"""Persistent embedding index: on-disk storage, incremental updates, cosine search.

Index artifacts live under ``<root>/.docwhisper/``:

- ``chunks.jsonl``    — one JSON object per chunk
- ``embeddings.npy``  — float32 matrix, row i is the vector for chunk i
- ``manifest.json``   — {relpath: mtime} of the files the index was built from
"""

import json
from dataclasses import asdict
from pathlib import Path
from typing import Protocol

import numpy as np

from docwhisper.models import Chunk, Hit

INDEX_DIR = ".docwhisper"

_CHUNKS_FILE = "chunks.jsonl"
_EMBEDDINGS_FILE = "embeddings.npy"
_MANIFEST_FILE = "manifest.json"


class Embedder(Protocol):
    """Anything that can turn text into vectors."""

    def embed(self, texts: list[str]) -> np.ndarray:
        """Embed documents. Returns a (n, d) float32 array."""
        ...

    def embed_query(self, text: str) -> np.ndarray:
        """Embed a single query. Returns a (d,) float32 array."""
        ...


class FastEmbedder:
    """Embedder backed by fastembed's BGE-small model.

    Embedding runs single-process: fastembed's multiprocess pool deadlocks on macOS.
    """

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5") -> None:
        from fastembed import TextEmbedding

        self._model = TextEmbedding(model_name=model_name)

    def embed(self, texts: list[str]) -> np.ndarray:
        vectors = list(self._model.embed(texts, batch_size=64))
        if not vectors:
            return np.zeros((0, 0), dtype=np.float32)
        return np.asarray(vectors, dtype=np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        vector = next(iter(self._model.query_embed([text])))
        return np.asarray(vector, dtype=np.float32)


class Index:
    """In-memory index of chunks + embedding vectors, persisted to ``<root>/.docwhisper``."""

    def __init__(
        self,
        chunks: list[Chunk] | None = None,
        vectors: np.ndarray | None = None,
        manifest: dict[str, float] | None = None,
    ) -> None:
        self.chunks: list[Chunk] = chunks if chunks is not None else []
        self.vectors: np.ndarray = (
            vectors if vectors is not None else np.zeros((0, 0), dtype=np.float32)
        )
        self.manifest: dict[str, float] = manifest if manifest is not None else {}

    @classmethod
    def exists(cls, root: Path) -> bool:
        """True only if a *complete* index (all three artifacts) is present.

        A bare or half-written ``.docwhisper`` directory (e.g. an ``index`` run
        interrupted after mkdir but before the files were written) does not
        count — it is treated as no index, so commands report a friendly error
        and ``index`` can rebuild it rather than crashing on a missing file.
        """
        index_dir = root / INDEX_DIR
        return all(
            (index_dir / name).is_file()
            for name in (_CHUNKS_FILE, _EMBEDDINGS_FILE, _MANIFEST_FILE)
        )

    @classmethod
    def load(cls, root: Path) -> "Index":
        """Load the index stored under ``root``; returns an empty Index if none (or partial)."""
        index_dir = root / INDEX_DIR
        if not cls.exists(root):
            return cls()
        chunks: list[Chunk] = []
        with (index_dir / _CHUNKS_FILE).open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    chunks.append(Chunk(**json.loads(line)))
        vectors = np.load(index_dir / _EMBEDDINGS_FILE)
        manifest_raw = json.loads((index_dir / _MANIFEST_FILE).read_text(encoding="utf-8"))
        manifest = {str(k): float(v) for k, v in manifest_raw.items()}
        return cls(chunks=chunks, vectors=vectors, manifest=manifest)

    def save(self, root: Path) -> None:
        """Write chunks.jsonl, embeddings.npy and manifest.json under ``root/.docwhisper``."""
        index_dir = root / INDEX_DIR
        index_dir.mkdir(parents=True, exist_ok=True)
        with (index_dir / _CHUNKS_FILE).open("w", encoding="utf-8") as f:
            for chunk in self.chunks:
                f.write(json.dumps(asdict(chunk), ensure_ascii=False) + "\n")
        np.save(index_dir / _EMBEDDINGS_FILE, np.asarray(self.vectors, dtype=np.float32))
        (index_dir / _MANIFEST_FILE).write_text(
            json.dumps(self.manifest, indent=2, sort_keys=True), encoding="utf-8"
        )

    def apply_changes(
        self,
        new_chunks: list[Chunk],
        vectors: np.ndarray,
        new_manifest: dict[str, float],
        removed_paths: set[str],
    ) -> None:
        """Replace chunks for changed files, drop removed files, adopt the new manifest.

        Every existing chunk whose path is in ``removed_paths`` (or in the paths
        of ``new_chunks``) is removed with its vector, then the new
        chunks/vectors are appended. ``removed_paths`` must include every
        re-processed file — even ones that produced no new chunks — so their
        stale content is evicted; see ``ingest_folder``.
        """
        stale_paths = {chunk.path for chunk in new_chunks} | removed_paths
        keep = [i for i, chunk in enumerate(self.chunks) if chunk.path not in stale_paths]
        kept_chunks = [self.chunks[i] for i in keep]
        kept_vectors = self.vectors[keep] if len(self.chunks) else self.vectors

        self.chunks = kept_chunks + list(new_chunks)
        new_vectors = np.asarray(vectors, dtype=np.float32)
        if kept_vectors.size == 0:
            self.vectors = (
                new_vectors if new_vectors.size else np.zeros((0, 0), dtype=np.float32)
            )
        elif new_vectors.size == 0:
            self.vectors = kept_vectors
        else:
            self.vectors = np.vstack([kept_vectors, new_vectors])
        self.manifest = dict(new_manifest)

    def search(self, query_vector: np.ndarray, top_k: int = 8) -> list[Hit]:
        """Return the ``top_k`` chunks most cosine-similar to ``query_vector``."""
        if not self.chunks or self.vectors.size == 0:
            return []
        matrix = np.asarray(self.vectors, dtype=np.float32)
        query = np.asarray(query_vector, dtype=np.float32).reshape(-1)

        matrix_norms = np.linalg.norm(matrix, axis=1)
        matrix_norms[matrix_norms == 0] = 1.0
        query_norm = float(np.linalg.norm(query)) or 1.0

        scores = (matrix @ query) / (matrix_norms * query_norm)
        order = np.argsort(-scores)[:top_k]
        return [Hit(chunk=self.chunks[i], score=float(scores[i])) for i in order]

    @property
    def files(self) -> int:
        """Number of files covered by the manifest."""
        return len(self.manifest)

    @property
    def size(self) -> int:
        """Number of chunks in the index."""
        return len(self.chunks)
