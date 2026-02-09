"""Tests for docwhisper.store.Index — no fastembed import anywhere here."""

from pathlib import Path

import numpy as np

from docwhisper.models import Chunk
from docwhisper.store import INDEX_DIR, Index


def _chunk(path: str, position: int, text: str) -> Chunk:
    return Chunk(id=f"{path}:{position}", path=path, position=position, text=text)


def _make_index() -> Index:
    chunks = [
        _chunk("a.md", 0, "alpha"),
        _chunk("a.md", 1, "beta"),
        _chunk("b.txt", 0, "gamma"),
    ]
    vectors = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=np.float32)
    manifest = {"a.md": 1.0, "b.txt": 2.0}
    return Index(chunks=chunks, vectors=vectors, manifest=manifest)


def test_load_missing_dir_gives_empty_index(tmp_path: Path) -> None:
    index = Index.load(tmp_path)
    assert index.chunks == []
    assert index.manifest == {}
    assert index.size == 0
    assert index.files == 0
    assert index.search(np.array([1.0, 0.0], dtype=np.float32)) == []


def test_save_load_round_trip(tmp_path: Path) -> None:
    index = _make_index()
    index.save(tmp_path)

    assert (tmp_path / INDEX_DIR / "chunks.jsonl").is_file()
    assert (tmp_path / INDEX_DIR / "embeddings.npy").is_file()
    assert (tmp_path / INDEX_DIR / "manifest.json").is_file()

    loaded = Index.load(tmp_path)
    assert loaded.chunks == index.chunks
    assert loaded.manifest == index.manifest
    np.testing.assert_array_equal(loaded.vectors, index.vectors)
    assert loaded.files == 2
    assert loaded.size == 3


def test_apply_changes_add_to_empty_index() -> None:
    index = Index()
    new_chunks = [_chunk("new.md", 0, "hello world")]
    vectors = np.array([[0.5, 0.5]], dtype=np.float32)
    index.apply_changes(new_chunks, vectors, {"new.md": 5.0}, removed_paths=set())

    assert index.chunks == new_chunks
    np.testing.assert_array_equal(index.vectors, vectors)
    assert index.manifest == {"new.md": 5.0}


def test_apply_changes_replaces_chunks_of_changed_file() -> None:
    index = _make_index()
    replacement = [_chunk("a.md", 0, "alpha v2")]
    vectors = np.array([[9.0, 9.0]], dtype=np.float32)
    index.apply_changes(replacement, vectors, {"a.md": 3.0, "b.txt": 2.0}, removed_paths=set())

    paths = [c.path for c in index.chunks]
    assert paths == ["b.txt", "a.md"]
    assert index.chunks[-1].text == "alpha v2"
    assert index.vectors.shape == (2, 2)
    np.testing.assert_array_equal(index.vectors[0], np.array([1.0, 1.0], dtype=np.float32))
    np.testing.assert_array_equal(index.vectors[1], np.array([9.0, 9.0], dtype=np.float32))
    assert index.manifest == {"a.md": 3.0, "b.txt": 2.0}


def test_apply_changes_deletes_removed_file() -> None:
    index = _make_index()
    index.apply_changes(
        [], np.zeros((0, 2), dtype=np.float32), {"b.txt": 2.0}, removed_paths={"a.md"}
    )

    assert [c.path for c in index.chunks] == ["b.txt"]
    assert index.vectors.shape == (1, 2)
    assert index.files == 1
    assert index.size == 1


def test_apply_changes_mixed_add_replace_delete() -> None:
    index = _make_index()
    new_chunks = [_chunk("b.txt", 0, "gamma v2"), _chunk("c.rst", 0, "delta")]
    vectors = np.array([[2.0, 0.0], [0.0, 3.0]], dtype=np.float32)
    new_manifest = {"b.txt": 9.0, "c.rst": 4.0}
    index.apply_changes(new_chunks, vectors, new_manifest, removed_paths={"a.md"})

    assert [c.path for c in index.chunks] == ["b.txt", "c.rst"]
    assert [c.text for c in index.chunks] == ["gamma v2", "delta"]
    np.testing.assert_array_equal(index.vectors, vectors)
    assert index.manifest == new_manifest


def test_search_ranks_by_cosine_similarity() -> None:
    # Hand-built 2-D vectors: cosine order relative to query [1, 0] is
    # exact match > 45 degrees > orthogonal.
    chunks = [
        _chunk("ortho.md", 0, "orthogonal"),
        _chunk("exact.md", 0, "exact"),
        _chunk("diag.md", 0, "diagonal"),
    ]
    # Non-unit magnitudes prove normalization: the exact-direction vector is tiny.
    vectors = np.array([[0.0, 5.0], [0.1, 0.0], [3.0, 3.0]], dtype=np.float32)
    index = Index(chunks=chunks, vectors=vectors, manifest={})

    hits = index.search(np.array([2.0, 0.0], dtype=np.float32), top_k=3)
    assert [h.chunk.path for h in hits] == ["exact.md", "diag.md", "ortho.md"]
    assert hits[0].score > hits[1].score > hits[2].score
    assert abs(hits[0].score - 1.0) < 1e-6
    assert abs(hits[2].score) < 1e-6


def test_search_respects_top_k() -> None:
    index = _make_index()
    hits = index.search(np.array([1.0, 0.0], dtype=np.float32), top_k=2)
    assert len(hits) == 2


def test_search_empty_index_returns_empty_list() -> None:
    assert Index().search(np.array([1.0, 0.0], dtype=np.float32)) == []
