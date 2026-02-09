# docwhisper — module contract (source of truth for parallel implementation)

Local semantic search CLI. No server, no external DB. Index artifacts live in
`<folder>/.docwhisper/` (chunks.jsonl + embeddings.npy + manifest.json).

Conventions: Python 3.12, ruff line-length 100 (`E,F,I,UP,B,SIM,RUF`), mypy-clean,
plain pytest (no asyncio needed anywhere — everything is synchronous).
`from __future__ import annotations` not needed. Type-hint everything public.

## src/docwhisper/models.py  (ALREADY WRITTEN — do not modify)

```python
@dataclass
class Chunk:
    id: str          # "<relpath>:<position>", e.g. "papers/attention.pdf:3"
    path: str        # path RELATIVE to the indexed root
    position: int    # chunk index within the file, 0-based
    text: str

@dataclass
class Hit:
    chunk: Chunk
    score: float     # cosine similarity, higher is better
```

## src/docwhisper/ingest.py  (Agent A)

```python
SUPPORTED = {".pdf", ".md", ".markdown", ".txt", ".rst"}

def extract_text(path: Path) -> str
    # .pdf via pypdf (join page texts with "\n"); others read as UTF-8
    # with errors="replace". Raises ValueError on unsupported suffix.

def chunk_text(text: str, max_chars: int = 1500, overlap: int = 200) -> list[str]
    # Split on paragraph boundaries (blank lines) where possible, packing
    # paragraphs up to max_chars; a paragraph longer than max_chars is split
    # hard with `overlap` chars of overlap between consecutive pieces.
    # No empty/whitespace-only chunks. Chunks < 30 chars are dropped.

def scan_files(root: Path) -> dict[str, float]
    # {relpath: mtime} for every SUPPORTED file under root, recursive,
    # skipping hidden dirs (any path component starting with ".").

def ingest_folder(root: Path, previous_manifest: dict[str, float]) -> tuple[list[Chunk], dict[str, float], set[str], set[str]]
    # Returns (chunks_for_changed_files, new_manifest, deleted, changed).
    # A file is "changed" if new to the manifest or mtime differs by > 1e-6.
    # Unchanged files are NOT re-extracted. deleted = in previous, not on disk.
    # changed = every re-processed file (incl. ones that now yield zero chunks or
    # whose extraction fails); caller evicts `deleted | changed` so stale content
    # never lingers. Extraction failures are logged via the module `logger` and
    # left out of the new manifest (retried next run) but DO appear in `changed`.
    # (Revised from a 3-tuple after the review swarm found chunkless re-indexed
    # files were never evicted.)
```

Tests in `tests/test_ingest.py`: chunk packing/overlap/min-size, scan_files hidden-dir
skipping, incremental semantics (changed/unchanged/deleted), extraction-error skip
(monkeypatch extract_text to raise for one file). Build tiny .md/.txt fixtures with
tmp_path; for PDF use pypdf to WRITE a one-page PDF fixture (pypdf.PdfWriter with a
blank page produces empty text — that's fine, assert extract_text returns str; do
NOT depend on network or binary fixture files).

## src/docwhisper/store.py  (Agent B)

```python
INDEX_DIR = ".docwhisper"

class Embedder(Protocol):
    def embed(self, texts: list[str]) -> np.ndarray      # (n, d) float32
    def embed_query(self, text: str) -> np.ndarray       # (d,)

class FastEmbedder:                                       # fastembed BGE-small
    # model_name param, default "BAAI/bge-small-en-v1.5". embed() uses
    # model.embed(texts, batch_size=64) SINGLE-PROCESS (multiprocess deadlocks
    # on macOS). embed_query() uses model.query_embed([text]).

class Index:
    # Holds: chunks list[Chunk], vectors np.ndarray (n, d), manifest dict[str, float]
    @classmethod
    def exists(cls, root: Path) -> bool         # True only if ALL 3 artifacts present
    @classmethod
    def load(cls, root: Path) -> "Index"        # empty Index if missing OR partial
    def save(self, root: Path) -> None          # chunks.jsonl, embeddings.npy, manifest.json
    def apply_changes(self, new_chunks: list[Chunk], vectors: np.ndarray,
                      new_manifest: dict[str, float], removed_paths: set[str]) -> None
        # Remove all existing chunks/vectors whose chunk.path is in
        # {paths of new_chunks} ∪ removed_paths, then append the new ones.
        # Replaces manifest wholesale. (removed_paths = deleted ∪ changed.)
    def search(self, query_vector: np.ndarray, top_k: int = 8) -> list[Hit]
        # Cosine similarity (normalize both sides; vectors may not be unit).
        # Empty index -> [].
    @property
    def files(self) -> int                       # len(manifest)
    @property
    def size(self) -> int                        # number of chunks
```

Tests in `tests/test_store.py`: save/load round-trip in tmp_path, apply_changes
add/replace/delete semantics, cosine ranking order with a hand-built 2-D fake
embedder (NO fastembed import in tests — construct Index state directly),
empty-index search.

## src/docwhisper/ask.py + src/docwhisper/cli.py  (Agent C)

ask.py:
```python
DEFAULT_MODEL = "llama3.1:8b"

def synthesize(question: str, hits: list[Hit], model: str = DEFAULT_MODEL,
               base_url: str = "http://localhost:11434") -> str
    # httpx POST {base_url}/api/chat, stream=False, temperature 0.1.
    # System prompt: answer ONLY from the provided excerpts; cite files inline
    # as [1], [2] mapping to the numbered excerpts; say so plainly if the
    # excerpts don't contain the answer. User content: question + numbered
    # excerpts, each headed "[n] <chunk.path>".
```

cli.py (argparse, `main(argv: list[str] | None = None) -> int`):
```
docwhisper index <folder>                 # ingest changed files, embed, save
docwhisper search <folder> "query" [-k 8] [--paths-only]
docwhisper ask <folder> "question" [-k 6] [--model ...]
docwhisper stats <folder>
```
- index: loads Index, ingest_folder, embeds ONLY changed chunks
  (FastEmbedder lazily imported so search/stats tests don't load the model),
  apply_changes, save. Prints "indexed N files (+A chunks, -D files removed),
  total C chunks". Exit 0.
- search: prints score, path, and a single-line 120-char snippet per hit
  (or just unique paths with --paths-only). Exit 0; exit 2 if no index.
- ask: search then synthesize; prints answer, then "Sources:" list of
  the hit paths. Exit 2 if no index, exit 3 if Ollama unreachable
  (httpx.ConnectError -> friendly message on stderr).
- stats: files, chunks, index dir size on disk. Exit 2 if no index.
- Embedder is constructed in ONE helper `_embedder()` so tests can
  monkeypatch it.

Tests in `tests/test_cli.py`: end-to-end index -> search -> stats in tmp_path
using a monkeypatched deterministic embedder (hash-based small vectors are fine
— same input text MUST give same vector); ask path with synthesize monkeypatched;
exit codes 2/3; --paths-only. NO network, NO fastembed model download in tests.
```

## Boundaries

- Agent A touches ONLY src/docwhisper/ingest.py and tests/test_ingest.py.
- Agent B touches ONLY src/docwhisper/store.py and tests/test_store.py.
- Agent C touches ONLY src/docwhisper/ask.py, src/docwhisper/cli.py, tests/test_cli.py.
- Nobody edits pyproject.toml, models.py, or another agent's files.
