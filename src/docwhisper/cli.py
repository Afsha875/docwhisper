"""docwhisper command-line interface: index, search, ask, stats."""

import argparse
import sys
from pathlib import Path

import httpx
import numpy as np

from docwhisper.ask import DEFAULT_MODEL, synthesize
from docwhisper.ingest import ingest_folder
from docwhisper.models import Hit
from docwhisper.store import INDEX_DIR, Embedder, Index

_SNIPPET_WIDTH = 120


def _embedder() -> Embedder:
    """Construct the embedding backend.

    FastEmbedder (and therefore the fastembed model) is imported lazily here so
    that commands and tests which never embed don't pay for model loading;
    tests monkeypatch this helper to inject a deterministic embedder.
    """
    from docwhisper.store import FastEmbedder

    return FastEmbedder()


def _has_index(folder: Path) -> bool:
    return Index.exists(folder)


def _no_index_error(folder: Path) -> int:
    print(
        f"error: no index found in {folder / INDEX_DIR} — run `docwhisper index` first",
        file=sys.stderr,
    )
    return 2


def _unique_paths(hits: list[Hit]) -> list[str]:
    seen: dict[str, None] = {}
    for hit in hits:
        seen.setdefault(hit.chunk.path, None)
    return list(seen)


def _cmd_index(folder: Path) -> int:
    index = Index.load(folder)
    chunks, new_manifest, deleted, changed = ingest_folder(folder, index.manifest)
    if chunks:
        vectors = _embedder().embed([chunk.text for chunk in chunks])
    else:
        dim = index.vectors.shape[1] if index.vectors.ndim == 2 else 0
        vectors = np.zeros((0, dim), dtype=np.float32)
    # Evict every re-processed file (`changed`) plus vanished ones (`deleted`),
    # so a file that now yields no chunks doesn't keep serving stale content.
    index.apply_changes(chunks, vectors, new_manifest, deleted | changed)
    index.save(folder)
    print(
        f"indexed {len(changed)} files (+{len(chunks)} chunks, "
        f"-{len(deleted)} files removed), total {index.size} chunks"
    )
    return 0


def _cmd_search(folder: Path, query: str, top_k: int, paths_only: bool) -> int:
    if not _has_index(folder):
        return _no_index_error(folder)
    index = Index.load(folder)
    hits = index.search(_embedder().embed_query(query), top_k=top_k)
    if paths_only:
        for path in _unique_paths(hits):
            print(path)
    else:
        for hit in hits:
            snippet = " ".join(hit.chunk.text.split())[:_SNIPPET_WIDTH]
            print(f"{hit.score:.3f}  {hit.chunk.path}  {snippet}")
    return 0


def _cmd_ask(folder: Path, question: str, top_k: int, model: str) -> int:
    if not _has_index(folder):
        return _no_index_error(folder)
    index = Index.load(folder)
    hits = index.search(_embedder().embed_query(question), top_k=top_k)
    try:
        answer = synthesize(question, hits, model=model)
    except httpx.HTTPError as exc:
        # Covers connect/read/timeout transport errors and non-2xx responses
        # (e.g. an unknown model 404) — all should fail gracefully, not crash.
        print(
            f"error: Ollama request failed ({type(exc).__name__}) — is `ollama serve` running "
            f"and is model {model!r} pulled?",
            file=sys.stderr,
        )
        return 3
    print(answer)
    print()
    print("Sources:")
    for path in _unique_paths(hits):
        print(f"  {path}")
    return 0


def _cmd_stats(folder: Path) -> int:
    if not _has_index(folder):
        return _no_index_error(folder)
    index = Index.load(folder)
    index_dir = folder / INDEX_DIR
    size_bytes = sum(p.stat().st_size for p in index_dir.rglob("*") if p.is_file())
    print(f"files:  {index.files}")
    print(f"chunks: {index.size}")
    print(f"size:   {size_bytes} bytes on disk ({index_dir})")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="docwhisper", description="Local semantic search over a folder of documents."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_index = sub.add_parser("index", help="ingest changed files, embed, save")
    p_index.add_argument("folder", type=Path)

    p_search = sub.add_parser("search", help="semantic search over the index")
    p_search.add_argument("folder", type=Path)
    p_search.add_argument("query")
    p_search.add_argument("-k", dest="top_k", type=int, default=8, help="number of hits")
    p_search.add_argument("--paths-only", action="store_true", help="print unique paths only")

    p_ask = sub.add_parser("ask", help="answer a question from the indexed documents")
    p_ask.add_argument("folder", type=Path)
    p_ask.add_argument("question")
    p_ask.add_argument("-k", dest="top_k", type=int, default=6, help="number of hits")
    p_ask.add_argument("--model", default=DEFAULT_MODEL, help="Ollama model name")

    p_stats = sub.add_parser("stats", help="show index statistics")
    p_stats.add_argument("folder", type=Path)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "index":
        return _cmd_index(args.folder)
    if args.command == "search":
        return _cmd_search(args.folder, args.query, args.top_k, args.paths_only)
    if args.command == "ask":
        return _cmd_ask(args.folder, args.question, args.top_k, args.model)
    return _cmd_stats(args.folder)


if __name__ == "__main__":
    raise SystemExit(main())
