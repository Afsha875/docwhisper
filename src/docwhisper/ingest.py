"""File discovery, text extraction, and chunking for docwhisper.

Turns a folder of documents into `Chunk` objects, doing incremental work: only
files that are new or whose mtime changed since the previous manifest are
re-extracted and re-chunked.
"""

import logging
import re
from pathlib import Path

from docwhisper.models import Chunk

logger = logging.getLogger(__name__)

SUPPORTED = {".pdf", ".md", ".markdown", ".txt", ".rst"}

_MIN_CHUNK_CHARS = 30
_PARAGRAPH_SPLIT = re.compile(r"\n\s*\n")


def extract_text(path: Path) -> str:
    """Extract plain text from a supported document.

    PDFs are read with pypdf (page texts joined with newlines); everything
    else is read as UTF-8 with undecodable bytes replaced.

    Raises:
        ValueError: if the file suffix is not in ``SUPPORTED``.
    """
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED:
        raise ValueError(f"unsupported file type {suffix!r}: {path}")
    if suffix == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(path)
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    return path.read_text(encoding="utf-8", errors="replace")


def _split_long(paragraph: str, max_chars: int, overlap: int) -> list[str]:
    """Hard-split an over-long paragraph into ``max_chars`` windows.

    Consecutive windows share ``overlap`` characters so that sentences cut at
    a window boundary still appear whole in one of the pieces.
    """
    step = max(max_chars - overlap, 1)
    windows: list[str] = []
    i = 0
    n = len(paragraph)
    while i < n:
        windows.append(paragraph[i : i + max_chars])
        if i + max_chars >= n:  # this window reached the end; another would be redundant
            break
        i += step
    return windows


def chunk_text(text: str, max_chars: int = 1500, overlap: int = 200) -> list[str]:
    """Split ``text`` into chunks of at most ``max_chars`` characters.

    Splits on paragraph boundaries (blank lines) where possible, packing
    whole paragraphs into each chunk. A single paragraph longer than
    ``max_chars`` is split hard with ``overlap`` characters of overlap
    between consecutive pieces. Chunks shorter than 30 characters (and
    empty/whitespace-only ones) are dropped.
    """
    pieces: list[str] = []
    for raw in _PARAGRAPH_SPLIT.split(text):
        paragraph = raw.strip()
        if not paragraph:
            continue
        if len(paragraph) <= max_chars:
            pieces.append(paragraph)
        else:
            pieces.extend(_split_long(paragraph, max_chars, overlap))

    chunks: list[str] = []
    current: list[str] = []
    current_len = 0
    for piece in pieces:
        # +2 accounts for the "\n\n" separator when joining paragraphs.
        extra = len(piece) + (2 if current else 0)
        if current and current_len + extra > max_chars:
            chunks.append("\n\n".join(current))
            current = []
            current_len = 0
            extra = len(piece)
        current.append(piece)
        current_len += extra
    if current:
        chunks.append("\n\n".join(current))

    return [c for c in chunks if len(c) >= _MIN_CHUNK_CHARS]


def scan_files(root: Path) -> dict[str, float]:
    """Map relative path -> mtime for every supported file under ``root``.

    Recursive; any path component starting with "." (hidden dirs/files) is
    skipped. Relative paths use "/" separators regardless of platform.
    """
    result: dict[str, float] = {}
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED:
            continue
        rel = path.relative_to(root)
        if any(part.startswith(".") for part in rel.parts):
            continue
        result[rel.as_posix()] = path.stat().st_mtime
    return result


def ingest_folder(
    root: Path, previous_manifest: dict[str, float]
) -> tuple[list[Chunk], dict[str, float], set[str], set[str]]:
    """Incrementally ingest ``root`` against a previous manifest.

    Returns ``(chunks_for_changed_files, new_manifest, deleted, changed)``:

    - ``deleted``: files present in ``previous_manifest`` but no longer on disk.
    - ``changed``: files re-processed this run (new, or mtime differs by more
      than 1e-6). Every changed file's *existing* index entries must be evicted
      — including files that now produce zero chunks (emptied or whittled below
      the minimum size) or whose extraction fails. Otherwise their old content
      would be served forever. The caller removes ``deleted | changed``.

    Unchanged files are not re-extracted. Files whose extraction raises are
    logged and left out of the new manifest so they are retried next run, but
    they still appear in ``changed`` so their stale chunks are dropped now.
    """
    current = scan_files(root)
    deleted = set(previous_manifest) - set(current)

    chunks: list[Chunk] = []
    new_manifest: dict[str, float] = {}
    changed: set[str] = set()
    for relpath, mtime in current.items():
        previous_mtime = previous_manifest.get(relpath)
        if previous_mtime is not None and abs(mtime - previous_mtime) <= 1e-6:
            new_manifest[relpath] = mtime
            continue
        changed.add(relpath)  # re-processing: evict old entries regardless of outcome
        try:
            text = extract_text(root / relpath)
        except Exception:
            logger.warning("skipping %s: extraction failed", relpath, exc_info=True)
            continue
        chunks.extend(
            Chunk(id=f"{relpath}:{position}", path=relpath, position=position, text=piece)
            for position, piece in enumerate(chunk_text(text))
        )
        new_manifest[relpath] = mtime

    return chunks, new_manifest, deleted, changed
