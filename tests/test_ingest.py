"""Tests for docwhisper.ingest: chunking, scanning, and incremental ingestion."""

import itertools
import os
from pathlib import Path

import pytest

from docwhisper import ingest
from docwhisper.ingest import chunk_text, extract_text, ingest_folder, scan_files


class TestChunkText:
    def test_packs_paragraphs_up_to_max_chars(self) -> None:
        paragraphs = [f"paragraph number {i} " + "x" * 80 for i in range(6)]
        text = "\n\n".join(paragraphs)
        chunks = chunk_text(text, max_chars=250)
        assert len(chunks) > 1
        assert all(len(c) <= 250 for c in chunks)
        # Every paragraph survives, whole, in exactly the order given.
        assert "\n\n".join(chunks).split("\n\n") == paragraphs

    def test_short_text_is_single_chunk(self) -> None:
        text = "one small paragraph that easily fits in a single chunk"
        assert chunk_text(text) == [text]

    def test_long_paragraph_split_hard_with_overlap(self) -> None:
        text = "".join(str(i % 10) for i in range(400))  # no blank lines anywhere
        chunks = chunk_text(text, max_chars=150, overlap=50)
        assert len(chunks) == 4
        assert all(len(c) <= 150 for c in chunks)
        for left, right in itertools.pairwise(chunks):
            assert left[-50:] == right[:50]
        # Stripping the overlap reconstructs the original text.
        assert chunks[0] + "".join(c[50:] for c in chunks[1:]) == text

    def test_no_redundant_final_window(self) -> None:
        # Regression: for this length the old range()-based split emitted a THIRD
        # window whose content was entirely within the overlap of the second —
        # zero new characters. The fix stops once a window reaches the end.
        text = "".join(str(i % 10) for i in range(230))
        chunks = chunk_text(text, max_chars=150, overlap=50)
        assert len(chunks) == 2  # was 3 before the fix
        assert len(chunks[-1]) > 50  # last window carries content beyond the overlap
        assert chunks[0] + chunks[1][50:] == text  # overlap-stripped reconstruction

    def test_drops_small_and_whitespace_chunks(self) -> None:
        assert chunk_text("tiny") == []
        # A tiny paragraph that cannot pack with its oversized neighbour is dropped.
        chunks = chunk_text("tiny\n\n   \n\n" + "z" * 100, max_chars=100)
        assert chunks == ["z" * 100]

    def test_empty_input(self) -> None:
        assert chunk_text("") == []
        assert chunk_text("   \n\n  \n ") == []


class TestExtractText:
    def test_reads_text_file_utf8(self, tmp_path: Path) -> None:
        f = tmp_path / "note.txt"
        f.write_text("hello docwhisper é", encoding="utf-8")
        assert extract_text(f) == "hello docwhisper é"

    def test_bad_bytes_are_replaced(self, tmp_path: Path) -> None:
        f = tmp_path / "weird.md"
        f.write_bytes(b"ok \xff\xfe bytes")
        text = extract_text(f)
        assert text.startswith("ok ")
        assert "�" in text

    def test_unsupported_suffix_raises(self, tmp_path: Path) -> None:
        f = tmp_path / "data.csv"
        f.write_text("a,b")
        with pytest.raises(ValueError, match="unsupported"):
            extract_text(f)

    def test_pdf_returns_str(self, tmp_path: Path) -> None:
        from pypdf import PdfWriter

        pdf = tmp_path / "blank.pdf"
        writer = PdfWriter()
        writer.add_blank_page(width=200, height=200)
        with pdf.open("wb") as fh:
            writer.write(fh)
        assert isinstance(extract_text(pdf), str)


class TestScanFiles:
    def test_finds_supported_files_recursively(self, tmp_path: Path) -> None:
        (tmp_path / "a.md").write_text("a")
        (tmp_path / "sub").mkdir()
        (tmp_path / "sub" / "b.txt").write_text("b")
        (tmp_path / "sub" / "c.py").write_text("c")  # unsupported
        manifest = scan_files(tmp_path)
        assert set(manifest) == {"a.md", "sub/b.txt"}
        assert all(isinstance(m, float) for m in manifest.values())

    def test_skips_hidden_dirs_and_files(self, tmp_path: Path) -> None:
        (tmp_path / ".docwhisper").mkdir()
        (tmp_path / ".docwhisper" / "hidden.md").write_text("hidden")
        (tmp_path / "sub" / ".secret").mkdir(parents=True)
        (tmp_path / "sub" / ".secret" / "deep.txt").write_text("deep")
        (tmp_path / ".dotfile.md").write_text("dot")
        (tmp_path / "visible.md").write_text("visible")
        assert set(scan_files(tmp_path)) == {"visible.md"}


class TestIngestFolder:
    @staticmethod
    def _write(root: Path, rel: str, text: str) -> Path:
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        return path

    def test_fresh_ingest(self, tmp_path: Path) -> None:
        self._write(tmp_path, "a.md", "first document body, comfortably long enough to keep")
        self._write(tmp_path, "sub/b.txt", "second document body, also long enough to keep")
        chunks, manifest, deleted, changed = ingest_folder(tmp_path, {})
        assert {c.path for c in chunks} == {"a.md", "sub/b.txt"}
        assert all(c.id == f"{c.path}:{c.position}" for c in chunks)
        assert set(manifest) == {"a.md", "sub/b.txt"}
        assert deleted == set()
        assert changed == {"a.md", "sub/b.txt"}  # new files count as changed

    def test_unchanged_files_not_reextracted(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._write(tmp_path, "a.md", "stable content that does not change between two runs")
        _, manifest, _, _ = ingest_folder(tmp_path, {})

        def boom(path: Path) -> str:
            raise AssertionError(f"extract_text called for unchanged file {path}")

        monkeypatch.setattr(ingest, "extract_text", boom)
        chunks, new_manifest, deleted, changed = ingest_folder(tmp_path, manifest)
        assert chunks == []
        assert new_manifest == manifest
        assert deleted == set()
        assert changed == set()  # nothing re-processed

    def test_changed_file_is_reingested(self, tmp_path: Path) -> None:
        path = self._write(tmp_path, "a.md", "original content, comfortably above minimum size")
        _, manifest, _, _ = ingest_folder(tmp_path, {})
        path.write_text("updated content, still comfortably above the minimum size")
        os.utime(path, (path.stat().st_atime, manifest["a.md"] + 5.0))
        chunks, new_manifest, deleted, changed = ingest_folder(tmp_path, manifest)
        assert [c.path for c in chunks] == ["a.md"]
        assert "updated content" in chunks[0].text
        assert new_manifest["a.md"] == pytest.approx(manifest["a.md"] + 5.0)
        assert changed == {"a.md"} and deleted == set()

    def test_emptied_file_is_marked_changed_for_eviction(self, tmp_path: Path) -> None:
        # Regression: a file re-processed to zero chunks must still be reported
        # as changed so its stale chunks get evicted downstream.
        path = self._write(tmp_path, "a.md", "content long enough to become one indexed chunk")
        chunks, manifest, _, _ = ingest_folder(tmp_path, {})
        assert len(chunks) == 1
        path.write_text("stub")  # < 30 chars -> zero chunks
        os.utime(path, (path.stat().st_atime, manifest["a.md"] + 5.0))
        chunks2, _, deleted, changed = ingest_folder(tmp_path, manifest)
        assert chunks2 == []
        assert changed == {"a.md"}  # re-processed even though it yielded nothing
        assert deleted == set()

    def test_deleted_file_reported(self, tmp_path: Path) -> None:
        path = self._write(tmp_path, "gone.txt", "this file is about to be deleted from disk")
        _, manifest, _, _ = ingest_folder(tmp_path, {})
        path.unlink()
        chunks, new_manifest, deleted, changed = ingest_folder(tmp_path, manifest)
        assert chunks == []
        assert "gone.txt" not in new_manifest
        assert deleted == {"gone.txt"}
        assert changed == set()

    def test_extraction_error_skips_file_with_warning(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        self._write(tmp_path, "good.md", "a healthy document whose extraction works just fine")
        self._write(tmp_path, "bad.md", "a document whose extraction is forced to fail below")
        real_extract = ingest.extract_text

        def flaky(path: Path) -> str:
            if path.name == "bad.md":
                raise RuntimeError("simulated extraction failure")
            return real_extract(path)

        monkeypatch.setattr(ingest, "extract_text", flaky)
        with caplog.at_level("WARNING", logger="docwhisper.ingest"):
            chunks, manifest, deleted, changed = ingest_folder(tmp_path, {})
        assert {c.path for c in chunks} == {"good.md"}
        assert set(manifest) == {"good.md"}  # bad.md retried next run
        assert deleted == set()
        # bad.md was still re-processed, so it's "changed" and its (nonexistent
        # here, but stale on a real re-index) chunks would be evicted.
        assert changed == {"good.md", "bad.md"}
        assert any("bad.md" in record.message for record in caplog.records)
