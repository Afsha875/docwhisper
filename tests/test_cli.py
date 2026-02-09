"""End-to-end CLI tests with a deterministic fake embedder (no network, no fastembed)."""

import hashlib
from pathlib import Path

import httpx
import numpy as np
import pytest

from docwhisper import cli
from docwhisper.models import Hit

DIM = 32

DOC_CATS = (
    "Cats are small carnivorous mammals that have lived alongside humans "
    "for thousands of years as companions."
)
DOC_SPACE = (
    "Rockets reach orbit by accelerating to roughly eight kilometres per second, "
    "fighting gravity and atmospheric drag the whole way up."
)


class FakeEmbedder:
    """Hash-seeded random unit vectors: identical text always maps to the same vector."""

    def _vec(self, text: str) -> np.ndarray:
        seed = int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big")
        vec = np.random.default_rng(seed).standard_normal(DIM).astype(np.float32)
        return vec / np.linalg.norm(vec)

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, DIM), dtype=np.float32)
        return np.stack([self._vec(t) for t in texts])

    def embed_query(self, text: str) -> np.ndarray:
        return self._vec(text)


@pytest.fixture
def fake_embedder(monkeypatch: pytest.MonkeyPatch) -> FakeEmbedder:
    embedder = FakeEmbedder()
    monkeypatch.setattr(cli, "_embedder", lambda: embedder)
    return embedder


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    (tmp_path / "cats.txt").write_text(DOC_CATS, encoding="utf-8")
    (tmp_path / "space.md").write_text(DOC_SPACE, encoding="utf-8")
    return tmp_path


def test_index_search_stats_end_to_end(
    corpus: Path, fake_embedder: FakeEmbedder, capsys: pytest.CaptureFixture[str]
) -> None:
    assert cli.main(["index", str(corpus)]) == 0
    out = capsys.readouterr().out
    assert "indexed 2 files" in out
    assert "total 2 chunks" in out

    # Querying with a document's exact text must rank that document first.
    assert cli.main(["search", str(corpus), DOC_CATS]) == 0
    lines = capsys.readouterr().out.strip().splitlines()
    assert len(lines) == 2
    assert "cats.txt" in lines[0]
    top_score = float(lines[0].split()[0])
    assert top_score > 0.9

    assert cli.main(["stats", str(corpus)]) == 0
    stats_out = capsys.readouterr().out
    assert "2" in stats_out  # files
    assert "chunks" in stats_out
    assert "bytes" in stats_out


def test_index_is_incremental(
    corpus: Path, fake_embedder: FakeEmbedder, capsys: pytest.CaptureFixture[str]
) -> None:
    assert cli.main(["index", str(corpus)]) == 0
    capsys.readouterr()
    # Nothing changed: second run should re-embed nothing and keep the totals.
    assert cli.main(["index", str(corpus)]) == 0
    out = capsys.readouterr().out
    assert "indexed 0 files (+0 chunks, -0 files removed), total 2 chunks" in out


def test_index_reports_deleted_files(
    corpus: Path, fake_embedder: FakeEmbedder, capsys: pytest.CaptureFixture[str]
) -> None:
    assert cli.main(["index", str(corpus)]) == 0
    capsys.readouterr()
    (corpus / "space.md").unlink()
    assert cli.main(["index", str(corpus)]) == 0
    out = capsys.readouterr().out
    assert "-1 files removed" in out
    assert "total 1 chunks" in out


def test_search_paths_only(
    corpus: Path, fake_embedder: FakeEmbedder, capsys: pytest.CaptureFixture[str]
) -> None:
    assert cli.main(["index", str(corpus)]) == 0
    capsys.readouterr()
    assert cli.main(["search", str(corpus), DOC_CATS, "--paths-only"]) == 0
    lines = capsys.readouterr().out.strip().splitlines()
    assert sorted(lines) == ["cats.txt", "space.md"]


def test_search_respects_k(
    corpus: Path, fake_embedder: FakeEmbedder, capsys: pytest.CaptureFixture[str]
) -> None:
    assert cli.main(["index", str(corpus)]) == 0
    capsys.readouterr()
    assert cli.main(["search", str(corpus), "anything at all", "-k", "1"]) == 0
    lines = capsys.readouterr().out.strip().splitlines()
    assert len(lines) == 1


@pytest.mark.parametrize("command", [["search", "{}", "q"], ["ask", "{}", "q"], ["stats", "{}"]])
def test_exit_2_when_no_index(
    tmp_path: Path,
    fake_embedder: FakeEmbedder,
    capsys: pytest.CaptureFixture[str],
    command: list[str],
) -> None:
    argv = [arg.format(tmp_path) for arg in command]
    assert cli.main(argv) == 2
    assert "no index" in capsys.readouterr().err


def test_ask_prints_answer_and_sources(
    corpus: Path,
    fake_embedder: FakeEmbedder,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert cli.main(["index", str(corpus)]) == 0
    capsys.readouterr()

    captured: dict[str, object] = {}

    def fake_synthesize(question: str, hits: list[Hit], model: str = "x") -> str:
        captured["question"] = question
        captured["hits"] = hits
        captured["model"] = model
        return "Cats are companions [1]."

    monkeypatch.setattr(cli, "synthesize", fake_synthesize)
    assert cli.main(["ask", str(corpus), "what are cats?", "--model", "test-model"]) == 0
    out = capsys.readouterr().out
    assert "Cats are companions [1]." in out
    assert "Sources:" in out
    assert "cats.txt" in out
    assert captured["question"] == "what are cats?"
    assert captured["model"] == "test-model"
    hits = captured["hits"]
    assert isinstance(hits, list) and all(isinstance(h, Hit) for h in hits)


@pytest.mark.parametrize(
    "error",
    [
        httpx.ConnectError("connection refused"),
        httpx.ReadTimeout("timed out"),  # ollama slow/hung mid-request
        httpx.ReadError("server closed connection"),  # ollama killed mid-request
        httpx.HTTPStatusError(  # unknown model -> 404
            "not found",
            request=httpx.Request("POST", "http://localhost:11434/api/chat"),
            response=httpx.Response(404),
        ),
    ],
)
def test_ask_exit_3_on_any_http_error(
    corpus: Path,
    fake_embedder: FakeEmbedder,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    error: httpx.HTTPError,
) -> None:
    assert cli.main(["index", str(corpus)]) == 0
    capsys.readouterr()

    def raise_error(*args: object, **kwargs: object) -> str:
        raise error

    monkeypatch.setattr(cli, "synthesize", raise_error)
    assert cli.main(["ask", str(corpus), "what are cats?"]) == 3
    assert "Ollama" in capsys.readouterr().err


def test_reindex_removes_stale_content_from_emptied_file(
    corpus: Path, fake_embedder: FakeEmbedder, capsys: pytest.CaptureFixture[str]
) -> None:
    # Regression (found by the review swarm): a file re-indexed down to nothing
    # must stop being searchable, and the re-index must not be a silent no-op.
    secret = corpus / "cats.txt"
    assert cli.main(["index", str(corpus)]) == 0
    capsys.readouterr()

    import os

    mtime = os.stat(secret).st_mtime
    secret.write_text("x")  # < 30 chars -> zero chunks
    os.utime(secret, (mtime + 10, mtime + 10))

    assert cli.main(["index", str(corpus)]) == 0
    out = capsys.readouterr().out
    assert "total 1 chunks" in out  # cats.txt's chunk is gone, only space.md remains

    assert cli.main(["search", str(corpus), DOC_CATS, "--paths-only"]) == 0
    remaining = capsys.readouterr().out.strip().splitlines()
    assert "cats.txt" not in remaining  # stale content no longer served


def test_partial_index_dir_does_not_crash(
    tmp_path: Path, fake_embedder: FakeEmbedder, capsys: pytest.CaptureFixture[str]
) -> None:
    # Regression: a bare/half-written .docwhisper dir must read as "no index"
    # (friendly exit 2), and `index` must be able to rebuild over it.
    (tmp_path / ".docwhisper").mkdir()
    (tmp_path / "doc.md").write_text(DOC_SPACE, encoding="utf-8")

    assert cli.main(["stats", str(tmp_path)]) == 2
    assert "no index" in capsys.readouterr().err

    assert cli.main(["index", str(tmp_path)]) == 0  # rebuilds cleanly
    capsys.readouterr()
    assert cli.main(["stats", str(tmp_path)]) == 0
