"""Answer synthesis over retrieved chunks via a local Ollama model."""

import httpx

from docwhisper.models import Hit

DEFAULT_MODEL = "llama3.1:8b"

_SYSTEM_PROMPT = (
    "You answer questions using ONLY the numbered document excerpts provided by the user. "
    "Cite the files you draw from inline as [1], [2], ... matching the excerpt numbers. "
    "If the excerpts do not contain the answer, say so plainly instead of guessing."
)


def synthesize(
    question: str,
    hits: list[Hit],
    model: str = DEFAULT_MODEL,
    base_url: str = "http://localhost:11434",
) -> str:
    """Ask a local Ollama model to answer ``question`` from the retrieved ``hits``.

    Sends a single non-streaming ``/api/chat`` request with the excerpts numbered
    ``[n] <path>`` so the model can cite them inline.
    """
    excerpts = "\n\n".join(
        f"[{i}] {hit.chunk.path}\n{hit.chunk.text}" for i, hit in enumerate(hits, start=1)
    )
    payload = {
        "model": model,
        "stream": False,
        "options": {"temperature": 0.1},
        "messages": [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Question: {question}\n\nExcerpts:\n\n{excerpts}",
            },
        ],
    }
    response = httpx.post(f"{base_url}/api/chat", json=payload, timeout=120.0)
    response.raise_for_status()
    return str(response.json()["message"]["content"]).strip()
