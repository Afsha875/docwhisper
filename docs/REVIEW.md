# Review findings & resolutions

docwhisper's three modules (`ingest`, `store`, `cli`) were built in parallel by
separate agents against `docs/CONTRACT.md`, integrated (35 tests green), then
put through three adversarial reviewers with distinct mandates: correctness,
robustness, and contract/packaging fidelity. Each reviewer had to *reproduce* a
defect before reporting it.

Four real defects survived that gauntlet. All are now fixed and pinned by
regression tests (named below; run `uv run pytest -k <name>`).

| # | Severity | Defect | Fix | Regression test |
|---|---|---|---|---|
| 1 | High | A file re-indexed to **zero chunks** (emptied, whittled below the 30-char minimum, or extraction now failing) had its mtime recorded but its old chunks never evicted — stale content served forever, and every later re-index a silent no-op. Root cause: `ingest_folder` returned only `(chunks, manifest, deleted)`; a changed-but-chunkless file was in neither `deleted` nor the new-chunk paths, so `apply_changes` never removed it. | `ingest_folder` now also returns `changed` (every re-processed path); the CLI evicts `deleted \| changed`. | `test_emptied_file_is_marked_changed_for_eviction`, `test_reindex_removes_stale_content_from_emptied_file` |
| 2 | Medium | A bare or half-written `.docwhisper/` directory (e.g. an `index` run interrupted after `mkdir` but before the files were written) crashed **every** command with an unhandled `FileNotFoundError` (exit 1) instead of the contracted friendly exit 2 — and blocked its own repair. | `Index.exists()` requires all three artifacts; `Index.load()` returns empty on a partial dir, so `index` rebuilds cleanly. | `test_partial_index_dir_does_not_crash` |
| 3 | Medium | `ask` caught only `httpx.ConnectError`. Ollama going down *mid-request*, a timeout, or an unknown-model 404 raised `ReadError` / `ReadTimeout` / `HTTPStatusError` — none subclasses of `ConnectError` — which escaped as a raw traceback (exit 1) instead of the contracted exit 3. | Broadened to `except httpx.HTTPError`, which covers all transport errors, timeouts, and non-2xx responses. | `test_ask_exit_3_on_any_http_error` (parametrized over four error types) |
| 4 | Low | `_split_long` emitted a redundant final window for some paragraph lengths — e.g. `chunk_text("…"*230, max_chars=150, overlap=50)` produced a 3rd chunk whose content lay entirely within the 2nd's overlap (zero new characters), duplicating embedded text. | Rewrote the hard-split as a `while` loop that stops once a window reaches the end of the paragraph. | `test_no_redundant_final_window` |

Findings the reviewers explicitly checked and cleared: cosine math (normalization,
non-unit/zero vectors, single-vector and empty index), save/load round-trips,
nested-directory relpaths and chunk-id uniqueness, hidden-directory skipping,
chunk packing/overlap invariants (fuzzed), argparse wiring, no fastembed import
or network access at test time, and lazy model loading keeping `search`/`stats`
fast.
