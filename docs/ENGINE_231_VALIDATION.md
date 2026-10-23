# Engine/UI 2.3.1 integration validation

Date: 2026-10-21. Branch: `feat/basal-ui-engine-231`.
Worktree: `/Users/pawelkiszczak/orca/workspaces/basal/basal-ui-engine-231`.

## Provenance and ownership

Started at committed UI base `7ea39dda27a91357a36b0c08840e4be020df3682`. Fetched upstream explicitly with
`git fetch upstream 621bc57004bde3df6cefa078e3421f21af7f795b` and merged that pinned commit as
`64850dbff160a8a913671643c62d8c8848ffc48d`. This includes 2.3.1 and the post-release facts fixes.
Merged the coordinator's subsequently committed UI base `8574fd3e8757b45ef9f068f03ddf85b4c0061c09` as
`80ff53b3d8ec74155b1e763214a15d2b428d4c5f`. Both are ancestors of the implementation commit containing this report.
The parent checkout and its dirty files were not edited or copied. No push or PR creation was performed.

The upstream conflicts in server construction were resolved around the fork's application, retaining authentication,
CORS, Decisions aliases and envelopes, request IDs, MCP model routing, and lifecycle hooks. Upstream request counting,
RequestError classification, OOM handling, engine changes, and startup changes were integrated.

## Requested scope

1. CPU packed is selectable with `mps`/`cpu`; eager remains the upstream CPU default. Omni defaults follow upstream
   supported Blackwell workstation and Hopper/FA3 rules. Unsupported selections retain reasons.
2. MLX packed defaults off, validates family/format/platform/mode/media eligibility, reaches the child and mode
   comparison caller, and survives save/restart/auto-load. Help explains the 512-token group limit and calibration caveat.
3. Existing batch selector exposes auto/compact/default/fine with backend validation, memory/startup help, preserved
   settings, and the existing engine-refusal/log path. Kernel acceptance remains the engine's responsibility.
4. Child health counters appear in `/api/status` and model rows. Null counters, idle, and unreachable differ. Polling
   is asynchronous, limited to one probe per child per five seconds, with 0.5-second request timeout and generation checks.
5. HTTP 422/503/500 remain distinct through both inference envelopes, Playground, gateway, Batch records, and job
   diagnostics. Request IDs and original messages are retained where supplied; no automatic request retries were added.
6. Actual report warning lines populate calibration results and file listings; jobs can finish `done_with_warnings`.
   Success does not imply confirmed certification. Reports are retained on failed calibration calls too.
7. State-cache help describes longest-prefix reuse and GPU budget competition; configured sizes remain runtime options.
   No cache-hit counters were invented.
8. Focused contract tests and the existing real-model/Chromium release gate were extended. Documentation was updated
   in SERVING, E2E, API, and WHATS_NEW.

## Commands and results

Environment: Apple M4 Max, 128 GiB system RAM, Python 3.13.11, PyTorch 2.13.0, MLX 0.32.3, Playwright 1.64.0.
`uv sync --extra test --extra e2e` created this worktree's virtualenv using the existing lockfile.

- `uv run pytest -q --ignore=tests/e2e`: **844 passed, 53 skipped**, 46.96 seconds. One upstream warning about
  uncompiled FlexAttention in its test. This run preceded one additional benchmark-diagnostic test and the final
  narrow saved-option validation adjustment; the affected suites were then rerun below.
- `uv run pytest -q tests/test_ui_engine231.py tests/test_ui_options.py tests/test_ui_bench.py tests/test_ui_calibration.py`:
  **62 passed**, 3.09 seconds, after the final implementation edits. This includes 32 integration contract cases.
- Real release commands used this environment prefix (offline, no downloads):
  `BASAL_E2E=1 HF_HUB_OFFLINE=1 BASAL_E2E_MODEL=Remek/basal-2.0-mini-MLX-8bit BASAL_E2E_BF16_MODEL=Remek/basal-2.0-mini`.
  With that prefix, `uv run pytest -q tests/e2e/test_calibration.py tests/e2e/test_load_options.py tests/e2e/test_lifecycle.py tests/e2e/test_batch.py`:
  **11 passed**, 52.34 seconds.
- With the same prefix, `uv run pytest -q tests/e2e/test_engine231.py`: **3 passed**, 21.14 seconds.
  CPU packed and MLX packed each loaded, answered, and restarted on real hardware using existing local model weights.
- With the same prefix, `uv run pytest -q tests/e2e/test_engine231.py tests/e2e/test_pages.py -k 'engine231 or load_options_dialog or batch_view'`:
  **5 passed, 17 deselected**, 22.41 seconds. Actual Chromium drove the UI. Injected HTTP errors, health edge cases,
  and warning text are explicitly simulated, while the app and ordinary inference paths are real.
- `git diff --check` passed. Both pinned upstream and the latest committed UI base passed ancestor checks.
  Real release fixtures verified no child processes survived teardown; a subsequent process check found none.

During development, the new release test initially failed on a nonexistent test helper and a Playwright route callback
signature. Both test-harness errors were fixed before the successful runs above. No failing acceptance case remains.

## Limits and review hotspots

NVIDIA/Blackwell, CUDA compact/fine performance and memory, TPU, ROCm, external engines, and physical GPU OOM recovery
were not accepted on real hardware. CUDA routing and status failures are covered with stubs; CPU and MLX inference
are real M4 Max checks. The 53 broad-suite skips are environment/hardware-gated tests, not passing evidence.
No CI, deployment, or production acceptance is claimed. No large model was downloaded; only existing model caches
were read, and settings/history/calibration/batch output used isolated temporary directories.

Review the merged `build_app` validation boundary, asynchronous health generation guard, report-warning parsing,
and `done_with_warnings` consumers. MLX mode comparisons now receive the saved packed option and record it per row;
other benchmark options keep their existing scope. Health figures are sampled and may lag by one polling interval.

## PR 9 regression follow-up

Shared request parsing now rejects invalid act `max_error` and threshold multi/tool `min`/`max` before inference,
returning 422 while retaining legacy numeric-string coercion. Internal engine failures remain 500 and OOM remains 503.
Long-lane cache misses use the existing gated layered prefill, so queued short requests can run between layers.
This recomputes the root even when a partial prefix is cached; exact long-lane hits and main-lane incremental reuse
remain enabled. Incremental long-lane reuse can return when prefix extension supports the scheduling gate.

- Red check: `.venv/bin/python -m pytest -q tests/test_http_status.py -k assembly_options tests/test_state_cache.py -k 'assembly_options or growing_state_in_long_lane'`:
  **6 failed, 3 passed, 25 deselected**, 1.41 seconds, reproducing five incorrect 500 responses and missing callbacks.
- Green check: `.venv/bin/python -m pytest -q tests/test_http_status.py tests/test_state_cache.py tests/test_long_lane.py tests/test_engine16.py tests/test_sets.py tests/test_tools.py tests/test_tasks.py`:
  **378 passed**, 5.25 seconds. The new cache regression checks scheduling callbacks, answer equality, and subsequent exact-hit reuse.
- Compatibility check: `.venv/bin/python -m pytest -q tests/test_server.py tests/test_api_compat.py tests/test_auth.py tests/test_mcp.py tests/test_ui_engine231.py tests/test_ui_gateway.py tests/test_engine.py tests/test_gemma4_graph.py tests/test_prefix_cache.py`:
  **188 passed**, 5.80 seconds, with no skips.

These follow-up checks use the existing virtualenv, HTTP stubs and tiny local CPU models; no downloads or real GPU,
browser, latency, or throughput acceptance was performed for these two fixes. The partial-prefix long-lane fallback
trades incremental prefill speed for preserved scheduling. Publication remains with the coordinator.

**Reverted:** the long-lane part of this follow-up. `GraphBackend.run_long` runs upstream's prefix extension
(`_extend_states`) again, as in rkinas/basal, and `test_growing_state_in_long_lane_still_yields_and_matches_uncached_answers`
is removed. The request validation above stays.
