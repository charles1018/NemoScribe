# FUTURE_AGENT_REPO_GUIDE.md

> **Audience**: future AI coding agents (Claude Code Opus/Sonnet/Haiku, Codex CLI, etc.) working on NemoScribe.
> **Written**: 2026-07-07, by a Claude Fable 5 session that inspected the full repository and ran the verification suite.
> **Status of this file**: untracked at creation time. Whether to commit it is the maintainer's decision (note: `CLAUDE.md`, `AGENTS.md`, and `.claude/` are deliberately kept local-only via `.git/info/exclude` — ask the user before committing this file, since they may want the same treatment).

---

## 1. Executive Summary

**What this repo is**: NemoScribe is a small (~3,700 lines of package code), single-maintainer Python CLI tool that converts video files into SRT subtitles using NVIDIA NeMo ASR (default model: `nvidia/parakeet-tdt-0.6b-v2`). It runs fully locally on an NVIDIA GPU, with optional VAD-based smart chunking, ITN, and LLM-based subtitle correction (OpenAI/Anthropic). Managed with `uv`; published at `github.com/charles1018/NemoScribe`; currently at v0.5.0.

**What is strong**:
- Unusually good agent-oriented documentation: `AGENTS.md` (local), `CLAUDE.md` (local), and `UPGRADE_NEMO.md` (committed) give explicit commands, pins, and validation steps.
- A self-contained regression suite (`tests/test_improvements.py`, 22 tests) that runs on CPU with no video files or API keys. Verified passing 22/22 in this session. `ruff check .` is clean.
- Dependency pins are documented *with reasons* (transformers/numpy/onnx/huggingface-hub), and there is a written log of failed experiments (GPU-PB context biasing, NGPU-LM, parakeet-unified) so agents don't re-try dead ends.
- Defensive compatibility shims against NeMo API drift (`_add_decoding_kwarg`, `_transcribe_with_hypotheses`, `_get_hypothesis_timestamps`).

**What is risky or unclear**:
- Some local-only documentation drift remains (`CLAUDE.md` still labels `--test full` as comprehensive unless manually corrected; the committed NeMo upgrade runbook was fixed on 2026-07-08 — see §3.1).
- CI is currently lint-only; full tests and E2E verification remain manual/local because full dependency installation is GPU/CUDA-heavy.
- LLM live-API behavior is only covered by fallback tests unless the maintainer explicitly runs a real paid-provider sample.

**Top 3 priorities for future agents** (details in §5):
1. **P0** — ✅ done 2026-07-08, commit `e3611c9`: fix the committed `--test full` documentation trap and remove the stray root `__init__.py`. Local `CLAUDE.md` wording still needs user-approved correction.
2. **P1** — ✅ done 2026-07-08, pending commit: refresh LLM post-processing model defaults and cost documentation using official provider docs.
3. **P1** — ✅ partially done 2026-07-08, pending commit: lint-only GitHub Actions workflow added. Full test CI remains a cost/dependency decision.

**The single most important warning**: *Never* upgrade `nemo-toolkit`, `torch`, `transformers`, `huggingface_hub`, `onnx`, CUDA indexes, or `uv.lock` by editing version strings alone. Read `UPGRADE_NEMO.md` first and follow its full validation checklist. The dependency matrix is fragile and every pin exists for a documented reason.

---

## 2. Repository Map

```
NemoScribe/
├── nemoscribe/            # The package. This is where almost all real work happens.
├── tests/                 # One integration-style test file with its own runner.
├── scripts/               # Standalone analysis/benchmark utilities.
├── docs/                  # Bilingual tuning guides.
├── tmp_outputs/           # Untracked local experiment outputs. Do not commit.
├── README.md / README.zh-TW.md   # User docs (bilingual, must stay in sync).
├── CHANGELOG.md           # Keep a Changelog format; update on user-visible changes.
├── UPGRADE_NEMO.md        # Committed runbook for NeMo stack upgrades. Read before dependency work.
├── V2_TEST_REPORT.md      # Historical LLM post-processing test evidence (reference only).
├── pyproject.toml         # Dependencies, pins, uv indexes, ruff config. High-risk file.
├── uv.lock                # Locked resolution. Only regenerate via `uv lock`, never hand-edit.
├── .env / .env.example    # API keys. NEVER read or print `.env` contents.
├── CLAUDE.md / AGENTS.md / .claude/   # LOCAL-ONLY (excluded via .git/info/exclude; absent in fresh clones).
└── .serena/               # Local tool config, excluded from git.
```

### `nemoscribe/` package (edit freely, with tests)

| File | Purpose | Notes for editors |
|------|---------|-------------------|
| `cli.py` (476 ln) | Hydra-style `key=value` arg parsing, orchestration (`main`, `process_videos`), type coercion (`_coerce_value`, `_set_typed_attr`, `parse_args`) | Nested key parsing supports exactly **two levels** (`parent.child`), not unlimited depth. If you ever nest config three levels deep, you must extend `parse_args` (`cli.py:405`). Update `tests/test_improvements.py` `cli`/`cli_list` tests when touching parsing. |
| `config.py` (318 ln) | All dataclass configs, rooted at `VideoToSRTConfig` | Every new field needs: type hint (parser relies on `get_type_hints`), default preserving backward compatibility, doc-table updates in 4 doc files (see §11), and a test. |
| `transcriber.py` (679 ln) | Model loading, decoding strategy setup, the main `transcribe_video()` pipeline (chunk → transcribe → clip → merge → dedup → ITN → LLM → SRT) | The largest and most load-bearing module. Contains NeMo version-compat logic — preserve the `_add_decoding_kwarg` / `_transcribe_with_hypotheses` fallback patterns. |
| `vad.py` (409 ln) | VAD inference, silence-gap detection, smart chunk splitting | Verified quality feature (76% max-segment reduction came from this + segment separators). Don't regress casually; validate with a real run. |
| `srt.py` (403 ln) | Hypothesis → segments (3-strategy fallback), SRT read/write, window clipping | `hypothesis_to_srt_segments` is the subtitle-quality heart. Segment tuples are `(start: float, end: float, text: str)` everywhere in the codebase. |
| `audio.py` (241 ln) | ffmpeg/ffprobe subprocess wrappers, `validate_media_path()` | `validate_media_path()` is the **security boundary** for subprocess calls. Do not weaken. All ffmpeg calls must go through validated paths. |
| `postprocess.py` (305 ln) | ITN (optional dep), overlapping-segment merge, dedup | Merge/dedup handles chunk-boundary artifacts; changing thresholds changes output subtitles — verify with a real run. |
| `llm_postprocess.py` (627 ln) | LLM correction with Agent Loop (LLM → validate → feedback → retry), provider routing | Must preserve segment count, order, and timestamps. AGENTS.md: "Treat ordering bugs as high severity." Anthropic/OpenAI batch loops are near-duplicates (refactor seam, §5 P2). Config class lives here (imported by `config.py`), not in `config.py`. |
| `log_utils.py` (130 ln) | Context manager filtering repetitive NeMo logs | Low risk. Filter patterns may need updating after NeMo upgrades. |
| `__init__.py` / `__main__.py` | Public API exports / `python -m nemoscribe` | Keep `__all__` in sync when adding public functions. `__version__` lives in `nemoscribe/__init__.py` and must match `pyproject.toml`. |

### `tests/`

- `tests/test_improvements.py` (1,482 ln): a **self-contained runner, not pytest**. `uv run python tests/test_improvements.py` runs all 22 tests; `--test <name>` runs one. Do not assume pytest exists (AGENTS.md explicitly says so). Tests are config/logic-level: they need the NeMo import stack but no GPU, no video files, no API keys, no model downloads.
- Who edits: any agent changing config, parsing, SRT logic, or LLM post-processing **must** add/update tests here.

### `scripts/` (standalone utilities, lower quality bar)

- `check_cuda.py` — CUDA sanity check (13 lines).
- `analyze_srt_stats.py`, `find_longest_segment.py` — import `parse_srt_timestamp` from `nemoscribe.srt` (the intended pattern).
- `evaluate_benchmark.py`, `analyze_quality.py` — **duplicate their own SRT parsing/normalization** instead of importing from the package (drift from the documented convention; cleanup candidate, §5 P2). Both take explicit `--test-dir`/`--reference` style args; reference SRTs live outside the repo at `~/dev/tools/claude/subtitle-workbench` (local machine only — unavailable in fresh clones/cloud).

### Do-not-touch / handle-with-care

- `uv.lock` — regenerate only, per `UPGRADE_NEMO.md`.
- `pyproject.toml` `[tool.uv] constraint-dependencies` and the cu130 index — every entry has a documented reason.
- `.env` — never read, print, or commit.
- `audio.validate_media_path()` and its call sites — security boundary.
- `tmp_outputs/`, generated `.srt`/`.wav` files — gitignored; never commit.

---

## 3. Current Workflow Diagnosis

### 3.1 The `--test full` trap (highest-impact drift, partially fixed)

- **Symptom**: Docs tell agents to run `uv run python tests/test_improvements.py --test full` as "comprehensive/all tests".
- **Evidence**: In `tests/test_improvements.py:1447`, `"full"` maps to `test_full_config` — a *single* test that checks a fully-populated config object. The actual full suite (22 tests) is the default `all` (no `--test` flag), see `run_all_tests` at `tests/test_improvements.py:1347`. Local `CLAUDE.md` says "full: All tests (19 total)" — wrong on both the mapping and the count. `UPGRADE_NEMO.md` was corrected on 2026-07-08, pending commit.
- **Why it hurts**: An agent doing a risky NeMo upgrade will run `--test full`, see one trivial pass, and conclude the suite is green. This defeats the entire purpose of the upgrade runbook.
- **Fix**: `UPGRADE_NEMO.md` now says to run with no `--test` flag. Local `CLAUDE.md` should be updated with user approval to say "run with no `--test` flag, or `--test all` (both run all 22 tests)". Note that `--test all` **already works** — it is the default (`tests/test_improvements.py:1399`) and is intercepted before `test_map` at `tests/test_improvements.py:1422` — so no runner change is needed.
- **Autonomous-safe?** Yes for docs (except `CLAUDE.md`, which is a local file — propose the edit to the user).

### 3.2 Stray tracked `__init__.py` at repo root — ✅ deleted 2026-07-08, pending commit

- **Symptom**: `/__init__.py` (repo root, git-tracked, on GitHub) carried an NVIDIA Apache-2.0 header and did `from .nemoscribe import extract_audio, main, process_videos, transcribe_video`.
- **Evidence**: `__init__.py:15` at repo root. This references a layout from before the package split (`nemoscribe.py` single file); today `extract_audio` lives in `nemoscribe.audio` and is not exported by `nemoscribe/__init__.py`. The file is dead code: the repo root is never imported as a package, and hatchling builds only the `nemoscribe/` package.
- **Why it hurts**: It confuses repo mapping (two `__init__.py` at different levels), carries a misleading license header, and an agent might "fix" it by re-exporting things instead of deleting it.
- **Fix**: Delete the root `__init__.py`. Verify with: `grep -rn --include="*.py" "from \.nemoscribe" .` (should return nothing after deletion; without `--include`, this guide file itself also matches), then `uv sync --python 3.12 && uv run python tests/test_improvements.py` and `uv run nemoscribe --help`.
- **Autonomous-safe?** Yes (with the verification above). It is tracked, so this is a commit-worthy change — follow the repo's commit conventions and let the user push.

### 3.3 Verification depends on unreproducible local assets

- **Symptom**: Real quality verification (WER benchmarks, VAD tuning, "do a real transcription run before concluding") requires: an NVIDIA GPU (maintainer has an RTX 3070 8GB), local video files (`~/Videos/...`), and reference SRTs in a sibling repo (`~/dev/tools/claude/subtitle-workbench`). None exist in CI or a fresh clone.
- **Evidence**: `AGENTS.md` End-to-End Runs section; `CLAUDE.md` benchmark sections; local wrapper `run_evaluate.sh` (local-only, in `.git/info/exclude`).
- **Why it hurts**: Agents either skip E2E verification (risky) or burn time hunting for files that don't exist in their environment. Cloud/worktree agents can never fully verify transcription quality.
- **Fix (partial)**: (a) Document the tiering explicitly (this file, §7): what can be verified anywhere (lint + 22 tests) vs. only on the maintainer's machine (GPU E2E, WER benchmarks). (b) Optionally add a tiny synthetic smoke asset (e.g., generate a 10-second WAV with `ffmpeg -f lavfi -i sine` plus a TTS-free silence/tone test that at least exercises `extract_audio`/chunking without ASR). Full ASR smoke testing in CI would download a 0.6B model — needs user judgment.
- **Autonomous-safe?** (a) yes; (b) requires user judgment on scope/cost.

**Other smaller drifts found (fix alongside 3.1):**
- `CLAUDE.md` claims CLI nested keys handle "unlimited depth (e.g., `parent.child.grandchild=value`)" — false; `parse_args` (`nemoscribe/cli.py:405-439`) handles exactly two levels. Harmless today (config is two levels deep) but a trap for anyone adding deeper nesting.
- `CLAUDE.md` claims analysis scripts "use shared `parse_srt_timestamp()` from `nemoscribe.srt`" — true only for 2 of the 4 SRT-consuming scripts (see §2).
- `README.md` badge/table says CUDA 13.0 (cu130) — consistent with `pyproject.toml` today, but re-check on every torch bump.

---

## 4. Architecture Understanding

### Runtime components and data flow

```
nemoscribe CLI (cli.py:main)
  └─ parse_args: Hydra-style key=value → VideoToSRTConfig (dataclasses, config.py)
  └─ process_videos (cli.py:148)
       ├─ check_ffmpeg / device+dtype selection (NeMo transcribe_utils)
       ├─ load_asr_model  ── HuggingFace/NGC download on first use (network!)
       ├─ setup_decoding_strategy  ── RNNT vs CTC branch, version-adaptive kwargs
       ├─ load_vad_model (optional; failure degrades gracefully to no-VAD)
       ├─ get_itn_normalizer (optional; graceful None)
       └─ per video → transcribe_video (transcriber.py:401)
            ├─ ffprobe duration → chunking decision (audio.max_chunk_duration, default 300s)
            ├─ [VAD path] extract full audio → run_vad_on_audio → speech segments
            │     → create_audio_chunks_with_vad (split at silences, vad.py)
            ├─ [no-VAD path] create_audio_chunks (fixed windows + 2s overlap, audio.py)
            ├─ per chunk: NeMo model.transcribe(timestamps=True) → Hypothesis
            │     → hypothesis_to_srt_segments (srt.py; segment-ts → word-ts → estimate fallback)
            │     → time-offset shift → clip_segments_to_window
            ├─ merge_overlapping_segments + deduplicate_segments (postprocess.py)
            ├─ apply_itn_to_segments (optional)
            ├─ postprocess_subtitles (optional LLM; network + API key; llm_postprocess.py)
            └─ write_srt_file
```

- **State**: none persisted. Temp audio in a `tempfile.TemporaryDirectory` (auto-cleaned). Model weights cached by HF/NGC in the user's home cache. ITN normalizer cached in a module-global dict. `.env` auto-loaded by `llm_postprocess.py` import (via python-dotenv if installed).
- **The universal data shape**: `List[Tuple[float, float, str]]` — `(start_sec, end_sec, text)`. Every pipeline stage consumes and produces this. Preserve it.
- **External services**: HuggingFace/NGC (model download, first run only), OpenAI/Anthropic APIs (only when `llm_postprocess.enabled=true`). Everything else is offline.
- **Build/deployment**: none. No CI, no PyPI publishing observed, no Docker. "Deployment" = users `git clone` + `uv sync --python 3.12`. Releases are git tags + CHANGELOG entries.
- **Important boundaries**:
  - NeMo API surface: enumerated exhaustively in `UPGRADE_NEMO.md:17-29`. Any NeMo upgrade must re-verify those imports.
  - Subprocess boundary: only ffmpeg/ffprobe, only via `validate_media_path()`-resolved paths.
  - LLM boundary: `postprocess_subtitles()` guarantees fallback-to-original on any failure; timestamps are structurally preserved because the LLM only returns text keyed by segment index.
- **Known unknowns (unverified in this session)**:
  - Windows support (README claims it; nothing here tested it).
  - Actual GPU behavior of current lockfile (no GPU run performed this session).
  - Whether NeMo 2.8.0 has since been released on PyPI (repo analysis dated 2026-04-04 said not yet; check before any upgrade).
  - Whether the hosted parakeet API mentioned in README still exists.

---

## 5. High-Leverage Improvement Roadmap

Ordered. Each item is scoped for one agent session.

### P0-1: Fix documentation/code drift around test invocation and CLI parsing claims — ✅ done 2026-07-08, pending commit
- **Problem**: §3.1 trap plus the "unlimited depth" and "shared parse_srt_timestamp" claims.
- **Why it matters**: These are instructions agents *will follow*; the `--test full` one directly undermines upgrade validation.
- **Direction**: Doc-only. In `UPGRADE_NEMO.md`, replace the `--test full` step with the no-flag full run (or `--test all`, which already works) and note it runs 22 tests. Propose (do not silently apply) matching corrections to local `CLAUDE.md`/`AGENTS.md` to the user.
- **Files**: `UPGRADE_NEMO.md`, user-approved edits to `CLAUDE.md`.
- **Steps**: (1) grep docs for `--test full`; (2) edit; (3) run the full suite to confirm the documented command works as written.
- **Verify**: `uv run ruff check .` clean; `uv run python tests/test_improvements.py` → 22/22.
- **Risks**: minimal. **Stop and ask**: only for `CLAUDE.md`/`AGENTS.md` wording (local files, user's own notes).

### P0-2: Delete the stray root `__init__.py` — ✅ done 2026-07-08, pending commit
- Covered in §3.2 (problem, direction, verification). One-file deletion + verification run. **Risks**: near zero; hatchling targets `nemoscribe/`. **Stop and ask**: no — but present the diff clearly and do not push.

### P1-1: Refresh LLM post-processing model defaults and cost docs — ✅ done 2026-07-08, pending commit
- **Problem**: The default model used to be `claude-3-5-sonnet-20241022`, and docs referenced outdated model generations and per-episode cost estimates. Old dated model IDs eventually get deprecated by providers, at which point LLM post-processing breaks at runtime for default-config users.
- **Why it matters**: This is the only feature that calls paid external APIs; a dead default model is a hard failure the fallback machinery cannot fully hide (every batch will error and fall back to uncorrected text).
- **Direction**: Verified provider docs on 2026-07-08. Default is now `claude-sonnet-5`; OpenAI examples use `gpt-5.4-mini`; docs now use per-MTok pricing instead of fixed per-episode estimates. Keep `provider`/`model` fully user-overridable (already true).
- **Files**: `nemoscribe/llm_postprocess.py`, `README.md`, `README.zh-TW.md`, `docs/TUNING_GUIDE.md`, `docs/TUNING_GUIDE.zh-TW.md`, `CHANGELOG.md`, tests `llm`/`llm_cli` in `tests/test_improvements.py` (they may assert the default model string — check `test_llm_config`).
- **Verify**: full test suite; if the user provides an API key, one real run of `llm_postprocess.enabled=true provider=anthropic` on a short SRT; otherwise state clearly that live-API behavior was not exercised.
- **Risks**: choosing a wrong/hallucinated model ID. Mitigation: fetch provider docs, or ask the user which models they pay for. **Stop and ask**: if unsure which provider tier the user wants as default.

### P1-2: Minimal CI (lint + CPU test suite) — lint-only ✅ done 2026-07-08, pending commit
- **Problem**: No `.github/workflows/`. Nothing prevented a broken commit.
- **Why it matters**: Every future agent currently re-runs verification manually; regressions reach `main` silently.
- **Direction**: `.github/workflows/lint.yml` now runs `uvx --from ruff==0.15.8 ruff check .` on push and pull requests. Full test CI is still deferred because the suite imports torch+NeMo; installing the pinned cu130 torch in CI is multi-GB and the `[[tool.uv.index]]` forces the CUDA wheel. Options for a future change: (a) full-deps CI with cache (slow first runs), (b) a CPU-torch override that must not change local GPU resolution, or (c) split NeMo-free tests.
- **Files**: `.github/workflows/lint.yml`; no `pyproject.toml` or lockfile changes.
- **Verify**: local `uvx --from ruff==0.15.8 ruff check .`; workflow should be checked after push.
- **Risks**: touching `pyproject.toml` indexes counts as NeMo-sensitive change → full UPGRADE_NEMO.md process. **Stop and ask**: before adding full test CI or any `pyproject.toml` change.

### P2-1: Deduplicate SRT parsing in `scripts/`
- **Problem**: `scripts/evaluate_benchmark.py:59` and `scripts/analyze_quality.py:42` re-implement SRT timestamp parsing and text normalization instead of importing from `nemoscribe.srt` (the other two scripts do it right).
- **Direction**: Import `parse_srt_timestamp` from `nemoscribe.srt`; consider moving a shared `parse_srt_file`/`normalize_text` into `nemoscribe/srt.py` if both scripts need it. Keep script CLIs unchanged.
- **Verify**: run both scripts against `tmp_outputs/unified_eval/*.srt` (present on maintainer's machine) or any generated SRT; identical output before/after (capture before-output first).
- **Risks**: subtle normalization differences between the duplicated copies — diff them before unifying; if they intentionally differ, document why instead of merging. **Stop and ask**: no.

### P2-2: Unify the Anthropic/OpenAI batch loops in `llm_postprocess.py`
- **Problem**: `postprocess_subtitles_anthropic` (`llm_postprocess.py:434`) and `postprocess_subtitles_openai` (`llm_postprocess.py:499`) are ~60-line near-identical copies differing only in client construction and availability flag.
- **Direction**: Extract one `_postprocess_with_provider(segments, config, api_key, provider)`; keep the two public function names as thin wrappers (they're exported implicitly via module use — check `nemoscribe/__init__.py`; only `postprocess_subtitles` is exported, so internals are free to move).
- **Verify**: `--test llm_parsing`, `--test llm_validation`, `--test llm_fallback`, `--test llm_validation_fallback`, then full suite.
- **Risks**: low; pure refactor with test coverage. **Stop and ask**: no.

### P2-3: NeMo 2.8.x upgrade (when released on PyPI)
- **Problem**: Pinned `<2.8`; 2.8 brings a relevant perf fix (forced decoder reinstantiation with `timestamps=True`, upstream #15298). Compatibility pre-analysis (2026-04-04, in local CLAUDE.md) found the used API surface stable.
- **Direction**: Follow `UPGRADE_NEMO.md` **to the letter** (mode decision, `uv lock`, import sanity block, focused tests, full suite, real GPU smoke test). Update `<2.8` → `<2.9`, re-check transformers constraint compatibility.
- **Verify**: everything in UPGRADE_NEMO.md §Required Validation Steps; the real-transcription step needs the maintainer's GPU machine.
- **Risks**: transitive churn (transformers/huggingface_hub conflicts are the known failure mode). **Stop and ask**: before merging if any "Escalate before merging" condition in UPGRADE_NEMO.md triggers; also confirm 2.8 is actually on PyPI first.

### P3: Deferred product ideas (need user judgment — do not start unprompted)
- LLM V3 improvements (conservative mode, cross-batch name context, user character hints via `.claude/character_names_example.txt`, number guard) — deliberately deferred at v0.4.0.
- Speaker diarization (idea noted in maintainer's memory/watchlist).
- Multilingual support via parakeet-tdt-0.6b-v3 (kept as *option*, not default — v2 beat v3 on English; do not flip the default).

---

## 6. Suggested Future Agent Tasks

### Task A (documentation/agent instructions): "Sync verification docs with the real test runner" — mostly done 2026-07-08, pending commit
- **Goal**: No committed or local doc misdescribes how to run tests.
- **Motivation**: §3.1 — `--test full` ≠ full suite; test count is 22, not 19.
- **Scope**: `UPGRADE_NEMO.md` is fixed; propose text for `CLAUDE.md`/`AGENTS.md` to the user. No code changes needed (`--test all` already runs the full suite).
- **Out of scope**: converting to pytest; renaming existing test IDs; README test tables unless wrong.
- **Acceptance criteria**: `grep -rn '\-\-test full' *.md docs/` returns only lines that correctly describe `test_full_config`; full suite still 22/22.
- **Commands**: `uv run python tests/test_improvements.py` ; `uv run python tests/test_improvements.py --test full` ; `uv run ruff check .`
- **Report format**: changed files list, before/after excerpt of each corrected claim, test summary line, note on which CLAUDE.md changes were only *proposed*.

### Task B (tests/verification): "Add regression tests for the CLI parser's actual depth behavior"
- **Goal**: Lock in (and make explicit) that `parse_args` supports two levels, warns on unknown keys, and rejects deeper nesting gracefully.
- **Motivation**: The "unlimited depth" doc claim (§3.3 extras) means someone may rely on it; a test makes reality discoverable.
- **Scope**: extend `test_cli_config_override` (or add `test_cli_nested_depth`) in `tests/test_improvements.py`: cases for `vad.onset=0.2` (works), `a.b.c=1` (warning, no crash), unknown parent/child (warning), `subtitle.word_gap_threshold=null` (None), enum-style and empty-list coercions if not already covered.
- **Out of scope**: implementing 3-level support (that's a feature decision).
- **Acceptance criteria**: new assertions pass; full suite 22+/22+ green; runner `--test` help string updated if a new test name is added.
- **Commands**: `uv run python tests/test_improvements.py --test cli` ; full suite ; `uv run ruff check .`
- **Report format**: list of new cases, suite summary, any surprising parser behavior found.

### Task C (architectural risk): "Execute the NeMo 2.8 upgrade runbook" (only when 2.8 is on PyPI)
- **Goal**: Land `nemo_toolkit[asr]>=2.8,<2.9` with full validation.
- **Motivation**: perf fix for chunked transcription; staying one major behind accumulates risk.
- **Scope / commands / acceptance**: exactly `UPGRADE_NEMO.md` (all five validation stages). GPU smoke test must run on the maintainer's machine — if you are a cloud agent, stop after stage 3 and hand off with explicit remaining steps.
- **Out of scope**: any other dependency bump in the same change; model default changes.
- **Report format**: mode chosen (PyPI/repo tracking), lockfile diff summary, each validation stage's exact command + result, escalation triggers hit (if any).

### Task D (developer workflow): "Lint-only GitHub Actions workflow" (after user approval)
- **Goal**: `ruff check .` runs on every push/PR.
- **Motivation**: cheapest possible regression net given the heavy dependency stack (§P1-2 option (a)).
- **Scope**: `.github/workflows/lint.yml` using `astral-sh/setup-uv` or plain `pipx run ruff==<pinned>`; pin the ruff version to match local (`ruff --version` locally first).
- **Out of scope**: running the test suite in CI (separate decision, §P1-2); modifying `pyproject.toml`.
- **Acceptance criteria**: workflow green on a test branch; no changes to lockfile or pyproject.
- **Commands**: local `uv run ruff check .` must match CI result.
- **Report format**: workflow file content, run URL/result, confirmation nothing else changed.

### Task E (cleanup, justified): "De-duplicate scripts' SRT parsing" — see roadmap P2-1 for full spec. Only do this when touching those scripts anyway or when explicitly asked; it is justified by the documented-but-violated convention, not urgency.

---

## 7. Verification Strategy

**Tier 1 — any environment (verified working in this session, 2026-07-07):**

| Purpose | Command | Session result |
|---|---|---|
| Install | `uv sync --python 3.12` | not re-run (venv already present); documented in AGENTS.md |
| Lint | `uv run ruff check .` | ✅ "All checks passed!" |
| Full test suite | `uv run python tests/test_improvements.py` | ✅ `Total: 22 \| Passed: 22 \| Failed: 0` (~1–2 min incl. NeMo import) |
| Single test | `uv run python tests/test_improvements.py --test <name>` | valid names: baseline, vad, itn, decoding, nemo_api, segmentation, merging, performance, ab_test, metrics, srt, srt_edge, path, cli, cli_list, llm, llm_cli, llm_validation, llm_parsing, llm_fallback, llm_validation_fallback, full — plus `all` (the default), which runs the whole 22-test suite |
| CLI smoke (no model download) | `uv run nemoscribe --help` | prints help + full config YAML |

⚠️ `--test full` runs ONE test (`test_full_config`). The full suite is the **no-flag** invocation.

**Tier 2 — maintainer's machine only (GPU + local media):**
- CUDA check: `uv run python scripts/check_cuda.py`
- Real transcription: `uv run nemoscribe video_path=<file>` (+ VAD flags from README). First run downloads ~0.6B model from HF (network, ~minutes).
- Quality benchmarks: `scripts/evaluate_benchmark.py --test-dir <dir> --reference <ref.srt>` (references in `~/dev/tools/claude/subtitle-workbench`, local-only); SRT stats via `scripts/analyze_srt_stats.py` / `find_longest_segment.py`.
- 8GB-GPU caveat: with `compute_dtype=float32` on dialogue-dense content, drop `audio.max_chunk_duration` to ~120 to avoid OOM.

**Unavailable / not runnable here:**
- No typecheck command exists (no mypy/pyright config). Don't invent one; if adding, that's a user-approved workflow change.
- No build/packaging pipeline to verify (hatchling config exists but nothing publishes).
- LLM live-API paths need `OPENAI_API_KEY`/`ANTHROPIC_API_KEY` (a `.env` exists locally — never print it; only check key *presence* via `python -c "import os;print(bool(os.getenv('ANTHROPIC_API_KEY')))"` inside `uv run` if strictly needed).
- ITN tests self-skip gracefully if `nemo_text_processing` isn't installed (`uv sync --extra itn` to enable).

**Rule of thumb**: config/parsing/SRT-logic changes → Tier 1 is sufficient. Anything touching VAD behavior, chunking, decoding, merging thresholds, or dependencies → Tier 1 **plus** a Tier 2 real run before declaring success (AGENTS.md Change Discipline requires this; if you can't run Tier 2, say so explicitly and hand off).

---

## 8. Risk Register

| # | Risk | Evidence | Impact | Likelihood | Mitigation | Agent-safe? | Human needed when |
|---|------|----------|--------|-----------|------------|-------------|-------------------|
| 1 | Careless NeMo-stack upgrade breaks resolution or decoding | Fragile pins documented in `pyproject.toml` + local CLAUDE.md tables (transformers <4.58, numpy <2, onnx <1.18, hf-hub <1.0) | High (tool unusable) | Medium (agents love bumping deps) | Hard rule: read `UPGRADE_NEMO.md` first; never edit version strings alone | Yes, if runbook followed | Any "Escalate before merging" trigger; GPU smoke test |
| 2 | Agent runs `--test full`, believes suite is green | §3.1 | High (false verification) | Medium until local docs are fixed | `UPGRADE_NEMO.md` fixed; local `CLAUDE.md` still needs user-approved correction | Yes | — |
| 3 | Subtitle-quality regression invisible to the test suite | Tests are config/logic-level; quality only measurable via GPU benchmark runs | Medium-High | Medium | Require Tier 2 run for pipeline-behavior changes; keep VAD/segment-separator defaults untouched | Partially | Judging WER/subjective quality trade-offs |
| 4 | LLM default model gets deprecated by provider | Default model IDs are external provider contracts | Medium (feature silently degrades to no-op corrections) | Medium over time | Re-check official provider docs before changing defaults | Yes, with doc verification | Choosing default provider/cost tier |
| 5 | Secrets leakage | Real `.env` in repo root (gitignored); auto-loaded on import | High | Low | Never `cat .env`; never commit; AGENTS.md already forbids printing | Yes (by abstaining) | — |
| 6 | LLM post-processing reorders/miscounts segments after a refactor | Past bug fixed in commit `f6b9936`; AGENTS.md marks ordering as high severity | High (subtitles desync) | Low-Medium | Keep count/order/timestamp invariants; run all four `llm_*` tests | Yes | — |
| 7 | Doc quadruplication drifts (README ×2, TUNING_GUIDE ×2) | 4 parallel files must stay in sync per Documentation Sync rules | Low-Medium (user confusion) | High | Always edit all four together; grep for the option name across docs | Yes | — |
| 8 | Fresh-clone agents lack CLAUDE.md/AGENTS.md context | `.git/info/exclude` hides them; GitHub copy has neither | Medium (conventions invisible) | Certain for cloud agents | This file (if the user commits it) is the only in-repo agent guide besides UPGRADE_NEMO.md | — | User decides whether to commit this file |
| 9 | `merge_overlapping_segments`/`deduplicate_segments` drop legitimate rapid dialogue | "prefer_longer" keeps one text over overlaps (`postprocess.py:219`); Jaccard dedup at 0.8 | Low (accepted trade-off) | Low | Don't tune thresholds without benchmark evidence | No (needs benchmarks) | Threshold changes |
| 10 | Re-attempting documented dead ends | GPU-PB, NGPU-LM, parakeet-unified/v3-as-default all evaluated & rejected with data | Medium (wasted sessions) | Medium | Check "Failed Optimizations" in local CLAUDE.md / §5 P3 here before proposing model/decoding changes | Yes (by reading first) | Re-opening a rejected direction requires user request |

---

## 9. Agent Operating Guidelines For This Repo

1. **Session start**: read (in order) `AGENTS.md` (if present locally), this file, then the specific module you'll touch. For dependency work, `UPGRADE_NEMO.md` is mandatory. In a fresh clone without local files, this file + `UPGRADE_NEMO.md` + `README.md` are your context.
2. **Language protocol**: talk to the user in Traditional Chinese (Taiwan); write code, comments, commit messages, and docs-in-repo in English.
3. **Tooling protocol**: `uv` for everything Python (`uv run`, `uv sync`, `uv lock`). Never bare `python`/`pip`/`pytest`. Python 3.12.
4. **Scope selection**: this codebase is small and coherent — prefer surgical edits to one module + its tests. AGENTS.md Change Discipline: no unrelated cleanup in the same patch. If a fix wants to touch >3 of {config.py, cli.py, transcriber.py, srt.py}, re-check whether you're over-scoping.
5. **Anti-over-refactoring rule**: the compatibility shims (`_add_decoding_kwarg`, `_transcribe_with_hypotheses`, `_get_hypothesis_timestamps`, the try/except optional imports in `postprocess.py`/`llm_postprocess.py`) look redundant but are deliberate NeMo/SDK version armor. Do not "simplify" them away.
6. **Config changes ripple**: new/changed config field ⇒ `config.py` (or `llm_postprocess.py` for LLM config) + consuming module + test + README.md + README.zh-TW.md + docs/TUNING_GUIDE.md + docs/TUNING_GUIDE.zh-TW.md + CHANGELOG.md. Batch-read them in parallel, edit in one pass.
7. **Uncertainty handling**: if a claim in any doc conflicts with code, trust the code, verify by running it, and fix the doc (or propose the fix for local-only files). Mark anything you couldn't run as unverified in your report.
8. **Progress reporting**: end every session with: files changed, exact commands run + outcomes (paste the test summary line), what was NOT verified and why, remaining risks.
9. **Completion bar**: Tier 1 verification (lint + full 22-test suite) green is the minimum for ANY code change. Pipeline-behavior changes additionally need a Tier 2 real run or an explicit handoff saying it's pending.
10. **Never without asking**: push/publish; edit `.env`; change `pyproject.toml` pins/indexes; commit `CLAUDE.md`/`AGENTS.md`/`.claude/` (they're local by design); change default model (`pretrained_name`) or default VAD/segmentation values; delete user media/benchmark outputs; commit anything under `tmp_outputs/` or generated `.srt` files.
11. **Commits**: only when the user asks. Conventional-ish style observed in history (`feat:`, `fix:`, `docs:`, `chore:`). Keep code and doc changes in one logical commit or a code+docs pair, matching existing history.

---

## 10. Model Routing Recommendations

Exact future model availability cannot be verified from this repo; routing below is by capability class.

| Capability class | Route these tasks | Notes |
|---|---|---|
| Cheap/small (e.g., Haiku-class) | Doc-sync edits across the 4 parallel doc files; CHANGELOG entries; running the verification suite and reporting; grep-style audits (e.g., finding stale claims) | Give it exact file lists and acceptance greps; it should not make judgment calls about pipeline behavior |
| Mid (e.g., Sonnet-class) | Roadmap P0-1, P0-2, P2-1, P2-2; adding tests (Task B); the lint-only CI (Task D, post-approval) | The test suite is a strong safety net for these |
| High-end coding (e.g., Opus/Codex-class) | NeMo 2.8 upgrade (Task C); anything touching `transcriber.py` decoding logic, VAD chunking math, or segment merge/dedup semantics; LLM default refresh with live-doc verification (P1-1) | These require reasoning about upstream API behavior and quality trade-offs |
| Fresh-context reviewer | After any change to `srt.py` segmentation, `llm_postprocess.py` invariants, or dependency bumps: one review pass focused on (a) segment tuple invariants, (b) order/count/timestamp preservation, (c) doc-sync completeness | Reviewer should read this file's §11 first |
| Human/product | Default model changes, VAD default tuning, CI budget, LLM V3 features, diarization, multilingual defaults, whether to commit this file | See §12 |

**Escalate** (cheap→high-end) when: a change requires editing `pyproject.toml`/`uv.lock`; a NeMo API behaves differently than documented; the test suite fails for reasons unrelated to your change. **Stop retrying** after two failed attempts at the same fix — write up findings and hand off instead; this repo's failure modes (dependency conflicts, GPU-specific crashes) usually need either the runbook or the physical machine, not more attempts.

---

## 11. Implementation Guardrails

- **Never hand-edit**: `uv.lock` (regenerate via `uv lock` only, and only within an UPGRADE_NEMO.md-compliant change).
- **NeMo-sensitive (full runbook required)**: `pyproject.toml` dependency ranges, `constraint-dependencies`, `[[tool.uv.index]]` CUDA index, `tool.uv.sources`.
- **Security-sensitive**: `nemoscribe/audio.py` `validate_media_path()` and every subprocess call site; `.env` (read-never).
- **Local-only, do not commit**: `CLAUDE.md`, `AGENTS.md`, `.claude/`, `.serena/`, `run_evaluate.sh`, `HANDOFF_*.md`, `CODE_REVIEW.md` (enforced by `.git/info/exclude` — note this file does not exist in fresh clones, so cloud agents must not "helpfully" commit those files if the user syncs them another way).
- **Never commit**: `tmp_outputs/`, `*.srt`/`*.wav`/media, `evaluation_report.txt`, model caches (all gitignored — keep it that way).
- **Tests that must move with behavior**: `cli`/`cli_list` ↔ `cli.py` parsing; `llm`, `llm_cli`, `llm_validation`, `llm_parsing`, `llm_fallback`, `llm_validation_fallback` ↔ `llm_postprocess.py`; `srt`, `srt_edge` ↔ `srt.py`; `decoding`, `nemo_api` ↔ `transcriber.py`; `path` ↔ `audio.py`; `segmentation`, `merging` ↔ `vad.py`/`postprocess.py`; `ab_test` ↔ A/B helpers in `cli.py`.
- **Patterns to preserve**: `(start, end, text)` tuple shape end-to-end; graceful-degradation pattern (optional deps import-guarded, features fall back with a warning, pipeline never crashes because an optional feature is missing); version-adaptive NeMo kwargs; per-file MIT header in `nemoscribe/*.py`; backward-compatible defaults (new features ship disabled).
- **Patterns to phase out (opportunistically, not proactively)**: duplicated SRT parsing in `scripts/evaluate_benchmark.py`/`analyze_quality.py`; duplicated provider batch loops in `llm_postprocess.py`.
- **Version stamps**: `pyproject.toml version` and `nemoscribe/__init__.py __version__` must match on release commits.

---

## 12. Open Questions For The User

**Product behavior**
- Should LLM post-processing default provider/model change, and to what cost tier? (P1-1 blocker)
- Are the V3 LLM ideas (conservative mode, cross-batch names, character hints, number guard) still wanted, and in what priority?
- Is multilingual (parakeet v3) demand real, or is English-only fine indefinitely?

**Architecture**
- When NeMo 2.8 ships on PyPI, upgrade immediately or wait a patch release?
- Is speaker diarization (watchlist idea) in scope for this tool or a separate project?

**Deployment**
- Is CI wanted at all, and with what budget? Lint-only vs. full-suite-with-cache? (P1-2 blocker)
- Any intent to publish to PyPI? (Would change packaging/verification requirements.)

**Data/security**
- Confirm/commit the root `__init__.py` deletion diff (it's public on GitHub; removal changes nothing functionally but is a visible history change).
- Should this guide file be committed to the public repo, kept local via `.git/info/exclude`, or trimmed before committing?

**UX/design**
- `output_path` is silently ignored when multiple videos match (`cli.py:119`) — warn, error, or keep?

**Maintenance preferences**
- Keep the bespoke test runner, or is a pytest migration acceptable someday? (Current stance inferred from AGENTS.md: keep runner, don't assume pytest.)
- Which corrections may agents apply to local `CLAUDE.md`/`AGENTS.md` directly vs. propose-only?

---

## 13. Recommended Next Session Prompt

```
Read FUTURE_AGENT_REPO_GUIDE.md at the repo root first, then AGENTS.md if it
exists locally. Do not modify dependencies or pyproject.toml in this session.

If the 2026-07-08 P0 patch is not committed yet, review and verify it first.
Otherwise pick ONE remaining roadmap item from §5 of the guide; confirm this
scope with me in one short message before editing.

Then:
1. Implement minimally — no unrelated cleanup.
2. Verify: `uv run ruff check .` and the FULL suite
   `uv run python tests/test_improvements.py` (expect 22/22; note that
   `--test full` is a single test, not the suite). For P0-2 also run
   `uv run nemoscribe --help`.
3. Do not commit or push unless I approve the diff.
4. Report: files changed, exact commands run with their summary output, any
   failures verbatim, anything you could not verify (e.g., GPU runs), and
   remaining risks. Reply to me in Traditional Chinese (Taiwan); keep all
   code/docs/commit text in English.
```

---

## 14. Maintenance Protocol For This Report

- **Update autonomously** (in the same session as the change that invalidates it): the Repository Map (§2) after file moves/deletes; Verification tables (§7) after command changes; marking roadmap items done (append "✅ done <date>, commit <sha>" to the item — don't delete it for one release cycle); test counts/names.
- **Requires user confirmation**: changing priorities/ordering in §5; adding new P0/P1 items; anything in §12; committing this file for the first time.
- **Recording lessons learned**: add a dated bullet to a `## Lessons Learned` section (create it at the end when first needed). One line per lesson, with a `file:line` or commit reference. If it's a dependency lesson, put it in `UPGRADE_NEMO.md` instead — that's the canonical home.
- **Avoiding bloat**: this file should stay under ~500 lines of substance. When a section doubles in size, move detail out (dependency detail → `UPGRADE_NEMO.md`; test docs → a header comment in `tests/test_improvements.py`; historical evidence → `V2_TEST_REPORT.md`-style dated reports) and leave a pointer.
- **Splitting**: if CI is ever added, spin verification into `docs/VERIFICATION.md`; if the roadmap grows past ~10 open items, spin it into `docs/ROADMAP.md`.
- **Deleting guidance**: remove any claim the moment you verify it's false (don't strike-through — delete and note it in Lessons Learned). Completed roadmap items may be deleted one release after completion.

---

## 15. Final Notes To Future Agents

**Three non-obvious pieces of advice:**

1. **The test suite passing tells you the plumbing works, not that subtitles are good.** Every quality decision in this repo (VAD thresholds, `segment_separators`, merge strategies, the 30%/60% LLM similarity thresholds) was validated by benchmarking real TV episodes against reference SRTs on the maintainer's GPU. If your change could alter *what text appears when*, a green suite is necessary but not sufficient — say so in your report and request a Tier 2 run.

2. **This project's history is mostly a record of things that *didn't* work.** GPU-PB context biasing, NGPU-LM fusion, parakeet-unified, v3-as-English-default — all tried, measured, and rejected with data. The single highest-waste failure mode for a future agent is proposing one of these again because it sounds promising. When you have an optimization idea, first search this file, `CHANGELOG.md`, and (if local) `CLAUDE.md`'s "Failed Optimizations" section.

3. **The graceful-degradation pattern is the product.** Users run this on messy machines: no ITN install, no API key, VAD model download failures, 8GB GPUs. Every optional feature failing must produce a warning and a still-usable SRT, never a crash. When adding features, wire the failure path first; when fixing bugs, check you haven't turned a degrade-and-warn into a raise.

**How this guidance will most likely decay**: a NeMo/torch upgrade lands and changes commands, pins, or API behavior described here — and the upgrading agent updates `pyproject.toml`/`UPGRADE_NEMO.md` but not this file. Second most likely: the maintainer edits local `CLAUDE.md` conventions that fresh-clone agents (who only see this file) never learn.

**How to prevent it**: treat §2, §7, and §11 as part of the definition-of-done for any structural or dependency change (§14 lists what to update autonomously). If you're the upgrading agent, budget five minutes for this file.

**When repo reality conflicts with this report**: the code and the actual command output win, always. Verify the discrepancy by running the relevant command, fix or delete the stale claim here (per §14), and mention the correction in your session report. This document was accurate on 2026-07-07 for commit `1bdac73`; it is a map, not the territory.
