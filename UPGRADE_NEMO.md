# NeMo Upgrade Runbook

This document defines the required process for upgrading NemoScribe's NeMo-based stack.

Use it whenever updating any of the following:

- `nemo-toolkit`
- `torch`
- CUDA wheel index / CUDA version
- `transformers`, `huggingface_hub`, `onnx`, or other NeMo-sensitive constraints
- `uv.lock`

The goal is to make future Codex CLI / Claude Code upgrades deterministic and safe.

## Current Integration Surface

NemoScribe depends on these NeMo APIs directly:

- `nemo.collections.asr.models.ASRModel`
- `nemo.collections.asr.models.EncDecClassificationModel`
- `nemo.collections.asr.parts.submodules.rnnt_decoding.RNNTDecodingConfig`
- `nemo.collections.asr.parts.submodules.ctc_decoding.CTCDecodingConfig`
- `nemo.collections.asr.parts.utils.rnnt_utils.Hypothesis`
- `nemo.collections.asr.parts.utils.vad_utils.init_frame_vad_model`
- `nemo.collections.asr.parts.utils.vad_utils.generate_vad_segment_table_per_tensor`
- `nemo.collections.asr.parts.utils.transcribe_utils.get_inference_device`
- `nemo.collections.asr.parts.utils.transcribe_utils.get_inference_dtype`
- `nemo.collections.asr.metrics.wer.word_error_rate_detail`
- `nemo.utils.logging`

These call sites must be re-checked on every NeMo upgrade:

- `nemoscribe/transcriber.py`
- `nemoscribe/vad.py`
- `nemoscribe/cli.py`
- `nemoscribe/srt.py`
- `scripts/analyze_quality.py`
- `tests/test_improvements.py`

## Upgrade Modes

Choose one mode before making changes.

### 1. PyPI release tracking

Use this when the target NeMo version is officially published.

Expected change:

- Update `nemo_toolkit[asr]` version range in `pyproject.toml`
- Re-resolve `uv.lock`

This is the default and preferred mode.

### 2. Upstream repo tracking

Use this only when NemoScribe must consume a newer NeMo commit before PyPI release.

Expected change:

- Replace the PyPI spec with a git or local path source in `pyproject.toml`
- Record the exact upstream commit SHA in the PR / commit message
- Re-resolve `uv.lock`

Do not describe this as a normal package bump. It is a source-tracking change.

## High-Risk Areas

### Packaging and dependency model

Recent NeMo upstream changes moved dependency ownership into `pyproject.toml` and `uv.lock`.
Legacy `requirements/*.txt` files were removed upstream.

Implications for NemoScribe:

- Do not assume upstream `requirements/*.txt` still exist
- Prefer reading upstream `pyproject.toml` for the current dependency matrix
- Re-check local constraints after every NeMo bump

### Torch and CUDA matrix

NemoScribe currently pins its own Torch/CUDA strategy in `pyproject.toml`.
This must be reviewed together with every NeMo version update.

Check all of the following:

- `torch` version range
- CUDA wheel index URL under `[[tool.uv.index]]`
- `tool.uv.sources` entries for `torch`, `torchvision`, `torchaudio`
- README installation examples that mention CUDA versions
- Real-device stability for `compute_dtype` and decoding settings

Do not bump NeMo alone if the upstream NeMo line now expects a different Torch/CUDA combination.

### Transitive dependency churn

Recent upstream NeMo changes removed some older dependencies and reworked extras.
Lockfile changes may be large and valid.

Pay special attention to:

- `transformers`
- `huggingface_hub`
- `lightning`
- `lhotse`
- `onnx`
- `numpy`
- `pyannote-*`

If `pyannote-*` disappears after a NeMo bump, that may be expected rather than a regression.

### ASR decoding behavior

Recent upstream work heavily changed transducer decoding internals.
NemoScribe uses local compatibility logic for decoding config fields, but behavior still needs validation.

Re-test:

- RNNT timestamps
- CTC timestamps
- `segment_separators`
- `segment_gap_threshold`
- `rnnt_fused_batch_size`
- chunked transcription output consistency

## Required File Review

Before making any version changes, review:

- `~/dev/tools/claude/NeMo/pyproject.toml`
- `~/dev/tools/claude/NeMo/CLAUDE.md`
- `pyproject.toml`
- `uv.lock`
- `README.md`
- `README.zh-TW.md`
- `CLAUDE.md`

## Required Edit Checklist

When upgrading, update these local files as needed:

- `pyproject.toml`
- `uv.lock`
- `README.md`
- `README.zh-TW.md`
- `CLAUDE.md`
- `CHANGELOG.md` if user-visible install or compatibility expectations changed

At minimum, verify that docs do not mention stale NeMo/Torch/CUDA assumptions.

## Required Validation Steps

Do not stop after dependency resolution. Run validation.

### 1. Dependency resolution

Run:

```bash
uv lock
uv sync --python 3.12
```

If using a non-default extra for validation, also run:

```bash
uv sync --python 3.12 --extra itn --extra benchmark --extra llm
```

### 2. Import-level sanity checks

Run:

```bash
uv run python - <<'PY'
from nemo.collections.asr.models import ASRModel, EncDecClassificationModel
from nemo.collections.asr.parts.submodules.rnnt_decoding import RNNTDecodingConfig
from nemo.collections.asr.parts.submodules.ctc_decoding import CTCDecodingConfig
from nemo.collections.asr.parts.utils.rnnt_utils import Hypothesis
from nemo.collections.asr.parts.utils.transcribe_utils import get_inference_device, get_inference_dtype
from nemo.collections.asr.parts.utils.vad_utils import init_frame_vad_model, generate_vad_segment_table_per_tensor
from nemo.collections.asr.metrics.wer import word_error_rate_detail
print("NeMo import surface OK")
PY
```

### 3. Automated regression tests

Run at least these focused checks:

```bash
uv run python tests/test_improvements.py --test baseline
uv run python tests/test_improvements.py --test vad
uv run python tests/test_improvements.py --test decoding
uv run python tests/test_improvements.py --test segmentation
uv run python tests/test_improvements.py --test srt
uv run python tests/test_improvements.py --test cli
```

Run the full regression suite when the bump is substantial. This currently runs all 22 tests; do not use `--test full` for this because that is only the single full-config test.

```bash
uv run python tests/test_improvements.py
```

### 4. Real transcription smoke test

Run one real sample on the target machine:

```bash
uv run nemoscribe video_path="sample.mp4"
```

Also test the higher-risk path:

```bash
uv run nemoscribe video_path="sample.mp4" \
  compute_dtype=float32 \
  vad.enabled=true \
  decoding.rnnt_fused_batch_size=0
```

Check:

- command completes successfully
- subtitle file is generated
- timestamps are monotonic
- segment splitting still looks sensible
- no obvious hallucination increase
- performance did not collapse unexpectedly

### 5. Optional quality script

If benchmark references are available, run:

```bash
uv run python scripts/analyze_quality.py
```

## Decision Rules

Use these rules during upgrades.

### Acceptable changes

- large `uv.lock` diff caused by upstream dependency cleanup
- removal of transitive `pyannote-*` packages
- minor version bumps in NeMo-owned dependencies
- README updates required by new CUDA or install guidance

### Escalate before merging

- import failures on the documented NeMo API surface
- changed transcription output shape or missing timestamps
- decoding config warnings that disable key functionality
- VAD output collapsing to empty or near-empty segments
- Torch/CUDA mismatch requiring undocumented local hacks
- `transformers` / `huggingface_hub` resolver conflict

## Notes for Agent-Driven Upgrades

When Codex CLI or Claude Code performs this upgrade, it should:

1. Read this file first.
2. Inspect upstream NeMo `pyproject.toml` and `CLAUDE.md`.
3. Decide whether the upgrade is PyPI-tracking or repo-tracking.
4. Update `pyproject.toml` conservatively.
5. Rebuild `uv.lock`.
6. Run the required validation steps.
7. Update README / CLAUDE docs so they match the new environment contract.

Do not submit an upgrade that only changes version strings without re-locking and validating behavior.

## Reference Notes

### huggingface-hub 1.0 Breaking Changes

Key changes that affect this project:
- HTTP backend: `requests` → `httpx`
- CLI: `huggingface-cli` → `hf`
- Removed: TensorFlow support, `hf_transfer`
- Error handling: `requests.HTTPError` → `httpx.HTTPError`

**Wait for**: transformers 5.0 release (will support huggingface-hub 1.0+)

**References**:
- [huggingface-hub v1.0 Migration Guide](https://huggingface.co/docs/huggingface_hub/v1.1.0/concepts/migration)
- [huggingface-hub v1.0 Release Notes](https://github.com/huggingface/huggingface_hub/releases/tag/v1.0.0)

### NeMo 2.8.0 Compatibility Analysis (2026-04-04)

NeMo 2.8.0 is not yet released on PyPI (source available at `~/dev/tools/claude/NeMo`).
Our API surface is **fully compatible** with 2.8.0 — no breaking changes to imports we use.

**Our NeMo API surface (all stable in 2.8.0):**
- `nemo.collections.asr.models.ASRModel` — model loading, `from_pretrained()`, `restore_from()`, `transcribe()`
- `nemo.collections.asr.models.EncDecClassificationModel` — VAD model (now inherits from `EncDecSpeakerLabelModel`)
- `nemo.collections.asr.parts.submodules.rnnt_decoding.RNNTDecodingConfig` — decoding config
- `nemo.collections.asr.parts.submodules.ctc_decoding.CTCDecodingConfig` — CTC decoding config
- `nemo.collections.asr.parts.utils.rnnt_utils.Hypothesis` — transcription result type
- `nemo.collections.asr.parts.utils.vad_utils.{init_frame_vad_model, generate_vad_segment_table_per_tensor}`
- `nemo.collections.asr.parts.utils.transcribe_utils.{get_inference_device, get_inference_dtype}`

**Key NeMo 2.8.0 changes (NOT affecting us):**
- Removed: k2 aligner, quartnet, msdd, slu models — we don't use these
- Added: ASR-EOU models — for streaming, not relevant for offline transcription
- Removed: torchaudio dependency — we use librosa directly
- Removed: deprecated NLP/TTS/other collections — ASR collection unchanged

**Beneficial fix in 2.8.0 (available after upgrade):**
- Fix forced decoder reinstantiation with `timestamps=True` (#15298) — performance improvement for chunked transcription

**When 2.8.0 is released:** Update `pyproject.toml` constraint from `<2.8` to `<2.9`
