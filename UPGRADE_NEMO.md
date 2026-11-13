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

- the target release's upstream `pyproject.toml` and `CLAUDE.md` (use a local
  `~/dev/tools/claude/NeMo` or `Speech` checkout when present; otherwise clone
  the exact official release tag to a temporary directory)
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

Run the full regression suite when the bump is substantial. The no-flag invocation runs every test; do not use `--test full` for this because that is only the single full-config test.

```bash
uv run python tests/test_improvements.py
```

### 4. Real transcription smoke test

Run one real sample on the target machine. The sample must be long enough to
be split into several chunks (longer than `audio.max_chunk_duration`, 300 s by
default; a full TV episode is ideal). A clip of a few seconds does not exercise
chunked transcription and has missed a hard crash before (see the 2026-11-13
reference note).

```bash
uv run nemoscribe video_path="episode.mkv"
```

Also test the higher-risk paths:

```bash
uv run nemoscribe video_path="episode.mkv" \
  compute_dtype=float32 \
  vad.enabled=true

# NeMo's CUDA graph decoder (off by default in NemoScribe). Re-test it on
# every upgrade; if it completes on a full episode, the default can be revisited.
uv run nemoscribe video_path="episode.mkv" \
  decoding.use_cuda_graph_decoder=true
```

A CUDA failure aborts the whole process (exit code 134) instead of raising a
Python exception, and systemd-coredump may then write a multi-GB core file to
the root volume. Check free disk space before running the CUDA graph test.

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

### NeMo Speech 3.0 Compatibility Analysis (2026-08-12)

NeMo Speech 3.0.0 is an official stable PyPI release. NemoScribe therefore uses
**PyPI release tracking**, not a Git source. The release tag inspected during
the upgrade was `v3.0.0` at commit
`fd6a877539710e2b98f28c43272ff81312f83417`.

The repository split moved the active speech collections to
`NVIDIA-NeMo/Speech`. NeMo 3.0 retains every API in NemoScribe's integration
surface, including RNNT/CTC decoding dataclasses, `Hypothesis`, ASR/VAD model
loading, VAD utilities, inference device/dtype helpers, and WER metrics.

Validated local matrix:

- Python 3.12
- `nemo-toolkit[asr]` 3.0.0
- PyTorch 2.13.0 + CUDA 13.2
- Transformers 5.15.0 + huggingface-hub 1.27.0
- NumPy 2.5.2
- ONNX 1.22.0
- RTX 3070 Laptop GPU, driver 595.84

The upstream actively-tested baseline uses PyTorch 2.12 with CUDA 12.6/13.2,
but its install contract supports user-selected PyTorch 2.7+. NemoScribe's
newer PyTorch 2.13 line was accepted only after import checks, the full 22-test
suite, and baseline plus VAD real-transcription smoke tests passed on CUDA.

Hydra Core 1.3.2 and OmegaConf 2.3.0 remain exact compatibility pins because
NeMo Speech 3.0 caps those versions upstream. The Transformers 5 migration
allows huggingface-hub 1.x; NumPy 2 and the current ONNX/ml-dtypes line are now
validated instead of carrying the NeMo 2.x compatibility pins.

Known upstream security constraint as of 2026-08-12: NeMo Speech 3.0 requires
`lightning<=2.4.0`, and released Lightning versions through 2.6.5 are affected
by [CVE-2026-58659](https://github.com/advisories/GHSA-qqmf-gpg7-g8gw) when
`LightningModule.load_from_checkpoint()` loads an attacker-controlled
`_instantiator`. Lightning 2.6.6 (2026-09-10) contains the fix, but NeMo 3.0.0
cannot use it (see the 2026-11-13 note below). ASR `.nemo` models use `ASRModel.restore_from()`, and
NemoScribe explicitly rejects `.ckpt` VAD paths before NeMo's VAD helper can
reach the affected Lightning API. Continue treating all accepted model archives
as trusted input, and revisit the Lightning constraint when NeMo permits a
stable patched release.

Expected lockfile cleanup includes removal of legacy non-speech, deployment,
Optuna, and `pyannote-*` packages. Do not re-add them unless a direct feature
requires them.

### Upstream Status Check and CUDA Graph Decoder Finding (2026-11-13)

No dependency was changed in this check.

Upstream status:

- `nemo-toolkit` 3.0.0 (2026-08-07, tag `v3.0.0`, commit `fd6a8775`) is still the
  latest release on both PyPI and GitHub. There is no 3.0.x patch release.
- Upstream `main` (`079e4877`, 2026-11-12) is versioned 3.1.0 and is 283 commits
  ahead of the tag. It is not a release; do not track it.
- `main` requires `lightning>=2.6.6,<2.7`, `pytorch-lightning>=2.6.6,<2.7`,
  `hydra-core>=1.3.8,<1.4`, `omegaconf>=2.3,<2.4`, and pins `lhotse==2.0.0a6`
  (upstream #16504). The exact `hydra-core==1.3.2` pin here must be relaxed when
  that release ships.
- Lightning 2.6.6 fixes CVE-2026-58659, but NeMo 3.0.0 caps `lightning<=2.4.0`
  and upstream reported NeMo as broken on Lightning 2.6 before #16504, so the
  patched Lightning cannot be forced onto 3.0.0. NemoScribe's `.ckpt` rejection
  remains the mitigation until the next NeMo release.
- Unreleased fixes on `main` that touch NemoScribe's paths:
  - #16098: batched TDT greedy decoding stored a stale token duration in the
    PyTorch implementation. It only occurs with more than one utterance per
    batch; NemoScribe transcribes with `batch_size=1`, so it is not affected.
  - #16096: `GreedyTDTInfer` skipped a frame when the symbol budget ran out.
    NemoScribe uses the batched `greedy_batch` strategy, so it is not affected.
  - #15714 (issue #16576): `change_attention_model()` adds randomly initialized
    attention biases to models trained with `use_bias: false`. This includes the
    default `parakeet-tdt-0.6b-v2`. NemoScribe only calls it when
    `audio.long_audio_threshold` is lowered from its disabled default; leave it
    disabled on NeMo 3.0.0.

CUDA graph decoder:

- On an RTX 3070 Laptop GPU (driver 620.70) with PyTorch 2.13.0+cu132, NeMo's
  CUDA graph implementation of TDT greedy decoding (`full_graph` mode) aborted
  with `CUDA error: an illegal memory access` in 7 of 7 runs of a 42-minute
  episode, at the second or third chunk, with bfloat16 and with float32.
- With `greedy.use_cuda_graph_decoder=false` the same episode completed in all
  8 transcriptions (about 160x realtime, 2.9-3.5 GiB peak allocated), with
  reproducible output. On an 11-second clip both decoders produce byte-identical subtitles.
- `fused_batch_size` only sets the joint's fused loss/WER step. Values `-1` and
  `0` are equivalent in NeMo 3.0 and neither disables CUDA graphs, so the old
  `rnnt_fused_batch_size=0` recommendation was not a real workaround.
- The 2026-08-12 smoke test used an 11-second clip, which is a single chunk and
  does not trigger the crash. Use a multi-chunk video for upgrade validation.
