# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- Upgraded the runtime stack to NeMo Speech 3.0, PyTorch 2.13 with CUDA 13.2, Transformers 5.15, NumPy 2.5, and ONNX 1.22; the minimum supported Python version is now 3.12
- Upgraded optional ITN, benchmark, and LLM dependencies to their latest stable lines, including `nemo-text-processing` 1.2, jiwer 4, OpenAI 3, Anthropic 0.121, and json-repair 0.63
- Changed the default LLM post-processing provider and model to OpenAI `gpt-5.4-mini` (previously Anthropic `claude-3-5-sonnet-20241022`); the OpenAI path was verified with live API calls. To keep using Anthropic, set both `llm_postprocess.provider=anthropic` and `llm_postprocess.model` (for example `claude-sonnet-5`)
- Reused the shared SRT timestamp parser in benchmark/quality scripts; the parser now accepts both comma and dot millisecond separators
- Deduplicated the Anthropic/OpenAI LLM subtitle batch loop while preserving provider-specific client setup and fallback behavior

### Added

- Added a GitHub Actions workflow that checks `uv.lock` and runs `ruff check .` via `uvx`; refreshed the action and Ruff versions

### Fixed

- Chunked transcription no longer deletes real dialogue: merging and deduplication of chunk-overlap duplicates is now restricted to segments that straddle a chunk boundary. Previously, adjacent sentences less than 0.1 s apart and legitimately repeated lines anywhere in the video could be dropped (129 words in a 42-minute test episode; WER against the official subtitles improved from 16.00% to 13.78%)
- Full-length videos no longer abort with `CUDA error: an illegal memory access`: NeMo's CUDA graph decoder is now disabled by default through the new `decoding.use_cuda_graph_decoder` option (default `false`). On an RTX 3070 Laptop GPU with NeMo Speech 3.0 and PyTorch 2.13 it crashed in every run of a 42-minute episode, while the PyTorch decoder completes at about 160x realtime with identical output. Set `decoding.use_cuda_graph_decoder=true` to opt back in
- Corrected the `decoding.rnnt_fused_batch_size` documentation and log message: it sets NeMo's joint fused batch size and never controlled CUDA graphs, so `rnnt_fused_batch_size=0` was not a working stability workaround
- LLM post-processing with Anthropic models now reads the text blocks of a response instead of assuming the first block is text; a leading thinking block previously failed every attempt and silently fell back to the uncorrected subtitles
- LLM post-processing with OpenAI now sends `max_completion_tokens` instead of the deprecated `max_tokens`, which reasoning models reject, and treats an empty reply as a failed attempt

### Security

- Masked `llm_postprocess.api_key` in the configuration printed at startup; an API key passed on the command line was previously written to the log in clear text
- Rejected local Lightning `.ckpt` VAD models before NeMo can route them through the vulnerable checkpoint loader described by CVE-2026-58659; use a NeMo registry model or trusted local `.nemo` archive

### Documentation

- Documented a known model limitation of `parakeet-tdt-0.6b-v2`: repeated words and false starts (disfluencies) may be dropped; use `parakeet-tdt-1.1b` when verbatim disfluencies matter
- Mentioned the free hosted API on build.nvidia.com as a no-GPU way to try the default model
- GPU memory table: 8GB GPUs with `compute_dtype=float32` may OOM at the default `max_chunk_duration=300` on dialogue-dense content; recommend dropping to 120 (verified on Yellowstone S03E01, RTX 3070 8GB)
- Updated installation and upgrade guidance for the NVIDIA-NeMo/Speech repository split, Python 3.12+, and the CUDA 13.2 package index

## [0.5.0] - 2026-06-11

### Added

- **VAD A/B comparison mode**: `ab_test.vad=true` generates both VAD and no-VAD outputs (`video.vad.srt` / `video.no_vad.srt`) in a single run, sharing the same ASR settings
- `decoding.segment_gap_threshold`: timing-based segment splitting that complements punctuation-based `segment_separators` (exposes NeMo's `RNNTDecodingConfig`/`CTCDecodingConfig` field)

### Changed

- Upgraded NeMo toolkit to `>=2.7.3,<2.8` and pinned core runtime dependencies (`torch>=2.11,<2.12`, tightened optional dependency ranges for `itn`, `benchmark`, and `llm` extras)
- Decoding options are now applied version-adaptively: unsupported NeMo decoding config fields are skipped with a debug log, and both `segment_separators` and the legacy `segment_seperators` spellings are accepted upstream
- Upgraded optional LLM dependencies in the lockfile (anthropic, openai, json-repair)

### Fixed

- LLM post-processing now preserves subtitle segment order in its output
- `decoding.segment_gap_threshold` now validates as a positive integer instead of silently accepting `0` or negative values
- Subtitle segmentation now preserves punctuation-based `segment_separators` when `segment_gap_threshold` is also enabled

### Documentation

- Added `UPGRADE_NEMO.md`: runbook for upgrading the NeMo dependency stack (version policy, lockfile re-resolution, validation steps)
- Clarified VAD tuning guidance in `TUNING_GUIDE.md`, including when to prefer no-VAD output on clean drama/movie audio
- Clarified that `segment_gap_threshold` is frame-based, must be positive, and can be combined with punctuation splitting
- Documented the Chicago Fire drama validation profile: `compute_dtype=float32`, `decoding.rnnt_fused_batch_size=0`, and `decoding.segment_gap_threshold=20`
- Synced test inventory and coverage notes in README

## [0.4.0] - 2026-03-26

### Added

- **LLM Post-processing**: Fix transcription errors (character names, proper nouns, homophones) using OpenAI or Anthropic LLMs
  - Agent Loop pattern: LLM → Validate → Feedback → Retry for reliable parsing
  - JSON structured output with `json-repair` for robust response parsing
  - Similarity validation to prevent excessive changes (adaptive thresholds)
  - Supports OpenAI (GPT-4o, GPT-4o-mini) and Anthropic (Claude 3.5 Sonnet, Claude 3 Opus)
  - Automatic `.env` file loading for API keys via `python-dotenv`
  - Graceful fallback to original text on any error
- New optional dependency group: `uv sync --extra llm` (anthropic, openai, python-dotenv, json-repair)
- LLM unit tests: config, CLI override, validation, JSON parsing, fallback (no API key required)

### Changed

- Version bump to 0.4.0

### Known Limitations

- LLM may over-correct ~10% of segments (mostly minor/cosmetic changes)
- Cost per episode: ~$0.06 (GPT-4o-mini) to ~$0.24 (Claude 3.5 Sonnet)
- Semantic errors (e.g., wrong word choice) remain challenging for LLM to fix

## [0.3.0] - 2025-12-28

### Added

- `segment_separators` configuration for punctuation-based segment splitting (verified: reduces max segment duration by 76%)
- CLI parser now warns about unknown config keys to catch typos (e.g., `vad.onst` instead of `vad.onset`)
- Enum type support in CLI type coercion with case-insensitive parsing
- Comprehensive docstrings for CLI helper functions (`_is_optional_type`, `_unwrap_optional`, `_coerce_value`, `_set_typed_attr`)
- Shared `parse_srt_timestamp()` utility in `nemoscribe.srt` module
- Analysis scripts for SRT quality (`analyze_srt_stats.py`, `find_longest_segment.py`)

### Changed

- ITNNormalizer type now imports actual type when `nemo_text_processing` is available, falls back to `Any` otherwise
- Improved type checking with conditional imports for optional dependencies

### Security

- Path validation for all ffmpeg/ffprobe subprocess calls prevents injection attacks
- Path normalization and validation to prevent path traversal attacks

### Fixed

- Chunk extraction errors now raise exceptions instead of silently failing
- Duration detection failures provide clear error messages
- CLI parser handles invalid nested config keys properly
- TemporaryDirectory context manager ensures cleanup even on exceptions

### Documentation

- Updated CLAUDE.md with CLI parser features, security improvements, and analysis scripts
- Expanded test coverage documentation (srt_edge_cases, path_validation, cli_config_override)

## [0.2.1] - 2025-12-22

### Added

- Parameter benchmark tools for testing and optimizing VAD settings (`scripts/evaluate_benchmark.py`)

### Fixed

- Error handling for temp directory cleanup on Windows
- RTFx division by zero when audio duration is very short
- Batch processing now continues after single file errors

### Changed

- Add type hints to VAD module functions
- Simplified gitignore configuration

## [0.2.0] - 2025-12-14

### Changed

- **Project Structure**: Refactored from single-file `nemoscribe.py` (2100+ lines) to modular package structure
  - `nemoscribe/config.py` - Configuration dataclasses
  - `nemoscribe/audio.py` - Audio processing (ffmpeg)
  - `nemoscribe/vad.py` - Voice Activity Detection
  - `nemoscribe/transcriber.py` - ASR model and transcription
  - `nemoscribe/srt.py` - SRT formatting and output
  - `nemoscribe/postprocess.py` - ITN and segment merging
  - `nemoscribe/log_utils.py` - Log filtering
  - `nemoscribe/cli.py` - CLI entry point
- **Entry Point**: Changed from `nemoscribe:main` to `nemoscribe.cli:main`

### Added

- `nemoscribe/__init__.py` - Package definition with public API exports
- `nemoscribe/__main__.py` - Support for `python -m nemoscribe`
- Project structure section in README files

### Fixed

- Unicode character encoding issue in test output for Windows terminals

### Notes

- 100% CLI backward compatible - all existing commands work unchanged
- Integration test coverage added for the initial modular release

## [0.1.0] - 2025-12-13

### Added

- Initial release
- Video to SRT subtitle conversion using NVIDIA NeMo ASR models
- Support for Parakeet-TDT models (0.6b-v2, 0.6b-v3, 1.1b)
- Voice Activity Detection (VAD) for filtering non-speech content
- Smart segmentation at silence boundaries
- Inverse Text Normalization (ITN) support
- CUDA graphs optimization for faster inference
- Batch processing for directories
- Long audio support (up to 3 hours) with chunked inference
- Configurable subtitle formatting (line length, duration, word gaps)
- Performance measurement (RTFx calculation)
- Quality analysis tool (`scripts/analyze_quality.py`)
