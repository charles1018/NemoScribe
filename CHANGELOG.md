# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
- All 10 unit tests pass

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
