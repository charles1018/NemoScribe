# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
