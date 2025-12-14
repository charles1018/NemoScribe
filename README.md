# NemoScribe

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![CUDA](https://img.shields.io/badge/CUDA-13.0-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![GitHub stars](https://img.shields.io/github/stars/charles1018/NemoScribe?style=social)](https://github.com/charles1018/NemoScribe)

**English** | [繁體中文](README.zh-TW.md)

Convert video files to SRT subtitles using NVIDIA NeMo ASR models with accurate word-level timestamps. Supports up to 3 hours of audio through chunked inference.

Built on [NVIDIA NeMo](https://github.com/NVIDIA/NeMo) framework with [Parakeet-TDT](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2) as the default model.

## Table of Contents

- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage Examples](#usage-examples)
- [Configuration Reference](#configuration-reference)
- [Recommended Models](#recommended-models)
- [Long Audio Support](#long-audio-support)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)
- [References](#references)

## Features

- **Accurate Timestamps**: Word-level and segment-level timestamps from NeMo ASR models
- **Long Audio Support**: Process videos up to 3 hours with automatic chunking
- **Voice Activity Detection (VAD)**: Filter non-speech content to reduce hallucinations
- **Smart Segmentation**: Split audio at silence boundaries, not mid-speech
- **Inverse Text Normalization (ITN)**: Convert spoken forms to written forms ("twenty five" → "25")
- **CUDA Optimized**: CUDA graphs enabled by default for faster inference
- **Batch Processing**: Process entire directories of videos

## Requirements

| Requirement | Details |
|-------------|---------|
| **OS** | Windows 10/11, Linux |
| **Python** | 3.10+ (3.12 recommended, avoid 3.13) |
| **Package Manager** | [uv](https://docs.astral.sh/uv/) (recommended) |
| **CUDA Toolkit** | Default cu130 (13.0). PyTorch also supports 12.6/12.8. |
| **FFmpeg** | Required for audio extraction |
| **Hardware** | NVIDIA GPU with CUDA (recommended) |

### FFmpeg Installation

- **Windows**: Download from [gyan.dev](https://www.gyan.dev/ffmpeg/builds/), extract, add `bin` folder to PATH
- **Linux**: `sudo apt install ffmpeg`

## Installation

### 1. Install uv

```powershell
# Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

```bash
# Linux/macOS
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Clone the Repository

```bash
git clone https://github.com/charles1018/NemoScribe.git
cd NemoScribe
```

### 3. Install Dependencies

```bash
uv sync --python 3.12
```

### 4. Configure CUDA (Strongly Recommended)

By default, `uv sync` may install CPU-only PyTorch. **GPU acceleration is strongly recommended** for reasonable transcription speed. The project is pre-configured to use CUDA 13.0, so GPU users only need to run `uv sync`.

> **Note**: PyTorch officially supports CUDA 12.6, 12.8, and 13.0. See [PyTorch Get Started](https://pytorch.org/get-started/locally/) for details.

If you need a different CUDA version, modify `pyproject.toml`:

**CUDA 13.0 (Default, Recommended):**
```toml
[[tool.uv.index]]
name = "pytorch"
url = "https://download.pytorch.org/whl/cu130"
explicit = true

[tool.uv.sources]
torch = { index = "pytorch" }
torchvision = { index = "pytorch" }
torchaudio = { index = "pytorch" }
```

**CUDA 12.8:**
```toml
[[tool.uv.index]]
name = "pytorch"
url = "https://download.pytorch.org/whl/cu128"
explicit = true
```

**CUDA 12.6:**
```toml
[[tool.uv.index]]
name = "pytorch"
url = "https://download.pytorch.org/whl/cu126"
explicit = true
```

Then re-sync:
```bash
uv sync
```

### 5. Verify Setup

```bash
uv run python scripts/check_cuda.py
# Expected output: CUDA available: True
```

## Quick Start

```bash
# Basic usage
uv run nemoscribe video_path="video.mp4"

# With VAD (recommended for better quality)
uv run nemoscribe video_path="video.mp4" vad.enabled=true

# Batch processing
uv run nemoscribe video_dir=/path/to/videos/ output_dir=/path/to/subtitles/
```

> **📖 Advanced Tuning:** For optimal parameter configurations for different scenarios (drama, news, technical tutorials), see [TUNING_GUIDE.md](docs/TUNING_GUIDE.md).

## Usage Examples

### Subtitle Formatting

```bash
uv run nemoscribe video_path=video.mp4 \
  subtitle.max_chars_per_line=32 \
  subtitle.max_segment_duration=3.0 \
  subtitle.word_gap_threshold=0.5

# Disable word gap splitting
uv run nemoscribe video_path=video.mp4 subtitle.word_gap_threshold=null
```

### Device and Precision

```bash
# Force CPU
uv run nemoscribe video_path=video.mp4 cuda=-1

# Specific GPU
uv run nemoscribe video_path=video.mp4 cuda=0

# Force float32 precision
uv run nemoscribe video_path=video.mp4 compute_dtype=float32
```

### VAD Configuration

```bash
# Enable VAD with smart segmentation
uv run nemoscribe video_path=video.mp4 \
  vad.enabled=true \
  audio.smart_segmentation=true

# Adjust VAD sensitivity (optimized for drama/movie)
uv run nemoscribe video_path=video.mp4 \
  vad.enabled=true \
  vad.onset=0.2 \
  vad.offset=0.1
```

### ITN (Inverse Text Normalization)

```bash
# Enable ITN (requires nemo_text_processing)
uv run nemoscribe video_path=video.mp4 postprocessing.enable_itn=true

# For models with auto-capitalization
uv run nemoscribe video_path=video.mp4 \
  postprocessing.enable_itn=true \
  postprocessing.itn_input_case=cased

# Install ITN dependency
uv add nemo_text_processing
```

**ITN Examples:**
- `"twenty five dollars"` → `"$25"`
- `"january first twenty twenty five"` → `"January 1, 2025"`
- `"three point one four"` → `"3.14"`
- `"the meeting is at ten thirty am"` → `"the meeting is at 10:30 a.m."`

### Performance Measurement

```bash
uv run nemoscribe video_path=video.mp4 performance.calculate_rtfx=true
# Example output: RTFx=15.2x realtime (transcribed 600s in 39.5s)
```

## Configuration Reference

### Main Options

| Option | Default | Description |
|--------|---------|-------------|
| `video_path` | - | Path to input video file |
| `video_dir` | - | Path to directory containing videos |
| `output_path` | auto | Output SRT file path |
| `output_dir` | auto | Output directory for batch processing |
| `pretrained_name` | `nvidia/parakeet-tdt-0.6b-v2` | Pretrained ASR model |
| `model_path` | - | Path to local .nemo checkpoint |
| `cuda` | auto | CUDA device ID (None=auto, negative=CPU) |
| `compute_dtype` | auto | `float32`, `bfloat16`, or `float16` |
| `overwrite` | true | Overwrite existing SRT files |

### Subtitle Formatting (`subtitle.*`)

| Option | Default | Description |
|--------|---------|-------------|
| `max_chars_per_line` | 42 | Maximum characters per subtitle line |
| `max_segment_duration` | 5.0 | Maximum seconds per subtitle segment |
| `word_gap_threshold` | 0.8 | New segment if word gap >= this (seconds) |

### Audio Processing (`audio.*`)

| Option | Default | Description |
|--------|---------|-------------|
| `sample_rate` | 16000 | Audio sample rate for ASR |
| `max_chunk_duration` | 300.0 | Max chunk size (5 min, safe for 8GB GPU) |
| `chunk_overlap` | 2.0 | Overlap between chunks (seconds) |
| `smart_segmentation` | true | Use VAD-based optimal split points |
| `min_silence_for_split` | 0.3 | Minimum silence duration for split point |
| `prefer_longer_silence` | true | Prefer splitting at longer silences |

### VAD Configuration (`vad.*`)

| Option | Default | Description |
|--------|---------|-------------|
| `enabled` | false | Enable Voice Activity Detection |
| `model` | `vad_multilingual_frame_marblenet` | VAD model name |
| `onset` | 0.3 | Speech detection onset threshold (0-1) |
| `offset` | 0.3 | Speech detection offset threshold (0-1) |
| `pad_onset` | 0.2 | Padding before speech segments (seconds) |
| `pad_offset` | 0.2 | Padding after speech segments (seconds) |
| `min_duration_on` | 0.2 | Minimum speech segment duration |
| `min_duration_off` | 0.2 | Minimum non-speech gap to merge |

### Decoding Optimization (`decoding.*`)

| Option | Default | Description |
|--------|---------|-------------|
| `rnnt_fused_batch_size` | -1 | CUDA graphs: -1=enabled, 0=disabled |
| `rnnt_timestamp_type` | "all" | Timestamp type: "char", "word", "segment", "all" |
| `ctc_timestamp_type` | "all" | CTC timestamp type |

### Post-processing (`postprocessing.*`)

| Option | Default | Description |
|--------|---------|-------------|
| `enable_itn` | false | Enable Inverse Text Normalization |
| `itn_lang` | "en" | Language for ITN |
| `itn_input_case` | "lower_cased" | Input case: "lower_cased" or "cased" |

### Performance (`performance.*`)

| Option | Default | Description |
|--------|---------|-------------|
| `calculate_rtfx` | false | Calculate Real-Time Factor (RTFx) |
| `warmup_steps` | 1 | Warmup iterations before timing |

### Logging (`logging.*`)

| Option | Default | Description |
|--------|---------|-------------|
| `verbose` | false | Show all NeMo internal logs (useful for debugging) |
| `suppress_repetitive_logs` | true | Suppress repetitive NeMo logs during chunk processing |

## Recommended Models

| Model | Speed | Accuracy | Features |
|-------|-------|----------|----------|
| `nvidia/parakeet-tdt-0.6b-v2` | Fast | **Best (EN)** | **Default**. 1.69% WER, auto-punctuation |
| `nvidia/parakeet-tdt-0.6b-v3` | Fast | Excellent | Multilingual (25 languages), auto language detection |
| `nvidia/parakeet-tdt-1.1b` | Medium | Best | Highest accuracy, no auto-punctuation |
| `nvidia/parakeet-ctc-1.1b` | Fastest | Good | Fastest inference |
| `nvidia/canary-1b-v2` | Medium | Good | Multilingual, supports translation |

### Model Selection Guide

- **English subtitles**: `parakeet-tdt-0.6b-v2` (default, best out-of-box experience)
- **Multilingual**: `parakeet-tdt-0.6b-v3` (25 languages, auto-detection)
- **Highest accuracy**: `parakeet-tdt-1.1b` (lowest WER, but no punctuation)
- **Fastest speed**: `parakeet-ctc-1.1b`
- **Translation**: `canary-1b-v2` (25 languages, transcription + translation)

> **Note**: `parakeet-tdt-1.1b` produces lowercase output without punctuation. The script automatically uses word-level timestamps to generate fine-grained subtitles.

## Long Audio Support

The script uses **audio chunking** to handle videos of any length:

- Automatically splits long audio into smaller chunks (default: 5 minutes)
- Chunks overlap (default: 2 seconds) to ensure accurate boundaries
- Merges subtitles from all chunks, handling duplicates automatically
- Long-audio attention tweaks are gated by `audio.long_audio_threshold` (default disables; lower to enable)

**GPU Memory Recommendations:**

| GPU VRAM | `max_chunk_duration` |
|----------|---------------------|
| 8GB | 300 (default) |
| 16GB | 600 |
| 24GB+ | 0 (no chunking) |

## Timestamp Priority

The script obtains timestamps in this priority order:

1. **Segment-level**: Direct segment timestamps from model (most accurate)
2. **Word-level**: Word timestamps grouped by line length/duration/gaps
3. **Fallback**: Estimated by speech rate (~150 words/min) when no timestamps available

> **Auto Fallback**: If average segment length exceeds `max_segment_duration * 2` (e.g., models without punctuation), the script automatically switches to word-level timestamps.

## Project Structure

```
nemoscribe/
├── __init__.py        # Package entry, version info
├── __main__.py        # python -m nemoscribe support
├── cli.py             # CLI parsing and entry point
├── config.py          # All dataclass configurations
├── audio.py           # Audio processing with ffmpeg
├── vad.py             # Voice Activity Detection
├── transcriber.py     # ASR model and transcription
├── srt.py             # SRT formatting and output
├── postprocess.py     # ITN, segment merging
└── log_utils.py       # Log filtering
```

## Supported Video Formats

`.mp4`, `.mkv`, `.avi`, `.mov`, `.webm`, `.m4v`

## Example Output

```srt
1
00:00:00,120 --> 00:00:03,450
Welcome to our show today.

2
00:00:03,680 --> 00:00:07,200
We have an exciting episode planned for you.

3
00:00:07,450 --> 00:00:11,800
Let's get started with our first topic.
```

## Testing

```bash
# Run all tests
uv run python tests/test_improvements.py

# Run specific test
uv run python tests/test_improvements.py --test vad
uv run python tests/test_improvements.py --test itn
uv run python tests/test_improvements.py --test segmentation
uv run python tests/test_improvements.py --test metrics

# Available tests: baseline, vad, itn, decoding, segmentation, merging, performance, metrics, srt, full
```

### Test Coverage

- **baseline_config**: Default configuration backward compatibility
- **vad_config**: VAD configuration correctness
- **itn_functions**: ITN normalization functionality
- **decoding_config**: Decoding configuration (CUDA graphs)
- **smart_segmentation**: Smart segmentation logic
- **segment_merging**: Overlapping segment merging
- **performance_config**: Performance configuration
- **quality_metrics**: WER/CER calculation
- **srt_formatting**: SRT formatting
- **full_config**: Complete configuration combination

## Quality Metrics

Calculate transcription quality using NeMo's official tools:

```python
from tests.test_improvements import calculate_transcription_quality

result = calculate_transcription_quality(
    hypothesis="transcribed text",
    reference="ground truth text"
)
print(f"WER: {result['wer']:.2%}")
print(f"CER: {result['cer']:.2%}")
```

Output includes: `wer`, `cer`, `insertion_rate`, `deletion_rate`, `substitution_rate`

## Troubleshooting

### CUDA Out of Memory

Reduce chunk size:
```bash
uv run nemoscribe video_path=video.mp4 audio.max_chunk_duration=180.0
```

### Timestamps Not Accurate

Use a model with timestamp support (`parakeet-tdt-*` recommended) and adjust segmentation parameters:
```bash
uv run nemoscribe video_path=video.mp4 \
  subtitle.max_segment_duration=3.0 \
  subtitle.word_gap_threshold=0.5
```

### Model Download Slow

Models are automatically downloaded from HuggingFace/NGC on first use. For slow connections:
```bash
# Use HuggingFace mirror (China mainland)
export HF_ENDPOINT=https://hf-mirror.com
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository at [github.com/charles1018/NemoScribe](https://github.com/charles1018/NemoScribe)
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

For bug reports and feature requests, please [open an issue](https://github.com/charles1018/NemoScribe/issues).

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

NemoScribe is built upon the following open-source projects:

- **[NVIDIA NeMo](https://github.com/NVIDIA/NeMo)** - Neural Modules toolkit for conversational AI (Apache 2.0 License)
- **[Parakeet-TDT](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2)** - NVIDIA's state-of-the-art ASR model (CC-BY-4.0 License)

We thank NVIDIA for making these excellent tools and models available to the community.

## References

### Model Resources

| Resource | Description |
|----------|-------------|
| [nvidia/parakeet-tdt-0.6b-v2](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2) | Default model, architecture and best practices |
| [nvidia/parakeet-tdt-0.6b-v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) | Multilingual version, 25 languages |
| [nvidia/canary-1b-v2](https://huggingface.co/nvidia/canary-1b-v2) | Multilingual with translation support |
| [HuggingFace Space Demo](https://huggingface.co/spaces/nvidia/parakeet-tdt-0.6b-v2) | Official demo with long audio handling |

### NeMo Framework References

| File Path | Description |
|-----------|-------------|
| `examples/asr/transcribe_speech.py` | Main architecture reference |
| `nemo/collections/asr/parts/utils/transcribe_utils.py` | Core utilities: `get_inference_device()`, `get_inference_dtype()` |
| `nemo/collections/asr/parts/utils/rnnt_utils.py` | `Hypothesis` class, timestamp data structure |

### Key Implementation Details

**Long Audio Optimization** (from HuggingFace Space):
```python
# Switch to local attention for memory efficiency on audio >8 minutes
model.change_attention_model("rel_pos_local_attn", [256, 256])
model.change_subsampling_conv_chunking_factor(1)  # 1 = auto select
```

**Timestamp Data Structure** (from `Hypothesis`):
```python
{
    'segment': [{'start': float, 'end': float, 'segment': str}, ...],
    'word': [{'start': float, 'end': float, 'word': str}, ...],
    'char': [...]  # character-level timestamps
}
```

### Documentation

- [NeMo ASR Documentation](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/asr/intro.html)
- [NeMo GitHub Repository](https://github.com/NVIDIA/NeMo)
- [Parakeet Model Card](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/nemo/models/parakeet-tdt-0.6b-v2)
