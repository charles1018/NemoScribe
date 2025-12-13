# NemoScribe Modularization Design

## Overview

**Goal**: Refactor the 2100+ line `nemoscribe.py` into a modular, layered architecture for better maintainability and extensibility.

**Constraints**:
- CLI parameter format must be fully backward compatible
- Maintain Hydra-style `key=value` syntax
- No public Python API (internal modularization only)

**Design Decisions**:
- Architecture: Functional layered architecture
- API: CLI only (no Python library API)
- Compatibility: 100% backward compatible with existing CLI parameters

## Target Structure

```
nemoscribe/
├── __init__.py        # Package entry, version info (~10 lines)
├── __main__.py        # python -m nemoscribe support (~5 lines)
├── cli.py             # CLI parsing and entry point (~150 lines)
├── config.py          # All dataclass configurations (~250 lines)
├── audio.py           # Audio processing with ffmpeg (~200 lines)
├── vad.py             # Voice Activity Detection (~250 lines)
├── transcriber.py     # ASR model and transcription (~300 lines)
├── srt.py             # SRT formatting and output (~200 lines)
├── postprocess.py     # ITN, segment merging (~150 lines)
└── logging.py         # Log filtering (~100 lines)
```

## Module Specifications

### Module 1: `config.py` - Configuration Definitions

**Responsibility**: Centralize all dataclass configuration classes

**Content** (from original lines 93-341):
```python
@dataclass
class SubtitleConfig: ...      # Subtitle formatting settings

@dataclass
class AudioConfig: ...         # Audio processing settings

@dataclass
class VADConfig: ...           # VAD settings

@dataclass
class PostProcessingConfig: ...# ITN settings

@dataclass
class DecodingConfig: ...      # Decoding strategy settings

@dataclass
class PerformanceConfig: ...   # Performance measurement settings

@dataclass
class LoggingConfig: ...       # Logging settings

@dataclass
class VideoToSRTConfig: ...    # Main config (combines all above)
```

**Public Interface**:
- Other modules import via `from nemoscribe.config import VideoToSRTConfig, VADConfig`
- Configuration structure remains unchanged for CLI backward compatibility

**Dependencies**: None (base module)

---

### Module 2: `audio.py` - Audio Processing

**Responsibility**: Handle all ffmpeg and audio splitting operations

**Content** (from original lines 348-478):
```python
def check_ffmpeg() -> bool:
    """Check if ffmpeg is available"""

def get_media_duration(file_path: str) -> float:
    """Get media file duration using ffprobe"""

def extract_audio(
    video_path: str,
    audio_path: str,
    sample_rate: int = 16000,
    start_time: Optional[float] = None,
    duration: Optional[float] = None,
) -> bool:
    """Extract audio from video to WAV format"""

def create_audio_chunks(
    video_path: str,
    output_dir: str,
    total_duration: float,
    sample_rate: int = 16000,
    max_chunk_duration: float = 600.0,
    overlap: float = 2.0,
) -> List[Tuple[str, float, float, float]]:
    """Split long audio into chunks (fixed duration splitting)"""
```

**Dependencies**: None (base module, only stdlib `subprocess`, `os`)

---

### Module 3: `vad.py` - Voice Activity Detection

**Responsibility**: VAD model loading, speech detection, smart split point calculation

**Content** (from original lines 486-841):
```python
def load_vad_model(model_name: str, device: torch.device):
    """Load VAD model"""

def run_vad_on_audio(
    audio_path: str,
    vad_model,
    vad_cfg: VADConfig,
    device: torch.device,
) -> List[Tuple[float, float]]:
    """Run VAD on audio, return speech segment list"""

def get_silence_gaps_from_speech(
    speech_segments: List[Tuple[float, float]],
    total_duration: float,
    min_silence_for_split: float = 0.3,
) -> List[Tuple[float, float, float]]:
    """Extract silence gaps from speech segments"""

def find_optimal_split_points(
    speech_segments: List[Tuple[float, float]],
    total_duration: float,
    max_chunk_duration: float,
    min_silence_for_split: float = 0.3,
    prefer_longer_silence: bool = True,
) -> List[float]:
    """Find optimal split points based on silence regions"""

def create_audio_chunks_with_vad(
    video_path: str,
    output_dir: str,
    total_duration: float,
    speech_segments: List[Tuple[float, float]],
    audio_cfg: AudioConfig,
) -> List[Tuple[str, float, float, float]]:
    """Smart audio splitting using VAD information"""
```

**Dependencies**:
- `from nemoscribe.config import VADConfig, AudioConfig`
- `from nemoscribe.audio import extract_audio`

---

### Module 4: `srt.py` - SRT Formatting and Output

**Responsibility**: SRT subtitle formatting, timestamp conversion, segment processing

**Content** (from original lines 1137-1358):
```python
def format_srt_timestamp(seconds: float) -> str:
    """Convert seconds to SRT timestamp format (HH:MM:SS,mmm)"""

def hypothesis_to_srt_segments(
    hypothesis: Hypothesis,
    max_chars_per_line: int = 42,
    max_segment_duration: float = 5.0,
    word_gap_threshold: Optional[float] = 0.8,
) -> List[Tuple[float, float, str]]:
    """Convert NeMo Hypothesis to SRT segments"""

def write_srt_file(
    segments: List[Tuple[float, float, str]],
    output_path: str,
) -> None:
    """Write segments to SRT file"""

def clip_segments_to_window(
    segments: List[Tuple[float, float, str]],
    window_start: float,
    window_end: float,
    tolerance: float = 0.15,
) -> List[Tuple[float, float, str]]:
    """Clip segments to fit time window (handle chunk boundaries)"""
```

**Dependencies**:
- Type hint only: `nemo.collections.asr.parts.utils.rnnt_utils.Hypothesis`
- No internal module dependencies (pure functions)

---

### Module 5: `postprocess.py` - Post-processing

**Responsibility**: ITN normalization, overlapping segment merging, deduplication

**Content** (from original lines 853-951, 1360-1505):
```python
# === ITN (Inverse Text Normalization) ===
_ITN_NORMALIZER_CACHE = {}  # Global cache

def get_itn_normalizer(lang: str = "en", input_case: str = "lower_cased"):
    """Get ITN normalizer (with caching)"""

def apply_itn(text: str, normalizer) -> str:
    """Apply ITN to single text"""

def apply_itn_to_segments(
    segments: List[Tuple[float, float, str]],
    normalizer,
) -> List[Tuple[float, float, str]]:
    """Apply ITN to all segments"""

# === Segment Merging and Deduplication ===
def merge_overlapping_segments(
    all_segments: List[Tuple[float, float, str]],
    overlap_threshold: float = 0.1,
    merge_strategy: str = "prefer_longer",
) -> List[Tuple[float, float, str]]:
    """Merge overlapping segments from different chunks"""

def deduplicate_segments(
    segments: List[Tuple[float, float, str]],
    similarity_threshold: float = 0.8,
) -> List[Tuple[float, float, str]]:
    """Remove duplicate segments based on text similarity"""
```

**Dependencies**:
- No internal module dependencies (pure functions)
- Optional: `nemo_text_processing` (for ITN feature)

---

### Module 6: `transcriber.py` - ASR Model and Transcription

**Responsibility**: ASR model loading, decoding strategy setup, core transcription logic

**Content** (from original lines 958-1024, 1513-1915):
```python
# === Decoding Strategy ===
def setup_decoding_strategy(asr_model: ASRModel, cfg: DecodingConfig) -> None:
    """Configure decoding strategy based on model type (CUDA graphs etc.)"""

# === Model Loading ===
def load_asr_model(
    pretrained_name: str,
    model_path: Optional[str],
    device: torch.device,
    compute_dtype: torch.dtype,
) -> Tuple[ASRModel, str]:
    """Load ASR model"""

def apply_long_audio_settings(model: ASRModel) -> bool:
    """Apply long audio optimization settings (local attention)"""

def revert_long_audio_settings(model: ASRModel) -> None:
    """Revert long audio settings"""

# === Transcription ===
def transcribe_audio_chunk(
    audio_path: str,
    asr_model: ASRModel,
    cfg: VideoToSRTConfig,
    time_offset: float = 0.0,
    suppress_logs: bool = False,
) -> List[Tuple[float, float, str]]:
    """Transcribe single audio chunk"""

def transcribe_video(
    video_path: str,
    output_path: str,
    asr_model: ASRModel,
    cfg: VideoToSRTConfig,
    device: torch.device,
    vad_model=None,
    itn_normalizer=None,
) -> str:
    """Transcribe complete video (main orchestration function)"""
```

**Dependencies**:
- `from nemoscribe.config import VideoToSRTConfig, DecodingConfig`
- `from nemoscribe.audio import extract_audio, create_audio_chunks, get_media_duration`
- `from nemoscribe.vad import run_vad_on_audio, create_audio_chunks_with_vad`
- `from nemoscribe.srt import hypothesis_to_srt_segments, clip_segments_to_window, write_srt_file`
- `from nemoscribe.postprocess import apply_itn_to_segments, merge_overlapping_segments, deduplicate_segments`
- `from nemoscribe.logging import suppress_repetitive_nemo_logs`

---

### Module 7: `logging.py` - Log Management

**Responsibility**: NeMo log filtering, repetitive message suppression

**Content** (from original lines 1031-1129):
```python
class NeMoLogFilter:
    """
    Custom log filter to suppress repetitive NeMo internal logs

    Filtered message types:
    - "Using RNNT Loss"
    - "Joint fused batch size"
    - "Timestamps requested, setting decoding timestamps"
    - Lhotse dataloader warnings
    """

    FILTER_PATTERNS = [...]

    def __init__(self): ...
    def filter(self, record) -> bool: ...


# Global filter instance
_nemo_log_filter = NeMoLogFilter()


class suppress_repetitive_nemo_logs:
    """
    Context manager to temporarily suppress repetitive NeMo logs

    Usage:
        with suppress_repetitive_nemo_logs():
            model.transcribe(...)
    """

    def __init__(self, enabled: bool = True): ...
    def __enter__(self): ...
    def __exit__(self, exc_type, exc_val, exc_tb): ...
```

**Dependencies**:
- Python stdlib `logging` only
- No internal module dependencies

---

### Module 8: `cli.py` - CLI Entry Point

**Responsibility**: Parse CLI arguments, orchestrate processing flow, program entry point

**Content** (from original lines 1917-2122):
```python
def process_videos(cfg: VideoToSRTConfig) -> List[str]:
    """
    Main orchestration function for video processing

    Responsibilities:
    - Validate inputs
    - Setup device and dtype
    - Load ASR model and VAD model
    - Initialize ITN
    - Collect video files
    - Call transcribe_video() for each
    """

def parse_args(args: List[str], cfg: VideoToSRTConfig) -> VideoToSRTConfig:
    """
    Parse Hydra-style key=value arguments
    (Extracted from original main() for testability)
    """

def main() -> int:
    """
    CLI entry point

    - Create default config
    - Parse command line arguments
    - Call process_videos()
    - Output results
    """

if __name__ == "__main__":
    sys.exit(main())
```

**Dependencies**:
- `from nemoscribe.config import VideoToSRTConfig`
- `from nemoscribe.audio import check_ffmpeg`
- `from nemoscribe.vad import load_vad_model`
- `from nemoscribe.transcriber import load_asr_model, setup_decoding_strategy, transcribe_video`
- `from nemoscribe.postprocess import get_itn_normalizer`

---

### Module 9: `__init__.py` - Package Definition

**Responsibility**: Define package info, control public interface

**Content**:
```python
"""
NemoScribe - Video to SRT Subtitle Generator

Convert video files to SRT subtitles using NVIDIA NeMo ASR models.
"""

__version__ = "0.1.0"
__author__ = "charles1018"

# Only export CLI entry point
from nemoscribe.cli import main

__all__ = ["main", "__version__"]
```

---

### Module 10: `__main__.py` - Module Execution Support

**Responsibility**: Support `python -m nemoscribe` execution

**Content**:
```python
"""
Allow running as: python -m nemoscribe
"""
from nemoscribe.cli import main

if __name__ == "__main__":
    main()
```

---

## Dependency Graph

```
                    ┌─────────────┐
                    │   cli.py    │
                    └──────┬──────┘
                           │
           ┌───────────────┼───────────────┐
           │               │               │
           ▼               ▼               ▼
    ┌─────────────┐ ┌─────────────┐ ┌─────────────┐
    │transcriber.py│ │  vad.py    │ │postprocess.py│
    └──────┬──────┘ └──────┬──────┘ └─────────────┘
           │               │
     ┌─────┴─────┬─────────┤
     │           │         │
     ▼           ▼         ▼
┌─────────┐ ┌─────────┐ ┌─────────┐
│ srt.py  │ │audio.py │ │logging.py│
└─────────┘ └─────────┘ └─────────┘

           ┌─────────────┐
           │  config.py  │  ◄── Used by all modules
           └─────────────┘
```

## pyproject.toml Changes

```toml
[project.scripts]
nemoscribe = "nemoscribe.cli:main"  # Changed from: nemoscribe:main
```

## Migration Notes

1. **File Structure Change**:
   - Old: Single `nemoscribe.py` at project root
   - New: `nemoscribe/` package directory

2. **Import Path Change**:
   - Entry point changes from `nemoscribe:main` to `nemoscribe.cli:main`
   - Internal imports use `from nemoscribe.xxx import yyy`

3. **Backward Compatibility**:
   - All CLI commands remain identical
   - All configuration parameters unchanged
   - User scripts require no modifications

4. **Testing**:
   - Existing tests should work with minimal path adjustments
   - Each module can now be unit tested independently
