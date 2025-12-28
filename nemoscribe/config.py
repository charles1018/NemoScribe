# MIT License
#
# Copyright (c) 2025 charles1018
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""
Configuration dataclasses for NemoScribe.

This module contains all configuration classes using Python dataclasses
for Hydra-style structured configuration.
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class SubtitleConfig:
    """Configuration for subtitle formatting."""

    # Maximum characters per subtitle line
    max_chars_per_line: int = 42

    # Maximum duration (seconds) for a single subtitle segment
    max_segment_duration: float = 5.0

    # Start new subtitle if word gap exceeds this threshold (seconds)
    # Set to None to disable gap-based splitting
    word_gap_threshold: Optional[float] = 0.8


@dataclass
class AudioConfig:
    """Configuration for audio processing."""

    # Sample rate for audio extraction (Hz)
    sample_rate: int = 16000

    # Duration threshold (seconds) to enable long audio optimizations
    # For audio longer than this, local attention + chunking will be enabled
    # Note: This feature may cause dtype issues on some systems, disabled by default
    # Use chunking (max_chunk_duration) instead for reliable processing
    long_audio_threshold: float = 99999.0  # Disabled by default, use chunking instead

    # Maximum chunk duration for GPUs with limited memory (seconds)
    # Set to 0 to disable chunking (requires large GPU memory)
    # For 8GB GPU: ~300s recommended
    # For 16GB GPU: ~600s recommended
    # For 24GB+ GPU: can set to 0 (no chunking)
    max_chunk_duration: float = 300.0  # 5 minutes per chunk (safe for 8GB GPU)

    # Overlap between chunks (seconds) for better boundary handling
    chunk_overlap: float = 2.0

    # Smart Segmentation: Use VAD-based optimal split points when VAD is enabled
    # This improves chunking by:
    # 1. Finding silence gaps between speech segments
    # 2. Splitting at the longest silence within allowed duration
    # 3. Avoiding cuts in the middle of speech
    smart_segmentation: bool = True  # Auto-enabled when VAD is active

    # Minimum silence duration to consider as a valid split point (seconds)
    # Shorter silences will be ignored
    min_silence_for_split: float = 0.3

    # Prefer splitting at longer silences when multiple options exist
    prefer_longer_silence: bool = True


@dataclass
class VADConfig:
    """
    Configuration for Voice Activity Detection (VAD).

    VAD filters non-speech segments (music, noise, silence) before ASR to reduce
    hallucinations and improve accuracy. Based on NeMo's frame_vad_inference_postprocess.yaml.

    When enabled, VAD will:
    1. Run frame-level speech detection on audio
    2. Apply smoothing and post-processing
    3. Generate speech segment timestamps
    4. Use these segments for smarter audio chunking
    """

    # Master switch - disabled by default for backward compatibility
    enabled: bool = False

    # VAD model name (NGC pretrained or local .nemo path)
    # Options: "vad_multilingual_frame_marblenet" (recommended), "vad_multilingual_marblenet"
    model: str = "vad_multilingual_frame_marblenet"

    # Frame parameters (must match pretrained model expectations)
    # window_length_in_sec must be 0.0 for frame-VAD models
    window_length_in_sec: float = 0.0
    # Frame shift - 0.02s (20ms) for pretrained NeMo VAD models
    shift_length_in_sec: float = 0.02

    # Post-processing parameters (from frame_vad_infer_postprocess.yaml)
    # Onset threshold for detecting beginning of speech (0-1)
    onset: float = 0.3
    # Offset threshold for detecting end of speech (0-1)
    offset: float = 0.3
    # Padding before each speech segment (seconds)
    pad_onset: float = 0.2
    # Padding after each speech segment (seconds)
    pad_offset: float = 0.2
    # Minimum speech segment duration to keep (seconds)
    min_duration_on: float = 0.2
    # Minimum non-speech gap to merge (seconds)
    min_duration_off: float = 0.2
    # Filter short speech first before merging gaps
    filter_speech_first: bool = True


@dataclass
class PostProcessingConfig:
    """
    Configuration for text post-processing.

    Includes Inverse Text Normalization (ITN) for converting spoken forms to written forms.
    Based on NeMo's streaming inference pipeline (asr_streaming_infer.py).

    ITN Examples:
    - "twenty five dollars" -> "$25"
    - "january first twenty twenty five" -> "January 1st, 2025"
    - "three point one four" -> "3.14"
    """

    # Inverse Text Normalization - disabled by default (requires nemo_text_processing)
    enable_itn: bool = False

    # Language for ITN (currently only "en" is well supported)
    itn_lang: str = "en"

    # Input case handling: "lower_cased" or "cased"
    # Use "lower_cased" for models without auto-capitalization (e.g., parakeet-tdt-1.1b)
    # Use "cased" for models with auto-capitalization (e.g., parakeet-tdt-0.6b-v2)
    itn_input_case: str = "lower_cased"


@dataclass
class DecodingConfig:
    """
    Configuration for ASR decoding optimization.

    Based on NeMo's transcribe_speech.py (lines 166-175, 308-348).
    Supports both RNNT/TDT and CTC models with appropriate settings.

    Key optimizations:
    - CUDA graphs (fused_batch_size=-1) for faster inference on GPU
    - Timestamp type selection for word/segment level output
    - Segment separators for punctuation-based splitting
    """

    # Enable CUDA graphs for RNNT/TDT models
    # -1 enables CUDA graphs (recommended for GPU)
    # 0 disables CUDA graphs
    # >0 sets fixed batch size for fused operations
    rnnt_fused_batch_size: int = -1

    # Timestamp type: "char", "word", "segment", or "all" (default)
    # "word" - word-level timestamps only
    # "segment" - sentence/segment-level timestamps only
    # "all" - both word and segment timestamps (recommended for SRT)
    rnnt_timestamp_type: str = "all"
    ctc_timestamp_type: str = "all"

    # Compute timestamps (required for SRT generation)
    # Setting to None lets the model decide based on default
    compute_timestamps: Optional[bool] = None

    # Segment separators for punctuation-based splitting
    # When set, segments will be split at these punctuation marks
    # This helps reduce long segments in drama/movie content
    # Set to empty list to disable punctuation-based splitting
    segment_separators: List[str] = field(default_factory=lambda: [".", "?", "!"])


@dataclass
class PerformanceConfig:
    """
    Configuration for performance measurement and optimization.

    Based on NeMo's transcribe_speech.py (lines 209-211, 380-396, 490-495).
    Useful for benchmarking different configurations.
    """

    # Calculate Real-Time Factor (RTFx)
    # RTFx = audio_duration / processing_time
    # RTFx > 1.0 means faster than real-time
    calculate_rtfx: bool = False

    # Number of warmup iterations before measuring (recommended: 1)
    # Warmup helps stabilize GPU performance for accurate measurements
    warmup_steps: int = 1


@dataclass
class LoggingConfig:
    """
    Configuration for logging behavior.

    Controls verbosity of NeMo internal logs during transcription.
    """

    # Verbose mode: show all NeMo internal logs (useful for debugging)
    # When False, suppresses repetitive decoder initialization logs during chunk processing
    verbose: bool = False

    # Suppress specific repetitive NeMo logs during chunk transcription
    # These logs are emitted every time transcribe() is called:
    # - "Using RNNT Loss : tdt"
    # - "Joint fused batch size <= 0"
    # - "Timestamps requested, setting decoding timestamps to True"
    suppress_repetitive_logs: bool = True


@dataclass
class VideoToSRTConfig:
    """
    NemoScribe transcription configuration.
    """

    # Model configuration
    model_path: Optional[str] = None  # Path to .nemo file
    pretrained_name: str = "nvidia/parakeet-tdt-0.6b-v2"  # Pretrained model name

    # Input configuration
    video_path: Optional[str] = None  # Single video file
    video_dir: Optional[str] = None  # Directory of videos

    # Output configuration
    output_path: Optional[str] = None  # Output SRT file (for single video)
    output_dir: Optional[str] = None  # Output directory (for batch processing)

    # Video file extensions to process
    video_extensions: List[str] = field(
        default_factory=lambda: [".mp4", ".mkv", ".avi", ".mov", ".webm", ".m4v"]
    )

    # Device configuration
    cuda: Optional[int] = None  # CUDA device ID (None=auto, negative=CPU)
    allow_mps: bool = False  # Allow Apple Silicon MPS

    # Precision configuration
    compute_dtype: Optional[str] = None  # float32, bfloat16, float16 (None=auto)
    matmul_precision: str = "high"  # Matrix multiplication precision

    # Subtitle formatting
    subtitle: SubtitleConfig = field(default_factory=SubtitleConfig)

    # Audio processing
    audio: AudioConfig = field(default_factory=AudioConfig)

    # Voice Activity Detection (disabled by default for backward compatibility)
    vad: VADConfig = field(default_factory=VADConfig)

    # Text post-processing (ITN, etc.) - disabled by default
    postprocessing: PostProcessingConfig = field(default_factory=PostProcessingConfig)

    # Decoding optimization (CUDA graphs, timestamp types)
    decoding: DecodingConfig = field(default_factory=DecodingConfig)

    # Performance measurement
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)

    # Logging configuration
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    # Overwrite existing SRT files
    overwrite: bool = True
