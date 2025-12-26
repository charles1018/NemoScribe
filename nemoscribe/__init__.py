"""
NemoScribe - Video to SRT Subtitle Generator

Convert video files to SRT subtitles using NVIDIA NeMo ASR models.
"""

__version__ = "0.2.1"
__author__ = "charles1018"

# CLI entry point
from nemoscribe.cli import main

# Configuration classes (for testing and advanced usage)
from nemoscribe.config import (
    AudioConfig,
    DecodingConfig,
    LoggingConfig,
    PerformanceConfig,
    PostProcessingConfig,
    SubtitleConfig,
    VADConfig,
    VideoToSRTConfig,
)

# SRT formatting functions
from nemoscribe.srt import (
    clip_segments_to_window,
    format_srt_timestamp,
    hypothesis_to_srt_segments,
    write_srt_file,
)

# Post-processing functions
from nemoscribe.postprocess import (
    apply_itn,
    apply_itn_to_segments,
    deduplicate_segments,
    get_itn_normalizer,
    merge_overlapping_segments,
)

# VAD functions
from nemoscribe.vad import (
    find_optimal_split_points,
    get_silence_gaps_from_speech,
)

__all__ = [
    # Version info
    "__version__",
    # CLI
    "main",
    # Config classes
    "VideoToSRTConfig",
    "SubtitleConfig",
    "AudioConfig",
    "VADConfig",
    "PostProcessingConfig",
    "DecodingConfig",
    "PerformanceConfig",
    "LoggingConfig",
    # SRT functions
    "format_srt_timestamp",
    "hypothesis_to_srt_segments",
    "write_srt_file",
    "clip_segments_to_window",
    # Post-processing functions
    "get_itn_normalizer",
    "apply_itn",
    "apply_itn_to_segments",
    "merge_overlapping_segments",
    "deduplicate_segments",
    # VAD functions
    "find_optimal_split_points",
    "get_silence_gaps_from_speech",
]
