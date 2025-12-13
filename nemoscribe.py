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
NemoScribe - Video to SRT Subtitle Generator

Convert video files to SRT subtitles using NVIDIA NeMo ASR models with accurate timestamps.
Built on NeMo framework with Parakeet-TDT as the default model.

Supports up to 3 hours of audio using local attention and chunked inference.

Usage:
    # Single video
    nemoscribe video_path=/path/to/video.mp4

    # With VAD (recommended)
    nemoscribe video_path=/path/to/video.mp4 vad.enabled=true

    # With specific output path
    nemoscribe video_path=/path/to/video.mp4 output_path=/path/to/output.srt

    # Process directory
    nemoscribe video_dir=/path/to/videos/ output_dir=/path/to/subtitles/

Recommended Models:
    - nvidia/parakeet-tdt-0.6b-v2 (default, best English accuracy, auto-punctuation)
    - nvidia/parakeet-tdt-0.6b-v3 (multilingual, 25 languages)
    - nvidia/parakeet-tdt-1.1b (highest accuracy, no auto-punctuation)
    - nvidia/parakeet-ctc-1.1b (fastest inference)

Requirements:
    - ffmpeg (for audio extraction)
    - NeMo toolkit with ASR support
    - NVIDIA GPU (recommended)

Acknowledgements:
    This project uses the following open-source software and models:
    - NVIDIA NeMo (https://github.com/NVIDIA/NeMo) - Apache 2.0 License
    - Parakeet-TDT model by NVIDIA (https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2) - CC-BY-4.0 License
"""

import gc
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from typing import List, Optional, Tuple

import torch
from omegaconf import OmegaConf

from nemo.collections.asr.models import ASRModel
from nemo.collections.asr.parts.utils.rnnt_utils import Hypothesis
from nemo.collections.asr.parts.utils.transcribe_utils import (
    get_inference_device,
    get_inference_dtype,
)
from nemo.collections.asr.parts.utils.vad_utils import (
    generate_vad_segment_table_per_tensor,
    init_frame_vad_model,
)
from nemo.utils import logging


# =============================================================================
# Configuration (using dataclasses for Hydra-style structured config)
# =============================================================================


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


# =============================================================================
# Audio/Video Processing Utilities
# =============================================================================


def check_ffmpeg() -> bool:
    """Check if ffmpeg is available."""
    try:
        subprocess.run(
            ["ffmpeg", "-version"],
            capture_output=True,
            check=True,
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def get_media_duration(file_path: str) -> float:
    """Get media file duration in seconds using ffprobe."""
    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        file_path,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, check=True, text=True)
        return float(result.stdout.strip())
    except (subprocess.CalledProcessError, ValueError) as e:
        logging.warning(f"Could not get duration for {file_path}: {e}")
        return 0.0


def extract_audio(
    video_path: str,
    audio_path: str,
    sample_rate: int = 16000,
    start_time: Optional[float] = None,
    duration: Optional[float] = None,
) -> bool:
    """
    Extract audio from video file using ffmpeg.

    Args:
        video_path: Input video file path
        audio_path: Output audio file path
        sample_rate: Target sample rate (default: 16000 Hz for ASR)
        start_time: Start time in seconds (optional)
        duration: Duration in seconds (optional)

    Returns:
        True if successful
    """
    cmd = ["ffmpeg"]

    # Add seek before input for efficiency
    if start_time is not None:
        cmd.extend(["-ss", str(start_time)])

    cmd.extend(["-i", video_path])

    if duration is not None:
        cmd.extend(["-t", str(duration)])

    cmd.extend([
        "-vn",  # No video
        "-acodec", "pcm_s16le",  # PCM 16-bit little-endian
        "-ar", str(sample_rate),  # Sample rate
        "-ac", "1",  # Mono
        "-y",  # Overwrite output
        audio_path,
    ])

    try:
        subprocess.run(cmd, capture_output=True, check=True)
        return True
    except subprocess.CalledProcessError as e:
        logging.error(f"Failed to extract audio: {e.stderr.decode()}")
        return False


def create_audio_chunks(
    video_path: str,
    output_dir: str,
    total_duration: float,
    sample_rate: int = 16000,
    max_chunk_duration: float = 600.0,
    overlap: float = 2.0,
) -> List[Tuple[str, float, float]]:
    """
    Split video audio into chunks for transcription.

    Args:
        video_path: Input video file
        output_dir: Directory to store audio chunks
        total_duration: Total video duration in seconds
        sample_rate: Audio sample rate
        max_chunk_duration: Maximum duration per chunk (seconds)
        overlap: Overlap between chunks (seconds)

    Returns:
        List of (audio_path, chunk_start_time, chunk_end_time) tuples
        where chunk_start_time and chunk_end_time are the times in the original video
    """
    chunks = []
    chunk_idx = 0
    current_start = 0.0

    while current_start < total_duration:
        chunk_end = min(current_start + max_chunk_duration, total_duration)

        # Calculate extraction parameters with overlap
        extract_start = max(0.0, current_start - overlap) if current_start > 0 else 0.0
        extract_end = min(total_duration, chunk_end + overlap) if chunk_end < total_duration else total_duration
        extract_duration = extract_end - extract_start

        audio_path = os.path.join(output_dir, f"chunk_{chunk_idx:04d}.wav")

        if extract_audio(
            video_path,
            audio_path,
            sample_rate,
            start_time=extract_start,
            duration=extract_duration,
        ):
            # Store: audio_path, original video start time, original video end time
            chunks.append((audio_path, current_start, chunk_end, extract_start))
            chunk_idx += 1
        else:
            logging.warning(f"Failed to extract chunk {chunk_idx}")

        current_start = chunk_end

    return chunks


# =============================================================================
# VAD (Voice Activity Detection) Utilities
# =============================================================================


def load_vad_model(
    model_name: str,
    device: torch.device,
):
    """
    Load VAD model from pretrained name or local path.

    Args:
        model_name: NGC model name or path to .nemo file
        device: Target device

    Returns:
        Loaded VAD model
    """
    logging.info(f"Loading VAD model: {model_name}")
    vad_model = init_frame_vad_model(model_name)
    vad_model = vad_model.to(device)
    vad_model.eval()
    return vad_model


def run_vad_on_audio(
    audio_path: str,
    vad_model,
    vad_cfg: VADConfig,
    device: torch.device,
) -> List[Tuple[float, float]]:
    """
    Run Voice Activity Detection on an audio file.

    Uses NeMo's frame-VAD model to detect speech segments.
    Based on speech_to_text_with_vad.py pattern.

    Args:
        audio_path: Path to audio file (WAV, 16kHz mono)
        vad_model: Loaded VAD model
        vad_cfg: VAD configuration
        device: Computation device

    Returns:
        List of (start_time, end_time) tuples for speech segments
    """
    import librosa

    # Load audio with librosa (NeMo's preferred audio loading library)
    # This returns a 1D numpy array
    audio_data, sample_rate = librosa.load(audio_path, sr=16000, mono=True)

    # Convert to tensor with batch dimension: [1, samples]
    waveform = torch.tensor(audio_data, dtype=torch.float32).unsqueeze(0).to(device)
    input_length = torch.tensor([waveform.shape[1]], device=device)

    # Run VAD inference
    with torch.inference_mode():
        with torch.amp.autocast(device.type):
            log_probs = vad_model(input_signal=waveform, input_signal_length=input_length)

        # Convert log probabilities to probabilities
        probs = torch.softmax(log_probs, dim=-1)

        # Handle different output shapes
        if len(probs.shape) == 3:
            probs = probs.squeeze(0)  # [1, T, C] -> [T, C]

        # Get speech probability (class 1)
        speech_probs = probs[:, 1].cpu()

    # Convert frame predictions to speech segments using NeMo's official utility
    per_args = {
        'frame_length_in_sec': vad_cfg.shift_length_in_sec,
        'onset': vad_cfg.onset,
        'offset': vad_cfg.offset,
        'pad_onset': vad_cfg.pad_onset,
        'pad_offset': vad_cfg.pad_offset,
        'min_duration_on': vad_cfg.min_duration_on,
        'min_duration_off': vad_cfg.min_duration_off,
        'filter_speech_first': 1.0 if vad_cfg.filter_speech_first else 0.0,
    }

    speech_segments_tensor = generate_vad_segment_table_per_tensor(speech_probs, per_args)

    # Convert tensor to list of tuples
    # Output format: [[start, end, duration], ...]
    speech_segments = []
    if speech_segments_tensor.numel() > 0:
        for seg in speech_segments_tensor:
            start = float(seg[0])
            end = float(seg[1])
            speech_segments.append((start, end))

    return speech_segments


def get_silence_gaps_from_speech(
    speech_segments: List[Tuple[float, float]],
    total_duration: float,
    min_silence_for_split: float = 0.3,
) -> List[Tuple[float, float, float]]:
    """
    Extract silence gaps from speech segments.

    Based on NeMo's vad_utils.py get_nonspeech_segments() pattern.

    Args:
        speech_segments: List of (start, end) speech segment tuples
        total_duration: Total audio duration in seconds
        min_silence_for_split: Minimum silence duration to consider (seconds)

    Returns:
        List of (start, end, duration) tuples for silence gaps
    """
    silence_gaps = []
    prev_end = 0.0

    for start, end in sorted(speech_segments, key=lambda x: x[0]):
        if start > prev_end:
            gap_duration = start - prev_end
            if gap_duration >= min_silence_for_split:
                silence_gaps.append((prev_end, start, gap_duration))
        prev_end = max(prev_end, end)

    # Check for trailing silence
    if total_duration > prev_end + min_silence_for_split:
        trailing_duration = total_duration - prev_end
        silence_gaps.append((prev_end, total_duration, trailing_duration))

    return silence_gaps


def find_optimal_split_points(
    speech_segments: List[Tuple[float, float]],
    total_duration: float,
    max_chunk_duration: float,
    min_silence_for_split: float = 0.3,
    prefer_longer_silence: bool = True,
) -> List[float]:
    """
    Find optimal points to split audio based on silence regions.

    This function implements smart segmentation by:
    1. Identifying all silence gaps from VAD speech segments
    2. For each chunk boundary, finding the best silence gap within the allowed range
    3. Preferring longer silences for cleaner splits

    Based on NeMo's vad_utils.py generate_vad_segment_table() patterns.

    Args:
        speech_segments: List of (start, end) speech segment tuples from VAD
        total_duration: Total audio duration in seconds
        max_chunk_duration: Maximum allowed chunk duration (seconds)
        min_silence_for_split: Minimum silence duration to consider for splitting
        prefer_longer_silence: If True, prefer splitting at longer silences

    Returns:
        List of split points (timestamps) including 0.0 and total_duration
    """
    if not speech_segments:
        # No speech detected - use fixed intervals
        split_points = [0.0]
        current = max_chunk_duration
        while current < total_duration:
            split_points.append(current)
            current += max_chunk_duration
        split_points.append(total_duration)
        return split_points

    # Get all silence gaps
    silence_gaps = get_silence_gaps_from_speech(
        speech_segments, total_duration, min_silence_for_split
    )

    if not silence_gaps:
        # No suitable silence gaps found - use fixed intervals
        logging.debug("No silence gaps found for smart segmentation, using fixed intervals")
        split_points = [0.0]
        current = max_chunk_duration
        while current < total_duration:
            split_points.append(current)
            current += max_chunk_duration
        split_points.append(total_duration)
        return split_points

    # Build split points by finding optimal silences
    split_points = [0.0]
    current_chunk_start = 0.0

    while current_chunk_start < total_duration:
        target_end = current_chunk_start + max_chunk_duration

        if target_end >= total_duration:
            # Last chunk - just extend to end
            break

        # Find all silences within the valid range
        # Allow some flexibility: search from 50% to 150% of max_chunk_duration
        search_start = current_chunk_start + (max_chunk_duration * 0.5)
        search_end = current_chunk_start + (max_chunk_duration * 1.2)

        candidates = []
        for gap_start, gap_end, gap_duration in silence_gaps:
            # Check if this silence gap is within our search range
            gap_midpoint = (gap_start + gap_end) / 2.0
            if search_start <= gap_midpoint <= search_end:
                candidates.append((gap_midpoint, gap_duration))

        if candidates:
            if prefer_longer_silence:
                # Sort by duration (descending) to prefer longer silences
                candidates.sort(key=lambda x: x[1], reverse=True)
            else:
                # Sort by proximity to target
                candidates.sort(key=lambda x: abs(x[0] - target_end))

            best_split = candidates[0][0]
            split_points.append(best_split)
            current_chunk_start = best_split
        else:
            # No silence found in range - look for any silence after current position
            fallback_candidates = [
                (gap_start, gap_end, gap_duration)
                for gap_start, gap_end, gap_duration in silence_gaps
                if gap_start > current_chunk_start + (max_chunk_duration * 0.3)
            ]

            if fallback_candidates:
                # Take the first available silence
                gap_start, gap_end, _ = fallback_candidates[0]
                split_points.append((gap_start + gap_end) / 2.0)
                current_chunk_start = split_points[-1]
            else:
                # Force split at max duration boundary
                forced_split = current_chunk_start + max_chunk_duration
                if forced_split < total_duration:
                    split_points.append(forced_split)
                    current_chunk_start = forced_split
                else:
                    break

    # Ensure final boundary
    if split_points[-1] < total_duration:
        split_points.append(total_duration)

    return split_points


def create_audio_chunks_with_vad(
    video_path: str,
    output_dir: str,
    total_duration: float,
    speech_segments: List[Tuple[float, float]],
    audio_cfg: AudioConfig,
) -> List[Tuple[str, float, float, float]]:
    """
    Create audio chunks based on VAD-detected speech segments with smart segmentation.

    This function uses optimal split point detection to:
    1. Find silence gaps between speech segments
    2. Split at the best silence within each chunk's allowed duration
    3. Prefer longer silences for cleaner audio boundaries
    4. Avoid cutting in the middle of speech

    Based on NeMo's vad_utils.py and speech_to_text_buffered_infer_rnnt.py patterns.

    Args:
        video_path: Input video file
        output_dir: Directory to store audio chunks
        total_duration: Total video duration in seconds
        speech_segments: List of (start, end) speech segment tuples from VAD
        audio_cfg: Audio configuration with segmentation settings

    Returns:
        List of (audio_path, chunk_start, chunk_end, extract_start) tuples
    """
    sample_rate = audio_cfg.sample_rate
    max_chunk_duration = audio_cfg.max_chunk_duration
    min_silence_for_split = audio_cfg.min_silence_for_split
    prefer_longer_silence = audio_cfg.prefer_longer_silence
    smart_segmentation = audio_cfg.smart_segmentation

    if not speech_segments:
        logging.warning("No speech segments detected by VAD, falling back to fixed chunking")
        return create_audio_chunks(
            video_path, output_dir, total_duration, sample_rate, max_chunk_duration
        )

    if smart_segmentation:
        # Use optimal split point detection
        split_points = find_optimal_split_points(
            speech_segments,
            total_duration,
            max_chunk_duration,
            min_silence_for_split,
            prefer_longer_silence,
        )
        logging.info(
            f"Smart segmentation: found {len(split_points) - 1} optimal chunks "
            f"(split at {len(split_points) - 2} silence points)"
        )
    else:
        # Legacy behavior: simple midpoint-based splitting
        silence_gaps = []
        prev_end = 0.0
        for start, end in speech_segments:
            if start > prev_end:
                gap_duration = start - prev_end
                if gap_duration >= min_silence_for_split:
                    silence_gaps.append((prev_end + start) / 2)
            prev_end = end

        if total_duration > prev_end + min_silence_for_split:
            silence_gaps.append(total_duration)

        # Determine chunk boundaries (legacy logic)
        split_points = [0.0]
        current_chunk_start = 0.0

        for silence_midpoint in silence_gaps:
            chunk_duration = silence_midpoint - current_chunk_start
            if chunk_duration >= max_chunk_duration:
                split_points.append(silence_midpoint)
                current_chunk_start = silence_midpoint
            elif silence_midpoint - split_points[-1] > max_chunk_duration * 1.5:
                forced_split = split_points[-1] + max_chunk_duration
                split_points.append(forced_split)
                current_chunk_start = forced_split

        if split_points[-1] < total_duration:
            split_points.append(total_duration)

    # Extract audio for each chunk
    chunks = []
    chunk_overlap = audio_cfg.chunk_overlap

    for i in range(len(split_points) - 1):
        chunk_start = split_points[i]
        chunk_end = split_points[i + 1]

        # Add overlap for boundary handling
        extract_start = max(0.0, chunk_start - chunk_overlap) if chunk_start > 0 else 0.0
        extract_end = min(total_duration, chunk_end + chunk_overlap)
        extract_duration = extract_end - extract_start

        audio_path = os.path.join(output_dir, f"chunk_{i:04d}.wav")

        if extract_audio(
            video_path,
            audio_path,
            sample_rate,
            start_time=extract_start,
            duration=extract_duration,
        ):
            chunks.append((audio_path, chunk_start, chunk_end, extract_start))
        else:
            logging.warning(f"Failed to extract VAD chunk {i}")

    return chunks


# =============================================================================
# ITN (Inverse Text Normalization) Utilities
# =============================================================================


# Global cache for ITN normalizer to avoid reloading
_ITN_NORMALIZER_CACHE = {}


def get_itn_normalizer(
    lang: str = "en",
    input_case: str = "lower_cased",
):
    """
    Get ITN normalizer if available, otherwise return None.

    Uses lazy loading with caching to avoid repeated initialization.
    Handles missing nemo_text_processing gracefully.

    Based on NeMo's asr_streaming_infer.py pattern.

    Args:
        lang: Language code (default: "en")
        input_case: Input case handling ("lower_cased" or "cased")

    Returns:
        InverseNormalizer instance or None if unavailable
    """
    cache_key = f"{lang}_{input_case}"

    if cache_key in _ITN_NORMALIZER_CACHE:
        return _ITN_NORMALIZER_CACHE[cache_key]

    try:
        from nemo_text_processing.inverse_text_normalization.inverse_normalize import InverseNormalizer

        logging.info(f"Initializing ITN normalizer (lang={lang}, input_case={input_case})...")
        normalizer = InverseNormalizer(lang=lang, input_case=input_case)
        _ITN_NORMALIZER_CACHE[cache_key] = normalizer
        logging.info("ITN normalizer initialized successfully")
        return normalizer

    except ImportError:
        logging.warning(
            "nemo_text_processing not installed. ITN disabled. "
            "Install with: pip install nemo_text_processing"
        )
        _ITN_NORMALIZER_CACHE[cache_key] = None
        return None

    except Exception as e:
        logging.warning(f"Failed to initialize ITN normalizer: {e}. ITN disabled.")
        _ITN_NORMALIZER_CACHE[cache_key] = None
        return None


def apply_itn(text: str, normalizer) -> str:
    """
    Apply Inverse Text Normalization to text.

    Converts spoken forms to written forms:
    - "twenty five dollars" -> "$25"
    - "january first" -> "January 1st"
    - "three point one four" -> "3.14"

    Args:
        text: Input text in spoken form
        normalizer: ITN normalizer instance (from get_itn_normalizer)

    Returns:
        Normalized text, or original text if normalization fails
    """
    if normalizer is None:
        return text

    if not text or not text.strip():
        return text

    try:
        normalized = normalizer.normalize(text, verbose=False)
        return normalized
    except Exception as e:
        logging.debug(f"ITN normalization failed for '{text[:50]}...': {e}")
        return text


def apply_itn_to_segments(
    segments: List[Tuple[float, float, str]],
    normalizer,
) -> List[Tuple[float, float, str]]:
    """
    Apply ITN to all segments.

    Args:
        segments: List of (start, end, text) tuples
        normalizer: ITN normalizer instance

    Returns:
        Segments with ITN applied to text
    """
    if normalizer is None:
        return segments

    return [
        (start, end, apply_itn(text, normalizer))
        for start, end, text in segments
    ]


# =============================================================================
# Decoding Strategy Utilities
# =============================================================================


def setup_decoding_strategy(
    asr_model: ASRModel,
    cfg: DecodingConfig,
) -> None:
    """
    Configure model decoding strategy based on model type.

    Follows pattern from NeMo's transcribe_speech.py (lines 308-348).
    Applies optimizations like CUDA graphs for RNNT models.

    Args:
        asr_model: Loaded ASR model
        cfg: Decoding configuration
    """
    if not hasattr(asr_model, 'change_decoding_strategy'):
        logging.debug("Model does not support decoding strategy changes")
        return

    # Check model type - RNNT/TDT models have 'joint' module
    if hasattr(asr_model, 'joint'):
        # RNNT/TDT model
        try:
            from nemo.collections.asr.parts.submodules.rnnt_decoding import RNNTDecodingConfig

            # Build decoding config following transcribe_speech.py pattern
            decoding_cfg = RNNTDecodingConfig(
                fused_batch_size=cfg.rnnt_fused_batch_size,
                rnnt_timestamp_type=cfg.rnnt_timestamp_type,
            )

            # Set compute_timestamps if specified
            if cfg.compute_timestamps is not None:
                decoding_cfg.compute_timestamps = cfg.compute_timestamps

            asr_model.change_decoding_strategy(decoding_cfg)

            cuda_graphs_status = "enabled" if cfg.rnnt_fused_batch_size == -1 else "disabled"
            logging.info(
                f"Applied RNNT decoding config: "
                f"fused_batch_size={cfg.rnnt_fused_batch_size} (CUDA graphs {cuda_graphs_status}), "
                f"timestamp_type={cfg.rnnt_timestamp_type}"
            )

        except Exception as e:
            logging.warning(f"Failed to apply RNNT decoding config: {e}")

    else:
        # CTC model
        try:
            from nemo.collections.asr.parts.submodules.ctc_decoding import CTCDecodingConfig

            decoding_cfg = CTCDecodingConfig(
                ctc_timestamp_type=cfg.ctc_timestamp_type,
            )

            if cfg.compute_timestamps is not None:
                decoding_cfg.compute_timestamps = cfg.compute_timestamps

            asr_model.change_decoding_strategy(decoding_cfg)

            logging.info(
                f"Applied CTC decoding config: timestamp_type={cfg.ctc_timestamp_type}"
            )

        except Exception as e:
            logging.warning(f"Failed to apply CTC decoding config: {e}")


# =============================================================================
# Logging Utilities
# =============================================================================


class NeMoLogFilter:
    """
    Custom log filter to suppress repetitive NeMo internal logs.

    Filters out messages that are emitted on every transcribe() call:
    - RNNT Loss configuration logs
    - Joint fused batch size warnings
    - Timestamp decoding messages
    """

    # Patterns to filter (substrings that identify repetitive logs)
    FILTER_PATTERNS = [
        # RNNT/TDT decoder initialization logs
        "Using RNNT Loss",
        "Joint fused batch size",
        "Will temporarily disable fused batch step",
        "Timestamps requested, setting decoding timestamps",
        "Loss tdt_kwargs",
        # Lhotse dataloader warnings (repeated per chunk)
        "The following configuration keys are ignored by Lhotse dataloader",
        "You are using a non-tarred dataset and requested tokenization",
        "pretokenize=False in dataloader config",
    ]

    def __init__(self):
        self.enabled = False

    def filter(self, record) -> bool:
        """Return False to suppress the log record, True to allow it."""
        if not self.enabled:
            return True

        message = record.getMessage()
        for pattern in self.FILTER_PATTERNS:
            if pattern in message:
                return False
        return True


# Global filter instance for reuse
_nemo_log_filter = NeMoLogFilter()


class suppress_repetitive_nemo_logs:
    """
    Context manager to temporarily suppress repetitive NeMo logs.

    Usage:
        with suppress_repetitive_nemo_logs():
            # NeMo operations here will have filtered logs
            model.transcribe(...)

    This reduces log noise during chunk-by-chunk transcription where
    the same initialization messages would be printed 40+ times.
    """

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.loggers_modified = []

    def __enter__(self):
        if not self.enabled:
            return self

        _nemo_log_filter.enabled = True

        # Add filter to relevant NeMo loggers
        import logging as std_logging

        # NeMo uses these logger names
        logger_names = [
            "nemo_logger",
            "nemo.collections.asr",
            "nemo.collections.asr.models",
            "nemo.collections.asr.parts.submodules.rnnt_decoding",
            "nemo.collections.asr.parts.submodules.rnnt_greedy_decoding",
        ]

        for name in logger_names:
            logger = std_logging.getLogger(name)
            if _nemo_log_filter not in logger.filters:
                logger.addFilter(_nemo_log_filter)
                self.loggers_modified.append(logger)

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self.enabled:
            return

        _nemo_log_filter.enabled = False

        # Remove filter from loggers
        for logger in self.loggers_modified:
            if _nemo_log_filter in logger.filters:
                logger.removeFilter(_nemo_log_filter)

        self.loggers_modified.clear()
        return False


# =============================================================================
# SRT Formatting Utilities
# =============================================================================


def format_srt_timestamp(seconds: float) -> str:
    """
    Convert seconds to SRT timestamp format (HH:MM:SS,mmm).

    Args:
        seconds: Time in seconds

    Returns:
        Formatted timestamp string
    """
    if seconds < 0:
        seconds = 0

    td = timedelta(seconds=seconds)
    total_seconds = int(td.total_seconds())
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    milliseconds = int((seconds - total_seconds) * 1000)

    return f"{hours:02d}:{minutes:02d}:{secs:02d},{milliseconds:03d}"


def hypothesis_to_srt_segments(
    hypothesis: Hypothesis,
    max_chars_per_line: int = 42,
    max_segment_duration: float = 5.0,
    word_gap_threshold: Optional[float] = 0.8,
) -> List[Tuple[float, float, str]]:
    """
    Convert NeMo Hypothesis to SRT segments.

    Prioritizes timestamp sources:
    1. segment-level timestamps (if available)
    2. word-level timestamps (grouped by line length/duration/gaps)
    3. Fallback: proportional allocation based on text length

    Args:
        hypothesis: NeMo transcription Hypothesis object
        max_chars_per_line: Maximum characters per subtitle line
        max_segment_duration: Maximum duration for a subtitle segment
        word_gap_threshold: Create new segment if gap between words exceeds this

    Returns:
        List of (start_time, end_time, text) tuples
    """
    segments = []

    # Get timestamp data from hypothesis
    timestamps = getattr(hypothesis, "timestep", None) or getattr(hypothesis, "timestamp", None) or {}

    # Check if segments are too coarse (e.g., models without punctuation produce one segment per chunk)
    # In that case, prefer word-level timestamps for finer granularity
    use_word_level = False
    if "segment" in timestamps and "word" in timestamps:
        seg_list = timestamps["segment"]
        if seg_list:
            avg_seg_duration = sum(s["end"] - s["start"] for s in seg_list) / len(seg_list)
            # If average segment is too long, use word-level instead
            if avg_seg_duration > max_segment_duration * 2:
                use_word_level = True

    # Strategy 1: Use segment-level timestamps if available and not too coarse
    if "segment" in timestamps and not use_word_level:
        for seg in timestamps["segment"]:
            text = seg.get("segment", seg.get("text", "")).strip()
            if text:
                start = seg["start"]
                end = seg["end"]
                segments.append((start, end, text))
        return segments

    # Strategy 2: Use word-level timestamps
    if "word" in timestamps:
        words_data = timestamps["word"]
        current_words = []
        current_start = None
        current_end = None

        for word_info in words_data:
            word = word_info.get("word", word_info.get("text", "")).strip()
            if not word:
                continue

            start = word_info["start"]
            end = word_info["end"]

            # Check for gap-based split
            if (
                current_words
                and current_end is not None
                and word_gap_threshold is not None
                and (start - current_end) >= word_gap_threshold
            ):
                text = " ".join(current_words)
                if text:
                    segments.append((current_start, current_end, text))
                current_words = []
                current_start = None
                current_end = None

            if current_start is None:
                current_start = start

            current_words.append(word)
            current_end = end

            # Check for length/duration-based split
            current_text = " ".join(current_words)
            current_duration = current_end - current_start

            if len(current_text) >= max_chars_per_line or current_duration >= max_segment_duration:
                if current_text:
                    segments.append((current_start, current_end, current_text))
                current_words = []
                current_start = None
                current_end = None

        # Flush remaining words
        if current_words:
            text = " ".join(current_words)
            if text:
                segments.append((current_start, current_end, text))

        return segments

    # Strategy 3: Fallback - proportional allocation
    text = getattr(hypothesis, "text", "") or ""
    text = text.strip()
    if not text:
        return segments

    words = text.split()
    if not words:
        return segments

    # Estimate duration based on typical speech rate (~150 words/min = 2.5 words/sec)
    estimated_duration = len(words) / 2.5
    time_per_word = estimated_duration / len(words)

    current_words = []
    current_start = 0.0

    for idx, word in enumerate(words):
        current_words.append(word)
        current_text = " ".join(current_words)
        word_end_time = (idx + 1) * time_per_word

        if len(current_text) >= max_chars_per_line:
            segments.append((
                current_start,
                word_end_time,
                current_text,
            ))
            current_words = []
            current_start = word_end_time

    # Flush remaining
    if current_words:
        text = " ".join(current_words)
        segments.append((
            current_start,
            estimated_duration,
            text,
        ))

    return segments


def write_srt_file(
    segments: List[Tuple[float, float, str]],
    output_path: str,
) -> None:
    """
    Write segments to SRT file.

    Args:
        segments: List of (start_time, end_time, text) tuples
        output_path: Output file path
    """
    with open(output_path, "w", encoding="utf-8") as f:
        for idx, (start, end, text) in enumerate(segments, start=1):
            f.write(f"{idx}\n")
            f.write(f"{format_srt_timestamp(start)} --> {format_srt_timestamp(end)}\n")
            f.write(f"{text}\n")
            f.write("\n")


def clip_segments_to_window(
    segments: List[Tuple[float, float, str]],
    window_start: float,
    window_end: float,
    tolerance: float = 0.15,
) -> List[Tuple[float, float, str]]:
    """
    Clip segments to fit within a time window.

    Used when processing overlapping audio chunks to avoid duplicate subtitles.

    Args:
        segments: List of (start, end, text) tuples
        window_start: Window start time (in original video time)
        window_end: Window end time (in original video time)
        tolerance: Tolerance for boundary clipping (seconds)

    Returns:
        Filtered and clipped segments
    """
    clipped = []
    for start, end, text in segments:
        # Skip segments entirely outside window
        if end < window_start - tolerance or start > window_end + tolerance:
            continue

        # Clip to window boundaries
        clipped_start = max(start, window_start)
        clipped_end = min(end, window_end)

        if clipped_end > clipped_start and text.strip():
            clipped.append((clipped_start, clipped_end, text))

    return clipped


def merge_overlapping_segments(
    all_segments: List[Tuple[float, float, str]],
    overlap_threshold: float = 0.1,
    merge_strategy: str = "prefer_longer",
) -> List[Tuple[float, float, str]]:
    """
    Merge segments from different chunks that overlap in time.

    This function handles the overlap regions between adjacent chunks by:
    1. Detecting segments that overlap in time
    2. Choosing the best segment or merging them based on strategy
    3. Removing duplicate content from chunk boundaries

    Based on NeMo's vad_utils.py merge_overlap_segment() pattern.

    Args:
        all_segments: List of (start, end, text) tuples from all chunks
        overlap_threshold: Time threshold (seconds) to consider segments as overlapping
        merge_strategy: How to handle overlaps:
            - "prefer_longer": Keep the segment with longer text (default)
            - "prefer_earlier": Keep the earlier segment
            - "merge_text": Combine text from both segments

    Returns:
        List of merged (start, end, text) tuples
    """
    if not all_segments:
        return []

    # Sort by start time
    sorted_segments = sorted(all_segments, key=lambda x: x[0])

    merged = []
    i = 0

    while i < len(sorted_segments):
        current_start, current_end, current_text = sorted_segments[i]

        # Look ahead for overlapping segments
        j = i + 1
        overlapping = [(current_start, current_end, current_text)]

        while j < len(sorted_segments):
            next_start, next_end, next_text = sorted_segments[j]

            # Check for overlap
            if next_start < current_end + overlap_threshold:
                overlapping.append((next_start, next_end, next_text))
                # Extend current_end to track the overlap region
                current_end = max(current_end, next_end)
                j += 1
            else:
                break

        if len(overlapping) == 1:
            # No overlap, keep as is
            merged.append(overlapping[0])
        else:
            # Handle overlapping segments
            if merge_strategy == "prefer_longer":
                # Keep the segment with the longest text
                best = max(overlapping, key=lambda x: len(x[2]))
                # Extend to cover the full range
                full_start = min(s[0] for s in overlapping)
                full_end = max(s[1] for s in overlapping)
                merged.append((full_start, full_end, best[2]))

            elif merge_strategy == "prefer_earlier":
                # Keep the first segment but extend its duration
                first = overlapping[0]
                full_end = max(s[1] for s in overlapping)
                merged.append((first[0], full_end, first[2]))

            elif merge_strategy == "merge_text":
                # Combine text from all overlapping segments
                full_start = min(s[0] for s in overlapping)
                full_end = max(s[1] for s in overlapping)
                # Concatenate unique words
                all_words = []
                for _, _, text in overlapping:
                    words = text.split()
                    for word in words:
                        if not all_words or word != all_words[-1]:
                            all_words.append(word)
                merged_text = " ".join(all_words)
                merged.append((full_start, full_end, merged_text))

            else:
                # Fallback to prefer_longer
                best = max(overlapping, key=lambda x: len(x[2]))
                merged.append(best)

        i = j

    return merged


def deduplicate_segments(
    segments: List[Tuple[float, float, str]],
    similarity_threshold: float = 0.8,
) -> List[Tuple[float, float, str]]:
    """
    Remove duplicate or near-duplicate segments based on text similarity.

    This is useful after merging overlapping segments to clean up any
    remaining duplicates caused by chunk boundaries.

    Args:
        segments: List of (start, end, text) tuples
        similarity_threshold: Jaccard similarity threshold (0-1) to consider duplicates

    Returns:
        Deduplicated list of segments
    """
    if not segments:
        return []

    def jaccard_similarity(text1: str, text2: str) -> float:
        """Calculate Jaccard similarity between two texts."""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        if not words1 or not words2:
            return 0.0
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        return intersection / union if union > 0 else 0.0

    # Sort by start time
    sorted_segments = sorted(segments, key=lambda x: x[0])
    deduplicated = [sorted_segments[0]]

    for i in range(1, len(sorted_segments)):
        current = sorted_segments[i]
        prev = deduplicated[-1]

        # Check if current segment is similar to previous
        similarity = jaccard_similarity(current[2], prev[2])

        if similarity >= similarity_threshold:
            # Skip duplicate, but extend previous segment's end time if needed
            if current[1] > prev[1]:
                deduplicated[-1] = (prev[0], current[1], prev[2])
        else:
            deduplicated.append(current)

    return deduplicated


# =============================================================================
# Main Transcription Logic
# =============================================================================


def load_asr_model(
    pretrained_name: str,
    model_path: Optional[str],
    device: torch.device,
    compute_dtype: torch.dtype,
) -> Tuple[ASRModel, str]:
    """
    Load ASR model from pretrained name or checkpoint.

    Args:
        pretrained_name: HuggingFace/NGC model name
        model_path: Path to .nemo checkpoint (overrides pretrained_name)
        device: Target device
        compute_dtype: Computation dtype

    Returns:
        Tuple of (model, model_name)
    """
    if model_path is not None:
        logging.info(f"Loading model from checkpoint: {model_path}")
        asr_model = ASRModel.restore_from(model_path, map_location="cpu")
        model_name = Path(model_path).stem
    else:
        logging.info(f"Loading pretrained model: {pretrained_name}")
        asr_model = ASRModel.from_pretrained(pretrained_name, map_location="cpu")
        model_name = pretrained_name

    asr_model = asr_model.to(device)
    asr_model = asr_model.to(compute_dtype)
    asr_model.eval()

    return asr_model, model_name


def apply_long_audio_settings(model: ASRModel) -> bool:
    """
    Apply optimized settings for long audio transcription.

    Based on HuggingFace Space app.py:
    - Switch to local attention for memory efficiency
    - Enable subsampling conv chunking

    Args:
        model: ASR model

    Returns:
        True if settings were applied successfully
    """
    try:
        if hasattr(model, "change_attention_model"):
            model.change_attention_model("rel_pos_local_attn", [256, 256])
            logging.info("Applied local attention model for long audio")

        if hasattr(model, "change_subsampling_conv_chunking_factor"):
            model.change_subsampling_conv_chunking_factor(1)  # 1 = auto select
            logging.info("Applied subsampling conv chunking for long audio")

        return True
    except Exception as e:
        logging.warning(f"Could not apply long audio settings: {e}")
        return False


def revert_long_audio_settings(model: ASRModel) -> None:
    """
    Revert to default attention settings after long audio transcription.

    Args:
        model: ASR model
    """
    try:
        if hasattr(model, "change_attention_model"):
            model.change_attention_model("rel_pos")

        if hasattr(model, "change_subsampling_conv_chunking_factor"):
            model.change_subsampling_conv_chunking_factor(-1)  # -1 = disable

        logging.info("Reverted long audio settings")
    except Exception as e:
        logging.warning(f"Failed to revert long audio settings: {e}")


def transcribe_audio_chunk(
    audio_path: str,
    asr_model: ASRModel,
    cfg: VideoToSRTConfig,
    time_offset: float = 0.0,
    suppress_logs: bool = False,
) -> List[Tuple[float, float, str]]:
    """
    Transcribe a single audio chunk and return segments with time offset applied.

    Args:
        audio_path: Path to audio file
        asr_model: Loaded ASR model
        cfg: Configuration
        time_offset: Time offset to add to all timestamps
        suppress_logs: If True, suppress repetitive NeMo logs during transcription

    Returns:
        List of (start_time, end_time, text) tuples with offset applied
    """
    with suppress_repetitive_nemo_logs(enabled=suppress_logs):
        with torch.inference_mode():
            transcriptions = asr_model.transcribe(
                [audio_path],
                batch_size=1,
                timestamps=True,
                return_hypotheses=True,
            )

    # Handle different return formats
    if isinstance(transcriptions, tuple):
        transcriptions = transcriptions[0]

    if isinstance(transcriptions, list) and len(transcriptions) > 0:
        hypothesis = transcriptions[0]
    else:
        hypothesis = transcriptions

    # Convert to SRT segments
    segments = hypothesis_to_srt_segments(
        hypothesis,
        max_chars_per_line=cfg.subtitle.max_chars_per_line,
        max_segment_duration=cfg.subtitle.max_segment_duration,
        word_gap_threshold=cfg.subtitle.word_gap_threshold,
    )

    # Apply time offset
    if time_offset != 0.0:
        segments = [(start + time_offset, end + time_offset, text) for start, end, text in segments]

    return segments


def transcribe_video(
    video_path: str,
    output_path: str,
    asr_model: ASRModel,
    cfg: VideoToSRTConfig,
    device: torch.device,
    vad_model=None,
    itn_normalizer=None,
) -> str:
    """
    Transcribe a single video file to SRT.

    Args:
        video_path: Input video file path
        output_path: Output SRT file path
        asr_model: Loaded ASR model
        cfg: Configuration
        device: Computation device
        vad_model: Optional loaded VAD model for speech segment detection
        itn_normalizer: Optional ITN normalizer for text post-processing

    Returns:
        Path to generated SRT file
    """
    import time

    logging.info(f"Processing: {video_path}")

    # Create temporary directory for audio files
    tmp_dir = tempfile.mkdtemp(prefix="nemoscribe_")
    long_audio_applied = False

    # RTFx measurement variables
    transcription_start_time = None
    transcription_times = []

    try:
        # Get video duration
        video_duration = get_media_duration(video_path)
        logging.info(f"Video duration: {video_duration:.1f} seconds ({video_duration/60:.1f} minutes)")

        # Determine if we need chunking
        max_chunk = cfg.audio.max_chunk_duration
        need_chunking = max_chunk > 0 and video_duration > max_chunk

        # Run VAD if enabled to detect speech segments
        speech_segments = None
        if cfg.vad.enabled and vad_model is not None:
            logging.info("Running Voice Activity Detection...")
            # Extract full audio for VAD analysis
            vad_audio_path = os.path.join(tmp_dir, "vad_audio.wav")
            if extract_audio(video_path, vad_audio_path, cfg.audio.sample_rate):
                speech_segments = run_vad_on_audio(
                    vad_audio_path,
                    vad_model,
                    cfg.vad,
                    device,
                )
                if speech_segments:
                    total_speech = sum(end - start for start, end in speech_segments)
                    logging.info(
                        f"VAD detected {len(speech_segments)} speech segments, "
                        f"total speech: {total_speech:.1f}s ({total_speech/video_duration*100:.1f}% of video)"
                    )
                else:
                    logging.warning("VAD detected no speech segments")
            else:
                logging.warning("Failed to extract audio for VAD, falling back to fixed chunking")

        if need_chunking:
            # Process in chunks for limited GPU memory
            num_chunks = int((video_duration + max_chunk - 1) // max_chunk)
            logging.info(f"Splitting into ~{num_chunks} chunks (max {max_chunk}s each) for GPU memory efficiency")

            # Use VAD-aware chunking if speech segments are available
            if speech_segments is not None:
                smart_mode = "smart" if cfg.audio.smart_segmentation else "basic"
                logging.info(f"Using VAD-aware chunking ({smart_mode} segmentation, splitting at silence boundaries)")
                chunks = create_audio_chunks_with_vad(
                    video_path,
                    tmp_dir,
                    video_duration,
                    speech_segments,
                    audio_cfg=cfg.audio,
                )
            else:
                # Fall back to fixed-duration chunking
                chunks = create_audio_chunks(
                    video_path,
                    tmp_dir,
                    video_duration,
                    sample_rate=cfg.audio.sample_rate,
                    max_chunk_duration=max_chunk,
                    overlap=cfg.audio.chunk_overlap,
                )

            if not chunks:
                raise RuntimeError(f"Failed to create audio chunks from {video_path}")

            # Apply long audio optimizations for each chunk if needed
            chunk_duration = max_chunk
            if chunk_duration > cfg.audio.long_audio_threshold:
                logging.info(f"Chunk duration > {cfg.audio.long_audio_threshold}s, applying long audio optimizations...")
                long_audio_applied = apply_long_audio_settings(asr_model)

            # Transcribe each chunk
            all_segments = []

            # Warmup for RTFx measurement (run first chunk without timing if requested)
            warmup_done = False
            if cfg.performance.calculate_rtfx and cfg.performance.warmup_steps > 0 and len(chunks) > 0:
                logging.info("Running warmup step for RTFx measurement...")
                warmup_audio, warmup_start, warmup_end, warmup_extract = chunks[0]
                _ = transcribe_audio_chunk(
                    warmup_audio, asr_model, cfg, time_offset=warmup_extract,
                    suppress_logs=cfg.logging.suppress_repetitive_logs
                )
                warmup_done = True
                # Clear GPU caches after warmup
                if device.type == "cuda":
                    torch.cuda.synchronize()
                    gc.collect()
                    torch.cuda.empty_cache()

            # Start timing for RTFx (after warmup)
            if cfg.performance.calculate_rtfx:
                if device.type == "cuda":
                    torch.cuda.synchronize()
                transcription_start_time = time.perf_counter()

            for chunk_idx, (audio_path, window_start, window_end, extract_start) in enumerate(chunks):
                logging.info(f"Transcribing chunk {chunk_idx + 1}/{len(chunks)} "
                           f"({window_start:.1f}s - {window_end:.1f}s)")

                # Time offset: timestamps in audio file are relative to extract_start
                # We need to convert them to original video time
                time_offset = extract_start

                # Suppress repetitive logs after first chunk (first chunk shows full logs for debugging)
                suppress_logs = cfg.logging.suppress_repetitive_logs and chunk_idx > 0

                chunk_segments = transcribe_audio_chunk(
                    audio_path,
                    asr_model,
                    cfg,
                    time_offset=time_offset,
                    suppress_logs=suppress_logs,
                )

                # Clip segments to the non-overlapping window
                chunk_segments = clip_segments_to_window(
                    chunk_segments,
                    window_start,
                    window_end,
                    tolerance=0.15,
                )

                all_segments.extend(chunk_segments)

                # Clear GPU memory between chunks
                if device.type == "cuda":
                    gc.collect()
                    torch.cuda.empty_cache()

            segments = all_segments

            # Merge overlapping segments from chunk boundaries
            original_count = len(segments)
            segments = merge_overlapping_segments(
                segments,
                overlap_threshold=0.1,
                merge_strategy="prefer_longer",
            )

            # Deduplicate similar segments (may occur at chunk boundaries)
            segments = deduplicate_segments(segments, similarity_threshold=0.8)

            if len(segments) < original_count:
                logging.debug(
                    f"Segment cleanup: {original_count} -> {len(segments)} "
                    f"(merged {original_count - len(segments)} overlapping/duplicate segments)"
                )

            # Stop timing for RTFx (chunked)
            if cfg.performance.calculate_rtfx and transcription_start_time is not None:
                if device.type == "cuda":
                    torch.cuda.synchronize()
                elapsed = time.perf_counter() - transcription_start_time
                transcription_times.append(elapsed)

        else:
            # Single chunk processing
            audio_path = os.path.join(tmp_dir, "audio.wav")
            logging.info("Extracting audio...")
            if not extract_audio(video_path, audio_path, cfg.audio.sample_rate):
                raise RuntimeError(f"Failed to extract audio from {video_path}")

            # Apply long audio optimizations if needed
            if video_duration > cfg.audio.long_audio_threshold:
                logging.info(f"Audio longer than {cfg.audio.long_audio_threshold}s, applying long audio optimizations...")
                long_audio_applied = apply_long_audio_settings(asr_model)

            # Warmup for RTFx measurement
            if cfg.performance.calculate_rtfx and cfg.performance.warmup_steps > 0:
                logging.info("Running warmup step for RTFx measurement...")
                _ = transcribe_audio_chunk(
                    audio_path, asr_model, cfg, time_offset=0.0,
                    suppress_logs=cfg.logging.suppress_repetitive_logs
                )
                if device.type == "cuda":
                    torch.cuda.synchronize()
                    gc.collect()
                    torch.cuda.empty_cache()

            # Start timing for RTFx
            if cfg.performance.calculate_rtfx:
                if device.type == "cuda":
                    torch.cuda.synchronize()
                transcription_start_time = time.perf_counter()

            # Transcribe
            logging.info("Transcribing audio...")
            segments = transcribe_audio_chunk(audio_path, asr_model, cfg, time_offset=0.0)

            # Stop timing for RTFx (single)
            if cfg.performance.calculate_rtfx and transcription_start_time is not None:
                if device.type == "cuda":
                    torch.cuda.synchronize()
                elapsed = time.perf_counter() - transcription_start_time
                transcription_times.append(elapsed)

        # Sort segments by start time
        segments.sort(key=lambda x: x[0])

        # Apply ITN post-processing if enabled
        if itn_normalizer is not None:
            logging.info("Applying Inverse Text Normalization (ITN)...")
            segments = apply_itn_to_segments(segments, itn_normalizer)

        # Write SRT file
        write_srt_file(segments, output_path)
        logging.info(f"Generated SRT: {output_path} ({len(segments)} subtitles)")

        # Report RTFx if measured
        if cfg.performance.calculate_rtfx and transcription_times and video_duration > 0:
            total_transcription_time = sum(transcription_times)
            rtfx = video_duration / total_transcription_time
            logging.info(
                f"Performance: RTFx={rtfx:.2f}x realtime "
                f"(transcribed {video_duration:.1f}s in {total_transcription_time:.2f}s)"
            )

        return output_path

    finally:
        # Revert long audio settings
        if long_audio_applied:
            revert_long_audio_settings(asr_model)

        # Cleanup temporary files
        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir)

        # Clear GPU memory
        if device.type == "cuda":
            gc.collect()
            torch.cuda.empty_cache()


def process_videos(cfg: VideoToSRTConfig) -> List[str]:
    """
    Process video(s) based on configuration.

    Args:
        cfg: Configuration

    Returns:
        List of generated SRT file paths
    """
    # Validate inputs
    if cfg.video_path is None and cfg.video_dir is None:
        raise ValueError("Either video_path or video_dir must be specified")

    if not check_ffmpeg():
        raise RuntimeError("ffmpeg is required but not found. Please install ffmpeg.")

    # Setup device
    torch.set_float32_matmul_precision(cfg.matmul_precision)
    device = get_inference_device(cuda=cfg.cuda, allow_mps=cfg.allow_mps)
    logging.info(f"Using device: {device}")

    # Setup dtype
    compute_dtype = get_inference_dtype(compute_dtype=cfg.compute_dtype, device=device)
    logging.info(f"Using dtype: {compute_dtype}")

    # Load model
    asr_model, model_name = load_asr_model(
        cfg.pretrained_name,
        cfg.model_path,
        device,
        compute_dtype,
    )
    logging.info(f"Model loaded: {model_name}")

    # Setup decoding strategy (CUDA graphs, timestamp types)
    setup_decoding_strategy(asr_model, cfg.decoding)

    # Load VAD model if enabled
    vad_model = None
    if cfg.vad.enabled:
        try:
            vad_model = load_vad_model(cfg.vad.model, device)
            logging.info(f"VAD model loaded: {cfg.vad.model}")
        except Exception as e:
            logging.warning(f"Failed to load VAD model: {e}. Continuing without VAD.")
            cfg.vad.enabled = False

    # Initialize ITN normalizer if enabled
    itn_normalizer = None
    if cfg.postprocessing.enable_itn:
        itn_normalizer = get_itn_normalizer(
            lang=cfg.postprocessing.itn_lang,
            input_case=cfg.postprocessing.itn_input_case,
        )
        if itn_normalizer is None:
            logging.warning("ITN requested but unavailable. Continuing without ITN.")
        else:
            logging.info(f"ITN normalizer ready (lang={cfg.postprocessing.itn_lang})")

    # Collect video files
    video_files = []

    if cfg.video_path is not None:
        video_path = Path(cfg.video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
        video_files.append(video_path)
    else:
        video_dir = Path(cfg.video_dir)
        if not video_dir.is_dir():
            raise NotADirectoryError(f"Not a directory: {video_dir}")

        for ext in cfg.video_extensions:
            video_files.extend(video_dir.glob(f"*{ext}"))
            video_files.extend(video_dir.glob(f"*{ext.upper()}"))

        video_files = sorted(set(video_files))

    if not video_files:
        logging.warning("No video files found")
        return []

    logging.info(f"Found {len(video_files)} video file(s)")

    # Determine output paths
    output_dir = None
    if cfg.output_dir is not None:
        output_dir = Path(cfg.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    # Process each video
    generated_files = []

    for video_path in video_files:
        # Determine output path
        if cfg.output_path is not None and len(video_files) == 1:
            srt_path = Path(cfg.output_path)
        elif output_dir is not None:
            srt_path = output_dir / video_path.with_suffix(".srt").name
        else:
            srt_path = video_path.with_suffix(".srt")

        # Skip if exists and not overwriting
        if srt_path.exists() and not cfg.overwrite:
            logging.info(f"Skipping (exists): {srt_path}")
            continue

        try:
            result = transcribe_video(
                str(video_path),
                str(srt_path),
                asr_model,
                cfg,
                device,
                vad_model=vad_model,
                itn_normalizer=itn_normalizer,
            )
            generated_files.append(result)
        except Exception as e:
            logging.error(f"Failed to process {video_path}: {e}")
            import traceback
            traceback.print_exc()

    return generated_files


# =============================================================================
# Entry Point
# =============================================================================


def main() -> int:
    """
    NemoScribe entry point.

    Parses command-line arguments in Hydra style (key=value).
    """
    # Parse command line arguments manually (Hydra-style)
    args = sys.argv[1:]

    # Build config from defaults
    cfg = VideoToSRTConfig()

    # Parse overrides
    for arg in args:
        if "=" not in arg:
            if arg in ("--help", "-h"):
                print(__doc__)
                print("\nConfiguration options:")
                print(OmegaConf.to_yaml(OmegaConf.structured(cfg)))
                sys.exit(0)
            continue

        key, value = arg.split("=", 1)

        # Handle nested keys (e.g., subtitle.max_chars_per_line)
        if "." in key:
            parts = key.split(".")
            if len(parts) == 2:
                parent, child = parts
                if hasattr(cfg, parent):
                    parent_obj = getattr(cfg, parent)
                    if hasattr(parent_obj, child):
                        # Convert value to appropriate type
                        current_value = getattr(parent_obj, child)
                        if current_value is None or isinstance(current_value, str):
                            if value.lower() == "null" or value.lower() == "none":
                                setattr(parent_obj, child, None)
                            else:
                                setattr(parent_obj, child, value)
                        elif isinstance(current_value, bool):
                            setattr(parent_obj, child, value.lower() in ("true", "1", "yes"))
                        elif isinstance(current_value, int):
                            setattr(parent_obj, child, int(value))
                        elif isinstance(current_value, float):
                            setattr(parent_obj, child, float(value))
        else:
            if hasattr(cfg, key):
                current_value = getattr(cfg, key)
                if current_value is None or isinstance(current_value, str):
                    if value.lower() == "null" or value.lower() == "none":
                        setattr(cfg, key, None)
                    else:
                        setattr(cfg, key, value)
                elif isinstance(current_value, bool):
                    setattr(cfg, key, value.lower() in ("true", "1", "yes"))
                elif isinstance(current_value, int):
                    setattr(cfg, key, int(value))
                elif isinstance(current_value, float):
                    setattr(cfg, key, float(value))
                elif isinstance(current_value, list):
                    setattr(cfg, key, value.split(","))

    logging.info(f"Configuration:\n{OmegaConf.to_yaml(OmegaConf.structured(cfg))}")

    generated = process_videos(cfg)
    if generated:
        print("\nGenerated files:")
        for f in generated:
            print(f)
    return 0


if __name__ == "__main__":
    main()
