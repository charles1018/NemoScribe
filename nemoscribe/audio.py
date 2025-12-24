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
Audio processing utilities for NemoScribe.

This module handles all ffmpeg operations for audio extraction and splitting.
"""

import os
import subprocess
from typing import List, Optional, Tuple

from nemo.utils import logging


def _check_binary(binary: str) -> bool:
    try:
        subprocess.run([binary, "-version"], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def check_ffmpeg() -> bool:
    """Check if ffmpeg and ffprobe are available."""
    return _check_binary("ffmpeg") and _check_binary("ffprobe")


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
) -> List[Tuple[str, float, float, float]]:
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
        List of (audio_path, chunk_start_time, chunk_end_time, extract_start_time) tuples
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
            # Store: audio_path, original video start time, original video end time, extract start
            chunks.append((audio_path, current_start, chunk_end, extract_start))
            chunk_idx += 1
        else:
            raise RuntimeError(f"Failed to extract chunk {chunk_idx} from {video_path}")

        current_start = chunk_end

    return chunks
