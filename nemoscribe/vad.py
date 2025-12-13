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
Voice Activity Detection (VAD) utilities for NemoScribe.

This module handles:
- VAD model loading
- Speech segment detection
- Smart audio splitting based on silence gaps
"""

import os
from typing import List, Tuple

import torch

from nemo.collections.asr.parts.utils.vad_utils import (
    generate_vad_segment_table_per_tensor,
    init_frame_vad_model,
)
from nemo.utils import logging

from nemoscribe.audio import create_audio_chunks, extract_audio
from nemoscribe.config import AudioConfig, VADConfig


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
