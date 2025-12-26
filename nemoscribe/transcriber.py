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
ASR transcription module for NemoScribe.

This module handles:
- ASR model loading and configuration
- Decoding strategy setup (CUDA graphs, timestamp types)
- Audio transcription with chunking support
- Long audio optimization settings
"""

import gc
import os
import tempfile
import time
from pathlib import Path
from typing import List, Optional, Tuple

import torch

from nemo.collections.asr.models import ASRModel, EncDecClassificationModel
from nemo.utils import logging

from nemoscribe.audio import create_audio_chunks, extract_audio, get_media_duration
from nemoscribe.config import DecodingConfig, VideoToSRTConfig
from nemoscribe.log_utils import suppress_repetitive_nemo_logs
from nemoscribe.postprocess import ITNNormalizer, apply_itn_to_segments, deduplicate_segments, merge_overlapping_segments
from nemoscribe.srt import clip_segments_to_window, hypothesis_to_srt_segments, write_srt_file
from nemoscribe.vad import create_audio_chunks_with_vad, run_vad_on_audio


# =============================================================================
# Decoding Strategy Configuration
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
# Model Loading
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


# =============================================================================
# Transcription Functions
# =============================================================================


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
    vad_model: Optional[EncDecClassificationModel] = None,
    itn_normalizer: Optional[ITNNormalizer] = None,
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
    logging.info(f"Processing: {video_path}")

    long_audio_applied = False

    # RTFx measurement variables
    transcription_start_time = None
    transcription_times = []

    # Use TemporaryDirectory context manager for guaranteed cleanup
    with tempfile.TemporaryDirectory(prefix="nemoscribe_") as tmp_dir:
        try:
            # Get video duration
            video_duration = get_media_duration(video_path)
            if video_duration <= 0:
                raise RuntimeError(
                    "Failed to determine video duration. Ensure ffprobe is available "
                    "and the input file is valid."
                )
            logging.info(
                f"Video duration: {video_duration:.1f} seconds ({video_duration/60:.1f} minutes)"
            )

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
                if total_transcription_time > 0:
                    rtfx = video_duration / total_transcription_time
                    logging.info(
                        f"Performance: RTFx={rtfx:.2f}x realtime "
                        f"(transcribed {video_duration:.1f}s in {total_transcription_time:.2f}s)"
                    )
                else:
                    logging.warning("RTFx calculation skipped: transcription time too small")

            return output_path

        finally:
            # Revert long audio settings
            if long_audio_applied:
                revert_long_audio_settings(asr_model)

            # Clear GPU memory
            if device.type == "cuda":
                gc.collect()
                torch.cuda.empty_cache()
