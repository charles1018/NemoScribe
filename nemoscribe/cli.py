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
CLI entry point for NemoScribe.

This module handles:
- Command-line argument parsing (Hydra-style key=value)
- Video processing orchestration
- Model loading and initialization
"""

import sys
from pathlib import Path
from typing import Any, List, Union, get_args, get_origin, get_type_hints

import torch
from omegaconf import OmegaConf

from nemo.collections.asr.parts.utils.transcribe_utils import (
    get_inference_device,
    get_inference_dtype,
)
from nemo.utils import logging

from nemoscribe.audio import check_ffmpeg
from nemoscribe.config import VideoToSRTConfig
from nemoscribe.postprocess import get_itn_normalizer
from nemoscribe.transcriber import load_asr_model, setup_decoding_strategy, transcribe_video
from nemoscribe.vad import load_vad_model


# Module docstring for help output
_HELP_TEXT = """
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
"""


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
        raise RuntimeError(
            "ffmpeg/ffprobe are required but not found. Please install ffmpeg."
        )

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
            continue

    return generated_files


_NONE_STRINGS = {"null", "none"}


def _is_optional_type(target_type: Any) -> bool:
    origin = get_origin(target_type)
    if origin is None:
        return False
    args = get_args(target_type)
    return origin is Union and type(None) in args


def _unwrap_optional(target_type: Any) -> Any:
    if not _is_optional_type(target_type):
        return target_type
    return next(t for t in get_args(target_type) if t is not type(None))


def _coerce_value(value: str, target_type: Any) -> Any:
    value_lower = value.lower()
    if value_lower in _NONE_STRINGS and _is_optional_type(target_type):
        return None

    target_type = _unwrap_optional(target_type)
    origin = get_origin(target_type)
    args = get_args(target_type)

    if origin in (list, List):
        item_type = args[0] if args else str
        if value == "":
            return []
        return [_coerce_value(part, item_type) for part in value.split(",")]

    if target_type is bool:
        return value_lower in ("true", "1", "yes", "y", "on")
    if target_type is int:
        return int(value)
    if target_type is float:
        return float(value)
    if target_type is str:
        return value
    if target_type is Any:
        return value

    return value


def _set_typed_attr(obj: Any, attr_name: str, value: str) -> bool:
    hints = get_type_hints(type(obj))
    if attr_name not in hints:
        return False
    target_type = hints[attr_name]
    setattr(obj, attr_name, _coerce_value(value, target_type))
    return True


def parse_args(args: List[str], cfg: VideoToSRTConfig) -> VideoToSRTConfig:
    """
    Parse Hydra-style key=value arguments into configuration.

    Args:
        args: List of command-line arguments
        cfg: Base configuration to modify

    Returns:
        Modified configuration
    """
    for arg in args:
        if "=" not in arg:
            continue

        key, value = arg.split("=", 1)

        # Handle nested keys (e.g., subtitle.max_chars_per_line)
        if "." in key:
            parent, child = key.split(".", 1)
            if hasattr(cfg, parent):
                parent_obj = getattr(cfg, parent)
                if hasattr(parent_obj, child):
                    _set_typed_attr(parent_obj, child, value)
                else:
                    logging.warning(f"Unknown config key: '{key}' ('{child}' not found in {parent})")
            else:
                logging.warning(f"Unknown config key: '{key}' ('{parent}' not found in config)")
        else:
            if hasattr(cfg, key):
                _set_typed_attr(cfg, key, value)
            else:
                logging.warning(f"Unknown config key: '{key}'")

    return cfg


def main() -> int:
    """
    NemoScribe entry point.

    Parses command-line arguments in Hydra style (key=value).
    """
    # Parse command line arguments manually (Hydra-style)
    args = sys.argv[1:]

    # Build config from defaults
    cfg = VideoToSRTConfig()

    # Check for help flag
    for arg in args:
        if arg in ("--help", "-h"):
            print(_HELP_TEXT)
            print("\nConfiguration options:")
            print(OmegaConf.to_yaml(OmegaConf.structured(cfg)))
            return 0

    # Parse overrides
    cfg = parse_args(args, cfg)

    logging.info(f"Configuration:\n{OmegaConf.to_yaml(OmegaConf.structured(cfg))}")

    generated = process_videos(cfg)
    if generated:
        print("\nGenerated files:")
        for f in generated:
            print(f)
    return 0


if __name__ == "__main__":
    sys.exit(main())
