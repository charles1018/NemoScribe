"""
NemoScribe - Video to SRT Subtitle Generator

Convert video files to SRT subtitles using NVIDIA NeMo ASR models.
"""

__version__ = "0.1.0"
__author__ = "charles1018"

# Only export CLI entry point
from nemoscribe.cli import main

__all__ = ["main", "__version__"]
