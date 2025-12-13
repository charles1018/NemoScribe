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
Logging utilities for NemoScribe.

This module provides log filtering to suppress repetitive NeMo internal logs.

Note: Named log_utils.py to avoid conflict with Python's built-in logging module.
"""

import logging as std_logging


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
