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
SRT subtitle formatting utilities for NemoScribe.

This module handles SRT file generation and timestamp formatting.
"""

from datetime import timedelta
from typing import Dict, List, Optional, Tuple

from nemo.collections.asr.parts.utils.rnnt_utils import Hypothesis


def validate_segment_gap_threshold(segment_gap_threshold_frames: Optional[int]) -> None:
    """
    Validate frame-based segment gap splitting configuration.

    Args:
        segment_gap_threshold_frames: Optional gap threshold in decoder frames

    Raises:
        ValueError: If the threshold is not a positive integer
    """
    if segment_gap_threshold_frames is not None and segment_gap_threshold_frames <= 0:
        raise ValueError("decoding.segment_gap_threshold must be a positive integer or None")


def _word_ends_segment(word: str, segment_separators: Optional[List[str]]) -> bool:
    """
    Check whether a decoded word should terminate the current segment.

    Args:
        word: Decoded word text
        segment_separators: Tokens that should terminate a segment

    Returns:
        True if the word ends a segment, False otherwise
    """
    if not segment_separators:
        return False
    return word in segment_separators or any(word.endswith(separator) for separator in segment_separators)


def _word_gap_exceeds_threshold(
    previous_word: Dict,
    current_word: Dict,
    segment_gap_threshold_frames: Optional[int],
) -> bool:
    """
    Check whether two words should be split based on decoder frame offsets.

    Args:
        previous_word: Previous decoded word timestamp entry
        current_word: Current decoded word timestamp entry
        segment_gap_threshold_frames: Gap threshold in decoder frames

    Returns:
        True when the inter-word gap exceeds the configured threshold
    """
    if segment_gap_threshold_frames is None:
        return False

    previous_end_offset = previous_word.get("end_offset")
    current_start_offset = current_word.get("start_offset")
    if previous_end_offset is None or current_start_offset is None:
        return False

    return (current_start_offset - previous_end_offset) >= segment_gap_threshold_frames


def _segments_from_word_timestamps(
    words_data: List[Dict],
    max_chars_per_line: int,
    max_segment_duration: float,
    word_gap_threshold: Optional[float],
    segment_separators: Optional[List[str]],
    segment_gap_threshold_frames: Optional[int],
) -> List[Tuple[float, float, str]]:
    """
    Convert word timestamps into subtitle segments.

    Supports punctuation-based splitting, frame-gap splitting, subtitle gap
    splitting, and the existing line length / duration guards.
    """
    validate_segment_gap_threshold(segment_gap_threshold_frames)

    segments: List[Tuple[float, float, str]] = []
    current_words: List[str] = []
    current_start: Optional[float] = None
    current_end: Optional[float] = None
    previous_word_info: Optional[Dict] = None
    previous_word_text: Optional[str] = None

    def flush_current_segment() -> None:
        nonlocal current_words, current_start, current_end
        if current_words and current_start is not None and current_end is not None:
            text = " ".join(current_words)
            if text:
                segments.append((current_start, current_end, text))
        current_words = []
        current_start = None
        current_end = None

    for word_info in words_data:
        word = word_info.get("word", word_info.get("text", "")).strip()
        if not word:
            continue

        start = word_info["start"]
        end = word_info["end"]

        should_split_before_word = False
        if current_words and previous_word_info is not None and current_end is not None:
            if word_gap_threshold is not None and (start - current_end) >= word_gap_threshold:
                should_split_before_word = True
            elif _word_gap_exceeds_threshold(previous_word_info, word_info, segment_gap_threshold_frames):
                should_split_before_word = True
            elif previous_word_text is not None and _word_ends_segment(previous_word_text, segment_separators):
                should_split_before_word = True

        if should_split_before_word:
            flush_current_segment()

        if current_start is None:
            current_start = start

        current_words.append(word)
        current_end = end

        current_text = " ".join(current_words)
        current_duration = current_end - current_start
        if len(current_text) >= max_chars_per_line or current_duration >= max_segment_duration:
            flush_current_segment()

        previous_word_info = word_info
        previous_word_text = word

    flush_current_segment()
    return segments


def parse_srt_timestamp(timestamp: str) -> float:
    """
    Convert SRT timestamp format (HH:MM:SS,mmm) to seconds.

    Args:
        timestamp: SRT timestamp string (e.g., "00:01:23,456")

    Returns:
        Time in seconds (float)

    Example:
        >>> parse_srt_timestamp("00:01:23,456")
        83.456
    """
    h, m, s = timestamp.split(":")
    s, ms = s.split(",")
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000


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
    segment_separators: Optional[List[str]] = None,
    segment_gap_threshold_frames: Optional[int] = None,
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
        segment_separators: Split segments after words ending in these tokens
        segment_gap_threshold_frames: Split segments when inter-word frame gaps exceed this

    Returns:
        List of (start_time, end_time, text) tuples
    """
    validate_segment_gap_threshold(segment_gap_threshold_frames)
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

    # NeMo's native segment builder treats frame-gap splitting and punctuation
    # splitting as mutually exclusive. When both are configured, rebuild segments
    # from word timestamps so both rules can stay active.
    if segment_gap_threshold_frames is not None and segment_separators and "word" in timestamps:
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
        return _segments_from_word_timestamps(
            words_data=timestamps["word"],
            max_chars_per_line=max_chars_per_line,
            max_segment_duration=max_segment_duration,
            word_gap_threshold=word_gap_threshold,
            segment_separators=segment_separators,
            segment_gap_threshold_frames=segment_gap_threshold_frames,
        )

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
