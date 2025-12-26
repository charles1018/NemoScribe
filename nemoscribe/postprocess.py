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
Post-processing utilities for NemoScribe.

This module handles:
- Inverse Text Normalization (ITN)
- Segment merging and deduplication
"""

from typing import Any, Dict, List, Optional, Tuple

from nemo.utils import logging


# Type alias for ITN normalizer (InverseNormalizer from nemo_text_processing)
ITNNormalizer = Any

# =============================================================================
# ITN (Inverse Text Normalization) Utilities
# =============================================================================

# Global cache for ITN normalizer to avoid reloading
_ITN_NORMALIZER_CACHE: Dict[str, Optional[ITNNormalizer]] = {}


def get_itn_normalizer(
    lang: str = "en",
    input_case: str = "lower_cased",
) -> Optional[ITNNormalizer]:
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


def apply_itn(text: str, normalizer: Optional[ITNNormalizer]) -> str:
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
    normalizer: Optional[ITNNormalizer],
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
# Segment Merging and Deduplication
# =============================================================================


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
