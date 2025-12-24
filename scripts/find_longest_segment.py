#!/usr/bin/env python3
"""Find the longest segment in SRT file."""

import re
from pathlib import Path


def parse_srt_timestamp(timestamp: str) -> float:
    """Convert SRT timestamp to seconds."""
    h, m, s = timestamp.split(":")
    s, ms = s.split(",")
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000


def find_longest_segment(srt_path: str) -> None:
    """Find and display the longest segment."""
    content = Path(srt_path).read_text(encoding="utf-8")

    # Split by double newline to get segments
    segments = content.strip().split("\n\n")

    longest_duration = 0
    longest_segment = None

    for segment in segments:
        lines = segment.strip().split("\n")
        if len(lines) < 3:
            continue

        # Line 0: number, Line 1: timestamp, Line 2+: text
        timestamp_line = lines[1]
        match = re.match(r"(\d{2}:\d{2}:\d{2},\d{3}) --> (\d{2}:\d{2}:\d{2},\d{3})", timestamp_line)

        if match:
            start_str, end_str = match.groups()
            start = parse_srt_timestamp(start_str)
            end = parse_srt_timestamp(end_str)
            duration = end - start

            if duration > longest_duration:
                longest_duration = duration
                longest_segment = segment

    if longest_segment:
        print(f"Longest segment: {longest_duration:.2f} seconds")
        print("=" * 60)
        print(longest_segment)
        print("=" * 60)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python find_longest_segment.py <srt_file>")
        sys.exit(1)

    find_longest_segment(sys.argv[1])
