#!/usr/bin/env python3
"""Analyze SRT file statistics."""

import re
from pathlib import Path
from typing import List, Tuple


def parse_srt_timestamp(timestamp: str) -> float:
    """Convert SRT timestamp to seconds."""
    # Format: 00:00:00,000
    h, m, s = timestamp.split(":")
    s, ms = s.split(",")
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000


def analyze_srt(srt_path: str) -> dict:
    """Analyze SRT file and return statistics."""
    content = Path(srt_path).read_text(encoding="utf-8")

    # Parse segments
    segments = []
    pattern = r"\d+\n(\d{2}:\d{2}:\d{2},\d{3}) --> (\d{2}:\d{2}:\d{2},\d{3})"

    for match in re.finditer(pattern, content):
        start_str, end_str = match.groups()
        start = parse_srt_timestamp(start_str)
        end = parse_srt_timestamp(end_str)
        duration = end - start
        segments.append((start, end, duration))

    # Calculate statistics
    durations = [d for _, _, d in segments]
    durations.sort(reverse=True)

    stats = {
        "segments": len(segments),
        "duration_max_seconds": round(max(durations), 2) if durations else 0,
        "duration_gt_6s": sum(1 for d in durations if d > 6.0),
        "duration_gt_10s": sum(1 for d in durations if d > 10.0),
        "top_10_durations": [round(d, 2) for d in durations[:10]],
    }

    return stats


if __name__ == "__main__":
    import sys
    import json

    if len(sys.argv) < 2:
        print("Usage: python analyze_srt_stats.py <srt_file>")
        sys.exit(1)

    srt_file = sys.argv[1]
    stats = analyze_srt(srt_file)

    print(json.dumps(stats, indent=2))
