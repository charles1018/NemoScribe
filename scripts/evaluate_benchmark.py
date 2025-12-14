#!/usr/bin/env python3
# MIT License
# Copyright (c) 2025 charles1018
"""
NemoScribe Parameter Benchmark Evaluation Script

Compares generated SRT files against a reference (human-made) subtitle file.
Calculates WER (Word Error Rate) and timestamp offset metrics.

Usage:
    cd <project_directory>
    uv run python scripts/evaluate_benchmark.py

Requirements:
    uv sync --extra benchmark  (installs jiwer)
"""

import re
import argparse
from dataclasses import dataclass
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path
from typing import Optional

try:
    from jiwer import wer, cer
except ImportError:
    print("ERROR: jiwer not installed. Run: uv sync --extra benchmark")
    exit(1)


@dataclass
class SubtitleEntry:
    """A single subtitle entry."""
    index: int
    start_time: float  # in seconds
    end_time: float    # in seconds
    text: str


@dataclass
class EvaluationResult:
    """Evaluation result for a single SRT file."""
    filename: str
    onset: float
    offset: float
    wer_score: float
    cer_score: float
    avg_start_offset: float
    avg_end_offset: float
    max_start_offset: float
    max_end_offset: float
    matched_segments: int
    total_ref_segments: int
    total_hyp_segments: int
    combined_score: float


def parse_srt_timestamp(timestamp: str) -> float:
    """Convert SRT timestamp (HH:MM:SS,mmm) to seconds."""
    match = re.match(r'(\d{2}):(\d{2}):(\d{2})[,.](\d{3})', timestamp.strip())
    if not match:
        raise ValueError(f"Invalid timestamp format: {timestamp}")
    hours, minutes, seconds, millis = map(int, match.groups())
    return hours * 3600 + minutes * 60 + seconds + millis / 1000


def parse_srt_file(filepath: Path) -> list[SubtitleEntry]:
    """Parse an SRT file and return list of SubtitleEntry."""
    entries = []
    content = filepath.read_text(encoding='utf-8-sig')  # Handle BOM

    # Split by double newline (subtitle blocks)
    blocks = re.split(r'\n\s*\n', content.strip())

    for block in blocks:
        lines = block.strip().split('\n')
        if len(lines) < 3:
            continue

        try:
            # First line: index
            index = int(lines[0].strip())

            # Second line: timestamps
            time_match = re.match(
                r'(\d{2}:\d{2}:\d{2}[,.]\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}[,.]\d{3})',
                lines[1].strip()
            )
            if not time_match:
                continue

            start_time = parse_srt_timestamp(time_match.group(1))
            end_time = parse_srt_timestamp(time_match.group(2))

            # Remaining lines: text
            text = ' '.join(lines[2:]).strip()

            entries.append(SubtitleEntry(
                index=index,
                start_time=start_time,
                end_time=end_time,
                text=text
            ))
        except (ValueError, IndexError):
            continue

    return entries


def normalize_text(text: str) -> str:
    """Normalize text for comparison."""
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation except apostrophes
    text = re.sub(r"[^\w\s']", ' ', text)
    # Normalize whitespace
    text = ' '.join(text.split())
    return text


def align_segments(ref_entries: list[SubtitleEntry],
                   hyp_entries: list[SubtitleEntry],
                   similarity_threshold: float = 0.5) -> list[tuple[SubtitleEntry, Optional[SubtitleEntry]]]:
    """
    Align hypothesis segments to reference segments based on text similarity.
    Returns list of (ref_entry, matched_hyp_entry or None).
    """
    alignments = []
    used_hyp_indices = set()

    for ref_entry in ref_entries:
        ref_text = normalize_text(ref_entry.text)
        best_match = None
        best_score = 0
        best_idx = -1

        # Search in a time window around the reference
        for i, hyp_entry in enumerate(hyp_entries):
            if i in used_hyp_indices:
                continue

            # Only consider hypotheses within reasonable time range
            time_diff = abs(hyp_entry.start_time - ref_entry.start_time)
            if time_diff > 30:  # 30 second tolerance
                continue

            hyp_text = normalize_text(hyp_entry.text)
            similarity = SequenceMatcher(None, ref_text, hyp_text).ratio()

            if similarity > best_score and similarity >= similarity_threshold:
                best_score = similarity
                best_match = hyp_entry
                best_idx = i

        if best_match:
            used_hyp_indices.add(best_idx)
        alignments.append((ref_entry, best_match))

    return alignments


def evaluate_srt(ref_path: Path, hyp_path: Path) -> EvaluationResult:
    """Evaluate a hypothesis SRT against reference SRT."""
    # Extract onset/offset from filename
    filename = hyp_path.stem
    onset_match = re.search(r'onset([\d.]+)', filename)
    offset_match = re.search(r'offset([\d.]+)', filename)
    onset = float(onset_match.group(1)) if onset_match else 0.0
    offset = float(offset_match.group(1)) if offset_match else 0.0

    # Parse files
    ref_entries = parse_srt_file(ref_path)
    hyp_entries = parse_srt_file(hyp_path)

    # Calculate WER and CER on full text
    ref_text = ' '.join(normalize_text(e.text) for e in ref_entries)
    hyp_text = ' '.join(normalize_text(e.text) for e in hyp_entries)

    wer_score = wer(ref_text, hyp_text)
    cer_score = cer(ref_text, hyp_text)

    # Align segments and calculate timing offsets
    alignments = align_segments(ref_entries, hyp_entries)

    start_offsets = []
    end_offsets = []
    matched_count = 0

    for ref_entry, hyp_entry in alignments:
        if hyp_entry is not None:
            matched_count += 1
            start_offsets.append(abs(hyp_entry.start_time - ref_entry.start_time))
            end_offsets.append(abs(hyp_entry.end_time - ref_entry.end_time))

    # Calculate timing metrics
    if start_offsets:
        avg_start_offset = sum(start_offsets) / len(start_offsets)
        avg_end_offset = sum(end_offsets) / len(end_offsets)
        max_start_offset = max(start_offsets)
        max_end_offset = max(end_offsets)
    else:
        avg_start_offset = avg_end_offset = max_start_offset = max_end_offset = float('inf')

    # Calculate combined score
    # WER weight: 0.7, Timing weight: 0.3
    # Timing accuracy = 1 - normalized_avg_offset (cap at 5 seconds)
    avg_offset = (avg_start_offset + avg_end_offset) / 2
    timing_accuracy = max(0, 1 - avg_offset / 5.0)
    combined_score = (1 - min(wer_score, 1.0)) * 0.7 + timing_accuracy * 0.3

    return EvaluationResult(
        filename=filename,
        onset=onset,
        offset=offset,
        wer_score=wer_score,
        cer_score=cer_score,
        avg_start_offset=avg_start_offset,
        avg_end_offset=avg_end_offset,
        max_start_offset=max_start_offset,
        max_end_offset=max_end_offset,
        matched_segments=matched_count,
        total_ref_segments=len(ref_entries),
        total_hyp_segments=len(hyp_entries),
        combined_score=combined_score
    )


def generate_report(results: list[EvaluationResult],
                    ref_path: Path,
                    output_path: Path) -> str:
    """Generate evaluation report."""
    # Sort by combined score (descending)
    results_sorted = sorted(results, key=lambda r: r.combined_score, reverse=True)

    lines = []
    lines.append("=" * 70)
    lines.append("NemoScribe Parameter Benchmark Evaluation Report")
    lines.append("=" * 70)
    lines.append(f"Evaluation time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Reference subtitle: {ref_path.name}")
    lines.append(f"Total tests evaluated: {len(results)}")
    lines.append("")

    # Summary table
    lines.append("-" * 70)
    lines.append(f"{'Rank':<5} {'Params':<22} {'WER':<8} {'CER':<8} {'Timing':<10} {'Score':<8}")
    lines.append("-" * 70)

    for i, r in enumerate(results_sorted, 1):
        params = f"onset={r.onset} offset={r.offset}"
        timing = f"{r.avg_start_offset:.2f}s"
        lines.append(
            f"{i:<5} {params:<22} {r.wer_score*100:>6.1f}% {r.cer_score*100:>6.1f}% "
            f"{timing:<10} {r.combined_score:.3f}"
        )

    lines.append("-" * 70)
    lines.append("")

    # Best result
    best = results_sorted[0]
    lines.append("=" * 70)
    lines.append("BEST PARAMETERS")
    lines.append("=" * 70)
    lines.append(f"  vad.onset = {best.onset}")
    lines.append(f"  vad.offset = {best.offset}")
    lines.append("")
    lines.append("Metrics:")
    lines.append(f"  Word Error Rate (WER): {best.wer_score*100:.2f}%")
    lines.append(f"  Char Error Rate (CER): {best.cer_score*100:.2f}%")
    lines.append(f"  Avg Start Offset: {best.avg_start_offset:.3f}s")
    lines.append(f"  Avg End Offset: {best.avg_end_offset:.3f}s")
    lines.append(f"  Matched Segments: {best.matched_segments}/{best.total_ref_segments}")
    lines.append(f"  Combined Score: {best.combined_score:.3f}")
    lines.append("")

    # Detailed results
    lines.append("=" * 70)
    lines.append("DETAILED RESULTS")
    lines.append("=" * 70)

    for i, r in enumerate(results_sorted, 1):
        lines.append("")
        lines.append(f"#{i} {r.filename}")
        lines.append(f"    WER: {r.wer_score*100:.2f}%, CER: {r.cer_score*100:.2f}%")
        lines.append(f"    Timing - Start: avg={r.avg_start_offset:.3f}s max={r.max_start_offset:.3f}s")
        lines.append(f"    Timing - End: avg={r.avg_end_offset:.3f}s max={r.max_end_offset:.3f}s")
        lines.append(f"    Segments: {r.matched_segments} matched / {r.total_ref_segments} ref / {r.total_hyp_segments} hyp")
        lines.append(f"    Combined Score: {r.combined_score:.3f}")

    lines.append("")
    lines.append("=" * 70)
    lines.append("Interpretation Guide:")
    lines.append("  - WER < 10%: Excellent transcription quality")
    lines.append("  - WER 10-20%: Good quality, minor errors")
    lines.append("  - WER 20-30%: Acceptable, noticeable errors")
    lines.append("  - WER > 30%: Poor quality, needs improvement")
    lines.append("  - Timing < 0.5s: Excellent sync")
    lines.append("  - Timing 0.5-1.0s: Good sync")
    lines.append("  - Timing > 1.0s: Noticeable delay")
    lines.append("=" * 70)

    report = '\n'.join(lines)

    # Write report
    output_path.write_text(report, encoding='utf-8')

    return report


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate NemoScribe parameter benchmark results'
    )
    parser.add_argument(
        '--test-dir',
        type=Path,
        default=Path(r'<test_directory>'),
        help='Directory containing test SRT files'
    )
    parser.add_argument(
        '--reference',
        type=Path,
        default=Path(r'<video_directory>\Chicago.Fire.S12E01.1080p.WEB.h264-ETHEL.ENG.srt'),
        help='Reference (human-made) SRT file'
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=None,
        help='Output report path (default: test-dir/evaluation_report.txt)'
    )

    args = parser.parse_args()

    # Validate paths
    if not args.test_dir.exists():
        print(f"ERROR: Test directory not found: {args.test_dir}")
        return 1

    if not args.reference.exists():
        print(f"ERROR: Reference file not found: {args.reference}")
        return 1

    # Find test SRT files
    test_files = list(args.test_dir.glob('onset*.srt'))
    if not test_files:
        print(f"ERROR: No test SRT files found in {args.test_dir}")
        print("Expected pattern: onset*_offset*.srt")
        return 1

    print(f"Found {len(test_files)} test files")
    print(f"Reference: {args.reference.name}")
    print()

    # Evaluate each file
    results = []
    for i, test_file in enumerate(sorted(test_files), 1):
        print(f"[{i}/{len(test_files)}] Evaluating {test_file.name}...")
        try:
            result = evaluate_srt(args.reference, test_file)
            results.append(result)
            print(f"         WER: {result.wer_score*100:.1f}%, Score: {result.combined_score:.3f}")
        except Exception as e:
            print(f"         ERROR: {e}")

    if not results:
        print("ERROR: No valid results")
        return 1

    # Generate report
    output_path = args.output or (args.test_dir / 'evaluation_report.txt')
    print()
    print(f"Generating report: {output_path}")
    report = generate_report(results, args.reference, output_path)

    print()
    print(report)

    return 0


if __name__ == '__main__':
    exit(main())
