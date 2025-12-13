#!/usr/bin/env python3
# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
SRT Quality Analysis Tool

Compare auto-generated subtitles with human reference subtitles.
Calculates WER, CER, and generates detailed analysis reports.

Usage:
    python analyze_quality.py --auto /path/to/auto.srt --reference /path/to/human.srt
"""

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


@dataclass
class SRTEntry:
    """Single SRT subtitle entry."""
    index: int
    start_time: float  # seconds
    end_time: float    # seconds
    text: str


def parse_srt_time(time_str: str) -> float:
    """Convert SRT timestamp to seconds."""
    # Format: HH:MM:SS,mmm
    match = re.match(r'(\d{2}):(\d{2}):(\d{2})[,.](\d{3})', time_str.strip())
    if not match:
        return 0.0
    h, m, s, ms = match.groups()
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000


def parse_srt(filepath: str) -> List[SRTEntry]:
    """Parse an SRT file into list of entries."""
    entries = []

    with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
        content = f.read()

    # Split by blank lines (double newlines)
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

            start_time = parse_srt_time(time_match.group(1))
            end_time = parse_srt_time(time_match.group(2))

            # Remaining lines: text (may contain HTML tags for formatting)
            text = ' '.join(lines[2:])
            # Remove HTML tags like <i>, </i>, <b>, etc.
            text = re.sub(r'<[^>]+>', '', text)

            entries.append(SRTEntry(
                index=index,
                start_time=start_time,
                end_time=end_time,
                text=text.strip()
            ))
        except (ValueError, IndexError):
            continue

    return entries


def extract_full_text(entries: List[SRTEntry]) -> str:
    """Extract all text from SRT entries, joined by space."""
    texts = [e.text for e in entries if e.text.strip()]
    return ' '.join(texts)


def normalize_text(text: str) -> str:
    """Normalize text for comparison (lowercase, remove punctuation)."""
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation except apostrophes (for contractions)
    text = re.sub(r"[^\w\s']", ' ', text)
    # Collapse multiple spaces
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def calculate_wer_cer(hypothesis: str, reference: str) -> Dict:
    """
    Calculate WER and CER using NeMo metrics.
    Falls back to simple implementation if NeMo not available.
    """
    try:
        from nemo.collections.asr.metrics.wer import word_error_rate_detail

        wer, tokens, ins_rate, del_rate, sub_rate = word_error_rate_detail(
            hypotheses=[hypothesis], references=[reference], use_cer=False
        )

        cer, char_tokens, c_ins, c_del, c_sub = word_error_rate_detail(
            hypotheses=[hypothesis], references=[reference], use_cer=True
        )

        return {
            'wer': wer * 100,  # Convert to percentage
            'cer': cer * 100,
            'word_tokens': tokens,
            'char_tokens': char_tokens,
            'insertions': ins_rate * 100,
            'deletions': del_rate * 100,
            'substitutions': sub_rate * 100,
        }
    except ImportError:
        # Fallback: simple Levenshtein-based WER
        return calculate_wer_simple(hypothesis, reference)


def calculate_wer_simple(hypothesis: str, reference: str) -> Dict:
    """Simple WER calculation without NeMo dependency."""
    hyp_words = hypothesis.split()
    ref_words = reference.split()

    # Dynamic programming for edit distance
    m, n = len(ref_words), len(hyp_words)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref_words[i-1] == hyp_words[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])

    wer = dp[m][n] / m * 100 if m > 0 else 0

    # Similar for CER
    hyp_chars = list(hypothesis.replace(' ', ''))
    ref_chars = list(reference.replace(' ', ''))
    m, n = len(ref_chars), len(hyp_chars)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref_chars[i-1] == hyp_chars[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])

    cer = dp[m][n] / m * 100 if m > 0 else 0

    return {
        'wer': wer,
        'cer': cer,
        'word_tokens': len(ref_words),
        'char_tokens': len(ref_chars),
        'insertions': 0,
        'deletions': 0,
        'substitutions': 0,
    }


def find_matching_segments(
    auto_entries: List[SRTEntry],
    ref_entries: List[SRTEntry],
    tolerance: float = 2.0
) -> List[Tuple[Optional[SRTEntry], Optional[SRTEntry]]]:
    """
    Match auto segments with reference segments by timestamp overlap.

    Args:
        auto_entries: Auto-generated subtitles
        ref_entries: Human reference subtitles
        tolerance: Time tolerance in seconds for matching

    Returns:
        List of (auto_entry, ref_entry) tuples, None if no match
    """
    matched = []
    used_ref = set()

    for auto in auto_entries:
        best_match = None
        best_overlap = 0

        for i, ref in enumerate(ref_entries):
            if i in used_ref:
                continue

            # Calculate time overlap
            overlap_start = max(auto.start_time, ref.start_time)
            overlap_end = min(auto.end_time, ref.end_time)
            overlap = max(0, overlap_end - overlap_start)

            # Also consider proximity if no overlap
            if overlap == 0:
                proximity = min(
                    abs(auto.start_time - ref.start_time),
                    abs(auto.end_time - ref.end_time)
                )
                if proximity <= tolerance:
                    overlap = tolerance - proximity

            if overlap > best_overlap:
                best_overlap = overlap
                best_match = (i, ref)

        if best_match:
            used_ref.add(best_match[0])
            matched.append((auto, best_match[1]))
        else:
            matched.append((auto, None))

    # Add unmatched reference entries
    for i, ref in enumerate(ref_entries):
        if i not in used_ref:
            matched.append((None, ref))

    return matched


def analyze_errors(
    auto_entries: List[SRTEntry],
    ref_entries: List[SRTEntry]
) -> Dict:
    """Analyze common error patterns."""

    auto_text = normalize_text(extract_full_text(auto_entries))
    ref_text = normalize_text(extract_full_text(ref_entries))

    auto_words = set(auto_text.split())
    ref_words = set(ref_text.split())

    # Words in auto but not in reference (potential insertions/errors)
    extra_words = auto_words - ref_words
    # Words in reference but not in auto (potential deletions/misses)
    missing_words = ref_words - auto_words

    return {
        'extra_words': sorted(list(extra_words))[:50],  # Top 50
        'missing_words': sorted(list(missing_words))[:50],
        'auto_word_count': len(auto_text.split()),
        'ref_word_count': len(ref_text.split()),
    }


def generate_sample_comparisons(
    auto_entries: List[SRTEntry],
    ref_entries: List[SRTEntry],
    num_samples: int = 10
) -> List[Dict]:
    """Generate sample side-by-side comparisons at different timestamps."""

    samples = []
    matched = find_matching_segments(auto_entries, ref_entries)

    # Filter to only pairs where both exist
    pairs = [(a, r) for a, r in matched if a and r]

    if not pairs:
        return samples

    # Sample at regular intervals
    step = max(1, len(pairs) // num_samples)

    for i in range(0, len(pairs), step):
        if len(samples) >= num_samples:
            break

        auto, ref = pairs[i]

        # Calculate similarity
        auto_norm = normalize_text(auto.text)
        ref_norm = normalize_text(ref.text)

        samples.append({
            'timestamp': f"{int(auto.start_time//60):02d}:{int(auto.start_time%60):02d}",
            'auto_text': auto.text,
            'ref_text': ref.text,
            'match': auto_norm == ref_norm
        })

    return samples


def format_time(seconds: float) -> str:
    """Format seconds as HH:MM:SS."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def generate_report(
    auto_path: str,
    ref_path: str,
    output_path: Optional[str] = None
) -> str:
    """Generate comprehensive quality analysis report."""

    print(f"Parsing auto-generated SRT: {auto_path}")
    auto_entries = parse_srt(auto_path)

    print(f"Parsing reference SRT: {ref_path}")
    ref_entries = parse_srt(ref_path)

    print("Extracting and normalizing text...")
    auto_text = extract_full_text(auto_entries)
    ref_text = extract_full_text(ref_entries)

    auto_norm = normalize_text(auto_text)
    ref_norm = normalize_text(ref_text)

    print("Calculating WER/CER metrics...")
    metrics = calculate_wer_cer(auto_norm, ref_norm)

    print("Analyzing error patterns...")
    errors = analyze_errors(auto_entries, ref_entries)

    print("Generating sample comparisons...")
    samples = generate_sample_comparisons(auto_entries, ref_entries, num_samples=15)

    # Calculate timing stats
    auto_duration = max(e.end_time for e in auto_entries) if auto_entries else 0
    ref_duration = max(e.end_time for e in ref_entries) if ref_entries else 0

    # Build report
    report = []
    report.append("=" * 80)
    report.append("NEMOSCRIBE QUALITY ANALYSIS REPORT")
    report.append("=" * 80)
    report.append("")
    report.append("## File Information")
    report.append(f"  Auto-generated: {Path(auto_path).name}")
    report.append(f"  Reference:      {Path(ref_path).name}")
    report.append("")
    report.append("## Statistics")
    report.append(f"  Auto subtitle count:    {len(auto_entries)}")
    report.append(f"  Reference subtitle count: {len(ref_entries)}")
    report.append(f"  Auto word count:        {errors['auto_word_count']}")
    report.append(f"  Reference word count:   {errors['ref_word_count']}")
    report.append(f"  Auto duration:          {format_time(auto_duration)}")
    report.append(f"  Reference duration:     {format_time(ref_duration)}")
    report.append("")
    report.append("## Quality Metrics")
    report.append(f"  Word Error Rate (WER):  {metrics['wer']:.2f}%")
    report.append(f"  Character Error Rate:   {metrics['cer']:.2f}%")
    report.append(f"  Insertion Rate:         {metrics['insertions']:.2f}%")
    report.append(f"  Deletion Rate:          {metrics['deletions']:.2f}%")
    report.append(f"  Substitution Rate:      {metrics['substitutions']:.2f}%")
    report.append("")
    report.append("## Interpretation")

    wer = metrics['wer']
    if wer < 10:
        quality = "Excellent"
        desc = "接近專業人工轉錄品質"
    elif wer < 20:
        quality = "Good"
        desc = "大部分內容準確，適合一般觀看"
    elif wer < 30:
        quality = "Fair"
        desc = "可理解但有明顯錯誤，需人工校對"
    elif wer < 50:
        quality = "Poor"
        desc = "錯誤較多，需大量修正"
    else:
        quality = "Very Poor"
        desc = "準確度不足，建議重新處理"

    report.append(f"  Quality Rating:         {quality}")
    report.append(f"  Description:            {desc}")
    report.append("")
    report.append("## Sample Comparisons")
    report.append("-" * 80)

    for i, sample in enumerate(samples, 1):
        match_icon = "✓" if sample['match'] else "✗"
        report.append(f"[{sample['timestamp']}] {match_icon}")
        report.append(f"  AUTO: {sample['auto_text'][:100]}")
        report.append(f"  REF:  {sample['ref_text'][:100]}")
        report.append("")

    report.append("-" * 80)
    report.append("")
    report.append("## Error Analysis")
    report.append("")
    report.append("### Words in Auto but NOT in Reference (potential errors):")
    extra_sample = errors['extra_words'][:30]
    report.append(f"  {', '.join(extra_sample)}")
    report.append("")
    report.append("### Words in Reference but NOT in Auto (missed words):")
    missing_sample = errors['missing_words'][:30]
    report.append(f"  {', '.join(missing_sample)}")
    report.append("")
    report.append("=" * 80)
    report.append("END OF REPORT")
    report.append("=" * 80)

    report_text = '\n'.join(report)

    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report_text)
        print(f"\nReport saved to: {output_path}")

    return report_text


def main():
    parser = argparse.ArgumentParser(
        description="Analyze SRT subtitle quality by comparing with reference"
    )
    parser.add_argument(
        '--auto', '-a',
        required=True,
        help='Path to auto-generated SRT file'
    )
    parser.add_argument(
        '--reference', '-r',
        required=True,
        help='Path to human reference SRT file'
    )
    parser.add_argument(
        '--output', '-o',
        help='Path to save report (optional)'
    )

    args = parser.parse_args()

    report = generate_report(args.auto, args.reference, args.output)
    print("\n" + report)


if __name__ == "__main__":
    main()
