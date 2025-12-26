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
Integration tests for NemoScribe improvements (Phases 1-4).

This script tests each improvement feature individually and in combination
to ensure quality and backward compatibility.

Usage:
    # Run all tests
    uv run python test_improvements.py

    # Run specific test
    uv run python test_improvements.py --test vad

    # Test with custom audio file
    uv run python test_improvements.py --audio /path/to/audio.wav

Based on NeMo's speech_to_text_eval.py for WER/CER calculation.
"""

import argparse
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


@dataclass
class TestResult:
    """Test result container."""

    name: str
    passed: bool
    message: str
    metrics: Optional[Dict] = None


def check_imports() -> Tuple[bool, str]:
    """Check if required imports are available."""
    missing = []

    try:
        import torch
    except ImportError:
        missing.append("torch")

    try:
        from nemo.collections.asr.models import ASRModel
    except ImportError:
        missing.append("nemo_toolkit")

    if missing:
        return False, f"Missing dependencies: {', '.join(missing)}"

    return True, "All imports available"


def calculate_transcription_quality(
    hypothesis: str,
    reference: str,
) -> Dict:
    """
    Calculate transcription quality metrics.

    Based on NeMo's speech_to_text_eval.py and eval_utils.py.

    Args:
        hypothesis: Predicted transcription text
        reference: Ground truth transcription text

    Returns:
        dict with WER, CER, and other metrics
    """
    from nemo.collections.asr.metrics.wer import word_error_rate, word_error_rate_detail

    # Calculate WER with detail
    wer, tokens, ins_rate, del_rate, sub_rate = word_error_rate_detail(
        hypotheses=[hypothesis], references=[reference], use_cer=False
    )

    # Calculate CER
    cer, char_tokens, _, _, _ = word_error_rate_detail(
        hypotheses=[hypothesis], references=[reference], use_cer=True
    )

    return {
        "wer": wer,
        "cer": cer,
        "tokens": tokens,
        "char_tokens": char_tokens,
        "insertion_rate": ins_rate,
        "deletion_rate": del_rate,
        "substitution_rate": sub_rate,
        "hypothesis_words": len(hypothesis.split()),
        "reference_words": len(reference.split()),
    }


def test_baseline_config() -> TestResult:
    """Test that baseline configuration produces valid output."""
    from nemoscribe import VideoToSRTConfig

    try:
        # Create default config
        cfg = VideoToSRTConfig()

        # Verify all new features are disabled by default
        checks = [
            ("VAD disabled", not cfg.vad.enabled),
            ("ITN disabled", not cfg.postprocessing.enable_itn),
            ("Smart segmentation enabled (but only activates with VAD)", cfg.audio.smart_segmentation),
            ("RTFx disabled", not cfg.performance.calculate_rtfx),
        ]

        failed = [name for name, passed in checks if not passed]
        if failed:
            return TestResult(
                name="baseline_config",
                passed=False,
                message=f"Default config checks failed: {failed}",
            )

        return TestResult(
            name="baseline_config",
            passed=True,
            message="Default configuration is backward compatible",
        )

    except Exception as e:
        return TestResult(
            name="baseline_config",
            passed=False,
            message=f"Exception: {e}",
        )


def test_vad_config() -> TestResult:
    """Test VAD configuration dataclass."""
    from nemoscribe import VADConfig

    try:
        # Test default values
        vad_cfg = VADConfig()
        assert not vad_cfg.enabled, "VAD should be disabled by default"
        assert vad_cfg.model == "vad_multilingual_frame_marblenet"
        assert vad_cfg.onset == 0.3
        assert vad_cfg.offset == 0.3

        # Test custom configuration
        custom_cfg = VADConfig(
            enabled=True,
            onset=0.5,
            offset=0.4,
            min_duration_on=0.5,
        )
        assert custom_cfg.enabled
        assert custom_cfg.onset == 0.5

        return TestResult(
            name="vad_config",
            passed=True,
            message="VADConfig works correctly",
        )

    except AssertionError as e:
        return TestResult(
            name="vad_config",
            passed=False,
            message=f"Assertion failed: {e}",
        )
    except Exception as e:
        return TestResult(
            name="vad_config",
            passed=False,
            message=f"Exception: {e}",
        )


def test_itn_functions() -> TestResult:
    """Test ITN (Inverse Text Normalization) functions."""
    from nemoscribe import apply_itn, get_itn_normalizer

    try:
        # Test graceful fallback when nemo_text_processing is not installed
        normalizer = get_itn_normalizer("en", "lower_cased")

        if normalizer is None:
            return TestResult(
                name="itn_functions",
                passed=True,
                message="ITN normalizer unavailable (expected if nemo_text_processing not installed)",
            )

        # Test ITN normalization
        test_cases = [
            ("twenty five", "25"),  # Number
            ("three point one four", "3.14"),  # Decimal
        ]

        for input_text, expected_substring in test_cases:
            result = apply_itn(input_text, normalizer)
            if expected_substring not in result:
                return TestResult(
                    name="itn_functions",
                    passed=False,
                    message=f"ITN failed: '{input_text}' -> '{result}' (expected substring: '{expected_substring}')",
                )

        # Test with None normalizer
        result = apply_itn("test text", None)
        assert result == "test text", "apply_itn should return original text when normalizer is None"

        # Test with empty text
        result = apply_itn("", normalizer)
        assert result == "", "apply_itn should handle empty text"

        return TestResult(
            name="itn_functions",
            passed=True,
            message="ITN functions work correctly",
        )

    except Exception as e:
        return TestResult(
            name="itn_functions",
            passed=False,
            message=f"Exception: {e}",
        )


def test_decoding_config() -> TestResult:
    """Test decoding configuration dataclass."""
    from nemoscribe import DecodingConfig

    try:
        cfg = DecodingConfig()

        # CUDA graphs should be enabled by default
        assert cfg.rnnt_fused_batch_size == -1, "CUDA graphs should be enabled by default"
        assert cfg.rnnt_timestamp_type == "all"
        assert cfg.ctc_timestamp_type == "all"

        return TestResult(
            name="decoding_config",
            passed=True,
            message="DecodingConfig works correctly with CUDA graphs enabled",
        )

    except AssertionError as e:
        return TestResult(
            name="decoding_config",
            passed=False,
            message=f"Assertion failed: {e}",
        )
    except Exception as e:
        return TestResult(
            name="decoding_config",
            passed=False,
            message=f"Exception: {e}",
        )


def test_smart_segmentation() -> TestResult:
    """Test smart segmentation functions."""
    from nemoscribe import (
        find_optimal_split_points,
        get_silence_gaps_from_speech,
    )

    try:
        # Test speech segments -> silence gaps conversion
        speech_segments = [
            (1.0, 3.0),   # Speech from 1s to 3s
            (5.0, 8.0),   # Speech from 5s to 8s
            (10.0, 12.0), # Speech from 10s to 12s
        ]
        total_duration = 15.0

        silence_gaps = get_silence_gaps_from_speech(
            speech_segments, total_duration, min_silence_for_split=0.3
        )

        # Expected silences: [0-1], [3-5], [8-10], [12-15]
        assert len(silence_gaps) == 4, f"Expected 4 silence gaps, got {len(silence_gaps)}"

        # Test optimal split point detection
        split_points = find_optimal_split_points(
            speech_segments,
            total_duration,
            max_chunk_duration=5.0,
            min_silence_for_split=0.3,
        )

        # Should start at 0 and end at total_duration
        assert split_points[0] == 0.0, "First split point should be 0.0"
        assert split_points[-1] == total_duration, f"Last split point should be {total_duration}"

        # Split points should be in silence regions
        for sp in split_points[1:-1]:
            in_silence = any(
                start <= sp <= end for start, end, _ in silence_gaps
            )
            # Allow for boundary cases
            assert in_silence or sp in [0.0, total_duration], f"Split point {sp} not in silence region"

        return TestResult(
            name="smart_segmentation",
            passed=True,
            message=f"Smart segmentation works correctly ({len(split_points)} split points)",
        )

    except AssertionError as e:
        return TestResult(
            name="smart_segmentation",
            passed=False,
            message=f"Assertion failed: {e}",
        )
    except Exception as e:
        return TestResult(
            name="smart_segmentation",
            passed=False,
            message=f"Exception: {e}",
        )


def test_segment_merging() -> TestResult:
    """Test overlapping segment merging functions."""
    from nemoscribe import deduplicate_segments, merge_overlapping_segments

    try:
        # Test overlapping segments
        segments = [
            (0.0, 2.0, "Hello world"),
            (1.8, 3.5, "Hello world today"),  # Overlaps with first
            (5.0, 7.0, "Another segment"),
        ]

        merged = merge_overlapping_segments(segments, overlap_threshold=0.5)
        assert len(merged) == 2, f"Expected 2 merged segments, got {len(merged)}"

        # Test deduplication
        segments_with_dups = [
            (0.0, 2.0, "Hello world"),
            (2.5, 4.0, "Hello world"),  # Near-duplicate text
            (5.0, 7.0, "Different text"),
        ]

        deduped = deduplicate_segments(segments_with_dups, similarity_threshold=0.8)
        assert len(deduped) == 2, f"Expected 2 deduplicated segments, got {len(deduped)}"

        return TestResult(
            name="segment_merging",
            passed=True,
            message="Segment merging and deduplication work correctly",
        )

    except AssertionError as e:
        return TestResult(
            name="segment_merging",
            passed=False,
            message=f"Assertion failed: {e}",
        )
    except Exception as e:
        return TestResult(
            name="segment_merging",
            passed=False,
            message=f"Exception: {e}",
        )


def test_performance_config() -> TestResult:
    """Test performance configuration dataclass."""
    from nemoscribe import PerformanceConfig

    try:
        cfg = PerformanceConfig()

        # RTFx should be disabled by default
        assert not cfg.calculate_rtfx, "RTFx should be disabled by default"
        assert cfg.warmup_steps == 1, "Default warmup steps should be 1"

        # Test enabling RTFx
        rtfx_cfg = PerformanceConfig(calculate_rtfx=True, warmup_steps=2)
        assert rtfx_cfg.calculate_rtfx
        assert rtfx_cfg.warmup_steps == 2

        return TestResult(
            name="performance_config",
            passed=True,
            message="PerformanceConfig works correctly",
        )

    except AssertionError as e:
        return TestResult(
            name="performance_config",
            passed=False,
            message=f"Assertion failed: {e}",
        )
    except Exception as e:
        return TestResult(
            name="performance_config",
            passed=False,
            message=f"Exception: {e}",
        )


def test_quality_metrics() -> TestResult:
    """Test transcription quality metrics calculation."""
    try:
        # Test perfect match
        result = calculate_transcription_quality(
            hypothesis="hello world",
            reference="hello world",
        )
        assert result["wer"] == 0.0, f"Expected WER=0 for perfect match, got {result['wer']}"
        assert result["cer"] == 0.0, f"Expected CER=0 for perfect match, got {result['cer']}"

        # Test with errors
        result = calculate_transcription_quality(
            hypothesis="hello word",  # 'world' -> 'word' (substitution)
            reference="hello world",
        )
        assert result["wer"] > 0.0, "Expected WER > 0 for imperfect match"
        assert result["substitution_rate"] > 0.0, "Expected substitution errors"

        # Test with insertions/deletions
        result = calculate_transcription_quality(
            hypothesis="hello beautiful world",  # Extra word
            reference="hello world",
        )
        assert result["insertion_rate"] > 0.0, "Expected insertion errors"

        return TestResult(
            name="quality_metrics",
            passed=True,
            message="Quality metrics calculation works correctly",
            metrics={"example_wer": result["wer"]},
        )

    except AssertionError as e:
        return TestResult(
            name="quality_metrics",
            passed=False,
            message=f"Assertion failed: {e}",
        )
    except Exception as e:
        return TestResult(
            name="quality_metrics",
            passed=False,
            message=f"Exception: {e}",
        )


def test_srt_formatting() -> TestResult:
    """Test SRT timestamp and segment formatting."""
    from nemoscribe import clip_segments_to_window, format_srt_timestamp

    try:
        # Test timestamp formatting
        assert format_srt_timestamp(0.0) == "00:00:00,000"
        assert format_srt_timestamp(61.5) == "00:01:01,500"
        assert format_srt_timestamp(3661.123) == "01:01:01,123"

        # Test segment clipping
        segments = [
            (0.0, 2.0, "Before window"),
            (5.0, 8.0, "In window"),
            (10.0, 12.0, "After window"),
        ]

        clipped = clip_segments_to_window(segments, window_start=4.0, window_end=9.0)
        assert len(clipped) == 1, f"Expected 1 segment in window, got {len(clipped)}"
        assert clipped[0][2] == "In window"

        return TestResult(
            name="srt_formatting",
            passed=True,
            message="SRT formatting and clipping work correctly",
        )

    except AssertionError as e:
        return TestResult(
            name="srt_formatting",
            passed=False,
            message=f"Assertion failed: {e}",
        )
    except Exception as e:
        return TestResult(
            name="srt_formatting",
            passed=False,
            message=f"Exception: {e}",
        )


def test_srt_edge_cases() -> TestResult:
    """Test SRT timestamp formatting edge cases."""
    from nemoscribe import format_srt_timestamp

    try:
        # Test negative values (should clamp to 0)
        result = format_srt_timestamp(-1.0)
        assert result == "00:00:00,000", f"Negative time should clamp to 0, got {result}"

        # Test very large values (e.g., 24+ hours)
        result = format_srt_timestamp(90061.999)  # 25 hours, 1 min, 1.999 sec
        assert "25:01:01" in result, f"Large time should work, got {result}"

        # Test millisecond precision (allow for floating-point rounding)
        # 1.5 seconds should produce ,500
        result = format_srt_timestamp(1.5)
        assert result == "00:00:01,500", f"Expected 500ms, got {result}"

        # Test half-second boundary
        result = format_srt_timestamp(2.25)
        assert result == "00:00:02,250", f"Expected 250ms, got {result}"

        # Test exactly on second boundary
        result = format_srt_timestamp(60.0)
        assert result == "00:01:00,000", f"Expected exact minute, got {result}"

        # Test hour boundary
        result = format_srt_timestamp(3600.0)
        assert result == "01:00:00,000", f"Expected exact hour, got {result}"

        return TestResult(
            name="srt_edge_cases",
            passed=True,
            message="SRT edge cases handled correctly",
        )

    except AssertionError as e:
        return TestResult(
            name="srt_edge_cases",
            passed=False,
            message=f"Assertion failed: {e}",
        )
    except Exception as e:
        return TestResult(
            name="srt_edge_cases",
            passed=False,
            message=f"Exception: {e}",
        )


def test_path_validation() -> TestResult:
    """Test path validation for audio module."""
    from nemoscribe.audio import validate_media_path

    try:
        # Test empty path
        try:
            validate_media_path("")
            assert False, "Empty path should raise ValueError"
        except ValueError:
            pass  # Expected

        # Test whitespace-only path
        try:
            validate_media_path("   ")
            assert False, "Whitespace path should raise ValueError"
        except ValueError:
            pass  # Expected

        # Test non-existent file
        try:
            validate_media_path("/nonexistent/path/to/file.mp4", must_exist=True)
            assert False, "Non-existent file should raise FileNotFoundError"
        except FileNotFoundError:
            pass  # Expected

        # Test with must_exist=False (path doesn't need to exist)
        result = validate_media_path("/some/path/file.mp4", must_exist=False)
        assert result is not None, "Path should be returned when must_exist=False"

        # Test valid path (use temp file)
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            temp_path = f.name
            f.write(b"test")

        try:
            result = validate_media_path(temp_path, must_exist=True)
            assert result.exists(), "Resolved path should exist"
            assert result.is_absolute(), "Resolved path should be absolute"
        finally:
            Path(temp_path).unlink()  # Cleanup

        return TestResult(
            name="path_validation",
            passed=True,
            message="Path validation works correctly",
        )

    except AssertionError as e:
        return TestResult(
            name="path_validation",
            passed=False,
            message=f"Assertion failed: {e}",
        )
    except Exception as e:
        return TestResult(
            name="path_validation",
            passed=False,
            message=f"Exception: {e}",
        )


def test_cli_config_override() -> TestResult:
    """Test CLI configuration override parsing."""
    from nemoscribe import VideoToSRTConfig
    from nemoscribe.cli import parse_args

    try:
        # Test basic string override
        cfg = VideoToSRTConfig()
        cfg = parse_args(["video_path=test.mp4"], cfg)
        assert cfg.video_path == "test.mp4", f"Expected test.mp4, got {cfg.video_path}"

        # Test boolean override (true)
        cfg = VideoToSRTConfig()
        cfg = parse_args(["vad.enabled=true"], cfg)
        assert cfg.vad.enabled is True, f"Expected True, got {cfg.vad.enabled}"

        # Test boolean override (false)
        cfg = VideoToSRTConfig()
        cfg = parse_args(["vad.enabled=false"], cfg)
        assert cfg.vad.enabled is False, f"Expected False, got {cfg.vad.enabled}"

        # Test numeric float override
        cfg = VideoToSRTConfig()
        cfg = parse_args(["vad.onset=0.3"], cfg)
        assert cfg.vad.onset == 0.3, f"Expected 0.3, got {cfg.vad.onset}"

        # Test numeric int override
        cfg = VideoToSRTConfig()
        cfg = parse_args(["audio.max_chunk_duration=120"], cfg)
        assert cfg.audio.max_chunk_duration == 120, f"Expected 120, got {cfg.audio.max_chunk_duration}"

        # Test null value
        cfg = VideoToSRTConfig()
        cfg = parse_args(["subtitle.word_gap_threshold=null"], cfg)
        assert cfg.subtitle.word_gap_threshold is None, f"Expected None, got {cfg.subtitle.word_gap_threshold}"

        # Test multiple overrides
        cfg = VideoToSRTConfig()
        cfg = parse_args(["video_path=video.mp4", "vad.enabled=true", "vad.onset=0.2"], cfg)
        assert cfg.video_path == "video.mp4"
        assert cfg.vad.enabled is True
        assert cfg.vad.onset == 0.2

        return TestResult(
            name="cli_config_override",
            passed=True,
            message="CLI config overrides parsed correctly",
        )

    except AssertionError as e:
        return TestResult(
            name="cli_config_override",
            passed=False,
            message=f"Assertion failed: {e}",
        )
    except Exception as e:
        return TestResult(
            name="cli_config_override",
            passed=False,
            message=f"Exception: {e}",
        )


def test_full_config() -> TestResult:
    """Test full VideoToSRTConfig with all sub-configs."""
    from nemoscribe import VideoToSRTConfig

    try:
        # Create config with all features enabled
        cfg = VideoToSRTConfig()
        cfg.vad.enabled = True
        cfg.postprocessing.enable_itn = True
        cfg.performance.calculate_rtfx = True
        cfg.audio.smart_segmentation = True

        # Verify all sub-configs exist
        assert hasattr(cfg, "vad"), "Missing vad config"
        assert hasattr(cfg, "postprocessing"), "Missing postprocessing config"
        assert hasattr(cfg, "decoding"), "Missing decoding config"
        assert hasattr(cfg, "performance"), "Missing performance config"
        assert hasattr(cfg, "audio"), "Missing audio config"
        assert hasattr(cfg, "subtitle"), "Missing subtitle config"

        # Verify nested settings
        assert cfg.vad.enabled
        assert cfg.postprocessing.enable_itn
        assert cfg.performance.calculate_rtfx
        assert cfg.audio.smart_segmentation

        return TestResult(
            name="full_config",
            passed=True,
            message="Full VideoToSRTConfig with all features works correctly",
        )

    except AssertionError as e:
        return TestResult(
            name="full_config",
            passed=False,
            message=f"Assertion failed: {e}",
        )
    except Exception as e:
        return TestResult(
            name="full_config",
            passed=False,
            message=f"Exception: {e}",
        )


def run_all_tests() -> List[TestResult]:
    """Run all tests and return results."""
    tests = [
        ("baseline_config", test_baseline_config),
        ("vad_config", test_vad_config),
        ("itn_functions", test_itn_functions),
        ("decoding_config", test_decoding_config),
        ("smart_segmentation", test_smart_segmentation),
        ("segment_merging", test_segment_merging),
        ("performance_config", test_performance_config),
        ("quality_metrics", test_quality_metrics),
        ("srt_formatting", test_srt_formatting),
        ("srt_edge_cases", test_srt_edge_cases),
        ("path_validation", test_path_validation),
        ("cli_config_override", test_cli_config_override),
        ("full_config", test_full_config),
    ]

    results = []
    for name, test_func in tests:
        print(f"Running test: {name}...")
        try:
            result = test_func()
            results.append(result)
            status = "[PASS]" if result.passed else "[FAIL]"
            print(f"  {status}: {result.message}")
        except Exception as e:
            results.append(TestResult(name=name, passed=False, message=f"Unexpected error: {e}"))
            print(f"  [FAIL]: Unexpected error: {e}")

    return results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Test NemoScribe improvements")
    parser.add_argument(
        "--test",
        type=str,
        help="Run specific test (vad, itn, decoding, segmentation, metrics, all)",
        default="all",
    )
    parser.add_argument(
        "--audio",
        type=str,
        help="Path to test audio file for integration tests",
        default=None,
    )
    args = parser.parse_args()

    # Check imports first
    imports_ok, msg = check_imports()
    if not imports_ok:
        print(f"Import check failed: {msg}")
        print("Please run tests using: uv run python test_improvements.py")
        sys.exit(1)

    print("=" * 60)
    print("NemoScribe Improvements Integration Tests")
    print("=" * 60)
    print()

    # Run tests
    if args.test == "all":
        results = run_all_tests()
    else:
        test_map = {
            "baseline": test_baseline_config,
            "vad": test_vad_config,
            "itn": test_itn_functions,
            "decoding": test_decoding_config,
            "segmentation": test_smart_segmentation,
            "merging": test_segment_merging,
            "performance": test_performance_config,
            "metrics": test_quality_metrics,
            "srt": test_srt_formatting,
            "srt_edge": test_srt_edge_cases,
            "path": test_path_validation,
            "cli": test_cli_config_override,
            "full": test_full_config,
        }

        if args.test not in test_map:
            print(f"Unknown test: {args.test}")
            print(f"Available: {', '.join(test_map.keys())}")
            sys.exit(1)

        result = test_map[args.test]()
        results = [result]

    # Summary
    print()
    print("=" * 60)
    print("Summary")
    print("=" * 60)

    passed = sum(1 for r in results if r.passed)
    failed = sum(1 for r in results if not r.passed)

    print(f"Total: {len(results)} | Passed: {passed} | Failed: {failed}")
    print()

    if failed > 0:
        print("Failed tests:")
        for r in results:
            if not r.passed:
                print(f"  - {r.name}: {r.message}")
        sys.exit(1)
    else:
        print("All tests passed!")
        sys.exit(0)


if __name__ == "__main__":
    main()
