# Parameter Benchmark Design

Date: 2025-12-14

## Goal

Find optimal VAD parameters for drama/movie transcription by testing different onset/offset combinations and evaluating against human-made subtitles.

## Test Configuration

### Variable Parameters (9 combinations)
- `vad.onset`: 0.1, 0.2, 0.3
- `vad.offset`: 0.1, 0.2, 0.3

### Fixed Parameters (current best settings)
```
vad.enabled=true
vad.model="vad_multilingual_frame_marblenet"
vad.min_duration_on=0.1
vad.filter_speech_first=false
audio.max_chunk_duration=60
audio.smart_segmentation=true
decoding.rnnt_timestamp_type="segment"
```

### Test Material
- Video: `C:\Users\charles\Videos\Captures\Chicago.Fire.S12E01.1080p.WEB.h264-ETHEL[EZTVx.to].mkv`
- Reference: `C:\Users\charles\Videos\Captures\Chicago.Fire.S12E01.1080p.WEB.h264-ETHEL.ENG.srt`

## File Structure

```
C:\Users\charles\Videos\Captures\param_test\
├── onset0.1_offset0.1.srt
├── onset0.1_offset0.2.srt
├── onset0.1_offset0.3.srt
├── onset0.2_offset0.1.srt
├── onset0.2_offset0.2.srt
├── onset0.2_offset0.3.srt
├── onset0.3_offset0.1.srt
├── onset0.3_offset0.2.srt
├── onset0.3_offset0.3.srt
└── evaluation_report.txt
```

## Implementation

### 1. run_benchmark.bat
- Location: `C:\Users\charles\Videos\Captures\param_test\run_benchmark.bat`
- Iterates through onset/offset combinations
- Calls `uv run nemoscribe` for each combination
- Estimated runtime: ~1 hour (9 tests × 5-8 min each)

### 2. scripts/evaluate_benchmark.py
- Location: `C:\Users\charles\Videos\NemoScribe\scripts\evaluate_benchmark.py`
- Parses SRT files
- Calculates WER using `jiwer` library
- Analyzes timestamp offsets
- Generates ranked evaluation report

## Evaluation Metrics

### Word Error Rate (WER)
- Concatenates all subtitle text
- Computes insertions, deletions, substitutions
- Lower is better

### Timestamp Offset
- Aligns sentences using text similarity
- Measures start/end time differences
- Reports: mean, max, standard deviation

### Combined Score
```
score = (1 - WER) * 0.7 + timing_accuracy * 0.3
```

## Usage

```batch
# Step 1: Run benchmark (from param_test directory)
run_benchmark.bat

# Step 2: Evaluate results (from project directory)
cd C:\Users\charles\Videos\NemoScribe
uv run python scripts/evaluate_benchmark.py
```
