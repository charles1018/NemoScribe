# VAD Long Segment Analysis

**Date**: 2025-12-24
**Test Video**: Chicago.Fire.S12E01.1080p.WEB.h264-ETHEL[EZTVx.to].mkv (41.8 minutes)

## Problem Statement

When enabling VAD for drama/movie transcription, segments became excessively long (up to 46.96 seconds), containing multiple rapid dialogue exchanges that should be separate subtitles.

## Root Cause Analysis

### Two-Layer Problem

1. **VAD Layer - Gap Merging**
   - **Parameter**: `vad.min_duration_off` (default: 0.2s)
   - **Behavior**: Merges non-speech gaps shorter than threshold
   - **Impact**: Short pauses between dialogues (0.1-0.2s) are converted to speech, merging separate dialogues

2. **ASR Layer - Segment Timestamping**
   - **Parameter**: `decoding.rnnt_timestamp_type="segment"`
   - **Behavior**: Treats entire VAD speech segments as single units
   - **Impact**: Doesn't split on sentence boundaries (periods, question marks, exclamation marks)

### Example from Analysis

**30.48s segment** (chunk #473) contained:
```
"years ago gibbs nowadays he can barely make it to his mailbox" |
"why two ambush" |
"oh this one's a temporary relocation" |
"why from where" |
"hey where were you guys based before 17" |
"we were sent here after that fire shut the place down for renovations" |
"Son of a bitch, the package was meant for them."
```

7+ dialogue exchanges merged into one subtitle.

## Testing Results

| Configuration | Max Duration | Segments >6s | Segments >10s | Conclusion |
|--------------|--------------|--------------|---------------|------------|
| Baseline (no VAD) | 11.28s | 20 | 2 | ✅ Best, but no noise filtering |
| Drama default (VAD) | 46.96s | 29 | 4 | ❌ Worst - gap merging issue |
| min_duration_off=0.05 | 30.48s | 27 | 3 | ✅ **Selected** - 35% improvement |
| min_duration_off=0.01 | 30.48s | 27 | 3 | No further improvement |
| word timestamps | 31.25s | 68 | 25 | ❌ Worse - more long segments |

## Solution (Option 3 - Accept Current State)

### Optimized Parameters

```bash
vad.enabled=true
vad.model="vad_multilingual_frame_marblenet"
vad.onset=0.2                     # From benchmark (best WER)
vad.offset=0.1                    # From benchmark
vad.min_duration_on=0.1           # Keep short speech
vad.min_duration_off=0.05         # KEY: Preserve short pauses
vad.pad_onset=0.1                 # Reduced from 0.2
vad.pad_offset=0.1                # Reduced from 0.2
vad.filter_speech_first=false
audio.max_chunk_duration=60
audio.smart_segmentation=true
decoding.rnnt_timestamp_type="segment"
```

### Trade-offs

**Advantages**:
- 35% reduction in max segment duration (46.96s → 30.48s)
- Reduced >10s segments from 4 to 3
- VAD effectively filters noise and music
- Speech detection more accurate (455 segments vs over-merged)

**Limitations**:
- Occasional 30s segments remain (rapid dialogue scenes)
- Segments don't split on sentence boundaries within continuous speech
- Acceptable for drama content where dialogue pacing varies

## Alternative Solutions (Not Pursued)

### Option 1: Use Default Timestamp Type
```bash
# Remove decoding.rnnt_timestamp_type="segment"
# Let ASR split on punctuation marks
```

**Pros**: Should split 30s segment into 7+ shorter ones
**Cons**: Untested, may over-segment, unclear if compatible with VAD
**Status**: Not tested, available for future experimentation

### Option 2: Post-Processing Split
Implement logic in `srt.py`:
1. Detect long segments (>10s)
2. Find longest word gaps in segment
3. Split at natural pauses

**Pros**: Full control over splitting logic
**Cons**: Requires development, adds complexity
**Status**: Deferred - current solution acceptable

## Technical Details

### NeMo VAD `filtering()` Function

Located in: `NeMo/nemo/collections/asr/parts/utils/vad_utils.py:612-679`

```python
def filtering(speech_segments, per_args):
    # Step 1: Remove short speech segments (< min_duration_on)
    speech_segments = filter_short_segments(speech_segments, min_duration_on)

    # Step 2: Find gaps between speech
    non_speech_segments = get_gap_segments(speech_segments)

    # Step 3: Find short gaps (< min_duration_off)
    short_non_speech = filter_short_segments(non_speech_segments, min_duration_off)

    # Step 4: CONVERT SHORT GAPS TO SPEECH (causing merge!)
    speech_segments = torch.cat((speech_segments, short_non_speech), 0)

    # Step 5: Merge overlapping segments
    speech_segments = merge_overlap_segment(speech_segments)
```

**Key Insight**: Short gaps are not just ignored—they are **converted to speech**, causing adjacent segments to merge.

### Segment vs Word Timestamps

- **`segment`**: ASR outputs one timestamp per VAD speech segment
  - Pros: Natural speech boundaries
  - Cons: Doesn't split rapid dialogues

- **`word`**: ASR outputs timestamp per word
  - Pros: Precise word timing
  - Cons: Requires word gap detection, produced worse results (68 >6s segments)

- **`all`** (default): Both word and segment timestamps
  - Should theoretically split on punctuation
  - Not tested in this analysis

## Recommendations

1. **For drama/movie content**: Use optimized parameters (min_duration_off=0.05)
2. **Monitor**: Check if 30s segments are acceptable for your use case
3. **Future work**: Test Option 1 (default timestamp type) if stricter segment limits needed
4. **Fallback**: Disable VAD if segment length is critical (baseline: 11.28s max)

## Files

- Analysis log: `.claude/long_segment_attempts.json`
- Test outputs:
  - `Chicago.Fire.S12E01.*.drama.srt` (baseline)
  - `Chicago.Fire.S12E01.*.drama.minoff0.05.srt` (optimized)
  - `Chicago.Fire.S12E01.*.drama.minoff0.01.srt` (over-optimized)
  - `Chicago.Fire.S12E01.*.drama.word.srt` (word timestamps)
- Analysis scripts:
  - `scripts/analyze_srt_stats.py`
  - `scripts/find_longest_segment.py`

## References

- NeMo VAD documentation: `NeMo/examples/asr/conf/vad/frame_vad_infer_postprocess.yaml`
- NeMo VAD utilities: `NeMo/nemo/collections/asr/parts/utils/vad_utils.py`
- VAD+ASR pipeline: `NeMo/examples/asr/asr_vad/speech_to_text_with_vad.py`
