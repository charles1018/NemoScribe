# parakeet-tdt-0.6b-v2 Parameter Tuning Guide (Parakeet-TDT)

**English** | [繁體中文](TUNING_GUIDE.zh-TW.md)

This guide helps you tune the parameters of the **NVIDIA NeMo (Parakeet-TDT)** model to produce AI subtitles that outperform typical off-the-shelf GUI tools.

The recommended parameters below have been verified through real-world testing and effectively address the most common AI subtitle problems: **hallucinations** (text appearing out of nowhere), **misheard names**, and **fragmented sentence splitting**.

> **📋 Prerequisite:** Install NemoScribe first — see [README.md](../README.md).

---

## 📑 Table of Contents

- [⚡ Quick Start (TL;DR)](#-quick-start-tldr)
- [🏆 Core Strategy: Why Tune?](#-core-strategy-why-tune)
- [🎬 Scenario A: Drama & Movies](#-scenario-a-drama--movies)
- [📰 Scenario B: News & Interviews](#-scenario-b-news--interviews)
- [💻 Scenario C: Tech Tutorials](#-scenario-c-tech-tutorials)
- [🔧 Base Config](#-base-config)
- [🤖 LLM Post-processing](#-llm-post-processing)
- [⚡ Batch Processing](#-batch-processing)
- [❓ FAQ](#-faq)

---

## ⚡ Quick Start (TL;DR)

**First time? Generate both VAD and no-VAD candidates:**

```bash
uv run nemoscribe \
    video_path="your_video.mp4" \
    output_path="output.srt" \
    compute_dtype=float32 \
    decoding.rnnt_fused_batch_size=0 \
    decoding.segment_gap_threshold=20 \
    ab_test.vad=true
```

This writes `output.vad.srt` and `output.no_vad.srt`. Pick the version that reads better; for more precise results, follow the scenario-specific settings below based on your content type.

---

## 🏆 Core Strategy: Why Tune?

Typical AI subtitle tools run with "defaults", which is like sifting sand with a coarse sieve. Our strategy is:
1.  **Compare VAD vs no-VAD**: VAD filters noise and reduces hallucinations, but may also cut subtle dialogue; no-VAD can be more complete on cleanly recorded drama/movie audio.
2.  **Shorter chunking**: Let the model "refresh" every minute to stay focused.
3.  **Better timestamp logic**: Force the model to output complete-sentence timings.

---

## 🎬 Scenario A: Drama & Movies
**Best for:** TV series, movies, anime.
**Characteristics:** Complex background audio, fast speech, cries for help / breathy voices / whispers.

### 🏆 Recommended Workflow (compare candidates first)

For drama/movie content, do not assume VAD is always better. Generate both VAD and no-VAD candidates with the same ASR settings, then pick the more natural one:

```bash
uv run nemoscribe \
    video_path="your_video.mkv" \
    output_path="output.srt" \
    compute_dtype=float32 \
    decoding.rnnt_fused_batch_size=0 \
    decoding.segment_gap_threshold=20 \
    ab_test.vad=true
```

This writes:

```text
output.vad.srt
output.no_vad.srt
```

Typical advantages of the VAD version:
- Fewer "out of nowhere" subtitles when there is lots of background audio, music, or silence
- Split points tend to land on silence

Typical advantages of the no-VAD version:
- May preserve more short lines and subtle dialogue on clean recordings
- Avoids missing words caused by VAD misdetection

### VAD Candidate Parameters

If you decide to use VAD, here are candidate VAD parameters for drama/movie content:

```bash
uv run nemoscribe \
    video_path="your_video.mkv" \
    output_path="output.srt" \
    compute_dtype=float32 \
    vad.enabled=true \
    vad.model="vad_multilingual_frame_marblenet" \
    vad.onset=0.2 \
    vad.offset=0.1 \
    vad.min_duration_on=0.1 \
    vad.min_duration_off=0.05 \
    vad.pad_onset=0.1 \
    vad.pad_offset=0.1 \
    vad.filter_speech_first=false \
    audio.max_chunk_duration=60 \
    audio.smart_segmentation=true \
    decoding.rnnt_fused_batch_size=0 \
    decoding.segment_gap_threshold=20
```

This VAD parameter set addresses the following issues:
- **Hallucinations**: the model inventing dialogue during silence or background music
- **Missed speech**: subtle cries for help, whispers, and breathy voices being ignored
- **Fragmented splitting**: one sentence being cut into multiple incomplete subtitles

> **Chicago Fire validation (2026-05-05)**: Tested on `Chicago Fire S12E01` with an RTX 3070 Laptop GPU and NeMo 2.7.3, the stable and reproducible CUDA combination is `compute_dtype=float32` + `decoding.rnnt_fused_batch_size=0`. In this sample, the no-VAD version had slightly more segments and words with a slightly lower rough WER; the VAD version still kept long segments under control and reduced background-audio risk. We therefore recommend most users start with `ab_test.vad=true`.

> **Note**: `decoding.rnnt_timestamp_type="all"` and `decoding.segment_separators=[".", "?", "!"]` are defaults and need no manual setting. In the Chicago Fire test, `segment_gap_threshold=20` further split overly long segments while preserving punctuation-based splitting.

### Parameter Details
| Parameter | Recommended | Why |
| :--- | :--- | :--- |
| `vad.onset` | `0.2` | **Test-verified optimum**. Balances sensitivity and accuracy — lowest WER and most accurate timestamps. |
| `vad.offset` | `0.1` | **Lower end-of-speech threshold**. Ensures complete sentence endings are captured. |
| `vad.min_duration_on` | `0.1` | Keeps very short utterances; below 0.1s is usually noise. |
| `vad.min_duration_off` | `0.05` | **Prevents dialogue merging**. The default 0.2s merges short pauses (e.g. breaths between lines), creating 40+ second mega-segments. Lowering to 0.05 improves this by 35%. |
| `vad.pad_onset` | `0.1` | Reduced from the default 0.2 to cut leading padding and avoid overlap. |
| `vad.pad_offset` | `0.1` | Reduced from the default 0.2 to cut trailing padding and avoid overlap. |
| `vad.filter_speech_first` | `false` | **No aggressive filtering**. Avoids deleting dialogue buried in noisy backgrounds. |
| `compute_dtype` | `float32` | **Chicago Fire verified stable value**. More stable than `bfloat16` on this RTX 3070 Laptop GPU. |
| `decoding.rnnt_fused_batch_size` | `0` | **Disables CUDA graphs**. Avoids CUDA illegal memory access during warmup / first-chunk transcription in the Chicago Fire run. |
| `decoding.segment_separators` | `[".", "?", "!"]` | **Punctuation splitting**. Splits segments at sentence boundaries to avoid overly long subtitles. |
| `decoding.segment_gap_threshold` | `20` | **Chicago Fire verified optimum**. Longest subtitle dropped from 30.48s to 12.48s; `15/10` did not reduce it further. |
| `postprocessing.enable_itn` | `false` | Drama dialogue usually doesn't need numbers converted to digits. |

### Before / After

The difference with the recommended parameters:

| Issue | ❌ Before | ✅ After |
| :--- | :--- | :--- |
| Hallucination | "Thank you for watching" appears during silence | Correctly stays silent, no extra text |
| Missed speech | A background "Help!" cry is ignored | Subtle cries are captured |
| Splitting | "I can't" / "believe this" (two fragments) | "I can't believe this." (one complete sentence) |

---

## 📰 Scenario B: News & Interviews
**Best for:** News reports, studio interviews, documentaries.
**Characteristics:** Clean recording, steady speech rate, quiet background.

### Parameter Details
| Parameter | Recommended | Why |
| :--- | :--- | :--- |
| `vad.onset` | `0.5` | **Standard threshold**. Filters out the anchor's breaths and paper shuffling. |
| `vad.min_duration_on` | `0.2` | Sentences are usually complete; no need to catch very short sounds. |
| `vad.filter_speech_first` | `true` | **Enable filtering**. Keeps the main voice cleaner. |
| `postprocessing.enable_itn` | `true` | **Must-have**. Automatically converts "January first" to "Jan 1st". |

**Example command for news/interviews:**
```bash
uv run nemoscribe \
    video_path="your_news_video.mp4" \
    output_path="your_news_subtitle.srt" \
    vad.enabled=true \
    vad.model="vad_multilingual_frame_marblenet" \
    vad.onset=0.5 \
    vad.offset=0.1 \
    vad.min_duration_on=0.2 \
    vad.filter_speech_first=true \
    audio.max_chunk_duration=60 \
    audio.smart_segmentation=true \
    postprocessing.enable_itn=true
```

---

## 💻 Scenario C: Tech Tutorials
**Best for:** Programming tutorials, AI model walkthroughs, software demos.
**Characteristics:** Full of version numbers / parameters / port numbers, pauses while the speaker operates, home-recording environments.

### Core Strategy:
The worst failure mode for tech videos is spelling out "Windows Eleven" as words. This mode focuses on **number formatting** and **preserving thinking pauses**.

| Parameter | Recommended | Why |
| :--- | :--- | :--- |
| `vad.onset` | `0.3` | **Middle ground**. More sensitive than news mode so filler words after a pause (e.g. "uh... so then we...") aren't cut, but not so sensitive that keyboard noise is recorded. |
| `vad.offset` | `0.2` | Slightly delayed cut-off to fit a think-while-talking rhythm. |
| `postprocessing.enable_itn` | `true` | **Absolutely essential!** This is what makes tutorials readable.<br>Effect:<br>❌ "Python three point ten"<br>✅ "Python 3.10"<br>❌ "Port eight thousand eighty"<br>✅ "Port 8080" |
| `audio.max_chunk_duration` | `60` | Keep 60 seconds to avoid drift over long monologues. |

**Example command for tech tutorials:**
```bash
uv run nemoscribe \
    video_path="your_tutorial_video.mp4" \
    output_path="your_tutorial_subtitle.srt" \
    vad.enabled=true \
    vad.model="vad_multilingual_frame_marblenet" \
    vad.onset=0.3 \
    vad.offset=0.2 \
    vad.min_duration_on=0.1 \
    vad.filter_speech_first=true \
    audio.max_chunk_duration=60 \
    audio.smart_segmentation=true \
    postprocessing.enable_itn=true
```

---

## 🔧 Base Config

Whatever the scenario, these parameters are recommended as a baseline:

| Parameter | Recommended | Effect |
| :--- | :--- | :--- |
| `audio.max_chunk_duration` | `60` | Forces a split every 60 seconds to avoid model fatigue. |
| `audio.smart_segmentation` | `true` | Splits intelligently at silence. |
| `decoding.rnnt_timestamp_type` | `"all"` | Outputs all timestamp types (default). Works best with segment_separators. |
| `decoding.segment_separators` | `[".", "?", "!"]` | Splits segments at punctuation (default). **Verified**: reduces long segments from 46.96s to 11.28s. Set to an empty list to disable. |
| `decoding.segment_gap_threshold` | `None` | Gap-based segment splitting (unit: frames, must be a positive integer). Forces a new segment when the gap between two consecutive words exceeds the threshold; when `segment_separators` is also enabled, NemoScribe keeps punctuation splits and applies gap splits on top. |
| `ab_test.vad` | `true` | When unsure whether VAD helps, generate both VAD / no-VAD candidates first. |

---

## 🤖 LLM Post-processing

**When to use:** Names and proper nouns are misrecognized in the subtitles, or you need higher consistency.

ASR models have inherent limits in recognizing names and proper nouns (e.g. hearing "Kylie Estevez" as "Alias of us"). LLM post-processing fixes this class of errors with a large language model.

### Prerequisites

```bash
# Install LLM dependencies
uv sync --extra llm

# Configure the API key
cp .env.example .env
# Edit .env and add: OPENAI_API_KEY=sk-... or ANTHROPIC_API_KEY=sk-ant-...
```

### Recommended Setup

| Provider | Model | Quality | Cost/episode | When to use |
|--------|------|------|---------|----------|
| OpenAI | `gpt-4o-mini` | Good | ~$0.06 | **First choice**: best cost/quality ratio |
| OpenAI | `gpt-4o` | Excellent | ~$0.30 | When you need higher quality |
| Anthropic | `claude-3-5-sonnet-20241022` | Excellent | ~$0.24 | If you prefer Anthropic |

### Usage Example

```bash
# VAD + LLM (full recommended pipeline)
uv run nemoscribe \
    video_path="your_video.mkv" \
    vad.enabled=true \
    vad.onset=0.2 \
    vad.offset=0.1 \
    vad.min_duration_off=0.05 \
    vad.pad_onset=0.1 \
    vad.pad_offset=0.1 \
    llm_postprocess.enabled=true \
    llm_postprocess.provider=openai \
    llm_postprocess.model=gpt-4o-mini
```

### Parameter Reference

| Parameter | Recommended | Description |
| :--- | :--- | :--- |
| `llm_postprocess.enabled` | `true` | Enable LLM correction |
| `llm_postprocess.provider` | `openai` | Provider: `openai` or `anthropic` |
| `llm_postprocess.model` | `gpt-4o-mini` | Model name |
| `llm_postprocess.batch_size` | `20` | Subtitle segments per LLM request. Larger gives more context but is slower |
| `llm_postprocess.max_retries` | `3` | Max retries when validation fails |

### Known Limitations

- **Over-correction**: ~10% of segments may be modified unnecessarily (mostly minor changes)
- **Semantic errors**: LLMs struggle with semantic-level mistakes (e.g. changing "breach" to "bridge")
- **Number drift**: numbers occasionally change (e.g. "thirty seconds" → "40 seconds")
- **Cost**: requires a paid API, but it's cheap (~$0.06/episode with GPT-4o-mini)

### How It Works

1. Subtitles are sent to the LLM in batches (20 segments per batch)
2. The LLM returns corrections in JSON format
3. Correction magnitude is validated (similarity check to prevent over-editing)
4. On validation failure, feedback is provided and the batch is retried (up to 3 times)
5. If anything fails, the original subtitles are used as a graceful fallback

---

## ⚡ Batch Processing

Process a whole folder of videos at once (drama/movie mode shown):

```bash
uv run nemoscribe \
    video_dir="C:\Path\To\Season1" \
    output_dir="C:\Path\To\Subtitles" \
    vad.enabled=true \
    vad.onset=0.2 \
    vad.offset=0.1 \
    vad.min_duration_on=0.1 \
    vad.min_duration_off=0.05 \
    vad.pad_onset=0.1 \
    vad.pad_offset=0.1 \
    vad.filter_speech_first=false \
    audio.max_chunk_duration=60 \
    audio.smart_segmentation=true
```

---

## ❓ FAQ

### Q: How long does a one-hour video take?
**A:** It depends on your GPU. On an RTX 3080, a one-hour video takes roughly 3–5 minutes. Enabling VAD adds a little processing time but greatly improves quality.

### Q: What if I get CUDA out of memory?
**A:** Try a shorter chunk duration:
```bash
audio.max_chunk_duration=30  # down from 60 seconds
```

### Q: Subtitles contain garbled text or strange symbols?
**A:** This is usually an encoding issue. NemoScribe writes SRT files in UTF-8 — make sure your player or editor supports UTF-8.

### Q: Why is some dialogue still missed?
**A:** Try lowering the VAD sensitivity threshold:
```bash
vad.onset=0.15  # from 0.2 down to 0.15, more sensitive
```
Note: going too low (below ~0.1) may pick up noise as speech and degrade timestamp accuracy.

If what's still missing after VAD tuning is **repeated words or false starts** (e.g. stuttering, shouting the same name repeatedly), this is a known limitation of the `parakeet-tdt-0.6b-v2` model itself (a regression vs the 1.1b model, see [official discussion #8](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2/discussions/8)) and cannot be fixed with parameters. If you need verbatim disfluencies, switch to `pretrained_name="nvidia/parakeet-tdt-1.1b"`.

### Q: Can it handle non-English videos?
**A:** The `parakeet-tdt-0.6b-v2` model is optimized for English. For other languages, use:
- `nvidia/parakeet-tdt-0.6b-v3`: 25 languages, automatic language detection
- `nvidia/canary-1b-v2`: 25 languages, with translation support

Usage:
```bash
uv run nemoscribe video_path="video.mp4" pretrained_name="nvidia/parakeet-tdt-0.6b-v3"
```

### Q: Subtitle segments are too long (over 30 seconds)?
**A:** This usually happens in fast-dialogue scenes. Try the following:

1. **Confirm punctuation splitting is enabled** (on by default):
   ```bash
   decoding.segment_separators=".,?,!"
   ```

2. **Lower VAD's min_duration_off** to keep more gaps between lines:
   ```bash
   vad.min_duration_off=0.05  # default 0.2
   ```

3. **Use `segment_gap_threshold` for gap-based splitting**:
   ```bash
   decoding.segment_gap_threshold=20  # split when the inter-word gap exceeds 20 frames
   ```

4. If you only want gap-based splitting without punctuation splits, additionally disable:
   ```bash
   decoding.segment_separators=
   ```

5. If very long segments remain, it's likely continuous rapid dialogue with no silence gaps — which is expected.

### Q: How do I verify CUDA/GPU is working?
**A:** Run:
```bash
uv run python scripts/check_cuda.py
```
