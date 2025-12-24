# NemoScribe

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![CUDA](https://img.shields.io/badge/CUDA-13.0-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![GitHub stars](https://img.shields.io/github/stars/charles1018/NemoScribe?style=social)](https://github.com/charles1018/NemoScribe)

[English](README.md) | **繁體中文**

使用 NVIDIA NeMo ASR 模型將影片檔案轉換為 SRT 字幕，支援精確的字詞級時間戳記。透過分段推論可處理長達 3 小時的音訊。

基於 [NVIDIA NeMo](https://github.com/NVIDIA/NeMo) 框架建構，預設使用 [Parakeet-TDT](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2) 模型。

## 目錄

- [功能特色](#功能特色)
- [系統需求](#系統需求)
- [安裝步驟](#安裝步驟)
- [快速開始](#快速開始)
- [使用範例](#使用範例)
- [設定參考](#設定參考)
- [建議模型](#建議模型)
- [長音訊支援](#長音訊支援)
- [疑難排解](#疑難排解)
- [貢獻指南](#貢獻指南)
- [授權條款](#授權條款)
- [致謝](#致謝)
- [參考資源](#參考資源)

## 功能特色

- **精確時間戳記**：從 NeMo ASR 模型取得字詞級與片段級時間戳記
- **長音訊支援**：透過自動分段處理長達 3 小時的影片
- **語音活動偵測 (VAD)**：過濾非語音內容以減少幻覺
- **智慧分段**：在靜音處分割音訊，而非語音中間
- **逆文字正規化 (ITN)**：將口語形式轉換為書寫形式（「twenty five」→「25」）
- **CUDA 最佳化**：預設啟用 CUDA graphs 以加速推論
- **批次處理**：處理整個目錄的影片檔案

## 系統需求

| 需求 | 說明 |
|------|------|
| **作業系統** | Windows 10/11、Linux |
| **Python** | 3.10+（建議使用 3.12，避免 3.13）|
| **套件管理器** | [uv](https://docs.astral.sh/uv/)（建議）|
| **CUDA Toolkit** | 預設 cu130（13.0）。PyTorch 亦支援 12.6/12.8。|
| **FFmpeg** | 音訊擷取必備 |
| **硬體** | NVIDIA GPU 搭配 CUDA（建議）|

### FFmpeg 安裝

- **Windows**：從 [gyan.dev](https://www.gyan.dev/ffmpeg/builds/) 下載，解壓縮後將 `bin` 資料夾加入 PATH
- **Linux**：`sudo apt install ffmpeg`

## 安裝步驟

### 1. 安裝 uv

```powershell
# Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

```bash
# Linux/macOS
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. 複製儲存庫

```bash
git clone https://github.com/charles1018/NemoScribe.git
cd NemoScribe
```

### 3. 安裝相依套件

```bash
uv sync --python 3.12
```

### 4. 設定 CUDA（強烈建議）

預設情況下，`uv sync` 可能會安裝僅支援 CPU 的 PyTorch。**強烈建議啟用 GPU 加速**以獲得合理的轉錄速度。本專案已預先設定使用 CUDA 13.0，GPU 使用者只需執行 `uv sync` 即可。

> **注意**：PyTorch 官方支援 CUDA 12.6、12.8 和 13.0。詳情請參考 [PyTorch Get Started](https://pytorch.org/get-started/locally/)。

如需使用不同的 CUDA 版本，請修改 `pyproject.toml`：

**CUDA 13.0（預設，建議）：**
```toml
[[tool.uv.index]]
name = "pytorch"
url = "https://download.pytorch.org/whl/cu130"
explicit = true

[tool.uv.sources]
torch = { index = "pytorch" }
torchvision = { index = "pytorch" }
torchaudio = { index = "pytorch" }
```

**CUDA 12.8：**
```toml
[[tool.uv.index]]
name = "pytorch"
url = "https://download.pytorch.org/whl/cu128"
explicit = true
```

**CUDA 12.6：**
```toml
[[tool.uv.index]]
name = "pytorch"
url = "https://download.pytorch.org/whl/cu126"
explicit = true
```

然後重新同步：
```bash
uv sync
```

### 5. 驗證設定

```bash
uv run python scripts/check_cuda.py
# 預期輸出：CUDA available: True
```

## 快速開始

```bash
# 基本用法
uv run nemoscribe video_path="video.mp4"

# 啟用 VAD（建議以獲得更好的品質）
uv run nemoscribe video_path="video.mp4" vad.enabled=true

# 批次處理
uv run nemoscribe video_dir=/path/to/videos/ output_dir=/path/to/subtitles/
```

> **📖 進階調校：** 針對不同場景（戲劇、新聞、技術教學）的最佳參數設定，請參考 [TUNING_GUIDE.md](docs/TUNING_GUIDE.md)。

## 使用範例

### 字幕格式設定

```bash
uv run nemoscribe video_path=video.mp4 \
  subtitle.max_chars_per_line=32 \
  subtitle.max_segment_duration=3.0 \
  subtitle.word_gap_threshold=0.5

# 停用字詞間隙分割
uv run nemoscribe video_path=video.mp4 subtitle.word_gap_threshold=null
```

### 裝置與精度設定

```bash
# 強制使用 CPU
uv run nemoscribe video_path=video.mp4 cuda=-1

# 指定 GPU
uv run nemoscribe video_path=video.mp4 cuda=0

# 強制使用 float32 精度
uv run nemoscribe video_path=video.mp4 compute_dtype=float32
```

### VAD 設定

```bash
# 啟用 VAD 與智慧分段
uv run nemoscribe video_path=video.mp4 \
  vad.enabled=true \
  audio.smart_segmentation=true

# 調整 VAD 靈敏度（戲劇/電影最佳化設定）
uv run nemoscribe video_path=video.mp4 \
  vad.enabled=true \
  vad.onset=0.2 \
  vad.offset=0.1 \
  vad.min_duration_off=0.05 \
  vad.pad_onset=0.1 \
  vad.pad_offset=0.1
```

### ITN（逆文字正規化）

```bash
# 啟用 ITN（需要 nemo_text_processing）
uv run nemoscribe video_path=video.mp4 postprocessing.enable_itn=true

# 針對有自動大寫的模型
uv run nemoscribe video_path=video.mp4 \
  postprocessing.enable_itn=true \
  postprocessing.itn_input_case=cased

# 安裝 ITN 相依套件
uv add nemo_text_processing
```

**ITN 範例：**
- `"twenty five dollars"` → `"$25"`
- `"january first twenty twenty five"` → `"January 1, 2025"`
- `"three point one four"` → `"3.14"`
- `"the meeting is at ten thirty am"` → `"the meeting is at 10:30 a.m."`

### 效能測量

```bash
uv run nemoscribe video_path=video.mp4 performance.calculate_rtfx=true
# 範例輸出：RTFx=15.2x realtime (transcribed 600s in 39.5s)
```

## 設定參考

### 主要選項

| 選項 | 預設值 | 說明 |
|------|--------|------|
| `video_path` | - | 輸入影片檔案路徑 |
| `video_dir` | - | 包含影片的目錄路徑 |
| `output_path` | 自動 | 輸出 SRT 檔案路徑 |
| `output_dir` | 自動 | 批次處理的輸出目錄 |
| `pretrained_name` | `nvidia/parakeet-tdt-0.6b-v2` | 預訓練 ASR 模型 |
| `model_path` | - | 本機 .nemo 檢查點路徑 |
| `cuda` | 自動 | CUDA 裝置 ID（None=自動，負數=CPU）|
| `compute_dtype` | 自動 | `float32`、`bfloat16` 或 `float16` |
| `overwrite` | true | 覆寫現有 SRT 檔案 |

### 字幕格式設定 (`subtitle.*`)

| 選項 | 預設值 | 說明 |
|------|--------|------|
| `max_chars_per_line` | 42 | 每行字幕最大字元數 |
| `max_segment_duration` | 5.0 | 每個字幕片段最大秒數 |
| `word_gap_threshold` | 0.8 | 字詞間隙 >= 此值時建立新片段（秒）|

### 音訊處理 (`audio.*`)

| 選項 | 預設值 | 說明 |
|------|--------|------|
| `sample_rate` | 16000 | ASR 音訊取樣率 |
| `max_chunk_duration` | 300.0 | 最大分段大小（5 分鐘，適合 8GB GPU）|
| `chunk_overlap` | 2.0 | 分段間重疊（秒）|
| `smart_segmentation` | true | 使用基於 VAD 的最佳分割點 |
| `min_silence_for_split` | 0.3 | 分割點的最小靜音時長 |
| `prefer_longer_silence` | true | 優先在較長的靜音處分割 |

### VAD 設定 (`vad.*`)

| 選項 | 預設值 | 說明 |
|------|--------|------|
| `enabled` | false | 啟用語音活動偵測 |
| `model` | `vad_multilingual_frame_marblenet` | VAD 模型名稱 |
| `onset` | 0.3 | 語音偵測起始門檻值 (0-1) |
| `offset` | 0.3 | 語音偵測結束門檻值 (0-1) |
| `pad_onset` | 0.2 | 語音片段前 padding（秒）|
| `pad_offset` | 0.2 | 語音片段後 padding（秒）|
| `min_duration_on` | 0.2 | 最小語音片段時長 |
| `min_duration_off` | 0.2 | 合併的最小非語音間隙 |

### 解碼最佳化 (`decoding.*`)

| 選項 | 預設值 | 說明 |
|------|--------|------|
| `rnnt_fused_batch_size` | -1 | CUDA graphs：-1=啟用，0=停用 |
| `rnnt_timestamp_type` | "all" | 時間戳記類型："char"、"word"、"segment"、"all" |
| `ctc_timestamp_type` | "all" | CTC 時間戳記類型 |

### 後處理 (`postprocessing.*`)

| 選項 | 預設值 | 說明 |
|------|--------|------|
| `enable_itn` | false | 啟用逆文字正規化 |
| `itn_lang` | "en" | ITN 語言 |
| `itn_input_case` | "lower_cased" | 輸入大小寫："lower_cased" 或 "cased" |

### 效能 (`performance.*`)

| 選項 | 預設值 | 說明 |
|------|--------|------|
| `calculate_rtfx` | false | 計算即時係數 (RTFx) |
| `warmup_steps` | 1 | 計時前的暖機迭代次數 |

### 日誌 (`logging.*`)

| 選項 | 預設值 | 說明 |
|------|--------|------|
| `verbose` | false | 顯示所有 NeMo 內部日誌（除錯用）|
| `suppress_repetitive_logs` | true | 在分段處理期間抑制重複的 NeMo 日誌 |

## 建議模型

| 模型 | 速度 | 準確度 | 特色 |
|------|------|--------|------|
| `nvidia/parakeet-tdt-0.6b-v2` | 快 | **最佳（英文）** | **預設**。WER 1.69%，自動標點 |
| `nvidia/parakeet-tdt-0.6b-v3` | 快 | 優秀 | 多語言（25 種語言），自動語言偵測 |
| `nvidia/parakeet-tdt-1.1b` | 中等 | 最佳 | 最高準確度，無自動標點 |
| `nvidia/parakeet-ctc-1.1b` | 最快 | 良好 | 推論速度最快 |
| `nvidia/canary-1b-v2` | 中等 | 良好 | 多語言，支援翻譯 |

### 模型選擇指南

- **英文字幕**：`parakeet-tdt-0.6b-v2`（預設，開箱即用體驗最佳）
- **多語言**：`parakeet-tdt-0.6b-v3`（25 種語言，自動偵測）
- **最高準確度**：`parakeet-tdt-1.1b`（最低 WER，但無標點）
- **最快速度**：`parakeet-ctc-1.1b`
- **翻譯功能**：`canary-1b-v2`（25 種語言，轉錄 + 翻譯）

> **注意**：`parakeet-tdt-1.1b` 產生的輸出為無標點的小寫文字。程式會自動使用字詞級時間戳記來產生細緻的字幕。

## 長音訊支援

程式使用**音訊分段**來處理任意長度的影片：

- 自動將長音訊分割成較小的分段（預設：5 分鐘）
- 分段間重疊（預設：2 秒）以確保邊界準確
- 自動合併所有分段的字幕，處理重複內容
- 長音訊注意力調校由 `audio.long_audio_threshold` 控制（預設停用；調低數值即可啟用）

**GPU 記憶體建議：**

| GPU VRAM | `max_chunk_duration` |
|----------|---------------------|
| 8GB | 300（預設）|
| 16GB | 600 |
| 24GB+ | 0（不分段）|

## 時間戳記優先順序

程式按以下優先順序取得時間戳記：

1. **片段級**：直接從模型取得片段時間戳記（最準確）
2. **字詞級**：依行長/時長/間隙分組的字詞時間戳記
3. **備援**：當無時間戳記時，依語速估算（約 150 字/分鐘）

> **自動備援**：若平均片段長度超過 `max_segment_duration * 2`（例如無標點的模型），程式會自動切換至字詞級時間戳記。

## 專案結構

```
nemoscribe/
├── __init__.py        # 套件入口，版本資訊
├── __main__.py        # python -m nemoscribe 支援
├── cli.py             # CLI 解析與入口點
├── config.py          # 所有 dataclass 設定
├── audio.py           # 音訊處理 (ffmpeg)
├── vad.py             # 語音活動偵測
├── transcriber.py     # ASR 模型與轉錄
├── srt.py             # SRT 格式化與輸出
├── postprocess.py     # ITN、片段合併
└── log_utils.py       # 日誌過濾
```

## 支援的影片格式

`.mp4`、`.mkv`、`.avi`、`.mov`、`.webm`、`.m4v`

## 輸出範例

```srt
1
00:00:00,120 --> 00:00:03,450
Welcome to our show today.

2
00:00:03,680 --> 00:00:07,200
We have an exciting episode planned for you.

3
00:00:07,450 --> 00:00:11,800
Let's get started with our first topic.
```

## 測試

```bash
# 執行所有測試
uv run python tests/test_improvements.py

# 執行特定測試
uv run python tests/test_improvements.py --test vad
uv run python tests/test_improvements.py --test itn
uv run python tests/test_improvements.py --test segmentation
uv run python tests/test_improvements.py --test metrics

# 可用測試：baseline, vad, itn, decoding, segmentation, merging, performance, metrics, srt, full
```

### 測試涵蓋範圍

- **baseline_config**：預設設定的向下相容性
- **vad_config**：VAD 設定正確性
- **itn_functions**：ITN 正規化功能
- **decoding_config**：解碼設定（CUDA graphs）
- **smart_segmentation**：智慧分段邏輯
- **segment_merging**：重疊片段合併
- **performance_config**：效能設定
- **quality_metrics**：WER/CER 計算
- **srt_formatting**：SRT 格式化
- **full_config**：完整設定組合

## 品質指標

使用 NeMo 官方工具計算轉錄品質：

```python
from tests.test_improvements import calculate_transcription_quality

result = calculate_transcription_quality(
    hypothesis="transcribed text",
    reference="ground truth text"
)
print(f"WER: {result['wer']:.2%}")
print(f"CER: {result['cer']:.2%}")
```

輸出包含：`wer`、`cer`、`insertion_rate`、`deletion_rate`、`substitution_rate`

## 疑難排解

### CUDA 記憶體不足

減少分段大小：
```bash
uv run nemoscribe video_path=video.mp4 audio.max_chunk_duration=180.0
```

### 時間戳記不準確

使用支援時間戳記的模型（建議 `parakeet-tdt-*`）並調整分段參數：
```bash
uv run nemoscribe video_path=video.mp4 \
  subtitle.max_segment_duration=3.0 \
  subtitle.word_gap_threshold=0.5
```

### 模型下載緩慢

模型會在首次使用時自動從 HuggingFace/NGC 下載。對於網路較慢的情況：
```bash
# 使用 HuggingFace 鏡像（中國大陸）
export HF_ENDPOINT=https://hf-mirror.com
```

## 貢獻指南

歡迎貢獻！請隨時提交 Pull Request。

1. 在 [github.com/charles1018/NemoScribe](https://github.com/charles1018/NemoScribe) Fork 此儲存庫
2. 建立您的功能分支（`git checkout -b feature/amazing-feature`）
3. 提交您的變更（`git commit -m 'Add some amazing feature'`）
4. 推送到分支（`git push origin feature/amazing-feature`）
5. 開啟 Pull Request

如需回報問題或提出功能建議，請[開啟 Issue](https://github.com/charles1018/NemoScribe/issues)。

## 授權條款

本專案採用 MIT 授權條款 - 詳見 [LICENSE](LICENSE) 檔案。

## 致謝

NemoScribe 建構於以下開源專案之上：

- **[NVIDIA NeMo](https://github.com/NVIDIA/NeMo)** - 對話式 AI 的神經模組工具包（Apache 2.0 授權）
- **[Parakeet-TDT](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2)** - NVIDIA 最先進的 ASR 模型（CC-BY-4.0 授權）

感謝 NVIDIA 將這些優秀的工具和模型提供給社群。

## 參考資源

### 模型資源

| 資源 | 說明 |
|------|------|
| [nvidia/parakeet-tdt-0.6b-v2](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2) | 預設模型，架構與最佳實踐 |
| [nvidia/parakeet-tdt-0.6b-v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) | 多語言版本，25 種語言 |
| [nvidia/canary-1b-v2](https://huggingface.co/nvidia/canary-1b-v2) | 多語言並支援翻譯 |
| [HuggingFace Space 示範](https://huggingface.co/spaces/nvidia/parakeet-tdt-0.6b-v2) | 官方示範，包含長音訊處理 |

### NeMo 框架參考

| 檔案路徑 | 說明 |
|----------|------|
| `examples/asr/transcribe_speech.py` | 主要架構參考 |
| `nemo/collections/asr/parts/utils/transcribe_utils.py` | 核心工具：`get_inference_device()`、`get_inference_dtype()` |
| `nemo/collections/asr/parts/utils/rnnt_utils.py` | `Hypothesis` 類別，時間戳記資料結構 |

### 關鍵實作細節

**長音訊最佳化**（來自 HuggingFace Space）：
```python
# 對於 >8 分鐘的音訊，切換到局部注意力以提高記憶體效率
model.change_attention_model("rel_pos_local_attn", [256, 256])
model.change_subsampling_conv_chunking_factor(1)  # 1 = 自動選擇
```

**時間戳記資料結構**（來自 `Hypothesis`）：
```python
{
    'segment': [{'start': float, 'end': float, 'segment': str}, ...],
    'word': [{'start': float, 'end': float, 'word': str}, ...],
    'char': [...]  # 字元級時間戳記
}
```

### 文件

- [NeMo ASR 文件](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/asr/intro.html)
- [NeMo GitHub 儲存庫](https://github.com/NVIDIA/NeMo)
- [Parakeet 模型卡](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/nemo/models/parakeet-tdt-0.6b-v2)
