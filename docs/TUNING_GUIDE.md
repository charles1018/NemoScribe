# parakeet-tdt-0.6b-v2 模型最佳化參數指南 (Parakeet-TDT)

這份指南旨在幫助使用者透過調整 **NVIDIA NeMo (Parakeet-TDT)** 模型的參數，產出品質超越一般市面 GUI 工具的 AI 字幕。

經過實測驗證，使用本指南建議的參數，能有效解決 AI 字幕常見的**幻覺（無中生有）**、**人名聽錯**以及**斷句破碎**等問題。

> **📋 前置條件：** 使用前請先完成 NemoScribe 安裝，詳見 [README.md](../README.md)。

---

## 📑 目錄

- [⚡ 快速開始](#-快速開始懶人包)
- [🏆 核心策略](#-核心策略為什麼需要調整)
- [🎬 場景 A：戲劇與電影](#-場景-a戲劇與電影-dramamovie)
- [📰 場景 B：新聞與訪談](#-場景-b新聞與訪談-newsinterview)
- [💻 場景 C：技術教學與解說](#-場景-c技術教學與解說-techtutorial)
- [🔧 通用基礎參數](#-通用基礎參數-base-config)
- [🤖 LLM 後處理](#-llm-後處理-llm-post-processing)
- [⚡ 批次處理指令](#-批次處理指令-batch-processing)
- [❓ 常見問題 FAQ](#-常見問題-faq)

---

## ⚡ 快速開始（懶人包）

**第一次使用？直接複製這個指令：**

```bash
uv run nemoscribe \
    video_path="您的影片.mp4" \
    output_path="輸出字幕.srt" \
    vad.enabled=true \
    vad.onset=0.2 \
    vad.offset=0.1 \
    audio.max_chunk_duration=60 \
    audio.smart_segmentation=true
```

想要更精確的結果？請根據您的影片類型，參考下方的場景設定。

---

## 🏆 核心策略：為什麼需要調整？

一般的 AI 字幕工具通常使用「預設值」，這就像是用一把粗糙的篩子去濾沙。我們的策略是：
1.  **開啟 VAD（語音偵測）**：依據場景調整靈敏度，過濾雜訊或保留細節。
2.  **縮短切分長度 (Chunking)**：讓 AI 每分鐘「刷新」一次大腦，保持專注力。
3.  **優化時間軸邏輯**：強制 AI 輸出完整的句子時間點。

---

## 🎬 場景 A：戲劇與電影 (Drama/Movie)
**適用：** 美劇、電影、動漫。
**特徵：** 背景音複雜、語速快、有呼救聲/氣音/低語。

### 🏆 推薦參數（實測驗證最佳解）

這是針對戲劇/電影場景經過多次實測驗證的最佳參數組合，能有效解決以下問題：
- **幻覺問題**：AI 在靜音或背景音樂處「無中生有」產生不存在的對話
- **遺漏問題**：細微的呼救聲、低語、氣音被忽略
- **斷句破碎**：一句話被切成多段不完整的字幕

**這是處理影劇類影片最常用的推薦設定。**

```bash
uv run nemoscribe \
    video_path="您的影片.mkv" \
    output_path="輸出字幕.srt" \
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

> **Chicago Fire 實測（2026-04-04）**：以 `Chicago Fire S12E01`、RTX 3070 Laptop GPU、NeMo 2.7.2 實跑，穩定可重現的組合為 `compute_dtype=float32` + `decoding.rnnt_fused_batch_size=0`。在這個條件下，`decoding.segment_gap_threshold=20` 可把最長字幕段落從 30.48s 壓到 12.48s；往下調到 `15` 或 `10` 都沒有再降低最長段落，只會增加段落數。

> **注意**：`decoding.rnnt_timestamp_type="all"` 和 `decoding.segment_separators=[".", "?", "!"]` 為預設值，無需手動設定。Chicago Fire 實測中，`segment_gap_threshold=20` 會在保留標點分段的前提下，進一步補切過長段落。

### 參數詳解
| 參數 | 推薦值 | 原因 |
| :--- | :--- | :--- |
| `vad.onset` | `0.2` | **經測試驗證最佳值**。平衡靈敏度與準確度，WER 最低且時間戳記最準確。 |
| `vad.offset` | `0.1` | **較低的結束門檻**。確保捕捉完整語句結尾。 |
| `vad.min_duration_on` | `0.1` | 保留極短促的發音，低於 0.1 秒通常是雜訊。 |
| `vad.min_duration_off` | `0.05` | **防止對話合併**。預設 0.2s 會將短停頓（如對話間的換氣）合併，造成 40+ 秒的超長片段。降至 0.05 可改善 35%。 |
| `vad.pad_onset` | `0.1` | 從預設 0.2 降低，減少段落前的 padding，避免重疊。 |
| `vad.pad_offset` | `0.1` | 從預設 0.2 降低，減少段落後的 padding，避免重疊。 |
| `vad.filter_speech_first` | `false` | **不強行過濾**。避免誤刪背景吵雜的對話。 |
| `compute_dtype` | `float32` | **Chicago Fire 實測穩定值**。在這台 RTX 3070 Laptop GPU 上比 `bfloat16` 更穩定。 |
| `decoding.rnnt_fused_batch_size` | `0` | **關閉 CUDA graphs**。Chicago Fire 實跑時可避免 warmup/首段轉寫階段的 CUDA illegal memory access。 |
| `decoding.segment_separators` | `[".", "?", "!"]` | **標點分割**。在句子結尾處分割段落，避免超長字幕。 |
| `decoding.segment_gap_threshold` | `20` | **Chicago Fire 實測最佳值**。最長字幕從 30.48s 降到 12.48s；`15/10` 沒有再降低最長段。 |
| `postprocessing.enable_itn` | `false` | 戲劇對白通常不需要將數字轉為阿拉伯數字。 |

### 效果展示

使用推薦參數前後的差異：

| 問題類型 | ❌ 優化前 | ✅ 優化後 |
| :--- | :--- | :--- |
| 幻覺 | 靜音處出現 "Thank you for watching" | 正確保持靜音，無多餘文字 |
| 遺漏 | 背景中的 "Help!" 呼救聲被忽略 | 成功捕捉到細微的呼救聲 |
| 斷句 | "I can't" / "believe this" (分成兩段) | "I can't believe this." (完整一句) |

---

## 📰 場景 B：新聞與訪談 (News/Interview)
**適用：** 新聞報導、攝影棚訪談、紀錄片。
**特徵：** 收音清晰、語速穩定、背景乾淨。

### 參數詳解
| 參數 | 推薦值 | 原因 |
| :--- | :--- | :--- |
| `vad.onset` | `0.5` | **標準門檻**。過濾掉主播的換氣聲、翻紙聲。 |
| `vad.min_duration_on` | `0.2` | 語句通常完整，不需要抓短音。 |
| `vad.filter_speech_first` | `true` | **開啟過濾**。讓主要人聲更純淨。 |
| `postprocessing.enable_itn` | `true` | **必開**。將 "January first" 自動轉為 "Jan 1st"。 |

**新聞/訪談專用指令範例：**
```bash
uv run nemoscribe \
    video_path="您的新聞影片.mp4" \
    output_path="您的新聞字幕.srt" \
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

## 💻 場景 C：技術教學與解說 (Tech/Tutorial)
**適用：** 程式教學、AI 模型解說、軟體操作演示。
**特徵：** 充滿版本號/參數/埠號、講者會有操作時的停頓、居家錄音環境。

### 核心策略：
技術影片最怕把 "Windows Eleven" 寫成單字。此模式專注於**數字格式化**與**保留思考停頓**。

| 參數 | 推薦值 | 原因 |
| :--- | :--- | :--- |
| `vad.onset` | `0.3` | **中庸設定**。比新聞靈敏一點，避免切掉講者思考後的發語詞（如 "呃... 然後我們..."），但又不至於錄下鍵盤聲。 |
| `vad.offset` | `0.2` | 稍微延後切斷，適應邊想邊講的節奏。 |
| `postprocessing.enable_itn` | `true` | **絕對核心！** 這是看懂教學的關鍵。<br>效果：<br>❌ "Python three point ten"<br>✅ "Python 3.10"<br>❌ "Port eight thousand eighty"<br>✅ "Port 8080" |
| `audio.max_chunk_duration` | `60` | 維持 60 秒，避免長篇大論導致的飄移。 |

**技術教學專用指令範例：**
```bash
uv run nemoscribe \
    video_path="您的教學影片.mp4" \
    output_path="您的教學字幕.srt" \
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

## 🔧 通用基礎參數 (Base Config)

無論哪種場景，以下參數建議**永遠保持**：

| 參數 | 推薦值 | 影響 |
| :--- | :--- | :--- |
| `audio.max_chunk_duration` | `60` | 強制每 60 秒切一段，避免模型疲勞。 |
| `audio.smart_segmentation` | `true` | 聰明地在靜音處切分。 |
| `decoding.rnnt_timestamp_type` | `"all"` | 輸出所有時間戳記類型（預設值）。配合 segment_separators 使用效果最佳。 |
| `decoding.segment_separators` | `[".", "?", "!"]` | 在標點處分割段落（預設值）。**已驗證**：可將長段落從 46.96s 降至 11.28s。設為空清單可停用。 |
| `decoding.segment_gap_threshold` | `None` | 基於詞間隔的段落分割（單位：幀，需為正整數）。當兩個連續詞之間的間隔超過此閾值時，強制分割為新段落；若同時啟用 `segment_separators`，NemoScribe 會保留標點分段並額外套用 gap 分段。 |
| `vad.enabled` | `true` | **永遠開啟**。這是避免幻覺（Hallucination）的唯一解法。 |

---

## 🤖 LLM 後處理 (LLM Post-processing)

**適用情境：** 字幕中人名、專有名詞辨識有誤，或需要更高的一致性。

ASR 模型在辨識人名和專有名詞時有先天限制（例如將 "Kylie Estevez" 聽成 "Alias of us"）。LLM 後處理透過大型語言模型修正這類錯誤。

### 前置條件

```bash
# 安裝 LLM 相依套件
uv sync --extra llm

# 設定 API 金鑰
cp .env.example .env
# 編輯 .env，加入：OPENAI_API_KEY=sk-... 或 ANTHROPIC_API_KEY=sk-ant-...
```

### 推薦設定

| 提供商 | 模型 | 品質 | 成本/集 | 建議場景 |
|--------|------|------|---------|----------|
| OpenAI | `gpt-4o-mini` | 良好 | ~$0.06 | **首選**：性價比最高 |
| OpenAI | `gpt-4o` | 優秀 | ~$0.30 | 需要更高品質時 |
| Anthropic | `claude-3-5-sonnet-20241022` | 優秀 | ~$0.24 | 偏好 Anthropic 時 |

### 使用範例

```bash
# 搭配 VAD + LLM（完整推薦流程）
uv run nemoscribe \
    video_path="您的影片.mkv" \
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

### 參數說明

| 參數 | 推薦值 | 說明 |
| :--- | :--- | :--- |
| `llm_postprocess.enabled` | `true` | 啟用 LLM 修正 |
| `llm_postprocess.provider` | `openai` | 提供商：`openai` 或 `anthropic` |
| `llm_postprocess.model` | `gpt-4o-mini` | 模型名稱 |
| `llm_postprocess.batch_size` | `20` | 每次送給 LLM 的字幕段落數。增大可提供更多上下文但較慢 |
| `llm_postprocess.max_retries` | `3` | 驗證失敗時的最大重試次數 |

### 已知限制

- **過度修正**：約 10% 的字幕段落可能被不必要地修改（多為輕微變動）
- **語意錯誤**：LLM 難以修正語意層面的錯誤（例如將 "breach" 誤改為 "bridge"）
- **數字漂移**：偶爾會改變數字（例如 "thirty seconds" → "40 seconds"）
- **成本**：需要付費 API，但成本很低（GPT-4o-mini 約每集 $0.06）

### 運作原理

1. 將字幕分批（每批 20 段）送給 LLM
2. LLM 以 JSON 格式回傳修正結果
3. 驗證修正幅度（相似度檢查，防止過度修改）
4. 若驗證失敗，提供回饋並重試（最多 3 次）
5. 任何環節失敗時，自動降級使用原始字幕

---

## ⚡ 批次處理指令 (Batch Processing)

一次處理整個資料夾的影片（以戲劇/電影模式為例）：

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

## ❓ 常見問題 FAQ

### Q: 處理一小時的影片大約需要多久？
**A:** 取決於您的 GPU 效能。以 RTX 3080 為例，一小時影片約需 3-5 分鐘處理時間。開啟 VAD 會稍微增加處理時間，但能大幅提升品質。

### Q: GPU 記憶體不足 (CUDA out of memory) 怎麼辦？
**A:** 嘗試縮短切分長度：
```bash
audio.max_chunk_duration=30  # 從 60 秒改為 30 秒
```

### Q: 字幕出現亂碼或奇怪符號？
**A:** 這通常是編碼問題。NemoScribe 輸出的 SRT 檔案為 UTF-8 編碼，請確認您的播放器或編輯器支援 UTF-8。

### Q: 為什麼有些對話還是被漏掉了？
**A:** 嘗試調低 VAD 靈敏度門檻：
```bash
vad.onset=0.15  # 從 0.2 調到 0.15，更加敏感
```
注意：過低（如 0.1 以下）可能會把雜音也辨識進來，且時間戳記準確度會下降。

### Q: 可以處理非英文的影片嗎？
**A:** `parakeet-tdt-0.6b-v2` 模型專為英文優化。處理其他語言建議使用：
- `nvidia/parakeet-tdt-0.6b-v3`：支援 25 種語言，自動語言偵測
- `nvidia/canary-1b-v2`：支援 25 種語言，並可進行翻譯

使用方式：
```bash
uv run nemoscribe video_path="影片.mp4" pretrained_name="nvidia/parakeet-tdt-0.6b-v3"
```

### Q: 字幕段落太長（超過 30 秒）怎麼辦？
**A:** 這通常發生在快速對話場景，嘗試以下方法：

1. **確認標點分割已啟用**（預設開啟）：
   ```bash
   decoding.segment_separators=".,?,!"
   ```

2. **降低 VAD 的 min_duration_off** 來保留更多對話間隙：
   ```bash
   vad.min_duration_off=0.05  # 預設 0.2
   ```

3. **使用 `segment_gap_threshold` 基於詞間隔分割**：
   ```bash
   decoding.segment_gap_threshold=20  # 當詞間隔超過 20 幀時分割
   ```

4. 若只想依 gap 分段、不想保留標點切段，可額外停用：
   ```bash
   decoding.segment_separators=
   ```

5. 如果仍有超長段落，這可能是連續快速對話沒有靜音間隙的正常現象。

### Q: 如何確認 CUDA/GPU 是否正常運作？
**A:** 執行以下指令檢查：
```bash
uv run python scripts/check_cuda.py
```
