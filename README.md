# focus_keeper

全地端、低延遲的**單人頭肩構圖監測**。即時處理攝影機或影片，在下列情況示警：

- 主角臉部離開畫面或偏離主要構圖區
- 畫面出現第二人以上
- 影像來源停止更新（USB 當掉、驅動凍結、虛擬攝影機停格）

不做人臉辨識、身分比對、全身偵測，也不做任何雲端傳輸。影像預設不落地。

**純 CPU、無 GPU、執行期不連網。** 相依只有 3 個套件，全部可封閉商用。

## 能力界線（使用前必讀）

系統偵測的是**人臉**，不是人。以下**不在**能力範圍內：

| 不能判讀 | 原因 |
|---|---|
| 螢幕前是不是「本人」 | 不做人臉辨識與身分比對（設計上排除） |
| 是否在「專注看畫面」 | 不量視線方向與注意力 |
| 真人 vs 照片／螢幕影像 | 已實測：靜止照片會被判為「一人在場」 |
| 「人在但沒面向鏡頭」vs「人真的離開」 | 兩者都表現為臉偵測不到 |
| 背對鏡頭的第二人 | 沒有臉即不計入人數 |

**定位是輔助提示**，不具防偽能力，不應在有規避動機的場合作為控制依據。

---

## 快速開始

完整安裝、使用與疑難排解：**[INSTALL.md](INSTALL.md)**

```bash
pip install -e .          # 或下載 Releases 的獨立執行檔（不需 Python）

# 取得模型權重並驗證 SHA-256（一次性；執行期不會自動下載）
python scripts/fetch_model.py

focus-keeper --source 0                  # 攝影機
focus-keeper --source media/sample.mp4   # 影片
```

預覽視窗按 `q` 或 `ESC` 結束。沒有攝影機時可產生示範影片：

```bash
python scripts/make_sample_video.py
```

### CLI

| 選項 | 說明 |
|---|---|
| `--source` | `0` = 攝影機 index；其餘視為影片路徑 |
| `--config` | 設定檔路徑，預設 `./config.yaml`。未指定的欄位與區塊沿用內建預設，可只寫要覆蓋的部分 |
| `--detector` | `yunet`（預設）或 `mediapipe` |
| `--log` | JSONL 事件記錄路徑；`--log none` 停用 |
| `--report` | 把效能與版本資訊寫成 JSON |
| `--no-display` | 不開預覽視窗 |
| `--duration` / `--max-frames` | 執行秒數／影格數上限 |

離場代碼：`0` 正常、`2` 啟動失敗（設定／模型／來源）、`3` 攝影機持續停頓而中止。

---

## 事件

| 事件 | 精確含意 | 預設門檻 |
|---|---|---|
| `NORMAL` | 恰好一張符合門檻的臉，位於構圖區內，畫面有更新 | — |
| `MULTI_PERSON` | 偵測到 ≥ 2 張臉 | 500 ms |
| `PRIMARY_MISSING` | 主角**臉部**連續偵測不到 | 1000 ms |
| `PRIMARY_LEFT` | 同上（升級門檻） | 2000 ms |
| `PRIMARY_OUTSIDE_ROI` | 臉可見但偏離主要構圖區 | 800 ms |
| `FEED_FROZEN` | 畫面連續無變化 → **其他判定皆不可信** | 3000 ms |

> ⚠️ 系統偵測的是**人臉**，不是人。`PRIMARY_LEFT` 的真實含意是「主角臉部偵測不到」，
> 不等於「人離開了」——背對鏡頭、大角度側身、低頭都會觸發。

**單幀漏偵測不會示警。** 示警前要求條件**連續**成立（任何一次不成立就重新起算）；
示警後才套用解除遲滯（連續不成立 200 ms 才關閉事件）。兩段方向相反的邏輯不可共用。

---

## 架構

```
FramePacket(image, timestamp_ms, frame_id)
  sources → detectors → tracker → rules → pipeline → cli / overlay / eventlog
  影像來源   偵測        主角身分   時間判定   分析核心    介面 / 疊圖 / 記錄
```

| 模組 | 職責 |
|---|---|
| `sources.py` | 攝影機（槽位固定 1）／影片，統一時間軸 |
| `detectors/` | 偵測器介面與實作，只回框與信心值 |
| `tracker.py` | 主角身分維持 |
| `rules.py` | 時間門檻判定（**不 import OpenCV**） |
| `config.py` | `AppConfig.load()`＝設定的唯一載入與驗證入口 |
| `validation.py` | 共用驗證基礎（葉模組，避免循環匯入） |
| `pipeline.py` | 分析核心（無 CLI、無疊圖、無檔案輸出） |
| `metrics.py` / `eventlog.py` / `overlay.py` / `cli.py` | 量測 / JSONL / 疊圖 / 介面 |

單向依賴，規則層不含任何模型知識（不 import OpenCV），可用純合成資料測試。
攝影機與影片共用同一個 `AnalysisPipeline`，差別只在時間戳來源，因此離線重跑可重現。

| 要做的事 | 只改這裡 |
|---|---|
| 換偵測器 | `focus_keeper/detectors/` 加一個檔 + `__init__.py` 註冊一行 |
| 調門檻／ROI | `config.yaml`（打錯欄位名會直接報錯，不會靜默忽略） |
| 加新事件 | `focus_keeper/rules.py` 的 `EventType` + `_conditions` |

### 偵測器

| 路線 | 組合 | 授權 | 狀態 |
|---|---|---|---|
| A（預設） | OpenCV `FaceDetectorYN` + YuNet ONNX | 程式 Apache-2.0／權重 MIT | ✅ 已實機驗證 |
| B | MediaPipe BlazeFace | Apache-2.0 | ⚠️ 僅供**離線交叉驗證**，需隔離環境 |
| C | 自訓任意角度 `head` 模型 | — | ⛔ 未實作，僅保留介面 |

**路線 B 不能與 A 共存於同一環境**：所有 `mediapipe` 版本都會連帶安裝
`opencv-contrib-python`，與釘住的 `opencv-python` 搶同一個 `cv2` 套件名；
`mediapipe<=0.10.21` 還要求 `numpy<2`。安裝方式見 `requirements.txt` 註解。

不使用 Ultralytics 套件或權重；YOLOX 官方預訓練權重亦不列入供應鏈。

---

## 實測摘要

真實素材（AMI Meeting Corpus，25 分 04 秒，15,043 次判定）與本機攝影機：

| 項目 | 結果 |
|---|---|
| 有人在場時漏偵測率 | **2.78%**；27 段連續漏偵測中 22 段（81%）被時間門檻吸收，不示警 |
| 空景正確判定無人 | **98.8%** |
| 示警延遲（超出門檻後） | **0–95 ms**（掃描整個取樣格），要求 ≤ 250 ms |
| 有效判定頻率 | **10.0 FPS**（持續 3 分鐘、1,801 次判定） |
| pipeline P95 延遲 | **4.15 ms**（單執行緒、持續運轉），要求 < 200 ms |
| 端到端 P95 延遲 | **4.43 ms** |
| 長時運轉 | 取樣視窗有上界，成本不隨時間成長（見下） |
| 執行期相依體積 | 約 142 MB（幾乎全是 OpenCV） |

重跑評測（需自行取得素材）：

```bash
python scripts/evaluate_real.py <影片> --absent "0-148,571-731"
```

`--absent` 是**已目視確認**的空景區間。未先確認 ground truth 就評測，
數字會差一個量級（多數素材頭尾都是空景）。

**已知失效模式**：側臉超過約 ±60°、手遮臉、低頭看資料、臉面積比 < 0.004。
時間門檻可吸收短於 1 秒的漏偵測；持續側身會被判為離場。

**長時運轉**：延遲統計的取樣視窗上限為 36,000 次判定（10 FPS 下約 1 小時）。
計數、平均、最大值對整段執行精確；`p50`／`p95` 為視窗內值，`--report` 會
一併輸出 `percentile_window_samples` 以免誤讀。疊圖的 p95 每 10 次判定才
重算一次——先前每格重算，8 小時後單格成本 6.29 ms，比整個偵測（2.5 ms）還高。

---

## 測試

```bash
pytest tests/ -q      # 138 項
```

缺模型權重時，需要真實推論的測試會自動 skip（其餘仍全過）。

---

## 授權與隱私

- 模型權重的 SHA-256 釘在 `config.yaml`，**不符即拒絕啟動**
- 執行期**沒有任何下載程式路徑**；權重只能由 `scripts/fetch_model.py` 明確取得
- **不儲存影像、臉部裁切或 embedding**；JSONL 只記錄事件型別、時間、信心值、人數
- 每筆 JSONL 記錄含 `logged_at`（ISO 8601 含時區）。這是應用層記錄，**不是防篡改稽核軌跡**

第三方授權（程式碼／權重／評測資料分列）：[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md)
