# 第三方授權聲明

依規格 §9，程式碼、模型權重、訓練資料**分別**列出。
本專案為可封閉商用（proprietary）交付，任何新增相依都必須先通過本頁的授權檢核。

最後更新：2026-07-29

---

## 1. 程式碼相依

| 套件 | 版本 | 授權 | 商用（封閉原始碼） | 用途 |
|---|---|---|---|---|
| [opencv-python](https://github.com/opencv/opencv-python) | 4.12.0.88 | Apache-2.0 | ✅ 可 | 影像擷取、`FaceDetectorYN` 推論、疊圖 |
| [NumPy](https://numpy.org/) | 2.2.6 | BSD-3-Clause | ✅ 可 | 陣列運算 |
| [PyYAML](https://pyyaml.org/) | 6.0.3 | MIT | ✅ 可 | 設定檔解析 |
| [pytest](https://pytest.org/) | 9.0.2 | MIT | ✅ 可（僅開發期） | 測試 |
| [mediapipe](https://github.com/google-ai-edge/mediapipe) | 未安裝（選用） | Apache-2.0 | ✅ 可 | 路線 B 對照驗證 |

Apache-2.0 需保留授權條款與 NOTICE；散布時請一併附上本檔與各套件的 LICENSE。

---

## 2. 模型權重

權重與程式碼**分別授權**，不得假設兩者相同。

### 2.1 使用中：YuNet（路線 A，預設）

| 項目 | 內容 |
|---|---|
| 檔名 | `face_detection_yunet_2023mar.onnx` |
| 大小 | 232,589 bytes |
| SHA-256 | `8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4` |
| 來源 | [opencv_zoo](https://github.com/opencv/opencv_zoo/tree/4.10.0/models/face_detection_yunet)，tag `4.10.0` |
| 授權 | **MIT**（`models/face_detection_yunet/LICENSE`） |
| 原始作者 | Wei Wu、Wenyong Huang、Yuantao Feng 等（深圳大學） |
| 取得日期 | 2026-07-29 |
| 商用（封閉原始碼） | ✅ 可，需保留 MIT 授權條款與著作權聲明 |

SHA-256 已釘在 `config.yaml`；不符即拒絕啟動。

### 2.2 選用：BlazeFace Short Range（路線 B，對照用）

| 項目 | 內容 |
|---|---|
| 檔名 | `blaze_face_short_range.tflite` |
| 來源 | Google MediaPipe 模型庫，版本目錄 `/float16/1/` |
| 授權 | Apache-2.0（依 MediaPipe 模型卡） |
| 商用（封閉原始碼） | ✅ 可 |
| 狀態 | **尚未取得**。實際採用前須以 `scripts/fetch_model.py --detector mediapipe` 下載，並把日期、版本目錄與 SHA-256 補進本表。 |

---

## 3. 訓練資料

| 模型 | 訓練資料 | 授權狀態 | 對本專案的影響 |
|---|---|---|---|
| YuNet | WIDER FACE 等公開臉部資料集 | 上游未於模型授權中對訓練資料另作聲明 | 本專案**僅使用推論權重**，不重新散布訓練資料，亦不進行再訓練 |
| BlazeFace | Google 內部與公開資料集 | 同上，模型卡未逐項列出 | 同上 |
| 路線 C（未實作） | **須自行標註** | 待定 | 見下節 |

**本專案自身不含任何訓練資料，也不產生訓練資料**：影像預設不落地，不儲存臉部裁切或 embedding。

### 3.1 評測資料（非訓練用）

| 項目 | 內容 |
|---|---|
| 資料集 | AMI Meeting Corpus |
| 使用檔案 | `IS1000a.Closeup1.avi`（低解析度 DivX，43,448,508 bytes） |
| 來源 | <https://groups.inf.ed.ac.uk/ami/download/> |
| 授權 | **CC BY 4.0**（Creative Commons 姓名標示 4.0 國際） |
| 商用 | ✅ 允許，**須標示出處** |
| 用途 | 僅作**推論評測**，不做訓練、不做微調、不重新散布 |
| 取得日期 | 2026-07-29 |

**必要標示**：本專案的準確度評測使用 AMI Meeting Corpus
（<https://groups.inf.ed.ac.uk/ami/>），依 CC BY 4.0 授權使用。

評測產物僅為統計數字（`logs/ami_*.json`），**不含任何影格或臉部裁切**。
影片檔本身不進 git（見 `.gitignore` 的 `media/`）。

---

## 4. 明確排除的供應鏈

| 項目 | 排除原因 |
|---|---|
| **Ultralytics**（YOLOv5/v8/11/26 套件與預訓練權重） | AGPL-3.0；封閉商用須另購 Enterprise License。規格 update 版已將其自選型中移除。 |
| **YOLOX 官方預訓練權重** | 程式碼為 Apache-2.0，但官方預訓練權重的授權需個別確認；本案不列入推薦供應鏈。 |
| 任何執行期自動下載的權重 | 違反規格 §9：無法在交付當下固定版本與雜湊。 |

---

## 5. 路線 C（升級路線，本版未實作）的授權前提

若 A/B 無法涵蓋大角度轉頭或遮臉，才啟動 Darknet/YOLO + 自訓 `head` 單類別模型：

1. **框架**：Darknet/YOLO（Apache-2.0），可封閉商用。
2. **權重**：**不得**直接沿用第三方預訓練權重並假設其與框架同授權。只接受兩種來源之一：
   - 以內部標註資料自行訓練；或
   - 取得權重持有者的**書面**商用授權。
3. **訓練資料**：需確認肖像權／個資合規（含蒐集告知與同意），並在本檔補列資料來源與授權。

---

## 6. 隱私與資料處理聲明

- 全程地端運算，**無任何雲端傳輸**。
- 不做人臉辨識、身分比對或 1:N 比對。
- 不儲存影像、臉部裁切或 embedding；JSONL 只記錄事件型別、時間、信心值與人數等中繼資料。
- 偵測器回傳的 landmark 僅供畫面標示與品質判斷，不落地。
