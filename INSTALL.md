# 安裝與使用

兩種方式，選一個。

---

## 方式 A：原始碼（需 Python 3.10+）

```bash
git clone https://github.com/ZeroAlcoholic/focus-keeper.git
cd focus-keeper
pip install -e .
```

模型權重已隨倉庫提供（232 KB），**安裝與執行都不需要連網**。

```bash
python -m focus_keeper.cli --config config.demo.yaml --source 0
```

## 方式 B：獨立執行檔（不需 Python，僅 Windows x64）

1. 到 [Releases](https://github.com/ZeroAlcoholic/focus-keeper/releases) 下載 `focus-keeper-v0.1.0-win64.zip`（58 MB）
2. 解壓到任意位置
3. 在該資料夾開命令提示字元：

```
focus-keeper.exe --config config.demo.yaml --source 0
```

> 未經程式碼簽章，企業防毒可能攔截。被擋就改用方式 A。

---

## 確認裝好了

```bash
python -m focus_keeper.cli --source 0 --no-display --duration 10 --report check.json
```

預期：`有效推論 FPS` 約 10、`pipeline 延遲 P95` 遠低於 200 ms。

沒有攝影機時：

```bash
python scripts/make_sample_video.py                              # 產生示範影片
python -m focus_keeper.cli --source media/sample.mp4             # 跑它
```

---

## 使用

`q` 或 `ESC` 結束。

| 選項 | 說明 |
|---|---|
| `--source 0` | 攝影機（數字是裝置 index） |
| `--source <路徑>` | 影片檔 |
| `--config config.demo.yaml` | DEMO 參數（時間門檻較短，現場不用乾等） |
| `--no-display` | 不開預覽視窗 |
| `--log <路徑>` | JSONL 事件記錄；`none` 停用 |
| `--report <路徑>` | 效能與版本資訊寫成 JSON（回報問題時附上這個） |
| `--duration <秒>` | 執行秒數上限 |

離場代碼：`0` 正常、`2` 啟動失敗、`3` 攝影機持續停頓而中止。

### 畫面判讀

左上角 `STATUS` 一行，顏色即語意：

| 顯示 | 顏色 | 含意 |
|---|---|---|
| `NORMAL` | 綠 | 一人在畫面中、構圖正常 |
| `MULTI_PERSON` | 橘 | 偵測到 2 張以上的臉 |
| `PRIMARY_OUTSIDE_ROI` | 橘 | 臉可見但偏離構圖區 |
| `PRIMARY_MISSING` | 紅 | 主角臉部連續 1 秒偵測不到 |
| `PRIMARY_LEFT` | 紅 | 同上，連續 2 秒 |
| `FEED_FROZEN` | 洋紅 | 畫面停格 → **其他判定皆不可信** |

**紅色不等於「人離開了」** —— 真實含意是「臉偵測不到」。背對鏡頭、大角度側身、
低頭看資料都會變紅。完整界線見 [README](README.md#能力界線使用前必讀)。

### 調參數

改 `config.yaml`（每個欄位都有註解說明）。打錯欄位名會直接報錯並列出可用欄位，
不會靜默忽略。只寫要覆蓋的部分即可，其餘沿用內建預設。

---

## 疑難排解

| 症狀 | 原因與處置 |
|---|---|
| `找不到 YuNet 模型檔` | 執行 `python scripts/fetch_model.py` |
| `無法開啟攝影機 index=0` | 鏡頭被其他程式占用，或 index 不是 0（外接／虛擬攝影機） |
| 攝影機開啟很慢 | 在 `config.yaml` 把 `sources.camera.backend` 改成 `msmf`（實測 0.6 秒 vs `dshow` 1.3 秒） |
| 一直顯示 `PRIMARY_LEFT` | 臉太小或角度太偏。臉需佔畫面 1.5% 以上；一般坐在筆電前的距離都符合 |
| 畫面文字是英文 | OpenCV 字型不支援中文，屬預期 |
| 延遲偏高 | 在 `config.yaml` 把 `detectors.yunet.detect_width` 降到 256（省 40% 算力，漏偵測率 +0.32pp） |

回報問題時請附上 `--report` 產生的 JSON，裡面有套件版本、模型 SHA-256 與設備資訊。
