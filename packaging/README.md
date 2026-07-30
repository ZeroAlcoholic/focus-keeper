# 散布方式

兩種給同事測試的方式，選一個或兩個都給。

| | A. 原始碼（repo） | B. 獨立執行檔 |
|---|---|---|
| 對象 | 會裝 Python 的人 | 不想／不能裝 Python 的人 |
| 下載大小 | clone **0.5 MB** + pip 抓相依 | **58 MB**（zip，解壓後 146 MB） |
| 前置作業 | 需要 Python 3.10+ | **無** |
| 可改參數 | ✅ | ✅（`config.yaml` 在 exe 旁邊） |
| 可看／改程式 | ✅ | ❌ |
| 建置者需求 | — | 需在 **Windows** 上建置（不可跨平台） |

---

## A. 原始碼散布

倉庫已就緒，同事只要三步：

```bash
git clone <repo-url>
cd focus_keeper
pip install -e .

python -m focus_keeper.cli --source 0
```

模型權重（232 KB、MIT）已隨倉庫提供，**不需連網**。
沒有攝影機時：`python scripts/make_sample_video.py` 產生示範影片。

倉庫共 38 個檔、約 0.5 MB。內部文件（需求規格、詳細能力評估）不在倉庫內。

---

## B. 獨立執行檔

### 建置（在 Windows 上，用隔離環境避免污染日常環境）

```powershell
python -m venv buildenv
buildenv\Scripts\python -m pip install -e . pyinstaller==6.11.1
buildenv\Scripts\python -m PyInstaller packaging\focus_keeper.spec `
    --distpath packaging\dist --workpath packaging\build --noconfirm
```

### 交付內容

把 `packaging\dist\focus-keeper\` 整個資料夾打包 zip，並在其中放入：

```
focus-keeper\
├── focus-keeper.exe
├── _internal\                              (PyInstaller 產出，勿動)
├── config.yaml                             ← 需手動複製
├── config.demo.yaml                        ← 需手動複製
└── models\
    └── face_detection_yunet_2023mar.onnx   ← 需手動複製
```

`config.yaml` 與 `models/` **刻意不打進 exe**：

1. 使用者要能調門檻而不必重新打包
2. 規格 §9 要求權重版本可稽核 —— 放在檔案系統上才能驗 SHA-256

`focus_keeper/config.py` 的 `APP_DIR` 在 frozen 模式下指向 **exe 所在目錄**，
所以從任何工作目錄啟動都找得到（已實測）。

### 同事的使用方式

```
focus-keeper.exe --config config.demo.yaml --source 0
```

### 為什麼用 onedir 而不是 onefile

- onefile 每次啟動都要把約 140 MB 解壓到暫存目錄，冷啟動慢好幾秒 —— 對背景監測是明顯的體驗損失
- onedir 只有第一次載入 DLL 的成本，之後由 OS 檔案快取負責
- 出問題時 onedir 能直接看出缺哪個 DLL，onefile 很難診斷

### 體積（實測）

**同事下載的是 zip，不是解壓後的目錄** —— 這兩個數字差很多：

| 版本 | 解壓後 | **下載（zip）** | 能讀影片檔 |
|---|---|---|---|
| 完整版（建議） | 146.4 MB | **58.1 MB** | ✅ |
| 僅攝影機版 | 119.4 MB | **46.6 MB** | ❌ |

組成：`cv2.pyd` 67.7 MB、ffmpeg DLL 27.0 MB、numpy OpenBLAS 19.4 MB、
`python312.dll` 7.1 MB、其他約 26 MB。

**建議用完整版。** 僅攝影機版只省 11.5 MB 下載，卻要放棄影片檔支援——
那正是同事沒鏡頭時的測試備案。要建僅攝影機版，在 spec 的 `binaries`
過濾掉 `opencv_videoio_ffmpeg*.dll` 即可（已實測仍正常，10.06 FPS）。

UPX 壓縮刻意不用：體積省得不多，卻明顯提高防毒誤判率。

再往下只剩「換掉 OpenCV」一條路（onnxruntime 跑 YuNet + 另尋擷取與繪圖），
估計可到 25–35 MB。但那要自己實作 YuNet 的 anchor 解碼與 NMS，等於把已驗證
的偵測環節換成重寫版本，所有實測數字失效、需重跑驗收。以省 20 MB 不划算。

### 已知的實務門檻

1. **防毒誤判**：PyInstaller 產出的未簽章 exe 在企業環境常被 Windows Defender 或
   端點防護攔下。若要正式發給多位同事，建議做**程式碼簽章**。
2. **不可跨平台**：Windows 上建置只能給 Windows。
3. **首次啟動較慢**：需載入約 100 MB 的 DLL。
4. **exe 內看不到版本資訊**：診斷時請同事執行 `--report out.json`，
   裡面有套件版本、模型 SHA-256、設備資訊。

---

## 驗證交付物

不論哪種方式，請同事先跑這個確認環境正常：

```
focus-keeper.exe --source 0 --no-display --duration 10 --report check.json
```

預期：有效 FPS 約 10、`pipeline P95` 遠低於 200 ms。
若人在鏡頭前則事件數應為 0；若不在，會看到 `PRIMARY_MISSING` / `PRIMARY_LEFT`
（這是正確行為）。
