# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller 規格檔：產生獨立執行檔（onedir）。

刻意用 onedir 而非 onefile：
  * onefile 每次啟動都要把 ~140 MB 解壓到暫存目錄，冷啟動慢好幾秒，
    對「開機就跑的背景監測」是明顯的體驗損失。
  * onedir 只有第一次載入 DLL 的成本，之後由 OS 檔案快取負責。
  * 出問題時 onedir 可以直接看到缺哪個 DLL，onefile 很難診斷。

config.yaml 與 models/ **不打進執行檔**，而是放在 exe 旁邊：
  * 使用者要能調門檻而不必重新打包
  * 規格 §9 要求權重版本可稽核；放在檔案系統上才能驗 SHA-256
  focus_keeper/config.py 的 APP_DIR 在 frozen 模式下會指向 exe 所在目錄。
"""

import sys
from pathlib import Path

ROOT = Path(SPECPATH).parent

a = Analysis(
    [str(ROOT / "packaging" / "entry.py")],
    pathex=[str(ROOT)],
    binaries=[],
    datas=[],
    hiddenimports=["focus_keeper.detectors.yunet_face"],
    # 路線 B 只在隔離環境用，不進獨立執行檔；連帶排除拖進來的重相依。
    excludes=[
        "mediapipe", "matplotlib", "tkinter", "PyQt5", "PyQt6", "PySide2", "PySide6",
        "IPython", "pytest", "setuptools", "pip", "scipy", "pandas",
    ],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz, a.scripts, [],
    exclude_binaries=True,
    name="focus-keeper",
    console=True,
    disable_windowed_traceback=False,
    upx=False,          # UPX 壓縮會提高防毒誤判率，體積省得不多，不划算
)
coll = COLLECT(
    exe, a.binaries, a.datas,
    strip=False, upx=False,
    name="focus-keeper",
)
