"""獨立執行檔的進入點。

PyInstaller 需要一個實體腳本檔作為分析起點；直接指向套件的 console_script
在某些版本會漏掉相依。
"""

from focus_keeper.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
