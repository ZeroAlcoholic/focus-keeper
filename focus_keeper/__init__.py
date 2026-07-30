"""即時主人物頭像監測 MVP。

模組分層（規格 §3）：

    sources -> detectors -> tracker -> rules -> app(overlay + JSONL)

上層可替換偵測器，規則層不得綁定任何模型。
"""

__version__ = "0.1.0"
