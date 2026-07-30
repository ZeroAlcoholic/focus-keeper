"""測試共用工具。

多數測試以**合成偵測結果**驅動 tracker／rules，不需模型也不需攝影機，
因此在 CI 上可完全重現。需要真實模型的測試會自動 skip。
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable, Iterable

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from focus_keeper.detectors.base import BBox, Detection, Detector  # noqa: E402
from focus_keeper.synthetic import draw_synthetic_face  # noqa: E402,F401

FRAME_W, FRAME_H = 1280, 720
YUNET_MODEL = ROOT / "models" / "face_detection_yunet_2023mar.onnx"

requires_yunet = pytest.mark.skipif(
    not YUNET_MODEL.is_file(),
    reason="需要 models/face_detection_yunet_2023mar.onnx，請先執行 scripts/fetch_model.py",
)




def face(
    cx: float,
    cy: float,
    size: float = 160.0,
    score: float = 0.95,
    *,
    aspect: float = 0.8,
) -> Detection:
    """以中心點與高度建立一個臉框偵測（預設寬高比 0.8，接近真實臉框）。"""
    w = size * aspect
    h = size
    return Detection(bbox=BBox(cx - w / 2, cy - h / 2, w, h), score=score)


def centered_face(score: float = 0.95, size: float = 160.0) -> Detection:
    return face(FRAME_W / 2, FRAME_H / 2, size=size, score=score)


class TimelineDetector(Detector):
    """測試替身：偵測內容是**時間的函數**，不是呼叫次數的函數。

    這點很重要——若內容綁在「第幾次被呼叫」，兩條時間軸只要處理幀數差一，
    場景轉換就整整平移一個推論週期，測到的會是測試自身的假象，
    而不是分析核心對時間軸的一致性。

    使用方式：在呼叫 ``pipeline.process(packet)`` 前先設好 ``now_ms``。
    """

    name = "timeline"

    def __init__(self, content_fn: Callable[[int], Iterable[Detection]]) -> None:
        self._content_fn = content_fn
        self.now_ms = 0
        self.calls = 0

    def detect(self, image) -> list[Detection]:  # noqa: ANN001 - 測試替身
        self.calls += 1
        return list(self._content_fn(self.now_ms))

    def describe(self) -> dict:
        return {"detector": self.name, "calls": self.calls}
