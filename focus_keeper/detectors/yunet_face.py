"""路線 A（預設）：OpenCV ``FaceDetectorYN`` + YuNet ONNX。

授權：OpenCV Apache-2.0；YuNet 模型目錄 MIT（見 THIRD_PARTY_NOTICES.md）。

依規格 §9：**執行時不得自動下載權重**。模型檔必須事先以
``python scripts/fetch_model.py`` 取得並通過 SHA-256 驗證。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .base import (
    BBox,
    Detection,
    Detector,
    DetectorUnavailableError,
    ModelIntegrityError,
    sha256_of,
)

__all__ = ["YuNetFaceDetector"]


class YuNetFaceDetector(Detector):
    """YuNet 臉部偵測器。

    Parameters
    ----------
    model_path:
        本機 ONNX 權重路徑；不存在即拋 :class:`DetectorUnavailableError`。
    detect_width:
        送入模型前的等比縮放寬度；``None`` 表示使用原始影格尺寸。
        降低此值是規格 §7「延遲超標」時的第一順位手段。
    score_threshold:
        模型內部門檻，刻意設得比業務層 ``min_confidence`` 寬鬆，
        讓 NMS 有足夠候選；最終取捨由規則層的 ``min_confidence`` 決定。
    expected_sha256:
        設定後即強制比對，避免權重被替換。
    """

    name = "yunet"

    def __init__(
        self,
        model_path: str | Path,
        *,
        detect_width: int | None = 320,
        score_threshold: float = 0.5,
        nms_threshold: float = 0.3,
        top_k: int = 50,
        expected_sha256: str | None = None,
        backend_id: int = cv2.dnn.DNN_BACKEND_OPENCV,
        target_id: int = cv2.dnn.DNN_TARGET_CPU,
    ) -> None:
        self._model_path = Path(model_path)
        if not self._model_path.is_file():
            raise DetectorUnavailableError(
                f"找不到 YuNet 模型檔：{self._model_path}\n"
                "請先執行：python scripts/fetch_model.py（本專案禁止執行期自動下載權重）"
            )

        self._sha256 = sha256_of(self._model_path)
        if expected_sha256 and self._sha256.lower() != expected_sha256.lower():
            raise ModelIntegrityError(
                f"YuNet 模型 SHA-256 不符。\n  期望：{expected_sha256}\n  實際：{self._sha256}"
            )

        self._detect_width = detect_width
        self._score_threshold = float(score_threshold)
        self._nms_threshold = float(nms_threshold)
        self._top_k = int(top_k)
        self._input_size: tuple[int, int] | None = None

        self._impl = cv2.FaceDetectorYN.create(
            model=str(self._model_path),
            config="",
            input_size=(320, 320),
            score_threshold=self._score_threshold,
            nms_threshold=self._nms_threshold,
            top_k=self._top_k,
            backend_id=backend_id,
            target_id=target_id,
        )

    # ------------------------------------------------------------------ #

    def _prepare(self, image: np.ndarray) -> tuple[np.ndarray, float]:
        """回傳 (送入模型的影像, 還原到原始座標的縮放倍率)。"""
        height, width = image.shape[:2]
        if self._detect_width is None or width <= self._detect_width:
            return image, 1.0
        scale = self._detect_width / float(width)
        new_size = (self._detect_width, max(1, int(round(height * scale))))
        resized = cv2.resize(image, new_size, interpolation=cv2.INTER_LINEAR)
        return resized, 1.0 / scale

    def detect(self, image: np.ndarray) -> list[Detection]:
        prepared, restore = self._prepare(image)
        height, width = prepared.shape[:2]
        if self._input_size != (width, height):
            self._impl.setInputSize((width, height))
            self._input_size = (width, height)

        _, raw = self._impl.detect(prepared)
        if raw is None:
            return []

        detections: list[Detection] = []
        for row in raw:
            x, y, w, h = (float(v) for v in row[0:4])
            score = float(row[-1])
            landmarks = tuple(
                (float(row[4 + 2 * i]) * restore, float(row[5 + 2 * i]) * restore)
                for i in range(5)
            )
            detections.append(
                Detection(
                    bbox=BBox(x, y, w, h).scaled(restore),
                    score=score,
                    landmarks=landmarks,
                )
            )
        return detections

    def describe(self) -> dict[str, Any]:
        return {
            "detector": self.name,
            "route": "A",
            "backend": f"opencv-python {cv2.__version__} FaceDetectorYN",
            "model_path": str(self._model_path),
            "model_sha256": self._sha256,
            "model_bytes": self._model_path.stat().st_size,
            "detect_width": self._detect_width,
            "score_threshold": self._score_threshold,
            "nms_threshold": self._nms_threshold,
            "top_k": self._top_k,
            "license": "code: Apache-2.0 (OpenCV) / weights: MIT (opencv_zoo face_detection_yunet)",
        }
