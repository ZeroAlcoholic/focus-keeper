"""路線 B（備援）：MediaPipe Face Detector（BlazeFace Short Range）。

授權：MediaPipe Apache-2.0；模型權重需自行留存來源與版本
（見 THIRD_PARTY_NOTICES.md）。

用途限定為規格 §7 的「以 B 對照 A」交叉驗證，不是預設路線。
MediaPipe 為選用相依：未安裝時本模組仍可被 import，只有建構時才報錯。
依規格 §9，模型檔同樣**不得於執行時自動下載**。
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

__all__ = ["MediaPipeFaceDetector"]


class MediaPipeFaceDetector(Detector):
    """BlazeFace Short Range 臉部偵測器（MediaPipe Tasks API）。"""

    name = "mediapipe"

    def __init__(
        self,
        model_path: str | Path,
        *,
        detect_width: int | None = 320,
        score_threshold: float = 0.5,
        nms_threshold: float = 0.3,
        expected_sha256: str | None = None,
    ) -> None:
        try:
            import mediapipe as mp
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision as mp_vision
        except ImportError as exc:  # pragma: no cover - 取決於環境
            raise DetectorUnavailableError(
                "未安裝 mediapipe。\n"
                "⚠️ 不要裝進本專案的執行環境：mediapipe 會連帶安裝 "
                "opencv-contrib-python（與釘住的 opencv-python 搶同一個 cv2 套件名），"
                "且 0.10.21 以下還會把 numpy 降級到 <2。\n"
                "路線 B 僅供離線交叉驗證，請另建隔離環境：\n"
                "    python -m venv mpenv\n"
                "    mpenv/Scripts/python -m pip install mediapipe==1.0.0 PyYAML\n"
                "    mpenv/Scripts/python scripts/fetch_model.py --detector mediapipe"
            ) from exc

        self._mp = mp
        self._model_path = Path(model_path)
        if not self._model_path.is_file():
            raise DetectorUnavailableError(
                f"找不到 BlazeFace 模型檔：{self._model_path}\n"
                "請先執行：python scripts/fetch_model.py --detector mediapipe"
            )

        self._sha256 = sha256_of(self._model_path)
        if expected_sha256 and self._sha256.lower() != expected_sha256.lower():
            raise ModelIntegrityError(
                f"BlazeFace 模型 SHA-256 不符。\n  期望：{expected_sha256}\n  實際：{self._sha256}"
            )

        self._detect_width = detect_width
        self._score_threshold = float(score_threshold)
        self._nms_threshold = float(nms_threshold)

        options = mp_vision.FaceDetectorOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(self._model_path)),
            running_mode=mp_vision.RunningMode.IMAGE,
            min_detection_confidence=self._score_threshold,
            min_suppression_threshold=self._nms_threshold,
        )
        self._impl = mp_vision.FaceDetector.create_from_options(options)

    # ------------------------------------------------------------------ #

    def _prepare(self, image: np.ndarray) -> tuple[np.ndarray, float]:
        height, width = image.shape[:2]
        if self._detect_width is None or width <= self._detect_width:
            return image, 1.0
        scale = self._detect_width / float(width)
        new_size = (self._detect_width, max(1, int(round(height * scale))))
        return cv2.resize(image, new_size, interpolation=cv2.INTER_LINEAR), 1.0 / scale

    def detect(self, image: np.ndarray) -> list[Detection]:
        prepared, restore = self._prepare(image)
        rgb = cv2.cvtColor(prepared, cv2.COLOR_BGR2RGB)
        mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        result = self._impl.detect(mp_image)

        detections: list[Detection] = []
        for det in result.detections:
            box = det.bounding_box
            score = float(det.categories[0].score) if det.categories else 0.0
            landmarks = tuple(
                (float(kp.x) * prepared.shape[1] * restore, float(kp.y) * prepared.shape[0] * restore)
                for kp in (det.keypoints or ())
            )
            detections.append(
                Detection(
                    bbox=BBox(
                        float(box.origin_x),
                        float(box.origin_y),
                        float(box.width),
                        float(box.height),
                    ).scaled(restore),
                    score=score,
                    landmarks=landmarks,
                )
            )
        return detections

    def describe(self) -> dict[str, Any]:
        try:
            import mediapipe as mp

            version = mp.__version__
        except Exception:  # pragma: no cover
            version = "unknown"
        return {
            "detector": self.name,
            "route": "B",
            "backend": f"mediapipe {version} FaceDetector (BlazeFace Short Range)",
            "model_path": str(self._model_path),
            "model_sha256": self._sha256,
            "model_bytes": self._model_path.stat().st_size,
            "detect_width": self._detect_width,
            "score_threshold": self._score_threshold,
            "nms_threshold": self._nms_threshold,
            "license": "code: Apache-2.0 (MediaPipe) / weights: 需留存來源與版本，見 THIRD_PARTY_NOTICES.md",
        }

    def close(self) -> None:
        closer = getattr(self._impl, "close", None)
        if callable(closer):
            closer()
