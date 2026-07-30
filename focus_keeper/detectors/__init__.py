"""偵測器套件與工廠。

新增路線（例如規格 §2 的 C：Darknet/YOLO head 模型）時，只需在此註冊，
規則層與追蹤層不需改動。
"""

from __future__ import annotations

from typing import Any, Callable, Mapping

from ..validation import ConfigError
from .base import (
    BBox,
    Detection,
    Detector,
    DetectorUnavailableError,
    ModelIntegrityError,
    filter_detections,
    sha256_of,
)

__all__ = [
    "BBox",
    "Detection",
    "Detector",
    "DetectorUnavailableError",
    "ModelIntegrityError",
    "filter_detections",
    "sha256_of",
    "build_detector",
    "available_detectors",
]


def _build_yunet(cfg: Mapping[str, Any]) -> Detector:
    from .yunet_face import YuNetFaceDetector

    return YuNetFaceDetector(
        model_path=cfg["model_path"],
        detect_width=cfg.get("detect_width", 320),
        score_threshold=cfg.get("score_threshold", 0.5),
        nms_threshold=cfg.get("nms_threshold", 0.3),
        top_k=cfg.get("top_k", 50),
        expected_sha256=cfg.get("sha256") or None,
    )


def _build_mediapipe(cfg: Mapping[str, Any]) -> Detector:
    from .mediapipe_face import MediaPipeFaceDetector

    return MediaPipeFaceDetector(
        model_path=cfg["model_path"],
        detect_width=cfg.get("detect_width", 320),
        score_threshold=cfg.get("score_threshold", 0.5),
        nms_threshold=cfg.get("nms_threshold", 0.3),
        expected_sha256=cfg.get("sha256") or None,
    )


_REGISTRY: dict[str, Callable[[Mapping[str, Any]], Detector]] = {
    "yunet": _build_yunet,
    "mediapipe": _build_mediapipe,
}


def available_detectors() -> tuple[str, ...]:
    return tuple(sorted(_REGISTRY))


def build_detector(name: str, detectors_cfg: Mapping[str, Any]) -> Detector:
    """依名稱建立偵測器；``detectors_cfg`` 為 config.yaml 的 ``detectors`` 區塊。"""
    key = name.strip().lower()
    if key not in _REGISTRY:
        raise ConfigError(f"未知的 detector：{name!r}，可用：{', '.join(available_detectors())}")
    section = detectors_cfg.get(key)
    if section is None:
        raise ConfigError(f"config.yaml 缺少 detectors.{key} 區塊")
    return _REGISTRY[key](section)
