"""Detector 抽象層。

規則層與追蹤層只依賴本檔的 ``BBox`` / ``Detection`` / ``Detector``，
不得引用任何具體模型（YuNet、MediaPipe、Darknet）之型別或行為。
"""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

__all__ = [
    "BBox",
    "Detection",
    "Detector",
    "DetectorUnavailableError",
    "ModelIntegrityError",
    "sha256_of",
]


class DetectorUnavailableError(RuntimeError):
    """所需套件或模型檔不存在時拋出。"""


class ModelIntegrityError(RuntimeError):
    """模型檔 SHA-256 與設定不符時拋出。"""


def sha256_of(path: str | Path, chunk_size: int = 1 << 20) -> str:
    """回傳檔案 SHA-256（小寫十六進位），供授權／版本回報使用。"""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class BBox:
    """像素座標的軸對齊框，原點在左上。"""

    x: float
    y: float
    w: float
    h: float

    @property
    def cx(self) -> float:
        return self.x + self.w / 2.0

    @property
    def cy(self) -> float:
        return self.y + self.h / 2.0

    @property
    def area(self) -> float:
        return max(0.0, self.w) * max(0.0, self.h)

    @property
    def xyxy(self) -> tuple[float, float, float, float]:
        return (self.x, self.y, self.x + self.w, self.y + self.h)

    def to_int(self) -> tuple[int, int, int, int]:
        return (int(round(self.x)), int(round(self.y)), int(round(self.w)), int(round(self.h)))

    def intersection_area(self, other: "BBox") -> float:
        ax1, ay1, ax2, ay2 = self.xyxy
        bx1, by1, bx2, by2 = other.xyxy
        iw = min(ax2, bx2) - max(ax1, bx1)
        ih = min(ay2, by2) - max(ay1, by1)
        if iw <= 0.0 or ih <= 0.0:
            return 0.0
        return iw * ih

    def iou(self, other: "BBox") -> float:
        inter = self.intersection_area(other)
        if inter <= 0.0:
            return 0.0
        union = self.area + other.area - inter
        return inter / union if union > 0.0 else 0.0

    def contains_point(self, px: float, py: float) -> bool:
        x1, y1, x2, y2 = self.xyxy
        return x1 <= px <= x2 and y1 <= py <= y2

    def expand(
        self,
        *,
        left: float = 0.0,
        right: float = 0.0,
        top: float = 0.0,
        bottom: float = 0.0,
    ) -> "BBox":
        """以自身寬高為單位向四方外擴（例：``left=1.1`` 代表左擴 1.1 倍框寬）。"""
        dx_l = self.w * left
        dx_r = self.w * right
        dy_t = self.h * top
        dy_b = self.h * bottom
        return BBox(self.x - dx_l, self.y - dy_t, self.w + dx_l + dx_r, self.h + dy_t + dy_b)

    def clip(self, width: float, height: float) -> "BBox":
        x1 = min(max(self.x, 0.0), width)
        y1 = min(max(self.y, 0.0), height)
        x2 = min(max(self.x + self.w, 0.0), width)
        y2 = min(max(self.y + self.h, 0.0), height)
        return BBox(x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1))

    def scaled(self, factor: float) -> "BBox":
        """整體座標縮放（用於偵測解析度還原到原始影格）。"""
        return BBox(self.x * factor, self.y * factor, self.w * factor, self.h * factor)


@dataclass(frozen=True)
class Detection:
    """單一臉／頭部偵測結果。

    ``landmarks`` 僅供繪圖與品質判斷，**不得**用於身分比對或儲存。
    """

    bbox: BBox
    score: float
    landmarks: tuple[tuple[float, float], ...] = field(default=())

    def area_ratio(self, frame_width: int, frame_height: int) -> float:
        frame_area = float(frame_width) * float(frame_height)
        if frame_area <= 0.0:
            return 0.0
        return self.bbox.area / frame_area


class Detector(ABC):
    """所有偵測器共同介面。"""

    #: 供設定檔與報表使用的短名稱。
    name: str = "detector"

    @abstractmethod
    def detect(self, image: Any) -> list[Detection]:
        """輸入 BGR ndarray，回傳原始影像座標系的偵測清單（未做業務過濾）。"""

    @abstractmethod
    def describe(self) -> dict[str, Any]:
        """回報套件版本、模型路徑與 SHA-256、輸入尺寸等，供交付回報使用。"""

    def close(self) -> None:  # pragma: no cover - 預設無資源需釋放
        return None

    def __enter__(self) -> "Detector":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def filter_detections(
    detections: Sequence[Detection],
    *,
    frame_width: int,
    frame_height: int,
    min_confidence: float,
    min_face_area_ratio: float,
) -> list[Detection]:
    """套用規格 §4 的有效臉框門檻，回傳依信心值遞減排序的結果。

    這是**主角資格**用的門檻：夠大、夠清楚，才適合被追成主角。
    「畫面裡還有沒有別人」請改用 :func:`filter_presence`——用同一把尺會
    把站得較遠的第二人濾掉，變成漏報。
    """
    kept = [
        d
        for d in detections
        if d.score >= min_confidence
        and d.area_ratio(frame_width, frame_height) >= min_face_area_ratio
    ]
    kept.sort(key=lambda d: d.score, reverse=True)
    return kept


def filter_presence(
    detections: Sequence[Detection],
    *,
    frame_width: int,
    frame_height: int,
    min_confidence: float,
    min_face_area_ratio: float,
) -> list[Detection]:
    """**在場人數**用的門檻，刻意比主角資格寬鬆。

    產品語意是「畫面必須是單人主構圖」，因此只要畫面裡還有第二個人就該
    示警——即使那個人站得較遠、臉較小，不足以當主角。兩者共用同一個門檻
    會造成漏報。
    """
    kept = [
        d
        for d in detections
        if d.score >= min_confidence
        and d.area_ratio(frame_width, frame_height) >= min_face_area_ratio
    ]
    kept.sort(key=lambda d: d.score, reverse=True)
    return kept
