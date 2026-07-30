"""主角追蹤（規格 §4）。

要點：
* 主角初始化＝「最大且最接近中央」的有效臉框。
* 之後以 IoU、中心距離、框面積變化維持；**不得每幀重新選主角**。
* 短暫漏偵測不視為換人，也不視為離場——僅標記 ``visible=False``，
  由 :mod:`src.rules` 依時間門檻決定是否示警。

本模組只依賴 :mod:`src.detectors.base` 的資料型別，不含任何模型知識。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Any, Mapping, Sequence

from .detectors.base import BBox, Detection
from .validation import ConfigError, as_mapping, check_fields

__all__ = ["TrackerConfig", "PrimarySubject", "TrackerResult", "PrimarySubjectTracker"]


@dataclass(frozen=True)
class TrackerConfig:
    """追蹤參數；權重之和不必為 1，比對時會正規化。"""

    iou_weight: float = 0.5
    center_weight: float = 0.3
    area_weight: float = 0.2
    #: 綜合比對分數低於此值視為不是同一人。
    min_match_score: float = 0.35
    #: 單次更新允許的中心位移上限（以影格對角線為單位）。
    max_center_shift_ratio: float = 0.25
    #: 單次更新允許的面積比（大/小）上限。
    max_area_ratio: float = 3.0
    #: 連續看不到主角超過此毫秒數即釋放主角身分，允許重新初始化。
    #: 必須 >= rules.primary_left_ms，否則會在示警前就把身分讓給別人。
    lost_after_ms: int = 3000
    #: 主角框的指數平滑係數（0=不平滑，越大越黏）。
    smoothing: float = 0.35

    @classmethod
    def from_dict(cls, cfg: Mapping[str, Any] | None) -> "TrackerConfig":
        cfg = as_mapping("tracker", cfg)
        check_fields("tracker", cfg, cls.__dataclass_fields__)
        return cls(**cfg)


@dataclass(frozen=True)
class PrimarySubject:
    """目前主角狀態的快照。"""

    bbox: BBox
    score: float
    #: 本次更新是否真的被偵測到（False = 沿用上次框，處於漏偵測寬限期）。
    visible: bool
    first_seen_ms: int
    last_seen_ms: int
    #: 自上次真正被偵測到起經過的毫秒數。
    missing_ms: int
    #: 成功匹配的更新次數，用於判斷追蹤是否已穩定。
    hits: int
    track_id: int


@dataclass
class TrackerResult:
    primary: PrimarySubject | None
    person_count: int
    detections: tuple[Detection, ...] = field(default=())
    #: 本次是否（重新）初始化主角。
    initialized: bool = False
    #: 本次是否釋放主角身分。
    released: bool = False


def _center_distance(a: BBox, b: BBox) -> float:
    return math.hypot(a.cx - b.cx, a.cy - b.cy)


def _area_consistency(a: BBox, b: BBox) -> float:
    aa, ab = a.area, b.area
    if aa <= 0.0 or ab <= 0.0:
        return 0.0
    return min(aa, ab) / max(aa, ab)


class PrimarySubjectTracker:
    """單一主角追蹤器。"""

    def __init__(self, config: TrackerConfig | None = None) -> None:
        self.config = config or TrackerConfig()
        self._primary: PrimarySubject | None = None
        self._next_track_id = 1

    # ------------------------------------------------------------------ #
    # 查詢
    # ------------------------------------------------------------------ #

    @property
    def primary(self) -> PrimarySubject | None:
        return self._primary

    def reset(self) -> None:
        self._primary = None

    # ------------------------------------------------------------------ #
    # 初始化
    # ------------------------------------------------------------------ #

    def _selection_score(
        self, det: Detection, frame_width: int, frame_height: int
    ) -> float:
        """「最大且最接近中央」的綜合分數。"""
        frame_area = float(frame_width) * float(frame_height)
        size_term = math.sqrt(det.bbox.area / frame_area) if frame_area > 0 else 0.0
        half_diag = math.hypot(frame_width, frame_height) / 2.0
        offset = math.hypot(det.bbox.cx - frame_width / 2.0, det.bbox.cy - frame_height / 2.0)
        center_term = 1.0 - min(1.0, offset / half_diag) if half_diag > 0 else 0.0
        return 0.5 * size_term + 0.5 * center_term

    def _initialize(
        self,
        detections: Sequence[Detection],
        timestamp_ms: int,
        frame_width: int,
        frame_height: int,
    ) -> PrimarySubject:
        best = max(
            detections,
            key=lambda d: self._selection_score(d, frame_width, frame_height),
        )
        subject = PrimarySubject(
            bbox=best.bbox,
            score=best.score,
            visible=True,
            first_seen_ms=timestamp_ms,
            last_seen_ms=timestamp_ms,
            missing_ms=0,
            hits=1,
            track_id=self._next_track_id,
        )
        self._next_track_id += 1
        return subject

    # ------------------------------------------------------------------ #
    # 維持
    # ------------------------------------------------------------------ #

    def _match_score(self, current: BBox, candidate: BBox, frame_diag: float) -> float:
        cfg = self.config
        iou = current.iou(candidate)
        shift = _center_distance(current, candidate)
        closeness = 1.0 - min(1.0, shift / frame_diag) if frame_diag > 0 else 0.0
        area_term = _area_consistency(current, candidate)
        total_weight = cfg.iou_weight + cfg.center_weight + cfg.area_weight
        if total_weight <= 0.0:
            return 0.0
        return (
            cfg.iou_weight * iou + cfg.center_weight * closeness + cfg.area_weight * area_term
        ) / total_weight

    def _find_match(
        self,
        detections: Sequence[Detection],
        current: BBox,
        frame_width: int,
        frame_height: int,
    ) -> Detection | None:
        cfg = self.config
        frame_diag = math.hypot(frame_width, frame_height)
        best: Detection | None = None
        best_score = 0.0
        for det in detections:
            # 硬性門檻：位移與面積變化超過上限，直接視為不同人。
            if frame_diag > 0 and _center_distance(current, det.bbox) > cfg.max_center_shift_ratio * frame_diag:
                continue
            consistency = _area_consistency(current, det.bbox)
            if consistency <= 0.0 or (1.0 / consistency) > cfg.max_area_ratio:
                continue
            score = self._match_score(current, det.bbox, frame_diag)
            if score >= cfg.min_match_score and score > best_score:
                best, best_score = det, score
        return best

    def _smooth(self, previous: BBox, observed: BBox) -> BBox:
        alpha = max(0.0, min(1.0, self.config.smoothing))
        if alpha <= 0.0:
            return observed
        return BBox(
            alpha * previous.x + (1 - alpha) * observed.x,
            alpha * previous.y + (1 - alpha) * observed.y,
            alpha * previous.w + (1 - alpha) * observed.w,
            alpha * previous.h + (1 - alpha) * observed.h,
        )

    # ------------------------------------------------------------------ #
    # 主要進入點
    # ------------------------------------------------------------------ #

    def update(
        self,
        detections: Sequence[Detection],
        *,
        timestamp_ms: int,
        frame_size: tuple[int, int],
    ) -> TrackerResult:
        """以本影格的有效偵測更新主角狀態。

        ``detections`` 應已由 :func:`src.detectors.base.filter_detections`
        過濾過（min_confidence／min_face_area_ratio）。
        ``frame_size`` 為 ``(width, height)``。
        """
        frame_width, frame_height = frame_size
        detections = tuple(detections)
        person_count = len(detections)

        # 尚無主角：只有在有有效臉框時才初始化。
        if self._primary is None:
            if not detections:
                return TrackerResult(primary=None, person_count=0, detections=detections)
            self._primary = self._initialize(detections, timestamp_ms, frame_width, frame_height)
            return TrackerResult(
                primary=self._primary,
                person_count=person_count,
                detections=detections,
                initialized=True,
            )

        current = self._primary
        matched = self._find_match(detections, current.bbox, frame_width, frame_height)

        if matched is not None:
            self._primary = replace(
                current,
                bbox=self._smooth(current.bbox, matched.bbox),
                score=matched.score,
                visible=True,
                last_seen_ms=timestamp_ms,
                missing_ms=0,
                hits=current.hits + 1,
            )
            return TrackerResult(
                primary=self._primary, person_count=person_count, detections=detections
            )

        # 沒有匹配：進入漏偵測寬限期，沿用最後已知框。
        missing_ms = max(0, timestamp_ms - current.last_seen_ms)
        if missing_ms > self.config.lost_after_ms:
            self._primary = None
            # 主角身分已釋放；本幀不立即改認新主角，下一幀才重新初始化，
            # 避免「離場當下畫面剛好有路人」造成瞬間換人。
            return TrackerResult(
                primary=None,
                person_count=person_count,
                detections=detections,
                released=True,
            )

        self._primary = replace(current, visible=False, missing_ms=missing_ms)
        return TrackerResult(
            primary=self._primary, person_count=person_count, detections=detections
        )
