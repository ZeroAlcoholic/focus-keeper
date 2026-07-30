"""時間規則引擎（規格 §4）。

事件：``NORMAL`` / ``MULTI_PERSON`` / ``PRIMARY_MISSING`` /
``PRIMARY_LEFT`` / ``PRIMARY_OUTSIDE_ROI``。

核心設計：每個異常條件各自維護一個「連續成立起始時間」，成立時間
達到 ``hold_ms`` 才開真事件——**單幀漏偵測不得示警**。條件轉為不成立後
還要連續維持 ``release_ms`` 才關事件（遲滯），避免單幀誤偵測讓事件抖動。

本模組不引用任何模型或 OpenCV，可純以合成資料測試。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable, Mapping

from .detectors.base import BBox
from .tracker import PrimarySubject
from .validation import ConfigError, as_mapping, check_fields

__all__ = [
    "EventType",
    "ROI",
    "ShoulderBoxConfig",
    "RuleConfig",
    "Event",
    "RuleUpdate",
    "TemporalRuleEngine",
    "estimate_shoulder_box",
]


class EventType(str, Enum):
    NORMAL = "NORMAL"
    MULTI_PERSON = "MULTI_PERSON"
    PRIMARY_MISSING = "PRIMARY_MISSING"
    PRIMARY_LEFT = "PRIMARY_LEFT"
    PRIMARY_OUTSIDE_ROI = "PRIMARY_OUTSIDE_ROI"
    #: 畫面停格——影像來源已經不再更新（USB 當掉、驅動凍結、虛擬攝影機停住）。
    #: 這時所有其他判定都失去意義：停在有人的那一格會永遠回報 NORMAL，
    #: 監測實際上已經死了卻不會有任何跡象。無聲失效必須可被看見。
    FEED_FROZEN = "FEED_FROZEN"


#: 顯示用優先序（數字大者優先），僅影響畫面上的單一狀態文字，
#: 不影響 JSONL——JSONL 記錄所有同時成立的事件。
_PRIORITY: dict[EventType, int] = {
    EventType.NORMAL: 0,
    EventType.MULTI_PERSON: 1,
    EventType.PRIMARY_OUTSIDE_ROI: 2,
    EventType.PRIMARY_MISSING: 3,
    EventType.PRIMARY_LEFT: 4,
    # 最高優先：畫面停格時，其他判定都不可信，操作者必須先知道這件事。
    EventType.FEED_FROZEN: 5,
}


@dataclass(frozen=True)
class ROI:
    """正規化（0~1）的主要構圖區。"""

    x: float = 0.10
    y: float = 0.02
    w: float = 0.80
    h: float = 0.96

    def to_pixels(self, frame_width: int, frame_height: int) -> BBox:
        return BBox(
            self.x * frame_width,
            self.y * frame_height,
            self.w * frame_width,
            self.h * frame_height,
        )

    @classmethod
    def from_dict(cls, cfg: Mapping[str, Any] | None) -> "ROI":
        cfg = as_mapping("rules.roi", cfg)
        check_fields("rules.roi", cfg, cls.__dataclass_fields__)
        roi = cls(**cfg)
        if not (0.0 <= roi.x < 1.0 and 0.0 <= roi.y < 1.0):
            raise ConfigError("roi.x / roi.y 必須落在 [0, 1)")
        if not (0.0 < roi.w <= 1.0 and 0.0 < roi.h <= 1.0):
            raise ConfigError("roi.w / roi.h 必須落在 (0, 1]")
        return roi


@dataclass(frozen=True)
class ShoulderBoxConfig:
    """由臉框外擴推估的肩上構圖框比例（以臉框寬高為單位）。

    僅供 ROI／構圖判斷與畫面標示，**不宣稱為肩部模型偵測結果**。
    """

    left: float = 0.85
    right: float = 0.85
    top: float = 0.45
    bottom: float = 1.60

    @classmethod
    def from_dict(cls, cfg: Mapping[str, Any] | None) -> "ShoulderBoxConfig":
        cfg = as_mapping("rules.shoulder_box", cfg)
        check_fields("rules.shoulder_box", cfg, cls.__dataclass_fields__)
        return cls(**cfg)


def estimate_shoulder_box(
    face: BBox, cfg: ShoulderBoxConfig, frame_width: int, frame_height: int
) -> BBox:
    """由臉框外擴推估肩上構圖框，並裁切到影格範圍內。"""
    return face.expand(
        left=cfg.left, right=cfg.right, top=cfg.top, bottom=cfg.bottom
    ).clip(frame_width, frame_height)


@dataclass(frozen=True)
class RuleConfig:
    """時間門檻與構圖判定參數（對應規格 §4 的 config 區塊）。"""

    multi_person_hold_ms: int = 500
    primary_missing_warn_ms: int = 1000
    primary_left_ms: int = 2000
    primary_outside_roi_ms: int = 800
    #: 畫面連續多久沒有任何變化就判定為停格。0 表示停用此檢查。
    #: 真人在鏡頭前必定有微動（呼吸、眨眼、感測器雜訊），連續數秒完全不變
    #: 幾乎只會是來源本身停止更新。
    feed_frozen_ms: int = 3000
    #: 條件轉為不成立後，需連續維持多久才關閉事件（遲滯）。
    event_release_ms: int = 200
    #: 觸發多人事件的人數下限。
    multi_person_min_count: int = 2
    #: 肩上構圖框落在 ROI 內的最低面積比例。
    min_roi_overlap: float = 0.45
    roi: ROI = field(default_factory=ROI)
    shoulder_box: ShoulderBoxConfig = field(default_factory=ShoulderBoxConfig)

    @classmethod
    def from_dict(cls, cfg: Mapping[str, Any] | None) -> "RuleConfig":
        cfg = as_mapping("rules", cfg)
        roi = ROI.from_dict(cfg.pop("roi", None))
        shoulder = ShoulderBoxConfig.from_dict(cfg.pop("shoulder_box", None))
        check_fields(
            "rules", cfg, set(cls.__dataclass_fields__) - {"roi", "shoulder_box"}
        )
        instance = cls(roi=roi, shoulder_box=shoulder, **cfg)
        if instance.primary_left_ms < instance.primary_missing_warn_ms:
            raise ConfigError("primary_left_ms 必須 >= primary_missing_warn_ms")
        return instance


@dataclass
class Event:
    """一次異常事件；``end_ms`` 為 ``None`` 代表仍在進行中。"""

    type: EventType
    #: 條件開始連續成立的時間。
    condition_start_ms: int
    #: 跨過 hold 門檻、實際示警的那一幀時間。
    triggered_ms: int
    end_ms: int | None = None
    #: 事件期間最大人數。
    max_person_count: int = 0
    #: 事件觸發當下主角的信心值（無主角時為 None）。
    trigger_confidence: float | None = None
    #: 事件期間主角信心值的最小值。
    min_confidence: float | None = None

    @property
    def hold_ms(self) -> int:
        """從條件成立到示警實際經過的毫秒數。"""
        return self.triggered_ms - self.condition_start_ms

    def alert_latency_ms(self, threshold_ms: int) -> int:
        """超出門檻後才示警的額外延遲。

        .. warning::
           基準 ``condition_start_ms`` 本身是「第一個**觀測到**條件成立的取樣點」，
           與 ``triggered_ms`` 被同一個取樣格量化，因此本值幾乎恆為 0，
           **量不到取樣造成的延遲**。真實延遲要以異常實際發生的時刻為基準：

               真實延遲 = triggered_ms - (真實異常時刻 + threshold_ms)

           真實異常時刻只有離線標註才知道，執行期無從得知，所以這裡只能回報
           觀測基準值。實測掃描全取樣格後，真實延遲為 0~95 ms（10 FPS 推論、
           30 FPS 來源），仍在 250 ms 驗收上限內；理論上界＝一個推論週期＋
           一個來源影格週期。見 tests/test_pipeline.py::TestTrueAlertLatency。
        """
        return max(0, self.hold_ms - threshold_ms)

    def duration_ms(self, now_ms: int | None = None) -> int:
        end = self.end_ms if self.end_ms is not None else now_ms
        if end is None:
            return 0
        return max(0, end - self.condition_start_ms)

    def to_record(self, threshold_ms: int | None = None) -> dict[str, Any]:
        record: dict[str, Any] = {
            "event": self.type.value,
            "condition_start_ms": self.condition_start_ms,
            "triggered_ms": self.triggered_ms,
            "end_ms": self.end_ms,
            "duration_ms": self.duration_ms(),
            "max_person_count": self.max_person_count,
            "trigger_confidence": self.trigger_confidence,
            "min_confidence": self.min_confidence,
        }
        if threshold_ms is not None:
            record["threshold_ms"] = threshold_ms
            record["alert_latency_ms"] = self.alert_latency_ms(threshold_ms)
        return record


@dataclass
class RuleUpdate:
    """單次更新的結果。"""

    timestamp_ms: int
    status: EventType
    active: tuple[Event, ...] = field(default=())
    started: tuple[Event, ...] = field(default=())
    ended: tuple[Event, ...] = field(default=())
    person_count: int = 0
    #: 目前主角的肩上構圖框（無主角時為 None），供 overlay 使用。
    shoulder_box: BBox | None = None
    roi_box: BBox | None = None
    primary_outside_roi: bool = False


class _Condition:
    """單一異常條件的時間狀態機。"""

    __slots__ = ("event_type", "hold_ms", "release_ms", "since_ms", "false_since_ms", "event")

    def __init__(self, event_type: EventType, hold_ms: int, release_ms: int) -> None:
        self.event_type = event_type
        self.hold_ms = int(hold_ms)
        self.release_ms = int(release_ms)
        self.since_ms: int | None = None
        self.false_since_ms: int | None = None
        self.event: Event | None = None

    def update(
        self,
        *,
        active: bool,
        timestamp_ms: int,
        person_count: int,
        confidence: float | None,
    ) -> tuple[Event | None, Event | None]:
        """回傳 ``(started_event, ended_event)``。"""
        started: Event | None = None
        ended: Event | None = None

        if active:
            self.false_since_ms = None
            if self.since_ms is None:
                self.since_ms = timestamp_ms
            if self.event is None and (timestamp_ms - self.since_ms) >= self.hold_ms:
                self.event = Event(
                    type=self.event_type,
                    condition_start_ms=self.since_ms,
                    triggered_ms=timestamp_ms,
                    max_person_count=person_count,
                    trigger_confidence=confidence,
                    min_confidence=confidence,
                )
                started = self.event
        elif self.event is None:
            # 尚未示警：門檻要求的是**連續**成立，任何一次不成立就打斷、重新起算。
            #
            # 這裡不能套用 release_ms 遲滯。若套用，偵測時有時無（例如側臉、
            # 手遮臉造成信心值在門檻邊界跳動）的主角會被當成連續不在場：
            # 50% 閃爍會在 1.1 秒誤報 PRIMARY_MISSING、2.1 秒誤報 PRIMARY_LEFT，
            # 而當事人其實有一半的時間清清楚楚在畫面上。
            self.since_ms = None
            self.false_since_ms = None
        else:
            # 已示警：反過來要求**連續**不成立 release_ms 才關閉事件，
            # 避免單幀誤偵測讓進行中的事件關掉再重開。
            if self.false_since_ms is None:
                self.false_since_ms = timestamp_ms
            if (timestamp_ms - self.false_since_ms) >= self.release_ms:
                self.event.end_ms = timestamp_ms
                ended = self.event
                self.event = None
                self.since_ms = None
                self.false_since_ms = None

        if self.event is not None:
            self.event.max_person_count = max(self.event.max_person_count, person_count)
            if confidence is not None:
                current = self.event.min_confidence
                self.event.min_confidence = (
                    confidence if current is None else min(current, confidence)
                )

        return started, ended

    def force_close(self, timestamp_ms: int) -> Event | None:
        if self.event is None:
            return None
        self.event.end_ms = timestamp_ms
        closed = self.event
        self.event = None
        self.since_ms = None
        self.false_since_ms = None
        return closed


class TemporalRuleEngine:
    """把每幀的追蹤結果轉成有時間門檻的事件序列。"""

    def __init__(self, config: RuleConfig | None = None) -> None:
        self.config = config or RuleConfig()
        release = self.config.event_release_ms
        self._conditions: dict[EventType, _Condition] = {
            EventType.MULTI_PERSON: _Condition(
                EventType.MULTI_PERSON, self.config.multi_person_hold_ms, release
            ),
            EventType.PRIMARY_MISSING: _Condition(
                EventType.PRIMARY_MISSING, self.config.primary_missing_warn_ms, release
            ),
            EventType.PRIMARY_LEFT: _Condition(
                EventType.PRIMARY_LEFT, self.config.primary_left_ms, release
            ),
            EventType.PRIMARY_OUTSIDE_ROI: _Condition(
                EventType.PRIMARY_OUTSIDE_ROI, self.config.primary_outside_roi_ms, release
            ),
        }
        if self.config.feed_frozen_ms > 0:
            self._conditions[EventType.FEED_FROZEN] = _Condition(
                EventType.FEED_FROZEN, self.config.feed_frozen_ms, release
            )

    # ------------------------------------------------------------------ #

    def threshold_for(self, event_type: EventType) -> int | None:
        condition = self._conditions.get(event_type)
        return condition.hold_ms if condition else None

    def is_outside_roi(
        self, face: BBox, frame_width: int, frame_height: int
    ) -> tuple[bool, BBox, BBox]:
        """回傳 ``(是否偏離主要構圖區, 肩上構圖框, ROI 像素框)``。"""
        cfg = self.config
        roi_px = cfg.roi.to_pixels(frame_width, frame_height)
        shoulder = estimate_shoulder_box(face, cfg.shoulder_box, frame_width, frame_height)

        center_inside = roi_px.contains_point(face.cx, face.cy)
        overlap = (
            shoulder.intersection_area(roi_px) / shoulder.area if shoulder.area > 0 else 0.0
        )
        outside = (not center_inside) or overlap < cfg.min_roi_overlap
        return outside, shoulder, roi_px

    # ------------------------------------------------------------------ #

    def update(
        self,
        *,
        timestamp_ms: int,
        person_count: int,
        primary: PrimarySubject | None,
        frame_size: tuple[int, int],
        frame_static: bool = False,
    ) -> RuleUpdate:
        frame_width, frame_height = frame_size
        cfg = self.config

        primary_visible = primary is not None and primary.visible
        confidence = primary.score if (primary is not None and primary.visible) else None

        shoulder_box: BBox | None = None
        roi_box = cfg.roi.to_pixels(frame_width, frame_height)
        outside_roi = False
        if primary is not None and primary.visible:
            outside_roi, shoulder_box, roi_box = self.is_outside_roi(
                primary.bbox, frame_width, frame_height
            )

        flags: dict[EventType, bool] = {
            EventType.MULTI_PERSON: person_count >= cfg.multi_person_min_count,
            # 主角不在畫面上：沒有主角，或處於漏偵測寬限期。
            EventType.PRIMARY_MISSING: not primary_visible,
            EventType.PRIMARY_LEFT: not primary_visible,
            # 只有看得到主角時才談「偏離構圖區」；看不到的情況屬 MISSING/LEFT。
            EventType.PRIMARY_OUTSIDE_ROI: outside_roi,
            EventType.FEED_FROZEN: frame_static,
        }

        started: list[Event] = []
        ended: list[Event] = []
        for event_type, condition in self._conditions.items():
            s, e = condition.update(
                active=flags.get(event_type, False),
                timestamp_ms=timestamp_ms,
                person_count=person_count,
                confidence=confidence,
            )
            if s is not None:
                started.append(s)
            if e is not None:
                ended.append(e)

        active = tuple(c.event for c in self._conditions.values() if c.event is not None)
        status = max(
            (e.type for e in active), key=lambda t: _PRIORITY[t], default=EventType.NORMAL
        )

        return RuleUpdate(
            timestamp_ms=timestamp_ms,
            status=status,
            active=active,
            started=tuple(started),
            ended=tuple(ended),
            person_count=person_count,
            shoulder_box=shoulder_box,
            roi_box=roi_box,
            primary_outside_roi=outside_roi,
        )

    def flush(self, timestamp_ms: int) -> tuple[Event, ...]:
        """串流結束時關閉所有仍在進行的事件。"""
        closed = [c.force_close(timestamp_ms) for c in self._conditions.values()]
        return tuple(e for e in closed if e is not None)

    def active_events(self) -> Iterable[Event]:
        return (c.event for c in self._conditions.values() if c.event is not None)
