"""時間規則引擎（規格 §4、§6）。

驗收重點：
* 單幀漏偵測不得示警。
* 門檻到達後 250 ms 內示警。
* 第二人持續超過門檻才觸發。
"""

from __future__ import annotations

import pytest
from conftest import FRAME_H, FRAME_W, face

from focus_keeper.validation import ConfigError
from focus_keeper.detectors.base import BBox
from focus_keeper.rules import (
    ROI,
    EventType,
    RuleConfig,
    ShoulderBoxConfig,
    TemporalRuleEngine,
    estimate_shoulder_box,
)
from focus_keeper.tracker import PrimarySubject

FRAME = (FRAME_W, FRAME_H)
#: 10 FPS 推論 -> 每 100 ms 一次判定。
STEP_MS = 100
MAX_ALERT_LATENCY_MS = 250


def subject(
    *, visible: bool = True, cx: float = FRAME_W / 2, cy: float = FRAME_H / 2,
    size: float = 160.0, score: float = 0.9, timestamp_ms: int = 0,
) -> PrimarySubject:
    bbox = face(cx, cy, size=size).bbox
    return PrimarySubject(
        bbox=bbox, score=score, visible=visible, first_seen_ms=0,
        last_seen_ms=timestamp_ms, missing_ms=0, hits=1, track_id=1,
    )


def engine(**overrides) -> TemporalRuleEngine:
    return TemporalRuleEngine(RuleConfig(**overrides))


def feed(
    rules: TemporalRuleEngine, frames, *, start_ms: int = 0, step_ms: int = STEP_MS
):
    """把 ``(person_count, primary)`` 序列以固定節拍送進規則引擎。"""
    updates = []
    for i, (person_count, primary) in enumerate(frames):
        updates.append(
            rules.update(
                timestamp_ms=start_ms + i * step_ms,
                person_count=person_count,
                primary=primary,
                frame_size=FRAME,
            )
        )
    return updates


def started_types(updates) -> list[EventType]:
    return [e.type for u in updates for e in u.started]


def first_event(updates, event_type: EventType):
    for update in updates:
        for event in update.started:
            if event.type is event_type:
                return event
    return None


class TestNormal:
    def test_stable_primary_stays_normal(self) -> None:
        rules = engine()
        updates = feed(rules, [(1, subject(timestamp_ms=i * STEP_MS)) for i in range(40)])
        assert all(u.status is EventType.NORMAL for u in updates)
        assert started_types(updates) == []


class TestSingleFrameMiss:
    def test_single_dropped_frame_never_alerts(self) -> None:
        """規格：單幀漏偵測不得示警。"""
        rules = engine()
        frames = []
        for i in range(60):
            missed = i % 7 == 3  # 每 7 幀漏 1 幀（100 ms），遠短於 1000 ms 門檻
            frames.append((0 if missed else 1, subject(visible=not missed)))
        updates = feed(rules, frames)
        assert started_types(updates) == []

    def test_two_consecutive_misses_still_below_threshold(self) -> None:
        rules = engine(primary_missing_warn_ms=1000)
        frames = [(1, subject())] * 5 + [(0, subject(visible=False))] * 2 + [(1, subject())] * 20
        updates = feed(rules, frames)
        assert started_types(updates) == []


class TestPrimaryMissing:
    def test_fires_after_threshold_within_250ms(self) -> None:
        rules = engine(primary_missing_warn_ms=1000)
        frames = [(1, subject())] * 3 + [(0, subject(visible=False))] * 20
        updates = feed(rules, frames)

        event = first_event(updates, EventType.PRIMARY_MISSING)
        assert event is not None
        assert event.hold_ms >= 1000
        assert event.alert_latency_ms(1000) <= MAX_ALERT_LATENCY_MS

    def test_escalates_to_left(self) -> None:
        rules = engine(primary_missing_warn_ms=1000, primary_left_ms=2000)
        frames = [(1, subject())] * 3 + [(0, None)] * 40
        updates = feed(rules, frames)

        missing = first_event(updates, EventType.PRIMARY_MISSING)
        left = first_event(updates, EventType.PRIMARY_LEFT)
        assert missing is not None and left is not None
        assert left.triggered_ms > missing.triggered_ms
        assert left.alert_latency_ms(2000) <= MAX_ALERT_LATENCY_MS
        # LEFT 優先序高於 MISSING，畫面狀態應顯示 LEFT。
        assert updates[-1].status is EventType.PRIMARY_LEFT

    def test_recovery_closes_event(self) -> None:
        rules = engine(primary_missing_warn_ms=1000, event_release_ms=200)
        frames = [(1, subject())] * 3 + [(0, None)] * 15 + [(1, subject())] * 10
        updates = feed(rules, frames)

        ended = [e for u in updates for e in u.ended if e.type is EventType.PRIMARY_MISSING]
        assert len(ended) == 1
        assert ended[0].end_ms is not None
        assert updates[-1].status is EventType.NORMAL

    def test_release_hysteresis_ignores_single_stray_frame(self) -> None:
        """事件進行中出現單幀誤偵測，不得把事件關掉再重開。"""
        rules = engine(primary_missing_warn_ms=1000, event_release_ms=200)
        frames = (
            [(1, subject())] * 3
            + [(0, None)] * 15
            + [(1, subject())]        # 單幀（100 ms）雜訊，短於 release 200 ms
            + [(0, None)] * 15
        )
        updates = feed(rules, frames)
        assert started_types(updates).count(EventType.PRIMARY_MISSING) == 1
        assert [e for u in updates for e in u.ended] == []


class TestFlickerDoesNotAccumulate:
    """回歸（2026-07-29 code review 第 1 項，HIGH）。

    原本「示警前的連續累積」與「示警後的解除遲滯」共用同一段邏輯，導致
    偵測時有時無的主角被當成連續不在場：50% 閃爍會在 1.1 秒誤報
    PRIMARY_MISSING、2.1 秒誤報 PRIMARY_LEFT，而當事人有一半的時間
    清清楚楚在畫面上。這正是真實素材上側臉／手遮臉時的行為
    （信心值在 min_confidence 邊界跳動）。

    修正後：門檻要求**連續**成立，任何一次不成立就重新起算。
    """

    @staticmethod
    def _fire(pattern, frames: int = 60):
        rules = engine()
        fired = []
        for i in range(frames):
            seen = pattern[i % len(pattern)]
            update = rules.update(
                timestamp_ms=i * STEP_MS,
                person_count=1 if seen else 0,
                primary=subject(visible=seen),
                frame_size=FRAME,
            )
            fired.extend(e.type for e in update.started)
        return fired

    @pytest.mark.parametrize(
        "pattern,label",
        [
            ([True, False], "50%"),
            ([True, True, False], "67%"),
            ([True, True, True, False], "75%"),
            ([True] * 6 + [False], "86%"),
            ([True, False, False], "33%"),
        ],
    )
    def test_intermittent_detection_never_alerts(self, pattern, label: str) -> None:
        """只要主角**曾經**被看到就打斷連續性，不得累積成離場示警。"""
        assert self._fire(pattern) == [], f"{label} 偵測率的閃爍造成誤報"

    def test_genuine_absence_still_alerts(self) -> None:
        """修正不得把真正的離場也一起吃掉。"""
        fired = self._fire([False])
        assert EventType.PRIMARY_MISSING in fired
        assert EventType.PRIMARY_LEFT in fired

    def test_one_detection_resets_the_clock(self) -> None:
        """離場計時途中出現一次偵測，門檻必須重新起算。"""
        rules = engine(primary_missing_warn_ms=1000)
        # 缺 900 ms -> 出現 1 次 -> 再缺 900 ms：兩段都不足 1000 ms。
        frames = (
            [(0, subject(visible=False))] * 9
            + [(1, subject(visible=True))]
            + [(0, subject(visible=False))] * 9
        )
        updates = feed(rules, frames)
        assert started_types(updates) == []

        # 但重新起算後若真的連續缺滿門檻，仍要示警。
        more = feed(
            rules,
            [(0, subject(visible=False))] * 15,
            start_ms=len(frames) * STEP_MS,
        )
        assert EventType.PRIMARY_MISSING in started_types(more)


class TestMultiPerson:
    def test_sustained_second_person_triggers(self) -> None:
        rules = engine(multi_person_hold_ms=500)
        frames = [(1, subject())] * 5 + [(2, subject())] * 15
        updates = feed(rules, frames)

        event = first_event(updates, EventType.MULTI_PERSON)
        assert event is not None
        assert event.hold_ms >= 500
        assert event.alert_latency_ms(500) <= MAX_ALERT_LATENCY_MS
        assert event.max_person_count == 2

    def test_brief_passerby_does_not_trigger(self) -> None:
        rules = engine(multi_person_hold_ms=500)
        frames = [(1, subject())] * 5 + [(2, subject())] * 4 + [(1, subject())] * 15
        updates = feed(rules, frames)
        assert EventType.MULTI_PERSON not in started_types(updates)

    def test_records_peak_person_count(self) -> None:
        rules = engine(multi_person_hold_ms=500)
        frames = [(2, subject())] * 8 + [(4, subject())] * 5 + [(2, subject())] * 5
        updates = feed(rules, frames)
        event = first_event(updates, EventType.MULTI_PERSON)
        assert event is not None
        assert event.max_person_count == 4


class TestFeedFrozen:
    """畫面停格（2026-07-29 新增）。

    這是監測系統最危險的失效：影像來源停止更新時畫面不會變黑，而是停在
    最後一格；若那一格有人，所有判定會永遠回報 NORMAL——監測早就死了
    卻毫無跡象。無聲失效必須可被看見。
    """

    def test_static_feed_alerts(self) -> None:
        rules = engine(feed_frozen_ms=3000)
        frames = [(1, subject())] * 60
        updates = []
        for i, (count, primary) in enumerate(frames):
            updates.append(
                rules.update(
                    timestamp_ms=i * STEP_MS, person_count=count,
                    primary=primary, frame_size=FRAME, frame_static=True,
                )
            )
        event = first_event(updates, EventType.FEED_FROZEN)
        assert event is not None, "畫面持續停格卻沒有示警"
        assert event.hold_ms >= 3000
        assert event.alert_latency_ms(3000) <= MAX_ALERT_LATENCY_MS

    def test_occasional_duplicate_frame_does_not_alert(self) -> None:
        """真實攝影機會偶爾送出完全相同的影格（實測 min diff = 0.0000），
        單格相同不得被當成停格。"""
        rules = engine(feed_frozen_ms=3000)
        updates = []
        for i in range(80):
            updates.append(
                rules.update(
                    timestamp_ms=i * STEP_MS, person_count=1, primary=subject(),
                    frame_size=FRAME, frame_static=(i % 4 == 0),
                )
            )
        assert EventType.FEED_FROZEN not in started_types(updates)

    def test_takes_priority_over_other_states(self) -> None:
        """停格時其他判定都不可信，畫面狀態必須顯示停格。"""
        rules = engine(feed_frozen_ms=500)
        for i in range(20):
            update = rules.update(
                timestamp_ms=i * STEP_MS, person_count=3, primary=subject(),
                frame_size=FRAME, frame_static=True,
            )
        assert update.status is EventType.FEED_FROZEN

    def test_can_be_disabled(self) -> None:
        rules = engine(feed_frozen_ms=0)
        updates = []
        for i in range(80):
            updates.append(
                rules.update(
                    timestamp_ms=i * STEP_MS, person_count=1, primary=subject(),
                    frame_size=FRAME, frame_static=True,
                )
            )
        assert EventType.FEED_FROZEN not in started_types(updates)


class TestRoi:
    def test_centered_subject_is_inside(self) -> None:
        rules = engine()
        outside, shoulder, roi_px = rules.is_outside_roi(
            face(FRAME_W / 2, FRAME_H / 2, size=160).bbox, FRAME_W, FRAME_H
        )
        assert outside is False
        assert shoulder.area > 0
        assert roi_px.w == pytest.approx(FRAME_W * 0.80)

    def test_edge_subject_is_outside(self) -> None:
        rules = engine()
        outside, _, _ = rules.is_outside_roi(
            face(40, FRAME_H / 2, size=160).bbox, FRAME_W, FRAME_H
        )
        assert outside is True

    def test_outside_roi_event_respects_threshold(self) -> None:
        rules = engine(primary_outside_roi_ms=800)
        drifted = subject(cx=30)
        frames = [(1, subject())] * 3 + [(1, drifted)] * 15
        updates = feed(rules, frames)

        event = first_event(updates, EventType.PRIMARY_OUTSIDE_ROI)
        assert event is not None
        assert event.hold_ms >= 800
        assert event.alert_latency_ms(800) <= MAX_ALERT_LATENCY_MS

    def test_brief_drift_does_not_trigger(self) -> None:
        rules = engine(primary_outside_roi_ms=800)
        frames = [(1, subject())] * 3 + [(1, subject(cx=30))] * 5 + [(1, subject())] * 10
        updates = feed(rules, frames)
        assert EventType.PRIMARY_OUTSIDE_ROI not in started_types(updates)

    def test_invisible_primary_is_not_reported_as_outside_roi(self) -> None:
        """看不到主角時屬 MISSING/LEFT，不得同時被判成偏離構圖區。"""
        rules = engine()
        frames = [(1, subject())] * 3 + [(0, subject(visible=False, cx=30))] * 20
        updates = feed(rules, frames)
        assert EventType.PRIMARY_OUTSIDE_ROI not in started_types(updates)


class TestShoulderBox:
    def test_expands_around_face(self) -> None:
        cfg = ShoulderBoxConfig(left=1.0, right=1.0, top=0.5, bottom=2.0)
        f = BBox(500, 300, 100, 120)
        shoulder = estimate_shoulder_box(f, cfg, FRAME_W, FRAME_H)
        assert shoulder.x == pytest.approx(400)
        assert shoulder.w == pytest.approx(300)
        assert shoulder.y == pytest.approx(240)
        assert shoulder.h == pytest.approx(120 + 60 + 240)

    def test_clipped_to_frame(self) -> None:
        cfg = ShoulderBoxConfig()
        shoulder = estimate_shoulder_box(BBox(0, 0, 100, 120), cfg, FRAME_W, FRAME_H)
        assert shoulder.x >= 0 and shoulder.y >= 0
        assert shoulder.x + shoulder.w <= FRAME_W
        assert shoulder.y + shoulder.h <= FRAME_H


class TestFlushAndConfig:
    def test_flush_closes_open_events(self) -> None:
        rules = engine(primary_missing_warn_ms=1000, primary_left_ms=2000)
        # 25 幀 x 100 ms = 0..2400 ms，足以同時開啟 MISSING(1000) 與 LEFT(2000)。
        feed(rules, [(0, None)] * 25)
        closed = rules.flush(9999)
        assert {e.type for e in closed} >= {EventType.PRIMARY_MISSING, EventType.PRIMARY_LEFT}
        assert all(e.end_ms == 9999 for e in closed)

    def test_rejects_inconsistent_thresholds(self) -> None:
        with pytest.raises(ValueError):
            RuleConfig.from_dict({"primary_missing_warn_ms": 3000, "primary_left_ms": 1000})

    def test_rejects_unknown_field(self) -> None:
        with pytest.raises(ConfigError):
            RuleConfig.from_dict({"typo_ms": 100})

    def test_roi_bounds_validated(self) -> None:
        with pytest.raises(ValueError):
            ROI.from_dict({"w": 0.0})
