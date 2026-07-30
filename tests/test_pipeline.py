"""分析核心：節流、事件記錄，以及 Camera／MP4 共用同一核心的可重現性。

作法：用 :class:`~conftest.ScriptedDetector` 取代真實模型，讓「同一段內容」
分別以「影片時間軸」與「攝影機時間軸」餵進 :class:`AnalysisPipeline`，
比較事件序列是否一致——差異只能來自時間戳來源，不能來自核心邏輯。
"""

from __future__ import annotations

import json
import time

import numpy as np
import pytest
from conftest import FRAME_H, FRAME_W, TimelineDetector, face

from focus_keeper.validation import ConfigError
from focus_keeper.config import PipelineConfig
from focus_keeper.eventlog import EventLogger
from focus_keeper.metrics import LatencyStats, percentile
from focus_keeper.pipeline import AnalysisPipeline
from focus_keeper.rules import EventType, RuleConfig, TemporalRuleEngine
from focus_keeper.sources import FramePacket
from focus_keeper.tracker import PrimarySubjectTracker, TrackerConfig

BLANK = np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8)
#: 推論週期（10 FPS）。
INFERENCE_PERIOD_MS = 100
#: 來源影格週期（30 FPS）。
SOURCE_PERIOD_MS = 34
#: 攝影機時間戳的抖動振幅。
JITTER_MS = 3
#: 兩條時間軸的示警時間差上限，由取樣量化推導而來：
#:   條件起點本身可差 (來源週期 + 抖動)；若門檻跨越點落在取樣格之間，
#:   示警會被推到下一個推論取樣點，再多一個推論週期。
#: 這是取樣量化的必然結果，不是核心邏輯不一致——同一份輸入仍然逐位元相同
#: （見 test_same_input_is_bit_identical）。真正的驗收指標是 250 ms 示警上限，
#: 由 test_alert_latency_within_acceptance_on_both_timelines 直接檢查。
TIMELINE_TOLERANCE_MS = INFERENCE_PERIOD_MS + SOURCE_PERIOD_MS + JITTER_MS
MAX_ALERT_LATENCY_MS = 250
#: 攝影機時間戳抖動樣式，振幅不超過 JITTER_MS。
JITTER_PATTERN = (-3, 1, 2, -1, 3, 0, -2)


def scenario(timestamp_ms: int) -> list:
    """情境（時間的函數）：主角在中央 -> 1.5~4.5 秒離場 -> 回來 -> 6 秒後多出第二人。"""
    if 1500 <= timestamp_ms < 4500:
        return []
    if timestamp_ms >= 6000:
        return [face(FRAME_W / 2, FRAME_H / 2), face(300, 300, size=140)]
    return [face(FRAME_W / 2, FRAME_H / 2)]


def empty_scenario(timestamp_ms: int) -> list:
    return []


def make_pipeline(content_fn=scenario, *, inference_fps: float = 10.0) -> AnalysisPipeline:
    return AnalysisPipeline(
        detector=TimelineDetector(content_fn),
        tracker=PrimarySubjectTracker(TrackerConfig(smoothing=0.0)),
        rules=TemporalRuleEngine(RuleConfig()),
        config=PipelineConfig(inference_fps=inference_fps),
    )


def packet(frame_id: int, timestamp_ms: int) -> FramePacket:
    return FramePacket(
        image=BLANK, timestamp_ms=timestamp_ms, frame_id=frame_id,
        capture_monotonic=time.perf_counter(),
    )


def feed_stream(pipeline: AnalysisPipeline, timestamps) -> None:
    for frame_id, ts in enumerate(timestamps):
        pipeline.detector.now_ms = ts
        pipeline.process(packet(frame_id, ts))


def run_stream(pipeline: AnalysisPipeline, timestamps) -> list[tuple]:
    """回傳可比對的事件簽章序列。"""
    signature = []
    for frame_id, ts in enumerate(timestamps):
        pipeline.detector.now_ms = ts
        result = pipeline.process(packet(frame_id, ts))
        if result is None:
            continue
        for event in result.rule_update.started:
            signature.append(("start", event.type.value, event.condition_start_ms, event.triggered_ms))
        for event in result.rule_update.ended:
            signature.append(("end", event.type.value, event.end_ms))
    return signature


class TestThrottling:
    def test_holds_target_rate_on_30fps_source(self) -> None:
        pipeline = make_pipeline(empty_scenario, inference_fps=10.0)
        timestamps = [int(round(i * 1000 / 30)) for i in range(300)]  # 30 FPS，10 秒
        feed_stream(pipeline, timestamps)

        # 10 秒 @ 10 FPS 目標 -> 約 100 幀（容許節拍對齊誤差）。
        assert 95 <= pipeline.processed_frames <= 101
        assert pipeline.skipped_frames == len(timestamps) - pipeline.processed_frames

    def test_awkward_source_rate_does_not_undershoot(self) -> None:
        """25 FPS 來源不會因為對不齊節拍而掉到 8.3 FPS。"""
        pipeline = make_pipeline(empty_scenario, inference_fps=10.0)
        feed_stream(pipeline, [i * 40 for i in range(250)])  # 25 FPS，10 秒
        assert pipeline.processed_frames >= 95

    def test_does_not_burst_after_a_stall(self) -> None:
        """來源停頓後不得補跑累積影格（規格 §7：不得累積影格）。"""
        pipeline = make_pipeline(empty_scenario, inference_fps=10.0)
        pipeline.process(packet(0, 0))
        pipeline.process(packet(1, 5000))          # 停頓 5 秒後的第一格
        before = pipeline.processed_frames
        pipeline.process(packet(2, 5010))          # 緊接著的下一格必須被節流
        assert pipeline.processed_frames == before

    def test_first_frame_always_processed(self) -> None:
        pipeline = make_pipeline(empty_scenario)
        assert pipeline.process(packet(0, 12345)) is not None


class TestSharedCore:
    def test_camera_and_video_timelines_agree(self) -> None:
        """同一核心、同一內容：影片時間軸與攝影機時間軸的事件序列必須一致。

        影片：嚴格 30 FPS 的整齊時間戳。
        攝影機：同樣約 30 FPS，但每格有 +-3 ms 的抖動（真實擷取的樣子）。
        """
        video_ts = [int(round(i * 1000 / 30)) for i in range(240)]
        jitter = [JITTER_PATTERN[i % len(JITTER_PATTERN)] for i in range(240)]
        assert max(abs(j) for j in jitter) <= JITTER_MS
        camera_ts = [max(0, t + j) for t, j in zip(video_ts, jitter)]

        video_events = run_stream(make_pipeline(), video_ts)
        camera_events = run_stream(make_pipeline(), camera_ts)

        # 強主張：事件的種類與先後順序必須完全一致。
        assert [e[0:2] for e in video_events] == [e[0:2] for e in camera_events]
        # 弱主張：取樣點不同，時間戳差異不超過取樣量化的必然上限。
        for v, c in zip(video_events, camera_events):
            for a, b in zip(v[2:], c[2:]):
                assert abs(a - b) <= TIMELINE_TOLERANCE_MS

    @pytest.mark.parametrize("jittered", [False, True])
    def test_alert_latency_within_acceptance_on_both_timelines(self, jittered: bool) -> None:
        """驗收條件：門檻到達後 250 ms 內示警——兩種時間軸都要成立。"""
        base = [int(round(i * 1000 / 30)) for i in range(240)]
        if jittered:
            base = [
                max(0, t + JITTER_PATTERN[i % len(JITTER_PATTERN)])
                for i, t in enumerate(base)
            ]

        pipeline = make_pipeline()
        checked = 0
        for frame_id, ts in enumerate(base):
            pipeline.detector.now_ms = ts
            result = pipeline.process(packet(frame_id, ts))
            if result is None:
                continue
            for event in result.rule_update.started:
                threshold = pipeline.rules.threshold_for(event.type)
                assert threshold is not None
                assert event.alert_latency_ms(threshold) <= MAX_ALERT_LATENCY_MS
                checked += 1
        assert checked >= 3, "情境應至少觸發 MISSING / LEFT / MULTI_PERSON 三個事件"

    def test_same_input_is_bit_identical(self) -> None:
        timestamps = [int(round(i * 1000 / 30)) for i in range(240)]
        assert run_stream(make_pipeline(), timestamps) == run_stream(
            make_pipeline(), timestamps
        )

    def test_expected_event_sequence(self) -> None:
        timestamps = [int(round(i * 1000 / 30)) for i in range(240)]
        events = run_stream(make_pipeline(), timestamps)
        types = [(kind, name) for kind, name, *_ in events]

        assert ("start", "PRIMARY_MISSING") in types
        assert ("start", "PRIMARY_LEFT") in types
        assert ("end", "PRIMARY_MISSING") in types
        assert ("start", "MULTI_PERSON") in types
        # 離場先於復歸，復歸先於多人。
        assert types.index(("start", "PRIMARY_MISSING")) < types.index(("end", "PRIMARY_MISSING"))
        assert types.index(("end", "PRIMARY_MISSING")) < types.index(("start", "MULTI_PERSON"))


class TestTrueAlertLatency:
    """證偽輪補洞（2026-07-29）：用 ground truth 量真實示警延遲。

    兩個先前的盲點：

    1. 原情境把場景轉換點放在 2.0／5.0／7.0 秒，剛好落在 10 FPS 的取樣格上，
       系統性避開了量化最差情況。
    2. 更根本的是，``Event.alert_latency_ms`` 的基準 ``condition_start_ms``
       本身就是「第一個**觀測到**條件成立的取樣點」。兩端被同一個取樣格量化，
       這個指標在構造上幾乎恆為 0——**它量不到取樣延遲**。

    這裡改用測試才知道的真實異常時刻當基準：

        真實延遲 = triggered_ms - (真實異常時刻 + 門檻)
    """

    @staticmethod
    def _leave_at(leave_ms: int):
        def content(timestamp_ms: int) -> list:
            return [] if timestamp_ms >= leave_ms else [face(FRAME_W / 2, FRAME_H / 2)]

        return content

    @staticmethod
    def _true_latency(leave_ms: int) -> int:
        """從主角**真的**離場算起，超出門檻多久才示警。"""
        pipeline = make_pipeline(TestTrueAlertLatency._leave_at(leave_ms))
        threshold = pipeline.rules.threshold_for(EventType.PRIMARY_MISSING)
        assert threshold is not None
        for frame_id in range(240):
            ts = int(round(frame_id * 1000 / 30))
            pipeline.detector.now_ms = ts
            result = pipeline.process(packet(frame_id, ts))
            if result is None:
                continue
            for event in result.rule_update.started:
                if event.type is EventType.PRIMARY_MISSING:
                    return event.triggered_ms - (leave_ms + threshold)
        pytest.fail(f"leave_ms={leave_ms} 未觸發 PRIMARY_MISSING")
        raise AssertionError("unreachable")

    #: 相對於取樣格的偏移，涵蓋整個推論週期。
    OFFSETS = (0, 17, 33, 50, 67, 83, 99)

    @pytest.mark.parametrize("offset", OFFSETS)
    def test_true_latency_within_acceptance(self, offset: int) -> None:
        latency = self._true_latency(2000 + offset)
        assert 0 <= latency <= MAX_ALERT_LATENCY_MS, (
            f"轉換點偏移 {offset} ms 時，真實示警延遲 {latency} ms 不在 [0, 250] 內"
        )

    def test_the_sweep_actually_exercises_quantization(self) -> None:
        """守門：若整個掃描都量到 0，代表這組測試沒測到東西。"""
        latencies = [self._true_latency(2000 + o) for o in self.OFFSETS]
        assert max(latencies) > 0, (
            f"整個掃描都量到 0 ms，未觸及量化最差情況：{latencies}"
        )
        # 上界＝一個推論週期＋一個來源影格週期。
        assert max(latencies) <= INFERENCE_PERIOD_MS + SOURCE_PERIOD_MS, latencies

    def test_reported_metric_understates_true_latency(self) -> None:
        """釘住這個落差，避免日後有人再把 JSONL 的 alert_latency_ms 當成真實延遲。"""
        pipeline = make_pipeline(self._leave_at(2001))
        threshold = pipeline.rules.threshold_for(EventType.PRIMARY_MISSING)
        for frame_id in range(240):
            ts = int(round(frame_id * 1000 / 30))
            pipeline.detector.now_ms = ts
            result = pipeline.process(packet(frame_id, ts))
            if result is None:
                continue
            for event in result.rule_update.started:
                if event.type is EventType.PRIMARY_MISSING:
                    reported = event.alert_latency_ms(threshold)
                    true = event.triggered_ms - (2001 + threshold)
                    assert reported == 0, "觀測基準下確實看不到延遲"
                    assert true > reported, (
                        "真實延遲應大於 JSONL 回報值；若相等代表量化被繞過了"
                    )
                    return
        pytest.fail("未觸發 PRIMARY_MISSING")


class TestEventLogger:
    def test_writes_jsonl_with_required_fields(self, tmp_path) -> None:
        pipeline = make_pipeline()
        log_path = tmp_path / "events.jsonl"
        logger = EventLogger(log_path, pipeline.rules)
        logger.session_start({"detector": "timeline"})

        for frame_id in range(240):
            ts = int(round(frame_id * 1000 / 30))
            pipeline.detector.now_ms = ts
            result = pipeline.process(packet(frame_id, ts))
            if result is not None:
                logger.log_update(result.rule_update)
        logger.log_closed(pipeline.flush(8000))
        logger.close()

        records = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
        assert records[0]["type"] == "session_start"

        starts = [r for r in records if r["type"] == "event_start"]
        assert starts, "應至少記錄一個事件"
        for record in starts:
            # 規格 §5：事件、起訖時間、信心值、人物數
            assert record["event"] in {e.value for e in EventType}
            assert isinstance(record["condition_start_ms"], int)
            assert isinstance(record["triggered_ms"], int)
            assert "max_person_count" in record
            assert "trigger_confidence" in record
            assert record["alert_latency_ms"] <= 250

        ends = [r for r in records if r["type"] == "event_end"]
        assert ends and all(r["end_ms"] is not None for r in ends)

    def test_disabled_logger_is_noop(self) -> None:
        rules = TemporalRuleEngine(RuleConfig())
        logger = EventLogger(None, rules)
        logger.session_start({"a": 1})
        logger.close()
        assert logger.path is None


class TestLatencyStats:
    def test_percentile_uses_nearest_rank(self) -> None:
        values = [float(v) for v in range(1, 101)]
        assert percentile(values, 50) == 50.0
        assert percentile(values, 95) == 95.0
        assert percentile([], 95) == 0.0
        assert percentile([7.0], 95) == 7.0

    def test_summary_shape(self) -> None:
        pipeline = make_pipeline()
        stats = LatencyStats()
        for frame_id in range(30):
            ts = frame_id * 100
            pipeline.detector.now_ms = ts
            result = pipeline.process(packet(frame_id, ts))
            if result is not None:
                stats.add(result)
        summary = stats.summary()
        assert summary["processed_frames"] == pipeline.processed_frames
        assert set(summary["pipeline_ms"]) == {"mean", "p50", "p95", "max"}
        assert summary["effective_inference_fps"] > 0


class TestPresenceThreshold:
    """在場人數與主角資格必須用不同的尺（2026-07-29 使用者裁決）。

    產品語意＝「畫面必須是單人主構圖」，所以只要畫面裡還有第二個人就該示警，
    即使他站得較遠、臉較小、不足以當主角。共用同一把尺會造成漏報。
    """

    # face() 的寬高比是 0.8，所以面積比 = 0.8 * size^2 / (1280*720)。
    #   size=160 -> 0.0222  通過主角門檻 0.015
    #   size=100 -> 0.0087  過不了主角門檻，但過得了在場門檻 0.004  <- 要測的就是這個
    PRIMARY_SIZE = 160
    FAR_SIZE = 100

    @staticmethod
    def _with_far_second(timestamp_ms: int) -> list:
        return [
            face(FRAME_W / 2, FRAME_H / 2, size=TestPresenceThreshold.PRIMARY_SIZE),
            face(220, 240, size=TestPresenceThreshold.FAR_SIZE),
        ]

    def test_fixture_sits_between_the_two_thresholds(self) -> None:
        """守門：夾具若不落在兩個門檻之間，下面的測試就什麼也沒測到。"""
        cfg = PipelineConfig()
        far = face(220, 240, size=self.FAR_SIZE)
        ratio = far.area_ratio(FRAME_W, FRAME_H)
        assert cfg.effective_presence_area_ratio < ratio < cfg.min_face_area_ratio, (
            f"遠處第二人面積比 {ratio:.5f} 未落在 "
            f"({cfg.effective_presence_area_ratio}, {cfg.min_face_area_ratio}) 之間"
        )

    def _run(self, cfg: PipelineConfig, frames: int = 20):
        pipeline = AnalysisPipeline(
            detector=TimelineDetector(self._with_far_second),
            tracker=PrimarySubjectTracker(TrackerConfig(smoothing=0.0)),
            rules=TemporalRuleEngine(RuleConfig()),
            config=cfg,
        )
        last = None
        fired = []
        for frame_id in range(frames):
            ts = frame_id * 100
            pipeline.detector.now_ms = ts
            result = pipeline.process(packet(frame_id, ts))
            if result is None:
                continue
            last = result
            fired.extend(e.type for e in result.rule_update.started)
        return last, fired

    def test_far_second_person_counts_and_alerts(self) -> None:
        last, fired = self._run(PipelineConfig())
        assert last is not None
        assert len(last.detections) == 1, "遠處第二人不該有主角資格"
        assert len(last.presence) == 2, "遠處第二人必須被算進在場人數"
        assert last.rule_update.person_count == 2
        assert EventType.MULTI_PERSON in fired

    def test_disabling_split_restores_old_behaviour(self) -> None:
        """兩個門檻設 None 時退回舊行為（共用主角門檻）——遠處第二人被忽略。"""
        cfg = PipelineConfig(
            presence_min_confidence=None, presence_min_face_area_ratio=None
        )
        last, fired = self._run(cfg)
        assert last is not None
        assert len(last.presence) == 1
        assert last.rule_update.person_count == 1
        assert EventType.MULTI_PERSON not in fired

    def test_overlay_draws_presence_not_just_primary_candidates(self) -> None:
        """人數說 2 人，畫面就得看得到 2 個框，不能自相矛盾。"""
        last, _ = self._run(PipelineConfig())
        assert last is not None
        assert last.rule_update.person_count == len(last.presence)

    def test_presence_threshold_cannot_be_stricter_than_primary(self) -> None:
        with pytest.raises(ValueError):
            PipelineConfig.from_dict({"presence_min_confidence": 0.95})
        with pytest.raises(ValueError):
            PipelineConfig.from_dict({"presence_min_face_area_ratio": 0.5})


class TestPipelineConfig:
    def test_rejects_bad_values(self) -> None:
        with pytest.raises(ValueError):
            PipelineConfig.from_dict({"inference_fps": 0})
        with pytest.raises(ValueError):
            PipelineConfig.from_dict({"min_confidence": 1.5})
        with pytest.raises(ConfigError):
            PipelineConfig.from_dict({"unknown": 1})

    def test_spec_defaults(self) -> None:
        cfg = PipelineConfig()
        assert cfg.inference_fps == 10.0
        assert cfg.min_confidence == 0.70          # update 版由 0.65 提高
        assert cfg.min_face_area_ratio == 0.015
