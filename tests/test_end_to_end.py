"""真實模型的端到端驗收（規格 §6）。

以 `scripts/make_sample_video.py` 同款情境產生 MP4，走完整條路徑：

    VideoSource -> YuNet(ONNX) -> PrimarySubjectTracker -> TemporalRuleEngine

驗的是「整條管線在真實推論下會不會產生正確的事件序列與時序」。

**界線**：影片用的是合成臉，因此這裡驗的是**流程正確性與推論成本**，
不是偵測準確度。準確度必須以真實素材另行驗收（見 README「已知限制」）。
"""

from __future__ import annotations

import numpy as np
import pytest
from conftest import draw_synthetic_face, requires_yunet

import cv2

from focus_keeper.config import AppConfig, PipelineConfig, load_config
from focus_keeper.metrics import LatencyStats
from focus_keeper.pipeline import AnalysisPipeline
from focus_keeper.detectors import build_detector
from focus_keeper.rules import EventType, RuleConfig, TemporalRuleEngine
from focus_keeper.sources import VideoSource
from focus_keeper.tracker import PrimarySubjectTracker, TrackerConfig

WIDTH, HEIGHT, FPS = 960, 540, 30
CENTER = (WIDTH // 2, 270)
SECOND = (170, 285)

#: 情境時間表（秒）。
T_LEAVE, T_RETURN, T_SECOND, T_END = 2.0, 5.0, 7.0, 10.0
MAX_ALERT_LATENCY_MS = 250
#: 事件時間的容許誤差＝一個推論週期＋一個來源影格週期。
TIMING_TOLERANCE_MS = 100 + 34


def build_scenario_video(path) -> str:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (WIDTH, HEIGHT))
    assert writer.isOpened(), "無法建立 VideoWriter（缺 mp4v 編碼器）"
    total = int(T_END * FPS)
    for i in range(total):
        t = i / FPS
        frame = np.full((HEIGHT, WIDTH, 3), 210, np.uint8)
        if t < T_LEAVE or t >= T_RETURN:
            draw_synthetic_face(frame, *CENTER, scale=1.0)
        if t >= T_SECOND:
            draw_synthetic_face(frame, *SECOND, scale=1.0)
        writer.write(cv2.GaussianBlur(frame, (5, 5), 0))
    writer.release()
    return str(path)


@pytest.fixture(scope="module")
def scenario_video(tmp_path_factory):
    return build_scenario_video(tmp_path_factory.mktemp("e2e") / "scenario.mp4")


@pytest.fixture(scope="module")
def run_result(scenario_video):
    """以 config.yaml 的正式參數跑完整段影片，回傳事件與效能統計。"""
    cfg = load_config()
    detector = build_detector("yunet", cfg["detectors"])
    pipeline = AnalysisPipeline(
        detector=detector,
        tracker=PrimarySubjectTracker(TrackerConfig.from_dict(cfg.get("tracker"))),
        rules=TemporalRuleEngine(RuleConfig.from_dict(cfg.get("rules"))),
        config=PipelineConfig.from_dict(cfg.get("pipeline")),
    )
    stats = LatencyStats()
    started, ended, frames = [], [], []
    try:
        with VideoSource(scenario_video) as source:
            for packet in source:
                result = pipeline.process(packet)
                if result is None:
                    continue
                stats.add(result)
                frames.append(result)
                started.extend(result.rule_update.started)
                ended.extend(result.rule_update.ended)
        ended.extend(pipeline.flush(frames[-1].timestamp_ms if frames else 0))
    finally:
        detector.close()
    return {
        "pipeline": pipeline, "stats": stats,
        "started": started, "ended": ended, "frames": frames,
    }


def first(events, event_type: EventType):
    return next((e for e in events if e.type is event_type), None)


@requires_yunet
class TestDetectionWorks:
    def test_primary_is_found_and_kept(self, run_result) -> None:
        frames = run_result["frames"]
        assert frames, "應至少處理一格"

        early = [f for f in frames if f.timestamp_ms < T_LEAVE * 1000]
        visible = [f for f in early if f.primary is not None and f.primary.visible]
        assert len(visible) >= len(early) - 1, "主角在場期間應幾乎每次判定都被追蹤到"

        # 主角應該就是中央那張臉。
        cx = [f.primary.bbox.cx for f in visible]
        assert min(cx) > CENTER[0] - 80 and max(cx) < CENTER[0] + 80

    def test_second_person_is_detected(self, run_result) -> None:
        late = [f for f in run_result["frames"] if f.timestamp_ms >= (T_SECOND + 0.5) * 1000]
        assert late
        counts = [f.rule_update.person_count for f in late]
        assert max(counts) >= 2, f"第二人未被偵測到，人數序列={sorted(set(counts))}"

    def test_primary_identity_survives_second_person(self, run_result) -> None:
        """規格 §4：出現第二人時不得改認主角。"""
        after_return = [
            f for f in run_result["frames"]
            if f.timestamp_ms >= T_RETURN * 1000 and f.primary is not None
        ]
        assert after_return
        track_ids = {f.primary.track_id for f in after_return}
        assert len(track_ids) == 1, f"主角身分在第二人出現後被換掉：{track_ids}"


@requires_yunet
class TestEventSequence:
    def test_missing_then_left(self, run_result) -> None:
        missing = first(run_result["started"], EventType.PRIMARY_MISSING)
        left = first(run_result["started"], EventType.PRIMARY_LEFT)
        assert missing is not None, "主角離場 3 秒卻沒有 PRIMARY_MISSING"
        assert left is not None, "主角離場 3 秒卻沒有 PRIMARY_LEFT"

        # 條件應在主角離場（2.0 s）當下起算。
        assert abs(missing.condition_start_ms - T_LEAVE * 1000) <= TIMING_TOLERANCE_MS
        # 門檻：missing 1000 ms -> 約 3.0 s；left 2000 ms -> 約 4.0 s。
        assert abs(missing.triggered_ms - 3000) <= TIMING_TOLERANCE_MS
        assert abs(left.triggered_ms - 4000) <= TIMING_TOLERANCE_MS

    def test_multi_person_fires_after_hold(self, run_result) -> None:
        event = first(run_result["started"], EventType.MULTI_PERSON)
        assert event is not None, "第二人持續 3 秒卻沒有 MULTI_PERSON"
        assert abs(event.condition_start_ms - T_SECOND * 1000) <= TIMING_TOLERANCE_MS
        assert abs(event.triggered_ms - (T_SECOND * 1000 + 500)) <= TIMING_TOLERANCE_MS
        assert event.max_person_count >= 2

    def test_missing_closes_on_return(self, run_result) -> None:
        closed = [e for e in run_result["ended"] if e.type is EventType.PRIMARY_MISSING]
        assert closed, "主角回來後 PRIMARY_MISSING 應被關閉"
        assert abs(closed[0].end_ms - T_RETURN * 1000) <= TIMING_TOLERANCE_MS + 200

    def test_no_spurious_events_while_stable(self, run_result) -> None:
        """主角穩定在中央的期間不得有任何事件。"""
        spurious = [
            e for e in run_result["started"]
            if e.condition_start_ms < T_LEAVE * 1000
        ]
        assert spurious == [], f"穩定期出現誤報：{[e.type.value for e in spurious]}"

    def test_alert_latency_within_acceptance(self, run_result) -> None:
        rules = run_result["pipeline"].rules
        for event in run_result["started"]:
            threshold = rules.threshold_for(event.type)
            assert threshold is not None
            assert event.alert_latency_ms(threshold) <= MAX_ALERT_LATENCY_MS, (
                f"{event.type.value} 超出門檻後 {event.alert_latency_ms(threshold)} ms 才示警"
            )


@requires_yunet
class TestPerformance:
    """規格 §6：有效推論 >= 10 FPS、P95 延遲 < 200 ms（本機 CPU）。"""

    def test_meets_acceptance_targets(self, run_result) -> None:
        stats = run_result["stats"]
        summary = stats.summary()
        p95 = summary["pipeline_ms"]["p95"]

        assert p95 < 200.0, f"P95 pipeline 延遲 {p95} ms 超出 200 ms 上限"
        assert 1000.0 / p95 >= 10.0, f"P95 延遲 {p95} ms 無法支撐 10 FPS"
        print(
            "\n[e2e] processed={processed_frames} "
            "mean={pipeline_ms[mean]}ms p95={pipeline_ms[p95]}ms "
            "eff_fps={effective_inference_fps}".format(**summary)
        )

    def test_throttle_holds_target_rate(self, run_result) -> None:
        pipeline = run_result["pipeline"]
        # 10 秒 @ 10 FPS -> 約 100 次判定。
        assert 90 <= pipeline.processed_frames <= 105
        assert pipeline.skipped_frames > 0, "30 FPS 來源應有影格被節流跳過"
