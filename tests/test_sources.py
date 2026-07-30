"""影像來源與端到端效能（規格 §3、§6）。

攝影機測試在 CI／無硬體環境不可靠，因此這裡以真實 MP4 走完整條路徑
（VideoSource -> YuNet -> tracker -> rules），量出本機的有效 FPS 與 P95 延遲。
"""

from __future__ import annotations

import numpy as np
import pytest
from conftest import YUNET_MODEL, requires_yunet

import cv2

from focus_keeper.config import PipelineConfig
from focus_keeper.metrics import LatencyStats, percentile
from focus_keeper.pipeline import AnalysisPipeline
from focus_keeper.detectors import build_detector
from focus_keeper.rules import RuleConfig, TemporalRuleEngine
from focus_keeper.sources import CameraSource, SourceError, VideoSource, open_source
from focus_keeper.tracker import PrimarySubjectTracker, TrackerConfig

#: 規格 §6 驗收條件。
MIN_EFFECTIVE_FPS = 10.0
MAX_P95_LATENCY_MS = 200.0


def write_video(path, *, frames: int = 90, fps: int = 30, size=(640, 360)) -> str:
    """產生一段可重複讀取的測試影片（內容為移動方塊，非真人）。"""
    width, height = size
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    assert writer.isOpened(), "無法建立 VideoWriter（缺 mp4v 編碼器）"
    rng = np.random.default_rng(20260729)
    for i in range(frames):
        frame = rng.integers(0, 60, size=(height, width, 3), dtype=np.uint8)
        x = 40 + (i * 5) % (width - 120)
        cv2.rectangle(frame, (x, 120), (x + 80, 220), (200, 200, 200), -1)
        writer.write(frame)
    writer.release()
    return str(path)


@pytest.fixture(scope="module")
def sample_video(tmp_path_factory):
    return write_video(tmp_path_factory.mktemp("media") / "sample.mp4")


class TestVideoSource:
    def test_reads_all_frames_with_monotonic_timestamps(self, sample_video) -> None:
        with VideoSource(sample_video) as source:
            packets = list(source)

        assert len(packets) >= 85          # 編碼器可能省略末端幾格
        assert [p.frame_id for p in packets] == list(range(len(packets)))
        timestamps = [p.timestamp_ms for p in packets]
        assert timestamps[0] == 0
        assert all(b > a for a, b in zip(timestamps, timestamps[1:])), "時間戳必須嚴格遞增"

    def test_timestamps_are_reproducible(self, sample_video) -> None:
        with VideoSource(sample_video) as a, VideoSource(sample_video) as b:
            first = [(p.frame_id, p.timestamp_ms) for p in a]
            second = [(p.frame_id, p.timestamp_ms) for p in b]
        assert first == second

    def test_timestamps_match_declared_fps(self, sample_video) -> None:
        with VideoSource(sample_video) as source:
            packets = list(source)
        deltas = [b.timestamp_ms - a.timestamp_ms for a, b in zip(packets, packets[1:])]
        assert all(30 <= d <= 37 for d in deltas), f"30 FPS 應約 33 ms/格，實得 {set(deltas)}"

    def test_missing_file_raises(self, tmp_path) -> None:
        with pytest.raises(SourceError):
            VideoSource(tmp_path / "does_not_exist.mp4")

    def test_describe_reports_geometry(self, sample_video) -> None:
        with VideoSource(sample_video) as source:
            info = source.describe()
        assert info["kind"] == "video"
        assert info["frame_width"] == 640 and info["frame_height"] == 360
        assert info["fps"] == pytest.approx(30.0, abs=0.5)


class TestOpenSource:
    def test_digit_selects_camera(self, monkeypatch) -> None:
        created = {}

        def fake_camera(index, **kwargs):
            created["index"] = index
            created["kwargs"] = kwargs
            return "camera-stub"

        monkeypatch.setattr("focus_keeper.sources.CameraSource", fake_camera)
        from focus_keeper.config import CameraConfig, SourcesConfig

        cfg = SourcesConfig(camera=CameraConfig(fps=15))
        assert open_source("2", cfg) == "camera-stub"
        assert created["index"] == 2
        assert created["kwargs"]["fps"] == 15

    def test_path_selects_video(self, sample_video) -> None:
        source = open_source(sample_video)
        try:
            assert isinstance(source, VideoSource)
        finally:
            source.close()

    def test_bad_camera_backend_rejected(self) -> None:
        with pytest.raises(SourceError):
            CameraSource(index=0, backend="not-a-backend")


@requires_yunet
class TestEndToEndPerformance:
    """規格 §6：有效推論 >= 10 FPS、P95 延遲 < 200 ms。

    注意：測試影片不含真人臉，因此量到的是**推論成本**（YuNet 前向傳遞
    與整條管線的固定開銷），不是偵測準確度。準確度需以真實素材另行驗收。
    """

    def test_meets_latency_and_fps_targets(self, sample_video) -> None:
        detector = build_detector(
            "yunet", {"yunet": {"model_path": str(YUNET_MODEL), "detect_width": 320}}
        )
        pipeline = AnalysisPipeline(
            detector=detector,
            tracker=PrimarySubjectTracker(TrackerConfig()),
            rules=TemporalRuleEngine(RuleConfig()),
            config=PipelineConfig(inference_fps=10.0),
        )
        stats = LatencyStats()
        try:
            with VideoSource(sample_video) as source:
                for packet in source:
                    result = pipeline.process(packet)
                    if result is not None:
                        stats.add(result)
        finally:
            detector.close()

        assert stats.count > 0
        p95 = percentile(stats.pipeline_ms, 95)
        # 單幀處理必須足以支撐 10 FPS（<= 100 ms），且 P95 遠低於 200 ms 上限。
        assert p95 < MAX_P95_LATENCY_MS, f"P95 pipeline 延遲 {p95:.1f} ms 超標"
        assert 1000.0 / p95 >= MIN_EFFECTIVE_FPS, (
            f"P95 延遲 {p95:.1f} ms 無法支撐 {MIN_EFFECTIVE_FPS} FPS"
        )

    def test_full_video_run_via_cli(self, sample_video, tmp_path) -> None:
        """--no-display 的無頭執行必須成功並寫出 JSONL。"""
        from focus_keeper.cli import main

        log_path = tmp_path / "events.jsonl"
        exit_code = main(
            [
                "--source", sample_video,
                "--no-display",
                "--quiet",
                "--log", str(log_path),
                "--report", str(tmp_path / "report.json"),
            ]
        )
        assert exit_code == 0
        assert log_path.is_file()

        import json

        records = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
        assert records[0]["type"] == "session_start"
        assert records[-1]["type"] == "session_end"
        assert records[0]["detector"]["detector"] == "yunet"
        assert records[0]["configured_detector"] == "yunet"
        assert records[-1]["metrics"]["processed_frames"] > 0
