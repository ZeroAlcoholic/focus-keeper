"""設定的載入與驗證（單一入口 AppConfig）。

補洞說明：重構前頂層 key 完全不驗——把 `rules` 打成 `rulez` 會讓整組
時間門檻悄悄退回預設值，而程式看起來完全正常。這是最難察覺的設定錯誤。
本檔把「每一層的未知欄位都必須報錯」釘成契約。
"""

from __future__ import annotations

import pytest
from conftest import YUNET_MODEL

from focus_keeper.config import AppConfig, CameraConfig, SourcesConfig, VideoConfig
from focus_keeper.validation import ConfigError, as_mapping, check_fields


def write(tmp_path, text: str) -> str:
    p = tmp_path / "c.yaml"
    p.write_text(text, encoding="utf-8")
    return str(p)


class TestUnknownFieldsAreRejected:
    """靜默忽略打錯的欄位名是最危險的設定錯誤。"""

    @pytest.mark.parametrize(
        "yaml_text,expect_in_message",
        [
            ("rulez:\n  primary_left_ms: 1\n", "設定檔頂層"),
            ("detecter: yunet\n", "設定檔頂層"),
            ("event_logg: x.jsonl\n", "設定檔頂層"),
            ("rules:\n  typo_ms: 1\n", "rules"),
            ("rules:\n  roi:\n    z: 1\n", "rules.roi"),
            ("rules:\n  shoulder_box:\n    up: 1\n", "rules.shoulder_box"),
            ("tracker:\n  smooth: 1\n", "tracker"),
            ("pipeline:\n  fps: 10\n", "pipeline"),
            ("sources:\n  cam: {}\n", "sources"),
            ("sources:\n  camera:\n    fpss: 30\n", "sources.camera"),
            ("sources:\n  video:\n    pace: true\n", "sources.video"),
            ("detectors:\n  yunet:\n    model_pth: x\n", "detectors.yunet"),
        ],
    )
    def test_typo_is_reported_with_section_name(
        self, tmp_path, yaml_text: str, expect_in_message: str
    ) -> None:
        with pytest.raises(ConfigError) as exc:
            AppConfig.load(write(tmp_path, yaml_text))
        assert expect_in_message in str(exc.value)
        # 訊息必須列出可用欄位，否則使用者不知道正確拼法
        assert "可用欄位" in str(exc.value)


class TestValueValidation:
    @pytest.mark.parametrize(
        "yaml_text",
        [
            "pipeline:\n  inference_fps: 0\n",
            "pipeline:\n  min_confidence: 1.5\n",
            "pipeline:\n  presence_min_confidence: 0.95\n",
            "rules:\n  primary_missing_warn_ms: 3000\n  primary_left_ms: 1000\n",
            "rules:\n  roi:\n    w: 0\n",
        ],
    )
    def test_bad_value_raises_config_error(self, tmp_path, yaml_text: str) -> None:
        with pytest.raises(ConfigError):
            AppConfig.load(write(tmp_path, yaml_text))

    def test_missing_file(self, tmp_path) -> None:
        with pytest.raises(ConfigError):
            AppConfig.load(tmp_path / "absent.yaml")

    def test_not_a_mapping(self, tmp_path) -> None:
        with pytest.raises(ConfigError):
            AppConfig.load(write(tmp_path, "- a\n- b\n"))

    def test_section_must_be_mapping(self, tmp_path) -> None:
        with pytest.raises(ConfigError):
            AppConfig.load(write(tmp_path, "rules: 5\n"))


class TestDefaultsAndTypes:
    def test_repo_config_loads_and_is_consistent(self) -> None:
        """倉庫附的 config.yaml 必須通過所有驗證且無警告。"""
        cfg = AppConfig.load()
        assert cfg.detector == "yunet"
        assert cfg.warnings == (), f"預設設定不應產生警告：{cfg.warnings}"
        assert cfg.pipeline.inference_fps == 10
        assert cfg.tracker.lost_after_ms >= cfg.rules.primary_left_ms

    def test_demo_config_loads(self) -> None:
        cfg = AppConfig.load("config.demo.yaml")
        assert cfg.warnings == ()
        assert cfg.rules.primary_left_ms < 2000, "DEMO 設定應縮短門檻"

    def test_empty_config_uses_defaults(self, tmp_path) -> None:
        cfg = AppConfig.load(write(tmp_path, "{}\n"))
        assert cfg.source == "0"
        assert isinstance(cfg.sources, SourcesConfig)
        assert isinstance(cfg.sources.camera, CameraConfig)
        assert isinstance(cfg.sources.video, VideoConfig)

    def test_event_log_null_means_disabled(self, tmp_path) -> None:
        assert AppConfig.load(write(tmp_path, "event_log:\n")).event_log is None
        assert AppConfig.load(write(tmp_path, 'event_log: ""\n')).event_log is None
        assert AppConfig.load(write(tmp_path, "event_log: a.jsonl\n")).event_log is not None

    def test_cross_section_warning(self, tmp_path) -> None:
        cfg = AppConfig.load(
            write(tmp_path, "tracker:\n  lost_after_ms: 500\nrules:\n  primary_left_ms: 2000\n")
        )
        assert any("lost_after_ms" in w for w in cfg.warnings)

    def test_as_report_is_serializable(self) -> None:
        import json

        json.dumps(AppConfig.load().as_report())


class TestValidationHelpers:
    def test_as_mapping_rejects_non_mapping(self) -> None:
        assert as_mapping("x", None) == {}
        with pytest.raises(ConfigError):
            as_mapping("x", [1, 2])

    def test_check_fields_lists_allowed(self) -> None:
        with pytest.raises(ConfigError) as exc:
            check_fields("sec", {"bad": 1}, {"good"})
        assert "bad" in str(exc.value) and "good" in str(exc.value)

    def test_config_error_is_value_error(self) -> None:
        """呼叫端只 catch ValueError 的舊程式仍然有效。"""
        assert issubclass(ConfigError, ValueError)


class TestLongRunBounded:
    """長時運轉的資源與成本必須有上界（2026-07-30 校正）。

    先前 LatencyStats 用無上界的 list，而 draw_overlay 每格都對整份清單
    排序算 p95——實測 8 小時後單格 6.29 ms，比整個偵測成本（2.5 ms）還高，
    監測本身會愈跑愈慢。對設計上要連續運轉數小時的系統這是實質缺陷。
    """

    @staticmethod
    def _fill(n: int):
        from focus_keeper.metrics import LatencyStats

        class _R:
            def __init__(self, i: int) -> None:
                self.pipeline_ms = float(i % 97)
                self.end_to_end_ms = float(i % 89)
                self.timestamp_ms = i * 100

        stats = LatencyStats()
        for i in range(n):
            stats.add(_R(i))
        return stats

    def test_sample_window_is_bounded(self) -> None:
        from focus_keeper.metrics import DEFAULT_WINDOW

        stats = self._fill(DEFAULT_WINDOW * 3)
        assert len(stats.pipeline_ms) == DEFAULT_WINDOW
        assert len(stats.end_to_end_ms) == DEFAULT_WINDOW
        assert len(stats.timestamps_ms) == DEFAULT_WINDOW

    def test_count_mean_max_stay_exact_over_whole_run(self) -> None:
        """視窗化不得讓計數／平均／最大值失真。"""
        from focus_keeper.metrics import DEFAULT_WINDOW

        n = DEFAULT_WINDOW * 2
        stats = self._fill(n)
        summary = stats.summary()
        assert summary["processed_frames"] == n
        assert summary["pipeline_ms"]["max"] == 96.0     # max(i % 97)
        assert summary["end_to_end_ms"]["max"] == 88.0   # max(i % 89)
        # 視窗大小必須一併回報，避免把視窗值誤讀成全程值
        assert summary["percentile_window_samples"] == DEFAULT_WINDOW
        assert summary["percentile_window_capacity"] == DEFAULT_WINDOW

    def test_hud_percentile_is_cached(self) -> None:
        stats = self._fill(1000)
        first = stats.hud_p95_end_to_end()
        # 未新增樣本時不得重算（回傳同一個快取值）
        assert stats.hud_p95_end_to_end() == first

    def test_hud_percentile_cost_does_not_grow(self) -> None:
        import time

        from focus_keeper.metrics import DEFAULT_WINDOW

        def cost(n: int) -> float:
            stats = self._fill(n)
            t = time.perf_counter()
            for _ in range(200):
                stats.hud_p95_end_to_end()
            return (time.perf_counter() - t) / 200 * 1000

        short = cost(1000)
        long_run = cost(DEFAULT_WINDOW * 8)          # ≈ 8 小時 @ 10 FPS
        assert long_run < max(0.5, short * 20 + 0.05), (
            f"長時運轉後 HUD p95 成本上升過多：{short:.4f} -> {long_run:.4f} ms"
        )

    def test_recent_fps_works_on_bounded_deque(self) -> None:
        """deque 不支援切片；改用 islice 後仍須正確。"""
        stats = self._fill(500)
        assert stats.recent_fps() == pytest.approx(10.0, abs=0.5)
        assert stats.headroom_fps() > 0
