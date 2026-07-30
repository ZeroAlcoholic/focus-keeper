"""CLI 錯誤處理契約。

補洞說明：2026-07-29 的乾淨拷貝驗證發現，模型／來源錯誤會乾淨收場（`[error]` + exit 2），
但**設定錯誤會噴 Python traceback + exit 1**。使用者第一次打錯設定路徑就會撞到。
本檔把「所有可預期的啟動失敗都必須 exit 2 且不得有 traceback」釘成契約。
"""

from __future__ import annotations

import pytest
from conftest import YUNET_MODEL

from focus_keeper.cli import main

BAD_EXIT = 2


def write(tmp_path, name: str, text: str) -> str:
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return str(path)


def run_cli(capsys, *argv) -> tuple[int, str]:
    code = main(list(argv))
    captured = capsys.readouterr()
    return code, captured.out + captured.err


class TestCleanFailures:
    def test_missing_config_file(self, tmp_path, capsys) -> None:
        code, out = run_cli(
            capsys, "--config", str(tmp_path / "nope.yaml"), "--source", "0", "--no-display"
        )
        assert code == BAD_EXIT
        assert "找不到設定檔" in out
        assert "Traceback" not in out

    def test_config_is_not_a_mapping(self, tmp_path, capsys) -> None:
        cfg = write(tmp_path, "list.yaml", "- a\n- b\n")
        code, out = run_cli(capsys, "--config", cfg, "--source", "0", "--no-display")
        assert code == BAD_EXIT
        assert "Traceback" not in out

    def test_partial_config_merges_with_defaults(self, tmp_path) -> None:
        """回歸（2026-07-30）：只寫片段的設定必須可用。

        先前 `detectors` 是唯一沒有程式內建預設的區塊，導致部分設定會以
        「缺少 detectors.yunet 區塊」失敗。當時我據此在 README 與 --help
        寫下「--config 是整份取代、不與預設值合併」——那是**錯的**，
        只是那個缺陷造成的假象。實際上每一層都與內建預設合併。
        """
        from focus_keeper.config import AppConfig

        cfg = AppConfig.load(
            write(tmp_path, "partial.yaml", "pipeline:\n  inference_fps: 5\n")
        )
        assert cfg.pipeline.inference_fps == 5               # 指定值
        assert cfg.pipeline.min_confidence == 0.70           # 同區未指定 -> 預設
        assert cfg.rules.primary_left_ms == 2000             # 整區未指定 -> 預設
        assert set(cfg.detectors) == {"yunet", "mediapipe"}  # 整區未指定 -> 內建預設
        assert cfg.source == "0"

    def test_partial_section_keeps_sibling_defaults(self, tmp_path) -> None:
        from focus_keeper.config import AppConfig

        cfg = AppConfig.load(write(tmp_path, "p2.yaml", "rules:\n  primary_left_ms: 3000\n"))
        assert cfg.rules.primary_left_ms == 3000
        assert cfg.rules.primary_missing_warn_ms == 1000

    def test_unknown_config_field(self, tmp_path, capsys) -> None:
        cfg = write(
            tmp_path,
            "typo.yaml",
            "detectors:\n  yunet:\n    model_path: x.onnx\n"
            "rules:\n  typo_ms: 100\n",
        )
        code, out = run_cli(capsys, "--config", cfg, "--source", "0", "--no-display")
        assert code == BAD_EXIT
        assert "Traceback" not in out

    def test_invalid_threshold_value(self, tmp_path, capsys) -> None:
        cfg = write(
            tmp_path,
            "bad_fps.yaml",
            "detectors:\n  yunet:\n    model_path: x.onnx\n"
            "pipeline:\n  inference_fps: 0\n",
        )
        code, out = run_cli(capsys, "--config", cfg, "--source", "0", "--no-display")
        assert code == BAD_EXIT
        assert "Traceback" not in out

    def test_missing_model_file(self, tmp_path, capsys) -> None:
        cfg = write(
            tmp_path,
            "nomodel.yaml",
            f"detectors:\n  yunet:\n    model_path: {tmp_path.as_posix()}/absent.onnx\n",
        )
        code, out = run_cli(capsys, "--config", cfg, "--source", "0", "--no-display")
        assert code == BAD_EXIT
        assert "fetch_model.py" in out
        assert "Traceback" not in out

    @pytest.mark.skipif(not YUNET_MODEL.is_file(), reason="需要 YuNet 權重")
    def test_sha256_mismatch(self, tmp_path, capsys) -> None:
        cfg = write(
            tmp_path,
            "badsha.yaml",
            f"detectors:\n  yunet:\n    model_path: {YUNET_MODEL.as_posix()}\n"
            f"    sha256: \"{'0' * 64}\"\n",
        )
        code, out = run_cli(capsys, "--config", cfg, "--source", "0", "--no-display")
        assert code == BAD_EXIT
        assert "SHA-256" in out
        assert "Traceback" not in out

    @pytest.mark.skipif(not YUNET_MODEL.is_file(), reason="需要 YuNet 權重")
    def test_missing_video_source(self, tmp_path, capsys) -> None:
        cfg = write(
            tmp_path,
            "ok.yaml",
            f"detectors:\n  yunet:\n    model_path: {YUNET_MODEL.as_posix()}\n",
        )
        code, out = run_cli(
            capsys, "--config", cfg, "--source", str(tmp_path / "absent.mp4"), "--no-display"
        )
        assert code == BAD_EXIT
        assert "找不到影片檔" in out
        assert "Traceback" not in out


class TestSourceConfigValidation:
    """回歸（code review 第 5 項）：sources 區塊先前不驗欄位名，
    打錯字會變成建構子 TypeError（traceback + exit 1），違反上面的契約。"""

    @pytest.mark.parametrize(
        "section,body",
        [
            ("camera", "sources:\n  camera:\n    fpss: 30\n"),
            ("video", "sources:\n  video:\n    pace_real: true\n"),
            ("camera", "sources:\n  camera: 30\n"),
        ],
    )
    def test_unknown_or_malformed_source_field(self, tmp_path, capsys, section, body) -> None:
        source = "0" if section == "camera" else str(tmp_path / "x.mp4")
        cfg = write(
            tmp_path,
            f"src_{section}_{abs(hash(body)) % 9999}.yaml",
            f"detectors:\n  yunet:\n    model_path: {YUNET_MODEL.as_posix()}\n" + body,
        )
        code, out = run_cli(capsys, "--config", cfg, "--source", source, "--no-display")
        assert code == BAD_EXIT
        assert "Traceback" not in out

    @pytest.mark.skipif(not YUNET_MODEL.is_file(), reason="需要 YuNet 權重")
    def test_null_event_log_disables_logging_cleanly(self, tmp_path, capsys) -> None:
        """`event_log:` 留空是使用者關閉記錄的自然寫法，不得變成 Path(None)。"""
        cfg = write(
            tmp_path,
            "nolog.yaml",
            f"detectors:\n  yunet:\n    model_path: {YUNET_MODEL.as_posix()}\n"
            "event_log:\n",
        )
        code, out = run_cli(
            capsys, "--config", cfg, "--source", str(tmp_path / "absent.mp4"), "--no-display"
        )
        # 影片不存在所以仍是 exit 2，重點是不能因為 event_log 為 None 就 traceback
        assert code == BAD_EXIT
        assert "Traceback" not in out
        assert "找不到影片檔" in out


class TestWarnings:
    @pytest.mark.skipif(not YUNET_MODEL.is_file(), reason="需要 YuNet 權重")
    def test_warns_when_lost_after_is_shorter_than_left(self, tmp_path, capsys) -> None:
        """tracker.lost_after_ms < rules.primary_left_ms 會在示警前換主角。"""
        cfg = write(
            tmp_path,
            "warn.yaml",
            f"detectors:\n  yunet:\n    model_path: {YUNET_MODEL.as_posix()}\n"
            "tracker:\n  lost_after_ms: 500\n"
            "rules:\n  primary_left_ms: 2000\n",
        )
        code, out = run_cli(
            capsys, "--config", cfg, "--source", str(tmp_path / "absent.mp4"), "--no-display"
        )
        assert "[warn]" in out
        assert "lost_after_ms" in out
        assert code == BAD_EXIT  # 影片不存在，但警告仍要先出現
