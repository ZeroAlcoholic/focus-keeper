"""一次性模型取得腳本（規格 §9）。

執行期的偵測器**不會**下載任何權重；權重必須由本腳本明確、手動取得，
並印出 SHA-256 讓你填回 ``config.yaml`` 與 ``THIRD_PARTY_NOTICES.md``。

用法::

    python scripts/fetch_model.py                      # 路線 A：YuNet（預設）
    python scripts/fetch_model.py --detector mediapipe # 路線 B：BlazeFace
    python scripts/fetch_model.py --verify-only        # 只重算現有檔案的 SHA-256
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"

#: 權重來源。URL 固定到特定 tag／版本目錄，不使用會隨時間變動的 branch HEAD。
SOURCES: dict[str, dict[str, str]] = {
    "yunet": {
        "filename": "face_detection_yunet_2023mar.onnx",
        "url": (
            "https://github.com/opencv/opencv_zoo/raw/4.10.0/"
            "models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
        ),
        "license": "MIT（opencv_zoo models/face_detection_yunet 目錄 LICENSE）",
        "upstream": "https://github.com/opencv/opencv_zoo/tree/4.10.0/models/face_detection_yunet",
    },
    "mediapipe": {
        "filename": "blaze_face_short_range.tflite",
        "url": (
            "https://storage.googleapis.com/mediapipe-models/face_detector/"
            "blaze_face_short_range/float16/1/blaze_face_short_range.tflite"
        ),
        "license": "Apache-2.0（MediaPipe 模型卡；請自行留存下載日期與版本目錄 /1/）",
        "upstream": "https://ai.google.dev/edge/mediapipe/solutions/vision/face_detector",
    },
}


def sha256_of(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_suffix(target.suffix + ".part")
    print(f"[fetch] {url}")
    with urllib.request.urlopen(url, timeout=60) as response:
        if response.status != 200:
            raise RuntimeError(f"HTTP {response.status}")
        data = response.read()
    temp.write_bytes(data)
    temp.replace(target)
    print(f"[ok]    寫入 {target}（{len(data):,} bytes）")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="取得並驗證本機模型權重")
    parser.add_argument(
        "--detector", default="yunet", choices=sorted(SOURCES),
        help="要取得的權重（預設 yunet，即路線 A）",
    )
    parser.add_argument("--expect-sha256", default=None, help="下載後比對此雜湊，不符即失敗")
    parser.add_argument("--verify-only", action="store_true", help="不下載，只計算現有檔案雜湊")
    parser.add_argument("--force", action="store_true", help="檔案已存在時仍重新下載")
    args = parser.parse_args(argv)

    spec = SOURCES[args.detector]
    target = MODELS_DIR / spec["filename"]

    if args.verify_only:
        if not target.is_file():
            print(f"[error] 檔案不存在：{target}", file=sys.stderr)
            return 1
    elif target.is_file() and not args.force:
        print(f"[skip]  已存在：{target}（要重抓請加 --force）")
    else:
        try:
            download(spec["url"], target)
        except Exception as exc:
            print(f"[error] 下載失敗：{exc}", file=sys.stderr)
            print(
                "        可手動下載後放到 models/ 目錄，再執行 --verify-only：\n"
                f"        {spec['url']}",
                file=sys.stderr,
            )
            return 1

    digest = sha256_of(target)
    size = target.stat().st_size

    if args.expect_sha256 and digest.lower() != args.expect_sha256.lower():
        print(
            f"[error] SHA-256 不符\n  期望：{args.expect_sha256}\n  實際：{digest}",
            file=sys.stderr,
        )
        return 1

    print("\n=== 請填回設定與授權文件 ===")
    print(f"detector     : {args.detector}")
    print(f"model_path   : models/{spec['filename']}")
    print(f"bytes        : {size:,}")
    print(f"sha256       : {digest}")
    print(f"license      : {spec['license']}")
    print(f"upstream     : {spec['upstream']}")
    print(f"\n寫入 config.yaml → detectors.{args.detector}.sha256: \"{digest}\"")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
