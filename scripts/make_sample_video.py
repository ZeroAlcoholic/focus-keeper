"""產生驗收用的示範影片，讓沒有攝影機的環境也能重現整條流程。

情境（預設 10 秒 @ 30 FPS）::

    0.0 - 2.0 s   主角在中央
    2.0 - 5.0 s   主角離場        -> PRIMARY_MISSING(3.0s) / PRIMARY_LEFT(4.0s)
    5.0 - 7.0 s   主角回到中央     -> 事件關閉
    7.0 -10.0 s   出現第二人       -> MULTI_PERSON(7.5s)

**影片使用合成臉，不是真人**：可用來驗流程與時序，不能用來驗偵測準確度。

用法::

    python scripts/make_sample_video.py                 # 產生 media/sample.mp4
    python scripts/make_sample_video.py --out foo.mp4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from focus_keeper.synthetic import draw_synthetic_face  # noqa: E402

WIDTH, HEIGHT, FPS = 960, 540, 30
CENTER = (WIDTH // 2, 270)
SECOND = (170, 285)
T_LEAVE, T_RETURN, T_SECOND, T_END = 2.0, 5.0, 7.0, 10.0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="產生驗收用示範影片（合成臉）")
    parser.add_argument("--out", default=str(ROOT / "media" / "sample.mp4"))
    parser.add_argument("--seconds", type=float, default=T_END)
    args = parser.parse_args(argv)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (WIDTH, HEIGHT))
    if not writer.isOpened():
        print("[error] 無法建立 VideoWriter（缺 mp4v 編碼器）", file=sys.stderr)
        return 1

    total = int(args.seconds * FPS)
    for i in range(total):
        t = i / FPS
        frame = np.full((HEIGHT, WIDTH, 3), 210, np.uint8)
        if t < T_LEAVE or t >= T_RETURN:
            draw_synthetic_face(frame, *CENTER, scale=1.0)
        if t >= T_SECOND:
            draw_synthetic_face(frame, *SECOND, scale=1.0)
        writer.write(cv2.GaussianBlur(frame, (5, 5), 0))
    writer.release()

    print(f"[ok] 已寫入 {out}（{total} 格 / {args.seconds} 秒 / {WIDTH}x{HEIGHT} @ {FPS} FPS）")
    print("     預期事件：PRIMARY_MISSING ~3.0s、PRIMARY_LEFT ~4.0s、MULTI_PERSON ~7.5s")
    print(f"     驗證：python -m focus_keeper.cli --source {out} --no-display --report report.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
