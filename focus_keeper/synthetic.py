"""合成測試素材產生器。

放在套件內而非 tests/：`scripts/make_sample_video.py` 需要它，
產品腳本不應依賴測試程式碼（依賴方向必須單向）。
"""

from __future__ import annotations

import cv2

__all__ = ["draw_synthetic_face"]


def draw_synthetic_face(image, cx: int, cy: int, scale: float = 1.0):
    """在影像上畫一張**合成**臉，YuNet 可穩定偵測（score 約 0.72~0.86）。

    這是測試素材，不是真人。合成臉比真實臉難偵測，因此若真實素材上通過門檻，
    合成臉多半也會通過——但**準確度不能靠合成臉驗收**，只能驗流程是否接通。
    """
    a, b = int(95 * scale), int(125 * scale)
    cv2.ellipse(image, (cx, cy), (a, b), 0, 0, 360, (188, 170, 158), -1)
    cv2.ellipse(
        image, (cx, cy - int(130 * scale)), (int(100 * scale), int(70 * scale)),
        0, 0, 360, (60, 45, 40), -1,
    )
    for dx in (-int(36 * scale), int(36 * scale)):
        cv2.ellipse(
            image, (cx + dx, cy - int(25 * scale)), (int(18 * scale), int(11 * scale)),
            0, 0, 360, (250, 250, 250), -1,
        )
        cv2.circle(image, (cx + dx, cy - int(25 * scale)), max(2, int(7 * scale)), (40, 30, 25), -1)
    cv2.ellipse(
        image, (cx, cy + int(12 * scale)), (int(9 * scale), int(22 * scale)),
        0, 0, 360, (168, 150, 140), -1,
    )
    cv2.ellipse(
        image, (cx, cy + int(62 * scale)), (int(32 * scale), int(13 * scale)),
        0, 0, 180, (120, 80, 80), -1,
    )
    return image
