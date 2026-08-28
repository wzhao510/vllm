# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Generate test images of different resolutions for EPD TTFT benchmark.

Qwen3-VL 视觉 token 数大致按 (H/16) * (W/16) / 4 估算
(16x16 patch, 4 patch merge 成 1 token).
"""
import os
import random

from PIL import Image

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

# (filename, (W, H), expected_visual_tokens_approx)
IMAGES = [
    ("small_128.jpg", (160, 160), 100),    # ~100 tokens
    ("mid_512.jpg", (384, 384), 576),      # ~576 tokens
    ("large_2048.jpg", (720, 720), 2025),  # ~2025 tokens
]


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    for name, size, expected in IMAGES:
        path = os.path.join(OUT_DIR, name)
        # 用纯色 + 少量随机像素,保证三张图 hash 不同
        # (相同 hash 会导致 cache 误命中,无法独立测三组数据)
        img = Image.new("RGB", size, color=(128, 64, 200))
        pixels = img.load()
        for _ in range(200):
            x = random.randint(0, size[0] - 1)
            y = random.randint(0, size[1] - 1)
            pixels[x, y] = (
                random.randint(0, 255),
                random.randint(0, 255),
                random.randint(0, 255),
            )
        img.save(path, "JPEG", quality=85)
        print(f"Generated {path}  size={size}  expected_tokens~={expected}")


if __name__ == "__main__":
    main()
