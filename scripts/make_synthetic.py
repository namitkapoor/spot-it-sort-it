"""Generate a toy dataset (washers with/without scratches) to smoke-test the pipeline before the event."""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def washer(rng: np.random.Generator, defect: bool) -> np.ndarray:
    img = np.full((256, 256, 3), 200, np.uint8) + rng.integers(0, 10, (256, 256, 3), dtype=np.uint8)
    c = (128 + int(rng.integers(-4, 5)), 128 + int(rng.integers(-4, 5)))
    cv2.circle(img, c, 90, (90, 90, 95), -1)
    cv2.circle(img, c, 35, (200, 200, 200), -1)
    if defect:
        kind = rng.integers(0, 3)
        if kind == 0:  # scratch
            a = rng.uniform(0, 2 * np.pi)
            p1 = (int(c[0] + 50 * np.cos(a)), int(c[1] + 50 * np.sin(a)))
            p2 = (int(c[0] + 85 * np.cos(a + 0.5)), int(c[1] + 85 * np.sin(a + 0.5)))
            cv2.line(img, p1, p2, (230, 230, 230), 3)
        elif kind == 1:  # chip
            a = rng.uniform(0, 2 * np.pi)
            cv2.circle(img, (int(c[0] + 90 * np.cos(a)), int(c[1] + 90 * np.sin(a))), 18, (200, 200, 200), -1)
        else:  # stain
            cv2.circle(img, (c[0] + int(rng.integers(-60, 60)), c[1] + 60), 12, (30, 60, 140), -1)
    return img


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/synthetic_washer", type=Path)
    ap.add_argument("--good", type=int, default=60)
    ap.add_argument("--defect", type=int, default=15)
    args = ap.parse_args()
    rng = np.random.default_rng(0)
    for split, n, bad in (("good", args.good, False), ("defect", args.defect, True)):
        (args.out / split).mkdir(parents=True, exist_ok=True)
        for i in range(n):
            cv2.imwrite(str(args.out / split / f"{i:03d}.png"), washer(rng, bad))
    print(f"wrote {args.good} good + {args.defect} defect -> {args.out}")


if __name__ == "__main__":
    main()
