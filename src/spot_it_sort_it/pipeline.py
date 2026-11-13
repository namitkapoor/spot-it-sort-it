"""Camera -> Anomalib (NPU) -> VLM (GPU, only when needed) -> arm (CPU)."""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import cv2

from .arm import make_arm
from .camera import frames
from .config import load_config
from .decider import Decider, heatmap_overlay
from .detector import AnomalyDetector


def main() -> None:
    ap = argparse.ArgumentParser(description="Spot it, sort it")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--source", help="override camera.source (index, URL or image folder)")
    ap.add_argument("--set", nargs="*", default=[], help="override config, e.g. detector.device=CPU")
    args = ap.parse_args()

    cfg = load_config(args.config, args.set)
    src = args.source if args.source is not None else cfg["camera"]["source"]
    src = int(src) if isinstance(src, str) and src.isdigit() else src

    d = cfg["detector"]
    detector = AnomalyDetector(d["model_xml"], d["device"], d["threshold"], d["borderline_margin"])
    decider = Decider(**cfg["decider"])
    arm = make_arm(cfg["arm"]["driver"])
    print(f"detector on {detector.device} | decider on {decider.device or 'rules'} | arm {cfg['arm']['driver']}")

    log_path = Path(cfg["log"]["csv"])
    log_path.parent.mkdir(parents=True, exist_ok=True)
    frame_dir = log_path.parent / "frames"
    frame_dir.mkdir(exist_ok=True)
    with log_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "score", "anomalous", "action", "defect", "reason", "source",
                    "detect_ms", "decide_ms", "arm_ms", "total_ms"])
        for fid, img in frames(src, cfg["camera"]["width"], cfg["camera"]["height"]):
            t0 = time.perf_counter()
            det = detector(img)
            dec = decider(img, det)
            bin_name = cfg["arm"]["good_bin"] if dec.is_good else cfg["arm"]["reject_bin"]
            arm_ms = arm.place(bin_name)
            total = (time.perf_counter() - t0) * 1000
            print(f"{Path(fid).name:>28}  score={det.score:.3f}  {dec.action:<15} ({dec.source})  "
                  f"det={det.latency_ms:.0f}ms dec={dec.latency_ms:.0f}ms total={total:.0f}ms")
            w.writerow([fid, f"{det.score:.4f}", det.is_anomalous, dec.action, dec.defect, dec.reason, dec.source,
                        f"{det.latency_ms:.1f}", f"{dec.latency_ms:.1f}", f"{arm_ms:.1f}", f"{total:.1f}"])
            if cfg["log"]["save_frames"] and det.is_anomalous:
                cv2.imwrite(str(frame_dir / f"{Path(fid).stem}_{dec.action}.jpg"), heatmap_overlay(img, det.anomaly_map))
    print(f"log -> {log_path}")


if __name__ == "__main__":
    main()
