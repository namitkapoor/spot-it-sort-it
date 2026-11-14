"""Per-item decision log required for judging: one line per item — number, decision, confidence, timestamp.

Each decision is written to disk BEFORE the arm is told to move. An item stays "pending" (logged, not yet
picked) until the arm reports it, so the same object seen again in later frames is not logged twice.
Latency for scoring = arm_start - seen (item enters view -> first arm motion).
"""
from __future__ import annotations

import csv
import os
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

DEST = {"normal": "box", "anomalous": "reject", "hold": "hold"}
FIELDS = ["item", "decision", "destination", "confidence", "score", "timestamp", "seen", "arm_start", "latency_ms",
          "center_x", "center_y", "box", "frames", "run"]


def iso(t: float) -> str:
    return datetime.fromtimestamp(t, timezone.utc).isoformat(timespec="milliseconds")


def confidence(label: str, score: float, box_lo: float, reject_hi: float) -> float:
    """0.5 at the edge of the hold band, rising to 1.0 the further the score is from it."""
    if label == "normal":
        return round(0.5 + 0.5 * min(1.0, (box_lo - score) / max(box_lo, 1e-6)), 3)
    if label == "anomalous":
        return round(0.5 + 0.5 * min(1.0, (score - reject_hi) / max(reject_hi, 1e-6)), 3)
    mid = (box_lo + reject_hi) / 2  # hold: 0.5 at the band edges, lower towards the middle
    return round(0.5 * abs(score - mid) / max((reject_hi - box_lo) / 2, 1e-6), 3)


@dataclass
class Item:
    item: int
    decision: str
    destination: str
    confidence: float
    score: float
    seen: float
    decided: float
    center: tuple[int, int]
    box: list[int]
    frames: int
    arm_start: float | None = None


@dataclass
class ItemLog:
    root: Path = field(default_factory=lambda: Path(os.environ.get("VIAM_MODULE_DATA", "/root/sortit-logs")))
    match_px: int = 45
    run: str = ""
    items: list[Item] = field(default_factory=list)

    def __post_init__(self) -> None:
        self._lock = threading.Lock()
        self.root.mkdir(parents=True, exist_ok=True)
        if not self.run:
            self.new_run()

    @property
    def path(self) -> Path:
        return self.root / f"items-{self.run}.csv"

    def new_run(self) -> str:
        with self._lock:
            self.run = datetime.now().strftime("%Y%m%d-%H%M%S")
            self.items = []
            with self.path.open("w", newline="") as f:
                csv.writer(f).writerow(FIELDS)
        return self.run

    def pending(self) -> list[Item]:
        return [i for i in self.items if i.arm_start is None]

    def _known(self, center: tuple[int, int]) -> Item | None:
        for it in self.pending():
            if abs(it.center[0] - center[0]) <= self.match_px and abs(it.center[1] - center[1]) <= self.match_px:
                return it
        return None

    def record(self, objs, seen: float, frames: int, box_lo: float, reject_hi: float) -> list[Item]:
        """Log every object not already pending. Returns the newly logged items (written before returning)."""
        new = []
        with self._lock:
            for o in objs:
                if self._known(o.center):
                    continue
                it = Item(len(self.items) + 1, o.label, DEST[o.label], confidence(o.label, o.score, box_lo, reject_hi),
                          round(o.score, 4), seen, time.time(), o.center, [o.x1, o.y1, o.x2, o.y2], frames)
                self.items.append(it)
                new.append(it)
            self._rewrite()
        return new

    def arm_start(self, item: int) -> Item:
        with self._lock:
            it = next(i for i in self.items if i.item == item)
            it.arm_start = time.time()
            self._rewrite()
            return it

    def _rewrite(self) -> None:
        tmp = self.path.with_suffix(".tmp")
        with tmp.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(FIELDS)
            for i in self.items:
                lat = round((i.arm_start - i.seen) * 1000) if i.arm_start else ""
                w.writerow([i.item, i.decision, i.destination, i.confidence, i.score, iso(i.decided), iso(i.seen),
                            iso(i.arm_start) if i.arm_start else "", lat, i.center[0], i.center[1],
                            " ".join(map(str, i.box)), i.frames, self.run])
            f.flush()
            os.fsync(f.fileno())
        tmp.replace(self.path)

    def as_dicts(self) -> list[dict]:
        return [{"item": i.item, "decision": i.decision, "destination": i.destination, "confidence": i.confidence,
                 "score": i.score, "timestamp": iso(i.decided), "seen": iso(i.seen),
                 "arm_start": iso(i.arm_start) if i.arm_start else None,
                 "latency_ms": round((i.arm_start - i.seen) * 1000) if i.arm_start else None,
                 "center": list(i.center), "box": i.box} for i in self.items]
