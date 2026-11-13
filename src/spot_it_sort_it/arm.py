"""Stage 3 — physical sorting. The arm SDK is revealed at the event; implement `Arm` for it."""
from __future__ import annotations

import time
from abc import ABC, abstractmethod


class Arm(ABC):
    @abstractmethod
    def place(self, bin_name: str) -> float:
        """Pick the current item and drop it in `bin_name`. Return elapsed ms."""


class MockArm(Arm):
    def place(self, bin_name: str) -> float:
        t0 = time.perf_counter()
        print(f"  [arm] -> {bin_name}")
        return (time.perf_counter() - t0) * 1000


def make_arm(driver: str) -> Arm:
    if driver == "mock":
        return MockArm()
    raise NotImplementedError(f"Arm driver '{driver}' not implemented yet — add it in arm.py")
