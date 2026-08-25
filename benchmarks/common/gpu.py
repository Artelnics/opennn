"""Reading the GPU's state, and waiting for it to be worth measuring on.

`gpu_state` and `measure_idle` were byte-identical in all three energy runners
and are lifted unchanged. `wait_for_idle` is not a lift: `cooldown` existed in
three forms with different meanings -- the capacity runners wait for VRAM to
drain, the energy runners wait for VRAM *and* for power to settle back near
idle, and one of the three also returns immediately on CPU. Those are different
questions, so this takes the power bound as an option rather than picking one
family's answer.

Why any of it matters: the engineering audit measured the same bf16
configuration at 6,994 and 8,682 samples/s an hour apart on one machine. Clock
and thermal state move throughput by more than most of the differences these
benchmarks are trying to detect, so a run that does not record the state it
measured in cannot be compared with one taken at another time.
"""

from __future__ import annotations

import re
import subprocess
import time
from typing import Any

__all__ = ["gpu_state", "measure_idle", "used_mib", "wait_for_idle"]

_STATE_FIELDS = (
    "clocks.current.sm,clocks.max.sm,clocks.current.memory,"
    "temperature.gpu,power.limit,power.draw,clocks_throttle_reasons.active"
)

_STATE_KEYS = [
    "sm_clock_mhz", "sm_clock_max_mhz", "mem_clock_mhz",
    "temp_c", "power_limit_w", "power_draw_w", "throttle_reasons",
]


def gpu_state() -> dict[str, Any]:
    """Clocks, temperature, power and any active throttle reason.

    Recorded beside a measurement so a number that looks wrong later can be
    checked against the state it was taken in.
    """
    try:
        out = subprocess.run(
            ["nvidia-smi", f"--query-gpu={_STATE_FIELDS}",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5).stdout.strip().split("\n")[0]

        state: dict[str, Any] = {}
        for key, value in zip(_STATE_KEYS, [v.strip() for v in out.split(",")]):
            try:
                state[key] = float(value)
            except ValueError:
                state[key] = value
        return state
    except Exception:
        return {"error": "nvidia-smi gpu-state query failed"}


def measure_idle(seconds: float = 5.0) -> float:
    """Mean idle power draw in watts, sampled at 100 ms.

    The 30.0 fallback is what the energy runners have always used when
    nvidia-smi says nothing; it is a placeholder, not a measurement, and an
    energy figure resting on it should be treated as unmeasured.
    """
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits",
             "-lms", "100"],
            capture_output=True, text=True, timeout=seconds).stdout
    except subprocess.TimeoutExpired as expired:
        out = expired.stdout.decode() if isinstance(expired.stdout, bytes) else (expired.stdout or "")

    values = [float(x) for x in out.split() if re.fullmatch(r"[0-9.]+", x)]
    return sum(values) / len(values) if values else 30.0


def used_mib() -> float:
    """VRAM in use, in MiB. Raises if nvidia-smi is unavailable."""
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=10).stdout.strip().splitlines()[0]
    return float(out)


def wait_for_idle(seconds: float = 30.0,
                  mib_threshold: float = 1200.0,
                  idle_watts: float | None = None,
                  watt_margin: float = 12.0) -> bool:
    """Wait until the previous run has let go of the card. True if it settled.

    VRAM is always waited on. Power only when `idle_watts` is given, which is
    what separates the energy runners' cooldown from the capacity ones': an
    energy measurement taken while the card is still coming down from the last
    one reads high, whereas a capacity measurement only needs the memory back.
    """
    deadline = time.time() + seconds

    while time.time() < deadline:
        try:
            query = "memory.used,power.draw" if idle_watts is not None else "memory.used"
            out = subprocess.run(
                ["nvidia-smi", f"--query-gpu={query}",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=10).stdout.strip().splitlines()[0]

            readings = [float(x) for x in out.split(",")]
        except Exception:
            # No nvidia-smi, or no GPU: nothing to wait for.
            return True

        if readings[0] <= mib_threshold:
            if idle_watts is None or readings[1] <= idle_watts + watt_margin:
                return True

        time.sleep(0.5)

    return False
