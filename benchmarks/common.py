"""Everything the suite needs that is not a model definition.

PLAN.md step 1. One file, four concerns: where a result came from, where the
binaries are, what the GPU was doing, and how a prediction is scored.

It is one file because the alternative was five, and because none of these is
big enough to be worth finding. What matters is that each fact has exactly one
implementation -- the suite it replaces had `versions()` in nine forms and a
binary lookup copied into eighteen files, none of which knew about the build
directory the instructions told you to create.
"""

from __future__ import annotations

import math
import os
import platform
import re
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

# --------------------------------------------------------------------------
# Provenance: what a result can be checked against later
# --------------------------------------------------------------------------

def run_text(command: list[str], cwd: Path | None = None, timeout: int = 120) -> str:
    """Stdout of `command`, or "" if it fails. Never raises: provenance is
    worth collecting on a machine missing a tool, just not worth crashing for."""
    try:
        completed = subprocess.run(command, cwd=cwd, capture_output=True,
                                   text=True, timeout=timeout)
        return completed.stdout.strip() if completed.returncode == 0 else ""
    except Exception:
        return ""

def repo_root(start: Path | None = None) -> Path:
    here = (start or Path(__file__)).resolve().parent
    root = run_text(["git", "-C", str(here), "rev-parse", "--show-toplevel"])
    return Path(root).resolve() if root else here.parent

REPO_ROOT = repo_root()
BENCHMARKS = Path(__file__).resolve().parent
RESULTS = BENCHMARKS / "results"

def git_metadata(root: Path | None = None) -> dict[str, Any]:
    """Commit, branch, and whether the tree was dirty.

    `dirty` decides whether a result can ever be regenerated, so it is recorded
    as a boolean plus a capped sample -- a very dirty tree should not be able
    to fill the artifact with its own status output.
    """
    target = str(root or REPO_ROOT)
    status = run_text(["git", "-C", target, "status", "--porcelain"])
    lines = status.splitlines() if status else []

    return {
        "commit": run_text(["git", "-C", target, "rev-parse", "HEAD"]),
        "branch": run_text(["git", "-C", target, "rev-parse", "--abbrev-ref", "HEAD"]),
        "dirty": bool(lines),
        "dirty_count": len(lines),
        "dirty_sample": lines[:20],
    }

def framework_versions(python: str | None = None) -> dict[str, Any]:
    """Python, the frameworks, what they were built against, and the GPU.

    Each framework is imported in a subprocess: asking torch its version in
    this process would load CUDA into the one about to time something.
    """
    interpreter = python or sys.executable

    versions: dict[str, Any] = {
        "python": sys.version.split()[0],
        "python_executable": interpreter,
        "platform": platform.platform(),
    }

    probes = {
        "torch": "import torch;print(torch.__version__);print(torch.version.cuda);"
                 "print(torch.backends.cudnn.version());print(torch.cuda.get_arch_list())",
        "tensorflow": "import tensorflow as tf;b=tf.sysconfig.get_build_info();"
                      "print(tf.__version__);print(b.get('cuda_version'));"
                      "print(b.get('cudnn_version'))",
    }

    for name, script in probes.items():
        completed = subprocess.run([interpreter, "-c", script],
                                   capture_output=True, text=True, timeout=180)

        if completed.returncode != 0:
            # Why it is absent is provenance too: "not installed" and
            # "installed but fails to import" are different facts.
            reason = completed.stderr.strip().splitlines()
            versions[f"{name}_error"] = reason[-1] if reason else "probe failed"
            continue

        lines = completed.stdout.strip().splitlines()
        if not lines:
            continue

        versions[name] = lines[0]
        if len(lines) > 2:
            versions[f"{name}_built_cuda"] = lines[1]
            versions[f"{name}_built_cudnn"] = lines[2]
        if len(lines) > 3:
            versions[f"{name}_arch_list"] = lines[3]

    nvcc = re.search(r"release\s+([0-9.]+)", run_text(["nvcc", "--version"], timeout=15))
    if nvcc:
        versions["cuda_nvcc"] = nvcc.group(1)

    gpu = run_text(["nvidia-smi", "--query-gpu=name,driver_version,power.limit",
                    "--format=csv,noheader"], timeout=15)
    if gpu:
        versions["gpu"] = gpu.splitlines()[0].strip()

    return versions

def file_info(path: Path) -> dict[str, Any]:
    """Identity of an input file, so a result names the data it measured."""
    path = Path(path)
    info: dict[str, Any] = {"path": str(path), "exists": path.exists()}

    if info["exists"]:
        stat = path.stat()
        info.update(bytes=stat.st_size, mtime=stat.st_mtime)

    return info

SESSION_ENV = "OPENNN_BENCH_SESSION"

def session_id() -> str:
    """The session a run belongs to. Numbers compare only within one.

    A session spans several launches, so it cannot be generated per process:
    export `$OPENNN_BENCH_SESSION` once and everything under it agrees.
    Without it each process gets its own `adhoc-` id, which is deliberately
    awkward, because an artifact whose session nothing else shares is exactly
    what an un-anchored run is.
    """
    return os.environ.get(SESSION_ENV) or f"adhoc-{os.getpid()}"

def result_destination(dirty: bool | None = None) -> Path:
    """The evidence store, or `scratch/` when the tree is dirty.

    Enforced here rather than asked for in prose: the suite this replaces
    stated the rule and checked it nowhere, which is how 39 of its 107
    artifacts came to be dirty-tree results filed as reproducible ones.
    """
    if dirty is None:
        dirty = bool(git_metadata().get("dirty", True))

    destination = RESULTS / "scratch" if dirty else RESULTS
    destination.mkdir(parents=True, exist_ok=True)
    return destination

# --------------------------------------------------------------------------
# Binaries: finding the compiled programs
# --------------------------------------------------------------------------

BUILD_DIRS = ("build-bench", "build", "build-benchmarks", "build-gpu", "build-cuda")
CONFIGS = ("", "Release", "RelWithDebInfo")

def candidate_names(base: str) -> list[str]:
    return [base + ".exe", base] if os.name == "nt" else [base, base + ".exe"]

def find_binary(base: str, root: Path | None = None) -> tuple[str, bool]:
    """Locate benchmark program `base`. Returns `(path, found)`.

    `$OPENNN_<NAME>_BIN` overrides one program, `$OPENNN_BIN` overrides all.
    An override is honoured even when the path does not exist, and `found`
    says which -- a caller should be able to report "you pointed me at a
    binary that is not there" rather than silently running a different one.
    """
    stem = base[len("opennn_"):] if base.startswith("opennn_") else base

    for name in (f"OPENNN_{stem.upper()}_BIN", "OPENNN_BIN"):
        override = os.environ.get(name)
        if override:
            return override, Path(override).exists()

    base_dir = root or REPO_ROOT
    for build in BUILD_DIRS:
        for config in CONFIGS:
            directory = base_dir / build / "bin" / config if config else base_dir / build / "bin"
            for name in candidate_names(base):
                if (directory / name).exists():
                    return str(directory / name), True

    return str(base_dir / BUILD_DIRS[0] / "bin" / candidate_names(base)[0]), False

# --------------------------------------------------------------------------
# GPU: state, and the sampler every run carries
# --------------------------------------------------------------------------

_STATE_FIELDS = ("clocks.current.sm,clocks.max.sm,clocks.current.memory,"
                 "temperature.gpu,power.limit,power.draw,clocks_throttle_reasons.active")

_STATE_KEYS = ("sm_clock_mhz", "sm_clock_max_mhz", "mem_clock_mhz",
               "temp_c", "power_limit_w", "power_draw_w", "throttle_reasons")

def gpu_state() -> dict[str, Any]:
    """Clocks, temperature, power and any active throttle reason, so a number
    that looks wrong later can be checked against the state it was taken in."""
    line = run_text(["nvidia-smi", f"--query-gpu={_STATE_FIELDS}",
                     "--format=csv,noheader,nounits"], timeout=10)
    if not line:
        return {"error": "nvidia-smi unavailable"}

    state: dict[str, Any] = {}
    for key, value in zip(_STATE_KEYS, [v.strip() for v in line.splitlines()[0].split(",")]):
        try:
            state[key] = float(value)
        except ValueError:
            state[key] = value

    return state

def used_mib() -> float:
    """VRAM in use, MiB. 0.0 when there is no GPU to ask."""
    line = run_text(["nvidia-smi", "--query-gpu=memory.used",
                     "--format=csv,noheader,nounits"], timeout=10)
    return float(line.splitlines()[0]) if line else 0.0

def wait_for_idle(seconds: float = 30.0, mib_threshold: float = 1200.0,
                  idle_watts: float | None = None, watt_margin: float = 12.0) -> bool:
    """Wait for the previous run to let go of the card. True if it settled.

    VRAM is always waited on; power only when `idle_watts` is given. An energy
    measurement taken while the card is still coming down from the last run
    reads high, whereas a capacity measurement only needs the memory back.
    """
    deadline = time.time() + seconds

    while time.time() < deadline:
        query = "memory.used,power.draw" if idle_watts is not None else "memory.used"
        line = run_text(["nvidia-smi", f"--query-gpu={query}",
                         "--format=csv,noheader,nounits"], timeout=10)
        if not line:
            return True                      # no GPU: nothing to wait for

        readings = [float(x) for x in line.splitlines()[0].split(",")]
        if readings[0] <= mib_threshold:
            if idle_watts is None or readings[1] <= idle_watts + watt_margin:
                return True

        time.sleep(0.5)

    return False

class Monitor:
    """Samples memory and power for the life of a run.

    This is what lets speed, peak memory and energy come from one execution
    instead of three. The suite this replaces ran the same binary twice --
    once timed, once with a power meter attached -- and filed the results in
    two folders, in two thermal states, as two numbers that could not be
    cross-referenced.

    Sampling always runs, because the cost is one subprocess and the
    alternative is how the separate energy benchmark came to exist.

        with Monitor() as monitor:
            ... run the engine ...
        monitor.peak_mib, monitor.energy_joules(start, end)
    """

    # Below this many samples inside the window, energy is reported as
    # unmeasured rather than as a number. A short run integrated from two
    # readings is not a small energy figure, it is no energy figure.
    MIN_WINDOW_SAMPLES = 4

    def __init__(self, interval_ms: int = 20, measure_idle_first: bool = True,
                 device: str = "cuda"):
        self.interval_ms = interval_ms
        self.device = device
        self.samples: list[tuple[float, float, float]] = []    # unix, MiB, watts
        self.idle_mib = 0.0
        self.idle_watts = 0.0
        self.peak_rss_mib = 0.0
        self._measure_idle = measure_idle_first
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._process: subprocess.Popen | None = None

    def watch_rss(self, pid: int) -> None:
        """Track a child's peak resident set, for CPU runs.

        VmHWM is a high-water mark the kernel maintains, so reading it once
        before the child exits gives its true peak -- polling frequency does
        not matter, only that we read it at all.
        """
        try:
            with open(f"/proc/{pid}/status") as handle:
                for line in handle:
                    if line.startswith("VmHWM:"):
                        self.peak_rss_mib = max(self.peak_rss_mib,
                                                float(line.split()[1]) / 1024.0)
                        return
        except (OSError, ValueError, IndexError):
            pass

    def __enter__(self) -> "Monitor":
        # On CPU there is nothing on the card worth sampling: the GPU sits
        # idle, and reporting its memory and draw as the run's would be a
        # measurement of the wrong device rather than a missing one.
        if self.device != "cuda":
            return self

        if self._measure_idle:
            self.idle_mib, self.idle_watts = self._read_once()

        # One long-lived nvidia-smi streaming with -lms, not a subprocess per
        # sample. Spawning one costs 20-50 ms, which was slower than the
        # interval it was supposedly honouring, so short runs landed no
        # samples inside their own timed window at all.
        try:
            self._process = subprocess.Popen(
                ["nvidia-smi", "--query-gpu=memory.used,power.draw",
                 "--format=csv,noheader,nounits", "-lms", str(self.interval_ms)],
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
        except Exception:
            self._process = None
            return self

        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_) -> None:
        self._stop.set()
        if self._process:
            self._process.terminate()
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._process.kill()
        if self._thread:
            self._thread.join(timeout=5)

    @staticmethod
    def _read_once() -> tuple[float, float]:
        line = run_text(["nvidia-smi", "--query-gpu=memory.used,power.draw",
                         "--format=csv,noheader,nounits"], timeout=10)
        if not line:
            return 0.0, 0.0
        try:
            parts = line.splitlines()[0].split(",")
            return float(parts[0]), float(parts[1])
        except (ValueError, IndexError):
            return 0.0, 0.0

    def _loop(self) -> None:
        assert self._process and self._process.stdout
        for line in self._process.stdout:
            if self._stop.is_set():
                break
            try:
                mib, watts = (float(x) for x in line.split(","))
            except ValueError:
                continue
            self.samples.append((time.time(), mib, watts))

    @property
    def peak_mib(self) -> float:
        """Peak device-used memory over the run, minus the idle reading.

        Whole-device on purpose: it counts the CUDA context and the
        allocator's cached blocks, because that memory really is unavailable
        to anything else. `torch.cuda.max_memory_allocated()` excludes both
        and has no OpenNN equivalent, so it never appears in a comparison.
        """
        if not self.samples:
            return 0.0
        return max(mib for _, mib, _ in self.samples) - self.idle_mib

    def energy_joules(self, start: float | None = None, end: float | None = None) -> float:
        """Board energy over [start, end], trapezoid-integrated.

        The window is the engine's own timed region, taken from the marks it
        prints, so warmup and data loading are outside it. Idle draw is not
        subtracted: the question is what the run costs to perform, and the
        card being on is part of that.
        """
        window = [(t, w) for t, _, w in self.samples
                  if (start is None or t >= start) and (end is None or t <= end)]
        if len(window) < 2:
            return 0.0

        return sum((window[i + 1][0] - window[i][0]) * (window[i + 1][1] + window[i][1]) / 2.0
                   for i in range(len(window) - 1))

    def summary(self, start: float | None = None, end: float | None = None) -> dict[str, Any]:
        if self.device != "cuda":
            # Peak RSS is the CPU counterpart of device-used memory. Energy has
            # none here: it would need RAPL, which is not wired up, and the
            # GPU's draw during a CPU run is the idle card.
            return {
                "peak_mib": round(self.peak_rss_mib, 1),
                "memory_metric": "process_peak_rss",
                "energy_joules": None,
                "energy_wh": None,
                "energy_measurable": False,
                "energy_note": "CPU run: no RAPL counter wired up",
                "samples": 0,
                "window_samples": 0,
            }

        window = [w for t, _, w in self.samples
                  if (start is None or t >= start) and (end is None or t <= end)]

        # A window too short to sample has no energy figure, and saying 0.0 Wh
        # would be a claim rather than an absence. Lengthen the run -- more
        # epochs or repeats -- if the energy cell matters.
        measurable = len(window) >= self.MIN_WINDOW_SAMPLES
        joules = self.energy_joules(start, end) if measurable else None

        return {
            "peak_mib": round(self.peak_mib, 1),
            "memory_metric": "device_used_minus_idle",
            "idle_mib": round(self.idle_mib, 1),
            "idle_watts": round(self.idle_watts, 1),
            "energy_joules": round(joules, 2) if measurable else None,
            "energy_wh": round(joules / 3600.0, 5) if measurable else None,
            "mean_watts": round(sum(window) / len(window), 1) if window else None,
            "samples": len(self.samples),
            "window_samples": len(window),
            "energy_measurable": measurable,
        }

# --------------------------------------------------------------------------
# CPU: which cores, and what state they are in
# --------------------------------------------------------------------------

def core_layout() -> dict[str, list[int]]:
    """Split the CPUs into performance and efficiency cores by peak frequency.

    A hybrid Intel part runs its E-cores materially slower than its P-cores --
    4,200 MHz against 5,400 on this machine, about 22% -- and the scheduler
    decides which a thread gets. That is worse than clock drift for a
    benchmark, because it is discrete and per-thread: two identical runs can
    differ by a fifth purely on placement, and nothing in the result would say
    so. Pinning to P-cores removes the variable rather than averaging it.
    """
    frequencies: dict[int, int] = {}

    for path in Path("/sys/devices/system/cpu").glob("cpu[0-9]*/cpufreq/cpuinfo_max_freq"):
        try:
            frequencies[int(path.parent.parent.name[3:])] = int(path.read_text())
        except (OSError, ValueError):
            continue

    if not frequencies:
        return {"performance": [], "efficiency": []}

    fastest = max(frequencies.values())

    # 100 MHz of slack: P-cores in one package differ slightly from each other
    # (5,400 and 5,300 here), and that is not the split being looked for.
    return {
        "performance": sorted(c for c, f in frequencies.items() if f >= fastest - 100_000),
        "efficiency": sorted(c for c, f in frequencies.items() if f < fastest - 100_000),
    }

def physical_cores(cpus: list[int]) -> int:
    """How many distinct physical cores `cpus` covers.

    SMT siblings share execution units, so for compute-bound work the useful
    thread count is the physical core count, not the logical one.
    """
    groups = set()

    for cpu in cpus:
        path = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list")
        groups.add(path.read_text().strip() if path.exists() else str(cpu))

    return len(groups) or len(cpus)

def cpu_state() -> dict[str, Any]:
    """Governor, turbo and core layout, recorded beside a CPU measurement.

    None of these can be set without root, so the artifact records what they
    were rather than asserting what they should be. A reader can then tell a
    locked-down run from an opportunistic one.
    """
    def read(path: str) -> str | None:
        try:
            return Path(path).read_text().strip()
        except OSError:
            return None

    layout = core_layout()
    no_turbo = read("/sys/devices/system/cpu/intel_pstate/no_turbo")

    return {
        "model": next((line.split(":", 1)[1].strip()
                       for line in Path("/proc/cpuinfo").read_text().splitlines()
                       if line.startswith("model name")), ""),
        "governor": read("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor"),
        "scaling_driver": read("/sys/devices/system/cpu/cpu0/cpufreq/scaling_driver"),
        "turbo_enabled": None if no_turbo is None else no_turbo == "0",
        "performance_cores": layout["performance"],
        "efficiency_cores": layout["efficiency"],
    }

# --------------------------------------------------------------------------
# Metrics: one scoring, every engine
# --------------------------------------------------------------------------

def roc_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    y = np.asarray(y_true, dtype=np.float64).reshape(-1)
    s = np.asarray(scores, dtype=np.float64).reshape(-1)
    positives = y >= 0.5
    n_pos = int(positives.sum())
    n_neg = int(y.size - n_pos)

    if n_pos == 0 or n_neg == 0:
        return float("nan")

    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(y.size, dtype=np.float64)
    sorted_scores = s[order]

    i = 0
    while i < y.size:
        j = i + 1
        while j < y.size and sorted_scores[j] == sorted_scores[i]:
            j += 1
        ranks[order[i:j]] = (i + 1 + j) / 2.0
        i = j

    return (float(ranks[positives].sum()) - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)

def binary_metrics(y_true: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    y = np.asarray(y_true, dtype=np.float64).reshape(-1)
    p = np.clip(np.asarray(probabilities, dtype=np.float64).reshape(-1), 1.0e-7, 1.0 - 1.0e-7)

    return {
        "test_accuracy": float(((p >= 0.5) == (y >= 0.5)).mean()),
        "test_log_loss": float(-(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)).mean()),
        "test_roc_auc": float(roc_auc(y, p)),
    }

def agrees(values: list[float], tolerance: float) -> bool:
    """Whether every engine's number sits within `tolerance` of their mean.

    The cross-engine quality gate: engines will never match exactly -- different
    RNG, different initialisation, different kernel order -- so agreement is a
    band, not equality. A cell failing this reports no speed number, because a
    speed win bought by computing something different is not a speed win.
    """
    usable = [v for v in values if not math.isnan(v)]
    if len(usable) < 2:
        return True

    mean = sum(usable) / len(usable)
    return mean == 0 or all(abs(v - mean) / abs(mean) <= tolerance for v in usable)
