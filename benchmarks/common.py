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

import bisect
import ctypes
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

def clocks_locked() -> bool:
    """Whether the GPU clock has been pinned for measurement.

    Inferred from persistence mode, which `gpu_clocks.sh lock` enables
    alongside `-lgc`. It is a proxy rather than a direct read: the applications
    -clock query this would otherwise use reports "Requested functionality has
    been deprecated" on this driver. A false positive needs someone to enable
    persistence by hand and not lock the clock, which is not an accident
    anyone has.
    """
    return run_text(["nvidia-smi", "--query-gpu=persistence_mode",
                     "--format=csv,noheader"], timeout=10).strip() == "Enabled"

def cpu_busy_fraction(seconds: float = 1.0,
                     cores: list[int] | None = None) -> float:
    """How much of the machine is already working, sampled now.

    Not the load average, which is useless here: it decays over minutes, so
    after the suite's own previous launch it reads 21 on a machine that is
    idle. This is the instantaneous non-idle fraction, quiet measuring under
    0.01 and one saturated core about one over the thread count.

    `cores` narrows it to the ones the run can actually be disturbed by. A CPU
    cell is `taskset`-pinned to the P-cores, so work stranded on an E-core
    cannot touch it -- but the aggregate line counts that work anyway, and the
    cell is then filed as scratch for interference it was structurally immune
    to. That is not hypothetical: a browser parked on the E-cores sent
    cpu-lstm-infer and cuda-transformer-infer to scratch on 2026-09-01 at 6.0%
    and 3.7%, with every other gate passing.

    Left as the whole machine when `cores` is None, which is right for a CUDA
    cell: nothing pins those, and their input pipeline can use any core.
    """
    wanted = None if cores is None else {f"cpu{index}" for index in cores}

    def snapshot() -> tuple[int, int]:
        total = idle = 0
        for line in Path("/proc/stat").read_text().split("\n"):
            fields = line.split()
            if not fields or not fields[0].startswith("cpu"):
                continue
            if wanted is None:
                if fields[0] != "cpu":                   # the aggregate line
                    continue
            elif fields[0] not in wanted:
                continue
            values = [int(v) for v in fields[1:]]
            total += sum(values)
            idle += values[3] + values[4]                # idle + iowait
        return total, idle

    total_before, idle_before = snapshot()
    time.sleep(seconds)
    total_after, idle_after = snapshot()

    elapsed = total_after - total_before
    if elapsed <= 0:
        return 0.0

    return max(0.0, 1.0 - (idle_after - idle_before) / elapsed)


# One busy core on this 28-thread part is about 0.036, and a quiet machine
# measures under 0.01, so this trips on roughly a single competing process.
BUSY_THRESHOLD = float(os.environ.get("OPENNN_BENCH_BUSY_THRESHOLD", "0.03"))


class ForeignActivity:
    """CPU time spent by everything that is not the launch, second by second.

    `cpu_busy_fraction` reads the machine before the first launch and after
    the last, and nothing in between -- the one place a long window can be
    hurt. That was found the expensive way: on 2026-09-02 an editor drawing a
    conversation on the E-cores, while a 2.4 s dense-training window ran on
    the P-cores, cost the launch-bound step 4-12% on both engines and left
    both edge samples quiet. This watches every second of a launch and keeps
    the worst one.

    Foreign means outside the launch: not the runner, not the launched
    process, not its descendants (PyTorch's compile workers are children of
    its driver), and not kernel threads, whose time under a CNN cell is the
    driver's interrupt work for the launch itself. Charging by process rather
    than reading /proc/stat is what makes that separation possible. Only
    live processes are charged, so one that starts and exits between two
    samples is missed -- at a second per sample that is under 1/28 of the
    machine for under a second, well below the threshold. For a pinned CPU
    cell only work last seen on the watched cores counts, as with the edge
    samples. The cost is one pass over /proc a second, a few milliseconds.
    """

    KERNEL_THREAD = 0x00200000                           # PF_KTHREAD

    def __init__(self, launched_pid: int, cores: list[int] | None = None,
                 interval: float = 1.0):
        self.own = {os.getpid(), launched_pid}
        self.cores = None if cores is None else set(cores)
        self.interval = interval
        self.samples: list[tuple[float, float, float]] = []   # start, end, fraction
        self.ncpus = os.cpu_count() or 1
        self.clk_tck = os.sysconf("SC_CLK_TCK")
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    @staticmethod
    def _processes() -> dict[int, tuple[int, int, int, int]]:
        """pid -> (ppid, flags, cpu ticks, last cpu)."""
        out = {}
        for entry in os.scandir("/proc"):
            if not entry.name.isdigit():
                continue
            try:
                with open(f"/proc/{entry.name}/stat") as handle:
                    text = handle.read()
            except OSError:
                continue
            rest = text[text.rindex(")") + 2:].split()
            # fields after the command name: state ppid ... flags(6) utime(11)
            # stime(12) ... processor(36)
            ticks = int(rest[11]) + int(rest[12])
            out[int(entry.name)] = (int(rest[1]), int(rest[6]), ticks, int(rest[36]))
        return out

    def _foreign_ticks(self, procs: dict) -> dict[int, int]:
        tree = set(self.own)
        grew = True
        while grew:                                      # descendants, to any depth
            grew = False
            for pid, (ppid, _, _, _) in procs.items():
                if ppid in tree and pid not in tree:
                    tree.add(pid)
                    grew = True
        return {pid: ticks for pid, (_, flags, ticks, cpu) in procs.items()
                if pid not in tree and not flags & self.KERNEL_THREAD
                and (self.cores is None or cpu in self.cores)}

    def _loop(self) -> None:
        previous = self._foreign_ticks(self._processes())
        mark = time.time()
        while not self._stop.wait(self.interval):
            current = self._foreign_ticks(self._processes())
            now = time.time()
            used = sum(max(0, ticks - previous.get(pid, ticks)) for pid, ticks in current.items())
            watched = self.ncpus if self.cores is None else len(self.cores)
            self.samples.append((mark, now, used / (self.clk_tck * (now - mark) * watched)))
            previous, mark = current, now

    def __enter__(self) -> "ForeignActivity":
        self._thread.start()
        return self

    def __exit__(self, *_) -> None:
        self._stop.set()
        self._thread.join()

    def worst(self, start: float | None = None, end: float | None = None) -> dict:
        """The busiest foreign second overlapping [start, end] (unix), or the
        whole launch when a mark is missing."""
        inside = [s for s in self.samples
                  if start is None or end is None or (s[1] > start and s[0] < end)]
        if not inside:
            return {"max": 0.0, "at": None, "seconds": 0,
                    "window": "timed" if start and end else "whole"}
        peak = max(inside, key=lambda s: s[2])
        return {"max": round(peak[2], 4), "at": round(peak[0], 3), "seconds": len(inside),
                "window": "timed" if start and end else "whole"}


def result_destination(dirty: bool | None = None, device: str = "cuda",
                       busy: bool = False) -> Path:
    """The evidence store, or `scratch/` when the run cannot be evidence.

    Two conditions, both enforced here rather than asked for in prose.

    A dirty tree, because a result that cannot be regenerated is not evidence.
    The suite this replaces stated that rule and checked it nowhere, which is
    how 39 of its 107 artifacts came to be dirty-tree results filed as
    reproducible ones.

    And a machine that was not quiet, because a competing process does not
    slow both engines equally -- a single browser tab at one core cost 35% of
    achievable memory bandwidth here, which moved a bandwidth-bound GEMM step
    by that much while leaving cache-resident ones untouched.

    And an unlocked GPU clock, for the same reason at one remove: this card
    drifts about 8% across a day, so margins under ~2% are not resolvable while
    it floats, and a transformer run read 986 and 482 samples/s for identical
    work fifteen minutes apart. Numbers taken that way are worth having --
    they catch gross regressions and prove the plumbing -- but they are not
    worth citing, and the filesystem should be the thing that remembers the
    difference.
    """
    if dirty is None:
        dirty = bool(git_metadata().get("dirty", True))

    unlocked = device == "cuda" and not clocks_locked()
    destination = (RESULTS / "scratch"
                   if (dirty or unlocked or bool(busy)) else RESULTS)
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

# --------------------------------------------------------------------------
# CPU: the RAPL energy counter
# --------------------------------------------------------------------------

RAPL_ROOT = Path("/sys/class/powercap")

def rapl_domain() -> dict[str, Any] | None:
    """The CPU package energy counter, or None when it cannot be read.

    `package-0` and not `core`: the package domain covers the cores *and* the
    uncore -- the memory controller, the ring, the caches -- and a benchmark
    that moves data pays for those as surely as it pays for the multipliers.
    Reporting the core domain alone would flatter a bandwidth-bound run by
    charging it only for arithmetic. Both are listed here so the artifact can
    say which was used rather than leaving the reader to guess.

    Returns None rather than raising when the counter is absent (AMD without
    the module, a VM) or unreadable, which is its state on a stock kernel:
    `energy_uj` is 0400 root-only after CVE-2020-8694, the PLATYPUS
    side-channel. Making it readable is a deliberate act by whoever set the
    machine up, and a run on a machine where nobody did should say the figure
    is missing, not invent one.
    """
    for entry in sorted(RAPL_ROOT.glob("intel-rapl:*")):
        name_file, energy_file = entry / "name", entry / "energy_uj"
        try:
            if name_file.read_text().strip() != "package-0":
                continue
            energy_file.read_text()                     # readable, not just present
            wrap = int((entry / "max_energy_range_uj").read_text().strip())
        except (OSError, ValueError):
            continue
        return {"path": energy_file, "name": "package-0", "wrap_uj": wrap}

    return None

def read_rapl_uj(domain: dict[str, Any]) -> int | None:
    try:
        return int(domain["path"].read_text().strip())
    except (OSError, ValueError):
        return None

# --------------------------------------------------------------------------
# GPU: the driver's own power samples
# --------------------------------------------------------------------------

class Nvml:
    """The NVML calls the monitor needs, bound with ctypes.

    This replaces `nvidia-smi --query-gpu=power.draw -lms 20`, and the reason
    is what `power.draw` is: on Ampere and later it is the driver's
    *one-second moving average* (`power.draw.average`; nvidia-smi documents
    it). Polled at 20 ms it still cannot see inside a second, so a 70 ms
    burst of GEMMs in an otherwise idle second reads as 45 W however finely
    it is sampled -- which is what every sub-second cell in the store had
    been reporting as its energy, memory-bound bursts at 200+ W included.

    The driver keeps its own ring of about 120 instantaneous board-power
    samples, one every 20 ms, timestamped by it (`nvmlDeviceGetSamples`,
    `NVML_TOTAL_POWER_SAMPLES`). Draining that ring is the one user-space
    source of sub-second power on this driver, and it is what the monitor
    integrates. The alternatives were measured and rejected on the
    reference machine, driver 610.43:

    - `power.draw.instant` (`NVML_FI_DEV_POWER_INSTANT`) refreshes every
      500 ms, as does `nvmlDeviceGetPowerUsage`.
    - `nvmlDeviceGetTotalEnergyConsumption` is a synchronous firmware query
      (~4 ms) that stalls the accumulator it reads: polled every 10 ms it
      under-counts a steady 233 W load by 60%, every 100 ms by 10%, every
      500 ms by 1%. Read once at each end of a run it is right, and that is
      the only way it is used here -- as a whole-run cross-check.
    """

    TOTAL_POWER_SAMPLES = 0          # nvmlSamplingType_t
    NOT_FOUND = 6                    # nvmlReturn_t: nothing newer than the timestamp

    class Sample(ctypes.Structure):  # nvmlSample_t
        _fields_ = [("timestamp_us", ctypes.c_ulonglong), ("value", ctypes.c_ulonglong)]

    class Memory(ctypes.Structure):  # nvmlMemory_v2_t
        _fields_ = [("version", ctypes.c_uint), ("total", ctypes.c_ulonglong),
                    ("reserved", ctypes.c_ulonglong), ("free", ctypes.c_ulonglong),
                    ("used", ctypes.c_ulonglong)]

    def __init__(self, index: int = 0):
        self.lib = ctypes.CDLL("libnvidia-ml.so.1")
        if self.lib.nvmlInit_v2() != 0:
            raise OSError("nvmlInit failed")
        self.handle = ctypes.c_void_p()
        if self.lib.nvmlDeviceGetHandleByIndex_v2(index, ctypes.byref(self.handle)) != 0:
            self.lib.nvmlShutdown()
            raise OSError(f"NVML has no device {index}")
        self._ring = (self.Sample * 256)()          # the driver's ring holds ~120
        self._memory = self.Memory()
        self._memory.version = (2 << 24) | ctypes.sizeof(self.Memory)

    def close(self) -> None:
        self.lib.nvmlShutdown()

    def memory_used_mib(self) -> float | None:
        if self.lib.nvmlDeviceGetMemoryInfo_v2(self.handle, ctypes.byref(self._memory)) != 0:
            return None
        return self._memory.used / 2**20

    def power_samples(self, newer_than_us: int) -> list[tuple[int, float]]:
        """(driver timestamp in unix microseconds, watts), oldest first, for
        every sample the driver took after `newer_than_us`."""
        value_type, count = ctypes.c_uint(), ctypes.c_uint(len(self._ring))
        rc = self.lib.nvmlDeviceGetSamples(
            self.handle, self.TOTAL_POWER_SAMPLES, ctypes.c_ulonglong(newer_than_us),
            ctypes.byref(value_type), ctypes.byref(count), self._ring)
        if rc != 0:
            return []
        # nvmlValue_t is a union; a power sample is a 32-bit milliwatt count
        # and the upper half of the word is left uncleared.
        return [(s.timestamp_us, (s.value & 0xFFFFFFFF) / 1e3)
                for s in self._ring[:count.value]]

    def energy_mj(self) -> int | None:
        """The firmware energy counter. Read sparingly -- see the class note."""
        value = ctypes.c_ulonglong()
        if self.lib.nvmlDeviceGetTotalEnergyConsumption(self.handle, ctypes.byref(value)) != 0:
            return None
        return value.value

# The name the CPU path files its memory reading under. It lives here, and
# run.py imports it, because it is a gate as well as a label: it says which
# peak a host baseline may be subtracted from, and the GPU path's
# `device_used_minus_idle` is not one -- that reading has already had the idle
# card taken off it. Keeping the literal in two files is what let it drift: the
# metric was renamed when the reading moved from total to anonymous RSS (see
# Monitor.watch_rss), run.py's gate went on asking for the old
# "process_peak_rss", and from that rename until now no CPU cell had a baseline
# subtracted at all.
HOST_MEMORY_METRIC = "process_peak_anonymous_rss"

# The stdout field a driver has to print for `workload_mib` to be computable:
# the framework baseline in the *same* quantity the peak is read in, anonymous
# resident pages, sampled once before the framework does any work.
#
# Deliberately not `baseline_rss_mib`, which the eight family drivers print
# today from /proc/self/statm's resident field (footprint's pair prints the
# same reading as `baseline_ram_mb`) -- that is total RSS, and total-minus-
# anonymous is not a workload figure but a negative number. On the 2026-09-03
# publish round it is negative for at least one engine on every cpu-* cell
# (cpu-lstm-infer: OpenNN 189.2 MiB peak against a 218.6 MiB baseline, PyTorch
# 458.7 against 671.9), so subtracting it would clamp the published workload to
# zero rather than correct it. Until a driver emits this field the runner says
# so in the artifact instead of subtracting the wrong one.
HOST_BASELINE_FIELD = "baseline_anonymous_rss_mib"

class Monitor:
    """Samples memory and power for the life of a run.

    This is what lets speed, peak memory and energy come from one execution
    instead of three. The suite this replaces ran the same binary twice --
    once timed, once with a power meter attached -- and filed the results in
    two folders, in two thermal states, as two numbers that could not be
    cross-referenced.

    Sampling always runs, because the cost is one thread and the alternative
    is how the separate energy benchmark came to exist.

        with Monitor() as monitor:
            ... run the engine ...
        monitor.peak_mib, monitor.energy_joules(start, end)
    """

    # Below this many power samples inside the window, energy is reported as
    # unmeasured rather than as a number. The driver samples every 20 ms, so
    # this is a one-second window, which keeps the two partial intervals at
    # its edges under 2% of it. A shorter run has no energy figure: a burst
    # of GEMMs integrated from three readings is not a small energy figure,
    # it is no energy figure. Lengthen the run -- more epochs or repeats.
    MIN_WINDOW_SAMPLES = 50

    def __init__(self, interval_ms: int = 20, measure_idle_first: bool = True,
                 device: str = "cuda"):
        # Memory is polled every `interval_ms`; the driver's power ring is
        # drained every `power_drain_ms`. The ring keeps ~2.4 s of 20 ms
        # samples, so one drain a second loses nothing and costs a fiftieth
        # of the calls. Neither rate moves a result: drained at 20 ms, at
        # 1 s and not at all, a launch-bound dense cell read the same
        # throughput (interleaved, 2026-09-02).
        self.interval_ms = int(os.environ.get("OPENNN_BENCH_MONITOR_MS", interval_ms))
        self.power_drain_ms = int(os.environ.get("OPENNN_BENCH_POWER_DRAIN_MS", 1000))
        self.device = device
        self.memory_samples: list[tuple[float, float]] = []    # unix, MiB
        self.power_samples: list[tuple[float, float]] = []     # unix, watts
        self.power_metric: str | None = None
        self.idle_mib = 0.0
        self.idle_watts = 0.0
        self.peak_rss_mib = 0.0
        self.peak_file_mib = 0.0
        # The firmware energy counter differenced over the monitor's whole
        # life: warmup, load and teardown included, so never the window's
        # figure -- but a bound on it, from an independent instrument.
        self.run_energy_joules: float | None = None
        self._measure_idle = measure_idle_first
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._process: subprocess.Popen | None = None
        self._nvml: Nvml | None = None
        self._counter_start_mj: int | None = None
        # (unix, cumulative joules) from the RAPL package counter, CPU runs only.
        self.rapl_samples: list[tuple[float, float]] = []
        self.rapl: dict[str, Any] | None = None

    def watch_rss(self, pid: int) -> None:
        """Track a child's peak *anonymous* resident set, for CPU runs.

        Anonymous, not total. RssFile counts pages backed by a mapped file,
        and OpenNN's CSV reader mmaps its input rather than copying it -- so
        total RSS charges it ~150 MiB for a 151 MB file it never allocated,
        while an engine that reads the same file into heap is charged the same
        amount for memory the kernel cannot reclaim. Counting mapped pages
        penalises the cheaper strategy.

        Measured on the same 500k-row cell: by total RSS, OpenNN 428 MiB and
        PyTorch 875; by anonymous, 270 and 563. Same runs, and only the second
        pair answers "how much memory does this demand".

        RssAnon has no kernel-maintained high-water mark, so unlike VmHWM this
        has to be sampled -- hence polling rather than one read at the end.
        """
        try:
            with open(f"/proc/{pid}/status") as handle:
                for line in handle:
                    if line.startswith("RssAnon:"):
                        self.peak_rss_mib = max(self.peak_rss_mib,
                                                float(line.split()[1]) / 1024.0)
                    elif line.startswith("RssFile:"):
                        self.peak_file_mib = max(self.peak_file_mib,
                                                 float(line.split()[1]) / 1024.0)
        except (OSError, ValueError, IndexError):
            pass

    def __enter__(self) -> "Monitor":
        # On CPU there is nothing on the card worth sampling: the GPU sits
        # idle, and reporting its memory and draw as the run's would be a
        # measurement of the wrong device rather than a missing one. The CPU's
        # own package counter is worth sampling, and is where its energy
        # figure comes from.
        if self.device != "cuda":
            self.rapl = rapl_domain()
            if self.rapl:
                self._thread = threading.Thread(target=self._rapl_loop, daemon=True)
                self._thread.start()
            return self

        if self._measure_idle:
            self.idle_mib, self.idle_watts = self._read_once()

        # The driver's own 20 ms power samples, drained in-process. See Nvml
        # for why nothing nvidia-smi prints can replace them.
        try:
            self._nvml = Nvml()
        except OSError:
            self._nvml = None

        if self._nvml:
            self.power_metric = "nvml_power_samples"
            self._counter_start_mj = self._nvml.energy_mj()
            self._thread = threading.Thread(target=self._nvml_loop, daemon=True)
            self._thread.start()
            return self

        # Without libnvidia-ml: one long-lived nvidia-smi streaming with -lms,
        # not a subprocess per sample -- spawning one costs 20-50 ms. What it
        # streams is the one-second average, so a figure from this path is
        # only meaningful for a window many seconds long, and the artifact
        # names the path so the two are never read as the same instrument.
        self.power_metric = "nvidia_smi_power_draw_1s_average"
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
        if self._nvml:
            end_mj = self._nvml.energy_mj()
            if self._counter_start_mj is not None and end_mj is not None:
                self.run_energy_joules = (end_mj - self._counter_start_mj) / 1e3
            self._nvml.close()
            self._nvml = None

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

    def _nvml_loop(self) -> None:
        """Memory at the interval; power as the driver sampled it.

        The ring holds about 2.4 s, so draining it at the memory interval
        leaves two orders of magnitude of slack. Timestamps are the driver's,
        not the drain's: a sample is filed at the moment it was taken.
        """
        assert self._nvml
        newest_us = int(time.time() * 1e6)

        def drain() -> None:
            nonlocal newest_us
            fresh = self._nvml.power_samples(newest_us)
            if fresh:
                newest_us = fresh[-1][0]
                self.power_samples.extend((us / 1e6, watts) for us, watts in fresh)

        last_drain = time.monotonic()
        while not self._stop.wait(self.interval_ms / 1000.0):
            mib = self._nvml.memory_used_mib()
            if mib is not None:
                self.memory_samples.append((time.time(), mib))
            if time.monotonic() - last_drain >= self.power_drain_ms / 1000.0:
                drain()
                last_drain = time.monotonic()
        drain()

    def _rapl_loop(self) -> None:
        """Accumulate the package energy counter into monotonic joules.

        RAPL reports energy consumed, not power, and the register is narrow
        enough to wrap during a run -- `max_energy_range_uj` is about 65 kJ on
        this part, which at a 60 W package is roughly 18 minutes, well inside a
        transformer cell. Each step is therefore taken modulo the wrap point
        before it is accumulated, so a wrap reads as a small positive step
        rather than as a large negative one that would silently cancel out
        most of the run's energy.
        """
        assert self.rapl
        wrap = self.rapl["wrap_uj"]
        previous = read_rapl_uj(self.rapl)
        total_uj = 0

        if previous is None:
            return

        self.rapl_samples.append((time.time(), 0.0))

        while not self._stop.wait(self.interval_ms / 1000.0):
            current = read_rapl_uj(self.rapl)
            if current is None:
                continue
            total_uj += (current - previous) % (wrap + 1)
            previous = current
            self.rapl_samples.append((time.time(), total_uj / 1e6))

    def rapl_joules(self, start: float | None = None,
                    end: float | None = None) -> float | None:
        """Package energy over [start, end], by difference of the counter.

        A counter is differenced, not integrated: the trapezoid rule the GPU
        path uses is for power samples, and applying it to cumulative energy
        would be wrong. Endpoints are interpolated linearly so the window is
        the engine's own timed region rather than the nearest sample to it.
        """
        if len(self.rapl_samples) < 2:
            return None

        def at(when: float | None, default_index: int) -> float:
            if when is None:
                return self.rapl_samples[default_index][1]
            if when <= self.rapl_samples[0][0]:
                return self.rapl_samples[0][1]
            if when >= self.rapl_samples[-1][0]:
                return self.rapl_samples[-1][1]
            for i in range(len(self.rapl_samples) - 1):
                (t0, j0), (t1, j1) = self.rapl_samples[i], self.rapl_samples[i + 1]
                if t0 <= when <= t1:
                    span = t1 - t0
                    return j0 if span <= 0 else j0 + (j1 - j0) * (when - t0) / span
            return self.rapl_samples[-1][1]

        return max(at(end, -1) - at(start, 0), 0.0)

    def rapl_window_samples(self, start: float | None, end: float | None) -> int:
        return len([t for t, _ in self.rapl_samples
                    if (start is None or t >= start) and (end is None or t <= end)])

    def _loop(self) -> None:
        assert self._process and self._process.stdout
        for line in self._process.stdout:
            if self._stop.is_set():
                break
            try:
                mib, watts = (float(x) for x in line.split(","))
            except ValueError:
                continue
            now = time.time()
            self.memory_samples.append((now, mib))
            self.power_samples.append((now, watts))

    @property
    def peak_mib(self) -> float:
        """Peak device-used memory over the run, minus the idle reading.

        Whole-device on purpose: it counts the CUDA context and the
        allocator's cached blocks, because that memory really is unavailable
        to anything else. `torch.cuda.max_memory_allocated()` excludes both
        and has no OpenNN equivalent, so it never appears in a comparison.
        """
        if not self.memory_samples:
            return 0.0
        return max(mib for _, mib in self.memory_samples) - self.idle_mib

    def window_power(self, start: float | None, end: float | None) -> list[float]:
        return [w for t, w in self.power_samples
                if (start is None or t >= start) and (end is None or t <= end)]

    def energy_joules(self, start: float | None = None, end: float | None = None) -> float:
        """Board energy over [start, end], trapezoid-integrated.

        The window is the engine's own timed region, taken from the marks it
        prints, so warmup and data loading are outside it. The samples inside
        the window are integrated as they are; the partial interval at each
        edge is integrated too, with the power at the edge interpolated
        between the two samples that straddle it, so the figure covers the
        window and not the nearest samples to it. Idle draw is not
        subtracted: the question is what the run costs to perform, and the
        card being on is part of that.
        """
        samples = self.power_samples
        if len(samples) < 2:
            return 0.0
        lo = samples[0][0] if start is None else max(start, samples[0][0])
        hi = samples[-1][0] if end is None else min(end, samples[-1][0])
        if hi <= lo:
            return 0.0

        times = [t for t, _ in samples]

        def power_at(when: float) -> float:
            i = bisect.bisect_left(times, when)
            if i == 0:
                return samples[0][1]
            if i >= len(samples):
                return samples[-1][1]
            (t0, w0), (t1, w1) = samples[i - 1], samples[i]
            return w0 if t1 <= t0 else w0 + (w1 - w0) * (when - t0) / (t1 - t0)

        points = ([(lo, power_at(lo))]
                  + [(t, w) for t, w in samples if lo < t < hi]
                  + [(hi, power_at(hi))])
        return sum((t1 - t0) * (w1 + w0) / 2.0
                   for (t0, w0), (t1, w1) in zip(points, points[1:]))

    def summary(self, start: float | None = None, end: float | None = None) -> dict[str, Any]:
        if self.device != "cuda":
            # Peak RSS is the CPU counterpart of device-used memory. Energy is
            # the RAPL package counter over the same timed window the GPU path
            # integrates board power over -- never the GPU's draw, which during
            # a CPU run is an idle card.
            window_samples = self.rapl_window_samples(start, end)
            measurable = self.rapl is not None and window_samples >= self.MIN_WINDOW_SAMPLES
            joules = self.rapl_joules(start, end) if measurable else None

            if self.rapl is None:
                note = ("CPU run: no readable RAPL counter. energy_uj is root-only "
                        "after CVE-2020-8694; grant read access to measure it")
            elif not measurable:
                note = (f"CPU run: timed window held {window_samples} RAPL samples, "
                        f"fewer than the {self.MIN_WINDOW_SAMPLES} required")
            else:
                note = None

            return {
                "peak_mib": round(self.peak_rss_mib, 1),
                "peak_file_backed_mib": round(self.peak_file_mib, 1),
                "memory_metric": HOST_MEMORY_METRIC,
                "energy_joules": round(joules, 4) if joules is not None else None,
                "energy_wh": round(joules / 3600.0, 6) if joules is not None else None,
                "energy_measurable": bool(measurable),
                "energy_note": note,
                # Package, not core: the uncore is part of what a run costs.
                # Recorded per run so a CPU figure is never silently compared
                # against the GPU's whole-board one.
                "energy_domain": self.rapl["name"] if self.rapl else None,
                "energy_metric": "rapl_package_energy" if self.rapl else None,
                "samples": len(self.rapl_samples),
                "window_samples": window_samples,
            }

        window = self.window_power(start, end)

        # A window too short to sample has no energy figure, and saying 0.0 Wh
        # would be a claim rather than an absence. Lengthen the run -- more
        # epochs or repeats -- if the energy cell matters.
        measurable = len(window) >= self.MIN_WINDOW_SAMPLES
        joules = self.energy_joules(start, end) if measurable else None

        notes = []
        if not measurable:
            notes.append(f"timed window held {len(window)} power samples, fewer than "
                         f"the {self.MIN_WINDOW_SAMPLES} required at the driver's 20 ms")
        if self.power_metric != "nvml_power_samples":
            notes.append("power is nvidia-smi's one-second average: libnvidia-ml "
                         "could not be loaded")

        return {
            "peak_mib": round(self.peak_mib, 1),
            "memory_metric": "device_used_minus_idle",
            "idle_mib": round(self.idle_mib, 1),
            "idle_watts": round(self.idle_watts, 1),
            "energy_joules": round(joules, 2) if measurable else None,
            "energy_wh": round(joules / 3600.0, 5) if measurable else None,
            "mean_watts": round(sum(window) / len(window), 1) if window else None,
            "samples": len(self.power_samples),
            "window_samples": len(window),
            "energy_measurable": measurable,
            "energy_note": "; ".join(notes) or None,
            # Board, and which reading of it: the driver's own 20 ms samples
            # or nvidia-smi's one-second average. They are not the same
            # instrument below a window of many seconds, and the artifact
            # says which one a figure came from.
            "energy_domain": "board",
            "energy_metric": self.power_metric,
            "run_energy_joules": (round(self.run_energy_joules, 1)
                                  if self.run_energy_joules is not None else None),
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
