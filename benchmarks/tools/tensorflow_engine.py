# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Artificial Intelligence Techniques, S.L. and Roberto López González

"""Shared TensorFlow runtime for the `*_tf.py` family drivers.

The TensorFlow drivers run each family the way the Keras documentation runs a
model: `model.compile()` with its defaults and `model.fit()` for training, and
the same compiled predict step Keras builds for `model.predict()` for
inference. Nothing is tuned per workload. Keras's own default decides whether
XLA is used (`jit_compile="auto"`): on for a GPU unless a layer opts out -- the
LSTM opts out to reach cuDNN -- and off on a CPU-only machine.

What is set here, and why each is not a performance choice:

- `TF_FORCE_GPU_ALLOW_GROWTH=true`. By default TensorFlow reserves nearly the
  whole card at start-up, so the protocol's whole-device memory reading would
  report ~15 GiB for every cell. Growth lets the reading show what the
  workload uses. It does not change the kernels that run.
- The GPU is hidden for `cpu` cells, so TensorFlow behaves as on a CPU-only
  machine instead of silently placing work on the card.
- cuSOLVER is preloaded from the NVIDIA pip wheels when TensorFlow's own
  search misses it (TF 2.21 does), otherwise TensorFlow refuses the GPU.
- `strict` disables TF32, as it does for the other engines; `fp32` keeps the
  TensorFlow default, which allows TF32 like the other engines' `fp32`.
- `bf16` uses Keras's `mixed_bfloat16` policy for training (the counterpart of
  autocast) and the pure `bfloat16` policy for inference (the counterpart of
  PT_INFER_CAST=weights); TF_INFER_CAST=autocast keeps mixed for inference.

Call `start()` before anything imports TensorFlow.
"""

from __future__ import annotations

import ctypes
import importlib.util
import os
import sys
import time
from pathlib import Path

from common import report_anonymous_baseline  # noqa: E402 -- tools/ is on the path

SEED = 42
INFER_CAST = os.environ.get("TF_INFER_CAST", "weights")

tf = None
keras = None


def _preload_cusolver() -> str:
    """Load cuSOLVER by absolute path so TensorFlow's later dlopen by soname finds it.

    TF 2.21 locates cuBLAS, cuDNN, cuFFT and the rest of the NVIDIA wheels by
    itself but not cuSOLVER, and without it registers no GPU at all. A library
    already loaded under its soname satisfies the later lookup.
    """
    spec = importlib.util.find_spec("nvidia")
    if spec is None or not spec.submodule_search_locations:
        return "none"

    for root in spec.submodule_search_locations:
        candidate = Path(root) / "cusolver" / "lib" / "libcusolver.so.11"
        if candidate.exists():
            ctypes.CDLL(str(candidate), mode=ctypes.RTLD_GLOBAL)
            return candidate.name
    return "none"


def start(device: str, precision: str):
    """Configure the process for `device`/`precision`, import TensorFlow, return (tf, keras)."""
    global tf, keras

    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")
    preloaded = "none"

    if device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    else:
        os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
        # The PTX TensorFlow ships for GPUs it has no binaries for is compiled
        # by the driver once and kept in its cache; a cache too small to hold
        # it recompiles at every start. Startup only -- never in a timed window.
        os.environ.setdefault("CUDA_CACHE_MAXSIZE", str(4 << 30))
        preloaded = _preload_cusolver()

    import tensorflow as _tf
    import keras as _keras

    tf, keras = _tf, _keras

    if device == "cuda":
        gpus = tf.config.list_physical_devices("GPU")
        if not gpus:
            raise SystemExit("cuda requested but TensorFlow registers no GPU")
        details = tf.config.experimental.get_device_details(gpus[0])
        capability = details.get("compute_capability")
        build = tf.sysconfig.get_build_info()
        shipped = build.get("cuda_compute_capabilities", [])
        native = capability is not None and f"sm_{capability[0]}{capability[1]}" in shipped
        print(f"tf_gpu={details.get('device_name', 'unknown').replace(' ', '_')}")
        print(f"tf_native_kernels={'yes' if native else 'no'}"
              f" tf_built_caps={','.join(shipped)}")

    # The runner sets a matched thread count for every engine on CPU cells;
    # honour it, or the comparison becomes a thread-count comparison.
    threads = (os.environ.get("TF_NUM_INTRAOP_THREADS")
               or os.environ.get("OMP_NUM_THREADS"))
    if threads:
        tf.config.threading.set_intra_op_parallelism_threads(int(threads))

    if precision == "strict":
        tf.config.experimental.enable_tensor_float_32_execution(False)

    print(f"tf_version={tf.__version__} keras_version={keras.__version__}")
    print(f"tf32={'on' if tf.config.experimental.tensor_float_32_execution_enabled() else 'off'}"
          f" memory_growth={os.environ.get('TF_FORCE_GPU_ALLOW_GROWTH', 'default')}"
          f" preloaded={preloaded}", flush=True)
    return tf, keras


def set_policy(precision: str, mode: str) -> str:
    """Keras dtype policy for a cell. Returns the policy name."""
    if precision != "bf16":
        name = "float32"
    elif mode == "infer" and INFER_CAST == "weights":
        name = "bfloat16"
    else:
        name = "mixed_bfloat16"
    keras.mixed_precision.set_dtype_policy(name)
    return name


def compute_dtype() -> str:
    return keras.mixed_precision.dtype_policy().compute_dtype


def device_name(device: str) -> str:
    return "/GPU:0" if device == "cuda" else "/CPU:0"


def report_blas() -> None:
    """CPU BLAS TensorFlow dispatches to, printed like the other engines print theirs."""
    try:
        from tensorflow.python.util import _pywrap_util_port

        onednn = _pywrap_util_port.IsMklEnabled()
    except Exception:
        onednn = False
    print(f"blas={'onednn' if onednn else 'eigen'}")


def report_baselines() -> None:
    """Total and anonymous resident memory, TensorFlow imported and configured
    but no data loaded and no model built -- the point every engine reports at."""
    print(f"baseline_rss_mib={resident_mib():.1f}")
    report_anonymous_baseline()


def resident_mib() -> float:
    """Process resident memory in MiB from Linux procfs; zero when unavailable."""
    try:
        with open("/proc/self/statm") as handle:
            return int(handle.read().split()[1]) * os.sysconf("SC_PAGE_SIZE") / (1024.0 * 1024.0)
    except Exception:
        return 0.0


def sync() -> None:
    """Wait for every queued device operation, so a timestamp marks completed work."""
    tf.test.experimental.sync_devices()


def trainable_parameters(model) -> int:
    """Trainable values -- what the other engines count. Batch-norm running
    statistics are non-trainable in Keras and buffers in PyTorch: neither counts."""
    total = 0
    for weight in model.trainable_weights:
        count = 1
        for dimension in weight.shape:
            count *= int(dimension)
        total += count
    return total


def seed() -> None:
    keras.utils.set_random_seed(SEED)


def execution(model) -> str:
    """How Keras chose to run the compiled model: its own `jit_compile="auto"` decision."""
    return "xla" if model.jit_compile else "graph"


def predict_step(model):
    """The step `model.predict()` compiles, without gathering every output on the host.

    `predict()` copies all outputs back and concatenates them. The other
    engines time the forward pass and discard its result, and for the
    transformer that copy would be a 666 MB transfer per batch, so the
    comparison times the same compiled step without it.
    """
    @tf.function(jit_compile=model.jit_compile, reduce_retracing=True)
    def step(*inputs):
        return model(list(inputs) if len(inputs) > 1 else inputs[0], training=False)

    return step


def is_oom(error: BaseException) -> bool:
    text = str(error)
    return (isinstance(error, tf.errors.ResourceExhaustedError)
            or "RESOURCE_EXHAUSTED" in text or "Out of memory" in text
            or "OOM when allocating" in text)


def epoch_timer(warmup: int):
    """A Keras callback timing each epoch after `warmup` untimed ones.

    It marks the window the runner integrates energy over, and synchronizes
    the device at both edges of every timed epoch, as the other drivers do.
    """

    class EpochTimer(keras.callbacks.Callback):
        def __init__(self):
            super().__init__()
            self.times: list[float] = []
            self._mark = 0.0

        def on_epoch_begin(self, epoch, logs=None):
            if epoch < warmup:
                return
            sync()
            if epoch == warmup:
                print(f"TIMED_START_UNIX={time.time():.3f}", flush=True)
            self._mark = time.perf_counter()

        def on_epoch_end(self, epoch, logs=None):
            if epoch < warmup:
                return
            sync()
            self.times.append(time.perf_counter() - self._mark)

        def on_train_end(self, logs=None):
            print(f"TIMED_END_UNIX={time.time():.3f}", flush=True)

    return EpochTimer()


def batches_of(text: str) -> list[int]:
    return [int(part) for part in text.split(",") if part]


def report_opened(path: str, role: str | None = None) -> None:
    """Announce the file actually opened, not the one passed in."""
    key = f"dataset_{role}" if role else "dataset_opened"
    print(f"{key}={Path(path).resolve()}", flush=True)


def oom_exit(error: BaseException) -> int:
    print(f"fits=0\nreason={str(error).splitlines()[0] if str(error) else type(error).__name__}"
          "\nRESULT=OOM", flush=True)
    return 1


def usage(doc: str | None) -> int:
    print(doc, file=sys.stderr)
    return 2
