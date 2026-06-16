# Model export to standalone code: OpenNN vs PyTorch vs TensorFlow

The previous notes measured what it costs to *run* a model with each library. This one is about a
different question: once a model is trained, **can you ship it as code that runs with no
machine-learning runtime at all?** On targets where you cannot install a framework — a
microcontroller-class device, a piece of firmware, a foreign codebase in another language — the
ideal artifact is not a model file plus a runtime, but plain source you compile into your
application.

## Contents

- [The result](#the-result)
- [What OpenNN produces (measured)](#what-opennn-produces-measured)
- [What PyTorch offers instead](#what-pytorch-offers-instead)
- [Why it matters](#why-it-matters)
- [Caveats](#caveats)
- [References](#references)

## The result

|  | OpenNN | PyTorch | TensorFlow |
| --- | --- | --- | --- |
| **Exports the trained model as standalone source?** | **yes — C, Python, JavaScript, PHP** | no | no |
| **Runs with no ML runtime?** | **yes** (C export needs only a C compiler) | no — needs libtorch or onnxruntime | no — needs the TF / TFLite runtime |

OpenNN turns a trained network into a self-contained function in your target language. PyTorch's
export paths (TorchScript, ONNX) produce a *model file* that still requires a runtime — `libtorch`
or ONNX Runtime — to execute.

## What OpenNN produces

We trained a small MLP on `sum.csv` (100 inputs → 64 → 1) and called OpenNN's exporter:

```
ModelExpression(&network).save("model.c", ModelExpression::ProgrammingLanguage::C);
```

This emits **`model.c`** — 531 lines, ~231 KB — a complete program with `calculate_outputs()` and a
`main()`. It then compiles and runs with nothing but a C compiler and the C standard library:

```
$ gcc -O2 -o model model.c -lm        # no OpenNN, no libtorch, no Python
$ ldd model
        libm.so.6        libc.so.6        (+ the loader)
$ env -i PATH=/nonexistent ./model     # empty environment
These are your outputs:
variable_101: 11.393875
```

The compiled program is ~236 KB and depends only on `libc`/`libm` — present on every system. There
is no model file to load and no framework to install; the weights are baked into the source. OpenNN
can emit the same model as **Python, JavaScript, or PHP** too (the Python export uses only NumPy;
the C export needs nothing beyond the standard library).

## What PyTorch offers instead

PyTorch has no equivalent "export to compilable source." Its two export paths both yield a
*serialized model* that needs a runtime to run:

- **TorchScript** (`torch.jit.save` → `.pt`): loaded and executed by `libtorch` — the same ~442 MB
  CPU runtime measured in the `CPU size benchmark`.
- **ONNX** (`torch.onnx.export` → `.onnx`): runs under ONNX Runtime or another inference engine,
  which must be installed on the target.

Both are good for portability *between ML runtimes*, but neither lets you drop a trained model into
a C/JS/PHP codebase, or a device with no ML stack, as plain source.

**TensorFlow** is the same: its export formats — SavedModel and TFLite (`.tflite`) — are serialized
models executed by the TensorFlow or TFLite runtime. TFLite targets small devices, but it is still
an interpreter you ship and link, not compilable source with the weights baked in.

## Why it matters

- **No-runtime targets:** firmware, microcontrollers, or sandboxed environments where you cannot
  install a 442 MB runtime can still run the model as compiled C.
- **Embedding in other languages/codebases:** paste the exported function into a C, JavaScript, or
  PHP project — no FFI, no bindings, no service call.
- **Auditability & longevity:** the exported source is human-readable and has no version-pinned
  runtime to rot; it will compile years later with a standard toolchain.
- **Tiny artifacts:** a few hundred KB of source vs. a model file plus a runtime.

## Caveats

- This is a **deployment-format** comparison, not a speed one. The generated code is a
  straightforward implementation of the forward pass; it is meant for portability, not maximum
  throughput.
- The C export covers the layers used by standard feed-forward / scaling / bounding models; very
  large or exotic architectures produce correspondingly large source and are better served by a
  runtime.
- PyTorch's TorchScript/ONNX are the right tools when the target *can* host an inference runtime —
  they preserve a broad operator set. The point here is the no-runtime case.
- Measured on Linux x86_64; the exported C is portable to any platform with a C compiler.

## References

- [OpenNN](https://www.opennn.net/).
- [PyTorch TorchScript documentation](https://pytorch.org/docs/stable/jit.html).
- [PyTorch ONNX export documentation](https://pytorch.org/docs/stable/onnx.html).
- [TensorFlow Lite](https://www.tensorflow.org/lite).
- [ONNX Runtime](https://onnxruntime.ai/).
