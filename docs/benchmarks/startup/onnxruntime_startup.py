# ONNX Runtime startup-latency benchmark: import onnxruntime, load a small MLP
# (10 -> 64 -> 1) from an .onnx file, run one inference, print the result, exit.
# ONNX Runtime is an inference engine: the model is loaded from disk, not built
# or trained here (it cannot train). Generate model_startup.onnx once beforehand.

import numpy as np
import onnxruntime as ort

session = ort.InferenceSession("model_startup.onnx", providers=["CPUExecutionProvider"])
x = np.ones((1, 10), dtype=np.float32)
output = session.run(None, {session.get_inputs()[0].name: x})

print("prediction", float(output[0][0, 0]))
