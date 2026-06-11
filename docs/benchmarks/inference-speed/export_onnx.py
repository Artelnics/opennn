# Build the inference-speed MLP and export it to model_inference.onnx so the
# ONNX Runtime benchmark has a model to load. ONNX Runtime is an inference
# engine: it cannot define or train a model, only run one exported from
# elsewhere. We export the same 2-layer MLP (F -> F -> 1, tanh then linear) the
# other three frameworks build, with a dynamic batch axis.
#
#   usage:  python export_onnx.py <features> [out_path]

import sys

import torch


def main():
    features = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    out_path = sys.argv[2] if len(sys.argv) > 2 else "model_inference.onnx"

    torch.manual_seed(42)
    model = torch.nn.Sequential(
        torch.nn.Linear(features, features),
        torch.nn.Tanh(),
        torch.nn.Linear(features, 1),
    )
    model.eval()

    dummy = torch.zeros(1, features, dtype=torch.float32)
    torch.onnx.export(
        model,
        dummy,
        out_path,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        opset_version=17,
    )
    print(f"wrote {out_path} (features={features})")


if __name__ == "__main__":
    main()
