//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   S W I G L U   O P E R A T O R   S O U R C E
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "opennn/neural_network/operators/swiglu_operator.h"
#include "opennn/core/tensor_operations.h"
#include "opennn/neural_network/forward_propagation.h"
#include "opennn/neural_network/back_propagation.h"
#ifdef OPENNN_HAS_CUDA
#include "opennn/core/cuda/kernel_activation.cuh"
#endif

namespace opennn
{

#ifdef OPENNN_HAS_CUDA

static void swiglu_forward_gpu(const TensorView& gate, const TensorView& up, TensorView& output)
{
    const int n = to_int(gate.size());
    output.dispatch([&]<typename T>() {
        swiglu_forward_cuda<T>(n, gate.as<T>(), up.as<T>(), output.as<T>());
    });
}

static void swiglu_backward_gpu(const TensorView& output_delta, const TensorView& gate, const TensorView& up,
                                TensorView& gate_delta, TensorView& up_delta)
{
    const int n = to_int(output_delta.size());
    output_delta.dispatch([&]<typename T>() {
        T* gate_delta_data = gate_delta.empty() ? nullptr : gate_delta.as<T>();
        T* up_delta_data   = up_delta.empty()   ? nullptr : up_delta.as<T>();
        swiglu_backward_cuda<T>(n, output_delta.as<T>(), gate.as<T>(), up.as<T>(),
                                gate_delta_data, up_delta_data);
    });
}

#else

OPENNN_CUDA_TEMPLATE_STUB(swiglu_forward_gpu)
OPENNN_CUDA_TEMPLATE_STUB(swiglu_backward_gpu)

#endif

static void swiglu_forward_cpu(const TensorView& gate, const TensorView& up, TensorView& output)
{
    const Index n = gate.size();
    const float* g = gate.as<float>();
    const float* u = up.as<float>();
    float* o       = output.as<float>();

    const bool parallel = n >= 65536;

    #pragma omp parallel for schedule(static) if(parallel)
    for (Index i = 0; i < n; ++i)
    {
        const float gi = g[i];
        const float silu = gi / (1.0f + expf(-gi));
        o[i] = silu * u[i];
    }
}

static void swiglu_backward_cpu(const TensorView& output_delta, const TensorView& gate, const TensorView& up,
                         TensorView& gate_delta, TensorView& up_delta)
{
    const Index n = output_delta.size();
    const float* d = output_delta.as<float>();
    const float* g = gate.as<float>();
    const float* u = up.as<float>();
    float* dg = gate_delta.empty() ? nullptr : gate_delta.as<float>();
    float* du = up_delta.empty()   ? nullptr : up_delta.as<float>();

    const bool parallel = n >= 65536;

    #pragma omp parallel for schedule(static) if(parallel)
    for (Index i = 0; i < n; ++i)
    {
        const float gi  = g[i];
        const float sig = 1.0f / (1.0f + expf(-gi));
        const float silu = gi * sig;
        if (du) du[i] = d[i] * silu;

        if (dg) dg[i] = d[i] * u[i] * sig * (1.0f + gi * (1.0f - sig));
    }
}

void swiglu_forward(const TensorView& gate, const TensorView& up, TensorView& output)
{
    if (gate.is_cuda()) { swiglu_forward_gpu(gate, up, output); return; }
    swiglu_forward_cpu(gate, up, output);
}

void swiglu_backward(const TensorView& output_delta, const TensorView& gate, const TensorView& up,
                     TensorView& gate_delta, TensorView& up_delta)
{
    if (output_delta.is_cuda()) { swiglu_backward_gpu(output_delta, gate, up, gate_delta, up_delta); return; }
    swiglu_backward_cpu(output_delta, gate, up, gate_delta, up_delta);
}

void SwiGLUOperator::forward_propagate(ForwardPropagation& forward_propagation, size_t layer, ForwardPropagationMode  )
{
    const TensorView& gate = get_input(forward_propagation, layer, 0);
    const TensorView& up   = get_input(forward_propagation, layer, 1);
    TensorView& output     = get_output(forward_propagation, layer);

    swiglu_forward(gate, up, output);
}

void SwiGLUOperator::back_propagate(ForwardPropagation& forward_propagation, BackPropagation& back_propagation, size_t layer) const
{
    const TensorView& gate         = get_input(forward_propagation, layer, 0);
    const TensorView& up           = get_input(forward_propagation, layer, 1);
    const TensorView& output_delta = get_output_delta(back_propagation, layer);

    TensorView& gate_delta = get_input_delta(back_propagation, layer, 0);
    TensorView& up_delta   = get_input_delta(back_propagation, layer, 1);

    swiglu_backward(output_delta, gate, up, gate_delta, up_delta);
}

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
