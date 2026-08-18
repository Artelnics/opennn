//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   D R O P O U T   O P E R A T O R   S O U R C E
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "opennn/neural_network/operators/dropout_operator.h"
#include "opennn/core/json.h"
#include "opennn/core/random_utilities.h"
#include "opennn/core/tensor_operations.h"
#include "opennn/neural_network/forward_propagation.h"
#include "opennn/neural_network/back_propagation.h"
#ifdef OPENNN_HAS_CUDA
#include "opennn/core/cuda/kernel_activation.cuh"
#endif

namespace opennn
{

// Defined below: against the CUDA kernels, or as throwing stubs.
static void dropout_forward_gpu(TensorView&, Buffer&, float);
static void dropout_backward_gpu(TensorView&, const Buffer&, float);

static void dropout_forward_cpu(TensorView& output, Buffer& mask, float rate)
{
    const Index element_count = output.size();
    mask.resize_bytes(element_count * Index(sizeof(float)), Device::CPU);
    if (element_count == 0) return;

    const float keep_scale = 1.0f / (1.0f - rate);
    float* output_data = output.as<float>();
    VectorMap mask_values = mask.as_vector();

    set_random_uniform(mask_values, 0.0f, 1.0f);

    const bool parallel = element_count >= 65536;

    #pragma omp parallel for schedule(static) if(parallel)
    for (Index i = 0; i < element_count; ++i)
    {
        const float keep_value = mask_values(i) < rate ? 0.0f : keep_scale;
        mask_values(i) = keep_value;
        output_data[i] *= keep_value;
    }
}

void dropout_forward(TensorView& output, Buffer& mask, float rate)
{
    if (rate <= 0.0f) return;
    if (output.is_cuda()) { dropout_forward_gpu(output, mask, rate); return; }
    dropout_forward_cpu(output, mask, rate);
}

void dropout_backward(TensorView& delta, const Buffer& mask, float rate)
{
    if (rate <= 0.0f) return;
    if (delta.is_cuda()) { dropout_backward_gpu(delta, mask, rate); return; }
    delta.as_vector().array() *= mask.as_vector().array();
}

#ifdef OPENNN_HAS_CUDA

static void dropout_forward_gpu(TensorView& output, Buffer& mask, float rate)
{
    const Index element_count = output.size();
    if (mask.get_device() != Device::CUDA || mask.byte_size() < element_count)
        mask.resize_bytes(element_count, Device::CUDA);

    const unsigned long long seed = static_cast<unsigned long long>(random_integer(0, 1 << 30));

    output.dispatch([&]<typename T>()
    {
        dropout_forward_cuda<T>(element_count, output.as<T>(), mask.as<uint8_t>(), rate, seed);
    });
}

static void dropout_backward_gpu(TensorView& delta, const Buffer& mask, float rate)
{
    delta.dispatch([&]<typename T>()
    {
        dropout_backward_cuda<T>(delta.size(), delta.as<T>(), delta.as<T>(), mask.as<uint8_t>(), rate);
    });
}

#else

OPENNN_CUDA_STUB(void, dropout_forward_gpu, (TensorView&, Buffer&, float))
OPENNN_CUDA_STUB(void, dropout_backward_gpu, (TensorView&, const Buffer&, float))

#endif

void DropoutOperator::set_rate(float new_rate)
{
    throw_if(new_rate < 0.0f || new_rate >= 1.0f,
             "Dropout rate must be in [0, 1).");

    rate = new_rate;
}

void DropoutOperator::forward_propagate(ForwardPropagation& forward_propagation, size_t layer, bool is_training)
{
    if (!is_training || !active()) return;

    TensorView& output = get_output(forward_propagation, layer);
    dropout_forward(output, mask, rate);
}

void DropoutOperator::back_propagate(ForwardPropagation&, BackPropagation& back_propagation, size_t layer) const
{
    if (!active()) return;
    dropout_backward(get_output_delta(back_propagation, layer), mask, rate);
}

void DropoutOperator::to_JSON(JsonWriter& w) const
{
    if (rate > 0.0f)
        add_json_field(w, "DropoutRate", rate);
}

void DropoutOperator::from_JSON(const Json* parent)
{
    if (parent && parent->has("DropoutRate"))
        set_rate(read_json_float(parent, "DropoutRate"));
}

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
