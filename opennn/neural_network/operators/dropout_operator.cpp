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

static void dropout_forward_gpu(TensorView&, TensorView&, float);
static void dropout_backward_gpu(TensorView&, const TensorView&, float);

static void validate_dropout_mask(const TensorView& values,
                                  const TensorView& mask)
{
    throw_if(!mask.is_int8() || mask.get_device() != values.get_device()
             || mask.size() < values.size(),
             "Dropout mask must provide one INT8 value per element on the same device.");
}

static void dropout_forward_cpu(TensorView& output, TensorView& mask, float rate)
{
    const Index element_count = output.size();
    if (element_count == 0) return;

    const float keep_scale = 1.0f / (1.0f - rate);
    float* output_data = output.as<float>();
    uint8_t* mask_values = mask.as<uint8_t>();

    set_random_bernoulli(span<uint8_t>(mask_values, size_t(element_count)),
                         1.0f - rate);

    const bool parallel = element_count >= 65536;

    #pragma omp parallel for schedule(static) if(parallel)
    for (Index i = 0; i < element_count; ++i)
    {
        output_data[i] *= mask_values[i] ? keep_scale : 0.0f;
    }
}

void dropout_forward(TensorView& output, TensorView& mask, float rate)
{
    if (rate <= 0.0f) return;
    validate_dropout_mask(output, mask);
    if (output.is_cuda()) { dropout_forward_gpu(output, mask, rate); return; }
    dropout_forward_cpu(output, mask, rate);
}

void dropout_backward(TensorView& delta, const TensorView& mask, float rate)
{
    if (rate <= 0.0f) return;
    validate_dropout_mask(delta, mask);
    if (delta.is_cuda()) { dropout_backward_gpu(delta, mask, rate); return; }

    const float keep_scale = 1.0f / (1.0f - rate);
    float* delta_values = delta.as<float>();
    const uint8_t* mask_values = mask.as<uint8_t>();
    for (Index i = 0; i < delta.size(); ++i)
        delta_values[i] *= mask_values[i] ? keep_scale : 0.0f;
}

#ifdef OPENNN_HAS_CUDA

static unsigned long long* dropout_seed_state()
{
    static Buffer state = []
    {
        Buffer buffer(Device::CUDA);
        buffer.resize_bytes(Index(sizeof(unsigned long long)), Device::CUDA);

        const unsigned long long initial =
            static_cast<unsigned long long>(random_integer(0, 1 << 30));

        device::copy_async(buffer.data(), &initial, Index(sizeof(initial)),
                           device::CopyKind::HostToDevice,
                           device::get_compute_stream());
        device::synchronize(device::get_compute_stream());

        return buffer;
    }();

    return state.as<unsigned long long>();
}

static void dropout_forward_gpu(TensorView& output, TensorView& mask, float rate)
{
    const Index element_count = output.size();

    unsigned long long* const seed_state = dropout_seed_state();

    advance_dropout_seed_cuda(seed_state);

    output.dispatch([&]<typename T>()
    {
        dropout_forward_cuda<T>(element_count, output.as<T>(), mask.as<uint8_t>(), rate, seed_state);
    });
}

static void dropout_backward_gpu(TensorView& delta, const TensorView& mask, float rate)
{
    delta.dispatch([&]<typename T>()
    {
        dropout_backward_cuda<T>(delta.size(), delta.as<T>(), delta.as<T>(), mask.as<uint8_t>(), rate);
    });
}

#else

OPENNN_CUDA_STUB(void, dropout_forward_gpu, (TensorView&, TensorView&, float))
OPENNN_CUDA_STUB(void, dropout_backward_gpu, (TensorView&, const TensorView&, float))

#endif

void DropoutOperator::set_rate(float new_rate)
{
    throw_if(new_rate < 0.0f || new_rate >= 1.0f,
             "Dropout rate must be in [0, 1).");

    rate = new_rate;
}

void DropoutOperator::forward_propagate(ForwardPropagation& forward_propagation, size_t layer, ForwardPropagationMode pass)
{
    if (!is_training(pass) || !active()) return;

    throw_if(!mask_slot, "DropoutOperator: mask slot was not planned.");
    TensorView& output = get_output(forward_propagation, layer);
    dropout_forward(output, forward_propagation.slots[layer][*mask_slot], rate);
}

void DropoutOperator::back_propagate(ForwardPropagation& forward_propagation,
                                     BackPropagation& back_propagation,
                                     size_t layer) const
{
    if (!active()) return;
    throw_if(!mask_slot, "DropoutOperator: mask slot was not planned.");
    dropout_backward(get_output_delta(back_propagation, layer),
                     forward_propagation.slots[layer][*mask_slot], rate);
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
