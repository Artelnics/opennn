//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   A C T I V A T I O N   O P E R A T O R   S O U R C E
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "activation_operator.h"
#include "device_backend.h"
#include "json.h"
#include "random_utilities.h"
#include "tensor_operations.h"
#include "string_utilities.h"
#include "forward_propagation.h"
#include "back_propagation.h"
#include "profiler.h"

namespace opennn
{

#ifdef OPENNN_HAS_CUDA

namespace
{

void configure_activation_descriptor(cudnnActivationDescriptor_t& descriptor,
                                     ActivationOp::Function function)
{
    if (!descriptor) CHECK_CUDNN(cudnnCreateActivationDescriptor(&descriptor));
    CHECK_CUDNN(cudnnSetActivationDescriptor(descriptor,
                                             ActivationOp::to_cudnn_mode(function),
                                             CUDNN_PROPAGATE_NAN,
                                             0.0));
}

void destroy_activation_descriptor(cudnnActivationDescriptor_t& descriptor)
{
    if (!descriptor) return;

    cudnnDestroyActivationDescriptor(descriptor);
    descriptor = nullptr;
}

}

#else

namespace
{

void configure_activation_descriptor(cudnnActivationDescriptor_t&, ActivationOp::Function)
{
}

void destroy_activation_descriptor(cudnnActivationDescriptor_t&)
{
}

}

#endif


cudnnActivationMode_t ActivationOp::to_cudnn_mode(Function function)
{
    using enum Function;
    switch (function)
    {
    case Sigmoid: return CUDNN_ACTIVATION_SIGMOID;
    case Tanh:    return CUDNN_ACTIVATION_TANH;
    case ReLU:    return CUDNN_ACTIVATION_RELU;
    case Identity:
    case Softmax: return CUDNN_ACTIVATION_IDENTITY;
    }

    return CUDNN_ACTIVATION_IDENTITY;
}

void ActivationOp::set_function(Function new_function)
{
    function = new_function;
    configure_activation_descriptor(descriptor, function);
}

void ActivationOp::set_function(const string& name)
{
    set_function(from_string(name));
}

void ActivationOp::forward_propagate(ForwardPropagation& fp, size_t layer, bool /*is_training*/)
{
    PROFILE_SCOPE("op:activation_fwd");
    TensorView& output = get_output(fp, layer);
    if (output.empty()) return;

    if (forward_fused && output.is_cuda()) return;

    if (!input_slots.empty() && input_slots[0] != output_slots[0])
        copy(get_input(fp, layer), output);

    activation_forward(output, function);
}

void ActivationOp::apply_delta(const TensorView& outputs, TensorView& delta) const
{
    activation_backward(outputs, delta, function);
}

void ActivationOp::back_propagate(ForwardPropagation& fp, BackPropagation& bp, size_t layer) const
{
    PROFILE_SCOPE("op:activation_bwd");
    if (backward_fused && get_output_delta(bp, layer).is_cuda()) return;

    const auto& slots = output_slots_backward.empty() ? output_slots : output_slots_backward;
    const TensorView& outputs = fp.forward_slots[layer][slots[0]];

    const bool standalone = !input_slots.empty() && input_slots[0] != output_slots[0];
    if (standalone)
    {
        const TensorView& output_delta = get_output_delta(bp, layer);
        TensorView& input_delta        = get_input_delta(bp, layer);
        if (input_delta.empty()) return;
        copy(output_delta, input_delta);
        apply_delta(outputs, input_delta);
    }
    else
    {
        TensorView& delta = get_output_delta(bp, layer);
        apply_delta(outputs, delta);
    }
}

void ActivationOp::destroy_cuda()
{
    destroy_activation_descriptor(descriptor);
}

void ActivationOp::to_JSON(JsonWriter& w) const
{
    add_json_field(w, "Activation", ActivationOp::to_string(function));
}

void ActivationOp::from_JSON(const Json* parent)
{
    if (parent && parent->has("Activation"))
        set_function(read_json_string(parent, "Activation"));
}

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
