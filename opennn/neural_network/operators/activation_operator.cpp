//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   A C T I V A T I O N   O P E R A T O R   S O U R C E
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "opennn/neural_network/operators/activation_operator.h"
#include "opennn/core/json.h"
#include "opennn/core/tensor_operations.h"
#include "opennn/neural_network/forward_propagation.h"
#include "opennn/neural_network/back_propagation.h"
#include "opennn/core/profiler.h"

namespace opennn
{

void ActivationOperator::forward_propagate(ForwardPropagation& forward_propagation, size_t layer, ForwardPropagationMode)
{
    PROFILE_SCOPE("op:activation_fwd");

    TensorView& output = get_output(forward_propagation, layer);

    if (output.empty() || forward_fused)
        return;

    if (input_slots.empty() || input_slots[0] != output_slots[0])
        copy(get_input(forward_propagation, layer), output);

    activation_forward(output, activation_function);

    if (save_slot != SIZE_MAX)
    {

        TensorView& saved = forward_propagation.slots[layer][save_slot];
        if (!saved.empty()) copy(output, saved);
    }
}

void ActivationOperator::back_propagate(ForwardPropagation& forward_propagation, BackPropagation& back_propagation, size_t layer) const
{
    PROFILE_SCOPE("op:activation_bwd");

    TensorView& output_delta = get_output_delta(back_propagation, layer);

    const bool fused_by_consumer = backward_fused_by_consumer
        && forward_propagation.drelu_fused_by_layer[layer] != 0;
    if ((backward_fused || fused_by_consumer) && output_delta.is_cuda())
        return;

    const bool needs_input = activation_needs_input(activation_function);
    const size_t read_slot = (save_slot != SIZE_MAX) ? save_slot : output_slots[0];

    const TensorView& outputs = needs_input
        ? get_input(forward_propagation, layer)
        : forward_propagation.slots[layer][read_slot];

    if (!input_slots.empty() && input_slots[0] != output_slots[0])
    {
        TensorView& input_delta = get_input_delta(back_propagation, layer);
        if (input_delta.empty()) return;
        copy(output_delta, input_delta);
        activation_backward(outputs, input_delta, activation_function);
    }
    else
    {
        activation_backward(outputs, output_delta, activation_function);
    }
}

void ActivationOperator::to_JSON(JsonWriter& w) const
{
    add_json_field(w, "Activation", ActivationOperator::to_string(activation_function));
}

void ActivationOperator::from_JSON(const Json* parent)
{
    if (parent && parent->has("Activation"))
        set_activation_function(read_json_string(parent, "Activation"));
}

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
