// SPDX-License-Identifier: LGPL-2.1-or-later
// Copyright (C) 2005-2026 Artificial Intelligence, SL.

#include "opennn/network/layers/c2psa_layer.h"
#include "opennn/registry.h"

namespace opennn
{

C2PSA::C2PSA(const Shape& new_input_shape, const string& new_label)
    : Layer(LayerType::C2PSA, Trainability::Trainable)
{
    operators = {&c2psa};
    set(new_input_shape, new_label);
}

void C2PSA::set(const Shape& new_input_shape, const string& new_label)
{
    if (!new_input_shape.empty())
        check_rank(new_input_shape, {3}, "C2PSA", "input");
    input_shape = new_input_shape;
    set_label(new_label);
    configure_operator();
}

void C2PSA::configure_operator()
{
    if (input_shape.empty() || input_shape[2] < 2) return;
    c2psa.set(input_shape[0], input_shape[1], input_shape[2]);
    c2psa.output_slots = {Output};
    c2psa.input_delta_slots = {InputDelta};
    c2psa.forward_scratch_slot = ForwardScratch;
    c2psa.backward_scratch_slot = BackwardScratch;
}

vector<TensorSpec> C2PSA::get_forward_specs(Index batch_size) const
{
    if (input_shape.empty()) return {};
    const Index tokens = input_shape[0] * input_shape[1];
    const Index C      = input_shape[2];
    const Index half_c = C / 2;
    return {
        {{batch_size, tokens, half_c}, compute_dtype},
        {{batch_size, tokens, half_c}, compute_dtype},
        {{batch_size, tokens, half_c}, compute_dtype},
        {{batch_size, tokens, tokens}, compute_dtype},
        {{batch_size, tokens, half_c}, compute_dtype},
        {{batch_size, tokens, C},      compute_dtype},
        {get_compute_device() == Device::CUDA
            ? Shape{batch_size, tokens, half_c}
            : Shape{}, compute_dtype},
        {{batch_size, input_shape[0], input_shape[1], C}, compute_dtype},
    };
}

vector<TensorSpec> C2PSA::get_backward_specs(Index batch_size) const
{
    if (input_shape.empty()) return {};
    const Index tokens = input_shape[0] * input_shape[1];
    const Index channels = input_shape[2];
    const Index half_channels = channels / 2;
    return {
        {Shape{batch_size}.append(get_input_shape()), compute_dtype},
        {get_compute_device() == Device::CUDA
            ? Shape{batch_size, tokens, channels + tokens + 5 * half_channels}
            : Shape{}, compute_dtype},
    };
}

}
