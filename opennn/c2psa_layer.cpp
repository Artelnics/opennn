//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   C 2 P S A   L A Y E R   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "registry.h"
#include "c2psa_layer.h"
#include "json.h"

namespace opennn
{

C2PSA::C2PSA(const Shape& new_input_shape, const string& new_label)
    : Layer(LayerType::C2PSA, true)
{
    operators = {&c2psa};
    set(new_input_shape, new_label);
}

Shape C2PSA::get_output_shape() const
{
    return input_shape;  // C2PSA preserves spatial shape and channel count
}

void C2PSA::set(const Shape& new_input_shape, const string& new_label)
{
    if (!new_input_shape.empty())
        check_rank(new_input_shape, {3}, "C2PSA", "input");
    input_shape = new_input_shape;
    set_label(new_label);
    configure_operator();
}

void C2PSA::set_input_shape(const Shape& new_input_shape)
{
    check_rank(new_input_shape, {3}, "C2PSA", "input");
    input_shape = new_input_shape;
    configure_operator();
}

void C2PSA::configure_operator()
{
    if (input_shape.empty() || input_shape[2] < 2) return;
    c2psa.set(input_shape[0], input_shape[1], input_shape[2]);
}

vector<TensorSpec> C2PSA::get_forward_specs(Index batch_size) const
{
    if (input_shape.empty()) return {};
    const Index tokens = input_shape[0] * input_shape[1];
    const Index C      = input_shape[2];
    const Index half_c = C / 2;
    return {
        {{batch_size, tokens, half_c}, compute_dtype},                          // x_attn_flat  [slot 1]
        {{batch_size, tokens, half_c}, compute_dtype},                          // Q            [slot 2]
        {{batch_size, tokens, half_c}, compute_dtype},                          // K            [slot 3]
        {{batch_size, tokens, tokens}, compute_dtype},                          // attn_weights [slot 4]
        {{batch_size, tokens, half_c}, compute_dtype},                          // V            [slot 5]
        {{batch_size, tokens, C},      compute_dtype},                          // concat_result[slot 6]
        {{batch_size, input_shape[0], input_shape[1], C}, compute_dtype},      // output       [slot 7]
    };
}

vector<TensorSpec> C2PSA::get_backward_specs(Index batch_size) const
{
    if (input_shape.empty()) return {};
    return {{Shape{batch_size}.append(get_input_shape()), compute_dtype}};
}

void C2PSA::read_JSON_body(const Json* /*root*/)
{
    // input_shape is set via set_input_shape from the network topology; no extra fields
}

void C2PSA::write_JSON_body(JsonWriter&) const
{
    // No extra fields beyond input_shape (stored in the network JSON)
}

REGISTER(Layer, C2PSA, "C2PSA")

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
