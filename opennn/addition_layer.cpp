//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   A D D I T I O N   L A Y E R   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "registry.h"
#include "addition_layer.h"
#include "math_utilities.h"
#include "forward_propagation.h"
#include "back_propagation.h"

namespace opennn
{

Addition::Addition(const Shape& new_input_shape, const string& new_name, Index new_inputs_number)
    : Layer(LayerType::Addition)
{
    operators = {&add};

    set(new_input_shape, new_name, new_inputs_number);
}

vector<TensorSpec> Addition::get_backward_specs(Index batch_size) const
{
    return vector<TensorSpec>(inputs_number, {Shape{batch_size}.append(input_shape), compute_dtype});
}

void Addition::set(const Shape& new_input_shape, const string& new_label, Index new_inputs_number)
{
    check_rank(new_input_shape, {2, 3}, "Addition", "input");

    if (new_inputs_number < 2)
        throw runtime_error("Addition: inputs_number must be >= 2.");

    input_shape = new_input_shape;
    inputs_number = new_inputs_number;

    // Output is at slot inputs_number (after all inputs). Delta slots:
    //   slot 0           = output delta (incoming from downstream)
    //   slots 1..N       = input deltas (one per source)
    vector<size_t> input_delta_slots(inputs_number);
    for (Index i = 0; i < inputs_number; ++i)
        input_delta_slots[size_t(i)] = size_t(i + 1);
    add.input_delta_slots = input_delta_slots;

    set_label(new_label);
}

REGISTER(Layer, Addition, "Addition")

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
