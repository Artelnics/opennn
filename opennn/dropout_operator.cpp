//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   D R O P O U T   O P E R A T O R   S O U R C E
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "dropout_operator.h"
#include "json.h"
#include "tensor_operations.h"
#include "forward_propagation.h"
#include "back_propagation.h"

namespace opennn
{

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
        add_json_field(w, "DropoutRate", to_string(rate));
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
