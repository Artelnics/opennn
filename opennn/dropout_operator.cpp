//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   D R O P O U T   O P E R A T O R   S O U R C E
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "dropout_operator.h"
#include "json.h"
#include "random_utilities.h"
#include "tensor_operations.h"
#include "string_utilities.h"
#include "forward_propagation.h"
#include "back_propagation.h"
#include "profiler.h"

namespace opennn
{

void DropoutOp::set_rate(float new_rate)
{
    throw_if(new_rate < 0.0f || new_rate >= 1.0f,
             "Dropout rate must be in [0, 1).");

    rate = new_rate;
}

void DropoutOp::forward_propagate(ForwardPropagation& fp, size_t layer, bool is_training)
{
    if (!is_training || !active()) return;

    auto& forward_slots = fp.forward_slots[layer];
    TensorView& output = get_output(fp, layer);

    if (!save_slots.empty())
        copy(output, forward_slots[save_slots[0]]);

    dropout_forward(output, mask, rate);
}

void DropoutOp::back_propagate(ForwardPropagation&, BackPropagation& bp, size_t layer) const
{
    if (!active()) return;
    dropout_backward(get_output_delta(bp, layer), mask, rate);
}

void DropoutOp::to_JSON(JsonWriter& w) const
{
    if (rate > 0.0f)
        add_json_field(w, "DropoutRate", to_string(rate));
}

void DropoutOp::from_JSON(const Json* parent)
{
    if (parent && parent->has("DropoutRate"))
        set_rate(float(read_json_float(parent, "DropoutRate")));
}

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
