//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   F L A T   O P E R A T O R   S O U R C E
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "flat_operator.h"
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

void FlatOp::forward_propagate(ForwardPropagation& fp, size_t layer, bool /*is_training*/)
{
    copy(get_input(fp, layer), get_output(fp, layer));
}

void FlatOp::back_propagate(ForwardPropagation&, BackPropagation& bp, size_t layer) const
{
    TensorView& input_delta = get_input_delta(bp, layer);
    if (input_delta.empty()) return;

    copy(get_output_delta(bp, layer), input_delta);
}

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
