//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   M E R G E   O P E R A T O R   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#pragma once

#include "operator.h"

namespace opennn
{

struct MergeOp : Operator
{
    Index heads_number = 0;
    Index query_sequence_length = 0;
    Index head_dimension = 0;
    Type  compute_dtype = Type::FP32;

    void set(Index heads_number, Index query_sequence_length, Index head_dimension, Type compute_dtype);

    void forward_propagate(ForwardPropagation& fp, size_t layer, bool is_training) override;
    void back_propagate(ForwardPropagation& fp, BackPropagation& bp, size_t layer) const override;
};

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
