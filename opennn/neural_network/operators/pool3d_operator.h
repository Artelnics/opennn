//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   P O O L 3 D   O P E R A T O R   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#pragma once

#include "opennn/neural_network/operators/operator.h"

namespace opennn
{

// The pooling kernels themselves. Pool3dOperator selects one per Method;
// they live here rather than in core because it is their only caller.
//
// The trailing valid_lengths is one length per sequence, or nullptr for a batch
// that carries no padding record. Given, it is where each sequence ends; absent,
// the sequence length is read off the data by treating an all-zero token row as
// padding, which is only true of a row nothing downstream of the Embedding has
// touched. It is deliberately not defaulted: a caller that has the lengths and
// forgets to pass them gets the guess, silently, and that is the failure this
// parameter exists to end.
void max_pooling_3d_forward(const TensorView&, TensorView&, TensorView&, bool, SequenceLengths);
void average_pooling_3d_forward(const TensorView&, TensorView&, SequenceLengths);
void max_pooling_3d_backward(const TensorView&, const TensorView&, TensorView&);
void average_pooling_3d_backward(const TensorView&, const TensorView&, TensorView&, SequenceLengths);
void first_token_3d_forward(const TensorView&, TensorView&);
void first_token_3d_backward(const TensorView&, TensorView&);

struct Pool3dOperator : Operator
{
    enum Method { Max, Average, First };
    Method method = Average;

    void forward_propagate(ForwardPropagation&, size_t, bool) override;
    void back_propagate(ForwardPropagation&, BackPropagation&, size_t) const override;
};

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
