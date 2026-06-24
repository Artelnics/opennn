//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   P O O L   O P E R A T O R   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#pragma once

#include "operator.h"

namespace opennn
{

struct PoolOp : Operator
{
    enum Method { Max, Average };

    Index input_height = 0;
    Index input_width = 0;
    Index input_channels = 0;

    Index pool_height = 1;
    Index pool_width = 1;
    Index row_stride = 1;
    Index column_stride = 1;
    Index padding_height = 0;
    Index padding_width = 0;

    Method method = Max;

    CudnnDescriptor<cudnnPoolingDescriptor_t> pooling_descriptor;

    void set(Index input_h, Index input_w, Index input_c,
             Index pool_h, Index pool_w,
             Index row_stride, Index column_stride,
             Index padding_h, Index padding_w,
             Method method);

    PoolOp() = default;
    PoolOp(const PoolOp&) = delete;
    PoolOp& operator=(const PoolOp&) = delete;

    void forward_propagate(ForwardPropagation& fp, size_t layer, bool is_training) override;
    void back_propagate(ForwardPropagation& fp, BackPropagation& bp, size_t layer) const override;
};

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
