//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   C O N V O L U T I O N   O P E R A T O R   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#pragma once

#include "operator.h"

namespace opennn
{

struct ConvolutionOperator : Operator
{
    Index input_height = 0;
    Index input_width = 0;

    Index kernels_number = 0;
    Index kernel_height = 0;
    Index kernel_width = 0;
    Index kernel_channels = 0;
    Index row_stride = 1;
    Index column_stride = 1;

    Index padding_height = 0;
    Index padding_width = 0;

    bool use_bias = true;

    TensorView weights;
    TensorView bias;

    TensorView weight_gradient;
    TensorView bias_gradient;

    bool fuse_relu = false;

    bool weights_relinked = true;

    bool is_pointwise() const noexcept
    {
        return kernel_height == 1 && kernel_width == 1
            && row_stride == 1 && column_stride == 1
            && padding_height == 0 && padding_width == 0;
    }

#ifdef OPENNN_HAS_CUDA
    struct ConvGraphCache;
    mutable unique_ptr<ConvGraphCache> conv_graph_cache;

    void apply_gpu_folded(const TensorView& input,
                          const TensorView& folded_weights,
                          const TensorView& folded_bias,
                          bool relu, TensorView& output);
#endif

    void set(Index, Index,
             Index, Index, Index, Index,
             Index, Index,
             Index, Index,
             Type);

    vector<TensorSpec> parameter_specs() const override;
    void link_parameters(span<const TensorView>) override;
    void link_gradients (span<const TensorView>) override;

    void set_parameters_random() override;
    void set_parameters_glorot() override;

    ~ConvolutionOperator() override;

    ConvolutionOperator();
    ConvolutionOperator(const ConvolutionOperator&) = delete;
    ConvolutionOperator& operator=(const ConvolutionOperator&) = delete;

    void forward_propagate(ForwardPropagation&, size_t, bool) override;
    void back_propagate(ForwardPropagation&, BackPropagation&, size_t) const override;

private:
    void apply_cpu(const TensorView&, TensorView&);
    void apply_gpu(const TensorView&, TensorView&);

    void apply_delta_cpu(const TensorView&, const TensorView&,
                         TensorView&) const;
    void apply_delta_gpu(const TensorView&, const TensorView&,
                         TensorView&) const;
};

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
