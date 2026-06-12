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

struct ConvolutionOp : Operator
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

    Type compute_dtype = Type::FP32;

    TensorView weights;
    TensorView bias;

    TensorView weight_gradient;
    TensorView bias_gradient;

    cudnnActivationDescriptor_t fused_activation = nullptr;

    cudnnFilterDescriptor_t      kernel_descriptor      = nullptr;
    cudnnConvolutionDescriptor_t convolution_descriptor = nullptr;

    cudnnConvolutionFwdAlgo_t       algorithm_forward = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
    cudnnConvolutionBwdDataAlgo_t   algorithm_data    = CUDNN_CONVOLUTION_BWD_DATA_ALGO_0;
    cudnnConvolutionBwdFilterAlgo_t algorithm_filter  = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0;

    size_t cudnn_workspace_size_ = 0;

    Index planned_batch_size = 0;

    struct ConvGraphCache;
    mutable unique_ptr<ConvGraphCache> conv_graph_cache;

    void set(Index input_h, Index input_w,
             Index kernels_n, Index kernel_h, Index kernel_w, Index kernel_c,
             Index row_stride, Index column_stride,
             Index padding_h, Index padding_w,
             Type compute_dtype);

    vector<TensorSpec> parameter_specs() const override;
    void link_parameters(span<const TensorView> views) override;
    void link_gradients (span<const TensorView> views) override;

    void set_parameters_random() override;
    void set_parameters_glorot() override;

    void destroy_cuda() override;

    ~ConvolutionOp() override;

    ConvolutionOp();
    ConvolutionOp(const ConvolutionOp&) = delete;
    ConvolutionOp& operator=(const ConvolutionOp&) = delete;

    void forward_propagate(ForwardPropagation& fp, size_t layer, bool is_training) override;
    void back_propagate(ForwardPropagation& fp, BackPropagation& bp, size_t layer) const override;

    void apply(const TensorView& input, TensorView& output, cudnnActivationDescriptor_t fused_activation = nullptr);

    void apply_delta(const TensorView& input,
                     const TensorView& output_delta,
                     TensorView& input_delta) const;

private:
    void apply_cpu(const TensorView& input, TensorView& output);
    void apply_gpu(const TensorView& input, TensorView& output, cudnnActivationDescriptor_t fused_activation = nullptr);

    void apply_delta_cpu(const TensorView& input, const TensorView& output_delta,
                         TensorView& input_delta) const;
    void apply_delta_gpu(const TensorView& input, const TensorView& output_delta,
                         TensorView& input_delta) const;
    void plan_convolution_algorithms(const TensorView& input, const TensorView& output);

    array<pair<Index, Index>, 4> nhwc_padding() const;
};

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
