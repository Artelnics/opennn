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

    CudnnDescriptor<cudnnActivationDescriptor_t> fused_activation;

    CudnnDescriptor<cudnnFilterDescriptor_t>      kernel_descriptor;
    CudnnDescriptor<cudnnConvolutionDescriptor_t> convolution_descriptor;

    cudnnConvolutionFwdAlgo_t       algorithm_forward = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
    cudnnConvolutionBwdDataAlgo_t   algorithm_data    = CUDNN_CONVOLUTION_BWD_DATA_ALGO_0;
    cudnnConvolutionBwdFilterAlgo_t algorithm_filter  = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0;

    size_t cudnn_workspace_size_ = 0;

    Index planned_batch_size = 0;

#ifdef OPENNN_HAS_CUDA
    struct ConvGraphCache;
    mutable unique_ptr<ConvGraphCache> conv_graph_cache;
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
    void plan_convolution_algorithms(const TensorView&, const TensorView&);
};

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
