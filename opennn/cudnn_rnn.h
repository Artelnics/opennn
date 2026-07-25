//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   C U D N N   R N N   S T A T E   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#pragma once

#include "operator.h"

namespace opennn
{

inline constexpr int RNN_MAX_LINEAR_LAYERS = 8;

#ifdef OPENNN_HAS_CUDA

struct CudnnRnnConfig
{
    cudnnRNNMode_t cell_mode;
    int num_linear_layers;
    const char* persist_env_var;
};

#endif

struct CudnnRnnState
{
protected:
    mutable Buffer weight_space_buf  {Device::CUDA};
    mutable Buffer dweight_space_buf {Device::CUDA};
    mutable Buffer workspace_buf     {Device::CUDA};
    mutable Buffer reserve_space_buf {Device::CUDA};
    mutable Buffer y_buf             {Device::CUDA};
    mutable Buffer dy_buf            {Device::CUDA};
    mutable Buffer dx_scratch_buf    {Device::CUDA};

    mutable CudnnDescriptor<cudnnRNNDescriptor_t>     rnn_desc;
    mutable CudnnDescriptor<cudnnDropoutDescriptor_t> dropout_desc;
    mutable Buffer dropout_states_buf{Device::CUDA};

    mutable CudnnRnnShapeSlot shape_slots_[RNN_SHAPE_SLOTS];
    mutable int active_shape_ = -1;
    mutable int shape_stamp_  = 0;
    CudnnRnnShapeSlot& active_shape() const { return shape_slots_[active_shape_]; }

    mutable Index cached_input_features  = -1;
    mutable Index cached_output_features = -1;

    mutable float* cudnn_w_ptrs_[RNN_MAX_LINEAR_LAYERS]  = {};
    mutable float* cudnn_b_ptrs_[RNN_MAX_LINEAR_LAYERS]  = {};
    mutable float* cudnn_gw_ptrs_[RNN_MAX_LINEAR_LAYERS] = {};
    mutable float* cudnn_gb_ptrs_[RNN_MAX_LINEAR_LAYERS] = {};

    mutable bool persist_algo_failed_ = false;
    mutable bool persist_algo_active_ = false;

#ifdef OPENNN_HAS_CUDA
    void cudnn_setup_(const CudnnRnnConfig&,
                      Index input_features, Index output_features, Index time_steps,
                      Index batch_size, bool for_training) const;
    void cudnn_setup_attempt_(const CudnnRnnConfig&,
                              Index input_features, Index output_features, Index time_steps,
                              Index batch_size, bool for_training) const;
    void cudnn_pack_weights_(int num_linear_layers,
                             Index input_features, Index output_features,
                             const TensorView* const* weights,
                             const TensorView* const* biases) const;
    void cudnn_unpack_gradients_(int num_linear_layers,
                                 Index input_features, Index output_features,
                                 const TensorView* const* weight_gradients,
                                 const TensorView* const* bias_gradients) const;
#endif
};

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
