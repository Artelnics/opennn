//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   C U D N N   R N N   S T A T E   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#pragma once

#include "opennn/neural_network/operators/operator.h"

#include <functional>

namespace opennn
{

template<typename Handle>
struct CudnnDescriptor
{
    Handle handle = nullptr;
#ifdef OPENNN_HAS_CUDA
    cudnnStatus_t (*deleter)(Handle) = nullptr;
#else
    void (*deleter)(Handle) = nullptr;
#endif

    CudnnDescriptor() = default;

    CudnnDescriptor(CudnnDescriptor&& other) noexcept
        : handle(other.handle), deleter(other.deleter)
    {
        other.handle = nullptr;
        other.deleter = nullptr;
    }

    CudnnDescriptor& operator=(CudnnDescriptor&& other) noexcept
    {
        if (this != &other)
        {
            reset();
            handle = other.handle;
            deleter = other.deleter;
            other.handle = nullptr;
            other.deleter = nullptr;
        }
        return *this;
    }

    CudnnDescriptor(const CudnnDescriptor&) = delete;
    CudnnDescriptor& operator=(const CudnnDescriptor&) = delete;

    ~CudnnDescriptor() { reset(); }

    void reset()
    {
        if (handle && deleter) deleter(handle);
        handle = nullptr;
        deleter = nullptr;
    }

    operator Handle() const { return handle; }
    explicit operator bool() const { return handle != nullptr; }
};

inline constexpr int RNN_SHAPE_SLOTS = 3;

struct CudnnRnnShapeSlot
{
    Index batch = -1;
    Index time  = -1;
    int   stamp = 0;
    bool  training_ready = false;
    CudnnDescriptor<cudnnRNNDataDescriptor_t> x_desc;
    CudnnDescriptor<cudnnRNNDataDescriptor_t> y_desc;
    CudnnDescriptor<cudnnTensorDescriptor_t>  h_desc;
    CudnnDescriptor<cudnnTensorDescriptor_t>  c_desc;
    Buffer seq_host{Device::CPU};
    Buffer seq_dev {Device::CUDA};
};

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

    void cudnn_rnn_forward_(bool is_training, bool has_cell_state,
                            const void* x, void* y,
                            const function<void()>& reconfigure) const;
    void cudnn_rnn_backward_(bool has_cell_state,
                             const void* x, const void* y, const void* dy,
                             void* dx) const;

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
