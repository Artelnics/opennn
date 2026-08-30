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
#include <mutex>

namespace opennn
{

inline constexpr int RNN_SHAPE_SLOTS = 3;

struct CudnnRnnShapeSlot
{
    Index batch = -1;
    Index time  = -1;
    Index input_features = -1;
    int  stamp = 0;
    bool training_ready = false;
    bool time_major = true;
    Index workspace_bytes = 0;
    Index reserve_space_bytes = 0;
    CudnnDescriptor<cudnnRNNDataDescriptor_t> x_desc;
    CudnnDescriptor<cudnnRNNDataDescriptor_t> y_desc;
    CudnnDescriptor<cudnnTensorDescriptor_t>  h_desc;
    CudnnDescriptor<cudnnTensorDescriptor_t>  c_desc;
    Buffer seq_host{Device::CPU};
    Buffer seq_dev {Device::CUDA};
};

#ifdef OPENNN_HAS_CUDA

struct CudnnRnnConfig
{
    cudnnRNNMode_t cell_mode;
    Type data_type = Type::FP32;
};

#endif

struct CudnnRnnState
{
protected:
    struct BackendState
    {
        mutex access_mutex;
        CudnnDescriptor<cudnnRNNDescriptor_t> rnn_desc;
        CudnnDescriptor<cudnnDropoutDescriptor_t> dropout_desc;
        Buffer dropout_states{Device::CUDA};
        CudnnRnnShapeSlot shape_slots[RNN_SHAPE_SLOTS];
        int shape_stamp = 0;
        Index cached_input_features = -1;
        Index cached_output_features = -1;
        Type cached_data_type = Type::Auto;
        Index weight_space_bytes = 0;
        bool persist_algo_failed = false;
        bool persist_algo_active = false;
        bool double_bias = false;
        bool packed_layout = false;

        const void* zeroed_weight_space = nullptr;
        Index zeroed_weight_space_bytes = 0;
        Index zeroed_input_features = -1;
        bool zeroed_double_bias = false;

        // What the packed weight space currently holds. cuDNN wants the weights
        // in its own layout, and copying them there is pure data movement that
        // repeats every forward pass even when nothing has moved: measured at
        // 3.4% of the LSTM layer. Packing is skipped when the destination
        // buffer, the shape and the network's parameter version all still
        // match. Version 0 means nothing has been packed yet.
        const void* packed_weight_space = nullptr;
        uint64_t packed_parameters_version = 0;
        Index packed_input_features = -1;
        Index packed_output_features = -1;
    };

    unique_lock<mutex> lock_backend_state() const
    {
        return unique_lock(backend_state.access_mutex);
    }

#ifdef OPENNN_HAS_CUDA

    void cudnn_rnn_forward_(const CudnnRnnShapeSlot&,
                            bool is_training, bool has_cell_state,
                            const void* x, void* y,
                            Buffer& forward_state,
                            const function<CudnnRnnShapeSlot&()>& reconfigure) const;
    void cudnn_rnn_backward_(const CudnnRnnShapeSlot&,
                             bool has_cell_state,
                             const void* x, const void* y, const void* dy,
                             void* dx,
                             const Buffer& forward_state,
                             Buffer& backward_scratch) const;

    CudnnRnnShapeSlot& cudnn_setup_(const CudnnRnnConfig&,
                                    Index input_features, Index output_features,
                                    Index time_steps, Index batch_size,
                                    bool for_training) const;
    CudnnRnnShapeSlot& cudnn_setup_attempt_(const CudnnRnnConfig&,
                                            Index input_features, Index output_features,
                                            Index time_steps, Index batch_size,
                                            bool for_training) const;
    void cudnn_copy_weight_regions_(int num_linear_layers,
                                    Index input_features,
                                    Index output_features,
                                    const TensorView* const* matrices,
                                    const TensorView* const* vectors,
                                    Buffer& packed_weights,
                                    bool to_cudnn) const;
    void cudnn_pack_weights_(int num_linear_layers,
                             Index input_features, Index output_features,
                             const TensorView* const* weights,
                             const TensorView* const* biases,
                             Buffer& forward_state,
                             uint64_t parameters_version) const;
    void cudnn_unpack_gradients_(int num_linear_layers,
                                 Index input_features, Index output_features,
                                 const TensorView* const* weight_gradients,
                                 const TensorView* const* bias_gradients,
                                 Buffer& backward_scratch) const;

    void prepare_cudnn_forward_state_(Buffer&, bool is_training,
                                      const CudnnRnnShapeSlot&) const;

    // What the two recurrent layers do not share: the shape they configure
    // cuDNN with, and how their weights pack into cuDNN's layout.
    virtual CudnnRnnShapeSlot& ensure_cudnn_setup_(Index batch_size, bool for_training) const = 0;
    virtual void pack_weights_to_cudnn_(Buffer&, uint64_t parameters_version) const = 0;
    virtual void unpack_gradients_from_cudnn_(Buffer&) const = 0;

    // What they do share. Driving cuDNN is the same sequence either way --
    // configure, pack, transpose into time-major, run, transpose back -- and
    // both layers had written it out, ~150 lines apiece, differing only in the
    // cell state the LSTM carries and the plain RNN does not.
    struct CudnnRnnDims
    {
        Index input_features  = 0;
        Index output_features = 0;
        Index time_steps      = 0;
        bool  return_sequences = false;
        bool  has_cell_state   = false;
    };

    void drive_cudnn_forward_(const CudnnRnnDims&,
                              const TensorView& input,
                              TensorView& sequence_output,
                              TensorView& output,
                              TensorView& cudnn_input_sequence,
                              TensorView& cudnn_output_sequence,
                              Buffer& forward_state,
                              bool is_training,
                              uint64_t parameters_version) const;

    void drive_cudnn_backward_(const CudnnRnnDims&,
                               const TensorView& input,
                               const TensorView& sequence_output,
                               const TensorView& output_delta,
                               const TensorView& cudnn_input_sequence,
                               const TensorView& cudnn_output_sequence,
                               TensorView& input_delta,
                               TensorView& sequence_delta_scratch,
                               TensorView& input_delta_scratch,
                               const Buffer& forward_state,
                               Buffer& backward_scratch) const;
#endif

    mutable BackendState backend_state;
};

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
