//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   B A T C H   S T R U C T   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#pragma once

#include "tensor_utilities.h"
#include "dataset.h"

namespace opennn
{

/// @brief Minibatch container holding pinned host/device buffers and views into a Dataset.
struct Batch
{
    /// @brief Constructs a batch sized for @p samples_number samples drawn from @p dataset.
    /// @param samples_number Maximum number of samples this batch can hold.
    /// @param dataset Source dataset used to discover variable shapes (non-owning).
    Batch(const Index = 0, const Dataset* = nullptr);
    ~Batch();

    Batch(const Batch&)            = delete;
    Batch& operator=(const Batch&) = delete;
    Batch(Batch&&)                 = delete;
    Batch& operator=(Batch&&)      = delete;

    /// @brief Reconfigures the batch for a new size or dataset; reuses allocations when possible.
    /// @param samples_number Maximum number of samples this batch can hold.
    /// @param dataset Source dataset used to discover variable shapes (non-owning).
    void set(const Index = 0, const Dataset* = nullptr);

    /// @brief Loads the indicated samples from the dataset into the batch buffers.
    /// @param sample_indices Indices of the dataset rows to load.
    /// @param input_indices Indices of input variables in the dataset.
    /// @param decoder_indices Indices of decoder-side input variables (may be empty).
    /// @param target_indices Indices of target variables in the dataset.
    /// @param is_training Marks the batch as training (controls augmentation/dropout caches).
    /// @param parallelize_samples If true, copies samples in parallel.
    void fill(const vector<Index>&,
              const vector<Index>&,
              const vector<Index>&,
              const vector<Index>&,
              bool is_training = true,
              bool parallelize_samples = true);

    /// @brief Returns the tensor views over the input buffer (device on GPU mode, host on CPU mode).
    const vector<TensorView>& get_inputs() const
    {
#ifdef OPENNN_HAS_CUDA
        if (is_gpu()) return input_views_cache;
#endif
        return input_views_host_cache;
    }

    /// @brief Returns the tensor view over the target buffer (device on GPU mode, host on CPU mode).
    const TensorView& get_targets() const
    {
#ifdef OPENNN_HAS_CUDA
        if (is_gpu()) return target_view_cache;
#endif
        return target_view_host_cache;
    }

    /// @brief Returns the current sample count (set by fill(); may be < samples_number).
    Index get_samples_number() const;

    /// @brief Prints a human-readable summary of the batch shapes and contents.
    void print() const;

    /// @brief Returns true when the batch is uninitialized or holds zero samples.
    bool is_empty() const;

    Index samples_number = 0;
    Index current_sample_count = 0;     // set by fill(); may be < samples_number

    const Dataset* dataset = nullptr;

    Buffer input;
    Shape input_shape;

    Buffer decoder;
    Shape decoder_shape;

    Buffer target;
    Shape target_shape;

    int input_contiguous = -1;
    int decoder_contiguous = -1;
    int target_contiguous = -1;

#ifdef OPENNN_HAS_CUDA
    /// @brief Asynchronously copies a slice of host buffers to device memory.
    /// @param sample_count Number of samples to copy.
    /// @param stream CUDA stream used for the asynchronous copy.
    /// @param fp32_staging Optional FP32 staging buffer used when promoting from lower precision.
    void copy_device_async(const Index, cudaStream_t, float* fp32_staging);
#endif

    Index get_input_elements() const { return samples_number * input_features_number; }

    vector<TensorView> input_views_host_cache;
    TensorView target_view_host_cache;

    vector<TensorView> input_views_cache;       // GPU views; populated only on CUDA mode
    TensorView target_view_cache;

    Index input_features_number = 0;
    Index decoder_features_number = 0;
    Index target_features_number = 0;

    float* inputs_host = nullptr;
    float* decoder_host = nullptr;
    float* targets_host = nullptr;

    Index inputs_host_allocated_size = 0;
    Index decoder_host_allocated_size = 0;
    Index targets_host_allocated_size = 0;

    bool needs_fp32_staging = false;
};

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
