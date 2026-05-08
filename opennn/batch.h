//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   B A T C H   S T R U C T   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

/**
 * @file batch.h
 * @brief Declares the Batch struct used to feed mini-batches into
 *        ForwardPropagation and BackPropagation.
 */

#pragma once

#include "tensor_utilities.h"
#include "dataset.h"

namespace opennn
{


/**
 * @struct Batch
 * @brief Owns the host-side and (optional) device-side buffers for one
 *        mini-batch of dataset samples.
 *
 * The Batch is filled once per training step from a Dataset using fill().
 * Its TensorViews (target_view, input_views) are then handed to the
 * NeuralNetwork's forward / backward passes. On CUDA-enabled builds the
 * batch maintains parallel host and device buffers; copy_device_async()
 * uploads the host data on a user-supplied CUDA stream so it can overlap
 * with compute.
 */
struct Batch
{
    /**
     * @brief Constructs a batch sized for a given dataset.
     * @param samples_number Number of samples per batch.
     * @param dataset Dataset whose shapes determine the batch buffer sizes.
     */
    Batch(const Index samples_number = 0, const Dataset* dataset = nullptr);

    /**
     * @brief (Re)allocates buffers for a given batch size and dataset.
     * @param samples_number Number of samples per batch.
     * @param dataset Dataset whose shapes determine the batch buffer sizes.
     */
    void set(const Index samples_number = 0, const Dataset* dataset = nullptr);

    /**
     * @brief Fills the host-side buffers from the bound dataset.
     * @param sample_indices Indices of samples copied into this batch.
     * @param input_feature_indices Dataset columns used as inputs.
     * @param decoder_feature_indices Dataset columns used as decoder inputs
     *                                (empty when the model has no decoder).
     * @param target_feature_indices Dataset columns used as targets.
     * @param augment True to apply Dataset::augment_inputs() to the input buffer.
     */
    void fill(const vector<Index>& sample_indices,
              const vector<Index>& input_feature_indices,
              const vector<Index>& decoder_feature_indices,
              const vector<Index>& target_feature_indices,
              bool augment = false);

    /**
     * @brief Returns the input TensorViews for the active device.
     * @return Device-side views on CUDA builds, host-side views otherwise.
     */
    const vector<TensorView>& get_inputs() const
    {
#ifdef OPENNN_HAS_CUDA
        if (Configuration::instance().is_gpu()) return input_views_cache;
#endif
        return input_views_host_cache;
    }

    /**
     * @brief Returns the target TensorView for the active device.
     * @return Device-side view on CUDA builds, host-side view otherwise.
     */
    const TensorView& get_targets() const
    {
#ifdef OPENNN_HAS_CUDA
        if (Configuration::instance().is_gpu()) return target_view_cache;
#endif
        return target_view_host_cache;
    }

    /** @brief Number of samples currently held in the batch. */
    Index get_samples_number() const;

    /** @brief Prints a human-readable summary of the batch buffers to stdout. */
    void print() const;

    /** @brief Whether the batch holds zero samples. */
    bool is_empty() const;

    /** @brief Number of samples currently held in the batch. */
    Index samples_number = 0;

    /** @brief Dataset whose shapes determined the batch buffer sizes; not owned. */
    const Dataset* dataset = nullptr;

    /** @brief Owning storage for the input tensor (host or device). */
    Buffer input;
    /** @brief Shape of @ref input (per-sample dimensions). */
    Shape input_shape;

    /** @brief Owning storage for the decoder tensor (encoder-decoder models). */
    Buffer decoder;
    /** @brief Shape of @ref decoder. */
    Shape decoder_shape;

    /** @brief Owning storage for the target tensor. */
    Buffer target;
    /** @brief Shape of @ref target. */
    Shape target_shape;

    /** @brief Stride hint passed to Dataset::fill_inputs(); -1 to ignore. */
    int input_contiguous = -1;
    /** @brief Stride hint passed to Dataset::fill_inputs() for the decoder; -1 to ignore. */
    int decoder_contiguous = -1;
    /** @brief Stride hint passed to Dataset::fill_targets(); -1 to ignore. */
    int target_contiguous = -1;

    /**
     * @brief Asynchronously copies the host buffers to the device on @p stream.
     * @param sample_count Number of samples to copy from the host buffer.
     * @param stream CUDA stream used for the asynchronous copy.
     */
    void copy_device_async(const Index sample_count, cudaStream_t stream);

    /** @brief Host-side input TensorViews (one per input feature group). */
    vector<TensorView> input_views_host_cache;
    /** @brief Host-side target TensorView. */
    TensorView target_view_host_cache;

    /** @brief Device-side input TensorViews; populated only on CUDA mode. */
    vector<TensorView> input_views_cache;
    /** @brief Device-side target TensorView; populated only on CUDA mode. */
    TensorView target_view_cache;

    /** @brief Total number of input features across all input feature groups. */
    Index num_input_features = 0;
    /** @brief Total number of decoder features. */
    Index num_decoder_features = 0;
    /** @brief Total number of target features. */
    Index num_target_features = 0;

    /** @brief Pinned host pointer used to stage inputs into the device buffer. */
    float* inputs_host = nullptr;
    /** @brief Pinned host pointer used to stage decoder inputs into the device buffer. */
    float* decoder_host = nullptr;
    /** @brief Pinned host pointer used to stage targets into the device buffer. */
    float* targets_host = nullptr;

    /** @brief Allocation capacity of @ref inputs_host, in floats. */
    Index inputs_host_allocated_size = 0;
    /** @brief Allocation capacity of @ref decoder_host, in floats. */
    Index decoder_host_allocated_size = 0;
    /** @brief Allocation capacity of @ref targets_host, in floats. */
    Index targets_host_allocated_size = 0;

    /** @brief Device-resident FP32 staging buffer used during mixed-precision uploads. */
    Buffer inputs_fp32_staging{Device::CUDA};
};

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
