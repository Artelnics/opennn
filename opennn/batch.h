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


struct Batch
{
    Batch(const Index = 0, const Dataset* = nullptr);

    void set(const Index = 0, const Dataset* = nullptr);

    void fill(const vector<Index>&,
              const vector<Index>&,
              const vector<Index>&,
              const vector<Index>&,
              bool augment = false);

    vector<TensorView> get_inputs() const;

    TensorView get_targets() const;

    Index get_samples_number() const;

    void print() const;

    bool is_empty() const;

    Index samples_number = 0;

    const Dataset* dataset = nullptr;

    Shape input_shape;
    VectorR input_vector;

    Shape decoder_shape;
    VectorR decoder_vector;

    Shape target_shape;
    VectorR target_vector;
};

#ifdef CUDA

struct BatchCuda
{
    BatchCuda(const Index = 0, Dataset* = nullptr);
    ~BatchCuda();

    void set(const Index, Dataset*);

    void fill(const vector<Index>&,
              const vector<Index>&,
              const vector<Index>&,
              const vector<Index>&);

    void fill_host(const vector<Index>&,
                   const vector<Index>&,
                   const vector<Index>&,
                   const vector<Index>&);

    vector<TensorView> get_inputs_device() const;
    TensorView get_targets_device() const;

    Index get_samples_number() const;

    MatrixR get_inputs_from_device() const;
    MatrixR get_decoder_from_device() const;
    MatrixR get_targets_from_device() const;

    void copy_device(const Index);
    void copy_device_async(const Index, cudaStream_t);

    void print() const;

    bool is_empty() const;

    Index samples_number = 0;
    Index num_input_features = 0;
    Index num_decoder_features = 0;
    Index num_target_features = 0;

    Dataset* dataset = nullptr;

    Shape input_shape;
    Shape decoder_shape;
    Shape target_shape;

    float* inputs_host = nullptr;
    float* decoder_host = nullptr;
    float* targets_host = nullptr;

    Index inputs_host_allocated_size = 0;
    Index decoder_host_allocated_size = 0;
    Index targets_host_allocated_size = 0;

    TensorCuda inputs_device;
    TensorCuda decoder_device;
    TensorCuda targets_device;
};

#endif

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
