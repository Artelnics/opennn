//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   B A C K   P R O P A G A T I O N   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#pragma once

#include "tensor_utilities.h"

namespace opennn
{

class Loss;
class NeuralNetwork;

struct BackwardEdge
{
    size_t consumer_idx;
    size_t port;
};

struct BackPropagation
{
    BackPropagation(const Index = 0, Loss* = nullptr);

    virtual ~BackPropagation()
    {
#ifdef OPENNN_WITH_CUDA
        if(errors_device) { cudaFree(errors_device); errors_device = nullptr; }
#endif
    }

    void set(const Index = 0, Loss* = nullptr);

    void allocate_device();

    const NeuralNetwork* get_neural_network() const;

    void accumulate_output_deltas(size_t layer_index);

    NeuralNetwork* neural_network = nullptr;

    Memory gradient;
    vector<vector<TensorView>> gradient_views;

    Memory backward;
    vector<vector<vector<TensorView>>> delta_views;

    Memory per_layer_output_deltas;
    vector<Shape> per_layer_output_delta_shapes;
    vector<vector<BackwardEdge>> backward_edges;

    vector<vector<TensorView>> get_layer_gradients() const;

    TensorView get_output_deltas() const;

    TensorView get_output_deltas_active() const
    {
#ifdef OPENNN_WITH_CUDA
        return Device::instance().is_gpu() ? get_output_deltas_device() : get_output_deltas();
#else
        return get_output_deltas();
#endif
    }

    void print() const;

    Index batch_size = 0;

    Loss* loss = nullptr;

    type error = type(0);
    Index active_tokens_count = 0;
    MatrixR errors;
    Memory output_deltas;
    Shape output_delta_dimensions;

    Tensor0 accuracy;
    MatrixR predictions;

    MatrixB matches;
    MatrixB mask;

    bool built_mask = false;
    type loss_value = type(0);

#ifdef OPENNN_WITH_CUDA

    float* errors_device = nullptr;

    TensorView output_deltas_view_device;

    const TensorView& get_output_deltas_device() const;

#endif
};

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.