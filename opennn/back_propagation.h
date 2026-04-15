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

struct BackPropagation
{
    BackPropagation(const Index = 0, Loss* = nullptr);

    virtual ~BackPropagation() = default;

    void set(const Index = 0, Loss* = nullptr);

    void allocate_device();

    const NeuralNetwork* get_neural_network() const;

    NeuralNetwork* neural_network = nullptr;

    Memory gradient;
    vector<vector<TensorView>> gradient_views;

    Memory backward;
    vector<vector<vector<TensorView>>> backward_views;

    vector<vector<TensorView>> get_layer_gradients() const;

    TensorView get_output_gradients() const;

    void print() const;

    Index batch_size = 0;

    Loss* loss = nullptr;

    type error;
    Index active_tokens_count = 0;
    MatrixR errors;
    Memory output_gradients;
    Shape output_gradient_dimensions;

    Tensor0 accuracy;
    MatrixR predictions;

    MatrixB matches;
    MatrixB mask;

    bool built_mask = false;
    type loss_value = type(0);

#ifdef CUDA

    float* errors_device = nullptr;
    float* error_device = nullptr;

    TensorView get_output_gradients_device() const;

    void free_cuda();

#endif
};

}
