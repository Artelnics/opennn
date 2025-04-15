#ifndef BACKPROPAGATION_H
#define BACKPROPAGATION_H

#include "neural_network_back_propagation.h"
#include "loss_index.h"

namespace opennn
{

struct BackPropagation
{
    BackPropagation(const Index& = 0, LossIndex* = nullptr);

    void set(const Index& = 0, LossIndex* = nullptr);

    vector<vector<pair<type*, dimensions>>> get_layer_delta_pairs() const;

    pair<type*, dimensions> get_output_deltas_pair() const;

    void print() const;

    Index samples_number = 0;

    LossIndex* loss_index = nullptr;

    NeuralNetworkBackPropagation neural_network;

    Tensor<type, 0> error;
    type regularization = type(0);
    type loss = type(0);

    Tensor<type, 2> errors;
    Tensor<type, 2> errors_weights;

    Tensor<type, 1> output_deltas;
    dimensions output_deltas_dimensions;

    Tensor<type, 1> parameters;

    Tensor<type, 1> gradient;
    Tensor<type, 1> regularization_gradient;

    Tensor<type, 0> accuracy;
    Tensor<type, 2> predictions;
    Tensor<bool, 2> matches;
    Tensor<bool, 2> mask;
    bool built_mask = false;
};


#ifdef OPENNN_CUDA_test

struct BackPropagationCuda
{
    BackPropagationCuda(const Index& = 0, LossIndex* = nullptr);

    void set(const Index& = 0, LossIndex* = nullptr);

    vector<vector<pair<type*, dimensions>>> get_layer_delta_pairs_device() const;

    pair<type*, dimensions> get_output_deltas_pair_device() const;

    void print();

    void free();

    Index samples_number = 0;

    LossIndex* loss_index = nullptr;

    NeuralNetworkBackPropagationCuda neural_network;

    Tensor<type, 0> error;
    type regularization = type(0);
    type loss = type(0);

    cudnnReduceTensorDescriptor_t reduce_tensor_descriptor;
    void* workspace = nullptr;
    size_t workspaceSize = 0;

    float* numerator = nullptr;
    float* numerator_2 = nullptr;
    float* numerator_3 = nullptr;
    float* outputs_plus_epsilon = nullptr;
    float* one_minus_targets = nullptr;
    float* one_minus_outputs = nullptr;
    float* numerator_reduce = nullptr;
    cudnnTensorDescriptor_t outputs_tensor_descriptor = nullptr;
    cudnnTensorDescriptor_t output_reduce_tensor_descriptor = nullptr;

    float* errors = nullptr;

    float* output_deltas = nullptr;
    dimensions output_deltas_dimensions;

    float* parameters = nullptr;
    float* parameters_square = nullptr;
    cudnnTensorDescriptor_t parameters_tensor_descriptor = nullptr;
    Tensor<type, 1> parameters_host;

    float* gradient = nullptr;
    cudnnTensorDescriptor_t gradient_tensor_descriptor = nullptr;
    float* regularization_gradient = nullptr;

    float* ones = nullptr;
    float one = 1.0f;

    // @todo
    Tensor<type, 0> accuracy;
    float* predictions = nullptr;
    float* matches = nullptr;
    float* mask = nullptr;
    bool built_mask = false;

    cudnnOpTensorDescriptor_t operator_sum_descriptor = nullptr;
    cudnnOpTensorDescriptor_t operator_multiplication_descriptor = nullptr;
    cudnnOpTensorDescriptor_t operator_square_root_descriptor = nullptr;
};

#endif

}
#endif
