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

    Index batch_samples_number = 0;

    LossIndex* loss_index = nullptr;

    NeuralNetworkBackPropagation neural_network;

    Tensor<type, 0> error;
    type regularization = type(0);
    type loss = type(0);

    Tensor<type, 2> errors;

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

}
#endif
