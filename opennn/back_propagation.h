#ifndef BACKPROPAGATION_H
#define BACKPROPAGATION_H

#include "neural_network_back_propagation.h"
#include "loss_index.h"

using namespace std;
using namespace Eigen;

namespace opennn
{

struct BackPropagation
{
    explicit BackPropagation() {}

    explicit BackPropagation(const Index& new_batch_samples_number, LossIndex* new_loss_index)
    {
        set(new_batch_samples_number, new_loss_index);
    }

    void set(const Index& new_batch_samples_number, LossIndex* new_loss_index);

    vector<vector<pair<type*, dimensions>>> get_layer_delta_pairs() const;

    pair<type*, dimensions> get_output_deltas_pair() const;

    void print() const;

    Index batch_samples_number = 0;

    LossIndex* loss_index = nullptr;

    NeuralNetworkBackPropagation neural_network;

    type error = type(0);
    type regularization = type(0);
    type loss = type(0);

    Tensor<type, 2> errors;

    Tensor<type, 1> output_deltas;
    dimensions output_deltas_dimensions;

    Tensor<type, 1> parameters;

    Tensor<type, 1> gradient;
    Tensor<type, 1> regularization_gradient;

    type accuracy = type(0);
    Tensor<type, 2> predictions;
    Tensor<bool, 2> matches;
    Tensor<bool, 2> mask;
    bool built_mask = false;
};

}
#endif
