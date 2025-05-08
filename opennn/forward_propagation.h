#ifndef FORWARDPROPAGATION_H
#define FORWARDPROPAGATION_H

#include "neural_network.h"

using namespace std;
using namespace Eigen;

namespace opennn
{

struct ForwardPropagation
{
    ForwardPropagation(const Index& = 0, NeuralNetwork* = nullptr);

    void set(const Index& = 0, NeuralNetwork* = nullptr);

    pair<type*, dimensions> get_last_trainable_layer_outputs_pair() const;

    vector<vector<pair<type*, dimensions>>> get_layer_input_pairs(const vector<pair<type*, dimensions>>&, const bool&) const;

    void print() const;

    Index samples_number = 0;

    NeuralNetwork* neural_network = nullptr;

    vector<unique_ptr<LayerForwardPropagation>> layers;
};


#ifdef OPENNN_CUDA

struct ForwardPropagationCuda
{
    ForwardPropagationCuda(const Index& = 0, NeuralNetwork* = nullptr);

    void set(const Index& = 0, NeuralNetwork* = nullptr);

    pair<type*, dimensions> get_last_trainable_layer_outputs_pair_device() const;

    vector<vector<pair<type*, dimensions>>> get_layer_input_pairs_device(const vector<pair<type*, dimensions>>&, const bool&) const;

    void print();

    void free();

    Index samples_number = 0;

    NeuralNetwork* neural_network = nullptr;
    
    vector<unique_ptr<LayerForwardPropagationCuda>> layers;
};

#endif

}
#endif
