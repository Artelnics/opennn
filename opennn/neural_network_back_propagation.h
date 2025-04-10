#ifndef NEURALNETWORKBACKPROPAGATION_H
#define NEURALNETWORKBACKPROPAGATION_H

#include "neural_network.h"

namespace opennn
{

struct NeuralNetworkBackPropagation
{
    NeuralNetworkBackPropagation(const Index& = 0, NeuralNetwork* = nullptr);

    void set(const Index& = 0, NeuralNetwork* = nullptr);

    const vector<unique_ptr<LayerBackPropagation>>& get_layers() const;

    void print() const;

    NeuralNetwork* get_neural_network() const;

    Index batch_size = 0;

    NeuralNetwork* neural_network = nullptr;

    vector<unique_ptr<LayerBackPropagation>> layers;
};


#ifdef OPENNN_CUDA_test

struct NeuralNetworkBackPropagationCuda
{
    NeuralNetworkBackPropagationCuda(const Index& = 0, NeuralNetwork* = nullptr);

    void set(const Index& = 0, NeuralNetwork* = nullptr);

    const vector<unique_ptr<LayerBackPropagationCuda>>& get_layers() const;

    void print();

    void free();

    Index batch_size = 0;

    NeuralNetwork* neural_network = nullptr;
    
    vector<unique_ptr<LayerBackPropagationCuda>> layers;
};

#endif

}
#endif
