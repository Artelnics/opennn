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

    NeuralNetwork* get_neural_network() const;

    void print() const;

    Index batch_size = 0;

    NeuralNetwork* neural_network = nullptr;

    vector<unique_ptr<LayerBackPropagation>> layers;
};


#ifdef OPENNN_CUDA_test

struct NeuralNetworkBackPropagationCuda
{
    NeuralNetworkBackPropagationCuda(const Index& = 0, NeuralNetwork* = nullptr);

    void set(const Index& = 0, NeuralNetwork* = nullptr);

    void print();

    void free();

    Index batch_size = 0;

    NeuralNetwork* neural_network = nullptr;
    
    vector<unique_ptr<LayerBackPropagation>> layers;
};

#endif

}
#endif
