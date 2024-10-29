#ifndef NEURALNETWORKBACKPROPAGATION_H
#define NEURALNETWORKBACKPROPAGATION_H

#include <string>

#include "neural_network.h"

using namespace std;
using namespace Eigen;

namespace opennn
{

struct NeuralNetworkBackPropagation
{
    NeuralNetworkBackPropagation(NeuralNetwork* new_neural_network = nullptr);

    void set(const Index& = 0, NeuralNetwork* = nullptr);

    const vector<unique_ptr<LayerBackPropagation>>& get_layers() const;

    NeuralNetwork* get_neural_network() const;

    void print() const;

    Index batch_samples_number = 0;

    NeuralNetwork* neural_network = nullptr;

    vector<unique_ptr<LayerBackPropagation>> layers;
};


}
#endif
