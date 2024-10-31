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
    NeuralNetworkBackPropagation() {}

    NeuralNetworkBackPropagation(NeuralNetwork* new_neural_network)
    {
        neural_network = new_neural_network;
    }

    void set(const Index& new_batch_samples_number, NeuralNetwork* new_neural_network);

    const vector<unique_ptr<LayerBackPropagation>>& get_layers() const
    {
        return layers;
    }

    NeuralNetwork* get_neural_network() const
    {
        return neural_network;
    }

    void print() const
    {
        cout << "Neural network back-propagation" << endl;

        const Index layers_number = layers.size();       

        for(Index i = 0; i < layers_number; i++)
        {
            cout << "Layer " << i << ": ";
            cout << neural_network->get_layer(i)->get_type_string() << endl;
            
            if(!layers[i]) continue;

            layers[i]->print();
        }
    }

    Index batch_samples_number = 0;

    NeuralNetwork* neural_network = nullptr;

    vector<unique_ptr<LayerBackPropagation>> layers;
};


}
#endif
