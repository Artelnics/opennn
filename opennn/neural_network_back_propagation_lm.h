#ifndef NEURALNETWORKBACKPROPAGATIONLM_H
#define NEURALNETWORKBACKPROPAGATIONLM_H

#include "neural_network.h"

namespace opennn
{

struct NeuralNetworkBackPropagationLM
{
    NeuralNetworkBackPropagationLM(NeuralNetwork* new_neural_network = nullptr)
    {
        neural_network = new_neural_network;
    }

    void set(const Index& = 0, NeuralNetwork* = nullptr);

    const vector<unique_ptr<LayerBackPropagationLM>>& get_layers() const 
    {
        return layers;
    }

    NeuralNetwork* get_neural_network() const
    {
        return neural_network;
    }


    void print()
    {
        const Index layers_number = layers.size();

        cout << "Layers number: " << layers_number << endl;

        for(Index i = 0; i < layers_number; i++)
        {
            cout << "Layer " << i + 1 << endl;

            layers[i]->print();
        }
    }

    Index batch_samples_number = 0;

    NeuralNetwork* neural_network = nullptr;

    vector<unique_ptr<LayerBackPropagationLM>> layers;
};

}
#endif
