#ifndef FORWARDPROPAGATION_H
#define FORWARDPROPAGATION_H

#include <string>

#include "neural_network.h"

using namespace std;
using namespace Eigen;

namespace opennn
{

struct ForwardPropagation
{
    ForwardPropagation() {}

    ForwardPropagation(const Index& new_batch_samples_number, NeuralNetwork* new_neural_network)
    {
        set(new_batch_samples_number, new_neural_network);
    }

    virtual ~ForwardPropagation()
    {
        const Index layers_number = layers.size();

        for(Index i = 0; i < layers_number; i++)
        {
            delete layers(i);
        }
    }


    void set(const Index& new_batch_samples_number, NeuralNetwork* new_neural_network);

    pair<type*, dimensions> get_last_trainable_layer_outputs_pair() const;


    void print() const
    {
        cout << "Neural network forward propagation" << endl;

        const Index layers_number = layers.size();

        cout << "Layers number: " << layers_number << endl;

        for(Index i = 0; i < layers_number; i++)
        {
            cout << "Layer " << i + 1 << ": " << layers(i)->layer->get_name() << endl;

            layers(i)->print();
        }
    }

    Index batch_samples_number = 0;

    NeuralNetwork* neural_network = nullptr;

    Tensor<LayerForwardPropagation*, 1> layers;
};

}
#endif
