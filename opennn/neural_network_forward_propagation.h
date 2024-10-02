#ifndef FORWARDPROPAGATION_H
#define FORWARDPROPAGATION_H

//#include <string>

#include "neural_network.h"
#include "batch.h"

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
            delete layers[i];
    }

    void set(const Index& new_batch_samples_number, NeuralNetwork* new_neural_network);

    pair<type*, dimensions> get_last_trainable_layer_outputs_pair() const;

    vector<vector<pair<type*, dimensions>>> get_layers_inputs(const Batch& batch) const 
    {
        vector<vector<pair<type*, dimensions>>> layers_inputs(neural_network->get_layers().size());
        vector<pair<type*, dimensions>>();
        for (Index i = 0; i < layers_inputs.size(); ++i)
        {
            // Handle different input scenarios based on whether the layer is input, context, or a hidden layer
            if (neural_network->is_input_layer(neural_network->get_layers_input_indices()[i]))
            {
                layers_inputs[i].push_back(batch.get_inputs_pair()(0)); // Example: batch inputs
            }
            else if (neural_network->is_context_layer(neural_network->get_layers_input_indices()[i]))
            {
                layers_inputs[i].push_back(batch.get_inputs_pair()(1)); // Contextual input
            }
            else
            {
                // Use outputs from the previous layers as inputs
                for (Index j = 0; j < neural_network->get_layers_input_indices()[i].size(); ++j)
                {
                    Index input_index = neural_network->get_layers_input_indices()[i][j];
                    layers_inputs[i].push_back(layers[input_index]->get_outputs_pair());
                }
            }
        }

        return layers_inputs;
    }

    void print() const
    {
        cout << "Neural network forward propagation" << endl;

        const Index layers_number = layers.size();

        cout << "Layers number: " << layers_number << endl;

        for(Index i = 0; i < layers_number; i++)
        {
            cout << "Layer " << i + 1 << ": " << layers[i]->layer->get_name() << endl;

            layers[i]->print();
        }
    }

    Index batch_samples_number = 0;

    NeuralNetwork* neural_network = nullptr;

    Tensor<LayerForwardPropagation*, 1> layers;
};

}
#endif
