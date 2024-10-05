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

    virtual ~BackPropagation();

    void set(const Index& new_batch_samples_number, LossIndex* new_loss_index);

    //void set_layers_outputs_indices(const vector<vector<Index>>&);

    vector<vector<pair<type*, dimensions>>> get_layers_deltas(const Index last_trainable_layer_index, const Index first_trainable_layer_index) const 
    {
        vector<vector<pair<type*, dimensions>>> layers_deltas(neural_network.get_neural_network()->get_layers().size());

        for (Index i = last_trainable_layer_index; i >= first_trainable_layer_index; --i)
        {
            if (i == last_trainable_layer_index)
            {
                // Use output deltas as initial deltas
                layers_deltas[i].push_back(get_output_deltas_pair());
            }
            else
            {
                for (Index j = 0; j < neural_network.get_neural_network()->get_layers_output_indices()[i].size(); ++j)
                {
                    Index output_index = neural_network.get_neural_network()->get_layers_output_indices()[i][j];
                    Index input_index = loss_index->find_input_index(neural_network.get_neural_network()->get_layers_input_indices()[output_index], i);
                    
                    // Use the input derivatives from the previous layer as the deltas for the current layer
                    layers_deltas[i].push_back(neural_network.get_layers()[output_index]->get_inputs_derivatives_pair()[input_index]);
                }
            }
        }

        return layers_deltas;
    }


    pair<type*, dimensions> get_output_deltas_pair() const;

    void print() const
    {
        cout << "Back-propagation" << endl
             << "Errors:" << endl
             << errors << endl
             << "Error:" << endl
             << error << endl
             << "Regularization:" << endl
             << regularization << endl
             << "Loss:" << endl
             << loss << endl
             << "Gradient:" << endl
             << gradient << endl;

        neural_network.print();
    }

    //vector<vector<Index>> layers_outputs_indices;

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
