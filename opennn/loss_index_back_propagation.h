#ifndef LOSSINDEXBACKPROPAGATION_H
#define LOSSINDEXBACKPROPAGATION_H

#include <string>

#include "loss_index.h"

using namespace std;
using namespace Eigen;

namespace opennn
{

/// Set of loss value and gradient vector of the loss index.
/// A method returning this structure might be implemented more efficiently than the loss and gradient methods separately.

struct BackPropagation
{
    /// Default constructor.

    explicit BackPropagation() {}

    explicit BackPropagation(const Index& new_batch_samples_number, LossIndex* new_loss_index_pointer)
    {
        set(new_batch_samples_number, new_loss_index_pointer);
    }

    virtual ~BackPropagation();

    void set(const Index& new_batch_samples_number, LossIndex* new_loss_index_pointer)
    {
        loss_index_pointer = new_loss_index_pointer;

        batch_samples_number = new_batch_samples_number;

        // Neural network

        NeuralNetwork* neural_network_pointer = loss_index_pointer->get_neural_network_pointer();

        const Index parameters_number = neural_network_pointer->get_parameters_number();

        const Index outputs_number = neural_network_pointer->get_outputs_number();

        // First order loss

        neural_network.set(batch_samples_number, neural_network_pointer);

        error = type(0);

        loss = type(0);

        errors.resize(batch_samples_number, outputs_number);

        parameters = neural_network_pointer->get_parameters();

        gradient.resize(parameters_number);

        regularization_gradient.resize(parameters_number);
    }

    pair<type*, dimensions> get_output_deltas_pair() const
    {
        const NeuralNetwork* neural_network_pointer = loss_index_pointer->get_neural_network_pointer();

        const Index trainable_layers_number = neural_network_pointer->get_trainable_layers_number();

        return neural_network.layers(trainable_layers_number - 1)->get_deltas_pair();
    }


    void print() const
    {
        cout << "Loss index back-propagation" << endl;

        cout << "Errors:" << endl;
        cout << errors << endl;

        cout << "Error:" << endl;
        cout << error << endl;

        cout << "Regularization:" << endl;
        cout << regularization << endl;

        cout << "Loss:" << endl;
        cout << loss << endl;

        cout << "Gradient:" << endl;
        cout << gradient << endl;

        neural_network.print();
    }

    Index batch_samples_number = 0;

    LossIndex* loss_index_pointer = nullptr;

    NeuralNetworkBackPropagation neural_network;

    type error = type(0);
    type regularization = type(0);
    type loss = type(0);

    Tensor<type, 2> errors;

    Tensor<type, 1> parameters;

    Tensor<type, 1> gradient;
    Tensor<type, 1> regularization_gradient;
};

}
#endif
