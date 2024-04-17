#ifndef BACKPROPAGATION_H
#define BACKPROPAGATION_H

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

    explicit BackPropagation(const Index& new_batch_samples_number, LossIndex* new_loss_index)
    {
        set(new_batch_samples_number, new_loss_index);
    }

    virtual ~BackPropagation();

    void set(const Index& new_batch_samples_number, LossIndex* new_loss_index);

    void set_layers_outputs_indices(Tensor<Tensor<Index, 1>, 1>&);

    pair<type*, dimensions> get_output_deltas_pair() const;

    void print() const
    {
        cout << "Back-propagation" << endl;

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

    Tensor<Tensor<Index, 1>, 1> layers_outputs_indices;

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
};

}
#endif
