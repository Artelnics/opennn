//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   N U M E R I C A L   D E R I V A T I V E S   ( T E S T   H E L P E R )

#include "numerical_derivatives.h"

#include "../opennn/dataset.h"
#include "../opennn/neural_network.h"
#include "../opennn/forward_propagation.h"
#include "../opennn/back_propagation.h"
#include "../opennn/batch.h"

#include <Eigen/Dense>

namespace opennn
{

float calculate_numerical_error(Loss& loss)
{
    NeuralNetwork* neural_network = loss.get_neural_network();
    Dataset*       dataset        = loss.get_dataset();

    if (!neural_network) throw runtime_error("calculate_numerical_error: neural network is not set.");
    if (!dataset)        throw runtime_error("calculate_numerical_error: dataset is not set.");

    const Index samples_number = dataset->get_samples_number("Training");

    const vector<Index> training_indices         = dataset->get_sample_indices("Training");
    const vector<Index> input_feature_indices    = dataset->get_feature_indices("Input");
    const vector<Index> decoder_feature_indices  = dataset->get_feature_indices("Decoder");
    const vector<Index> target_feature_indices   = dataset->get_feature_indices("Target");

    Batch batch(samples_number, dataset);
    batch.fill(training_indices, input_feature_indices, decoder_feature_indices, target_feature_indices);

    ForwardPropagation forward_propagation(samples_number, neural_network);
    neural_network->forward_propagate(batch.get_inputs(), forward_propagation);

    BackPropagation back_propagation(samples_number, &loss);
    back_propagation.error = loss.calculate_error(batch, forward_propagation).error;

    return back_propagation.error;
}

VectorR calculate_gradient(Loss& loss)
{
    NeuralNetwork* neural_network = loss.get_neural_network();
    Dataset*       dataset        = loss.get_dataset();

    if (!neural_network) throw runtime_error("calculate_gradient: neural network is not set.");
    if (!dataset)        throw runtime_error("calculate_gradient: dataset is not set.");

    const Index samples_number = dataset->get_samples_number("Training");

    const vector<Index> training_indices         = dataset->get_sample_indices("Training");
    const vector<Index> input_feature_indices    = dataset->get_feature_indices("Input");
    const vector<Index> decoder_feature_indices  = dataset->get_feature_indices("Decoder");
    const vector<Index> target_feature_indices   = dataset->get_feature_indices("Target");

    Batch batch(samples_number, dataset);
    batch.fill(training_indices, input_feature_indices, decoder_feature_indices, target_feature_indices);

    ForwardPropagation forward_propagation(samples_number, neural_network);
    BackPropagation    back_propagation(samples_number, &loss);

    Map<const VectorR, AlignedMax> parameters(neural_network->get_parameters_data(),
                                               neural_network->get_parameters_size());

    neural_network->forward_propagate(batch.get_inputs(), parameters, forward_propagation);

    loss.back_propagate(batch, forward_propagation, back_propagation);

    return Map<const VectorR, AlignedMax>(back_propagation.gradient.as<float>(),
                                          back_propagation.gradient.size_in_floats());
}

VectorR calculate_numerical_gradient(Loss& loss)
{
    NeuralNetwork* neural_network = loss.get_neural_network();
    Dataset*       dataset        = loss.get_dataset();

    if (!neural_network) throw runtime_error("calculate_numerical_gradient: neural network is not set.");
    if (!dataset)        throw runtime_error("calculate_numerical_gradient: dataset is not set.");

    const Index samples_number = dataset->get_samples_number("Training");

    const vector<Index> training_indices         = dataset->get_sample_indices("Training");
    const vector<Index> input_feature_indices    = dataset->get_feature_indices("Input");
    const vector<Index> decoder_feature_indices  = dataset->get_feature_indices("Decoder");
    const vector<Index> target_feature_indices   = dataset->get_feature_indices("Target");

    Batch batch(samples_number, dataset);
    batch.fill(training_indices, input_feature_indices, decoder_feature_indices, target_feature_indices);

    ForwardPropagation forward_propagation(samples_number, neural_network);
    BackPropagation    back_propagation(samples_number, &loss);

    VectorMap parameters(neural_network->get_parameters_data(),
                         neural_network->get_parameters_size());

    const Index parameters_number = parameters.size();

    VectorR perturbed = parameters;
    VectorR numerical_gradient = VectorR::Zero(parameters_number);

    for (Index i = 0; i < parameters_number; ++i)
    {
        const float h = Loss::calculate_h(parameters(i));

        perturbed(i) += h;
        neural_network->forward_propagate(batch.get_inputs(), perturbed, forward_propagation);
        back_propagation.error = loss.calculate_error(batch, forward_propagation).error;
        const float error_forward = back_propagation.error;

        perturbed(i) -= 2.0f * h;
        neural_network->forward_propagate(batch.get_inputs(), perturbed, forward_propagation);
        back_propagation.error = loss.calculate_error(batch, forward_propagation).error;
        const float error_backward = back_propagation.error;

        perturbed(i) += h;

        numerical_gradient(i) = (error_forward - error_backward) / float(2 * h);
    }

    return numerical_gradient;
}

VectorR calculate_numerical_input_deltas(Loss& loss)
{
    NeuralNetwork* neural_network = loss.get_neural_network();
    Dataset*       dataset        = loss.get_dataset();

    if (!neural_network) throw runtime_error("calculate_numerical_input_deltas: neural network is not set.");
    if (!dataset)        throw runtime_error("calculate_numerical_input_deltas: dataset is not set.");

    const Index samples_number = dataset->get_samples_number("Training");
    const Index values_number  = neural_network->get_inputs_number() * samples_number;

    const vector<Index> sample_indices         = dataset->get_sample_indices("Training");
    const vector<Index> input_feature_indices  = dataset->get_feature_indices("Input");
    const vector<Index> target_feature_indices = dataset->get_feature_indices("Target");

    Batch batch(samples_number, dataset);
    batch.fill(sample_indices, input_feature_indices, {}, target_feature_indices);

    ForwardPropagation forward_propagation(samples_number, neural_network);
    BackPropagation    back_propagation(samples_number, &loss);

    VectorR numerical_inputs_gradients = VectorR::Zero(values_number);

    const vector<TensorView>& input_views = batch.get_inputs();
    TensorMap4 inputs_vector = input_views[0].as_tensor<4>();

    for (Index i = 0; i < values_number; ++i)
    {
        const float h = Loss::calculate_h(inputs_vector(i));

        input_views[0].as<float>()[i] += h;
        neural_network->forward_propagate(input_views, forward_propagation);
        back_propagation.error = loss.calculate_error(batch, forward_propagation).error;
        const float error_forward = back_propagation.error;

        input_views[0].as<float>()[i] -= 2 * h;
        neural_network->forward_propagate(input_views, forward_propagation);
        back_propagation.error = loss.calculate_error(batch, forward_propagation).error;
        const float error_backward = back_propagation.error;

        input_views[0].as<float>()[i] += h;

        numerical_inputs_gradients(i) = (error_forward - error_backward) / float(2 * h);
    }

    return numerical_inputs_gradients;
}

MatrixR calculate_numerical_hessian(Loss& loss)
{
    // Stub matching the original Loss method — to be filled in when needed.
    const VectorR gradient = calculate_numerical_gradient(loss);
    const Index parameters_number = gradient.size();
    return MatrixR::Zero(parameters_number, parameters_number);
}

MatrixR calculate_inverse_hessian(Loss& loss)
{
    MatrixR numerical_hessian = calculate_numerical_hessian(loss);
    const Index parameters_number = numerical_hessian.rows();

    using MatrixType = Matrix<float, Dynamic, Dynamic, ColMajor>;
    Map<MatrixType> hessian_map(numerical_hessian.data(), parameters_number, parameters_number);

    FullPivLU<MatrixType> const hessian_decomposition(hessian_map);

    if (!hessian_decomposition.isInvertible())
    {
        MatrixType hessian_damped = hessian_map
            + MatrixType::Identity(parameters_number, parameters_number) * 1e-4f;

        FullPivLU<MatrixType> const hessian_decomposition_damped(hessian_damped);

        const MatrixType hessian_map_inverse = hessian_decomposition_damped.inverse();

        MatrixR hessian_inverse(parameters_number, parameters_number);
        Map<MatrixType>(hessian_inverse.data(), parameters_number, parameters_number) = hessian_map_inverse;

        return hessian_inverse;
    }

    const MatrixType hessian_map_inverse = hessian_decomposition.inverse();
    MatrixR hessian_inverse(parameters_number, parameters_number);
    Map<MatrixType>(hessian_inverse.data(), parameters_number, parameters_number) = hessian_map_inverse;

    return hessian_inverse;
}

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
