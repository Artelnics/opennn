//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   W E I G T H E D   S Q U A R E D   E R R O R   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "registry.h"
#include "tensor_utilities.h"
#include "dataset.h"
#include "variable.h"
#include "neural_network.h"
#include "weighted_squared_error.h"

namespace opennn
{

WeightedSquaredError::WeightedSquaredError(const NeuralNetwork* new_neural_network, const Dataset* new_dataset)
    : Loss(new_neural_network, new_dataset)
{
    set_default();
}


void WeightedSquaredError::set(NeuralNetwork* new_neural_network, Dataset* new_dataset)
{
    set_neural_network(new_neural_network);
    set_dataset(new_dataset);
}


type WeightedSquaredError::get_positives_weight() const
{
    return positives_weight;
}


type WeightedSquaredError::get_negatives_weight() const
{
    return negatives_weight;
}


type WeightedSquaredError::get_normalizaton_coefficient() const
{
    return normalization_coefficient;
}


void WeightedSquaredError::set_default()
{
    name = "WeightedSquaredError";

    negatives_weight = type(-1.0);
    positives_weight = type(-1.0);
    normalization_coefficient = type(-1.0);

    if(!has_dataset() || dataset->get_samples_number() == 0)
        return;

    set_weights();
    set_normalization_coefficient();
}


void WeightedSquaredError::set_positives_weight(const type new_positives_weight)
{
    positives_weight = new_positives_weight;
}


void WeightedSquaredError::set_negatives_weight(const type new_negatives_weight)
{
    negatives_weight = new_negatives_weight;
}


void WeightedSquaredError::set_weights(const type new_positives_weight, type new_negatives_weight)
{
    positives_weight = new_positives_weight;
    negatives_weight = new_negatives_weight;
}


void WeightedSquaredError::set_weights()
{
    if(!dataset) return;

    const vector<Variable>& target_variables
        = dataset->get_variables("Target");

    if(target_variables.empty())
    {
        positives_weight = type(1);
        negatives_weight = type(1);

        return;
    }

    if(target_variables.size() == 1 && target_variables[0].is_binary())
    {
        const VectorI target_distribution = dataset->calculate_target_distribution();

        const Index negatives = target_distribution[0];
        const Index positives = target_distribution[1];

        if(positives == 0 || negatives == 0)
        {
            positives_weight = type(1);
            negatives_weight = type(1);

            return;
        }

        negatives_weight = type(1);
        positives_weight = type(negatives)/type(positives);

        return;
    }

    positives_weight = type(1);
    negatives_weight = type(1);
}


void WeightedSquaredError::set_normalization_coefficient()
{
    if(!dataset)
    {
        normalization_coefficient = type(1);
        return;
    }

    const vector<Variable>& target_variables
        = dataset->get_variables("Target");

    if(target_variables.empty())
    {
        normalization_coefficient = type(1);
        return;
    }

    if(target_variables.size() == 1 && target_variables[0].is_binary())
    {
        const vector<Index> target_feature_indices = dataset->get_feature_indices("Target");

        const Index negatives = dataset->calculate_used_negatives(target_feature_indices[0]);

        normalization_coefficient = type(negatives)*negatives_weight*type(0.5);

        return;
    }

    normalization_coefficient = type(1);
}


void WeightedSquaredError::set_dataset(const Dataset* new_dataset)
{
    dataset = const_cast<Dataset*>(new_dataset);

    set_weights();

    set_normalization_coefficient();
}


void WeightedSquaredError::calculate_error(const Batch& batch,
                                           const ForwardPropagation& forward_propagation,
                                           BackPropagation& back_propagation) const
{
    // Dataset

    const Index total_samples_number = dataset->get_samples_number();

    // Batch

    const Index samples_number = batch.get_samples_number();

    const MatrixMap targets = matrix_map(batch.get_targets());

    // Forward propagation

    const MatrixMap outputs = matrix_map(forward_propagation.get_last_trainable_layer_outputs());

    // Back propagation

    MatrixR& errors = back_propagation.errors;
    MatrixR& errors_weights = back_propagation.errors_weights;
    type& error = back_propagation.error;

    errors = outputs - targets;

    errors_weights.array() = (targets.array() == type(0)).select(
        MatrixR::Constant(targets.rows(), targets.cols(), negatives_weight),
        MatrixR::Constant(targets.rows(), targets.cols(), positives_weight)
        );

    const type coefficient = type(total_samples_number) / (type(samples_number) * normalization_coefficient);

    error = (errors.array().square() * back_propagation.errors_weights.array()).sum() * coefficient;
}


void WeightedSquaredError::calculate_output_gradients(const Batch& batch,
                                                  ForwardPropagation&,
                                                  BackPropagation& back_propagation) const
{
    // Dataset

    const Index total_samples_number = dataset->get_samples_number();

    // Batch

    const Index batch_size = batch.target_shape[0];

    // Back propagation

    const MatrixR& errors = back_propagation.errors;

    const MatrixR& errors_weights = back_propagation.errors_weights;

    MatrixMap output_gradients = matrix_map(back_propagation.get_output_gradients());

    const type coefficient = type(2 * total_samples_number) / (type(batch_size) * normalization_coefficient);

    output_gradients.array() = coefficient * (errors_weights.array() * errors.array());
}


void WeightedSquaredError::to_XML(XMLPrinter& printer) const
{
    printer.OpenElement("WeightedSquaredError");

    add_xml_element(printer, "PositivesWeight", to_string(positives_weight));
    add_xml_element(printer, "NegativesWeight", to_string(negatives_weight));

    printer.CloseElement();
}


void WeightedSquaredError::from_XML(const XMLDocument& document)
{
    const XMLElement* root_element = document.FirstChildElement("WeightedSquaredError");

    if(!root_element)
        throw runtime_error("Weighted squared element is nullptr.\n");

    set_positives_weight(read_xml_type(root_element, "PositivesWeight"));
    set_negatives_weight(read_xml_type(root_element, "NegativesWeight"));
}


#ifdef CUDA

void WeightedSquaredError::calculate_error(const BatchCuda& batch,
                                           const ForwardPropagationCuda& forward_propagation,
                                           BackPropagationCuda& back_propagation) const
{
    // Dataset

    const Index total_samples_number = dataset->get_samples_number();

    // Batch

    const Index samples_number = batch.get_samples_number();
    const type* targets = batch.targets_device.data;

    // Forward propagation

    const float* outputs = forward_propagation.get_last_trainable_layer_outputs_device().data;

    // Back propagation

    type& error = back_propagation.error;
    float* error_device = back_propagation.error_device;
    float* errors = back_propagation.errors;

    const size_t size = samples_number * forward_propagation.layers[neural_network->get_last_trainable_layer_index()]->layer->get_outputs_number();

    const cudnnTensorDescriptor_t output_tensor_descriptor = back_propagation.output_gradients.get_descriptor();
    const cudnnTensorDescriptor_t output_reduce_tensor_descriptor = back_propagation.output_reduce_tensor_descriptor;
    const cudnnReduceTensorDescriptor_t reduce_tensor_descriptor = back_propagation.reduce_tensor_descriptor;

    calculate_weighted_squared_error_cuda(size, errors, targets, outputs, positives_weight, negatives_weight);

    cudnnReduceTensor(get_cudnn_handle(),
                      reduce_tensor_descriptor,
                      nullptr,
                      0,
                      back_propagation.workspace,
                      back_propagation.workspace_size,
                      &alpha_one,
                      output_tensor_descriptor,
                      errors,
                      &beta_zero,
                      output_reduce_tensor_descriptor,
                      error_device);

    CHECK_CUDA(cudaMemcpy(&error, error_device, sizeof(float), cudaMemcpyDeviceToHost));

    const type coefficient = type(total_samples_number) / (type(samples_number) * normalization_coefficient);
    error *= coefficient;

    if (isnan(error)) throw runtime_error("\nError is NAN.");
}


void WeightedSquaredError::calculate_output_gradients(const BatchCuda& batch,
                                                      ForwardPropagationCuda& forward_propagation,
                                                      BackPropagationCuda& back_propagation) const
{
    // Dataset

    const Index total_samples_number = dataset->get_samples_number();

    // Batch

    const Index batch_size = batch.target_shape[0];
    const type* targets = batch.targets_device.data;

    // Forward propagation

    const float* outputs = forward_propagation.get_last_trainable_layer_outputs_device().data;

    // Back propagation

    float* output_gradients = back_propagation.get_output_gradients_device().data;

    const size_t size = batch_size * forward_propagation.layers[neural_network->get_last_trainable_layer_index()]->layer->get_outputs_number();

    const type coefficient = type(2 * total_samples_number) / (type(batch_size) * normalization_coefficient);

    calculate_weighted_squared_error_delta_cuda(size, output_gradients, targets, outputs, positives_weight, negatives_weight, coefficient);
}

#endif

REGISTER(Loss, WeightedSquaredError, "WeightedSquaredError");

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// This library is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 2.1 of the License, or any later version.
// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.
// You should have received a copy of the GNU Lesser General Public
// License along with this library; if not, write to the Free Software
// Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
