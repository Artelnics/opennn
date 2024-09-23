//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   C R O S S   E N T R O P Y   E R R O R   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "cross_entropy_error.h"
#include "neural_network_forward_propagation.h"
#include "back_propagation.h"

namespace opennn
{

CrossEntropyError::CrossEntropyError() : LossIndex()
{
}


CrossEntropyError::CrossEntropyError(NeuralNetwork* new_neural_network, DataSet* new_data_set)
    : LossIndex(new_neural_network, new_data_set)
{
}


void CrossEntropyError::calculate_error(const Batch& batch,
                     const ForwardPropagation& forward_propagation,
                     BackPropagation& back_propagation) const
{      
    const Index outputs_number = neural_network->get_outputs_number();

    (outputs_number == 1)
        ? calculate_binary_error(batch, forward_propagation, back_propagation)
        : calculate_multiple_error(batch, forward_propagation, back_propagation);
}


void CrossEntropyError::calculate_binary_error(const Batch& batch,
                                               const ForwardPropagation& forward_propagation,
                                               BackPropagation& back_propagation) const
{
    // Batch

    const Index batch_samples_number = batch.get_batch_samples_number();

    const pair<type*, dimensions> targets_pair = batch.get_targets_pair();

    const TensorMap<Tensor<type, 2>> targets(targets_pair.first, targets_pair.second[0], targets_pair.second[1]);

    // Forward propagation

    const pair<type*, dimensions> outputs_pair = forward_propagation.get_last_trainable_layer_outputs_pair();

    const TensorMap<Tensor<type, 2>> outputs(outputs_pair.first, outputs_pair.second[0], outputs_pair.second[1]);

    // Back propagation

    type& error = back_propagation.error;

    Tensor<type, 0> cross_entropy_error;

    cross_entropy_error.device(*thread_pool_device) 
        = ((targets * outputs.log() + (type(1) - targets) * ((type(1) - outputs).log())).sum()) / type(-batch_samples_number);

    error = cross_entropy_error();

    if(isnan(error)) throw runtime_error("\nError is NAN.");
}


void CrossEntropyError::calculate_multiple_error(const Batch& batch,
                                                 const ForwardPropagation& forward_propagation,
                                                 BackPropagation& back_propagation) const
{
    // Batch

    const Index batch_samples_number = batch.get_batch_samples_number();
    
    const pair<type*, dimensions> targets_pair = batch.get_targets_pair();

    const TensorMap<Tensor<type, 2>> targets(targets_pair.first, targets_pair.second[0], targets_pair.second[1]);

    // Forward propagation

    const pair<type*, dimensions> outputs_pair = forward_propagation.get_last_trainable_layer_outputs_pair();
    
    const TensorMap<Tensor<type, 2>> outputs(outputs_pair.first, outputs_pair.second[0], outputs_pair.second[1]);

    // Back propagation

    const Index layers_number = back_propagation.neural_network.layers.size();

    ProbabilisticLayerBackPropagation* probabilistic_layer_back_propagation =
        static_cast<ProbabilisticLayerBackPropagation*>(back_propagation.neural_network.layers(layers_number - 1));

    probabilistic_layer_back_propagation->targets = targets;

    type& error = back_propagation.error;

    Tensor<type, 0> cross_entropy_error;

    cross_entropy_error.device(*thread_pool_device) = (targets*outputs.log()).sum() / type(-1/**batch_samples_number*/);

    error = cross_entropy_error();

    if(isnan(error)) throw runtime_error("\nError is NAN.");
}


void CrossEntropyError::calculate_output_delta(const Batch& batch,
                                               ForwardPropagation& forward_propagation,
                                               BackPropagation& back_propagation) const
{
     const Index outputs_number = neural_network->get_outputs_number();

     (outputs_number == 1)
         ? calculate_binary_output_delta(batch, forward_propagation, back_propagation)
         : calculate_multiple_output_delta(batch, forward_propagation, back_propagation);
}


void CrossEntropyError::calculate_binary_output_delta(const Batch& batch,
                                                      ForwardPropagation& forward_propagation,
                                                      BackPropagation& back_propagation) const
{
    // Neural network

    const Index last_trainable_layer_index = neural_network->get_last_trainable_layer_index();

    // Batch

    const Index batch_samples_number = batch.get_batch_samples_number();

    const pair<type*, dimensions> targets_pair = batch.get_targets_pair();

    const TensorMap<Tensor<type, 2>> targets(targets_pair.first, targets_pair.second[0], targets_pair.second[1]);

    // Forward propagation

    const ProbabilisticLayerForwardPropagation* probabilistic_layer_forward_propagation
            = static_cast<ProbabilisticLayerForwardPropagation*>(forward_propagation.layers(last_trainable_layer_index));

    const Tensor<type, 2>& outputs = probabilistic_layer_forward_propagation->outputs;

    // Back propagation

    const pair<type*, dimensions> output_deltas_pair = back_propagation.get_output_deltas_pair();

    TensorMap<Tensor<type, 2>> output_deltas(output_deltas_pair.first, output_deltas_pair.second[0], output_deltas_pair.second[1]);

    output_deltas.device(*thread_pool_device)
            = (-targets/outputs + (type(1) - targets)/(type(1) - outputs))/type(batch_samples_number);
}


void CrossEntropyError::calculate_multiple_output_delta(const Batch& batch,
                                                        ForwardPropagation& forward_propagation,
                                                        BackPropagation& back_propagation) const
{
    const Index last_trainable_layer_index = neural_network->get_last_trainable_layer_index();

    const Index batch_samples_number = batch.get_batch_samples_number();

    const pair<type*, dimensions> targets_pair = batch.get_targets_pair();

    const TensorMap<Tensor<type, 2>> targets(targets_pair.first, targets_pair.second[0], targets_pair.second[1]);

    const pair<type*, dimensions> outputs_pair = forward_propagation.layers(last_trainable_layer_index)->get_outputs_pair();

    const TensorMap<Tensor<type, 2>> outputs(outputs_pair.first, outputs_pair.second[0], outputs_pair.second[1]);

    const pair<type*, dimensions> output_deltas_pair = back_propagation.get_output_deltas_pair();

    TensorMap<Tensor<type, 2>> output_deltas(output_deltas_pair.first, output_deltas_pair.second[0], output_deltas_pair.second[1]);

    const type coefficient = -type(1) / type(batch_samples_number);

    output_deltas.device(*thread_pool_device) = (targets/outputs)*coefficient;
}


string CrossEntropyError::get_loss_method() const
{
    return "CROSS_ENTROPY_ERROR";
}


string CrossEntropyError::get_error_type_text() const
{
    return "Cross entropy error";
}


void CrossEntropyError::to_XML(tinyxml2::XMLPrinter& file_stream) const
{
    // Error type

    file_stream.OpenElement("CrossEntropyError");

    file_stream.CloseElement();
}


void CrossEntropyError::from_XML(const tinyxml2::XMLDocument& document)
{
    const tinyxml2::XMLElement* root_element = document.FirstChildElement("CrossEntropyError");

    if(!root_element)
        throw runtime_error("Cross entropy error element is nullptr.\n");

    // Regularization

    tinyxml2::XMLDocument regularization_document;

    const tinyxml2::XMLElement* regularization_element = root_element->FirstChildElement("Regularization");

    tinyxml2::XMLNode* element_clone = regularization_element->DeepClone(&regularization_document);

    regularization_document.InsertFirstChild(element_clone);

    regularization_from_XML(regularization_document);
}

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2024 Artificial Intelligence Techniques, SL.
//
// This library is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 2.1 of the License, or any later version.
//
// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.

// You should have received a copy of the GNU Lesser General Public
// License along with this library; if not, write to the Free Software
// Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
