//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   C R O S S   E N T R O P Y   E R R O R   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "cross_entropy_error.h"
#include "neural_network_forward_propagation.h"
#include "loss_index_back_propagation.h"

namespace opennn
{

/// Default constructor.
/// It creates a default cross-entropy error term object,
/// which is not associated with any neural network and not measured on any data set.
/// It also initializes all the rest of the class members to their default values.

CrossEntropyError::CrossEntropyError() : LossIndex()
{
}


/// Neural network and data set constructor.
/// It creates a cross-entropy error term object associated with a neural network and measured on a data set.
/// It also initializes all the rest of the class members to their default values:
/// @param new_neural_network: Pointer to a neural network object.
/// @param new_data_set: Pointer to a data set object.

CrossEntropyError::CrossEntropyError(NeuralNetwork* new_neural_network, DataSet* new_data_set)
    : LossIndex(new_neural_network, new_data_set)
{
}


/// \brief CrossEntropyError::calculate_error.
/// \param batch
/// \param forward_propagation
/// \param back_propagation

void CrossEntropyError::calculate_error(const Batch& batch,
                     const ForwardPropagation& forward_propagation,
                     BackPropagation& back_propagation) const
{      
    const Index outputs_number = neural_network->get_outputs_number();

    if(outputs_number == 1)
    {
        calculate_binary_error(batch, forward_propagation, back_propagation);
    }
    else
    {
        calculate_multiple_error(batch, forward_propagation, back_propagation);
    }
}


void CrossEntropyError::calculate_binary_error(const Batch& batch,
                                               const ForwardPropagation& forward_propagation,
                                               BackPropagation& back_propagation) const
{
    // Batch

    const Index batch_samples_number = batch.get_batch_samples_number();

    const pair<type*, dimensions> targets_pair = batch.get_targets_pair();

    const TensorMap<Tensor<type, 2>> targets(targets_pair.first, targets_pair.second[0][0], targets_pair.second[0][1]);

    // Forward propagation

    const pair<type*, dimensions> outputs_pair = forward_propagation.get_last_trainable_layer_outputs_pair();

    const TensorMap<Tensor<type, 2>> outputs(outputs_pair.first, outputs_pair.second[0][0], outputs_pair.second[0][1]);

    // Back propagation

    type& error = back_propagation.error;

    Tensor<type, 0> cross_entropy_error;

    cross_entropy_error.device(*thread_pool_device) 
        = -((targets * outputs.log() + (type(1) - targets) * ((type(1) - outputs).log())).sum()) / type(batch_samples_number);

    error = cross_entropy_error();

    if (isnan(error)) throw runtime_error("Error is NAN.");
}


void CrossEntropyError::calculate_multiple_error(const Batch& batch,
                                                 const ForwardPropagation& forward_propagation,
                                                 BackPropagation& back_propagation) const
{
    // Batch

    const Index batch_samples_number = batch.get_batch_samples_number();

    const pair<type*, dimensions> targets_pair = batch.get_targets_pair();

    const TensorMap<Tensor<type, 2>> targets(targets_pair.first, targets_pair.second[0][0], targets_pair.second[0][1]);

    // Forward propagation

    const pair<type*, dimensions> outputs_pair = forward_propagation.get_last_trainable_layer_outputs_pair();

    const TensorMap<Tensor<type, 2>> outputs(outputs_pair.first, outputs_pair.second[0][0], outputs_pair.second[0][1]);

    // Back propagation

    type& error = back_propagation.error;

    Tensor<type, 0> cross_entropy_error;
    
    cross_entropy_error.device(*thread_pool_device) = -(targets*outputs.log()).sum();

    error = cross_entropy_error()/type(batch_samples_number);

    if (isnan(error)) throw runtime_error("Error is NAN.");
}


void CrossEntropyError::calculate_output_delta(const Batch& batch,
                                               ForwardPropagation& forward_propagation,
                                               BackPropagation& back_propagation) const
{
     const Index outputs_number = neural_network->get_outputs_number();

     if(outputs_number == 1)
     {
         calculate_binary_output_delta(batch, forward_propagation, back_propagation);
     }
     else
     {
         calculate_multiple_output_delta(batch, forward_propagation, back_propagation);
     }
}


void CrossEntropyError::calculate_binary_output_delta(const Batch& batch,
                                                      ForwardPropagation& forward_propagation,
                                                      BackPropagation& back_propagation) const
{
    const Index trainable_layers_number = neural_network->get_trainable_layers_number();
    const Index last_trainable_layer_index = neural_network->get_last_trainable_layer_index();

    const ProbabilisticLayerForwardPropagation* probabilistic_layer_forward_propagation
            = static_cast<ProbabilisticLayerForwardPropagation*>(forward_propagation.layers(last_trainable_layer_index));

    ProbabilisticLayerBackPropagation* probabilistic_layer_back_propagation
            = static_cast<ProbabilisticLayerBackPropagation*>(back_propagation.neural_network.layers(trainable_layers_number-1));

    const Index batch_samples_number = batch.get_batch_samples_number();

    const pair<type*, dimensions> targets_pair = batch.get_targets_pair();

    const TensorMap<Tensor<type, 2>> targets(targets_pair.first, targets_pair.second[0][0], targets_pair.second[0][1]);

    const Tensor<type, 2>& outputs = probabilistic_layer_forward_propagation->outputs;

    Tensor<type, 2>& deltas = probabilistic_layer_back_propagation->deltas;

    deltas.device(*thread_pool_device)
            = (-targets/outputs + (type(1) - targets)/(type(1) - outputs))/type(batch_samples_number);
}


void CrossEntropyError::calculate_multiple_output_delta(const Batch& batch,
                                                        ForwardPropagation& forward_propagation,
                                                        BackPropagation& back_propagation) const
{
    const Index trainable_layers_number = neural_network->get_trainable_layers_number();
    const Index last_trainable_layer_index = neural_network->get_last_trainable_layer_index();

    ProbabilisticLayerBackPropagation* probabilistic_layer_back_propagation
            = static_cast<ProbabilisticLayerBackPropagation*>(back_propagation.neural_network.layers(trainable_layers_number-1));

    const Index batch_samples_number = batch.get_batch_samples_number();

    const pair<type*, dimensions> targets_pair = batch.get_targets_pair();

    const TensorMap<Tensor<type, 2>> targets(targets_pair.first, targets_pair.second[0][0], targets_pair.second[0][1]);

    const pair<type*, dimensions> outputs_pair = forward_propagation.layers(last_trainable_layer_index)->get_outputs_pair();

    const TensorMap<Tensor<type, 2>> outputs(outputs_pair.first, outputs_pair.second[0][1], outputs_pair.second[0][1]);

    Tensor<type, 2>& deltas = probabilistic_layer_back_propagation->deltas;

    const type coefficient = -type(1) / type(batch_samples_number);

    deltas.device(*thread_pool_device) = (targets/outputs)*coefficient;
}


/// Returns a string with the name of the cross-entropy error loss type, "CROSS_ENTROPY_ERROR".

string CrossEntropyError::get_error_type() const
{
    return "CROSS_ENTROPY_ERROR";
}


/// Returns a string with the name of the cross-entropy error loss type in text format.

string CrossEntropyError::get_error_type_text() const
{
    return "Cross entropy error";
}


/// Serializes the cross-entropy error object into an XML document of the TinyXML library without keeping the DOM tree in memory.
/// See the OpenNN manual for more information about the format of this document

void CrossEntropyError::write_XML(tinyxml2::XMLPrinter& file_stream) const
{
    // Error type

    file_stream.OpenElement("CrossEntropyError");

    file_stream.CloseElement();
}


/// Deserializes a TinyXML document into this cross-entropy object.
/// @param document TinyXML document containing the member data.

void CrossEntropyError::from_XML(const tinyxml2::XMLDocument& document)
{
    const tinyxml2::XMLElement* root_element = document.FirstChildElement("CrossEntropyError");

    if(!root_element)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: CrossEntropyError class.\n"
               << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
               << "Cross entropy error element is nullptr.\n";

        throw runtime_error(buffer.str());
    }

    // Regularization

    tinyxml2::XMLDocument regularization_document;
    tinyxml2::XMLNode* element_clone;

    const tinyxml2::XMLElement* regularization_element = root_element->FirstChildElement("Regularization");

    element_clone = regularization_element->DeepClone(&regularization_document);

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
