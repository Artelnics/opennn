//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   C R O S S   E N T R O P Y   E R R O R   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "cross_entropy_error.h"

namespace OpenNN
{

/// Default constructor.
/// It creates a default cross entropy error term object,
/// which is not associated to any neural network and not measured on any data set.
/// It also initializes all the rest of class members to their default values.

CrossEntropyError::CrossEntropyError() : LossIndex()
{
}


/// Neural network and data set constructor.
/// It creates a cross entropy error term object associated to a neural network and measured on a data set.
/// It also initializes all the rest of class members to their default values:
/// @param new_neural_network_pointer: Pointer to a neural network object.
/// @param new_data_set_pointer: Pointer to a data set object.

CrossEntropyError::CrossEntropyError(NeuralNetwork* new_neural_network_pointer, DataSet* new_data_set_pointer)
    : LossIndex(new_neural_network_pointer, new_data_set_pointer)
{
}


/// Destructor.

CrossEntropyError::~CrossEntropyError()
{
}


////// \brief CrossEntropyError::calculate_error.
////// \param batch
////// \param forward_propagation
////// \param back_propagation
void CrossEntropyError::calculate_error(const DataSet::Batch& batch,
                     const NeuralNetwork::ForwardPropagation& forward_propagation,
                     LossIndex::BackPropagation& back_propagation) const
{
    const Index outputs_number = neural_network_pointer->get_outputs_number();

    if(outputs_number == 1)
    {
        calculate_binary_error(batch, forward_propagation, back_propagation);
    }
    else
    {
        calculate_multiple_error(batch, forward_propagation, back_propagation);
    }
}


void CrossEntropyError::calculate_binary_error(const DataSet::Batch& batch,
                                               const NeuralNetwork::ForwardPropagation& forward_propagation,
                                               LossIndex::BackPropagation& back_propagation) const
{
    const Index batch_samples_number = batch.inputs_2d.dimension(0);

    const Index trainable_layers_number = neural_network_pointer->get_trainable_layers_number();

    const Tensor<type, 2>& outputs = forward_propagation.layers[trainable_layers_number-1].activations_2d;
    const Tensor<type, 2>& targets = batch.targets_2d;

    Tensor<type, 0> cross_entropy_error;
    cross_entropy_error.device(*thread_pool_device) = -(targets*(outputs.log())).sum() - ((1-targets)*((1-outputs).log())).sum();

    back_propagation.error = cross_entropy_error()/static_cast<type>(batch_samples_number);
}


void CrossEntropyError::calculate_multiple_error(const DataSet::Batch& batch,
                                                 const NeuralNetwork::ForwardPropagation& forward_propagation,
                                                 LossIndex::BackPropagation& back_propagation) const
{
    const Index batch_samples_number = batch.inputs_2d.dimension(0);

    const Index trainable_layers_number = neural_network_pointer->get_trainable_layers_number();

    const Tensor<type, 2>& outputs = forward_propagation.layers[trainable_layers_number-1].activations_2d;
    const Tensor<type, 2>& targets = batch.targets_2d;

    Tensor<type, 0> cross_entropy_error;
    cross_entropy_error.device(*thread_pool_device) = -(targets*(outputs.log())).sum();

    back_propagation.error = cross_entropy_error()/static_cast<type>(batch_samples_number);
}


void CrossEntropyError::calculate_output_gradient(const DataSet::Batch& batch,
                               const NeuralNetwork::ForwardPropagation& forward_propagation,
                               BackPropagation& back_propagation) const
{
     #ifdef __OPENNN_DEBUG__

     check();

     #endif

     const Index outputs_number = neural_network_pointer->get_outputs_number();

     if(outputs_number == 1)
     {
         calculate_binary_output_gradient(batch, forward_propagation, back_propagation);
     }
     else
     {
         calculate_multiple_output_gradient(batch, forward_propagation, back_propagation);
     }
}


void CrossEntropyError::calculate_binary_output_gradient(const DataSet::Batch& batch,
                                                         const NeuralNetwork::ForwardPropagation& forward_propagation,
                                                         BackPropagation& back_propagation) const
{
    const Index batch_samples_number = batch.inputs_2d.dimension(0);

    const Index trainable_layers_number = neural_network_pointer->get_trainable_layers_number();

    const Tensor<type, 2>& targets = batch.targets_2d;
    const Tensor<type, 2>& outputs = forward_propagation.layers[trainable_layers_number-1].activations_2d;

    back_propagation.output_gradient.device(*thread_pool_device) = static_cast<type>(1)/static_cast<type>(batch_samples_number) *
            (static_cast<type>(-1)*(targets/outputs) + (static_cast<type>(1) - targets)/(static_cast<type>(1) - outputs));
}


void CrossEntropyError::calculate_multiple_output_gradient(const DataSet::Batch& batch,
                                                           const NeuralNetwork::ForwardPropagation& forward_propagation,
                                                           BackPropagation& back_propagation) const
{
    const Index batch_samples_number = batch.inputs_2d.dimension(0);

    const Index trainable_layers_number = neural_network_pointer->get_trainable_layers_number();

    const Tensor<type, 2>& targets = batch.targets_2d;
    const Tensor<type, 2>& outputs = forward_propagation.layers[trainable_layers_number-1].activations_2d;

    back_propagation.output_gradient.device(*thread_pool_device) = static_cast<type>(1)/static_cast<type>(batch_samples_number) *(-targets/outputs);
}

/// Returns a string with the name of the cross entropy error loss type, "CROSS_ENTROPY_ERROR".

string CrossEntropyError::get_error_type() const
{
    return "CROSS_ENTROPY_ERROR";
}


/// Returns a string with the name of the cross entropy error loss type in text format.

string CrossEntropyError::get_error_type_text() const
{
    return "Cross entropy error";
}


/// Serializes the cross entropy error object into a XML document of the TinyXML library without keep the DOM tree in memory.
/// See the OpenNN manual for more information about the format of this document

void CrossEntropyError::write_XML(tinyxml2::XMLPrinter& file_stream) const
{
    // Error type

    file_stream.OpenElement("CrossEntropyError");

    file_stream.CloseElement();
}


/// Deserializes a TinyXML document into this cross entropy object.
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

        throw logic_error(buffer.str());
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
// Copyright(C) 2005-2020 Artificial Intelligence Techniques, SL.
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
