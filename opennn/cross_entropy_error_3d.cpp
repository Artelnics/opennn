//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   C R O S S   E N T R O P Y   E R R O R   3 D   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "cross_entropy_error_3d.h"
#include "neural_network_forward_propagation.h"
#include "loss_index_back_propagation.h"

namespace opennn
{

/// Default constructor.
/// It creates a default cross-entropy error term object,
/// which is not associated with any neural network and not measured on any data set.
/// It also initializes all the rest of the class members to their default values.

CrossEntropyError3D::CrossEntropyError3D() : LossIndex()
{
}


/// Neural network and data set constructor.
/// It creates a cross-entropy error term object associated with a neural network and measured on a data set.
/// It also initializes all the rest of the class members to their default values:
/// @param new_neural_network_pointer: Pointer to a neural network object.
/// @param new_data_set_pointer: Pointer to a data set object.

CrossEntropyError3D::CrossEntropyError3D(NeuralNetwork* new_neural_network_pointer, DataSet* new_data_set_pointer)
    : LossIndex(new_neural_network_pointer, new_data_set_pointer)
{
}


/// \brief CrossEntropyError3D::calculate_error.
/// \param batch
/// \param forward_propagation
/// \param back_propagation

void CrossEntropyError3D::calculate_error(const DataSetBatch& batch,
                     const ForwardPropagation& forward_propagation,
                     LossIndexBackPropagation& back_propagation) const
{
    const Index batch_samples_number = batch.get_batch_samples_number();

    const Index last_trainable_layer_index = neural_network_pointer->get_last_trainable_layer_index();

    const pair<type*, dimensions> outputs_pair = forward_propagation.layers(last_trainable_layer_index)->get_outputs_pair();

    const TensorMap<Tensor<type, 3>> outputs_map(outputs_pair.first, outputs_pair.second[0][0],
                                                                     outputs_pair.second[0][1],
                                                                     outputs_pair.second[0][2]);

    const pair<type*, dimensions> targets_pair = batch.get_targets_pair();

    const TensorMap<Tensor<type, 3>> targets_map(targets_pair.first, targets_pair.second[0][0],
                                                                     targets_pair.second[0][1],
                                                                     targets_pair.second[0][2]);

    Tensor<type, 0> cross_entropy_error;
    cross_entropy_error.device(*thread_pool_device) = -( targets_map * outputs_map.log() ).sum();

    back_propagation.error = cross_entropy_error()/type(batch_samples_number);

    if(is_nan(back_propagation.error))
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: cross_entropy_error class.\n"
               << "void calculate_multiple_error(const DataSetBatch&, const NeuralNetworkForwardPropagation&,LossIndexBackPropagation&) method.\n"
               << "NAN values found in back propagation error.";

        throw invalid_argument(buffer.str());
    }
}

void CrossEntropyError3D::calculate_error(pair<type*, dimensions>& outputs_pair, const Tensor<type, 3>& targets, type& error) const
{
    TensorMap<Tensor<type, 3>> outputs_map(outputs_pair.first, outputs_pair.second[0][0],
                                                               outputs_pair.second[0][1],
                                                               outputs_pair.second[0][2]);

    Index batch_samples_number = outputs_map.dimension(0);

    Tensor<type, 0> cross_entropy_error;
    cross_entropy_error.device(*thread_pool_device) = -(targets * outputs_map.log()).sum();

    error = cross_entropy_error() / type(batch_samples_number);
}

Tensor<type, 1> CrossEntropyError3D::calculate_numerical_gradient(const Tensor<type, 3>& inputs, const Tensor<type, 3>& targets)
{
    Index samples_number = inputs.dimension(0);

    pair<type*, dimensions> inputs_pair = get_pair(inputs);
    
    ForwardPropagation forward_propagation(samples_number, neural_network_pointer);
    
    pair<type*, dimensions> outputs_pair;

    const Tensor<type, 1> parameters = neural_network_pointer->get_parameters();
    const Index last_trainable_layer_index = neural_network_pointer->get_last_trainable_layer_index();

    const Index parameters_number = parameters.size();

    type h;
    Tensor<type, 1> parameters_forward(parameters);
    Tensor<type, 1> parameters_backward(parameters);

    type error_forward;
    type error_backward;

    Tensor<type, 1> numerical_gradient(parameters_number);
    numerical_gradient.setConstant(type(0));

    for (Index i = 0; i < parameters_number; i++)
    {
        h = calculate_h(parameters(i));

        parameters_forward(i) += h;
        
        neural_network_pointer->forward_propagate(inputs_pair,
            parameters_forward,
            forward_propagation);

        outputs_pair = forward_propagation.layers(last_trainable_layer_index)->get_outputs_pair();

        calculate_error(outputs_pair, targets, error_forward);

        parameters_forward(i) -= h;

        parameters_backward(i) -= h;

        neural_network_pointer->forward_propagate(inputs_pair,
                                                  parameters_backward,
                                                  forward_propagation);

        outputs_pair = forward_propagation.layers(last_trainable_layer_index)->get_outputs_pair();

        calculate_error(outputs_pair, targets, error_backward);

        parameters_backward(i) += h;

        numerical_gradient(i) = (error_forward - error_backward) / (type(2) * h);
    }

    return numerical_gradient;
}


void CrossEntropyError3D::calculate_output_delta(const DataSetBatch& batch,
                                               ForwardPropagation& forward_propagation,
                                               LossIndexBackPropagation& back_propagation) const
{
    const Index trainable_layers_number = neural_network_pointer->get_trainable_layers_number();
    const Index last_trainable_layer_index = neural_network_pointer->get_last_trainable_layer_index();

    ProbabilisticLayer3DBackPropagation* probabilistic_layer_back_propagation
            = static_cast<ProbabilisticLayer3DBackPropagation*>(back_propagation.neural_network.layers(trainable_layers_number-1));

    const Index batch_samples_number = batch.get_batch_samples_number();

    const pair<type*, dimensions> outputs_pair = forward_propagation.layers(last_trainable_layer_index)->get_outputs_pair();

    const TensorMap<Tensor<type, 3>> outputs_map(outputs_pair.first, outputs_pair.second[0][0],
                                                                     outputs_pair.second[0][1],
                                                                     outputs_pair.second[0][2]);

    const pair<type*, dimensions> targets_pair = batch.get_targets_pair();

    const TensorMap<Tensor<type, 3>> targets_map(targets_pair.first, targets_pair.second[0][0],
                                                                     targets_pair.second[0][1],
                                                                     targets_pair.second[0][2]);

    Tensor<type, 3>& deltas = probabilistic_layer_back_propagation->deltas;

    deltas.device(*thread_pool_device) = (-targets_map/outputs_map)/type(batch_samples_number);
}


/// Returns a string with the name of the cross-entropy error loss type, "CROSS_ENTROPY_ERROR".

string CrossEntropyError3D::get_error_type() const
{
    return "CROSS_ENTROPY_ERROR_3D";
}


/// Returns a string with the name of the cross-entropy error loss type in text format.

string CrossEntropyError3D::get_error_type_text() const
{
    return "Cross entropy error 3D";
}


/// Serializes the cross-entropy error object into an XML document of the TinyXML library without keeping the DOM tree in memory.
/// See the OpenNN manual for more information about the format of this document

void CrossEntropyError3D::write_XML(tinyxml2::XMLPrinter& file_stream) const
{
    // Error type

    file_stream.OpenElement("CrossEntropyError3D");

    file_stream.CloseElement();
}


/// Deserializes a TinyXML document into this cross-entropy object.
/// @param document TinyXML document containing the member data.

void CrossEntropyError3D::from_XML(const tinyxml2::XMLDocument& document)
{
    const tinyxml2::XMLElement* root_element = document.FirstChildElement("CrossEntropyError3D");

    if(!root_element)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: CrossEntropyError3D class.\n"
               << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
               << "Cross entropy error element is nullptr.\n";

        throw invalid_argument(buffer.str());
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
