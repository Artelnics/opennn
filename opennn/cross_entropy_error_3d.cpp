//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   C R O S S   E N T R O P Y   E R R O R   3 D   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "cross_entropy_error_3d.h"
#include "probabilistic_layer_3d.h"
#include "neural_network_forward_propagation.h"
#include "back_propagation.h"

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
/// @param new_neural_network: Pointer to a neural network object.
/// @param new_data_set: Pointer to a data set object.

CrossEntropyError3D::CrossEntropyError3D(NeuralNetwork* new_neural_network, DataSet* new_data_set)
    : LossIndex(new_neural_network, new_data_set)
{
}


/// \brief CrossEntropyError3D::calculate_error.
/// \param batch
/// \param forward_propagation
/// \param back_propagation

void CrossEntropyError3D::calculate_error(const Batch& batch,
                                          const ForwardPropagation& forward_propagation,
                                          BackPropagation& back_propagation) const
{
    // Batch

    const Index batch_samples_number = batch.get_batch_samples_number();

    const pair<type*, dimensions> targets_pair = batch.get_targets_pair();

    const Index outputs_number = targets_pair.second[1];

    const TensorMap<Tensor<type, 2>> targets(targets_pair.first,
                                             batch_samples_number,
                                             outputs_number);
    
    // Forward propagation
    
    const pair<type*, dimensions> outputs_pair = forward_propagation.get_last_trainable_layer_outputs_pair();

    const Index outputs_depth = outputs_pair.second[2];
    
    const TensorMap<Tensor<type, 3>> outputs(outputs_pair.first, 
                                             batch_samples_number,
                                             outputs_number,
                                             outputs_depth);
    
    // Back propagation

    const Index layers_number = back_propagation.neural_network.layers.size();
    
    ProbabilisticLayer3DBackPropagation* probabilistic_layer_3d_back_propagation =
        static_cast<ProbabilisticLayer3DBackPropagation*>(back_propagation.neural_network.layers(layers_number - 1));
        
    probabilistic_layer_3d_back_propagation->targets = targets;
    
    Tensor<type, 2>& errors = back_propagation.errors;
    Tensor<type, 2>& predictions = back_propagation.predictions;
    Tensor<bool, 2>& matches = back_propagation.matches;
    Tensor<bool, 2>& mask = back_propagation.mask;
    bool& built_mask = back_propagation.built_mask;
    
    Tensor<type, 0> cross_entropy_error;

    if(!built_mask)
    {
        mask.device(*thread_pool_device) = targets != targets.constant(0);
        built_mask = true;
    }

    const Tensor<type, 0> mask_sum = mask.cast<type>().sum();

    #pragma omp parallel for

    for(Index i = 0; i < batch_samples_number; i++)
        for(Index j = 0; j < outputs_number; j++)
            errors(i, j) = -log(outputs(i, j, Index(targets(i, j))));

    errors.device(*thread_pool_device) = errors * mask.cast<type>();

    cross_entropy_error.device(*thread_pool_device) = errors.sum();

    back_propagation.error = cross_entropy_error(0) / mask_sum(0);

    // Masked accuracy
    
    predictions.device(*thread_pool_device) = outputs.argmax(2).cast<type>();

    matches.device(*thread_pool_device) = predictions == targets;
    
    matches.device(*thread_pool_device) = matches && mask;

    Tensor<type, 0> accuracy;

    accuracy.device(*thread_pool_device) = matches.cast<type>().sum() / mask_sum(0);

    back_propagation.accuracy = accuracy(0);
    
    if(isnan(back_propagation.error)) throw runtime_error("Error is NAN");
}


void CrossEntropyError3D::calculate_output_delta(const Batch& batch,
                                               ForwardPropagation& forward_propagation,
                                               BackPropagation& back_propagation) const
{
    // ProbabilisticLayer3D does not have deltas. Error combinations derivatives are calculated directly.
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
