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

CrossEntropyError3D::CrossEntropyError3D() : LossIndex()
{
}


CrossEntropyError3D::CrossEntropyError3D(NeuralNetwork* new_neural_network, DataSet* new_data_set)
    : LossIndex(new_neural_network, new_data_set)
{
}


void CrossEntropyError3D::calculate_error(const Batch& batch,
                                          const ForwardPropagation& forward_propagation,
                                          BackPropagation& back_propagation) const
{
    // Batch

    const Index batch_samples_number = batch.get_batch_samples_number();

    const pair<type*, dimensions> targets_pair = batch.get_targets_pair();

    const Index outputs_number = targets_pair.second[1];

    const TensorMap<Tensor<type, 2>> targets = tensor_map_2(targets_pair);
    
    // Forward propagation
    
    const pair<type*, dimensions> outputs_pair = forward_propagation.get_last_trainable_layer_outputs_pair();

    const TensorMap<Tensor<type, 3>> outputs = tensor_map_3(outputs_pair);
    
    // Back propagation

    const Index layers_number = back_propagation.neural_network.layers.size();
    
    ProbabilisticLayer3DBackPropagation* probabilistic_layer_3d_back_propagation =
        static_cast<ProbabilisticLayer3DBackPropagation*>(back_propagation.neural_network.layers[layers_number - 1].get());
        
    probabilistic_layer_3d_back_propagation->targets = targets;
    
    Tensor<type, 2>& errors = back_propagation.errors;
    Tensor<type, 2>& predictions = back_propagation.predictions;
    Tensor<bool, 2>& matches = back_propagation.matches;
    Tensor<bool, 2>& mask = back_propagation.mask;
    bool& built_mask = back_propagation.built_mask;
    
    Tensor<type, 0>& error = back_propagation.error;

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

    error.device(*thread_pool_device) = errors.sum() / mask_sum(0);

    // Masked accuracy
    
    predictions.device(*thread_pool_device) = outputs.argmax(2).cast<type>();

    matches.device(*thread_pool_device) = predictions == targets;
    
    matches.device(*thread_pool_device) = matches && mask;

    Tensor<type, 0> accuracy;

    accuracy.device(*thread_pool_device) = matches.cast<type>().sum() / mask_sum(0);

    back_propagation.accuracy = accuracy(0);
    
    if(isnan(error())) throw runtime_error("Error is NAN");
}


void CrossEntropyError3D::calculate_output_delta(const Batch& batch,
                                                 ForwardPropagation& forward_propagation,
                                                 BackPropagation& back_propagation) const
{
    // ProbabilisticLayer3D does not have deltas. Error combinations derivatives are calculated directly.
}



string CrossEntropyError3D::get_loss_method() const


{
    return "CROSS_ENTROPY_ERROR_3D";
}


string CrossEntropyError3D::get_error_type_text() const
{
    return "Cross entropy error 3D";
}


void CrossEntropyError3D::to_XML(tinyxml2::XMLPrinter& file_stream) const
{
    // Error type

    file_stream.OpenElement("CrossEntropyError3D");

    file_stream.CloseElement();
}


void CrossEntropyError3D::from_XML(const tinyxml2::XMLDocument& document)
{
    const tinyxml2::XMLElement* root_element = document.FirstChildElement("CrossEntropyError3D");

    if(!root_element)
        throw runtime_error("Cross entropy error element is nullptr.\n");

    // Regularization

    tinyxml2::XMLDocument regularization_document;
    const tinyxml2::XMLElement* regularization_element = root_element->FirstChildElement("Regularization");
    regularization_document.InsertFirstChild(regularization_element->DeepClone(&regularization_document));
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
