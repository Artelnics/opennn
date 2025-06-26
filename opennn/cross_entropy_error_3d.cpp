//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   C R O S S   E N T R O P Y   E R R O R   3 D   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "registry.h"
#include "tensors.h"
#include "neural_network.h"
#include "cross_entropy_error_3d.h"
#include "probabilistic_layer_3d.h"

namespace opennn
{

CrossEntropyError3d::CrossEntropyError3d(NeuralNetwork* new_neural_network, Dataset* new_data_set)
    : LossIndex(new_neural_network, new_data_set)
{
}


void CrossEntropyError3d::calculate_binary_error(const Batch& batch,
                                                 const ForwardPropagation& forward_propagation,
                                                 BackPropagation& back_propagation) const
{
/*
    // Batch

    const Index samples_number = batch.get_samples_number();

    const pair<type*, dimensions> targets_pair = batch.get_target_pair();

    const TensorMap<Tensor<type, 2>> targets = tensor_map<2>(targets_pair);

    // Forward propagation

    const pair<type*, dimensions> outputs_pair = forward_propagation.get_last_trainable_layer_outputs_pair();

    const TensorMap<Tensor<type, 2>> outputs = tensor_map<2>(outputs_pair);

    // Back propagation

    Tensor<type, 0>& error = back_propagation.error;

    error.device(*thread_pool_device)
        = ((targets * (outputs + epsilon).log() + (type(1) - targets) * ((type(1) - outputs + epsilon).log())).sum()) / type(-samples_number);

    if(isnan(error())) throw runtime_error("\nError is NAN.");
*/
}


void CrossEntropyError3d::calculate_multiple_error(const Batch& batch,
                                                   const ForwardPropagation& forward_propagation,
                                                   BackPropagation& back_propagation) const
{
    // @todo
}


void CrossEntropyError3d::calculate_error(const Batch& batch,
                                          const ForwardPropagation& forward_propagation,
                                          BackPropagation& back_propagation) const
{
    const Index outputs_number = neural_network->get_outputs_number();

    outputs_number == 1
        ? calculate_binary_error(batch, forward_propagation, back_propagation)
        : calculate_multiple_error(batch, forward_propagation, back_propagation);
}


void CrossEntropyError3d::calculate_output_delta(const Batch&,
                                                 ForwardPropagation&,
                                                 BackPropagation&) const
{
    // Probabilistic3d does not have deltas. 
    // Error combinations derivatives are calculated directly.
}


string CrossEntropyError3d::get_name() const
{
    return "CrossEntropyError3d";
}


void CrossEntropyError3d::to_XML(XMLPrinter& file_stream) const
{
    file_stream.OpenElement("CrossEntropyError3d");
    file_stream.CloseElement();
}


void CrossEntropyError3d::from_XML(const XMLDocument& document)
{
    const XMLElement* root_element = document.FirstChildElement("CrossEntropyError3d");

    if(!root_element)
        throw runtime_error("Cross entropy error element is nullptr.\n");

    // Regularization

    XMLDocument regularization_document;
    const XMLElement* regularization_element = root_element->FirstChildElement("Regularization");
    regularization_document.InsertFirstChild(regularization_element->DeepClone(&regularization_document));
    regularization_from_XML(regularization_document);
}


#ifdef OPENNN_CUDA

void CrossEntropyError3d::calculate_error_cuda(const BatchCuda& batch_cuda,
                                               const ForwardPropagationCuda& forward_propagation_cuda,
                                               BackPropagationCuda& back_propagation_cuda) const
{
    throw runtime_error("CUDA calculate_error_cuda not implemented for loss index type: CrossEntropyError3d");
}


void CrossEntropyError3d::calculate_output_delta_cuda(const BatchCuda& batch_cuda,
                                                      ForwardPropagationCuda& forward_propagation_cuda,
                                                      BackPropagationCuda& back_propagation_cuda) const
{
    throw runtime_error("CUDA calculate_output_delta_cuda not implemented for loss index type: CrossEntropyError3d");
}

#endif

REGISTER(LossIndex, CrossEntropyError3d, "CrossEntropyError3d");

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2025 Artificial Intelligence Techniques, SL.
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
