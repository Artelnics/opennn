//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   C R O S S   E N T R O P Y   E R R O R   3 D   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "registry.h"
#include "dataset.h"
#include "neural_network.h"
#include "cross_entropy_error_3d.h"

namespace opennn
{

CrossEntropyError3d::CrossEntropyError3d(const NeuralNetwork* new_neural_network, const Dataset* new_dataset)
    : Loss(new_neural_network, new_dataset)
{
    name = "CrossEntropyError3d";
}


void CrossEntropyError3d::calculate_binary_error(const Batch& batch,
                                                 const ForwardPropagation& forward_propagation,
                                                 BackPropagation& back_propagation) const
{
    /*
    const TensorView targets_view = batch.get_targets();
    const MatrixMap targets = matrix_map(targets_view);

    const TensorView outputs_view = forward_propagation.get_last_trainable_layer_outputs();
    const TensorMap3 outputs = tensor_map<3>(outputs_view);

    const Index batch_size = outputs.dimension(0);
    const Index sequence_length = outputs.dimension(1);

    constexpr type epsilon = numeric_limits<type>::epsilon();

    // 3. Prepare Masking
    // In sequence tasks, we ignore padding. We assume target 0 is padding if built_mask is used.
    // If you want to include all tokens, you can skip the mask multiplication.

    back_propagation.mask.device(get_device()) = (targets != targets.constant(0.0f));
    const MatrixB& mask = back_propagation.mask;

    // 4. Reshape outputs to [Batch, Sequence] to match targets
    auto outputs_2d = outputs.reshape(array_2(batch_size, sequence_length));

    // 5. Calculate element-wise Binary Cross Entropy:
    // Loss = -(target * log(output) + (1 - target) * log(1 - output))

    // We reuse the errors member in back_propagation to store element-wise loss
    Tensor2& elementwise_loss = back_propagation.errors;

    elementwise_loss.device(get_device()) = -(targets * (outputs_2d + epsilon).log() +
        (targets.constant(1.0f) - targets) * (targets.constant(1.0f) - outputs_2d + epsilon).log());

    // 6. Aggregate Error
    // Sum only the non-masked (non-padding) elements
    Tensor0 total_masked_loss;
    total_masked_loss.device(get_device()) = (elementwise_loss * mask.cast<type>()).sum();

    Tensor0 active_elements;
    active_elements.device(get_device()) = mask.cast<type>().sum();


    // Average the error over the number of non-padded tokens
    if (active_elements() > 0.0f)
        back_propagation.error.device(get_device()) = total_masked_loss / active_elements();
    else
        back_propagation.error.setZero();
    */
}


void CrossEntropyError3d::calculate_multiple_error(const Batch& batch,
                                                   const ForwardPropagation& forward_propagation,
                                                   BackPropagation& back_propagation) const
{
    /*
    const TensorView targets_view = batch.get_targets();
    const MatrixMap targets = matrix_map(targets_view);

    const TensorView outputs_view = forward_propagation.get_last_trainable_layer_outputs();
    const TensorMap3 outputs = tensor_map<3>(outputs_view);

    const Index batch_size = outputs.dimension(0);
    const Index sequence_length = outputs.dimension(1);
    const Index vocabulary_size = outputs.dimension(2);

    // 3. Prepare Masking
    // We assume index 0 is the [PAD] token.
    // The mask is true for actual words and false for padding.
    back_propagation.mask.device(get_device()) = (targets != targets.constant(0.0f));
    const MatrixB& mask = back_propagation.mask;

    type total_log_loss = 0.0f;
    Index active_tokens_count = 0;

    // 4. Calculate Cross Entropy Sum
    // We need to find the probability the model assigned to the *correct* index.
    // In C++, a nested loop with OpenMP is the most efficient way to handle this 3D indexing.

    constexpr type epsilon = numeric_limits<type>::epsilon();

    #pragma omp parallel for reduction(+:total_log_loss, active_tokens_count)
    for(Index i = 0; i < batch_size; ++i)
    {
        for(Index j = 0; j < sequence_length; ++j)
        {
            // Only calculate loss if the token is not padding
            if (mask(i, j))
            {
                const Index target_index = static_cast<Index>(targets(i, j));

                // Safety check for vocabulary bounds
                if (target_index >= 0 && target_index < vocabulary_size)
                {
                    // Loss = -log(probability_of_correct_class)
                    const type probability = outputs(i, j, target_index);
                    total_log_loss -= log(probability + epsilon);
                    active_tokens_count++;
                }
            }
        }
    }

    // 5. Final Loss Calculation
    // Average the loss across all non-padding tokens in the batch

    active_tokens_count > 0
        ? back_propagation.error.setValues({total_log_loss/static_cast<type>(active_tokens_count)})
        : back_propagation.error.setZero();
    */
}



void CrossEntropyError3d::calculate_error(const Batch& batch,
                                          const ForwardPropagation& forward_propagation,
                                          BackPropagation& back_propagation) const
{
    /*
    const Index outputs_number = neural_network->get_outputs_number();

    outputs_number == 1
        ? calculate_binary_error(batch, forward_propagation, back_propagation)
        : calculate_multiple_error(batch, forward_propagation, back_propagation);

    if (isnan(back_propagation.error))
        throw runtime_error("Error is NAN.");
    */
}


void CrossEntropyError3d::calculate_output_gradients(const Batch&,
                                                 ForwardPropagation&,
                                                 BackPropagation&) const
{
    // Dense3d with softmax does not have output_gradients.
    // Error combinations derivatives are calculated directly.
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

void CrossEntropyError3d::calculate_error(const BatchCuda&,
                                               const ForwardPropagationCuda&,
                                               BackPropagationCuda&) const
{
    throw runtime_error("CUDA calculate_error not implemented for loss index type: CrossEntropyError3d");
}


void CrossEntropyError3d::calculate_output_gradients(const BatchCuda&,
                                                      ForwardPropagationCuda&,
                                                      BackPropagationCuda&) const
{
    throw runtime_error("CUDA calculate_output_gradients not implemented for loss index type: CrossEntropyError3d");
}

#endif

REGISTER(Loss, CrossEntropyError3d, "CrossEntropyError3d");

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
