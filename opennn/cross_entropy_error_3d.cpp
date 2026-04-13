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
    const TensorView outputs_view = forward_propagation.get_last_trainable_layer_outputs();
    TensorMap3 outputs = tensor_map<3>(outputs_view);

    const TensorView targets_view = batch.get_targets();
    TensorMap2 targets(targets_view.data, outputs_view.shape[0], outputs_view.shape[1]);

    const Index batch_size = outputs.dimension(0);
    const Index sequence_length = outputs.dimension(1);

    type total_loss = 0.0f;
    Index active_tokens_count = 0;
    Index correct_tokens_count = 0;

#pragma omp parallel for reduction(+:total_loss, active_tokens_count, correct_tokens_count)
    for(Index i = 0; i < batch_size; ++i)
    {
        for(Index j = 0; j < sequence_length; ++j)
        {
            const type target_val = targets(i, j);

            if(target_val >= 0.0f)
            {
                const type y = outputs(i, j, 0);
                const type t = target_val;

                total_loss -= (t * log(y + EPSILON)
                               + (type(1.0) - t) * log(type(1.0) - y + EPSILON));

                active_tokens_count++;

                const type predicted = y >= type(0.5) ? type(1.0) : type(0.0);

                if(predicted == t)
                    correct_tokens_count++;
            }
        }
    }

    back_propagation.error = active_tokens_count > 0
                                 ? total_loss / static_cast<type>(active_tokens_count)
                                 : type(0);

    if(active_tokens_count > 0)
        back_propagation.accuracy.setValues(
            {static_cast<type>(correct_tokens_count) / static_cast<type>(active_tokens_count)});
    else
        back_propagation.accuracy.setZero();
}


void CrossEntropyError3d::calculate_multiple_error(const Batch& batch,
                                                   const ForwardPropagation& forward_propagation,
                                                   BackPropagation& back_propagation) const
{
    const TensorView outputs_view = forward_propagation.get_last_trainable_layer_outputs();
    TensorMap3 outputs = tensor_map<3>(outputs_view);

    const TensorView targets_view = batch.get_targets();
    TensorMap2 targets(targets_view.data, outputs_view.shape[0], outputs_view.shape[1]);

    const Index batch_size = outputs.dimension(0);
    const Index sequence_length = outputs.dimension(1);
    const Index vocabulary_size = outputs.dimension(2);

    type total_log_loss = 0.0f;
    Index active_tokens_count = 0;
    Index correct_tokens_count = 0;

#pragma omp parallel for reduction(+:total_log_loss, active_tokens_count, correct_tokens_count)
    for(Index i = 0; i < batch_size; ++i)
    {
        for(Index j = 0; j < sequence_length; ++j)
        {
            const Index target_index = static_cast<Index>(targets(i, j));

            if(target_index > 0 && target_index < vocabulary_size)
            {
                const type probability = outputs(i, j, target_index);
                total_log_loss -= log(probability + EPSILON);
                active_tokens_count++;

                Index best_index = 0;
                type best_value = outputs(i, j, 0);

                for(Index k = 1; k < vocabulary_size; ++k)
                {
                    if(outputs(i, j, k) > best_value)
                    {
                        best_value = outputs(i, j, k);
                        best_index = k;
                    }
                }

                if(best_index == target_index)
                    correct_tokens_count++;
            }
        }
    }

    back_propagation.error = active_tokens_count > 0
                                 ? total_log_loss / static_cast<type>(active_tokens_count)
                                 : type(0);

    if(active_tokens_count > 0)
        back_propagation.accuracy.setValues(
            {static_cast<type>(correct_tokens_count) / static_cast<type>(active_tokens_count)});
    else
        back_propagation.accuracy.setZero();
}


void CrossEntropyError3d::calculate_error(const Batch& batch,
                                          const ForwardPropagation& forward_propagation,
                                          BackPropagation& back_propagation) const
{
    const Index outputs_number = neural_network->get_output_shape().back();

    outputs_number == 1
        ? calculate_binary_error(batch, forward_propagation, back_propagation)
        : calculate_multiple_error(batch, forward_propagation, back_propagation);

    if (isnan(back_propagation.error))
        throw runtime_error("Error is NAN.");
}


void CrossEntropyError3d::calculate_output_gradients(const Batch& batch,
                                                     ForwardPropagation& forward_propagation,
                                                     BackPropagation& back_propagation) const
{
    const TensorView outputs_view = forward_propagation.get_last_trainable_layer_outputs();
    TensorMap3 outputs = tensor_map<3>(outputs_view);

    const TensorView targets_view = batch.get_targets();
    TensorMap2 targets(targets_view.data, outputs_view.shape[0], outputs_view.shape[1]);

    TensorMap3 output_gradients(back_propagation.output_gradients.data(),
                                outputs_view.shape[0], outputs_view.shape[1], outputs_view.shape[2]);

    const Index batch_size = outputs.dimension(0);
    const Index sequence_length = outputs.dimension(1);
    const Index vocabulary_size = outputs.dimension(2);

    Index active_tokens_count = 0;

    if(vocabulary_size == 1)
    {
#pragma omp parallel for reduction(+:active_tokens_count)
        for(Index i = 0; i < batch_size; ++i)
            for(Index j = 0; j < sequence_length; ++j)
                if(targets(i, j) >= 0.0f)
                    active_tokens_count++;
    }
    else
    {
#pragma omp parallel for reduction(+:active_tokens_count)
        for(Index i = 0; i < batch_size; ++i)
            for(Index j = 0; j < sequence_length; ++j)
            {
                const Index target_index = static_cast<Index>(targets(i, j));
                if(target_index > 0 && target_index < vocabulary_size)
                    active_tokens_count++;
            }
    }

    const type scale = active_tokens_count > 0
                           ? type(1.0) / static_cast<type>(active_tokens_count)
                           : type(0.0);

    if(vocabulary_size == 1)
    {
#pragma omp parallel for
        for(Index i = 0; i < batch_size; ++i)
        {
            for(Index j = 0; j < sequence_length; ++j)
            {
                const type target_val = targets(i, j);

                if(target_val >= 0.0f)
                {
                    const type out = outputs(i, j, 0);
                    const type term1 = (type(1.0) - target_val) / (type(1.0) - out + EPSILON);
                    const type term2 = target_val / (out + EPSILON);
                    output_gradients(i, j, 0) = (term1 - term2) * scale;
                }
                else
                {
                    output_gradients(i, j, 0) = type(0.0);
                }
            }
        }
    }
    else
    {
#pragma omp parallel for
        for(Index i = 0; i < batch_size; ++i)
        {
            for(Index j = 0; j < sequence_length; ++j)
            {
                const Index target_index = static_cast<Index>(targets(i, j));

                if(target_index > 0 && target_index < vocabulary_size)
                {
                    for(Index k = 0; k < vocabulary_size; ++k)
                    {
                        if(k == target_index)
                            output_gradients(i, j, k) = (outputs(i, j, k) - type(1.0)) * scale;
                        else
                            output_gradients(i, j, k) = outputs(i, j, k) * scale;
                    }
                }
                else
                {
                    for(Index k = 0; k < vocabulary_size; ++k)
                        output_gradients(i, j, k) = type(0.0);
                }
            }
        }
    }
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

void CrossEntropyError3d::calculate_error(const BatchCuda& batch,
                                          const ForwardPropagationCuda& forward_propagation,
                                          BackPropagationCuda& back_propagation) const
{
    const Index outputs_number = neural_network->get_output_shape().back();

    outputs_number == 1
        ? calculate_binary_error(batch, forward_propagation, back_propagation)
        : calculate_multiple_error(batch, forward_propagation, back_propagation);
}


void CrossEntropyError3d::calculate_binary_error(const BatchCuda& batch,
                                                 const ForwardPropagationCuda& forward_propagation,
                                                 BackPropagationCuda& back_propagation) const
{
    const Shape layer_output_shape = neural_network->get_output_shape();

    const int batch_size = static_cast<int>(batch.get_samples_number());
    const int seq_length = static_cast<int>(layer_output_shape[0]);

    const type* targets_device = batch.targets_device.data;
    const float* outputs_device =
        const_cast<ForwardPropagationCuda&>(forward_propagation)
            .get_last_trainable_layer_outputs_device().data;

    type& error = back_propagation.error;
    float* error_device = back_propagation.error_device;
    float* errors = back_propagation.errors;

    const size_t token_count = static_cast<size_t>(batch_size) * static_cast<size_t>(seq_length);

    const cudnnTensorDescriptor_t output_reduce_tensor_descriptor = back_propagation.output_reduce_tensor_descriptor;
    const cudnnReduceTensorDescriptor_t reduce_tensor_descriptor = back_propagation.reduce_tensor_descriptor;

    void* workspace = back_propagation.workspace;
    const size_t workspace_size = back_propagation.workspace_size;

    TensorViewCuda token_errors_view(Shape{batch_size, seq_length});
    const cudnnTensorDescriptor_t token_errors_descriptor = token_errors_view.get_descriptor();

    cross_entropy_3d_binary_forward_cuda(token_count,
                                         batch_size,
                                         seq_length,
                                         outputs_device,
                                         targets_device,
                                         errors,
                                         EPSILON);

    CHECK_CUDNN(cudnnReduceTensor(get_cudnn_handle(),
                                  reduce_tensor_descriptor,
                                  nullptr, 0,
                                  workspace, workspace_size,
                                  &alpha_one,
                                  token_errors_descriptor, errors,
                                  &beta_zero,
                                  output_reduce_tensor_descriptor, error_device));

    float total_loss = 0.0f;
    CHECK_CUDA(cudaMemcpy(&total_loss, error_device, sizeof(float), cudaMemcpyDeviceToHost));

    CHECK_CUDA(cudaMemset(error_device, 0, 2 * sizeof(float)));

    cross_entropy_3d_binary_counts_cuda(token_count,
                                        outputs_device,
                                        targets_device,
                                        error_device);

    float counts_host[2] = {0.0f, 0.0f};
    CHECK_CUDA(cudaMemcpy(counts_host, error_device, 2 * sizeof(float), cudaMemcpyDeviceToHost));

    const Index active_tokens_count = static_cast<Index>(counts_host[0]);
    const Index correct_tokens_count = static_cast<Index>(counts_host[1]);

    back_propagation.active_tokens_count = active_tokens_count;

    error = active_tokens_count > 0
                ? static_cast<type>(total_loss / static_cast<float>(active_tokens_count))
                : type(0);

    if(active_tokens_count > 0)
        back_propagation.accuracy.setValues(
            {static_cast<type>(correct_tokens_count) / static_cast<type>(active_tokens_count)});
    else
        back_propagation.accuracy.setZero();

    if(isnan(error))
        throw runtime_error("\nError is NAN.");
}


void CrossEntropyError3d::calculate_multiple_error(const BatchCuda& batch,
                                                   const ForwardPropagationCuda& forward_propagation,
                                                   BackPropagationCuda& back_propagation) const
{
    const Shape layer_output_shape = neural_network->get_output_shape();

    const int batch_size = static_cast<int>(batch.get_samples_number());
    const int seq_length = static_cast<int>(layer_output_shape[0]);
    const int vocab_size = static_cast<int>(layer_output_shape[1]);

    const type* targets_device = batch.targets_device.data;
    const float* outputs_device =
        const_cast<ForwardPropagationCuda&>(forward_propagation)
            .get_last_trainable_layer_outputs_device().data;

    type& error = back_propagation.error;
    float* error_device = back_propagation.error_device;
    float* errors = back_propagation.errors;

    const size_t n = static_cast<size_t>(batch_size)
                     * static_cast<size_t>(seq_length)
                     * static_cast<size_t>(vocab_size);

    const size_t token_count = static_cast<size_t>(batch_size)
                               * static_cast<size_t>(seq_length);

    const cudnnTensorDescriptor_t output_reduce_tensor_descriptor = back_propagation.output_reduce_tensor_descriptor;
    const cudnnReduceTensorDescriptor_t reduce_tensor_descriptor = back_propagation.reduce_tensor_descriptor;

    void* workspace = back_propagation.workspace;
    const size_t workspace_size = back_propagation.workspace_size;

    TensorViewCuda token_errors_view(Shape{batch_size, seq_length});
    const cudnnTensorDescriptor_t token_errors_descriptor = token_errors_view.get_descriptor();

    cross_entropy_3d_multiple_forward_cuda(n,
                                           batch_size,
                                           seq_length,
                                           vocab_size,
                                           outputs_device,
                                           targets_device,
                                           errors,
                                           EPSILON);

    CHECK_CUDNN(cudnnReduceTensor(get_cudnn_handle(),
                                  reduce_tensor_descriptor,
                                  nullptr, 0,
                                  workspace, workspace_size,
                                  &alpha_one,
                                  token_errors_descriptor, errors,
                                  &beta_zero,
                                  output_reduce_tensor_descriptor, error_device));

    float total_loss = 0.0f;
    CHECK_CUDA(cudaMemcpy(&total_loss, error_device, sizeof(float), cudaMemcpyDeviceToHost));

    CHECK_CUDA(cudaMemset(error_device, 0, 2 * sizeof(float)));

    cross_entropy_3d_multiple_counts_cuda(token_count,
                                          vocab_size,
                                          outputs_device,
                                          targets_device,
                                          error_device);

    float counts_host[2] = {0.0f, 0.0f};
    CHECK_CUDA(cudaMemcpy(counts_host, error_device, 2 * sizeof(float), cudaMemcpyDeviceToHost));

    const Index active_tokens_count = static_cast<Index>(counts_host[0]);
    const Index correct_tokens_count = static_cast<Index>(counts_host[1]);

    back_propagation.active_tokens_count = active_tokens_count;

    error = active_tokens_count > 0
                ? static_cast<type>(total_loss / static_cast<float>(active_tokens_count))
                : type(0);

    if(active_tokens_count > 0)
        back_propagation.accuracy.setValues(
            {static_cast<type>(correct_tokens_count) / static_cast<type>(active_tokens_count)});
    else
        back_propagation.accuracy.setZero();

    if(isnan(error))
        throw runtime_error("\nError is NAN.");
}


void CrossEntropyError3d::calculate_output_gradients(const BatchCuda& batch,
                                                     ForwardPropagationCuda& forward_propagation,
                                                     BackPropagationCuda& back_propagation) const
{
    const Index outputs_number = neural_network->get_output_shape().back();

    outputs_number == 1
        ? calculate_binary_output_gradients(batch, forward_propagation, back_propagation)
        : calculate_multiple_output_gradients(batch, forward_propagation, back_propagation);
}


void CrossEntropyError3d::calculate_binary_output_gradients(const BatchCuda& batch,
                                                            ForwardPropagationCuda& forward_propagation,
                                                            BackPropagationCuda& back_propagation) const
{
    const Shape layer_output_shape = neural_network->get_output_shape();

    const int batch_size = static_cast<int>(batch.get_samples_number());
    const int seq_length = static_cast<int>(layer_output_shape[0]);

    const type* targets_device = batch.targets_device.data;
    const float* outputs = forward_propagation.get_last_trainable_layer_outputs_device().data;
    float* output_gradients = back_propagation.get_output_gradients_device().data;

    const size_t size = static_cast<size_t>(batch_size) * static_cast<size_t>(seq_length);

    const float scale_factor = back_propagation.active_tokens_count > 0
                                   ? (1.0f / static_cast<float>(back_propagation.active_tokens_count))
                                   : 0.0f;

    cross_entropy_3d_binary_backward_cuda(size,
                                          batch_size,
                                          seq_length,
                                          outputs,
                                          targets_device,
                                          output_gradients,
                                          scale_factor);
}


void CrossEntropyError3d::calculate_multiple_output_gradients(const BatchCuda& batch,
                                                              ForwardPropagationCuda& forward_propagation,
                                                              BackPropagationCuda& back_propagation) const
{
    const Shape layer_output_shape = neural_network->get_output_shape();

    const int batch_size = static_cast<int>(batch.get_samples_number());
    const int seq_length = static_cast<int>(layer_output_shape[0]);
    const int vocab_size = static_cast<int>(layer_output_shape[1]);

    const type* targets_device = batch.targets_device.data;
    const float* outputs = forward_propagation.get_last_trainable_layer_outputs_device().data;
    float* output_gradients = back_propagation.get_output_gradients_device().data;

    const size_t size = static_cast<size_t>(batch_size)
                        * static_cast<size_t>(seq_length)
                        * static_cast<size_t>(vocab_size);

    const float scale_factor = back_propagation.active_tokens_count > 0
                                   ? (1.0f / static_cast<float>(back_propagation.active_tokens_count))
                                   : 0.0f;

    cross_entropy_3d_multiple_backward_cuda(size,
                                            batch_size,
                                            seq_length,
                                            vocab_size,
                                            outputs,
                                            targets_device,
                                            output_gradients,
                                            scale_factor);
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
