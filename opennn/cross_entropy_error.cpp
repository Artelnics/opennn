//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   C R O S S   E N T R O P Y   E R R O R   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "registry.h"
#include "dataset.h"
#include "neural_network.h"
#include "cross_entropy_error.h"

namespace opennn
{

CrossEntropyError2d::CrossEntropyError2d(const NeuralNetwork* new_neural_network, const Dataset* new_dataset)
    : Loss(new_neural_network, new_dataset)
{
    name = "CrossEntropyError2d";
}


void CrossEntropyError2d::calculate_error(const Batch& batch,
                                          const ForwardPropagation& forward_propagation,
                                          BackPropagation& back_propagation) const
{
    const Index outputs_number = neural_network->get_outputs_number();

    outputs_number == 1
        ? calculate_binary_error(batch, forward_propagation, back_propagation)
        : calculate_multiple_error(batch, forward_propagation, back_propagation);
}


void CrossEntropyError2d::calculate_binary_error(const Batch& batch,
                                                 const ForwardPropagation& forward_propagation,
                                                 BackPropagation& back_propagation) const
{
    // Batch

    const Index samples_number = batch.get_samples_number();
    const TensorView targets_view = batch.get_targets();
    const MatrixMap targets = matrix_map(targets_view);

    // Forward propagation

    const TensorView outputs_view = forward_propagation.get_last_trainable_layer_outputs();
    const MatrixMap outputs = matrix_map(outputs_view);

    // Back propagation

    const type epsilon = numeric_limits<type>::epsilon();

    back_propagation.error = (targets.array() * (outputs.array() + epsilon).log() +
                               (1.0f - targets.array()) * (1.0f - outputs.array() + epsilon).log()
                               ).sum()/ static_cast<type>(-samples_number);

    if(isnan(back_propagation.error)) throw runtime_error("\nError is NAN.");
}


void CrossEntropyError2d::calculate_multiple_error(const Batch& batch,
                                                   const ForwardPropagation& forward_propagation,
                                                   BackPropagation& back_propagation) const
{
    // Batch

    const Index samples_number = batch.get_samples_number();
    const MatrixMap targets = matrix_map(batch.get_targets());

    // Forward propagation

    const MatrixMap outputs = matrix_map(forward_propagation.get_last_trainable_layer_outputs());

    // Back propagation

    const type epsilon = numeric_limits<type>::epsilon();

    type& error = back_propagation.error;

    error = (targets.array() * (outputs.array() + epsilon).log()).sum() / static_cast<type>(-samples_number);

    if(isnan(error)) throw runtime_error("\nError is NAN.");}


void CrossEntropyError2d::calculate_output_gradients(const Batch& batch,
                                                     ForwardPropagation& forward_propagation,
                                                     BackPropagation& back_propagation) const
{
    neural_network->get_outputs_number() == 1
        ? calculate_binary_output_gradients(batch, forward_propagation, back_propagation)
        : calculate_multiple_output_gradients(batch, forward_propagation, back_propagation);
}


void CrossEntropyError2d::calculate_binary_output_gradients(const Batch& batch,
                                                            ForwardPropagation& forward_propagation,
                                                            BackPropagation& back_propagation) const
{
    // Batch

    const Index samples_number = batch.get_samples_number();

    const TensorView targets_view = batch.get_targets();

    const MatrixMap targets = matrix_map(targets_view);

    // Forward propagation

    const TensorView outputs_view = forward_propagation.get_last_trainable_layer_outputs();

    const MatrixMap outputs = matrix_map(outputs_view);

    // Back propagation

    MatrixMap output_gradients = matrix_map(back_propagation.get_output_gradients());

    const type epsilon = numeric_limits<type>::epsilon();

    output_gradients.array() = (-targets.array() / (outputs.array() + epsilon) +
                                   (1.0f - targets.array()) / (1.0f - outputs.array() + epsilon)
                                   ) / static_cast<type>(samples_number);
}


void CrossEntropyError2d::calculate_multiple_output_gradients(const Batch& batch,
                                                          ForwardPropagation& forward_propagation,
                                                          BackPropagation& back_propagation) const
{
    // Batch

    const Index samples_number = batch.get_samples_number();

    const TensorView targets_view = batch.get_targets();

    const MatrixMap targets = matrix_map(targets_view);

    // Forward propagation

    const TensorView outputs_view = forward_propagation.get_last_trainable_layer_outputs();

    const MatrixMap outputs = matrix_map(outputs_view);

    // Back propagation

    MatrixMap output_gradients = matrix_map(back_propagation.get_output_gradients());

    output_gradients = (outputs - targets) / type(samples_number);
}


void CrossEntropyError2d::to_XML(XMLPrinter& file_stream) const
{
    file_stream.OpenElement("CrossEntropyError2d");

    file_stream.CloseElement();
}


void CrossEntropyError2d::from_XML(const XMLDocument& document)
{
    const XMLElement* root_element = document.FirstChildElement("CrossEntropyError2d");

    if(!root_element)
        throw runtime_error("Cross entropy error element is nullptr.\n");
}


#ifdef OPENNN_CUDA

void CrossEntropyError2d::calculate_error(const BatchCuda& batch,
                                               const ForwardPropagationCuda& forward_propagation,
                                               BackPropagationCuda& back_propagation) const
{
    const Index outputs_number = neural_network->get_outputs_number();

    outputs_number == 1
        ? calculate_binary_error(batch, forward_propagation, back_propagation)
        : calculate_multiple_error(batch, forward_propagation, back_propagation);
}


void CrossEntropyError2d::calculate_binary_error(const BatchCuda& batch,
                                                      const ForwardPropagationCuda& forward_propagation,
                                                      BackPropagationCuda& back_propagation) const
{
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

    calculate_binary_cross_entropy_cuda(size, errors, targets, outputs, numeric_limits<type>::epsilon());

    cudnnReduceTensor(get_cudnn_handle(),
                      reduce_tensor_descriptor,
                      nullptr,
                      0,
                      back_propagation.workspace,
                      back_propagation.workspace_size,
                      &alpha,
                      output_tensor_descriptor,
                      errors,
                      &beta,
                      output_reduce_tensor_descriptor,
                      error_device);

    CHECK_CUDA(cudaMemcpy(&error, error_device, sizeof(float), cudaMemcpyDeviceToHost));

    error = error / type(-samples_number);

    if (isnan(error)) throw runtime_error("\nError is NAN.");
}


void CrossEntropyError2d::calculate_multiple_error(const BatchCuda& batch,
                                                        const ForwardPropagationCuda& forward_propagation,
                                                        BackPropagationCuda& back_propagation) const
{
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
    void* workspace = back_propagation.workspace;
    const size_t workspace_size = back_propagation.workspace_size;

    calculate_multiple_cross_entropy_cuda(size, errors, targets, outputs, numeric_limits<type>::epsilon());

    const float alpha = 1.0f;
    const float beta = 0.0f;

    cudnnReduceTensor(get_cudnn_handle(),
                      reduce_tensor_descriptor,
                      nullptr, 0,
                      workspace, workspace_size,
                      &alpha,
                      output_tensor_descriptor, errors,
                      &beta,
                      output_reduce_tensor_descriptor, error_device);

    CHECK_CUDA(cudaMemcpy(&error, error_device, sizeof(type), cudaMemcpyDeviceToHost));

    error = error / type(-samples_number);

    if (isnan(error)) throw runtime_error("\nError is NAN.");
}


void CrossEntropyError2d::calculate_output_gradients(const BatchCuda& batch,
                                                      ForwardPropagationCuda& forward_propagation,
                                                      BackPropagationCuda& back_propagation) const
{
    const Index outputs_number = neural_network->get_outputs_number();

    outputs_number == 1
        ? calculate_binary_output_gradients(batch, forward_propagation, back_propagation)
        : calculate_multiple_output_gradients(batch, forward_propagation, back_propagation);
}


void CrossEntropyError2d::calculate_binary_output_gradients(const BatchCuda& batch,
                                                             ForwardPropagationCuda& forward_propagation,
                                                             BackPropagationCuda& back_propagation) const
{
    // Batch

    const Index samples_number = batch.get_samples_number();

    const type* targets = batch.targets_device.data;

    // Forward propagation

    const float* outputs = forward_propagation.get_last_trainable_layer_outputs_device().data;

    const size_t size = samples_number * forward_propagation.layers[neural_network->get_last_trainable_layer_index()]->layer->get_outputs_number();

    // Back propagation

    float* output_gradients = back_propagation.get_output_gradient_views_device().data;

    const type scaling_factor = 1.0f / static_cast<type>(samples_number);

    calculate_binary_cross_entropy_delta_cuda(size, output_gradients, targets, outputs, numeric_limits<type>::epsilon(), scaling_factor);
}


void CrossEntropyError2d::calculate_multiple_output_gradients(const BatchCuda& batch,
                                                               ForwardPropagationCuda& forward_propagation,
                                                               BackPropagationCuda& back_propagation) const
{
    // Batch

    const Index samples_number = batch.get_samples_number();

    const type* targets = batch.targets_device.data;

    // Forward propagation

    const float* outputs = forward_propagation.get_last_trainable_layer_outputs_device().data;

    // Back propagation

    float* output_gradients = back_propagation.get_output_gradient_views_device().data;

    const size_t size = samples_number * forward_propagation.layers[neural_network->get_last_trainable_layer_index()]->layer->get_outputs_number();

    const float scale_factor = 1.0f / static_cast<float>(samples_number);

    calculate_multiple_cross_entropy_delta_cuda(size, output_gradients, targets, outputs, scale_factor);
}

#endif

REGISTER(Loss, CrossEntropyError2d, "CrossEntropyError2d");

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
