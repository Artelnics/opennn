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
    : LossIndex(new_neural_network, new_dataset)
{
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
}


void CrossEntropyError2d::calculate_multiple_error(const Batch& batch,
                                                   const ForwardPropagation& forward_propagation,
                                                   BackPropagation& back_propagation) const
{
    // Batch

    const Index samples_number = batch.get_samples_number();

    const pair<type*, dimensions> targets_pair = batch.get_target_pair();

    const TensorMap<Tensor<type, 2>> targets = tensor_map<2>(targets_pair);

    // Forward propagation

    const pair<type*, dimensions> outputs_pair = forward_propagation.get_last_trainable_layer_outputs_pair();

    const TensorMap<Tensor<type, 2>> outputs = tensor_map<2>(outputs_pair);

    // Back propagation

    Tensor<type, 0>& error = back_propagation.error;

    error.device(*thread_pool_device) = (targets*(outputs + epsilon).log()).sum() / type(-samples_number);

    if(isnan(error())) throw runtime_error("\nError is NAN.");
}


void CrossEntropyError2d::calculate_output_delta(const Batch& batch,
                                                 ForwardPropagation& forward_propagation,
                                                 BackPropagation& back_propagation) const
{
    const Index outputs_number = neural_network->get_outputs_number();

    outputs_number == 1
        ? calculate_binary_output_delta(batch, forward_propagation, back_propagation)
        : calculate_multiple_output_delta(batch, forward_propagation, back_propagation);
}


void CrossEntropyError2d::calculate_binary_output_delta(const Batch& batch,
                                                        ForwardPropagation& forward_propagation,
                                                        BackPropagation& back_propagation) const
{
    // Batch

    const Index samples_number = batch.get_samples_number();

    const pair<type*, dimensions> targets_pair = batch.get_target_pair();

    const TensorMap<Tensor<type, 2>> targets = tensor_map<2>(targets_pair);

    // Forward propagation

    const pair<type*, dimensions> outputs_pair = forward_propagation.get_last_trainable_layer_outputs_pair();

    const TensorMap<Tensor<type, 2>> outputs = tensor_map<2>(outputs_pair);

    // Back propagation

    const pair<type*, dimensions> output_deltas_pair = back_propagation.get_output_deltas_pair();

    TensorMap<Tensor<type, 2>> output_deltas = tensor_map<2>(output_deltas_pair);

    output_deltas.device(*thread_pool_device)
        = (-targets/(outputs + epsilon) + (type(1) - targets)/(type(1) - outputs + epsilon))/type(samples_number);
}


void CrossEntropyError2d::calculate_multiple_output_delta(const Batch& batch,
                                                          ForwardPropagation& forward_propagation,
                                                          BackPropagation& back_propagation) const
{
    // Batch

    const Index samples_number = batch.get_samples_number();

    const pair<type*, dimensions> targets_pair = batch.get_target_pair();

    const TensorMap<Tensor<type, 2>> targets = tensor_map<2>(targets_pair);

    // Forward propagation

    const pair<type*, dimensions> outputs_pair = forward_propagation.get_last_trainable_layer_outputs_pair();

    const TensorMap<Tensor<type, 2>> outputs = tensor_map<2>(outputs_pair);

    // Back propagation

    const pair<type*, dimensions> output_deltas_pair = back_propagation.get_output_deltas_pair();

    TensorMap<Tensor<type, 2>> output_deltas = tensor_map<2>(output_deltas_pair);

    output_deltas.device(*thread_pool_device) = (outputs - targets) / type(samples_number);
}


string CrossEntropyError2d::get_name() const
{
    return "CrossEntropyError2d";
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
    /*
    // Regularization

    XMLDocument regularization_document;

    const XMLElement* regularization_element = root_element->FirstChildElement("Regularization");
    regularization_document.InsertFirstChild(regularization_element->DeepClone(&regularization_document));

    regularization_from_XML(regularization_document);
    */
}


#ifdef OPENNN_CUDA

void CrossEntropyError2d::calculate_error_cuda(const BatchCuda& batch_cuda,
                                               const ForwardPropagationCuda& forward_propagation_cuda,
                                               BackPropagationCuda& back_propagation_cuda) const
{
    const Index outputs_number = neural_network->get_outputs_number();

    outputs_number == 1
        ? calculate_binary_error_cuda(batch_cuda, forward_propagation_cuda, back_propagation_cuda)
        : calculate_multiple_error_cuda(batch_cuda, forward_propagation_cuda, back_propagation_cuda);
}


void CrossEntropyError2d::calculate_binary_error_cuda(const BatchCuda& batch_cuda,
                                                      const ForwardPropagationCuda& forward_propagation_cuda,
                                                      BackPropagationCuda& back_propagation_cuda) const
{
    // Batch

    const Index samples_number = batch_cuda.get_samples_number();

    const type* targets = batch_cuda.targets_device;

    // Forward propagation

    const float* outputs = forward_propagation_cuda.get_last_trainable_layer_outputs_device();

    // Back propagation

    Tensor<type, 0>& error = back_propagation_cuda.error;
    float* error_device = back_propagation_cuda.error_device;
    float* errors = back_propagation_cuda.errors;

    const size_t size = samples_number * forward_propagation_cuda.layers[neural_network->get_last_trainable_layer_index()]->layer->get_outputs_number();

    const cudnnTensorDescriptor_t& output_tensor_descriptor = back_propagation_cuda.output_tensor_descriptor;
    const cudnnTensorDescriptor_t& output_reduce_tensor_descriptor = back_propagation_cuda.output_reduce_tensor_descriptor;

    const cudnnReduceTensorDescriptor_t& reduce_tensor_descriptor = back_propagation_cuda.reduce_tensor_descriptor;
    void* workspace = back_propagation_cuda.workspace;
    size_t workspaceSize = back_propagation_cuda.workspaceSize;

    const float alpha = 1.0f;
    const float beta = 0.0f;

    calculate_binary_cross_entropy_cuda(size, errors, targets, outputs, epsilon);

    cudnnReduceTensor(cudnn_handle,
                      reduce_tensor_descriptor,
                      nullptr, 0,
                      workspace, workspaceSize,
                      &alpha,
                      output_tensor_descriptor, errors,
                      &beta,
                      output_reduce_tensor_descriptor, error_device);

    CHECK_CUDA(cudaMemcpy(error.data(), error_device, sizeof(float), cudaMemcpyDeviceToHost));

    error = error / type(-samples_number);

    if (isnan(error())) throw runtime_error("\nError is NAN.");
}


void CrossEntropyError2d::calculate_multiple_error_cuda(const BatchCuda& batch_cuda,
                                                        const ForwardPropagationCuda& forward_propagation_cuda,
                                                        BackPropagationCuda& back_propagation_cuda) const
{
    // Batch

    const Index samples_number = batch_cuda.get_samples_number();

    const type* targets = batch_cuda.targets_device;

    // Forward propagation

    const float* outputs = forward_propagation_cuda.get_last_trainable_layer_outputs_device();

    // Back propagation

    Tensor<type, 0>& error = back_propagation_cuda.error;
    float* error_device = back_propagation_cuda.error_device;
    float* errors = back_propagation_cuda.errors;

    const size_t size = samples_number * forward_propagation_cuda.layers[neural_network->get_last_trainable_layer_index()]->layer->get_outputs_number();

    const cudnnTensorDescriptor_t& output_tensor_descriptor = back_propagation_cuda.output_tensor_descriptor;
    const cudnnTensorDescriptor_t& output_reduce_tensor_descriptor = back_propagation_cuda.output_reduce_tensor_descriptor;

    const cudnnReduceTensorDescriptor_t& reduce_tensor_descriptor = back_propagation_cuda.reduce_tensor_descriptor;
    void* workspace = back_propagation_cuda.workspace;
    size_t workspaceSize = back_propagation_cuda.workspaceSize;

    calculate_multiple_cross_entropy_cuda(size, errors, targets, outputs, epsilon);

    const float alpha = 1.0f;
    const float beta = 0.0f;

    cudnnReduceTensor(cudnn_handle,
                      reduce_tensor_descriptor,
                      nullptr, 0,
                      workspace, workspaceSize,
                      &alpha,
                      output_tensor_descriptor, errors,
                      &beta,
                      output_reduce_tensor_descriptor, error_device);

    CHECK_CUDA(cudaMemcpy(error.data(), error_device, sizeof(type), cudaMemcpyDeviceToHost));

    error = error / type(-samples_number);

    if (isnan(error())) throw runtime_error("\nError is NAN.");
}


void CrossEntropyError2d::calculate_output_delta_cuda(const BatchCuda& batch_cuda,
                                                      ForwardPropagationCuda& forward_propagation_cuda,
                                                      BackPropagationCuda& back_propagation_cuda) const
{
    const Index outputs_number = neural_network->get_outputs_number();

    outputs_number == 1
        ? calculate_binary_output_delta_cuda(batch_cuda, forward_propagation_cuda, back_propagation_cuda)
        : calculate_multiple_output_delta_cuda(batch_cuda, forward_propagation_cuda, back_propagation_cuda);
}


void CrossEntropyError2d::calculate_binary_output_delta_cuda(const BatchCuda& batch_cuda,
                                                             ForwardPropagationCuda& forward_propagation_cuda,
                                                             BackPropagationCuda& back_propagation_cuda) const
{
    // Batch

    const Index samples_number = batch_cuda.get_samples_number();

    const type* targets = batch_cuda.targets_device;

    // Forward propagation

    const float* outputs = forward_propagation_cuda.get_last_trainable_layer_outputs_device();

    const size_t size = samples_number * forward_propagation_cuda.layers[neural_network->get_last_trainable_layer_index()]->layer->get_outputs_number();

    // Back propagation

    float* output_deltas = back_propagation_cuda.get_output_deltas_device();

    const type scaling_factor = 1.0f / static_cast<type>(samples_number);

    calculate_binary_cross_entropy_delta_cuda(size, output_deltas, targets, outputs, epsilon, scaling_factor);
}


void CrossEntropyError2d::calculate_multiple_output_delta_cuda(const BatchCuda& batch_cuda,
                                                               ForwardPropagationCuda& forward_propagation_cuda,
                                                               BackPropagationCuda& back_propagation_cuda) const
{
    // Batch

    const Index samples_number = batch_cuda.get_samples_number();

    const type* targets = batch_cuda.targets_device;

    // Forward propagation

    const float* outputs = forward_propagation_cuda.get_last_trainable_layer_outputs_device();

    // Back propagation

    float* output_deltas = back_propagation_cuda.get_output_deltas_device();

    const size_t size = samples_number * forward_propagation_cuda.layers[neural_network->get_last_trainable_layer_index()]->layer->get_outputs_number();

    const float scale_factor = 1.0f / static_cast<float>(samples_number);

    calculate_multiple_cross_entropy_delta_cuda(size, output_deltas, targets, outputs, scale_factor);
}

#endif

REGISTER(LossIndex, CrossEntropyError2d, "CrossEntropyError2d");

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
