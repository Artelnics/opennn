//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   C R O S S   E N T R O P Y   E R R O R   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "registry.h"
#include "tensors.h"
#include "dataset.h"
#include "neural_network.h"
#include "loss_index.h"
#include "cross_entropy_error.h"

namespace opennn
{

CrossEntropyError2d::CrossEntropyError2d(NeuralNetwork* new_neural_network, Dataset* new_data_set)
    : LossIndex(new_neural_network, new_data_set)
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

    // cout << "samples_number: " << samples_number << endl;
    // cout << "targets_pair_first: " << targets_pair.first << endl;
    // cout << "targets_pair_second: " << endl;
    // print_vector(targets_pair.second);

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

    const size_t size = samples_number * forward_propagation_cuda.layers[neural_network->get_last_trainable_layer_index()]->layer->get_outputs_number();

    const cudnnTensorDescriptor_t& output_tensor_descriptor = back_propagation_cuda.output_tensor_descriptor;
    const cudnnTensorDescriptor_t& output_reduce_tensor_descriptor = back_propagation_cuda.output_reduce_tensor_descriptor;

    const cudnnOpTensorDescriptor_t& operator_multiplication_descriptor = back_propagation_cuda.operator_multiplication_descriptor;
    const cudnnOpTensorDescriptor_t& operator_sum_descriptor = back_propagation_cuda.operator_sum_descriptor;

    const cudnnReduceTensorDescriptor_t& reduce_tensor_descriptor = back_propagation_cuda.reduce_tensor_descriptor;
    void* workspace = back_propagation_cuda.workspace;
    size_t workspaceSize = back_propagation_cuda.workspaceSize;

    float* numerator = back_propagation_cuda.numerator;
    float* numerator_2 = back_propagation_cuda.numerator_2;
    float* numerator_3 = back_propagation_cuda.numerator_3;
    float* outputs_plus_epsilon = back_propagation_cuda.outputs_plus_epsilon;
    float* one_minus_targets = back_propagation_cuda.one_minus_targets;
    float* one_minus_outputs = back_propagation_cuda.one_minus_outputs;
    float* numerator_reduce = back_propagation_cuda.numerator_reduce;
    float* ones = back_propagation_cuda.ones;

    const float alpha = 1.0f;
    const float alpha_minus_one = -1.0f;
    const float beta = 0.0f;

    cudaMemcpy(one_minus_outputs, ones, size * sizeof(float), cudaMemcpyDeviceToDevice);
    cudaMemcpy(one_minus_targets, ones, size * sizeof(float), cudaMemcpyDeviceToDevice);

    cublasSaxpy(cublas_handle, size, &alpha_minus_one, outputs, 1, one_minus_outputs, 1); // 1 - outputs
    cublasSaxpy(cublas_handle, size, &epsilon, ones, 1, one_minus_outputs, 1); // (1 - outputs) + epsilon

    cublasSaxpy(cublas_handle, size, &alpha_minus_one, targets, 1, one_minus_targets, 1); // 1 - targets

    log(size, one_minus_outputs, numerator_2); // (1 - outputs).log()

    // (1 - targets) * (1 - outputs).log()
    cudnnOpTensor(cudnn_handle,
        operator_multiplication_descriptor,
        &alpha,
        output_tensor_descriptor,
        one_minus_targets,
        &alpha,
        output_tensor_descriptor,
        numerator_2,
        &beta,
        output_tensor_descriptor,
        numerator);

    // outputs + epsilon
    cudaMemcpy(outputs_plus_epsilon, outputs, size * sizeof(float), cudaMemcpyDeviceToDevice);
    cublasSaxpy(cublas_handle, size, &epsilon, ones, 1, outputs_plus_epsilon, 1);

    // outputs.log()
    log(size, outputs_plus_epsilon, numerator_2);

    // target * outputs.log()
    cudnnOpTensor(cudnn_handle,
        operator_multiplication_descriptor,
        &alpha,
        output_tensor_descriptor,
        targets,
        &alpha,
        output_tensor_descriptor,
        numerator_2,
        &beta,
        output_tensor_descriptor,
        numerator_3);

    // (target * outputs.log()) + ((1 - targets) * (1 - outputs).log())
    cudnnOpTensor(cudnn_handle,
        operator_sum_descriptor,
        &alpha,
        output_tensor_descriptor,
        numerator_3,
        &alpha,
        output_tensor_descriptor,
        numerator,
        &beta,
        output_tensor_descriptor,
        numerator_2);

    // (target * outputs.log()) + ((1 - targets) * (1 - outputs).log()).sum()
    cudnnReduceTensor(cudnn_handle,
        reduce_tensor_descriptor,
        nullptr, 0,
        workspace, workspaceSize,
        &alpha,
        output_tensor_descriptor, numerator_2,
        &beta,
        output_reduce_tensor_descriptor, numerator_reduce);

    cudaMemcpy(error.data(), numerator_reduce, sizeof(type), cudaMemcpyDeviceToHost);

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

    const size_t size = samples_number * forward_propagation_cuda.layers[neural_network->get_last_trainable_layer_index()]->layer->get_outputs_number();

    const cudnnTensorDescriptor_t& output_tensor_descriptor = back_propagation_cuda.output_tensor_descriptor;
    const cudnnTensorDescriptor_t& output_reduce_tensor_descriptor = back_propagation_cuda.output_reduce_tensor_descriptor;

    const cudnnOpTensorDescriptor_t& operator_multiplication_descriptor = back_propagation_cuda.operator_multiplication_descriptor;

    const cudnnReduceTensorDescriptor_t& reduce_tensor_descriptor = back_propagation_cuda.reduce_tensor_descriptor;
    void* workspace = back_propagation_cuda.workspace;
    size_t workspaceSize = back_propagation_cuda.workspaceSize;

    float* numerator = back_propagation_cuda.numerator;
    float* numerator_2 = back_propagation_cuda.numerator_2;
    float* numerator_reduce = back_propagation_cuda.numerator_reduce;
    float* outputs_plus_epsilon = back_propagation_cuda.outputs_plus_epsilon;
    float* ones = back_propagation_cuda.ones;

    float alpha = 1.0f;
    const float beta = 0.0f;

    // outputs + epsilon
    cudaMemcpy(outputs_plus_epsilon, outputs, size * sizeof(float), cudaMemcpyDeviceToDevice);
    cublasSaxpy(cublas_handle, size, &epsilon, ones, 1, outputs_plus_epsilon, 1);

    // (outputs + epsilon).log()
    log(size, outputs_plus_epsilon, numerator_2);

    // targets * ((outputs + epsilon).log())
    cudnnOpTensor(cudnn_handle,
        operator_multiplication_descriptor,
        &alpha,
        output_tensor_descriptor,
        numerator_2,
        &alpha,
        output_tensor_descriptor,
        targets,
        &beta,
        output_tensor_descriptor,
        numerator);

    // (targets * ((outputs + epsilon).log())).sum()
    cudnnReduceTensor(cudnn_handle,
        reduce_tensor_descriptor,
        nullptr, 0,
        workspace, workspaceSize,
        &alpha,
        output_tensor_descriptor, numerator,
        &beta,
        output_reduce_tensor_descriptor, numerator_reduce);

    cudaMemcpy(error.data(), numerator_reduce, sizeof(type), cudaMemcpyDeviceToHost);

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

    const cudnnTensorDescriptor_t& output_tensor_descriptor = back_propagation_cuda.output_tensor_descriptor;

    const cudnnOpTensorDescriptor_t& operator_sum_descriptor = back_propagation_cuda.operator_sum_descriptor;

    float* numerator = back_propagation_cuda.numerator;
    float* numerator_2 = back_propagation_cuda.numerator_2;
    float* numerator_3 = back_propagation_cuda.numerator_3;

    // Already calculated in calculate_error_cuda
    float* one_minus_targets = back_propagation_cuda.one_minus_targets;
    float* one_minus_outputs = back_propagation_cuda.one_minus_outputs;

    const type alpha = 1.0f;
    const type beta = 0.0f;
    const type beta_minus_one = -1.0f;

    const type scaling_factor = type(1.0) / type(samples_number);

    // (one_minus_targets) / (one_minus_outputs)
    division(size, one_minus_targets, one_minus_outputs, numerator);

    // -targets
    cudnnOpTensor(cudnn_handle,
        operator_sum_descriptor,
        &beta,
        output_tensor_descriptor,
        numerator_2,
        &beta_minus_one,
        output_tensor_descriptor,
        targets,
        &beta,
        output_tensor_descriptor,
        numerator_2);

    // (-targets / (outputs)
    division(size, numerator_2, outputs, numerator_3);

    // (-targets/outputs) + (one_minus_targets/one_minus_outputs)
    cudnnOpTensor(cudnn_handle,
        operator_sum_descriptor,
        &alpha,
        output_tensor_descriptor,
        numerator_3,
        &alpha,
        output_tensor_descriptor,
        numerator,
        &beta,
        output_tensor_descriptor,
        output_deltas);

    // output_deltas / samples_number
    cublasSscal(cublas_handle, size, &scaling_factor, output_deltas, 1);
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

    const cudnnTensorDescriptor_t& output_tensor_descriptor = back_propagation_cuda.output_tensor_descriptor;

    const cudnnOpTensorDescriptor_t& operator_sum_descriptor = back_propagation_cuda.operator_sum_descriptor;

    const type alpha = 1.0f;
    const type beta = 0.0f;
    const type beta_minus_one = -1.0f;

    // outputs - targets
    cudnnOpTensor(cudnn_handle,
        operator_sum_descriptor,
        &alpha,
        output_tensor_descriptor,
        outputs,
        &beta_minus_one,
        output_tensor_descriptor,
        targets,
        &beta,
        output_tensor_descriptor,
        output_deltas);

    // (outputs - targets) / samples_number
    const float scale_factor = 1.0f / static_cast<float>(samples_number);
    cudnnScaleTensor(cudnn_handle,output_tensor_descriptor,output_deltas,&scale_factor);
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
