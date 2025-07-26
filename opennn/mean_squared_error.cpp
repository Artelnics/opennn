//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   M E A N   S Q U A R E D   E R R O R   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "registry.h"
#include "tensors.h"
#include "dataset.h"
#include "neural_network.h"
#include "mean_squared_error.h"

namespace opennn
{

MeanSquaredError::MeanSquaredError(const NeuralNetwork* new_neural_network, const Dataset* new_dataset)
    : LossIndex(new_neural_network, new_dataset)
{
}


void MeanSquaredError::calculate_error(const Batch& batch,
                                       const ForwardPropagation& forward_propagation,
                                       BackPropagation& back_propagation) const
{
    if (!neural_network)
        throw runtime_error("MeanSquaredError: Neural network pointer is null.");

    const Index outputs_number = neural_network->get_outputs_number();
    const Index samples_number = batch.get_samples_number();

    if (outputs_number == 0 || samples_number == 0)
        throw runtime_error("MeanSquaredError: outputs_number or samples_number is zero.");

    const TensorMap<Tensor<type, 2>> targets = tensor_map<2>(batch.get_target_pair());
    
    const pair<type*, dimensions> outputs_pair = forward_propagation.get_last_trainable_layer_outputs_pair();

    const TensorMap<Tensor<type, 2>> outputs = tensor_map<2>(outputs_pair);

    // Back propagation
    
    Tensor<type, 2>& errors = back_propagation.errors;

    Tensor<type, 0>& error = back_propagation.error;

    if(outputs.dimension(0) != targets.dimension(0))
        throw runtime_error("MeanSquaredError: outputs and target dimension 0 do not match: " + to_string(outputs.dimension(0)) + " " + to_string(targets.dimension(0)));

    if(outputs.dimension(1) != targets.dimension(1))
        throw runtime_error("MeanSquaredError: outputs and target dimension 1 do not match: " + to_string(outputs.dimension(1)) + " " + to_string(targets.dimension(1)));

    errors.device(*thread_pool_device) = outputs - targets;
    
    error.device(*thread_pool_device) = errors.contract(errors, axes(0,0,1,1)) / type(samples_number * outputs_number);

    if(isnan(error())) throw runtime_error("\nError is NAN.");
}


void MeanSquaredError::calculate_error_lm(const Batch& batch,
                                          const ForwardPropagation&,
                                          BackPropagationLM& back_propagation) const
{
    const Index outputs_number = neural_network->get_outputs_number();
    
    const Index samples_number = batch.get_samples_number();

    Tensor<type, 1>& squared_errors = back_propagation.squared_errors;

    Tensor<type, 0>& error = back_propagation.error;

    error.device(*thread_pool_device) = squared_errors.square().sum() / type(samples_number * outputs_number);

    if(isnan(error())) throw runtime_error("\nError is NAN.");
}


void MeanSquaredError::calculate_output_delta(const Batch& batch,
                                              ForwardPropagation&,
                                              BackPropagation& back_propagation) const
{
    const Index outputs_number = neural_network->get_outputs_number();

    // Batch

    const Index samples_number = batch.get_samples_number();

    // Back propagation

    const Tensor<type, 2>& errors = back_propagation.errors;

    const pair<type*, dimensions> output_deltas_pair = back_propagation.get_output_deltas_pair();

    TensorMap<Tensor<type, 2>> output_deltas = tensor_map<2>(output_deltas_pair);

    output_deltas.device(*thread_pool_device) = errors / type(0.5 * outputs_number * samples_number);
}


void MeanSquaredError::calculate_output_delta_lm(const Batch&,
                                                 ForwardPropagation&,
                                                 BackPropagationLM& back_propagation) const
{
    const Tensor<type, 2>& errors = back_propagation.errors;
    const Tensor<type, 1>& squared_errors = back_propagation.squared_errors;

    const pair<type*, dimensions> output_deltas_pair = back_propagation.get_output_deltas_pair();

    TensorMap<Tensor<type, 2>> output_deltas = tensor_map<2>(output_deltas_pair);

    output_deltas.device(*thread_pool_device) = errors;
    divide_columns(thread_pool_device.get(), output_deltas, squared_errors);
}


void MeanSquaredError::calculate_error_gradient_lm(const Batch& batch,
                                                   BackPropagationLM& back_propagation_lm) const
{
    const Index outputs_number = neural_network->get_outputs_number();

    const Index samples_number = outputs_number * batch.get_samples_number();

    const type coefficient = type(2)/type(samples_number);

    const Tensor<type, 1>& squared_errors = back_propagation_lm.squared_errors;
    const Tensor<type, 2>& squared_errors_jacobian = back_propagation_lm.squared_errors_jacobian;

    Tensor<type, 1>& gradient = back_propagation_lm.gradient;

    gradient.device(*thread_pool_device) = squared_errors_jacobian.contract(squared_errors, axes(0,0))*coefficient;
}


void MeanSquaredError::calculate_error_hessian_lm(const Batch& batch,
                                                  BackPropagationLM& back_propagation_lm) const
{

    const Index outputs_number = neural_network->get_outputs_number();

    const Index samples_number = batch.get_samples_number();

    const type coefficient = type(2.0)/type(outputs_number*samples_number);

    Tensor<type, 2>& hessian = back_propagation_lm.hessian;

    const Tensor<type, 2>& squared_errors_jacobian = back_propagation_lm.squared_errors_jacobian;

    hessian.device(*thread_pool_device) = squared_errors_jacobian.contract(squared_errors_jacobian, axes(0,0))*coefficient;
}


string MeanSquaredError::get_name() const
{
    return "MeanSquaredError";
}


void MeanSquaredError::from_XML(const XMLDocument& document)
{
    const XMLElement* root_element = document.FirstChildElement("MeanSquaredError");

    if(!root_element)
        throw runtime_error("Mean squared element is nullptr.\n");

    // Regularization

    XMLDocument regularization_document;
    const XMLElement* regularization_element = root_element->FirstChildElement("Regularization");
    regularization_document.InsertFirstChild(regularization_element->DeepClone(&regularization_document));
    regularization_from_XML(regularization_document);
}


void MeanSquaredError::to_XML(XMLPrinter& file_stream) const
{
    file_stream.OpenElement("MeanSquaredError");

    file_stream.CloseElement();
}


#ifdef OPENNN_CUDA

void MeanSquaredError::calculate_error_cuda(const BatchCuda& batch_cuda,
                                            const ForwardPropagationCuda& forward_propagation_cuda,
                                            BackPropagationCuda& back_propagation_cuda) const
{

    const Index outputs_number = neural_network->get_outputs_number();

    // Batch 

    const Index samples_number = batch_cuda.get_samples_number();

    const type* targets = batch_cuda.targets_device;

    // Forward propagation

    const type* outputs = forward_propagation_cuda.get_last_trainable_layer_outputs_device();

    // Back propagatioin

    type* errors_device = back_propagation_cuda.errors;

    Tensor<type,0>& error = back_propagation_cuda.error;

    const cudnnTensorDescriptor_t& output_tensor_descriptor = back_propagation_cuda.output_tensor_descriptor;

    const cudnnOpTensorDescriptor_t& operator_sum_descriptor = back_propagation_cuda.operator_sum_descriptor;

    float alpha = 1.0f;
    float alpha_minus_one = -1.0f;
    const float beta = 0.0f;
    
    cudnnOpTensor(cudnn_handle,
        operator_sum_descriptor,
        &alpha_minus_one,
        output_tensor_descriptor,
        targets,
        &alpha,
        output_tensor_descriptor,
        outputs,
        &beta,
        output_tensor_descriptor,
        errors_device);

    cublasSdot(cublas_handle, samples_number * outputs_number, errors_device, 1, errors_device, 1, &error(0));

    const type coefficient = type(2.0)/type(samples_number * outputs_number);

    error(0) = error(0) * coefficient;

    if (isnan(error())) throw runtime_error("\nError is NAN.");
}


void MeanSquaredError::calculate_output_delta_cuda(const BatchCuda& batch_cuda,
                                                   ForwardPropagationCuda& forward_propagation_cuda,
                                                   BackPropagationCuda& back_propagation_cuda) const
{
    const Index outputs_number = neural_network->get_outputs_number();

    // Batch

    const Index samples_number = batch_cuda.get_samples_number();

    // Back propagation

    type* errors_device = back_propagation_cuda.errors;

    float* output_deltas_device = back_propagation_cuda.get_output_deltas_device();

    const type coefficient = type(2.0) / type(outputs_number * samples_number);

    cudaMemcpy(output_deltas_device, errors_device, outputs_number * samples_number * sizeof(float), cudaMemcpyDeviceToDevice);

    cublasSscal(cublas_handle, outputs_number * samples_number, &coefficient, output_deltas_device, 1);
}

#endif

REGISTER(LossIndex, MeanSquaredError, "MeanSquaredError");

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
