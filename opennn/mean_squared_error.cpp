//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   M E A N   S Q U A R E D   E R R O R   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "tensors.h"
#include "mean_squared_error.h"
#include "forward_propagation.h"
#include "back_propagation.h"

namespace opennn
{

MeanSquaredError::MeanSquaredError(NeuralNetwork* new_neural_network, DataSet* new_data_set)
    : LossIndex(new_neural_network, new_data_set)
{
}


void MeanSquaredError::calculate_error(const Batch& batch,
                                       const ForwardPropagation& forward_propagation,
                                       BackPropagation& back_propagation) const
{
    const Index outputs_number = neural_network->get_outputs_number();

    // Batch

    const Index samples_number = batch.get_samples_number();

    const pair<type*, dimensions> targets_pair = batch.get_target_pair();

    const TensorMap<Tensor<type, 2>> targets = tensor_map_2(targets_pair);

    // Forward propagation
    
    const pair<type*, dimensions> outputs_pair = forward_propagation.get_last_trainable_layer_outputs_pair();

    const TensorMap<Tensor<type, 2>> outputs = tensor_map_2(outputs_pair);

    // Back propagation
    
    Tensor<type, 2>& errors = back_propagation.errors;

    Tensor<type, 0>& error = back_propagation.error;
    
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

    TensorMap<Tensor<type, 2>> output_deltas = tensor_map_2(output_deltas_pair);

    output_deltas.device(*thread_pool_device) = errors / type(0.5 * outputs_number * samples_number);
}


void MeanSquaredError::calculate_output_delta_lm(const Batch&,
                                                 ForwardPropagation&,
                                                 BackPropagationLM& back_propagation) const
{
    const Tensor<type, 2>& errors = back_propagation.errors;
    const Tensor<type, 1>& squared_errors = back_propagation.squared_errors;

    const pair<type*, dimensions> output_deltas_pair = back_propagation.get_output_deltas_pair();

    TensorMap<Tensor<type, 2>> output_deltas = tensor_map_2(output_deltas_pair);

    // output_deltas.device(*thread_pool_device) = errors
    //     / squared_errors.reshape(array<Index, 2>({1, squared_errors.size()}))
    //                     .broadcast(array<Index, 2>({output_deltas.dimension(0), 1}));

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


string MeanSquaredError::get_loss_method() const
{
    return "MEAN_SQUARED_ERROR";
}


string MeanSquaredError::get_error_type_text() const
{
    return "Mean squared error";
}


void MeanSquaredError::to_XML(XMLPrinter& file_stream) const
{
    file_stream.OpenElement("MeanSquaredError");

    file_stream.CloseElement();
}


#ifdef OPENNN_CUDA_test

void MeanSquaredError::calculate_error_cuda(const BatchCuda& batch_cuda,
                                            const ForwardPropagationCuda& forward_propagation_cuda,
                                            BackPropagationCuda& back_propagation_cuda) const
{

    const Index outputs_number = neural_network->get_outputs_number();

    // Batch 

    const Index samples_number = batch_cuda.get_samples_number();

    const type* targets = batch_cuda.targets_device;

    // Forward propagation

    const pair<type*, dimensions> outputs_pair = forward_propagation_cuda.get_last_trainable_layer_outputs_pair_device();

    const type* outputs = outputs_pair.first;

    // Back propagatioin

    type* errors_device = back_propagation_cuda.errors;

    Tensor<type,0>& error = back_propagation_cuda.error;

    const cudnnTensorDescriptor_t& outputs_tensor_descriptor = back_propagation_cuda.outputs_tensor_descriptor;

    const cudnnOpTensorDescriptor_t& operator_sum_descriptor = back_propagation_cuda.operator_sum_descriptor;

    float alpha = 1.0f;
    float alpha_minus_one = -1.0f;
    const float beta = 0.0f;
    
    cudnnOpTensor(cudnn_handle,
        operator_sum_descriptor,
        &alpha_minus_one,
        outputs_tensor_descriptor,
        targets,
        &alpha,
        outputs_tensor_descriptor,
        outputs,
        &beta,
        outputs_tensor_descriptor,
        errors_device);

    float mean_square_error = 0.0f;

    cublasSdot(cublas_handle, samples_number * outputs_number, errors_device, 1, errors_device, 1, &mean_square_error);

    const type coefficient = type(2.0)/type(samples_number * outputs_number);

    error(0) = mean_square_error * coefficient;

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

    const pair<type*, dimensions> output_deltas_pair_device = back_propagation_cuda.get_output_deltas_pair_device();

    type* output_deltas_device = output_deltas_pair_device.first;

    const type coefficient = type(2.0) / type(outputs_number * samples_number);

    cudaMemcpy(output_deltas_device, errors_device, outputs_number * samples_number * sizeof(float), cudaMemcpyDeviceToDevice);

    cublasSscal(cublas_handle, outputs_number * samples_number, &coefficient, output_deltas_device, 1);
}

#endif

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
