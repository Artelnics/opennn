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
    : Loss(new_neural_network, new_dataset)
{
    name = "MeanSquaredError";
}


void MeanSquaredError::calculate_error(const Batch& batch,
                                       const ForwardPropagation& forward_propagation,
                                       BackPropagation& back_propagation) const
{
    if(!neural_network)
        throw runtime_error("MeanSquaredError: Neural network pointer is null.");

    const Index outputs_number = neural_network->get_outputs_number();
    const Index samples_number = batch.get_samples_number();

    if (outputs_number == 0 || samples_number == 0)
        throw runtime_error("MeanSquaredError: outputs_number or samples_number is zero.");

    const MatrixMap targets = matrix_map(batch.get_targets());
    const TensorView outputs_view = forward_propagation.get_last_trainable_layer_outputs();
    const MatrixMap outputs = matrix_map(outputs_view);

    MatrixR& errors = back_propagation.errors;

    if(outputs.rows() != targets.rows() || outputs.cols() != targets.cols())
        throw runtime_error("MeanSquaredError: outputs (" + to_string(outputs.rows()) + "x" + to_string(outputs.cols()) +
                            ") and targets (" + to_string(targets.rows()) + "x" + to_string(targets.cols()) + ") dimensions do not match.");

    errors = outputs - targets;

    back_propagation.error = errors.squaredNorm() / static_cast<type>(samples_number * outputs_number);

    if(isnan(back_propagation.error)) throw runtime_error("\nError is NAN.");
}


void MeanSquaredError::calculate_error_lm(const Batch&,
                                          const ForwardPropagation&,
                                          BackPropagationLM& back_propagation) const
{
    const VectorR& squared_errors = back_propagation.squared_errors;

    type& error = back_propagation.error;

    error = squared_errors.squaredNorm() * static_cast<type>(0.5);

    if(isnan(error)) throw runtime_error("\nError is NAN.");
}


void MeanSquaredError::calculate_output_gradients(const Batch& batch,
                                              ForwardPropagation&,
                                              BackPropagation& back_propagation) const
{
    const Index outputs_number = neural_network->get_outputs_number();

    // Batch

    const Index samples_number = batch.get_samples_number();

    // Back propagation

    const MatrixR& errors = back_propagation.errors;

    MatrixMap output_gradients = matrix_map(back_propagation.get_output_gradients());

    output_gradients = errors / type(0.5 * outputs_number * samples_number);
}


void MeanSquaredError::calculate_output_gradients_lm(const Batch&,
                                                 ForwardPropagation&,
                                                 BackPropagationLM& back_propagation) const
{
    MatrixMap output_gradients = matrix_map(back_propagation.get_output_gradients());

    output_gradients.array() = back_propagation.errors.array() /
                               (back_propagation.squared_errors.array() + numeric_limits<type>::epsilon());
}


void MeanSquaredError::calculate_error_gradient_lm(const Batch&,
                                                   BackPropagationLM& back_propagation_lm) const
{
    const VectorR& squared_errors = back_propagation_lm.squared_errors;

    const MatrixR& squared_errors_jacobian = back_propagation_lm.squared_errors_jacobian;

    VectorR& gradient = back_propagation_lm.gradient;

    gradient.noalias() = squared_errors_jacobian.transpose() * squared_errors;
}


void MeanSquaredError::calculate_error_hessian_lm(const Batch&,
                                                  BackPropagationLM& back_propagation_lm) const
{
    const MatrixR& squared_errors_jacobian = back_propagation_lm.squared_errors_jacobian;

    MatrixR& hessian = back_propagation_lm.hessian;

    hessian.noalias() = squared_errors_jacobian.transpose() * squared_errors_jacobian;
}


void MeanSquaredError::from_XML(const XMLDocument& document)
{
    const XMLElement* root_element = document.FirstChildElement("MeanSquaredError");

    if(!root_element)
        throw runtime_error("Mean squared element is nullptr.\n");
}


void MeanSquaredError::to_XML(XMLPrinter& file_stream) const
{
    file_stream.OpenElement("MeanSquaredError");

    file_stream.CloseElement();
}


#ifdef OPENNN_CUDA

void MeanSquaredError::calculate_error(const BatchCuda& batch,
                                            const ForwardPropagationCuda& forward_propagation,
                                            BackPropagationCuda& back_propagation) const
{
    const Index outputs_number = neural_network->get_outputs_number();

    // Batch

    const Index samples_number = batch.get_samples_number();

    const type* targets = batch.targets_device.data;

    // Forward propagation

    const TensorViewCuda outputs = forward_propagation.get_last_trainable_layer_outputs_device();

    // Back propagatioin

    type* errors_device = back_propagation.errors;

    type& error = back_propagation.error;

    const cudnnTensorDescriptor_t output_tensor_descriptor = back_propagation.output_gradients.get_descriptor();

    const float alpha_minus_one = -1.0f;

    cudnnOpTensor(get_cudnn_handle(),
                  get_operator_sum_descriptor(),
                  &alpha_minus_one,
                  output_tensor_descriptor,
                  targets,
                  &alpha,
                  output_tensor_descriptor,
                  outputs.data,
                  &beta,
                  output_tensor_descriptor,
                  errors_device);

    CHECK_CUBLAS(cublasSdot(get_cublas_handle(), samples_number * outputs_number, errors_device, 1, errors_device, 1, &error));

    const type coefficient = type(1.0)/type(samples_number * outputs_number);

    error *= coefficient;

    if (isnan(error)) throw runtime_error("\nError is NAN.");
}


void MeanSquaredError::calculate_output_gradients(const BatchCuda& batch,
                                                   ForwardPropagationCuda& forward_propagation,
                                                   BackPropagationCuda& back_propagation) const
{
    const Index outputs_number = neural_network->get_outputs_number();

    // Batch

    const Index samples_number = batch.get_samples_number();

    // Back propagation

    type* errors_device = back_propagation.errors;

    float* output_gradients_device = back_propagation.get_output_gradient_views_device().data;

    const type coefficient = type(2.0) / type(outputs_number * samples_number);

    cudaMemcpy(output_gradients_device, errors_device, outputs_number * samples_number * sizeof(float), cudaMemcpyDeviceToDevice);

    CHECK_CUBLAS(cublasSscal(get_cublas_handle(), outputs_number * samples_number, &coefficient, output_gradients_device, 1));
}

#endif

REGISTER(Loss, MeanSquaredError, "MeanSquaredError");

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
