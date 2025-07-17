//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   N O R M A L I Z E D   S Q U A R E D   E R R O R   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "registry.h"
#include "tensors.h"
#include "dataset.h"
#include "normalized_squared_error.h"

namespace opennn
{

NormalizedSquaredError::NormalizedSquaredError(const NeuralNetwork* new_neural_network, const Dataset* new_dataset)
    : LossIndex(new_neural_network, new_dataset)
{
    set_normalization_coefficient();
}


void NormalizedSquaredError::set_dataset(const Dataset* new_dataset)
{
    dataset = const_cast<Dataset*>(new_dataset);

    neural_network->has("Recurrent")
        ? set_time_series_normalization_coefficient()
        : set_normalization_coefficient();
}


void NormalizedSquaredError::set_normalization_coefficient()
{
    if (!has_dataset() || dataset->get_samples_number() <= 1)
    {
        normalization_coefficient = type(1);
        return;
    }

    const Tensor<type, 1> training_target_means = dataset->calculate_means("Training", "Target");
    const Tensor<type, 2> training_target_data = dataset->get_data("Training", "Target");

    normalization_coefficient = calculate_normalization_coefficient(training_target_data, training_target_means);
}


void NormalizedSquaredError::set_time_series_normalization_coefficient()
{
    const Tensor<type, 2> targets = dataset->get_data_variables("Target");

    const Index rows = targets.dimension(0)-1;
    const Index columns = targets.dimension(1);

    Tensor<type, 2> targets_t(rows, columns);
    Tensor<type, 2> targets_t_1(rows, columns);

    for(Index i = 0; i < columns; i++)
        memcpy(targets_t_1.data() + targets_t_1.dimension(0) * i,
               targets.data() + targets.dimension(0) * i,
               rows * sizeof(type));

    for(Index i = 0; i < columns; i++)
        memcpy(targets_t.data() + targets_t.dimension(0) * i,
               targets.data() + targets.dimension(0) * i + 1,
               rows * sizeof(type));

    normalization_coefficient = calculate_time_series_normalization_coefficient(targets_t_1, targets_t);
}


type NormalizedSquaredError::calculate_time_series_normalization_coefficient(const Tensor<type, 2>& targets_t_1,
                                                                             const Tensor<type, 2>& targets_t) const
{
    const Index target_samples_number = targets_t_1.dimension(0);
    const Index target_variables_number = targets_t_1.dimension(1);

    type new_normalization_coefficient = type(0);

    #pragma omp parallel for reduction(+:new_normalization_coefficient)

    for(Index i = 0; i < target_samples_number; i++)
        for(Index j = 0; j < target_variables_number; j++)
            new_normalization_coefficient += (targets_t_1(i, j) - targets_t(i, j)) * (targets_t_1(i, j) - targets_t(i, j));

    return new_normalization_coefficient;
}



void NormalizedSquaredError::set_default()
{
    if(has_neural_network() && has_dataset() && dataset->get_samples_number() != 0)
        set_normalization_coefficient();
    else
        normalization_coefficient = type(NAN);
}


type NormalizedSquaredError::calculate_normalization_coefficient(const Tensor<type, 2>& targets,
                                                                 const Tensor<type, 1>& targets_mean) const
{
    const Index rows_number = targets.dimension(0);

    Tensor<type, 0> new_normalization_coefficient;
    new_normalization_coefficient.setZero();

    for(Index i = 0; i < rows_number; i++)
        new_normalization_coefficient.device(*thread_pool_device)
        += (targets.chip(i, 0) - targets_mean).square().sum();
    
    return new_normalization_coefficient() < NUMERIC_LIMITS_MIN
        ? type(1)
        : new_normalization_coefficient();
}


void NormalizedSquaredError::calculate_error(const Batch& batch,
                                             const ForwardPropagation& forward_propagation,
                                             BackPropagation& back_propagation) const
{
    const Index total_samples_number = dataset->get_used_samples_number();

    // Batch

    const Index samples_number = batch.get_samples_number();

    const pair<type*, dimensions> targets_pair = batch.get_target_pair();

    const TensorMap<Tensor<type, 2>> targets = tensor_map<2>(targets_pair);

    // Forward propagation

    const pair<type*, dimensions> outputs_pair = forward_propagation.get_last_trainable_layer_outputs_pair();

    const TensorMap<Tensor<type, 2>> outputs = tensor_map<2>(outputs_pair);

    // Back propagation

    Tensor<type, 2>& errors = back_propagation.errors;

    Tensor<type,0>& error = back_propagation.error;

    errors.device(*thread_pool_device) = outputs - targets;

    const type coefficient = type(total_samples_number) / type(samples_number * normalization_coefficient);

    error.device(*thread_pool_device) =  errors.contract(errors, axes(0,0,1,1)) * coefficient;

    if(isnan(error())) throw runtime_error("\nError is NAN.");
}


void NormalizedSquaredError::calculate_error_lm(const Batch& batch,
                                                const ForwardPropagation&,
                                                BackPropagationLM& back_propagation) const
{
    const Index total_samples_number = dataset->get_samples_number();

    // Batch

    const Index samples_number = batch.get_samples_number();

    // Back propagation

    Tensor<type, 1>& squared_errors = back_propagation.squared_errors;
    Tensor<type, 0>& error = back_propagation.error;

    const type coefficient = type(total_samples_number) / type(samples_number * normalization_coefficient);

    error.device(*thread_pool_device) = squared_errors.square().sum() * coefficient;

    if(isnan(error())) throw runtime_error("\nError is NAN.");
}


void NormalizedSquaredError::calculate_output_delta(const Batch& batch,
                                                    ForwardPropagation&,
                                                    BackPropagation& back_propagation) const
{
    // Data set

    const Index total_samples_number = dataset->get_samples_number();

    // Batch

    const Index samples_number = batch.get_samples_number();

    // Back propagation

    const Tensor<type, 2>& errors = back_propagation.errors;

    const pair<type*, dimensions> delta_pairs = back_propagation.get_output_deltas_pair();  

    TensorMap<Tensor<type, 2>> deltas = tensor_map<2>(delta_pairs);

    const type coefficient = type(2*total_samples_number) / (type(samples_number)*normalization_coefficient);

    deltas.device(*thread_pool_device) = coefficient*errors;
}


void NormalizedSquaredError::calculate_output_delta_lm(const Batch& ,
                                                       ForwardPropagation&,
                                                       BackPropagationLM & back_propagation) const
{
    const Tensor<type, 2>& errors = back_propagation.errors;
    const Tensor<type, 1>& squared_errors = back_propagation.squared_errors;

    const pair<type*, dimensions> output_deltas_pair = back_propagation.get_output_deltas_pair();

    TensorMap<Tensor<type, 2>> output_deltas = tensor_map<2>(output_deltas_pair);

    output_deltas.device(*thread_pool_device) = errors;
    divide_columns(thread_pool_device.get(), output_deltas, squared_errors);
}


void NormalizedSquaredError::calculate_error_gradient_lm(const Batch& batch,
                                                         BackPropagationLM& back_propagation_lm) const
{
    const Index total_samples_number = dataset->get_samples_number();

    // Batch

    const Index samples_number = batch.get_samples_number();

    // Back propagation

    Tensor<type, 1>& gradient = back_propagation_lm.gradient;

    const Tensor<type, 1>& squared_errors = back_propagation_lm.squared_errors;
    const Tensor<type, 2>& squared_errors_jacobian = back_propagation_lm.squared_errors_jacobian;

    const type coefficient = type(2* total_samples_number)/type(samples_number* normalization_coefficient);

    gradient.device(*thread_pool_device) = squared_errors_jacobian.contract(squared_errors, axes(0,0))*coefficient;
}


void NormalizedSquaredError::calculate_error_hessian_lm(const Batch& batch,
                                                        BackPropagationLM& back_propagation_lm) const
{
    const Index total_samples_number = dataset->get_samples_number();

    // Batch

    const Index samples_number = batch.get_samples_number();

    // Back propagation

    const Tensor<type, 2>& squared_errors_jacobian = back_propagation_lm.squared_errors_jacobian;

    Tensor<type, 2>& hessian = back_propagation_lm.hessian;

    const type coefficient = type(2)/((type(samples_number)/type(total_samples_number))*normalization_coefficient);

    hessian.device(*thread_pool_device) = squared_errors_jacobian.contract(squared_errors_jacobian, axes(0,0))*coefficient;
}


string NormalizedSquaredError::get_name() const
{
    return "NormalizedSquaredError";
}


void NormalizedSquaredError::to_XML(XMLPrinter& file_stream) const
{
    file_stream.OpenElement("NormalizedSquaredError");

    file_stream.CloseElement();
}


void NormalizedSquaredError::from_XML(const XMLDocument& document)
{
    const XMLElement* root_element = document.FirstChildElement("NormalizedSquaredError");

    if(!root_element)
        throw runtime_error("Normalized squared element is nullptr.\n");
}


#ifdef OPENNN_CUDA

void NormalizedSquaredError::calculate_error_cuda(const BatchCuda& batch_cuda,
                                                  const ForwardPropagationCuda& forward_propagation_cuda,
                                                  BackPropagationCuda& back_propagation_cuda) const
{
    throw runtime_error("CUDA calculate_error_cuda not implemented for loss index type: NormalizedSquaredError");
}


void NormalizedSquaredError::calculate_output_delta_cuda(const BatchCuda& batch_cuda,
                                                         ForwardPropagationCuda& forward_propagation_cuda,
                                                         BackPropagationCuda& back_propagation_cuda) const
{
    throw runtime_error("CUDA calculate_output_delta_cuda not implemented for loss index type: NormalizedSquaredError");
}

#endif

REGISTER(LossIndex, NormalizedSquaredError, "NormalizedSquaredError");

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
