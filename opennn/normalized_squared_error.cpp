//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   N O R M A L I Z E D   S Q U A R E D   E R R O R   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "tensors.h"
#include "normalized_squared_error.h"
#include "forward_propagation.h"
#include "back_propagation.h"

namespace opennn
{

NormalizedSquaredError::NormalizedSquaredError(NeuralNetwork* new_neural_network, DataSet* new_data_set)
    : LossIndex(new_neural_network, new_data_set)
{
    set_normalization_coefficient();
}


void NormalizedSquaredError::set_data_set(DataSet* new_data_set)
{
    data_set = new_data_set;

    if(neural_network->has(Layer::Type::Recurrent)
        || neural_network->has(Layer::Type::LongShortTermMemory))
        set_time_series_normalization_coefficient();
    else
        set_normalization_coefficient();
}


void NormalizedSquaredError::set_normalization_coefficient()
{
    if (!has_data_set() || data_set->get_samples_number() == 0)
    {
        normalization_coefficient = type(1);
        return;
    }

    const Tensor<type, 1> training_target_means = data_set->calculate_means(DataSet::SampleUse::Training, DataSet::VariableUse::Target); 
    const Tensor<type, 2> training_target_data = data_set->get_data(DataSet::SampleUse::Training, DataSet::VariableUse::Target);

    normalization_coefficient = calculate_normalization_coefficient(training_target_data, training_target_means);
}


void NormalizedSquaredError::set_time_series_normalization_coefficient()
{
    const Tensor<type, 2> targets = data_set->get_data(DataSet::VariableUse::Target);

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
    if(has_neural_network() && has_data_set() && data_set->get_samples_number() != 0)
        set_normalization_coefficient();
    else
        normalization_coefficient = type(NAN);
}


type NormalizedSquaredError::calculate_normalization_coefficient(const Tensor<type, 2>& targets,
                                                                 const Tensor<type, 1>& targets_mean) const
{
    const Index rows_number = targets.dimension(0);

    Tensor<type, 0> new_normalization_coefficient;

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
    const Index total_samples_number = data_set->get_used_samples_number();

    // Batch

    const Index batch_samples_number = batch.get_samples_number();

    const pair<type*, dimensions> targets_pair = batch.get_target_pair();

    const TensorMap<Tensor<type, 2>> targets = tensor_map_2(targets_pair);

    // Forward propagation

    const pair<type*, dimensions> outputs_pair = forward_propagation.get_last_trainable_layer_outputs_pair();

    const TensorMap<Tensor<type, 2>> outputs = tensor_map_2(outputs_pair);

    // Back propagation

    Tensor<type, 2>& errors = back_propagation.errors;

    Tensor<type,0>& error = back_propagation.error;

    errors.device(*thread_pool_device) = outputs - targets;

    const type coefficient = type(total_samples_number) / type(batch_samples_number * normalization_coefficient);

    error.device(*thread_pool_device) =  errors.contract(errors, SSE) * coefficient;

    if(isnan(error())) throw runtime_error("\nError is NAN.");
}


void NormalizedSquaredError::calculate_error_lm(const Batch& batch,
                                                const ForwardPropagation&,
                                                BackPropagationLM& back_propagation) const
{
    const Index total_samples_number = data_set->get_samples_number();

    // Batch

    const Index batch_samples_number = batch.get_samples_number();

    // Back propagation

    Tensor<type, 1>& squared_errors = back_propagation.squared_errors;
    Tensor<type, 0>& error = back_propagation.error;

    const type coefficient = type(total_samples_number) / type(batch_samples_number * normalization_coefficient);

    error.device(*thread_pool_device) = squared_errors.square().sum() * coefficient;

    if(isnan(error())) throw runtime_error("\nError is NAN.");
}


void NormalizedSquaredError::calculate_output_delta(const Batch& batch,
                                                    ForwardPropagation&,
                                                    BackPropagation& back_propagation) const
{
    // Data set

    const Index total_samples_number = data_set->get_samples_number();

    // Batch

    const Index batch_samples_number = batch.get_samples_number();

    // Back propagation

    const Tensor<type, 2>& errors = back_propagation.errors;

    const pair<type*, dimensions> delta_pairs = back_propagation.get_output_deltas_pair();  

    TensorMap<Tensor<type, 2>> deltas = tensor_map_2(delta_pairs);

    const type coefficient = type(2*total_samples_number) / (type(batch_samples_number)*normalization_coefficient);

    deltas.device(*thread_pool_device) = coefficient*errors;
}


void NormalizedSquaredError::calculate_output_delta_lm(const Batch& ,
                                                       ForwardPropagation&,
                                                       BackPropagationLM & back_propagation) const
{
    const Tensor<type, 2>& errors = back_propagation.errors;
    const Tensor<type, 1>& squared_errors = back_propagation.squared_errors;

    const pair<type*, dimensions> output_deltas_pair = back_propagation.get_output_deltas_pair();

    TensorMap<Tensor<type, 2>> output_deltas = tensor_map_2(output_deltas_pair);

    output_deltas.device(*thread_pool_device) = errors;

    divide_columns(thread_pool_device.get(), output_deltas, squared_errors);
}


void NormalizedSquaredError::calculate_error_gradient_lm(const Batch& batch,
                                                         BackPropagationLM& back_propagation_lm) const
{
    const Index total_samples_number = data_set->get_samples_number();

    // Batch

    const Index batch_samples_number = batch.get_samples_number();

    // Back propagation

    Tensor<type, 1>& gradient = back_propagation_lm.gradient;

    const Tensor<type, 1>& squared_errors = back_propagation_lm.squared_errors;
    const Tensor<type, 2>& squared_errors_jacobian = back_propagation_lm.squared_errors_jacobian;

    const type coefficient = type(2* total_samples_number)/type(batch_samples_number* normalization_coefficient);

    gradient.device(*thread_pool_device) = squared_errors_jacobian.contract(squared_errors, AT_B)*coefficient;
}


void NormalizedSquaredError::calculate_error_hessian_lm(const Batch& batch,
                                                        BackPropagationLM& back_propagation_lm) const
{
    const Index total_samples_number = data_set->get_samples_number();

    // Batch

    const Index batch_samples_number = batch.get_samples_number();

    // Back propagation

    const Tensor<type, 2>& squared_errors_jacobian = back_propagation_lm.squared_errors_jacobian;

    Tensor<type, 2>& hessian = back_propagation_lm.hessian;

    const type coefficient = type(2)/((type(batch_samples_number)/type(total_samples_number))*normalization_coefficient);

    hessian.device(*thread_pool_device) = squared_errors_jacobian.contract(squared_errors_jacobian, AT_B)*coefficient;
}


string NormalizedSquaredError::get_loss_method() const
{
    return "NORMALIZED_SQUARED_ERROR";
}


string NormalizedSquaredError::get_error_type_text() const
{
    return "Normalized squared error";
}


void NormalizedSquaredError::to_XML(XMLPrinter& file_stream) const
{
    file_stream.OpenElement("NormalizedSquaredError");

    file_stream.CloseElement();
}


void NormalizedSquaredError::from_XML(const XMLDocument& document) const
{
    const XMLElement* root_element = document.FirstChildElement("NormalizedSquaredError");

    if(!root_element)
        throw runtime_error("Normalized squared element is nullptr.\n");
}

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2024 Artificial Intelligence Techniques, SL.
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
