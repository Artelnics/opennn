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
    : Loss(new_neural_network, new_dataset)
{
    name = "NormalizedSquaredError";

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
    if(!has_dataset() || dataset->get_samples_number() <= 1)
    {
        normalization_coefficient = type(1);
        return;
    }

    const VectorR training_target_means = dataset->calculate_means("Training", "Target");
    const MatrixR training_target_data = dataset->get_data("Training", "Target");

    normalization_coefficient = calculate_normalization_coefficient(training_target_data, training_target_means);
}


void NormalizedSquaredError::set_time_series_normalization_coefficient()
{
    const MatrixR targets = dataset->get_feature_data("Target");
    const Index rows = targets.rows() - 1;

    normalization_coefficient = (targets.topRows(rows) - targets.bottomRows(rows)).squaredNorm();
}


void NormalizedSquaredError::set_default()
{
    if(has_neural_network() && has_dataset() && dataset->get_samples_number() != 0)
        set_normalization_coefficient();
    else
        normalization_coefficient = type(NAN);
}


type NormalizedSquaredError::calculate_normalization_coefficient(const MatrixR& targets,
                                                                 const VectorR& targets_mean) const
{
    const type new_normalization_coefficient = (targets.rowwise() - targets_mean.transpose()).squaredNorm();

    return (new_normalization_coefficient < NUMERIC_LIMITS_MIN)
               ? static_cast<type>(1)
               : new_normalization_coefficient;
}


void NormalizedSquaredError::calculate_error(const Batch& batch,
                                             const ForwardPropagation& forward_propagation,
                                             BackPropagation& back_propagation) const
{
    const Index total_samples_number = dataset->get_used_samples_number();

    // Batch

    const Index samples_number = batch.get_samples_number();
    const MatrixMap targets = matrix_map(batch.get_targets());

    // Forward propagation

    const TensorView outputs_view = forward_propagation.get_last_trainable_layer_outputs();
    const MatrixMap outputs = matrix_map(outputs_view);

    // Back propagation

    MatrixR& errors = back_propagation.errors;

    errors = outputs - targets;

    const type coefficient = static_cast<type>(total_samples_number) /
                             static_cast<type>(samples_number * normalization_coefficient);

    back_propagation.error = errors.squaredNorm() * coefficient;

    if(isnan(back_propagation.error)) throw runtime_error("\nError is NAN.");
}


void NormalizedSquaredError::calculate_error_lm(const Batch&,
                                                const ForwardPropagation&,
                                                BackPropagationLM& back_propagation) const
{
    VectorR& squared_errors = back_propagation.squared_errors;

    type& error = back_propagation.error;

    error = squared_errors.squaredNorm() * static_cast<type>(0.5);

    if(isnan(error)) throw runtime_error("\nError is NAN.");
}


void NormalizedSquaredError::calculate_output_gradients(const Batch& batch,
                                                        ForwardPropagation&,
                                                        BackPropagation& back_propagation) const
{
    const Index total_samples_number = dataset->get_used_samples_number();
    const Index samples_number = batch.get_samples_number();

    const MatrixR& errors = back_propagation.errors;

    const TensorView output_gradient_views = back_propagation.get_output_gradients();

    MatrixMap output_gradients = matrix_map(output_gradient_views);

    const type coefficient = static_cast<type>(2.0 * total_samples_number) /
                             static_cast<type>(samples_number * normalization_coefficient);

    output_gradients = coefficient * errors;
}


void NormalizedSquaredError::calculate_squared_errors_lm(const Batch& batch,
                                                         const ForwardPropagation&,
                                                         BackPropagationLM& back_propagation_lm) const
{
    const Index total_samples_number = dataset->get_used_samples_number();
    const Index batch_size = batch.get_samples_number();

    const MatrixR& errors = back_propagation_lm.errors;

    VectorR& squared_errors = back_propagation_lm.squared_errors;

    const type coefficient = sqrt(static_cast<type>(2.0 * total_samples_number) /
                                  static_cast<type>(batch_size * normalization_coefficient));

    squared_errors = errors.reshaped() * coefficient;
}


void NormalizedSquaredError::calculate_output_gradients_lm(const Batch& batch,
                                                       ForwardPropagation&,
                                                       BackPropagationLM & back_propagation) const
{
    // @todo Is this correct?

    const Index total_samples_number = dataset->get_used_samples_number();
    const Index batch_samples_number = batch.get_samples_number();

    MatrixMap output_gradients = matrix_map(back_propagation.get_output_gradients());

    const type coefficient = sqrt(type(2.0 * total_samples_number) / (type(batch_samples_number) * normalization_coefficient));

    output_gradients.setConstant(coefficient);
}


void NormalizedSquaredError::calculate_error_gradient_lm(const Batch&,
                                                         BackPropagationLM& back_propagation_lm) const
{
    VectorR& gradient = back_propagation_lm.gradient;

    const VectorR& squared_errors = back_propagation_lm.squared_errors;

    const MatrixR& squared_errors_jacobian = back_propagation_lm.squared_errors_jacobian;

    gradient.noalias() = squared_errors_jacobian.transpose() * squared_errors;
}


void NormalizedSquaredError::calculate_error_hessian_lm(const Batch&,
                                                        BackPropagationLM& back_propagation_lm) const
{
    const MatrixR& squared_errors_jacobian = back_propagation_lm.squared_errors_jacobian;

    MatrixR& hessian = back_propagation_lm.hessian;

    hessian.noalias() = squared_errors_jacobian.transpose() * squared_errors_jacobian;
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

void NormalizedSquaredError::calculate_error(const BatchCuda&,
                                                  const ForwardPropagationCuda&,
                                                  BackPropagationCuda&) const
{
    throw runtime_error("CUDA calculate_error not implemented for loss index type: NormalizedSquaredError");
}


void NormalizedSquaredError::calculate_output_gradients(const BatchCuda&,
                                                         ForwardPropagationCuda&,
                                                         BackPropagationCuda&) const
{
    throw runtime_error("CUDA calculate_output_gradients not implemented for loss index type: NormalizedSquaredError");
}

#endif

REGISTER(Loss, NormalizedSquaredError, "NormalizedSquaredError");

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
