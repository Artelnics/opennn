//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   M E A N   S Q U A R E D   E R R O R   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "mean_squared_error.h"
#include "loss_index_back_propagation.h"

namespace opennn
{

/// Default constructor.
/// It creates a mean squared error term not associated with any
/// neural network and not measured on any data set.
/// It also initializes all the rest of the class members to their default values.

MeanSquaredError::MeanSquaredError() : LossIndex()
{
}


/// Neural network and data set constructor.
/// It creates a mean squared error term object associated with a
/// neural network and measured on a data set.
/// It also initializes all the rest of the class members to their default values.
/// @param new_neural_network_pointer Pointer to a neural network object.
/// @param new_data_set_pointer Pointer to a data set object.

MeanSquaredError::MeanSquaredError(NeuralNetwork* new_neural_network_pointer, DataSet* new_data_set_pointer)
    : LossIndex(new_neural_network_pointer, new_data_set_pointer)
{
}


/// \brief MeanSquaredError::calculate_error
/// \param batch
/// \param forward_propagation
/// \param back_propagation

void MeanSquaredError::calculate_error(const DataSetBatch& batch,
                                       const ForwardPropagation& forward_propagation,
                                       BackPropagation& back_propagation) const
{
    calculate_errors(batch, forward_propagation, back_propagation);


    const Index outputs_number = neural_network_pointer->get_outputs_number();
    const Index batch_samples_number = batch.get_batch_samples_number();

    // This line was needed in convolutional branch: 
    // const Index batch_samples_number = batch.inputs_2d.dimension(0) > 0 ? batch.inputs_2d.dimension(0) : batch.inputs_4d.dimension(0);

    const type coefficient = batch_samples_number > type(0) ? type(batch_samples_number) : type(1);

    Tensor<type, 0> sum_squared_error;

    sum_squared_error.device(*thread_pool_device) = back_propagation.errors.contract(back_propagation.errors, SSE);

    back_propagation.error = sum_squared_error(0)/coefficient;

    if(isnan(back_propagation.error)) throw invalid_argument("Error is NAN.");
}


void MeanSquaredError::calculate_error_lm(const DataSetBatch& batch,
                                          const ForwardPropagation&,
                                          LossIndexBackPropagationLM& back_propagation) const
{
    Tensor<type, 0> sum_squared_error;

    const Index outputs_number = neural_network_pointer->get_outputs_number();

    const Index batch_samples_number = batch.get_batch_samples_number();

    sum_squared_error.device(*thread_pool_device) = (back_propagation.squared_errors*back_propagation.squared_errors).sum();

    const type coefficient = type(1)/type(batch_samples_number*outputs_number);

    back_propagation.error = coefficient*sum_squared_error(0);
}


void MeanSquaredError::calculate_output_delta(const DataSetBatch& batch,
                                              ForwardPropagation&,
                                              BackPropagation& back_propagation) const
{
     #ifdef OPENNN_DEBUG
     check();
     #endif

     const Index trainable_layers_number = neural_network_pointer->get_trainable_layers_number();

     const Tensor<type, 2>& errors = back_propagation.errors;

     const LayerBackPropagation* output_layer_back_propagation = back_propagation.neural_network.layers(trainable_layers_number-1);

     // Check if works for convolutional

     const Index outputs_number = neural_network_pointer->get_outputs_number();

     const Index batch_samples_number = batch.get_batch_samples_number();

//     This line was written in convolutional. Without it, batch samples number was 0.
//     const Index batch_samples_number = batch.inputs_2d.dimension(0) == 0 ? batch.inputs_4d.dimension(0) : batch.inputs_2d.dimension(0);

     const type coefficient = type(2.0)/type(outputs_number*batch_samples_number);
     
     const pair<type*, dimensions> deltas_pair = output_layer_back_propagation->get_deltas_pair();

     TensorMap<Tensor<type, 2>> deltas(deltas_pair.first, deltas_pair.second[0][0], deltas_pair.second[0][1]);

     deltas.device(*thread_pool_device) = coefficient*errors;
}


void MeanSquaredError::calculate_output_delta_lm(const DataSetBatch&,
                                                 ForwardPropagation&,
                                                 LossIndexBackPropagationLM& loss_index_back_propagation) const
{
#ifdef OPENNN_DEBUG
    check();
#endif

    const Index trainable_layers_number = neural_network_pointer->get_trainable_layers_number();

    LayerBackPropagationLM* output_layer_back_propagation = loss_index_back_propagation.neural_network.layers(trainable_layers_number-1);

    const Layer* output_layer_pointer = output_layer_back_propagation->layer_pointer;

    const Layer::Type output_layer_type = output_layer_pointer->get_type();

    if(output_layer_type != Layer::Type::Perceptron && output_layer_type != Layer::Type::Probabilistic)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: MeanSquaredError class.\n"
               << "Levenberg-Marquardt can only be used with Perceptron and Probabilistic layers.\n";

        throw invalid_argument(buffer.str());
    }

    copy(execution::par,
         loss_index_back_propagation.errors.data(),
         loss_index_back_propagation.errors.data() + loss_index_back_propagation.errors.size(),
         output_layer_back_propagation->deltas.data());

    divide_columns(thread_pool_device,
                   output_layer_back_propagation->deltas,
                   loss_index_back_propagation.squared_errors);
}


void MeanSquaredError::calculate_error_gradient_lm(const DataSetBatch& batch,
                                             LossIndexBackPropagationLM& loss_index_back_propagation_lm) const
{
#ifdef OPENNN_DEBUG

    check();

#endif

    const Index outputs_number = neural_network_pointer->get_outputs_number();

    const Index batch_samples_number = outputs_number * batch.get_batch_samples_number();

    const type coefficient = type(2)/type(batch_samples_number);

    Tensor<type, 1>& gradient = loss_index_back_propagation_lm.gradient;

    const Tensor<type, 1>& squared_errors = loss_index_back_propagation_lm.squared_errors;
    const Tensor<type, 2>& squared_errors_jacobian = loss_index_back_propagation_lm.squared_errors_jacobian;

    gradient.device(*thread_pool_device)
        = (squared_errors_jacobian.contract(squared_errors, AT_B))*coefficient;
}


void MeanSquaredError::calculate_error_hessian_lm(const DataSetBatch& batch,
                                                       LossIndexBackPropagationLM& loss_index_back_propagation_lm) const
{
     #ifdef OPENNN_DEBUG
     check();
     #endif

     const Index outputs_number = neural_network_pointer->get_outputs_number();

     const Index batch_samples_number = outputs_number * batch.get_batch_samples_number();

     const type coefficient = type(2.0)/type(batch_samples_number);

     Tensor<type, 2>& hessian = loss_index_back_propagation_lm.hessian;

     const Tensor<type, 2>& squared_errors_jacobian = loss_index_back_propagation_lm.squared_errors_jacobian;

     hessian.device(*thread_pool_device)
         = (squared_errors_jacobian.contract(squared_errors_jacobian, AT_B))*coefficient;
}


/// Returns a string with the name of the mean squared error loss type, "MEAN_SQUARED_ERROR".

string MeanSquaredError::get_error_type() const
{
    return "MEAN_SQUARED_ERROR";
}


/// Returns a string with the name of the mean squared error loss type in text format.

string MeanSquaredError::get_error_type_text() const
{
    return "Mean squared error";
}


/// Serializes the cross-entropy error object into an XML document of the TinyXML library without keeping the DOM tree in memory.
/// See the OpenNN manual for more information about the format of this document

void MeanSquaredError::write_XML(tinyxml2::XMLPrinter& file_stream) const
{
    // Error type

    file_stream.OpenElement("MeanSquaredError");

    file_stream.CloseElement();
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
