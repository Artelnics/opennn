//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   M E A N   S Q U A R E D   E R R O R   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "mean_squared_error.h"

namespace OpenNN
{

/// Default constructor.
/// It creates a mean squared error term not associated to any
/// neural network and not measured on any data set.
/// It also initializes all the rest of class members to their default values.

MeanSquaredError::MeanSquaredError() : LossIndex()
{
}


/// Neural network constructor.
/// It creates a mean squared error term object associated to a
/// neural network object but not measured on any data set object.
/// It also initializes all the rest of class members to their default values.
/// @param new_neural_network_pointer Pointer to a neural network object.

MeanSquaredError::MeanSquaredError(NeuralNetwork* new_neural_network_pointer)
    : LossIndex(new_neural_network_pointer)
{
}


/// Data set constructor.
/// It creates a mean squared error term not associated to any
/// neural network but to be measured on a given data set object.
/// It also initializes all the rest of class members to their default values.
/// @param new_data_set_pointer Pointer to a data set object.

MeanSquaredError::MeanSquaredError(DataSet* new_data_set_pointer)
    : LossIndex(new_data_set_pointer)
{
}


/// Neural network and data set constructor.
/// It creates a mean squared error term object associated to a
/// neural network and measured on a data set.
/// It also initializes all the rest of class members to their default values.
/// @param new_neural_network_pointer Pointer to a neural network object.
/// @param new_data_set_pointer Pointer to a data set object.

MeanSquaredError::MeanSquaredError(NeuralNetwork* new_neural_network_pointer, DataSet* new_data_set_pointer)
    : LossIndex(new_neural_network_pointer, new_data_set_pointer)
{
}


/// XML constructor.
/// It creates a mean squared error object with all pointers set to nullptr.
/// The object members are loaded by means of a XML document.
/// Please be careful with the format of that file, which is specified in the OpenNN manual.
/// @param mean_squared_error_document TinyXML document with the mean squared error elements.

MeanSquaredError::MeanSquaredError(const tinyxml2::XMLDocument& mean_squared_error_document)
    : LossIndex(mean_squared_error_document)
{
    from_XML(mean_squared_error_document);
}


/// Copy constructor.
/// It creates a copy of an existing mean squared error object.
/// @param other_mean_squared_error Mean squared error object to be copied.

MeanSquaredError::MeanSquaredError(const MeanSquaredError& other_mean_squared_error)
    : LossIndex(other_mean_squared_error)
{
}


/// Destructor.

MeanSquaredError::~MeanSquaredError()
{
}



void MeanSquaredError::calculate_error(const DataSet::Batch& batch,
                     const NeuralNetwork::ForwardPropagation& forward_propagation,
                     LossIndex::BackPropagation& back_propagation) const
{
    Tensor<type, 0> sum_squared_error;

    const Index batch_instances_number = batch.inputs_2d.dimension(0);

    const Index trainable_layers_number = neural_network_pointer->get_trainable_layers_number();

    const Tensor<type, 2>& outputs = forward_propagation.layers(trainable_layers_number-1).activations_2d;
    const Tensor<type, 2>& targets = batch.targets_2d;

    Tensor<type, 2> errors(batch_instances_number, outputs.dimension(1));

    errors.device(*thread_pool_device) = outputs - targets;

    sum_squared_error.device(*thread_pool_device) = errors.contract(errors, SSE);

    back_propagation.error = sum_squared_error(0)/static_cast<type>(batch_instances_number);
}


void MeanSquaredError::calculate_error_terms(const DataSet::Batch& batch,
                                                   const NeuralNetwork::ForwardPropagation& forward_propagation,
                                                   SecondOrderLoss& second_order_loss) const
{

    const Index trainable_layers_number = neural_network_pointer->get_trainable_layers_number();

    const Index batch_instances_number = batch.get_instances_number();

    const Tensor<type, 2>& outputs = forward_propagation.layers(trainable_layers_number-1).activations_2d;
    const Tensor<type, 2>& targets = batch.targets_2d;

    second_order_loss.error_terms.resize(outputs.dimension(0));
    const Eigen::array<int, 1> rows_sum = {Eigen::array<int, 1>({1})};

    second_order_loss.error_terms.device(*thread_pool_device) = ((outputs - targets).square().sum(rows_sum)).sqrt();

    Tensor<type, 0> error;
    error.device(*thread_pool_device) = second_order_loss.error_terms.contract(second_order_loss.error_terms, AT_B);

    second_order_loss.error = error()/static_cast<type>(batch_instances_number);
}


void MeanSquaredError::calculate_output_gradient(const DataSet::Batch& batch,
                               const NeuralNetwork::ForwardPropagation& forward_propagation,
                               BackPropagation& back_propagation) const
{
     #ifdef __OPENNN_DEBUG__

     check();

     #endif

     const Index batch_instances_number = batch.inputs_2d.dimension(0);

     const type coefficient = static_cast<type>(2.0)/static_cast<type>(batch_instances_number);

     const Index trainable_layers_number = neural_network_pointer->get_trainable_layers_number();

     const Tensor<type, 2>& outputs = forward_propagation.layers(trainable_layers_number-1).activations_2d;
     const Tensor<type, 2>& targets = batch.targets_2d;

     Tensor<type, 2> errors(outputs.dimension(0), outputs.dimension(1));

     errors.device(*thread_pool_device) = outputs - targets;

     back_propagation.output_gradient.device(*thread_pool_device) = coefficient*errors;
}


void MeanSquaredError::calculate_Jacobian_gradient(const DataSet::Batch& batch,
                                                   LossIndex::SecondOrderLoss& second_order_loss) const
{
#ifdef __OPENNN_DEBUG__

    check();

#endif

    const Index batch_instances_number = batch.get_instances_number();

    const type coefficient = static_cast<type>(2)/static_cast<type>(batch_instances_number);

    second_order_loss.gradient.device(*thread_pool_device) = second_order_loss.error_Jacobian.contract(second_order_loss.error_terms, AT_B);

    second_order_loss.gradient.device(*thread_pool_device) = coefficient*second_order_loss.gradient;
}



void MeanSquaredError::calculate_hessian_approximation(const DataSet::Batch& batch, LossIndex::SecondOrderLoss& second_order_loss) const
{
     #ifdef __OPENNN_DEBUG__

     check();

     #endif

     const Index batch_instances_number = batch.inputs_2d.dimension(0);

     const type coefficient = (static_cast<type>(2.0)/static_cast<type>(batch_instances_number));

     second_order_loss.hessian.device(*thread_pool_device) = second_order_loss.error_Jacobian.contract(second_order_loss.error_Jacobian, AT_B);

     second_order_loss.hessian.device(*thread_pool_device) = coefficient*second_order_loss.hessian;
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


/// Serializes the cross entropy error object into a XML document of the TinyXML library without keep the DOM tree in memory.
/// See the OpenNN manual for more information about the format of this document

void MeanSquaredError::write_XML(tinyxml2::XMLPrinter& file_stream) const
{
    // Error type

    file_stream.OpenElement("MeanSquaredError");

    file_stream.CloseElement();

    // Regularization

//    write_regularization_XML(file_stream);
}

}


// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2020 Artificial Intelligence Techniques, SL.
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
