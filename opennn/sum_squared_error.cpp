//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   S U M   S Q U A R E D   E R R O R   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "sum_squared_error.h"

namespace OpenNN
{

/// Default constructor.
/// It creates a sum squared error term not associated to any neural network and not measured on any data set.
/// It also initializes all the rest of class members to their default values.

SumSquaredError::SumSquaredError() : LossIndex()
{
}


/// Neural network and data set constructor.
/// It creates a sum squared error associated to a neural network and measured on a data set.
/// It also initializes all the rest of class members to their default values.
/// @param new_neural_network_pointer Pointer to a neural network object.
/// @param new_data_set_pointer Pointer to a data set object.

SumSquaredError::SumSquaredError(NeuralNetwork* new_neural_network_pointer, DataSet* new_data_set_pointer)
    : LossIndex(new_neural_network_pointer, new_data_set_pointer)
{
}


/// Destructor.

SumSquaredError::~SumSquaredError()
{
}


void SumSquaredError::calculate_error(const DataSet::Batch& batch,
                     const NeuralNetwork::ForwardPropagation& forward_propagation,
                     LossIndex::BackPropagation& back_propagation) const
{
    Tensor<type, 0> sum_squared_error;

    const Index trainable_layers_number = neural_network_pointer->get_trainable_layers_number();

    const Tensor<type, 2>& outputs = forward_propagation.layers(trainable_layers_number-1).activations_2d;
    const Tensor<type, 2>& targets = batch.targets_2d;

    back_propagation.errors.device(*thread_pool_device) = outputs - targets;

    sum_squared_error.device(*thread_pool_device) = back_propagation.errors.contract(back_propagation.errors, SSE);

    back_propagation.error = sum_squared_error(0);

    return;
}


void SumSquaredError::calculate_error_terms(const DataSet::Batch& batch,
                                            const NeuralNetwork::ForwardPropagation& forward_propagation,
                                            SecondOrderLoss& second_order_loss) const
{
    const Index trainable_layers_number = neural_network_pointer->get_trainable_layers_number();

    const Tensor<type, 2>& outputs = forward_propagation.layers(trainable_layers_number-1).activations_2d;
    const Tensor<type, 2>& targets = batch.targets_2d;

    second_order_loss.error_terms.resize(outputs.dimension(0));
    const Eigen::array<int, 1> rows_sum = {Eigen::array<int, 1>({1})};

    second_order_loss.error_terms.device(*thread_pool_device) = ((outputs - targets).square().sum(rows_sum)).sqrt();

    Tensor<type, 0> error;
    error.device(*thread_pool_device) = second_order_loss.error_terms.contract(second_order_loss.error_terms, AT_B);

    second_order_loss.error = error();
}


void SumSquaredError::calculate_output_gradient(const DataSet::Batch& batch,
                               const NeuralNetwork::ForwardPropagation& forward_propagation,
                               BackPropagation& back_propagation) const
{
     #ifdef __OPENNN_DEBUG__

     check();

     #endif

     const type coefficient = static_cast<type>(2.0);

     const Index trainable_layers_number = neural_network_pointer->get_trainable_layers_number();

     const Tensor<type, 2>& outputs = forward_propagation.layers(trainable_layers_number-1).activations_2d;
     const Tensor<type, 2>& targets = batch.targets_2d;

     back_propagation.errors.device(*thread_pool_device) = outputs - targets;

     back_propagation.output_gradient.device(*thread_pool_device) = coefficient*back_propagation.errors;

}

void SumSquaredError::calculate_Jacobian_gradient(const DataSet::Batch& ,
                                    LossIndex::SecondOrderLoss& second_order_loss) const
{
#ifdef __OPENNN_DEBUG__

    check();

#endif

    const type coefficient = (static_cast<type>(2.0));

    second_order_loss.gradient.device(*thread_pool_device) = second_order_loss.error_terms_Jacobian.contract(second_order_loss.error_terms, AT_B);

    second_order_loss.gradient.device(*thread_pool_device) = coefficient*second_order_loss.gradient;

}

// Hessian method

void SumSquaredError::calculate_hessian_approximation(const DataSet::Batch&, LossIndex::SecondOrderLoss& second_order_loss) const
{
     #ifdef __OPENNN_DEBUG__

     check();

     #endif

     const type coefficient = (static_cast<type>(2.0));

     second_order_loss.hessian.device(*thread_pool_device) = second_order_loss.error_terms_Jacobian.contract(second_order_loss.error_terms_Jacobian, AT_B);

     second_order_loss.hessian.device(*thread_pool_device) = coefficient*second_order_loss.hessian;
}


/// Returns a string with the name of the sum squared error loss type, "SUM_SQUARED_ERROR".

string SumSquaredError::get_error_type() const
{
    return "SUM_SQUARED_ERROR";
}


/// Returns a string with the name of the sum squared error loss type in text format.

string SumSquaredError::get_error_type_text() const
{
    return "Sum squared error";
}


/// Serializes the cross entropy error object into a XML document of the TinyXML library without keep the DOM tree in memory.
/// See the OpenNN manual for more information about the format of this document

void SumSquaredError::write_XML(tinyxml2::XMLPrinter& file_stream) const
{
    // Error type

    file_stream.OpenElement("SumSquaredError");

    file_stream.CloseElement();
}


/// Loads a sum squared error object from a XML document.
/// @param document TinyXML document containing the members of the object.

void SumSquaredError::from_XML(const tinyxml2::XMLDocument& document)
{
    const tinyxml2::XMLElement* root_element = document.FirstChildElement("SumSquaredError");

    if(!root_element)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: SumSquaredError class.\n"
               << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
               << "Sum squared element is nullptr.\n";

        throw logic_error(buffer.str());
    }
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
