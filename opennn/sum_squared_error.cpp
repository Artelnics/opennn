//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   S U M   S Q U A R E D   E R R O R   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "sum_squared_error.h"

namespace opennn
{

/// Default constructor.
/// It creates a sum squared error term not associated with any neural network and not measured on any data set.
/// It also initializes all the rest of the class members to their default values.

SumSquaredError::SumSquaredError() : LossIndex()
{
}


/// Neural network and data set constructor.
/// It creates a sum squared error associated with a neural network and measured on a data set.
/// It also initializes all the rest of the class members to their default values.
/// @param new_neural_network_pointer Pointer to a neural network object.
/// @param new_data_set_pointer Pointer to a data set object.

SumSquaredError::SumSquaredError(NeuralNetwork* new_neural_network_pointer, DataSet* new_data_set_pointer)
    : LossIndex(new_neural_network_pointer, new_data_set_pointer)
{
}


void SumSquaredError::calculate_error(const DataSetBatch&,
                     const NeuralNetworkForwardPropagation&,
                     LossIndexBackPropagation& back_propagation) const
{
    Tensor<type, 0> sum_squared_error;

    sum_squared_error.device(*thread_pool_device) = back_propagation.errors.contract(back_propagation.errors, SSE);

    back_propagation.error = sum_squared_error(0);
}


void SumSquaredError::calculate_error_lm(const DataSetBatch&,
                     const NeuralNetworkForwardPropagation&,
                     LossIndexBackPropagationLM& back_propagation) const
{
    Tensor<type, 0> sum_squared_error;

    sum_squared_error.device(*thread_pool_device) = (back_propagation.squared_errors*back_propagation.squared_errors).sum();

    back_propagation.error = sum_squared_error(0);
}


void SumSquaredError::calculate_output_delta(const DataSetBatch&,
                                             NeuralNetworkForwardPropagation&,
                                             LossIndexBackPropagation& back_propagation) const
{
     #ifdef OPENNN_DEBUG

     check();

     #endif

     const Index trainable_layers_number = neural_network_pointer->get_trainable_layers_number();

     LayerBackPropagation* output_layer_back_propagation = back_propagation.neural_network.layers(trainable_layers_number-1);

     const type coefficient = static_cast<type>(2.0);

     TensorMap<Tensor<type, 2>> deltas(output_layer_back_propagation->deltas_data, output_layer_back_propagation->deltas_dimensions(0), output_layer_back_propagation->deltas_dimensions(1));

     deltas.device(*thread_pool_device) = coefficient*back_propagation.errors;
}


void SumSquaredError::calculate_output_delta_lm(const DataSetBatch&,
                                                NeuralNetworkForwardPropagation&,
                                                LossIndexBackPropagationLM& loss_index_back_propagation) const
{
#ifdef OPENNN_DEBUG

    check();

#endif

    const Index trainable_layers_number = neural_network_pointer->get_trainable_layers_number();

    LayerBackPropagationLM* output_layer_back_propagation = loss_index_back_propagation.neural_network.layers(trainable_layers_number-1);

    const Layer* output_layer_pointer = output_layer_back_propagation->layer_pointer;

    if(output_layer_pointer->get_type() != Layer::Type::Perceptron && output_layer_pointer->get_type() != Layer::Type::Probabilistic)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: MeanSquaredError class.\n"
               << "Levenberg-Marquardt can only be used with Perceptron and Probabilistic layers.\n";

        throw invalid_argument(buffer.str());
    }

    copy(loss_index_back_propagation.errors.data(),
         loss_index_back_propagation.errors.data() + loss_index_back_propagation.errors.size(),
         output_layer_back_propagation->deltas.data());

    divide_columns(output_layer_back_propagation->deltas, loss_index_back_propagation.squared_errors);
}


void SumSquaredError::calculate_error_gradient_lm(const DataSetBatch& ,
                                            LossIndexBackPropagationLM& loss_index_back_propagation_lm) const
{
#ifdef OPENNN_DEBUG

    check();

#endif

    const type coefficient = (static_cast<type>(2.0));

    loss_index_back_propagation_lm.gradient.device(*thread_pool_device)
            = loss_index_back_propagation_lm.squared_errors_jacobian.contract(loss_index_back_propagation_lm.squared_errors, AT_B);

    loss_index_back_propagation_lm.gradient.device(*thread_pool_device)
            = coefficient*loss_index_back_propagation_lm.gradient;
}


void SumSquaredError::calculate_error_hessian_lm(const DataSetBatch&,
                                                      LossIndexBackPropagationLM& loss_index_back_propagation_lm) const
{
     #ifdef OPENNN_DEBUG

     check();

     #endif

     const type coefficient = static_cast<type>(2.0);

     loss_index_back_propagation_lm.hessian.device(*thread_pool_device)
             = loss_index_back_propagation_lm.squared_errors_jacobian.contract(loss_index_back_propagation_lm.squared_errors_jacobian, AT_B);

     loss_index_back_propagation_lm.hessian.device(*thread_pool_device)
             = coefficient*loss_index_back_propagation_lm.hessian;
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


/// Serializes the cross-entropy error object into an XML document of the TinyXML library without keeping the DOM tree in memory.
/// See the OpenNN manual for more information about the format of this document

void SumSquaredError::write_XML(tinyxml2::XMLPrinter& file_stream) const
{
    // Error type

    file_stream.OpenElement("SumSquaredError");

    file_stream.CloseElement();
}


/// Loads a sum squared error object from an XML document.
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

        throw invalid_argument(buffer.str());
    }
}

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2022 Artificial Intelligence Techniques, SL.
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
