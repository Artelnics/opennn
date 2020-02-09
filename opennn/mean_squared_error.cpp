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


/// This method separates training instances and calculates batches from the dataset.
/// It also calculates the outputs and the sum squared error from the targets and outputs.
/// Returns a sum squared error of the training instances.

type MeanSquaredError::calculate_training_error() const
{

#ifdef __OPENNN_DEBUG__

check();

#endif

    //Neural network

     bool is_forecasting = false;

    if(neural_network_pointer->has_long_short_term_memory_layer() || neural_network_pointer->has_recurrent_layer()) is_forecasting = true;

    // Data set

    const Tensor<Index, 2> training_batches = data_set_pointer->get_training_batches(!is_forecasting);

    const Index training_instances_number = data_set_pointer->get_training_instances_number();

    const Index batches_number = training_batches.size();

    const Index batch_instances_number = data_set_pointer->get_batch_instances_number();

    const Index inputs_number = data_set_pointer->get_input_variables_number();
    const Index targets_number = data_set_pointer->get_target_variables_number();

    Tensor<type, 2> inputs(batch_instances_number, inputs_number);
    Tensor<type, 2> targets(batch_instances_number, targets_number);
    Tensor<type, 2> outputs(batch_instances_number, targets_number);

    // Mean squared error

    type training_error = static_cast<type>(0.0);

    #pragma omp parallel for reduction(+ : training_error)

    for(Index i = 0; i < batches_number; i++)
    {
        inputs = data_set_pointer->get_input_data(training_batches.chip(i,0));
        targets = data_set_pointer->get_target_data(training_batches.chip(i,0));

        outputs = neural_network_pointer->calculate_trainable_outputs(inputs);

        const type batch_error = sum_squared_error(outputs, targets);

        training_error += batch_error;
    }

    return training_error/training_instances_number;
}


type MeanSquaredError::calculate_training_error(const Tensor<type, 1>& parameters) const
{
#ifdef __OPENNN_DEBUG__

check();

#endif

    //Neural network

     bool is_forecasting = false;

    if(neural_network_pointer->has_long_short_term_memory_layer() || neural_network_pointer->has_recurrent_layer()) is_forecasting = true;

    // Data set

    const Tensor<Index, 2> training_batches = data_set_pointer->get_training_batches(!is_forecasting);
    const Index training_instances_number = data_set_pointer->get_training_instances_number();

    const Index batches_number = training_batches.size();

    const Index batch_instances_number = data_set_pointer->get_batch_instances_number();

    const Index inputs_number = data_set_pointer->get_input_variables_number();
    const Index targets_number = data_set_pointer->get_target_variables_number();

    Tensor<type, 2> inputs(batch_instances_number, inputs_number);
    Tensor<type, 2> targets(batch_instances_number, targets_number);
    Tensor<type, 2> outputs(batch_instances_number, targets_number);

    // Mean squared error

    type training_error = static_cast<type>(0.0);

     #pragma omp parallel for reduction(+ : training_error)

    for(Index i = 0; i < batches_number; i++)
    {
        inputs = data_set_pointer->get_input_data(training_batches.chip(i,0));
        targets = data_set_pointer->get_target_data(training_batches.chip(i,0));

        outputs = neural_network_pointer->calculate_trainable_outputs(inputs, parameters);

        const type batch_error = sum_squared_error(outputs, targets);

        training_error += batch_error;
    }

    return training_error/training_instances_number;
}


/// This method separates selection instances and calculates batches from the dataset.
/// It also calculates the outputs and the sum squared error from the targets and outputs.
/// Returns a sum squared error of the training instances.

type MeanSquaredError::calculate_selection_error() const
{

#ifdef __OPENNN_DEBUG__

check();

#endif

    // Data set

    const Index selection_instances_number = data_set_pointer->get_selection_instances_number();

     //Neural network

     bool is_forecasting = false;

    if(neural_network_pointer->has_long_short_term_memory_layer() || neural_network_pointer->has_recurrent_layer()) is_forecasting = true;

    // Data set

    const Tensor<Index, 2> selection_batches = data_set_pointer->get_selection_batches(!is_forecasting);

    const Index batches_number = selection_batches.size();

    const Index batch_instances_number = data_set_pointer->get_batch_instances_number();

    const Index inputs_number = data_set_pointer->get_input_variables_number();
    const Index targets_number = data_set_pointer->get_target_variables_number();

    Tensor<type, 2> inputs(batch_instances_number, inputs_number);
    Tensor<type, 2> targets(batch_instances_number, targets_number);
    Tensor<type, 2> outputs(batch_instances_number, targets_number);

    type selection_error = static_cast<type>(0.0);

     #pragma omp parallel for reduction(+ : selection_error)

    for(Index i = 0; i < batches_number; i++)
    {
        inputs = data_set_pointer->get_input_data(selection_batches.chip(i,0));
        targets = data_set_pointer->get_target_data(selection_batches.chip(i,0));

        outputs = neural_network_pointer->calculate_trainable_outputs(inputs);

        const type batch_error = sum_squared_error(outputs, targets);

        selection_error += batch_error;
    }

    return selection_error/selection_instances_number;
}


/// This method calculates the mean squared error of the given batch.
/// Returns the mean squared error of this batch.
/// @param batch_indices Indices of the batch instances corresponding to the dataset.

type MeanSquaredError::calculate_batch_error(const Tensor<Index, 1>& batch_indices) const
{
#ifdef __OPENNN_DEBUG__

check();

#endif

    // Loss index

    const Tensor<type, 2> inputs = data_set_pointer->get_input_data(batch_indices);
    const Tensor<type, 2> targets = data_set_pointer->get_target_data(batch_indices);

    const Tensor<type, 2> outputs = neural_network_pointer->calculate_trainable_outputs(inputs);

    return sum_squared_error(outputs, targets);
}


type MeanSquaredError::calculate_batch_error(const Tensor<Index, 1>& batch_indices,
                                               const Tensor<type, 1>& parameters) const
{
#ifdef __OPENNN_DEBUG__

check();

#endif

    // Loss index

    const Tensor<type, 2> inputs = data_set_pointer->get_input_data(batch_indices);

    const Tensor<type, 2> targets = data_set_pointer->get_target_data(batch_indices);

    const Tensor<type, 2> outputs = neural_network_pointer->calculate_trainable_outputs(inputs, parameters);

    return sum_squared_error(outputs, targets);
}


/// This method calculates the first order loss.
/// It is used for optimization of parameters during training.
/// Returns a first order terms loss structure, which contains the values and the Jacobian of the error terms function.

LossIndex::BackPropagation MeanSquaredError::calculate_back_propagation() const
{
#ifdef __OPENNN_DEBUG__

check();

#endif

    // Neural network

    const Index layers_number = neural_network_pointer->get_trainable_layers_number();

     bool is_forecasting = false;

    if(neural_network_pointer->has_long_short_term_memory_layer() || neural_network_pointer->has_recurrent_layer()) is_forecasting = true;

    // Data set

    const Tensor<Index, 2> training_batches = data_set_pointer->get_training_batches(!is_forecasting);

    const Index training_instances_number = data_set_pointer->get_training_instances_number();

    const Index batches_number = training_batches.size();

    BackPropagation back_propagation(this);

    // Eigen stuff
/*

     #pragma omp parallel for

    for(Index i = 0; i < batches_number; i++)
    {
        const Tensor<type, 2> inputs = data_set_pointer->get_input_data(training_batches.chip(i,0));
        const Tensor<type, 2> targets = data_set_pointer->get_target_data(training_batches.chip(i,0));

        const Tensor<Layer::ForwardPropagation, 1> forward_propagation
                = neural_network_pointer->calculate_forward_propagation(inputs);

        const Tensor<type, 1> error_terms
                = calculate_training_error_terms(forward_propagation[layers_number-1].activations, targets);

//        const Tensor<type, 2> output_gradient = (forward_propagation[layers_number-1].activations - targets).divide(error_terms, 0);

        const Tensor<Tensor<type, 2>, 1> layers_delta
                = calculate_layers_delta(forward_propagation,
                                         output_gradient);

        const Tensor<type, 2> error_terms_Jacobian
                = calculate_error_terms_Jacobian(inputs, forward_propagation, layers_delta);

//        const Tensor<type, 2> error_terms_Jacobian_transpose = error_terms_Jacobian.calculate_transpose();

        const Tensor<type, 0> loss = error_terms.contract(error_terms, product_vector_vector);//dot(error_terms, error_terms);

        const Tensor<type, 1> gradient = error_terms_Jacobian.contract(error_terms,product_matrix_vector);//dot(error_terms_Jacobian_transpose, error_terms);

          #pragma omp critical
        {
            back_propagation.loss += loss(0);
            back_propagation.gradient += gradient;
         }
    }

    back_propagation.loss /= static_cast<type>(training_instances_number);
    back_propagation.gradient = (2.0/static_cast<type>(training_instances_number))*back_propagation.gradient;
*/
    return back_propagation;
}


/// Returns loss vector of the error terms function for the mean squared error.
/// It uses the error back-propagation method.
/// @param outputs Tensor with the values of the outputs.
/// @param targets Tensor with the values of the targets.

Tensor<type, 1> MeanSquaredError::calculate_training_error_terms(const Tensor<type, 2>& outputs, const Tensor<type, 2>& targets) const
{
   // Control sentence

   #ifdef __OPENNN_DEBUG__

   check();

   #endif
/*
   return error_rows(outputs, targets);
  */

    return Tensor<type, 1>();

}


/// This method separates instances from training.
/// It also calculates the outputs and the sum squared error from the targets and outputs.
/// Returns a mean squared error of the training instances.
/// @param parameters

Tensor<type, 1> MeanSquaredError::calculate_training_error_terms(const Tensor<type, 1>& parameters) const
{
#ifdef __OPENNN_DEBUG__

check();

#endif

    const Tensor<type, 2> inputs = data_set_pointer->get_training_input_data();

    const Tensor<type, 2> targets = data_set_pointer->get_training_target_data();

    const Tensor<type, 2> outputs = neural_network_pointer->calculate_trainable_outputs(inputs, parameters);

    const Index training_instances_number = data_set_pointer->get_training_instances_number();
/*
    return error_rows(outputs, targets)/static_cast<type>(training_instances_number);
*/
    return Tensor<type, 1>();
}


/// This method calculates the second order loss.
/// It is used for optimization of parameters during training.
/// Returns a second order terms loss structure, which contains the values and the Hessian of the error terms function.

LossIndex::SecondOrderLoss MeanSquaredError::calculate_terms_second_order_loss() const
{

#ifdef __OPENNN_DEBUG__

check();

#endif
/*
    // Neural network

    const Index layers_number = neural_network_pointer->get_trainable_layers_number();

    const Index parameters_number = neural_network_pointer->get_parameters_number();

     bool is_forecasting = false;

    if(neural_network_pointer->has_long_short_term_memory_layer() || neural_network_pointer->has_recurrent_layer()) is_forecasting = true;

    // Data set

    const Tensor<Index, 2> training_batches = data_set_pointer->get_training_batches(!is_forecasting);

    const Index training_instances_number = data_set_pointer->get_training_instances_number();

    const Index batches_number = training_batches.size();

    SecondOrderLoss terms_second_order_loss(parameters_number);

    // Eigen stuff

     #pragma omp parallel for

    for(Index i = 0; i < batches_number; i++)
    {
        const Tensor<type, 2> inputs = data_set_pointer->get_input_data(training_batches.chip(i,0));
        const Tensor<type, 2> targets = data_set_pointer->get_target_data(training_batches.chip(i,0));

        const Tensor<Layer::ForwardPropagation, 1> forward_propagation = neural_network_pointer->calculate_forward_propagation(inputs);

        const Tensor<type, 1> error_terms = calculate_training_error_terms(forward_propagation[layers_number-1].activations, targets);

        const Tensor<type, 2> output_gradient = (forward_propagation[layers_number-1].activations - targets).divide(error_terms, 0);

        const Tensor<Tensor<type, 2>, 1> layers_delta = calculate_layers_delta(forward_propagation, output_gradient);

        const Tensor<type, 2> error_terms_Jacobian = calculate_error_terms_Jacobian(inputs, forward_propagation, layers_delta);

//        const Tensor<type, 2> error_terms_Jacobian_transpose = error_terms_Jacobian.calculate_transpose();

        //dot(error_terms, error_terms);

        const Tensor<type, 0> loss = error_terms.contract(error_terms, product_vector_vector);

        //dot(error_terms_Jacobian_transpose, error_terms);

        const Tensor<type, 1> gradient = error_terms_Jacobian.contract(error_terms, product_matrix_transpose_vector);

        Tensor<type, 2> hessian_approximation = error_terms_Jacobian.contract(error_terms_Jacobian, product_matrix_transpose_matrix);// = error_terms_Jacobian.dot(error_terms_Jacobian);
        //hessian_approximation.dot(error_terms_Jacobian_transpose, error_terms_Jacobian);

        #pragma omp critical
        {
            terms_second_order_loss.loss += loss(0);
            terms_second_order_loss.gradient += gradient;
            terms_second_order_loss.hessian += hessian_approximation;
         }

    }

    terms_second_order_loss.loss /= static_cast<type>(training_instances_number);
    terms_second_order_loss.gradient *= (static_cast<type>(2.0)/static_cast<type>(training_instances_number));
    terms_second_order_loss.hessian *= (static_cast<type>(2.0)/static_cast<type>(training_instances_number));

    if(regularization_method != RegularizationMethod::NoRegularization)
    {
        terms_second_order_loss.loss += calculate_regularization();
        terms_second_order_loss.gradient += calculate_regularization_gradient();
        terms_second_order_loss.hessian += calculate_regularization_hessian();
    }

    return terms_second_order_loss;
  */
    const Index parameters_number = neural_network_pointer->get_parameters_number();
    SecondOrderLoss terms_second_order_loss(parameters_number);
    return  terms_second_order_loss;
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


/// Serializes the mean squared error object into a XML document of the TinyXML library.
/// See the OpenNN manual for more information about the format of this document->

tinyxml2::XMLDocument* MeanSquaredError::to_XML() const
{
   ostringstream buffer;

   tinyxml2::XMLDocument* document = new tinyxml2::XMLDocument;

   // Mean squared error

   tinyxml2::XMLElement* mean_squared_error_element = document->NewElement("MeanSquaredError");

   document->InsertFirstChild(mean_squared_error_element);

   // Display
//   {
//      tinyxml2::XMLElement* element = document->NewElement("Display");
//      mean_squared_error_element->LinkEndChild(element);

//      buffer.str("");
//      buffer << display;

//      tinyxml2::XMLText* text = document->NewText(buffer.str().c_str());
//      element->LinkEndChild(text);
//   }

   return document;
}


/// Serializes the cross entropy error object into a XML document of the TinyXML library without keep the DOM tree in memory.
/// See the OpenNN manual for more information about the format of this document

void MeanSquaredError::write_XML(tinyxml2::XMLPrinter& file_stream) const
{
    // Error type

    file_stream.OpenElement("Error");

    file_stream.PushAttribute("Type", "MEAN_SQUARED_ERROR");

    file_stream.CloseElement();

    // Regularization

    write_regularization_XML(file_stream);
}


type MeanSquaredError::sum_squared_error(const Tensor<type, 2>& outputs, const Tensor<type, 2>& targets) const
{
    const auto error = targets - outputs;

    const Eigen::array<IndexPair<Index>, 2> product_dimensions = { IndexPair<Index>(0, 0), IndexPair<Index>(1, 1) };

    const Tensor<type, 0> sse = error.contract(error, product_dimensions);

    return sse(0);
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
