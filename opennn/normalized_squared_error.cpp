//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   N O R M A L I Z E D   S Q U A R E D   E R R O R   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "normalized_squared_error.h"

namespace OpenNN
{

/// Default constructor.
/// It creates a normalized squared error term object not associated to any
/// neural network and not measured on any data set.
/// It also initializes all the rest of class members to their default values.

NormalizedSquaredError::NormalizedSquaredError() : LossIndex()
{
    set_default();
}


/// Neural network constructor.
/// It creates a normalized squared error term associated to a neural network object but not measured on any data set.
/// It also initializes all the rest of class members to their default values.
/// @param new_neural_network_pointer Pointer to a neural network object.

NormalizedSquaredError::NormalizedSquaredError(NeuralNetwork* new_neural_network_pointer)
    : LossIndex(new_neural_network_pointer)
{
    set_default();
}


/// Data set constructor.
/// It creates a normalized squared error term not associated to any
/// neural network but to be measured on a data set object.
/// It also initializes all the rest of class members to their default values.
/// @param new_data_set_pointer Pointer to a data set object.

NormalizedSquaredError::NormalizedSquaredError(DataSet* new_data_set_pointer)
    : LossIndex(new_data_set_pointer)
{
    set_default();
}


/// Neural network and data set constructor.
/// It creates a normalized squared error term associated to a neural network and measured on a data set.
/// It also initializes all the rest of class members to their default values.
/// @param new_neural_network_pointer Pointer to a neural network object.
/// @param new_data_set_pointer Pointer to a data set object.

NormalizedSquaredError::NormalizedSquaredError(NeuralNetwork* new_neural_network_pointer, DataSet* new_data_set_pointer)
    : LossIndex(new_neural_network_pointer, new_data_set_pointer)
{
    set_default();
}


/// XML constructor.
/// It creates a normalized squared error not associated to any neural network and not measured on any data set.
/// It also sets all the rest of class members from a TinyXML document->
/// @param normalized_squared_error_document XML document with the class members.

NormalizedSquaredError::NormalizedSquaredError(const tinyxml2::XMLDocument& normalized_squared_error_document)
    : LossIndex(normalized_squared_error_document)
{
    set_default();

    from_XML(normalized_squared_error_document);
}


/// Destructor.

NormalizedSquaredError::~NormalizedSquaredError()
{
}


/// Returns the normalization coefficient.

type NormalizedSquaredError::get_normalization_coefficient() const
{
    return normalization_coefficient;
}


/// Returns the selection normalization coefficient.

type NormalizedSquaredError::get_selection_normalization_coefficient() const
{
    return selection_normalization_coefficient;
}


/// Sets the normalization coefficient from training instances.
/// This method calculates the normalization coefficient of the dataset.

void NormalizedSquaredError::set_normalization_coefficient()
{
    // Data set

    const Tensor<type, 1> targets_mean = data_set_pointer->calculate_used_targets_mean();

    //Targets matrix

    const Tensor<type, 2> targets = data_set_pointer->get_target_data();

    //Normalization coefficient

    normalization_coefficient = calculate_normalization_coefficient(targets, targets_mean);
}

/// Sets the normalization coefficient.
/// @param new_normalization_coefficient New normalization coefficient to be set.

void NormalizedSquaredError::set_normalization_coefficient(const type& new_normalization_coefficient)
{
    normalization_coefficient = new_normalization_coefficient;
}


/// Sets the normalization coefficient from selection instances.
/// This method calculates the normalization coefficient of the dataset.

void NormalizedSquaredError::set_selection_normalization_coefficient()
{
    // Data set

    const Tensor<Index, 1> selection_indices = data_set_pointer->get_selection_instances_indices();

    const Index selection_instances_number = selection_indices.size();

    if(selection_instances_number == 0) return;

    const Tensor<type, 1> selection_targets_mean = data_set_pointer->calculate_selection_targets_mean();

    const Tensor<type, 2> targets = data_set_pointer->get_selection_target_data();

    // Normalization coefficient

    selection_normalization_coefficient = calculate_normalization_coefficient(targets, selection_targets_mean);
}


/// Sets the normalization coefficient from selection instances.
/// @param new_normalization_coefficient New normalization coefficient to be set.

void NormalizedSquaredError::set_selection_normalization_coefficient(const type& new_selection_normalization_coefficient)
{
    selection_normalization_coefficient = new_selection_normalization_coefficient;
}


/// Sets the default values.

void NormalizedSquaredError::set_default()
{
    if(has_neural_network() && has_data_set() && data_set_pointer->has_data())
    {
        set_normalization_coefficient();
        set_selection_normalization_coefficient();
    }
    else
    {
        normalization_coefficient = -1;
        selection_normalization_coefficient = -1;
    }
}

/// Returns the normalization coefficient to be used for the loss of the error.
/// This is measured on the training instances of the data set.
/// @param targets Matrix with the targets values from dataset.
/// @param targets_mean Vector with the means of the given targets.

type NormalizedSquaredError::calculate_normalization_coefficient(const Tensor<type, 2>& targets, const Tensor<type, 1>& targets_mean) const
{

#ifdef __OPENNN_DEBUG__

    check();

    const Index means_number = targets_mean.dimension(0);
    const Index targets_number = targets.dimension(1);

    if(targets_number != means_number)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: NormalizedquaredError function.\n"
               << "type calculate_normalization_coefficient(const Tensor<type, 2>& targets, const Tensor<type, 1>& targets_mean) function.\n"
               << " The columns number of targets("<< targets_number <<") must be equal("<< means_number<<").\n";

        throw logic_error(buffer.str());
    }
#endif

    const Index size = targets.dimension(0);

    type normalization_coefficient = 0;

    for(Index i = 0; i < size; i++)
    {
        Tensor<type, 0> norm_1 = (targets.chip(i,0) - targets_mean).square().sum();

        normalization_coefficient += norm_1(0);
    }

    return normalization_coefficient;
}

/// Returns loss vector of the error terms function for the normalized squared error.
/// It uses the error back-propagation method.

Tensor<type, 1> NormalizedSquaredError::calculate_training_error_terms(const Tensor<type, 2>& outputs, const Tensor<type, 2>& targets) const
{
#ifdef __OPENNN_DEBUG__

    check();

#endif

    /*
       return error_rows(outputs, targets);
    */
    return Tensor<type, 1>();
}


/// Returns loss vector of the error terms function for the normalized squared error.
/// It uses the error back-propagation method.
/// @param parameters Neural network parameters.

Tensor<type, 1> NormalizedSquaredError::calculate_training_error_terms(const Tensor<type, 1>& parameters) const
{
#ifdef __OPENNN_DEBUG__

    check();

#endif

    const Tensor<type, 2> inputs = data_set_pointer->get_training_input_data();

    const Tensor<type, 2> targets = data_set_pointer->get_training_target_data();

    const Tensor<type, 2> outputs = neural_network_pointer->calculate_trainable_outputs(inputs, parameters);
    /*
        return error_rows(outputs, targets)/normalization_coefficient;
    */
    return Tensor<type, 1>();

}


/// This method calculates the second order loss.
/// It is used for optimization of parameters during training.
/// Returns a second order terms loss structure, which contains the values and the Hessian of the error terms function.

LossIndex::SecondOrderLoss NormalizedSquaredError::calculate_terms_second_order_loss() const
{
#ifdef __OPENNN_DEBUG__

    check();

#endif

    // Neural network

    const Index layers_number = neural_network_pointer->get_trainable_layers_number();

    const Index parameters_number = neural_network_pointer->get_parameters_number();

    bool is_forecasting = false;

    if(neural_network_pointer->has_long_short_term_memory_layer() || neural_network_pointer->has_recurrent_layer()) is_forecasting = true;

    // Data set


    SecondOrderLoss terms_second_order_loss(parameters_number);
/*
    const Tensor<Index, 2> training_batches = data_set_pointer->get_training_batches(!is_forecasting);

    const Index batches_number = training_batches.size();

    // Eigen stuff

    #pragma omp parallel for

    for(Index i = 0; i < batches_number; i++)
    {
        const Tensor<type, 2> inputs = data_set_pointer->get_input_data(training_batches.chip(i,0));
        const Tensor<type, 2> targets = data_set_pointer->get_target_data(training_batches.chip(i,0));

                const Tensor<Layer::ForwardPropagation, 1> forward_propagation = neural_network_pointer->forward_propagate(inputs);

                const Tensor<type, 1> error_terms = calculate_training_error_terms(forward_propagation[layers_number-1].activations_2d, targets);

                const Tensor<type, 2> output_gradient = (forward_propagation[layers_number-1].activations_2d - targets).divide(error_terms, 0);

                const Tensor<Tensor<type, 2>, 1> layers_delta = calculate_layers_delta(forward_propagation, output_gradient);

                const Tensor<type, 2> error_terms_Jacobian = calculate_error_terms_Jacobian(inputs, forward_propagation, layers_delta);

        //        const Tensor<type, 2> error_terms_Jacobian_transpose = error_terms_Jacobian.calculate_transpose();

        //        const type loss = dot(error_terms, error_terms);

                const Tensor<type, 0> loss = error_terms.contract(error_terms, product_vector_vector);

        //        const Tensor<type, 1> gradient = dot(error_terms_Jacobian_transpose, error_terms);

                const Tensor<type, 1> gradient = error_terms_Jacobian.contract(error_terms, product_matrix_vector);

                Tensor<type, 2> hessian_approximation = error_terms_Jacobian.contract(error_terms_Jacobian, product_matrix_matrix);
                //hessian_approximation.dot(error_terms_Jacobian_transpose, error_terms_Jacobian);

                  #pragma omp critical
                {
                    terms_second_order_loss.loss += loss(0);
                    terms_second_order_loss.gradient += gradient;
                    terms_second_order_loss.hessian += hessian_approximation;
                 }

    }

    terms_second_order_loss.loss /= normalization_coefficient;
    terms_second_order_loss.gradient = (static_cast<type>(2.0)/normalization_coefficient)*terms_second_order_loss.gradient;
    terms_second_order_loss.hessian = (static_cast<type>(2.0)/normalization_coefficient)*terms_second_order_loss.hessian;

    if(regularization_method != RegularizationMethod::NoRegularization)
    {
//        terms_second_order_loss.loss += calculate_regularization();
//        terms_second_order_loss.gradient += calculate_regularization_gradient();
//        terms_second_order_loss.hessian += calculate_regularization_hessian();
    }
*/
    return terms_second_order_loss;
}


/// Returns a string with the name of the normalized squared error loss type, "NORMALIZED_SQUARED_ERROR".

string NormalizedSquaredError::get_error_type() const
{
    return "NORMALIZED_SQUARED_ERROR";
}


/// Returns a string with the name of the normalized squared error loss type in text format.

string NormalizedSquaredError::get_error_type_text() const
{
    return "Normalized squared error";
}


/// Serializes the normalized squared error object into a XML document of the TinyXML library.
/// See the OpenNN manual for more information about the format of this element.

tinyxml2::XMLDocument* NormalizedSquaredError::to_XML() const
{
    ostringstream buffer;

    tinyxml2::XMLDocument* document = new tinyxml2::XMLDocument;

    // Normalized squared error

    tinyxml2::XMLElement* normalized_squared_error_element = document->NewElement("NormalizedSquaredError");

    document->InsertFirstChild(normalized_squared_error_element);

    // Display
//   {
//      tinyxml2::XMLElement* display_element = document->NewElement("Display");
//      normalized_squared_error_element->LinkEndChild(display_element);

//      buffer.str("");
//      buffer << display;

//      tinyxml2::XMLText* display_text = document->NewText(buffer.str().c_str());
//      display_element->LinkEndChild(display_text);
//   }

    return document;
}


/// Serializes the cross entropy error object into a XML document of the TinyXML library without keep the DOM tree in memory.
/// See the OpenNN manual for more information about the format of this document

void NormalizedSquaredError::write_XML(tinyxml2::XMLPrinter& file_stream) const
{
    // Error type

    file_stream.OpenElement("Error");

    file_stream.PushAttribute("Type", "NORMALIZED_SQUARED_ERROR");

    file_stream.CloseElement();

    // Regularization

    write_regularization_XML(file_stream);
}


/// Loads a root mean squared error object from a XML document.
/// @param document Pointer to a TinyXML document with the object data.

void NormalizedSquaredError::from_XML(const tinyxml2::XMLDocument& document)
{
    const tinyxml2::XMLElement* root_element = document.FirstChildElement("NormalizedSquaredError");

    if(!root_element)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: NormalizedSquaredError class.\n"
               << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
               << "Normalized squared element is nullptr.\n";

        throw logic_error(buffer.str());
    }

    // Regularization

    tinyxml2::XMLDocument regularization_document;
    tinyxml2::XMLNode* element_clone;

    const tinyxml2::XMLElement* regularization_element = root_element->FirstChildElement("Regularization");

    element_clone = regularization_element->DeepClone(&regularization_document);

    regularization_document.InsertFirstChild(element_clone);

    regularization_from_XML(regularization_document);
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
