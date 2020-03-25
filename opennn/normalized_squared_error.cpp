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

double NormalizedSquaredError::get_normalization_coefficient() const
{
    return normalization_coefficient;
}


/// Sets the normalization coefficient from training instances.
/// This method calculates the normalization coefficient of the dataset.

void NormalizedSquaredError::set_normalization_coefficient()
{
    // Data set stuff

    const Vector<size_t> training_indices = data_set_pointer->get_training_instances_indices();

    const size_t training_instances_number = training_indices.size();    

    const Vector<size_t> targets_indices = data_set_pointer->get_target_variables_indices();

    const Vector<double> training_targets_mean = data_set_pointer->calculate_training_targets_mean();

    // Normalized squared error stuff

    double new_normalization_coefficient = 0.0;

     #pragma omp parallel for reduction(+ : new_normalization_coefficient)

    for(int i = 0; i < static_cast<int>(training_instances_number); i++)
    {
        const size_t training_index = training_indices[static_cast<size_t>(i)];

       // Target vector

       const Vector<double> targets = data_set_pointer->get_instance_data(training_index, targets_indices);

       // Normalization coefficient

       new_normalization_coefficient += sum_squared_error(targets, training_targets_mean);
    }

    normalization_coefficient = new_normalization_coefficient;
}


/// Sets the normalization coefficient.
/// @param new_normalization_coefficient New normalization coefficient to be set.

void NormalizedSquaredError::set_normalization_coefficient(const double& new_normalization_coefficient)
{
    normalization_coefficient = new_normalization_coefficient;
}


/// Sets the normalization coefficient from selection instances.
/// This method calculates the normalization coefficient of the dataset.

void NormalizedSquaredError::set_selection_normalization_coefficient()
{
    // Data set stuff

//

    const Vector<size_t> selection_indices = data_set_pointer->get_selection_instances_indices();

    const size_t selection_instances_number = selection_indices.size();

    if(selection_instances_number == 0) return;

    

    const Vector<size_t> targets_indices = data_set_pointer->get_target_variables_indices();

    const Vector<double> selection_targets_mean = data_set_pointer->calculate_selection_targets_mean();

    // Normalized squared error stuff

    double new_selection_normalization_coefficient = 0.0;

     #pragma omp parallel for reduction(+ : new_selection_normalization_coefficient)

    for(int i = 0; i < static_cast<int>(selection_instances_number); i++)
    {
        const size_t selection_index = selection_indices[static_cast<size_t>(i)];

       // Target vector

       const Vector<double> targets = data_set_pointer->get_instance_data(selection_index, targets_indices);

       // Normalization coefficient

       new_selection_normalization_coefficient += sum_squared_error(targets, selection_targets_mean);
    }

    selection_normalization_coefficient = new_selection_normalization_coefficient;
}


/// Sets the normalization coefficient from selection instances.
/// @param new_normalization_coefficient New normalization coefficient to be set.

void NormalizedSquaredError::set_selection_normalization_coefficient(const double& new_selection_normalization_coefficient)
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

double NormalizedSquaredError::calculate_normalization_coefficient(const Matrix<double>& targets, const Vector<double>& targets_mean) const
{
#ifdef __OPENNN_DEBUG__

check();

#endif

   return sum_squared_error(targets, targets_mean);
}


/// This method separates training instances and calculates batches from the dataset.
/// It also calculates the outputs and the sum squared error from the targets and outputs.
/// Returns a sum squared error of the training instances.

double NormalizedSquaredError::calculate_training_error() const
{
#ifdef __OPENNN_DEBUG__

check();

#endif

    //Neural network

     bool is_forecasting = false;

    if(neural_network_pointer->has_long_short_term_memory_layer() || neural_network_pointer->has_recurrent_layer()) is_forecasting = true;

    // Data set

    const Vector<Vector<size_t>> training_batches = data_set_pointer->get_training_batches(!is_forecasting);

    const size_t batches_number = training_batches.size();

    const size_t batch_instances_number = data_set_pointer->get_batch_instances_number();

    const size_t inputs_number = data_set_pointer->get_input_variables_number();
    const size_t targets_number = data_set_pointer->get_target_variables_number();

    Tensor<double> inputs(batch_instances_number, inputs_number);
    Tensor<double> targets(batch_instances_number, targets_number);
    Tensor<double> outputs(batch_instances_number, targets_number);

    double training_error = 0.0;

     #pragma omp parallel for reduction(+ : training_error)

    for(int i = 0; i < static_cast<int>(batches_number); i++)
    {
        inputs = data_set_pointer->get_input_data(training_batches[static_cast<size_t>(i)]);
        targets = data_set_pointer->get_target_data(training_batches[static_cast<size_t>(i)]);

        outputs = neural_network_pointer->calculate_trainable_outputs(inputs);

        const double batch_error = sum_squared_error(outputs, targets);

        training_error += batch_error;
    }

    return training_error / normalization_coefficient;
}


double NormalizedSquaredError::calculate_training_error(const Vector<double>& parameters) const
{
#ifdef __OPENNN_DEBUG__

check();

#endif

    //Neural network

     bool is_forecasting = false;

    if(neural_network_pointer->has_long_short_term_memory_layer() || neural_network_pointer->has_recurrent_layer()) is_forecasting = true;

    // Data set

    const Vector<Vector<size_t>> training_batches = data_set_pointer->get_training_batches(!is_forecasting);

    const size_t batches_number = training_batches.size();

    const size_t batch_instances_number = data_set_pointer->get_batch_instances_number();

    const size_t inputs_number = data_set_pointer->get_input_variables_number();
    const size_t targets_number = data_set_pointer->get_target_variables_number();

    Tensor<double> inputs(batch_instances_number, inputs_number);
    Tensor<double> targets(batch_instances_number, targets_number);
    Tensor<double> outputs(batch_instances_number, targets_number);

    double training_error = 0.0;

     #pragma omp parallel for reduction(+ : training_error)

    for(int i = 0; i < static_cast<int>(batches_number); i++)
    {
        inputs = data_set_pointer->get_input_data(training_batches[static_cast<size_t>(i)]);
        targets = data_set_pointer->get_target_data(training_batches[static_cast<size_t>(i)]);

        outputs = neural_network_pointer->calculate_trainable_outputs(inputs, parameters);

        const double batch_error = sum_squared_error(outputs, targets);

        training_error += batch_error;
    }

    return training_error / normalization_coefficient;
}


/// This method separates selection instances and calculates batches from the dataset.
/// It also calculates the outputs and the sum squared error from the targets and outputs.
/// Returns a sum squared error of the training instances.

double NormalizedSquaredError::calculate_selection_error() const
{
#ifdef __OPENNN_DEBUG__

check();

#endif

    //Neural network

     bool is_forecasting = false;

    if(neural_network_pointer->has_long_short_term_memory_layer() || neural_network_pointer->has_recurrent_layer()) is_forecasting = true;

    // Data set

    const Vector<Vector<size_t>> selection_batches = data_set_pointer->get_selection_batches(!is_forecasting);

    const size_t batches_number = selection_batches.size();

    double selection_error = 0.0;

    const size_t batch_instances_number = data_set_pointer->get_batch_instances_number();

    const size_t inputs_number = data_set_pointer->get_input_variables_number();
    const size_t targets_number = data_set_pointer->get_target_variables_number();

    Tensor<double> inputs(batch_instances_number, inputs_number);
    Tensor<double> targets(batch_instances_number, targets_number);
    Tensor<double> outputs(batch_instances_number, targets_number);

     #pragma omp parallel for reduction(+ : selection_error)

    for(int i = 0; i < static_cast<int>(batches_number); i++)
    {
        inputs = data_set_pointer->get_input_data(selection_batches[static_cast<size_t>(i)]);
        targets = data_set_pointer->get_target_data(selection_batches[static_cast<size_t>(i)]);

        outputs = neural_network_pointer->calculate_trainable_outputs(inputs);

        const double batch_error = sum_squared_error(outputs, targets);

        selection_error += batch_error;
    }

    return selection_error / selection_normalization_coefficient;
}


/// This method calculates the mean squared error of the given batch.
/// Returns the mean squared error of this batch.
/// @param batch_indices Indices of the batch instances corresponding to the dataset.

double NormalizedSquaredError::calculate_batch_error(const Vector<size_t>& batch_indices) const
{
#ifdef __OPENNN_DEBUG__

check();

#endif

    // Data set

    const Tensor<double> inputs = data_set_pointer->get_input_data(batch_indices);

    const Tensor<double> targets = data_set_pointer->get_target_data(batch_indices);

    const Tensor<double> outputs = neural_network_pointer->calculate_trainable_outputs(inputs);

    const double batch_error = sum_squared_error(outputs, targets);

    return batch_error;
}


double NormalizedSquaredError::calculate_batch_error(const Vector<size_t>& batch_indices, const Vector<double>& parameters) const
{
#ifdef __OPENNN_DEBUG__

check();

#endif

    // Data set

    const Tensor<double> inputs = data_set_pointer->get_input_data(batch_indices);
    const Tensor<double> targets = data_set_pointer->get_target_data(batch_indices);

    const Tensor<double> outputs = neural_network_pointer->calculate_trainable_outputs(inputs, parameters);

    const double batch_error = sum_squared_error(outputs, targets);

    return batch_error;
}


/// This method calculates the gradient of the output error function, necessary for backpropagation.
/// Returns the gradient value.
/// @param outputs Tensor with the values of the outputs from the neural network.
/// @param targets Tensor with the values of the targets from the dataset.

Tensor<double> NormalizedSquaredError::calculate_output_gradient(const Tensor<double>& outputs, const Tensor<double>& targets) const
{
#ifdef __OPENNN_DEBUG__

check();

#endif

     return (outputs-targets)*2.0 / normalization_coefficient;
}


/// This method calculates the first order loss.
/// It is used for optimization of parameters during training.
/// Returns a first order terms loss structure, which contains the values and the Jacobian of the error terms function.

LossIndex::FirstOrderLoss NormalizedSquaredError::calculate_first_order_loss() const
{
#ifdef __OPENNN_DEBUG__

check();

#endif

    // Neural network

    const size_t layers_number = neural_network_pointer->get_trainable_layers_number();

    const size_t parameters_number = neural_network_pointer->get_parameters_number();

     bool is_forecasting = false;

    if(neural_network_pointer->has_long_short_term_memory_layer() || neural_network_pointer->has_recurrent_layer()) is_forecasting = true;

    // Data set

    const Vector<Vector<size_t>> training_batches = data_set_pointer->get_training_batches(!is_forecasting);

    const size_t batches_number = training_batches.size();

    FirstOrderLoss first_order_loss(parameters_number);

     #pragma omp parallel for

    for(int i = 0; i < static_cast<int>(batches_number); i++)
    {
        const Tensor<double> inputs = data_set_pointer->get_input_data(training_batches[static_cast<unsigned>(i)]);

        const Tensor<double> targets = data_set_pointer->get_target_data(training_batches[static_cast<unsigned>(i)]);

        const Vector<Layer::FirstOrderActivations> forward_propagation
                = neural_network_pointer->calculate_trainable_forward_propagation(inputs);

        const Vector<double> error_terms
                = calculate_training_error_terms(forward_propagation[layers_number-1].activations, targets);

        const Tensor<double> output_gradient
                = (forward_propagation[layers_number-1].activations - targets).divide(error_terms, 0);

        const Vector<Tensor<double>> layers_delta = calculate_layers_delta(forward_propagation, output_gradient);

        const Matrix<double> error_terms_Jacobian
                = calculate_error_terms_Jacobian(inputs, forward_propagation, layers_delta);

        const Matrix<double> error_terms_Jacobian_transpose = error_terms_Jacobian.calculate_transpose();

        const double loss = dot(error_terms, error_terms);

        const Vector<double> gradient = dot(error_terms_Jacobian_transpose, error_terms);

          #pragma omp critical
        {
            first_order_loss.loss += loss;
            first_order_loss.gradient += gradient;
         }
    }

    first_order_loss.loss /= normalization_coefficient;
    first_order_loss.gradient *= (2.0/normalization_coefficient);

    // Regularization

    if(regularization_method != RegularizationMethod::NoRegularization)
    {
        first_order_loss.loss += regularization_weight*calculate_regularization();
        first_order_loss.gradient += calculate_regularization_gradient()*regularization_weight;
    }

    return first_order_loss;
}


/// This method calculates the first order loss for the selected batch.
/// Returns a first order terms loss structure, which contains the values and the Jacobian of the error terms function.
/// @param batch_indices Indices of the batch instances corresponding to the dataset.

LossIndex::FirstOrderLoss NormalizedSquaredError::calculate_batch_first_order_loss(const Vector<size_t>& batch_indices) const
{
#ifdef __OPENNN_DEBUG__

check();

#endif

    // Neural network

    const size_t layers_number = neural_network_pointer->get_trainable_layers_number();

    const size_t parameters_number = neural_network_pointer->get_parameters_number();

    FirstOrderLoss first_order_loss(parameters_number);

    const Tensor<double> inputs = data_set_pointer->get_input_data(batch_indices);

    const Tensor<double> targets = data_set_pointer->get_target_data(batch_indices);

    const Vector<Layer::FirstOrderActivations> forward_propagation = neural_network_pointer->calculate_trainable_forward_propagation(inputs);

    const Tensor<double> output_gradient = calculate_output_gradient(forward_propagation[layers_number-1].activations, targets);

    const Vector<Tensor<double>> layers_delta = calculate_layers_delta(forward_propagation, output_gradient);

    const Vector<double> batch_gradient = calculate_error_gradient(inputs, forward_propagation, layers_delta);

    const double batch_error = sum_squared_error(forward_propagation[layers_number-1].activations, targets);

    first_order_loss.loss = batch_error / normalization_coefficient;
    first_order_loss.gradient += batch_gradient;

    // Regularization

    if(regularization_method != RegularizationMethod::NoRegularization)
    {
        first_order_loss.loss += regularization_weight*calculate_regularization();
        first_order_loss.gradient += calculate_regularization_gradient()*regularization_weight;
    }

    return first_order_loss;
}


/// Returns loss vector of the error terms function for the normalized squared error.
/// It uses the error back-propagation method.

Vector<double> NormalizedSquaredError::calculate_training_error_terms(const Tensor<double>& outputs, const Tensor<double>& targets) const
{
#ifdef __OPENNN_DEBUG__

check();

#endif

   // Control sentence

   #ifdef __OPENNN_DEBUG__

   check();

   #endif

   return error_rows(outputs, targets);
}


/// Returns loss vector of the error terms function for the normalized squared error.
/// It uses the error back-propagation method.
/// @param parameters Neural network parameters.

Vector<double> NormalizedSquaredError::calculate_training_error_terms(const Vector<double>& parameters) const
{
#ifdef __OPENNN_DEBUG__

check();

#endif

    const Tensor<double> inputs = data_set_pointer->get_training_input_data();

    const Tensor<double> targets = data_set_pointer->get_training_target_data();

    const Tensor<double> outputs = neural_network_pointer->calculate_trainable_outputs(inputs, parameters);

    return error_rows(outputs, targets)/normalization_coefficient;
}


/// Returns the squared errors of the training instances. 

Vector<double> NormalizedSquaredError::calculate_squared_errors() const
{
   #ifdef __OPENNN_DEBUG__ 

   check();

   #endif

   // Data set stuff

   const Vector<size_t> training_indices = data_set_pointer->get_training_instances_indices();

   const size_t training_instances_number = training_indices.size();

   // Calculate

   Vector<double> squared_errors(training_instances_number);

   // Main loop

    #pragma omp parallel for

   for(int i = 0; i < static_cast<int>(training_instances_number); i++)
   {
      const size_t training_index = training_indices[static_cast<size_t>(i)];

      const Tensor<double> inputs = data_set_pointer->get_instance_input_data(training_index);

      const Tensor<double> outputs = neural_network_pointer->calculate_trainable_outputs(inputs);

      const Tensor<double> targets = data_set_pointer->get_instance_target_data(training_index);

      squared_errors[static_cast<size_t>(i)] = sum_squared_error(outputs, targets);
   }

   return squared_errors;
}


/// Returns a vector with the indices of the instances which have the maximum error.
/// @param maximal_errors_number Number of instances required.

Vector<size_t> NormalizedSquaredError::calculate_maximal_errors(const size_t& maximal_errors_number) const
{
    #ifdef __OPENNN_DEBUG__

    check();

    const size_t training_instances_number = data_set_pointer->get_training_instances_number();

    if(maximal_errors_number > training_instances_number)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: NormalizedquaredError class.\n"
               << "Vector<size_t> calculate_maximal_errors() const method.\n"
               << "Number of maximal errors(" << maximal_errors_number << ") must be equal or less than number of training instances(" << training_instances_number << ").\n";

       throw logic_error(buffer.str());
    }

    #endif

    return maximal_indices(calculate_squared_errors(), maximal_errors_number);
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

    const size_t layers_number = neural_network_pointer->get_trainable_layers_number();

    const size_t parameters_number = neural_network_pointer->get_parameters_number();

     bool is_forecasting = false;

    if(neural_network_pointer->has_long_short_term_memory_layer() || neural_network_pointer->has_recurrent_layer()) is_forecasting = true;

    // Data set

    const Vector<Vector<size_t>> training_batches = data_set_pointer->get_training_batches(!is_forecasting);

    const size_t batches_number = training_batches.size();

    SecondOrderLoss terms_second_order_loss(parameters_number);

     #pragma omp parallel for

    for(int i = 0; i < static_cast<int>(batches_number); i++)
    {
        const Tensor<double> inputs = data_set_pointer->get_input_data(training_batches[static_cast<unsigned>(i)]);
        const Tensor<double> targets = data_set_pointer->get_target_data(training_batches[static_cast<unsigned>(i)]);

        const Vector<Layer::FirstOrderActivations> forward_propagation = neural_network_pointer->calculate_trainable_forward_propagation(inputs);

        const Vector<double> error_terms = calculate_training_error_terms(forward_propagation[layers_number-1].activations, targets);

        const Tensor<double> output_gradient = (forward_propagation[layers_number-1].activations - targets).divide(error_terms, 0);

        const Vector<Tensor<double>> layers_delta = calculate_layers_delta(forward_propagation, output_gradient);

        const Matrix<double> error_terms_Jacobian = calculate_error_terms_Jacobian(inputs, forward_propagation, layers_delta);

        const Matrix<double> error_terms_Jacobian_transpose = error_terms_Jacobian.calculate_transpose();

        const double loss = dot(error_terms, error_terms);

        const Vector<double> gradient = dot(error_terms_Jacobian_transpose, error_terms);

        Matrix<double> hessian_approximation;
        //hessian_approximation.dot(error_terms_Jacobian_transpose, error_terms_Jacobian);

          #pragma omp critical
        {
            terms_second_order_loss.loss += loss;
            terms_second_order_loss.gradient += gradient;
            terms_second_order_loss.hessian += hessian_approximation;
         }
    }

    terms_second_order_loss.loss /= normalization_coefficient;
    terms_second_order_loss.gradient *= (2.0/normalization_coefficient);
    terms_second_order_loss.hessian *= (2.0/normalization_coefficient);

    if(regularization_method == RegularizationMethod::NoRegularization)
    {
        terms_second_order_loss.loss += calculate_regularization();
        terms_second_order_loss.gradient += calculate_regularization_gradient();
        terms_second_order_loss.hessian += calculate_regularization_hessian();
    }

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
// Copyright(C) 2005-2019 Artificial Intelligence Techniques, SL.
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
