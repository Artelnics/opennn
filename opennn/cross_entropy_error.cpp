/****************************************************************************************************************/
/*                                                                                                              */
/*   OpenNN: Open Neural Networks Library                                                                       */
/*   www.opennn.net                                                                                             */
/*                                                                                                              */
/*   C R O S S   E N T R O P Y   E R R O R   C L A S S                                                          */
/*                                                                                                              */
/*   Artificial Intelligence Techniques SL                                                                      */
/*   artelnics@artelnics.com                                                                                    */
/*                                                                                                              */
/****************************************************************************************************************/

// OpenNN includes

#include "cross_entropy_error.h"

#ifdef __OPENNN_CUDA__
#include <cuda_runtime.h>
#include <cublas_v2.h>

void freeCUDA(double* A_d);

void calculateFirstOrderLossCUDA(const std::vector<double*> weights_d, const std::vector<size_t> weights_rows_numbers, const std::vector<size_t> weights_columns_numbers,
                                const std::vector<double*> biases_d, const std::vector<size_t> bias_rows_numbers,
                                const double* input_data_h, const size_t input_rows, const size_t input_columns,
                                const double* target_data_h, const size_t target_rows, const size_t target_columns,
                                std::vector<double*> error_gradient_data,
                                double* output_data_h, const size_t output_rows, const size_t output_columns,
                                const std::vector<std::string> layers_activations, const std::string loss_method,
                                const std::vector<double> loss_parameters = vector<double>());

void calculateOutputsCUDA(const std::vector<double*> weights_d, const std::vector<size_t> weights_rows_numbers, const std::vector<size_t> weights_columns_numbers,
                          const std::vector<double*> biases_d, const std::vector<size_t> bias_rows_numbers,
                          const double* input_data_h, const size_t input_rows, const size_t input_columns,
                          double* output_data_h, const size_t output_rows, const size_t output_columns,
                          const std::vector<std::string> layers_activations);

#endif

namespace OpenNN
{

// DEFAULT CONSTRUCTOR

/// Default constructor. 
/// It creates a default cross entropy error term object, 
/// which is not associated to any neural network and not measured on any data set.
/// It also initializes all the rest of class members to their default values.

CrossEntropyError::CrossEntropyError() : LossIndex()
{
}


// NEURAL NETWORK CONSTRUCTOR

/// Neural network constructor. 
/// It creates a cross entropy error term associated to a neural network but not measured on any data set.
/// It also initializes all the rest of class members to their default values.
/// @param new_neural_network_pointer Pointer to a neural network object.

CrossEntropyError::CrossEntropyError(NeuralNetwork* new_neural_network_pointer)
 : LossIndex(new_neural_network_pointer)
{
}


// DATA SET CONSTRUCTOR

/// Data set constructor. 
/// It creates a cross entropy error not associated to any neural network but to be measured on a data set object.
/// It also initializes all the rest of class members to their default values.
/// @param new_data_set_pointer Pointer to a data set object.

CrossEntropyError::CrossEntropyError(DataSet* new_data_set_pointer) 
: LossIndex(new_data_set_pointer)
{
}


// NEURAL NETWORK AND DATA SET CONSTRUCTOR

/// Neural network and data set constructor. 
/// It creates a cross entropy error term object associated to a neural network and measured on a data set.
/// It also initializes all the rest of class members to their default values:
/// @param new_neural_network_pointer: Pointer to a neural network object.
/// @param new_data_set_pointer: Pointer to a data set object.

CrossEntropyError::CrossEntropyError(NeuralNetwork* new_neural_network_pointer, DataSet* new_data_set_pointer)
: LossIndex(new_neural_network_pointer, new_data_set_pointer)
{
}


// XML CONSTRUCTOR

/// XML constructor. 
/// It creates a cross entropy error not associated to any neural network and not measured on any data set.
/// It also sets all the rest of class members from a TinyXML document->
/// @param sum_squared_error_document XML document with the class members. 

CrossEntropyError::CrossEntropyError(const tinyxml2::XMLDocument& sum_squared_error_document)
 : LossIndex(sum_squared_error_document)
{
    from_XML(sum_squared_error_document);
}


// COPY CONSTRUCTOR

/// Copy constructor. 
/// It creates a cross entropy error not associated to any neural network and not measured on any data set.
/// It also sets all the rest of class members from another cross-entropy error object.
/// @param new_cross_entropy_error Object to be copied. 

CrossEntropyError::CrossEntropyError(const CrossEntropyError& new_cross_entropy_error)
 : LossIndex(new_cross_entropy_error)
{

}


// DESTRUCTOR

/// Destructor.

CrossEntropyError::~CrossEntropyError() 
{
}


// ASSIGNMENT OPERATOR

/// Assignment operator. 
/// @param other_cross_entropy_error Object to be copied. 

CrossEntropyError& CrossEntropyError::operator = (const CrossEntropyError& other_cross_entropy_error)
{
   if(this != &other_cross_entropy_error) 
   {
      *neural_network_pointer = *other_cross_entropy_error.neural_network_pointer;
      *data_set_pointer = *other_cross_entropy_error.data_set_pointer;
      display = other_cross_entropy_error.display;
   }

   return(*this);

}

// EQUAL TO OPERATOR

/// Equal to operator. 
/// If compares this object with another object of the same class, and returns true if they are equal, and false otherwise. 
/// @param other_cross_entropy_error Object to be compared with. 

bool CrossEntropyError::operator == (const CrossEntropyError& other_cross_entropy_error) const
{
   if(*neural_network_pointer == *other_cross_entropy_error.neural_network_pointer
   && display == other_cross_entropy_error.display)    
   {
      return(true);
   }
   else
   {
      return(false);  
   }

}


// METHODS


double CrossEntropyError::calculate_training_error() const
{
#ifdef __OPENNN_DEBUG__

check();

#endif

    // Multilayer perceptron

    const MultilayerPerceptron* multilayer_perceptron_pointer = neural_network_pointer->get_multilayer_perceptron_pointer();

    // Data set

    const Vector< Vector<size_t> > training_batches = data_set_pointer->get_instances_pointer()->get_training_batches(batch_size);

    const size_t batches_number = training_batches.size();

    double training_error = 0.0;

    #pragma omp parallel for reduction(+ : training_error)

    for(int i = 0; i < static_cast<int>(batches_number); i++)
    {
        const Matrix<double> inputs = data_set_pointer->get_inputs(training_batches[static_cast<unsigned>(i)]);
        const Matrix<double> targets = data_set_pointer->get_targets(training_batches[static_cast<unsigned>(i)]);

        Matrix<double> outputs = multilayer_perceptron_pointer->calculate_outputs(inputs);

        const double batch_error = outputs.calculate_cross_entropy_error(targets);

        training_error += batch_error;
    }

    return training_error;
}


double CrossEntropyError::calculate_selection_error() const
{
#ifdef __OPENNN_DEBUG__

check();

#endif

    // Multilayer perceptron

    const MultilayerPerceptron* multilayer_perceptron_pointer = neural_network_pointer->get_multilayer_perceptron_pointer();

    // Data set

    const Vector< Vector<size_t> > selection_batches = data_set_pointer->get_instances_pointer()->get_selection_batches(batch_size);

    const size_t batches_number = selection_batches.size();

    double selection_error = 0.0;

    #pragma omp parallel for reduction(+ : selection_error)

    for(int i = 0; i < static_cast<int>(batches_number); i++)
    {
        const Matrix<double> inputs = data_set_pointer->get_inputs(selection_batches[static_cast<unsigned>(i)]);
        const Matrix<double> targets = data_set_pointer->get_targets(selection_batches[static_cast<unsigned>(i)]);

        Matrix<double> outputs = multilayer_perceptron_pointer->calculate_outputs(inputs);

        const double batch_error = outputs.calculate_cross_entropy_error(targets);

        selection_error += batch_error;
    }

    return selection_error;
}


double CrossEntropyError::calculate_training_error(const Vector<double>& parameters) const
{
#ifdef __OPENNN_DEBUG__

check();

#endif

    // Multilayer perceptron

    const MultilayerPerceptron* multilayer_perceptron_pointer = neural_network_pointer->get_multilayer_perceptron_pointer();

    // Data set

    const Vector< Vector<size_t> > training_batches = data_set_pointer->get_instances_pointer()->get_training_batches(batch_size);

    const size_t batches_number = training_batches.size();

    double training_error = 0.0;

    #pragma omp parallel for reduction(+ : training_error)

    for(int i = 0; i < static_cast<int>(batches_number); i++)
    {
        const Matrix<double> inputs = data_set_pointer->get_inputs(training_batches[static_cast<unsigned>(i)]);
        const Matrix<double> targets = data_set_pointer->get_targets(training_batches[static_cast<unsigned>(i)]);

        Matrix<double> outputs = multilayer_perceptron_pointer->calculate_outputs(inputs, parameters);

        const double batch_error = outputs.calculate_cross_entropy_error(targets);

        training_error += batch_error;
    }

    return training_error;
}


double CrossEntropyError::calculate_batch_error(const Vector<size_t>& batch_indices) const
{
#ifdef __OPENNN_DEBUG__

check();

#endif

    // Multilayer perceptron

    const MultilayerPerceptron* multilayer_perceptron_pointer = neural_network_pointer->get_multilayer_perceptron_pointer();

    // Data set

    const Matrix<double> inputs = data_set_pointer->get_inputs(batch_indices);
    const Matrix<double> targets = data_set_pointer->get_targets(batch_indices);

    Matrix<double> outputs = multilayer_perceptron_pointer->calculate_outputs(inputs);

    const double batch_error = outputs.calculate_cross_entropy_error(targets);

    return batch_error;
}

double CrossEntropyError::calculate_batch_error_cuda(const Vector<size_t>& batch_indices, const MultilayerPerceptron::Pointers& pointers) const
{
    double batch_error = 0.0;

#ifdef __OPENNN_CUDA__

    const size_t layers_number = pointers.architecture.size() - 1;

    const Matrix<double> inputs_matrix = data_set_pointer->get_inputs(batch_indices);
    const double* input_data = inputs_matrix.data();
    const size_t input_rows = inputs_matrix.get_rows_number();
    const size_t input_columns = inputs_matrix.get_columns_number();

    Matrix<double> outputs(inputs_matrix.get_rows_number(), pointers.architecture[layers_number]);
    double* output_data = outputs.data();
    const size_t output_rows = inputs_matrix.get_rows_number();
    const size_t output_columns = pointers.architecture[layers_number];

    vector<size_t> weights_rows_numbers(layers_number);
    vector<size_t> weights_columns_numbers(layers_number);

    vector<size_t> bias_rows_numbers(layers_number);

    for(size_t i = 0; i < layers_number; i++)
    {
        weights_rows_numbers[i] = pointers.architecture[i];
        weights_columns_numbers[i] = pointers.architecture[i+1];

        bias_rows_numbers[i] = pointers.architecture[i+1];
    }

    calculateOutputsCUDA(pointers.weights_pointers.to_std_vector(), weights_rows_numbers, weights_columns_numbers,
                         pointers.biases_pointers.to_std_vector(), bias_rows_numbers,
                         input_data, input_rows, input_columns,
                         output_data, output_rows, output_columns,
                         pointers.layer_activations.to_std_vector());

    const Matrix<double> targets_matrix = data_set_pointer->get_targets(batch_indices);

    batch_error = outputs.calculate_cross_entropy_error(targets_matrix);

#endif

    return batch_error;
}

/// Returns the cross entropy error function gradient of a multilayer perceptron on a data set.
/// It uses the error back-propagation method.

Vector<double> CrossEntropyError::calculate_training_error_gradient() const
{
#ifdef __OPENNN_DEBUG__

check();

#endif

    // Neural network

    const MultilayerPerceptron* multilayer_perceptron_pointer = neural_network_pointer->get_multilayer_perceptron_pointer();

    const size_t layers_number = multilayer_perceptron_pointer->get_layers_number();

    const size_t parameters_number = multilayer_perceptron_pointer->get_parameters_number();

    // Data set

    const size_t training_instances_number = data_set_pointer->get_instances().get_training_instances_number();

    const Vector< Vector<size_t> > training_batches = data_set_pointer->get_instances_pointer()->get_training_batches(batch_size);

    const size_t batches_number = training_batches.size();

    // Loss index

    Vector<double> training_error_gradient(parameters_number, 0.0);

    #pragma omp parallel for

    for(int i = 0; i < static_cast<int>(batches_number); i++)
    {
        const Matrix<double> inputs = data_set_pointer->get_inputs(training_batches[static_cast<unsigned>(i)]);
        const Matrix<double> targets = data_set_pointer->get_targets(training_batches[static_cast<unsigned>(i)]);

        const MultilayerPerceptron::FirstOrderForwardPropagation first_order_forward_propagation
                = multilayer_perceptron_pointer->calculate_first_order_forward_propagation(inputs);

        const Matrix<double> output_gradient
                = calculate_output_gradient(first_order_forward_propagation.layers_activations[layers_number-1], targets);

        const Vector< Matrix<double> > layers_delta
                = calculate_layers_delta(first_order_forward_propagation.layers_activation_derivatives, output_gradient);

        const Vector<double> batch_gradient
                = calculate_error_gradient(inputs, first_order_forward_propagation.layers_activations, layers_delta);

        #pragma omp critical

        training_error_gradient += batch_gradient;
    }

    return training_error_gradient / static_cast<double>(training_instances_number);
}

LossIndex::FirstOrderLoss CrossEntropyError::calculate_batch_first_order_loss_cuda(const Vector<size_t>& batch_indices,
                                                                                   const MultilayerPerceptron::Pointers& pointers, const Vector<double*>& data_device) const
{
    FirstOrderLoss first_order_loss;

#ifdef __OPENNN_CUDA__

    const size_t instances_number = batch_indices.size();
    const size_t layers_number = pointers.architecture.size() - 1;

    const size_t inputs_number = data_set_pointer->get_variables().get_inputs_number();
    const size_t targets_number = data_set_pointer->get_variables().get_targets_number();

    const Matrix<double> targets_matrix = data_set_pointer->get_targets(batch_indices);

    Matrix<double> outputs(instances_number, pointers.architecture[layers_number]);
    double* output_data = outputs.data();
    const size_t output_rows = instances_number;
    const size_t output_columns = pointers.architecture[layers_number];

    vector<size_t> weights_rows_numbers(layers_number);
    vector<size_t> weights_columns_numbers(layers_number);

    vector<size_t> bias_rows_numbers(layers_number);

    size_t parameters_number = 0;

    for(size_t i = 0; i < layers_number; i++)
    {
        weights_rows_numbers[i] = pointers.architecture[i];
        weights_columns_numbers[i] = pointers.architecture[i+1];

        bias_rows_numbers[i] = pointers.architecture[i+1];

        parameters_number += pointers.architecture[i]*pointers.architecture[i+1] + pointers.architecture[i+1];
    }

    first_order_loss.gradient.set(parameters_number);
    vector<double*> error_gradient_data(2*layers_number);

    size_t index = 0;

    for(size_t i = 0; i < layers_number; i++)
    {
        error_gradient_data[2*i] = first_order_loss.gradient.data() + index;
        index += weights_rows_numbers[i]*weights_columns_numbers[i];

        error_gradient_data[2*i+1] = first_order_loss.gradient.data() + index;
        index += bias_rows_numbers[i];
    }

    vector<double> loss_parameters;

    string loss_method = write_error_term_type();

    calculateFirstOrderLossCUDA(pointers.weights_pointers.to_std_vector(), weights_rows_numbers, weights_columns_numbers,
                               pointers.biases_pointers.to_std_vector(), bias_rows_numbers,
                               data_device[0], instances_number, inputs_number,
                               data_device[1], instances_number, targets_number,
                               error_gradient_data,
                               output_data, output_rows, output_columns,
                               pointers.layer_activations.to_std_vector(), loss_method, loss_parameters);

    const double batch_error = outputs.calculate_cross_entropy_error(targets_matrix);

    first_order_loss.loss = batch_error;

    // Regularization

    if(regularization_method != RegularizationMethod::None)
    {
        first_order_loss.loss += calculate_regularization();
        first_order_loss.gradient += calculate_regularization_gradient();
    }

#endif

    return first_order_loss;
}

/// Returns the cross-entropy error function output gradient of a multilayer perceptron on a data set.
/// It uses the error back-propagation method.
/// @param outputs Matrix of outputs of the neural network.
/// @param targets Matrix of targets of the data set.

Matrix<double> CrossEntropyError::calculate_output_gradient(const Matrix<double>& outputs, const Matrix<double>& targets) const
{
#ifdef __OPENNN_DEBUG__

check();

#endif

    return (targets/outputs)*(-1.0) + (targets*(-1.0) + 1.0)/(outputs*(-1.0) + 1.0);
}


/// Returns a string with the name of the cross entropy error loss type, "CROSS_ENTROPY_ERROR".

string CrossEntropyError::write_error_term_type() const
{
   return("CROSS_ENTROPY_ERROR");
}


/// Serializes the cross entropy error object into a XML document of the TinyXML library. 
/// See the OpenNN manual for more information about the format of this document-> 

tinyxml2::XMLDocument* CrossEntropyError::to_XML() const
{
   ostringstream buffer;

   tinyxml2::XMLDocument* document = new tinyxml2::XMLDocument;

   // Cross entropy error 

   tinyxml2::XMLElement* cross_entropy_error_element = document->NewElement("CrossEntropyError");

   document->InsertFirstChild(cross_entropy_error_element);

   // Display

//   {
//      tinyxml2::XMLElement* display_element = document->NewElement("Display");
//      cross_entropy_error_element->LinkEndChild(display_element);

//      buffer.str("");
//      buffer << display;

//      tinyxml2::XMLText* display_text = document->NewText(buffer.str().c_str());
//      display_element->LinkEndChild(display_text);
//   }

   return(document);
}


/// Serializes the cross entropy error object into a XML document of the TinyXML library without keep the DOM tree in memory.
/// See the OpenNN manual for more information about the format of this document

void CrossEntropyError::write_XML(tinyxml2::XMLPrinter& file_stream) const
{
    // Error type

    file_stream.OpenElement("Error");

    file_stream.PushAttribute("Type", "CROSS_ENTROPY_ERROR");

    file_stream.CloseElement();

    // Regularization

    write_regularization_XML(file_stream);
}


/// Deserializes a TinyXML document into this cross entropy object.
/// @param document TinyXML document containing the member data.

void CrossEntropyError::from_XML(const tinyxml2::XMLDocument& document)
{
    const tinyxml2::XMLElement* root_element = document.FirstChildElement("CrossEntropyError");

    if(!root_element)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: NormalizedSquaredError class.\n"
               << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
               << "Cross entropy element is nullptr.\n";

        throw logic_error(buffer.str());
    }

    // Regularization

    tinyxml2::XMLDocument regularization_document;
    tinyxml2::XMLNode* element_clone;

    const tinyxml2::XMLElement* regularization_element = root_element->FirstChildElement("Regularization");

    element_clone = regularization_element->DeepClone(&regularization_document);

    regularization_document.InsertFirstChild(element_clone);

    regularization_from_XML(regularization_document);

//    const tinyxml2::XMLElement* root_element = document.FirstChildElement("CrossEntropyError");

//    if(!root_element)
//    {
//        ostringstream buffer;

//        buffer << "OpenNN Exception: CrossEntropyError class.\n"
//               << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
//               << "Cross entropy error element is nullptr.\n";

//        throw logic_error(buffer.str());
//    }

//  // Display
//  {
//     const tinyxml2::XMLElement* display_element = root_element->FirstChildElement("Display");

//     if(display_element)
//     {
//        const string new_display_string = display_element->GetText();

//        try
//        {
//           set_display(new_display_string != "0");
//        }
//        catch(const logic_error& e)
//        {
//           cerr << e.what() << endl;
//        }
//     }
//  }
}

}


// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2018 Artificial Intelligence Techniques, SL.
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
