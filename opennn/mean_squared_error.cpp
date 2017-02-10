/****************************************************************************************************************/
/*                                                                                                              */
/*   OpenNN: Open Neural Networks Library                                                                       */
/*   www.opennn.net                                                                                             */
/*                                                                                                              */
/*   M E A N   S Q U A R E D   E R R O R   C L A S S                                                            */
/*                                                                                                              */
/*   Roberto Lopez                                                                                              */
/*   Artelnics - Making intelligent use of data                                                                 */
/*   robertolopez@artelnics.com                                                                                 */
/*                                                                                                              */
/****************************************************************************************************************/

// OpenNN includes

#include "mean_squared_error.h"

namespace OpenNN
{
// DEFAULT CONSTRUCTOR

/// Default constructor. 
/// It creates a mean squared error term not associated to any 
/// neural network and not measured on any data set.
/// It also initializes all the rest of class members to their default values.

MeanSquaredError::MeanSquaredError(void) : ErrorTerm()
{
}


// NEURAL NETWORK CONSTRUCTOR

/// Neural network constructor. 
/// It creates a mean squared error term object associated to a 
/// neural network object but not measured on any data set object.
/// It also initializes all the rest of class members to their default values.
/// @param new_neural_network_pointer Pointer to a neural network object.

MeanSquaredError::MeanSquaredError(NeuralNetwork* new_neural_network_pointer)
: ErrorTerm(new_neural_network_pointer)
{
}


// DATA SET CONSTRUCTOR

/// Data set constructor. 
/// It creates a mean squared error term not associated to any 
/// neural network but to be measured on a given data set object.
/// It also initializes all the rest of class members to their default values.
/// @param new_data_set_pointer Pointer to a data set object.

MeanSquaredError::MeanSquaredError(DataSet* new_data_set_pointer)
: ErrorTerm(new_data_set_pointer)
{
}


// NEURAL NETWORK AND DATA SET CONSTRUCTOR

/// Neural network and data set constructor. 
/// It creates a mean squared error term object associated to a 
/// neural network and measured on a data set.
/// It also initializes all the rest of class members to their default values.
/// @param new_neural_network_pointer Pointer to a neural network object.
/// @param new_data_set_pointer Pointer to a data set object.

MeanSquaredError::MeanSquaredError(NeuralNetwork* new_neural_network_pointer, DataSet* new_data_set_pointer)
: ErrorTerm(new_neural_network_pointer, new_data_set_pointer)
{
}


// XML CONSTRUCTOR

/// XML constructor. 
/// It creates a mean squared error object with all pointers set to NULL. 
/// The object members are loaded by means of a XML document.
/// Please be careful with the format of that file, which is specified in the OpenNN manual.
/// @param mean_squared_error_document TinyXML document with the mean squared error elements.

MeanSquaredError::MeanSquaredError(const tinyxml2::XMLDocument& mean_squared_error_document)
 : ErrorTerm(mean_squared_error_document)
{
}


// COPY CONSTRUCTOR

/// Copy constructor. 
/// It creates a copy of an existing mean squared error object. 
/// @param other_mean_squared_error Mean squared error object to be copied.

MeanSquaredError::MeanSquaredError(const MeanSquaredError& other_mean_squared_error)
: ErrorTerm(other_mean_squared_error)
{
}


// DESTRUCTOR

/// Destructor.

MeanSquaredError::~MeanSquaredError(void)
{
}


// METHODS

// void check(void) const method

/// Checks that there are a neural network and a data set associated to the mean squared error, 
/// and that the numbers of inputs and outputs in the neural network are equal to the numbers of inputs and targets in the data set. 
/// If some of the above conditions is not hold, the method throws an exception. 

void MeanSquaredError::check(void) const
{
   std::ostringstream buffer;

   // Neural network stuff

   if(!neural_network_pointer)
   {
      buffer << "OpenNN Exception: MeanSquaredError class.\n"
             << "void check(void) const method.\n"
             << "Pointer to neural network is NULL.\n";

      throw std::logic_error(buffer.str());	  
   }

   const MultilayerPerceptron* multilayer_perceptron_pointer = neural_network_pointer->get_multilayer_perceptron_pointer();

   if(!multilayer_perceptron_pointer)
   {
      buffer << "OpenNN Exception: MeanSquaredError class.\n"
             << "void check(void) const method.\n"
             << "Pointer to multilayer perceptron is NULL.\n";

      throw std::logic_error(buffer.str());	  
   }

   const size_t inputs_number = multilayer_perceptron_pointer->get_inputs_number();
   const size_t outputs_number = multilayer_perceptron_pointer->get_outputs_number();

   if(inputs_number == 0)
   {
      buffer << "OpenNN Exception: MeanSquaredError class.\n"
             << "void check(void) const method.\n"
             << "Number of inputs in multilayer perceptron object is zero.\n";

      throw std::logic_error(buffer.str());	  
   }

   if(outputs_number == 0)
   {
      buffer << "OpenNN Exception: MeanSquaredError class.\n"
             << "void check(void) const method.\n"
             << "Number of outputs in multilayer perceptron object is zero.\n";

      throw std::logic_error(buffer.str());	  
   }

   // Data set stuff

   if(!data_set_pointer)
   {
      buffer << "OpenNN Exception: MeanSquaredError class.\n"
             << "void check(void) const method.\n"
             << "Pointer to data set is NULL.\n";

      throw std::logic_error(buffer.str());	  
   }

   // Sum squared error stuff

   const Variables& variables = data_set_pointer->get_variables();

   const size_t data_set_inputs_number = variables.count_inputs_number();
   const size_t data_set_targets_number = variables.count_targets_number();

   if(inputs_number != data_set_inputs_number)
   {
      buffer << "OpenNN Exception: MeanSquaredError class.\n"
             << "void check(void) const method.\n"
             << "Number of inputs in multilayer perceptron must be equal to number of inputs in data set.\n";

      throw std::logic_error(buffer.str());	  
   }

   if(outputs_number != data_set_targets_number)
   {
      buffer << "OpenNN Exception: MeanSquaredError class.\n"
             << "void check(void) const method.\n"
             << "Number of outputs in multilayer perceptron must be equal to number of targets in data set.\n";

      throw std::logic_error(buffer.str());
   }
}


// double calculate_error(void) const method

/// Returns the mean squared error of a neural network on a data set.

double MeanSquaredError::calculate_error(void) const
{
   // Control sentence

   #ifdef __OPENNN_DEBUG__ 

   check();

   #endif

   // Neural network stuff 

   const MultilayerPerceptron* multilayer_perceptron_pointer = neural_network_pointer->get_multilayer_perceptron_pointer();

   const size_t inputs_number = multilayer_perceptron_pointer->get_inputs_number();
   const size_t outputs_number = multilayer_perceptron_pointer->get_outputs_number();

   // Data set stuff 

   const Instances& instances = data_set_pointer->get_instances();

   const size_t training_instances_number = instances.count_training_instances_number();

   const Vector<size_t> training_indices = instances.arrange_training_indices();

   size_t training_index;

   const Variables& variables = data_set_pointer->get_variables();

   const Vector<size_t> inputs_indices = variables.arrange_inputs_indices();
   const Vector<size_t> targets_indices = variables.arrange_targets_indices();

   // Mean squared error stuff

   Vector<double> inputs(inputs_number);
   Vector<double> outputs(outputs_number);
   Vector<double> targets(outputs_number);

   int i = 0;

   double sum_squared_error = 0.0;

   #pragma omp parallel for private(i, training_index, inputs, outputs, targets) reduction(+:sum_squared_error)

   for(i = 0; i < (int)training_instances_number; i++)
   {
       training_index = training_indices[i];

      // Input vector

      inputs = data_set_pointer->get_instance(training_index, inputs_indices);

      // Output vector

      outputs = multilayer_perceptron_pointer->calculate_outputs(inputs);

      // Target vector

      targets = data_set_pointer->get_instance(training_index, targets_indices);

      // Sum squared error

	  sum_squared_error += outputs.calculate_sum_squared_error(targets);
   }

   return(sum_squared_error/(double)training_instances_number);
}


// double calculate_error(const Vector<double>&) const method

/// Returns which would be the error term of a neural network for an hypothetical
/// vector of parameters. It does not set that vector of parameters to the neural network. 
/// @param parameters Vector of potential parameters for the neural network associated to the error term.

double MeanSquaredError::calculate_error(const Vector<double>& parameters) const
{
   // Control sentence (if debug)

   #ifdef __OPENNN_DEBUG__ 

   check();

   #endif

   #ifdef __OPENNN_DEBUG__ 

   const size_t size = parameters.size();

   const size_t parameters_number = neural_network_pointer->count_parameters_number();

   if(size != parameters_number)
   {
      std::ostringstream buffer;

      buffer << "OpenNN Exception: MeanSquaredError class.\n"
             << "double calculate_error(const Vector<double>&) const method.\n"
             << "Size (" << size << ") must be equal to number of parameters (" << parameters_number << ").\n";

      throw std::logic_error(buffer.str());
   }

   #endif

   // Neural network stuff

   const MultilayerPerceptron* multilayer_perceptron_pointer = neural_network_pointer->get_multilayer_perceptron_pointer();

   const size_t inputs_number = multilayer_perceptron_pointer->get_inputs_number();
   const size_t outputs_number = multilayer_perceptron_pointer->get_outputs_number();

   // Data set stuff

   const Instances& instances = data_set_pointer->get_instances();

   const size_t training_instances_number = instances.count_training_instances_number();

   const Vector<size_t> training_indices = instances.arrange_training_indices();

   size_t training_index;

   const Variables& variables = data_set_pointer->get_variables();

   const Vector<size_t> inputs_indices = variables.arrange_inputs_indices();
   const Vector<size_t> targets_indices = variables.arrange_targets_indices();

   // Mean squared error stuff

   Vector<double> inputs(inputs_number);
   Vector<double> outputs(outputs_number);
   Vector<double> targets(outputs_number);

   double sum_squared_error = 0.0;

   int i = 0;

   #pragma omp parallel for private(i, training_index, inputs, outputs, targets) reduction(+:sum_squared_error)

   for(i = 0; i < (int)training_instances_number; i++)
   {
       training_index = training_indices[i];

      // Input vector

      inputs = data_set_pointer->get_instance(training_index, inputs_indices);

      // Output vector

      outputs = multilayer_perceptron_pointer->calculate_outputs(inputs, parameters);

      // Target vector

      targets = data_set_pointer->get_instance(training_index, targets_indices);

      // Sum squared error

      sum_squared_error += outputs.calculate_sum_squared_error(targets);
   }

   return(sum_squared_error/(double)training_instances_number);
}


// double calculate_selection_error(void) const method

/// Returns the mean squared error of the multilayer perceptron measured on the selection instances of the 
/// data set.

double MeanSquaredError::calculate_selection_error(void) const
{
   // Control sentence (if debug)

   #ifdef __OPENNN_DEBUG__ 

   check();

   #endif

   const MultilayerPerceptron* multilayer_perceptron_pointer = neural_network_pointer->get_multilayer_perceptron_pointer();

   const size_t inputs_number = multilayer_perceptron_pointer->get_inputs_number();
   const size_t outputs_number = multilayer_perceptron_pointer->get_outputs_number();

   const Instances& instances = data_set_pointer->get_instances();

   const size_t selection_instances_number = instances.count_selection_instances_number();

   if(selection_instances_number == 0)
   {
      return(0.0);
   }

   const Vector<size_t> selection_indices = instances.arrange_selection_indices();

   size_t selection_index;

   const Variables& variables = data_set_pointer->get_variables();

   const Vector<size_t> inputs_indices = variables.arrange_inputs_indices();
   const Vector<size_t> targets_indices = variables.arrange_targets_indices();


      Vector<double> inputs(inputs_number);
      Vector<double> outputs(outputs_number);
      Vector<double> targets(outputs_number);

      double selection_loss = 0.0;

      int i = 0;

      #pragma omp parallel for private(i, selection_index, inputs, outputs, targets) reduction(+:selection_loss)

      for(i = 0; i < (int)selection_instances_number; i++)
      {
          selection_index = selection_indices[i];

         // Input vector

         inputs = data_set_pointer->get_instance(selection_index, inputs_indices);

         // Output vector

         outputs = multilayer_perceptron_pointer->calculate_outputs(inputs);

         // Target vector

         targets = data_set_pointer->get_instance(selection_index, targets_indices);

         // Sum of squares error

         selection_loss += outputs.calculate_sum_squared_error(targets);
      }

      return(selection_loss/(double)selection_instances_number);
}


// Vector<double> calculate_output_gradient(const Vector<double>&, const Vector<double>&) const method

Vector<double> MeanSquaredError::calculate_output_gradient(const Vector<double>& output, const Vector<double>& target) const
{
    const Instances& instances = data_set_pointer->get_instances();

    const size_t training_instances_number = instances.count_training_instances_number();

    const Vector<double> output_gradient = (output-target)*(2.0/(double)training_instances_number);

    return(output_gradient);
}


// Matrix<double> calculate_output_Hessian(const Vector<double>&, const Vector<double>&) const method

Matrix<double> MeanSquaredError::calculate_output_Hessian(const Vector<double>&, const Vector<double>&) const
{
    const Instances& instances = data_set_pointer->get_instances();

    const size_t training_instances_number = instances.count_training_instances_number();

    const size_t outputs_number = neural_network_pointer->get_multilayer_perceptron_pointer()->get_outputs_number();

    Matrix<double> output_Hessian(outputs_number, outputs_number);
    output_Hessian.initialize_diagonal(2.0/training_instances_number);

    return(output_Hessian);
}


// FirstOrderPerformance calculate_first_order_loss(void) const method

/// @todo

ErrorTerm::FirstOrderPerformance MeanSquaredError::calculate_first_order_loss(void) const
{
    // Control sentence

    #ifdef __OPENNN_DEBUG__

    check();

    #endif

   FirstOrderPerformance first_order_loss;

   first_order_loss.loss = calculate_error();
   first_order_loss.gradient = calculate_gradient();

   return(first_order_loss);
}


// SecondOrderloss calculate_second_order_loss(void) const method

/// @todo

ErrorTerm::SecondOrderPerformance MeanSquaredError::calculate_second_order_loss(void) const
{
    // Control sentence

    #ifdef __OPENNN_DEBUG__

    check();

    #endif

   SecondOrderPerformance second_order_loss;

   second_order_loss.loss = calculate_error();
   second_order_loss.gradient = calculate_gradient();
   second_order_loss.Hessian = calculate_Hessian();

   return(second_order_loss);
}


// Vector<double> calculate_terms(void) const method

/// Returns loss vector of the error terms function for the mean squared error.
/// It uses the error back-propagation method.

Vector<double> MeanSquaredError::calculate_terms(void) const
{
   // Control sentence

   #ifdef __OPENNN_DEBUG__ 

   check();

   #endif

   // Neural network stuff

   const MultilayerPerceptron* multilayer_perceptron_pointer = neural_network_pointer->get_multilayer_perceptron_pointer();

   const size_t inputs_number = multilayer_perceptron_pointer->get_inputs_number();
   const size_t outputs_number = multilayer_perceptron_pointer->get_outputs_number();

   // Data set stuff

   const Instances& instances = data_set_pointer->get_instances();

   const size_t training_instances_number = instances.count_training_instances_number();

   const Vector<size_t> training_indices = instances.arrange_training_indices();

   size_t training_index;

   const Variables& variables = data_set_pointer->get_variables();

   const Vector<size_t> inputs_indices = variables.arrange_inputs_indices();
   const Vector<size_t> targets_indices = variables.arrange_targets_indices();

   // Mean squared error stuff

   Vector<double> error_terms(training_instances_number);

   Vector<double> inputs(inputs_number);
   Vector<double> outputs(outputs_number);
   Vector<double> targets(outputs_number);

   int i = 0;

   #pragma omp parallel for private(i, training_index, inputs, outputs, targets)

   for(i = 0; i < (int)training_instances_number; i++)
   {
       training_index = training_indices[i];

      // Input vector

      inputs = data_set_pointer->get_instance(training_index, inputs_indices);

      // Output vector

      outputs = multilayer_perceptron_pointer->calculate_outputs(inputs);

      // Target vector

      targets = data_set_pointer->get_instance(training_index, targets_indices);

      // Error

      error_terms[i] = outputs.calculate_distance(targets);
   }

   return(error_terms/sqrt((double)training_instances_number));
}


// Vector<double> calculate_terms(const Vector<double>&) const method

/// Returns which would be the error terms loss vector of a multilayer perceptron for an hypothetical vector of multilayer perceptron parameters.
/// It does not set that vector of parameters to the multilayer perceptron. 
/// @param network_parameters Vector of a potential multilayer_perceptron_pointer parameters for the multilayer perceptron associated to the loss functional.

Vector<double> MeanSquaredError::calculate_terms(const Vector<double>& network_parameters) const
{
   // Control sentence (if debug)

   #ifdef __OPENNN_DEBUG__ 

   check();

   #endif

   #ifdef __OPENNN_DEBUG__ 

   std::ostringstream buffer;

   const size_t size = network_parameters.size();

   const size_t parameters_number = neural_network_pointer->count_parameters_number();

   if(size != parameters_number)
   {
      buffer << "OpenNN Exception: MeanSquaredError class.\n"
             << "double calculate_terms(const Vector<double>&) const method.\n"
             << "Size (" << size << ") must be equal to number of multilayer perceptron parameters (" << parameters_number << ").\n";

      throw std::logic_error(buffer.str());
   }

   #endif

   NeuralNetwork neural_network_copy(*neural_network_pointer);

   neural_network_copy.set_parameters(network_parameters);

   MeanSquaredError mean_squared_error_copy(*this);

   mean_squared_error_copy.set_neural_network_pointer(&neural_network_copy);

   return(mean_squared_error_copy.calculate_terms());
}


// Matrix<double> calculate_terms_Jacobian(void) const method

/// Returns the Jacobian matrix of the mean squared error function, whose elements are given by the 
/// derivatives of the squared errors data set with respect to the multilayer perceptron parameters.

Matrix<double> MeanSquaredError::calculate_terms_Jacobian(void) const
{
   // Control sentence

   #ifdef __OPENNN_DEBUG__ 

   check();

   #endif

   // Neural network stuff

   const MultilayerPerceptron* multilayer_perceptron_pointer = neural_network_pointer->get_multilayer_perceptron_pointer();

   const size_t inputs_number = multilayer_perceptron_pointer->get_inputs_number();
   const size_t outputs_number = multilayer_perceptron_pointer->get_outputs_number();

   const size_t layers_number = multilayer_perceptron_pointer->get_layers_number();

   const size_t neural_parameters_number = multilayer_perceptron_pointer->count_parameters_number();

   Vector< Vector< Vector<double> > > first_order_forward_propagation(2);

   Vector<double> particular_solution;
   Vector<double> homogeneous_solution;

   const bool has_conditions_layer = neural_network_pointer->has_conditions_layer();

   const ConditionsLayer* conditions_layer_pointer = has_conditions_layer ? neural_network_pointer->get_conditions_layer_pointer() : NULL;

   // Data set stuff

   const Instances& instances = data_set_pointer->get_instances();

   const size_t training_instances_number = instances.count_training_instances_number();

   const Vector<size_t> training_indices = instances.arrange_training_indices();

   size_t training_index;

   const Variables& variables = data_set_pointer->get_variables();

   const Vector<size_t> inputs_indices = variables.arrange_inputs_indices();
   const Vector<size_t> targets_indices = variables.arrange_targets_indices();

   Vector<double> inputs(inputs_number);
   Vector<double> targets(outputs_number);

   // Loss index

   Vector<double> term(outputs_number);
   double term_norm;

   Vector<double> output_gradient(outputs_number);

   Vector< Vector<double> > layers_delta(layers_number);
   Vector<double> point_gradient(neural_parameters_number);

   Matrix<double> terms_Jacobian(training_instances_number, neural_parameters_number);

   // Main loop

   int i = 0;

#pragma omp parallel for private(i, training_index, inputs, targets, first_order_forward_propagation,  \
 term, term_norm, output_gradient, layers_delta, particular_solution, homogeneous_solution, point_gradient)

   for(i = 0; i < (int)training_instances_number; i++)
   {
       training_index = training_indices[i];

      inputs = data_set_pointer->get_instance(training_index, inputs_indices);

      targets = data_set_pointer->get_instance(training_index, targets_indices);

      first_order_forward_propagation = multilayer_perceptron_pointer->calculate_first_order_forward_propagation(inputs);

      const Vector< Vector<double> >& layers_activation = first_order_forward_propagation[0];
      const Vector< Vector<double> >& layers_activation_derivative = first_order_forward_propagation[1];

      if(!has_conditions_layer)
      {
         const Vector<double>& outputs = first_order_forward_propagation[0][layers_number-1]; 

         term = (outputs-targets);
         term_norm = term.calculate_norm();

         if(term_norm == 0.0)
         {
             output_gradient.set(outputs_number, 0.0);
         }
         else
         {
            output_gradient = term/term_norm;
         }

         layers_delta = calculate_layers_delta(layers_activation_derivative, output_gradient);
      }
      else
      {
         particular_solution = conditions_layer_pointer->calculate_particular_solution(inputs);
         homogeneous_solution = conditions_layer_pointer->calculate_homogeneous_solution(inputs);

         term = (particular_solution+homogeneous_solution*layers_activation[layers_number-1] - targets)/sqrt((double)training_instances_number);              
         term_norm = term.calculate_norm();

         if(term_norm == 0.0)
         {
             output_gradient.set(outputs_number, 0.0);
         }
         else
         {
            output_gradient = term/term_norm;
         }

         layers_delta = calculate_layers_delta(layers_activation_derivative, homogeneous_solution, output_gradient);
	  }

      point_gradient = calculate_point_gradient(inputs, layers_activation, layers_delta);

      terms_Jacobian.set_row(i, point_gradient);
  }

   return(terms_Jacobian/sqrt((double)training_instances_number));
}


// FirstOrderTerms calculate_first_order_terms(void) const method

/// Returns a first order terms loss structure, which contains the values and the Jacobian of the error terms function.

/// @todo

MeanSquaredError::FirstOrderTerms MeanSquaredError::calculate_first_order_terms(void) const
{
    // Control sentence (if debug)

    #ifdef __OPENNN_DEBUG__

    check();

    #endif

   FirstOrderTerms first_order_terms;

   first_order_terms.terms = calculate_terms();

   first_order_terms.Jacobian = calculate_terms_Jacobian();

   return(first_order_terms);
}


// std::string write_error_term_type(void) const method

/// Returns a string with the name of the mean squared error loss type, "MEAN_SQUARED_ERROR".

std::string MeanSquaredError::write_error_term_type(void) const
{
   return("MEAN_SQUARED_ERROR");
}


// tinyxml2::XMLDocument* to_XML(void) const method 

/// Serializes the mean squared error object into a XML document of the TinyXML library. 
/// See the OpenNN manual for more information about the format of this document-> 

tinyxml2::XMLDocument* MeanSquaredError::to_XML(void) const
{
   std::ostringstream buffer;

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

   return(document);
}


// void write_XML(tinyxml2::XMLPrinter &) const method

void MeanSquaredError::write_XML(tinyxml2::XMLPrinter&) const
{
    //file_stream.OpenElement("MeanSquaredError");

    //file_stream.CloseElement();
}


}

// OpenNN: Open Neural Networks Library.
// Copyright (c) 2005-2016 Roberto Lopez.
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
