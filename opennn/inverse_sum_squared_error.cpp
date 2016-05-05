/****************************************************************************************************************/
/*                                                                                                              */
/*   OpenNN: Open Neural Networks Library                                                                       */
/*   www.opennn.net                                                                                             */
/*                                                                                                              */
/*   I N V E R S E   S U M   S Q U A R E D   E R R O R   C L A S S                                              */
/*                                                                                                              */
/*   Roberto Lopez                                                                                              */
/*   Artelnics - Making intelligent use of data                                                                 */
/*   robertolopez@artelnics.com                                                                                 */
/*                                                                                                              */
/****************************************************************************************************************/


// OpenNN includes

#include "inverse_sum_squared_error.h"


namespace OpenNN
{

// DEFAULT CONSTRUCTOR

/// Default constructor. 
/// It creates an inverse sum squared error performance term not associated to any neural network and not measured on any mathematical model and data set.
/// It also initializes all the rest of class members to their default values.

InverseSumSquaredError::InverseSumSquaredError(void) : PerformanceTerm()
{
   construct_numerical_differentiation();

   set_default();
}

// NEURAL NETWORK CONSTRUCTOR

/// Neural network constructor. 
/// It creates an inverse sum squared error associated to a neural network but neither measured on a mathematical model nor on a data set.
/// It also initializes all the rest of class members to their default values.
/// @param new_neural_network_pointer Pointer to a neural network object.

InverseSumSquaredError::InverseSumSquaredError(NeuralNetwork* new_neural_network_pointer)
 : PerformanceTerm(new_neural_network_pointer)
{
   construct_numerical_differentiation();

   set_default();
}


// GENERAL CONSTRUCTOR

/// Neural network, mathematical model and data set constructor. 
/// It creates an inverse sum squared error associated to a neural network and measured on a mathematical model and a data set.
/// It also initializes all the rest of class members to their default values.
/// @param new_neural_network_pointer Pointer to a neural network object.
/// @param new_mathematical_model_pointer Pointer to a mathematical model object.
/// @param new_data_set_pointer Pointer to a data set object.

InverseSumSquaredError::InverseSumSquaredError(NeuralNetwork* new_neural_network_pointer, MathematicalModel* new_mathematical_model_pointer, DataSet* new_data_set_pointer)
 : PerformanceTerm(new_neural_network_pointer, new_mathematical_model_pointer, new_data_set_pointer)
{
   construct_numerical_differentiation();

   set_default();
}


// XML CONSTRUCTOR

/// XML constructor. 
/// It creates an inverse sum squared error performance term not associated to any neural network and not measured on any mathematical model and data set.
/// It initializes all the member data from a XML document.
/// @param inverse_sum_squared_error_document Pointer to a TinyXML document containing the inverse sum squared error data. 

InverseSumSquaredError::InverseSumSquaredError(const tinyxml2::XMLDocument& inverse_sum_squared_error_document)
 : PerformanceTerm(inverse_sum_squared_error_document)
{
   construct_numerical_differentiation();

   set_default();
}


// DESTRUCTOR

/// Destructor. 
/// It does not delete any object. 

InverseSumSquaredError::~InverseSumSquaredError(void) 
{
}


// METHODS

// const UnknownsMethod& get_unknowns_method(void) const method

/// This returns the method for entering the unknown values or functions into the mathematical model. 

const InverseSumSquaredError::UnknownsMethod& InverseSumSquaredError::get_unknowns_method(void) const
{
   return(unknowns_method);
}


// std::string write_unknowns_method(void) const method

/// This returns a string with the name of the method for entering the unknown values or functions into the mathematical model. 

std::string InverseSumSquaredError::write_unknowns_method(void) const
{
   switch(unknowns_method)
   {
      case LookUpTable:
      {
         return("LookUpTable");
      }
      break;

      case IndependentParametersValues:
      {
         return("IndependentParametersValues");
      }
      break;

      default:
      {
         std::ostringstream buffer;

         buffer << "OpenNN Exception: InverseSumSquaredError class.\n"
                << "std::string write_unknowns_method(void) const method.\n"
                << "Unknown property method.\n";
 
	     throw std::logic_error(buffer.str());
      }
      break;
   }   
}


// void set_unknowns_method(const UnknownsMethod&) method

/// This sets a new the method for entering the unknown values or functions into the mathematical model. 
/// @param new_unknowns_method Unknowns method (independent parameters, look-up table, ...).

void InverseSumSquaredError::set_unknowns_method(const UnknownsMethod& new_unknowns_method)
{
   unknowns_method = new_unknowns_method;
}


// void set_unknowns_method(const std::string&) method

/// This sets a new the method for entering the unknown values or functions into the mathematical model. 
/// @param new_unknowns_method String with the name of the unknowns method (independent parameters, look-up table, ...).

void InverseSumSquaredError::set_unknowns_method(const std::string& new_unknowns_method)
{
   if(new_unknowns_method == "LookUpTable")
   {
      set_unknowns_method(LookUpTable);
   }
   else if(new_unknowns_method == "IndependentParametersValues")
   {
      set_unknowns_method(IndependentParametersValues);
   }
   else
   {
      std::ostringstream buffer;

      buffer << "OpenNN Exception: ScalingLayer class.\n"
             << "void set_unknowns_method(const std::string&) method.\n"
			 << "Unknown property method: " << new_unknowns_method << ".\n";

	  throw std::logic_error(buffer.str());
   }
}


// void set_default(void) method

/// Sets the default values of the inverse sum squared error:
/// <ul>
/// <li> Unknowns method: Independent parameters values.
/// <li> Display: true. 
/// </ul>

void InverseSumSquaredError::set_default(void)
{
   unknowns_method = IndependentParametersValues;

   display = true;
}


// void check(void) const method

/// Checks that there are a neural network, a mathemacial model and a data set associated to the inverse sum squared error. 
/// If some of the above conditions is not hold, the method throws an exception. 

void InverseSumSquaredError::check(void) const
{
   std::ostringstream buffer;

   // Neural network stuff

   if(!neural_network_pointer)
   {
      buffer << "OpenNN Exception: InverseSumSquaredError class.\n"
             << "void check(void) const method.\n"
             << "Pointer to neural network is NULL.\n";

      throw std::logic_error(buffer.str());	  
   }

   const MultilayerPerceptron* multilayer_perceptron_pointer = neural_network_pointer->get_multilayer_perceptron_pointer();

   if(!multilayer_perceptron_pointer)
   {
      buffer << "OpenNN Exception: InverseSumSquaredError class.\n"
             << "void check(void) const method.\n"
             << "Pointer to multilayer perceptron is NULL.\n";

      throw std::logic_error(buffer.str());	  
   }

   const size_t inputs_number = multilayer_perceptron_pointer->get_inputs_number();
   const size_t outputs_number = multilayer_perceptron_pointer->get_outputs_number();

   if(inputs_number == 0)
   {
      buffer << "OpenNN Exception: InverseSumSquaredError class.\n"
             << "void check(void) const method.\n"
             << "Number of inputs in multilayer perceptron object is zero.\n";

      throw std::logic_error(buffer.str());	  
   }

   if(outputs_number == 0)
   {
      buffer << "OpenNN Exception: InverseSumSquaredError class.\n"
             << "void check(void) const method.\n"
             << "Number of outputs in multilayer perceptron object is zero.\n";

      throw std::logic_error(buffer.str());	  
   }

   // Mathematical model stuff

   if(!mathematical_model_pointer)
   {
      buffer << "OpenNN Exception: InverseSumSquaredError class.\n"
             << "void check(void) const method.\n"
             << "Pointer to mathematical model is NULL.\n";

      throw std::logic_error(buffer.str());	  
   }

   // Data set stuff 

   if(!data_set_pointer)
   {
      buffer << "OpenNN Exception: InverseSumSquaredError class.\n"
             << "void check(void) const method.\n"
             << "Pointer to data set is NULL.\n";

      throw std::logic_error(buffer.str());	  
   }

   // Final solutions error stuff

}


// double calculate_performance(void) const method

/// @todo

double InverseSumSquaredError::calculate_performance(void) const
{
   // Control sentence

   #ifdef __OPENNN_DEBUG__ 

   check();

   #endif

   // Data set stuff

   const Matrix<double> training_input_data = data_set_pointer->arrange_training_input_data();
   const Matrix<double> training_target_data = data_set_pointer->arrange_training_target_data();

   const size_t training_instances_number = training_input_data.get_rows_number();

   // Mathematical model stuff 

   const Matrix<double> training_solution_data = mathematical_model_pointer->calculate_dependent_variables(*neural_network_pointer, training_input_data);

   return(training_solution_data.calculate_sum_squared_error(training_target_data)/(double)training_instances_number);
}


// double calculate_performance(const Vector<double>&) const method

double InverseSumSquaredError::calculate_performance(const Vector<double>& potential_parameters) const
{
   // Control sentence (if debug)

   #ifdef __OPENNN_DEBUG__ 

   check();

   #endif

   #ifdef __OPENNN_DEBUG__ 

   const size_t size = potential_parameters.size();

   const size_t parameters_number = neural_network_pointer->count_parameters_number();

   if(size != parameters_number)
   {
      std::ostringstream buffer;

      buffer << "OpenNN Exception: InverseSumSquaredError class.\n"
             << "double calculate_performance(const Vector<double>&) const method.\n"
             << "Size (" << size << ") must be equal to number of parameters (" << parameters_number << ").\n";

      throw std::logic_error(buffer.str());	  
   }

   #endif

   NeuralNetwork neural_network_copy(*neural_network_pointer);

   neural_network_copy.set_parameters(potential_parameters);

   InverseSumSquaredError inverse_sum_squared_error_copy(*this);

   inverse_sum_squared_error_copy.set_neural_network_pointer(&neural_network_copy);

   return(inverse_sum_squared_error_copy.calculate_performance());
}


// double calculate_selection_performance(void) const method

/// @todo

double InverseSumSquaredError::calculate_selection_performance(void) const
{
   // Control sentence (if debug)

   #ifdef __OPENNN_DEBUG__ 

   check();

   #endif

   // Neural network

   const MultilayerPerceptron* multilayer_perceptron_pointer = neural_network_pointer->get_multilayer_perceptron_pointer();

   const size_t inputs_number = multilayer_perceptron_pointer->get_inputs_number();
   const size_t outputs_number = multilayer_perceptron_pointer->get_outputs_number();

   // Data set

   const Instances& instances = data_set_pointer->get_instances();

   const size_t selection_instances_number = instances.count_selection_instances_number();

   const Vector<size_t> selection_indices = instances.arrange_selection_indices();

   size_t selection_index;

   const Variables& variables = data_set_pointer->get_variables();

   const Vector<size_t> inputs_indices = variables.arrange_inputs_indices();
   const Vector<size_t> targets_indices = variables.arrange_targets_indices();

   // Performance functional

   Vector<double> inputs(inputs_number);
   Vector<double> outputs(outputs_number);
   Vector<double> targets(outputs_number);

   double selection_objective = 0.0;

   int i = 0;

   #pragma omp parallel for private(i, selection_index, inputs, outputs, targets) reduction(+ : selection_objective)

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

      selection_objective += outputs.calculate_sum_squared_error(targets);           
   }

   return(selection_objective);
}


// std::string write_performance_term_type(void) const method

/// Returns a string with the name of the inverser sum squared error performance type, "INVERSE_SUM_SQUARED_ERROR".

std::string InverseSumSquaredError::write_performance_term_type(void) const
{
   return("INVERSE_SUM_SQUARED_ERROR");
}


// tinyxml2::XMLDocument* to_XML(void) const method 

/// Serializes the inverse sum squared error object into a XML document of the TinyXML library. 
/// See the OpenNN manual for more information about the format of this document-> 

tinyxml2::XMLDocument* InverseSumSquaredError::to_XML(void) const
{
   std::ostringstream buffer;

   tinyxml2::XMLDocument* document = new tinyxml2::XMLDocument;

   // Inverse sum squared error

   tinyxml2::XMLElement* inverse_sum_squared_error_element = document->NewElement("InverseSumSquaredError");

   document->InsertFirstChild(inverse_sum_squared_error_element);

   // Numerical differentiation

   if(numerical_differentiation_pointer)
   {
	  tinyxml2::XMLElement* element = numerical_differentiation_pointer->to_XML()->FirstChildElement();
      inverse_sum_squared_error_element->LinkEndChild(element);
   }

   // Unknowns method
   {
      tinyxml2::XMLElement* element = document->NewElement("UnknownsMethod");
      inverse_sum_squared_error_element->LinkEndChild(element);

      tinyxml2::XMLText* text = document->NewText(write_unknowns_method().c_str());
      element->LinkEndChild(text);
   }

   // Display

//   {
//      tinyxml2::XMLElement* display_element = document->NewElement("Display");
//      inverse_sum_squared_error_element->LinkEndChild(display_element);

//      buffer.str("");
//      buffer << display;

//      tinyxml2::XMLText* display_text = document->NewText(buffer.str().c_str());
//      display_element->LinkEndChild(display_text);
//   }

   return(document);

}


// void from_XML(const tinyxml2::XMLDocument&) method

/// @todo

void InverseSumSquaredError::from_XML(const tinyxml2::XMLDocument&)   
{

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
