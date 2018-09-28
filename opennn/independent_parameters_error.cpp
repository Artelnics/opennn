/****************************************************************************************************************/
/*                                                                                                              */
/*   OpenNN: Open Neural Networks Library                                                                       */
/*   www.opennn.net                                                                                             */
/*                                                                                                              */
/*   I N D E P E N D E N T   P A R A M E T E R S   E R R O R   C L A S S                                        */
/*                                                                                                              */
/*   Roberto Lopez                                                                                              */
/*   Artelnics - Making intelligent use of data                                                                 */
/*   robertolopez@artelnics.com                                                                                 */
/*                                                                                                              */
/****************************************************************************************************************/

// OpenNN includes

#include "independent_parameters_error.h"


namespace OpenNN
{

// DEFAULT CONSTRUCTOR

/// Default constructor. 
/// It creates a independent parameters error performance term with all pointers initialized to NULL.
/// It also initializes all the rest of class members to their default values.

IndependentParametersError::IndependentParametersError(void) 
 : PerformanceTerm()
{
   construct_numerical_differentiation();

   set_default();
}


// NEURAL NETWORK CONSTRUCTOR

/// Neural network constructor. 
/// It creates a independent parameters error performance term associated to a neural network.
/// It also initializes all the rest of class members to their default values.
/// @param new_neural_network_pointer Pointer to a neural network object.

IndependentParametersError::IndependentParametersError(NeuralNetwork* new_neural_network_pointer)
: PerformanceTerm(new_neural_network_pointer)
{
   construct_numerical_differentiation();

   set_default();
}


// XML CONSTRUCTOR

/// XML constructor. 
/// It creates a independent parameters error performance term with all pointers initialized to NULL.
/// It also loads the rest of class members from a XML document.
/// @param independent_parameters_error_document TinyXML document of a independent parameters values object.

IndependentParametersError::IndependentParametersError(const tinyxml2::XMLDocument& independent_parameters_error_document)
 : PerformanceTerm(independent_parameters_error_document)
{
   construct_numerical_differentiation();

   set_default();
}


// DESTRUCTOR

/// Destructor.

IndependentParametersError::~IndependentParametersError(void)
{
}


// ASSIGNMENT OPERATOR

// FinalSolutionsError& operator = (const FinalSolutionsError&) method

/// Assignment operator.  

IndependentParametersError& IndependentParametersError::operator = (const IndependentParametersError& other_independent_parameters_error)
{
   if(this != &other_independent_parameters_error) 
   {
      *neural_network_pointer = *other_independent_parameters_error.neural_network_pointer;
      *data_set_pointer = *other_independent_parameters_error.data_set_pointer;
      *mathematical_model_pointer = *other_independent_parameters_error.mathematical_model_pointer;
      *numerical_differentiation_pointer = *other_independent_parameters_error.numerical_differentiation_pointer;
      display = other_independent_parameters_error.display;

      target_independent_parameters = other_independent_parameters_error.target_independent_parameters;
      independent_parameters_errors_weights = other_independent_parameters_error.independent_parameters_errors_weights;
   }

   return(*this);
}


// EQUAL TO OPERATOR

// bool operator == (const FinalSolutionsError&) const method

/// Equal to operator. 

bool IndependentParametersError::operator == (const IndependentParametersError& other_independent_parameters_error) const
{
   if(*neural_network_pointer == *other_independent_parameters_error.neural_network_pointer
   && *mathematical_model_pointer == *other_independent_parameters_error.mathematical_model_pointer
   && *numerical_differentiation_pointer == *other_independent_parameters_error.numerical_differentiation_pointer
   && display == other_independent_parameters_error.display    
   && target_independent_parameters == other_independent_parameters_error.target_independent_parameters
   && independent_parameters_errors_weights == other_independent_parameters_error.independent_parameters_errors_weights)
   {
      return(true);
   }
   else
   {
      return(false);  
   }
}


// METHODS


// const Vector<double>& get_target_independent_parameters(void) const method

/// Returns the desired values for the independent parameter. 

const Vector<double>& IndependentParametersError::get_target_independent_parameters(void) const
{
   return(target_independent_parameters);
}


// const double& get_target_independent_parameter(const size_t&) const method

/// Returns the desired value of a single independent parameter. 
/// @param i Index of independent parameter. 

const double& IndependentParametersError::get_target_independent_parameter(const size_t& i) const
{
   return(target_independent_parameters[i]);
}


// const Vector<double>& get_independent_parameters_errors_weights(void) const method

/// Returns the weight for each error between the actual independent parameters and their target values. 

const Vector<double>& IndependentParametersError::get_independent_parameters_errors_weights(void) const
{
   return(independent_parameters_errors_weights);
}


// const double& get_independent_parameter_error_weight(const size_t&) const method

/// Returns the weight for a singel error between an independent parameters and its target value. 
/// @param i Index of independent parameter parameter. 

const double& IndependentParametersError::get_independent_parameter_error_weight(const size_t& i) const
{
   return(independent_parameters_errors_weights[i]);   
}


// void set_target_independent_parameters(const Vector<double>&) method

/// Sets new desired values for the independent parameters. 
/// @param new_target_independent_parameters Vector of desired values for the independent parameters. 

void IndependentParametersError::set_target_independent_parameters(const Vector<double>& new_target_independent_parameters)
{
   target_independent_parameters = new_target_independent_parameters;
}


// void set_target_independent_parameter(const size_t&, const double&) method

/// Sets the desired value of a single independent parameter. 
/// @param i Index of independent parameter. 
/// @param new_target_independent_parameter Desired value for that parameter. 

void IndependentParametersError::set_target_independent_parameter(const size_t& i, const double& new_target_independent_parameter)
{
   target_independent_parameters[i] = new_target_independent_parameter;
}


// void set_independent_parameters_errors_weights(const Vector<double>&) method

/// Sets new weights for each error between the actual independent parameters and their target values. 
/// @param new_independent_parameters_errors_weights Vector of weights, with size the number of independent parameters.

void IndependentParametersError::set_independent_parameters_errors_weights(const Vector<double>& new_independent_parameters_errors_weights) 
{
   independent_parameters_errors_weights = new_independent_parameters_errors_weights;
}


// void set_independent_parameter_error_weight(const size_t&, const double&) method

/// Sets a new weight for the error between a single independent parameter and its target value. 
/// @param i Index of independent parameter. 
/// @param new_independent_parameter_error_weight Weight value.

void IndependentParametersError::set_independent_parameter_error_weight(const size_t& i, const double& new_independent_parameter_error_weight)
{
   independent_parameters_errors_weights[i] = new_independent_parameter_error_weight;
}


// void set_default(void) method

/// Sets the default values for this object:
/// <ul>
/// <li> Target independent parameters: 0 for all parameters. 
/// <li> Errors weights: 1 for all errors. 
/// <li> Display: True. 
/// </ul>

void IndependentParametersError::set_default(void)
{
   if(neural_network_pointer)
   {
      if(neural_network_pointer->has_independent_parameters())
	  {
         const IndependentParameters* independent_parameters_pointer = neural_network_pointer->get_independent_parameters_pointer();

         const size_t independent_parameters_number  = independent_parameters_pointer->get_parameters_number();

         target_independent_parameters.set(independent_parameters_number, 0.0);

		 independent_parameters_errors_weights.set(independent_parameters_number, 1.0);
	  }
   }
   else
   {
       target_independent_parameters.set();
       independent_parameters_errors_weights.set();
   }

   display = true;
}


// void check(void) const method

/// Checks that there are a neural network and a data set associated to the sum squared error, 
/// and that the number of independent parameters in the neural network is equal to the number of size of the target independent parameters in the performance term. 
/// If some of the above conditions is not hold, the method throws an exception. 

void IndependentParametersError::check(void) const
{
   std::ostringstream buffer;

   // Neural network stuff

   if(!neural_network_pointer)
   {
      buffer << "OpenNN Exception: IndependentParametersError class.\n"
             << "void check(void) const method.\n"
             << "Pointer to neural network is NULL.\n";

      throw std::logic_error(buffer.str());	  
   }

   const IndependentParameters* independent_parameters_pointer = neural_network_pointer->get_independent_parameters_pointer();

   if(!independent_parameters_pointer)
   {
      buffer << "OpenNN Exception: IndependentParametersError class.\n"
             << "void check(void) const method.\n"
             << "Pointer to independent parameters is NULL.\n";

      throw std::logic_error(buffer.str());	  
   }

   const size_t independent_parameters_number = independent_parameters_pointer->get_parameters_number();

   if(independent_parameters_number == 0)
   {
      buffer << "OpenNN Exception: IndependentParametersError class.\n"
             << "void check(void) const method.\n"
             << "Number of independent parameters is zero.\n";

      throw std::logic_error(buffer.str());	  
   }

   // Mathematical model stuff

   if(!mathematical_model_pointer)
   {
      buffer << "OpenNN Exception: IndependentParametersError class.\n"
             << "void check(void) const method.\n"
             << "Pointer to mathematical model is NULL.\n";

      throw std::logic_error(buffer.str());	  
   }

   // Independent parameters error stuff

   const size_t target_independent_parameters_size = target_independent_parameters.size();

   if(target_independent_parameters_size != independent_parameters_number)
   {
      buffer << "OpenNN Exception: IndependentParametersError class." << std::endl
             << "double calculate_performance(void) const method." << std::endl
             << "Size of target independent parameters must be equal to number of independent parameters." << std::endl;

      throw std::logic_error(buffer.str());	  
   }

   const size_t independent_parameters_errors_weights_size = independent_parameters_errors_weights.size();

   if(independent_parameters_errors_weights_size != independent_parameters_number)
   {
      buffer << "OpenNN Exception: IndependentParametersError class." << std::endl
             << "double calculate_performance(void) const method." << std::endl
             << "Size of independent parameters errors weights must be equal to number of independent parameters." << std::endl;

      throw std::logic_error(buffer.str());	  
   }

}


// double calculate_performance(void) const method

/// Returns the dot product between the independent parameters vector and its targets vector.

double IndependentParametersError::calculate_performance(void) const
{
   // Neural network stuff

   #ifdef __OPENNN_DEBUG__ 

   check();

   #endif

   // Neural network stuff

   const IndependentParameters* independent_parameters_pointer = neural_network_pointer->get_independent_parameters_pointer();

   const Vector<double> independent_parameters = independent_parameters_pointer->get_parameters();

   const Vector<double> independent_parameters_error = independent_parameters - target_independent_parameters;

   return((independent_parameters_errors_weights*independent_parameters_error*independent_parameters_error).calculate_sum());
}


// double calculate_performance(const Vector<double>&) const method

/// @todo

double IndependentParametersError::calculate_performance(const Vector<double>& parameters) const  
{
   // Neural network stuff

   #ifdef __OPENNN_DEBUG__ 

   check();

   #endif


#ifdef __OPENNN_DEBUG__ 

   const size_t size = parameters.size();

   const size_t parameters_number = neural_network_pointer->count_parameters_number();

   if(size != parameters_number)
   {
      std::ostringstream buffer;

      buffer << "OpenNN Exception: IndependentParametersError class." << std::endl
             << "double calculate_performance(const Vector<double>&) const method." << std::endl
             << "Size of parameters (" << size << ") must be equal to number of parameters (" << parameters_number << ")." << std::endl;

      throw std::logic_error(buffer.str());
   }

   #endif

   NeuralNetwork neural_network_copy(*neural_network_pointer);

   neural_network_copy.set_parameters(parameters);

   IndependentParametersError independent_parameters_error_copy(*this);

   independent_parameters_error_copy.set_neural_network_pointer(&neural_network_copy);

   return(independent_parameters_error_copy.calculate_performance());
}


// Vector<double> calculate_gradient(void) const method

Vector<double> IndependentParametersError::calculate_gradient(void) const
{
   // Neural network stuff

   #ifdef __OPENNN_DEBUG__ 

   check();

   #endif

   const size_t parameters_number = neural_network_pointer->count_parameters_number();

   const IndependentParameters* independent_parameters_pointer = neural_network_pointer->get_independent_parameters_pointer();

   const size_t independent_parameters_number = independent_parameters_pointer->get_parameters_number();

   const size_t neural_parameters_number = parameters_number - independent_parameters_number;

   Vector<double> multilayer_perceptron_gradient(neural_parameters_number, 0.0);

   const Vector<double> independent_parameters = independent_parameters_pointer->get_parameters();

   const Vector<double> independent_parameters_gradient = (independent_parameters - target_independent_parameters)*2.0;

   return(multilayer_perceptron_gradient.assemble(independent_parameters_gradient));
}


// Matrix<double> calculate_Hessian(void) const method

Matrix<double> IndependentParametersError::calculate_Hessian(void) const   
{
   // Neural network stuff

   #ifdef __OPENNN_DEBUG__ 

   check();

   #endif

   const size_t parameters_number = neural_network_pointer->count_parameters_number();

   const IndependentParameters* independent_parameters_pointer = neural_network_pointer->get_independent_parameters_pointer();

   const size_t independent_parameters_number = independent_parameters_pointer->get_parameters_number();

   const size_t neural_parameters_number = parameters_number - independent_parameters_number;

   Matrix<double> Hessian(parameters_number, parameters_number, 0.0);

   for(size_t i = neural_parameters_number; i < parameters_number; i++)
   {
      for(size_t j = neural_parameters_number; j < parameters_number; j++)
      {
         Hessian(i,j) = 2.0;
      }     
   }

   return(Hessian);
}


// std::string write_performance_term_type(void) const method

/// Returns a string with the name of the independent parameters error performance type, "INDEPENDENT_PARAMETERS_ERROR".

std::string IndependentParametersError::write_performance_term_type(void) const
{
   return("INDEPENDENT_PARAMETERS_ERROR");
}


// std::string write_information(void) const method

std::string IndependentParametersError::write_information(void) const
{
   std::ostringstream buffer;

   buffer << "Independent parameters error: " << calculate_performance() << "\n";

   return(buffer.str());
}


// tinyxml2::XMLDocument* to_XML(void) method method 

/// Returns a representation of the independent parameters error object, in XML format.
/// @todo Add numerical differentiation tag.

tinyxml2::XMLDocument* IndependentParametersError::to_XML(void) const
{
   std::ostringstream buffer;

   tinyxml2::XMLDocument* document = new tinyxml2::XMLDocument;

   // Independent parameters error

   tinyxml2::XMLElement* independent_parameters_error_element = document->NewElement("IndependentParametersError");

   document->InsertFirstChild(independent_parameters_error_element);

   // Numerical differentiation

//   if(numerical_differentiation_pointer)
//   {
//	  tinyxml2::XMLElement* element = numerical_differentiation_pointer->to_XML()->FirstChildElement();
//      independent_parameters_error_element->LinkEndChild(element);
//   }

   // Target independent parameters 

   {
   tinyxml2::XMLElement* element = document->NewElement("TargetIndependentParamters");
   independent_parameters_error_element->LinkEndChild(element);

   buffer.str("");
   buffer << target_independent_parameters;

   tinyxml2::XMLText* text = document->NewText(buffer.str().c_str());
   element->LinkEndChild(text);
   }

   // Independent parameters errors weights

   {
   tinyxml2::XMLElement* element = document->NewElement("IndependentParametersErrorsWeights");
   independent_parameters_error_element->LinkEndChild(element);

   buffer.str("");
   buffer << independent_parameters_errors_weights;

   tinyxml2::XMLText* text = document->NewText(buffer.str().c_str());
   element->LinkEndChild(text);
   }

   // Display

//   {
//   tinyxml2::XMLElement* display_element = document->NewElement("Display");
//   independent_parameters_error_element->LinkEndChild(display_element);

//   buffer.str("");
//   buffer << display;

//   tinyxml2::XMLText* display_text = document->NewText(buffer.str().c_str());
//   display_element->LinkEndChild(display_text);
//   }

   return(document);
}


// void from_XML(const tinyxml2::XMLDocument&) method

// This method loads a sum squared error object from a XML file. 
// @param element Name of XML sum squared error file. 

void IndependentParametersError::from_XML(const tinyxml2::XMLDocument& document)
{
    const tinyxml2::XMLElement* root_element = document.FirstChildElement("IndependentParametersError");

    if(!root_element)
    {
        std::ostringstream buffer;

        buffer << "OpenNN Exception: IndependentParametersError class.\n"
               << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
               << "Independent parameters error element is NULL.\n";

        throw std::logic_error(buffer.str());
    }

  // Display
  {
     const tinyxml2::XMLElement* display_element = root_element->FirstChildElement("Display");

     if(display_element)
     {
        const std::string new_display_string = display_element->GetText();

        try
        {
           set_display(new_display_string != "0");
        }
        catch(const std::logic_error& e)
        {
           std::cout << e.what() << std::endl;
        }
     }
  }
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
