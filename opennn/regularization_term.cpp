/****************************************************************************************************************/
/*                                                                                                              */
/*   OpenNN: Open Neural Networks Library                                                                       */
/*   www.opennn.net                                                                                             */
/*                                                                                                              */
/*   R E G U L A R I Z A T I O N   T E R M   C L A S S                                                          */
/*                                                                                                              */
/*   Roberto Lopez                                                                                              */
/*   Artelnics - Making intelligent use of data                                                                 */
/*   robertolopez@artelnics.com                                                                                 */
/*                                                                                                              */
/****************************************************************************************************************/

// OpenNN includes

#include "regularization_term.h"

namespace OpenNN
{

// DEFAULT CONSTRUCTOR

/// Default constructor. 
/// It creates a default error term object, with all pointers initialized to NULL.
/// It also initializes all the rest of class members to their default values.

RegularizationTerm::RegularizationTerm(void)
 : neural_network_pointer(NULL), 
   numerical_differentiation_pointer(NULL)
{
   set_default();
}


// NEURAL NETWORK CONSTRUCTOR

/// Neural network constructor. 
/// It creates a error term object associated to a neural network object. 
/// The rest of pointers are initialized to NULL.
/// It also initializes all the rest of class members to their default values.
/// @param new_neural_network_pointer Pointer to a neural network object.

RegularizationTerm::RegularizationTerm(NeuralNetwork* new_neural_network_pointer)
 : neural_network_pointer(new_neural_network_pointer), 
   numerical_differentiation_pointer(NULL)
{
   set_default();
}


// XML CONSTRUCTOR

/// XML constructor. 
/// It creates a default error term object, with all pointers initialized to NULL.
/// It also loads all the rest of class members from a XML document.
/// @param error_term_document Pointer to a TinyXML document with the object data.

RegularizationTerm::RegularizationTerm(const tinyxml2::XMLDocument& error_term_document)
 : neural_network_pointer(NULL), 
   numerical_differentiation_pointer(NULL)
{
   set_default();

   from_XML(error_term_document);
}


// COPY CONSTRUCTOR

/// Copy constructor. 
/// It creates a copy of an existing error term object. 
/// @param other_error_term Error term object to be copied.

RegularizationTerm::RegularizationTerm(const RegularizationTerm& other_error_term)
 : neural_network_pointer(NULL), 
   numerical_differentiation_pointer(NULL)
{
   neural_network_pointer = other_error_term.neural_network_pointer;

   if(other_error_term.numerical_differentiation_pointer)
   {
      numerical_differentiation_pointer = new NumericalDifferentiation(*other_error_term.numerical_differentiation_pointer);
   }

   display = other_error_term.display;  
}


// DESTRUCTOR

/// Destructor.
/// It deletes the numerical differentiation object composing this error term object. 

RegularizationTerm::~RegularizationTerm(void)
{
   delete numerical_differentiation_pointer;
}


// ASSIGNMENT OPERATOR

// RegularizationTerm& operator = (const RegularizationTerm&) method

/// Assignment operator. 
/// It assigns to this error term object the members from another error term object. 
/// @param other_error_term Error term object to be copied. 

RegularizationTerm& RegularizationTerm::operator = (const RegularizationTerm& other_error_term)
{
   if(this != &other_error_term) 
   {
      neural_network_pointer = other_error_term.neural_network_pointer;

      if(other_error_term.numerical_differentiation_pointer == NULL)
      {
          delete numerical_differentiation_pointer;

          numerical_differentiation_pointer = NULL;
      }
      else
      {
            numerical_differentiation_pointer = new NumericalDifferentiation(*other_error_term.numerical_differentiation_pointer);
      }

      display = other_error_term.display;
   }

   return(*this);
}


// EQUAL TO OPERATOR

// bool operator == (const RegularizationTerm&) const method

/// Equal to operator. 
/// It compares this object to another object. 
/// The return is true if both objects have the same member data, and false otherwise. 

bool RegularizationTerm::operator == (const RegularizationTerm& other_error_term) const
{
   if(neural_network_pointer != other_error_term.neural_network_pointer)
   {
       return(false);
   }
/*
   else if((numerical_differentiation_pointer == NULL && other_error_term.numerical_differentiation_pointer !=NULL)
        || (numerical_differentiation_pointer != NULL && other_error_term.numerical_differentiation_pointer ==NULL))
   {
       return(false);
   }
   else if(numerical_differentiation_pointer != NULL)
   {
       if(&numerical_differentiation_pointer != &other_error_term.numerical_differentiation_pointer)
       {
            return(false);
       }
   }
*/
   else if(display != other_error_term.display)
   {
      return(false);
   }

   return(true);

}


// METHODS

// const bool& get_display(void) const method

/// Returns true if messages from this class can be displayed on the screen, or false if messages
/// from this class can't be displayed on the screen.

const bool& RegularizationTerm::get_display(void) const
{
   return(display);
}


// bool has_neural_network(void) const method

/// Returns true if this error term has a neural network associated,
/// and false otherwise.

bool RegularizationTerm::has_neural_network(void) const
{
    if(neural_network_pointer)
    {
        return(true);
    }
    else
    {
        return(false);
    }
}


// bool has_numerical_differentiation(void) const method

/// Returns true if this error term object contains a numerical differentiation object,
/// and false otherwise.

bool RegularizationTerm::has_numerical_differentiation(void) const
{
    if(numerical_differentiation_pointer)
    {
        return(true);
    }
    else
    {
        return(false);
    }
}


// void set(void) method

/// Sets all the member pointers to NULL (neural network, data set, mathematical model and numerical differentiation).
/// It also initializes all the rest of class members to their default values.

void RegularizationTerm::set(void)
{
   neural_network_pointer = NULL;
   numerical_differentiation_pointer = NULL;

   set_default();
}


// void set(NeuralNetwork*) method

/// Sets all the member pointers to NULL, but the neural network, which set to a given pointer.
/// It also initializes all the rest of class members to their default values.
/// @param new_neural_network_pointer Pointer to a neural network object.

void RegularizationTerm::set(NeuralNetwork* new_neural_network_pointer)
{
   neural_network_pointer = new_neural_network_pointer;
   numerical_differentiation_pointer = NULL;

   set_default();
}


// void set(const RegularizationTerm&) method

/// Sets to this error term object the members of another error term object.
/// @param other_error_term Error term to be copied. 

void RegularizationTerm::set(const RegularizationTerm& other_error_term)
{
   neural_network_pointer = other_error_term.neural_network_pointer;

   if(other_error_term.numerical_differentiation_pointer)
   {
      numerical_differentiation_pointer = new NumericalDifferentiation(*other_error_term.numerical_differentiation_pointer);
   }

   display = other_error_term.display;  
}


// void set_neural_network_pointer(NeuralNetwork*) method

/// Sets a pointer to a neural network object which is to be associated to the error term.
/// @param new_neural_network_pointer Pointer to a neural network object to be associated to the error term.

void RegularizationTerm::set_neural_network_pointer(NeuralNetwork* new_neural_network_pointer)
{
   neural_network_pointer = new_neural_network_pointer;
}


// void set_numerical_differentiation_pointer(NumericalDifferentiation*) method

/// Sets a new numerical differentiation pointer in this error term object.
/// @param new_numerical_differentiation_pointer Pointer to a numerical differentiation object. 

void RegularizationTerm::set_numerical_differentiation_pointer(NumericalDifferentiation* new_numerical_differentiation_pointer)
{
   numerical_differentiation_pointer = new_numerical_differentiation_pointer;
}


// void set_default(void) method

/// Sets the members of the error term to their default values:
/// <ul>
/// <li> Display: true.
/// </ul>

void RegularizationTerm::set_default(void)
{
   display = true;
}


// void set_display(const bool&) method

/// Sets a new display value.
/// If it is set to true messages from this class are to be displayed on the screen;
/// if it is set to false messages from this class are not to be displayed on the screen.
/// @param new_display Display value.

void RegularizationTerm::set_display(const bool& new_display)
{
   display = new_display;
}


// void construct_numerical_differentiation(void) method

/// This method constructs the numerical differentiation object which composes the error term class. 

void RegularizationTerm::construct_numerical_differentiation(void)
{
   if(numerical_differentiation_pointer == NULL)
   {
      numerical_differentiation_pointer = new NumericalDifferentiation();
   }
}


// void delete_numerical_differentiation_pointer(void) method

/// This method deletes the numerical differentiation object which composes the error term class. 

void RegularizationTerm::delete_numerical_differentiation_pointer(void)
{
   delete numerical_differentiation_pointer;

   numerical_differentiation_pointer = NULL;
}


// void check(void) const method

/// Checks that there is a neural network associated to the error term.
/// If some of the above conditions is not hold, the method throws an exception. 

void RegularizationTerm::check(void) const
{
   std::ostringstream buffer;

   if(!neural_network_pointer)
   {
      buffer << "OpenNN Exception: RegularizationTerm class.\n"
             << "void check(void) const.\n"
             << "Pointer to neural network is NULL.\n";

      throw std::logic_error(buffer.str());	  
   }
}


// Vector<double> calculate_gradient(void) const method

/// Returns the default gradient vector of the error term.
/// It uses numerical differentiation.

Vector<double> RegularizationTerm::calculate_gradient(void) const
{
   // Neural network stuff

   #ifdef __OPENNN_DEBUG__

   check();

   #endif

   // Loss index stuff

   #ifdef __OPENNN_DEBUG__

   std::ostringstream buffer;

   if(!numerical_differentiation_pointer)
   {
      buffer << "OpenNN Exception: RegularizationTerm class.\n"
             << "Vector<double> calculate_gradient(void) const method.\n"
             << "Numerical differentiation pointer is NULL.\n";

      throw std::logic_error(buffer.str());	  
   }

   #endif

   const Vector<double> parameters = neural_network_pointer->arrange_parameters();

   return(numerical_differentiation_pointer->calculate_gradient(*this, &RegularizationTerm::calculate_regularization, parameters));
}


// Vector<double> calculate_gradient(const Vector<double>&) const method

/// Returns the default gradient vector of the error term.
/// It uses numerical differentiation.

Vector<double> RegularizationTerm::calculate_gradient(const Vector<double>& parameters) const
{
   // Neural network stuff

   #ifdef __OPENNN_DEBUG__

   check();

   #endif

   // Loss index stuff

   #ifdef __OPENNN_DEBUG__

   std::ostringstream buffer;

   if(!numerical_differentiation_pointer)
   {
      buffer << "OpenNN Exception: RegularizationTerm class.\n"
             << "Vector<double> calculate_gradient(const Vector<double>&) const method.\n"
             << "Numerical differentiation pointer is NULL.\n";

      throw std::logic_error(buffer.str());
   }

   #endif

   return(numerical_differentiation_pointer->calculate_gradient(*this, &RegularizationTerm::calculate_regularization, parameters));
}


// Matrix<double> calculate_Hessian(void) const method

/// @todo

Matrix<double> RegularizationTerm::calculate_Hessian(void) const
{
   // Neural network stuff

   #ifdef __OPENNN_DEBUG__

   check();

   #endif

   const Vector<double> parameters = neural_network_pointer->arrange_parameters();

   return(numerical_differentiation_pointer->calculate_Hessian(*this, &RegularizationTerm::calculate_regularization, parameters));
}


// Matrix<double> calculate_Hessian(const Vector<double>&) const method

/// @todo

Matrix<double> RegularizationTerm::calculate_Hessian(const Vector<double>& parameters) const
{
   // Neural network stuff

   #ifdef __OPENNN_DEBUG__

   check();

   #endif

   return(numerical_differentiation_pointer->calculate_Hessian(*this, &RegularizationTerm::calculate_regularization, parameters));
}


// std::string write_error_term_type(void) const method

/// Returns a string with the default type of error term, "USER_PERFORMANCE_TERM".

std::string RegularizationTerm::write_error_term_type(void) const
{
   return("USER_REGULARIZATION_TERM");
}


// std::string write_information(void) const method

/// Returns a string with the default information of the error term.
/// It will be used by the training strategy to monitor the training process. 
/// By default this information is empty. 

std::string RegularizationTerm::write_information(void) const
{
   return("");
}


// std::string to_string(void) const method

/// Returns the default string representation of a error term.

std::string RegularizationTerm::to_string(void) const
{
   std::ostringstream buffer;

   buffer << "Error term\n";
          //<< "Display: " << display << "\n";

   return(buffer.str());
}


// tinyxml2::XMLDocument* to_XML(void) const method 

/// Serializes a default error term object into a XML document of the TinyXML library.
/// See the OpenNN manual for more information about the format of this document.

tinyxml2::XMLDocument* RegularizationTerm::to_XML(void) const
{
   std::ostringstream buffer;

   tinyxml2::XMLDocument* document = new tinyxml2::XMLDocument;

   // Error term

   tinyxml2::XMLElement* root_element = document->NewElement("RegularizationTerm");

   document->InsertFirstChild(root_element);

   return(document);
}


// void write_XML(tinyxml2::XMLPrinter&) const method

/// Serializes the regularization term object into a XML document of the TinyXML library without keep the DOM tree in memory.
/// See the OpenNN manual for more information about the format of this document.

void RegularizationTerm::write_XML(tinyxml2::XMLPrinter& file_stream) const
{
    file_stream.OpenElement("RegularizationTerm");

    file_stream.CloseElement();
}


// void from_XML(const tinyxml2::XMLDocument&) method

/// Loads a default error term from a XML document.
/// @param document TinyXML document containing the error term members.

void RegularizationTerm::from_XML(const tinyxml2::XMLDocument& document)
{
   // Display warnings

   const tinyxml2::XMLElement* display_element = document.FirstChildElement("Display");

   if(display_element)
   {
      std::string new_display_string = display_element->GetText();           

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


// size_t calculate_Kronecker_delta(const size_t&, const size_t&) const method

/// Returns the Knronecker delta of two integers a and b, which equals 1 if they are equal and 0 otherwise.
/// @param a First integer.
/// @param b Second integer.

size_t RegularizationTerm::calculate_Kronecker_delta(const size_t& a, const size_t& b) const
{
   if(a == b)
   {
      return(1);
   }
   else
   {
      return(0);
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
