//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   N U M E R I C A L   D I F F E R E N T I A T I O N   C L A S S         
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "numerical_differentiation.h"

namespace OpenNN
{

/// Default constructor. 
/// It creates a numerical differentiation object with the default members. 

NumericalDifferentiation::NumericalDifferentiation()
{ 
   set_default();
}


/// Copy constructor. 

NumericalDifferentiation::NumericalDifferentiation(const NumericalDifferentiation& other_numerical_differentiation)
{ 
   set(other_numerical_differentiation);
}


/// Destructor.

NumericalDifferentiation::~NumericalDifferentiation()
{

}


/// Returns the method used for numerical differentiation(forward differences or central differences).

const NumericalDifferentiation::NumericalDifferentiationMethod& NumericalDifferentiation::get_numerical_differentiation_method() const
{
   return(numerical_differentiation_method);                           
}


/// Returns a string with the name of the method to be used for numerical differentiation. 

string NumericalDifferentiation::write_numerical_differentiation_method() const
{
   switch(numerical_differentiation_method)
   {
      case ForwardDifferences:
      {
         return "ForwardDifferences";
	  }

      case CentralDifferences:
      {
         return "CentralDifferences";
 	  }
   }

   return string();
}


/// Returns the number of precision digits required for the derivatives. 

const size_t& NumericalDifferentiation::get_precision_digits() const
{
   return(precision_digits);
}


/// Returns the flag used by this class for displaying or not displaying warnings.

const bool& NumericalDifferentiation::get_display() const
{
   return display;
}


/// Sets the members of this object to be equal to those of another object. 
/// @param other_numerical_differentiation Numerical differentiation object to be copied. 

void NumericalDifferentiation::set(const NumericalDifferentiation& other_numerical_differentiation)
{
   numerical_differentiation_method = other_numerical_differentiation.numerical_differentiation_method;

   precision_digits = other_numerical_differentiation.precision_digits;

   display = other_numerical_differentiation.display;

}


/// Sets the method to be used for numerical differentiation(forward differences or central differences).
/// @param new_numerical_differentiation_method New numerical differentiation method.

void NumericalDifferentiation::set_numerical_differentiation_method
(const NumericalDifferentiation::NumericalDifferentiationMethod& new_numerical_differentiation_method)
{
   numerical_differentiation_method = new_numerical_differentiation_method;
}


/// Sets the method to be used for the numerical differentiation.
/// The argument is a string with the name of the numerical differentiation method. 
/// @param new_numerical_differentiation_method Numerical differentiation method name string.

void NumericalDifferentiation::set_numerical_differentiation_method(const string& new_numerical_differentiation_method)
{
   if(new_numerical_differentiation_method == "ForwardDifferences")
   {
      numerical_differentiation_method = ForwardDifferences;
   }
   else if(new_numerical_differentiation_method == "CentralDifferences")
   {
      numerical_differentiation_method = CentralDifferences;
   }
   else
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: NumericalDifferentiation class.\n"
             << "void set_numerical_differentiation_method(const string&) method.\n"
			 << "Unknown numerical differentiation method: " << new_numerical_differentiation_method << ".\n";

      throw logic_error(buffer.str());
   }	
}


/// Sets a new flag for displaying warnings from this class or not. 
/// @param new_display Display flag. 

void NumericalDifferentiation::set_display(const bool& new_display)
{
   display = new_display;
}


/// Sets a new number of precision digits required for the derivatives. 
/// @param new_precision_digits Number of precision digits. 

void NumericalDifferentiation::set_precision_digits(const size_t& new_precision_digits)
{
   precision_digits = new_precision_digits;
}


/// Sets default values for the members of this object:
/// <ul> 
/// <li> Numerical differentiation method: Central differences.
/// <li> Precision digits: 6.
/// <li> Display: true.
/// </ul>

void NumericalDifferentiation::set_default()
{
   numerical_differentiation_method = CentralDifferences;

   precision_digits = 6;

   display = true;
}


/// Calculates a proper step size for computing the derivatives, as a function of the inputs point value. 
/// @param x Input value. 

double NumericalDifferentiation::calculate_h(const double& x) const
{
   const double eta = pow(10.0,-1*static_cast<int>(precision_digits));

   return(sqrt(eta)*(1.0 + abs(x)));
}


/// Calculates a vector of step sizes for computing the derivatives, as a function of a vector of inputs. 
/// @param x Input vector. 

Vector<double> NumericalDifferentiation::calculate_h(const Vector<double>& x) const
{
   const double eta = pow(10.0,-1*static_cast<int>(precision_digits));

   const size_t n = x.size();

   Vector<double> h(n);

   for(size_t i = 0; i < n; i++)
   {
      h[i] = sqrt(eta)*(1.0 + abs(x[i]));
   }
 
   return(h);
}


/// Calculates a tensor of step sizes for computing the derivatives, as a function of a vector of inputs.
/// @param x Input tensor.

Tensor<double> NumericalDifferentiation::calculate_h(const Tensor<double>& x) const
{
   const double eta = pow(10.0,-1*static_cast<int>(precision_digits));

   const size_t n = x.size();

   const Vector<size_t> dimensions = x.get_dimensions();

   Tensor<double> h(dimensions);

   for(size_t i = 0; i < n; i++)
   {
      h[i] = sqrt(eta)*(1.0 + abs(x[i]));
   }

   return(h);
}


Vector<double> NumericalDifferentiation::calculate_backward_differences_derivatives(const Vector<double>& x,
                                                                                    const Vector<double>& y) const
{
    

    #ifdef __OPENNN_DEBUG__

    const size_t x_size = x.size();
    const size_t y_size = y.size();

    if(x_size != y_size)
    {
       ostringstream buffer;

       buffer << "OpenNN Exception: NumericalDifferentiation class.\n"
              << "Vector<double> calculate_backward_differences_derivatives(const Vector<double>&, const Vector<double>&) const method.\n"
              << "Size of independent variable must be equal to size of dependent variable.\n";

       throw logic_error(buffer.str());
    }

    #endif

    const size_t size = x.size();

    Vector<double> derivatives(size);
    derivatives[0] = 0.0;

    for(size_t i = 1; i < size; i++)
    {
        const double numerator = y[i] - y[i-1];
        const double denominator = x[i] - x[i-1];

        if(denominator != 0.0)
        {
            derivatives[i] = numerator/denominator;
        }
        else
        {
            ostringstream buffer;

            buffer << "OpenNN Exception: NumericalDifferentiation class.\n"
                   << "Vector<double> calculate_backward_differences_derivatives(const Vector<double>&, const Vector<double>&) const method.\n"
                   << "Denominator is equal to 0.\n";

            throw logic_error(buffer.str());
        }
    }

    return derivatives;
}


/// Serializes this numerical differentiation object into a XML document-> 

tinyxml2::XMLDocument* NumericalDifferentiation::to_XML() const  
{
   tinyxml2::XMLDocument* document = new tinyxml2::XMLDocument;

   tinyxml2::XMLElement* element = nullptr;
   tinyxml2::XMLText* text = nullptr;

   ostringstream buffer;

   // Numerical differentiation

   tinyxml2::XMLElement* root_element = document->NewElement("NumericalDifferentiation");

   document->InsertFirstChild(root_element);

   // Numerical differentiation method
   
   element = document->NewElement("NumericalDifferentiationMethod");
   root_element->LinkEndChild(element);

   text = document->NewText(write_numerical_differentiation_method().c_str());
   element->LinkEndChild(text);

   // Precision digits
   
   element = document->NewElement("PrecisionDigits");
   root_element->LinkEndChild(element);

   buffer.str("");
   buffer << precision_digits;

   text = document->NewText(buffer.str().c_str());
   element->LinkEndChild(text);
   
   // Display
   
   element = document->NewElement("Display");
   root_element->LinkEndChild(element);

   buffer.str("");
   buffer << display;

   text = document->NewText(buffer.str().c_str());
   element->LinkEndChild(text);

   return document;
}


/// Serializes the numerical differentiation object into a XML document of the TinyXML library without keep the DOM tree in memory.
/// See the OpenNN manual for more information about the format of this document.

void NumericalDifferentiation::write_XML(tinyxml2::XMLPrinter&) const
{

}


/// Deserializes the object from a XML document.
/// @param document TinyXML document with the member data.

void NumericalDifferentiation::from_XML(const tinyxml2::XMLDocument& document)
{
    const tinyxml2::XMLElement* root_element = document.FirstChildElement("NumericalDifferentiation");

    if(!root_element)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: NumericalDifferentiation class.\n"
               << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
               << "Numerical differentiation element is nullptr.\n";

        throw logic_error(buffer.str());
    }

  // Numerical differentiation method
  {
     const tinyxml2::XMLElement* element = root_element->FirstChildElement("NumericalDifferentiationMethod");

     if(element)
     {
        const string new_numerical_differentiation_method = element->GetText();

        try
        {
           set_numerical_differentiation_method(new_numerical_differentiation_method);
        }
        catch(const logic_error& e)
        {
           cerr << e.what() << endl;
        }
     }
  }

  // Precision digits
  {
     const tinyxml2::XMLElement* element = root_element->FirstChildElement("PrecisionDigits");

     if(element)
     {
        const size_t new_precision_digits = static_cast<size_t>(atoi(element->GetText()));

        try
        {
           set_precision_digits(new_precision_digits);
        }
        catch(const logic_error& e)
        {
           cerr << e.what() << endl;
        }
     }
  }


  // Display
  {
     const tinyxml2::XMLElement* element = root_element->FirstChildElement("Display");

     if(element)
     {
        const string new_display = element->GetText();

        try
        {
           set_display(new_display != "0");
        }
        catch(const logic_error& e)
        {
           cerr << e.what() << endl;
        }
     }
  }
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
