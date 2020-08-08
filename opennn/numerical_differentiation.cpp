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


/// Destructor.

NumericalDifferentiation::~NumericalDifferentiation()
{

}


/// Returns the method used for numerical differentiation(forward differences or central differences).

const NumericalDifferentiation::NumericalDifferentiationMethod& NumericalDifferentiation::get_numerical_differentiation_method() const
{
    return numerical_differentiation_method ;
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

const Index& NumericalDifferentiation::get_precision_digits() const
{
    return precision_digits;
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

void NumericalDifferentiation:: set_numerical_differentiation_method(const string& new_numerical_differentiation_method)
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

void NumericalDifferentiation::set_precision_digits(const Index& new_precision_digits)
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


type NumericalDifferentiation::calculate_eta() const
{
    return pow(static_cast<type>(10.0), static_cast<type>(-1.0*precision_digits));
}


/// Calculates a proper step size for computing the derivatives, as a function of the inputs point value.
/// @param x Input value.

type NumericalDifferentiation::calculate_h(const type& x) const
{
    const type eta = calculate_eta();

    return sqrt(eta)*(static_cast<type>(1.0) + abs(x));
}


/// Calculates a vector of step sizes for computing the derivatives, as a function of a vector of inputs.
/// @param x Input vector.

Tensor<type, 1> NumericalDifferentiation::calculate_h(const Tensor<type, 1>& x) const
{
    const type eta = calculate_eta();

    const Index n = x.size();

    Tensor<type, 1> h(n);

    for(Index i = 0; i < n; i++)
    {
        h(i) = sqrt(eta)*(1 + abs(x(i)));
    }

    return h;
}


/// Calculates a tensor of step sizes for computing the derivatives, as a function of a vector of inputs.
/// @param x Input tensor.

Tensor<type, 2> NumericalDifferentiation::calculate_h(const Tensor<type, 2>& x) const
{
    const type eta = calculate_eta();

    const Index n = x.size();

    const auto& dimensions = x.dimensions();

    Tensor<type, 2> h(dimensions);

    Tensor<type, 2> y = x.abs();

    for(Index i = 0; i < n; i++)
    {
        h(i) = sqrt(eta)*(1 + y(i));
    }

    return h;
}


Tensor<type, 4> NumericalDifferentiation::calculate_h(const Tensor<type, 4>& x) const
{
    const type eta = calculate_eta();

    const Index n = x.size();

    const auto& dimensions = x.dimensions();

    Tensor<type, 4> h(dimensions);

    Tensor<type, 4> y = x.abs();

    for(Index i = 0; i < n; i++)
    {
        h(i) = sqrt(eta)*(1 + y(i));
    }

    return h;
}


Tensor<type, 1> NumericalDifferentiation::calculate_backward_differences_derivatives(const Tensor<type, 1>& x,
        const Tensor<type, 1>& y) const
{
#ifdef __OPENNN_DEBUG__

    const Index x_size = x.size();
    const Index y_size = y.size();

    if(x_size != y_size)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: NumericalDifferentiation class.\n"
               << "Tensor<type, 1> calculate_backward_differences_derivatives(const Tensor<type, 1>&, const Tensor<type, 1>&) const method.\n"
               << "Size of independent variable must be equal to size of dependent variable.\n";

        throw logic_error(buffer.str());
    }

#endif

    const Index size = x.size();

    Tensor<type, 1> derivatives(size);
    derivatives[0] = 0;

    for(Index i = 1; i < size; i++)
    {
        const type numerator = y(i) - y[i-1];
        const type denominator = x(i) - x[i-1];

        if(abs(denominator) > numeric_limits<float>::min())
        {
            derivatives(i) = numerator/denominator;
        }
        else
        {
            ostringstream buffer;

            buffer << "OpenNN Exception: NumericalDifferentiation class.\n"
                   << "Tensor<type, 1> calculate_backward_differences_derivatives(const Tensor<type, 1>&, const Tensor<type, 1>&) const method.\n"
                   << "Denominator is equal to 0.\n";

            throw logic_error(buffer.str());
        }
    }

    return derivatives;
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
            const Index new_precision_digits = static_cast<Index>(atoi(element->GetText()));

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
