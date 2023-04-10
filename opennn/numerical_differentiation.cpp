//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   N U M E R I C A L   D I F F E R E N T I A T I O N   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "numerical_differentiation.h"

namespace opennn
{

/// Default constructor.
/// It creates a numerical differentiation object with the default members.

NumericalDifferentiation::NumericalDifferentiation()
{
    set_default();
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
    precision_digits = 6;

    display = true;
}


type NumericalDifferentiation::calculate_eta() const
{
    return pow(static_cast<type>(10.0), static_cast<type>(-1.0)*type(precision_digits));
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
        h(i) = sqrt(eta)*(type(1) + abs(x(i)));
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
        h(i) = sqrt(eta)*(type(1) + y(i));
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
        h(i) = sqrt(eta)*(type(1) + y(i));
    }

    return h;
}

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2022 Artificial Intelligence Techniques, SL.
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
