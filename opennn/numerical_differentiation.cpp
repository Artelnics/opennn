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

NumericalDifferentiation::NumericalDifferentiation()
{
    set_default();
}


const Index& NumericalDifferentiation::get_precision_digits() const
{
    return precision_digits;
}


const bool& NumericalDifferentiation::get_display() const
{
    return display;
}


void NumericalDifferentiation::set_display(const bool& new_display)
{
    display = new_display;
}


void NumericalDifferentiation::set_precision_digits(const Index& new_precision_digits)
{
    precision_digits = new_precision_digits;
}


void NumericalDifferentiation::set_default()
{
    precision_digits = 6;

    display = true;
}


type NumericalDifferentiation::calculate_eta() const
{
    return pow(type(10.0), type(-1.0)*type(precision_digits));
}


type NumericalDifferentiation::calculate_h(const type& x) const
{
    const type eta = calculate_eta();

    return sqrt(eta)*(type(1) + abs(x));
}


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
// Copyright(C) 2005-2024 Artificial Intelligence Techniques, SL.
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
