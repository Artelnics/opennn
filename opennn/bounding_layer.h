//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   B O U N D I N G   L A Y E R   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#ifndef BOUNDINGLAYER_H
#define BOUNDINGLAYER_H

#include "layer.h"

namespace opennn
{

class Bounding final : public Layer
{

public:

    Bounding(const dimensions& = {0}, const string& = "bounding_layer");

    enum class BoundingMethod{NoBounding, Bounding};

    dimensions get_input_dimensions() const override;
    dimensions get_output_dimensions() const override;

    const BoundingMethod& get_bounding_method() const;

    string get_bounding_method_string() const;

    const Tensor<type, 1>& get_lower_bounds() const;
    type get_lower_bound(const Index&) const;

    const Tensor<type, 1>& get_upper_bounds() const;
    type get_upper_bound(const Index&) const;

    void set(const dimensions & = { 0 }, const string & = "bounding_layer");

    void set_input_dimensions(const dimensions&) override;
    void set_output_dimensions(const dimensions&) override;

    void set_bounding_method(const BoundingMethod&);
    void set_bounding_method(const string&);

    void set_lower_bounds(const Tensor<type, 1>&);
    void set_lower_bound(const Index&, const type&);

    void set_upper_bounds(const Tensor<type, 1>&);
    void set_upper_bound(const Index&, const type&);

    // Lower and upper bounds

    void forward_propagate(const vector<TensorView>&,
                           unique_ptr<LayerForwardPropagation>&,
                           const bool&) override;

    // Expression

    string get_expression(const vector<string>& = vector<string>(), const vector<string>& = vector<string>()) const override;

    // Serialization

    void print() const override;

    void from_XML(const XMLDocument&) override;

    void to_XML(XMLPrinter&) const override;

private:

    BoundingMethod bounding_method = BoundingMethod::Bounding;

    Tensor<type, 1> lower_bounds;

    Tensor<type, 1> upper_bounds;
};


struct BoundingForwardPropagation final : LayerForwardPropagation
{
    BoundingForwardPropagation(const Index& = 0, Layer* = nullptr);

    TensorView get_output_pair() const override;

    void set(const Index& = 0, Layer* = nullptr) override;

    void print() const override;

    Tensor<type, 2> outputs;
};

}

#endif


// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2025 Artificial Intelligence Techniques, SL.
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
