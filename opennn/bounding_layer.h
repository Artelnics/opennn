//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   B O U N D I N G   L A Y E R   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#pragma once

#include "layer.h"

namespace opennn
{

class Bounding final : public Layer
{

public:

    Bounding(const Shape& = {0}, const string& = "bounding_layer");

    enum class BoundingMethod{NoBounding, Bounding};

    Shape get_input_shape() const override;
    Shape get_output_shape() const override;

    const BoundingMethod& get_bounding_method() const;

    string get_bounding_method_string() const;

    const VectorR& get_lower_bounds() const;
    type get_lower_bound(const Index) const;

    const VectorR& get_upper_bounds() const;
    type get_upper_bound(const Index) const;

    void set(const Shape& = { 0 }, const string & = "bounding_layer");

    void set_input_shape(const Shape&) override;
    void set_output_shape(const Shape&) override;

    void set_bounding_method(const BoundingMethod&);
    void set_bounding_method(const string&);

    void set_lower_bounds(const VectorR&);
    void set_lower_bound(const Index, type);

    void set_upper_bounds(const VectorR&);
    void set_upper_bound(const Index, type);

    // Lower and upper bounds

    void forward_propagate(const vector<TensorView>&,
                           unique_ptr<LayerForwardPropagation>&,
                           bool) override;

#ifdef OPENNN_CUDA
    void forward_propagate(const vector<TensorViewCuda>&,
                                unique_ptr<LayerForwardPropagationCuda>&,
                                bool) override;
#endif

    // Expression

    string get_expression(const vector<string>& = vector<string>(), const vector<string>& = vector<string>()) const override;

    // Serialization

    void print() const override;

    void from_XML(const XMLDocument&) override;

    void to_XML(XMLPrinter&) const override;

private:

    BoundingMethod bounding_method = BoundingMethod::Bounding;

    VectorR lower_bounds;

    VectorR upper_bounds;
};


struct BoundingForwardPropagation final : LayerForwardPropagation
{
    BoundingForwardPropagation(const Index = 0, Layer* = nullptr);

    void initialize() override;

    void print() const override;
};


#ifdef OPENNN_CUDA

struct BoundingForwardPropagationCuda final : public LayerForwardPropagationCuda
{
    BoundingForwardPropagationCuda(const Index = 0, Layer* = nullptr);

    void initialize() override;

    void print() const override;
};

#endif

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// This library is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 2.1 of the License, or any later version.
// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.
// You should have received a copy of the GNU Lesser General Public
// License along with this library; if not, write to the Free Software
// Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
