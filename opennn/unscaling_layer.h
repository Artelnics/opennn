//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   U N S C A L I N G   L A Y E R   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#pragma once

#include "layer.h"
#include "scaling.h"

namespace opennn
{

class Unscaling final : public Layer
{

public:

    Unscaling(const Shape& = {0}, const string& = "unscaling_layer");

    Shape get_input_shape() const override;
    Shape get_output_shape() const override;

    vector<Descriptives> get_descriptives() const;

    VectorR get_minimums() const;
    VectorR get_maximums() const;

    vector<string> get_scalers() const;

    void set(const Index = 0, const string& = "unscaling_layer");

    void set_input_shape(const Shape&) override;
    void set_output_shape(const Shape&) override;

    void set_descriptives(const vector<Descriptives>&);

    void set_min_max_range(const type, const type);

    void set_scalers(const vector<string>&);
    void set_scalers(const string&);

    void forward_propagate(const vector<TensorView>&,
                           unique_ptr<LayerForwardPropagation>&,
                           bool) override;
#ifdef OPENNN_CUDA
    void forward_propagate(const vector<TensorViewCuda>&,
                           unique_ptr<LayerForwardPropagationCuda>&,
                           bool) override;
#endif

    void print() const override;

    void from_XML(const XMLDocument&) override;
    void to_XML(XMLPrinter&) const override;

    string get_expression(const vector<string>& = vector<string>(), const vector<string>& = vector<string>()) const override;

private:

    vector<Descriptives> descriptives;

    vector<string> scalers;

    type min_range;
    type max_range;
};


struct UnscalingForwardPropagation final : LayerForwardPropagation
{
    UnscalingForwardPropagation(const Index = 0, Layer* = nullptr);

    void initialize() override;

    void print() const override;
};


#ifdef OPENNN_CUDA

struct UnscalingForwardPropagationCuda final : public LayerForwardPropagationCuda
{
    UnscalingForwardPropagationCuda(const Index = 0, Layer* = nullptr);

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
