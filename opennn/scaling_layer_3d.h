//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   S C A L I N G   L A Y E R   3 D   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#ifndef SCALINGLAYER3D_H
#define SCALINGLAYER3D_H

#include "layer.h"
#include "statistics.h"
#include "scaling.h"

namespace opennn
{

class Scaling3d final : public Layer
{

public:

    Scaling3d(const dimensions& = {0, 0});

    dimensions get_input_dimensions() const override;
    dimensions get_output_dimensions() const override;

    vector<Descriptives> get_descriptives() const;
    Descriptives get_descriptives(const Index&) const;

    Tensor<type, 1> get_minimums() const;
    Tensor<type, 1> get_maximums() const;
    Tensor<type, 1> get_means() const;
    Tensor<type, 1> get_standard_deviations() const;

    vector<string> get_scaling_methods() const;

    void set(const dimensions& = {0, 0});

    void set_input_dimensions(const dimensions&) override;
    void set_output_dimensions(const dimensions&) override;

    void set_descriptives(const vector<Descriptives>&);

    void set_min_max_range(const type& min, const type& max);

    void set_scalers(const vector<string>&);
    void set_scalers(const string&);

    void forward_propagate(const vector<TensorView>&,
                           unique_ptr<LayerForwardPropagation>&,
                           const bool&) override;

    Tensor<type, 3> calculate_outputs(const Tensor<type, 3>& inputs) const;

    string get_expression(const vector<string>& = vector<string>(), const vector<string>& = vector<string>()) const override;

    void print() const override;

    void from_XML(const XMLDocument&) override;
    void to_XML(XMLPrinter&) const override;

private:

    dimensions input_dimensions;

    vector<Descriptives> descriptives;

    vector<string> scalers;

    type min_range;
    type max_range;
};


struct Scaling3dForwardPropagation final : LayerForwardPropagation
{
    Scaling3dForwardPropagation(const Index& = 0, Layer* = nullptr);

    TensorView get_output_pair() const override;

    void set(const Index& = 0, Layer* = nullptr) override;

    void print() const override;

    Tensor<type, 3> outputs;
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
