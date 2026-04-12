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

    vector<Shape> get_forward_shapes(const Index batch_size) const override
    {
        return {Shape{batch_size}.append(get_output_shape())};
    }

    const VectorR& get_minimums() const { return minimums; }
    const VectorR& get_maximums() const { return maximums; }

    void set(const Index = 0, const string& = "unscaling_layer");

    void set_input_shape(const Shape&) override;
    void set_output_shape(const Shape&) override;

    void set_descriptives(const vector<Descriptives>&);

    void set_min_max_range(const type, const type);

    void set_scalers(const vector<string>&);
    void set_scalers(const string&);

    void forward_propagate(ForwardPropagation&, size_t, bool) override;

    void calculate_coefficients();

    void print() const override;

    void from_XML(const XmlDocument&) override;
    void to_XML(XmlPrinter&) const override;

    string get_expression(const vector<string>& = vector<string>(), const vector<string>& = vector<string>()) const override;

private:

    VectorR means;
    VectorR standard_deviations;
    VectorR minimums;
    VectorR maximums;

    VectorR multipliers;
    VectorR offsets;

    vector<string> scalers;

    type min_range = -1.0f;
    type max_range = 1.0f;
};

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
