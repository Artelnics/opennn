//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   B O U N D I N G   L A Y E R   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#pragma once

#include "layer.h"
#include "operators.h"

namespace opennn
{

class Bounding final : public Layer
{
public:

    using BoundingMethod = Bound::Method;

    Bounding(const Shape& = {0}, const string& = "bounding_layer");

    Shape get_input_shape() const override { return output_shape; }
    Shape get_output_shape() const override;

    const BoundingMethod& get_bounding_method() const { return bound.method; }

    VectorR get_lower_bounds() const;
    VectorR get_upper_bounds() const;

    void set(const Shape& = {0}, const string& = "bounding_layer");

    void set_input_shape(const Shape&) override;
    void set_output_shape(const Shape&) override;

    void set_bounding_method(const BoundingMethod&);
    void set_bounding_method(const string&);

    void set_lower_bounds(const VectorR&);
    void set_lower_bound(Index, float);

    void set_upper_bounds(const VectorR&);
    void set_upper_bound(Index, float);

    void read_JSON_body(const Json*) override;
    void write_JSON_body(JsonWriter&) const override;

private:

    Shape output_shape;

    Bound bound;

    static const EnumMap<BoundingMethod>& bounding_method_map();
};

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
