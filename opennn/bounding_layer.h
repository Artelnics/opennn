//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   B O U N D I N G   L A Y E R   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#pragma once

#include "layer.h"
#include "forward_propagation.h"
#include "back_propagation.h"

namespace opennn
{

class Bounding final : public Layer
{
public:

    enum class BoundingMethod {NoBounding, Bounding};

private:

    BoundingMethod bounding_method = BoundingMethod::Bounding;

    VectorR lower_bounds;
    VectorR upper_bounds;

    enum Forward {Inputs = 0, Outputs = 1};

    vector<Shape> get_forward_shapes(const Index batch_size) const override
    {
        return {Shape{batch_size}.append(get_output_shape())};
    }

public:

    Bounding(const Shape& = {0}, const string& = "bounding_layer");

    // Getters

    Shape get_output_shape() const override;

    const BoundingMethod& get_bounding_method() const;

    const VectorR& get_lower_bounds() const;
    const VectorR& get_upper_bounds() const;

    // Setters

    void set(const Shape& = {0}, const string& = "bounding_layer");

    void set_input_shape(const Shape&) override;
    void set_output_shape(const Shape&) override;

    void set_bounding_method(const BoundingMethod&);
    void set_bounding_method(const string&);

    void set_lower_bounds(const VectorR&);
    void set_lower_bound(const Index, type);

    void set_upper_bounds(const VectorR&);
    void set_upper_bound(const Index, type);

    // Forward propagation

    void forward_propagate(ForwardPropagation&, size_t, bool) noexcept override;

    // Serialization

    void from_XML(const XmlDocument&) override;
    void to_XML(XmlPrinter&) const override;
};

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
