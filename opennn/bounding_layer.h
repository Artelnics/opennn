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

    Bounding(const Shape& = {0}, const string& = "bounding_layer");

    enum class BoundingMethod{NoBounding, Bounding};

    Shape get_input_shape() const override { return output_shape; }

    Shape get_output_shape() const override;

    vector<Shape> get_forward_shapes(const Index batch_size) const override
    {
        return {Shape{batch_size}.append(get_output_shape())};
    }

    const BoundingMethod& get_bounding_method() const;

    // Return by value — zero-copy would require VectorMap; VectorR copy is cheap here
    // (bounds are output-sized, typically small) and lets const& callers bind safely.
    VectorR get_lower_bounds() const;
    VectorR get_upper_bounds() const;

    enum States {Lower, Upper};

    vector<Shape> get_state_shapes() const override
    {
        if(bounding_method == BoundingMethod::NoBounding || output_shape.empty() || output_shape[0] == 0)
            return {};
        return {Shape{output_shape[0]}, Shape{output_shape[0]}};
    }

    type* link_states(type* pointer) override;

    void set(const Shape& = { 0 }, const string & = "bounding_layer");

    void set_input_shape(const Shape&) override;
    void set_output_shape(const Shape&) override;

    void set_bounding_method(const BoundingMethod&);
    void set_bounding_method(const string&);

    // Setters require the layer to be compiled first (states arena must be allocated).
    void set_lower_bounds(const VectorR&);
    void set_lower_bound(const Index, type);

    void set_upper_bounds(const VectorR&);
    void set_upper_bound(const Index, type);

    // Forward

    void forward_propagate(ForwardPropagation&, size_t, bool) noexcept override;

    // Serialization (two-phase: from_XML parses config; load_state_from_XML parses bounds after compile)

    void from_XML(const XmlDocument&) override;

    void load_state_from_XML(const XmlDocument&) override;

    void to_XML(XmlPrinter&) const override;

private:

    enum Forward {Input, Output};

    Shape output_shape;

    BoundingMethod bounding_method = BoundingMethod::Bounding;
};

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
