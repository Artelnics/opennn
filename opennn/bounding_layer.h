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

    const VectorR& get_lower_bounds() const;
    const VectorR& get_upper_bounds() const;

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

    void set_lower_bounds(const VectorR&);
    void set_lower_bound(const Index, type);

    void set_upper_bounds(const VectorR&);
    void set_upper_bound(const Index, type);

    // Forward

    void forward_propagate(ForwardPropagation&, size_t, bool) noexcept override;

    // Serialization

    void from_XML(const XmlDocument&) override;

    void to_XML(XmlPrinter&) const override;

private:

    enum Forward {Input, Output};

    Shape output_shape;

    BoundingMethod bounding_method = BoundingMethod::Bounding;

    // @todo Remove these VectorR. They exist only as pre-compile staging: NN::from_XML
    // calls Bounding::from_XML before NN::compile() runs link_states(), so states[]
    // is still empty when bounds are parsed. link_states() flushes these into
    // states[Lower]/states[Upper] and they become dead weight afterwards. To remove
    // them, NN::from_XML needs a two-phase parse (collect raw values, run compile(),
    // then let layers populate their states via a post-compile hook).
    VectorR lower_bounds;
    VectorR upper_bounds;
};

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
