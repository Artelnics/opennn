//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   F L A T T E N   L A Y E R   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#pragma once

#include "layer.h"
#include "math_utilities.h"
#include "forward_propagation.h"
#include "back_propagation.h"

namespace opennn
{

class Flatten final : public Layer
{

public:

    Flatten(const Shape& new_input_shape = {})
    {
        set(new_input_shape);
    }

    Shape get_input_shape() const override { return input_shape; }

    Shape get_output_shape() const override { return { input_shape.size() }; }

    void set(const Shape& new_input_shape)
    {
        input_shape = new_input_shape;
        set_label("flatten_layer");
        name = "Flatten";
        layer_type = LayerType::Flatten;

        if (!input_shape.empty()
            && input_shape.rank != 1 && input_shape.rank != 2 && input_shape.rank != 3)
            throw runtime_error("Flatten layer supports input rank 1, 2 or 3 (got "
                                + to_string(input_shape.rank) + ").");
    }

    void set_input_shape(const Shape& new_input_shape) override
    {
        set(new_input_shape);
    }

    vector<pair<Shape, Type>> get_forward_specs(const Index batch_size) const override
    {
        return {{Shape{batch_size}.append(get_output_shape()), activation_dtype}};
    }

    vector<pair<Shape, Type>> get_backward_specs(Index batch_size) const override
    {
        return {{Shape{batch_size}.append(input_shape), activation_dtype}};
    }

    void forward_propagate(ForwardPropagation& forward_propagation, size_t layer, bool) noexcept override
    {
        auto& forward_views = forward_propagation.views[layer];

        copy(forward_views[Input][0], forward_views[Output][0]);
    }

    void back_propagate(ForwardPropagation&,
                        BackPropagation& back_propagation,
                        size_t layer) const noexcept override
    {
        auto& delta_views = back_propagation.delta_views[layer];

        copy(delta_views[OutputDelta][0], delta_views[InputDelta][0]);
    }

    void from_XML(const XmlDocument& document) override
    {
        const XmlElement* element = document.first_child_element("Flatten");
        if (!element) throw runtime_error(name + " element is nullptr.");

        set(string_to_shape(read_xml_string(element, "InputDimensions")));
    }

    void to_XML(XmlPrinter& printer) const override
    {
        printer.open_element("Flatten");

        write_xml(printer, {
            {"InputDimensions", shape_to_string(input_shape)}
        });

        printer.close_element();
    }

private:

    Shape input_shape;

    enum Forward {Input, Output};
    enum Backward {OutputDelta, InputDelta};
};

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
