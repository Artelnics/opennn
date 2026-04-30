//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   A D D I T I O N   L A Y E R   C L A S S   H E A D E R
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

class Addition final : public Layer
{

public:

    Addition(const Shape& new_input_shape = {}, const string& new_name = "")
    {
        set(new_input_shape, new_name);
    }

    Shape get_input_shape() const override { return input_shape; }

    Shape get_output_shape() const override { return input_shape; }

    vector<pair<Shape, Type>> get_forward_specs(Index batch_size) const override
    {
        return {{Shape{batch_size}.append(input_shape), activation_dtype}};
    }

    vector<pair<Shape, Type>> get_backward_specs(Index batch_size) const override
    {
        const Type act = activation_dtype;
        return {
            /*InputDelta0*/ {Shape{batch_size}.append(input_shape), act},
            /*InputDelta1*/ {Shape{batch_size}.append(input_shape), act},
        };
    }

    void set(const Shape& new_input_shape, const string& new_label)
    {
        input_shape = new_input_shape;
        label = new_label;
        name = "Addition";
        layer_type = LayerType::Addition;

        if (!input_shape.empty() && input_shape.rank != 2 && input_shape.rank != 3)
            throw runtime_error("Addition layer supports input rank 2 or 3 (got "
                                + to_string(input_shape.rank) + ").");
    }

    void forward_propagate(ForwardPropagation& forward_propagation, size_t layer, bool) noexcept override
    {
        auto& forward_views = forward_propagation.views[layer];

        addition(forward_views[Input][0], forward_views[Input][1], forward_views[Output][0]);
    }

    void back_propagate(ForwardPropagation&,
                        BackPropagation& back_propagation,
                        size_t layer) const noexcept override
    {
        auto& delta_views = back_propagation.delta_views[layer];

        copy(delta_views[OutputDelta][0], delta_views[InputDelta0][0]);
        copy(delta_views[OutputDelta][0], delta_views[InputDelta1][0]);
    }

    void from_XML(const XmlDocument& document) override
    {
        const XmlElement* element = document.first_child_element("Addition");
        if (!element) throw runtime_error(name + " element is nullptr.");

        const string new_label = read_xml_string(element, "Label");
        const Shape new_input_shape = string_to_shape(read_xml_string(element, "InputDimensions"));

        set(new_input_shape, new_label);
    }

    void to_XML(XmlPrinter& printer) const override
    {
        printer.open_element("Addition");

        write_xml(printer, {
            {"Label", label},
            {"InputDimensions", shape_to_string(input_shape)}
        });

        printer.close_element();
    }

    void set_input_shape(const Shape& shape) override { set(shape, label); }

private:

    Shape input_shape;

    enum Forward {Input, Output};
    enum Backward {OutputDelta, InputDelta0, InputDelta1};
};

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
