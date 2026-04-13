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

template<int Rank>
class Addition final : public Layer
{

public:

    Addition(const Shape& new_input_shape = {}, const string& new_name = "")
    {
        set(new_input_shape, new_name);
    }

    Shape get_output_shape() const override
    {
        return input_shape;
    }

    void set(const Shape& new_input_shape, const string& new_label)
    {
        if(!new_input_shape.empty() && new_input_shape.rank != Rank - 1)
            throw runtime_error("Input shape rank for AdditionLayer<" + to_string(Rank) + "> must be " + to_string(Rank - 1) + " (without batch dimension).");

        input_shape = new_input_shape;

        label = new_label;

        if constexpr (Rank == 3) {
            name = "Addition3d";
            layer_type = LayerType::Addition3d;
        } else if constexpr (Rank == 4) {
            name = "Addition4d";
            layer_type = LayerType::Addition4d;
        } else {
            throw runtime_error("Addition layer not implemented for rank: " + to_string(Rank));
        }
    }

    void forward_propagate(ForwardPropagation& forward_propagation, size_t layer, bool) override
    {
        const TensorView& input_1 = forward_propagation.views[layer][Inputs][0];
        const TensorView& input_2 = forward_propagation.views[layer][Inputs][1];

        TensorView& output = forward_propagation.views[layer][Outputs][0];

        addition(input_1, input_2, output);
    }

    void back_propagate(ForwardPropagation&,
                        BackPropagation& back_propagation,
                        size_t layer) const override
    {
        const TensorView& output_gradient = back_propagation.backward_views[layer][OutputGradients][0];

        TensorView& input_gradient_0 = back_propagation.backward_views[layer][InputGradients][0];
        TensorView& input_gradient_1 = back_propagation.backward_views[layer][InputGradients][1];

        copy(output_gradient, input_gradient_0);
        copy(output_gradient, input_gradient_1);
    }

    void from_XML(const XmlDocument& document) override
    {
        const XmlElement* element = document.first_child_element("Addition");
        if(!element) throw runtime_error(name + " element is nullptr.");

        const string new_label = read_xml_string(element, "Label");
        const Shape new_input_shape = string_to_shape(read_xml_string(element, "InputDimensions"));

        set(new_input_shape, new_label);
    }

    void to_XML(XmlPrinter& printer) const override
    {
        printer.open_element("Addition");

        add_xml_element(printer, "Label", label);
        add_xml_element(printer, "InputDimensions", shape_to_string(input_shape));

        printer.close_element();
    }

private:

    enum Forward {Inputs, Outputs};
    enum Backward {OutputGradients, InputGradients};
};

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
