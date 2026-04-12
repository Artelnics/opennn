//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   F L A T T E N   L A Y E R   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#pragma once

#include "tensor_utilities.h"
#include "math_utilities.h"
#include "forward_propagation.h"
#include "back_propagation.h"

namespace opennn
{

template<int Rank>
class Flatten final : public Layer
{

public:

    Flatten(const Shape& new_input_shape = {})
    {
        set(new_input_shape);
    }

    Shape get_output_shape() const override
    {
        return { input_shape.size() };
    }

    void set(const Shape& new_input_shape)
    {
        if (new_input_shape.rank != Rank - 1)
            throw runtime_error("Error: Input shape size must match layer Rank in FlattenLayer::set().");

        name = "Flatten" + to_string(Rank) + "d";

        set_label("flatten_layer");

        input_shape = new_input_shape;
    }

    void set_input_shape(const Shape& new_input_shape) override
    {
        input_shape = new_input_shape;
    }

    vector<Shape> get_forward_shapes(const Index batch_size) const override
    {
        return {Shape{batch_size}.append(get_output_shape())}; // {batch, total_count}
    }

    vector<Shape> get_backward_shapes(Index batch_size) const override
    {
        return {Shape{batch_size}.append(input_shape)};
    }

    
    void forward_propagate(ForwardPropagation& forward_propagation, size_t layer, bool) override
    {
        const TensorView& input = forward_propagation.views[layer][Inputs][0];
        TensorView& output = forward_propagation.views[layer][Outputs][0];

        copy(input, output);
    }
    
    
    void back_propagate(ForwardPropagation&,
                        BackPropagation& back_propagation,
                        size_t layer) const override
    {
        const TensorView& output_gradient = back_propagation.backward_views[layer][OutputGradients][0];
        TensorView& input_gradient = back_propagation.backward_views[layer][InputGradients][0];

        copy(output_gradient, input_gradient);
    }
    
    // Serialization

    void from_XML(const XmlDocument& document) override
    {
        const XmlElement* element = get_xml_root(document, "Flatten");

        const Index input_height = read_xml_index(element, "InputHeight");
        const Index input_width = read_xml_index(element, "InputWidth");

        if constexpr (Rank == 3)
        {
            const Index input_channels = read_xml_index(element, "InputChannels");
            set({input_height, input_width, input_channels});
        }
        else
            set({input_height, input_width});

    }

    void to_XML(XmlPrinter& printer) const override
    {
        printer.open_element("Flatten");

        add_xml_element(printer, "InputHeight", to_string(get_input_height()));
        add_xml_element(printer, "InputWidth", to_string(get_input_width()));
        if constexpr (Rank == 3)
            add_xml_element(printer, "InputChannels", to_string(get_input_channels()));

        printer.close_element();
    }

private:

    Index get_input_height() const
    {
        if constexpr (Rank < 2)
            throw logic_error("get_input_height() requires Rank ≥ 2.");
        return input_shape[0];
    }

    Index get_input_width() const
    {
        if constexpr (Rank < 2)
            throw logic_error("get_input_width() requires Rank >= 2.");
        return input_shape[1];
    }

    Index get_input_channels() const
    {
        if constexpr (Rank < 3)
            throw logic_error("get_input_channels() requires Rank >= 3.");
        return input_shape[2];
    }

    enum Forward {Inputs, Outputs};
    enum Backward {OutputGradients, InputGradients};
};

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
