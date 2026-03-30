//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   F L A T T E N   L A Y E R   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#pragma once

#include "layer.h"
#include "tensor_utilities.h"
#include "math_utilities.h"
#include "neural_network.h"
#include "loss.h"

namespace opennn
{

#ifdef CUDA

template<int Rank> struct FlattenForwardPropagationCuda;
template<int Rank> struct FlattenBackPropagationCuda;

#endif // CUDA


template<int Rank>
class Flatten final : public Layer
{

public:

    Flatten(const Shape& new_input_shape = {})
    {
        set(new_input_shape);
    }


    Shape get_input_shape() const override
    {
        return input_shape;
    }


    Shape get_output_shape() const override
    {
        if (input_shape.empty() || input_shape[0] == 0)
            return {0};

        return { (Index)accumulate(input_shape.begin(), input_shape.end(), (size_t)1, multiplies<size_t>()) };
    }


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


    void set(const Shape& new_input_shape)
    {
        if (new_input_shape.size() != Rank - 1)
            throw runtime_error("Error: Input shape size must match layer Rank in FlattenLayer::set().");

        name = "Flatten" + to_string(Rank) + "d";

        set_label("flatten_layer");

        input_shape = new_input_shape;
    }


    vector<Shape> get_forward_shapes() const override
    {
        return {};
    }


    vector<Shape> get_backward_shapes() const override
    {
        /*
        input_gradients = {{nullptr, Shape{batch_size}.append(flatten_layer->get_input_shape())}};
        */

        return {};
    }

    
    void forward_propagate(ForwardPropagation& forward_propagation, size_t layer, bool) override
    {
        const TensorView& input = forward_propagation.views[layer][Inputs][0];
        TensorView& output = forward_propagation.views[layer][Outputs][0];

        opennn::copy(input, output);
    }
    
    
    void back_propagate(ForwardPropagation&,
                        BackPropagation& back_propagation,
                        size_t layer) const override
    {
        const TensorView& output_gradient = back_propagation.backward_views[layer][OutputGradients][0];
        TensorView& input_gradient = back_propagation.backward_views[layer][InputGradients][0];

        opennn::copy(output_gradient, input_gradient);
    }
    
    // Serialization

    void from_XML(const XMLDocument& document) override
    {
        const XMLElement* element = document.FirstChildElement("Flatten");

        if(!element)
            throw runtime_error("Flatten2d element is nullptr.\n");

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

    void to_XML(XMLPrinter& printer) const override
    {
        printer.OpenElement("Flatten");

        add_xml_element(printer, "InputHeight", to_string(get_input_height()));
        add_xml_element(printer, "InputWidth", to_string(get_input_width()));
        if constexpr (Rank == 3)
            add_xml_element(printer, "InputChannels", to_string(get_input_channels()));

        printer.CloseElement();
    }

    void print() const override
    {
        cout << "Flatten layer" << endl
             << "Input shape: " << input_shape << endl
             << "Output shape: " << get_output_shape() << endl;
    }

private:

    enum Forward {Inputs, Outputs};
    enum Backward {OutputGradients, InputGradients};

    Shape input_shape;
};

void reference_flatten_layer();
}


// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// This library is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 2.1 of the License, or any later version.
// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.
// You should have received a copy of the GNU Lesser General Public
// License along with this library; if not, write to the Free Software
// Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
