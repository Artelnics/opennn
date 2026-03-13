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

namespace opennn
{

template<int Rank> struct FlattenForwardPropagation;
template<int Rank> struct FlattenBackPropagation;

#ifdef OPENNN_CUDA

template<int Rank> struct FlattenForwardPropagationCuda;
template<int Rank> struct FlattenBackPropagationCuda;

#endif // OPENNN_CUDA


template<int Rank>
class Flatten final : public Layer
{

public:

    Flatten(const Shape& new_input_shape = {} )
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

    // Forward propagation
    
    void forward_propagate(const vector<TensorView>& input_views,
                           unique_ptr<LayerForwardPropagation>& layer_forward_propagation,
                           bool) override
    {
        FlattenForwardPropagation<Rank>* forward_propagation =
            static_cast<FlattenForwardPropagation<Rank>*>(layer_forward_propagation.get());

        const size_t bytes_to_copy = input_views[0].size() * sizeof(type);

        if (input_views[0].data != forward_propagation->outputs.data)
            memcpy(forward_propagation->outputs.data, input_views[0].data, bytes_to_copy);
    }
    
    // Back-propagation
    
    void back_propagate(const vector<TensorView>&,
                        const vector<TensorView>& output_gradient_views,
                        unique_ptr<LayerForwardPropagation>&,
                        unique_ptr<LayerBackPropagation>& layer_back_propagation) const override
    {
        FlattenBackPropagation<Rank>* flatten_back_propagation =
            static_cast<FlattenBackPropagation<Rank>*>(layer_back_propagation.get());

        type* source_ptr = output_gradient_views[0].data;
        type* dest_ptr = flatten_back_propagation->input_gradients[0].data;

        const size_t bytes_to_copy = flatten_back_propagation->input_gradients[0].size() * sizeof(type);

        if (source_ptr && dest_ptr && source_ptr != dest_ptr)
            memcpy(dest_ptr, source_ptr, bytes_to_copy);
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

#ifdef OPENNN_CUDA

public:

    void forward_propagate(const vector<TensorViewCuda>& input_views,
                           unique_ptr<LayerForwardPropagationCuda>& forward_propagation,
                           bool)
    {
        type* inputs = input_views[0].data;
        type* outputs = forward_propagation->outputs.data;

        const size_t bytes_to_copy = input_views[0].size() * sizeof(type);

        CHECK_CUDA(cudaMemcpy(outputs,
                              inputs,
                              bytes_to_copy,
                              cudaMemcpyDeviceToDevice));
    }


    void back_propagate(const vector<TensorViewCuda>& input_views,
                        const vector<TensorViewCuda>& output_gradient_views,
                        unique_ptr<LayerForwardPropagationCuda>& forward_propagation,
                        unique_ptr<LayerBackPropagationCuda>& back_propagation) const
    {
        type* input_gradients = output_gradient_views[0].data;
        type* output_gradients = back_propagation->input_gradients[0].data;

        const size_t bytes_to_copy = back_propagation->input_gradients[0].size() * sizeof(type);

        CHECK_CUDA(cudaMemcpy(input_gradients,
                              output_gradients,
                              bytes_to_copy,
                              cudaMemcpyDeviceToDevice));
    }

#endif

private:

    Shape input_shape;
};


template<int Rank>
struct FlattenForwardPropagation final : LayerForwardPropagation
{
    FlattenForwardPropagation(const Index new_batch_size = 0, Layer* new_layer = nullptr)
    {
        set(new_batch_size, new_layer);
    }

    void initialize() override
    {
        const Shape output_shape = layer->get_output_shape();
        outputs.shape = {batch_size, output_shape[0]};
        outputs_memory.resize(outputs.shape.count());
        outputs.data = outputs_memory.data();
    }
    VectorR outputs_memory;

    void print() const override
    {
        cout << "Flatten Outputs Dimensions:" << endl << outputs.shape << endl;
    }
};


template<int Rank>
struct FlattenBackPropagation final : LayerBackPropagation
{
    FlattenBackPropagation(const Index new_batch_size = 0, Layer* new_layer = nullptr)
    {
        set(new_batch_size, new_layer);
    }

    void initialize() override
    {
        const Flatten<Rank>* flatten_layer = static_cast<const Flatten<Rank>*>(layer);

        const Shape input_shape = flatten_layer->get_input_shape();

        Shape full_input_shape = { batch_size };
        full_input_shape.insert(full_input_shape.end(), input_shape.begin(), input_shape.end());

        input_gradients_memory.resize(1);
        input_gradients_memory[0].resize(full_input_shape.count());
        input_gradients.resize(1);
        input_gradients[0].data = input_gradients_memory[0].data();
        input_gradients[0].shape = full_input_shape;
    }

    void print() const override
    {        
        cout << "Flatten Deltas Dimensions:" << endl << input_gradients[0].shape << endl;
    }
};


#ifdef OPENNN_CUDA

template<int Rank>
struct FlattenForwardPropagationCuda : public LayerForwardPropagationCuda
{
    FlattenForwardPropagationCuda(const Index new_batch_size = 0, Layer* new_layer = nullptr)
    {
        set(new_batch_size, new_layer);
    }

    void initialize() override
    {
        const Index outputs_number = layer->get_outputs_number();
        outputs.set_descriptor({batch_size, outputs_number});
    }
};


template<int Rank>
struct FlattenBackPropagationCuda : public LayerBackPropagationCuda
{
    FlattenBackPropagationCuda(const Index new_batch_size = 0, Layer* new_layer = nullptr)
    {
        set(new_batch_size, new_layer);
    }

    void initialize() override
    {
        const Shape input_shape = layer->get_input_shape();

        Shape full_input_shape = { batch_size };
        full_input_shape.insert(full_input_shape.end(), input_shape.begin(), input_shape.end());

        input_gradients.resize(1);
        input_gradients[0].resize(full_input_shape);
    }
};

#endif

void reference_flatten_layer();

}

#pragma once

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
