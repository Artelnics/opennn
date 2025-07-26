//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   A D D I T I O N   L A Y E R   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#ifndef ADDITIONLAYER_H
#define ADDITIONLAYER_H

#include "layer.h"
#include "tensors.h"

namespace opennn
{

template<int Rank> struct AdditionForwardPropagation;
template<int Rank> struct AdditionBackPropagation;

template<int Rank>
class Addition : public Layer
{

public:

    Addition(const dimensions& new_input_dimensions = dimensions({}), const string& new_name = "")
    {
        set(new_input_dimensions, new_name);
    }

    dimensions get_input_dimensions() const override
    {
        return input_dimensions;
    }

    dimensions get_output_dimensions() const override
    {
        return input_dimensions;
    }


    void set(const dimensions& new_input_dimensions, const string& new_label)
    {
        if (!new_input_dimensions.empty() && new_input_dimensions.size() != Rank)
            throw runtime_error("Input dimensions rank for AdditionLayer<" + to_string(Rank) + "> must be " + to_string(Rank));

        input_dimensions = new_input_dimensions;

        label = new_label;

        name = "Addition";
    }


    void forward_propagate(const vector<pair<type*, dimensions>>& input_pairs,
                           unique_ptr<LayerForwardPropagation>& layer_forward_propagation,
                           const bool&) override
    {

        if (input_pairs.size() != 2)
            throw runtime_error(name + " layer requires exactly two inputs.");

        if (input_pairs[0].second != input_pairs[1].second)
            throw runtime_error("Input dimensions for " + name + " must be identical.");

        const TensorMap<Tensor<type, Rank>> input_1 = tensor_map<Rank>(input_pairs[0]);
        const TensorMap<Tensor<type, Rank>> input_2 = tensor_map<Rank>(input_pairs[1]);

        AdditionForwardPropagation<Rank>* this_forward_propagation =
            static_cast<AdditionForwardPropagation<Rank>*>(layer_forward_propagation.get());

        Tensor<type, Rank>& outputs = this_forward_propagation->outputs;
        outputs.device(*thread_pool_device) = input_1 + input_2;

    }

    void back_propagate(const vector<pair<type*, dimensions>>&,
                        const vector<pair<type*, dimensions>>& delta_pairs,
                        unique_ptr<LayerForwardPropagation>&,
                        unique_ptr<LayerBackPropagation>& back_propagation) const override
    {
        if (delta_pairs.size() != 1)
            throw runtime_error(name + " backpropagation requires exactly one delta input.");

        const TensorMap<Tensor<type, Rank>> deltas = tensor_map<Rank>(delta_pairs[0]);

        AdditionBackPropagation<Rank>* this_back_propagation =
            static_cast<AdditionBackPropagation<Rank>*>(back_propagation.get());

        // The gradient of an addition is 1, so the incoming delta is passed back to both inputs.
        this_back_propagation->input_1_derivatives.device(*thread_pool_device) = deltas;
        this_back_propagation->input_2_derivatives.device(*thread_pool_device) = deltas;

    }

    void from_XML(const XMLDocument& document) override
    {
        const XMLElement* element = document.FirstChildElement("Addition");
        if (!element) throw runtime_error(name + " element is nullptr.");

        const string new_label = read_xml_string(element, "Label");
        const dimensions new_input_dimensions = string_to_dimensions(read_xml_string(element, "InputDimensions"));

        set(new_input_dimensions, new_label);
    }


    void to_XML(XMLPrinter& printer) const override
    {
        printer.OpenElement("Addition");

        add_xml_element(printer, "Label", label);
        add_xml_element(printer, "InputDimensions", dimensions_to_string(input_dimensions));

        printer.CloseElement();
    }

#ifdef OPENNN_CUDA

public:

    void forward_propagate_cuda(const vector<float*>&,
                                unique_ptr<LayerForwardPropagationCuda>&,
                                const bool&) override;

    void back_propagate_cuda(const vector<float*>&,
                             const vector<float*>&,
                             unique_ptr<LayerForwardPropagationCuda>&,
                             unique_ptr<LayerBackPropagationCuda>&) const override;

#endif

private:

    dimensions input_dimensions;
};


template<int Rank>
struct AdditionForwardPropagation : LayerForwardPropagation
{
    AdditionForwardPropagation(const Index& new_batch_size = 0, Layer* new_layer = nullptr)
        : LayerForwardPropagation()
    {
        set(new_batch_size, new_layer);
    }

    pair<type*, dimensions> get_output_pair() const override
    {
        const dimensions output_dimensions = layer->get_output_dimensions();

        dimensions full_dimensions = {batch_size};
        full_dimensions.insert(full_dimensions.end(), output_dimensions.begin(), output_dimensions.end());

        return {(type*)outputs.data(), full_dimensions};
    }


    void set(const Index& new_batch_size, Layer* new_layer) override
    {
        if (!new_layer) return;

        layer = new_layer;
        batch_size = new_batch_size;

        const dimensions output_dimensions = layer->get_output_dimensions();

        array<Index, Rank> full_dimensions;
        full_dimensions[0] = batch_size;

        for(int i = 0; i < Rank; ++i)
            full_dimensions[i+1] = output_dimensions[i];

        outputs.resize(DSizes<Index, Rank>(full_dimensions));
    }

    void print() const override
    {

    }

    Tensor<type, Rank> outputs;
};


template<int Rank>
struct AdditionBackPropagation : LayerBackPropagation
{
    AdditionBackPropagation(const Index& new_batch_size = 0, Layer* new_layer = nullptr)
        : LayerBackPropagation()
    {
        set(new_batch_size, new_layer);
    }

    vector<pair<type*, dimensions>> get_input_derivative_pairs() const override
    {
        const dimensions input_dimensions = layer->get_input_dimensions();
        dimensions full_dimensions = {batch_size};
        full_dimensions.insert(full_dimensions.end(), input_dimensions.begin(), input_dimensions.end());

        return {{(type*)input_1_derivatives.data(), full_dimensions},
                {(type*)input_2_derivatives.data(), full_dimensions}};
    }


    void set(const Index& new_batch_size, Layer* new_layer) override
    {
        if (!new_layer) return;

        layer = new_layer;
        batch_size = new_batch_size;

        const dimensions input_dimensions = layer->get_input_dimensions();

        array<Index, Rank + 1> full_dimensions;
        full_dimensions[0] = batch_size;

        for(int i = 0; i < Rank; ++i)
            full_dimensions[i+1] = input_dimensions[i];

        // auto d_sizes = DSizes<Index, Rank>(full_dimensions);
        // input_1_derivatives.resize(d_sizes);
        // input_2_derivatives.resize(d_sizes);
    }

    void print() const override
    {

    }

    Tensor<type, Rank> input_1_derivatives;
    Tensor<type, Rank> input_2_derivatives;
};


#ifdef OPENNN_CUDA

struct Addition3dForwardPropagationCuda : public LayerForwardPropagationCuda
{
    Addition3dForwardPropagationCuda(const Index & = 0, Layer* = nullptr);

    void set(const Index & = 0, Layer* = nullptr) override;

    void print() const override;
    
    void free() override;
};


struct Addition3dBackPropagationCuda : public LayerBackPropagationCuda
{
    Addition3dBackPropagationCuda(const Index & = 0, Layer* = nullptr);

    vector<float*> get_input_derivatives_device() override;

    void set(const Index & = 0, Layer* = nullptr) override;
    
    void print() const override;

    void free() override;
    
    float* inputs_1_derivatives = nullptr;
    float* inputs_2_derivatives = nullptr;
};

#endif
}

#endif


// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2025 Artificial Intelligence Techniques, SL.
//
// This library is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 2.1 of the License, or any later version.
//
// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.

// You should have received a copy of the GNU Lesser General Public
// License along with this library; if not, write to the Free Software

// Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
