//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   F L A T T E N   L A Y E R   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#ifndef FLATTENLAYER_H
#define FLATTENLAYER_H

#include "layer.h"
#include "tensors.h"

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

    Flatten(const dimensions& new_input_dimensions = {} )
    {
        set(new_input_dimensions);
    }


    dimensions get_input_dimensions() const override
    {
        return input_dimensions;
    }


    dimensions get_output_dimensions() const override
    {
        if (input_dimensions.empty() || input_dimensions[0] == 0)
            return {0};

        return { (Index)accumulate(input_dimensions.begin(), input_dimensions.end(), (size_t)1, multiplies<size_t>()) };
    }


    Index get_input_height() const
    {

        if constexpr (Rank < 2)
            throw logic_error("get_input_height() requires Rank ≥ 2.");

        return input_dimensions[0];
    }


    Index get_input_width() const
    {
        if constexpr (Rank < 2)
            throw logic_error("get_input_width() requires Rank >= 2.");

        return input_dimensions[1];
    }


    Index get_input_channels() const
    {
        if constexpr (Rank < 3)
            throw logic_error("get_input_channels() requires Rank >= 3.");

        return input_dimensions[2];
    }


    void set(const dimensions& new_input_dimensions)
    {
        if (new_input_dimensions.size() != Rank - 1)
            throw runtime_error("Error: Input dimensions size must match layer Rank in FlattenLayer::set().");

        name = "Flatten" + to_string(Rank) + "d";

        set_label("flatten_layer");

        input_dimensions = new_input_dimensions;
    }

    // Forward propagation

    void forward_propagate(const vector<TensorView>& input_views,
                           unique_ptr<LayerForwardPropagation>& layer_forward_propagation,
                           const bool&) override
    {
        const Index batch_size = layer_forward_propagation->batch_size;
        const Index outputs_number = get_outputs_number();

        FlattenForwardPropagation<Rank>* forward_prop =
            static_cast<FlattenForwardPropagation<Rank>*>(layer_forward_propagation.get());

        forward_prop->outputs = TensorMap<Tensor<type, 2>>(input_views[0].data, batch_size, outputs_number);
    }

    // Back-propagation

    void back_propagate(const vector<TensorView>&,
                        const vector<TensorView>& delta_views,
                        unique_ptr<LayerForwardPropagation>&,
                        unique_ptr<LayerBackPropagation>& back_propagation) const override
    {
        const Index batch_size = back_propagation->batch_size;
        const Index outputs_number = get_outputs_number();

        FlattenBackPropagation<Rank>* back_prop =
            static_cast<FlattenBackPropagation<Rank>*>(back_propagation.get());

        memcpy(back_prop->input_deltas.data(),
               delta_views[0].data,
               (batch_size * outputs_number * sizeof(type)));
    }

    // Serialization

    void from_XML(const XMLDocument& document) override
    {

        const XMLElement* element = document.FirstChildElement("Flatten");
        if (!element) throw runtime_error("Flatten2d element is nullptr.\n");

        const Index input_height = read_xml_value<Index>(element, "InputHeight");
        const Index input_width = read_xml_value<Index>(element, "InputWidth");

        if constexpr (Rank == 3){
            const Index input_channels = read_xml_value<Index>(element, "InputChannels");
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
        cout << "Flatten layer" << endl;

        cout << "Input dimensions: ";
        print_vector(input_dimensions);

        cout << "Output dimensions: ";
        print_vector(get_output_dimensions());
    }

#ifdef OPENNN_CUDA

public:

    void forward_propagate_cuda(const vector<float*>& inputs_device,
                                unique_ptr<LayerForwardPropagationCuda>& forward_propagation_cuda,
                                const bool&)
    {
        FlattenForwardPropagationCuda<Rank>* fp_cuda =
            static_cast<FlattenForwardPropagationCuda<Rank>*>(forward_propagation_cuda.get());

        const Index batch_size = fp_cuda->batch_size;
        const Index outputs_number = get_outputs_number();

        if constexpr (Rank == 4)
        {
            const Index height = get_input_height();
            const Index width = get_input_width();
            const Index channels = get_input_channels();

            type* reordered_inputs = fp_cuda->reordered_inputs;
            type* outputs_device = fp_cuda->outputs;

            invert_reorder_inputs_cuda(inputs_device[0], reordered_inputs, batch_size, channels, height, width);

            reorganize_inputs_cuda(reordered_inputs, outputs_device, batch_size, outputs_number);
            //reorganize_inputs_cuda(inputs_device[0], outputs_device, batch_size, outputs_number);
        }
        else
            CHECK_CUDA(cudaMemcpy(fp_cuda->outputs, inputs_device[0], batch_size * outputs_number * sizeof(type), cudaMemcpyDeviceToDevice));
    }


    void back_propagate_cuda(const vector<float*>&,
                             const vector<float*>& deltas_device,
                             unique_ptr<LayerForwardPropagationCuda>&,
                             unique_ptr<LayerBackPropagationCuda>& back_propagation_cuda) const
    {
        FlattenBackPropagationCuda<Rank>* flatten_layer_back_propagation_cuda =
            static_cast<FlattenBackPropagationCuda<Rank>*>(back_propagation_cuda.get());

        const Index batch_size = flatten_layer_back_propagation_cuda->batch_size;
        const Index outputs_number = get_outputs_number();

        type* input_deltas = flatten_layer_back_propagation_cuda->input_deltas;

        reorganize_deltas_cuda(deltas_device[0], input_deltas, batch_size, outputs_number);
    }

#endif

private:

    dimensions input_dimensions;
};

template<int Rank>
struct FlattenForwardPropagation final : LayerForwardPropagation
{
    FlattenForwardPropagation(const Index& new_batch_size = 0, Layer* new_layer = nullptr)
    {
        set(new_batch_size, new_layer);
    }


    TensorView get_output_pair() const override
    {
        const dimensions output_dimensions = layer->get_output_dimensions();

        return {(type*)outputs.data(), {batch_size, output_dimensions[0]}};
    }


    void set(const Index& new_batch_size = 0, Layer* new_layer = nullptr) override
    {
        if (!new_layer) return;

        batch_size = new_batch_size;
        layer = new_layer;

        const dimensions output_dimensions = layer->get_output_dimensions();
        outputs.resize(batch_size, output_dimensions[0]);
    }


    void print() const override
    {
        cout << "Flatten Outputs:" << endl << outputs.dimensions() << endl;
    }


    Tensor<type, 2> outputs;
};


template<int Rank>
struct FlattenBackPropagation final : LayerBackPropagation
{
    FlattenBackPropagation(const Index& new_batch_size = 0, Layer* new_layer = nullptr)
    {
        set(new_batch_size, new_layer);
    }


    vector<TensorView> get_input_derivative_views() const override
    {
        const Flatten<Rank>* layer_ptr = static_cast<const Flatten<Rank>*>(layer);
        const dimensions input_dimensions = layer_ptr->get_input_dimensions();

        dimensions full_dimensions;
        full_dimensions.reserve(Rank + 1);
        full_dimensions.push_back(batch_size);
        full_dimensions.insert(full_dimensions.end(), input_dimensions.begin(), input_dimensions.end());

        return {{(type*)(input_deltas.data()), full_dimensions}};
    }

    void set(const Index& new_batch_size = 0, Layer* new_layer = nullptr) override
    {
        if (!new_layer) return;

        batch_size = new_batch_size;
        layer = new_layer;

        const Flatten<Rank>* layer_ptr = static_cast<const Flatten<Rank>*>(layer);
        const dimensions input_dimensions = layer_ptr->get_input_dimensions();

        array<Index, Rank + 1> resize_dimensions;
        resize_dimensions[0] = batch_size;

        for(int i = 0; i < Rank; ++i)
            resize_dimensions[i + 1] = input_dimensions[i];

        input_deltas.resize(resize_dimensions);
    }


    void print() const override
    {
        cout << "Flatten Input derivatives:" << endl << input_deltas.dimensions() << endl;
    }

    Tensor<type, Rank> input_deltas;
};


#ifdef OPENNN_CUDA

template<int Rank>
struct FlattenForwardPropagationCuda : public LayerForwardPropagationCuda
{
    FlattenForwardPropagationCuda(const Index & = 0, Layer* = nullptr);

    void set(const Index & = 0, Layer* = nullptr) override;

    void free() override;

    type* reordered_inputs = nullptr;
};


template<int Rank>
struct FlattenBackPropagationCuda : public LayerBackPropagationCuda
{
    FlattenBackPropagationCuda(const Index & = 0, Layer* = nullptr);

    void set(const Index & = 0, Layer* = nullptr) override;

    void free() override;
};


template<int Rank>
FlattenForwardPropagationCuda<Rank>::FlattenForwardPropagationCuda(const Index& new_batch_size, Layer* new_layer)
{
    set(new_batch_size, new_layer);
}


template<int Rank>
void FlattenForwardPropagationCuda<Rank>::set(const Index& new_batch_size, Layer* new_layer)
{
    if (!new_layer) return;

    layer = new_layer;
    batch_size = new_batch_size;

    const Index inputs_number = layer->get_inputs_number();
    const Index outputs_number = layer->get_outputs_number();

    if constexpr (Rank == 4)
    {
        CHECK_CUDA(cudaMalloc(&reordered_inputs, batch_size * inputs_number * sizeof(float)));
        //CUDA_MALLOC_AND_REPORT(reordered_inputs, batch_size * inputs_number * sizeof(float));
    }

    CHECK_CUDA(cudaMalloc(&outputs, batch_size * outputs_number * sizeof(float)));
    //CUDA_MALLOC_AND_REPORT(outputs, batch_size * outputs_number * sizeof(float));
}


template<int Rank>
void FlattenForwardPropagationCuda<Rank>::free()
{
    if (outputs) cudaFree(outputs);
    outputs = nullptr;

    if (reordered_inputs) cudaFree(reordered_inputs);
    reordered_inputs = nullptr;
}


template<int Rank>
FlattenBackPropagationCuda<Rank>::FlattenBackPropagationCuda(const Index& new_batch_size, Layer* new_layer)
{
    set(new_batch_size, new_layer);
}


template<int Rank>
void FlattenBackPropagationCuda<Rank>::set(const Index& new_batch_size, Layer* new_layer)
{
    if (!new_layer) return;

    layer = new_layer;
    batch_size = new_batch_size;

    const size_t inputs_number = layer->get_inputs_number();

    CHECK_CUDA(cudaMalloc(&input_deltas, batch_size * inputs_number * sizeof(float)));
    //CUDA_MALLOC_AND_REPORT(input_deltas, batch_size * inputs_number * sizeof(float));
}


template<int Rank>
void FlattenBackPropagationCuda<Rank>::free()
{
    if (input_deltas) cudaFree(input_deltas);
    input_deltas = nullptr;
}

#endif

void reference_flatten_layer();

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
