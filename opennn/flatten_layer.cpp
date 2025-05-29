//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   F L A T T E N   L A Y E R   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "flatten_layer.h"
#include "tensors.h"

namespace opennn
{

Flatten::Flatten(const dimensions& new_input_dimensions) : Layer()
{
    set(new_input_dimensions);
}


dimensions Flatten::get_input_dimensions() const
{
    return input_dimensions;
}


dimensions Flatten::get_output_dimensions() const
{
    return { input_dimensions[0] * input_dimensions[1] * input_dimensions[2] };
}


Index Flatten::get_input_height() const
{
    return input_dimensions[0];
}


Index Flatten::get_input_width() const
{
    return input_dimensions[1];
}


Index Flatten::get_input_channels() const
{
    return input_dimensions[2];
}


void Flatten::set(const dimensions& new_input_dimensions)
{
    layer_type = Type::Flatten;

    set_name("flatten_layer");

    input_dimensions = new_input_dimensions;
}


void Flatten::forward_propagate(const vector<pair<type*, dimensions>>& input_pairs,
                                unique_ptr<LayerForwardPropagation>& layer_forward_propagation,
                                const bool&)
{
    const Index batch_size = layer_forward_propagation->batch_size;
    const Index outputs_number = get_outputs_number();

    FlattenForwardPropagation* flatten_layer_forward_propagation =
            static_cast<FlattenForwardPropagation*>(layer_forward_propagation.get());

    flatten_layer_forward_propagation->outputs = TensorMap<Tensor<type, 2>>(input_pairs[0].first, batch_size, outputs_number);
}


void Flatten::back_propagate(const vector<pair<type*, dimensions>>& input_pairs,
                             const vector<pair<type*, dimensions>>& delta_pairs,
                             unique_ptr<LayerForwardPropagation>&,
                             unique_ptr<LayerBackPropagation>& back_propagation) const
{
    const Index batch_size = input_pairs[0].second[0];
    const Index height = input_pairs[0].second[1];
    const Index width = input_pairs[0].second[2];
    const Index channels = input_pairs[0].second[3];

    FlattenBackPropagation* flatten_layer_back_propagation =
        static_cast<FlattenBackPropagation*>(back_propagation.get());

    flatten_layer_back_propagation->input_derivatives = TensorMap<Tensor<type, 4>>(delta_pairs[0].first,
        batch_size, height, width, channels);
}


void Flatten::to_XML(XMLPrinter& printer) const
{
    printer.OpenElement("Flatten");

    add_xml_element(printer, "InputHeight", to_string(get_input_height()));
    add_xml_element(printer, "InputWidth", to_string(get_input_width()));
    add_xml_element(printer, "InputChannels", to_string(get_input_channels()));

    printer.CloseElement(); 
}


void Flatten::from_XML(const XMLDocument& document)
{
    const XMLElement* flatten_layer_element = document.FirstChildElement("Flatten");

    if (!flatten_layer_element) 
        throw runtime_error("Flatten element is nullptr.\n");

    const Index input_height = read_xml_index(flatten_layer_element, "InputHeight");
    const Index input_width = read_xml_index(flatten_layer_element, "InputWidth");
    const Index input_channels = read_xml_index(flatten_layer_element, "InputChannels");

    set({ input_height, input_width, input_channels });
}


void Flatten::print() const
{
    cout << "Flatten layer" << endl;

    cout << "Input dimensions: " << endl;
    print_vector(input_dimensions);

    cout << "Output dimensions: " << endl;
    print_vector(get_output_dimensions());
}


FlattenForwardPropagation::FlattenForwardPropagation(const Index& new_batch_size, Layer* new_layer)
    : LayerForwardPropagation()
{
    set(new_batch_size, new_layer);
}


pair<type*, dimensions> FlattenForwardPropagation::get_outputs_pair() const
{
    const dimensions output_dimensions = layer->get_output_dimensions();

    return {(type*)outputs.data(), {batch_size, output_dimensions[0]}};
}


void FlattenForwardPropagation::set(const Index& new_batch_size, Layer* new_layer)
{
    batch_size = new_batch_size;

    layer = new_layer;

    const dimensions output_dimensions = layer->get_output_dimensions();

    outputs.resize(batch_size, output_dimensions[0]);
}


void FlattenForwardPropagation::print() const
{
    cout << "Flatten Outputs dimensions:" << endl
         << outputs.dimensions() << endl;
}


void FlattenBackPropagation::set(const Index& new_batch_size, Layer* new_layer)
{
    layer = new_layer;

    batch_size = new_batch_size;

    const Flatten* flatten_layer = static_cast<Flatten*>(layer);

    const dimensions input_dimensions = flatten_layer->get_input_dimensions();

    input_derivatives.resize(batch_size,
                             input_dimensions[0],
                             input_dimensions[1],
                             input_dimensions[2]);
}


void FlattenBackPropagation::print() const
{
    cout << "Flatten Input derivatives:" << endl
         << input_derivatives.dimensions() << endl;
}


FlattenBackPropagation::FlattenBackPropagation(const Index& new_batch_size, Layer* new_layer)
    : LayerBackPropagation()
{
    set(new_batch_size, new_layer);
}


vector<pair<type*, dimensions>> FlattenBackPropagation::get_input_derivative_pairs() const
{
    const Flatten* flatten_layer = static_cast<Flatten*>(layer);

    const dimensions input_dimensions = flatten_layer->get_input_dimensions();

    return {{(type*)(input_derivatives.data()),
            {batch_size, input_dimensions[0], input_dimensions[1], input_dimensions[2]}}};
}


#ifdef OPENNN_CUDA

void Flatten::forward_propagate_cuda(const vector<pair<type*, dimensions>>& input_pairs_device,
                                     unique_ptr<LayerForwardPropagationCuda>& forward_propagation_cuda,
                                     const bool&)
{
    // Inputs

    const Index batch_size = input_pairs_device[0].second[0];
    const Index height = input_pairs_device[0].second[1];
    const Index width = input_pairs_device[0].second[2];
    const Index channels = input_pairs_device[0].second[3];

    const Index outputs_number = get_outputs_number();
    
    type* inputs_device = input_pairs_device[0].first;

    // Forward propagation

    FlattenForwardPropagationCuda* flatten_layer_forward_propagation_cuda =
        static_cast<FlattenForwardPropagationCuda*>(forward_propagation_cuda.get());

    type* reordered_inputs = flatten_layer_forward_propagation_cuda->reordered_inputs;
    type* outputs_device = flatten_layer_forward_propagation_cuda->outputs;

    invert_reorder_inputs_cuda(inputs_device, reordered_inputs, batch_size, channels, height, width);

    reorganize_inputs_cuda(reordered_inputs, outputs_device, batch_size, outputs_number);  
}


void Flatten::back_propagate_cuda(const vector<pair<type*, dimensions>>& input_pairs_device,
                                  const vector<pair<type*, dimensions>>& deltas_pair_device,
                                  unique_ptr<LayerForwardPropagationCuda>& forward_propagation_cuda,
                                  unique_ptr<LayerBackPropagationCuda>& back_propagation_cuda) const
{
    // Inputs

    const Index batch_size = input_pairs_device[0].second[0];

    const Index outputs_number = get_outputs_number();

    type* deltas_device = deltas_pair_device[0].first;

    // Back propagation

    FlattenBackPropagationCuda* flatten_layer_back_propagation_cuda =
        static_cast<FlattenBackPropagationCuda*>(back_propagation_cuda.get());

    type* input_derivatives = flatten_layer_back_propagation_cuda->input_derivatives;

    reorganize_deltas_cuda(deltas_device, input_derivatives, batch_size, outputs_number);
}


// CUDA structs

FlattenForwardPropagationCuda::FlattenForwardPropagationCuda(const Index& new_batch_samples_number, Layer* new_layer)
    : LayerForwardPropagationCuda()
{
    set(new_batch_samples_number, new_layer);
}


void FlattenForwardPropagationCuda::set(const Index& new_batch_samples_number, Layer* new_layer)
{
    if (new_batch_samples_number == 0) return;

    layer = new_layer;

    batch_size = new_batch_samples_number;

    const Flatten* flatten_layer = static_cast<Flatten*>(layer);

    const Index inputs_number = flatten_layer->get_inputs_number();
    const Index outputs_number = flatten_layer->get_outputs_number();

    // Reordered inputs

    if (cudaMalloc(&reordered_inputs, batch_size * inputs_number * sizeof(float)) != cudaSuccess)
        cout << "outputs allocation error" << endl;

    // Outputs

    if (cudaMalloc(&outputs, batch_size * outputs_number * sizeof(float)) != cudaSuccess)
        cout << "outputs allocation error" << endl;
}


void FlattenForwardPropagationCuda::print() const
{
    const dimensions output_dimensions = layer->get_output_dimensions();

    cout << "Flatten Outputs:" << endl
        << matrix_from_device(outputs, batch_size, output_dimensions[0]) << endl;
}


void FlattenForwardPropagationCuda::free()
{
    cudaFree(reordered_inputs);
    cudaFree(outputs);
}


pair<type*, dimensions> FlattenForwardPropagationCuda::get_outputs_pair_device() const
{
    const dimensions output_dimensions = layer->get_output_dimensions();

    return { (type*)outputs, {batch_size, output_dimensions[0]} };
}


FlattenBackPropagationCuda::FlattenBackPropagationCuda(const Index& new_batch_samples_number, Layer* new_layer)
    : LayerBackPropagationCuda()
{
    set(new_batch_samples_number, new_layer);
}


vector<pair<type*, dimensions>> FlattenBackPropagationCuda::get_input_derivative_pairs_device() const
{
    const Flatten* flatten_layer = static_cast<Flatten*>(layer);

    const dimensions input_dimensions = flatten_layer->get_input_dimensions();

    return { {input_derivatives, {batch_size, input_dimensions[0], input_dimensions[1], input_dimensions[2]}} };
}


void FlattenBackPropagationCuda::set(const Index& new_batch_samples_number, Layer* new_layer)
{
    if (new_batch_samples_number == 0) return;

    layer = new_layer;

    batch_size = new_batch_samples_number;

    const Flatten* flatten_layer = static_cast<Flatten*>(new_layer);

    dimensions input_dimensions = flatten_layer->get_input_dimensions();

    // Input derivatives

    if (cudaMalloc(&input_derivatives, batch_size * input_dimensions[0] * input_dimensions[1] * input_dimensions[2] * sizeof(float)) != cudaSuccess)
        cout << "input_derivatives flatten layer back propagation allocation error" << endl;
}


void FlattenBackPropagationCuda::print() const
{
    const dimensions input_dimensions = layer->get_input_dimensions();

    cout << "Flatten Input derivatives:" << endl
        << matrix_4d_from_device(input_derivatives, batch_size, input_dimensions[0], input_dimensions[1], input_dimensions[2]) << endl;
}


void FlattenBackPropagationCuda::free()
{
    cudaFree(input_derivatives);
}

#endif

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2023 Artificial Intelligence Techniques, SL.
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
