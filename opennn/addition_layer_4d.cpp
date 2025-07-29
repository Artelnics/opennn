//   OpenNN: Open Neural Networks Library
//   www.opennnn.net
//
//   A D D I T I O N   L A Y E R   4 D   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com
/*
#include "registry.h"
#include "tensors.h"
#include "addition_layer_4d.h"

namespace opennn
{

Addition4d::Addition4d(const dimensions& new_input_dimensions, 
                       const string& new_name) : Layer()
{
     set(new_input_dimensions, new_name);
}


Index Addition4d::get_input_height() const
{
    return input_dimensions[0];
}


Index Addition4d::get_input_width() const
{
    return input_dimensions[1];
}


Index Addition4d::get_input_channels() const
{
    return input_dimensions[2];
}


dimensions Addition4d::get_input_dimensions() const
{
    return input_dimensions;
}


dimensions Addition4d::get_output_dimensions() const
{
    return input_dimensions;
}


void Addition4d::set(const dimensions& new_input_dimensions, 
                     const string& new_name)
{
    if (new_input_dimensions.size() != 3)
        throw runtime_error("Input dimensions for Addition4d layer must have 3 elements: {height, width, channels}.");

    input_dimensions = new_input_dimensions;

    label = new_name;

    name = "Addition4d";
}


void Addition4d::set_input_dimensions(const dimensions& new_input_dimensions)
{
    input_dimensions = new_input_dimensions;
}


void Addition4d::forward_propagate(const vector<pair<type*, dimensions>>& input_pairs,
                                   unique_ptr<LayerForwardPropagation>& layer_forward_propagation,
                                   const bool&)
{
    if (input_pairs.size() != 2)
        throw runtime_error("Addition4d layer requires exactly two inputs.");
    if (input_pairs[0].second != input_pairs[1].second)
        throw runtime_error("Input dimensions for Addition4d layer must be identical.");

    const TensorMap<Tensor<type, 4>> input_1 = tensor_map<4>(input_pairs[0]);
    const TensorMap<Tensor<type, 4>> input_2 = tensor_map<4>(input_pairs[1]);

    Addition4dForwardPropagation* this_forward_propagation =
        static_cast<Addition4dForwardPropagation*>(layer_forward_propagation.get());

    Tensor<type, 4>& outputs = this_forward_propagation->outputs;

    outputs.device(*thread_pool_device) = input_1 + input_2;
}


void Addition4d::back_propagate(const vector<pair<type*, dimensions>>&,
                                const vector<pair<type*, dimensions>>& delta_pairs,
                                unique_ptr<LayerForwardPropagation>&,
                                unique_ptr<LayerBackPropagation>& back_propagation) const
{
    if (delta_pairs.size() != 1)
        throw runtime_error("Addition4d backpropagation requires exactly one delta input.");

    const TensorMap<Tensor<type, 4>> deltas = tensor_map<4>(delta_pairs[0]);

    Addition4dBackPropagation* this_back_propagation =
        static_cast<Addition4dBackPropagation*>(back_propagation.get());

    Tensor<type, 4>& input_1_derivatives = this_back_propagation->input_1_derivatives;
    Tensor<type, 4>& input_2_derivatives = this_back_propagation->input_2_derivatives;

    input_1_derivatives.device(*thread_pool_device) = deltas;
    input_2_derivatives.device(*thread_pool_device) = deltas;
}


void Addition4d::from_XML(const XMLDocument& document)
{
    const XMLElement* addition_layer_element = document.FirstChildElement("Addition4d");
    if (!addition_layer_element)
        throw runtime_error("Addition4d element is nullptr.\n");

    const string new_label = read_xml_string(addition_layer_element, "Label");
    const dimensions new_dims = string_to_dimensions(read_xml_string(addition_layer_element, "InputDimensions"));

    set(new_dims, new_label);
}


void Addition4d::to_XML(XMLPrinter& printer) const
{
    printer.OpenElement("Addition4d");
    add_xml_element(printer, "Label", label);
    add_xml_element(printer, "InputDimensions", dimensions_to_string(input_dimensions));
    printer.CloseElement();
}


Addition4dForwardPropagation::Addition4dForwardPropagation(const Index& new_batch_size, Layer* new_layer)
    : LayerForwardPropagation()
{
    set(new_batch_size, new_layer);
}


pair<type*, dimensions> Addition4dForwardPropagation::get_output_pair() const
{
    const Addition4d* addition_layer = static_cast<Addition4d*>(layer);

    const dimensions dims = addition_layer->get_output_dimensions();

    return { (type*)outputs.data(), {batch_size, dims[0], dims[1], dims[2]} };
}


void Addition4dForwardPropagation::set(const Index& new_batch_size, Layer* new_layer)
{
    if (!new_layer) return;

    layer = new_layer;

    batch_size = new_batch_size;

    const Addition4d* addition_layer = static_cast<Addition4d*>(layer);

    outputs.resize(batch_size,
                   addition_layer->get_input_height(),
                   addition_layer->get_input_width(),
                   addition_layer->get_input_channels());
}


void Addition4dForwardPropagation::print() const
{
    cout << "Addition4dForwardPropagation:" << endl;
    cout << "Outputs dimensions: " << outputs.dimensions() << endl;
}


Addition4dBackPropagation::Addition4dBackPropagation(const Index& new_batch_size, Layer* new_layer)
    : LayerBackPropagation()
{
    set(new_batch_size, new_layer);
}


vector<pair<type*, dimensions>> Addition4dBackPropagation::get_input_derivative_pairs() const
{
    const Addition4d* addition_layer = static_cast<Addition4d*>(layer);

    const dimensions dims = addition_layer->get_input_dimensions();

    const dimensions full_dims = { batch_size, dims[0], dims[1], dims[2] };

    return { {(type*)input_1_derivatives.data(), full_dims},
             {(type*)input_2_derivatives.data(), full_dims} };
}


void Addition4dBackPropagation::set(const Index& new_batch_size, Layer* new_layer)
{
    if (!new_layer) return;

    layer = new_layer;

    batch_size = new_batch_size;

    const Addition4d* addition_layer = static_cast<Addition4d*>(layer);

    const Index height = addition_layer->get_input_height();
    const Index width = addition_layer->get_input_width();
    const Index channels = addition_layer->get_input_channels();

    input_1_derivatives.resize(batch_size, height, width, channels);

    input_2_derivatives.resize(batch_size, height, width, channels);    
}

   
void Addition4dBackPropagation::print() const   
{
    cout << "Addition4dBackPropagation:" << endl;
    cout << "input_1_derivatives dimensions: " << input_1_derivatives.dimensions() << endl;
    cout << "input_2_derivatives dimensions: " << input_2_derivatives.dimensions() << endl;
}


#ifdef OPENNN_CUDA

void Addition4d::forward_propagate_cuda(const vector<float*>& inputs_device,
                                        unique_ptr<LayerForwardPropagationCuda>& forward_propagation_cuda,
                                        const bool&)
{
    if (inputs_device.size() != 2)
        throw runtime_error("Addition4d layer requires exactly two inputs for CUDA propagation.");

    const type* input_1_device = inputs_device[0];
    const type* input_2_device = inputs_device[1];

    Addition4dForwardPropagationCuda* addition_layer_3d_forward_propagation_cuda =
        static_cast<Addition4dForwardPropagationCuda*>(forward_propagation_cuda.get());
    
    type* outputs_device = addition_layer_3d_forward_propagation_cuda->outputs;

    const size_t total_elements = static_cast<size_t>(addition_layer_3d_forward_propagation_cuda->batch_size) *
                                  get_input_height() *
                                  get_input_width() *
                                  get_input_channels();

    addition_cuda(total_elements, input_1_device, input_2_device, outputs_device);
}


void Addition4d::back_propagate_cuda(const vector<float*>&,
                                     const vector<float*>& deltas_device,
                                     unique_ptr<LayerForwardPropagationCuda>&,
                                     unique_ptr<LayerBackPropagationCuda>& back_propagation_cuda) const
{
    if (deltas_device.size() != 1)
        throw runtime_error("Addition4d backpropagation requires exactly one delta input for CUDA.");

    Addition4dBackPropagationCuda* addition_layer_3d_back_propagation =
        static_cast<Addition4dBackPropagationCuda*>(back_propagation_cuda.get());

    type* inputs_1_derivatives = addition_layer_3d_back_propagation->inputs_1_derivatives;
    type* inputs_2_derivatives = addition_layer_3d_back_propagation->inputs_2_derivatives;

    const size_t total_elements = static_cast<size_t>(addition_layer_3d_back_propagation->batch_size) *
                                  get_input_height() *
                                  get_input_width() *
                                  get_input_channels();

    CHECK_CUDA(cudaMemcpy(inputs_1_derivatives, deltas_device[0], total_elements * sizeof(type), cudaMemcpyDeviceToDevice));
    CHECK_CUDA(cudaMemcpy(inputs_2_derivatives, deltas_device[0], total_elements * sizeof(type), cudaMemcpyDeviceToDevice));
}


Addition4dForwardPropagationCuda::Addition4dForwardPropagationCuda(const Index& new_batch_size, Layer* new_layer)
    : LayerForwardPropagationCuda()
{
    set(new_batch_size, new_layer);
}


void Addition4dForwardPropagationCuda::set(const Index& new_batch_size, Layer* new_layer)
{
    if (!new_layer) return;

    cout << "Addition4dForwardPropagationCuda set:" << endl;
    
    batch_size = new_batch_size;

    layer = new_layer;
 
    const Addition4d* addition_layer = static_cast<const Addition4d*>(layer);

    const size_t total_elements = static_cast<size_t>(batch_size) *
                                  addition_layer->get_input_height() *
                                  addition_layer->get_input_width() *
                                  addition_layer->get_input_channels();

    //CHECK_CUDA(cudaMalloc(&outputs, total_elements * sizeof(float)));
    CUDA_MALLOC_AND_REPORT(outputs, total_elements * sizeof(type));
}


void Addition4dForwardPropagationCuda::print() const 
{ 

}


void Addition4dForwardPropagationCuda::free()
{
    if (outputs) cudaFree(outputs);

    outputs = nullptr;
}


Addition4dBackPropagationCuda::Addition4dBackPropagationCuda(const Index& new_batch_size, Layer* new_layer)
    : LayerBackPropagationCuda()
{
    set(new_batch_size, new_layer);
}


vector<float*> Addition4dBackPropagationCuda::get_input_derivatives_device()
{
    return { inputs_1_derivatives, inputs_2_derivatives };
}


void Addition4dBackPropagationCuda::set(const Index& new_batch_size, Layer* new_layer)
{
    if (!new_layer) return;

    cout << "Addition4dBackPropagationCuda set:" << endl;
    
    batch_size = new_batch_size;
        
    layer = new_layer;

    const Addition4d* addition_layer = static_cast<const Addition4d*>(layer);
        
    const size_t total_elements = static_cast<size_t>(batch_size) *
                                  addition_layer->get_input_height() *
                                  addition_layer->get_input_width() *
                                  addition_layer->get_input_channels();

    //CHECK_CUDA(cudaMalloc(&inputs_1_derivatives, total_elements * sizeof(float)));
    CUDA_MALLOC_AND_REPORT(inputs_1_derivatives, total_elements * sizeof(type));
    //CHECK_CUDA(cudaMalloc(&inputs_2_derivatives, total_elements * sizeof(float)));
    CUDA_MALLOC_AND_REPORT(inputs_2_derivatives, total_elements * sizeof(type));
}


void Addition4dBackPropagationCuda::print() const
{ 

}


void Addition4dBackPropagationCuda::free()
{
    if (inputs_1_derivatives) cudaFree(inputs_1_derivatives);
    if (inputs_2_derivatives) cudaFree(inputs_2_derivatives);

    inputs_1_derivatives = nullptr;
    inputs_2_derivatives = nullptr;
}


    REGISTER(LayerForwardPropagationCuda, Addition4dForwardPropagationCuda, "Addition4d")
    REGISTER(LayerBackPropagationCuda, Addition4dBackPropagationCuda, "Addition4d")

#endif // OPENNN_CUDA

    REGISTER(Layer, Addition4d, "Addition4d")
    REGISTER(LayerForwardPropagation, Addition4dForwardPropagation, "Addition4d")
    REGISTER(LayerBackPropagation, Addition4dBackPropagation, "Addition4d")

} // namespace opennn
*/
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
