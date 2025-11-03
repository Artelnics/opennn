//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   S C A L I N G   L A Y E R   4 D   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "registry.h"
#include "tensors.h"
#include "scaling_layer_4d.h"

namespace opennn
{

Scaling4d::Scaling4d(const dimensions& new_input_dimensions) : Layer()
{
    set(new_input_dimensions);
}


dimensions Scaling4d::get_input_dimensions() const
{
    return input_dimensions;
}


dimensions Scaling4d::get_output_dimensions() const
{
    return input_dimensions;
}


void Scaling4d::set(const dimensions& new_input_dimensions)
{
    input_dimensions = new_input_dimensions;

    name = "Scaling4d";

    label = "scaling_layer_4d";

    is_trainable = false;
}


void Scaling4d::forward_propagate(const vector<TensorView>& input_views,
                                  unique_ptr<LayerForwardPropagation>& forward_propagation,
                                  const bool&)
{
    Scaling4dForwardPropagation* this_forward_propagation =
        static_cast<Scaling4dForwardPropagation*>(forward_propagation.get());

    const TensorMap<Tensor<type, 4>> inputs = tensor_map<4>(input_views[0]);

    Tensor<type, 4>& outputs = this_forward_propagation->outputs;

    outputs.device(*thread_pool_device) = inputs/type(255);
}


void Scaling4d::print() const
{
    cout << "Scaling layer 4D" << endl;

    print_vector(input_dimensions);
}


void Scaling4d::to_XML(XMLPrinter& printer) const
{
    printer.OpenElement("Scaling4d");

    add_xml_element(printer, "InputDimensions", dimensions_to_string(input_dimensions));

    printer.CloseElement();
}


void Scaling4d::from_XML(const XMLDocument& document)
{
    const XMLElement* scaling_layer_element = document.FirstChildElement("Scaling4d");

    if(!scaling_layer_element)
        throw runtime_error("Scaling layer element is nullptr.\n");

    set(string_to_dimensions(read_xml_string(scaling_layer_element, "InputDimensions")));

}


Scaling4dForwardPropagation::Scaling4dForwardPropagation(const Index& new_batch_size, Layer* new_layer)
    : LayerForwardPropagation()
{
    set(new_batch_size, new_layer);
}


TensorView Scaling4dForwardPropagation::get_output_pair() const
{
    const Scaling4d* scaling_layer_4d = static_cast<Scaling4d*>(layer);

    const dimensions output_dimensions = scaling_layer_4d->get_output_dimensions();

    return {(type*)outputs.data(), {batch_size, output_dimensions[0], output_dimensions[1], output_dimensions[2]}};
}


void Scaling4dForwardPropagation::initialize()
{
    const dimensions output_dimensions = layer->get_output_dimensions();

    outputs.resize(batch_size, output_dimensions[0], output_dimensions[1], output_dimensions[2]);
}


void Scaling4dForwardPropagation::print() const
{
    cout << "Scaling Outputs:" << endl
         << outputs.dimensions() << endl;
}


#ifdef  OPENNN_CUDA

void Scaling4d::forward_propagate_cuda(const vector<float*>& inputs_device,
                                       unique_ptr<LayerForwardPropagationCuda>& forward_propagation_cuda,
                                       const bool&)
{
    Scaling4dForwardPropagationCuda* scaling_layer_forward_propagation =
        static_cast<Scaling4dForwardPropagationCuda*>(forward_propagation_cuda.get());

    const size_t size = get_inputs_number() * scaling_layer_forward_propagation->batch_size;

    type* outputs_device = scaling_layer_forward_propagation->outputs;
    type* scalar_device = scaling_layer_forward_propagation->scalar_device;

    division(size, inputs_device[0], scalar_device, outputs_device);
}


// CUDA structs

Scaling4dForwardPropagationCuda::Scaling4dForwardPropagationCuda(const Index& new_batch_size, Layer* new_layer)
    : LayerForwardPropagationCuda()
{
    set(new_batch_size, new_layer);
}


void Scaling4dForwardPropagationCuda::set(const Index& new_batch_size, Layer* new_layer)
{
    if (!new_layer) return;

    layer = new_layer;
    batch_size = new_batch_size;

    const Scaling4d* flatten_layer = static_cast<Scaling4d*>(layer);

    const Index outputs_number = flatten_layer->get_outputs_number();

    const size_t size = batch_size * outputs_number;

    Tensor<float, 1> scalar_host(size);
    scalar_host.setConstant(255);

    CHECK_CUDA(cudaMalloc(&scalar_device, size * sizeof(float)));
    //CUDA_MALLOC_AND_REPORT(scalar_device, size * sizeof(float));

    CHECK_CUDA(cudaMemcpy(scalar_device, scalar_host.data(), size * sizeof(float), cudaMemcpyHostToDevice));

    CHECK_CUDA(cudaMalloc(&outputs, size * sizeof(float)));
    //CUDA_MALLOC_AND_REPORT(outputs, size * sizeof(float));
}


void Scaling4dForwardPropagationCuda::print() const
{
    const dimensions output_dimensions = layer->get_output_dimensions();

    cout << "Scaling4d Outputs:" << endl
         << matrix_from_device(outputs, batch_size, output_dimensions[0]) << endl;
}


void Scaling4dForwardPropagationCuda::free()
{
    cudaFree(scalar_device);
    cudaFree(outputs);

    scalar_device = nullptr;
    outputs = nullptr;
}


REGISTER(LayerForwardPropagationCuda, Scaling4dForwardPropagationCuda, "Scaling4d")

#endif

REGISTER(Layer, Scaling4d, "Scaling4d")
REGISTER(LayerForwardPropagation, Scaling4dForwardPropagation, "Scaling4d")

}

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
