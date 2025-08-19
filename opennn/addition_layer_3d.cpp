//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   A D D I T I O N   L A Y E R   3 D   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "registry.h"
#include "addition_layer_3d.h"

namespace opennn
{
/*
Addition3d::Addition3d(const dimensions& new_input_dimensions,
                       const string& new_name) : Layer()
{
    set(new_input_dimensions[0], new_input_dimensions[1], new_name);
}


Index Addition3d::get_sequence_length() const
{
    return sequence_length;
}


Index Addition3d::get_embedding_dimension() const
{
    return embedding_dimension;
}


dimensions Addition3d::get_input_dimensions() const
{
    return { sequence_length, embedding_dimension };
}


dimensions Addition3d::get_output_dimensions() const
{
    return { sequence_length, embedding_dimension };
}


void Addition3d::set(const Index& new_sequence_length,
                     const Index& new_embedding_dimension,
                     const string& new_label)
{
    sequence_length = new_sequence_length;

    embedding_dimension = new_embedding_dimension;

    label = new_label;

    name = "Addition3d";
}


void Addition3d::forward_propagate(const vector<TensorView>& input_views,
                                   unique_ptr<LayerForwardPropagation>& layer_forward_propagation,
                                   const bool&)
{
    const TensorMap<Tensor<type, 3>> positional_encodings = tensor_map<3>(input_views[0]);

    const TensorMap<Tensor<type, 3>> input_embeddings = tensor_map<3>(input_views[1]);

    Addition3dForwardPropagation* this_forward_propagation =
        static_cast<Addition3dForwardPropagation*>(layer_forward_propagation.get());

    Tensor<type, 3>& outputs = this_forward_propagation->outputs;

    outputs.device(*thread_pool_device) = positional_encodings + input_embeddings;
}


void Addition3d::back_propagate(const vector<TensorView>&,
                                const vector<TensorView>& delta_views,
                                unique_ptr<LayerForwardPropagation>&,
                                unique_ptr<LayerBackPropagation>& back_propagation) const
{
    const TensorMap<Tensor<type, 3>> deltas = tensor_map<3>(delta_views[0]);

    // Back propagation

    Addition3dBackPropagation* addition_3d_back_propagation =
        static_cast<Addition3dBackPropagation*>(back_propagation.get());

    Tensor<type, 3>& input_1_derivatives = addition_3d_back_propagation->input_1_derivatives;

    Tensor<type, 3>& input_2_derivatives = addition_3d_back_propagation->input_2_derivatives;

    input_1_derivatives.device(*thread_pool_device) = deltas;
    input_2_derivatives.device(*thread_pool_device) = deltas;
}


void Addition3d::from_XML(const XMLDocument& document)
{
    const auto* addition_layer_element = document.FirstChildElement("Addition3d");

    if (!addition_layer_element)
        throw runtime_error("Addition3d element is nullptr.\n");

    const string new_name = read_xml_string(addition_layer_element, "Name");
    const Index new_sequence_lenght = read_xml_index(addition_layer_element, "SequenceLength");
    const Index new_embedding_legth = read_xml_index(addition_layer_element, "EmbeddingLength");

    set(new_sequence_lenght, new_embedding_legth, new_name);
}


void Addition3d::to_XML(XMLPrinter& printer) const
{
    printer.OpenElement("Addition3d");

    add_xml_element(printer, "Label", label);
    add_xml_element(printer, "SequenceLength", to_string(get_sequence_length()));
    add_xml_element(printer, "EmbeddingLength", to_string(get_embedding_dimension()));

    printer.CloseElement();
}


Addition3dForwardPropagation::Addition3dForwardPropagation(const Index& new_batch_size, 
                                                                     Layer* new_layer)
    : LayerForwardPropagation()
{
    set(new_batch_size, new_layer);
}


TensorView Addition3dForwardPropagation::get_output_pair() const
{
    Addition3d* addition_3d = static_cast<Addition3d*>(layer);

    const Index sequence_length = addition_3d->get_sequence_length();
    const Index embedding_dimension = addition_3d->get_embedding_dimension();

    return {(type*)outputs.data(), {batch_size, sequence_length, embedding_dimension}};
}


void Addition3dForwardPropagation::set(const Index& new_batch_size, Layer* new_layer)
{
    if (!new_layer) return;

    layer = new_layer;

    Addition3d* addition_layer_3d = static_cast<Addition3d*>(layer);

    batch_size = new_batch_size;

    const Index sequence_length = addition_layer_3d->get_sequence_length();
    const Index embedding_dimension = addition_layer_3d->get_embedding_dimension();

    outputs.resize(batch_size, sequence_length, embedding_dimension);
}


void Addition3dForwardPropagation::print() const
{
    cout << "Addition3dForwardPropagation:" << endl;

    cout << "Outputs:" << endl
         << outputs << endl;
}


void Addition3dBackPropagation::set(const Index& new_batch_size, Layer* new_layer)
{
    if (!new_layer) return;

    batch_size = new_batch_size;

    layer = new_layer;

    Addition3d* addition_layer_3d = static_cast<Addition3d*>(layer);

    const Index sequence_length = addition_layer_3d->get_sequence_length();
    const Index embedding_dimension = addition_layer_3d->get_embedding_dimension();

    input_1_derivatives.resize(batch_size, sequence_length, embedding_dimension);
    input_2_derivatives.resize(batch_size, sequence_length, embedding_dimension);
}


void Addition3dBackPropagation::print() const
{
    cout << "Addition3dBackPropagation:" << endl;

    cout << "input_1_derivatives:" << endl
        << input_1_derivatives << endl;
    cout << "input_2_derivatives:" << endl
        << input_2_derivatives << endl;
}


Addition3dBackPropagation::Addition3dBackPropagation(const Index& new_batch_size, Layer* new_layer)
    : LayerBackPropagation()
{
    set(new_batch_size, new_layer);
}


vector<TensorView> Addition3dBackPropagation::get_input_derivative_views() const
{
    Addition3d* addition_layer_3d = static_cast<Addition3d*>(layer);

    const Index sequence_length = addition_layer_3d->get_sequence_length();
    const Index embedding_dimension = addition_layer_3d->get_embedding_dimension();

    return
    {{(type*)input_1_derivatives.data(), {batch_size, sequence_length, embedding_dimension}},
     {(type*)input_2_derivatives.data(), {batch_size, sequence_length, embedding_dimension}}};
}
*/
#ifdef OPENNN_CUDA
/*
void Addition3d::forward_propagate_cuda(const vector<float*>& inputs_device,
                                        unique_ptr<LayerForwardPropagationCuda>& forward_propagation_cuda,
                                        const bool& is_training)
{
    Addition3dForwardPropagationCuda* addition_layer_3d_forward_propagation_cuda
        = static_cast<Addition3dForwardPropagationCuda*>(forward_propagation_cuda.get());

    const Index batch_size = addition_layer_3d_forward_propagation_cuda->batch_size;

    type* outputs = addition_layer_3d_forward_propagation_cuda->outputs;

    const size_t total_elements = batch_size * sequence_length * embedding_dimension;

    addition_cuda(total_elements, inputs_device[0], inputs_device[1], outputs);
}


void Addition3d::back_propagate_cuda(const vector<float*>&,
                                     const vector<float*>& deltas_device,
                                     unique_ptr<LayerForwardPropagationCuda>& forward_propagation_cuda,
                                     unique_ptr<LayerBackPropagationCuda>& back_propagation_cuda) const
{
    Addition3dBackPropagationCuda* addition_layer_3d_back_propagation =
        static_cast<Addition3dBackPropagationCuda*>(back_propagation_cuda.get());

    const Index batch_size = addition_layer_3d_back_propagation->batch_size;

    type* inputs_1_derivatives = addition_layer_3d_back_propagation->inputs_1_derivatives;
    type* inputs_2_derivatives = addition_layer_3d_back_propagation->inputs_2_derivatives;

    const size_t total_elements = batch_size * sequence_length * embedding_dimension;

    CHECK_CUDA(cudaMemcpy(inputs_1_derivatives, deltas_device[0], total_elements * sizeof(type), cudaMemcpyDeviceToDevice));
    CHECK_CUDA(cudaMemcpy(inputs_2_derivatives, deltas_device[0], total_elements * sizeof(type), cudaMemcpyDeviceToDevice));
}


// CUDA structs

Addition3dForwardPropagationCuda::Addition3dForwardPropagationCuda(const Index& new_batch_size, Layer* new_layer)
    : LayerForwardPropagationCuda()
{
    set(new_batch_size, new_layer);
}


void Addition3dForwardPropagationCuda::set(const Index& new_batch_size, Layer* new_layer)
{
    if (!new_layer) return;

    cout << "Addition3dForwardPropagationCuda set:" << endl;

    batch_size = new_batch_size;

    layer = new_layer;

    const Addition3d* addition_layer = static_cast<const Addition3d*>(layer);

    const size_t total_elements = static_cast<size_t>(batch_size) *
        addition_layer->get_sequence_length() *
        addition_layer->get_embedding_dimension();

    // Outputs

    //CHECK_CUDA(cudaMalloc(&outputs, total_elements * sizeof(float)));
    CUDA_MALLOC_AND_REPORT(outputs, total_elements * sizeof(float));
}


void Addition3dForwardPropagationCuda::print() const
{
    cout << "Addition3dForwardPropagationCuda:" << endl;

    const Addition3d* addition_layer = static_cast<const Addition3d*>(layer);

    cout << "Outputs dimensions:" << endl
        << "[ " << batch_size << " , " << addition_layer->get_sequence_length() << " , " << addition_layer->get_embedding_dimension() << " ]" << endl;

    cout << "Outputs:" << endl
        << matrix_3d_from_device(outputs, batch_size, addition_layer->get_sequence_length(), addition_layer->get_embedding_dimension()) << endl;
}


void Addition3dForwardPropagationCuda::free()
{
    if (outputs) cudaFree(outputs);
    outputs = nullptr;
}


Addition3dBackPropagationCuda::Addition3dBackPropagationCuda(const Index& new_batch_size, Layer* new_layer)
    : LayerBackPropagationCuda()
{
    set(new_batch_size, new_layer);
}


vector<float*> Addition3dBackPropagationCuda::get_input_derivatives_device()
{
    return { inputs_1_derivatives, inputs_2_derivatives };
}


void Addition3dBackPropagationCuda::set(const Index& new_batch_size, Layer* new_layer)
{
    if (!new_layer) return;

    cout << "Addition3dBackPropagationCuda set:" << endl;

    batch_size = new_batch_size;

    layer = new_layer;

    const Addition3d* addition_layer = static_cast<const Addition3d*>(layer);

    const size_t total_elements = static_cast<size_t>(batch_size) *
        addition_layer->get_sequence_length() *
        addition_layer->get_embedding_dimension();

    // Input derivatives

    //CHECK_CUDA(cudaMalloc(&inputs_1_derivatives, total_elements * sizeof(float)));
    CUDA_MALLOC_AND_REPORT(inputs_1_derivatives, total_elements * sizeof(float));
    //CHECK_CUDA(cudaMalloc(&inputs_2_derivatives, total_elements * sizeof(float)));
    CUDA_MALLOC_AND_REPORT(inputs_2_derivatives, total_elements * sizeof(float));
}


void Addition3dBackPropagationCuda::print() const
{
    cout << "Addition3dBackPropagationCuda:" << endl;

    const Addition3d* addition_layer = static_cast<const Addition3d*>(layer);

    cout << "input derivatives dimensions:" << endl
        << "[ "<< batch_size << " , " << addition_layer->get_sequence_length() << " , " << addition_layer->get_embedding_dimension() << " ]" << endl;
}


void Addition3dBackPropagationCuda::free()
{
    if (inputs_1_derivatives) cudaFree(inputs_1_derivatives);
    if (inputs_2_derivatives) cudaFree(inputs_2_derivatives);

    inputs_1_derivatives = nullptr;
    inputs_2_derivatives = nullptr;
}
*/

#endif

    using Addition3d = Addition<3>;
    using Addition4d = Addition<4>;

    using AdditionForwardPropagation3d = AdditionForwardPropagation<3>;
    using AdditionForwardPropagation4d = AdditionForwardPropagation<4>;

    using AdditionBackPropagation3d = AdditionBackPropagation<3>;
    using AdditionBackPropagation4d = AdditionBackPropagation<4>;

    REGISTER(Layer, Addition3d, "Addition3d")
    REGISTER(LayerForwardPropagation, AdditionForwardPropagation3d, "Addition3d")
    REGISTER(LayerBackPropagation, AdditionBackPropagation3d, "Addition3d")

    REGISTER(Layer, Addition4d, "Addition4d")
    REGISTER(LayerForwardPropagation, AdditionForwardPropagation4d, "Addition4d")
    REGISTER(LayerBackPropagation, AdditionBackPropagation4d, "Addition4d")

#ifdef OPENNN_CUDA

    using AdditionForwardPropagationCuda3d = AdditionForwardPropagationCuda<3>;
    using AdditionBackPropagationCuda3d = AdditionBackPropagationCuda<3>;

    using AdditionForwardPropagationCuda4d = AdditionForwardPropagationCuda<4>;
    using AdditionBackPropagationCuda4d = AdditionBackPropagationCuda<4>;

    REGISTER(LayerForwardPropagationCuda, AdditionForwardPropagationCuda3d, "Addition3d")
    REGISTER(LayerBackPropagationCuda, AdditionBackPropagationCuda3d, "Addition3d")

    REGISTER(LayerForwardPropagationCuda, AdditionForwardPropagationCuda4d, "Addition4d")
    REGISTER(LayerBackPropagationCuda, AdditionBackPropagationCuda4d, "Addition4d")

#endif

    template class Addition<3>;
    template class Addition<4>;

    template struct AdditionForwardPropagation<3>;
    template struct AdditionForwardPropagation<4>;

    template struct AdditionBackPropagation<3>;
    template struct AdditionBackPropagation<4>;

#ifdef OPENNN_CUDA

    template struct AdditionForwardPropagationCuda<3>;
    template struct AdditionForwardPropagationCuda<4>;

    template struct AdditionBackPropagationCuda<3>;
    template struct AdditionBackPropagationCuda<4>;

#endif

    // Linker fix: Ensures the static registration macros in this file are run.
    void reference_addition_layer() { }

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
