//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   B O U N D I N G   L A Y E R   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "registry.h"
#include "tensors.h"
#include "bounding_layer.h"
#include "tinyxml2.h"

namespace opennn
{

Bounding::Bounding(const dimensions& output_dimensions, const string& new_name) : Layer()
{
    set(output_dimensions, new_name);
}


const Bounding::BoundingMethod& Bounding::get_bounding_method() const
{
    return bounding_method;
}


dimensions Bounding::get_input_dimensions() const
{
    return { lower_bounds.dimension(0) };
}


type Bounding::get_lower_bound(const Index& i) const
{
    return lower_bounds[i];
}


const Tensor<type, 1>& Bounding::get_lower_bounds() const
{
    return lower_bounds;
}


dimensions Bounding::get_output_dimensions() const
{
    return { lower_bounds.dimension(0) };
}


type Bounding::get_upper_bound(const Index& i) const
{
    return upper_bounds(i);
}


const Tensor<type, 1>& Bounding::get_upper_bounds() const
{
    return upper_bounds;
}


void Bounding::set(const dimensions& new_output_dimensions, const string& new_label)
{
    set_output_dimensions(new_output_dimensions);

    label = new_label;

    bounding_method = BoundingMethod::Bounding;

    name = "Bounding";

    is_trainable = false;
}


void Bounding::set_bounding_method(const BoundingMethod& new_method)
{
    bounding_method = new_method;
}


void Bounding::set_bounding_method(const string& new_method_string)
{
    if(new_method_string == "NoBounding" || new_method_string == "No bounding")
        bounding_method = BoundingMethod::NoBounding;
    else if(new_method_string == "Positive outputs" || new_method_string == "Data range" || new_method_string == "Bounding")
        bounding_method = BoundingMethod::Bounding;
    else
        throw runtime_error("Unknown bounding method: " + new_method_string + ".\n");
}


void Bounding::set_input_dimensions(const dimensions& new_input_dimensions)
{
    lower_bounds.resize(new_input_dimensions[0]);
    upper_bounds.resize(new_input_dimensions[0]);
}


void Bounding::set_lower_bound(const Index& index, const type& new_lower_bound)
{
    const dimensions output_dimensions = get_output_dimensions();

    if(lower_bounds.size() != output_dimensions[0])
    {
        lower_bounds.resize(output_dimensions[0]);
        lower_bounds.setConstant(-numeric_limits<type>::max());
    }

    lower_bounds[index] = new_lower_bound;
}


void Bounding::set_lower_bounds(const Tensor<type, 1>& new_lower_bounds)
{
    lower_bounds = new_lower_bounds;
}


void Bounding::set_output_dimensions(const dimensions& new_output_dimensions)
{
    lower_bounds.resize(new_output_dimensions[0]);
    upper_bounds.resize(new_output_dimensions[0]);

    lower_bounds.setConstant(-numeric_limits<type>::max());
    upper_bounds.setConstant(numeric_limits<type>::max());
}


void Bounding::set_upper_bounds(const Tensor<type, 1>& new_upper_bounds)
{
    upper_bounds = new_upper_bounds;
}


void Bounding::set_upper_bound(const Index& index, const type& new_upper_bound)
{
    const dimensions output_dimensions = get_output_dimensions();

    if(upper_bounds.size() != output_dimensions[0])
    {
        upper_bounds.resize(output_dimensions[0]);
        upper_bounds.setConstant(numeric_limits<type>::max());
    }

    upper_bounds[index] = new_upper_bound;
}


void Bounding::forward_propagate(const vector<TensorView>& input_views,
                                 unique_ptr<LayerForwardPropagation>& forward_propagation,
                                 const bool&)
{
    const TensorMap<Tensor<type,2>> inputs = tensor_map<2>(input_views[0]);

    BoundingForwardPropagation* this_forward_propagation =
        static_cast<BoundingForwardPropagation*>(forward_propagation.get());

    Tensor<type,2>& outputs = this_forward_propagation->outputs;

    if(bounding_method == BoundingMethod::NoBounding)
    {
        outputs.device(*thread_pool_device) = inputs;
        return;
    }

    const Index rows_number = inputs.dimension(0);
    const Index columns_number = inputs.dimension(1);

#pragma omp parallel for
    for (Index j = 0; j < columns_number; j++)
    {
        const type& lower_bound = lower_bounds(j);
        const type& upper_bound = upper_bounds(j);

        for (Index i = 0; i < rows_number; i++)
            outputs(i, j) = clamp(inputs(i, j), lower_bound, upper_bound);
    }
}


string Bounding::get_bounding_method_string() const
{
    if(bounding_method == BoundingMethod::Bounding)
        return "Bounding";
    else if(bounding_method == BoundingMethod::NoBounding)
        return "NoBounding";
    else
        throw runtime_error("Unknown bounding method.\n");
}


string Bounding::get_expression(const vector<string>& new_input_names, const vector<string>& new_output_names) const
{
    const vector<string> input_names = new_input_names.empty()
                                           ? get_default_input_names()
                                           : new_input_names;

    const vector<string> output_names = new_output_names.empty()
                                            ? get_default_output_names()
                                            : new_output_names;


    if (bounding_method == BoundingMethod::NoBounding)
        return string();

    ostringstream buffer;

    buffer.precision(10);

    const dimensions output_dimensions = get_output_dimensions();

    for(Index i = 0; i < output_dimensions[0]; i++)
        buffer << output_names[i] << " = max(" << lower_bounds[i] << ", " << input_names[i] << ")\n"
               << output_names[i] << " = min(" << upper_bounds[i] << ", " << output_names[i] << ")\n";

    return buffer.str();
}


void Bounding::print() const
{
    cout << "Bounding layer" << endl
         << "Lower bounds: " << lower_bounds << endl
         << "Upper bounds: " << upper_bounds << endl;
}


void Bounding::to_XML(XMLPrinter& printer) const
{
    printer.OpenElement("Bounding");

    const dimensions output_dimensions = get_input_dimensions();

    add_xml_element(printer, "NeuronsNumber", to_string(output_dimensions[0]));

    for (Index i = 0; i < output_dimensions[0]; i++)
    {
        printer.OpenElement("Item");
        printer.PushAttribute("Index", unsigned(i + 1));

        add_xml_element(printer, "LowerBound", to_string(lower_bounds[i]));
        add_xml_element(printer, "UpperBound", to_string(upper_bounds[i]));

        printer.CloseElement();
    }

    add_xml_element(printer, "BoundingMethod", get_bounding_method_string());

    printer.CloseElement();
}


void Bounding::from_XML(const XMLDocument& document)
{
    const auto* root_element = document.FirstChildElement("Bounding");

    if (!root_element)
        throw runtime_error("Bounding element is nullptr.\n");

    const Index neurons_number = read_xml_index(root_element, "NeuronsNumber");

    set({ neurons_number });

    const auto* item_element = root_element->FirstChildElement("Item");

    for (Index i = 0; i < neurons_number && item_element; i++)
    {
        unsigned index = 0;
        item_element->QueryUnsignedAttribute("Index", &index);

        if (index != i + 1)
            throw runtime_error("Index " + to_string(index) + " is incorrect.\n");

        lower_bounds[index - 1] = read_xml_type(item_element, "LowerBound");
        upper_bounds[index - 1] = read_xml_type(item_element, "UpperBound");

        item_element = item_element->NextSiblingElement("Item");
    }

    set_bounding_method(read_xml_string(root_element, "BoundingMethod"));
}


BoundingForwardPropagation::BoundingForwardPropagation(const Index& new_batch_size, Layer* new_layer)
    : LayerForwardPropagation()
{
    set(new_batch_size, new_layer);
}


TensorView BoundingForwardPropagation::get_output_pair() const
{
    const dimensions output_dimensions = layer->get_output_dimensions();

    return { (type*)outputs.data(), { batch_size, output_dimensions[0]}};
}


void BoundingForwardPropagation::initialize()
{
    const Index neurons_number = static_cast<Bounding*>(layer)->get_output_dimensions()[0];

    outputs.resize(batch_size, neurons_number);
}


void BoundingForwardPropagation::print() const
{
    cout << "Outputs:" << endl
         << outputs << endl;
}

REGISTER(Layer, Bounding, "Bounding")
REGISTER(LayerForwardPropagation, BoundingForwardPropagation, "Bounding")


#ifdef OPENNN_CUDA

void Bounding::forward_propagate_cuda(const vector<float*>& inputs_device,
                                      unique_ptr<LayerForwardPropagationCuda>& forward_propagation_cuda,
                                      const bool&)
{
    BoundingForwardPropagationCuda* this_forward_propagation =
        static_cast<BoundingForwardPropagationCuda*>(forward_propagation_cuda.get());

    const size_t size = get_outputs_number() * this_forward_propagation->batch_size * sizeof(float);

    // @todo Implement bounding in CUDA
}


BoundingForwardPropagationCuda::BoundingForwardPropagationCuda(const Index& new_batch_size, Layer* new_layer)
    : LayerForwardPropagationCuda()
{
    set(new_batch_size, new_layer);
}


void BoundingForwardPropagationCuda::set(const Index& new_batch_size, Layer* new_layer)
{
    if (!new_layer) return;

    layer = new_layer;
    batch_size = new_batch_size;

    const size_t size = layer->get_outputs_number() * batch_size;

    //cudaMalloc(&outputs, size * sizeof(float));
}


void BoundingForwardPropagationCuda::print() const
{
    const Index outputs_number = layer->get_outputs_number();

    cout << "Bounding CUDA Outputs (pass-through):" << endl
        << matrix_from_device(outputs, batch_size, outputs_number) << endl;
}


void BoundingForwardPropagationCuda::free()
{
    cudaFree(outputs);
    outputs = nullptr;
}

REGISTER(LayerForwardPropagationCuda, BoundingForwardPropagationCuda, "Bounding")

#endif

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
