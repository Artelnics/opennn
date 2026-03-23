//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   A D D I T I O N   L A Y E R   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#pragma once

#include "layer.h"
#include "tensor_utilities.h"

namespace opennn
{

template<int Rank> struct AdditionForwardPropagation;
template<int Rank> struct AdditionBackPropagation;

#ifdef OPENNN_CUDA
template<int Rank> struct AdditionForwardPropagationCuda;
template<int Rank> struct AdditionBackPropagationCuda;
#endif

template<int Rank>
class Addition final : public Layer
{

public:

    Addition(const Shape& new_input_shape = {}, const string& new_name = "")
    {
        set(new_input_shape, new_name);
    }

    Shape get_input_shape() const override
    {
        return input_shape;
    }

    Shape get_output_shape() const override
    {
        return input_shape;
    }


    void set(const Shape& new_input_shape, const string& new_label)
    {
        if(!new_input_shape.empty() && new_input_shape.size() != Rank)
            throw runtime_error("Input shape rank for AdditionLayer<" + to_string(Rank) + "> must be " + to_string(Rank));

        input_shape = new_input_shape;

        label = new_label;

        name = "Addition";
    }


    void forward_propagate(unique_ptr<LayerForwardPropagation>& forward_propagation,
                           bool) override
    {
        const TensorMapR<Rank> input_1 = tensor_map<Rank>(forward_propagation->inputs[0]);
        const TensorMapR<Rank> input_2 = tensor_map<Rank>(forward_propagation->inputs[1]);

        TensorMapR<Rank> outputs = tensor_map<Rank>(forward_propagation->outputs);

        outputs.device(get_device()) = input_1 + input_2;
    }


    void back_propagate(unique_ptr<LayerForwardPropagation>&,
                        unique_ptr<LayerBackPropagation>& back_propagation) const override
    {
        if (back_propagation->output_gradients.size() != 1)
            throw runtime_error(name + " backpropagation requires exactly one delta input.");

        const TensorMapR<Rank> output_gradients = tensor_map<Rank>(back_propagation->output_gradients[0]);

        TensorMapR<Rank> input_gradients_0 = tensor_map<Rank>(back_propagation->input_gradients[0]);
        TensorMapR<Rank> input_gradients_1 = tensor_map<Rank>(back_propagation->input_gradients[1]);

        input_gradients_0.device(get_device()) = output_gradients;
        input_gradients_1.device(get_device()) = output_gradients;
    }


    void from_XML(const XMLDocument& document) override
    {
        const XMLElement* element = document.FirstChildElement("Addition");
        if(!element) throw runtime_error(name + " element is nullptr.");

        const string new_label = read_xml_string(element, "Label");
        const Shape new_input_shape = string_to_shape(read_xml_string(element, "InputDimensions"));

        set(new_input_shape, new_label);
    }


    void to_XML(XMLPrinter& printer) const override
    {
        printer.OpenElement("Addition");

        add_xml_element(printer, "Label", label);
        add_xml_element(printer, "InputDimensions", shape_to_string(input_shape));

        printer.CloseElement();
    }

#ifdef OPENNN_CUDA

public:

    void forward_propagate(unique_ptr<LayerForwardPropagationCuda>& forward_propagation,
                           bool) override
    {
        const size_t total_elements = forward_propagation->batch_size * get_inputs_number();

        // @todo substitute addition_cuda by cudnn function similar as follows
/*
        const float alpha_minus_one = -1.0f;

        cudnnOpTensor(cudnn_handle,
                      operator_sum_descriptor,
                      &alpha_minus_one,
                      output_tensor_descriptor,
                      targets,
                      &alpha,
                      output_tensor_descriptor,
                      outputs,
                      &beta,
                      output_tensor_descriptor,
                      errors_device);

*/
        addition_cuda(total_elements,
                      forward_propagation->inputs[0].data,
                      forward_propagation->inputs[1].data,
                      forward_propagation->outputs.data);
    }


    void back_propagate(unique_ptr<LayerForwardPropagationCuda>&,
                        unique_ptr<LayerBackPropagationCuda>& back_propagation) const override
    {
        if (output_gradients.size() != 1)
            throw runtime_error(name + " backpropagation requires exactly one delta input for CUDA.");

        AdditionBackPropagationCuda<Rank>* this_back_propagation =
            static_cast<AdditionBackPropagationCuda<Rank>*>(back_propagation.get());

        const size_t inputs_number = get_inputs_number();
        const size_t total_elements = static_cast<size_t>(back_propagation->batch_size) * inputs_number;

        CHECK_CUDA(cudaMemcpy(this_back_propagation->input_gradients[0].data, output_gradients[0].data, total_elements * sizeof(type), cudaMemcpyDeviceToDevice));
        CHECK_CUDA(cudaMemcpy(this_back_propagation->input_gradients[1].data, output_gradients[0].data, total_elements * sizeof(type), cudaMemcpyDeviceToDevice));
    }

#endif

private:

    Shape input_shape;
};


template<int Rank>
struct AdditionForwardPropagation final : LayerForwardPropagation
{
    AdditionForwardPropagation(const Index new_batch_size = 0, Layer* new_layer = nullptr)
        : LayerForwardPropagation()
    {
        set(new_batch_size, new_layer);
    }


    void initialize() override
    {
        outputs.shape = Shape{batch_size}.append(layer->get_output_shape());
    }


    void print() const override
    {
        cout << "Addition Forward Propagation:" << endl
             << "Outputs shape: " << outputs.shape << endl
             << "Outputs data:" << endl << outputs.data << endl;
    }
};


template<int Rank>
struct AdditionBackPropagation final : LayerBackPropagation
{
    AdditionBackPropagation(const Index new_batch_size = 0, Layer* new_layer = nullptr)
        : LayerBackPropagation()
    {
        set(new_batch_size, new_layer);
    }

    void initialize() override
    {
        const Shape shape = Shape{batch_size}.append(layer->get_input_shape());

        input_gradients = {{nullptr,shape}, {nullptr,shape}};
    }


    void print() const override
    {
        cout << "Addition Back Propagation:" << endl;

        if(input_gradients.size() >= 1)
        {
            cout << "Input 1 Deltas shape: " << input_gradients[0].shape << endl;
            cout << input_gradients[0].data << endl;
        }

        if(input_gradients.size() >= 2)
        {
            cout << "Input 2 Deltas shape: " << input_gradients[1].shape << endl;
            cout << input_gradients[1].data << endl;
        }
    }
};


#ifdef OPENNN_CUDA

template<int Rank>
struct AdditionForwardPropagationCuda : public LayerForwardPropagationCuda
{
    AdditionForwardPropagationCuda(const Index new_batch_size = 0, Layer* new_layer = nullptr)
        : LayerForwardPropagationCuda()
    {
        set(new_batch_size, new_layer);
    }

    void initialize() override
    {
        outputs.set_descriptor(Shape{batch_size}.append(layer->get_output_shape()));
    }

    void print() const override
    {
        // @todo
    }
};


template<int Rank>
struct AdditionBackPropagationCuda : public LayerBackPropagationCuda
{
    AdditionBackPropagationCuda(const Index new_batch_size = 0, Layer* new_layer = nullptr)
        : LayerBackPropagationCuda()
    {
        set(new_batch_size, new_layer);
    }


    void initialize() override
    {        
        const Shape shape = Shape{batch_size}.append(layer->get_input_shape());

        input_gradients = {TensorViewCuda(shape), TensorViewCuda(shape)};
    }


    void print() const override
    {
        // @todo
    }
};

#endif // OPENNN_CUDA

void reference_addition_layer();

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
