//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   P O O L I N G   L A Y E R   3 D   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "registry.h"
#include "tensor_utilities.h"
#include "pooling_layer_3d.h"

namespace opennn
{

Pooling3d::Pooling3d(const Shape& new_input_shape,
                     const PoolingMethod& new_pooling_method,
                     const string& new_name) : Layer()
{
    set(new_input_shape, new_pooling_method, new_name);
}


Shape Pooling3d::get_output_shape() const
{
    return {input_shape[1]};
}


Pooling3d::PoolingMethod Pooling3d::get_pooling_method() const
{
    return pooling_method;
}


string Pooling3d::write_pooling_method() const
{
    return pooling_method == PoolingMethod::MaxPooling ? "MaxPooling" : "AveragePooling";
}


void Pooling3d::set(const Shape& new_input_shape, const PoolingMethod& new_pooling_method, const string& new_label)
{
    name = "Pooling3d";
    input_shape = new_input_shape;
    pooling_method = new_pooling_method;
    set_label(new_label);
}


void Pooling3d::set_input_shape(const Shape& new_input_shape)
{
    input_shape = new_input_shape;
}


void Pooling3d::set_pooling_method(const PoolingMethod& new_pooling_method)
{
    pooling_method = new_pooling_method;
}


void Pooling3d::set_pooling_method(const string& new_pooling_method)
{
    if (new_pooling_method == "MaxPooling") pooling_method = PoolingMethod::MaxPooling;
    else if (new_pooling_method == "AveragePooling") pooling_method = PoolingMethod::AveragePooling;
    else throw runtime_error("Unknown pooling type: " + new_pooling_method);
}


void Pooling3d::forward_propagate(ForwardPropagation& forward_propagation, size_t, bool is_training)
{
/*
    const TensorMap3 inputs = tensor_map<3>(forward_propagation->inputs[0]);
    MatrixMap outputs = matrix_map(forward_propagation->outputs);

    auto* pooling_forward_propagation = static_cast<Pooling3dForwardPropagation*>(forward_propagation.get());

    const Index batch_size = inputs.dimension(0);
    const Index sequence_length = inputs.dimension(1);
    const Index features = inputs.dimension(2);

    if(pooling_method == PoolingMethod::MaxPooling)
    {
        MatrixI& maximal_indices = pooling_forward_propagation->maximal_indices;

        #pragma omp parallel for
        for(Index b = 0; b < batch_size; ++b)
        {
            outputs.row(b).setConstant(-numeric_limits<type>::infinity());

            for(Index s = 0; s < sequence_length; ++s)
            {
                for(Index f = 0; f < features; ++f)
                {
                    const type value = inputs(b, s, f);

                    if(value > outputs(b, f))
                    {
                        outputs(b, f) = value;
                        if(is_training) maximal_indices(b, f) = s;
                    }
                }
            }
        }
    }
    else // AveragePooling
    {
        #pragma omp parallel for
        for(Index b = 0; b < batch_size; ++b)
        {
            outputs.row(b).setZero();

            Index valid_count = 0;

            for(Index s = 0; s < sequence_length; ++s)
            {
                bool is_padding = true;

                for(Index f = 0; f < features; ++f)
                {
                    if(inputs(b, s, f) != type(0))
                    {
                        is_padding = false;
                        break;
                    }
                }

                if(!is_padding)
                {
                    for(Index f = 0; f < features; ++f)
                        outputs(b, f) += inputs(b, s, f);

                    ++valid_count;
                }
            }

            if(valid_count > 0)
                outputs.row(b) /= static_cast<type>(valid_count);
        }
    }
*/
}


void Pooling3d::back_propagate(ForwardPropagation& forward_propagation,
                               BackPropagation& back_propagation,
                               size_t index) const
{
/*
    const TensorMap3 inputs = tensor_map<3>(forward_propagation->inputs[0]);
    const MatrixMap delta = matrix_map(back_propagation->output_gradients[0]);

    Pooling3dForwardPropagation* pooling_forward_propagation = static_cast<Pooling3dForwardPropagation*>(forward_propagation.get());
    Pooling3dBackPropagation* pooling_back_propagation = static_cast<Pooling3dBackPropagation*>(back_propagation.get());

    TensorMap3 input_derivatives = tensor_map<3>(pooling_back_propagation->input_gradients[0]);

    input_derivatives.setZero();

    const Index batch_size = inputs.dimension(0);
    const Index sequence_length = inputs.dimension(1);
    const Index features = inputs.dimension(2);

    if(pooling_method == PoolingMethod::MaxPooling)
    {
        const MatrixI& maximal_indices = pooling_forward_propagation->maximal_indices;

        #pragma omp parallel for
        for(Index b = 0; b < batch_size; ++b)
        {
            for(Index f = 0; f < features; ++f)
            {
                const Index max_idx = maximal_indices(b, f);
                input_derivatives(b, max_idx, f) = delta(b, f);
            }
        }
    }
    else // AveragePooling
    {
        #pragma omp parallel for
        for(Index b = 0; b < batch_size; ++b)
        {
            Index valid_count = 0;

            for(Index s = 0; s < sequence_length; ++s)
            {
                bool is_padding = true;

                for(Index f = 0; f < features; ++f)
                {
                    if(inputs(b, s, f) != type(0))
                    {
                        is_padding = false;
                        break;
                    }
                }

                if(!is_padding)
                    ++valid_count;
            }

            if(valid_count == 0) continue;

            const type inverse_valid_count = type(1.0) / static_cast<type>(valid_count);

            for(Index s = 0; s < sequence_length; ++s)
            {
                bool is_padding = true;

                for(Index f = 0; f < features; ++f)
                {
                    if(inputs(b, s, f) != type(0))
                    {
                        is_padding = false;
                        break;
                    }
                }

                if(!is_padding)
                {
                    for(Index f = 0; f < features; ++f)
                        input_derivatives(b, s, f) = delta(b, f) * inverse_valid_count;
                }
            }
        }
    }
*/
}


void Pooling3d::to_XML(XMLPrinter& printer) const
{
    printer.OpenElement("Pooling3d");
    add_xml_element(printer, "InputDimensions", shape_to_string(get_input_shape()));
    add_xml_element(printer, "PoolingMethod", write_pooling_method());
    printer.CloseElement();
}


void Pooling3d::from_XML(const XMLDocument& document)
{
    const XMLElement* element = document.FirstChildElement("Pooling3d");
    if(!element) throw runtime_error("Pooling3d element is nullptr.");

    set_input_shape(string_to_shape(read_xml_string(element, "InputDimensions")));
    set_pooling_method(read_xml_string(element, "PoolingMethod"));
}


void Pooling3d::print() const
{
    cout << "Pooling3d layer" << endl
         << "Input shape: " << shape_to_string(input_shape) << endl
         << "Output shape: " << shape_to_string(get_output_shape()) << endl
         << "Pooling Method: " << write_pooling_method() << endl;
}

#ifdef CUDA

void Pooling3dForwardPropagationCuda::initialize()
{
    const Pooling3d* pooling_layer = static_cast<Pooling3d*>(layer);
    const Index features = pooling_layer->get_output_shape()[0];

    outputs.set_descriptor({batch_size, features});

    if (pooling_layer->get_pooling_method() == Pooling3d::PoolingMethod::MaxPooling)
        maximal_indices_device.resize({batch_size, features});
}


void Pooling3dForwardPropagationCuda::free()
{
    maximal_indices_device.free();
}


void Pooling3dBackPropagationCuda::initialize()
{
    const Pooling3d* pooling_layer = static_cast<Pooling3d*>(layer);

    const Shape layer_input_dimensions = pooling_layer->get_input_shape();
    const Index sequence_length = layer_input_dimensions[0];
    const Index features = layer_input_dimensions[1];

    input_gradients = {TensorView({batch_size, sequence_length, features})};
}

#endif

REGISTER(Layer, Pooling3d, "Pooling3d")

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
