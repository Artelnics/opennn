//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   P O O L I N G   L A Y E R   3 D   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "pooling_layer_3d.h"
#include "tensors.h"

namespace opennn
{

Pooling3d::Pooling3d(const dimensions& new_input_dimensions,
                     const PoolingMethod& new_pooling_method,
                     const string& new_name) : Layer()
{
    set(new_input_dimensions, new_pooling_method, new_name);
}


dimensions Pooling3d::get_input_dimensions() const
{
    return input_dimensions;
}


dimensions Pooling3d::get_output_dimensions() const
{
    return {input_dimensions[1]};
}


Pooling3d::PoolingMethod Pooling3d::get_pooling_method() const
{
    return pooling_method;
}


string Pooling3d::write_pooling_method() const
{
    return pooling_method == PoolingMethod::MaxPooling ? "MaxPooling" : "AveragePooling";
}


void Pooling3d::set(const dimensions& new_input_dimensions, const PoolingMethod& new_pooling_method, const string& new_name)
{
    layer_type = Layer::Type::Pooling3d;
    input_dimensions = new_input_dimensions;
    pooling_method = new_pooling_method;
    set_name(new_name);
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


void Pooling3d::forward_propagate(const vector<pair<type*, dimensions>>& input_pairs,
                                  unique_ptr<LayerForwardPropagation>& layer_forward_propagation,
                                  const bool& is_training)
{
    Pooling3dForwardPropagation* pooling_layer_forward_propagation =
        static_cast<Pooling3dForwardPropagation*>(layer_forward_propagation.get());

    const TensorMap<Tensor<type, 3>> inputs = tensor_map<3>(input_pairs[0]);
    Tensor<type, 2>& outputs = pooling_layer_forward_propagation->outputs;

    const Index batch_size = inputs.dimension(0);
    const Index sequence_length = inputs.dimension(1);
    const Index features = inputs.dimension(2);

    if (pooling_method == PoolingMethod::MaxPooling)
    {
        Tensor<Index, 2>& maximal_indices = pooling_layer_forward_propagation->maximal_indices;

        #pragma omp parallel for
        for (Index batch_index = 0; batch_index < batch_size; ++batch_index)
        {
            for (Index feature_index = 0; feature_index < features; ++feature_index)
            {
                type   max_val = -std::numeric_limits<type>::infinity();
                Index  max_idx = 0;

                for (Index seq_index = 0; seq_index < sequence_length; ++seq_index)
                {
                    const type value = inputs(batch_index, seq_index, feature_index);
                    if (value > max_val)
                    {
                        max_val = value;
                        max_idx = seq_index;
                    }
                }

                outputs(batch_index, feature_index) = max_val;
                if (is_training) maximal_indices(batch_index, feature_index) = max_idx;
            }
        }
    }
    else // AveragePooling
    {
        outputs.device(*thread_pool_device) =
            inputs.mean(array<Index, 1>({1}))
                  .reshape(array<Index, 2>({batch_size, features}));
    }
}


void Pooling3d::back_propagate(const vector<pair<type*, dimensions>>& input_pairs,
                               const vector<pair<type*, dimensions>>& delta_pairs,
                               unique_ptr<LayerForwardPropagation>& forward_propagation,
                               unique_ptr<LayerBackPropagation>& back_propagation) const
{
    const TensorMap<Tensor<type, 3>> input_tensor_map  = tensor_map<3>(input_pairs[0]);
    const TensorMap<Tensor<type, 2>> delta_tensor_map  = tensor_map<2>(delta_pairs[0]);

    Pooling3dForwardPropagation* forward_layer  =
        static_cast<Pooling3dForwardPropagation*>(forward_propagation.get());
    Pooling3dBackPropagation* backward_layer =
        static_cast<Pooling3dBackPropagation*>(back_propagation.get());

    backward_layer->input_derivatives.setZero();

    const Index batch_size = input_tensor_map.dimension(0);
    const Index sequence_length = input_tensor_map.dimension(1);
    const Index number_of_features = input_tensor_map.dimension(2);

    if (pooling_method == PoolingMethod::MaxPooling)
    {
        for (Index batch_index = 0; batch_index < batch_size; ++batch_index)
        {
            for (Index feature_index = 0; feature_index < number_of_features; ++feature_index)
            {
                const Index maximum_index =
                    forward_layer->maximal_indices(batch_index, feature_index);

                backward_layer->input_derivatives(batch_index,
                                                  maximum_index,
                                                  feature_index) +=
                    delta_tensor_map(batch_index, feature_index);
            }
        }
    }
    else // AveragePooling
    {
        const array<Index, 3> broadcast_shape = {1, sequence_length, 1};
        const array<Index, 3> reshape_dimensions = {batch_size, 1, number_of_features};

        backward_layer->input_derivatives.device(*thread_pool_device) +=
            delta_tensor_map.reshape(reshape_dimensions)
                            .broadcast(broadcast_shape)
                            / static_cast<type>(sequence_length);
    }
}


void Pooling3dForwardPropagation::set(const Index& new_batch_size, Layer* new_layer)
{
    batch_size = new_batch_size;
    layer = new_layer;

    Pooling3d* pooling_layer = static_cast<Pooling3d*>(new_layer);

    const Index features = pooling_layer->get_output_dimensions()[0];
    outputs.resize(batch_size, features);

    if (pooling_layer->get_pooling_method() == Pooling3d::PoolingMethod::MaxPooling)
    {
        maximal_indices.resize(batch_size, features);
    }
}


void Pooling3dBackPropagation::set(const Index& new_batch_size, Layer* new_layer)
{
    batch_size = new_batch_size;
    layer = static_cast<Pooling3d*>(new_layer);

    const dimensions layer_input_dimensions = layer->get_input_dimensions();

    input_derivatives.resize(batch_size,
                             layer_input_dimensions[0],
                             layer_input_dimensions[1]);
    input_derivatives.setZero();
}


Pooling3dForwardPropagation::Pooling3dForwardPropagation(const Index& new_batch_size, Layer* new_layer)
    : LayerForwardPropagation()
{
    set(new_batch_size, new_layer);
}


pair<type*, dimensions> Pooling3dForwardPropagation::get_outputs_pair() const
{
    return {(type*)outputs.data(), {batch_size, layer->get_output_dimensions()[0]}};
}


Pooling3dBackPropagation::Pooling3dBackPropagation(const Index& new_batch_size, Layer* new_layer)
    : LayerBackPropagation()
{
    set(new_batch_size, new_layer);
}


vector<pair<type*, dimensions>> Pooling3dBackPropagation::get_input_derivative_pairs() const
{
    const auto input_dims = layer->get_input_dimensions();
    return {{(type*)input_derivatives.data(), {batch_size, input_dims[0], input_dims[1]}}};
}


void Pooling3d::to_XML(XMLPrinter& printer) const
{
    printer.OpenElement("Pooling3d");
    add_xml_element(printer, "InputDimensions", dimensions_to_string(get_input_dimensions()));
    add_xml_element(printer, "PoolingMethod", write_pooling_method());
    printer.CloseElement();
}

void Pooling3d::from_XML(const XMLDocument& document)
{
    const XMLElement* element = document.FirstChildElement("Pooling3d");
    if (!element) throw runtime_error("Pooling3d element is nullptr.");

    set_input_dimensions(string_to_dimensions(read_xml_string(element, "InputDimensions")));
    set_pooling_method(read_xml_string(element, "PoolingMethod"));
}


void Pooling3d::print() const
{
    cout << "Pooling3d layer" << endl;
    cout << "Input dimensions: " << dimensions_to_string(input_dimensions) << endl;
    cout << "Output dimensions: " << dimensions_to_string(get_output_dimensions()) << endl;
    cout << "Pooling Method: " << write_pooling_method() << endl;
}

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
