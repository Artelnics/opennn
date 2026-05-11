//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   F O R W A R D   P R O P A G A T I O N   S O U R C E
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "forward_propagation.h"
#include "neural_network.h"

namespace opennn
{

namespace {

TensorView layer_output_view(const vector<vector<vector<TensorView>>>& views, Index layer_index)
{
    if (layer_index < 0
        || size_t(layer_index) >= views.size()
        || views[layer_index].size() <= 1
        || views[layer_index].back().empty())
        return {};
    return views[layer_index].back()[0];
}

}

ForwardPropagation::ForwardPropagation(const Index new_batch_size, NeuralNetwork* new_neural_network)
{
    set(new_batch_size, new_neural_network);
}

void ForwardPropagation::set(const Index new_batch_size, NeuralNetwork* new_neural_network)
{
    batch_size = new_batch_size;
    neural_network = new_neural_network;

    if (!neural_network) throw runtime_error("neural network is not set.");

    const auto& layers = neural_network->get_layers();
    const size_t layers_number = layers.size();
    const vector<vector<Shape>> forward_shapes = neural_network->get_forward_shapes(batch_size);

    views.resize(layers_number);

    const bool on_gpu = is_gpu();
    const Device device = on_gpu ? Device::CUDA : Device::CPU;

    const auto forward_dtypes = collect_layer_dtypes(layers, batch_size, on_gpu,
                                                     &Layer::get_forward_dtypes);

    const Index total_bytes = aligned_total_bytes(forward_shapes, forward_dtypes);

    if (total_bytes > 0)
    {
        data.resize_bytes(total_bytes, device);
        data.setZero();
    }

    uint8_t* cursor = data.as<uint8_t>();
    for (Index i = 0; i < layers_number; ++i)
    {
        const vector<Shape>& shapes = forward_shapes[i];
        const size_t slots = shapes.size();
        views[i].assign(slots + 1, vector<TensorView>(1));

        for (size_t j = 0; j < slots; ++j)
        {
            const Shape& slot_shape = shapes[j];

            if (slot_shape.size() > 0)
            {
                views[i][j + 1][0] = TensorView(cursor, slot_shape, forward_dtypes[i][j]);
                cursor += get_aligned_bytes(slot_shape.size() * type_bytes(forward_dtypes[i][j]));
            }
        }
    }

    const auto& layer_input_indices = neural_network->get_layer_input_indices();
    for (Index i = 0; i < layers_number; ++i)
    {
        const vector<Index>& input_indices = layer_input_indices[i];
        views[i][0].resize(input_indices.size());

        for (size_t k = 0; k < input_indices.size(); ++k)
        {
            const Index producer = input_indices[k];
            if (producer < 0) continue;

            const size_t output_slot = forward_shapes[producer].size();
            if (output_slot == 0) continue;

            const TensorView& source = views[producer][output_slot][0];
            if (source.empty()) continue;

            views[i][0][k] = source;
        }
    }
}

TensorView ForwardPropagation::get_last_trainable_layer_outputs() const
{
    if (!neural_network) return {};
    return layer_output_view(views, neural_network->get_last_trainable_layer_index());
}

TensorView ForwardPropagation::get_outputs() const
{
    if (!neural_network) return {};
    const Index last = Index(neural_network->get_layers_number()) - 1;
    const TensorView v = layer_output_view(views, last);
    return v.empty() ? get_last_trainable_layer_outputs() : v;
}

void ForwardPropagation::print() const
{
    cout << "Neural network forward propagation" << "\n";

    const size_t layers_number = neural_network->get_layers_number();

    cout << "Layers number: " << layers_number << "\n";

    for (Index i = 0; i < layers_number; ++i)
        cout << "Layer " << i + 1 << ": " << neural_network->get_layer(i)->get_label() << "\n";
}

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
