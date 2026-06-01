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

ForwardPropagation::ForwardPropagation(const Index new_batch_size, NeuralNetwork* new_neural_network)
{
    set(new_batch_size, new_neural_network);
}

void ForwardPropagation::set(const Index new_batch_size, NeuralNetwork* new_neural_network)
{
    batch_size = new_batch_size;
    neural_network = new_neural_network;

    throw_if(!neural_network, "neural network is not set.");

    const auto& layers = neural_network->get_layers();
    const size_t layers_number = layers.size();
    views.resize(layers_number);

    const auto forward_specs = neural_network->get_forward_specs(batch_size);

    if (const Index total_bytes = get_aligned_bytes(forward_specs); total_bytes > 0)
    {
        data.resize_bytes(total_bytes, neural_network->get_device());
        data.setZero();
    }

    uint8_t* cursor = data.as<uint8_t>();
    for (size_t i = 0; i < layers_number; ++i)
    {
        const auto& specs = forward_specs[i];
        views[i].assign(specs.size() + 1, vector<TensorView>(1));

        for (size_t j = 0; j < specs.size(); ++j)
        {
            const auto& [shape, dtype] = specs[j];
            if (shape.size() == 0) continue;
            views[i][j + 1][0] = TensorView(cursor, shape, dtype, data.device_type);
            cursor += get_aligned_bytes(shape.size(), dtype);
        }
    }

    const auto& source_layers = neural_network->get_source_layers();
    for (size_t i = 0; i < layers_number; ++i)
    {
        const vector<Index>& sources = source_layers[i];
        views[i][0].resize(sources.size());

        for (size_t j = 0; j < sources.size(); ++j)
        {
            const Index source_layer = sources[j];
            if (source_layer < 0) continue;

            const size_t output_slot = forward_specs[source_layer].size();
            if (output_slot == 0) continue;

            if (const TensorView& source = views[source_layer][output_slot][0]; !source.empty())
                views[i][0][j] = source;
        }
    }
}

TensorView ForwardPropagation::get_last_trainable_layer_outputs() const
{
    if (!neural_network) return {};

    const Index layer_index = neural_network->get_last_trainable_layer_index();
    
    if (layer_index < 0
        || size_t(layer_index) >= views.size()
        || views[layer_index].size() <= 1)
        return {};

    const TensorView& v = views[layer_index].back()[0];
    return v.empty() ? TensorView{} : v;
}

TensorView ForwardPropagation::get_outputs() const
{
    if (!neural_network) return {};

    const Index last = Index(neural_network->get_layers_number()) - 1;
    
    if (last >= 0
        && size_t(last) < views.size()
        && views[last].size() > 1)
    {
        const TensorView& v = views[last].back()[0];
        if (!v.empty()) return v;
    }

    return get_last_trainable_layer_outputs();
}

void ForwardPropagation::print() const
{
    cout << "Neural network forward propagation" << "\n";

    const size_t layers_number = neural_network->get_layers_number();

    cout << "Layers number: " << layers_number << "\n";

    for (size_t i = 0; i < layers_number; ++i)
        cout << "Layer " << i + 1 << ": " << neural_network->get_layer(static_cast<Index>(i))->get_label() << "\n";
}

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
