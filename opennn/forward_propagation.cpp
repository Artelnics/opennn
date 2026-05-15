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

    if (!neural_network) throw runtime_error("neural network is not set.");

    const auto& layers = neural_network->get_layers();
    const size_t layers_number = layers.size();
    views.resize(layers_number);

    const vector<vector<Shape>> forward_shapes = neural_network->get_forward_shapes(batch_size);

    const vector<vector<Type>> forward_dtypes = neural_network->get_forward_dtypes(batch_size);

    if (const Index total_bytes = get_aligned_bytes(forward_shapes, forward_dtypes); total_bytes > 0)
    {
        const Device device = current_device();
        data.resize_bytes(total_bytes, device);
        data.setZero();
    }

    uint8_t* cursor = data.as<uint8_t>();
    for (size_t i = 0; i < layers_number; ++i)
    {
        const vector<Shape>& shapes = forward_shapes[i];
        const vector<Type>& dtypes = forward_dtypes[i];
        views[i].assign(shapes.size() + 1, vector<TensorView>(1));

        for (size_t j = 0; j < shapes.size(); ++j)
        {
            if (shapes[j].size() == 0) continue;
            views[i][j + 1][0] = TensorView(cursor, shapes[j], dtypes[j]);
            cursor += get_aligned_bytes(shapes[j].size(), dtypes[j]);
        }
    }

    const auto& layer_input_indices = neural_network->get_layer_input_indices();
    for (size_t i = 0; i < layers_number; ++i)
    {
        const vector<Index>& input_indices = layer_input_indices[i];
        const size_t inputs_number = input_indices.size();
        views[i][0].resize(inputs_number);

        for (size_t k = 0; k < inputs_number; ++k)
        {
            const Index producer = input_indices[k];
            if (producer < 0) continue;

            const size_t output_slot = forward_shapes[producer].size();
            if (output_slot == 0) continue;

            const TensorView& source = views[producer][output_slot][0];
            if (!source.empty()) views[i][0][k] = source;
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
