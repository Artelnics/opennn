//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   F O R W A R D   P R O P A G A T I O N   S O U R C E
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "forward_propagation.h"
#include "neural_network.h"
#include "convolutional_layer.h"
#include "dense_layer.h"
#include "multihead_attention_layer.h"

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
    const vector<vector<Shape>> forward_shapes = neural_network->get_forward_shapes(batch_size);

    views.resize(layers_number);

#ifdef OPENNN_WITH_CUDA
    const bool is_gpu = Configuration::instance().is_gpu();

    if (is_gpu)
    {
        vector<vector<Type>> forward_dtypes(layers_number);
        for (Index i = 0; i < layers_number; ++i)
            forward_dtypes[i] = layers[i]->get_forward_dtypes(batch_size);

        Index total_bytes = 0;
        for (Index i = 0; i < layers_number; ++i)
        {
            const vector<Shape>& shapes = forward_shapes[i];
            for (size_t j = 0; j < shapes.size(); ++j)
                if (shapes[j].size() > 0)
                    total_bytes += get_aligned_bytes(shapes[j].size() * type_bytes(forward_dtypes[i][j]));
        }

        if (total_bytes > 0)
        {
            data.resize_bytes(total_bytes, Device::CUDA);
            data.setZero();
        }

        uint8_t* cursor = (total_bytes > 0) ? data.as<uint8_t>() : nullptr;
        for (Index i = 0; i < layers_number; ++i)
        {
            const vector<Shape>& shapes = forward_shapes[i];
            const size_t slots = shapes.size();
            views[i].resize(slots + 1);

            for (size_t j = 0; j < slots; ++j)
            {
                const Shape& slot_shape = shapes[j];
                views[i][j + 1].resize(1);

                if (slot_shape.size() > 0)
                {
                    views[i][j + 1][0] = TensorView(cursor, slot_shape, forward_dtypes[i][j]);
                    if (cursor) cursor += get_aligned_bytes(slot_shape.size() * type_bytes(forward_dtypes[i][j]));
                }
            }
        }
    }
    else
#endif
    {
        const Index total_size = aligned_total_elements(forward_shapes);

        if (total_size > 0)
        {
            data.resize_bytes(total_size * Index(sizeof(float)), Device::CPU);
            data.setZero();
        }

        float* pointer = (total_size > 0) ? data.as<float>() : nullptr;
        for (Index i = 0; i < layers_number; ++i)
        {
            const vector<Shape>& shapes = forward_shapes[i];
            const size_t slots = shapes.size();
            views[i].resize(slots + 1);

            for (size_t j = 0; j < slots; ++j)
            {
                const Shape& slot_shape = shapes[j];
                views[i][j + 1].resize(1);

                if (slot_shape.size() > 0)
                {
                    views[i][j + 1][0] = TensorView(pointer, slot_shape);
                    if (pointer) pointer += get_aligned_size(slot_shape.size());
                }
            }
        }
    }

    const auto& layer_input_indices = neural_network->get_layer_input_indices();
    for (Index i = 0; i < layers_number; ++i)
    {
        const vector<Index>& input_indices = layer_input_indices[i];
        const size_t input_indices_size = input_indices.size();
        views[i][0].resize(input_indices_size);

        for (size_t k = 0; k < input_indices_size; ++k)
        {
            const Index producer_index = input_indices[k];

            if (producer_index >= 0)
            {
                const size_t output_slot = forward_shapes[producer_index].size();

                if (output_slot > 0 && producer_index < ssize(views)
                    && !views[producer_index][output_slot].empty())
                {
                    views[i][0][k] = views[producer_index][output_slot][0];
                }
            }
        }
    }

#ifdef OPENNN_WITH_CUDA
    if (is_gpu)
    {
        for (auto& layer : layers)
        {
            if (layer->get_type() == LayerType::Convolutional)
            {
                if (auto* conv = dynamic_cast<Convolutional*>(layer.get()))
                    conv->init_cuda(batch_size);
            }
            else if (layer->get_type() == LayerType::Dense)
            {
                if (auto* dense = dynamic_cast<Dense*>(layer.get()))
                    dense->init_cuda(batch_size);
            }
        }
    }
#endif
}

TensorView ForwardPropagation::get_last_trainable_layer_outputs() const
{
    const Index last_trainable_layer_index = neural_network->get_last_trainable_layer_index();

    if (last_trainable_layer_index < 0
       || static_cast<size_t>(last_trainable_layer_index) >= views.size()
       || views[last_trainable_layer_index].size() <= 1
       || views[last_trainable_layer_index].back().empty())
        return {};

    return views[last_trainable_layer_index].back()[0];
}

TensorView ForwardPropagation::get_outputs() const
{
    if (!neural_network || views.empty()) return {};

    const size_t layers_number = neural_network->get_layers_number();

    if (layers_number == 0
       || layers_number - 1 >= views.size()
       || views[layers_number - 1].size() < 2
       || views[layers_number - 1].back().empty())
    {
        return get_last_trainable_layer_outputs();
    }

    return views[layers_number - 1].back()[0];
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
