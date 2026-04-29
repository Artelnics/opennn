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

    if(!neural_network) throw runtime_error("ForwardPropagation error: neural network is not set.");

    const vector<unique_ptr<Layer>>& nn_layers = neural_network->get_layers();
    const size_t layers_number = nn_layers.size();

    const vector<vector<Shape>> forward_shapes = neural_network->get_forward_shapes(batch_size);

    Index total_size = 0;

    for(const auto& layer_shapes : forward_shapes)
        for(const Shape& shape : layer_shapes)
            total_size += get_aligned_size(shape.size());

    // Forward arena is pure scratch — no Eigen-based init reads from it before
    // the first forward pass. When the resolved device is CUDA we skip the CPU
    // allocation entirely and let allocate_device() build the GPU buffer; the
    // view structure is still populated below so input wiring works.
    const bool gpu_mode = Configuration::instance().is_gpu();

    if(total_size > 0 && !gpu_mode)
    {
        data.resize_bytes(total_size * Index(sizeof(type)), DeviceType::CPU);
        data.setZero();
    }

    views.resize(layers_number);
    type* pointer = (total_size > 0 && !gpu_mode) ? data.as<type>() : nullptr;

    for(Index i = 0; i < layers_number; ++i)
    {
        const vector<Shape>& shapes = forward_shapes[i];
        const size_t slots = shapes.size();

        views[i].resize(slots + 1);

        for(size_t j = 0; j < slots; ++j)
        {
            const Shape& s = shapes[j];
            views[i][j + 1].resize(1);

            // Always set the shape so downstream wiring sees a non-empty view.
            // `data` may be null in GPU mode (allocate_device fills it later).
            if(s.size() > 0)
            {
                views[i][j + 1][0] = TensorView(pointer, s);

                if(pointer) pointer += get_aligned_size(s.size());
            }
        }
    }

    const auto& layer_input_indices = neural_network->get_layer_input_indices();

    for(Index i = 0; i < layers_number; ++i)
    {
        const vector<Index>& input_indices = layer_input_indices[i];
        const size_t input_indices_size = input_indices.size();
        views[i][0].resize(input_indices_size);

        for(size_t k = 0; k < input_indices_size; ++k)
        {
            const Index j = input_indices[k];

            if(j >= 0)
            {
                const size_t output_slot = forward_shapes[j].size();

                if(output_slot > 0 && j < ssize(views)
                    && !views[j][output_slot].empty())
                {
                    views[i][0][k] = views[j][output_slot][0];
                }
            }
        }
    }
}

void ForwardPropagation::allocate_device()
{
#ifdef OPENNN_WITH_CUDA
    if(!neural_network || batch_size <= 0) return;

    const vector<vector<Shape>> forward_shapes = neural_network->get_forward_shapes(batch_size);
    const auto& nn_layers = neural_network->get_layers();
    const auto& layer_input_indices = neural_network->get_layer_input_indices();
    const size_t layers_number = nn_layers.size();

    vector<vector<cudnnDataType_t>> forward_dtypes(layers_number);
    for(Index i = 0; i < layers_number; ++i)
        forward_dtypes[i] = nn_layers[i]->get_forward_dtypes(batch_size);

    Index total_bytes = 0;
    for(Index i = 0; i < layers_number; ++i)
    {
        const vector<Shape>& shapes = forward_shapes[i];
        for(size_t j = 0; j < shapes.size(); ++j)
            if(shapes[j].size() > 0)
                total_bytes += get_aligned_bytes(shapes[j].size() * dtype_bytes(forward_dtypes[i][j]));
    }

    if(total_bytes == 0) return;

    data.resize_bytes(total_bytes, DeviceType::CUDA);
    data.setZero();

    uint8_t* cursor = data.as<uint8_t>();

    for(Index i = 0; i < layers_number; ++i)
    {
        const vector<Shape>& shapes = forward_shapes[i];

        for(size_t j = 0; j < shapes.size(); ++j)
        {
            const Shape& s = shapes[j];

            if(s.size() > 0)
            {
                views[i][j + 1][0].data  = cursor;
                views[i][j + 1][0].dtype = forward_dtypes[i][j];
                cursor += get_aligned_bytes(s.size() * dtype_bytes(forward_dtypes[i][j]));
            }
        }
    }

    for(Index i = 0; i < layers_number; ++i)
    {
        const vector<Index>& input_idx = layer_input_indices[i];

        for(size_t k = 0; k < input_idx.size(); ++k)
        {
            const Index j = input_idx[k];

            if(j >= 0)
            {
                const size_t output_slot = forward_shapes[j].size();

                if(output_slot > 0 && j < ssize(views)
                    && !views[j][output_slot].empty())
                {
                    views[i][0][k] = views[j][output_slot][0];
                }
            }
        }
    }

    for(auto& layer : neural_network->get_layers())
    {
        if(layer->get_type() == LayerType::Convolutional)
        {
            if(auto* conv = dynamic_cast<Convolutional*>(layer.get()))
                conv->init_cuda(batch_size);
        }
        else if(layer->get_type() == LayerType::Dense2d)
        {
            if(auto* dense = dynamic_cast<Dense<2>*>(layer.get()))
                dense->init_cuda(batch_size);
        }
        else if(layer->get_type() == LayerType::Dense3d)
        {
            if(auto* dense = dynamic_cast<Dense<3>*>(layer.get()))
                dense->init_cuda(batch_size);
        }
        else if(layer->get_type() == LayerType::MultiHeadAttention)
        {
            if(auto* mha = dynamic_cast<MultiHeadAttention*>(layer.get()))
                mha->init_cuda(batch_size);
        }
    }
#endif
}

TensorView ForwardPropagation::get_last_trainable_layer_outputs() const
{
    const Index last_trainable_layer_index = neural_network->get_last_trainable_layer_index();

    if(last_trainable_layer_index < 0
       || static_cast<size_t>(last_trainable_layer_index) >= views.size()
       || views[last_trainable_layer_index].size() <= 1
       || views[last_trainable_layer_index].back().empty())
        return {};

    return views[last_trainable_layer_index].back()[0];
}

vector<vector<TensorView>> ForwardPropagation::get_layer_input_views(const vector<TensorView>&,
                                                                     bool) const
{
    const size_t layers_number = neural_network->get_layers_number();

    if (layers_number == 0) return {};

    vector<vector<TensorView>> layer_input_views(layers_number);

    for (Index i = 0; i < layers_number; ++i)
        if(i < views.size() && !views[i].empty())
            layer_input_views[i] = views[i][0];

    return layer_input_views;
}

TensorView ForwardPropagation::get_outputs() const
{
    if(!neural_network || views.empty()) return {};

    const size_t layers_number = neural_network->get_layers_number();

    if(layers_number == 0
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

    for(Index i = 0; i < layers_number; ++i)
        cout << "Layer " << i + 1 << ": " << neural_network->get_layer(i)->get_label() << "\n";
}

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
