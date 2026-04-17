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
        for(const Shape& s : layer_shapes)
            total_size += get_aligned_size(s.size());

    if(total_size > 0)
    {
        data.resize(total_size);
        data.setZero();
    }

    views.resize(layers_number);
    type* pointer = (total_size > 0) ? data.data() : nullptr;

    for(Index i = 0; i < layers_number; ++i)
    {
        const vector<Shape>& shapes = forward_shapes[i];
        const size_t slots = shapes.size();

        views[i].resize(slots + 1);

        for(size_t j = 0; j < slots; ++j)
        {
            const Shape& s = shapes[j];
            views[i][j + 1].resize(1);

            if(s.size() > 0 && pointer)
            {
                views[i][j + 1][0] = TensorView(pointer, s);

                pointer += get_aligned_size(s.size());
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
    if(!neural_network || batch_size <= 0 || data.size() == 0) return;

    data.resize_device(data.size());
    data.setZero_device();

    const vector<vector<Shape>> forward_shapes = neural_network->get_forward_shapes(batch_size);
    const auto& layer_input_indices = neural_network->get_layer_input_indices();
    const size_t layers_number = neural_network->get_layers().size();

    type* dev_pointer = data.device();

    for(Index i = 0; i < layers_number; ++i)
    {
        const vector<Shape>& shapes = forward_shapes[i];

        for(size_t j = 0; j < shapes.size(); ++j)
        {
            const Shape& s = shapes[j];

            if(s.size() > 0)
            {
                views[i][j + 1][0].data = dev_pointer;
                views[i][j + 1][0].set_descriptor(s);
                dev_pointer += get_aligned_size(s.size());
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
            Convolutional* conv = static_cast<Convolutional*>(layer.get());
            conv->init_cuda(batch_size);
        }
        else if(layer->get_type() == LayerType::Dense2d)
        {
            Dense<2>* dense = static_cast<Dense<2>*>(layer.get());
            dense->init_cuda(batch_size);
        }
        else if(layer->get_type() == LayerType::Dense3d)
        {
            Dense<3>* dense = static_cast<Dense<3>*>(layer.get());
            dense->init_cuda(batch_size);
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

    for(Index i = 0; i < layers_number; i++)
        cout << "Layer " << i + 1 << ": " << neural_network->get_layer(i)->get_label() << "\n";
}

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
