//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   B A C K   P R O P A G A T I O N   S O U R C E
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "back_propagation.h"
#include "loss.h"
#include "neural_network.h"
#include "forward_propagation.h"
#include "math_utilities.h"

namespace opennn
{

BackPropagation::BackPropagation(const Index new_batch_size, Loss* new_loss)
{
    set(new_batch_size, new_loss);
}

void BackPropagation::set(const Index new_batch_size, Loss* new_loss)
{
    batch_size = new_batch_size;
    loss = new_loss;

    if(!loss)
        throw runtime_error("BackPropagation error: loss is not set.");

    const NeuralNetwork* neural_network = loss->get_neural_network();

    if(!neural_network)
        throw runtime_error("BackPropagation error: neural network is not set.");

    const auto& layers = neural_network->get_layers();
    const size_t layers_number = layers.size();
    const vector<vector<Shape>> parameter_shapes = neural_network->get_parameter_shapes();
    const vector<vector<Shape>> backward_shapes  = neural_network->get_backward_shapes(batch_size);
    const Shape output_shape = neural_network->get_output_shape();
    const Index last_trainable_layer_index = neural_network->get_last_trainable_layer_index();
    const auto& layer_input_indices = neural_network->get_layer_input_indices();

    output_delta_dimensions = Shape({batch_size}).append(output_shape);

    loss_value = float(0);
    error = float(0);
    accuracy.setZero();

    backward_edges.assign(layers_number, {});
    for(size_t consumer_index = 0; consumer_index < layers_number; ++consumer_index)
    {
        const vector<Index>& inputs = layer_input_indices[consumer_index];
        for(size_t port_index = 0; port_index < inputs.size(); ++port_index)
        {
            const Index producer = inputs[port_index];
            if(producer >= 0 && static_cast<size_t>(producer) < layers_number)
                backward_edges[producer].push_back({consumer_index, port_index});
        }
    }

    per_layer_output_delta_shapes.assign(layers_number, Shape());
    for(Index i = 0; i < layers_number; ++i)
    {
        if(i == last_trainable_layer_index)        continue;
        if(backward_edges[i].size() <= 1)          continue;
        const Shape output_shape_i = layers[i]->get_output_shape();
        if(output_shape_i.empty())                 continue;
        per_layer_output_delta_shapes[i] = Shape({batch_size}).append(output_shape_i);
    }

    gradient_views.resize(layers_number);
    delta_views.resize(layers_number);

#ifdef OPENNN_WITH_CUDA
    const bool is_gpu = Configuration::instance().is_gpu();
    if(is_gpu)
    {
        const Type activation_dtype = neural_network->get_training_type();
        const Index activation_bytes = type_bytes(activation_dtype);

        vector<vector<Type>> backward_dtypes(layers_number);
        for(Index i = 0; i < layers_number; ++i)
            backward_dtypes[i] = layers[i]->get_backward_dtypes(batch_size);

        const Index total_gradient_floats = aligned_total_elements(parameter_shapes);
        gradient.resize_bytes(total_gradient_floats * Index(sizeof(float)), Device::CUDA);
        gradient.setZero();

        float* g_ptr = gradient.as<float>();
        for(Index i = 0; i < layers_number; ++i)
        {
            const vector<Shape>& layer_param_shapes = parameter_shapes[i];
            gradient_views[i].resize(layer_param_shapes.size());

            for(size_t j = 0; j < layer_param_shapes.size(); ++j)
            {
                const Shape& slot_shape = layer_param_shapes[j];
                if(slot_shape.size() > 0)
                {
                    gradient_views[i][j] = TensorView(g_ptr, slot_shape, Type::FP32);
                    g_ptr += get_aligned_size(slot_shape.size());
                }
            }
        }

        Index total_backward_bytes = 0;
        for(Index i = 0; i < layers_number; ++i)
        {
            const vector<Shape>& shapes = backward_shapes[i];
            for(size_t j = 0; j < shapes.size(); ++j)
                if(shapes[j].size() > 0)
                    total_backward_bytes += get_aligned_bytes(shapes[j].size() * type_bytes(backward_dtypes[i][j]));
        }

        if(total_backward_bytes > 0)
        {
            backward.resize_bytes(total_backward_bytes, Device::CUDA);
            backward.setZero();
        }

        uint8_t* b_cursor = (total_backward_bytes > 0) ? backward.as<uint8_t>() : nullptr;
        for(Index i = 0; i < layers_number; ++i)
        {
            const vector<Shape>& shapes = backward_shapes[i];
            const size_t slots = shapes.size();

            delta_views[i].resize(slots + 1);
            delta_views[i][0].resize(1);

            for(size_t j = 0; j < slots; ++j)
            {
                const Shape& slot_shape = shapes[j];
                delta_views[i][j + 1].resize(1);

                if(slot_shape.size() > 0)
                {
                    delta_views[i][j + 1][0] = TensorView(b_cursor, slot_shape, backward_dtypes[i][j]);
                    if(b_cursor) b_cursor += get_aligned_bytes(slot_shape.size() * type_bytes(backward_dtypes[i][j]));
                }
            }
        }

        const Index total_output_delta_elems = batch_size * output_shape.size();
        output_deltas.resize_bytes(total_output_delta_elems * activation_bytes, Device::CUDA);
        output_deltas.setZero();

        Index total_og_bytes = 0;
        for(Index i = 0; i < layers_number; ++i)
            if(!per_layer_output_delta_shapes[i].empty())
                total_og_bytes += get_aligned_bytes(per_layer_output_delta_shapes[i].size() * activation_bytes);

        if(total_og_bytes > 0)
        {
            per_layer_output_deltas.resize_bytes(total_og_bytes, Device::CUDA);
            per_layer_output_deltas.setZero();
        }

        for(Index i = 0; i < layers_number; ++i)
        {
            if(delta_views[i].empty()) continue;

            if(i == last_trainable_layer_index)
            {
                delta_views[i][0][0] = TensorView(output_deltas.as<float>(),
                                                  output_delta_dimensions,
                                                  activation_dtype);
            }
            else if(!backward_edges[i].empty())
            {
                if(backward_edges[i].size() > 1
                   && !per_layer_output_delta_shapes[i].empty()
                   && per_layer_output_deltas.as<uint8_t>())
                {
                    uint8_t* og_cursor = per_layer_output_deltas.as<uint8_t>();
                    for(size_t k = 0; k < i; ++k)
                        if(!per_layer_output_delta_shapes[k].empty())
                            og_cursor += get_aligned_bytes(per_layer_output_delta_shapes[k].size()
                                                           * activation_bytes);

                    delta_views[i][0][0].data  = og_cursor;
                    delta_views[i][0][0].shape = per_layer_output_delta_shapes[i];
                    delta_views[i][0][0].type = activation_dtype;
                }
                else
                {
                    const BackwardEdge& edge = backward_edges[i].front();
                    const size_t slot = 1 + edge.port;
                    if(edge.consumer_idx < delta_views.size()
                       && slot < delta_views[edge.consumer_idx].size()
                       && !delta_views[edge.consumer_idx][slot].empty())
                    {
                        delta_views[i][0][0] = delta_views[edge.consumer_idx][slot][0];
                    }
                }
            }
        }

        if(errors_device) { cudaFree(errors_device); errors_device = nullptr; }
        const Index outputs_number = output_shape[0];
        CHECK_CUDA(cudaMalloc(&errors_device, batch_size * outputs_number * sizeof(float)));

        output_deltas_view_device = TensorView(output_deltas.as<float>(),
                                               output_delta_dimensions,
                                               activation_dtype);
        return;
    }
#endif

    const Index total_parameters_size = aligned_total_elements(parameter_shapes);
    if(total_parameters_size > 0)
    {
        gradient.resize_bytes(total_parameters_size * Index(sizeof(float)), Device::CPU);
        gradient.setZero();
    }

    float* g_ptr = (total_parameters_size > 0) ? gradient.as<float>() : nullptr;
    for(Index i = 0; i < layers_number; ++i)
    {
        const vector<Shape>& layer_param_shapes = parameter_shapes[i];
        gradient_views[i].resize(layer_param_shapes.size());

        for(size_t j = 0; j < layer_param_shapes.size(); ++j)
        {
            const Shape& slot_shape = layer_param_shapes[j];
            if(slot_shape.size() > 0)
            {
                gradient_views[i][j] = TensorView(g_ptr, slot_shape, Type::FP32);
                if(g_ptr) g_ptr += get_aligned_size(slot_shape.size());
            }
        }
    }

    const Index total_backward_size = aligned_total_elements(backward_shapes);
    if(total_backward_size > 0)
    {
        backward.resize_bytes(total_backward_size * Index(sizeof(float)), Device::CPU);
        backward.setZero();
    }

    float* b_ptr = (total_backward_size > 0) ? backward.as<float>() : nullptr;
    for(Index i = 0; i < layers_number; ++i)
    {
        const vector<Shape>& shapes = backward_shapes[i];
        const size_t slots = shapes.size();

        delta_views[i].resize(slots + 1);
        delta_views[i][0].resize(1);

        for(size_t j = 0; j < slots; ++j)
        {
            const Shape& slot_shape = shapes[j];
            delta_views[i][j + 1].resize(1);

            if(slot_shape.size() > 0)
            {
                delta_views[i][j + 1][0] = TensorView(b_ptr, slot_shape);
                if(b_ptr) b_ptr += get_aligned_size(slot_shape.size());
            }
        }
    }

    const Index total_output_elements = output_shape.size() * batch_size;
    if(total_output_elements > 0)
    {
        output_deltas.resize_bytes(total_output_elements * Index(sizeof(float)), Device::CPU);
        output_deltas.setZero();
    }

    Index total_output_delta_size = 0;
    for(Index i = 0; i < layers_number; ++i)
        if(!per_layer_output_delta_shapes[i].empty())
            total_output_delta_size += get_aligned_size(per_layer_output_delta_shapes[i].size());

    if(total_output_delta_size > 0)
    {
        per_layer_output_deltas.resize_bytes(total_output_delta_size * Index(sizeof(float)), Device::CPU);
        per_layer_output_deltas.setZero();
    }

    float* og_ptr = (total_output_delta_size > 0) ? per_layer_output_deltas.as<float>() : nullptr;
    for(Index i = 0; i < layers_number; ++i)
    {
        if(delta_views[i].empty()) continue;

        if(i == last_trainable_layer_index)
        {
            delta_views[i][0][0] = TensorView(output_deltas.as<float>(), output_delta_dimensions);
        }
        else if(!backward_edges[i].empty())
        {
            if(backward_edges[i].size() > 1 && og_ptr && !per_layer_output_delta_shapes[i].empty())
            {
                delta_views[i][0][0] = TensorView(og_ptr, per_layer_output_delta_shapes[i]);
                og_ptr += get_aligned_size(per_layer_output_delta_shapes[i].size());
            }
            else
            {
                const BackwardEdge& edge = backward_edges[i].front();
                const size_t slot = 1 + edge.port;
                if(edge.consumer_idx < delta_views.size()
                   && slot < delta_views[edge.consumer_idx].size()
                   && !delta_views[edge.consumer_idx][slot].empty())
                {
                    delta_views[i][0][0] = delta_views[edge.consumer_idx][slot][0];
                }
            }
        }
    }
}

void BackPropagation::accumulate_output_deltas(size_t layer_index)
{
    if(layer_index >= delta_views.size()) return;
    if(delta_views[layer_index].empty()) return;
    if(backward_edges[layer_index].size() <= 1) return;

    TensorView& destination = delta_views[layer_index][0][0];
    if(!destination.data) return;

    bool seeded = false;
    for(const BackwardEdge& edge : backward_edges[layer_index])
    {
        const size_t slot = 1 + edge.port;

        if(edge.consumer_idx >= delta_views.size()) continue;
        const auto& consumer_views = delta_views[edge.consumer_idx];
        if(slot >= consumer_views.size() || consumer_views[slot].empty()) continue;

        const TensorView& source = consumer_views[slot][0];
        if(!source.data || source.size() != destination.size()) continue;

        if(!seeded)
        {
            copy(source, destination);
            seeded = true;
        }
        else
        {
            addition(destination, source, destination);
        }
    }

    if(!seeded)
        destination.fill(0.0f);
}

const NeuralNetwork* BackPropagation::get_neural_network() const
{
    return neural_network;
}

vector<vector<TensorView>> BackPropagation::get_layer_gradients() const
{
    const NeuralNetwork* neural_network_ptr = loss->get_neural_network();

    const size_t layers_number = neural_network_ptr->get_layers_number();

    vector<vector<TensorView>> layer_gradient_views(layers_number);

    return layer_gradient_views;
}

TensorView BackPropagation::get_output_deltas() const
{
#ifdef OPENNN_WITH_CUDA
    if (Configuration::instance().is_gpu())
        return output_deltas_view_device;
#endif
    return {const_cast<float*>(output_deltas.as<float>()), output_delta_dimensions};
}

void BackPropagation::print() const
{
    cout << "Back-propagation" << "\n"
         << "Error: " << error << "\n"
         << "Loss:  " << loss_value << "\n";
}

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.