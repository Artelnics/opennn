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

    const size_t layers_number = neural_network->get_layers_number();

    const vector<vector<Shape>> parameter_shapes = neural_network->get_parameter_shapes();

    Index total_parameters_size = 0;

    for(const auto& layer_shapes : parameter_shapes)
        for(const Shape& s : layer_shapes)
            total_parameters_size += get_aligned_size(s.size());

    gradient.resize(total_parameters_size);
    gradient.setZero();

    gradient_views.resize(layers_number);
    type* g_ptr = (total_parameters_size > 0) ? gradient.data() : nullptr;

    for(size_t i = 0; i < layers_number; ++i)
    {
        const vector<Shape>& layer_param_shapes = parameter_shapes[i];
        gradient_views[i].resize(layer_param_shapes.size());

        for(size_t j = 0; j < layer_param_shapes.size(); ++j)
        {
            const Shape& s = layer_param_shapes[j];
            if(s.size() > 0 && g_ptr)
            {
                gradient_views[i][j] = TensorView(g_ptr, s);
                g_ptr += get_aligned_size(s.size());
            }
        }
    }

    const vector<vector<Shape>> backward_shapes = neural_network->get_backward_shapes(batch_size);

    Index total_backward_size = 0;

    for(const auto& layer_shapes : backward_shapes)
        for(const Shape& s : layer_shapes)
            total_backward_size += get_aligned_size(s.size());

    backward.resize(total_backward_size);
    backward.setZero();

    backward_views.resize(layers_number);
    type* b_ptr = (total_backward_size > 0) ? backward.data() : nullptr;

    for(size_t i = 0; i < layers_number; ++i)
    {
        const vector<Shape>& shapes = backward_shapes[i];
        const size_t slots = shapes.size();

        backward_views[i].resize(slots + 1);
        backward_views[i][0].resize(1);

        for(size_t j = 0; j < slots; ++j)
        {
            const Shape& s = shapes[j];
            backward_views[i][j + 1].resize(1);

            if(s.size() > 0 && b_ptr)
            {
                backward_views[i][j + 1][0] = TensorView(b_ptr, s);
                b_ptr += get_aligned_size(s.size());
            }
        }
    }

    const Shape output_shape = neural_network->get_output_shape();
    const Index outputs_number = output_shape[0];

    loss_value = type(0);
    error = type(0);
    built_mask = false;
    accuracy.setZero();

    errors.resize(batch_size, outputs_number);

    output_gradient_dimensions = Shape({batch_size}).append(output_shape);

    const Index total_output_elements = output_shape.size() * batch_size;
    output_gradients.resize(total_output_elements);
    output_gradients.setZero();

    const Index last_trainable_layer_index = neural_network->get_last_trainable_layer_index();
    const auto& layer_input_indices = neural_network->get_layer_input_indices();

    backward_edges.assign(layers_number, {});
    for(size_t c = 0; c < layers_number; ++c)
    {
        const vector<Index>& inputs = layer_input_indices[c];
        for(size_t p = 0; p < inputs.size(); ++p)
        {
            const Index producer = inputs[p];
            if(producer >= 0 && static_cast<size_t>(producer) < layers_number)
                backward_edges[producer].push_back({c, p});
        }
    }

    per_layer_output_gradient_shapes.assign(layers_number, Shape());
    Index total_output_gradient_size = 0;
    const auto& layers_ref = neural_network->get_layers();
    for(size_t i = 0; i < layers_number; ++i)
    {
        if(static_cast<Index>(i) == last_trainable_layer_index) continue;
        const Shape output_shape_i = layers_ref[i]->get_output_shape();
        if(output_shape_i.empty()) continue;
        per_layer_output_gradient_shapes[i] = Shape({batch_size}).append(output_shape_i);
        total_output_gradient_size += get_aligned_size(per_layer_output_gradient_shapes[i].size());
    }

    per_layer_output_gradients.resize(total_output_gradient_size);
    per_layer_output_gradients.setZero();

    type* og_ptr = (total_output_gradient_size > 0) ? per_layer_output_gradients.data() : nullptr;

    for(size_t i = 0; i < layers_number; ++i)
    {
        if(backward_views[i].empty()) continue;

        if(static_cast<Index>(i) == last_trainable_layer_index)
        {
            backward_views[i][0][0] = TensorView(output_gradients.data(), output_gradient_dimensions);
        }
        else if(!backward_edges[i].empty())
        {
            if(backward_edges[i].size() > 1 && og_ptr && !per_layer_output_gradient_shapes[i].empty())
            {
                // Multi-consumer: dedicated buffer in per_layer_output_gradients (accumulation path)
                backward_views[i][0][0] = TensorView(og_ptr, per_layer_output_gradient_shapes[i]);
                og_ptr += get_aligned_size(per_layer_output_gradient_shapes[i].size());
            }
            else
            {
                // Single-consumer: alias to consumer's input gradient (no accumulation needed)
                const BackwardEdge& edge = backward_edges[i].front();
                const size_t slot = 1 + edge.port;
                if(edge.consumer_idx < backward_views.size()
                   && slot < backward_views[edge.consumer_idx].size()
                   && !backward_views[edge.consumer_idx][slot].empty())
                {
                    backward_views[i][0][0] = backward_views[edge.consumer_idx][slot][0];
                }
            }
        }
    }
}

void BackPropagation::accumulate_output_gradients(size_t layer_index)
{
    if(layer_index >= backward_views.size()) return;
    if(backward_views[layer_index].empty()) return;
    if(backward_edges[layer_index].size() <= 1) return;  // Single consumer: already wired, no accumulation needed

    TensorView& output_grad = backward_views[layer_index][0][0];
    if(!output_grad.data) return;

    const Index n = output_grad.size();
    type* out_ptr = output_grad.data;

#ifndef OPENNN_WITH_CUDA
    std::fill(out_ptr, out_ptr + n, type(0));

    for(const BackwardEdge& edge : backward_edges[layer_index])
    {
        const size_t slot = 1 + edge.port;
        if(edge.consumer_idx >= backward_views.size()) continue;
        const auto& consumer_views = backward_views[edge.consumer_idx];
        if(slot >= consumer_views.size()) continue;
        if(consumer_views[slot].empty()) continue;
        const TensorView& src = consumer_views[slot][0];
        if(!src.data) continue;
        if(src.size() != n) continue;

        for(Index k = 0; k < n; ++k)
            out_ptr[k] += src.data[k];
    }
#else
    cudaMemset(out_ptr, 0, n * sizeof(float));

    for(const BackwardEdge& edge : backward_edges[layer_index])
    {
        const size_t slot = 1 + edge.port;
        if(edge.consumer_idx >= backward_views.size()) continue;
        const auto& consumer_views = backward_views[edge.consumer_idx];
        if(slot >= consumer_views.size()) continue;
        if(consumer_views[slot].empty()) continue;
        const TensorView& src = consumer_views[slot][0];
        if(!src.data) continue;
        if(src.size() != n) continue;

        addition_cuda(n, out_ptr, src.data, out_ptr);
    }
#endif
}

void BackPropagation::allocate_device()
{
#ifdef OPENNN_WITH_CUDA
    if(!loss || batch_size <= 0)
        return;

    const NeuralNetwork* neural_network = loss->get_neural_network();
    if(!neural_network)
        throw runtime_error("BackPropagation error: neural network is not set.");

    const size_t layers_number = neural_network->get_layers_number();
    const Shape output_shape = neural_network->get_output_shape();
    const Index outputs_number = output_shape[0];

    gradient.resize_device(gradient.size());
    gradient.setZero_device();
    backward.resize_device(backward.size());
    backward.setZero_device();
    output_gradients.resize_device(output_gradients.size());
    output_gradients.setZero_device();

    const vector<vector<Shape>> parameter_shapes = neural_network->get_parameter_shapes();

    if(gradient.size() > 0)
    {
        type* dev_g_ptr = gradient.device();

        for(size_t i = 0; i < layers_number; ++i)
        {
            const vector<Shape>& layer_param_shapes = parameter_shapes[i];

            for(size_t j = 0; j < layer_param_shapes.size(); ++j)
            {
                const Shape& s = layer_param_shapes[j];
                if(s.size() > 0 && j < gradient_views[i].size())
                {
                    gradient_views[i][j].data = dev_g_ptr;
                    gradient_views[i][j].set_descriptor(s);
                    dev_g_ptr += get_aligned_size(s.size());
                }
            }
        }
    }

    const vector<vector<Shape>> backward_shapes = neural_network->get_backward_shapes(batch_size);
    const Index last_trainable_layer_index = neural_network->get_last_trainable_layer_index();

    // Allocate per-layer output gradient buffers on device (for multi-consumer gradient accumulation)
    if(per_layer_output_gradients.size() > 0)
    {
        per_layer_output_gradients.resize_device(per_layer_output_gradients.size());
        per_layer_output_gradients.setZero_device();
    }

    if(backward.size() > 0)
    {
        type* dev_b_ptr = backward.device();

        for(size_t i = 0; i < layers_number; ++i)
        {
            const vector<Shape>& shapes = backward_shapes[i];

            for(size_t j = 0; j < shapes.size(); ++j)
            {
                const Shape& s = shapes[j];
                if(s.size() > 0)
                {
                    backward_views[i][j + 1][0].data = dev_b_ptr;
                    backward_views[i][j + 1][0].set_descriptor(s);
                    dev_b_ptr += get_aligned_size(s.size());
                }
            }
        }

        for(size_t i = 0; i < layers_number; ++i)
        {
            if(backward_views[i].empty()) continue;

            if(static_cast<Index>(i) == last_trainable_layer_index)
            {
                TensorView og_view(output_gradients.device(), output_gradient_dimensions);
                og_view.set_descriptor(output_gradient_dimensions);
                backward_views[i][0][0] = og_view;
            }
            else if(!backward_edges[i].empty())
            {
                if(backward_edges[i].size() > 1 && !per_layer_output_gradient_shapes[i].empty()
                   && per_layer_output_gradients.device())
                {
                    // Multi-consumer: use dedicated device buffer for accumulation
                    type* dev_og = per_layer_output_gradients.device();
                    for(size_t k = 0; k < i; k++)
                        if(!per_layer_output_gradient_shapes[k].empty())
                            dev_og += get_aligned_size(per_layer_output_gradient_shapes[k].size());

                    backward_views[i][0][0].data = dev_og;
                    backward_views[i][0][0].set_descriptor(per_layer_output_gradient_shapes[i]);
                }
                else
                {
                    const BackwardEdge& edge = backward_edges[i].front();
                    const size_t slot = 1 + edge.port;
                    if(edge.consumer_idx < backward_views.size()
                       && slot < backward_views[edge.consumer_idx].size()
                       && !backward_views[edge.consumer_idx][slot].empty())
                    {
                        backward_views[i][0][0] = backward_views[edge.consumer_idx][slot][0];
                    }
                }
            }
        }
    }

    CHECK_CUDA(cudaMalloc(&errors_device, batch_size * outputs_number * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&error_device, 2 * sizeof(float)));

    output_gradients_view_device = TensorView(output_gradients.device(), output_gradient_dimensions);
    output_gradients_view_device.set_descriptor(output_gradient_dimensions);
#endif
}


#ifdef OPENNN_WITH_CUDA

const TensorView& BackPropagation::get_output_gradients_device() const
{
    return output_gradients_view_device;
}

void BackPropagation::free_cuda()
{
    if(errors_device) { cudaFree(errors_device); errors_device = nullptr; }
    if(error_device) { cudaFree(error_device); error_device = nullptr; }
}

#endif


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

TensorView BackPropagation::get_output_gradients() const
{
    return {const_cast<type*>(output_gradients.data()), output_gradient_dimensions};
}

void BackPropagation::print() const
{
    cout << "Back-propagation" << "\n"
         << "Errors:" << "\n"
         << errors << "\n"
         << "Error:" << "\n"
         << error << "\n"
         << "Loss:" << "\n"
         << loss_value << "\n";
}

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
