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

    const size_t layers_number = neural_network->get_layers_number();

    const vector<vector<Shape>> parameter_shapes = neural_network->get_parameter_shapes();

    Index total_parameters_size = 0;

    for(const auto& layer_shapes : parameter_shapes)
        for(const Shape& s : layer_shapes)
            total_parameters_size += get_aligned_size(s.size());

    gradient.setZero(total_parameters_size);

    gradient_views.resize(layers_number);
    type* g_ptr = (total_parameters_size > 0) ? gradient.data() : nullptr;

    for(Index i = 0; i < layers_number; ++i)
    {
        const vector<Shape>& layer_param_shapes = parameter_shapes[i];
        gradient_views[i].resize(layer_param_shapes.size());

        for(size_t j = 0; j < layer_param_shapes.size(); ++j)
        {
            const Shape& s = layer_param_shapes[j];
            if(s.size() > 0 && g_ptr)
            {
                gradient_views[i][j] = TensorView(g_ptr, s, CUDNN_DATA_FLOAT);
                g_ptr += get_aligned_size(s.size());
            }
        }
    }

    const vector<vector<Shape>> backward_shapes = neural_network->get_backward_shapes(batch_size);

    Index total_backward_size = 0;

    for(const auto& layer_shapes : backward_shapes)
        for(const Shape& s : layer_shapes)
            total_backward_size += get_aligned_size(s.size());

    backward.setZero(total_backward_size);

    backward_views.resize(layers_number);
    type* b_ptr = (total_backward_size > 0) ? backward.data() : nullptr;

    for(Index i = 0; i < layers_number; ++i)
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
    output_gradients.setZero(total_output_elements);

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
    for(Index i = 0; i < layers_number; ++i)
    {
        if(i == last_trainable_layer_index) continue;
        if(backward_edges[i].size() <= 1) continue;
        const Shape output_shape_i = layers_ref[i]->get_output_shape();
        if(output_shape_i.empty()) continue;
        per_layer_output_gradient_shapes[i] = Shape({batch_size}).append(output_shape_i);
        total_output_gradient_size += get_aligned_size(per_layer_output_gradient_shapes[i].size());
    }

    per_layer_output_gradients.setZero(total_output_gradient_size);

    type* og_ptr = (total_output_gradient_size > 0) ? per_layer_output_gradients.data() : nullptr;

    for(Index i = 0; i < layers_number; ++i)
    {
        if(backward_views[i].empty()) continue;

        if(i == last_trainable_layer_index)
        {
            backward_views[i][0][0] = TensorView(output_gradients.data(), output_gradient_dimensions);
        }
        else if(!backward_edges[i].empty())
        {
            if(backward_edges[i].size() > 1 && og_ptr && !per_layer_output_gradient_shapes[i].empty())
            {
                backward_views[i][0][0] = TensorView(og_ptr, per_layer_output_gradient_shapes[i]);
                og_ptr += get_aligned_size(per_layer_output_gradient_shapes[i].size());
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

void BackPropagation::accumulate_output_gradients(size_t layer_index)
{
    if(layer_index >= backward_views.size()) return;
    if(backward_views[layer_index].empty()) return;
    if(backward_edges[layer_index].size() <= 1) return;

    TensorView& destination = backward_views[layer_index][0][0];
    if(!destination.data) return;

    destination.fill(0.0f);

    for(const BackwardEdge& edge : backward_edges[layer_index])
    {
        const size_t slot = 1 + edge.port;

        if(edge.consumer_idx >= backward_views.size()) continue;
        const auto& consumer_views = backward_views[edge.consumer_idx];
        if(slot >= consumer_views.size() || consumer_views[slot].empty()) continue;

        const TensorView& source = consumer_views[slot][0];
        if(!source.data || source.size() != destination.size()) continue;

        addition(destination, source, destination);
    }
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

    {
        const auto& nn_layers_for_sizing = neural_network->get_layers();
        const vector<vector<Shape>> backward_shapes_for_sizing = neural_network->get_backward_shapes(batch_size);
        Index total_backward_bytes = 0;
        for(Index i = 0; i < layers_number; ++i)
        {
            const vector<Shape>& shapes = backward_shapes_for_sizing[i];
            const vector<cudnnDataType_t> dtypes = nn_layers_for_sizing[i]->get_backward_dtypes(batch_size);
            for(size_t j = 0; j < shapes.size(); ++j)
                if(shapes[j].size() > 0)
                    total_backward_bytes += get_aligned_bytes(shapes[j].size() * dtype_bytes(dtypes[j]));
        }
        backward.resize_device_bytes(total_backward_bytes);
        backward.setZero_device();
    }

    output_gradients.resize_device_bytes(output_gradients.size() * Index(dtype_bytes(CUDNN_ACTIVATION_DTYPE)));
    output_gradients.setZero_device();

    const vector<vector<Shape>> parameter_shapes = neural_network->get_parameter_shapes();

    if(gradient.size() > 0)
    {
        type* dev_g_ptr = gradient.device();

        for(Index i = 0; i < layers_number; ++i)
        {
            const vector<Shape>& layer_param_shapes = parameter_shapes[i];

            for(size_t j = 0; j < layer_param_shapes.size(); ++j)
            {
                const Shape& s = layer_param_shapes[j];
                if(s.size() > 0 && j < gradient_views[i].size())
                {
                    gradient_views[i][j].data = dev_g_ptr;
                    dev_g_ptr += get_aligned_size(s.size());
                }
            }
        }
    }

    const vector<vector<Shape>> backward_shapes = neural_network->get_backward_shapes(batch_size);
    const Index last_trainable_layer_index = neural_network->get_last_trainable_layer_index();

    if(per_layer_output_gradients.size() > 0)
    {
        Index total_og_bytes = 0;
        for(Index i = 0; i < layers_number; ++i)
            if(!per_layer_output_gradient_shapes[i].empty())
                total_og_bytes += get_aligned_bytes(per_layer_output_gradient_shapes[i].size()
                                                    * dtype_bytes(CUDNN_ACTIVATION_DTYPE));

        per_layer_output_gradients.resize_device_bytes(total_og_bytes);
        per_layer_output_gradients.setZero_device();
    }

    if(backward.size() > 0)
    {
        uint8_t* dev_b_cursor = backward.device_bytes();

        for(Index i = 0; i < layers_number; ++i)
        {
            const vector<Shape>& shapes = backward_shapes[i];
            const vector<cudnnDataType_t> dtypes = neural_network->get_layers()[i]->get_backward_dtypes(batch_size);

            for(size_t j = 0; j < shapes.size(); ++j)
            {
                const Shape& s = shapes[j];
                if(s.size() > 0)
                {
                    backward_views[i][j + 1][0].data  = reinterpret_cast<type*>(dev_b_cursor);
                    backward_views[i][j + 1][0].dtype = dtypes[j];
                    dev_b_cursor += get_aligned_bytes(s.size() * dtype_bytes(dtypes[j]));
                }
            }
        }

        for(Index i = 0; i < layers_number; ++i)
        {
            if(backward_views[i].empty()) continue;

            if(i == last_trainable_layer_index)
            {
                backward_views[i][0][0] = TensorView(output_gradients.device(), output_gradient_dimensions);
            }
            else if(!backward_edges[i].empty())
            {
                if(backward_edges[i].size() > 1 && !per_layer_output_gradient_shapes[i].empty()
                   && per_layer_output_gradients.device_bytes())
                {
                    uint8_t* og_cursor = per_layer_output_gradients.device_bytes();
                    for(size_t k = 0; k < i; ++k)
                        if(!per_layer_output_gradient_shapes[k].empty())
                            og_cursor += get_aligned_bytes(per_layer_output_gradient_shapes[k].size()
                                                           * dtype_bytes(CUDNN_ACTIVATION_DTYPE));

                    backward_views[i][0][0].data  = reinterpret_cast<type*>(og_cursor);
                    backward_views[i][0][0].shape = per_layer_output_gradient_shapes[i];
                    backward_views[i][0][0].dtype = CUDNN_ACTIVATION_DTYPE;
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

    if(errors_device) { cudaFree(errors_device); errors_device = nullptr; }
    CHECK_CUDA(cudaMalloc(&errors_device, batch_size * outputs_number * sizeof(float)));

    output_gradients_view_device = TensorView(output_gradients.device(), output_gradient_dimensions);
#endif
}


#ifdef OPENNN_WITH_CUDA

const TensorView& BackPropagation::get_output_gradients_device() const
{
    return output_gradients_view_device;
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
