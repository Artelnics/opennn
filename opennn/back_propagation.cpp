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

    const Index total_parameters_size = aligned_total_elements(parameter_shapes);

    // gradient/backward/output_deltas are pure scratch — no Eigen-based init
    // reads them before the first backward pass, so the CPU allocation is
    // pointless when we'll run on GPU. allocate_device() builds the GPU buffer
    // and rebuilds the views; here we only need the view structures sized.
    const bool gpu_mode = Configuration::instance().is_gpu();

    if(!gpu_mode)
    {
        gradient.resize_bytes(total_parameters_size * Index(sizeof(type)), DeviceType::CPU);
        gradient.setZero();
    }

    gradient_views.resize(layers_number);
    type* g_ptr = (total_parameters_size > 0 && !gpu_mode) ? gradient.as<type>() : nullptr;

    for(Index i = 0; i < layers_number; ++i)
    {
        const vector<Shape>& layer_param_shapes = parameter_shapes[i];
        gradient_views[i].resize(layer_param_shapes.size());

        for(size_t j = 0; j < layer_param_shapes.size(); ++j)
        {
            const Shape& s = layer_param_shapes[j];
            if(s.size() > 0)
            {
                // GPU mode: g_ptr is null; allocate_device() will fill the data
                // pointer. Shape is set unconditionally so consumers don't see empty.
                gradient_views[i][j] = TensorView(g_ptr, s, CUDNN_DATA_FLOAT);
                if(g_ptr) g_ptr += get_aligned_size(s.size());
            }
        }
    }

    const vector<vector<Shape>> backward_shapes = neural_network->get_backward_shapes(batch_size);

    const Index total_backward_size = aligned_total_elements(backward_shapes);

    if(!gpu_mode)
    {
        backward.resize_bytes(total_backward_size * Index(sizeof(type)), DeviceType::CPU);
        backward.setZero();
    }

    delta_views.resize(layers_number);
    type* b_ptr = (total_backward_size > 0 && !gpu_mode) ? backward.as<type>() : nullptr;

    for(Index i = 0; i < layers_number; ++i)
    {
        const vector<Shape>& shapes = backward_shapes[i];
        const size_t slots = shapes.size();

        delta_views[i].resize(slots + 1);
        delta_views[i][0].resize(1);

        for(size_t j = 0; j < slots; ++j)
        {
            const Shape& s = shapes[j];
            delta_views[i][j + 1].resize(1);

            if(s.size() > 0)
            {
                // GPU mode: b_ptr is null; allocate_device() will fill data.
                delta_views[i][j + 1][0] = TensorView(b_ptr, s);
                if(b_ptr) b_ptr += get_aligned_size(s.size());
            }
        }
    }

    const Shape output_shape = neural_network->get_output_shape();

    loss_value = type(0);
    error = type(0);
    accuracy.setZero();

    output_delta_dimensions = Shape({batch_size}).append(output_shape);

    const Index total_output_elements = output_shape.size() * batch_size;
    if(!gpu_mode)
    {
        output_deltas.resize_bytes(total_output_elements * Index(sizeof(type)), DeviceType::CPU);
        output_deltas.setZero();
    }

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

    per_layer_output_delta_shapes.assign(layers_number, Shape());
    Index total_output_delta_size = 0;
    const auto& layers_ref = neural_network->get_layers();
    for(Index i = 0; i < layers_number; ++i)
    {
        if(i == last_trainable_layer_index) continue;
        if(backward_edges[i].size() <= 1) continue;
        const Shape output_shape_i = layers_ref[i]->get_output_shape();
        if(output_shape_i.empty()) continue;
        per_layer_output_delta_shapes[i] = Shape({batch_size}).append(output_shape_i);
        total_output_delta_size += get_aligned_size(per_layer_output_delta_shapes[i].size());
    }

    if(!gpu_mode)
    {
        per_layer_output_deltas.resize_bytes(total_output_delta_size * Index(sizeof(type)), DeviceType::CPU);
        per_layer_output_deltas.setZero();
    }

    type* og_ptr = (total_output_delta_size > 0 && !gpu_mode) ? per_layer_output_deltas.as<type>() : nullptr;

    for(Index i = 0; i < layers_number; ++i)
    {
        if(delta_views[i].empty()) continue;

        if(i == last_trainable_layer_index)
        {
            delta_views[i][0][0] = TensorView(output_deltas.as<type>(), output_delta_dimensions);
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

    // Copy the first valid source instead of fill(0) + addition. Saves one full
    // memset of the destination (B·S·E · sizeof(T) bytes) plus one cudnnOpTensor
    // launch per multi-consumer layer. The remaining edges still accumulate
    // via addition into the seeded destination.
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

    // No edge contributed (every one was filtered out): zero the destination so
    // downstream layers see the correct "no gradient" value. Rare path.
    if(!seeded)
        destination.fill(0.0f);
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

    // Activation dtype for the output_delta / per_layer_output_delta arenas. Reads from
    // the network's resolved Configuration. FP32 if Float32 training, BF16 if BP16.
    const cudnnDataType_t activation_dtype = neural_network->get_training_cudnn_dtype();
    const Index activation_bytes = dtype_bytes(activation_dtype);

    // Compute sizes from shapes — set() may have skipped the CPU allocation when
    // running in GPU mode, so we can't read sizes from the CPU buffers.
    const Index total_gradient_floats   = aligned_total_elements(neural_network->get_parameter_shapes());
    const Index total_output_delta_elems = batch_size * output_shape.size();

    gradient.resize_bytes(total_gradient_floats * Index(sizeof(float)), DeviceType::CUDA);
    gradient.setZero();

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
        backward.resize_bytes(total_backward_bytes, DeviceType::CUDA);
        backward.setZero();
    }

    output_deltas.resize_bytes(total_output_delta_elems * activation_bytes, DeviceType::CUDA);
    output_deltas.setZero();

    const vector<vector<Shape>> parameter_shapes = neural_network->get_parameter_shapes();

    if(gradient.size() > 0)
    {
        type* dev_g_ptr = gradient.as<type>();

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

    {
        Index total_og_bytes = 0;
        for(Index i = 0; i < layers_number; ++i)
            if(!per_layer_output_delta_shapes[i].empty())
                total_og_bytes += get_aligned_bytes(per_layer_output_delta_shapes[i].size()
                                                    * activation_bytes);

        if(total_og_bytes > 0)
        {
            per_layer_output_deltas.resize_bytes(total_og_bytes, DeviceType::CUDA);
            per_layer_output_deltas.setZero();
        }
    }

    if(backward.size() > 0)
    {
        uint8_t* dev_b_cursor = backward.as<uint8_t>();

        for(Index i = 0; i < layers_number; ++i)
        {
            const vector<Shape>& shapes = backward_shapes[i];
            const vector<cudnnDataType_t> dtypes = neural_network->get_layers()[i]->get_backward_dtypes(batch_size);

            for(size_t j = 0; j < shapes.size(); ++j)
            {
                const Shape& s = shapes[j];
                if(s.size() > 0)
                {
                    delta_views[i][j + 1][0].data  = dev_b_cursor;
                    delta_views[i][j + 1][0].dtype = dtypes[j];
                    dev_b_cursor += get_aligned_bytes(s.size() * dtype_bytes(dtypes[j]));
                }
            }
        }

        for(Index i = 0; i < layers_number; ++i)
        {
            if(delta_views[i].empty()) continue;

            if(i == last_trainable_layer_index)
            {
                delta_views[i][0][0] = TensorView(output_deltas.as<type>(),
                                                  output_delta_dimensions,
                                                  activation_dtype);
            }
            else if(!backward_edges[i].empty())
            {
                if(backward_edges[i].size() > 1 && !per_layer_output_delta_shapes[i].empty()
                   && per_layer_output_deltas.as<uint8_t>())
                {
                    uint8_t* og_cursor = per_layer_output_deltas.as<uint8_t>();
                    for(size_t k = 0; k < i; ++k)
                        if(!per_layer_output_delta_shapes[k].empty())
                            og_cursor += get_aligned_bytes(per_layer_output_delta_shapes[k].size()
                                                           * activation_bytes);

                    delta_views[i][0][0].data  = og_cursor;
                    delta_views[i][0][0].shape = per_layer_output_delta_shapes[i];
                    delta_views[i][0][0].dtype = activation_dtype;
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

    if(errors_device) { cudaFree(errors_device); errors_device = nullptr; }
    CHECK_CUDA(cudaMalloc(&errors_device, batch_size * outputs_number * sizeof(float)));

    output_deltas_view_device = TensorView(output_deltas.as<type>(),
                                           output_delta_dimensions,
                                           activation_dtype);
#endif
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
    return {const_cast<type*>(output_deltas.as<type>()), output_delta_dimensions};
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