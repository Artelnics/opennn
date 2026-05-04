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

    if (!loss)
        throw runtime_error("loss is not set.");

    const NeuralNetwork* neural_network = loss->get_neural_network();

    if (!neural_network)
        throw runtime_error("neural network is not set.");

    const auto& layers = neural_network->get_layers();
    const size_t layers_number = layers.size();
    const vector<vector<Shape>> parameter_shapes = neural_network->get_parameter_shapes();
    const vector<vector<Shape>> backward_shapes  = neural_network->get_backward_shapes(batch_size);
    const Shape output_shape = neural_network->get_output_shape();
    const Index last_trainable_layer_index = neural_network->get_last_trainable_layer_index();
    const auto& layer_input_indices = neural_network->get_layer_input_indices();

    output_delta_dimensions = Shape({batch_size}).append(output_shape);

    loss_value = 0.0f;
    error = 0.0f;
    accuracy.setZero();

    backward_edges.assign(layers_number, {});
    for (size_t consumer_index = 0; consumer_index < layers_number; ++consumer_index)
    {
        const vector<Index>& inputs = layer_input_indices[consumer_index];
        for (size_t port_index = 0; port_index < inputs.size(); ++port_index)
        {
            const Index producer = inputs[port_index];
            if (producer >= 0 && static_cast<size_t>(producer) < layers_number)
                backward_edges[producer].push_back({consumer_index, port_index});
        }
    }

    per_layer_output_delta_shapes.assign(layers_number, Shape());
    for (Index i = 0; i < layers_number; ++i)
    {
        if (i == last_trainable_layer_index)        continue;
        if (backward_edges[i].size() <= 1)          continue;
        const Shape output_shape_i = layers[i]->get_output_shape();
        if (output_shape_i.empty())                 continue;
        per_layer_output_delta_shapes[i] = Shape({batch_size}).append(output_shape_i);
    }

    gradient_views.resize(layers_number);
    delta_views.resize(layers_number);

    const bool is_gpu = Configuration::instance().is_gpu();
    const Device device = is_gpu ? Device::CUDA : Device::CPU;

    const Type activation_dtype = is_gpu ? neural_network->get_training_type() : Type::FP32;
    const Index activation_bytes = type_bytes(activation_dtype);

    vector<vector<Type>> backward_dtypes(layers_number);
    for (Index i = 0; i < layers_number; ++i)
    {
        backward_dtypes[i] = layers[i]->get_backward_dtypes(batch_size);
        if (!is_gpu)
            std::fill(backward_dtypes[i].begin(), backward_dtypes[i].end(), Type::FP32);
    }

    const Index total_gradient_bytes = aligned_total_elements(parameter_shapes) * Index(sizeof(float));
    if (total_gradient_bytes > 0)
    {
        gradient.resize_bytes(total_gradient_bytes, device);
        gradient.setZero();
    }

    uint8_t* g_cursor = gradient.as<uint8_t>();
    for (Index i = 0; i < layers_number; ++i)
    {
        const vector<Shape>& layer_param_shapes = parameter_shapes[i];
        gradient_views[i].resize(layer_param_shapes.size());

        for (size_t j = 0; j < layer_param_shapes.size(); ++j)
        {
            const Shape& slot_shape = layer_param_shapes[j];
            if (slot_shape.size() > 0)
            {
                gradient_views[i][j] = TensorView(g_cursor, slot_shape, Type::FP32);
                g_cursor += get_aligned_bytes(slot_shape.size() * Index(sizeof(float)));
            }
        }
    }

    const Index total_backward_bytes = aligned_total_bytes(backward_shapes, backward_dtypes);

    if (total_backward_bytes > 0)
    {
        backward.resize_bytes(total_backward_bytes, device);
        backward.setZero();
    }

    uint8_t* b_cursor = backward.as<uint8_t>();
    for (Index i = 0; i < layers_number; ++i)
    {
        const vector<Shape>& shapes = backward_shapes[i];
        const size_t slots = shapes.size();

        delta_views[i].assign(slots + 1, vector<TensorView>(1));

        for (size_t j = 0; j < slots; ++j)
        {
            const Shape& slot_shape = shapes[j];

            if (slot_shape.size() > 0)
            {
                delta_views[i][j + 1][0] = TensorView(b_cursor, slot_shape, backward_dtypes[i][j]);
                b_cursor += get_aligned_bytes(slot_shape.size() * type_bytes(backward_dtypes[i][j]));
            }
        }
    }

    const Index total_output_delta_elems = batch_size * output_shape.size();
    if (total_output_delta_elems > 0)
    {
        output_deltas.resize_bytes(total_output_delta_elems * activation_bytes, device);
        output_deltas.setZero();
    }

    const Index total_og_bytes = aligned_total_bytes(per_layer_output_delta_shapes, activation_dtype);

    if (total_og_bytes > 0)
    {
        per_layer_output_deltas.resize_bytes(total_og_bytes, device);
        per_layer_output_deltas.setZero();
    }

    uint8_t* og_cursor = per_layer_output_deltas.as<uint8_t>();
    for (Index i = 0; i < layers_number; ++i)
    {
        if (delta_views[i].empty()) continue;

        if (i == last_trainable_layer_index)
        {
            delta_views[i][0][0] = TensorView(output_deltas.as<uint8_t>(),
                                              output_delta_dimensions, activation_dtype);
            continue;
        }

        if (backward_edges[i].empty()) continue;

        if (backward_edges[i].size() > 1 && !per_layer_output_delta_shapes[i].empty())
        {
            delta_views[i][0][0] = TensorView(og_cursor,
                                              per_layer_output_delta_shapes[i], activation_dtype);
            og_cursor += get_aligned_bytes(per_layer_output_delta_shapes[i].size() * activation_bytes);
            continue;
        }

        const BackwardEdge& edge = backward_edges[i].front();
        const size_t slot = 1 + edge.port;
        if (edge.consumer_idx < delta_views.size()
            && slot < delta_views[edge.consumer_idx].size()
            && !delta_views[edge.consumer_idx][slot].empty())
        {
            delta_views[i][0][0] = delta_views[edge.consumer_idx][slot][0];
        }
    }

#ifdef OPENNN_WITH_CUDA
    if (is_gpu)
    {
        const Index outputs_number = output_shape[0];
        errors_device.resize_bytes(batch_size * outputs_number * Index(sizeof(float)), Device::CUDA);

        output_deltas_view_device = TensorView(output_deltas.as<float>(),
                                               output_delta_dimensions,
                                               activation_dtype);
    }
#endif
}

void BackPropagation::accumulate_output_deltas(size_t layer_index)
{
    if (layer_index >= delta_views.size()) return;
    if (delta_views[layer_index].empty()) return;
    if (backward_edges[layer_index].size() <= 1) return;

    TensorView& destination = delta_views[layer_index][0][0];
    if (!destination.data) return;

    vector<const TensorView*> sources;
    sources.reserve(backward_edges[layer_index].size());
    for (const BackwardEdge& edge : backward_edges[layer_index])
    {
        const size_t slot = 1 + edge.port;

        if (edge.consumer_idx >= delta_views.size()) continue;
        const auto& consumer_views = delta_views[edge.consumer_idx];
        if (slot >= consumer_views.size() || consumer_views[slot].empty()) continue;

        const TensorView& source = consumer_views[slot][0];
        if (!source.data || source.size() != destination.size()) continue;

        sources.push_back(&source);
    }

    if (sources.empty())
    {
        destination.fill(0.0f);
        return;
    }

    if (sources.size() == 1)
    {
        copy(*sources[0], destination);
        return;
    }

#ifdef OPENNN_WITH_CUDA
    if (Configuration::instance().is_gpu())
    {
        copy(*sources[0], destination);
        for (size_t s = 1; s < sources.size(); ++s)
            addition(destination, *sources[s], destination);
        return;
    }
#endif

    const Index n = destination.size();
    float* dst = destination.as<float>();
    const size_t k = sources.size();

    if (k == 2)
    {
        const float* p0 = sources[0]->as<float>();
        const float* p1 = sources[1]->as<float>();
        #pragma omp parallel for
        for (Index i = 0; i < n; ++i) dst[i] = p0[i] + p1[i];
        return;
    }

    vector<const float*> ptrs;
    ptrs.reserve(k);
    for (size_t s = 0; s < k; ++s) ptrs.push_back(sources[s]->as<float>());

    #pragma omp parallel for
    for (Index i = 0; i < n; ++i)
    {
        float sum = ptrs[0][i];
        for (size_t s = 1; s < k; ++s) sum += ptrs[s][i];
        dst[i] = sum;
    }
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