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
#include "cuda_dispatch.h"

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

    neural_network = loss->get_neural_network();

    if (!neural_network)
        throw runtime_error("neural network is not set.");

    loss_value = 0.0f;
    error = 0.0f;
    accuracy = 0.0f;

    const auto& layers = neural_network->get_layers();
    const size_t layers_number = layers.size();
    const auto parameter_specs = neural_network->get_parameter_specs();
    const auto backward_specs  = neural_network->get_backward_specs(batch_size);
    const auto& layer_input_indices = neural_network->get_layer_input_indices();

    backward_edges.assign(layers_number, {});

    for (size_t i = 0; i < layers_number; ++i)
    {
        const vector<Index>& input_indices = layer_input_indices[i];
        for (size_t j = 0; j < input_indices.size(); ++j)
            if (const Index producer = input_indices[j]; producer >= 0 && size_t(producer) < layers_number)
                backward_edges[producer].push_back({i, j});
    }

    if (const Index first = neural_network->get_first_trainable_layer_index(); first >= 0)
    {
        const auto& ops = layers[first]->get_operators();
        if (!ops.empty()) ops[0]->input_delta_slots.clear();
    }

    const Device device = current_device();

    if (const Index gradient_bytes = get_aligned_bytes(parameter_specs, Type::FP32); gradient_bytes > 0)
    {
        gradient.resize_bytes(gradient_bytes, device);
        gradient.setZero();
    }

    gradient_views.resize(layers_number);

    float* pointer = gradient.as<float>();
    for (size_t i = 0; i < layers_number; ++i)
        pointer = layers[i]->link_gradients(pointer, gradient_views[i]);

    setup_delta_pool(backward_specs);
}

void BackPropagation::setup_delta_pool(const vector<vector<pair<Shape, Type>>>& backward_specs)
{
    const auto& layers = neural_network->get_layers();
    const Index layers_number = neural_network->get_layers_number();
    const Index first_trainable_layer_index = neural_network->get_first_trainable_layer_index();
    const Index last_trainable_layer_index = neural_network->get_last_trainable_layer_index();
    const auto& layer_input_indices = neural_network->get_layer_input_indices();
    const Type compute_dtype = is_gpu() ? neural_network->get_training_type() : Type::FP32;

    vector<Index> entry_layers;
    vector<size_t> entry_slots;
    vector<Index> entry_offsets;
    vector<Shape> entry_shapes;
    vector<Type> entry_dtypes;
    vector<Index> entry_births;
    vector<Index> entry_deaths;
    vector<Index> entry_bytes;

    vector<pair<size_t, size_t>> alias_target(size_t(layers_number), {SIZE_MAX, SIZE_MAX});

    Index max_step = 0;

    const Shape output_delta_shape = Shape({batch_size}).append(neural_network->get_output_shape());
    if (output_delta_shape.size() != 0)
    {
        entry_layers.push_back(last_trainable_layer_index);
        entry_slots.push_back(0);
        entry_offsets.push_back(Index(-1));
        entry_shapes.push_back(output_delta_shape);
        entry_dtypes.push_back(compute_dtype);
        entry_births.push_back(0);
        entry_deaths.push_back(0);
        entry_bytes.push_back(get_aligned_bytes(output_delta_shape.size(), compute_dtype));
    }

    for (Index layer_index = first_trainable_layer_index; layer_index <= last_trainable_layer_index; ++layer_index)
    {
        const auto& specs = backward_specs[layer_index];
        const auto& input_indices = layer_input_indices[layer_index];

        for (size_t j = 0; j < specs.size(); ++j)
        {
            const auto& [shape, dtype] = specs[j];
            if (shape.size() == 0) continue;

            const Index birth = last_trainable_layer_index - layer_index;
            const Index producer = (j < input_indices.size()) ? input_indices[j] : Index(-1);
            const bool producer_is_trainable = producer >= first_trainable_layer_index
                                            && producer <= last_trainable_layer_index;
            const Index death = producer_is_trainable ? last_trainable_layer_index - producer : birth;

            entry_layers.push_back(layer_index);
            entry_slots.push_back(j + 1);
            entry_offsets.push_back(Index(-1));
            entry_shapes.push_back(shape);
            entry_dtypes.push_back(dtype);
            entry_births.push_back(birth);
            entry_deaths.push_back(death);
            entry_bytes.push_back(get_aligned_bytes(shape.size(), dtype));

            max_step = max({max_step, birth, death});
        }
    }

    for (Index layer_index = first_trainable_layer_index; layer_index < last_trainable_layer_index; ++layer_index)
    {
        const auto& layer_backward_edges = backward_edges[layer_index];

        if (layer_backward_edges.empty()) continue;

        if (layer_backward_edges.size() == 1)
        {
            const auto& edge = layer_backward_edges.front();
            alias_target[layer_index] = {edge.first, edge.second + 1};
            continue;
        }

        const Shape output_shape = layers[layer_index]->get_output_shape();
        if (output_shape.empty()) continue;

        const Shape delta_shape = Shape({batch_size}).append(output_shape);
        const Index step = last_trainable_layer_index - layer_index;

        entry_layers.push_back(layer_index);
        entry_slots.push_back(0);
        entry_offsets.push_back(Index(-1));
        entry_shapes.push_back(delta_shape);
        entry_dtypes.push_back(compute_dtype);
        entry_births.push_back(step);
        entry_deaths.push_back(step);
        entry_bytes.push_back(get_aligned_bytes(delta_shape.size(), compute_dtype));

        max_step = max(max_step, step);
    }

    vector<vector<size_t>> births_by_step(size_t(max_step + 1));
    vector<vector<size_t>> deaths_by_step(size_t(max_step + 1));

    for (size_t id = 0; id < entry_layers.size(); ++id)
    {
        births_by_step[size_t(entry_births[id])].push_back(id);
        deaths_by_step[size_t(entry_deaths[id])].push_back(id);
    }

    vector<pair<Index, Index>> free_list = {{0, numeric_limits<Index>::max()}};
    Index peak_bytes = 0;

    for (Index step = 0; step <= max_step; ++step)
    {
        for (size_t id : births_by_step[size_t(step)])
        {
            const Index bytes = entry_bytes[id];
            Index offset = Index(-1);

            for (auto it = free_list.begin(); it != free_list.end(); ++it)
            {
                if (it->second < bytes) continue;

                offset = it->first;

                if (it->second == bytes)
                    free_list.erase(it);
                else
                {
                    it->first += bytes;
                    it->second -= bytes;
                }

                break;
            }

            entry_offsets[id] = offset;
            peak_bytes = max(peak_bytes, offset + bytes);
        }

        for (size_t id : deaths_by_step[size_t(step)])
        {
            auto it = free_list.begin();
            while (it != free_list.end() && it->first < entry_offsets[id])
                ++it;

            it = free_list.insert(it, {entry_offsets[id], entry_bytes[id]});

            if (it + 1 != free_list.end() && it->first + it->second == (it + 1)->first)
            {
                it->second += (it + 1)->second;
                free_list.erase(it + 1);
            }

            if (it != free_list.begin() && (it - 1)->first + (it - 1)->second == it->first)
            {
                (it - 1)->second += it->second;
                free_list.erase(it);
            }
        }
    }

    delta_views.resize(layers_number);
    for (Index i = 0; i < layers_number; ++i)
        delta_views[i].assign(backward_specs[i].size() + 1, TensorView{});

    if (peak_bytes > 0)
    {
        delta_pool.resize_bytes(peak_bytes, current_device());
        delta_pool.setZero();
    }

    uint8_t* const base = delta_pool.as<uint8_t>();

    for (size_t id = 0; id < entry_layers.size(); ++id)
        delta_views[entry_layers[id]][entry_slots[id]] =
            TensorView(base + entry_offsets[id], entry_shapes[id], entry_dtypes[id]);

    for (Index i = 0; i < layers_number; ++i)
        if (const auto [consumer, slot] = alias_target[i];
            consumer != SIZE_MAX && !delta_views[consumer][slot].empty())
            delta_views[i][0] = delta_views[consumer][slot];
}

void BackPropagation::accumulate_output_deltas(size_t layer_index)
{
    const auto& layer_backward_edges = backward_edges[layer_index];
    if (layer_backward_edges.size() <= 1) return;

    TensorView& destination = delta_views[layer_index][0];
    if (!destination.data) return;

    destination.setZero();

    for (const auto& edge : layer_backward_edges)
    {
        const TensorView& source = delta_views[edge.first][1 + edge.second];
        if (!source.data || source.size() != destination.size()) continue;

        add(destination, source, destination);
    }
}

TensorView& BackPropagation::get_output_delta()
{
    return delta_views[neural_network->get_last_trainable_layer_index()][0];
}

const TensorView& BackPropagation::get_output_delta() const
{
    return delta_views[neural_network->get_last_trainable_layer_index()][0];
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
