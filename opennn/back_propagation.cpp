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
    loss_pointer = new_loss;

    throw_if(!loss_pointer, "loss is not set.");

    neural_network = loss_pointer->get_neural_network();

    throw_if(!neural_network, "neural network is not set.");

    error = 0.0f;
    accuracy = 0.0f;
    regularization = 0.0f;
    loss = 0.0f;

    const auto& layers = neural_network->get_layers();
    const size_t layers_number = layers.size();
    const auto parameter_specs = neural_network->get_parameter_specs();
    const auto backward_specs  = neural_network->get_backward_specs(batch_size);
    const auto& source_layers = neural_network->get_source_layers();

    consumer_edges.assign(layers_number, {});

    for (size_t i = 0; i < layers_number; ++i)
    {
        const vector<Index>& sources = source_layers[i];
        for (size_t j = 0; j < sources.size(); ++j)
            if (const Index source_layer = sources[j]; source_layer >= 0
            && size_t(source_layer) < layers_number)
                consumer_edges[source_layer].push_back({i, j});
    }

    const Index gradient_bytes = get_aligned_bytes(parameter_specs, Type::FP32);
    gradient.resize_bytes(gradient_bytes, neural_network->get_device());
    gradient.setZero();

    gradient_views.resize(layers_number);

    float* pointer = gradient.as<float>();
    for (size_t i = 0; i < layers_number; ++i)
        pointer = layers[i]->link_gradients(pointer, gradient_views[i], gradient.device_type);

    setup_delta_pool(backward_specs);
}

void BackPropagation::setup_delta_pool(const vector<vector<TensorSpec>>& backward_specs)
{
    struct DeltaEntry
    {
        Index      layer;
        size_t     slot;
        TensorSpec spec;
        Index      first_step;
        Index      last_step;
        Index      byte_offset = -1;
    };

    const auto& layers = neural_network->get_layers();
    const Index layers_number = neural_network->get_layers_number();
    const Index first_trainable_layer_index = neural_network->get_first_trainable_layer_index();
    const Index last_trainable_layer_index = neural_network->get_last_trainable_layer_index();
    const auto& source_layers = neural_network->get_source_layers();
    const Type compute_dtype = neural_network->is_gpu()
        ? neural_network->get_training_type()
        : Type::FP32;

    vector<DeltaEntry> delta_entries;

    const Index last_backward_step = last_trainable_layer_index - first_trainable_layer_index;

    const Shape output_delta_shape = Shape({batch_size}).append(layers[last_trainable_layer_index]->get_output_shape());

    if (output_delta_shape.size() != 0)
        delta_entries.push_back({last_trainable_layer_index, 0, {output_delta_shape, compute_dtype}, 0, 0});

    for (Index layer_index = first_trainable_layer_index; layer_index <= last_trainable_layer_index; ++layer_index)
    {
        const auto& specs = backward_specs[layer_index];
        const auto& sources = source_layers[layer_index];

        for (size_t j = 0; j < specs.size(); ++j)
        {
            const auto& [shape, dtype] = specs[j];
            if (shape.size() == 0) continue;

            const Index first_step = last_trainable_layer_index - layer_index;
            const Index source_layer = (j < sources.size()) ? sources[j] : Index(-1);
            const bool source_layer_is_trainable = source_layer >= first_trainable_layer_index
                                                && source_layer <= last_trainable_layer_index;

            const bool is_input_delta = j < sources.size();
            // Deltas into external or non-trainable sources are not consumed.
            if (is_input_delta && !source_layer_is_trainable) continue;

            const Index last_step = source_layer_is_trainable ? last_trainable_layer_index - source_layer : first_step;

            delta_entries.push_back({layer_index, j + 1, {shape, dtype}, first_step, last_step});
        }
    }

    for (Index layer_index = first_trainable_layer_index; layer_index < last_trainable_layer_index; ++layer_index)
    {
        const auto& edges = consumer_edges[layer_index];

        const bool has_multiple_consumers = edges.size() > 1;
        const bool is_detached_detection_layer = layers[layer_index]->get_type() == LayerType::Detection
                                              && edges.empty();

        if (!has_multiple_consumers && !is_detached_detection_layer) continue;

        const Shape output_shape = layers[layer_index]->get_output_shape();
        if (output_shape.empty()) continue;

        const Shape delta_shape = Shape({batch_size}).append(output_shape);
        const Index last_step = last_trainable_layer_index - layer_index;
        const Index first_step = is_detached_detection_layer ? Index(0) : last_step;

        delta_entries.push_back({layer_index, 0, {delta_shape, compute_dtype}, first_step, last_step});
    }

    vector<vector<size_t>> entries_starting_at_backward_step(size_t(last_backward_step + 1));
    vector<vector<size_t>> entries_ending_at_backward_step(size_t(last_backward_step + 1));

    for (size_t entry_index = 0; entry_index < delta_entries.size(); ++entry_index)
    {
        entries_starting_at_backward_step[size_t(delta_entries[entry_index].first_step)].push_back(entry_index);
        entries_ending_at_backward_step[size_t(delta_entries[entry_index].last_step)].push_back(entry_index);
    }

    vector<pair<Index, Index>> free_blocks = {{0, numeric_limits<Index>::max()}};
    Index peak_bytes = 0;

    for (Index backward_step = 0; backward_step <= last_backward_step; ++backward_step)
    {
        for (size_t entry_index : entries_starting_at_backward_step[size_t(backward_step)])
        {
            const Index entry_bytes = get_aligned_bytes(delta_entries[entry_index].spec);
            Index byte_offset = Index(-1);

            auto it = ranges::find_if(free_blocks, [entry_bytes](const auto& block) { return block.second >= entry_bytes; });

            if (it != free_blocks.end())
            {
                byte_offset = it->first;

                if (it->second == entry_bytes)
                    free_blocks.erase(it);
                else
                {
                    it->first  += entry_bytes;
                    it->second -= entry_bytes;
                }
            }

            delta_entries[entry_index].byte_offset = byte_offset;
            peak_bytes = max(peak_bytes, byte_offset + entry_bytes);
        }

        for (size_t entry_index : entries_ending_at_backward_step[size_t(backward_step)])
        {
            const Index entry_bytes = get_aligned_bytes(delta_entries[entry_index].spec);

            auto it = ranges::lower_bound(free_blocks, delta_entries[entry_index].byte_offset, {}, &pair<Index, Index>::first);

            it = free_blocks.insert(it, {delta_entries[entry_index].byte_offset, entry_bytes});

            if (it + 1 != free_blocks.end() && it->first + it->second == (it + 1)->first)
            {
                it->second += (it + 1)->second;
                free_blocks.erase(it + 1);
            }

            if (it != free_blocks.begin() && (it - 1)->first + (it - 1)->second == it->first)
            {
                (it - 1)->second += it->second;
                free_blocks.erase(it);
            }
        }
    }

    layer_output_deltas.assign(size_t(layers_number), TensorView{});
    backward_slots.assign(size_t(layers_number), {});
    for (Index i = 0; i < layers_number; ++i)
        backward_slots[i].assign(backward_specs[i].size() + 1, TensorView{});

    delta_pool.resize_bytes(peak_bytes, neural_network->get_device());
    delta_pool.setZero();

    uint8_t* const base = delta_pool.as<uint8_t>();

    for (const auto& entry : delta_entries)
    {
        TensorView delta_view(base + entry.byte_offset,
                              entry.spec.shape,
                              entry.spec.dtype,
                              delta_pool.device_type);

        if (entry.slot == 0)
            layer_output_deltas[entry.layer] = delta_view;
        else
            backward_slots[entry.layer][entry.slot] = delta_view;
    }

    for (Index i = first_trainable_layer_index; i < last_trainable_layer_index; ++i)
    {
        const auto& edges = consumer_edges[i];
        if (edges.size() != 1) continue;

        const auto& [consumer_layer, input_position] = edges.front();
        const size_t slot = input_position + 1;
        const auto& consumer_deltas = backward_slots[consumer_layer];

        if (slot < consumer_deltas.size() && !consumer_deltas[slot].empty())
            layer_output_deltas[i] = consumer_deltas[slot];
    }
}

void BackPropagation::accumulate_output_deltas(size_t layer_index)
{
    const auto& edges = consumer_edges[layer_index];
    if (edges.size() <= 1) return;

    TensorView& destination = layer_output_deltas[layer_index];
    if (!destination.data) return;

    destination.setZero();

    for (const auto& [consumer_layer, input_position] : edges)
    {
        const TensorView& source = backward_slots[consumer_layer][1 + input_position];

        if (!source.data || source.size() != destination.size()) continue;

        add(destination, source, destination);
    }
}

TensorView& BackPropagation::get_output_delta()
{
    return layer_output_deltas[neural_network->get_last_trainable_layer_index()];
}

const TensorView& BackPropagation::get_output_delta() const
{
    return layer_output_deltas[neural_network->get_last_trainable_layer_index()];
}

void BackPropagation::print() const
{
    cout << "Back-propagation" << "\n"
         << "Error: " << error << "\n"
         << "Regularization: " << regularization << "\n"
         << "Loss:  " << loss << "\n";
}

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
