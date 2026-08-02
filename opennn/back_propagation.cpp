//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   B A C K   P R O P A G A T I O N   S O U R C E
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "back_propagation.h"
#include "memory_pool.h"
#include "loss.h"
#include "neural_network.h"
#include "forward_propagation.h"
#include "tensor_operations.h"
#include "memory_debug.h"

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

    throw_if(neural_network->get_training_type() == Type::INT8,
             "INT8 is inference-only; training requires FP32 or BF16.");

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
    memory_debug::record("backward", "BackPropagation::gradient", gradient_bytes,
                         format("batch={}", batch_size));

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
    vector<bool> aliases_residual_delta(size_t(layers_number), false);
    Index aliased_residual_delta_bytes = 0;

    for (Index layer_index = first_trainable_layer_index;
         layer_index <= last_trainable_layer_index;
         ++layer_index)
    {
        const auto& specs = backward_specs[size_t(layer_index)];
        const auto& sources = source_layers[size_t(layer_index)];
        const bool aliases = layers[size_t(layer_index)]->allows_input_delta_alias()
            && specs.size() == 2 && sources.size() == 2
            && sources[0] >= first_trainable_layer_index
            && sources[0] < sources[1]
            && sources[1] <= last_trainable_layer_index
            && !specs[0].shape.empty() && specs[1].shape == specs[0].shape
            && specs[1].dtype == specs[0].dtype
            && layers[size_t(sources[1])]->preserves_output_delta_during_backward();
        aliases_residual_delta[size_t(layer_index)] = aliases;
        if (aliases) aliased_residual_delta_bytes += get_aligned_bytes(specs[1]);
    }

    const auto is_passthrough = [&](Index layer_index)
    {
        return backward_specs[size_t(layer_index)].empty()
            && layers[layer_index]->get_forward_specs(batch_size).empty();
    };

    const Shape output_delta_shape = Shape({batch_size}).append(layers[last_trainable_layer_index]->get_output_shape());

    Index loss_delta_consumer = last_trainable_layer_index;
    while (loss_delta_consumer >= 0 && is_passthrough(loss_delta_consumer)
           && !source_layers[loss_delta_consumer].empty() && source_layers[loss_delta_consumer][0] >= 0)
        loss_delta_consumer = source_layers[loss_delta_consumer][0];

    if (output_delta_shape.size() != 0 && !loss_pointer->output_delta_overwrites_outputs())
        delta_entries.push_back({last_trainable_layer_index, 0, {output_delta_shape, compute_dtype}, 0,
                                 last_trainable_layer_index - loss_delta_consumer});

    for (Index layer_index = first_trainable_layer_index; layer_index <= last_trainable_layer_index; ++layer_index)
    {
        const auto& specs = backward_specs[layer_index];
        const auto& sources = source_layers[layer_index];

        for (size_t j = 0; j < specs.size(); ++j)
        {
            const auto& [shape, dtype] = specs[j];
            if (shape.empty()) continue;
            if (j == 1 && aliases_residual_delta[size_t(layer_index)])
                continue;

            const Index first_step = last_trainable_layer_index - layer_index;

            Index source_layer = (j < sources.size()) ? sources[j] : Index(-1);
            while (source_layer >= 0 && is_passthrough(source_layer)
                   && !source_layers[source_layer].empty() && source_layers[source_layer][0] >= 0)
                source_layer = source_layers[source_layer][0];

            const bool source_layer_is_trainable = source_layer >= first_trainable_layer_index
                                                && source_layer <= last_trainable_layer_index;

            const bool is_input_delta = j < sources.size();
            if (is_input_delta && !source_layer_is_trainable) continue;

            const Index last_step = source_layer_is_trainable ? last_trainable_layer_index - source_layer : first_step;

            delta_entries.push_back({layer_index, j + 1, {shape, dtype}, first_step, last_step});
        }
    }

    const pair<size_t, size_t> no_consumer_delta{SIZE_MAX, SIZE_MAX};
    vector<pair<size_t, size_t>> reusable_consumer_deltas(
        size_t(layers_number),
        no_consumer_delta);

    for (Index layer_index = first_trainable_layer_index; layer_index < last_trainable_layer_index; ++layer_index)
    {
        const auto& edges = consumer_edges[layer_index];

        const bool has_multiple_consumers = edges.size() > 1;
        const Shape output_shape = layers[layer_index]->get_output_shape();
        const Shape delta_shape = Shape({batch_size}).append(output_shape);
        const auto reusable_delta = has_multiple_consumers
            ? ranges::find_if(
                  edges,
                  [&](const auto& edge)
                  {
                      const auto [consumer_layer, input_position] = edge;
                      const auto& specs = backward_specs[consumer_layer];
                      return input_position < specs.size()
                          && !specs[input_position].shape.empty()
                          && specs[input_position].shape == delta_shape;
                  })
            : edges.end();
        if (reusable_delta != edges.end())
            reusable_consumer_deltas[size_t(layer_index)] = *reusable_delta;

        const auto layer_type = layers[layer_index]->get_type();
        const bool is_detached_detection_layer =
            (layer_type == LayerType::Detection || layer_type == LayerType::DetectionV8)
            && edges.empty();

        if ((!has_multiple_consumers || reusable_delta != edges.end())
            && !is_detached_detection_layer)
            continue;
        if (output_shape.empty()) continue;

        const Index last_step = last_trainable_layer_index - layer_index;
        const Index first_step = is_detached_detection_layer ? Index(0) : last_step;

        delta_entries.push_back({layer_index, 0, {delta_shape, compute_dtype}, first_step, last_step});
    }

    vector<MemoryPoolEntry> lifetime_entries;
    lifetime_entries.reserve(delta_entries.size());
    for (const DeltaEntry& entry : delta_entries)
        lifetime_entries.push_back({get_aligned_bytes(entry.spec),
                                    entry.first_step,
                                    entry.last_step});

    const bool compact_pool_supported =
        neural_network->supports_compact_cnn_memory_layout();
    const MemoryPoolPlan pool_plan = plan_memory_pool(
        lifetime_entries,
        compact_pool_supported
            ? MemoryPoolStrategy::Compact
            : MemoryPoolStrategy::Chronological);
    layer_output_deltas.assign(size_t(layers_number), TensorView{});
    backward_slots.assign(size_t(layers_number), {});
    for (Index i = 0; i < layers_number; ++i)
        backward_slots[i].assign(backward_specs[i].size() + 1, TensorView{});

    delta_pool.resize_bytes(pool_plan.peak_bytes, neural_network->get_device());
    delta_pool.setZero();
    memory_debug::record("backward", "BackPropagation::delta_pool", pool_plan.peak_bytes,
                         format("batch={},planner={}",
                                batch_size,
                                compact_pool_supported ? "compact" : "chronological"));
    memory_debug::record("backward.delta_pool_analysis", "live_bytes_lower_bound",
                         pool_plan.lower_bound_live_bytes,
                         format("batch={},entries={}", batch_size, delta_entries.size()));
    memory_debug::record("backward.delta_pool_analysis", "allocator_fragmentation_overhead",
                         pool_plan.fragmentation_bytes(),
                         format("batch={},entries={}", batch_size, delta_entries.size()));

    uint8_t* const base = delta_pool.as<uint8_t>();

    for (size_t i = 0; i < delta_entries.size(); ++i)
    {
        const DeltaEntry& entry = delta_entries[i];
        const TensorView delta_view(base + pool_plan.byte_offsets[i],
                              entry.spec.shape,
                              entry.spec.dtype,
                              delta_pool.device_type);

        if (entry.slot == 0)
            layer_output_deltas[entry.layer] = delta_view;
        else
            backward_slots[entry.layer][entry.slot] = delta_view;
    }

    for (Index layer_index = first_trainable_layer_index;
         layer_index <= last_trainable_layer_index;
         ++layer_index)
        if (aliases_residual_delta[size_t(layer_index)])
            backward_slots[size_t(layer_index)][2] =
                backward_slots[size_t(layer_index)][1];

    if (aliased_residual_delta_bytes > 0)
        memory_debug::record(
            "backward.delta_alias",
            "residual_input_delta_bytes",
            aliased_residual_delta_bytes,
            format("batch={}", batch_size));

    for (Index i = first_trainable_layer_index; i < last_trainable_layer_index; ++i)
    {
        const auto& edges = consumer_edges[i];
        if (edges.empty()) continue;

        if (edges.size() > 1)
        {
            const auto [consumer_layer, input_position] =
                reusable_consumer_deltas[size_t(i)];
            if (consumer_layer != SIZE_MAX)
            {
                const size_t slot = input_position + 1;
                layer_output_deltas[i] =
                    backward_slots[consumer_layer][slot];
            }

            continue;
        }

        size_t consumer_layer = edges.front().first;
        size_t input_position = edges.front().second;
        while (is_passthrough(Index(consumer_layer))
               && consumer_edges[consumer_layer].size() == 1)
        {
            input_position = consumer_edges[consumer_layer].front().second;
            consumer_layer = consumer_edges[consumer_layer].front().first;
        }

        const size_t slot = input_position + 1;
        const auto& consumer_deltas = backward_slots[consumer_layer];

        TensorView delta_view;
        if (slot < consumer_deltas.size() && !consumer_deltas[slot].empty())
            delta_view = consumer_deltas[slot];
        else if (is_passthrough(Index(consumer_layer))
                 && !layer_output_deltas[consumer_layer].empty())
            delta_view = layer_output_deltas[consumer_layer];
        else
            continue;

        delta_view.shape = Shape{batch_size}.append(layers[i]->get_output_shape());
        layer_output_deltas[i] = delta_view;
    }
}

void BackPropagation::accumulate_output_deltas(size_t layer_index)
{
    const auto& edges = consumer_edges[layer_index];
    if (edges.size() <= 1) return;

    TensorView& destination = layer_output_deltas[layer_index];
    if (!destination.data) return;

    const TensorView* first_source = nullptr;
    bool destination_is_source = false;
    for (const auto& [consumer_layer, input_position] : edges)
    {
        const TensorView& source = backward_slots[consumer_layer][1 + input_position];
        if (source.data && source.size() == destination.size())
        {
            if (!first_source) first_source = &source;
            destination_is_source |= source.data == destination.data;
        }
    }

    if (!first_source) { destination.setZero(); return; }
    if (!destination_is_source) copy(*first_source, destination);
    for (const auto& [consumer_layer, input_position] : edges)
    {
        const TensorView& source = backward_slots[consumer_layer][1 + input_position];
        if (!source.data || source.size() != destination.size()
            || source.data == destination.data
            || (!destination_is_source && &source == first_source))
            continue;

        if (destination.is_cuda())
            add(destination, source, destination);
        else
            destination.as_vector() += source.as_vector();
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

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
