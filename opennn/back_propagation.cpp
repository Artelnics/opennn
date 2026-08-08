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

namespace
{

vector<bool> find_passthrough_layers(const vector<unique_ptr<Layer>>& layers,
                                     const vector<vector<TensorSpec>>& backward_specs,
                                     Index batch_size)
{
    vector<bool> passthrough(layers.size());
    for (size_t i = 0; i < layers.size(); ++i)
        passthrough[i] = backward_specs[i].empty()
            && layers[i]->get_forward_specs(batch_size).empty();
    return passthrough;
}

}

BackPropagation::BackPropagation(const Index new_batch_size, Loss* new_loss,
                                 ForwardPropagation* forward_propagation)
{
    set(new_batch_size, new_loss, forward_propagation);
}

vector<vector<pair<size_t, size_t>>> BackPropagation::make_consumer_edges(
    const NeuralNetwork& network)
{
    const auto& source_layers = network.get_source_layers();
    const size_t layers_number = network.get_layers().size();

    vector<vector<pair<size_t, size_t>>> edges(layers_number);
    for (size_t i = 0; i < layers_number; ++i)
    {
        const vector<Index>& sources = source_layers[i];
        for (size_t j = 0; j < sources.size(); ++j)
            if (const Index source_layer = sources[j]; source_layer >= 0
            && size_t(source_layer) < layers_number)
                edges[source_layer].push_back({i, j});
    }
    return edges;
}

void BackPropagation::set(const Index new_batch_size, Loss* new_loss,
                          ForwardPropagation* forward_propagation)
{
    batch_size = new_batch_size;
    loss = new_loss;

    throw_if(!loss, "loss is not set.");

    neural_network = loss->get_neural_network();

    throw_if(!neural_network, "neural network is not set.");

    throw_if(neural_network->get_training_type() == Type::INT8,
             "INT8 is inference-only; training requires FP32 or BF16.");

    error = 0.0f;
    accuracy = 0.0f;
    regularization = 0.0f;
    loss_value = 0.0f;

    const auto& layers = neural_network->get_layers();
    const size_t layers_number = layers.size();
    const auto parameter_specs = neural_network->get_parameter_specs();
    const auto backward_specs  = neural_network->get_backward_specs(batch_size);
    const auto& source_layers = neural_network->get_source_layers();

    consumer_edges = make_consumer_edges(*neural_network);

    const Index gradient_bytes = get_aligned_bytes(parameter_specs, Type::FP32);
    gradient.resize_bytes(gradient_bytes, neural_network->get_device());
    gradient.setZero();
    memory_debug::record("backward", "BackPropagation::gradient", gradient_bytes,
                         format("batch={}", batch_size));

    gradient_views.resize(layers_number);

    float* pointer = gradient.as<float>();
    for (size_t i = 0; i < layers_number; ++i)
        pointer = layers[i]->link_gradients(pointer, gradient_views[i], gradient.device_type);

    if (forward_propagation && forward_propagation->joint_delta_plan.valid)
    {
        const auto& joint = forward_propagation->joint_delta_plan;
        delta_pool.resize_bytes(0, neural_network->get_device());
        bind_delta_views(joint.layout, joint.offsets,
                         forward_propagation->data.as<uint8_t>(),
                         forward_propagation->data.device_type,
                         backward_specs);
        return;
    }

    setup_delta_pool(backward_specs);
}

BackPropagation::DeltaLayout BackPropagation::build_delta_entries(
    const NeuralNetwork& network, const Loss& loss_function, Index batch_size,
    const vector<vector<TensorSpec>>& backward_specs,
    const vector<vector<pair<size_t, size_t>>>& consumer_edges)
{
    const auto& layers = network.get_layers();
    const Index first_trainable_layer_index = network.get_first_trainable_layer_index();
    const Index last_trainable_layer_index = network.get_last_trainable_layer_index();
    const auto& source_layers = network.get_source_layers();
    const auto is_trainable_layer = [&](Index layer_index)
    {
        return layer_index >= first_trainable_layer_index
            && layer_index <= last_trainable_layer_index;
    };

    const Type compute_dtype = activation_dtype(network.get_training_type());

    DeltaLayout layout;
    vector<DeltaEntry>& delta_entries = layout.entries;
    vector<bool>& aliases_residual_delta = layout.aliases_residual_delta;
    aliases_residual_delta.assign(layers.size(), false);

    for (Index layer_index = first_trainable_layer_index;
         layer_index <= last_trainable_layer_index;
         ++layer_index)
    {
        const size_t index = size_t(layer_index);
        const auto& specs = backward_specs[index];
        const auto& sources = source_layers[index];

        if (!layers[index]->allows_input_delta_alias()
            || specs.size() != 2
            || sources.size() != 2)
            continue;

        if (!is_trainable_layer(sources[0])
            || !is_trainable_layer(sources[1])
            || sources[0] >= sources[1])
            continue;

        if (specs[0].shape.empty() || specs[1] != specs[0])
            continue;

        if (!layers[size_t(sources[1])]->preserves_output_delta_during_backward())
            continue;

        aliases_residual_delta[index] = true;
        layout.aliased_residual_delta_bytes += get_aligned_bytes(specs[1]);
    }

    const vector<bool> passthrough =
        find_passthrough_layers(layers, backward_specs, batch_size);
    const auto resolve_through_passthrough = [&](Index layer_index)
    {
        while (layer_index >= 0 && passthrough[size_t(layer_index)]
               && !source_layers[layer_index].empty() && source_layers[layer_index][0] >= 0)
            layer_index = source_layers[layer_index][0];
        return layer_index;
    };

    const Shape output_delta_shape = Shape({batch_size}).append(layers[last_trainable_layer_index]->get_output_shape());

    const Index loss_delta_consumer = resolve_through_passthrough(last_trainable_layer_index);

    if (output_delta_shape.size() > 0 && !loss_function.output_delta_overwrites_outputs())
        delta_entries.push_back({last_trainable_layer_index, 0, {output_delta_shape, compute_dtype}, 0,
                                 last_trainable_layer_index - loss_delta_consumer});

    for (Index layer_index = first_trainable_layer_index; layer_index <= last_trainable_layer_index; ++layer_index)
    {
        const size_t index = size_t(layer_index);
        const auto& specs = backward_specs[index];
        const auto& sources = source_layers[index];

        for (size_t slot = 0; slot < specs.size(); ++slot)
        {
            const auto& [shape, dtype] = specs[slot];
            if (shape.empty() || (slot == 1 && aliases_residual_delta[index]))
                continue;

            const Index first_step = last_trainable_layer_index - layer_index;
            Index last_step = first_step;

            if (slot < sources.size())
            {
                const Index source_layer = resolve_through_passthrough(sources[slot]);
                if (!is_trainable_layer(source_layer)) continue;
                last_step = last_trainable_layer_index - source_layer;
            }

            delta_entries.push_back({layer_index, slot + 1, {shape, dtype}, first_step, last_step});
        }
    }

    const pair<size_t, size_t> no_consumer_delta{SIZE_MAX, SIZE_MAX};
    vector<pair<size_t, size_t>>& reusable_consumer_deltas = layout.reusable_consumer_deltas;
    reusable_consumer_deltas.assign(layers.size(), no_consumer_delta);

    for (Index layer_index = first_trainable_layer_index; layer_index < last_trainable_layer_index; ++layer_index)
    {
        const size_t index = size_t(layer_index);
        const auto& edges = consumer_edges[index];
        const bool detached_detection =
            edges.empty()
            && is_one_of(layers[index]->get_type(),
                         LayerType::Detection, LayerType::DetectionV8);

        if (!detached_detection && edges.size() <= 1) continue;

        const Shape output_shape = layers[index]->get_output_shape();
        const Shape delta_shape = Shape({batch_size}).append(output_shape);

        if (!detached_detection)
        {
            const auto reusable_delta = ranges::find_if(edges, [&](const auto& edge)
            {
                const auto [consumer_layer, input_position] = edge;
                const auto& specs = backward_specs[consumer_layer];
                return input_position < specs.size()
                    && !specs[input_position].shape.empty()
                    && specs[input_position].shape == delta_shape;
            });

            if (reusable_delta != edges.end())
            {
                reusable_consumer_deltas[index] = *reusable_delta;
                continue;
            }
        }

        if (output_shape.empty()) continue;

        const Index last_step = last_trainable_layer_index - layer_index;
        const Index first_step = detached_detection ? Index(0) : last_step;

        delta_entries.push_back({layer_index, 0, {delta_shape, compute_dtype}, first_step, last_step});
    }

    return layout;
}

vector<MemoryPoolEntry> BackPropagation::to_pool_entries(const vector<DeltaEntry>& delta_entries,
                                                         Index step_offset)
{
    vector<MemoryPoolEntry> lifetime_entries;
    lifetime_entries.reserve(delta_entries.size());
    for (const DeltaEntry& entry : delta_entries)
        lifetime_entries.push_back({get_aligned_bytes(entry.spec),
                                    entry.first_step + step_offset,
                                    entry.last_step + step_offset});
    return lifetime_entries;
}

void BackPropagation::setup_delta_pool(const vector<vector<TensorSpec>>& backward_specs)
{
    const Index first_trainable_layer_index = neural_network->get_first_trainable_layer_index();
    const Index last_trainable_layer_index = neural_network->get_last_trainable_layer_index();
    const Index layers_number = neural_network->get_layers_number();

    const DeltaLayout layout = build_delta_entries(
        *neural_network, *loss, batch_size, backward_specs, consumer_edges);
    const vector<DeltaEntry>& delta_entries = layout.entries;

    const vector<MemoryPoolEntry> lifetime_entries = to_pool_entries(delta_entries);

    memory_debug::record_pool_lifetimes(
        "backward", lifetime_entries,
        format("first_trainable={},last_trainable={},layers={}",
               first_trainable_layer_index,
               last_trainable_layer_index,
               layers_number));

    const bool compact_pool_supported =
        neural_network->supports_compact_cnn_memory_layout();
    const MemoryPoolPlan pool_plan = plan_memory_pool(
        lifetime_entries,
        compact_pool_supported
            ? MemoryPoolStrategy::Compact
            : MemoryPoolStrategy::Chronological);
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

    bind_delta_views(layout, pool_plan.byte_offsets,
                     delta_pool.as<uint8_t>(), delta_pool.device_type,
                     backward_specs);
}

void BackPropagation::bind_delta_views(const DeltaLayout& layout,
                                       const vector<Index>& byte_offsets,
                                       uint8_t* base, Device device,
                                       const vector<vector<TensorSpec>>& backward_specs)
{
    const auto& layers = neural_network->get_layers();
    const Index layers_number = neural_network->get_layers_number();
    const Index first_trainable_layer_index = neural_network->get_first_trainable_layer_index();
    const Index last_trainable_layer_index = neural_network->get_last_trainable_layer_index();

    const vector<DeltaEntry>& delta_entries = layout.entries;
    const vector<bool>& aliases_residual_delta = layout.aliases_residual_delta;
    const Index aliased_residual_delta_bytes = layout.aliased_residual_delta_bytes;
    const vector<pair<size_t, size_t>>& reusable_consumer_deltas = layout.reusable_consumer_deltas;

    const vector<bool> passthrough =
        find_passthrough_layers(layers, backward_specs, batch_size);

    layer_output_deltas.assign(size_t(layers_number), TensorView{});
    backward_slots.assign(size_t(layers_number), {});
    for (Index i = 0; i < layers_number; ++i)
        backward_slots[i].assign(backward_specs[i].size() + 1, TensorView{});

    for (size_t i = 0; i < delta_entries.size(); ++i)
    {
        const DeltaEntry& entry = delta_entries[i];
        const TensorView delta_view(base + byte_offsets[i],
                              entry.spec.shape,
                              entry.spec.dtype,
                              device);

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
            if (consumer_layer == SIZE_MAX) continue;

            const size_t slot = input_position + 1;
            layer_output_deltas[i] =
                backward_slots[consumer_layer][slot];
            continue;
        }

        size_t consumer_layer = edges.front().first;
        size_t input_position = edges.front().second;
        while (passthrough[consumer_layer]
               && consumer_edges[consumer_layer].size() == 1)
        {
            input_position = consumer_edges[consumer_layer].front().second;
            consumer_layer = consumer_edges[consumer_layer].front().first;
        }

        const size_t slot = input_position + 1;
        const auto& consumer_deltas = backward_slots[consumer_layer];

        if (slot < consumer_deltas.size() && !consumer_deltas[slot].empty())
            layer_output_deltas[i] = consumer_deltas[slot];
        else if (passthrough[consumer_layer]
                 && !layer_output_deltas[consumer_layer].empty())
            layer_output_deltas[i] = layer_output_deltas[consumer_layer];
        else
            continue;

        layer_output_deltas[i].shape =
            Shape{batch_size}.append(layers[i]->get_output_shape());
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
