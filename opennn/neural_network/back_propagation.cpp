//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   B A C K   P R O P A G A T I O N   S O U R C E
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "opennn/neural_network/back_propagation.h"
#include "opennn/registry.h"
#include "opennn/core/memory_pool.h"
#include "opennn/training_strategy/loss.h"
#include "opennn/neural_network/neural_network.h"
#include "opennn/core/tensor_operations.h"
#include "opennn/core/memory_debug.h"

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

BackPropagation::BackPropagation(const Index new_batch_size,
                                 Loss& new_loss,
                                 Buffer* external_arena,
                                 span<const Index> arena_offsets)
{
    set(new_batch_size, new_loss, external_arena, arena_offsets);
}

NeuralNetwork* BackPropagation::get_neural_network() const
{
    return loss ? loss->get_neural_network() : nullptr;
}

const NeuralNetwork& BackPropagation::require_network() const
{
    throw_if(!loss, "BackPropagation: loss is not set.");

    const NeuralNetwork* const network = loss->get_neural_network();
    throw_if(!network, "BackPropagation: the loss has no neural network.");

    return *network;
}

vector<vector<pair<size_t, size_t>>> BackPropagation::make_consumer_edges() const
{
    const NeuralNetwork& network = require_network();

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

void BackPropagation::set(const Index new_batch_size, Loss& new_loss,
                          Buffer* external_arena,
                          span<const Index> arena_offsets)
{
    batch_size = new_batch_size;
    loss = &new_loss;

    const NeuralNetwork& neural_network = require_network();

    throw_if(neural_network.get_training_type() == Type::INT8,
             "INT8 is inference-only; training requires FP32 or BF16.");

    output_delta_layer_index = neural_network.get_last_trainable_layer_index();

    metrics.reset();

    setup_gradient();

    DeltaPlan plan = build_delta_plan();

    if (external_arena && !arena_offsets.empty())
    {
        arena.resize_bytes(0, neural_network.get_device());
        bind_deltas(plan.layout, arena_offsets,
                    external_arena->as<uint8_t>(),
                    external_arena->device_type,
                    plan.backward_specs);
        return;
    }

    setup_arena(plan.backward_specs, plan.layout);
}

void BackPropagation::setup_gradient()
{
    const NeuralNetwork& neural_network = require_network();

    const auto& layers = neural_network.get_layers();
    const auto parameter_specs = neural_network.get_parameter_specs();

    const Index gradient_bytes = get_aligned_bytes(parameter_specs, Type::FP32);
    gradient.resize_bytes(gradient_bytes, neural_network.get_device());
    gradient.setZero();
    memory_debug::record("backward", "BackPropagation::gradient", gradient_bytes,
                         format("batch={}", batch_size));

    float* pointer = gradient.as<float>();
    for (size_t i = 0; i < layers.size(); ++i)
        pointer = layers[i]->link_gradients(pointer, gradient.device_type);
}

BackPropagation::DeltaLayout BackPropagation::build_delta_layout(
    const vector<vector<TensorSpec>>& backward_specs) const
{
    const NeuralNetwork& network = require_network();
    const auto& layers = network.get_layers();
    const auto& source_layers = network.get_source_layers();

    const Index first_layer = network.get_first_trainable_layer_index();
    const Index last_layer = network.get_last_trainable_layer_index();
    const Type compute_dtype = activation_dtype(network.get_training_type());

    const auto is_trainable = [&](Index layer)
    {
        return layer >= first_layer && layer <= last_layer;
    };

    DeltaLayout layout;
    layout.passthrough_layers =
        find_passthrough_layers(layers, backward_specs, batch_size);

    layout.aliases_residual_delta.assign(layers.size(), false);
    layout.reusable_consumer_deltas.assign(
        layers.size(), {SIZE_MAX, SIZE_MAX});

    const auto resolve_source = [&](Index layer)
    {
        while (layer >= 0
               && layout.passthrough_layers[size_t(layer)]
               && !source_layers[layer].empty()
               && source_layers[layer][0] >= 0)
        {
            layer = source_layers[layer][0];
        }

        return layer;
    };

    const auto make_delta_shape = [&](Index layer)
    {
        return Shape{batch_size}.append(layers[layer]->get_output_shape());
    };

    // Residual delta aliases
    for (Index layer = first_layer; layer <= last_layer; ++layer)
    {
        const size_t i = size_t(layer);
        const auto& specs = backward_specs[i];
        const auto& sources = source_layers[i];

        if (!layers[i]->allows_input_delta_alias()
            || specs.size() != 2
            || sources.size() != 2
            || !is_trainable(sources[0])
            || !is_trainable(sources[1])
            || sources[0] >= sources[1]
            || specs[0].shape.empty()
            || specs[1] != specs[0]
            || !layers[size_t(sources[1])]->preserves_output_delta_during_backward())
        {
            continue;
        }

        layout.aliases_residual_delta[i] = true;
        layout.aliased_residual_delta_bytes += get_aligned_bytes(specs[1]);
    }

    // Loss output delta
    const Shape output_delta_shape = make_delta_shape(last_layer);
    const Index loss_consumer = resolve_source(last_layer);

    if (output_delta_shape.size() > 0 && !loss->output_delta_overwrites_outputs())
    {
        layout.entries.push_back({
            last_layer,
            0,
            {output_delta_shape, compute_dtype},
            0,
            last_layer - loss_consumer
        });
    }

    // Layer input deltas
    for (Index layer = first_layer; layer <= last_layer; ++layer)
    {
        const size_t i = size_t(layer);
        const auto& specs = backward_specs[i];
        const auto& sources = source_layers[i];

        for (size_t slot = 0; slot < specs.size(); ++slot)
        {
            const auto& spec = specs[slot];

            if (spec.shape.empty()
                || (slot == 1 && layout.aliases_residual_delta[i]))
            {
                continue;
            }

            const Index first_step = last_layer - layer;
            Index last_step = first_step;

            if (slot < sources.size())
            {
                const Index source = resolve_source(sources[slot]);

                if (!is_trainable(source))
                    continue;

                last_step = last_layer - source;
            }

            layout.entries.push_back({
                layer,
                slot + 1,
                spec,
                first_step,
                last_step
            });
        }
    }

    // Reuse consumer deltas where possible
    for (Index layer = first_layer; layer < last_layer; ++layer)
    {
        const size_t i = size_t(layer);
        const auto& edges = consumer_edges[i];

        const bool detached_detection =
            edges.empty()
            && is_one_of(layers[i]->get_type(),
                         LayerType::Detection,
                         LayerType::DetectionV8);

        if (!detached_detection && edges.size() <= 1)
            continue;

        const Shape output_shape = layers[i]->get_output_shape();
        const Shape delta_shape = Shape{batch_size}.append(output_shape);

        if (!detached_detection)
        {
            const auto reusable = ranges::find_if(
                edges,
                [&](const auto& edge)
                {
                    const auto [consumer, input] = edge;
                    const auto& specs = backward_specs[consumer];

                    return input < specs.size()
                        && !specs[input].shape.empty()
                        && specs[input].shape == delta_shape;
                });

            if (reusable != edges.end())
            {
                layout.reusable_consumer_deltas[i] = *reusable;
                continue;
            }
        }

        if (output_shape.empty())
            continue;

        const Index last_step = last_layer - layer;

        layout.entries.push_back({
            layer,
            0,
            {delta_shape, compute_dtype},
            detached_detection ? Index{0} : last_step,
            last_step
        });
    }

    return layout;
}

vector<MemoryPoolEntry> BackPropagation::to_pool_entries(const vector<DeltaEntry>& delta_entries,
                                                         Index step_offset)
{
    vector<MemoryPoolEntry> lifetime_entries;
    lifetime_entries.reserve(delta_entries.size());

    ranges::transform(delta_entries, back_inserter(lifetime_entries),
                      [step_offset](const DeltaEntry& entry)
                      {
                          return MemoryPoolEntry{get_aligned_bytes(entry.spec),
                                                 entry.first_step + step_offset,
                                                 entry.last_step + step_offset};
                      });
    return lifetime_entries;
}

vector<MemoryPoolEntry> BackPropagation::make_co_planned_lifetimes(
    Loss& new_loss, const Index new_batch_size)
{
    BackPropagation planner;
    planner.loss = &new_loss;
    planner.batch_size = new_batch_size;

    const NeuralNetwork& neural_network = planner.require_network();
    const DeltaPlan plan = planner.build_delta_plan();

    const Index backward_base = Index(2 * neural_network.get_layers_number() - 1);
    const Index step_offset =
        backward_base - neural_network.get_last_trainable_layer_index();

    return to_pool_entries(plan.layout.entries, step_offset);
}

BackPropagation::DeltaPlan BackPropagation::build_delta_plan()
{
    const NeuralNetwork& neural_network = require_network();

    consumer_edges = make_consumer_edges();

    DeltaPlan plan;
    plan.backward_specs = neural_network.get_backward_specs(batch_size);
    plan.layout = build_delta_layout(plan.backward_specs);

    return plan;
}

void BackPropagation::setup_arena(const vector<vector<TensorSpec>>& backward_specs,
                                  const DeltaLayout& layout)
{
    const NeuralNetwork& neural_network = require_network();

    const Index first_trainable_layer_index = neural_network.get_first_trainable_layer_index();
    const Index last_trainable_layer_index = neural_network.get_last_trainable_layer_index();
    const Index layers_number = neural_network.get_layers_number();

    const vector<DeltaEntry>& delta_entries = layout.entries;

    const vector<MemoryPoolEntry> lifetime_entries = to_pool_entries(delta_entries);

    memory_debug::record_pool_lifetimes(
        "backward", lifetime_entries,
        format("first_trainable={},last_trainable={},layers={}",
               first_trainable_layer_index,
               last_trainable_layer_index,
               layers_number));

    const bool compact_pool_supported =
        neural_network.supports_compact_cnn_memory_layout();
    const MemoryPoolPlan pool_plan = plan_memory_pool(
        lifetime_entries,
        compact_pool_supported
            ? MemoryPoolStrategy::Compact
            : MemoryPoolStrategy::Chronological);
    arena.resize_bytes(pool_plan.peak_bytes, neural_network.get_device());
    arena.setZero();
    memory_debug::record("backward", "BackPropagation::arena", pool_plan.peak_bytes,
                         format("batch={},planner={}",
                                batch_size,
                                compact_pool_supported ? "compact" : "chronological"));
    memory_debug::record("backward.arena_analysis", "live_bytes_lower_bound",
                         pool_plan.lower_bound_live_bytes,
                         format("batch={},entries={}", batch_size, delta_entries.size()));
    memory_debug::record("backward.arena_analysis", "allocator_fragmentation_overhead",
                         pool_plan.fragmentation_bytes(),
                         format("batch={},entries={}", batch_size, delta_entries.size()));

    bind_deltas(layout, pool_plan.byte_offsets,
                arena.as<uint8_t>(), arena.device_type,
                backward_specs);
}

void BackPropagation::bind_deltas(const DeltaLayout& layout,
                                  span<const Index> byte_offsets,
                                  uint8_t* base, Device device,
                                  const vector<vector<TensorSpec>>& backward_specs)
{
    const NeuralNetwork& neural_network = require_network();

    const auto& layers = neural_network.get_layers();
    const Index layers_number = neural_network.get_layers_number();
    const Index first_layer = neural_network.get_first_trainable_layer_index();
    const Index last_layer = neural_network.get_last_trainable_layer_index();

    output_deltas.assign(size_t(layers_number), TensorView{});
    slots.assign(size_t(layers_number), {});

    for (Index i = 0; i < layers_number; ++i)
        slots[i].resize(backward_specs[i].size() + 1);

    for (size_t i = 0; i < layout.entries.size(); ++i)
    {
        const DeltaEntry& entry = layout.entries[i];

        TensorView delta(base + byte_offsets[i],
                         entry.spec.shape,
                         entry.spec.dtype,
                         device);

        if (entry.slot == 0)
            output_deltas[entry.layer] = delta;
        else
            slots[entry.layer][entry.slot] = delta;
    }

    for (Index i = first_layer; i <= last_layer; ++i)
        if (layout.aliases_residual_delta[size_t(i)])
            slots[i][2] = slots[i][1];

    if (layout.aliased_residual_delta_bytes > 0)
    {
        memory_debug::record(
            "backward.delta_alias",
            "residual_input_delta_bytes",
            layout.aliased_residual_delta_bytes,
            format("batch={}", batch_size));
    }

    for (Index layer_index = first_layer; layer_index < last_layer; ++layer_index)
    {
        const size_t index = size_t(layer_index);
        const auto& edges = consumer_edges[index];

        if (edges.empty())
            continue;

        if (edges.size() > 1)
        {
            const auto [consumer_layer, input_position] =
                layout.reusable_consumer_deltas[index];

            if (consumer_layer != SIZE_MAX)
                output_deltas[index] = slots[consumer_layer][input_position + 1];

            continue;
        }

        auto [consumer_layer, input_position] = edges.front();

        while (layout.passthrough_layers[consumer_layer]
               && consumer_edges[consumer_layer].size() == 1)
            tie(consumer_layer, input_position) =
                consumer_edges[consumer_layer].front();

        const size_t slot = input_position + 1;

        if (slot < slots[consumer_layer].size()
            && !slots[consumer_layer][slot].empty())
            output_deltas[index] = slots[consumer_layer][slot];
        else if (layout.passthrough_layers[consumer_layer]
                 && !output_deltas[consumer_layer].empty())
            output_deltas[index] = output_deltas[consumer_layer];
        else
            continue;

        output_deltas[index].shape =
            Shape{batch_size}.append(layers[index]->get_output_shape());
    }
}

void BackPropagation::accumulate_output_deltas(size_t layer_index)
{
    const auto& edges = consumer_edges[layer_index];
    if (edges.size() <= 1) return;

    TensorView& destination = output_deltas[layer_index];
    if (!destination.data) return;

    const auto source = [&](const auto& edge) -> const TensorView&
    {
        return slots[edge.first][1 + edge.second];
    };

    const auto valid = [&](const auto& edge)
    {
        const TensorView& s = source(edge);
        return s.data && s.size() == destination.size();
    };

    // Prefer destination itself as the initial accumulated value.
    auto first = std::ranges::find_if(edges, [&](const auto& edge)
    {
        const TensorView& s = source(edge);
        return valid(edge) && s.data == destination.data;
    });

    if (first == edges.end())
        first = std::ranges::find_if(edges, valid);

    if (first == edges.end())
    {
        destination.setZero();
        return;
    }

    const TensorView& first_source = source(*first);

    if (first_source.data != destination.data)
        copy(first_source, destination);

    for (auto it = edges.begin(); it != edges.end(); ++it)
    {
        const TensorView& s = source(*it);

        if (it == first || !valid(*it) || s.data == destination.data)
            continue;

        if (destination.is_cuda())
            add(destination, s, destination);
        else
            destination.as_vector() += s.as_vector();
    }
}

TensorView& BackPropagation::get_output_delta()
{
    throw_if(output_deltas.empty(),
             "BackPropagation::get_output_delta: deltas are not bound.");
    return output_deltas[output_delta_layer_index];
}

const TensorView& BackPropagation::get_output_delta() const
{
    throw_if(output_deltas.empty(),
             "BackPropagation::get_output_delta: deltas are not bound.");
    return output_deltas[output_delta_layer_index];
}

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
