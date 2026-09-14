// SPDX-License-Identifier: LGPL-2.1-or-later
// Copyright (C) 2005-2026 Artificial Intelligence, SL.

#include "opennn/network/forward_propagation.h"
#include "opennn/network/training_arena_plan.h"
#include "opennn/registry.h"
#include "opennn/network/network.h"
#include "opennn/core/memory_debug.h"
#include "opennn/core/device_backend.h"
#include "opennn/core/memory_pool.h"
#include "opennn/core/profiler.h"
#include "opennn/core/string_utilities.h"

#ifndef OPENNN_NO_VISION
#include "opennn/network/layers/grouped_query_attention_layer.h"
#endif

namespace opennn
{

bool ForwardPropagation::reserve_kv_cache(const Index required, const Index preserved_tokens)
{
#ifdef OPENNN_NO_VISION
    (void)required;
    (void)preserved_tokens;
    return false;
#else
    if (!network || mode != ForwardPropagationMode::Inference
        || batch_size != 1 || !network->is_gpu()
        || network->get_training_type() != Type::BF16) return false;

    const Index limit = network->get_input_shape()[0];
    throw_if(required < 1 || required > limit || preserved_tokens < 0
             || preserved_tokens > required,
             "ForwardPropagation::reserve_kv_cache: invalid capacity/prefix {}/{} (limit {}).",
             required, preserved_tokens, limit);
    Index capacity = min(Index(256), limit);
    while (capacity < required) capacity = min(capacity * 2, limit);

    struct Replacement { size_t layer; Buffer storage{Device::CUDA}; };
    vector<Replacement> replacements;
    const auto& layers = network->get_layers();
    const DeviceStream stream = device::get_compute_stream();
    try
    {
        for (size_t i = 0; i < layers.size(); ++i)
        {
            const auto* attention = dynamic_cast<const GroupedQueryAttention*>(layers[i].get());
            if (!attention || !attention->uses_compact_inference()
                || attention->get_compute_dtype() != Type::BF16) continue;

            const Index row_bytes = attention->get_kv_heads() * attention->get_head_dim()
                * Index(sizeof(uint16_t));
            Buffer& old = (*layer_session_state_storage)[i];
            const Index old_half = old.byte_size() / 2;
            throw_if(old.byte_size() % (2 * row_bytes) != 0
                     || preserved_tokens * row_bytes > old_half,
                     "ForwardPropagation::reserve_kv_cache: invalid preserved prefix at layer {}.", i);
            if (old_half >= capacity * row_bytes) continue;

            replacements.push_back({i, Buffer{Device::CUDA}});
            Buffer& next = replacements.back().storage;
            next.resize_bytes(2 * capacity * row_bytes, Device::CUDA);
            device::set_zero_async(next.data(), next.byte_size(), stream);
            if (preserved_tokens > 0)
            {
                device::copy_async(next.data(), old.data(), preserved_tokens * row_bytes,
                                   device::CopyKind::DeviceToDevice, stream);
                device::copy_async(next.as<char>() + capacity * row_bytes,
                                   old.as<char>() + old_half, preserved_tokens * row_bytes,
                                   device::CopyKind::DeviceToDevice, stream);
            }
        }
        if (replacements.empty()) return false;
        device::synchronize(stream);
    }
    catch (const exception& error)
    {
        device::synchronize(stream);
        const device::CudaBlockCacheBypass release_failed_growth;
        replacements.clear();
        throw runtime_error(format("Qwen KV growth to {} tokens failed; previous cache retained: {}",
                                   capacity, error.what()));
    }
    reset_cuda_graph();
    for (auto& replacement : replacements)
        (*layer_session_state_storage)[replacement.layer].swap(replacement.storage);
    // Superseded buckets cannot be reused while this session keeps growing.
    // Return them instead of retaining another KV cache in the block pool.
    const device::CudaBlockCacheBypass release_old_buckets;
    replacements.clear();
    return true;
#endif
}

void ForwardPropagation::release_inference_storage()
{
    throw_if(mode != ForwardPropagationMode::Inference,
             "ForwardPropagation: cannot trim training storage.");
    reset_cuda_graph();
    inputs.clear();
    slots.clear();
    capacity_inputs.clear();
    capacity_slots.clear();
    staged_inputs.clear();
    staged_input_storage.clear();
    layer_state_storage.clear();
    layer_session_state_storage.reset();
    layer_pinned_storage.clear();
    device_valid_length_storage.clear();
    output_window.reset();
    for (Buffer& buffer : inference_graph_workspaces)
        buffer.resize_bytes(0, Device::CUDA);
    position_device.resize_bytes(0, Device::CUDA);
    position_pinned = {};
    loss_workspace.resize_bytes(0, Device::CUDA);
    loss_target_workspace.resize_bytes(0, Device::CUDA);
    arena.resize_bytes(0, arena.get_device());
    vector<vector<uint16_t>>{}.swap(host_bf16_input_scratch);
    vector<uint16_t>{}.swap(host_bf16_output_scratch);
    past_length = 0;
}

static Index resolve_producer(const vector<vector<TensorSpec>>& forward_specs,
                              const vector<vector<Index>>& source_layers,
                              Index source_layer)
{
    Index resolved = source_layer;
    while (resolved >= 0 && forward_specs[size_t(resolved)].empty())
    {
        const auto& upstream = source_layers[size_t(resolved)];
        if (upstream.empty())
        {
            resolved = -1;
            break;
        }
        resolved = upstream.front();
    }
    return resolved;
}

static vector<Index> find_early_output_release_steps(
    const vector<unique_ptr<Layer>>& layers,
    const vector<vector<pair<size_t, size_t>>>& consumers,
    const vector<vector<TensorSpec>>& forward_specs,
    Index& released_bytes)
{
    vector<Index> release_steps(layers.size(), Index(-1));
    released_bytes = 0;

    for (size_t producer_index = 0; producer_index < layers.size(); ++producer_index)
    {
        const auto& output_consumers = consumers[producer_index];
        if (output_consumers.empty()) continue;

        Index release_step = -1;
        if (!layers[producer_index]->backward_uses_forward_output())
        {
            const bool releasable = ranges::all_of(
                output_consumers,
                [&](const auto& edge)
                {
                    const auto [consumer_index, input_position] = edge;
                    return !layers[consumer_index]->backward_uses_input(
                        input_position);
                });

            if (releasable)
                for (const auto& edge : output_consumers)
                    release_step = max(release_step, Index(edge.first));
        }

        if (release_step < 0) continue;

        release_steps[producer_index] = release_step;
        if (!forward_specs[producer_index].empty())
            released_bytes += get_aligned_bytes(forward_specs[producer_index].back());
    }

    return release_steps;
}

struct ForwardCapacities
{
    Index sequence = 0;
    Index output = 0;
    Index output_layer = -1;
};

static ForwardCapacities apply_inference_shape_policy(
    vector<vector<TensorSpec>>& specs,
    const Shape& input_shape,
    const InferenceShapePolicy& policy)
{
    const Index model_sequence = input_shape.empty() ? 0 : input_shape[0];
    ForwardCapacities result;
    result.sequence = policy.sequence_capacity > 0 ? policy.sequence_capacity : model_sequence;

    throw_if(policy.sequence_capacity > model_sequence,
             "ForwardPropagation::set: sequence capacity {} exceeds the network capacity {}.",
             policy.sequence_capacity, model_sequence);

    if(policy.sequence_capacity > 0)
        for(auto& layer_specs : specs)
            for(TensorSpec& spec : layer_specs)
                if(spec.shape.get_rank() >= 2 && spec.shape[1] == model_sequence)
                    spec.shape.set_dimension(1, result.sequence);

    for(const size_t i : views::iota(size_t(0), specs.size()) | views::reverse)
        if(!specs[i].empty())
        {
            result.output_layer = Index(i);
            break;
        }

    result.output = policy.final_output_capacity > 0
                  ? policy.final_output_capacity
                  : result.sequence;

    throw_if(policy.final_output_capacity > 0 && policy.sequence_capacity <= 0,
             "ForwardPropagation::set: final_output_capacity requires an explicit sequence_capacity.");
    throw_if(result.output > result.sequence,
             "ForwardPropagation::set: final output capacity {} exceeds sequence capacity {}.",
             result.output, result.sequence);

    if(policy.final_output_capacity > 0 && result.output_layer >= 0)
    {
        TensorSpec& output = specs[size_t(result.output_layer)].back();
        throw_if(output.shape.get_rank() < 2 || output.shape[1] != result.sequence,
                 "ForwardPropagation::set: final output does not expose a sequence dimension compatible with compact inference.");
        output.shape.set_dimension(1, result.output);
    }

    return result;
}

static void elide_inference_slots(const vector<unique_ptr<Layer>>& layers,
                                  vector<vector<TensorSpec>>& specs,
                                  Device device,
                                  Index batch_size)
{
    for(size_t i = 0; i < layers.size(); ++i)
    {
#ifndef OPENNN_NO_VISION
        const auto* attention = dynamic_cast<const GroupedQueryAttention*>(layers[i].get());
#endif
        for(size_t j = 0; j < specs[i].size(); ++j)
        {
            bool elidable = layers[i]->get_forward_slot_kind(j + 1) == ForwardSlotKind::TrainingOnly
                         || layers[i]->is_forward_slot_inference_elidable(j + 1, device);
#ifndef OPENNN_NO_VISION
            elidable = elidable
                    || (attention && attention->is_forward_slot_inference_elidable(j + 1, device, batch_size));
#endif
            if(elidable) specs[i][j] = {};
        }
    }
}

static vector<Index> find_inference_output_release_steps(
    const vector<vector<TensorSpec>>& specs,
    const vector<vector<Index>>& sources,
    span<const Index> retained_outputs,
    Index last_trainable_layer)
{
    const size_t layers_number = specs.size();
    const Index final_step = layers_number == 0 ? 0 : Index(layers_number - 1);
    vector<Index> last_consumers(layers_number);
    vector<bool> has_consumers(layers_number, false);
    iota(last_consumers.begin(), last_consumers.end(), Index(0));

    for(size_t consumer = 0; consumer < layers_number; ++consumer)
        for(const Index source : sources[consumer])
        {
            const Index producer = resolve_producer(specs, sources, source);
            if(producer < 0) continue;
            has_consumers[size_t(producer)] = true;
            last_consumers[size_t(producer)] = max(last_consumers[size_t(producer)], Index(consumer));
        }

    vector<bool> observable(layers_number, false);
    for(size_t i = 0; i < layers_number; ++i)
        observable[i] = !has_consumers[i];

    const auto retain = [&](Index layer)
    {
        if(layer < 0 || size_t(layer) >= layers_number) return;
        const Index producer = resolve_producer(specs, sources, layer);
        if(producer >= 0) observable[size_t(producer)] = true;
    };

    retain(Index(layers_number) - 1);
    retain(last_trainable_layer);
    for(const Index retained : retained_outputs)
    {
        throw_if(retained < 0 || size_t(retained) >= layers_number,
                 "ForwardPropagation::set: retained output layer {} is out of range (network has {} layers).",
                 retained, layers_number);
        retain(retained);
    }

    for(size_t i = 0; i < layers_number; ++i)
        last_consumers[i] = observable[i] ? final_step : last_consumers[i];

    return last_consumers;
}

struct ForwardPoolLayout
{
    vector<vector<Index>> slot_offsets;
    vector<vector<Index>> transient_slot_offsets;
    Index activation_bytes = 0;
    Index transient_bytes = 0;
    Index logical_bytes = 0;
    Index logical_persistent_bytes = 0;
    Index lower_bound_bytes = 0;
    Index fragmentation_bytes = 0;
    Index overlaid_scratch_bytes = 0;
    size_t overlaid_recompute_slots = 0;
};

struct PooledForwardSlots
{
    vector<pair<size_t, size_t>> persistent;
    vector<pair<size_t, size_t>> transient;
    vector<MemoryPoolEntry> lifetimes;
    Index transient_bytes = 0;
};

static bool is_transient_forward_slot(
    const ForwardPropagationMode mode,
    const vector<unique_ptr<Layer>>& layers,
    const vector<size_t>& recomputable_slots,
    const size_t layer,
    const size_t slot)
{
    return is_training(mode)
        && (layers[layer]->get_forward_slot_kind(slot + 1)
                == ForwardSlotKind::Transient
            || recomputable_slots[layer] == slot);
}

static ForwardPoolLayout initialize_forward_pool_layout(
    const vector<unique_ptr<Layer>>& layers,
    const vector<vector<TensorSpec>>& specs,
    const vector<size_t>& recomputable_slots,
    const ForwardPropagationMode mode)
{
    ForwardPoolLayout layout;
    layout.slot_offsets.resize(specs.size());
    layout.transient_slot_offsets.resize(specs.size());

    for(size_t i = 0; i < specs.size(); ++i)
    {
        layout.slot_offsets[i].assign(specs[i].size(), Index(-1));
        layout.transient_slot_offsets[i].assign(specs[i].size(), Index(-1));
        throw_if(recomputable_slots[i] != SIZE_MAX
                 && recomputable_slots[i] >= specs[i].size(),
                 "ForwardPropagation::set: invalid recomputable slot for layer {}.", i);

        for(size_t j = 0; j < specs[i].size(); ++j)
        {
            if(specs[i][j].shape.empty()) continue;
            const Index bytes = get_aligned_bytes(specs[i][j]);
            layout.logical_bytes += bytes;
            if(is_transient_forward_slot(mode, layers, recomputable_slots, i, j))
                throw_if(j + 1 == specs[i].size(),
                         "ForwardPropagation::set: a layer output cannot be a transient slot.");
            else
                layout.logical_persistent_bytes += bytes;
        }
    }
    return layout;
}

static PooledForwardSlots collect_forward_slots(
    const vector<unique_ptr<Layer>>& layers,
    const vector<vector<TensorSpec>>& specs,
    const vector<size_t>& recomputable_slots,
    const ForwardPropagationMode mode,
    const vector<Index>& output_release_steps)
{
    PooledForwardSlots pooled;
    const bool training = is_training(mode);
    const Index backward_base = backward_step(Index(specs.size()), 0);

    for(size_t i = 0; i < specs.size(); ++i)
        for(size_t j = 0; j < specs[i].size(); ++j)
        {
            const TensorSpec& spec = specs[i][j];
            if(spec.shape.empty()
               || is_transient_forward_slot(mode, layers, recomputable_slots, i, j))
                continue;

            const bool output = j + 1 == specs[i].size();
            const Index last_step = training
                ? (output && output_release_steps[i] >= 0
                    ? output_release_steps[i] : backward_base - Index(i))
                : (output ? output_release_steps[i] : Index(i));
            pooled.persistent.emplace_back(i, j);
            pooled.lifetimes.push_back({get_aligned_bytes(spec), Index(i), last_step});
        }

    if(!training) return pooled;
    for(size_t i = 0; i < specs.size(); ++i)
        for(size_t j = 0; j < specs[i].size(); ++j)
        {
            const TensorSpec& spec = specs[i][j];
            if(spec.shape.empty()
               || !is_transient_forward_slot(mode, layers, recomputable_slots, i, j)
               || recomputable_slots[i] == j)
                continue;

            const Index bytes = get_aligned_bytes(spec);
            pooled.transient.emplace_back(i, j);
            pooled.lifetimes.push_back({bytes, Index(i), Index(i)});
            pooled.transient_bytes += bytes;
        }
    return pooled;
}

static void apply_forward_pool_plan(
    ForwardPoolLayout& layout,
    const PooledForwardSlots& pooled,
    const MemoryPoolPlan& plan)
{
    for(size_t i = 0; i < pooled.persistent.size(); ++i)
        layout.slot_offsets[pooled.persistent[i].first][pooled.persistent[i].second]
            = plan.byte_offsets[i];

    const size_t transient_base = pooled.persistent.size();
    for(size_t i = 0; i < pooled.transient.size(); ++i)
        layout.transient_slot_offsets[pooled.transient[i].first][pooled.transient[i].second]
            = plan.byte_offsets[transient_base + i];

    layout.activation_bytes = plan.peak_bytes;
    layout.lower_bound_bytes = plan.lower_bound_live_bytes;
    layout.fragmentation_bytes = plan.fragmentation_bytes();
}

static Index place_unplanned_transient_slots(
    ForwardPoolLayout& layout,
    const vector<unique_ptr<Layer>>& layers,
    const vector<vector<TensorSpec>>& specs,
    const vector<size_t>& recomputable_slots,
    const ForwardPropagationMode mode)
{
    Index block_bytes = 0;
    for(size_t i = 0; i < specs.size(); ++i)
    {
        Index layer_bytes = 0;
        for(size_t j = 0; j < specs[i].size(); ++j)
        {
            if(!is_transient_forward_slot(mode, layers, recomputable_slots, i, j)
               || specs[i][j].shape.empty()
               || layout.transient_slot_offsets[i][j] >= 0)
                continue;
            layout.transient_slot_offsets[i][j] = layout.activation_bytes + layer_bytes;
            layer_bytes += get_aligned_bytes(specs[i][j]);
        }
        block_bytes = max(block_bytes, layer_bytes);
    }
    return block_bytes;
}

static void plan_training_forward_pool(
    ForwardPoolLayout& layout,
    PooledForwardSlots pooled,
    const vector<vector<TensorSpec>>& specs,
    const vector<size_t>& recomputable_slots,
    const span<const MemoryPoolEntry> co_planned_lifetimes,
    const bool exhaustive_training_plan,
    const size_t early_release_outputs,
    const Index early_release_logical_bytes,
    const Index batch_size,
    vector<Index>& co_planned_offsets)
{
    memory_debug::record_pool_lifetimes(
        "forward", pooled.lifetimes,
        format("layers={},batch={}", specs.size(), batch_size));
    const size_t forward_entry_count = pooled.lifetimes.size();
    pooled.lifetimes.insert(pooled.lifetimes.end(),
                            co_planned_lifetimes.begin(), co_planned_lifetimes.end());

    const MemoryPoolPlan plan = [&]
    {
        PROFILE_SCOPE_HOST("fp:set:plan");
        return exhaustive_training_plan
            ? plan_memory_pool_best(pooled.lifetimes)
            : plan_memory_pool(pooled.lifetimes,
                early_release_outputs > 0
                    ? MemoryPoolStrategy::Compact
                    : MemoryPoolStrategy::Chronological);
    }();
    apply_forward_pool_plan(layout, pooled, plan);

    if(!co_planned_lifetimes.empty())
    {
        co_planned_offsets.assign(plan.byte_offsets.begin() + forward_entry_count,
                                  plan.byte_offsets.end());
        const Index bytes = accumulate(co_planned_lifetimes.begin(), co_planned_lifetimes.end(),
                                       Index(0), [](const Index total, const MemoryPoolEntry& entry)
                                       { return total + entry.bytes; });
        memory_debug::record("forward.joint_plan", "co_planned_entries_in_arena", bytes,
                             format("batch={},entries={}", batch_size,
                                    co_planned_lifetimes.size()));
    }

    if(pooled.transient_bytes > 0)
        memory_debug::record("forward.transient_pool", "lifetime_planned_scratch",
                             pooled.transient_bytes,
                             format("batch={},entries={}", batch_size, pooled.transient.size()));

    const Index backward_base = backward_step(Index(specs.size()), 0);
    for(size_t i = 0; i < specs.size(); ++i)
    {
        const size_t slot = recomputable_slots[i];
        if(slot == SIZE_MAX || specs[i][slot].shape.empty()) continue;
        const Index bytes = get_aligned_bytes(specs[i][slot]);
        const Index second_step = backward_base - Index(i);
        const Index offset = find_memory_pool_overlay(
            pooled.lifetimes, plan, bytes, Index(i), second_step);
        if(offset >= 0)
        {
            layout.transient_slot_offsets[i][slot] = offset;
            ++layout.overlaid_recompute_slots;
            layout.overlaid_scratch_bytes += bytes;
        }
        memory_debug::record("forward.recompute_entry", format("{}:{}", i, slot), bytes,
                             format("first={},second={},overlaid={}", i, second_step,
                                    offset >= 0 ? 1 : 0));
    }

    if(early_release_outputs > 0)
    {
        memory_debug::record("forward.training_lifetime_reuse",
                             "early_release_output_bytes", early_release_logical_bytes,
                             format("batch={},layers={}", batch_size, early_release_outputs));
        memory_debug::record("forward.training_lifetime_reuse",
                             "allocated_persistent_bytes", layout.activation_bytes,
                             format("batch={},lower_bound_mib={:.2f}", batch_size,
                                    double(layout.lower_bound_bytes) / (1024.0 * 1024.0)));
    }
}

static void record_forward_pool_metrics(
    const ForwardPoolLayout& layout,
    const ForwardPropagationMode mode,
    const Index batch_size,
    const size_t recomputed_layers)
{
    if(layout.transient_bytes > 0)
        memory_debug::record("forward.transient_pool", "shared_block", layout.transient_bytes,
                             format("batch={}", batch_size));
    if(layout.overlaid_recompute_slots > 0)
        memory_debug::record("forward.training_recomputation", "overlaid_scratch_bytes",
                             layout.overlaid_scratch_bytes,
                             format("batch={},layers={}", batch_size,
                                    layout.overlaid_recompute_slots));

    if(!is_training(mode))
    {
        memory_debug::record("forward.inference_pool_analysis", "logical_persistent_bytes",
                             layout.logical_persistent_bytes, format("batch={}", batch_size));
        memory_debug::record("forward.inference_pool_analysis", "live_bytes_lower_bound",
                             layout.lower_bound_bytes, format("batch={}", batch_size));
        memory_debug::record("forward.inference_pool_analysis", "allocator_fragmentation_overhead",
                             layout.fragmentation_bytes, format("batch={}", batch_size));
        memory_debug::record("forward.inference_pool_analysis", "saved_bytes",
                             layout.logical_bytes - layout.activation_bytes,
                             format("batch={}", batch_size));
    }
    else if(recomputed_layers > 0)
    {
        const Index allocated_bytes = layout.activation_bytes + layout.transient_bytes;
        memory_debug::record("forward.training_recomputation", "logical_forward_bytes",
                             layout.logical_bytes, format("batch={}", batch_size));
        memory_debug::record("forward.training_recomputation", "allocated_forward_bytes",
                             allocated_bytes,
                             format("batch={},layers={}", batch_size, recomputed_layers));
        memory_debug::record("forward.training_recomputation", "saved_bytes",
                             layout.logical_bytes - allocated_bytes,
                             format("batch={}", batch_size));
    }
}

ForwardPropagation::ForwardPropagation(const Index new_batch_size,
                                       Network* new_network,
                                       const ForwardPropagationMode new_mode,
                                       const InferenceShapePolicy new_shape_policy,
                                       const bool new_inputs_pre_scaled,
                                       const span<const MemoryPoolEntry> co_planned_lifetimes,
                                       const bool exhaustive_training_plan)
{
    set(new_batch_size, new_network, nullptr, new_mode,
        new_shape_policy, new_inputs_pre_scaled, co_planned_lifetimes,
        exhaustive_training_plan);
}

ForwardPropagation::~ForwardPropagation()
{
    PROFILE_SCOPE_HOST("fp:dtor");
}

void ForwardPropagation::set(const Index new_batch_size,
                             Network* new_network,
                             Buffer* external_storage,
                             const bool new_inputs_pre_scaled,
                             TrainingArenaPlan& plan)
{
    set(new_batch_size, new_network, external_storage,
        ForwardPropagationMode::Training, InferenceShapePolicy{},
        new_inputs_pre_scaled, plan.co_planned_lifetimes(),
        plan.uses_joint_gradient());
    plan.bind_offsets(co_planned_offsets);
}

void ForwardPropagation::stage_position(DeviceStream stream)
{
#ifdef OPENNN_HAS_CUDA
    if (!position_pinned)
    {
        position_pinned.resize_bytes(Index(sizeof(int)));
        position_device.resize_bytes(Index(sizeof(int)), Device::CUDA);
    }

    *position_pinned.as<int>() = int(past_length);
    device::copy_async(position_device.data(),
                       position_pinned.data(),
                       Index(sizeof(int)),
                       device::CopyKind::HostToDevice, stream);
#else
    (void)stream;
#endif
}

void ForwardPropagation::set(
    const Index new_batch_size,
    Network* new_network,
    Buffer* external_storage,
    const ForwardPropagationMode new_mode,
    const InferenceShapePolicy new_shape_policy,
    const bool new_inputs_pre_scaled,
    const span<const MemoryPoolEntry> co_planned_lifetimes,
    const bool exhaustive_training_plan)
{
    throw_if(!new_network,
             "neural network is not set.");

    throw_if(new_mode != ForwardPropagationMode::Inference
             && (new_shape_policy.sequence_capacity > 0
                 || new_shape_policy.final_output_capacity > 0),
             "ForwardPropagation::set: compact capacities are inference-only.");

    throw_if(new_mode != ForwardPropagationMode::Inference
             && !new_shape_policy.retained_output_layers.empty(),
             "ForwardPropagation::set: retained outputs are inference-only; "
             "training keeps every activation alive for the backward pass.");

    PROFILE_SCOPE_HOST("fp:set");

    reset_cuda_graph();
    co_planned_offsets.clear();

    batch_size = new_batch_size;
    network = new_network;
    mode = new_mode;
    past_length = 0;

    const auto& layers = network->get_layers();
    const size_t layers_number = layers.size();

    position_staging_required = ranges::any_of(
        layers, [](const unique_ptr<Layer>& layer)
        {
            return layer && layer->uses_sequence_position();
        });

    staged_input_storage.clear();
    layer_state_storage.clear();
    layer_session_state_storage = make_shared<vector<Buffer>>();
    layer_pinned_storage.clear();
    staged_inputs.clear();
    host_bf16_input_scratch.clear();
    passthrough_overrides.clear();
    device_valid_length_storage.clear();
    output_window.reset();

    inputs.assign(layers_number, {});
    slots.assign(layers_number, {});
    drelu_fused_by_layer.assign(layers_number, uint8_t{0});
    layer_state_storage.reserve(layers_number);
    layer_session_state_storage->reserve(layers_number);
    for (size_t i = 0; i < layers_number; ++i)
    {
        layer_state_storage.emplace_back(network->get_device());
        layer_session_state_storage->emplace_back(
            network->get_device());
    }
    layer_pinned_storage.resize(layers_number);
    valid_lengths.assign(layers_number, {});
    device_valid_lengths.assign(layers_number, nullptr);
    device_valid_length_storage.resize(layers_number);

    auto forward_specs = [&]
    {
        PROFILE_SCOPE_HOST("fp:set:specs");
        return network->get_forward_specs(batch_size);
    }();

    throw_if(forward_specs.size() != layers_number,
             "ForwardPropagation::set: forward specs size ({}) does not match layers number ({}).",
             forward_specs.size(),
             layers_number);

    const auto& source_layers = network->get_source_layers();

    throw_if(source_layers.size() != layers_number,
             "ForwardPropagation::set: source layers size ({}) does not match layers number ({}).",
             source_layers.size(),
             layers_number);

    execution_start_layer = 0;
    if (new_inputs_pre_scaled)
        while (size_t(execution_start_layer) < layers_number
              && layers[size_t(execution_start_layer)]->skip_for_pre_scaled_input())
            ++execution_start_layer;

    for (Index i = 0; i < execution_start_layer; ++i)
        forward_specs[size_t(i)].clear();

    const ForwardCapacities capacities = apply_inference_shape_policy(
        forward_specs, network->get_input_shape(), new_shape_policy);
    sequence_capacity = capacities.sequence;
    final_output_capacity = capacities.output;
    final_output_layer = capacities.output_layer;

    recomputable_slots.assign(layers_number, SIZE_MAX);

    if(is_training(mode)
       && network->get_training_activation_recomputation())
    {
        ranges::transform(
            layers,
            recomputable_slots.begin(),
            [](const auto& layer)
            {
                const size_t slot = layer->get_recomputable_forward_slot();
                return slot == SIZE_MAX ? SIZE_MAX : slot - 1;
            });
    }

    if(!is_training(mode))
        elide_inference_slots(layers, forward_specs, network->get_device(), batch_size);

    Index early_release_logical_bytes = 0;
    const vector<Index> output_release_steps = is_training(mode)
        ? find_early_output_release_steps(layers, network->get_consumer_edges(),
                                          forward_specs, early_release_logical_bytes)
        : vector<Index>(layers_number, Index(-1));
    const size_t early_release_outputs = ranges::count_if(
        output_release_steps, [](const Index step) { return step >= 0; });

    ForwardPoolLayout pool = initialize_forward_pool_layout(
        layers, forward_specs, recomputable_slots, mode);
    if(is_training(mode))
    {
        PooledForwardSlots pooled = collect_forward_slots(
            layers, forward_specs, recomputable_slots, mode, output_release_steps);
        plan_training_forward_pool(
            pool, move(pooled), forward_specs, recomputable_slots,
            co_planned_lifetimes, exhaustive_training_plan,
            early_release_outputs, early_release_logical_bytes,
            batch_size, co_planned_offsets);
        pool.transient_bytes = place_unplanned_transient_slots(
            pool, layers, forward_specs, recomputable_slots, mode);
    }
    else
    {
        const vector<Index> release_steps = find_inference_output_release_steps(
            forward_specs, source_layers, new_shape_policy.retained_output_layers,
            network->get_last_trainable_layer_index());
        const PooledForwardSlots pooled = collect_forward_slots(
            layers, forward_specs, recomputable_slots, mode, release_steps);
        const MemoryPoolPlan plan = [&]
        {
            PROFILE_SCOPE_HOST("fp:set:plan");
            return plan_memory_pool(pooled.lifetimes, MemoryPoolStrategy::Compact);
        }();
        apply_forward_pool_plan(pool, pooled, plan);
    }

    const Index total_bytes = pool.activation_bytes + pool.transient_bytes;
    if(external_storage
       && external_storage->get_device() == network->get_device()
       && external_storage->byte_size() >= total_bytes)
        arena.set_view(external_storage->data(), total_bytes,
                       external_storage->get_device());
    else
    {
        PROFILE_SCOPE_HOST("fp:set:alloc");
        arena.resize_bytes(total_bytes, network->get_device());
    }
    {
        PROFILE_SCOPE_HOST("fp:set:zero");
        arena.setZero();
    }

    memory_debug::record(
        arena.owns_memory() ? "forward" : "forward.aliased",
        "ForwardPropagation::arena", arena.owns_memory() ? total_bytes : 0,
        format("batch={},mode={}", batch_size,
               is_training(mode) ? "training" : "inference"));
    const size_t recomputed_layers = ranges::count_if(
        recomputable_slots, [](const size_t slot) { return slot != SIZE_MAX; });
    record_forward_pool_metrics(pool, mode, batch_size, recomputed_layers);

    device::set_conv_workspace_auto_limit_bytes(bind_slots(
        forward_specs, pool.slot_offsets, pool.transient_slot_offsets));

    capacity_inputs = inputs;
    capacity_slots = slots;
    active_sequence_length = sequence_capacity;

    if(new_shape_policy.final_output_capacity > 0)
        output_window.emplace(OutputWindow{Buffer{}, 0, 0});

    if(new_shape_policy.sequence_capacity > 0)
        set_active_sequence_length(sequence_capacity);
}

Index ForwardPropagation::bind_slots(
    const vector<vector<TensorSpec>>& forward_specs,
    const vector<vector<Index>>& slot_offsets,
    const vector<vector<Index>>& transient_slot_offsets)
{
    const auto& layers = network->get_layers();
    const auto& source_layers = network->get_source_layers();
    uint8_t* const arena_base = arena.as<uint8_t>();

    Index max_layer_bytes = 0;

    for (size_t i = 0; i < forward_specs.size(); ++i)
    {
        const auto& specs = forward_specs[i];
        max_layer_bytes = max(max_layer_bytes, get_aligned_bytes(specs));

        slots[i].assign(specs.size() + 1, TensorView{});

        Index layer_logical_bytes = 0;
        for (size_t j = 0; j < specs.size(); ++j)
        {
            const auto& [shape, dtype] = specs[j];
            if (shape.empty()) continue;

            const bool transient = transient_slot_offsets[i][j] >= 0;
            const Index offset = transient ? transient_slot_offsets[i][j]
                                           : slot_offsets[i][j];
            throw_if(offset < 0,
                     "ForwardPropagation::set: no planned offset for layer {} slot {}.",
                     i, j);

            slots[i][j + 1] =
                TensorView(arena_base + offset, shape, dtype, arena.get_device());

            if (!transient) layer_logical_bytes += get_aligned_bytes(specs[j]);
        }

        if (layer_logical_bytes > 0)
            memory_debug::record("forward.layer",
                                 format("{}:{}", i, layers[i]->get_label()),
                                 layer_logical_bytes,
                                 format("batch={}", batch_size));

        const vector<Index>& sources = source_layers[i];
        inputs[i].assign(sources.size(), TensorView{});

        for (size_t j = 0; j < sources.size(); ++j)
        {
            const Index source_layer = sources[j];
            if (source_layer < 0) continue;

            if (!forward_specs[source_layer].empty())
            {
                inputs[i][j] = slots[source_layer].back();
                continue;
            }

            const Index resolved =
                resolve_producer(forward_specs, source_layers, source_layer);

            if (resolved < 0)
            {
                passthrough_overrides.emplace_back(i, j, size_t(-resolved - 1));
                continue;
            }

            TensorView view = slots[resolved].back();
            if (!view.empty())
                view = view.reshape(Shape{view.get_shape()[0]}
                    .append(layers[source_layer]->get_output_shape()));
            inputs[i][j] = view;
        }
    }

    return max_layer_bytes;
}

uint64_t ForwardPropagation::get_parameters_version() const
{
    return network ? network->get_parameters_version() : 0;
}

void ForwardPropagation::recompute_for_backward(Index layer_index)
{
    if (layer_index < 0
        || size_t(layer_index) >= recomputable_slots.size()
        || recomputable_slots[size_t(layer_index)] == SIZE_MAX)
        return;

    network->get_layers()[size_t(layer_index)]
        ->recompute_forward_slot(*this, size_t(layer_index));
}

void ForwardPropagation::set_active_sequence_length(Index length)
{
    throw_if(length < 1 || length > sequence_capacity,
             "ForwardPropagation::set_active_sequence_length: length {} is "
             "outside [1, {}].",
             length, sequence_capacity);

    reset_cuda_graph();

    inputs = capacity_inputs;
    slots = capacity_slots;
    active_sequence_length = length;

    const auto shrink_sequence = [this, length](TensorView& view)
    {
        if (!view.empty() && view.get_rank() >= 2
            && view.get_shape()[1] == sequence_capacity)
        {
            Shape active_shape = view.get_shape();
            active_shape.set_dimension(1, length);
            view = view.reshape_prefix(active_shape);
        }
    };

    for (auto& layer_slots : slots)
        for (auto& slot : layer_slots) shrink_sequence(slot);

    for (auto& layer_inputs : inputs)
        for (auto& view : layer_inputs) shrink_sequence(view);

    if (output_window)
    {
        const Index count = min(final_output_capacity, length);
        set_output_sequence_window(length - count, count);
    }
}

void ForwardPropagation::share_session_state_from(
    const ForwardPropagation& source)
{
    throw_if(!network || network != source.network,
             "ForwardPropagation::share_session_state_from requires both "
             "propagations to execute the same network.");
    throw_if(mode != ForwardPropagationMode::Inference
             || source.mode != ForwardPropagationMode::Inference,
             "ForwardPropagation::share_session_state_from is inference-only.");
    throw_if(!layer_session_state_storage
             || !source.layer_session_state_storage
             || layer_session_state_storage->size()
                    != source.layer_session_state_storage->size(),
             "ForwardPropagation::share_session_state_from: layer counts do "
             "not match.");

    reset_cuda_graph();
    layer_session_state_storage = source.layer_session_state_storage;
}

void ForwardPropagation::set_output_sequence_window(Index start, Index count)
{
    throw_if(!output_window,
             "ForwardPropagation::set_output_sequence_window requires a "
             "compact final output capacity.");
    throw_if(start < 0 || count < 1
             || start + count > active_sequence_length,
             "ForwardPropagation::set_output_sequence_window: window [{}, {}) "
             "is outside the active sequence length {}.",
             start, start + count, active_sequence_length);
    throw_if(count > final_output_capacity,
             "ForwardPropagation::set_output_sequence_window: {} rows exceed "
             "the final output capacity {}.",
             count, final_output_capacity);
    throw_if(final_output_layer < 0
             || size_t(final_output_layer) >= inputs.size()
             || inputs[size_t(final_output_layer)].empty(),
             "ForwardPropagation::set_output_sequence_window: final layer has "
             "no input view.");

    reset_cuda_graph();

    TensorView& input = inputs[size_t(final_output_layer)].front();
    const TensorView& capacity_input =
        capacity_inputs[size_t(final_output_layer)].front();
    throw_if(capacity_input.empty() || capacity_input.get_rank() < 2,
             "ForwardPropagation::set_output_sequence_window: final layer "
             "input is not sequence-shaped.");

    const Shape& capacity_shape = capacity_input.get_shape();
    const Index row_bytes =
        capacity_shape.size() / capacity_shape[0] / capacity_shape[1]
        * type_bytes(capacity_input.get_type());

    OutputWindow& window = *output_window;
    window.start = start;
    window.count = count;

    Shape window_shape = capacity_shape;
    window_shape.set_dimension(1, count);

    if (batch_size == 1)
    {
        window.input.resize_bytes(0, capacity_input.get_device());
        input = TensorView(static_cast<char*>(capacity_input.get_data()) + start * row_bytes,
                           window_shape,
                           capacity_input.get_type(),
                           capacity_input.get_device());
    }
    else
    {
        window.input.resize_bytes(batch_size * count * row_bytes,
                                  capacity_input.get_device());
        input = TensorView(window.input.data(),
                           window_shape,
                           capacity_input.get_type(),
                           capacity_input.get_device());
    }

    TensorView& output = slots[size_t(final_output_layer)].back();
    const TensorView& capacity_output = capacity_slots[size_t(final_output_layer)].back();
    Shape output_shape = capacity_output.get_shape();
    output_shape.set_dimension(1, count);
    output = capacity_output.reshape_prefix(output_shape);
}

void ForwardPropagation::gather_output_window()
{
    if (!output_window || output_window->input.empty()) return;

    const OutputWindow& window = *output_window;

    const TensorView& capacity_input =
        capacity_inputs[size_t(final_output_layer)].front();

    const Shape& capacity_shape = capacity_input.get_shape();
    const Index sequence = capacity_shape[1];
    const Index row_bytes = capacity_shape.size() / capacity_shape[0]
                          / sequence * type_bytes(capacity_input.get_type());
    const Index window_bytes = window.count * row_bytes;

    for (Index sample = 0; sample < batch_size; ++sample)
        device::copy_async(
            static_cast<char*>(window.input.data()) + sample * window_bytes,
            static_cast<const char*>(capacity_input.get_data())
                + (sample * sequence + window.start) * row_bytes,
            window_bytes,
            capacity_input.get_device(), capacity_input.get_device(),
            device::get_compute_stream());
}

TensorView ForwardPropagation::get_layer_outputs(const Index layer) const
{
    if (!network || layer < 0 || size_t(layer) >= slots.size())
        return {};

    const auto& layer_slots = slots[size_t(layer)];
    if (!layer_slots.empty() && !layer_slots.back().empty())
        return layer_slots.back();

    if (size_t(layer) >= inputs.size() || inputs[size_t(layer)].empty())
        return {};

    TensorView input = inputs[size_t(layer)].front();
    if (!input.empty())
        input = input.reshape(Shape{input.get_shape()[0]}.append(
            network->get_layers()[size_t(layer)]->get_output_shape()));
    return input;
}

TensorView ForwardPropagation::get_last_trainable_layer_outputs() const
{
    return network
        ? get_layer_outputs(network->get_last_trainable_layer_index())
        : TensorView{};
}

Index ForwardPropagation::valid_lengths_source(const size_t layer, const size_t input_ordinal) const
{
    if (!network) return -1;

    const auto& source_layers = network->get_source_layers();
    if (layer >= source_layers.size()) return -1;

    const vector<Index>& sources = source_layers[layer];
    if (input_ordinal >= sources.size()) return -1;

    const Index source = sources[input_ordinal];
    return (source < 0 || size_t(source) >= valid_lengths.size()) ? -1 : source;
}

const vector<Index>* ForwardPropagation::input_valid_lengths(const size_t layer,
                                                             const size_t input_ordinal) const
{
    const Index source = valid_lengths_source(layer, input_ordinal);
    if (source < 0) return nullptr;

    const vector<Index>& lengths = valid_lengths[size_t(source)];

    return lengths.empty() ? nullptr : &lengths;
}

const int* ForwardPropagation::input_device_valid_lengths(const size_t layer,
                                                          const size_t input_ordinal) const
{
    const Index source = valid_lengths_source(layer, input_ordinal);
    return source < 0 ? nullptr : device_valid_lengths[size_t(source)];
}

SequenceLengths ForwardPropagation::input_sequence_lengths(const size_t layer,
                                                          const size_t input_ordinal) const
{
    return {input_valid_lengths(layer, input_ordinal),
            input_device_valid_lengths(layer, input_ordinal)};
}

int* ForwardPropagation::device_valid_lengths_slot(const size_t layer, const Index requested_batch_size)
{
    Buffer& storage = device_valid_length_storage[layer];
    if (storage.get_device() != Device::CUDA)
        storage.resize_bytes(requested_batch_size * Index(sizeof(int)), Device::CUDA);
    else
        storage.grow_to(requested_batch_size * Index(sizeof(int)));

    device_valid_lengths[layer] = storage.as<int>();
    return storage.as<int>();
}

void ForwardPropagation::inherit_valid_lengths(const size_t layer)
{
    if (layer >= valid_lengths.size()) return;

    const vector<Index>* source_lengths = input_valid_lengths(layer, 0);
    const int* device_source_lengths = input_device_valid_lengths(layer, 0);
    if (!source_lengths && !device_source_lengths) return;

    const auto& layers = network->get_layers();
    const Index source = network->get_source_layers()[layer][0];

    const Shape output_shape = layers[layer]->get_output_shape();
    const Shape source_shape = layers[size_t(source)]->get_output_shape();

    if (output_shape.get_rank() < 2 || source_shape.get_rank() < 2) return;
    if (output_shape[0] != source_shape[0]) return;

    if (source_lengths) valid_lengths[layer] = *source_lengths;
    device_valid_lengths[layer] = device_source_lengths;
}

TensorView ForwardPropagation::get_outputs() const
{
    if (!network) return {};

    const Index last = Index(network->get_layers_number()) - 1;
    TensorView output = get_layer_outputs(last);
    return output.empty() ? get_last_trainable_layer_outputs() : output;
}

void ForwardPropagation::set_cuda_graph(bool enabled)
{
    use_cuda_graph = enabled;
    cuda_graph_failed = false;
    if (!enabled) reset_cuda_graph();
}

void ForwardPropagation::reset_cuda_graph() noexcept
{
    inference_graph_exec.reset();
    captured_input_pointers.clear();
    cuda_graph_warmup_calls = 0;
    inference_graph_workspace_requirements = {};
}

void ForwardPropagation::prepare_cuda_graph_workspaces()
{
    for(size_t i = 0; i < inference_graph_workspaces.size(); ++i)
    {
        Buffer& buffer = inference_graph_workspaces[i];
        const Index growth = inference_graph_workspace_requirements[i] - buffer.byte_size();

        if(growth <= 0) continue;

        buffer.grow_to(inference_graph_workspace_requirements[i]);

        memory_debug::record("forward.graph_workspace",
                             device::graph_workspace_labels[i],
                             growth,
                             format("batch={}", batch_size));
    }
}

bool ForwardPropagation::cuda_graph_workspaces_need_growth() const noexcept
{
    for (size_t i = 0; i < inference_graph_workspaces.size(); ++i)
        if (inference_graph_workspace_requirements[i]
            > inference_graph_workspaces[i].byte_size())
            return true;

    return false;
}

device::GraphWorkspaceViews 
ForwardPropagation::get_cuda_graph_workspace_views() const noexcept
{
    device::GraphWorkspaceViews views{};

    for (size_t i = 0; i < views.size(); ++i)
        views[i] = {inference_graph_workspaces[i].data(),
                    inference_graph_workspaces[i].byte_size()};

    return views;
}

}
