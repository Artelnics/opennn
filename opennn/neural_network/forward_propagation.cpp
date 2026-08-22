//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   F O R W A R D   P R O P A G A T I O N   S O U R C E
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "opennn/neural_network/forward_propagation.h"
#include "opennn/registry.h"
#include "opennn/neural_network/neural_network.h"
#include "opennn/core/memory_debug.h"
#include "opennn/core/device_backend.h"
#include "opennn/core/memory_pool.h"
#include "opennn/core/profiler.h"
#include "opennn/core/string_utilities.h"

namespace opennn
{

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
    const vector<vector<Index>>& source_layers,
    const vector<vector<TensorSpec>>& forward_specs,
    Index& released_bytes)
{
    vector<vector<pair<size_t, size_t>>> consumers(layers.size());
    for (size_t consumer = 0; consumer < source_layers.size(); ++consumer)
        for (size_t input = 0; input < source_layers[consumer].size(); ++input)
        {
            const Index source = source_layers[consumer][input];
            if (source >= 0)
                consumers[size_t(source)].push_back({consumer, input});
        }

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

ForwardPropagation::ForwardPropagation(const Index new_batch_size,
                                       NeuralNetwork* new_neural_network,
                                       const ForwardPropagationMode new_mode,
                                       const InferenceShapePolicy new_shape_policy,
                                       const bool new_inputs_pre_scaled,
                                       const span<const MemoryPoolEntry> co_planned_lifetimes)
{
    set(new_batch_size, new_neural_network, nullptr, new_mode,
        new_shape_policy, new_inputs_pre_scaled, co_planned_lifetimes);
}

ForwardPropagation::~ForwardPropagation()
{
    PROFILE_SCOPE_HOST("fp:dtor");
}

void ForwardPropagation::stage_position(cudaStream_t stream)
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
    NeuralNetwork* new_neural_network,
    Buffer* external_storage,
    const ForwardPropagationMode new_mode,
    const InferenceShapePolicy new_shape_policy,
    const bool new_inputs_pre_scaled,
    const span<const MemoryPoolEntry> co_planned_lifetimes)
{
    throw_if(!new_neural_network,
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
    neural_network = new_neural_network;
    mode = new_mode;
    past_length = 0;

    const auto& layers = neural_network->get_layers();
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
    valid_lengths.clear();
    device_valid_lengths.clear();
    device_valid_length_storage.clear();
    output_window.reset();

    inputs.resize(layers_number);
    slots.resize(layers_number);
    drelu_fused_by_layer.assign(layers_number, uint8_t{0});
    layer_state_storage.reserve(layers_number);
    layer_session_state_storage->reserve(layers_number);
    for (size_t i = 0; i < layers_number; ++i)
    {
        layer_state_storage.emplace_back(neural_network->get_device());
        layer_session_state_storage->emplace_back(
            neural_network->get_device());
    }
    layer_pinned_storage.resize(layers_number);
    valid_lengths.resize(layers_number);
    device_valid_lengths.assign(layers_number, nullptr);
    device_valid_length_storage.resize(layers_number);

    auto forward_specs = [&]
    {
        PROFILE_SCOPE_HOST("fp:set:specs");
        return neural_network->get_forward_specs(batch_size);
    }();

    throw_if(forward_specs.size() != layers_number,
             "ForwardPropagation::set: forward specs size ({}) does not match layers number ({}).",
             forward_specs.size(),
             layers_number);

    const auto& source_layers = neural_network->get_source_layers();

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

    const Shape model_input_shape = neural_network->get_input_shape();

    const Index model_sequence_capacity =
        model_input_shape.empty() ? Index(0) : model_input_shape[0];

    sequence_capacity =
        new_shape_policy.sequence_capacity > 0
        ? new_shape_policy.sequence_capacity
        : model_sequence_capacity;

    throw_if(new_shape_policy.sequence_capacity > model_sequence_capacity,
             "ForwardPropagation::set: sequence capacity {} exceeds the "
             "network capacity {}.",
             new_shape_policy.sequence_capacity,
             model_sequence_capacity);

    if(new_shape_policy.sequence_capacity > 0)
    {
        for(auto& layer_specs : forward_specs)
        {
            for(TensorSpec& spec : layer_specs)
            {
                if(spec.shape.get_rank() >= 2
                   && spec.shape[1] == model_sequence_capacity)
                {
                    spec.shape.set_dimension(1, sequence_capacity);
                }
            }
        }
    }

    final_output_layer = -1;

    for(const size_t i :
        views::iota(size_t(0), layers_number) | views::reverse)
    {
        if(forward_specs[i].empty()) continue;

        final_output_layer = Index(i);
        break;
    }

    final_output_capacity =
        new_shape_policy.final_output_capacity > 0
        ? new_shape_policy.final_output_capacity
        : sequence_capacity;

    throw_if(new_shape_policy.final_output_capacity > 0
             && new_shape_policy.sequence_capacity <= 0,
             "ForwardPropagation::set: final_output_capacity requires an "
             "explicit sequence_capacity.");

    throw_if(final_output_capacity > sequence_capacity,
             "ForwardPropagation::set: final output capacity {} exceeds "
             "sequence capacity {}.",
             final_output_capacity,
             sequence_capacity);

    if(new_shape_policy.final_output_capacity > 0
       && final_output_layer >= 0)
    {
        TensorSpec& output_spec =
            forward_specs[size_t(final_output_layer)].back();

        throw_if(output_spec.shape.get_rank() < 2
                 || output_spec.shape[1] != sequence_capacity,
                 "ForwardPropagation::set: final output does not expose a "
                 "sequence dimension compatible with compact inference.");

        output_spec.shape.set_dimension(1, final_output_capacity);
    }

    const bool is_training =
        mode == ForwardPropagationMode::Training;

    recomputable_slots.assign(layers_number, SIZE_MAX);

    if(is_training
       && neural_network->get_training_activation_recomputation())
    {
        ranges::transform(
            layers,
            recomputable_slots.begin(),
            [](const auto& layer)
            {
                return layer->get_recomputable_forward_slot();
            });
    }

    if(!is_training)
    {
        for(size_t i = 0; i < layers_number; ++i)
        {
            for(size_t j = 0; j < forward_specs[i].size(); ++j)
            {
                if(layers[i]->get_forward_slot_kind(j)
                   == ForwardSlotKind::TrainingOnly)
                {
                    forward_specs[i][j] = {};
                }
            }
        }
    }

    const auto is_transient_slot =
        [&](const size_t layer, const size_t slot)
    {
        return is_training
            && (layers[layer]->get_forward_slot_kind(slot)
                    == ForwardSlotKind::Transient
                || recomputable_slots[layer] == slot);
    };

    Index early_release_logical_bytes = 0;

    const vector<Index> output_release_steps =
        is_training
        ? find_early_output_release_steps(
              layers,
              source_layers,
              forward_specs,
              early_release_logical_bytes)
        : vector<Index>(layers_number, Index(-1));

    const size_t early_release_outputs =
        ranges::count_if(
            output_release_steps,
            [](const Index step)
            {
                return step >= 0;
            });

    vector<vector<Index>> slot_offsets(layers_number);
    vector<vector<Index>> transient_slot_offsets(layers_number);

    Index logical_total_bytes = 0;
    Index logical_persistent_bytes = 0;

    for(size_t i = 0; i < layers_number; ++i)
    {
        slot_offsets[i].assign(
            forward_specs[i].size(),
            Index(-1));

        transient_slot_offsets[i].assign(
            forward_specs[i].size(),
            Index(-1));

        throw_if(recomputable_slots[i] != SIZE_MAX
                 && recomputable_slots[i] >= forward_specs[i].size(),
                 "ForwardPropagation::set: invalid recomputable slot for layer {}.",
                 i);

        for(size_t j = 0; j < forward_specs[i].size(); ++j)
        {
            const TensorSpec& spec = forward_specs[i][j];

            if(spec.shape.empty()) continue;

            const Index bytes = get_aligned_bytes(spec);

            logical_total_bytes += bytes;

            if(is_transient_slot(i, j))
            {
                throw_if(
                    j + 1 == forward_specs[i].size(),
                    "ForwardPropagation::set: a layer output cannot be a transient slot.");
            }
            else
            {
                logical_persistent_bytes += bytes;
            }
        }
    }

    Index activation_pool_bytes = 0;
    Index lower_bound_live_bytes = 0;
    Index fragmentation_bytes = 0;
    Index transient_block_bytes = 0;

    size_t overlaid_recompute_slots = 0;
    Index overlaid_scratch_bytes = 0;

    const auto place_transient_slots = [&]() -> Index
    {
        Index block_bytes = 0;

        for(size_t i = 0; i < layers_number; ++i)
        {
            Index layer_bytes = 0;

            for(size_t j = 0; j < forward_specs[i].size(); ++j)
            {
                if(!is_transient_slot(i, j)
                   || forward_specs[i][j].shape.empty()
                   || transient_slot_offsets[i][j] >= 0)
                {
                    continue;
                }

                transient_slot_offsets[i][j] =
                    activation_pool_bytes + layer_bytes;

                layer_bytes +=
                    get_aligned_bytes(forward_specs[i][j]);
            }

            block_bytes = max(block_bytes, layer_bytes);
        }

        return block_bytes;
    };

    vector<pair<size_t, size_t>> pooled_slots;
    vector<MemoryPoolEntry> pooled_lifetimes;

    const auto collect_pooled_slots = [&](auto&& last_step_for)
    {
        for(size_t i = 0; i < layers_number; ++i)
        {
            for(size_t j = 0; j < forward_specs[i].size(); ++j)
            {
                const TensorSpec& spec = forward_specs[i][j];

                if(spec.shape.empty() || is_transient_slot(i, j))
                    continue;

                const bool is_output =
                    j + 1 == forward_specs[i].size();

                pooled_slots.emplace_back(i, j);

                pooled_lifetimes.push_back(
                    {get_aligned_bytes(spec),
                     Index(i),
                     last_step_for(i, is_output)});
            }
        }
    };

    const auto apply_pool_plan =
        [&](const MemoryPoolPlan& plan)
    {
        for(size_t i = 0; i < pooled_slots.size(); ++i)
        {
            slot_offsets[pooled_slots[i].first]
                        [pooled_slots[i].second] =
                plan.byte_offsets[i];
        }

        activation_pool_bytes = plan.peak_bytes;
        lower_bound_live_bytes = plan.lower_bound_live_bytes;
        fragmentation_bytes = plan.fragmentation_bytes();
    };

    if(is_training)
    {
        const Index backward_base =
            backward_step(Index(layers_number), 0);

        collect_pooled_slots(
            [&](const size_t i, const bool is_output)
            {
                return is_output && output_release_steps[i] >= 0
                    ? output_release_steps[i]
                    : backward_base - Index(i);
            });

        memory_debug::record_pool_lifetimes(
            "forward",
            pooled_lifetimes,
            format("layers={},batch={}",
                   layers_number,
                   batch_size));

        const size_t forward_entry_count =
            pooled_lifetimes.size();

        pooled_lifetimes.insert(
            pooled_lifetimes.end(),
            co_planned_lifetimes.begin(),
            co_planned_lifetimes.end());

        // Chronological is load-bearing here, not a default. Activation
        // recomputation relies on a scratch slot landing on top of a future
        // activation, which first_step ordering produces and largest-first does
        // not. Forcing Compact was measured: the joint arena gets strictly smaller
        // (an MLP drops batch * outputs * sizeof(float); ResNet-50 and Transformer
        // are byte-identical, already taking that branch), but it breaks the
        // recompute aliasing pinned by
        // ForwardPropagationMemoryTest.TrainingRecomputeScratchUsesFutureActivations
        // and YoloOverfit.CSPGradientFlowsAndLossDecreases then stops learning.
        // Do not simplify this to always-Compact.

        const MemoryPoolPlan persistent_plan = [&]
        {
            PROFILE_SCOPE_HOST("fp:set:plan");
            return plan_memory_pool(
                pooled_lifetimes,
                early_release_outputs > 0
                    ? MemoryPoolStrategy::Compact
                    : MemoryPoolStrategy::Chronological);
        }();

        apply_pool_plan(persistent_plan);

        if(!co_planned_lifetimes.empty())
        {
            co_planned_offsets.assign(
                persistent_plan.byte_offsets.begin()
                    + forward_entry_count,
                persistent_plan.byte_offsets.end());

            Index co_planned_bytes = 0;
            for(const MemoryPoolEntry& entry : co_planned_lifetimes)
                co_planned_bytes += entry.bytes;

            memory_debug::record(
                "forward.joint_plan",
                "delta_entries_in_arena",
                co_planned_bytes,
                format("batch={},entries={}",
                       batch_size,
                       co_planned_lifetimes.size()));
        }

        for(size_t i = 0; i < layers_number; ++i)
        {
            const size_t slot =
                recomputable_slots[i];

            if(slot == SIZE_MAX
               || forward_specs[i][slot].shape.empty())
            {
                continue;
            }

            const Index bytes =
                get_aligned_bytes(forward_specs[i][slot]);

            const Index backward_step =
                backward_base - Index(i);

            const Index overlay_offset = find_memory_pool_overlay(
                pooled_lifetimes,
                persistent_plan,
                bytes,
                Index(i),
                backward_step);

            if(overlay_offset >= 0)
            {
                transient_slot_offsets[i][slot] =
                    overlay_offset;

                ++overlaid_recompute_slots;
                overlaid_scratch_bytes += bytes;
            }

            memory_debug::record(
                "forward.recompute_entry",
                format("{}:{}", i, slot),
                bytes,
                format("first={},second={},overlaid={}",
                       i,
                       backward_step,
                       overlay_offset >= 0 ? 1 : 0));
        }

        transient_block_bytes =
            place_transient_slots();

        if(early_release_outputs > 0)
        {
            memory_debug::record(
                "forward.training_lifetime_reuse",
                "early_release_output_bytes",
                early_release_logical_bytes,
                format("batch={},layers={}",
                       batch_size,
                       early_release_outputs));

            memory_debug::record(
                "forward.training_lifetime_reuse",
                "allocated_persistent_bytes",
                activation_pool_bytes,
                format(
                    "batch={},lower_bound_mib={:.2f}",
                    batch_size,
                    double(lower_bound_live_bytes)
                        / (1024.0 * 1024.0)));
        }
    }
    else
    {
        const Index final_step =
            layers_number == 0
            ? 0
            : Index(layers_number - 1);

        vector<Index> last_consumers(layers_number);
        vector<bool> has_consumers(layers_number, false);

        iota(
            last_consumers.begin(),
            last_consumers.end(),
            Index(0));

        for(size_t consumer = 0;
            consumer < layers_number;
            ++consumer)
        {
            for(const Index source_layer :
                source_layers[consumer])
            {
                const Index producer =
                    resolve_producer(
                        forward_specs,
                        source_layers,
                        source_layer);

                if(producer < 0) continue;

                has_consumers[size_t(producer)] = true;

                last_consumers[size_t(producer)] =
                    max(last_consumers[size_t(producer)],
                        Index(consumer));
            }
        }

        vector<bool> externally_observable(
            layers_number,
            false);

        for(size_t i = 0; i < layers_number; ++i)
        {
            if(!has_consumers[i])
            {
                externally_observable[i] = true;
            }
        }

        const auto mark_resolved_output =
            [&](const Index layer_index)
        {
            if(layer_index < 0
               || size_t(layer_index) >= layers_number)
            {
                return;
            }

            const Index producer =
                resolve_producer(
                    forward_specs,
                    source_layers,
                    layer_index);

            if(producer >= 0)
                externally_observable[size_t(producer)] = true;
        };

        mark_resolved_output(
            Index(layers_number) - 1);

        mark_resolved_output(
            neural_network->get_last_trainable_layer_index());

        for(const Index retained :
            new_shape_policy.retained_output_layers)
        {
            throw_if(
                retained < 0
                || size_t(retained) >= layers_number,
                "ForwardPropagation::set: retained output layer {} is out "
                "of range (network has {} layers).",
                retained,
                layers_number);

            mark_resolved_output(retained);
        }

        collect_pooled_slots(
            [&](const size_t i, const bool is_output)
            {
                if(!is_output)
                    return Index(i);

                return externally_observable[i]
                    ? final_step
                    : last_consumers[i];
            });

        apply_pool_plan([&]
        {
            PROFILE_SCOPE_HOST("fp:set:plan");
            return plan_memory_pool(pooled_lifetimes, MemoryPoolStrategy::Compact);
        }());
    }

    const Index total_bytes =
        activation_pool_bytes + transient_block_bytes;

    if(external_storage
       && external_storage->get_device()
              == neural_network->get_device()
       && external_storage->byte_size() >= total_bytes)
    {
        arena.set_view(
            external_storage->data(),
            total_bytes,
            external_storage->get_device());
    }
    else
    {
        PROFILE_SCOPE_HOST("fp:set:alloc");
        arena.resize_bytes(
            total_bytes,
            neural_network->get_device());
    }

    {
        PROFILE_SCOPE_HOST("fp:set:zero");
        arena.setZero();
    }

    memory_debug::record(
        arena.owns_memory() ? "forward" : "forward.aliased",
        "ForwardPropagation::arena",
        arena.owns_memory() ? total_bytes : 0,
        format("batch={},mode={}",
               batch_size,
               is_training ? "training" : "inference"));

    if(transient_block_bytes > 0)
    {
        memory_debug::record(
            "forward.transient_pool",
            "shared_block",
            transient_block_bytes,
            format("batch={}", batch_size));
    }

    if(overlaid_recompute_slots > 0)
    {
        memory_debug::record(
            "forward.training_recomputation",
            "overlaid_scratch_bytes",
            overlaid_scratch_bytes,
            format("batch={},layers={}",
                   batch_size,
                   overlaid_recompute_slots));
    }

    const size_t recomputed_layers =
        ranges::count_if(
            recomputable_slots,
            [](const size_t slot)
            {
                return slot != SIZE_MAX;
            });

    if(!is_training)
    {
        memory_debug::record(
            "forward.inference_pool_analysis",
            "logical_persistent_bytes",
            logical_persistent_bytes,
            format("batch={}", batch_size));

        memory_debug::record(
            "forward.inference_pool_analysis",
            "live_bytes_lower_bound",
            lower_bound_live_bytes,
            format("batch={}", batch_size));

        memory_debug::record(
            "forward.inference_pool_analysis",
            "allocator_fragmentation_overhead",
            fragmentation_bytes,
            format("batch={}", batch_size));

        memory_debug::record(
            "forward.inference_pool_analysis",
            "saved_bytes",
            logical_total_bytes - activation_pool_bytes,
            format("batch={}", batch_size));
    }
    else if(recomputed_layers > 0)
    {
        memory_debug::record(
            "forward.training_recomputation",
            "logical_forward_bytes",
            logical_total_bytes,
            format("batch={}", batch_size));

        memory_debug::record(
            "forward.training_recomputation",
            "allocated_forward_bytes",
            total_bytes,
            format("batch={},layers={}",
                   batch_size,
                   recomputed_layers));

        memory_debug::record(
            "forward.training_recomputation",
            "saved_bytes",
            logical_total_bytes - total_bytes,
            format("batch={}", batch_size));
    }

    device::set_conv_workspace_auto_limit_bytes(
        bind_slots(
            forward_specs,
            slot_offsets,
            transient_slot_offsets));

    capacity_inputs = inputs;
    capacity_slots = slots;
    active_sequence_length = sequence_capacity;

    if(new_shape_policy.final_output_capacity > 0)
        output_window.emplace();

    if(new_shape_policy.sequence_capacity > 0)
        set_active_sequence_length(sequence_capacity);
}

Index ForwardPropagation::bind_slots(
    const vector<vector<TensorSpec>>& forward_specs,
    const vector<vector<Index>>& slot_offsets,
    const vector<vector<Index>>& transient_slot_offsets)
{
    const auto& layers = neural_network->get_layers();
    const auto& source_layers = neural_network->get_source_layers();
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
        inputs[i].resize(sources.size());

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

void ForwardPropagation::recompute_for_backward(Index layer_index)
{
    if (layer_index < 0
        || size_t(layer_index) >= recomputable_slots.size()
        || recomputable_slots[size_t(layer_index)] == SIZE_MAX)
        return;

    neural_network->get_layers()[size_t(layer_index)]
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
    throw_if(!neural_network || neural_network != source.neural_network,
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
    if (!neural_network || layer < 0 || size_t(layer) >= slots.size())
        return {};

    const auto& layer_slots = slots[size_t(layer)];
    if (!layer_slots.empty() && !layer_slots.back().empty())
        return layer_slots.back();

    if (size_t(layer) >= inputs.size() || inputs[size_t(layer)].empty())
        return {};

    TensorView input = inputs[size_t(layer)].front();
    if (!input.empty())
        input = input.reshape(Shape{input.get_shape()[0]}.append(
            neural_network->get_layers()[size_t(layer)]->get_output_shape()));
    return input;
}

TensorView ForwardPropagation::get_last_trainable_layer_outputs() const
{
    return neural_network
        ? get_layer_outputs(neural_network->get_last_trainable_layer_index())
        : TensorView{};
}

// The layer whose record feeds one of `layer`'s inputs, or -1 when that input
// is one of the network's own inputs: raw token ids that nothing has had the
// chance to describe yet.
Index ForwardPropagation::valid_lengths_source(const size_t layer, const size_t input_ordinal) const
{
    if (!neural_network) return -1;

    const auto& source_layers = neural_network->get_source_layers();
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

    // Re-inherited on every pass. The copy used to be taken once and then kept
    // forever, so a second batch with different padding left every layer below
    // the first consumer masking against the first batch's lengths. An
    // Embedding overwrites its own entry as it runs, and its source is a
    // network input, so it is unaffected by re-inheriting here.

    const vector<Index>* source_lengths = input_valid_lengths(layer, 0);
    const int* device_source_lengths = input_device_valid_lengths(layer, 0);
    if (!source_lengths && !device_source_lengths) return;

    // The record travels only as far as the sequence it describes. A layer that
    // pools the sequence away, or reshapes it, ends the record here rather than
    // handing on lengths for something that no longer exists.
    const auto& layers = neural_network->get_layers();
    const Index source = neural_network->get_source_layers()[layer][0];

    const Shape output_shape = layers[layer]->get_output_shape();
    const Shape source_shape = layers[size_t(source)]->get_output_shape();

    if (output_shape.get_rank() < 2 || source_shape.get_rank() < 2) return;
    if (output_shape[0] != source_shape[0]) return;

    if (source_lengths) valid_lengths[layer] = *source_lengths;
    device_valid_lengths[layer] = device_source_lengths;
}

TensorView ForwardPropagation::get_outputs() const
{
    if (!neural_network) return {};

    const Index last = Index(neural_network->get_layers_number()) - 1;
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

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
