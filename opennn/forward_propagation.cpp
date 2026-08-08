//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   F O R W A R D   P R O P A G A T I O N   S O U R C E
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "forward_propagation.h"
#include "neural_network.h"
#include "memory_debug.h"
#include "device_backend.h"
#include "memory_pool.h"
#include "string_utilities.h"

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
                                       const vector<MemoryPoolEntry>* co_planned_lifetimes)
{
    set(new_batch_size, new_neural_network, nullptr, new_mode,
        new_shape_policy, new_inputs_pre_scaled, co_planned_lifetimes);
}

ForwardPropagation::~ForwardPropagation()
{
#ifdef OPENNN_HAS_CUDA
    if (position_pinned) device::deallocate_pinned_host(position_pinned);
#endif
}

void ForwardPropagation::stage_position(cudaStream_t stream)
{
#ifdef OPENNN_HAS_CUDA
    if (!position_pinned)
    {
        position_pinned = device::allocate_pinned_host(Index(sizeof(int)));
        position_device.resize_bytes(Index(sizeof(int)), Device::CUDA);
    }

    *static_cast<int*>(position_pinned) = int(past_length);
    device::copy_async(position_device.data, position_pinned, Index(sizeof(int)),
                       device::CopyKind::HostToDevice, stream);
#else
    (void)stream;
#endif
}

void ForwardPropagation::set(const Index new_batch_size,
                             NeuralNetwork* new_neural_network,
                             Buffer* external_storage,
                             const ForwardPropagationMode new_mode,
                             const InferenceShapePolicy new_shape_policy,
                             const bool new_inputs_pre_scaled,
                             const vector<MemoryPoolEntry>* co_planned_lifetimes)
{
    throw_if(!new_neural_network, "neural network is not set.");
    throw_if(new_mode != ForwardPropagationMode::Inference
             && (new_shape_policy.sequence_capacity > 0
                 || new_shape_policy.final_output_capacity > 0),
             "ForwardPropagation::set: compact capacities are inference-only.");

    throw_if(new_mode != ForwardPropagationMode::Inference
             && !new_shape_policy.retained_output_layers.empty(),
             "ForwardPropagation::set: retained outputs are inference-only; "
             "training keeps every activation alive for the backward pass.");

    reset_cuda_graph();

    batch_size = new_batch_size;
    neural_network = new_neural_network;
    mode = new_mode;
    inference_shape_policy = new_shape_policy;
    inputs_pre_scaled = new_inputs_pre_scaled;

    const auto& layers = neural_network->get_layers();
    const size_t layers_number = layers.size();
    device_input_buffers.clear();
    device_input_views.clear();
    host_bf16_input_scratch.clear();
    passthrough_overrides.clear();
    input_views.resize(layers_number);
    forward_slots.resize(layers_number);

    auto forward_specs = neural_network->get_forward_specs(batch_size);

    throw_if(forward_specs.size() != layers_number,
             "ForwardPropagation::set: forward specs size ({}) does not match layers number ({}).",
                    forward_specs.size(), layers_number);

    const auto& source_layers = neural_network->get_source_layers();

    throw_if(source_layers.size() != layers_number,
             "ForwardPropagation::set: source layers size ({}) does not match layers number ({}).",
                    source_layers.size(), layers_number);

    if (mode == ForwardPropagationMode::Training && inputs_pre_scaled)
        for (size_t i = 0;
             i < layers_number && layers[i]->get_type() == LayerType::Scaling;
             ++i)

            forward_specs[i].clear();

    const Shape model_input_shape = neural_network->get_input_shape();
    const Index model_sequence_capacity =
        model_input_shape.empty() ? Index(0) : model_input_shape[0];
    sequence_capacity = new_shape_policy.sequence_capacity > 0
        ? new_shape_policy.sequence_capacity
        : model_sequence_capacity;
    throw_if(new_shape_policy.sequence_capacity > model_sequence_capacity,
             "ForwardPropagation::set: sequence capacity {} exceeds the "
             "network capacity {}.",
             new_shape_policy.sequence_capacity, model_sequence_capacity);

    if (new_shape_policy.sequence_capacity > 0)
        for (auto& layer_specs : forward_specs)
            for (TensorSpec& spec : layer_specs)
                if (spec.shape.rank >= 2
                    && spec.shape[1] == model_sequence_capacity)
                    spec.shape[1] = sequence_capacity;

    final_output_layer = -1;
    for (Index i = Index(layers_number) - 1; i >= 0; --i)
        if (!forward_specs[size_t(i)].empty())
        {
            final_output_layer = i;
            break;
        }

    final_output_capacity = new_shape_policy.final_output_capacity > 0
        ? new_shape_policy.final_output_capacity
        : sequence_capacity;
    throw_if(new_shape_policy.final_output_capacity > 0
             && new_shape_policy.sequence_capacity <= 0,
             "ForwardPropagation::set: final_output_capacity requires an "
             "explicit sequence_capacity.");
    throw_if(final_output_capacity > sequence_capacity,
             "ForwardPropagation::set: final output capacity {} exceeds "
             "sequence capacity {}.",
             final_output_capacity, sequence_capacity);

    if (new_shape_policy.final_output_capacity > 0
        && final_output_layer >= 0)
    {
        TensorSpec& output_spec =
            forward_specs[size_t(final_output_layer)].back();
        throw_if(output_spec.shape.rank < 2
                 || output_spec.shape[1] != sequence_capacity,
                 "ForwardPropagation::set: final output does not expose a "
                 "sequence dimension compatible with compact inference.");
        output_spec.shape[1] = final_output_capacity;
    }

    const bool is_training = mode == ForwardPropagationMode::Training;

    recomputable_forward_slots.assign(layers_number, SIZE_MAX);
    if (is_training && neural_network->get_training_activation_recomputation())
        ranges::transform(layers, recomputable_forward_slots.begin(),
                          [](const auto& layer) { return layer->get_recomputable_forward_slot(); });

    if (!is_training)
        for (size_t i = 0; i < layers_number; ++i)
            for (size_t j = 0; j < forward_specs[i].size(); ++j)
                if (layers[i]->get_forward_slot_kind(j) == ForwardSlotKind::TrainingOnly)
                    forward_specs[i][j] = {};

    const auto is_transient_slot = [&](size_t layer, size_t slot)
    {
        return is_training
            && (layers[layer]->get_forward_slot_kind(slot) == ForwardSlotKind::Transient
                || recomputable_forward_slots[layer] == slot);
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
        ranges::count_if(output_release_steps,
                         [](const Index step) { return step >= 0; });

    const bool recompute_overlay_allowed =
        neural_network->supports_compact_cnn_memory_layout()
        || early_release_outputs > 0;

    vector<vector<Index>> slot_offsets(layers_number);
    vector<vector<Index>> transient_slot_offsets(layers_number);
    Index logical_total_bytes = 0;
    Index logical_persistent_bytes = 0;

    for (size_t i = 0; i < layers_number; ++i)
    {
        slot_offsets[i].assign(forward_specs[i].size(), Index(-1));
        transient_slot_offsets[i].assign(forward_specs[i].size(), Index(-1));
        throw_if(recomputable_forward_slots[i] != SIZE_MAX
                 && recomputable_forward_slots[i] >= forward_specs[i].size(),
                 "ForwardPropagation::set: invalid recomputable slot for layer {}.",
                 i);

        for (size_t j = 0; j < forward_specs[i].size(); ++j)
        {
            const auto& spec = forward_specs[i][j];
            if (spec.shape.empty()) continue;

            const Index bytes = get_aligned_bytes(spec);
            logical_total_bytes += bytes;
            if (is_transient_slot(i, j))
                throw_if(j + 1 == forward_specs[i].size(),
                         "ForwardPropagation::set: a layer output cannot be a transient slot.");
            else
                logical_persistent_bytes += bytes;
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
        for (size_t i = 0; i < layers_number; ++i)
        {
            Index layer_bytes = 0;
            for (size_t j = 0; j < forward_specs[i].size(); ++j)
                if (is_transient_slot(i, j)
                    && !forward_specs[i][j].shape.empty()
                    && transient_slot_offsets[i][j] < 0)
                {
                    transient_slot_offsets[i][j] =
                        activation_pool_bytes + layer_bytes;
                    layer_bytes += get_aligned_bytes(forward_specs[i][j]);
                }
            block_bytes = max(block_bytes, layer_bytes);
        }
        return block_bytes;
    };

    vector<pair<size_t, size_t>> pooled_slots;
    vector<MemoryPoolEntry> pooled_lifetimes;
    const auto collect_pooled_slots = [&](auto&& last_step_for)
    {
        for (size_t i = 0; i < layers_number; ++i)
            for (size_t j = 0; j < forward_specs[i].size(); ++j)
            {
                const TensorSpec& spec = forward_specs[i][j];
                if (spec.shape.empty() || is_transient_slot(i, j)) continue;

                const bool is_output = j + 1 == forward_specs[i].size();
                pooled_slots.push_back({i, j});
                pooled_lifetimes.push_back({get_aligned_bytes(spec),
                                            Index(i),
                                            last_step_for(i, is_output)});
            }
    };

    const auto apply_pool_plan = [&](const MemoryPoolPlan& plan)
    {
        for (size_t i = 0; i < pooled_slots.size(); ++i)
            slot_offsets[pooled_slots[i].first][pooled_slots[i].second] =
                plan.byte_offsets[i];

        activation_pool_bytes  = plan.peak_bytes;
        lower_bound_live_bytes = plan.lower_bound_live_bytes;
        fragmentation_bytes    = plan.fragmentation_bytes();
    };

    if (is_training)
    {

        const Index backward_base = Index(2 * layers_number - 1);
        collect_pooled_slots([&](size_t i, bool is_output)
        {
            return is_output && output_release_steps[i] >= 0
                ? output_release_steps[i]
                : backward_base - Index(i);
        });

        memory_debug::record_pool_lifetimes(
            "forward", pooled_lifetimes,
            format("layers={},batch={}", layers_number, batch_size));

        co_planned_block = {};
        const size_t forward_entry_count = pooled_lifetimes.size();
        if (co_planned_lifetimes)
            for (const MemoryPoolEntry& entry : *co_planned_lifetimes)
            {
                co_planned_block.bytes += entry.bytes;
                pooled_lifetimes.push_back(entry);
            }

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
        const MemoryPoolPlan persistent_plan = plan_memory_pool(
            pooled_lifetimes,
            early_release_outputs > 0
                ? MemoryPoolStrategy::Compact
                : MemoryPoolStrategy::Chronological);

        apply_pool_plan(persistent_plan);

        if (co_planned_lifetimes)
        {
            co_planned_block.offsets.assign(
                persistent_plan.byte_offsets.begin() + forward_entry_count,
                persistent_plan.byte_offsets.end());
            co_planned_block.valid = true;
            memory_debug::record("forward.joint_plan", "delta_entries_in_arena",
                                 co_planned_block.bytes,
                                 format("batch={},entries={}", batch_size,
                                        co_planned_lifetimes->size()));
        }

        for (size_t i = 0; i < layers_number; ++i)
        {
            const size_t slot = recomputable_forward_slots[i];
            if (slot == SIZE_MAX || forward_specs[i][slot].shape.empty())
                continue;

            const Index bytes = get_aligned_bytes(forward_specs[i][slot]);
            const Index backward_step = backward_base - Index(i);
            const Index overlay_offset = recompute_overlay_allowed
                ? find_memory_pool_overlay(pooled_lifetimes, persistent_plan,
                                           bytes, Index(i), backward_step)
                : Index(-1);

            if (overlay_offset >= 0)
            {
                transient_slot_offsets[i][slot] = overlay_offset;
                ++overlaid_recompute_slots;
                overlaid_scratch_bytes += bytes;
            }

            memory_debug::record(
                "forward.recompute_entry", format("{}:{}", i, slot), bytes,
                format("first={},second={},overlaid={}",
                       i, backward_step, overlay_offset >= 0 ? 1 : 0));
        }

        transient_block_bytes = place_transient_slots();

        if (early_release_outputs > 0)
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
                format("batch={},lower_bound_mib={:.2f}",
                       batch_size,
                       double(lower_bound_live_bytes) / (1024.0 * 1024.0)));
        }
    }
    else
    {
        const Index final_step = layers_number == 0 ? 0 : Index(layers_number - 1);
        vector<Index> last_consumers(layers_number);
        vector<bool> has_consumers(layers_number, false);
        iota(last_consumers.begin(), last_consumers.end(), Index(0));

        for (size_t consumer = 0; consumer < layers_number; ++consumer)
            for (const Index source_layer : source_layers[consumer])
            {
                const Index producer = resolve_producer(
                    forward_specs, source_layers, source_layer);
                if (producer < 0) continue;

                has_consumers[size_t(producer)] = true;
                last_consumers[size_t(producer)] =
                    max(last_consumers[size_t(producer)], Index(consumer));
            }

        vector<bool> externally_observable(layers_number, false);
        for (size_t i = 0; i < layers_number; ++i)
            if (!has_consumers[i]
                || is_one_of(layers[i]->get_type(), LayerType::Detection, LayerType::DetectionV8))
                externally_observable[i] = true;

        const auto mark_resolved_output = [&](Index layer_index)
        {
            if (layer_index < 0 || size_t(layer_index) >= layers_number) return;

            const Index producer =
                resolve_producer(forward_specs, source_layers, layer_index);
            if (producer >= 0) externally_observable[size_t(producer)] = true;
        };

        mark_resolved_output(Index(layers_number) - 1);
        mark_resolved_output(neural_network->get_last_trainable_layer_index());

        for (const Index retained : inference_shape_policy.retained_output_layers)
        {
            throw_if(retained < 0 || size_t(retained) >= layers_number,
                     "ForwardPropagation::set: retained output layer {} is out "
                     "of range (network has {} layers).",
                     retained, layers_number);
            mark_resolved_output(retained);
        }

        collect_pooled_slots([&](size_t i, bool is_output)
        {
            if (!is_output) return Index(i);
            return externally_observable[i] ? final_step : last_consumers[i];
        });

        apply_pool_plan(plan_memory_pool(pooled_lifetimes, MemoryPoolStrategy::Compact));

    }

    const Index total_bytes = activation_pool_bytes + transient_block_bytes;

    if (external_storage
        && external_storage->device_type == neural_network->get_device()
        && external_storage->bytes >= total_bytes)
        data.set_view(external_storage->data, total_bytes, external_storage->device_type);
    else
        data.resize_bytes(total_bytes, neural_network->get_device());
    data.setZero();

    memory_debug::record(data.owns ? "forward" : "forward.aliased",
                         "ForwardPropagation::data",
                         data.owns ? total_bytes : 0,
                         format("batch={},mode={}",
                                batch_size,
                                is_training ? "training" : "inference"));
    if (transient_block_bytes > 0)
        memory_debug::record("forward.transient_pool", "shared_block",
                             transient_block_bytes,
                             format("batch={}", batch_size));
    if (overlaid_recompute_slots > 0)
        memory_debug::record("forward.training_recomputation",
                             "overlaid_scratch_bytes",
                             overlaid_scratch_bytes,
                             format("batch={},layers={}",
                                    batch_size,
                                    overlaid_recompute_slots));
    if (!is_training)
    {
        memory_debug::record("forward.inference_pool_analysis", "logical_persistent_bytes",
                             logical_persistent_bytes,
                             format("batch={}", batch_size));
        memory_debug::record("forward.inference_pool_analysis", "live_bytes_lower_bound",
                             lower_bound_live_bytes,
                             format("batch={}", batch_size));
        memory_debug::record("forward.inference_pool_analysis", "allocator_fragmentation_overhead",
                             fragmentation_bytes,
                             format("batch={}", batch_size));
        memory_debug::record("forward.inference_pool_analysis", "saved_bytes",
                             logical_total_bytes - activation_pool_bytes,
                             format("batch={}", batch_size));
    }
    else if (ranges::any_of(recomputable_forward_slots,
                            [](size_t slot) { return slot != SIZE_MAX; }))
    {
        const size_t recomputed_layers = ranges::count_if(
            recomputable_forward_slots,
            [](size_t slot) { return slot != SIZE_MAX; });
        memory_debug::record("forward.training_recomputation",
                             "logical_forward_bytes",
                             logical_total_bytes,
                             format("batch={}", batch_size));
        memory_debug::record("forward.training_recomputation",
                             "allocated_forward_bytes",
                             total_bytes,
                             format("batch={},layers={}",
                                    batch_size,
                                    recomputed_layers));
        memory_debug::record("forward.training_recomputation",
                             "saved_bytes",
                             logical_total_bytes - total_bytes,
                             format("batch={}", batch_size));
    }

    device::set_conv_workspace_auto_limit_bytes(
        bind_slot_views(forward_specs, slot_offsets, transient_slot_offsets));

    capacity_input_views = input_views;
    capacity_forward_slots = forward_slots;
    active_sequence_length = sequence_capacity;

    if (new_shape_policy.sequence_capacity > 0)
    {
        set_active_sequence_length(sequence_capacity);
        const Index count = min(final_output_capacity, sequence_capacity);
        set_output_sequence_window(sequence_capacity - count, count);
    }
}

Index ForwardPropagation::bind_slot_views(
    const vector<vector<TensorSpec>>& forward_specs,
    const vector<vector<Index>>& slot_offsets,
    const vector<vector<Index>>& transient_slot_offsets)
{
    const auto& layers = neural_network->get_layers();
    const auto& source_layers = neural_network->get_source_layers();
    uint8_t* const pool_base = data.as<uint8_t>();

    Index max_layer_bytes = 0;

    for (size_t i = 0; i < forward_specs.size(); ++i)
    {
        const auto& specs = forward_specs[i];
        max_layer_bytes = max(max_layer_bytes, get_aligned_bytes(specs));

        forward_slots[i].assign(specs.size() + 1, TensorView{});

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

            forward_slots[i][j + 1] =
                TensorView(pool_base + offset, shape, dtype, data.device_type);

            if (!transient) layer_logical_bytes += get_aligned_bytes(specs[j]);
        }

        if (layer_logical_bytes > 0)
            memory_debug::record("forward.layer",
                                 format("{}:{}", i, layers[i]->get_label()),
                                 layer_logical_bytes,
                                 format("batch={}", batch_size));

        const vector<Index>& sources = source_layers[i];
        input_views[i].resize(sources.size());

        for (size_t j = 0; j < sources.size(); ++j)
        {
            const Index source_layer = sources[j];
            if (source_layer < 0) continue;

            if (!forward_specs[source_layer].empty())
            {
                input_views[i][j] = forward_slots[source_layer].back();
                continue;
            }

            const Index resolved =
                resolve_producer(forward_specs, source_layers, source_layer);

            if (resolved < 0)
            {
                passthrough_overrides.emplace_back(i, j, size_t(-resolved - 1));
                continue;
            }

            TensorView view = forward_slots[resolved].back();
            if (!view.empty())
                view.shape = Shape{view.shape[0]}
                    .append(layers[source_layer]->get_output_shape());
            input_views[i][j] = view;
        }
    }

    return max_layer_bytes;
}

void ForwardPropagation::recompute_for_backward(Index layer_index)
{
    if (layer_index < 0
        || size_t(layer_index) >= recomputable_forward_slots.size()
        || recomputable_forward_slots[size_t(layer_index)] == SIZE_MAX)
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

    input_views = capacity_input_views;
    forward_slots = capacity_forward_slots;
    active_sequence_length = length;

    const auto shrink_sequence = [this, length](TensorView& view)
    {
        if (!view.empty() && view.get_rank() >= 2
            && view.shape[1] == sequence_capacity)
            view.shape[1] = length;
    };

    for (auto& layer_slots : forward_slots)
        for (auto& slot : layer_slots) shrink_sequence(slot);

    for (auto& layer_inputs : input_views)
        for (auto& view : layer_inputs) shrink_sequence(view);

    if (inference_shape_policy.final_output_capacity > 0)
    {
        const Index count = min(final_output_capacity, length);
        set_output_sequence_window(length - count, count);
    }
}

void ForwardPropagation::set_output_sequence_window(Index start, Index count)
{
    throw_if(inference_shape_policy.final_output_capacity <= 0,
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
             || size_t(final_output_layer) >= input_views.size()
             || input_views[size_t(final_output_layer)].empty(),
             "ForwardPropagation::set_output_sequence_window: final layer has "
             "no input view.");

    reset_cuda_graph();

    TensorView& input = input_views[size_t(final_output_layer)].front();
    const TensorView& capacity_input =
        capacity_input_views[size_t(final_output_layer)].front();
    throw_if(capacity_input.empty() || capacity_input.get_rank() < 2,
             "ForwardPropagation::set_output_sequence_window: final layer "
             "input is not sequence-shaped.");

    const Index row_bytes =
        capacity_input.shape.size() / capacity_input.shape[0]
        / capacity_input.shape[1] * type_bytes(capacity_input.type);

    output_window_start = start;
    output_window_count = count;

    input = capacity_input;
    input.shape[1] = count;

    if (batch_size == 1)
    {
        output_window_input.resize_bytes(0, capacity_input.device);
        input.data = static_cast<char*>(capacity_input.data) + start * row_bytes;
    }
    else
    {
        output_window_input.resize_bytes(batch_size * count * row_bytes,
                                         capacity_input.device);
        input.data = output_window_input.data;
    }

    TensorView& output = forward_slots[size_t(final_output_layer)].back();
    output = capacity_forward_slots[size_t(final_output_layer)].back();
    output.shape[1] = count;
}

void ForwardPropagation::gather_output_window()
{
    if (output_window_input.empty()) return;

    const TensorView& capacity_input =
        capacity_input_views[size_t(final_output_layer)].front();

    const Index sequence = capacity_input.shape[1];
    const Index row_bytes = capacity_input.shape.size() / capacity_input.shape[0]
                          / sequence * type_bytes(capacity_input.type);
    const Index window_bytes = output_window_count * row_bytes;

    for (Index sample = 0; sample < batch_size; ++sample)
        device::copy_async(
            static_cast<char*>(output_window_input.data) + sample * window_bytes,
            static_cast<const char*>(capacity_input.data)
                + (sample * sequence + output_window_start) * row_bytes,
            window_bytes,
            capacity_input.device, capacity_input.device,
            device::get_compute_stream());
}

static TensorView get_layer_outputs(const ForwardPropagation& propagation,
                                    const Index layer)
{
    if (!propagation.neural_network || layer < 0
        || size_t(layer) >= propagation.forward_slots.size())
        return {};

    const auto& slots = propagation.forward_slots[size_t(layer)];
    if (!slots.empty() && !slots.back().empty()) return slots.back();

    if (size_t(layer) >= propagation.input_views.size()
        || propagation.input_views[size_t(layer)].empty())
        return {};

    TensorView input = propagation.input_views[size_t(layer)].front();
    if (!input.empty())
        input.shape = Shape{input.shape[0]}.append(
            propagation.neural_network->get_layers()[size_t(layer)]
                ->get_output_shape());
    return input;
}

TensorView ForwardPropagation::get_last_trainable_layer_outputs() const
{
    return neural_network
        ? get_layer_outputs(*this,
                            neural_network->get_last_trainable_layer_index())
        : TensorView{};
}

TensorView ForwardPropagation::get_outputs() const
{
    if (!neural_network) return {};

    const Index last = Index(neural_network->get_layers_number()) - 1;
    TensorView output = get_layer_outputs(*this, last);
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
    for (size_t i = 0; i < inference_graph_workspaces.size(); ++i)
    {
        Buffer& buffer = inference_graph_workspaces[i];
        const Index before = buffer.bytes;
        buffer.grow_to(inference_graph_workspace_requirements[i]);
        if (buffer.bytes > before)
            memory_debug::record("forward.graph_workspace",
                                 device::graph_workspace_labels[i],
                                 buffer.bytes - before,
                                 format("batch={}", batch_size));
    }
}

bool ForwardPropagation::cuda_graph_workspaces_need_growth() const noexcept
{
    for (size_t i = 0; i < inference_graph_workspaces.size(); ++i)
        if (inference_graph_workspace_requirements[i]
                > inference_graph_workspaces[i].bytes)
            return true;

    return false;
}

device::GraphWorkspaceViews
ForwardPropagation::get_cuda_graph_workspace_views() const noexcept
{
    device::GraphWorkspaceViews views{};

    for (size_t i = 0; i < views.size(); ++i)
        views[i] = {inference_graph_workspaces[i].data,
                    inference_graph_workspaces[i].bytes};

    return views;
}

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
