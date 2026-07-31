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

ForwardPropagation::ForwardPropagation(const Index new_batch_size,
                                       NeuralNetwork* new_neural_network,
                                       const ForwardPropagationMode new_mode,
                                       const InferenceShapePolicy new_shape_policy)
{
    set(new_batch_size, new_neural_network, nullptr, new_mode,
        new_shape_policy);
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

void ForwardPropagation::set(const Index new_batch_size, NeuralNetwork* new_neural_network,
                             Buffer* external_storage,
                             const ForwardPropagationMode new_mode,
                             const InferenceShapePolicy new_shape_policy)
{
    throw_if(!new_neural_network, "neural network is not set.");
    throw_if(new_mode != ForwardPropagationMode::Inference
             && (new_shape_policy.sequence_capacity > 0
                 || new_shape_policy.final_output_capacity > 0),
             "ForwardPropagation::set: compact capacities are inference-only.");

    reset_cuda_graph();

    batch_size = new_batch_size;
    neural_network = new_neural_network;
    mode = new_mode;
    inference_shape_policy = new_shape_policy;

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

    vector<vector<Index>> slot_offsets(layers_number);
    Index logical_persistent_bytes = 0;
    Index transient_block_bytes = 0;

    for (size_t i = 0; i < layers_number; ++i)
    {
        slot_offsets[i].assign(forward_specs[i].size(), Index(-1));
        for (size_t j = 0; j < forward_specs[i].size(); ++j)
        {
            const auto& spec = forward_specs[i][j];
            if (spec.shape.empty()) continue;

            if (layers[i]->is_forward_slot_transient(j))
            {
                throw_if(j + 1 == forward_specs[i].size(),
                         "ForwardPropagation::set: a layer output cannot be a transient slot.");
                transient_block_bytes = max(transient_block_bytes, get_aligned_bytes(spec));
            }
            else
                logical_persistent_bytes += get_aligned_bytes(spec);
        }
    }

    Index activation_pool_bytes = logical_persistent_bytes;
    Index lower_bound_live_bytes = logical_persistent_bytes;
    Index fragmentation_bytes = 0;

    if (mode == ForwardPropagationMode::Training)
    {
        Index cursor_offset = 0;
        for (size_t i = 0; i < layers_number; ++i)
            for (size_t j = 0; j < forward_specs[i].size(); ++j)
            {
                const TensorSpec& spec = forward_specs[i][j];
                if (spec.shape.empty() || layers[i]->is_forward_slot_transient(j))
                    continue;

                slot_offsets[i][j] = cursor_offset;
                cursor_offset += get_aligned_bytes(spec);
            }
    }
    else
    {
        struct ForwardEntry
        {
            size_t layer = 0;
            size_t slot = 0;
            MemoryPoolEntry lifetime;
        };

        const Index final_step = layers_number == 0 ? 0 : Index(layers_number - 1);
        vector<Index> last_consumers(layers_number);
        vector<bool> has_consumers(layers_number, false);
        for (size_t i = 0; i < layers_number; ++i)
            last_consumers[i] = Index(i);

        for (size_t consumer = 0; consumer < layers_number; ++consumer)
            for (const Index source_layer : source_layers[consumer])
            {
                const Index producer = resolve_producer(forward_specs, source_layers, source_layer);
                if (producer < 0) continue;

                has_consumers[size_t(producer)] = true;
                last_consumers[size_t(producer)] =
                    max(last_consumers[size_t(producer)], Index(consumer));
            }

        vector<bool> externally_observable(layers_number, false);
        for (size_t i = 0; i < layers_number; ++i)
            if (!has_consumers[i] || layers[i]->get_type() == LayerType::Detection)
                externally_observable[i] = true;

        const auto mark_resolved_output = [&](Index layer_index)
        {
            if (layer_index < 0 || size_t(layer_index) >= layers_number) return;

            Index producer = layer_index;
            if (forward_specs[size_t(producer)].empty())
            {
                const auto& sources = source_layers[size_t(producer)];
                producer = sources.empty() ? Index(-1) : resolve_producer(forward_specs, source_layers, sources.front());
            }

            if (producer >= 0) externally_observable[size_t(producer)] = true;
        };

        mark_resolved_output(Index(layers_number) - 1);
        mark_resolved_output(neural_network->get_last_trainable_layer_index());

        vector<ForwardEntry> forward_entries;
        vector<MemoryPoolEntry> lifetime_entries;

        for (size_t i = 0; i < layers_number; ++i)
            for (size_t j = 0; j < forward_specs[i].size(); ++j)
            {
                const TensorSpec& spec = forward_specs[i][j];
                if (spec.shape.empty() || layers[i]->is_forward_slot_transient(j))
                    continue;

                const bool is_output = j + 1 == forward_specs[i].size();
                Index last_step = Index(i);
                if (is_output)
                {
                    last_step = last_consumers[i];
                    if (externally_observable[i]) last_step = final_step;
                }

                const MemoryPoolEntry lifetime{get_aligned_bytes(spec),
                                               Index(i),
                                               last_step};
                forward_entries.push_back({i, j, lifetime});
                lifetime_entries.push_back(lifetime);
            }

        const MemoryPoolPlan pool_plan = plan_memory_pool(lifetime_entries);
        for (size_t entry_index = 0; entry_index < forward_entries.size(); ++entry_index)
        {
            const ForwardEntry& entry = forward_entries[entry_index];
            slot_offsets[entry.layer][entry.slot] = pool_plan.byte_offsets[entry_index];
        }

        activation_pool_bytes = pool_plan.peak_bytes;
        lower_bound_live_bytes = pool_plan.lower_bound_live_bytes;
        fragmentation_bytes = pool_plan.fragmentation_bytes();
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
                                mode == ForwardPropagationMode::Training
                                    ? "training"
                                    : "inference"));
    if (transient_block_bytes > 0)
        memory_debug::record("forward.transient_pool", "shared_block",
                             transient_block_bytes,
                             format("batch={}", batch_size));
    if (mode == ForwardPropagationMode::Inference)
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
                             logical_persistent_bytes - activation_pool_bytes,
                             format("batch={}", batch_size));
    }

    uint8_t* const pool_base = data.as<uint8_t>();
    uint8_t* const transient_base = pool_base
        ? pool_base + activation_pool_bytes
        : nullptr;
    Index max_layer_bytes = 0;
    for (size_t i = 0; i < layers_number; ++i)
    {
        const auto& specs = forward_specs[i];

        const Index layer_bytes = get_aligned_bytes(specs);
        max_layer_bytes = std::max(max_layer_bytes, layer_bytes);

        forward_slots[i].assign(specs.size() + 1, TensorView{});

        Index layer_logical_bytes = 0;
        for (size_t j = 0; j < specs.size(); ++j)
        {
            const auto& [shape, dtype] = specs[j];
            if (shape.empty()) continue;

            if (layers[i]->is_forward_slot_transient(j))
            {
                forward_slots[i][j + 1] = TensorView(transient_base, shape, dtype, data.device_type);
                continue;
            }

            throw_if(slot_offsets[i][j] < 0,
                     "ForwardPropagation::set: missing memory-pool offset for layer {} slot {}.",
                            i, j);
            forward_slots[i][j + 1] =
                TensorView(pool_base + slot_offsets[i][j], shape, dtype, data.device_type);
            layer_logical_bytes += get_aligned_bytes(specs[j]);
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

            const Index resolved = resolve_producer(forward_specs, source_layers, source_layer);

            if (resolved >= 0)
            {

                TensorView view = forward_slots[resolved].back();
                if (!view.empty())
                    view.shape = Shape{view.shape[0]}
                        .append(layers[source_layer]->get_output_shape());
                input_views[i][j] = view;
            }
            else
                passthrough_overrides.emplace_back(i, j, size_t(-resolved - 1));
        }
    }

    device::set_conv_workspace_auto_limit_bytes(max_layer_bytes);

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
    throw_if(batch_size != 1,
             "ForwardPropagation::set_output_sequence_window currently "
             "supports batch size 1.");
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

    const Index row_elements =
        capacity_input.shape.size() / capacity_input.shape[0]
        / capacity_input.shape[1];
    input = capacity_input;
    input.data = static_cast<char*>(capacity_input.data)
        + start * row_elements * type_bytes(capacity_input.type);
    input.shape[1] = count;

    TensorView& output = forward_slots[size_t(final_output_layer)].back();
    output = capacity_forward_slots[size_t(final_output_layer)].back();
    output.shape[1] = count;
}

TensorView ForwardPropagation::get_last_trainable_layer_outputs() const
{
    if (!neural_network) return {};

    const Index layer_index = neural_network->get_last_trainable_layer_index();

    if (layer_index < 0
        || size_t(layer_index) >= forward_slots.size()
        || forward_slots[layer_index].size() < 1)
        return {};

    const TensorView& v = forward_slots[layer_index].back();
    if (!v.empty()) return v;

    if (size_t(layer_index) < input_views.size() && !input_views[layer_index].empty())
    {
        TensorView input_view = input_views[layer_index].front();
        if (!input_view.empty())
        {
            input_view.shape = Shape{input_view.shape[0]}
                .append(neural_network->get_layers()[layer_index]->get_output_shape());
            return input_view;
        }
    }

    return {};
}

TensorView ForwardPropagation::get_outputs() const
{
    if (!neural_network) return {};

    const Index last = Index(neural_network->get_layers_number()) - 1;

    if (last >= 0
        && size_t(last) < forward_slots.size()
        && forward_slots[last].size() > 1)
    {
        const TensorView& v = forward_slots[last].back();
        if (!v.empty()) return v;
    }

    if (last >= 0 && size_t(last) < input_views.size() && !input_views[last].empty())
    {
        TensorView input_view = input_views[last].front();
        if (!input_view.empty())
        {
            input_view.shape = Shape{input_view.shape[0]}
                .append(neural_network->get_layers()[last]->get_output_shape());
            return input_view;
        }
    }

    return get_last_trainable_layer_outputs();
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
    const auto grow_and_record = [&](Buffer& buffer, Index required_bytes,
                                     const char* label)
    {
        const Index before = buffer.bytes;
        buffer.grow_to(required_bytes);
        if (buffer.bytes > before)
            memory_debug::record("forward.graph_workspace", label,
                                 buffer.bytes - before,
                                 format("batch={}", batch_size));
    };

    grow_and_record(inference_graph_shared_scratch,
                    inference_graph_workspace_requirements.shared_scratch,
                    "shared_scratch");
    grow_and_record(inference_graph_bf16_input,
                    inference_graph_workspace_requirements.bf16_input,
                    "bf16_input");
    grow_and_record(inference_graph_bf16_gradient,
                    inference_graph_workspace_requirements.bf16_gradient,
                    "bf16_gradient");
    grow_and_record(inference_graph_bf16_to_fp32,
                    inference_graph_workspace_requirements.bf16_to_fp32,
                    "bf16_to_fp32");
    grow_and_record(inference_graph_int8_dequant,
                    inference_graph_workspace_requirements.int8_dequant,
                    "int8_dequant");
}

bool ForwardPropagation::cuda_graph_workspaces_need_growth() const noexcept
{
    return inference_graph_workspace_requirements.shared_scratch
               > inference_graph_shared_scratch.bytes
        || inference_graph_workspace_requirements.bf16_input
               > inference_graph_bf16_input.bytes
        || inference_graph_workspace_requirements.bf16_gradient
               > inference_graph_bf16_gradient.bytes
        || inference_graph_workspace_requirements.bf16_to_fp32
               > inference_graph_bf16_to_fp32.bytes
        || inference_graph_workspace_requirements.int8_dequant
               > inference_graph_int8_dequant.bytes;
}

device::GraphWorkspaceViews
ForwardPropagation::get_cuda_graph_workspace_views() const noexcept
{
    return {
        inference_graph_shared_scratch.data,
        inference_graph_shared_scratch.bytes,
        inference_graph_bf16_input.data,
        inference_graph_bf16_input.bytes,
        inference_graph_bf16_gradient.data,
        inference_graph_bf16_gradient.bytes,
        inference_graph_bf16_to_fp32.data,
        inference_graph_bf16_to_fp32.bytes,
        inference_graph_int8_dequant.data,
        inference_graph_int8_dequant.bytes
    };
}

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
