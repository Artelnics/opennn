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

namespace opennn
{

ForwardPropagation::ForwardPropagation(const Index new_batch_size, NeuralNetwork* new_neural_network)
{
    set(new_batch_size, new_neural_network);
}

void ForwardPropagation::set(const Index new_batch_size, NeuralNetwork* new_neural_network,
                             Buffer* external_storage)
{
    throw_if(!new_neural_network, "neural network is not set.");

    // Re-allocating the activation arena invalidates every pointer a captured
    // inference graph baked in.
    reset_cuda_graph();

    batch_size = new_batch_size;
    neural_network = new_neural_network;

    const auto& layers = neural_network->get_layers();
    const size_t layers_number = layers.size();
    device_input_buffers.clear();
    passthrough_overrides.clear();
    input_views.resize(layers_number);
    forward_slots.resize(layers_number);

    const auto forward_specs = neural_network->get_forward_specs(batch_size);

    throw_if(forward_specs.size() != layers_number,
             format("ForwardPropagation::set: forward specs size ({}) does not match layers number ({}).",
                    forward_specs.size(), layers_number));

    const auto& source_layers = neural_network->get_source_layers();

    throw_if(source_layers.size() != layers_number,
             format("ForwardPropagation::set: source layers size ({}) does not match layers number ({}).",
                    source_layers.size(), layers_number));

    // Transient slots (e.g. attention head split/merge staging) never carry
    // data across an operator invocation, and forward/backward execute the
    // operators serially, so every transient slot across all layers can view
    // ONE shared max-sized block at the arena tail -- the same reasoning as
    // the shared cuDNN workspace. Outputs can never be transient: consumers
    // read them through input_views in forward and backward.
    Index persistent_bytes = 0;
    Index transient_block_bytes = 0;

    for (size_t i = 0; i < layers_number; ++i)
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
                persistent_bytes += get_aligned_bytes(spec);
        }

    const Index total_bytes = persistent_bytes + transient_block_bytes;

    // Overlay this propagation on an external buffer (e.g. the training
    // ForwardPropagation's, when validation is temporally disjoint) when it
    // fits and lives on the same device; otherwise own the allocation.
    if (external_storage
        && external_storage->device_type == neural_network->get_device()
        && external_storage->bytes >= total_bytes)
        data.set_view(external_storage->data, total_bytes, external_storage->device_type);
    else
        data.resize_bytes(total_bytes, neural_network->get_device());
    data.setZero();
    // Only owned storage counts as new VRAM; an aliased view reuses an existing
    // buffer, so report it separately (zero new bytes) to keep the profiler honest.
    memory_debug::record(data.owns ? "forward" : "forward.aliased",
                         "ForwardPropagation::data",
                         data.owns ? total_bytes : 0,
                         format("batch={}", batch_size));
    if (transient_block_bytes > 0)
        memory_debug::record("forward.transient_pool", "shared_block",
                             transient_block_bytes,
                             format("batch={}", batch_size));

    uint8_t* const transient_base = data.as<uint8_t>() + persistent_bytes;
    uint8_t* cursor = data.as<uint8_t>();
    Index max_layer_bytes = 0;
    for (size_t i = 0; i < layers_number; ++i)
    {
        const auto& specs = forward_specs[i];
        // Full spec bytes (transients included) -- this feeds the AUTO conv
        // workspace limit below, which sizes actual per-layer work.
        const Index layer_bytes = get_aligned_bytes(specs);
        max_layer_bytes = std::max(max_layer_bytes, layer_bytes);

        forward_slots[i].assign(specs.size() + 1, TensorView{});

        Index layer_persistent_bytes = 0;
        for (size_t j = 0; j < specs.size(); ++j)
        {
            const auto& [shape, dtype] = specs[j];
            if (shape.empty()) continue;

            if (layers[i]->is_forward_slot_transient(j))
            {
                forward_slots[i][j + 1] = TensorView(transient_base, shape, dtype, data.device_type);
                continue;
            }

            forward_slots[i][j + 1] = TensorView(cursor, shape, dtype, data.device_type);
            cursor += get_aligned_bytes(specs[j]);
            layer_persistent_bytes += get_aligned_bytes(specs[j]);
        }

        if (layer_persistent_bytes > 0)
            memory_debug::record("forward.layer",
                                 format("{}:{}", i, layers[i]->get_label()),
                                 layer_persistent_bytes,
                                 format("batch={}", batch_size));

        // validate_source_indices guarantees source < i, so its slots are already assigned.
        const vector<Index>& sources = source_layers[i];
        input_views[i].resize(sources.size());

        for (size_t j = 0; j < sources.size(); ++j)
        {
            const Index source_layer = sources[j];
            if (source_layer < 0) continue;  // external input — set in forward_propagate

            if (!forward_specs[source_layer].empty())
            {
                input_views[i][j] = forward_slots[source_layer].back();
                continue;
            }

            // Passthrough layer (empty specs): follow the chain upstream
            Index resolved = source_layer;
            while (resolved >= 0 && forward_specs[resolved].empty())
            {
                const auto& up = source_layers[resolved];
                if (up.empty()) { resolved = -1; break; }
                resolved = up[0];
            }

            if (resolved >= 0)
                input_views[i][j] = forward_slots[resolved].back();
            else
                passthrough_overrides.emplace_back(i, j, size_t(-resolved - 1));
        }
    }

    // AUTO conv-workspace value = largest single-layer activation. Only consulted
    // when the cap mode is AUTO (set_conv_workspace_cap(<0)); harmless otherwise.
    device::set_conv_workspace_auto_limit_bytes(max_layer_bytes);
}

TensorView ForwardPropagation::get_last_trainable_layer_outputs() const
{
    if (!neural_network) return {};

    const Index layer_index = neural_network->get_last_trainable_layer_index();
    
    if (layer_index < 0
        || size_t(layer_index) >= forward_slots.size()
        || forward_slots[layer_index].size() <= 1)
        return {};

    const TensorView& v = forward_slots[layer_index].back();
    return v.empty() ? TensorView{} : v;
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

    // A passthrough final layer (e.g. Scaling/Unscaling with None scalers)
    // allocates no forward slot: its output is its input view.
    if (last >= 0 && size_t(last) < input_views.size() && !input_views[last].empty())
    {
        const TensorView& input_view = input_views[last].front();
        if (!input_view.empty()) return input_view;
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
}

void ForwardPropagation::print() const
{
    cout << "Neural network forward propagation\n";

    if (!neural_network)
    {
        cout << "Neural network is not set.\n";
        return;
    }

    const size_t layers_number = neural_network->get_layers_number();

    cout << "Layers number: " << layers_number << "\n";

    for (size_t i = 0; i < layers_number; ++i)
        cout << "Layer " << i + 1 << ": " << neural_network->get_layer(static_cast<Index>(i))->get_label() << "\n";
}

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
