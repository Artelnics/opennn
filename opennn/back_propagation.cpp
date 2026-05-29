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

    if (!loss_pointer)
        throw runtime_error("loss is not set.");

    neural_network = loss_pointer->get_neural_network();

    if (!neural_network)
        throw runtime_error("neural network is not set.");

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
        Index      birth;
        Index      death;
        Index      offset = -1;
    };

    const auto& layers = neural_network->get_layers();
    const Index layers_number = neural_network->get_layers_number();
    const Index first_trainable_layer_index = neural_network->get_first_trainable_layer_index();
    const Index last_trainable_layer_index = neural_network->get_last_trainable_layer_index();
    const auto& source_layers = neural_network->get_source_layers();
    const Type compute_dtype = neural_network->is_gpu()
        ? neural_network->get_training_type()
        : Type::FP32;

    vector<DeltaEntry> deltas;

    const Index max_step = last_trainable_layer_index - first_trainable_layer_index;

    const Shape output_delta_shape = Shape({batch_size}).append(layers[last_trainable_layer_index]->get_output_shape());

    if (output_delta_shape.size() != 0)
        deltas.push_back({last_trainable_layer_index, 0, {output_delta_shape, compute_dtype}, 0, 0});

    for (Index layer_index = first_trainable_layer_index; layer_index <= last_trainable_layer_index; ++layer_index)
    {
        const auto& specs = backward_specs[layer_index];
        const auto& sources = source_layers[layer_index];

        for (size_t j = 0; j < specs.size(); ++j)
        {
            const auto& [shape, dtype] = specs[j];
            if (shape.size() == 0) continue;

            const Index birth = last_trainable_layer_index - layer_index;
            const Index source_layer = (j < sources.size()) ? sources[j] : Index(-1);
            const bool source_layer_is_trainable = source_layer >= first_trainable_layer_index
                                                && source_layer <= last_trainable_layer_index;

            const Index death = source_layer_is_trainable ? last_trainable_layer_index - source_layer : birth;

            deltas.push_back({layer_index, j + 1, {shape, dtype}, birth, death});
        }
    }

    for (Index layer_index = first_trainable_layer_index; layer_index < last_trainable_layer_index; ++layer_index)
    {
        const auto& edges = consumer_edges[layer_index];

        if (edges.empty()) continue;
        if (edges.size() == 1) continue;

        const Shape output_shape = layers[layer_index]->get_output_shape();
        if (output_shape.empty()) continue;

        const Shape delta_shape = Shape({batch_size}).append(output_shape);
        const Index step = last_trainable_layer_index - layer_index;

        deltas.push_back({layer_index, 0, {delta_shape, compute_dtype}, step, step});
    }

    // YOLO multi-head (FPN): every Detection layer is a training target —
    // the Loss writes a per-head output-delta to slot 0 of each. Pool
    // would otherwise only back the last-trainable Detection's slot 0.
    // Lifetime: born at step 0 (loss writes all heads at once at the start of
    // backward), dies when its layer is walked.
    for (Index layer_index = first_trainable_layer_index; layer_index < last_trainable_layer_index; ++layer_index)
    {
        if (layers[layer_index]->get_type() != LayerType::Detection) continue;
        if (!consumer_edges[layer_index].empty()) continue;

        const Shape output_shape = layers[layer_index]->get_output_shape();
        if (output_shape.empty()) continue;

        const Shape delta_shape = Shape({batch_size}).append(output_shape);
        const Index step = last_trainable_layer_index - layer_index;

        deltas.push_back({layer_index, 0, {delta_shape, compute_dtype}, 0, step});
    }

    vector<vector<size_t>> births_by_step(size_t(max_step + 1));
    vector<vector<size_t>> deaths_by_step(size_t(max_step + 1));

    for (size_t id = 0; id < deltas.size(); ++id)
    {
        births_by_step[size_t(deltas[id].birth)].push_back(id);
        deaths_by_step[size_t(deltas[id].death)].push_back(id);
    }

    vector<pair<Index, Index>> free_list = {{0, numeric_limits<Index>::max()}};
    Index peak_bytes = 0;

    for (Index step = 0; step <= max_step; ++step)
    {
        for (size_t id : births_by_step[size_t(step)])
        {
            const Index bytes = get_aligned_bytes(deltas[id].spec.shape.size(), deltas[id].spec.dtype);
            Index offset = Index(-1);

            auto it = ranges::find_if(free_list, [bytes](const auto& block) { return block.second >= bytes; });

            if (it != free_list.end())
            {
                offset = it->first;
                if (it->second == bytes)
                    free_list.erase(it);
                else
                {
                    it->first  += bytes;
                    it->second -= bytes;
                }
            }

            deltas[id].offset = offset;
            peak_bytes = max(peak_bytes, offset + bytes);
        }

        for (size_t id : deaths_by_step[size_t(step)])
        {
            const Index bytes = get_aligned_bytes(deltas[id].spec.shape.size(), deltas[id].spec.dtype);

            auto it = ranges::lower_bound(free_list, deltas[id].offset, {}, &pair<Index, Index>::first);
            
            it = free_list.insert(it, {deltas[id].offset, bytes});

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
        delta_views[i].resize(backward_specs[i].size() + 1);

    delta_pool.resize_bytes(peak_bytes, neural_network->get_device());
    delta_pool.setZero();

    uint8_t* const base = delta_pool.as<uint8_t>();

    for (const auto& delta : deltas)
        delta_views[delta.layer][delta.slot] = TensorView(base + delta.offset,
                                                          delta.spec.shape,
                                                          delta.spec.dtype,
                                                          delta_pool.device_type);

    for (Index i = first_trainable_layer_index; i < last_trainable_layer_index; ++i)
    {
        const auto& edges = consumer_edges[i];
        if (edges.size() != 1) continue;

        const auto& [consumer_layer, input_position] = edges.front();
        const size_t slot = input_position + 1;
        const auto& consumer_deltas = delta_views[consumer_layer];

        if (slot < consumer_deltas.size() && !consumer_deltas[slot].empty())
            delta_views[i][0] = consumer_deltas[slot];
    }
}

void BackPropagation::accumulate_output_deltas(size_t layer_index)
{
    const auto& edges = consumer_edges[layer_index];
    if (edges.size() <= 1) return;

    TensorView& destination = delta_views[layer_index][0];
    if (!destination.data) return;

    destination.setZero();

    for (const auto& [consumer_layer, input_position] : edges)
    {
        const TensorView& source = delta_views[consumer_layer][1 + input_position];
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
         << "Regularization: " << regularization << "\n"
         << "Loss:  " << loss << "\n";
}

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
