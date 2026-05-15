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
#include "cuda_dispatch.h"

namespace opennn
{

BackPropagation::DeltaPoolPlan BackPropagation::compute_delta_pool_plan(
    const vector<vector<Shape>>& backward_shapes,
    const vector<vector<Type>>& backward_dtypes) const
{
    const auto& layers = neural_network->get_layers();
    const Index layers_number = Index(layers.size());
    const Index first_trainable_layer_index = neural_network->get_first_trainable_layer_index();
    const Index last_trainable_layer_index = neural_network->get_last_trainable_layer_index();
    const auto& layer_input_indices = neural_network->get_layer_input_indices();
    const Shape output_delta_shape = Shape({batch_size}).append(neural_network->get_output_shape());
    const Type compute_dtype = is_gpu() ? neural_network->get_training_type() : Type::FP32;

    DeltaPoolPlan delta_pool_plan;
    delta_pool_plan.alias_target.assign(size_t(layers_number), {SIZE_MAX, SIZE_MAX});

    const Index max_step = last_trainable_layer_index - first_trainable_layer_index;

    auto step_of = [&](Index layer_index) { return last_trainable_layer_index - layer_index; };

    auto add = [&](Index layer, size_t slot, const Shape& shape, Type dtype,
                   Index birth, Index death)
    {
        const Index bytes = shape.size() * type_bytes(dtype);
        if (bytes <= 0) return;
        delta_pool_plan.entries.push_back({layer, slot, Index(-1), shape, dtype,
                                get_aligned_bytes(bytes), birth, death});
    };

    add(last_trainable_layer_index, 0,
        output_delta_shape, compute_dtype,
        0, step_of(last_trainable_layer_index));

    for (Index layer_index = first_trainable_layer_index; layer_index <= last_trainable_layer_index; ++layer_index)
    {
        const auto& shapes = backward_shapes[layer_index];
        const auto& dtypes = backward_dtypes[layer_index];
        const auto& input_indices = layer_input_indices[layer_index];

        for (size_t j = 0; j < shapes.size(); ++j)
        {
            const Index birth = step_of(layer_index);
            const Index producer = (j < input_indices.size()) ? input_indices[j] : Index(-1);
            const bool producer_is_trainable = producer >= first_trainable_layer_index
                                            && producer <= last_trainable_layer_index;
            const Index death = producer_is_trainable ? step_of(producer) : birth;

            add(layer_index, j + 1, shapes[j], dtypes[j], birth, death);
        }
    }

    for (Index layer_index = first_trainable_layer_index; layer_index < last_trainable_layer_index; ++layer_index)
    {
        const auto& edges = backward_edges[layer_index];

        if (edges.size() == 1)
        {
            const BackwardEdge& edge = edges.front();
            delta_pool_plan.alias_target[layer_index] = {edge.consumer_layer_index, edge.consumer_input_index + 1};
            continue;
        }

        if (edges.size() < 2) continue;

        const Shape output_shape = layers[layer_index]->get_output_shape();
        if (output_shape.empty()) continue;

        add(layer_index, 0, Shape({batch_size}).append(output_shape), compute_dtype,
            step_of(layer_index), step_of(layer_index));
    }

    vector<vector<size_t>> births_by_step(size_t(max_step + 1));
    vector<vector<size_t>> deaths_by_step(size_t(max_step + 1));

    for (size_t id = 0; id < delta_pool_plan.entries.size(); ++id)
    {
        births_by_step[size_t(delta_pool_plan.entries[id].birth)].push_back(id);
        deaths_by_step[size_t(delta_pool_plan.entries[id].death)].push_back(id);
    }

    struct FreeChunk { Index offset; Index bytes; };
    
    vector<FreeChunk> free_list = {{0, numeric_limits<Index>::max()}};

    auto release = [&](Index offset, Index bytes)
    {
        auto it = lower_bound(free_list.begin(), free_list.end(), offset,
            [](const FreeChunk& c, Index off) { return c.offset < off; });
        it = free_list.insert(it, {offset, bytes});

        if (it + 1 != free_list.end() && it->offset + it->bytes == (it + 1)->offset)
        {
            it->bytes += (it + 1)->bytes;
            free_list.erase(it + 1);
        }
        
        if (it != free_list.begin() && (it - 1)->offset + (it - 1)->bytes == it->offset)
        {
            (it - 1)->bytes += it->bytes;
            free_list.erase(it);
        }
    };

    auto acquire = [&](Index bytes) -> Index
    {
        for (auto it = free_list.begin(); it != free_list.end(); ++it)
        {
            if (it->bytes < bytes) continue;
            const Index off = it->offset;
            if (it->bytes == bytes) free_list.erase(it);
            else { it->offset += bytes; it->bytes -= bytes; }
            return off;
        }
        return Index(-1);
    };

    Index peak = 0;

    for (Index k = 0; k <= max_step; ++k)
    {
        if (k > 0)
            for (size_t id : deaths_by_step[size_t(k - 1)])
                release(delta_pool_plan.entries[id].offset, delta_pool_plan.entries[id].bytes);

        for (size_t id : births_by_step[size_t(k)])
        {
            const Index off = acquire(delta_pool_plan.entries[id].bytes);
            delta_pool_plan.entries[id].offset = off;

            const Index end = off + delta_pool_plan.entries[id].bytes;
            if (end > peak) peak = end;
        }
    }

    delta_pool_plan.peak_bytes = peak;

    return delta_pool_plan;
}

BackPropagation::BackPropagation(const Index new_batch_size, Loss* new_loss)
{
    set(new_batch_size, new_loss);
}

void BackPropagation::set(const Index new_batch_size, Loss* new_loss)
{
    batch_size = new_batch_size;
    loss = new_loss;

    if (!loss)
        throw runtime_error("loss is not set.");

    neural_network = loss->get_neural_network();

    if (!neural_network)
        throw runtime_error("neural network is not set.");

    loss_value = 0.0f;
    error = 0.0f;
    accuracy = 0.0f;

    const auto& layers = neural_network->get_layers();
    const size_t layers_number = layers.size();
    const vector<vector<Shape>> parameter_shapes = neural_network->get_parameter_shapes();
    const vector<vector<Shape>> backward_shapes  = neural_network->get_backward_shapes(batch_size);
    const vector<vector<Type>>  backward_dtypes  = neural_network->get_backward_dtypes(batch_size);
    const auto& layer_input_indices = neural_network->get_layer_input_indices();

    backward_edges.assign(layers_number, {});

    for (size_t i = 0; i < layers_number; ++i)
    {
        const vector<Index>& input_indices = layer_input_indices[i];
        for (size_t j = 0; j < input_indices.size(); ++j)
            if (const Index producer = input_indices[j]; size_t(producer) < layers_number)
                backward_edges[producer].push_back({i, j});
    }

    const Device device = current_device();

    if (const Index gradient_bytes = get_aligned_bytes(parameter_shapes, Type::FP32); gradient_bytes > 0)
    {
        gradient.resize_bytes(gradient_bytes, device);
        gradient.setZero();
    }

    gradient_views.resize(layers_number);

    float* pointer = gradient.as<float>();
    for (Index i = 0; i < layers_number; ++i)
        pointer = layers[i]->link_gradients(pointer, gradient_views[i]);

    const DeltaPoolPlan delta_pool_plan = compute_delta_pool_plan(backward_shapes, backward_dtypes);

    delta_views.resize(layers_number);

    for (Index i = 0; i < layers_number; ++i)
        delta_views[i].assign(backward_shapes[i].size() + 1, TensorView{});

    if (delta_pool_plan.peak_bytes > 0)
    {
        delta_pool.resize_bytes(delta_pool_plan.peak_bytes, device);
        delta_pool.setZero();
    }

    uint8_t* base = delta_pool.as<uint8_t>();

    for (const DeltaPoolEntry& entry : delta_pool_plan.entries)
        delta_views[entry.layer][entry.slot] = TensorView(base + entry.offset, entry.shape, entry.dtype);

    for (Index i = 0; i < layers_number; ++i)
        if (const auto [consumer, slot] = delta_pool_plan.alias_target[i];
            consumer < delta_views.size()
            && slot < delta_views[consumer].size()
            && !delta_views[consumer][slot].empty())
            delta_views[i][0] = delta_views[consumer][slot];
}

void BackPropagation::accumulate_output_deltas(size_t layer_index)
{
    if (layer_index >= delta_views.size()) return;

    const auto& edges = backward_edges[layer_index];
    if (edges.size() <= 1) return;

    TensorView& destination = delta_views[layer_index][0];
    if (!destination.data) return;

    destination.setZero();

    for (const BackwardEdge& edge : edges)
    {
        const size_t slot = 1 + edge.consumer_input_index;

        if (edge.consumer_layer_index >= delta_views.size()) continue;

        const auto& consumer_views = delta_views[edge.consumer_layer_index];

        if (slot >= consumer_views.size() || consumer_views[slot].empty()) continue;

        const TensorView& source = consumer_views[slot];
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
         << "Loss:  " << loss_value << "\n";
}

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
