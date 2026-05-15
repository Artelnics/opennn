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

namespace {

using BackwardEdge = BackPropagation::BackwardEdge;

struct DeltaPoolEntry
{
    Index layer;
    size_t slot;
    Index offset;
    Shape shape;
    Type dtype;
    Index bytes;
    Index birth;
    Index death;
};

struct DeltaPoolPlan
{
    Index peak_bytes = 0;
    vector<DeltaPoolEntry> entries;
    vector<pair<size_t, size_t>> alias_target;
};

DeltaPoolPlan compute_delta_pool_plan(
    Index layers_number,
    Index first_trainable_layer_index,
    Index last_trainable_layer_index,
    const vector<vector<Index>>& layer_input_indices,
    const vector<vector<BackwardEdge>>& backward_edges,
    const vector<vector<Shape>>& backward_shapes,
    const vector<vector<Type>>& backward_dtypes,
    const vector<Shape>& layer_output_delta_shapes,
    const Shape& output_delta_shape,
    Type compute_dtype)
{
    DeltaPoolPlan delta_pool_plan;
    delta_pool_plan.alias_target.assign(size_t(layers_number), {SIZE_MAX, SIZE_MAX});

    if (last_trainable_layer_index < 0
        || first_trainable_layer_index < 0
        || last_trainable_layer_index < first_trainable_layer_index)
        return delta_pool_plan;

    const Index max_step = last_trainable_layer_index - first_trainable_layer_index;

    auto step_of = [&](Index layer_index) { return last_trainable_layer_index - layer_index; };
    auto in_range = [&](Index layer_index)
    {
        return layer_index >= first_trainable_layer_index
            && layer_index <= last_trainable_layer_index;
    };

    for (Index layer_index = first_trainable_layer_index; layer_index < last_trainable_layer_index; ++layer_index)
    {
        if (backward_edges[layer_index].size() != 1) continue;
        const BackwardEdge& edge = backward_edges[layer_index].front();
        delta_pool_plan.alias_target[layer_index] = {edge.consumer_layer_index, edge.consumer_input_index + 1};
    }

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
            const Index death = in_range(producer) ? step_of(producer) : birth;

            add(layer_index, j + 1, shapes[j], dtypes[j], birth, death);
        }
    }

    for (Index layer_index = first_trainable_layer_index; layer_index < last_trainable_layer_index; ++layer_index)
    {
        if (backward_edges[layer_index].size() <= 1) continue;
        if (layer_index >= (Index)layer_output_delta_shapes.size()) continue;
        if (layer_output_delta_shapes[layer_index].empty()) continue;

        add(layer_index, 0, layer_output_delta_shapes[layer_index], compute_dtype,
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

    const NeuralNetwork* neural_network = loss->get_neural_network();

    if (!neural_network)
        throw runtime_error("neural network is not set.");

    const Shape output_delta_shape = Shape({batch_size}).append(neural_network->get_output_shape());

    loss_value = 0.0f;
    error = 0.0f;
    accuracy = 0.0f;

    const auto& layers = neural_network->get_layers();
    const size_t layers_number = layers.size();
    const vector<vector<Shape>> parameter_shapes = neural_network->get_parameter_shapes();
    const vector<vector<Shape>> backward_shapes  = neural_network->get_backward_shapes(batch_size);
    const Index first_trainable_layer_index = neural_network->get_first_trainable_layer_index();
    const Index last_trainable_layer_index = neural_network->get_last_trainable_layer_index();
    const auto& layer_input_indices = neural_network->get_layer_input_indices();

    backward_edges.assign(layers_number, {});

    for (size_t layer_index = 0; layer_index < layers_number; ++layer_index)
    {
        const vector<Index>& input_indices = layer_input_indices[layer_index];
        for (size_t input_index = 0; input_index < input_indices.size(); ++input_index)
        {
            const Index producer = input_indices[input_index];
            if (producer >= 0 && size_t(producer) < layers_number)
                backward_edges[producer].push_back({layer_index, input_index});
        }
    }

    vector<Shape> layer_output_delta_shapes(layers_number);

    for (Index layer_index = 0; layer_index < (Index)layers_number; ++layer_index)
    {
        if (layer_index == last_trainable_layer_index) continue;
        if (backward_edges[layer_index].size() <= 1) continue;

        const Shape output_shape = layers[layer_index]->get_output_shape();
        if (!output_shape.empty())
            layer_output_delta_shapes[layer_index] = Shape({batch_size}).append(output_shape);
    }

    gradient_views.resize(layers_number);

    const Device device = is_gpu() ? Device::CUDA : Device::CPU;

    const auto backward_dtypes = neural_network->get_backward_dtypes(batch_size);

    const Index gradient_bytes = aligned_total_elements(parameter_shapes) * Index(sizeof(float));
    
    if (gradient_bytes > 0)
    {
        gradient.resize_bytes(gradient_bytes, device);
        gradient.setZero();
    }

    uint8_t* cursor = gradient.as<uint8_t>();
    for (Index layer_index = 0; layer_index < (Index)layers_number; ++layer_index)
    {
        const vector<Shape>& shapes = parameter_shapes[layer_index];
        gradient_views[layer_index].resize(shapes.size());

        for (size_t j = 0; j < shapes.size(); ++j)
        {
            if (shapes[j].size() == 0) continue;
            gradient_views[layer_index][j] = TensorView(cursor, shapes[j], Type::FP32);
            cursor += get_aligned_bytes(shapes[j].size() * Index(sizeof(float)));
        }

        layers[layer_index]->redistribute_parameter_gradients_to_operators(gradient_views[layer_index]);
    }

    const Type compute_dtype = is_gpu() ? neural_network->get_training_type() : Type::FP32;

    const DeltaPoolPlan delta_pool_plan = compute_delta_pool_plan(
        Index(layers_number),
        first_trainable_layer_index,
        last_trainable_layer_index,
        layer_input_indices,
        backward_edges,
        backward_shapes,
        backward_dtypes,
        layer_output_delta_shapes,
        output_delta_shape,
        compute_dtype);

    delta_views.resize(layers_number);

    for (Index layer_index = 0; layer_index < (Index)layers_number; ++layer_index)
        delta_views[layer_index].assign(backward_shapes[layer_index].size() + 1, TensorView{});

    if (delta_pool_plan.peak_bytes > 0)
    {
        delta_pool.resize_bytes(delta_pool_plan.peak_bytes, device);
        delta_pool.setZero();
    }

    uint8_t* base = delta_pool.as<uint8_t>();
    for (const DeltaPoolEntry& e : delta_pool_plan.entries)
        delta_views[e.layer][e.slot] = TensorView(base + e.offset, e.shape, e.dtype);

    for (Index layer_index = 0; layer_index < layers_number; ++layer_index)
    {
        const auto [consumer, slot] = delta_pool_plan.alias_target[layer_index];
        if (consumer < delta_views.size()
            && slot < delta_views[consumer].size()
            && !delta_views[consumer][slot].empty())
            delta_views[layer_index][0] = delta_views[consumer][slot];
    }
}

void BackPropagation::accumulate_output_deltas(size_t layer_index)
{
    if (layer_index >= delta_views.size()) return;

    const auto& edges = backward_edges[layer_index];
    if (edges.size() <= 1) return;

    TensorView& destination = delta_views[layer_index][0];
    if (!destination.data) return;

    bool started = false;
    for (const BackwardEdge& edge : edges)
    {
        const size_t slot = 1 + edge.consumer_input_index;

        if (edge.consumer_layer_index >= delta_views.size()) continue;
        const auto& consumer_views = delta_views[edge.consumer_layer_index];
        if (slot >= consumer_views.size() || consumer_views[slot].empty()) continue;

        const TensorView& source = consumer_views[slot];
        if (!source.data || source.size() != destination.size()) continue;

        if (!started) { copy(source, destination); started = true; }
        else          { add(destination, source, destination); }
    }

    if (!started) destination.fill(0.0f);
}

TensorView& BackPropagation::get_output_delta()
{
    return delta_views[loss->get_neural_network()->get_last_trainable_layer_index()][0];
}

const TensorView& BackPropagation::get_output_delta() const
{
    return delta_views[loss->get_neural_network()->get_last_trainable_layer_index()][0];
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
