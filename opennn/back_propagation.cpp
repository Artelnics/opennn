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

#include <algorithm>
#include <limits>

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
    const vector<Shape>& per_layer_output_delta_shapes,
    const Shape& output_delta_dimensions,
    Type compute_dtype)
{
    DeltaPoolPlan plan;
    plan.alias_target.assign(size_t(layers_number), {SIZE_MAX, SIZE_MAX});

    if (last_trainable_layer_index < 0
        || first_trainable_layer_index < 0
        || last_trainable_layer_index < first_trainable_layer_index)
        return plan;

    const Index max_step = last_trainable_layer_index - first_trainable_layer_index;

    auto step_of = [&](Index layer_index) { return last_trainable_layer_index - layer_index; };
    auto in_range = [&](Index layer_index)
    {
        return layer_index >= first_trainable_layer_index
            && layer_index <= last_trainable_layer_index;
    };

    for (Index i = first_trainable_layer_index; i < last_trainable_layer_index; ++i)
    {
        if (backward_edges[i].size() != 1) continue;
        const BackwardEdge& edge = backward_edges[i].front();
        plan.alias_target[i] = {edge.consumer_index, edge.port + 1};
    }

    struct LiveRange { Index entry_index; Index bytes; Index birth; Index death; };
    vector<LiveRange> ranges;

    auto add = [&](Index layer, size_t slot, const Shape& shape, Type dtype,
                   Index birth, Index death)
    {
        const Index bytes = shape.size() * type_bytes(dtype);
        if (bytes <= 0) return;
        plan.entries.push_back({layer, slot, Index(-1), shape, dtype});
        ranges.push_back({Index(plan.entries.size() - 1),
                          get_aligned_bytes(bytes), birth, death});
    };

    add(last_trainable_layer_index, 0,
        output_delta_dimensions, compute_dtype,
        0, step_of(last_trainable_layer_index));

    for (Index i = first_trainable_layer_index; i <= last_trainable_layer_index; ++i)
    {
        const auto& shapes = backward_shapes[i];
        const auto& dtypes = backward_dtypes[i];
        const auto& inputs = layer_input_indices[i];

        for (size_t j = 0; j < shapes.size(); ++j)
        {
            const Index birth = step_of(i);
            const Index producer = (j < inputs.size()) ? inputs[j] : Index(-1);
            const Index death = in_range(producer) ? step_of(producer) : birth;

            add(i, j + 1, shapes[j], dtypes[j], birth, death);
        }
    }

    for (Index i = first_trainable_layer_index; i < last_trainable_layer_index; ++i)
    {
        if (backward_edges[i].size() <= 1) continue;
        if (i >= (Index)per_layer_output_delta_shapes.size()) continue;
        if (per_layer_output_delta_shapes[i].empty()) continue;

        add(i, 0, per_layer_output_delta_shapes[i], compute_dtype,
            step_of(i), step_of(i));
    }

    vector<vector<size_t>> births_by_step(size_t(max_step + 1));
    vector<vector<size_t>> deaths_by_step(size_t(max_step + 1));
    for (size_t id = 0; id < ranges.size(); ++id)
    {
        births_by_step[size_t(ranges[id].birth)].push_back(id);
        deaths_by_step[size_t(ranges[id].death)].push_back(id);
    }

    struct FreeChunk { Index offset; Index bytes; };
    vector<FreeChunk> free_list = {{0, std::numeric_limits<Index>::max()}};

    auto release = [&](Index offset, Index bytes)
    {
        auto it = std::lower_bound(free_list.begin(), free_list.end(), offset,
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
                release(plan.entries[ranges[id].entry_index].offset, ranges[id].bytes);

        for (size_t id : births_by_step[size_t(k)])
        {
            const Index off = acquire(ranges[id].bytes);
            plan.entries[ranges[id].entry_index].offset = off;

            const Index end = off + ranges[id].bytes;
            if (end > peak) peak = end;
        }
    }

    plan.peak_bytes = peak;

    return plan;
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

    const Shape output_shape = neural_network->get_output_shape();
    output_delta_dimensions = Shape({batch_size}).append(output_shape);

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
    for (size_t consumer_index = 0; consumer_index < layers_number; ++consumer_index)
    {
        const vector<Index>& inputs = layer_input_indices[consumer_index];
        for (size_t port_index = 0; port_index < inputs.size(); ++port_index)
        {
            const Index producer = inputs[port_index];
            if (producer >= 0 && static_cast<size_t>(producer) < layers_number)
                backward_edges[producer].push_back({consumer_index, port_index});
        }
    }

    vector<Shape> per_layer_output_delta_shapes(layers_number);
    for (size_t i = 0; i < layers_number; ++i)
    {
        const Index layer_index = static_cast<Index>(i);
        if (layer_index == last_trainable_layer_index) continue;
        if (backward_edges[i].size() <= 1)          continue;
        const Shape output_shape_i = layers[i]->get_output_shape();
        if (output_shape_i.empty())                 continue;
        per_layer_output_delta_shapes[i] = Shape({batch_size}).append(output_shape_i);
    }

    gradient_views.resize(layers_number);

    const Device device = is_gpu() ? Device::CUDA : Device::CPU;
    const Type compute_dtype = is_gpu() ? neural_network->get_training_type() : Type::FP32;

    const auto backward_dtypes = neural_network->get_backward_dtypes(batch_size);

    const Index total_gradient_bytes = aligned_total_elements(parameter_shapes) * Index(sizeof(float));
    if (total_gradient_bytes > 0)
    {
        gradient.resize_bytes(total_gradient_bytes, device);
        gradient.setZero();
    }

    uint8_t* g_cursor = gradient.as<uint8_t>();
    for (size_t i = 0; i < layers_number; ++i)
    {
        const vector<Shape>& layer_param_shapes = parameter_shapes[i];
        gradient_views[i].resize(layer_param_shapes.size());

        for (size_t j = 0; j < layer_param_shapes.size(); ++j)
        {
            const Shape& slot_shape = layer_param_shapes[j];
            if (slot_shape.size() == 0) continue;

            gradient_views[i][j] = TensorView(g_cursor, slot_shape, Type::FP32);
            g_cursor += get_aligned_bytes(slot_shape.size() * Index(sizeof(float)));
        }

        layers[i]->redistribute_parameter_gradients_to_operators(gradient_views[i]);
    }

    const DeltaPoolPlan plan = compute_delta_pool_plan(
        Index(layers_number),
        first_trainable_layer_index,
        last_trainable_layer_index,
        layer_input_indices,
        backward_edges,
        backward_shapes,
        backward_dtypes,
        per_layer_output_delta_shapes,
        output_delta_dimensions,
        compute_dtype);

    delta_views.assign(layers_number, {});
    for (size_t i = 0; i < layers_number; ++i)
        delta_views[i].assign(backward_shapes[i].size() + 1, TensorView{});

    if (plan.peak_bytes > 0)
    {
        delta_pool.resize_bytes(plan.peak_bytes, device);
        delta_pool.setZero();
    }

    uint8_t* base = delta_pool.as<uint8_t>();
    for (const DeltaPoolEntry& e : plan.entries)
        delta_views[e.layer][e.slot] = TensorView(base + e.offset, e.shape, e.dtype);

    for (size_t i = 0; i < layers_number; ++i)
    {
        const auto [consumer, slot] = plan.alias_target[i];
        if (consumer < delta_views.size()
            && slot < delta_views[consumer].size()
            && !delta_views[consumer][slot].empty())
            delta_views[i][0] = delta_views[consumer][slot];
    }
}

void BackPropagation::accumulate_output_deltas(size_t layer_index)
{
    if (layer_index >= delta_views.size()) return;
    if (backward_edges[layer_index].size() <= 1) return;

    TensorView& destination = delta_views[layer_index][0];
    if (!destination.data) return;

    vector<const TensorView*> sources;
    sources.reserve(backward_edges[layer_index].size());
    for (const BackwardEdge& edge : backward_edges[layer_index])
    {
        const size_t slot = 1 + edge.port;

        if (edge.consumer_index >= delta_views.size()) continue;
        const auto& consumer_views = delta_views[edge.consumer_index];
        if (slot >= consumer_views.size() || consumer_views[slot].empty()) continue;

        const TensorView& source = consumer_views[slot];
        if (!source.data || source.size() != destination.size()) continue;

        sources.push_back(&source);
    }

    if (sources.empty())
    {
        destination.fill(0.0f);
        return;
    }

    if (sources.size() == 1)
    {
        copy(*sources[0], destination);
        return;
    }

    IF_GPU({
        copy(*sources[0], destination);
        for (size_t s = 1; s < sources.size(); ++s)
            add(destination, *sources[s], destination);
        return;
    });

    const Index n = destination.size();
    float* dst = destination.as<float>();
    const size_t k = sources.size();

    vector<const float*> ptrs(k);
    transform(sources.begin(), sources.end(), ptrs.begin(),
              [](const TensorView* tv) { return tv->as<float>(); });

    #pragma omp parallel for
    for (Index i = 0; i < n; ++i)
    {
        float sum = ptrs[0][i];
        for (size_t s = 1; s < k; ++s) sum += ptrs[s][i];
        dst[i] = sum;
    }
}

TensorView& BackPropagation::get_output_deltas()
{
    return delta_views[loss->get_neural_network()->get_last_trainable_layer_index()][0];
}

const TensorView& BackPropagation::get_output_deltas() const
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
