// SPDX-License-Identifier: LGPL-2.1-or-later
// Copyright (C) 2005-2026 Artificial Intelligence, SL.

#include "opennn/network/network.h"
#include "opennn/network/network_internal.h"

#include <algorithm>
#include <cmath>
#include <utility>

#include "opennn/core/cuda/kernel_cast.cuh"
#include "opennn/core/cuda/kernel_tensor.cuh"
#include "opennn/core/memory_debug.h"
#include "opennn/core/profiler.h"
#include "opennn/core/string_utilities.h"
#include "opennn/core/tensor_types.h"
#include "opennn/core/variable.h"
#include "opennn/network/back_propagation.h"
#include "opennn/network/forward_propagation.h"
#include "opennn/network/layers/dense_layer.h"
#include "opennn/network/model_expression.h"
#include "opennn/network/operators/combination_operator.h"
#include "opennn/registry.h"

namespace opennn
{

using network_detail::validate_source_indices;
using network_detail::validate_source_arity;

#ifdef OPENNN_HAS_CUDA
using network_detail::quantization_channel;
using network_detail::finalize_int8_scales;
using network_detail::quantize_int8_host;
#endif

namespace
{

#ifdef OPENNN_HAS_CUDA
vector<CombinationOperator*> get_combination_operators(Layer& layer)
{
    vector<CombinationOperator*> combinations;
    combinations.reserve(layer.get_operators().size());

    for (Operator* op : layer.get_operators())
        if (auto* combination = dynamic_cast<CombinationOperator*>(op))
            combinations.push_back(combination);

    return combinations;
}
#endif

void wire_drelu_fusions(vector<unique_ptr<Layer>>& layers,
                        const vector<vector<Index>>& source_layers,
                        const vector<vector<pair<size_t, size_t>>>& consumer_edges,
                        Device device,
                        Type training_type)
{
    for (auto& layer : layers)
        if (auto* dense = dynamic_cast<Dense*>(layer.get()))
        {
            dense->reset_drelu_fusion();
            dense->reset_single_output_relu_fusion();
        }

    if (device != Device::CUDA || !is_one_of(training_type, Type::FP32, Type::BF16))
        return;

    const bool drelu_enabled = env_flag_enabled("OPENNN_DRELU_FUSION");

    for (size_t i = 0; i < source_layers.size(); ++i)
    {
        const auto& sources = source_layers[i];
        if (sources.size() != 1 || sources[0] < 0) continue;
        if (consumer_edges[size_t(sources[0])].size() != 1) continue;

        auto* consumer = dynamic_cast<Dense*>(layers[i].get());
        auto* producer = dynamic_cast<Dense*>(layers[size_t(sources[0])].get());

        if (!consumer || !producer) continue;

        if (!consumer->try_wire_single_output_relu_fusion(*producer, sources[0]) && drelu_enabled)
            consumer->try_wire_drelu_fusion(*producer, sources[0]);
    }
}

}

Network::Network()
    : Network(NetworkTask::Generic)
{
}

Network::Network(NetworkTask new_task)
    : task(new_task)
{
    clear();
}

Network::Network(const filesystem::path& file_name)
    : Network(file_name, NetworkTask::Generic)
{
}

Network::Network(const filesystem::path& file_name, NetworkTask new_task)
    : Network(new_task)
{
    load(file_name);
}

Index Network::add_layer(unique_ptr<Layer> layer, const vector<Index>& sources)
{
    throw_if(!layer, "Network: cannot add a null layer.");

    const Index old_layers_number = get_layers_number() - 1;

    if (!layers.empty())
        throw_if(!layers.back()->allows_successors(),
                 "No layers can be added after a {} layer.\n",
                 layers.back()->get_name());

    const vector<Index> resolved_sources = sources.empty()
        ? vector<Index>{old_layers_number}
        : sources;

    validate_source_indices(resolved_sources, ssize(layers), ssize(layers));
    validate_source_arity(*layer, resolved_sources, ssize(layers));

    layers.push_back(std::move(layer));

    source_layers.push_back(resolved_sources);

    linked_gradient_base   = nullptr;

    return ssize(layers) - 1;
}

void Network::compile()
{
    if (get_layers_number() == 0) return;
    compile(Configuration::instance().resolve());
}

void Network::compile(const Device device)
{
    if (get_layers_number() == 0) return;
    compile(Configuration::instance().resolve_for(device));
}

void Network::compile(EffectiveConfig new_config, const bool allocate_parameter_master)
{
    mark_parameters_changed();

    config = new_config;

    stale_configuration_warned = false;

    for (auto& layer : layers)
    {
        layer->set_compute_device(get_device());
        layer->set_compute_dtype(get_training_type());
    }

    // The BF16 inference loader writes compact device storage and releases the
    // fp32 master without reading it, so a network compiled for that loader
    // skips the master: for a 4B model it is 16 GiB of host memory that would
    // only be zeroed and freed. With no master there is nothing to link; the
    // loader links the operators once its storage exists.
    parameters.resize_bytes(allocate_parameter_master
                            ? get_aligned_bytes(get_parameter_specs(), Type::FP32)
                            : Index(0),
                            Device::CPU);
    parameters.setZero();

    clear_low_precision_parameter_storage();

    if (allocate_parameter_master) link_parameters();

    states.resize_bytes(get_states_size() * Index(sizeof(float)), Device::CPU);
    states.setZero();

    link_states();

    for (auto& layer : layers)
        for (Operator* op : layer->get_operators())
            op->initialize_states();

    wire_drelu_fusions(layers, source_layers, get_consumer_edges(),
                       get_device(), get_training_type());
}

void Network::clear_low_precision_parameter_storage()
{
    parameters_bf16_mirror.resize_bytes(0, Device::CUDA);
    parameters_bf16_mirror_compact = false;
    parameters_fp32_inference_storage.resize_bytes(0, Device::CUDA);
    parameters_int8_storage.resize_bytes(0, Device::CUDA);
}

void Network::warn_if_stale_configuration() const
{
    if (stale_configuration_warned
        || config.generation == Configuration::instance().get_generation())
        return;

    stale_configuration_warned = true;

    logging::warning() << "Warning: Configuration::set() was called after this network was compiled, "
            "so it has no effect on it. The network keeps the settings resolved at "
            "compile() time; call Configuration::set() before constructing the network.\n";
}

bool Network::has(const string& name) const
{
    return has(string_to_layer_type(name));
}

bool Network::has(LayerType type) const
{
    return ranges::any_of(layers,
                          [type](const unique_ptr<Layer>& layer) {return layer->get_type() == type;});
}

bool Network::has_recurrent_layers() const
{
    return ranges::any_of(layers, [](const unique_ptr<Layer>& layer)
    {
        return layer->is_recurrent();
    });
}

const unique_ptr<Layer>& Network::get_layer(const string& label) const
{
    auto it = ranges::find_if(layers,
                              [&label](const unique_ptr<Layer>& layer) { return layer->get_label() == label; });

    if (it != layers.end())
        return *it;

    throw runtime_error("Layer not found in neural network");
}

Index Network::get_layer_index(const string& new_label) const
{
    if (contains({"Dataset", "decoder"}, new_label))
        return -1;

    if (new_label == "input")
        return -2;

    auto it = ranges::find_if(layers,
                              [&new_label](const unique_ptr<Layer>& layer) { return layer->get_label() == new_label; });

    if (it != layers.end())
        return distance(layers.begin(), it);

    throw runtime_error(format("Layer not found: {}", new_label));
}

vector<vector<pair<size_t, size_t>>> Network::get_consumer_edges() const
{
    vector<vector<pair<size_t, size_t>>> edges(layers.size());

    for (size_t consumer = 0; consumer < source_layers.size(); ++consumer)
        for (size_t input = 0; input < source_layers[consumer].size(); ++input)
            if (const Index source = source_layers[consumer][input]; source >= 0)
                edges[size_t(source)].push_back({consumer, input});

    return edges;
}

const Layer* Network::get_first(const string& name) const
{
    return get_first(string_to_layer_type(name));
}

const Layer* Network::get_first(LayerType type) const
{
    auto it = ranges::find_if(layers,
                              [type](const unique_ptr<Layer>& layer) { return layer->get_type() == type; });

    return it != layers.end() ? it->get() : nullptr;
}

Layer* Network::get_first(const string& name)
{
    return get_first(string_to_layer_type(name));
}

Layer* Network::get_first(LayerType type)
{
    return const_cast<Layer*>(static_cast<const Network*>(this)->get_first(type));
}

static void define_variables_from_names(vector<Variable>& variables,
                                        const vector<string>& names,
                                        VariableRole role)
{
    variables.assign(names.size(), Variable());

    for (size_t i = 0; i < names.size(); ++i)
    {
        variables[i].name = names[i];
        variables[i].role = role;
        variables[i].type = VariableType::Numeric;
    }
}

static void set_variable_names(vector<Variable>& variables, const vector<string>& new_names)
{

    if (ranges::any_of(variables,
                       [](const Variable& v) { return !v.is_categorical() && v.features > 1; }))
        return define_variables_from_names(variables, new_names,
                                           variables.empty() ? VariableRole::None : variables[0].role);

    const size_t total = new_names.size();
    size_t name_index = 0;
    for (size_t i = 0; i < variables.size(); ++i)
    {
        if (variables[i].is_categorical())
        {
            const size_t num_cats = variables[i].get_categories_number();
            throw_if(name_index + num_cats > total,
                     "set_variable_names: not enough names for categorical variable {} (need {}, have {}).",
                            i, num_cats, total - name_index);
            variables[i].categories.assign(new_names.begin() + name_index,
                                           new_names.begin() + name_index + num_cats);
            name_index += num_cats;
        }
        else
        {
            throw_if(name_index >= total,
                     "set_variable_names: not enough names for scalar variable {}.", i);
            variables[i].name = new_names[name_index];
            ++name_index;
        }
    }

    throw_if(name_index != total,
             "set_variable_names: received {} names but variables expected {}.",
                    total, name_index);
}

void Network::set_input_names(const vector<string>& new_input_names)
{
    if (input_variables.empty() && !new_input_names.empty())
        return define_variables_from_names(input_variables, new_input_names, VariableRole::Input);

    set_variable_names(input_variables, new_input_names);
}

void Network::set_output_names(const vector<string>& new_output_names)
{
    if (output_variables.empty() && !new_output_names.empty())
        return define_variables_from_names(output_variables, new_output_names, VariableRole::Target);

    set_variable_names(output_variables, new_output_names);
}

void Network::set_input_shape(const Shape& new_input_shape)
{
    constexpr Index primary_external_source = -1;

    if (get_features_number(input_variables) != new_input_shape.size())
    {
        input_variables.assign(1, Variable());
        input_variables[0].name = "input";
        input_variables[0].role = VariableRole::Input;
        input_variables[0].type = VariableType::Numeric;
        input_variables[0].features = new_input_shape.size();
    }

    const Index layers_number = get_layers_number();
    for (Index i = 0; i < layers_number; ++i)
    {
        const vector<Index>& sources = source_layers[i];
        if (sources.size() != 1) continue;

        const Index source = sources[0];
        if (source == primary_external_source)
            layers[i]->set_input_shape(new_input_shape);
        else if (source >= 0)
            layers[i]->set_input_shape(layers[source]->get_output_shape());
    }
}

void Network::clear()
{
    layers.clear();

    source_layers.clear();

    input_variables.clear();

    output_variables.clear();

    linked_gradient_base   = nullptr;
}

void Network::steal_from(Network& src)
{
    mark_parameters_changed();

    clear();
    task             = src.task;
    layers           = std::move(src.layers);
    source_layers    = std::move(src.source_layers);
    input_variables  = std::move(src.input_variables);
    output_variables = std::move(src.output_variables);
    linked_gradient_base       = nullptr;
    src.linked_gradient_base   = nullptr;
    link_parameters();
}

Shape Network::get_input_shape() const
{
    if (layers.empty())
        return {};

    return layers[0]->get_input_shape();
}

Shape Network::get_output_shape() const
{
    if (layers.empty())
        return {};

    return layers.back()->get_output_shape();
}

ActivationFunction Network::get_output_activation() const
{
    const Index last_index = get_last_trainable_layer_index();
    if (last_index < 0 || static_cast<size_t>(last_index) >= layers.size())
        return ActivationFunction::Identity;

    return layers[last_index]->get_output_activation();
}

Index Network::get_parameters_number() const
{
    return transform_reduce(layers.begin(), layers.end(), Index(0), plus<>{},
        [](const unique_ptr<Layer>& layer) { return layer->get_parameters_number(); });
}

// Scanned rather than cached. Every caller is setup -- BackPropagation::set,
// the delta layout, the two lifetime planners and setup_arena -- so the scan
// runs a handful of times per training run over a few dozen layers. The cache
// this replaces was invalidated on add_layer, clear, steal_from and from_JSON,
// all structural changes, and so missed Layer::set_is_trainable: freezing a
// layer after any query left the stale range in place, and a fine-tune that
// froze, trained, unfroze and trained again kept back-propagating over the
// frozen range.
Index Network::get_first_trainable_layer_index() const
{
    const auto trainable = ranges::find_if(
        layers, [](const unique_ptr<Layer>& layer) { return layer->get_is_trainable(); });

    return trainable == layers.end() ? -1 : distance(layers.begin(), trainable);
}

Index Network::get_last_trainable_layer_index() const
{
    for (Index i = get_layers_number() - 1; i >= 0; --i)
        if (layers[i]->get_is_trainable()) return i;

    return -1;
}

Index Network::get_layers_number(const string& name) const
{
    return get_layers_number(string_to_layer_type(name));
}

Index Network::get_layers_number(LayerType type) const
{
    return ranges::count_if(layers,
                            [type](const unique_ptr<Layer>& layer) {return layer->get_type() == type;});
}

static bool upload_host_vector(Buffer& buffer, const VectorR& values)
{
    const Index byte_count = values.size() * Index(sizeof(float));

    if (buffer.get_device() == Device::CUDA)
    {
        buffer.resize_bytes(byte_count, Device::CUDA);
        if (byte_count > 0)
        {
            cudaStream_t stream = device::get_compute_stream();
            device::copy_async(buffer.data(), values.data(), byte_count,
                               device::CopyKind::HostToDevice,
                               stream);
            device::synchronize(stream);
        }
        return true;
    }

    buffer.resize_bytes(byte_count, Device::CPU);
    if (byte_count > 0)
        memcpy(buffer.data(), values.data(), static_cast<size_t>(byte_count));
    return false;
}

void Network::set_parameters(const VectorR& new_parameters)
{
    mark_parameters_changed();

    throw_if(fp32_master_released(),
             "Network::set_parameters: the fp32 parameter master was released for "
             "quantized inference; reload the model before replacing parameters.");

    throw_if(new_parameters.size() == 0,
             "Network::set_parameters: refusing to apply an empty parameter vector.");

    const Index expected_size = get_parameters_buffer_size();
    throw_if(expected_size > 0 && new_parameters.size() != expected_size,
             "Network::set_parameters: size mismatch (got {}, expected {}). Make sure the network is compiled with the same architecture as the one that produced this snapshot.", new_parameters.size(), expected_size);

    parameters_fp32_inference_storage.resize_bytes(0, Device::CUDA);

    if (upload_host_vector(parameters, new_parameters))
        cast_parameters_to_bf16();

    link_parameters();
}

void Network::set_states(const VectorR& new_states)
{
    const Index expected_size = get_states_buffer_size();

    if (expected_size == 0)
    {
        throw_if(new_states.size() != 0, "Network::set_states: network has no state buffer.");
        return;
    }

    throw_if(new_states.size() != expected_size,
             "Network::set_states: size mismatch (got {}, expected {}).", new_states.size(), expected_size);

    upload_host_vector(states, new_states);

    link_states();
}

void Network::initialize_parameters(void (Operator::*initializer)())
{
    mark_parameters_changed();

    const HostParametersGuard guard(*this);
    const HostStatesGuard states_guard(*this);

    for (const auto& layer : layers)
        for (Operator* op : layer->get_operators())
            (op->*initializer)();
}

namespace
{

TensorView single_input_view(const MatrixR& inputs)
{
    return TensorView(const_cast<float*>(inputs.data()),
                      {inputs.rows(), inputs.cols()}, Type::FP32);
}

TensorView single_input_view(const Tensor3& inputs)
{
    return TensorView(const_cast<float*>(inputs.data()),
                      {inputs.dimension(0), inputs.dimension(1), inputs.dimension(2)},
                      Type::FP32);
}

TensorView single_input_view(const Tensor4& inputs)
{
    return TensorView(const_cast<float*>(inputs.data()),
                      {inputs.dimension(0), inputs.dimension(1), inputs.dimension(2),
                       inputs.dimension(3)},
                      Type::FP32);
}

}

void Network::calculate_outputs(const MatrixR& inputs, MatrixR& outputs)
{
    calculate_outputs(vector<TensorView>{single_input_view(inputs)}, outputs);
}

void Network::calculate_outputs(const Tensor3& inputs, MatrixR& outputs)
{
    calculate_outputs(vector<TensorView>{single_input_view(inputs)}, outputs);
}

void Network::calculate_outputs(const Tensor4& inputs, MatrixR& outputs)
{
    calculate_outputs(vector<TensorView>{single_input_view(inputs)}, outputs);
}

void Network::calculate_outputs(const Tensor3& inputs_1, const Tensor3& inputs_2,
                                      Tensor3& outputs)
{
    if (get_layers_number() == 0)
    {
        outputs = Tensor3();
        return;
    }

    warn_if_stale_configuration();

    const Index batch_size = inputs_1.dimension(0);

    ForwardPropagation forward_propagation(batch_size, this,
                                           ForwardPropagationMode::Inference);

    const vector<TensorView> input_views = {single_input_view(inputs_1),
                                            single_input_view(inputs_2)};

    forward_propagate(input_views, forward_propagation, ForwardPropagationMode::Inference);

    if (!is_gpu())
    {
        outputs = forward_propagation.get_outputs().as_tensor<3>();
        return;
    }

    const TensorView out = forward_propagation.get_outputs();

    throw_if(out.get_shape().get_rank() < 3,
             "calculate_outputs(Tensor3, Tensor3): expected rank-3 output, got rank {}",
             out.get_shape().get_rank());

    const Shape& shape = out.get_shape();

    if (outputs.dimension(0) != shape[0]
        || outputs.dimension(1) != shape[1]
        || outputs.dimension(2) != shape[2])
        outputs.resize(shape[0], shape[1], shape[2]);

    copy_device_to_host_float(out.get_data(), out.get_type(), out.size(),
                              outputs.data(), device::get_compute_stream(),
                              forward_propagation.host_bf16_output_scratch);
}

Tensor3 Network::calculate_outputs(const Tensor3& inputs_1, const Tensor3& inputs_2)
{
    Tensor3 outputs;
    calculate_outputs(inputs_1, inputs_2, outputs);
    return outputs;
}

MatrixR Network::calculate_outputs(const vector<TensorView>& input_views)
{
    if (layers.empty() || input_views.empty()) return {};

    warn_if_stale_configuration();

    const Index batch_size = input_views[0].get_shape()[0];

    if (is_gpu())
    {
        ForwardPropagation forward_propagation(batch_size, this,
                                               ForwardPropagationMode::Inference);
        return calculate_outputs_device(input_views, forward_propagation);
    }

    constexpr Index tile_budget_bytes = Index(1024) * 1024 * 1024;

    const Index row_bytes = max(Index(1), get_aligned_bytes(get_forward_specs(1)));
    const Index tile_rows_max = clamp((tile_budget_bytes / row_bytes) & ~Index(15),
                                      Index(16), Index(65536));

    const bool tileable = batch_size > tile_rows_max
        && ranges::all_of(input_views,
            [batch_size](const TensorView& view)
            {
                return view.get_shape().get_rank() >= 2
                    && view.get_shape()[0] == batch_size
                    && view.is_fp32()
                    && !view.is_cuda();
            });

    if (!tileable)
    {
        ForwardPropagation forward_propagation(batch_size, this,
                                               ForwardPropagationMode::Inference);
        forward_propagate(input_views, forward_propagation, ForwardPropagationMode::Inference);
        return forward_propagation.get_outputs().as_matrix();
    }

    ForwardPropagation tile_propagation(tile_rows_max, this,
                                        ForwardPropagationMode::Inference);
    unique_ptr<ForwardPropagation> tail_propagation;

    MatrixR outputs;

    for (Index start = 0; start < batch_size; start += tile_rows_max)
    {
        const Index rows = min(tile_rows_max, batch_size - start);

        ForwardPropagation* propagation = &tile_propagation;
        if (rows != tile_rows_max)
        {
            tail_propagation = make_unique<ForwardPropagation>(
                rows, this, ForwardPropagationMode::Inference);
            propagation = tail_propagation.get();
        }

        vector<TensorView> tile_views;
        tile_views.reserve(input_views.size());
        for (const TensorView& view : input_views)
        {
            Shape tile_shape = view.get_shape();
            tile_shape.set_dimension(0, rows);
            const Index row_elements = view.size() / batch_size;
            tile_views.emplace_back(view.as<float>() + start * row_elements,
                                    tile_shape, Type::FP32);
        }

        forward_propagate(tile_views, *propagation, ForwardPropagationMode::Inference);

        const TensorView tile_outputs = propagation->get_outputs();
        const Index output_columns = tile_outputs.size() / rows;
        if (outputs.size() == 0)
            outputs.resize(batch_size, output_columns);

        memcpy(outputs.data() + start * output_columns, tile_outputs.get_data(),
               size_t(rows) * size_t(output_columns) * sizeof(float));
    }

    return outputs;
}

void Network::calculate_outputs(const vector<TensorView>& input_views,
                                      MatrixR& outputs)
{
    if (layers.empty() || input_views.empty())
    {
        outputs.resize(0, 0);
        return;
    }

    warn_if_stale_configuration();

    if (!is_gpu())
    {
        outputs = calculate_outputs(input_views);
        return;
    }

    const Index batch_size = input_views[0].get_shape()[0];

    ForwardPropagation forward_propagation(batch_size, this,
                                           ForwardPropagationMode::Inference);

    calculate_outputs_device(input_views, forward_propagation, outputs);
}

MatrixR Network::calculate_outputs(const MatrixR& inputs)
{
    return calculate_outputs(vector<TensorView>{single_input_view(inputs)});
}

MatrixR Network::calculate_outputs(const Tensor3& inputs)
{
    return calculate_outputs(vector<TensorView>{single_input_view(inputs)});
}

MatrixR Network::calculate_outputs(const Tensor4& inputs)
{
    return calculate_outputs(vector<TensorView>{single_input_view(inputs)});
}

void Network::forward_propagate(const vector<TensorView>& input_view,
                                      ForwardPropagation& forward_propagation,
                                      ForwardPropagationMode pass) const
{
    throw_if(parameters.size_in_floats() != get_aligned_size(get_parameter_specs()),
             "Network shapes changed since compile(); call compile() again.");

    const Index first_layer_index = forward_propagation.get_execution_start_layer();
    const Index last_layer_index = get_layers_number() - 1;

#ifdef OPENNN_HAS_CUDA
    if (is_gpu())
    {
        Network* self = const_cast<Network*>(this);

        const bool needs_parameter_device_copy =
            parameters.get_device() != Device::CUDA
            || (!parameters.empty() && !low_precision_storage_ready());

        if (needs_parameter_device_copy)
            self->copy_parameters_device();

        self->copy_states_device();

        vector<TensorView>& device_inputs =
            forward_propagation.staged_inputs;
        device_inputs.assign(input_view.begin(), input_view.end());
        forward_propagation.staged_input_storage.resize(input_view.size());

        const bool uses_bf16_activations =
            activation_dtype(config.training_type) == Type::BF16;
        if (uses_bf16_activations)
            forward_propagation.host_bf16_input_scratch.resize(input_view.size());

        const auto external_input_allows_bf16_cast = [&](size_t input_index)
        {
            const Index external_source = -static_cast<Index>(input_index) - 1;

            for (size_t layer_index = 0; layer_index < source_layers.size(); ++layer_index)
                for (size_t source_index = 0;
                     source_index < source_layers[layer_index].size();
                     ++source_index)
                {
                    if (source_layers[layer_index][source_index] == external_source
                        && !layers[layer_index]->allows_bf16_input_cast(source_index))
                        return false;
                }

            return true;
        };

        cudaStream_t stream = device::get_compute_stream();
        bool inputs_staged = false;

        if (forward_propagation.needs_position_staging())
            forward_propagation.stage_position(stream);

        for (size_t i = 0; i < input_view.size(); ++i)
        {
            const TensorView& source = input_view[i];
            if (source.empty()) continue;
            if (source.is_cuda()) continue;

            throw_if(source.get_device() == Device::Auto,
                     "Network::forward_propagate: input device must be CPU or CUDA.");

            const bool cast_input_to_bf16 = uses_bf16_activations
                                         && source.is_fp32()
                                         && external_input_allows_bf16_cast(i);

            Buffer& input_buffer = forward_propagation.staged_input_storage[i];
            const auto ensure_cuda_capacity = [&](Index required_bytes)
            {
                if (input_buffer.get_device() != Device::CUDA)
                    input_buffer.resize_bytes(required_bytes, Device::CUDA);
                else
                    input_buffer.grow_to(required_bytes);
            };

            if (cast_input_to_bf16)
            {
                const Index n = source.size();
                vector<uint16_t>& bf16_cpu = forward_propagation.host_bf16_input_scratch[i];
                bf16_cpu.resize(size_t(n));
                float_2_bfloat16_host(n, source.as<float>(), bf16_cpu.data());
                ensure_cuda_capacity(n * Index(sizeof(uint16_t)));
                device::copy_async(input_buffer.data(),
                                   bf16_cpu.data(),
                                   size_t(n) * sizeof(uint16_t),
                                   device::CopyKind::HostToDevice,
                                   stream);
            }
            else
            {
                ensure_cuda_capacity(source.byte_size());
                device::copy_async(input_buffer.data(),
                                   source.get_data(),
                                   source.byte_size(),
                                   device::CopyKind::HostToDevice,
                                   stream);
            }

            device_inputs[i] = TensorView(input_buffer.data(),
                                          source.get_shape(),
                                          cast_input_to_bf16 ? Type::BF16 : source.get_type(),
                                          Device::CUDA);
            inputs_staged = true;
        }

        forward_propagate(device_inputs, forward_propagation, pass, first_layer_index, last_layer_index);

        if (inputs_staged)
            device::synchronize(stream);

        return;
    }
#endif

    forward_propagate(input_view, forward_propagation, pass, first_layer_index, last_layer_index);
}

void Network::forward_propagate(const vector<TensorView>& input_view,
                                      ForwardPropagation& forward_propagation,
                                      ForwardPropagationMode pass,
                                      Index first_layer_index,
                                      Index last_layer_index) const
{
    throw_if(pass == ForwardPropagationMode::Training
             && forward_propagation.mode != ForwardPropagationMode::Training,
             "Network::forward_propagate: an inference ForwardPropagation "
             "cannot be used for training.");

    const auto pick_input = [&](size_t input_index) -> const TensorView& {
        throw_if(input_index >= input_view.size(),
                 "Network::forward_propagate: input index {} out of range (have {} inputs). Network wiring expects more inputs than were provided.",
                        input_index, input_view.size());
        return input_view[input_index];
    };

    if (first_layer_index > last_layer_index && last_layer_index >= 0)
    {
        auto& final_inputs = forward_propagation.inputs[size_t(last_layer_index)];
        if (!final_inputs.empty()) final_inputs.front() = pick_input(0);
        return;
    }

    for (const auto& [layer_i, source_j, ext_idx] : forward_propagation.passthrough_overrides)
        if (Index(layer_i) >= first_layer_index)
            forward_propagation.inputs[layer_i][source_j] = pick_input(ext_idx);

    for (Index i = first_layer_index; i <= last_layer_index; ++i)
    {
        const vector<Index>& sources = source_layers[i];
        auto& input_slot = forward_propagation.inputs[i];

        for (size_t source_index = 0; source_index < sources.size(); ++source_index)
        {
            const Index source_layer = sources[source_index];

            if (source_layer < 0)
                input_slot[source_index] = pick_input(size_t(-source_layer - 1));
            else if (source_layer < forward_propagation.get_execution_start_layer())
                input_slot[source_index] = pick_input(source_index);
        }

        if (i == forward_propagation.get_final_output_layer())
            forward_propagation.gather_output_window();

        PROFILE_SCOPE("fwd:" + layers[i]->get_name());
        layers[i]->forward_propagate(forward_propagation, i, pass);

        forward_propagation.inherit_valid_lengths(size_t(i));
    }
}

void Network::forward_propagate(const vector<TensorView>& input_view,
                                      const VectorR& new_parameters,
                                      ForwardPropagation& forward_propagation)
{

    const Device original_parameters_device = parameters.get_device();
    const Index parameters_size = get_parameters_buffer_size();
    VectorR saved_parameters(parameters_size);
    if (parameters.get_device() == Device::CUDA)
    {
        cudaStream_t stream = device::get_compute_stream();
        device::copy_async(saved_parameters.data(), parameters.data(),
                           parameters_size * Index(sizeof(float)),
                           device::CopyKind::DeviceToHost, stream);
        device::synchronize(stream);
    }
    else
        memcpy(saved_parameters.data(), parameters.data(),
               size_t(parameters_size) * sizeof(float));

    set_parameters(new_parameters);
    forward_propagate(input_view, forward_propagation, ForwardPropagationMode::Training);
    set_parameters(saved_parameters);

    if (parameters.get_device() != original_parameters_device)
    {
        if (original_parameters_device == Device::CPU)
            copy_parameters_host();
        else if (original_parameters_device == Device::CUDA)
            copy_parameters_device();
    }
}

Network::ParameterSlotTotals Network::for_each_parameter_slot(
    const function<void(const ParameterSlot&)>& visit,
    const function<void(Layer&)>& begin_layer) const
{
    ParameterSlotTotals totals{};
    Index master_elements{};

    for (const auto& layer : layers)
    {
        if (begin_layer)
            begin_layer(*layer);

        const auto specs = layer->get_parameter_specs();
        const auto quantization = layer->get_parameter_quantization();
        const Layer::TiedWeight tie = layer->get_tied_weight();

        for (size_t i = 0; i < specs.size(); ++i)
        {
            const auto& [shape, dtype] = specs[i];

            ParameterSlot slot;
            slot.layer = layer.get();
            slot.shape = shape;
            slot.dtype = dtype;
            slot.tied = tie.source && i == tie.spec_index;
            slot.master_offset = master_elements;
            slot.bf16_offset = totals.bf16_elements;
            slot.int8_offset = totals.int8_elements;
            slot.fp32_offset = totals.fp32_elements;

            if (!shape.empty() && dtype == Type::INT8 && !slot.tied)
            {
                const Operator::SlotQuantization q =
                    i < quantization.size()
                        ? quantization[i]
                        : Operator::SlotQuantization{};

                throw_if(q.channels <= 0 || shape.size() % q.channels != 0,
                         "Network: INT8 parameter slot without per-channel "
                         "quantization metadata in layer \"{}\".",
                         layer->get_label());

                slot.scale_channels = q.channels;
                slot.scale_axis = q.axis;
            }

            if (visit)
                visit(slot);

            if (shape.empty())
                continue;

            const Index aligned = get_aligned_size(shape.size());
            master_elements += aligned;

            if (slot.tied)
                continue;

            if (dtype == Type::INT8)
            {
                totals.int8_elements += aligned;
                totals.fp32_elements += get_aligned_size(slot.scale_channels);
            }
            else if (dtype == Type::BF16)
            {
                totals.bf16_elements += aligned;
            }
            else
            {
                totals.fp32_elements += aligned;
            }
        }
    }

    return totals;
}

void Network::allocate_compact_parameter_storage(const ParameterSlotTotals& totals)
{
    parameters_bf16_mirror.resize_bytes(
        totals.bf16_elements * Index(sizeof(bfloat16)), Device::CUDA);
    parameters_fp32_inference_storage.resize_bytes(
        totals.fp32_elements * Index(sizeof(float)), Device::CUDA);
    parameters_int8_storage.resize_bytes(totals.int8_elements, Device::CUDA);
    parameters_bf16_mirror_compact = true;
}

#ifdef OPENNN_HAS_CUDA

void Network::use_compact_parameter_storage()
{
    void* compact_storage = parameters_bf16_mirror.data();
    if (!compact_storage) compact_storage = parameters_int8_storage.data();
    if (!compact_storage) compact_storage = parameters_fp32_inference_storage.data();

    throw_if(!compact_storage,
             "Network: compact inference parameter storage is empty.");

    // The view keeps the master's logical size, which forward_propagate checks
    // against the layer specs; it is computed from the specs here because a
    // network compiled for the loader never had the master.
    const Index master_bytes = get_aligned_bytes(get_parameter_specs(), Type::FP32);
    parameters.resize_bytes(0, Device::CPU);
    parameters.set_view(compact_storage, master_bytes, Device::CUDA);
    link_parameters();
    activate_transposed_inference_weights();
}

#endif

vector<string> Network::get_layer_labels() const
{
    vector<string> layer_labels(layers.size());
    ranges::transform(layers, layer_labels.begin(),
                      [](const unique_ptr<Layer>& layer) { return layer->get_label(); });
    return layer_labels;
}

void Network::link_parameters()
{
    linked_gradient_base = nullptr;

    const ParameterStorage storage = get_parameter_storage();

    const bool low_precision_live = storage == ParameterStorage::DeviceMasterWithMirror
                                 || storage == ParameterStorage::DeviceCompact;

    float* fp32_base = parameters.as<float>();

    // The compact fp32 storage only exists once the master has gone; before
    // that the master is the fp32 source and this buffer is empty.
    float* fp32_inference_base =
        storage == ParameterStorage::DeviceCompact
        && !parameters_fp32_inference_storage.empty()
        ? parameters_fp32_inference_storage.as<float>()
        : nullptr;

    bfloat16* bf16_mirror_base = low_precision_live && !parameters_bf16_mirror.empty()
        ? parameters_bf16_mirror.as<bfloat16>()
        : nullptr;

    int8_t* int8_base = low_precision_live && !parameters_int8_storage.empty()
        ? parameters_int8_storage.as<int8_t>()
        : nullptr;

    Layer* current_layer = nullptr;

    for_each_parameter_slot([&](const ParameterSlot& slot)
    {
        auto& param_views = slot.layer->get_parameter_views();
        auto& param_scales = slot.layer->get_parameter_scales();

        if (slot.shape.empty())
        {
            param_views.emplace_back();
            param_scales.emplace_back();
            return;
        }

        const Type expected_type =
            slot.dtype == Type::INT8 && int8_base != nullptr ? Type::INT8
            : slot.dtype == Type::BF16 && bf16_mirror_base != nullptr ? Type::BF16
            : Type::FP32;

        if (slot.tied)
        {
            const Layer::TiedWeight tie = slot.layer->get_tied_weight();
            const auto& source_views = tie.source->get_parameter_views();
            throw_if(source_views.size() <= tie.source_spec_index
                     || source_views[tie.source_spec_index].empty(),
                     "Network::link_parameters: tied weight source is not linked.");
            const TensorView& source = source_views[tie.source_spec_index];
            throw_if(source.size() != slot.shape.size(),
                     "Network::link_parameters: tied weight sizes do not match.");
            throw_if(source.get_type() != expected_type,
                     "Network::link_parameters: tied weight dtype mismatch "
                     "(the source table must be stored in the consumer's compute dtype).");

            param_views.emplace_back(source);

            const auto& source_scales = tie.source->get_parameter_scales();
            param_scales.emplace_back(source_scales.size() > tie.source_spec_index
                                      ? source_scales[tie.source_spec_index]
                                      : TensorView{});
            return;
        }

        float* const fp32_slot = fp32_base ? fp32_base + slot.master_offset : nullptr;

        void* slot_ptr = fp32_slot;
        Type view_type = Type::FP32;
        Device view_device = parameters.get_device();
        TensorView scale_view;

        if (slot.dtype == Type::INT8 && int8_base != nullptr)
        {
            throw_if(fp32_inference_base == nullptr,
                     "Network::link_parameters: INT8 parameters require compact FP32 scale storage.");

            slot_ptr = int8_base + slot.int8_offset;
            view_type = Type::INT8;
            view_device = Device::CUDA;
            scale_view = TensorView(fp32_inference_base + slot.fp32_offset,
                                    Shape{slot.scale_channels}, Type::FP32, Device::CUDA);
        }
        else if (slot.dtype == Type::BF16 && bf16_mirror_base != nullptr)
        {
            slot_ptr = bf16_mirror_base
                + (parameters_bf16_mirror_compact ? slot.bf16_offset : slot.master_offset);
            view_type = Type::BF16;
            view_device = Device::CUDA;
        }
        else if (fp32_inference_base != nullptr)
        {
            float* const compact_slot = fp32_inference_base + slot.fp32_offset;
            throw_if(!is_aligned(compact_slot),
                     "Network::link_parameters: unaligned compact fp32 parameter memory.");

            slot_ptr = compact_slot;
            view_type = Type::FP32;
            view_device = Device::CUDA;
        }
        else
        {
            throw_if(!is_aligned(fp32_slot),
                     "Network::link_parameters: unaligned parameter memory.");
        }

        param_views.emplace_back(slot_ptr, slot.shape, view_type, view_device);
        param_scales.emplace_back(scale_view);
    },
    [&](Layer& layer)
    {
        if (current_layer) current_layer->redistribute_parameters_to_operators();
        layer.get_parameter_views().clear();
        layer.get_parameter_scales().clear();
        current_layer = &layer;
    });

    if (current_layer) current_layer->redistribute_parameters_to_operators();
}

void Network::link_gradients(const Buffer& gradient) const
{
    void* const base = gradient.data();

    if (!base || base == linked_gradient_base) return;

    float* pointer = static_cast<float*>(base);

    for (const auto& layer : layers)
        pointer = layer->link_gradients(pointer, gradient.get_device());

    linked_gradient_base = base;
}

void Network::link_gradients(
    const span<const TensorView> layer_gradients) const
{
    throw_if(layer_gradients.size() != layers.size(),
             "Network::link_gradients: got {} layer-gradient views for "
             "{} layers.",
             layer_gradients.size(), layers.size());

    for(size_t i = 0; i < layers.size(); ++i)
    {
        const TensorView& storage = layer_gradients[i];
        const Index expected_elements =
            get_aligned_size(layers[i]->get_parameter_specs());

        throw_if(storage.size() != expected_elements,
                 "Network::link_gradients: layer {} has {} gradient "
                 "elements, expected {}.",
                 i, storage.size(), expected_elements);

        float* const begin = storage.empty()
            ? nullptr
            : storage.as<float>();
        float* const end = layers[i]->link_gradients(
            begin, storage.empty() ? parameters.get_device()
                                   : storage.get_device());

        const float* const expected_end = expected_elements > 0
            ? begin + expected_elements
            : begin;
        throw_if(end != expected_end,
                 "Network::link_gradients: layer {} linked an unexpected "
                 "gradient extent.", i);
    }

    // A later contiguous layout must relink even if its base happens to equal
    // one of the arena-backed layer slices.
    linked_gradient_base = nullptr;
}

void Network::link_states()
{
    const Device state_device = states.empty()
        ? parameters.get_device()
        : states.get_device();

    link_states(state_device);
}

void Network::link_states(Device device)
{
    float* state_pointer = states.as<float>();

    for (auto& layer : layers)
        state_pointer = layer->link_states(state_pointer, device);
}

#ifdef OPENNN_HAS_CUDA

void Network::copy_parameters_device()
{
    throw_if(config.device != Device::CUDA,
             "Network::copy_parameters_device: the network is compiled for the CPU.");

    if (parameters.empty())
        return clear_low_precision_parameter_storage();

    if (fp32_master_released())
    {
        const bool bf16_released = config.training_type == Type::BF16 && !parameters_bf16_mirror.empty();
        const bool int8_released = config.training_type == Type::INT8 && !parameters_int8_storage.empty();
        throw_if(!bf16_released && !int8_released,
                 "Network::copy_parameters_device: parameters are a non-owning view.");
        return link_parameters();
    }

    if (config.training_type == Type::INT8)
    {
        throw_if(parameters.get_device() != Device::CPU || !parameters.owns_memory(),
                 "Network::copy_parameters_device: INT8 inference requires "
                 "a host FP32 master to quantize.");
        return upload_parameters_int8_inference();
    }

    cudaStream_t stream = device::get_compute_stream();
    parameters.migrate_to(Device::CUDA, stream);

    if (config.training_type == Type::BF16)
    {
        parameters_bf16_mirror.resize_bytes(parameters.size_in_floats() * Index(sizeof(bfloat16)), Device::CUDA);
        parameters_bf16_mirror_compact = false;
        parameters_fp32_inference_storage.resize_bytes(0, Device::CUDA);
        parameters_int8_storage.resize_bytes(0, Device::CUDA);
        cast_parameters_to_bf16();
    }
    else
        clear_low_precision_parameter_storage();

    link_parameters();
}

void Network::cast_parameters_to_bf16()
{
    if (parameters_bf16_mirror.empty() || parameters.empty() || !parameters.owns_memory()) return;

    cast_fp32_to_bf16(parameters.size_in_floats(),
                           parameters.as<float>(),
                           parameters_bf16_mirror.as<bfloat16>());
}

void Network::release_bf16_fp32_parameter_master_for_inference()
{
    const bool can_release_parameter_master =
        config.training_type == Type::BF16
        && parameters.get_device() == Device::CUDA
        && !parameters.empty()
        && !parameters_bf16_mirror.empty()
        && parameters.owns_memory();

    if (!can_release_parameter_master) return;

    const auto specs = get_parameter_specs();

    Index fp32_keep_floats = 0;
    for (const auto& layer_specs : specs)
        for (const auto& [shape, dtype] : layer_specs)
            if (!shape.empty() && dtype != Type::BF16)
                fp32_keep_floats += get_aligned_size(shape.size());

    if (fp32_keep_floats > 0)
    {
        parameters_fp32_inference_storage.resize_bytes(fp32_keep_floats * Index(sizeof(float)), Device::CUDA);

        cudaStream_t stream = device::get_compute_stream();
        float* const source_base = parameters.as<float>();
        float* const destination_base = parameters_fp32_inference_storage.as<float>();

        Index source_offset = 0;
        Index destination_offset = 0;

        for (const auto& layer_specs : specs)
            for (const auto& [shape, dtype] : layer_specs)
            {
                if (shape.empty()) continue;

                const Index aligned = get_aligned_size(shape.size());
                if (dtype != Type::BF16)
                {
                    device::copy_async(destination_base + destination_offset,
                                       source_base + source_offset,
                                       aligned * Index(sizeof(float)),
                                       device::CopyKind::DeviceToDevice,
                                       stream);
                    destination_offset += aligned;
                }
                source_offset += aligned;
            }

        device::synchronize(stream);
        memory_debug::record("parameters",
                             "fp32_compact_inference",
                             parameters_fp32_inference_storage.byte_size(),
                             "bf16_release");
    }
    else
    {
        parameters_fp32_inference_storage.resize_bytes(0, Device::CUDA);
    }

    const Index fp32_master_bytes = parameters.byte_size();
    parameters.resize_bytes(0, Device::CUDA);
    parameters.set_view(parameters_bf16_mirror.data(),
                        fp32_master_bytes,
                        Device::CUDA);
    link_parameters();
}

void Network::upload_parameters_bf16_inference()
{
    const bool can_upload_low_precision_parameters =
        config.device == Device::CUDA
        && is_one_of(config.training_type, Type::BF16, Type::INT8)
        && !parameters.empty()
        && parameters.get_device() == Device::CPU
        && parameters.owns_memory();

    if (!can_upload_low_precision_parameters)
        return copy_parameters_device();

    cudaStream_t stream = device::get_compute_stream();
    const float* const host_fp32 = parameters.as<float>();

    const ParameterSlotTotals totals = for_each_parameter_slot({});
    allocate_compact_parameter_storage(totals);
    uint16_t* const mirror = parameters_bf16_mirror.as<uint16_t>();
    float* const fp32_compact = parameters_fp32_inference_storage.as<float>();
    int8_t* const int8_storage = parameters_int8_storage.as<int8_t>();

    vector<uint16_t> host_bf16;
    vector<int8_t> host_int8;
    vector<float> host_scales;

    for_each_parameter_slot([&](const ParameterSlot& slot)
    {
        if (slot.shape.empty() || slot.tied) return;

        const Index size = slot.shape.size();
        const float* const source = host_fp32 + slot.master_offset;

        if (slot.dtype == Type::INT8 && int8_storage)
        {
            const Index channels = slot.scale_channels;
            const Index row_length = size / channels;

            host_scales.assign(size_t(channels), 0.0f);
            for (Index i = 0; i < size; ++i)
            {
                const Index channel = quantization_channel(i, row_length, channels, slot.scale_axis);
                host_scales[size_t(channel)] = max(host_scales[size_t(channel)], abs(source[i]));
            }
            finalize_int8_scales(host_scales);

            host_int8.resize(size_t(size));
            quantize_int8_host(source, size, 0, row_length, channels, slot.scale_axis,
                               host_scales.data(), host_int8.data());

            device::copy_async(int8_storage + slot.int8_offset, host_int8.data(),
                               size, Device::CPU, Device::CUDA, stream);
            device::copy_async(fp32_compact + slot.fp32_offset, host_scales.data(),
                               channels * Index(sizeof(float)), Device::CPU, Device::CUDA, stream);
            device::synchronize(stream);
        }
        else if (slot.dtype == Type::BF16 && mirror)
        {
            host_bf16.resize(static_cast<size_t>(size));
            ranges::transform(span<const float>(source, static_cast<size_t>(size)),
                              host_bf16.begin(), float_to_bfloat16_host);
            device::copy_async(mirror + slot.bf16_offset, host_bf16.data(),
                               size * Index(sizeof(uint16_t)), Device::CPU, Device::CUDA, stream);
            device::synchronize(stream);
        }
        else if (fp32_compact)
            device::copy_async(fp32_compact + slot.fp32_offset, source,
                               size * Index(sizeof(float)), Device::CPU, Device::CUDA, stream);
    });
    device::synchronize(stream);

    use_compact_parameter_storage();
}

void Network::upload_parameters_int8_inference()
{
    throw_if(config.training_type != Type::INT8,
             "Network::upload_parameters_int8_inference: "
             "the network must be compiled with an INT8 configuration.");
    upload_parameters_bf16_inference();
}

void Network::activate_transposed_inference_weights()
{
    PROFILE_SCOPE_HOST("load:transpose_inference_weights");

    const bool int8_training = get_training_type() == Type::INT8;

    // The weights are chosen before anything is enqueued so that one scratch,
    // sized for the largest of them, serves every transpose.
    vector<CombinationOperator*> pending;
    Index scratch_bytes = 0;

    for (const auto& layer : layers)
    {
        const bool has_tied_weight = bool(layer->get_tied_weight().source);
        const vector<CombinationOperator*> combinations =
            get_combination_operators(*layer);

        for (CombinationOperator* combination : combinations)
        {
            const TensorView& weight = combination->weights;
            const bool automatic_int8 = int8_training && weight.is_int8()
                && combination->fused_activation == ActivationFunction::Identity;
            const bool configured = combinations.size() == 1
                && combination->transposed_inference_preferred && !weight.is_int8();

            if (has_tied_weight || combination->transposed_inference_active
                || combination->tied_transposed || combination->use_bias
                || (!automatic_int8 && !configured)
                || !weight.is_cuda() || weight.get_rank() != 2)
                continue;

            pending.push_back(combination);
            scratch_bytes = max(scratch_bytes, weight.byte_size());
        }
    }

    if (pending.empty()) return;

    cudaStream_t stream = device::get_compute_stream();

    Buffer scratch{Device::CUDA};
    scratch.resize_bytes(scratch_bytes, Device::CUDA);

    // The stream orders each transpose after the copy that drained the scratch
    // for the previous weight, so one wait at the end replaces the one per
    // weight that put 36 host round trips into a Qwen3 load. The wait also
    // runs on the way out of an exception: the scratch must not be released
    // with a kernel still reading it.
    try
    {
        for (CombinationOperator* combination : pending)
        {
            const TensorView& weight = combination->weights;
            const Shape& shape = weight.get_shape();
            if (weight.is_int8())
                transpose_2d_cuda<int8_t>(shape[0], shape[1],
                                          weight.as<int8_t>(), scratch.as<int8_t>());
            else
                weight.dispatch([&]<typename T>()
                {
                    transpose_2d_cuda<T>(shape[0], shape[1],
                                         weight.as<T>(), scratch.as<T>());
                });
            device::copy_async(weight.get_data(), scratch.data(), weight.byte_size(),
                               device::CopyKind::DeviceToDevice, stream);
            combination->transposed_inference_active = true;
        }
    }
    catch (...)
    {
        device::synchronize(stream);
        throw;
    }

    device::synchronize(stream);
}

void Network::copy_parameters_host()
{
    mark_parameters_changed();

    if (parameters.empty())
        return clear_low_precision_parameter_storage();

    throw_if(fp32_master_released(),
             "Network::copy_parameters_host: the fp32 CUDA parameter master "
             "was released for quantized inference and cannot be copied back.");

    parameters.migrate_to(Device::CPU, device::get_compute_stream());
    clear_low_precision_parameter_storage();

    for (const auto& layer : layers)
        for (CombinationOperator* combination : get_combination_operators(*layer))
            combination->transposed_inference_active = false;

    link_parameters();
}

Network::DeviceResidency Network::get_device_residency() const noexcept
{
    return {parameters.data(),
            parameters_bf16_mirror.data(),
            parameters_fp32_inference_storage.data(),
            parameters_int8_storage.data(),
            states.data()};
}

void Network::copy_states_device()
{
    if (!states.empty())
        states.migrate_to(Device::CUDA, device::get_compute_stream());

    link_states(Device::CUDA);
}

void Network::copy_states_host()
{
    if (!states.empty())
        states.migrate_to(Device::CPU, device::get_compute_stream());

    link_states(Device::CPU);
}

void Network::calculate_outputs_device(const vector<TensorView>& input_views_cpu,
                                             ForwardPropagation& forward_propagation,
                                             MatrixR& outputs)
{
    forward_propagate(input_views_cpu, forward_propagation, ForwardPropagationMode::Inference);

    const TensorView out_view = forward_propagation.get_outputs();

    const Index batch_size = input_views_cpu[0].get_shape()[0];
    const Index out_cols = out_view.size() / batch_size;

    if (Index(outputs.rows()) != batch_size || Index(outputs.cols()) != out_cols)
        outputs.resize(batch_size, out_cols);

    cudaStream_t stream = device::get_compute_stream();
    copy_device_to_host_float(out_view.get_data(), out_view.get_type(), out_view.size(),
                              outputs.data(), stream,
                              forward_propagation.host_bf16_output_scratch);
}

MatrixR Network::calculate_outputs_device(const vector<TensorView>& input_views_cpu,
                                                ForwardPropagation& forward_propagation)
{
    MatrixR result;
    calculate_outputs_device(input_views_cpu, forward_propagation, result);
    return result;
}

namespace
{

bool same_input_pointers(const vector<TensorView>& inputs,
                         const vector<const void*>& captured)
{
    if (captured.size() != inputs.size()) return false;

    for (size_t i = 0; i < inputs.size(); ++i)
        if (inputs[i].get_data() != captured[i]) return false;

    return true;
}

constexpr Index inference_graph_warmup_calls = 2;

}

TensorView Network::calculate_outputs_resident(const vector<TensorView>& gpu_inputs,
                                                     ForwardPropagation& forward_propagation,
                                                     bool upload_parameters)
{

    if (upload_parameters)
    {
        const DeviceResidency before = get_device_residency();

        copy_parameters_device();
        copy_states_device();

        if (get_device_residency() != before)
            forward_propagation.reset_cuda_graph();
    }

    if (!forward_propagation.use_cuda_graph || forward_propagation.cuda_graph_failed)
    {
        forward_propagate(gpu_inputs, forward_propagation, ForwardPropagationMode::Inference);
        return forward_propagation.get_outputs();
    }

    const cudaStream_t compute = device::get_compute_stream();

    if (forward_propagation.inference_graph_exec)
    {
        if (same_input_pointers(gpu_inputs, forward_propagation.captured_input_pointers))
        {
            PROFILE_SCOPE_HOST("inference:graph_launch");

            if (forward_propagation.position_pinned)
                *forward_propagation.position_pinned.as<int>() =
                    int(forward_propagation.past_length);
            device::launch_graph(forward_propagation.inference_graph_exec, compute);
            return forward_propagation.get_outputs();
        }

        forward_propagate(gpu_inputs, forward_propagation, ForwardPropagationMode::Inference);
        return forward_propagation.get_outputs();
    }

    {
        device::CudaGraphWorkspaceScope workspace_measurement(
            forward_propagation.inference_graph_workspace_requirements);
        forward_propagate(gpu_inputs, forward_propagation, ForwardPropagationMode::Inference);
    }

    if (++forward_propagation.cuda_graph_warmup_calls < inference_graph_warmup_calls)
        return forward_propagation.get_outputs();

    if (env_flag_enabled("OPENNN_GRAPH_TIMING"))
    {
        forward_propagation.cuda_graph_failed = true;
        logging::warning() << "Network::calculate_outputs_resident: OPENNN_GRAPH_TIMING "
                "event timing cannot be captured; continuing eager.\n";
        return forward_propagation.get_outputs();
    }

    const bool profiler_was_enabled = profiler::is_enabled();
    profiler::set_enabled(false);

    forward_propagation.prepare_cuda_graph_workspaces();
    const device::GraphWorkspaceViews graph_workspace_views =
        forward_propagation.get_cuda_graph_workspace_views();

    try
    {
        device::synchronize(compute);
        device::CudaAllocationGrowthGuard growth_guard(true);
        device::CudaGraphWorkspaceScope stable_workspaces(
            forward_propagation.inference_graph_workspace_requirements,
            &graph_workspace_views);

        // One eager pass with the stable workspaces already installed, so the
        // captured pass is the second time every library sees these exact
        // pointers. The two warmup calls above ran against the thread-local
        // workspaces, so a vendor call that initialises something lazily on
        // first sight of a buffer -- cuDNN's RNN engine was the suspect -- was
        // still doing it inside the capture, where an allocation is illegal and
        // surfaces as an opaque status from whichever call triggered it. This
        // also decides that question for good: if capture still fails after an
        // identical eager call succeeded, first touch was not the cause.
        forward_propagate(gpu_inputs, forward_propagation, ForwardPropagationMode::Inference);
        device::synchronize(compute);

        device::StreamCapture capture(compute);

        forward_propagate(gpu_inputs, forward_propagation, ForwardPropagationMode::Inference);

        capture.end(forward_propagation.inference_graph_exec);

        forward_propagation.captured_input_pointers.resize(gpu_inputs.size());
        ranges::transform(gpu_inputs, forward_propagation.captured_input_pointers.begin(),
                          [](const auto& gpu_input) { return gpu_input.get_data(); });
    }
    catch (const exception& capture_error)
    {
        if (forward_propagation.cuda_graph_workspaces_need_growth())
        {

            forward_propagation.inference_graph_exec.reset();
            forward_propagation.captured_input_pointers.clear();
            forward_propagation.cuda_graph_warmup_calls =
                inference_graph_warmup_calls - 1;
        }
        else
        {
            forward_propagation.reset_cuda_graph();
            forward_propagation.cuda_graph_failed = true;
            logging::warning() << "Network::calculate_outputs_resident: cuda graph capture "
                    "unavailable (" << capture_error.what() << "); continuing eager.\n";
        }
    }

    if (forward_propagation.inference_graph_exec)
        release_thread_workspaces();

    profiler::set_enabled(profiler_was_enabled);

    return forward_propagation.get_outputs();
}

#else

void Network::copy_parameters_device() OPENNN_CUDA_STUB_BODY(Network::copy_parameters_device)

void Network::cast_parameters_to_bf16() OPENNN_CUDA_STUB_BODY(Network::cast_parameters_to_bf16)

void Network::release_bf16_fp32_parameter_master_for_inference()
{
}

void Network::upload_parameters_bf16_inference()
{
}

void Network::upload_parameters_int8_inference() OPENNN_CUDA_STUB_BODY(Network::upload_parameters_int8_inference)

void Network::copy_parameters_host()
{
    link_parameters();
}

void Network::copy_states_device() OPENNN_CUDA_STUB_BODY(Network::copy_states_device)

void Network::copy_states_host()
{
    link_states(Device::CPU);
}

MatrixR Network::calculate_outputs_device(const vector<TensorView>&,
                                                ForwardPropagation&) OPENNN_CUDA_STUB_BODY(Network::calculate_outputs_device)

void Network::calculate_outputs_device(const vector<TensorView>&,
                                             ForwardPropagation&,
                                             MatrixR&) OPENNN_CUDA_STUB_BODY(Network::calculate_outputs_device)

TensorView Network::calculate_outputs_resident(const vector<TensorView>&,
                                                     ForwardPropagation&,
                                                     bool) OPENNN_CUDA_STUB_BODY(Network::calculate_outputs_resident)

#endif

}
