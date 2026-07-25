//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   N E U R A L   N E T W O R K   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "registry.h"
#include "tensor_types.h"
#include "neural_network.h"
#include "profiler.h"
#include "dense_layer.h"
#include "scaling_layer.h"
#include "flatten_layer.h"
#include "convolutional_layer.h"
#include "image_processing.h"
#include "addition_layer.h"
#include "embedding_layer.h"
#include "tokenizer_layer.h"
#include "variable.h"
#include "string_utilities.h"
#include "forward_propagation.h"
#include "back_propagation.h"
#include "model_expression.h"
#include "memory_debug.h"

#include <algorithm>
#include "kernel.cuh"

namespace opennn
{

static vector<Index> string_to_source_indices(const string&);
static void validate_source_indices(const vector<Index>&, Index, Index);
static void validate_source_arity(const Layer&, const vector<Index>&, Index);

NeuralNetwork::NeuralNetwork()
{
    clear();
}

NeuralNetwork::NeuralNetwork(const filesystem::path& file_name)
{
    load(file_name);
}

void NeuralNetwork::add_layer(unique_ptr<Layer> layer, const vector<Index>& sources)
{
    const Index old_layers_number = get_layers_number() - 1;

    if (!layers.empty()) validate_type(layers.back()->get_type());

    const vector<Index> resolved_sources = sources.empty()
        ? vector<Index>{old_layers_number}
        : sources;

    validate_source_indices(resolved_sources, ssize(layers), ssize(layers));
    validate_source_arity(*layer, resolved_sources, ssize(layers));

    layers.push_back(move(layer));

    source_layers.push_back(resolved_sources);

    first_trainable_cache_ = -1;
    last_trainable_cache_  = -1;
}

void NeuralNetwork::compile()
{
    compile(Configuration::instance().resolve().device);
}

void NeuralNetwork::compile(const Device device)
{
    if (get_layers_number() == 0) return;

    config = Configuration::instance().resolve();
    config.device = device;
    if (device != Device::CUDA) config.training_type = Type::FP32;

    stale_configuration_warned = false;

    for (auto& layer : layers)
    {
        layer->set_compute_device(get_device());
        layer->set_compute_dtype(get_training_type());
    }

    parameters.resize_bytes(get_aligned_bytes(get_parameter_specs(), Type::FP32), Device::CPU);
    parameters.setZero();

    parameters_bf16_mirror.resize_bytes(0, Device::CUDA);
    parameters_fp32_inference_storage.resize_bytes(0, Device::CUDA);

    link_parameters();

    states.resize_bytes(get_states_size() * Index(sizeof(float)), Device::CPU);
    states.setZero();

    link_states();

}

void NeuralNetwork::validate_type(LayerType type) const
{
    throw_if(type == LayerType::Bounding,
             "No layers can be added after a bounding layer.\n");
}

void NeuralNetwork::warn_if_stale_configuration() const
{
    if (stale_configuration_warned
        || config.generation == Configuration::instance().get_generation())
        return;

    stale_configuration_warned = true;

    cerr << "Warning: Configuration::set() was called after this network was compiled, "
            "so it has no effect on it. The network keeps the settings resolved at "
            "compile() time; call Configuration::set() before constructing the network.\n";
}

bool NeuralNetwork::has(const string& name) const
{
    return has(string_to_layer_type(name));
}

bool NeuralNetwork::has(LayerType type) const
{
    return ranges::any_of(layers,
                          [type](const unique_ptr<Layer>& layer) {return layer->get_type() == type;});
}

vector<string> NeuralNetwork::get_input_feature_names() const
{
    return get_variable_feature_names(input_variables);
}

vector<string> NeuralNetwork::get_output_feature_names() const
{
    return get_variable_feature_names(output_variables);
}

const unique_ptr<Layer>& NeuralNetwork::get_layer(const string& label) const
{
    auto it = ranges::find_if(layers,
                              [&label](const unique_ptr<Layer>& layer) { return layer->get_label() == label; });

    if (it != layers.end())
        return *it;

    throw runtime_error("Layer not found in neural network");
}

Index NeuralNetwork::get_layer_index(const string& new_label) const
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

const Layer* NeuralNetwork::get_first(const string& name) const
{
    return get_first(string_to_layer_type(name));
}

const Layer* NeuralNetwork::get_first(LayerType type) const
{
    auto it = ranges::find_if(layers,
                              [type](const unique_ptr<Layer>& layer) { return layer->get_type() == type; });

    return it != layers.end() ? it->get() : nullptr;
}

Layer* NeuralNetwork::get_first(const string& name)
{
    return get_first(string_to_layer_type(name));
}

Layer* NeuralNetwork::get_first(LayerType type)
{
    return const_cast<Layer*>(static_cast<const NeuralNetwork*>(this)->get_first(type));
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

void NeuralNetwork::set_input_names(const vector<string>& new_input_names)
{
    if (input_variables.empty() && !new_input_names.empty())
        return define_variables_from_names(input_variables, new_input_names, VariableRole::Input);

    set_variable_names(input_variables, new_input_names);
}

void NeuralNetwork::set_output_names(const vector<string>& new_output_names)
{
    if (output_variables.empty() && !new_output_names.empty())
        return define_variables_from_names(output_variables, new_output_names, VariableRole::Target);

    set_variable_names(output_variables, new_output_names);
}

void NeuralNetwork::set_input_shape(const Shape& new_input_shape)
{
    if (get_features_number(input_variables) != new_input_shape.size())
    {
        input_variables.assign(1, Variable());
        input_variables[0].name = "input";
        input_variables[0].role = VariableRole::Input;
        input_variables[0].type = VariableType::Numeric;
        input_variables[0].features = new_input_shape.size();
    }

    if (Layer* scaling = get_first(LayerType::Scaling))
        scaling->set_input_shape(new_input_shape);

    layers[get_first_trainable_layer_index()]->set_input_shape(new_input_shape);

    const Index layers_number = get_layers_number();
    for (Index i = 0; i < layers_number; ++i)
    {
        const vector<Index>& sources = source_layers[i];
        if (sources.size() == 1 && sources[0] >= 0)
            layers[i]->set_input_shape(layers[sources[0]]->get_output_shape());
    }
}

void NeuralNetwork::clear()
{
    layers.clear();

    source_layers.clear();

    input_variables.clear();

    output_variables.clear();

    first_trainable_cache_ = -1;
    last_trainable_cache_  = -1;
}

void NeuralNetwork::steal_from(NeuralNetwork& src)
{
    clear();
    layers           = std::move(src.layers);
    source_layers    = std::move(src.source_layers);
    input_variables  = std::move(src.input_variables);
    output_variables = std::move(src.output_variables);
    first_trainable_cache_ = src.first_trainable_cache_;
    last_trainable_cache_  = src.last_trainable_cache_;
    src.first_trainable_cache_ = -1;
    src.last_trainable_cache_  = -1;
    link_parameters();
}

static vector<Index> string_to_source_indices(const string& text)
{
    return parse_number_list<Index>(text, "SourceLayers");
}

static void validate_source_indices(const vector<Index>& sources, Index layer_index, Index layers_count)
{
    for (Index src : sources)
    {
        if (src < 0) continue;
        throw_if(src >= layers_count || src >= layer_index,
                 "NeuralNetwork::set_source_layers: source index {} is not a previous layer for layer {}.", src, layer_index);
    }
}

static void validate_source_arity(const Layer& layer,
                                  const vector<Index>& sources,
                                  Index layer_index)
{
    if (const auto* addition = dynamic_cast<const Addition*>(&layer);
        addition && ssize(sources) != addition->get_inputs_number())
        throw runtime_error(format("NeuralNetwork::set_source_layers: Addition layer {} expects {} sources, got {}.", layer_index, addition->get_inputs_number(), sources.size()));

    if (const auto* convolutional = dynamic_cast<const Convolutional*>(&layer);
        convolutional && convolutional->get_residual() && ssize(sources) != 2)
        throw runtime_error(format("NeuralNetwork::set_source_layers: residual Convolutional layer {} expects 2 sources, got {}.", layer_index, sources.size()));
}

void NeuralNetwork::set_source_layers(const vector<vector<Index>>& new_source_layers)
{
    throw_if(ssize(new_source_layers) != ssize(layers),
             "NeuralNetwork::set_source_layers: outer size ({}) must match layers count ({}).", new_source_layers.size(), layers.size());

    for (Index i = 0; i < ssize(new_source_layers); ++i)
    {
        validate_source_indices(new_source_layers[i], i, ssize(layers));
        validate_source_arity(*layers[i], new_source_layers[i], i);
    }

    source_layers = new_source_layers;
}

void NeuralNetwork::set_source_layers(const Index layer_index, const vector<Index>& new_sources)
{
    throw_if(layer_index < 0 || layer_index >= ssize(layers),
             "NeuralNetwork::set_source_layers: layer index {} out of range.", layer_index);

    validate_source_indices(new_sources, layer_index, ssize(layers));
    validate_source_arity(*layers[layer_index], new_sources, layer_index);

    source_layers[layer_index] = new_sources;
}

void NeuralNetwork::set_source_layers(const string& layer_label,
                                      const vector<string>& new_source_labels)
{
    vector<Index> new_sources(new_source_labels.size());

    ranges::transform(new_source_labels, new_sources.begin(),
                      [this](const string& label) { return get_layer_index(label); });

    set_source_layers(get_layer_index(layer_label), new_sources);
}

void NeuralNetwork::set_source_layers(const string& layer_label,
                                      initializer_list<string> new_source_labels_list)
{
    set_source_layers(layer_label, vector<string>(new_source_labels_list));
}

void NeuralNetwork::set_source_layers(const string& layer_label, const string& new_source_label)
{
    const Index layer_index = get_layer_index(layer_label);

    set_source_layers(layer_index, {get_layer_index(new_source_label)});
}

Index NeuralNetwork::get_inputs_number() const
{
    if (layers.empty())
        return 0;

    if (get_first(LayerType::Embedding))
        return get_layer(0)->get_inputs_number();

    for (const LayerType type : {LayerType::Recurrent, LayerType::LongShortTermMemory})
        if (const Layer* layer = get_first(type))
            return layer->get_input_shape()[1];

    return layers[0]->get_input_shape().size();
}

Index NeuralNetwork::get_outputs_number() const
{
    if (layers.empty()) return 0;

    return layers.back()->get_output_shape().size();
}

Shape NeuralNetwork::get_input_shape() const
{
    if (layers.empty())
        return {};

    return layers[0]->get_input_shape();
}

Shape NeuralNetwork::get_output_shape() const
{
    if (layers.empty())
        return {};

    return layers.back()->get_output_shape();
}

ActivationFunction NeuralNetwork::get_output_activation() const
{
    const Index last_index = get_last_trainable_layer_index();
    if (last_index < 0 || static_cast<size_t>(last_index) >= layers.size())
        return ActivationFunction::Identity;

    return layers[last_index]->get_output_activation();
}

Index NeuralNetwork::get_parameters_number() const
{
    return transform_reduce(layers.begin(), layers.end(), Index(0), plus<>{},
        [](const unique_ptr<Layer>& layer) { return layer->get_parameters_number(); });
}

Index NeuralNetwork::get_first_trainable_layer_index() const
{
    if (first_trainable_cache_ >= 0) return first_trainable_cache_;

    auto it = ranges::find_if(layers,
                              [](const unique_ptr<Layer>& layer) { return layer->get_is_trainable(); });

    throw_if(it == layers.end(),
             "The neural network has no trainable layers: get_first_trainable_layer_index.");

    first_trainable_cache_ = distance(layers.begin(), it);
    return first_trainable_cache_;
}

Index NeuralNetwork::get_last_trainable_layer_index() const
{
    if (last_trainable_cache_ >= 0) return last_trainable_cache_;

    const Index layers_number = get_layers_number();
    for (Index i = layers_number - 1; i >= 0; --i)
        if (layers[i]->get_is_trainable())
            return last_trainable_cache_ = i;

    throw runtime_error("The neural network has no trainable layers: get_last_trainable_layer_index");
}

Index NeuralNetwork::get_layers_number(const string& name) const
{
    return get_layers_number(string_to_layer_type(name));
}

Index NeuralNetwork::get_layers_number(LayerType type) const
{
    return ranges::count_if(layers,
                            [type](const unique_ptr<Layer>& layer) {return layer->get_type() == type;});
}

static bool upload_host_vector(Buffer& buffer, const VectorR& values)
{
    const Index byte_count = values.size() * Index(sizeof(float));

    if (buffer.device_type == Device::CUDA)
    {
        buffer.resize_bytes(byte_count, Device::CUDA);
        if (byte_count > 0)
        {
            cudaStream_t stream = Backend::get_compute_stream();
            device::copy_async(buffer.data, values.data(), byte_count,
                               device::CopyKind::HostToDevice,
                               stream);
            device::synchronize(stream);
        }
        return true;
    }

    buffer.resize_bytes(byte_count, Device::CPU);
    if (byte_count > 0)
        memcpy(buffer.data, values.data(), static_cast<size_t>(byte_count));
    return false;
}

void NeuralNetwork::set_parameters(const VectorR& new_parameters)
{
    throw_if(new_parameters.size() == 0,
             "NeuralNetwork::set_parameters: refusing to apply an empty parameter vector.");

    const Index expected_size = get_parameters_size();
    throw_if(expected_size > 0 && new_parameters.size() != expected_size,
             "NeuralNetwork::set_parameters: size mismatch (got {}, expected {}). Make sure the network is compiled with the same architecture as the one that produced this snapshot.", new_parameters.size(), expected_size);

    parameters_fp32_inference_storage.resize_bytes(0, Device::CUDA);

    if (upload_host_vector(parameters, new_parameters))
        cast_parameters_to_bf16();

    link_parameters();
}

void NeuralNetwork::set_states(const VectorR& new_states)
{
    const Index expected_size = get_states_buffer_size();

    if (expected_size == 0)
    {
        throw_if(new_states.size() != 0, "NeuralNetwork::set_states: network has no state buffer.");
        return;
    }

    throw_if(new_states.size() != expected_size,
             "NeuralNetwork::set_states: size mismatch (got {}, expected {}).", new_states.size(), expected_size);

    upload_host_vector(states, new_states);

    link_states();
}

void NeuralNetwork::initialize_parameters(void (Operator::*initializer)())
{
    const HostParametersGuard guard(*this);

    for (const auto& layer : layers)
        for (Operator* op : layer->get_operators())
            (op->*initializer)();
}

void NeuralNetwork::set_parameters_random()
{
    initialize_parameters(&Operator::set_parameters_random);
}

void NeuralNetwork::set_parameters_glorot()
{
    initialize_parameters(&Operator::set_parameters_glorot);
}

void NeuralNetwork::set_parameters_pytorch()
{
    initialize_parameters(&Operator::set_parameters_pytorch);
}

Tensor3 NeuralNetwork::calculate_outputs(const Tensor3& inputs_1, const Tensor3& inputs_2)
{
    const Index layers_number = get_layers_number();

    if (layers_number == 0)
        return {};

    warn_if_stale_configuration();

    const Index batch_size = inputs_1.dimension(0);

    ForwardPropagation forward_propagation(batch_size, this,
                                           ForwardPropagationMode::Inference);

    const vector<TensorView> input_views = {TensorView(const_cast<float*>(inputs_1.data()), {{inputs_1.dimension(0), inputs_1.dimension(1), inputs_1.dimension(2)}}),
                                            TensorView(const_cast<float*>(inputs_2.data()), {{inputs_2.dimension(0), inputs_2.dimension(1), inputs_2.dimension(2)}})};

    if (is_gpu())
    {
        const MatrixR result_matrix = calculate_outputs_device(input_views, forward_propagation);
        const TensorView out = forward_propagation.get_outputs();
        throw_if(out.shape.rank < 3,
                 "calculate_outputs(Tensor3, Tensor3): expected rank-3 output, got rank {}",
                        out.shape.rank);
        Tensor3 result(out.shape[0], out.shape[1], out.shape[2]);
        memcpy(result.data(), result_matrix.data(),
                    size_t(result.size()) * sizeof(float));
        return result;
    }

    forward_propagate(input_views, forward_propagation, false);

    return forward_propagation.get_outputs().as_tensor<3>();
}

MatrixR NeuralNetwork::calculate_outputs(const vector<TensorView>& input_views)
{
    if (layers.empty() || input_views.empty()) return {};

    warn_if_stale_configuration();

    const Index batch_size = input_views[0].shape[0];

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
        && all_of(input_views.begin(), input_views.end(),
                  [batch_size](const TensorView& view)
                  {
                      return view.shape.rank >= 2
                          && view.shape[0] == batch_size
                          && view.is_fp32()
                          && !view.is_cuda();
                  });

    if (!tileable)
    {
        ForwardPropagation forward_propagation(batch_size, this,
                                               ForwardPropagationMode::Inference);
        forward_propagate(input_views, forward_propagation, false);
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
            Shape tile_shape = view.shape;
            tile_shape[0] = rows;
            const Index row_elements = view.size() / batch_size;
            tile_views.emplace_back(view.as<float>() + start * row_elements,
                                    tile_shape, Type::FP32);
        }

        forward_propagate(tile_views, *propagation, false);

        const TensorView tile_outputs = propagation->get_outputs();
        const Index output_columns = tile_outputs.size() / rows;
        if (outputs.size() == 0)
            outputs.resize(batch_size, output_columns);

        memcpy(outputs.data() + start * output_columns, tile_outputs.data,
               size_t(rows) * size_t(output_columns) * sizeof(float));
    }

    return outputs;
}

MatrixR NeuralNetwork::calculate_outputs(const MatrixR& inputs)
{
    return calculate_outputs(vector<TensorView>{TensorView(const_cast<float*>(inputs.data()), {inputs.rows(), inputs.cols()}, Type::FP32)});
}

MatrixR NeuralNetwork::calculate_outputs(const Tensor3& inputs)
{
    return calculate_outputs(vector<TensorView>{TensorView(const_cast<float*>(inputs.data()), {inputs.dimension(0), inputs.dimension(1), inputs.dimension(2)}, Type::FP32)});
}

MatrixR NeuralNetwork::calculate_outputs(const Tensor4& inputs)
{
    return calculate_outputs(vector<TensorView>{TensorView(const_cast<float*>(inputs.data()), {inputs.dimension(0), inputs.dimension(1), inputs.dimension(2), inputs.dimension(3)}, Type::FP32)});
}

void NeuralNetwork::forward_propagate(const vector<TensorView>& input_view,
                                      ForwardPropagation& forward_propagation,
                                      bool is_training) const
{
    throw_if(parameters.size_in_floats() != get_aligned_size(get_parameter_specs()),
             "Network shapes changed since compile(); call compile() again.");

    Index first_layer_index = 0;
    if (is_training || forward_propagation.inputs_pre_scaled)
        while (first_layer_index < get_layers_number()
               && layers[first_layer_index]->get_type() == LayerType::Scaling)
            ++first_layer_index;
    const Index last_layer_index = get_layers_number() - 1;

#ifdef OPENNN_HAS_CUDA
    if (is_gpu())
    {
        NeuralNetwork* self = const_cast<NeuralNetwork*>(this);

        if (parameters.device_type != Device::CUDA
            || (config.training_type == Type::BF16 && !parameters.empty() && parameters_bf16_mirror.empty()))
            self->copy_parameters_device();

        self->copy_states_device();

        vector<TensorView> input_views_device = input_view;
        forward_propagation.device_input_buffers.resize(input_view.size());
        forward_propagation.host_bf16_input_scratch.resize(input_view.size());

        const auto input_feeds_token_ids = [&](size_t input_index)
        {
            const Index external_source = -static_cast<Index>(input_index) - 1;

            for (size_t layer_index = 0; layer_index < source_layers.size(); ++layer_index)
                for (const Index source : source_layers[layer_index])
                    if (source == external_source
                        && (layers[layer_index]->get_type() == LayerType::Embedding
                            || layers[layer_index]->get_type() == LayerType::Tokenizer))
                        return true;

            return false;
        };

        cudaStream_t stream = Backend::get_compute_stream();
        bool staged_inputs = false;

        if (has(LayerType::GroupedQueryAttention))
            forward_propagation.stage_position(stream);

        for (size_t i = 0; i < input_view.size(); ++i)
        {
            const TensorView& source = input_view[i];
            if (source.empty()) continue;
            if (source.is_cuda()) continue;

            throw_if(source.device == Device::Auto,
                     "NeuralNetwork::forward_propagate: input device must be CPU or CUDA.");

            const bool cast_input_to_bf16 = config.training_type == Type::BF16
                                         && source.is_fp32()
                                         && !input_feeds_token_ids(i);

            Buffer& input_buffer = forward_propagation.device_input_buffers[i];

            if (cast_input_to_bf16)
            {
                const Index n = source.size();
                vector<uint16_t>& bf16_cpu = forward_propagation.host_bf16_input_scratch[i];
                bf16_cpu.resize(size_t(n));
                const float* src = source.as<float>();
                uint16_t* dst = bf16_cpu.data();
                #pragma omp parallel for if(n > 4096)
                for (Index j = 0; j < n; ++j)
                    dst[j] = static_cast<uint16_t>(bit_cast<uint32_t>(src[j]) >> 16);
                input_buffer.resize_bytes(n * Index(sizeof(uint16_t)), Device::CUDA);
                device::copy_async(input_buffer.data,
                                   bf16_cpu.data(),
                                   size_t(n) * sizeof(uint16_t),
                                   device::CopyKind::HostToDevice,
                                   stream);
                input_views_device[i].type = Type::BF16;
            }
            else
            {
                input_buffer.resize_bytes(source.byte_size(), Device::CUDA);
                device::copy_async(input_buffer.data,
                                   source.data,
                                   source.byte_size(),
                                   device::CopyKind::HostToDevice,
                                   stream);
            }

            input_views_device[i].data = input_buffer.data;
            input_views_device[i].device = Device::CUDA;
            staged_inputs = true;
        }

        forward_propagate(input_views_device, forward_propagation, is_training, first_layer_index, last_layer_index);

        if (staged_inputs)
            device::synchronize(stream);

        return;
    }
#endif

    forward_propagate(input_view, forward_propagation, is_training, first_layer_index, last_layer_index);
}

void NeuralNetwork::forward_propagate(const vector<TensorView>& input_view,
                                      ForwardPropagation& forward_propagation,
                                      bool is_training,
                                      Index first_layer_index,
                                      Index last_layer_index) const
{
    throw_if(is_training
             && forward_propagation.mode != ForwardPropagationMode::Training,
             "NeuralNetwork::forward_propagate: an inference ForwardPropagation "
             "cannot be used for training.");

    const auto pick_input = [&](size_t input_index) -> const TensorView& {
        throw_if(input_index >= input_view.size(),
                 "NeuralNetwork::forward_propagate: input index {} out of range (have {} input views). Network wiring expects more inputs than were provided.",
                        input_index, input_view.size());
        return input_view[input_index];
    };

    for (const auto& [layer_i, source_j, ext_idx] : forward_propagation.passthrough_overrides)
        if (Index(layer_i) >= first_layer_index)
            forward_propagation.input_views[layer_i][source_j] = pick_input(ext_idx);

    for (Index i = first_layer_index; i <= last_layer_index; ++i)
    {
        const vector<Index>& sources = source_layers[i];
        auto& input_slot = forward_propagation.input_views[i];

        for (size_t source_index = 0; source_index < sources.size(); ++source_index)
        {
            const Index source_layer = sources[source_index];

            if (source_layer < 0)
                input_slot[source_index] = pick_input(size_t(-source_layer - 1));
            else if ((is_training || forward_propagation.inputs_pre_scaled)
                     && source_layer < first_layer_index)
                input_slot[source_index] = pick_input(source_index);
        }

        PROFILE_SCOPE("fwd:" + layers[i]->get_name());
        layers[i]->forward_propagate(forward_propagation, i, is_training);
    }
}

void NeuralNetwork::forward_propagate(const vector<TensorView>& input_view,
                                      const VectorR& new_parameters,
                                      ForwardPropagation& forward_propagation)
{

    const Device original_parameters_device = parameters.device_type;
    const Index parameters_size = get_parameters_size();
    VectorR saved_parameters(parameters_size);
    if (parameters.device_type == Device::CUDA)
    {
        cudaStream_t stream = Backend::get_compute_stream();
        device::copy_async(saved_parameters.data(), parameters.data,
                           parameters_size * Index(sizeof(float)),
                           device::CopyKind::DeviceToHost, stream);
        device::synchronize(stream);
    }
    else
        memcpy(saved_parameters.data(), parameters.data,
               size_t(parameters_size) * sizeof(float));

    set_parameters(new_parameters);
    forward_propagate(input_view, forward_propagation, true);
    set_parameters(saved_parameters);

    if (parameters.device_type != original_parameters_device)
    {
        if (original_parameters_device == Device::CPU)
            copy_parameters_host();
        else if (original_parameters_device == Device::CUDA)
            copy_parameters_device();
    }
}

MatrixR NeuralNetwork::calculate_text_outputs(const Tensor<string, 1>& input_documents)
{
    const auto* tokenizer_layer = dynamic_cast<const Tokenizer*>(get_first(LayerType::Tokenizer));

    throw_if(!tokenizer_layer,
             "calculate_text_outputs: network has no Tokenizer layer.\n");

    const TokenizerOperator* tokenizer = tokenizer_layer->get_tokenizer();

    throw_if(!tokenizer || tokenizer->get_vocabulary_size() == 0,
             "calculate_text_outputs: the Tokenizer layer has no vocabulary; call set_tokenizer() first.\n");

    const Index sequence_length = tokenizer_layer->get_output_shape()[0];
    const Index batch_size = input_documents.size();

    MatrixR inputs = MatrixR::Zero(batch_size, sequence_length);

    for (Index i = 0; i < batch_size; ++i)
    {
        const vector<Index> ids = tokenizer->encode_sequence(input_documents.data()[i], sequence_length);

        for (Index j = 0; j < min(ssize(ids), sequence_length); ++j)
            inputs(i, j) = float(ids[size_t(j)]);
    }

    return calculate_outputs(inputs);
}

void NeuralNetwork::to_JSON(JsonWriter& printer) const
{

    const HostStatesGuard guard(*const_cast<NeuralNetwork*>(this),
                                parameters.device_type == Device::CUDA);

    const Index inputs_number = get_inputs_number();
    const Index layers_number = get_layers_number();
    const Index outputs_number = get_outputs_number();

    const auto write_variables_array = [&printer](const vector<Variable>& variables, const char* tag)
    {
        printer.begin_array(tag);

        for (size_t i = 0; i < variables.size(); ++i)
        {
            const Variable& variable = variables[i];

            printer.begin_array_object();
            add_json_field(printer, "Index", i + 1);
            add_json_field(printer, "Text", variable.name);

            if (variable.features > 1)
                add_json_field(printer, "Features", variable.features);

            if (variable.is_categorical())
                add_json_field(printer, "Categories", vector_to_string(variable.categories, ";"));

            printer.end_array_object();
        }

        printer.end_array();
    };

    printer.open_element("NeuralNetwork");

    printer.open_element("Inputs");
    add_json_field(printer, "InputsNumber", inputs_number);
    write_variables_array(input_variables, "Input");
    printer.close_element();

    printer.open_element("Layers");
    add_json_field(printer, "LayersNumber", layers_number);

    printer.begin_array("Items");
    for (Index i = 0; i < layers_number; ++i)
    {
        printer.begin_array_object();
        layers[i]->to_JSON(printer);
        printer.end_array_object();
    }
    printer.end_array();

    printer.open_element("SourceLayers");
    printer.begin_array("SourceLayer");
    for (size_t i = 0; i < source_layers.size(); ++i)
    {
        printer.begin_array_object();
        add_json_field(printer, "LayerIndex", i);
        add_json_field(printer, "Text", vector_to_string(source_layers[i]));
        printer.end_array_object();
    }
    printer.end_array();
    printer.close_element();

    printer.close_element();

    printer.open_element("Outputs");
    const Index outputs_count = has(LayerType::Embedding)
                              ? outputs_number
                              : get_features_number(output_variables);
    add_json_field(printer, "OutputsNumber", outputs_count);
    write_variables_array(output_variables, "Output");
    printer.close_element();
    printer.close_element();
}

void NeuralNetwork::from_JSON(const JsonDocument& document)
{
    [[maybe_unused]] static const bool _layers_registered = []() { register_classes(); return true; }();

    const Json* neural_network_element = get_json_root(document, "NeuralNetwork");

    const auto read_variables_array = [](const Json* parent, const char* tag,
                                         vector<Variable>& variables, const char* role)
    {
        const Json* items = parent->find(tag);
        const Index entries_number = (items && items->is_array())
                                   ? Index(items->array_value.size())
                                   : 0;

        variables.assign(size_t(entries_number), Variable());

        for_json_items(parent, tag, entries_number, [&](Index i, const Json* element) {
            Variable& variable = variables[size_t(i)];

            variable.name = read_json_string(element, "Text");
            variable.set_role(role);
            variable.features = element->find("Features") ? read_json_index(element, "Features") : 1;

            if (element->find("Categories"))
            {
                variable.type = VariableType::Categorical;
                variable.categories = get_tokens(read_json_string(element, "Categories"), ";");
            }
        });
    };

    if (const Json* inputs_element = neural_network_element->find("Inputs"); inputs_element)
        read_variables_array(inputs_element, "Input", input_variables, "Input");

    const Json* layers_container = neural_network_element->find("Layers");
    throw_if(!layers_container, "layers container is nullptr.");

    const Index layers_number = read_json_index(layers_container, "LayersNumber");

    layers.clear();
    source_layers.clear();
    layers.reserve(layers_number);
    first_trainable_cache_ = -1;
    last_trainable_cache_  = -1;

    const Json* items_array = layers_container->find("Items");
    if (items_array && items_array->is_array())
    {
        for (const Json& item : items_array->array_value)
        {
            if (!item.is_object() || item.object_value.empty()) continue;

            const string& tag_name = item.object_value[0].first;

            unique_ptr<Layer> layer = Registry<Layer>::instance().create(tag_name);
            throw_if(!layer,
                     "Layer '{}' not found in Registry. "
                            "Ensure the layer file is linked and REGISTER macro is used.",
                            tag_name);

            JsonDocument layer_doc;
            layer_doc.root = item;
            layer->from_JSON(layer_doc);

            layers.push_back(move(layer));
        }
    }

    source_layers.resize(layers.size());

    if (const Json* source_layers_element = layers_container->find("SourceLayers"); source_layers_element)
    {
        const Json* indices_array = source_layers_element->find("SourceLayer");
        if (indices_array && indices_array->is_array())
        {
            for (const Json& entry : indices_array->array_value)
            {
                const long layer_index = read_json_index(&entry, "LayerIndex");
                const string text   = read_json_string(&entry, "Text");
                if (text.empty()) continue;

                throw_if(layer_index < 0 || layer_index >= ssize(layers),
                         "NeuralNetwork::from_JSON: SourceLayer index {} out of range (have {} layers).", layer_index, layers.size());

                set_source_layers(layer_index, string_to_source_indices(text));
            }
        }
    }

    if (const Json* outputs_element = neural_network_element->find("Outputs"); outputs_element)
        read_variables_array(outputs_element, "Output", output_variables, "Target");

    compile();

    if (items_array && items_array->is_array())
    {
        Index layer_index = 0;
        for (const Json& item : items_array->array_value)
        {
            if (!item.is_object() || item.object_value.empty()) continue;
            if (layer_index >= ssize(layers)) break;

            JsonDocument layer_doc;
            layer_doc.root = item;
            layers[layer_index]->load_state_from_JSON(layer_doc);
            ++layer_index;
        }
    }

    const Json* parameters_element = neural_network_element->find("Parameters");
    const string parameters_text   = parameters_element ? read_json_string(parameters_element, "Values") : string();
    if (parameters_text.empty()) return;

    VectorR json_parameters;
    string_to_vector(parameters_text, json_parameters);

    if (json_parameters.size() != parameters.size_in_floats())
    {
        cout << "Warning: JSON parameter size (" << json_parameters.size()
             << ") differs from Compiled size (" << parameters.size_in_floats() << ").\n";
    }

    const Index elements_to_copy = min(parameters.size_in_floats(), json_parameters.size());

    const HostParametersGuard guard(*this);
    std::copy(json_parameters.data(), json_parameters.data() + elements_to_copy, parameters.as<float>());
}

void NeuralNetwork::save(const filesystem::path& file_name) const
{
    ofstream file(file_name);

    if (!file.is_open())
        return;

    JsonWriter printer;
    to_JSON(printer);
    file << printer.c_str();

    filesystem::path binary_path = file_name;
    binary_path.replace_extension(".bin");

    save_parameters_binary(binary_path);
}

static ofstream open_binary_output(const filesystem::path& file_name)
{
    ofstream file(file_name, ios::binary);

    throw_if(!file.is_open(),
             "Cannot open binary file for writing: {}\n", file_name.string());

    return file;
}

static void write_binary_payload(ofstream& file, const filesystem::path& file_name,
                                 const void* data, Index byte_count)
{
    if (byte_count > 0)
        file.write(static_cast<const char*>(data), byte_count);

    throw_if(!file, "Error writing binary file: {}\n", file_name.string());
}

static ifstream open_binary_input(const filesystem::path& file_name,
                                  uintmax_t expected_bytes, const char* caller)
{
    ifstream file(file_name, ios::binary);

    throw_if(!file.is_open(),
             "Cannot open binary file: {}\n", file_name.string());

    const uintmax_t file_bytes = filesystem::file_size(file_name);
    throw_if(file_bytes != expected_bytes,
             "NeuralNetwork::{}: size mismatch for {} (got {} bytes, expected {} bytes).",
                    caller,
                    file_name.string(),
                    file_bytes,
                    expected_bytes);

    return file;
}

void NeuralNetwork::save_parameters_binary(const filesystem::path& file_name) const
{
    ofstream file = open_binary_output(file_name);

    const HostParametersGuard guard(*const_cast<NeuralNetwork*>(this));

    write_binary_payload(file, file_name, parameters.as<float>(),
                         parameters.size_in_floats() * Index(sizeof(float)));
}

void NeuralNetwork::save_states_binary(const filesystem::path& file_name) const
{
    ofstream file = open_binary_output(file_name);

    const HostStatesGuard guard(*const_cast<NeuralNetwork*>(this));

    write_binary_payload(file, file_name, states.data, states.bytes);
}

void NeuralNetwork::load(const filesystem::path& file_name)
{
    clear();

    from_JSON(load_json_file(file_name));

    filesystem::path binary_path = file_name;
    binary_path.replace_extension(".bin");

    if (filesystem::exists(binary_path))
        load_parameters_binary(binary_path);
}

void NeuralNetwork::load_parameters_binary(const filesystem::path& file_name)
{
    const Index parameters_number = parameters.size_in_floats();

    ifstream file = open_binary_input(file_name,
                                      uintmax_t(parameters_number) * sizeof(float),
                                      "load_parameters_binary");

    {
        const HostParametersGuard guard(*this);
        file.read(reinterpret_cast<char*>(parameters.as<float>()), parameters_number * sizeof(float));
    }

    throw_if(!file, "Error reading binary file: {}", file_name.string());
}

void NeuralNetwork::load_states_binary(const filesystem::path& file_name)
{
    ifstream file = open_binary_input(file_name, uintmax_t(states.bytes),
                                      "load_states_binary");

    {
        const HostStatesGuard guard(*this);

        if (states.bytes > 0)
            file.read(reinterpret_cast<char*>(states.data), states.bytes);

        if (!guard.was_on_device)
            link_states();
    }

    throw_if(!file, "Error reading binary file: {}", file_name.string());
}

vector<string> NeuralNetwork::get_layer_labels() const
{
    vector<string> layer_labels(layers.size());
    ranges::transform(layers, layer_labels.begin(),
                      [](const unique_ptr<Layer>& layer) { return layer->get_label(); });
    return layer_labels;
}

void NeuralNetwork::link_parameters()
{
    float* fp32_base = parameters.as<float>();
    float* fp32_inference_base =
        parameters.device_type == Device::CUDA
        && !parameters.owns
        && !parameters_fp32_inference_storage.empty()
        ? parameters_fp32_inference_storage.as<float>()
        : nullptr;

    bfloat16* bf16_mirror_base = (parameters.device_type == Device::CUDA && !parameters_bf16_mirror.empty())
        ? parameters_bf16_mirror.as<bfloat16>()
        : nullptr;

    Index offset = 0;
    Index fp32_inference_offset = 0;
    Index bf16_mirror_offset = 0;

    for (auto& layer : layers)
    {
        const auto specs = layer->get_parameter_specs();
        auto& param_views = layer->get_parameter_views();
        param_views.clear();

        const Layer::TiedWeight tie = layer->get_tied_weight();

        for (size_t spec_index = 0; spec_index < specs.size(); ++spec_index)
        {
            const auto& [shape, slot_dtype] = specs[spec_index];
            if (shape.empty())
            {
                param_views.emplace_back();
                continue;
            }

            const Index aligned = get_aligned_size(shape.size());

            const Type expected_type = slot_dtype == Type::BF16 && bf16_mirror_base != nullptr
                ? Type::BF16 : Type::FP32;

            if (tie.source && spec_index == tie.spec_index)
            {

                const auto& source_views = tie.source->get_parameter_views();
                throw_if(source_views.size() <= tie.source_spec_index
                         || source_views[tie.source_spec_index].empty(),
                         "NeuralNetwork::link_parameters: tied weight source is not linked.");
                const TensorView& source = source_views[tie.source_spec_index];
                throw_if(source.size() != shape.size(),
                         "NeuralNetwork::link_parameters: tied weight sizes do not match.");
                throw_if(source.type != expected_type,
                         "NeuralNetwork::link_parameters: tied weight dtype mismatch "
                         "(the source table must be stored in the consumer's compute dtype).");

                param_views.emplace_back(source);
                offset += aligned;
                continue;
            }

            float* const fp32_slot = fp32_base ? fp32_base + offset : nullptr;

            void* slot_ptr = fp32_slot;
            Type view_type = Type::FP32;
            Device view_device = parameters.device_type;

            if (slot_dtype == Type::BF16 && bf16_mirror_base != nullptr)
            {
                slot_ptr = bf16_mirror_base + (parameters_bf16_mirror_compact ? bf16_mirror_offset : offset);
                view_type = Type::BF16;
                view_device = Device::CUDA;
                bf16_mirror_offset += aligned;
            }
            else if (fp32_inference_base != nullptr)
            {
                float* const compact_slot = fp32_inference_base + fp32_inference_offset;
                throw_if(!is_aligned(compact_slot),
                         "NeuralNetwork::link_parameters: unaligned compact fp32 parameter memory.");

                slot_ptr = compact_slot;
                view_type = Type::FP32;
                view_device = Device::CUDA;
                fp32_inference_offset += aligned;
            }
            else
            {
                throw_if(!is_aligned(fp32_slot),
                         "NeuralNetwork::link_parameters: unaligned parameter memory.");
            }

            param_views.emplace_back(slot_ptr, shape, view_type, view_device);
            offset += aligned;
        }

        layer->redistribute_parameters_to_operators();
    }
}

void NeuralNetwork::link_states()
{
    const Device state_device = states.empty()
        ? parameters.device_type
        : states.device_type;

    link_states(state_device);
}

void NeuralNetwork::link_states(Device device)
{
    float* state_pointer = states.as<float>();

    for (auto& layer : layers)
        state_pointer = layer->link_states(state_pointer, device);
}

#ifdef OPENNN_HAS_CUDA

void NeuralNetwork::copy_parameters_device()
{
    if (parameters.empty())
    {
        parameters_bf16_mirror.resize_bytes(0, Device::CUDA);
        parameters_fp32_inference_storage.resize_bytes(0, Device::CUDA);
        return;
    }

    if (parameters.device_type == Device::CUDA && !parameters.owns)
    {
        throw_if(config.training_type != Type::BF16 || parameters_bf16_mirror.empty(),
                 "NeuralNetwork::copy_parameters_device: parameters are a non-owning view.");
        link_parameters();
        return;
    }

    cudaStream_t stream = Backend::get_compute_stream();
    parameters.migrate_to(Device::CUDA, stream);

    if (config.training_type == Type::BF16)
    {
        parameters_bf16_mirror.resize_bytes(parameters.size_in_floats() * Index(sizeof(bfloat16)), Device::CUDA);
        parameters_bf16_mirror_compact = false;
        cast_parameters_to_bf16();
    }
    else
    {
        parameters_bf16_mirror.resize_bytes(0, Device::CUDA);
        parameters_bf16_mirror_compact = false;
    }
    parameters_fp32_inference_storage.resize_bytes(0, Device::CUDA);

    link_parameters();
}

void NeuralNetwork::cast_parameters_to_bf16()
{
    if (parameters_bf16_mirror.empty()) return;
    if (parameters.empty())      return;
    if (parameters.device_type == Device::CUDA && !parameters.owns) return;

    cast_fp32_to_bf16(parameters.size_in_floats(),
                           parameters.as<float>(),
                           parameters_bf16_mirror.as<bfloat16>());
}

#ifdef OPENNN_HAS_CUDA

static inline uint16_t float_to_bfloat16_host(float value)
{
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    bits += 0x7FFFu + ((bits >> 16) & 1u);
    return static_cast<uint16_t>(bits >> 16);
}
#endif

void NeuralNetwork::upload_parameters_bf16_inference()
{
#ifdef OPENNN_HAS_CUDA
    if (config.device != Device::CUDA
        || config.training_type != Type::BF16
        || parameters.empty()
        || parameters.device_type != Device::CPU
        || !parameters.owns)
    {
        copy_parameters_device();
        return;
    }

    cudaStream_t stream = Backend::get_compute_stream();
    const float* const host_fp32 = parameters.as<float>();

    Index bf16_keep = 0;
    Index fp32_keep = 0;
    for (const auto& layer : layers)
    {
        const Layer::TiedWeight tie = layer->get_tied_weight();
        const auto layer_specs = layer->get_parameter_specs();
        for (size_t spec_index = 0; spec_index < layer_specs.size(); ++spec_index)
        {
            const auto& [shape, dtype] = layer_specs[spec_index];
            if (shape.empty() || (tie.source && spec_index == tie.spec_index)) continue;
            (dtype == Type::BF16 ? bf16_keep : fp32_keep) += get_aligned_size(shape.size());
        }
    }

    parameters_bf16_mirror.resize_bytes(bf16_keep * Index(sizeof(bfloat16)), Device::CUDA);
    parameters_fp32_inference_storage.resize_bytes(fp32_keep * Index(sizeof(float)), Device::CUDA);
    parameters_bf16_mirror_compact = true;
    uint16_t* const mirror = bf16_keep > 0 ? parameters_bf16_mirror.as<uint16_t>() : nullptr;
    float* const fp32_compact = fp32_keep > 0 ? parameters_fp32_inference_storage.as<float>() : nullptr;

    vector<uint16_t> host_bf16;
    Index offset = 0;
    Index bf16_offset = 0;
    Index fp32_offset = 0;

    for (const auto& layer : layers)
    {
        const Layer::TiedWeight tie = layer->get_tied_weight();
        const auto layer_specs = layer->get_parameter_specs();
        for (size_t spec_index = 0; spec_index < layer_specs.size(); ++spec_index)
        {
            const auto& [shape, dtype] = layer_specs[spec_index];
            if (shape.empty()) continue;

            const Index size = shape.size();
            const Index aligned = get_aligned_size(size);

            if (tie.source && spec_index == tie.spec_index)
            {
                offset += aligned;
                continue;
            }

            if (dtype == Type::BF16 && mirror)
            {
                host_bf16.resize(static_cast<size_t>(size));
                for (Index i = 0; i < size; ++i)
                    host_bf16[static_cast<size_t>(i)] = float_to_bfloat16_host(host_fp32[offset + i]);
                device::copy_async(mirror + bf16_offset, host_bf16.data(),
                                   size * Index(sizeof(uint16_t)), Device::CPU, Device::CUDA, stream);
                device::synchronize(stream);
                bf16_offset += aligned;
            }
            else if (fp32_compact)
            {
                device::copy_async(fp32_compact + fp32_offset, host_fp32 + offset,
                                   size * Index(sizeof(float)), Device::CPU, Device::CUDA, stream);
                fp32_offset += aligned;
            }

            offset += aligned;
        }
    }
    device::synchronize(stream);

    const Index master_bytes = parameters.bytes;
    parameters.resize_bytes(0, Device::CPU);
    parameters.set_view(parameters_bf16_mirror.data, master_bytes, Device::CUDA);

    link_parameters();
#endif
}

void NeuralNetwork::copy_parameters_host()
{
    if (parameters.empty())
    {
        parameters_bf16_mirror.resize_bytes(0, Device::CUDA);
        parameters_fp32_inference_storage.resize_bytes(0, Device::CUDA);
        return;
    }

    throw_if(parameters.device_type == Device::CUDA && !parameters.owns,
             "NeuralNetwork::copy_parameters_host: the fp32 CUDA parameter master "
             "was released for BF16 inference and cannot be copied back.");

    parameters.migrate_to(Device::CPU, Backend::get_compute_stream());
    parameters_bf16_mirror.resize_bytes(0, Device::CUDA);
    parameters_bf16_mirror_compact = false;
    parameters_fp32_inference_storage.resize_bytes(0, Device::CUDA);

    link_parameters();
}

void NeuralNetwork::copy_states_device()
{
    if (!states.empty())
        states.migrate_to(Device::CUDA, Backend::get_compute_stream());

    link_states(Device::CUDA);
}

void NeuralNetwork::copy_states_host()
{
    if (!states.empty())
        states.migrate_to(Device::CPU, Backend::get_compute_stream());

    link_states(Device::CPU);
}

MatrixR NeuralNetwork::calculate_outputs_device(const vector<TensorView>& input_views_cpu,
                                                ForwardPropagation& forward_propagation)
{
    forward_propagate(input_views_cpu, forward_propagation, false);

    const TensorView out_view = forward_propagation.get_outputs();

    const Index batch_size = input_views_cpu[0].shape[0];
    const Index out_cols = out_view.size() / batch_size;
    MatrixR result(batch_size, out_cols);

    cudaStream_t stream = Backend::get_compute_stream();
    copy_device_to_host_float(out_view.data, out_view.type, out_view.size(),
                              result.data(), stream);

    return result;
}

namespace
{

bool same_input_pointers(const vector<TensorView>& inputs,
                         const vector<const void*>& captured)
{
    if (captured.size() != inputs.size()) return false;

    for (size_t i = 0; i < inputs.size(); ++i)
        if (inputs[i].data != captured[i]) return false;

    return true;
}

constexpr Index inference_graph_warmup_calls = 2;

}

TensorView NeuralNetwork::calculate_outputs_resident(const vector<TensorView>& gpu_inputs,
                                                     ForwardPropagation& forward_propagation,
                                                     bool upload_parameters)
{

    if (upload_parameters)
    {
        copy_parameters_device();
        copy_states_device();

        forward_propagation.reset_cuda_graph();
    }

    if (!forward_propagation.use_cuda_graph || forward_propagation.cuda_graph_failed)
    {
        forward_propagate(gpu_inputs, forward_propagation, false);
        return forward_propagation.get_outputs();
    }

    const cudaStream_t compute = Backend::get_compute_stream();

    if (forward_propagation.inference_graph_exec)
    {
        if (same_input_pointers(gpu_inputs, forward_propagation.captured_input_pointers))
        {
            PROFILE_SCOPE_HOST("inference:graph_launch");

            if (forward_propagation.position_pinned)
                *static_cast<int*>(forward_propagation.position_pinned) = int(forward_propagation.past_length);
            device::launch_graph(forward_propagation.inference_graph_exec, compute);
            return forward_propagation.get_outputs();
        }

        forward_propagate(gpu_inputs, forward_propagation, false);
        return forward_propagation.get_outputs();
    }

    {
        device::CudaGraphWorkspaceScope workspace_measurement(
            forward_propagation.inference_graph_workspace_requirements);
        forward_propagate(gpu_inputs, forward_propagation, false);
    }

    if (++forward_propagation.cuda_graph_warmup_calls < inference_graph_warmup_calls)
        return forward_propagation.get_outputs();

    if (env_flag_enabled("OPENNN_GRAPH_TIMING"))
    {
        forward_propagation.cuda_graph_failed = true;
        cerr << "NeuralNetwork::calculate_outputs_resident: OPENNN_GRAPH_TIMING "
                "event timing cannot be captured; continuing eager.\n";
        return forward_propagation.get_outputs();
    }

    const bool profiler_was_enabled = ::opennn::enabled();
    ::opennn::enabled() = false;

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
        device::StreamCapture capture(compute);

        forward_propagate(gpu_inputs, forward_propagation, false);

        const device::GraphHandle graph = capture.end();
        device::instantiate_or_update(forward_propagation.inference_graph_exec, graph.get());

        forward_propagation.captured_input_pointers.resize(gpu_inputs.size());
        for (size_t i = 0; i < gpu_inputs.size(); ++i)
            forward_propagation.captured_input_pointers[i] = gpu_inputs[i].data;
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
            cerr << "NeuralNetwork::calculate_outputs_resident: cuda graph capture "
                    "unavailable (" << capture_error.what() << "); continuing eager.\n";
        }
    }

    ::opennn::enabled() = profiler_was_enabled;

    return forward_propagation.get_outputs();
}

#else

void NeuralNetwork::copy_parameters_device()
{
    throw runtime_error("NeuralNetwork::copy_parameters_device requires CUDA support.");
}

void NeuralNetwork::cast_parameters_to_bf16()
{
    throw runtime_error("NeuralNetwork::cast_parameters_to_bf16 requires CUDA support.");
}

void NeuralNetwork::upload_parameters_bf16_inference()
{
}

void NeuralNetwork::copy_parameters_host()
{
    link_parameters();
}

void NeuralNetwork::copy_states_device()
{
    throw runtime_error("NeuralNetwork::copy_states_device requires CUDA support.");
}

void NeuralNetwork::copy_states_host()
{
    link_states(Device::CPU);
}

MatrixR NeuralNetwork::calculate_outputs_device(const vector<TensorView>&,
                                                ForwardPropagation&)
{
    throw runtime_error("NeuralNetwork::calculate_outputs_device requires CUDA support.");
}

TensorView NeuralNetwork::calculate_outputs_resident(const vector<TensorView>&,
                                                     ForwardPropagation&,
                                                     bool)
{
    throw runtime_error("NeuralNetwork::calculate_outputs_resident requires CUDA support.");
}

#endif

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
