//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   N E U R A L   N E T W O R K   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "registry.h"
#include "tensor_utilities.h"
#include "neural_network.h"
#include "profiler.h"
#include "dense_layer.h"
#include "scaling_layer.h"
#include "flatten_layer.h"
#include "cuda_dispatch.h"
#include "convolutional_layer.h"
#include "image_utilities.h"
#include "addition_layer.h"
#include "embedding_layer.h"
#include "variable.h"
#include "string_utilities.h"
#include "forward_propagation.h"
#include "back_propagation.h"
#include "model_expression.h"

namespace opennn
{

NeuralNetwork::NeuralNetwork()
{
    set_default();
}

NeuralNetwork::NeuralNetwork(const filesystem::path& file_name)
{
    load(file_name);
}

void NeuralNetwork::add_layer(unique_ptr<Layer> layer, const vector<Index>& input_indices)
{
    const Index old_layers_number = get_layers_number() - 1;

    if (!layers.empty()) validate_type(layers.back()->get_type());

    layers.push_back(move(layer));

    layer_input_indices.push_back(input_indices.empty()
        ? vector<Index>(1, old_layers_number )
        : input_indices);

    first_trainable_cache_ = -1;
    last_trainable_cache_  = -1;
}

void NeuralNetwork::compile()
{
    if (get_layers_number() == 0) return;

    config = Configuration::instance().resolve();

    for (auto& layer : layers)
        layer->set_compute_dtype(get_training_type());

    parameters.resize_bytes(aligned_total_elements(get_parameter_shapes()) * Index(sizeof(float)),
                            Device::CPU);
    parameters.setZero();

    float* pointer = parameters.as<float>();
    for (auto& layer : layers)
        pointer = layer->link_parameters(pointer);

    states.resize_bytes(get_states_size() * Index(sizeof(float)), Device::CPU);
    states.setZero();

    float* state_pointer = states.as<float>();
    for (auto& layer : layers)
        state_pointer = layer->link_states(state_pointer);

    const Index first = get_first_trainable_layer_index();
    if (first >= 0 && size_t(first) < layers.size())
    {
        const auto& ops = layers[first]->get_operators();
        if (!ops.empty()) ops[0]->input_delta_slots.clear();
    }
}

void NeuralNetwork::validate_type(LayerType type) const
{
    if (type == LayerType::Bounding)
        throw runtime_error("No layers can be added after a bounding layer.\n");
}

bool NeuralNetwork::has(const string& name) const
{
    return has(string_to_layer_type(name));
}

bool NeuralNetwork::has(LayerType type) const
{
    return any_of(layers.begin(), layers.end(),
                  [type](const unique_ptr<Layer>& layer) {return layer->get_type() == type;});
}

static vector<string> get_feature_names_from(const vector<Variable>& variables)
{
    vector<string> feature_names;
    feature_names.reserve(variables.size());

    for (const auto& variable : variables)
    {
        const vector<string> names = variable.get_names();
        feature_names.insert(feature_names.end(), names.begin(), names.end());
    }

    return feature_names;
}

vector<string> NeuralNetwork::get_input_feature_names() const
{
    return get_feature_names_from(input_variables);
}

vector<string> NeuralNetwork::get_output_feature_names() const
{
    return get_feature_names_from(output_variables);
}

const unique_ptr<Layer>& NeuralNetwork::get_layer(const string& label) const
{
    auto it = find_if(layers.begin(), layers.end(),
                      [&label](const unique_ptr<Layer>& layer) { return layer->get_label() == label; });

    if (it != layers.end())
        return *it;

    throw runtime_error("Layer not found in neural network");
}

Index NeuralNetwork::get_layer_index(const string& new_label) const
{
    if (new_label == "Dataset" || new_label == "decoder")
        return -1;

    if (new_label == "input")
        return -2;

    auto it = find_if(layers.begin(), layers.end(),
                      [&new_label](const unique_ptr<Layer>& layer) { return layer->get_label() == new_label; });

    if (it != layers.end())
        return distance(layers.begin(), it);

    throw runtime_error("Layer not found: " + new_label);
}

vector<vector<Index>> NeuralNetwork::get_layer_output_indices() const
{
    const Index layers_number = ssize(layer_input_indices);

    vector<vector<Index>> layer_output_indices(layers_number);

    for (Index i = 0; i < layers_number; ++i)
        for (const Index input_index : layer_input_indices[i])
            if (input_index >= 0)
                layer_output_indices[input_index].push_back(i);

    for (auto& outputs : layer_output_indices)
        if (outputs.empty())
            outputs.push_back(-1);

    return layer_output_indices;
}

const Layer* NeuralNetwork::get_first(const string& name) const
{
    return get_first(string_to_layer_type(name));
}

const Layer* NeuralNetwork::get_first(LayerType type) const
{
    auto it = find_if(layers.begin(), layers.end(),
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

static void set_variable_names(vector<Variable>& variables, const vector<string>& new_names)
{
    Index name_index = 0;
    for (size_t i = 0; i < variables.size(); ++i)
    {
        if (variables[i].is_categorical())
        {
            const size_t num_cats = variables[i].get_categories_number();
            variables[i].categories.assign(new_names.begin() + name_index, new_names.begin() + name_index + num_cats);
            name_index += num_cats;
        }
        else
        {
            variables[i].name = new_names[name_index];
            ++name_index;
        }
    }
}

void NeuralNetwork::set_input_names(const vector<string>& new_input_names)
{
    set_variable_names(input_variables, new_input_names);
}

void NeuralNetwork::set_output_names(const vector<string>& new_output_names)
{
    set_variable_names(output_variables, new_output_names);
}

void NeuralNetwork::set_input_shape(const Shape& new_input_shape)
{
    input_variables.resize(new_input_shape.size());

    if (has(LayerType::Scaling))
        get_first(LayerType::Scaling)->set_input_shape(new_input_shape);

    layers[get_first_trainable_layer_index()]->set_input_shape(new_input_shape);

    const Index layers_number = get_layers_number();
    for (Index i = 0; i < layers_number; ++i)
    {
        const vector<Index>& inputs = layer_input_indices[i];
        if (inputs.size() == 1 && inputs[0] >= 0)
            layers[i]->set_input_shape(layers[inputs[0]]->get_output_shape());
    }
}

void NeuralNetwork::set_default()
{
    layers.clear();

    layer_input_indices.clear();

    input_variables.clear();

    output_variables.clear();
}

void NeuralNetwork::set_layer_input_indices(const string& layer_label,
                                            const vector<string>& new_layer_input_labels)
{
    vector<Index> new_layer_input_indices(new_layer_input_labels.size());

    transform(new_layer_input_labels.begin(), new_layer_input_labels.end(), new_layer_input_indices.begin(),
              [this](const string& label) { return get_layer_index(label); });

    layer_input_indices[get_layer_index(layer_label)] = new_layer_input_indices;
}

void NeuralNetwork::set_layer_input_indices(const string& layer_label,
                                            initializer_list<string> new_layer_input_labels_list)
{
    set_layer_input_indices(layer_label, vector<string>(new_layer_input_labels_list));
}

void NeuralNetwork::set_layer_input_indices(const string& layer_label, const string& new_layer_input_labels)
{
    const Index layer_index = get_layer_index(layer_label);

    layer_input_indices[layer_index] = {get_layer_index(new_layer_input_labels)};
}

Index NeuralNetwork::get_inputs_number() const
{
    if (layers.empty())
        return 0;

    if (has(LayerType::Embedding))
        return get_layer(0)->get_inputs_number();

    if (has(LayerType::Recurrent))
        return get_first(LayerType::Recurrent)->get_input_shape()[1];

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

Activation::Function NeuralNetwork::get_output_activation() const
{
    const Index last_index = get_last_trainable_layer_index();
    if (last_index < 0 || static_cast<size_t>(last_index) >= layers.size())
        return Activation::Function::Identity;

    return layers[last_index]->get_output_activation();
}

Index NeuralNetwork::get_parameters_number() const
{
    Index parameters_number = 0;

    for (const auto& layer : layers)
        parameters_number += layer->get_parameters_number();

    return parameters_number;
}

vector<Index> NeuralNetwork::get_layer_parameter_numbers() const
{
    const Index layers_number = get_layers_number();

    vector<Index> layer_parameter_numbers(layers_number);

    #pragma omp parallel for

    for (int i = 0; i < layers_number; ++i)
        layer_parameter_numbers[i] = layers[i]->get_parameters_number();

    return layer_parameter_numbers;
}

Index NeuralNetwork::get_first_trainable_layer_index() const
{
    if (first_trainable_cache_ >= 0) return first_trainable_cache_;

    auto it = find_if(layers.begin(), layers.end(),
                      [](const unique_ptr<Layer>& layer) { return layer->get_is_trainable(); });

    if (it == layers.end())
        throw runtime_error("The neural network has no trainable layers: get_first_trainable_layer_index.");

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
    return count_if(layers.begin(), layers.end(),
                    [type](const unique_ptr<Layer>& layer) {return layer->get_type() == type;});
}

void NeuralNetwork::set_parameters(const VectorR& new_parameters)
{
    const Index byte_count = new_parameters.size() * Index(sizeof(float));

#ifdef OPENNN_HAS_CUDA
    if (parameters.device_type == Device::CUDA)
    {
        parameters.resize_bytes(byte_count, Device::CUDA);
        if (byte_count > 0)
            CHECK_CUDA(cudaMemcpy(parameters.data, new_parameters.data(), byte_count, cudaMemcpyHostToDevice));
        cast_parameters_to_bf16();
        return;
    }
#endif

    parameters.resize_bytes(byte_count, Device::CPU);
    if (byte_count > 0)
        std::memcpy(parameters.data, new_parameters.data(), static_cast<size_t>(byte_count));
}

void NeuralNetwork::set_parameters_random()
{
#ifdef OPENNN_HAS_CUDA
    const bool was_on_device = (parameters.device_type == Device::CUDA);
    if (was_on_device) copy_parameters_host();
#endif

    for (const auto& layer : layers)
        for (Operator* op : layer->get_operators())
            op->set_parameters_random();

#ifdef OPENNN_HAS_CUDA
    if (was_on_device) copy_parameters_device();
#endif
}

void NeuralNetwork::set_parameters_glorot()
{
    const Index layers_number = get_layers_number();

#ifdef OPENNN_HAS_CUDA
    const bool was_on_device = (parameters.device_type == Device::CUDA);
    if (was_on_device) copy_parameters_host();
#endif

    #pragma omp parallel for
    for (int i = 0; i < layers_number; ++i)
        for (Operator* op : layers[i]->get_operators())
            op->set_parameters_glorot();

#ifdef OPENNN_HAS_CUDA
    if (was_on_device) copy_parameters_device();
#endif
}

Tensor3 NeuralNetwork::calculate_outputs(const Tensor3& inputs_1, const Tensor3& inputs_2)
{
    const Index layers_number = get_layers_number();

    if (layers_number == 0)
        return Tensor3();

    const Index batch_size = inputs_1.dimension(0);

    ForwardPropagation forward_propagation(batch_size, this);

    const vector<TensorView> input_views = {TensorView(const_cast<float*>(inputs_1.data()), {{inputs_1.dimension(0), inputs_1.dimension(1), inputs_1.dimension(2)}}),
                                            TensorView(const_cast<float*>(inputs_2.data()), {{inputs_2.dimension(0), inputs_2.dimension(1), inputs_2.dimension(2)}})};

    forward_propagate(input_views, forward_propagation, false);

    return forward_propagation.get_outputs().as_tensor<3>();
}

MatrixR NeuralNetwork::calculate_outputs(const vector<TensorView>& input_views)
{
    if (layers.empty() || input_views.empty()) return {};

    const Index batch_size = input_views[0].shape[0];
    ForwardPropagation forward_propagation(batch_size, this);

    IF_GPU({ return calculate_outputs_device(input_views, forward_propagation); });

    forward_propagate(input_views, forward_propagation, false);

    const size_t layers_count = get_layers_number();
    const TensorView out_view = (layers_count > 0
                           && layers_count - 1 < forward_propagation.views.size()
                           && forward_propagation.views[layers_count - 1].size() > 1
                           && !forward_propagation.views[layers_count - 1].back().empty())
                          ? forward_propagation.views[layers_count - 1].back()[0]
                          : forward_propagation.get_last_trainable_layer_outputs();

    return out_view.as_matrix();
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
    if (parameters.size_in_floats() != aligned_total_elements(get_parameter_shapes()))
        throw runtime_error("Network shapes changed since compile(); call compile() again.");

    const Index first_layer_index = is_training ? get_first_trainable_layer_index() : 0;
    const Index last_layer_index  = is_training ? get_last_trainable_layer_index()  : get_layers_number() - 1;

    const auto pick_input = [&](size_t k) -> const TensorView& {
        return input_view[k < input_view.size() ? k : 0];
    };

    for (Index i = first_layer_index; i <= last_layer_index; ++i)
    {
        const vector<Index>& input_indices = layer_input_indices[i];
        auto& input_slot = forward_propagation.views[i][0];

        for (size_t k = 0; k < input_indices.size(); ++k)
        {
            const Index current_input = input_indices[k];

            if (current_input < 0)
                input_slot[k] = pick_input(size_t(-current_input - 1));
            else if (is_training && current_input < first_layer_index)
                input_slot[k] = pick_input(k);
        }

        PROFILE_SCOPE("fwd:" + layers[i]->get_name());
        layers[i]->forward_propagate(forward_propagation, i, is_training);
    }
}

void NeuralNetwork::forward_propagate(const vector<TensorView>& input_view,
                                      const VectorR& new_parameters,
                                      ForwardPropagation& forward_propagation)
{
    VectorMap parameters_view(parameters.as<float>(), parameters.size_in_floats());

    const VectorR saved_parameters = parameters_view;

    parameters_view = new_parameters;

    forward_propagate(input_view, forward_propagation, true);

    parameters_view = saved_parameters;
}

MatrixR NeuralNetwork::calculate_directional_inputs(const Index direction,
                                                    const VectorR& point,
                                                    float minimum,
                                                    float maximum,
                                                    Index points_number) const
{
    const Index inputs_number = get_inputs_number();

    MatrixR directional_inputs(points_number, inputs_number);

    VectorR inputs = point;

    for (Index i = 0; i < points_number; ++i)
    {
        inputs(direction) = minimum + (maximum - minimum)*float(i)/float(points_number-1);
        directional_inputs.row(i) = inputs.transpose();
    }

    return directional_inputs;
}

Index NeuralNetwork::calculate_image_output(const filesystem::path& image_path)
{
    Tensor3 image = load_image(image_path);

    const auto* scaling_layer = dynamic_cast<Scaling*>(get_first(LayerType::Scaling));
    if (!scaling_layer || scaling_layer->get_input_shape().rank != 3)
        throw runtime_error("Expected 4D image Scaling layer.");

    const Index height = scaling_layer->get_input_shape()[0];
    const Index width = scaling_layer->get_input_shape()[1];
    const Index channels = scaling_layer->get_input_shape()[2];

    const Index current_height = image.dimension(0);
    const Index current_width = image.dimension(1);
    const Index current_channels = image.dimension(2);

    if (current_channels != channels)
        throw runtime_error("Different channels number " + image_path.string() + "\n");

    if (current_height != height || current_width != width)
        image = resize_image(image, height, width);

    Tensor4 input_data(1, height, width, channels);

    const Index pixels_number = height * width * channels;

    #pragma omp parallel for
    for (Index j = 0; j < pixels_number; ++j)
        input_data(j) = image(j);

    const Matrix outputs = calculate_outputs(input_data);

    Index predicted_index = 0;

    if (outputs.size() > 1)
        predicted_index = maximal_index(outputs.row(0));
    else
        predicted_index = outputs(0);

    return predicted_index;
}

MatrixR NeuralNetwork::calculate_text_outputs(const Tensor<string, 1>& input_documents)
{
    if (layers[0]->get_type() != LayerType::Embedding)
        throw runtime_error("First layer must be Embedding for text processing.\n");

    if (input_variables.empty() || input_variables[0].categories.empty())
        throw runtime_error("input_variables[0] does not contain the vocabulary.\n");

    const Index batch_size = input_documents.size();
    const auto* embedding_layer = dynamic_cast<const Embedding*>(get_layer(0).get());
    if (!embedding_layer) throw runtime_error("Expected Embedding layer at index 0.");
    const Index sequence_length = embedding_layer->get_sequence_length();

    const vector<string>& vocabulary = input_variables[0].categories;
    unordered_map<string, Index> vocabulary_map;
    vocabulary_map.reserve(vocabulary.size());

    for (size_t i = 0; i < vocabulary.size(); ++i)
        vocabulary_map[vocabulary[i]] = i;

    MatrixR inputs = MatrixR::Zero(batch_size, sequence_length);

    for (Index i = 0; i < batch_size; ++i)
    {
        const vector<string> tokens = tokenize(input_documents.data()[i]);
        const size_t tokens_number = tokens.size();

        if (sequence_length > 0)
            inputs(i, 0) = 2.0f; // START_INDEX

        for (size_t j = 0; j < tokens_number; ++j)
        {
            if (1 + j >= static_cast<size_t>(sequence_length)) break;

            const auto it = vocabulary_map.find(tokens[j]);

            inputs(i, 1 + j) = (it != vocabulary_map.end())
                                   ? static_cast<float>(it->second)
                                   : 1.0f; // UNK_INDEX
        }

        if (1 + tokens_number < static_cast<size_t>(sequence_length))
            inputs(i, 1 + tokens_number) = 3.0f; // END_INDEX
    }

    return calculate_outputs(inputs);
}

void NeuralNetwork::to_JSON(JsonWriter& printer) const
{
    const Index inputs_number = get_inputs_number();
    const Index layers_number = get_layers_number();
    const Index outputs_number = get_outputs_number();

    vector<string> input_names = get_input_feature_names();
    while (ssize(input_names) < inputs_number)
        input_names.push_back("input_" + to_string(input_names.size() + 1));

    vector<string> output_names = get_output_feature_names();
    while (ssize(output_names) < outputs_number)
        output_names.push_back("output_" + to_string(output_names.size() + 1));

    printer.open_element("NeuralNetwork");

    // Input

    printer.open_element("Inputs");
    add_json_field(printer, "InputsNumber", to_string(inputs_number));
    printer.begin_array("Input");
    for (Index i = 0; i < inputs_number; ++i)
    {
        printer.begin_array_object();
        add_json_field(printer, "Index", to_string(i + 1));
        add_json_field(printer, "Text",  input_names[i]);
        printer.end_array_object();
    }
    printer.end_array();
    printer.close_element();

    // Layers

    printer.open_element("Layers");
    add_json_field(printer, "LayersNumber", to_string(layers_number));

    printer.begin_array("Items");
    for (Index i = 0; i < layers_number; ++i)
    {
        printer.begin_array_object();
        layers[i]->to_JSON(printer);
        printer.end_array_object();
    }
    printer.end_array();

    printer.open_element("LayerInputIndices");
    printer.begin_array("LayerInputsIndices");
    for (size_t i = 0; i < layer_input_indices.size(); ++i)
    {
        printer.begin_array_object();
        add_json_field(printer, "LayerIndex", to_string(i));
        add_json_field(printer, "Text", vector_to_string(layer_input_indices[i]));
        printer.end_array_object();
    }
    printer.end_array();
    printer.close_element();

    printer.close_element();

    // Outputs

    printer.open_element("Outputs");
    const Index outputs_count = has(LayerType::Embedding) ? outputs_number : output_names.size();
    add_json_field(printer, "OutputsNumber", to_string(outputs_count));
    printer.begin_array("Output");
    for (Index i = 0; i < outputs_count; ++i)
    {
        printer.begin_array_object();
        add_json_field(printer, "Index", to_string(i + 1));
        add_json_field(printer, "Text",  output_names[i]);
        printer.end_array_object();
    }
    printer.end_array();
    printer.close_element();

    // Parameters

    printer.open_element("Parameters");
    if (parameters.size_in_floats() > 0)
    {
        const Index params_size = parameters.size_in_floats();
        const float* params_data = parameters.as<float>();
        VectorR params_host_snapshot;
#ifdef OPENNN_HAS_CUDA
        if (parameters.device_type == Device::CUDA)
        {
            params_host_snapshot.resize(params_size);
            CHECK_CUDA(cudaMemcpy(params_host_snapshot.data(), params_data,
                                  params_size * sizeof(float), cudaMemcpyDeviceToHost));
            params_data = params_host_snapshot.data();
        }
#endif
        const Map<const VectorR, AlignedMax> parameters_view(params_data, params_size);
        add_json_field(printer, "Values", vector_to_string(parameters_view, " "));
    }
    printer.close_element();

    printer.close_element();
}

void NeuralNetwork::from_JSON(const JsonDocument& document)
{
    [[maybe_unused]] static const bool _layers_registered = []() { register_classes(); return true; }();

    const Json* neural_network_element = get_json_root(document, "NeuralNetwork");

    const Json* inputs_element = neural_network_element->first_child("Inputs");
    if (inputs_element)
    {
        const Index inputs_number = read_json_index(inputs_element, "InputsNumber");
        input_variables.resize(inputs_number);

        for_json_items(inputs_element, "Input", inputs_number, [this](Index i, const Json* element) {
            input_variables[i].name = read_json_string(element, "Text");
        });
    }

    const Json* layers_container = neural_network_element->first_child("Layers");
    if (!layers_container)
        throw runtime_error("layers container is nullptr.");

    const Index layers_number = read_json_index(layers_container, "LayersNumber");

    layers.clear();
    layer_input_indices.clear();
    layers.reserve(layers_number);
    layer_input_indices.resize(layers_number);

    const Json* items_array = layers_container->find("Items");
    if (items_array && items_array->is_array())
    {
        for (const Json& item : items_array->array_value)
        {
            if (!item.is_object() || item.object_value.empty()) continue;

            const string& tag_name = item.object_value[0].first;

            unique_ptr<Layer> layer = Registry<Layer>::instance().create(tag_name);
            if (!layer)
                throw runtime_error("Layer '" + tag_name + "' not found in Registry. "
                                                          "Ensure the layer file is linked and REGISTER macro is used.");

            JsonDocument layer_doc;
            layer_doc.root = item;
            layer->from_JSON(layer_doc);

            layers.push_back(move(layer));
        }

        first_trainable_cache_ = -1;
        last_trainable_cache_  = -1;
    }

    const Json* connectivity_element = layers_container->find("LayerInputIndices");
    if (connectivity_element)
    {
        const Json* indices_array = connectivity_element->find("LayerInputsIndices");
        if (indices_array && indices_array->is_array())
        {
            for (const Json& entry : indices_array->array_value)
            {
                const long layer_index = read_json_index(&entry, "LayerIndex");
                const string text   = read_json_string(&entry, "Text");
                if (layer_index >= 0 && layer_index < ssize(layers) && !text.empty())
                {
                    Shape shape = string_to_shape(text, " ");
                    layer_input_indices[layer_index] = vector<Index>(shape.begin(), shape.end());
                }
            }
        }
    }

    const Json* outputs_element = neural_network_element->first_child("Outputs");
    if (outputs_element)
    {
        const Index outputs_number = read_json_index(outputs_element, "OutputsNumber");
        output_variables.resize(outputs_number);

        for_json_items(outputs_element, "Output", outputs_number, [this](Index i, const Json* element) {
            output_variables[i].name = read_json_string(element, "Text");
        });
    }

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

    const Json* parameters_element = neural_network_element->first_child("Parameters");
    const string parameters_text   = parameters_element ? read_json_string(parameters_element, "Values") : string();
    if (!parameters_text.empty())
    {
        VectorR xml_parameters;
        string_to_vector(parameters_text, xml_parameters);

        if (xml_parameters.size() > 0)
        {
            if (xml_parameters.size() != parameters.size_in_floats())
            {
                cout << "Warning: XML parameter size (" << xml_parameters.size()
                     << ") differs from Compiled size (" << parameters.size_in_floats() << ").\n";
            }

            const Index elements_to_copy = min(parameters.size_in_floats(), xml_parameters.size());

#ifdef OPENNN_HAS_CUDA
            const bool was_on_device = (parameters.device_type == Device::CUDA);
            if (was_on_device) copy_parameters_host();
#endif
            std::copy(xml_parameters.data(), xml_parameters.data() + elements_to_copy, parameters.as<float>());
#ifdef OPENNN_HAS_CUDA
            if (was_on_device) copy_parameters_device();
#endif
        }
    }
}

void NeuralNetwork::save(const filesystem::path& file_name) const
{
    ofstream file(file_name);

    if (!file.is_open())
        throw runtime_error("Cannot open file: " + file_name.string());

    JsonWriter printer;
    to_JSON(printer);
    file << printer.c_str();
}

void NeuralNetwork::save_parameters(const filesystem::path& file_name) const
{
    ofstream file(file_name);

    if (!file.is_open())
        throw runtime_error("Cannot open parameters data file.\n");

    const Index params_size = parameters.size_in_floats();
    const float* params_data = parameters.as<float>();
    VectorR params_host_snapshot;
#ifdef OPENNN_HAS_CUDA
    if (parameters.device_type == Device::CUDA)
    {
        params_host_snapshot.resize(params_size);
        CHECK_CUDA(cudaMemcpy(params_host_snapshot.data(), params_data,
                              params_size * sizeof(float), cudaMemcpyDeviceToHost));
        params_data = params_host_snapshot.data();
    }
#endif
    const Map<const VectorR, AlignedMax> parameters_view(params_data, params_size);
    file << parameters_view << "\n";

    file.close();
}

void NeuralNetwork::load(const filesystem::path& file_name)
{
    set_default();

    from_JSON(load_json_file(file_name));
}

void NeuralNetwork::load_parameters_binary(const filesystem::path& file_name)
{
    ifstream file(file_name, ios::binary);

    if (!file.is_open())
        throw runtime_error("Cannot open binary file: " + file_name.string() + "\n");

    const Index parameters_number = parameters.size_in_floats();

#ifdef OPENNN_HAS_CUDA
    const bool was_on_device = (parameters.device_type == Device::CUDA);
    if (was_on_device) copy_parameters_host();
#endif
    file.read(reinterpret_cast<char*>(parameters.as<float>()), parameters_number * sizeof(float));
#ifdef OPENNN_HAS_CUDA
    if (was_on_device) copy_parameters_device();
#endif

    if (!file)
        throw runtime_error("Error reading binary file: " + file_name.string());

}

void NeuralNetwork::save_outputs(MatrixR& inputs, const filesystem::path& file_name)
{
    const MatrixR outputs = calculate_outputs(inputs);

    ofstream file(file_name);

    if (!file.is_open())
        throw runtime_error("Cannot open " + file_name.string() + " file.\n");

    const vector<string> output_names = get_output_feature_names();

    const Index outputs_number = get_outputs_number();
    const Index batch_size = inputs.rows();

    for (size_t i = 0; i < size_t(outputs_number); ++i)
    {
        file << output_names[i];

        if (i != output_names.size() - 1)
            file << ";";
    }

    file << "\n";

    for (Index i = 0; i < batch_size; ++i)
    {
        for (Index j = 0; j < outputs_number; ++j)
        {
            file << outputs(i, j);

            if (j != outputs_number-1)
                file << ";";
        }

        file << "\n";
    }

    file.close();
}

void NeuralNetwork::save_outputs(Tensor3& inputs_3d, const filesystem::path& file_name)
{
    const MatrixR outputs = calculate_outputs(inputs_3d);

    const Index batch_size = inputs_3d.dimension(0);
    const Index past_time_steps = inputs_3d.dimension(1);
    const Index features_number = inputs_3d.dimension(2);

    Tensor2 last_time_step_inputs(batch_size, features_number);

    last_time_step_inputs = inputs_3d.chip(past_time_steps - 1, 1);

    ofstream file(file_name);

    if (!file.is_open())
        throw runtime_error("Cannot open " + file_name.string() + " file.\n");

    const vector<string> output_names = get_output_feature_names();
    const Index outputs_number = get_outputs_number();

    const vector<string> input_names = get_input_feature_names();
    for (const auto& name : input_names)
        file << name << ";";

    for (size_t i = 0; i < size_t(outputs_number); ++i)
    {
        file << output_names[i];
        if (i != output_names.size() - 1)
            file << ";";
    }
    file << "\n";

    for (Index i = 0; i < batch_size; ++i)
    {
        for (Index j = 0; j < features_number; ++j)
            file << last_time_step_inputs(i, j) << ";";

        for (Index j = 0; j < outputs_number; ++j)
        {
            file << outputs(i, j);
            if (j != outputs_number - 1)
                file << ";";
        }
        file << "\n";
    }

    file.close();
}

vector<string> NeuralNetwork::get_layer_labels() const
{
    vector<string> layer_labels(layers.size());
    transform(layers.begin(), layers.end(), layer_labels.begin(),
              [](const unique_ptr<Layer>& layer) { return layer->get_label(); });
    return layer_labels;
}

vector<string> NeuralNetwork::get_names_string() const
{
    vector<string> names(layers.size());
    transform(layers.begin(), layers.end(), names.begin(),
              [](const unique_ptr<Layer>& layer) { return layer->get_name(); });
    return names;
}

#ifdef OPENNN_HAS_CUDA

void NeuralNetwork::copy_parameters_device()
{
    if (parameters.empty()) return;

    parameters.migrate_to(Device::CUDA);

    if(config.training_type  == Type::BF16 ||
       config.inference_type == Type::BF16)
    {
        parameters_bf16.resize_bytes(parameters.size_in_floats() * Index(sizeof(__nv_bfloat16)), Device::CUDA);
        cast_parameters_to_bf16();
    }

    link_parameters();
}

void NeuralNetwork::cast_parameters_to_bf16()
{
    if (parameters_bf16.empty()) return;
    if (parameters.empty())      return;

    cast_fp32_to_bf16_cuda(parameters.size_in_floats(),
                           parameters.as<float>(),
                           parameters_bf16.as<__nv_bfloat16>());
}

void NeuralNetwork::copy_parameters_host()
{
    if (parameters.empty()) return;

    parameters.migrate_to(Device::CPU);

    link_parameters();
}

void NeuralNetwork::link_parameters()
{
    // Ignore the BF16 mirror when active Configuration is CPU (host can't read GPU memory).
    float* fp32_ptr           = parameters.as<float>();
    const bool use_bf16_mirror =
        is_gpu() && parameters_bf16.bytes > 0;
    __nv_bfloat16* bf16_ptr = use_bf16_mirror
                                ? parameters_bf16.as<__nv_bfloat16>()
                                : nullptr;

    for (auto& layer : layers)
    {
        const auto specs = layer->get_parameter_specs();
        auto& param_views = layer->get_parameter_views();

        for (size_t i = 0; i < specs.size(); ++i)
        {
            const auto& [shape, slot_dtype] = specs[i];
            if (shape.empty()) continue;

            const Index aligned = get_aligned_size(shape.size());

            if (i < param_views.size())
            {
                const bool use_bf16 = (slot_dtype == Type::BF16 && bf16_ptr != nullptr);
                void* slot_ptr = use_bf16 ? static_cast<void*>(bf16_ptr)
                                          : static_cast<void*>(fp32_ptr);
                param_views[i] = TensorView(slot_ptr, shape, use_bf16 ? Type::BF16 : Type::FP32);
            }

            fp32_ptr += aligned;
            if (bf16_ptr != nullptr) bf16_ptr += aligned;
        }

        layer->redistribute_parameters_to_operators();
    }
}

void NeuralNetwork::copy_states_device()
{
    if (states.empty()) return;

    states.migrate_to(Device::CUDA);

    link_states();
}

void NeuralNetwork::copy_states_host()
{
    if (states.empty()) return;

    states.migrate_to(Device::CPU);

    link_states();
}

void NeuralNetwork::link_states()
{
    float* state_pointer = states.as<float>();
    if (!state_pointer) return;

    for (auto& layer : layers)
        state_pointer = layer->link_states(state_pointer);
}

MatrixR NeuralNetwork::calculate_outputs_device(const vector<TensorView>& input_views_cpu,
                                                ForwardPropagation& forward_propagation)
{
    copy_parameters_device();
    copy_states_device();

    const Index input_size = input_views_cpu[0].size();
    float* input_device = nullptr;
    CHECK_CUDA(cudaMalloc(&input_device, input_size * sizeof(float)));
    cudaStream_t stream = Backend::get_compute_stream();
    CHECK_CUDA(cudaMemcpyAsync(input_device,
                               input_views_cpu[0].data,
                               input_size * sizeof(float),
                               cudaMemcpyHostToDevice,
                               stream));

    vector<TensorView> input_views_gpu = input_views_cpu;
    input_views_gpu[0].data = input_device;

    forward_propagate(input_views_gpu, forward_propagation, false);

    const size_t layers_count = get_layers_number();
    const TensorView out_view = (layers_count > 0
                           && layers_count - 1 < forward_propagation.views.size()
                           && forward_propagation.views[layers_count - 1].size() > 1
                           && !forward_propagation.views[layers_count - 1].back().empty())
                          ? forward_propagation.views[layers_count - 1].back()[0]
                          : forward_propagation.get_last_trainable_layer_outputs();

    const Index batch_size = input_views_cpu[0].shape[0];
    const Index out_cols = out_view.size() / batch_size;
    MatrixR result(batch_size, out_cols);

    vector<uint16_t> staging;
    const bool output_is_bf16 = (out_view.type == Type::BF16);

    if (output_is_bf16)
    {
        const Index size = out_view.size();
        staging.resize(static_cast<size_t>(size));
        CHECK_CUDA(cudaMemcpyAsync(staging.data(),
                                   out_view.data,
                                   size * sizeof(uint16_t),
                                   cudaMemcpyDeviceToHost,
                                   stream));
    }
    else
    {
        CHECK_CUDA(cudaMemcpyAsync(result.data(),
                                   out_view.data,
                                   out_view.size() * sizeof(float),
                                   cudaMemcpyDeviceToHost,
                                   stream));
    }

    CHECK_CUDA(cudaStreamSynchronize(stream));

    if (output_is_bf16)
    {
        float* destination = result.data();
        for (Index i = 0; i < out_view.size(); ++i)
        {
            const uint32_t bits = static_cast<uint32_t>(staging[size_t(i)]) << 16;
            std::memcpy(&destination[i], &bits, sizeof(float));
        }
    }

    CHECK_CUDA(cudaFree(input_device));

    return result;
}

#endif

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
