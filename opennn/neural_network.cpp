//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   N E U R A L   N E T W O R K   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "registry.h"
//#include "string_utilities.h"
#include "tensor_utilities.h"
#include "neural_network.h"
#include "dense_layer.h"
#include "scaling_layer.h"
#include "flatten_layer.h"
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
    const Index old_layers_number = static_cast<Index>(get_layers_number()) - 1;

    if (!layers.empty()) validate_type(layers.back()->get_type());

    layers.push_back(std::move(layer));

    layer_input_indices.push_back(input_indices.empty()
        ? vector<Index>(1, old_layers_number )
        : input_indices);
}

void NeuralNetwork::compile()
{
    const size_t layers_number = get_layers_number();

    if (layers_number == 0) return;

    for (size_t i = 0; i < layers_number; ++i)
    {
        const vector<Index>& inputs = layer_input_indices[i];

        if (inputs.size() == 1 && inputs[0] >= 0)
            layers[i]->set_input_shape(layers[inputs[0]]->get_output_shape());
    }

    Index total_parameters = 0;
    for (const auto& layer : layers)
        for (const Shape& s : layer->get_parameter_shapes())
            total_parameters += get_aligned_size(s.size());

    parameters.resize(total_parameters);
    parameters.setZero();

    type* pointer = parameters.data();

    for (auto& layer : layers)
        pointer = layer->link_parameters(pointer);
}

void NeuralNetwork::validate_type(LayerType type) const
{
    if(type == LayerType::Bounding)
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

static vector<string> get_feature_names_from(const vector<Variable>& vars)
{
    vector<string> feature_names;
    feature_names.reserve(vars.size());

    for (const auto& var : vars)
    {
        const vector<string> names = var.get_names();
        feature_names.insert(feature_names.end(), names.begin(), names.end());
    }

    return feature_names;
}

const vector<string> NeuralNetwork::get_input_feature_names() const
{
    return get_feature_names_from(input_variables);
}

const vector<string> NeuralNetwork::get_output_feature_names() const
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
    if(new_label == "Dataset" || new_label == "decoder")
        return -1;

    if(new_label == "input")
        return -2;

    auto it = find_if(layers.begin(), layers.end(),
                      [&new_label](const unique_ptr<Layer>& layer) { return layer->get_label() == new_label; });

    if (it != layers.end())
        return distance(layers.begin(), it);

    throw runtime_error("Layer not found: " + new_label);
}

vector<vector<Index>> NeuralNetwork::get_layer_output_indices() const
{
    const size_t layers_number = layer_input_indices.size();

    vector<vector<Index>> layer_output_indices(layers_number);

    for(size_t i = 0; i < layers_number; i++)
        for(const Index input_index : layer_input_indices[i])
            if (input_index >= 0)
                layer_output_indices[input_index].push_back(static_cast<Index>(i));

    for(auto& outputs : layer_output_indices)
        if(outputs.empty())
            outputs.push_back(-1);

    return layer_output_indices;
}

Index NeuralNetwork::find_input_index(const vector<Index>& layer_inputs_indices, Index layer_index) const
{
    auto it = find(layer_inputs_indices.begin(), layer_inputs_indices.end(), layer_index);
    return (it != layer_inputs_indices.end()) ? distance(layer_inputs_indices.begin(), it) : -1;
}

const Layer* NeuralNetwork::get_first(const string& name) const
{
    return get_first(string_to_layer_type(name));
}

const Layer* NeuralNetwork::get_first(LayerType type) const
{
    auto it = find_if(layers.begin(), layers.end(),
                      [type](const unique_ptr<Layer>& layer) { return layer->get_type() == type; });

    if (it != layers.end())
        return it->get();

    throw runtime_error("Layer not found in Neural Network: " + layer_type_to_string(type));
}

Layer* NeuralNetwork::get_first(const string& name)
{
    return const_cast<Layer*>(const_cast<const NeuralNetwork*>(this)->get_first(name));
}

Layer* NeuralNetwork::get_first(LayerType type)
{
    return const_cast<Layer*>(const_cast<const NeuralNetwork*>(this)->get_first(type));
}

static void set_variable_names(vector<Variable>& vars, const vector<string>& new_names)
{
    Index j = 0;
    for(size_t i = 0; i < vars.size(); ++i)
    {
        if(vars[i].is_categorical())
        {
            const size_t num_cats = vars[i].get_categories_number();
            vars[i].categories.assign(new_names.begin() + j, new_names.begin() + j + num_cats);
            j += num_cats;
        }
        else
        {
            vars[i].name = new_names[j];
            j++;
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
    const Index total_inputs = new_input_shape.size();
    input_variables.resize(total_inputs);

    if(has(LayerType::Scaling2d))
        get_first(LayerType::Scaling2d)->set_input_shape(new_input_shape);
    else if(has(LayerType::Scaling3d))
        get_first(LayerType::Scaling3d)->set_input_shape(new_input_shape);

    layers[get_first_trainable_layer_index()]->set_input_shape(new_input_shape);
}

void NeuralNetwork::set_default()
{
    reference_all_layers();

    layers.clear();

    layer_input_indices.clear();

    input_variables.clear();

    output_variables.clear();
}

void NeuralNetwork::set_layer_input_indices(const string& layer_label,
                                            const vector<string>& new_layer_input_labels)
{
    const Index layer_index = get_layer_index(layer_label);

    const Index size = new_layer_input_labels.size();

    vector<Index> new_layer_input_indices(size);

    for(Index i = 0; i < size; i++)
        new_layer_input_indices[i] = get_layer_index(new_layer_input_labels[i]);

    layer_input_indices[layer_index] = new_layer_input_indices;
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
    if(layers.empty())
        return 0;

    if(has(LayerType::Embedding))
        return get_layer(0)->get_inputs_number();

    if(has(LayerType::Recurrent))
        return get_first(LayerType::Recurrent)->get_input_shape()[1];

    const Shape input_shape = layers[0]->get_input_shape();

    return input_shape.size();
}

Index NeuralNetwork::get_outputs_number() const
{
    if(layers.empty()) return 0;

    return layers.back()->get_output_shape().size();
}

Shape NeuralNetwork::get_input_shape() const
{
    if(layers.empty())
        return {};

    return layers[0]->get_input_shape();
}

Shape NeuralNetwork::get_output_shape() const
{
    if(layers.empty())
        return {};

    return layers.back()->get_output_shape();
}

ActivationFunction NeuralNetwork::get_output_activation() const
{
    const Index last_idx = get_last_trainable_layer_index();
    if(last_idx < 0 || static_cast<size_t>(last_idx) >= layers.size())
        return ActivationFunction::Linear;

    return layers[last_idx]->get_output_activation();
}

Index NeuralNetwork::get_parameters_number() const
{
    Index parameters_number = 0;

    const size_t layers_number = get_layers_number();

    for(size_t i = 0; i < layers_number; i++)
        parameters_number += layers[i]->get_parameters_number();

    return parameters_number;
}

vector<Index> NeuralNetwork::get_layer_parameter_numbers() const
{
    const size_t layers_number = get_layers_number();

    vector<Index> layer_parameter_numbers(layers_number);

    #pragma omp parallel for

    for(size_t i = 0; i < layers_number; i++)
        layer_parameter_numbers[i] = layers[i]->get_parameters_number();

    return layer_parameter_numbers;
}

Index NeuralNetwork::get_first_trainable_layer_index() const
{
    auto it = find_if(layers.begin(), layers.end(),
                      [](const unique_ptr<Layer>& layer) { return layer->get_is_trainable(); });

    if (it != layers.end())
        return distance(layers.begin(), it);

    throw runtime_error("The neural network has no trainable layers: get_first_trainable_layer_index.");
}

Index NeuralNetwork::get_last_trainable_layer_index() const
{
    const size_t layers_number = get_layers_number();

    for(size_t i = layers_number; i-- > 0; )
        if (layers[i]->get_is_trainable())
            return static_cast<Index>(i);

    throw runtime_error("The neural network has no trainable layers: get_last_trainable_layer_index");
}

size_t NeuralNetwork::get_layers_number(const string& name) const
{
    return get_layers_number(string_to_layer_type(name));
}

size_t NeuralNetwork::get_layers_number(LayerType type) const
{
    return count_if(layers.begin(), layers.end(),
                    [type](const unique_ptr<Layer>& layer) {return layer->get_type() == type;});
}

void NeuralNetwork::set_parameters_random()
{
    const size_t layers_number = get_layers_number();

    for(size_t i = 0; i < layers_number; i++)
        layers[i]->set_parameters_random();
}

void NeuralNetwork::set_parameters_glorot()
{
    const size_t layers_number = get_layers_number();

    #pragma omp parallel for
    for(size_t i = 0; i < layers_number; i++)
        layers[i]->set_parameters_glorot();
}

Tensor3 NeuralNetwork::calculate_outputs(const Tensor3& inputs_1, const Tensor3& inputs_2)
{
    const size_t layers_number = get_layers_number();

    if (layers_number == 0)
        return Tensor3();

    const Index batch_size = inputs_1.dimension(0);

    ForwardPropagation forward_propagation(batch_size, this);

    const vector<TensorView> input_views = {TensorView(const_cast<type*>(inputs_1.data()), {{inputs_1.dimension(0), inputs_1.dimension(1), inputs_1.dimension(2)}}),
                                            TensorView(const_cast<type*>(inputs_2.data()), {{inputs_2.dimension(0), inputs_2.dimension(1), inputs_2.dimension(2)}})};

    forward_propagate(input_views, forward_propagation, false);

    return tensor_map<3>(forward_propagation.get_outputs());
}

MatrixR NeuralNetwork::calculate_outputs(const vector<TensorView>& input_views)
{
    if(layers.empty() || input_views.empty()) return {};

    const Index batch_size = input_views[0].shape[0];
    ForwardPropagation fp(batch_size, this);

#ifdef OPENNN_WITH_CUDA
    if (Device::instance().is_gpu())
        return calculate_outputs_device(input_views, fp);
#endif

    forward_propagate(input_views, fp, false);

    const size_t layers_count = get_layers_number();
    const TensorView out_view = (layers_count > 0
                           && layers_count - 1 < fp.views.size()
                           && fp.views[layers_count - 1].size() > 1
                           && !fp.views[layers_count - 1].back().empty())
                          ? fp.views[layers_count - 1].back()[0]
                          : fp.get_last_trainable_layer_outputs();

    return MatrixMap(out_view.data, batch_size, out_view.size() / batch_size);
}

MatrixR NeuralNetwork::calculate_outputs(const MatrixR& inputs) 
{
    return calculate_outputs({TensorView(const_cast<type*>(inputs.data()), {inputs.rows(), inputs.cols()})});
}

MatrixR NeuralNetwork::calculate_outputs(const Tensor3& inputs) 
{
    return calculate_outputs({TensorView(const_cast<type*>(inputs.data()), {inputs.dimension(0), inputs.dimension(1), inputs.dimension(2)})});
}

MatrixR NeuralNetwork::calculate_outputs(const Tensor4& inputs) 
{
    return calculate_outputs({TensorView(const_cast<type*>(inputs.data()), {inputs.dimension(0), inputs.dimension(1), inputs.dimension(2), inputs.dimension(3)})});
}

void NeuralNetwork::forward_propagate(const vector<TensorView>& input_view,
                                      ForwardPropagation& forward_propagation,
                                      bool is_training) const
{
    const size_t layers_number = get_layers_number();

    const Index first_layer_index = is_training
                                        ? get_first_trainable_layer_index()
                                        : 0;

    const Index last_layer_index = is_training
                                       ? get_last_trainable_layer_index()
                                       : static_cast<Index>(layers_number) - 1;

    for (Index layer_index = first_layer_index; layer_index <= last_layer_index; ++layer_index)
    {
        const vector<Index>& input_indices = layer_input_indices[layer_index];

        if(static_cast<size_t>(layer_index) >= forward_propagation.views.size()
           || forward_propagation.views[layer_index].empty())
            continue;

        auto& input_slot = forward_propagation.views[layer_index][0];
        input_slot.resize(input_indices.size());

        for(size_t k = 0; k < input_indices.size(); ++k)
        {
            const Index current_input = input_indices[k];

            if(current_input < 0)
            {
                const size_t input_view_index = static_cast<size_t>((-current_input) - 1) < input_view.size()
                    ? static_cast<size_t>((-current_input) - 1) : 0;

                input_slot[k] = input_view[input_view_index];
            }
            else if(is_training && current_input < first_layer_index)
            {
                const size_t input_view_index = (k < input_view.size()) ? k : 0;
                input_slot[k] = input_view[input_view_index];
            }
            // else: already wired in ForwardPropagation::set() to upstream output
        }
    }

    for(Index i = first_layer_index; i <= last_layer_index; i++)
        layers[i]->forward_propagate(forward_propagation, i, is_training);
}

void NeuralNetwork::forward_propagate(const vector<TensorView>& input_view,
                                      const VectorR& new_parameters,
                                      ForwardPropagation& forward_propagation)
{
    VectorR& params = get_parameters();

    // Save current values without changing buffer address (Layer TensorViews
    // are bound to params.data()). Swap would relocate the buffer and break them.
    const VectorR saved_parameters = params;

    params = new_parameters;

    forward_propagate(input_view, forward_propagation, true);

    params = saved_parameters;
}

MatrixR NeuralNetwork::calculate_directional_inputs(const Index direction,
                                                    const VectorR& point,
                                                    type minimum,
                                                    type maximum,
                                                    Index points_number) const
{
    const Index inputs_number = get_inputs_number();

    MatrixR directional_inputs(points_number, inputs_number);

    VectorR inputs(inputs_number);

    inputs = point;

    for(Index i = 0; i < points_number; i++)
    {
        inputs(direction) = minimum + (maximum - minimum)*type(i)/type(points_number-1);

        for(Index j = 0; j < inputs_number; j++)
            directional_inputs(i, j) = inputs(j);
    }

    return directional_inputs;
}

Index NeuralNetwork::calculate_image_output(const filesystem::path& image_path)
{
    Tensor3 image = load_image(image_path);

    const auto* scaling_layer = dynamic_cast<Scaling<4>*>(get_first(LayerType::Scaling4d));
    if(!scaling_layer) throw runtime_error("Expected Scaling<4> layer.");

    const Index height = scaling_layer->get_input_shape()[0];
    const Index width = scaling_layer->get_input_shape()[1];
    const Index channels = scaling_layer->get_input_shape()[2];

    const Index current_height = image.dimension(0);
    const Index current_width = image.dimension(1);
    const Index current_channels = image.dimension(2);

    if (current_channels != channels)
        throw runtime_error("Error: Different channels number " + image_path.string() + "\n");

    if(current_height != height || current_width != width)
        image = resize_image(image, height, width);

    Tensor4 input_data(1, height, width, channels);

    const Index pixels_number = height * width * channels;

    #pragma omp parallel for
    for(Index j = 0; j < pixels_number; j++)
        input_data(j) = image(j);

    const Matrix outputs = calculate_outputs(input_data);

    Index predicted_index = 0;

    if (outputs.size() > 1)
    {
        type max_value = outputs(0);

        for(Index i = 1; i < outputs.cols(); ++i)
        {
            if (outputs(i) > max_value)
            {
                max_value = outputs(i);
                predicted_index = i;
            }
        }
    }
    else
        predicted_index = outputs(0);

    return predicted_index;
}

MatrixR NeuralNetwork::calculate_text_outputs(const Tensor<string, 1>& input_documents)
{
    if(layers[0]->get_type() != LayerType::Embedding)
        throw runtime_error("Error: First layer must be Embedding for text processing.\n");

    if(input_variables.empty() || input_variables[0].categories.empty())
        throw runtime_error("Error: input_variables[0] does not contain the vocabulary.\n");

    const Index batch_size = input_documents.size();
    const auto* embedding_layer = dynamic_cast<const Embedding*>(get_layer(0).get());
    if(!embedding_layer) throw runtime_error("Expected Embedding layer at index 0.");
    const Index sequence_length = embedding_layer->get_sequence_length();

    const vector<string>& vocabulary = input_variables[0].categories;
    unordered_map<string, Index> vocabulary_map;
    vocabulary_map.reserve(vocabulary.size());

    for(size_t i = 0; i < vocabulary.size(); ++i)
        vocabulary_map[vocabulary[i]] = i;

    MatrixR inputs(batch_size, sequence_length);
    inputs.setConstant(0.0f);

    for(Index i = 0; i < batch_size; ++i)
    {
        const string input_data = input_documents.data()[i];
        const vector<string> tokens = tokenize(input_data);
        const size_t tokens_number = tokens.size();

        if (sequence_length > 0)
            inputs(i, 0) = 2.0f; // START_INDEX

        for(size_t j = 0; j < tokens_number; j++)
        {
            if (1 + j >= static_cast<size_t>(sequence_length)) break;

            const auto it = vocabulary_map.find(tokens[j]);

            inputs(i, 1 + j) = (it != vocabulary_map.end())
                                   ? static_cast<type>(it->second)
                                   : 1.0f; // UNK_INDEX
        }

        if (1 + tokens_number < static_cast<size_t>(sequence_length))
            inputs(i, 1 + tokens_number) = 3.0f; // END_INDEX
    }

    MatrixR outputs = calculate_outputs(inputs);

    return outputs;
}

void NeuralNetwork::to_XML(XmlPrinter& printer) const
{
    const Index inputs_number = get_inputs_number();
    const size_t layers_number = get_layers_number();
    const Index outputs_number = get_outputs_number();

    vector<string> input_names = get_input_feature_names();
    while (input_names.size() < static_cast<size_t>(inputs_number))
        input_names.push_back("input_" + to_string(input_names.size() + 1));

    vector<string> output_names = get_output_feature_names();
    while (output_names.size() < static_cast<size_t>(outputs_number))
        output_names.push_back("output_" + to_string(output_names.size() + 1));

    printer.open_element("NeuralNetwork");

    // Input

    printer.open_element("Inputs");

    add_xml_element(printer, "InputsNumber", to_string(inputs_number));

    for(Index i = 0; i < inputs_number; i++)
        add_xml_element_attribute(printer, "Input", input_names[i], "Index", to_string(i + 1));

    printer.close_element();

    // Layers

    printer.open_element("Layers");

    add_xml_element(printer, "LayersNumber", to_string(layers_number));

    for(size_t i = 0; i < layers_number; i++)
        layers[i]->to_XML(printer);

    // Layer input indices

    printer.open_element("LayerInputIndices");

    for(size_t i = 0; i < layer_input_indices.size(); i++)
        add_xml_element_attribute(printer, "LayerInputsIndices", vector_to_string(layer_input_indices[i]), "LayerIndex", to_string(i));

    printer.close_element();

    printer.close_element();

    // Outputs

    printer.open_element("Outputs");

    const Index outputs_count = has(LayerType::Embedding) ? outputs_number : output_names.size();
    add_xml_element(printer, "OutputsNumber", to_string(outputs_count));

    for(Index i = 0; i < outputs_count; i++)
        add_xml_element_attribute(printer, "Output", output_names[i], "Index", to_string(i + 1));

    printer.close_element();

    // Paramaters

    printer.open_element("Parameters");

    if (parameters.size() > 0)
        printer.push_text(vector_to_string(parameters.vector, " ").c_str());

    printer.close_element();

    printer.close_element();
}

void NeuralNetwork::from_XML(const XmlDocument& document)
{
    const XmlElement* neural_network_element = get_xml_root(document, "NeuralNetwork");

    // 1. Load Input Variables
    const XmlElement* inputs_element = neural_network_element->first_child_element("Inputs");
    if(inputs_element)
    {
        const Index inputs_number = read_xml_index(inputs_element, "InputsNumber");
        input_variables.resize(inputs_number);

        for_xml_items(inputs_element, "Input", inputs_number, [this](Index i, const XmlElement* el){
            if(el->get_text())
                input_variables[i].name = el->get_text();
        });
    }

    // 2. Load Layers Topology
    const XmlElement* layers_container = neural_network_element->first_child_element("Layers");
    if(!layers_container)
        throw runtime_error("NeuralNetwork error: layers container is nullptr.");

    const Index layers_number = read_xml_index(layers_container, "LayersNumber");

    layers.clear();
    layer_input_indices.clear();
    // Pre-reserve to avoid frequent reallocations during add_layer
    layers.reserve(layers_number);
    layer_input_indices.resize(layers_number);

    // Iterate through children of <Layers>, skipping the <LayersNumber> tag
    const XmlElement* layer_element = layers_container->first_child_element();
    while(layer_element)
    {
        string tag_name = layer_element->name();

        // Skip metadata tags, only process actual Layer types
        if(tag_name != "LayersNumber" && tag_name != "LayerInputIndices")
        {
            // Use the Registry to create the specific layer type (Dense, Scaling, etc.)
            unique_ptr<Layer> layer = Registry<Layer>::instance().create(tag_name);

            if (!layer)
                throw runtime_error("Layer type '" + tag_name + "' not found in Registry. "
                                                                "Ensure the layer file is linked and REGISTER macro is used.");

            // Create a temporary sub-document for the layer to parse its own data
            XmlDocument layer_doc;
            layer_doc.insert_first_child(layer_element->deep_clone(&layer_doc));
            layer->from_XML(layer_doc);

            // Add to the network
            layers.push_back(std::move(layer));
        }
        layer_element = layer_element->next_sibling_element();
    }

    // 3. Load Connectivity (Layer Input Indices)
    const XmlElement* connectivity_element = layers_container->first_child_element("LayerInputIndices");
    if(connectivity_element)
    {
        const XmlElement* indices_element = connectivity_element->first_child_element("LayerInputsIndices");
        while(indices_element)
        {
            int layer_idx = -1;
            indices_element->query_int_attribute("LayerIndex", &layer_idx);

            if(layer_idx >= 0 && layer_idx < static_cast<int>(layers.size()))
            {
                const char* text = indices_element->get_text();
                if(text)
                {
                    Shape s = string_to_shape(text, " ");
                    layer_input_indices[layer_idx] = vector<Index>(s.begin(), s.end());
                }
            }
            indices_element = indices_element->next_sibling_element("LayerInputsIndices");
        }
    }

    // 4. Load Output Variables
    const XmlElement* outputs_element = neural_network_element->first_child_element("Outputs");
    if(outputs_element)
    {
        const Index outputs_number = read_xml_index(outputs_element, "OutputsNumber");
        output_variables.resize(outputs_number);

        for_xml_items(outputs_element, "Output", outputs_number, [this](Index i, const XmlElement* el){
            if(el->get_text())
                output_variables[i].name = el->get_text();
        });
    }

    // 5. Global Settings

    // 6. COMPILE Topology
    // This establishes input/output shapes and allocates the 'parameters' vector
    // with correct ALIGN_BYTES alignment for vectorized operations.
    compile();

    // 7. Load Flattened Parameters
    const XmlElement* parameters_element = neural_network_element->first_child_element("Parameters");
    if(parameters_element && parameters_element->get_text())
    {
        VectorR xml_parameters;
        string_to_vector(parameters_element->get_text(), xml_parameters);

        if (xml_parameters.size() > 0)
        {
            if(xml_parameters.size() != parameters.size())
            {
                // This usually happens if the XML was generated with a different architecture
                cout << "Warning: XML parameter size (" << xml_parameters.size()
                     << ") differs from Compiled size (" << parameters.size() << ").\n";
            }

            const Index elements_to_copy = min(parameters.size(), xml_parameters.size());
            // Since 'parameters' is already linked to layer views via compile(),
            // copying into 'parameters' updates all layers simultaneously.
            std::copy(xml_parameters.data(), xml_parameters.data() + elements_to_copy, parameters.data());
        }
    }
}

void NeuralNetwork::save(const filesystem::path& file_name) const
{
    ofstream file(file_name);

    if(!file.is_open())
        throw runtime_error("Cannot open file: " + file_name.string());

    XmlPrinter printer;
    to_XML(printer);
    file << printer.c_str();
}

void NeuralNetwork::save_parameters(const filesystem::path& file_name) const
{
    ofstream file(file_name);

    if(!file.is_open())
        throw runtime_error("Cannot open parameters data file.\n");

    file << parameters.vector << "\n";

    file.close();
}

void NeuralNetwork::load(const filesystem::path& file_name)
{
    set_default();

    from_XML(load_xml_file(file_name));
}

void NeuralNetwork::load_parameters_binary(const filesystem::path& file_name)
{
    ifstream file(file_name, ios::binary);

    if(!file.is_open())
        throw runtime_error("Cannot open binary file: " + file_name.string() + "\n");

    const Index parameters_number = parameters.size();

//    VectorR new_parameters(parameters_number);

    file.read(reinterpret_cast<char*>(parameters.data()), parameters_number * sizeof(type));

    if(!file)
        throw runtime_error("Error reading binary file: " + file_name.string());

//    set_parameters(new_parameters);
}

void NeuralNetwork::save_outputs(MatrixR& inputs, const filesystem::path& file_name)
{
    const MatrixR outputs = calculate_outputs(inputs);

    ofstream file(file_name);

    if(!file.is_open())
        throw runtime_error("Cannot open " + file_name.string() + " file.\n");

    const vector<string> output_names = get_output_feature_names();

    const Index outputs_number = get_outputs_number();
    const Index batch_size = inputs.rows();

    for(size_t i = 0; i < size_t(outputs_number); i++)
    {
        file << output_names[i];

        if(i != output_names.size() - 1) 
            file << ";";
    }

    file << "\n";

    for(Index i = 0; i < batch_size; i++)
    {
        for(Index j = 0; j < outputs_number; j++)
        {
            file << outputs(i, j);

            if(j != outputs_number-1) 
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

    if(!file.is_open())
        throw runtime_error("Cannot open " + file_name.string() + " file.\n");

    const vector<string> output_names = get_output_feature_names();
    const Index outputs_number = get_outputs_number();

    const vector<string> input_names = get_input_feature_names();
    for(const auto& name : input_names)
        file << name << ";";

    for(size_t i = 0; i < size_t(outputs_number); ++i)
    {
        file << output_names[i];
        if (i != output_names.size() - 1)
            file << ";";
    }
    file << "\n";

    for(Index i = 0; i < batch_size; ++i)
    {
        for(Index j = 0; j < features_number; ++j)
            file << last_time_step_inputs(i, j) << ";";

        for(Index j = 0; j < outputs_number; ++j)
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
    const size_t layers_number = get_layers_number();

    vector<string> layer_labels(layers_number);

    for(size_t i = 0; i < layers_number; i++)
        layer_labels[i] = layers[i]->get_label();

    return layer_labels;
}

vector<string> NeuralNetwork::get_names_string() const
{
    const size_t layers_number = get_layers_number();

    vector<string> names(layers_number);

    for(size_t i = 0; i < layers_number; i++)
        names[i] = layers[i]->get_name();

    return names;
}




#ifdef OPENNN_WITH_CUDA

void NeuralNetwork::copy_parameters_device()
{
    if(parameters.empty()) return;

    parameters.resize_device(parameters.size());

    CHECK_CUDA(cudaMemcpy(parameters.device(),
                          parameters.vector.data(),
                          parameters.size() * sizeof(float),
                          cudaMemcpyHostToDevice));
}

void NeuralNetwork::copy_parameters_host()
{
    if(parameters.empty()) return;

    CHECK_CUDA(cudaMemcpy(parameters.vector.data(),
                          parameters.device(),
                          parameters.size() * sizeof(float),
                          cudaMemcpyDeviceToHost));
}

void NeuralNetwork::link_parameters_device()
{
    type* dev_ptr = parameters.device();

    for(auto& layer : layers)
    {
        const vector<Shape> shapes = layer->get_parameter_shapes();
        auto& param_views = layer->get_parameter_views();

        for(size_t i = 0; i < shapes.size(); ++i)
        {
            if(shapes[i].empty()) continue;

            if(i < param_views.size())
            {
                param_views[i].data = dev_ptr;
                param_views[i].set_descriptor(shapes[i]);
            }

            dev_ptr += get_aligned_size(shapes[i].size());
        }
    }
}

void NeuralNetwork::link_parameters_cpu()
{
    type* cpu_ptr = parameters.data();

    for(auto& layer : layers)
    {
        const vector<Shape> shapes = layer->get_parameter_shapes();
        auto& param_views = layer->get_parameter_views();

        for(size_t i = 0; i < shapes.size(); ++i)
        {
            if(shapes[i].empty()) continue;

            if(i < param_views.size())
                param_views[i] = TensorView(cpu_ptr, shapes[i]);

            cpu_ptr += get_aligned_size(shapes[i].size());
        }
    }
}

MatrixR NeuralNetwork::calculate_outputs_device(const vector<TensorView>& input_views_cpu,
                                                ForwardPropagation& fp)
{
    // Parameters → device, layer views → device pointers
    copy_parameters_device();
    link_parameters_device();

    // ForwardPropagation buffers → device, fp.views → device pointers
    fp.allocate_device();

    // Upload inputs CPU → GPU (allocate temp device buffer)
    const Index input_size = input_views_cpu[0].size();
    type* input_device = nullptr;
    CHECK_CUDA(cudaMalloc(&input_device, input_size * sizeof(type)));
    CHECK_CUDA(cudaMemcpy(input_device,
                          input_views_cpu[0].data,
                          input_size * sizeof(type),
                          cudaMemcpyHostToDevice));

    vector<TensorView> input_views_gpu = input_views_cpu;
    input_views_gpu[0].data = input_device;
    input_views_gpu[0].set_descriptor(input_views_cpu[0].shape);

    // Forward on GPU (math_utilities dispatches via Device::is_gpu)
    forward_propagate(input_views_gpu, fp, false);

    // Pick last layer output (same logic as CPU path)
    const size_t layers_count = get_layers_number();
    const TensorView out_view = (layers_count > 0
                           && layers_count - 1 < fp.views.size()
                           && fp.views[layers_count - 1].size() > 1
                           && !fp.views[layers_count - 1].back().empty())
                          ? fp.views[layers_count - 1].back()[0]
                          : fp.get_last_trainable_layer_outputs();

    // Download outputs GPU → CPU
    const Index batch_size = input_views_cpu[0].shape[0];
    const Index out_cols = out_view.size() / batch_size;
    MatrixR result(batch_size, out_cols);
    CHECK_CUDA(cudaMemcpy(result.data(),
                          out_view.data,
                          out_view.size() * sizeof(type),
                          cudaMemcpyDeviceToHost));

    // Free temp input buffer and restore CPU layer views
    cudaFree(input_device);
    link_parameters_cpu();

    return result;
}

#endif

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
