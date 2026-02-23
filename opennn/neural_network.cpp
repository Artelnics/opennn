//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   N E U R A L   N E T W O R K   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "registry.h"
#include "strings_utilities.h"
#include "images.h"
#include "tensors.h"
#include "neural_network.h"
#include "dense_layer.h"
#include "scaling_layer.h"
#include "scaling_layer.h"
#include "flatten_layer.h"
#include "convolutional_layer.h"
#include "addition_layer.h"
#include "embedding_layer.h"

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

    if (old_layers_number >= 0)
    {
        const string& name = layers[old_layers_number]->get_name();
        if(!validate_name(name)) return;
    }

    layers.push_back(std::move(layer));

    layer_input_indices.push_back(input_indices.empty()
        ? vector<Index>(1, old_layers_number )
        : input_indices);
}


vector<vector<TensorView*>> NeuralNetwork::get_layer_parameter_views()
{
    const Index layers_number = get_layers_number();

    vector<vector<TensorView*>> layer_parameter_views(layers_number);

    for(Index i = 0; i < layers_number; i++)
        layer_parameter_views[i] = layers[i]->get_parameter_views();

    return layer_parameter_views;
}


void NeuralNetwork::compile()
{
    const vector<vector<TensorView*>> layer_parameter_views = get_layer_parameter_views();

    const Index parameters_size = get_size(layer_parameter_views);

    if (parameters_size == 0) return;

    parameters.resize(parameters_size);
    parameters.setZero();

    link(parameters.data(), layer_parameter_views);

#ifdef OPENNN_CUDA

    const vector<vector<TensorViewCuda*>> layer_parameter_views_device = get_layer_parameter_views_device();

    allocate_parameters_device();

    link(parameters_device.data, layer_parameter_views_device);

#endif
}


bool NeuralNetwork::validate_name(const string& name) const
{
    if(name == "Bounding")
        throw runtime_error("No layers can be added after a bounding layer.\n");

    return true;
}


void NeuralNetwork::reference_all_layers()
{
    reference_dense_layer();
    reference_scaling_layer();
    reference_flatten_layer();
    reference_addition_layer();
    // Add more template layers
}


bool NeuralNetwork::has(const string& name) const
{
    return any_of(layers.begin(), layers.end(),
                  [&](const unique_ptr<Layer>& layer) {return layer->get_name() == name;});
}


bool NeuralNetwork::is_empty() const
{
    return layers.empty();
}


const vector<string>& NeuralNetwork::get_feature_names() const
{
    return input_names;
}


Index NeuralNetwork::get_input_index(const string& input_name) const
{
    for(Index i = 0; i < Index(input_names.size()); i++)
        if(input_names[i] == input_name)
            return i;

    throw runtime_error("Input name not found: " + input_name);
}


const vector<string>& NeuralNetwork::get_output_names() const
{
    return output_names;
}


Index NeuralNetwork::get_output_index(const string& output_name) const
{
    for(Index i = 0; i < Index(output_names.size()); i++)
        if(output_names[i] == output_name)
            return i;

    throw runtime_error("Output name not found: " + output_name);
}


const vector<unique_ptr<Layer>>& NeuralNetwork::get_layers() const
{
    return layers;
}


const unique_ptr<Layer>& NeuralNetwork::get_layer(const Index layer_index) const
{
    return layers[layer_index];
}


const unique_ptr<Layer>& NeuralNetwork::get_layer(const string& label) const
{
    const vector<string> labels = get_layer_labels();

    for(Index i = 0; i < Index(labels.size()); i++)
        if(labels[i] == label)
            return layers[i];

    throw runtime_error("Layer not found in neural network");
}


Index NeuralNetwork::get_layer_index(const string& new_label) const
{
    if(new_label == "Dataset" || new_label == "decoder")
        return -1;

    if(new_label == "input")
        return -2;

    const Index layers_number = get_layers_number();

    for(Index i = 0; i < layers_number; i++)
        if(layers[i]->get_label() == new_label)
            return i;

    throw runtime_error("Layer not found: " + new_label);
}


const vector<vector<Index>>& NeuralNetwork::get_layer_input_indices() const
{
    return layer_input_indices;
}


vector<vector<Index>> NeuralNetwork::get_layer_output_indices() const
{
    const Index layers_number = layer_input_indices.size();

    vector<vector<Index>> layer_output_indices(layers_number);

    for(Index i = 0; i < layers_number; i++)
        for(Index input_index : layer_input_indices[i])
            if (input_index >= 0)
                layer_output_indices[input_index].push_back(i);

    for(auto& outputs : layer_output_indices)
        if(outputs.empty())
            outputs.push_back(-1);

    return layer_output_indices;
}


Index NeuralNetwork::find_input_index(const vector<Index>& layer_inputs_indices, Index layer_index) const
{
    for(Index i = 0; i < Index(layer_inputs_indices.size()); i++)
        if (layer_inputs_indices[i] == layer_index)
            return i;

    return -1;
}


Layer* NeuralNetwork::get_first(const string& name) const
{
    for(const unique_ptr<Layer>& layer : layers)
        if(layer->get_name() == name)
            return layer.get();

    throw runtime_error("Layer not found in Neural Network: " + name);
}


bool NeuralNetwork::get_display() const
{
    return display;
}

const vector<string>&NeuralNetwork::get_input_vocabulary() const
{
    return input_vocabulary;
}


const vector<string>&NeuralNetwork::get_output_vocabulary() const
{
    return output_vocabulary;
}


void NeuralNetwork::set(const filesystem::path& file_name)
{
    load(file_name);
}


void NeuralNetwork::set_feature_names(const vector<string>& new_feature_names)
{
    input_names = new_feature_names;
}


void NeuralNetwork::set_output_names(const vector<string>& new_output_namess)
{
    output_names = new_output_namess;
}


void NeuralNetwork::set_input_shape(const Shape& new_input_shape)
{
    const Index total_inputs = new_input_shape.count();
    input_names.resize(total_inputs);

    if(has("Scaling2d"))
    {
        Scaling<2>* scaling_layer = static_cast<Scaling<2>*>(get_first("Scaling2d"));
        scaling_layer->set_input_shape(new_input_shape);
    }
    else if(has("Scaling3d"))
    {
        Scaling<3>* scaling_layer = static_cast<Scaling<3>*>(get_first("Scaling3d"));
        scaling_layer->set_input_shape(new_input_shape);
    }

    layers[get_first_trainable_layer_index()].get()->set_input_shape(new_input_shape);
}


void NeuralNetwork::set_default()
{
    reference_all_layers();

    display = true;

    layers.clear();

    layer_input_indices.clear();

    input_names.clear();

    output_names.clear();
}


void NeuralNetwork::set_layers_number(const Index new_layers_number)
{
    layers.resize(new_layers_number);
    layer_input_indices.resize(new_layers_number);
}


void NeuralNetwork::set_layer_input_indices(const vector<vector<Index>>& new_layer_input_indices)
{
    layer_input_indices = new_layer_input_indices;
}


void NeuralNetwork::set_layer_input_indices(const Index layer_index, const vector<Index>& new_layer_input_indices)
{
    layer_input_indices[layer_index] = new_layer_input_indices;
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
                                            const initializer_list<string>& new_layer_input_labels_list)
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

    if(has("Embedding"))
        return get_layer(0)->get_inputs_number();

    const Shape input_shape = layers[0]->get_input_shape();

    return input_shape.count();
}


Index NeuralNetwork::get_outputs_number() const
{
    if(layers.empty()) 
        return 0;

    const Layer* last_layer = layers[layers.size() - 1].get();

    const Shape output_shape = last_layer->get_output_shape();

    return output_shape.count();
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

    return layers[layers.size() - 1]->get_output_shape();
}


Index NeuralNetwork::get_parameters_number() const
{
    Index parameters_number = 0;

    const Index layers_number = get_layers_number();

    for(Index i = 0; i < layers_number; i++)
        parameters_number += layers[i]->get_parameters_number();

    return parameters_number;
}


vector<Index> NeuralNetwork::get_layer_parameter_numbers() const
{
    const Index layers_number = get_layers_number();

    vector<Index> layer_parameter_numbers(layers_number);

    #pragma omp parallel for 

    for(Index i = 0; i < layers_number; i++)
        layer_parameter_numbers[i] = layers[i]->get_parameters_number();

    return layer_parameter_numbers;
}


void NeuralNetwork::set_parameters(const VectorR& new_parameters)
{
    parameters = new_parameters;

    link(parameters.data(), get_layer_parameter_views());
}


void NeuralNetwork::set_display(bool new_display)
{
    display = new_display;
}


void NeuralNetwork::set_input_vocabulary(const vector<string>& new_input_vocabulary)
{
    input_vocabulary = new_input_vocabulary;
}


void NeuralNetwork::set_output_vocabulary(const vector<string>& new_output_vocabulary)
{
    output_vocabulary = new_output_vocabulary;
}


Index NeuralNetwork::get_layers_number() const
{
    return layers.size();
}


Index NeuralNetwork::get_first_trainable_layer_index() const
{
    const Index layers_number = get_layers_number();

    for(Index i = 0; i < layers_number; i++)
        if (layers[i]->get_is_trainable())
            return i;

    throw runtime_error("The neural network has no trainable layers: get_first_trainable_layer_index.");
}


Index NeuralNetwork::get_last_trainable_layer_index() const
{
    const Index layers_number = get_layers_number();

    for(Index i = layers_number-1; i >= 0 ; i--)
        if (layers[i]->get_is_trainable())
            return i;

    throw runtime_error("The neural network has no trainable layers: get_last_trainable_layer_index");
}


Index NeuralNetwork::get_layers_number(const string& name) const
{
    return count_if(layers.begin(), layers.end(),
                    [&](const unique_ptr<Layer>& layer) {return layer->get_name() == name;});
}


void NeuralNetwork::set_parameters_random()
{
    const Index layers_number = get_layers_number();

    for(Index i = 0; i < layers_number; i++)
        layers[i]->set_parameters_random();
}


void NeuralNetwork::set_parameters_glorot()
{
    const Index layers_number = get_layers_number();

    #pragma omp parallel for
    for(Index i = 0; i < layers_number; i++)
        layers[i]->set_parameters_glorot();
}


Tensor3 NeuralNetwork::calculate_outputs(const Tensor3& inputs_1, const Tensor3& inputs_2)
{
    const Index layers_number = get_layers_number();

    if (layers_number == 0)
        return Tensor3();

    const Index batch_size = inputs_1.dimension(0);

    ForwardPropagation forward_propagation(batch_size, this);

    const vector<TensorView> input_views = {TensorView((type*)inputs_1.data(), {{inputs_1.dimension(0), inputs_1.dimension(1), inputs_1.dimension(2)}}),
                                            TensorView((type*)inputs_2.data(), {{inputs_2.dimension(0), inputs_2.dimension(1), inputs_2.dimension(2)}})};

    forward_propagate(input_views, forward_propagation, false);

    const vector<string> layer_labels = get_layer_labels();

    const TensorView outputs_view = forward_propagation.get_outputs();

    return tensor_map<3>(outputs_view);
}


TensorView NeuralNetwork::run_internal_forward_propagation(const type *data, const Shape &shape)
{
    ForwardPropagation forward_propagation(shape[0], this);
    TensorView input_view(const_cast<type*>(data), shape);

    forward_propagate({input_view}, forward_propagation, false);

    return forward_propagation.layers.back()->get_outputs();
}


MatrixR NeuralNetwork::calculate_outputs(const MatrixR& inputs)
{
    TensorView out = run_internal_forward_propagation(inputs.data(), {inputs.rows(), inputs.cols()});

    return MatrixMap(out.data, out.shape[0], out.shape[1]);
}


MatrixR NeuralNetwork::calculate_outputs(const Tensor3& inputs)
{
    TensorView out = run_internal_forward_propagation(inputs.data(), {inputs.dimension(0), inputs.dimension(1), inputs.dimension(2)});

    return MatrixMap(out.data, out.shape[0], out.shape[1]);
}


MatrixR NeuralNetwork::calculate_outputs(const Tensor4& inputs)
{
    TensorView out = run_internal_forward_propagation(inputs.data(), {inputs.dimension(0), inputs.dimension(1), inputs.dimension(2), inputs.dimension(3)});
    return MatrixMap(out.data, out.shape[0], out.shape[1]);
}


void NeuralNetwork::forward_propagate(const vector<TensorView>& input_view,
                                      ForwardPropagation& forward_propagation,
                                      bool is_training) const
{
    const Index layers_number = get_layers_number();

    const Index first_layer_index = is_training
                                  ? get_first_trainable_layer_index()
                                  : 0;

    const Index last_layer_index = is_training
                                 ? get_last_trainable_layer_index()
                                 : layers_number - 1;

    const vector<vector<TensorView>> layer_input_views
        = forward_propagation.get_layer_input_views(input_view, is_training);

    for(Index i = first_layer_index; i <= last_layer_index; i++)
        layers[i]->forward_propagate(layer_input_views[i],
            forward_propagation.layers[i],
            is_training);
}


void NeuralNetwork::forward_propagate(const vector<TensorView>& input_view,
                                      const VectorR& new_parameters,
                                      ForwardPropagation& forward_propagation)
{
    const VectorR original_parameters = get_parameters();

    set_parameters(new_parameters);

    forward_propagate(input_view, forward_propagation, true);

    set_parameters(original_parameters);
}


string NeuralNetwork::get_expression() const
{
    const Index layers_number = get_layers_number();

    const vector<string> layer_labels = get_layer_labels();

    vector<string> new_feature_names = get_feature_names();
    vector<string> new_output_names = get_output_names();

    const Index inputs_number = input_names.size();
    const Index outputs_number = output_names.size();

    for(Index i = 0; i < inputs_number; i++)
        input_names[i].empty()
            ? new_feature_names[i] = "input_" + to_string(i)
            : new_feature_names[i] = input_names[i];

    ostringstream buffer;

    for(Index i = 0; i < layers_number; i++)
    {
        if (i == layers_number - 1)
        {
            for(Index j = 0; j < outputs_number; j++)
                new_output_names[j] = !output_names[j].empty()
                      ? output_names[j]
                      : "output_" + to_string(i);

            buffer << layers[i]->get_expression(new_feature_names, new_output_names) << endl;
        }
        else
        {
            const Index layer_neurons_number = layers[i]->get_outputs_number();

            new_output_names.resize(layer_neurons_number);
            
            for(Index j = 0; j < layer_neurons_number; j++)
                new_output_names[j] = (layer_labels[i] == "scaling_layer")
                      ? "scaled_" + input_names[j]
                      : layer_labels[i] + "_output_" + to_string(j);

            buffer << layers[i]->get_expression(new_feature_names, new_output_names) << endl;
            new_feature_names = new_output_names;
        }
    }

    return buffer.str();
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
    // Tensor3 image = load_image(image_path);

    // Scaling4d* scaling_layer = static_cast<Scaling4d*>(get_first("Scaling4d"));

    // const Index height = scaling_layer->get_input_shape()[0];
    // const Index width = scaling_layer->get_input_shape()[1];
    // const Index channels = scaling_layer->get_input_shape()[2];

    // const Index current_height = image.dimension(0);
    // const Index current_width = image.dimension(1);
    // const Index current_channels = image.dimension(2);

    // if (current_channels != channels)
    //     throw runtime_error("Error: Different channels number " + image_path.string() + "\n");

    // if(current_height != height || current_width != width)
    //     image = resize_image(image, height, width);

    // Tensor4 input_data(1, height, width, channels);

    // const Index pixels_number = height * width * channels;

    // #pragma omp parallel for
    // for(Index j = 0; j < pixels_number; j++)
    //     input_data(j) = image(j);

    // const Matrix outputs = calculate_outputs(input_data);

    // Index predicted_index = -1;

    // if (outputs.size() > 1)
    // {
    //     type max_value = outputs(0);

    //     for(Index i = 1; i < outputs.dimension(1); ++i)
    //     {
    //         if (outputs(i) > max_value)
    //         {
    //             max_value = outputs(i);
    //             predicted_index = i;
    //         }
    //     }
    // }
    // else
    //     predicted_index = outputs(0);

    // return predicted_index;
    return 0;
}


MatrixR NeuralNetwork::calculate_text_outputs(const Tensor<string, 1>&input_documents)
{
    const Index batch_size = input_documents.dimension(0);

    const Embedding* embedding_layer = static_cast<const Embedding*>(get_layer(0).get());

    const Index sequence_length = embedding_layer->get_sequence_length();

    unordered_map<string, Index> vocabulary_map;
    vocabulary_map.reserve(input_vocabulary.size());
    for(Index i = 0; i < (Index)input_vocabulary.size(); ++i)
        vocabulary_map[input_vocabulary[i]] = i;

    MatrixR inputs(batch_size, sequence_length);
    inputs.setConstant(0.0);

    #pragma omp parallel for
    for(Index i = 0; i < batch_size; ++i)
    {
        const vector<string> tokens = tokenize(input_documents(i));

        Index current_index = 0;

        // Start
        if(current_index < sequence_length)
        {
            inputs(i, current_index) = 2;
            current_index++;
        }

        // Tokens
        for(const string& token : tokens)
        {
            if(current_index >= sequence_length - 1)
                break;

            const auto it = vocabulary_map.find(token);

            if(it != vocabulary_map.end())
                inputs(i, current_index) = (type)it->second;
            else
                inputs(i, current_index) = 1;

            current_index++;
        }

        // End
        if(current_index < sequence_length)
            inputs(i, current_index) = 3;
    }

    return calculate_outputs(inputs);
}


Tensor<string, 2> NeuralNetwork::get_dense2d_layers_information() const
{
    const Index layers_number = get_layers_number();

    Index dense2d_layers_number = 0;

    for(Index i = 0; i < layers_number; i++)
        if (layers[i]->get_name() == "Dense2d" && layers[i]->get_label().find("classification") == string::npos)
            dense2d_layers_number++;

    Tensor<string, 2> information(dense2d_layers_number, 4);

    Index dense2d_layer_index = 0;

    for(Index i = 0; i < layers_number; i++)
    {
        const string& name = layers[i]->get_name();
        const string label = layers[i]->get_label();

        if (name != "Dense2d" || label.find("classification") != string::npos)
            continue;

        information(dense2d_layer_index, 0) = label;
        information(dense2d_layer_index, 1) = to_string(layers[i]->get_input_shape()[0]);
        information(dense2d_layer_index, 2) = to_string(layers[i]->get_output_shape()[0]);

        const Dense<2>* dense2d_layer = static_cast<Dense<2>*>(layers[i].get());

        information(dense2d_layer_index, 3) = dense2d_layer->get_activation_function();

        dense2d_layer_index++;
    }

    return information;
}


void NeuralNetwork::to_XML(XMLPrinter& printer) const
{
    const Index inputs_number = get_inputs_number();
    const Index layers_number = get_layers_number();
    const Index outputs_number = get_outputs_number();

    printer.OpenElement("NeuralNetwork");

    // Features

    printer.OpenElement("Features");

    add_xml_element(printer, "FeaturesNumber", to_string(inputs_number));

    for(Index i = 0; i < inputs_number; i++)
        add_xml_element_attribute(printer, "Feature", input_names[i], "Index", to_string(i + 1));

    printer.CloseElement();

    // Layers

    printer.OpenElement("Layers");

    add_xml_element(printer, "LayersNumber", to_string(layers_number));

    for(Index i = 0; i < layers_number; i++)
        layers[i]->to_XML(printer);

    // Layer input indices

    printer.OpenElement("LayerInputIndices");

    for(Index i = 0; i < Index(layer_input_indices.size()); i++) 
        add_xml_element_attribute(printer, "LayerInputsIndices", vector_to_string(layer_input_indices[i]), "LayerIndex", to_string(i));

    printer.CloseElement();

    printer.CloseElement();

    // Outputs

    printer.OpenElement("Outputs");

    const Index outputs_count = has("Embedding") ? outputs_number : output_names.size();
    add_xml_element(printer, "OutputsNumber", to_string(outputs_count));

    for(Index i = 0; i < outputs_count; i++)
        add_xml_element_attribute(printer, "Output", output_names[i], "Index", to_string(i + 1));

    printer.CloseElement();

    add_xml_element(printer, "Display", to_string(display));

    printer.CloseElement();
}


void NeuralNetwork::from_XML(const XMLDocument& document)
{
    const XMLElement* neural_network_element = document.FirstChildElement("NeuralNetwork");

    if(!neural_network_element)
        throw runtime_error("Neural network element is nullptr.\n");

    features_from_XML(neural_network_element->FirstChildElement("Features"));
    layers_from_XML(neural_network_element->FirstChildElement("Layers"));
    outputs_from_XML(neural_network_element->FirstChildElement("Outputs"));
    set_display(read_xml_bool(neural_network_element, "Display"));
}


void NeuralNetwork::features_from_XML(const XMLElement* features_element)
{
    if(!features_element)
        throw runtime_error("Features element is nullptr.\n");

    const Index new_features_number = read_xml_index(features_element, "FeaturesNumber");
    input_names.resize(new_features_number);

    // Feature names

    const XMLElement* start_element = features_element->FirstChildElement("FeaturesNumber");

    for(Index i = 0; i < new_features_number; i++)
    {
        const XMLElement* feature_element = start_element->NextSiblingElement("Feature");
        start_element = feature_element;

        if(feature_element->Attribute("Index") != to_string(i+1))
            throw runtime_error("Feature index number (" + to_string(i+1) + ") does not match (" + feature_element->Attribute("Item") + ").\n");

        if(feature_element->GetText())
            input_names[i] = feature_element->GetText();
    }
}


void NeuralNetwork::layers_from_XML(const XMLElement* layers_element)
{
    if(!layers_element)
        throw runtime_error("Layers element is nullptr.\n");

    const Index layers_number = read_xml_index(layers_element, "LayersNumber");

    layers.clear();
    layer_input_indices.clear();

    const XMLElement* start_element = layers_element->FirstChildElement("LayersNumber");

    for(Index i = 0; i < layers_number; i++)
    {
        const XMLElement* layer_element = start_element->NextSiblingElement();

        if(!layer_element)
            throw runtime_error("Layer element is nullptr.");

        const string name_string = layer_element->Name();
        unique_ptr<Layer> layer = Registry<Layer>::instance().create(name_string);

        XMLDocument layer_document;
        XMLNode* element_clone = layer_element->DeepClone(&layer_document);
        layer_document.InsertFirstChild(element_clone);

        layer->from_XML(layer_document);
        add_layer(std::move(layer));

        start_element = layer_element;
    }

    const XMLElement* layer_input_indices_element = layers_element->FirstChildElement("LayerInputIndices");
    if(!layer_input_indices_element)
        throw runtime_error("LayerInputIndices element is nullptr.\n");

    for(const XMLElement* layer_inputs_indices_element = layer_input_indices_element->FirstChildElement("LayerInputsIndices");
         layer_inputs_indices_element;
         layer_inputs_indices_element = layer_inputs_indices_element->NextSiblingElement("LayerInputsIndices"))
    {
        int layer_index;
        if (layer_inputs_indices_element->QueryIntAttribute("LayerIndex", &layer_index) != tinyxml2::XML_SUCCESS)
            throw runtime_error("Error: LayerIndex attribute missing or invalid.\n");

        const char* text = layer_inputs_indices_element->GetText();
        if(!text)
            throw runtime_error("Text is nullptr for LayerInputsIndices element.");

        Shape input_shape = string_to_shape(string(text), " ");
        layer_input_indices[layer_index] = vector<Index>(input_shape.begin(), input_shape.end());
    }
}


void NeuralNetwork::outputs_from_XML(const XMLElement* outputs_element)
{
    if(!outputs_element)
        throw runtime_error("Outputs element is nullptr.\n");

    const Index new_outputs_number = read_xml_index(outputs_element, "OutputsNumber");

    output_names.resize(new_outputs_number);

    const XMLElement* outputs_number_element = outputs_element->FirstChildElement("OutputsNumber");

    const XMLElement* start_element = outputs_number_element;

    for(Index i = 0; i < new_outputs_number; i++)
    {
        const XMLElement* output_element = start_element->NextSiblingElement("Output");
        start_element = output_element;

        if(output_element->Attribute("Index") != to_string(i+1))
            throw runtime_error("Output index number (" + to_string(i+1) + ") does not match (" + output_element->Attribute("Item") + ").\n");

        if(output_element->GetText())
            output_names[i] = output_element->GetText();
    }
}


void NeuralNetwork::print() const
{
    cout << "Neural network" << endl;

    cout << "Features number: " << get_inputs_number() << endl;

    const Index layers_number = get_layers_number();

    cout << "Layers number: " << layers_number << endl;

    for(Index i = 0; i < layers_number; i++)
        cout << "\nLayer " << i << ": " << layers[i]->get_name() << endl;
    
    cout << "Outputs number: " << get_outputs_number() << endl;

    cout << "Outputs:" << endl
         << get_output_names();

    cout << "Parameters number: " << get_parameters_number() << endl;

    // if(!input_vocabulary.empty())
    //     cout << "Input vocabulary" << input_vocabulary <<

}


void NeuralNetwork::save(const filesystem::path& file_name) const
{
    ofstream file(file_name);

    if(!file.is_open())
        return;

    XMLPrinter printer;
    to_XML(printer);
    file << printer.CStr();
}


void NeuralNetwork::save_parameters(const filesystem::path& file_name) const
{
    ofstream file(file_name);

    if(!file.is_open())
        throw runtime_error("Cannot open parameters data file.\n");

    file << parameters << endl;

    file.close();
}


void NeuralNetwork::load(const filesystem::path& file_name)
{
    set_default();

    XMLDocument document;

    if (document.LoadFile(file_name.string().c_str()))
        throw runtime_error("Cannot load XML file " + file_name.string() + ".\n");

    from_XML(document);
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

    const vector<string> output_names = get_output_names();

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

    const vector<string> output_names = get_output_names();
    const Index outputs_number = get_outputs_number();

    const vector<string> input_names = get_feature_names();
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
    const Index layers_number = get_layers_number();

    vector<string> layer_labels(layers_number);

    for(Index i = 0; i < layers_number; i++)
        layer_labels[i] = layers[i]->get_label();

    return layer_labels;
}


vector<string> NeuralNetwork::get_names_string() const
{
    const Index layers_number = get_layers_number();

    vector<string> names(layers_number);

    for(Index i = 0; i < layers_number; i++)
        names[i] = layers[i]->get_name();

    return names;
}


NeuralNetworkBackPropagation::NeuralNetworkBackPropagation(const Index new_batch_size, 
                                                           NeuralNetwork* new_neural_network)
{
    set(new_batch_size, new_neural_network);
}


void NeuralNetworkBackPropagation::set(const Index new_batch_size, NeuralNetwork* new_neural_network)
{
    batch_size = new_batch_size;

    neural_network = new_neural_network;

    if(!neural_network) return;

    const vector<unique_ptr<Layer>>& neural_network_layers = neural_network->get_layers();
    const Index first_traineable_layer_number = neural_network->get_first_trainable_layer_index();
    const Index last_traineable_layer_number = neural_network->get_last_trainable_layer_index();

    const Index layers_number = neural_network_layers.size();

    layers.resize(layers_number);

    for(Index i = first_traineable_layer_number; i <= last_traineable_layer_number; i++)
    {
        layers[i] = Registry<LayerBackPropagation>::instance().create(neural_network_layers[i]->get_name());
        layers[i]->set(batch_size, neural_network_layers[i].get());
    }

    const vector<vector<TensorView*>> layer_gradient_views = get_layer_gradient_views();

    const Index gradient_size = get_size(layer_gradient_views);

    if (gradient_size == 0) return;

    gradient.resize(gradient_size);
    gradient.setZero();

    link(gradient.data(), layer_gradient_views);
}


const vector<unique_ptr<LayerBackPropagation>>& NeuralNetworkBackPropagation::get_layers() const
{
    return layers;
}


vector<vector<TensorView *>> NeuralNetworkBackPropagation::get_layer_gradient_views()
{
    const size_t layers_number = layers.size();

    vector<vector<TensorView*>> layer_gradient_views(layers_number);

    for (size_t i = 0; i < layers_number; i++)
        if (layers[i])
            layer_gradient_views[i] = layers[i]->get_gradient_views();


    return layer_gradient_views;
}


NeuralNetwork* NeuralNetworkBackPropagation::get_neural_network() const
{
    return neural_network;
}


void NeuralNetworkBackPropagation::print() const
{
    cout << "Neural network back-propagation" << endl;

    const Index layers_number = layers.size();

    for(Index i = 0; i < layers_number; i++)
    {
        cout << "Layer " << i << ": "
             << neural_network->get_layer(i)->get_name() << endl;

        if(!layers[i]) continue;

        layers[i]->print();
    }
}


ForwardPropagation::ForwardPropagation(const Index new_batch_size, NeuralNetwork* new_neural_network)
{
    set(new_batch_size, new_neural_network);
}


void ForwardPropagation::set(const Index new_samples_number, NeuralNetwork* new_neural_network)
{
    samples_number = new_samples_number;

    neural_network = new_neural_network;

    if(!neural_network) throw runtime_error("There is no neural network.");

    const vector<unique_ptr<Layer>>& neural_network_layers = neural_network->get_layers();

    const Index layers_number = neural_network_layers.size();

    layers.resize(layers_number);

    for(Index i = 0; i < layers_number; i++)
    {
        layers[i] = Registry<LayerForwardPropagation>::instance().create(neural_network_layers[i]->get_name());
        layers[i]->set(samples_number, neural_network_layers[i].get());
    }

    const vector<vector<TensorView*>> layer_workspace_views = get_layer_workspace_views();

    const Index workspace_size = get_size(layer_workspace_views);

    if (workspace_size == 0) return;

    workspace.resize(workspace_size);
    workspace.setZero();

    link(workspace.data(), layer_workspace_views);
}


vector<vector<TensorView*>> ForwardPropagation::get_layer_workspace_views()
{
    const Index layers_number = neural_network->get_layers_number();

    vector<vector<TensorView*>> layer_workspace_views(layers_number);

    for (Index i = 0; i < layers_number; i++)
        layer_workspace_views[i] = layers[i]->get_workspace_views();

    return layer_workspace_views;
}


TensorView ForwardPropagation::get_last_trainable_layer_outputs() const
{
    const Index last_trainable_layer_index = neural_network->get_last_trainable_layer_index();

    const unique_ptr<LayerForwardPropagation>& layer_forward_propagation = layers[last_trainable_layer_index];

    return layer_forward_propagation->get_outputs();
}


vector<vector<TensorView>> ForwardPropagation::get_layer_input_views(const vector<TensorView>& batch_input_views,
                                                                     bool is_training) const
{
    const Index layers_number = neural_network->get_layers_number();

    if (layers_number == 0)
        return {};

    const vector<vector<Index>>& layer_input_indices = neural_network->get_layer_input_indices();

    vector<vector<TensorView>> layer_input_views(layers_number);

    layer_input_views[0] = batch_input_views;

    const Index first_trainable_layer_index = neural_network->get_first_trainable_layer_index();

    for(Index layer_index = first_trainable_layer_index; layer_index < layers_number; layer_index++)
    {
        if((layer_index == first_trainable_layer_index && is_training) || layer_index == 0)
        {
            layer_input_views[layer_index] = batch_input_views;
            continue;
        }

        const vector<Index>& input_layer_indices = layer_input_indices[layer_index];
        layer_input_views[layer_index].resize(input_layer_indices.size());

        for(Index input_index = 0; input_index < static_cast<Index>(input_layer_indices.size()); input_index++)
        {
            const Index input_layer_index = input_layer_indices[input_index];
            layer_input_views[layer_index][input_index] = layers[input_layer_index]->get_outputs();
        }
    }

    return layer_input_views;
}

TensorView ForwardPropagation::get_outputs()
{
    return layers.back()->get_outputs();
}


void ForwardPropagation::print() const
{
    cout << "Neural network forward propagation" << endl;

    const Index layers_number = layers.size();

    cout << "Layers number: " << layers_number << endl;

    for(Index i = 0; i < layers_number; i++)
    {
        cout << "Layer " << i + 1 << ": " << neural_network->get_layer(i)->get_label() << endl;

        layers[i]->print();
    }
}


NeuralNetworkBackPropagationLM::NeuralNetworkBackPropagationLM(NeuralNetwork *new_neural_network)
{
    neural_network = new_neural_network;
}

void NeuralNetworkBackPropagationLM::set(const Index new_batch_size,
                                         NeuralNetwork* new_neural_network)
{
    batch_size = new_batch_size;
    neural_network = new_neural_network;

    if(!neural_network) return;

    const Index layers_number = neural_network->get_layers_number();
    const vector<unique_ptr<Layer>>& neural_network_layers = neural_network->get_layers();

    const Index outputs_number = neural_network->get_outputs_number();
    const Index total_error_terms = batch_size * outputs_number;

    layers.clear();
    layers.resize(layers_number);

    const Index first_trainable = neural_network->get_first_trainable_layer_index();
    const Index last_trainable = neural_network->get_last_trainable_layer_index();

    for(Index i = first_trainable; i <= last_trainable; i++)
        layers[i] = make_unique<DenseBackPropagationLM>(total_error_terms, neural_network_layers[i].get());

    const vector<vector<TensorView*>> layer_workspace_views = get_layer_workspace_views();
    const Index workspace_size = get_size(layer_workspace_views);

    if(workspace_size == 0) return;

    workspace.resize(workspace_size);
    workspace.setZero();

    link(workspace.data(), layer_workspace_views);
}

const vector<unique_ptr<LayerBackPropagationLM> >&NeuralNetworkBackPropagationLM::get_layers() const
{
    return layers;
}

NeuralNetwork *NeuralNetworkBackPropagationLM::get_neural_network() const
{
    return neural_network;
}


vector<vector<TensorView *>> NeuralNetworkBackPropagationLM::get_layer_workspace_views()
{
    const Index layers_number = neural_network->get_layers_number();

    vector<vector<TensorView*>> layer_workspace_views(layers_number);

    for(Index i = 0; i < layers_number; i++)
        if(layers[i])
            layer_workspace_views[i] = layers[i]->get_gradient_views();

    return layer_workspace_views;
}


void NeuralNetworkBackPropagationLM::print()
{
    const Index layers_number = layers.size();

    cout << "Layers number: " << layers_number << endl;

    for(Index i = 0; i < layers_number; i++)
    {
        cout << "Layer " << i + 1 << endl;

        layers[i]->print();
    }
}


#ifdef OPENNN_CUDA

void NeuralNetwork::allocate_parameters_device()
{
    parameters_device.resize({ parameters.size() });
}


void NeuralNetwork::copy_parameters_device()
{
    // Convolutional weights need custom order for gpu
    for (const unique_ptr<Layer>& layer : layers)
        if (auto* conv = dynamic_cast<Convolutional*>(layer.get()))
            conv->reorder_weights_for_cudnn();

    CHECK_CUDA(cudaMemcpy(parameters_device.data, parameters.data(), parameters.size() * sizeof(type), cudaMemcpyHostToDevice));

    for (const unique_ptr<Layer>& layer : layers)
        if (auto* conv = dynamic_cast<Convolutional*>(layer.get()))
            conv->reorder_weights_for_cudnn();
}


void NeuralNetwork::copy_parameters_host()
{
    CHECK_CUDA(cudaMemcpy(parameters.data(), parameters_device.data, parameters.size() * sizeof(type), cudaMemcpyDeviceToHost));

    // Convolutional weights need custom order for gpu
    for (const unique_ptr<Layer>& layer : layers)
        if (auto* conv = dynamic_cast<Convolutional*>(layer.get()))
            conv->reorder_weights_for_cudnn();
}


vector<vector<TensorViewCuda*>> NeuralNetwork::get_layer_parameter_views_device()
{
    const Index layers_number = get_layers_number();

    vector<vector<TensorViewCuda*>> layer_parameter_views(layers_number);

    for(Index i = 0; i < layers_number; i++)
        layer_parameter_views[i] = layers[i]->get_parameter_views_device();

    return layer_parameter_views;
}


void NeuralNetwork::forward_propagate(const vector<TensorViewCuda>& input_views_device,
                                      ForwardPropagationCuda& forward_propagation,
                                      bool is_training) const
{
    const Index layers_number = get_layers_number();

    const Index first_layer_index = is_training
                                        ? get_first_trainable_layer_index()
                                        : 0;

    const Index last_layer_index = is_training
                                       ? get_last_trainable_layer_index()
                                       : layers_number - 1;

    const vector<vector<TensorViewCuda>> layer_input_views_device
        = forward_propagation.get_layer_input_views_device(input_views_device, is_training);

    for (Index i = first_layer_index; i <= last_layer_index; i++)
        layers[i]->forward_propagate(layer_input_views_device[i],
                                     forward_propagation.layers[i],
                                     is_training);
}


TensorViewCuda NeuralNetwork::calculate_outputs(TensorViewCuda input_device, Index batch_size)
{
    if (layers.empty())
        return TensorViewCuda();

    ForwardPropagationCuda forward_propagation(batch_size, this);

    forward_propagate({ input_device }, forward_propagation, false);

    return forward_propagation.get_last_trainable_layer_outputs_device();
}


TensorCuda &NeuralNetwork::get_parameters_device()
{
    return parameters_device;
}


ForwardPropagationCuda::ForwardPropagationCuda(const Index new_batch_size, NeuralNetwork* new_neural_network)
{
    set(new_batch_size, new_neural_network);
}


void ForwardPropagationCuda::set(const Index new_samples_number, NeuralNetwork* new_neural_network)
{
    samples_number = new_samples_number;

    neural_network = new_neural_network;

    if(!neural_network) throw runtime_error("There is no neural network.");

    const vector<unique_ptr<Layer>>& neural_network_layers = neural_network->get_layers();

    const Index layers_number = neural_network_layers.size();

    layers.resize(layers_number);

    for(Index i = 0; i < layers_number; i++)
    {
        const auto& current_layer = neural_network_layers[i];

        layers[i] = Registry<LayerForwardPropagationCuda>::instance().create(current_layer->get_name());
        layers[i]->set(samples_number, current_layer.get());
    }

    const vector<vector<TensorViewCuda*>> layer_workspace_views = get_layer_workspace_views_device();

    const Index workspace_size = get_size(layer_workspace_views);

    if (workspace_size == 0) return;

    workspace.resize({workspace_size});

    link(workspace.data, layer_workspace_views);
}


vector<vector<TensorViewCuda*>> ForwardPropagationCuda::get_layer_workspace_views_device()
{
    const Index layers_number = neural_network->get_layers_number();

    vector<vector<TensorViewCuda*>> layer_workspace_views(layers_number);

    for (Index i = 0; i < layers_number; i++)
        layer_workspace_views[i] = layers[i]->get_workspace_views();

    return layer_workspace_views;
}


TensorViewCuda ForwardPropagationCuda::get_last_trainable_layer_outputs_device() const
{
    const Index last_trainable_layer_index = neural_network->get_last_trainable_layer_index();

    const unique_ptr<LayerForwardPropagationCuda>& layer_forward_propagation = layers[last_trainable_layer_index];

    return layer_forward_propagation->get_outputs();
}


vector<vector<TensorViewCuda>> ForwardPropagationCuda::get_layer_input_views_device(const vector<TensorViewCuda>& batch_input_views,
                                                                                    bool is_training) const
{
    const Index layers_number = neural_network->get_layers_number();

    if (layers_number == 0)
        return {};

    const vector<vector<Index>>& layer_input_indices = neural_network->get_layer_input_indices();

    vector<vector<TensorViewCuda>> layer_input_views(layers_number);

    layer_input_views[0] = batch_input_views;

    const Index first_trainable_layer_index = neural_network->get_first_trainable_layer_index();

    for(Index layer_index = first_trainable_layer_index; layer_index < layers_number; layer_index++)
    {
        if((layer_index == first_trainable_layer_index && is_training) || layer_index == 0)
        {
            layer_input_views[layer_index] = batch_input_views;
            continue;
        }

        const vector<Index>& input_layer_indices = layer_input_indices[layer_index];
        layer_input_views[layer_index].resize(input_layer_indices.size());

        for(Index input_index = 0; input_index < static_cast<Index>(input_layer_indices.size()); input_index++)
        {
            const Index input_layer_index = input_layer_indices[input_index];
            layer_input_views[layer_index][input_index] = layers[input_layer_index]->get_outputs();
        }
    }

    return layer_input_views;
}


void ForwardPropagationCuda::print()
{
    const Index layers_number = layers.size();

    cout << "Layers number: " << layers_number << endl;

    for(Index i = 0; i < layers_number; i++)
    {
        cout << "Layer " << i + 1 << endl;

        layers[i]->print();
    }
}


void ForwardPropagationCuda::free()
{
    for (unique_ptr<LayerForwardPropagationCuda>& layer : layers)
        if (layer) layer->free();
}


NeuralNetworkBackPropagationCuda::NeuralNetworkBackPropagationCuda(const Index new_batch_size, NeuralNetwork* new_neural_network)
{
    set(new_batch_size, new_neural_network);
}


void NeuralNetworkBackPropagationCuda::set(const Index new_batch_size, NeuralNetwork* new_neural_network)
{
    batch_size = new_batch_size;

    neural_network = new_neural_network;

    if (!neural_network) return;

    const vector<unique_ptr<Layer>>& neural_network_layers = neural_network->get_layers();

    const Index layers_number = neural_network_layers.size();
    const Index first_traineable_layer_number = neural_network->get_first_trainable_layer_index();
    const Index last_traineable_layer_number = neural_network->get_last_trainable_layer_index();

    layers.resize(layers_number);

    for (Index i = first_traineable_layer_number; i <= last_traineable_layer_number; i++)
    {
        layers[i] = Registry<LayerBackPropagationCuda>::instance().create(neural_network_layers[i]->get_name());
        layers[i]->set(batch_size, neural_network_layers[i].get());
    }

    const vector<vector<TensorViewCuda*>> layer_workspace_views = get_layer_workspace_views_device();

    const Index workspace_size = get_size(layer_workspace_views);

    if (workspace_size == 0) return;

    workspace.resize({workspace_size});

    link(workspace.data, layer_workspace_views);
}


const vector<unique_ptr<LayerBackPropagationCuda>>& NeuralNetworkBackPropagationCuda::get_layers() const
{
    return layers;
}


vector<vector<TensorViewCuda*>> NeuralNetworkBackPropagationCuda::get_layer_workspace_views_device()
{
    vector<vector<TensorViewCuda*>> layer_workspace_views(layers.size());
    Index i = 0;

    for (const auto& layer_bp : layers)
    {
        if (layer_bp)
            layer_workspace_views[i] = layer_bp->get_workspace_views();

        i++;
    }

    return layer_workspace_views;
    
}


void NeuralNetworkBackPropagationCuda::print()
{
    const Index layers_number = layers.size();

    cout << "Layers number: " << layers_number << endl;

    for(Index i = 0; i < layers_number; i++)
    {
        cout << "Layer " << i + 1 << endl;

        layers[i]->print();
    }
}


void NeuralNetworkBackPropagationCuda::free()
{
    for (unique_ptr<LayerBackPropagationCuda>& layer : layers)
        if (layer) layer->free();
}

#endif

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// This library is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 2.1 of the License, or any later version.
// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.
// You should have received a copy of the GNU Lesser General Public
// License along with this library; if not, write to the Free Software
// Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
