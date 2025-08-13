//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   N E U R A L   N E T W O R K   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "registry.h"
#include "images.h"
#include "neural_network.h"
#include "perceptron_layer.h"
#include "scaling_layer_2d.h"
#include "flatten_layer.h"
#include "addition_layer_3d.h"

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


bool NeuralNetwork::validate_name(const string& name) const
{
    if(name == "Bounding")
        throw runtime_error("No layers can be added after a bounding layer.\n");

    return true;
}


void NeuralNetwork::reference_all_layers()
{
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


const vector<string>& NeuralNetwork::get_input_names() const
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
    for(size_t i = 0; i < output_names.size(); i++)
        if(output_names[i] == output_name)
            return i;

    throw runtime_error("Output name not found: " + output_name);
}


const vector<unique_ptr<Layer>>& NeuralNetwork::get_layers() const
{
    return layers;
}


const unique_ptr<Layer>& NeuralNetwork::get_layer(const Index& layer_index) const
{
    return layers[layer_index];
}


const unique_ptr<Layer>& NeuralNetwork::get_layer(const string& label) const
{
    const vector<string> labels = get_layer_labels();

    for(size_t i = 0; i < labels.size(); i++)
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

    for (Index i = 0; i < layers_number; i++)
        for (Index input_index : layer_input_indices[i])
            if (input_index >= 0)
                layer_output_indices[input_index].push_back(i);

    for(auto& outputs : layer_output_indices)
        if(outputs.empty())
            outputs.push_back(-1);

    return layer_output_indices;
}


Index NeuralNetwork::find_input_index(const vector<Index>& layer_inputs_indices, const Index& layer_index) const
{
    for (Index i = 0; i < Index(layer_inputs_indices.size()); i++)
        if (layer_inputs_indices[i] == layer_index)
            return i;

    return -1;
}


Layer* NeuralNetwork::get_first(const string& name) const
{
    for(const unique_ptr<Layer>& layer : layers)
        if(layer->get_name() == name)
            return layer.get();

    throw runtime_error("Neural network is empty.");
}


const bool& NeuralNetwork::get_display() const
{
    return display;
}


void NeuralNetwork::set(const filesystem::path& file_name)
{
    load(file_name);
}


void NeuralNetwork::set_input_names(const vector<string>& new_input_names)
{
    input_names = new_input_names;
}


void NeuralNetwork::set_output_names(const vector<string>& new_output_namess)
{
    output_names = new_output_namess;
}


void NeuralNetwork::set_input_dimensions(const dimensions& new_input_dimensions)
{
    input_names.resize(new_input_dimensions[0]);

    if(has("Scaling2d"))
    {
        Scaling2d* scaling_layer = static_cast<Scaling2d*>(get_first("Scaling2d"));

        scaling_layer->set_input_dimensions(new_input_dimensions);
    }

    layers[get_first_trainable_layer_index()].get()->set_input_dimensions(new_input_dimensions);
}


void NeuralNetwork::set_default()
{
    reference_all_layers();

    display = true;

    layer_input_indices.clear();
}


void NeuralNetwork::set_threads_number(const int& new_threads_number)
{
    for (const unique_ptr<Layer>& layer : layers)
        layer->set_threads_number(new_threads_number);
}


void NeuralNetwork::set_layers_number(const Index& new_layers_number)
{
    layers.resize(new_layers_number);
    layer_input_indices.resize(new_layers_number);
}


void NeuralNetwork::set_layer_input_indices(const vector<vector<Index>>& new_layer_input_indices)
{
    layer_input_indices = new_layer_input_indices;
}


void NeuralNetwork::set_layer_inputs_indices(const Index& layer_index, const vector<Index>& new_layer_input_indices)
{
    layer_input_indices[layer_index] = new_layer_input_indices;
}


void NeuralNetwork::set_layer_inputs_indices(const string& layer_label,
                                             const vector<string>& new_layer_input_labels)
{
    const Index layer_index = get_layer_index(layer_label);

    const Index size = new_layer_input_labels.size();

    vector<Index> new_layer_input_indices(size);

    for(Index i = 0; i < size; i++)
        new_layer_input_indices[i] = get_layer_index(new_layer_input_labels[i]);

    layer_input_indices[layer_index] = new_layer_input_indices;
}


void NeuralNetwork::set_layer_inputs_indices(const string& layer_label,
                                             const initializer_list<string>& new_layer_input_labels_list)
{
    set_layer_inputs_indices(layer_label, vector<string>(new_layer_input_labels_list));
}


void NeuralNetwork::set_layer_inputs_indices(const string& layer_label, const string& new_layer_input_labels)
{
    const Index layer_index = get_layer_index(layer_label);

    layer_input_indices[layer_index] = {get_layer_index(new_layer_input_labels)};
}


Index NeuralNetwork::get_inputs_number() const
{
    if(layers.empty())
        return 0;

    if(has("Embedding"))
        return input_names.size();

    const dimensions input_dimensions = layers[0]->get_input_dimensions();

    return accumulate(input_dimensions.begin(), input_dimensions.end(), Index(1), multiplies<Index>());
}


Index NeuralNetwork::get_outputs_number() const
{
    if(layers.empty()) 
        return 0;

    const Layer* last_layer = layers[layers.size() - 1].get();

    const dimensions output_dimensions = last_layer->get_output_dimensions();

    return accumulate(output_dimensions.begin(), output_dimensions.end(), Index(1), multiplies<Index>());
}


dimensions NeuralNetwork::get_input_dimensions() const
{
    if(layers.empty())
        return {};

    return layers[0]->get_input_dimensions();
}


dimensions NeuralNetwork::get_output_dimensions() const
{
    if(layers.empty()) 
        return {};

    return layers[layers.size() - 1]->get_output_dimensions();
}


Index NeuralNetwork::get_parameters_number() const
{
    Index parameters_number = 0;

    for (const auto& layer : layers)
        parameters_number += layer->get_parameters_number();

    return parameters_number;
}


void NeuralNetwork::get_parameters(Tensor<type, 1>& parameters) const
{
    const Index parameters_number = get_parameters_number();

    parameters.resize(parameters_number);

    Index position = 0;

    for (const unique_ptr<Layer>& layer : layers)
    {
        const vector<pair<type*, Index>> layer_parameter_pairs = layer->get_parameter_pairs();

        for(const pair<type*, Index>& parameter_pair : layer_parameter_pairs)
        {
            memcpy(parameters.data() + position, parameter_pair.first, parameter_pair.second * sizeof(type));
            position += parameter_pair.second;
        }
    }
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


void NeuralNetwork::set_parameters(const Tensor<type, 1>& new_parameters)
{
    if (new_parameters.size() != get_parameters_number())
        throw runtime_error("New parameters size is not equal to parameters size.");

    Index index = 0;

    for (const unique_ptr<Layer>& layer : layers)
    {
        const vector<pair<type *, Index> > layer_parameter_pairs = layer->get_parameter_pairs();

        for (const pair<type*, Index>& parameter_pair : layer_parameter_pairs)
        {
            memcpy(parameter_pair.first, new_parameters.data() + index, parameter_pair.second * sizeof(type));
            index += parameter_pair.second;
        }
    }
}


void NeuralNetwork::set_display(const bool& new_display)
{
    display = new_display;
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

    #pragma omp parallel for
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


void NeuralNetwork::forward_propagate(const vector<pair<type*, dimensions>>& input_pair,
                                      ForwardPropagation& forward_propagation,
                                      const bool& is_training) const
{   
    const Index layers_number = get_layers_number();

    Index first_layer_index = 0;
    Index last_layer_index = layers_number-1;

    if(is_training)
    {
        first_layer_index = get_first_trainable_layer_index();
        last_layer_index = get_last_trainable_layer_index();
    }

    const vector<vector<pair<type*, dimensions>>> layer_input_pairs
        = forward_propagation.get_layer_input_pairs(input_pair, is_training);

    for (Index i = first_layer_index; i <= last_layer_index; i++)
        layers[i]->forward_propagate(layer_input_pairs[i],
                                     forward_propagation.layers[i],
                                     is_training);
}


void NeuralNetwork::forward_propagate(const vector<pair<type*, dimensions>>& input_pair,
                                      const Tensor<type, 1>& new_parameters,
                                      ForwardPropagation& forward_propagation)
{
    Tensor<type, 1> original_parameters;
    get_parameters(original_parameters);

    set_parameters(new_parameters);

    forward_propagate(input_pair, forward_propagation, true);

    set_parameters(original_parameters);
}


string NeuralNetwork::get_expression() const
{
    const Index layers_number = get_layers_number();

    const vector<string> layer_labels = get_layer_labels();

    vector<string> new_input_names = get_input_names();
    vector<string> new_output_names = get_output_names();

    const Index inputs_number = input_names.size();
    const Index outputs_number = output_names.size();

    for (int i = 0; i < inputs_number; i++)
        input_names[i].empty()
            ? new_input_names[i] = "input_" + to_string(i)
            : new_input_names[i] = input_names[i];

    ostringstream buffer;

    for (Index i = 0; i < layers_number; i++)
    {
        if (i == layers_number - 1)
        {
            for (int j = 0; j < outputs_number; j++)
                new_output_names[j] = !output_names[j].empty()
                      ? output_names[j]
                      : "output_" + to_string(i);

            buffer << layers[i]->get_expression(new_input_names, new_output_names) << endl;
        }
        else
        {
            const Index layer_neurons_number = layers[i]->get_outputs_number();

            new_output_names.resize(layer_neurons_number);
            
            for (Index j = 0; j < layer_neurons_number; j++)
                new_output_names[j] = (layer_labels[i] == "scaling_layer")
                      ? "scaled_" + input_names[j]
                      : layer_labels[i] + "_output_" + to_string(j);

            buffer << layers[i]->get_expression(new_input_names, new_output_names) << endl;
            new_input_names = new_output_names;
        }
    }

    string expression = buffer.str();

    //replace(expression, "+-", "-");
    return expression;
}


Tensor<type, 2> NeuralNetwork::calculate_scaled_outputs(type* scaled_inputs_data, Tensor<Index, 1>& inputs_dimensions)
{
    const Index inputs_dimensions_number = inputs_dimensions.size();

    if(inputs_dimensions_number == 2)
    {
        Tensor<type, 2> scaled_outputs;
        Tensor<type, 2> last_layer_outputs;

        Tensor<Index, 1> outputs_dimensions;
        Tensor<Index, 1> last_layer_outputs_dimensions;

        const Index layers_number = get_layers_number();

        if(layers_number == 0)
        {
            scaled_outputs = TensorMap<Tensor<type,2>>(scaled_inputs_data, inputs_dimensions[0], inputs_dimensions[1]);
            return scaled_outputs;
        }

        scaled_outputs.resize(inputs_dimensions[0], layers[0]->get_outputs_number());

        outputs_dimensions = get_dimensions(scaled_outputs);

        ForwardPropagation forward_propagation(inputs_dimensions[0], this);

        bool is_training = false;

        if(layers[0]->get_name() == "Scaling2d")
        {
            pair<type*, dimensions> scaled_inputs_tensor(scaled_inputs_data, {inputs_dimensions[0], inputs_dimensions[1]});

            const Tensor<Index, 0> size = inputs_dimensions.prod();

            memcpy(scaled_inputs_tensor.first, scaled_inputs_data, static_cast<size_t>(size(0)*sizeof(type)) );

            layers[0]->forward_propagate({scaled_inputs_tensor}, forward_propagation.layers[0], is_training);

            const pair<type*, dimensions> outputs_pair = forward_propagation.layers[0]->get_output_pair();
            scaled_outputs = tensor_map<2>(outputs_pair);
        }
        else
        {
            scaled_outputs = TensorMap<Tensor<type,2>>(scaled_inputs_data, inputs_dimensions[0], inputs_dimensions[1]);
        }

        last_layer_outputs = scaled_outputs;

        last_layer_outputs_dimensions = get_dimensions(last_layer_outputs);

        for(Index i = 1; i < layers_number; i++)
        {
            if(layers[i]->get_name() != "Unscaling" && layers[i]->get_name() != "Scaling2d")
            {
                scaled_outputs.resize(inputs_dimensions[0], layers[0]->get_outputs_number());

                outputs_dimensions = get_dimensions(scaled_outputs);

                pair<type*, dimensions> inputs_tensor(last_layer_outputs.data(), {last_layer_outputs_dimensions[0], last_layer_outputs_dimensions[1]});

                const Tensor<Index, 0> sizeT = last_layer_outputs_dimensions.prod();

                memcpy(inputs_tensor.first, last_layer_outputs.data() , static_cast<size_t>(sizeT(0)*sizeof(type)) );

                layers[i]->forward_propagate({inputs_tensor}, forward_propagation.layers[i], is_training);

                scaled_outputs = tensor_map<2>(forward_propagation.layers[i]->get_output_pair());

                last_layer_outputs = scaled_outputs;
                last_layer_outputs_dimensions = get_dimensions(last_layer_outputs);
            }
        }

        return scaled_outputs;
    }
    else if(inputs_dimensions_number == 4)
    { 
        /// @todo
        return Tensor<type, 2>();
    }
    else
    {
        return Tensor<type, 2>();
    }

}


Tensor<type, 2> NeuralNetwork::calculate_directional_inputs(const Index& direction,
                                                            const Tensor<type, 1>& point,
                                                            const type& minimum,
                                                            const type& maximum,
                                                            const Index& points_number) const
{
    const Index inputs_number = get_inputs_number();

    Tensor<type, 2> directional_inputs(points_number, inputs_number);

    Tensor<type, 1> inputs(inputs_number);

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
    // Tensor<type, 3> image = read_bmp_image(image_path);

    // Scaling4d* scaling_layer_4d = static_cast<Scaling4d*>(get_first("Scaling4d"));

    // const Index height = scaling_layer_4d->get_input_dimensions()[0];
    // const Index width = scaling_layer_4d->get_input_dimensions()[1];
    // const Index channels = scaling_layer_4d->get_input_dimensions()[2];

    // const Index current_height = image.dimension(0);
    // const Index current_width = image.dimension(1);
    // const Index current_channels = image.dimension(2);

    // if (current_channels != channels)
    //     throw runtime_error("Error: Different channels number " + image_path.string() + "\n");

    // if(current_height != height || current_width != width)
    //     image = resize_image(image, height, width);

    // Tensor<type, 4> input_data(1, height, width, channels);

    // const Index pixels_number = height * width * channels;

    // #pragma omp parallel for
    // for (Index j = 0; j < pixels_number; j++)
    //     input_data(j) = image(j);

    // const Tensor<type, 2> outputs = calculate_outputs<4,2>(input_data);

    // Index predicted_index = -1;

    // if (outputs.size() > 1)
    // {
    //     type max_value = outputs(0);

    //     for (Index i = 1; i < outputs.dimension(1); ++i)
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


Tensor<string, 2> NeuralNetwork::get_dense2d_layers_information() const
{
    const Index layers_number = get_layers_number();

    Index dense2d_layers_number = 0;

    for(Index i = 0; i < layers_number; i++)
        if (layers[i]->get_name() == "Dense2d" && layers[i]->get_label().find("classification") == std::string::npos)
            dense2d_layers_number++;

    Tensor<string, 2> information(dense2d_layers_number, 4);

    Index dense2d_layer_index = 0;

    for(Index i = 0; i < layers_number; i++)
    {
        const string& name = layers[i]->get_name();
        const string label = layers[i]->get_label();

        if (name != "Dense2d" || label.find("classification") != std::string::npos)
            continue;

        information(dense2d_layer_index, 0) = label;
        information(dense2d_layer_index, 1) = to_string(layers[i]->get_input_dimensions()[0]);
        information(dense2d_layer_index, 2) = to_string(layers[i]->get_output_dimensions()[0]);

        const Dense2d* dense2d_layer = static_cast<Dense2d*>(layers[i].get());

        information(dense2d_layer_index, 3) = dense2d_layer->get_activation_function();

        dense2d_layer_index++;
    }

    return information;
}


Tensor<string, 2> NeuralNetwork::get_probabilistic_layer_information() const
{
    const Index layers_number = get_layers_number();

    Index probabilistic_layers_number = 0;

    for(Index i = 0; i < layers_number; i++)
        if (layers[i]->get_label().find("classification") != std::string::npos)
            probabilistic_layers_number++;

    Tensor<string, 2> information(probabilistic_layers_number,4);

    Index probabilistic_layer_index = 0;

    for(Index i = 0; i < layers_number; i++)
    {
        const string& name = layers[i]->get_name();
        const string label = layers[i]->get_label();

        if (name != "Dense2d" || label.find("dense2d") != std::string::npos)
            continue;

        information(probabilistic_layer_index, 0) = label;
        information(probabilistic_layer_index, 1) = to_string(layers[i]->get_input_dimensions()[0]);
        information(probabilistic_layer_index, 2) = to_string(layers[i]->get_output_dimensions()[0]);

        const Dense2d* dense_2d = static_cast<Dense2d*>(layers[i].get());

        information(probabilistic_layer_index, 3) = dense_2d->get_activation_function();

        probabilistic_layer_index++;
    }

    return information;
}


void NeuralNetwork::to_XML(XMLPrinter& printer) const
{
    const Index inputs_number = get_inputs_number();
    const Index layers_number = get_layers_number();
    const Index outputs_number = get_outputs_number();

    printer.OpenElement("NeuralNetwork");

    // Inputs

    printer.OpenElement("Inputs");

    add_xml_element(printer, "InputsNumber", to_string(inputs_number));

    for (Index i = 0; i < inputs_number; i++)
        add_xml_element_attribute(printer, "Input", input_names[i], "Index", to_string(i + 1));

    printer.CloseElement();

    // Layers

    printer.OpenElement("Layers");

    add_xml_element(printer, "LayersNumber", to_string(layers_number));

    for (Index i = 0; i < layers_number; i++)
        layers[i]->to_XML(printer);

    // Layer input indices

    printer.OpenElement("LayerInputIndices");

    for (Index i = 0; i < Index(layer_input_indices.size()); i++) 
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
    //set();

    const XMLElement* neural_network_element = document.FirstChildElement("NeuralNetwork");

    if(!neural_network_element)
        throw runtime_error("Neural network element is nullptr.\n");

    inputs_from_XML(neural_network_element->FirstChildElement("Inputs"));
    layers_from_XML(neural_network_element->FirstChildElement("Layers"));
    outputs_from_XML(neural_network_element->FirstChildElement("Outputs"));
    set_display(read_xml_bool(neural_network_element, "Display"));
}


void NeuralNetwork::inputs_from_XML(const XMLElement* inputs_element)
{
    if(!inputs_element)
        throw runtime_error("Inputs element is nullptr.\n");

    const Index new_inputs_number = read_xml_index(inputs_element, "InputsNumber");
    input_names.resize(new_inputs_number);

    // Inputs names

    const XMLElement* start_element = inputs_element->FirstChildElement("InputsNumber");

    for(Index i = 0; i < new_inputs_number; i++)
    {
        const XMLElement* input_element = start_element->NextSiblingElement("Input");

        if(input_element->Attribute("Index") != to_string(i+1))
            throw runtime_error("Input index number (" + to_string(i+1) + ") does not match (" + input_element->Attribute("Item") + ").\n");

        if(!input_element->GetText())
            throw runtime_error("Input text is nullptr.");

        input_names[i] = input_element->GetText();

        start_element = input_element;
    }
}


void NeuralNetwork::layers_from_XML(const XMLElement* layers_element)
{
    if (!layers_element)
        throw runtime_error("Layers element is nullptr.\n");

    const Index layers_number = read_xml_index(layers_element, "LayersNumber");

    layers.clear();

    const XMLElement* start_element = layers_element->FirstChildElement("LayersNumber");

    for (Index i = 0; i < layers_number; i++)
    {
        const XMLElement* layer_element = start_element->NextSiblingElement();

        if (!layer_element)
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
    if (!layer_input_indices_element)
        throw runtime_error("LayerInputIndices element is nullptr.\n");

    layer_input_indices.resize(layers.size());
    layer_input_indices.clear();

    for (const XMLElement* layer_inputs_indices_element = layer_input_indices_element->FirstChildElement("LayerInputsIndices");
         layer_inputs_indices_element;
         layer_inputs_indices_element = layer_inputs_indices_element->NextSiblingElement("LayerInputsIndices"))
    {
        int layer_index;
        if (layer_inputs_indices_element->QueryIntAttribute("LayerIndex", &layer_index) != tinyxml2::XML_SUCCESS)
            throw runtime_error("Error: LayerIndex attribute missing or invalid.\n");

        const char* text = layer_inputs_indices_element->GetText();
        if (!text)
            throw runtime_error("Text is nullptr for LayerInputsIndices element.");

        const vector<Index> input_index = string_to_dimensions(string(text), " ");
        if (layer_index >= layer_input_indices.size())
            layer_input_indices.push_back(input_index);
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

    cout << "Inputs number: " << get_inputs_number() << endl;

    print_vector(get_input_names());

    const Index layers_number = get_layers_number();

    cout << "Layers number: " << layers_number << endl;

    for(Index i = 0; i < layers_number; i++)
    {
        cout << endl
             << "Layer " << i << ": " << endl;
        layers[i]->print();
    }

    cout << "Outputs number: " << get_outputs_number() << endl;

    cout << "Outputs:" << endl;
    print_vector(get_output_names());

    cout << "Parameters number: " << get_parameters_number() << endl;
}


void NeuralNetwork::save(const filesystem::path& file_name) const
{
    ofstream file(file_name);

    if (!file.is_open())
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

    // Tensor<type, 1> parameters;
    // get_parameters(parameters);

    // file << parameters << endl;

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

    const Index parameters_number = get_parameters_number();

    Tensor<type, 1> new_parameters(parameters_number);

    file.read(reinterpret_cast<char*>(new_parameters.data()), parameters_number * sizeof(type));

    if (!file)
        throw runtime_error("Error reading binary file: " + file_name.string());

    set_parameters(new_parameters);
}


void NeuralNetwork::save_outputs(Tensor<type, 2>& inputs, const filesystem::path& file_name)
{
    const Tensor<type, 2> outputs = calculate_outputs<2,2>(inputs);

    ofstream file(file_name);

    if(!file.is_open())
        throw runtime_error("Cannot open " + file_name.string() + " file.\n");

    const vector<string> output_names = get_output_names();

    const Index outputs_number = get_outputs_number();
    const Index batch_size = inputs.dimension(0);

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


NeuralNetworkBackPropagation::NeuralNetworkBackPropagation(const Index& new_batch_size, 
                                                           NeuralNetwork* new_neural_network)
{
    set(new_batch_size, new_neural_network);
}


void NeuralNetworkBackPropagation::set(const Index& new_batch_size, NeuralNetwork* new_neural_network)
{
    batch_size = new_batch_size;

    neural_network = new_neural_network;

    if(!neural_network) return;

    const vector<unique_ptr<Layer>>& neural_network_layers = neural_network->get_layers();
    const Index first_traineable_layer_number = neural_network->get_first_trainable_layer_index();
    const Index last_traineable_layer_number = neural_network->get_last_trainable_layer_index();

    const Index layers_number = neural_network_layers.size();

    layers.resize(layers_number);

    for (Index i = first_traineable_layer_number; i <= last_traineable_layer_number; i++)
    {
        layers[i] = Registry<LayerBackPropagation>::instance().create(neural_network_layers[i]->get_name());
        layers[i]->set(batch_size, neural_network_layers[i].get());
    }
}


const vector<unique_ptr<LayerBackPropagation>>& NeuralNetworkBackPropagation::get_layers() const
{
    return layers;
}


NeuralNetwork* NeuralNetworkBackPropagation::get_neural_network() const
{
    return neural_network;
}


void NeuralNetworkBackPropagation::print() const
{
    cout << "Neural network back-propagation" << endl;

    const Index layers_number = layers.size();

    for (Index i = 0; i < layers_number; i++)
    {
        cout << "Layer " << i << ": "
             << neural_network->get_layer(i)->get_name() << endl;

        if (!layers[i]) continue;

        layers[i]->print();
    }
}


ForwardPropagation::ForwardPropagation(const Index& new_batch_size, NeuralNetwork* new_neural_network)
{
    set(new_batch_size, new_neural_network);
}


void ForwardPropagation::set(const Index& new_samples_number, NeuralNetwork* new_neural_network)
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

}


pair<type*, dimensions> ForwardPropagation::get_last_trainable_layer_outputs_pair() const
{
    const Index last_trainable_layer_index = neural_network->get_last_trainable_layer_index();

    const unique_ptr<LayerForwardPropagation>& layer_forward_propagation = layers[last_trainable_layer_index];

    return layer_forward_propagation->get_output_pair();
}


vector<vector<pair<type*, dimensions>>> ForwardPropagation::get_layer_input_pairs(const vector<pair<type*, dimensions>>& batch_input_pairs,
                                                                                  const bool& is_training) const
{
    const Index layers_number = neural_network->get_layers_number();

    if (layers_number == 0)
        return {};

    const vector<vector<Index>>& layer_input_indices = neural_network->get_layer_input_indices();

    vector<vector<pair<type*, dimensions>>> layer_input_pairs(layers_number);

    layer_input_pairs[0] = batch_input_pairs;

    const Index first_trainable_layer_index = neural_network->get_first_trainable_layer_index();

    for (Index layer_index = first_trainable_layer_index; layer_index < layers_number; layer_index++)
    {
        if ((layer_index == first_trainable_layer_index && is_training) || layer_index == 0)
        {
            layer_input_pairs[layer_index] = batch_input_pairs;
            continue;
        }

        const vector<Index>& input_layer_indices = layer_input_indices[layer_index];
        layer_input_pairs[layer_index].resize(input_layer_indices.size());

        for (Index input_index = 0; input_index < static_cast<Index>(input_layer_indices.size()); input_index++)
        {
            const Index input_layer_index = input_layer_indices[input_index];
            layer_input_pairs[layer_index][input_index] = layers[input_layer_index]->get_output_pair();
        }
    }

    return layer_input_pairs;
}


void ForwardPropagation::print() const
{
    cout << "Neural network forward propagation" << endl;

    const Index layers_number = layers.size();

    cout << "Layers number: " << layers_number << endl;

    for (Index i = 0; i < layers_number; i++)
    {
        cout << "Layer " << i + 1 << ": " << neural_network->get_layer(i)->get_label() << endl;

        layers[i]->print();
    }
}


void NeuralNetworkBackPropagationLM::set(const Index& new_batch_size, 
                                         NeuralNetwork* new_neural_network)
{
    batch_size = new_batch_size;

    neural_network = new_neural_network;

    const Index layers_number = neural_network->get_layers_number();

    const vector<unique_ptr<Layer>>& neural_network_layers = neural_network->get_layers();

    layers.resize(layers_number);

    for(Index i = 0; i < layers_number; i++)
        if(neural_network_layers[i]->get_name() == "Dense2d")
            layers[i] = make_unique<Dense2dLayerBackPropagationLM>(batch_size, neural_network_layers[i].get());
}


#ifdef OPENNN_CUDA

void NeuralNetwork::allocate_parameters_device()
{
    const vector<unique_ptr<Layer>>& layers = get_layers();

    const Index layers_number = layers.size();

    if (layers_number == 0) return;

    const Index first_trainable_layer_index = get_first_trainable_layer_index();
    const Index last_trainable_layer_index = get_last_trainable_layer_index();

    for (Index i = first_trainable_layer_index; i <= last_trainable_layer_index; i++)
        layers[i]->allocate_parameters_device();
}


void NeuralNetwork::free_parameters_device()
{
    const vector<unique_ptr<Layer>>& layers = get_layers();

    const Index layers_number = layers.size();

    if (layers_number == 0) return;

    const Index first_trainable_layer_index = get_first_trainable_layer_index();
    const Index last_trainable_layer_index = get_last_trainable_layer_index();

    for (Index i = first_trainable_layer_index; i <= last_trainable_layer_index; i++)
        layers[i]->free_parameters_device();
}


void NeuralNetwork::copy_parameters_device()
{
    const vector<unique_ptr<Layer>>& layers = get_layers();

    const Index layers_number = layers.size();

    if (layers_number == 0) return;

    const Index first_trainable_layer_index = get_first_trainable_layer_index();
    const Index last_trainable_layer_index = get_last_trainable_layer_index();

    for (Index i = first_trainable_layer_index; i <= last_trainable_layer_index; i++)
        layers[i]->copy_parameters_device();
}


void NeuralNetwork::copy_parameters_host()
{
    const vector<unique_ptr<Layer>>& layers = get_layers();

    const Index layers_number = layers.size();

    if (layers_number == 0) return;

    const Index first_trainable_layer_index = get_first_trainable_layer_index();
    const Index last_trainable_layer_index = get_last_trainable_layer_index();

    for (Index i = first_trainable_layer_index; i <= last_trainable_layer_index; i++)
        layers[i]->copy_parameters_host();
}


void NeuralNetwork::forward_propagate_cuda(const vector<float*>& input_device,
                                           ForwardPropagationCuda& forward_propagation_cuda,
                                           const bool& is_training) const
{
    const Index layers_number = get_layers_number();

    const Index first_trainable_layer_index = get_first_trainable_layer_index();
    const Index last_trainable_layer_index = get_last_trainable_layer_index();

    const Index first_layer_index = is_training ? first_trainable_layer_index : 0;
    const Index last_layer_index = is_training ? last_trainable_layer_index : layers_number - 1;

    const vector<vector<float*>> layer_input_device = forward_propagation_cuda.get_layer_inputs_device(input_device, is_training);

    for (Index i = first_layer_index; i <= last_layer_index; i++)
        layers[i]->forward_propagate_cuda(layer_input_device[i],
                                          forward_propagation_cuda.layers[i],
                                          is_training);
}


float* NeuralNetwork::calculate_outputs_cuda(float* input_device, const Index& batch_size)
{
    if (layers.empty())
        return nullptr;

    ForwardPropagationCuda forward_propagation_cuda(batch_size, this);

    forward_propagate_cuda({ input_device }, forward_propagation_cuda, false);

    return forward_propagation_cuda.get_last_trainable_layer_outputs_device();
}


void NeuralNetwork::create_cuda() const
{
    const vector<unique_ptr<Layer>>& neural_network_layers = get_layers();

    for (const unique_ptr<Layer>& layer : neural_network_layers)
        layer->create_cuda();
}


void NeuralNetwork::destroy_cuda() const
{
    const vector<unique_ptr<Layer>>& neural_network_layers = get_layers();

    for (const unique_ptr<Layer>& layer : neural_network_layers)
        layer->destroy_cuda();
}


ForwardPropagationCuda::ForwardPropagationCuda(const Index& new_batch_size, NeuralNetwork* new_neural_network)
{
    set(new_batch_size, new_neural_network);
}


void ForwardPropagationCuda::set(const Index& new_samples_number, NeuralNetwork* new_neural_network)
{
    samples_number = new_samples_number;

    neural_network = new_neural_network;

    if (!neural_network) throw runtime_error("There is no neural network.");

    const vector<unique_ptr<Layer>>& neural_network_layers = neural_network->get_layers();

    const Index layers_number = neural_network_layers.size();

    layers.resize(layers_number);

    for (Index i = 0; i < layers_number; i++)
    {
        layers[i] = Registry<LayerForwardPropagationCuda>::instance().create(neural_network_layers[i]->get_name());
        layers[i]->set(samples_number, neural_network_layers[i].get());
    }
}


float* ForwardPropagationCuda::get_last_trainable_layer_outputs_device() const
{
    const Index last_trainable_layer_index = neural_network->get_last_trainable_layer_index();

    const unique_ptr<LayerForwardPropagationCuda>& layer_forward_propagation = layers[last_trainable_layer_index];

    return layer_forward_propagation->get_output_device();
}


vector<vector<float*>> ForwardPropagationCuda::get_layer_inputs_device(const vector<float*>& batch_input_device, const bool& is_training) const
{
    const Index layers_number = neural_network->get_layers_number();

    if (layers_number == 0)
        return vector<vector<float*>>();

    const vector<vector<Index>>& layer_input_indices = neural_network->get_layer_input_indices();

    const Index first_trainable_layer_index = neural_network->get_first_trainable_layer_index();
    const Index last_trainable_layer_index = neural_network->get_last_trainable_layer_index();

    const Index first_layer_index = is_training ? first_trainable_layer_index : 0;
    const Index last_layer_index = is_training ? last_trainable_layer_index : layers_number - 1;

    vector<vector<float*>> layer_input_device(layers_number);

    layer_input_device[0] = batch_input_device;

    for (Index i = first_layer_index; i <= last_layer_index; i++)
    {
        const vector<Index>& this_layer_input_indices = layer_input_indices[i];

        layer_input_device[i].resize(1);

        if (false/*neural_network->get_model_type_string() == "TextClassification"*/)
        {
            if (i == first_trainable_layer_index)
            {
                vector<float*> batch_input_pairs1;
                batch_input_pairs1.push_back(batch_input_device[0]);
                layer_input_device[i] = batch_input_pairs1;
                continue;
            }

            if (i == first_trainable_layer_index + 1)
            {
                vector<float*> batch_input_pairs2;
                batch_input_pairs2.push_back(batch_input_device[1]);
                layer_input_device[i] = batch_input_pairs2;
                continue;
            }
        }
        else
        {
            if ((i == first_trainable_layer_index && is_training) || i == 0)
            {
                layer_input_device[i] = batch_input_device;
                continue;
            }
        };

        const Index this_layer_inputs_number = this_layer_input_indices.size();

        layer_input_device[i].resize(this_layer_inputs_number);

        for (Index j = 0; j < this_layer_inputs_number; j++)
        {
            const Index this_layer_input_index = this_layer_input_indices[j];

            layer_input_device[i][j] = layers[this_layer_input_index]->get_output_device();
        }
    }

    return layer_input_device;
}


void ForwardPropagationCuda::print()
{
    const Index layers_number = layers.size();

    cout << "Layers number: " << layers_number << endl;

    for (Index i = 0; i < layers_number; i++)
    {
        cout << "Layer " << i + 1 << endl;

        layers[i]->print();
    }
}


void ForwardPropagationCuda::free()
{
    const Index layers_number = layers.size();

    for (Index i = 0; i < layers_number; i++)
        if (!layers[i])
            layers[i]->free();
}


NeuralNetworkBackPropagationCuda::NeuralNetworkBackPropagationCuda(const Index& new_batch_size, NeuralNetwork* new_neural_network)
{
    set(new_batch_size, new_neural_network);
}


void NeuralNetworkBackPropagationCuda::set(const Index& new_batch_size, NeuralNetwork* new_neural_network)
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
}


const vector<unique_ptr<LayerBackPropagationCuda>>& NeuralNetworkBackPropagationCuda::get_layers() const
{
    return layers;
}


void NeuralNetworkBackPropagationCuda::print()
{
    const Index layers_number = layers.size();

    cout << "Layers number: " << layers_number << endl;

    for (Index i = 0; i < layers_number; i++)
    {
        cout << "Layer " << i + 1 << endl;

        layers[i]->print();
    }
}


void NeuralNetworkBackPropagationCuda::free()
{
    const Index layers_number = layers.size();

    for (Index i = 0; i < layers_number; i++)
        if (!layers[i])
            layers[i]->free();
}

#endif

} // Namespace

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2025 Artificial Intelligence Techniques, SL.
//
// This library is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 2.1 of the License, or any later version.
//
// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.

// You should have received a copy of the GNU Lesser General Public
// License along with this library; if not, write to the Free Software
// Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
