//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   N E U R A L   N E T W O R K   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <numeric>

#include "tensors.h"
#include "neural_network.h"
#include "neural_network_forward_propagation.h"
#include "neural_network_back_propagation.h"
#include "neural_network_back_propagation_lm.h"
#include "strings_utilities.h"
#include "config.h"
#include "layer.h"
#include "perceptron_layer.h"
#include "perceptron_layer_3d.h"
#include "scaling_layer_2d.h"
#include "scaling_layer_4d.h"
#include "addition_layer_3d.h"
#include "normalization_layer_3d.h"
#include "unscaling_layer.h"
#include "bounding_layer.h"
#include "probabilistic_layer.h"
#include "probabilistic_layer_3d.h"
#include "convolutional_layer.h"
#include "flatten_layer.h"
#include "embedding_layer.h"
#include "multihead_attention_layer.h"

namespace opennn
{

NeuralNetwork::NeuralNetwork() : layers(0)
{
    set();
}


NeuralNetwork::NeuralNetwork(const NeuralNetwork::ModelType& model_type, 
                             const dimensions& input_dimensions,
                             const dimensions& complexity_dimensions,
                             const dimensions& output_dimensions)
{
    set(model_type, input_dimensions, complexity_dimensions, output_dimensions);
}


NeuralNetwork::NeuralNetwork(const string& file_name)
{
    load(file_name);
}


NeuralNetwork::~NeuralNetwork()
{
//    delete_layers();
}


//void NeuralNetwork::delete_layers()
//{
//    layers.resize(0);
//}


void NeuralNetwork::add_layer(unique_ptr<Layer> layer, const string& name, const vector<Index>& input_indices)
{
    const Layer::Type layer_type = layer->get_type();

    if(!validate_layer_type(layer_type)) return;

    const Index old_layers_number = get_layers_number();

    layers.push_back(std::move(layer));

    layer_input_indices.push_back(input_indices.empty() ? std::vector<Index>(1, old_layers_number - 1) : input_indices);

    layers[old_layers_number]->set_name(name);
}


bool NeuralNetwork::validate_layer_type(const Layer::Type layer_type) const
{
    if(has_bounding_layer())
        throw runtime_error("No layers can be added after a bounding layer.\n");

    return true;
}


bool NeuralNetwork::has_scaling_layer_2d() const
{
    const Index layers_number = get_layers_number();

    for(Index i = 0; i < layers_number; i++)
        if(layers[i]->get_type() == Layer::Type::Scaling2D) 
            return true;

    return false;
}


bool NeuralNetwork::has_scaling_layer_4d() const
{
    const Index layers_number = get_layers_number();

    for(Index i = 0; i < layers_number; i++)
        if(layers[i]->get_type() == Layer::Type::Scaling4D) 
            return true;

    return false;
}


bool NeuralNetwork::has_long_short_term_memory_layer() const
{
    const Index layers_number = get_layers_number();

    for(Index i = 0; i < layers_number; i++)
        if(layers[i]->get_type() == Layer::Type::LongShortTermMemory) 
            return true;

    return false;
}


bool NeuralNetwork::has_convolutional_layer() const
{
    const Index layers_number = get_layers_number();

    for(Index i = 0; i < layers_number; i++)
        if(layers[i]->get_type() == Layer::Type::Convolutional) 
            return true;

    return false;
}


bool NeuralNetwork::has_flatten_layer() const
{
    const Index layers_number = get_layers_number();

    for(Index i = 0; i < layers_number; i++)
        if(layers[i]->get_type() == Layer::Type::Flatten) 
            return true;

    return false;
}


bool NeuralNetwork::has_recurrent_layer() const
{
    const Index layers_number = get_layers_number();

    for(Index i = 0; i < layers_number; i++)
        if(layers[i]->get_type() == Layer::Type::Recurrent) 
            return true;

    return false;
}


bool NeuralNetwork::has_unscaling_layer() const
{
    const Index layers_number = get_layers_number();

    for(Index i = 0; i < layers_number; i++)
        if(layers[i]->get_type() == Layer::Type::Unscaling) 
            return true;

    return false;
}


bool NeuralNetwork::has_bounding_layer() const
{
    const Index layers_number = get_layers_number();

    for(Index i = 0; i < layers_number; i++)
        if(layers[i]->get_type() == Layer::Type::Bounding) 
            return true;

    return false;
}


bool NeuralNetwork::has_probabilistic_layer() const
{
    const Index layers_number = get_layers_number();

    for(Index i = 0; i < layers_number; i++)
        if(layers[i]->get_type() == Layer::Type::Probabilistic) 
            return true;

    return false;
}


bool NeuralNetwork::is_empty() const
{
    return layers.empty();
}


const Tensor<string, 1>& NeuralNetwork::get_input_names() const
{
    return inputs_name;
}


string NeuralNetwork::get_input_name(const Index& index) const
{
    return inputs_name[index];
}


Index NeuralNetwork::get_input_index(const string& name) const
{
    for(Index i = 0; i < inputs_name.size(); i++)
        if(inputs_name(i) == name) 
            return i;

    return 0;
}


NeuralNetwork::ModelType NeuralNetwork::get_model_type() const
{
    return model_type;
}


string NeuralNetwork::get_model_type_string() const
{
    switch (model_type)
    {
        case ModelType::AutoAssociation:
            return "AutoAssociation";
        case ModelType::Approximation:
            return "Approximation";
        case ModelType::Classification:
            return "Classification";
        case ModelType::Forecasting:
            return "Forecasting";
        case ModelType::TextClassification:
            return "TextClassification";
        case ModelType::ImageClassification:
            return "ImageClassification";
        default:
            throw runtime_error("Unkown model type");
    }
}


const Tensor<string, 1>& NeuralNetwork::get_output_names() const
{
    return output_names;
}


string NeuralNetwork::get_output_name(const Index& index) const
{
    return output_names[index];
}


Index NeuralNetwork::get_output_index(const string& name) const
{
    for(Index i = 0; i < output_names.size(); i++)
        if(output_names(i) == name) return i;

    return 0;
}


const vector<unique_ptr<Layer>>& NeuralNetwork::get_layers() const
{
    return layers;
}


const unique_ptr<Layer>& NeuralNetwork::get_layer(const Index& layer_index) const
{
    return layers[layer_index];
}


const unique_ptr<Layer>& NeuralNetwork::get_layer(const string& name) const
{
    const Tensor<string, 1> layer_names = get_layer_names();

    for(Index i = 0; i < layer_names.size(); i++)
        if(layer_names(i) == name)
            return layers[i];

    throw runtime_error("Layer not found in neural network");
}


vector<Layer*> NeuralNetwork::get_trainable_layers() const
{
    const Index trainable_layers_number = get_trainable_layers_number();

    vector<Layer*> trainable_layers(trainable_layers_number);

    const Index layers_number = get_layers_number();

    Index index = 0;

    for (Index i = 0; i < layers_number; i++)
    {
        const Layer::Type layer_type = layers[i]->get_type();

        if (layer_type != Layer::Type::Scaling2D &&
            layer_type != Layer::Type::Scaling4D &&
            layer_type != Layer::Type::Unscaling &&
            layer_type != Layer::Type::Bounding)
        {
            trainable_layers[index++] = layers[i].get();
        }
    }
    return trainable_layers;
}


Index NeuralNetwork::get_layer_index(const string& name) const
{
    const Index layers_number = get_layers_number();

    if(name == "dataset" || name == "input")
        return -1;
    else if(name == "context")
        return -2;

    for(Index i = 0; i < layers_number; i++)
        if(layers[i]->get_name() == name)
            return i;

    return 0;
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
    {
        for (Index k = 0; k < layer_input_indices[i].size(); k++)
        {
            const Index input_index = layer_input_indices[i][k];
            layer_output_indices[input_index].push_back(i);
        }
    }
/*
    Index layer_count = 0;

    for(Index i = 0; i < layers_number; i++)
    {
        for(Index j = 0; j < layers_number; j++)
            for(Index k = 0; k < layer_input_indices[j].size(); k++)
                if(layer_input_indices[j][k] == i)
                    layer_count++;

        layer_output_indices[i].resize(layer_count);

        layer_count = 0;

        for(Index j = 0; j < layers_number; j++)
        {
            for(Index k = 0; k < layer_input_indices[j].size(); k++)
            {
                if(layer_input_indices[j][k] == i)
                {
                    layer_output_indices[i][layer_count++] = j;
                }
            }
        }

        layer_count = 0;
    }
*/
    return layer_output_indices;
}


Index NeuralNetwork::find_input_index(const vector<Index>& layer_inputs_indices, const Index& layer_index) const
{
    // @todo not sure what it does. Rename variables

    for (Index i = 0; i < layer_inputs_indices.size(); i++)
        if (layer_inputs_indices[i] == layer_index)
            return i;

    return -1;
}


ScalingLayer2D* NeuralNetwork::get_scaling_layer_2d() const
{
    const Index layers_number = get_layers_number();

    for(Index i = 0; i < layers_number; i++)
        if(layers[i]->get_type() == Layer::Type::Scaling2D)
            return dynamic_cast<ScalingLayer2D*>(layers[i].get());

    throw runtime_error("No scaling layer 2d in neural network.\n");
}


ScalingLayer4D* NeuralNetwork::get_scaling_layer_4d() const
{
    const Index layers_number = get_layers_number();

    for(Index i = 0; i < layers_number; i++)
        if(layers[i]->get_type() == Layer::Type::Scaling4D)
            return dynamic_cast<ScalingLayer4D*>(layers[i].get());

    throw runtime_error("No scaling layer in neural network.\n");
}


UnscalingLayer* NeuralNetwork::get_unscaling_layer() const
{
    const Index layers_number = get_layers_number();

    for(Index i = 0; i < layers_number; i++)
        if(layers[i]->get_type() == Layer::Type::Unscaling)
            return dynamic_cast<UnscalingLayer*>(layers[i].get());

    throw runtime_error("No unscaling layer in neural network.\n");
}


BoundingLayer* NeuralNetwork::get_bounding_layer() const
{
    const Index layers_number = get_layers_number();

    for(Index i = 0; i < layers_number; i++)
        if(layers[i]->get_type() == Layer::Type::Bounding)
            return dynamic_cast<BoundingLayer*>(layers[i].get());

    throw runtime_error("No bounding layer in neural network.\n");
}


// FlattenLayer* NeuralNetwork::get_flatten_layer() const
// {
//     const Index layers_number = get_layers_number();

//     for(Index i = 0; i < layers_number; i++)
//     {
//         if(layers[i]->get_type() == Layer::Type::Flatten)
//         {
//             return dynamic_cast<FlattenLayer*>(layers[i]);
//         }
//     }

//     throw runtime_error("No flatten layer in neural network.\n");
// }


//ConvolutionalLayer* NeuralNetwork::get_convolutional_layer() const
//{
//    const Index layers_number = get_layers_number();
//
//    for(Index i = 0; i < layers_number; i++)
//        if(layers[i]->get_type() == Layer::Type::Convolutional)
//            return dynamic_cast<ConvolutionalLayer*>(layers[i]);
//
//    throw runtime_error("No convolutional layer in neural network.\n");
//}


// PoolingLayer* NeuralNetwork::get_pooling_layer() const
// {
//     const Index layers_number = get_layers_number();

//     for(Index i = 0; i < layers_number; i++)
//         if(layers[i]->get_type() == Layer::Type::PoolingLayer)
//             return dynamic_cast<PoolingLayer*>(layers[i]);

//     throw runtime_error("No pooling layer in neural network.\n");
// }


ProbabilisticLayer* NeuralNetwork::get_probabilistic_layer() const
{
    const Index layers_number = get_layers_number();

    for(Index i = 0; i < layers_number; i++)
        if(layers[i]->get_type() == Layer::Type::Probabilistic)
            return dynamic_cast<ProbabilisticLayer*>(layers[i].get());

    throw runtime_error("No probabilistic layer in neural network.\n");
}


LongShortTermMemoryLayer* NeuralNetwork::get_long_short_term_memory_layer() const
{
    const Index layers_number = get_layers_number();

    for(Index i = 0; i < layers_number; i++)
        if(layers[i]->get_type() == Layer::Type::LongShortTermMemory)
            return dynamic_cast<LongShortTermMemoryLayer*>(layers[i].get());

    throw runtime_error("No long-short-term memory layer in neural network.\n");
}


RecurrentLayer* NeuralNetwork::get_recurrent_layer() const
{
    const Index layers_number = get_layers_number();

    for(Index i = 0; i < layers_number; i++)
        if(layers[i]->get_type() == Layer::Type::Recurrent)
            return dynamic_cast<RecurrentLayer*>(layers[i].get());

    throw runtime_error("No recurrent layer in neural network.\n");
}


const bool& NeuralNetwork::get_display() const
{
    return display;
}


void NeuralNetwork::set()
{
    inputs_name.resize(0);

    output_names.resize(0);

    layers.resize(0);

    set_default();
}


void NeuralNetwork::set(const NeuralNetwork::ModelType& new_model_type,
                        const dimensions& input_dimensions, 
                        const dimensions& complexity_dimensions,
                        const dimensions& output_dimensions)
{
    set_default();

    layers.resize(0);

    model_type = new_model_type;

    const Index complexity_size = complexity_dimensions.size();

    const Index inputs_number = accumulate(input_dimensions.begin(), input_dimensions.end(), 1, multiplies<Index>());

    inputs_name.resize(inputs_number);

    for(Index i = 0; i < inputs_number; i++)
        inputs_name(i) = "input_" + to_string(i+1);

    if(model_type == ModelType::Approximation)
    {
        add_layer(make_unique<ScalingLayer2D>(input_dimensions));

        for(Index i = 0; i < complexity_size; i++)
        {
            const dimensions neurons_number = {complexity_dimensions[i]};

            add_layer(make_unique<PerceptronLayer>(get_output_dimensions(), neurons_number, PerceptronLayer::ActivationFunction::HyperbolicTangent),
                      "perceptron_layer_" + to_string(i+1));
        }

        add_layer(make_unique<PerceptronLayer>(get_output_dimensions(), output_dimensions, PerceptronLayer::ActivationFunction::Linear), "perceptron_layer_" + to_string(complexity_size+1));

        add_layer(make_unique<UnscalingLayer>(output_dimensions));

        add_layer(make_unique<BoundingLayer>(output_dimensions));
    }
    else if(model_type == ModelType::Classification || model_type == ModelType::TextClassification)
    {
        for (Index i = 0; i < complexity_size; i++)
        {
            const dimensions neurons_number = { complexity_dimensions[i] };
            add_layer(make_unique<PerceptronLayer>(get_output_dimensions(), neurons_number, PerceptronLayer::ActivationFunction::HyperbolicTangent),
                "perceptron_layer_" + to_string(i + 1));
        }

        add_layer(make_unique<ProbabilisticLayer>(get_output_dimensions(), output_dimensions), "probabilistic_layer");
    }
    else if(model_type == ModelType::Forecasting)
    {
        add_layer(make_unique<ScalingLayer2D>(input_dimensions));

        add_layer(make_unique<UnscalingLayer>(output_dimensions));

        add_layer(make_unique<BoundingLayer>(output_dimensions));
    }
    else if(model_type == ModelType::ImageClassification)
    {
        add_layer(make_unique<ScalingLayer4D>(input_dimensions));
        
        for (Index i = 0; i < complexity_size; i++)
        {
            const dimensions kernel_dimensions = {3, 3, get_output_dimensions()[2], complexity_dimensions[i]};

            add_layer(make_unique<ConvolutionalLayer>(get_output_dimensions(), kernel_dimensions),
                      "convolutional_layer_" + to_string(i+1));

            const dimensions pool_dimensions = {1, 1};

            add_layer(make_unique<PoolingLayer>(get_output_dimensions(), pool_dimensions),
                "pooling_layer_" + to_string(i + 1));
        }

        add_layer(make_unique<FlattenLayer>(get_output_dimensions()));

        //const dimensions neurons_number = { complexity_dimensions[complexity_dimensions.size()]*2 };
        //add_layer(make_unique<PerceptronLayer>(get_output_dimensions(), neurons_number, PerceptronLayer::ActivationFunction::RectifiedLinear), "perceptron_layer");

        add_layer(make_unique<ProbabilisticLayer>(get_output_dimensions(), output_dimensions), "probabilistic_layer");
    }
    else if(model_type == ModelType::AutoAssociation)
    {
        /*
        add_layer(make_unique<ScalingLayer2D>(inputs_number));

        const Index mapping_neurons_number = 10;
        const Index bottle_neck_neurons_number = complexity_dimensions[0];

        add_layer(make_unique<PerceptronLayer>(input_dimensions[0], mapping_neurons_number, PerceptronLayer::ActivationFunction::HyperbolicTangent),
                  "mapping_layer");

        add_layer(make_unique<PerceptronLayer>(mapping_neurons_number, bottle_neck_neurons_number, PerceptronLayer::ActivationFunction::Linear),
                  "bottleneck_layer");

        add_layer(make_unique<PerceptronLayer>(bottle_neck_neurons_number, mapping_neurons_number, PerceptronLayer::ActivationFunction::HyperbolicTangent),
                  "demapping_layer");

        add_layer(make_unique<PerceptronLayer>(mapping_neurons_number, output_dimensions[0], PerceptronLayer::ActivationFunction::Linear),
                  "output_layer");

        add_layer(make_unique<UnscalingLayer>(output_dimensions[0]));
*/
    }

    const Index outputs_number = accumulate(output_dimensions.begin(), output_dimensions.end(), 1, multiplies<Index>());

    output_names.resize(outputs_number);

    for(Index i = 0; i < outputs_number; i++)
        output_names(i) = "output_" + to_string(i+1);
}


void NeuralNetwork::set(const string& file_name)
{
    layers.resize(0);

    load(file_name);
}


void NeuralNetwork::set_model_type(const NeuralNetwork::ModelType& new_model_type)
{
    model_type = new_model_type;
}


void NeuralNetwork::set_model_type_string(const string& new_model_type)
{
    if(new_model_type == "Approximation")
        set_model_type(ModelType::Approximation);
    else if(new_model_type == "Classification")
        set_model_type(ModelType::Classification);
    else if(new_model_type == "Forecasting")
        set_model_type(ModelType::Forecasting);
    else if(new_model_type == "ImageClassification")
        set_model_type(ModelType::ImageClassification);
    else if(new_model_type == "TextClassification")
        set_model_type(ModelType::TextClassification);
    else if(new_model_type == "AutoAssociation")
        set_model_type(ModelType::AutoAssociation);
    else
        throw runtime_error("Neural Network class exception:\n"
                            "void set_model_type_string(const string&)\n"
                            "Unknown model type: " + new_model_type + "\n");
}


void NeuralNetwork::set_inputs_names(const Tensor<string, 1>& new_inputs_names)
{
    inputs_name = new_inputs_names;
}


void NeuralNetwork::set_output_namess(const Tensor<string, 1>& new_output_namess)
{
    output_names = new_output_namess;
}


void NeuralNetwork::set_inputs_number(const Index& new_inputs_number)
{
    inputs_name.resize(new_inputs_number);

    if(has_scaling_layer_2d())
    {
        ScalingLayer2D* scaling_layer_2d = get_scaling_layer_2d();

        scaling_layer_2d->set_inputs_number(new_inputs_number);
    }

    const Index trainable_layers_number = get_trainable_layers_number();
    vector<Layer*> trainable_layers = get_trainable_layers();

    if(trainable_layers_number > 0)
        trainable_layers[0]->set_inputs_number(new_inputs_number);
}


// void NeuralNetwork::set_inputs_number(const Tensor<bool, 1>& inputs)
// {
//     if(layers.dimension(0) == 0) return;

//     Index new_inputs_number = 0;

//     for(Index i = 0; i < inputs.dimension(0); i++)
//     {
//         if(inputs(i)) new_inputs_number++;
//     }

//     set_inputs_number(new_inputs_number);
// }


void NeuralNetwork::set_default()
{
    display = true;

    layer_input_indices = vector<vector<Index>>();

    const int n = omp_get_max_threads();

    thread_pool = new ThreadPool(n);
    thread_pool_device = new ThreadPoolDevice(thread_pool, n);
}


void NeuralNetwork::set_threads_number(const int& new_threads_number)
{
    const Index layers_number = get_layers_number();

    for(Index i = 0; i < layers_number; i++)
        layers[i]->set_threads_number(new_threads_number);
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


void NeuralNetwork::set_layer_inputs_indices(const string& name, const Tensor<string, 1>& new_layer_inputs_names)
{
    const Index layer_index = get_layer_index(name);

    const Index size = new_layer_inputs_names.size();

    vector<Index> new_layer_input_indices(size);

    for(Index i = 0; i < size; i++)
        new_layer_input_indices[i] = get_layer_index(new_layer_inputs_names(i));

    layer_input_indices[layer_index] = new_layer_input_indices;
}


void NeuralNetwork::set_layer_inputs_indices(const string& name, const initializer_list<string>& new_layer_inputs_names_list)
{
    Tensor<string, 1> new_layer_inputs_names(new_layer_inputs_names_list.size());
    new_layer_inputs_names.setValues(new_layer_inputs_names_list);

    set_layer_inputs_indices(name, new_layer_inputs_names);
}


void NeuralNetwork::set_layer_inputs_indices(const string& name, const string& new_layer_inputs_name)
{
    const Index layer_index = get_layer_index(name);

    vector<Index> new_layer_input_indices(1);

    new_layer_input_indices[0] = get_layer_index(new_layer_inputs_name);

    layer_input_indices[layer_index] = new_layer_input_indices;
}


PerceptronLayer* NeuralNetwork::get_first_perceptron_layer() const
{
    const Index layers_number = get_layers_number();

    for(Index i = 0; i < layers_number; i++)
        if(layers[i]->get_type() == Layer::Type::Perceptron)
            return static_cast<PerceptronLayer*>(layers[i].get());

    return nullptr;
}


Index NeuralNetwork::get_inputs_number() const
{
    if(!layers.empty())
        return layers[0]->get_inputs_number();

    return 0;
}


Index NeuralNetwork::get_outputs_number() const
{
    if(layers.size() == 0) return 0;

    const Layer* last_layer = layers[layers.size() - 1].get();

    const dimensions output_dimensions = last_layer->get_output_dimensions();

    const Index outputs_rank = output_dimensions.size();

    Index outputs_number = 1;

    for(Index i = 0; i < outputs_rank; i++)
        outputs_number *= output_dimensions[i];

    return outputs_number;
}


dimensions NeuralNetwork::get_output_dimensions() const
{
    if(layers.size() == 0) 
        return {};

    const Layer* last_layer = layers[layers.size() - 1].get();

    return last_layer->get_output_dimensions();
}


Tensor<Index, 1> NeuralNetwork::get_architecture() const
{
    const Index layers_number = get_layers_number();
    const Index inputs_number = get_inputs_number();

    if(layers_number == 0 || inputs_number == 0) 
        return Tensor<Index, 1>();

    Tensor<Index, 1> architecture(layers_number);

    for(Index i = 0; i < layers_number; i++)
        architecture(i) = layers[i]->get_neurons_number();

    return architecture;
}


Index NeuralNetwork::get_parameters_number() const
{
    Index parameters_number = 0;

    for(Index i = 0; i < layers.size(); i++)
        parameters_number += layers[i]->get_parameters_number();

    return parameters_number;
}


Tensor<type, 1> NeuralNetwork::get_parameters() const
{
    const Index parameters_number = get_parameters_number();

    Tensor<type, 1> parameters(parameters_number);

    const Index trainable_layers_number = get_trainable_layers_number();

    Index position = 0;

    for(Index i = 0; i < trainable_layers_number; i++)
    {
        const Tensor<type, 1> layer_parameters = layers[i]->get_parameters();

        // @todo use memcpy

        for(Index j = 0; j < layer_parameters.size(); j++)
            parameters(j + position) = layer_parameters(j);

        position += layer_parameters.size();
    }

    return parameters;
}


vector<Index> NeuralNetwork::get_layers_parameters_numbers() const
{
    const Index layers_number = get_layers_number();

    vector<Index> layers_parameters_number(layers_number);

    for(Index i = 0; i < layers_number; i++)
        layers_parameters_number[i] = layers[i]->get_parameters_number();

    return layers_parameters_number;
}


vector<Index> NeuralNetwork::get_trainable_layers_parameters_numbers() const
{
    const Index trainable_layers_number = get_trainable_layers_number();

    const vector<Layer*> trainable_layers = get_trainable_layers();

    vector<Index> trainable_layers_parameters_number(trainable_layers_number);

    for(Index i = 0; i < trainable_layers_number; i++)
        trainable_layers_parameters_number[i] = trainable_layers[i]->get_parameters_number();

    return trainable_layers_parameters_number;
}


void NeuralNetwork::set_parameters(const Tensor<type, 1>& new_parameters) const
{
    const Index trainable_layers_number = get_trainable_layers_number();

    const vector<Layer*> trainable_layers = get_trainable_layers();

    const vector<Index> trainable_layers_parameters_numbers = get_trainable_layers_parameters_numbers();

    Index index = 0;

    // @todo parallelize

    for(Index i = 0; i < trainable_layers_number; i++)
    {
        trainable_layers[i]->set_parameters(new_parameters, index);

        index += trainable_layers_parameters_numbers[i];
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


Index NeuralNetwork::get_trainable_layers_number() const
{
    const Index layers_number = get_layers_number();

    Index count = 0;

    for(Index i = 0; i < layers_number; i++)
    {
        if(layers[i]->get_type() != Layer::Type::Scaling2D
        && layers[i]->get_type() != Layer::Type::Scaling4D
        && layers[i]->get_type() != Layer::Type::Unscaling
        && layers[i]->get_type() != Layer::Type::Bounding)
            count++;
    }

    return count;
}


Index NeuralNetwork::get_first_trainable_layer_index() const
{
    const Index layers_number = get_layers_number();

    Layer::Type layer_type;

    for(Index i = 0; i < layers_number; i++)
    {
        layer_type = layers[i]->get_type();

        if(layer_type != Layer::Type::Scaling2D
        && layer_type != Layer::Type::Scaling4D
        && layer_type != Layer::Type::Unscaling
        && layer_type != Layer::Type::Bounding)
            return i;
    }

    return 0;
}


Index NeuralNetwork::get_last_trainable_layer_index() const
{
    const Index layers_number = get_layers_number();

    Layer::Type layer_type;

    for(Index i = layers_number-1; i >= 0 ; i--)
    {
        layer_type = layers[i]->get_type();

        if(layer_type != Layer::Type::Scaling2D
        && layer_type != Layer::Type::Scaling4D
        && layer_type != Layer::Type::Unscaling
        && layer_type != Layer::Type::Bounding)
            return i;
    }

    return 0;
}


Index NeuralNetwork::get_perceptron_layers_number() const
{
    const Index layers_number = get_layers_number();

    Index count = 0;

    #pragma omp parallel for reduction(+: count)

    for(Index i = 0; i < layers_number; i++)
        if(layers[i]->get_type() == Layer::Type::Perceptron)
            count++;

    return count;
}


Index NeuralNetwork::get_probabilistic_layers_number() const
{
    const Index layers_number = get_layers_number();

    Index count = 0;

    #pragma omp parallel for reduction(+: count)

    for(Index i = 0; i < layers_number; i++)
        if(layers[i]->get_type() == Layer::Type::Probabilistic)
            count++;

    return count;
}


Index NeuralNetwork::get_long_short_term_memory_layers_number() const
{
    const Index layers_number = get_layers_number();

    Index count = 0;

    #pragma omp parallel for reduction(+: count)

    for(Index i = 0; i < layers_number; i++)
        if(layers[i]->get_type() == Layer::Type::LongShortTermMemory)
            count++;
 
    return count;
}


Index NeuralNetwork::get_flatten_layers_number() const
{
    const Index layers_number = get_layers_number();

    Index count = 0;

    #pragma omp parallel for reduction(+: count)

    for(Index i = 0; i < layers_number; i++)
        if(layers[i]->get_type() == Layer::Type::Flatten)
            count++;

    return count;
}


//Index NeuralNetwork::get_convolutional_layers_number() const
//{
//    const Index layers_number = get_layers_number();
//
//    Index count = 0;
//
//    for(Index i = 0; i < layers_number; i++)
//    {
//        if(layers[i]->get_type() == Layer::Type::Convolutional)
//        {
//            count++;
//        }
//    }
//
//    return count;
//}


Index NeuralNetwork::get_pooling_layers_number() const
{
    const Index layers_number = get_layers_number();

    Index count = 0;

    #pragma omp parallel for reduction(+: count)

    for(Index i = 0; i < layers_number; i++)
        if(layers[i]->get_type() == Layer::Type::Pooling)
            count++;

    return count;
}


Index NeuralNetwork::get_recurrent_layers_number() const
{
    const Index layers_number = get_layers_number();

    Index count = 0;

    #pragma omp parallel for reduction(+: count)

    for(Index i = 0; i < layers_number; i++)
        if(layers[i]->get_type() == Layer::Type::Recurrent)
            count++;

    return count;
}


bool NeuralNetwork::is_input_layer(const vector<Index>& this_layer_inputs_indices) const
{
    const Index input_layers_number = this_layer_inputs_indices.size();

    for(Index i = 0; i < input_layers_number; i++)
        if(this_layer_inputs_indices[i] == -1)
            return true;

    return false;
}


bool NeuralNetwork::is_context_layer(const vector<Index>& this_layer_inputs_indices) const
{   
    // @todo Is this ok?
    const Index layers_number = get_layers_number();

    for(Index i = 0; i < layers_number; i++)
        if(this_layer_inputs_indices[i] == -2)
            return true;

    return false;
}


void NeuralNetwork::set_parameters_constant(const type& value) const
{
    const Index layers_number = get_layers_number();

    #pragma omp parallel for
    for(Index i = 0; i < layers_number; i++)
        layers[i]->set_parameters_constant(value);
}


void NeuralNetwork::set_parameters_random() const
{
    const Index layers_number = get_layers_number();

    const vector<unique_ptr<Layer>>& layers = get_layers();

    #pragma omp parallel for
    for(Index i = 0; i < layers_number; i++)
        layers[i]->set_parameters_random();
}


void NeuralNetwork::forward_propagate(const vector<pair<type*, dimensions>>& input_pairs,
                                      ForwardPropagation& forward_propagation,
                                      const bool& is_training) const
{
    const Index layers_number = get_layers_number();

    const vector<unique_ptr<Layer>>& layers = get_layers();

    const vector<vector<Index>>& layer_input_indices = get_layer_input_indices();

    vector<vector<pair<type*, dimensions>>> layer_input_pairs;
       // = forward_propagation.get_layer_input_pairs(const Batch & batch) const

    const Index first_trainable_layer_index = get_first_trainable_layer_index();
    const Index last_trainable_layer_index = get_last_trainable_layer_index();

    const Index first_layer_index = is_training ? first_trainable_layer_index : 0;
    const Index last_layer_index = is_training ? last_trainable_layer_index : layers_number - 1;
    
    for(Index i = first_layer_index; i <= last_layer_index; i++)
    {
        layers[i]->forward_propagate(layer_input_pairs[i],
                                     forward_propagation.layers[i],
                                     is_training);
    }
}


void NeuralNetwork::forward_propagate(const vector<pair<type*, dimensions>>& input_pairs,
                                      const Tensor<type, 1>& new_parameters,
                                      ForwardPropagation& forward_propagation) const
{
    const Tensor<type, 1> original_parameters = get_parameters();

    set_parameters(new_parameters);

    const bool is_training = true;

    forward_propagate(input_pairs, forward_propagation, is_training);

    set_parameters(original_parameters);
}


Tensor<type, 2> NeuralNetwork::calculate_outputs(const Tensor<type, 2>& inputs)
{
    const Index batch_samples_number = inputs.dimension(0);
    const Index inputs_number = inputs.dimension(1);

    ForwardPropagation forward_propagation(batch_samples_number, this);

    const pair<type*, dimensions> input_pair((type*)inputs.data(), {{batch_samples_number, inputs_number}});

    forward_propagate({input_pair}, forward_propagation);

    const Index layers_number = get_layers_number();

    if(layers_number == 0) return Tensor<type, 2>();
    
    const pair<type*, dimensions> outputs_pair 
        = forward_propagation.layers[layers_number - 1]->get_outputs_pair();

    return tensor_map_2(outputs_pair);
}


Tensor<type, 2> NeuralNetwork::calculate_outputs(const Tensor<type, 4>& inputs)
{
    const Index batch_samples_number = inputs.dimension(0);

    ForwardPropagation forward_propagation(batch_samples_number, this);

    const pair<type*, dimensions> input_pair((type*)inputs.data(), { {inputs.dimension(0), inputs.dimension(1), inputs.dimension(2), inputs.dimension(3)}});

    forward_propagate({input_pair}, forward_propagation);

    const Index layers_number = get_layers_number();

    if(layers_number == 0) return Tensor<type, 2>();

    const pair<type*, dimensions> outputs_pair 
        = forward_propagation.layers[layers_number - 1]->get_outputs_pair();

    return tensor_map_2(outputs_pair);
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


Tensor<string, 2> NeuralNetwork::get_information() const
{

    const Index trainable_layers_number = get_trainable_layers_number();

    Tensor<string, 2> information(trainable_layers_number, 3);

    const vector<Layer*> trainable_layers = get_trainable_layers();

    for(Index i = 0; i < trainable_layers_number; i++)
    {
        information(i,0) = to_string(trainable_layers[i]->get_inputs_number());
        information(i,1) = to_string(trainable_layers[i]->get_neurons_number());

        const string layer_type = trainable_layers[i]->get_type_string();

        if(layer_type == "PerceptronLayer")
        {
            const PerceptronLayer* perceptron_layer = static_cast<PerceptronLayer*>(trainable_layers[i]);

            information(i,2) = perceptron_layer->write_activation_function();
        }
        else if(layer_type == "ProbabilisticLayer")
        {
            const ProbabilisticLayer* probabilistic_layer = static_cast<ProbabilisticLayer*>(trainable_layers[i]);

            information(i,2) = probabilistic_layer->write_activation_function();
        }
        else if(layer_type == "LongShortTermMemoryLayer")
        {
            const LongShortTermMemoryLayer* long_short_term_memory_layer = static_cast<LongShortTermMemoryLayer*>(trainable_layers[i]);

            information(i,2) = long_short_term_memory_layer->write_activation_function();
        }
        else if(layer_type == "RecurrentLayer")
        {
            const RecurrentLayer* recurrent_layer = static_cast<RecurrentLayer*>(trainable_layers[i]);

            information(i,2) = recurrent_layer->write_activation_function();
        }
        else
        {
            information(i,2) = "No activation function";
        }
    }

    return information;
}


Tensor<string, 2> NeuralNetwork::get_perceptron_layers_information() const
{
    const Index layers_number = get_layers_number();

    const Index perceptron_layers_number = get_perceptron_layers_number();

    Tensor<string, 2> information(perceptron_layers_number, 3);

    Index perceptron_layer_index = 0;

    for(Index i = 0; i < layers_number; i++)
    {
        const Layer::Type layer_type = layers[i]->get_type();

        if(layer_type == Layer::Type::Perceptron)
        {
            information(perceptron_layer_index,0) = to_string(layers[i]->get_inputs_number());
            information(perceptron_layer_index,1) = to_string(layers[i]->get_neurons_number());

            const PerceptronLayer* perceptron_layer = static_cast<PerceptronLayer*>(layers[i].get());

            information(perceptron_layer_index, 2) = perceptron_layer->write_activation_function();

            perceptron_layer_index++;
        }
    }

    return information;
}


Tensor<string, 2> NeuralNetwork::get_probabilistic_layer_information() const
{
    const Index layers_number = get_layers_number();

    const Index probabilistic_layers_number = get_probabilistic_layers_number();

    Tensor<string, 2> information(probabilistic_layers_number, 3);

    Index probabilistic_layer_index = 0;

    for(Index i = 0; i < layers_number; i++)
    {
        const Layer::Type layer_type = layers[i]->get_type();

        if(layer_type == Layer::Type::Probabilistic)
        {
            information(probabilistic_layer_index,0) = to_string(layers[i]->get_inputs_number());
            information(probabilistic_layer_index,1) = to_string(layers[i]->get_neurons_number());

            const ProbabilisticLayer* probabilistic_layer = static_cast<ProbabilisticLayer*>(layers[i].get());

            information(probabilistic_layer_index,2) = probabilistic_layer->write_activation_function();

            probabilistic_layer_index++;
        }
    }

    return information;
}


void NeuralNetwork::to_XML(tinyxml2::XMLPrinter& file_stream) const
{
    file_stream.OpenElement("NeuralNetwork");

    // Inputs

    file_stream.OpenElement("Inputs");

    // Inputs number

    const Index inputs_number = get_inputs_number();

    file_stream.OpenElement("InputsNumber");   
    file_stream.PushText(to_string(inputs_number).c_str());
    file_stream.CloseElement();

    // Inputs names

    for(Index i = 0; i < inputs_number; i++)
    {
        if(inputs_name.size() != inputs_number)
            throw runtime_error("Size of inputs name is not equal to inputs number");

        file_stream.OpenElement("Input");
        file_stream.PushAttribute("Index", to_string(i+1).c_str());
        file_stream.PushText(inputs_name[i].c_str());
        file_stream.CloseElement();
    }

    // Inputs (end tag)

    file_stream.CloseElement();

    // Layers

    file_stream.OpenElement("Layers");

    // Layers number

    const Index layers_number = get_layers_number();

    file_stream.OpenElement("LayersNumber");
    file_stream.PushText(to_string(layers_number).c_str());
    file_stream.CloseElement();

    // Layers

    for(Index i = 0; i < layers_number; i++)
    {
        layers[i]->to_XML(file_stream);
    }
/*
    ostringstream buffer;

    // Layers inputs indices

    file_stream.OpenElement("LayersInputsIndices");

    for(Index i = 0; i < layer_input_indices.size(); i++)
    {
        file_stream.OpenElement("LayerInputsIndices");

        file_stream.PushAttribute("LayerIndex", to_string(i+1).c_str());

        const Tensor<Index, 1>& indices = layer_input_indices[i];

        buffer.str("");

        for(Index j = 0; j < indices.size(); j++)
        {
            buffer << indices(j);

            if(j != indices.size() - 1)
                buffer << " ";
        }

        file_stream.PushText(buffer.str().c_str());

        file_stream.CloseElement();
    }

    file_stream.CloseElement();
*/
    // Layers (end tag)

    file_stream.CloseElement();

    // Ouputs

    file_stream.OpenElement("Outputs");

    // Outputs number

    const Index outputs_number = output_names.size();
    file_stream.OpenElement("OutputsNumber");
    file_stream.PushText(to_string(outputs_number).c_str());
    file_stream.CloseElement();

    // Outputs names

    for(Index i = 0; i < output_names.size(); i++)
    {
        file_stream.OpenElement("Output");

        file_stream.PushAttribute("Index", to_string(i+1).c_str());

        file_stream.PushText(output_names[i].c_str());

        file_stream.CloseElement();
    }

    //Outputs (end tag)

    file_stream.CloseElement();

    // Neural network (end tag)

    file_stream.CloseElement();
}


void NeuralNetwork::from_XML(const tinyxml2::XMLDocument& document)
{
    set();

    const tinyxml2::XMLElement* neural_network_element = document.FirstChildElement("NeuralNetwork");

    if(!neural_network_element)
        throw runtime_error("Neural network element is nullptr.\n");

    // Inputs

    const tinyxml2::XMLElement* inputs_element = neural_network_element->FirstChildElement("Inputs");

    if(!inputs_element)
        throw runtime_error("Inputs element is nullptr.");

    tinyxml2::XMLDocument inputs_document;
    tinyxml2::XMLNode* inputs_element_clone = inputs_element->DeepClone(&inputs_document);

    inputs_document.InsertFirstChild(inputs_element_clone);

    inputs_from_XML(inputs_document);

    // Layers

    const tinyxml2::XMLElement* layers_element = neural_network_element->FirstChildElement("Layers");

    if(!layers_element)
        throw runtime_error("Layers element is nullptr.");

    tinyxml2::XMLDocument layers_document;
    tinyxml2::XMLNode* layers_element_clone = layers_element->DeepClone(&layers_document);

    layers_document.InsertFirstChild(layers_element_clone);

    layers_from_XML(layers_document);

    // Outputs

    const tinyxml2::XMLElement* outputs_element = neural_network_element->FirstChildElement("Outputs");

    if(!outputs_element)
        throw runtime_error("Outputs element is nullptr.");

    tinyxml2::XMLDocument outputs_document;
    tinyxml2::XMLNode* outputs_element_clone = outputs_element->DeepClone(&outputs_document);

    outputs_document.InsertFirstChild(outputs_element_clone);

    outputs_from_XML(outputs_document);
    
    // Display

    const tinyxml2::XMLElement* display_element = neural_network_element->FirstChildElement("Display");

    if(display_element)
        set_display(display_element->GetText() != string("0"));
}


void NeuralNetwork::inputs_from_XML(const tinyxml2::XMLDocument& document)
{
    const tinyxml2::XMLElement* inputs_element = document.FirstChildElement("Inputs");

    if(!inputs_element)
        throw runtime_error("Inputs element is nullptr.\n");

    // Inputs number

    const tinyxml2::XMLElement* inputs_number_element = inputs_element->FirstChildElement("InputsNumber");

    if(!inputs_number_element)
        throw runtime_error("Inputs number element is nullptr.\n");

    const Index new_inputs_number = Index(atoi(inputs_number_element->GetText()));
    inputs_name.resize(new_inputs_number);

    // if(inputs_number_element->GetText())
    //     set_inputs_number(inputs_number);

    // Inputs names

    const tinyxml2::XMLElement* start_element = inputs_number_element;

    for(Index i = 0; i < new_inputs_number; i++)
    {
        const tinyxml2::XMLElement* input_element = start_element->NextSiblingElement("Input");

        if(input_element->Attribute("Index") != to_string(i+1))
            throw runtime_error("Input index number (" + to_string(i+1) + ") does not match (" + input_element->Attribute("Item") + ").\n");

        if(!input_element->GetText())
            throw runtime_error("Input text is nullptr.");

        inputs_name(i) = input_element->GetText();

        start_element = input_element;
    }
}


void NeuralNetwork::layers_from_XML(const tinyxml2::XMLDocument& document)
{
    const tinyxml2::XMLElement* layers_element = document.FirstChildElement("Layers");

    if(!layers_element)
        throw runtime_error("Layers element is nullptr.\n");

    // Layers types

    const tinyxml2::XMLElement* layers_number_element = layers_element->FirstChildElement("LayersNumber");

    if(!layers_number_element)
        throw runtime_error("LayersNumber element is nullptr.\n");

    const Index layers_number = Index(atoi(layers_number_element->GetText()));

    // Add layers

    const tinyxml2::XMLElement* start_element = layers_number_element;

    for(Index i = 0; i < layers_number; i++)
    {
        const tinyxml2::XMLElement* layer_element = start_element->NextSiblingElement();

        if(!layer_element)
             throw runtime_error("Layer element is nullptr.");

        const string layer_type = layer_element->Name();

        tinyxml2::XMLDocument layer_document;
        tinyxml2::XMLNode* element_clone = layer_element->DeepClone(&layer_document);
        layer_document.InsertFirstChild(element_clone);

        if(layer_type == "ScalingLayer2D")
        {
            unique_ptr<ScalingLayer2D> scaling_layer = make_unique<ScalingLayer2D>();
            scaling_layer->from_XML(layer_document);
            add_layer(std::move(scaling_layer));
        }
        else if(layer_type == "Scaling4D")
        {
            unique_ptr<ScalingLayer4D> scaling_layer = make_unique<ScalingLayer4D>();
            scaling_layer->from_XML(layer_document);
            add_layer(std::move(scaling_layer));
        }
        else if(layer_type == "ConvolutionalLayer")
        {
            unique_ptr<ConvolutionalLayer> convolutional_layer = make_unique<ConvolutionalLayer>();
            convolutional_layer->from_XML(layer_document);
            add_layer(std::move(convolutional_layer));
        }
        else if(layer_type == "PerceptronLayer")
        {
            unique_ptr<PerceptronLayer> perceptron_layer = make_unique<PerceptronLayer>();
            perceptron_layer->from_XML(layer_document);
            add_layer(std::move(perceptron_layer));
        }
        else if(layer_type == "PerceptronLayer3D")
        {
            unique_ptr<PerceptronLayer3D> perceptron_layer_3d = make_unique<PerceptronLayer3D>();
            perceptron_layer_3d->from_XML(layer_document);
            add_layer(std::move(perceptron_layer_3d));
        }
        else if(layer_type == "PoolingLayer")
        {
            unique_ptr<PoolingLayer> pooling_layer = make_unique<PoolingLayer>();
            pooling_layer->from_XML(layer_document);
            add_layer(std::move(pooling_layer));
        }
        else if(layer_type == "ProbabilisticLayer")
        {
            unique_ptr<ProbabilisticLayer> probabilistic_layer = make_unique<ProbabilisticLayer>();
            probabilistic_layer->from_XML(layer_document);
            add_layer(std::move(probabilistic_layer));
        }
        else if(layer_type == "ProbabilisticLayer3D")
        {
            unique_ptr<ProbabilisticLayer3D> probabilistic_layer_3d = make_unique<ProbabilisticLayer3D>();
            probabilistic_layer_3d->from_XML(layer_document);
            add_layer(std::move(probabilistic_layer_3d));
        }
        else if(layer_type == "LongShortTermMemoryLayer")
        {
            unique_ptr<LongShortTermMemoryLayer> long_short_term_memory_layer = make_unique<LongShortTermMemoryLayer>();
            long_short_term_memory_layer->from_XML(layer_document);
            add_layer(std::move(long_short_term_memory_layer));
        }
        else if(layer_type == "RecurrentLayer")
        {
            unique_ptr<RecurrentLayer> recurrent_layer = make_unique<RecurrentLayer>();
            recurrent_layer->from_XML(layer_document);
            add_layer(std::move(recurrent_layer));
        }
        else if(layer_type == "UnscalingLayer")
        {
            unique_ptr<UnscalingLayer> unscaling_layer = make_unique<UnscalingLayer>();
            unscaling_layer->from_XML(layer_document);
            add_layer(std::move(unscaling_layer));
        }
        else if(layer_type == "BoundingLayer")
        {
            unique_ptr<BoundingLayer> bounding_layer = make_unique<BoundingLayer>();
            bounding_layer->from_XML(layer_document);
            add_layer(std::move(bounding_layer));
        }
        else if(layer_type == "EmbeddingLayer")
        {
            unique_ptr<EmbeddingLayer> embedding_layer = make_unique<EmbeddingLayer>();
            embedding_layer->from_XML(layer_document);
            add_layer(std::move(embedding_layer));
        }
        else if(layer_type == "MultiheadAttentionLayer")
        {
            unique_ptr<MultiheadAttentionLayer> multihead_attention_layer = make_unique<MultiheadAttentionLayer>();
            multihead_attention_layer->from_XML(layer_document);
            add_layer(std::move(multihead_attention_layer));
        }
        else if(layer_type == "AdditionLayer3D")
        {
            unique_ptr<AdditionLayer3D> addition_layer_3d = make_unique<AdditionLayer3D>();
            addition_layer_3d->from_XML(layer_document);
            add_layer(std::move(addition_layer_3d));
        }
        else if(layer_type == "NormalizationLayer3D")
        {
            unique_ptr<NormalizationLayer3D> normalization_layer_3d = make_unique<NormalizationLayer3D>();
            normalization_layer_3d->from_XML(layer_document);
            add_layer(std::move(normalization_layer_3d));
        }
        else
        {
            throw runtime_error("Unknown layer type: " + layer_type);
        }

        start_element = layer_element;
    }

    // Layers inputs indices
/*
    const tinyxml2::XMLElement* layer_input_indices_element = layers_element->FirstChildElement("LayersInputsIndices");

    if(!layer_input_indices_element)
        throw runtime_error("LayersInputsIndices element is nullptr.\n");

    layer_input_indices.resize(layers.size());

    for(const tinyxml2::XMLElement* layer_inputs_indices_element = layer_input_indices_element->FirstChildElement("LayerInputsIndices");
        layer_inputs_indices_element;
        layer_inputs_indices_element = layer_inputs_indices_element->NextSiblingElement("LayerInputsIndices"))
    {
        if(layer_inputs_indices_element->GetText())
        {
//            const Index layer_index = Index(stoi(layer_inputs_indices_element->Attribute("LayerIndex"))) - 1;
//            const string indices_string = layer_inputs_indices_element->GetText();
//            layer_input_indices[layer_index] = to_type_vector(indices_string, ' ').cast<Index>();
        }
    }
*/
}


void NeuralNetwork::outputs_from_XML(const tinyxml2::XMLDocument& document)
{
    ostringstream buffer;

    const tinyxml2::XMLElement* root_element = document.FirstChildElement("Outputs");

    if(!root_element)
        throw runtime_error("Outputs element is nullptr.\n");

    // Outputs number
    
    const tinyxml2::XMLElement* outputs_number_element = root_element->FirstChildElement("OutputsNumber");

    if(!outputs_number_element)
        throw runtime_error("Outputs number element is nullptr.\n");

    Index new_outputs_number = 0;

    if(outputs_number_element->GetText())
        new_outputs_number = Index(atoi(outputs_number_element->GetText()));

    // Outputs names

    const tinyxml2::XMLElement* start_element = outputs_number_element;

    output_names.resize(new_outputs_number);

    for(Index i = 0; i < new_outputs_number; i++)
    {
        const tinyxml2::XMLElement* output_element = start_element->NextSiblingElement("Output");
        start_element = output_element;

        if(output_element->Attribute("Index") != to_string(i+1))
            throw runtime_error("Output index number (" + to_string(i+1) + ") does not match (" + output_element->Attribute("Item") + ").\n");

        if(output_element->GetText())
            output_names(i) = output_element->GetText();
    }
}


void NeuralNetwork::print() const
{
    cout << "Neural network" << endl;

    if(model_type != ModelType::ImageClassification)
    {
        cout << "Inputs:" << endl
             << get_input_names() << endl;
    }

    const Index layers_number = get_layers_number();       

    cout << "Layers number: " << layers_number << endl;

    for(Index i = 0; i < layers_number; i++)
    {
        cout << endl
             << "Layer " << i << ": " << endl;
        layers[i]->print();

    }

    cout << "Outputs:" << endl
         << get_output_names() << endl;

    cout << "Parameters:" << endl
         << get_parameters_number() << endl;
}


void NeuralNetwork::save(const string& file_name) const
{
    FILE * file = fopen(file_name.c_str(), "w");

    if(file)
    {
        tinyxml2::XMLPrinter printer(file);
        to_XML(printer);
        fclose(file);
    }
}


void NeuralNetwork::save_parameters(const string& file_name) const
{
    ofstream file(file_name.c_str());

    if(!file.is_open())
        throw runtime_error("Cannot open parameters data file.\n");

    const Tensor<type, 1> parameters = get_parameters();

    file << parameters << endl;

    // Close file

    file.close();
}


void NeuralNetwork::load(const string& file_name)
{
    set_default();

    tinyxml2::XMLDocument document;

    if(document.LoadFile(file_name.c_str()))
        throw runtime_error("Cannot load XML file " + file_name + ".\n");

    from_XML(document);
}


void NeuralNetwork::load_parameters_binary(const string& file_name)
{
    ifstream file;

    file.open(file_name.c_str(), ios::binary);

    if(!file.is_open())
        throw runtime_error("Cannot open binary file: " + file_name + "\n");

    streamsize size = sizeof(type);

    const Index parameters_number = get_parameters_number();

    Tensor<type, 1> new_parameters(parameters_number);

    type value = 0;

    for(Index i = 0; i < parameters_number; i++)
    {
        file.read(reinterpret_cast<char*>(&value), size);

        new_parameters(i) = value;
    }

    set_parameters(new_parameters);
}


void NeuralNetwork::save_expression_c(const string& file_name) const
{
    ofstream file(file_name.c_str());

    if(!file.is_open())
        throw runtime_error("Cannot open expression text file.\n");
    /*
    file << write_expression_c();
    */
    file.close();
}


void NeuralNetwork::save_expression_api(const string& file_name) const
{
    ofstream file(file_name.c_str());

    if(!file.is_open())
        throw runtime_error("Cannot open expression text file.\n");
    /*
    file << write_expression_api();
    */
    file.close();
}


void NeuralNetwork::save_expression_javascript(const string& file_name) const
{
    ofstream file(file_name.c_str());

    if(!file.is_open())
        throw runtime_error("Cannot open expression text file.\n");
    /*
    file << write_expression_javascript();
    */
    file.close();
}


void NeuralNetwork::save_expression_python(const string& file_name) const
{
    ofstream file(file_name.c_str());

    if(!file.is_open())
        throw runtime_error("Cannot open expression text file.\n");
/*
    file << write_expression_python();
*/
    file.close();
}


void NeuralNetwork::save_outputs(Tensor<type, 2>& inputs, const string & file_name)
{
    const Tensor<type, 2> outputs = calculate_outputs(inputs);

    ofstream file(file_name.c_str());

    if(!file.is_open())
        throw runtime_error("Cannot open " + file_name + " file.\n");

    const Tensor<string, 1> output_names = get_output_names();

    const Index outputs_number = get_outputs_number();
    const Index samples_number = inputs.dimension(0);

    for(Index i = 0; i < outputs_number; i++)
    {
        file << output_names[i];

        if(i != output_names.size()-1) 
            file << ";";
    }

    file << "\n";

    for(Index i = 0; i < samples_number; i++)
    {
        for(Index j = 0; j < outputs_number; j++)
        {
            file << outputs(i, j);

            if(j != outputs_number-1) file << ";";
        }

        file << "\n";
    }

    file.close();
}


Tensor<string, 1> NeuralNetwork::get_layer_names() const
{
    const Index layers_number = get_layers_number();

    Tensor<string, 1> layer_names(layers_number);

    for(Index i = 0; i < layers_number; i++)
        layer_names[i] = layers[i]->get_name();

    return layer_names;
}


Tensor<string, 1> NeuralNetwork::get_layer_types_string() const
{
    const Index layers_number = get_layers_number();

    Tensor<string, 1> layer_types(layers_number);

    for(Index i = 0; i < layers_number; i++)
        layer_types[i] = layers[i]->get_type_string();

    return layer_types;
}


void NeuralNetworkBackPropagation::set(const Index& new_batch_samples_number, NeuralNetwork* new_neural_network)
{
    batch_samples_number = new_batch_samples_number;

    neural_network = new_neural_network;

    const vector<unique_ptr<Layer>>& neural_network_layers = neural_network->get_layers();

    const Index layers_number = neural_network_layers.size();

    layers.resize(layers_number);
    layers.setConstant(nullptr);

    for(Index i = 0; i < layers_number; i++)
    {
        switch (neural_network_layers[i]->get_type())
        {
        case Layer::Type::Perceptron:
            layers[i] = new PerceptronLayerBackPropagation(batch_samples_number, neural_network_layers[i].get());
        break;

        case Layer::Type::PerceptronLayer3D:
            layers[i] = new PerceptronLayer3DBackPropagation(batch_samples_number, neural_network_layers[i].get());
        break;

        case Layer::Type::Probabilistic:
            layers[i] = new ProbabilisticLayerBackPropagation(batch_samples_number, neural_network_layers[i].get());
        break;

        case Layer::Type::Probabilistic3D:
            layers[i] = new ProbabilisticLayer3DBackPropagation(batch_samples_number, neural_network_layers[i].get());
        break;

        case Layer::Type::Recurrent:
            layers[i] = new RecurrentLayerBackPropagation(batch_samples_number, neural_network_layers[i].get());
        break;

        case Layer::Type::LongShortTermMemory:
            layers[i] = new LongShortTermMemoryLayerBackPropagation(batch_samples_number, neural_network_layers[i].get());
        break;

        case Layer::Type::Convolutional:
            layers[i] = new ConvolutionalLayerBackPropagation(batch_samples_number, neural_network_layers[i].get());
        break;

        case Layer::Type::Pooling:
            layers[i] = new PoolingLayerBackPropagation(batch_samples_number, neural_network_layers[i].get());
        break;

        case Layer::Type::Flatten:
            layers[i] = new FlattenLayerBackPropagation(batch_samples_number, neural_network_layers[i].get());
        break;

        case Layer::Type::Embedding:
            layers[i] = new EmbeddingLayerBackPropagation(batch_samples_number, neural_network_layers[i].get());
        break;

        case Layer::Type::MultiheadAttention:
            layers[i] = new MultiheadAttentionLayerBackPropagation(batch_samples_number, neural_network_layers[i].get());
        break;

        case Layer::Type::Addition3D:
            layers[i] = new AdditionLayer3DBackPropagation(batch_samples_number, neural_network_layers[i].get());
        break;

        case Layer::Type::Normalization3D:
            layers[i] = new NormalizationLayer3DBackPropagation(batch_samples_number, neural_network_layers[i].get());
        break;

        default: break;
        }
    }
}


void ForwardPropagation::set(const Index& new_batch_samples_number, NeuralNetwork* new_neural_network)
{
    batch_samples_number = new_batch_samples_number;

    neural_network = new_neural_network;

    const vector<unique_ptr<Layer>>& neural_network_layers = neural_network->get_layers();

    const Index layers_number = neural_network_layers.size();

    layers.resize(layers_number, nullptr);

    for(Index i = 0; i < layers_number; i++)
    {
        switch (neural_network_layers[i]->get_type())
        {
        case Layer::Type::Perceptron:
            layers[i] = new PerceptronLayerForwardPropagation(batch_samples_number, neural_network_layers[i].get());
        break;

        case Layer::Type::PerceptronLayer3D:
            layers[i] = new PerceptronLayer3DForwardPropagation(batch_samples_number, neural_network_layers[i].get());
        break;

        case Layer::Type::Probabilistic:
            layers[i] = new ProbabilisticLayerForwardPropagation(batch_samples_number, neural_network_layers[i].get());
        break;

        case Layer::Type::Probabilistic3D:
            layers[i] = new ProbabilisticLayer3DForwardPropagation(batch_samples_number, neural_network_layers[i].get());
        break;

        case Layer::Type::Recurrent:
            layers[i] = new RecurrentLayerForwardPropagation(batch_samples_number, neural_network_layers[i].get());
        break;

        case Layer::Type::LongShortTermMemory:
            layers[i] = new LongShortTermMemoryLayerForwardPropagation(batch_samples_number, neural_network_layers[i].get());
        break;

        case Layer::Type::Convolutional:
            layers[i] = new ConvolutionalLayerForwardPropagation(batch_samples_number, neural_network_layers[i].get());
        break;

        case Layer::Type::Pooling:
            layers[i] = new PoolingLayerForwardPropagation(batch_samples_number, neural_network_layers[i].get());
        break;

        case Layer::Type::Flatten:
            layers[i] = new FlattenLayerForwardPropagation(batch_samples_number, neural_network_layers[i].get());
        break;

        case Layer::Type::Scaling2D:
            layers[i] = new ScalingLayer2DForwardPropagation(batch_samples_number, neural_network_layers[i].get());
        break;

        case Layer::Type::Scaling4D:
            layers[i] = new ScalingLayer4DForwardPropagation(batch_samples_number, neural_network_layers[i].get());
        break;

        case Layer::Type::Unscaling:
            layers[i] = new UnscalingLayerForwardPropagation(batch_samples_number, neural_network_layers[i].get());
        break;

        case Layer::Type::Bounding:
            layers[i] = new BoundingLayerForwardPropagation(batch_samples_number, neural_network_layers[i].get());
        break;

        case Layer::Type::Embedding:
            layers[i] = new EmbeddingLayerForwardPropagation(batch_samples_number, neural_network_layers[i].get());
        break;

        case Layer::Type::MultiheadAttention:
            layers[i] = new MultiheadAttentionLayerForwardPropagation(batch_samples_number, neural_network_layers[i].get());
        break;

        case Layer::Type::Addition3D:
            layers[i] = new AdditionLayer3DForwardPropagation(batch_samples_number, neural_network_layers[i].get());
        break;

        case Layer::Type::Normalization3D:
            layers[i] = new NormalizationLayer3DForwardPropagation(batch_samples_number, neural_network_layers[i].get());
        break;

        default: cout << "Default" << endl; break;
        }
    }
}


pair<type*, dimensions> ForwardPropagation::get_last_trainable_layer_outputs_pair() const
{
    const Index last_trainable_layer_index = neural_network->get_last_trainable_layer_index();

    return layers[last_trainable_layer_index]->get_outputs_pair();
}


vector<vector<pair<type*, dimensions>>> ForwardPropagation::get_layer_input_pairs(const Batch& batch) const
{
    const Index layers_number = neural_network->get_layers_number();

    vector<vector<pair<type*, dimensions>>> layer_input_pairs(layers_number);

    const vector<vector<Index>>& layer_input_indices = neural_network->get_layer_input_indices();

    const vector<pair<type*, dimensions>> input_pairs = batch.get_input_pairs();

    for (Index i = 0; i < layers_number; i++)
    {
        const vector<Index>& this_layer_input_indices = layer_input_indices[i];

        if (neural_network->is_input_layer(this_layer_input_indices))
        {
            layer_input_pairs[i].push_back(input_pairs[0]);
        }
        else if (neural_network->is_context_layer(this_layer_input_indices))
        {
            layer_input_pairs[i].push_back(input_pairs[1]);
        }
        else
        {
            const Index this_layer_inputs_number = this_layer_input_indices.size();

            for (Index j = 0; j < this_layer_inputs_number; j++)
            {
                const Index this_layer_input_index = this_layer_input_indices[j];

                layer_input_pairs[i].push_back(layers[this_layer_input_index]->get_outputs_pair());
            }
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
        cout << "Layer " << i + 1 << ": " << layers[i]->layer->get_name() << endl;

        layers[i]->print();
    }
}



void NeuralNetworkBackPropagationLM::set(const Index new_batch_samples_number, NeuralNetwork* new_neural_network)
{

    batch_samples_number = new_batch_samples_number;

    neural_network = new_neural_network;

    const Index layers_number = neural_network->get_layers_number();
    const Index trainable_layers_number = neural_network->get_trainable_layers_number();

    layers.resize(trainable_layers_number);
/*
    for(Index i = 0; i < trainable_layers_number; i++)
    {
        switch (trainable_layers(i)->get_type())
        {
        case Layer::Type::Perceptron:
            layers[i] = new PerceptronLayerBackPropagationLM(batch_samples_number, trainable_layers(i).get());
            break;

        case Layer::Type::Probabilistic:
            layers[i] = new ProbabilisticLayerBackPropagationLM(batch_samples_number, trainable_layers(i).get());
            break;

        default:
            throw runtime_error("Levenberg-Marquardt can only be used with Perceptron and Probabilistic layers.\n");
        }
    }
*/
}
}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2024 Artificial Intelligence Techniques, SL.
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
