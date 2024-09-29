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
    
NeuralNetwork::NeuralNetwork()
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


NeuralNetwork::NeuralNetwork(const Tensor<Layer*, 1>& new_layers)
{
    set();

    layers = new_layers;
}


NeuralNetwork::~NeuralNetwork()
{
    delete_layers();
}


void NeuralNetwork::delete_layers()
{
    const Index layers_number = get_layers_number();

    for(Index i = 0;  i < layers_number; i++)
    {
        delete layers[i];

        layers[i] = nullptr;
    }

    layers.resize(0);
}


void NeuralNetwork::add_layer(Layer* layer, const string& name)
{
    const Layer::Type layer_type = layer->get_type();

    if(!validate_layer_type(layer_type)) return;

    const Index old_layers_number = get_layers_number();

    const Tensor<Layer*, 1> old_layers = get_layers();

    const Tensor<Tensor<Index, 1>, 1> old_layers_inputs_indices = get_layers_input_indices();

    layers.resize(old_layers_number + 1);

    for(Index i = 0; i < old_layers_number; i++)
        layers(i) = old_layers(i);

    layers(old_layers_number) = layer;

    layers_inputs_indices.resize(old_layers_number+1);

    for(Index i = 0; i < old_layers_number; i++)
        layers_inputs_indices(i) = old_layers_inputs_indices(i);

    Tensor<Index, 1> new_layer_inputs_indices(1);
    new_layer_inputs_indices(0) = old_layers_number-1;

    layers_inputs_indices(old_layers_number) = new_layer_inputs_indices;

    // if (layer_type == Layer::Type::Flatten)
    // {
    //     if (old_layers_number > 0)
    //     {
    //         Layer* previous_layer = old_layers(old_layers_number - 1);

    //         if (previous_layer->get_type() == Layer::Type::Convolutional)
    //         {
    //             ConvolutionalLayer* convolutional_layer = static_cast<ConvolutionalLayer*>(previous_layer);
    //         }
    //     }
    // }

//    layer.set_layer .set_layer_name(name);
}


bool NeuralNetwork::validate_layer_type(const Layer::Type layer_type)
{
    const Index layers_number = layers.size();

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
    return layers.dimension(0) == 0;
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


Tensor<Layer*, 1> NeuralNetwork::get_layers() const
{
    return layers;
}


Layer* NeuralNetwork::get_layer(const Index& layer_index) const
{
    return layers(layer_index);
}


Layer* NeuralNetwork::get_layer(const string& name) const
{
    const Tensor<string, 1> layer_names = get_layer_names();

    for(Index i = 0; i < layer_names.size(); i++)
        if(layer_names(i) == name)
            return layers(i);

    return nullptr;
}


Tensor<Layer*, 1> NeuralNetwork::get_trainable_layers() const
{
//    const Index layers_number = get_layers_number();

    const Index trainable_layers_number = get_trainable_layers_number();

    Tensor<Layer*, 1> trainable_layers(trainable_layers_number);

    copy_if(layers.data(), layers.data() + layers.size(), trainable_layers.data(), [](Layer* layer)
    {
        const Layer::Type layer_type = layer->get_type();

        return layer_type != Layer::Type::Scaling2D
            && layer_type != Layer::Type::Scaling4D
            && layer_type != Layer::Type::Unscaling
            && layer_type != Layer::Type::Bounding;
    });

    // Index index = 0;

    // Layer::Type layer_type;

    // for(Index i = 0; i < layers_number; i++)
    // {
    //     layer_type = layers(i)->get_type();

    //     if(layer_type != Layer::Type::Scaling2D
    //     && layer_type != Layer::Type::Scaling4D
    //     && layer_type != Layer::Type::Unscaling
    //     && layer_type != Layer::Type::Bounding)
    //     {
    //         trainable_layers(index) = layers(i);
    //         index++;
    //     }
    // }

    return trainable_layers;
}


Index NeuralNetwork::get_layer_index(const string& name) const
{
    const Index layers_number = get_layers_number();

    if(name == "dataset" || name == "input")
        return -1;

    if(name == "context")
        return -2;

    for(Index i = 0; i < layers_number; i++)
        if(layers(i)->get_name() == name)
            return i;

    return 0;
}


// Tensor<Index, 1> NeuralNetwork::get_trainable_layers_indices() const
// {
//     const Index layers_number = get_layers_number();

//     const Index trainable_layers_number = get_trainable_layers_number();

//     Tensor<Index, 1> trainable_layers_indices(trainable_layers_number);

//     Index trainable_layer_index = 0;

//     for(Index i = 0; i < layers_number; i++)
//     {
//         if(layers[i]->get_type() != Layer::Type::Scaling2D
//         && layers[i]->get_type() != Layer::Type::Scaling4D
//         && layers[i]->get_type() != Layer::Type::Unscaling
//         && layers[i]->get_type() != Layer::Type::Bounding)
//         {
//             trainable_layers_indices[trainable_layer_index] = i;
//             trainable_layer_index++;
//         }
//     }

//     return trainable_layers_indices;
// }


const Tensor<Tensor<Index, 1>, 1>& NeuralNetwork::get_layers_input_indices() const
{
    return layers_inputs_indices;
}


ScalingLayer2D* NeuralNetwork::get_scaling_layer_2d() const
{
    const Index layers_number = get_layers_number();

    for(Index i = 0; i < layers_number; i++)
        if(layers[i]->get_type() == Layer::Type::Scaling2D)
            return dynamic_cast<ScalingLayer2D*>(layers[i]);

    throw runtime_error("No scaling layer 2d in neural network.\n");
}


ScalingLayer4D* NeuralNetwork::get_scaling_layer_4d() const
{
    const Index layers_number = get_layers_number();

    for(Index i = 0; i < layers_number; i++)
        if(layers[i]->get_type() == Layer::Type::Scaling4D)
            return dynamic_cast<ScalingLayer4D*>(layers[i]);

    throw runtime_error("No scaling layer in neural network.\n");
}


UnscalingLayer* NeuralNetwork::get_unscaling_layer() const
{
    const Index layers_number = get_layers_number();

    for(Index i = 0; i < layers_number; i++)
        if(layers[i]->get_type() == Layer::Type::Unscaling)
            return dynamic_cast<UnscalingLayer*>(layers[i]);

    throw runtime_error("No unscaling layer in neural network.\n");
}


BoundingLayer* NeuralNetwork::get_bounding_layer() const
{
    const Index layers_number = get_layers_number();

    for(Index i = 0; i < layers_number; i++)
        if(layers[i]->get_type() == Layer::Type::Bounding)
            return dynamic_cast<BoundingLayer*>(layers[i]);

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
//    {
//        if(layers[i]->get_type() == Layer::Type::Convolutional)
//        {
//            return dynamic_cast<ConvolutionalLayer*>(layers[i]);
//        }
//    }
//
//    throw runtime_error("No convolutional layer in neural network.\n");
//}


// PoolingLayer* NeuralNetwork::get_pooling_layer() const
// {
//     const Index layers_number = get_layers_number();

//     for(Index i = 0; i < layers_number; i++)
//     {
//         if(layers[i]->get_type() == Layer::Type::PoolingLayer)
//         {
//             return dynamic_cast<PoolingLayer*>(layers[i]);
//         }
//     }

//     throw runtime_error("No pooling layer in neural network.\n");
// }


ProbabilisticLayer* NeuralNetwork::get_probabilistic_layer() const
{
    const Index layers_number = get_layers_number();

    for(Index i = 0; i < layers_number; i++)
        if(layers[i]->get_type() == Layer::Type::Probabilistic)
            return dynamic_cast<ProbabilisticLayer*>(layers[i]);

    throw runtime_error("No probabilistic layer in neural network.\n");
}


LongShortTermMemoryLayer* NeuralNetwork::get_long_short_term_memory_layer() const
{
    const Index layers_number = get_layers_number();

    for(Index i = 0; i < layers_number; i++)
        if(layers[i]->get_type() == Layer::Type::LongShortTermMemory)
            return dynamic_cast<LongShortTermMemoryLayer*>(layers[i]);

    throw runtime_error("No long-short-term memory layer in neural network.\n");
}


RecurrentLayer* NeuralNetwork::get_recurrent_layer() const
{
    const Index layers_number = get_layers_number();

    for(Index i = 0; i < layers_number; i++)
        if(layers[i]->get_type() == Layer::Type::Recurrent)
            return dynamic_cast<RecurrentLayer*>(layers[i]);

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

    delete_layers();

    set_default();
}


void NeuralNetwork::set(const NeuralNetwork::ModelType& new_model_type,
                        const dimensions& input_dimensions, 
                        const dimensions& complexity_dimensions,
                        const dimensions& output_dimensions)
{
    delete_layers();

    model_type = new_model_type;

    const Index complexity_size = complexity_dimensions.size();

    const Index inputs_number = accumulate(input_dimensions.begin(), input_dimensions.end(), 1, multiplies<Index>());

    inputs_name.resize(inputs_number);

    for(Index i = 0; i < inputs_number; i++)
        inputs_name(i) = "input_" + to_string(i+1);

    if(model_type == ModelType::Approximation)
    {
        add_layer(new ScalingLayer2D(input_dimensions));

        for(Index i = 0; i < complexity_size; i++)
            add_layer(new PerceptronLayer(get_output_dimensions(), {complexity_dimensions[i]}, PerceptronLayer::ActivationFunction::HyperbolicTangent),
                      "perceptron_layer_" + to_string(i+1));

        add_layer(new PerceptronLayer(get_output_dimensions(), output_dimensions, PerceptronLayer::ActivationFunction::Linear), "perceptron_layer_" + to_string(complexity_size+1));

        add_layer(new UnscalingLayer(output_dimensions));

        add_layer(new BoundingLayer(output_dimensions));
    }
    else if(model_type == ModelType::Classification || model_type == ModelType::TextClassification)
    {
        for(Index i = 0; i < complexity_size; i++)
            add_layer(new PerceptronLayer(get_output_dimensions(), {complexity_dimensions[i]}, PerceptronLayer::ActivationFunction::HyperbolicTangent),
                      "perceptron_layer_" + to_string(i+1));

        add_layer(new ProbabilisticLayer(get_output_dimensions(), output_dimensions), "probabilistic_layer");
    }
    else if(model_type == ModelType::Forecasting)
    {
        add_layer(new ScalingLayer2D(inputs_number));

        add_layer(new UnscalingLayer(output_dimensions));

        add_layer(new BoundingLayer(output_dimensions));
    }
    else if(model_type == ModelType::ImageClassification)
    {
        add_layer(new ScalingLayer4D(input_dimensions));

        for(Index i = 0; i < complexity_size; i++)
        {
            const dimensions kernel_dimensions = {3, 3, get_output_dimensions()[2], complexity_dimensions[i]};

            add_layer(new ConvolutionalLayer(get_output_dimensions(), kernel_dimensions),
                      "convolutional_layer_" + to_string(i+1));

            const dimensions pool_dimensions = {2, 2};

            add_layer(new PoolingLayer(get_output_dimensions(), pool_dimensions),
                      "pooling_layer_" + to_string(i+1));
        }

        add_layer(new FlattenLayer(get_output_dimensions()));

        add_layer(new ProbabilisticLayer(get_output_dimensions(), output_dimensions), "probabilistic_layer");
    }
    else if(model_type == ModelType::AutoAssociation)
    {
/*
        add_layer(new ScalingLayer2D(inputs_number));

        const Index mapping_neurons_number = 10;
        const Index bottle_neck_neurons_number = complexity_dimensions[0];

        add_layer(new PerceptronLayer(input_dimensions[0], mapping_neurons_number, PerceptronLayer::ActivationFunction::HyperbolicTangent),
                  "mapping_layer");

        add_layer(new PerceptronLayer(mapping_neurons_number, bottle_neck_neurons_number, PerceptronLayer::ActivationFunction::Linear),
                  "bottleneck_layer");

        add_layer(new PerceptronLayer(bottle_neck_neurons_number, mapping_neurons_number, PerceptronLayer::ActivationFunction::HyperbolicTangent),
                  "demapping_layer");

        add_layer(new PerceptronLayer(mapping_neurons_number, output_dimensions[0], PerceptronLayer::ActivationFunction::Linear),
                  "output_layer");

        add_layer(new UnscalingLayer(output_dimensions[0]));
*/
    }

    const Index outputs_number = accumulate(output_dimensions.begin(), output_dimensions.end(), 1, multiplies<Index>());

    output_names.resize(outputs_number);

    for(Index i = 0; i < outputs_number; i++)
        output_names(i) = "output_" + to_string(i+1);

    set_default();
}


void NeuralNetwork::set(const string& file_name)
{
    delete_layers();

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
    Tensor<Layer*, 1> trainable_layers = get_trainable_layers();

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

    const int n = omp_get_max_threads();

    thread_pool = new ThreadPool(n);
    thread_pool_device = new ThreadPoolDevice(thread_pool, n);
}


void NeuralNetwork::set_threads_number(const int& new_threads_number)
{
    const Index layers_number = get_layers_number();

    for(Index i = 0; i < layers_number; i++)
        layers(i)->set_threads_number(new_threads_number);
}


void NeuralNetwork::set_layers_number(const Index& new_layers_number)
{
    layers.resize(new_layers_number);
    layers_inputs_indices.resize(new_layers_number);
}


void NeuralNetwork::set_layers(Tensor<Layer*, 1>& new_layers)
{
    layers = new_layers;
}


void NeuralNetwork::set_layers_inputs_indices(const Tensor<Tensor<Index, 1>, 1>& new_layers_inputs_indices)
{
    layers_inputs_indices = new_layers_inputs_indices;
}


void NeuralNetwork::set_layer_inputs_indices(const Index& layer_index, const Tensor<Index, 1>& new_layer_inputs_indices)
{
    layers_inputs_indices(layer_index) = new_layer_inputs_indices;
}


void NeuralNetwork::set_layer_inputs_indices(const string& name, const Tensor<string, 1>& new_layer_inputs_names)
{
    const Index layer_index = get_layer_index(name);

    const Index size = new_layer_inputs_names.size();

    Tensor<Index, 1> new_layer_inputs_indices(size);

    for(Index i = 0; i < size; i++)
    {
        new_layer_inputs_indices(i) = get_layer_index(new_layer_inputs_names(i));
    }

    layers_inputs_indices(layer_index) = new_layer_inputs_indices;
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

    Tensor<Index, 1> new_layer_inputs_indices(1);

    new_layer_inputs_indices(0) = get_layer_index(new_layer_inputs_name);

    layers_inputs_indices(layer_index) = new_layer_inputs_indices;
}


PerceptronLayer* NeuralNetwork::get_first_perceptron_layer() const
{
    const Index layers_number = get_layers_number();

    for(Index i = 0; i < layers_number; i++)
        if(layers(i)->get_type() == Layer::Type::Perceptron)
            return static_cast<PerceptronLayer*>(layers[i]);

    return nullptr;
}


Index NeuralNetwork::get_inputs_number() const
{
    if(layers.dimension(0) != 0)
        return layers(0)->get_inputs_number();

    return 0;
}


Index NeuralNetwork::get_outputs_number() const
{
    if(layers.size() == 0) return 0;

    const Layer* last_layer = layers[layers.size() - 1];

    const dimensions output_dimensions = last_layer->get_output_dimensions();

    const Index outputs_rank = output_dimensions.size();

    Index outputs_number = 1;

    for(Index i = 0; i < outputs_rank; i++)
        outputs_number *= output_dimensions[i];

    return outputs_number;
}


dimensions NeuralNetwork::get_output_dimensions() const
{
    if(layers.size() == 0) return {};

    const Layer* last_layer = layers[layers.size() - 1];

    return last_layer->get_output_dimensions();
}


Tensor<Index, 1> NeuralNetwork::get_architecture() const
{
    const Index layers_number = get_layers_number();

    const Index inputs_number = get_inputs_number();

    if(layers_number == 0 || inputs_number == 0) return Tensor<Index, 1>();

    Tensor<Index, 1> architecture(layers_number);

    for(Index i = 0; i < layers_number; i++)
        architecture(i) = layers(i)->get_neurons_number();

    return architecture;
}


Index NeuralNetwork::get_parameters_number() const
{
    const Tensor<Layer*, 1> trainable_layers = get_trainable_layers();

    Index parameters_number = 0;

    for(Index i = 0; i < trainable_layers.size(); i++)
        if(trainable_layers[i] == nullptr)
            cout << "Layer " << i << " is nullptr." << endl;
        else
            parameters_number += trainable_layers[i]->get_parameters_number();

    return parameters_number;
}


Tensor<type, 1> NeuralNetwork::get_parameters() const
{
    const Index parameters_number = get_parameters_number();

    Tensor<type, 1> parameters(parameters_number);

    const Index trainable_layers_number = get_trainable_layers_number();

    const Tensor<Layer*, 1> trainable_layers = get_trainable_layers();

    Index position = 0;

    for(Index i = 0; i < trainable_layers_number; i++)
    {
        const Tensor<type, 1> layer_parameters = trainable_layers(i)->get_parameters();

        // @todo use memcpy

        for(Index j = 0; j < layer_parameters.size(); j++)
            parameters(j + position) = layer_parameters(j);

        position += layer_parameters.size();
    }

    return parameters;
}


Tensor<Index, 1> NeuralNetwork::get_layers_parameters_numbers() const
{
    const Index layers_number = get_layers_number();

    Tensor<Index, 1> layers_parameters_number(layers_number);

    for(Index i = 0; i < layers_number; i++)
        layers_parameters_number[i] = layers(i)->get_parameters_number();

    return layers_parameters_number;
}


Tensor<Index, 1> NeuralNetwork::get_trainable_layers_parameters_numbers() const
{
    const Index trainable_layers_number = get_trainable_layers_number();

    const Tensor<Layer*, 1> trainable_layers = get_trainable_layers();

    Tensor<Index, 1> trainable_layers_parameters_number(trainable_layers_number);

    for(Index i = 0; i < trainable_layers_number; i++)
        trainable_layers_parameters_number[i] = trainable_layers[i]->get_parameters_number();

    return trainable_layers_parameters_number;
}


void NeuralNetwork::set_parameters(const Tensor<type, 1>& new_parameters) const
{
    const Index trainable_layers_number = get_trainable_layers_number();

    const Tensor<Layer*, 1> trainable_layers = get_trainable_layers();

    const Tensor<Index, 1> trainable_layers_parameters_numbers = get_trainable_layers_parameters_numbers();

    Index index = 0;

    // @todo parallelize

    for(Index i = 0; i < trainable_layers_number; i++)
    {
        trainable_layers(i)->set_parameters(new_parameters, index);

        index += trainable_layers_parameters_numbers(i);
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
        if(layers(i)->get_type() != Layer::Type::Scaling2D
        && layers(i)->get_type() != Layer::Type::Scaling4D
        && layers(i)->get_type() != Layer::Type::Unscaling
        && layers(i)->get_type() != Layer::Type::Bounding)
        {
            count++;
        }
    }

    return count;
}


Index NeuralNetwork::get_first_trainable_layer_index() const
{
    const Index layers_number = get_layers_number();

    Layer::Type layer_type;

    for(Index i = 0; i < layers_number; i++)
    {
        layer_type = layers(i)->get_type();

        if(layer_type != Layer::Type::Scaling2D
        && layer_type != Layer::Type::Scaling4D
        && layer_type != Layer::Type::Unscaling
        && layer_type != Layer::Type::Bounding)
        {
            return i;
        }
    }

    return 0;
}


Index NeuralNetwork::get_last_trainable_layer_index() const
{
    const Index layers_number = get_layers_number();

    Layer::Type layer_type;

    for(Index i = layers_number-1; i >= 0 ; i--)
    {
        layer_type = layers(i)->get_type();

        if(layer_type != Layer::Type::Scaling2D
        && layer_type != Layer::Type::Scaling4D
        && layer_type != Layer::Type::Unscaling
        && layer_type != Layer::Type::Bounding)
        {
            return i;
        }
    }

    return 0;
}


Index NeuralNetwork::get_perceptron_layers_number() const
{
    const Index layers_number = get_layers_number();

    Index count = 0;

    #pragma omp parallel for reduction(+: count)

    for(Index i = 0; i < layers_number; i++)
        if(layers(i)->get_type() == Layer::Type::Perceptron)
            count++;

    return count;
}


Index NeuralNetwork::get_probabilistic_layers_number() const
{
    const Index layers_number = get_layers_number();

    Index count = 0;

    #pragma omp parallel for reduction(+: count)

    for(Index i = 0; i < layers_number; i++)
        if(layers(i)->get_type() == Layer::Type::Probabilistic)
            count++;

    return count;
}


Index NeuralNetwork::get_long_short_term_memory_layers_number() const
{
    const Index layers_number = get_layers_number();

    Index count = 0;

    #pragma omp parallel for reduction(+: count)

    for(Index i = 0; i < layers_number; i++)
        if(layers(i)->get_type() == Layer::Type::LongShortTermMemory)
            count++;
 
    return count;
}


Index NeuralNetwork::get_flatten_layers_number() const
{
    const Index layers_number = get_layers_number();

    Index count = 0;

    #pragma omp parallel for reduction(+: count)

    for(Index i = 0; i < layers_number; i++)
        if(layers(i)->get_type() == Layer::Type::Flatten)
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
//        if(layers(i)->get_type() == Layer::Type::Convolutional)
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
        if(layers(i)->get_type() == Layer::Type::Pooling)
            count++;

    return count;
}


Index NeuralNetwork::get_recurrent_layers_number() const
{
    const Index layers_number = get_layers_number();

    Index count = 0;

    #pragma omp parallel for reduction(+: count)

    for(Index i = 0; i < layers_number; i++)
        if(layers(i)->get_type() == Layer::Type::Recurrent)
            count++;

    return count;
}


bool NeuralNetwork::is_input_layer(const Tensor<Index, 1>& layer_inputs_indices) const
{
    for(Index i = 0; i < layer_inputs_indices.size(); i++)
        if(layer_inputs_indices(i) == -1) 
            return true;

    return false;
}


bool NeuralNetwork::is_context_layer(const Tensor<Index, 1>& layer_inputs_indices) const
{
    for(Index i = 0; i < layer_inputs_indices.size(); i++)
        if(layer_inputs_indices(i) == -2) 
            return true;

    return false;
}


void NeuralNetwork::set_parameters_constant(const type& value) const
{
    const Index trainable_layers_number = get_trainable_layers_number();

    const Tensor<Layer*, 1> trainable_layers = get_trainable_layers();

    #pragma omp parallel for

    for(Index i = 0; i < trainable_layers_number; i++)
        trainable_layers[i]->set_parameters_constant(value);
}


void NeuralNetwork::set_parameters_random() const
{
    const Index layers_number = get_layers_number();

    Tensor<Layer*, 1> layers = get_layers();

    #pragma omp parallel for

    for(Index i = 0; i < layers_number; i++)
        layers[i]->set_parameters_random();
}


void NeuralNetwork::forward_propagate(const Tensor<pair<type*, dimensions>, 1>& inputs_pair,
                                      ForwardPropagation& forward_propagation,
                                      const bool& is_training) const
{
    const Index layers_number = get_layers_number();

    const Tensor<Layer*, 1> layers = get_layers();

    const Tensor<Tensor<Index, 1>, 1> layers_inputs_indices = get_layers_input_indices();

    const Index first_trainable_layer_index = get_first_trainable_layer_index();
    const Index last_trainable_layer_index = get_last_trainable_layer_index();

    Index first_layer_index;
    Index last_layer_index;
    
    if(is_training)
    {
        first_layer_index = first_trainable_layer_index;
        last_layer_index = last_trainable_layer_index;
    }
    else
    {
        first_layer_index = 0;
        last_layer_index = layers_number - 1;
    }

    Tensor<pair<type*, dimensions>, 1> layer_inputs;
    
    for(Index i = first_layer_index; i <= last_layer_index; i++)
    {
        const Tensor<Index, 1> layer_input_indices = layers_inputs_indices(i);
        const Index layer_inputs_number = layer_input_indices.size();

        if(i == first_layer_index || is_input_layer(layer_input_indices))
        {
            layer_inputs.resize(1);

            layer_inputs(0) = inputs_pair(0);
        }
        else if(is_context_layer(layer_input_indices))
        {
            layer_inputs.resize(1);

            layer_inputs(0) = inputs_pair(1);
        }
        else
        {
            layer_inputs.resize(layer_inputs_number);

            for(Index j = 0; j < layer_inputs_number; j++)
            {
                layer_inputs(j) = forward_propagation.layers(layers_inputs_indices(i)(j))->get_outputs_pair();
            }
        }

        layers(i)->forward_propagate(layer_inputs,
                                     forward_propagation.layers(i),
                                     is_training);
    }
}


void NeuralNetwork::forward_propagate(const Tensor<pair<type*, dimensions>, 1>& inputs_pair,
                                      const Tensor<type, 1>& new_parameters,
                                      ForwardPropagation& forward_propagation) const
{
    const Tensor<type, 1> original_parameters = get_parameters();

    set_parameters(new_parameters);

    const bool is_training = true;

    forward_propagate(inputs_pair, forward_propagation, is_training);

    set_parameters(original_parameters);
}


Tensor<type, 2> NeuralNetwork::calculate_outputs(const Tensor<type, 2>& inputs)
{
    const Index batch_samples_number = inputs.dimension(0);
    const Index inputs_number = inputs.dimension(1);

    ForwardPropagation forward_propagation(batch_samples_number, this);

    const pair<type*, dimensions> inputs_pair((type*)inputs.data(), {{batch_samples_number, inputs_number}});

    forward_propagate(tensor_wrapper(inputs_pair), forward_propagation);

    const Index layers_number = get_layers_number();

    if(layers_number == 0) return Tensor<type, 2>();
    
    const pair<type*, dimensions> outputs_pair = forward_propagation.layers(layers_number - 1)->get_outputs_pair();

    const TensorMap<Tensor<type, 2>> outputs(outputs_pair.first, outputs_pair.second[0], outputs_pair.second[1]);

    return outputs;
}


Tensor<type, 2> NeuralNetwork::calculate_outputs(const Tensor<type, 4>& inputs)
{
    const Index batch_samples_number = inputs.dimension(0);

    ForwardPropagation forward_propagation(batch_samples_number, this);

    const pair<type*, dimensions> inputs_pair((type*)inputs.data(), { {inputs.dimension(0), inputs.dimension(1), inputs.dimension(2), inputs.dimension(3)}});

    forward_propagate(tensor_wrapper(inputs_pair), forward_propagation);

    const Index layers_number = get_layers_number();

    if(layers_number == 0) return Tensor<type, 2>();

    const pair<type*, dimensions> outputs_pair = forward_propagation.layers(layers_number - 1)->get_outputs_pair();

    const TensorMap<Tensor<type, 2>> outputs(outputs_pair.first, outputs_pair.second[0], outputs_pair.second[1]);

    return outputs;
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

    Tensor<Layer*, 1> trainable_layers = get_trainable_layers();

    for(Index i = 0; i < trainable_layers_number; i++)
    {
        information(i,0) = to_string(trainable_layers(i)->get_inputs_number());
        information(i,1) = to_string(trainable_layers(i)->get_neurons_number());

        const string layer_type = trainable_layers(i)->get_type_string();

        if(layer_type == "PerceptronLayer")
        {
            const PerceptronLayer* perceptron_layer = static_cast<PerceptronLayer*>(trainable_layers(i));

            information(i,2) = perceptron_layer->write_activation_function();
        }
        else if(layer_type == "ProbabilisticLayer")
        {
            const ProbabilisticLayer* probabilistic_layer = static_cast<ProbabilisticLayer*>(trainable_layers(i));

            information(i,2) = probabilistic_layer->write_activation_function();
        }
        else if(layer_type == "LongShortTermMemoryLayer")
        {
            const LongShortTermMemoryLayer* long_short_term_memory_layer = static_cast<LongShortTermMemoryLayer*>(trainable_layers(i));

            information(i,2) = long_short_term_memory_layer->write_activation_function();
        }
        else if(layer_type == "RecurrentLayer")
        {
            const RecurrentLayer* recurrent_layer = static_cast<RecurrentLayer*>(trainable_layers(i));

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
    const Index trainable_layers_number = get_trainable_layers_number();

    const Index perceptron_layers_number = get_perceptron_layers_number();

    Tensor<string, 2> information(perceptron_layers_number, 3);

    Tensor<Layer*, 1> trainable_layers = get_trainable_layers();

    Index perceptron_layer_index = 0;

    for(Index i = 0; i < trainable_layers_number; i++)
    {
        const string layer_type = trainable_layers(i)->get_type_string();

        if(layer_type == "PerceptronLayer")
        {
            information(perceptron_layer_index,0) = to_string(trainable_layers(i)->get_inputs_number());
            information(perceptron_layer_index,1) = to_string(trainable_layers(i)->get_neurons_number());

            const PerceptronLayer* perceptron_layer = static_cast<PerceptronLayer*>(trainable_layers(i));

            information(perceptron_layer_index, 2) = perceptron_layer->write_activation_function();

            perceptron_layer_index++;
        }
    }

    return information;
}


Tensor<string, 2> NeuralNetwork::get_probabilistic_layer_information() const
{
    const Index trainable_layers_number = get_trainable_layers_number();

    const Index probabilistic_layers_number = get_probabilistic_layers_number();

    Tensor<string, 2> information(probabilistic_layers_number, 3);

    Tensor<Layer*, 1> trainable_layers = get_trainable_layers();

    Index probabilistic_layer_index = 0;

    for(Index i = 0; i < trainable_layers_number; i++)
    {
        const string layer_type = trainable_layers(i)->get_type_string();

        if(layer_type == "ProbabilisticLayer")
        {
            information(probabilistic_layer_index,0) = to_string(trainable_layers(i)->get_inputs_number());
            information(probabilistic_layer_index,1) = to_string(trainable_layers(i)->get_neurons_number());

            const ProbabilisticLayer* probabilistic_layer = static_cast<ProbabilisticLayer*>(trainable_layers(i));

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

    for(Index i = 0; i < layers_inputs_indices.size(); i++)
    {
        file_stream.OpenElement("LayerInputsIndices");

        file_stream.PushAttribute("LayerIndex", to_string(i+1).c_str());

        const Tensor<Index, 1>& indices = layers_inputs_indices(i);

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
            ScalingLayer2D* scaling_layer = new ScalingLayer2D();
            scaling_layer->from_XML(layer_document);
            add_layer(scaling_layer);
        }
        else if(layer_type == "Scaling4D")
        {
            ScalingLayer4D* scaling_layer = new ScalingLayer4D();
            scaling_layer->from_XML(layer_document);
            add_layer(scaling_layer);
        }
        else if(layer_type == "ConvolutionalLayer")
        {
            ConvolutionalLayer* convolutional_layer = new ConvolutionalLayer();
            convolutional_layer->from_XML(layer_document);
            add_layer(convolutional_layer);
        }
        else if(layer_type == "PerceptronLayer")
        {
            PerceptronLayer* perceptron_layer = new PerceptronLayer();
            perceptron_layer->from_XML(layer_document);
            add_layer(perceptron_layer);
        }
        else if(layer_type == "PerceptronLayer3D")
        {
            PerceptronLayer3D* perceptron_layer_3d = new PerceptronLayer3D();
            perceptron_layer_3d->from_XML(layer_document);
            add_layer(perceptron_layer_3d);
        }
        else if(layer_type == "PoolingLayer")
        {
            PoolingLayer* pooling_layer = new PoolingLayer();
            pooling_layer->from_XML(layer_document);
            add_layer(pooling_layer);
        }
        else if(layer_type == "ProbabilisticLayer")
        {
            ProbabilisticLayer* probabilistic_layer = new ProbabilisticLayer();
            probabilistic_layer->from_XML(layer_document);
            add_layer(probabilistic_layer);
        }
        else if(layer_type == "ProbabilisticLayer3D")
        {
            ProbabilisticLayer3D* probabilistic_layer_3d = new ProbabilisticLayer3D();
            probabilistic_layer_3d->from_XML(layer_document);
            add_layer(probabilistic_layer_3d);
        }
        else if(layer_type == "LongShortTermMemoryLayer")
        {
            LongShortTermMemoryLayer* long_short_term_memory_layer = new LongShortTermMemoryLayer();
            long_short_term_memory_layer->from_XML(layer_document);
            add_layer(long_short_term_memory_layer);
        }
        else if(layer_type == "RecurrentLayer")
        {
            RecurrentLayer* recurrent_layer = new RecurrentLayer();
            recurrent_layer->from_XML(layer_document);
            add_layer(recurrent_layer);
        }
        else if(layer_type == "UnscalingLayer")
        {
            UnscalingLayer* unscaling_layer = new UnscalingLayer();
            unscaling_layer->from_XML(layer_document);
            add_layer(unscaling_layer);
        }
        else if(layer_type == "BoundingLayer")
        {
            BoundingLayer* bounding_layer = new BoundingLayer();
            bounding_layer->from_XML(layer_document);
            add_layer(bounding_layer);
        }
        else if(layer_type == "EmbeddingLayer")
        {
            EmbeddingLayer* embedding_layer = new EmbeddingLayer();
            embedding_layer->from_XML(layer_document);
            add_layer(embedding_layer);
        }
        else if(layer_type == "MultiheadAttentionLayer")
        {
            MultiheadAttentionLayer* multihead_attention_layer = new MultiheadAttentionLayer();
            multihead_attention_layer->from_XML(layer_document);
            add_layer(multihead_attention_layer); 
        }
        else if(layer_type == "AdditionLayer3D")
        {
            AdditionLayer3D* addition_layer_3d = new AdditionLayer3D();
            addition_layer_3d->from_XML(layer_document);
            add_layer(addition_layer_3d);
        }
        else if(layer_type == "NormalizationLayer3D")
        {
            NormalizationLayer3D* normalization_layer_3d = new NormalizationLayer3D();
            normalization_layer_3d->from_XML(layer_document);
            add_layer(normalization_layer_3d);
        }
        else
        {
            throw runtime_error("Unknown layer type: " + layer_type);
        }

        start_element = layer_element;
    }

    // Layers inputs indices
/*
    const tinyxml2::XMLElement* layers_inputs_indices_element = layers_element->FirstChildElement("LayersInputsIndices");

    if(!layers_inputs_indices_element)
        throw runtime_error("LayersInputsIndices element is nullptr.\n");

    layers_inputs_indices.resize(layers.size());

    for(const tinyxml2::XMLElement* layer_inputs_indices_element = layers_inputs_indices_element->FirstChildElement("LayerInputsIndices");
        layer_inputs_indices_element;
        layer_inputs_indices_element = layer_inputs_indices_element->NextSiblingElement("LayerInputsIndices"))
    {
        if(layer_inputs_indices_element->GetText())
        {
//            const Index layer_index = Index(stoi(layer_inputs_indices_element->Attribute("LayerIndex"))) - 1;
//            const string indices_string = layer_inputs_indices_element->GetText();
//            layers_inputs_indices(layer_index) = to_type_vector(indices_string, ' ').cast<Index>();
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


string NeuralNetwork::write_expression() const
{
    const Index layers_number = get_layers_number();

    const Tensor<Layer*, 1> layers = get_layers();
    const Tensor<string, 1> layer_names = get_layer_names();

    Tensor<string, 1> output_namess_vector;
    Tensor<string, 1> inputs_names_vector;
    inputs_names_vector = inputs_name;
    string aux_name;

    for(int i = 0; i < inputs_name.dimension(0); i++)
    {
        if(!inputs_names_vector[i].empty())
        {
            aux_name = inputs_name[i];
            inputs_names_vector(i) = replace_non_allowed_programming_expressions(aux_name);
        }
        else
        {
            inputs_names_vector(i) = "input_" + to_string(i);
        }
    }

    Index layer_neurons_number;

    Tensor<string, 1> scaled_inputs_names(inputs_name.dimension(0));
    Tensor<string, 1> unscaled_output_namess(inputs_name.dimension(0));

    ostringstream buffer;

    for(Index i = 0; i < layers_number; i++)
    {
        if(i == layers_number-1)
        {
            output_namess_vector = output_names;

            for(int j = 0; j < output_names.dimension(0); j++)
            {
                if(!output_namess_vector[j].empty())
                {
                    aux_name = output_names[j];
                    output_namess_vector(j) = replace_non_allowed_programming_expressions(aux_name);
                }
                else
                {
                    output_namess_vector(j) = "output_" + to_string(i);
                }
            }
            buffer << layers[i]->write_expression(inputs_names_vector, output_namess_vector) << endl;
        }
        else
        {
            layer_neurons_number = layers[i]->get_neurons_number();
            output_namess_vector.resize(layer_neurons_number);

            for(Index j = 0; j < layer_neurons_number; j++)
            {
                if(layer_names(i) == "scaling_layer")
                {
                    aux_name = inputs_name(j);
                    output_namess_vector(j) = "scaled_" + replace_non_allowed_programming_expressions(aux_name);
                    scaled_inputs_names(j) = output_namess_vector(j);
                }
                else
                {
                    output_namess_vector(j) =  layer_names(i) + "_output_" + to_string(j);
                }
            }
            buffer << layers[i]->write_expression(inputs_names_vector, output_namess_vector) << endl;
            inputs_names_vector = output_namess_vector;
            unscaled_output_namess = inputs_names_vector;
        }
    }

    string expression = buffer.str();

    replace(expression, "+-", "-");

    return expression;
}


string NeuralNetwork::write_expression_c() const
{
    string aux;
    ostringstream buffer;
    ostringstream outputs_buffer;

    Tensor<string, 1> inputs =  get_input_names();
    Tensor<string, 1> outputs = get_output_names();
    Tensor<string, 1> found_tokens;

    int cell_states_counter = 0;
    int hidden_state_counter = 0;
    int LSTM_number = get_long_short_term_memory_layers_number();

    bool logistic     = false;
    bool ReLU         = false;
    bool ExpLinear    = false;
    bool SExpLinear   = false;
    bool HSigmoid     = false;
    bool SoftPlus     = false;
    bool SoftSign     = false;

    buffer << "// Artificial Intelligence Techniques SL\t" << endl
           << "// artelnics@artelnics.com\t" << endl
           << "// Your model has been exported to this c file." << endl
           << "// You can manage it with the main method, where you \t" << endl
           << "// can change the values of your inputs. For example:" << endl
           << "// if we want to add these 3 values (0.3, 2.5 and 1.8)" << endl
           << "// to our 3 inputs (Input_1, Input_2 and Input_1), the" << endl
           << "// main program has to look like this:" << endl
           << "// \t" << endl
           << "// int main(){ " << endl
           << "// \t" << "vector<float> inputs(3);"<< endl
           << "// \t" << endl
           << "// \t" << "const float asdas  = 0.3;" << endl
           << "// \t" << "inputs[0] = asdas;"        << endl
           << "// \t" << "const float input2 = 2.5;" << endl
           << "// \t" << "inputs[1] = input2;"       << endl
           << "// \t" << "const float input3 = 1.8;" << endl
           << "// \t" << "inputs[2] = input3;"       << endl
           << "// \t" << ". . ." << endl
           << "// \n" << endl
           << "// Inputs Names:" <<endl;

    Tensor<Tensor<string,1>, 1> inputs_outputs_buffer = fix_input_output_variables(inputs, outputs, buffer);

    for(Index i = 0; i < inputs_outputs_buffer(0).dimension(0);i++)
        inputs(i) = inputs_outputs_buffer(0)(i);

    for(Index i = 0; i < inputs_outputs_buffer(1).dimension(0);i++)
        outputs(i) = inputs_outputs_buffer(1)(i);

    buffer << inputs_outputs_buffer(2)[0]
           << "\n" << endl
           << "#include <iostream>" << endl
           << "#include <vector>" << endl
           << "#include <math.h>" << endl
           << "#include <stdio.h>" << endl
           << "\n" << endl
           << "using namespace std;" << endl
           << "\n" << endl;

    string token;
    string expression = write_expression();

    if(model_type == ModelType::AutoAssociation)
    {
    // Delete intermediate calculations

    // sample_autoassociation_distance
    {
        const string word_to_delete = "sample_autoassociation_distance =";

        const size_t index = expression.find(word_to_delete);

        if(index != string::npos)
        {
            expression.erase(index, string::npos);
        }

    }

    // sample_autoassociation_variables_distance
    {
        const string word_to_delete = "sample_autoassociation_variables_distance =";

        const size_t index = expression.find(word_to_delete);

        if(index != string::npos)
            expression.erase(index, string::npos);
    }
    }

    stringstream ss(expression);

    Tensor<string, 1> tokens;

    while(getline(ss, token, '\n'))
    {
        if(token.size() > 1 && token.back() == '{')
            break;

        if(token.size() > 1 && token.back() != ';')
            token += ';';

        push_back_string(tokens, token);
    }

    for(int i = 0; i < tokens.dimension(0); i++)
    {
        string t = tokens(i);

        const string word = get_word_from_token(t);

        if(word.size() > 1 && !find_string_in_tensor(found_tokens, word))
            push_back_string(found_tokens, word);
    }

    const string target_string0("Logistic");
    const string target_string1("ReLU");
    const string target_string4("ExponentialLinear");
    const string target_string5("SELU");
    const string target_string6("HardSigmoid");
    const string target_string7("SoftPlus");
    const string target_string8("SoftSign");

    for(int i = 0; i < tokens.dimension(0); i++)
    {
        const string t = tokens(i);

        const size_t substring_length0 = t.find(target_string0);
        const size_t substring_length1 = t.find(target_string1);
        const size_t substring_length4 = t.find(target_string4);
        const size_t substring_length5 = t.find(target_string5);
        const size_t substring_length6 = t.find(target_string6);
        const size_t substring_length7 = t.find(target_string7);
        const size_t substring_length8 = t.find(target_string8);

        if(substring_length0 < t.size() && substring_length0!=0){ logistic = true; }
        if(substring_length1 < t.size() && substring_length1!=0){ ReLU = true; }
        if(substring_length4 < t.size() && substring_length4!=0){ ExpLinear = true; }
        if(substring_length5 < t.size() && substring_length5!=0){ SExpLinear = true; }
        if(substring_length6 < t.size() && substring_length6!=0){ HSigmoid = true; }
        if(substring_length7 < t.size() && substring_length7!=0){ SoftPlus = true; }
        if(substring_length8 < t.size() && substring_length8!=0){ SoftSign = true; }
    }

    if(logistic)
    {
        buffer << "float Logistic (float x) {" << endl
               << "float z = 1/(1+exp(-x));" << endl
               << "return z;" << endl
               << "}" << endl
               << "\n" << endl;
    }

    if(ReLU)
    {
        buffer << "float ReLU(float x) {" << endl
               << "float z = max(0, x);" << endl
               << "return z;" << endl
               << "}" << endl
               << "\n" << endl;
    }

    if(ExpLinear)
    {
        buffer << "float ExponentialLinear(float x) {" << endl
               << "float z;" << endl
               << "float alpha = 1.67326;" << endl
               << "if(x>0){" << endl
               << "z = x;" << endl
               << "}else{" << endl
               << "z = alpha*(exp(x)-1);" << endl
               << "}" << endl
               << "return z;" << endl
               << "}" << endl
               << "\n" << endl;
    }

    if(SExpLinear)
    {
        buffer << "float SELU(float x) {" << endl
               << "float z;" << endl
               << "float alpha  = 1.67326;" << endl
               << "float lambda = 1.05070;" << endl
               << "if(x > 0){" << endl
               << "z = lambda*x;" << endl
               << "}else{" << endl
               << "z = lambda*alpha*(exp(x)-1);" << endl
               << "}" << endl
               << "return z;" << endl
               << "}" << endl
               << "\n" << endl;
    }

    if(HSigmoid)
    {
        buffer << "float HardSigmoid(float x) {" << endl
               << "float z = 1/(1+exp(-x));" << endl
               << "return z;" << endl
               << "}" << endl
               << "\n" << endl;
    }

    if(SoftPlus)
    {
        buffer << "float SoftPlus(float x) {" << endl
               << "float z = log(1+exp(x));" << endl
               << "return z;" << endl
               << "}" << endl
               << "\n" << endl;
    }

    if(SoftSign)
    {
        buffer << "float SoftSign(float x) {" << endl
               << "float z = x/(1+abs(x));" << endl
               << "return z;" << endl
               << "}" << endl
               << "\n" << endl;
    }

    if(LSTM_number > 0)
    {
        for(int i = 0; i < found_tokens.dimension(0); i++)
        {
            const string token = found_tokens(i);

            if(token.find("cell_state") == 0)
                cell_states_counter += 1;

            if(token.find("hidden_state") == 0)
                hidden_state_counter += 1;
        }

        buffer << "struct LSTMMemory" << endl
               << "{" << endl
               << "\t" << "int current_combinations_derivatives = 3;" << endl
               << "\t" << "int time_step_counter = 1;" << endl;

        for(int i = 0; i < hidden_state_counter; i++)
            buffer << "\t" << "float hidden_state_" << to_string(i) << " = type(0);" << endl;

        for(int i = 0; i < cell_states_counter; i++)
            buffer << "\t" << "float cell_states_" << to_string(i) << " = type(0);" << endl;

        buffer << "} lstm; \n\n" << endl
               << "vector<float> calculate_outputs(const vector<float>& inputs, LSTMMemory& lstm)" << endl;
    }
    else
    {
        buffer << "vector<float> calculate_outputs(const vector<float>& inputs)" << endl;
    }

    buffer << "{" << endl;

    for(int i = 0; i < inputs.dimension(0); i++)
    {
        if(inputs[i].empty())
            buffer << "\t" << "const float " << "input_" << to_string(i) << " = " << "inputs[" << to_string(i) << "];" << endl;
        else
            buffer << "\t" << "const float " << inputs[i] << " = " << "inputs[" << to_string(i) << "];" << endl;
    }

    if(LSTM_number>0)
    {
        buffer << "\n\tif(lstm.time_step_counter%lstm.current_combinations_derivatives == 0 ){" << endl
               << "\t\t" << "lstm.time_step_counter = 1;" << endl;

        for(int i = 0; i < hidden_state_counter; i++)
        {
            buffer << "\t\t" << "lstm.hidden_state_" << to_string(i) << " = type(0);" << endl;
        }

        for(int i = 0; i < cell_states_counter; i++)
            buffer << "\t\t" << "lstm.cell_states_" << to_string(i) << " = type(0);" << endl;

        buffer << "\t}" << endl;
    }

    buffer << "" << endl;

    for(int i = 0; i < tokens.dimension(0); i++)
    {
        const string t = tokens(i);

        if(t.size() <= 1)
            outputs_buffer << "" << endl;
        else
            outputs_buffer << "\t" << t << endl;
    }

    const string keyword = "double";

    string outputs_espresion = outputs_buffer.str();

    replace_substring_in_string(found_tokens, outputs_espresion, keyword);

    if(LSTM_number>0)
    {
        replace_all_appearances(outputs_espresion, "(t)", "");
        replace_all_appearances(outputs_espresion, "(t-1)", "");
        replace_all_appearances(outputs_espresion, "double cell_state", "cell_state");
        replace_all_appearances(outputs_espresion, "double hidden_state", "hidden_state");
        replace_all_appearances(outputs_espresion, "cell_state"  , "lstm.cell_state");
        replace_all_appearances(outputs_espresion, "hidden_state", "lstm.hidden_state");
    }

    buffer << outputs_espresion;

    const Tensor<string, 1> fixed_outputs = fix_write_expression_outputs(expression, outputs, "c");

    for(int i = 0; i < fixed_outputs.dimension(0); i++)
        buffer << fixed_outputs(i) << endl;

    buffer << "\t" << "vector<float> out(" << outputs.size() << ");" << endl;

    for(int i = 0; i < outputs.dimension(0); i++)
        if(outputs[i].empty())
            buffer << "\t" << "out[" << to_string(i) << "] = " << "output" << to_string(i) << ";"<< endl;
        else
            buffer << "\t" << "out[" << to_string(i) << "] = " << outputs[i] << ";" << endl;

    if(LSTM_number)
        buffer << "\n\t" << "lstm.time_step_counter += 1;" << endl;

    buffer << "\n\t" << "return out;" << endl
           << "}"  << endl
           << "\n" << endl
           << "int main(){ \n" << endl
           << "\t" << "vector<float> inputs(" << to_string(inputs.size()) << "); \n" << endl;

    for(int i = 0; i < inputs.dimension(0); i++)
    {
        if(inputs[i].empty())
        {
            buffer << "\t" << "const float " << "input_" << to_string(i) <<" =" << " /*enter your value here*/; " << endl
                   << "\t" << "inputs[" << to_string(i) << "] = " << "input_" << to_string(i) << ";" << endl;
        }
        else
        {
            buffer << "\t" << "const float " << inputs[i] << " =" << " /*enter your value here*/; " << endl
                   << "\t" << "inputs[" << to_string(i) << "] = " << inputs[i] << ";" << endl;
        }
    }

    buffer << "" << endl;

    if(LSTM_number > 0)
    {
        buffer << "\t"   << "LSTMMemory lstm;" << "\n" << endl
               << "\t"   << "vector<float> outputs(" << outputs.size() <<");" << endl
               << "\n\t" << "outputs = calculate_outputs(inputs, lstm);" << endl;
    }
    else
    {
        buffer << "\t"   << "vector<float> outputs(" << outputs.size() <<");" << endl
               << "\n\t" << "outputs = calculate_outputs(inputs);" << endl;
    }

    buffer << "" << endl
           << "\t" << "printf(\"These are your outputs:\\n\");" << endl;

    for(int i = 0; i < outputs.dimension(0); i++)
    {
        if(outputs[i].empty())
            buffer << "\t" << "printf( \"output" << to_string(i) << ":" << " %f \\n\", "<< "outputs[" << to_string(i) << "]" << ");" << endl;
        else
            buffer << "\t" << "printf( \""<< output_names[i] << ":" << " %f \\n\", "<< "outputs[" << to_string(i) << "]" << ");" << endl;
    }

    buffer << "\n\t" << "return 0;" << endl
           << "} \n" << endl;

    const string out = buffer.str();
    //replace_all_appearances(out, "double double double", "double");
    //replace_all_appearances(out, "double double", "double");
    return out;
}


string NeuralNetwork::write_expression_api() const
{
    ostringstream buffer;
    Tensor<string, 1> found_tokens;
    Tensor<string, 1> inputs =  get_input_names();
    Tensor<string, 1> outputs = get_output_names();

    int LSTM_number = get_long_short_term_memory_layers_number();
    int cell_states_counter = 0;
    int hidden_state_counter = 0;

    bool logistic     = false;
    bool ReLU         = false;
    bool ExpLinear    = false;
    bool SExpLinear   = false;
    bool HSigmoid     = false;
    bool SoftPlus     = false;
    bool SoftSign     = false;

    buffer << "<!DOCTYPE html>" << endl
           << "<!--" << endl
           << "Artificial Intelligence Techniques SL\t" << endl
           << "artelnics@artelnics.com\t" << endl
           << "" << endl
           << "Your model has been exported to this php file." << endl
           << "You can manage it writting your parameters in the url of your browser.\t" << endl
           << "Example:" << endl
           << "" << endl
           << "\turl = http://localhost/API_example/\t" << endl
           << "\tparameters in the url = http://localhost/API_example/?num=5&num=2&...\t" << endl
           << "\tTo see the ouput refresh the page" << endl
           << "" << endl
           << "\tInputs Names: \t" << endl;

    Tensor<Tensor<string,1>, 1> inputs_outputs_buffer = fix_input_output_variables(inputs, outputs, buffer);

    for(Index i = 0; i < inputs_outputs_buffer(0).dimension(0);i++)
        inputs(i) = inputs_outputs_buffer(0)(i);

    for(Index i = 0; i < inputs_outputs_buffer(1).dimension(0);i++)
        outputs(i) = inputs_outputs_buffer(1)(i);

    buffer << inputs_outputs_buffer(2)[0]
        << "" << endl
        << "-->\t" << endl
        << "" << endl
        << "<html lang = \"en\">\n" << endl
        << "<head>\n" << endl
        << "<title>Rest API Client Side Demo</title>\n " << endl
        << "<meta charset = \"utf-8\">" << endl
        << "<meta name = \"viewport\" content = \"width=device-width, initial-scale=1\">" << endl
        << "<link rel = \"stylesheet\" href = \"https://maxcdn.bootstrapcdn.com/bootstrap/3.3.7/css/bootstrap.min.css\">" << endl
        << "<script src = \"https://ajax.googleapis.com/ajax/libs/jquery/3.2.0/jquery.min.js\"></script>" << endl
        << "<script src = \"https://maxcdn.bootstrapcdn.com/bootstrap/3.3.7/js/bootstrap.min.js\"></script>" << endl
        << "</head>" << endl
        << "<style>" << endl
        << ".btn{" << endl
        << "background-color: #7393B3" << endl // Gray
        << "border: none;" << endl
        << "color: white;" << endl
        << "padding: 15px 32px;" << endl
        << "text-align: center;" << endl
        << "font-size: 16px;" << endl
        << "}" << endl
        << "</style>" << endl
        << "<body>" << endl
        << "<div class = \"container\">" << endl
        << "<br></br>" << endl
        << "<div class = \"form-group\">" << endl
        << "<p>" << endl
        << "follow the steps defined in the \"index.php\" file" << endl
        << "</p>" << endl
        << "<p>" << endl
        << "Refresh the page to see the prediction" << endl
        << "</p>" << endl
        << "</div>" << endl
        << "<h4>" << endl
        << "<?php" << "\n" << endl;

    string token;
    string expression = write_expression();

    if(model_type == ModelType::AutoAssociation)
    {
    // Delete intermediate calculations

    // sample_autoassociation_distance
    {
        string word_to_delete = "sample_autoassociation_distance =";

        size_t index = expression.find(word_to_delete);

        if(index != string::npos)
            expression.erase(index, string::npos);
    }

    // sample_autoassociation_variables_distance
    {
        string word_to_delete = "sample_autoassociation_variables_distance =";

        size_t index = expression.find(word_to_delete);

        if(index != string::npos)
        {
            expression.erase(index, string::npos);
        }
    }
    }

    stringstream ss(expression);
    Tensor<string, 1> tokens;

    while(getline(ss, token, '\n'))
    {
        if(token.size() > 1 && token.back() == '{') break;
        if(token.size() > 1 && token.back() != ';') token += ';';

        if(token.size() < 2) continue;

        push_back_string(tokens, token);

    }

    string word;

    for(int i = 0; i < tokens.dimension(0); i++)
    {
        string t = tokens(i);
        word = get_word_from_token(t);

        if(word.size() > 1)
            push_back_string(found_tokens, word);
    }

    if(LSTM_number > 0)
    {
        for(int i = 0; i < found_tokens.dimension(0); i++)
        {
            const string t = found_tokens(i);

            if(token.find("cell_state") == 0)
                cell_states_counter += 1;

            if(token.find("hidden_state") == 0)
                hidden_state_counter += 1;
        }

        buffer << "class NeuralNetwork{" << endl
               << "public $time_steps = 3;" << endl
               << "public $time_step_counter = 1;" << endl;

        for(int i = 0; i < hidden_state_counter; i++)
            buffer << "public $" << "hidden_state_" << to_string(i) << " = type(0);" << endl;

        for(int i = 0; i < cell_states_counter; i++)
            buffer << "public $" << "cell_states_" << to_string(i) << " = type(0);" << endl;

        buffer << "}" << endl
               << "$nn = new NeuralNetwork;" << endl;
    }

    buffer << "session_start();" << endl
           << "if(isset($_SESSION['lastpage']) && $_SESSION['lastpage'] == __FILE__) { " << endl
           << "if(isset($_SERVER['HTTPS']) && $_SERVER['HTTPS'] === 'on') " << endl
           << "\t$url = \"https://\"; " << endl
           << "else" << endl
           << "\t$url = \"http://\"; " << endl
           << "\n" << endl
           << "$url.= $_SERVER['HTTP_HOST'];" << endl
           << "$url.= $_SERVER['REQUEST_URI'];" << endl
           << "$url_components = parse_url($url);" << endl
           << "parse_str($url_components['query'], $params);" << endl
           << "\n" << endl;

    for(int i = 0; i < inputs.dimension(0); i++)
    {
        if(inputs[i].empty())
        {
            buffer << "$num"    + to_string(i) << " = " << "$params['num" + to_string(i) << "'];" << endl
                   << "$input_" + to_string(i) << " = intval(" << "$num"  + to_string(i) << ");"  << endl;
        }
        else
        {
            buffer << "$num" + to_string(i) << " = " << "$params['num" + to_string(i) << "'];" << endl
                   << "$" << inputs[i]      << " = intval(" << "$num"  + to_string(i) << ");"  << endl;
        }
    }

    buffer << "if(" << endl;

    for(int i = 0; i < inputs.dimension(0); i++)
        if(i != inputs.dimension(0)-1)
            buffer << "is_numeric(" << "$" << "num" + to_string(i) << ") &&" << endl;
        else
            buffer << "is_numeric(" << "$" << "num" + to_string(i) << "))" << endl;

    buffer << "{" << endl
           << "$status=200;" << endl
           << "$status_msg = 'valid parameters';" << endl
           << "}" << endl
           << "else" << endl
           << "{" << endl
           << "$status =400;" << endl
           << "$status_msg = 'invalid parameters';" << endl
           << "}"   << endl;

    if(LSTM_number>0)
    {
        buffer << "if($nn->time_step_counter % $nn->current_combinations_derivatives === 0 ){" << endl
               << "$nn->current_combinations_derivatives = 3;" << endl
               << "$nn->time_step_counter = 1;" << endl;

        for(int i = 0; i < hidden_state_counter; i++)
            buffer << "$nn->" << "hidden_state_" << to_string(i) << " = type(0);" << endl;

        for(int i = 0; i < cell_states_counter; i++)
            buffer << "$nn->" << "cell_states_" << to_string(i) << " = type(0);" << endl;

        buffer << "}" << endl;
    }

    buffer << "\n" << endl;

    string target_string0("Logistic");
    string target_string1("ReLU");
    string target_string4("ExponentialLinear");
    string target_string5("SELU");
    string target_string6("HardSigmoid");
    string target_string7("SoftPlus");
    string target_string8("SoftSign");

    size_t substring_length0;
    size_t substring_length1;
    size_t substring_length2;
    size_t substring_length3;
    size_t substring_length4;
    size_t substring_length5;
    size_t substring_length6;
    size_t substring_length7;
    size_t substring_length8;

    string new_word;

    Tensor<string, 1> found_tokens_and_input_names = concatenate_string_tensors(inputs, found_tokens);
    found_tokens_and_input_names = sort_string_tensor(found_tokens_and_input_names);

    for(int i = 0; i < tokens.dimension(0); i++)
    {
        string t = tokens(i);

        substring_length0 = t.find(target_string0);
        substring_length1 = t.find(target_string1);
        substring_length4 = t.find(target_string4);
        substring_length5 = t.find(target_string5);
        substring_length6 = t.find(target_string6);
        substring_length7 = t.find(target_string7);
        substring_length8 = t.find(target_string8);

        if(substring_length0 < t.size() && substring_length0!=0){ logistic     = true; }
        if(substring_length1 < t.size() && substring_length1!=0){ ReLU         = true; }
        if(substring_length4 < t.size() && substring_length4!=0){ ExpLinear    = true; }
        if(substring_length5 < t.size() && substring_length5!=0){ SExpLinear   = true; }
        if(substring_length6 < t.size() && substring_length6!=0){ HSigmoid     = true; }
        if(substring_length7 < t.size() && substring_length7!=0){ SoftPlus     = true; }
        if(substring_length8 < t.size() && substring_length8!=0){ SoftSign     = true; }

        for(int i = 0; i < found_tokens_and_input_names.dimension(0); i++)
        {
            new_word.clear();

            new_word = "$" + found_tokens_and_input_names[i];

            replace_all_word_appearances(t, found_tokens_and_input_names[i], new_word);
        }

        if(LSTM_number > 0)
        {
            replace_all_appearances(t, "(t)"     , "");
            replace_all_appearances(t, "(t-1)"   , "");
            replace_all_appearances(t, "hidden_" , "$hidden_");
            replace_all_appearances(t, "cell_"   , "$cell_");
            replace_all_appearances(t, "$hidden_", "$nn->hidden_");
            replace_all_appearances(t, "$cell_"  , "$nn->cell_");
        }

        buffer << t << endl;
    }

    const Tensor<string, 1> fixed_outputs = fix_write_expression_outputs(expression, outputs, "php");

    for(int i = 0; i < fixed_outputs.dimension(0); i++)
        buffer << fixed_outputs(i) << endl;

    buffer << "if($status === 200){" << endl
           << "$response = ['status' => $status,  'status_message' => $status_msg" << endl;

    for(int i = 0; i < outputs.dimension(0); i++)
        buffer << ", '" << outputs(i) << "' => " << "$" << outputs[i] << endl;

    buffer << "];" << endl
           << "}" << endl
           << "else" << endl
           << "{" << endl
           << "$response = ['status' => $status,  'status_message' => $status_msg" << "];" << endl
           << "}" << endl;

    if(LSTM_number>0)
        buffer << "$nn->time_step_counter += 1;" << endl;

    buffer << "\n" << endl
           << "$json_response_pretty = json_encode($response, JSON_PRETTY_PRINT);" << endl
           << "echo nl2br(\"\\n\" . $json_response_pretty . \"\\n\");" << endl
           << "}else{" << endl
           << "echo \"New page\";" << endl
           << "}" << endl
           << "$_SESSION['lastpage'] = __FILE__;" << endl
           << "?>" << endl
           << "\n" << endl;

    if(logistic)
    {
        buffer << "<?php" << endl
               << "function Logistic(int $x) {" << endl
               << "$z = 1/(1+exp(-$x));" << endl
               << "return $z;" << endl
               << "}" << endl
               << "?>" << endl
               << "\n" << endl;
    }

    if(ReLU)
    {
        buffer << "<?php" << endl
               << "function ReLU(int $x) {" << endl
               << "$z = max(0, $x);" << endl
               << "return $z;" << endl
               << "}" << endl
               << "?>" << endl
               << "\n" << endl;
    }

    if(ExpLinear)
    {
        buffer << "<?php" << endl
               << "function ExponentialLinear(int $x) {" << endl
               << "$alpha = 1.6732632423543772848170429916717;" << endl
               << "if($x>0){" << endl
               << "$z=$x;" << endl
               << "}else{" << endl
               << "$z=$alpha*(exp($x)-1);" << endl
               << "}" << endl
               << "return $z;" << endl
               << "}" << endl
               << "?>" << endl
               << "\n" << endl;
    }

    if(SExpLinear)
    {
        buffer << "<?php" << endl
               << "function SELU(int $x) {" << endl
               << "$alpha  = 1.67326;" << endl
               << "$lambda = 1.05070;" << endl
               << "if($x>0){" << endl
               << "$z=$lambda*$x;" << endl
               << "}else{" << endl
               << "$z=$lambda*$alpha*(exp($x)-1);" << endl
               << "}" << endl
               << "return $z;" << endl
               << "}" << endl
               << "?>" << endl
               << "\n" << endl;
    }

    if(HSigmoid)
    {
        buffer << "<?php" << endl
               << "function HardSigmoid(int $x) {" << endl
               << "$z=1/(1+exp(-$x));" << endl
               << "return $z;" << endl
               << "}" << endl
               << "?>" << endl
               << "\n" << endl;
    }

    if(SoftPlus)
    {
        buffer << "<?php" << endl
               << "function SoftPlus(int $x) {" << endl
               << "$z=log(1+exp($x));" << endl
               << "return $z;" << endl
               << "}" << endl
               << "?>" << endl
               << "\n" << endl;
    }

    if(SoftSign)
    {
        buffer << "<?php" << endl
               << "function SoftSign(int $x) {" << endl
               << "$z=$x/(1+abs($x));" << endl
               << "return $z;" << endl
               << "}" << endl
               << "?>" << endl
               << "\n" << endl;
    }

    buffer << "</h4>" << endl
           << "</div>" << endl
           << "</body>" << endl
           << "</html>" << endl;

    string out = buffer.str();

    replace_all_appearances(out, "$$", "$");
    replace_all_appearances(out, "_$", "_");

    return out;
}


string NeuralNetwork::write_expression_javascript() const
{
    Tensor<string, 1> tokens;
    Tensor<string, 1> found_tokens;
    Tensor<string, 1> found_mathematical_expressions;
    Tensor<string, 1> inputs =  get_input_names();
    Tensor<string, 1> outputs = get_output_names();

    ostringstream buffer_to_fix;

    string token;
    string expression = write_expression();

    const int maximum_output_variable_numbers = 5;

    if(model_type == ModelType::AutoAssociation)
    {
    // Delete intermediate calculations

    // sample_autoassociation_distance
    {
        string word_to_delete = "sample_autoassociation_distance =";

        size_t index = expression.find(word_to_delete);

        if(index != string::npos)
            expression.erase(index, string::npos);
    }

    // sample_autoassociation_variables_distance
    {
        string word_to_delete = "sample_autoassociation_variables_distance =";

        size_t index = expression.find(word_to_delete);

        if(index != string::npos)
            expression.erase(index, string::npos);
    }
    }

    stringstream ss(expression);

    int cell_states_counter = 0;
    int hidden_state_counter = 0;
    int LSTM_number = get_long_short_term_memory_layers_number();

    bool logistic     = false;
    bool ReLU         = false;
    bool ExpLinear    = false;
    bool SExpLinear   = false;
    bool HSigmoid     = false;
    bool SoftPlus     = false;
    bool SoftSign     = false;

    buffer_to_fix << "<!--" << endl
                  << "Artificial Intelligence Techniques SL\t" << endl
                  << "artelnics@artelnics.com\t" << endl
                  << "" << endl
                  << "Your model has been exported to this JavaScript file." << endl
                  << "You can manage it with the main method, where you \t" << endl
                  << "can change the values of your inputs. For example:" << endl
                  << "" << endl
                  << "if we want to add these 3 values (0.3, 2.5 and 1.8)" << endl
                  << "to our 3 inputs (Input_1, Input_2 and Input_1), the" << endl
                  << "main program has to look like this:" << endl
                  << "\t" << endl
                  << "int neuralNetwork(){ " << endl
                  << "\t" << "vector<float> inputs(3);"<< endl
                  << "\t" << endl
                  << "\t" << "const float asdas  = 0.3;" << endl
                  << "\t" << "inputs[0] = asdas;"        << endl
                  << "\t" << "const float input2 = 2.5;" << endl
                  << "\t" << "inputs[1] = input2;"       << endl
                  << "\t" << "const float input3 = 1.8;" << endl
                  << "\t" << "inputs[2] = input3;"       << endl
                  << "\t" << ". . ." << endl
                  << "\n" << endl
                  << "Inputs Names:" <<endl;
     
     Tensor<Tensor<string,1>, 1> inputs_outputs_buffer = fix_input_output_variables(inputs, outputs, buffer_to_fix);

    for(Index i = 0; i < inputs_outputs_buffer(0).dimension(0);i++)
        inputs(i) = inputs_outputs_buffer(0)(i);

    for(Index i = 0; i < inputs_outputs_buffer(1).dimension(0);i++)
        outputs(i) = inputs_outputs_buffer(1)(i);

    ostringstream buffer;

    buffer << inputs_outputs_buffer(2)[0]
           << "-->" << endl
           << "\n" << endl
           << "<!DOCTYPE HTML>" << endl
           << "<html lang=\"en\">" << endl
           << "\n" << endl
           << "<head>" << endl
           << "<link href=\"https://www.neuraldesigner.com/assets/css/neuraldesigner.css\" rel=\"stylesheet\" />" << endl
           << "<link href=\"https://www.neuraldesigner.com/images/fav.ico\" rel=\"shortcut icon\" type=\"image/x-icon\" />" << endl
           << "</head>" << endl
           << "\n" << endl
           << "<style>" << endl
           << "" << endl
           << "body {" << endl
           << "display: flex;" << endl
           << "justify-content: center;" << endl
           << "align-items: center;" << endl
           << "min-height: 100vh;" << endl
           << "margin: 0;" << endl
           << "background-color: #f0f0f0;" << endl
           << "font-family: Arial, sans-serif;" << endl
           << "}" << endl
           << "" << endl
           << ".form {" << endl
           << "border-collapse: collapse;" << endl
           << "width: 80%; " << endl
           << "max-width: 600px; " << endl
           << "margin: 0 auto; " << endl
           << "background-color: #fff; " << endl
           << "box-shadow: 0px 0px 10px rgba(0, 0, 0, 0.1); " << endl
           << "border: 1px solid #777; " << endl
           << "border-radius: 5px; " << endl
           << "}" << endl
           << "" << endl
           << "input[type=\"number\"] {" << endl
           << "width: 60px; " << endl
           << "text-align: center; " << endl
           << "}" << endl
           << "" << endl
           << ".form th," << endl
           << ".form td {" << endl
           << "padding: 10px;" << endl
           << "text-align: center;" << endl
           << "font-family: \"Helvetica Neue\", Helvetica, Arial, sans-serif; " << endl
           << "}" << endl
           << "" << endl
           << "" << endl
           << ".btn {" << endl
           << "background-color: #5da9e9;" << endl
           << "border: none;" << endl
           << "color: white;" << endl
           << "text-align: center;" << endl
           << "font-size: 16px;" << endl
           << "margin: 4px;" << endl
           << "cursor: pointer;" << endl
           << "padding: 10px 20px;" << endl
           << "border-radius: 5px;" << endl
           << "transition: background-color 0.3s ease;" << endl
           << "font-family: \"Helvetica Neue\", Helvetica, Arial, sans-serif;" << endl
           << "}" << endl
           << "" << endl
           << "" << endl
           << ".btn:hover {" << endl
           << "background-color: #4b92d3; " << endl
           << "}" << endl
           << "" << endl
           << "" << endl
           << "input[type=\"range\"]::-webkit-slider-runnable-track {" << endl
           << "background: #5da9e9;" << endl
           << "height: 0.5rem;" << endl
           << "}" << endl
           << "" << endl
           << "" << endl
           << "input[type=\"range\"]::-moz-range-track {" << endl
           << "background: #5da9e9;" << endl
           << "height: 0.5rem;" << endl
           << "}" << endl
           << "" << endl
           << "" << endl
           << ".tabla {" << endl
           << "width: 100%;" << endl
           << "padding: 5px;" << endl
           << "margin: 0; " << endl
           << "}" << endl
           << "" << endl
           << "" << endl
           << ".form th {" << endl
           << "background-color: #f2f2f2;" << endl
           << "font-family: \"Helvetica Neue\", Helvetica, Arial, sans-serif;" << endl
           << "}" << endl
           << "</style>" << endl
           << "\n" << endl
           << "<body>" << endl
           << "\n" << endl
           << "<section>" << endl
           << "<br/>" << endl
           << "\n" << endl
           << "<div align=\"center\" style=\"display:block;text-align: center;\">" << endl
           << "<!-- MENU OPTIONS HERE  -->" << endl
           << "<form style=\"display: inline-block;margin-left: auto; margin-right: auto;\">" << endl
           << "\n" << endl
           << "<table border=\"1px\" class=\"form\">" << endl
           << "\n" << endl
           << "INPUTS" << endl;

    if(has_scaling_layer_2d())
    {
        const Tensor<Descriptives, 1> inputs_descriptives = get_scaling_layer_2d()->get_descriptives();

        for(int i = 0; i < inputs.dimension(0); i++)
        {
            buffer << "<!-- "<< to_string(i) <<"scaling layer -->" << endl
                   << "<tr style=\"height:3.5em\">" << endl
                   << "<td> " << inputs_name[i] << " </td>" << endl
                   << "<td style=\"text-align:center\">" << endl
                   << "<input type=\"range\" id=\"" << inputs[i] << "\" value=\"" << (inputs_descriptives(i).minimum + inputs_descriptives(i).maximum)/2 << "\" min=\"" << inputs_descriptives(i).minimum << "\" max=\"" << inputs_descriptives(i).maximum << "\" step=\"" << (inputs_descriptives(i).maximum - inputs_descriptives(i).minimum)/type(100) << "\" onchange=\"updateTextInput1(this.value, '" << inputs[i] << "_text')\" />" << endl
                   << "<input class=\"tabla\" type=\"number\" id=\"" << inputs[i] << "_text\" value=\"" << (inputs_descriptives(i).minimum + inputs_descriptives(i).maximum)/2 << "\" min=\"" << inputs_descriptives(i).minimum << "\" max=\"" << inputs_descriptives(i).maximum << "\" step=\"" << (inputs_descriptives(i).maximum - inputs_descriptives(i).minimum)/type(100) << "\" onchange=\"updateTextInput1(this.value, '" << inputs[i] << "')\">" << endl
                   << "</td>" << endl
                   << "</tr>" << endl
                   << "\n" << endl;
        }
    }
    else
    {
        for(int i = 0; i < inputs.dimension(0); i++)
        {
            buffer << "<!-- "<< to_string(i) <<"no scaling layer -->" << endl
                   << "<tr style=\"height:3.5em\">" << endl
                   << "<td> " << inputs_name[i] << " </td>" << endl
                   << "<td style=\"text-align:center\">" << endl
                   << "<input type=\"range\" id=\"" << inputs[i] << "\" value=\"0\" min=\"-1\" max=\"1\" step=\"0.01\" onchange=\"updateTextInput1(this.value, '" << inputs[i] << "_text')\" />" << endl
                   << "<input class=\"tabla\" type=\"number\" id=\"" << inputs[i] << "_text\" value=\"0\" min=\"-1\" max=\"1\" step=\"0.01\" onchange=\"updateTextInput1(this.value, '" << inputs[i] << "')\">" << endl
                   << "</td>" << endl
                   << "</tr>" << endl
                   << "\n" << endl;
        }
    }

    buffer << "</table>" << endl
           << "</form>" << endl
           << "\n" << endl;

    if(outputs.dimension(0) > maximum_output_variable_numbers)
    {
        buffer << "<!-- HIDDEN INPUTS -->" << endl;

        for(int i = 0; i < outputs.dimension(0); i++)
            buffer << "<input type=\"hidden\" id=\"" << outputs[i] << "\" value=\"\">" << endl;

        buffer << "\n" << endl;
    }

    buffer << "<div align=\"center\">" << endl
           << "<!-- BUTTON HERE -->" << endl
           << "<button class=\"btn\" onclick=\"neuralNetwork()\">calculate outputs</button>" << endl
           << "</div>" << endl
           << "\n" << endl
           << "<br/>" << endl
           << "\n" << endl
           << "<table border=\"1px\" class=\"form\">" << endl
           << "OUTPUTS" << endl;

    if(outputs.dimension(0) > maximum_output_variable_numbers)
    {
        buffer << "<tr style=\"height:3.5em\">" << endl
               << "<td> Target </td>" << endl
               << "<td>" << endl
               << "<select id=\"category_select\" onchange=\"updateSelectedCategory()\">" << endl;

        for(int i = 0; i < outputs.dimension(0); i++)
            buffer << "<option value=\"" << outputs[i] << "\">" << output_names[i] << "</option>" << endl;

        buffer << "</select>" << endl
               << "</td>" << endl
               << "</tr>" << endl
               << "\n" << endl
               << "<tr style=\"height:3.5em\">" << endl
               << "<td> Value </td>" << endl
               << "<td>" << endl
               << "<input style=\"text-align:right; padding-right:20px;\" id=\"selected_value\" value=\"\" type=\"text\"  disabled/>" << endl
               << "</td>" << endl
               << "</tr>" << endl
               << "\n" << endl;
    }
    else
    {
        for(int i = 0; i < outputs.dimension(0); i++)
        {
            buffer << "<tr style=\"height:3.5em\">" << endl
                   << "<td> " << output_names[i] << " </td>" << endl
                   << "<td>" << endl
                   << "<input style=\"text-align:right; padding-right:20px;\" id=\"" << outputs[i] << "\" value=\"\" type=\"text\"  disabled/>" << endl
                   << "</td>" << endl
                   << "</tr>" << endl
                   << "\n" << endl;
        }
    }

    buffer << "</table>" << endl
           << "\n" << endl
           << "</form>" << endl
           << "</div>" << endl
           << "\n" << endl
           << "</section>" << endl
           << "\n" << endl
           << "<script>" << endl;

    if(outputs.dimension(0) > maximum_output_variable_numbers)
    {
        buffer << "function updateSelectedCategory() {" << endl
               << "\tvar selectedCategory = document.getElementById(\"category_select\").value;" << endl
               << "\tvar selectedValueElement = document.getElementById(\"selected_value\");" << endl;

        for(int i = 0; i < outputs.dimension(0); i++) 
        {
            buffer << "\tif(selectedCategory === \"" << outputs[i] << "\") {" << endl
                   << "\t\tselectedValueElement.value = document.getElementById(\"" << outputs[i] << "\").value;" << endl
                   << "\t}" << endl;
        }

        buffer << "}" << endl
               << "\n" << endl;
    }

    buffer << "function neuralNetwork()" << endl
           << "{" << endl
           << "\t" << "var inputs = [];" << endl;

    for(int i = 0; i < inputs.dimension(0); i++)
        buffer << "\t" << "var " << inputs[i] << " =" << " document.getElementById(\"" << inputs[i] << "\").value; " << endl
               << "\t" << "inputs.push(" << inputs[i] << ");" << endl;

    buffer << "\n" << "\t" << "var outputs = calculate_outputs(inputs); " << endl;

    for(int i = 0; i < outputs.dimension(0); i++)
        buffer << "\t" << "var " << outputs[i] << " = document.getElementById(\"" << outputs[i] << "\");" << endl
               << "\t" << outputs[i] << ".value = outputs[" << to_string(i) << "].toFixed(4);" << endl;

    if(outputs.dimension(0) > maximum_output_variable_numbers)
        buffer << "\t" << "updateSelectedCategory();" << endl;
    //else
    //{
    //    for(int i = 0; i < outputs.dimension(0); i++)
    //    {
    //        buffer << "\t" << "var " << outputs[i] << " = document.getElementById(\"" << outputs[i] << "\");" << endl;
    //        buffer << "\t" << outputs[i] << ".value = outputs[" << to_string(i) << "].toFixed(4);" << endl;
    //    }
    //}

    buffer << "\t" << "update_LSTM();" << endl
           << "}" << "\n" << endl;

    while(getline(ss, token, '\n'))
    {
        if(token.size() > 1 && token.back() == '{')
            break; 

        if(token.size() > 1 && token.back() != ';')
            token += ';'; 

        push_back_string(tokens, token);
    }

    buffer << "function calculate_outputs(inputs)" << endl
           << "{" << endl;

    for(int i = 0; i < inputs.dimension(0); i++)
        buffer << "\t" << "var " << inputs[i] << " = " << "+inputs[" << to_string(i) << "];" << endl;

    buffer << "" << endl;

    for(int i = 0; i < tokens.dimension(0); i++)
    {
        const string word = get_word_from_token(tokens(i));

        if(word.size() > 1)
            push_back_string(found_tokens, word);
    }

    if(LSTM_number > 0)
    {
        for(int i = 0; i < found_tokens.dimension(0); i++)
        {
            token = found_tokens(i);

            if(token.find("cell_state") == 0)
                cell_states_counter += 1;

            if(token.find("hidden_state") == 0)
                hidden_state_counter += 1;
        }

        buffer << "\t" << "if(time_step_counter % current_combinations_derivatives == 0 ){" << endl
               << "\t\t" << "time_step_counter = 1" << endl;

        for(int i = 0; i < hidden_state_counter; i++)
            buffer << "\t\t" << "hidden_state_" << to_string(i) << " = 0" << endl;

        for(int i = 0; i < cell_states_counter; i++)
            buffer << "\t\t" << "cell_states_" << to_string(i) << " = 0" << endl;

        buffer << "\t}\n" << endl;
    }

    string target_string_0("Logistic");
    string target_string_1("ReLU");
    string target_string_4("ExponentialLinear");
    string target_string_5("SELU");
    string target_string_6("HardSigmoid");
    string target_string_7("SoftPlus");
    string target_string_8("SoftSign");

    string sufix = "Math.";

    push_back_string(found_mathematical_expressions, "exp");
    push_back_string(found_mathematical_expressions, "tanh");
    push_back_string(found_mathematical_expressions, "max");
    push_back_string(found_mathematical_expressions, "min");

    for(int i = 0; i < tokens.dimension(0); i++)
    {
        string t = tokens(i);

        const size_t substring_length_0 = t.find(target_string_0);
        const size_t substring_length_1 = t.find(target_string_1);
        const size_t substring_length_4 = t.find(target_string_4);
        const size_t substring_length_5 = t.find(target_string_5);
        const size_t substring_length_6 = t.find(target_string_6);
        const size_t substring_length_7 = t.find(target_string_7);
        const size_t substring_length_8 = t.find(target_string_8);

        if(substring_length_1 < t.size() && substring_length_1!=0){ ReLU = true; }
        if(substring_length_0 < t.size() && substring_length_0!=0){ logistic = true; }
        if(substring_length_6 < t.size() && substring_length_6!=0){ HSigmoid = true; }
        if(substring_length_7 < t.size() && substring_length_7!=0){ SoftPlus = true; }
        if(substring_length_8 < t.size() && substring_length_8!=0){ SoftSign = true; }
        if(substring_length_4 < t.size() && substring_length_4!=0){ ExpLinear = true; }
        if(substring_length_5 < t.size() && substring_length_5!=0){ SExpLinear = true; }

        for(int i = 0; i < found_mathematical_expressions.dimension(0); i++)
        {
            string key_word = found_mathematical_expressions(i);
            string new_word;

            new_word = sufix + key_word;
            replace_all_appearances(t, key_word, new_word);
        }

        if(t.size() <= 1)
            buffer << "" << endl;
        else
            buffer << "\t" << "var " << t << endl;
    }

    if(LSTM_number>0)
        buffer << "\t" << "time_step_counter += 1" << "\n" << endl;

    const Tensor<string, 1> fixed_outputs = fix_write_expression_outputs(expression, outputs, "javascript");

    for(int i = 0; i < fixed_outputs.dimension(0); i++)
        buffer << fixed_outputs(i) << endl;

    buffer << "\t" << "var out = [];" << endl;

    for(int i = 0; i < outputs.dimension(0); i++)
        buffer << "\t" << "out.push(" << outputs[i] << ");" << endl;

    buffer << "\n\t" << "return out;" << endl
           << "}" << "\n" << endl;

    if(LSTM_number > 0)
    {
        buffer << "\t" << "var steps = 3;            " << endl
               << "\t" << "var current_combinations_derivatives = steps;   " << endl
               << "\t" << "var time_step_counter = 1;" << endl;

        for(int i = 0; i < hidden_state_counter; i++)
            buffer << "\t" << "var " << "var hidden_state_" << to_string(i) << " = 0" << endl;

        for(int i = 0; i < cell_states_counter; i++)
            buffer << "\t" << "var " << "var cell_states_" << to_string(i) << " = 0" << endl;

        buffer << "\n" << endl;
    }

    if(logistic)
    {
        buffer << "function Logistic(x) {" << endl
               << "\tvar z = 1/(1+Math.exp(x));" << endl
               << "\treturn z;" << endl
               << "}" << endl
               << "\n" << endl;
    }

    if(ReLU)
    {
        buffer << "function ReLU(x) {" << endl
               << "\tvar z = Math.max(0, x);" << endl
               << "\treturn z;" << endl
               << "}" << endl
               << "\n" << endl;
    }

    if(ExpLinear)
    {
        buffer << "function ExponentialLinear(x) {" << endl
               << "\tvar alpha = 1.67326;" << endl
               << "\tif(x>0){" << endl
               << "\t\tvar z = x;" << endl
               << "\t}else{" << endl
               << "\t\tvar z = alpha*(Math.exp(x)-1);" << endl
               << "\t}" << endl
               << "\treturn z;" << endl
               << "}" << endl
               << "\n" << endl;
    }

    if(SExpLinear)
    {
        buffer << "function SELU(x) {" << endl
               << "\tvar alpha  = 1.67326;" << endl
               << "\tvar lambda = 1.05070;" << endl
               << "\tif(x>0){" << endl
               << "\t\tvar z = lambda*x;" << endl
               << "\t}else{" << endl
               << "\t\tvar z = lambda*alpha*(Math.exp(x)-1);" << endl
               << "\t}" << endl
               << "return z;" << endl
               << "}" << endl
               << "\n" << endl;
    }

    if(HSigmoid)
    {
        buffer << "function HardSigmoid(x) {" << endl
               << "\tvar z=1/(1+Math.exp(-x));" << endl
               << "\treturn z;" << endl
               << "}" << endl
               << "\n" << endl;
    }

    if(SoftPlus)
    {
        buffer << "function SoftPlus(int x) {" << endl
               << "\tvar z=log(1+Math.exp(x));" << endl
               << "\treturn z;" << endl
               << "}" << endl
               << "\n" << endl;
    }

    if(SoftSign)
    {
        buffer << "function SoftSign(x) {" << endl
               << "\tvar z=x/(1+Math.abs(x));" << endl
               << "\treturn z;" << endl
               << "}" << endl
               << "\n" << endl;
    }

    buffer << "function updateTextInput1(val, id)" << endl
           << "{" << endl
           << "\t"<< "document.getElementById(id).value = val;" << endl
           << "}" << endl
           << "\n" << endl
           << "window.onresize = showDiv;" << endl
           << "\n" << endl
           << "</script>" << endl
           << "\n" << endl
           << "<!--script src=\"https://www.neuraldesigner.com/app/htmlparts/footer.js\"></script-->" << endl
           << "\n" << endl
           << "</body>" << endl
           << "\n" << endl
           << "</html>" << endl;

    string out = buffer.str();

    if(LSTM_number>0)
    {
        replace_all_appearances(out, "(t)", "");
        replace_all_appearances(out, "(t-1)", "");
        replace_all_appearances(out, "var cell_state"  , "cell_state");
        replace_all_appearances(out, "var hidden_state", "hidden_state");
    }

    return out;
}


string NeuralNetwork::write_expression_python() const
{
    ostringstream buffer;

    Tensor<string, 1> found_tokens;
    Tensor<string, 1> found_mathematical_expressions;

    Tensor<string, 1> inputs =  get_input_names();
    Tensor<string, 1> original_inputs =  get_input_names();
    Tensor<string, 1> outputs = get_output_names();

//    const Index layers_number = get_layers_number();

    int LSTM_number = get_long_short_term_memory_layers_number();
    int cell_states_counter = 0;
    int hidden_state_counter = 0;

    bool logistic     = false;
    bool ReLU         = false;
    bool ExpLinear    = false;
    bool SExpLinear   = false;
    bool HSigmoid     = false;
    bool SoftPlus     = false;
    bool SoftSign     = false;

    buffer << "\'\'\' " << endl
           << "Artificial Intelligence Techniques SL\t" << endl
           << "artelnics@artelnics.com\t" << endl
           << "" << endl
           << "Your model has been exported to this python file."  << endl
           << "You can manage it with the 'NeuralNetwork' class.\t" << endl
           << "Example:" << endl
           << "" << endl
           << "\tmodel = NeuralNetwork()\t" << endl
           << "\tsample = [input_1, input_2, input_3, input_4, ...]\t" << endl
           << "\toutputs = model.calculate_outputs(sample)" << endl
           << "\n" << endl
           << "Inputs Names: \t" << endl;

    Tensor<Tensor<string,1>, 1> inputs_outputs_buffer = fix_input_output_variables(inputs, outputs, buffer);

    for(Index i = 0; i < inputs_outputs_buffer(0).dimension(0);i++)
    {
        inputs(i) = inputs_outputs_buffer(0)(i);
        buffer << "\t" << i << ") " << inputs(i) << endl;
    }

    for(Index i = 0; i < inputs_outputs_buffer(1).dimension(0);i++)
        outputs(i) = inputs_outputs_buffer(1)(i);

    buffer << "\n" << endl
           << "You can predict with a batch of samples using calculate_batch_output method\t" << endl
           << "IMPORTANT: input batch must be <class 'numpy.ndarray'> type\t" << endl
           << "Example_1:\t" << endl
           << "\tmodel = NeuralNetwork()\t" << endl
           << "\tinput_batch = np.array([[1, 2], [4, 5]])\t" << endl
           << "\toutputs = model.calculate_batch_output(input_batch)" << endl
           << "Example_2:\t" << endl
           << "\tinput_batch = pd.DataFrame( {'col1': [1, 2], 'col2': [3, 4]})\t" << endl
           << "\toutputs = model.calculate_batch_output(input_batch.values)" << endl
           << "\'\'\' " << endl
           << "\n" << endl;

    Tensor<string, 1> tokens;

    string expression = write_expression();
    string token;

    stringstream ss(expression);

    while(getline(ss, token, '\n'))
    {
        if(token.size() > 1 && token.back() == '{'){ break; }
        if(token.size() > 1 && token.back() != ';'){ token += ';'; }

        push_back_string(tokens, token);
    }

    const string target_string0("Logistic");
    const string target_string1("ReLU");
    const string target_string4("ExponentialLinear");
    const string target_string5("SELU");
    const string target_string6("HardSigmoid");
    const string target_string7("SoftPlus");
    const string target_string8("SoftSign");

    for(int i = 0; i < tokens.dimension(0); i++)
    {
        string word;
        string t = tokens(i);

        const size_t substring_length0 = t.find(target_string0);
        const size_t substring_length1 = t.find(target_string1);
        const size_t substring_length4 = t.find(target_string4);
        const size_t substring_length5 = t.find(target_string5);
        const size_t substring_length6 = t.find(target_string6);
        const size_t substring_length7 = t.find(target_string7);
        const size_t substring_length8 = t.find(target_string8);

        if(substring_length0 < t.size() && substring_length0!=0){ logistic = true; }
        if(substring_length1 < t.size() && substring_length1!=0){ ReLU = true; }
        if(substring_length4 < t.size() && substring_length4!=0){ ExpLinear = true; }
        if(substring_length5 < t.size() && substring_length5!=0){ SExpLinear = true; }
        if(substring_length6 < t.size() && substring_length6!=0){ HSigmoid = true; }
        if(substring_length7 < t.size() && substring_length7!=0){ SoftPlus = true; }
        if(substring_length8 < t.size() && substring_length8!=0){ SoftSign = true; }

        word = get_word_from_token(t);

        if(word.size() > 1)
            push_back_string(found_tokens, word);
    }

    for(int i = 0; i< found_tokens.dimension(0); i++)
    {
        const string token = found_tokens(i);

        if(token.find("cell_state") == 0)
            cell_states_counter += 1;

        if(token.find("hidden_state") == 0)
            hidden_state_counter += 1;
    }

    buffer << "import numpy as np" << endl
           << "\n" << endl;
/*
    if(model_type == ModelType::AutoAssociation)
    {
        buffer << "def calculate_distances(input, output):" << endl;
        buffer << "\t" << "return (np.linalg.norm(np.array(input)-np.array(output)))/len(input)" << endl;

        buffer << "\n" << endl;

        buffer << "def calculate_variables_distances(input, output):" << endl;
        buffer << "\t" << "length_vector = len(input)" << endl;
        buffer << "\t" << "variables_distances = [None] * length_vector" << endl;
        buffer << "\t" << "for i in range(length_vector):" << endl;
        buffer << "\t\t" << "variables_distances[i] = (np.linalg.norm(np.array(input[i])-np.array(output[i])))" << endl;
        buffer << "\t" << "return variables_distances" << endl;

        buffer << "\n" << endl;
    }
*/
    buffer << "class NeuralNetwork:" << endl;
/*
    if(model_type == ModelType::AutoAssociation)
    {
        buffer << "\t" << "minimum = " << to_string(distances_descriptives.minimum) << endl;
        buffer << "\t" << "first_quartile = " << to_string(auto_associative_distances_box_plot.first_quartile) << endl;
        buffer << "\t" << "median = " << to_string(auto_associative_distances_box_plot.median) << endl;
        buffer << "\t" << "mean = " << to_string(distances_descriptives.mean) << endl;
        buffer << "\t" << "third_quartile = "  << to_string(auto_associative_distances_box_plot.third_quartile) << endl;
        buffer << "\t" << "maximum = " << to_string(distances_descriptives.maximum) << endl;
        buffer << "\t" << "standard_deviation = " << to_string(distances_descriptives.standard_deviation) << endl;
        buffer << "\n" << endl;
    }
*/
    if(LSTM_number > 0)
    {
        buffer << "\t" << "def __init__(self, ts = 1):" << endl
               << "\t\t" << "self.inputs_number = " << to_string(inputs.size()) << endl
               << "\t\t" << "self.current_combinations_derivatives = ts" << endl;

        for(int i = 0; i < hidden_state_counter; i++)
            buffer << "\t\t" << "self.hidden_state_" << to_string(i) << " = 0" << endl;

        for(int i = 0; i < cell_states_counter; i++)
            buffer << "\t\t" << "self.cell_states_" << to_string(i) << " = 0" << endl;

        buffer << "\t\t" << "self.time_step_counter = 1" << endl;
    }
    else
    {
        string inputs_list;

        for(int i = 0; i < original_inputs.size();i++)
        {
            inputs_list += "'" + original_inputs(i) + "'";

            if(i < original_inputs.size() - 1)
                inputs_list += ", ";
        }

        buffer << "\t" << "def __init__(self):" << endl
               << "\t\t" << "self.inputs_number = " << to_string(inputs.size()) << endl
               << "\t\t" << "self.inputs_name = [" << inputs_list << "]" << endl;
    }

    buffer << "\n" << endl;

    if(logistic)
    {
        buffer << "\tdef Logistic (x):" << endl
               << "\t\t" << "z = 1/(1+np.exp(-x))" << endl
               << "\t\t" << "return z" << endl
               << "\n" << endl;
    }

    if(ReLU)
    {
        buffer << "\tdef ReLU (x):" << endl
               << "\t\t" << "z = max(0, x)" << endl
               << "\t\t" << "return z" << endl
               << "\n" << endl;
    }

    if(ExpLinear)
    {
        buffer << "\tdef ExponentialLinear (x):" << endl
               << "\t\t"   << "float alpha = 1.67326" << endl
               << "\t\t"   << "if(x>0):" << endl
               << "\t\t\t" << "z = x" << endl
               << "\t\t"   << "else:" << endl
               << "\t\t\t" << "z = alpha*(np.exp(x)-1)" << endl
               << "\t\t"   << "return z" << endl
               << "\n" << endl;
    }

    if(SExpLinear)
    {
        buffer << "\tdef SELU (x):" << endl
               << "\t\t"   << "float alpha = 1.67326" << endl
               << "\t\t"   << "float lambda = 1.05070" << endl
               << "\t\t"   << "if(x>0):" << endl
               << "\t\t\t" << "z = lambda*x" << endl
               << "\t\t"   << "else:" << endl
               << "\t\t\t" << "z = lambda*alpha*(np.exp(x)-1)" << endl
               << "\t\t"   << "return z" << endl
               << "\n" << endl;
    }

    if(HSigmoid)
    {
        buffer << "\tdef HardSigmoid (x):" << endl
               << "\t\t"   <<  "z = 1/(1+np.exp(-x))" << endl
               << "\t\t"   <<  "return z" << endl
               << "\n" << endl;
    }

    if(SoftPlus)
    {
        buffer << "\tdef SoftPlus (x):" << endl
               << "\t\t"   << "z = log(1+np.exp(x))" << endl
               << "\t\t"   << "return z" << endl
               << "\n" << endl;
    }

    if(SoftSign)
    {
        buffer << "\tdef SoftSign (x):" << endl
               << "\t\t"   << "z = x/(1+abs(x))" << endl
               << "\t\t"   << "return z" << endl
               << "\n" << endl;
    }

    buffer << "\t" << "def calculate_outputs(self, inputs):" << endl;

    for(int i = 0; i < inputs.dimension(0); i++)
        buffer << "\t\t" << inputs[i] << " = " << "inputs[" << to_string(i) << "]" << endl;

    if(LSTM_number > 0)
    {
        buffer << "\n\t\t" << "if(self.time_step_counter % self.current_combinations_derivatives == 0 ):" << endl
               << "\t\t\t" << "self.t = 1" << endl;

        for(int i = 0; i < hidden_state_counter; i++)
            buffer << "\t\t\t" << "self.hidden_state_" << to_string(i) << " = 0" << endl;

        for(int i = 0; i < cell_states_counter; i++)
            buffer << "\t\t\t" << "self.cell_states_" << to_string(i) << " = 0" << endl;
    }

    buffer << "" << endl;

    found_tokens.resize(0);
    push_back_string(found_tokens, "log");
    push_back_string(found_tokens, "exp");
    push_back_string(found_tokens, "tanh");

    push_back_string(found_mathematical_expressions, "Logistic");
    push_back_string(found_mathematical_expressions, "ReLU");
    push_back_string(found_mathematical_expressions, "ExponentialLinear");
    push_back_string(found_mathematical_expressions, "SELU");
    push_back_string(found_mathematical_expressions, "HardSigmoid");
    push_back_string(found_mathematical_expressions, "SoftPlus");
    push_back_string(found_mathematical_expressions, "SoftSign");

    string sufix;
    string new_word;
    string key_word ;

    for(int i = 0; i < tokens.dimension(0); i++)
    {
        string t = tokens(i);

        sufix = "np.";
        new_word = ""; 
        key_word = "";

        for(int i = 0; i < found_tokens.dimension(0); i++)
        {
            key_word = found_tokens(i);
            new_word = sufix + key_word;
            replace_all_appearances(t, key_word, new_word);
        }

        sufix = "NeuralNetwork.";
        new_word = ""; 
        key_word = "";

        for(int i = 0; i < found_mathematical_expressions.dimension(0); i++)
        {
            key_word = found_mathematical_expressions(i);
            new_word = sufix + key_word;
            replace_all_appearances(t, key_word, new_word);
        }

        if(LSTM_number>0)
        {
            replace_all_appearances(t, "(t)", "");
            replace_all_appearances(t, "(t-1)", "");
            replace_all_appearances(t, "cell_state", "self.cell_state");
            replace_all_appearances(t, "hidden_state", "self.hidden_state");
        }

        buffer << "\t\t" << t << endl;
    }

    const Tensor<string, 1> fixed_outputs = fix_write_expression_outputs(expression, outputs, "python");

    if(model_type != ModelType::AutoAssociation)
        for(int i = 0; i < fixed_outputs.dimension(0); i++)
            buffer << "\t\t" << fixed_outputs(i) << endl;

    buffer << "\t\t" << "out = " << "[None]*" << outputs.size() << "\n" << endl;

    for(int i = 0; i < outputs.dimension(0); i++)
        buffer << "\t\t" << "out[" << to_string(i) << "] = " << outputs[i] << endl;

    if(LSTM_number>0)
        buffer << "\n\t\t" << "self.time_step_counter += 1" << endl;

    if(model_type != ModelType::AutoAssociation)
        buffer << "\n\t\t" << "return out;" << endl;
    else
        buffer << "\n\t\t" << "return out, sample_autoassociation_distance, sample_autoassociation_variables_distance;" << endl;

    buffer << "\n" << endl
           << "\t" << "def calculate_batch_output(self, input_batch):" << endl
           << "\t\toutput_batch = [None]*input_batch.shape[0]\n" << endl
           << "\t\tfor i in range(input_batch.shape[0]):\n" << endl;

    if(has_recurrent_layer())
        buffer << "\t\t\tif(i%self.current_combinations_derivatives == 0):\n" << endl
               << "\t\t\t\tself.hidden_states = "+to_string(get_recurrent_layer()->get_neurons_number())+"*[0]\n" << endl;

    if(has_long_short_term_memory_layer())
        buffer << "\t\t\tif(i%self.current_combinations_derivatives == 0):\n" << endl
               << "\t\t\t\tself.hidden_states = "+to_string(get_long_short_term_memory_layer()->get_neurons_number())+"*[0]\n" << endl
               << "\t\t\t\tself.cell_states = "+to_string(get_long_short_term_memory_layer()->get_neurons_number())+"*[0]\n" << endl;

    buffer << "\t\t\tinputs = list(input_batch[i])\n" << endl
           << "\t\t\toutput = self.calculate_outputs(inputs)\n" << endl
           << "\t\t\toutput_batch[i] = output\n"<< endl
           << "\t\treturn output_batch\n"<<endl
           << "def main():" << endl
           << "\n\tinputs = []" << "\n" << endl;

    for(Index i = 0; i < inputs.size(); i++)
        buffer << "\t" << inputs(i) << " = " << "#- ENTER YOUR VALUE HERE -#" << endl
               << "\t" << "inputs.append(" << inputs(i) << ")" << "\n" << endl;

    buffer << "\t" << "nn = NeuralNetwork()" << endl
           << "\t" << "outputs = nn.calculate_outputs(inputs)" << endl
           << "\t" << "print(outputs)" << endl
           << "\n" << "main()" << endl;

    string out = buffer.str();

    replace(out, ";", "");

    return out;
}


void NeuralNetwork::save_expression_c(const string& file_name) const
{
    ofstream file(file_name.c_str());

    if(!file.is_open())
        throw runtime_error("Cannot open expression text file.\n");

    file << write_expression_c();

    file.close();
}


void NeuralNetwork::save_expression_api(const string& file_name) const
{
    ofstream file(file_name.c_str());

    if(!file.is_open())
        throw runtime_error("Cannot open expression text file.\n");

    file << write_expression_api();

    file.close();
}


void NeuralNetwork::save_expression_javascript(const string& file_name) const
{
    ofstream file(file_name.c_str());

    if(!file.is_open())
        throw runtime_error("Cannot open expression text file.\n");

    file << write_expression_javascript();

    file.close();
}


void NeuralNetwork::save_expression_python(const string& file_name) const
{
    ofstream file(file_name.c_str());

    if(!file.is_open())
        throw runtime_error("Cannot open expression text file.\n");

    file << write_expression_python();

    file.close();
}


void NeuralNetwork::save_outputs(Tensor<type, 2>& inputs, const string & file_name)
{
    Tensor<type, 2> outputs = calculate_outputs(inputs);

    ofstream file(file_name.c_str());

    if(!file.is_open())
        throw runtime_error("Cannot open " + file_name + " file.\n");

    const Tensor<string, 1> output_names = get_output_names();

    const Index outputs_number = get_outputs_number();
    const Index samples_number = inputs.dimension(0);

    for(Index i = 0; i < outputs_number; i++)
    {
        file << output_names[i];

        if(i != output_names.size()-1) file << ";";
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


Layer* NeuralNetwork::get_last_trainable_layer() const
{
    if(layers.size() == 0) return nullptr;

    Tensor<Layer*, 1> trainable_layers = get_trainable_layers();

    const Index trainable_layers_number = get_trainable_layers_number();

    return trainable_layers(trainable_layers_number-1);
}


void NeuralNetworkBackPropagation::set(const Index& new_batch_samples_number, NeuralNetwork* new_neural_network)
{
    batch_samples_number = new_batch_samples_number;

    neural_network = new_neural_network;

    const Tensor<Layer*, 1> neural_network_layers = neural_network->get_layers();

    const Index layers_number = neural_network_layers.size();

    layers.resize(layers_number);
    layers.setConstant(nullptr);

    for(Index i = 0; i < layers_number; i++)
    {
        switch (neural_network_layers(i)->get_type())
        {
        case Layer::Type::Perceptron:
            layers(i) = new PerceptronLayerBackPropagation(batch_samples_number, neural_network_layers(i));
        break;

        case Layer::Type::PerceptronLayer3D:
            layers(i) = new PerceptronLayer3DBackPropagation(batch_samples_number, neural_network_layers(i));
        break;

        case Layer::Type::Probabilistic:
            layers(i) = new ProbabilisticLayerBackPropagation(batch_samples_number, neural_network_layers(i));
        break;

        case Layer::Type::Probabilistic3D:
            layers(i) = new ProbabilisticLayer3DBackPropagation(batch_samples_number, neural_network_layers(i));
        break;

        case Layer::Type::Recurrent:
            layers(i) = new RecurrentLayerBackPropagation(batch_samples_number, neural_network_layers(i));
        break;

        case Layer::Type::LongShortTermMemory:
            layers(i) = new LongShortTermMemoryLayerBackPropagation(batch_samples_number, neural_network_layers(i));
        break;

        case Layer::Type::Convolutional:
            layers(i) = new ConvolutionalLayerBackPropagation(batch_samples_number, neural_network_layers(i));
        break;

        case Layer::Type::Pooling:
            layers(i) = new PoolingLayerBackPropagation(batch_samples_number, neural_network_layers(i));
        break;

        case Layer::Type::Flatten:
            layers(i) = new FlattenLayerBackPropagation(batch_samples_number, neural_network_layers(i));
        break;

        case Layer::Type::Embedding:
            layers(i) = new EmbeddingLayerBackPropagation(batch_samples_number, neural_network_layers(i));
        break;

        case Layer::Type::MultiheadAttention:
            layers(i) = new MultiheadAttentionLayerBackPropagation(batch_samples_number, neural_network_layers(i));
        break;

        case Layer::Type::Addition3D:
            layers(i) = new AdditionLayer3DBackPropagation(batch_samples_number, neural_network_layers(i));
        break;

        case Layer::Type::Normalization3D:
            layers(i) = new NormalizationLayer3DBackPropagation(batch_samples_number, neural_network_layers(i));
        break;

        default: break;
        }
    }
}


void ForwardPropagation::set(const Index& new_batch_samples_number, NeuralNetwork* new_neural_network)
{
    batch_samples_number = new_batch_samples_number;

    neural_network = new_neural_network;

    const Tensor<Layer*, 1> neural_network_layers = neural_network->get_layers();

    const Index layers_number = neural_network_layers.size();

    layers.resize(layers_number);
    layers.setConstant(nullptr);

    for(Index i = 0; i < layers_number; i++)
    {
        switch (neural_network_layers(i)->get_type())
        {
        case Layer::Type::Perceptron:
            layers(i) = new PerceptronLayerForwardPropagation(batch_samples_number, neural_network_layers(i));
        break;

        case Layer::Type::PerceptronLayer3D:
            layers(i) = new PerceptronLayer3DForwardPropagation(batch_samples_number, neural_network_layers(i));
        break;

        case Layer::Type::Probabilistic:
            layers(i) = new ProbabilisticLayerForwardPropagation(batch_samples_number, neural_network_layers(i));
        break;

        case Layer::Type::Probabilistic3D:
            layers(i) = new ProbabilisticLayer3DForwardPropagation(batch_samples_number, neural_network_layers(i));
        break;

        case Layer::Type::Recurrent:
            layers(i) = new RecurrentLayerForwardPropagation(batch_samples_number, neural_network_layers(i));
        break;

        case Layer::Type::LongShortTermMemory:
            layers(i) = new LongShortTermMemoryLayerForwardPropagation(batch_samples_number, neural_network_layers(i));
        break;

        case Layer::Type::Convolutional:
            layers(i) = new ConvolutionalLayerForwardPropagation(batch_samples_number, neural_network_layers(i));
        break;

        case Layer::Type::Pooling:
            layers(i) = new PoolingLayerForwardPropagation(batch_samples_number, neural_network_layers(i));
        break;

        case Layer::Type::Flatten:
            layers(i) = new FlattenLayerForwardPropagation(batch_samples_number, neural_network_layers(i));
        break;

        case Layer::Type::Scaling2D:
            layers(i) = new ScalingLayer2DForwardPropagation(batch_samples_number, neural_network_layers(i));
        break;

        case Layer::Type::Scaling4D:
            layers(i) = new ScalingLayer4DForwardPropagation(batch_samples_number, neural_network_layers(i));
        break;

        case Layer::Type::Unscaling:
            layers(i) = new UnscalingLayerForwardPropagation(batch_samples_number, neural_network_layers(i));
        break;

        case Layer::Type::Bounding:
            layers(i) = new BoundingLayerForwardPropagation(batch_samples_number, neural_network_layers(i));
        break;

        case Layer::Type::RegionProposal:
            //                layers(i) = new RegionProposalLayerForwardPropagation(batch_samples_number, neural_network_layers(i));
        break;

        case Layer::Type::Embedding:
            layers(i) = new EmbeddingLayerForwardPropagation(batch_samples_number, neural_network_layers(i));
        break;

        case Layer::Type::MultiheadAttention:
            layers(i) = new MultiheadAttentionLayerForwardPropagation(batch_samples_number, neural_network_layers(i));
        break;

        case Layer::Type::Addition3D:
            layers(i) = new AdditionLayer3DForwardPropagation(batch_samples_number, neural_network_layers(i));
        break;

        case Layer::Type::Normalization3D:
            layers(i) = new NormalizationLayer3DForwardPropagation(batch_samples_number, neural_network_layers(i));
        break;

        default: cout << "Default" << endl; break;
        }
    }
}


pair<type*, dimensions> ForwardPropagation::get_last_trainable_layer_outputs_pair() const
{
    const Index last_trainable_layer_index = neural_network->get_last_trainable_layer_index();

    return layers(last_trainable_layer_index)->get_outputs_pair();
}


void NeuralNetworkBackPropagationLM::set(const Index new_batch_samples_number, NeuralNetwork* new_neural_network)
{
    batch_samples_number = new_batch_samples_number;

    neural_network = new_neural_network;

    const Tensor<Layer*, 1> trainable_layers = neural_network->get_trainable_layers();

    const Index trainable_layers_number = trainable_layers.size();

    layers.resize(trainable_layers_number);

    for(Index i = 0; i < trainable_layers_number; i++)
    {
        switch (trainable_layers(i)->get_type())
        {
        case Layer::Type::Perceptron:
            layers(i) = new PerceptronLayerBackPropagationLM(batch_samples_number, trainable_layers(i));

            break;

        case Layer::Type::Probabilistic:
            layers(i) = new ProbabilisticLayerBackPropagationLM(batch_samples_number, trainable_layers(i));

            break;

        default:
            throw runtime_error("Levenberg-Marquardt can only be used with Perceptron and Probabilistic layers.\n");
        }
    }
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
