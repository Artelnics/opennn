//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   N E U R A L   N E T W O R K   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "tensors.h"
#include "images.h"
#include "neural_network.h"
#include "forward_propagation.h"
#include "neural_network_back_propagation.h"
#include "neural_network_back_propagation_lm.h"
#include "layer.h"
#include "perceptron_layer.h"
#include "perceptron_layer_3d.h"
#include "pooling_layer.h"
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
#include "recurrent_layer.h"
#include "long_short_term_memory_layer.h"

namespace opennn
{

NeuralNetwork::NeuralNetwork(const NeuralNetwork::ModelType& model_type, 
                             const dimensions& input_dimensions,
                             const dimensions& complexity_dimensions,
                             const dimensions& output_dimensions)
{
    set(model_type, input_dimensions, complexity_dimensions, output_dimensions);
}


NeuralNetwork::NeuralNetwork(const filesystem::path& file_name)
{
    load(file_name);
}


void NeuralNetwork::add_layer(unique_ptr<Layer> layer, const vector<Index>& input_indices)
{
    const Layer::Type layer_type = layer->get_type();

    if(!validate_layer_type(layer_type)) return;

    const Index old_layers_number = get_layers_number();

    layers.push_back(std::move(layer));

    layer_input_indices.push_back(input_indices.empty()

        ? vector<Index>(1, old_layers_number - 1)

        : input_indices);
}


bool NeuralNetwork::validate_layer_type(const Layer::Type& layer_type) const
{
    if(has(Layer::Type::Bounding))
        throw runtime_error("No layers can be added after a bounding layer.\n");

    return true;
}


bool NeuralNetwork::has(const Layer::Type& layer_type) const
{
    const Index layers_number = get_layers_number();

    for (Index i = 0; i < layers_number; i++)
        if (layers[i]->get_type() == layer_type)
            return true;

    return false;
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


NeuralNetwork::ModelType NeuralNetwork::get_model_type() const
{
    return model_type;
}


string NeuralNetwork::get_model_type_string() const
{
    switch (model_type)
    {
        case ModelType::Default:
            return "Default";
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


const unique_ptr<Layer>& NeuralNetwork::get_layer(const string& layer_name) const
{
    const vector<string> layer_names = get_layer_names();

    for(size_t i = 0; i < layer_names.size(); i++)
        if(layer_names[i] == layer_name)
            return layers[i];

    throw runtime_error("Layer not found in neural network");
}


Index NeuralNetwork::get_layer_index(const string& layer_name) const
{
    if(layer_name == "dataset" || layer_name == "input")
        return -1;

    if(layer_name == "context")
        return -2;

    const Index layers_number = get_layers_number();

    for(Index i = 0; i < layers_number; i++)
        if(layers[i]->get_name() == layer_name)
            return i;

    throw runtime_error("Layer not found" + layer_name);
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
        for (size_t j = 0; j < layer_input_indices[i].size(); j++)
        {
            const Index input_index = layer_input_indices[i][j];

            if (input_index != -1) 
                layer_output_indices[input_index].push_back(i);
        }
    }

    for (Index i = 0; i < layers_number; i++)
        if (layer_output_indices[i].empty()) 
            layer_output_indices[i].push_back(-1);

    return layer_output_indices;
}


Index NeuralNetwork::find_input_index(const vector<Index>& layer_inputs_indices, const Index& layer_index) const
{
    for (Index i = 0; i < Index(layer_inputs_indices.size()); i++)
        if (layer_inputs_indices[i] == layer_index)
            return i;

    return -1;
}


Layer* NeuralNetwork::get_first(const Layer::Type& layer_type) const
{
    const Index layers_number = get_layers_number();

    for(Index i = 0; i < layers_number; i++)
        if(layers[i]->get_type() == layer_type)
            return layers[i].get();

    throw runtime_error("Neural network does not have layer type.");
}


const bool& NeuralNetwork::get_display() const
{
    return display;
}


void NeuralNetwork::set(const NeuralNetwork::ModelType& new_model_type,
                        const dimensions& input_dimensions, 
                        const dimensions& complexity_dimensions,
                        const dimensions& output_dimensions)
{
    set_default();

    layers.resize(0);

    model_type = new_model_type;

    const Index inputs_number = accumulate(input_dimensions.begin(), 
                                           input_dimensions.end(), 
                                           1, 
                                           multiplies<Index>());

    input_names.resize(inputs_number);

    for(Index i = 0; i < inputs_number; i++)
        input_names[i] = "input_" + to_string(i+1);

    switch(model_type)
    {    
    case ModelType::Approximation:
        set_approximation(input_dimensions, complexity_dimensions, output_dimensions);
        break;
    
    case ModelType::Classification: 
        set_classification(input_dimensions, complexity_dimensions, output_dimensions);
        break;

    case ModelType::TextClassification:
        set_classification(input_dimensions, complexity_dimensions, output_dimensions);
        break;

    case ModelType::Forecasting:
        set_forecasting(input_dimensions, complexity_dimensions, output_dimensions);
        break;

    case ModelType::ImageClassification:
        set_image_classification(input_dimensions, complexity_dimensions, output_dimensions);
        break;
    
    case ModelType::AutoAssociation:
        set_auto_association(input_dimensions, complexity_dimensions, output_dimensions);
        break;

/*
    case ModelType::TextClassificationTransformer:
        set_text_classification_transformer(input_dimensions, complexity_dimensions, output_dimensions);
        break;
*/

    default:
        break;

    }

    const Index outputs_number = accumulate(output_dimensions.begin(), 
                                            output_dimensions.end(), 
                                            1, 
                                            multiplies<Index>());

    output_names.resize(outputs_number);

    for(Index i = 0; i < outputs_number; i++)
        output_names[i] = "output_" + to_string(i+1);
}


void NeuralNetwork::set_approximation(const dimensions& input_dimensions, 
                                      const dimensions& complexity_dimensions, 
                                      const dimensions& output_dimensions)
{

    const Index complexity_size = complexity_dimensions.size();

    add_layer(make_unique<ScalingLayer2D>(input_dimensions));

    for (Index i = 0; i < complexity_size; i++)
        add_layer(make_unique<PerceptronLayer>(get_output_dimensions(),
                                               dimensions{ complexity_dimensions[i] },
                                               PerceptronLayer::ActivationFunction::RectifiedLinear,
                                               "perceptron_layer_" + to_string(i + 1)));

    add_layer(make_unique<PerceptronLayer>(get_output_dimensions(),
                                           output_dimensions,
                                           PerceptronLayer::ActivationFunction::Linear,
                                           "perceptron_layer_" + to_string(complexity_size + 1)));

    add_layer(make_unique<UnscalingLayer>(output_dimensions));

    add_layer(make_unique<BoundingLayer>(output_dimensions));
}


void NeuralNetwork::set_classification(const dimensions& input_dimensions, 
                                       const dimensions& complexity_dimensions, 
                                       const dimensions& output_dimensions)
{
    const Index complexity_size = complexity_dimensions.size();

    add_layer(make_unique<ScalingLayer2D>(input_dimensions));

    for (Index i = 0; i < complexity_size; i++)
        add_layer(make_unique<PerceptronLayer>(get_output_dimensions(),
                                               dimensions{complexity_dimensions[i]},
                                               PerceptronLayer::ActivationFunction::HyperbolicTangent,
                                               "perceptron_layer_" + to_string(i + 1)));

    add_layer(make_unique<ProbabilisticLayer>(get_output_dimensions(),
                                              output_dimensions,
                                              "probabilistic_layer"));
}


void NeuralNetwork::set_forecasting(const dimensions& input_dimensions, 
                                    const dimensions& complexity_dimensions, 
                                    const dimensions& output_dimensions)
{
    add_layer(make_unique<ScalingLayer2D>(input_dimensions));

    add_layer(make_unique<RecurrentLayer>(get_output_dimensions(),
        dimensions{ complexity_dimensions[0] }));

    add_layer(make_unique<PerceptronLayer>(get_output_dimensions(),
        output_dimensions,
        PerceptronLayer::ActivationFunction::HyperbolicTangent,
        "recurrent_layer"));

    add_layer(make_unique<UnscalingLayer>(output_dimensions));

    add_layer(make_unique<BoundingLayer>(output_dimensions));
}


void NeuralNetwork::set_auto_association(const dimensions& input_dimensions, 
                                         const dimensions& complexity_dimensions, 
                                         const dimensions& output_dimensions)
{
    add_layer(make_unique<ScalingLayer2D>(input_dimensions));

    const Index mapping_neurons_number = 10;
    const Index bottle_neck_neurons_number = complexity_dimensions[0];

    add_layer(make_unique<PerceptronLayer>(input_dimensions, 
                                           dimensions{mapping_neurons_number}, 
                                           PerceptronLayer::ActivationFunction::HyperbolicTangent,
                                           "mapping_layer"));

    add_layer(make_unique<PerceptronLayer>(dimensions{ mapping_neurons_number },
                                           dimensions{ bottle_neck_neurons_number },
                                           PerceptronLayer::ActivationFunction::Linear,
                                           "bottleneck_layer"));

    add_layer(make_unique<PerceptronLayer>(dimensions{ bottle_neck_neurons_number },
                                           dimensions{ mapping_neurons_number },
                                           PerceptronLayer::ActivationFunction::HyperbolicTangent,
                                           "demapping_layer"));

    add_layer(make_unique<PerceptronLayer>(dimensions{ mapping_neurons_number },
                                           dimensions{ output_dimensions }, 
                                           PerceptronLayer::ActivationFunction::Linear,
                                           "output_layer"));

    add_layer(make_unique<UnscalingLayer>(output_dimensions));
}


void NeuralNetwork::set_image_classification(const dimensions& input_dimensions, 
                                             const dimensions& complexity_dimensions, 
                                             const dimensions& output_dimensions)
{
    if (input_dimensions.size() != 3)
        throw runtime_error("Input dimensions size is not 3.");

    const Index complexity_size = complexity_dimensions.size();

    add_layer(make_unique<ScalingLayer4D>(input_dimensions));

    for (Index i = 0; i < complexity_size; i++)
    {
        const dimensions kernel_dimensions = { 3, 3, get_output_dimensions()[2], complexity_dimensions[i] };
        const dimensions convolution_stride_dimensions = { 1, 1 };
        const ConvolutionalLayer::ConvolutionType convolution_type = ConvolutionalLayer::ConvolutionType::Valid;

        //add_layer(make_unique<ConvolutionalLayer>(get_output_dimensions(),
        //                                          kernel_dimensions,
        //                                          ConvolutionalLayer::ActivationFunction::RectifiedLinear,
        //                                          convolution_stride_dimensions,
        //                                          convolution_type,
        //                                          "convolutional_layer_" + to_string(i+1)));

        const dimensions pool_dimensions = { 2, 2 };
        const dimensions pooling_stride_dimensions = { 2, 2 };
        const dimensions padding_dimensions = { 0, 0 };
        const PoolingLayer::PoolingMethod pooling_method = PoolingLayer::PoolingMethod::MaxPooling;

        add_layer(make_unique<PoolingLayer>(get_output_dimensions(),
                                            pool_dimensions,
                                            pooling_stride_dimensions,
                                            padding_dimensions,
                                            pooling_method,
                                            "pooling_layer_" + to_string(i + 1)));
    }

    add_layer(make_unique<FlattenLayer>(get_output_dimensions()));

    add_layer(make_unique<ProbabilisticLayer>(get_output_dimensions(),
                                              output_dimensions,
                                              "probabilistic_layer"));
}

/*
void NeuralNetwork::set_text_classification_transformer(const dimensions& input_dimensions,
                                         const dimensions& complexity_dimensions,
                                                        const dimensions& output_dimensions)
{

    layers.resize(0);

    // input_names.resize(input_length + context_length);

    const Index complexity_size = complexity_dimensions.size();

    // Embedding Layers

    const Index embedding_depth = 32;
    const Index perceptron_depth = 32;
    const Index heads_number = 2;
    const type dropout_rate = 0;

    unique_ptr<EmbeddingLayer> embedding_layer
        = make_unique<EmbeddingLayer>(input_dimensions[0],
                                      input_dimensions[1],
                                      embedding_depth,
                                      true);

    embedding_layer->set_dropout_rate(dropout_rate);
    embedding_layer->set_name("embedding");
    //name = embedding_layer->get_name();
    add_layer(std::move(embedding_layer));
    set_layer_inputs_indices("embedding", "input");

    // cout<<get_output_dimensions().size()<<endl;
    // cout<<get_output_dimensions()[0]<<endl;
    // cout<<get_output_dimensions()[1]<<endl;
    // Encoder

    for(Index i = 0; i < complexity_size; i++)
    {
        // Multi head attention

        unique_ptr<MultiheadAttentionLayer> self_attention_layer =
            make_unique<MultiheadAttentionLayer>(input_dimensions[1],
                                                 input_dimensions[1],
                                                 embedding_depth,
                                                 heads_number);

        self_attention_layer->set_dropout_rate(dropout_rate);
        self_attention_layer->set_name("self_attention_" + to_string(i+1));
        //name = input_self_attention_layer->get_name();

        add_layer(std::move(self_attention_layer));

        if(i == 0)
            set_layer_inputs_indices("self_attention_1", {"embedding", "embedding"});
        else
            set_layer_inputs_indices("self_attention_" + to_string(i+1), { "perceptron_normalization_" + to_string(i), "perceptron_normalization_" + to_string(i) });

        // Addition

        unique_ptr<AdditionLayer3D> self_attention_addition_layer
            = make_unique<AdditionLayer3D>(input_dimensions[1], embedding_depth);

        self_attention_addition_layer->set_name("self_attention_addition_" + to_string(i+1));
        //name = self_attention_addition_layer->get_name();

        add_layer(std::move(self_attention_addition_layer));

        if(i == 0)
            set_layer_inputs_indices("self_attention_addition_" + to_string(i+1), { "embedding", "self_attention_" + to_string(i+1) });
        else
            set_layer_inputs_indices("self_attention_addition_" + to_string(i+1), { "perceptron_normalization_" + to_string(i), "self_attention_" + to_string(i+1) });

        // Normalization

        unique_ptr<NormalizationLayer3D> self_attention_normalization_layer
            = make_unique<NormalizationLayer3D>(input_dimensions[1], embedding_depth);

        self_attention_normalization_layer->set_name("self_attention_normalization_" + to_string(i+1));
        //name = self_attention_normalization_layer->get_name();

        add_layer(std::move(self_attention_normalization_layer));

        set_layer_inputs_indices("self_attention_normalization_" + to_string(i+1), "self_attention_addition_" + to_string(i+1));

        // Perceptron

        unique_ptr<PerceptronLayer3D> encoder_internal_perceptron_layer
            = make_unique<PerceptronLayer3D>(input_dimensions[1], embedding_depth, perceptron_depth, PerceptronLayer3D::ActivationFunction::RectifiedLinear);

        encoder_internal_perceptron_layer->set_name("encoder_internal_perceptron_" + to_string(i+1));
        //name = encoder_internal_perceptron_layer->get_name();

        add_layer(std::move(encoder_internal_perceptron_layer));

        set_layer_inputs_indices("encoder_internal_perceptron_" + to_string(i+1), "self_attention_normalization_" + to_string(i+1));

        // Perceptron

        unique_ptr<PerceptronLayer3D> encoder_external_perceptron_layer =
            make_unique<PerceptronLayer3D>(input_dimensions[1], perceptron_depth, embedding_depth, PerceptronLayer3D::ActivationFunction::RectifiedLinear);

        encoder_external_perceptron_layer->set_dropout_rate(dropout_rate);
        encoder_external_perceptron_layer->set_name("encoder_external_perceptron_" + to_string(i+1));
        //name = encoder_external_perceptron_layer->get_name();

        add_layer(std::move(encoder_external_perceptron_layer));

        set_layer_inputs_indices("encoder_external_perceptron_" + to_string(i+1), "encoder_internal_perceptron_" + to_string(i+1));

        // Addition

        unique_ptr<AdditionLayer3D> encoder_perceptron_addition_layer
            = make_unique<AdditionLayer3D>(input_dimensions[1], embedding_depth);

        encoder_perceptron_addition_layer->set_name("encoder_perceptron_addition_" + to_string(i+1));
        //name = encoder_perceptron_addition_layer->get_name();

        add_layer(std::move(encoder_perceptron_addition_layer));

        set_layer_inputs_indices("encoder_perceptron_addition_" + to_string(i+1), { "self_attention_normalization_" + to_string(i+1), "encoder_external_perceptron_" + to_string(i+1) });

        // Normalization

        unique_ptr<NormalizationLayer3D> encoder_perceptron_normalization_layer
            = make_unique<NormalizationLayer3D>(input_dimensions[1], embedding_depth);

        encoder_perceptron_normalization_layer->set_name("encoder_perceptron_normalization_" + to_string(i+1));
        //name = encoder_perceptron_normalization_layer->get_name();

        add_layer(std::move(encoder_perceptron_normalization_layer));

        set_layer_inputs_indices("encoder_perceptron_normalization_" + to_string(i+1), "encoder_perceptron_addition_" + to_string(i+1));

    }

        // Global Average Pooling

        const dimensions pool_dimensions = { 1, input_dimensions[1] };
        const dimensions pooling_stride_dimensions = { 1, input_dimensions[1] };
        const dimensions padding_dimensions = { 0, 0 };
        const PoolingLayer::PoolingMethod pooling_method = PoolingLayer::PoolingMethod::AveragePooling;

        add_layer(make_unique<PoolingLayer>(get_output_dimensions(),
                                            pool_dimensions,
                                            pooling_stride_dimensions,
                                            padding_dimensions,
                                            pooling_method,
                                            "pooling_layer"));

        set_layer_inputs_indices("global_average_pooling", "encoder_perceptron_normalization_" + to_string(complexity_size));

        add_layer(make_unique<PerceptronLayer>(get_output_dimensions(),
                                               output_dimensions,
                                               PerceptronLayer::ActivationFunction::Logistic,
                                               "perceptron_layer_" + to_string(complexity_size + 1)));

        set_layer_inputs_indices("perceptron_layer_" + to_string(complexity_size + 1), "global_average_pooling");
    }
*/
};


void NeuralNetwork::set(const filesystem::path& file_name)
{
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
        throw runtime_error("Unknown model type: " + new_model_type + "\n");
}


void NeuralNetwork::set_input_names(const vector<string>& new_input_namess)
{
    input_names = new_input_namess;
}


void NeuralNetwork::set_output_namess(const vector<string>& new_output_namess)
{
    output_names = new_output_namess;
}


void NeuralNetwork::set_input_dimensions(const dimensions& new_input_dimensions)
{
/*
    input_names.resize(new_inputs_number);

    if(has(Layer::Type::Scaling2D))
    {
        ScalingLayer2D* scaling_layer_2d = get_scaling_layer_2d();

        scaling_layer_2d->set_inputs_number(new_inputs_number);
    }

    const Index first_trainable_layer_index = get_first_trainable_layer_index();

    layers[first_trainable_layer_index]->set_inputs_number(new_inputs_number);
*/
}


void NeuralNetwork::set_default()
{
    display = true;

    layer_input_indices = vector<vector<Index>>();
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


void NeuralNetwork::set_layer_inputs_indices(const string& layer_name,
                                             const vector<string>& new_layer_input_names)
{
    const Index layer_index = get_layer_index(layer_name);

    const Index size = new_layer_input_names.size();

    vector<Index> new_layer_input_indices(size);

    for(Index i = 0; i < size; i++)
        new_layer_input_indices[i] = get_layer_index(new_layer_input_names[i]);

    layer_input_indices[layer_index] = new_layer_input_indices;
}


void NeuralNetwork::set_layer_inputs_indices(const string& layer_name, 
                                             const initializer_list<string>& new_layer_input_names_list)
{
    const vector<string> new_layer_input_names = new_layer_input_names_list;

    set_layer_inputs_indices(layer_name, new_layer_input_names);
}


void NeuralNetwork::set_layer_inputs_indices(const string& layer_name, const string& new_layer_input_names)
{
    const Index layer_index = get_layer_index(layer_name);

    layer_input_indices[layer_index] = {get_layer_index(new_layer_input_names)};
}


Index NeuralNetwork::get_inputs_number() const
{
    if(layers.empty())
        return 0;

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


dimensions NeuralNetwork::get_output_dimensions() const
{
    if(layers.empty()) 
        return {};

    return layers[layers.size() - 1]->get_output_dimensions();
}


Index NeuralNetwork::get_parameters_number() const
{
    Index parameters_number = 0;

    #pragma omp parallel for reduction(+: parameters_number)

    for(Index i = 0; i < Index(layers.size()); i++)
        parameters_number += layers[i]->get_parameters_number();

    return parameters_number;
}


Tensor<type, 1> NeuralNetwork::get_parameters() const
{
    const Index parameters_number = get_parameters_number();

    Tensor<type, 1> parameters(parameters_number);

    Index position = 0;

    for (const auto& layer : layers)
    {
        const Tensor<type, 1> layer_parameters = layer->get_parameters();

        memcpy(parameters.data() + position,
               layer_parameters.data(),
               layer_parameters.size() * sizeof(type));

        position += layer_parameters.size();
    }

    return parameters;
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


void NeuralNetwork::set_parameters(const Tensor<type, 1>& new_parameters) const
{
    const Index layers_number = get_layers_number();

    const vector<Index> layer_parameter_numbers = get_layer_parameter_numbers();

    Index index = 0;

    for(Index i = 0; i < layers_number; i++)
    {
        layers[i]->set_parameters(new_parameters, index);

        index += layer_parameter_numbers[i];
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


bool NeuralNetwork::is_trainable(const Layer::Type& layer_type)
{
    return layer_type != Layer::Type::Scaling2D &&
           layer_type != Layer::Type::Scaling4D &&
           layer_type != Layer::Type::Unscaling &&
           layer_type != Layer::Type::Bounding;
}


Index NeuralNetwork::get_first_trainable_layer_index() const
{
    const Index layers_number = get_layers_number();

    for(Index i = 0; i < layers_number; i++)
        if (is_trainable(layers[i]->get_type())) 
            return i;

    throw runtime_error("The neural network has no trainable layers.");
}


Index NeuralNetwork::get_last_trainable_layer_index() const
{
    const Index layers_number = get_layers_number();

    for(Index i = layers_number-1; i >= 0 ; i--)
        if (is_trainable(layers[i]->get_type()))
            return i;

    throw runtime_error("The neural network has no trainable layers.");
}

Index NeuralNetwork::get_layers_number(const Layer::Type& layer_type) const
{
    const Index layers_number = get_layers_number();

    Index count = 0;

    #pragma omp parallel for reduction(+: count)

    for(Index i = 0; i < layers_number; i++)
        if(layers[i]->get_type() == layer_type)
            count++;

    return count;
}


// bool NeuralNetwork::is_input_layer(const vector<Index>& this_layer_inputs_indices) const
// {
//     const Index input_layers_number = this_layer_inputs_indices.size();

//     for(Index i = 0; i < input_layers_number; i++)
//         if(this_layer_inputs_indices[i] == -1)
//             return true;

//     return false;
// }


// bool NeuralNetwork::is_context_layer(const vector<Index>& this_layer_inputs_indices) const
// {
//     // @todo Is this ok?
//     const Index layers_number = get_layers_number();

//     for(Index i = 0; i < layers_number; i++)
//         if(this_layer_inputs_indices[i] == -2)
//             return true;

//     return false;
// }


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

    #pragma omp parallel for
    for(Index i = 0; i < layers_number; i++)
        layers[i]->set_parameters_random();
}


void NeuralNetwork::forward_propagate(const vector<pair<type*, dimensions>>& input_pair,
                                      ForwardPropagation& forward_propagation,
                                      const bool& is_training) const
{
    const Index layers_number = get_layers_number();

    const Index first_trainable_layer_index = get_first_trainable_layer_index();
    const Index last_trainable_layer_index = get_last_trainable_layer_index();

    const Index first_layer_index = is_training ? first_trainable_layer_index : 0;
    const Index last_layer_index = is_training ? last_trainable_layer_index : layers_number - 1;

    const vector<vector<pair<type*, dimensions>>> layer_input_pairs = forward_propagation.get_layer_input_pairs(input_pair);

    for (Index i = first_layer_index; i <= last_layer_index; i++){

        layers[i]->forward_propagate(layer_input_pairs[i],
                                     forward_propagation.layers[i],
                                     is_training);}

}


void NeuralNetwork::forward_propagate(const vector<pair<type*, dimensions>>& input_pair,
                                      const Tensor<type, 1>& new_parameters,
                                      ForwardPropagation& forward_propagation) const
{
    const Tensor<type, 1> original_parameters = get_parameters();

    set_parameters(new_parameters);

    const bool is_training = true;

    forward_propagate(input_pair, forward_propagation, is_training);

    set_parameters(original_parameters);
}


string NeuralNetwork::get_expression() const
{
    const Index layers_number = get_layers_number();

    const vector<string> layer_names = get_layer_names();

    vector<string> new_input_names = get_input_names();
    vector<string> new_output_names = get_output_names();

    const Index inputs_number = input_names.size();
    const Index outputs_number = output_names.size();

    for (int i = 0; i < inputs_number; i++)
        input_names[i].empty()
            ? new_input_names[i] = "input_" + to_string(i)
            : new_input_names[i] = "XXX"/*replace_non_allowed_programming_expressions(input_names[i])*/;

    vector<string> scaled_input_names(inputs_number);
    vector<string> unscaled_output_names(outputs_number);

    ostringstream buffer;

    for (Index i = 0; i < layers_number; i++)
        buffer << layers[i]->get_expression();

/*
    if (i == layers_number - 1)
    {
    for (int j = 0; j < output_names.size(); j++)
    if (!output_names[j].empty())
  ;//  output_names[j] = replace_non_allowed_programming_expressions(output_names[j]);
    else
    output_names[j] = "output_" + to_string(i);

    //buffer << layers[i]->get_expression(input_names, output_names) << endl;
    }
    else
    {

//    layer_neurons_number = layers[i]->get_neurons_number();
//    output_namess_vector.resize(layer_neurons_number);

    for (Index j = 0; j < layer_neurons_number; j++)
    {
    if (layer_names[i] == "scaling_layer")
    {
//    output_names[j] = "scaled_" + replace_non_allowed_programming_expressions(input_names[j]);
    scaled_input_names[j] = output_names[j];
    }
    else
    {
    output_names[j] = layer_names[i] + "_output_" + to_string(j);
    }
    }

    //            buffer << layers[i]->get_expression(input_names, output_names) << endl;
    //            input_namess_vector = output_namess_vector;
    //            unscaled_output_names = input_namess_vector;
    }
    }
*/
    string expression = buffer.str();

    //replace(expression, "+-", "-");
    return expression;
}


Tensor<type, 2> NeuralNetwork::calculate_outputs(const Tensor<type, 2>& inputs)
{
    const Index layers_number = get_layers_number();

    if (layers_number == 0)
        return Tensor<type, 2>();

    const Index batch_samples_number = inputs.dimension(0);
    const Index inputs_number = inputs.dimension(1);

    ForwardPropagation forward_propagation(batch_samples_number, this);

    const pair<type*, dimensions> input_pair((type*)inputs.data(), {{batch_samples_number, inputs_number}});

    forward_propagate({input_pair}, forward_propagation);

    //forward_propagation.print();

    const pair<type*, dimensions> outputs_pair
        = forward_propagation.layers[layers_number - 1]->get_outputs_pair();

    return tensor_map_2(outputs_pair);
}


Tensor<type, 2> NeuralNetwork::calculate_outputs(const Tensor<type, 4>& inputs)
{
    const Index layers_number = get_layers_number();

    if (layers_number == 0) 
        return Tensor<type, 2>();

    const Index batch_samples_number = inputs.dimension(0);

    ForwardPropagation forward_propagation(batch_samples_number, this);

    const pair<type*, dimensions> input_pair((type*)inputs.data(), { {inputs.dimension(0), inputs.dimension(1), inputs.dimension(2), inputs.dimension(3)}});

    forward_propagate({input_pair}, forward_propagation);

    const pair<type*, dimensions> outputs_pair 
        = forward_propagation.layers[layers_number - 1]->get_outputs_pair();

    return tensor_map_2(outputs_pair);
}


Tensor<type, 2> NeuralNetwork::calculate_scaled_outputs(const Tensor<type, 2>&)
{
    return Tensor<type, 2>();
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


Index NeuralNetwork::calculate_image_output(const string& image_path)
{
    const Tensor<unsigned char, 3> image_data = read_bmp_image(image_path);

    ScalingLayer4D* scaling_layer_4d = static_cast<ScalingLayer4D*>(get_first(Layer::Type::Scaling4D));

    const Index height = scaling_layer_4d->get_input_dimensions()[0];
    const Index width = scaling_layer_4d->get_input_dimensions()[1];
    const Index image_channels = scaling_layer_4d->get_input_dimensions()[2];

    const Index current_height = image_data.dimension(0);
    const Index current_width = image_data.dimension(1);
    const Index current_channels = image_data.dimension(2);

    if (current_channels != image_channels)
        throw runtime_error("Error: Different channels number " + image_path + "\n");

    Tensor<unsigned char, 3> resized_image_data(height, width, image_channels);
/*
    (current_height != height || current_width != width)
        ? bilinear_interpolation_resize_image(image_data, resized_image_data, height, width)
        : resized_image_data = image_data;
*/
    Tensor<type, 4> input_data(1, height, width, image_channels);

    const Index pixels_number = height * width * image_channels;

#pragma omp parallel for
    for (Index j = 0; j < pixels_number; j++)
        input_data(j) = resized_image_data(j);

    const Tensor<type, 2> outputs = calculate_outputs(input_data);

    Index predicted_index = -1;

    if (outputs.size() > 1)
    {
        type max_value = outputs(0);
        for (Index i = 1; i < outputs.dimension(1); ++i) {
            if (outputs(i) > max_value) {
                max_value = outputs(i);
                predicted_index = i;
            }
        }
    }
    else
        predicted_index = outputs(0);

    return predicted_index;
}


Tensor<string, 2> NeuralNetwork::get_perceptron_layers_information() const
{
    const Index layers_number = get_layers_number();

    const Index perceptron_layers_number = get_layers_number(Layer::Type::Perceptron);

    Tensor<string, 2> information(perceptron_layers_number, 3);

    Index perceptron_layer_index = 0;

    for(Index i = 0; i < layers_number; i++)
    {
        const Layer::Type layer_type = layers[i]->get_type();

        if (layer_type != Layer::Type::Perceptron) 
            continue;

        information(perceptron_layer_index, 0) = to_string(layers[i]->get_input_dimensions()[0]);
        information(perceptron_layer_index, 1) = to_string(layers[i]->get_output_dimensions()[0]);

        const PerceptronLayer* perceptron_layer = static_cast<PerceptronLayer*>(layers[i].get());

        information(perceptron_layer_index, 2) = perceptron_layer->get_activation_function_string();

        perceptron_layer_index++;
    }

    return information;
}


Tensor<string, 2> NeuralNetwork::get_probabilistic_layer_information() const
{
    const Index layers_number = get_layers_number();

    const Index probabilistic_layers_number = get_layers_number(Layer::Type::Probabilistic);

    Tensor<string, 2> information(probabilistic_layers_number, 3);

    Index probabilistic_layer_index = 0;

    for(Index i = 0; i < layers_number; i++)
    {
        const Layer::Type layer_type = layers[i]->get_type();

        if (layer_type != Layer::Type::Probabilistic) 
            continue;

        information(probabilistic_layer_index,0) = to_string(layers[i]->get_input_dimensions()[0]);
        information(probabilistic_layer_index,1) = to_string(layers[i]->get_output_dimensions()[0]);

        const ProbabilisticLayer* probabilistic_layer = static_cast<ProbabilisticLayer*>(layers[i].get());

        information(probabilistic_layer_index,2) = probabilistic_layer->get_activation_function_string();

        probabilistic_layer_index++;
    }

    return information;
}


void NeuralNetwork::to_XML(XMLPrinter& printer) const
{
    const Index inputs_number = get_inputs_number();
    const Index layers_number = get_layers_number();
    const Index outputs_number = output_names.size();

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

    add_xml_element(printer, "OutputsNumber", to_string(outputs_number));

    for (Index i = 0; i < outputs_number; i++) 
        add_xml_element_attribute(printer, "Output", output_names[i], "Index", to_string(i + 1));

    printer.CloseElement(); 

    printer.CloseElement();
}


void NeuralNetwork::from_XML(const XMLDocument& document)
{
    set();

    const XMLElement* neural_network_element = document.FirstChildElement("NeuralNetwork");

    if(!neural_network_element)
        throw runtime_error("Neural network element is nullptr.\n");

    inputs_from_XML(neural_network_element->FirstChildElement("Inputs"));
    layers_from_XML(neural_network_element->FirstChildElement("Layers"));
    outputs_from_XML(neural_network_element->FirstChildElement("Outputs"));
    //set_display(read_xml_bool(neural_network_element, "Display"));
}


void NeuralNetwork::inputs_from_XML(const XMLElement* inputs_element)
{
    if(!inputs_element)
        throw runtime_error("Inputs element is nullptr.\n");

    const Index new_inputs_number = read_xml_index(inputs_element, "InputsNumber");

    input_names.resize(new_inputs_number);

    const XMLElement* inputs_number_element = inputs_element->FirstChildElement("InputsNumber");

    // Inputs names

    const XMLElement* start_element = inputs_number_element;

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
    if(!layers_element)
        throw runtime_error("Layers element is nullptr.\n");

    const Index layers_number = read_xml_index(layers_element, "LayersNumber");

    // Add layers

    const XMLElement* start_element = layers_element->FirstChildElement("LayersNumber");

    for(Index i = 0; i < layers_number; i++)
    {
        const XMLElement* layer_element = start_element->NextSiblingElement();

        if(!layer_element)
             throw runtime_error("Layer element is nullptr.");

        const string layer_type_string = layer_element->Name();

        XMLDocument layer_document;
        XMLNode* element_clone = layer_element->DeepClone(&layer_document);
        layer_document.InsertFirstChild(element_clone);

        if (layer_type_string == "Scaling2D") {
            unique_ptr<ScalingLayer2D> scaling_layer = make_unique<ScalingLayer2D>();
            scaling_layer->from_XML(layer_document);
            add_layer(std::move(scaling_layer));
        }
        else if (layer_type_string == "Scaling4D") {
            unique_ptr<ScalingLayer4D> scaling_layer = make_unique<ScalingLayer4D>();
            scaling_layer->from_XML(layer_document);
            add_layer(std::move(scaling_layer));
        }
        else if (layer_type_string == "Convolutional") {
            unique_ptr<ConvolutionalLayer> convolutional_layer = make_unique<ConvolutionalLayer>();
            convolutional_layer->from_XML(layer_document);
            add_layer(std::move(convolutional_layer));
        }
        else if (layer_type_string == "Perceptron") {
            unique_ptr<PerceptronLayer> perceptron_layer = make_unique<PerceptronLayer>();
            perceptron_layer->from_XML(layer_document);
            add_layer(std::move(perceptron_layer));
        }
        else if (layer_type_string == "Perceptron3D") {
            unique_ptr<PerceptronLayer3D> perceptron_layer_3d = make_unique<PerceptronLayer3D>();
            perceptron_layer_3d->from_XML(layer_document);
            add_layer(std::move(perceptron_layer_3d));
        }
        else if (layer_type_string == "Pooling") {
            unique_ptr<PoolingLayer> pooling_layer = make_unique<PoolingLayer>();
            pooling_layer->from_XML(layer_document);
            add_layer(std::move(pooling_layer));
        }
        else if (layer_type_string == "Flatten") {
            unique_ptr<FlattenLayer> flatten_layer = make_unique<FlattenLayer>();
            flatten_layer->from_XML(layer_document);
            add_layer(std::move(flatten_layer));
        }
        else if (layer_type_string == "Probabilistic") {
            unique_ptr<ProbabilisticLayer> probabilistic_layer = make_unique<ProbabilisticLayer>();
            probabilistic_layer->from_XML(layer_document);
            add_layer(std::move(probabilistic_layer));
        }
        else if (layer_type_string == "Probabilistic3D") {
            unique_ptr<ProbabilisticLayer3D> probabilistic_layer_3d = make_unique<ProbabilisticLayer3D>();
            probabilistic_layer_3d->from_XML(layer_document);
            add_layer(std::move(probabilistic_layer_3d));
        }
        else if (layer_type_string == "LongShortTermMemory") {
            unique_ptr<LongShortTermMemoryLayer> long_short_term_memory_layer = make_unique<LongShortTermMemoryLayer>();
            long_short_term_memory_layer->from_XML(layer_document);
            add_layer(std::move(long_short_term_memory_layer));
        }
        else if (layer_type_string == "Recurrent") {
            unique_ptr<RecurrentLayer> recurrent_layer = make_unique<RecurrentLayer>();
            recurrent_layer->from_XML(layer_document);
            add_layer(std::move(recurrent_layer));
        }
        else if (layer_type_string == "Unscaling") {
            unique_ptr<UnscalingLayer> unscaling_layer = make_unique<UnscalingLayer>();
            unscaling_layer->from_XML(layer_document);
            add_layer(std::move(unscaling_layer));
        }
        else if (layer_type_string == "Bounding") {
            unique_ptr<BoundingLayer> bounding_layer = make_unique<BoundingLayer>();
            bounding_layer->from_XML(layer_document);
            add_layer(std::move(bounding_layer));
        }
        else if (layer_type_string == "Embedding") {
            unique_ptr<EmbeddingLayer> embedding_layer = make_unique<EmbeddingLayer>();
            embedding_layer->from_XML(layer_document);
            add_layer(std::move(embedding_layer));
        }
        else if (layer_type_string == "MultiheadAttention") {
            unique_ptr<MultiheadAttentionLayer> multihead_attention_layer = make_unique<MultiheadAttentionLayer>();           
            multihead_attention_layer->from_XML(layer_document);
            add_layer(std::move(multihead_attention_layer));
        }
        else if (layer_type_string == "Addition3D") {
            unique_ptr<AdditionLayer3D> addition_layer_3d = std::make_unique<AdditionLayer3D>();
            addition_layer_3d->from_XML(layer_document);
            add_layer(std::move(addition_layer_3d));
        }
        else if (layer_type_string == "Normalization3D") {
            unique_ptr<NormalizationLayer3D> normalization_layer_3d = make_unique<NormalizationLayer3D>();
            normalization_layer_3d->from_XML(layer_document);
            add_layer(std::move(normalization_layer_3d));
        }
        else {
            throw runtime_error("Unknown layer type");
        }

        start_element = layer_element;
        
    }

    // Layers inputs indices

    const XMLElement* layer_input_indices_element = layers_element->FirstChildElement("LayerInputIndices");

    if(!layer_input_indices_element)
        throw runtime_error("LayerInputIndices element is nullptr.\n");

    layer_input_indices.clear(); // @todo .clear because they are already saved from Add layers for (is this code needed?)
    layer_input_indices.resize(layers.size());

    for (const tinyxml2::XMLElement* layer_inputs_indices_element = layer_input_indices_element->FirstChildElement("LayerInputsIndices");
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
    cout << "Neural network" << endl
         << "Model type:" << endl
         << get_model_type_string() << endl;

    print_vector(get_input_names());

    if(model_type != ModelType::ImageClassification)
    {
        cout << "Inputs:" << endl;
        print_vector(get_input_names());
    }
    const Index layers_number = get_layers_number();       

    cout << "Layers number: " << layers_number << endl;

    for(Index i = 0; i < layers_number; i++)
    {
        cout << endl
             << "Layer " << i << ": " << endl;
        layers[i]->print();
    }

    cout << "Outputs:" << endl;
    print_vector(get_output_names());

    cout << "Parameters number:" << endl
         << get_parameters_number() << endl;
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

    const Tensor<type, 1> parameters = get_parameters();

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

    const Index parameters_number = get_parameters_number();

    Tensor<type, 1> new_parameters(parameters_number);

    file.read(reinterpret_cast<char*>(new_parameters.data()), parameters_number * sizeof(type));

    if (!file)
        throw runtime_error("Error reading binary file: " + file_name.string());

    set_parameters(new_parameters);
}


void NeuralNetwork::save_expression(const ProgrammingLanguage& programming_language,
                                    const filesystem::path& file_name) const
{
    ofstream file(file_name);

    if(!file.is_open())
        throw runtime_error("Cannot open expression text file.\n");
    /*
    file << get_expression_c();
    */
    file.close();
}


void NeuralNetwork::save_outputs(Tensor<type, 2>& inputs, const filesystem::path& file_name)
{
    const Tensor<type, 2> outputs = calculate_outputs(inputs);

    ofstream file(file_name);

    if(!file.is_open())
        throw runtime_error("Cannot open " + file_name.string() + " file.\n");

    //const vector<string> output_names = get_output_names();

    const Index outputs_number = get_outputs_number();
    const Index samples_number = inputs.dimension(0);

    for(size_t i = 0; i < size_t(outputs_number); i++)
    {
        file << output_names[i];

        if(i != output_names.size() - 1) 
            file << ";";
    }

    file << "\n";

    for(Index i = 0; i < samples_number; i++)
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


vector<string> NeuralNetwork::get_layer_names() const
{
    const Index layers_number = get_layers_number();

    vector<string> layer_names(layers_number);

    for(Index i = 0; i < layers_number; i++)
        layer_names[i] = layers[i]->get_name();

    return layer_names;
}


vector<string> NeuralNetwork::get_layer_types_string() const
{
    const Index layers_number = get_layers_number();

    vector<string> layer_types(layers_number);

    for(Index i = 0; i < layers_number; i++)
        layer_types[i] = layers[i]->get_type_string();

    return layer_types;
}


NeuralNetworkBackPropagation::NeuralNetworkBackPropagation(const Index& new_batch_samples_number, 
                                                           NeuralNetwork* new_neural_network)
{
    set(new_batch_samples_number, new_neural_network);
}


void NeuralNetworkBackPropagation::set(const Index& new_batch_samples_number, NeuralNetwork* new_neural_network)
{
    batch_samples_number = new_batch_samples_number;

    neural_network = new_neural_network;

    if(!neural_network) return;

    const vector<unique_ptr<Layer>>& neural_network_layers = neural_network->get_layers();

    const Index layers_number = neural_network_layers.size();

    layers.resize(layers_number);

    for(Index i = 0; i < layers_number; i++)
    {
        switch (neural_network_layers[i]->get_type())
        {
        case Layer::Type::Perceptron:
            layers[i] = make_unique< PerceptronLayerBackPropagation>(batch_samples_number, neural_network_layers[i].get());
        break;

        case Layer::Type::Perceptron3D:
            layers[i] = make_unique < PerceptronLayer3DBackPropagation>(batch_samples_number, neural_network_layers[i].get());
        break;

        case Layer::Type::Probabilistic:
            layers[i] = make_unique < ProbabilisticLayerBackPropagation>(batch_samples_number, neural_network_layers[i].get());
        break;

        case Layer::Type::Probabilistic3D:
            layers[i] = make_unique < ProbabilisticLayer3DBackPropagation>(batch_samples_number, neural_network_layers[i].get());
        break;

        case Layer::Type::Recurrent:
            layers[i] = make_unique < RecurrentLayerBackPropagation>(batch_samples_number, neural_network_layers[i].get());
        break;

        case Layer::Type::LongShortTermMemory:
            layers[i] = make_unique < LongShortTermMemoryLayerBackPropagation>(batch_samples_number, neural_network_layers[i].get());
        break;

        case Layer::Type::Convolutional:
            layers[i] = make_unique < ConvolutionalLayerBackPropagation>(batch_samples_number, neural_network_layers[i].get());
        break;

        case Layer::Type::Pooling:
            layers[i] = make_unique < PoolingLayerBackPropagation>(batch_samples_number, neural_network_layers[i].get());
        break;

        case Layer::Type::Flatten:
            layers[i] = make_unique < FlattenLayerBackPropagation>(batch_samples_number, neural_network_layers[i].get());
        break;

        case Layer::Type::Embedding:
            layers[i] = make_unique < EmbeddingLayerBackPropagation>(batch_samples_number, neural_network_layers[i].get());
        break;

        case Layer::Type::MultiheadAttention:
            layers[i] = make_unique < MultiheadAttentionLayerBackPropagation>(batch_samples_number, neural_network_layers[i].get());
        break;

        case Layer::Type::Addition3D:
            layers[i] = make_unique < AdditionLayer3DBackPropagation>(batch_samples_number, neural_network_layers[i].get());
        break;

        case Layer::Type::Normalization3D:
            layers[i] = make_unique < NormalizationLayer3DBackPropagation>(batch_samples_number, neural_network_layers[i].get());
        break;

        default: break;
        }
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
             << neural_network->get_layer(i)->get_type_string() << endl;

        if (!layers[i]) continue;

        layers[i]->print();
    }
}


ForwardPropagation::ForwardPropagation(const Index& new_batch_samples_number,
                                       NeuralNetwork* new_neural_network)
{
    set(new_batch_samples_number, new_neural_network);
}


void ForwardPropagation::set(const Index& new_batch_samples_number, NeuralNetwork* new_neural_network)
{
    batch_samples_number = new_batch_samples_number;

    neural_network = new_neural_network;

    if(!neural_network) return;

    const vector<unique_ptr<Layer>>& neural_network_layers = neural_network->get_layers();

    const Index layers_number = neural_network_layers.size();

    layers.resize(layers_number);

    for(Index i = 0; i < layers_number; i++)
    {
        switch (neural_network_layers[i]->get_type())
        {
        case Layer::Type::Perceptron:
            layers[i] = make_unique<PerceptronLayerForwardPropagation>(batch_samples_number, neural_network_layers[i].get());
        break;
        
        case Layer::Type::Perceptron3D:
            layers[i] = make_unique<PerceptronLayer3DForwardPropagation>(batch_samples_number, neural_network_layers[i].get());
        break;

        case Layer::Type::Probabilistic:
            layers[i] = make_unique<ProbabilisticLayerForwardPropagation>(batch_samples_number, neural_network_layers[i].get());
        break;

        case Layer::Type::Probabilistic3D:
            layers[i] = make_unique<ProbabilisticLayer3DForwardPropagation>(batch_samples_number, neural_network_layers[i].get());
        break;

        case Layer::Type::Recurrent:
            layers[i] = make_unique<RecurrentLayerForwardPropagation>(batch_samples_number, neural_network_layers[i].get());
        break;

        case Layer::Type::LongShortTermMemory:
            layers[i] = make_unique<LongShortTermMemoryLayerForwardPropagation>(batch_samples_number, neural_network_layers[i].get());
        break;

        case Layer::Type::Convolutional:
            layers[i] = make_unique<ConvolutionalLayerForwardPropagation>(batch_samples_number, neural_network_layers[i].get());
        break;

        case Layer::Type::Pooling:
            layers[i] = make_unique<PoolingLayerForwardPropagation>(batch_samples_number, neural_network_layers[i].get());
        break;

        case Layer::Type::Flatten:
            layers[i] = make_unique<FlattenLayerForwardPropagation>(batch_samples_number, neural_network_layers[i].get());
        break;

        case Layer::Type::Scaling2D:
            layers[i] = make_unique<ScalingLayer2DForwardPropagation>(batch_samples_number, neural_network_layers[i].get());
        break;

        case Layer::Type::Scaling4D:
            layers[i] = make_unique<ScalingLayer4DForwardPropagation>(batch_samples_number, neural_network_layers[i].get());
        break;

        case Layer::Type::Unscaling:
            layers[i] = make_unique<UnscalingLayerForwardPropagation>(batch_samples_number, neural_network_layers[i].get());
        break;

        case Layer::Type::Bounding:
            layers[i] = make_unique<BoundingLayerForwardPropagation>(batch_samples_number, neural_network_layers[i].get());
        break;

        case Layer::Type::Embedding:
            layers[i] = make_unique<EmbeddingLayerForwardPropagation>(batch_samples_number, neural_network_layers[i].get());
        break;

        case Layer::Type::MultiheadAttention:
            layers[i] = make_unique<MultiheadAttentionLayerForwardPropagation>(batch_samples_number, neural_network_layers[i].get());
        break;

        case Layer::Type::Addition3D:
            layers[i] = make_unique<AdditionLayer3DForwardPropagation>(batch_samples_number, neural_network_layers[i].get());
        break;

        case Layer::Type::Normalization3D:
            layers[i] = make_unique<NormalizationLayer3DForwardPropagation>(batch_samples_number, neural_network_layers[i].get());
        break;

        default: cout << "Default" << endl; break;
        }
    }
}


pair<type*, dimensions> ForwardPropagation::get_last_trainable_layer_outputs_pair() const
{
    const Index last_trainable_layer_index = neural_network->get_last_trainable_layer_index();

    const unique_ptr<LayerForwardPropagation>& layer_forward_propagation = layers[last_trainable_layer_index];

    return layer_forward_propagation->get_outputs_pair();
}


vector<vector<pair<type*, dimensions>>> ForwardPropagation::get_layer_input_pairs(const vector<pair<type*, dimensions>>& batch_input_pairs) const
{
    const Index layers_number = neural_network->get_layers_number();

    if (layers_number == 0) 
        return vector<vector<pair<type*, dimensions>>>();

    const vector<vector<Index>>& layer_input_indices = neural_network->get_layer_input_indices();

    vector<vector<pair<type*, dimensions>>> layer_input_pairs(layers_number);

    //layer_input_pairs[0] = batch_input_pairs;

    const Index first_trainable_layer_index = neural_network->get_first_trainable_layer_index();

    for (Index i = first_trainable_layer_index; i < layers_number; i++)
    {
        const vector<Index>& this_layer_input_indices = layer_input_indices[i];

        layer_input_pairs[i].resize(1);

        // if (i == first_trainable_layer_index)
        // {
        //     layer_input_pairs[i] = batch_input_pairs;
        //     continue;
        // }

        if(neural_network->get_model_type_string() == "TextClassification"){

            if (i == first_trainable_layer_index)
            {   vector<pair<type*, dimensions>> batch_input_pairs1;
                batch_input_pairs1.push_back(batch_input_pairs[0]);
                layer_input_pairs[i] = batch_input_pairs1;
                continue;
            }

            if (i == first_trainable_layer_index+1)
            {
                vector<pair<type*, dimensions>> batch_input_pairs2;
                batch_input_pairs2.push_back(batch_input_pairs[1]);
                layer_input_pairs[i] = batch_input_pairs2;
                continue;
            }
        } else {
            if (i == first_trainable_layer_index)
            {
                layer_input_pairs[i] = batch_input_pairs;
                continue;
            }
        };

        const Index this_layer_inputs_number = this_layer_input_indices.size();
        layer_input_pairs[i].resize(this_layer_inputs_number);

        for (Index j = 0; j < this_layer_inputs_number; j++)
        {
            const Index this_layer_input_index = this_layer_input_indices[j];

            layer_input_pairs[i][j] = layers[this_layer_input_index]->get_outputs_pair();

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
        cout << "Layer " << i + 1 << ": " << neural_network->get_layer(i)->get_name() << endl;

        layers[i]->print();
    }
}


void NeuralNetworkBackPropagationLM::set(const Index& new_batch_samples_number, 
                                         NeuralNetwork* new_neural_network)
{
    batch_samples_number = new_batch_samples_number;

    neural_network = new_neural_network;

    const Index layers_number = neural_network->get_layers_number();

    const vector<unique_ptr<Layer>>& neural_network_layers = neural_network->get_layers();

    layers.resize(layers_number);

    for(Index i = 0; i < layers_number; i++)
    {
        switch (neural_network_layers[i]->get_type())
        {
        case Layer::Type::Perceptron:
            layers[i] = make_unique<PerceptronLayerBackPropagationLM>(batch_samples_number, neural_network_layers[i].get());
            break;

        case Layer::Type::Probabilistic:
            layers[i] = make_unique<ProbabilisticLayerBackPropagationLM>(batch_samples_number, neural_network_layers[i].get());
            break;

        default:
            continue;
            //throw runtime_error("Levenberg-Marquardt can only be used with Perceptron and Probabilistic layers.\n");
        }
    }
}

// Namespace

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
