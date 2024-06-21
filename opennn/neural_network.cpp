//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   N E U R A L   N E T W O R K   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "neural_network.h"
#include "neural_network_forward_propagation.h"
#include "neural_network_back_propagation.h"
#include "neural_network_back_propagation_lm.h"

namespace opennn
{

/// Default constructor.
/// It creates an empty neural network object.
/// It also initializes all pointers in the object to nullptr.
/// Finally, it initializes the rest of the members to their default values.
    
NeuralNetwork::NeuralNetwork()
{
    set();
}


/// Type of model and architecture of the Neural Network constructor.
/// It creates a neural network object with the given model type and architecture.
/// It initializes the rest of the members to their default values.
/// @param model_type Type of problem to be solved with the neural network
/// (Approximation, Classification, Forecasting, ImageClassification, TextClassification, AutoAssociation).
/// @param architecture Architecture of the neural network({inputs_number, hidden_neurons_number, outputs_number}).

NeuralNetwork::NeuralNetwork(const NeuralNetwork::ModelType& model_type, const Tensor<Index, 1>& architecture)
{
    set(model_type, architecture);
}


NeuralNetwork::NeuralNetwork(const NeuralNetwork::ModelType& model_type, const initializer_list<Index>& architecture_list)
{
    Tensor<Index, 1> architecture(architecture_list.size());
    architecture.setValues(architecture_list);

    set(model_type, architecture);
}


/// (Convolutional layer) constructor.
/// It creates a neural network object with the given parameters.
/// Note that this method is only valid when our problem presents convolutional layers.

NeuralNetwork::NeuralNetwork(const Tensor<Index, 1>& new_inputs_dimensions,
                             const Index& new_blocks_number,
                             const Tensor<Index, 1>& new_filters_dimensions,
                             const Index& new_outputs_number)
{
    set(new_inputs_dimensions, new_blocks_number, new_filters_dimensions, new_outputs_number);
}


/// File constructor.
/// It creates a neural network object by loading its members from an XML-type file.
/// @param file_name Name of neural network file.

NeuralNetwork::NeuralNetwork(const string& file_name)
{
    load(file_name);
}


/// XML constructor.
/// It creates a neural network object by loading its members from an XML document.
/// @param document TinyXML document containing the neural network data.

NeuralNetwork::NeuralNetwork(const tinyxml2::XMLDocument& document)
{
    from_XML(document);
}


/// Layers constructor.
/// It creates a neural network object by
/// It also sets the rest of the members to their default values.

NeuralNetwork::NeuralNetwork(const Tensor<Layer*, 1>& new_layers)
{
    set();

    layers = new_layers;
}


/// Destructor.

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


/// Add a new layer to the Neural Network model.
/// @param layer The layer that will be added.

void NeuralNetwork::add_layer(Layer* layer)
{
    if(has_bounding_layer())
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: NeuralNetwork class.\n"
               << "NeuralNetwork::add_layer() method.\n"
               << "No layers can be added after a bounding layer.\n";

        print();

        throw runtime_error(buffer.str());
    }
    
    const Layer::Type layer_type = layer->get_type();

    if(check_layer_type(layer_type))
    {
        const Index old_layers_number = get_layers_number();
 
        // Layers pointers
        
        Tensor<Layer*, 1> old_layers = get_layers();
        
        layers.resize(old_layers_number + 1);

        for(Index i = 0; i < old_layers_number; i++) layers(i) = old_layers(i);

        layers(old_layers_number) = layer;
        
        // Layers inputs indices

        Tensor<Tensor<Index, 1>, 1> old_layers_inputs_indices = get_layers_inputs_indices();

        layers_inputs_indices.resize(old_layers_number+1);

        for(Index i = 0; i < old_layers_number; i++) 
            layers_inputs_indices(i) = old_layers_inputs_indices(i);

        Tensor<Index, 1> new_layer_inputs_indices(1);
        new_layer_inputs_indices(0) = old_layers_number-1;

        layers_inputs_indices(old_layers_number) = new_layer_inputs_indices;
    }
    else
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: NeuralNetwork class.\n"
               << "void add_layer(const Layer*) method.\n"
               << "Layer type " << layer->get_type_string() << " cannot be added in position " << layers.size()
               << " in the network architecture.\n";

        throw runtime_error(buffer.str());
    }

}


/// Check if a given layer type can be added to the structure of the neural network.
/// LSTM and Recurrent layers can only be added at the beginning.
/// @param layer_type Type of new layer to be added.

bool NeuralNetwork::check_layer_type(const Layer::Type layer_type)
{
    const Index layers_number = layers.size();

    if(layers_number > 1 && (layer_type == Layer::Type::Recurrent || layer_type == Layer::Type::LongShortTermMemory))
    {
        return false;
    }
    else if(layers_number == 1 && (layer_type == Layer::Type::Recurrent || layer_type == Layer::Type::LongShortTermMemory))
    {
        const Layer::Type first_layer_type = layers[0]->get_type();

        if(first_layer_type != Layer::Type::Scaling2D) return false;
    }

    return true;
}


/// Returns true if the neural network object has a scaling layer object inside,
/// and false otherwise.

bool NeuralNetwork::has_scaling_layer() const
{
    const Index layers_number = get_layers_number();

    for(Index i = 0; i < layers_number; i++)
    {
        if(layers[i]->get_type() == Layer::Type::Scaling2D) return true;
    }

    return false;
}

/// Returns true if the neural network object has a scaling layer 4d object inside,
/// and false otherwise.

bool NeuralNetwork::has_scaling_4d_layer() const
{
    const Index layers_number = get_layers_number();

    for (Index i = 0; i < layers_number; i++)
    {
        if (layers[i]->get_type() == Layer::Type::Scaling4D) return true;
    }

    return false;
}


/// Returns true if the neural network object has a long short-term memory layer object inside,
/// and false otherwise.

bool NeuralNetwork::has_long_short_term_memory_layer() const
{
    const Index layers_number = get_layers_number();

    for(Index i = 0; i < layers_number; i++)
    {
        if(layers[i]->get_type() == Layer::Type::LongShortTermMemory) return true;
    }

    return false;
}


/// Returns true if the neural network object has a convolutional object inside,
/// and false otherwise.

bool NeuralNetwork::has_convolutional_layer() const
{
    const Index layers_number = get_layers_number();

    for(Index i = 0; i < layers_number; i++)
    {
        if(layers[i]->get_type() == Layer::Type::Convolutional) return true;
    }

    return false;
}

/// Returns true if the neural network object has a flaten object inside,
/// and false otherwise.

bool NeuralNetwork::has_flatten_layer() const
{
    const Index layers_number = get_layers_number();

    for(Index i = 0; i < layers_number; i++)
    {
        if(layers[i]->get_type() == Layer::Type::Flatten) return true;
    }

    return false;
}



/// Returns true if the neural network object has a recurrent layer object inside,
/// and false otherwise.

bool NeuralNetwork::has_recurrent_layer() const
{
    const Index layers_number = get_layers_number();

    for(Index i = 0; i < layers_number; i++)
    {
        if(layers[i]->get_type() == Layer::Type::Recurrent) return true;
    }

    return false;
}


/// Returns true if the neural network object has an unscaling layer object inside,
/// and false otherwise.

bool NeuralNetwork::has_unscaling_layer() const
{
    const Index layers_number = get_layers_number();

    for(Index i = 0; i < layers_number; i++)
    {
        if(layers[i]->get_type() == Layer::Type::Unscaling) return true;
    }

    return false;
}


/// Returns true if the neural network object has a bounding layer object inside,
/// and false otherwise.

bool NeuralNetwork::has_bounding_layer() const
{
    const Index layers_number = get_layers_number();

    for(Index i = 0; i < layers_number; i++)
    {
        if(layers[i]->get_type() == Layer::Type::Bounding) return true;
    }

    return false;
}


/// Returns true if the neural network object has a probabilistic layer object inside,
/// and false otherwise.

bool NeuralNetwork::has_probabilistic_layer() const
{
    const Index layers_number = get_layers_number();

    for(Index i = 0; i < layers_number; i++)
    {
        if(layers[i]->get_type() == Layer::Type::Probabilistic) return true;
    }

    return false;
}


/// Returns true if the neural network object is empty,
/// and false otherwise.

bool NeuralNetwork::is_empty() const
{
    if(layers.dimension(0) == 0) return true;

    return false;
}


/// Returns a string vector with the names of the variables used as inputs.

const Tensor<string, 1>& NeuralNetwork::get_inputs_names() const
{
    return inputs_names;
}


/// Returns a string with the name of the variable used as inputs on a certain index.
/// @param index Index of the variable to be examined.

string NeuralNetwork::get_input_name(const Index& index) const
{
    return inputs_names[index];
}


/// Returns the index of the variable with a given name.
/// @param name Name of the variable to be examined.

Index NeuralNetwork::get_input_index(const string& name) const
{
    for(Index i = 0; i < inputs_names.size(); i++)
    {
        if(inputs_names(i) == name) return i;
    }

    return 0;
}

NeuralNetwork::ModelType NeuralNetwork::get_model_type() const
{
    return model_type;
}


string NeuralNetwork::get_model_type_string() const
{
    if (model_type == ModelType::AutoAssociation)
    {
        return "AutoAssociation";
    }
    else if(model_type == ModelType::Approximation)
    {
        return "Approximation";
    }
    else if(model_type == ModelType::Classification)
    {
        return "Classification";
    }
    else if(model_type == ModelType::Forecasting)
    {
        return "Forecasting";
    }
    else if(model_type == ModelType::TextClassification)
    {
        return "TextClassification";
    }
    else if(model_type == ModelType::ImageClassification)
    {
        return "ImageClassification";
    }
    else
    {
        throw runtime_error("Unkown model type");
    }
}


/// Returns a string vector with the names of the variables used as outputs.

const Tensor<string, 1>& NeuralNetwork::get_outputs_names() const
{
    return outputs_names;
}


/// Returns a string with the name of the variable used as outputs on a certain index.
/// @param index Index of the variable to be examined.

string NeuralNetwork::get_output_name(const Index& index) const
{
    return outputs_names[index];
}


/// Returns the index of the variable with a given name.
/// @param name Name of the variable to be examined.

Index NeuralNetwork::get_output_index(const string& name) const
{
    for(Index i = 0; i < outputs_names.size(); i++)
    {
        if(outputs_names(i) == name) return i;
    }

    return 0;
}


/// Returns a pointer to the layers object composing this neural network object.

Tensor<Layer*, 1> NeuralNetwork::get_layers() const
{
    return layers;
}


Layer* NeuralNetwork::get_layer(const Index& layer_index) const
{
    return layers(layer_index);
}

Layer* NeuralNetwork::get_layer(const string& layer_name) const
{
    Tensor<string, 1> layers_names = get_layers_names();

    for (Index i = 0; i < layers_names.size(); i++)
    {
        if (layers_names(i) == layer_name)    return layers(i);
    }

    return nullptr;
}


/// Returns a pointer to the trainable layers object composing this neural network object.

Tensor<Layer*, 1> NeuralNetwork::get_trainable_layers() const
{
    const Index layers_number = get_layers_number();

    const Index trainable_layers_number = get_trainable_layers_number();

    Tensor<Layer*, 1> trainable_layers(trainable_layers_number);

    Index index = 0;

    Layer::Type layer_type;

    for(Index i = 0; i < layers_number; i++)
    {
        layer_type = layers(i)->get_type();

        if(layer_type != Layer::Type::Scaling2D
        && layer_type != Layer::Type::Scaling4D
        && layer_type != Layer::Type::Unscaling
        && layer_type != Layer::Type::Bounding)
        {
            trainable_layers(index) = layers(i);
            index++;
        }
    }

    return trainable_layers;
}


Index NeuralNetwork::get_layer_index(const string& layer_name) const
{
    const Index layers_number = get_layers_number();

    if(layer_name == "dataset" || layer_name == "input")
    {
        return -1;
    }

    if (layer_name == "context")
    {
        return -2;
    }

    for(Index i = 0; i < layers_number; i++)
    {
        if(layers(i)->get_name() == layer_name)
        {
            return i;
        }
    }

    return 0;
}


/// Returns a vector with the indices of the trainable layers.

Tensor<Index, 1> NeuralNetwork::get_trainable_layers_indices() const
{
    const Index layers_number = get_layers_number();

    const Index trainable_layers_number = get_trainable_layers_number();

    Tensor<Index, 1> trainable_layers_indices(trainable_layers_number);

    Index trainable_layer_index = 0;

    for(Index i = 0; i < layers_number; i++)
    {
        if(layers[i]->get_type() != Layer::Type::Scaling2D
        && layers[i]->get_type() != Layer::Type::Unscaling
        && layers[i]->get_type() != Layer::Type::Bounding)
        {
            trainable_layers_indices[trainable_layer_index] = i;
            trainable_layer_index++;
        }
    }

    return trainable_layers_indices;
}


const Tensor<Tensor<Index, 1>, 1>& NeuralNetwork::get_layers_inputs_indices() const
{
    return layers_inputs_indices;
}


/// Returns a pointer to the scaling layer object composing this neural network object.

ScalingLayer2D* NeuralNetwork::get_scaling_layer_2d() const
{
    const Index layers_number = get_layers_number();

    for(Index i = 0; i < layers_number; i++)
    {
        if(layers[i]->get_type() == Layer::Type::Scaling2D)
        {
            return dynamic_cast<ScalingLayer2D*>(layers[i]);
        }
    }

    ostringstream buffer;

    buffer << "OpenNN Exception: NeuralNetwork class.\n"
           << "ScalingLayer2D* get_scaling_layer_2d() const method.\n"
           << "No scaling layer in neural network.\n";

    throw runtime_error(buffer.str());
}


ScalingLayer4D* NeuralNetwork::get_scaling_layer_4d() const
{
    const Index layers_number = get_layers_number();

    for(Index i = 0; i < layers_number; i++)
    {
        if(layers[i]->get_type() == Layer::Type::Scaling4D)
        {
            return dynamic_cast<ScalingLayer4D*>(layers[i]);
        }
    }

    ostringstream buffer;

    buffer << "OpenNN Exception: NeuralNetwork class.\n"
           << "ScalingLayer* get_scaling_layer_4d() const method.\n"
           << "No scaling layer in neural network.\n";

    throw runtime_error(buffer.str());
}

/// Returns a pointer to the unscaling layers object composing this neural network object.

UnscalingLayer* NeuralNetwork::get_unscaling_layer() const
{
    const Index layers_number = get_layers_number();

    for(Index i = 0; i < layers_number; i++)
    {
        if(layers[i]->get_type() == Layer::Type::Unscaling)
        {
            return dynamic_cast<UnscalingLayer*>(layers[i]);
        }
    }

    ostringstream buffer;

    buffer << "OpenNN Exception: NeuralNetwork class.\n"
           << "UnscalingLayer* get_unscaling_layer() const method.\n"
           << "No unscaling layer in neural network.\n";

    throw runtime_error(buffer.str());
}


/// Returns a pointer to the bounding layer object composing this neural network object.

BoundingLayer* NeuralNetwork::get_bounding_layer() const
{
    const Index layers_number = get_layers_number();

    for(Index i = 0; i < layers_number; i++)
    {
        if(layers[i]->get_type() == Layer::Type::Bounding)
        {
            return dynamic_cast<BoundingLayer*>(layers[i]);
        }
    }

    ostringstream buffer;

    buffer << "OpenNN Exception: NeuralNetwork class.\n"
           << "BoundingLayer* get_bounding_layer() const method.\n"
           << "No bounding layer in neural network.\n";

    throw runtime_error(buffer.str());
}

/// Returns a pointer to the flatten layer object composing this neural network object.

FlattenLayer* NeuralNetwork::get_flatten_layer() const
{
    const Index layers_number = get_layers_number();

    for(Index i = 0; i < layers_number; i++)
    {
        if(layers[i]->get_type() == Layer::Type::Flatten)
        {
            return dynamic_cast<FlattenLayer*>(layers[i]);
        }
    }

    ostringstream buffer;

    buffer << "OpenNN Exception: NeuralNetwork class.\n"
           << "BoundingLayer* get_flatten_layer() const method.\n"
           << "No flatten layer in neural network.\n";

    throw runtime_error(buffer.str());
}


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
//    ostringstream buffer;
//
//    buffer << "OpenNN Exception: NeuralNetwork class.\n"
//           << "ConvolutionalLayer* get_convolutional_layer() const method.\n"
//           << "No convolutional layer in neural network.\n";
//
//    throw runtime_error(buffer.str());
//}


PoolingLayer* NeuralNetwork::get_pooling_layer() const
{
    const Index layers_number = get_layers_number();

    for(Index i = 0; i < layers_number; i++)
    {
        if(layers[i]->get_type() == Layer::Type::Pooling)
        {
            return dynamic_cast<PoolingLayer*>(layers[i]);
        }
    }

    ostringstream buffer;

    buffer << "OpenNN Exception: NeuralNetwork class.\n"
           << "PoolingLayer* get_pooling_layer() const method.\n"
           << "No pooling layer in neural network.\n";

    throw runtime_error(buffer.str());
}


/// Returns a pointer to the first probabilistic layer composing this neural network.

ProbabilisticLayer* NeuralNetwork::get_probabilistic_layer() const
{
    const Index layers_number = get_layers_number();

    for(Index i = 0; i < layers_number; i++)
    {
        if(layers[i]->get_type() == Layer::Type::Probabilistic)
        {
            return dynamic_cast<ProbabilisticLayer*>(layers[i]);
        }
    }

    ostringstream buffer;

    buffer << "OpenNN Exception: NeuralNetwork class.\n"
           << "ProbabilisticLayer* get_probabilistic_layer() const method.\n"
           << "No probabilistic layer in neural network.\n";

    throw runtime_error(buffer.str());
}

/// Returns a pointer to the long short-term memory layer of this neural network, if it exits.

LongShortTermMemoryLayer* NeuralNetwork::get_long_short_term_memory_layer() const
{
    const Index layers_number = get_layers_number();

    for(Index i = 0; i < layers_number; i++)
    {
        if(layers[i]->get_type() == Layer::Type::LongShortTermMemory)
        {
            return dynamic_cast<LongShortTermMemoryLayer*>(layers[i]);
        }
    }

    ostringstream buffer;

    buffer << "OpenNN Exception: NeuralNetwork class.\n"
           << "LongShortTermMemoryLayer* get_long_short_term_memory_layer() const method.\n"
           << "No long-short-term memory layer in neural network.\n";

    throw runtime_error(buffer.str());
}


/// Returns a pointer to the recurrent layer of this neural network, if it exits.

RecurrentLayer* NeuralNetwork::get_recurrent_layer() const
{
    const Index layers_number = get_layers_number();

    for(Index i = 0; i < layers_number; i++)
    {
        if(layers[i]->get_type() == Layer::Type::Recurrent)
        {
            return dynamic_cast<RecurrentLayer*>(layers[i]);
        }
    }

    ostringstream buffer;

    buffer << "OpenNN Exception: NeuralNetwork class.\n"
           << "RecurrentLayer* get_recurrent_layer() const method.\n"
           << "No recurrent layer in neural network.\n";

    throw runtime_error(buffer.str());
}


/// Returns true if messages from this class are displayed on the screen, or false if messages
/// from this class are not displayed on the screen.

const bool& NeuralNetwork::get_display() const
{
    return display;
}


/// This method deletes all the pointers in the neural network.
/// It also sets the rest of the members to their default values.

void NeuralNetwork::set()
{
    inputs_names.resize(0);

    outputs_names.resize(0);

    delete_layers();

    set_default();
}


/// Sets a new neural network with a given neural network architecture.
/// It also sets the rest of the members to their default values.
/// @param architecture Architecture of the neural network.

void NeuralNetwork::set(const NeuralNetwork::ModelType& model_type, const Tensor<Index, 1>& architecture)
{
    delete_layers();

    if(architecture.size() <= 1) return;

    const Index size = architecture.size();

    const Index inputs_number = architecture[0];
    const Index outputs_number = architecture[size-1];

    inputs_names.resize(inputs_number);

    ScalingLayer2D* scaling_layer_2d = new ScalingLayer2D(inputs_number);
    add_layer(scaling_layer_2d);

    if(model_type == ModelType::Approximation)
    {
        for(Index i = 0; i < size-1; i++)
        {
            PerceptronLayer* perceptron_layer = new PerceptronLayer(architecture[i], architecture[i+1]);
            perceptron_layer->set_name("perceptron_layer_" + to_string(i+1));

            add_layer(perceptron_layer);
            if(i == size-2) perceptron_layer->set_activation_function(PerceptronLayer::ActivationFunction::Linear);
        }

        UnscalingLayer* unscaling_layer = new UnscalingLayer(outputs_number);

        add_layer(unscaling_layer);

        BoundingLayer* bounding_layer = new BoundingLayer(outputs_number);

        add_layer(bounding_layer);
    }
    else if(model_type == ModelType::Classification || model_type == ModelType::TextClassification)
    {

        for(Index i = 0; i < size-2; i++)
        {
            PerceptronLayer* perceptron_layer = new PerceptronLayer(architecture[i], architecture[i+1]);

            perceptron_layer->set_name("perceptron_layer_" + to_string(i+1));

            add_layer(perceptron_layer);
        }

        ProbabilisticLayer* probabilistic_layer = new ProbabilisticLayer(architecture[size-2], architecture[size-1]);

        add_layer(probabilistic_layer);
    }
    else if(model_type == ModelType::Forecasting)
    {
        LongShortTermMemoryLayer* long_short_term_memory_layer = new LongShortTermMemoryLayer(architecture[0], architecture[1]);
        // RecurrentLayer* recurrent_layer = new RecurrentLayer(architecture[0], architecture[1]);

        add_layer(long_short_term_memory_layer);

        for(Index i = 1 ; i < size-1 ; i++)
        {
            PerceptronLayer* perceptron_layer = new PerceptronLayer(architecture[i], architecture[i+1]);

            perceptron_layer->set_name("perceptron_layer_" + to_string(i+1));

            add_layer(perceptron_layer);

            if(i == size-2) perceptron_layer->set_activation_function(PerceptronLayer::ActivationFunction::Linear);
        }

        UnscalingLayer* unscaling_layer = new UnscalingLayer(architecture[size-1]);

        add_layer(unscaling_layer);

        BoundingLayer* bounding_layer = new BoundingLayer(outputs_number);

        add_layer(bounding_layer);
    }
    else if(model_type == ModelType::ImageClassification)
    {
        // Use the set mode build specifically for image classification
    }
    else if(model_type == ModelType::AutoAssociation)
    {
        const Index mapping_neurons_number = 10;
        const Index bottle_neck_neurons_number = architecture[1];
        const Index target_variables_number = architecture[2];

        PerceptronLayer *mapping_layer = new PerceptronLayer(architecture[0], mapping_neurons_number, PerceptronLayer::ActivationFunction::HyperbolicTangent);
        mapping_layer->set_name("mapping_layer");
        PerceptronLayer *bottle_neck_layer = new PerceptronLayer(mapping_neurons_number, bottle_neck_neurons_number, PerceptronLayer::ActivationFunction::Linear);
        bottle_neck_layer->set_name("bottle_neck_layer");
        PerceptronLayer *demapping_layer = new PerceptronLayer(bottle_neck_neurons_number, mapping_neurons_number, PerceptronLayer::ActivationFunction::HyperbolicTangent);
        demapping_layer->set_name("demapping_layer");
        PerceptronLayer *output_layer = new PerceptronLayer(mapping_neurons_number, target_variables_number, PerceptronLayer::ActivationFunction::Linear);
        output_layer->set_name("output_layer");
        UnscalingLayer *unscaling_layer = new UnscalingLayer(target_variables_number);

        add_layer(mapping_layer);
        add_layer(bottle_neck_layer);
        add_layer(demapping_layer);
        add_layer(output_layer);
        add_layer(unscaling_layer);
    }

    outputs_names.resize(outputs_number);

    set_default();
}


void NeuralNetwork::set(const NeuralNetwork::ModelType& model_type, const initializer_list<Index>& architecture_list)
{
    Tensor<Index, 1> architecture(architecture_list.size());
    architecture.setValues(architecture_list);

    set_model_type(model_type);

    set(model_type, architecture);
}


/// Sets a new neural network with a given convolutional neural network architecture (CNN).
/// It also sets the rest of the members to their default values.
/// @param input_variables_dimensions Define the dimensions of the input varibales.
/// @param blocks_number Number of blocks.
/// @param filters_dimensions Architecture of the neural network.
/// @param outputs_number Architecture of the neural network.

void NeuralNetwork::set(const Tensor<Index, 1>& input_variables_dimensions,
                        const Index& blocks_number,
                        const Tensor<Index, 1>& filters_dimensions,
                        const Index& outputs_number)
{
    delete_layers();

    ScalingLayer4D* scaling_layer = new ScalingLayer4D(input_variables_dimensions);
    add_layer(scaling_layer);

    dimensions outputs_dimensions = scaling_layer->get_outputs_dimensions();  

//    for(Index i = 0; i < blocks_number; i++)
//    {
        // Check convolutional
        //ConvolutionalLayer* convolutional_layer = new ConvolutionalLayer(outputs_dimensions, filters_dimensions);
        //convolutional_layer->set_name("convolutional_layer_1" /* + to_string(1) */); // This change the initial name of the table.

    //add_layer(convolutional_layer);
    //outputs_dimensions = convolutional_layer->get_outputs_dimensions();

//        // Pooling layer 1

//        PoolingLayer* pooling_layer = new PoolingLayer(outputs_dimensions);
//        pooling_layer->set_name("pooling_layer_" + to_string(i+1));

//        add_layer(pooling_layer);

//        outputs_dimensions = pooling_layer->get_outputs_dimensions();


    FlattenLayer* flatten_layer = new FlattenLayer(outputs_dimensions);
    add_layer(flatten_layer);

    outputs_dimensions = flatten_layer->get_outputs_dimensions();

    const Index product = outputs_dimensions[0] * outputs_dimensions[1] * outputs_dimensions[2] * outputs_dimensions[3];

    PerceptronLayer* perceptron_layer = new PerceptronLayer(product, 3);
    perceptron_layer->set_name("perceptron_layer_1");
    add_layer(perceptron_layer);

    const Index perceptron_layer_outputs = perceptron_layer->get_neurons_number();

    ProbabilisticLayer* probabilistic_layer = new ProbabilisticLayer(perceptron_layer_outputs, outputs_number);
    add_layer(probabilistic_layer);
}


/// Sets the neural network members by loading them from an XML file.
/// @param file_name Neural network XML file_name.

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
    {
        set_model_type(ModelType::Approximation);
    }
    else if(new_model_type == "Classification")
    {
        set_model_type(ModelType::Classification);
    }
    else if(new_model_type == "Forecasting")
    {
        set_model_type(ModelType::Forecasting);
    }
    else if(new_model_type == "ImageClassification")
    {
        set_model_type(ModelType::ImageClassification);
    }
    else if(new_model_type == "TextClassification")
    {
        set_model_type(ModelType::TextClassification);
    }
    else if(new_model_type == "AutoAssociation")
    {
        set_model_type(ModelType::AutoAssociation);
    }
    else
    {
        const string message =
                "Neural Network class exception:\n"
                "void set_model_type_string(const string&)\n"
                "Unknown project type: " + new_model_type + "\n";

        throw runtime_error(message);
    }
}


/// Sets the names of inputs in neural network
/// @param new_inputs_names Tensor with the new names of inputs.

void NeuralNetwork::set_inputs_names(const Tensor<string, 1>& new_inputs_names)
{
    inputs_names = new_inputs_names;
}


/// Sets the names of outputs in neural network.
/// @param new_outputs_names Tensor with the new names of outputs.

void NeuralNetwork::set_outputs_names(const Tensor<string, 1>& new_outputs_names)
{
    outputs_names = new_outputs_names;
}


/// Sets the new inputs number of this neural network object.
/// @param new_inputs_number Number of inputs.

void NeuralNetwork::set_inputs_number(const Index& new_inputs_number)
{
#ifdef OPENNN_DEBUG

    if(new_inputs_number == 0)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: NeuralNetwork class.\n"
               << "void set_inputs_number(const Index&) method.\n"
               << "The number of inputs (" << new_inputs_number << ") must be greater than 0.\n";

        throw runtime_error(buffer.str());
    }

#endif

    inputs_names.resize(new_inputs_number);

    if(has_scaling_layer())
    {
        ScalingLayer2D* scaling_layer_2d = get_scaling_layer_2d();

        scaling_layer_2d->set_inputs_number(new_inputs_number);
    }

    const Index trainable_layers_number = get_trainable_layers_number();
    Tensor<Layer*, 1> trainable_layers = get_trainable_layers();

    if(trainable_layers_number > 0)
    {
        trainable_layers[0]->set_inputs_number(new_inputs_number);
    }
}


/// Sets the new inputs number of this neural network object.
/// @param inputs Boolean vector containing the number of inputs.

void NeuralNetwork::set_inputs_number(const Tensor<bool, 1>& inputs)
{
    if(layers.dimension(0) == 0) return;

    Index new_inputs_number = 0;

    for(Index i = 0; i < inputs.dimension(0); i++)
    {
        if(inputs(i)) new_inputs_number++;
    }

    set_inputs_number(new_inputs_number);
}


/// Sets those members which are not pointer to their default values.

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
    {
        layers(i)->set_threads_number(new_threads_number);
    }
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


void NeuralNetwork::set_layer_inputs_indices(const string& layer_name, const Tensor<string, 1>& new_layer_inputs_names)
{
    const Index layer_index = get_layer_index(layer_name);

    const Index size = new_layer_inputs_names.size();

    Tensor<Index, 1> new_layer_inputs_indices(size);

    for(Index i = 0; i < size; i++)
    {
        new_layer_inputs_indices(i) = get_layer_index(new_layer_inputs_names(i));
    }

    layers_inputs_indices(layer_index) = new_layer_inputs_indices;
}


void NeuralNetwork::set_layer_inputs_indices(const string& layer_name, const initializer_list<string>& new_layer_inputs_names_list)
{
    Tensor<string, 1> new_layer_inputs_names(new_layer_inputs_names_list.size());
    new_layer_inputs_names.setValues(new_layer_inputs_names_list);

    set_layer_inputs_indices(layer_name, new_layer_inputs_names);
}


void NeuralNetwork::set_layer_inputs_indices(const string& layer_name, const string& new_layer_inputs_name)
{
    const Index layer_index = get_layer_index(layer_name);

    Tensor<Index, 1> new_layer_inputs_indices(1);

    new_layer_inputs_indices(0) = get_layer_index(new_layer_inputs_name);

    layers_inputs_indices(layer_index) = new_layer_inputs_indices;
}


PerceptronLayer* NeuralNetwork::get_first_perceptron_layer() const
{
    const Index layers_number = get_layers_number();

    for(Index i = 0; i < layers_number; i++)
    {
        if(layers(i)->get_type() == Layer::Type::Perceptron)
        {
            return static_cast<PerceptronLayer*>(layers[i]);
        }
    }

    return nullptr;
}


/// Returns the number of inputs to the neural network.

Index NeuralNetwork::get_inputs_number() const
{
    if(layers.dimension(0) != 0)
    {
        return layers(0)->get_inputs_number();
    }

    return 0;
}


Index NeuralNetwork::get_outputs_number() const
{
    if(layers.size() > 0)
    {
        const Layer* last_layer = layers[layers.size() - 1];

        const dimensions outputs_dimensions = last_layer->get_outputs_dimensions();

        Index outputs_number = 1;

        for (Index i = 0; i < static_cast<Index>(outputs_dimensions.size()); i++)    outputs_number *= outputs_dimensions[i];

        return outputs_number;
    }

    return 0;
}


dimensions NeuralNetwork::get_outputs_dimensions() const
{
    if (layers.size() > 0)
    {
        const Layer* last_layer = layers[layers.size() - 1];

        return last_layer->get_outputs_dimensions();
    }

    return {};
}


Tensor<Index, 1> NeuralNetwork::get_trainable_layers_neurons_numbers() const
{
    const Index trainable_layers_number = get_trainable_layers_number();

    Tensor<Index, 1> layers_neurons_number(trainable_layers_number);

    Index count = 0;

    for(Index i = 0; i < layers.size(); i++)
    {
        if(layers(i)->get_type() != Layer::Type::Scaling2D
        && layers(i)->get_type() != Layer::Type::Unscaling
        && layers(i)->get_type() != Layer::Type::Bounding)
        {
            layers_neurons_number(count) = layers[i]->get_neurons_number();

            count++;
        }
    }

    return layers_neurons_number;
}


Tensor<Index, 1> NeuralNetwork::get_trainable_layers_inputs_numbers() const
{
    const Index trainable_layers_number = get_trainable_layers_number();

    Tensor<Index, 1> layers_neurons_number(trainable_layers_number);

    Index count = 0;

    for(Index i = 0; i < layers.size(); i++)
    {
        if(layers(i)->get_type() != Layer::Type::Scaling2D
        && layers(i)->get_type() != Layer::Type::Unscaling
        && layers(i)->get_type() != Layer::Type::Bounding)
        {
            layers_neurons_number(count) = layers[i]->get_inputs_number();

            count++;
        }
    }

    return layers_neurons_number;
}


/// Returns a vector with the architecture of the neural network.
/// The elements of this vector are as follows:
/// <UL>
/// <LI> Number of scaling neurons (if there is a scaling layer).</LI>
/// <LI> Multilayer perceptron architecture(if there is a neural network).</LI>
/// <LI> Number of unscaling neurons (if there is an unscaling layer).</LI>
/// <LI> Number of probabilistic neurons (if there is a probabilistic layer).</LI>
/// <LI> Number of bounding neurons (if there is a bounding layer).</LI>
/// </UL>

Tensor<Index, 1> NeuralNetwork::get_architecture() const
{
    const Index layers_number = get_layers_number();

    Tensor<Index, 1> architecture(layers_number);

    const Index inputs_number = get_inputs_number();

    if(inputs_number == 0) return architecture;

    if(layers_number > 0)
    {
        for(Index i = 0; i < layers_number; i++)
        {
            architecture(i) = layers(i)->get_neurons_number();
        }
    }

    return architecture;
}


/// Returns the number of parameters in the neural network
/// The number of parameters is the sum of all the neural network parameters (biases and synaptic weights).

//Index NeuralNetwork::get_parameters_number() const
//{
//    cout << "----- get_parameters_number -----" << endl;

//    const Tensor<Layer*, 1> trainable_layers = get_trainable_layers();

//    Index parameters_number = 0;

//    for(Index i = 0; i < trainable_layers.size(); i++)
//    {
//        parameters_number += trainable_layers[i]->get_parameters_number();
//    }

//    return parameters_number;
//}

Index NeuralNetwork::get_parameters_number() const
{
    const Tensor<Layer*, 1> trainable_layers = get_trainable_layers();

    Index parameters_number = 0;

    for(Index i = 0; i < trainable_layers.size(); i++)
    {
        if(trainable_layers[i] == nullptr)
        {
            cout << "Layer " << i << " is nullptr." << endl;
        }
        else
        {
            parameters_number += trainable_layers[i]->get_parameters_number();
        }
    }

    return parameters_number;
}


/// Returns the values of the parameters in the neural network as a single vector.
/// This contains all the neural network parameters (biases and synaptic weights).

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

        /// @todo use memcpy

        for(Index j = 0; j < layer_parameters.size(); j++)
        {
            parameters(j + position) = layer_parameters(j);
        }

        position += layer_parameters.size();
    }

    return parameters;
}


Tensor<Index, 1> NeuralNetwork::get_layers_parameters_numbers() const
{
    const Index layers_number = get_layers_number();

    Tensor<Index, 1> layers_parameters_number(layers_number);

    for (Index i = 0; i < layers_number; i++)
    {
        layers_parameters_number[i] = layers(i)->get_parameters_number();
    }

    return layers_parameters_number;
}


Tensor<Index, 1> NeuralNetwork::get_trainable_layers_parameters_numbers() const
{
    const Index trainable_layers_number = get_trainable_layers_number();

    const Tensor<Layer*, 1> trainable_layers = get_trainable_layers();

    Tensor<Index, 1> trainable_layers_parameters_number(trainable_layers_number);

    for(Index i = 0; i < trainable_layers_number; i++)
    {
        trainable_layers_parameters_number[i] = trainable_layers[i]->get_parameters_number();
    }

    return trainable_layers_parameters_number;
}


/// Sets all the parameters(biases and synaptic weights) from a single vector.
/// @param new_parameters New set of parameter values.

void NeuralNetwork::set_parameters(const Tensor<type, 1>& new_parameters) const
{
    const Index trainable_layers_number = get_trainable_layers_number();

    const Tensor<Layer*, 1> trainable_layers = get_trainable_layers();

    const Tensor<Index, 1> trainable_layers_parameters_numbers = get_trainable_layers_parameters_numbers();

    Index index = 0;

    for(Index i = 0; i < trainable_layers_number; i++)
    {
        trainable_layers(i)->set_parameters(new_parameters, index);

        index += trainable_layers_parameters_numbers(i);
    }
}


/// Sets a new display value.
/// If it is set to true messages from this class are displayed on the screen;
/// if it is set to false messages from this class are not displayed on the screen.
/// @param new_display Display value.

void NeuralNetwork::set_display(const bool& new_display)
{
    display = new_display;
}


/// Returns the number of layers in the neural network.
/// That includes perceptron, scaling, unscaling, inputs trending, outputs trending, bounding, probabilistic or conditions layers.

Index NeuralNetwork::get_layers_number() const
{
    return layers.size();
}


Tensor<Index, 1> NeuralNetwork::get_layers_neurons_numbers() const
{
    Tensor<Index, 1> layers_neurons_number(layers.size());

    for(Index i = 0; i < layers.size(); i++)
    {
        layers_neurons_number(i) = layers[i]->get_neurons_number();
    }

    return layers_neurons_number;
}


Index NeuralNetwork::get_trainable_layers_number() const
{
    const Index layers_number = get_layers_number();

    Index count = 0;

    for(Index i = 0; i < layers_number; i++)
    {
        if(layers(i)->get_type() != Layer::Type::Scaling2D
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

    for(Index i = 0; i < layers_number; i++)
    {
        if(layers(i)->get_type() == Layer::Type::Perceptron)
        {
            count++;
        }
    }

    return count;
}


Index NeuralNetwork::get_probabilistic_layers_number() const
{
    const Index layers_number = get_layers_number();

    Index count = 0;

    for(Index i = 0; i < layers_number; i++)
    {
        if(layers(i)->get_type() == Layer::Type::Probabilistic)
        {
            count++;
        }
    }

    return count;
}


Index NeuralNetwork::get_long_short_term_memory_layers_number() const
{
    const Index layers_number = get_layers_number();

    Index count = 0;

    for(Index i = 0; i < layers_number; i++)
    {
        if(layers(i)->get_type() == Layer::Type::LongShortTermMemory)
        {
            count++;
        }
    }

    return count;
}


Index NeuralNetwork::get_flatten_layers_number() const
{
    const Index layers_number = get_layers_number();

    Index count = 0;

    for(Index i = 0; i < layers_number; i++)
    {
        if(layers(i)->get_type() == Layer::Type::Flatten)
        {
            count++;
        }
    }

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

    for(Index i = 0; i < layers_number; i++)
    {
        if(layers(i)->get_type() == Layer::Type::Pooling)
        {
            count++;
        }
    }

    return count;
}


Index NeuralNetwork::get_recurrent_layers_number() const
{
    const Index layers_number = get_layers_number();

    Index count = 0;

    for(Index i = 0; i < layers_number; i++)
    {
        if(layers(i)->get_type() == Layer::Type::Recurrent)
        {
            count++;
        }
    }

    return count;
}


bool NeuralNetwork::is_input_layer(const Tensor<Index, 1>& layer_inputs_indices) const
{
    for (Index i = 0; i < layer_inputs_indices.size(); i++)
        if (layer_inputs_indices(i) == -1) return true;

    return false;
}

bool NeuralNetwork::is_context_layer(const Tensor<Index, 1>& layer_inputs_indices) const
{
    for (Index i = 0; i < layer_inputs_indices.size(); i++)
        if (layer_inputs_indices(i) == -2) return true;

    return false;
}


/// Initializes all the biases and synaptic weights with a given value.

void NeuralNetwork::set_parameters_constant(const type& value) const
{
    const Index trainable_layers_number = get_trainable_layers_number();

    const Tensor<Layer*, 1> trainable_layers = get_trainable_layers();

    for(Index i = 0; i < trainable_layers_number; i++)
    {
        trainable_layers[i]->set_parameters_constant(value);
    }
}


/// Initializes all the parameters in the newtork(biases and synaptic weiths + independent
/// parameters) at random with values comprised between a given minimum and a given maximum values.
/// @param minimum Minimum initialization value.
/// @param maximum Maximum initialization value.

void NeuralNetwork::set_parameters_random() const
{
    const Index layers_number = get_layers_number();

    Tensor<Layer*, 1> layers = get_layers();

    for(Index i = 0; i < layers_number; i++)
    {
        layers[i]->set_parameters_random();

    }
}


/// Returns the norm of the vector of parameters.

type NeuralNetwork::calculate_parameters_norm() const
{
    const Tensor<type, 1> parameters = get_parameters();

    Tensor<type, 0> parameters_norm;

    parameters_norm.device(*thread_pool_device) = parameters.square().sum().sqrt();

    return parameters_norm(0);
}


/// Calculates the forward propagation in the neural network.
/// @param batch Batch of data set that contains the inputs and targets to be trained.
/// @param foward_propagation Is a NeuralNetwork class structure where save the necessary parameters of forward propagation.

void NeuralNetwork::forward_propagate(const Tensor<pair<type*, dimensions>, 1>& inputs_pair,
                                      ForwardPropagation& forward_propagation,
                                      const bool& is_training) const
{
    const Index layers_number = get_layers_number();

    const Tensor<Layer*, 1> layers = get_layers();

    const Tensor<Tensor<Index, 1>, 1> layers_inputs_indices = get_layers_inputs_indices();

    const Index first_trainable_layer_index = get_first_trainable_layer_index();
    const Index last_trainable_layer_index = get_last_trainable_layer_index();

    Index first_layer_index;
    Index last_layer_index;

    if (is_training)
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
    
    for (Index i = first_layer_index; i <= last_layer_index; i++)
    {
        if (i == first_layer_index || is_input_layer(layers_inputs_indices(i)))
        {
            layer_inputs.resize(1);

            layer_inputs(0) = inputs_pair(0);
        }
        else if (is_context_layer(layers_inputs_indices(i)))
        {
            layer_inputs.resize(1);

            layer_inputs(0) = inputs_pair(1);
        }
        else
        {
            layer_inputs.resize(layers_inputs_indices(i).size());

            for (Index j = 0; j < layers_inputs_indices(i).size(); j++)
            {
                layer_inputs(j) = forward_propagation.layers(layers_inputs_indices(i)(j))->get_outputs_pair();
            }
        }
        
        layers(i)->forward_propagate(layer_inputs,
                                     forward_propagation.layers(i),
                                     is_training);
    }
}


/// Calculates the forward propagation in the neural network.
/// @param batch Batch of data set that contains the inputs and targets to be trained.
/// @param parameters Parameters of neural network.
/// @param foward_propagation Is a NeuralNetwork class structure where save the necessary parameters of forward propagation.

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


/// Calculates the outputs vector from the neural network in response to an inputs vector.
/// The activity for that is the following:
/// <ul>
/// <li> Check inputs range.
/// <li> Calculate scaled inputs.
/// <li> Calculate forward propagation.
/// <li> Calculate unscaled outputs.
/// <li> Apply boundary conditions.
/// <li> Calculate bounded outputs.
/// </ul>
/// @param inputs Set of inputs to the neural network.

Tensor<type, 2> NeuralNetwork::calculate_outputs(const Tensor<type, 2>& inputs)
{
    const Index batch_samples_number = inputs.dimension(0);
    const Index inputs_number = inputs.dimension(1);
    const Index outputs_number = get_outputs_number();

    ForwardPropagation neural_network_forward_propagation(batch_samples_number, this);

    const pair<type*, dimensions> inputs_pair((type*)inputs.data(), {{batch_samples_number, inputs_number}});

    forward_propagate(tensor_wrapper(inputs_pair), neural_network_forward_propagation);

    const Index layers_number = get_layers_number();

    if(layers_number == 0) return Tensor<type, 2>();
    
    const pair<type*, dimensions> outputs_pair = neural_network_forward_propagation.layers(layers_number - 1)->get_outputs_pair();

    const TensorMap<Tensor<type, 2>> outputs(outputs_pair.first, outputs_pair.second[0], outputs_pair.second[1]);

    return outputs;
}


Tensor<type, 2> NeuralNetwork::calculate_outputs(const Tensor<type, 4>& inputs)
{
    const Index batch_samples_number = inputs.dimension(0);
    const Index outputs_number = get_outputs_number();

    ForwardPropagation neural_network_forward_propagation(batch_samples_number, this);

    const pair<type*, dimensions> inputs_pair((type*)inputs.data(), { {inputs.dimension(0), inputs.dimension(1), inputs.dimension(2), inputs.dimension(3)} });

    forward_propagate(tensor_wrapper(inputs_pair), neural_network_forward_propagation);

    const Index layers_number = get_layers_number();

    if (layers_number == 0) return Tensor<type, 2>();

    const pair<type*, dimensions> outputs_pair = neural_network_forward_propagation.layers(layers_number - 1)->get_outputs_pair();

    const TensorMap<Tensor<type, 2>> outputs(outputs_pair.first, outputs_pair.second[0], outputs_pair.second[1]);

    return outputs;
}

/// Calculates the input data necessary to compute the output data from the neural network in some direction.
/// @param direction Input index (must be between 0 and number of inputs - 1).
/// @param point Input point through the directional input passes.
/// @param minimum Minimum value of the input with the above index.
/// @param maximum Maximum value of the input with the above index.
/// @param points_number Number of points in the directional input data set.

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
        {
            directional_inputs(i,j) = inputs(j);
        }
    }

    return directional_inputs;
}


/// For each layer: inputs, neurons, activation function.

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

        if(layer_type == "Perceptron")
        {
            const PerceptronLayer* perceptron_layer = static_cast<PerceptronLayer*>(trainable_layers(i));

            information(i,2) = perceptron_layer->write_activation_function();
        }
        else if(layer_type == "Probabilistic")
        {
            const ProbabilisticLayer* probabilistic_layer = static_cast<ProbabilisticLayer*>(trainable_layers(i));

            information(i,2) = probabilistic_layer->write_activation_function();
        }
        else if(layer_type == "LongShortTermMemory")
        {
            const LongShortTermMemoryLayer* long_short_term_memory_layer = static_cast<LongShortTermMemoryLayer*>(trainable_layers(i));

            information(i,2) = long_short_term_memory_layer->write_activation_function();
        }
        else if(layer_type == "Recurrent")
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


/// For each perceptron layer: inputs, neurons, activation function

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

        if(layer_type == "Perceptron")
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


/// For each probabilistic layer: inputs, neurons, activation function

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

        if(layer_type == "Probabilistic")
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


/// Serializes the neural network object into an XML document of the TinyXML library without keeping the DOM tree in memory.
/// See the OpenNN manual for more information about the format of this document.

void NeuralNetwork::write_XML(tinyxml2::XMLPrinter& file_stream) const
{
    ostringstream buffer;

    file_stream.OpenElement("NeuralNetwork");

    // Inputs

    file_stream.OpenElement("Inputs");

    // Inputs number

    file_stream.OpenElement("InputsNumber");

    buffer.str("");
    //    buffer << get_inputs_number();
    buffer << inputs_names.size();

    file_stream.PushText(buffer.str().c_str());

    file_stream.CloseElement();

    // Inputs names

    for(Index i = 0; i < inputs_names.size(); i++)
    {
        file_stream.OpenElement("Input");

        file_stream.PushAttribute("Index", to_string(i+1).c_str());

        file_stream.PushText(inputs_names[i].c_str());

        file_stream.CloseElement();
    }

    // Inputs (end tag)

    file_stream.CloseElement();

    // Layers

    file_stream.OpenElement("Layers");

    // Layers number

    //cout << "Layers types" << endl;

    file_stream.OpenElement("LayersTypes");

    buffer.str("");

    for(Index i = 0; i < layers.size(); i++)
    {
        buffer << layers[i]->get_type_string();
        if(i != (layers.size()-1)) buffer << " ";
    }

    file_stream.PushText(buffer.str().c_str());

    file_stream.CloseElement();

    // Layers information

    for(Index i = 0; i < layers.size(); i++)
    {
        layers[i]->write_XML(file_stream);
    }

    // Layers inputs indices

    cout << "Layers Inputs Indices" << endl;

    file_stream.OpenElement("LayersInputsIndices");

    for (Index i = 0; i < layers_inputs_indices.size(); i++)
    {
        file_stream.OpenElement("LayerInputsIndices");

        file_stream.PushAttribute("LayerIndex", to_string(i + 1).c_str());

        const Tensor<Index, 1>& indices = layers_inputs_indices(i);
        buffer.str("");
        for (Index j = 0; j < indices.size(); j++)
        {
            buffer << indices(j);
            if (j != (indices.size() - 1)) buffer << " ";
        }

        file_stream.PushText(buffer.str().c_str());

        file_stream.CloseElement();
    }

    file_stream.CloseElement();

    // Layers (end tag)

    file_stream.CloseElement();

    // Ouputs

    file_stream.OpenElement("Outputs");

    // Outputs number

    //cout << "Outputs number" << endl;
    //const Index outputs_number = get_outputs_number();
    const Index outputs_number = outputs_names.size();

    file_stream.OpenElement("OutputsNumber");

    buffer.str("");
    buffer << outputs_number;

    file_stream.PushText(buffer.str().c_str());

    file_stream.CloseElement();

    // Outputs names

    //cout << "Outputs names" << endl;
    for(Index i = 0; i < outputs_names.size(); i++)
    {
        file_stream.OpenElement("Output");

        file_stream.PushAttribute("Index", to_string(i+1).c_str());

        file_stream.PushText(outputs_names[i].c_str());

        file_stream.CloseElement();
    }

    //Outputs (end tag)

    file_stream.CloseElement();

    // Neural network (end tag)

    file_stream.CloseElement();
}


/// Deserializes a TinyXML document into this neural network object.
/// @param document XML document containing the member data.

void NeuralNetwork::from_XML(const tinyxml2::XMLDocument& document)
{
    ostringstream buffer;

    const tinyxml2::XMLElement* root_element = document.FirstChildElement("NeuralNetwork");

    if(!root_element)
    {
        buffer << "OpenNN Exception: NeuralNetwork class.\n"
               << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
               << "Neural network element is nullptr.\n";

        throw runtime_error(buffer.str());
    }
    
    // Inputs
    {
        const tinyxml2::XMLElement* element = root_element->FirstChildElement("Inputs");

        if(element)
        {
            tinyxml2::XMLDocument inputs_document;
            tinyxml2::XMLNode* element_clone;

            element_clone = element->DeepClone(&inputs_document);

            inputs_document.InsertFirstChild(element_clone);

            inputs_from_XML(inputs_document);
        }
    }
    
    // Layers
    {
        const tinyxml2::XMLElement* element = root_element->FirstChildElement("Layers");

        if(element)
        {
            tinyxml2::XMLDocument layers_document;
            tinyxml2::XMLNode* element_clone;

            element_clone = element->DeepClone(&layers_document);

            layers_document.InsertFirstChild(element_clone);

            layers_from_XML(layers_document);
        }
    }
    
    // Outputs
    {
        const tinyxml2::XMLElement* element = root_element->FirstChildElement("Outputs");

        if(element)
        {
            tinyxml2::XMLDocument outputs_document;
            tinyxml2::XMLNode* element_clone;

            element_clone = element->DeepClone(&outputs_document);

            outputs_document.InsertFirstChild(element_clone);

            outputs_from_XML(outputs_document);
        }
    }
    
    // Display
    {
        const tinyxml2::XMLElement* element = root_element->FirstChildElement("Display");

        if(element)
        {
            const string new_display_string = element->GetText();

            try
            {
                set_display(new_display_string != "0");
            }
            catch(const exception& e)
            {
                cerr << e.what() << endl;
            }
        }
    }
}


void NeuralNetwork::inputs_from_XML(const tinyxml2::XMLDocument& document)
{
    ostringstream buffer;

    const tinyxml2::XMLElement* root_element = document.FirstChildElement("Inputs");

    if(!root_element)
    {
        buffer << "OpenNN Exception: NeuralNetwork class.\n"
               << "void inputs_from_XML(const tinyxml2::XMLDocument&) method.\n"
               << "Inputs element is nullptr.\n";

        throw runtime_error(buffer.str());
    }

    // Inputs number

    const tinyxml2::XMLElement* inputs_number_element = root_element->FirstChildElement("InputsNumber");

    if(!inputs_number_element)
    {
        buffer << "OpenNN Exception: NeuralNetwork class.\n"
               << "void inputs_from_XML(const tinyxml2::XMLDocument&) method.\n"
               << "Inputs number element is nullptr.\n";

        throw runtime_error(buffer.str());
    }

    Index new_inputs_number = 0;

    if(inputs_number_element->GetText())
    {
        new_inputs_number = Index(atoi(inputs_number_element->GetText()));

        set_inputs_number(new_inputs_number);
    }

    // Inputs names

    const tinyxml2::XMLElement* start_element = inputs_number_element;

    if(new_inputs_number > 0)
    {
        for(Index i = 0; i < new_inputs_number; i++)
        {
            const tinyxml2::XMLElement* input_element = start_element->NextSiblingElement("Input");
            start_element = input_element;

            if(input_element->Attribute("Index") != to_string(i+1))
            {
                buffer << "OpenNN Exception: NeuralNetwork class.\n"
                       << "void inputs_from_XML(const tinyxml2::XMLDocument&) method.\n"
                       << "Input index number (" << i+1 << ") does not match (" << input_element->Attribute("Item") << ").\n";

                throw runtime_error(buffer.str());
            }

            if(!input_element->GetText())
            {
                inputs_names(i) = "";
            }
            else
            {
                inputs_names(i) = input_element->GetText();
            }
        }
    }
}


void NeuralNetwork::layers_from_XML(const tinyxml2::XMLDocument& document)
{
    ostringstream buffer;

    const tinyxml2::XMLElement* root_element = document.FirstChildElement("Layers");

    if(!root_element)
    {
        buffer << "OpenNN Exception: NeuralNetwork class.\n"
               << "void layers_from_XML(const tinyxml2::XMLDocument&) method.\n"
               << "Layers element is nullptr.\n";

        throw runtime_error(buffer.str());
    }

    // Layers types

    const tinyxml2::XMLElement* layers_types_element = root_element->FirstChildElement("LayersTypes");

    if(!layers_types_element)
    {
        buffer << "OpenNN Exception: NeuralNetwork class.\n"
               << "void layers_from_XML(const tinyxml2::XMLDocument&) method.\n"
               << "Layers types element is nullptr.\n";

        throw runtime_error(buffer.str());
    }

    Tensor<string, 1> layers_types;

    if(layers_types_element->GetText())
    {
        layers_types = get_tokens(layers_types_element->GetText(), ' ');
    }

    // Add layers

    const tinyxml2::XMLElement* start_element = layers_types_element;

    for(Index i = 0; i < layers_types.size(); i++)
    {
        if(layers_types(i) == "Scaling2D")
        {
            ScalingLayer2D* scaling_layer = new ScalingLayer2D();

            const tinyxml2::XMLElement* scaling_element = start_element->NextSiblingElement("ScalingLayer2D");

            start_element = scaling_element;

            if(scaling_element)
            {
                tinyxml2::XMLDocument scaling_document;
                tinyxml2::XMLNode* element_clone;

                element_clone = scaling_element->DeepClone(&scaling_document);

                scaling_document.InsertFirstChild(element_clone);

                scaling_layer->from_XML(scaling_document);
            }

            add_layer(scaling_layer);
        }
        else if(layers_types(i) == "Scaling4D")
        {
            ScalingLayer4D* scaling_layer = new ScalingLayer4D();

            const tinyxml2::XMLElement* scaling_element = start_element->NextSiblingElement("ScalingLayer4D");

            start_element = scaling_element;

            if(scaling_element)
            {
                tinyxml2::XMLDocument scaling_document;
                tinyxml2::XMLNode* element_clone;

                element_clone = scaling_element->DeepClone(&scaling_document);

                scaling_document.InsertFirstChild(element_clone);

                scaling_layer->from_XML(scaling_document);
            }

            add_layer(scaling_layer);
        }
        //else if(layers_types(i) == "Convolutional")
        //{
        //    ConvolutionalLayer* convolutional_layer = new ConvolutionalLayer();

        //    const tinyxml2::XMLElement* convolutional_element = start_element->NextSiblingElement("ConvolutionalLayer");
        //    start_element = convolutional_element;

        //    if(convolutional_element)
        //    {
        //        tinyxml2::XMLDocument convolutional_document;
        //        tinyxml2::XMLNode* element_clone;

        //        element_clone = convolutional_element->DeepClone(&convolutional_document);

        //        convolutional_document.InsertFirstChild(element_clone);

        //        convolutional_layer->from_XML(convolutional_document);
        //    }

        //    add_layer(convolutional_layer);
        //}
        else if(layers_types(i) == "Perceptron")
        {
            PerceptronLayer* perceptron_layer = new PerceptronLayer();

            const tinyxml2::XMLElement* perceptron_element = start_element->NextSiblingElement("PerceptronLayer");
            start_element = perceptron_element;

            if(perceptron_element)
            {
                tinyxml2::XMLDocument perceptron_document;
                tinyxml2::XMLNode* element_clone;

                element_clone = perceptron_element->DeepClone(&perceptron_document);

                perceptron_document.InsertFirstChild(element_clone);

                perceptron_layer->from_XML(perceptron_document);
            }

            add_layer(perceptron_layer);
        }
        else if (layers_types(i) == "Perceptron3D")
        {
            PerceptronLayer3D* perceptron_layer_3d = new PerceptronLayer3D();

            const tinyxml2::XMLElement* perceptron_element = start_element->NextSiblingElement("PerceptronLayer3D");
            start_element = perceptron_element;

            if (perceptron_element)
            {
                tinyxml2::XMLDocument perceptron_document;
                tinyxml2::XMLNode* element_clone;

                element_clone = perceptron_element->DeepClone(&perceptron_document);

                perceptron_document.InsertFirstChild(element_clone);

                perceptron_layer_3d->from_XML(perceptron_document);
            }

            add_layer(perceptron_layer_3d);
        }
        else if(layers_types(i) == "Pooling")
        {
            PoolingLayer* pooling_layer = new PoolingLayer();

            const tinyxml2::XMLElement* pooling_element = start_element->NextSiblingElement("PoolingLayer");
            start_element = pooling_element;

            if(pooling_element)
            {
                tinyxml2::XMLDocument pooling_document;
                tinyxml2::XMLNode* element_clone;

                element_clone = pooling_element->DeepClone(&pooling_document);

                pooling_document.InsertFirstChild(element_clone);

                pooling_layer->from_XML(pooling_document);
            }

            add_layer(pooling_layer);
        }
        else if(layers_types(i) == "Probabilistic")
        {
            ProbabilisticLayer* probabilistic_layer = new ProbabilisticLayer();

            const tinyxml2::XMLElement* probabilistic_element = start_element->NextSiblingElement("ProbabilisticLayer");
            start_element = probabilistic_element;

            if(probabilistic_element)
            {
                tinyxml2::XMLDocument probabilistic_document;
                tinyxml2::XMLNode* element_clone;

                element_clone = probabilistic_element->DeepClone(&probabilistic_document);

                probabilistic_document.InsertFirstChild(element_clone);
                probabilistic_layer->from_XML(probabilistic_document);
            }

            add_layer(probabilistic_layer);
        }
        else if(layers_types(i) == "Probabilistic3D")
        {
            ProbabilisticLayer3D* probabilistic_layer_3d = new ProbabilisticLayer3D();

            const tinyxml2::XMLElement* probabilistic_element = start_element->NextSiblingElement("ProbabilisticLayer3D");
            start_element = probabilistic_element;

            if(probabilistic_element)
            {
                tinyxml2::XMLDocument probabilistic_document;
                tinyxml2::XMLNode* element_clone;

                element_clone = probabilistic_element->DeepClone(&probabilistic_document);

                probabilistic_document.InsertFirstChild(element_clone);
                probabilistic_layer_3d->from_XML(probabilistic_document);
            }

            add_layer(probabilistic_layer_3d);
        }
        else if(layers_types(i) == "LongShortTermMemory")
        {
            LongShortTermMemoryLayer* long_short_term_memory_layer = new LongShortTermMemoryLayer();

            const tinyxml2::XMLElement* long_short_term_memory_element = start_element->NextSiblingElement("LongShortTermMemoryLayer");
            start_element = long_short_term_memory_element;

            if(long_short_term_memory_element)
            {
                tinyxml2::XMLDocument long_short_term_memory_document;
                tinyxml2::XMLNode* element_clone;

                element_clone = long_short_term_memory_element->DeepClone(&long_short_term_memory_document);

                long_short_term_memory_document.InsertFirstChild(element_clone);

                long_short_term_memory_layer->from_XML(long_short_term_memory_document);
            }

            add_layer(long_short_term_memory_layer);
        }
        else if(layers_types(i) == "Recurrent")
        {
            RecurrentLayer* recurrent_layer = new RecurrentLayer();

            const tinyxml2::XMLElement* recurrent_element = start_element->NextSiblingElement("RecurrentLayer");
            start_element = recurrent_element;

            if(recurrent_element)
            {
                tinyxml2::XMLDocument recurrent_document;
                tinyxml2::XMLNode* element_clone;

                element_clone = recurrent_element->DeepClone(&recurrent_document);

                recurrent_document.InsertFirstChild(element_clone);

                recurrent_layer->from_XML(recurrent_document);
            }

            add_layer(recurrent_layer);
        }
        else if(layers_types(i) == "Unscaling")
        {
            UnscalingLayer* unscaling_layer = new UnscalingLayer();

            const tinyxml2::XMLElement* unscaling_element = start_element->NextSiblingElement("UnscalingLayer");
            start_element = unscaling_element;

            if(unscaling_element)
            {
                tinyxml2::XMLDocument unscaling_document;
                tinyxml2::XMLNode* element_clone;

                element_clone = unscaling_element->DeepClone(&unscaling_document);

                unscaling_document.InsertFirstChild(element_clone);

                unscaling_layer->from_XML(unscaling_document);
            }

            add_layer(unscaling_layer);
        }
        else if(layers_types(i) == "Bounding")
        {
            BoundingLayer* bounding_layer = new BoundingLayer();

            const tinyxml2::XMLElement* bounding_element = start_element->NextSiblingElement("BoundingLayer");

            start_element = bounding_element;

            if(bounding_element)
            {
                tinyxml2::XMLDocument bounding_document;
                tinyxml2::XMLNode* element_clone;

                element_clone = bounding_element->DeepClone(&bounding_document);

                bounding_document.InsertFirstChild(element_clone);

                bounding_layer->from_XML(bounding_document);
            }

            add_layer(bounding_layer);
        }
        else if(layers_types(i) == "Embedding")
        {
            EmbeddingLayer* embedding_layer = new EmbeddingLayer();

            const tinyxml2::XMLElement* embedding_element = start_element->NextSiblingElement("EmbeddingLayer");

            start_element = embedding_element;

            if(embedding_element)
            {
                tinyxml2::XMLDocument embedding_document;
                tinyxml2::XMLNode* element_clone;

                element_clone = embedding_element->DeepClone(&embedding_document);

                embedding_document.InsertFirstChild(element_clone);

                embedding_layer->from_XML(embedding_document);
            }

            add_layer(embedding_layer);
        }
        else if(layers_types(i) == "MultiheadAttention")
        {
            MultiheadAttentionLayer* multihead_attention_layer = new MultiheadAttentionLayer();

            const tinyxml2::XMLElement* multihead_attention_element = start_element->NextSiblingElement("MultiheadAttentionLayer");

            start_element = multihead_attention_element;
            
            if(multihead_attention_element)
            {
                tinyxml2::XMLDocument multihead_attention_document;
                tinyxml2::XMLNode* element_clone;

                element_clone = multihead_attention_element->DeepClone(&multihead_attention_document);
                
                multihead_attention_document.InsertFirstChild(element_clone);

                multihead_attention_layer->from_XML(multihead_attention_document);
            }

            add_layer(multihead_attention_layer); 
        }
        else if(layers_types(i) == "Addition3D")
        {
            AdditionLayer3D* addition_layer_3d = new AdditionLayer3D();
            
            const tinyxml2::XMLElement* addition_element = start_element->NextSiblingElement("AdditionLayer3D");

            start_element = addition_element;

            if(addition_element)
            {
                tinyxml2::XMLDocument addition_document;
                tinyxml2::XMLNode* element_clone;

                element_clone = addition_element->DeepClone(&addition_document);

                addition_document.InsertFirstChild(element_clone);

                addition_layer_3d->from_XML(addition_document);
            }
            
            add_layer(addition_layer_3d);
        }
        else if(layers_types(i) == "Normalization3D")
        {
            NormalizationLayer3D* normalization_layer_3d = new NormalizationLayer3D();

            const tinyxml2::XMLElement* normalization_element = start_element->NextSiblingElement("NormalizationLayer3D");

            start_element = normalization_element;

            if(normalization_element)
            {
                tinyxml2::XMLDocument normalization_document;
                tinyxml2::XMLNode* element_clone;

                element_clone = normalization_element->DeepClone(&normalization_document);

                normalization_document.InsertFirstChild(element_clone);

                normalization_layer_3d->from_XML(normalization_document);
            }

            add_layer(normalization_layer_3d);
        }
    }

    // Layers inputs indices

    const tinyxml2::XMLElement* layers_inputs_indices_element = root_element->FirstChildElement("LayersInputsIndices");

    if (!layers_inputs_indices_element)
    {
        buffer << "OpenNN Exception: NeuralNetwork class.\n"
            << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
            << "LayersInputsIndices element is nullptr.\n";

        throw runtime_error(buffer.str());
    }

    layers_inputs_indices.resize(layers.size());

    for (const tinyxml2::XMLElement* layer_inputs_indices_element = layers_inputs_indices_element->FirstChildElement("LayerInputsIndices");
        layer_inputs_indices_element;
        layer_inputs_indices_element = layer_inputs_indices_element->NextSiblingElement("LayerInputsIndices"))
    {
        if (layer_inputs_indices_element->GetText())
        {
            Index layer_index = Index(stoi(layer_inputs_indices_element->Attribute("LayerIndex"))) - 1;
            const string indices_string = layer_inputs_indices_element->GetText();
            layers_inputs_indices(layer_index) = to_type_vector(indices_string, ' ').cast<Index>();
        }
    }
}


void NeuralNetwork::outputs_from_XML(const tinyxml2::XMLDocument& document)
{
    ostringstream buffer;

    const tinyxml2::XMLElement* root_element = document.FirstChildElement("Outputs");

    if(!root_element)
    {
        buffer << "OpenNN Exception: NeuralNetwork class.\n"
               << "void outputs_from_XML(const tinyxml2::XMLDocument&) method.\n"
               << "Outputs element is nullptr.\n";

        throw runtime_error(buffer.str());
    }

    // Outputs number
    
    const tinyxml2::XMLElement* outputs_number_element = root_element->FirstChildElement("OutputsNumber");

    if(!outputs_number_element)
    {
        buffer << "OpenNN Exception: NeuralNetwork class.\n"
               << "void outputs_from_XML(const tinyxml2::XMLDocument&) method.\n"
               << "Outputs number element is nullptr.\n";

        throw runtime_error(buffer.str());
    }

    Index new_outputs_number = 0;

    if(outputs_number_element->GetText())
    {
        new_outputs_number = Index(atoi(outputs_number_element->GetText()));
    }

    // Outputs names
    /*
    const tinyxml2::XMLElement* start_element = outputs_number_element;

    if(new_outputs_number > 0)
    {
        outputs_names.resize(new_outputs_number);

        for(Index i = 0; i < new_outputs_number; i++)
        {
            const tinyxml2::XMLElement* output_element = start_element->NextSiblingElement("Output");
            start_element = output_element;
            
            if(output_element->Attribute("Index") != to_string(i+1))
            {
                buffer << "OpenNN Exception: NeuralNetwork class.\n"
                       << "void outputs_from_XML(const tinyxml2::XMLDocument&) method.\n"
                       << "Output index number (" << i+1 << ") does not match (" << output_element->Attribute("Item") << ").\n";

                throw runtime_error(buffer.str());
            }
            
            if(!output_element->GetText())
            {
                outputs_names(i) = "";
            }
            else
            {
                outputs_names(i) = output_element->GetText();
            }
        }
    }
    */
}


/// Prints to the screen the most important information about the neural network object.

void NeuralNetwork::print() const
{
    cout << "Neural network" << endl;

    const Index layers_number = get_layers_number();

    cout << "Layers number: " << layers_number << endl;

    for(Index i = 0; i < layers_number; i++)
    {
        cout << "Layer " << i << ": " << layers[i]->get_neurons_number()
             << " " << layers[i]->get_type_string() << " neurons" << endl;
    }
}


/// Saves to an XML file the members of a neural network object.
/// @param file_name Name of neural network XML file.

void NeuralNetwork::save(const string& file_name) const
{
    FILE * file = fopen(file_name.c_str(), "w");

    if(file)
    {
        tinyxml2::XMLPrinter printer(file);
        write_XML(printer);
        fclose(file);
    }
}


/// Saves to a data file the parameters of a neural network object.
/// @param file_name Name of the parameters data file.

void NeuralNetwork::save_parameters(const string& file_name) const
{
    std::ofstream file(file_name.c_str());

    if(!file.is_open())
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: NeuralNetwork class.\n"
               << "void save_parameters(const string&) const method.\n"
               << "Cannot open parameters data file.\n";

        throw runtime_error(buffer.str());
    }

    const Tensor<type, 1> parameters = get_parameters();

    file << parameters << endl;

    // Close file

    file.close();
}


/// Loads from an XML file the members for this neural network object.
/// Please mind about the file format, which is specified in the User's Guide.
/// @param file_name Name of neural network XML file.

void NeuralNetwork::load(const string& file_name)
{
    set_default();

    tinyxml2::XMLDocument document;

    if(document.LoadFile(file_name.c_str()))
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: NeuralNetwork class.\n"
               << "void load(const string&) method.\n"
               << "Cannot load XML file " << file_name << ".\n";

        throw runtime_error(buffer.str());

    }

    from_XML(document);
}


/// Loads the neural network parameters from a data file.
/// The format of this file is just a sequence of numbers.
/// @param file_name Name of the parameters data file.

void NeuralNetwork::
load_parameters_binary(const string& file_name)
{

    std::ifstream file;

    file.open(file_name.c_str(), ios::binary);

    if(!file.is_open())
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: NeuralNetwork template.\n"
               << "void load_parameters_binary(const string&) method.\n"
               << "Cannot open binary file: " << file_name << "\n";

        throw runtime_error(buffer.str());
    }

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


/// Returns a string with the mathematical expression that represents the neural network.

string NeuralNetwork::write_expression() const
{

//    NeuralNetwork::ModelType model_type = get_model_type();

    const Index layers_number = get_layers_number();

    const Tensor<Layer*, 1> layers = get_layers();
    const Tensor<string, 1> layers_names = get_layers_names();

    Tensor<string, 1> outputs_names_vector;
    Tensor<string, 1> inputs_names_vector;
    inputs_names_vector = inputs_names;
    string aux_name = "";

    for(int i = 0; i < inputs_names.dimension(0); i++)
    {
        if(!inputs_names_vector[i].empty())
        {
            aux_name = inputs_names[i];
            inputs_names_vector(i) = replace_non_allowed_programming_expressions(aux_name);
        }
        else
        {
            inputs_names_vector(i) = "input_" + to_string(i);
        }
    }

    Index layer_neurons_number;

    Tensor<string, 1> scaled_inputs_names(inputs_names.dimension(0));
    Tensor<string, 1> unscaled_outputs_names(inputs_names.dimension(0));

    ostringstream buffer;

    for(Index i = 0; i < layers_number; i++)
    {
        if(i == layers_number-1)
        {
            outputs_names_vector = outputs_names;

            for(int j = 0; j < outputs_names.dimension(0); j++)
            {
                if(!outputs_names_vector[j].empty())
                {
                    aux_name = outputs_names[j];
                    outputs_names_vector(j) = replace_non_allowed_programming_expressions(aux_name);
                }
                else
                {
                    outputs_names_vector(j) = "output_" + to_string(i);
                }
            }
            buffer << layers[i]->write_expression(inputs_names_vector, outputs_names_vector) << endl;
        }
        else
        {
            layer_neurons_number = layers[i]->get_neurons_number();
            outputs_names_vector.resize(layer_neurons_number);

            for(Index j = 0; j < layer_neurons_number; j++)
            {
                if(layers_names(i) == "scaling_layer")
                {
                    aux_name = inputs_names(j);
                    outputs_names_vector(j) = "scaled_" + replace_non_allowed_programming_expressions(aux_name);
                    scaled_inputs_names(j) = outputs_names_vector(j);
                }
                else
                {
                    outputs_names_vector(j) =  layers_names(i) + "_output_" + to_string(j);
                }
            }
            buffer << layers[i]->write_expression(inputs_names_vector, outputs_names_vector) << endl;
            inputs_names_vector = outputs_names_vector;
            unscaled_outputs_names = inputs_names_vector;
        }
    }

    string expression = buffer.str();

    replace(expression, "+-", "-");

    return expression;
}


/// Returns a string with the c function of the expression represented by the neural network.

string NeuralNetwork::write_expression_c() const
{
    string aux = "";
    ostringstream buffer;
    ostringstream outputs_buffer;

    Tensor<string, 1> inputs =  get_inputs_names();
    Tensor<string, 1> outputs = get_outputs_names();
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

    buffer << "/*" << endl;
    buffer << "Artificial Intelligence Techniques SL\t" << endl;
    buffer << "artelnics@artelnics.com\t" << endl;
    buffer << "" << endl;
    buffer << "Your model has been exported to this c file." << endl;
    buffer << "You can manage it with the main method, where you \t" << endl;
    buffer << "can change the values of your inputs. For example:" << endl;
    buffer << "" << endl;

    buffer << "if we want to add these 3 values (0.3, 2.5 and 1.8)" << endl;
    buffer << "to our 3 inputs (Input_1, Input_2 and Input_1), the" << endl;
    buffer << "main program has to look like this:" << endl;
    buffer << "\t" << endl;
    buffer << "int main(){ " << endl;
    buffer << "\t" << "vector<float> inputs(3);"<< endl;
    buffer << "\t" << endl;
    buffer << "\t" << "const float asdas  = 0.3;" << endl;
    buffer << "\t" << "inputs[0] = asdas;"        << endl;
    buffer << "\t" << "const float input2 = 2.5;" << endl;
    buffer << "\t" << "inputs[1] = input2;"       << endl;
    buffer << "\t" << "const float input3 = 1.8;" << endl;
    buffer << "\t" << "inputs[2] = input3;"       << endl;
    buffer << "\t" << ". . ." << endl;
    buffer << "\n" << endl;
    buffer << "Inputs Names:" <<endl;

    Tensor<Tensor<string,1>, 1> inputs_outputs_buffer = fix_input_output_variables(inputs, outputs, buffer);

    for(Index i = 0; i < inputs_outputs_buffer(0).dimension(0);++i)
        inputs(i) = inputs_outputs_buffer(0)(i);

    for(Index i = 0; i < inputs_outputs_buffer(1).dimension(0);++i)
        outputs(i) = inputs_outputs_buffer(1)(i);

    buffer << inputs_outputs_buffer(2)[0];
    buffer << "*/" << endl;
    buffer << "\n" << endl;
    buffer << "#include <iostream>" << endl;
    buffer << "#include <vector>" << endl;
    buffer << "#include <math.h>" << endl;
    buffer << "#include <stdio.h>" << endl;
    buffer << "\n" << endl;
    buffer << "using namespace std;" << endl;
    buffer << "\n" << endl;

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
        {
            expression.erase(index, string::npos);
        }

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
        if(token.size() > 1 && token.back() == '{')
        {
            break;
        }
        if(token.size() > 1 && token.back() != ';')
        {
            token += ';';
        }
        push_back_string(tokens, token);
    }

    for(int i = 0; i < tokens.dimension(0); i++)
    {
        string t = tokens(i);
        string word = get_word_from_token(t);

        if(word.size() > 1 && !find_string_in_tensor(found_tokens, word))
        {
            push_back_string(found_tokens, word);
        }

    }

    string target_string0("Logistic");
    string target_string1("ReLU");
    string target_string4("ExponentialLinear");
    string target_string5("SELU");
    string target_string6("HardSigmoid");
    string target_string7("SoftPlus");
    string target_string8("SoftSign");

    for(int i = 0; i < tokens.dimension(0); i++)
    {
        string t = tokens(i);

        size_t substring_length0 = t.find(target_string0);
        size_t substring_length1 = t.find(target_string1);
        size_t substring_length4 = t.find(target_string4);
        size_t substring_length5 = t.find(target_string5);
        size_t substring_length6 = t.find(target_string6);
        size_t substring_length7 = t.find(target_string7);
        size_t substring_length8 = t.find(target_string8);

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
        buffer << "float Logistic (float x) {" << endl;
        buffer << "float z = 1/(1+exp(-x));" << endl;
        buffer << "return z;" << endl;
        buffer << "}" << endl;
        buffer << "\n" << endl;
    }

    if(ReLU)
    {
        buffer << "float ReLU(float x) {" << endl;
        buffer << "float z = max(0, x);" << endl;
        buffer << "return z;" << endl;
        buffer << "}" << endl;
        buffer << "\n" << endl;
    }

    if(ExpLinear)
    {
        buffer << "float ExponentialLinear(float x) {" << endl;
        buffer << "float z;" << endl;
        buffer << "float alpha = 1.67326;" << endl;
        buffer << "if(x>0){" << endl;
        buffer << "z = x;" << endl;
        buffer << "}else{" << endl;
        buffer << "z = alpha*(exp(x)-1);" << endl;
        buffer << "}" << endl;
        buffer << "return z;" << endl;
        buffer << "}" << endl;
        buffer << "\n" << endl;
    }

    if(SExpLinear)
    {
        buffer << "float SELU(float x) {" << endl;
        buffer << "float z;" << endl;
        buffer << "float alpha  = 1.67326;" << endl;
        buffer << "float lambda = 1.05070;" << endl;
        buffer << "if(x > 0){" << endl;
        buffer << "z = lambda*x;" << endl;
        buffer << "}else{" << endl;
        buffer << "z = lambda*alpha*(exp(x)-1);" << endl;
        buffer << "}" << endl;
        buffer << "return z;" << endl;
        buffer << "}" << endl;
        buffer << "\n" << endl;
    }

    if(HSigmoid)
    {
        buffer << "float HardSigmoid(float x) {" << endl;
        buffer << "float z = 1/(1+exp(-x));" << endl;
        buffer << "return z;" << endl;
        buffer << "}" << endl;
        buffer << "\n" << endl;
    }

    if(SoftPlus)
    {
        buffer << "float SoftPlus(float x) {" << endl;
        buffer << "float z = log(1+exp(x));" << endl;
        buffer << "return z;" << endl;
        buffer << "}" << endl;
        buffer << "\n" << endl;
    }

    if(SoftSign)
    {
        buffer << "float SoftSign(float x) {" << endl;
        buffer << "float z = x/(1+abs(x));" << endl;
        buffer << "return z;" << endl;
        buffer << "}" << endl;
        buffer << "\n" << endl;
    }

    if(LSTM_number>0)
    {
        for(int i = 0; i < found_tokens.dimension(0); i++)
        {
            string token = found_tokens(i);

            if(token.find("cell_state") == 0)
            {
                cell_states_counter += 1;
            }

            if(token.find("hidden_state") == 0)
            {
                hidden_state_counter += 1;
            }
        }

        buffer << "struct LSTMMemory" << endl;
        buffer << "{" << endl;
        buffer << "\t" << "int time_steps = 3;" << endl;
        buffer << "\t" << "int time_step_counter = 1;" << endl;

        for(int i = 0; i < hidden_state_counter; i++)
        {
            buffer << "\t" << "float hidden_state_" << to_string(i) << " = type(0);" << endl;
        }

        for(int i = 0; i < cell_states_counter; i++)
        {
            buffer << "\t" << "float cell_states_" << to_string(i) << " = type(0);" << endl;
        }

        buffer << "} lstm; \n\n" << endl;
        buffer << "vector<float> calculate_outputs(const vector<float>& inputs, LSTMMemory& lstm)" << endl;
    }
    else
    {
        buffer << "vector<float> calculate_outputs(const vector<float>& inputs)" << endl;
    }

    buffer << "{" << endl;

    for(int i = 0; i < inputs.dimension(0); i++)
    {
        if(inputs[i].empty())
        {
            buffer << "\t" << "const float " << "input_" << to_string(i) << " = " << "inputs[" << to_string(i) << "];" << endl;
        }
        else
        {
            buffer << "\t" << "const float " << inputs[i] << " = " << "inputs[" << to_string(i) << "];" << endl;
        }
    }

    if(LSTM_number>0)
    {
        buffer << "\n\tif(lstm.time_step_counter%lstm.time_steps == 0 ){" << endl;
        buffer << "\t\t" << "lstm.time_step_counter = 1;" << endl;

        for(int i = 0; i < hidden_state_counter; i++)
        {
            buffer << "\t\t" << "lstm.hidden_state_" << to_string(i) << " = type(0);" << endl;
        }

        for(int i = 0; i < cell_states_counter; i++)
        {
            buffer << "\t\t" << "lstm.cell_states_" << to_string(i) << " = type(0);" << endl;
        }

        buffer << "\t}" << endl;
    }

    buffer << "" << endl;

    for(int i = 0; i < tokens.dimension(0); i++)
    {
        string t = tokens(i);

        if(t.size()<=1)
        {
            outputs_buffer << "" << endl;
        }
        else
        {
            outputs_buffer << "\t" << t << endl;
        }
    }

    const string keyword = "double";
    string outputs_espresion = outputs_buffer.str();
    replace_substring_in_string (found_tokens, outputs_espresion, keyword);

    if(LSTM_number>0)
    {
        replace_all_appearances(outputs_espresion, "(t)", "");
        replace_all_appearances(outputs_espresion, "(t-1)", "");
        replace_all_appearances(outputs_espresion, "double cell_state", "cell_state");
        replace_all_appearances(outputs_espresion, "double hidden_state", "hidden_state");
        replace_all_appearances(outputs_espresion, "cell_state"  , "lstm.cell_state"  );
        replace_all_appearances(outputs_espresion, "hidden_state", "lstm.hidden_state");
    }

    buffer << outputs_espresion;

    const Tensor<string, 1> fixed_outputs = fix_write_expression_outputs(expression, outputs, "c");

    for(int i = 0; i < fixed_outputs.dimension(0); i++)
    {
        buffer << fixed_outputs(i) << endl;
    }

    buffer << "\t" << "vector<float> out(" << outputs.size() << ");" << endl;

    for(int i = 0; i < outputs.dimension(0); i++)
    {
        if(outputs[i].empty())
        {
            buffer << "\t" << "out[" << to_string(i) << "] = " << "output" << to_string(i) << ";"<< endl;
        }
        else
        {
            buffer << "\t" << "out[" << to_string(i) << "] = " << outputs[i] << ";" << endl;
        }
    }

    if(LSTM_number)
    {
        buffer << "\n\t" << "lstm.time_step_counter += 1;" << endl;
    }

    buffer << "\n\t" << "return out;" << endl;
    buffer << "}"  << endl;
    buffer << "\n" << endl;
    buffer << "int main(){ \n" << endl;
    buffer << "\t" << "vector<float> inputs(" << to_string(inputs.size()) << "); \n" << endl;

    for(int i = 0; i < inputs.dimension(0); i++)
    {
        if(inputs[i].empty())
        {
            buffer << "\t" << "const float " << "input_" << to_string(i) <<" =" << " /*enter your value here*/; " << endl;
            buffer << "\t" << "inputs[" << to_string(i) << "] = " << "input_" << to_string(i) << ";" << endl;
        }
        else
        {
            buffer << "\t" << "const float " << inputs[i] << " =" << " /*enter your value here*/; " << endl;
            buffer << "\t" << "inputs[" << to_string(i) << "] = " << inputs[i] << ";" << endl;
        }
    }

    buffer << "" << endl;

    if(LSTM_number > 0)
    {
        buffer << "\t"   << "LSTMMemory lstm;" << "\n" << endl;
        buffer << "\t"   << "vector<float> outputs(" << outputs.size() <<");" << endl;
        buffer << "\n\t" << "outputs = calculate_outputs(inputs, lstm);" << endl;
    }
    else
    {
        buffer << "\t"   << "vector<float> outputs(" << outputs.size() <<");" << endl;
        buffer << "\n\t" << "outputs = calculate_outputs(inputs);" << endl;
    }
    buffer << "" << endl;
    buffer << "\t" << "printf(\"These are your outputs:\\n\");" << endl;

    for(int i = 0; i < outputs.dimension(0); i++)
    {
        if(outputs[i].empty())
        {
            buffer << "\t" << "printf( \"output" << to_string(i) << ":" << " %f \\n\", "<< "outputs[" << to_string(i) << "]" << ");" << endl;
        }
        else
        {
            buffer << "\t" << "printf( \""<< outputs_names[i] << ":" << " %f \\n\", "<< "outputs[" << to_string(i) << "]" << ");" << endl;
        }
    }

    buffer << "\n\t" << "return 0;" << endl;
    buffer << "} \n" << endl;

    string out = buffer.str();
    //replace_all_appearances(out, "double double double", "double");
    //replace_all_appearances(out, "double double", "double");
    return out;
}


/// Returns a string that conatins an API composed by an html script (the index page), and a php scipt
/// that contains a function of the expression represented by the neural network.

string NeuralNetwork::write_expression_api() const
{
    ostringstream buffer;
    Tensor<string, 1> found_tokens;
    Tensor<string, 1> inputs =  get_inputs_names();
    Tensor<string, 1> outputs = get_outputs_names();

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

    buffer << "<!DOCTYPE html>" << endl;
    buffer << "<!--" << endl;
    buffer << "Artificial Intelligence Techniques SL\t" << endl;
    buffer << "artelnics@artelnics.com\t" << endl;
    buffer << "" << endl;
    buffer << "Your model has been exported to this php file." << endl;
    buffer << "You can manage it writting your parameters in the url of your browser.\t" << endl;
    buffer << "Example:" << endl;
    buffer << "" << endl;
    buffer << "\turl = http://localhost/API_example/\t" << endl;
    buffer << "\tparameters in the url = http://localhost/API_example/?num=5&num=2&...\t" << endl;
    buffer << "\tTo see the ouput refresh the page" << endl;
    buffer << "" << endl;
    buffer << "\tInputs Names: \t" << endl;

    Tensor<Tensor<string,1>, 1> inputs_outputs_buffer = fix_input_output_variables(inputs, outputs, buffer);

    for(Index i = 0; i < inputs_outputs_buffer(0).dimension(0);++i)
        inputs(i) = inputs_outputs_buffer(0)(i);

    for(Index i = 0; i < inputs_outputs_buffer(1).dimension(0);++i)
        outputs(i) = inputs_outputs_buffer(1)(i);

    buffer << inputs_outputs_buffer(2)[0];
    buffer << "" << endl;
    buffer << "-->\t" << endl;
    buffer << "" << endl;
    buffer << "<html lang = \"en\">\n" << endl;
    buffer << "<head>\n" << endl;
    buffer << "<title>Rest API Client Side Demo</title>\n " << endl;
    buffer << "<meta charset = \"utf-8\">" << endl;
    buffer << "<meta name = \"viewport\" content = \"width=device-width, initial-scale=1\">" << endl;
    buffer << "<link rel = \"stylesheet\" href = \"https://maxcdn.bootstrapcdn.com/bootstrap/3.3.7/css/bootstrap.min.css\">" << endl;
    buffer << "<script src = \"https://ajax.googleapis.com/ajax/libs/jquery/3.2.0/jquery.min.js\"></script>" << endl;
    buffer << "<script src = \"https://maxcdn.bootstrapcdn.com/bootstrap/3.3.7/js/bootstrap.min.js\"></script>" << endl;
    buffer << "</head>" << endl;

    buffer << "<style>" << endl;
    buffer << ".btn{" << endl;
    buffer << "background-color: #7393B3" << endl; // Gray
    buffer << "border: none;" << endl;
    buffer << "color: white;" << endl;
    buffer << "padding: 15px 32px;" << endl;
    buffer << "text-align: center;" << endl;
    buffer << "font-size: 16px;" << endl;
    buffer << "}" << endl;
    buffer << "</style>" << endl;

    buffer << "<body>" << endl;
    buffer << "<div class = \"container\">" << endl;
    buffer << "<br></br>" << endl;
    buffer << "<div class = \"form-group\">" << endl;
    buffer << "<p>" << endl;
    buffer << "follow the steps defined in the \"index.php\" file" << endl;
    buffer << "</p>" << endl;
    buffer << "<p>" << endl;
    buffer << "Refresh the page to see the prediction" << endl;
    buffer << "</p>" << endl;
    buffer << "</div>" << endl;
    buffer << "<h4>" << endl;
    buffer << "<?php" << "\n" << endl;

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
        {
            expression.erase(index, string::npos);
        }
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
        {
            push_back_string(found_tokens, word);
        }
    }

    if(LSTM_number > 0)
    {
        for(int i = 0; i < found_tokens.dimension(0); i++)
        {
            const string t = found_tokens(i);

            if(token.find("cell_state") == 0)
            {
                cell_states_counter += 1;
            }

            if(token.find("hidden_state") == 0)
            {
                hidden_state_counter += 1;
            }
        }

        buffer << "class NeuralNetwork{" << endl;
        buffer << "public $time_steps = 3;" << endl;
        buffer << "public $time_step_counter = 1;" << endl;

        for(int i = 0; i < hidden_state_counter; i++)
        {
            buffer << "public $" << "hidden_state_" << to_string(i) << " = type(0);" << endl;
        }

        for(int i = 0; i < cell_states_counter; i++)
        {
            buffer << "public $" << "cell_states_" << to_string(i) << " = type(0);" << endl;
        }

        buffer << "}" << endl;
        buffer << "$nn = new NeuralNetwork;" << endl;
    }

    buffer << "session_start();" << endl;
    buffer << "if(isset($_SESSION['lastpage']) && $_SESSION['lastpage'] == __FILE__) { " << endl;
    buffer << "if(isset($_SERVER['HTTPS']) && $_SERVER['HTTPS'] === 'on') " << endl;
    buffer << "\t$url = \"https://\"; " << endl;
    buffer << "else" << endl;
    buffer << "\t$url = \"http://\"; " << endl;
    buffer << "\n" << endl;
    buffer << "$url.= $_SERVER['HTTP_HOST'];" << endl;
    buffer << "$url.= $_SERVER['REQUEST_URI'];" << endl;
    buffer << "$url_components = parse_url($url);" << endl;
    buffer << "parse_str($url_components['query'], $params);" << endl;
    buffer << "\n" << endl;

    for(int i = 0; i < inputs.dimension(0); i++)
    {
        if(inputs[i].empty())
        {
            buffer << "$num"    + to_string(i) << " = " << "$params['num" + to_string(i) << "'];" << endl;
            buffer << "$input_" + to_string(i) << " = intval(" << "$num"  + to_string(i) << ");"  << endl;
        }
        else
        {
            buffer << "$num" + to_string(i) << " = " << "$params['num" + to_string(i) << "'];" << endl;
            buffer << "$" << inputs[i]      << " = intval(" << "$num"  + to_string(i) << ");"  << endl;
        }
    }

    buffer << "if(" << endl;

    for(int i = 0; i < inputs.dimension(0); i++)
    {
        if(i != inputs.dimension(0)-1)
        {
            buffer << "is_numeric(" << "$" << "num" + to_string(i) << ") &&" << endl;
        }
        else
        {
            buffer << "is_numeric(" << "$" << "num" + to_string(i) << ") )" << endl;
        }
    }

    buffer << "{" << endl;
    buffer << "$status=200;" << endl;
    buffer << "$status_msg = 'valid parameters';" << endl;
    buffer << "}" << endl;
    buffer << "else" << endl;
    buffer << "{" << endl;
    buffer << "$status =400;" << endl;
    buffer << "$status_msg = 'invalid parameters';" << endl;
    buffer << "}"   << endl;

    if(LSTM_number>0)
    {
        buffer << "if( $nn->time_step_counter % $nn->time_steps === 0 ){" << endl;
        buffer << "$nn->time_steps = 3;" << endl;
        buffer << "$nn->time_step_counter = 1;" << endl;

        for(int i = 0; i < hidden_state_counter; i++)
        {
            buffer << "$nn->" << "hidden_state_" << to_string(i) << " = type(0);" << endl;
        }

        for(int i = 0; i < cell_states_counter; i++)
        {
            buffer << "$nn->" << "cell_states_" << to_string(i) << " = type(0);" << endl;
        }
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

        if(LSTM_number>0)
        {
            replace_all_appearances(t, "(t)"     , "");
            replace_all_appearances(t, "(t-1)"   , "");
            replace_all_appearances(t, "hidden_" , "$hidden_");
            replace_all_appearances(t, "cell_"   , "$cell_"  );
            replace_all_appearances(t, "$hidden_", "$nn->hidden_");
            replace_all_appearances(t, "$cell_"  , "$nn->cell_"  );
        }

        buffer << t << endl;

        //side = 0;
    }

    const Tensor<string, 1> fixed_outputs = fix_write_expression_outputs(expression, outputs, "php");

    for(int i = 0; i < fixed_outputs.dimension(0); i++)
    {
        buffer << fixed_outputs(i) << endl;
    }

    buffer << "if($status === 200){" << endl;
    buffer << "$response = ['status' => $status,  'status_message' => $status_msg" << endl;

    for(int i = 0; i < outputs.dimension(0); i++)
    {
        buffer << ", '" << outputs(i) << "' => " << "$" << outputs[i] << endl;
    }

    buffer << "];" << endl;
    buffer << "}" << endl;
    buffer << "else" << endl;
    buffer << "{" << endl;
    buffer << "$response = ['status' => $status,  'status_message' => $status_msg" << "];" << endl;
    buffer << "}" << endl;

    if(LSTM_number>0)
    {
        buffer << "$nn->time_step_counter += 1;" << endl;
    }

    buffer << "\n" << endl;
    buffer << "$json_response_pretty = json_encode($response, JSON_PRETTY_PRINT);" << endl;
    buffer << "echo nl2br(\"\\n\" . $json_response_pretty . \"\\n\");" << endl;
    buffer << "}else{" << endl;
    buffer << "echo \"New page\";" << endl;
    buffer << "}" << endl;
    buffer << "$_SESSION['lastpage'] = __FILE__;" << endl;
    buffer << "?>" << endl;
    buffer << "\n" << endl;

    if(logistic)
    {
        buffer << "<?php" << endl;
        buffer << "function Logistic(int $x) {" << endl;
        buffer << "$z = 1/(1+exp(-$x));" << endl;
        buffer << "return $z;" << endl;
        buffer << "}" << endl;
        buffer << "?>" << endl;
        buffer << "\n" << endl;
    }

    if(ReLU)
    {
        buffer << "<?php" << endl;
        buffer << "function ReLU(int $x) {" << endl;
        buffer << "$z = max(0, $x);" << endl;
        buffer << "return $z;" << endl;
        buffer << "}" << endl;
        buffer << "?>" << endl;
        buffer << "\n" << endl;
    }

    if(ExpLinear)
    {
        buffer << "<?php" << endl;
        buffer << "function ExponentialLinear(int $x) {" << endl;
        buffer << "$alpha = 1.6732632423543772848170429916717;" << endl;
        buffer << "if($x>0){" << endl;
        buffer << "$z=$x;" << endl;
        buffer << "}else{" << endl;
        buffer << "$z=$alpha*(exp($x)-1);" << endl;
        buffer << "}" << endl;
        buffer << "return $z;" << endl;
        buffer << "}" << endl;
        buffer << "?>" << endl;
        buffer << "\n" << endl;
    }

    if(SExpLinear)
    {
        buffer << "<?php" << endl;
        buffer << "function SELU(int $x) {" << endl;
        buffer << "$alpha  = 1.67326;" << endl;
        buffer << "$lambda = 1.05070;" << endl;
        buffer << "if($x>0){" << endl;
        buffer << "$z=$lambda*$x;" << endl;
        buffer << "}else{" << endl;
        buffer << "$z=$lambda*$alpha*(exp($x)-1);" << endl;
        buffer << "}" << endl;
        buffer << "return $z;" << endl;
        buffer << "}" << endl;
        buffer << "?>" << endl;
        buffer << "\n" << endl;
    }

    if(HSigmoid)
    {
        buffer << "<?php" << endl;
        buffer << "function HardSigmoid(int $x) {" << endl;
        buffer << "$z=1/(1+exp(-$x));" << endl;
        buffer << "return $z;" << endl;
        buffer << "}" << endl;
        buffer << "?>" << endl;
        buffer << "\n" << endl;
    }

    if(SoftPlus)
    {
        buffer << "<?php" << endl;
        buffer << "function SoftPlus(int $x) {" << endl;
        buffer << "$z=log(1+exp($x));" << endl;
        buffer << "return $z;" << endl;
        buffer << "}" << endl;
        buffer << "?>" << endl;
        buffer << "\n" << endl;
    }

    if(SoftSign)
    {
        buffer << "<?php" << endl;
        buffer << "function SoftSign(int $x) {" << endl;
        buffer << "$z=$x/(1+abs($x));" << endl;
        buffer << "return $z;" << endl;
        buffer << "}" << endl;
        buffer << "?>" << endl;
        buffer << "\n" << endl;
    }

    buffer << "</h4>" << endl;
    buffer << "</div>" << endl;
    buffer << "</body>" << endl;
    buffer << "</html>" << endl;

    string out = buffer.str();

    replace_all_appearances(out, "$$", "$");
    replace_all_appearances(out, "_$", "_");

    return out;

}

/// Returns a string with the javaScript function of the expression represented by the neural network.

string NeuralNetwork::write_expression_javascript() const
{
    Tensor<string, 1> tokens;
    Tensor<string, 1> found_tokens;
    Tensor<string, 1> found_mathematical_expressions;
    Tensor<string, 1> inputs =  get_inputs_names();
    Tensor<string, 1> outputs = get_outputs_names();

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
        {
            expression.erase(index, string::npos);
        }
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

    buffer_to_fix << "<!--" << endl;
    buffer_to_fix << "Artificial Intelligence Techniques SL\t" << endl;
    buffer_to_fix << "artelnics@artelnics.com\t" << endl;
    buffer_to_fix << "" << endl;
    buffer_to_fix << "Your model has been exported to this JavaScript file." << endl;
    buffer_to_fix << "You can manage it with the main method, where you \t" << endl;
    buffer_to_fix << "can change the values of your inputs. For example:" << endl;
    buffer_to_fix << "" << endl;
    buffer_to_fix << "if we want to add these 3 values (0.3, 2.5 and 1.8)" << endl;
    buffer_to_fix << "to our 3 inputs (Input_1, Input_2 and Input_1), the" << endl;
    buffer_to_fix << "main program has to look like this:" << endl;
    buffer_to_fix << "\t" << endl;
    buffer_to_fix << "int neuralNetwork(){ " << endl;
    buffer_to_fix << "\t" << "vector<float> inputs(3);"<< endl;
    buffer_to_fix << "\t" << endl;
    buffer_to_fix << "\t" << "const float asdas  = 0.3;" << endl;
    buffer_to_fix << "\t" << "inputs[0] = asdas;"        << endl;
    buffer_to_fix << "\t" << "const float input2 = 2.5;" << endl;
    buffer_to_fix << "\t" << "inputs[1] = input2;"       << endl;
    buffer_to_fix << "\t" << "const float input3 = 1.8;" << endl;
    buffer_to_fix << "\t" << "inputs[2] = input3;"       << endl;
    buffer_to_fix << "\t" << ". . ." << endl;
    buffer_to_fix << "\n" << endl;
    buffer_to_fix << "Inputs Names:" <<endl;

    Tensor<Tensor<string,1>, 1> inputs_outputs_buffer = fix_input_output_variables(inputs, outputs, buffer_to_fix);

    for(Index i = 0; i < inputs_outputs_buffer(0).dimension(0);++i)
        inputs(i) = inputs_outputs_buffer(0)(i);

    for(Index i = 0; i < inputs_outputs_buffer(1).dimension(0);++i)
        outputs(i) = inputs_outputs_buffer(1)(i);

    ostringstream buffer;

    buffer << inputs_outputs_buffer(2)[0];
    buffer << "-->" << endl;
    buffer << "\n" << endl;
    buffer << "<!DOCTYPE HTML>" << endl;
    buffer << "<html lang=\"en\">" << endl;
    buffer << "\n" << endl;
    buffer << "<head>" << endl;
    buffer << "<link href=\"https://www.neuraldesigner.com/assets/css/neuraldesigner.css\" rel=\"stylesheet\" />" << endl;
    buffer << "<link href=\"https://www.neuraldesigner.com/images/fav.ico\" rel=\"shortcut icon\" type=\"image/x-icon\" />" << endl;
    buffer << "</head>" << endl;
    buffer << "\n" << endl;
    buffer << "<style>" << endl;
    buffer << "" << endl;
    buffer << "body {" << endl;
    buffer << "display: flex;" << endl;
    buffer << "justify-content: center;" << endl;
    buffer << "align-items: center;" << endl;
    buffer << "min-height: 100vh;" << endl;
    buffer << "margin: 0;" << endl;
    buffer << "background-color: #f0f0f0;" << endl;
    buffer << "font-family: Arial, sans-serif;" << endl;
    buffer << "}" << endl;
    buffer << "" << endl;
    buffer << ".form {" << endl;
    buffer << "border-collapse: collapse;" << endl;
    buffer << "width: 80%; " << endl;
    buffer << "max-width: 600px; " << endl;
    buffer << "margin: 0 auto; " << endl;
    buffer << "background-color: #fff; " << endl;
    buffer << "box-shadow: 0px 0px 10px rgba(0, 0, 0, 0.1); " << endl;
    buffer << "border: 1px solid #777; " << endl;
    buffer << "border-radius: 5px; " << endl;
    buffer << "}" << endl;
    buffer << "" << endl;
    buffer << "input[type=\"number\"] {" << endl;
    buffer << "width: 60px; " << endl;
    buffer << "text-align: center; " << endl;
    buffer << "}" << endl;
    buffer << "" << endl;
    buffer << ".form th," << endl;
    buffer << ".form td {" << endl;
    buffer << "padding: 10px;" << endl;
    buffer << "text-align: center;" << endl;
    buffer << "font-family: \"Helvetica Neue\", Helvetica, Arial, sans-serif; " << endl;
    buffer << "}" << endl;
    buffer << "" << endl;
    buffer << "" << endl;
    buffer << ".btn {" << endl;
    buffer << "background-color: #5da9e9;" << endl;
    buffer << "border: none;" << endl;
    buffer << "color: white;" << endl;
    buffer << "text-align: center;" << endl;
    buffer << "font-size: 16px;" << endl;
    buffer << "margin: 4px;" << endl;
    buffer << "cursor: pointer;" << endl;
    buffer << "padding: 10px 20px;" << endl;
    buffer << "border-radius: 5px;" << endl;
    buffer << "transition: background-color 0.3s ease;" << endl;
    buffer << "font-family: \"Helvetica Neue\", Helvetica, Arial, sans-serif;" << endl;
    buffer << "}" << endl;
    buffer << "" << endl;
    buffer << "" << endl;
    buffer << ".btn:hover {" << endl;
    buffer << "background-color: #4b92d3; " << endl;
    buffer << "}" << endl;
    buffer << "" << endl;
    buffer << "" << endl;
    buffer << "input[type=\"range\"]::-webkit-slider-runnable-track {" << endl;
    buffer << "background: #5da9e9;" << endl;
    buffer << "height: 0.5rem;" << endl;
    buffer << "}" << endl;
    buffer << "" << endl;
    buffer << "" << endl;
    buffer << "input[type=\"range\"]::-moz-range-track {" << endl;
    buffer << "background: #5da9e9;" << endl;
    buffer << "height: 0.5rem;" << endl;
    buffer << "}" << endl;
    buffer << "" << endl;
    buffer << "" << endl;
    buffer << ".tabla {" << endl;
    buffer << "width: 100%;" << endl;
    buffer << "padding: 5px;" << endl;
    buffer << "margin: 0; " << endl;
    buffer << "}" << endl;
    buffer << "" << endl;
    buffer << "" << endl;
    buffer << ".form th {" << endl;
    buffer << "background-color: #f2f2f2;" << endl;
    buffer << "font-family: \"Helvetica Neue\", Helvetica, Arial, sans-serif;" << endl;
    buffer << "}" << endl;

    buffer << "</style>" << endl;
    buffer << "\n" << endl;

    buffer << "<body>" << endl;
    buffer << "\n" << endl;
    buffer << "<section>" << endl;
    buffer << "<br/>" << endl;
    buffer << "\n" << endl;
    buffer << "<div align=\"center\" style=\"display:block;text-align: center;\">" << endl;
    buffer << "<!-- MENU OPTIONS HERE  -->" << endl;
    buffer << "<form style=\"display: inline-block;margin-left: auto; margin-right: auto;\">" << endl;
    buffer << "\n" << endl;
    buffer << "<table border=\"1px\" class=\"form\">" << endl;
    buffer << "\n" << endl;
    buffer << "INPUTS" << endl;

    if(has_scaling_layer())
    {
        const Tensor<Descriptives, 1> inputs_descriptives = get_scaling_layer_2d()->get_descriptives();

        for(int i = 0; i < inputs.dimension(0); i++)
        {
            buffer << "<!-- "<< to_string(i) <<"scaling layer -->" << endl;
            buffer << "<tr style=\"height:3.5em\">" << endl;
            buffer << "<td> " << inputs_names[i] << " </td>" << endl;
            buffer << "<td style=\"text-align:center\">" << endl;
            buffer << "<input type=\"range\" id=\"" << inputs[i] << "\" value=\"" << (inputs_descriptives(i).minimum + inputs_descriptives(i).maximum)/2 << "\" min=\"" << inputs_descriptives(i).minimum << "\" max=\"" << inputs_descriptives(i).maximum << "\" step=\"" << (inputs_descriptives(i).maximum - inputs_descriptives(i).minimum)/type(100) << "\" onchange=\"updateTextInput1(this.value, '" << inputs[i] << "_text')\" />" << endl;
            buffer << "<input class=\"tabla\" type=\"number\" id=\"" << inputs[i] << "_text\" value=\"" << (inputs_descriptives(i).minimum + inputs_descriptives(i).maximum)/2 << "\" min=\"" << inputs_descriptives(i).minimum << "\" max=\"" << inputs_descriptives(i).maximum << "\" step=\"" << (inputs_descriptives(i).maximum - inputs_descriptives(i).minimum)/type(100) << "\" onchange=\"updateTextInput1(this.value, '" << inputs[i] << "')\">" << endl;
            buffer << "</td>" << endl;
            buffer << "</tr>" << endl;
            buffer << "\n" << endl;
        }
    }
    else
    {
        for(int i = 0; i < inputs.dimension(0); i++)
        {
            buffer << "<!-- "<< to_string(i) <<"no scaling layer -->" << endl;
            buffer << "<tr style=\"height:3.5em\">" << endl;
            buffer << "<td> " << inputs_names[i] << " </td>" << endl;
            buffer << "<td style=\"text-align:center\">" << endl;
            buffer << "<input type=\"range\" id=\"" << inputs[i] << "\" value=\"0\" min=\"-1\" max=\"1\" step=\"0.01\" onchange=\"updateTextInput1(this.value, '" << inputs[i] << "_text')\" />" << endl;
            buffer << "<input class=\"tabla\" type=\"number\" id=\"" << inputs[i] << "_text\" value=\"0\" min=\"-1\" max=\"1\" step=\"0.01\" onchange=\"updateTextInput1(this.value, '" << inputs[i] << "')\">" << endl;
            buffer << "</td>" << endl;
            buffer << "</tr>" << endl;
            buffer << "\n" << endl;
        }
    }

    buffer << "</table>" << endl;
    buffer << "</form>" << endl;
    buffer << "\n" << endl;

    if(outputs.dimension(0) > maximum_output_variable_numbers)
    {
        buffer << "<!-- HIDDEN INPUTS -->" << endl;
        for(int i = 0; i < outputs.dimension(0); i++)
        {
            buffer << "<input type=\"hidden\" id=\"" << outputs[i] << "\" value=\"\">" << endl;
        }
        buffer << "\n" << endl;
    }

    buffer << "<div align=\"center\">" << endl;
    buffer << "<!-- BUTTON HERE -->" << endl;
    buffer << "<button class=\"btn\" onclick=\"neuralNetwork()\">calculate outputs</button>" << endl;
    buffer << "</div>" << endl;
    buffer << "\n" << endl;
    buffer << "<br/>" << endl;
    buffer << "\n" << endl;
    buffer << "<table border=\"1px\" class=\"form\">" << endl;
    buffer << "OUTPUTS" << endl;

    if(outputs.dimension(0) > maximum_output_variable_numbers)
    {
        buffer << "<tr style=\"height:3.5em\">" << endl;
        buffer << "<td> Target </td>" << endl;
        buffer << "<td>" << endl;
        buffer << "<select id=\"category_select\" onchange=\"updateSelectedCategory()\">" << endl;

        for(int i = 0; i < outputs.dimension(0); i++)
        {
            buffer << "<option value=\"" << outputs[i] << "\">" << outputs_names[i] << "</option>" << endl;
        }

        buffer << "</select>" << endl;
        buffer << "</td>" << endl;
        buffer << "</tr>" << endl;
        buffer << "\n" << endl;

        buffer << "<tr style=\"height:3.5em\">" << endl;
        buffer << "<td> Value </td>" << endl;
        buffer << "<td>" << endl;
        buffer << "<input style=\"text-align:right; padding-right:20px;\" id=\"selected_value\" value=\"\" type=\"text\"  disabled/>" << endl;
        buffer << "</td>" << endl;
        buffer << "</tr>" << endl;
        buffer << "\n" << endl;
    }
    else
    {
        for(int i = 0; i < outputs.dimension(0); i++)
        {
            buffer << "<tr style=\"height:3.5em\">" << endl;
            buffer << "<td> " << outputs_names[i] << " </td>" << endl;
            buffer << "<td>" << endl;
            buffer << "<input style=\"text-align:right; padding-right:20px;\" id=\"" << outputs[i] << "\" value=\"\" type=\"text\"  disabled/>" << endl;
            buffer << "</td>" << endl;
            buffer << "</tr>" << endl;
            buffer << "\n" << endl;
        }
    }

    buffer << "</table>" << endl;
    buffer << "\n" << endl;
    buffer << "</form>" << endl;
    buffer << "</div>" << endl;
    buffer << "\n" << endl;
    buffer << "</section>" << endl;
    buffer << "\n" << endl;

    buffer << "<script>" << endl;

    if(outputs.dimension(0) > maximum_output_variable_numbers)
    {
        buffer << "function updateSelectedCategory() {" << endl;
        buffer << "\tvar selectedCategory = document.getElementById(\"category_select\").value;" << endl;
        buffer << "\tvar selectedValueElement = document.getElementById(\"selected_value\");" << endl;

        for(int i = 0; i < outputs.dimension(0); i++) {
            buffer << "\tif(selectedCategory === \"" << outputs[i] << "\") {" << endl;
            buffer << "\t\tselectedValueElement.value = document.getElementById(\"" << outputs[i] << "\").value;" << endl;
            buffer << "\t}" << endl;
        }

        buffer << "}" << endl;
        buffer << "\n" << endl;
    }

    buffer << "function neuralNetwork()" << endl;
    buffer << "{" << endl;
    buffer << "\t" << "var inputs = [];" << endl;

    for(int i = 0; i < inputs.dimension(0); i++)
    {
        buffer << "\t" << "var " << inputs[i] << " =" << " document.getElementById(\"" << inputs[i] << "\").value; " << endl;
        buffer << "\t" << "inputs.push(" << inputs[i] << ");" << endl;
    }

    buffer << "\n" << "\t" << "var outputs = calculate_outputs(inputs); " << endl;

    for(int i = 0; i < outputs.dimension(0); i++)
    {
        buffer << "\t" << "var " << outputs[i] << " = document.getElementById(\"" << outputs[i] << "\");" << endl;
        buffer << "\t" << outputs[i] << ".value = outputs[" << to_string(i) << "].toFixed(4);" << endl;
    }

    if(outputs.dimension(0) > maximum_output_variable_numbers)
    {
        buffer << "\t" << "updateSelectedCategory();" << endl;
    }
    //else
    //{
    //    for(int i = 0; i < outputs.dimension(0); i++)
    //    {
    //        buffer << "\t" << "var " << outputs[i] << " = document.getElementById(\"" << outputs[i] << "\");" << endl;
    //        buffer << "\t" << outputs[i] << ".value = outputs[" << to_string(i) << "].toFixed(4);" << endl;
    //    }
    //}

    buffer << "\t" << "update_LSTM();" << endl;
    buffer << "}" << "\n" << endl;

    while(getline(ss, token, '\n'))
    {
        if(token.size() > 1 && token.back() == '{'){ break; }
        if(token.size() > 1 && token.back() != ';'){ token += ';'; }
        push_back_string(tokens, token);
    }

    buffer << "function calculate_outputs(inputs)" << endl;
    buffer << "{" << endl;

    for(int i = 0; i < inputs.dimension(0); i++)
    {
        buffer << "\t" << "var " << inputs[i] << " = " << "+inputs[" << to_string(i) << "];" << endl;
    }

    buffer << "" << endl;

    for(int i = 0; i < tokens.dimension(0); i++)
    {
        string word = get_word_from_token(tokens(i));

        if(word.size() > 1)
        {
            push_back_string(found_tokens, word);
        }
    }

    if(LSTM_number > 0)
    {
        for(int i = 0; i < found_tokens.dimension(0); i++)
        {
            token = found_tokens(i);

            if(token.find("cell_state") == 0)
            {
                cell_states_counter += 1;
            }

            if(token.find("hidden_state") == 0)
            {
                hidden_state_counter += 1;
            }
        }

        buffer << "\t" << "if( time_step_counter % time_steps == 0 ){" << endl;
        buffer << "\t\t" << "time_step_counter = 1" << endl;

        for(int i = 0; i < hidden_state_counter; i++)
        {
            buffer << "\t\t" << "hidden_state_" << to_string(i) << " = 0" << endl;
        }

        for(int i = 0; i < cell_states_counter; i++)
        {
            buffer << "\t\t" << "cell_states_" << to_string(i) << " = 0" << endl;
        }

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
            string new_word = "";

            new_word = sufix + key_word;
            replace_all_appearances(t, key_word, new_word);
        }

        if(t.size() <= 1)
        {
            buffer << "" << endl;
        }
        else
        {
            buffer << "\t" << "var " << t << endl;
        }
    }

    if(LSTM_number>0)
    {
        buffer << "\t" << "time_step_counter += 1" << "\n" << endl;
    }

    const Tensor<string, 1> fixed_outputs = fix_write_expression_outputs(expression, outputs, "javascript");

    for(int i = 0; i < fixed_outputs.dimension(0); i++)
    {
        buffer << fixed_outputs(i) << endl;
    }

    buffer << "\t" << "var out = [];" << endl;

    for(int i = 0; i < outputs.dimension(0); i++)
    {
        buffer << "\t" << "out.push(" << outputs[i] << ");" << endl;
    }

    buffer << "\n\t" << "return out;" << endl;
    buffer << "}" << "\n" << endl;

    if(LSTM_number>0)
    {
        buffer << "\t" << "var steps = 3;            " << endl;
        buffer << "\t" << "var time_steps = steps;   " << endl;
        buffer << "\t" << "var time_step_counter = 1;" << endl;

        for(int i = 0; i < hidden_state_counter; i++)
        {
            buffer << "\t" << "var " << "var hidden_state_" << to_string(i) << " = 0" << endl;
        }

        for(int i = 0; i < cell_states_counter; i++)
        {
            buffer << "\t" << "var " << "var cell_states_" << to_string(i) << " = 0" << endl;
        }

        buffer << "\n" << endl;
    }

    if(logistic)
    {
        buffer << "function Logistic(x) {" << endl;
        buffer << "\tvar z = 1/(1+Math.exp(x));" << endl;
        buffer << "\treturn z;" << endl;
        buffer << "}" << endl;
        buffer << "\n" << endl;
    }

    if(ReLU)
    {
        buffer << "function ReLU(x) {" << endl;
        buffer << "\tvar z = Math.max(0, x);" << endl;
        buffer << "\treturn z;" << endl;
        buffer << "}" << endl;
        buffer << "\n" << endl;
    }

    if(ExpLinear)
    {
        buffer << "function ExponentialLinear(x) {" << endl;
        buffer << "\tvar alpha = 1.67326;" << endl;
        buffer << "\tif(x>0){" << endl;
        buffer << "\t\tvar z = x;" << endl;
        buffer << "\t}else{" << endl;
        buffer << "\t\tvar z = alpha*(Math.exp(x)-1);" << endl;
        buffer << "\t}" << endl;
        buffer << "\treturn z;" << endl;
        buffer << "}" << endl;
        buffer << "\n" << endl;
    }

    if(SExpLinear)
    {
        buffer << "function SELU(x) {" << endl;
        buffer << "\tvar alpha  = 1.67326;" << endl;
        buffer << "\tvar lambda = 1.05070;" << endl;
        buffer << "\tif(x>0){" << endl;
        buffer << "\t\tvar z = lambda*x;" << endl;
        buffer << "\t}else{" << endl;
        buffer << "\t\tvar z = lambda*alpha*(Math.exp(x)-1);" << endl;
        buffer << "\t}" << endl;
        buffer << "return z;" << endl;
        buffer << "}" << endl;
        buffer << "\n" << endl;
    }

    if(HSigmoid)
    {
        buffer << "function HardSigmoid(x) {" << endl;
        buffer << "\tvar z=1/(1+Math.exp(-x));" << endl;
        buffer << "\treturn z;" << endl;
        buffer << "}" << endl;
        buffer << "\n" << endl;
    }

    if(SoftPlus)
    {
        buffer << "function SoftPlus(int x) {" << endl;
        buffer << "\tvar z=log(1+Math.exp(x));" << endl;
        buffer << "\treturn z;" << endl;
        buffer << "}" << endl;
        buffer << "\n" << endl;
    }

    if(SoftSign)
    {
        buffer << "function SoftSign(x) {" << endl;
        buffer << "\tvar z=x/(1+Math.abs(x));" << endl;
        buffer << "\treturn z;" << endl;
        buffer << "}" << endl;
        buffer << "\n" << endl;
    }

    buffer << "function updateTextInput1(val, id)" << endl;
    buffer << "{" << endl;
    buffer << "\t"<< "document.getElementById(id).value = val;" << endl;
    buffer << "}" << endl;
    buffer << "\n" << endl;
    buffer << "window.onresize = showDiv;" << endl;
    buffer << "\n" << endl;
    buffer << "</script>" << endl;
    buffer << "\n" << endl;
    buffer << "<!--script src=\"https://www.neuraldesigner.com/app/htmlparts/footer.js\"></script-->" << endl;
    buffer << "\n" << endl;
    buffer << "</body>" << endl;
    buffer << "\n" << endl;
    buffer << "</html>" << endl;

    string out = buffer.str();

    if(LSTM_number>0)
    {
        replace_all_appearances(out, "(t)", "");
        replace_all_appearances(out, "(t-1)", "");
        replace_all_appearances(out, "var cell_state"  , "cell_state"  );
        replace_all_appearances(out, "var hidden_state", "hidden_state");
    }

    return out;
}


/// Returns a string with the python function of the expression represented by the neural network.

string NeuralNetwork::write_expression_python() const
{
    ostringstream buffer;

    Tensor<string, 1> found_tokens;
    Tensor<string, 1> found_mathematical_expressions;

    Tensor<string, 1> inputs =  get_inputs_names();
    Tensor<string, 1> original_inputs =  get_inputs_names();
    Tensor<string, 1> outputs = get_outputs_names();

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

    buffer << "\'\'\' " << endl;
    buffer << "Artificial Intelligence Techniques SL\t" << endl;
    buffer << "artelnics@artelnics.com\t" << endl;
    buffer << "" << endl;
    buffer << "Your model has been exported to this python file."  << endl;
    buffer << "You can manage it with the 'NeuralNetwork' class.\t" << endl;
    buffer << "Example:" << endl;
    buffer << "" << endl;
    buffer << "\tmodel = NeuralNetwork()\t" << endl;
    buffer << "\tsample = [input_1, input_2, input_3, input_4, ...]\t" << endl;
    buffer << "\toutputs = model.calculate_outputs(sample)" << endl;
    buffer << "\n" << endl;
    buffer << "Inputs Names: \t" << endl;

    Tensor<Tensor<string,1>, 1> inputs_outputs_buffer = fix_input_output_variables(inputs, outputs, buffer);

    for(Index i = 0; i < inputs_outputs_buffer(0).dimension(0);++i)
    {
        inputs(i) = inputs_outputs_buffer(0)(i);
        buffer << "\t" << i << ") " << inputs(i) << endl;
    }

    for(Index i = 0; i < inputs_outputs_buffer(1).dimension(0);++i)
        outputs(i) = inputs_outputs_buffer(1)(i);

    buffer << "\n" << endl;

    buffer << "You can predict with a batch of samples using calculate_batch_output method\t" << endl;
    buffer << "IMPORTANT: input batch must be <class 'numpy.ndarray'> type\t" << endl;
    buffer << "Example_1:\t" << endl;
    buffer << "\tmodel = NeuralNetwork()\t" << endl;
    buffer << "\tinput_batch = np.array([[1, 2], [4, 5]])\t" << endl;
    buffer << "\toutputs = model.calculate_batch_output(input_batch)" << endl;
    buffer << "Example_2:\t" << endl;
    buffer << "\tinput_batch = pd.DataFrame( {'col1': [1, 2], 'col2': [3, 4]})\t" << endl;
    buffer << "\toutputs = model.calculate_batch_output(input_batch.values)" << endl;

    buffer << "\'\'\' " << endl;
    buffer << "\n" << endl;

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
        string word = "";
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
        {
            push_back_string(found_tokens, word);
        }
    }

    for(int i = 0; i< found_tokens.dimension(0); i++)
    {
        const string token = found_tokens(i);

        if(token.find("cell_state") == 0)
        {
            cell_states_counter += 1;
        }
        if(token.find("hidden_state") == 0)
        {
            hidden_state_counter += 1;
        }
    }

    buffer << "import numpy as np" << endl;
    buffer << "\n" << endl;
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
        buffer << "\t" << "def __init__(self, ts = 1):" << endl;
        buffer << "\t\t" << "self.inputs_number = " << to_string(inputs.size()) << endl;
        buffer << "\t\t" << "self.time_steps = ts" << endl;

        for(int i = 0; i < hidden_state_counter; i++)
        {
            buffer << "\t\t" << "self.hidden_state_" << to_string(i) << " = 0" << endl;
        }

        for(int i = 0; i < cell_states_counter; i++)
        {
            buffer << "\t\t" << "self.cell_states_" << to_string(i) << " = 0" << endl;
        }

        buffer << "\t\t" << "self.time_step_counter = 1" << endl;
    }
    else
    {
        string inputs_list;

        for(int i = 0; i < original_inputs.size();++i)
        {
            inputs_list += "'" + original_inputs(i) + "'";

            if(i < original_inputs.size() - 1)
            {
                inputs_list += ", ";
            }
        }

        buffer << "\t" << "def __init__(self):" << endl;
        buffer << "\t\t" << "self.inputs_number = " << to_string(inputs.size()) << endl;
        buffer << "\t\t" << "self.inputs_names = [" << inputs_list << "]" << endl;

    }

    buffer << "\n" << endl;

    if(logistic)
    {
        buffer << "\tdef Logistic (x):" << endl;
        buffer << "\t\t" << "z = 1/(1+np.exp(-x))" << endl;
        buffer << "\t\t" << "return z" << endl;
        buffer << "\n" << endl;
    }

    if(ReLU)
    {
        buffer << "\tdef ReLU (x):" << endl;
        buffer << "\t\t" << "z = max(0, x)" << endl;
        buffer << "\t\t" << "return z" << endl;
        buffer << "\n" << endl;
    }

    if(ExpLinear)
    {
        buffer << "\tdef ExponentialLinear (x):" << endl;
        buffer << "\t\t"   << "float alpha = 1.67326" << endl;
        buffer << "\t\t"   << "if(x>0):" << endl;
        buffer << "\t\t\t" << "z = x" << endl;
        buffer << "\t\t"   << "else:" << endl;
        buffer << "\t\t\t" << "z = alpha*(np.exp(x)-1)" << endl;
        buffer << "\t\t"   << "return z" << endl;
        buffer << "\n" << endl;
    }

    if(SExpLinear)
    {
        buffer << "\tdef SELU (x):" << endl;
        buffer << "\t\t"   << "float alpha = 1.67326" << endl;
        buffer << "\t\t"   << "float lambda = 1.05070" << endl;
        buffer << "\t\t"   << "if(x>0):" << endl;
        buffer << "\t\t\t" << "z = lambda*x" << endl;
        buffer << "\t\t"   << "else:" << endl;
        buffer << "\t\t\t" << "z = lambda*alpha*(np.exp(x)-1)" << endl;
        buffer << "\t\t"   << "return z" << endl;
        buffer << "\n" << endl;
    }

    if(HSigmoid)
    {
        buffer << "\tdef HardSigmoid (x):" << endl;
        buffer << "\t\t"   <<  "z = 1/(1+np.exp(-x))" << endl;
        buffer << "\t\t"   <<  "return z" << endl;
        buffer << "\n" << endl;
    }

    if(SoftPlus)
    {
        buffer << "\tdef SoftPlus (x):" << endl;
        buffer << "\t\t"   << "z = log(1+np.exp(x))" << endl;
        buffer << "\t\t"   << "return z" << endl;
        buffer << "\n" << endl;
    }

    if(SoftSign)
    {
        buffer << "\tdef SoftSign (x):" << endl;
        buffer << "\t\t"   << "z = x/(1+abs(x))" << endl;
        buffer << "\t\t"   << "return z" << endl;
        buffer << "\n" << endl;
    }

    buffer << "\t" << "def calculate_outputs(self, inputs):" << endl;

    for(int i = 0; i < inputs.dimension(0); i++)
    {
        buffer << "\t\t" << inputs[i] << " = " << "inputs[" << to_string(i) << "]" << endl;
    }

    if(LSTM_number>0)
    {
        buffer << "\n\t\t" << "if( self.time_step_counter % self.time_steps == 0 ):" << endl;
        buffer << "\t\t\t" << "self.t = 1" << endl;

        for(int i = 0; i < hidden_state_counter; i++)
        {
            buffer << "\t\t\t" << "self.hidden_state_" << to_string(i) << " = 0" << endl;
        }

        for(int i = 0; i < cell_states_counter; i++)
        {
            buffer << "\t\t\t" << "self.cell_states_" << to_string(i) << " = 0" << endl;
        }
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
        new_word = ""; key_word = "";

        for(int i = 0; i < found_tokens.dimension(0); i++)
        {
            key_word = found_tokens(i);
            new_word = sufix + key_word;
            replace_all_appearances(t, key_word, new_word);
        }

        sufix = "NeuralNetwork.";
        new_word = ""; key_word = "";

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
    {
        for(int i = 0; i < fixed_outputs.dimension(0); i++)
        {
            buffer << "\t\t" << fixed_outputs(i) << endl;
        }
    }

    buffer << "\t\t" << "out = " << "[None]*" << outputs.size() << "\n" << endl;

    for(int i = 0; i < outputs.dimension(0); i++)
    {
        buffer << "\t\t" << "out[" << to_string(i) << "] = " << outputs[i] << endl;
    }

    if(LSTM_number>0)
    {
        buffer << "\n\t\t" << "self.time_step_counter += 1" << endl;
    }

    if(model_type != ModelType::AutoAssociation)
    {
        buffer << "\n\t\t" << "return out;" << endl;
    }
    else
    {
        buffer << "\n\t\t" << "return out, sample_autoassociation_distance, sample_autoassociation_variables_distance;" << endl;
    }

    buffer << "\n" << endl;
    buffer << "\t" << "def calculate_batch_output(self, input_batch):" << endl;
    buffer << "\t\toutput_batch = [None]*input_batch.shape[0]\n" << endl;
    buffer << "\t\tfor i in range(input_batch.shape[0]):\n" << endl;

    if(has_recurrent_layer())
    {
        buffer << "\t\t\tif(i%self.time_steps == 0):\n" << endl;
        buffer << "\t\t\t\tself.hidden_states = "+to_string(get_recurrent_layer()->get_neurons_number())+"*[0]\n" << endl;
    }

    if(has_long_short_term_memory_layer())
    {
        buffer << "\t\t\tif(i%self.time_steps == 0):\n" << endl;
        buffer << "\t\t\t\tself.hidden_states = "+to_string(get_long_short_term_memory_layer()->get_neurons_number())+"*[0]\n" << endl;
        buffer << "\t\t\t\tself.cell_states = "+to_string(get_long_short_term_memory_layer()->get_neurons_number())+"*[0]\n" << endl;
    }

    buffer << "\t\t\tinputs = list(input_batch[i])\n" << endl;
    buffer << "\t\t\toutput = self.calculate_outputs(inputs)\n" << endl;
    buffer << "\t\t\toutput_batch[i] = output\n"<< endl;
    buffer << "\t\treturn output_batch\n"<<endl;

    buffer << "def main():" << endl;
    buffer << "\n\tinputs = []" << "\n" << endl;

    for(Index i = 0; i < inputs.size(); ++i)
    {
        buffer << "\t" << inputs(i) << " = " << "#- ENTER YOUR VALUE HERE -#" << endl;
        buffer << "\t" << "inputs.append(" << inputs(i) << ")" << "\n" << endl;
    }

    buffer << "\t" << "nn = NeuralNetwork()" << endl;
    buffer << "\t" << "outputs = nn.calculate_outputs(inputs)" << endl;
    buffer << "\t" << "print(outputs)" << endl;

    buffer << "\n" << "main()" << endl;

    string out = buffer.str();

    replace(out, ";", "");

    return out;
}


/// Saves the mathematical expression represented by the neural network to a text file.
/// @param file_name Name of the expression text file.

void NeuralNetwork::save_expression_c(const string& file_name) const
{
    std::ofstream file(file_name.c_str());

    if(!file.is_open())
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: NeuralNetwork class.\n"
               << "void  save_expression(const string&) method.\n"
               << "Cannot open expression text file.\n";

        throw runtime_error(buffer.str());
    }

    file << write_expression_c();

    file.close();
}


/// Saves the api function of the expression represented by the neural network to a text file.
/// @param file_name Name of the expression text file.

void NeuralNetwork::save_expression_api(const string& file_name) const
{
    std::ofstream file(file_name.c_str());

    if(!file.is_open())
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: NeuralNetwork class.\n"
               << "void  save_expression_api(const string&) method.\n"
               << "Cannot open expression text file.\n";

        throw runtime_error(buffer.str());
    }

    file << write_expression_api();

    file.close();
}


/// Saves the javascript function of the expression represented by the neural network to a text file.
/// @param file_name Name of the expression text file.

void NeuralNetwork::save_expression_javascript(const string& file_name) const
{
    std::ofstream file(file_name.c_str());

    if(!file.is_open())
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: NeuralNetwork class.\n"
               << "void  save_expression_javascript(const string&) method.\n"
               << "Cannot open expression text file.\n";

        throw runtime_error(buffer.str());
    }

    file << write_expression_javascript();

    file.close();
}


/// Saves the python function of the expression represented by the neural network to a text file.
/// @param file_name Name of the expression text file.

void NeuralNetwork::save_expression_python(const string& file_name) const
{
    std::ofstream file(file_name.c_str());

    if(!file.is_open())
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: NeuralNetwork class.\n"
               << "void  save_expression_python(const string&) method.\n"
               << "Cannot open expression text file.\n";

        throw runtime_error(buffer.str());
    }

    file << write_expression_python();

    file.close();
}


/// Saves a csv file containing the outputs for a set of given inputs.
/// @param inputs Inputs to calculate the outputs.
/// @param file_name Name of the data file

void NeuralNetwork::save_outputs(Tensor<type, 2>& inputs, const string & file_name)
{
    Tensor<type, 2> outputs = calculate_outputs(inputs);

    std::ofstream file(file_name.c_str());

    if(!file.is_open())
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: NeuralNetwork class.\n"
               << "void save_outputs(const string&) method.\n"
               << "Cannot open " << file_name << " file.\n";

        throw runtime_error(buffer.str());
    }

    const Tensor<string, 1> outputs_names = get_outputs_names();

    const Index outputs_number = get_outputs_number();
    const Index samples_number = inputs.dimension(0);

    for(Index i = 0; i < outputs_number; i++)
    {
        file << outputs_names[i];

        if(i != outputs_names.size()-1) file << ";";
    }

    file << "\n";

    for(Index i = 0; i < samples_number; i++)
    {
        for(Index j = 0; j < outputs_number; j++)
        {
            file << outputs(i,j);

            if(j != outputs_number-1) file << ";";
        }

        file << "\n";
    }

    file.close();
}


Tensor<string, 1> NeuralNetwork::get_layers_names() const
{
    const Index layers_number = get_layers_number();

    Tensor<string, 1> layers_names(layers_number);

    for(Index i = 0; i < layers_number; i++)
    {
        layers_names[i] = layers[i]->get_name();
    }

    return layers_names;
}


Layer* NeuralNetwork::get_last_trainable_layer() const
{
    if(layers.size() == 0) return nullptr;

    Tensor<Layer*, 1> trainable_layers = get_trainable_layers();

    const Index trainable_layers_number = get_trainable_layers_number();

    return trainable_layers(trainable_layers_number-1);
}


Layer* NeuralNetwork::get_last_layer() const
{
    if (layers.size() > 0)
    {
        Layer* last_layer = layers[layers.size() - 1];

        return last_layer;
    }

    return nullptr;
}


void NeuralNetworkBackPropagation::set(const Index& new_batch_samples_number, NeuralNetwork* new_neural_network)
{
    batch_samples_number = new_batch_samples_number;

    neural_network = new_neural_network;

    const Tensor<Layer*, 1> neural_network_layers = neural_network->get_layers();

    const Index layers_number = neural_network_layers.size();

    layers.resize(layers_number);
    layers.setConstant(nullptr);

    for (Index i = 0; i < layers_number; i++)
    {
        switch (neural_network_layers(i)->get_type())
        {
        case Layer::Type::Perceptron:
        {
            layers(i) = new PerceptronLayerBackPropagation(batch_samples_number, neural_network_layers(i));
        }
        break;

        case Layer::Type::Perceptron3D:
        {
            layers(i) = new PerceptronLayer3DBackPropagation(batch_samples_number, neural_network_layers(i));
        }
        break;

        case Layer::Type::Probabilistic:
        {
            layers(i) = new ProbabilisticLayerBackPropagation(batch_samples_number, neural_network_layers(i));
        }
        break;

        case Layer::Type::Probabilistic3D:
        {
            layers(i) = new ProbabilisticLayer3DBackPropagation(batch_samples_number, neural_network_layers(i));
        }
        break;

        case Layer::Type::Recurrent:
        {
            layers(i) = new RecurrentLayerBackPropagation(batch_samples_number, neural_network_layers(i));
        }
        break;

        case Layer::Type::LongShortTermMemory:
        {
            layers(i) = new LongShortTermMemoryLayerBackPropagation(batch_samples_number, neural_network_layers(i));
        }
        break;

        case Layer::Type::Convolutional:
        {
            layers(i) = new ConvolutionalLayerBackPropagation(batch_samples_number, neural_network_layers(i));
        }
        break;

        case Layer::Type::Pooling:
        {
            layers(i) = new PoolingLayerBackPropagation(batch_samples_number, neural_network_layers(i));
        }
        break;

        case Layer::Type::Flatten:
        {
            layers(i) = new FlattenLayerBackPropagation(batch_samples_number, neural_network_layers(i));
        }
        break;

        case Layer::Type::Embedding:
        {
            layers(i) = new EmbeddingLayerBackPropagation(batch_samples_number, neural_network_layers(i));
        }
        break;

        case Layer::Type::MultiheadAttention:
        {
            layers(i) = new MultiheadAttentionLayerBackPropagation(batch_samples_number, neural_network_layers(i));
        }
        break;

        case Layer::Type::Addition3D:
        {
            layers(i) = new AdditionLayer3DBackPropagation(batch_samples_number, neural_network_layers(i));
        }
        break;

        case Layer::Type::Normalization3D:
        {
            layers(i) = new NormalizationLayer3DBackPropagation(batch_samples_number, neural_network_layers(i));
        }
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

    for (Index i = 0; i < layers_number; i++)
    {
        switch (neural_network_layers(i)->get_type())
        {
        case Layer::Type::Perceptron:
        {
            layers(i) = new PerceptronLayerForwardPropagation(batch_samples_number, neural_network_layers(i));
        }
        break;

        case Layer::Type::Perceptron3D:
        {
            layers(i) = new PerceptronLayer3DForwardPropagation(batch_samples_number, neural_network_layers(i));
        }
        break;

        case Layer::Type::Probabilistic:
        {
            layers(i) = new ProbabilisticLayerForwardPropagation(batch_samples_number, neural_network_layers(i));
        }
        break;

        case Layer::Type::Probabilistic3D:
        {
            layers(i) = new ProbabilisticLayer3DForwardPropagation(batch_samples_number, neural_network_layers(i));
        }
        break;

        case Layer::Type::Recurrent:
        {
            layers(i) = new RecurrentLayerForwardPropagation(batch_samples_number, neural_network_layers(i));
        }
        break;

        case Layer::Type::LongShortTermMemory:
        {
            layers(i) = new LongShortTermMemoryLayerForwardPropagation(batch_samples_number, neural_network_layers(i));
        }
        break;

        case Layer::Type::Convolutional:
        {
            layers(i) = new ConvolutionalLayerForwardPropagation(batch_samples_number, neural_network_layers(i));
        }
        break;

        case Layer::Type::Pooling:
        {
            layers(i) = new PoolingLayerForwardPropagation(batch_samples_number, neural_network_layers(i));
        }
        break;

        case Layer::Type::Flatten:
        {
            layers(i) = new FlattenLayerForwardPropagation(batch_samples_number, neural_network_layers(i));
        }
        break;

        case Layer::Type::Scaling2D:
        {
            layers(i) = new ScalingLayer2DForwardPropagation(batch_samples_number, neural_network_layers(i));
        }
        break;

        case Layer::Type::Unscaling:
        {
            layers(i) = new UnscalingLayerForwardPropagation(batch_samples_number, neural_network_layers(i));
        }
        break;

        case Layer::Type::Bounding:
        {
            layers(i) = new BoundingLayerForwardPropagation(batch_samples_number, neural_network_layers(i));
        }
        break;

        case Layer::Type::RegionProposal:
        {
            //                layers(i) = new RegionProposalLayerForwardPropagation(batch_samples_number, neural_network_layers(i));
        }
        break;

        case Layer::Type::Embedding:
        {
            layers(i) = new EmbeddingLayerForwardPropagation(batch_samples_number, neural_network_layers(i));

        }
        break;

        case Layer::Type::MultiheadAttention:
        {
            layers(i) = new MultiheadAttentionLayerForwardPropagation(batch_samples_number, neural_network_layers(i));

        }
        break;

        case Layer::Type::Addition3D:
        {
            layers(i) = new AdditionLayer3DForwardPropagation(batch_samples_number, neural_network_layers(i));

        }
        break;

        case Layer::Type::Normalization3D:
        {
            layers(i) = new NormalizationLayer3DForwardPropagation(batch_samples_number, neural_network_layers(i));

        }
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

    for (Index i = 0; i < trainable_layers_number; i++)
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
        {
            ostringstream buffer;

            buffer << "OpenNN Exception: NeuralNetwork class.\n"
                << "Levenberg-Marquardt can only be used with Perceptron and Probabilistic layers.\n";

            throw invalid_argument(buffer.str());
        }
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
