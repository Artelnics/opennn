//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   N E U R A L   N E T W O R K   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "neural_network.h"

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

NeuralNetwork::NeuralNetwork(const NeuralNetwork::ProjectType& model_type, const Tensor<Index, 1>& architecture)
{
    set(model_type, architecture);
}


NeuralNetwork::NeuralNetwork(const NeuralNetwork::ProjectType& model_type, const initializer_list<Index>& architecture_list)
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

NeuralNetwork::NeuralNetwork(const Tensor<Layer*, 1>& new_layers_pointers)
{
    set();

    layers_pointers = new_layers_pointers;
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
        delete layers_pointers[i];

        layers_pointers[i] = nullptr;
    }

    layers_pointers.resize(0);
}


/// Add a new layer to the Neural Network model.
/// @param layer The layer that will be added.

void NeuralNetwork::add_layer(Layer* layer_pointer)
{
    if(has_bounding_layer())
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: NeuralNetwork class.\n"
               << "NeuralNetwork::add_layer() method.\n"
               << "No layers can be added after a bounding layer.\n";

        print();

        throw invalid_argument(buffer.str());
    }

    if(has_probabilistic_layer())
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: NeuralNetwork class.\n"
               << "NeuralNetwork::add_layer() method.\n"
               << "No layers can be added after a probabilistic layer.\n";

        throw invalid_argument(buffer.str());
    }

    //    if(layer_pointer->get_type_string() == "Pooling")
    //    {
    //        ostringstream buffer;

    //        buffer << "OpenNN Exception: NeuralNetwork class.\n"
    //               << "NeuralNetwork::add_layer() method.\n"
    //               << "Pooling Layer is not available yet. It will be included in future versions.\n";

    //        throw invalid_argument(buffer.str());
    //    }

    //    if(layer_pointer->get_type_string() == "Convolutional")
    //    {
    //        ostringstream buffer;

    //        buffer << "OpenNN Exception: NeuralNetwork class.\n"
    //               << "NeuralNetwork::add_layer() method.\n"
    //               << "Convolutional Layer is not available yet. It will be included in future versions.\n";

    //        throw invalid_argument(buffer.str());
    //    }

    const Layer::Type layer_type = layer_pointer->get_type();

    if(check_layer_type(layer_type))
    {
        const Index old_layers_number = get_layers_number();

        Tensor<Layer*, 1> old_layers_pointers = get_layers_pointers();

        layers_pointers.resize(old_layers_number+1);

        for(Index i = 0; i < old_layers_number; i++) layers_pointers(i) = old_layers_pointers(i);

        layers_pointers(old_layers_number) = layer_pointer;
    }
    else
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: NeuralNetwork class.\n"
               << "void add_layer(const Layer*) method.\n"
               << "Layer type " << layer_pointer->get_type_string() << " cannot be added in position " << layers_pointers.size()
               << " in the neural network architecture.\n";

        throw invalid_argument(buffer.str());
    }
}


/// Check if a given layer type can be added to the structure of the neural network.
/// LSTM and Recurrent layers can only be added at the beginning.
/// @param layer_type Type of new layer to be added.

bool NeuralNetwork::check_layer_type(const Layer::Type layer_type)
{
    const Index layers_number = layers_pointers.size();

    if(layers_number > 1 && (layer_type == Layer::Type::Recurrent || layer_type == Layer::Type::LongShortTermMemory))
    {
        return false;
    }
    else if(layers_number == 1 && (layer_type == Layer::Type::Recurrent || layer_type == Layer::Type::LongShortTermMemory))
    {
        const Layer::Type first_layer_type = layers_pointers[0]->get_type();

        if(first_layer_type != Layer::Type::Scaling) return false;
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
        if(layers_pointers[i]->get_type() == Layer::Type::Scaling) return true;
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
        if(layers_pointers[i]->get_type() == Layer::Type::LongShortTermMemory) return true;
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
        if(layers_pointers[i]->get_type() == Layer::Type::Convolutional) return true;
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
        if(layers_pointers[i]->get_type() == Layer::Type::Flatten) return true;
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
        if(layers_pointers[i]->get_type() == Layer::Type::Recurrent) return true;
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
        if(layers_pointers[i]->get_type() == Layer::Type::Unscaling) return true;
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
        if(layers_pointers[i]->get_type() == Layer::Type::Bounding) return true;
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
        if(layers_pointers[i]->get_type() == Layer::Type::Probabilistic) return true;
    }

    return false;
}


/// Returns true if the neural network object is empty,
/// and false otherwise.

bool NeuralNetwork::is_empty() const
{
    if(layers_pointers.dimension(0) == 0) return true;

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

NeuralNetwork::ProjectType NeuralNetwork::get_project_type() const
{
    return project_type;
}


string NeuralNetwork::get_project_type_string() const
{
    if(project_type == ProjectType::Approximation)
    {
        return "Approximation";
    }
    else if(project_type == ProjectType::Classification)
    {
        return "Classification";
    }
    else if(project_type == ProjectType::Forecasting)
    {
        return "Forecasting";
    }
    else if(project_type == ProjectType::ImageClassification)
    {
        return "ImageClassification";
    }
    else if(project_type == ProjectType::TextGeneration)
    {
        return "TextGeneration";
    }
    else if(project_type == ProjectType::AutoAssociation)
    {
        return "AutoAssociation";
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

Tensor<Layer*, 1> NeuralNetwork::get_layers_pointers() const
{
    return layers_pointers;
}


Layer* NeuralNetwork::get_layer_pointer(const Index& layer_index) const
{
    return layers_pointers(layer_index);
}


/// Returns a pointer to the trainable layers object composing this neural network object.

Tensor<Layer*, 1> NeuralNetwork::get_trainable_layers_pointers() const
{
    const Index layers_number = get_layers_number();

    const Index trainable_layers_number = get_trainable_layers_number();

    Tensor<Layer*, 1> trainable_layers_pointers(trainable_layers_number);

    Index index = 0;

    for(Index i = 0; i < layers_number; i++)
    {
        if(layers_pointers[i]->get_type() != Layer::Type::Scaling
                && layers_pointers[i]->get_type() != Layer::Type::Unscaling
                && layers_pointers[i]->get_type() != Layer::Type::Bounding)
        {
            trainable_layers_pointers[index] = layers_pointers[i];
            index++;
        }
    }

    return trainable_layers_pointers;
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
        if(layers_pointers[i]->get_type() != Layer::Type::Scaling
                && layers_pointers[i]->get_type() != Layer::Type::Unscaling
                && layers_pointers[i]->get_type() != Layer::Type::Bounding)
        {
            trainable_layers_indices[trainable_layer_index] = i;
            trainable_layer_index++;
        }
    }

    return trainable_layers_indices;
}


/// Returns a pointer to the scaling layer object composing this neural network object.

ScalingLayer* NeuralNetwork::get_scaling_layer_pointer() const
{
    const Index layers_number = get_layers_number();

    for(Index i = 0; i < layers_number; i++)
    {
        if(layers_pointers[i]->get_type() == Layer::Type::Scaling)
        {
            return dynamic_cast<ScalingLayer*>(layers_pointers[i]);
        }
    }

    ostringstream buffer;

    buffer << "OpenNN Exception: NeuralNetwork class.\n"
           << "ScalingLayer* get_scaling_layer_pointer() const method.\n"
           << "No scaling layer in neural network.\n";

    throw invalid_argument(buffer.str());
}


/// Returns a pointer to the unscaling layers object composing this neural network object.

UnscalingLayer* NeuralNetwork::get_unscaling_layer_pointer() const
{
    const Index layers_number = get_layers_number();

    for(Index i = 0; i < layers_number; i++)
    {
        if(layers_pointers[i]->get_type() == Layer::Type::Unscaling)
        {
            return dynamic_cast<UnscalingLayer*>(layers_pointers[i]);
        }
    }

    ostringstream buffer;

    buffer << "OpenNN Exception: NeuralNetwork class.\n"
           << "UnscalingLayer* get_unscaling_layer_pointer() const method.\n"
           << "No unscaling layer in neural network.\n";

    throw invalid_argument(buffer.str());
}


/// Returns a pointer to the bounding layer object composing this neural network object.

BoundingLayer* NeuralNetwork::get_bounding_layer_pointer() const
{
    const Index layers_number = get_layers_number();

    for(Index i = 0; i < layers_number; i++)
    {
        if(layers_pointers[i]->get_type() == Layer::Type::Bounding)
        {
            return dynamic_cast<BoundingLayer*>(layers_pointers[i]);
        }
    }

    ostringstream buffer;

    buffer << "OpenNN Exception: NeuralNetwork class.\n"
           << "BoundingLayer* get_bounding_layer_pointer() const method.\n"
           << "No bounding layer in neural network.\n";

    throw invalid_argument(buffer.str());
}

/// Returns a pointer to the flatten layer object composing this neural network object.

FlattenLayer* NeuralNetwork::get_flatten_layer_pointer() const
{
    const Index layers_number = get_layers_number();

    for(Index i = 0; i < layers_number; i++)
    {
        if(layers_pointers[i]->get_type() == Layer::Type::Flatten)
        {
            return dynamic_cast<FlattenLayer*>(layers_pointers[i]);
        }
    }

    ostringstream buffer;

    buffer << "OpenNN Exception: NeuralNetwork class.\n"
           << "BoundingLayer* get_flatten_layer_pointer() const method.\n"
           << "No flatten layer in neural network.\n";

    throw invalid_argument(buffer.str());
}


ConvolutionalLayer* NeuralNetwork::get_convolutional_layer_pointer() const
{
    const Index layers_number = get_layers_number();

    for(Index i = 0; i < layers_number; i++)
    {
        if(layers_pointers[i]->get_type() == Layer::Type::Convolutional)
        {
            return dynamic_cast<ConvolutionalLayer*>(layers_pointers[i]);
        }
    }

    ostringstream buffer;

    buffer << "OpenNN Exception: NeuralNetwork class.\n"
           << "ConvolutionalLayer* get_convolutional_layer_pointer() const method.\n"
           << "No convolutional layer in neural network.\n";

    throw invalid_argument(buffer.str());
}


PoolingLayer* NeuralNetwork::get_pooling_layer_pointer() const
{
    const Index layers_number = get_layers_number();

    for(Index i = 0; i < layers_number; i++)
    {
        if(layers_pointers[i]->get_type() == Layer::Type::Pooling)
        {
            return dynamic_cast<PoolingLayer*>(layers_pointers[i]);
        }
    }

    ostringstream buffer;

    buffer << "OpenNN Exception: NeuralNetwork class.\n"
           << "PoolingLayer* get_pooling_layer_pointer() const method.\n"
           << "No pooling layer in neural network.\n";

    throw invalid_argument(buffer.str());
}


/// Returns a pointer to the first probabilistic layer composing this neural network.

ProbabilisticLayer* NeuralNetwork::get_probabilistic_layer_pointer() const
{
    const Index layers_number = get_layers_number();

    for(Index i = 0; i < layers_number; i++)
    {
        if(layers_pointers[i]->get_type() == Layer::Type::Probabilistic)
        {
            return dynamic_cast<ProbabilisticLayer*>(layers_pointers[i]);
        }
    }

    ostringstream buffer;

    buffer << "OpenNN Exception: NeuralNetwork class.\n"
           << "ProbabilisticLayer* get_probabilistic_layer_pointer() const method.\n"
           << "No probabilistic layer in neural network.\n";

    throw invalid_argument(buffer.str());
}

/// Returns a pointer to the long short-term memory layer of this neural network, if it exits.

LongShortTermMemoryLayer* NeuralNetwork::get_long_short_term_memory_layer_pointer() const
{
    const Index layers_number = get_layers_number();

    for(Index i = 0; i < layers_number; i++)
    {
        if(layers_pointers[i]->get_type() == Layer::Type::LongShortTermMemory)
        {
            return dynamic_cast<LongShortTermMemoryLayer*>(layers_pointers[i]);
        }
    }

    ostringstream buffer;

    buffer << "OpenNN Exception: NeuralNetwork class.\n"
           << "LongShortTermMemoryLayer* get_long_short_term_memory_layer_pointer() const method.\n"
           << "No long-short-term memory layer in neural network.\n";

    throw invalid_argument(buffer.str());
}


/// Returns a pointer to the recurrent layer of this neural network, if it exits.

RecurrentLayer* NeuralNetwork::get_recurrent_layer_pointer() const
{
    const Index layers_number = get_layers_number();

    for(Index i = 0; i < layers_number; i++)
    {
        if(layers_pointers[i]->get_type() == Layer::Type::Recurrent)
        {
            return dynamic_cast<RecurrentLayer*>(layers_pointers[i]);
        }
    }

    ostringstream buffer;

    buffer << "OpenNN Exception: NeuralNetwork class.\n"
           << "RecurrentLayer* get_recurrent_layer_pointer() const method.\n"
           << "No recurrent layer in neural network.\n";

    throw invalid_argument(buffer.str());
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

void NeuralNetwork::set(const NeuralNetwork::ProjectType& model_type, const Tensor<Index, 1>& architecture)
{
    delete_layers();

    if(architecture.size() <= 1) return;

    const Index size = architecture.size();

    const Index inputs_number = architecture[0];
    const Index outputs_number = architecture[size-1];

    inputs_names.resize(inputs_number);

    ScalingLayer* scaling_layer_pointer = new ScalingLayer(inputs_number);

    this->add_layer(scaling_layer_pointer);

    if(model_type == ProjectType::Approximation)
    {
        for(Index i = 0; i < size-1; i++)
        {
            PerceptronLayer* perceptron_layer_pointer = new PerceptronLayer(architecture[i], architecture[i+1]);
            perceptron_layer_pointer->set_name("perceptron_layer_" + to_string(i+1));

            this->add_layer(perceptron_layer_pointer);

            if(i == size-2) perceptron_layer_pointer->set_activation_function(PerceptronLayer::ActivationFunction::Linear);
        }

        UnscalingLayer* unscaling_layer_pointer = new UnscalingLayer(outputs_number);

        this->add_layer(unscaling_layer_pointer);

        BoundingLayer* bounding_layer_pointer = new BoundingLayer(outputs_number);

        this->add_layer(bounding_layer_pointer);
    }
    else if(model_type == ProjectType::Classification || model_type == ProjectType::TextClassification)
    {
        for(Index i = 0; i < size-2; i++)
        {
            PerceptronLayer* perceptron_layer_pointer = new PerceptronLayer(architecture[i], architecture[i+1]);

            perceptron_layer_pointer->set_name("perceptron_layer_" + to_string(i+1));

            this->add_layer(perceptron_layer_pointer);
        }

        ProbabilisticLayer* probabilistic_layer_pointer = new ProbabilisticLayer(architecture[size-2], architecture[size-1]);

        this->add_layer(probabilistic_layer_pointer);
    }
    else if(model_type == ProjectType::Forecasting)
    {
        //                LongShortTermMemoryLayer* long_short_term_memory_layer_pointer = new LongShortTermMemoryLayer(architecture[0], architecture[1]);
        //                RecurrentLayer* long_short_term_memory_layer_pointer = new RecurrentLayer(architecture[0], architecture[1]);

        //                this->add_layer(long_short_term_memory_layer_pointer);

        for(Index i = 0 /* 1 when lstm layer*/; i < size-1 /*size-1 when lstm layer*/; i++)
        {
            PerceptronLayer* perceptron_layer_pointer = new PerceptronLayer(architecture[i], architecture[i+1]);

            perceptron_layer_pointer->set_name("perceptron_layer_" + to_string(i+1));

            this->add_layer(perceptron_layer_pointer);

            if(i == size-2) perceptron_layer_pointer->set_activation_function(PerceptronLayer::ActivationFunction::Linear);
        }

        UnscalingLayer* unscaling_layer_pointer = new UnscalingLayer(architecture[size-1]);

        this->add_layer(unscaling_layer_pointer);

        BoundingLayer* bounding_layer_pointer = new BoundingLayer(outputs_number);

        this->add_layer(bounding_layer_pointer);
    }
    else if(model_type == ProjectType::ImageClassification)
    {
        // Use the set mode build specifically for image classification
    }
    else if(model_type == ProjectType::TextGeneration)
    {
        LongShortTermMemoryLayer* long_short_term_memory_layer_pointer = new LongShortTermMemoryLayer(architecture[0], architecture[1]);

        ProbabilisticLayer* probabilistic_layer_pointer = new ProbabilisticLayer(architecture[1], architecture[2]);

        this->add_layer(long_short_term_memory_layer_pointer);
        this->add_layer(probabilistic_layer_pointer);
    }
    else if(model_type == ProjectType::AutoAssociation)
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

        this->add_layer(mapping_layer);
        this->add_layer(bottle_neck_layer);
        this->add_layer(demapping_layer);
        this->add_layer(output_layer);
        this->add_layer(unscaling_layer);
    }

    outputs_names.resize(outputs_number);

    set_default();
}


void NeuralNetwork::set(const NeuralNetwork::ProjectType& model_type, const initializer_list<Index>& architecture_list)
{
    Tensor<Index, 1> architecture(architecture_list.size());
    architecture.setValues(architecture_list);

    set_project_type(model_type);

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

    ScalingLayer* scaling_layer = new ScalingLayer(input_variables_dimensions);

    this->add_layer(scaling_layer);

    Tensor<Index, 1> outputs_dimensions = scaling_layer->get_outputs_dimensions();

    for(Index i = 0; i < blocks_number; i++)
    {
        ConvolutionalLayer* convolutional_layer = new ConvolutionalLayer(outputs_dimensions, filters_dimensions);
        this->add_layer(convolutional_layer);

        outputs_dimensions = convolutional_layer->get_outputs_dimensions();

        // Pooling layer 1

        PoolingLayer* pooling_layer_1 = new PoolingLayer(outputs_dimensions);
        this->add_layer(pooling_layer_1);

        outputs_dimensions = pooling_layer_1->get_outputs_dimensions();
    }

    FlattenLayer* flatten_layer = new FlattenLayer(outputs_dimensions);
    this->add_layer(flatten_layer);

    outputs_dimensions = flatten_layer->get_outputs_dimensions();

    const Tensor<Index, 0> outputs_dimensions_prod = outputs_dimensions.prod();

    PerceptronLayer* perceptron_layer = new PerceptronLayer(outputs_dimensions_prod(0), 3);
    perceptron_layer->set_name("perceptron_layer_1");
    this->add_layer(perceptron_layer);

    const Index perceptron_layer_outputs = perceptron_layer->get_neurons_number();

    ProbabilisticLayer* probabilistic_layer = new ProbabilisticLayer(perceptron_layer_outputs, outputs_number);
    this->add_layer(probabilistic_layer);
}


/// Sets the neural network members by loading them from an XML file.
/// @param file_name Neural network XML file_name.

void NeuralNetwork::set(const string& file_name)
{
    delete_layers();

    load(file_name);
}

void NeuralNetwork::set_project_type(const NeuralNetwork::ProjectType& new_project_type)
{
    project_type = new_project_type;
}

void NeuralNetwork::set_project_type_string(const string& new_project_type)
{
    if(new_project_type == "Approximation")
    {
        set_project_type(ProjectType::Approximation);
    }
    else if(new_project_type == "Classification")
    {
        set_project_type(ProjectType::Classification);
    }
    else if(new_project_type == "Forecasting")
    {
        set_project_type(ProjectType::Forecasting);
    }
    else if(new_project_type == "ImageClassification")
    {
        set_project_type(ProjectType::ImageClassification);
    }
    else if(new_project_type == "TextClassification")
    {
        set_project_type(ProjectType::TextClassification);
    }
    else if(new_project_type == "AutoAssociation")
    {
        set_project_type(ProjectType::AutoAssociation);
    }
    else
    {
        const string message =
                "Neural Network class exception:\n"
                "void set_project_type_string(const string&)\n"
                "Unknown project type: " + new_project_type + "\n";

        throw logic_error(message);
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

        throw invalid_argument(buffer.str());
    }

#endif

    inputs_names.resize(new_inputs_number);

    if(has_scaling_layer())
    {
        ScalingLayer* scaling_layer_pointer = get_scaling_layer_pointer();

        scaling_layer_pointer->set_inputs_number(new_inputs_number);
    }

    const Index trainable_layers_number = get_trainable_layers_number();
    Tensor<Layer*, 1> trainable_layers_pointers = get_trainable_layers_pointers();

    if(trainable_layers_number > 0)
    {
        trainable_layers_pointers[0]->set_inputs_number(new_inputs_number);
    }
}


/// Sets the new inputs number of this neural network object.
/// @param inputs Boolean vector containing the number of inputs.

void NeuralNetwork::set_inputs_number(const Tensor<bool, 1>& inputs)
{
    if(layers_pointers.dimension(0) == 0) return;

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
}


void NeuralNetwork::set_threads_number(const int& new_threads_number)
{
    const Index layers_number = get_layers_number();

    for(Index i = 0; i < layers_number; i++)
    {
        layers_pointers(i)->set_threads_number(new_threads_number);
    }
}


void NeuralNetwork::set_layers_pointers(Tensor<Layer*, 1>& new_layers_pointers)
{
    layers_pointers = new_layers_pointers;
}


PerceptronLayer* NeuralNetwork::get_first_perceptron_layer_pointer() const
{
    const Index layers_number = get_layers_number();

    for(Index i = 0; i < layers_number; i++)
    {
        if(layers_pointers(i)->get_type() == Layer::Type::Perceptron)
        {
            return static_cast<PerceptronLayer*>(layers_pointers[i]);
        }
    }

    return nullptr;
}


/// Returns the number of inputs to the neural network.

Index NeuralNetwork::get_inputs_number() const
{
    if(layers_pointers.dimension(0) != 0)
    {
        return layers_pointers(0)->get_inputs_number();
    }

    return 0;
}


Index NeuralNetwork::get_outputs_number() const
{
    if(layers_pointers.size() > 0)
    {
        const Layer* last_layer = layers_pointers[layers_pointers.size()-1];

        return last_layer->get_neurons_number();
    }

    return 0;
}


Tensor<Index, 1> NeuralNetwork::get_trainable_layers_neurons_numbers() const
{
    const Index trainable_layers_number = get_trainable_layers_number();

    Tensor<Index, 1> layers_neurons_number(trainable_layers_number);

    Index count = 0;

    for(Index i = 0; i < layers_pointers.size(); i++)
    {
        if(layers_pointers(i)->get_type() != Layer::Type::Scaling
                && layers_pointers(i)->get_type() != Layer::Type::Unscaling
                && layers_pointers(i)->get_type() != Layer::Type::Bounding)
        {
            layers_neurons_number(count) = layers_pointers[i]->get_neurons_number();

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

    for(Index i = 0; i < layers_pointers.size(); i++)
    {
        if(layers_pointers(i)->get_type() != Layer::Type::Scaling
                && layers_pointers(i)->get_type() != Layer::Type::Unscaling
                && layers_pointers(i)->get_type() != Layer::Type::Bounding)
        {
            layers_neurons_number(count) = layers_pointers[i]->get_inputs_number();

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
            architecture(i) = layers_pointers(i)->get_neurons_number();
        }
    }

    return architecture;
}


/// Returns the number of parameters in the neural network
/// The number of parameters is the sum of all the neural network parameters (biases and synaptic weights).

Index NeuralNetwork::get_parameters_number() const
{
    const Tensor<Layer*, 1> trainable_layers_pointers = get_trainable_layers_pointers();

    Index parameters_number = 0;

    for(Index i = 0; i < trainable_layers_pointers.size(); i++)
    {
        parameters_number += trainable_layers_pointers[i]->get_parameters_number();
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

    const Tensor<Layer*, 1> trainable_layers_pointers = get_trainable_layers_pointers();

    Index position = 0;

    for(Index i = 0; i < trainable_layers_number; i++)
    {
        const Tensor<type, 1> layer_parameters = trainable_layers_pointers(i)->get_parameters();

        for(Index j = 0; j < layer_parameters.size(); j++)
        {
            parameters(j + position) = layer_parameters(j);
        }

        position += layer_parameters.size();
    }

    return parameters;
}


Tensor< Tensor< TensorMap< Tensor<type, 1> >*, 1>, 1> NeuralNetwork::get_layers_parameters()
{
    const Index trainable_layers_number = get_trainable_layers_number();
    const Tensor<Layer*, 1> trainable_layers_pointers = get_trainable_layers_pointers();

    Tensor< Tensor< TensorMap< Tensor<type, 1> >*, 1>, 1> layers_parameters(trainable_layers_number);

    for(Index i = 0; i < trainable_layers_number; i++)
    {
        layers_parameters(i) = trainable_layers_pointers(i)->get_layer_parameters();

    }

    return layers_parameters;
}


Tensor<Index, 1> NeuralNetwork::get_trainable_layers_parameters_numbers() const
{
    const Index trainable_layers_number = get_trainable_layers_number();

    const Tensor<Layer*, 1> trainable_layers_pointers = get_trainable_layers_pointers();

    Tensor<Index, 1> trainable_layers_parameters_number(trainable_layers_number);

    for(Index i = 0; i < trainable_layers_number; i++)
    {
        trainable_layers_parameters_number[i] = trainable_layers_pointers[i]->get_parameters_number();
    }

    return trainable_layers_parameters_number;
}


Tensor<Tensor<type, 1>, 1> NeuralNetwork::get_trainable_layers_parameters(const Tensor<type, 1>& parameters) const
{
    const Index trainable_layers_number = get_trainable_layers_number();

    const Tensor<Index, 1> trainable_layers_parameters_number = get_trainable_layers_parameters_numbers();

    Tensor<Tensor<type, 1>, 1> trainable_layers_parameters(trainable_layers_number);

    Index index = 0;

    for(Index i = 0; i < trainable_layers_number; i++)
    {

        trainable_layers_parameters(i).resize(trainable_layers_parameters_number(i));

        trainable_layers_parameters(i) = parameters.slice(Eigen::array<Eigen::Index, 1>({index}), Eigen::array<Eigen::Index, 1>({trainable_layers_parameters_number(i)}));

        index += trainable_layers_parameters_number(i);

    }

    return trainable_layers_parameters;
}


/// Sets all the parameters(biases and synaptic weights) from a single vector.
/// @param new_parameters New set of parameter values.

void NeuralNetwork::set_parameters(Tensor<type, 1>& new_parameters) const
{
#ifdef OPENNN_DEBUG

    const Index size = new_parameters.size();

    const Index parameters_number = get_parameters_number();

    if(size < parameters_number)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: NeuralNetwork class.\n"
               << "void set_parameters(const Tensor<type, 1>&) method.\n"
               << "Size (" << size << ") must be greater or equal to number of parameters (" << parameters_number << ").\n";

        throw invalid_argument(buffer.str());
    }

#endif

    const Index trainable_layers_number = get_trainable_layers_number();

    const Tensor<Layer*, 1> trainable_layers_pointers = get_trainable_layers_pointers();

    const Tensor<Index, 1> trainable_layers_parameters_numbers = get_trainable_layers_parameters_numbers();

    Index index = 0;

    for(Index i = 0; i < trainable_layers_number; i++)
    {
        trainable_layers_pointers(i)->set_parameters(new_parameters, index);

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
    return layers_pointers.size();
}


Tensor<Index, 1> NeuralNetwork::get_layers_neurons_numbers() const
{
    Tensor<Index, 1> layers_neurons_number(layers_pointers.size());

    for(Index i = 0; i < layers_pointers.size(); i++)
    {
        layers_neurons_number(i) = layers_pointers[i]->get_neurons_number();
    }

    return layers_neurons_number;
}


Index NeuralNetwork::get_trainable_layers_number() const
{
    const Index layers_number = get_layers_number();

    Index count = 0;

    for(Index i = 0; i < layers_number; i++)
    {
        if(layers_pointers(i)->get_type() != Layer::Type::Scaling
                && layers_pointers(i)->get_type() != Layer::Type::Unscaling
                && layers_pointers(i)->get_type() != Layer::Type::Flatten
                && layers_pointers(i)->get_type() != Layer::Type::Bounding)
        {
            count++;
        }
    }

    return count;
}


Index NeuralNetwork::get_perceptron_layers_number() const
{
    const Index layers_number = get_layers_number();

    Index count = 0;

    for(Index i = 0; i < layers_number; i++)
    {
        if(layers_pointers(i)->get_type() == Layer::Type::Perceptron)
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
        if(layers_pointers(i)->get_type() == Layer::Type::Probabilistic)
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
        if(layers_pointers(i)->get_type() == Layer::Type::LongShortTermMemory)
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
        if(layers_pointers(i)->get_type() == Layer::Type::Flatten)
        {
            count++;
        }
    }

    return count;
}


Index NeuralNetwork::get_convolutional_layers_number() const
{
    const Index layers_number = get_layers_number();

    Index count = 0;

    for(Index i = 0; i < layers_number; i++)
    {
        if(layers_pointers(i)->get_type() == Layer::Type::Convolutional)
        {
            count++;
        }
    }

    return count;
}


Index NeuralNetwork::get_pooling_layers_number() const
{
    const Index layers_number = get_layers_number();

    Index count = 0;

    for(Index i = 0; i < layers_number; i++)
    {
        if(layers_pointers(i)->get_type() == Layer::Type::Pooling)
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
        if(layers_pointers(i)->get_type() == Layer::Type::Recurrent)
        {
            count++;
        }
    }

    return count;
}


/// Initializes all the biases and synaptic weights with a given value.

void NeuralNetwork::set_parameters_constant(const type& value) const
{
    const Index trainable_layers_number = get_trainable_layers_number();

    const Tensor<Layer*, 1> trainable_layers_pointers = get_trainable_layers_pointers();

    for(Index i = 0; i < trainable_layers_number; i++)
    {
        trainable_layers_pointers[i]->set_parameters_constant(value);
    }
}


/// Initializes all the parameters in the newtork(biases and synaptic weiths + independent
/// parameters) at random with values comprised between a given minimum and a given maximum values.
/// @param minimum Minimum initialization value.
/// @param maximum Maximum initialization value.

void NeuralNetwork::set_parameters_random() const
{
    const Index trainable_layers_number = get_trainable_layers_number();

    Tensor<Layer*, 1> trainable_layers_pointers = get_trainable_layers_pointers();

    for(Index i = 0; i < trainable_layers_number; i++)
    {
        trainable_layers_pointers[i]->set_parameters_random();
    }
}


/// Returns the norm of the vector of parameters.

type NeuralNetwork::calculate_parameters_norm() const
{
    const Tensor<type, 1> parameters = get_parameters();

    const Tensor<type, 0> parameters_norm = parameters.square().sum().sqrt();

    return parameters_norm(0);
}


/// Perturbate parameters of the neural network.
/// @param perturbation Maximum distance of perturbation.

void NeuralNetwork::perturbate_parameters(const type& perturbation)
{
#ifdef OPENNN_DEBUG

    if(perturbation < 0)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: NeuralNetwork class.\n"
               << "void perturbate_parameters(const type&) method.\n"
               << "Perturbation must be equal or greater than 0.\n";

        throw invalid_argument(buffer.str());
    }

#endif

    Tensor<type, 1> parameters = get_parameters();

    parameters = parameters+perturbation;

    set_parameters(parameters);
}


/// Calculates the forward propagation in the neural network.
/// @param batch DataSetBatch of data set that contains the inputs and targets to be trained.
/// @param foward_propagation Is a NeuralNetwork class structure where save the necessary parameters of forward propagation.

void NeuralNetwork::forward_propagate(DataSetBatch& batch,
                                      NeuralNetworkForwardPropagation& forward_propagation) const
{
    const Tensor<Layer*, 1> trainable_layers_pointers = get_trainable_layers_pointers();

    const Index trainable_layers_number = trainable_layers_pointers.size();

    trainable_layers_pointers(0)->forward_propagate(batch.inputs_data, batch.inputs_dimensions, forward_propagation.layers(0));

    for(Index i = 1; i < trainable_layers_number; i++)
    {
        trainable_layers_pointers(i)->forward_propagate(forward_propagation.layers(i-1)->outputs_data,
                                                        forward_propagation.layers(i-1)->outputs_dimensions,
                                                        forward_propagation.layers(i));
    }
}


/// Calculates the forward propagation in the neural network.
/// @param batch DataSetBatch of data set that contains the inputs and targets to be trained.
/// @param paramters Parameters of neural network.
/// @param foward_propagation Is a NeuralNetwork class structure where save the necessary parameters of forward propagation.

void NeuralNetwork::forward_propagate(const DataSetBatch& batch,
                                      Tensor<type, 1>& parameters,
                                      NeuralNetworkForwardPropagation& forward_propagation) const
{
    const Tensor<Layer*, 1> trainable_layers_pointers = get_trainable_layers_pointers();

    const Index trainable_layers_number = trainable_layers_pointers.size();

    const Index parameters_number = trainable_layers_pointers(0)->get_parameters_number();

    Tensor<type, 1> potential_parameters = TensorMap<Tensor<type, 1>>(parameters.data(), parameters_number);

    trainable_layers_pointers(0)->forward_propagate(batch.inputs_data, batch.inputs_dimensions,  potential_parameters, forward_propagation.layers(0));

    Index index = parameters_number;

    for(Index i = 1; i < trainable_layers_number; i++)
    {
        const Index parameters_number = trainable_layers_pointers(i)->get_parameters_number();

        Tensor<type, 1> potential_parameters = TensorMap<Tensor<type, 1>>(parameters.data() + index, parameters_number);

        trainable_layers_pointers(i)->forward_propagate(forward_propagation.layers(i-1)->outputs_data,
                                                        forward_propagation.layers(i-1)->outputs_dimensions,
                                                        potential_parameters,
                                                        forward_propagation.layers(i));

        index += parameters_number;
    }
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

Tensor<type, 2> NeuralNetwork::calculate_outputs(type* inputs_data, const Tensor<Index, 1>& inputs_dimensions)
{
#ifdef OPENNN_DEBUG
    cout << "inputs dimensions: " << inputs_dimensions << endl;

    if(inputs_dimensions(1) != get_inputs_number())
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: NeuralNetwork class.\n"
               << "void calculate_outputs(type* inputs_data, Tensor<Index, 1>& inputs_dimensions, type* outputs_data, Tensor<Index, 1>& outputs_dimensions) method.\n"
               << "Inputs columns number must be equal to " << get_inputs_number() << ", (" << inputs_dimensions(1) << ").\n";

        throw invalid_argument(buffer.str());
    }
#endif

    const Index inputs_dimensions_number = inputs_dimensions.size();

    if(inputs_dimensions_number == 2)
    {
        Tensor<type, 2> outputs;
        Tensor<type, 2> last_layer_outputs;

        Tensor<Index, 1> outputs_dimensions;
        Tensor<Index, 1> last_layer_outputs_dimensions;

        const Index layers_number = get_layers_number();

        if(layers_number == 0)
        {
            const Tensor<Index, 0> inputs_size = inputs_dimensions.prod();
            outputs = TensorMap<Tensor<type,2>>(inputs_data, inputs_dimensions(0), inputs_dimensions(1));
            return outputs;
        }

        outputs.resize(inputs_dimensions(0),layers_pointers(0)->get_neurons_number());
        outputs_dimensions = get_dimensions(outputs);

        layers_pointers(0)->calculate_outputs(inputs_data, inputs_dimensions, outputs.data(), outputs_dimensions);

        last_layer_outputs = outputs;
        last_layer_outputs_dimensions = get_dimensions(last_layer_outputs);

        for(Index i = 1; i < layers_number; i++)
        {
            outputs.resize(inputs_dimensions(0),layers_pointers(i)->get_neurons_number());
            outputs_dimensions = get_dimensions(outputs);

            layers_pointers(i)->calculate_outputs(last_layer_outputs.data(), last_layer_outputs_dimensions, outputs.data(), outputs_dimensions);

            last_layer_outputs = outputs;
            last_layer_outputs_dimensions = get_dimensions(last_layer_outputs);
        }

        return outputs;
    }
    else if(inputs_dimensions_number == 4)
    {
        /// @todo
    }
    else
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: NeuralNetwork class.\n"
               << "void calculate_outputs(type* inputs_data, Tensor<Index, 1>& inputs_dimensions, type* outputs_data, Tensor<Index, 1>& outputs_dimensions) method.\n"
               << "Inputs dimensions number (" << inputs_dimensions_number << ") must be 2 or 4.\n";

        throw invalid_argument(buffer.str());
    }
}


Tensor<type, 2> NeuralNetwork::calculate_scaled_outputs(type* scaled_inputs_data, Tensor<Index, 1>& inputs_dimensions)
{
#ifdef OPENNN_DEBUG
    if(inputs_dimensions(1) != get_inputs_number())
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: NeuralNetwork class.\n"
               << "void calculate_outputs(type* inputs_data, Tensor<Index, 1>& inputs_dimensions, type* outputs_data, Tensor<Index, 1>& outputs_dimensions) method.\n"
               << "Inputs columns number must be equal to " << get_inputs_number() << ", (inputs number).\n";

        throw invalid_argument(buffer.str());
    }
#endif

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
            const Tensor<Index, 0> inputs_size = inputs_dimensions.prod();
            scaled_outputs = TensorMap<Tensor<type,2>>(scaled_inputs_data, inputs_dimensions(0), inputs_dimensions(1));
            return scaled_outputs;
        }

        scaled_outputs.resize(inputs_dimensions(0),layers_pointers(0)->get_neurons_number());

        outputs_dimensions = get_dimensions(scaled_outputs);

        if(layers_pointers(0)->get_type_string() != "Scaling")
        {
            layers_pointers(0)->calculate_outputs(scaled_inputs_data, inputs_dimensions, scaled_outputs.data(), outputs_dimensions);
        }
        else
        {
            scaled_outputs = TensorMap<Tensor<type,2>>(scaled_inputs_data, inputs_dimensions(0), inputs_dimensions(1));
        }

        last_layer_outputs = scaled_outputs;

        last_layer_outputs_dimensions = get_dimensions(last_layer_outputs);

        for(Index i = 1; i < layers_number; i++)
        {
            if(layers_pointers(i)->get_type_string() != "Unscaling" || layers_pointers(i)->get_type_string() != "Scaling")
            {
                scaled_outputs.resize(inputs_dimensions(0),layers_pointers(i)->get_neurons_number());
                outputs_dimensions = get_dimensions(scaled_outputs);

                layers_pointers(i)->calculate_outputs(last_layer_outputs.data(), last_layer_outputs_dimensions, scaled_outputs.data(), outputs_dimensions);

                last_layer_outputs = scaled_outputs;
                last_layer_outputs_dimensions = get_dimensions(last_layer_outputs);
            }
        }

        return scaled_outputs;
    }
    else if(inputs_dimensions_number == 4)
    {
        /// @todo
    }
    else
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: NeuralNetwork class.\n"
               << "void calculate_outputs(type* inputs_data, Tensor<Index, 1>& inputs_dimensions, type* outputs_data, Tensor<Index, 1>& outputs_dimensions) method.\n"
               << "Inputs dimensions number (" << inputs_dimensions_number << ") must be 2 or 4.\n";

        throw invalid_argument(buffer.str());
    }

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
        inputs(direction) = minimum + (maximum - minimum)*static_cast<type>(i)/static_cast<type>(points_number-1);

        for(Index j = 0; j < inputs_number; j++)
        {
            directional_inputs(i,j) = inputs(j);
        }
    }

    return directional_inputs;
}


/// Generates a text output based on the neural network and some input letters given by the user.
/// @param text_generation_alphabet TextGenerationAlphabet object used for the text generation model
/// @param input_string Input string given by the user
/// @param max_length Maximum length of the returned string
/// @param one_word Boolean, if true returns just one word, if false returns a phrase

string NeuralNetwork::calculate_text_outputs(TextGenerationAlphabet& text_generation_alphabet, const string& input_string, const Index& max_length, const bool& one_word)
{
    string result = one_word ? generate_word(text_generation_alphabet, input_string, max_length) : generate_phrase(text_generation_alphabet, input_string, max_length);

    return result;
}


/// @todo

string NeuralNetwork::generate_word(TextGenerationAlphabet& text_generation_alphabet, const string& first_letters, const Index& length)
{
    ostringstream buffer;

    buffer << "OpenNN Exception: NeuralNetwork class.\n"
           << "string generate_word(TextGenerationAlphabet&, const string&, const Index&) method.\n"
           << "This method is not implemented yet.\n";

    throw invalid_argument(buffer.str());

    return string();

    // Under development

    //    const Index alphabet_length = text_generation_alphabet.get_alphabet_length();

    //    if(first_letters.length()*alphabet_length != get_inputs_number())
    //    {
    //        ostringstream buffer;

    //        buffer << "OpenNN Exception: NeuralNetwork class.\n"
    //               << "string generate_word(TextGenerationAlphabet&, const string&, const Index&) method.\n"
    //               << "Input string length must be equal to " << int(get_inputs_number()/alphabet_length) << "\n";

    //        throw invalid_argument(buffer.str());
    //    }


    //    string result = first_letters;

    //    // 1. Input letters to one hot encode

    //    Tensor<type, 2> input_data = text_generation_alphabet.multiple_one_hot_encode(first_letters);

    //    Tensor<Index, 1> input_dimensions = get_dimensions(input_data);

    //    Tensor<string, 1> punctuation_signs(6); // @todo change for multiple letters predicted

    //    punctuation_signs.setValues({" ",",",".","\n",":",";"});

    //    // 2. Loop for forecasting the following letter in function of the last letters

    //    do{
    //        Tensor<type, 2> output = calculate_outputs(input_data.data(), input_dimensions);

    //        string letter = text_generation_alphabet.multiple_one_hot_decode(output);

    //        if(!contains(punctuation_signs, letter))
    //        {
    //            result += letter;

    //            input_data = text_generation_alphabet.multiple_one_hot_encode(result.substr(result.length() - first_letters.length()));
    //        }

    //    }while(result.length() < length);

    //    return result;
}


/// @todo

string NeuralNetwork::generate_phrase(TextGenerationAlphabet& text_generation_alphabet, const string& first_letters, const Index& length)
{
    const Index alphabet_length = text_generation_alphabet.get_alphabet_length();

    if(first_letters.length()*alphabet_length != get_inputs_number())
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: NeuralNetwork class.\n"
               << "string generate_word(TextGenerationAlphabet&, const string&, const Index&) method.\n"
               << "Input string length must be equal to " << int(get_inputs_number()/alphabet_length) << "\n";

        throw invalid_argument(buffer.str());
    }

    string result = first_letters;

    Tensor<type, 2> input_data = text_generation_alphabet.multiple_one_hot_encode(first_letters);

    Tensor<Index, 1> input_dimensions = get_dimensions(input_data);

    do{
        Tensor<type, 2> input_data(get_inputs_number(), 1);
        input_data.setZero();
        Tensor<Index, 1> input_dimensions = get_dimensions(input_data);

        Tensor<type, 2> output = calculate_outputs(input_data.data(), input_dimensions);

        string letter = text_generation_alphabet.multiple_one_hot_decode(output);

        result += letter;

        input_data = text_generation_alphabet.multiple_one_hot_encode(result.substr(result.length() - first_letters.length()));

    }while(result.length() < length);

    return result;
}


/// For each layer: inputs, neurons, activation function.
/// @todo Complete for the rest of the layers.

Tensor<string, 2> NeuralNetwork::get_information() const
{
    const Index trainable_layers_number = get_trainable_layers_number();

    Tensor<string, 2> information(trainable_layers_number, 3);

    Tensor<Layer*, 1> trainable_layers_pointers = get_trainable_layers_pointers();

    for(Index i = 0; i < trainable_layers_number; i++)
    {
        information(i,0) = to_string(trainable_layers_pointers(i)->get_inputs_number());
        information(i,1) = to_string(trainable_layers_pointers(i)->get_neurons_number());

        const string layer_type = trainable_layers_pointers(i)->get_type_string();

        if(layer_type == "Perceptron")
        {
            const PerceptronLayer* perceptron_layer = static_cast<PerceptronLayer*>(trainable_layers_pointers(i));

            information(i,2) = perceptron_layer->write_activation_function();
        }
        else if(layer_type == "Probabilistic")
        {
            const ProbabilisticLayer* probabilistic_layer = static_cast<ProbabilisticLayer*>(trainable_layers_pointers(i));

            information(i,2) = probabilistic_layer->write_activation_function();
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

    Tensor<Layer*, 1> trainable_layers_pointers = get_trainable_layers_pointers();

    Index perceptron_layer_index = 0;

    for(Index i = 0; i < trainable_layers_number; i++)
    {
        const string layer_type = trainable_layers_pointers(i)->get_type_string();

        if(layer_type == "Perceptron")
        {
            information(perceptron_layer_index,0) = to_string(trainable_layers_pointers(i)->get_inputs_number());
            information(perceptron_layer_index,1) = to_string(trainable_layers_pointers(i)->get_neurons_number());

            const PerceptronLayer* perceptron_layer = static_cast<PerceptronLayer*>(trainable_layers_pointers(i));

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

    Tensor<Layer*, 1> trainable_layers_pointers = get_trainable_layers_pointers();

    Index probabilistic_layer_index = 0;

    for(Index i = 0; i < trainable_layers_number; i++)
    {
        const string layer_type = trainable_layers_pointers(i)->get_type_string();

        if(layer_type == "Probabilistic")
        {
            information(probabilistic_layer_index,0) = to_string(trainable_layers_pointers(i)->get_inputs_number());
            information(probabilistic_layer_index,1) = to_string(trainable_layers_pointers(i)->get_neurons_number());

            const ProbabilisticLayer* probabilistic_layer = static_cast<ProbabilisticLayer*>(trainable_layers_pointers(i));

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

    file_stream.OpenElement("LayersTypes");

    buffer.str("");

    for(Index i = 0; i < layers_pointers.size(); i++)
    {
        buffer << layers_pointers[i]->get_type_string();
        if(i != (layers_pointers.size()-1)) buffer << " ";
    }

    file_stream.PushText(buffer.str().c_str());

    file_stream.CloseElement();

    // Layers information

    for(Index i = 0; i < layers_pointers.size(); i++)
    {
        layers_pointers[i]->write_XML(file_stream);
    }

    // Layers (end tag)

    file_stream.CloseElement();

    // Ouputs

    file_stream.OpenElement("Outputs");

    // Outputs number

    const Index outputs_number = outputs_names.size();

    file_stream.OpenElement("OutputsNumber");

    buffer.str("");
    buffer << outputs_number;

    file_stream.PushText(buffer.str().c_str());

    file_stream.CloseElement();

    // Outputs names

    for(Index i = 0; i < outputs_number; i++)
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

        throw invalid_argument(buffer.str());
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
            catch(const invalid_argument& e)
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

        throw invalid_argument(buffer.str());
    }

    // Inputs number

    const tinyxml2::XMLElement* inputs_number_element = root_element->FirstChildElement("InputsNumber");

    if(!inputs_number_element)
    {
        buffer << "OpenNN Exception: NeuralNetwork class.\n"
               << "void inputs_from_XML(const tinyxml2::XMLDocument&) method.\n"
               << "Inputs number element is nullptr.\n";

        throw invalid_argument(buffer.str());
    }

    Index new_inputs_number = 0;

    if(inputs_number_element->GetText())
    {
        new_inputs_number = static_cast<Index>(atoi(inputs_number_element->GetText()));

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

                throw invalid_argument(buffer.str());
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

        throw invalid_argument(buffer.str());
    }

    // Layers types

    const tinyxml2::XMLElement* layers_types_element = root_element->FirstChildElement("LayersTypes");

    if(!layers_types_element)
    {
        buffer << "OpenNN Exception: NeuralNetwork class.\n"
               << "void layers_from_XML(const tinyxml2::XMLDocument&) method.\n"
               << "Layers types element is nullptr.\n";

        throw invalid_argument(buffer.str());
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
        if(layers_types(i) == "Scaling")
        {
            ScalingLayer* scaling_layer = new ScalingLayer();

            const tinyxml2::XMLElement* scaling_element = start_element->NextSiblingElement("ScalingLayer");
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
        else if(layers_types(i) == "Convolutional")
        {
            ConvolutionalLayer* convolutional_layer = new ConvolutionalLayer();

            const tinyxml2::XMLElement* convolutional_element = start_element->NextSiblingElement("ConvolutionalLayer");
            start_element = convolutional_element;

            if(convolutional_element)
            {
                tinyxml2::XMLDocument convolutional_document;
                tinyxml2::XMLNode* element_clone;

                element_clone = convolutional_element->DeepClone(&convolutional_document);

                convolutional_document.InsertFirstChild(element_clone);

                convolutional_layer->from_XML(convolutional_document);
            }

            add_layer(convolutional_layer);
        }
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

        throw invalid_argument(buffer.str());
    }

    // Outputs number

    const tinyxml2::XMLElement* outputs_number_element = root_element->FirstChildElement("OutputsNumber");

    if(!outputs_number_element)
    {
        buffer << "OpenNN Exception: NeuralNetwork class.\n"
               << "void inputs_from_XML(const tinyxml2::XMLDocument&) method.\n"
               << "Outputs number element is nullptr.\n";

        throw invalid_argument(buffer.str());
    }

    Index new_outputs_number = 0;

    if(outputs_number_element->GetText())
    {
        new_outputs_number = static_cast<Index>(atoi(outputs_number_element->GetText()));
    }

    // Outputs names

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

                throw invalid_argument(buffer.str());
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
}


/// Prints to the screen the most important information about the neural network object.

void NeuralNetwork::print() const
{
    cout << "Neural network" << endl;

    const Index layers_number = get_layers_number();

    cout << "Layers number: " << layers_number << endl;

    for(Index i = 0; i < layers_number; i++)
    {
        cout << "Layer " << i+1 << ": " << layers_pointers[i]->get_neurons_number()
             << " " << layers_pointers[i]->get_type_string() << " neurons" << endl;
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

        throw invalid_argument(buffer.str());
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

        throw invalid_argument(buffer.str());

    }

    from_XML(document);
}


/// Loads the neural network parameters from a data file.
/// The format of this file is just a sequence of numbers.
/// @param file_name Name of the parameters data file.

void NeuralNetwork::load_parameters_binary(const string& file_name)
{
    std::ifstream file;

    file.open(file_name.c_str(), ios::binary);

    if(!file.is_open())
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: NeuralNetwork template.\n"
               << "void load_parameters_binary(const string&) method.\n"
               << "Cannot open binary file: " << file_name << "\n";

        throw invalid_argument(buffer.str());
    }

    streamsize size = sizeof(float);

    const Index parameters_number = get_parameters_number();

    Tensor<type, 1> new_parameters(parameters_number);

    type value;

    for(Index i = 0; i < parameters_number; i++)
    {
        file.read(reinterpret_cast<char*>(&value), size);

        new_parameters(i) = value;
    }

    set_parameters(new_parameters);
}


/// Returns a string with the c function of the expression represented by the neural network.

string NeuralNetwork::write_expression_c() const
{

    //get_scaling_layer_pointer()->get_descriptives()

    int LSTM_number = get_long_short_term_memory_layers_number();
    int cell_state_counter = 0;
    int hidden_state_counter = 0;

    vector<std::string> found_tokens;
    ostringstream buffer;
    ostringstream calculate_outputs_buffer;

    string aux = "";

    bool logistic     = false;
    bool ReLU         = false;
    bool Threshold    = false;
    bool SymThreshold = false;
    bool ExpLinear    = false;
    bool SExpLinear   = false;
    bool HSigmoid     = false;
    bool SoftPlus     = false;
    bool SoftSign     = false;


    buffer << "/**" << endl;
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

    /*
    const Tensor<string, 1> inputs = get_inputs_names();
    const Tensor<string, 1> outputs = get_outputs_names();

    for (int i = 0; i < inputs.dimension(0); i++)
    {
        if (inputs[i].empty())
        {
            buffer << "\t" << to_string(i) + ") " << "input_" + to_string(i) << endl;
        }
        else
        {
            buffer << "\t" << to_string(i) + ") " << inputs[i] << endl;
        }
    }
    */

    const Tensor<string, 1> inputs_base = get_inputs_names();
    Tensor<string, 1> inputs(inputs_base.dimension(0));

    const Tensor<string, 1> outputs_base = get_outputs_names();
    Tensor<string, 1> outputs(outputs_base.dimension(0));

    string input_name_aux;
    string output_name_aux;

    for (int i = 0; i < inputs_base.dimension(0); i++)
    {
        if (inputs_base[i].empty())
        {
            buffer << "\t" << to_string(i) + ") " << "input_" + to_string(i) << endl;
        }
        else
        {
            input_name_aux = inputs_base[i];
            input_name_aux = replace_non_allowed_programming_characters(input_name_aux);
            inputs(i) = input_name_aux;
            buffer << "\t" << to_string(i) + ") " << inputs(i) << endl;
        }
    }

    for (int i = 0; i < outputs_base.dimension(0); i++)
    {
        if (outputs_base[i].empty())
        {
            buffer << "\t" << to_string(i) + ") " << "ouput_" + to_string(i) << endl;
        }
        else
        {
            output_name_aux = outputs_base[i];
            output_name_aux = replace_non_allowed_programming_characters(output_name_aux);
            outputs(i) = output_name_aux;
        }
    }

    buffer << "*/" << endl;
    buffer << "\n" << endl;
    buffer << "#include <iostream>" << endl;
    buffer << "#include <vector>" << endl;
    buffer << "#include <math.h>" << endl;
    buffer << "#include <stdio.h>" << endl;
    buffer << "\n" << endl;
    buffer << "using namespace std;" << endl;
    buffer << "\n" << endl;

    string expression = write_expression();
    vector<std::string> tokens;
    std::string token;
    std::stringstream ss(expression);

    while (getline(ss, token, '\n'))
    {
        if (token.size() > 1 && token.back() == '{'){ break; }
        if (token.size() > 1 && token.back() != ';'){ token += ';'; }
        tokens.push_back(token);
    }

    for (auto& s : tokens)
    {
        string word = "";
        for (char& c : s)
        {
            if ( c!=' ' && c!='=' ){ word += c; }
            else { break; }
        }
        if (word.size() > 1)
        {
            if ( find(found_tokens.begin(), found_tokens.end(), word) == found_tokens.end() )
                found_tokens.push_back(word);
        }
    }

    std::string target_string0("Logistic");
    std::string target_string1("ReLU");
    std::string target_string2("Threshold");
    std::string target_string3("SymmetricThreshold");
    std::string target_string4("ExponentialLinear");
    std::string target_string5("ScaledExponentialLinear");
    std::string target_string6("HardSigmoid");
    std::string target_string7("SoftPlus");
    std::string target_string8("SoftSign");

    for (auto& t:tokens)
    {
        size_t substring_length0 = t.find(target_string0);
        size_t substring_length1 = t.find(target_string1);
        size_t substring_length2 = t.find(target_string2);
        size_t substring_length3 = t.find(target_string3);
        size_t substring_length4 = t.find(target_string4);
        size_t substring_length5 = t.find(target_string5);
        size_t substring_length6 = t.find(target_string6);
        size_t substring_length7 = t.find(target_string7);
        size_t substring_length8 = t.find(target_string8);

        if (substring_length0 < t.size() && substring_length0!=0){ logistic = true; }
        if (substring_length1 < t.size() && substring_length1!=0){ ReLU = true; }
        if (substring_length2 < t.size() && substring_length2!=0){ Threshold = true; }
        if (substring_length3 < t.size() && substring_length3!=0){ SymThreshold = true; }
        if (substring_length4 < t.size() && substring_length4!=0){ ExpLinear = true; }
        if (substring_length5 < t.size() && substring_length5!=0){ SExpLinear = true; }
        if (substring_length6 < t.size() && substring_length6!=0){ HSigmoid = true; }
        if (substring_length7 < t.size() && substring_length7!=0){ SoftPlus = true; }
        if (substring_length8 < t.size() && substring_length8!=0){ SoftSign = true; }
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

    if(Threshold)
    {
        buffer << "float Threshold(float x) {" << endl;
        buffer << "float z;" << endl;
        buffer << "if (x < 0) {" << endl;
        buffer << "z = 0;" << endl;
        buffer << "}else{" << endl;
        buffer << "z = 1;" << endl;
        buffer << "}" << endl;
        buffer << "return z;" << endl;
        buffer << "}" << endl;
        buffer << "\n" << endl;
    }

    if(SymThreshold)
    {
        buffer << "float SymmetricThreshold(float x) {" << endl;
        buffer << "float z" << endl;
        buffer << "if (x < 0) {" << endl;
        buffer << "z = -1;" << endl;
        buffer << "}else{" << endl;
        buffer << "z=1;" << endl;
        buffer << "}" << endl;
        buffer << "return z;" << endl;
        buffer << "}" << endl;
        buffer << "\n" << endl;
    }

    if(ExpLinear)
    {
        buffer << "float ExponentialLinear(float x) {" << endl;
        buffer << "float z;" << endl;
        buffer << "float alpha = 1.67326;" << endl;
        buffer << "if (x>0){" << endl;
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
        buffer << "float ScaledExponentialLinear(float x) {" << endl;
        buffer << "float z;" << endl;
        buffer << "float alpha  = 1.67326;" << endl;
        buffer << "float lambda = 1.05070;" << endl;
        buffer << "if (x > 0){" << endl;
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
        for (string & token: found_tokens)
        {
            if (token.find("cell_state") == 0)
            {
                cell_state_counter += 1;
            }

            if (token.find("hidden_state") == 0)
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
            buffer << "\t" << "float hidden_state_" << to_string(i) << " = 0;" << endl;
        }

        for(int i = 0; i < cell_state_counter; i++)
        {
            buffer << "\t" << "float cell_state_" << to_string(i) << " = 0;" << endl;
        }

        buffer << "} lstm; \n\n" << endl;
        buffer << "vector<float> calculate_outputs(const vector<float>& inputs, LSTMMemory& lstm)" << endl;
    }
    else
    {
        buffer << "vector<float> calculate_outputs(const vector<float>& inputs)" << endl;
    }


    buffer << "{" << endl;

    for (int i = 0; i < inputs.dimension(0); i++)
    {
        if (inputs[i].empty())
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
            buffer << "\t\t" << "lstm.hidden_state_" << to_string(i) << " = 0;" << endl;
        }

        for(int i = 0; i < cell_state_counter; i++)
        {
            buffer << "\t\t" << "lstm.cell_state_" << to_string(i) << " = 0;" << endl;
        }

        buffer << "\t}" << endl;
    }

    buffer << "" << endl;

    //USING BUFFER2
    for (auto& t:tokens)
    {
        if (t.size()<=1)
        {
            calculate_outputs_buffer << "" << endl;
        }
        else
        {
            //aux = "const float " + t;
            calculate_outputs_buffer << "\t" << t << endl;
        }
    }


    string calculate_outputs_string = calculate_outputs_buffer.str();
    for (auto& found_token: found_tokens){
        string toReplace(found_token);

        //el fallo es el cont float, con double si que funciona
        string newword = "double " + found_token;
        size_t pos = calculate_outputs_string.find(toReplace);
        calculate_outputs_string.replace(pos, toReplace.length(), newword);
    }

    if(LSTM_number>0)
    {
        replace_all_appearances(calculate_outputs_string, "(t)", "");
        replace_all_appearances(calculate_outputs_string, "(t-1)", "");
        replace_all_appearances(calculate_outputs_string, "double cell_state", "cell_state");
        replace_all_appearances(calculate_outputs_string, "double hidden_state", "hidden_state");
        replace_all_appearances(calculate_outputs_string, "cell_state"  , "lstm.cell_state"  );
        replace_all_appearances(calculate_outputs_string, "hidden_state", "lstm.hidden_state");
    }
    buffer << calculate_outputs_string;

    buffer << "\t" << "vector<float> out(" << outputs.size() << ");" << endl;
    for (int i = 0; i < outputs.dimension(0); i++)
    {
        if (outputs[i].empty())
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

    for (int i = 0; i < inputs.dimension(0); i++)
    {
        if (inputs[i].empty())
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

    for (int i = 0; i < outputs.dimension(0); i++)
    {
        if (outputs[i].empty())
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
    return out;
}


string NeuralNetwork::write_expression() const
{
    const Index layers_number = get_layers_number();

    const Tensor<Layer*, 1> layers_pointers = get_layers_pointers();
    const Tensor<string, 1> layers_names = get_layers_names();

    Tensor<string, 1> outputs_names_vector;
    Tensor<string, 1> inputs_names_vector;
    inputs_names_vector = inputs_names;
    string aux_name = "";
    for (int i = 0; i < inputs_names.dimension(0); i++)
    {
        if (!inputs_names_vector[i].empty())
        {
            aux_name = inputs_names[i];
            inputs_names_vector(i) = replace_non_allowed_programming_characters(aux_name);
        }
        else
        {
            inputs_names_vector(i) = "input_" + to_string(i);
        }
    }

    Index layer_neurons_number;

    ostringstream buffer;

    for(Index i = 0; i < layers_number; i++)
    {
        if(i == layers_number-1)
        {
            outputs_names_vector = outputs_names;
            for (int i = 0; i < outputs_names.dimension(0); i++)
            {
                if (!outputs_names_vector[i].empty())
                {
                    aux_name = outputs_names[i];
                    outputs_names_vector(i) = replace_non_allowed_programming_characters(aux_name);
                }
                else
                {
                    outputs_names_vector(i) = "output_" + to_string(i);
                }
            }
            buffer << layers_pointers[i]->write_expression(inputs_names_vector, outputs_names_vector) << endl;
        }
        else
        {
            layer_neurons_number = layers_pointers[i]->get_neurons_number();
            outputs_names_vector.resize(layer_neurons_number);

            for(Index j = 0; j < layer_neurons_number; j++)
            {
                if(layers_names(i) == "scaling_layer")
                {
                    aux_name = inputs_names(j);
                    outputs_names_vector(j) = "scaled_" + replace_non_allowed_programming_characters(aux_name);
                }
                else
                {
                    outputs_names_vector(j) =  layers_names(i) + "_output_" + to_string(j);
                }
            }
            buffer << layers_pointers[i]->write_expression(inputs_names_vector, outputs_names_vector) << endl;
            inputs_names_vector = outputs_names_vector;
        }
    }

    string expression = buffer.str();

    replace(expression, "+-", "-");

    return expression;
}


/// Returns a string that conatins an API composed by an html script (the index page), and a php scipt
/// that contains a function of the expression represented by the neural network.

string NeuralNetwork::write_expression_api() const
{
    vector<std::string> found_tokens;
    ostringstream buffer;

    int LSTM_number = get_long_short_term_memory_layers_number();
    int cell_state_counter = 0;
    int hidden_state_counter = 0;

    bool logistic     = false;
    bool ReLU         = false;
    bool Threshold    = false;
    bool SymThreshold = false;
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

    const Tensor<string, 1> inputs_base = get_inputs_names();
    Tensor<string, 1> inputs(inputs_base.dimension(0));

    const Tensor<string, 1> outputs_base = get_outputs_names();
    Tensor<string, 1> outputs(outputs_base.dimension(0));

    string input_name_aux;
    string output_name_aux;

    for (int i = 0; i < inputs_base.dimension(0); i++)
    {
        if (inputs_base[i].empty())
        {
            inputs(i) = "input_" + to_string(i);
            buffer << "\t" << to_string(i) + ") " << inputs(i) << endl;
        }
        else
        {
            input_name_aux = inputs_base[i];
            input_name_aux = replace_non_allowed_programming_characters(input_name_aux);
            inputs(i) = input_name_aux;
            buffer << "\t" << to_string(i) + ") " << inputs(i) << endl;
        }
    }

    for (int i = 0; i < outputs_base.dimension(0); i++)
    {
        if (outputs_base[i].empty())
        {
            outputs(i) = "output_" + to_string(i);
            buffer << "\t" << to_string(i) + ") " << outputs(i) << endl;
        }
        else
        {
            output_name_aux = outputs_base[i];
            output_name_aux = replace_non_allowed_programming_characters(output_name_aux);
            outputs(i) = output_name_aux;
        }
    }

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

    string expression = write_expression();
    vector<std::string> tokens;
    std::string token;
    std::stringstream ss(expression);

    while (getline(ss, token, '\n'))
    {
        if (token.size() > 1 && token.back() == '{'){ break; }
        if (token.size() > 1 && token.back() != ';'){ token += ';'; }
        tokens.push_back(token);
    }

    for (auto& s : tokens)
    {
        string word = "";
        for (char& c : s)
        {
            if ( c!=' ' && c!='=' ){ word += c; }
            else { break; }
        }
        if (word.size() > 1)
        {
            found_tokens.push_back(word);
        }
    }

    if(LSTM_number>0)
    {
        for (string & token: found_tokens)
        {
            if (token.find("cell_state") == 0)
            {
                cell_state_counter += 1;
            }

            if (token.find("hidden_state") == 0)
            {
                hidden_state_counter += 1;
            }
        }

        buffer << "class NeuralNetwork{" << endl;
        buffer << "public $time_steps = 3;" << endl;
        buffer << "public $time_step_counter = 1;" << endl;

        for(int i = 0; i < hidden_state_counter; i++)
        {
            buffer << "public $" << "hidden_state_" << to_string(i) << " = 0;" << endl;
        }

        for(int i = 0; i < cell_state_counter; i++)
        {
            buffer << "public $" << "cell_state_" << to_string(i) << " = 0;" << endl;
        }

        buffer << "}" << endl;
        buffer << "$nn = new NeuralNetwork;" << endl;
    }

    buffer << "session_start();" << endl;
    buffer << "if (isset($_SESSION['lastpage']) && $_SESSION['lastpage'] == __FILE__) { " << endl;
    buffer << "if (isset($_SERVER['HTTPS']) && $_SERVER['HTTPS'] === 'on') " << endl;
    buffer << "\t$url = \"https://\"; " << endl;
    buffer << "else" << endl;
    buffer << "\t$url = \"http://\"; " << endl;
    buffer << "\n" << endl;
    buffer << "$url.= $_SERVER['HTTP_HOST'];" << endl;
    buffer << "$url.= $_SERVER['REQUEST_URI'];" << endl;
    buffer << "$url_components = parse_url($url);" << endl;
    buffer << "parse_str($url_components['query'], $params);" << endl;
    buffer << "\n" << endl;

    for (int i = 0; i < inputs.dimension(0); i++)
    {
        if (inputs[i].empty())
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

    for (int i = 0; i < inputs.dimension(0); i++)
    {
        if (i != inputs.dimension(0)-1)
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
            buffer << "$nn->" << "hidden_state_" << to_string(i) << " = 0;" << endl;
        }

        for(int i = 0; i < cell_state_counter; i++)
        {
            buffer << "$nn->" << "cell_state_" << to_string(i) << " = 0;" << endl;
        }
        buffer << "}" << endl;
    }

    buffer << "\n" << endl;

    std::string target_string0("Logistic");
    std::string target_string1("ReLU");
    std::string target_string2("Threshold");
    std::string target_string3("SymmetricThreshold");
    std::string target_string4("ExponentialLinear");
    std::string target_string5("ScaledExponentialLinear");
    std::string target_string6("HardSigmoid");
    std::string target_string7("SoftPlus");
    std::string target_string8("SoftSign");

    size_t substring_length0;
    size_t substring_length1;
    size_t substring_length2;
    size_t substring_length3;
    size_t substring_length4;
    size_t substring_length5;
    size_t substring_length6;
    size_t substring_length7;
    size_t substring_length8;

    vector<string> words_in_senteces;
    string new_word;

    for (auto& t:tokens)
    {
        substring_length0 = t.find(target_string0);
        substring_length1 = t.find(target_string1);
        substring_length2 = t.find(target_string2);
        substring_length3 = t.find(target_string3);
        substring_length4 = t.find(target_string4);
        substring_length5 = t.find(target_string5);
        substring_length6 = t.find(target_string6);
        substring_length7 = t.find(target_string7);
        substring_length8 = t.find(target_string8);

        if (substring_length0 < t.size() && substring_length0!=0){ logistic     = true; }
        if (substring_length1 < t.size() && substring_length1!=0){ ReLU         = true; }
        if (substring_length2 < t.size() && substring_length2!=0){ Threshold    = true; }
        if (substring_length3 < t.size() && substring_length3!=0){ SymThreshold = true; }
        if (substring_length4 < t.size() && substring_length4!=0){ ExpLinear    = true; }
        if (substring_length5 < t.size() && substring_length5!=0){ SExpLinear   = true; }
        if (substring_length6 < t.size() && substring_length6!=0){ HSigmoid     = true; }
        if (substring_length7 < t.size() && substring_length7!=0){ SoftPlus     = true; }
        if (substring_length8 < t.size() && substring_length8!=0){ SoftSign     = true; }

        for (auto& key_word : found_tokens)
        {
            new_word.clear();
            new_word = "$" + key_word;
            replace_all_appearances(t, key_word, new_word);
        }

        for (int i = 0; i < inputs.dimension(0); i++)
        {
            new_word.clear();
            new_word = "$" + inputs[i];
            replace_all_appearances(t, inputs[i], new_word);
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
    }

    buffer << "if ($status === 200){" << endl;
    buffer << "$response = ['status' => $status,  'status_message' => $status_msg" << endl;

    for (int i = 0; i < outputs.dimension(0); i++)
    {
        buffer << ", '" << outputs(i) << "' => " << "$" << outputs[i] << endl;
    }

    buffer << "];" << endl;
    buffer << "}" << endl;
    buffer << "else" << endl;
    buffer << "{" << endl;
    buffer << "$response = ['status' => $status,  'status_message' => $status_msg" << "];" << endl;
    buffer << "}" << endl;

    if (LSTM_number>0)
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

    if(Threshold)
    {
        buffer << "<?php" << endl;
        buffer << "function Threshold(int $x) {" << endl;
        buffer << "if ($x < 0) {" << endl;
        buffer << "$z = 0;" << endl;
        buffer << "}else{" << endl;
        buffer << "$z=1;" << endl;
        buffer << "}" << endl;
        buffer << "return $z;" << endl;
        buffer << "}" << endl;
        buffer << "?>" << endl;
        buffer << "\n" << endl;
    }

    if(SymThreshold)
    {
        buffer << "<?php" << endl;
        buffer << "function SymmetricThreshold(int $x) {" << endl;
        buffer << "if ($x < 0) {" << endl;
        buffer << "$z = -1;" << endl;
        buffer << "}else{" << endl;
        buffer << "$z=1;" << endl;
        buffer << "}" << endl;
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
        buffer << "if ($x>0){" << endl;
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
        buffer << "function ScaledExponentialLinear(int $x) {" << endl;
        buffer << "$alpha  = 1.67326;" << endl;
        buffer << "$lambda = 1.05070;" << endl;
        buffer << "if ($x>0){" << endl;
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

    vector<std::string> found_tokens;
    vector<std::string> found_tokens2;
    ostringstream buffer;

    string expression = write_expression();
    vector<std::string> tokens;
    std::string token;
    std::stringstream ss(expression);

    int LSTM_number = get_long_short_term_memory_layers_number();
    int cell_state_counter = 0;
    int hidden_state_counter = 0;

    bool logistic     = false;
    bool ReLU         = false;
    bool Threshold    = false;
    bool SymThreshold = false;
    bool ExpLinear    = false;
    bool SExpLinear   = false;
    bool HSigmoid     = false;
    bool SoftPlus     = false;
    bool SoftSign     = false;

    buffer << "<!--" << endl;
    buffer << "Artificial Intelligence Techniques SL\t" << endl;
    buffer << "artelnics@artelnics.com\t" << endl;
    buffer << "" << endl;
    buffer << "Your model has been exported to this JavaScript file." << endl;
    buffer << "You can manage it with the main method, where you \t" << endl;
    buffer << "can change the values of your inputs. For example:" << endl;
    buffer << "can change the values of your inputs. For example:" << endl;
    buffer << "" << endl;
    buffer << "if we want to add these 3 values (0.3, 2.5 and 1.8)" << endl;
    buffer << "to our 3 inputs (Input_1, Input_2 and Input_1), the" << endl;
    buffer << "main program has to look like this:" << endl;
    buffer << "\t" << endl;
    buffer << "int neuralNetwork(){ " << endl;
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

    const Tensor<string, 1> inputs_base = get_inputs_names();
    Tensor<string, 1> inputs(inputs_base.dimension(0));

    const Tensor<string, 1> outputs_base = get_outputs_names();
    Tensor<string, 1> outputs(outputs_base.dimension(0));

    string input_name_aux;
    string output_name_aux;

    for (int i = 0; i < inputs_base.dimension(0); i++)
    {
        if (inputs_base[i].empty())
        {
            inputs(i) = "input_" + to_string(i);
            buffer << "\t" << to_string(i) + ") " << inputs(i) << endl;
        }
        else
        {
            input_name_aux = inputs_base[i];
            input_name_aux = replace_non_allowed_programming_characters(input_name_aux);
            inputs(i) = input_name_aux;
            buffer << "\t" << to_string(i) + ") " << inputs(i) << endl;
        }
    }

    for (int i = 0; i < outputs_base.dimension(0); i++)
    {
        if (outputs_base[i].empty())
        {
            outputs(i) = "output_" + to_string(i);
            buffer << "\t" << to_string(i) + ") " << outputs(i) << endl;
        }
        else
        {
            output_name_aux = outputs_base[i];
            output_name_aux = replace_non_allowed_programming_characters(output_name_aux);
            outputs(i) = output_name_aux;
        }
    }

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

    if (has_scaling_layer())
    {
        Tensor<Descriptives, 1>  inputs_descriptives = get_scaling_layer_pointer()->get_descriptives();

        for (int i = 0; i < inputs.dimension(0); i++)
        {
            buffer << "<!-- "<< to_string(i) <<"scaling layer -->" << endl;
            buffer << "<tr style=\"height:3.5em\">" << endl;
            buffer << "<td> " << inputs_names[i] << " </td>" << endl;
            buffer << "<td style=\"text-align:center\">" << endl;
            buffer << "<input type=\"range\" id=\"" << inputs[i] << "\" value=\"" << (inputs_descriptives(0).minimum + inputs_descriptives(0).maximum)/2 << "\" min=\"" << inputs_descriptives(0).minimum << "\" max=\"" << inputs_descriptives(0).maximum << "\" step=\"0.01\" onchange=\"updateTextInput1(this.value, '" << inputs[i] << "_text')\" />" << endl;
            buffer << "<input class=\"tabla\" type=\"number\" id=\"" << inputs[i] << "_text\" value=\"" << (inputs_descriptives(0).minimum + inputs_descriptives(0).maximum)/2 << "\" min=\"" << inputs_descriptives(0).minimum << "\" max=\"" << inputs_descriptives(0).maximum << "\" step=\"0.01\" onchange=\"updateTextInput1(this.value, '" << inputs[i] << "')\">" << endl;
            buffer << "</td>" << endl;
            buffer << "</tr>" << endl;
            buffer << "\n" << endl;
        }
    }
    else
    {
        for (int i = 0; i < inputs.dimension(0); i++)
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
    buffer << "<div align=\"center\">" << endl;
    buffer << "<!-- BUTTON HERE -->" << endl;
    buffer << "<button class=\"btn\" onclick=\"neuralNetwork()\">calculate outputs</button>" << endl;
    buffer << "</div>" << endl;
    buffer << "\n" << endl;
    buffer << "<br/>" << endl;
    buffer << "\n" << endl;
    buffer << "<table border=\"1px\" class=\"form\">" << endl;
    buffer << "OUTPUTS" << endl;

    for (int i = 0; i < outputs.dimension(0); i++)
    {
        buffer << "<tr style=\"height:3.5em\">" << endl;
        buffer << "<td> " << outputs_names[i] << " </td>" << endl;
        buffer << "<td>" << endl;
        buffer << "<input style=\"text-align:right; padding-right:20px;\" id=\"" << outputs[i] << "\" value=\"\" type=\"text\"  disabled/>" << endl;
        buffer << "</td>" << endl;
        buffer << "</tr>" << endl;
        buffer << "\n" << endl;
    }

    buffer << "</table>" << endl;
    buffer << "\n" << endl;
    buffer << "</form>" << endl;
    buffer << "</div>" << endl;
    buffer << "\n" << endl;
    buffer << "</section>" << endl;
    buffer << "\n" << endl;
    buffer << "<script>" << endl;
    buffer << "function neuralNetwork()" << endl;
    buffer << "{" << endl;
    buffer << "\t" << "var inputs = [];" << endl;

    for (int i = 0; i < inputs.dimension(0); i++)
    {
        buffer << "\t" << "var " << inputs[i] << " =" << " document.getElementById(\"" << inputs[i] << "\").value; " << endl;
        buffer << "\t" << "inputs.push(" << inputs[i] << ");" << endl;
    }

    buffer << "\n" << "\t" << "var outputs = calculate_outputs(inputs); " << endl;

    for (int i = 0; i < outputs.dimension(0); i++)
    {
        buffer << "\t" << "var " << outputs[i] << " = document.getElementById(\"" << outputs[i] << "\");" << endl;
        buffer << "\t" << outputs[i] << ".value = outputs[" << to_string(i) << "].toFixed(4);" << endl;
    }

    buffer << "\t" << "update_LSTM();" << endl;
    buffer << "}" << "\n" << endl;

    while (getline(ss, token, '\n'))
    {
        if (token.size() > 1 && token.back() == '{'){ break; }
        if (token.size() > 1 && token.back() != ';'){ token += ';'; }
        tokens.push_back(token);
    }

    buffer << "function calculate_outputs(inputs)" << endl;
    buffer << "{" << endl;

    for (int i = 0; i < inputs.dimension(0); i++)
    {
        buffer << "\t" << "var " << inputs[i] << " = " << "inputs[" << to_string(i) << "];" << endl;
    }

    buffer << "" << endl;

    for (auto& t:tokens)
    {
        string word = "";
        for (char& c : t)
        {
            if ( c!=' ' && c!='=' ){ word += c; }
            else { break; }
        }
        if (word.size() > 1)
        {
            found_tokens.push_back(word);
        }
    }

    if(LSTM_number>0)
    {
        for (string & token: found_tokens)
        {
            if (token.find("cell_state") == 0)
            {
                cell_state_counter += 1;
            }

            if (token.find("hidden_state") == 0)
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

        for(int i = 0; i < cell_state_counter; i++)
        {
            buffer << "\t\t" << "cell_state_" << to_string(i) << " = 0" << endl;
        }

        buffer << "\t}\n" << endl;
    }

    std::string target_string0("Logistic");
    std::string target_string1("ReLU");
    std::string target_string2("Threshold");
    std::string target_string3("SymmetricThreshold");
    std::string target_string4("ExponentialLinear");
    std::string target_string5("ScaledExponentialLinear");
    std::string target_string6("HardSigmoid");
    std::string target_string7("SoftPlus");
    std::string target_string8("SoftSign");

    found_tokens2.push_back("exp");
    found_tokens2.push_back("tanh");
    found_tokens2.push_back("max");
    found_tokens2.push_back("min");
    string sufix = "Math.";

    for (auto& t:tokens)
    {
        size_t substring_length0 = t.find(target_string0);
        size_t substring_length1 = t.find(target_string1);
        size_t substring_length2 = t.find(target_string2);
        size_t substring_length3 = t.find(target_string3);
        size_t substring_length4 = t.find(target_string4);
        size_t substring_length5 = t.find(target_string5);
        size_t substring_length6 = t.find(target_string6);
        size_t substring_length7 = t.find(target_string7);
        size_t substring_length8 = t.find(target_string8);

        if (substring_length0 < t.size() && substring_length0!=0){ logistic = true; }
        if (substring_length1 < t.size() && substring_length1!=0){ ReLU = true; }
        if (substring_length2 < t.size() && substring_length2!=0){ Threshold = true; }
        if (substring_length3 < t.size() && substring_length3!=0){ SymThreshold = true; }
        if (substring_length4 < t.size() && substring_length4!=0){ ExpLinear = true; }
        if (substring_length5 < t.size() && substring_length5!=0){ SExpLinear = true; }
        if (substring_length6 < t.size() && substring_length6!=0){ HSigmoid = true; }
        if (substring_length7 < t.size() && substring_length7!=0){ SoftPlus = true; }
        if (substring_length8 < t.size() && substring_length8!=0){ SoftSign = true; }

        for (auto& key_word : found_tokens2)
        {
            string new_word = "";
            new_word = sufix + key_word;
            replace_all_appearances(t, key_word, new_word);
        }

        if (t.size()<=1)
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

    buffer << "\t" << "var out = [];" << endl;

    for (int i = 0; i < outputs.dimension(0); i++)
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

        for(int i = 0; i < cell_state_counter; i++)
        {
            buffer << "\t" << "var " << "var cell_state_" << to_string(i) << " = 0" << endl;
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

    if(Threshold)
    {
        buffer << "function Threshold(x) {" << endl;
        buffer << "\tif (x < 0) {" << endl;
        buffer << "\t\tvar z = 0;" << endl;
        buffer << "\t}else{" << endl;
        buffer << "\t\tvar z = 1;" << endl;
        buffer << "\t}" << endl;
        buffer << "\treturn z;" << endl;
        buffer << "}" << endl;
        buffer << "\n" << endl;
    }

    if(SymThreshold)
    {
        buffer << "function SymmetricThreshold(x) {" << endl;
        buffer << "\tif (x < 0) {" << endl;
        buffer << "\t\tvar z = -1;" << endl;
        buffer << "\t}else{" << endl;
        buffer << "\t\tvar z=1;" << endl;
        buffer << "\t}" << endl;
        buffer << "\treturn z;" << endl;
        buffer << "}" << endl;
        buffer << "\n" << endl;
    }

    if(ExpLinear)
    {
        buffer << "function ExponentialLinear(x) {" << endl;
        buffer << "\tvar alpha = 1.67326;" << endl;
        buffer << "\tif (x>0){" << endl;
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
        buffer << "function ScaledExponentialLinear(x) {" << endl;
        buffer << "\tvar alpha  = 1.67326;" << endl;
        buffer << "\tvar lambda = 1.05070;" << endl;
        buffer << "\tif (x>0){" << endl;
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
    vector<std::string> found_tokens;
    vector<std::string> found_tokens2;

    int LSTM_number = get_long_short_term_memory_layers_number();
    int cell_state_counter = 0;
    int hidden_state_counter = 0;

    bool logistic     = false;
    bool ReLU         = false;
    bool Threshold    = false;
    bool SymThreshold = false;
    bool ExpLinear    = false;
    bool SExpLinear   = false;
    bool HSigmoid     = false;
    bool SoftPlus     = false;
    bool SoftSign     = false;

    buffer << "\'\'\' " << endl;
    buffer << "Artificial Intelligence Techniques SL\t" << endl;
    buffer << "artelnics@artelnics.com\t" << endl;
    buffer << "" << endl;
    buffer << "Your model has been exported to this python file." << endl;
    buffer << "You can manage it with the main method where you \t" << endl;
    buffer << "can change the values of your inputs. For example:" << endl;
    buffer << "" << endl;
    buffer << "if we want to add these 3 values (0.3, 2.5 and 1.8)" << endl;
    buffer << "to our 3 inputs (Input_1, Input_2 and Input_1), the" << endl;
    buffer << "main program has to look like this:" << endl;
    buffer << "" << endl;
    buffer << "def main ():" << endl;
    buffer << "\t" << "#default_val = 3.1416" << endl;
    buffer << "\t" << "inputs = [None]*3" << endl;
    buffer << "\t" <<  "" << endl;
    buffer << "\t" << "Id_1 = 0.3" << endl;
    buffer << "\t" << "Id_1 = 2.5" << endl;
    buffer << "\t" << "Id_1 = 1.8" << endl;
    buffer << "\t" << "" << endl;
    buffer << "\t" << "inputs[0] = Input_1" << endl;
    buffer << "\t" << "inputs[1] = Input_2" << endl;
    buffer << "\t" << "inputs[2] = Input_3" << endl;
    buffer << "\t" << ". . ." << endl;
    buffer << "\n" << endl;
    buffer << "Inputs Names: \t" << endl;

    const Tensor<string, 1> inputs_base = get_inputs_names();
    Tensor<string, 1> inputs(inputs_base.dimension(0));

    const Tensor<string, 1> outputs_base = get_outputs_names();
    Tensor<string, 1> outputs(outputs_base.dimension(0));

    string input_name_aux;
    string output_name_aux;

    for (int i = 0; i < inputs_base.dimension(0); i++)
    {
        if (inputs_base[i].empty())
        {
            inputs(i) = "input_" + to_string(i);
            buffer << "\t" << to_string(i) + ") " << inputs(i) << endl;
        }
        else
        {
            input_name_aux = inputs_base[i];
            input_name_aux = replace_non_allowed_programming_characters(input_name_aux);
            inputs(i) = input_name_aux;
            buffer << "\t" << to_string(i) + ") " << inputs(i) << endl;
        }
    }

    for (int i = 0; i < outputs_base.dimension(0); i++)
    {
        if (outputs_base[i].empty())
        {
            outputs(i) = "output_" + to_string(i);
            buffer << "\t" << to_string(i) + ") " << outputs(i) << endl;
        }
        else
        {
            output_name_aux = outputs_base[i];
            output_name_aux = replace_non_allowed_programming_characters(output_name_aux);
            outputs(i) = output_name_aux;
        }
    }

    buffer << "\n" << endl;
    buffer << "\'\'\' " << endl;
    buffer << "\n" << endl;

    string expression = write_expression();
    vector<std::string> tokens;
    std::string token;
    std::stringstream ss(expression);

    while (getline(ss, token, '\n'))
    {
        if (token.size() > 1 && token.back() == '{'){ break; }
        if (token.size() > 1 && token.back() != ';'){ token += ';'; }
        tokens.push_back(token);
    }

    std::string target_string0("Logistic");
    std::string target_string1("ReLU");
    std::string target_string2("Threshold");
    std::string target_string3("SymmetricThreshold");
    std::string target_string4("ExponentialLinear");
    std::string target_string5("ScaledExponentialLinear");
    std::string target_string6("HardSigmoid");
    std::string target_string7("SoftPlus");
    std::string target_string8("SoftSign");

    for (auto& t:tokens)
    {
        size_t substring_length0 = t.find(target_string0);
        size_t substring_length1 = t.find(target_string1);
        size_t substring_length2 = t.find(target_string2);
        size_t substring_length3 = t.find(target_string3);
        size_t substring_length4 = t.find(target_string4);
        size_t substring_length5 = t.find(target_string5);
        size_t substring_length6 = t.find(target_string6);
        size_t substring_length7 = t.find(target_string7);
        size_t substring_length8 = t.find(target_string8);

        if (substring_length0 < t.size() && substring_length0!=0){ logistic = true; }
        if (substring_length1 < t.size() && substring_length1!=0){ ReLU = true; }
        if (substring_length2 < t.size() && substring_length2!=0){ Threshold = true; }
        if (substring_length3 < t.size() && substring_length3!=0){ SymThreshold = true; }
        if (substring_length4 < t.size() && substring_length4!=0){ ExpLinear = true; }
        if (substring_length5 < t.size() && substring_length5!=0){ SExpLinear = true; }
        if (substring_length6 < t.size() && substring_length6!=0){ HSigmoid = true; }
        if (substring_length7 < t.size() && substring_length7!=0){ SoftPlus = true; }
        if (substring_length8 < t.size() && substring_length8!=0){ SoftSign = true; }

        string word = "";

        for (char& c : t)
        {
            if ( c!=' ' && c!='=' ){ word += c; }
            else { break; }
        }

        if (word.size() > 1)
            found_tokens.push_back(word);
    }

    for (string & token: found_tokens)
    {
        if (token.find("cell_state") == 0)
            cell_state_counter += 1;

        if (token.find("hidden_state") == 0)
            hidden_state_counter += 1;
    }

    buffer << "import math" << endl;
    buffer << "import numpy as np" << endl;
    buffer << "\n" << endl;
    buffer << "class NeuralNetwork:" << endl;

    if (LSTM_number > 0)
    {
        buffer << "\t" << "def __init__(self, ts = 0):" << endl;
        buffer << "\t\t" << "self.inputs_number = " << to_string(inputs.size()) << endl;
        buffer << "\t\t" << "self.time_steps = ts" << endl;

        for(int i = 0; i < hidden_state_counter; i++)
        {
            buffer << "\t\t" << "self.hidden_state_" << to_string(i) << " = 0" << endl;
        }

        for(int i = 0; i < cell_state_counter; i++)
        {
            buffer << "\t\t" << "self.cell_state_" << to_string(i) << " = 0" << endl;
        }

        buffer << "\t\t" << "self.time_step_counter = 1" << endl;
    }
    else
    {
        buffer << "\t" << "def __init__(self):" << endl;
        buffer << "\t\t" << "self.inputs_number = " << to_string(inputs.size()) << endl;
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

    if(Threshold)
    {
        buffer << "\tdef Threshold (x):" << endl;
        buffer << "\t\t"   << "if (x < 0):" << endl;
        buffer << "\t\t\t" << "z = 0" << endl;
        buffer << "\t\t"   << "else:" << endl;
        buffer << "\t\t\t" << "z = 1" << endl;
        buffer << "\t\t"   << "return z" << endl;
        buffer << "\n" << endl;
    }

    if(SymThreshold)
    {
        buffer << "\tdef SymmetricThreshold (x):" << endl;
        buffer << "\t\t"   << "if (x < 0):" << endl;
        buffer << "\t\t\t" << "z = -1" << endl;
        buffer << "\t\t"   << "else:" << endl;
        buffer << "\t\t\t" << "z = 1" << endl;
        buffer << "\t\t"   << "return z" << endl;
        buffer << "\n" << endl;
    }

    if(ExpLinear)
    {
        buffer << "\tdef ExponentialLinear (x):" << endl;
        buffer << "\t\t"   << "float alpha = 1.67326" << endl;
        buffer << "\t\t"   << "if (x>0):" << endl;
        buffer << "\t\t\t" << "z = x" << endl;
        buffer << "\t\t"   << "else:" << endl;
        buffer << "\t\t\t" << "z = alpha*(np.exp(x)-1)" << endl;
        buffer << "\t\t"   << "return z" << endl;
        buffer << "\n" << endl;
    }

    if(SExpLinear)
    {
        buffer << "\tdef ScaledExponentialLinear (x):" << endl;
        buffer << "\t\t"   << "float alpha = 1.67326" << endl;
        buffer << "\t\t"   << "float lambda = 1.05070" << endl;
        buffer << "\t\t"   << "if (x>0):" << endl;
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

    for (int i = 0; i < inputs.dimension(0); i++)
    {
        buffer << "\t\t" << inputs[i] << " = " << "inputs[" << to_string(i) << "]" << endl;
    }

    if (LSTM_number>0)
    {
        buffer << "\n\t\t" << "if( self.time_step_counter % self.time_steps == 0 ):" << endl;
        buffer << "\t\t\t" << "self.t = 1" << endl;

        for(int i = 0; i < hidden_state_counter; i++)
        {
            buffer << "\t\t\t" << "self.hidden_state_" << to_string(i) << " = 0" << endl;
        }

        for(int i = 0; i < cell_state_counter; i++)
        {
            buffer << "\t\t\t" << "self.cell_state_" << to_string(i) << " = 0" << endl;
        }
    }

    buffer << "" << endl;

    found_tokens.clear();
    found_tokens.push_back("exp");
    found_tokens.push_back("tanh");

    found_tokens2.push_back("Logistic");
    found_tokens2.push_back("ReLU");
    found_tokens2.push_back("Threshold");
    found_tokens2.push_back("SymmetricThreshold");
    found_tokens2.push_back("ExponentialLinear");
    found_tokens2.push_back("ScaledExponentialLinear");
    found_tokens2.push_back("HardSigmoid");
    found_tokens2.push_back("SoftPlus");
    found_tokens2.push_back("SoftSign");

    string sufix;
    string new_word;

    for (auto& t:tokens)
    {
        new_word = "";
        sufix = "np.";

        for (auto& key_word : found_tokens)
        {
            new_word = "";
            new_word = sufix + key_word;
            replace_all_appearances(t, key_word, new_word);
        }

        new_word = "";
        sufix = "NeuralNetwork.";

        for (auto& key_word : found_tokens2)
        {
            new_word = "";
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

    buffer << "\t\t" << "out = " << "[None]*" << outputs.size() << "" << endl;

    for (int i = 0; i < outputs.dimension(0); i++)
    {
        buffer << "\t\t" << "out[" << to_string(i) << "] = " << outputs[i] << endl;
    }

    if(LSTM_number>0)
    {
        buffer << "\n\t\t" << "self.time_step_counter += 1" << endl;
    }

    buffer << "\n\t\t" << "return out;" << endl;
    buffer << "\n" << endl;
    buffer << "\t"   << "def main (self):" << endl;
    buffer << "\t\t" << "default_val = 3.1416" << endl;
    buffer << "\t\t" << "inputs = [None]*" << to_string(inputs.size()) << "\n" << endl;

    for (int i = 0; i < inputs.dimension(0); i++)
    {
        buffer << "\t\t" << inputs[i] << " = " << "default_val" << "\t" << "#Change this value" << endl;
        buffer << "\t\t" << "inputs[" << to_string(i) << "] = " << inputs[i] << "\n" << endl;
    }

    buffer << "" << endl;
    buffer << "\t\t" << "outputs = NeuralNetwork.calculate_outputs(self, inputs)" << endl;
    buffer << "" << endl;
    buffer << "\t\t" << "print(\"\\nThese are your outputs:\\n\")" << endl;

    for (int i = 0; i < outputs.dimension(0); i++)
    {
        buffer << "\t\t" << "print( \""<< "\\t " << outputs[i] << ":\" "<< "+ " << "str(outputs[" << to_string(i) << "])" << " + " << "\"\\n\" )" << endl;
    }

    buffer << "\n";

    if (LSTM_number>0){
        buffer << "steps = 3" << endl;
        buffer << "nn = NeuralNetwork(steps)" << endl;
        buffer << "nn.main()" << endl;
    }
    else
    {
        buffer << "nn = NeuralNetwork()" << endl;
        buffer << "nn.main()" << endl;
    }

    string out = buffer.str();
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

        throw invalid_argument(buffer.str());
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

        throw invalid_argument(buffer.str());
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

        throw invalid_argument(buffer.str());
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

        throw invalid_argument(buffer.str());
    }

    file << write_expression_python();

    file.close();
}


/// Saves a csv file containing the outputs for a set of given inputs.
/// @param inputs Inputs to calculate the outputs.
/// @param file_name Name of the data file

void NeuralNetwork::save_outputs(Tensor<type, 2>& inputs, const string & file_name)
{
    Tensor<Index, 1> inputs_dimensions = get_dimensions(inputs);

    Tensor<type, 2> outputs(inputs_dimensions(0), get_outputs_number());

    Tensor<Index, 1> outputs_dimensions = get_dimensions(outputs);

    calculate_outputs(inputs.data(), inputs_dimensions, outputs.data(), outputs_dimensions);

    std::ofstream file(file_name.c_str());

    if(!file.is_open())
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: NeuralNetwork class.\n"
               << "void  save_expression_python(const string&) method.\n"
               << "Cannot open expression text file.\n";

        throw invalid_argument(buffer.str());
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
        layers_names[i] = layers_pointers[i]->get_name();
    }

    return layers_names;
}


Layer* NeuralNetwork::get_last_trainable_layer_pointer() const
{
    if(layers_pointers.size() == 0) return nullptr;

    Tensor<Layer*, 1> trainable_layers_pointers = get_trainable_layers_pointers();

    const Index trainable_layers_number = get_trainable_layers_number();

    return trainable_layers_pointers(trainable_layers_number-1);
}

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2021 Artificial Intelligence Techniques, SL.
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
