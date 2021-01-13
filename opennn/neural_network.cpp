//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   N E U R A L   N E T W O R K   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "neural_network.h"

namespace OpenNN
{

/// Default constructor.
/// It creates an empty neural network object.
/// All pointers in the object are initialized to nullptr.
/// The rest of members are initialized to their default values.

NeuralNetwork::NeuralNetwork()
{
    set();
}


/// Type of model and architecture of the Neural Network constructor.
/// It creates a neural network object with the given model type and architecture.
/// The rest of members are initialized to their default values.
/// @param model_type Type of problem to be solved with the neural network
/// (Approximation, Classification, Forecasting, ImageApproximation, ImageClassification).
/// @param architecture Architecture of the neural network({inputs_number, hidden_neurons_number, outputs_number}).

NeuralNetwork::NeuralNetwork(const NeuralNetwork::ProjectType& model_type, const Tensor<Index, 1>& architecture)
{
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
/// Please be careful with the format of that file, which is specified in the OpenNN manual.
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
/// It also sets the rest of members to their default values.

NeuralNetwork::NeuralNetwork(const Tensor<Layer*, 1>& new_layers_pointers)
{
    set();

    layers_pointers = new_layers_pointers;
}


/// Destructor.

NeuralNetwork::~NeuralNetwork()
{
    const Index layers_number = get_layers_number();

    for(Index i = 0;  i < layers_number; i++)
    {
        delete layers_pointers[i];

        layers_pointers[i] = nullptr;
    }
}


/// Add a new layer to the Neural Network model.
/// @param layer The layer that will be added.
/// @todo break the software.

void NeuralNetwork::add_layer(Layer* layer_pointer)
{

    if(layer_pointer->get_type_string() == "Recurrent" || layer_pointer->get_type_string() == "LongShortTermMemory"){
        ostringstream buffer;

        buffer << "OpenNN Exception: NeuralNetwork class.\n"
               << "NeuralNetwork::add_layer() method.\n"
               << "Long Short Term Memory Layer and Recurrent Layer are not available yet. Both of them will be included in future versions.\n";

        throw logic_error(buffer.str());

    }
    if(layer_pointer->get_type_string() == "Convolutional"){
        ostringstream buffer;

        buffer << "OpenNN Exception: NeuralNetwork class.\n"
               << "NeuralNetwork::add_layer() method.\n"
               << "Convolutional Layer is not available yet. It will be included in future versions.!!\n";

        throw logic_error(buffer.str());
    }

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
               << "Layer type (" << layer_pointer->get_type_string() << ") cannot be added in position " << layers_pointers.size()
               << " to the neural network architecture.\n";

        throw logic_error(buffer.str());
    }
}


/// Check if a given layer type can be added to the structure of the neural network.
/// LSTM and Recurrent layers can only be added at the beginning.
/// @param layer_type Type of new layer to be added.

bool NeuralNetwork::check_layer_type(const Layer::Type layer_type)
{
    const Index layers_number = layers_pointers.size();

    if(layers_number > 1 && (layer_type == Layer::Recurrent || layer_type == Layer::LongShortTermMemory))
    {
        return false;
    }
    else if(layers_number == 1 && (layer_type == Layer::Recurrent || layer_type == Layer::LongShortTermMemory))
    {
        const Layer::Type first_layer_type = layers_pointers[0]->get_type();

        if(first_layer_type != Layer::Scaling) return false;
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
        if(layers_pointers[i]->get_type() == Layer::Scaling) return true;
    }

    return false;
}


/// Returns true if the neural network object has a principal components layer object inside,
/// and false otherwise.

bool NeuralNetwork::has_principal_components_layer() const
{
    const Index layers_number = get_layers_number();

    for(Index i = 0; i < layers_number; i++)
    {
        if(layers_pointers[i]->get_type() == Layer::PrincipalComponents) return true;
    }

    return false;
}


/// Returns true if the neural network object has a long short term memory layer object inside,
/// and false otherwise.

bool NeuralNetwork::has_long_short_term_memory_layer() const
{
    const Index layers_number = get_layers_number();

    for(Index i = 0; i < layers_number; i++)
    {
        if(layers_pointers[i]->get_type() == Layer::LongShortTermMemory) return true;
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
        if(layers_pointers[i]->get_type() == Layer::Convolutional) return true;
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
        if(layers_pointers[i]->get_type() == Layer::Recurrent) return true;
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
        if(layers_pointers[i]->get_type() == Layer::Unscaling) return true;
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
        if(layers_pointers[i]->get_type() == Layer::Bounding) return true;
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
        if(layers_pointers[i]->get_type() == Layer::Probabilistic) return true;
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


/// Returns a pointer to the trainable layers object composing this neural network object.

Tensor<Layer*, 1> NeuralNetwork::get_trainable_layers_pointers() const
{
    const Index layers_number = get_layers_number();

    const Index trainable_layers_number = get_trainable_layers_number();

    Tensor<Layer*, 1> trainable_layers_pointers(trainable_layers_number);

    Index trainable_layer_index = 0;

    for(Index i = 0; i < layers_number; i++)
    {
        if(layers_pointers[i]->get_type() != Layer::Scaling
        && layers_pointers[i]->get_type() != Layer::Unscaling
        && layers_pointers[i]->get_type() != Layer::Bounding)
        {
            trainable_layers_pointers[trainable_layer_index] = layers_pointers[i];
            trainable_layer_index++;
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
        if(layers_pointers[i]->get_type() != Layer::Scaling
        && layers_pointers[i]->get_type() != Layer::Unscaling
        && layers_pointers[i]->get_type() != Layer::Bounding)
        {
            trainable_layers_indices[trainable_layer_index] = i;
            trainable_layer_index++;
        }
    }

    return trainable_layers_indices;
}


/// Returns a pointer to the scaling layers object composing this neural network object.

ScalingLayer* NeuralNetwork::get_scaling_layer_pointer() const
{
    const Index layers_number = get_layers_number();

    for(Index i = 0; i < layers_number; i++)
    {
        if(layers_pointers[i]->get_type() == Layer::Scaling)
        {
            return dynamic_cast<ScalingLayer*>(layers_pointers[i]);
        }
    }

    ostringstream buffer;

    buffer << "OpenNN Exception: NeuralNetwork class.\n"
           << "ScalingLayer* get_scaling_layer_pointer() const method.\n"
           << "No scaling layer in neural network.\n";

    throw logic_error(buffer.str());
}


/// Returns a pointer to the unscaling layers object composing this neural network object.

UnscalingLayer* NeuralNetwork::get_unscaling_layer_pointer() const
{
    const Index layers_number = get_layers_number();

    for(Index i = 0; i < layers_number; i++)
    {
        if(layers_pointers[i]->get_type() == Layer::Unscaling)
        {
            return dynamic_cast<UnscalingLayer*>(layers_pointers[i]);
        }
    }

    ostringstream buffer;

    buffer << "OpenNN Exception: NeuralNetwork class.\n"
           << "UnscalingLayer* get_unscaling_layer_pointer() const method.\n"
           << "No unscaling layer in neural network.\n";

    throw logic_error(buffer.str());
}


/// Returns a pointer to the bounding layers object composing this neural network object.

BoundingLayer* NeuralNetwork::get_bounding_layer_pointer() const
{
    const Index layers_number = get_layers_number();

    for(Index i = 0; i < layers_number; i++)
    {
        if(layers_pointers[i]->get_type() == Layer::Bounding)
        {
            return dynamic_cast<BoundingLayer*>(layers_pointers[i]);
        }
    }

    ostringstream buffer;

    buffer << "OpenNN Exception: NeuralNetwork class.\n"
           << "BoundingLayer* get_bounding_layer_pointer() const method.\n"
           << "No bounding layer in neural network.\n";

    throw logic_error(buffer.str());
}


/// Returns a pointer to the probabilistic layers object composing this neural network object.

ProbabilisticLayer* NeuralNetwork::get_probabilistic_layer_pointer() const
{
    const Index layers_number = get_layers_number();

    for(Index i = 0; i < layers_number; i++)
    {
        if(layers_pointers[i]->get_type() == Layer::Probabilistic)
        {
            return dynamic_cast<ProbabilisticLayer*>(layers_pointers[i]);
        }
    }

    ostringstream buffer;

    buffer << "OpenNN Exception: NeuralNetwork class.\n"
           << "ProbabilisticLayer* get_probabilistic_layer_pointer() const method.\n"
           << "No probabilistic layer in neural network.\n";

    throw logic_error(buffer.str());
}


/// Returns a pointer to the main components of the layers object composing this neural network object.

PrincipalComponentsLayer* NeuralNetwork::get_principal_components_layer_pointer() const
{
    const Index layers_number = get_layers_number();

    for(Index i = 0; i < layers_number; i++)
    {
        if(layers_pointers[i]->get_type() == Layer::PrincipalComponents)
        {
            return dynamic_cast<PrincipalComponentsLayer*>(layers_pointers[i]);
        }
    }

    ostringstream buffer;

    buffer << "OpenNN Exception: NeuralNetwork class.\n"
           << "PrincipalComponentsLayer* get_principal_components_layer_pointer() const method.\n"
           << "No principal components layer in neural network.\n";

    throw logic_error(buffer.str());
}


/// Returns a pointer to the long short term memory layer of this neural network, if exits.

LongShortTermMemoryLayer* NeuralNetwork::get_long_short_term_memory_layer_pointer() const
{
    const Index layers_number = get_layers_number();

    for(Index i = 0; i < layers_number; i++)
    {
        if(layers_pointers[i]->get_type() == Layer::LongShortTermMemory)
        {
            return dynamic_cast<LongShortTermMemoryLayer*>(layers_pointers[i]);
        }
    }

    ostringstream buffer;

    buffer << "OpenNN Exception: NeuralNetwork class.\n"
           << "LongShortTermMemoryLayer* get_long_short_term_memory_layer_pointer() const method.\n"
           << "No long-short term memory layer in neural network.\n";

    throw logic_error(buffer.str());
}


/// Returns a pointer to the recurrent layer of this neural network, if exits.

RecurrentLayer* NeuralNetwork::get_recurrent_layer_pointer() const
{
    const Index layers_number = get_layers_number();

    for(Index i = 0; i < layers_number; i++)
    {
        if(layers_pointers[i]->get_type() == Layer::Recurrent)
        {
            return dynamic_cast<RecurrentLayer*>(layers_pointers[i]);
        }
    }

    ostringstream buffer;

    buffer << "OpenNN Exception: NeuralNetwork class.\n"
           << "RecurrentLayer* get_recurrent_layer_pointer() const method.\n"
           << "No recurrent layer in neural network.\n";

    throw logic_error(buffer.str());
}


/// Returns true if messages from this class are to be displayed on the screen, or false if messages
/// from this class are not to be displayed on the screen.

const bool& NeuralNetwork::get_display() const
{
    return display;
}


/// This method deletes all the pointers in the neural network.
/// It also sets the rest of members to their default values.

void NeuralNetwork::set()
{
    inputs_names.resize(0);

    outputs_names.resize(0);

    layers_pointers.resize(0);

    set_default();
}


/// Sets a new neural network with a given neural network architecture.
/// It also sets the rest of members to their default values.
/// @param architecture Architecture of the neural network.

void NeuralNetwork::set(const NeuralNetwork::ProjectType& model_type, const Tensor<Index, 1>& architecture)
{
    layers_pointers.resize(0);

    if(architecture.size() <= 1) return;

    const Index size = architecture.size();

    const Index inputs_number = architecture[0];
    const Index outputs_number = architecture[size-1];

    inputs_names.resize(inputs_number);

    ScalingLayer* scaling_layer_pointer = new ScalingLayer(inputs_number);

    this->add_layer(scaling_layer_pointer);

    if(model_type == Approximation)
    {
        for(Index i = 0; i < size-1; i++)
        {
            PerceptronLayer* perceptron_layer_pointer = new PerceptronLayer(architecture[i], architecture[i+1], i);

            this->add_layer(perceptron_layer_pointer);

            if(i == size-2) perceptron_layer_pointer->set_activation_function(PerceptronLayer::Linear);
        }

        UnscalingLayer* unscaling_layer_pointer = new UnscalingLayer(outputs_number);

        this->add_layer(unscaling_layer_pointer);

        BoundingLayer* bounding_layer_pointer = new BoundingLayer(outputs_number);

        this->add_layer(bounding_layer_pointer);
    }
    else if(model_type == Classification)
    {
        for(Index i = 0; i < size-2; i++)
        {
            PerceptronLayer* perceptron_layer_pointer = new PerceptronLayer(architecture[i], architecture[i+1], i);

            this->add_layer(perceptron_layer_pointer);
        }

        ProbabilisticLayer* probabilistic_layer_pointer = new ProbabilisticLayer(architecture[size-2], architecture[size-1]);

        this->add_layer(probabilistic_layer_pointer);
    }
    else if(model_type == Forecasting)
    {
//        LongShortTermMemoryLayer* long_short_term_memory_layer_pointer = new LongShortTermMemoryLayer(architecture[0], architecture[1]);

//        this->add_layer(long_short_term_memory_layer_pointer);


        RecurrentLayer* recurrent_layer_pointer = new RecurrentLayer(architecture[0], architecture[1]);

        this->add_layer(recurrent_layer_pointer);

        for(Index i = 1; i < size-1; i++)
        {
            PerceptronLayer* perceptron_layer_pointer = new PerceptronLayer(architecture[i], architecture[i+1], i);

            this->add_layer(perceptron_layer_pointer);
        }

        UnscalingLayer* unscaling_layer_pointer = new UnscalingLayer(architecture[size-1]);

        this->add_layer(unscaling_layer_pointer);
    }

    outputs_names.resize(outputs_number);

    set_default();

}


/// Sets a new neural network with a given convolutional neural network architecture (CNN).
/// It also sets the rest of members to their default values.
/// @param input_variables_dimensions Define the dimensions of the input varibales.
/// @param blocks_number Number of blocks.
/// @param filters_dimensions Architecture of the neural network.
/// @param outputs_number Architecture of the neural network.
/// @todo

void NeuralNetwork::set(const Tensor<Index, 1>& input_variables_dimensions,
                        const Index& blocks_number,
                        const Tensor<Index, 1>& filters_dimensions,
                        const Index& outputs_number)
{
    layers_pointers.resize(0);

    ScalingLayer* scaling_layer = new ScalingLayer(input_variables_dimensions);
    this->add_layer(scaling_layer);

    Tensor<Index, 1> outputs_dimensions = scaling_layer->get_outputs_dimensions();

    for(Index i = 0; i < blocks_number; i++)
    {
        ConvolutionalLayer* convolutional_layer = new ConvolutionalLayer(outputs_dimensions, filters_dimensions);
        add_layer(convolutional_layer);

        outputs_dimensions = convolutional_layer->get_outputs_dimensions();

        // Pooling layer 1

        PoolingLayer* pooling_layer_1 = new PoolingLayer(outputs_dimensions);
        add_layer(pooling_layer_1);

        outputs_dimensions = pooling_layer_1->get_outputs_dimensions();
    }

    //    PerceptronLayer* perceptron_layer = new PerceptronLayer(outputs_dimensions.sum(), 18);
    const Tensor<Index, 0> outputs_dimensions_sum = outputs_dimensions.sum();

    PerceptronLayer* perceptron_layer = new PerceptronLayer(outputs_dimensions_sum(0), 18);
    add_layer(perceptron_layer);

    const Index perceptron_layer_outputs = perceptron_layer->get_neurons_number();

    ProbabilisticLayer* probabilistic_layer = new ProbabilisticLayer(perceptron_layer_outputs, outputs_number);
    add_layer(probabilistic_layer);
}


/// Sets the neural network members by loading them from a XML file.
/// @param file_name Neural network XML file_name.

void NeuralNetwork::set(const string& file_name)
{
    layers_pointers.resize(0);

    load(file_name);
}


/// Sets the members of this neural network object with those from other neural network object.
/// @param other_neural_network Neural network object to be copied.

void NeuralNetwork::set(const NeuralNetwork& other_neural_network)
{
    layers_pointers.resize(0);

    if(this == &other_neural_network) return;

    inputs_names = other_neural_network.inputs_names;

    outputs_names = other_neural_network.outputs_names;

    layers_pointers = other_neural_network.layers_pointers;

    display = other_neural_network.display;

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
#ifdef __OPENNN_DEBUG__

    if(new_inputs_number == 0)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: NeuralNetwork class.\n"
               << "void set_inputs_number(const Index&) method.\n"
               << "The number of inputs(" << new_inputs_number << ") must be greater than 0.\n";

        throw logic_error(buffer.str());
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
        if(inputs(i) == true) new_inputs_number++;
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
        if(layers_pointers(i)->get_type() == Layer::Perceptron)
        {
            return dynamic_cast<PerceptronLayer*>(layers_pointers[i]);
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
        Layer* last_layer = layers_pointers[layers_pointers.size()-1];

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
        if(layers_pointers(i)->get_type() != Layer::Scaling
                && layers_pointers(i)->get_type() != Layer::Unscaling
                && layers_pointers(i)->get_type() != Layer::Bounding)
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
        if(layers_pointers(i)->get_type() != Layer::Scaling
                && layers_pointers(i)->get_type() != Layer::Unscaling
                && layers_pointers(i)->get_type() != Layer::Bounding)
        {
            layers_neurons_number(count) = layers_pointers[i]->get_inputs_number();

            count++;
        }

    }

    return layers_neurons_number;
}


Tensor<Index, 1> NeuralNetwork::get_trainable_layers_synaptic_weight_numbers() const
{
    const Index trainable_layers_number = get_trainable_layers_number();

    Tensor<Index, 1> layers_neurons_number(trainable_layers_number);

    Index count = 0;

    for(Index i = 0; i < layers_pointers.size(); i++)
    {
        if(layers_pointers(i)->get_type() != Layer::Scaling
                && layers_pointers(i)->get_type() != Layer::Unscaling
                && layers_pointers(i)->get_type() != Layer::Bounding)
        {
            layers_neurons_number(count) = layers_pointers[i]->get_synaptic_weights_number();

            count++;
        }

    }

    return layers_neurons_number;
}

/// Returns a vector with the architecture of the neural network.
/// The elements of this vector are as follows;
/// <UL>
/// <LI> Number of scaling neurons(if there is a scaling layer).</LI>
/// <LI> Number of principal components neurons(if there is a principal components layer).</LI>
/// <LI> Multilayer perceptron architecture(if there is a neural network).</LI>
/// <LI> Number of conditions neurons(if there is a conditions layer).</LI>
/// <LI> Number of unscaling neurons(if there is an unscaling layer).</LI>
/// <LI> Number of probabilistic neurons(if there is a probabilistic layer).</LI>
/// <LI> Number of bounding neurons(if there is a bounding layer).</LI>
/// </UL>

Tensor<Index, 1> NeuralNetwork::get_architecture() const
{
    const Index layers_number = get_layers_number();

    Tensor<Index, 1> architecture(layers_number);

    const Index inputs_number = get_inputs_number();

    if(inputs_number == 0)
    {
        return architecture;
    }

    //    architecture.push_back(inputs_number);

    //    architecture(0) = inputs_number;

    if(layers_number > 0)
    {
        for(Index i = 0; i < layers_number; i++)
        {
            architecture(i) = layers_pointers(i)->get_neurons_number(); //.push_back(layers_pointers[i]->get_neurons_number());
        }
    }

    return architecture;
}


/// Returns the number of parameters in the neural network
/// The number of parameters is the sum of all the neural network parameters(biases and synaptic weights) and independent parameters.

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


/// Returns the number of parameters in the neural network
/// The number of parameters is the sum of all the neural network trainable parameters(biases and synaptic weights) and independent parameters.

Index NeuralNetwork::get_trainable_parameters_number() const
{
    const Index trainable_layers_number = get_trainable_layers_number();

    const Tensor<Layer*, 1> trainable_layers_pointers = get_trainable_layers_pointers();

    Index trainable_parameters_number = 0;

    for(Index i = 0; i < trainable_layers_number; i++)
    {
        trainable_parameters_number += trainable_layers_pointers[i]->get_parameters_number();
    }

    return trainable_parameters_number;
}


/// Returns the values of the parameters in the neural network as a single vector.
/// This contains all the neural network parameters(biases and synaptic weights) and preprocessed independent parameters.

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


/// Sets all the parameters(neural_network_pointer parameters and independent parameters) from a single vector.
/// @param new_parameters New set of parameter values.

void NeuralNetwork::set_parameters(Tensor<type, 1>& new_parameters)
{
#ifdef __OPENNN_DEBUG__

    const Index size = new_parameters.size();

    const Index parameters_number = get_parameters_number();

    if(size < parameters_number)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: NeuralNetwork class.\n"
               << "void set_parameters(const Tensor<type, 1>&) method.\n"
               << "Size (" << size << ") must be grater or equal to number of parameters (" << parameters_number << ").\n";

        throw logic_error(buffer.str());
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
/// If it is set to true messages from this class are to be displayed on the screen;
/// if it is set to false messages from this class are not to be displayed on the screen.
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
        if(layers_pointers(i)->get_type() != Layer::Scaling
                && layers_pointers(i)->get_type() != Layer::Unscaling
                && layers_pointers(i)->get_type() != Layer::Bounding)
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
        if(layers_pointers(i)->get_type() == Layer::Perceptron)
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
        if(layers_pointers(i)->get_type() == Layer::Probabilistic)
        {
            count++;
        }
    }

    return count;
}



/// Initializes all the neural and the independent parameters with a given value.

void NeuralNetwork::set_parameters_constant(const type& value)
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

void NeuralNetwork::set_parameters_random()
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
#ifdef __OPENNN_DEBUG__

    if(perturbation < 0)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: NeuralNetwork class.\n"
               << "void perturbate_parameters(const type&) method.\n"
               << "Perturbation must be equal or greater than 0.\n";

        throw logic_error(buffer.str());
    }

#endif

    Tensor<type, 1> parameters = get_parameters();

    parameters = parameters + perturbation;

    set_parameters(parameters);
}


/// Calculates the forward propagation in the neural network.
/// @param batch Batch of data set that contains the inputs and targets to be trained.
/// @param foward_propagation Is a NeuralNetwork class structure where save the neccesary paraneters of forward propagation.

void NeuralNetwork::forward_propagate(const DataSet::Batch& batch,
                                      ForwardPropagation& forward_propagation) const
{
    const Tensor<Layer*, 1> trainable_layers_pointers = get_trainable_layers_pointers();

    const Index trainable_layers_number = trainable_layers_pointers.size();

    if(trainable_layers_pointers(0)->get_type() == Layer::Convolutional)
    {

        trainable_layers_pointers(0)->forward_propagate(batch.inputs_4d, forward_propagation.layers(0));
    }
    else
    {
        trainable_layers_pointers(0)->forward_propagate(batch.inputs_2d, forward_propagation.layers(0));
    }

    for(Index i = 1; i < trainable_layers_number; i++)
    {
         trainable_layers_pointers(i)->forward_propagate(forward_propagation.layers(i-1).activations_2d,
                                                                     forward_propagation.layers(i));
    }
}


/// Calculates the forward propagation in the neural network.
/// @param batch Batch of data set that contains the inputs and targets to be trained.
/// @param paramters Parameters of neural network.
/// @param foward_propagation Is a NeuralNetwork class structure where save the neccesary paraneters of forward propagation.

void NeuralNetwork::forward_propagate(const DataSet::Batch& batch,
                                      Tensor<type, 1>& parameters,
                                      ForwardPropagation& forward_propagation) const
{
    const Tensor<Layer*, 1> trainable_layers_pointers = get_trainable_layers_pointers();

    const Index trainable_layers_number = trainable_layers_pointers.size();

    const Index parameters_number = trainable_layers_pointers(0)->get_parameters_number();

    const TensorMap<Tensor<type, 1>> potential_parameters(parameters.data(), parameters_number);

    if(trainable_layers_pointers(0)->get_type() == Layer::Convolutional)
    {
        trainable_layers_pointers(0)->forward_propagate(batch.inputs_4d, potential_parameters, forward_propagation.layers(0));
    }
    else
    {
        trainable_layers_pointers(0)->forward_propagate(batch.inputs_2d, potential_parameters, forward_propagation.layers(0));
    }

    Index index = parameters_number;

    for(Index i = 1; i < trainable_layers_number; i++)
    {
        const Index parameters_number = trainable_layers_pointers(i)->get_parameters_number();

        const TensorMap<Tensor<type, 1>> potential_parameters(parameters.data() + index, parameters_number);

        trainable_layers_pointers(i)->forward_propagate(forward_propagation.layers(i-1).activations_2d,
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
/// <li> Apply boundary condtions.
/// <li> Calculate bounded outputs.
/// </ul>
/// @param inputs Set of inputs to the neural network.

Tensor<type, 2> NeuralNetwork::calculate_outputs(const Tensor<type, 2>& inputs)
{
#ifdef __OPENNN_DEBUG__

    const Index inputs_dimensions_number = inputs.rank();

    if(inputs_dimensions_number != 2 && inputs_dimensions_number != 4)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: NeuralNetwork class.\n"
               << "Tensor<type, 2> calculate_outputs(const Tensor<type, 2>&) const method.\n"
               << "Inputs dimensions number (" << inputs_dimensions_number << ") must be 2 or 4.\n";

        throw logic_error(buffer.str());
    }

#endif

    Tensor<type, 2> outputs;

    const Index layers_number = get_layers_number();

    if(layers_number == 0) return inputs;

    outputs = layers_pointers(0)->calculate_outputs(inputs);

    for(Index i = 1; i < layers_number; i++)
    {
        outputs = layers_pointers(i)->calculate_outputs(outputs);
    }

    return outputs;
}




Tensor<type, 2> NeuralNetwork::calculate_outputs(const Tensor<type, 4>& inputs)
{
#ifdef __OPENNN_DEBUG__

    const Index inputs_dimensions_number = inputs.rank();

    if(inputs_dimensions_number != 2 && inputs_dimensions_number != 4)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: NeuralNetwork class.\n"
               << "Tensor<type, 2> calculate_outputs(const Tensor<type, 2>&) const method.\n"
               << "Inputs dimensions number (" << inputs_dimensions_number << ") must be 2 or 4.\n";

        throw logic_error(buffer.str());
    }

#endif

    Tensor<type, 4> outputs_4d;
    Tensor<type, 2> outputs, inputs_2d;

    const Index layers_number = get_layers_number();

    if(layers_number == 0) return inputs_2d;

    // First layer output
    if(layers_pointers(1)->get_type() == Layer::Convolutional)
    {
        outputs_4d = layers_pointers(0)->calculate_outputs_4D(inputs);
    }
    else
    {
        outputs = layers_pointers(0)->calculate_outputs_from4D(inputs);
    }

    for(Index i = 1; i < layers_number; i++)
    {
        if(layers_pointers(i + 1)->get_type() == Layer::Convolutional)
        {
            outputs_4d = layers_pointers(i)->calculate_outputs_4D(outputs_4d);
        }
        else
        {
            if(layers_pointers(i)->get_type() != Layer::Convolutional && layers_pointers(i)->get_type() != Layer::Pooling)
            {
                outputs = layers_pointers(i)->calculate_outputs(outputs);
            }
            else
            {
                outputs = layers_pointers(i)->calculate_outputs_from4D(outputs_4d);
            }
        }
    }

    return outputs;
}


/// Calculates the input data which is necessary to compute the output data from the neural network in some direction.
/// @param direction Input index(must be between 0 and number of inputs - 1).
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
        inputs(direction) = minimum + (maximum-minimum)*static_cast<type>(i)/static_cast<type>(points_number-1);

        for(Index j = 0; j < inputs_number; j++)
        {
            directional_inputs(i,j) = inputs(j);
        }
    }

    return directional_inputs;
}


/// For each layer: inputs, neurons, activation function

Tensor<string, 2> NeuralNetwork::get_information() const
{
    const Index trainable_layers_number = get_trainable_layers_number();

    Tensor<string, 2> information(trainable_layers_number, 3);

    Tensor<Layer*, 1> trainable_layers_pointers = get_trainable_layers_pointers();

    for(Index i = 0; i < trainable_layers_number; i++)
    {
        information(i,0) = std::to_string(trainable_layers_pointers(i)->get_inputs_number());
        information(i,1) = std::to_string(trainable_layers_pointers(i)->get_neurons_number());
//        information(i,2) = trainable_layers_pointers(i)->get_type_string();

        const string layer_type = trainable_layers_pointers(i)->get_type_string();

        if(layer_type == "Perceptron")
        {
            PerceptronLayer* perceptron_layer = static_cast<PerceptronLayer*>(trainable_layers_pointers(i));

            information(i,2) = perceptron_layer->write_activation_function();
        }
        else
        {
            //@todo rest of the layers
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
            information(perceptron_layer_index,0) = std::to_string(trainable_layers_pointers(i)->get_inputs_number());
            information(perceptron_layer_index,1) = std::to_string(trainable_layers_pointers(i)->get_neurons_number());

            PerceptronLayer* perceptron_layer = static_cast<PerceptronLayer*>(trainable_layers_pointers(i));

            information(perceptron_layer_index,2) = perceptron_layer->write_activation_function();

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
            information(probabilistic_layer_index,0) = std::to_string(trainable_layers_pointers(i)->get_inputs_number());
            information(probabilistic_layer_index,1) = std::to_string(trainable_layers_pointers(i)->get_neurons_number());

            ProbabilisticLayer* probabilistic_layer = static_cast<ProbabilisticLayer*>(trainable_layers_pointers(i));

            information(probabilistic_layer_index,2) = probabilistic_layer->write_activation_function();

            probabilistic_layer_index++;
        }
    }

    return information;
}


/// Serializes the neural network object into a XML document of the TinyXML library without keep the DOM tree in memory.
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
    buffer << get_inputs_number();

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

        throw logic_error(buffer.str());
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
            catch(const logic_error& e)
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

        throw logic_error(buffer.str());
    }

    // Inputs number

    const tinyxml2::XMLElement* inputs_number_element = root_element->FirstChildElement("InputsNumber");

    if(!inputs_number_element)
    {
        buffer << "OpenNN Exception: NeuralNetwork class.\n"
               << "void inputs_from_XML(const tinyxml2::XMLDocument&) method.\n"
               << "Inputs number element is nullptr.\n";

        throw logic_error(buffer.str());
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

            if(input_element->Attribute("Index") != std::to_string(i+1))
            {
                buffer << "OpenNN Exception: NeuralNetwork class.\n"
                       << "void inputs_from_XML(const tinyxml2::XMLDocument&) method.\n"
                       << "Input index number (" << i+1 << ") does not match (" << input_element->Attribute("Item") << ").\n";

                throw logic_error(buffer.str());
            }

//            inputs_names(i) = input_element->GetText();
            if(!input_element->GetText()){
                inputs_names(i) = "";
            }
            else{
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

        throw logic_error(buffer.str());
    }

    // Layers types

    const tinyxml2::XMLElement* layers_types_element = root_element->FirstChildElement("LayersTypes");

    if(!layers_types_element)
    {
        buffer << "OpenNN Exception: NeuralNetwork class.\n"
               << "void layers_from_XML(const tinyxml2::XMLDocument&) method.\n"
               << "Layers types element is nullptr.\n";

        throw logic_error(buffer.str());
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
        else if(layers_types(i) == "PrincipalComponents")
        {
            PrincipalComponentsLayer* principal_components_layer = new PrincipalComponentsLayer();

            const tinyxml2::XMLElement* principal_components_element = start_element->NextSiblingElement("PrincipalComponentsLayer");
            start_element = principal_components_element;

            if(principal_components_element)
            {
                tinyxml2::XMLDocument principal_components_document;
                tinyxml2::XMLNode* element_clone;

                element_clone = principal_components_element->DeepClone(&principal_components_document);

                principal_components_document.InsertFirstChild(element_clone);

                principal_components_layer->from_XML(principal_components_document);
            }

            add_layer(principal_components_layer);
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

        throw logic_error(buffer.str());
    }

    // Outputs number

    const tinyxml2::XMLElement* outputs_number_element = root_element->FirstChildElement("OutputsNumber");

    if(!outputs_number_element)
    {
        buffer << "OpenNN Exception: NeuralNetwork class.\n"
               << "void inputs_from_XML(const tinyxml2::XMLDocument&) method.\n"
               << "Outputs number element is nullptr.\n";

        throw logic_error(buffer.str());
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

            if(output_element->Attribute("Index") != std::to_string(i+1))
            {
                buffer << "OpenNN Exception: NeuralNetwork class.\n"
                       << "void outputs_from_XML(const tinyxml2::XMLDocument&) method.\n"
                       << "Output index number (" << i+1 << ") does not match (" << output_element->Attribute("Item") << ").\n";

                throw logic_error(buffer.str());
            }

            if(!output_element->GetText()){
                outputs_names(i) = "";
            }else{
                outputs_names(i) = output_element->GetText();
            }

        }
    }
}


/// Prints to the screen the members of a neural network object in a XML-type format.
/// @todo

void NeuralNetwork::print() const
{
}


void NeuralNetwork::print_summary() const
{
    const Index layers_number = get_layers_number();

    cout << "Layers number: " << layers_number << endl;

    for(Index i = 0; i < layers_number; i++)
    {
        cout << "Layer " << i+1 << ": " << layers_pointers[i]->get_neurons_number()
             << " " << layers_pointers[i]->get_type_string() << " neurons" << endl;

    }
}


/// Saves to a XML file the members of a neural network object.
/// @param file_name Name of neural network XML file.

void NeuralNetwork::save(const string& file_name) const
{

    FILE *pFile;
//    errno_t err;

//    err = fopen_s(&pFile, file_name.c_str(), "w");
    pFile = fopen(file_name.c_str(), "w");

    tinyxml2::XMLPrinter document(pFile);

    write_XML(document);

    fclose(pFile);
}


/// Saves to a data file the parameters of a neural network object.
/// @param file_name Name of parameters data file.

void NeuralNetwork::save_parameters(const string& file_name) const
{
    ofstream file(file_name.c_str());

    if(!file.is_open())
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: NeuralNetwork class.\n"
               << "void save_parameters(const string&) const method.\n"
               << "Cannot open parameters data file.\n";

        throw logic_error(buffer.str());
    }

    const Tensor<type, 1> parameters = get_parameters();

    file << parameters << endl;

    // Close file

    file.close();
}


/// Loads from a XML file the members for this neural network object.
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

        throw logic_error(buffer.str());

    }

    from_XML(document);
}


/// Loads the neural network parameters from a data file.
/// The format of this file is just a sequence of numbers.
/// @param file_name Name of parameters data file.
/// @todo
void NeuralNetwork::load_parameters(const string& file_name)
{
    ifstream file(file_name.c_str());

    if(!file.is_open())
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: NeuralNetwork class.\n"
               << "void load_parameters(const string&) method.\n"
               << "Cannot open parameters data file.\n";

        throw logic_error(buffer.str());
    }

    const Index parameters_number = get_parameters_number();

    Tensor<type, 1> new_parameters(parameters_number);
    /*
        new_parameters.load(file_name);
    */
    set_parameters(new_parameters);

    file.close();
}


/// Loads the neural network parameters from a data file.
/// The format of this file is just a sequence of numbers.
/// @param file_name Name of parameters data file.

void NeuralNetwork::load_parameters_binary(const string& file_name)
{
    ifstream file;

    file.open(file_name.c_str(), ios::binary);

    if(!file.is_open())
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: NeuralNetwork template.\n"
               << "void load_parameters_binary(const string&) method.\n"
               << "Cannot open binary file: " << file_name << "\n";

        throw logic_error(buffer.str());
    }

    streamsize size = sizeof(double);

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
    const Index layers_number = get_layers_number();

    Tensor<Layer*, 1> layers_pointers = get_layers_pointers();
    Tensor<string, 1> layers_names = get_layers_names();

    ostringstream buffer;

    buffer <<"/*"<<endl;
    buffer <<"Artificial Intelligence Techniques SL\t"<<endl;
    buffer <<"artelnics@artelnics.com\t"<<endl;
    buffer <<""<<endl;
    buffer <<"Your model has been exported to this file." <<endl;
    buffer <<"You can manage it with the 'neural network' method.\t"<<endl;
    buffer <<"Example:"<<endl;
    buffer <<""<<endl;
    buffer <<"\tvector<float> sample(n);\t"<<endl;
    buffer <<"\tsample[0] = 1;\t"<<endl;
    buffer <<"\tsample[1] = 2;\t"<<endl;
    buffer <<"\tsample[n] = 10;\t"<<endl;
    buffer <<"\tvector<float> outputs = neural_network(sample);"<<endl;
    buffer <<""<<endl;
    buffer <<"Notice that only one sample is allowed as input. Batch of inputs are not yet implement,\t"<<endl;
    buffer <<"however you can loop through neural network function in order to get multiple outputs.\t"<<endl;
    buffer <<"*/"<<endl;
    buffer <<""<<endl;


    buffer << "#include <vector>\n" << endl;

    buffer << "using namespace std;\n" << endl;

    for(Index i = 0; i < layers_number; i++)
    {

        buffer << layers_pointers[i]->write_expression_c() << endl;
    }

    buffer << "vector<float> neural_network(const vector<float>& inputs)\n{" << endl;

    buffer << "\tvector<float> outputs;\n" << endl;

    if(layers_number > 0)
    {
        buffer << "\toutputs = " << layers_names[0] << "(inputs);\n";
    }

    for(Index i = 1; i < layers_number; i++)
    {
        buffer << "\toutputs = " << layers_names[i] << "(outputs);\n";
    }

    buffer << "\n\treturn outputs;\n}" << endl;

    buffer << "int main(){return 0;}" << endl;


    string expression = buffer.str();

    replace(expression, "+-", "-");
    replace(expression, "-+", "-");
    replace(expression, "--", "+");

    return expression;
}


string NeuralNetwork::write_expression(const Tensor<string, 1>& inputs_names, const Tensor<string, 1>& outputs_names) const
{
    const Index layers_number = get_layers_number();

    Tensor<Layer*, 1> layers_pointers = get_layers_pointers();
    Tensor<string, 1> layers_names = get_layers_names();

    Tensor<string, 1> temporal_outputs_names;
    Tensor<string, 1> temporal_inputs_names;
    temporal_inputs_names = inputs_names;

    Index layer_neurons_number;

    ostringstream buffer;

    for(Index i = 0; i < layers_number; i++)
    {
        if(i == layers_number-1)
        {
            temporal_outputs_names = outputs_names;
            buffer << layers_pointers[i]->write_expression(temporal_inputs_names, temporal_outputs_names) << endl;
        }
        else{
            layer_neurons_number = layers_pointers[i]->get_neurons_number();
            temporal_outputs_names.resize(layer_neurons_number);


            for(Index j = 0; j < layer_neurons_number; j++)
            {
                if(layers_names(i) == "scaling_layer")
                {
                    temporal_outputs_names(j) = "scaled_" + inputs_names(j);
                }
                else{
                    temporal_outputs_names(j) =  layers_names(i) + "_output_" + to_string(j);
                }
            }
            buffer << layers_pointers[i]->write_expression(temporal_inputs_names, temporal_outputs_names) << endl;

            temporal_inputs_names = temporal_outputs_names;
        }
    }

    string expression = buffer.str();

    replace(expression, "+-", "-");

    return expression;
}


/// Returns a string with the python function of the expression represented by the neural network.

string NeuralNetwork::write_expression_python() const
{
    const Index layers_number = get_layers_number();

    Tensor<Layer*, 1> layers_pointers = get_layers_pointers();
    Tensor<string, 1> layers_names = get_layers_names();

    ostringstream buffer;

    buffer <<"'''"<<endl;
    buffer <<"Artificial Intelligence Techniques SL\t"<<endl;
    buffer <<"artelnics@artelnics.com\t"<<endl;
    buffer <<""<<endl;
    buffer <<"Your model has been exported to this python file." <<endl;
    buffer <<"You can manage it with the 'neural network' method.\t"<<endl;
    buffer <<"Example:"<<endl;
    buffer <<""<<endl;
    buffer <<"\tsample = [input_1, input_2, input_3, input_4, ...] 	 \t"<<endl;
    buffer <<"\toutputs = neural_network(sample)"<<endl;
    buffer <<""<<endl;
    buffer <<"\tInputs Names: \t"<<endl;
    buffer <<"\t" << get_inputs_names() << endl;
    buffer <<""<<endl;
    buffer <<"Notice that only one sample is allowed as input. Batch of inputs are not yet implement,\t"<<endl;
    buffer <<"however you can loop through neural network function in order to get multiple outputs.\t"<<endl;
    buffer <<"'''"<<endl;
    buffer <<""<<endl;
    buffer << "import numpy as np\n" << endl;

    for(Index i = 0; i  < layers_number; i++)
    {
        buffer << layers_pointers[i]->write_expression_python() << endl;
    }

    buffer << "def neural_network(inputs):\n" << endl;

    buffer << "\toutputs = [None] * len(inputs)\n" << endl;

    if(layers_number > 0)
    {
        buffer << "\toutputs = " << layers_names[0] << "(inputs)\n";
    }

    for(Index i = 1; i < layers_number; i++)
    {
        buffer << "\toutputs = " << layers_names[i] << "(outputs)\n";
    }

    buffer << "\n\treturn outputs;\n" << endl;

    string expression = buffer.str();

    replace(expression, "+-", "-");
    replace(expression, "-+", "-");
    replace(expression, "--", "+");

    return expression;
}


/// Saves the mathematical expression represented by the neural network to a text file.
/// @param file_name Name of expression text file.

void NeuralNetwork::save_expression_c(const string& file_name)
{
    ofstream file(file_name.c_str());

    if(!file.is_open())
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: NeuralNetwork class.\n"
               << "void  save_expression(const string&) method.\n"
               << "Cannot open expression text file.\n";

        throw logic_error(buffer.str());
    }

    file << write_expression_c();

    file.close();
}


/// Saves the python function of the expression represented by the neural network to a text file.
/// @param file_name Name of expression text file.

void NeuralNetwork::save_expression_python(const string& file_name)
{
    ofstream file(file_name.c_str());

    if(!file.is_open())
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: NeuralNetwork class.\n"
               << "void  save_expression_python(const string&) method.\n"
               << "Cannot open expression text file.\n";

        throw logic_error(buffer.str());
    }

    file << write_expression_python();

    file.close();
}


/// Saves a csv file containing the outputs for a set of given inputs.
/// @param inputs Inputs to calculate the outputs.
/// @param file_name Name of data file

void NeuralNetwork::save_outputs(const Tensor<type, 2> & inputs, const string & file_name)
{
    const Tensor<type, 2> outputs = calculate_outputs(inputs);

    ofstream file(file_name.c_str());

    if(!file.is_open())
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: NeuralNetwork class.\n"
               << "void  save_expression_python(const string&) method.\n"
               << "Cannot open expression text file.\n";

        throw logic_error(buffer.str());
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


/// Saves a set of input-output values from the neural network to a data file.
/// @param file_name Name of data file.
/// @todo

void NeuralNetwork::save_data(const string& file_name) const
{
    const Index inputs_number = get_inputs_number();

#ifdef __OPENNN_DEBUG__

    ostringstream buffer;

    //    if(!neural_network_pointer)
    //    {
    //        buffer << "OpenNN Exception: NeuralNetwork class.\n"
    //               << "void save_data(const string&) const method.\n"
    //               << "Pointer to neural network is nullptr.\n";

    //        throw logic_error(buffer.str());
    //    }

    if(inputs_number != 1)
    {
        buffer << "OpenNN Exception: NeuralNetwork class.\n"
               << "void save_data(const string&) const method.\n"
               << "Number of inputs is not 1.\n";

        throw logic_error(buffer.str());
    }

    //    if(!scaling_layer_pointer)
    //    {
    //        buffer << "OpenNN Exception: NeuralNetwork class.\n"
    //               << "void save_data(const string&) const method.\n"
    //               << "Pointer to scaling layer is nullptr.\n";

    //        throw logic_error(buffer.str());
    //    }

#endif

    const Index outputs_number = get_outputs_number();

    const Index variables_number = inputs_number + outputs_number;

    //    const Tensor<Descriptives, 1> scaling_layer_descriptives = scaling_layer_pointer->get_descriptives();

    const Index points_number = 101;

    Tensor<type, 2> data(points_number, variables_number);

    Tensor<type, 1> inputs(inputs_number);
    Tensor<type, 1> outputs(outputs_number);
    Tensor<type, 1> row(variables_number);

    Tensor<type, 1> increments(inputs_number);

    for(Index i = 0; i < inputs_number; i++)
    {
        //        inputs[i] = scaling_layer_descriptives[i].minimum;
        //        increments[i] = (scaling_layer_descriptives[i].maximum - scaling_layer_descriptives[i].minimum)/static_cast<type>(points_number-1.0);
    }
    /*
        for(Index i = 0; i < points_number; i++)
        {
    //        outputs = calculate_outputs(inputs.to_column_matrix());

            row = inputs.assemble(outputs);

            data.set_row(i, row);

            inputs += increments;
        }

        data.save_csv(file_name);
    */
}


Layer* NeuralNetwork::get_output_layer_pointer() const
{
    if(layers_pointers.dimension(0) == 0)
    {
        return nullptr;
    }
    else
    {
        const Index layers_number = get_layers_number();

        return layers_pointers[layers_number-1];
    }

    return nullptr;
}


Layer* NeuralNetwork::get_layer_pointer(const Index& index) const
{
    return layers_pointers[index];
}

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2020 Artificial Intelligence Techniques, SL.
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
