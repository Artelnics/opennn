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


/// Copy constructor. 
/// It creates a copy of an existing neural network object. 
/// @param other_neural_network Neural network object to be copied.

NeuralNetwork::NeuralNetwork(const NeuralNetwork& other_neural_network)
{
    set(other_neural_network);
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

        if(first_layer_type != Layer::Scaling)
        {
            return false;
        }
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

Tensor<string, 1> NeuralNetwork::get_inputs_names() const
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
        if(inputs_names(i) == name)
        {
            return i;
            break;
        }
    }
    return 0;
}


/// Returns a string vector with the names of the variables used as outputs.

Tensor<string, 1> NeuralNetwork::get_outputs_names() const
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
        if(outputs_names(i) == name)
        {
            return i;
            break;
        }
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
            /*trainable_layers_indices.push_back(i);*/
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

    return nullptr;
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

    return nullptr;
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

    return nullptr;
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

    return nullptr;
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

    return nullptr;
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

    return nullptr;
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

    return nullptr;
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
            PerceptronLayer* perceptron_layer_pointer = new PerceptronLayer(architecture[i], architecture[i+1]);

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
            PerceptronLayer* perceptron_layer_pointer = new PerceptronLayer(architecture[i], architecture[i+1]);

            this->add_layer(perceptron_layer_pointer);
        }

        ProbabilisticLayer* probabilistic_layer_pointer = new ProbabilisticLayer(architecture[size-2], architecture[size-1]);

        this->add_layer(probabilistic_layer_pointer);
    }
    else if(model_type == Forecasting)
    {
        LongShortTermMemoryLayer* long_short_term_memory_layer_pointer = new LongShortTermMemoryLayer(architecture[0], architecture[1]);

        this->add_layer(long_short_term_memory_layer_pointer);

        for(Index i = 1; i < size-1; i++)
        {
            PerceptronLayer* perceptron_layer_pointer = new PerceptronLayer(architecture[i], architecture[i+1]);

            this->add_layer(perceptron_layer_pointer);
        }

        UnscalingLayer* unscaling_layer_pointer = new UnscalingLayer(architecture[size-1]);

        this->add_layer(unscaling_layer_pointer);
    }

    outputs_names.resize(outputs_number);

    set_default();

}


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


void NeuralNetwork::set_inputs_names(const Tensor<string, 1>& new_inputs_names)
{
    inputs_names = new_inputs_names;
}


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
/*
    const Index new_inputs_number = inputs.count_equal_to(true);

    set_inputs_number(new_inputs_number);
*/
}


/// Sets those members which are not pointer to their default values.

void NeuralNetwork::set_default()
{
    display = true;
}


void NeuralNetwork::set_device_pointer(Device* new_device_pointer)
{  
    const Index layers_number = get_layers_number();

    for(Index i = 0; i < layers_number; i++)
    {
        layers_pointers[i]->set_device_pointer(new_device_pointer);
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
        return layers_pointers[0]->get_inputs_number();
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

        const Tensor<type, 1> layer_parameters = trainable_layers_pointers[i]->get_parameters();

        for(Index i = 0; i < layer_parameters.size(); i++)
        {
            parameters(i + position) = layer_parameters(i);
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

void NeuralNetwork::set_parameters(const Tensor<type, 1>& new_parameters)
{
#ifdef __OPENNN_DEBUG__

    const Index size = new_parameters.size();

    const Index parameters_number = get_parameters_number();

    if(size != parameters_number)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: NeuralNetwork class.\n"
               << "void set_parameters(const Tensor<type, 1>&) method.\n"
               << "Size (" << size << ") must be equal to number of parameters (" << parameters_number << ").\n";

        throw logic_error(buffer.str());
    }

#endif

    const Index trainable_layers_number = get_trainable_layers_number();

    const Tensor<Layer*, 1> trainable_layers_pointers = get_trainable_layers_pointers();

//    const Tensor<Tensor<type, 1>, 1> layers_parameters = get_trainable_layers_parameters(new_parameters);

    Index index = 0;

    for(Index i = 0; i < trainable_layers_number; i++)
    {
//        if(trainable_layers_pointers[i]->get_type() == Layer::Pooling) continue;

        trainable_layers_pointers(i)->insert_parameters(index, new_parameters);

        index += 0;
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

    Tensor<Index, 1> layers_neurons_number;
    /*

    for(Index i = 0; i < layers_pointers.size(); i++)
    {
        layers_neurons_number.push_back(layers_pointers[i]->get_neurons_number());
    }
    */
    return layers_neurons_number;
}


Index NeuralNetwork::get_trainable_layers_number() const
{
    const Index layers_number = get_layers_number();

    Index count = 0;

    for(Index i = 0; i < layers_number; i++)
    {
        if(layers_pointers[i]->get_type() != Layer::Scaling
        && layers_pointers[i]->get_type() != Layer::Unscaling
        && layers_pointers[i]->get_type() != Layer::Bounding)
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
/*
    const type parameters_norm = l2_norm(parameters);

    return parameters_norm;
*/
    return 0;
}


/// Returns a descriptives structure of the parameters vector.
/// That contains the minimum, maximum, mean and standard deviation values of the parameters.

Descriptives NeuralNetwork::calculate_parameters_descriptives() const
{
    const Tensor<type, 1> parameters = get_parameters();

    return descriptives(parameters);

}


/// Returns a histogram structure of the parameters vector.
/// That will be used for looking at the distribution of the parameters.
/// @param bins_number Number of bins in the histogram(10 by default).

Histogram NeuralNetwork::calculate_parameters_histogram(const Index& bins_number) const
{
    const Tensor<type, 1> parameters = get_parameters();

    return histogram(parameters, bins_number);

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

    Tensor<type, 1> parameters_perturbation(parameters);

    parameters_perturbation.setRandom();

    parameters += parameters_perturbation;

    set_parameters(parameters);
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

//    const Index inputs_number = get_inputs_number();

//    const Index inputs_dimension = inputs.dimension(1);

//    if(inputs_size != inputs_number)
//    {
//        ostringstream buffer;

//        buffer << "OpenNN Exception: NeuralNetwork class.\n"
//               << "Tensor<type, 2> calculate_outputs(const Tensor<type, 2>&) const method.\n"
//               << "Dimension of inputs (" <<  << ") must be equal to number of inputs.\n";

//        throw logic_error(buffer.str());
//    }

#endif

    const Index layers_number = get_layers_number();

    if(layers_number == 0) return inputs;

    cout<<inputs;

    Tensor<type, 2> outputs = layers_pointers(0)->calculate_outputs(inputs);

    for(Index i = 1; i < layers_number; i++)
    {
        outputs = layers_pointers(i)->calculate_outputs(outputs);
    }

    return outputs;
}


Tensor<type, 2> NeuralNetwork::calculate_trainable_outputs(const Tensor<type, 2>& inputs) const
{
#ifdef __OPENNN_DEBUG__

    ///@todo check for convolutional

//    const Index inputs_dimensions_number = inputs.rank();

//    if(inputs_dimensions_number != 2)
//    {
//        ostringstream buffer;

//        buffer << "OpenNN Exception: NeuralNetwork class.\n"
//               << "Tensor<type, 2> calculate_trainable_outputs(const Tensor<type, 2>&) const method.\n"
//               << "Inputs dimensions number (" << inputs_dimensions_number << ") must be 2.\n";

//        throw logic_error(buffer.str());
//    }

//    const Index inputs_number = get_inputs_number();

//    const Index inputs_columns_number = inputs.dimension(1);

//    if(inputs_columns_number != inputs_number)
//    {
//        ostringstream buffer;

//        buffer << "OpenNN Exception: NeuralNetwork class.\n"
//               << "Tensor<type, 2> calculate_outputs(const Tensor<type, 2>&) const method.\n"
//               << "Number of columns (" << inputs_columns_number << ") must be equal to number of inputs (" << inputs_number << ").\n";

//        throw logic_error(buffer.str());
//    }

#endif

    const Index trainable_layers_number = get_trainable_layers_number();

    const Tensor<Layer*, 1> trainable_layers_pointers = get_trainable_layers_pointers();

    Tensor<type, 2> outputs = trainable_layers_pointers[0]->calculate_outputs(inputs);

    for(Index i = 1; i < trainable_layers_number; i++)
    {
        outputs = trainable_layers_pointers[i]->calculate_outputs(outputs);
    }

    return outputs;
}


Tensor<type, 2> NeuralNetwork::calculate_trainable_outputs(const Tensor<type, 2>& inputs,
                                                           const Tensor<type, 1>& parameters) const
{
    const Index batch_size = inputs.dimension(0);

    const Index trainable_layers_number = get_trainable_layers_number();

    #ifdef __OPENNN_DEBUG__

        if(trainable_layers_number == 0)
        {
            ostringstream buffer;

            buffer << "OpenNN Exception: NeuralNetwork class.\n"
                   << "Tensor<type, 2> calculate_outputs(const Tensor<type, 2>&, cons Tensor<type, 1>&) const method.\n"
                   << "This neural network has not got any layer.\n";

            throw logic_error(buffer.str());
        }

    #endif

    const Tensor<Layer*, 1> trainable_layers_pointers = get_trainable_layers_pointers();

    const Tensor<Tensor<type, 1>, 1> trainable_layers_parameters = get_trainable_layers_parameters(parameters);

    Tensor<type, 2> outputs(batch_size, trainable_layers_pointers[0]->get_neurons_number());

    if(trainable_layers_pointers[0]->get_type() == OpenNN::Layer::Type::Pooling)
    {
        outputs = trainable_layers_pointers[0]->calculate_outputs(inputs);
    }

    else outputs = trainable_layers_pointers[0]->calculate_outputs(inputs, trainable_layers_parameters[0]);

    for(Index i = 1; i < trainable_layers_number; i++)
    {
        outputs.resize(batch_size, trainable_layers_pointers[i]->get_neurons_number());

        if(trainable_layers_pointers[i]->get_type() == OpenNN::Layer::Type::Pooling)
        {
            outputs = trainable_layers_pointers[i]->calculate_outputs(outputs);
        }
        else outputs = trainable_layers_pointers[i]->calculate_outputs(outputs, trainable_layers_parameters[i]);
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
/*
    for(Index i = 0; i < points_number; i++)
    {
        inputs[direction] = minimum + (maximum-minimum)*i/static_cast<type>(points_number-1);

        directional_inputs.set_row(i, inputs);
    }
*/
    return directional_inputs;
}


/// Calculates the histogram of the outputs with random inputs.
/// @param points_number Number of random instances to evaluate the neural network.
/// @param bins_number Number of bins for the histograms.
/// @todo

Tensor<Histogram, 1> NeuralNetwork::calculate_outputs_histograms(const Index& points_number, const Index& bins_number)
{
    const Index inputs_number = get_inputs_number();

    Tensor<type, 2> inputs(points_number, inputs_number);
/*
    if(scaling_layer_pointer == nullptr)
    {
    }
    else
    {
        const Tensor<ScalingLayer::ScalingMethod, 1> scaling_methods = scaling_layer_pointer->get_scaling_methods();

        for(Index i = 0; i < scaling_methods.size(); i++)
        {
            Tensor<type, 1> input_column(points_number, 0.0);

            if(scaling_methods[i] == ScalingLayer::NoScaling)
            {
                input_column.setRandom();
            }
            else if(scaling_methods[i] == ScalingLayer::MinimumMaximum)
            {
                type minimum = scaling_layer_pointer->get_descriptives(i).minimum;
                type maximum = scaling_layer_pointer->get_descriptives(i).maximum;

                input_column.setRandom(minimum, maximum);
            }
            else if(scaling_methods[i] == ScalingLayer::MeanStandardDeviation)
            {
                type mean = scaling_layer_pointer->get_descriptives(i).mean;
                type standard_deviation = scaling_layer_pointer->get_descriptives(i).standard_deviation;

                input_column.setRandom(mean, standard_deviation);
            }
            else if(scaling_methods[i] == ScalingLayer::StandardDeviation)
            {
                type mean = scaling_layer_pointer->get_descriptives(i).mean;
                type standard_deviation = scaling_layer_pointer->get_descriptives(i).standard_deviation;

                input_column.setRandom(mean, standard_deviation);
            }

            inputs.set_column(i, input_column, "");
        }
    }

    const Tensor<type, 2> outputs = calculate_outputs(inputs);

    return histograms(outputs.to_matrix(), bins_number);
*/
    return Tensor<Histogram, 1>();
}


/// Calculates the histogram of the outputs with a matrix of given inputs.
/// @param inputs Matrix of the data to evaluate the neural network.
/// @param bins_number Number of bins for the histograms.

Tensor<Histogram, 1> NeuralNetwork::calculate_outputs_histograms(const Tensor<type, 2>& inputs, const Index& bins_number)
{
   const Tensor<type, 2> outputs = calculate_outputs(inputs);

   return histograms(outputs, bins_number);
}


/// Returns a string representation of the current neural network object.

string NeuralNetwork::object_to_string() const
{
    ostringstream buffer;

    buffer << "Neural network:\n";
/*
    buffer << "Inputs names:\n";
    buffer << inputs_names << endl;
*/
    // Layers

    const Index layers_number = get_layers_number();

    buffer << "Layers number: " << layers_number << endl;

    for(Index i = 0; i < layers_number; i++)
    {
        buffer << "Layer " << i+1 << ":" << endl;

        buffer << layers_pointers[i]->object_to_string() << endl;
    }
/*
    buffer << "Outputs names:\n";
    buffer << outputs_names << endl;
*/
    return buffer.str();
}


///@todo

Tensor<string, 2> NeuralNetwork::get_information() const
{
    return Tensor<string, 2>();
}


/// Serializes the neural network object into a XML document of the TinyXML library.
/// See the OpenNN manual for more information about the format of this element.
///@todo

tinyxml2::XMLDocument* NeuralNetwork::to_XML() const
{
    tinyxml2::XMLDocument* document = new tinyxml2::XMLDocument;

    tinyxml2::XMLElement* neural_network_element = document->NewElement("NeuralNetwork");

    document->InsertFirstChild(neural_network_element);

    ostringstream buffer;

    // Inputs

//    if(inputs_pointer)
//    {
//        tinyxml2::XMLDocument* inputs_document = inputs_pointer->to_XML();

//        const tinyxml2::XMLElement* inputs_element = inputs_document->FirstChildElement("Inputs");

//        tinyxml2::XMLNode* node = inputs_element->DeepClone(document);

//        neural_network_element->InsertEndChild(node);

//        delete inputs_document;
//    }

    // Outputs

//    if(outputs_pointer)
//    {
//        const tinyxml2::XMLDocument* outputs_document = outputs_pointer->to_XML();

//        const tinyxml2::XMLElement* outputs_element = outputs_document->FirstChildElement("Outputs");

//        tinyxml2::XMLNode* node = outputs_element->DeepClone(document);

//        neural_network_element->InsertEndChild(node);

//        delete outputs_document;
//    }

    //   // Display warnings
    //   {
    //      tinyxml2::XMLElement* display_element = document->NewElement("Display");
    //      neural_network_element->LinkEndChild(display_element);

    //      buffer.str("");
    //      buffer << display;

    //      tinyxml2::XMLText* display_text = document->NewText(buffer.str().c_str());
    //      display_element->LinkEndChild(display_text);
    //   }

    return document;
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


/// Prints to the screen the members of a neural network object in a XML-type format.

void NeuralNetwork::print() const
{
    if(display) cout << object_to_string();
}


void NeuralNetwork::print_summary() const
{
    const Index layers_number = get_layers_number();

    cout << "Layers number: " << layers_number << endl;

    for(Index i = 0; i < layers_number; i++)
    {
        cout << "Layer " << i+1 << ": " << layers_pointers[i]->get_type_string() << endl;
    }
}


/// Saves to a XML file the members of a neural network object.
/// @param file_name Name of neural network XML file.

void NeuralNetwork::save(const string& file_name) const
{
    tinyxml2::XMLDocument* document = to_XML();

    document->SaveFile(file_name.c_str());

    delete document;
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


/// Returns a string with the expression of the function represented by the neural network.

string NeuralNetwork::write_expression() const
{
    ostringstream buffer;

    const Index inputs_number = get_inputs_number();
    const Index outputs_number = get_outputs_number();

    Tensor<string, 1> inputs_names = get_inputs_names();
    Tensor<string, 1> outputs_names = get_outputs_names();
/*
    cout << "Inputs names: " << inputs_names << endl;
    cout << "Outputs names: " << outputs_names << endl;

    Index position = 0;

    string search;
    string replace;

    for(Index i = 0; i < inputs_number; i++)
    {
        position = 0;

        search = "(";
        replace = "_";

        while((position = inputs_names[i].find(search, position)) != string::npos)
        {
            inputs_names[i].replace(position, search.length(), replace);
            position += replace.length();
        }

        string::iterator end_pos = remove(inputs_names[i].begin(), inputs_names[i].end(), ' ');
        inputs_names[i].erase(end_pos, inputs_names[i].end());

        position = 0;

        search = "-";
        replace = "_";

        while((position = inputs_names[i].find(search, position)) != string::npos)
        {
            inputs_names[i].replace(position, search.length(), replace);
            position += replace.length();
        }

        position = 0;

        search = "/";
        replace = "_";

        while((position = inputs_names[i].find(search, position)) != string::npos)
        {
            inputs_names[i].replace(position, search.length(), replace);
            position += replace.length();
        }

        position = 0;

        search = ")";
        replace = "_";

        while((position = inputs_names[i].find(search, position)) != string::npos)
        {
            inputs_names[i].replace(position, search.length(), replace);
            position += replace.length();
        }

        position = 0;

        search = ":";
        replace = "_";

        while((position = inputs_names[i].find(search, position)) != string::npos)
        {
            inputs_names[i].replace(position, search.length(), replace);
            position += replace.length();
        }
    }

    for(Index i = 0; i < outputs_number; i++)
    {
        position = 0;

        search = "(";
        replace = "_";

        while((position = outputs_names[i].find(search, position)) != string::npos)
        {
            outputs_names[i].replace(position, search.length(), replace);
            position += replace.length();
        }

        string::iterator end_pos = remove(outputs_names[i].begin(), outputs_names[i].end(), ' ');
        outputs_names[i].erase(end_pos, outputs_names[i].end());

        position = 0;

        search = "-";
        replace = "_";

        while((position = outputs_names[i].find(search, position)) != string::npos)
        {
            outputs_names[i].replace(position, search.length(), replace);
            position += replace.length();
        }

        position = 0;

        search = "/";
        replace = "_";

        while((position = outputs_names[i].find(search, position)) != string::npos)
        {
            outputs_names[i].replace(position, search.length(), replace);
            position += replace.length();
        }

        position = 0;

        search = ")";
        replace = "_";

        while((position = outputs_names[i].find(search, position)) != string::npos)
        {
            outputs_names[i].replace(position, search.length(), replace);
            position += replace.length();
        }

        position = 0;

        search = ":";
        replace = "_";

        while((position = outputs_names[i].find(search, position)) != string::npos)
        {
            outputs_names[i].replace(position, search.length(), replace);
            position += replace.length();
        }

    }

    // Scaled inputs

    Tensor<string, 1> scaled_inputs_name(inputs_names.size());

    for(Index i = 0; i < inputs_names.size(); i++)
    {
        buffer.str("");

        buffer << "scaled_" << inputs_names[i];

        scaled_inputs_name[i] = buffer.str();
    }

    // Principal components

    Tensor<string, 1> principal_components_name(inputs_number);

    for(Index i = 0; i < inputs_number; i++)
    {
        buffer.str("");

        buffer << "principal_component_" <<(i+1);

        principal_components_name[i] = buffer.str();
    }

    // Scaled outputs

    Tensor<string, 1> scaled_outputs_name(outputs_names.size());

    for(Index i = 0; i < outputs_names.size(); i++)
    {
        buffer.str("");

        buffer << "scaled_" << outputs_names[i];

        scaled_outputs_name[i] = buffer.str();
    }

    // Non probabilistic outputs

    Tensor<string, 1> non_probabilistic_outputs_name(outputs_number);

    for(Index i = 0; i < outputs_number; i++)
    {
        buffer.str("");

        buffer << "non_probabilistic_" << outputs_names[i];

        non_probabilistic_outputs_name[i] = buffer.str();
    }

    buffer.str("");

    ///@todo write expression for each layer

//    // Scaling layer
//    if(has_scaling_layer())
//    {
//        buffer << scaling_layer_pointer->write_expression(inputs_name, scaled_inputs_name);
//    }
//    // Principal components layer
//    if(has_principal_components_layer())
//    {
//        buffer << principal_components_layer_pointer->write_expression(scaled_inputs_name, principal_components_name);
//    }
//    // Multilayer perceptron
//    if(has_multilayer_perceptron())
//    {
//        if(scaling_layer_pointer && unscaling_layer_pointer)
//        {
//            if(has_principal_components_layer() && principal_components_layer_pointer->write_principal_components_method() != "NoPrincipalComponents")
//            {
//                buffer << multilayer_perceptron_pointer->write_expression(principal_components_name, scaled_outputs_name);
//            }
//            else
//            {
//                buffer << multilayer_perceptron_pointer->write_expression(scaled_inputs_name, scaled_outputs_name);
//            }
//        }
//        else if(scaling_layer_pointer && probabilistic_layer_pointer)
//        {
//            if(has_principal_components_layer() && principal_components_layer_pointer->write_principal_components_method() != "NoPrincipalComponents")
//            {
//                buffer << multilayer_perceptron_pointer->write_expression(principal_components_name, scaled_outputs_name);
//            }
//            else
//            {
//                buffer << multilayer_perceptron_pointer->write_expression(scaled_inputs_name, non_probabilistic_outputs_name);
//            }
//        }
//        else
//        {
//            buffer << multilayer_perceptron_pointer->write_expression(inputs_name, outputs_name);
//        }
//    }
//    // Outputs unscaling
//    if(has_unscaling_layer())
//    {
//        buffer << unscaling_layer_pointer->write_expression(scaled_outputs_name, outputs_name);
//    }
//    // Outputs trending layer
//    if(has_outputs_trending_layer())
//    {
//        buffer << outputs_trending_layer_pointer->write_expression(outputs_name, outputs_name);
//    }
//    // Probabilistic layer
//    if(has_probabilistic_layer())
//    {
//        buffer << probabilistic_layer_pointer->write_expression(non_probabilistic_outputs_name, outputs_name);
//    }
//    // Bounding layer
//    if(has_bounding_layer())
//    {
//        buffer << bounding_layer_pointer->write_expression(outputs_name, outputs_name);
//    }

    string expression = buffer.str();

    position = 0;

    search = "--";
    replace = "+";

    while((position = expression.find(search, position)) != string::npos)
    {
        expression.replace(position, search.length(), replace);
        position += replace.length();
    }

    position = 0;

    search = "+-";
    replace = "-";

    while((position = expression.find(search, position)) != string::npos)
    {
        expression.replace(position, search.length(), replace);
        position += replace.length();
    }

    position = 0;

    search = "\n-";
    replace = "-";

    while((position = expression.find(search, position)) != string::npos)
    {
        expression.replace(position, search.length(), replace);
        position += replace.length();
    }

    position = 0;

    search = "\n+";
    replace = "+";

    while((position = expression.find(search, position)) != string::npos)
    {
        expression.replace(position, search.length(), replace);
        position += replace.length();
    }

    position = 0;

    search = "\"";
    replace = "";

    while((position = expression.find(search, position)) != string::npos)
    {
        expression.replace(position, search.length(), replace);
        position += replace.length();
    }

    return expression;
    */

    return "";
}


/// Returns a string with the expression of the function represented by the neural network.

string NeuralNetwork::write_mathematical_expression_php() const
{
    ostringstream buffer;

    const Index inputs_number = get_inputs_number();
    const Index outputs_number = get_outputs_number();

    Tensor<string, 1> inputs_names = get_inputs_names();
    Tensor<string, 1> outputs_names = get_outputs_names();

    Index position = 0;

    string search;
    string replace;

    for(Index i = 0; i < inputs_number; i++)
    {
        position = 0;

        search = "(";
        replace = "_";

        while((position = inputs_names[i].find(search, position)) != string::npos)
        {
            inputs_names[i].replace(position, search.length(), replace);
            position += replace.length();
        }

        string::iterator end_pos = remove(inputs_names[i].begin(), inputs_names[i].end(), ' ');
        inputs_names[i].erase(end_pos, inputs_names[i].end());

        position = 0;

        search = "-";
        replace = "_";

        while((position = inputs_names[i].find(search, position)) != string::npos)
        {
            inputs_names[i].replace(position, search.length(), replace);
            position += replace.length();
        }

        position = 0;

        search = "/";
        replace = "_";

        while((position = inputs_names[i].find(search, position)) != string::npos)
        {
            inputs_names[i].replace(position, search.length(), replace);
            position += replace.length();
        }

        position = 0;

        search = ")";
        replace = "_";

        while((position = inputs_names[i].find(search, position)) != string::npos)
        {
            inputs_names[i].replace(position, search.length(), replace);
            position += replace.length();
        }
    }

    for(Index i = 0; i < outputs_number; i++)
    {
        position = 0;

        search = "(";
        replace = "_";

        while((position = outputs_names[i].find(search, position)) != string::npos)
        {
            outputs_names[i].replace(position, search.length(), replace);
            position += replace.length();
        }

        string::iterator end_pos = remove(outputs_names[i].begin(), outputs_names[i].end(), ' ');
        outputs_names[i].erase(end_pos, outputs_names[i].end());

        position = 0;

        search = "-";
        replace = "_";

        while((position = outputs_names[i].find(search, position)) != string::npos)
        {
            outputs_names[i].replace(position, search.length(), replace);
            position += replace.length();
        }

        position = 0;

        search = "/";
        replace = "_";

        while((position = outputs_names[i].find(search, position)) != string::npos)
        {
            outputs_names[i].replace(position, search.length(), replace);
            position += replace.length();
        }

        position = 0;

        search = ")";
        replace = "_";

        while((position = outputs_names[i].find(search, position)) != string::npos)
        {
            outputs_names[i].replace(position, search.length(), replace);
            position += replace.length();
        }
    }

    // Scaled inputs

    Tensor<string, 1> scaled_inputs_name(inputs_names.size());

    for(Index i = 0; i < inputs_names.size(); i++)
    {
        buffer.str("");

        buffer << "$scaled_" << inputs_names[i];

        scaled_inputs_name[i] = buffer.str();
    }

    // Principal components

    Tensor<string, 1> principal_components_name(inputs_number);

    for(Index i = 0; i < inputs_number; i++)
    {
        buffer.str("");

        buffer << "$principal_component_" <<(i+1);

        principal_components_name[i] = buffer.str();
    }

    // Scaled outputs

    Tensor<string, 1> scaled_outputs_name(outputs_number);

    for(Index i = 0; i < outputs_number; i++)
    {
        buffer.str("");

        buffer << "$scaled_" << outputs_names[i];

        scaled_outputs_name[i] = buffer.str();
    }

    // Non probabilistic outputs

    Tensor<string, 1> non_probabilistic_outputs_name(outputs_number);

    for(Index i = 0; i < outputs_number; i++)
    {
        buffer.str("");

        buffer << "$non_probabilistic_" << outputs_names[i];

        non_probabilistic_outputs_name[i] = buffer.str();
    }

    buffer.str("");

    for(Index i = 0; i < inputs_names.size(); i++)
    {
        inputs_names[i] = "$"+inputs_names[i];
    }

    for(Index i = 0; i < outputs_names.size(); i++)
    {
        outputs_names[i] = "$"+outputs_names[i];
    }

    string expression = buffer.str();

    position = 0;

    search = "--";
    replace = "+";

    while((position = expression.find(search, position)) != string::npos)
    {
        expression.replace(position, search.length(), replace);
        position += replace.length();
    }

    position = 0;

    search = "+-";
    replace = "-";

    while((position = expression.find(search, position)) != string::npos)
    {
        expression.replace(position, search.length(), replace);
        position += replace.length();
    }

    position = 0;

    search = "\n-";
    replace = "-";

    while((position = expression.find(search, position)) != string::npos)
    {
        expression.replace(position, search.length(), replace);
        position += replace.length();
    }

    position = 0;

    search = "\n+";
    replace = "+";

    while((position = expression.find(search, position)) != string::npos)
    {
        expression.replace(position, search.length(), replace);
        position += replace.length();
    }

    position = 0;

    search = "\"";
    replace = "";

    while((position = expression.find(search, position)) != string::npos)
    {
        expression.replace(position, search.length(), replace);
        position += replace.length();
    }

    return expression;
}


/// Returns a string with the python function of the expression represented by the neural network.

string NeuralNetwork::write_expression_python() const
{
    ostringstream buffer;

    const Index inputs_number = get_inputs_number();
    const Index outputs_number = get_outputs_number();

    Tensor<string, 1> inputs_names = get_inputs_names();
    Tensor<string, 1> outputs_names = get_outputs_names();

    Index pos;

    string search;
    string replace;

    for(Index i = 0; i < inputs_number; i++)
    {
        string::iterator end_pos = remove(inputs_names[i].begin(), inputs_names[i].end(), ' ');
        inputs_names[i].erase(end_pos, inputs_names[i].end());

        pos = 0;

        search = "-";
        replace = "_";

        while((pos = inputs_names[i].find(search, pos)) != string::npos)
        {
            inputs_names[i].replace(pos, search.length(), replace);
            pos += replace.length();
        }

        pos = 0;

        search = "(";
        replace = "_";

        while((pos = inputs_names[i].find(search, pos)) != string::npos)
        {
            inputs_names[i].replace(pos, search.length(), replace);
            pos += replace.length();
        }

        pos = 0;

        search = ")";
        replace = "_";

        while((pos = inputs_names[i].find(search, pos)) != string::npos)
        {
            inputs_names[i].replace(pos, search.length(), replace);
            pos += replace.length();
        }

        pos = 0;

        search = "+";
        replace = "_";

        while((pos = inputs_names[i].find(search, pos)) != string::npos)
        {
            inputs_names[i].replace(pos, search.length(), replace);
            pos += replace.length();
        }

        pos = 0;

        search = "*";
        replace = "_";

        while((pos = inputs_names[i].find(search, pos)) != string::npos)
        {
            inputs_names[i].replace(pos, search.length(), replace);
            pos += replace.length();
        }

        pos = 0;

        search = "/";
        replace = "_";

        while((pos = inputs_names[i].find(search, pos)) != string::npos)
        {
            inputs_names[i].replace(pos, search.length(), replace);
            pos += replace.length();
        }
    }

    for(Index i = 0; i < outputs_number; i++)
    {
        string::iterator end_pos = remove(outputs_names[i].begin(), outputs_names[i].end(), ' ');
        outputs_names[i].erase(end_pos, outputs_names[i].end());

        pos = 0;

        search = "-";
        replace = "_";

        while((pos = outputs_names[i].find(search, pos)) != string::npos)
        {
            outputs_names[i].replace(pos, search.length(), replace);
            pos += replace.length();
        }

        pos = 0;

        search = "(";
        replace = "_";

        while((pos = outputs_names[i].find(search, pos)) != string::npos)
        {
            outputs_names[i].replace(pos, search.length(), replace);
            pos += replace.length();
        }

        pos = 0;

        search = ")";
        replace = "_";

        while((pos = outputs_names[i].find(search, pos)) != string::npos)
        {
            outputs_names[i].replace(pos, search.length(), replace);
            pos += replace.length();
        }

        pos = 0;

        search = "+";
        replace = "_";

        while((pos = outputs_names[i].find(search, pos)) != string::npos)
        {
            outputs_names[i].replace(pos, search.length(), replace);
            pos += replace.length();
        }

        pos = 0;

        search = "*";
        replace = "_";

        while((pos = outputs_names[i].find(search, pos)) != string::npos)
        {
            outputs_names[i].replace(pos, search.length(), replace);
            pos += replace.length();
        }

        pos = 0;

        search = "/";
        replace = "_";

        while((pos = outputs_names[i].find(search, pos)) != string::npos)
        {
            outputs_names[i].replace(pos, search.length(), replace);
            pos += replace.length();
        }
    }

    Tensor<PerceptronLayer::ActivationFunction, 1> activations;

//    const Index layers_number = get_layers_number();

//    for(Index i = 0; i < layers_number; i++)
//        activations.push_back(layers_pointers[i].get_activation_function());

    buffer.str("");

    buffer << "#!/usr/bin/python\n\n";
/*
    if(activations.contains(PerceptronLayer::Threshold))
    {
        buffer << "def Threshold(x) : \n"
                  "   if x < 0 : \n"
                  "       return 0\n"
                  "   else : \n"
                  "       return 1\n\n";
    }

    if(activations.contains(PerceptronLayer::SymmetricThreshold))
    {
        buffer << "def SymmetricThreshold(x) : \n"
                  "   if x < 0 : \n"
                  "       return -1\n"
                  "   else : \n"
                  "       return 1\n\n";
    }

    if(activations.contains(PerceptronLayer::Logistic))
    {
        buffer << "from math import exp\n"
                  "def Logistic(x) : \n"
                  "   return(1/(1+exp(-x))) \n\n";
    }

    if(activations.contains(PerceptronLayer::HyperbolicTangent))
    {
        buffer << "from math import tanh\n\n";
    }
*/
//    if(has_probabilistic_layer())
//    {
//        type decision_threshold = probabilistic_layer_pointer->get_decision_threshold();

//        switch(probabilistic_layer_pointer->get_activation_function())
//        {
//            case ProbabilisticLayer::Probability :
//            {
//                buffer << "def Binary(x) : \n"
//                          "   if x < " << decision_threshold << " : \n"
//                          "       return 0\n"
//                          "   else : \n"
//                          "       return 1\n\n";
//            }
//            break;
//        case ProbabilisticLayer::Binary :
//        {
//            buffer << "def Probability(x) : \n"
//                      "   if x < 0 :\n"
//                      "       return 0\n"
//                      "   elif x > 1 :\n"
//                      "       return 1\n"
//                      "   else : \n"
//                      "       return x\n\n";
//        }
//            break;
//        case ProbabilisticLayer::Competitive :
//        {
//            buffer << "def Competitive(";
//            for(Index i = 0; i < outputs_number; i++)
//            {
//                buffer << "x" << i;

//                if(i != outputs_number - 1)
//                    buffer << ", ";
//            }
//            buffer << ") :\n";

//            buffer << "   inputs = [";
//            for(Index i = 0; i < outputs_number; i++)
//            {
//                buffer << "x" << i;

//                if(i != outputs_number - 1)
//                    buffer << ", ";
//            }
//            buffer << "]\n";
//            buffer << "   competitive = [0 for i in range(" << outputs_number << ")]\n"
//                                                                                  "   maximal_index = inputs.index(max(inputs))\n"
//                                                                                  "   competitive[maximal_index] = 1\n"
//                                                                                  "   return competitive\n\n";
//        }

//            break;
//        case ProbabilisticLayer::Softmax :
//        {
//            buffer << "from math import exp\n"
//                      "def Softmax(";
//            for(Index i = 0; i < outputs_number; i++)
//            {
//                buffer << "x" << i;

//                if(i != outputs_number - 1)
//                    buffer << ", ";
//            }
//            buffer << ") :\n";

//            buffer << "   inputs = [";
//            for(Index i = 0; i < outputs_number; i++)
//            {
//                buffer << "x" << i;

//                if(i != outputs_number - 1)
//                    buffer << ", ";
//            }
//            buffer << "]\n";
//            buffer << "   softmax = [0 for i in range(" << outputs_number << ")]\n"
//            "   sum = 0\n"
//            "   for i in range(" << outputs_number << ") :\n"
//            "       sum += exp(inputs[i])\n"
//            "   for i in range(" << outputs_number << ") :\n"
//                                                                                                                                                                    "       softmax[i] = exp(inputs[i])/sum\n";
//            buffer << "   return softmax\n\n";
//        }
//            break;

//        case ProbabilisticLayer::NoProbabilistic :
//            break;
//        }
//    }

    buffer << "def expression(inputs) : \n\n    ";

    buffer << "if type(inputs) != list:\n    "
           << "   print('Argument must be a list')\n    "
           << "   exit()\n    ";

    buffer << "if len(inputs) != " << inputs_number << ":\n    "
           << "   print('Incorrect number of inputs')\n    "
           << "   exit()\n    ";

    for(Index i = 0; i < inputs_number; i++)
    {
        buffer << inputs_names[i] << "=inputs[" << i << "]\n    ";
    }

    string neural_network_expression = write_expression();

    pos = 0;

    search = "\n";
    replace = "\n    ";

    while((pos = neural_network_expression.find(search, pos)) != string::npos)
    {
        neural_network_expression.replace(pos, search.length(), replace);
        pos += replace.length();
    }

    buffer << neural_network_expression;

    buffer << "\n    return ";

    for(Index i = 0; i < outputs_number; i++)
    {
        buffer << outputs_names[i];

        if(i != outputs_number - 1)
            buffer << ", ";
    }

    buffer << " \n";
    string expression = buffer.str();

    pos = 0;

    search = "\"";
    replace = "";

    while((pos = expression.find(search, pos)) != string::npos)
    {
        expression.replace(pos, search.length(), replace);
        pos += replace.length();
    }

    pos = 0;

    search = ";";
    replace = "";

    while((pos = expression.find(search, pos)) != string::npos)
    {
        expression.replace(pos, search.length(), replace);
        pos += replace.length();
    }

    return expression;

}


/// Returns a string with the php function of the expression represented by the neural network.
/// @todo

string NeuralNetwork::write_expression_php() const
{
    ostringstream buffer;

    const Index inputs_number = get_inputs_number();
    const Index outputs_number = get_outputs_number();

    Tensor<string, 1> inputs_names = get_inputs_names();
    Tensor<string, 1> outputs_names = get_outputs_names();

    Index pos;

    string search;
    string replace;

    for(Index i = 0; i < inputs_number; i++)
    {
        string::iterator end_pos = remove(inputs_names[i].begin(), inputs_names[i].end(), ' ');
        inputs_names[i].erase(end_pos, inputs_names[i].end());

        pos = 0;

        search = "-";
        replace = "_";

        while((pos = inputs_names[i].find(search, pos)) != string::npos)
        {
            inputs_names[i].replace(pos, search.length(), replace);
            pos += replace.length();
        }

        pos = 0;

        search = "(";
        replace = "_";

        while((pos = inputs_names[i].find(search, pos)) != string::npos)
        {
            inputs_names[i].replace(pos, search.length(), replace);
            pos += replace.length();
        }

        pos = 0;

        search = ")";
        replace = "_";

        while((pos = inputs_names[i].find(search, pos)) != string::npos)
        {
            inputs_names[i].replace(pos, search.length(), replace);
            pos += replace.length();
        }

        pos = 0;

        search = "+";
        replace = "_";

        while((pos = inputs_names[i].find(search, pos)) != string::npos)
        {
            inputs_names[i].replace(pos, search.length(), replace);
            pos += replace.length();
        }

        pos = 0;

        search = ":";
        replace = "_";


        while((pos = inputs_names[i].find(search, pos)) != string::npos)
        {
            inputs_names[i].replace(pos, search.length(), replace);
            pos += replace.length();
        }

        pos = 0;

        search = "*";
        replace = "_";

        while((pos = inputs_names[i].find(search, pos)) != string::npos)
        {
            inputs_names[i].replace(pos, search.length(), replace);
            pos += replace.length();
        }

        pos = 0;

        search = "/";
        replace = "_";

        while((pos = inputs_names[i].find(search, pos)) != string::npos)
        {
            inputs_names[i].replace(pos, search.length(), replace);
            pos += replace.length();
        }
    }

    for(Index i = 0; i < outputs_number; i++)
    {
        string::iterator end_pos = remove(outputs_names[i].begin(), outputs_names[i].end(), ' ');
        outputs_names[i].erase(end_pos, outputs_names[i].end());

        pos = 0;

        search = "-";
        replace = "_";

        while((pos = outputs_names[i].find(search, pos)) != string::npos)
        {
            outputs_names[i].replace(pos, search.length(), replace);
            pos += replace.length();
        }

        pos = 0;

        search = "(";
        replace = "_";

        while((pos = outputs_names[i].find(search, pos)) != string::npos)
        {
            outputs_names[i].replace(pos, search.length(), replace);
            pos += replace.length();
        }

        pos = 0;

        search = ")";
        replace = "_";

        while((pos = outputs_names[i].find(search, pos)) != string::npos)
        {
            outputs_names[i].replace(pos, search.length(), replace);
            pos += replace.length();
        }

        pos = 0;

        search = ":";
        replace = "_";

        while((pos = outputs_names[i].find(search, pos)) != string::npos)
        {
            outputs_names[i].replace(pos, search.length(), replace);
            pos += replace.length();
        }

        pos = 0;

        search = "+";
        replace = "_";

        while((pos = outputs_names[i].find(search, pos)) != string::npos)
        {
            outputs_names[i].replace(pos, search.length(), replace);
            pos += replace.length();
        }

        pos = 0;

        search = "*";
        replace = "_";

        while((pos = outputs_names[i].find(search, pos)) != string::npos)
        {
            outputs_names[i].replace(pos, search.length(), replace);
            pos += replace.length();
        }

        pos = 0;

        search = "/";
        replace = "_";

        while((pos = outputs_names[i].find(search, pos)) != string::npos)
        {
            outputs_names[i].replace(pos, search.length(), replace);
            pos += replace.length();
        }
    }

    Tensor<PerceptronLayer::ActivationFunction, 1> activations;

//    const Index layers_number = get_layers_number();

//    for(Index i = 0; i < layers_number; i++)
//        activations.push_back(layers_pointers[i]->get_activation_function());

    buffer.str("");
/*
    if(activations.contains(PerceptronLayer::Threshold))
    {
        buffer << "function Threshold($x)\n"
                  "{\n"
                  "   if($x < 0)\n"
                  "   {\n"
                  "       return 0;\n"
                  "   }\n"
                  "   else\n"
                  "   {\n"
                  "       return 1;\n"
                  "   }\n"
                  "}\n\n";
    }

    if(activations.contains(PerceptronLayer::SymmetricThreshold))
    {
        buffer << "function SymmetricThreshold(&x)\n"
                  "{\n"
                  "   if($x < 0)\n"
                  "   {\n"
                  "       return -1;\n"
                  "   }\n"
                  "   else\n"
                  "   {\n"
                  "       return 1;\n"
                  "   }\n"
                  "}\n\n";
    }

    if(activations.contains(PerceptronLayer::Logistic))
    {
        buffer << "function Logistic($x)\n"
                  "{\n"
                  "   return(1/(1+exp(-$x)));"
                  "}\n\n";
    }
*/
    buffer << "function expression($inputs)\n"
              "{\n";

    buffer << "   if(!is_array($inputs))\n"
              "   {\n"
              "       throw new \\InvalidArgumentException('Argument must be a list.', 1);\n"
              "   }\n";

    buffer << "   if(count($inputs) != " << inputs_number << ")\n"
              "   {\n"
              "       throw new \\InvalidArgumentException('Incorrect number of inputs.', 2);\n"
              "   }\n\n";

    for(Index i = 0; i < inputs_names.size(); i++)
    {
        inputs_names[i] = "$_"+inputs_names[i];
    }

    for(Index i = 0; i < outputs_names.size(); i++)
    {
        outputs_names[i] = "$"+outputs_names[i];
    }

    for(Index i = 0; i < inputs_number; i++)
    {
        buffer << "   " << inputs_names[i] << "=$inputs[" << i << "];\n";
    }

    string neural_network_expression = write_mathematical_expression_php();

    pos = 0;

    search = "\n";
    replace = "\n    ";

    while((pos = neural_network_expression.find(search, pos)) != string::npos)
    {
        neural_network_expression.replace(pos, search.length(), replace);
        pos += replace.length();
    }

    buffer << neural_network_expression;

    buffer << "\n    return ";

    for(Index i = 0; i < outputs_number; i++)
    {
        buffer << outputs_names[i];

        if(i != outputs_number - 1)
            buffer << ", ";
        else
            buffer << ";";
    }

    buffer << " \n";
    buffer << "}";

    string expression = buffer.str();

    pos = 0;

    search = "\"";
    replace = "";

    while((pos = expression.find(search, pos)) != string::npos)
    {
        expression.replace(pos, search.length(), replace);
        pos += replace.length();
    }

//    pos = 0;

//    search = ";";
//    replace = "";

//    while((pos = expression.find(search, pos)) != string::npos)
//    {
//        expression.replace(pos, search.length(), replace);
//        pos += replace.length();
//    }

    return expression;
}


/// Returns a string with the R function of the expression represented by the neural network.
/// @todo

string NeuralNetwork::write_expression_R() const
{
    ostringstream buffer;

    const Index inputs_number = get_inputs_number();
    const Index outputs_number = get_outputs_number();

    Tensor<string, 1> inputs_names = get_inputs_names();
    Tensor<string, 1> outputs_names = get_outputs_names();

    Index pos = 0;

    string search;
    string replace;

    for(Index i = 0; i < inputs_number; i++)
    {
        pos = 0;

        search = "-";
        replace = "_";

        while((pos = inputs_names[i].find(search, pos)) != string::npos)
        {
            inputs_names[i].replace(pos, search.length(), replace);
            pos += replace.length();
        }

        string::iterator end_pos = remove(inputs_names[i].begin(), inputs_names[i].end(), ' ');
        inputs_names[i].erase(end_pos, inputs_names[i].end());

        pos = 0;

        search = "(";
        replace = "_";

        while((pos = inputs_names[i].find(search, pos)) != string::npos)
        {
            inputs_names[i].replace(pos, search.length(), replace);
            pos += replace.length();
        }

        pos = 0;

        search = ":";
        replace = "_";

        while((pos = inputs_names[i].find(search, pos)) != string::npos)
        {
            inputs_names[i].replace(pos, search.length(), replace);
            pos += replace.length();
        }

        pos = 0;

        search = ")";
        replace = "_";

        while((pos = inputs_names[i].find(search, pos)) != string::npos)
        {
            inputs_names[i].replace(pos, search.length(), replace);
            pos += replace.length();
        }

        pos = 0;

        search = "+";
        replace = "_";

        while((pos = inputs_names[i].find(search, pos)) != string::npos)
        {
            inputs_names[i].replace(pos, search.length(), replace);
            pos += replace.length();
        }

        pos = 0;

        search = "*";
        replace = "_";

        while((pos = inputs_names[i].find(search, pos)) != string::npos)
        {
            inputs_names[i].replace(pos, search.length(), replace);
            pos += replace.length();
        }

        pos = 0;

        search = "/";
        replace = "_";

        while((pos = inputs_names[i].find(search, pos)) != string::npos)
        {
            outputs_names[i].replace(pos, search.length(), replace);
            pos += replace.length();
        }

        pos = 0;

        search = ":";
        replace = "_";

        while((pos = inputs_names[i].find(search, pos)) != string::npos)
        {
            inputs_names[i].replace(pos, search.length(), replace);
            pos += replace.length();
        }

    }

    for(Index i = 0; i < outputs_number; i++)
    {
        pos = 0;

        search = "-";
        replace = "_";

        while((pos = outputs_names[i].find(search, pos)) != string::npos)
        {
            outputs_names[i].replace(pos, search.length(), replace);
            pos += replace.length();
        }

        string::iterator end_pos = remove(outputs_names[i].begin(), outputs_names[i].end(), ' ');
        outputs_names[i].erase(end_pos, outputs_names[i].end());

        pos = 0;

        search = "(";
        replace = "_";

        while((pos = outputs_names[i].find(search, pos)) != string::npos)
        {
            outputs_names[i].replace(pos, search.length(), replace);
            pos += replace.length();
        }

        pos = 0;

        search = ")";
        replace = "_";

        while((pos = outputs_names[i].find(search, pos)) != string::npos)
        {
            outputs_names[i].replace(pos, search.length(), replace);
            pos += replace.length();
        }

        pos = 0;

        search = "+";
        replace = "_";

        while((pos = outputs_names[i].find(search, pos)) != string::npos)
        {
            outputs_names[i].replace(pos, search.length(), replace);
            pos += replace.length();
        }

        pos = 0;

        search = "*";
        replace = "_";

        while((pos = outputs_names[i].find(search, pos)) != string::npos)
        {
            outputs_names[i].replace(pos, search.length(), replace);
            pos += replace.length();
        }

        pos = 0;

        search = "/";
        replace = "_";

        while((pos = outputs_names[i].find(search, pos)) != string::npos)
        {
            outputs_names[i].replace(pos, search.length(), replace);
            pos += replace.length();
        }
    }

    Tensor<PerceptronLayer::ActivationFunction, 1> activations;

//    const Index layers_number = get_layers_number();

//    for(Index i = 0; i < layers_number; i++)
//	{
//        activations.push_back(layers_pointers[i]->get_activation_function());
//	}

    buffer.str("");
/*
    if(activations.contains(PerceptronLayer::Threshold))
    {
        buffer << "Threshold <- function(x) { \n"
                  "   if(x < 0)  0 \n"
                  "   else 1 \n"
                  "}\n\n";
    }

    if(activations.contains(PerceptronLayer::SymmetricThreshold))
    {
        buffer << "SymmetricThreshold <- function(x) { \n"
                  "   if(x < 0)  -1 \n"
                  "   else 1 \n"
                  "}\n\n";
    }

    if(activations.contains(PerceptronLayer::Logistic))
    {
        buffer << "Logistic <- function(x) { \n"
                  "   1/(1+exp(-x))\n"
                  "}\n\n";
    }
*/
    buffer << "expression <- function(inputs) {\n\n    ";

    buffer << "if(length(inputs) != " << inputs_number << ") {\n    "
           << "   print('Incorrect number of inputs')\n    "
           << "   return )\n    "
              "}\n    ";

    for(Index i = 0; i < inputs_number; i++)
    {
        buffer << inputs_names[i] << "=inputs[" << i+1 << "]\n    ";
    }

    string neural_network_expression = write_expression();

    pos = 0;

    search = "\n";
    replace = "\n    ";

    while((pos = neural_network_expression.find(search, pos)) != string::npos)
    {
        neural_network_expression.replace(pos, search.length(), replace);
        pos += replace.length();
    }

    pos = 0;

    search = ";";
    replace = "";

    while((pos = neural_network_expression.find(search, pos)) != string::npos)
    {
        neural_network_expression.replace(pos, search.length(), replace);
        pos += replace.length();
    }

    pos = 0;

    search = "=";
    replace = "<-";

    while((pos = neural_network_expression.find(search, pos)) != string::npos)
    {
        neural_network_expression.replace(pos, search.length(), replace);
        pos += replace.length();
    }

    ostringstream outputs;

    outputs << "outputs <- c(";

    for(Index i = 0; i < outputs_number; i++)
    {
        outputs << outputs_names[i];

        if(i != outputs_number - 1)
            outputs << ", ";
    }

    outputs << ")\n    ";

    buffer << neural_network_expression;

    buffer << outputs.str();

    buffer << "outputs \n} \n";

    string expression = buffer.str();

    pos = 0;

    search = "\"";
    replace = "";

    while((pos = expression.find(search, pos)) != string::npos)
    {
        expression.replace(pos, search.length(), replace);
        pos += replace.length();
    }

    return expression;
}


/// Saves the mathematical expression represented by the neural network to a text file.
/// @param file_name Name of expression text file.

void NeuralNetwork::save_expression(const string& file_name)
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

    file << write_expression();

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


/// Saves the R function of the expression represented by the neural network to a text file.
/// @param file_name Name of expression text file.

void NeuralNetwork::save_expression_R(const string& file_name)
{
    ofstream file(file_name.c_str());

    if(!file.is_open())
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: NeuralNetwork class.\n"
               << "void  save_expression_R(const string&) method.\n"
               << "Cannot open expression text file.\n";

        throw logic_error(buffer.str());
    }

    file << write_expression_R();

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
