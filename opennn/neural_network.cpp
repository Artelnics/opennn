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

NeuralNetwork::NeuralNetwork(const NeuralNetwork::ProjectType& model_type, const Vector<size_t>& architecture)
{
    set(model_type, architecture);
}


/// (Convolutional layer) constructor.
/// It creates a neural network object with the given parameters.
/// Note that this method is only valid when our problem presents convolutional layers.

NeuralNetwork::NeuralNetwork(const Vector<size_t>& new_inputs_dimensions,
                             const size_t& new_blocks_number,
                             const Vector<size_t>& new_filters_dimensions,
                             const size_t& new_outputs_number)
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

NeuralNetwork::NeuralNetwork(const Vector<Layer*>& new_layers_pointers)
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
    const size_t layers_number = get_layers_number();

    for(size_t i = 0;  i < layers_number; i++)
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
    const Layer::LayerType layer_type = layer_pointer->get_type();

    if(check_layer_type(layer_type))
    {
        layers_pointers.push_back(layer_pointer);
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

bool NeuralNetwork::check_layer_type(const Layer::LayerType layer_type)
{
    const size_t layers_number = layers_pointers.size();

    if(layers_number > 1 && (layer_type == Layer::Recurrent || layer_type == Layer::LongShortTermMemory))
    {
        return false;
    }
    else if(layers_number == 1 && (layer_type == Layer::Recurrent || layer_type == Layer::LongShortTermMemory))
    {
        const Layer::LayerType first_layer_type = layers_pointers[0]->get_type();

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
    const size_t layers_number = get_layers_number();

    for(size_t i = 0; i < layers_number; i++)
    {
        if(layers_pointers[i]->get_type() == Layer::Scaling) return true;
    }

    return false;
}


/// Returns true if the neural network object has a principal components layer object inside,
/// and false otherwise.

bool NeuralNetwork::has_principal_components_layer() const
{
    const size_t layers_number = get_layers_number();

    for(size_t i = 0; i < layers_number; i++)
    {
        if(layers_pointers[i]->get_type() == Layer::PrincipalComponents) return true;
    }

    return false;
}


/// Returns true if the neural network object has a long short term memory layer object inside,
/// and false otherwise.

bool NeuralNetwork::has_long_short_term_memory_layer() const
{
    const size_t layers_number = get_layers_number();

    for(size_t i = 0; i < layers_number; i++)
    {
        if(layers_pointers[i]->get_type() == Layer::LongShortTermMemory) return true;
    }

    return false;
}



/// Returns true if the neural network object has a recurrent layer object inside,
/// and false otherwise.

bool NeuralNetwork::has_recurrent_layer() const
{
    const size_t layers_number = get_layers_number();

    for(size_t i = 0; i < layers_number; i++)
    {
        if(layers_pointers[i]->get_type() == Layer::Recurrent) return true;
    }

    return false;
}


/// Returns true if the neural network object has an unscaling layer object inside,
/// and false otherwise.

bool NeuralNetwork::has_unscaling_layer() const
{
    const size_t layers_number = get_layers_number();

    for(size_t i = 0; i < layers_number; i++)
    {
        if(layers_pointers[i]->get_type() == Layer::Unscaling) return true;
    }

    return false;
}


/// Returns true if the neural network object has a bounding layer object inside,
/// and false otherwise.

bool NeuralNetwork::has_bounding_layer() const
{
    const size_t layers_number = get_layers_number();

    for(size_t i = 0; i < layers_number; i++)
    {
        if(layers_pointers[i]->get_type() == Layer::Bounding) return true;
    }

    return false;
}


/// Returns true if the neural network object has a probabilistic layer object inside,
/// and false otherwise.

bool NeuralNetwork::has_probabilistic_layer() const
{
    const size_t layers_number = get_layers_number();

    for(size_t i = 0; i < layers_number; i++)
    {
        if(layers_pointers[i]->get_type() == Layer::Probabilistic) return true;
    }

    return false;
}


/// Returns true if the neural network object is empty,
/// and false otherwise.

bool NeuralNetwork::is_empty() const
{
    return layers_pointers.empty();
}


/// Returns a string vector with the names of the variables used as inputs.

Vector<string> NeuralNetwork::get_inputs_names() const
{
    return inputs_names;
}


/// Returns a string with the name of the variable used as inputs on a certain index.
/// @param index Index of the variable to be examined.

string NeuralNetwork::get_input_name(const size_t& index) const
{
    return inputs_names[index];
}


/// Returns the index of the variable with a given name.
/// @param name Name of the variable to be examined.

size_t NeuralNetwork::get_input_index(const string& name) const
{
    return inputs_names.get_first_index(name);
}


/// Returns a string vector with the names of the variables used as outputs.

Vector<string> NeuralNetwork::get_outputs_names() const
{
    return outputs_names;
}


/// Returns a string with the name of the variable used as outputs on a certain index.
/// @param index Index of the variable to be examined.

string NeuralNetwork::get_output_name(const size_t& index) const
{
    return outputs_names[index];
}


/// Returns the index of the variable with a given name.
/// @param name Name of the variable to be examined.

size_t NeuralNetwork::get_output_index(const string& name) const
{
    return outputs_names.get_first_index(name);
}


/// Returns a pointer to the layers object composing this neural network object.

Vector<Layer*> NeuralNetwork::get_layers_pointers() const
{
    return layers_pointers;
}


/// Returns a pointer to the trainable layers object composing this neural network object.

Vector<Layer*> NeuralNetwork::get_trainable_layers_pointers() const
{
    const size_t layers_number = get_layers_number();

    Vector<Layer*> trainable_layers_pointers;

    for(size_t i = 0; i < layers_number; i++)
    {
        if(layers_pointers[i]->get_type() != Layer::Scaling
        && layers_pointers[i]->get_type() != Layer::Unscaling
        && layers_pointers[i]->get_type() != Layer::Bounding)
        {
            trainable_layers_pointers.push_back(layers_pointers[i]);
        }
    }

    return trainable_layers_pointers;
}


/// Returns a vector with the indices of the trainable layers.

Vector<size_t> NeuralNetwork::get_trainable_layers_indices() const
{
    const size_t layers_number = get_layers_number();

    Vector<size_t> trainable_layers_indices;

    for(size_t i = 0; i < layers_number; i++)
    {
        if(layers_pointers[i]->get_type() != Layer::Scaling
        && layers_pointers[i]->get_type() != Layer::Unscaling
        && layers_pointers[i]->get_type() != Layer::Bounding)
        {
            trainable_layers_indices.push_back(i);
        }
    }

    return trainable_layers_indices;
}


/// Returns a pointer to the scaling layers object composing this neural network object.

ScalingLayer* NeuralNetwork::get_scaling_layer_pointer() const
{
    const size_t layers_number = get_layers_number();

    for(size_t i = 0; i < layers_number; i++)
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
    const size_t layers_number = get_layers_number();

    for(size_t i = 0; i < layers_number; i++)
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
    const size_t layers_number = get_layers_number();

    for(size_t i = 0; i < layers_number; i++)
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
    const size_t layers_number = get_layers_number();

    for(size_t i = 0; i < layers_number; i++)
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
    const size_t layers_number = get_layers_number();

    for(size_t i = 0; i < layers_number; i++)
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
    const size_t layers_number = get_layers_number();

    for(size_t i = 0; i < layers_number; i++)
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
    const size_t layers_number = get_layers_number();

    for(size_t i = 0; i < layers_number; i++)
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
    inputs_names.set();

    outputs_names.set();

    layers_pointers.set();

    set_default();
}


/// Sets a new neural network with a given neural network architecture.
/// It also sets the rest of members to their default values. 
/// @param architecture Architecture of the neural network.

void NeuralNetwork::set(const NeuralNetwork::ProjectType& model_type, const Vector<size_t>& architecture)
{    
    layers_pointers.set();

    const size_t size = architecture.size();

    const size_t inputs_number = architecture.get_first();
    const size_t outputs_number = architecture.get_last();

    inputs_names.set(inputs_number);

    ScalingLayer* scaling_layer_pointer = new ScalingLayer(inputs_number);

    this->add_layer(scaling_layer_pointer);

    if(model_type == Approximation)
    {
        for(size_t i = 0; i < size-1; i++)
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
        for(size_t i = 0; i < size-2; i++)
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

        for(size_t i = 1; i < size-1; i++)
        {
            PerceptronLayer* perceptron_layer_pointer = new PerceptronLayer(architecture[i], architecture[i+1]);

            this->add_layer(perceptron_layer_pointer);
        }

        UnscalingLayer* unscaling_layer_pointer = new UnscalingLayer(architecture[size-1]);

        this->add_layer(unscaling_layer_pointer);
    }

    outputs_names.set(outputs_number);

    set_default();
}


void NeuralNetwork::set(const Vector<size_t>& inputs_dimensions,
                        const size_t& blocks_number,
                        const Vector<size_t>& filters_dimensions,
                        const size_t& outputs_number)
{
    layers_pointers.set();

    ScalingLayer* scaling_layer = new ScalingLayer(inputs_dimensions);
    this->add_layer(scaling_layer);

    Vector<size_t> outputs_dimensions = scaling_layer->get_outputs_dimensions();

    for(size_t i = 0; i < blocks_number; i++)
    {
        ConvolutionalLayer* convolutional_layer = new ConvolutionalLayer(outputs_dimensions, filters_dimensions);
        add_layer(convolutional_layer);

        outputs_dimensions = convolutional_layer->get_outputs_dimensions();

        // Pooling layer 1

        PoolingLayer* pooling_layer_1 = new PoolingLayer(outputs_dimensions);
        add_layer(pooling_layer_1);

        outputs_dimensions = pooling_layer_1->get_outputs_dimensions();
    }

    PerceptronLayer* perceptron_layer = new PerceptronLayer(outputs_dimensions.calculate_sum(), 18);
    add_layer(perceptron_layer);

    const size_t perceptron_layer_outputs = perceptron_layer->get_neurons_number();

    ProbabilisticLayer* probabilistic_layer = new ProbabilisticLayer(perceptron_layer_outputs, outputs_number);
    add_layer(probabilistic_layer);
}


/// Sets the neural network members by loading them from a XML file.
/// @param file_name Neural network XML file_name.

void NeuralNetwork::set(const string& file_name)
{
    layers_pointers.set();

     load(file_name);
}


/// Sets the members of this neural network object with those from other neural network object.
/// @param other_neural_network Neural network object to be copied. 

void NeuralNetwork::set(const NeuralNetwork& other_neural_network)
{
    layers_pointers.set();

    if(this == &other_neural_network) return;

    inputs_names = other_neural_network.inputs_names;

    outputs_names = other_neural_network.outputs_names;

    layers_pointers = other_neural_network.layers_pointers;

    display = other_neural_network.display;
}


void NeuralNetwork::set_inputs_names(const Vector<string>& new_inputs_names)
{
    inputs_names = new_inputs_names;
}


void NeuralNetwork::set_outputs_names(const Vector<string>& new_outputs_names)
{
    outputs_names = new_outputs_names;
}


/// Sets the new inputs number of this neural network object.
/// @param new_inputs_number Number of inputs.

void NeuralNetwork::set_inputs_number(const size_t& new_inputs_number)
{
#ifdef __OPENNN_DEBUG__

    if(new_inputs_number == 0)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: NeuralNetwork class.\n"
               << "void set_inputs_number(const size_t& ) method.\n"
               << "The number of inputs(" << new_inputs_number << ") must be greater than 0.\n";

        throw logic_error(buffer.str());
    }

#endif

    inputs_names.set(new_inputs_number);

    if(has_scaling_layer())
    {
        ScalingLayer* scaling_layer_pointer = get_scaling_layer_pointer();

        scaling_layer_pointer->set_inputs_number(new_inputs_number);
    }

    const size_t trainable_layers_number = get_trainable_layers_number();
    Vector<Layer*> trainable_layers_pointers = get_trainable_layers_pointers();

    if(trainable_layers_number > 0)
    {
        trainable_layers_pointers[0]->set_inputs_number(new_inputs_number);
    }
}


/// Sets the new inputs number of this neural network object.
/// @param inputs Boolean vector containing the number of inputs.

void NeuralNetwork::set_inputs_number(const Vector<bool>& inputs)
{
    if(layers_pointers.empty()) return;

    const size_t new_inputs_number = inputs.count_equal_to(true);

    set_inputs_number(new_inputs_number);
}


/// Sets those members which are not pointer to their default values.

void NeuralNetwork::set_default()
{
    display = true;
}


void NeuralNetwork::set_layers_pointers(Vector<Layer*>& new_layers_pointers)
{
    layers_pointers = new_layers_pointers;
}


PerceptronLayer* NeuralNetwork::get_first_perceptron_layer_pointer() const
{
    const size_t layers_number = get_layers_number();

    for(size_t i = 0; i < layers_number; i++)
    {
        if(layers_pointers[i]->get_type() == Layer::Perceptron)
        {
            return dynamic_cast<PerceptronLayer*>(layers_pointers[i]);
        }
    }

    return nullptr;
}


/// Returns the number of inputs to the neural network.

size_t NeuralNetwork::get_inputs_number() const
{
    if(!layers_pointers.empty())
    {
        return layers_pointers[0]->get_inputs_number();
    }

    return 0;
}


size_t NeuralNetwork::get_outputs_number() const
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

Vector<size_t> NeuralNetwork::get_architecture() const
{
    Vector<size_t> architecture;

    const size_t inputs_number = get_inputs_number();

    if(inputs_number == 0)
    {
        return(architecture);
    }

    architecture.push_back(inputs_number);

    const size_t layers_number = get_layers_number();

    if(layers_number > 0)
    {
        for(size_t i = 0; i < layers_number; i++)
        {
            architecture.push_back(layers_pointers[i]->get_neurons_number());
        }
    }
    return architecture;
}


/// Returns the number of parameters in the neural network
/// The number of parameters is the sum of all the neural network parameters(biases and synaptic weights) and independent parameters.

size_t NeuralNetwork::get_parameters_number() const
{
    const Vector<Layer*> trainable_layers_pointers = get_trainable_layers_pointers();

    size_t parameters_number = 0;

    for(size_t i = 0; i < trainable_layers_pointers.size(); i++)
    {
        parameters_number += trainable_layers_pointers[i]->get_parameters_number();
    }

    return parameters_number;
}


/// Returns the number of parameters in the neural network
/// The number of parameters is the sum of all the neural network trainable parameters(biases and synaptic weights) and independent parameters.

size_t NeuralNetwork::get_trainable_parameters_number() const
{
    const size_t trainable_layers_number = get_trainable_layers_number();

    const Vector<Layer*> trainable_layers_pointers = get_trainable_layers_pointers();

    size_t trainable_parameters_number = 0;

    for(size_t i = 0; i < trainable_layers_number; i++)
    {
        trainable_parameters_number += trainable_layers_pointers[i]->get_parameters_number();
    }

    return trainable_parameters_number;
}


/// Returns the values of the parameters in the neural network as a single vector.
/// This contains all the neural network parameters(biases and synaptic weights) and preprocessed independent parameters.

Vector<double> NeuralNetwork::get_parameters() const
{
    const size_t parameters_number = get_parameters_number();

    Vector<double> parameters(parameters_number);

    const size_t trainable_layers_number = get_trainable_layers_number();

    const Vector<Layer*> trainable_layers_pointers = get_trainable_layers_pointers();

    size_t position = 0;

    for(size_t i = 0; i < trainable_layers_number; i++)
    {
        const Vector<double> layer_parameters = trainable_layers_pointers[i]->get_parameters();

        parameters.embed(position, layer_parameters);

        position += layer_parameters.size();
    }

    return parameters;
}


Vector<size_t> NeuralNetwork::get_trainable_layers_parameters_numbers() const
{
    const size_t trainable_layers_number = get_trainable_layers_number();

    const Vector<Layer*> trainable_layers_pointers = get_trainable_layers_pointers();

    Vector<size_t> trainable_layers_parameters_number(trainable_layers_number);

    for(size_t i = 0; i < trainable_layers_number; i++)
    {
        trainable_layers_parameters_number[i] = trainable_layers_pointers[i]->get_parameters_number();
    }

    return trainable_layers_parameters_number;
}


Vector<Vector<double>> NeuralNetwork::get_trainable_layers_parameters(const Vector<double>& parameters) const
{
    const size_t trainable_layers_number = get_trainable_layers_number();

    const Vector<size_t> trainable_layers_parameters_number = get_trainable_layers_parameters_numbers();

    Vector<Vector<double>> trainable_layers_parameters(trainable_layers_number);

    size_t index = 0;

    for(size_t i = 0; i < trainable_layers_number; i++)
    {
        trainable_layers_parameters[i] = parameters.get_subvector(index, index + trainable_layers_parameters_number[i]-1);

        index += trainable_layers_parameters_number[i];
    }

    return trainable_layers_parameters;
}


/// Sets all the parameters(neural_network_pointer parameters and independent parameters) from a single vector.
/// @param new_parameters New set of parameter values. 

void NeuralNetwork::set_parameters(const Vector<double>& new_parameters)
{
#ifdef __OPENNN_DEBUG__

    const size_t size = new_parameters.size();

    const size_t parameters_number = get_parameters_number();

    if(size != parameters_number)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: NeuralNetwork class.\n"
               << "void set_parameters(const Vector<double>&) method.\n"
               << "Size (" << size << ") must be equal to number of parameters (" << parameters_number << ").\n";

        throw logic_error(buffer.str());
    }

#endif

    const size_t trainable_layers_number = get_trainable_layers_number();

    Vector<Layer*> trainable_layers_pointers = get_trainable_layers_pointers();

    const Vector<size_t> trainable_layers_parameters_number = get_trainable_layers_parameters_numbers();

    size_t position = 0;

    for(size_t i = 0; i < trainable_layers_number; i++)
    {
        if(trainable_layers_pointers[i]->get_type() == Layer::Pooling)
        {
            continue;
        }

        const Vector<double> layer_parameters = new_parameters.get_subvector(position, position+trainable_layers_parameters_number[i]-1);

        trainable_layers_pointers[i]->set_parameters(layer_parameters);

        position += trainable_layers_parameters_number[i];
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

size_t NeuralNetwork::get_layers_number() const
{
    return layers_pointers.size();
}


Vector<size_t> NeuralNetwork::get_layers_neurons_numbers() const
{
    Vector<size_t> layers_neurons_number;

    for(size_t i = 0; i < layers_pointers.size(); i++)
    {
        layers_neurons_number.push_back(layers_pointers[i]->get_neurons_number());
    }
    return layers_neurons_number;
}


size_t NeuralNetwork::get_trainable_layers_number() const
{
    const size_t layers_number = get_layers_number();

    size_t count = 0;

    for(size_t i = 0; i < layers_number; i++)
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

void NeuralNetwork::initialize_parameters(const double& value)
{
    const size_t trainable_layers_number = get_trainable_layers_number();

    const Vector<Layer*> trainable_layers_pointers = get_trainable_layers_pointers();

    for(size_t i = 0; i < trainable_layers_number; i++)
    {
        trainable_layers_pointers[i]->initialize_parameters(value);
    }
}


/// Initializes all the parameters in the newtork(biases and synaptic weiths + independent
/// parameters) at random with values comprised between a given minimum and a given maximum values.
/// @param minimum Minimum initialization value.
/// @param maximum Maximum initialization value.

void NeuralNetwork::randomize_parameters_uniform(const double& minimum, const double& maximum)
{
    const size_t trainable_layers_number = get_trainable_layers_number();

    Vector<Layer*> trainable_layers_pointers = get_trainable_layers_pointers();

    for(size_t i = 0; i < trainable_layers_number; i++)
    {
       trainable_layers_pointers[i]->randomize_parameters_uniform(minimum, maximum);
    }
}


/// Initializes all the parameters in the newtork(biases and synaptic weiths + independent
/// parameters) at random with values chosen from a normal distribution with a given mean and a given standard 
/// deviation.
/// @param mean Mean of normal distribution.
/// @param standard_deviation Standard deviation of normal distribution.

void NeuralNetwork::randomize_parameters_normal(const double& mean, const double& standard_deviation)
{
    const size_t trainable_layers_number = get_trainable_layers_number();

    Vector<Layer*> trainable_layers_pointers = get_trainable_layers_pointers();

    for(size_t i = 0; i < trainable_layers_number; i++)
    {
         trainable_layers_pointers[i]->randomize_parameters_normal(mean, standard_deviation);
    }
}


/// Returns the norm of the vector of parameters.

double NeuralNetwork::calculate_parameters_norm() const
{
    const Vector<double> parameters = get_parameters();

    const double parameters_norm = l2_norm(parameters);

    return(parameters_norm);
}


/// Returns a descriptives structure of the parameters vector.
/// That contains the minimum, maximum, mean and standard deviation values of the parameters.

Descriptives NeuralNetwork::calculate_parameters_descriptives() const
{
    const Vector<double> parameters = get_parameters();

    return descriptives(parameters);
}


/// Returns a histogram structure of the parameters vector.
/// That will be used for looking at the distribution of the parameters.
/// @param bins_number Number of bins in the histogram(10 by default).

Histogram NeuralNetwork::calculate_parameters_histogram(const size_t& bins_number) const
{
    const Vector<double> parameters = get_parameters();

    return histogram(parameters, bins_number);
}


/// Perturbate parameters of the neural network.
/// @param perturbation Maximum distance of perturbation.

void NeuralNetwork::perturbate_parameters(const double& perturbation)
{
#ifdef __OPENNN_DEBUG__

    if(perturbation < 0)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: NeuralNetwork class.\n"
               << "void perturbate_parameters(const double&) method.\n"
               << "Perturbation must be equal or greater than 0.\n";

        throw logic_error(buffer.str());
    }

#endif

    Vector<double> parameters = get_parameters();

    Vector<double> parameters_perturbation(parameters);

    parameters_perturbation.randomize_uniform(-perturbation,perturbation);

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

Tensor<double> NeuralNetwork::calculate_outputs(const Tensor<double>& inputs)
{
#ifdef __OPENNN_DEBUG__

    const size_t inputs_dimensions_number = inputs.get_dimensions_number();

    if(inputs_dimensions_number != 2 && inputs_dimensions_number != 4)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: NeuralNetwork class.\n"
               << "Tensor<double> calculate_outputs(const Tensor<double>&) const method.\n"
               << "Inputs dimensions number (" << inputs_dimensions_number << ") must be 2 or 4.\n";

        throw logic_error(buffer.str());
    }

//    const size_t inputs_number = get_inputs_number();

//    const size_t inputs_dimension = inputs.get_dimension(1);

//    if(inputs_size != inputs_number)
//    {
//        ostringstream buffer;

//        buffer << "OpenNN Exception: NeuralNetwork class.\n"
//               << "Tensor<double> calculate_outputs(const Tensor<double>&) const method.\n"
//               << "Dimension of inputs (" <<  << ") must be equal to number of inputs.\n";

//        throw logic_error(buffer.str());
//    }

#endif

    const size_t layers_number = get_layers_number();

    if(layers_number == 0) return inputs;

    Tensor<double> outputs = layers_pointers[0]->calculate_outputs(inputs);

    for(size_t i = 1; i < layers_number; i++)
    {
        outputs = layers_pointers[i]->calculate_outputs(outputs);
    }

    return outputs;
}


Tensor<double> NeuralNetwork::calculate_trainable_outputs(const Tensor<double>& inputs) const
{
#ifdef __OPENNN_DEBUG__

    ///@todo check for convolutional

//    const size_t inputs_dimensions_number = inputs.get_dimensions_number();

//    if(inputs_dimensions_number != 2)
//    {
//        ostringstream buffer;

//        buffer << "OpenNN Exception: NeuralNetwork class.\n"
//               << "Tensor<double> calculate_trainable_outputs(const Tensor<double>&) const method.\n"
//               << "Inputs dimensions number (" << inputs_dimensions_number << ") must be 2.\n";

//        throw logic_error(buffer.str());
//    }

//    const size_t inputs_number = get_inputs_number();

//    const size_t inputs_columns_number = inputs.get_dimension(1);

//    if(inputs_columns_number != inputs_number)
//    {
//        ostringstream buffer;

//        buffer << "OpenNN Exception: NeuralNetwork class.\n"
//               << "Tensor<double> calculate_outputs(const Tensor<double>&) const method.\n"
//               << "Number of columns (" << inputs_columns_number << ") must be equal to number of inputs (" << inputs_number << ").\n";

//        throw logic_error(buffer.str());
//    }

#endif

    const size_t trainable_layers_number = get_trainable_layers_number();

    const Vector<Layer*> trainable_layers_pointers = get_trainable_layers_pointers();

    Tensor<double> outputs = trainable_layers_pointers[0]->calculate_outputs(inputs);

    for(size_t i = 1; i < trainable_layers_number; i++)
    {
        outputs = trainable_layers_pointers[i]->calculate_outputs(outputs);
    }

    return outputs;
}


Tensor<double> NeuralNetwork::calculate_trainable_outputs(const Tensor<double>& inputs,
                                                          const Vector<double>& parameters) const
{
    const size_t trainable_layers_number = get_trainable_layers_number();

    #ifdef __OPENNN_DEBUG__

        if(trainable_layers_number == 0)
        {
            ostringstream buffer;

            buffer << "OpenNN Exception: NeuralNetwork class.\n"
                   << "Tensor<double> calculate_outputs(const Tensor<double>&, cons Vector<double>&) const method.\n"
                   << "This neural network has not got any layer.\n";

            throw logic_error(buffer.str());
        }

    #endif

    const Vector<Layer*> trainable_layers_pointers = get_trainable_layers_pointers();

    const Vector<Vector<double>> trainable_layers_parameters = get_trainable_layers_parameters(parameters);

    Tensor<double> outputs;

    if(trainable_layers_pointers[0]->get_type() == OpenNN::Layer::LayerType::Pooling)
    {
        outputs = trainable_layers_pointers[0]->calculate_outputs(inputs);
    }
    else outputs = trainable_layers_pointers[0]->calculate_outputs(inputs, trainable_layers_parameters[0]);

    for(size_t i = 1; i < trainable_layers_number; i++)
    {
        if(trainable_layers_pointers[i]->get_type() == OpenNN::Layer::LayerType::Pooling)
        {
            outputs = trainable_layers_pointers[i]->calculate_outputs(outputs);
        }
        else outputs = trainable_layers_pointers[i]->calculate_outputs(outputs, trainable_layers_parameters[i]);
    }

    return outputs;
}


Eigen::MatrixXd NeuralNetwork::calculate_outputs_eigen(const Eigen::MatrixXd& inputs_eigen)
{
    const size_t points_number = static_cast<size_t>(inputs_eigen.rows());
    const size_t inputs_number = get_inputs_number();

    Tensor<double> inputs(points_number, inputs_number);

    Eigen::Map<Eigen::MatrixXd> aux(static_cast<double*>(inputs.data()),
                                    static_cast<Eigen::Index>(points_number),
                                    static_cast<Eigen::Index>(inputs_number));

    aux = inputs_eigen;

    Tensor<double> outputs = calculate_outputs(inputs);

    const Eigen::Map<Eigen::MatrixXd> outputs_eigen(static_cast<double*>(outputs.data()),
                                                    static_cast<Eigen::Index>(points_number),
                                                    static_cast<Eigen::Index>(outputs.get_dimension(1)));

    return outputs_eigen;
}


/// Calculates the input data which is necessary to compute the output data from the neural network in some direction.
/// @param direction Input index(must be between 0 and number of inputs - 1).
/// @param point Input point through the directional input passes.
/// @param minimum Minimum value of the input with the above index.
/// @param maximum Maximum value of the input with the above index.
/// @param points_number Number of points in the directional input data set.

Matrix<double> NeuralNetwork::calculate_directional_inputs(const size_t& direction,
                                                           const Vector<double>& point,
                                                           const double& minimum,
                                                           const double& maximum,
                                                           const size_t& points_number) const
{
    const size_t inputs_number = get_inputs_number();

    Matrix<double> directional_inputs(points_number, inputs_number);

    Vector<double> inputs(inputs_number);

    inputs = point;

    for(size_t i = 0; i < points_number; i++)
    {
        inputs[direction] = minimum + (maximum-minimum)*i/static_cast<double>(points_number-1);

        directional_inputs.set_row(i, inputs);
    }

    return directional_inputs;
}


/// Calculates the histogram of the outputs with random inputs.
/// @param points_number Number of random instances to evaluate the neural network.
/// @param bins_number Number of bins for the histograms.
/// @todo

Vector<Histogram> NeuralNetwork::calculate_outputs_histograms(const size_t& points_number, const size_t& bins_number)
{
    const size_t inputs_number = get_inputs_number();

    Tensor<double> inputs(points_number, inputs_number);
/*
    if(scaling_layer_pointer == nullptr)
    {
    }
    else
    {
        const Vector<ScalingLayer::ScalingMethod> scaling_methods = scaling_layer_pointer->get_scaling_methods();

        for(size_t i = 0; i < scaling_methods.size(); i++)
        {
            Vector<double> input_column(points_number, 0.0);

            if(scaling_methods[i] == ScalingLayer::NoScaling)
            {
                input_column.randomize_uniform();
            }
            else if(scaling_methods[i] == ScalingLayer::MinimumMaximum)
            {
                double minimum = scaling_layer_pointer->get_descriptives(i).minimum;
                double maximum = scaling_layer_pointer->get_descriptives(i).maximum;

                input_column.randomize_uniform(minimum, maximum);
            }
            else if(scaling_methods[i] == ScalingLayer::MeanStandardDeviation)
            {
                double mean = scaling_layer_pointer->get_descriptives(i).mean;
                double standard_deviation = scaling_layer_pointer->get_descriptives(i).standard_deviation;

                input_column.randomize_normal(mean, standard_deviation);
            }
            else if(scaling_methods[i] == ScalingLayer::StandardDeviation)
            {
                double mean = scaling_layer_pointer->get_descriptives(i).mean;
                double standard_deviation = scaling_layer_pointer->get_descriptives(i).standard_deviation;

                input_column.randomize_normal(mean, standard_deviation);
            }

            inputs.set_column(i, input_column, "");
        }
    }
*/
    const Tensor<double> outputs = calculate_outputs(inputs);

    return histograms(outputs.to_matrix(), bins_number);

}


/// Calculates the histogram of the outputs with a matrix of given inputs.
/// @param inputs Matrix of the data to evaluate the neural network.
/// @param bins_number Number of bins for the histograms.

Vector<Histogram> NeuralNetwork::calculate_outputs_histograms(const Tensor<double>& inputs, const size_t& bins_number)
{
    const Tensor<double> outputs = calculate_outputs(inputs);

   return histograms(outputs.to_matrix(), bins_number);
}


/// Returns a string representation of the current neural network object.

string NeuralNetwork::object_to_string() const
{
    ostringstream buffer;

    buffer << "Neural network:\n";

    buffer << "Inputs names:\n";
    buffer << inputs_names << endl;

    // Layers

    const size_t layers_number = get_layers_number();

    buffer << "Layers number: " << layers_number << endl;

    for(size_t i = 0; i < layers_number; i++)
    {
        buffer << "Layer " << i+1 << ":" << endl;

        buffer << layers_pointers[i]->object_to_string() << endl;
    }

    buffer << "Outputs names:\n";
    buffer << outputs_names << endl;

    return buffer.str();
}


///@todo

Matrix<string> NeuralNetwork::get_information() const
{
    return Matrix<string>();
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

/*
/// Serializes the neural network object into a XML document of the TinyXML library without keep the DOM tree in memory.
/// See the OpenNN manual for more information about the format of this document.

void NeuralNetwork::write_XML(tinyxml2::XMLPrinter& file_stream) const
{
    file_stream.OpenElement("NeuralNetwork");

    // Inputs

//    if(inputs_pointer)
//    {
//        inputs_pointer->write_XML(file_stream);
//    }

    // Outputs

//    if(outputs_pointer)
//    {
//        outputs_pointer->write_XML(file_stream);
//    }

    file_stream.CloseElement();
}
*/

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

    for(size_t i = 0; i < inputs_names.size(); i++)
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

    for(size_t i = 0; i < layers_pointers.size(); i++)
    {
        buffer << layers_pointers[i]->get_type_string();
        if(i != (layers_pointers.size()-1)) buffer << " ";
    }

    file_stream.PushText(buffer.str().c_str());

    file_stream.CloseElement();

    // Layers information

    for(size_t i = 0; i < layers_pointers.size(); i++)
    {
        layers_pointers[i]->write_XML(file_stream);
    }

    // Layers (end tag)

    file_stream.CloseElement();

    // Ouputs

    file_stream.OpenElement("Outputs");

    // Outputs number

    const size_t outputs_number = outputs_names.size();

    file_stream.OpenElement("OutputsNumber");

    buffer.str("");
    buffer << outputs_number;

    file_stream.PushText(buffer.str().c_str());

    file_stream.CloseElement();

    // Outputs names

    for(size_t i = 0; i < outputs_number; i++)
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
/*
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
*/

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

    cout << "inputs" << endl;

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
    cout << "outputs" << endl;

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

    size_t new_inputs_number = 0;

    if(inputs_number_element->GetText())
    {
        new_inputs_number = static_cast<size_t>(atoi(inputs_number_element->GetText()));

        set_inputs_number(new_inputs_number);
    }

    // Inputs names

    const tinyxml2::XMLElement* start_element = inputs_number_element;

    if(new_inputs_number > 0)
    {
        for(size_t i = 0; i < new_inputs_number; i++)
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

            inputs_names[i] = input_element->GetText();
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

    Vector<string> layers_types;

    if(layers_types_element->GetText())
    {
        layers_types = get_tokens(layers_types_element->GetText(), ' ');
    }

    // Add layers

    const tinyxml2::XMLElement* start_element = layers_types_element;

    for(size_t i = 0; i < layers_types.size(); i++)
    {
        if(layers_types[i] == "Scaling")
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
        else if(layers_types[i] == "Convolutional")
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

//                convolutional_layer->from_XML(convolutional_document);
            }

            add_layer(convolutional_layer);
        }
        else if(layers_types[i] == "Perceptron")
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
        else if(layers_types[i] == "Pooling")
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

//                pooling_layer->from_XML(pooling_document);
            }

            add_layer(pooling_layer);
        }
        else if(layers_types[i] == "Probabilistic")
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
        else if(layers_types[i] == "LongShortTermMemory")
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

//                long_short_term_memory_layer->from_XML(long_short_term_memory_document);
            }

            add_layer(long_short_term_memory_layer);
        }
        else if(layers_types[i] == "Recurrent")
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

//                recurrent_layer->from_XML(recurrent_document);
            }

            add_layer(recurrent_layer);
        }
        else if(layers_types[i] == "Unscaling")
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
        else if(layers_types[i] == "Bounding")
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
        else if(layers_types[i]== "PrincipalComponents")
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

    size_t new_outputs_number = 0;

    if(outputs_number_element->GetText())
    {
        new_outputs_number = static_cast<size_t>(atoi(outputs_number_element->GetText()));
    }

    // Outputs names

    const tinyxml2::XMLElement* start_element = outputs_number_element;

    if(new_outputs_number > 0)
    {
        outputs_names.resize(new_outputs_number);

        for(size_t i = 0; i < new_outputs_number; i++)
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

            outputs_names[i] = output_element->GetText();
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
    const size_t layers_number = get_layers_number();

    cout << "Layers number: " << layers_number << endl;

    for(size_t i = 0; i < layers_number; i++)
    {
        cout << "Layer " << i+1 << ": " << layers_pointers[i]->get_type_string() << endl;
    }
}


/// Saves to a XML file the members of a neural network object.
/// @param file_name Name of neural network XML file.

void NeuralNetwork::save(const string& file_name) const
{
    FILE* file = fopen(file_name.c_str(), "w");

    tinyxml2::XMLPrinter filestream(file);
    write_XML(filestream);

    fclose(file);
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

    const Vector<double> parameters = get_parameters();

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

    const size_t parameters_number = get_parameters_number();

    Vector<double> new_parameters(parameters_number);

    new_parameters.load(file_name);

    set_parameters(new_parameters);

    file.close();
}


/// Returns a string with the expression of the function represented by the neural network.

string NeuralNetwork::write_expression() const
{
    ostringstream buffer;

    const size_t inputs_number = get_inputs_number();
    const size_t outputs_number = get_outputs_number();

    Vector<string> inputs_names = get_inputs_names();
    Vector<string> outputs_names = get_outputs_names();

    cout << "Inputs names: " << inputs_names << endl;
    cout << "Outputs names: " << outputs_names << endl;

    size_t position = 0;

    string search;
    string replace;

    for(size_t i = 0; i < inputs_number; i++)
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

    for(size_t i = 0; i < outputs_number; i++)
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

    Vector<string> scaled_inputs_name(inputs_names.size());

    for(size_t i = 0; i < inputs_names.size(); i++)
    {
        buffer.str("");

        buffer << "scaled_" << inputs_names[i];

        scaled_inputs_name[i] = buffer.str();
    }

    // Principal components

    Vector<string> principal_components_name(inputs_number);

    for(size_t i = 0; i < inputs_number; i++)
    {
        buffer.str("");

        buffer << "principal_component_" <<(i+1);

        principal_components_name[i] = buffer.str();
    }

    // Scaled outputs

    Vector<string> scaled_outputs_name(outputs_names.size());

    for(size_t i = 0; i < outputs_names.size(); i++)
    {
        buffer.str("");

        buffer << "scaled_" << outputs_names[i];

        scaled_outputs_name[i] = buffer.str();
    }

    // Non probabilistic outputs

    Vector<string> non_probabilistic_outputs_name(outputs_number);

    for(size_t i = 0; i < outputs_number; i++)
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
}


/// Returns a string with the expression of the function represented by the neural network.

string NeuralNetwork::write_mathematical_expression_php() const
{
    ostringstream buffer;

    const size_t inputs_number = get_inputs_number();
    const size_t outputs_number = get_outputs_number();

    Vector<string> inputs_names = get_inputs_names();
    Vector<string> outputs_names = get_outputs_names();

    size_t position = 0;

    string search;
    string replace;

    for(size_t i = 0; i < inputs_number; i++)
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

    for(size_t i = 0; i < outputs_number; i++)
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

    Vector<string> scaled_inputs_name(inputs_names.size());

    for(size_t i = 0; i < inputs_names.size(); i++)
    {
        buffer.str("");

        buffer << "$scaled_" << inputs_names[i];

        scaled_inputs_name[i] = buffer.str();
    }

    // Principal components

    Vector<string> principal_components_name(inputs_number);

    for(size_t i = 0; i < inputs_number; i++)
    {
        buffer.str("");

        buffer << "$principal_component_" <<(i+1);

        principal_components_name[i] = buffer.str();
    }

    // Scaled outputs

    Vector<string> scaled_outputs_name(outputs_number);

    for(size_t i = 0; i < outputs_number; i++)
    {
        buffer.str("");

        buffer << "$scaled_" << outputs_names[i];

        scaled_outputs_name[i] = buffer.str();
    }

    // Non probabilistic outputs

    Vector<string> non_probabilistic_outputs_name(outputs_number);

    for(size_t i = 0; i < outputs_number; i++)
    {
        buffer.str("");

        buffer << "$non_probabilistic_" << outputs_names[i];

        non_probabilistic_outputs_name[i] = buffer.str();
    }

    buffer.str("");

    for(size_t i = 0; i < inputs_names.size(); i++)
    {
        inputs_names[i] = "$"+inputs_names[i];
    }

    for(size_t i = 0; i < outputs_names.size(); i++)
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

    const size_t inputs_number = get_inputs_number();
    const size_t outputs_number = get_outputs_number();

    Vector<string> inputs_names = get_inputs_names();
    Vector<string> outputs_names = get_outputs_names();

    size_t pos;

    string search;
    string replace;

    for(size_t i = 0; i < inputs_number; i++)
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

    for(size_t i = 0; i < outputs_number; i++)
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

    Vector<PerceptronLayer::ActivationFunction> activations;

//    const size_t layers_number = get_layers_number();

//    for(size_t i = 0; i < layers_number; i++)
//        activations.push_back(layers_pointers[i].get_activation_function());

    buffer.str("");

    buffer << "#!/usr/bin/python\n\n";

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

//    if(has_probabilistic_layer())
//    {
//        double decision_threshold = probabilistic_layer_pointer->get_decision_threshold();

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
//            for(size_t i = 0; i < outputs_number; i++)
//            {
//                buffer << "x" << i;

//                if(i != outputs_number - 1)
//                    buffer << ", ";
//            }
//            buffer << ") :\n";

//            buffer << "   inputs = [";
//            for(size_t i = 0; i < outputs_number; i++)
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
//            for(size_t i = 0; i < outputs_number; i++)
//            {
//                buffer << "x" << i;

//                if(i != outputs_number - 1)
//                    buffer << ", ";
//            }
//            buffer << ") :\n";

//            buffer << "   inputs = [";
//            for(size_t i = 0; i < outputs_number; i++)
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

    for(size_t i = 0; i < inputs_number; i++)
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

    for(size_t i = 0; i < outputs_number; i++)
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

    const size_t inputs_number = get_inputs_number();
    const size_t outputs_number = get_outputs_number();

    Vector<string> inputs_names = get_inputs_names();
    Vector<string> outputs_names = get_outputs_names();

    size_t pos;

    string search;
    string replace;

    for(size_t i = 0; i < inputs_number; i++)
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

    for(size_t i = 0; i < outputs_number; i++)
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

    Vector<PerceptronLayer::ActivationFunction> activations;

//    const size_t layers_number = get_layers_number();

//    for(size_t i = 0; i < layers_number; i++)
//        activations.push_back(layers_pointers[i]->get_activation_function());

    buffer.str("");

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

    for(size_t i = 0; i < inputs_names.size(); i++)
    {
        inputs_names[i] = "$_"+inputs_names[i];
    }

    for(size_t i = 0; i < outputs_names.size(); i++)
    {
        outputs_names[i] = "$"+outputs_names[i];
    }

    for(size_t i = 0; i < inputs_number; i++)
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

    for(size_t i = 0; i < outputs_number; i++)
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

    const size_t inputs_number = get_inputs_number();
    const size_t outputs_number = get_outputs_number();

    Vector<string> inputs_names = get_inputs_names();
    Vector<string> outputs_names = get_outputs_names();

    size_t pos = 0;

    string search;
    string replace;

    for(size_t i = 0; i < inputs_number; i++)
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

    for(size_t i = 0; i < outputs_number; i++)
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

    Vector<PerceptronLayer::ActivationFunction> activations;

//    const size_t layers_number = get_layers_number();

//    for(size_t i = 0; i < layers_number; i++)
//	{
//        activations.push_back(layers_pointers[i]->get_activation_function());
//	}

    buffer.str("");

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

    buffer << "expression <- function(inputs) {\n\n    ";

    buffer << "if(length(inputs) != " << inputs_number << ") {\n    "
           << "   print('Incorrect number of inputs')\n    "
           << "   return )\n    "
              "}\n    ";

    for(size_t i = 0; i < inputs_number; i++)
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

    for(size_t i = 0; i < outputs_number; i++)
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
    const size_t inputs_number = get_inputs_number();

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

    const size_t outputs_number = get_outputs_number();

    const size_t variables_number = inputs_number + outputs_number;

//    const Vector<Descriptives> scaling_layer_descriptives = scaling_layer_pointer->get_descriptives();

    const size_t points_number = 101;

    Matrix<double> data(points_number, variables_number);

    Vector<double> inputs(inputs_number);
    Vector<double> outputs(outputs_number);
    Vector<double> row(variables_number);

    Vector<double> increments(inputs_number);

    for(size_t i = 0; i < inputs_number; i++)
    {
//        inputs[i] = scaling_layer_descriptives[i].minimum;
//        increments[i] = (scaling_layer_descriptives[i].maximum - scaling_layer_descriptives[i].minimum)/static_cast<double>(points_number-1.0);
    }

    for(size_t i = 0; i < points_number; i++)
    {
//        outputs = calculate_outputs(inputs.to_column_matrix());

        row = inputs.assemble(outputs);

        data.set_row(i, row);

        inputs += increments;
    }

    data.save_csv(file_name);
}


Vector<Layer::FirstOrderActivations> NeuralNetwork::calculate_trainable_forward_propagation(const Tensor<double>& inputs) const
{
    const size_t trainable_layers_number = get_trainable_layers_number();

    Vector<Layer*> trainable_layers_pointers = get_trainable_layers_pointers();

    Vector<Layer::FirstOrderActivations> forward_propagation(trainable_layers_number);

    // First layer

    forward_propagation[0] = trainable_layers_pointers[0]->calculate_first_order_activations(inputs);

    // Rest of layers

    for(size_t i = 1; i < trainable_layers_number; i++)
    {
        forward_propagation[i] = trainable_layers_pointers[i]->calculate_first_order_activations(forward_propagation[i-1].activations);
    }

    return forward_propagation;
}


Layer* NeuralNetwork::get_output_layer_pointer() const
{
    if(layers_pointers.empty())
        return nullptr;
    else
        return layers_pointers.get_last();
}


Layer* NeuralNetwork::get_layer_pointer(const size_t& index) const
{
    return layers_pointers[index];
}

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2019 Artificial Intelligence Techniques, SL.
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
