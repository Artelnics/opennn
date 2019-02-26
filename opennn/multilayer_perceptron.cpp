/****************************************************************************************************************/
/*                                                                                                              */
/*   OpenNN: Open Neural MultilayerPerceptrons Library                                                          */
/*   www.opennn.net                                                                                             */
/*                                                                                                              */
/*   M U L T I L A Y E R   P E R C E P T R O N   C L A S S                                                      */
/*                                                                                                              */
/*   Artificial Intelligence Techniques SL                                                                      */
/*   artelnics@artelnics.com                                                                                    */
/*                                                                                                              */
/****************************************************************************************************************/

// OpenNN includes

#include "multilayer_perceptron.h"

#ifdef __OPENNN_CUDA__
#include <cuda_runtime.h>
#include <cublas_v2.h>

void initCUDA();

int mallocCUDA(double** A_d, int nBytes);
int memcpyCUDA(double* A_d, const double* A_h, int nBytes);
int getHostVector(const double* A_d, double* A_h, int nBytes);
void freeCUDA(double* A_d);

void updateParametersCUDA(std::vector<double*> weights_d, const std::vector<size_t> weights_rows_numbers, const std::vector<size_t> weights_columns_numbers,
                          std::vector<double*> biases_d, const std::vector<size_t> bias_rows_numbers,
                          const double* gradient_h, const size_t parameters_number);

void updateParametersSgdCUDA(std::vector<double*> weights_d, const std::vector<size_t> weights_rows_numbers, const std::vector<size_t> weights_columns_numbers,
                             std::vector<double*> biases_d, const std::vector<size_t> bias_rows_numbers,
                             const double* gradient_d, const size_t parameters_number,
                             const double& momentum, const bool& nesterov, const double& initial_learning_rate,
                             const double& initial_decay, const size_t& learning_rate_iteration, double*& last_increment);

#endif

#define numeric_to_string( x ) static_cast< ostringstream & >( \
   ( ostringstream() << dec << x ) ).str()

namespace OpenNN
{

// DEFAULT CONSTRUCTOR

/// Default constructor. 
///	It creates a multilayer perceptron object witout any layer.
/// This constructor also initializes the rest of class members to their default values.

MultilayerPerceptron::MultilayerPerceptron()
{
    set();
}


// LAYERS CONSTRUCTOR

/// Layers constructor. 
/// It creates a multilayer perceptron object with given layers of perceptrons. 
/// This constructor also initializes the rest of class members to their default values.

MultilayerPerceptron::MultilayerPerceptron(const Vector<PerceptronLayer>& new_layers)
{
    set(new_layers);
}


// NETWORK ARCHITECTURE CONSTRUCTOR

/// Architecture constructor. 
/// It creates a multilayer perceptron object with an arbitrary deep learning architecture.
/// The architecture is represented by a vector of integers.
/// The first element is the number of inputs.
/// The rest of elements are the number of perceptrons in the subsequent layers. 
/// The multilayer perceptron parameters are initialized at random. 
/// @param new_architecture Vector of integers representing the architecture of the multilayer perceptron.

MultilayerPerceptron::MultilayerPerceptron(const Vector<size_t>& new_architecture)
{
    set(new_architecture);
}

// NETWORK ARCHITECTURE CONSTRUCTOR

/// Architecture constructor.
/// It creates a multilayer perceptron object with an arbitrary deep learning architecture.
/// The architecture is represented by a vector of integers.
/// The first element is the number of inputs.
/// The rest of elements are the number of perceptrons in the subsequent layers.
/// The multilayer perceptron parameters are initialized at random.
/// @param new_architecture Vector of integers representing the architecture of the multilayer perceptron.

MultilayerPerceptron::MultilayerPerceptron(const Vector<int>& new_architecture)
{
    const size_t architecture_size = new_architecture.size();

    Vector<size_t> new_architecture_size_t(architecture_size);

    for(size_t i = 0; i < architecture_size; i++)
    {
        new_architecture_size_t[i] = static_cast<size_t>(new_architecture[i]);
    }

    set(new_architecture_size_t);
}

// ONE LAYER CONSTRUCTOR

/// One layer constructor. 
/// It creates a one-layer perceptron object. 
/// The multilayer perceptron parameters are initialized at random. 
/// This constructor also initializes the rest of class members to their default values:
/// <ul>
/// <li> PerceptronLayer activation function: Linear.
/// </ul> 
/// @param new_inputs_number Number of inputs in the layer.
/// @param new_neurons_number Number of neurons in the layer.

MultilayerPerceptron::MultilayerPerceptron(const size_t& new_inputs_number, const size_t& new_neurons_number)
{
    set(new_inputs_number, new_neurons_number);
}


// TWO LAYERS CONSTRUCTOR

/// Two layers constructor. 
/// It creates a multilayer perceptron object with a hidden layer of
/// perceptrons and an outputs layer of perceptrons. 
/// The multilayer perceptron parameters are initialized at random. 
/// This constructor also initializes the rest of class members to their default values.
/// @param new_inputs_number Number of inputs in the multilayer perceptron
/// @param new_hidden_neurons_number Number of neurons in the hidden layer of the multilayer perceptron
/// @param new_outputs_number Number of outputs neurons.

MultilayerPerceptron::MultilayerPerceptron(const size_t& new_inputs_number, const size_t& new_hidden_neurons_number, const size_t& new_outputs_number)
{
    set(new_inputs_number, new_hidden_neurons_number, new_outputs_number);

    set_default();
}


// COPY CONSTRUCTOR

/// Copy constructor. 
/// It creates a copy of an existing multilayer perceptron object. 
/// @param other_multilayer_perceptron Multilayer perceptron object to be copied.

MultilayerPerceptron::MultilayerPerceptron(const MultilayerPerceptron& other_multilayer_perceptron)
{
    set(other_multilayer_perceptron);
}


// DESTRUCTOR

/// Destructor.
/// This destructor does not delete any pointer. 

MultilayerPerceptron::~MultilayerPerceptron()
{
}


// ASSIGNMENT OPERATOR

/// Assignment operator. 
/// It assigns to this object the members of an existing multilayer perceptron object.
/// @param other_multilayer_perceptron Multilayer perceptron object to be assigned.

MultilayerPerceptron& MultilayerPerceptron::operator = (const MultilayerPerceptron& other_multilayer_perceptron)
{
    if(this != &other_multilayer_perceptron)
    {
        layers = other_multilayer_perceptron.layers;

        display = other_multilayer_perceptron.display;
    }

    return(*this);
}


// EQUAL TO OPERATOR


/// Equal to operator. 
/// It compares this object with another object of the same class. 
/// It returns true if the members of the two objects have the same values, and false otherwise.
/// @ param other_multilayer_perceptron Multilayer perceptron to be compared with.

bool MultilayerPerceptron::operator == (const MultilayerPerceptron& other_multilayer_perceptron) const
{
    if(layers == other_multilayer_perceptron.layers
            && display == other_multilayer_perceptron.display)
    {
        return(true);
    }
    else
    {
        return(false);
    }
}


// METHODS


/// Returns the layers of the multilayer perceptron 
/// The format is a reference to the vector of vectors of perceptrons.
/// Note that each layer might have a different size.

const Vector<PerceptronLayer>& MultilayerPerceptron::get_layers() const 
{
    return(layers);
}


/// Returns a reference to the vector of perceptrons in a single layer.
/// @param i Index of layer.

const PerceptronLayer& MultilayerPerceptron::get_layer(const size_t& i) const
{
    // Control sentence(if debug)

#ifdef __OPENNN_DEBUG__

    const size_t layers_number = get_layers_number();

    if(i >= layers_number)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: MultilayerPerceptron class.\n"
               << "const PerceptronLayer get_layer(const size_t&) const method.\n"
               << "Index of layer(" << i << ") must be less than number of layers(" << layers_number << ").\n";

        throw logic_error(buffer.str());
    }

#endif

    return(layers[i]);
}


/// Returns a pointer to a given layer of perceptrons. 
/// @param i Index of perceptron layer. 

PerceptronLayer* MultilayerPerceptron::get_layer_pointer(const size_t& i)
{
    // Control sentence(if debug)

#ifdef __OPENNN_DEBUG__

    const size_t layers_number = get_layers_number();

    if(i >= layers_number)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: MultilayerPerceptron class.\n"
               << "PerceptronLayer* get_layer_pointer(const size_t&) const method.\n"
               << "Index of layer(" << i << ") must be less than number of layers(" << layers_number << ").\n";

        throw logic_error(buffer.str());
    }

#endif

    return(&layers[i]);
}


/// Returns the total number of perceptrons in the multilayer perceptron.
/// This is equal to the sum of the perceptrons of all layers. 

size_t MultilayerPerceptron::get_perceptrons_number() const
{
    const Vector<size_t> layers_perceptrons_number = get_layers_perceptrons_numbers();

    return(layers_perceptrons_number.calculate_sum());
}


/// Returns a vector of size the number of layers, where each element is equal to the total number of neurons in the current and all the previous layers. 

Vector<size_t> MultilayerPerceptron::count_cumulative_perceptrons_number() const
{
    const size_t layers_number = get_layers_number();

    Vector<size_t> cumulative_neurons_number(layers_number);

    if(layers_number != 0)
    {
        const Vector<size_t> layers_size = get_layers_perceptrons_numbers();

        cumulative_neurons_number[0] = layers_size[0];

        for(size_t i = 1; i < layers_number; i++)
        {
            cumulative_neurons_number[i] = cumulative_neurons_number[i-1] + layers_size[i];
        }
    }

    return(cumulative_neurons_number);
}


/// Returns a vector of integers with size the number of layers, 
/// where each element contains the number of parameters in the corresponding layer.

Vector<size_t> MultilayerPerceptron::get_layers_parameters_number() const
{
    const size_t layers_number = get_layers_number();

    Vector<size_t> layers_parameters_number(layers_number);

    for(size_t i = 0; i < layers_number; i++)
    {
        layers_parameters_number[i] = layers[i].get_parameters_number();
    }

    return(layers_parameters_number);
}


/// Returns a vector of integers with size the number of layers,
/// where each element contains the total number of parameters in the corresponding and the previous layers.

Vector<size_t> MultilayerPerceptron::count_layers_cumulative_parameters_number() const
{
    const size_t layers_number = get_layers_number();

    Vector<size_t> layers_cumulative_parameters_number(layers_number);

    layers_cumulative_parameters_number[0] = layers[0].get_parameters_number();

    for(size_t i = 1; i < layers_number; i++)
    {
        layers_cumulative_parameters_number[i] = layers_cumulative_parameters_number[i-1] + layers[i].get_parameters_number();
    }

    return(layers_cumulative_parameters_number);
}


/// Returns the bias values from the neurons in all the layers. 
/// The format is a vector of vectors of real values. 
/// The size of this vector is the number of layers.
/// The size of each subvector is the number of neurons in the corresponding layer. 

Vector< Vector<double> > MultilayerPerceptron::get_layers_biases() const
{
    const size_t layers_number = get_layers_number();

    Vector< Vector<double> > layers_biases(layers_number);

    for(size_t i = 0; i < layers_number; i++)
    {
        layers_biases[i] = layers[i].get_biases();
    }

    return(layers_biases);
}


/// Returns the synaptic weight values from the neurons in all the layers. 
/// The format is a vector of matrices of real values. 
/// The size of this vector is the number of layers.
/// The number of rows of each sub_matrix is the number of neurons in the corresponding layer. 
/// The number of columns of each sub_matrix is the number of inputs to the corresponding layer. 

Vector< Matrix<double> > MultilayerPerceptron::get_layers_synaptic_weights() const
{
    const size_t layers_number = get_layers_number();

    Vector< Matrix<double> > layers_synaptic_weights(layers_number);

    for(size_t i = 0; i < layers_number; i++)
    {
        layers_synaptic_weights[i] = layers[i].get_synaptic_weights();
    }

    return(layers_synaptic_weights);
}


/// Returns the neural parameter values(biases and synaptic weights) from the neurons in all 
/// the layers. 
/// The format is a vector of vector of real values.
/// The size of this vector is the number of layers.
/// The number of rows of each sub_matrix is the number of neurons in the corresponding layer. 
/// The number of columns of each sub_matrix is the number of parameters(inputs + 1) to the corresponding layer. 

//Vector< Vector<double> > MultilayerPerceptron::get_layers_parameters() const
//{
//    const size_t layers_number = get_layers_number();

//    Vector< Vector<double> > layers_parameters(layers_number);

//    for(size_t i = 0; i < layers_number; i++)
//    {
//        layers_parameters[i] = layers[i].get_parameters();
//    }

//    return(layers_parameters);
//}


//MultilayerPerceptron::LayersParameters MultilayerPerceptron::get_layers_parameters(const Vector<double>& parameters) const
//{
//    const size_t layers_number = get_layers_number();

//    const Vector< Vector<double> > layers_parameters = get_layers_parameters();

//    LayersParameters layers_parameters_structure(layers_number);

//    size_t layer_parameters_number;
//    size_t inputs_number;
//    size_t perceptrons_number;
//    size_t position = 0;

//    for(size_t i = 0; i < layers_number; i++)
//    {
//        layer_parameters_number = layers[i].get_parameters_number();
//        inputs_number = layers[i].get_inputs_number();
//        perceptrons_number = layers[i].get_perceptrons_number();

//        layers_parameters_structure.synaptic_weights[i] =
//                parameters.get_subvector(position, position + layer_parameters_number - perceptrons_number-1).to_matrix(inputs_number, perceptrons_number);


//        layers_parameters_structure.biases[i] =
//                parameters.get_subvector(position + layer_parameters_number - perceptrons_number, position + layer_parameters_number-1);

//        position += layer_parameters_number;
//    }

//    return layers_parameters_structure;
//}


/// Returns the number of parameters(biases and synaptic weights) in the multilayer perceptron. 

size_t MultilayerPerceptron::get_parameters_number() const
{
    const size_t layers_number = get_layers_number();

    size_t parameters_number = 0;

    for(size_t i = 0; i < layers_number; i++)
    {
        parameters_number += layers[i].get_parameters_number();
    }

    return(parameters_number);
}


/// Returns the values of all the biases and synaptic weights in the multilayer perceptron as a single vector.

Vector<double> MultilayerPerceptron::get_parameters() const
{
    const size_t layers_number = get_layers_number();

    const size_t parameters_number = get_parameters_number();

    Vector<double> parameters(parameters_number);

    size_t position = 0;

    for(size_t i = 0; i < layers_number; i++)
    {
        const Vector<double> layer_parameters = layers[i].get_parameters();
        const size_t layer_parameters_number = layer_parameters.size();

        parameters.tuck_in(position, layer_parameters);
        position += layer_parameters_number;
    }

    return(parameters);
}


/// Returns the statistics of all the biases and synaptic weights in the multilayer perceptron.

Vector<double> MultilayerPerceptron::get_parameters_statistics() const
{
    Vector<double> parameters_statistics(4, 0.0);

    const Vector<double> parameters = get_parameters();

    parameters_statistics[0] = parameters.calculate_minimum();
    parameters_statistics[1] = parameters.calculate_maximum();
    parameters_statistics[2] = parameters.calculate_mean();
    parameters_statistics[3] = parameters.calculate_standard_deviation();

    return(parameters_statistics);
}


/// Returns the number of parameters for each layer in this multilayer perceptron.
/// The format is a vector with size the number of layers.
/// Each element contains the number of parameters(biases and synaptic weights) in the corresponding layer.

Vector<size_t> MultilayerPerceptron::get_layers_parameters_numbers() const
{
    const size_t layers_number = get_layers_number();

    Vector<size_t> layers_parameters_numbers(layers_number);

    for(size_t i = 0; i < layers_number; i++)
    {
        layers_parameters_numbers[i] = layers[i].get_parameters_number();
    }

    return(layers_parameters_numbers);
}


/// Returns the index of the layer at which a perceptron belongs to. 
/// @param neuron_index Index of the neuron. 

size_t MultilayerPerceptron::get_layer_index(const size_t& neuron_index) const
{
    // Control sentence(if debug)

#ifdef __OPENNN_DEBUG__

    const size_t neurons_number = get_perceptrons_number();

    if(neuron_index >= neurons_number)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: MultilayerPerceptron class.\n"
               << "int get_layer_index(const size_t&) const method.\n"
               << "Index of neuron must be less than number of neurons.\n";

        throw logic_error(buffer.str());
    }

#endif

    const Vector<size_t> cumulative_neurons_number = count_cumulative_perceptrons_number();

    const size_t layer_index = cumulative_neurons_number.calculate_cumulative_index(neuron_index);

    return(layer_index);
}


/// Returns the index of a neuron, given the layer it belongs and its position in that layer. 
/// @param layer_index Index of layer. 
/// @param perceptron_position Position on the perceptron in that layer. 

size_t MultilayerPerceptron::get_perceptron_index(const size_t& layer_index, const size_t& perceptron_position) const
{
    if(layer_index == 0)
    {
        return(perceptron_position);
    }
    else
    {
        const Vector<size_t> cumulative_neurons_number = count_cumulative_perceptrons_number();

        return(cumulative_neurons_number[layer_index-1] + perceptron_position);
    }
}


/// Returns the index in the vector of parameters of a bias. 
/// @param layer_index Index of layer.
/// @param perceptron_index Index of perceptron within that layer.

size_t MultilayerPerceptron::get_layer_bias_index(const size_t& layer_index, const size_t& /*perceptron_index*/) const
{  
    // Control sentence(if debug)

#ifdef __OPENNN_DEBUG__

    const size_t layers_number = get_layers_number();

    if(layer_index >= layers_number)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: MultilayerPerceptron class.\n"
               << "size_t get_layer_bias_index(const size_t&, const size_t&) const method.\n"
               << "Index of layer(" << layer_index << ") must be less than number of layers(" << layers_number << ").\n";

        throw logic_error(buffer.str());
    }

//    const size_t layer_perceptrons_number = layers[layer_index].get_perceptrons_number();

//    if(perceptron_index >= layer_perceptrons_number)
//    {
//        ostringstream buffer;

//        buffer << "OpenNN Exception: MultilayerPerceptron class.\n"
//               << "size_t get_layer_bias_index(const size_t&, const size_t&) const method.\n"
//               << "Index of perceptron must be less than number of perceptrons in that layer.\n";

//        throw logic_error(buffer.str());
//    }

#endif

    size_t layer_bias_index = 0;

    // Previous layers

    for(size_t i = 0; i < layer_index; i++)
    {
        layer_bias_index += layers[i].get_parameters_number();
    }

    // Previous layer neurons

//    for(size_t j = 0; j < perceptron_index; j++)
//    {
//        layer_bias_index += layers[layer_index].get_perceptron(j).get_parameters_number();
//    }

    return(layer_bias_index);
}


/// Returns the index in the vector of parameters of a synaptic weight.
/// @param layer_index Index of layer.
/// @param perceptron_index Index of perceptron within that layer.
/// @param input_index Index of inputs within that perceptron.

size_t MultilayerPerceptron::get_layer_synaptic_weight_index
(const size_t& layer_index, const size_t& perceptron_index, const size_t& input_index) const
{
    // Control sentence(if debug)

#ifdef __OPENNN_DEBUG__

    const size_t layers_number = get_layers_number();

    if(layer_index >= layers_number)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: MultilayerPerceptron class.\n"
               << "size_t get_layer_synaptic_weight_index(const size_t&, const size_t&, const size_t&) method.\n"
               << "Index of layer(" << layer_index << ") must be less than number of layers(" << layers_number << ").\n";

        throw logic_error(buffer.str());
    }

    const size_t layer_perceptrons_number = layers[layer_index].get_perceptrons_number();

    if(perceptron_index >= layer_perceptrons_number)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: MultilayerPerceptron class.\n"
               << "size_t get_layer_synaptic_weight_index(const size_t&, const size_t&, const size_t&) method.\n"
               << "Index of perceptron must be less than number of perceptrons in layer.\n";

        throw logic_error(buffer.str());
    }

    const size_t layer_inputs_number = layers[layer_index].get_inputs_number();

    if(input_index >= layer_inputs_number)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: MultilayerPerceptron class.\n"
               << "size_t get_layer_synaptic_weight_index(const size_t&, const size_t&, const size_t&) method.\n"
               << "Index of inputs must be less than number of inputs in perceptron.\n";

        throw logic_error(buffer.str());
    }

#endif

    size_t layer_synaptic_weight_index = 0;

    // Previous layers

    if(layer_index > 0)
    {
        for(size_t i = 0; i < layer_index-1; i++)
        {
//            layer_synaptic_weight_index += layers[layer_index].get_parameters_number();
        }
    }

    // Previous layer neurons

    if(perceptron_index > 0)
    {
        for(size_t i = 0; i < perceptron_index-1; i++)
        {
//            layer_synaptic_weight_index += layers[layer_index].get_perceptron(i).get_parameters_number();
        }
    }

    // Hidden neuron bias

    layer_synaptic_weight_index += 1;

    // Synaptic weight index

    layer_synaptic_weight_index += input_index;

    return(layer_synaptic_weight_index);
}


/// Returns the layer, neuron and parameter indices of a neural parameter.
/// @param parameter_index Index of parameter within the parameters vector. 

Vector<size_t> MultilayerPerceptron::get_parameter_indices(const size_t& parameter_index) const
{
    // Control sentence(if debug)

#ifdef __OPENNN_DEBUG__

    const size_t parameters_number = get_parameters_number();

    if(parameter_index >= parameters_number)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: MultilayerPerceptron class.\n"
               << "Vector<int> get_parameter_indices(const size_t&) const method.\n"
               << "Index of neural parameter must be less than number of multilayer perceptron parameters.\n";

        throw logic_error(buffer.str());
    }

#endif

    return(get_parameters_indices().get_row(parameter_index));
}


/// Returns a matrix with the indices of the multilayer perceptron parameters.
/// That indices include the layer index, the neuron index and the parameter index. 
/// The number of rows is the number of multilayer perceptron parameters.
/// The number of columns is 3. 

Matrix<size_t> MultilayerPerceptron::get_parameters_indices() const
{
    size_t perceptron_parameters_number;

    const size_t layers_number = get_layers_number();
    const Vector<size_t> layers_size = get_layers_perceptrons_numbers();

    const size_t parameters_number = get_parameters_number();

    Matrix<size_t> parameters_indices(parameters_number, 3);

    size_t parameter_index = 0;

    for(size_t i = 0; i < layers_number; i++)
    {
        for(size_t j = 0; j < layers_size[i]; j++)
        {
            perceptron_parameters_number = layers[i].get_inputs_number() + 1;

            for(size_t k = 0; k < perceptron_parameters_number; k++)
            {
                parameters_indices(parameter_index,0) = i;
                parameters_indices(parameter_index,1) = j;
                parameters_indices(parameter_index,2) = k;
                parameter_index++;
            }
        }
    }

    return(parameters_indices);
}


/// Returns the activation function of every layer in a single vector. 

Vector<PerceptronLayer::ActivationFunction> MultilayerPerceptron::get_layers_activation_function() const
{
    const size_t layers_number = get_layers_number();

    Vector<PerceptronLayer::ActivationFunction> layers_activation_function(layers_number);

    for(size_t i = 0; i < layers_number; i++)
    {
        layers_activation_function[i] = layers[i].get_activation_function();
    }

    return(layers_activation_function);
}


/// Returns a vector of strings with the name of the activation functions for the layers. 
/// The size of this vector is the number of layers. 

Vector<string> MultilayerPerceptron::write_layers_activation_function() const
{
    const size_t layers_number = get_layers_number();

    Vector<string> layers_activation_function(layers_number);

    for(size_t i = 0; i < layers_number; i++)
    {
        layers_activation_function[i] = layers[i].write_activation_function();
    }

    return(layers_activation_function);
}


/// Returns true if messages from this class are to be displayed on the screen, or false if messages 
/// from this class are not to be displayed on the screen.

const bool& MultilayerPerceptron::get_display() const
{
    return(display);
}


/// Sets those members not related to the multilayer perceptron architecture to their default values:
/// <ul>
/// <li> First perceptron layers activation function: Hyperbolic tangent.
/// <li> Last perceptron layer activation function: Linear.
/// <li> Display: True.
/// </ul>

void MultilayerPerceptron::set_default()
{
    // Multilayer perceptron architecture

    const size_t layers_number = get_layers_number();

    if(layers_number > 0)
    {
        for(size_t i = 0; i < layers_number-1; i++)
        {
            layers[i].set_activation_function(PerceptronLayer::HyperbolicTangent);
        }

        layers[layers_number-1].set_activation_function(PerceptronLayer::Linear);
    }

    // Display messages

    set_display(true);
}


/// Sets an empty multilayer_perceptron_pointer architecture. 

void MultilayerPerceptron::set()
{
    layers.set();
}


/// Sets a multilayer_perceptron_pointer architecture with given layers of perceptrons. 
/// @param new_layers Vector of vectors of perceptrons, which represent the multilayer perceptron architecture.

void MultilayerPerceptron::set(const Vector<PerceptronLayer>& new_layers)
{
    layers = new_layers;
}


/// Sets a deep learning architecture for the multilayer perceptron.
/// The architecture is represented as a vector of integers.
/// The number of layers is the size of that vector minus one. 
/// The first element in the vector represents the number of inputs. 
/// The rest of elements represent the corresponding number of perceptrons in each layer. 
/// All the parameters of the multilayer perceptron are initialized at random.
/// @param new_architecture Architecture of the multilayer perceptron.

void MultilayerPerceptron::set(const Vector<size_t>& new_architecture)
{
    ostringstream buffer;

    const size_t new_architecture_size = new_architecture.size();

    if(new_architecture_size == 0)
    {
        set();
    }
    else if(new_architecture_size == 1)
    {
        buffer << "OpenNN Exception: MultilayerPerceptron class.\n"
               << "void set_architecture(const Vector<size_t>&) method.\n"
               << "Size of architecture cannot be one.\n";

        throw logic_error(buffer.str());
    }
    else
    {
        for(size_t i = 0; i < new_architecture_size; i++)
        {
            if(new_architecture[i] == 0)
            {
                buffer << "OpenNN Exception: MultilayerPerceptron class.\n"
                       << "void set_architecture(const Vector<size_t>&) method.\n"
                       << "Size " << i << " must be greater than zero.\n";

                throw logic_error(buffer.str());
            }
        }

        const size_t new_layers_number = new_architecture_size-1;
        layers.set(new_layers_number);

        // First layer

        for(size_t i = 0; i < new_layers_number; i++)
        {
            layers[i].set(new_architecture[i], new_architecture[i+1]);
        }

        // Activation

        for(size_t i = 0; i < new_layers_number-1; i++)
        {
            layers[i].set_activation_function(PerceptronLayer::HyperbolicTangent);
        }

        layers[new_layers_number-1].set_activation_function(PerceptronLayer::Linear);
    }
}   


/// Sets a deep learning architecture for the multilayer perceptron.
/// The architecture is represented as a vector of integers.
/// The number of layers is the size of that vector minus one.
/// The first element in the vector represents the number of inputs.
/// The rest of elements represent the corresponding number of perceptrons in each layer.
/// All the parameters of the multilayer perceptron are initialized at random.
/// @param new_architecture Architecture of the multilayer perceptron.

void MultilayerPerceptron::set(const Vector<int>& new_architecture)
{
    ostringstream buffer;

    const size_t new_architecture_size = new_architecture.size();

    if(new_architecture_size == 0)
    {
        set();
    }
    else if(new_architecture_size == 1)
    {
        buffer << "OpenNN Exception: MultilayerPerceptron class.\n"
               << "void set_architecture(const Vector<size_t>&) method.\n"
               << "Size of architecture cannot be one.\n";

        throw logic_error(buffer.str());
    }
    else
    {
        for(size_t i = 0; i < new_architecture_size; i++)
        {
            if(new_architecture[i] == 0)
            {
                buffer << "OpenNN Exception: MultilayerPerceptron class.\n"
                       << "void set_architecture(const Vector<size_t>&) method.\n"
                       << "Size " << i << " must be greater than zero.\n";

                throw logic_error(buffer.str());
            }
        }

        const size_t new_layers_number = new_architecture_size-1;
        layers.set(new_layers_number);

        // First layer

        for(size_t i = 0; i < new_layers_number; i++)
        {
            layers[i].set(static_cast<size_t>(new_architecture[i]), static_cast<size_t>(new_architecture[i+1]));
        }

        // Activation

        for(size_t i = 0; i < new_layers_number-1; i++)
        {
            layers[i].set_activation_function(PerceptronLayer::HyperbolicTangent);
        }

        layers[new_layers_number-1].set_activation_function(PerceptronLayer::Linear);
    }
}


/// Sets a new architecture with just one layer. 
/// @param new_inputs_number Number of inputs to the multilayer perceptron.
/// @param new_perceptrons_number Number of perceptrons in the unique layer. This is also the number of outputs.

void MultilayerPerceptron::set(const size_t& new_inputs_number, const size_t& new_perceptrons_number)
{
    // Control sentence(if debug)

#ifdef __OPENNN_DEBUG__

    ostringstream buffer;

    if(new_inputs_number == 0)
    {
        buffer << "OpenNN Exception: MultilayerPerceptron class.\n"
               << "void set(const size_t&, const size_t&) method.\n"
               << "Number of inputs cannot be zero.\n";

        throw logic_error(buffer.str());
    }
    else if(new_perceptrons_number == 0)
    {
        buffer << "OpenNN Exception: MultilayerPerceptron class.\n"
               << "void set_architecture(const size_t&, const size_t&) method.\n"
               << "Number of perceptrons cannot be zero.\n";

        throw logic_error(buffer.str());
    }

#endif

    layers.set(1);

    layers[0].set(new_inputs_number, new_perceptrons_number);
}


/// Sets a new multilayer_perceptron_pointer architecture with two layers, a hidden layer and an outputs layer. 
/// @param new_inputs_number Number of inputs to the multilayer perceptron.
/// @param new_hidden_neurons_number Number of neurons in the hidden layer. 
/// @param new_outputs_number Number of outputs from the multilayer perceptron.

void MultilayerPerceptron::set(const size_t& new_inputs_number, const size_t& new_hidden_neurons_number, const size_t& new_outputs_number)
{
    // Control sentence(if debug)

#ifdef __OPENNN_DEBUG__

    ostringstream buffer;

    if(new_inputs_number == 0)
    {
        buffer << "OpenNN Exception: MultilayerPerceptron class.\n"
               << "void set(const size_t&, const size_t&, const size_t&) method.\n"
               << "Number of inputs must be greater than zero.\n";

        throw logic_error(buffer.str());
    }
    else if(new_hidden_neurons_number == 0)
    {
        buffer << "OpenNN Exception: MultilayerPerceptron class.\n"
               << "void set(const size_t&, const size_t&, const size_t&) method.\n"
               << "Number of hidden neurons must be greater than zero.\n";

        throw logic_error(buffer.str());
    }
    else if(new_outputs_number == 0)
    {
        buffer << "OpenNN Exception: MultilayerPerceptron class.\n"
               << "void set(const size_t&, const size_t&, const size_t&) method.\n"
               << "Number of outputs must be greater than zero.\n";

        throw logic_error(buffer.str());
    }

#endif

    layers.set(2);

    layers[0].set(new_inputs_number, new_hidden_neurons_number);
    layers[0].set_activation_function(PerceptronLayer::HyperbolicTangent);

    layers[1].set(new_hidden_neurons_number, new_outputs_number);
    layers[1].set_activation_function(PerceptronLayer::Linear);
}


/// Sets the members of this object to be the members of another object of the same class. 
/// @param other_multilayer_perceptron Object to be copied. 

void MultilayerPerceptron::set(const MultilayerPerceptron& other_multilayer_perceptron)
{
    layers = other_multilayer_perceptron.layers;

    display = other_multilayer_perceptron.display;
}


/// This method set a new number of inputs in the multilayer perceptron. 
/// @param new_inputs_number Number of inputs. 

void MultilayerPerceptron::set_inputs_number(const size_t& new_inputs_number)
{
    if(is_empty())
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: MultilayerPerceptron class.\n"
               << "void set_inputs_number(const size_t&) method.\n"
               << "Multilayer perceptron is empty.\n";

        throw logic_error(buffer.str());
    }

    layers[0].set_inputs_number(new_inputs_number);
}


/// Sets the size of the layers of the multilayer perceptron.
/// It neither modifies the number of inputs nor the number of outputs. 
/// @param new_layers_size New numbers of neurons for the layers of the multilayer perceptron
/// The number of elements of this vector is the number of layers. 

void MultilayerPerceptron::set_layers_perceptrons_number(const Vector<size_t>& new_layers_size)
{
    const Vector<size_t> inputs_number(1, get_inputs_number());

    set(inputs_number.assemble(new_layers_size));
}


/// Sets the size of the layer of the multilayer perceptron when this is unique. 
/// All the parameters of the multilayer perceptron are initialized at random.
/// @param layer_index Index of layer.
/// @param new_layer_perceptrons_number New numbers of neurons for that layer of the multilayer perceptron

void MultilayerPerceptron::set_layer_perceptrons_number(const size_t& layer_index, const size_t& new_layer_perceptrons_number)
{
    const size_t layer_inputs_number = layers[layer_index].get_inputs_number();

    layers[layer_index].set_perceptrons_number(new_layer_perceptrons_number);

    layers[layer_index].set_inputs_number(layer_inputs_number);

    const size_t layers_number = get_layers_number();

    if(layer_index < layers_number-1)
    {
        layers[layer_index+1].set_inputs_number(new_layer_perceptrons_number);
    }
}


/// Sets a new multilayer_perceptron_pointer architecture by means of a pack of layers of perceptrons. 
/// @param new_layers Vector o vectors of perceptron neurons representing a multilayer_perceptron_pointer architecture getd in layers.

void MultilayerPerceptron::set_layers(const Vector<PerceptronLayer>& new_layers)
{
    layers = new_layers;
}


/// Sets all the biases of the layers in the multilayer perceptron
/// The format is a vector of vectors of real numbers. 
/// The size of this vector is the number of layers.
/// The size of each subvector is the number of neurons in the corresponding layer. 
/// @param new_layers_biases New set of biases in the layers. 

void MultilayerPerceptron::set_layers_biases(const Vector< Vector<double> >& new_layers_biases)
{
    const size_t layers_number = get_layers_number();

    // Control sentence(if debug)

#ifdef __OPENNN_DEBUG__

    const size_t size = new_layers_biases.size();

    if(size != layers_number)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: MultilayerPerceptron class.\n"
               << "void set_layers_biases(const Vector< Vector<double> >&) method.\n"
               << "Size(" << size << ") must be equal to number of layers(" << layers_number << ").\n";

        throw logic_error(buffer.str());
    }

#endif

    const Vector<size_t> layers_size = get_layers_perceptrons_numbers();

    for(size_t i = 0; i < layers_number; i++)
    {
        layers[i].set_biases(new_layers_biases[i]);
    }
}


/// Sets all the synaptic weights of the layers in the multilayer perceptron
/// The format is a vector of matrices of real numbers. 
/// The size of this vector is the number of layers.
/// The number of rows of each sub_matrix is the number of neurons in the corresponding layer. 
/// The number of columns of each sub_matrix is the number of inputs to the corresponding layer. 
/// @param new_layers_synaptic_weights New set of synaptic weights in the layers. 

void MultilayerPerceptron::set_layers_synaptic_weights(const Vector< Matrix<double> >& new_layers_synaptic_weights)
{
    const size_t layers_number = get_layers_number();

    // Control sentence(if debug)

#ifdef __OPENNN_DEBUG__

    const size_t size = new_layers_synaptic_weights.size();

    if(size != layers_number)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: MultilayerPerceptron class.\n"
               << "void set_layers_synaptic_weights(const Vector< Matrix<double> >&) method.\n"
               << "Size must be equal to number of layers.\n";

        throw logic_error(buffer.str());
    }

#endif

    for(size_t i = 0; i < layers_number; i++)
    {
        layers[i].set_synaptic_weights(new_layers_synaptic_weights[i]);
    }

}


/// Sets the parameters of a single layer.
/// @param i Index of layer.
/// @param new_layer_parameters Parameters of corresponding layer.

void MultilayerPerceptron::set_layer_parameters(const size_t i, const Vector<double>& new_layer_parameters)
{
    layers[i].set_parameters(new_layer_parameters);
}


/// Sets the multilayer perceptron parameters of all layers.
/// The argument is a vector of vectors of real numbers. 
/// The number of elements is the number of layers. 
/// Each element contains the vector of parameters of a single layer
/// @param new_layers_parameters New vector of layers parameters. 

void MultilayerPerceptron::set_layers_parameters(const Vector< Vector<double> >& new_layers_parameters)
{
    const size_t layers_number = get_layers_number();

    // Control sentence(if debug)

#ifdef __OPENNN_DEBUG__

    const size_t new_layers_parameters_size = new_layers_parameters.size();

    if(new_layers_parameters_size != layers_number)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: MultilayerPerceptron class.\n"
               << "void set_layers_parameters(const Vector< Vector<double> >&) method.\n"
               << "Size of layer parameters must be equal to number of layers.\n";

        throw logic_error(buffer.str());
    }

#endif

    for(size_t i = 0; i < layers_number; i++)
    {
        layers[i].set_parameters(new_layers_parameters[i]);
    }
}


/// Sets all the biases and synaptic weights in the multilayer perceptron from a single vector.
/// @param new_parameters New set of biases and synaptic weights values. 

void MultilayerPerceptron::set_parameters(const Vector<double>& new_parameters)
{

    // Control sentence(if debug)

#ifdef __OPENNN_DEBUG__

    const size_t size = new_parameters.size();

    const size_t parameters_number = get_parameters_number();

    if(size != parameters_number)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: MultilayerPerceptron class.\n"
               << "void set_parameters(const Vector<double>&) method.\n"
               << "Size of parameters (" << size << ") must be equal to number of biases and synaptic weights (" << parameters_number << ").\n";

        throw logic_error(buffer.str());
    }

#endif

    const size_t layers_number = get_layers_number();

    size_t layer_parameters_number;
    Vector<double> layer_parameters;

    size_t position = 0;

    for(size_t i = 0; i < layers_number; i++)
    {
        layer_parameters_number = layers[i].get_parameters_number();

        layer_parameters = new_parameters.get_subvector(position, position + layer_parameters_number - 1);

        layers[i].set_parameters(layer_parameters);

        position += layer_parameters_number;
    }
}


/// This class sets a new activation(or transfer) function in all the layers of the multilayer perceptron 
/// @param new_layers_activation_function Activation function for the layers.
/// The size of this Vector must be equal to the number of layers, and each element corresponds
/// to the activation function of one layer. 

void MultilayerPerceptron::set_layers_activation_function
(const Vector<PerceptronLayer::ActivationFunction>& new_layers_activation_function)
{ 
    const size_t layers_number = get_layers_number();

    // Control sentence(if debug)

#ifdef __OPENNN_DEBUG__

    const size_t new_layers_activation_function_size = new_layers_activation_function.size();

    if(new_layers_activation_function_size != layers_number)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: MultilayerPerceptron class.\n"
               << "void set_layers_activation_function(const Vector<PerceptronLayer::ActivationFunction>&) method.\n"
               << "Size of activation function of layers must be equal to number of layers.\n";

        throw logic_error(buffer.str());
    }

#endif

    for(size_t i = 0; i < layers_number; i++)
    {
        layers[i].set_activation_function(new_layers_activation_function[i]);
    }
}


/// This class sets a new activation(or transfer) function in a single layer of the multilayer perceptron
/// @param i Index of layer.
/// @param new_layer_activation_function Activation function for that layer.

void MultilayerPerceptron::set_layer_activation_function
(const size_t& i, const PerceptronLayer::ActivationFunction& new_layer_activation_function)
{
    // Control sentence(if debug)

#ifdef __OPENNN_DEBUG__

    const size_t layers_number = get_layers_number();

    if(i >= layers_number)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: MultilayerPerceptron class.\n"
               << "void set_layer_activation_function(const size_t&, const Perceptron::ActivationFunction&) method.\n"
               << "Index of layer is equal or greater than number of layers.\n";

        throw logic_error(buffer.str());
    }

#endif

    layers[i].set_activation_function(new_layer_activation_function);
}


/// Sets a new activation(or transfer) function in all the layers. 
/// The argument is a string containing the name of the function("Logistic", "HyperbolicTangent", "Threshold", etc).
/// @param new_layers_activation_function Activation function for the layers. 

void MultilayerPerceptron::set_layers_activation_function(const Vector<string>& new_layers_activation_function)
{
    const size_t layers_number = get_layers_number();

    // Control sentence(if debug)

#ifdef __OPENNN_DEBUG__

    const size_t size = new_layers_activation_function.size();

    if(size != layers_number)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: MultilayerPerceptron class.\n"
               << "void set_layer_activation_function(const Vector<string>&) method.\n"
               << "Size of layers activation function is not equal to number of layers.\n";

        throw logic_error(buffer.str());
    }

#endif


    for(size_t i = 0; i < layers_number; i++)
    {
        layers[i].set_activation_function(new_layers_activation_function[i]);
    }
}


/// Sets a new display value.
/// If it is set to true messages from this class are to be displayed on the screen;
/// if it is set to false messages from this class are not to be displayed on the screen.
/// @param new_display Display value.

void MultilayerPerceptron::set_display(const bool& new_display)
{
    display = new_display;
}


/// Returns true if the number of layers in the multilayer perceptron is zero, and false otherwise. 

bool MultilayerPerceptron::is_empty() const
{
    if(layers.empty())
    {
        return(true);
    }
    else
    {
        return(false);
    }
}


/// Grow an input in this multilayer perceptron object.

void MultilayerPerceptron::grow_input()
{
    const size_t layers_number = get_layers_number();

    if(layers_number > 0)
    {
        layers[0].grow_input();
    }
}


/// Add new perceptrons in a given layer of the multilayer perceptron.
/// @param layer_index Index of the layer to be grown.
/// @param perceptrons_number Number of perceptrons to add to the layer. The default value is 1.

void MultilayerPerceptron::grow_layer_perceptron(const size_t& layer_index, const size_t& perceptrons_number)
{
#ifdef __OPENNN_DEBUG__

    if(layer_index >= (layers.size()-1))
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: MultilayerPerceptron class.\n"
               << "void grow_layer_perceptron(const size_t&, const size_t&) method.\n"
               << "Index of layer is equal or greater than number of layers-1.\n";

        throw logic_error(buffer.str());
    }

    if(perceptrons_number == 0)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: MultilayerPerceptron class.\n"
               << "void grow_layer_perceptron(const size_t&, const size_t&) method.\n"
               << "Number of perceptrons must be greater than 0.\n";

        throw logic_error(buffer.str());
    }

#endif

    for(size_t i = 0; i < perceptrons_number; i++)
    {
        layers[layer_index].grow_perceptron();
        layers[layer_index+1].grow_input();
    }
}


/// Removes a given input to the multilayer perceptron.
/// @param index Index of input to be pruned.

void MultilayerPerceptron::prune_input(const size_t& index)
{
    // Control sentence(if debug)

#ifdef __OPENNN_DEBUG__

    const size_t inputs_number = get_inputs_number();

    if(index >= inputs_number)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: MultilayerPerceptron class.\n"
               << "void prune_input(const size_t&) method.\n"
               << "Index of input is equal or greater than number of inputs.\n";

        throw logic_error(buffer.str());
    }

#endif

    const size_t layers_number = get_layers_number();

    if(layers_number > 0)
    {
        layers[0].prune_input(index);
    }
}


/// Removes a given output neuron from the multilayer perceptron.
/// @param index Index of output to be pruned.

void MultilayerPerceptron::prune_output(const size_t& index)
{
    // Control sentence(if debug)

#ifdef __OPENNN_DEBUG__

    const size_t outputs_number = get_outputs_number();

    if(index >= outputs_number)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: MultilayerPerceptron class.\n"
               << "void prune_output(const size_t&) method.\n"
               << "Index of output is equal or greater than number of outputs.\n";

        throw logic_error(buffer.str());
    }

#endif

    const size_t layers_number = get_layers_number();

    if(layers_number > 0)
    {
        layers[layers_number-1].prune_perceptron(index);
    }
}


/// Removes a perceptron from the multilayer perceptron.
/// @param layer_index Index of the layer where is the perceptron to be removed.
/// @param perceptron_index Index of the perceptron to be pruned.

void MultilayerPerceptron::prune_layer_perceptron(const size_t& layer_index, const size_t& perceptron_index)
{
#ifdef __OPENNN_DEBUG__

    if(layer_index >= (layers.size()-1))
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: MultilayerPerceptron class.\n"
               << "void prune_layer(const size_t&, const size_t&) method.\n"
               << "Index of layer is equal or greater than number of layers-1.\n";

        throw logic_error(buffer.str());
    }

    if(perceptron_index >= layers[layer_index].get_perceptrons_number())
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: MultilayerPerceptron class.\n"
               << "void prune_layer(const size_t&, const size_t&) method.\n"
               << "Index of perceptron is equal or greater than number of perceptrons in the layer.\n";

        throw logic_error(buffer.str());
    }

#endif

    layers[layer_index].prune_perceptron(perceptron_index);
    layers[layer_index+1].prune_input(perceptron_index);
}


/// Sets a random architecture in the multilayer perceptron.
/// It also sets random activation functions for each layer.
/// This method is useful for testing purposes. 

void MultilayerPerceptron::initialize_random()
{
    const size_t architecture_size = rand()%10 + 2;

    Vector<size_t> architecture(architecture_size);

    for(size_t i = 0; i < architecture_size; i++)
    {
        architecture[i]  = rand()%10 + 1;
    }

    set(architecture);

    const size_t layers_number = get_layers_number();

    // Layers activation function

    for(size_t i = 0; i < layers_number; i++)
    {
        switch(rand()%5)
        {
        case 0:
        {
            layers[i].set_activation_function(PerceptronLayer::Logistic);
        }
            break;

        case 1:
        {
            layers[i].set_activation_function(PerceptronLayer::HyperbolicTangent);
        }
            break;

        case 2:
        {
            layers[i].set_activation_function(PerceptronLayer::Threshold);
        }
            break;

        case 3:
        {
            layers[i].set_activation_function(PerceptronLayer::SymmetricThreshold);
        }
            break;

        case 4:
        {
            layers[i].set_activation_function(PerceptronLayer::Linear);
        }
            break;

        default:
        {
            ostringstream buffer;

            buffer << "OpenNN Exception: MultilayerPerceptron class.\n"
                   << "void initialize_random() method.\n"
                   << "Unknown layer activation function.\n";

            throw logic_error(buffer.str());
        }
        }
    }

    // Display messages

    set_display(true);
}


/// Initializes the biases of all the perceptrons in the multilayer perceptron with a given value. 
/// @param value Biases initialization value. 

void MultilayerPerceptron::initialize_biases(const double& value)
{
    const size_t layers_number = get_layers_number();

    for(size_t i = 0; i < layers_number; i++)
    {
        layers[i].initialize_biases(value);
    }
}



/// Initializes the synaptic weights of all the perceptrons in the multilayer perceptron with a given value. 
/// @param value Synaptic weights initialization value. 

void MultilayerPerceptron::initialize_synaptic_weights(const double& value) 
{
    const size_t layers_number = get_layers_number();
    const Vector<size_t> layers_size = get_layers_perceptrons_numbers();

    for(size_t i = 0; i < layers_number; i++)
    {
        layers[i].initialize_synaptic_weights(value);
    }
}


void MultilayerPerceptron::initialize_synaptic_weights_Glorot()
{
    const size_t layers_number = get_layers_number();

    for(size_t i = 0; i < layers_number; i++)
    {
         double fan_in = static_cast<double>(layers[i].get_synaptic_weights().size());
         double fan_out;

        if(i ==layers_number-1){

         fan_out = 1.0;
        }
        else{

         fan_out = static_cast<double>(layers[i + 1].get_synaptic_weights().size());

        }

        const double limit = sqrt(6.0/(fan_in + fan_out));

        layers[i].initialize_synaptic_weights_Glorot(-limit,limit);
    }
}


/// Initializes all the biases and synaptic weights in the neural newtork with a given value.
/// @param value Multilayer perceptron parameters initialization value. 

void MultilayerPerceptron::initialize_parameters(const double& value)
{
    const size_t parameters_number = get_parameters_number();

    const Vector<double> parameters(parameters_number, value);

    set_parameters(parameters);
}


/// Initializes all the biases and synaptic weights in the neural newtork at random with values comprised 
/// between -1 and +1.

void MultilayerPerceptron::randomize_parameters_uniform()
{
    const size_t parameters_number = get_parameters_number();

    Vector<double> parameters(parameters_number);

    parameters.randomize_uniform();

    set_parameters(parameters);
}


/// Initializes all the biases and synaptic weights in the multilayer perceptron at random with values 
/// comprised between a minimum and a maximum values.
/// @param minimum Minimum initialization value.
/// @param maximum Maximum initialization value.

void MultilayerPerceptron::randomize_parameters_uniform(const double& minimum, const double& maximum)
{
    const size_t parameters_number = get_parameters_number();

    Vector<double> parameters(parameters_number);

    parameters.randomize_uniform(minimum, maximum);

    set_parameters(parameters);
}


/// Initializes all the biases and synaptic weights in the multilayer perceptron at random, with values 
/// comprised between different minimum and maximum numbers for each parameter.
/// @param minimum Vector of minimum initialization values.
/// @param maximum Vector of maximum initialization values.

void MultilayerPerceptron::randomize_parameters_uniform(const Vector<double>& minimum, const Vector<double>& maximum)
{
    const size_t parameters_number = get_parameters_number();

    Vector<double> parameters(parameters_number);

    parameters.randomize_uniform(minimum, maximum);

    set_parameters(parameters);
}


/// Initializes all the biases and synaptic weights in the multilayer perceptron at random, with values 
/// comprised between a different minimum and maximum numbers for each parameter.
/// All minimum are maximum initialization values must be given from a vector of two real vectors.
/// The first element must contain the minimum inizizalization value for each parameter.
/// The second element must contain the maximum inizizalization value for each parameter.
/// @param minimum_maximum Vector of minimum and maximum initialization values.

void MultilayerPerceptron::randomize_parameters_uniform(const Vector< Vector<double> >& minimum_maximum)
{
    const size_t parameters_number = get_parameters_number();

    Vector<double> parameters(parameters_number);

    parameters.randomize_uniform(minimum_maximum[0], minimum_maximum[1]);

    set_parameters(parameters);
}


/// Initializes all the biases and synaptic weights in the newtork with random values chosen from a 
/// normal distribution with mean 0 and standard deviation 1.

void MultilayerPerceptron::randomize_parameters_normal()
{
    const size_t parameters_number = get_parameters_number();

    Vector<double> parameters(parameters_number);

    parameters.randomize_normal();

    set_parameters(parameters);
}


/// Initializes all the biases and synaptic weights in the multilayer perceptron with random random values 
/// chosen from a normal distribution with a given mean and a given standard deviation.
/// @param mean Mean of normal distribution.
/// @param standard_deviation Standard deviation of normal distribution.

void MultilayerPerceptron::randomize_parameters_normal(const double& mean, const double& standard_deviation)
{
    const size_t parameters_number = get_parameters_number();

    Vector<double> parameters(parameters_number);

    parameters.randomize_normal(mean, standard_deviation);

    set_parameters(parameters);
}


/// Initializes all the biases an synaptic weights in the multilayer perceptron with random values chosen 
/// from normal distributions with different mean and standard deviation for each parameter.
/// @param mean Vector of mean values.
/// @param standard_deviation Vector of standard deviation values.

void MultilayerPerceptron::randomize_parameters_normal(const Vector<double>& mean, const Vector<double>& standard_deviation)
{
    const size_t parameters_number = get_parameters_number();

    Vector<double> parameters(parameters_number);

    parameters.randomize_normal(mean, standard_deviation);

    set_parameters(parameters);
}


/// Initializes all the biases and synaptic weights in the multilayer perceptron with random values chosen 
/// from normal distributions with different mean and standard deviation for each parameter.
/// All mean and standard deviation values are given from a vector of two real vectors.
/// The first element must contain the mean value for each parameter.
/// The second element must contain the standard deviation value for each parameter.
/// @param mean_standard_deviation Vector of mean and standard deviation values.

void MultilayerPerceptron::randomize_parameters_normal(const Vector< Vector<double> >& mean_standard_deviation)
{
    const size_t parameters_number = get_parameters_number();

    Vector<double> parameters(parameters_number);

    parameters.randomize_normal(mean_standard_deviation[0], mean_standard_deviation[1]);

    set_parameters(parameters);
}


/// Initializes the parameters at random with values chosen from a normal distribution with mean 0 and standard deviation 1.

void MultilayerPerceptron::initialize_parameters()
{
    randomize_parameters_normal();
}


/// Perturbate parameters of the multilayer perceptron.
/// @param perturbation Maximum distance of perturbation.

void MultilayerPerceptron::perturbate_parameters(const double& perturbation)
{
#ifdef __OPENNN_DEBUG__

    if(perturbation < 0)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: MultilayerPerceptron class.\n"
               << "void perturbate_parameters(const double&) method.\n"
               << "Perturbation must be equal or greater than 0.\n";

        throw logic_error(buffer.str());
    }

#endif

    Vector<double> parameters = get_parameters();

    Vector<double> parameters_perturbation(parameters);

    parameters_perturbation.randomize_uniform(-perturbation,perturbation);

    parameters = parameters + parameters_perturbation;

    set_parameters(parameters);
}


/// Returns the norm of the vector of multilayer perceptron parameters.

double MultilayerPerceptron::calculate_parameters_norm() const 
{
    const Vector<double> parameters = get_parameters();

    const double parameters_norm = parameters.calculate_L2_norm();

    return(parameters_norm);
}


/// Returns the partial derivatives of the outputs from each layer with respect to the inputs to the corresponding layer,
/// for a vector of inputs to the neural netwok.
/// The format of this quantity is a vector of matrices.
/// @param inputs Matrix of inputs to the multilayer perceptron

Vector< Vector< Matrix<double> > > MultilayerPerceptron::calculate_layers_Jacobian(const Matrix<double>& inputs) const
{
    const size_t layers_number = get_layers_number();

    Vector<Matrix<double> > layers_output(layers_number);
    Vector< Vector< Matrix<double> > > layers_Jacobian(layers_number);

    layers_output[0] = layers[0].calculate_outputs(inputs);

    layers_Jacobian[0] = layers[0].calculate_Jacobian(inputs);

    for(size_t i = 1; i < layers_number; i++)
    {
        layers_output[i] = layers[i].calculate_outputs(layers_output[i-1]);
        layers_Jacobian[i] = layers[i].calculate_Jacobian(layers_output[i-1]);
    }

    return(layers_Jacobian);
}


/// Returns the partial derivatives of the outputs from the last layer with respect to the inputs to the first layer.
/// That is, it computes the inputs-outputs partial derivatives of the raw multilayer perceptron.
/// @param inputs Vector of inputs to the first layer of the multilayer perceptron architecture.

Vector< Matrix<double> > MultilayerPerceptron::calculate_Jacobian(const Matrix<double>& inputs) const
{
#ifdef __OPENNN_DEBUG__
    const size_t size = inputs.size();
    const size_t inputs_number = get_inputs_number();
    if(size != inputs_number)
    {
        ostringstream buffer;
        buffer << "OpenNN Exception: MultilayerPerceptron class.\n"
               << "Matrix<double> calculate_Jacobian(const Vector<double>&) const method.\n"
               << "Size must be equal to number of inputs.\n";
        throw logic_error(buffer.str());
    }
#endif

    const size_t points_number = inputs.get_rows_number();
    const size_t layers_number = get_layers_number();

    if(layers_number == 0)
    {
        return Vector<Matrix<double>>();
    }

    Vector< Matrix<double> > Jacobian(points_number);

    const Vector< Vector< Matrix<double> > > layers_Jacobian = calculate_layers_Jacobian(inputs);

    Jacobian = layers_Jacobian[layers_number-1];

    for(size_t i = layers_number-2; i == 0; i--)
    {
        for(size_t j = 0; j < points_number; j++)
        {
            Jacobian[j] = Jacobian[j].dot(layers_Jacobian[i][j]);
        }
    }

    return(Jacobian);

/*
    const Vector< Matrix<double> >  layers_Jacobian = calculate_layers_Jacobian(inputs);
    Matrix<double> Jacobian = layers_Jacobian[layers_number-1];

    for(size_t i = layers_number-2; i == 0; i--)
    {
        Jacobian = Jacobian.dot(layers_Jacobian[i]);
    }
*/
}


Matrix<double> MultilayerPerceptron::calculate_outputs(const Matrix<double>& inputs) const
{
    #ifdef __OPENNN_DEBUG__

    const size_t inputs_number = get_inputs_number();

    const size_t inputs_columns_number = inputs.get_columns_number();

    if(inputs_columns_number != inputs_number)
    {
       ostringstream buffer;

       buffer << "OpenNN Exception: MultilayerPerceptron class.\n"
              << "Matrix<double> calculate_outputs(const Matrix<double>&) const method.\n"
              << "Number of columns of inputs matrix must be equal to number of inputs.\n";

       throw logic_error(buffer.str());
    }

    #endif

    const size_t layers_number = get_layers_number();

    if(layers_number == 0)
    {
        return Matrix<double>();
    }

    Matrix<double> outputs = layers[0].calculate_outputs(inputs);

    for(size_t i = 1; i < layers_number; i++)
    {
        outputs = layers[i].calculate_outputs(outputs);
    }

    return(outputs);
}


Matrix<double> MultilayerPerceptron::calculate_outputs(const Matrix<double>& inputs, const Vector<double>& parameters) const
{
    const size_t layers_number = get_layers_number();

    if(layers_number == 0)
    {
        return Matrix<double>();
    }

    const Vector<size_t> architecture = get_architecture();

    const LayersParameters layers_parameters(architecture, parameters);

    Matrix<double> outputs = layers[0].calculate_outputs(inputs, layers_parameters.biases[0], layers_parameters.synaptic_weights[0]);

    for(size_t i = 1; i < layers_number; i++)
    {
        outputs = layers[i].calculate_outputs(outputs, layers_parameters.biases[i], layers_parameters.synaptic_weights[i]);
    }

    return outputs;
}


MultilayerPerceptron::Pointers MultilayerPerceptron::host_to_device() const
{
    Pointers pointers;

    pointers.layers_number = get_layers_number();
    pointers.architecture = get_architecture();

    pointers.weights_pointers.set(pointers.layers_number);
    pointers.biases_pointers.set(pointers.layers_number);

    pointers.layer_activations.set(pointers.layers_number);

    for(size_t i = 0; i < pointers.layers_number; i++)
    {
        pointers.layer_activations[i] = get_layer(i).write_activation_function();
    }

#ifdef __OPENNN_CUDA__

    int error;

    for(size_t i = 0; i < pointers.layers_number; i++)
    {
        const int weights_num_bytes = pointers.architecture[i]*pointers.architecture[i+1]*sizeof(double);

        error = mallocCUDA(&pointers.weights_pointers[i], weights_num_bytes);
        if(error != 0)
        {
            for(size_t i = 0; i < pointers.layers_number; i++)
            {
                freeCUDA(pointers.weights_pointers[i]);
                freeCUDA(pointers.biases_pointers[i]);
            }

            ostringstream buffer;

            buffer << "OpenNN Exception: MultilayerPerceptron class.\n"
                   << "Pointers host_to_device() const method.\n"
                   << "CUDA memory reserve failed.\n";

            throw logic_error(buffer.str());
        }

        const int biases_num_bytes = pointers.architecture[i+1]*sizeof(double);

        error = mallocCUDA(&pointers.biases_pointers[i], biases_num_bytes);
        if(error != 0)
        {
            for(size_t i = 0; i < pointers.layers_number; i++)
            {
                freeCUDA(pointers.weights_pointers[i]);
                freeCUDA(pointers.biases_pointers[i]);
            }

            ostringstream buffer;

            buffer << "OpenNN Exception: MultilayerPerceptron class.\n"
                   << "Pointers host_to_device() const method.\n"
                   << "CUDA memory reserve failed.\n";

            throw logic_error(buffer.str());
        }

        error = memcpyCUDA(pointers.weights_pointers[i], get_layer(i).get_synaptic_weights().data(), weights_num_bytes);
        error += memcpyCUDA(pointers.biases_pointers[i], get_layer(i).get_biases().data(), biases_num_bytes);

        if(error != 0)
        {
            for(size_t i = 0; i < pointers.layers_number; i++)
            {
                freeCUDA(pointers.weights_pointers[i]);
                freeCUDA(pointers.biases_pointers[i]);
            }

            ostringstream buffer;

            buffer << "OpenNN Exception: MultilayerPerceptron class.\n"
                   << "Pointers host_to_device() const method.\n"
                   << "CUDA memory copy failed.\n";

            throw logic_error(buffer.str());
        }
    }

    pointers.CUDA_initialized = true;

#endif

    return pointers;
}

MultilayerPerceptron::Pointers::~Pointers()
{
#ifdef __OPENNN_CUDA__
    for(size_t i = 0; i < weights_pointers.size(); i++)
    {
        freeCUDA(weights_pointers[i]);
    }

    for(size_t i = 0; i < biases_pointers.size(); i++)
    {
        freeCUDA(biases_pointers[i]);
    }
#endif
}

Vector<double> MultilayerPerceptron::Pointers::get_parameters() const
{
    Vector<double> parameters;

#ifdef __OPENNN_CUDA__
    const size_t layers_number = architecture.size() - 1;

    size_t parameters_number = 0;

    for(size_t i = 0; i < layers_number; i++)
    {
        parameters_number += architecture[i]*architecture[i+1] + architecture[i+1];
    }

    parameters.set(parameters_number);
    double* parameters_data = parameters.data();

    size_t index = 0;

    for(size_t i = 0; i < layers_number; i++)
    {
        const int weights_num_bytes = architecture[i]*architecture[i+1]*sizeof(double);

        getHostVector(weights_pointers[i], parameters_data+index, weights_num_bytes);
        index += architecture[i]*architecture[i+1];

        const int biases_num_bytes = architecture[i+1]*sizeof(double);

        getHostVector(biases_pointers[i], parameters_data+index, biases_num_bytes);
        index += architecture[i+1];
    }
#endif

    return parameters;
}


void MultilayerPerceptron::Pointers::update_parameters(const Vector<double>& increment)
{
#ifdef __OPENNN_CUDA__
    const size_t layers_number = architecture.size() - 1;

    const double* increment_data = increment.data();

    vector<size_t> weights_rows_numbers(layers_number);
    vector<size_t> weights_columns_numbers(layers_number);

    vector<size_t> bias_rows_numbers(layers_number);

    size_t parameters_number = 0;

    for(size_t i = 0; i < layers_number; i++)
    {
        weights_rows_numbers[i] = architecture[i];
        weights_columns_numbers[i] = architecture[i+1];

        bias_rows_numbers[i] = architecture[i+1];

        parameters_number += architecture[i]*architecture[i+1] + architecture[i+1];
    }

    updateParametersCUDA(weights_pointers.to_std_vector(), weights_rows_numbers, weights_columns_numbers,
                         biases_pointers.to_std_vector(), bias_rows_numbers,
                         increment_data, parameters_number);
#endif
}

void MultilayerPerceptron::Pointers::update_parameters_sgd(const Vector<double*>& gradient_d, const double& momentum, const bool& nesterov, const double& initial_learning_rate,
                                                           const double& initial_decay, const size_t& learning_rate_iteration, const Vector<double>& last_increment)
{
#ifdef __OPENNN_CUDA__

#endif
}

/// Returns a vector of matrix, where each row contains the combination values of a layer in response
/// to an inputs to the multilayer perceptron
/// @param inputs Input values to the multilayer perceptron

Vector< Matrix<double> > MultilayerPerceptron::calculate_layers_combinations(const Matrix<double>& inputs) const
{
    // Control sentence(if debug)

#ifdef __OPENNN_DEBUG__

    const size_t inputs_size = inputs.get_columns_number();

    const size_t inputs_number = get_inputs_number();

    if(inputs_size != inputs_number)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: MultilayerPerceptron class.\n"
               << "Vector< Vector<double> > calculate_layers_combination(const Vector<double>&) const method.\n"
               << "Size must be equal to number of inputs.\n";

        throw logic_error(buffer.str());
    }

#endif

    const size_t layers_number = get_layers_number();

    Vector< Matrix<double> > layers_combination(layers_number);

    if(layers_number > 0)
    {
        Vector< Matrix<double> > layers_activation(layers_number);

        layers_combination[0] = layers[0].calculate_combinations(inputs);

        layers_activation[0] = layers[0].calculate_activations(layers_combination[0]);

        for(size_t i = 1; i < layers_number; i++)
        {
            layers_combination[i] = layers[i].calculate_combinations(layers_activation[i-1]);

            layers_activation[i] = layers[i].calculate_activations(layers_combination[i]);
        }
    }

    return(layers_combination);
}


/// Returns the combination parameters gradient of all neurons in the network architecture. 
/// The format is a vector of subvectors of subsubvectors.
/// The size of the vector is the number of layers.
/// The size of each subvector is the number of perceptrons in the layer. 
/// The size of each subsubvector is the number of inputs to the perceptron. 
/// That quantities will be useful for calculating some multilayer perceptron derivatives. 
/// @param layers_inputs Vector of subvectors with the inputs to each layer. 

Vector< Vector< Vector<double> > > MultilayerPerceptron::calculate_perceptrons_combination_parameters_gradient(const Vector< Vector<double> >& /*layers_inputs*/) const
{
    const size_t layers_number = get_layers_number();

    // Control sentence(if debug)

#ifdef __OPENNN_DEBUG__

//    const size_t layers_input_size = layers_inputs.size();

//    if(layers_input_size != layers_number)
//    {
//        ostringstream buffer;

//        buffer << "OpenNN Exception: MultilayerPerceptron class.\n"
//               << "Vector< Vector< Vector<double> > > calculate_perceptrons_combination_parameters_gradient(const Vector< Vector<double> >&) const method.\n"
//               << "Size must be equal to number of layers.\n";

//        throw logic_error(buffer.str());
//    }

#endif

    const Vector<size_t> layers_inputs_number = get_layers_inputs_number();

#ifdef __OPENNN_DEBUG__

//    for(size_t i = 0; i < layers_number; i++)
//    {
//        if(layers_inputs[i].size() != layers_inputs_number[i])
//        {
//            ostringstream buffer;

//            buffer << "OpenNN Exception: MultilayerPerceptron class.\n"
//                   << "Vector< Vector< Vector<double> > > calculate_perceptrons_combination_parameters_gradient(const Vector< Vector<double> >&) const method.\n"
//                   << "Size of inputs to layer " << i << " must be equal to size of that layer.\n";

//            throw logic_error(buffer.str());
//        }
//    }

#endif

    const Vector<size_t> layers_size = get_layers_perceptrons_numbers();

    Vector<double> dummy_layer_parameters;

    Vector< Vector< Vector<double> > > perceptrons_combination_gradient(layers_number);

    for(size_t i = 0; i < layers_number; i++)
    {
        perceptrons_combination_gradient[i].set(layers_size[i]);

        for(size_t j = 0; j < layers_size[i]; j++)
        {
//            const Perceptron& perceptron = layers[i].get_perceptron(j);

//            perceptrons_combination_gradient[i][j] = perceptron.calculate_combination_gradient(layers_inputs[i], dummy_layer_parameters);
        }
    }

    return(perceptrons_combination_gradient);
}


/// Returns a vector of matrix, where each element contains the activation derivatives of a layer in response
/// to an inputs to the multilayer perceptron
/// @param inputs Input values to the multilayer perceptron

Vector< Matrix<double> > MultilayerPerceptron::calculate_layers_activations_derivatives(const Matrix<double>& inputs) const
{
    // Control sentence(if debug)

#ifdef __OPENNN_DEBUG__

    const size_t inputs_size = inputs.get_columns_number();
    const size_t inputs_number = get_inputs_number();

    if(inputs_size != inputs_number)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: MultilayerPerceptron class.\n"
               << "Vector< Matrix<double> > calculate_layers_activation_derivatives(const Vector<double>&) const method.\n"
               << "Number of columns must be equal to number of inputs.\n";

        throw logic_error(buffer.str());
    }

#endif

    const size_t layers_number = get_layers_number();

    Vector< Matrix<double> > layers_combination(layers_number);
    Vector< Matrix<double> > layers_activation(layers_number);
    Vector< Matrix<double> > layers_activation_derivatives(layers_number);

    if(layers_number != 0)
    {
        layers_combination[0] = layers[0].calculate_combinations(inputs);
        layers_activation[0] = layers[0].calculate_activations(layers_combination[0]);
        layers_activation_derivatives[0] = layers[0].calculate_activations_derivatives(layers_combination[0]);

        for(size_t i = 1; i < layers_number; i++)
        {
            layers_combination[i] = layers[i].calculate_combinations(layers_activation[i-1]);
            layers_activation[i] = layers[i].calculate_activations(layers_combination[i]);
            layers_activation_derivatives[i] = layers[i].calculate_activations_derivatives(layers_combination[i]);
        }
    }

    return(layers_activation_derivatives);
}


MultilayerPerceptron::FirstOrderForwardPropagation MultilayerPerceptron::calculate_first_order_forward_propagation(const Matrix<double>& inputs) const
{
    // Control sentence(if debug)

#ifdef __OPENNN_DEBUG__

    const size_t inputs_columns_number = inputs.get_columns_number();

    const size_t inputs_number = get_inputs_number();

    if(inputs_columns_number != inputs_number)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: MultilayerPerceptron class.\n"
               << "Vector< Vector< Vector<double> > > calculate_first_order_forward_propagation(const Vector<double>&) const method.\n"
               << "Size must be equal to number of inputs.\n";

        throw logic_error(buffer.str());
    }

#endif

    const size_t layers_number = get_layers_number();

    FirstOrderForwardPropagation first_order_forward_propagation(layers_number);

    Matrix<double> layer_combinations = layers[0].calculate_combinations(inputs);

    first_order_forward_propagation.layers_activations[0] = layers[0].calculate_activations(layer_combinations);

    first_order_forward_propagation.layers_activation_derivatives[0] = layers[0].calculate_activations_derivatives(layer_combinations);

    for(size_t i = 1; i < layers_number; i++)
    {
        layer_combinations = layers[i].calculate_combinations(first_order_forward_propagation.layers_activations[i-1]);

        first_order_forward_propagation.layers_activations[i] = layers[i].calculate_activations(layer_combinations);

        first_order_forward_propagation.layers_activation_derivatives[i] = layers[i].calculate_activations_derivatives(layer_combinations);
    }

    return(first_order_forward_propagation);
}


/// Returns a string representation of the current multilayer perceptron object. 

string MultilayerPerceptron::object_to_string() const
{
    ostringstream buffer;

    buffer << "MultilayerPerceptron\n"
           << "Architecture: " << get_architecture() << "\n"
           << "Layers activation function: " << write_layers_activation_function() << "\n"
           << "Parameters: " << get_parameters() << "\n";
    //<< "Display: " << display << "\n";

    return(buffer.str());
}


/// Serializes the multilayer perceptron object into a XML document of the TinyXML library. 
/// See the OpenNN manual for more information about the format of this document-> 

tinyxml2::XMLDocument* MultilayerPerceptron::to_XML() const
{
    ostringstream buffer;

    tinyxml2::XMLDocument* document = new tinyxml2::XMLDocument;

    tinyxml2::XMLElement* multilayer_perceptron_element = document->NewElement("MultilayerPerceptron");

    document->InsertFirstChild(multilayer_perceptron_element);

    // Architecture
    {
        tinyxml2::XMLElement* architecture_element = document->NewElement("Architecture");
        multilayer_perceptron_element->LinkEndChild(architecture_element);

        string architecture_string = get_architecture().vector_to_string(' ');

        tinyxml2::XMLText* architecture_text = document->NewText(architecture_string.c_str());
        architecture_element->LinkEndChild(architecture_text);
    }

    // Layers activation function
    {
        tinyxml2::XMLElement* layers_activation_function_element = document->NewElement("LayersActivationFunction");
        multilayer_perceptron_element->LinkEndChild(layers_activation_function_element);

        string layers_activation_function_string = write_layers_activation_function().vector_to_string(' ');

        tinyxml2::XMLText* layers_activation_function_text = document->NewText(layers_activation_function_string.c_str());
        layers_activation_function_element->LinkEndChild(layers_activation_function_text);
    }

    // Parameters
    {
        tinyxml2::XMLElement* parameters_element = document->NewElement("Parameters");
        multilayer_perceptron_element->LinkEndChild(parameters_element);

        const string parameters_string = get_parameters().vector_to_string(' ');

        tinyxml2::XMLText* parameters_text = document->NewText(parameters_string.c_str());
        parameters_element->LinkEndChild(parameters_text);
    }

    // Display
    //   {
    //      tinyxml2::XMLElement* display_element = document->NewElement("Display");
    //      multilayer_perceptron_element->LinkEndChild(display_element);

    //      buffer.str("");
    //      buffer << display;

    //      tinyxml2::XMLText* display_text = document->NewText(buffer.str().c_str());
    //      display_element->LinkEndChild(display_text);
    //   }

    return(document);
}


/// Serializes the multilayer perceptron object into a XML document of the TinyXML library without keep the DOM tree in memory.
/// See the OpenNN manual for more information about the format of this document.

void MultilayerPerceptron::write_XML(tinyxml2::XMLPrinter& file_stream) const
{
    file_stream.OpenElement("MultilayerPerceptron");

    // Architecture

    file_stream.OpenElement("Architecture");

    file_stream.PushText(get_architecture().vector_to_string(' ').c_str());

    file_stream.CloseElement();

    // Layers activation function

    file_stream.OpenElement("LayersActivationFunction");

    file_stream.PushText(write_layers_activation_function().vector_to_string(' ').c_str());

    file_stream.CloseElement();

    // Parameters

    file_stream.OpenElement("Parameters");

    file_stream.PushText(get_parameters().vector_to_string(' ').c_str());

    file_stream.CloseElement();


    file_stream.CloseElement();
}


/// Deserializes a TinyXML document into this multilayer perceptron object.
/// @param document TinyXML document containing the member data.

void MultilayerPerceptron::from_XML(const tinyxml2::XMLDocument& document)
{
    const tinyxml2::XMLElement* root_element = document.FirstChildElement("MultilayerPerceptron");

    if(!root_element)
    {
        return;
    }

    // Architecture
    {
        const tinyxml2::XMLElement* architecture_element = root_element->FirstChildElement("Architecture");

        if(architecture_element)
        {
            const char* architecture_text = architecture_element->GetText();

            if(architecture_text)
            {
                Vector<size_t> new_architecture;
                new_architecture.parse(architecture_text);

                try
                {
                    set(new_architecture);
                }
                catch(const logic_error& e)
                {
                    cerr << e.what() << endl;
                }
            }
        }
    }

    // Layers activation function
    {
        const tinyxml2::XMLElement* layers_activation_function_element = root_element->FirstChildElement("LayersActivationFunction");

        if(layers_activation_function_element)
        {
            const char* layers_activation_function_text = layers_activation_function_element->GetText();

            if(layers_activation_function_text)
            {
                Vector<string> new_layers_activation_function;
                new_layers_activation_function.parse(layers_activation_function_text);

                try
                {
                    set_layers_activation_function(new_layers_activation_function);
                }
                catch(const logic_error& e)
                {
                    cerr << e.what() << endl;
                }
            }
        }
    }

    // Parameters
    {
        const tinyxml2::XMLElement* parameters_element = root_element->FirstChildElement("Parameters");

        if(parameters_element)
        {
            const char* parameters_text = parameters_element->GetText();

            if(parameters_text)
            {
                Vector<double> new_parameters;
                new_parameters.parse(parameters_text);

                try
                {
                    set_parameters(new_parameters);
                }
                catch(const logic_error& e)
                {
                    cerr << e.what() << endl;
                }
            }
        }
    }

    // Display
    {
        const tinyxml2::XMLElement* display_element = root_element->FirstChildElement("Display");

        if(display_element)
        {
            string new_display_string = display_element->GetText();

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


/// Serializes the perceptrons layers into a PMML document

void MultilayerPerceptron::to_PMML(tinyxml2::XMLElement* neural_network) const
{
    tinyxml2::XMLDocument* pmml_document = neural_network->GetDocument();

    PerceptronLayer::ActivationFunction neural_network_activation_function = get_layer(0).get_activation_function();

    bool equal_activation_function_for_all_layers = true;

    const size_t number_of_layers = get_layers_number();

    for(size_t i = 1; i < number_of_layers; i++ )
    {
        if(get_layer(i).get_activation_function() != neural_network_activation_function)
        {
            equal_activation_function_for_all_layers = false;

            break;
        }
    }

    // set any default activationFunction because is "required" by the pmml specification,
    // but is ignored when is defined for each layer
    switch(neural_network_activation_function)
    {
    case PerceptronLayer::Threshold:
        neural_network->SetAttribute("activationFunction","threshold");
        break;

    case PerceptronLayer::SymmetricThreshold:
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: MultilayerPerceptron class.\n"
               << "void to_PMML(tinyxml2::XMLElement*) const method.\n"
               << "Symmetric threshold activaton function is not supported by PMML.\n";

        throw logic_error(buffer.str());
    }

    case PerceptronLayer::Logistic:
        neural_network->SetAttribute("activationFunction","logistic");
        break;

    case PerceptronLayer::HyperbolicTangent:
        neural_network->SetAttribute("activationFunction","tanh");
        break;

    case PerceptronLayer::Linear:
        neural_network->SetAttribute("activationFunction","identity");
        break;

//    default:
//        break;
    }


    for(size_t layers_loop_i = 0; layers_loop_i < number_of_layers; layers_loop_i++)
    {
        // Create layer
        tinyxml2::XMLElement* neural_layer = pmml_document->NewElement("NeuralLayer");
        neural_network->LinkEndChild(neural_layer);

        const PerceptronLayer current_layer = get_layer(layers_loop_i);

        const size_t number_of_neurons = current_layer.get_perceptrons_number();

        neural_layer->SetAttribute("numberOfNeurons",static_cast<unsigned>(number_of_neurons));


        if(!equal_activation_function_for_all_layers)
        {
            switch(current_layer.get_activation_function())
            {
            case PerceptronLayer::Threshold:
                neural_layer->SetAttribute("activationFunction","threshold");
                break;

            case PerceptronLayer::SymmetricThreshold:

                //ostringstream buffer;

                //buffer << "OpenNN Exception: MultilayerPerceptron class.\n"
                //       << "void to_PMML(tinyxml2::XMLElement*) const method.\n"
                //       << "Symmetric threshold activaton function is not supported by PMML.\n";

                //throw logic_error(buffer.str());

                break;

            case PerceptronLayer::Logistic:
                neural_layer->SetAttribute("activationFunction","logistic");
                break;

            case PerceptronLayer::HyperbolicTangent:
                neural_layer->SetAttribute("activationFunction","tanh");
                break;

            case PerceptronLayer::Linear:
                neural_layer->SetAttribute("activationFunction","identity");
                break;

//            default:
//                break;
            }
        }

        // Layer neurons
//        for(size_t neurons_loop_i = 0; neurons_loop_i < number_of_neurons; neurons_loop_i++ )
//        {
//            tinyxml2::XMLElement* neuron = pmml_document->NewElement("Neuron");
//            neural_layer->LinkEndChild(neuron);

//            const Perceptron current_perceptron = current_layer.get_perceptron(neurons_loop_i);

//            stringstream buffer;
//            buffer << setprecision(15) << current_perceptron.get_bias();

//            neuron->SetAttribute("bias",buffer.str().c_str());

//            string neuron_id = number_to_string(layers_loop_i+1);
//            neuron_id.append(",");
//            neuron_id.append(number_to_string(neurons_loop_i));

//            neuron->SetAttribute("id",neuron_id.c_str());

//            // Neuron connections
//            const size_t number_of_connections = current_perceptron.get_inputs_number();

//            for(size_t connections_loop_i = 0; connections_loop_i < number_of_connections; connections_loop_i++)
//            {
//                tinyxml2::XMLElement* con = pmml_document->NewElement("Con");
//                neuron->LinkEndChild(con);

//                string connection_from = number_to_string(layers_loop_i);
//                connection_from.append(",");
//                connection_from.append(number_to_string(connections_loop_i));

//                con->SetAttribute("from",connection_from.c_str());

//                const double connection_weight = current_perceptron.get_synaptic_weight(connections_loop_i);

//                buffer.str(string());
//                buffer << setprecision(15) << connection_weight;

//                con->SetAttribute("weight",buffer.str().c_str());
//            }

            // End neuron connections
//        }
        // End layer neurons

    }
}


/// Serializes the multilayer perceptron object into a PMML document.
/// @param file_stream TinyXML file to append the multilayer perceptron object.
/// @param is_softmax_normalization_method True if softmax normalization is used, false otherwise.

void MultilayerPerceptron::write_PMML(tinyxml2::XMLPrinter& file_stream, bool is_softmax_normalization_method) const
{

    PerceptronLayer::ActivationFunction neural_network_activation_function = get_layer(0).get_activation_function();

    bool equal_activation_function_for_all_layers = true;

    const size_t number_of_layers = get_layers_number();


    for(size_t i = 1; i < number_of_layers; i++ )
    {
        if(get_layer(i).get_activation_function() != neural_network_activation_function)
        {
            equal_activation_function_for_all_layers = false;

            break;
        }
    }


    for(size_t layers_loop_i = 0; layers_loop_i < number_of_layers; layers_loop_i++)
    {
        file_stream.OpenElement("NeuralLayer");

        const PerceptronLayer current_layer = get_layer(layers_loop_i);

        const size_t number_of_neurons = current_layer.get_perceptrons_number();

        file_stream.PushAttribute("numberOfNeurons",static_cast<unsigned>(number_of_neurons));

        if(!equal_activation_function_for_all_layers)
        {
            switch(current_layer.get_activation_function())
            {
            case PerceptronLayer::Threshold:
                file_stream.PushAttribute("activationFunction", "threshold");
                break;

            case PerceptronLayer::SymmetricThreshold:
            {
                ostringstream buffer;

                buffer << "OpenNN Exception: MultilayerPerceptron class.\n"
                       << "void write_PMML(tinyxml2::XMLPrinter&, bool) const method.\n"
                       << "Symmetric threshold activaton function is not supported by PMML.\n";

                throw logic_error(buffer.str());
            }

            case PerceptronLayer::Logistic:
                file_stream.PushAttribute("activationFunction", "logistic");
                break;

            case PerceptronLayer::HyperbolicTangent:
                file_stream.PushAttribute("activationFunction", "tanh");
                break;

            case PerceptronLayer::Linear:
                file_stream.PushAttribute("activationFunction", "identity");
                break;

//            default:
//                break;
            }
        }

        // Put softmax normalizatoin method at output layer(the last) if needed
        if(layers_loop_i == (number_of_layers - 1))
            if(is_softmax_normalization_method)
                file_stream.PushAttribute("normalizationMethod","softmax");

        // Layer neurons
//        for(size_t neurons_loop_i = 0; neurons_loop_i < number_of_neurons; neurons_loop_i++ )
//        {
//            file_stream.OpenElement("Neuron");

//            const Perceptron current_perceptron = current_layer.get_perceptron(neurons_loop_i);

//            stringstream buffer;
//            buffer << setprecision(15) << current_perceptron.get_bias();

//            file_stream.PushAttribute("bias",buffer.str().c_str());

//            string neuron_id = number_to_string(layers_loop_i+1);
//            neuron_id.append(",");
//            neuron_id.append(number_to_string(neurons_loop_i));

//            file_stream.PushAttribute("id",neuron_id.c_str());

//            // Neuron connections
//            const size_t number_of_connections = current_perceptron.get_inputs_number();

//            for(size_t connections_loop_i = 0; connections_loop_i < number_of_connections; connections_loop_i++)
//            {
//                file_stream.OpenElement("Con");

//                string connection_from = number_to_string(layers_loop_i);
//                connection_from.append(",");
//                connection_from.append(number_to_string(connections_loop_i));

//                file_stream.PushAttribute("from",connection_from.c_str());

//                const double connection_weight = current_perceptron.get_synaptic_weight(connections_loop_i);

//                buffer.str(string());
//                buffer << setprecision(15) << connection_weight;

//                file_stream.PushAttribute("weight", buffer.str().c_str());

//                // Close Con
//                file_stream.CloseElement();
//            }

//            // Close Neuron
//            file_stream.CloseElement();
//        }

        // Close NeuralLayer
        file_stream.CloseElement();
    }
}


/// Deserializes a PMML document into this multilayer perceptron object.

void MultilayerPerceptron::from_PMML(const tinyxml2::XMLElement* neural_network)
{
    ostringstream buffer;


    const tinyxml2::XMLAttribute* attribute_activation_function_neural_network = neural_network->FindAttribute("activationFunction");

    if(!attribute_activation_function_neural_network)
    {
        buffer << "OpenNN Exception: MultilayerPerceptron class.\n"
               << "void from_PMML(const tinyxml2::XMLElement*) method.\n"
               << "Attibute \"activationFunction\" in NeuralNetwork element is nullptr.\n";

        throw logic_error(buffer.str());
    }

    string network_activation_function = string(attribute_activation_function_neural_network->Value());


    const tinyxml2::XMLElement* neural_layer = neural_network->FirstChildElement("NeuralLayer");

    const size_t number_of_layers = get_layers_number();

    Vector< Vector<double> > new_layers_biases(number_of_layers);

    Vector< Matrix<double> > new_synaptic_weights(number_of_layers);

    // layers
    for(size_t layer_i = 0; layer_i < number_of_layers; layer_i++)
    {
        const tinyxml2::XMLAttribute* attribute_activation_function_layer = neural_layer->FindAttribute("activationFunction");

        const PerceptronLayer current_layer = get_layer(layer_i);

        string activation_function_value;
        PerceptronLayer::ActivationFunction new_activation_function;

        if(!attribute_activation_function_layer)
        {
            activation_function_value = network_activation_function;
        }
        else
        {
            activation_function_value = string(attribute_activation_function_layer->Value());
        }

        if(activation_function_value == "tanh")
        {
            new_activation_function = PerceptronLayer::HyperbolicTangent;
        }
        else if(activation_function_value == "logistic")
        {
            new_activation_function = PerceptronLayer::Logistic;
        }
        else if(activation_function_value == "identity")
        {
            new_activation_function = PerceptronLayer::Linear;
        }
        else if(activation_function_value == "threshold")
        {
            new_activation_function = PerceptronLayer::Threshold;
        }
        else
        {
            buffer << "OpenNN Exception: MultilayerPerceptron class.\n"
                   << "void from_PMML(const tinyxml2::XMLElement*) method.\n"
                   << "Activation function: " << activation_function_value << " not supported.\n";

            throw logic_error(buffer.str());
        }

        set_layer_activation_function(layer_i, new_activation_function);


        const size_t current_layer_perceptrons_number = current_layer.get_perceptrons_number();

        Vector<double> current_layer_new_biases(current_layer_perceptrons_number);

        // All perceptrons in the same layer have the same number of inputs
        Matrix<double> current_layer_new_synaptic_weights(current_layer_perceptrons_number,current_layer.get_inputs_number());

        const tinyxml2::XMLElement* neuron = neural_layer->FirstChildElement("Neuron");


        // neurons
        for(size_t neuron_i = 0; neuron_i < current_layer_perceptrons_number; neuron_i++)
        {
            const tinyxml2::XMLAttribute* attribute_bias_neuron = neuron->FindAttribute("bias");

            if(!attribute_bias_neuron)
            {
                buffer << "OpenNN Exception: MultilayerPerceptron class.\n"
                       << "void from_PMML(const tinyxml2::XMLElement*) method.\n"
                       << "Attribute \"bias\" in Neuron element is nullptr.\n";

                throw logic_error(buffer.str());
            }

            string neuron_bias_string(attribute_bias_neuron->Value());

            if(neuron_bias_string == "")
            {
                buffer << "OpenNN Exception: MultilayerPerceptron class.\n"
                       << "void from_PMML(const tinyxml2::XMLElement*) method.\n"
                       << "Attribute \"bias\" in Neuron element is empty.\n";

                throw logic_error(buffer.str());
            }

            const double neuron_bias = atof(neuron_bias_string.c_str());

            current_layer_new_biases.at(neuron_i) = neuron_bias;


            const tinyxml2::XMLElement* con = neuron->FirstChildElement("Con");

//            const Perceptron current_perceptron = current_layer.get_perceptron(neuron_i);

//            Vector<double> current_perceptron_connections(current_perceptron.get_inputs_number());

//            // connections
//            for(size_t connection_i = 0; connection_i < current_perceptron.get_inputs_number(); connection_i++)
//            {
//                if(!con)
//                {
//                    buffer << "OpenNN Exception: MultilayerPerceptron class.\n"
//                           << "void from_PMML(const tinyxml2::XMLElement*) method.\n"
//                           << "Con in Neuron element is nullptr.\n";

//                    throw logic_error(buffer.str());
//                }

//                const tinyxml2::XMLAttribute* attribute_weight_connection = con->FindAttribute("weight");

//                if(!attribute_weight_connection)
//                {
//                    buffer << "OpenNN Exception: MultilayerPerceptron class.\n"
//                           << "void from_PMML(const tinyxml2::XMLElement*) method.\n"
//                           << "Attribute \"weight\" in Con in Neuron element is nullptr.\n";

//                    throw logic_error(buffer.str());
//                }

//                const string connection_weight_string(attribute_weight_connection->Value());

//                if(connection_weight_string == "")
//                {
//                    buffer << "OpenNN Exception: MultilayerPerceptron class.\n"
//                           << "void from_PMML(const tinyxml2::XMLElement*) method.\n"
//                           << "Attribute \"weight\" in Con in Neuron element is nullptr.\n";

//                    throw logic_error(buffer.str());
//                }

//                const double connection_weight = atof(connection_weight_string.c_str());

//                current_perceptron_connections.at(connection_i) = connection_weight;

                con = con->NextSiblingElement("Con");
//            }

//            current_layer_new_synaptic_weights.set_row(neuron_i,current_perceptron_connections);

//            neuron = neuron->NextSiblingElement("Neuron");
        }

//        new_synaptic_weights.at(layer_i) = current_layer_new_synaptic_weights;

//        new_layers_biases.at(layer_i) = current_layer_new_biases;

        neural_layer = neural_layer->NextSiblingElement("NeuralLayer");
    }

//    set_layers_biases(new_layers_biases);

//    set_layers_synaptic_weights(new_synaptic_weights);
}


/// Returns a string matrix with certain information about the multilayer perceptron,
/// which includes the number of inputs, the number of perceptrons and the activation function of all the layers.
/// The  number of rows is the number of layers, and the number of columns is three.
/// Each row in the matrix contains the information of a single layer.

Matrix<string> MultilayerPerceptron::write_information() const
{
    ostringstream buffer;

    const size_t layers_number = get_layers_number();

    if(layers_number == 0)
    {
        Matrix<string> information;

        return(information);

    }
    else
    {
        Matrix<string> information(layers_number, 3);

        for(size_t i = 0; i < layers_number; i++)
        {
            // Inputs number

            buffer.str("");
            buffer << layers[i].get_inputs_number();

            information(i,0) = buffer.str();

            // Perceptrons number

            buffer.str("");
            buffer << layers[i].get_perceptrons_number();

            information(i,1) = buffer.str();

            // Activation function

            information(i,2) = layers[i].write_activation_function();
        }

        return(information);

    }
}


/// Returns a string with the expression of the forward propagation process in a multilayer perceptron.
/// @param inputs_name Name of input variables.
/// @param outputs_name Name of output variables.

string MultilayerPerceptron::write_expression(const Vector<string>& inputs_name, const Vector<string>& outputs_name) const
{
    const size_t layers_number = get_layers_number();
    const Vector<size_t> layers_perceptrons_number = get_layers_perceptrons_numbers();

    ostringstream buffer;

    if(layers_number == 0)
    {
    }
    else if(layers_number == 1)
    {
        buffer << layers[0].write_expression(inputs_name, outputs_name) << "\n";
    }
    else
    {
        Vector< Vector<string> > layers_outputs_name(layers_number);

        for(size_t i = 0; i < layers_number; i++)
        {
            layers_outputs_name[i].set(layers_perceptrons_number[i]);

            for(size_t j = 0; j < layers_perceptrons_number[i]; j++)
            {
                ostringstream new_buffer;
                new_buffer << "y_" << i+1 << "_" << j+1;
                layers_outputs_name[i][j] = new_buffer.str();
            }
        }

//        cout << "Layers outputs name: " << layers_outputs_name << endl;
//        cout << "Inputs name: " << inputs_name << endl << endl;


        buffer << layers[0].write_expression(inputs_name, layers_outputs_name[0]);
//        cout << "Perceptrons Number: " << layers[0].get_perceptrons_number() << endl;

        for(size_t i = 1; i < layers_number-1; i++)
        {
            buffer << layers[i].write_expression(layers_outputs_name[i-1], layers_outputs_name[i]);
//            cout << "Perceptrons Number: " << layers[i].get_perceptrons_number() << endl;
        }

        buffer << layers[layers_number-1].write_expression(layers_outputs_name[layers_number-2], outputs_name);
//        cout << "Perceptrons Number: " << layers[layers_number-1].get_perceptrons_number() << endl;
    }

    return(buffer.str());
}


/// Returns a string with the php expression of the forward propagation process in a multilayer perceptron.
/// @param inputs_name Name of input variables.
/// @param outputs_name Name of output variables.

string MultilayerPerceptron::write_expression_php(const Vector<string>& inputs_name, const Vector<string>& outputs_name) const
{
    const size_t layers_number = get_layers_number();
    const Vector<size_t> layers_perceptrons_number = get_layers_perceptrons_numbers();

    ostringstream buffer;

    if(layers_number == 0)
    {
    }
    else if(layers_number == 1)
    {
        buffer << layers[0].write_expression(inputs_name, outputs_name) << "\n";
    }
    else
    {
        Vector< Vector<string> > layers_outputs_name(layers_number);

        for(size_t i = 0; i < layers_number; i++)
        {
            layers_outputs_name[i].set(layers_perceptrons_number[i]);

            for(size_t j = 0; j < layers_perceptrons_number[i]; j++)
            {
                ostringstream new_buffer;
                new_buffer << "$y_" << i+1 << "_" << j+1;
                layers_outputs_name[i][j] = new_buffer.str();
            }
        }

        buffer << layers[0].write_expression(inputs_name, layers_outputs_name[0]);

        for(size_t i = 1; i < layers_number-1; i++)
        {
            buffer << layers[i].write_expression(layers_outputs_name[i-1], layers_outputs_name[i]);
        }

        buffer << layers[layers_number-1].write_expression(layers_outputs_name[layers_number-2], outputs_name);
    }

    return(buffer.str());
}
}

// OpenNN: Open Neural MultilayerPerceptrons Library.
// Copyright(C) 2005-2018 Artificial Intelligence Techniques, SL.
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
