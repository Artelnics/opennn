//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   P R O B A B I L I S T I C   L A Y E R   3 D   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "strings_utilities.h"
#include "probabilistic_layer_3d.h"

namespace opennn
{

/// Default constructor.
/// It creates a probabilistic layer object with zero probabilistic neurons.
/// It does not has Synaptic weights or Biases

ProbabilisticLayer3D::ProbabilisticLayer3D()
{
    set();
}


/// Probabilistic neurons number constructor.
/// It creates a probabilistic layer with a given size.
/// @param new_neurons_number Number of neurons in the layer.

ProbabilisticLayer3D::ProbabilisticLayer3D(const Index& new_inputs_number, const Index& new_inputs_depth, const Index& new_neurons_number)
{
    set(new_inputs_number, new_inputs_depth, new_neurons_number);
}


Index ProbabilisticLayer3D::get_inputs_number() const
{
    return inputs_number;
}


Index ProbabilisticLayer3D::get_inputs_depth() const
{
    return synaptic_weights.dimension(0);
}


Index ProbabilisticLayer3D::get_neurons_number() const
{
    return biases.size();
}


dimensions ProbabilisticLayer3D::get_outputs_dimensions() const
{
    Index neurons_number = get_neurons_number();

    return { inputs_number, neurons_number };
}


Index ProbabilisticLayer3D::get_biases_number() const
{
    return biases.size();
}


/// Returns the number of layer's synaptic weights

Index ProbabilisticLayer3D::get_synaptic_weights_number() const
{
    return synaptic_weights.size();
}


/// Returns the decision threshold.

const type& ProbabilisticLayer3D::get_decision_threshold() const
{
    return decision_threshold;
}


/// Returns the method to be used for interpreting the outputs as probabilistic values.
/// The methods available for that are Binary, Probability, Competitive and Softmax.

const ProbabilisticLayer3D::ActivationFunction& ProbabilisticLayer3D::get_activation_function() const
{
    return activation_function;
}


/// Returns a string with the probabilistic method for the outputs
/// ("Competitive", "Softmax" or "NoProbabilistic").

string ProbabilisticLayer3D::write_activation_function() const
{
    if(activation_function == ActivationFunction::Competitive)
    {
        return "Competitive";
    }
    else if(activation_function == ActivationFunction::Softmax)
    {
        return "Softmax";
    }
    else
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: ProbabilisticLayer3D class.\n"
               << "string write_activation_function() const method.\n"
               << "Unknown probabilistic method.\n";

        throw runtime_error(buffer.str());
    }
}


/// Returns a string with the probabilistic method for the outputs to be included in some text
/// ("competitive", "softmax").

string ProbabilisticLayer3D::write_activation_function_text() const
{
    if(activation_function == ActivationFunction::Competitive)
    {
        return "competitive";
    }
    else if(activation_function == ActivationFunction::Softmax)
    {
        return "softmax";
    }
    else
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: ProbabilisticLayer3D class.\n"
               << "string write_activation_function_text() const method.\n"
               << "Unknown probabilistic method.\n";

        throw runtime_error(buffer.str());
    }
}


/// Returns true if messages from this class are displayed on the screen, or false if messages
/// from this class are not displayed on the screen.

const bool& ProbabilisticLayer3D::get_display() const
{
    return display;
}


/// Returns the biases of the layer.

const Tensor<type, 1>& ProbabilisticLayer3D::get_biases() const
{
    return biases;
}


/// Returns the synaptic weights of the layer.

const Tensor<type, 2>& ProbabilisticLayer3D::get_synaptic_weights() const
{
    return synaptic_weights;
}


/// Returns the number of parameters (biases and synaptic weights) of the layer.

Index ProbabilisticLayer3D::get_parameters_number() const
{
    return biases.size() + synaptic_weights.size();
}


/// Returns a single vector with all the layer parameters.
/// The format is a vector of real values.
/// The size is the number of parameters in the layer.

Tensor<type, 1> ProbabilisticLayer3D::get_parameters() const
{
    Tensor<type, 1> parameters(synaptic_weights.size() + biases.size());

    copy(/*execution::par,*/ 
        synaptic_weights.data(),
        synaptic_weights.data() + synaptic_weights.size(),
        parameters.data());

    copy(/*execution::par,*/ 
        biases.data(),
        biases.data() + biases.size(),
        parameters.data() + synaptic_weights.size());

    return parameters;
}


/// Sets a probabilistic layer with zero probabilistic neurons.
/// It also sets the rest of the members to their default values.

void ProbabilisticLayer3D::set()
{
    inputs_number = 0;
    
    biases.resize(0);

    synaptic_weights.resize(0,0);

    set_default();
}


/// Resizes the size of the probabilistic layer.
/// It also sets the rest of the class members to their default values.
/// @param new_neurons_number New size for the probabilistic layer.

void ProbabilisticLayer3D::set(const Index& new_inputs_number, const Index& new_inputs_depth, const Index& new_neurons_number)
{
    inputs_number = new_inputs_number;

    biases.resize(new_neurons_number);

    synaptic_weights.resize(new_inputs_depth, new_neurons_number);

    set_parameters_random();

    set_default();
}


/// Sets this object to be equal to another object of the same class.
/// @param other_probabilistic_layer Probabilistic layer object to be copied.

void ProbabilisticLayer3D::set(const ProbabilisticLayer3D& other_probabilistic_layer)
{
    set_default();

    activation_function = other_probabilistic_layer.activation_function;

    decision_threshold = other_probabilistic_layer.decision_threshold;

    display = other_probabilistic_layer.display;
}

void ProbabilisticLayer3D::set_name(const string& new_layer_name)
{
    layer_name = new_layer_name;
}


void ProbabilisticLayer3D::set_inputs_number(const Index& new_inputs_number)
{
    inputs_number = new_inputs_number;
}


void ProbabilisticLayer3D::set_inputs_depth(const Index& new_inputs_depth)
{
    const Index neurons_number = get_neurons_number();

    biases.resize(neurons_number);

    synaptic_weights.resize(new_inputs_depth, neurons_number);
}


void ProbabilisticLayer3D::set_neurons_number(const Index& new_neurons_number)
{
    const Index inputs_depth = get_inputs_depth();

    biases.resize(new_neurons_number);

    synaptic_weights.resize(inputs_depth, new_neurons_number);
}


void ProbabilisticLayer3D::set_biases(const Tensor<type, 1>& new_biases)
{
    biases = new_biases;
}


void ProbabilisticLayer3D::set_synaptic_weights(const Tensor<type, 2>& new_synaptic_weights)
{
    synaptic_weights = new_synaptic_weights;
}


void ProbabilisticLayer3D::set_parameters(const Tensor<type, 1>& new_parameters, const Index& index)
{
    const Index biases_number = biases.size();
    const Index synaptic_weights_number = synaptic_weights.size();

    copy(/*execution::par,*/ 
        new_parameters.data() + index,
        new_parameters.data() + index + synaptic_weights_number,
        synaptic_weights.data());

    copy(/*execution::par,*/ 
        new_parameters.data() + index + synaptic_weights_number,
        new_parameters.data() + index + synaptic_weights_number + biases_number,
        biases.data());
}


/// Sets a new threshold value for discriminating between two classes.
/// @param new_decision_threshold New discriminating value. It must be comprised between 0 and 1.

void ProbabilisticLayer3D::set_decision_threshold(const type& new_decision_threshold)
{
#ifdef OPENNN_DEBUG

    if(new_decision_threshold <= 0)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: ProbabilisticLayer3D class.\n"
               << "void set_decision_threshold(const type&) method.\n"
               << "Decision threshold(" << decision_threshold << ") must be greater than zero.\n";

        throw runtime_error(buffer.str());
    }
    else if(new_decision_threshold >= 1)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: ProbabilisticLayer3D class.\n"
               << "void set_decision_threshold(const type&) method.\n"
               << "Decision threshold(" << decision_threshold << ") must be less than one.\n";

        throw runtime_error(buffer.str());
    }

#endif

    decision_threshold = new_decision_threshold;
}


/// Sets the members to their default values:
/// <ul>
/// <li> Probabilistic method: Softmax.
/// <li> Display: True.
/// </ul>

void ProbabilisticLayer3D::set_default()
{
    layer_name = "probabilistic_layer_3d";
    
    layer_type = Layer::Type::Probabilistic3D;
    
    const Index neurons_number = get_neurons_number();

    activation_function = ActivationFunction::Softmax;

    decision_threshold = type(0.5);

    display = true;
}


/// Sets the chosen method for probabilistic postprocessing.
/// Current probabilistic methods include Binary, Probability, Competitive and Softmax.
/// @param new_activation_function Method for interpreting the outputs as probabilistic values.

void ProbabilisticLayer3D::set_activation_function(const ActivationFunction& new_activation_function)
{
#ifdef OPENNN_DEBUG

    const Index neurons_number = get_neurons_number();

    if(neurons_number == 1 && new_activation_function == ActivationFunction::Competitive)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: ProbabilisticLayer3D class.\n"
               << "void set_activation_function(const ActivationFunction&) method.\n"
               << "Activation function cannot be Competitive when the number of neurons is 1.\n";

        throw runtime_error(buffer.str());
    }

    if(neurons_number == 1 && new_activation_function == ActivationFunction::Softmax)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: ProbabilisticLayer3D class.\n"
               << "void set_activation_function(const ActivationFunction&) method.\n"
               << "Activation function cannot be Softmax when the number of neurons is 1.\n";

        throw runtime_error(buffer.str());
    }

    if(neurons_number != 1 && new_activation_function == ActivationFunction::Binary)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: ProbabilisticLayer3D class.\n"
               << "void set_activation_function(const ActivationFunction&) method.\n"
               << "Activation function cannot be Binary when the number of neurons is greater than 1.\n";

        throw runtime_error(buffer.str());
    }

    if(neurons_number != 1 && new_activation_function == ActivationFunction::Logistic)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: ProbabilisticLayer3D class.\n"
               << "void set_activation_function(const ActivationFunction&) method.\n"
               << "Activation function cannot be Logistic when the number of neurons is greater than 1.\n";

        throw runtime_error(buffer.str());
    }

#endif

    activation_function = new_activation_function;
}


/// Sets a new method for probabilistic processing from a string with the name.
/// Current probabilistic methods include Competitive and Softmax.
/// @param new_activation_function Method for interpreting the outputs as probabilistic values.

void ProbabilisticLayer3D::set_activation_function(const string& new_activation_function)
{
    if(new_activation_function == "Competitive")
    {
        set_activation_function(ActivationFunction::Competitive);
    }
    else if(new_activation_function == "Softmax")
    {
        set_activation_function(ActivationFunction::Softmax);
    }
    else
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: ProbabilisticLayer3D class.\n"
               << "void set_activation_function(const string&) method.\n"
               << "Unknown probabilistic method: " << new_activation_function << ".\n";

        throw runtime_error(buffer.str());
    }
}


/// Sets a new display value.
/// If it is set to true messages from this class are displayed on the screen;
/// if it is set to false messages from this class are not displayed on the screen.
/// @param new_display Display value.

void ProbabilisticLayer3D::set_display(const bool& new_display)
{
    display = new_display;
}


/// Initializes the biases of all the neurons in the probabilistic layer with a given value.
/// @param value Biases initialization value.

void ProbabilisticLayer3D::set_biases_constant(const type& value)
{
    biases.setConstant(value);
}


/// Initializes the synaptic weights of all the neurons in the probabilistic layer with a given value.
/// @param value Synaptic weights initialization value.

void ProbabilisticLayer3D::set_synaptic_weights_constant(const type& value)
{
    synaptic_weights.setConstant(value);
}


void ProbabilisticLayer3D::set_synaptic_weights_constant_Glorot()
{
    synaptic_weights.setRandom();
}


/// Initializes all the biases and synaptic weights in the neural newtork with a given value.
/// @param value Parameters initialization value.

void ProbabilisticLayer3D::set_parameters_constant(const type& value)
{
    biases.setConstant(value);

    synaptic_weights.setConstant(value);
}


/// Initializes all the biases and synaptic weights in the neural newtork at random with values comprised
/// between -1 and +1.

void ProbabilisticLayer3D::set_parameters_random()
{
    biases.setRandom();

    synaptic_weights.setRandom();
}


/// Initializes the biases to zeroes and the synaptic weights with the Glorot Uniform initializer

void ProbabilisticLayer3D::set_parameters_glorot()
{
    biases.setZero();
    
    const type limit = sqrt(6 / type(get_inputs_depth() + get_neurons_number()));

    const type minimum = -limit;
    const type maximum = limit;
    
#pragma omp parallel for

    for (Index i = 0; i < synaptic_weights.size(); i++)
    {
        const type random = static_cast<type>(rand() / (RAND_MAX + 1.0));

        synaptic_weights(i) = minimum + (maximum - minimum) * random;
    }
}


void ProbabilisticLayer3D::insert_parameters(const Tensor<type, 1>& parameters, const Index&)
{
    const Index biases_number = get_biases_number();
    const Index synaptic_weights_number = get_synaptic_weights_number();

    copy(/*execution::par,*/ 
        parameters.data(),
         parameters.data() + biases_number,
         biases.data());

    copy(/*execution::par,*/ 
        parameters.data() + biases_number,
         parameters.data() + biases_number + synaptic_weights_number,
         synaptic_weights.data());
}


void ProbabilisticLayer3D::calculate_combinations(const Tensor<type, 3>& inputs,
                                                  Tensor<type, 3>& combinations) const
{
    combinations.device(*thread_pool_device) = inputs.contract(synaptic_weights, contraction_indices);

    sum_matrices(thread_pool_device, biases, combinations);
}


void ProbabilisticLayer3D::calculate_activations(const Tensor<type, 3>& combinations,
                                                 Tensor<type, 3>& activations) const
{
    switch(activation_function)
    {
    case ActivationFunction::Softmax: softmax(combinations, activations); return;

    default: return;
    }
}


void ProbabilisticLayer3D::forward_propagate(const Tensor<pair<type*, dimensions>, 1>& inputs_pair,
                                             LayerForwardPropagation* forward_propagation,
                                             const bool& is_training)
{
    const TensorMap<Tensor<type, 3>> inputs(inputs_pair(0).first, inputs_pair(0).second[0], inputs_pair(0).second[1], inputs_pair(0).second[2]);

    ProbabilisticLayer3DForwardPropagation* probabilistic_layer_3d_forward_propagation
            = static_cast<ProbabilisticLayer3DForwardPropagation*>(forward_propagation);
    
    Tensor<type, 3>& outputs = probabilistic_layer_3d_forward_propagation->outputs;
    
    calculate_combinations(inputs, outputs);

    if (is_training)
        calculate_activations(outputs, outputs);
    //else competitive(outputs, outputs);
}


void ProbabilisticLayer3D::back_propagate(const Tensor<pair<type*, dimensions>, 1>& inputs_pair,
                                          const Tensor<pair<type*, dimensions>, 1>& deltas_pair,
                                          LayerForwardPropagation* forward_propagation,
                                          LayerBackPropagation* back_propagation) const
{
    const TensorMap<Tensor<type, 3>> inputs(inputs_pair(0).first, 
                                            inputs_pair(0).second[0], 
                                            inputs_pair(0).second[1], 
                                            inputs_pair(0).second[2]);

    // Forward propagation

    const Index batch_samples_number = forward_propagation->batch_samples_number;

    ProbabilisticLayer3DForwardPropagation* probabilistic_layer_3d_forward_propagation =
            static_cast<ProbabilisticLayer3DForwardPropagation*>(forward_propagation);

    const Tensor<type, 3>& outputs = probabilistic_layer_3d_forward_propagation->outputs;

    // Back propagation

    ProbabilisticLayer3DBackPropagation* probabilistic_layer_3d_back_propagation =
            static_cast<ProbabilisticLayer3DBackPropagation*>(back_propagation);

    const Tensor<type, 2>& targets = probabilistic_layer_3d_back_propagation->targets;

    Tensor<type, 3>& error_combinations_derivatives = probabilistic_layer_3d_back_propagation->error_combinations_derivatives;

    Tensor<type, 1>& biases_derivatives = probabilistic_layer_3d_back_propagation->biases_derivatives;
    Tensor<type, 2>& synaptic_weights_derivatives = probabilistic_layer_3d_back_propagation->synaptic_weights_derivatives;

    Tensor<type, 3>& input_derivatives = probabilistic_layer_3d_back_propagation->input_derivatives;

    const Eigen::array<IndexPair<Index>, 2> double_contraction_indices = { IndexPair<Index>(0, 0), IndexPair<Index>(1, 1) };
    const Eigen::array<IndexPair<Index>, 1> single_contraction_indices = { IndexPair<Index>(2, 1) };

    calculate_error_combinations_derivatives(outputs, targets, error_combinations_derivatives);

    biases_derivatives.device(*thread_pool_device) 
        = error_combinations_derivatives.sum(Eigen::array<Index, 2>({ 0, 1 }));

    synaptic_weights_derivatives.device(*thread_pool_device) 
        = inputs.contract(error_combinations_derivatives, double_contraction_indices);

    input_derivatives.device(*thread_pool_device) 
        = error_combinations_derivatives.contract(synaptic_weights, single_contraction_indices);
}


void ProbabilisticLayer3D::calculate_error_combinations_derivatives(const Tensor<type, 3>& outputs, 
                                                                    const Tensor<type, 2>& targets,  
                                                                    Tensor<type, 3>& error_combinations_derivatives) const
{
    const Index batch_samples_number = outputs.dimension(0);
    const Index outputs_number = outputs.dimension(1);

    /// @todo Can we simplify this? For instance put the division in the last line somewhere else. 

    error_combinations_derivatives.device(*thread_pool_device) = outputs;

#pragma omp parallel for

    for (Index i = 0; i < batch_samples_number; i++)
        for (Index j = 0; j < outputs_number; j++)
            error_combinations_derivatives(i, j, Index(targets(i, j))) -= 1;
    
    error_combinations_derivatives.device(*thread_pool_device) = error_combinations_derivatives / type(batch_samples_number * outputs_number);
}


void ProbabilisticLayer3D::insert_gradient(LayerBackPropagation* back_propagation,
                                           const Index& index,
                                           Tensor<type, 1>& gradient) const
{
    const Index biases_number = get_biases_number();
    const Index synaptic_weights_number = get_synaptic_weights_number();

    const ProbabilisticLayer3DBackPropagation* probabilistic_layer_3d_back_propagation =
            static_cast<ProbabilisticLayer3DBackPropagation*>(back_propagation);

    const type* synaptic_weights_derivatives_data = probabilistic_layer_3d_back_propagation->synaptic_weights_derivatives.data();
    const type* biases_derivatives_data = probabilistic_layer_3d_back_propagation->biases_derivatives.data();

    copy(/*execution::par,*/ 
         synaptic_weights_derivatives_data,
         synaptic_weights_derivatives_data + synaptic_weights_number,
         gradient.data() + index);

    copy(/*execution::par,*/ 
         biases_derivatives_data,
         biases_derivatives_data + biases_number,
         gradient.data() + index + synaptic_weights_number);
}


/// Serializes the probabilistic layer object into an XML document of the TinyXML library without keeping the DOM tree in memory.
/// See the OpenNN manual for more information about the format of this document.

void ProbabilisticLayer3D::write_XML(tinyxml2::XMLPrinter& file_stream) const
{
    ostringstream buffer;

    // Probabilistic layer

    file_stream.OpenElement("ProbabilisticLayer3D");

    // Inputs number

    file_stream.OpenElement("InputsNumber");

    buffer.str("");
    buffer << get_inputs_number();

    file_stream.PushText(buffer.str().c_str());

    file_stream.CloseElement();

    // Neurons number

    file_stream.OpenElement("NeuronsNumber");

    buffer.str("");
    buffer << get_neurons_number();

    file_stream.PushText(buffer.str().c_str());

    file_stream.CloseElement();

    // Activation function

    file_stream.OpenElement("ActivationFunction");

    file_stream.PushText(write_activation_function().c_str());

    file_stream.CloseElement();

    // Parameters

    file_stream.OpenElement("Parameters");

    buffer.str("");

    const Tensor<type, 1> parameters = get_parameters();
    const Index parameters_size = parameters.size();

    for(Index i = 0; i < parameters_size; i++)
    {
        buffer << parameters(i);

        if(i != (parameters_size-1)) buffer << " ";
    }

    file_stream.PushText(buffer.str().c_str());

    file_stream.CloseElement();

    // Decision threshold

    file_stream.OpenElement("DecisionThreshold");

    buffer.str("");
    buffer << decision_threshold;

    file_stream.PushText(buffer.str().c_str());

    file_stream.CloseElement();

    // Probabilistic layer (end tag)

    file_stream.CloseElement();
}


/// Deserializes a TinyXML document into this probabilistic layer object.
/// @param document XML document containing the member data.

void ProbabilisticLayer3D::from_XML(const tinyxml2::XMLDocument& document)
{
    ostringstream buffer;

    // Probabilistic layer

    const tinyxml2::XMLElement* probabilistic_layer_element = document.FirstChildElement("ProbabilisticLayer3D");

    if(!probabilistic_layer_element)
    {
        buffer << "OpenNN Exception: ProbabilisticLayer3D class.\n"
               << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
               << "Probabilistic layer element is nullptr.\n";

        throw runtime_error(buffer.str());
    }

    // Inputs number

    const tinyxml2::XMLElement* inputs_number_element = probabilistic_layer_element->FirstChildElement("InputsNumber");

    if(!inputs_number_element)
    {
        buffer << "OpenNN Exception: ProbabilisticLayer3D class.\n"
               << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
               << "Inputs number element is nullptr.\n";

        throw runtime_error(buffer.str());
    }
/*
    Index new_inputs_number;

    if(inputs_number_element->GetText())
    {
        new_inputs_number = Index(stoi(inputs_number_element->GetText()));
    }
*/
    // Neurons number

    const tinyxml2::XMLElement* neurons_number_element = probabilistic_layer_element->FirstChildElement("NeuronsNumber");

    if(!inputs_number_element)
    {
        buffer << "OpenNN Exception: ProbabilisticLayer3D class.\n"
               << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
               << "Neurons number element is nullptr.\n";

        throw runtime_error(buffer.str());
    }
/*
    Index new_neurons_number;

    if(neurons_number_element->GetText())
    {
        new_neurons_number = Index(stoi(neurons_number_element->GetText()));
    }

    set(new_inputs_number, new_neurons_number);
*/
    // Activation function

    const tinyxml2::XMLElement* activation_function_element = probabilistic_layer_element->FirstChildElement("ActivationFunction");

    if(!activation_function_element)
    {
        buffer << "OpenNN Exception: ProbabilisticLayer3D class.\n"
               << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
               << "Activation function element is nullptr.\n";

        throw runtime_error(buffer.str());
    }

    if(activation_function_element->GetText())
    {
        set_activation_function(activation_function_element->GetText());
    }

    // Parameters

    const tinyxml2::XMLElement* parameters_element = probabilistic_layer_element->FirstChildElement("Parameters");

    if(!parameters_element)
    {
        buffer << "OpenNN Exception: ProbabilisticLayer3D class.\n"
               << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
               << "Parameters element is nullptr.\n";

        throw runtime_error(buffer.str());
    }

    if(parameters_element->GetText())
    {
        const string parameters_string = parameters_element->GetText();

        set_parameters(to_type_vector(parameters_string, ' '));
    }

    // Decision threshold

    const tinyxml2::XMLElement* decision_threshold_element = probabilistic_layer_element->FirstChildElement("DecisionThreshold");

    if(!decision_threshold_element)
    {
        buffer << "OpenNN Exception: ProbabilisticLayer3D class.\n"
               << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
               << "Decision threshold element is nullptr.\n";

        throw runtime_error(buffer.str());
    }

    if(decision_threshold_element->GetText())
    {
        set_decision_threshold(type(atof(decision_threshold_element->GetText())));
    }

    // Display

    const tinyxml2::XMLElement* display_element = probabilistic_layer_element->FirstChildElement("Display");

    if(display_element)
    {
        const string new_display_string = display_element->GetText();

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


pair<type*, dimensions> ProbabilisticLayer3DForwardPropagation::get_outputs_pair() const
{
    ProbabilisticLayer3D* probabilistic_layer_3d = static_cast<ProbabilisticLayer3D*>(layer);

    const Index neurons_number = probabilistic_layer_3d->get_neurons_number();

    const Index inputs_number = probabilistic_layer_3d->get_inputs_number();

    return pair<type*, dimensions>(outputs_data, { batch_samples_number, inputs_number, neurons_number });
}


void ProbabilisticLayer3DForwardPropagation::set(const Index& new_batch_samples_number, Layer* new_layer)
{
    layer = new_layer;

    ProbabilisticLayer3D* probabilistic_layer_3d = static_cast<ProbabilisticLayer3D*>(layer);

    batch_samples_number = new_batch_samples_number;

    const Index neurons_number = probabilistic_layer_3d->get_neurons_number();

    const Index inputs_number = probabilistic_layer_3d->get_inputs_number();
    
    outputs.resize(batch_samples_number, inputs_number, neurons_number);
    
    outputs_data = outputs.data();
}


void ProbabilisticLayer3DBackPropagation::set(const Index& new_batch_samples_number, Layer* new_layer)
{
    layer = new_layer;

    ProbabilisticLayer3D* probabilistic_layer_3d = static_cast<ProbabilisticLayer3D*>(layer);

    batch_samples_number = new_batch_samples_number;

    const Index neurons_number = probabilistic_layer_3d->get_neurons_number();
    const Index inputs_number = probabilistic_layer_3d->get_inputs_number();
    const Index inputs_depth = probabilistic_layer_3d->get_inputs_depth();

    targets.resize(batch_samples_number, inputs_number);

    biases_derivatives.resize(neurons_number);

    synaptic_weights_derivatives.resize(inputs_depth, neurons_number);

    error_combinations_derivatives.resize(batch_samples_number, inputs_number, neurons_number);

    input_derivatives.resize(batch_samples_number, inputs_number, inputs_depth);

    inputs_derivatives.resize(1);
    inputs_derivatives(0).first = input_derivatives.data();
    inputs_derivatives(0).second = { batch_samples_number, inputs_number, inputs_depth };
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
