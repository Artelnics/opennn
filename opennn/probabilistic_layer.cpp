//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   P R O B A B I L I S T I C   L A Y E R   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "tensors.h"
#include "strings_utilities.h"
#include "probabilistic_layer.h"

namespace opennn
{

ProbabilisticLayer::ProbabilisticLayer()
{
    set();
}


ProbabilisticLayer::ProbabilisticLayer(const Index& new_inputs_number, const Index& new_neurons_number)
{
    set(new_inputs_number, new_neurons_number);

    if(new_neurons_number > 1)
    {
        activation_function = ActivationFunction::Softmax;
    }
}


ProbabilisticLayer::ProbabilisticLayer(const dimensions& new_input_dimensions, const dimensions& new_output_dimensions)
{
    set(new_input_dimensions[0], new_output_dimensions[0]);

    if(new_output_dimensions[0] > 1)
    {
        activation_function = ActivationFunction::Softmax;
    }
}


void ProbabilisticLayer::set_name(const string& new_layer_name)
{
    name = new_layer_name;
}


Index ProbabilisticLayer::get_inputs_number() const
{
    return synaptic_weights.dimension(0);
}


Index ProbabilisticLayer::get_neurons_number() const
{
    return biases.size();
}


dimensions ProbabilisticLayer::get_output_dimensions() const
{
    Index neurons_number = get_neurons_number();

    return { neurons_number };
}


Index ProbabilisticLayer::get_biases_number() const
{
    return biases.size();
}


Index ProbabilisticLayer::get_synaptic_weights_number() const
{
    return synaptic_weights.size();
}


const type& ProbabilisticLayer::get_decision_threshold() const
{
    return decision_threshold;
}


const ProbabilisticLayer::ActivationFunction& ProbabilisticLayer::get_activation_function() const
{
    return activation_function;
}


string ProbabilisticLayer::write_activation_function() const
{
    if(activation_function == ActivationFunction::Binary)
    {
        return "Binary";
    }
    else if(activation_function == ActivationFunction::Logistic)
    {
        return "Logistic";
    }
    else if(activation_function == ActivationFunction::Competitive)
    {
        return "Competitive";
    }
    else if(activation_function == ActivationFunction::Softmax)
    {
        return "Softmax";
    }
    else
    {
        throw runtime_error("Unknown probabilistic method.\n");
    }
}


string ProbabilisticLayer::write_activation_function_text() const
{
    if(activation_function == ActivationFunction::Binary)
    {
        return "binary";
    }
    else if(activation_function == ActivationFunction::Logistic)
    {
        return "logistic";
    }
    else if(activation_function == ActivationFunction::Competitive)
    {
        return "competitive";
    }
    else if(activation_function == ActivationFunction::Softmax)
    {
        return "softmax";
    }
    else
    {
        throw runtime_error("Unknown probabilistic method.\n");
    }
}


const bool& ProbabilisticLayer::get_display() const
{
    return display;
}


const Tensor<type, 1>& ProbabilisticLayer::get_biases() const
{
    return biases;
}


const Tensor<type, 2>& ProbabilisticLayer::get_synaptic_weights() const
{
    return synaptic_weights;
}


Index ProbabilisticLayer::get_parameters_number() const
{
    return biases.size() + synaptic_weights.size();
}


Tensor<type, 1> ProbabilisticLayer::get_parameters() const
{
    Tensor<type, 1> parameters(synaptic_weights.size() + biases.size());

    copy(synaptic_weights.data(),
         synaptic_weights.data() + synaptic_weights.size(),
         parameters.data());

    copy(biases.data(),
         biases.data() + biases.size(),
         parameters.data() + synaptic_weights.size());

    return parameters;
}


void ProbabilisticLayer::set()
{
    biases.resize(0);

    synaptic_weights.resize(0,0);

    set_default();
}


void ProbabilisticLayer::set(const Index& new_inputs_number, const Index& new_neurons_number)
{
    biases.resize(new_neurons_number);

    synaptic_weights.resize(new_inputs_number, new_neurons_number);

    set_parameters_random();

    set_default();
}


void ProbabilisticLayer::set(const ProbabilisticLayer& other_probabilistic_layer)
{
    set_default();

    activation_function = other_probabilistic_layer.activation_function;

    decision_threshold = other_probabilistic_layer.decision_threshold;

    display = other_probabilistic_layer.display;
}


void ProbabilisticLayer::set_inputs_number(const Index& new_inputs_number)
{
    const Index neurons_number = get_neurons_number();

    biases.resize(neurons_number);

    synaptic_weights.resize(new_inputs_number, neurons_number);
}


void ProbabilisticLayer::set_neurons_number(const Index& new_neurons_number)
{
    const Index inputs_number = get_inputs_number();

    biases.resize(new_neurons_number);

    synaptic_weights.resize(inputs_number, new_neurons_number);
}


void ProbabilisticLayer::set_biases(const Tensor<type, 1>& new_biases)
{
    biases = new_biases;
}


void ProbabilisticLayer::set_synaptic_weights(const Tensor<type, 2>& new_synaptic_weights)
{
    synaptic_weights = new_synaptic_weights;
}


void ProbabilisticLayer::set_parameters(const Tensor<type, 1>& new_parameters, const Index& index)
{
    const Index biases_number = biases.size();
    const Index synaptic_weights_number = synaptic_weights.size();

    copy(new_parameters.data() + index,
         new_parameters.data() + index + synaptic_weights_number,
         synaptic_weights.data());

    copy(new_parameters.data() + index + synaptic_weights_number,
         new_parameters.data() + index + synaptic_weights_number + biases_number,
         biases.data());
}


void ProbabilisticLayer::set_decision_threshold(const type& new_decision_threshold)
{
    decision_threshold = new_decision_threshold;
}


void ProbabilisticLayer::set_default()
{
    name = "probabilistic_layer";

    layer_type = Layer::Type::Probabilistic;

    const Index neurons_number = get_neurons_number();

    if(neurons_number == 1)
    {
        activation_function = ActivationFunction::Logistic;
    }
    else
    {
        activation_function = ActivationFunction::Softmax;
    }

    decision_threshold = type(0.5);

    display = true;
}


void ProbabilisticLayer::set_activation_function(const ActivationFunction& new_activation_function)
{
#ifdef OPENNN_DEBUG

    const Index neurons_number = get_neurons_number();

    if(neurons_number == 1 && new_activation_function == ActivationFunction::Competitive)
        throw runtime_error("Activation function cannot be Competitive when the number of neurons is 1.\n");

    if(neurons_number == 1 && new_activation_function == ActivationFunction::Softmax)
        throw runtime_error("Activation function cannot be Softmax when the number of neurons is 1.\n");

    if(neurons_number != 1 && new_activation_function == ActivationFunction::Binary)
        throw runtime_error("Activation function cannot be Binary when the number of neurons is greater than 1.\n");

    if(neurons_number != 1 && new_activation_function == ActivationFunction::Logistic)
        throw runtime_error("Activation function cannot be Logistic when the number of neurons is greater than 1.\n");

#endif

    activation_function = new_activation_function;
}


void ProbabilisticLayer::set_activation_function(const string& new_activation_function)
{
    if(new_activation_function == "Binary")
    {
        set_activation_function(ActivationFunction::Binary);
    }
    else if(new_activation_function == "Logistic")
    {
        set_activation_function(ActivationFunction::Logistic);
    }
    else if(new_activation_function == "Competitive")
    {
        set_activation_function(ActivationFunction::Competitive);
    }
    else if(new_activation_function == "Softmax")
    {
        set_activation_function(ActivationFunction::Softmax);
    }
    else
    {
        throw runtime_error("Unknown probabilistic method: " + new_activation_function + ".\n");
    }
}


void ProbabilisticLayer::set_display(const bool& new_display)
{
    display = new_display;
}


void ProbabilisticLayer::set_parameters_constant(const type& value)
{
    biases.setConstant(value);

    synaptic_weights.setConstant(value);
}


void ProbabilisticLayer::set_parameters_random()
{
    set_random(biases);

    set_random(synaptic_weights);
}


void ProbabilisticLayer::insert_parameters(const Tensor<type, 1>& parameters, const Index&)
{
    const Index biases_number = get_biases_number();
    const Index synaptic_weights_number = get_synaptic_weights_number();

    copy(parameters.data(),
         parameters.data() + biases_number,
         biases.data());

    copy(parameters.data() + biases_number,
         parameters.data() + biases_number + synaptic_weights_number,
         synaptic_weights.data());
}


void ProbabilisticLayer::calculate_combinations(const Tensor<type, 2>& inputs,
                                                Tensor<type, 2>& combinations) const
{
    combinations.device(*thread_pool_device) = inputs.contract(synaptic_weights, A_B);

    sum_columns(thread_pool_device.get(), biases, combinations);
}


void ProbabilisticLayer::calculate_activations(const Tensor<type, 2>& combinations,
                                               Tensor<type, 2>& activations,
                                               Tensor<type, 1>& aux_rows) const
{
    switch(activation_function)
    {
    case ActivationFunction::Binary: binary(combinations, activations); return;

    case ActivationFunction::Logistic: logistic(activations); return;

    case ActivationFunction::Competitive: competitive(combinations, activations); return;

    case ActivationFunction::Softmax: softmax(combinations, activations, aux_rows); return;

    default: return;
    }
}


void ProbabilisticLayer::calculate_activations_derivatives(const Tensor<type, 2>& combinations,
                                                           Tensor<type, 2>& activations,
                                                           Tensor<type, 2>& activations_derivatives) const
{
    switch(activation_function)
    {
    case ActivationFunction::Logistic:
        
        logistic_derivatives(activations,
                             activations_derivatives);

        return;

    default:

        return;
    }
}


void ProbabilisticLayer::forward_propagate(const Tensor<pair<type*, dimensions>, 1>& inputs_pair,
                                           LayerForwardPropagation* forward_propagation,
                                           const bool& is_training)
{
    const Index neurons_number = get_neurons_number();

    const TensorMap<Tensor<type, 2>> inputs(inputs_pair(0).first, inputs_pair(0).second[0], inputs_pair(0).second[1]);

    ProbabilisticLayerForwardPropagation* probabilistic_layer_forward_propagation
            = static_cast<ProbabilisticLayerForwardPropagation*>(forward_propagation);

    Tensor<type, 2>& outputs = probabilistic_layer_forward_propagation->outputs;

    calculate_combinations(inputs, outputs);

    if(is_training)
    {
        if(neurons_number == 1)
        {
            Tensor<type, 2>& activations_derivatives = probabilistic_layer_forward_propagation->activations_derivatives;

            calculate_activations_derivatives(outputs,
                                              outputs,
                                              activations_derivatives);
        }
        else
        {
            Tensor<type, 1>& aux_rows = probabilistic_layer_forward_propagation->aux_rows;

            calculate_activations(outputs,
                                  outputs,
                                  aux_rows); 
        }
        
    }
    else
    {
        Tensor<type, 1>& aux_rows = probabilistic_layer_forward_propagation->aux_rows;

        calculate_activations(outputs,
                              outputs,
                              aux_rows);
    }
}


void ProbabilisticLayer::back_propagate(const Tensor<pair<type*, dimensions>, 1>& inputs_pair,
                                                  const Tensor<pair<type*, dimensions>, 1>& deltas_pair,
                                                  LayerForwardPropagation* forward_propagation,
                                                  LayerBackPropagation* back_propagation) const
{
    const Index samples_number = inputs_pair(0).second[0];
    const Index neurons_number = get_neurons_number();

    const TensorMap<Tensor<type, 2>> inputs(inputs_pair(0).first, samples_number, inputs_pair(0).second[1]);
    const TensorMap<Tensor<type, 2>> deltas(deltas_pair(0).first, samples_number, deltas_pair(0).second[1]);

    // Forward propagation

    ProbabilisticLayerForwardPropagation* probabilistic_layer_forward_propagation =
            static_cast<ProbabilisticLayerForwardPropagation*>(forward_propagation);

    const Tensor<type, 2>& outputs = probabilistic_layer_forward_propagation->outputs;

    // Back propagation

    ProbabilisticLayerBackPropagation* probabilistic_layer_back_propagation =
            static_cast<ProbabilisticLayerBackPropagation*>(back_propagation);

    const Tensor<type, 2>& targets = probabilistic_layer_back_propagation->targets;
    Tensor<type, 2>& input_derivatives = probabilistic_layer_back_propagation->input_derivatives;

    Tensor<type, 2>& error_combinations_derivatives = probabilistic_layer_back_propagation->error_combinations_derivatives;

    if(neurons_number == 1)
    {
        const Tensor<type, 2>& activations_derivatives = probabilistic_layer_forward_propagation->activations_derivatives;

        error_combinations_derivatives.device(*thread_pool_device) = deltas * activations_derivatives;
    }
    else
    {
        error_combinations_derivatives.device(*thread_pool_device) = outputs - targets;
    }

    Tensor<type, 1>& biases_derivatives = probabilistic_layer_back_propagation->biases_derivatives;

    Tensor<type, 2>& synaptic_weights_derivatives = probabilistic_layer_back_propagation->synaptic_weights_derivatives;

    const Eigen::array<Index, 1> sum_dimensions({0});

    synaptic_weights_derivatives.device(*thread_pool_device) = inputs.contract(error_combinations_derivatives, AT_B);

    biases_derivatives.device(*thread_pool_device) = error_combinations_derivatives.sum(sum_dimensions);

    input_derivatives.device(*thread_pool_device) = error_combinations_derivatives.contract(synaptic_weights, A_BT);
}


void ProbabilisticLayer::insert_gradient(LayerBackPropagation* back_propagation,
                                         const Index& index,
                                         Tensor<type, 1>& gradient) const
{
    const Index biases_number = get_biases_number();
    const Index synaptic_weights_number = get_synaptic_weights_number();

    const ProbabilisticLayerBackPropagation* probabilistic_layer_back_propagation =
            static_cast<ProbabilisticLayerBackPropagation*>(back_propagation);

    const type* synaptic_weights_derivatives_data = probabilistic_layer_back_propagation->synaptic_weights_derivatives.data();
    const type* biases_derivatives_data = probabilistic_layer_back_propagation->biases_derivatives.data();

    copy(synaptic_weights_derivatives_data,
         synaptic_weights_derivatives_data + synaptic_weights_number,
         gradient.data() + index);

    copy(biases_derivatives_data,
         biases_derivatives_data + biases_number,
         gradient.data() + index + synaptic_weights_number);
}


void ProbabilisticLayer::insert_squared_errors_Jacobian_lm(LayerBackPropagationLM* back_propagation,
                                                           const Index& index,
                                                           Tensor<type, 2>& squared_errors_Jacobian) const
{
    ProbabilisticLayerBackPropagationLM* probabilistic_layer_back_propagation_lm =
            static_cast<ProbabilisticLayerBackPropagationLM*>(back_propagation);

    const Index batch_samples_number = back_propagation->batch_samples_number;

    const Index parameters_number = get_parameters_number();

    type* squared_errors_Jacobian_data = probabilistic_layer_back_propagation_lm->squared_errors_Jacobian.data();

    copy(squared_errors_Jacobian_data,
         squared_errors_Jacobian_data + parameters_number*batch_samples_number,
         squared_errors_Jacobian_data + index);
}


void ProbabilisticLayer::to_XML(tinyxml2::XMLPrinter& file_stream) const
{
    ostringstream buffer;

    // Probabilistic layer

    file_stream.OpenElement("ProbabilisticLayer");

    // Inputs number

    file_stream.OpenElement("InputsNumber");
    file_stream.PushText(to_string(get_inputs_number()).c_str());
    file_stream.CloseElement();

    // Neurons number

    file_stream.OpenElement("NeuronsNumber");
    file_stream.PushText(to_string(get_neurons_number()).c_str());
    file_stream.CloseElement();

    // Activation function

    file_stream.OpenElement("ActivationFunction");
    file_stream.PushText(write_activation_function().c_str());
    file_stream.CloseElement();

    // Parameters

    file_stream.OpenElement("Parameters");
    file_stream.PushText(tensor_to_string(get_parameters()).c_str());
    file_stream.CloseElement();

    // Decision threshold

    file_stream.OpenElement("DecisionThreshold");
    file_stream.PushText(to_string(decision_threshold).c_str());
    file_stream.CloseElement();

    // Probabilistic layer (end tag)

    file_stream.CloseElement();
}


void ProbabilisticLayer::from_XML(const tinyxml2::XMLDocument& document)
{
    ostringstream buffer;

    // Probabilistic layer

    const tinyxml2::XMLElement* probabilistic_layer_element = document.FirstChildElement("ProbabilisticLayer");

    if(!probabilistic_layer_element)
        throw runtime_error("Probabilistic layer element is nullptr.\n");

    // Inputs number

    const tinyxml2::XMLElement* inputs_number_element = probabilistic_layer_element->FirstChildElement("InputsNumber");

    if(!inputs_number_element)
        throw runtime_error("Inputs number element is nullptr.\n");

    Index new_inputs_number;

    if(inputs_number_element->GetText())
        new_inputs_number = Index(stoi(inputs_number_element->GetText()));

    // Neurons number

    const tinyxml2::XMLElement* neurons_number_element = probabilistic_layer_element->FirstChildElement("NeuronsNumber");

    if(!inputs_number_element)
        throw runtime_error("Neurons number element is nullptr.\n");

    Index new_neurons_number;

    if(neurons_number_element->GetText())
        new_neurons_number = Index(stoi(neurons_number_element->GetText()));

    set(new_inputs_number, new_neurons_number);

    // Activation function

    const tinyxml2::XMLElement* activation_function_element = probabilistic_layer_element->FirstChildElement("ActivationFunction");

    if(!activation_function_element)
        throw runtime_error("Activation function element is nullptr.\n");

    if(activation_function_element->GetText())
        set_activation_function(activation_function_element->GetText());

    // Parameters

    const tinyxml2::XMLElement* parameters_element = probabilistic_layer_element->FirstChildElement("Parameters");

    if(!parameters_element)
        throw runtime_error("Parameters element is nullptr.\n");

    if(parameters_element->GetText())
        set_parameters(to_type_vector(parameters_element->GetText(), " "));

    // Decision threshold

    const tinyxml2::XMLElement* decision_threshold_element = probabilistic_layer_element->FirstChildElement("DecisionThreshold");

    if(!decision_threshold_element)
        throw runtime_error("Decision threshold element is nullptr.\n");

    if(decision_threshold_element->GetText())
        set_decision_threshold(type(atof(decision_threshold_element->GetText())));

    // Display

    const tinyxml2::XMLElement* display_element = probabilistic_layer_element->FirstChildElement("Display");

    if(display_element)
        set_display(display_element->GetText() != string("0"));
}


string ProbabilisticLayer::write_binary_expression(const Tensor<string, 1>& inputs_names, const Tensor<string, 1>& outputs_name) const
{
    ostringstream buffer;

    for(Index j = 0; j < outputs_name.size(); j++)
    {
        buffer << outputs_name(j) << " = binary(" << inputs_names(j) << ");\n";
    }
    return buffer.str();
}


string ProbabilisticLayer::write_logistic_expression(const Tensor<string, 1>& inputs_names,
                                                     const Tensor<string, 1>& outputs_name) const
{
    ostringstream buffer;

    for(Index j = 0; j < outputs_name.size(); j++)
    {
        buffer << outputs_name(j) << " = logistic(" << inputs_names(j) << ");\n";
    }
    return buffer.str();
}


string ProbabilisticLayer::write_competitive_expression(const Tensor<string, 1>& inputs_names, const Tensor<string, 1>& outputs_name) const
{
    ostringstream buffer;

    for(Index j = 0; j < outputs_name.size(); j++)
    {
        buffer << outputs_name(j) << " = competitive(" << inputs_names(j) << ");\n";
    }
    return buffer.str();
}


string ProbabilisticLayer::write_softmax_expression(const Tensor<string, 1>& inputs_names, const Tensor<string, 1>& outputs_name) const
{
    ostringstream buffer;

    for(Index j = 0; j < outputs_name.size(); j++)
    {
        buffer << outputs_name(j) << " = softmax(" << inputs_names(j) << ");\n";
    }

    return buffer.str();
}


string ProbabilisticLayer::write_combinations(const Tensor<string, 1>& inputs_names) const
{
    ostringstream buffer;

    const Index inputs_number = get_inputs_number();
    const Index neurons_number = get_neurons_number();

    for(Index i = 0; i < neurons_number; i++)
    {
        buffer << "probabilistic_layer_combinations_" << to_string(i) << " = " << biases(i);

        for(Index j = 0; j < inputs_number; j++)
        {
            buffer << " +" << synaptic_weights(j, i) << "*" << inputs_names(j) << "";
        }

        buffer << " " << endl;
    }

    buffer << "\t" << endl;

    return buffer.str();
}


string ProbabilisticLayer::write_activations(const Tensor<string, 1>& outputs_name) const
{
    ostringstream buffer;

    const Index neurons_number = get_neurons_number();

    for(Index i = 0; i < neurons_number; i++)
    {
        switch(activation_function)
        {
        case ActivationFunction::Binary:
        {
            buffer << "\tif" << "probabilistic_layer_combinations_" << to_string(i) << " < 0.5, " << outputs_name(i) << "= 0.0. Else " << outputs_name(i) << " = 1.0\n";
        }
            break;

        case ActivationFunction::Logistic:
        {
            buffer <<  outputs_name(i) << " = 1.0/(1.0 + exp(-" <<  "probabilistic_layer_combinations_" << to_string(i) << ") );\n";
        }
            break;

        case ActivationFunction::Competitive:
            if(i == 0)
            {
                buffer << "\tfor each probabilistic_layer_combinations_i:"<<endl;

                buffer <<"\t\tif probabilistic_layer_combinations_i is equal to max(probabilistic_layer_combinations_i):"<<endl;

                buffer <<"\t\t\tactivations[i] = 1"<<endl;

                buffer <<"\t\telse:"<<endl;

                buffer <<"\t\t\tactivations[i] = 0"<<endl;
            }

            break;

        case ActivationFunction::Softmax:

            if(i == 0)
            {
                buffer << "sum = ";

                for(Index i = 0; i < neurons_number; i++)
                {
                    buffer << "exp(probabilistic_layer_combinations_" << to_string(i) << ")";

                    if(i != neurons_number-1) buffer << " + ";
                }

                buffer << ";\n" << endl;

                for(Index i = 0; i < neurons_number; i++)
                {
                    buffer << outputs_name(i) << " = exp(probabilistic_layer_combinations_" << to_string(i) <<")/sum;\n";
                }

            }
            break;
        default:
            break;
        }
    }

    return buffer.str();
}


string ProbabilisticLayer::write_expression(const Tensor<string, 1>& inputs_names,
                                            const Tensor<string, 1>& outputs_name) const
{
    ostringstream buffer;

    buffer << write_combinations(inputs_names);

    buffer << write_activations(outputs_name);

    return buffer.str();
}

ProbabilisticLayerForwardPropagation::ProbabilisticLayerForwardPropagation()
    : LayerForwardPropagation()
{

}


ProbabilisticLayerForwardPropagation::ProbabilisticLayerForwardPropagation(
    const Index& new_batch_samples_number, Layer *new_layer)
    : LayerForwardPropagation()
{
    set(new_batch_samples_number, new_layer);
}


ProbabilisticLayerForwardPropagation::~ProbabilisticLayerForwardPropagation()
{

}


pair<type *, dimensions> ProbabilisticLayerForwardPropagation::get_outputs_pair() const
{
    const Index neurons_number = layer->get_neurons_number();

    return pair<type *, dimensions>(outputs_data, {{batch_samples_number, neurons_number}});
}


void ProbabilisticLayerForwardPropagation::set(const Index &new_batch_samples_number, Layer *new_layer) 
{
    layer = new_layer;

    batch_samples_number = new_batch_samples_number;

    const Index neurons_number = layer->get_neurons_number();

    outputs.resize(batch_samples_number, neurons_number);

    outputs_data = outputs.data();

    activations_derivatives.resize(0, 0);

    if(neurons_number == 1)
    {
        activations_derivatives.resize(batch_samples_number, neurons_number);
    }
    else
    {
        aux_rows.resize(batch_samples_number);
    }
}


void ProbabilisticLayerForwardPropagation::print() const 
{
    cout << "Probabilistic layer forward-propagation" << endl;

    cout << "Outputs:" << endl;
    cout << outputs << endl;

    const Index neurons_number = layer->get_neurons_number();

    if(neurons_number == 1)
    {
        cout << "Activations derivatives:" << endl;
        cout << activations_derivatives << endl;
    }
}


ProbabilisticLayerBackPropagation::ProbabilisticLayerBackPropagation() : LayerBackPropagation() 
{
}


ProbabilisticLayerBackPropagation::~ProbabilisticLayerBackPropagation() 
{

}


ProbabilisticLayerBackPropagation::ProbabilisticLayerBackPropagation(const Index &new_batch_samples_number, Layer *new_layer)
    : LayerBackPropagation() 
{
    set(new_batch_samples_number, new_layer);
}


void ProbabilisticLayerBackPropagation::set(const Index &new_batch_samples_number, Layer *new_layer) 
{
    layer = new_layer;

    batch_samples_number = new_batch_samples_number;

    const Index neurons_number = layer->get_neurons_number();
    const Index inputs_number = layer->get_inputs_number();

    if(neurons_number > 1)
        targets.resize(batch_samples_number, neurons_number);

    biases_derivatives.resize(neurons_number);

    synaptic_weights_derivatives.resize(inputs_number, neurons_number);

    deltas_row.resize(neurons_number);
    activations_derivatives_matrix.resize(neurons_number, neurons_number);

    error_combinations_derivatives.resize(batch_samples_number, neurons_number);

    error_combinations_derivatives.resize(batch_samples_number, neurons_number);

    input_derivatives.resize(batch_samples_number, inputs_number);

    inputs_derivatives.resize(1);
    inputs_derivatives(0).first = input_derivatives.data();
    inputs_derivatives(0).second = { batch_samples_number, inputs_number };
}


void ProbabilisticLayerBackPropagation::print() const 
{
    cout << "Biases derivatives:" << endl;
    cout << biases_derivatives << endl;

    cout << "Synaptic weights derivatives:" << endl;
    cout << synaptic_weights_derivatives << endl;
}


void ProbabilisticLayerBackPropagationLM::set(const Index& new_batch_samples_number, Layer* new_layer)
{
    layer = new_layer;

    batch_samples_number = new_batch_samples_number;

    const Index neurons_number = layer->get_neurons_number();
    const Index parameters_number = layer->get_parameters_number();

    //deltas.resize(batch_samples_number, neurons_number);
    deltas_row.resize(neurons_number);

    squared_errors_Jacobian.resize(batch_samples_number, parameters_number);

    error_combinations_derivatives.resize(batch_samples_number, neurons_number);
}
} // namespace opennn

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
