//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   P E R C E P T R O N   L A Y E R   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "perceptron_layer.h"
#include "tensors.h"
#include "strings_utilities.h"

namespace opennn
{

Perceptron::Perceptron(const dimensions& new_input_dimensions,
                       const dimensions& new_output_dimensions,
                       const Activation& new_activation_function,
                       const string& new_layer_name) : Layer()
{
    set(new_input_dimensions,
        new_output_dimensions,
        new_activation_function,
        new_layer_name);
}


dimensions Perceptron::get_input_dimensions() const
{
    return { weights.dimension(0) };
}


dimensions Perceptron::get_output_dimensions() const
{
    return { biases.size() };
}


void Perceptron::set_dropout_rate(const type& new_dropout_rate)
{
    dropout_rate = new_dropout_rate;
}


Index Perceptron::get_parameters_number() const
{
    return biases.size() + weights.size();
}


type Perceptron::get_dropout_rate() const
{
    return dropout_rate;
}


Tensor<type, 1> Perceptron::get_parameters() const
{
    const Index parameters_number = get_parameters_number();

    Tensor<type, 1> parameters(parameters_number);

    Index index = 0;

    copy_to_vector(parameters, weights, index);
    copy_to_vector(parameters, biases, index);

    return parameters;
}


const Perceptron::Activation& Perceptron::get_activation_function() const
{
    return activation_function;
}


string Perceptron::get_activation_function_string() const
{
    switch(activation_function)
    {
    case Activation::Logistic:
        return "Logistic";

    case Activation::HyperbolicTangent:
        return "HyperbolicTangent";

    case Activation::Linear:
        return "Linear";

    case Activation::RectifiedLinear:
        return "RectifiedLinear";

    case Activation::ScaledExponentialLinear:
        return "ScaledExponentialLinear";

    case Activation::SoftPlus:
        return "SoftPlus";

    case Activation::SoftSign:
        return "SoftSign";

    case Activation::HardSigmoid:
        return "HardSigmoid";

    case Activation::ExponentialLinear:
        return "ExponentialLinear";
    }

    return string();
}


void Perceptron::set(const dimensions& new_input_dimensions,
                          const dimensions& new_output_dimensions,
                          const Perceptron::Activation& new_activation_function,
                          const string& new_name)
{
    if (new_input_dimensions.size() != 1)
        throw runtime_error("Input dimensions size is not 1");

    if (new_output_dimensions.size() != 1)
        throw runtime_error("Output dimensions size is not 1");   

    biases.resize(new_output_dimensions[0]);    
    weights.resize(new_input_dimensions[0], new_output_dimensions[0]);    

    set_parameters_random();

    set_activation_function(new_activation_function);

    set_name(new_name);
    
    layer_type = Layer::Type::Perceptron;
}


void Perceptron::set_input_dimensions(const dimensions& new_input_dimensions)
{
    const Index inputs_number = new_input_dimensions[0];
    const Index outputs_number = get_outputs_number();

    biases.resize(outputs_number);

    weights.resize(inputs_number, outputs_number);
}


void Perceptron::set_output_dimensions(const dimensions& new_output_dimensions)
{
    const Index inputs_number = get_inputs_number();
    const Index neurons_number = new_output_dimensions[0];

    biases.resize(neurons_number);

    weights.resize(inputs_number, neurons_number);
}


void Perceptron::set_parameters(const Tensor<type, 1>& new_parameters, Index& index)
{   
    copy_from_vector(weights, new_parameters, index);
    copy_from_vector(biases, new_parameters, index);
}


void Perceptron::set_activation_function(const Perceptron::Activation& new_activation_function)
{
    activation_function = new_activation_function;
}


void Perceptron::set_activation_function(const string& new_activation_function_name)
{
    if(new_activation_function_name == "Logistic")
        activation_function = Activation::Logistic;
    else if(new_activation_function_name == "HyperbolicTangent")
        activation_function = Activation::HyperbolicTangent;
    else if(new_activation_function_name == "Linear")
        activation_function = Activation::Linear;
    else if(new_activation_function_name == "RectifiedLinear")
        activation_function = Activation::RectifiedLinear;
    else if(new_activation_function_name == "ScaledExponentialLinear")
        activation_function = Activation::ScaledExponentialLinear;
    else if(new_activation_function_name == "SoftPlus")
        activation_function = Activation::SoftPlus;
    else if(new_activation_function_name == "SoftSign")
        activation_function = Activation::SoftSign;
    else if(new_activation_function_name == "HardSigmoid")
        activation_function = Activation::HardSigmoid;
    else if(new_activation_function_name == "ExponentialLinear")
        activation_function = Activation::ExponentialLinear;
    else
        throw runtime_error("Unknown activation function: " + new_activation_function_name + ".\n");
}


void Perceptron::set_parameters_constant(const type& value)
{
    biases.setConstant(value);

    weights.setConstant(value);
}


void Perceptron::set_parameters_random()
{
    set_random(biases);

    set_random(weights);
}


void Perceptron::calculate_combinations(const Tensor<type, 2>& inputs,
                                        Tensor<type, 2>& combinations) const
{
    const Index batch_size = combinations.dimension(0);
    const Index outputs_number = biases.size();

    combinations.device(*thread_pool_device)
        = inputs.contract(weights, axes(1,0))
        + biases.reshape(array<Index, 2>({1, outputs_number}))
                .broadcast(array<Index, 2>({batch_size, 1}));
}

/*
void Perceptron::batch_normalization(Tensor<type, 1>& means, 
    Tensor<type, 1>& standard_deviations,
    const Tensor<type, 2>& inputs,
    Tensor<type, 2>& outputs) const
{
    const array<Index, 2> rows({outputs.dimension(0), 1 });

    const array<int, 1> axis_x({ 0 });

    means.device(*thread_pool_device) = outputs.mean(axis_x);

    standard_deviations.device(*thread_pool_device) 
        = (outputs - means.broadcast(rows)).square().mean(axis_x).sqrt();
    

    outputs = 
        shifts.broadcast(rows) 
        + (outputs - means.broadcast(rows))*scales.broadcast(rows)/standard_deviations.broadcast(rows);               

}
*/

void Perceptron::calculate_activations(Tensor<type, 2>& activations,
                                       Tensor<type, 2>& activation_derivatives) const
{
    switch(activation_function)
    {
    case Activation::Linear: linear(activations, activation_derivatives); return;

    case Activation::Logistic: logistic(activations, activation_derivatives);return;

    case Activation::HyperbolicTangent: hyperbolic_tangent(activations, activation_derivatives); return;

    case Activation::RectifiedLinear: rectified_linear(activations, activation_derivatives); return;

    case Activation::ScaledExponentialLinear: scaled_exponential_linear(activations, activation_derivatives); return;

    case Activation::SoftPlus: soft_plus(activations, activation_derivatives);return;

    case Activation::SoftSign: soft_sign(activations, activation_derivatives); return;

    case Activation::HardSigmoid: hard_sigmoid(activations, activation_derivatives); return;

    case Activation::ExponentialLinear: exponential_linear(activations, activation_derivatives); return;

    default: return;
    }
}


void Perceptron::forward_propagate(const vector<pair<type*, dimensions>>& input_pairs,
                                   unique_ptr<LayerForwardPropagation>& layer_forward_propagation,
                                   const bool& is_training)
{
    const TensorMap<Tensor<type, 2>> inputs = tensor_map_2(input_pairs[0]);

    PerceptronForwardPropagation* perceptron_layer_forward_propagation =
        static_cast<PerceptronForwardPropagation*>(layer_forward_propagation.get());

    Tensor<type, 2>& outputs = perceptron_layer_forward_propagation->outputs;

    calculate_combinations(inputs,
                           outputs);

    if(is_training)
    {
        Tensor<type, 2>& activation_derivatives = perceptron_layer_forward_propagation->activation_derivatives;

        calculate_activations(outputs, activation_derivatives);
    }
    else
    {
        calculate_activations(outputs, empty_2);
    }

    // @todo
// if(is_training && dropout_rate > type(0))
//     dropout(outputs);

}


void Perceptron::back_propagate(const vector<pair<type*, dimensions>>& input_pairs,
                                const vector<pair<type*, dimensions>>& delta_pairs,
                                unique_ptr<LayerForwardPropagation>& forward_propagation,
                                unique_ptr<LayerBackPropagation>& back_propagation) const
{
    const TensorMap<Tensor<type, 2>> inputs = tensor_map_2(input_pairs[0]);
    TensorMap<Tensor<type, 2>> deltas = tensor_map_2(delta_pairs[0]);

    // Forward propagation

    const PerceptronForwardPropagation* perceptron_layer_forward_propagation =
        static_cast<PerceptronForwardPropagation*>(forward_propagation.get());

    const Tensor<type, 2>& activation_derivatives = perceptron_layer_forward_propagation->activation_derivatives;

    // Back propagation

    PerceptronBackPropagation* perceptron_back_propagation =
        static_cast<PerceptronBackPropagation*>(back_propagation.get());
    
    Tensor<type, 2>& weight_derivatives = perceptron_back_propagation->weight_derivatives;

    Tensor<type, 1>& bias_derivatives = perceptron_back_propagation->bias_derivatives;

    const bool& is_first_layer = perceptron_back_propagation->is_first_layer;

    Tensor<type, 2>& input_derivatives = perceptron_back_propagation->input_derivatives;

    deltas.device(*thread_pool_device) = deltas * activation_derivatives;

    bias_derivatives.device(*thread_pool_device) = deltas.sum(array<Index, 1>({0}));

    weight_derivatives.device(*thread_pool_device) = inputs.contract(deltas, axes(0,0));

    if (!is_first_layer)
        input_derivatives.device(*thread_pool_device) = deltas.contract(weights, axes(0,0));
}


void Perceptron::back_propagate_lm(const vector<pair<type*, dimensions>>& input_pairs,
                                        const vector<pair<type*, dimensions>>& delta_pairs,
                                        unique_ptr<LayerForwardPropagation>& forward_propagation,
                                        unique_ptr<LayerBackPropagationLM>& back_propagation) const
{
    const TensorMap<Tensor<type, 2>> inputs = tensor_map_2(input_pairs[0]);
    TensorMap<Tensor<type, 2>> deltas = tensor_map_2(delta_pairs[0]);
    
    const Index inputs_number = get_inputs_number();
    const Index outputs_number = get_outputs_number();

    const Index weights_number = weights.size();

    // Forward propagation

    const PerceptronForwardPropagation* perceptron_layer_forward_propagation =
        static_cast<PerceptronForwardPropagation*>(forward_propagation.get());

    const Tensor<type, 2>& activation_derivatives
        = perceptron_layer_forward_propagation->activation_derivatives;

    // Back propagation

    PerceptronLayerBackPropagationLM* perceptron_layer_back_propagation_lm =
        static_cast<PerceptronLayerBackPropagationLM*>(back_propagation.get());

    Tensor<type, 2>& squared_errors_Jacobian = perceptron_layer_back_propagation_lm->squared_errors_Jacobian;

    const bool& is_first_layer = perceptron_layer_back_propagation_lm->is_first_layer;

    Tensor<type, 2>& input_derivatives = perceptron_layer_back_propagation_lm->input_derivatives;

    deltas.device(*thread_pool_device) = deltas * activation_derivatives;

    Index weight_index = 0;

    for(Index neuron_index = 0; neuron_index < outputs_number; neuron_index++)
    {
        const Tensor<type, 1> combination_delta_neuron = tensor_map_(deltas, neuron_index);

        for(Index input_index = 0; input_index < inputs_number; input_index++)
        {
            const Tensor<type, 1> input = inputs.chip(input_index,1);

            TensorMap<Tensor<type, 1>> squared_errors_jacobian_synaptic_weight 
                = tensor_map(squared_errors_Jacobian, weight_index++);

            squared_errors_jacobian_synaptic_weight.device(*thread_pool_device) 
                = combination_delta_neuron * input;
        }

        const Index bias_index = weights_number + neuron_index;

        TensorMap<Tensor<type, 1>> squared_errors_jacobian_bias 
            = tensor_map(squared_errors_Jacobian, bias_index);

        squared_errors_jacobian_bias.device(*thread_pool_device) = combination_delta_neuron;
    }

    if(!is_first_layer)
        input_derivatives.device(*thread_pool_device) = deltas.contract(weights, axes(1,1));
}


void Perceptron::insert_gradient(unique_ptr<LayerBackPropagation>& back_propagation,
                                 Index& index,
                                 Tensor<type, 1>& gradient) const
{
    PerceptronBackPropagation* perceptron_back_propagation =
        static_cast<PerceptronBackPropagation*>(back_propagation.get());

    copy_to_vector(gradient, perceptron_back_propagation->weight_derivatives, index);
    copy_to_vector(gradient, perceptron_back_propagation->bias_derivatives, index);
}


void Perceptron::insert_squared_errors_Jacobian_lm(unique_ptr<LayerBackPropagationLM>& back_propagation,
                                                   const Index& index,
                                                   Tensor<type, 2>& squared_errors_Jacobian) const
{
    const Index parameters_number = get_parameters_number();
    const Index batch_size = back_propagation->batch_size;

    PerceptronLayerBackPropagationLM* perceptron_layer_back_propagation_lm =
        static_cast<PerceptronLayerBackPropagationLM*>(back_propagation.get());

    type* this_squared_errors_Jacobian_data = perceptron_layer_back_propagation_lm->squared_errors_Jacobian.data();

    memcpy(squared_errors_Jacobian.data() + index,
           this_squared_errors_Jacobian_data,
           parameters_number * batch_size * sizeof(type));
}


string Perceptron::get_expression(const vector<string>& new_input_names,
                                       const vector<string>& new_output_names) const
{
    const vector<string> input_names = new_input_names.empty()
       ? get_default_input_names()
       : new_input_names;

    const vector<string> output_names = new_output_names.empty()
        ? get_default_output_names()
        : new_output_names;

    const Index inputs_number = get_inputs_number();
    const Index outputs_number = get_outputs_number();

    ostringstream buffer;

    for(Index j = 0; j < outputs_number; j++)
    {
        const TensorMap<Tensor<type, 1>> weights_column = tensor_map(weights, j);

        buffer << output_names[j] << " = " << get_activation_function_string_expression() << "( " << biases(j) << " + ";

        for(Index i = 0; i < inputs_number - 1; i++)
            buffer << "(" << weights_column(i) << "*" << input_names[i] << ") + ";

        buffer << "("  << weights_column(inputs_number - 1) << "*" << input_names[inputs_number - 1]  << ") );\n";
    }

    return buffer.str();
}


void Perceptron::print() const
{
    cout << "Perceptron layer" << endl
         << "Input dimensions: " << get_input_dimensions()[0] << endl
         << "Output dimensions: " << get_output_dimensions()[0] << endl
         << "Biases dimensions: " << biases.dimensions() << endl
         << "Synaptic weights dimensions: " << weights.dimensions() << endl;

    cout << "Biases:" << endl;
    cout << biases << endl;
    cout << "Synaptic weights:" << endl;
    cout << weights << endl;

    cout << "Activation function:" << endl;
    cout << get_activation_function_string() << endl;
}


void Perceptron::from_XML(const XMLDocument& document)
{
    const XMLElement* perceptron_layer_element = document.FirstChildElement("Perceptron");

    if(!perceptron_layer_element)
        throw runtime_error("Perceptron element is nullptr.\n");

    set_name(read_xml_string(perceptron_layer_element, "Name"));
    set_input_dimensions({ read_xml_index(perceptron_layer_element, "InputsNumber") });
    set_output_dimensions({ read_xml_index(perceptron_layer_element, "NeuronsNumber") });
    set_activation_function(read_xml_string(perceptron_layer_element, "Activation"));

    Index index = 0;

    set_parameters(to_type_vector(read_xml_string(perceptron_layer_element, "Parameters"), " "), index);
}


void Perceptron::to_XML(XMLPrinter& printer) const
{
    printer.OpenElement("Perceptron");

    add_xml_element(printer, "Name", name);
    add_xml_element(printer, "InputsNumber", to_string(get_input_dimensions()[0]));
    add_xml_element(printer, "NeuronsNumber", to_string(get_output_dimensions()[0]));
    add_xml_element(printer, "Activation", get_activation_function_string());
    add_xml_element(printer, "Parameters", tensor_to_string(get_parameters()));

    printer.CloseElement();  
}


string Perceptron::get_activation_function_string_expression() const
{
    switch(activation_function)
    {
    case Activation::Logistic:
        return "logistic";

    case Activation::HyperbolicTangent:
        return "tanh";

    case Activation::Linear:
        return string();

    case Activation::RectifiedLinear:
        return "ReLU";

    case Activation::ExponentialLinear:
        return "ELU";

    case Activation::ScaledExponentialLinear:
        return "SELU";

    case Activation::SoftPlus:
        return "soft_plus";

    case Activation::SoftSign:
        return "soft_sign";

    case Activation::HardSigmoid:
        return "hard_sigmoid";

    default:
        return string();
    }
}


void PerceptronForwardPropagation::set(const Index& new_batch_size, Layer *new_layer)
{
    layer = new_layer;
    
    batch_size = new_batch_size;

    if (!layer) return;

    const Index outputs_number = layer->get_outputs_number();

    outputs.resize(batch_size, outputs_number);

    activation_derivatives.resize(batch_size, outputs_number);

    activation_derivatives.setConstant((type)NAN);
}


pair<type *, dimensions> PerceptronForwardPropagation::get_outputs_pair() const
{
    const dimensions output_dimensions = layer->get_output_dimensions();
    
    return pair<type *, dimensions>((type*)outputs.data(), {{batch_size, output_dimensions[0]}});
}


PerceptronForwardPropagation::PerceptronForwardPropagation(const Index &new_batch_size,
                                                                     Layer *new_layer)
: LayerForwardPropagation()
{
    set(new_batch_size, new_layer);
}


void PerceptronForwardPropagation::print() const
{
    cout << "Outputs:" << endl
         << outputs << endl
         << "Activation derivatives:" << endl
         << activation_derivatives << endl;
}


PerceptronBackPropagation::PerceptronBackPropagation(const Index &new_batch_size, Layer *new_layer)
    : LayerBackPropagation()
{
    set(new_batch_size, new_layer);
}


void PerceptronBackPropagation::set(const Index&new_batch_size,
                                    Layer *new_layer)
{
    layer = new_layer;
    
    batch_size = new_batch_size;
    
    const Index outputs_number = layer->get_outputs_number();
    const Index inputs_number = layer->get_input_dimensions()[0];

    bias_derivatives.resize(outputs_number);
    bias_derivatives.setZero();

    weight_derivatives.resize(inputs_number, outputs_number);
    weight_derivatives.setZero();

    input_derivatives.resize(batch_size, inputs_number);
}


vector<pair<type*, dimensions>> PerceptronBackPropagation::get_input_derivative_pairs() const
{
    const Index inputs_number = layer->get_input_dimensions()[0];

    return { {(type*)(input_derivatives.data()), {batch_size, inputs_number}} };
}


void PerceptronBackPropagation::print() const
{
    cout << "Biases derivatives:" << endl
         << bias_derivatives << endl
         << "Synaptic weights derivatives:" << endl
         << weight_derivatives << endl;
}


PerceptronLayerBackPropagationLM::PerceptronLayerBackPropagationLM(const Index &new_batch_size,
                                                                   Layer *new_layer)
    : LayerBackPropagationLM()
{
    set(new_batch_size, new_layer);
}


void PerceptronLayerBackPropagationLM::set(const Index &new_samples_number, Layer *new_layer)
{
    layer = new_layer;

    batch_size = new_samples_number;

    const Index inputs_number = layer->get_input_dimensions()[0];
    const Index parameters_number = layer->get_parameters_number();

    squared_errors_Jacobian.resize(batch_size, parameters_number);

    input_derivatives.resize(batch_size, inputs_number);
}


vector<pair<type*, dimensions>> PerceptronLayerBackPropagationLM::get_input_derivative_pairs() const
{
    const Index inputs_number = layer->get_input_dimensions()[0];

    return {{(type*)(input_derivatives.data()), {batch_size, inputs_number}}};
}


void PerceptronLayerBackPropagationLM::print() const
{
    cout << "Squared errors Jacobian: " << endl
        << squared_errors_Jacobian << endl;
    cout << "Input derivatives: " << endl
        << input_derivatives << endl;
}


#ifdef OPENNN_CUDA_test

void Perceptron::forward_propagate_cuda(const vector<pair<type*, dimensions>>& inputs_pair_device,
                                        unique_ptr<LayerForwardPropagationCuda>& forward_propagation_cuda,
                                        const bool& is_training) //final
{
    // Perceptron layer

    const Index inputs_number = get_inputs_number();
    const Index outputs_number = get_outputs_number();

    // Inputs

    const Index batch_samples_number = inputs_pair_device[0].second[0];

    const type* inputs_device = inputs_pair_device[0].first;

    // Forward propagation

    PerceptronLayerForwardPropagationCuda* perceptron_layer_forward_propagation_cuda =
        static_cast<PerceptronLayerForwardPropagationCuda*>(forward_propagation_cuda.get());

    Perceptron* perceptron_layer = static_cast<Perceptron*>(perceptron_layer_forward_propagation_cuda->layer);

    type* combinations = perceptron_layer_forward_propagation_cuda->combinations;
    type* outputs = perceptron_layer_forward_propagation_cuda->outputs;

    const cudnnActivationDescriptor_t& activation_descriptor = perceptron_layer_forward_propagation_cuda->activation_descriptor;

    const cudnnTensorDescriptor_t& outputs_tensor_descriptor = perceptron_layer_forward_propagation_cuda->outputs_tensor_descriptor;
    const cudnnTensorDescriptor_t& outputs_batch_tensor_descriptor = perceptron_layer_forward_propagation_cuda->outputs_batch_tensor_descriptor;
    const cudnnTensorDescriptor_t& biases_batch_tensor_descriptor = perceptron_layer_forward_propagation_cuda->biases_batch_tensor_descriptor;

    // Combinations

    const float alpha = 1.0f;
    const float beta = 0.0f;

    cublasSgemm(cublas_handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        batch_samples_number, outputs_number, inputs_number,
        &alpha,
        inputs_device,
        batch_samples_number,
        weights_device,
        inputs_number,
        &beta,
        combinations,
        batch_samples_number);

    // @todo Improve by using cudnnAddTensor
    for (Index biases_index = 0; biases_index < outputs_number; biases_index++)
    {
        type* outputs_batch = combinations + biases_index * batch_samples_number;
        type* biases_batch = biases_device + biases_index;

        cudnnOpTensor(cudnn_handle,
            operator_sum_descriptor,
            &alpha,
            outputs_batch_tensor_descriptor,
            outputs_batch,
            &alpha,
            biases_batch_tensor_descriptor,
            biases_batch,
            &beta,
            outputs_batch_tensor_descriptor,
            outputs_batch);
    }

    // Activations

    if (perceptron_layer->get_activation_function() != Activation::Linear)
    {
        cudnnStatus_t activationStatus = cudnnActivationForward(cudnn_handle,
            activation_descriptor,
            &alpha,
            outputs_tensor_descriptor,
            combinations,
            &beta,
            outputs_tensor_descriptor,
            outputs);

        if (activationStatus != CUDNN_STATUS_SUCCESS)
            cout << "cudnnActivationForward failed: " << cudnnGetErrorString(activationStatus) << endl;
    }
    else
        cudaMemcpy(outputs, combinations, batch_samples_number * outputs_number * sizeof(type), cudaMemcpyDeviceToDevice);
}


void Perceptron::back_propagate_cuda(const vector<pair<type*, dimensions>>& inputs_pair_device,
                                     const vector<pair<type*, dimensions>>& deltas_pair_device,
                                     unique_ptr<LayerForwardPropagationCuda>& forward_propagation_cuda,
                                     unique_ptr<LayerBackPropagationCuda>& back_propagation_cuda) const
{
    // Perceptron layer

    const Index inputs_number = get_inputs_number();
    const Index outputs_number = get_outputs_number();

    // Inputs

    const Index batch_samples_number = inputs_pair_device[0].second[0];

    const type* inputs_device = inputs_pair_device[0].first;
    const type* deltas_device = deltas_pair_device[0].first;

    // Forward propagation

    PerceptronLayerForwardPropagationCuda* perceptron_layer_forward_propagation_cuda =
        static_cast<PerceptronLayerForwardPropagationCuda*>(forward_propagation_cuda.get());

    Perceptron* perceptron_layer = static_cast<Perceptron*>(perceptron_layer_forward_propagation_cuda->layer);

    float* combinations = perceptron_layer_forward_propagation_cuda->combinations;
    float* outputs = perceptron_layer_forward_propagation_cuda->outputs;

    const cudnnActivationDescriptor_t& activation_descriptor = perceptron_layer_forward_propagation_cuda->activation_descriptor;

    // Back propagation

    PerceptronLayerBackPropagationCuda* perceptron_layer_back_propagation =
        static_cast<PerceptronLayerBackPropagationCuda*>(back_propagation_cuda.get());

    float* ones = perceptron_layer_back_propagation->ones;
    float* error_combinations_derivatives = perceptron_layer_back_propagation->error_combinations_derivatives_cuda;
    float* biases_derivatives = perceptron_layer_back_propagation->biases_derivatives_cuda;
    float* weights_derivatives = perceptron_layer_back_propagation->weights_derivatives_cuda;
    float* inputs_derivatives = perceptron_layer_back_propagation->inputs_derivatives;

    const cudnnTensorDescriptor_t& deltas_tensor_descriptor = perceptron_layer_back_propagation->deltas_tensor_descriptor;
    const cudnnTensorDescriptor_t& error_combinations_derivatives_tensor_descriptor = perceptron_layer_back_propagation->error_combinations_derivatives_tensor_descriptor;

    const float alpha = 1.0f;
    float beta = 0.0f;

    // Error combinations derivatives

    if (perceptron_layer->get_activation_function() != Activation::Linear)
        cudnnActivationBackward(cudnn_handle,
            activation_descriptor,
            &alpha,
            error_combinations_derivatives_tensor_descriptor,
            outputs,
            deltas_tensor_descriptor,
            deltas_device,
            error_combinations_derivatives_tensor_descriptor,
            combinations,
            &beta,
            error_combinations_derivatives_tensor_descriptor,
            error_combinations_derivatives);
    else
        cudaMemcpy(error_combinations_derivatives, deltas_device, batch_samples_number * outputs_number * sizeof(type), cudaMemcpyDeviceToDevice);

    // Bias derivatives 
    //// @todo  Use cudnnReduceTensor instead of contract of ones
    cublasSgemm(cublas_handle,
        CUBLAS_OP_T, CUBLAS_OP_N,
        outputs_number,
        1,
        batch_samples_number,
        &alpha,
        error_combinations_derivatives,
        batch_samples_number,
        ones,
        batch_samples_number,
        &beta,
        biases_derivatives,
        outputs_number);

    // Synaptic weights derivatives

    cublasSgemm(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N,
        inputs_number,
        outputs_number,
        batch_samples_number,
        &alpha,
        inputs_device,
        batch_samples_number,
        error_combinations_derivatives,
        batch_samples_number,
        &beta,
        weights_derivatives,
        inputs_number);

    // Input derivatives

    cublasSgemm(cublas_handle,
        CUBLAS_OP_N, CUBLAS_OP_T,
        batch_samples_number,
        inputs_number,
        outputs_number,
        &alpha,
        error_combinations_derivatives,
        batch_samples_number,
        weights_device,
        inputs_number,
        &beta,
        inputs_derivatives,
        batch_samples_number);
}


void Perceptron::insert_gradient_cuda(unique_ptr<LayerBackPropagationCuda>& back_propagation_cuda, 
                                      Index& index, 
                                      float* gradient) const
{
    // Perceptron layer

    const Index weights_number = weights.size();
    const Index biases_number = biases.size();

    // Perceptron layer back propagation cuda

    PerceptronLayerBackPropagationCuda* perceptron_layer_back_propagation =
        static_cast<PerceptronLayerBackPropagationCuda*>(back_propagation_cuda.get());

    type* weights_derivatives_cuda = perceptron_layer_back_propagation->weights_derivatives_cuda;

    type* biases_derivatives_cuda = perceptron_layer_back_propagation->biases_derivatives_cuda;

    if (cudaMemcpy(gradient + index,
        weights_derivatives_cuda,
        size_t(weights_number) * sizeof(type),
        cudaMemcpyDeviceToDevice) != cudaSuccess)
        cout << "gradient (weights) copy error" << endl;

    if (cudaMemcpy(gradient + index + weights_number,
        biases_derivatives_cuda,
        size_t(biases_number) * sizeof(type),
        cudaMemcpyDeviceToDevice) != cudaSuccess)
        cout << "gradient (biases) copy error" << endl;
}


void Perceptron::set_parameters_cuda(const float* new_parameters, const Index& index)
{
    const Index weights_number = weights.size();
    const Index biases_number = biases.size();

    if (cudaMemcpy(weights_device,
        new_parameters + index,
        size_t(weights_number) * sizeof(type),
        cudaMemcpyDeviceToDevice) != cudaSuccess)
        cout << "biases copy error" << endl;

    if (cudaMemcpy(biases_device,
        new_parameters + weights_number + index,
        size_t(biases_number) * sizeof(type),
        cudaMemcpyDeviceToDevice) != cudaSuccess)
        cout << "synaptic weights copy error" << endl;
}


void Perceptron::get_parameters_cuda(const Tensor<type, 1>& new_parameters, const Index& index)
{
    /*
    const Index weights_number = get_weights_number();
    const Index biases_number = get_biases_number();

    if (cudaMemcpy(const_cast<void*>(static_cast<const void*>(new_parameters.data() + index)),
        weights_device, size_t(weights_number) * sizeof(type), cudaMemcpyDeviceToDevice) != cudaSuccess)
        cout << "new_parameters copy error" << endl;

    if (cudaMemcpy(const_cast<void*>(static_cast<const void*>(new_parameters.data() + biases_number + index)),
        biases_device, size_t(biases_number) * sizeof(type), cudaMemcpyDeviceToDevice) != cudaSuccess)
        cout << "new_parameters copy error" << endl;
        */
}


void Perceptron::allocate_parameters_device()
{
    const Index inputs_number = get_inputs_number();
    const Index outputs_number = get_outputs_number();

    if (cudaMalloc(&biases_device, outputs_number * sizeof(float)) != cudaSuccess)
        cout << "Biases allocation error" << endl;

    if (cudaMalloc(&weights_device, inputs_number * outputs_number * sizeof(float)) != cudaSuccess)
        cout << "Synaptic weights allocation error" << endl;
}


void Perceptron::free_parameters_device()
{
    cudaFree(biases_device);
    cudaFree(weights_device);

    biases_device = nullptr;
    weights_device = nullptr;
}


void Perceptron::copy_parameters_device()
{
    if (biases_device == nullptr)
        cout << "Biases device is null" << endl;

    if (weights_device == nullptr)
        cout << "Weights device is null" << endl;

    if (cudaMemcpy(biases_device, biases.data(), biases.size() * sizeof(type), cudaMemcpyHostToDevice) != cudaSuccess)
        cout << "Biases device copy error" << endl;

    if (cudaMemcpy(weights_device, weights.data(), weights.size() * sizeof(type), cudaMemcpyHostToDevice) != cudaSuccess)
        cout << "Weights device copy error" << endl;
}


void Perceptron::copy_parameters_host()
{
    if (biases_device == nullptr) 
        cout << "Biases is null" << endl;

    if (weights_device == nullptr) 
        cout << "Synaptic weights is null" << endl;

    if (cudaMemcpy(biases.data(), biases_device, biases.size() * sizeof(type), cudaMemcpyDeviceToHost) != cudaSuccess)
        cout << "Biases host copy error" << endl;

    if (cudaMemcpy(weights.data(), weights_device, weights.size() * sizeof(type), cudaMemcpyDeviceToHost) != cudaSuccess)
        cout << "Weights host copy error" << endl;
}


float* Perceptron::get_weights_device() const
{
    return weights_device;
}

float* Perceptron::get_biases_device() const
{
    return biases_device;
}


// CUDA structs

PerceptronLayerForwardPropagationCuda::PerceptronLayerForwardPropagationCuda(const Index& new_batch_samples_number, Layer* new_layer)
    : LayerForwardPropagationCuda()
{
    set(new_batch_samples_number, new_layer);
}


void PerceptronLayerForwardPropagationCuda::set(const Index& new_batch_samples_number, Layer* new_layer)
{
    batch_size = new_batch_samples_number;

    layer = new_layer;

    const Index outputs_number = layer->get_outputs_number();
    const Index inputs_number = layer->get_input_dimensions()[0];

    // Biases

    cudnnCreateTensorDescriptor(&biases_batch_tensor_descriptor);

    cudnnSetTensor4dDescriptor(biases_batch_tensor_descriptor,
        CUDNN_TENSOR_NCHW,
        CUDNN_DATA_FLOAT,
        1,
        1,
        1,
        1);

    // Outputs

    if (cudaMalloc(&combinations, batch_size * outputs_number * sizeof(float)) != cudaSuccess)
        cout << "combinations allocation error" << endl;

    if (cudaMalloc(&outputs, batch_size * outputs_number * sizeof(float)) != cudaSuccess)
        cout << "outputs allocation error" << endl;

    cudnnCreateTensorDescriptor(&outputs_tensor_descriptor);

    cudnnSetTensor4dDescriptor(outputs_tensor_descriptor,
        CUDNN_TENSOR_NCHW,
        CUDNN_DATA_FLOAT,
        batch_size,
        outputs_number,
        1,
        1);

    cudnnCreateTensorDescriptor(&outputs_batch_tensor_descriptor);

    cudnnSetTensor4dDescriptor(outputs_batch_tensor_descriptor,
        CUDNN_TENSOR_NCHW,
        CUDNN_DATA_FLOAT,
        batch_size,
        1,
        1,
        1);

    // Activations

    cudnnCreateActivationDescriptor(&activation_descriptor);

    Perceptron* perceptron_layer = static_cast<Perceptron*>(layer);

    switch (perceptron_layer->get_activation_function())
    {

    case Perceptron::Activation::Linear:
        cudnnSetActivationDescriptor(activation_descriptor, CUDNN_ACTIVATION_IDENTITY, CUDNN_PROPAGATE_NAN, 0.0);
        break;

    case Perceptron::Activation::Logistic:
        cudnnSetActivationDescriptor(activation_descriptor, CUDNN_ACTIVATION_SIGMOID, CUDNN_PROPAGATE_NAN, 0.0);
        break;

    case Perceptron::Activation::HyperbolicTangent:
        cudnnSetActivationDescriptor(activation_descriptor, CUDNN_ACTIVATION_TANH, CUDNN_PROPAGATE_NAN, 0.0);
        break;

    case Perceptron::Activation::RectifiedLinear:
        cudnnSetActivationDescriptor(activation_descriptor, CUDNN_ACTIVATION_RELU, CUDNN_PROPAGATE_NAN, 0.0);
        break;

    case Perceptron::Activation::ExponentialLinear:
        cudnnSetActivationDescriptor(activation_descriptor, CUDNN_ACTIVATION_ELU, CUDNN_PROPAGATE_NAN, 0.0);
        break;

        //case Perceptron::Activation::Swish:
        //    cudnnSetActivationDescriptor(activation_derivatives_descriptor, CUDNN_ACTIVATION_SWISH, CUDNN_PROPAGATE_NAN, 0.0);
        //    break;

        //case Perceptron::Activation::ClippedRectifiedLinear:
        //    cudnnSetActivationDescriptor(activation_derivatives_descriptor, CUDNN_ACTIVATION_CLIPPED_RELU, CUDNN_PROPAGATE_NAN, 0.0);
        //    break;

    }
}

void PerceptronLayerForwardPropagationCuda::print() const
{
    // @todo
}


void PerceptronLayerForwardPropagationCuda::free()
{
    cudaFree(outputs);

    cudnnDestroyActivationDescriptor(activation_descriptor);

    cudnnDestroyTensorDescriptor(outputs_tensor_descriptor);
    cudnnDestroyTensorDescriptor(outputs_batch_tensor_descriptor);
    cudnnDestroyTensorDescriptor(biases_batch_tensor_descriptor);
}


pair<type*, dimensions> PerceptronLayerForwardPropagationCuda::get_outputs_pair() const
{
    const dimensions output_dimensions = layer->get_output_dimensions();

    return pair<type*, dimensions>(outputs, { {batch_size, output_dimensions[0]} });
}


PerceptronLayerBackPropagationCuda::PerceptronLayerBackPropagationCuda(const Index& new_batch_samples_number, Layer* new_layer)
    : LayerBackPropagationCuda()
{
    set(new_batch_samples_number, new_layer);
}


void PerceptronLayerBackPropagationCuda::set(const Index& new_batch_samples_number, Layer* new_layer)
{
    batch_size = new_batch_samples_number;

    layer = new_layer;

    const Index outputs_number = layer->get_outputs_number();
    const Index inputs_number = layer->get_input_dimensions()[0];

    // Ones

    if (cudaMalloc(&ones, batch_size * sizeof(float)) != cudaSuccess)
        cout << "ones allocation error" << endl;

    for (Index i = 0; i < batch_size; i++) {
        cudaMemcpy(ones + i, &one, sizeof(float), cudaMemcpyHostToDevice);
    }

    // biases_derivatives_cuda

    if (cudaMalloc(&biases_derivatives_cuda, outputs_number * sizeof(float)) != cudaSuccess)
        cout << "biases_derivatives perceptron allocation error" << endl;

    // weights_derivatives_cuda

    if (cudaMalloc(&weights_derivatives_cuda, inputs_number * outputs_number * sizeof(float)) != cudaSuccess)
        cout << "weights_derivatives allocation error" << endl;

    // Inputs derivatives

    if (cudaMalloc(&inputs_derivatives, batch_size * inputs_number * sizeof(float)) != cudaSuccess)
        cout << "inputs derivatives allocation error" << endl;

    // Deltas

    cudnnCreateTensorDescriptor(&deltas_tensor_descriptor);

    cudnnSetTensor4dDescriptor(deltas_tensor_descriptor,
        CUDNN_TENSOR_NHWC,
        CUDNN_DATA_FLOAT,
        batch_size,
        outputs_number,
        1,
        1);

    // Error combinations derivatives

    if (cudaMalloc(&error_combinations_derivatives_cuda, batch_size * outputs_number * sizeof(float)) != cudaSuccess)
        cout << "error combinations derivatives allocation error" << endl;

    cudnnCreateTensorDescriptor(&error_combinations_derivatives_tensor_descriptor);

    cudnnSetTensor4dDescriptor(error_combinations_derivatives_tensor_descriptor,
        CUDNN_TENSOR_NHWC,
        CUDNN_DATA_FLOAT,
        batch_size,
        outputs_number,
        1,
        1);

}


vector<pair<type*, dimensions>> PerceptronLayerBackPropagationCuda::get_input_derivative_pairs_device() const
{
    const Index inputs_number = layer->get_input_dimensions()[0];

    return { {inputs_derivatives, {batch_size, inputs_number}} };
}


void PerceptronLayerBackPropagationCuda::print() const
{
    // @todo
}


void PerceptronLayerBackPropagationCuda::free()
{
    cudaFree(biases_derivatives_cuda);
    cudaFree(weights_derivatives_cuda);
    cudaFree(error_combinations_derivatives_cuda);
    cudaFree(inputs_derivatives);
    cudaFree(ones);

    cudnnDestroyTensorDescriptor(error_combinations_derivatives_tensor_descriptor);
    cudnnDestroyTensorDescriptor(deltas_tensor_descriptor);

}

#endif

} // namespace opennn


// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2025 Artificial Intelligence Techniques, SL.
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
