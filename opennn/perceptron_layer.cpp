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

Dense2d::Dense2d(const dimensions& new_input_dimensions,
                 const dimensions& new_output_dimensions,
                 const Activation& new_activation_function,
                 const string& new_layer_name) : Layer()
{
    set(new_input_dimensions, new_output_dimensions, new_activation_function, new_layer_name);
}


dimensions Dense2d::get_input_dimensions() const
{
    return { weights.dimension(0) };
}


dimensions Dense2d::get_output_dimensions() const
{
    return { biases.size() };
}


void Dense2d::set_dropout_rate(const type& new_dropout_rate)
{
    if (new_dropout_rate < type(0) || new_dropout_rate >= type(1))
        throw runtime_error("Dropout rate must be in [0,1).");
    dropout_rate = new_dropout_rate;
}


Index Dense2d::get_parameters_number() const
{
    return biases.size() + weights.size();
}


type Dense2d::get_dropout_rate() const
{
    return dropout_rate;
}


Tensor<type, 1> Dense2d::get_parameters() const
{
    const Index parameters_number = get_parameters_number();

    Tensor<type, 1> parameters(parameters_number);

    Index index = 0;

    copy_to_vector(parameters, weights, index);
    copy_to_vector(parameters, biases, index);

    return parameters;
}


const Dense2d::Activation& Dense2d::get_activation_function() const
{
    return activation_function;
}


string Dense2d::get_activation_function_string() const
{
    switch(activation_function)
    {
    case Activation::Logistic: return "Logistic";
    case Activation::HyperbolicTangent: return "HyperbolicTangent";
    case Activation::Linear: return "Linear";
    case Activation::RectifiedLinear: return "RectifiedLinear";
    case Activation::ExponentialLinear: return "ExponentialLinear";
    case Activation::Softmax: return "Softmax";
    }

    return string();
}


void Dense2d::set(const dimensions& new_input_dimensions,
                  const dimensions& new_output_dimensions,
                  const Dense2d::Activation& new_activation_function,
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
    
    layer_type = Layer::Type::Dense2d;

    #ifdef OPENNN_CUDA

    // Activations
    
    if (activation_function != Activation::Softmax)
    {
        cudnnCreateActivationDescriptor(&activation_descriptor);

        cudnnActivationMode_t activation = CUDNN_ACTIVATION_IDENTITY;

        switch (get_activation_function())
        {
        case Activation::Linear: activation = CUDNN_ACTIVATION_IDENTITY; break;
        case Activation::Logistic: activation = CUDNN_ACTIVATION_SIGMOID; break;
        case Activation::HyperbolicTangent: activation = CUDNN_ACTIVATION_TANH; break;
        case Activation::RectifiedLinear: activation = CUDNN_ACTIVATION_RELU; break;
        case Activation::ExponentialLinear: activation = CUDNN_ACTIVATION_ELU; break;

        default: break;
        }

        cudnnSetActivationDescriptor(activation_descriptor, activation, CUDNN_PROPAGATE_NAN, 0.0);
    }

    #endif
}


void Dense2d::set_input_dimensions(const dimensions& new_input_dimensions)
{
    const Index inputs_number = new_input_dimensions[0];
    const Index outputs_number = get_outputs_number();

    biases.resize(outputs_number);

    weights.resize(inputs_number, outputs_number);
}


void Dense2d::set_output_dimensions(const dimensions& new_output_dimensions)
{
    const Index inputs_number = get_inputs_number();
    const Index neurons_number = new_output_dimensions[0];

    biases.resize(neurons_number);

    weights.resize(inputs_number, neurons_number);
}


void Dense2d::set_parameters(const Tensor<type, 1>& new_parameters, Index& index)
{   
    copy_from_vector(weights, new_parameters, index);
    copy_from_vector(biases, new_parameters, index);
}


void Dense2d::set_activation_function(const Dense2d::Activation& new_activation_function)
{
    if (new_activation_function == Activation::Softmax)
    {
        get_output_dimensions()[0] == 1
            ? activation_function = Activation::Logistic
            : activation_function = Activation::Softmax;
    }
    else
        activation_function = new_activation_function;
}


void Dense2d::set_activation_function(const string& new_activation_function_name)
{
    if(new_activation_function_name == "Logistic")
        activation_function = Activation::Logistic;
    else if(new_activation_function_name == "HyperbolicTangent")
        activation_function = Activation::HyperbolicTangent;
    else if(new_activation_function_name == "Linear")
        activation_function = Activation::Linear;
    else if(new_activation_function_name == "RectifiedLinear")
        activation_function = Activation::RectifiedLinear;
    else if(new_activation_function_name == "ExponentialLinear")
        activation_function = Activation::ExponentialLinear;
    else if(new_activation_function_name == "Softmax")
        activation_function = Activation::Softmax;
    else
        throw runtime_error("Unknown activation function: " + new_activation_function_name + ".\n");
}


void Dense2d::set_parameters_constant(const type& value)
{
    biases.setConstant(value);
    weights.setConstant(value);
}


void Dense2d::set_parameters_random()
{
    set_random(biases);
    set_random(weights);
}


void Dense2d::calculate_combinations(const Tensor<type, 2>& inputs,
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
void Dense2d::batch_normalization(Tensor<type, 1>& means,
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

void Dense2d::calculate_activations(Tensor<type, 2>& activations,
                                    Tensor<type, 2>& activation_derivatives) const
{
    switch(activation_function)
    {
    case Activation::Linear: linear(activations, activation_derivatives); return;
    case Activation::Logistic: logistic(activations, activation_derivatives);return;
    case Activation::HyperbolicTangent: hyperbolic_tangent(activations, activation_derivatives); return;
    case Activation::RectifiedLinear: rectified_linear(activations, activation_derivatives); return;
    case Activation::ExponentialLinear: exponential_linear(activations, activation_derivatives); return;
    case Activation::Softmax: softmax(activations); return;
    default: return;
    }
}


void Dense2d::forward_propagate(const vector<pair<type*, dimensions>>& input_pairs,
                                unique_ptr<LayerForwardPropagation>& layer_forward_propagation,
                                const bool& is_training)
{
    const TensorMap<Tensor<type, 2>> inputs = tensor_map_2(input_pairs[0]);

    Dense2dForwardPropagation* dense2d_forward_propagation =
        static_cast<Dense2dForwardPropagation*>(layer_forward_propagation.get());

    Tensor<type, 2>& outputs = dense2d_forward_propagation->outputs;

    calculate_combinations(inputs,
                           outputs);

    is_training
        ? calculate_activations(outputs, dense2d_forward_propagation->activation_derivatives)
        : calculate_activations(outputs, empty_2);

    if(is_training && dropout_rate > type(0))
        dropout(outputs, dropout_rate);
}


void Dense2d::back_propagate(const vector<pair<type*, dimensions>>& input_pairs,
                             const vector<pair<type*, dimensions>>& delta_pairs,
                             unique_ptr<LayerForwardPropagation>& forward_propagation,
                             unique_ptr<LayerBackPropagation>& back_propagation) const
{
    const TensorMap<Tensor<type, 2>> inputs = tensor_map_2(input_pairs[0]);
    TensorMap<Tensor<type, 2>> deltas = tensor_map_2(delta_pairs[0]);

    // Forward propagation

    const Dense2dForwardPropagation* dense2d_layer_forward_propagation =
        static_cast<Dense2dForwardPropagation*>(forward_propagation.get());

    const Tensor<type, 2>& activation_derivatives = dense2d_layer_forward_propagation->activation_derivatives;

    // Back propagation

    Dense2dBackPropagation* dense2d_back_propagation =
        static_cast<Dense2dBackPropagation*>(back_propagation.get());
    
    Tensor<type, 2>& weight_derivatives = dense2d_back_propagation->weight_derivatives;

    Tensor<type, 1>& bias_derivatives = dense2d_back_propagation->bias_derivatives;

    const bool& is_first_layer = dense2d_back_propagation->is_first_layer;

    Tensor<type, 2>& input_derivatives = dense2d_back_propagation->input_derivatives;

    if(activation_function != Activation::Softmax)
        deltas.device(*thread_pool_device) = deltas * activation_derivatives;

    bias_derivatives.device(*thread_pool_device) = deltas.sum(array<Index, 1>({0}));

    weight_derivatives.device(*thread_pool_device) = inputs.contract(deltas, axes(0,0));

    if (!is_first_layer)
        input_derivatives.device(*thread_pool_device) = deltas.contract(weights, axes(1,1));
}


void Dense2d::back_propagate_lm(const vector<pair<type*, dimensions>>& input_pairs,
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

    const Dense2dForwardPropagation* dense2d_layer_forward_propagation =
        static_cast<Dense2dForwardPropagation*>(forward_propagation.get());

    const Tensor<type, 2>& activation_derivatives
        = dense2d_layer_forward_propagation->activation_derivatives;

    // Back propagation

    Dense2dLayerBackPropagationLM* dense2d_layer_back_propagation_lm =
        static_cast<Dense2dLayerBackPropagationLM*>(back_propagation.get());

    Tensor<type, 2>& squared_errors_Jacobian = dense2d_layer_back_propagation_lm->squared_errors_Jacobian;

    const bool& is_first_layer = dense2d_layer_back_propagation_lm->is_first_layer;

    Tensor<type, 2>& input_derivatives = dense2d_layer_back_propagation_lm->input_derivatives;

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


void Dense2d::insert_gradient(unique_ptr<LayerBackPropagation>& back_propagation,
                                 Index& index,
                                 Tensor<type, 1>& gradient) const
{
    Dense2dBackPropagation* dense2d_back_propagation =
        static_cast<Dense2dBackPropagation*>(back_propagation.get());

    copy_to_vector(gradient, dense2d_back_propagation->weight_derivatives, index);
    copy_to_vector(gradient, dense2d_back_propagation->bias_derivatives, index);
}


void Dense2d::insert_squared_errors_Jacobian_lm(unique_ptr<LayerBackPropagationLM>& back_propagation,
                                                   const Index& index,
                                                   Tensor<type, 2>& squared_errors_Jacobian) const
{
    const Index parameters_number = get_parameters_number();
    const Index batch_size = back_propagation->batch_size;

    Dense2dLayerBackPropagationLM* dense2d_layer_back_propagation_lm =
        static_cast<Dense2dLayerBackPropagationLM*>(back_propagation.get());

    type* this_squared_errors_Jacobian_data = dense2d_layer_back_propagation_lm->squared_errors_Jacobian.data();

    memcpy(squared_errors_Jacobian.data() + index,
           this_squared_errors_Jacobian_data,
           parameters_number * batch_size * sizeof(type));
}


string Dense2d::get_expression(const vector<string>& new_input_names,
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

        buffer << "(" << weights_column(inputs_number - 1) << "*" << input_names[inputs_number - 1] << ") );\n";
    }

    return buffer.str();
}


void Dense2d::print() const
{
    cout << "Dense2d layer" << endl
         << "Input dimensions: " << get_input_dimensions()[0] << endl
         << "Output dimensions: " << get_output_dimensions()[0] << endl
         << "Biases dimensions: " << biases.dimensions() << endl
         << "Weights dimensions: " << weights.dimensions() << endl;

    cout << "Biases:" << endl;
    cout << biases << endl;
    cout << "Weights:" << endl;
    cout << weights << endl;

    cout << "Activation function:" << endl;
    cout << get_activation_function_string() << endl;
}


void Dense2d::from_XML(const XMLDocument& document)
{
    const XMLElement* dense2d_layer_element = document.FirstChildElement("Dense2d");

    if(!dense2d_layer_element)
        throw runtime_error("Dense2d element is nullptr.\n");

    set_name(read_xml_string(dense2d_layer_element, "Name"));
    set_input_dimensions({ read_xml_index(dense2d_layer_element, "InputsNumber") });
    set_output_dimensions({ read_xml_index(dense2d_layer_element, "NeuronsNumber") });
    set_activation_function(read_xml_string(dense2d_layer_element, "Activation"));

    Index index = 0;

    set_parameters(to_type_vector(read_xml_string(dense2d_layer_element, "Parameters"), " "), index);
}


void Dense2d::to_XML(XMLPrinter& printer) const
{
    printer.OpenElement("Dense2d");

    add_xml_element(printer, "Name", name);
    add_xml_element(printer, "InputsNumber", to_string(get_input_dimensions()[0]));
    add_xml_element(printer, "NeuronsNumber", to_string(get_output_dimensions()[0]));
    add_xml_element(printer, "Activation", get_activation_function_string());
    add_xml_element(printer, "Parameters", tensor_to_string(get_parameters()));

    printer.CloseElement();  
}


string Dense2d::get_activation_function_string_expression() const
{
    switch(activation_function)
    {
    case Activation::Logistic: return "logistic";
    case Activation::HyperbolicTangent: return "tanh";
    case Activation::Linear: return string();
    case Activation::RectifiedLinear: return "ReLU";
    case Activation::ExponentialLinear: return "ELU";
    default: return string();
    }
}


void Dense2dForwardPropagation::set(const Index& new_batch_size, Layer *new_layer)
{
    layer = new_layer;
    
    batch_size = new_batch_size;

    if (!layer) return;

    const Index outputs_number = layer->get_outputs_number();

    outputs.resize(batch_size, outputs_number);

    activation_derivatives.resize(batch_size, outputs_number);

    activation_derivatives.setConstant((type)NAN);
}


pair<type *, dimensions> Dense2dForwardPropagation::get_outputs_pair() const
{
    const dimensions output_dimensions = layer->get_output_dimensions();
    
    return pair<type *, dimensions>((type*)outputs.data(), {{batch_size, output_dimensions[0]}});
}


Dense2dForwardPropagation::Dense2dForwardPropagation(const Index&new_batch_size,
                                                                     Layer *new_layer)
: LayerForwardPropagation()
{
    set(new_batch_size, new_layer);
}


void Dense2dForwardPropagation::print() const
{
    cout << "Outputs:" << endl
         << outputs << endl
         << "Activation derivatives:" << endl
         << activation_derivatives << endl;
}


Dense2dBackPropagation::Dense2dBackPropagation(const Index&new_batch_size, Layer *new_layer)
    : LayerBackPropagation()
{
    set(new_batch_size, new_layer);
}


void Dense2dBackPropagation::set(const Index&new_batch_size,
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


vector<pair<type*, dimensions>> Dense2dBackPropagation::get_input_derivative_pairs() const
{
    const Index inputs_number = layer->get_input_dimensions()[0];

    return { {(type*)(input_derivatives.data()), {batch_size, inputs_number}} };
}


void Dense2dBackPropagation::print() const
{
    cout << "Biases derivatives:" << endl
         << bias_derivatives << endl
         << "Synaptic weights derivatives:" << endl
         << weight_derivatives << endl;
}


Dense2dLayerBackPropagationLM::Dense2dLayerBackPropagationLM(const Index&new_batch_size,
                                                                   Layer *new_layer)
    : LayerBackPropagationLM()
{
    set(new_batch_size, new_layer);
}


void Dense2dLayerBackPropagationLM::set(const Index&new_samples_number, Layer *new_layer)
{
    layer = new_layer;

    batch_size = new_samples_number;

    const Index inputs_number = layer->get_input_dimensions()[0];
    const Index parameters_number = layer->get_parameters_number();

    squared_errors_Jacobian.resize(batch_size, parameters_number);

    input_derivatives.resize(batch_size, inputs_number);
}


vector<pair<type*, dimensions>> Dense2dLayerBackPropagationLM::get_input_derivative_pairs() const
{
    const Index inputs_number = layer->get_input_dimensions()[0];

    return {{(type*)(input_derivatives.data()), {batch_size, inputs_number}}};
}


void Dense2dLayerBackPropagationLM::print() const
{
    cout << "Squared errors Jacobian: " << endl
        << squared_errors_Jacobian << endl;
    cout << "Input derivatives: " << endl
        << input_derivatives << endl;
}


#ifdef OPENNN_CUDA

void Dense2d::forward_propagate_cuda(const vector<float*>& inputs_device,
                                     unique_ptr<LayerForwardPropagationCuda>& forward_propagation_cuda,
                                     const bool& is_training)
{
    // Dense2d layer

    const Index inputs_number = get_inputs_number();
    const Index outputs_number = get_outputs_number();

    // Forward propagation

    Dense2dForwardPropagationCuda* dense2d_layer_forward_propagation_cuda =
        static_cast<Dense2dForwardPropagationCuda*>(forward_propagation_cuda.get());

    const Index batch_size = dense2d_layer_forward_propagation_cuda->batch_size;

    type* combinations = dense2d_layer_forward_propagation_cuda->combinations;
    type* outputs = dense2d_layer_forward_propagation_cuda->outputs;

    const cudnnTensorDescriptor_t& output_tensor_descriptor = dense2d_layer_forward_propagation_cuda->output_tensor_descriptor;
    const cudnnTensorDescriptor_t& output_softmax_tensor_descriptor = dense2d_layer_forward_propagation_cuda->output_softmax_tensor_descriptor;

    const cudnnTensorDescriptor_t& biases_tensor_descriptor = dense2d_layer_forward_propagation_cuda->biases_tensor_descriptor;

    // Combinations

    cublasSgemm(cublas_handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        batch_size, outputs_number, inputs_number,
        &alpha,
        inputs_device[0],
        batch_size,
        weights_device,
        inputs_number,
        &beta,
        combinations,
        batch_size);

    cudnnStatus_t status = cudnnAddTensor(cudnn_handle,
        &alpha,
        biases_tensor_descriptor,
        biases_device,
        &beta_add,
        output_tensor_descriptor,
        combinations);

    if (status != CUDNN_STATUS_SUCCESS)
        cerr << "Dense2d CUDA: cudnnAddTensor failed. Error: " << cudnnGetErrorString(status) << endl;

    // Activations

    switch (activation_function)
    {
    case Activation::Linear:
        cudaMemcpy(outputs, combinations, batch_size * outputs_number * sizeof(type), cudaMemcpyDeviceToDevice);

        break;

    case Activation::Softmax:
        cudnnSoftmaxForward(cudnn_handle,
            CUDNN_SOFTMAX_ACCURATE,
            CUDNN_SOFTMAX_MODE_CHANNEL,
            &alpha,
            output_softmax_tensor_descriptor,
            combinations,
            &beta,
            output_softmax_tensor_descriptor,
            outputs);

        break;

    default:
        cudnnActivationForward(cudnn_handle,
            activation_descriptor,
            &alpha,
            output_tensor_descriptor,
            combinations,
            &beta,
            output_tensor_descriptor,
            outputs);

        break;
    }

    // Droput

    if (is_training && get_dropout_rate() > type(0))
    {
        status = cudnnDropoutForward(cudnn_handle,
            dense2d_layer_forward_propagation_cuda->dropout_descriptor,
            output_tensor_descriptor,
            outputs,
            output_tensor_descriptor,
            outputs,
            dense2d_layer_forward_propagation_cuda->dropout_reserve_space,
            dense2d_layer_forward_propagation_cuda->dropout_reserve_space_size);

        if (status != CUDNN_STATUS_SUCCESS)
            cout << "cudnnDropoutForward failed: " << cudnnGetErrorString(status) << endl;
    }
}


void Dense2d::back_propagate_cuda(const vector<float*>& inputs_device,
                                  const vector<float*>& deltas_device,
                                  unique_ptr<LayerForwardPropagationCuda>& forward_propagation_cuda,
                                  unique_ptr<LayerBackPropagationCuda>& back_propagation_cuda) const
{
    // Dense2d layer

    const Index inputs_number = get_inputs_number();
    const Index outputs_number = get_outputs_number();

    // Forward propagation

    Dense2dForwardPropagationCuda* dense2d_layer_forward_propagation_cuda =
        static_cast<Dense2dForwardPropagationCuda*>(forward_propagation_cuda.get());

    Dense2d* dense2d_layer = static_cast<Dense2d*>(dense2d_layer_forward_propagation_cuda->layer);

    const Index batch_size = dense2d_layer_forward_propagation_cuda->batch_size;

    type* combinations = dense2d_layer_forward_propagation_cuda->combinations;
    type* outputs = dense2d_layer_forward_propagation_cuda->outputs;

    // Back propagation

    Dense2dBackPropagationCuda* dense2d_layer_back_propagation =
        static_cast<Dense2dBackPropagationCuda*>(back_propagation_cuda.get());

    float* ones = dense2d_layer_back_propagation->ones;
    float* error_combinations_derivatives = dense2d_layer_back_propagation->combination_deltas_device;

    float* bias_derivatives = dense2d_layer_back_propagation->bias_derivatives_device;
    float* weight_derivatives = dense2d_layer_back_propagation->weight_derivatives_device;
    float* input_derivatives = dense2d_layer_back_propagation->input_derivatives;

    const cudnnTensorDescriptor_t& deltas_tensor_descriptor = dense2d_layer_back_propagation->deltas_tensor_descriptor;
    const cudnnTensorDescriptor_t& combination_deltas_tensor_descriptor = dense2d_layer_back_propagation->combination_deltas_tensor_descriptor;

    // Dropout

    if (get_dropout_rate() > type(0))
    {
        cudnnStatus_t dstatus = cudnnDropoutBackward(cudnn_handle,
            dense2d_layer_forward_propagation_cuda->dropout_descriptor,
            deltas_tensor_descriptor,
            deltas_device[0],
            deltas_tensor_descriptor,
            deltas_device[0],
            dense2d_layer_forward_propagation_cuda->dropout_reserve_space,
            dense2d_layer_forward_propagation_cuda->dropout_reserve_space_size);

        if (dstatus != CUDNN_STATUS_SUCCESS)
            cout << "cudnnDropoutBackward failed: " << cudnnGetErrorString(dstatus) << endl;
    }

    // Error combinations derivatives

    if (dense2d_layer->get_activation_function() != Activation::Linear && dense2d_layer->get_activation_function() != Activation::Softmax)
        cudnnActivationBackward(cudnn_handle,
            activation_descriptor,
            &alpha,
            combination_deltas_tensor_descriptor,
            outputs,
            deltas_tensor_descriptor,
            deltas_device[0],
            combination_deltas_tensor_descriptor,
            combinations,
            &beta,
            combination_deltas_tensor_descriptor,
            error_combinations_derivatives);
    else
        cudaMemcpy(error_combinations_derivatives, deltas_device[0], batch_size * outputs_number * sizeof(type), cudaMemcpyDeviceToDevice);

    // Bias derivatives

    cublasSgemm(cublas_handle,
        CUBLAS_OP_T, CUBLAS_OP_N,
        outputs_number,
        1,
        batch_size,
        &alpha,
        error_combinations_derivatives,
        batch_size,
        ones,
        batch_size,
        &beta,
        bias_derivatives,
        outputs_number);

    // Weight derivatives

    cublasSgemm(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N,
        inputs_number,
        outputs_number,
        batch_size,
        &alpha,
        inputs_device[0],
        batch_size,
        error_combinations_derivatives,
        batch_size,
        &beta,
        weight_derivatives,
        inputs_number);

    // Input derivatives

    cublasSgemm(cublas_handle,
        CUBLAS_OP_N, CUBLAS_OP_T,
        batch_size,
        inputs_number,
        outputs_number,
        &alpha,
        error_combinations_derivatives,
        batch_size,
        weights_device,
        inputs_number,
        &beta,
        input_derivatives,
        batch_size);
}


void Dense2d::insert_gradient_cuda(unique_ptr<LayerBackPropagationCuda>& back_propagation_cuda,
                                      Index& index, 
                                      float* gradient) const
{
    Dense2dBackPropagationCuda* dense2d_layer_back_propagation =
        static_cast<Dense2dBackPropagationCuda*>(back_propagation_cuda.get());

    copy_to_vector_cuda(gradient, dense2d_layer_back_propagation->weight_derivatives_device, weights.size(), index);
    copy_to_vector_cuda(gradient, dense2d_layer_back_propagation->bias_derivatives_device, biases.size(), index);
}


void Dense2d::set_parameters_cuda(const float* new_parameters, Index& index)
{
    copy_from_vector_cuda(weights_device, new_parameters, weights.size(), index);
    copy_from_vector_cuda(biases_device, new_parameters, biases.size(), index);
}


void Dense2d::allocate_parameters_device()
{
    const Index inputs_number = get_inputs_number();
    const Index outputs_number = get_outputs_number();

    CHECK_CUDA(cudaMalloc(&biases_device, outputs_number * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&weights_device, inputs_number * outputs_number * sizeof(float)));
}


void Dense2d::free_parameters_device()
{
    cudaFree(biases_device);
    cudaFree(weights_device);

    biases_device = nullptr;
    weights_device = nullptr;
}


void Dense2d::copy_parameters_device()
{
    if (!biases_device) cout << "Biases device is null" << endl;

    if (!weights_device) cout << "Weights device is null" << endl;

    CHECK_CUDA(cudaMemcpy(biases_device, biases.data(), biases.size() * sizeof(type), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(weights_device, weights.data(), weights.size() * sizeof(type), cudaMemcpyHostToDevice));
}


void Dense2d::copy_parameters_host()
{
    if (!biases_device) cout << "Biases is null" << endl;
    if (!weights_device) cout << "Synaptic weights is null" << endl;

    CHECK_CUDA(cudaMemcpy(biases.data(), biases_device, biases.size() * sizeof(type), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(weights.data(), weights_device, weights.size() * sizeof(type), cudaMemcpyDeviceToHost));
}


Dense2dForwardPropagationCuda::Dense2dForwardPropagationCuda(const Index& new_batch_size, Layer* new_layer)
    : LayerForwardPropagationCuda()
{
    set(new_batch_size, new_layer);
}


void Dense2dForwardPropagationCuda::set(const Index& new_batch_size, Layer* new_layer)
{
    batch_size = new_batch_size;

    layer = new_layer;

    Dense2d* dense2d_layer = static_cast<Dense2d*>(layer);

    const Index outputs_number = dense2d_layer->get_outputs_number();
    const Index inputs_number = dense2d_layer->get_input_dimensions()[0];

    // Biases

    cudnnCreateTensorDescriptor(&biases_tensor_descriptor);

    cudnnSetTensor4dDescriptor(biases_tensor_descriptor,
        CUDNN_TENSOR_NCHW,
        CUDNN_DATA_FLOAT,
        1,
        outputs_number,
        1,
        1);

    // Outputs

    CHECK_CUDA(cudaMalloc(&combinations, batch_size * outputs_number * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&outputs, batch_size * outputs_number * sizeof(float)));

    cudnnCreateTensorDescriptor(&output_softmax_tensor_descriptor);

    cudnnSetTensor4dDescriptor(output_softmax_tensor_descriptor,
        CUDNN_TENSOR_NCHW,
        CUDNN_DATA_FLOAT,
        1,
        outputs_number,
        batch_size,
        1);

    cudnnCreateTensorDescriptor(&output_tensor_descriptor);

    cudnnSetTensor4dDescriptor(output_tensor_descriptor,
        CUDNN_TENSOR_NCHW,
        CUDNN_DATA_FLOAT,
        batch_size,
        outputs_number,
        1,
        1);

    //Dropout

    if (dense2d_layer->get_dropout_rate() > type(0))
    {
        cudnnCreateDropoutDescriptor(&dropout_descriptor);

        cudnnDropoutGetStatesSize(dense2d_layer->get_cudnn_handle(), &dropout_states_size);

        cudaMalloc(&dropout_states, dropout_states_size);

        cudnnSetDropoutDescriptor(dropout_descriptor,
            dense2d_layer->get_cudnn_handle(),
            static_cast<float>(dense2d_layer->get_dropout_rate()),
            dropout_states,
            dropout_states_size,
            dropout_seed);

        cudnnDropoutGetReserveSpaceSize(output_tensor_descriptor, &dropout_reserve_space_size);
        cudaMalloc(&dropout_reserve_space, dropout_reserve_space_size);
    }
}

void Dense2dForwardPropagationCuda::print() const
{
    // @todo
}


void Dense2dForwardPropagationCuda::free()
{
    cudaFree(combinations);
    cudaFree(outputs);

    cudnnDestroyTensorDescriptor(output_softmax_tensor_descriptor);
    cudnnDestroyTensorDescriptor(output_tensor_descriptor);
    cudnnDestroyTensorDescriptor(biases_tensor_descriptor);

    if (dropout_reserve_space)
        cudaFree(dropout_reserve_space);
    if (dropout_descriptor)
        cudnnDestroyDropoutDescriptor(dropout_descriptor);
    if (dropout_states)
        cudaFree(dropout_states);
}


Dense2dBackPropagationCuda::Dense2dBackPropagationCuda(const Index& new_batch_size, Layer* new_layer)
    : LayerBackPropagationCuda()
{
    set(new_batch_size, new_layer);
}


void Dense2dBackPropagationCuda::set(const Index& new_batch_size, Layer* new_layer)
{
    batch_size = new_batch_size;

    layer = new_layer;

    const Index outputs_number = layer->get_outputs_number();
    const Index inputs_number = layer->get_input_dimensions()[0];

    // Ones

    CHECK_CUDA(cudaMalloc(&ones, batch_size * sizeof(float)));

    for (Index i = 0; i < batch_size; i++)
        CHECK_CUDA(cudaMemcpy(ones + i, &one, sizeof(float), cudaMemcpyHostToDevice));

    CHECK_CUDA(cudaMalloc(&bias_derivatives_device, outputs_number * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&weight_derivatives_device, inputs_number * outputs_number * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&input_derivatives, batch_size * inputs_number * sizeof(float)));

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

    CHECK_CUDA(cudaMalloc(&combination_deltas_device, batch_size * outputs_number * sizeof(float)));

    cudnnCreateTensorDescriptor(&combination_deltas_tensor_descriptor);

    cudnnSetTensor4dDescriptor(combination_deltas_tensor_descriptor,
        CUDNN_TENSOR_NHWC,
        CUDNN_DATA_FLOAT,
        batch_size,
        outputs_number,
        1,
        1);
}


void Dense2dBackPropagationCuda::print() const
{
    // @todo
}


void Dense2dBackPropagationCuda::free()
{
    cudaFree(bias_derivatives_device);
    cudaFree(weight_derivatives_device);
    cudaFree(combination_deltas_device);
    cudaFree(input_derivatives);
    cudaFree(ones);

    cudnnDestroyTensorDescriptor(combination_deltas_tensor_descriptor);
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
