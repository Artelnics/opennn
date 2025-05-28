//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   P R O B A B I L I S T I C   L A Y E R   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "probabilistic_layer.h"
#include "tensors.h"
#include "strings_utilities.h"

namespace opennn
{

Probabilistic::Probabilistic(const dimensions& new_input_dimensions,
                                       const dimensions& new_output_dimensions,
                                       const string& new_name) : Layer()
{
    set(new_input_dimensions, new_output_dimensions, new_name);
}


dimensions Probabilistic::get_input_dimensions() const
{
    return { weights.dimension(0) };
}


dimensions Probabilistic::get_output_dimensions() const
{
    return { biases.size() };
}


const type& Probabilistic::get_decision_threshold() const
{
    return decision_threshold;
}


const Probabilistic::Activation& Probabilistic::get_activation_function() const
{
    return activation_function;
}


string Probabilistic::get_activation_function_string() const
{
    if(activation_function == Activation::Binary)
        return "Binary";
    else if(activation_function == Activation::Logistic)
        return "Logistic";
    else if(activation_function == Activation::Competitive)
        return "Competitive";
    else if(activation_function == Activation::Softmax)
        return "Softmax";
    else
        throw runtime_error("Unknown probabilistic method.\n");
}


Index Probabilistic::get_parameters_number() const
{
    return biases.size() + weights.size();
}


Tensor<type, 1> Probabilistic::get_parameters() const
{
    const Index weights_number = weights.size();
    const Index biases_number = biases.size();

    Tensor<type, 1> parameters(weights_number + biases_number);

    Index index = 0;

    copy_to_vector(parameters, weights, index);
    copy_to_vector(parameters, biases, index);

    return parameters;
}


void Probabilistic::set(const dimensions& new_input_dimensions,
                             const dimensions& new_output_dimensions,
                             const string& new_name)
{
    if (new_input_dimensions.size() != 1)
        throw runtime_error("Input dimensions rank is not 1");

    if (new_output_dimensions.size() != 1)
        throw runtime_error("Output dimensions rank is not 1");

    biases.resize(new_output_dimensions[0]);
    weights.resize(new_input_dimensions[0], new_output_dimensions[0]);

    set_parameters_random();

    layer_type = Layer::Type::Probabilistic;

    new_output_dimensions[0] == 1
        ? activation_function = Activation::Logistic
        : activation_function = Activation::Softmax;

    decision_threshold = type(0.5);

    name = new_name;
}



void Probabilistic::set_input_dimensions(const dimensions& new_input_dimensions)
{
    const dimensions output_dimensions = get_output_dimensions();

    biases.resize(output_dimensions[0]);

    weights.resize(new_input_dimensions[0], output_dimensions[0]);
}


void Probabilistic::set_output_dimensions(const dimensions& new_output_dimensions)
{
    const dimensions input_dimensions = get_input_dimensions();

    biases.resize(new_output_dimensions[0]);

    weights.resize(input_dimensions[0], new_output_dimensions[0]);
}


void Probabilistic::set_parameters(const Tensor<type, 1>& new_parameters, Index& index)
{
    copy_from_vector(weights, new_parameters, index);
    copy_from_vector(biases, new_parameters, index);
}


void Probabilistic::set_decision_threshold(const type& new_decision_threshold)
{
    decision_threshold = new_decision_threshold;
}


void Probabilistic::set_activation_function(const Activation& new_activation_function)
{
    activation_function = new_activation_function;
}


void Probabilistic::set_activation_function(const string& new_activation_function)
{
    if(new_activation_function == "Binary")
        set_activation_function(Activation::Binary);
    else if(new_activation_function == "Logistic")
        set_activation_function(Activation::Logistic);
    else if(new_activation_function == "Competitive")
        set_activation_function(Activation::Competitive);
    else if(new_activation_function == "Softmax")
        set_activation_function(Activation::Softmax);
    else
        throw runtime_error("Unknown probabilistic method: " + new_activation_function + ".\n");
}


void Probabilistic::set_parameters_constant(const type& value)
{
    biases.setConstant(value);

    weights.setConstant(value);
}


void Probabilistic::set_parameters_random()
{
    set_random(biases);

    set_random(weights);
}


void Probabilistic::calculate_combinations(const Tensor<type, 2>& inputs,
                                                Tensor<type, 2>& combinations) const
{
    const Index batch_size = combinations.dimension(0);
    const Index outputs_number = biases.size();

    combinations.device(*thread_pool_device)
        = inputs.contract(weights, axes(1,0))
        + biases.reshape(array<Index, 2>({1, outputs_number}))
                .broadcast(array<Index, 2>({batch_size, 1}));
}

void Probabilistic::calculate_activations(Tensor<type, 2>& activations,Tensor<type, 2>& activation_derivatives) const
{
    switch (activation_function)
    {
    case Activation::Softmax:
        softmax(activations);
        return;
    case Activation::Logistic:
        logistic(activations, activation_derivatives);
        return;
    case Activation::Competitive:
        competitive(activations);
        return;
    default:
        return;
    }
}

void Probabilistic::forward_propagate(const vector<pair<type*, dimensions>>& input_pairs,
                                           unique_ptr<LayerForwardPropagation>& forward_propagation,
                                           const bool& is_training)
{
    const Index outputs_number = get_outputs_number();

    const TensorMap<Tensor<type, 2>> inputs = tensor_map_2(input_pairs[0]);

    ProbabilisticForwardPropagation* probabilistic_layer_forward_propagation =
        static_cast<ProbabilisticForwardPropagation*>(forward_propagation.get());

    Tensor<type, 2>& outputs = probabilistic_layer_forward_propagation->outputs;

    calculate_combinations(inputs, outputs);

    if (outputs_number == 1 && !is_training)
    {
        logistic(outputs, empty_2);
    }
    else if (outputs_number == 1 && is_training)
    {
        Tensor<type, 2>& activation_derivatives = probabilistic_layer_forward_propagation->activation_derivatives;

        logistic(outputs, activation_derivatives);
    }
    else if (outputs_number > 1)
    {
        softmax(outputs);
    }
    else
        calculate_activations(outputs, empty_2);
}


void Probabilistic::back_propagate(const vector<pair<type*, dimensions>>& input_pairs,
                                        const vector<pair<type*, dimensions>>& delta_pairs,
                                        unique_ptr<LayerForwardPropagation>& forward_propagation,
                                        unique_ptr<LayerBackPropagation>& back_propagation) const
{
    const Index outputs_number = get_outputs_number();

    const TensorMap<Tensor<type, 2>> inputs = tensor_map_2(input_pairs[0]);
    TensorMap<Tensor<type, 2>> deltas = tensor_map_2(delta_pairs[0]);

    // Forward propagation

    ProbabilisticForwardPropagation* probabilistic_layer_forward_propagation =
        static_cast<ProbabilisticForwardPropagation*>(forward_propagation.get());

    // Back propagation

    ProbabilisticBackPropagation* probabilistic_back_propagation =
            static_cast<ProbabilisticBackPropagation*>(back_propagation.get());

    Tensor<type, 2>& input_derivatives = probabilistic_back_propagation->input_derivatives;

    if(outputs_number == 1)
    {
        const Tensor<type, 2>& activation_derivatives = probabilistic_layer_forward_propagation->activation_derivatives;

        deltas.device(*thread_pool_device) = deltas * activation_derivatives;
    }

    Tensor<type, 1>& bias_derivatives = probabilistic_back_propagation->bias_derivatives;

    Tensor<type, 2>& weight_derivatives = probabilistic_back_propagation->weight_derivatives;

    weight_derivatives.device(*thread_pool_device) = inputs.contract(deltas, axes(0,0));

    bias_derivatives.device(*thread_pool_device) = deltas.sum(array<Index, 1>({0}));

    input_derivatives.device(*thread_pool_device) = deltas.contract(weights, axes(1,1));
}


void Probabilistic::insert_gradient(unique_ptr<LayerBackPropagation>& back_propagation,
                                         Index& index,
                                         Tensor<type, 1>& gradient) const
{
    const ProbabilisticBackPropagation* probabilistic_back_propagation =
        static_cast<ProbabilisticBackPropagation*>(back_propagation.get());

    copy_to_vector(gradient, probabilistic_back_propagation->weight_derivatives, index);
    copy_to_vector(gradient, probabilistic_back_propagation->bias_derivatives, index);
}


void Probabilistic::insert_squared_errors_Jacobian_lm(unique_ptr<LayerBackPropagationLM>& back_propagation,
                                                           const Index& index,
                                                           Tensor<type, 2>& squared_errors_Jacobian) const
{
    ProbabilisticLayerBackPropagationLM* probabilistic_layer_back_propagation_lm =
        static_cast<ProbabilisticLayerBackPropagationLM*>(back_propagation.get());

    const Index batch_size = back_propagation->batch_size;
    const Index parameters_number = get_parameters_number();

    type* this_squared_errors_Jacobian_data = probabilistic_layer_back_propagation_lm->squared_errors_Jacobian.data();

    memcpy(squared_errors_Jacobian.data() + index,
           this_squared_errors_Jacobian_data,
           parameters_number * batch_size *sizeof(type));
}


void Probabilistic::print() const
{
    cout << "Probabilistic layer" << endl
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


void Probabilistic::to_XML(XMLPrinter& printer) const
{
    printer.OpenElement("Probabilistic");

    add_xml_element(printer, "InputsNumber", to_string(get_input_dimensions()[0]));
    add_xml_element(printer, "NeuronsNumber", to_string(get_output_dimensions()[0]));
    add_xml_element(printer, "Activation", get_activation_function_string());
    add_xml_element(printer, "Parameters", tensor_to_string(get_parameters()));
    add_xml_element(printer, "DecisionThreshold", to_string(decision_threshold));

    printer.CloseElement();
}


void Probabilistic::from_XML(const XMLDocument& document)
{
    const XMLElement* probabilistic_layer_element = document.FirstChildElement("Probabilistic");

    if(!probabilistic_layer_element)
        throw runtime_error("Probabilistic layer element is nullptr.\n");

    const Index new_inputs_number = read_xml_index(probabilistic_layer_element, "InputsNumber");
    const Index new_neurons_number = read_xml_index(probabilistic_layer_element, "NeuronsNumber");

    set({ new_inputs_number }, { new_neurons_number });

    set_activation_function(read_xml_string(probabilistic_layer_element, "Activation"));

    Index index = 0;

    set_parameters(to_type_vector(read_xml_string(probabilistic_layer_element, "Parameters"), " "), index);
    set_decision_threshold(read_xml_type(probabilistic_layer_element, "DecisionThreshold"));
}


string Probabilistic::write_binary_expression(const vector<string>& input_names, const vector<string>& output_names) const
{
    ostringstream buffer;

    for(size_t j = 0; j < output_names.size(); j++)
        buffer << output_names[j] << " = binary(" << input_names[j] << ");\n";

    return buffer.str();
}


string Probabilistic::write_logistic_expression(const vector<string>& input_names,
                                                     const vector<string>& output_names) const
{
    ostringstream buffer;

    for(size_t j = 0; j < output_names.size(); j++)
        buffer << output_names[j] << " = logistic(" << input_names[j] << ");\n";

    return buffer.str();
}


string Probabilistic::write_competitive_expression(const vector<string>& input_names, const vector<string>& output_names) const
{
    ostringstream buffer;

    for(size_t j = 0; j < output_names.size(); j++)
        buffer << output_names[j] << " = competitive(" << input_names[j] << ");\n";

    return buffer.str();
}


string Probabilistic::write_softmax_expression(const vector<string>& input_names, const vector<string>& output_names) const
{
    ostringstream buffer;

    for(size_t j = 0; j < output_names.size(); j++)
        buffer << output_names[j] << " = softmax(" << input_names[j] << ");\n";

    return buffer.str();
}


string Probabilistic::write_combinations(const vector<string>& input_names) const
{
    ostringstream buffer;

    const Index inputs_number = get_inputs_number();
    const Index outputs_number = get_outputs_number();

    for(Index i = 0; i < outputs_number; i++)
    {
        buffer << "probabilistic_layer_combinations_" << to_string(i) << " = " << biases(i);

        for(Index j = 0; j < inputs_number; j++)
            buffer << " +" << weights(j, i) << "*" << input_names[j] << "";

        buffer << " " << endl;
    }

    buffer << "\t" << endl;

    return buffer.str();
}


string Probabilistic::write_activations(const vector<string>& output_names) const
{
    ostringstream buffer;

    const Index outputs_number = get_outputs_number();

    for(Index i = 0; i < outputs_number; i++)
    {
        switch(activation_function)
        {
        case Activation::Binary:
            buffer << "\tif" << "probabilistic_layer_combinations_" << to_string(i) << " < 0.5, " << output_names[i] << "= 0.0. Else " << output_names[i] << " = 1.0\n";
            break;

        case Activation::Logistic:
            buffer <<  output_names[i] << " = 1.0/(1.0 + exp(-" <<  "probabilistic_layer_combinations_" << to_string(i) << "));\n";
            break;

        case Activation::Competitive:
            if(i == 0)
                buffer << "\tfor each probabilistic_layer_combinations_i:" << endl
                       << "\t\tif probabilistic_layer_combinations_i is equal to max(probabilistic_layer_combinations_i):"<<endl
                       << "\t\t\tactivations[i] = 1"<<endl
                       << "\t\telse:"<<endl
                       << "\t\t\tactivations[i] = 0"<<endl;
            break;

        case Activation::Softmax:

            if (i == 0)
            {
                buffer << "sum = ";

                for (Index j = 0; j < outputs_number; j++)
                {
                    buffer << "exp(probabilistic_layer_combinations_" << to_string(j) << ")";

                    if (j != outputs_number - 1)
                        buffer << " + ";
                }

                buffer << ";\n" << endl;

                for (Index j = 0; j < outputs_number; j++)
                    buffer << output_names[j] << " = exp(probabilistic_layer_combinations_" << to_string(j) << ")/sum;\n";
            }
            break;
        default:
            break;
        }
    }

    return buffer.str();
}


string Probabilistic::get_expression(const vector<string>& input_names,
                                            const vector<string>& output_names) const
{
    ostringstream buffer;

    buffer << write_combinations(input_names)
           << write_activations(output_names);

    return buffer.str();
}


ProbabilisticForwardPropagation::ProbabilisticForwardPropagation(
    const Index& new_batch_size, Layer *new_layer)
    : LayerForwardPropagation()
{
    set(new_batch_size, new_layer);
}


pair<type *, dimensions> ProbabilisticForwardPropagation::get_outputs_pair() const
{
    const Index outputs_number = layer->get_outputs_number();

    return pair<type *, dimensions>((type*)outputs.data(), {{batch_size, outputs_number}});
}


void ProbabilisticForwardPropagation::set(const Index& new_batch_size, Layer *new_layer)
{
    layer = new_layer;

    batch_size = new_batch_size;

    const Index outputs_number = layer->get_outputs_number();

    outputs.resize(batch_size, outputs_number);

    activation_derivatives.resize(0, 0);

    if(outputs_number == 1)
        activation_derivatives.resize(batch_size, outputs_number);
}


void ProbabilisticForwardPropagation::print() const
{
    cout << "Probabilistic layer forward-propagation" << endl
         << "Outputs dimensions:" << endl
         << outputs.dimensions() << endl;

    const Index outputs_number = layer->get_outputs_number();

    if(outputs_number == 1)
       cout << "Activation derivatives:" << endl
            << activation_derivatives << endl;
}


ProbabilisticBackPropagation::ProbabilisticBackPropagation(const Index&new_batch_size, Layer *new_layer)
    : LayerBackPropagation()
{
    set(new_batch_size, new_layer);
}


void ProbabilisticBackPropagation::set(const Index& new_batch_size, Layer *new_layer)
{
    layer = new_layer;

    batch_size = new_batch_size;

    const Index outputs_number = layer->get_outputs_number();
    const Index inputs_number = layer->get_input_dimensions()[0];

    bias_derivatives.resize(outputs_number);

    weight_derivatives.resize(inputs_number, outputs_number);

    input_derivatives.resize(batch_size, inputs_number);
}


vector<pair<type*, dimensions>> ProbabilisticBackPropagation::get_input_derivative_pairs() const
{
    const Index inputs_number = layer->get_input_dimensions()[0];

    return {{(type*)(input_derivatives.data()), {batch_size, inputs_number}} };
}


void ProbabilisticBackPropagation::print() const
{
    cout << "Biases derivatives:" << endl
         << bias_derivatives << endl
         << "Synaptic weights derivatives:" << endl
         << weight_derivatives << endl;
}


ProbabilisticLayerBackPropagationLM::ProbabilisticLayerBackPropagationLM(const Index& new_batch_size, Layer* new_layer)
    : LayerBackPropagationLM()
{
    set(new_batch_size, new_layer);
}


vector<pair<type*, dimensions>> ProbabilisticLayerBackPropagationLM::get_input_derivative_pairs() const
{
    return vector<pair<type*, dimensions>>();
}


void ProbabilisticLayerBackPropagationLM::set(const Index& new_batch_size, Layer* new_layer)
{
    layer = new_layer;

    batch_size = new_batch_size;

    const Index parameters_number = layer->get_parameters_number();

    squared_errors_Jacobian.resize(batch_size, parameters_number);
}


void ProbabilisticLayerBackPropagationLM::print() const
{
    cout << "Squared errors Jacobian: " << endl
        << squared_errors_Jacobian << endl;
}


#ifdef OPENNN_CUDA

void Probabilistic::forward_propagate_cuda(const vector<pair<type*, dimensions>>& input_pairs_device,
                                                unique_ptr<LayerForwardPropagationCuda>& forward_propagation_cuda,
                                                const bool& is_training)
{
    // Probabilistic layer

    const Index inputs_number = get_inputs_number();
    const Index outputs_number = get_outputs_number();

    // Inputs

    const Index batch_size = input_pairs_device[0].second[0];

    const type* inputs_device = input_pairs_device[0].first;

    // Forward propagation

    ProbabilisticForwardPropagationCuda* probabilistic_layer_forward_propagation_cuda =
        static_cast<ProbabilisticForwardPropagationCuda*>(forward_propagation_cuda.get());

    float* outputs = probabilistic_layer_forward_propagation_cuda->outputs;

    const cudnnActivationDescriptor_t& activation_descriptor = probabilistic_layer_forward_propagation_cuda->activation_descriptor;

    const cudnnTensorDescriptor_t& output_tensor_descriptor = probabilistic_layer_forward_propagation_cuda->output_tensor_descriptor;
    const cudnnTensorDescriptor_t& output_tensor_descriptor = probabilistic_layer_forward_propagation_cuda->output_tensor_descriptor;
    const cudnnTensorDescriptor_t& outputs_softmax_tensor_descriptor = probabilistic_layer_forward_propagation_cuda->outputs_softmax_tensor_descriptor;
    const cudnnTensorDescriptor_t& outputs_batch_tensor_descriptor = probabilistic_layer_forward_propagation_cuda->outputs_batch_tensor_descriptor;
    const cudnnTensorDescriptor_t& biases_batch_tensor_descriptor = probabilistic_layer_forward_propagation_cuda->biases_batch_tensor_descriptor;

    cublasSgemm(cublas_handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        batch_size, outputs_number, inputs_number,
        &alpha,
        inputs_device,
        batch_size,
        weights_device,
        inputs_number,
        &beta,
        outputs,
        batch_size);

    // @todo Improve by using cudnnAddTensor

    for (Index biases_index = 0; biases_index < outputs_number; biases_index++)
    {
        type* outputs_batch = outputs + biases_index * batch_size;
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

    switch (activation_function)
    {
    case Activation::Logistic:
        cudnnActivationForward(cudnn_handle,
            activation_descriptor,
            &alpha,
            output_tensor_descriptor,
            combinations,
            output_tensor_descriptor,
            outputs,
            &beta,
            output_tensor_descriptor,
            outputs);

        break;

    case Activation::Softmax:
        cudnnSoftmaxForward(cudnn_handle,
            CUDNN_SOFTMAX_ACCURATE,
            CUDNN_SOFTMAX_MODE_CHANNEL,
            &alpha,
            outputs_softmax_tensor_descriptor,
            outputs,
            &beta,
            outputs_softmax_tensor_descriptor,
            outputs);

        break;
    }
}


void Probabilistic::back_propagate_cuda(const vector<pair<type*, dimensions>>& input_pairs_device,
                                             const vector<pair<type*, dimensions>>& deltas_pair_device,
                                             unique_ptr<LayerForwardPropagationCuda>& forward_propagation_cuda,
                                             unique_ptr<LayerBackPropagationCuda>& back_propagation_cuda) const
{
    // Probabilistic layer

    const Index inputs_number = get_inputs_number();
    const Index outputs_number = get_outputs_number();

    // Inputs

    const Index batch_size = input_pairs_device[0].second[0];

    const float* inputs_device = input_pairs_device[0].first;

    const float* deltas_device = deltas_pair_device[0].first;

    // Forward propagation

    ProbabilisticForwardPropagationCuda* probabilistic_layer_forward_propagation =
        static_cast<ProbabilisticForwardPropagationCuda*>(forward_propagation_cuda.get());

    const float* outputs = probabilistic_layer_forward_propagation->outputs;

    const cudnnActivationDescriptor_t& activation_descriptor = probabilistic_layer_forward_propagation->activation_descriptor;

    const cudnnTensorDescriptor_t& output_tensor_descriptor = probabilistic_layer_forward_propagation->output_tensor_descriptor;
    const cudnnTensorDescriptor_t& output_tensor_descriptor = probabilistic_layer_forward_propagation->output_tensor_descriptor;

    // Back propagation

    ProbabilisticBackPropagationCuda* probabilistic_layer_back_propagation =
        static_cast<ProbabilisticBackPropagationCuda*>(back_propagation_cuda.get());

    const float* ones = probabilistic_layer_back_propagation->ones;
    float* error_combinations_derivatives = probabilistic_layer_back_propagation->error_combinations_derivatives_device;
    float* weight_derivatives = probabilistic_layer_back_propagation->weight_derivatives_device;
    float* bias_derivatives = probabilistic_layer_back_propagation->bias_derivatives_device;
    float* input_derivatives = probabilistic_layer_back_propagation->input_derivatives;

    const cudnnTensorDescriptor_t& error_combinations_derivatives_tensor_descriptor = probabilistic_layer_back_propagation->error_combinations_derivatives_tensor_descriptor;

    // Error combinations

    if (outputs_number == 1)
    {
        cudnnActivationBackward(cudnn_handle,
            activation_descriptor,
            &alpha,
            error_combinations_derivatives_tensor_descriptor,
            outputs,
            output_tensor_descriptor,
            deltas_device,
            error_combinations_derivatives_tensor_descriptor,
            combinations,
            &beta,
            error_combinations_derivatives_tensor_descriptor,
            error_combinations_derivatives);
    }
    else
        cudaMemcpy(error_combinations_derivatives, deltas_device, batch_size * outputs_number * sizeof(float), cudaMemcpyDeviceToDevice);

    // Weight derivatives

    cublasSgemm(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N,
        inputs_number,
        outputs_number,
        batch_size,
        &alpha,
        inputs_device,
        batch_size,
        error_combinations_derivatives,
        batch_size,
        &beta,
        weight_derivatives,
        inputs_number);

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


void Probabilistic::insert_gradient_cuda(unique_ptr<LayerBackPropagationCuda>& back_propagation_cuda,
                                              Index& index,
                                              float* gradient) const
{
    ProbabilisticBackPropagationCuda* probabilistic_layer_back_propagation_cuda =
        static_cast<ProbabilisticBackPropagationCuda*>(back_propagation_cuda.get());

    copy_to_vector_cuda(gradient, probabilistic_layer_back_propagation_cuda->weight_derivatives_device, weights.size(), index);
    copy_to_vector_cuda(gradient, probabilistic_layer_back_propagation_cuda->bias_derivatives_device, biases.size(), index);
}


void Probabilistic::set_parameters_cuda(const float* new_parameters, Index& index)
{
    copy_from_vector_cuda(weights_device, new_parameters, weights.size(), index);
    copy_from_vector_cuda(biases_device, new_parameters, biases.size(), index);
}


void Probabilistic::allocate_parameters_device()
{
    const Index inputs_number = get_inputs_number();
    const Index outputs_number = get_outputs_number();

    if (cudaMalloc(&biases_device, outputs_number * sizeof(float)) != cudaSuccess)
        cout << "Biases allocation error" << endl;

    if (cudaMalloc(&weights_device, inputs_number * outputs_number * sizeof(float)) != cudaSuccess)
        cout << "Weights allocation error" << endl;
}


void Probabilistic::copy_parameters_device()
{
    if (!biases_device) cout << "Biases is null" << endl;

    if (!weights_device) cout << "Weights is null" << endl;

    if (cudaMemcpy(biases_device, biases.data(), biases.size() * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess)
        cout << "Biases copy error" << endl;

    if (cudaMemcpy(weights_device, weights.data(), weights.size() * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess)
        cout << "Weights copy error" << endl;
}


void Probabilistic::copy_parameters_host()
{
    if (!biases_device) cout << "Biases device is null" << endl;

    if (!weights_device) cout << "Weights device is null" << endl;

    if (cudaMemcpy(biases.data(), biases_device, biases.size() * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess)
        cout << "Biases copy error" << endl;

    if (cudaMemcpy(weights.data(), weights_device, weights.size() * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess)
        cout << "Weights copy error" << endl;
}


void Probabilistic::free_parameters_device()
{
    cudaFree(biases_device);
    cudaFree(weights_device);

    biases_device = nullptr;
    weights_device = nullptr;
}


ProbabilisticForwardPropagationCuda::ProbabilisticForwardPropagationCuda(const Index& new_batch_samples_number, Layer* new_layer)
    : LayerForwardPropagationCuda()
{
    set(new_batch_samples_number, new_layer);
}


void ProbabilisticForwardPropagationCuda::set(const Index& new_batch_samples_number, Layer* new_layer)
{
    batch_size = new_batch_samples_number;

    layer = new_layer;

    const Index outputs_number = layer->get_outputs_number();

    const Index inputs_number = layer->get_inputs_number();

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

    if (cudaMalloc(&outputs, batch_size * outputs_number * sizeof(float)) != cudaSuccess)
        cout << "outputs allocation error" << endl;

    cudnnCreateTensorDescriptor(&outputs_softmax_tensor_descriptor);

    cudnnSetTensor4dDescriptor(outputs_softmax_tensor_descriptor,
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

    cudnnCreateTensorDescriptor(&outputs_batch_tensor_descriptor);

    cudnnSetTensor4dDescriptor(outputs_batch_tensor_descriptor,
        CUDNN_TENSOR_NCHW,
        CUDNN_DATA_FLOAT,
        batch_size,
        1,
        1,
        1);

    // Activations

    if (outputs_number == 1)
    {
        cudnnCreateActivationDescriptor(&activation_descriptor);

        cudnnSetActivationDescriptor(activation_descriptor, CUDNN_ACTIVATION_SIGMOID, CUDNN_PROPAGATE_NAN, 0.0);
    }
}


void ProbabilisticForwardPropagationCuda::print() const
{
    const Index outputs_number = layer->get_outputs_number();

    cout << layer->get_type_string() + " forward propagation" << endl;

    cout << "Outputs (Softmax forward)" << endl;
    cout << matrix_from_device(outputs, batch_size, outputs_number) << endl;
}


void ProbabilisticForwardPropagationCuda::free()
{
    cudaFree(outputs);

    cudnnDestroyTensorDescriptor(output_tensor_descriptor);
    cudnnDestroyTensorDescriptor(outputs_softmax_tensor_descriptor);
    cudnnDestroyTensorDescriptor(outputs_batch_tensor_descriptor);
    cudnnDestroyTensorDescriptor(biases_batch_tensor_descriptor);
    cudnnDestroyActivationDescriptor(activation_descriptor);
}


pair<type*, dimensions> ProbabilisticForwardPropagationCuda::get_outputs_pair_device() const
{
    const Index outputs_number = layer->get_outputs_number();

    return pair<type*, dimensions>(outputs, { {batch_size, outputs_number} });
}


ProbabilisticBackPropagationCuda::ProbabilisticBackPropagationCuda(const Index& new_batch_samples_number, Layer* new_layer)
    : LayerBackPropagationCuda()
{
    set(new_batch_samples_number, new_layer);
}


void ProbabilisticBackPropagationCuda::set(const Index& new_batch_samples_number, Layer* new_layer)
{
    batch_size = new_batch_samples_number;

    layer = new_layer;

    const Index outputs_number = layer->get_outputs_number();

    const Index inputs_number = layer->get_inputs_number();

    // Sum

    cudnnCreateOpTensorDescriptor(&operator_sum_descriptor);

    cudnnSetOpTensorDescriptor(operator_sum_descriptor,
        CUDNN_OP_TENSOR_ADD,
        CUDNN_DATA_FLOAT,
        CUDNN_NOT_PROPAGATE_NAN);

    // Error combinations derivatives

    if (cudaMalloc(&error_combinations_derivatives_device, batch_size * outputs_number * sizeof(float)) != cudaSuccess)
        cout << "error combination derivatives allocation error" << endl;

    cudnnCreateTensorDescriptor(&error_combinations_derivatives_tensor_descriptor);

    cudnnSetTensor4dDescriptor(error_combinations_derivatives_tensor_descriptor,
        CUDNN_TENSOR_NHWC,
        CUDNN_DATA_FLOAT,
        batch_size,
        outputs_number,
        1,
        1);

    // bias_derivatives_device

    if (cudaMalloc(&bias_derivatives_device, outputs_number * sizeof(float)) != cudaSuccess)
        cout << "bias_derivatives probabilistic allocation error" << endl;

    // weight_derivatives_device

    if (cudaMalloc(&weight_derivatives_device, inputs_number * outputs_number * sizeof(float)) != cudaSuccess)
        cout << "weight_derivatives allocation error" << endl;

    // Input derivatives

    if (cudaMalloc(&input_derivatives, batch_size * inputs_number * sizeof(float)) != cudaSuccess)
        cout << "inputs derivatives allocation error" << endl;

    // Aux ones vector

    if (cudaMalloc(&ones, batch_size * sizeof(float)) != cudaSuccess)
        cout << "bias_derivatives allocation error" << endl;

    for (Index i = 0; i < batch_size; i++) {
        cudaMemcpy(ones + i, &one, sizeof(float), cudaMemcpyHostToDevice);
    }
}


vector<pair<type*, dimensions>> ProbabilisticBackPropagationCuda::get_input_derivative_pairs_device() const
{
    const Index inputs_number = layer->get_input_dimensions()[0];

    return { {input_derivatives, {batch_size, inputs_number}} };
}


void ProbabilisticBackPropagationCuda::print() const
{
    const Index inputs_number = layer->get_inputs_number();
    const Index outputs_number = layer->get_outputs_number();

    cout << layer->get_type_string() + " back propagation" << endl;

    cout << "bias_derivatives_device" << endl;
    cout << matrix_from_device(bias_derivatives_device, outputs_number, 1) << endl;

    cout << "weight_derivatives_device" << endl;
    cout << matrix_from_device(weight_derivatives_device, inputs_number, outputs_number) << endl;

    cout << "inputs derivatives" << endl;
    cout << matrix_from_device(input_derivatives, batch_size, outputs_number) << endl;

    cout << "error_combinations_derivatives" << endl;
    cout << matrix_from_device(error_combinations_derivatives_device, batch_size, outputs_number) << endl;
}


void ProbabilisticBackPropagationCuda::free()
{
    cudaFree(error_combinations_derivatives_device);
    cudaFree(bias_derivatives_device);
    cudaFree(weight_derivatives_device);
    cudaFree(input_derivatives);
    cudaFree(ones);

    cudnnDestroyOpTensorDescriptor(operator_sum_descriptor);
    cudnnDestroyTensorDescriptor(error_combinations_derivatives_tensor_descriptor);
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
