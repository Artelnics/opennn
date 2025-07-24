//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   P E R C E P T R O N   L A Y E R   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "registry.h"
#include "tensors.h"
#include "perceptron_layer.h"

namespace opennn
{

Dense2d::Dense2d(const dimensions& new_input_dimensions,
                 const dimensions& new_output_dimensions,
                 const string& new_activation_function,
                 const bool& new_batch_normalization,
                 const string& new_layer_label) : Layer()
{
    set(new_input_dimensions, new_output_dimensions, new_activation_function, new_batch_normalization, new_layer_label);
}


dimensions Dense2d::get_input_dimensions() const
{
    return { weights.dimension(0) };
}


dimensions Dense2d::get_output_dimensions() const
{
    return { biases.size() };
}


vector<pair<type*, Index>> Dense2d::get_parameter_pairs() const
{
    vector<pair<type*, Index>> parameter_pairs =
    {
        {(type*)(biases.data()), biases.size()},
        {(type*)(weights.data()), weights.size()}
    };

    if (batch_normalization)
    {
        parameter_pairs.push_back({ const_cast<type*>(scales.data()), scales.size() });
        parameter_pairs.push_back({ const_cast<type*>(offsets.data()), offsets.size() });
    }

    return parameter_pairs;
}


void Dense2d::set_dropout_rate(const type& new_dropout_rate)
{
    if (new_dropout_rate < type(0) || new_dropout_rate >= type(1))
        throw runtime_error("Dropout rate must be in [0,1).");

    dropout_rate = new_dropout_rate;
}


type Dense2d::get_dropout_rate() const
{
    return dropout_rate;
}


bool Dense2d::get_batch_normalization() const
{
    return batch_normalization;
}

Tensor<type, 1> Dense2d::get_scales() const
{
    return scales;
}

Tensor<type, 1> Dense2d::get_offsets() const
{
    return offsets;
}


const string& Dense2d::get_activation_function() const
{
    return activation_function;
}


void Dense2d::set(const dimensions& new_input_dimensions,
                  const dimensions& new_output_dimensions,
                  const string& new_activation_function,
                  const bool& new_batch_normalization,
                  const string& new_label)
{
    if (new_input_dimensions.size() != 1)
        throw runtime_error("Input dimensions size is not 1");

    if (new_output_dimensions.size() != 1)
        throw runtime_error("Output dimensions size is not 1");   

    biases.resize(new_output_dimensions[0]);    
    weights.resize(new_input_dimensions[0], new_output_dimensions[0]);    

    set_parameters_random();

    set_activation_function(new_activation_function);

    set_batch_normalization(new_batch_normalization);

    const Index outputs_number = get_outputs_number();

    if (batch_normalization)
    {
        scales.resize(outputs_number);
        scales.setConstant(1.0);

        offsets.resize(outputs_number);
        offsets.setZero();

        moving_means.resize(outputs_number);
        moving_means.setZero();

        moving_standard_deviations.resize(outputs_number);
        moving_standard_deviations.setZero();
    }

    set_label(new_label);
    
    name = "Dense2d";

    #ifdef OPENNN_CUDA

    if (batch_normalization)
    {
        cudnnCreateTensorDescriptor(&bn_tensor_descriptor);

        cudnnSetTensor4dDescriptor(bn_tensor_descriptor,
            CUDNN_TENSOR_NCHW,
            CUDNN_DATA_FLOAT,
            1, outputs_number, 1, 1);
    }

    // Activations
    
    if (activation_function != "Softmax")
    {
        cudnnCreateActivationDescriptor(&activation_descriptor);

        cudnnActivationMode_t activation = CUDNN_ACTIVATION_IDENTITY;

        if (activation_function == "Linear")
        {
            activation = CUDNN_ACTIVATION_IDENTITY;
            use_combinations = false;
        }
        else if (activation_function == "Logistic")
        {
            activation = CUDNN_ACTIVATION_SIGMOID;
            use_combinations = false;
        }
        else if (activation_function == "HyperbolicTangent")
        {
            activation = CUDNN_ACTIVATION_TANH;
            use_combinations = false;
        }
        else if (activation_function == "RectifiedLinear")
        {
            activation = CUDNN_ACTIVATION_RELU;
            use_combinations = false;
        }
        else if (activation_function == "ExponentialLinear")
        {
            activation = CUDNN_ACTIVATION_ELU;
            use_combinations = true;
        }
        else if (activation_function == "ClippedRelu")
        {
            activation = CUDNN_ACTIVATION_CLIPPED_RELU;
            use_combinations = true;
        }
        else if (activation_function == "Swish")
        {
            activation = CUDNN_ACTIVATION_SWISH;
            use_combinations = true;
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


void Dense2d::set_batch_normalization(const bool& new_batch_normalization)
{
    batch_normalization = new_batch_normalization;
}


void Dense2d::set_activation_function(const string& new_activation_function)
{
    if(new_activation_function == "Logistic"
    || new_activation_function == "HyperbolicTangent"
    || new_activation_function == "Linear"
    || new_activation_function == "RectifiedLinear"
    || new_activation_function == "ExponentialLinear"
    || new_activation_function == "Softmax")
        activation_function = new_activation_function;
    else
        throw runtime_error("Unknown activation function: " + new_activation_function);

    if (new_activation_function == "Softmax" && get_outputs_number() == 1)
        activation_function = "Logistic";
}


void Dense2d::calculate_combinations(const Tensor<type, 2>& inputs,
                                     Tensor<type, 2>& combinations) const
{
    const Index batch_size = combinations.dimension(0);
    const Index outputs_number = biases.size();

    combinations.device(*thread_pool_device)
        = inputs.contract(weights, axes(1,0))
        + biases.reshape(array_2(1, outputs_number))
                .broadcast(array_2(batch_size, 1));
}


void Dense2d::apply_batch_normalization(unique_ptr<LayerForwardPropagation>& layer_forward_propagation, const bool& is_training)
{
    // @todo cpu
}


void Dense2d::forward_propagate(const vector<pair<type*, dimensions>>& input_pairs,
                                unique_ptr<LayerForwardPropagation>& layer_forward_propagation,
                                const bool& is_training)
{
    const TensorMap<Tensor<type, 2>> inputs = tensor_map<2>(input_pairs[0]);

    Dense2dForwardPropagation* dense2d_forward_propagation =
        static_cast<Dense2dForwardPropagation*>(layer_forward_propagation.get());

    Tensor<type, 2>& outputs = dense2d_forward_propagation->outputs;

    calculate_combinations(inputs,
                           outputs);

    if (batch_normalization)
        apply_batch_normalization(layer_forward_propagation, is_training);

    is_training
        ? calculate_activations(activation_function, outputs, dense2d_forward_propagation->activation_derivatives)
        : calculate_activations(activation_function, outputs, empty_2);

    if(is_training && dropout_rate > type(0))
        dropout(outputs, dropout_rate);
}


void Dense2d::back_propagate(const vector<pair<type*, dimensions>>& input_pairs,
                             const vector<pair<type*, dimensions>>& delta_pairs,
                             unique_ptr<LayerForwardPropagation>& forward_propagation,
                             unique_ptr<LayerBackPropagation>& back_propagation) const
{
    const TensorMap<Tensor<type, 2>> inputs = tensor_map<2>(input_pairs[0]);
    TensorMap<Tensor<type, 2>> deltas = tensor_map<2>(delta_pairs[0]);

    // Forward propagation

    const Dense2dForwardPropagation* dense2d_layer_forward_propagation =
        static_cast<Dense2dForwardPropagation*>(forward_propagation.get());

    const Tensor<type, 2>& activation_derivatives = dense2d_layer_forward_propagation->activation_derivatives;

    // Back propagation

    Dense2dBackPropagation* dense2d_back_propagation =
        static_cast<Dense2dBackPropagation*>(back_propagation.get());
    
    Tensor<type, 2>& weight_deltas = dense2d_back_propagation->weight_deltas;

    Tensor<type, 1>& bias_deltas = dense2d_back_propagation->bias_deltas;

    const bool& is_first_layer = dense2d_back_propagation->is_first_layer;

    Tensor<type, 2>& input_deltas = dense2d_back_propagation->input_deltas;

    if(activation_function != "Softmax")
        deltas.device(*thread_pool_device) = deltas * activation_derivatives;

    //if (batch_normalization) // @todo cpu batch normalization backward

    bias_deltas.device(*thread_pool_device) = deltas.sum(array_1(0));

    weight_deltas.device(*thread_pool_device) = inputs.contract(deltas, axes(0,0));

    if (!is_first_layer)
        input_deltas.device(*thread_pool_device) = deltas.contract(weights, axes(1,1));
}


void Dense2d::back_propagate_lm(const vector<pair<type*, dimensions>>& input_pairs,
                                        const vector<pair<type*, dimensions>>& delta_pairs,
                                        unique_ptr<LayerForwardPropagation>& forward_propagation,
                                        unique_ptr<LayerBackPropagationLM>& back_propagation) const
{
    const TensorMap<Tensor<type, 2>> inputs = tensor_map<2>(input_pairs[0]);
    TensorMap<Tensor<type, 2>> deltas = tensor_map<2>(delta_pairs[0]);
    
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

    Tensor<type, 2>& input_deltas = dense2d_layer_back_propagation_lm->input_deltas;

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
        input_deltas.device(*thread_pool_device) = deltas.contract(weights, axes(1,1));
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

        buffer << output_names[j] << " = " << activation_function << "( " << biases(j) << " + ";

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

    //cout << "Biases:" << endl;
    //cout << biases << endl;
    //cout << "Weights:" << endl;
    //cout << weights << endl;

    cout << "Activation function:" << endl;
    cout << activation_function << endl;
}


void Dense2d::from_XML(const XMLDocument& document)
{
    const XMLElement* dense2d_layer_element = document.FirstChildElement("Dense2d");

    if(!dense2d_layer_element)
        throw runtime_error("Dense2d element is nullptr.\n");

    set_label(read_xml_string(dense2d_layer_element, "Label"));

    const Index inputs_number = read_xml_index(dense2d_layer_element, "InputsNumber");
    const Index neurons_number = read_xml_index(dense2d_layer_element, "NeuronsNumber");

    set_input_dimensions({ inputs_number });
    set_output_dimensions({ neurons_number });

    set_activation_function(read_xml_string(dense2d_layer_element, "Activation"));

    bool use_batch_normalization = false;
    const XMLElement* bn_element = dense2d_layer_element->FirstChildElement("BatchNormalization");
    if (bn_element && bn_element->GetText())
        use_batch_normalization = (string(bn_element->GetText()) == "true");
    set_batch_normalization(use_batch_normalization);
    if (batch_normalization)
    {
        scales.resize(neurons_number);
        offsets.resize(neurons_number);
        moving_means.resize(neurons_number);
        moving_standard_deviations.resize(neurons_number);

        string_to_tensor<type, 1>(read_xml_string(dense2d_layer_element, "Scales"), scales);
        string_to_tensor<type, 1>(read_xml_string(dense2d_layer_element, "Offsets"), offsets);
        string_to_tensor<type, 1>(read_xml_string(dense2d_layer_element, "MovingMeans"), moving_means);
        string_to_tensor<type, 1>(read_xml_string(dense2d_layer_element, "MovingStandardDeviations"), moving_standard_deviations);
    }

    string_to_tensor<type, 1>(read_xml_string(dense2d_layer_element, "Biases"), biases);
    string_to_tensor<type, 2>(read_xml_string(dense2d_layer_element, "Weights"), weights);
}


void Dense2d::to_XML(XMLPrinter& printer) const
{
    printer.OpenElement("Dense2d");

    add_xml_element(printer, "Label", label);
    add_xml_element(printer, "InputsNumber", to_string(get_input_dimensions()[0]));
    add_xml_element(printer, "NeuronsNumber", to_string(get_output_dimensions()[0]));
    add_xml_element(printer, "Activation", activation_function);
    add_xml_element(printer, "BatchNormalization", batch_normalization ? "true" : "false");
    if (batch_normalization)
    {
        add_xml_element(printer, "Scales", tensor_to_string<type, 1>(scales));
        add_xml_element(printer, "Offsets", tensor_to_string<type, 1>(offsets));
        add_xml_element(printer, "MovingMeans", tensor_to_string<type, 1>(moving_means));
        add_xml_element(printer, "MovingStandardDeviations", tensor_to_string<type, 1>(moving_standard_deviations));
    }
    add_xml_element(printer, "Biases", tensor_to_string<type, 1>(biases));
    add_xml_element(printer, "Weights", tensor_to_string<type, 2>(weights));

    printer.CloseElement();  
}


void Dense2dForwardPropagation::set(const Index& new_batch_size, Layer *new_layer)
{
    if (!new_layer) return;

    layer = new_layer;
    
    batch_size = new_batch_size;

    const Index outputs_number = layer->get_outputs_number();

    outputs.resize(batch_size, outputs_number);

    activation_derivatives.resize(batch_size, outputs_number);

    activation_derivatives.setConstant((type)NAN);

    const auto* dense_layer = static_cast<const Dense2d*>(layer);

    if (dense_layer->get_batch_normalization())
    {
        means.resize(outputs_number);
        standard_deviations.resize(outputs_number);
    }
}


pair<type *, dimensions> Dense2dForwardPropagation::get_output_pair() const
{
    const dimensions output_dimensions = layer->get_output_dimensions();
    
    return pair<type *, dimensions>((type*)outputs.data(), {{batch_size, output_dimensions[0]}});
}


Dense2dForwardPropagation::Dense2dForwardPropagation(const Index& new_batch_size, Layer *new_layer)
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


Dense2dBackPropagation::Dense2dBackPropagation(const Index& new_batch_size, Layer *new_layer)
    : LayerBackPropagation()
{
    set(new_batch_size, new_layer);
}


void Dense2dBackPropagation::set(const Index& new_batch_size, Layer *new_layer)
{
    if (!new_layer) return;

    batch_size = new_batch_size;

    layer = new_layer;

    const Index outputs_number = layer->get_outputs_number();
    const Index inputs_number = layer->get_input_dimensions()[0];

    bias_deltas.resize(outputs_number);
    bias_deltas.setZero();

    weight_deltas.resize(inputs_number, outputs_number);
    weight_deltas.setZero();

    input_deltas.resize(batch_size, inputs_number);

    const auto* dense_layer = static_cast<const Dense2d*>(layer);

    if (dense_layer->get_batch_normalization())
    {
        bn_scale_deltas.resize(outputs_number);
        bn_scale_deltas.setZero();

        bn_offset_deltas.resize(outputs_number);
        bn_offset_deltas.setZero();
    }
}


vector<pair<type*, dimensions>> Dense2dBackPropagation::get_input_derivative_pairs() const
{
    const Index inputs_number = layer->get_input_dimensions()[0];

    return { {(type*)(input_deltas.data()), {batch_size, inputs_number}} };
}


vector<pair<type*, Index>> Dense2dBackPropagation::get_parameter_delta_pairs() const
{
    const auto* dense_layer = static_cast<const Dense2d*>(layer);

    vector<pair<type*, Index>> delta_pairs =
    {
        { const_cast<type*>(bias_deltas.data()), bias_deltas.size() },
        { const_cast<type*>(weight_deltas.data()), weight_deltas.size() }
    };

    if (dense_layer->get_batch_normalization())
    {
        delta_pairs.push_back({ const_cast<type*>(bn_scale_deltas.data()), bn_scale_deltas.size() });
        delta_pairs.push_back({ const_cast<type*>(bn_offset_deltas.data()), bn_offset_deltas.size() });
    }

    return delta_pairs;
}


void Dense2dBackPropagation::print() const
{
    cout << "Biases derivatives:" << endl
         << bias_deltas << endl
         << "Synaptic weights derivatives:" << endl
         << weight_deltas << endl;
}


Dense2dLayerBackPropagationLM::Dense2dLayerBackPropagationLM(const Index& new_batch_size,
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

    input_deltas.resize(batch_size, inputs_number);
}


vector<pair<type*, dimensions>> Dense2dLayerBackPropagationLM::get_input_derivative_pairs() const
{
    const Index inputs_number = layer->get_input_dimensions()[0];

    return {{(type*)(input_deltas.data()), {batch_size, inputs_number}}};
}


void Dense2dLayerBackPropagationLM::print() const
{
    cout << "Squared errors Jacobian: " << endl
        << squared_errors_Jacobian << endl;
    cout << "Input derivatives: " << endl
        << input_deltas << endl;
}


void Dense2d::normalization(Tensor<type, 1> &means,
                            Tensor<type, 1> &standard_deviations,
                            const Tensor<type, 2> &inputs,
                            Tensor<type, 2> &outputs) const
{

    const array<Index, 2> rows({outputs.dimension(0), 1});

    const array<int, 1> axis_x({0});

    means.device(*thread_pool_device) = outputs.mean(axis_x);

    standard_deviations.device(*thread_pool_device)
        = (outputs - means.broadcast(rows)).square().mean(axis_x).sqrt();

    outputs = inputs;// -means.broadcast(array<Index, 2>({ outputs.dimension(0), 1 }));
        //shifts.broadcast(rows);
        //+ (outputs - means.broadcast(rows))*scales.broadcast(rows)/standard_deviations.broadcast(rows);
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

    type* outputs_buffer = use_combinations ? combinations : outputs;

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
        outputs_buffer,
        batch_size);

    cudnnStatus_t status = cudnnAddTensor(cudnn_handle,
        &alpha,
        biases_tensor_descriptor,
        biases_device,
        &beta_add,
        output_tensor_descriptor,
        outputs_buffer);

    if (status != CUDNN_STATUS_SUCCESS)
        cerr << "Dense2d CUDA: cudnnAddTensor failed. Error: " << cudnnGetErrorString(status) << endl;

    // Batch Normalization

    if (batch_normalization)
    {
        cudnnStatus_t bn_status;

        if (is_training)
        {
            bn_status = cudnnBatchNormalizationForwardTraining(
                cudnn_handle,
                CUDNN_BATCHNORM_PER_ACTIVATION,
                &alpha, &beta_add,
                output_tensor_descriptor,
                outputs_buffer,
                output_tensor_descriptor,
                outputs_buffer,
                bn_tensor_descriptor,
                bn_scale_device,
                bn_offset_device,
                momentum,
                bn_running_mean_device,
                bn_running_variance_device,
                epsilon,
                dense2d_layer_forward_propagation_cuda->bn_saved_mean,
                dense2d_layer_forward_propagation_cuda->bn_saved_inv_variance
            );
            if (bn_status != CUDNN_STATUS_SUCCESS)
                cout << "cudnnBatchNormalizationForwardTraining failed: " << cudnnGetErrorString(bn_status) << endl;
        }
        else
        {
            bn_status = cudnnBatchNormalizationForwardInference(
                cudnn_handle,
                CUDNN_BATCHNORM_PER_ACTIVATION,
                &alpha, &beta_add,
                output_tensor_descriptor,
                outputs_buffer,
                output_tensor_descriptor,
                outputs_buffer,
                bn_tensor_descriptor,
                bn_scale_device,
                bn_offset_device,
                bn_running_mean_device,
                bn_running_variance_device,
                epsilon
            );
            if (bn_status != CUDNN_STATUS_SUCCESS)
                cout << "cudnnBatchNormalizationForwardInference failed: " << cudnnGetErrorString(bn_status) << endl;
        }
    }

    // Activations

    if (activation_function == "Linear")
        cudaMemcpy(outputs, combinations, batch_size * outputs_number * sizeof(type), cudaMemcpyDeviceToDevice);
    else if (activation_function == "Softmax")
        cudnnSoftmaxForward(cudnn_handle,
            CUDNN_SOFTMAX_ACCURATE,
            CUDNN_SOFTMAX_MODE_CHANNEL,
            &alpha,
            output_softmax_tensor_descriptor,
            combinations,
            &beta,
            output_softmax_tensor_descriptor,
            outputs);
    else
        cudnnActivationForward(cudnn_handle,
            activation_descriptor,
            &alpha,
            output_tensor_descriptor,
            outputs_buffer,
            &beta,
            output_tensor_descriptor,
            outputs);

    // Droput

    if (is_training && activation_function != "Softmax" && get_dropout_rate() > type(0))
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

    float* bias_deltas = dense2d_layer_back_propagation->bias_deltas_device;
    float* weight_deltas = dense2d_layer_back_propagation->weight_deltas_device;
    float* input_deltas = dense2d_layer_back_propagation->input_deltas;

    const cudnnTensorDescriptor_t& deltas_tensor_descriptor = dense2d_layer_back_propagation->deltas_tensor_descriptor;

    // Dropout

    if (get_dropout_rate() > type(0) && activation_function != "Softmax")
    {
        cudnnStatus_t status = cudnnDropoutBackward(cudnn_handle,
            dense2d_layer_forward_propagation_cuda->dropout_descriptor,
            deltas_tensor_descriptor,
            deltas_device[0],
            deltas_tensor_descriptor,
            deltas_device[0],
            dense2d_layer_forward_propagation_cuda->dropout_reserve_space,
            dense2d_layer_forward_propagation_cuda->dropout_reserve_space_size);

        if (status != CUDNN_STATUS_SUCCESS)
            cout << "cudnnDropoutBackward failed: " << cudnnGetErrorString(status) << endl;
    }

    // Error combinations derivatives

    if (dense2d_layer->get_activation_function() != "Linear" && dense2d_layer->get_activation_function() != "Softmax")
    {
        if (use_combinations)
        {
            cudnnStatus_t status = cudnnActivationBackward(cudnn_handle,
                activation_descriptor,
                &alpha,
                deltas_tensor_descriptor,
                outputs,
                deltas_tensor_descriptor,
                deltas_device[0],
                deltas_tensor_descriptor,
                combinations,
                &beta,
                deltas_tensor_descriptor,
                deltas_device[0]);

            if (status != CUDNN_STATUS_SUCCESS)
                cout << "cudnnActivationBackward failed: " << cudnnGetErrorString(status) << endl;
        }
        else
        {
            cudnnStatus_t status = cudnnActivationBackward(cudnn_handle,
                activation_descriptor,
                &alpha,
                deltas_tensor_descriptor,
                outputs,
                deltas_tensor_descriptor,
                deltas_device[0],
                deltas_tensor_descriptor,
                outputs,
                &beta,
                deltas_tensor_descriptor,
                deltas_device[0]);

            if (status != CUDNN_STATUS_SUCCESS)
                cout << "cudnnActivationBackward failed: " << cudnnGetErrorString(status) << endl;
        }
    }

    // Batch Normalization

    if (batch_normalization)
    {
        cudnnStatus_t bn_status = cudnnBatchNormalizationBackward(
            cudnn_handle,
            CUDNN_BATCHNORM_PER_ACTIVATION,
            &alpha, &beta,
            &alpha, &beta,
            dense2d_layer_forward_propagation_cuda->output_tensor_descriptor,
            use_combinations ? combinations : outputs,
            deltas_tensor_descriptor,
            deltas_device[0],
            deltas_tensor_descriptor,
            deltas_device[0],
            bn_tensor_descriptor,
            bn_scale_device,
            dense2d_layer_back_propagation->bn_scale_deltas_device,
            dense2d_layer_back_propagation->bn_offset_deltas_device,
            epsilon,
            dense2d_layer_forward_propagation_cuda->bn_saved_mean,
            dense2d_layer_forward_propagation_cuda->bn_saved_inv_variance
        );
        if (bn_status != CUDNN_STATUS_SUCCESS)
            cout << "cudnnBatchNormalizationBackward failed: " << cudnnGetErrorString(bn_status) << endl;
    }

    // Bias derivatives

    cublasSgemm(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N,
        outputs_number,
        1,
        batch_size,
        &alpha,
        deltas_device[0],
        batch_size,
        ones,
        batch_size,
        &beta,
        bias_deltas,
        outputs_number);

    // Weight derivatives

    cublasSgemm(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N,
        inputs_number,
        outputs_number,
        batch_size,
        &alpha,
        inputs_device[0],
        batch_size,
        deltas_device[0],
        batch_size,
        &beta,
        weight_deltas,
        inputs_number);

    // Input derivatives

    cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T,
        batch_size,
        inputs_number,
        outputs_number,
        &alpha,
        deltas_device[0],
        batch_size,
        weights_device,
        inputs_number,
        &beta,
        input_deltas,
        batch_size);
}


vector<pair<float*, Index>> Dense2d::get_parameter_pair_device() const
{
    vector<pair<float*, Index>> parameter_pairs =
    {
        {biases_device, biases.size()},
        {weights_device, weights.size()}
    };

    if (batch_normalization)
    {
        parameter_pairs.push_back({ bn_scale_device, scales.size() });
        parameter_pairs.push_back({ bn_offset_device, offsets.size() });
    }

    return parameter_pairs;
}


void Dense2d::allocate_parameters_device()
{
    cout << "Dense2d allocate_parameters_device:" << endl;
    const Index inputs_number = get_inputs_number();
    const Index outputs_number = get_outputs_number();

    //CHECK_CUDA(cudaMalloc(&biases_device, outputs_number * sizeof(float)));
    CUDA_MALLOC_AND_REPORT(biases_device, outputs_number * sizeof(float));
    //CHECK_CUDA(cudaMalloc(&weights_device, inputs_number * outputs_number * sizeof(float)));
    CUDA_MALLOC_AND_REPORT(weights_device, inputs_number * outputs_number * sizeof(float));

    if (batch_normalization)
    {
        //CHECK_CUDA(cudaMalloc(&bn_scale_device, outputs_number * sizeof(float)));
        CUDA_MALLOC_AND_REPORT(bn_scale_device, outputs_number * sizeof(float));

        //CHECK_CUDA(cudaMalloc(&bn_offset_device, outputs_number * sizeof(float)));
        CUDA_MALLOC_AND_REPORT(bn_offset_device, outputs_number * sizeof(float));

        //CHECK_CUDA(cudaMalloc(&bn_running_mean_device, outputs_number * sizeof(float)));
        CUDA_MALLOC_AND_REPORT(bn_running_mean_device, outputs_number * sizeof(float));

        //CHECK_CUDA(cudaMalloc(&bn_running_variance_device, outputs_number * sizeof(float)));
        CUDA_MALLOC_AND_REPORT(bn_running_variance_device, outputs_number * sizeof(float));
    }
}


void Dense2d::free_parameters_device()
{
    cudaFree(biases_device);
    cudaFree(weights_device);

    biases_device = nullptr;
    weights_device = nullptr;

    if (batch_normalization)
    {
        cudaFree(bn_scale_device);
        cudaFree(bn_offset_device);
        cudaFree(bn_running_mean_device);
        cudaFree(bn_running_variance_device);

        bn_scale_device = nullptr;
        bn_offset_device = nullptr;
        bn_running_mean_device = nullptr;
        bn_running_variance_device = nullptr;

        cudnnDestroyTensorDescriptor(bn_tensor_descriptor);
        bn_tensor_descriptor = nullptr;
    }
}


void Dense2d::copy_parameters_device()
{
    if (!biases_device) cout << "Biases device is null" << endl;

    if (!weights_device) cout << "Weights device is null" << endl;

    CHECK_CUDA(cudaMemcpy(biases_device, biases.data(), biases.size() * sizeof(type), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(weights_device, weights.data(), weights.size() * sizeof(type), cudaMemcpyHostToDevice));

    if (batch_normalization)
    {
        CHECK_CUDA(cudaMemcpy(bn_scale_device, scales.data(), scales.size() * sizeof(type), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(bn_offset_device, offsets.data(), offsets.size() * sizeof(type), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(bn_running_mean_device, moving_means.data(), moving_means.size() * sizeof(type), cudaMemcpyHostToDevice));
        Tensor<type, 1> moving_variances = moving_standard_deviations.square();
        CHECK_CUDA(cudaMemcpy(bn_running_variance_device, moving_variances.data(), moving_variances.size() * sizeof(type), cudaMemcpyHostToDevice));
    }
}


void Dense2d::copy_parameters_host()
{
    if (!biases_device) cout << "Biases is null" << endl;
    if (!weights_device) cout << "Synaptic weights is null" << endl;

    CHECK_CUDA(cudaMemcpy(biases.data(), biases_device, biases.size() * sizeof(type), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(weights.data(), weights_device, weights.size() * sizeof(type), cudaMemcpyDeviceToHost));

    if (batch_normalization)
    {
        CHECK_CUDA(cudaMemcpy(scales.data(), bn_scale_device, scales.size() * sizeof(type), cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(offsets.data(), bn_offset_device, offsets.size() * sizeof(type), cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(moving_means.data(), bn_running_mean_device, moving_means.size() * sizeof(type), cudaMemcpyDeviceToHost));
        Tensor<type, 1> moving_variances(moving_standard_deviations.size());
        CHECK_CUDA(cudaMemcpy(moving_variances.data(), bn_running_variance_device, moving_variances.size() * sizeof(type), cudaMemcpyDeviceToHost));
        moving_standard_deviations = moving_variances.sqrt();
    }
}


Dense2dForwardPropagationCuda::Dense2dForwardPropagationCuda(const Index& new_batch_size, Layer* new_layer)
    : LayerForwardPropagationCuda()
{
    set(new_batch_size, new_layer);
}


void Dense2dForwardPropagationCuda::set(const Index& new_batch_size, Layer* new_layer)
{
    if (!new_layer) return;

    cout << "Dense2dForwardPropagationCuda set:" << endl;

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

    if (dense2d_layer->use_combinations)
    {
        //CHECK_CUDA(cudaMalloc(&combinations, batch_size * outputs_number * sizeof(float)));
        CUDA_MALLOC_AND_REPORT(combinations, batch_size * outputs_number * sizeof(float));
    }
    //CHECK_CUDA(cudaMalloc(&outputs, batch_size * outputs_number * sizeof(float)));
    CUDA_MALLOC_AND_REPORT(outputs, batch_size * outputs_number * sizeof(float));

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
        {
            random_device rd;
            auto now = chrono::high_resolution_clock::now().time_since_epoch().count();

            dropout_seed = (static_cast<unsigned long long>(rd()) << 1) ^ static_cast<unsigned long long>(now);
        }
        cudnnCreateDropoutDescriptor(&dropout_descriptor);

        cudnnDropoutGetStatesSize(dense2d_layer->get_cudnn_handle(), &dropout_states_size);

        CHECK_CUDA(cudaMalloc(&dropout_states, dropout_states_size));

        cudnnSetDropoutDescriptor(dropout_descriptor,
            dense2d_layer->get_cudnn_handle(),
            static_cast<float>(dense2d_layer->get_dropout_rate()),
            dropout_states,
            dropout_states_size,
            dropout_seed);

        cudnnDropoutGetReserveSpaceSize(output_tensor_descriptor, &dropout_reserve_space_size);
        CHECK_CUDA(cudaMalloc(&dropout_reserve_space, dropout_reserve_space_size));
    }

    // Batch Normalization

    if (dense2d_layer->get_batch_normalization())
    {
        //CHECK_CUDA(cudaMalloc(&bn_saved_mean, outputs_number * sizeof(float)));
        CUDA_MALLOC_AND_REPORT(bn_saved_mean, outputs_number * sizeof(float));
        //CHECK_CUDA(cudaMalloc(&bn_saved_inv_variance, outputs_number * sizeof(float)));
        CUDA_MALLOC_AND_REPORT(bn_saved_inv_variance, outputs_number * sizeof(float));
    }
}

void Dense2dForwardPropagationCuda::print() const
{
    // @todo
}


void Dense2dForwardPropagationCuda::free()
{
    if (combinations != nullptr)
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

    if (bn_saved_mean) cudaFree(bn_saved_mean);
    if (bn_saved_inv_variance) cudaFree(bn_saved_inv_variance);
}


Dense2dBackPropagationCuda::Dense2dBackPropagationCuda(const Index& new_batch_size, Layer* new_layer)
    : LayerBackPropagationCuda()
{
    set(new_batch_size, new_layer);
}


void Dense2dBackPropagationCuda::set(const Index& new_batch_size, Layer* new_layer)
{
    if (!new_layer) return;

    cout << "Dense2dBackPropagationCuda set:" << endl;

    batch_size = new_batch_size;

    layer = new_layer;

    const Index outputs_number = layer->get_outputs_number();
    const Index inputs_number = layer->get_input_dimensions()[0];

    // Ones

    CHECK_CUDA(cudaMalloc(&ones, batch_size * sizeof(float)));
    vector<float> ones_host(batch_size, 1.0f);
    CHECK_CUDA(cudaMemcpy(ones, ones_host.data(), batch_size * sizeof(float), cudaMemcpyHostToDevice));

    // Parameters

    //CHECK_CUDA(cudaMalloc(&bias_deltas_device, outputs_number * sizeof(float)));
    CUDA_MALLOC_AND_REPORT(bias_deltas_device, outputs_number * sizeof(float));
    //CHECK_CUDA(cudaMalloc(&weight_deltas_device, inputs_number * outputs_number * sizeof(float)));
    CUDA_MALLOC_AND_REPORT(weight_deltas_device, inputs_number * outputs_number * sizeof(float));

    // Input deltas
    
    //CHECK_CUDA(cudaMalloc(&input_deltas, batch_size * inputs_number * sizeof(float)));
    CUDA_MALLOC_AND_REPORT(input_deltas, batch_size * inputs_number * sizeof(float));

    // Deltas

    cudnnCreateTensorDescriptor(&deltas_tensor_descriptor);

    cudnnSetTensor4dDescriptor(deltas_tensor_descriptor,
        CUDNN_TENSOR_NCHW,
        CUDNN_DATA_FLOAT,
        batch_size,
        outputs_number,
        1,
        1);

    // Batch Normalization

    const auto* dense_layer = static_cast<const Dense2d*>(layer);

    if (dense_layer->get_batch_normalization())
    {
        //CHECK_CUDA(cudaMalloc(&bn_scale_deltas_device, outputs_number * sizeof(float)));
        CUDA_MALLOC_AND_REPORT(bn_scale_deltas_device, outputs_number * sizeof(float));
        //CHECK_CUDA(cudaMalloc(&bn_offset_deltas_device, outputs_number * sizeof(float)));
        CUDA_MALLOC_AND_REPORT(bn_offset_deltas_device, outputs_number * sizeof(float));
    }
}


vector<pair<float*, Index>> Dense2dBackPropagationCuda::get_parameter_delta_pair_device() const
{
    const auto* dense_layer = static_cast<const Dense2d*>(layer);
    const Index weight_deltas_size = dense_layer->get_input_dimensions()[0] * dense_layer->get_output_dimensions()[0];
    const Index bias_deltas_size = dense_layer->get_output_dimensions()[0];

    vector<pair<float*, Index>> delta_pairs =
    {
        { bias_deltas_device,   bias_deltas_size },
        { weight_deltas_device, weight_deltas_size }
    };

    if (dense_layer->get_batch_normalization())
    {
        delta_pairs.push_back({ bn_scale_deltas_device, dense_layer->get_scales().size() });
        delta_pairs.push_back({ bn_offset_deltas_device, dense_layer->get_offsets().size() });
    }

    return delta_pairs;
}


void Dense2dBackPropagationCuda::print() const
{
    // @todo
}


void Dense2dBackPropagationCuda::free()
{
    cudaFree(bias_deltas_device);
    cudaFree(weight_deltas_device);
    cudaFree(input_deltas);
    cudaFree(ones);
    if (bn_scale_deltas_device) cudaFree(bn_scale_deltas_device);
    if (bn_offset_deltas_device) cudaFree(bn_offset_deltas_device);

    cudnnDestroyTensorDescriptor(deltas_tensor_descriptor);
}

REGISTER(LayerForwardPropagationCuda, Dense2dForwardPropagationCuda, "Dense2d")
REGISTER(LayerBackPropagationCuda, Dense2dBackPropagationCuda, "Dense2d")

#endif

REGISTER(Layer, Dense2d, "Dense2d")
REGISTER(LayerForwardPropagation, Dense2dForwardPropagation, "Dense2d")
REGISTER(LayerBackPropagation, Dense2dBackPropagation, "Dense2d")

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
// License along with this library; if not, write to the Free Software Foundation.
// Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
