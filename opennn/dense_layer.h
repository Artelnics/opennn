//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   D E N S E   L A Y E R   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#pragma once

#include "layer.h"

namespace opennn
{

template<int Rank>
class Dense final : public Layer
{
private:

    Index neurons_number;

    bool batch_normalization = false;

    type momentum = type(0.9);

    type dropout_rate = type(0);

    DropoutArguments dropout_args;

    ActivationArguments activation_arguments;

    enum Parameters {Bias, Weight, Gamma, Beta};

    vector<Shape> get_parameter_shapes() const override
    {
        const Index input_dimension = input_shape.back();

        return {{neurons_number},                            // Biases
                {input_dimension, neurons_number},           // Weights
                {batch_normalization ? neurons_number : 0},  // Gammas
                {batch_normalization ? neurons_number : 0}}; // Betas
    }

    enum States {RunningMean, RunningVariance};

    vector<Shape> get_state_shapes() const override
    {
        if (!batch_normalization) return {};
        return {{neurons_number},   // RunningMean
                {neurons_number}};  // RunningVariance
    }

    enum Forward {Input, Combination, BatchNormMean, BatchNormInverseVariance, Output};

    vector<Shape> get_forward_shapes(const Index batch_size) const override
    {
        const Shape output_shape = Shape{batch_size}.append(get_output_shape());

        if(batch_normalization)
            return {output_shape,             // Combination
                    Shape{neurons_number},    // BatchNormMean
                    Shape{neurons_number},    // BatchNormInverseVariance
                    output_shape};            // Output
        else
            return {Shape{},                  // Combination (unused)
                    Shape{},                  // BatchNormMean (unused)
                    Shape{},                  // BatchNormInverseVariance (unused)
                    output_shape};            // Output
    }

    enum Backward {OutputGradients, InputGradients};

    vector<Shape> get_backward_shapes(Index batch_size) const override
    {
        return {Shape{batch_size}.append(get_input_shape())};
    }

public:

    Dense(const Shape& new_input_shape = {0},
          const Shape& new_output_shape = {0},
          const string& new_activation_function = "HyperbolicTangent",
          bool new_batch_normalization = false,
          const string& new_label = "dense_layer")
    {
        set(new_input_shape,
            new_output_shape,
            new_activation_function,
            new_batch_normalization,
            new_label);
    }

    ~Dense() override
    {
#ifdef OPENNN_WITH_CUDA
        if (activation_arguments.activation_descriptor)
            cudnnDestroyActivationDescriptor(activation_arguments.activation_descriptor);
        if (dropout_args.descriptor) cudnnDestroyDropoutDescriptor(dropout_args.descriptor);
        if (dropout_args.states) cudaFree(dropout_args.states);
        if (dropout_args.reserve_space) cudaFree(dropout_args.reserve_space);
#endif
    }

    Shape get_output_shape() const override
    {
        if constexpr (Rank == 2)
            return {neurons_number};
        else
            return {input_shape[0], neurons_number};
    }

    Index get_sequence_length() const
    {
        if constexpr (Rank == 3)
            return input_shape[0];
        else
            return 1;
    }

    const ActivationFunction& get_activation_function() const
    {
        return activation_arguments.activation_function;
    }

    ActivationFunction get_output_activation() const override
    {
        return activation_arguments.activation_function;
    }

    void set(const Shape& new_input_shape = {},
             const Shape& new_output_shape = {},
             const string& new_activation_function = "HyperbolicTangent",
             bool new_batch_normalization = false,
             const string& new_label = "dense_layer")
    {
        if (new_input_shape.rank() != Rank - 1)
            throw runtime_error("Input shape size must be " + to_string(Rank - 1));

        if (new_output_shape.rank() != 1)
            throw runtime_error("Output shape size is not 1");

        input_shape = new_input_shape;
        neurons_number = new_output_shape.back();

        set_activation_function(new_activation_function);

        set_batch_normalization(new_batch_normalization);

        set_label(new_label);

        name = "Dense" + to_string(Rank) + "d";
        layer_type = (Rank == 2) ? LayerType::Dense2d : LayerType::Dense3d;
    }

    void set_input_shape(const Shape& new_input_shape) override
    {
        if (new_input_shape.rank() != Rank - 1)
            throw runtime_error("Input shape size must be " + to_string(Rank - 1));

        input_shape = new_input_shape;
    }

    void set_output_shape(const Shape& new_output_shape) override
    {
        neurons_number = new_output_shape.back();
    }

    void set_batch_normalization(bool new_batch_normalization)
    {
        batch_normalization = new_batch_normalization;
    }

    void set_activation_function(const string& name)
    {
        ActivationFunction function = string_to_activation(name);

        if (function == ActivationFunction::Softmax && get_outputs_number() == 1)
            function = ActivationFunction::Sigmoid;

        activation_arguments.activation_function = function;

#ifdef OPENNN_WITH_CUDA

        cudnnActivationDescriptor_t& descriptor = activation_arguments.activation_descriptor;

        if (descriptor == nullptr && function != ActivationFunction::Softmax)
            cudnnCreateActivationDescriptor(&descriptor);

        cudnnActivationMode_t activation_mode = CUDNN_ACTIVATION_IDENTITY;
        double relu_ceiling = 0.0;

        switch(function)
        {
        case ActivationFunction::Linear:
            activation_mode = CUDNN_ACTIVATION_IDENTITY; break;
        case ActivationFunction::Sigmoid:
            activation_mode = CUDNN_ACTIVATION_SIGMOID; break;
        case ActivationFunction::HyperbolicTangent:
            activation_mode = CUDNN_ACTIVATION_TANH; break;
        case ActivationFunction::RectifiedLinear:
            activation_mode = CUDNN_ACTIVATION_RELU; break;
        case ActivationFunction::ScaledExponentialLinear:
            activation_mode = CUDNN_ACTIVATION_ELU; break;
        default: break;
        }

        if (function != ActivationFunction::Softmax)
            cudnnSetActivationDescriptor(descriptor, activation_mode, CUDNN_PROPAGATE_NAN, relu_ceiling);
#endif
    }

    void set_dropout_rate(const type new_dropout_rate)
    {
        if (new_dropout_rate < type(0) || new_dropout_rate >= type(1))
            throw runtime_error("Dropout rate must be in [0,1).");

        dropout_rate = new_dropout_rate;
        dropout_args.rate = new_dropout_rate;
    }

    void set_parameters_glorot() override
    {
        const type limit = sqrt(6.0 / (get_inputs_number() + get_outputs_number()));

        VectorMap(parameters[Bias].data, parameters[Bias].size()).setZero();

        set_random_uniform(VectorMap(parameters[Weight].data, parameters[Weight].size()), -limit, limit);

        VectorMap(parameters[Gamma].data, parameters[Gamma].size()).setConstant(1.0);

        VectorMap(parameters[Beta].data, parameters[Beta].size()).setZero();

        if (batch_normalization)
        {
            VectorMap(states[RunningMean].data, states[RunningMean].size()).setZero();
            VectorMap(states[RunningVariance].data, states[RunningVariance].size()).setOnes();
        }
    }

    void set_parameters_random() override
    {
        VectorMap(parameters[Bias].data, parameters[Bias].size()).setZero();

        set_random_uniform(VectorMap(parameters[Weight].data, parameters[Weight].size()));

        VectorMap(parameters[Gamma].data, parameters[Gamma].size()).setConstant(1.0);

        VectorMap(parameters[Beta].data, parameters[Beta].size()).setZero();

        if (batch_normalization)
        {
            VectorMap(states[RunningMean].data, states[RunningMean].size()).setZero();
            VectorMap(states[RunningVariance].data, states[RunningVariance].size()).setOnes();
        }
    }

#ifdef OPENNN_WITH_CUDA

    void init_cuda(Index batch_size)
    {
        // Dropout

        if (dropout_rate > type(0))
        {
            cudnnTensorDescriptor_t temp_desc;
            cudnnCreateTensorDescriptor(&temp_desc);

            const Index output_size = get_outputs_number();
            const Index seq_len = get_sequence_length();

            cudnnSetTensor4dDescriptor(temp_desc, CUDNN_TENSOR_NHWC, CUDNN_DATA_FLOAT,
                                       static_cast<int>(batch_size),
                                       static_cast<int>(output_size),
                                       static_cast<int>(seq_len),
                                       1);

            if (dropout_args.descriptor) { cudnnDestroyDropoutDescriptor(dropout_args.descriptor); dropout_args.descriptor = nullptr; }
            if (dropout_args.states) { cudaFree(dropout_args.states); dropout_args.states = nullptr; }
            if (dropout_args.reserve_space) { cudaFree(dropout_args.reserve_space); dropout_args.reserve_space = nullptr; }

            CHECK_CUDNN(cudnnCreateDropoutDescriptor(&dropout_args.descriptor));
            CHECK_CUDNN(cudnnDropoutGetStatesSize(Device::get_cudnn_handle(), &dropout_args.states_size));
            CHECK_CUDA(cudaMalloc(&dropout_args.states, dropout_args.states_size));
            CHECK_CUDNN(cudnnSetDropoutDescriptor(dropout_args.descriptor, Device::get_cudnn_handle(),
                                                  dropout_rate, dropout_args.states, dropout_args.states_size,
                                                  static_cast<unsigned long long>(random_integer(0, 1 << 30))));
            CHECK_CUDNN(cudnnDropoutGetReserveSpaceSize(temp_desc, &dropout_args.reserve_size));
            CHECK_CUDA(cudaMalloc(&dropout_args.reserve_space, dropout_args.reserve_size));

            dropout_args.rate = dropout_rate;

            cudnnDestroyTensorDescriptor(temp_desc);
        }
    }
#endif

    void forward_propagate(ForwardPropagation& forward_propagation, size_t layer, bool is_training) override
    {
        const TensorView& input = forward_propagation.views[layer][Input][0];
        TensorView& output = forward_propagation.views[layer][Output][0];

        const TensorView& weights = parameters[Weight];
        const TensorView& biases = parameters[Bias];

        const TensorView& gammas = parameters[Gamma];
        const TensorView& betas = parameters[Beta];

        if (batch_normalization)
        {
            TensorView& combination_output = forward_propagation.views[layer][Combination][0];
            combination(input, weights, biases, combination_output);

            if (is_training)
            {
                TensorView& bn_mean = forward_propagation.views[layer][BatchNormMean][0];
                TensorView& bn_inverse_variance = forward_propagation.views[layer][BatchNormInverseVariance][0];
                batch_normalization_training(combination_output, gammas, betas,
                                             states[RunningMean], states[RunningVariance],
                                             bn_mean, bn_inverse_variance,
                                             output, momentum);
            }
            else
            {
                batch_normalization_inference(combination_output, gammas, betas,
                                              states[RunningMean], states[RunningVariance],
                                              output);
            }
        }
        else
        {
            combination(input, weights, biases, output);
        }

        if (is_training && dropout_rate > type(0))
            dropout(output, dropout_args);

        activation(output, activation_arguments);
    }

    void back_propagate(ForwardPropagation& forward_propagation,
                        BackPropagation& back_propagation,
                        size_t layer) const override
    {
        const TensorView& input = forward_propagation.views[layer][Input][0];
        const TensorView& output = forward_propagation.views[layer][Output][0];

        TensorView& delta = back_propagation.backward_views[layer][OutputGradients][0];

        activation_gradient(output, delta, delta, activation_arguments);

        if (dropout_rate > type(0))
            dropout_gradient(delta, dropout_args, delta);

        if (batch_normalization)
        {
            const TensorView& bn_mean = forward_propagation.views[layer][BatchNormMean][0];
            const TensorView& bn_inverse_variance = forward_propagation.views[layer][BatchNormInverseVariance][0];
            const TensorView& gammas = parameters[Gamma];
            TensorView& gamma_gradient = back_propagation.gradient_views[layer][Gamma];
            TensorView& beta_gradient = back_propagation.gradient_views[layer][Beta];

            const TensorView& combination_output = forward_propagation.views[layer][Combination][0];
            batch_normalization_backward(combination_output, output, delta,
                                         bn_mean, bn_inverse_variance,
                                         gammas, gamma_gradient, beta_gradient,
                                         delta);
        }

        TensorView& bias_gradient = back_propagation.gradient_views[layer][Bias];
        TensorView& weight_gradient = back_propagation.gradient_views[layer][Weight];

        const Index total_rows = input.size() / input.shape[input.get_rank() - 1];
        TensorView input_2d(input.data, {total_rows, input.shape[input.get_rank() - 1]});
        TensorView delta_2d(delta.data, {total_rows, delta.shape[delta.get_rank() - 1]});

        multiply(input_2d, true, delta_2d, false, weight_gradient);
        sum(delta_2d, bias_gradient);

        if (!is_first_layer)
        {
            TensorView& input_gradient = back_propagation.backward_views[layer][InputGradients][0];
            TensorView ig_2d(input_gradient.data, {total_rows, input_gradient.shape[input_gradient.get_rank() - 1]});
            multiply(delta_2d, false, parameters[Weight], true, ig_2d);
        }
    }

    void from_XML(const XmlDocument& document) override
    {
        const XmlElement* dense_layer_element = get_xml_root(document, name);

        set_label(read_xml_string(dense_layer_element, "Label"));

        const Index inputs_number = read_xml_index(dense_layer_element, "InputsNumber");
        const Index neurons_number = read_xml_index(dense_layer_element, "NeuronsNumber");

        if constexpr (Rank == 3)
        {
            const Index input_sequence_length = read_xml_index(dense_layer_element, "InputSequenceLength");
            set_input_shape({ input_sequence_length, inputs_number });
        }
        else
            set_input_shape({ inputs_number });

        set_output_shape({ neurons_number });

        set_activation_function(read_xml_string(dense_layer_element, "Activation"));

        const XmlElement* bn_element = dense_layer_element->first_child_element("BatchNormalization");
        const bool use_batch_normalization = bn_element && bn_element->get_text()
                                          && string(bn_element->get_text()) == "true";
        set_batch_normalization(use_batch_normalization);

        if (batch_normalization && states.size() > RunningVariance)
        {
            VectorR tmp;

            string_to_vector(read_xml_string(dense_layer_element, "RunningMeans"), tmp);
            VectorMap(states[RunningMean].data, states[RunningMean].size()) = tmp;

            string_to_vector(read_xml_string(dense_layer_element, "RunningVariances"), tmp);
            VectorMap(states[RunningVariance].data, states[RunningVariance].size()) = tmp;
        }
    }

    void to_XML(XmlPrinter& printer) const override
    {
        printer.open_element(name.c_str());

        add_xml_element(printer, "Label", label);

        if constexpr (Rank == 3)
        {
            add_xml_element(printer, "InputSequenceLength", to_string(get_input_shape()[0]));
            add_xml_element(printer, "InputsNumber", to_string(get_input_shape()[1]));
            add_xml_element(printer, "NeuronsNumber", to_string(get_output_shape()[1]));
        }
        else
        {
            add_xml_element(printer, "InputsNumber", to_string(get_input_shape()[0]));
            add_xml_element(printer, "NeuronsNumber", to_string(get_output_shape()[0]));
        }

        add_xml_element(printer, "Activation", activation_function_map().to_string(activation_arguments.activation_function));
        add_xml_element(printer, "BatchNormalization", batch_normalization ? "true" : "false");

        if (batch_normalization && states.size() > RunningVariance)
        {
            add_xml_element(printer, "RunningMeans", vector_to_string(states[RunningMean].as_vector()));
            add_xml_element(printer, "RunningVariances", vector_to_string(states[RunningVariance].as_vector()));
        }

        printer.close_element();
    }
};

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
