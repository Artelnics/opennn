//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   D E N S E   L A Y E R   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#pragma once

#include "layer.h"
#include "math_utilities.h"
#include "forward_propagation.h"
#include "back_propagation.h"

namespace opennn
{

template<int Rank>
class Dense final : public Layer
{
private:

    Index neurons_number;
    ActivationFunction activation_function = ActivationFunction::HyperbolicTangent;

    VectorR running_means;
    VectorR running_variances;

    bool batch_normalization = false;

    type momentum = type(0.9);

    type dropout_rate = type(0);

#ifdef OPENNN_WITH_CUDA

    cudnnActivationDescriptor_t activation_descriptor = nullptr;
    cudnnDropoutDescriptor_t dropout_descriptor = nullptr;

#endif

    enum Parameters {Bias, Weight, Gamma, Beta};

    vector<Shape> get_parameter_shapes() const override
    {
        const Index input_dimension = input_shape.back();

        return {{neurons_number},                            // Biases
                {input_dimension, neurons_number},           // Weights
                {batch_normalization ? neurons_number : 0},  // Gammas
                {batch_normalization ? neurons_number : 0}}; // Betas
    }

    enum Forward {Input, NormalizedOutput, Output};

    vector<Shape> get_forward_shapes(const Index batch_size) const override
    {
        // views[layer] slots: 0=Input (wired), 1=NormalizedOutput, 2=Output
        // shapes[k] → slot k+1, so we return 2 shapes.
        const Shape output_shape = Shape{batch_size}.append(get_output_shape());

        if(batch_normalization)
            return {output_shape,   // slot 1: NormalizedOutput
                    output_shape};  // slot 2: Output
        else
            return {Shape{},        // slot 1: NormalizedOutput (unused, placeholder)
                    output_shape};  // slot 2: Output
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
          const string& new_label = "dense2d_layer")
    {
        set(new_input_shape, new_output_shape, new_activation_function, new_batch_normalization, new_label);
    }

    Dense(const Index input_sequence_length,
          Index embedding_dimension,
          Index feed_forward_dimension,
          const string& new_activation_function,
          const string& new_label)
    {
        set({input_sequence_length, embedding_dimension},
            {feed_forward_dimension},
            new_activation_function,
            false,
            new_label);
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
        return activation_function;
    }

    ActivationFunction get_output_activation() const override
    {
        return activation_function;
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

        const Index outputs_number = get_outputs_number();

        if (batch_normalization)
        {
            running_means.resize(outputs_number);
            running_variances.resize(outputs_number);
        }

        set_label(new_label);

        name = "Dense" + to_string(Rank) + "d";
        layer_type = (Rank == 2) ? LayerType::Dense2d : LayerType::Dense3d;

#ifdef OPENNN_WITH_CUDA
        // @todo batch normalization device descriptors
#endif
    }

    void set_parameters_glorot() override
    {
        const type limit = sqrt(6.0 / (get_inputs_number() + get_outputs_number()));

        VectorMap(parameters[Bias].data, parameters[Bias].size()).setZero();

        set_random_uniform(VectorMap(parameters[Weight].data, parameters[Weight].size()), -limit, limit);

        VectorMap(parameters[Gamma].data, parameters[Gamma].size()).setConstant(1.0);

        VectorMap(parameters[Beta].data, parameters[Beta].size()).setZero();
    }

    void set_parameters_random() override
    {
        VectorMap(parameters[Bias].data, parameters[Bias].size()).setZero();

        set_random_uniform(VectorMap(parameters[Weight].data, parameters[Weight].size()));

        VectorMap(parameters[Gamma].data, parameters[Gamma].size()).setConstant(1.0);

        VectorMap(parameters[Beta].data, parameters[Beta].size()).setZero();
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

    void set_activation_function(const string& name)
    {
        activation_function = string_to_activation(name);

#ifdef OPENNN_WITH_CUDA

        if (activation_descriptor == nullptr && activation_function != ActivationFunction::Softmax)
            cudnnCreateActivationDescriptor(&activation_descriptor);

        cudnnActivationMode_t activation_mode = CUDNN_ACTIVATION_IDENTITY;
        double relu_ceiling = 0.0;

        switch(activation_function)
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

        if (activation_function != ActivationFunction::Softmax)
            cudnnSetActivationDescriptor(activation_descriptor, activation_mode, CUDNN_PROPAGATE_NAN, relu_ceiling);

#endif
    }

    void set_dropout_rate(const type new_dropout_rate)
    {
        if (new_dropout_rate < type(0) || new_dropout_rate >= type(1))
            throw runtime_error("Dropout rate must be in [0,1).");

        dropout_rate = new_dropout_rate;
    }

    void set_batch_normalization(bool new_batch_normalization)
    {
        batch_normalization = new_batch_normalization;
    }

    void forward_propagate(ForwardPropagation& forward_propagation, size_t layer, bool is_training) override
    {
        const TensorView& input = forward_propagation.views[layer][Input][0];
        TensorView& output = forward_propagation.views[layer][Output][0];

        const TensorView& weights = parameters[Weight];
        const TensorView& biases = parameters[Bias];

        const TensorView& gammas = parameters[Gamma];
        const TensorView& betas = parameters[Beta];

        combination(input, weights, biases, output);
/*
        if (batch_normalization)
            is_training
                ? batch_normalization_training(output, gammas, betas, running_means, running_variances)
                : batch_normalization_inference(output, gammas, betas, running_means, running_variances);
*/
        ActivationArguments act_args;
        act_args.activation_function = activation_function;
#ifdef OPENNN_WITH_CUDA
        act_args.activation_descriptor = activation_descriptor;
#endif
        activation(output, act_args);

        if(is_training && dropout_rate > type(0))
            dropout(output, dropout_rate);
    }

    void back_propagate(ForwardPropagation& forward_propagation,
                        BackPropagation& back_propagation,
                        size_t layer) const override
    {
        const TensorView& input = forward_propagation.views[layer][Input][0];
        const TensorView& output = forward_propagation.views[layer][Output][0];

        TensorView& delta = back_propagation.backward_views[layer][OutputGradients][0];

#ifndef OPENNN_WITH_CUDA
        activation_gradient(output, delta, delta, activation_function);
#else
        activation_gradient(output, delta, delta, activation_function, activation_descriptor);
#endif

        TensorView& bias_gradient = back_propagation.gradient_views[layer][Bias];
        TensorView& weight_gradient = back_propagation.gradient_views[layer][Weight];

        // Flatten to 2D for gradient computation (handles Dense3d where input/delta are 3D)
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

    string get_expression(const vector<string>& new_input_names = vector<string>(),
                          const vector<string>& new_output_names = vector<string>()) const override
    {
        const vector<string> input_names = new_input_names.empty()
        ? get_default_feature_names()
        : new_input_names;

        const vector<string> output_names = new_output_names.empty()
                                                ? get_default_output_names()
                                                : new_output_names;

        const Index inputs_number = get_inputs_number();
        const Index outputs_number = get_outputs_number();

        if (parameters[Bias].data == nullptr || parameters[Weight].data == nullptr) return "";

        ostringstream buffer;
/*
        for(Index j = 0; j < outputs_number; j++)
        {
            buffer << output_names[j] << " = " << activation_function << "( " << parameters[Bias].data[j] << " + ";

            for(Index i = 0; i < inputs_number; i++)
            {
                const Index weight_index = i * outputs_number + j;

                buffer << "(" << parameters[Weight].data[weight_index] << "*" << input_names[i] << ")";

                if (i < inputs_number - 1) buffer << " + ";
            }

            buffer << " );\n";
        }
*/
        return buffer.str();
    }

    void from_XML(const XmlDocument& document) override
    {
        const XmlElement* dense2d_layer_element = get_xml_root(document, name);

        set_label(read_xml_string(dense2d_layer_element, "Label"));

        const Index inputs_number = read_xml_index(dense2d_layer_element, "InputsNumber");
        const Index neurons_number = read_xml_index(dense2d_layer_element, "NeuronsNumber");

        if constexpr (Rank == 3)
        {
            const Index input_sequence_length = read_xml_index(dense2d_layer_element, "InputSequenceLength");
            set_input_shape({ input_sequence_length, inputs_number });
        }
        else
            set_input_shape({ inputs_number });

        set_output_shape({ neurons_number });

        set_activation_function(read_xml_string(dense2d_layer_element, "Activation"));

        bool use_batch_normalization = false;
        const XmlElement* bn_element = dense2d_layer_element->first_child_element("BatchNormalization");

        if (bn_element && bn_element->get_text())
            use_batch_normalization = (string(bn_element->get_text()) == "true");
        set_batch_normalization(use_batch_normalization);

        if (batch_normalization)
        {
            running_means.resize(neurons_number);
            running_variances.resize(neurons_number);

            string_to_vector(read_xml_string(dense2d_layer_element, "RunningMeans"), running_means);
            string_to_vector(read_xml_string(dense2d_layer_element, "RunningStandardDeviations"), running_variances);
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
/*
        add_xml_element(printer, "Activation", activation_function);
        add_xml_element(printer, "BatchNormalization", batch_normalization ? "true" : "false");
*/
        if (batch_normalization)
        {
            add_xml_element(printer, "RunningMeans", vector_to_string(running_means));
            add_xml_element(printer, "RunningStandardDeviations", vector_to_string(running_variances));
        }

        printer.close_element();
    }

    bool use_combinations = true;
};

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
