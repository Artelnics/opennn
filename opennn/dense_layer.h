//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   D E N S E   L A Y E R   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#pragma once

#include "layer.h"
#include "random_utilities.h"
#include "math_utilities.h"
#include "loss.h"

namespace opennn
{
template<int Rank> class Dense;

struct DenseBackPropagationLM final : LayerBackPropagationLM
{
    void initialize() override
    {
        input_gradients = {{nullptr, Shape{batch_size}.append(layer->get_input_shape())}};

        squared_errors_Jacobian.shape = {batch_size, layer->get_parameters_number()};
    }

    vector<TensorView*> get_gradient_views() override
    {
        return {&squared_errors_Jacobian};
    }

    TensorView squared_errors_Jacobian;
};


struct Dense2dForwardPropagationLM;

template<int Rank>
class Dense final : public Layer
{
private:

    Index neurons_number;

    VectorR running_means;
    VectorR running_variances;

    bool batch_normalization = false;

    type momentum = type(0.9);

    string activation_function = "HyperbolicTangent";

    type dropout_rate = type(0);

#ifdef CUDA

    cudnnActivationDescriptor_t activation_descriptor = nullptr;
    cudnnDropoutDescriptor_t dropout_descriptor = nullptr;

#endif

    enum Parameters {Bias, Weight, Gamma, Beta};

    vector<Shape> get_parameter_shapes() const override
    {
        const Index input_dimension = input_shape.back();
        const Index output_dimension = get_outputs_number();

        return {{output_dimension},                            // Biases
                {input_dimension, output_dimension},           // Weights
                {batch_normalization ? output_dimension : 0},  // Gammas
                {batch_normalization ? output_dimension : 0}}; // Betas
    }

    enum Forward {Input, NormalizedOutput, Output};


    vector<Shape> get_forward_shapes(const Index batch_size) const override
    {
        const Index input_dimension = input_shape.back();
        const Index output_dimension = get_outputs_number();

        /*
        vector<Shape> shapes =
        {{batch_size, outputs_number},   // outputs
         {{batch_size, outputs_number}}; // activation_derivatives

        if (batch_normalization)
        {
            shapes.push_back({outputs_number});                // means
            shapes.push_back({outputs_number});                // standard_deviations
            shapes.push_back({batch_size, outputs_number});    // normalized_outputs
        }

        return shapes;
*/
        return {};
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
            return {parameters[Bias].size()};
        else
            return {input_shape[0], parameters[Bias].size()};
    }

    Index get_sequence_length() const
    {
        if constexpr (Rank == 3)
            return input_shape[0];
        else
            return 1;
    }

    type get_dropout_rate() const
    {
        return dropout_rate;
    }


    bool get_batch_normalization() const
    {
        return batch_normalization;
    }


    const string& get_activation_function() const
    {
        return activation_function;
    }


    void set(const Shape& new_input_shape = {},
             const Shape& new_output_shape = {},
             const string& new_activation_function = "HyperbolicTangent",
             bool new_batch_normalization = false,
             const string& new_label = "dense_layer")
    {
        if (new_input_shape.size() != Rank - 1)
            throw runtime_error("Input shape size must be " + to_string(Rank - 1));

        if (new_output_shape.size() != 1)
            throw runtime_error("Output shape size is not 1");

        input_shape = new_input_shape;

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

#ifdef CUDA

        cudnnDropoutDescriptor_t dropout_descriptor = nullptr;

        if (batch_normalization)
        {
            Shape batch_normalization_shape = { outputs_number };

            betas_device.set_descriptor(batch_normalization_shape);
            gammas_device.set_descriptor(batch_normalization_shape);
            running_means_device.resize(batch_normalization_shape);
            running_variances_device.resize(batch_normalization_shape);
        }

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
        if (new_input_shape.size() != Rank - 1)
            throw runtime_error("Input shape size must be " + to_string(Rank - 1));

        input_shape = new_input_shape;
    }


    void set_output_shape(const Shape& new_output_shape) override
    {
        neurons_number = new_output_shape.back();
    }


    void set_activation_function(const string& new_activation_function)
    {
        static const unordered_set<string> activation_functions =
            {"Sigmoid", "HyperbolicTangent", "Linear", "RectifiedLinear", "ScaledExponentialLinear", "Softmax","Logistic"};

        if (activation_functions.count(new_activation_function))
            if (get_output_shape()[0] == 1 && new_activation_function == "Softmax")
                activation_function = "Sigmoid";
            else
                activation_function = new_activation_function;
        else
            throw runtime_error("Unknown activation function: " + new_activation_function);

#ifdef CUDA

        if (activation_descriptor == nullptr && activation_function != "Softmax")
            cudnnCreateActivationDescriptor(&activation_descriptor);

        cudnnActivationMode_t activation_mode = CUDNN_ACTIVATION_IDENTITY;
        double relu_ceiling = 0.0;

        if (activation_function == "Linear")
        {
            activation_mode = CUDNN_ACTIVATION_IDENTITY;
            use_combinations = false;
        }
        else if (activation_function == "Sigmoid")
        {
            activation_mode = CUDNN_ACTIVATION_SIGMOID;
            use_combinations = false;
        }
        else if (activation_function == "HyperbolicTangent")
        {
            activation_mode = CUDNN_ACTIVATION_TANH;
            use_combinations = false;
        }
        else if (activation_function == "RectifiedLinear")
        {
            activation_mode = CUDNN_ACTIVATION_RELU;
            use_combinations = false;
        }
        else if (activation_function == "ScaledExponentialLinear")
        {
            activation_mode = CUDNN_ACTIVATION_ELU;
            use_combinations = true;
        }
        else if (activation_function == "ClippedRelu")
        {
            activation_mode = CUDNN_ACTIVATION_CLIPPED_RELU;
            use_combinations = true;
            relu_ceiling = 6.0;
        }
        else if (activation_function == "Swish")
        {
            activation_mode = CUDNN_ACTIVATION_SWISH;
            use_combinations = true;
        }
        else if (activation_function == "Softmax")
            use_combinations = true;

        if (activation_function != "Softmax")
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

        if (batch_normalization)
            is_training
                ? batch_normalization_training(output, gammas, betas, running_means, running_variances)
                : batch_normalization_inference(output, gammas, betas, running_means, running_variances);

        activation(output, activation_function);

        if(is_training && dropout_rate > type(0))
            dropout(output, dropout_rate);
    }


    void back_propagate(ForwardPropagation& forward_propagation,
                        BackPropagation& back_propagation,
                        size_t layer) const override
    {
        const TensorView& input = forward_propagation.views[layer][Input][0];
        const TensorView& output = forward_propagation.views[layer][Output][0];

        const TensorView& output_gradient = back_propagation.backward_views[layer][OutputGradients][0];

//        if (dropout_rate > type(0))
//            dropout_gradient(incoming_gradients, dropout_mask, dropout_rate, incoming_gradients);

        //output_gradient = activation_gradient(output, output_gradient)


        if (batch_normalization)
        {
            const TensorView& normalized_inputs = forward_propagation.views[layer][NormalizedOutput][0];
            const TensorView& standard_deviations = forward_propagation.views[layer][Output][1];

            TensorView& gamma_gradients = back_propagation.gradient_views[layer][Gamma];
            TensorView& beta_gradients = back_propagation.gradient_views[layer][Beta];
/*
            batch_normalization_backward(delta,
                                         normalized_inputs,
                                         standard_deviations,
                                         parameters[Gamma],
                                         gamma_gradients,
                                         beta_gradients,
                                         delta);
*/
        }

        TensorView& bias_gradient = back_propagation.gradient_views[layer][Bias];
        TensorView& weight_gradient = back_propagation.gradient_views[layer][Weight];

/*
        multiply(input, true, delta, false, weight_gradient);
        sum(delta, bias_gradient);

        if (!is_first_layer)
        {
            TensorView& input_gradient = back_propagation.backward_views[layer][InputGradients][0];
            multiply(delta, false, parameters[Weight], true, input_gradient);
        }
 */

    }

    void back_propagate(ForwardPropagation& forward_propagation,
                        BackPropagationLM& back_propagation,
                        size_t index) const override
    {
#ifndef CUDA
/*
        const Index inputs_number = get_input_features_number();
        const Index outputs_number = get_neurons_number();
        const Index batch_size = forward_propagation->inputs[0].size() / inputs_number;
        const Index biases_number = biases.size();

        const MatrixMap inputs(forward_propagation->inputs[0].data, batch_size, inputs_number);
        MatrixMap output_gradients(back_propagation->output_gradients[0].data, batch_size, outputs_number);

        MatrixMap jacobian = matrix_map(dense_bp_lm->squared_errors_Jacobian);
        jacobian.setZero();

        if(activation_function != "Softmax")
        {
            const MatrixMap activation_derivatives(dense_fp->activation_derivatives.data, batch_size, outputs_number);
            output_gradients.array() *= activation_derivatives.array();
        }

        const Index total_error_terms = jacobian.rows();
        const Index network_outputs = (batch_size > 0) ? total_error_terms / batch_size : 1;

        for(Index j = 0; j < outputs_number; j++)
            for(Index k = 0; k < network_outputs; ++k)
                for(Index s = 0; s < batch_size; ++s)
                    jacobian(s * network_outputs + k, j) = output_gradients(s, j);

        for(Index i = 0; i < inputs_number; i++)
        {
            for(Index j = 0; j < outputs_number; j++)
            {
                const Index weight_col = biases_number + i * outputs_number + j;

                const VectorR interaction = output_gradients.col(j).array() * inputs.col(i).array();

                for(Index k = 0; k < network_outputs; ++k)
                    for(Index s = 0; s < batch_size; ++s)
                        jacobian(s * network_outputs + k, weight_col) = interaction(s);
            }
        }

        if(!is_first_layer)
        {
            MatrixMap input_gradients(dense_bp_lm->input_gradients[0].data, batch_size, inputs_number);
            const MatrixMap weights = matrix_map(parameters[Weights]);

            input_gradients.noalias() = output_gradients * weights.transpose();
        }
*/
#else

#endif
    }


    void insert_squared_errors_Jacobian_lm(BackPropagationLM& back_propagation,
                                           Index start_column_index,
                                           MatrixR& global_jacobian) const override
    {
        const Index alignment_elements = EIGEN_MAX_ALIGN_BYTES / sizeof(type);
        const Index mask_elements = ~(alignment_elements - 1);
        const Index total_error_terms = global_jacobian.rows();

        Index global_offset = start_column_index;
        Index local_offset = 0;
/*
        MatrixMap layer_jacobian = matrix_map(dense_lm->squared_errors_Jacobian);

        const Index biases_size = biases.size();
        if(biases_size > 0)
        {
            global_jacobian.block(0, global_offset, total_error_terms, biases_size) =
                layer_jacobian.block(0, local_offset, total_error_terms, biases_size);

            local_offset += biases_size;
            global_offset += (biases_size + alignment_elements - 1) & mask_elements;
        }

        const Index weights_size = weights.size();

        if(weights_size > 0)
        {
            global_jacobian.block(0, global_offset, total_error_terms, weights_size) =
                layer_jacobian.block(0, local_offset, total_error_terms, weights_size);

            local_offset += weights_size;
            global_offset += (weights_size + alignment_elements - 1) & mask_elements;
        }

        if(!batch_normalization) return;

        const Index gammas_size = gammas.size();

        if(gammas_size > 0)
        {
            global_jacobian.block(0, global_offset, total_error_terms, gammas_size) =
                layer_jacobian.block(0, local_offset, total_error_terms, gammas_size);

            local_offset += gammas_size;
            global_offset += (gammas_size + alignment_elements - 1) & mask_elements;
        }

        const Index betas_size = betas.size();

        if(betas_size > 0)
        {
            global_jacobian.block(0, global_offset, total_error_terms, betas_size) =
                layer_jacobian.block(0, local_offset, total_error_terms, betas_size);

            local_offset += betas_size;
            global_offset += (betas_size + alignment_elements - 1) & mask_elements;
        }
*/
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

        return buffer.str();
    }


    void print() const override
    {
        cout << "Dense layer" << endl
             << "Input shape: " << get_input_shape() << endl
             << "Output shape: " << get_output_shape() << endl
             << "Biases shape: " << parameters[Bias].shape << endl
             << "Weights shape: " << parameters[Weight].shape << endl
             << "Activation function: " << activation_function << endl
             << "Batch normalization: " << (batch_normalization ? "True" : "False") << endl
             << "Dropout rate: " << dropout_rate << endl;
    }


    void from_XML(const XMLDocument& document) override
    {
        const XMLElement* dense2d_layer_element = document.FirstChildElement(name.c_str());

        if(!dense2d_layer_element)
            throw runtime_error(name + " element is nullptr.\n");

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
        const XMLElement* bn_element = dense2d_layer_element->FirstChildElement("BatchNormalization");

        if (bn_element && bn_element->GetText())
            use_batch_normalization = (string(bn_element->GetText()) == "true");
        set_batch_normalization(use_batch_normalization);

        if (batch_normalization)
        {
            running_means.resize(neurons_number);
            running_variances.resize(neurons_number);

            string_to_vector(read_xml_string(dense2d_layer_element, "RunningMeans"), running_means);
            string_to_vector(read_xml_string(dense2d_layer_element, "RunningStandardDeviations"), running_variances);
        }
    }


    void to_XML(XMLPrinter& printer) const override
    {
        printer.OpenElement(name.c_str());

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
        add_xml_element(printer, "Activation", activation_function);
        add_xml_element(printer, "BatchNormalization", batch_normalization ? "true" : "false");

        if (batch_normalization)
        {
            add_xml_element(printer, "RunningMeans", vector_to_string(running_means));
            add_xml_element(printer, "RunningStandardDeviations", vector_to_string(running_variances));
        }

        printer.CloseElement();
    }

    bool use_combinations = true;
};

void reference_dense_layer();

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// This library is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 2.1 of the License, or any later version.
// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.
// You should have received a copy of the GNU Lesser General Public
// License along with this library; if not, write to the Free Software
// Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
