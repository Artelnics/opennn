//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   D E N S E   L A Y E R   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#pragma once

#include "layer.h"
#include "profiler.h"

namespace opennn
{

class Dense final : public Layer
{
private:

    Shape input_shape;     // without-batch input shape; back() is feature count
    Index output_features;

    bool batch_normalization = false;

    float momentum = float(0.1);

    float dropout_rate = float(0);

    DropoutArguments dropout_arguments;

    ActivationArguments activation_arguments;

    enum Parameters {Bias, Weight, Gamma, Beta};

    vector<pair<Shape, Type>> get_parameter_specs() const override
    {
        const Type act = activation_dtype;
        return {
            /*Bias*/   {{output_features},                           act},
            /*Weight*/ {{get_input_features(), output_features},     act},
            /*Gamma*/  {{batch_normalization ? output_features : 0}, Type::FP32},
            /*Beta*/   {{batch_normalization ? output_features : 0}, Type::FP32},
        };
    }

    Index get_input_features() const
    {
        return input_shape.empty() ? 0 : input_shape.back();
    }

    enum States {RunningMean, RunningVariance};

    vector<pair<Shape, Type>> get_state_specs() const override
    {
        if (!batch_normalization) return {};
        return {
            /*RunningMean*/     {{output_features}, Type::FP32},
            /*RunningVariance*/ {{output_features}, Type::FP32},
        };
    }

    enum Forward {Input, Combination, BatchNormMean, BatchNormInverseVariance, Activation, Output};

    vector<pair<Shape, Type>> get_forward_specs(const Index batch_size) const override
    {
        const Shape output_shape = Shape{batch_size}.append(get_output_shape());
        const Type act = activation_dtype;

        const Shape activation_shape = (dropout_rate > float(0)) ? output_shape : Shape{};

        const Shape combination_shape = batch_normalization ? output_shape           : Shape{};
        const Shape bn_stat_shape     = batch_normalization ? Shape{output_features} : Shape{};

        return {
            /*Combination*/              {combination_shape, act},
            /*BatchNormMean*/            {bn_stat_shape,     Type::FP32},
            /*BatchNormInverseVariance*/ {bn_stat_shape,     Type::FP32},
            /*Activation*/               {activation_shape,  act},
            /*Output*/                   {output_shape,      act},
        };
    }

    enum Backward {OutputDelta, InputDelta};

    vector<pair<Shape, Type>> get_backward_specs(Index batch_size) const override
    {
        return {{Shape{batch_size}.append(get_input_shape()), activation_dtype}};
    }

    void init_dense_norm_defaults()
    {
        parameters[Bias].fill(0.0f);
        parameters[Gamma].fill(1.0f);
        parameters[Beta].fill(0.0f);
        if (batch_normalization && ssize(states) > RunningVariance)
        {
            states[RunningMean].fill(0.0f);
            states[RunningVariance].fill(1.0f);
        }
    }

public:

    Dense(const Shape& new_input_shape = {},
          const Shape& new_output_shape = {},
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
        destroy_cuda();
#endif
    }

    Shape get_input_shape() const override { return input_shape; }

    Shape get_output_shape() const override
    {
        Shape result = input_shape;
        if (result.empty()) return {output_features};
        result.back() = output_features;
        return result;
    }

    Index get_sequence_length() const { return (input_shape.rank == 2) ? input_shape[0] : Index(1); }

    const ActivationFunction& get_activation_function() const
    {
        return activation_arguments.activation_function;
    }

    ActivationFunction get_output_activation() const override
    {
        return activation_arguments.activation_function;
    }

    bool get_batch_normalization() const { return batch_normalization; }

    float get_momentum() const { return momentum; }

    // Setters

    void set(const Shape& new_input_shape = {},
             const Shape& new_output_shape = {},
             const string& new_activation_function = "HyperbolicTangent",
             bool new_batch_normalization = false,
             const string& new_label = "dense_layer")
    {
        is_trainable = true;
        layer_type = LayerType::Dense;
        name = "Dense";

        if (new_input_shape.empty() && new_output_shape.empty())
        {
            input_shape = {};
            output_features = 0;
            return;
        }

        if (new_input_shape.rank != 1 && new_input_shape.rank != 2)
            throw runtime_error("Dense input shape rank must be 1 or 2 (got "
                                + to_string(new_input_shape.rank) + ").");

        if (new_output_shape.rank != 1)
            throw runtime_error("Dense output shape rank must be 1.");

        input_shape = new_input_shape;
        output_features = new_output_shape.back();

        set_activation_function(new_activation_function);
        set_batch_normalization(new_batch_normalization);
        set_label(new_label);
    }

    void set_input_shape(const Shape& new_input_shape) override
    {
        if (new_input_shape.rank != 1 && new_input_shape.rank != 2)
            throw runtime_error("Dense input shape rank must be 1 or 2.");

        input_shape = new_input_shape;
    }

    void set_output_shape(const Shape& new_output_shape) override
    {
        output_features = new_output_shape.back();
    }

    void set_activation_function(const string& name)
    {
        ActivationFunction function = string_to_activation(name);

        if (function == ActivationFunction::Softmax && get_outputs_number() == 1)
            function = ActivationFunction::Sigmoid;

        activation_arguments.activation_function = function;

#ifdef OPENNN_WITH_CUDA
        if (activation_arguments.activation_descriptor && function != ActivationFunction::Softmax)
            cudnnSetActivationDescriptor(activation_arguments.activation_descriptor,
                                         to_cudnn_activation_mode(function),
                                         CUDNN_PROPAGATE_NAN, 0.0);
#endif
    }

    void set_batch_normalization(bool new_batch_normalization)
    {
        batch_normalization = new_batch_normalization;
    }

    void set_dropout_rate(const float new_dropout_rate)
    {
        if (new_dropout_rate < float(0) || new_dropout_rate >= float(1))
            throw runtime_error("Dropout rate must be in [0,1).");

        dropout_rate = new_dropout_rate;
        dropout_arguments.rate = new_dropout_rate;
    }

    void set_momentum(const float new_momentum)
    {
        if (new_momentum < float(0) || new_momentum >= float(1))
            throw runtime_error("Batch normalization momentum must be in [0,1).");

        momentum = new_momentum;
    }

    // Parameter initialization

    void set_parameters_glorot() override
    {
        const float limit = sqrt(6.0 / (get_inputs_number() + get_outputs_number()));
        set_random_uniform(VectorMap(parameters[Weight].as<float>(), parameters[Weight].size()), -limit, limit);
        init_dense_norm_defaults();
    }

    void set_parameters_random() override
    {
        set_random_uniform(VectorMap(parameters[Weight].as<float>(), parameters[Weight].size()));
        init_dense_norm_defaults();
    }

#ifdef OPENNN_WITH_CUDA

    void init_cuda(Index batch_size)
    {
        // Activation descriptor

        const ActivationFunction function = activation_arguments.activation_function;
        if (function != ActivationFunction::Softmax)
        {
            cudnnActivationDescriptor_t& descriptor = activation_arguments.activation_descriptor;
            if (!descriptor)
                cudnnCreateActivationDescriptor(&descriptor);
            cudnnSetActivationDescriptor(descriptor, to_cudnn_activation_mode(function), CUDNN_PROPAGATE_NAN, 0.0);
        }

        // Dropout

        if (dropout_rate > float(0))
        {
            cudnnTensorDescriptor_t temp_desc;
            cudnnCreateTensorDescriptor(&temp_desc);

            const Index output_size = get_outputs_number();

            // cuDNN 9 rejects BFLOAT16 in cudnnDropoutForward; pin to FP32.
            cudnnSetTensor4dDescriptor(temp_desc, CUDNN_TENSOR_NHWC, to_cudnn(Type::FP32),
                                       static_cast<int>(batch_size),
                                       static_cast<int>(output_size),
                                       static_cast<int>(get_sequence_length()),
                                       1);

            if (dropout_arguments.descriptor) { cudnnDestroyDropoutDescriptor(dropout_arguments.descriptor); dropout_arguments.descriptor = nullptr; }
            if (dropout_arguments.states) { cudaFree(dropout_arguments.states); dropout_arguments.states = nullptr; }
            if (dropout_arguments.reserve_space) { cudaFree(dropout_arguments.reserve_space); dropout_arguments.reserve_space = nullptr; }

            CHECK_CUDNN(cudnnCreateDropoutDescriptor(&dropout_arguments.descriptor));
            CHECK_CUDNN(cudnnDropoutGetStatesSize(Device::get_cudnn_handle(), &dropout_arguments.states_size));
            CHECK_CUDA(cudaMalloc(&dropout_arguments.states, dropout_arguments.states_size));
            CHECK_CUDNN(cudnnSetDropoutDescriptor(dropout_arguments.descriptor, Device::get_cudnn_handle(),
                                                  dropout_rate, dropout_arguments.states, dropout_arguments.states_size,
                                                  static_cast<unsigned long long>(random_integer(0, 1 << 30))));
            CHECK_CUDNN(cudnnDropoutGetReserveSpaceSize(temp_desc, &dropout_arguments.reserve_size));
            CHECK_CUDA(cudaMalloc(&dropout_arguments.reserve_space, dropout_arguments.reserve_size));

            dropout_arguments.rate = dropout_rate;

            cudnnDestroyTensorDescriptor(temp_desc);
        }
    }

    void destroy_cuda()
    {
        if (activation_arguments.activation_descriptor)
            cudnnDestroyActivationDescriptor(activation_arguments.activation_descriptor);
        if (dropout_arguments.descriptor) cudnnDestroyDropoutDescriptor(dropout_arguments.descriptor);
        if (dropout_arguments.states) cudaFree(dropout_arguments.states);
        if (dropout_arguments.reserve_space) cudaFree(dropout_arguments.reserve_space);
    }
#endif

    // Forward / back propagation

    void forward_propagate(ForwardPropagation& forward_propagation, size_t layer, bool is_training) noexcept override
    {
        auto& forward_views = forward_propagation.views[layer];

        if (batch_normalization)
        {
            combination(forward_views[Input][0],
                        parameters[Weight],
                        parameters[Bias],
                        forward_views[Combination][0]);

            if(is_training)
                batch_normalization_training(forward_views[Combination][0],
                                             parameters[Gamma],
                                             parameters[Beta],
                                             states[RunningMean],
                                             states[RunningVariance],
                                             forward_views[BatchNormMean][0],
                                             forward_views[BatchNormInverseVariance][0],
                                             forward_views[Output][0],
                                             momentum);
            else
                batch_normalization_inference(forward_views[Combination][0],
                                              parameters[Gamma],
                                              parameters[Beta],
                                              states[RunningMean],
                                              states[RunningVariance],
                                              forward_views[Output][0]);

            activation(forward_views[Output][0], activation_arguments);
        }
        else
        {
            ::opennn::profiler::ScopedTimer timer("dense_fwd:combination_activation");
            combination_activation(forward_views[Input][0],
                                   parameters[Weight], parameters[Bias],
                                   activation_arguments,
                                   forward_views[Output][0]);
        }

        if (is_training && dropout_rate > float(0))
        {
            ::opennn::profiler::ScopedTimer timer("dense_fwd:dropout_save_and_apply");
            copy(forward_views[Output][0], forward_views[Activation][0]);
            dropout(forward_views[Output][0], dropout_arguments);
        }
    }

    void back_propagate(ForwardPropagation& forward_propagation,
                        BackPropagation& back_propagation,
                        size_t layer) const noexcept override
    {
        auto& forward_views = forward_propagation.views[layer];
        auto& delta_views = back_propagation.delta_views[layer];
        auto& gradient_views = back_propagation.gradient_views[layer];

        const TensorView& input = forward_views[Input][0];
        const TensorView& output = forward_views[Output][0];

        TensorView& output_delta = delta_views[OutputDelta][0];

        if (dropout_rate > float(0))
        {
            ::opennn::profiler::ScopedTimer dropout_delta_timer("dense_bwd:01_dropout_delta");
            dropout_delta(output_delta, output_delta, dropout_arguments);
            ::opennn::profiler::ScopedTimer activation_delta_timer("dense_bwd:02_activation_delta_dropout");
            activation_delta(forward_views[Activation][0], output_delta, output_delta, activation_arguments);
        }
        else
        {
            ::opennn::profiler::ScopedTimer timer("dense_bwd:02_activation_delta");
            activation_delta(output, output_delta, output_delta, activation_arguments);
        }

        if (batch_normalization)
        {
            ::opennn::profiler::ScopedTimer timer("dense_bwd:03_batchnorm_backward");
            batch_normalization_backward(forward_views[Combination][0],
                                         output,
                                         output_delta,
                                         forward_views[BatchNormMean][0],
                                         forward_views[BatchNormInverseVariance][0],
                                         parameters[Gamma],
                                         gradient_views[Gamma],
                                         gradient_views[Beta],
                                         output_delta);
        }

        const Index total_rows = input.size() / input.shape.back();

        TensorView output_delta_2d = output_delta.reshape({total_rows, output_delta.shape.back()});
        TensorView input_2d        = input.reshape({total_rows, input.shape.back()});

        TensorView input_delta_2d;
        if (!is_first_layer)
        {
            TensorView& input_delta = delta_views[InputDelta][0];
            input_delta_2d = input_delta.reshape({total_rows, input_delta.shape.back()});
        }

        {
            ::opennn::profiler::ScopedTimer timer("dense_bwd:04_combination_gradient");
            combination_gradient(output_delta_2d,
                                 input_2d,
                                 parameters[Weight],
                                 input_delta_2d,
                                 gradient_views[Weight],
                                 gradient_views[Bias],
                                 false);
        }
    }

    // Serialization

    void from_XML(const XmlDocument& document) override
    {
        const XmlElement* dense_layer_element = document.first_child_element("Dense");
        if (!dense_layer_element) throw runtime_error(name + " element is nullptr.");

        set_label(read_xml_string(dense_layer_element, "Label"));

        set_input_shape(string_to_shape(read_xml_string(dense_layer_element, "InputDimensions")));
        set_output_shape({ read_xml_index(dense_layer_element, "NeuronsNumber") });

        set_activation_function(read_xml_string(dense_layer_element, "Activation"));
        set_batch_normalization(read_xml_bool(dense_layer_element, "BatchNormalization"));
    }

    void load_state_from_XML(const XmlDocument& document) override
    {
        if(!batch_normalization) return;

        const XmlElement* dense_layer_element = document.first_child_element("Dense");
        if (!dense_layer_element) throw runtime_error(name + " element is nullptr.");

        VectorR tmp;
        string_to_vector(read_xml_string(dense_layer_element, "RunningMeans"), tmp);
        if(tmp.size() == states[RunningMean].size() && states[RunningMean].data)
            VectorMap(states[RunningMean].as<float>(), states[RunningMean].size()) = tmp;

        string_to_vector(read_xml_string(dense_layer_element, "RunningVariances"), tmp);
        if(tmp.size() == states[RunningVariance].size() && states[RunningVariance].data)
            VectorMap(states[RunningVariance].as<float>(), states[RunningVariance].size()) = tmp;
    }

    void to_XML(XmlPrinter& printer) const override
    {
        printer.open_element("Dense");

        write_xml(printer, {
            {"Label", label},
            {"InputDimensions", shape_to_string(input_shape)},
            {"NeuronsNumber", to_string(output_features)},
            {"Activation", activation_to_string(activation_arguments.activation_function)},
            {"BatchNormalization", to_string(batch_normalization)}
        });

        if (batch_normalization)
            write_xml(printer, {
                {"RunningMeans", vector_to_string(states[RunningMean].as_vector())},
                {"RunningVariances", vector_to_string(states[RunningVariance].as_vector())}
            });

        printer.close_element();
    }
};

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
