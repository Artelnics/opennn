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

template<int Rank>
class Dense final : public Layer
{
private:

    Index input_features = 0;
    Index sequence_length = 0;
    Index output_features;

    bool batch_normalization = false;

    type momentum = type(0.1);

    type dropout_rate = type(0);

    DropoutArguments dropout_arguments;

    ActivationArguments activation_arguments;

    enum Parameters {Bias, Weight, Gamma, Beta};

    vector<Shape> get_parameter_shapes() const override
    {
        const Index input_dimension = input_features;

        return {{output_features},                            // Bias
                {input_dimension, output_features},           // Weight
                {batch_normalization ? output_features : 0},  // Gamma
                {batch_normalization ? output_features : 0}}; // Beta
    }

    // Bias and Weight follow the activation dtype (BF16 working copy when flag
    // is on); Gamma and Beta stay FP32 because cuDNN's batch normalization
    // expects FP32 scale/shift even when the activation tensors are BF16.
    vector<cudnnDataType_t> get_parameter_dtypes() const override
    {
        return {activation_dtype,   // Bias
                activation_dtype,   // Weight
                CUDNN_DATA_FLOAT,         // Gamma
                CUDNN_DATA_FLOAT};        // Beta
    }

    enum States {RunningMean, RunningVariance};

    vector<Shape> get_state_shapes() const override
    {
        if (!batch_normalization) return {};
        return {{output_features},   // RunningMean
                {output_features}};  // RunningVariance
    }

    enum Forward {Input, Combination, BatchNormMean, BatchNormInverseVariance, Activation, Output};

    vector<Shape> get_forward_shapes(const Index batch_size) const override
    {
        const Shape output_shape = Shape{batch_size}.append(get_output_shape());

        // Activation view stores the post-activation, pre-dropout value so that
        // back-propagation can compute the activation derivative after dropout
        // has overwritten the output tensor. Allocated only when dropout is in use.
        const Shape activation_shape = (dropout_rate > type(0)) ? output_shape : Shape{};

        if(batch_normalization)
            return {output_shape,             // Combination
                    Shape{output_features},   // BatchNormMean
                    Shape{output_features},   // BatchNormInverseVariance
                    activation_shape,         // Activation
                    output_shape};            // Output
        else
            return {Shape{},                  // Combination (unused)
                    Shape{},                  // BatchNormMean (unused)
                    Shape{},                  // BatchNormInverseVariance (unused)
                    activation_shape,         // Activation
                    output_shape};            // Output
    }

    vector<cudnnDataType_t> get_forward_dtypes(Index) const override
    {
        return {activation_dtype,  // Combination
                CUDNN_DATA_FLOAT,        // BatchNormMean
                CUDNN_DATA_FLOAT,        // BatchNormInverseVariance
                activation_dtype,  // Activation
                activation_dtype}; // Output
    }

    enum Backward {OutputDelta, InputDelta};

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
        destroy_cuda();
#endif
    }

    Shape get_input_shape() const override
    {
        if constexpr (Rank == 2)
            return {input_features};
        else
            return {sequence_length, input_features};
    }

    Shape get_output_shape() const override
    {
        if constexpr (Rank == 2)
            return {output_features};
        else
            return {sequence_length, output_features};
    }

    Index get_sequence_length() const { return sequence_length; }

    const ActivationFunction& get_activation_function() const
    {
        return activation_arguments.activation_function;
    }

    ActivationFunction get_output_activation() const override
    {
        return activation_arguments.activation_function;
    }

    bool get_batch_normalization() const { return batch_normalization; }

    type get_momentum() const { return momentum; }

    // Setters

    void set(const Shape& new_input_shape = {},
             const Shape& new_output_shape = {},
             const string& new_activation_function = "HyperbolicTangent",
             bool new_batch_normalization = false,
             const string& new_label = "dense_layer")
    {
        if (new_input_shape.size() == 0 && new_output_shape.size() == 0)
        {
            // Default construction for registry — will be configured via from_XML
            name = (Rank == 2) ? "Dense2d" : "Dense3d";
            layer_type = (Rank == 2) ? LayerType::Dense2d : LayerType::Dense3d;
            is_trainable = true;
            return;
        }

        if (new_input_shape.rank != Rank - 1)
            throw runtime_error("Input shape size must be " + to_string(Rank - 1));

        if (new_output_shape.rank != 1)
            throw runtime_error("Output shape size is not 1");

        if constexpr (Rank == 2)
            input_features = new_input_shape[0];
        else
        {
            sequence_length = new_input_shape[0];
            input_features = new_input_shape[1];
        }

        output_features = new_output_shape.back();

        set_activation_function(new_activation_function);

        set_batch_normalization(new_batch_normalization);

        set_label(new_label);

        name = "Dense" + to_string(Rank) + "d";
        layer_type = (Rank == 2) ? LayerType::Dense2d : LayerType::Dense3d;
    }

    void set_input_shape(const Shape& new_input_shape) override
    {
        if (new_input_shape.rank != Rank - 1)
            throw runtime_error("Input shape size must be " + to_string(Rank - 1));

        if constexpr (Rank == 2)
            input_features = new_input_shape[0];
        else
        {
            sequence_length = new_input_shape[0];
            input_features = new_input_shape[1];
        }
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

    void set_dropout_rate(const type new_dropout_rate)
    {
        if (new_dropout_rate < type(0) || new_dropout_rate >= type(1))
            throw runtime_error("Dropout rate must be in [0,1).");

        dropout_rate = new_dropout_rate;
        dropout_arguments.rate = new_dropout_rate;
    }

    void set_momentum(const type new_momentum)
    {
        if (new_momentum < type(0) || new_momentum >= type(1))
            throw runtime_error("Batch normalization momentum must be in [0,1).");

        momentum = new_momentum;
    }

    // Parameter initialization

    void set_parameters_glorot() override
    {
        const type limit = sqrt(6.0 / (get_inputs_number() + get_outputs_number()));
        set_random_uniform(VectorMap(parameters[Weight].template as<float>(), parameters[Weight].size()), -limit, limit);
        init_dense_norm_defaults();
    }

    void set_parameters_random() override
    {
        set_random_uniform(VectorMap(parameters[Weight].template as<float>(), parameters[Weight].size()));
        init_dense_norm_defaults();
    }

private:
    // Bias=0, Gamma=1, Beta=0; running stats reset (when batch norm active).
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

        if (dropout_rate > type(0))
        {
            cudnnTensorDescriptor_t temp_desc;
            cudnnCreateTensorDescriptor(&temp_desc);

            const Index output_size = get_outputs_number();

            cudnnSetTensor4dDescriptor(temp_desc, CUDNN_TENSOR_NHWC, activation_dtype,
                                       static_cast<int>(batch_size),
                                       static_cast<int>(output_size),
                                       static_cast<int>(Rank == 3 ? sequence_length : 1),
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
            ::opennn::profiler::ScopedTimer _t("dense3d_fwd:combination_activation");
            combination_activation(forward_views[Input][0],
                                   parameters[Weight], parameters[Bias],
                                   activation_arguments,
                                   forward_views[Output][0]);
        }

        if (is_training && dropout_rate > type(0))
        {
            ::opennn::profiler::ScopedTimer _t("dense3d_fwd:dropout_save_and_apply");
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

        if (dropout_rate > type(0))
        {
            ::opennn::profiler::ScopedTimer _t1("dense3d_bwd:01_dropout_delta");
            dropout_delta(output_delta, output_delta, dropout_arguments);
            ::opennn::profiler::ScopedTimer _t2("dense3d_bwd:02_activation_delta_dropout");
            activation_delta(forward_views[Activation][0], output_delta, output_delta, activation_arguments);
        }
        else
        {
            ::opennn::profiler::ScopedTimer _t("dense3d_bwd:02_activation_delta");
            activation_delta(output, output_delta, output_delta, activation_arguments);
        }

        if (batch_normalization)
        {
            ::opennn::profiler::ScopedTimer _t("dense3d_bwd:03_batchnorm_backward");
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
        TensorView input_2d           = input.reshape({total_rows, input.shape.back()});

        TensorView input_delta_2d;
        if (!is_first_layer)
        {
            TensorView& input_delta = delta_views[InputDelta][0];
            input_delta_2d = input_delta.reshape({total_rows, input_delta.shape.back()});
        }

        {
            ::opennn::profiler::ScopedTimer _t("dense3d_bwd:04_combination_gradient");
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
        const XmlElement* dense_layer_element = get_xml_root(document, name);

        set_label(read_xml_string(dense_layer_element, "Label"));

        const Index inputs_number = read_xml_index(dense_layer_element, "InputsNumber");
        const Index output_features = read_xml_index(dense_layer_element, "NeuronsNumber");

        if constexpr (Rank == 3)
        {
            const Index input_sequence_length = read_xml_index(dense_layer_element, "InputSequenceLength");
            set_input_shape({ input_sequence_length, inputs_number });
        }
        else
            set_input_shape({ inputs_number });

        set_output_shape({ output_features });

        set_activation_function(read_xml_string(dense_layer_element, "Activation"));

        const XmlElement* bn_element = dense_layer_element->first_child_element("BatchNormalization");
        const bool use_batch_normalization = bn_element && bn_element->get_text()
                                             && string(bn_element->get_text()) == "true";
        set_batch_normalization(use_batch_normalization);
    }

    // Phase 2: runs after NN::compile(), so states[] is allocated. Parses BN running
    // statistics directly into the arena — no staging required.
    void load_state_from_XML(const XmlDocument& document) override
    {
        if(!batch_normalization) return;

        const XmlElement* dense_layer_element = get_xml_root(document, name);

        VectorR tmp;
        string_to_vector(read_xml_string(dense_layer_element, "RunningMeans"), tmp);
        if(tmp.size() == states[RunningMean].size() && states[RunningMean].data)
            VectorMap(states[RunningMean].template as<float>(), states[RunningMean].size()) = tmp;

        string_to_vector(read_xml_string(dense_layer_element, "RunningVariances"), tmp);
        if(tmp.size() == states[RunningVariance].size() && states[RunningVariance].data)
            VectorMap(states[RunningVariance].template as<float>(), states[RunningVariance].size()) = tmp;
    }

    void to_XML(XmlPrinter& printer) const override
    {
        printer.open_element(name.c_str());

        if constexpr (Rank == 3)
            write_xml(printer, {
                {"Label", label},
                {"InputSequenceLength", to_string(get_input_shape()[0])},
                {"InputsNumber", to_string(get_input_shape()[1])},
                {"NeuronsNumber", to_string(get_output_shape()[1])},
                {"Activation", activation_to_string(activation_arguments.activation_function)},
                {"BatchNormalization", batch_normalization ? "true" : "false"}
            });
        else
            write_xml(printer, {
                {"Label", label},
                {"InputsNumber", to_string(get_input_shape()[0])},
                {"NeuronsNumber", to_string(get_output_shape()[0])},
                {"Activation", activation_to_string(activation_arguments.activation_function)},
                {"BatchNormalization", batch_normalization ? "true" : "false"}
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
