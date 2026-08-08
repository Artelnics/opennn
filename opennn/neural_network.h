//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   N E U R A L   N E T W O R K   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#pragma once

#include "configuration.h"
#include "layer.h"
#include "tensor_types.h"
#include "variable.h"

#include <functional>

namespace opennn
{

class NeuralNetwork
{

public:

    NeuralNetwork();

    virtual ~NeuralNetwork() = default;

    NeuralNetwork(const filesystem::path&);

    void add_layer(unique_ptr<Layer>,
                  const vector<Index>& = {});

    const Configuration::Resolved& get_config() const noexcept { return config; }
    Device get_device() const noexcept { return config.device; }
    bool is_gpu() const noexcept { return config.device == Device::CUDA; }
    bool is_cpu() const noexcept { return config.device == Device::CPU; }

    Type get_training_type()  const noexcept { return config.training_type; }

    void set_training_activation_recomputation(bool enabled) noexcept
    {
        training_activation_recomputation = enabled;
    }
    bool get_training_activation_recomputation() const noexcept
    {
        return training_activation_recomputation;
    }

    void warn_if_stale_configuration() const;

    vector<vector<TensorSpec>> get_parameter_specs() const
    {
        return collect_layer_specs([](const Layer& layer) { return layer.get_parameter_specs(); });
    }

    vector<vector<TensorSpec>> get_state_specs() const
    {
        return collect_layer_specs([](const Layer& layer) { return layer.get_state_specs(); });
    }

    vector<vector<TensorSpec>> get_forward_specs(Index batch_size) const
    {
        auto specs = collect_layer_specs([batch_size](const Layer& layer) { return layer.get_forward_specs(batch_size); });
        if (!is_gpu()) force_specs_to_fp32(specs);
        return specs;
    }

    vector<vector<TensorSpec>> get_backward_specs(Index batch_size) const
    {
        auto specs = collect_layer_specs([batch_size](const Layer& layer) { return layer.get_backward_specs(batch_size); });
        if (!is_gpu()) force_specs_to_fp32(specs);
        return specs;
    }

    Index get_states_size() const     { return get_aligned_size(get_state_specs()); }

    void compile();
    void compile(Device device);
    bool has(const string&) const;
    bool has(LayerType) const;
    bool has_recurrent_layers() const;
    bool supports_compact_cnn_memory_layout() const noexcept;

    bool is_empty() const noexcept { return layers.empty(); }

    float* get_parameters_data() { return parameters.as<float>(); }
    const float* get_parameters_data() const noexcept { return parameters.as<float>(); }
    Index get_parameters_size() const noexcept { return parameters.size_in_floats(); }
    Device get_parameters_device() const noexcept { return parameters.device_type; }
    float* get_states_data() { return states.as<float>(); }
    const float* get_states_data() const noexcept { return states.as<float>(); }
    Index get_states_buffer_size() const noexcept { return states.size_in_floats(); }

    const vector<Variable>& get_input_variables() const noexcept { return input_variables; }
    vector<string> get_input_feature_names() const;

    const vector<Variable>& get_output_variables() const noexcept { return output_variables; }
    vector<string> get_output_feature_names() const;

    const vector<unique_ptr<Layer>>& get_layers() const noexcept { return layers; }
    const unique_ptr<Layer>& get_layer(const Index layer_index) const { return layers[layer_index]; }
    const unique_ptr<Layer>& get_layer(const string&) const;

    Index get_layer_index(const string&) const;

    const vector<vector<Index>>& get_source_layers() const noexcept { return source_layers; }

    Layer* get_first(const string&);
    Layer* get_first(LayerType);
    const Layer* get_first(const string&) const;
    const Layer* get_first(LayerType) const;

    void set_input_variables(const vector<Variable>& new_input_variables) { input_variables = new_input_variables; }
    void set_output_variables(const vector<Variable>& new_output_variables) { output_variables = new_output_variables; }

    void set_input_names(const vector<string>&);
    void set_output_names(const vector<string>&);

    void set_input_shape(const Shape&);

    void clear();
    void steal_from(NeuralNetwork& src);

    Index get_layers_number() const noexcept { return ssize(layers); }
    Index get_layers_number(const string&) const;
    Index get_layers_number(LayerType) const;

    Index get_first_trainable_layer_index() const;
    Index get_last_trainable_layer_index() const;

    void invalidate_trainable_layer_cache() { first_trainable_cache_ = -1; last_trainable_cache_ = -1; }

    Index get_inputs_number() const;
    Index get_outputs_number() const;

    Shape get_input_shape() const;
    Shape get_output_shape() const;

    ActivationFunction get_output_activation() const;
    Index get_parameters_number() const;

    void set_parameters(const VectorR&);
    void set_states(const VectorR&);
    void set_parameters_random();
    void set_parameters_glorot();
    void set_parameters_pytorch();
    void link_parameters();
    void link_states();
    void link_states(Device);
    void wire_drelu_fusions();
    MatrixR calculate_outputs(const vector<TensorView>&);

    TensorView calculate_outputs_resident(const vector<TensorView>&,
                                          ForwardPropagation&,
                                          bool upload_parameters = true);

    MatrixR calculate_outputs(const MatrixR&);

    MatrixR calculate_outputs(const Tensor3&);

    MatrixR calculate_outputs(const Tensor4&);

    Tensor3 calculate_outputs(const Tensor3&, const Tensor3&);

    MatrixR calculate_text_outputs(const Tensor<string, 1>&);
    void from_JSON(const JsonDocument&);

    void to_JSON(JsonWriter&) const;

    void save(const filesystem::path&) const;
    void save_parameters_binary(const filesystem::path&) const;
    void save_states_binary(const filesystem::path&) const;

    void load(const filesystem::path&);
    void load_parameters_binary(const filesystem::path&);

    void load_parameters_bf16_inference_binary(const filesystem::path&);
    void load_states_binary(const filesystem::path&);

    void forward_propagate(const vector<TensorView>&,
                          ForwardPropagation&,
                          bool = false) const;

    void forward_propagate(const vector<TensorView>&,
                          ForwardPropagation&,
                          bool,
                          Index,
                          Index) const;

    void forward_propagate(const vector<TensorView>&,
                          const VectorR&,
                          ForwardPropagation&);

public:

    void cast_parameters_to_bf16();

    void release_bf16_fp32_parameter_master_for_inference();

    void upload_parameters_bf16_inference();

    void upload_parameters_int8_inference();

    bfloat16* get_parameters_bf16_mirror_data()
    {
        return config.training_type == Type::BF16 && parameters.owns
            ? parameters_bf16_mirror.as<bfloat16>()
            : nullptr;
    }

    void copy_parameters_device();
    void copy_parameters_host();

    void copy_states_device();
    void copy_states_host();

private:

    void compile(Configuration::Resolved);

    MatrixR calculate_outputs_device(const vector<TensorView>&, ForwardPropagation&);

public:

    vector<string> get_layer_labels() const;

private:

    struct HostParametersGuard
    {
        explicit HostParametersGuard(NeuralNetwork& n)
            : network(n), was_on_device(n.parameters.device_type == Device::CUDA)
        {
            if (was_on_device) network.copy_parameters_host();
        }

        ~HostParametersGuard() { if (was_on_device) network.copy_parameters_device(); }

        HostParametersGuard(const HostParametersGuard&) = delete;
        HostParametersGuard& operator=(const HostParametersGuard&) = delete;

        NeuralNetwork& network;
        const bool was_on_device;
    };

    struct HostStatesGuard
    {
        explicit HostStatesGuard(NeuralNetwork& n)
            : HostStatesGuard(n, n.states.device_type == Device::CUDA) {}

        HostStatesGuard(NeuralNetwork& n, bool stage)
            : network(n), was_on_device(stage)
        {
            if (was_on_device) network.copy_states_host();
        }

        ~HostStatesGuard() { if (was_on_device) network.copy_states_device(); }

        HostStatesGuard(const HostStatesGuard&) = delete;
        HostStatesGuard& operator=(const HostStatesGuard&) = delete;

        NeuralNetwork& network;
        const bool was_on_device;
    };

    void initialize_parameters(void (Operator::*)());

    void clear_low_precision_parameter_storage();
    void activate_transposed_inference_weights();

    struct ParameterSlot
    {
        Layer* layer = nullptr;
        Shape shape;
        Type dtype = Type::FP32;
        bool tied = false;
        Index scale_channels = 0;
        int   scale_axis = 0;
        Index master_offset = 0;
        Index bf16_offset = 0;
        Index int8_offset = 0;
        Index fp32_offset = 0;
    };

    struct ParameterSlotTotals
    {
        Index bf16_elements = 0;
        Index int8_elements = 0;
        Index fp32_elements = 0;
    };

    ParameterSlotTotals for_each_parameter_slot(
        const function<void(const ParameterSlot&)>& visit,
        const function<void(Layer&)>& begin_layer = {}) const;

    void allocate_compact_parameter_storage(const ParameterSlotTotals&);
    void use_compact_parameter_storage();

    void validate_type(LayerType) const;

    static void force_specs_to_fp32(vector<vector<TensorSpec>>& specs)
    {
        for (auto& layer_specs : specs)
            for (auto& spec : layer_specs)
                spec.dtype = Type::FP32;
    }

    template<typename Fn>
    vector<vector<TensorSpec>> collect_layer_specs(Fn fn) const
    {
        vector<vector<TensorSpec>> out(layers.size());
        ranges::transform(layers, out.begin(),
                          [&](const unique_ptr<Layer>& layer) { return fn(*layer); });
        return out;
    }

protected:

    vector<Variable> input_variables;
    vector<Variable> output_variables;

    vector<unique_ptr<Layer>> layers;

    vector<vector<Index>> source_layers;

    Buffer parameters;
    Buffer parameters_bf16_mirror{Device::CUDA};
    Buffer parameters_fp32_inference_storage{Device::CUDA};
    Buffer parameters_int8_storage{Device::CUDA};

    bool parameters_bf16_mirror_compact = false;

    Buffer states;

    Configuration::Resolved config;

    bool training_activation_recomputation = false;

    mutable bool stale_configuration_warned = false;

    mutable Index first_trainable_cache_ = -1;
    mutable Index last_trainable_cache_  = -1;
};

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
