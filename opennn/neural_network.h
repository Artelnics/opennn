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

    const Configuration::Resolved& get_config() const { return config; }
    Device get_device() const { return config.device; }
    bool is_gpu() const { return config.device == Device::CUDA; }
    bool is_cpu() const { return config.device == Device::CPU; }

    Type get_training_type()  const { return config.training_type; }

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
    bool has(const string&) const;
    bool has(LayerType) const;

    bool is_empty() const { return layers.empty(); }

    float* get_parameters_data() { return parameters.as<float>(); }
    const float* get_parameters_data() const { return parameters.as<float>(); }
    Index get_parameters_size() const { return parameters.size_in_floats(); }
    Device get_parameters_device() const { return parameters.device_type; }
    float* get_states_data() { return states.as<float>(); }
    const float* get_states_data() const { return states.as<float>(); }
    Index get_states_buffer_size() const { return states.size_in_floats(); }

    const vector<Variable>& get_input_variables() const { return input_variables; }
    vector<string> get_input_feature_names() const;

    const vector<Variable>& get_output_variables() const { return output_variables; }
    vector<string> get_output_feature_names() const;

    const vector<unique_ptr<Layer>>& get_layers() const { return layers; }
    const unique_ptr<Layer>& get_layer(const Index layer_index) const { return layers[layer_index]; }
    const unique_ptr<Layer>& get_layer(const string&) const;

    Index get_layer_index(const string&) const;

    const vector<vector<Index>>& get_source_layers() const { return source_layers; }
    vector<vector<Index>> get_consumer_layers() const;

    Layer* get_first(const string&);
    Layer* get_first(LayerType);
    const Layer* get_first(const string&) const;
    const Layer* get_first(LayerType) const;

    void set_source_layers(const vector<vector<Index>>&);
    void set_source_layers(const Index, const vector<Index>&);

    void set_source_layers(const string&, const vector<string>&);
    void set_source_layers(const string&, initializer_list<string>);
    void set_source_layers(const string&, const string&);

    void set_input_variables(const vector<Variable>& new_input_variables) { input_variables = new_input_variables; }
    void set_output_variables(const vector<Variable>& new_output_variables) { output_variables = new_output_variables; }

    void set_input_names(const vector<string>&);
    void set_output_names(const vector<string>&);

    void set_input_shape(const Shape&);

    void clear();

    Index get_layers_number() const { return ssize(layers); }
    Index get_layers_number(const string&) const;
    Index get_layers_number(LayerType) const;

    Index get_first_trainable_layer_index() const;
    Index get_last_trainable_layer_index() const;
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
    void link_parameters();
    void link_states();
    void link_states(Device);
    MatrixR calculate_outputs(const vector<TensorView>&);

    // Device-resident inference: input already on GPU, caller-owned persistent
    // ForwardPropagation (activations allocated once), parameters uploaded only
    // when upload_parameters=true (skip on repeat calls with unchanged weights).
    // Output is left on the GPU; returns its TensorView (no D2H). This is
    // the zero-per-call-overhead path -- the PyTorch-equivalent inference loop.
    TensorView calculate_outputs_resident(const vector<TensorView>&,
                                          ForwardPropagation&,
                                          bool upload_parameters = true);

    MatrixR calculate_outputs(const MatrixR&);

    MatrixR calculate_outputs(const Tensor3&);

    MatrixR calculate_outputs(const Tensor4&);

    MatrixR calculate_directional_inputs(const Index, const VectorR&, float, float, Index = 101) const;

    Tensor3 calculate_outputs(const Tensor3&, const Tensor3&);

    Index calculate_image_output(const filesystem::path&);

    MatrixR calculate_text_outputs(const Tensor<string, 1>&);
    void from_JSON(const JsonDocument&);

    void to_JSON(JsonWriter&) const;

    void save(const filesystem::path&) const;
    void save_parameters_binary(const filesystem::path&) const;
    void save_states_binary(const filesystem::path&) const;

    void load(const filesystem::path&);
    void load_parameters_binary(const filesystem::path&);
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

    bfloat16* get_parameters_bf16_mirror_data()
    {
        return parameters.device_type == Device::CUDA
            && config.training_type == Type::BF16
            && !parameters.empty()
            && !parameters_bf16_mirror.empty()
            ? parameters_bf16_mirror.as<bfloat16>()
            : nullptr;
    }

    void copy_parameters_device();
    void copy_parameters_host();

    void copy_states_device();
    void copy_states_host();

private:

    MatrixR calculate_outputs_device(const vector<TensorView>&, ForwardPropagation&);

public:

    vector<string> get_layer_labels() const;

private:

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

    Buffer states;

    Configuration::Resolved config;

    mutable Index first_trainable_cache_ = -1;
    mutable Index last_trainable_cache_  = -1;
};

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
