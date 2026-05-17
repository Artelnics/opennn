//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   N E U R A L   N E T W O R K   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#pragma once

#include "layer.h"
#include "tensor_utilities.h"
#include "variable.h"

namespace opennn
{

#include "forward_propagation.h"

class NeuralNetwork
{

public:

    NeuralNetwork();

    virtual ~NeuralNetwork() = default;

    NeuralNetwork(const filesystem::path&);

    void add_layer(unique_ptr<Layer>,
                  const vector<Index>& = vector<Index>());

    [[nodiscard]] const Configuration::Resolved& get_config() const { return config; }
    [[nodiscard]] bool is_gpu() const { return config.device == Device::CUDA; }
    [[nodiscard]] bool is_cpu() const { return config.device == Device::CPU; }

    [[nodiscard]] Type get_training_type()  const { return config.training_type; }
    [[nodiscard]] Type get_inference_type() const { return config.inference_type; }

    [[nodiscard]] vector<vector<pair<Shape, Type>>> get_parameter_specs() const { return collect_layer_specs([](const Layer& L) { return L.get_parameter_specs(); }); }
    [[nodiscard]] vector<vector<pair<Shape, Type>>> get_state_specs()     const { return collect_layer_specs([](const Layer& L) { return L.get_state_specs(); }); }
    [[nodiscard]] vector<vector<pair<Shape, Type>>> get_forward_specs(Index b) const
    {
        auto specs = collect_layer_specs([b](const Layer& L) { return L.get_forward_specs(b); });
        if (!is_gpu()) force_specs_to_fp32(specs);
        return specs;
    }
    [[nodiscard]] vector<vector<pair<Shape, Type>>> get_backward_specs(Index b) const
    {
        auto specs = collect_layer_specs([b](const Layer& L) { return L.get_backward_specs(b); });
        if (!is_gpu()) force_specs_to_fp32(specs);
        return specs;
    }

    [[nodiscard]] Index get_states_size() const     { return get_aligned_size(get_state_specs()); }
    [[nodiscard]] Index get_forward_size(Index b)  const { return get_aligned_size(get_forward_specs(b));  }
    [[nodiscard]] Index get_backward_size(Index b) const { return get_aligned_size(get_backward_specs(b)); }

    void compile();
    [[nodiscard]] bool has(const string&) const;
    [[nodiscard]] bool has(LayerType) const;

    [[nodiscard]] bool is_empty() const { return layers.empty(); }

    [[nodiscard]] float* get_parameters_data() { return parameters.as<float>(); }
    [[nodiscard]] const float* get_parameters_data() const { return parameters.as<float>(); }
    [[nodiscard]] Index get_parameters_size() const { return parameters.size_in_floats(); }

    [[nodiscard]] const vector<Variable>& get_input_variables() const { return input_variables; }
    [[nodiscard]] vector<string> get_input_feature_names() const;

    [[nodiscard]] const vector<Variable>& get_output_variables() const { return output_variables; }
    [[nodiscard]] vector<string> get_output_feature_names() const;

    [[nodiscard]] const vector<unique_ptr<Layer>>& get_layers() const { return layers; }
    [[nodiscard]] const unique_ptr<Layer>& get_layer(const Index i) const { return layers[i]; }
    [[nodiscard]] const unique_ptr<Layer>& get_layer(const string&) const;

    [[nodiscard]] Index get_layer_index(const string&) const;

    [[nodiscard]] const vector<vector<Index>>& get_layer_input_indices() const { return layer_input_indices; }
    [[nodiscard]] vector<vector<Index>> get_layer_output_indices() const;

    [[nodiscard]] Layer* get_first(const string&);
    [[nodiscard]] Layer* get_first(LayerType);
    [[nodiscard]] const Layer* get_first(const string&) const;
    [[nodiscard]] const Layer* get_first(LayerType) const;
    void set_layers_number(const Index new_layers_number) { layers.resize(new_layers_number); layer_input_indices.resize(new_layers_number); }

    void set_layer_input_indices(const vector<vector<Index>>& new_layer_input_indices) { layer_input_indices = new_layer_input_indices; }
    void set_layer_input_indices(const Index layer_index, const vector<Index>& new_input_indices) { layer_input_indices[layer_index] = new_input_indices; }

    void set_layer_input_indices(const string&, const vector<string>&);
    void set_layer_input_indices(const string&, initializer_list<string>);
    void set_layer_input_indices(const string&, const string&);

    void set_input_variables(const vector<Variable>& new_input_variables) { input_variables = new_input_variables; }
    void set_output_variables(const vector<Variable>& new_output_variables) { output_variables = new_output_variables; }

    void set_input_names(const vector<string>&);
    void set_output_names(const vector<string>&);

    void set_input_shape(const Shape&);

    void set_default();
    [[nodiscard]] Index get_layers_number() const { return ssize(layers); }
    [[nodiscard]] Index get_layers_number(const string&) const;
    [[nodiscard]] Index get_layers_number(LayerType) const;

    [[nodiscard]] Index get_first_trainable_layer_index() const;
    [[nodiscard]] Index get_last_trainable_layer_index() const;
    [[nodiscard]] Index get_inputs_number() const;
    [[nodiscard]] Index get_outputs_number() const;

    [[nodiscard]] Shape get_input_shape() const;
    [[nodiscard]] Shape get_output_shape() const;

    [[nodiscard]] ActivationOp::Function get_output_activation() const;
    [[nodiscard]] Index get_parameters_number() const;

    [[nodiscard]] vector<Index> get_layer_parameter_numbers() const;

    void set_parameters(const VectorR& new_parameters);
    void set_parameters_random();
    void set_parameters_glorot();
    void link_parameters();
    [[nodiscard]] MatrixR calculate_outputs(const vector<TensorView>&);

    [[nodiscard]] MatrixR calculate_outputs(const MatrixR&);

    [[nodiscard]] MatrixR calculate_outputs(const Tensor3&);

    [[nodiscard]] MatrixR calculate_outputs(const Tensor4&);

    [[nodiscard]] MatrixR calculate_directional_inputs(const Index, const VectorR&, float, float, Index = 101) const;

    [[nodiscard]] Tensor3 calculate_outputs(const Tensor3&, const Tensor3&);

    [[nodiscard]] Index calculate_image_output(const filesystem::path&);

    [[nodiscard]] MatrixR calculate_text_outputs(const Tensor<string, 1>&);
    void from_JSON(const JsonDocument&);

    void to_JSON(JsonWriter&) const;

    void save(const filesystem::path&) const;
    void save_parameters(const filesystem::path&) const;
    void save_parameters_binary(const filesystem::path&) const;

    void load(const filesystem::path&);
    void load_parameters_binary(const filesystem::path&);

    [[nodiscard]] vector<string> get_names_string() const;

    void save_outputs(MatrixR&, const filesystem::path&);
    void save_outputs(Tensor3&, const filesystem::path&);

    void forward_propagate(const vector<TensorView>&,
                          ForwardPropagation&,
                          bool = false) const;

    void forward_propagate(const vector<TensorView>&,
                          ForwardPropagation&,
                          bool is_training,
                          Index first_layer_index,
                          Index last_layer_index) const;

    void forward_propagate(const vector<TensorView>&,
                          const VectorR&,
                          ForwardPropagation&);

#ifdef OPENNN_HAS_CUDA

public:

    void cast_parameters_to_bf16();

    // Returns nullptr when no BF16 mirror is allocated (FP32-only mode), so
    // optimizer kernels can pass it straight through and skip the mirror write.
    [[nodiscard]] bfloat16* get_parameters_bf16_data()
    {
        return parameters_bf16.empty() ? nullptr : parameters_bf16.as<bfloat16>();
    }

    void copy_parameters_device();
    void copy_parameters_host();

    void copy_states_device();
    void copy_states_host();
    void link_states();

private:

    [[nodiscard]] MatrixR calculate_outputs_device(const vector<TensorView>&, ForwardPropagation&);

#endif

public:

    [[nodiscard]] vector<string> get_layer_labels() const;

private:

    void validate_type(LayerType) const;

    static void force_specs_to_fp32(vector<vector<pair<Shape, Type>>>& specs)
    {
        for (auto& layer_specs : specs)
            for (auto& spec : layer_specs)
                spec.second = Type::FP32;
    }

    template<typename Fn>
    [[nodiscard]] vector<vector<pair<Shape, Type>>> collect_layer_specs(Fn fn) const
    {
        vector<vector<pair<Shape, Type>>> out(layers.size());
        ranges::transform(layers, out.begin(),
                          [&](const unique_ptr<Layer>& l) { return fn(*l); });
        return out;
    }

protected:

    string name = "neural_network";

    vector<Variable> input_variables;
    vector<Variable> output_variables;

    vector<unique_ptr<Layer>> layers;

    vector<vector<Index>> layer_input_indices;

    Buffer parameters;
    Buffer parameters_bf16{Device::CUDA};

    Buffer states;

    Configuration::Resolved config;

    // Cached by get_first/last_trainable_layer_index after first computation.
    // Invalidated when the layer list changes (add_layer / clear).
    mutable Index first_trainable_cache_ = -1;
    mutable Index last_trainable_cache_  = -1;
};

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
