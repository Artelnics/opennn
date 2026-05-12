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

    const Configuration::Resolved& get_config() const { return config; }
    bool is_gpu() const { return config.device == Device::CUDA; }
    bool is_cpu() const { return config.device == Device::CPU; }

    Type get_training_type()  const { return config.training_type; }
    Type get_inference_type() const { return config.inference_type; }

    vector<vector<Shape>> get_parameter_shapes()        const { return collect_layer_shapes([](const Layer& L)            { return L.get_parameter_shapes(); }); }
    vector<vector<Shape>> get_state_shapes()            const { return collect_layer_shapes([](const Layer& L)            { return L.get_state_shapes(); }); }
    vector<vector<Shape>> get_forward_shapes(Index b)   const { return collect_layer_shapes([b](const Layer& L)           { return L.get_forward_shapes(b); }); }
    vector<vector<Shape>> get_backward_shapes(Index b)  const { return collect_layer_shapes([b](const Layer& L)           { return L.get_backward_shapes(b); }); }

    Index get_states_size() const     { return aligned_total_elements(get_state_shapes()); }
    Index get_forward_size(Index b)  const { return aligned_total_elements(get_forward_shapes(b));  }
    Index get_backward_size(Index b) const { return aligned_total_elements(get_backward_shapes(b)); }

    void compile();
    bool has(const string&) const;
    bool has(LayerType) const;

    bool is_empty() const { return layers.empty(); }

    float* get_parameters_data() { return parameters.as<float>(); }
    const float* get_parameters_data() const { return parameters.as<float>(); }
    Index get_parameters_size() const { return parameters.size_in_floats(); }

    const vector<Variable>& get_input_variables() const { return input_variables; }
    vector<string> get_input_feature_names() const;

    const vector<Variable>& get_output_variables() const { return output_variables; }
    vector<string> get_output_feature_names() const;

    const vector<unique_ptr<Layer>>& get_layers() const { return layers; }
    const unique_ptr<Layer>& get_layer(const Index i) const { return layers[i]; }
    const unique_ptr<Layer>& get_layer(const string&) const;

    Index get_layer_index(const string&) const;

    const vector<vector<Index>>& get_layer_input_indices() const { return layer_input_indices; }
    vector<vector<Index>> get_layer_output_indices() const;

    Layer* get_first(const string&);
    Layer* get_first(LayerType);
    const Layer* get_first(const string&) const;
    const Layer* get_first(LayerType) const;
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
    Index get_layers_number() const { return ssize(layers); }
    Index get_layers_number(const string&) const;
    Index get_layers_number(LayerType) const;

    Index get_first_trainable_layer_index() const;
    Index get_last_trainable_layer_index() const;
    Index get_inputs_number() const;
    Index get_outputs_number() const;

    Shape get_input_shape() const;
    Shape get_output_shape() const;

    ActivationOp::Function get_output_activation() const;
    Index get_parameters_number() const;

    vector<Index> get_layer_parameter_numbers() const;

    void set_parameters(const VectorR& new_parameters);
    void set_parameters_random();
    void set_parameters_glorot();
    MatrixR calculate_outputs(const vector<TensorView>&);

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
    void save_parameters(const filesystem::path&) const;
    void save_parameters_binary(const filesystem::path&) const;

    void load(const filesystem::path&);
    void load_parameters_binary(const filesystem::path&);

    vector<string> get_names_string() const;

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
    __nv_bfloat16* get_parameters_bf16_data()
    {
        return parameters_bf16.empty() ? nullptr : parameters_bf16.as<__nv_bfloat16>();
    }

    void copy_parameters_device();
    void copy_parameters_host();
    void link_parameters();

    void copy_states_device();
    void copy_states_host();
    void link_states();

private:

    MatrixR calculate_outputs_device(const vector<TensorView>&, ForwardPropagation&);

#endif

public:

    vector<string> get_layer_labels() const;

private:

    void validate_type(LayerType) const;

    // Gather a per-layer shape vector via the supplied accessor.
    template<typename Fn>
    vector<vector<Shape>> collect_layer_shapes(Fn fn) const
    {
        const Index n = get_layers_number();
        vector<vector<Shape>> out(n);
        for (Index i = 0; i < n; ++i) out[i] = fn(*layers[i]);
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
    vector<vector<vector<TensorView>>> parameter_views;

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
