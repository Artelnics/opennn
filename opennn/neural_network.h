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

    vector<vector<Shape>> get_parameter_shapes() const
    {
        const size_t layers_number = get_layers_number();
        vector<vector<Shape>> shapes(layers_number);

        for(size_t i = 0; i < layers_number; i++)
            shapes[i] = layers[i]->get_parameter_shapes();

        return shapes;
    }

    vector<vector<Shape>> get_forward_shapes(Index batch_size) const
    {
        const size_t layers_number = get_layers_number();
        vector<vector<Shape>> shapes(layers_number);

        for(size_t i = 0; i < layers_number; i++)
            shapes[i] = layers[i]->get_forward_shapes(batch_size);

        return shapes;
    }

    vector<vector<Shape>> get_backward_shapes(Index batch_size) const
    {
        const size_t layers_number = get_layers_number();
        vector<vector<Shape>> shapes(layers_number);

        for(size_t i = 0; i < layers_number; i++)
            shapes[i] = layers[i]->get_backward_shapes(batch_size);

        return shapes;
    }

    void compile();

    // Get

    bool has(const string&) const;
    bool has(LayerType) const;

    bool is_empty() const { return layers.empty(); }

    VectorR& get_parameters() { return parameters.vector; }
    const VectorR& get_parameters() const { return parameters.vector; }

    type* get_parameters_data() { return parameters.data(); }
    const type* get_parameters_data() const { return parameters.data(); }
    Index get_parameters_size() const { return parameters.size(); }

    const vector<Variable>& get_input_variables() const { return input_variables; }
    const vector<string> get_input_feature_names() const;

    const vector<Variable>& get_output_variables() const { return output_variables; }
    const vector<string> get_output_feature_names() const;

    const vector<unique_ptr<Layer>>& get_layers() const { return layers; }
    const unique_ptr<Layer>& get_layer(const Index i) const { return layers[i]; }
    const unique_ptr<Layer>& get_layer(const string&) const;

    Index get_layer_index(const string&) const;

    const vector<vector<Index>>& get_layer_input_indices() const { return layer_input_indices; }
    vector<vector<Index>> get_layer_output_indices() const;

    Index find_input_index(const vector<Index>&, Index) const;

    Layer* get_first(const string&);
    Layer* get_first(LayerType);
    const Layer* get_first(const string&) const;
    const Layer* get_first(LayerType) const;

    // Set

    void set_layers_number(const Index n) { layers.resize(n); layer_input_indices.resize(n); }

    void set_layer_input_indices(const vector<vector<Index>>& v) { layer_input_indices = v; }
    void set_layer_input_indices(const Index i, const vector<Index>& v) { layer_input_indices[i] = v; }

    void set_layer_input_indices(const string&, const vector<string>&);
    void set_layer_input_indices(const string&, initializer_list<string>);
    void set_layer_input_indices(const string&, const string&);

    void set_input_variables(const vector<Variable>& v) { input_variables = v; }
    void set_output_variables(const vector<Variable>& v) { output_variables = v; }

    void set_input_names(const vector<string>&);
    void set_output_names(const vector<string>&);

    void set_input_shape(const Shape&);

    void set_default();

    // Layers

    size_t get_layers_number() const { return layers.size(); }
    size_t get_layers_number(const string&) const;
    size_t get_layers_number(LayerType) const;

    Index get_first_trainable_layer_index() const;
    Index get_last_trainable_layer_index() const;

    // Architecture

    Index get_inputs_number() const;
    Index get_outputs_number() const;

    Shape get_input_shape() const;
    Shape get_output_shape() const;

    ActivationFunction get_output_activation() const;

    // Parameters

    Index get_parameters_number() const;

    vector<Index> get_layer_parameter_numbers() const;

    void set_parameters(const VectorR& p) { parameters.vector = p; }

    // Parameters initialization

    void set_parameters_random();
    void set_parameters_glorot();

    // Output

    MatrixR calculate_outputs(const vector<TensorView>&);

    MatrixR calculate_outputs(const MatrixR&);

    MatrixR calculate_outputs(const Tensor3&);

    MatrixR calculate_outputs(const Tensor4&);

    MatrixR calculate_directional_inputs(const Index, const VectorR&, type, type, Index = 101) const;

    Tensor3 calculate_outputs(const Tensor3&, const Tensor3&);

    Index calculate_image_output(const filesystem::path&);

    MatrixR calculate_text_outputs(const Tensor<string, 1>&);

    // Serialization

    void from_XML(const XmlDocument&);

    void to_XML(XmlPrinter&) const;

    void save(const filesystem::path&) const;
    void save_parameters(const filesystem::path&) const;

    void load(const filesystem::path&);
    void load_parameters_binary(const filesystem::path&);

    vector<string> get_names_string() const;

    void save_outputs(MatrixR&, const filesystem::path&);
    void save_outputs(Tensor3&, const filesystem::path&);

    void forward_propagate(const vector<TensorView>&,
                          ForwardPropagation&,
                          bool = false) const;

    void forward_propagate(const vector<TensorView>&,
                          const VectorR&,
                          ForwardPropagation&);

#ifdef OPENNN_WITH_CUDA

public:

    type* get_parameters_device() { return parameters.device(); }
    const type* get_parameters_device() const { return parameters.device(); }

    void copy_parameters_device();
    void copy_parameters_host();
    void link_parameters_device();
    void link_parameters_cpu();

    void copy_states_device();
    void copy_states_host();
    void link_states_device();
    void link_states_cpu();

private:

    MatrixR calculate_outputs_device(const vector<TensorView>&, ForwardPropagation&);

#endif

public:

    vector<string> get_layer_labels() const;

private:

    void validate_type(LayerType) const;

protected:

    string name = "neural_network";

    vector<Variable> input_variables;
    vector<Variable> output_variables;

    vector<unique_ptr<Layer>> layers;

    vector<vector<Index>> layer_input_indices;

    Memory parameters;
    vector<vector<vector<TensorView>>> parameter_views;

    Memory states; // non-trainable persistent state
};

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
