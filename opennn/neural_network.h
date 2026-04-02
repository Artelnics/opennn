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

class NeuralNetwork;

struct ForwardPropagation
{
    ForwardPropagation(const Index = 0, NeuralNetwork* = nullptr);

    void set(const Index = 0, NeuralNetwork* = nullptr);

    TensorView get_last_trainable_layer_outputs() const;

    vector<vector<TensorView>> get_layer_input_views(const vector<TensorView>&, bool) const;

    TensorView get_outputs();

    void print() const;

    Index batch_size = 0;

    NeuralNetwork* neural_network = nullptr;

    VectorR data;
    vector<vector<vector<TensorView>>> views;
};


class NeuralNetwork
{

public:

    NeuralNetwork();

    NeuralNetwork(const filesystem::path&);

    void add_layer(unique_ptr<Layer>,
                  const vector<Index>& = vector<Index>());

    vector<vector<Shape>> get_parameter_shapes() const
    {
        const Index layers_number = get_layers_number();
        vector<vector<Shape>> shapes(layers_number);

        for(Index i = 0; i < layers_number; i++)
            shapes[i] = layers[i]->get_parameter_shapes();

        return shapes;
    }

    vector<vector<Shape>> get_forward_shapes(Index batch_size) const
    {
        const Index layers_number = get_layers_number();
        vector<vector<Shape>> shapes(layers_number);

        for(Index i = 0; i < layers_number; i++)
            shapes[i] = layers[i]->get_forward_shapes(batch_size);

        return shapes;
    }

    vector<vector<Shape>> get_backward_shapes(Index batch_size) const
    {
        const Index layers_number = get_layers_number();
        vector<vector<Shape>> shapes(layers_number);

        for(Index i = 0; i < layers_number; i++)
            shapes[i] = layers[i]->get_backward_shapes(batch_size);

        return shapes;
    }


    void compile();

    bool validate_name(const string&) const;

    void reference_all_layers();

    // Get

    bool has(const string&) const;

    bool is_empty() const;

    VectorR& get_parameters();

    const vector<Variable>& get_input_variables() const;
    const vector<string> get_input_feature_names() const;

    const vector<Variable>& get_output_variables() const;
    const vector<string> get_output_feature_names() const;

    const vector<unique_ptr<Layer>>& get_layers() const;
    const unique_ptr<Layer>& get_layer(const Index) const;
    const unique_ptr<Layer>& get_layer(const string&) const;

    Index get_layer_index(const string&) const;

    const vector<vector<Index>>& get_layer_input_indices() const;
    vector<vector<Index>> get_layer_output_indices() const;

    Index find_input_index(const vector<Index>&, Index) const;

    Layer* get_first(const string&) const;

    bool get_display() const;

    // Set

    void set(const filesystem::path&);

    void set_layers_number(const Index);

    void set_layer_input_indices(const vector<vector<Index>>&);
    void set_layer_input_indices(const Index, const vector<Index>&);

    void set_layer_input_indices(const string&, const vector<string>&);
    void set_layer_input_indices(const string&, const initializer_list<string>&);
    void set_layer_input_indices(const string&, const string&);

    void set_input_variables(const vector<Variable>&);
    void set_output_variables(const vector<Variable>&);


    void set_input_names(const vector<string>&);
    void set_output_names(const vector<string>&);

    //@simone @todo void set_input_names(const vector<string>&);
    //se ci sono nuovinomi deve richiamare il cambio nome nelle variabili

    void set_input_shape(const Shape&);

    void set_default();

    void set_display(bool);

    // Layers

    Index get_layers_number() const;
    Index get_layers_number(const string&) const;

    Index get_first_trainable_layer_index() const;
    Index get_last_trainable_layer_index() const;

    // Architecture

    Index get_inputs_number() const;
    Index get_outputs_number() const;

    Shape get_input_shape() const;
    Shape get_output_shape() const;

    // Parameters

    Index get_parameters_number() const;

    vector<Index> get_layer_parameter_numbers() const;

    void set_parameters(const VectorR&);

    // Parameters initialization

    void set_parameters_random();
    void set_parameters_glorot();

    // Output

    Tensor3 calculate_outputs(const Tensor3&, const Tensor3&);

    MatrixR calculate_outputs(const MatrixR&);

    MatrixR calculate_outputs(const Tensor3&);

    MatrixR calculate_outputs(const Tensor4&);

    MatrixR calculate_directional_inputs(const Index, const VectorR&, type, type, Index = 101) const;

    Index calculate_image_output(const filesystem::path&);

    MatrixR calculate_text_outputs(const Tensor<string, 1>&);

    // Serialization

    Tensor<string, 2> get_dense2d_layers_information() const;

    void from_XML(const XMLDocument&);
    void inputs_from_XML(const XMLElement*);
    void layers_from_XML(const XMLElement*);
    void outputs_from_XML(const XMLElement*);

    void to_XML(XMLPrinter&) const;

    virtual void print() const;
    void save(const filesystem::path&) const;
    void save_parameters(const filesystem::path&) const;

    void load(const filesystem::path&);
    void load_parameters_binary(const filesystem::path&);

    vector<string> get_layer_labels() const;
    vector<string> get_names_string() const;

    void save_outputs(MatrixR&, const filesystem::path&);
    void save_outputs(Tensor3&, const filesystem::path&);

    void forward_propagate(const vector<TensorView>&,
                          ForwardPropagation&,
                          bool = false) const;

    void forward_propagate(const vector<TensorView>&,
                          const VectorR&,
                          ForwardPropagation&);

    string get_expression() const;

#ifdef CUDA

public:

    TensorCuda& get_parameters_device();

    vector<vector<TensorView*>> get_layer_parameter_views_device();

    void allocate_parameters_device();
    void copy_parameters_device();
    void copy_parameters_host();

    void forward_propagate(const vector<TensorView>&,
                           ForwardPropagationCuda&,
                           bool = false) const;

    void forward_propagate(const vector<TensorView>&,
                           const VectorR&,
                           ForwardPropagationCuda&);

    TensorView calculate_outputs(TensorView, Index);

protected:

    TensorCuda parameters_device;

#endif

protected:

    string name = "neural_network";

    vector<Variable> input_variables;
    vector<Variable> output_variables;

    vector<unique_ptr<Layer>> layers;

    vector<vector<Index>> layer_input_indices;

    bool display = true;

    VectorR parameters;
    vector<vector<vector<TensorView>>> parameter_views;

};


struct NeuralNetworkBackPropagationLM
{
    NeuralNetworkBackPropagationLM(NeuralNetwork* new_neural_network = nullptr);

    void set(const Index = 0, NeuralNetwork* = nullptr);

    const vector<unique_ptr<LayerBackPropagationLM>>& get_layers() const;

    NeuralNetwork* get_neural_network() const;

    void print();

    Index batch_size = 0;

    NeuralNetwork* neural_network = nullptr;

    vector<unique_ptr<LayerBackPropagationLM>> layers;

    VectorR gradient;

    VectorR workspace;
};

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
