//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   N E U R A L   N E T W O R K   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#pragma once

#include "layer.h"
#include "tensors.h"

namespace opennn
{

class NeuralNetwork;

struct NeuralNetworkBackPropagation;
struct NeuralNetworkBackPropagationLM;

#ifdef OPENNN_CUDA
struct ForwardPropagationCuda;
struct NeuralNetworkBackPropagationCuda;
#endif

struct ForwardPropagation
{
    ForwardPropagation(const Index = 0, NeuralNetwork* = nullptr);

    void set(const Index = 0, NeuralNetwork* = nullptr);

    vector<vector<TensorView*>> get_layer_workspace_views();

    TensorView get_last_trainable_layer_outputs() const;

    vector<vector<TensorView>> get_layer_input_views(const vector<TensorView>&, bool) const;

    TensorView get_outputs();

    void print() const;

    Index samples_number = 0;

    NeuralNetwork* neural_network = nullptr;

    vector<unique_ptr<LayerForwardPropagation>> layers;

    VectorR workspace;
};


#ifdef OPENNN_CUDA

struct ForwardPropagationCuda
{
    ForwardPropagationCuda(const Index = 0, NeuralNetwork* = nullptr);

    ~ForwardPropagationCuda() { free(); }

    void set(const Index = 0, NeuralNetwork* = nullptr);

    vector<vector<TensorViewCuda*>> get_layer_workspace_views_device();

    TensorViewCuda get_last_trainable_layer_outputs_device() const;

    vector<vector<TensorViewCuda>> get_layer_input_views_device(const vector<TensorViewCuda>&, bool) const;

    TensorViewCuda get_outputs()
    {
        return layers.back()->get_outputs();
    }

    void print();

    void free();

    Index samples_number = 0;

    NeuralNetwork* neural_network = nullptr;

    vector<unique_ptr<LayerForwardPropagationCuda>> layers;

    TensorCuda workspace;
};

#endif


class NeuralNetwork
{

public:

    NeuralNetwork();

    NeuralNetwork(const filesystem::path&);

    void add_layer(unique_ptr<Layer>,
                  const vector<Index>& = vector<Index>());

    vector<vector<TensorView*>> get_layer_parameter_views();

    void compile();

    bool validate_name(const string&) const;

    void reference_all_layers();

    // Get

    bool has(const string&) const;

    bool is_empty() const;

    VectorR& get_parameters()
    {
        return parameters;
    }

    const vector<string>& get_feature_names() const;
    Index get_input_index(const string&) const;

    const vector<string>& get_output_names() const;
    Index get_output_index(const string&) const;

    const vector<unique_ptr<Layer>>& get_layers() const;
    const unique_ptr<Layer>& get_layer(const Index) const;
    const unique_ptr<Layer>& get_layer(const string&) const;

    Index get_layer_index(const string&) const;

    const vector<vector<Index>>& get_layer_input_indices() const;
    vector<vector<Index>> get_layer_output_indices() const;

    Index find_input_index(const vector<Index>&, Index) const;

    Layer* get_first(const string&) const;

    bool get_display() const;

    const vector<string>& get_input_vocabulary() const;

    const vector<string>& get_output_vocabulary() const;

    // Set

    void set(const filesystem::path&);

    void set_layers_number(const Index);

    void set_layer_input_indices(const vector<vector<Index>>&);
    void set_layer_input_indices(const Index, const vector<Index>&);

    void set_layer_input_indices(const string&, const vector<string>&);
    void set_layer_input_indices(const string&, const initializer_list<string>&);
    void set_layer_input_indices(const string&, const string&);

    void set_feature_names(const vector<string>&);
    void set_output_names(const vector<string>&);

    void set_input_shape(const Shape&);

    void set_default();

    void set_display(bool);

    void set_input_vocabulary(const vector<string>&);
    void set_output_vocabulary(const vector<string>&);

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
/*
    template <Index input_rank, Index output_rank>
    Tensor<type, output_rank> calculate_outputs(const Tensor<type, input_rank>& inputs)
    {
        const Index layers_number = get_layers_number();

        if (layers_number == 0)
           return Tensor<type, output_rank>();

        const Index batch_size = inputs.dimension(0);

        ForwardPropagation forward_propagation(batch_size, this);

        Shape input_shape;

        for(Index i = 0; i < input_rank; ++i)
           input_shape.push_back(inputs.dimension(i));

        const TensorView input_view((type*)inputs.data(), input_shape);

        forward_propagate({input_view}, forward_propagation, false);

        const TensorView& output_view = forward_propagation.layers.back()->get_outputs();

        if constexpr (output_rank == 2)
        {
            if (output_view.rank() == 4)
            {
                const Index batch_size = output_view.shape[0];
                const Index features = output_view.size() / batch_size;

                if (reinterpret_cast<uintptr_t>(output_view.data) % EIGEN_MAX_ALIGN_BYTES != 0)
                    throw runtime_error("tensor_map alignment error: Pointer is not aligned.");

                return MatrixMap(output_view.data, batch_size, features);
            }

            return matrix_map(output_view);
        }
        else if constexpr (output_rank == 3)
           return tensor_map<3>(output_view);
        else if constexpr (output_rank == 4)
           return tensor_map<4>(output_view);
        else
           static_assert(output_rank >= 2 && output_rank <= 4, "Unsupported output rank.");

        return Tensor<type, output_rank>();
    }
*/
    Tensor3 calculate_outputs(const Tensor3&, const Tensor3&);

    TensorView run_internal_forward_propagation(const type*, const Shape&);

    MatrixR calculate_outputs(const MatrixR&);

    MatrixR calculate_outputs(const Tensor3&);

    MatrixR calculate_outputs(const Tensor4&);

    MatrixR calculate_directional_inputs(const Index, const VectorR&, type, type, Index = 101) const;

    Index calculate_image_output(const filesystem::path&);

    MatrixR calculate_text_outputs(const Tensor<string, 1>&);

    // Serialization

    Tensor<string, 2> get_dense2d_layers_information() const;

    void from_XML(const XMLDocument&);
    void features_from_XML(const XMLElement*);
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

#ifdef OPENNN_CUDA

public:

    TensorCuda& get_parameters_device();

    vector<vector<TensorViewCuda*>> get_layer_parameter_views_device();

    void allocate_parameters_device();
    void copy_parameters_device();
    void copy_parameters_host();

    void forward_propagate(const vector<TensorViewCuda>&,
                           ForwardPropagationCuda&,
                           bool = false) const;

    TensorViewCuda calculate_outputs(TensorViewCuda, Index);

protected:

    TensorCuda parameters_device;

#endif

protected:

    string name = "neural_network";

    vector<string> input_names;

    vector<string> output_names;

    vector<string> input_vocabulary;
    vector<string> output_vocabulary;

    vector<unique_ptr<Layer>> layers;

    vector<vector<Index>> layer_input_indices;

    bool display = true;

    VectorR parameters;
};


struct NeuralNetworkBackPropagation
{
    NeuralNetworkBackPropagation(const Index = 0, NeuralNetwork* = nullptr);

    void set(const Index = 0, NeuralNetwork* = nullptr);

    const vector<unique_ptr<LayerBackPropagation>>& get_layers() const;

    vector<vector<TensorView*>> get_layer_gradient_views();

    void print() const;

    NeuralNetwork* get_neural_network() const;

    Index batch_size = 0;

    NeuralNetwork* neural_network = nullptr;

    vector<unique_ptr<LayerBackPropagation>> layers;

    VectorR gradient;

    // @todo
    //VectorR input_gradients;
};


#ifdef OPENNN_CUDA

struct NeuralNetworkBackPropagationCuda
{
    NeuralNetworkBackPropagationCuda(const Index = 0, NeuralNetwork* = nullptr);

    void set(const Index = 0, NeuralNetwork* = nullptr);

    const vector<unique_ptr<LayerBackPropagationCuda>>& get_layers() const;

    vector<vector<TensorViewCuda*>> get_layer_workspace_views_device();

    void print();

    void free();

    Index batch_size = 0;

    NeuralNetwork* neural_network = nullptr;

    vector<unique_ptr<LayerBackPropagationCuda>> layers;

    TensorCuda workspace;
};

#endif


struct NeuralNetworkBackPropagationLM
{
    NeuralNetworkBackPropagationLM(NeuralNetwork* new_neural_network = nullptr);

    void set(const Index = 0, NeuralNetwork* = nullptr);

    const vector<unique_ptr<LayerBackPropagationLM>>& get_layers() const;

    NeuralNetwork* get_neural_network() const;

    vector<vector<TensorView*>> get_layer_workspace_views();

    void print();

    Index batch_size = 0;

    NeuralNetwork* neural_network = nullptr;

    vector<unique_ptr<LayerBackPropagationLM>> layers;

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
