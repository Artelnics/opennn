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

/// @brief Container of layers forming a feed-forward neural network, with parameter storage and I/O.
class NeuralNetwork
{

public:

    /// @brief Constructs an empty neural network.
    NeuralNetwork();

    virtual ~NeuralNetwork() = default;

    /// @brief Constructs a neural network and loads its definition from a JSON file.
    /// @param path Path to the JSON file describing the network.
    NeuralNetwork(const filesystem::path&);

    /// @brief Appends a layer to the network.
    /// @param layer Owning pointer to the layer; the network takes ownership.
    /// @param source_indices Indices of layers feeding into this one (empty means the previous layer).
    void add_layer(unique_ptr<Layer>,
                  const vector<Index>& = {});

    [[nodiscard]] const Configuration::Resolved& get_config() const { return config; }
    [[nodiscard]] bool is_gpu() const { return config.device == Device::CUDA; }
    [[nodiscard]] bool is_cpu() const { return config.device == Device::CPU; }

    [[nodiscard]] Type get_training_type()  const { return config.training_type; }
    [[nodiscard]] Type get_inference_type() const { return config.inference_type; }

    /// @brief Returns the tensor specs of trainable parameters for every layer.
    [[nodiscard]] vector<vector<TensorSpec>> get_parameter_specs() const { return collect_layer_specs([](const Layer& L) { return L.get_parameter_specs(); }); }

    /// @brief Returns the tensor specs of persistent layer state (e.g. running statistics).
    [[nodiscard]] vector<vector<TensorSpec>> get_state_specs()     const { return collect_layer_specs([](const Layer& L) { return L.get_state_specs(); }); }

    /// @brief Returns the tensor specs of the forward-propagation workspace for each layer.
    /// @param b Batch size used to size the per-layer activations.
    [[nodiscard]] vector<vector<TensorSpec>> get_forward_specs(Index b) const
    {
        auto specs = collect_layer_specs([b](const Layer& L) { return L.get_forward_specs(b); });
        if (!is_gpu()) force_specs_to_fp32(specs);
        return specs;
    }
    
    /// @brief Returns the tensor specs of the back-propagation workspace for each layer.
    /// @param b Batch size used to size the per-layer gradient buffers.
    [[nodiscard]] vector<vector<TensorSpec>> get_backward_specs(Index b) const
    {
        auto specs = collect_layer_specs([b](const Layer& L) { return L.get_backward_specs(b); });
        if (!is_gpu()) force_specs_to_fp32(specs);
        return specs;
    }

    /// @brief Returns the total byte size required to hold all persistent layer states.
    [[nodiscard]] Index get_states_size() const     { return get_aligned_size(get_state_specs()); }

    /// @brief Allocates buffers, resolves devices, and wires layer/operator views; call once after all layers are added.
    void compile();

    /// @brief Returns whether the network contains a layer with the given label.
    [[nodiscard]] bool has(const string&) const;

    /// @brief Returns whether the network contains at least one layer of the given type.
    [[nodiscard]] bool has(LayerType) const;

    [[nodiscard]] bool is_empty() const { return layers.empty(); }

    [[nodiscard]] float* get_parameters_data() { return parameters.as<float>(); }
    [[nodiscard]] const float* get_parameters_data() const { return parameters.as<float>(); }
    [[nodiscard]] Index get_parameters_size() const { return parameters.size_in_floats(); }

    [[nodiscard]] const vector<Variable>& get_input_variables() const { return input_variables; }

    /// @brief Returns the flat list of input feature names (expanding categorical variables).
    [[nodiscard]] vector<string> get_input_feature_names() const;

    [[nodiscard]] const vector<Variable>& get_output_variables() const { return output_variables; }

    /// @brief Returns the flat list of output feature names (expanding categorical variables).
    [[nodiscard]] vector<string> get_output_feature_names() const;

    [[nodiscard]] const vector<unique_ptr<Layer>>& get_layers() const { return layers; }
    [[nodiscard]] const unique_ptr<Layer>& get_layer(const Index i) const { return layers[i]; }

    /// @brief Returns the layer with the given label.
    /// @param label Label assigned to the layer (e.g. via Layer::set_label).
    [[nodiscard]] const unique_ptr<Layer>& get_layer(const string&) const;

    /// @brief Returns the index of the layer with the given label, or -1 if not found.
    [[nodiscard]] Index get_layer_index(const string&) const;

    [[nodiscard]] const vector<vector<Index>>& get_source_layers() const { return source_layers; }

    /// @brief Returns the inverse adjacency: for each layer, the indices of layers that consume its output.
    [[nodiscard]] vector<vector<Index>> get_consumer_layers() const;

    /// @brief Returns the first layer matching the given label, or nullptr if not found.
    [[nodiscard]] Layer* get_first(const string&);

    /// @brief Returns the first layer of the given type, or nullptr if not found.
    [[nodiscard]] Layer* get_first(LayerType);

    /// @copydoc NeuralNetwork::get_first(const string&)
    [[nodiscard]] const Layer* get_first(const string&) const;

    /// @copydoc NeuralNetwork::get_first(LayerType)
    [[nodiscard]] const Layer* get_first(LayerType) const;

    /// @brief Replaces the layer connectivity graph.
    /// @param new_source_layers For each layer index, the indices of its source layers.
    void set_source_layers(const vector<vector<Index>>& new_source_layers) { source_layers = new_source_layers; }

    /// @brief Replaces the source layers for one specific layer.
    void set_source_layers(const Index layer_index, const vector<Index>& new_sources) { source_layers[layer_index] = new_sources; }

    /// @brief Sets the source layers of a layer using labels for identification.
    /// @param target Label of the destination layer.
    /// @param sources Labels of the source layers feeding into the target.
    void set_source_layers(const string&, const vector<string>&);

    /// @copydoc NeuralNetwork::set_source_layers(const string&, const vector<string>&)
    void set_source_layers(const string&, initializer_list<string>);

    /// @brief Convenience overload for a single source layer.
    /// @param target Label of the destination layer.
    /// @param source Label of the source layer feeding into the target.
    void set_source_layers(const string&, const string&);

    void set_input_variables(const vector<Variable>& new_input_variables) { input_variables = new_input_variables; }
    void set_output_variables(const vector<Variable>& new_output_variables) { output_variables = new_output_variables; }

    /// @brief Sets the names of every input feature.
    void set_input_names(const vector<string>&);

    /// @brief Sets the names of every output feature.
    void set_output_names(const vector<string>&);

    /// @brief Sets the shape of the input of the first layer and propagates it through the graph.
    void set_input_shape(const Shape&);

    /// @brief Removes all layers and resets the network to an empty state.
    void clear();
    
    [[nodiscard]] Index get_layers_number() const { return ssize(layers); }

    /// @brief Returns the number of layers whose label contains the given substring.
    [[nodiscard]] Index get_layers_number(const string&) const;

    /// @brief Returns the number of layers of the given type.
    [[nodiscard]] Index get_layers_number(LayerType) const;

    /// @brief Returns the index of the first trainable layer (cached).
    [[nodiscard]] Index get_first_trainable_layer_index() const;

    /// @brief Returns the index of the last trainable layer (cached).
    [[nodiscard]] Index get_last_trainable_layer_index() const;

    /// @brief Returns the number of input features expected by the first layer.
    [[nodiscard]] Index get_inputs_number() const;

    /// @brief Returns the number of output features produced by the last layer.
    [[nodiscard]] Index get_outputs_number() const;

    /// @brief Returns the shape of the input of the first layer.
    [[nodiscard]] Shape get_input_shape() const;

    /// @brief Returns the shape of the output of the last layer.
    [[nodiscard]] Shape get_output_shape() const;

    /// @brief Returns the activation function of the output layer.
    [[nodiscard]] ActivationOp::Function get_output_activation() const;

    /// @brief Returns the total number of trainable parameters across all layers.
    [[nodiscard]] Index get_parameters_number() const;

    /// @brief Copies the contents of @p new_parameters into the network's parameter buffer.
    void set_parameters(const VectorR& new_parameters);

    /// @brief Initializes every parameter with random values.
    void set_parameters_random();

    /// @brief Initializes every parameter using Glorot (Xavier) initialization.
    void set_parameters_glorot();

    /// @brief Wires the contiguous parameter buffer to per-layer / per-operator views.
    void link_parameters();

    /// @brief Wires the contiguous state buffer to per-layer / per-operator views.
    void link_states();

    /// @brief Computes outputs for the given input tensor views.
    /// @param inputs Tensor views of the inputs (one per input variable).
    /// @return Matrix of outputs with one row per sample.
    [[nodiscard]] MatrixR calculate_outputs(const vector<TensorView>&);

    /// @brief Computes outputs for a 2D input matrix.
    [[nodiscard]] MatrixR calculate_outputs(const MatrixR&);

    /// @brief Computes outputs for a 3D input tensor.
    [[nodiscard]] MatrixR calculate_outputs(const Tensor3&);

    /// @brief Computes outputs for a 4D input tensor.
    [[nodiscard]] MatrixR calculate_outputs(const Tensor4&);

    /// @brief Generates samples by sweeping one input dimension across a range while keeping the others fixed.
    /// @param direction Index of the input to vary.
    /// @param point Baseline values for the remaining inputs.
    /// @param minimum Lower bound of the sweep range.
    /// @param maximum Upper bound of the sweep range.
    /// @param points_number Number of points sampled in the range.
    /// @return Matrix with one row per sampled point.
    [[nodiscard]] MatrixR calculate_directional_inputs(const Index, const VectorR&, float, float, Index = 101) const;

    /// @brief Computes outputs for an encoder/decoder model.
    /// @param encoder_input Encoder side input tensor.
    /// @param decoder_input Decoder side input tensor.
    [[nodiscard]] Tensor3 calculate_outputs(const Tensor3&, const Tensor3&);

    /// @brief Reads an image file and returns the predicted class index.
    [[nodiscard]] Index calculate_image_output(const filesystem::path&);

    /// @brief Tokenizes the given strings and returns the network's outputs.
    [[nodiscard]] MatrixR calculate_text_outputs(const Tensor<string, 1>&);

    /// @brief Restores the network architecture and parameters from a JSON document.
    void from_JSON(const JsonDocument&);

    /// @brief Serializes the network architecture and parameters to a JSON writer.
    void to_JSON(JsonWriter&) const;

    /// @brief Saves the full network (architecture + parameters) to a JSON file.
    void save(const filesystem::path&) const;

    /// @brief Saves only the parameter values to a JSON file.
    void save_parameters(const filesystem::path&) const;

    /// @brief Saves only the parameter values to a binary file.
    void save_parameters_binary(const filesystem::path&) const;

    /// @brief Loads the full network (architecture + parameters) from a JSON file.
    void load(const filesystem::path&);

    /// @brief Loads parameter values from a binary file produced by save_parameters_binary().
    void load_parameters_binary(const filesystem::path&);

    /// @brief Returns the labels of all layers as a vector of strings.
    [[nodiscard]] vector<string> get_names_string() const;

    /// @brief Writes the output matrix to a CSV file.
    void save_outputs(MatrixR&, const filesystem::path&);

    /// @brief Writes the 3D output tensor to a CSV file.
    void save_outputs(Tensor3&, const filesystem::path&);

    /// @brief Runs a forward pass over all layers.
    /// @param inputs Tensor views of the inputs (one per input variable).
    /// @param forward_propagation Workspace receiving per-layer activations.
    /// @param is_training If true, enables training-only behavior (dropout, batch-norm stats).
    void forward_propagate(const vector<TensorView>&,
                          ForwardPropagation&,
                          bool = false) const;

    /// @brief Runs a forward pass over a contiguous sub-range of layers.
    /// @param inputs Tensor views of the inputs.
    /// @param forward_propagation Workspace receiving per-layer activations.
    /// @param is_training Enables training-only behavior.
    /// @param first_layer_index First layer index (inclusive) to evaluate.
    /// @param last_layer_index Last layer index (inclusive) to evaluate.
    void forward_propagate(const vector<TensorView>&,
                          ForwardPropagation&,
                          bool is_training,
                          Index first_layer_index,
                          Index last_layer_index) const;

    /// @brief Runs a forward pass after temporarily overwriting the parameter buffer.
    /// @param inputs Tensor views of the inputs.
    /// @param parameters Replacement parameter values used for this pass.
    /// @param forward_propagation Workspace receiving per-layer activations.
    void forward_propagate(const vector<TensorView>&,
                          const VectorR&,
                          ForwardPropagation&);

#ifdef OPENNN_HAS_CUDA

public:

    /// @brief Casts the FP32 parameter buffer into the BF16 device mirror used by GPU kernels.
    void cast_parameters_to_bf16();

    // Returns nullptr when no BF16 mirror is allocated (FP32-only mode), so
    // optimizer kernels can pass it straight through and skip the mirror write.
    [[nodiscard]] bfloat16* get_parameters_bf16_data()
    {
        return parameters_bf16.empty() ? nullptr : parameters_bf16.as<bfloat16>();
    }

    /// @brief Copies the parameter buffer from host to device memory.
    void copy_parameters_device();

    /// @brief Copies the parameter buffer from device back to host memory.
    void copy_parameters_host();

    /// @brief Copies the state buffer from host to device memory.
    void copy_states_device();

    /// @brief Copies the state buffer from device back to host memory.
    void copy_states_host();

private:

    [[nodiscard]] MatrixR calculate_outputs_device(const vector<TensorView>&, ForwardPropagation&);

#endif

public:

    /// @brief Returns the labels of all layers in order.
    [[nodiscard]] vector<string> get_layer_labels() const;

private:

    void validate_type(LayerType) const;

    static void force_specs_to_fp32(vector<vector<TensorSpec>>& specs)
    {
        for (auto& layer_specs : specs)
            for (auto& spec : layer_specs)
                spec.dtype = Type::FP32;
    }

    template<typename Fn>
    [[nodiscard]] vector<vector<TensorSpec>> collect_layer_specs(Fn fn) const
    {
        vector<vector<TensorSpec>> out(layers.size());
        ranges::transform(layers, out.begin(),
                          [&](const unique_ptr<Layer>& l) { return fn(*l); });
        return out;
    }

protected:

    vector<Variable> input_variables;
    vector<Variable> output_variables;

    vector<unique_ptr<Layer>> layers;

    vector<vector<Index>> source_layers;

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
