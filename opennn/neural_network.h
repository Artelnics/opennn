//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   N E U R A L   N E T W O R K   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

/**
 * @file neural_network.h
 * @brief Declares the NeuralNetwork class.
 *
 * NeuralNetwork is the base class for every model in OpenNN. It owns an
 * ordered stack of Layer instances, the parameter buffer they share, and the
 * device/precision configuration used for forward and backward passes.
 */

#pragma once

#include "layer.h"
#include "tensor_utilities.h"
#include "variable.h"

namespace opennn
{

#include "forward_propagation.h"

/**
 * @class NeuralNetwork
 * @brief Stack of Layers forming a trainable model.
 *
 * The class is designed to be subclassed; the standard architectures
 * (ApproximationNetwork, ClassificationNetwork, ForecastingNetwork,
 * ImageClassificationNetwork, etc.) preconfigure the layer stack for common
 * tasks. Direct users can also call add_layer() and compile() to build custom
 * topologies.
 *
 * Holds a single parameter Buffer that all layers index into, plus a parallel
 * states Buffer for layers that maintain hidden state. CUDA mirrors are
 * available when compiled with OPENNN_HAS_CUDA.
 */
class NeuralNetwork
{

public:

    /**
     * @brief Default-constructs an empty network.
     *
     * No layers, no input/output variables, default device/precision config.
     */
    NeuralNetwork();

    /// Defaulted virtual destructor for safe polymorphic deletion.
    virtual ~NeuralNetwork() = default;

    /**
     * @brief Constructs a network by loading a serialized model from disk.
     * @param file_name Path to a JSON file produced by save().
     */
    NeuralNetwork(const filesystem::path& file_name);

    /**
     * @brief Appends a layer to the stack.
     * @param new_layer Owning pointer to the layer; ownership is transferred.
     * @param input_indices Indices of the layers whose outputs feed this layer.
     *                      Empty means "the previous layer".
     */
    void add_layer(unique_ptr<Layer> new_layer, const vector<Index>& input_indices = vector<Index>());

    /**
     * @brief Returns the resolved device/precision configuration.
     * @return Const reference to the configuration.
     */
    const Configuration::Resolved& get_config() const { return config; }

    /**
     * @brief Reports whether the network is configured to run on CUDA.
     * @return true when the configured device is Device::CUDA.
     */
    bool is_gpu() const { return config.device == Device::CUDA; }

    /**
     * @brief Reports whether the network is configured to run on CPU.
     * @return true when the configured device is Device::CPU.
     */
    bool is_cpu() const { return config.device == Device::CPU; }

    /**
     * @brief Returns the precision used for training.
     * @return Configured training Type (e.g. FP32, BF16).
     */
    Type get_training_type()  const { return config.training_type; }

    /**
     * @brief Returns the precision used for inference.
     * @return Configured inference Type (e.g. FP32, BF16, FP16).
     */
    Type get_inference_type() const { return config.inference_type; }

    /**
     * @brief Returns the parameter shapes of every layer.
     * @return Outer vector indexed by layer, inner by parameter group.
     */
    vector<vector<Shape>> get_parameter_shapes()        const { return collect_layer_shapes([](const Layer& L)            { return L.get_parameter_shapes(); }); }

    /**
     * @brief Returns the persistent state shapes of every layer.
     * @return Outer vector indexed by layer, inner by state group.
     */
    vector<vector<Shape>> get_state_shapes()            const { return collect_layer_shapes([](const Layer& L)            { return L.get_state_shapes(); }); }

    /**
     * @brief Returns the forward-propagation buffer shapes for a given batch size.
     * @param batch_size Number of samples per forward pass.
     * @return Outer vector indexed by layer, inner by buffer group.
     */
    vector<vector<Shape>> get_forward_shapes(Index batch_size)   const { return collect_layer_shapes([batch_size](const Layer& L)   { return L.get_forward_shapes(batch_size); }); }

    /**
     * @brief Returns the backward-propagation buffer shapes for a given batch size.
     * @param batch_size Number of samples per backward pass.
     * @return Outer vector indexed by layer, inner by buffer group.
     */
    vector<vector<Shape>> get_backward_shapes(Index batch_size)  const { return collect_layer_shapes([batch_size](const Layer& L)   { return L.get_backward_shapes(batch_size); }); }

    /**
     * @brief Returns the total size in floats of all persistent state buffers.
     * @return Element count summed across layers, accounting for alignment.
     */
    Index get_states_size() const     { return aligned_total_elements(get_state_shapes()); }

    /**
     * @brief Returns the total size in floats of the forward-propagation workspace.
     * @param batch_size Number of samples per forward pass.
     * @return Element count summed across layers, accounting for alignment.
     */
    Index get_forward_size(Index batch_size)  const { return aligned_total_elements(get_forward_shapes(batch_size));  }

    /**
     * @brief Returns the total size in floats of the backward-propagation workspace.
     * @param batch_size Number of samples per backward pass.
     * @return Element count summed across layers, accounting for alignment.
     */
    Index get_backward_size(Index batch_size) const { return aligned_total_elements(get_backward_shapes(batch_size)); }

    /**
     * @brief Allocates parameters and resolves layer-input indices.
     *
     * Should be called after the layer stack is fully assembled. Validates
     * input/output shape consistency and sets up the shared parameter buffer.
     */
    void compile();

    /**
     * @brief Reports whether the network contains a layer of the given name.
     * @param layer_name Layer type name (e.g. "Dense", "Recurrent").
     * @return true if any layer matches the name.
     */
    bool has(const string& layer_name) const;

    /**
     * @brief Reports whether the network contains a layer of the given type.
     * @param type Layer type enum.
     * @return true if any layer matches the type.
     */
    bool has(LayerType type) const;

    /**
     * @brief Reports whether the network has zero layers.
     * @return true when the layer stack is empty.
     */
    bool is_empty() const { return layers.empty(); }

    /**
     * @brief Returns the parameter buffer as a raw float pointer.
     * @return Pointer to the underlying float storage.
     */
    float* get_parameters_data() { return parameters.as<float>(); }

    /**
     * @brief Returns the parameter buffer as a raw float pointer (const overload).
     * @return Const pointer to the underlying float storage.
     */
    const float* get_parameters_data() const { return parameters.as<float>(); }

    /**
     * @brief Returns the parameter count in floats.
     * @return Number of float entries in the parameter buffer.
     */
    Index get_parameters_size() const { return parameters.size_in_floats(); }

    /**
     * @brief Returns the input variables describing each input feature.
     * @return Const reference to the input variables vector.
     */
    const vector<Variable>& get_input_variables() const { return input_variables; }

    /**
     * @brief Returns the names of the input features.
     * @return Vector of input feature names, one per input variable.
     */
    const vector<string> get_input_feature_names() const;

    /**
     * @brief Returns the output variables describing each output target.
     * @return Const reference to the output variables vector.
     */
    const vector<Variable>& get_output_variables() const { return output_variables; }

    /**
     * @brief Returns the names of the output features.
     * @return Vector of output feature names, one per output variable.
     */
    const vector<string> get_output_feature_names() const;

    /**
     * @brief Returns the layer stack.
     * @return Const reference to the vector of owned Layer pointers.
     */
    const vector<unique_ptr<Layer>>& get_layers() const { return layers; }

    /**
     * @brief Returns the layer at a given index.
     * @param i Zero-based index into the layer stack.
     * @return Const reference to the owned Layer pointer.
     */
    const unique_ptr<Layer>& get_layer(const Index i) const { return layers[i]; }

    /**
     * @brief Returns the layer with a given name.
     * @param layer_name Layer name.
     * @return Const reference to the owned Layer pointer.
     */
    const unique_ptr<Layer>& get_layer(const string& layer_name) const;

    /**
     * @brief Returns the index of the layer with a given name.
     * @param layer_name Layer name.
     * @return Zero-based index into the layer stack.
     */
    Index get_layer_index(const string& layer_name) const;

    /**
     * @brief Returns the per-layer input-layer indices.
     * @return Outer vector indexed by destination layer, inner by source layer.
     */
    const vector<vector<Index>>& get_layer_input_indices() const { return layer_input_indices; }

    /**
     * @brief Returns the per-layer output-layer indices.
     * @return Outer vector indexed by source layer, inner by destination layer.
     */
    vector<vector<Index>> get_layer_output_indices() const;

    /**
     * @brief Returns the first layer of a given type by name.
     * @param layer_name Layer type name (e.g. "Scaling", "Dense").
     * @return Pointer to the layer, or nullptr if none exists.
     */
    Layer* get_first(const string& layer_name);

    /**
     * @brief Returns the first layer of a given type.
     * @param type Layer type enum.
     * @return Pointer to the layer, or nullptr if none exists.
     */
    Layer* get_first(LayerType type);

    /**
     * @brief Returns the first layer of a given type by name (const overload).
     * @param layer_name Layer type name.
     * @return Const pointer to the layer, or nullptr.
     */
    const Layer* get_first(const string& layer_name) const;

    /**
     * @brief Returns the first layer of a given type (const overload).
     * @param type Layer type enum.
     * @return Const pointer to the layer, or nullptr.
     */
    const Layer* get_first(LayerType type) const;

    /**
     * @brief Resizes the layer stack.
     * @param new_layers_number New layer count; existing entries are preserved.
     */
    void set_layers_number(const Index new_layers_number) { layers.resize(new_layers_number); layer_input_indices.resize(new_layers_number); }

    /**
     * @brief Replaces the entire layer-input wiring.
     * @param new_layer_input_indices Outer vector indexed by destination layer.
     */
    void set_layer_input_indices(const vector<vector<Index>>& new_layer_input_indices) { layer_input_indices = new_layer_input_indices; }

    /**
     * @brief Replaces the input wiring of a single layer by index.
     * @param layer_index Index of the destination layer.
     * @param new_input_indices Indices of source layers feeding it.
     */
    void set_layer_input_indices(const Index layer_index, const vector<Index>& new_input_indices) { layer_input_indices[layer_index] = new_input_indices; }

    /**
     * @brief Replaces the input wiring of a single layer by name.
     * @param layer_name Name of the destination layer.
     * @param input_layer_names Names of layers feeding it.
     */
    void set_layer_input_indices(const string& layer_name, const vector<string>& input_layer_names);

    /**
     * @brief Replaces the input wiring of a single layer by name (initializer-list overload).
     * @param layer_name Name of the destination layer.
     * @param input_layer_names Names of layers feeding it.
     */
    void set_layer_input_indices(const string& layer_name, initializer_list<string> input_layer_names);

    /**
     * @brief Sets a single source layer as input to a destination layer.
     * @param layer_name Name of the destination layer.
     * @param input_layer_name Name of the source layer.
     */
    void set_layer_input_indices(const string& layer_name, const string& input_layer_name);

    /**
     * @brief Replaces the input variables.
     * @param new_input_variables New input variables vector.
     */
    void set_input_variables(const vector<Variable>& new_input_variables) { input_variables = new_input_variables; }

    /**
     * @brief Replaces the output variables.
     * @param new_output_variables New output variables vector.
     */
    void set_output_variables(const vector<Variable>& new_output_variables) { output_variables = new_output_variables; }

    /**
     * @brief Sets the names of the input features.
     * @param new_input_names One entry per input variable.
     */
    void set_input_names(const vector<string>& new_input_names);

    /**
     * @brief Sets the names of the output features.
     * @param new_output_names One entry per output variable.
     */
    void set_output_names(const vector<string>& new_output_names);

    /**
     * @brief Sets the input shape of the network.
     *
     * Reshapes the first layer accordingly. Useful when reusing an architecture
     * for a dataset with different feature dimensions.
     *
     * @param new_input_shape New input shape.
     */
    void set_input_shape(const Shape& new_input_shape);

    /**
     * @brief Resets non-architectural state to defaults.
     */
    void set_default();

    /**
     * @brief Returns the number of layers.
     * @return Layer count.
     */
    Index get_layers_number() const { return ssize(layers); }

    /**
     * @brief Returns the number of layers of a given type by name.
     * @param layer_name Layer type name.
     * @return Number of matching layers.
     */
    Index get_layers_number(const string& layer_name) const;

    /**
     * @brief Returns the number of layers of a given type.
     * @param type Layer type enum.
     * @return Number of matching layers.
     */
    Index get_layers_number(LayerType type) const;

    /**
     * @brief Returns the index of the first trainable layer.
     * @return Layer index, or -1 if no trainable layer exists.
     */
    Index get_first_trainable_layer_index() const;

    /**
     * @brief Returns the index of the last trainable layer.
     * @return Layer index, or -1 if no trainable layer exists.
     */
    Index get_last_trainable_layer_index() const;

    /**
     * @brief Returns the total number of input features.
     * @return Inputs count.
     */
    Index get_inputs_number() const;

    /**
     * @brief Returns the total number of output features.
     * @return Outputs count.
     */
    Index get_outputs_number() const;

    /**
     * @brief Returns the network's input shape.
     * @return Shape of the input tensor expected by the first layer.
     */
    Shape get_input_shape() const;

    /**
     * @brief Returns the network's output shape.
     * @return Shape of the output tensor produced by the last layer.
     */
    Shape get_output_shape() const;

    /**
     * @brief Returns the activation function of the last layer.
     * @return Activation::Function enum value.
     */
    Activation::Function get_output_activation() const;

    /**
     * @brief Returns the total number of trainable parameters.
     * @return Parameter count summed across layers.
     */
    Index get_parameters_number() const;

    /**
     * @brief Returns the parameter count of every layer.
     * @return Vector of parameter counts, one per layer.
     */
    vector<Index> get_layer_parameter_numbers() const;

    /**
     * @brief Replaces all trainable parameters.
     * @param new_parameters Flat vector with as many entries as get_parameters_number().
     */
    void set_parameters(const VectorR& new_parameters);

    /**
     * @brief Initializes parameters with uniform random values.
     */
    void set_parameters_random();

    /**
     * @brief Initializes parameters with Glorot (Xavier) uniform values.
     */
    void set_parameters_glorot();

    /**
     * @brief Computes outputs from a list of input tensor views.
     * @param inputs Input tensor views indexed by input slot.
     * @return Network outputs as a matrix [samples x outputs].
     */
    MatrixR calculate_outputs(const vector<TensorView>& inputs);

    /**
     * @brief Computes outputs for tabular inputs.
     * @param inputs Input matrix [samples x features].
     * @return Network outputs as a matrix [samples x outputs].
     */
    MatrixR calculate_outputs(const MatrixR& inputs);

    /**
     * @brief Computes outputs for rank-3 inputs (e.g. sequence data).
     * @param inputs Rank-3 input tensor.
     * @return Network outputs as a matrix [samples x outputs].
     */
    MatrixR calculate_outputs(const Tensor3& inputs);

    /**
     * @brief Computes outputs for rank-4 inputs (e.g. images).
     * @param inputs Rank-4 input tensor.
     * @return Network outputs as a matrix [samples x outputs].
     */
    MatrixR calculate_outputs(const Tensor4& inputs);

    /**
     * @brief Computes outputs along a directional sweep of one input.
     *
     * Holds the other inputs at @p point and varies input @p direction across
     * @p points_number values in [@p minimum, @p maximum].
     *
     * @param direction Index of the input to sweep.
     * @param point Reference point for all other inputs.
     * @param minimum Minimum value for the swept input.
     * @param maximum Maximum value for the swept input.
     * @param points_number Number of evaluation points along the sweep.
     * @return Matrix [points_number x outputs] with the network response.
     */
    MatrixR calculate_directional_inputs(const Index direction, const VectorR& point, float minimum, float maximum, Index points_number = 101) const;

    /**
     * @brief Computes outputs from two rank-3 inputs (encoder/decoder pair).
     * @param inputs Rank-3 input tensor.
     * @param context Rank-3 context tensor.
     * @return Rank-3 output tensor.
     */
    Tensor3 calculate_outputs(const Tensor3& inputs, const Tensor3& context);

    /**
     * @brief Runs the network on a single image and returns the predicted class.
     * @param image_path Path to the image file.
     * @return Predicted class index.
     */
    Index calculate_image_output(const filesystem::path& image_path);

    /**
     * @brief Runs the network on a batch of strings and returns class outputs.
     * @param texts One string per sample.
     * @return Output matrix [samples x outputs].
     */
    MatrixR calculate_text_outputs(const Tensor<string, 1>& texts);

    /**
     * @brief Restores the network from a JSON document.
     * @param document Parsed JSON produced by to_JSON().
     */
    void from_JSON(const JsonDocument& document);

    /**
     * @brief Serializes the network to JSON.
     * @param writer JSON writer that receives the network tree.
     */
    void to_JSON(JsonWriter& writer) const;

    /**
     * @brief Saves the full network (architecture + parameters) to a JSON file.
     * @param file_name Destination path.
     */
    void save(const filesystem::path& file_name) const;

    /**
     * @brief Saves only the parameters in binary form.
     * @param file_name Destination path.
     */
    void save_parameters(const filesystem::path& file_name) const;

    /**
     * @brief Loads the full network (architecture + parameters) from a JSON file.
     * @param file_name Source path.
     */
    void load(const filesystem::path& file_name);

    /**
     * @brief Loads parameters from a binary file produced by save_parameters().
     *
     * The architecture must already match the saved network.
     *
     * @param file_name Source path.
     */
    void load_parameters_binary(const filesystem::path& file_name);

    /**
     * @brief Returns the names of every input and output feature.
     * @return Concatenated vector of input feature names followed by output feature names.
     */
    vector<string> get_names_string() const;

    /**
     * @brief Saves a tabular outputs tensor to a CSV file.
     * @param outputs Outputs matrix [samples x outputs].
     * @param file_name Destination path.
     */
    void save_outputs(MatrixR& outputs, const filesystem::path& file_name);

    /**
     * @brief Saves a rank-3 outputs tensor to a CSV file.
     * @param outputs Rank-3 outputs tensor.
     * @param file_name Destination path.
     */
    void save_outputs(Tensor3& outputs, const filesystem::path& file_name);

    /**
     * @brief Runs the forward pass and writes intermediate activations into @p forward.
     * @param inputs Input tensor views.
     * @param forward Forward-propagation workspace; populated in place.
     * @param is_training Whether to enable training-time behavior (e.g. dropout).
     */
    void forward_propagate(const vector<TensorView>& inputs, ForwardPropagation& forward, bool is_training = false) const;

    /**
     * @brief Runs the forward pass with explicitly supplied parameters.
     *
     * Used by line-search style optimizers that probe along a direction.
     *
     * @param inputs Input tensor views.
     * @param parameters Parameter vector used for this evaluation.
     * @param forward Forward-propagation workspace; populated in place.
     */
    void forward_propagate(const vector<TensorView>& inputs, const VectorR& parameters, ForwardPropagation& forward);

#ifdef OPENNN_HAS_CUDA

public:

    /// Casts the parameter buffer to BF16 for mixed-precision training.
    void cast_parameters_to_bf16();

    /// Copies parameters from host to device memory.
    void copy_parameters_device();
    /// Copies parameters from device to host memory.
    void copy_parameters_host();
    /// Re-binds device-side parameter views after a parameter buffer reallocation.
    void link_parameters();

    /// Copies persistent layer state from host to device memory.
    void copy_states_device();
    /// Copies persistent layer state from device to host memory.
    void copy_states_host();
    /// Re-binds device-side state views after a state buffer reallocation.
    void link_states();

private:

    /**
     * @brief Device-side forward pass for inference.
     * @param inputs Input tensor views.
     * @param forward Forward-propagation workspace.
     * @return Network outputs as a matrix [samples x outputs].
     */
    MatrixR calculate_outputs_device(const vector<TensorView>& inputs, ForwardPropagation& forward);

#endif

public:

    /**
     * @brief Returns a label for every layer (name + key hyperparameters).
     * @return Layer labels suitable for printing.
     */
    vector<string> get_layer_labels() const;

private:

    /**
     * @brief Throws if the layer type is not allowed in the current architecture.
     * @param type Layer type to validate.
     */
    void validate_type(LayerType type) const;

    /**
     * @brief Gathers a per-layer shape vector via the supplied accessor.
     * @tparam Fn Callable taking a const Layer& and returning vector<Shape>.
     * @param fn Accessor function.
     * @return Outer vector indexed by layer, inner by shape group.
     */
    template<typename Fn>
    vector<vector<Shape>> collect_layer_shapes(Fn fn) const
    {
        const Index n = get_layers_number();
        vector<vector<Shape>> out(n);
        for (Index i = 0; i < n; ++i) out[i] = fn(*layers[i]);
        return out;
    }

protected:

    /// Network identifier; used as a JSON tag.
    string name = "neural_network";

    /// Description of every input feature (role, scaler, type).
    vector<Variable> input_variables;
    /// Description of every output feature (role, scaler, type).
    vector<Variable> output_variables;

    /// Owned layers in execution order.
    vector<unique_ptr<Layer>> layers;

    /// Per-destination-layer list of source-layer indices.
    vector<vector<Index>> layer_input_indices;

    /// Flat parameter buffer shared by all layers.
    Buffer parameters;
    /// BF16 mirror of @p parameters for mixed-precision CUDA training.
    Buffer parameters_bf16{Device::CUDA};
    /// Per-layer per-parameter-group views into @p parameters.
    vector<vector<vector<TensorView>>> parameter_views;

    /// Flat persistent-state buffer shared by stateful layers (e.g. Recurrent).
    Buffer states;

    /// Resolved device/precision configuration applied at compile time.
    Configuration::Resolved config;
};

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
