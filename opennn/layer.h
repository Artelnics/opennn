//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   L A Y E R   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

/**
 * @file layer.h
 * @brief Declares the Layer abstract base class and the LayerType enumeration.
 *
 * Every layer type in OpenNN (Dense, Convolutional, Recurrent, Embedding, ...)
 * derives from Layer and implements the shape, parameter and propagation
 * interface defined here. Layers are stateless with respect to batch data:
 * forward and backward intermediates are stored in ForwardPropagation and
 * BackPropagation objects owned by the caller.
 */

#pragma once

#include "tensor_utilities.h"
#include "math_utilities.h"
#include "random_utilities.h"
#include "string_utilities.h"
#include "forward_propagation.h"
#include "back_propagation.h"

namespace opennn
{

struct Operator;

/**
 * @enum LayerType
 * @brief Identifier for every concrete Layer subclass supported by OpenNN.
 *
 * Used for serialization, runtime type queries and the registry-based
 * factory that rebuilds layers from XML/JSON.
 */
enum class LayerType
{
    Addition,
    Bounding,
    Convolutional,
    ConvolutionalRelu,
    Dense,
    DenseRelu,
    Embedding,
    Flatten,
    MultiHeadAttention,
    Normalization3d,
    Pooling,
    Pooling3d,
    Recurrent,
    Scaling,
    Unscaling
};

/**
 * @brief Returns the singleton string<->enum mapping for LayerType values.
 * @return Reference to a process-wide EnumMap initialized on first call.
 */
inline const EnumMap<LayerType>& layer_type_map()
{
    static const vector<pair<LayerType, string>> entries = {
        {LayerType::Addition,           "Addition"},
        {LayerType::Bounding,           "Bounding"},
        {LayerType::Convolutional,      "Convolutional"},
        {LayerType::ConvolutionalRelu,  "ConvolutionalRelu"},
        {LayerType::Dense,              "Dense"},
        {LayerType::DenseRelu,          "DenseRelu"},
        {LayerType::Embedding,          "Embedding"},
        {LayerType::Flatten,            "Flatten"},
        {LayerType::MultiHeadAttention, "MultiHeadAttention"},
        {LayerType::Normalization3d,    "Normalization3d"},
        {LayerType::Pooling,            "Pooling"},
        {LayerType::Pooling3d,          "Pooling3d"},
        {LayerType::Recurrent,          "Recurrent"},
        {LayerType::Scaling,            "Scaling"},
        {LayerType::Unscaling,          "Unscaling"}
    };
    static const EnumMap<LayerType> map{entries};
    return map;
}

/**
 * @brief Converts a LayerType to its canonical string name.
 * @param type Layer type value.
 * @return Reference to the canonical string (e.g. "Dense", "Convolutional").
 */
inline const string& layer_type_to_string(LayerType type)
{
    return layer_type_map().to_string(type);
}

/**
 * @brief Parses a LayerType from its canonical string name.
 * @param name String to parse (case sensitive, must match a canonical name).
 * @return Matching LayerType; throws if the string is unrecognized.
 */
inline LayerType string_to_layer_type(const string& name)
{
    return layer_type_map().from_string(name);
}

/**
 * @def FORCE_INLINE
 * @brief Compiler-specific always-inline hint.
 *
 * Resolves to `__forceinline` on MSVC, `__attribute__((always_inline)) inline`
 * on GCC/Clang, and plain `inline` on other compilers.
 */
#ifdef _MSC_VER
#define FORCE_INLINE __forceinline
#elif defined(__GNUC__) || defined(__clang__)
#define FORCE_INLINE __attribute__((always_inline)) inline
#else
#define FORCE_INLINE inline
#endif

/**
 * @brief Extracts the shape component from a list of (Shape, Type) specs.
 * @param specs Vector of buffer specifications.
 * @return Vector of the shapes, in the same order as @p specs.
 */
inline vector<Shape> spec_shapes(const vector<pair<Shape, Type>>& specs)
{
    vector<Shape> result;
    result.reserve(specs.size());
    for (const auto& [shape, _] : specs) result.push_back(shape);
    return result;
}

/**
 * @brief Extracts the dtype component from a list of (Shape, Type) specs.
 * @param specs Vector of buffer specifications.
 * @return Vector of the dtypes, in the same order as @p specs.
 */
inline vector<Type> spec_dtypes(const vector<pair<Shape, Type>>& specs)
{
    vector<Type> result;
    result.reserve(specs.size());
    for (const auto& [_, type] : specs) result.push_back(type);
    return result;
}

/**
 * @class Layer
 * @brief Abstract base class for every layer in an OpenNN NeuralNetwork.
 *
 * A Layer declares:
 * - Its parameter, state, forward and backward buffer specifications, used
 *   by the network to size the shared memory arenas before training.
 * - Its input and output shapes (per-sample, batch dimension is added by
 *   the caller).
 * - Its forward and backward propagation routines, which read from and
 *   write into ForwardPropagation / BackPropagation views provided by the
 *   caller.
 * - JSON serialization hooks so the layer can be persisted as part of a
 *   NeuralNetwork file.
 *
 * Layers do not own batch-dependent buffers; they only own their parameter
 * and state TensorViews into externally allocated memory.
 */
class Layer
{

public:

    /** @brief Virtual destructor; subclasses are owned via unique_ptr<Layer>. */
    virtual ~Layer() = default;

    /**
     * @brief Returns the user-assigned label of this layer.
     * @return Reference to the label string (default "my_layer").
     */
    const string& get_label() const { return label; }

    /**
     * @brief Returns the canonical type name of this layer.
     * @return Reference to the name string (e.g. "Dense", "Convolutional").
     */
    const string& get_name() const { return name; }

    /**
     * @brief Returns the LayerType enumerator for this layer.
     * @return Layer type tag set by the concrete subclass.
     */
    LayerType get_type() const { return layer_type; }

    /**
     * @brief Sets the per-sample input shape of this layer.
     *
     * The argument is the new input shape, excluding the batch dimension.
     * Subclasses override this to also resize the output shape and any
     * shape-dependent parameters.
     */
    virtual void set_input_shape(const Shape&);

    /**
     * @brief Sets the per-sample output shape of this layer.
     *
     * The argument is the new output shape, excluding the batch dimension.
     * Subclasses override this to also resize parameters whose dimensions
     * depend on the output shape.
     */
    virtual void set_output_shape(const Shape&);

    /**
     * @brief Sets the human-readable label of this layer.
     * @param new_label Label moved into the layer.
     */
    void set_label(string new_label) { label = move(new_label); }

    /**
     * @brief Total number of trainable parameters in this layer.
     * @return Sum of the sizes of all parameter TensorViews.
     */
    Index get_parameters_number() const;

    /**
     * @brief Returns the operators that compose this layer, if any.
     * @return Vector of non-owning Operator pointers; empty for leaf layers.
     *
     * Composite layers (e.g. those mixing convolution and activation) override
     * this so that the base forward_propagate() can iterate through them.
     */
    virtual vector<Operator*> get_operators() { return {}; }

    /**
     * @brief Specifications of the trainable parameter tensors owned by this layer.
     * @return Vector of (Shape, Type) pairs, one per parameter tensor.
     */
    virtual vector<pair<Shape, Type>> get_parameter_specs() const;

    /**
     * @brief Specifications of the persistent state tensors of this layer.
     * @return Vector of (Shape, Type) pairs, one per state tensor (e.g. running
     *         statistics for batch normalization).
     */
    virtual vector<pair<Shape, Type>> get_state_specs() const;

    /**
     * @brief Specifications of the forward intermediate buffers for one batch.
     * @param batch_size Number of samples in the batch.
     * @return Vector of (Shape, Type) pairs. The last entry is the layer
     *         output, wired to downstream layers.
     */
    virtual vector<pair<Shape, Type>> get_forward_specs(Index batch_size) const
    {
        return {{Shape{batch_size}.append(get_output_shape()), compute_dtype}};
    }

    /**
     * @brief Specifications of the backward intermediate buffers for one batch.
     * @param batch_size Number of samples in the batch.
     * @return Vector of (Shape, Type) pairs. Empty for non-trainable layers.
     */
    virtual vector<pair<Shape, Type>> get_backward_specs(Index batch_size) const
    {
        if (!is_trainable) return {};
        return {{Shape{batch_size}.append(get_input_shape()), compute_dtype}};
    }

    /** @brief Shape-only view of get_parameter_specs(). */
    vector<Shape> get_parameter_shapes()        const { return spec_shapes(get_parameter_specs()); }
    /** @brief Shape-only view of get_state_specs(). */
    vector<Shape> get_state_shapes()            const { return spec_shapes(get_state_specs()); }
    /** @brief Shape-only view of get_forward_specs() for batch size @p b. */
    vector<Shape> get_forward_shapes(Index b)   const { return spec_shapes(get_forward_specs(b)); }
    /** @brief Shape-only view of get_backward_specs() for batch size @p b. */
    vector<Shape> get_backward_shapes(Index b)  const { return spec_shapes(get_backward_specs(b)); }

    /** @brief Dtype-only view of get_parameter_specs(). */
    vector<Type>  get_parameter_dtypes()        const { return spec_dtypes(get_parameter_specs()); }
    /** @brief Dtype-only view of get_forward_specs() for batch size @p b. */
    vector<Type>  get_forward_dtypes(Index b)   const { return spec_dtypes(get_forward_specs(b)); }
    /** @brief Dtype-only view of get_backward_specs() for batch size @p b. */
    vector<Type>  get_backward_dtypes(Index b)  const { return spec_dtypes(get_backward_specs(b)); }

    /**
     * @brief Per-sample input shape of the layer (excluding the batch dimension).
     * @return Shape required by forward_propagate() for a single sample.
     */
    virtual Shape get_input_shape() const = 0;

    /**
     * @brief Per-sample output shape of the layer (excluding the batch dimension).
     * @return Shape produced by forward_propagate() for a single sample.
     */
    virtual Shape get_output_shape() const = 0;

    /**
     * @brief Activation function fused at the end of this layer, if any.
     * @return Activation::Function::Identity by default; layers like
     *         DenseRelu / ConvolutionalRelu override this.
     */
    virtual Activation::Function get_output_activation() const { return Activation::Function::Identity; }

    /**
     * @brief Total number of scalar inputs per sample (product of input dims).
     * @return Flat input size.
     */
    Index get_inputs_number() const { return get_input_shape().size(); }

    /**
     * @brief Total number of scalar outputs per sample (product of output dims).
     * @return Flat output size.
     */
    Index get_outputs_number() const { return get_output_shape().size(); }

    /**
     * @brief Forward pass: reads inputs from @p fp and writes outputs into @p fp.
     * @param fp ForwardPropagation buffer slice owned by the caller.
     * @param layer Index of this layer inside the network.
     * @param is_training True during training (enables dropout, BN updates, etc.).
     *
     * The default implementation runs every Operator returned by
     * get_operators() in order. Leaf layers override this directly.
     */
    virtual void forward_propagate(ForwardPropagation& fp, size_t layer, bool is_training) noexcept
    {
        for (Operator* op : get_operators())
            op->forward_propagate(fp, layer, is_training);
    }

    /**
     * @brief Backward pass: propagates gradients through this layer.
     *
     * Receives the forward intermediates, the BackPropagation buffer in which
     * to accumulate gradients, and this layer's index inside the network.
     * Throws by default; concrete trainable layers must override.
     */
    virtual void back_propagate(ForwardPropagation&, BackPropagation&, size_t) const noexcept
    {
        throw runtime_error("back_propagate not implemented for layer type: " + name);
    }

    /**
     * @brief Loads the layer configuration (hyperparameters) from JSON.
     * @param document Parsed JSON document for this layer.
     */
    virtual void from_JSON(const JsonDocument& document);

    /**
     * @brief Subclass hook for parsing the body of from_JSON().
     *
     * Receives the raw JSON node for this layer. Overridden by concrete
     * layers to read their specific fields after the base class has read
     * the common ones.
     */
    virtual void read_JSON_body(const Json*) {}

    /**
     * @brief Loads parameter and state tensors from a JSON document.
     * @param document Parsed JSON document containing tensor data.
     */
    virtual void load_state_from_JSON(const JsonDocument& document);

    /**
     * @brief Writes the layer configuration to JSON.
     * @param writer Streaming JSON writer.
     */
    virtual void to_JSON(JsonWriter& writer) const;

    /**
     * @brief Subclass hook for emitting the body of to_JSON().
     *
     * Receives the streaming JSON writer; overridden by concrete layers
     * to emit their subclass-specific fields.
     */
    virtual void write_JSON_body(JsonWriter&) const {}

    /** @brief Prints a human-readable summary of the layer to stdout. */
    virtual void print() const {}

    /**
     * @brief Whether this layer has trainable parameters.
     * @return True if gradients should flow through this layer during training.
     */
    bool get_is_trainable() const { return is_trainable; }

    /**
     * @brief Numerical type used for forward/backward computation.
     * @return Compute dtype (FP32 by default; FP16/BF16 in mixed precision).
     */
    Type get_compute_dtype() const { return compute_dtype; }

    /**
     * @brief Sets the compute dtype and triggers on_compute_dtype_changed().
     * @param new_compute_dtype Desired compute dtype.
     */
    void set_compute_dtype(Type new_compute_dtype)
    {
        compute_dtype = new_compute_dtype;
        on_compute_dtype_changed();
    }

    /**
     * @brief Hook invoked after set_compute_dtype() mutates the dtype.
     *
     * Subclasses override this to reconfigure cached dtype-dependent
     * resources (CUDA descriptors, cached scratch buffers, etc.).
     */
    virtual void on_compute_dtype_changed() {}

    /**
     * @brief Wires this layer's parameter TensorViews onto an external buffer.
     * @param pointer Start of the parameter region inside the network's
     *                parameter arena.
     * @return Pointer advanced past the bytes consumed by this layer.
     */
    virtual float* link_parameters(float* pointer);

    /**
     * @brief Wires this layer's state TensorViews onto an external buffer.
     * @param pointer Start of the state region inside the network's
     *                state arena.
     * @return Pointer advanced past the bytes consumed by this layer.
     */
    virtual float* link_states(float* pointer);

    /** @brief Mutable access to this layer's parameter TensorViews. */
    vector<TensorView>& get_parameter_views() { return parameters; }
    /** @brief Read-only access to this layer's parameter TensorViews. */
    const vector<TensorView>& get_parameter_views() const { return parameters; }

    /** @brief Mutable access to this layer's state TensorViews. */
    vector<TensorView>& get_state_views() { return states; }
    /** @brief Read-only access to this layer's state TensorViews. */
    const vector<TensorView>& get_state_views() const { return states; }

    /**
     * @brief Forwards the current parameter views down to each composing Operator.
     *
     * Composite layers slice their shared parameter buffer and hand sub-views
     * to their Operators so the operators can run independently.
     */
    void redistribute_parameters_to_operators()
    {
        distribute_to_operators(parameters, &Operator::link_parameters, &Operator::parameter_specs);
    }

    /**
     * @brief Forwards externally provided gradient views down to each Operator.
     * @param gradient_views Per-parameter gradient TensorViews aligned with
     *                       this layer's parameter specs.
     */
    void redistribute_parameter_gradients_to_operators(vector<TensorView>& gradient_views)
    {
        distribute_to_operators(gradient_views, &Operator::link_gradients, &Operator::parameter_specs);
    }

    /**
     * @brief Forwards the current state views down to each composing Operator.
     */
    void redistribute_states_to_operators()
    {
        distribute_to_operators(states, &Operator::link_states, &Operator::state_specs);
    }

protected:

    /** @brief Default constructor; only invoked by subclasses. */
    Layer() = default;

    /** @brief User-visible label for this layer instance (default "my_layer"). */
    string label = "my_layer";

    /** @brief Canonical type name set by the subclass (e.g. "dense"). */
    string name = "layer";

    /** @brief Layer type tag set by the subclass. */
    LayerType layer_type = LayerType::Dense;

    /** @brief True if the layer has parameters that participate in training. */
    bool is_trainable = true;

    /** @brief True if this layer is the network's input layer. */
    bool is_first_layer = false;

    /** @brief Numerical type used for forward and backward computation. */
    Type compute_dtype = Type::FP32;

    /** @brief Parameter TensorViews bound to the network's parameter arena. */
    vector<TensorView> parameters;
    /** @brief State TensorViews bound to the network's state arena. */
    vector<TensorView> states;

    /**
     * @brief Builds @p views over a contiguous float buffer using @p shapes.
     * @param pointer Start of the buffer to slice.
     * @param shapes Per-view shapes.
     * @param views Output vector of TensorViews, populated in order.
     * @param tag Diagnostic label used in error messages.
     * @return Pointer advanced past the bytes covered by @p shapes.
     */
    float* link_views(float* pointer,
                      const vector<Shape>& shapes,
                      vector<TensorView>& views,
                      const char* tag) const;

    /**
     * @brief Generic helper used by the redistribute_*_to_operators() routines.
     * @param views Source TensorViews to slice.
     * @param link Operator member function that receives each sub-view list.
     * @param specs Operator member function returning the per-Operator spec list.
     */
    void distribute_to_operators(
        vector<TensorView>& views,
        void (Operator::*link)(const vector<TensorView>&),
        vector<pair<Shape, Type>> (Operator::*specs)() const);

    /** @brief Sub-layers, when this layer is itself a composite. */
    vector<unique_ptr<Layer>> layers;

};

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
