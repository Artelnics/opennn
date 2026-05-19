//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   L A Y E R   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

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

/// @brief Identifier of every concrete layer subclass shipped with OpenNN.
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

/// @brief Returns the bidirectional mapping between LayerType values and their string names.
[[nodiscard]] inline const EnumMap<LayerType>& layer_type_map()
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

/// @brief Returns the string name associated with the given LayerType.
[[nodiscard]] inline const string& layer_type_to_string(LayerType type)
{
    return layer_type_map().to_string(type);
}

/// @brief Returns the LayerType corresponding to the given string name.
[[nodiscard]] inline LayerType string_to_layer_type(const string& name)
{
    return layer_type_map().from_string(name);
}

/// @brief Throws if @p shape rank is not one of @p allowed.
/// @param shape Shape to validate (empty shapes are accepted).
/// @param allowed Acceptable rank values.
/// @param layer Layer name used in the error message.
/// @param what Description of the tensor being checked (e.g. "input").
inline void check_rank(const Shape& shape, initializer_list<int> allowed,
                       const char* layer, const char* what)
{
    if (shape.empty()) return;
    for (int r : allowed) if (int(shape.rank) == r) return;

    string allowed_str;
    auto it = allowed.begin();
    while (it != allowed.end())
    {
        if (!allowed_str.empty())
            allowed_str += (it + 1 == allowed.end()) ? " or " : ", ";
        allowed_str += to_string(*it);
        ++it;
    }

    throw runtime_error(format("{} layer supports {} rank {} (got {}).",
                               layer, what, allowed_str, shape.rank));
}

/// @brief Abstract base class for all OpenNN layers; orchestrates operators and shape propagation.
class Layer
{

public:

    virtual ~Layer() = default;

    [[nodiscard]] const string& get_label() const { return label; }

    [[nodiscard]] const string& get_name() const { return layer_type_to_string(layer_type); }

    [[nodiscard]] LayerType get_type() const { return layer_type; }

    /// @brief Sets the input shape; subclasses override to derive dependent dimensions.
    virtual void set_input_shape(const Shape&);

    /// @brief Sets the output shape; subclasses override when the output is user-configurable.
    virtual void set_output_shape(const Shape&);

    void set_label(string new_label) { label = move(new_label); }

    /// @brief Returns the total number of trainable parameters owned by this layer.
    [[nodiscard]] Index get_parameters_number() const;

    [[nodiscard]] const vector<Operator*>& get_operators() const { return operators; }

    /// @brief Returns the tensor specs of trainable parameters; subclasses override.
    [[nodiscard]] virtual vector<TensorSpec> get_parameter_specs() const;

    /// @brief Returns the tensor specs of persistent state (e.g. running mean/variance).
    [[nodiscard]] virtual vector<TensorSpec> get_state_specs()     const;

    /// @brief Returns the tensor specs of the forward workspace; defaults to a single output tensor.
    /// @param batch_size Batch size used to size each tensor.
    [[nodiscard]] virtual vector<TensorSpec> get_forward_specs(Index batch_size) const
    {
        return {{Shape{batch_size}.append(get_output_shape()), compute_dtype}};
    }

    /// @brief Returns the tensor specs of the backward workspace; empty for non-trainable layers.
    /// @param batch_size Batch size used to size each tensor.
    [[nodiscard]] virtual vector<TensorSpec> get_backward_specs(Index batch_size) const
    {
        if (!is_trainable) return {};
        return {{Shape{batch_size}.append(get_input_shape()), compute_dtype}};
    }

    /// @brief Returns the input shape stored by the layer.
    [[nodiscard]] virtual Shape get_input_shape() const { return input_shape; }

    /// @brief Returns the output shape; subclasses must implement this to expose their geometry.
    [[nodiscard]] virtual Shape get_output_shape() const = 0;

    /// @brief Returns the layer's output activation (Identity for most layers; overridden by Dense/Bounding).
    [[nodiscard]] virtual ActivationOp::Function get_output_activation() const { return ActivationOp::Function::Identity; }

    [[nodiscard]] Index get_inputs_number() const { return get_input_shape().size(); }

    [[nodiscard]] Index get_outputs_number() const { return get_output_shape().size(); }

    /// @brief Runs the forward pass by chaining the layer's operators in order.
    /// @param fp Forward propagation workspace.
    /// @param layer Index of this layer in the workspace.
    /// @param is_training If true, enables training-only behavior in child operators.
    virtual void forward_propagate(ForwardPropagation& fp, size_t layer, bool is_training) noexcept
    {
        for (Operator* op : get_operators())
            op->forward_propagate(fp, layer, is_training);
    }

    /// @brief Runs the backward pass by chaining the layer's operators in reverse order.
    /// @param fp Forward propagation workspace (read-only here).
    /// @param bp Back propagation workspace receiving gradients.
    /// @param i Index of this layer in the workspace.
    virtual void back_propagate(ForwardPropagation& fp, BackPropagation& bp, size_t i) const noexcept
    {
        for (Operator* op : views::reverse(get_operators()))
            op->back_propagate(fp, bp, i);
    }

    /// @brief Restores layer configuration and parameters from a JSON document.
    virtual void from_JSON(const JsonDocument& document);

    /// @brief Subclass hook reading the body section of the layer's JSON node.
    virtual void read_JSON_body(const Json*) {}

    /// @brief Restores persistent state (e.g. running statistics) from a JSON document.
    virtual void load_state_from_JSON(const JsonDocument& document);

    /// @brief Serializes layer configuration and parameters to a JSON writer.
    virtual void to_JSON(JsonWriter& writer) const;

    /// @brief Subclass hook writing the body section of the layer's JSON node.
    virtual void write_JSON_body(JsonWriter&) const {}

    /// @brief Returns a human-readable mathematical expression for this layer (empty by default).
    /// @param input_names Names assigned to the inputs.
    /// @param output_names Names assigned to the outputs.
    [[nodiscard]] virtual string write_expression(const vector<string>& /*input_names*/,
                                    const vector<string>& /*output_names*/) const { return {}; }

    /// @brief Prints a human-readable summary of the layer to standard output.
    virtual void print() const {}

    [[nodiscard]] bool get_is_trainable() const { return is_trainable; }

    [[nodiscard]] Type get_compute_dtype() const { return compute_dtype; }

    /// @brief Sets the compute dtype and notifies subclasses via on_compute_dtype_changed().
    void set_compute_dtype(Type new_compute_dtype)
    {
        compute_dtype = new_compute_dtype;
        on_compute_dtype_changed();
    }

    /// @brief Subclass hook invoked when the compute dtype changes; default is no-op.
    virtual void on_compute_dtype_changed() {}

    /// @brief Binds the persistent-state region of the shared buffer to operator views.
    /// @param pointer Start of the layer's slice in the shared state buffer.
    /// @return Pointer advanced past this layer's state region.
    virtual float* link_states(float* pointer);

    /// @brief Binds the gradient slice of the shared buffer to operator gradient views.
    /// @param pointer Start of the layer's slice in the shared gradient buffer.
    /// @param gradient_views Output list of views the optimizer iterates over.
    /// @return Pointer advanced past this layer's gradient region.
    float* link_gradients(float* pointer, vector<TensorView>& gradient_views);

    vector<TensorView>& get_parameter_views() { return parameters; }
    const vector<TensorView>& get_parameter_views() const { return parameters; }

    /// @brief Re-binds operator parameter views after the parameter buffer has been resized or moved.
    void redistribute_parameters_to_operators();

protected:

    Layer() = default;

    Layer(LayerType t, bool trainable = true)
        : layer_type(t), is_trainable(trainable) {}

    enum Forward {Input, Output};
    enum Backward {OutputDelta, InputDelta};

    string label = "my_layer";

    LayerType layer_type = LayerType::Dense;

    bool is_trainable = true;

    Shape input_shape;

    Type compute_dtype = Type::FP32;

    vector<TensorView> parameters;
    vector<TensorView> states;

    vector<Operator*> operators;

    float* link_views_to_operators(
        vector<TensorView>& views, float* pointer,
        vector<TensorSpec> (Operator::*specs_fn)() const,
        void (Operator::*link_fn)(span<const TensorView>));

};

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
