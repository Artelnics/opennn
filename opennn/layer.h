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
#include "operators.h"
#include "random_utilities.h"
#include "string_utilities.h"
#include "forward_propagation.h"
#include "back_propagation.h"

namespace opennn
{

struct Operator;

enum class LayerType
{
    Activation,
    Addition,
    Bounding,
    Convolutional,
    ConvolutionalRelu,
    Dense,
    DenseRelu,
    Detection,
    Embedding,
    Flatten,
    LongShortTermMemory,
    MultiHeadAttention,
    Normalization3d,
    NonMaxSuppression,
    Pooling,
    Pooling3d,
    Recurrent,
    Scaling,
    Unscaling
};

[[nodiscard]] inline const EnumMap<LayerType>& layer_type_map()
{
    static const vector<pair<LayerType, string>> entries = {
        {LayerType::Activation,         "Activation"},
        {LayerType::Addition,           "Addition"},
        {LayerType::Bounding,           "Bounding"},
        {LayerType::Convolutional,      "Convolutional"},
        {LayerType::ConvolutionalRelu,  "ConvolutionalRelu"},
        {LayerType::Dense,              "Dense"},
        {LayerType::DenseRelu,          "DenseRelu"},
        {LayerType::Detection,          "Detection"},
        {LayerType::Embedding,          "Embedding"},
        {LayerType::Flatten,            "Flatten"},
        {LayerType::LongShortTermMemory, "LongShortTermMemory"},
        {LayerType::MultiHeadAttention, "MultiHeadAttention"},
        {LayerType::Normalization3d,    "Normalization3d"},
        {LayerType::NonMaxSuppression,  "NonMaxSuppression"},
        {LayerType::Pooling,            "Pooling"},
        {LayerType::Pooling3d,          "Pooling3d"},
        {LayerType::Recurrent,          "Recurrent"},
        {LayerType::Scaling,            "Scaling"},
        {LayerType::Unscaling,          "Unscaling"}
    };
    static const EnumMap<LayerType> map{entries};
    return map;
}

[[nodiscard]] inline const string& layer_type_to_string(LayerType type)
{
    return layer_type_map().to_string(type);
}

[[nodiscard]] inline LayerType string_to_layer_type(const string& name)
{
    return layer_type_map().from_string(name);
}

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

class Layer
{

public:

    virtual ~Layer() = default;

    [[nodiscard]] const string& get_label() const { return label; }

    [[nodiscard]] const string& get_name() const { return layer_type_to_string(layer_type); }

    [[nodiscard]] LayerType get_type() const { return layer_type; }

    virtual void set_input_shape(const Shape&);
    virtual void set_output_shape(const Shape&);

    void set_label(string new_label) { label = move(new_label); }

    [[nodiscard]] Index get_parameters_number() const;
    [[nodiscard]] const vector<Operator*>& get_operators() const { return operators; }
    [[nodiscard]] virtual vector<TensorSpec> get_parameter_specs() const;
    [[nodiscard]] virtual vector<TensorSpec> get_state_specs()     const;
    [[nodiscard]] virtual vector<TensorSpec> get_forward_specs(Index batch_size) const
    {
        return {{Shape{batch_size}.append(get_output_shape()), compute_dtype}};
    }
    [[nodiscard]] virtual vector<TensorSpec> get_backward_specs(Index batch_size) const
    {
        if (!is_trainable) return {};
        return {{Shape{batch_size}.append(get_input_shape()), compute_dtype}};
    }

    [[nodiscard]] virtual Shape get_input_shape() const { return input_shape; }

    [[nodiscard]] virtual Shape get_output_shape() const = 0;

    [[nodiscard]] virtual ActivationFunction get_output_activation() const { return ActivationFunction::Identity; }

    [[nodiscard]] Index get_inputs_number() const { return get_input_shape().size(); }

    [[nodiscard]] Index get_outputs_number() const { return get_output_shape().size(); }
    
    virtual void forward_propagate(ForwardPropagation& fp, size_t layer, bool is_training) noexcept
    {
        for (Operator* op : get_operators())
            op->forward_propagate(fp, layer, is_training);
    }

    virtual void back_propagate(ForwardPropagation& fp, BackPropagation& bp, size_t i) const noexcept
    {
        for (Operator* op : views::reverse(get_operators()))
            op->back_propagate(fp, bp, i);
    }

    virtual void from_JSON(const JsonDocument& document);

    virtual void read_JSON_body(const Json*) {}

    virtual void load_state_from_JSON(const JsonDocument& document);

    virtual void to_JSON(JsonWriter& writer) const;

    virtual void write_JSON_body(JsonWriter&) const {}

    [[nodiscard]] virtual string write_expression(const vector<string>& /*input_names*/,
                                    const vector<string>& /*output_names*/) const { return {}; }

    virtual void print() const {}

    [[nodiscard]] bool get_is_trainable() const { return is_trainable; }

    [[nodiscard]] Type get_compute_dtype() const { return compute_dtype; }

    void set_compute_dtype(Type new_compute_dtype)
    {
        compute_dtype = new_compute_dtype;
        on_compute_dtype_changed();
    }

    virtual void on_compute_dtype_changed() {}

    virtual float* link_states(float* pointer);

    float* link_gradients(float* pointer, vector<TensorView>& gradient_views);

    vector<TensorView>& get_parameter_views() { return parameters; }
    const vector<TensorView>& get_parameter_views() const { return parameters; }

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
