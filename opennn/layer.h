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

enum class LayerType
{
    Addition,
    Bounding,
    Convolutional,
    Dense,
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

inline const EnumMap<LayerType>& layer_type_map()
{
    static const vector<pair<LayerType, string>> entries = {
        {LayerType::Addition,           "Addition"},
        {LayerType::Bounding,           "Bounding"},
        {LayerType::Convolutional,      "Convolutional"},
        {LayerType::Dense,              "Dense"},
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

inline const string& layer_type_to_string(LayerType type)
{
    return layer_type_map().to_string(type);
}

inline LayerType string_to_layer_type(const string& name)
{
    return layer_type_map().from_string(name);
}

#ifdef _MSC_VER
#define FORCE_INLINE __forceinline
#elif defined(__GNUC__) || defined(__clang__)
#define FORCE_INLINE __attribute__((always_inline)) inline
#else
#define FORCE_INLINE inline
#endif

class Layer
{

public:

    virtual ~Layer() = default;

    const string& get_label() const { return label; }

    const string& get_name() const { return name; }

    LayerType get_type() const { return layer_type; }

    virtual void set_input_shape(const Shape&);
    virtual void set_output_shape(const Shape&);

    void set_label(string new_label) { label = std::move(new_label); }

    virtual void set_parameters_random();

    virtual void set_parameters_glorot();

    Index get_parameters_number() const;

    virtual vector<pair<Shape, Type>> get_parameter_specs() const { return {}; }
    virtual vector<pair<Shape, Type>> get_state_specs()     const { return {}; }
    virtual vector<pair<Shape, Type>> get_forward_specs(Index) const { return {}; }
    virtual vector<pair<Shape, Type>> get_backward_specs(Index) const { return {}; }

    vector<Shape> get_parameter_shapes() const
    {
        vector<Shape> shapes;
        for (const auto& [shape, _] : get_parameter_specs()) shapes.push_back(shape);
        return shapes;
    }

    vector<Shape> get_state_shapes() const
    {
        vector<Shape> shapes;
        for (const auto& [shape, _] : get_state_specs()) shapes.push_back(shape);
        return shapes;
    }

    vector<Shape> get_forward_shapes(Index batch_size) const
    {
        vector<Shape> shapes;
        for (const auto& [shape, _] : get_forward_specs(batch_size)) shapes.push_back(shape);
        return shapes;
    }

    vector<Shape> get_backward_shapes(Index batch_size) const
    {
        vector<Shape> shapes;
        for (const auto& [shape, _] : get_backward_specs(batch_size)) shapes.push_back(shape);
        return shapes;
    }

    vector<Type> get_parameter_dtypes() const
    {
        vector<Type> dtypes;
        for (const auto& [_, type] : get_parameter_specs()) dtypes.push_back(type);
        return dtypes;
    }

    vector<Type> get_state_dtypes() const
    {
        vector<Type> dtypes;
        for (const auto& [_, type] : get_state_specs()) dtypes.push_back(type);
        return dtypes;
    }

    vector<Type> get_forward_dtypes(Index batch_size) const
    {
        vector<Type> dtypes;
        for (const auto& [_, type] : get_forward_specs(batch_size)) dtypes.push_back(type);
        return dtypes;
    }

    vector<Type> get_backward_dtypes(Index batch_size) const
    {
        vector<Type> dtypes;
        for (const auto& [_, type] : get_backward_specs(batch_size)) dtypes.push_back(type);
        return dtypes;
    }

    virtual Shape get_input_shape() const = 0;

    virtual Shape get_output_shape() const = 0;

    virtual Activation::Function get_output_activation() const { return Activation::Function::Identity; }

    Index get_inputs_number() const { return get_input_shape().size(); }

    Index get_outputs_number() const { return get_output_shape().size(); }

    // Forward propagation

    virtual void forward_propagate(ForwardPropagation&, size_t, bool) noexcept = 0;

    // Back propagation

    virtual void back_propagate(ForwardPropagation&, BackPropagation&, size_t) const noexcept
    {
        throw runtime_error("back_propagate not implemented for layer type: " + name);
    }

    virtual void from_JSON(const JsonDocument&) {}

    virtual void load_state_from_JSON(const JsonDocument&) {}

    virtual void to_JSON(JsonWriter&) const {}

    virtual void print() const {}

    vector<string> get_default_feature_names() const;

    vector<string> get_default_output_names() const;

    bool get_is_trainable() const { return is_trainable; }

    Type get_activation_dtype() const { return activation_dtype; }

    void set_activation_dtype(Type new_activation_dtype) { activation_dtype = new_activation_dtype; }

    virtual float* link_parameters(float* pointer);

    virtual float* link_states(float* pointer);

    vector<TensorView>& get_parameter_views() { return parameters; }
    const vector<TensorView>& get_parameter_views() const { return parameters; }

    vector<TensorView>& get_state_views() { return states; }
    const vector<TensorView>& get_state_views() const { return states; }

protected:

    Layer() = default;

    string label = "my_layer";

    string name = "layer";

    LayerType layer_type = LayerType::Dense;

    bool is_trainable = true;

    bool is_first_layer = false;

    Type activation_dtype = Type::FP32;

    vector<TensorView> parameters;
    vector<TensorView> states;

    Tensor2 empty_2;

    void add_gradients(const vector<TensorView>&) const;

    float* link_views(float* pointer,
                      const vector<Shape>& shapes,
                      vector<TensorView>& views,
                      const char* tag) const;
};

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
