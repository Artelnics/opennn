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

inline const string& layer_type_to_string(LayerType type)
{
    return layer_type_map().to_string(type);
}

inline LayerType string_to_layer_type(const string& name)
{
    return layer_type_map().from_string(name);
}

inline vector<Shape> spec_shapes(const vector<pair<Shape, Type>>& specs)
{
    vector<Shape> result;
    result.reserve(specs.size());
    for (const auto& [shape, _] : specs) result.push_back(shape);
    return result;
}

inline vector<Type> spec_dtypes(const vector<pair<Shape, Type>>& specs)
{
    vector<Type> result;
    result.reserve(specs.size());
    for (const auto& [_, type] : specs) result.push_back(type);
    return result;
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

    throw runtime_error(string(layer) + " layer supports " + what + " rank "
                        + allowed_str + " (got " + to_string(shape.rank) + ").");
}

class Layer
{

public:

    virtual ~Layer() = default;

    const string& get_label() const { return label; }

    const string& get_name() const { return name; }

    LayerType get_type() const { return layer_type; }

    virtual void set_input_shape(const Shape&);
    virtual void set_output_shape(const Shape&);

    void set_label(string new_label) { label = move(new_label); }

    Index get_parameters_number() const;
    const vector<Operator*>& get_operators() const { return operators; }
    virtual vector<pair<Shape, Type>> get_parameter_specs() const;
    virtual vector<pair<Shape, Type>> get_state_specs()     const;
    virtual vector<pair<Shape, Type>> get_forward_specs(Index batch_size) const
    {
        return {{Shape{batch_size}.append(get_output_shape()), compute_dtype}};
    }
    virtual vector<pair<Shape, Type>> get_backward_specs(Index batch_size) const
    {
        if (!is_trainable) return {};
        return {{Shape{batch_size}.append(get_input_shape()), compute_dtype}};
    }

    vector<Shape> get_parameter_shapes()        const { return spec_shapes(get_parameter_specs()); }
    vector<Shape> get_state_shapes()            const { return spec_shapes(get_state_specs()); }
    vector<Shape> get_forward_shapes(Index b)   const { return spec_shapes(get_forward_specs(b)); }
    vector<Shape> get_backward_shapes(Index b)  const { return spec_shapes(get_backward_specs(b)); }

    vector<Type>  get_forward_dtypes(Index b)   const { return spec_dtypes(get_forward_specs(b)); }
    vector<Type>  get_backward_dtypes(Index b)  const { return spec_dtypes(get_backward_specs(b)); }

    virtual Shape get_input_shape() const { return input_shape; }

    virtual Shape get_output_shape() const = 0;

    virtual ActivationOp::Function get_output_activation() const { return ActivationOp::Function::Identity; }

    Index get_inputs_number() const { return get_input_shape().size(); }

    Index get_outputs_number() const { return get_output_shape().size(); }
    
    virtual void forward_propagate(ForwardPropagation& fp, size_t layer, bool is_training) noexcept
    {
        for (Operator* op : get_operators())
            op->forward_propagate(fp, layer, is_training);
    }

    virtual void back_propagate(ForwardPropagation& fp, BackPropagation& bp, size_t i) const noexcept
    {
        const auto& ops = get_operators();
        for (auto it = ops.rbegin(); it != ops.rend(); ++it)
            (*it)->back_propagate(fp, bp, i);
    }

    virtual void from_JSON(const JsonDocument& document);

    virtual void read_JSON_body(const Json*) {}

    virtual void load_state_from_JSON(const JsonDocument& document);

    virtual void to_JSON(JsonWriter& writer) const;

    virtual void write_JSON_body(JsonWriter&) const {}

    virtual string write_expression(const vector<string>& /*input_names*/,
                                    const vector<string>& /*output_names*/) const { return string(); }

    virtual void print() const {}

    bool get_is_trainable() const { return is_trainable; }

    Type get_compute_dtype() const { return compute_dtype; }

    void set_compute_dtype(Type new_compute_dtype)
    {
        compute_dtype = new_compute_dtype;
        on_compute_dtype_changed();
    }

    virtual void on_compute_dtype_changed() {}

    virtual float* link_parameters(float* pointer);

    virtual float* link_states(float* pointer);

    vector<TensorView>& get_parameter_views() { return parameters; }
    const vector<TensorView>& get_parameter_views() const { return parameters; }

    vector<TensorView>& get_state_views() { return states; }
    const vector<TensorView>& get_state_views() const { return states; }

    void redistribute_parameters_to_operators()
    {
        distribute_to_operators(parameters, &Operator::link_parameters, &Operator::parameter_count);
    }

    void redistribute_parameter_gradients_to_operators(vector<TensorView>& gradient_views)
    {
        distribute_to_operators(gradient_views, &Operator::link_gradients, &Operator::parameter_count);
    }

protected:

    Layer() = default;

    Layer(string n, LayerType t, bool trainable = true)
        : name(move(n)), layer_type(t), is_trainable(trainable) {}

    enum Forward {Input, Output};
    enum Backward {OutputDelta, InputDelta};

    string label = "my_layer";

    string name = "layer";

    LayerType layer_type = LayerType::Dense;

    bool is_trainable = true;

    Shape input_shape;

    Type compute_dtype = Type::FP32;

    vector<TensorView> parameters;
    vector<TensorView> states;

    vector<Operator*> operators;

    void distribute_to_operators(
        vector<TensorView>& views,
        void (Operator::*link)(const vector<TensorView>&),
        size_t (Operator::*count)() const);

    float* link_views_to_operators(
        vector<TensorView>& views, float* pointer,
        vector<pair<Shape, Type>> (Operator::*specs_fn)() const,
        void (Operator::*link_fn)(const vector<TensorView>&));

    vector<unique_ptr<Layer>> layers;

};

inline vector<vector<Type>> collect_layer_dtypes(
    const vector<unique_ptr<Layer>>& layers,
    Index batch_size,
    bool is_gpu,
    vector<Type> (Layer::*getter)(Index) const)
{
    vector<vector<Type>> result(layers.size());
    
    for (size_t i = 0; i < layers.size(); ++i)
    {
        result[i] = (layers[i].get()->*getter)(batch_size);

        if (!is_gpu)
            std::fill(result[i].begin(), result[i].end(), Type::FP32);
    }
    return result;
}

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
