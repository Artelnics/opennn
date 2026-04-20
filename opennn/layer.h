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
    Addition3d,
    Addition4d,
    Bounding,
    Convolutional,
    Dense2d,
    Dense3d,
    Embedding,
    Flatten2d,
    Flatten3d,
    Flatten4d,
    MultiHeadAttention,
    Normalization3d,
    Pooling,
    Pooling3d,
    Recurrent,
    Scaling2d,
    Scaling3d,
    Scaling4d,
    Unscaling
};

inline const EnumMap<LayerType>& layer_type_map()
{
    static const vector<pair<LayerType, string>> entries = {
        {LayerType::Addition3d,         "Addition3d"},
        {LayerType::Addition4d,         "Addition4d"},
        {LayerType::Bounding,           "Bounding"},
        {LayerType::Convolutional,      "Convolutional"},
        {LayerType::Dense2d,            "Dense2d"},
        {LayerType::Dense3d,            "Dense3d"},
        {LayerType::Embedding,          "Embedding"},
        {LayerType::Flatten2d,          "Flatten2d"},
        {LayerType::Flatten3d,          "Flatten3d"},
        {LayerType::Flatten4d,          "Flatten4d"},
        {LayerType::MultiHeadAttention, "MultiHeadAttention"},
        {LayerType::Normalization3d,    "Normalization3d"},
        {LayerType::Pooling,            "Pooling"},
        {LayerType::Pooling3d,          "Pooling3d"},
        {LayerType::Recurrent,          "Recurrent"},
        {LayerType::Scaling2d,          "Scaling2d"},
        {LayerType::Scaling3d,          "Scaling3d"},
        {LayerType::Scaling4d,          "Scaling4d"},
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

    virtual vector<Shape> get_parameter_shapes() const { return {}; }

    virtual vector<Shape> get_state_shapes() const { return {}; }

    virtual vector<Shape> get_forward_shapes(Index) const { return {}; }

    virtual vector<Shape> get_backward_shapes(Index) const { return {}; }

    // Per-slot dtype for forward/backward TensorViews. Defaults to all ACTIVATION_DTYPE
    // (the flip target). Layers with FP32 tenants (layernorm stats, pooling indices,
    // valid masks, etc.) override to mark those slots as CUDNN_DATA_FLOAT.
    virtual vector<cudnnDataType_t> get_forward_dtypes(Index batch_size) const
    {
        return vector<cudnnDataType_t>(get_forward_shapes(batch_size).size(), CUDNN_ACTIVATION_DTYPE);
    }

    virtual vector<cudnnDataType_t> get_backward_dtypes(Index batch_size) const
    {
        return vector<cudnnDataType_t>(get_backward_shapes(batch_size).size(), CUDNN_ACTIVATION_DTYPE);
    }

    virtual Shape get_input_shape() const = 0;

    virtual Shape get_output_shape() const = 0;

    virtual ActivationFunction get_output_activation() const { return ActivationFunction::Linear; }

    Index get_inputs_number() const { return get_input_shape().size(); }

    Index get_outputs_number() const { return get_output_shape().size(); }

    // Forward propagation

    virtual void forward_propagate(ForwardPropagation&, size_t, bool) noexcept = 0;

    // Back propagation

    virtual void back_propagate(ForwardPropagation&, BackPropagation&, size_t) const noexcept {}

    virtual void from_XML(const tinyxml2::XmlDocument&) {}

    virtual void to_XML(tinyxml2::XmlPrinter&) const {}

    virtual void print() const {}

    vector<string> get_default_feature_names() const;

    vector<string> get_default_output_names() const;

    bool get_is_trainable() const { return is_trainable; }

    type* link_parameters(type* pointer);

    vector<TensorView>& get_parameter_views() { return parameters; }
    const vector<TensorView>& get_parameter_views() const { return parameters; }

    type* link_states(type* pointer);

    vector<TensorView>& get_state_views() { return states; }
    const vector<TensorView>& get_state_views() const { return states; }

protected:

    Layer() = default;

    string label = "my_layer";

    string name = "layer";

    LayerType layer_type = LayerType::Dense2d;

    bool is_trainable = true;

    bool is_first_layer = false;

    vector<TensorView> parameters;
    vector<TensorView> states; // non-trainable persistent state

    Tensor2 empty_2;
    Tensor3 empty_3;
    Tensor4 empty_4;
    
    void add_gradients(const vector<TensorView>&) const;
};

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
