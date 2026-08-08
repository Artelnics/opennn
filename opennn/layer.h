//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   L A Y E R   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#pragma once

#include "tensor_types.h"
#include "tensor_operations.h"
#include "operator.h"
#include "forward_propagation.h"
#include "back_propagation.h"

#include <ranges>

namespace opennn
{

enum class ForwardSlotKind { Pooled, Transient, TrainingOnly };

enum class LayerType
{
    Activation,
    Addition,
    Bounding,
    Concatenation,
    Convolutional,
    Dense,
    Detection,
    DetectionV8,
    Embedding,
    Flatten,
    LongShortTermMemory,
    MultiHeadAttention,
    Normalization3d,
    RMSNormalization3d,
    GroupedQueryAttention,
    NonMaxSuppression,
    Pooling,
    Pooling3d,
    Recurrent,
    Scaling,
    Tokenizer,
    Unscaling,
    Upsample,
    C2PSA
};

inline const EnumMap<LayerType>& layer_type_map()
{
    static const vector<pair<LayerType, string>> entries = {
        {LayerType::Activation,         "Activation"},
        {LayerType::Addition,           "Addition"},
        {LayerType::Bounding,           "Bounding"},
        {LayerType::Concatenation,      "Concatenation"},
        {LayerType::Concatenation,      "Concatenate"},
        {LayerType::Convolutional,      "Convolutional"},
        {LayerType::Dense,              "Dense"},
        {LayerType::Detection,          "Detection"},
        {LayerType::DetectionV8,        "DetectionV8"},
        {LayerType::Embedding,          "Embedding"},
        {LayerType::Flatten,            "Flatten"},
        {LayerType::LongShortTermMemory, "LongShortTermMemory"},
        {LayerType::MultiHeadAttention, "MultiHeadAttention"},
        {LayerType::Normalization3d,    "Normalization3d"},
        {LayerType::RMSNormalization3d, "RMSNormalization3d"},
        {LayerType::GroupedQueryAttention, "GroupedQueryAttention"},
        {LayerType::NonMaxSuppression,  "NonMaxSuppression"},
        {LayerType::Pooling,            "Pooling"},
        {LayerType::Pooling3d,          "Pooling3d"},
        {LayerType::Recurrent,          "Recurrent"},
        {LayerType::Scaling,            "Scaling"},
        {LayerType::Tokenizer,          "Tokenizer"},
        {LayerType::Unscaling,          "Unscaling"},
        {LayerType::Upsample,           "Upsample"},
        {LayerType::C2PSA,              "C2PSA"}
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

inline void check_rank(const Shape& shape, initializer_list<int> allowed,
                       const char* layer, const char* what)
{
    if (shape.empty()) return;
    if (ranges::any_of(allowed, [&](int r){ return int(shape.rank) == r; })) return;

    string allowed_str;
    for (int r : allowed)
        allowed_str += format("{}{}", allowed_str.empty() ? "" : "/", r);

    throw runtime_error(format("{} layer supports {} rank {} (got {}).",
                               layer, what, allowed_str, shape.rank));
}

class Layer
{

public:

    virtual ~Layer() = default;

    const string& get_label() const noexcept { return label; }

    const string& get_name() const { return layer_type_to_string(layer_type); }

    LayerType get_type() const noexcept { return layer_type; }

    // Which input ranks this layer can represent. Callers that build a shape
    // rather than receive one - neuron selection, for instance - must ask before
    // setting it, because set_input_shape below throws on a rank a layer cannot
    // take. The base refuses everything so a layer that forgets to declare its
    // ranks fails loudly the first time one is set, instead of accepting a shape
    // it cannot honour.
    virtual bool accepts_input_rank(Index) const { return false; }

    // Validation lives here and nowhere else, so no layer can omit it. What a
    // layer does with an accepted shape is its own business - apply_input_shape
    // below - but whether the shape is acceptable at all is answered uniformly.
    void set_input_shape(const Shape& new_input_shape)
    {
        throw_if(!new_input_shape.empty() && !accepts_input_rank(new_input_shape.rank),
                 "{} layer does not accept an input of rank {}.",
                 get_name(), new_input_shape.rank);

        apply_input_shape(new_input_shape);
    }

    virtual void set_output_shape(const Shape&) {}

    void set_label(string new_label) { label = move(new_label); }

    Index get_parameters_number() const;
    const vector<Operator*>& get_operators() const noexcept { return operators; }
    virtual vector<TensorSpec> get_parameter_specs() const;
    virtual vector<TensorSpec> get_state_specs()     const;
    virtual vector<TensorSpec> get_forward_specs(Index batch_size) const
    {
        return {{Shape{batch_size}.append(get_output_shape()), compute_dtype}};
    }
    virtual vector<TensorSpec> get_backward_specs(Index batch_size) const
    {
        if (!is_trainable) return {};
        return {{Shape{batch_size}.append(get_input_shape()), compute_dtype}};
    }

    virtual ForwardSlotKind get_forward_slot_kind(size_t) const
    {
        return ForwardSlotKind::Pooled;
    }
    virtual size_t get_recomputable_forward_slot() const noexcept { return SIZE_MAX; }
    virtual void recompute_forward_slot(ForwardPropagation&, size_t) {}
    virtual bool backward_uses_forward_output() const noexcept { return true; }
    virtual bool backward_uses_input(size_t) const noexcept { return true; }
    virtual bool preserves_output_delta_during_backward() const noexcept { return false; }
    virtual bool allows_input_delta_alias() const noexcept { return false; }

    virtual Shape get_input_shape() const noexcept { return input_shape; }

    virtual Shape get_output_shape() const = 0;

    virtual ActivationFunction get_output_activation() const { return ActivationFunction::Identity; }

    Index get_inputs_number() const noexcept { return get_input_shape().size(); }

    Index get_outputs_number() const { return get_output_shape().size(); }

    virtual void forward_propagate(ForwardPropagation& forward_propagation, size_t layer, bool is_training)
    {
        for (Operator* op : get_operators())
            op->forward_propagate(forward_propagation, layer, is_training);
    }

    virtual void back_propagate(ForwardPropagation& forward_propagation, BackPropagation& back_propagation, size_t i) const
    {
        for (Operator* op : views::reverse(get_operators()))
            op->back_propagate(forward_propagation, back_propagation, i);
    }

    virtual void from_JSON(const JsonDocument&);

    virtual void read_JSON_body(const Json*) {}

    virtual void on_loaded() {}

    virtual void load_state_from_JSON(const JsonDocument&);

    virtual void to_JSON(JsonWriter&) const;

    virtual void write_JSON_body(JsonWriter&) const {}

    virtual string write_expression(const vector<string>&  ,
                                    const vector<string>&  ) const { return {}; }

    bool get_is_trainable() const noexcept { return is_trainable; }
    void set_is_trainable(bool trainable) { is_trainable = trainable; }

    Type get_compute_dtype() const noexcept { return compute_dtype; }
    Device get_compute_device() const noexcept { return compute_device; }

    void set_compute_dtype(Type new_compute_dtype)
    {
        weights_dtype = new_compute_dtype;
        compute_dtype = activation_dtype(new_compute_dtype);
        on_compute_dtype_changed();

        for (Operator* op : operators)
            op->set_weights_dtype(weights_dtype);
    }

    void set_compute_device(Device new_compute_device) { compute_device = new_compute_device; }

    virtual void on_compute_dtype_changed() {}

    virtual float* link_states(float*, Device);

    float* link_gradients(float*, Device);

protected:

    // The layer's own reaction to an input shape that has already been
    // accepted. Assigning input_shape suits most layers; those that decompose
    // the shape or rebuild operators from it override this.
    virtual void apply_input_shape(const Shape& new_input_shape)
    {
        input_shape = new_input_shape;
    }


    static bool refresh_feature_storage(Buffer& storage, bool& dirty, Device device,
                                        Index features, Index columns,
                                        const function<void(float*)>& fill);

public:

    vector<TensorView>& get_parameter_views() { return parameters; }
    const vector<TensorView>& get_parameter_views() const noexcept { return parameters; }

    vector<TensorView>& get_parameter_scales() { return parameter_scales; }
    const vector<TensorView>& get_parameter_scales() const noexcept { return parameter_scales; }

    vector<Operator::SlotQuantization> get_parameter_quantization() const;

    struct TiedWeight { const Layer* source = nullptr; size_t spec_index = 0; size_t source_spec_index = 0; };
    virtual TiedWeight get_tied_weight() const { return {}; }

    void redistribute_parameters_to_operators();

protected:

    Layer() = default;

    Layer(LayerType t, bool trainable = true)
        : layer_type(t), is_trainable(trainable) {}

    enum Forward {Input, Output};

    string label = "my_layer";

    LayerType layer_type = LayerType::Dense;

    bool is_trainable = true;

    Shape input_shape;

    Type compute_dtype = Type::FP32;
    Type weights_dtype = Type::FP32;
    Device compute_device = Device::CPU;

    vector<TensorView> parameters;
    vector<TensorView> parameter_scales;
    vector<TensorView> states;

    vector<Operator*> operators;

    float* link_views_to_operators(
        vector<TensorView>&, float*,
        vector<TensorSpec> (Operator::*specs_fn)() const,
        void (Operator::*link_fn)(span<const TensorView>),
        Device);

};

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
