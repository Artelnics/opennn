//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   L A Y E R   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#pragma once

#include "opennn/core/tensor_types.h"
#include "opennn/core/tensor_operations.h"
#include "opennn/neural_network/operators/operator.h"
#include "opennn/neural_network/forward_propagation.h"
#include "opennn/neural_network/back_propagation.h"

#include <ranges>
#include <utility>

namespace opennn
{

enum class ForwardSlotKind { Pooled, Transient, TrainingOnly };
enum class BatchNormalization { No, Yes };
enum class Trainability { Frozen, Trainable };
enum class LayerType;

inline void check_rank(const Shape& shape, initializer_list<int> allowed,
                       const char* layer, const char* what)
{
    if (shape.empty()) return;
    if (ranges::any_of(allowed, [&](int r){ return int(shape.get_rank()) == r; })) return;

    string allowed_str;
    for (int r : allowed)
        allowed_str += format("{}{}", allowed_str.empty() ? "" : "/", r);

    throw runtime_error(format("{} layer supports {} rank {} (got {}).",
                               layer, what, allowed_str, shape.get_rank()));
}

class Layer
{

public:

    struct TiedWeight
    {
        const Layer* source = nullptr;
        size_t spec_index = 0;
        size_t source_spec_index = 0;
    };

    virtual ~Layer() = default;

    const string& get_label() const noexcept { return label; }

    const string& get_name() const;

    LayerType get_type() const noexcept { return layer_type; }

    virtual bool accepts_input_rank(Index) const { return false; }

    void set_input_shape(const Shape& new_input_shape)
    {
        throw_if(!new_input_shape.empty() && !accepts_input_rank(new_input_shape.get_rank()),
                 "{} layer does not accept an input of rank {}.",
                 get_name(), new_input_shape.get_rank());

        apply_input_shape(new_input_shape);
    }

    virtual void set_output_shape(const Shape&) {}

    void set_label(string new_label) { label = std::move(new_label); }

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

    virtual bool folds_input_delta_addend(size_t) const noexcept { return false; }

    virtual Shape get_input_shape() const noexcept { return input_shape; }

    virtual Shape get_output_shape() const = 0;

    virtual ActivationFunction get_output_activation() const { return ActivationFunction::Identity; }

    virtual Index get_sources_number() const noexcept { return 1; }

    virtual bool allows_successors() const noexcept { return true; }
    virtual bool is_recurrent() const noexcept { return false; }
    virtual bool skip_for_pre_scaled_input() const noexcept { return false; }
    virtual bool uses_sequence_position() const noexcept { return false; }

    virtual bool allows_bf16_input_cast(size_t) const noexcept { return true; }

    Index get_inputs_number() const noexcept { return get_input_shape().size(); }

    Index get_outputs_number() const { return get_output_shape().size(); }

    virtual void forward_propagate(ForwardPropagation& forward_propagation, size_t layer, ForwardPropagationMode pass)
    {
        for (Operator* op : get_operators())
            op->forward_propagate(forward_propagation, layer, pass);
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

    vector<TensorView>& get_parameter_views() { return parameters; }
    const vector<TensorView>& get_parameter_views() const noexcept { return parameters; }

    vector<TensorView>& get_parameter_scales() { return parameter_scales; }
    const vector<TensorView>& get_parameter_scales() const noexcept { return parameter_scales; }

    vector<Operator::SlotQuantization> get_parameter_quantization() const;

    virtual TiedWeight get_tied_weight() const { return {}; }
    virtual void set_tied_weight(const TiedWeight& tied_weight)
    {
        throw_if(tied_weight.source,
                 "{} layer does not support tied weights.", get_name());
    }

    void redistribute_parameters_to_operators();

private:

    const LayerType layer_type;

protected:

    virtual void apply_input_shape(const Shape& new_input_shape)
    {
        input_shape = new_input_shape;
    }

    static bool refresh_feature_storage(Buffer& storage, bool& dirty, Device device,
                                        Index features, Index columns,
                                        const function<void(float*)>& fill);

    Layer(LayerType type, Trainability trainability = Trainability::Trainable)
        : layer_type(type), is_trainable(trainability == Trainability::Trainable) {}

    enum Forward {Input, Output};

    string label = "my_layer";

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
