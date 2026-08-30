//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   O P E R A T O R   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#pragma once

#include "opennn/core/tensor_types.h"
#include "opennn/core/tensor_operations.h"
#include "opennn/core/enum_map.h"
#include "opennn/neural_network/forward_propagation.h"
#include "opennn/neural_network/back_propagation.h"

namespace opennn
{

class Json;
class JsonWriter;

inline bool link_views(span<const TensorView> views, initializer_list<TensorView*> targets)
{
    if (views.size() < targets.size()) return false;

    size_t index = 0;
    for (TensorView* target : targets) *target = views[index++];
    return true;
}

struct Operator
{
    virtual ~Operator() = default;

    virtual vector<TensorSpec> parameter_specs() const { return {}; }
    virtual vector<TensorSpec> state_specs()     const { return {}; }

    struct SlotQuantization { Index channels = 0; int axis = 0; };
    virtual vector<SlotQuantization> parameter_quantization() const { return {}; }

    // One trainable tensor and the gradient that shadows it, in the order the
    // layout hands the views over. An operator that declares its slots here
    // gets both link_parameters and link_gradients from the base. The two used
    // to be written as a matched pair per operator, each restating the same
    // use_bias / RMS / positional condition, so the pair could drift.
    struct ParameterSlot
    {
        TensorView* parameter = nullptr;
        TensorView* gradient  = nullptr;

        // Whether the layout supplies this slot at all. An absent slot is
        // reset so a projection that drops its bias, or an RMS norm that has
        // no beta, cannot keep a stale view. The exception is a view another
        // link step owns: the embedding's positional table is a state when it
        // is not a trained parameter, and link_states has already filled it.
        bool present = true;
        bool retain_when_absent = false;
    };

    virtual vector<ParameterSlot> parameter_slots() { return {}; }

    virtual void link_parameters(span<const TensorView> views) { link_slots(views, &ParameterSlot::parameter); }
    virtual void link_gradients (span<const TensorView> views) { link_slots(views, &ParameterSlot::gradient); }
    virtual void link_states    (span<const TensorView>) {}
    virtual void link_parameter_scales(span<const TensorView>) {}

    // Returns whether the layout supplied a view for every present slot, which
    // is what the old link_views reported and what two operators still act on.
    bool link_slots(span<const TensorView> views, TensorView* ParameterSlot::* member)
    {
        const vector<ParameterSlot> slots = parameter_slots();

        size_t needed = 0;
        for (const ParameterSlot& slot : slots) needed += slot.present ? 1 : 0;
        if (views.size() < needed) return false;

        size_t index = 0;
        for (const ParameterSlot& slot : slots)
        {
            TensorView* const target = slot.*member;
            if (slot.present)                 *target = views[index++];
            else if (!slot.retain_when_absent) *target = {};
        }

        return true;
    }

    virtual void initialize_states() {}

    virtual void set_weights_dtype(Type new_weights_dtype) { weights_dtype = new_weights_dtype; }

    virtual void set_parameters_random() {}
    virtual void set_parameters_glorot() {}

    virtual void set_parameters_pytorch() { set_parameters_glorot(); }

    virtual void forward_propagate(ForwardPropagation&, size_t, ForwardPropagationMode) {}
    virtual void back_propagate(ForwardPropagation&, BackPropagation&, size_t) const {}

    virtual void to_JSON  (JsonWriter&) const {}
    virtual void from_JSON(const Json*)       {}
    virtual void load_state_from_JSON(const Json*) {}

    Type compute_dtype = Type::FP32;
    Type weights_dtype = Type::FP32;

    vector<size_t> input_slots = {0};
    vector<size_t> output_slots = {1};

    vector<size_t> input_delta_slots = {1};
    vector<size_t> output_delta_slots = {0};

    TensorView& get_input(ForwardPropagation& forward_propagation, size_t layer, size_t slot_index = 0) const noexcept
    {
        const size_t slot = input_slots[slot_index];
        return slot == 0 ? forward_propagation.inputs[layer][0] : forward_propagation.slots[layer][slot];
    }

    vector<TensorView>& get_inputs(ForwardPropagation& forward_propagation, size_t layer) const noexcept
    {
        return forward_propagation.inputs[layer];
    }

    TensorView& get_output(ForwardPropagation& forward_propagation, size_t layer, size_t slot_index = 0) const noexcept
    {
        return forward_propagation.slots[layer][output_slots[slot_index]];
    }

    TensorView& get_output_delta(BackPropagation& back_propagation, size_t layer, size_t slot_index = 0) const noexcept
    {
        const size_t slot = output_delta_slots[slot_index];
        return slot == 0 ? back_propagation.output_deltas[layer] : back_propagation.slots[layer][slot];
    }

    TensorView& get_input_delta(BackPropagation& back_propagation, size_t layer, size_t slot_index = 0) const noexcept
    {
        return back_propagation.slots[layer][input_delta_slots[slot_index]];
    }
};

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
