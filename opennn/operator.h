//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   O P E R A T O R   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#pragma once

#include "tensor_types.h"
#include "tensor_operations.h"
#include "enum_map.h"
#include "forward_propagation.h"
#include "back_propagation.h"

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

    virtual void link_parameters(span<const TensorView>) {}
    virtual void link_gradients (span<const TensorView>) {}
    virtual void link_states    (span<const TensorView>) {}
    virtual void link_parameter_scales(span<const TensorView>) {}

    virtual void set_weights_dtype(Type new_weights_dtype) { weights_dtype = new_weights_dtype; }

    virtual void set_parameters_random() {}
    virtual void set_parameters_glorot() {}

    virtual void set_parameters_pytorch() { set_parameters_glorot(); }

    virtual void forward_propagate(ForwardPropagation&, size_t, bool) {}
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
        return slot == 0 ? forward_propagation.input_views[layer][0] : forward_propagation.forward_slots[layer][slot];
    }

    vector<TensorView>& get_inputs(ForwardPropagation& forward_propagation, size_t layer) const noexcept
    {
        return forward_propagation.input_views[layer];
    }

    TensorView& get_output(ForwardPropagation& forward_propagation, size_t layer, size_t slot_index = 0) const noexcept
    {
        return forward_propagation.forward_slots[layer][output_slots[slot_index]];
    }

    TensorView& get_output_delta(BackPropagation& back_propagation, size_t layer, size_t slot_index = 0) const noexcept
    {
        const size_t slot = output_delta_slots[slot_index];
        return slot == 0 ? back_propagation.layer_output_deltas[layer] : back_propagation.backward_slots[layer][slot];
    }

    TensorView& get_input_delta(BackPropagation& back_propagation, size_t layer, size_t slot_index = 0) const noexcept
    {
        return back_propagation.backward_slots[layer][input_delta_slots[slot_index]];
    }
};

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
