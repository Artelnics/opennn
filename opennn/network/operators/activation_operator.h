// SPDX-License-Identifier: LGPL-2.1-or-later
// Copyright (C) 2005-2026 Artificial Intelligence, SL.

#pragma once

#include "opennn/network/operators/operator.h"

namespace opennn
{

struct ActivationOperator : Operator
{
    static ActivationFunction from_string(const string& name) { return activation_function_from_string(name); }
    static const string& to_string(ActivationFunction function) { return activation_function_to_string(function); }

    ActivationFunction activation_function = ActivationFunction::Identity;

    optional<size_t> saved_output_slot;

    bool forward_fused = false;
    bool backward_fused = false;

    bool backward_fused_by_consumer = false;

    void set_activation_function(ActivationFunction new_function) { activation_function = new_function; }
    void set_activation_function(const string& name) { set_activation_function(from_string(name)); }

    void forward_propagate(ForwardPropagation&, size_t, ForwardPropagationMode) override;
    void back_propagate(ForwardPropagation&, BackPropagation&, size_t) const override;

    void to_JSON(JsonWriter&) const override;
    void from_JSON(const Json*) override;

    ActivationOperator() = default;
    ActivationOperator(const ActivationOperator&) = delete;
    ActivationOperator& operator=(const ActivationOperator&) = delete;
};

}
