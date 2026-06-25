//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   A C T I V A T I O N   O P E R A T O R   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#pragma once

#include "operator.h"

namespace opennn
{

struct ActivationOperator : Operator
{
    static const EnumMap<ActivationFunction>& map() { return activation_function_map(); }
    static ActivationFunction from_string(const string& name) { return activation_function_from_string(name); }
    static const string& to_string(ActivationFunction function) { return activation_function_to_string(function); }

    ActivationFunction activation_function = ActivationFunction::Identity;

    size_t save_slot = SIZE_MAX;

    bool forward_fused = false;
    bool backward_fused = false;

    void set_activation_function(ActivationFunction);
    void set_activation_function(const string&);

    void forward_propagate(ForwardPropagation&, size_t, bool) override;
    void back_propagate(ForwardPropagation&, BackPropagation&, size_t) const override;

    void to_JSON(JsonWriter&) const override;
    void from_JSON(const Json*) override;

    ActivationOperator() = default;
    ActivationOperator(const ActivationOperator&) = delete;
    ActivationOperator& operator=(const ActivationOperator&) = delete;
};

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
