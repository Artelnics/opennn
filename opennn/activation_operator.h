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

struct ActivationOp : Operator
{
    using Function = ActivationFunction;

    static const EnumMap<Function>& map() { return activation_function_map(); }
    static Function from_string(const string& name) { return activation_function_from_string(name); }
    static const string& to_string(Function function) { return activation_function_to_string(function); }
    static cudnnActivationMode_t to_cudnn_mode(Function function);

    Function function = Function::Identity;

    CudnnDescriptor<cudnnActivationDescriptor_t> descriptor;

    vector<size_t> output_slots_backward;

    bool forward_fused = false;
    bool backward_fused = false;

    void set_function(Function new_function);
    void set_function(const string& name);

    void forward_propagate(ForwardPropagation& fp, size_t layer, bool is_training) override;
    void back_propagate(ForwardPropagation& fp, BackPropagation& bp, size_t layer) const override;

    void apply_delta(const TensorView& outputs, TensorView& delta) const;

    void to_JSON(JsonWriter& w) const override;
    void from_JSON(const Json* parent) override;

    ActivationOp() = default;
    ActivationOp(const ActivationOp&) = delete;
    ActivationOp& operator=(const ActivationOp&) = delete;
};

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
