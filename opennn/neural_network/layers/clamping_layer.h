//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   C L A M P I N G   L A Y E R   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#pragma once

#include "opennn/neural_network/layers/layer.h"
#include "opennn/neural_network/operators/operator.h"

namespace opennn
{

void apply_clamping(const TensorView&, const TensorView&, const TensorView&, TensorView&);

struct ClampingOperator : Operator
{
    enum class Method { NoClamping, Clamping };

    Method method = Method::Clamping;

    TensorView lower;
    TensorView upper;

    void forward_propagate(ForwardPropagation&, size_t, bool) override;
};

class Clamping final : public Layer
{
public:

    using ClampingMethod = ClampingOperator::Method;

    Clamping(const Shape& = {0}, const string& = "clamping_layer");

    Shape get_input_shape() const noexcept override { return output_shape; }
    Shape get_output_shape() const noexcept override { return output_shape; }

    const ClampingMethod& get_clamping_method() const noexcept { return clamping.method; }

    VectorR get_lower_bounds() const { return Map<const VectorR>(lower_bounds.data(), ssize(lower_bounds)); }
    VectorR get_upper_bounds() const { return Map<const VectorR>(upper_bounds.data(), ssize(upper_bounds)); }

    void set(const Shape& = {0}, const string& = "clamping_layer");

    bool accepts_input_rank(Index rank) const override { return is_one_of(rank, 1, 2, 3); }
    bool allows_successors() const noexcept override { return false; }

    void apply_input_shape(const Shape& new_input_shape) override { set(new_input_shape, label); }

    void set_clamping_method(const ClampingMethod&);
    void set_clamping_method(const string&);

    void set_lower_bound(Index, float);

    void set_upper_bound(Index, float);

    float* link_states(float*, Device) override;

    void read_JSON_body(const Json*) override;
    void write_JSON_body(JsonWriter&) const override;

    string write_expression(const vector<string>&,
                            const vector<string>&) const override;

private:

    Shape output_shape;

    vector<float> lower_bounds;
    vector<float> upper_bounds;

    // Device mirror of the configured bounds. This is model state, not
    // per-execution scratch, and is refreshed only when configuration changes.
    Buffer op_storage;
    bool   op_storage_dirty = true;

    ClampingOperator clamping;

    void refresh_op_storage(Device);

    static const EnumMap<ClampingMethod>& clamping_method_map();
};

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
