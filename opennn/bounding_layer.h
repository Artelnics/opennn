//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   B O U N D I N G   L A Y E R   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#pragma once

#include "layer.h"
#include "operators.h"

namespace opennn
{

/// @brief Output bounding layer that clips outputs to configured lower and upper limits.
class Bounding final : public Layer
{
public:

    using BoundingMethod = BoundOp::Method;

    /// @brief Constructs a bounding layer for the given output shape and label.
    Bounding(const Shape& = {0}, const string& = "bounding_layer");

    Shape get_input_shape() const override { return output_shape; }
    Shape get_output_shape() const override { return output_shape; }

    const BoundingMethod& get_bounding_method() const { return bound.method; }

    /// @brief Returns the configured lower bound for each output variable.
    VectorR get_lower_bounds() const;
    /// @brief Returns the configured upper bound for each output variable.
    VectorR get_upper_bounds() const;

    /// @brief Reconfigures the layer with a new output shape and label.
    void set(const Shape& = {0}, const string& = "bounding_layer");

    /// @copydoc Layer::set_input_shape
    void set_input_shape(const Shape&) override;

    /// @brief Sets the bounding method enum used to clip outputs.
    void set_bounding_method(const BoundingMethod&);
    /// @brief Sets the bounding method from its string name.
    void set_bounding_method(const string&);

    /// @brief Sets the full vector of lower bounds, one entry per output variable.
    void set_lower_bounds(const VectorR&);
    /// @brief Sets the lower bound of a single output variable by index.
    void set_lower_bound(Index, float);

    /// @brief Sets the full vector of upper bounds, one entry per output variable.
    void set_upper_bounds(const VectorR&);
    /// @brief Sets the upper bound of a single output variable by index.
    void set_upper_bound(Index, float);

    /// @copydoc Layer::link_states
    float* link_states(float*) override;

    /// @copydoc Layer::read_JSON_body
    void read_JSON_body(const Json*) override;
    /// @copydoc Layer::write_JSON_body
    void write_JSON_body(JsonWriter&) const override;

    /// @copydoc Layer::write_expression
    string write_expression(const vector<string>& input_names,
                            const vector<string>& output_names) const override;

private:

    Shape output_shape;

    vector<float> lower_bounds;
    vector<float> upper_bounds;

    Buffer op_storage;
    bool   op_storage_dirty = true;

    BoundOp bound;

    void refresh_op_storage(Device device);

    static const EnumMap<BoundingMethod>& bounding_method_map();
};

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
