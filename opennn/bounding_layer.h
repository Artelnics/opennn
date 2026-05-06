//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   B O U N D I N G   L A Y E R   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#pragma once

#include "layer.h"
#include "forward_propagation.h"
#include "back_propagation.h"

namespace opennn
{

class Bounding final : public Layer
{
public:

    enum class BoundingMethod{NoBounding, Bounding};

    Bounding(const Shape& = {0}, const string& = "bounding_layer");

    Shape get_input_shape() const override { return output_shape; }
    Shape get_output_shape() const override;

    const BoundingMethod& get_bounding_method() const;

    VectorR get_lower_bounds() const;
    VectorR get_upper_bounds() const;

    vector<pair<Shape, Type>> get_forward_specs(Index batch_size) const override;
    vector<pair<Shape, Type>> get_state_specs() const override;

    void set(const Shape& = {0}, const string& = "bounding_layer");

    void set_input_shape(const Shape&) override;
    void set_output_shape(const Shape&) override;

    void set_bounding_method(const BoundingMethod&);
    void set_bounding_method(const string&);

    void set_lower_bounds(const VectorR&);
    void set_lower_bound(Index, float);

    void set_upper_bounds(const VectorR&);
    void set_upper_bound(Index, float);

    float* link_states(float* pointer) override;

    void forward_propagate(ForwardPropagation&, size_t, bool) noexcept override;

    void from_JSON(const JsonDocument&) override;
    void load_state_from_JSON(const JsonDocument&) override;
    void to_JSON(JsonWriter&) const override;

private:

    Shape output_shape;

    BoundingMethod bounding_method = BoundingMethod::Bounding;

    enum States {Lower, Upper};
    enum Forward {Input, Output};

    static const EnumMap<BoundingMethod>& bounding_method_map();
};

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
