//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   F L A T T E N   L A Y E R   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#pragma once

#include "layer.h"

namespace opennn
{

class Flatten final : public Layer
{
public:

    Flatten(const Shape& = {});

    Shape get_input_shape() const override { return input_shape; }
    Shape get_output_shape() const override { return { input_shape.size() }; }

    void set(const Shape&);
    void set_input_shape(const Shape& new_input_shape) override { set(new_input_shape); }

    vector<pair<Shape, Type>> get_forward_specs(Index batch_size) const override;
    vector<pair<Shape, Type>> get_backward_specs(Index batch_size) const override;

    void forward_propagate(ForwardPropagation&, size_t layer, bool) noexcept override;
    void back_propagate(ForwardPropagation&, BackPropagation&, size_t layer) const noexcept override;

    void from_JSON(const JsonDocument&) override;
    void to_JSON(JsonWriter&) const override;

private:

    Shape input_shape;

    enum Forward {Input, Output};
    enum Backward {OutputDelta, InputDelta};
};

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
