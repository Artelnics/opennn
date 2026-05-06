//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   A D D I T I O N   L A Y E R   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#pragma once

#include "layer.h"
#include "operators.h"

namespace opennn
{

class Addition final : public Layer
{
public:

    Addition(const Shape& = {}, const string& = "");

    Shape get_input_shape() const override { return input_shape; }
    Shape get_output_shape() const override { return input_shape; }

    vector<pair<Shape, Type>> get_backward_specs(Index batch_size) const override;

    vector<Operator*> get_operators() override { return {&add}; }

    void set(const Shape&, const string&);
    void set_input_shape(const Shape& shape) override { set(shape, label); }

    void back_propagate(ForwardPropagation&, BackPropagation&, size_t layer) const noexcept override;


private:

    Shape input_shape;

    Add add;

    enum Forward {Input, Output};
    enum Backward {OutputDelta, InputDelta0, InputDelta1};
};

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
