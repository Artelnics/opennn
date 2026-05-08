//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   F L A T T E N   L A Y E R   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#pragma once

#include "layer.h"
#include "operators.h"

namespace opennn
{

class Flatten final : public Layer
{
public:

    Flatten(const Shape& = {});

    Shape get_input_shape() const override { return input_shape; }
    Shape get_output_shape() const override { return { input_shape.size() }; }

    vector<Operator*> get_operators() override { return {&flat}; }

    void set(const Shape&);

    void set_input_shape(const Shape& new_input_shape) override { set(new_input_shape); }

private:

    Shape input_shape;

    Flat flat;

    enum Forward {Input, Output};
    enum Backward {OutputDelta, InputDelta};
};

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
