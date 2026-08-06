//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   C 2 P S A   L A Y E R   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#pragma once

#include "layer.h"
#include "c2psa_operator.h"

namespace opennn
{

class C2PSA final : public Layer
{
public:

    C2PSA(const Shape& = {}, const string& = "c2psa_layer");

    Shape get_output_shape() const override;

    void set(const Shape&, const string&);
    void set_input_shape(const Shape&) override;

    vector<TensorSpec> get_forward_specs(Index) const override;
    vector<TensorSpec> get_backward_specs(Index) const override;

private:

    C2PSAOperator c2psa;

    void configure_operator();
};

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
