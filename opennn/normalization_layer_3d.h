//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   N O R M A L I Z A T I O N   L A Y E R   3 D   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#pragma once

#include "layer.h"

namespace opennn
{

class Normalization3d final : public Layer
{

public:

    Normalization3d(const Shape& = Shape({0,0}),
                    const string& = "normalization_layer_3d");

    Index get_sequence_length() const;
    Index get_embedding_dimension() const;

    Shape get_input_shape() const override;
    Shape get_output_shape() const override;

    vector<Shape> get_parameter_shapes() const override;

    vector<Shape> get_forward_shapes(const Index batch_size) const override
    {
        return {{batch_size, sequence_length, embedding_dimension},  // Outputs
                {batch_size, sequence_length },                      // Means
                {batch_size, sequence_length },                      // StandardDeviations
                {batch_size, sequence_length, embedding_dimension}}; // NormalizedInputs
    }

    vector<Shape> get_backward_shapes(Index batch_size) const override
    {
        return {{ batch_size, sequence_length, embedding_dimension}};
    }

    void set(const Index = 0, Index = 0, const string& = "normalization_layer_3d");

    void set_parameters_random() override;
    void set_parameters_glorot() override;

    void forward_propagate(ForwardPropagation&, size_t, bool) override;

    void back_propagate(ForwardPropagation&, BackPropagation&, size_t) const override;

    void from_XML(const XMLDocument&) override;
    void to_XML(XMLPrinter&) const override;

#ifdef CUDA

protected:

    TensorView gammas_device;
    TensorView betas_device;

#endif

private:

    Index embedding_dimension;
    Index sequence_length;

    enum Parameters {Gammas, Betas};
    enum Forward {Inputs, Means, StandardDeviations, Outputs};
    enum Backward {OutputGradients, InputGradients};

};


#ifdef CUDA

struct Normalization3dForwardPropagationCuda : public LayerForwardPropagationCuda
{
    TensorCuda means_device;
    TensorCuda inv_variances_device;
};


struct Normalization3dBackPropagationCuda : public LayerBackPropagationCuda
{
    TensorView gamma_gradients;
    TensorView beta_gradients;
};

#endif

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// This library is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 2.1 of the License, or any later version.
// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.
// You should have received a copy of the GNU Lesser General Public
// License along with this library; if not, write to the Free Software
// Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
