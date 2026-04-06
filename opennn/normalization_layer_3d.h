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

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
