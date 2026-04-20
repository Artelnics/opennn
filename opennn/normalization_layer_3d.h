//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   N O R M A L I Z A T I O N   L A Y E R   3 D   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#pragma once

#include "layer.h"
#include "forward_propagation.h"
#include "back_propagation.h"

namespace opennn
{

class Normalization3d final : public Layer
{
private:

    Index embedding_dimension = 0;
    Index sequence_length = 0;

#ifdef OPENNN_WITH_CUDA
    TensorView gammas_device;
    TensorView betas_device;
#endif

    enum Parameters {Gammas, Betas};

    vector<Shape> get_parameter_shapes() const override;

    // View slots: 0=Inputs(wired), 1=Means, 2=StdDevs, 3=NormalizedInputs, 4=Outputs
    enum Forward {Inputs = 0, Means = 1, StandardDeviations = 2, NormalizedInputs = 3, Outputs = 4};

    // Forward shapes: intermediate buffers first, output last.
    // The last shape is the one wired to the downstream layer.
    vector<Shape> get_forward_shapes(const Index batch_size) const override
    {
        return {{batch_size, sequence_length },                      // slot 1: Means
                {batch_size, sequence_length },                      // slot 2: StandardDeviations
                {batch_size, sequence_length, embedding_dimension},  // slot 3: NormalizedInputs
                {batch_size, sequence_length, embedding_dimension}}; // slot 4: Outputs (LAST = wired downstream)
    }

    enum Backward {OutputGradients = 0, InputGradients = 1};

    vector<Shape> get_backward_shapes(Index batch_size) const override
    {
        return {{ batch_size, sequence_length, embedding_dimension}};
    }

public:

    Normalization3d(const Shape& = Shape({0,0}),
                    const string& = "normalization_layer_3d");

    // Getters

    Index get_sequence_length() const { return sequence_length; }
    Index get_embedding_dimension() const { return embedding_dimension; }

    Shape get_input_shape() const override;
    Shape get_output_shape() const override;

    // Setters

    void set(const Index = 0, Index = 0, const string& = "normalization_layer_3d");

    void set_input_shape(const Shape& new_input_shape) override
    {
        if(new_input_shape.rank() >= 2)
        {
            sequence_length = new_input_shape[0];
            embedding_dimension = new_input_shape[1];
        }
    }

    // Parameter initialization

    void set_parameters_random() override;
    void set_parameters_glorot() override;

    // Forward / back propagation

    void forward_propagate(ForwardPropagation&, size_t, bool) noexcept override;

    void back_propagate(ForwardPropagation&, BackPropagation&, size_t) const noexcept override;

    // Serialization

    void from_XML(const XmlDocument&) override;
    void to_XML(XmlPrinter&) const override;
};

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
