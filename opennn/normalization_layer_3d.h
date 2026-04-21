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

public:

    Normalization3d(const Shape& = Shape({0,0}),
                    const string& = "normalization_layer_3d");

    Index get_sequence_length() const { return sequence_length; }
    Index get_embedding_dimension() const { return embedding_dimension; }

    Shape get_input_shape() const override;
    Shape get_output_shape() const override;

    void set_input_shape(const Shape& new_input_shape) override
    {
        if(new_input_shape.rank() >= 2)
        {
            sequence_length = new_input_shape[0];
            embedding_dimension = new_input_shape[1];
        }
    }

    vector<Shape> get_parameter_shapes() const override;

    vector<Shape> get_forward_shapes(const Index batch_size) const override
    {
        return {{batch_size, sequence_length },                      // Means
                {batch_size, sequence_length },                      // StandardDeviations
                {batch_size, sequence_length, embedding_dimension},  // NormalizedInputs
                {batch_size, sequence_length, embedding_dimension}}; // Output
    }

    vector<cudnnDataType_t> get_forward_dtypes(Index) const override
    {
        return {CUDNN_DATA_FLOAT,        // Means
                CUDNN_DATA_FLOAT,        // StandardDeviations
                CUDNN_ACTIVATION_DTYPE,  // NormalizedInputs
                CUDNN_ACTIVATION_DTYPE}; // Output
    }

    vector<Shape> get_backward_shapes(Index batch_size) const override
    {
        return {{ batch_size, sequence_length, embedding_dimension}};
    }

    void set(const Index = 0, Index = 0, const string& = "normalization_layer_3d");

    void set_parameters_random() override;
    void set_parameters_glorot() override;

    void forward_propagate(ForwardPropagation&, size_t, bool) noexcept override;

    void back_propagate(ForwardPropagation&, BackPropagation&, size_t) const noexcept override;

    void from_XML(const XmlDocument&) override;
    void to_XML(XmlPrinter&) const override;

protected:

    TensorView gammas_device;
    TensorView betas_device;

private:

    Index embedding_dimension = 0;
    Index sequence_length = 0;

    enum Parameters {Gamma, Beta};

    enum Forward {Input = 0, Means = 1, StandardDeviations = 2, NormalizedInput = 3, Output = 4};

    enum Backward {OutputGradient = 0, InputGradient = 1};

};

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
