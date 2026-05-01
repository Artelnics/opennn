//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   N O R M A L I Z A T I O N   L A Y E R   3 D   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#pragma once

#include "layer.h"
#include "operators.h"
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
        if(new_input_shape.rank >= 2)
        {
            sequence_length = new_input_shape[0];
            embedding_dimension = new_input_shape[1];
        }
    }

    // Gamma and beta are 1-D and stay FP32: our layernorm CUDA kernels read
    // `const float* gamma`/`beta` and would not compile if these slots ever
    // changed type.
    vector<pair<Shape, Type>> get_parameter_specs() const override;

    vector<pair<Shape, Type>> get_forward_specs(const Index batch_size) const override
    {
        const Type act = activation_dtype;
        return {
            /*Means*/              {{batch_size, sequence_length},                      Type::FP32},
            /*StandardDeviations*/ {{batch_size, sequence_length},                      Type::FP32},
            /*NormalizedInputs*/   {{batch_size, sequence_length, embedding_dimension}, act},
            /*Output*/             {{batch_size, sequence_length, embedding_dimension}, act},
        };
    }

    vector<pair<Shape, Type>> get_backward_specs(Index batch_size) const override
    {
        return {{{batch_size, sequence_length, embedding_dimension}, activation_dtype}};
    }

    void set(const Index = 0, Index = 0, const string& = "normalization_layer_3d");

    void set_parameters_random() override;
    void set_parameters_glorot() override;

    void forward_propagate(ForwardPropagation&, size_t, bool) noexcept override;

    void back_propagate(ForwardPropagation&, BackPropagation&, size_t) const noexcept override;

    float* link_parameters(float* pointer) override;

    void from_JSON(const JsonDocument&) override;
    void to_JSON(JsonWriter&) const override;

private:

    Index embedding_dimension = 0;
    Index sequence_length = 0;

    LayerNorm layer_norm;

    enum Parameters {Gamma, Beta};

    enum Forward {Input = 0, Means = 1, StandardDeviations = 2, NormalizedInput = 3, Output = 4};

    enum Backward {OutputDelta = 0, InputDelta = 1};

};

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
