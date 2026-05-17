//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   N O R M A L I Z A T I O N   L A Y E R   3 D   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "registry.h"
#include "tensor_utilities.h"
#include "math_utilities.h"
#include "normalization_layer_3d.h"
#include "neural_network.h"
#include "loss.h"
#include "forward_propagation.h"
#include "back_propagation.h"

namespace opennn
{

Normalization3d::Normalization3d(const Shape& new_input_shape,
                                 const string& new_name)
    : Layer(LayerType::Normalization3d)
{
    operators = {&layer_norm};

    set(new_input_shape.dim_or_zero(0), new_input_shape.dim_or_zero(1), new_name);

    layer_norm.output_slots = {Means, StandardDeviations, NormalizedInput, Output};
}

Shape Normalization3d::get_input_shape() const
{
    return { sequence_length, embedding_dimension };
}

Shape Normalization3d::get_output_shape() const
{
    return { sequence_length, embedding_dimension };
}

vector<pair<Shape, Type>> Normalization3d::get_forward_specs(Index batch_size) const
{
    const Shape normalized_shape = is_gpu()
        ? Shape{}
        : Shape{batch_size, sequence_length, embedding_dimension};

    return {
        {{batch_size, sequence_length},                      Type::FP32},    // Means
        {{batch_size, sequence_length},                      Type::FP32},    // StandardDeviations
        {normalized_shape,                                   compute_dtype}, // NormalizedInputs
        {{batch_size, sequence_length, embedding_dimension}, compute_dtype}, // Output
    };
}

void Normalization3d::set(Index new_sequence_length,
                          Index new_embedding_dimension,
                          const string& new_label)
{
    sequence_length     = new_sequence_length;
    embedding_dimension = new_embedding_dimension;

    set_label(new_label);

    layer_norm.set(sequence_length, embedding_dimension);
}

void Normalization3d::set_input_shape(const Shape& new_input_shape)
{
    if (new_input_shape.rank < 2) return;
    set(new_input_shape[0], new_input_shape[1], label);
}

void Normalization3d::read_JSON_body(const Json* element)
{
    const Shape new_input_shape = string_to_shape(read_json_string(element, "InputDimensions"));

    set(new_input_shape.dim_or_zero(0), new_input_shape.dim_or_zero(1), get_label());
}

REGISTER(Layer, Normalization3d, "Normalization3d")

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
