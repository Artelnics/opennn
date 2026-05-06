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
                                 const string& new_name) : Layer()
{
    name = "Normalization3d";
    layer_type = LayerType::Normalization3d;

    const Index new_sequence_length     = new_input_shape.empty() ? Index(0) : new_input_shape[0];
    const Index new_embedding_dimension = new_input_shape.rank >= 2 ? new_input_shape[1] : Index(0);

    set(new_sequence_length, new_embedding_dimension, new_name);
}

// Getters

Shape Normalization3d::get_input_shape() const
{
    return { sequence_length, embedding_dimension };
}

Shape Normalization3d::get_output_shape() const
{
    return { sequence_length, embedding_dimension };
}

vector<Operator*> Normalization3d::get_operators()
{
    return {&layer_norm};
}

vector<pair<Shape, Type>> Normalization3d::get_forward_specs(Index batch_size) const
{
    const Shape normalized_shape = Configuration::instance().is_gpu()
        ? Shape{}
        : Shape{batch_size, sequence_length, embedding_dimension};

    return {
        {{batch_size, sequence_length},                      Type::FP32},    // Means
        {{batch_size, sequence_length},                      Type::FP32},    // StandardDeviations
        {normalized_shape,                                   compute_dtype}, // NormalizedInputs
        {{batch_size, sequence_length, embedding_dimension}, compute_dtype}, // Output
    };
}

vector<pair<Shape, Type>> Normalization3d::get_backward_specs(Index batch_size) const
{
    return {{{batch_size, sequence_length, embedding_dimension}, compute_dtype}};
}

// Setters

void Normalization3d::set(Index new_sequence_length,
                          Index new_embedding_dimension,
                          const string& new_label)
{
    sequence_length     = new_sequence_length;
    embedding_dimension = new_embedding_dimension;

    set_label(new_label);

    layer_norm.set(sequence_length, embedding_dimension);
}

// link_parameters() is inherited from Layer; the base auto-distributes
// {gamma, beta} to layer_norm.

void Normalization3d::set_parameters_random()
{
    layer_norm.init_defaults();
}

void Normalization3d::set_parameters_glorot()
{
    set_parameters_random();
}

// Forward / back propagation

void Normalization3d::forward_propagate(ForwardPropagation& forward_propagation, size_t layer, bool) noexcept
{
    auto& forward_views = forward_propagation.views[layer];

    layer_norm.apply(forward_views[Input][0],
                     forward_views[Means][0],
                     forward_views[StandardDeviations][0],
                     forward_views[NormalizedInput][0],
                     forward_views[Output][0],
                     forward_propagation.batch_size);
}

void Normalization3d::back_propagate(ForwardPropagation& forward_propagation,
                                     BackPropagation& back_propagation,
                                     size_t layer) const noexcept
{
    auto& forward_views = forward_propagation.views[layer];
    auto& delta_views = back_propagation.delta_views[layer];
    auto& gradient_views = back_propagation.gradient_views[layer];

    layer_norm.apply_delta(forward_views[Input][0],
                           delta_views[OutputDelta][0],
                           forward_views[Means][0],
                           forward_views[StandardDeviations][0],
                           forward_views[NormalizedInput][0],
                           gradient_views[Gamma],
                           gradient_views[Beta],
                           delta_views[InputDelta][0],
                           forward_propagation.batch_size);
}

// Serialization

void Normalization3d::from_JSON(const JsonDocument& document)
{
    const Json* element = get_json_root(document, "Normalization3d");

    const string new_name = read_json_string(element, "Label");
    const Index new_sequence_length = read_json_index(element, "SequenceLength");
    const Index new_embedding_dimension = read_json_index(element, "EmbeddingDimension");

    set(new_sequence_length, new_embedding_dimension, new_name);
}

void Normalization3d::to_JSON(JsonWriter& printer) const
{
    printer.open_element("Normalization3d");

    write_json(printer, {
        {"Label", label},
        {"SequenceLength", to_string(get_sequence_length())},
        {"EmbeddingDimension", to_string(get_embedding_dimension())}
    });

    printer.close_element();
}

REGISTER(Layer, Normalization3d, "Normalization3d")

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
