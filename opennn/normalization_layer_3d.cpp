//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   N O R M A L I Z A T I O N   L A Y E R   3 D   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "registry.h"
#include "tensor_types.h"
#include "normalization_layer_3d.h"

namespace opennn
{

Normalization3d::Normalization3d(const Shape& new_input_shape,
                                 const string& new_name)
    : Layer(LayerType::Normalization3d)
{
    operators = {&layer_normalization};

    set(new_input_shape.dim_or_zero(0), new_input_shape.dim_or_zero(1), new_name);

    layer_normalization.output_slots = {Means, StandardDeviations, NormalizedInput, Output};
}

Shape Normalization3d::get_input_shape() const noexcept
{
    return { sequence_length, embedding_dimension };
}

Shape Normalization3d::get_output_shape() const
{
    return { sequence_length, embedding_dimension };
}

vector<TensorSpec> Normalization3d::get_forward_specs(Index batch_size) const
{
    // The NormalizedInput slot is unused on CUDA in the plain path, but the
    // fused residual-add path stores the post-add sum there, so it must be sized.
    const bool need_sum = layer_normalization.fuse_add || get_compute_device() != Device::CUDA;
    return {
        {{batch_size, sequence_length},                      Type::FP32},
        {{batch_size, sequence_length},                      Type::FP32},
        {need_sum ? Shape{batch_size, sequence_length, embedding_dimension} : Shape{}, compute_dtype},
        {{batch_size, sequence_length, embedding_dimension}, compute_dtype},
    };
}

vector<TensorSpec> Normalization3d::get_backward_specs(Index batch_size) const
{
    // Fused norm has two source layers (main, residual), so the backward must
    // provide a gradient buffer for each, mirroring the Addition layer.
    const Index inputs = layer_normalization.fuse_add ? 2 : 1;
    return vector<TensorSpec>(size_t(inputs),
        {Shape{batch_size, sequence_length, embedding_dimension}, compute_dtype});
}

void Normalization3d::set_fuse_add(bool on)
{
    layer_normalization.fuse_add = on;
    // The compute reads the main input via slot 0 and the residual directly from
    // the second gathered source, so input_slots stays {0}. The backward routes a
    // gradient to each of the two source layers: slot 1 -> main, slot 2 -> residual.
    layer_normalization.input_delta_slots   = on ? vector<size_t>{1, 2} : vector<size_t>{1};
    layer_normalization.residual_delta_slot = on ? 2 : 0;
}

void Normalization3d::set(Index new_sequence_length,
                          Index new_embedding_dimension,
                          const string& new_label)
{
    sequence_length     = new_sequence_length;
    embedding_dimension = new_embedding_dimension;

    set_label(new_label);

    layer_normalization.set(sequence_length, embedding_dimension);
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

    if (element->has("FuseAdd"))
        set_fuse_add(read_json_bool(element, "FuseAdd"));
}

void Normalization3d::write_JSON_body(JsonWriter& printer) const
{
    write_json(printer, {
        {"FuseAdd", to_string(layer_normalization.fuse_add)}
    });
}

REGISTER(Layer, Normalization3d, "Normalization3d")

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
