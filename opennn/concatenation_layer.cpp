//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   C O N C A T E N A T I O N   L A Y E R   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "concatenation_layer.h"
#include "json.h"
#include "string_utilities.h"

#include "tensor_operations.h"
#include "forward_propagation.h"
#include "back_propagation.h"
#ifdef OPENNN_HAS_CUDA
#  include "kernel.cuh"
#endif

namespace opennn
{

void ConcatenationOperator::forward_propagate(ForwardPropagation& forward_propagation, size_t layer, bool)
{
    const vector<TensorView>& inputs = get_inputs(forward_propagation, layer);
    TensorView& output = get_output(forward_propagation, layer);

    throw_if(inputs.size() != input_channels.size(),
             "Concatenation: input count mismatch.");

    const Index total_channels = output.shape[3];

#ifdef OPENNN_HAS_CUDA
    if (output.is_cuda())
    {
        const Index batch_size = output.shape[0];
        const Index height     = inputs[0].shape[1];
        const Index width      = inputs[0].shape[2];
        Index ch_offset = 0;
        for (size_t i = 0; i < inputs.size(); ++i)
        {
            concat_forward_slice_cuda(to_int(batch_size), to_int(height), to_int(width),
                                      to_int(input_channels[i]), to_int(total_channels), to_int(ch_offset),
                                      inputs[i].as<float>(), output.as<float>());
            ch_offset += input_channels[i];
        }
        return;
    }
#endif

    float* dst = output.as<float>();
    const Index pixels = output.size() / total_channels;

    #pragma omp parallel for
    for (Index pixel = 0; pixel < pixels; ++pixel)
    {
        float* out_row = dst + pixel * total_channels;
        Index ch_offset = 0;
        for (size_t i = 0; i < inputs.size(); ++i)
        {
            const Index in_c = inputs[i].shape[3];
            memcpy(out_row + ch_offset, inputs[i].as<float>() + pixel * in_c, in_c * sizeof(float));
            ch_offset += in_c;
        }
    }
}

void ConcatenationOperator::back_propagate(ForwardPropagation&, BackPropagation& back_propagation, size_t layer) const
{
    const TensorView& output_delta = get_output_delta(back_propagation, layer);

    const auto& backward_slots = back_propagation.backward_slots[layer];
    const bool needs_input_delta = ranges::any_of(input_delta_slots, [&](size_t slot)
    {
        return slot < backward_slots.size() && !backward_slots[slot].empty();
    });

    if (!needs_input_delta) return;

    const Index total_channels = output_delta.shape[3];

#ifdef OPENNN_HAS_CUDA
    if (output_delta.is_cuda())
    {
        const Index batch_size = output_delta.shape[0];
        const Index height     = output_delta.shape[1];
        const Index width      = output_delta.shape[2];
        Index ch_offset = 0;
        for (size_t i = 0; i < input_channels.size(); ++i)
        {
            TensorView& in_delta = get_input_delta(back_propagation, layer, i);
            if (!in_delta.empty())
                concat_backward_slice_cuda(to_int(batch_size), to_int(height), to_int(width),
                                           to_int(input_channels[i]), to_int(total_channels), to_int(ch_offset),
                                           output_delta.as<float>(), in_delta.as<float>());
            ch_offset += input_channels[i];
        }
        return;
    }
#endif

    const float* delta = output_delta.as<float>();
    const Index pixels = output_delta.size() / total_channels;

    #pragma omp parallel for
    for (Index pixel = 0; pixel < pixels; ++pixel)
    {
        const float* delta_row = delta + pixel * total_channels;
        Index ch_offset = 0;
        for (size_t i = 0; i < input_channels.size(); ++i)
        {
            const Index in_c = input_channels[i];
            TensorView& in_delta = get_input_delta(back_propagation, layer, i);
            if (!in_delta.empty())
                memcpy(in_delta.as<float>() + pixel * in_c, delta_row + ch_offset, in_c * sizeof(float));
            ch_offset += in_c;
        }
    }
}

Concatenation::Concatenation(const Shape& new_input_shape,
                             const vector<Index>& per_input_channels,
                             const string& new_label)
    : Layer(LayerType::Concatenation)
{
    operators = {&concatenation};
    set(new_input_shape, per_input_channels, new_label);
}

Shape Concatenation::get_output_shape() const
{
    if (input_shape.empty()) return {};
    const Index total_channels = accumulate(concatenation.input_channels.begin(), concatenation.input_channels.end(), Index(0));
    return { input_shape[0], input_shape[1], total_channels };
}

vector<TensorSpec> Concatenation::get_backward_specs(Index batch_size) const
{
    vector<TensorSpec> specs;
    specs.reserve(concatenation.input_channels.size());
    for (const Index channels : concatenation.input_channels)
        specs.push_back({ Shape{batch_size, input_shape[0], input_shape[1], channels}, compute_dtype });
    return specs;
}

void Concatenation::set(const Shape& new_input_shape,
                        const vector<Index>& per_input_channels,
                        const string& new_label)
{
    check_rank(new_input_shape, {3}, "Concatenation", "input");

    input_shape = new_input_shape;
    concatenation.input_channels = per_input_channels;
    set_label(new_label);

    concatenation.input_delta_slots.resize(per_input_channels.size());
    iota(concatenation.input_delta_slots.begin(), concatenation.input_delta_slots.end(), size_t(1));

}

void Concatenation::set_input_shape(const Shape& new_input_shape)
{
    check_rank(new_input_shape, {3}, "Concatenation", "input");
    input_shape = new_input_shape;
}

const Json* Concatenation::legacy_body(const JsonDocument& document) const
{
    if (document.first_child(get_name())) return nullptr;
    return document.first_child("Concatenate");
}

void Concatenation::from_JSON(const JsonDocument& document)
{
    if (const Json* legacy = legacy_body(document))
    {
        Layer::from_JSON(JsonDocument::wrap(get_name(), *legacy));
        return;
    }

    Layer::from_JSON(document);
}

void Concatenation::load_state_from_JSON(const JsonDocument& document)
{
    if (const Json* legacy = legacy_body(document))
    {
        Layer::load_state_from_JSON(JsonDocument::wrap(get_name(), *legacy));
        return;
    }

    Layer::load_state_from_JSON(document);
}

void Concatenation::read_JSON_body(const Json* root)
{
    set(input_shape,
        parse_number_list<Index>(read_json_string(root, "InputChannels"), "InputChannels"),
        label);
}

void Concatenation::write_JSON_body(JsonWriter& writer) const
{
    add_json_field(writer, "InputChannels", vector_to_string(concatenation.input_channels, " "));
}

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
