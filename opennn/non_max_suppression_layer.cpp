//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   N O N   M A X   S U P P R E S S I O N   L A Y E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "registry.h"
#include "non_max_suppression_layer.h"
#include "json.h"

namespace opennn
{

NonMaxSuppression::NonMaxSuppression(const Shape& new_input_shape,
                                     Index new_boxes_per_cell,
                                     float new_confidence_threshold,
                                     float new_iou_threshold,
                                     const string& new_label)
    : Layer(LayerType::NonMaxSuppression, false)
{
    operators = {&nms};
    nms.input_slots = {Input};
    nms.output_slots = {Output};

    set(new_input_shape,
        new_boxes_per_cell,
        new_confidence_threshold,
        new_iou_threshold,
        new_label);
}

Shape NonMaxSuppression::get_output_shape() const
{
    if (input_shape.rank != 3) return {};
    return {input_shape[0] * input_shape[1] * nms.boxes_per_cell, 6};
}

void NonMaxSuppression::set(const Shape& new_input_shape,
                            Index new_boxes_per_cell,
                            float new_confidence_threshold,
                            float new_iou_threshold,
                            const string& new_label)
{
    if (!new_input_shape.empty())
        check_rank(new_input_shape, {3}, "NonMaxSuppression", "input");

    input_shape = new_input_shape;
    nms.boxes_per_cell = new_boxes_per_cell;
    nms.confidence_threshold = new_confidence_threshold;
    nms.iou_threshold = new_iou_threshold;
    set_label(new_label);
    configure_operator();
}

void NonMaxSuppression::set_input_shape(const Shape& new_input_shape)
{
    check_rank(new_input_shape, {3}, "NonMaxSuppression", "input");
    input_shape = new_input_shape;
    configure_operator();
}

void NonMaxSuppression::configure_operator()
{
    if (input_shape.empty()) return;
    nms.set(input_shape, nms.boxes_per_cell, nms.confidence_threshold, nms.iou_threshold);
}

void NonMaxSuppression::read_JSON_body(const Json* root)
{
    nms.boxes_per_cell = read_json_index(root, "BoxesPerCell");
    nms.confidence_threshold = read_json_type(root, "ConfidenceThreshold");
    nms.iou_threshold = read_json_type(root, "IouThreshold");
    configure_operator();
}

void NonMaxSuppression::write_JSON_body(JsonWriter& writer) const
{
    add_json_field(writer, "BoxesPerCell", to_string(nms.boxes_per_cell));
    add_json_field(writer, "ConfidenceThreshold", to_string(nms.confidence_threshold));
    add_json_field(writer, "IouThreshold", to_string(nms.iou_threshold));
}

REGISTER(Layer, NonMaxSuppression, "NonMaxSuppression")

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
