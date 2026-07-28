//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   D E T E C T I O N   V 8   L A Y E R   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "registry.h"
#include "detection_v8_layer.h"
#include "json.h"

namespace opennn
{

DetectionV8::DetectionV8(const Shape& new_input_shape, const string& new_label)
    : Layer(LayerType::DetectionV8)
{
    operators = {&detection};
    set(new_input_shape, new_label);
}

DetectionV8::DetectionV8(const Shape& new_input_shape, Index reg_max, const string& new_label)
    : Layer(LayerType::DetectionV8)
{
    operators = {&detection};
    set(new_input_shape, reg_max, new_label);
}

void DetectionV8::set(const Shape& new_input_shape, const string& new_label)
{
    set(new_input_shape, 1, new_label);
}

void DetectionV8::set(const Shape& new_input_shape, Index reg_max, const string& new_label)
{
    if (!new_input_shape.empty())
        check_rank(new_input_shape, {3}, "DetectionV8", "input");

    input_shape = new_input_shape;
    set_label(new_label);
    detection.reg_max = max(Index(1), reg_max);
    configure_operator();
}

void DetectionV8::set_input_shape(const Shape& new_input_shape)
{
    check_rank(new_input_shape, {3}, "DetectionV8", "input");
    input_shape = new_input_shape;
    configure_operator();
}

void DetectionV8::configure_operator()
{
    if (input_shape.empty()) return;
    detection.set(input_shape, detection.reg_max);
}

void DetectionV8::read_JSON_body(const Json* root)
{
    const Index classes  = read_json_index(root, "ClassesNumber");
    const Index gs       = read_json_index(root, "GridSize");
    const Index gw       = read_json_index(root, "GridWidth");
    const Json* rm_node  = root ? root->find("RegMax") : nullptr;
    const Index rm       = rm_node ? Index(read_json_index(root, "RegMax")) : Index(1);
    detection.reg_max    = max(Index(1), rm);
    const Index box_ch   = 4 * detection.reg_max;
    input_shape          = Shape{gs, gw, box_ch + classes};
    configure_operator();
}

void DetectionV8::write_JSON_body(JsonWriter& writer) const
{
    add_json_field(writer, "ClassesNumber", to_string(detection.classes_number));
    add_json_field(writer, "GridSize",      to_string(detection.grid_size));
    add_json_field(writer, "GridWidth",     to_string(detection.grid_width));
    add_json_field(writer, "RegMax",        to_string(detection.reg_max));
}

REGISTER(Layer, DetectionV8, "DetectionV8")

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
