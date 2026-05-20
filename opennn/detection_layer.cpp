//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   D E T E C T I O N   L A Y E R   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "registry.h"
#include "detection_layer.h"
#include "json.h"

namespace opennn
{

namespace
{

string anchors_to_string(const vector<array<float, 2>>& anchors)
{
    ostringstream buffer;
    for (size_t i = 0; i < anchors.size(); ++i)
    {
        if (i != 0) buffer << ' ';
        buffer << anchors[i][0] << ' ' << anchors[i][1];
    }
    return buffer.str();
}

vector<array<float, 2>> string_to_anchors(const string& text)
{
    istringstream stream(text);
    vector<array<float, 2>> anchors;
    float width = 0.0f;
    float height = 0.0f;

    while (stream >> width >> height)
        anchors.push_back({width, height});

    return anchors;
}

}

Detection::Detection(const Shape& new_input_shape,
                     const vector<array<float, 2>>& new_anchors,
                     const string& new_label)
    : Layer(LayerType::Detection)
{
    operators = {&detection};
    detection.input_slots = {Input};
    detection.output_slots = {Output};
    detection.input_delta_slots = {InputDelta};
    detection.output_delta_slots = {OutputDelta};

    set(new_input_shape, new_anchors, new_label);
}

void Detection::set(const Shape& new_input_shape,
                    const vector<array<float, 2>>& new_anchors,
                    const string& new_label)
{
    if (!new_input_shape.empty())
        check_rank(new_input_shape, {3}, "Detection", "input");

    input_shape = new_input_shape;
    set_label(new_label);
    detection.anchors = new_anchors;
    configure_operator();
}

void Detection::set_input_shape(const Shape& new_input_shape)
{
    check_rank(new_input_shape, {3}, "Detection", "input");
    input_shape = new_input_shape;
    configure_operator();
}

void Detection::set_anchors(const vector<array<float, 2>>& new_anchors)
{
    detection.anchors = new_anchors;
    configure_operator();
}

void Detection::configure_operator()
{
    if (input_shape.empty()) return;
    detection.set(input_shape, detection.anchors);
}

void Detection::read_JSON_body(const Json* root)
{
    set_anchors(string_to_anchors(read_json_string(root, "Anchors")));
}

void Detection::write_JSON_body(JsonWriter& writer) const
{
    add_json_field(writer, "Anchors", anchors_to_string(detection.anchors));
}

REGISTER(Layer, Detection, "Detection")

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
