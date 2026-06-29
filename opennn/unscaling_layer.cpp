//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   U N S C A L I N G   L A Y E R    C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "registry.h"
#include "string_utilities.h"
#include "unscaling_layer.h"
#include "json.h"

namespace opennn
{

Unscaling::Unscaling(const Shape& new_input_shape, const string& new_label)
    : Scaling(LayerType::Unscaling)
{
    scale_op.invert = true;
    set(new_input_shape.dim_or_zero(0), new_label);
}

void Unscaling::set(Index new_neurons_number, const string& new_label)
{
    set_label(new_label);

    descriptives.assign(size_t(new_neurons_number), Descriptives(-1.0f, 1.0f, 0.0f, 1.0f));
    scalers.assign(size_t(new_neurons_number), ScalerMethod::MinimumMaximum);
    min_range = -1.0f;
    max_range = 1.0f;
    op_storage_dirty = true;
}

void Unscaling::set_input_shape(const Shape& new_input_shape)
{
    set(new_input_shape.dim_or_zero(0));
}

void Unscaling::read_JSON_body(const Json* root_element)
{
    if (!root_element) return;

    if (root_element->has("MinRange"))
        min_range = parse_float(read_json_string(root_element, "MinRange"), "Unscaling: MinRange");
    if (root_element->has("MaxRange"))
        max_range = parse_float(read_json_string(root_element, "MaxRange"), "Unscaling: MaxRange");

    const Json* neurons_array = root_element->find("Neurons");
    if (!neurons_array || !neurons_array->is_array()) return;

    throw_if(ssize(neurons_array->array_value) != ssize(scalers),
             format("Unscaling::read_JSON_body: \"Neurons\" has {} entries, expected {}.",
                    neurons_array->array_value.size(), scalers.size()));

    for (size_t i = 0; i < neurons_array->array_value.size(); ++i)
    {
        const Json* neuron = &neurons_array->array_value[i];

        scalers[i] = string_to_scaler_method(read_json_string(neuron, "Scaler"));

        const string descriptives_text = read_json_string(neuron, "Descriptives");
        const vector<string> tokens = get_tokens(descriptives_text, " ");
        throw_if(tokens.size() < 4,
                 format("Unscaling::read_JSON_body: neuron {} \"Descriptives\" has {} tokens, expected 4.",
                        i, tokens.size()));
        descriptives[i].minimum            = parse_float(tokens[0], "Unscaling: Descriptives");
        descriptives[i].maximum            = parse_float(tokens[1], "Unscaling: Descriptives");
        descriptives[i].mean               = parse_float(tokens[2], "Unscaling: Descriptives");
        descriptives[i].standard_deviation = parse_float(tokens[3], "Unscaling: Descriptives");
    }

    op_storage_dirty = true;
    refresh_op_storage(op_storage_device);
}

void Unscaling::write_JSON_body(JsonWriter& printer) const
{
    add_json_field(printer, "MinRange", to_string(min_range));
    add_json_field(printer, "MaxRange", to_string(max_range));

    const Index features = ssize(descriptives);

    printer.begin_array("Neurons");
    for (Index i = 0; i < features; ++i)
    {
        printer.begin_array_object();

        ostringstream descriptives_stream;
        descriptives_stream.precision(10);
        descriptives_stream << descriptives[size_t(i)].minimum << ' '
                            << descriptives[size_t(i)].maximum << ' '
                            << descriptives[size_t(i)].mean    << ' '
                            << descriptives[size_t(i)].standard_deviation;

        add_json_field(printer, "Descriptives", descriptives_stream.str());
        add_json_field(printer, "Scaler", scaler_method_to_string(scalers[size_t(i)]));

        printer.end_array_object();
    }
    printer.end_array();
}

string Unscaling::write_expression(const vector<string>& input_names,
                                   const vector<string>& output_names) const
{
    const Index outputs_number = get_outputs_number();
    throw_if(outputs_number == 0 || ssize(scalers) != outputs_number,
             "Unscaling::write_expression: layer not configured.");

    ostringstream buffer;
    buffer.precision(10);

    for (Index i = 0; i < outputs_number; ++i)
    {
        const Descriptives& descriptive = descriptives[size_t(i)];
        using enum ScalerMethod;
        switch (scalers[size_t(i)])
        {
        case None:
            buffer << output_names[i] << " = " << input_names[i] << ";\n";
            break;
        case MinimumMaximum:
            if (abs(descriptive.minimum - descriptive.maximum) < EPSILON)
                buffer << output_names[i] << "=" << descriptive.minimum << ";\n";
            else
                buffer << output_names[i] << "=" << input_names[i] << "*"
                       << "(" << (descriptive.maximum - descriptive.minimum) / (max_range - min_range)
                       << ")+" << (descriptive.minimum - min_range * (descriptive.maximum - descriptive.minimum) / (max_range - min_range)) << ";\n";
            break;
        case MeanStandardDeviation:
            buffer << output_names[i] << "=" << input_names[i] << "*" << descriptive.standard_deviation << "+" << descriptive.mean << ";\n";
            break;
        case StandardDeviation:
            buffer << output_names[i] << "=" << input_names[i] << "*" << descriptive.standard_deviation << ";\n";
            break;
        case Logarithm:
            buffer << output_names[i] << "=" << "exp(" << input_names[i] << ");\n";
            break;
        case ImageMinMax:
            buffer << output_names[i] << "=" << input_names[i] << " * 255.0;\n";
            break;
        default:
            throw runtime_error("Unknown inputs scaling method.\n");
        }
    }

    string expression = buffer.str();
    replace(expression, "+-", "-");
    replace(expression, "--", "+");

    return expression;
}

REGISTER(Layer, Unscaling, "Unscaling")

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
