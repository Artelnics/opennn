//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   S C A L I N G   L A Y E R   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "registry.h"
#include "scaling_layer.h"
#include "math_utilities.h"
#include "forward_propagation.h"
#include "string_utilities.h"

namespace opennn
{

Scaling::Scaling(const Shape& new_input_shape)
    : Layer("Scaling", LayerType::Scaling, false)
{
    operators = {&scale_op};
    set(new_input_shape);
}
VectorR Scaling::get_minimums() const
{
    return scale_op.minimums.data ? scale_op.minimums.as_vector() : VectorR();
}

VectorR Scaling::get_maximums() const
{
    return scale_op.maximums.data ? scale_op.maximums.as_vector() : VectorR();
}

VectorR Scaling::get_means() const
{
    return scale_op.means.data ? scale_op.means.as_vector() : VectorR();
}

VectorR Scaling::get_standard_deviations() const
{
    return scale_op.standard_deviations.data ? scale_op.standard_deviations.as_vector() : VectorR();
}

void Scaling::set(const Shape& new_input_shape)
{
    input_shape = new_input_shape;

    set_label("scaling_layer");

    scale_op.input_slots  = {Input};
    scale_op.output_slots = {Output};
    scale_op.set(input_shape.size());

    if (input_shape.empty()) return;

    check_rank(input_shape, {1, 2, 3}, "Scaling", "input");

    scalers.assign(input_shape.size(), ScalerMethod::MeanStandardDeviation);

    set_min_max_range(-1.0f, 1.0f);
}

void Scaling::set_input_shape(const Shape& new_input_shape)
{
    set(new_input_shape);
}

void Scaling::set_descriptives(const vector<Descriptives>& new_descriptives)
{
    if (!scale_op.means.data)
        throw runtime_error("Scaling::set_descriptives: layer not compiled yet.");

    const Index descriptives_count = new_descriptives.size();
    if (descriptives_count != scale_op.means.size())
        throw runtime_error("Scaling::set_descriptives: size mismatch.");

    for (Index i = 0; i < descriptives_count; ++i)
    {
        scale_op.means.as<float>()[i]               = new_descriptives[i].mean;
        scale_op.standard_deviations.as<float>()[i] = new_descriptives[i].standard_deviation;
        scale_op.minimums.as<float>()[i]            = new_descriptives[i].minimum;
        scale_op.maximums.as<float>()[i]            = new_descriptives[i].maximum;
    }
}

void Scaling::set_scalers(const vector<string>& new_scalers)
{
    scalers.resize(new_scalers.size());
    transform(new_scalers.begin(), new_scalers.end(), scalers.begin(), string_to_scaler_method);
    flush_scalers_to_states();
}

void Scaling::set_scalers(const string& new_scaler)
{
    const ScalerMethod method = string_to_scaler_method(new_scaler);
    for (auto& scaler : scalers)
        scaler = method;
    flush_scalers_to_states();
}

void Scaling::forward_propagate(ForwardPropagation& forward_propagation, size_t layer, bool is_training) noexcept
{
    flush_scalers_to_states();
    for (Operator* op : get_operators())
        op->forward_propagate(forward_propagation, layer, is_training);
}

void Scaling::read_JSON_body(const Json* scaling_layer_element)
{
    const vector<string> scaler_names = get_tokens(read_json_string(scaling_layer_element, "Scalers"), " ");
    scalers.resize(scaler_names.size());
    transform(scaler_names.begin(), scaler_names.end(), scalers.begin(), string_to_scaler_method);

    scale_op.min_range = float(stof(read_json_string(scaling_layer_element, "MinRange")));
    scale_op.max_range = float(stof(read_json_string(scaling_layer_element, "MaxRange")));
}

void Scaling::write_JSON_body(JsonWriter& printer) const
{
    vector<string> scaler_names(scalers.size());
    transform(scalers.begin(), scalers.end(), scaler_names.begin(), scaler_method_to_string);

    write_json(printer, {
        {"Means", vector_to_string(scale_op.means.as_vector())},
        {"StandardDeviations", vector_to_string(scale_op.standard_deviations.as_vector())},
        {"Minimums", vector_to_string(scale_op.minimums.as_vector())},
        {"Maximums", vector_to_string(scale_op.maximums.as_vector())},
        {"Scalers", vector_to_string(scaler_names)},
        {"MinRange", to_string(scale_op.min_range)},
        {"MaxRange", to_string(scale_op.max_range)}
    });
}
void Scaling::flush_scalers_to_states()
{
    if (!scale_op.scalers.data) return;
    if (ssize(scalers) != scale_op.scalers.size()) return;
    for (size_t i = 0; i < scalers.size(); ++i)
        scale_op.scalers.as<float>()[i] = static_cast<float>(scalers[i]);
}

string Scaling::write_expression(const vector<string>& input_names,
                                 const vector<string>& /*output_names*/) const
{
    ostringstream buffer;
    buffer.precision(10);

    const Index outputs_number = get_outputs_number();
    const VectorR& minimums = get_minimums();
    const VectorR& maximums = get_maximums();
    const VectorR& means = get_means();
    const VectorR& standard_deviations = get_standard_deviations();
    const vector<ScalerMethod>& scalers_local = get_scalers();
    const float min_range = get_min_range();
    const float max_range = get_max_range();

    for (Index i = 0; i < outputs_number; ++i)
    {
        switch (scalers_local[i])
        {
        case ScalerMethod::None:
            buffer << "scaled_" << input_names[i] << " = " << input_names[i] << ";\n";
            break;
        case ScalerMethod::MinimumMaximum:
            buffer << "scaled_" << input_names[i]
                   << " = " << input_names[i] << "*(" << max_range << "-" << min_range << ")/("
                   << maximums[i] << "-(" << minimums[i] << "))-" << minimums[i] << "*("
                   << max_range << "-" << min_range << ")/("
                   << maximums[i] << "-" << minimums[i] << ")+" << min_range << ";\n";
            break;
        case ScalerMethod::MeanStandardDeviation:
            buffer << "scaled_" << input_names[i] << " = (" << input_names[i] << "-" << means[i] << ")/" << standard_deviations[i] << ";\n";
            break;
        case ScalerMethod::StandardDeviation:
            buffer << "scaled_" << input_names[i] << " = " << input_names[i] << "/(" << standard_deviations[i] << ");\n";
            break;
        case ScalerMethod::Logarithm:
            buffer << "scaled_" << input_names[i] << " = log(" << input_names[i] << ");\n";
            break;
        default:
            throw runtime_error("Unknown inputs scaling method.\n");
        }
    }

    string expression = buffer.str();

    expression = regex_replace(expression, regex("\\+-"), "-");
    expression = regex_replace(expression, regex("--"), "+");

    return expression;
}

REGISTER(Layer, Scaling, "Scaling")

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
