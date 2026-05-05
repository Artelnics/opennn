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
{
    set(new_input_shape);
}

vector<pair<Shape, Type>> Scaling::get_forward_specs(Index batch_size) const
{
    return {{Shape{batch_size}.append(input_shape), compute_dtype}}; // Output
}

vector<pair<Shape, Type>> Scaling::get_state_specs() const
{
    const Index features = input_shape.size();
    if (features == 0) return {};
    return {
        {Shape{features}, Type::FP32}, // Minimums
        {Shape{features}, Type::FP32}, // Maximums
        {Shape{features}, Type::FP32}, // Means
        {Shape{features}, Type::FP32}, // StandardDeviations
        {Shape{features}, Type::FP32}, // Scalers
    };
}

VectorR Scaling::get_minimums() const
{
    return (ssize(states) > Minimums && states[Minimums].data) ? states[Minimums].as_vector() : VectorR();
}

VectorR Scaling::get_maximums() const
{
    return (ssize(states) > Maximums && states[Maximums].data) ? states[Maximums].as_vector() : VectorR();
}

VectorR Scaling::get_means() const
{
    return (ssize(states) > Means && states[Means].data) ? states[Means].as_vector() : VectorR();
}

VectorR Scaling::get_standard_deviations() const
{
    return (ssize(states) > StandardDeviations && states[StandardDeviations].data) ? states[StandardDeviations].as_vector() : VectorR();
}

void Scaling::set(const Shape& new_input_shape)
{
    input_shape = new_input_shape;
    is_trainable = false;
    name = "Scaling";
    layer_type = LayerType::Scaling;

    if (input_shape.empty())
        return;

    if (input_shape.rank != 1 && input_shape.rank != 2 && input_shape.rank != 3)
        throw runtime_error("Scaling layer supports input rank 1, 2 or 3 (got "
                            + to_string(input_shape.rank) + ").");

    const Index new_inputs_number = input_shape.size();

    scalers.assign(new_inputs_number, ScalerMethod::MeanStandardDeviation);

    label = "scaling_layer";

    set_min_max_range(-1.0f, 1.0f);
}

float* Scaling::link_states(float* pointer)
{
    const bool needs_defaults = ssize(states) < 5 || states[Means].data == nullptr;

    float* next = Layer::link_states(pointer);

    if (!needs_defaults || ssize(states) < 5) return next;

    if (states[Means].data)
        states[Means].as_vector().setZero();
    if (states[StandardDeviations].data)
        states[StandardDeviations].as_vector().setOnes();
    if (states[Minimums].data)
        states[Minimums].as_vector().setConstant(-1.0f);
    if (states[Maximums].data)
        states[Maximums].as_vector().setOnes();
    if (states[Scalers].data && ssize(scalers) == states[Scalers].size())
        for (size_t i = 0; i < scalers.size(); ++i)
            states[Scalers].as<float>()[i] = static_cast<float>(scalers[i]);

    return next;
}

void Scaling::set_input_shape(const Shape& new_input_shape)
{
    set(new_input_shape);
}

void Scaling::set_output_shape(const Shape& new_output_shape)
{
    set_input_shape(new_output_shape);
}

void Scaling::set_descriptives(const vector<Descriptives>& new_descriptives)
{
    if (ssize(states) < 5 || !states[Means].data)
        throw runtime_error("Scaling::set_descriptives: layer not compiled yet.");

    const Index descriptives_count = new_descriptives.size();
    if (descriptives_count != states[Means].size())
        throw runtime_error("Scaling::set_descriptives: size mismatch.");

    for (Index i = 0; i < descriptives_count; ++i)
    {
        states[Means].as<float>()[i]              = new_descriptives[i].mean;
        states[StandardDeviations].as<float>()[i] = new_descriptives[i].standard_deviation;
        states[Minimums].as<float>()[i]           = new_descriptives[i].minimum;
        states[Maximums].as<float>()[i]           = new_descriptives[i].maximum;
    }
}

void Scaling::set_scalers(const vector<string>& new_scalers)
{
    scalers.resize(new_scalers.size());
    for (size_t i = 0; i < new_scalers.size(); ++i)
        scalers[i] = string_to_scaler_method(new_scalers[i]);
    flush_scalers_to_states();
}

void Scaling::set_scalers(const string& new_scaler)
{
    const ScalerMethod method = string_to_scaler_method(new_scaler);
    for (auto& scaler : scalers)
        scaler = method;
    flush_scalers_to_states();
}

void Scaling::flush_scalers_to_states()
{
    if (ssize(states) <= Scalers || !states[Scalers].data) return;
    if (ssize(scalers) != states[Scalers].size()) return;
    for (size_t i = 0; i < scalers.size(); ++i)
        states[Scalers].as<float>()[i] = static_cast<float>(scalers[i]);
}

void Scaling::forward_propagate(ForwardPropagation& forward_propagation, size_t layer, bool) noexcept
{
    auto& forward_views = forward_propagation.views[layer];

    if (states.size() < 5)
    {
        copy(forward_views[Input][0], forward_views[Output][0]);
        return;
    }

    scale(forward_views[Input][0],
          states[Minimums], states[Maximums],
          states[Means], states[StandardDeviations],
          states[Scalers],
          min_range, max_range,
          forward_views[Output][0]);
}

string Scaling::write_no_scaling_expression(const vector<string>& input_names, const vector<string>& output_names) const
{
    const Index inputs_number = get_output_shape().size();

    ostringstream buffer;
    buffer.precision(10);

    for (Index i = 0; i < inputs_number; ++i)
        buffer << output_names[i] << " = " << input_names[i] << ";\n";

    return buffer.str();
}

string Scaling::write_minimum_maximum_expression(const vector<string>& input_names, const vector<string>& output_names) const
{
    ostringstream buffer;
    buffer.precision(10);

    const Index inputs_number = get_output_shape().size();
    const float* mins = states[Minimums].as<float>();
    const float* maxs = states[Maximums].as<float>();
    for (Index i = 0; i < inputs_number; ++i)
        buffer << output_names[i] << " = 2*(" << input_names[i] << "-(" << mins[i]
               << "))/(" << maxs[i] << "-(" << mins[i] << "))-1;\n";

    return buffer.str();
}

string Scaling::write_mean_standard_deviation_expression(const vector<string>& input_names, const vector<string>& output_names) const
{
    ostringstream buffer;
    buffer.precision(10);

    const Index inputs_number = get_output_shape().size();
    const float* mns = states[Means].as<float>();
    const float* sds = states[StandardDeviations].as<float>();
    for (Index i = 0; i < inputs_number; ++i)
        buffer << output_names[i] << " = (" << input_names[i] << "-(" << mns[i]
               << "))/" << sds[i] << ";\n";

    return buffer.str();
}

string Scaling::write_standard_deviation_expression(const vector<string>& input_names, const vector<string>& output_names) const
{
    ostringstream buffer;
    buffer.precision(10);

    const Index inputs_number = get_output_shape().size();
    const float* sds = states[StandardDeviations].as<float>();
    for (Index i = 0; i < inputs_number; ++i)
        buffer << output_names[i] << " = " << input_names[i] << "/(" << sds[i] << ");\n";

    return buffer.str();
}

void Scaling::from_JSON(const JsonDocument& document)
{
    const Json* scaling_layer_element = document.first_child("Scaling");
    if (!scaling_layer_element) throw runtime_error(name + " element is nullptr.");

    set(string_to_shape(read_json_string(scaling_layer_element, "InputDimensions")));

    const vector<string> scaler_names = get_tokens(read_json_string(scaling_layer_element, "Scalers"), " ");
    scalers.resize(scaler_names.size());
    for (size_t i = 0; i < scaler_names.size(); ++i)
        scalers[i] = string_to_scaler_method(scaler_names[i]);

    min_range = float(stof(read_json_string(scaling_layer_element, "MinRange")));
    max_range = float(stof(read_json_string(scaling_layer_element, "MaxRange")));
}

void Scaling::load_state_from_JSON(const JsonDocument& document)
{
    if (ssize(states) < 5 || !states[Means].data) return;

    const Json* scaling_layer_element = document.first_child("Scaling");
    if (!scaling_layer_element) throw runtime_error(name + " element is nullptr.");

    VectorR tmp;
    string_to_vector(read_json_string(scaling_layer_element, "Means"), tmp);
    if (tmp.size() == states[Means].size())
        states[Means].as_vector() = tmp;

    string_to_vector(read_json_string(scaling_layer_element, "StandardDeviations"), tmp);
    if (tmp.size() == states[StandardDeviations].size())
        states[StandardDeviations].as_vector() = tmp;

    string_to_vector(read_json_string(scaling_layer_element, "Minimums"), tmp);
    if (tmp.size() == states[Minimums].size())
        states[Minimums].as_vector() = tmp;

    string_to_vector(read_json_string(scaling_layer_element, "Maximums"), tmp);
    if (tmp.size() == states[Maximums].size())
        states[Maximums].as_vector() = tmp;
}

void Scaling::to_JSON(JsonWriter& printer) const
{
    printer.open_element("Scaling");

    vector<string> scaler_names(scalers.size());
    for (size_t i = 0; i < scalers.size(); ++i)
        scaler_names[i] = scaler_method_to_string(scalers[i]);

    write_json(printer, {
        {"InputDimensions", shape_to_string(input_shape)},
        {"Means", vector_to_string(states[Means].as_vector())},
        {"StandardDeviations", vector_to_string(states[StandardDeviations].as_vector())},
        {"Minimums", vector_to_string(states[Minimums].as_vector())},
        {"Maximums", vector_to_string(states[Maximums].as_vector())},
        {"Scalers", vector_to_string(scaler_names)},
        {"MinRange", to_string(min_range)},
        {"MaxRange", to_string(max_range)}
    });

    printer.close_element();
}

REGISTER(Layer, Scaling, "Scaling")

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
