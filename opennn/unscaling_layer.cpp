//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   U N S C A L I N G   L A Y E R    C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "registry.h"
#include "string_utilities.h"
#include "tensor_utilities.h"
#include "math_utilities.h"
#include "neural_network.h"
#include "unscaling_layer.h"

namespace opennn
{

Unscaling::Unscaling(const Shape& new_input_shape, const string& new_label) : Layer()
{
    name = "Unscaling";
    layer_type = LayerType::Unscaling;
    is_trainable = false;

    set(new_input_shape.empty() ? Index(0) : new_input_shape[0], new_label);
}

// Getters

Shape Unscaling::get_input_shape() const
{
    return { ssize(scalers) };
}

Shape Unscaling::get_output_shape() const
{
    return { ssize(scalers) };
}

VectorR Unscaling::get_minimums() const
{
    return (ssize(states) > Minimums && states[Minimums].data) ? states[Minimums].as_vector() : VectorR();
}

VectorR Unscaling::get_maximums() const
{
    return (ssize(states) > Maximums && states[Maximums].data) ? states[Maximums].as_vector() : VectorR();
}

VectorR Unscaling::get_means() const
{
    return (ssize(states) > Means && states[Means].data) ? states[Means].as_vector() : VectorR();
}

VectorR Unscaling::get_standard_deviations() const
{
    return (ssize(states) > StandardDeviations && states[StandardDeviations].data) ? states[StandardDeviations].as_vector() : VectorR();
}

vector<pair<Shape, Type>> Unscaling::get_forward_specs(Index batch_size) const
{
    return {{Shape{batch_size}.append(get_output_shape()), Type::FP32}};
}

vector<pair<Shape, Type>> Unscaling::get_state_specs() const
{
    const Index features = ssize(scalers);
    if (features == 0) return {};
    return {
        {Shape{features}, Type::FP32}, // Minimums
        {Shape{features}, Type::FP32}, // Maximums
        {Shape{features}, Type::FP32}, // Means
        {Shape{features}, Type::FP32}, // StandardDeviations
        {Shape{features}, Type::FP32}, // Scalers
    };
}

// Setters

void Unscaling::set(Index new_neurons_number, const string& new_label)
{
    scalers.assign(new_neurons_number, ScalerMethod::MinimumMaximum);

    set_label(new_label);

    set_min_max_range(-1.0f, 1.0f);
}

void Unscaling::set_input_shape(const Shape& new_input_shape)
{
    set(new_input_shape[0]);
}

void Unscaling::set_output_shape(const Shape& /*new_output_shape*/)
{
}

void Unscaling::set_descriptives(const vector<Descriptives>& new_descriptives)
{
    if (ssize(states) < 5 || !states[Means].data)
        throw runtime_error("Unscaling::set_descriptives: layer not compiled yet.");

    const Index descriptives_count = new_descriptives.size();
    if (descriptives_count != states[Means].size())
        throw runtime_error("Unscaling::set_descriptives: size mismatch.");

    for (Index i = 0; i < descriptives_count; ++i)
    {
        states[Means].as<float>()[i]              = new_descriptives[i].mean;
        states[StandardDeviations].as<float>()[i] = new_descriptives[i].standard_deviation;
        states[Minimums].as<float>()[i]           = new_descriptives[i].minimum;
        states[Maximums].as<float>()[i]           = new_descriptives[i].maximum;
    }
}

void Unscaling::set_min_max_range(float min, float max)
{
    min_range = min;
    max_range = max;
}

void Unscaling::set_scalers(const vector<string>& new_scaler)
{
    scalers.resize(new_scaler.size());
    for (size_t i = 0; i < new_scaler.size(); ++i)
        scalers[i] = string_to_scaler_method(new_scaler[i]);
    flush_scalers_to_states();
}

void Unscaling::set_scalers(const string& new_scalers)
{
    const ScalerMethod method = string_to_scaler_method(new_scalers);
    for (auto& scaler : scalers)
        scaler = method;
    flush_scalers_to_states();
}

float* Unscaling::link_states(float* pointer)
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

// Forward propagation

void Unscaling::forward_propagate(ForwardPropagation& forward_propagation, size_t layer, bool) noexcept
{
    auto& forward_views = forward_propagation.views[layer];

    if (states.size() < 5)
    {
        copy(forward_views[Input][0], forward_views[Output][0]);
        return;
    }

    unscale(forward_views[Input][0],
            states[Minimums], states[Maximums],
            states[Means], states[StandardDeviations],
            states[Scalers],
            min_range, max_range,
            forward_views[Output][0]);
}

// Serialization

void Unscaling::print() const
{
    cout << "Unscaling layer" << "\n";
}

void Unscaling::from_JSON(const JsonDocument& document)
{
    const Json* root_element = get_json_root(document, "Unscaling");

    const Index neurons_number = read_json_index(root_element, "NeuronsNumber");

    set(neurons_number);

    const Json* neurons_array = root_element->find("Neurons");
    if (!neurons_array || !neurons_array->is_array()) return;

    for (Index i = 0; i < neurons_number && size_t(i) < neurons_array->array_value.size(); ++i)
    {
        const Json* neuron = &neurons_array->array_value[size_t(i)];
        scalers[i] = string_to_scaler_method(read_json_string(neuron, "Scaler"));
    }
}

void Unscaling::load_state_from_JSON(const JsonDocument& document)
{
    if (ssize(states) < 5 || !states[Means].data) return;

    const Json* root_element = get_json_root(document, "Unscaling");

    const Json* neurons_array = root_element->find("Neurons");
    if (!neurons_array || !neurons_array->is_array()) return;

    for (size_t i = 0; i < neurons_array->array_value.size() && Index(i) < states[Minimums].size(); ++i)
    {
        const Json* neuron = &neurons_array->array_value[i];
        const string descriptives = read_json_string(neuron, "Descriptives");
        const vector<string> tokens = get_tokens(descriptives, " ");
        if (tokens.size() >= 4)
        {
            states[Minimums].as<float>()[i]           = float(stof(tokens[0]));
            states[Maximums].as<float>()[i]           = float(stof(tokens[1]));
            states[Means].as<float>()[i]              = float(stof(tokens[2]));
            states[StandardDeviations].as<float>()[i] = float(stof(tokens[3]));
        }
    }
}

void Unscaling::to_JSON(JsonWriter& printer) const
{
    printer.open_element("Unscaling");

    const Shape output_shape = get_output_shape();
    add_json_field(printer, "NeuronsNumber", to_string(output_shape[0]));

    const bool have_state = (ssize(states) > StandardDeviations && states[Means].data);
    const float* mins = have_state ? states[Minimums].as<float>()           : nullptr;
    const float* maxs = have_state ? states[Maximums].as<float>()           : nullptr;
    const float* mns  = have_state ? states[Means].as<float>()              : nullptr;
    const float* sds  = have_state ? states[StandardDeviations].as<float>() : nullptr;

    printer.begin_array("Neurons");
    for (Index i = 0; i < output_shape[0]; ++i)
    {
        printer.begin_array_object();

        ostringstream descriptives_stream;
        descriptives_stream.precision(10);
        descriptives_stream << (mins ? mins[i] : -1.0f) << ' '
                            << (maxs ? maxs[i] : 1.0f)  << ' '
                            << (mns  ? mns[i]  : 0.0f)  << ' '
                            << (sds  ? sds[i]  : 1.0f);

        add_json_field(printer, "Descriptives", descriptives_stream.str());
        add_json_field(printer, "Scaler", scaler_method_to_string(scalers[i]));

        printer.end_array_object();
    }
    printer.end_array();

    printer.close_element();
}

// Helpers

void Unscaling::flush_scalers_to_states()
{
    if (ssize(states) <= Scalers || !states[Scalers].data) return;
    if (ssize(scalers) != states[Scalers].size()) return;
    for (size_t i = 0; i < scalers.size(); ++i)
        states[Scalers].as<float>()[i] = static_cast<float>(scalers[i]);
}

REGISTER(Layer, Unscaling, "Unscaling")

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
