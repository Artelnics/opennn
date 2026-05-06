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
    return unscale_op.minimums.data ? unscale_op.minimums.as_vector() : VectorR();
}

VectorR Unscaling::get_maximums() const
{
    return unscale_op.maximums.data ? unscale_op.maximums.as_vector() : VectorR();
}

VectorR Unscaling::get_means() const
{
    return unscale_op.means.data ? unscale_op.means.as_vector() : VectorR();
}

VectorR Unscaling::get_standard_deviations() const
{
    return unscale_op.standard_deviations.data ? unscale_op.standard_deviations.as_vector() : VectorR();
}

void Unscaling::set(Index new_neurons_number, const string& new_label)
{
    scalers.assign(new_neurons_number, ScalerMethod::MinimumMaximum);

    set_label(new_label);

    unscale_op.input_slots  = {Input};
    unscale_op.output_slots = {Output};
    unscale_op.set(new_neurons_number);

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
    if (!unscale_op.means.data)
        throw runtime_error("Unscaling::set_descriptives: layer not compiled yet.");

    const Index descriptives_count = new_descriptives.size();
    if (descriptives_count != unscale_op.means.size())
        throw runtime_error("Unscaling::set_descriptives: size mismatch.");

    for (Index i = 0; i < descriptives_count; ++i)
    {
        unscale_op.means.as<float>()[i]               = new_descriptives[i].mean;
        unscale_op.standard_deviations.as<float>()[i] = new_descriptives[i].standard_deviation;
        unscale_op.minimums.as<float>()[i]            = new_descriptives[i].minimum;
        unscale_op.maximums.as<float>()[i]            = new_descriptives[i].maximum;
    }
}

void Unscaling::set_min_max_range(float min, float max)
{
    unscale_op.min_range = min;
    unscale_op.max_range = max;
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

void Unscaling::forward_propagate(ForwardPropagation& forward_propagation, size_t layer, bool is_training) noexcept
{
    flush_scalers_to_states();
    for (Operator* op : get_operators())
        op->forward_propagate(forward_propagation, layer, is_training);
}
void Unscaling::print() const
{
    cout << "Unscaling layer" << "\n";
}

void Unscaling::read_JSON_body(const Json* root_element)
{
    const Json* neurons_array = root_element->find("Neurons");
    if (!neurons_array || !neurons_array->is_array()) return;

    const Index neurons_number = ssize(scalers);
    for (Index i = 0; i < neurons_number && size_t(i) < neurons_array->array_value.size(); ++i)
    {
        const Json* neuron = &neurons_array->array_value[size_t(i)];
        scalers[i] = string_to_scaler_method(read_json_string(neuron, "Scaler"));
    }
}

void Unscaling::write_JSON_body(JsonWriter& printer) const
{
    const Shape output_shape = get_output_shape();

    const bool have_state = unscale_op.means.data != nullptr;
    const float* mins = have_state ? unscale_op.minimums.as<float>()            : nullptr;
    const float* maxs = have_state ? unscale_op.maximums.as<float>()            : nullptr;
    const float* mns  = have_state ? unscale_op.means.as<float>()               : nullptr;
    const float* sds  = have_state ? unscale_op.standard_deviations.as<float>() : nullptr;

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
}
void Unscaling::flush_scalers_to_states()
{
    if (!unscale_op.scalers.data) return;
    if (ssize(scalers) != unscale_op.scalers.size()) return;
    for (size_t i = 0; i < scalers.size(); ++i)
        unscale_op.scalers.as<float>()[i] = static_cast<float>(scalers[i]);
}

REGISTER(Layer, Unscaling, "Unscaling")

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
