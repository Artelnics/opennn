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

Unscaling::Unscaling(const Shape& new_input_shape, const string& label)
    : Layer()
{
    set(new_input_shape[0], label);
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

// Setters

void Unscaling::set(const Index new_neurons_number, const string& new_label)
{
    scalers.assign(new_neurons_number, ScalerMethod::MinimumMaximum);

    label = new_label;

    set_min_max_range(float(-1), float(1));

    name = "Unscaling";
    layer_type = LayerType::Unscaling;

    is_trainable = false;
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
    if(ssize(states) < 5 || !states[Means].data)
        throw runtime_error("Unscaling::set_descriptives: layer not compiled yet.");

    const Index descriptives_count = new_descriptives.size();
    if(descriptives_count != states[Means].size())
        throw runtime_error("Unscaling::set_descriptives: size mismatch.");

    for(Index i = 0; i < descriptives_count; ++i)
    {
        states[Means].as<float>()[i]              = new_descriptives[i].mean;
        states[StandardDeviations].as<float>()[i] = new_descriptives[i].standard_deviation;
        states[Minimums].as<float>()[i]           = new_descriptives[i].minimum;
        states[Maximums].as<float>()[i]           = new_descriptives[i].maximum;
    }
}

void Unscaling::set_min_max_range(const float min, const float max)
{
    min_range = min;
    max_range = max;
}

void Unscaling::set_scalers(const vector<string>& new_scaler)
{
    scalers.resize(new_scaler.size());
    for(size_t i = 0; i < new_scaler.size(); ++i)
        scalers[i] = string_to_scaler_method(new_scaler[i]);
    flush_scalers_to_states();
}

void Unscaling::set_scalers(const string& new_scalers)
{
    const ScalerMethod method = string_to_scaler_method(new_scalers);
    for(auto& scaler : scalers)
        scaler = method;
    flush_scalers_to_states();
}

float* Unscaling::link_states(float* pointer)
{
    float* next = Layer::link_states(pointer);

    if(ssize(states) < 5) return next;

    if(states[Means].data)
        VectorMap(states[Means].as<float>(), states[Means].size()).setZero();
    if(states[StandardDeviations].data)
        VectorMap(states[StandardDeviations].as<float>(), states[StandardDeviations].size()).setOnes();
    if(states[Minimums].data)
        VectorMap(states[Minimums].as<float>(), states[Minimums].size()).setConstant(float(-1));
    if(states[Maximums].data)
        VectorMap(states[Maximums].as<float>(), states[Maximums].size()).setOnes();
    if(states[Scalers].data && ssize(scalers) == states[Scalers].size())
        for(size_t i = 0; i < scalers.size(); ++i)
            states[Scalers].as<float>()[i] = static_cast<float>(scalers[i]);

    return next;
}

void Unscaling::flush_scalers_to_states()
{
    if(ssize(states) <= Scalers || !states[Scalers].data) return;
    if(ssize(scalers) != states[Scalers].size()) return;
    for(size_t i = 0; i < scalers.size(); ++i)
        states[Scalers].as<float>()[i] = static_cast<float>(scalers[i]);
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

void Unscaling::from_XML(const XmlDocument& document)
{
    const XmlElement* root_element = get_xml_root(document, "Unscaling");

    const Index neurons_number = read_xml_index(root_element, "NeuronsNumber");

    set(neurons_number);

    const XmlElement* start_element = root_element->first_child_element("NeuronsNumber");

    for(Index i = 0; i < neurons_number; ++i) {
        const XmlElement* unscaling_neuron_element = start_element->next_sibling_element("UnscalingNeuron");
        if(!unscaling_neuron_element) {
            throw runtime_error("Unscaling neuron " + to_string(i + 1) + " is nullptr.\n");
        }

        unsigned index = 0;
        unscaling_neuron_element->query_unsigned_attribute("Index", &index);
        if (index != i + 1) {
            throw runtime_error("Index " + to_string(index) + " is not correct.\n");
        }

        scalers[i] = string_to_scaler_method(read_xml_string(unscaling_neuron_element, "Scaler"));

        start_element = unscaling_neuron_element;
    }
}

void Unscaling::load_state_from_XML(const XmlDocument& document)
{
    if(ssize(states) < 5 || !states[Means].data) return;

    const XmlElement* root_element = get_xml_root(document, "Unscaling");

    const Index neurons_number = read_xml_index(root_element, "NeuronsNumber");

    const XmlElement* start_element = root_element->first_child_element("NeuronsNumber");

    for(Index i = 0; i < neurons_number; ++i) {
        const XmlElement* unscaling_neuron_element = start_element->next_sibling_element("UnscalingNeuron");
        if(!unscaling_neuron_element) break;

        const XmlElement* descriptives_element = unscaling_neuron_element->first_child_element("Descriptives");
        if(descriptives_element && descriptives_element->get_text())
        {
            const vector<string> tokens = get_tokens(descriptives_element->get_text(), " ");
            if(tokens.size() >= 4 && i < states[Minimums].size())
            {
                states[Minimums].as<float>()[i]           = float(stof(tokens[0]));
                states[Maximums].as<float>()[i]           = float(stof(tokens[1]));
                states[Means].as<float>()[i]              = float(stof(tokens[2]));
                states[StandardDeviations].as<float>()[i] = float(stof(tokens[3]));
            }
        }

        start_element = unscaling_neuron_element;
    }
}

void Unscaling::to_XML(XmlPrinter& printer) const
{
    printer.open_element("Unscaling");

    const Shape output_shape = get_output_shape();

    add_xml_element(printer, "NeuronsNumber", to_string(output_shape[0]));

    const bool have_state = (ssize(states) > StandardDeviations && states[Means].data);
    const float* mins = have_state ? states[Minimums].as<float>()           : nullptr;
    const float* maxs = have_state ? states[Maximums].as<float>()           : nullptr;
    const float* mns  = have_state ? states[Means].as<float>()              : nullptr;
    const float* sds  = have_state ? states[StandardDeviations].as<float>() : nullptr;

    for(Index i = 0; i < output_shape[0]; ++i)
    {
        printer.open_element("UnscalingNeuron");
        printer.push_attribute("Index", int(i + 1));

        ostringstream descriptives_stream;
        descriptives_stream.precision(10);
        descriptives_stream << (mins ? mins[i] : float(-1)) << ' '
                            << (maxs ? maxs[i] : float(1))  << ' '
                            << (mns  ? mns[i]  : float(0))  << ' '
                            << (sds  ? sds[i]  : float(1));
        add_xml_element(printer, "Descriptives", descriptives_stream.str());

        add_xml_element(printer, "Scaler", scaler_method_to_string(scalers[i]));

        printer.close_element();
    }

    printer.close_element();
}

REGISTER(Layer, Unscaling, "Unscaling")

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
