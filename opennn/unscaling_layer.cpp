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
    const Index neurons_number = means.size();

    return {neurons_number};
}

Shape Unscaling::get_output_shape() const
{
    const Index neurons_number = means.size();

    return { neurons_number };
}

// Setters

void Unscaling::set(const Index new_neurons_number, const string& new_label)
{
    means = VectorR::Zero(new_neurons_number);
    standard_deviations.resize(new_neurons_number);
    standard_deviations.setOnes();
    minimums.resize(new_neurons_number);
    minimums.setConstant(type(-1.0));
    maximums.resize(new_neurons_number);
    maximums.setOnes();

    scalers.resize(new_neurons_number, ScalerMethod::MinimumMaximum);

    label = new_label;

    set_scalers("MinimumMaximum");

    set_min_max_range(type(-1), type(1));

    name = "Unscaling";
    layer_type = LayerType::Unscaling;

    is_trainable = false;
}

void Unscaling::set_input_shape(const Shape& new_input_shape)
{
    means.resize(new_input_shape[0]);
    standard_deviations.resize(new_input_shape[0]);
    minimums.resize(new_input_shape[0]);
    maximums.resize(new_input_shape[0]);
}

void Unscaling::set_output_shape(const Shape& /*new_output_shape*/)
{
}

void Unscaling::set_descriptives(const vector<Descriptives>& new_descriptives)
{
    const Index n = new_descriptives.size();
    means.resize(n);
    standard_deviations.resize(n);
    minimums.resize(n);
    maximums.resize(n);

    for(Index i = 0; i < n; ++i)
    {
        means[i] = new_descriptives[i].mean;
        standard_deviations[i] = new_descriptives[i].standard_deviation;
        minimums[i] = new_descriptives[i].minimum;
        maximums[i] = new_descriptives[i].maximum;
    }

    write_states();
}

void Unscaling::set_min_max_range(const type min, const type max)
{
    min_range = min;
    max_range = max;
}

void Unscaling::set_scalers(const vector<string>& new_scaler)
{
    scalers.resize(new_scaler.size());
    for(size_t i = 0; i < new_scaler.size(); ++i)
        scalers[i] = string_to_scaler_method(new_scaler[i]);
    write_states();
}

void Unscaling::set_scalers(const string& new_scalers)
{
    const ScalerMethod method = string_to_scaler_method(new_scalers);
    for(auto& scaler : scalers)
        scaler = method;
    write_states();
}

type* Unscaling::link_states(type* pointer)
{
    type* next = Layer::link_states(pointer);
    write_states();
    return next;
}

void Unscaling::write_states()
{
    if (states.size() < 5) return;

    if (means.size() == states[Means].size() && states[Means].data)
        VectorMap(states[Means].data, states[Means].size()) = means;
    if (standard_deviations.size() == states[StandardDeviations].size() && states[StandardDeviations].data)
        VectorMap(states[StandardDeviations].data, states[StandardDeviations].size()) = standard_deviations;
    if (minimums.size() == states[Minimums].size() && states[Minimums].data)
        VectorMap(states[Minimums].data, states[Minimums].size()) = minimums;
    if (maximums.size() == states[Maximums].size() && states[Maximums].data)
        VectorMap(states[Maximums].data, states[Maximums].size()) = maximums;
    if (ssize(scalers) == states[Scalers].size() && states[Scalers].data)
        for (size_t i = 0; i < scalers.size(); ++i)
            states[Scalers].data[i] = static_cast<type>(scalers[i]);
}

// Forward propagation

void Unscaling::forward_propagate(ForwardPropagation& forward_propagation, size_t layer, bool) noexcept
{
    auto& forward_views = forward_propagation.views[layer];

    // Unscaling has is_trainable=false and sits after the last trainable layer, so
    // NeuralNetwork::forward_propagate skips it entirely during training. This path
    // runs only for validation during training and for inference.
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

        const XmlElement* descriptives_element = unscaling_neuron_element->first_child_element("Descriptives");
        if(descriptives_element && descriptives_element->get_text())
        {
            const vector<string> tokens = get_tokens(descriptives_element->get_text(), " ");
            if(tokens.size() >= 4)
            {
                minimums(i)            = type(stof(tokens[0]));
                maximums(i)            = type(stof(tokens[1]));
                means(i)               = type(stof(tokens[2]));
                standard_deviations(i) = type(stof(tokens[3]));
            }
        }

        scalers[i] = string_to_scaler_method(read_xml_string(unscaling_neuron_element, "Scaler"));

        start_element = unscaling_neuron_element;
    }
}

void Unscaling::to_XML(XmlPrinter& printer) const
{
    printer.open_element("Unscaling");

    const Shape output_shape = get_output_shape();

    add_xml_element(printer, "NeuronsNumber", to_string(output_shape[0]));

    for(Index i = 0; i < output_shape[0]; ++i)
    {
        printer.open_element("UnscalingNeuron");
        printer.push_attribute("Index", int(i + 1));

        ostringstream descriptives_stream;
        descriptives_stream.precision(10);
        descriptives_stream << minimums(i) << ' '
                            << maximums(i) << ' '
                            << means(i) << ' '
                            << standard_deviations(i);
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
