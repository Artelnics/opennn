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




string Unscaling::get_expression(const vector<string>& new_input_names,
                                 const vector<string>& new_output_names) const
{
    const vector<string> input_names = new_input_names.empty()
        ? get_default_feature_names()
        : new_input_names;

    const vector<string> output_names = new_output_names.empty()
        ? get_default_output_names()
        : new_output_names;

    const Index outputs_number = get_outputs_number();

    ostringstream buffer;

    buffer.precision(10);
/*
    for(Index i = 0; i < outputs_number; i++)
    {
        const string& scaler = scalers[i];

        if(scaler == "None")
            buffer << output_names[i] << " = " << input_names[i] << ";\n";
        else if(scaler == "MinimumMaximum")
            if(abs(descriptives[i].minimum - descriptives[i].maximum) < EPSILON)
                buffer << output_names[i] << "=" << descriptives[i].minimum <<";\n";
            else
                buffer << output_names[i] << "=" << input_names[i] << "*(" << (descriptives[i].maximum - descriptives[i].minimum)/(max_range - min_range)
                << ")+" << (descriptives[i].minimum - min_range*(descriptives[i].maximum - descriptives[i].minimum)/(max_range - min_range)) << ";\n";
        else if(scaler == "MeanStandardDeviation")
            buffer << output_names[i] << "=" << input_names[i] << "*" << descriptives[i].standard_deviation <<"+"<< descriptives[i].mean <<";\n";
        else if(scaler == "StandardDeviation")
            buffer << output_names[i] << "=" <<  input_names[i] << "*" << descriptives[i].standard_deviation <<";\n";
        else if(scaler == "Logarithm")
            buffer << output_names[i] << "=" << "exp(" << input_names[i] << ");\n";
        else
            throw runtime_error("Unknown inputs scaling method.\n");
    }
*/
    string expression = buffer.str();

    replace(expression, "+-", "-");
    replace(expression, "--", "+");

    return expression;
}


void Unscaling::set_input_shape(const Shape& new_input_shape)
{
    means.resize(new_input_shape[0]);
    standard_deviations.resize(new_input_shape[0]);
    minimums.resize(new_input_shape[0]);
    maximums.resize(new_input_shape[0]);
}


void Unscaling::set_output_shape(const Shape& new_output_shape)
{
}


void Unscaling::set(const Index new_neurons_number, const string& new_label)
{
    means.resize(new_neurons_number);
    means.setZero();
    standard_deviations.resize(new_neurons_number);
    standard_deviations.setOnes();
    minimums.resize(new_neurons_number);
    minimums.setConstant(type(-1.0));
    maximums.resize(new_neurons_number);
    maximums.setOnes();
    multipliers.resize(new_neurons_number);
    offsets.resize(new_neurons_number);

    scalers.resize(new_neurons_number, "MinimumMaximum");

    label = new_label;

    set_scalers("MinimumMaximum");

    set_min_max_range(type(-1), type(1));

    calculate_coefficients();

    name = "Unscaling";

    is_trainable = false;
}


void Unscaling::set_min_max_range(const type min, const type max)
{
    min_range = min;
    max_range = max;
}


void Unscaling::set_descriptives(const vector<Descriptives>& new_descriptives)
{
    const Index n = new_descriptives.size();
    means.resize(n);
    standard_deviations.resize(n);
    minimums.resize(n);
    maximums.resize(n);
    multipliers.resize(n);
    offsets.resize(n);

    for(Index i = 0; i < n; ++i)
    {
        means[i] = new_descriptives[i].mean;
        standard_deviations[i] = new_descriptives[i].standard_deviation;
        minimums[i] = new_descriptives[i].minimum;
        maximums[i] = new_descriptives[i].maximum;
    }

    calculate_coefficients();
}


void Unscaling::set_scalers(const vector<string>& new_scaler)
{
    scalers = new_scaler;
}


void Unscaling::set_scalers(const string& new_scalers)
{
    for(string& scaler : scalers)
        scaler = new_scalers;
}


void Unscaling::calculate_coefficients()
{
    const Index n = scalers.size();
    for(Index i = 0; i < n; ++i)
    {
        const string& method = scalers[i];
        if(method == "MeanStandardDeviation") {
            multipliers[i] = standard_deviations[i] + EPSILON;
            offsets[i] = means[i];
        }
        else if(method == "MinimumMaximum") {
            const type range = (maximums[i] - minimums[i]) + EPSILON;
            multipliers[i] = range / (max_range - min_range);
            offsets[i] = minimums[i] - min_range * multipliers[i];
        }
        else { // None
            multipliers[i] = 1.0f;
            offsets[i] = 0.0f;
        }
    }
}


void Unscaling::forward_propagate(ForwardPropagation& forward_propagation, size_t layer, bool)
{
    const TensorView& input = forward_propagation.views[layer][0][0];
    TensorView& output = forward_propagation.views[layer][1][0];

    // Data targets are scaled in-place by Optimizer::set_scaling(),
    // so the unscaling layer just copies input to output.
    // The unscaling coefficients are stored for serialization/expression only.
    copy(input, output);
}


void Unscaling::print() const
{
    cout << "Unscaling layer" << endl;
}


void Unscaling::to_XML(XMLPrinter& printer) const
{
    printer.OpenElement("Unscaling");

    const Shape output_shape = get_output_shape();

    add_xml_element(printer, "NeuronsNumber", to_string(output_shape[0]));

    for(Index i = 0; i < output_shape[0]; i++)
    {
        printer.OpenElement("UnscalingNeuron");
        printer.PushAttribute("Index", int(i + 1));
        //add_xml_element(printer, "Descriptives", vector_to_string(descriptives[i].to_tensor()));
        add_xml_element(printer, "Scaler", scalers[i]);

        printer.CloseElement();
    }

    printer.CloseElement();
}


void Unscaling::from_XML(const XMLDocument& document)
{
    const XMLElement* root_element = get_xml_root(document, "Unscaling");

    const Index neurons_number = read_xml_index(root_element, "NeuronsNumber");

    set(neurons_number);

    const XMLElement* start_element = root_element->FirstChildElement("NeuronsNumber");

    for(Index i = 0; i < neurons_number; i++) {
        const XMLElement* unscaling_neuron_element = start_element->NextSiblingElement("UnscalingNeuron");
        if(!unscaling_neuron_element) {
            throw runtime_error("Unscaling neuron " + to_string(i + 1) + " is nullptr.\n");
        }

        unsigned index = 0;
        unscaling_neuron_element->QueryUnsignedAttribute("Index", &index);
        if (index != i + 1) {
            throw runtime_error("Index " + to_string(index) + " is not correct.\n");
        }

        const XMLElement* descriptives_element = unscaling_neuron_element->FirstChildElement("Descriptives");
/*
        if (descriptives_element->GetText())
        {
            const vector<string> splitted_descriptives = get_tokens(descriptives_element->GetText(), " ");
            descriptives[i].set(
                type(stof(splitted_descriptives[0])),
                type(stof(splitted_descriptives[1])),
                type(stof(splitted_descriptives[2])),
                type(stof(splitted_descriptives[3])));
        }
*/
        scalers[i] = read_xml_string(unscaling_neuron_element, "Scaler");

        start_element = unscaling_neuron_element;
    }
}

REGISTER(Layer, Unscaling, "Unscaling")

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
