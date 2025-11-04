//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   U N S C A L I N G   L A Y E R    C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "registry.h"
#include "strings_utilities.h"
#include "tensors.h"
#include "unscaling_layer.h"

namespace opennn
{

Unscaling::Unscaling(const dimensions& new_input_dimensions, const string& label)
    : Layer()
{
    set(new_input_dimensions[0], label);
}


dimensions Unscaling::get_input_dimensions() const
{
    const Index neurons_number = descriptives.size();

    return {neurons_number};
}


dimensions Unscaling::get_output_dimensions() const
{
    const Index neurons_number = descriptives.size();

    return { neurons_number };
}


vector<Descriptives> Unscaling::get_descriptives() const
{
    return descriptives;
}


Tensor<type, 1> Unscaling::get_minimums() const
{
    const Index outputs_number = get_outputs_number();

    Tensor<type, 1> minimums(outputs_number);

#pragma omp parallel for
    for(Index i = 0; i < outputs_number; i++)
        minimums[i] = descriptives[i].minimum;

    return minimums;
}


Tensor<type, 1> Unscaling::get_maximums() const
{
    const Index outputs_number = get_outputs_number();

    Tensor<type, 1> maximums(outputs_number);

#pragma omp parallel for
    for(Index i = 0; i < outputs_number; i++)
        maximums[i] = descriptives[i].maximum;

    return maximums;
}


vector<string> Unscaling::get_scalers() const
{
    return scalers;
}


string Unscaling::get_expression(const vector<string>& new_input_names,
                                 const vector<string>& new_output_names) const
{
    const vector<string> input_names = new_input_names.empty()
                                           ? get_default_input_names()
                                           : new_input_names;

    const vector<string> output_names = new_output_names.empty()
                                            ? get_default_output_names()
                                            : new_output_names;

    const Index outputs_number = get_outputs_number();

    ostringstream buffer;

    buffer.precision(10);

    for(Index i = 0; i < outputs_number; i++)
    {
        const string& scaler = scalers[i];

        if(scaler == "None")
            buffer << output_names[i] << " = " << input_names[i] << ";\n";
        else if(scaler == "MinimumMaximum")
            if(abs(descriptives[i].minimum - descriptives[i].maximum) < NUMERIC_LIMITS_MIN)
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

    string expression = buffer.str();

    replace(expression, "+-", "-");
    replace(expression, "--", "+");

    return expression;
}


void Unscaling::set_input_dimensions(const dimensions& new_input_dimensions)
{
    descriptives.resize(new_input_dimensions[0]);
}


void Unscaling::set_output_dimensions(const dimensions& new_output_dimensions)
{
    descriptives.resize(new_output_dimensions[0]);
}


void Unscaling::set(const Index& new_neurons_number, const string& new_label)
{
    descriptives.resize(new_neurons_number);

    for (auto& descriptive : descriptives)
        descriptive.set(type(-1.0), type(1), type(0), type(1));

    scalers.resize(new_neurons_number, "MinimumMaximum");

    label = new_label;

    set_scalers("MinimumMaximum");

    set_min_max_range(type(-1), type(1));

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
    descriptives = new_descriptives;
}


void Unscaling::set_scalers(const vector<string>& new_scaler)
{
    scalers = new_scaler;
}


void Unscaling::set_scalers(const string& new_scalers)
{
    for (string& scaler : scalers)
        scaler = new_scalers;
}


void Unscaling::forward_propagate(const vector<TensorView>& input_views,
                                  unique_ptr<LayerForwardPropagation>& forward_propagation,
                                  const bool&)
{
    const Index outputs_number = get_outputs_number();

    UnscalingForwardPropagation* this_forward_propagation =
        static_cast<UnscalingForwardPropagation*>(forward_propagation.get());

    const TensorMap<Tensor<type,2>> inputs = tensor_map<2>(input_views[0]);

    Tensor<type, 2>& outputs = this_forward_propagation->outputs;

    outputs = inputs;

    for(Index i = 0; i < outputs_number; i++)
    {
        const string& scaler = scalers[i];

        const Descriptives& descriptive = descriptives[i];

        if(abs(descriptives[i].standard_deviation) < NUMERIC_LIMITS_MIN)
            descriptives[i].standard_deviation = NUMERIC_LIMITS_MIN;

        if(scaler == "None")
            continue;
        else if(scaler == "MinimumMaximum")
            unscale_minimum_maximum(outputs, i, descriptive, min_range, max_range);
        else if(scaler == "MeanStandardDeviation")
            unscale_mean_standard_deviation(outputs, i, descriptive);
        else if(scaler == "StandardDeviation")
            unscale_standard_deviation(outputs, i, descriptive);
        else if(scaler == "Logarithm")
            unscale_logarithmic(outputs, i);
        else if(scaler == "ImageMinMax")
            unscale_image_minimum_maximum(outputs, i);
        else
            throw runtime_error("Unknown scaling method\n");
    }
}


void Unscaling::print() const
{
    cout << "Unscaling layer" << endl;

    const Index inputs_number = get_inputs_number();

    for(Index i = 0; i < inputs_number; i++)
    {
        cout << "Neuron " << i << endl
             << "string " << scalers[i] << endl;

        descriptives[i].print();
    }
}


void Unscaling::to_XML(XMLPrinter& printer) const
{
    printer.OpenElement("Unscaling");

    const dimensions output_dimensions = get_output_dimensions();

    add_xml_element(printer, "NeuronsNumber", to_string(output_dimensions[0]));

    for (Index i = 0; i < output_dimensions[0]; i++)
    {
        printer.OpenElement("UnscalingNeuron");
        printer.PushAttribute("Index", int(i + 1));
        add_xml_element(printer, "Descriptives", tensor_to_string<type, 1>(descriptives[i].to_tensor()));
        add_xml_element(printer, "Scaler", scalers[i]);

        printer.CloseElement();
    }

    printer.CloseElement();
}


void Unscaling::from_XML(const XMLDocument& document)
{
    const XMLElement* root_element = document.FirstChildElement("Unscaling");

    if(!root_element)
        throw runtime_error("Unscaling element is nullptr.\n");

    const Index neurons_number = read_xml_index(root_element, "NeuronsNumber");

    set(neurons_number);

    const XMLElement* start_element = root_element->FirstChildElement("NeuronsNumber");

    for (Index i = 0; i < neurons_number; i++) {
        const XMLElement* unscaling_neuron_element = start_element->NextSiblingElement("UnscalingNeuron");
        if (!unscaling_neuron_element) {
            throw runtime_error("Unscaling neuron " + to_string(i + 1) + " is nullptr.\n");
        }

        unsigned index = 0;
        unscaling_neuron_element->QueryUnsignedAttribute("Index", &index);
        if (index != i + 1) {
            throw runtime_error("Index " + to_string(index) + " is not correct.\n");
        }

        const XMLElement* descriptives_element = unscaling_neuron_element->FirstChildElement("Descriptives");

        if (descriptives_element->GetText())
        {
            const vector<string> splitted_descriptives = get_tokens(descriptives_element->GetText(), " ");
            descriptives[i].set(
                type(stof(splitted_descriptives[0])),
                type(stof(splitted_descriptives[1])),
                type(stof(splitted_descriptives[2])),
                type(stof(splitted_descriptives[3])));
        }

        scalers[i] = read_xml_string(unscaling_neuron_element, "Scaler");

        start_element = unscaling_neuron_element;
    }
}


UnscalingForwardPropagation::UnscalingForwardPropagation(const Index& new_batch_size, Layer* new_layer)
    : LayerForwardPropagation()
{
    set(new_batch_size, new_layer);
}


TensorView UnscalingForwardPropagation::get_output_pair() const
{
    const dimensions output_dimensions = layer->get_output_dimensions();

    return { (type*)outputs.data(), { batch_size, output_dimensions[0]}};
}


void UnscalingForwardPropagation::initialize()
{
    const dimensions output_dimensions = static_cast<Unscaling*>(layer)->get_output_dimensions();

    outputs.resize(batch_size, output_dimensions[0]);
}


void UnscalingForwardPropagation::print() const
{
    cout << "Outputs:" << endl
         << outputs << endl;
}

REGISTER(Layer, Unscaling, "Unscaling")
REGISTER(LayerForwardPropagation, UnscalingForwardPropagation, "Unscaling")

}


// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2025 Artificial Intelligence Techniques, SL.
//
// This library is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 2.1 of the License, or any later version.
//
// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.

// You should have received a copy of the GNU Lesser General Public
// License along with this library; if not, write to the Free Software
// Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
