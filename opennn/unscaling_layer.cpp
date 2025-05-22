//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   U N S C A L I N G   L A Y E R    C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "unscaling_layer.h"
#include "strings_utilities.h"
#include "tensors.h"
#include "descriptives.h"

namespace opennn
{

Unscaling::Unscaling(const dimensions& new_input_dimensions, const string& layer_name)
    : Layer()
{
    set(new_input_dimensions[0], layer_name);
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


vector<Scaler> Unscaling::get_unscaling_method() const
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
        const Scaler scaler = scalers[i];

        switch (scaler)
        { 

        case Scaler::None:
        
            buffer << output_names[i] << " = " << input_names[i] << ";\n";
            break;

        case Scaler::MinimumMaximum:
        
            if(abs(descriptives[i].minimum - descriptives[i].maximum) < NUMERIC_LIMITS_MIN)
            {
                buffer << output_names[i] << "=" << descriptives[i].minimum <<";\n";
            }
            else
            {
                const type slope = (descriptives[i].maximum-descriptives[i].minimum)/(max_range-min_range);

                const type intercept = descriptives[i].minimum - min_range*(descriptives[i].maximum-descriptives[i].minimum)/(max_range-min_range);

                buffer << output_names[i] << "=" << input_names[i] << "*" << slope << "+" << intercept<<";\n";
            }
            break;

        case Scaler::MeanStandardDeviation:
        
            buffer << output_names[i] << "=" << input_names[i] << "*" << descriptives[i].standard_deviation <<"+"<< descriptives[i].mean <<";\n";
        
            break;

        case  Scaler::StandardDeviation:
        
            buffer << output_names[i] << "=" <<  input_names[i] << "*" << descriptives[i].standard_deviation <<";\n";
        
            break;

        case Scaler::Logarithm:

            buffer << output_names[i] << "=" << "exp(" << input_names[i] << ");\n";
            
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


vector<string> Unscaling::write_unscaling_methods() const
{
    const Index outputs_number = get_outputs_number();

    vector<string> scaling_methods_strings(outputs_number);

    for (Index i = 0; i < outputs_number; i++)
        scaling_methods_strings[i] = scaler_to_string(scalers[i]);

    return scaling_methods_strings;
}


vector<string> Unscaling::write_unscaling_method_text() const
{
    const Index outputs_number = get_outputs_number();

    vector<string> scaling_methods_strings(outputs_number);

    for(Index i = 0; i < outputs_number; i++)
        if(scalers[i] == Scaler::None)
            scaling_methods_strings[i] = "no unscaling";
        else if(scalers[i] == Scaler::MinimumMaximum)
            scaling_methods_strings[i] = "minimum and maximum";
        else if(scalers[i] == Scaler::MeanStandardDeviation)
            scaling_methods_strings[i] = "mean and standard deviation";
        else if(scalers[i] == Scaler::StandardDeviation)
            scaling_methods_strings[i] = "standard deviation";
        else if(scalers[i] == Scaler::Logarithm)
            scaling_methods_strings[i] = "logarithm";
        else
            throw runtime_error("Unknown unscaling method.\n");

    return scaling_methods_strings;
}


void Unscaling::set_input_dimensions(const dimensions& new_input_dimensions)
{
    descriptives.resize(new_input_dimensions[0]);
}


void Unscaling::set_output_dimensions(const dimensions& new_output_dimensions)
{
    descriptives.resize(new_output_dimensions[0]);
}


void Unscaling::set(const Index& new_neurons_number, const string& new_name)
{
    descriptives.resize(new_neurons_number);

    for (Index i = 0; i < new_neurons_number; i++)
        descriptives[i].set(type(-1.0), type(1), type(0), type(1));

    scalers.resize(new_neurons_number, Scaler::MinimumMaximum);

    name = new_name;

    set_scalers(Scaler::MinimumMaximum);

    set_min_max_range(type(-1), type(1));

    layer_type = Type::Unscaling;
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


void Unscaling::set_scalers(const vector<Scaler>& new_unscaling_method)
{
    scalers = new_unscaling_method;
}


void Unscaling::set_scalers(const string& new_scaling_methods_string)
{
    if(new_scaling_methods_string == "None")
        set_scalers(Scaler::None);
    else if(new_scaling_methods_string == "MinimumMaximum")
        set_scalers(Scaler::MinimumMaximum);
    else if(new_scaling_methods_string == "MeanStandardDeviation")
        set_scalers(Scaler::MeanStandardDeviation);
    else if(new_scaling_methods_string == "StandardDeviation")
        set_scalers(Scaler::StandardDeviation);
    else if(new_scaling_methods_string == "Logarithm")
        set_scalers(Scaler::Logarithm);
    else
        throw runtime_error("set_scalers(const string& new_scaling_methods_string) method.\n");
}


void Unscaling::set_scalers(const vector<string>& new_scalers)
{
    const Index outputs_number = get_outputs_number();

    for(Index i = 0; i < outputs_number; i++)
        scalers[i] = string_to_scaler(new_scalers[i]);
}


void Unscaling::set_scalers(const Scaler& new_unscaling_method)
{
    const Index outputs_number = get_outputs_number();

    for(Index i = 0; i < outputs_number; i++)
        scalers[i] = new_unscaling_method;
}


bool Unscaling::is_empty() const
{
    return get_output_dimensions()[0] == 0;
}


void Unscaling::forward_propagate(const vector<pair<type*, dimensions>>& input_pairs,
                                       unique_ptr<LayerForwardPropagation>& forward_propagation,
                                       const bool&)
{
    const Index outputs_number = get_outputs_number();

    UnscalingForwardPropagation* unscaling_layer_forward_propagation =
            static_cast<UnscalingForwardPropagation*>(forward_propagation.get());

    const TensorMap<Tensor<type,2>> inputs = tensor_map_2(input_pairs[0]);

    Tensor<type, 2>& outputs = unscaling_layer_forward_propagation->outputs;

    outputs = inputs;

    for(Index i = 0; i < outputs_number; i++)
    {
        const Scaler& scaler = scalers[i];

        const Descriptives& descriptive = descriptives[i];

        if(abs(descriptives[i].standard_deviation) < NUMERIC_LIMITS_MIN)
            throw runtime_error("Standard deviation is zero.");

        switch(scaler)
        {
        case Scaler::None:
            continue;
        break;

        case Scaler::MinimumMaximum:
            unscale_minimum_maximum(outputs, i, descriptive, min_range, max_range);
        break;

        case Scaler::MeanStandardDeviation:
            unscale_mean_standard_deviation(outputs, i, descriptive);
        break;

        case Scaler::StandardDeviation:
            unscale_standard_deviation(outputs, i, descriptive);
        break;

        case Scaler::Logarithm:
            unscale_logarithmic(outputs, i);
        break;

        case Scaler::ImageMinMax:
            unscale_image_minimum_maximum(outputs, i);
            break;

        default:
            //break;
            throw runtime_error("Unknown scaling method\n");
        }
    }
}


vector<string> Unscaling::write_scalers_text() const
{
    const Index outputs_number = get_outputs_number();

    vector<string> scaling_methods_strings(outputs_number);

    for(Index i = 0; i < outputs_number; i++)
        if(scalers[i] == Scaler::None)
            scaling_methods_strings[i] = "no scaling";
        else if(scalers[i] == Scaler::MeanStandardDeviation)
            scaling_methods_strings[i] = "mean and standard deviation";
        else if(scalers[i] == Scaler::StandardDeviation)
            scaling_methods_strings[i] = "standard deviation";
        else if(scalers[i] == Scaler::MinimumMaximum)
            scaling_methods_strings[i] = "minimum and maximum";
        else if(scalers[i] == Scaler::Logarithm)
            scaling_methods_strings[i] = "Logarithm";
        else
            throw runtime_error("Unknown " + to_string(i) + " scaling method.\n");

    return scaling_methods_strings;
}


void Unscaling::print() const
{
    cout << "Unscaling layer" << endl;

    const Index inputs_number = get_inputs_number();

    const vector<string> scalers_text = write_scalers_text();

    for(Index i = 0; i < inputs_number; i++)
    {
        cout << "Neuron " << i << endl
             << "Scaler " << scalers_text[i] << endl;

        descriptives[i].print();
    }
}


void Unscaling::to_XML(XMLPrinter& printer) const
{
    printer.OpenElement("Unscaling");

    const dimensions output_dimensions = get_output_dimensions();

    add_xml_element(printer, "NeuronsNumber", to_string(output_dimensions[0]));

    const vector<string> scalers = write_unscaling_methods();

    for (Index i = 0; i < output_dimensions[0]; i++) 
    {
        printer.OpenElement("UnscalingNeuron");
        printer.PushAttribute("Index", int(i + 1));
        add_xml_element(printer, "Descriptives", tensor_to_string(descriptives[i].to_tensor()));
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

        scalers[i] = string_to_scaler(read_xml_string(unscaling_neuron_element, "Scaler"));

        start_element = unscaling_neuron_element;
    }
}


UnscalingForwardPropagation::UnscalingForwardPropagation(const Index& new_batch_size, Layer* new_layer)
    : LayerForwardPropagation()
{
    set(new_batch_size, new_layer);
}


pair<type*, dimensions> UnscalingForwardPropagation::get_outputs_pair() const
{
    const dimensions output_dimensions = layer->get_output_dimensions();

    return { (type*)outputs.data(), { batch_size, output_dimensions[0]}};
}


void UnscalingForwardPropagation::set(const Index& new_batch_size, Layer* new_layer)
{
    layer = new_layer;

    const dimensions output_dimensions = static_cast<Unscaling*>(layer)->get_output_dimensions();

    batch_size = new_batch_size;

    outputs.resize(batch_size, output_dimensions[0]);
}


void UnscalingForwardPropagation::print() const
{
    cout << "Outputs:" << endl
        << outputs << endl;
}

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
