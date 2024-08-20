//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   S C A L I N G   L A Y E R   4 D   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "tensors.h"
#include "scaling_layer_4d.h"
#include "strings_utilities.h"

namespace opennn
{

ScalingLayer4D::ScalingLayer4D() : Layer()
{
    set();
}


ScalingLayer4D::ScalingLayer4D(const Index& new_neurons_number) : Layer()
{
    set(new_neurons_number);
}


ScalingLayer4D::ScalingLayer4D(const dimensions& new_input_dimensions) : Layer()
{
    set(new_input_dimensions);
}


ScalingLayer4D::ScalingLayer4D(const Tensor<Descriptives, 1>& new_descriptives) : Layer()
{
    set(new_descriptives);
}


dimensions ScalingLayer4D::get_inputs_dimensions() const
{
    return input_dimensions;
}


dimensions ScalingLayer4D::get_output_dimensions() const
{
    return input_dimensions;
}


Index ScalingLayer4D::get_inputs_number() const
{
    return input_dimensions[0]*input_dimensions[1]*input_dimensions[2];
}


Index ScalingLayer4D::get_neurons_number() const
{
    return descriptives.size();
}


Tensor<Descriptives, 1> ScalingLayer4D::get_descriptives() const
{
    return descriptives;
}


Descriptives ScalingLayer4D::get_descriptives(const Index& index) const
{
    return descriptives(index);
}


Tensor<type, 1> ScalingLayer4D::get_minimums() const
{
    const Index neurons_number = get_neurons_number();

    Tensor<type, 1> minimums(neurons_number);

    for(Index i = 0; i < neurons_number; i++)
    {
        minimums[i] = descriptives[i].minimum;
    }

    return minimums;
}


Tensor<type, 1> ScalingLayer4D::get_maximums() const
{
    const Index neurons_number = get_neurons_number();

    Tensor<type, 1> maximums(neurons_number);

    for(Index i = 0; i < neurons_number; i++)
    {
        maximums[i] = descriptives[i].maximum;
    }

    return maximums;
}


Tensor<type, 1> ScalingLayer4D::get_means() const
{
    const Index neurons_number = get_neurons_number();

    Tensor<type, 1> means(neurons_number);

    for(Index i = 0; i < neurons_number; i++)
    {
        means[i] = descriptives[i].mean;
    }

    return means;
}


Tensor<type, 1> ScalingLayer4D::get_standard_deviations() const
{
    const Index neurons_number = get_neurons_number();

    Tensor<type, 1> standard_deviations(neurons_number);

    for(Index i = 0; i < neurons_number; i++)
    {
        standard_deviations[i] = descriptives[i].standard_deviation;
    }

    return standard_deviations;
}


Tensor<Scaler, 1> ScalingLayer4D::get_scaling_methods() const
{
    return scalers;
}


Tensor<string, 1> ScalingLayer4D::write_scalers() const
{
    const Index neurons_number = get_neurons_number();

    Tensor<string, 1> scaling_methods_strings(neurons_number);

    for(Index i = 0; i < neurons_number; i++)
    {
        if(scalers[i] == Scaler::None)
        {
            scaling_methods_strings[i] = "None";
        }
        else if(scalers[i] == Scaler::MinimumMaximum)
        {
            scaling_methods_strings[i] = "MinimumMaximum";
        }
        else if(scalers[i] == Scaler::MeanStandardDeviation)
        {
            scaling_methods_strings[i] = "MeanStandardDeviation";
        }
        else if(scalers[i] == Scaler::StandardDeviation)
        {
            scaling_methods_strings[i] = "StandardDeviation";
        }
        else if(scalers[i] == Scaler::Logarithm)
        {
            scaling_methods_strings[i] = "Logarithm";
        }
        else
        {
            throw runtime_error("Unknown " + to_string(i) + " scaling method.\n");
        }
    }

    return scaling_methods_strings;
}


Tensor<string, 1> ScalingLayer4D::write_scalers_text() const
{
    const Index neurons_number = get_neurons_number();

#ifdef OPENNN_DEBUG

    if(neurons_number == 0)
        throw runtime_error("Neurons number must be greater than 0.\n");

#endif

    Tensor<string, 1> scaling_methods_strings(neurons_number);

    for(Index i = 0; i < neurons_number; i++)
    {
        if(scalers[i] == Scaler::None)
        {
            scaling_methods_strings[i] = "no scaling";
        }
        else if(scalers[i] == Scaler::MeanStandardDeviation)
        {
            scaling_methods_strings[i] = "mean and standard deviation";
        }
        else if(scalers[i] == Scaler::StandardDeviation)
        {
            scaling_methods_strings[i] = "standard deviation";
        }
        else if(scalers[i] == Scaler::MinimumMaximum)
        {
            scaling_methods_strings[i] = "minimum and maximum";
        }
        else if(scalers[i] == Scaler::Logarithm)
        {
            scaling_methods_strings[i] = "Logarithm";
        }
        else
        {
            throw runtime_error("Unknown " + to_string(i) + " scaling method.\n");
        }
    }

    return scaling_methods_strings;
}


const bool& ScalingLayer4D::get_display() const
{
    return display;
}


void ScalingLayer4D::set()
{
    descriptives.resize(0);

    scalers.resize(0);

    set_default();
}


void ScalingLayer4D::set(const Index& new_inputs_number)
{
    descriptives.resize(new_inputs_number);

    scalers.resize(new_inputs_number);

    scalers.setConstant(Scaler::MeanStandardDeviation);

    set_default();
}


void ScalingLayer4D::set(const dimensions& new_input_dimensions)
{
    input_dimensions = new_input_dimensions;

    const Index inputs_number = get_inputs_number();

    descriptives.resize(inputs_number);

    scalers.resize(inputs_number);
    scalers.setConstant(Scaler::MinimumMaximum);

    set_default();
}


void ScalingLayer4D::set(const Tensor<Descriptives, 1>& new_descriptives)
{
    descriptives = new_descriptives;

    scalers.resize(new_descriptives.size());

    scalers.setConstant(Scaler::MeanStandardDeviation);

    set_neurons_number(new_descriptives.size());

    set_default();
}


void ScalingLayer4D::set(const Tensor<Descriptives, 1>& new_descriptives, const Tensor<Scaler, 1>& new_scalers)
{
    descriptives = new_descriptives;

    scalers = new_scalers;
}


void ScalingLayer4D::set(const tinyxml2::XMLDocument& new_scaling_layer_document)
{
    set_default();

    from_XML(new_scaling_layer_document);
}


void ScalingLayer4D::set_inputs_number(const Index& new_inputs_number)
{
    descriptives.resize(new_inputs_number);

    scalers.resize(new_inputs_number);

    scalers.setConstant(Scaler::MeanStandardDeviation);
}


void ScalingLayer4D::set_neurons_number(const Index& new_neurons_number)
{
    descriptives.resize(new_neurons_number);

    scalers.resize(new_neurons_number);

    scalers.setConstant(Scaler::MeanStandardDeviation);
}


void ScalingLayer4D::set_default()
{
    layer_name = "scaling_layer";

    set_scalers(Scaler::MinimumMaximum);

    set_min_max_range(type(0), type(255));

    set_display(true);

    layer_type = Type::Scaling4D;
}


void ScalingLayer4D::set_min_max_range(const type& min, const type& max)
{
    min_range = min;
    max_range = max;
}


void ScalingLayer4D::set_descriptives(const Tensor<Descriptives, 1>& new_descriptives)
{

#ifdef OPENNN_DEBUG

    const Index new_descriptives_size = new_descriptives.size();

    const Index neurons_number = get_neurons_number();

    if(new_descriptives_size != neurons_number)
        throw runtime_error("Size of descriptives (" + to_string(new_descriptives_size) + ") is not equal to number of scaling neurons (" + to_string(neurons_number) + ").\n");

#endif

    descriptives = new_descriptives;
}


void ScalingLayer4D::set_item_descriptives(const Index& i, const Descriptives& item_descriptives)
{
    descriptives(i) = item_descriptives;
}


void ScalingLayer4D::set_minimum(const Index& i, const type& new_minimum)
{
    descriptives(i).set_minimum(new_minimum);
}


void ScalingLayer4D::set_maximum(const Index& i, const type& new_maximum)
{
    descriptives(i).set_maximum(new_maximum);
}


void ScalingLayer4D::set_mean(const Index& i, const type& new_mean)
{
    descriptives(i).set_mean(new_mean);
}


void ScalingLayer4D::set_standard_deviation(const Index& i, const type& new_standard_deviation)
{
    descriptives(i).set_standard_deviation(new_standard_deviation);
}


void ScalingLayer4D::set_scalers(const Tensor<Scaler, 1>& new_scaling_methods)
{
#ifdef OPENNN_DEBUG

    const Index neurons_number = get_neurons_number();

    if(neurons_number == 0)
        throw runtime_error("Neurons number (" + to_string(neurons_number) + ") must be greater than 0.\n");

#endif

    scalers = new_scaling_methods;
}


void ScalingLayer4D::set_scalers(const Tensor<string, 1>& new_scaling_methods_string)
{
    const Index neurons_number = get_neurons_number();

#ifdef OPENNN_DEBUG

    if(neurons_number == 0)
        throw runtime_error("Neurons number (" + to_string(neurons_number) + ") must be greater than 0.\n");

#endif

    Tensor<Scaler, 1> new_scaling_methods(neurons_number);

    for(Index i = 0; i < neurons_number; i++)
    {
        if(new_scaling_methods_string(i) == "None")
        {
            new_scaling_methods(i) = Scaler::None;
        }
        else if(new_scaling_methods_string(i) == "MinimumMaximum")
        {
            new_scaling_methods(i) = Scaler::MinimumMaximum;
        }
        else if(new_scaling_methods_string(i) == "MeanStandardDeviation")
        {
            new_scaling_methods(i) = Scaler::MeanStandardDeviation;
        }
        else if(new_scaling_methods_string(i) == "StandardDeviation")
        {
            new_scaling_methods(i) = Scaler::StandardDeviation;
        }
        else if(new_scaling_methods_string(i) == "Logarithm")
        {
            new_scaling_methods(i) = Scaler::Logarithm;
        }
        else
        {
            throw runtime_error("Unknown scaling method: " + new_scaling_methods_string[i] + ".\n");
        }
    }

    set_scalers(new_scaling_methods);
}


void ScalingLayer4D::set_scaler(const Index& variable_index, const Scaler& new_scaler)
{
    scalers(variable_index) = new_scaler;
}


void ScalingLayer4D::set_scalers(const string& new_scaling_methods_string)
{
    const Index neurons_number = get_neurons_number();

#ifdef OPENNN_DEBUG

    if(neurons_number == 0)
        throw runtime_error("Neurons number (" + to_string(neurons_number) + ") must be greater than 0.\n");

#endif

    Tensor<Scaler, 1> new_scaling_methods(neurons_number);

    for(Index i = 0; i < neurons_number; i++)
    {
        if(new_scaling_methods_string == "None")
        {
            new_scaling_methods(i) = Scaler::None;
        }
        else if(new_scaling_methods_string == "MeanStandardDeviation")
        {
            new_scaling_methods(i) = Scaler::MeanStandardDeviation;
        }
        else if(new_scaling_methods_string == "MinimumMaximum")
        {
            new_scaling_methods(i) = Scaler::MinimumMaximum;
        }
        else if(new_scaling_methods_string == "StandardDeviation")
        {
            new_scaling_methods(i) = Scaler::StandardDeviation;
        }
        else if(new_scaling_methods_string == "Logarithm")
        {
            new_scaling_methods(i) = Scaler::Logarithm;
        }
        else
        {
            throw runtime_error("Unknown scaling method: " + to_string(new_scaling_methods_string[i]) + ".\n");
        }
    }

    set_scalers(new_scaling_methods);
}


void ScalingLayer4D::set_scalers(const Scaler& new_scaling_method)
{
    const Index neurons_number = get_neurons_number();

    for(Index i = 0; i < neurons_number; i++)
    {
        scalers(i) = new_scaling_method;
    }
}


void ScalingLayer4D::set_display(const bool& new_display)
{
    display = new_display;
}


bool ScalingLayer4D::is_empty() const
{
    const Index inputs_number = get_neurons_number();

    if(inputs_number == 0)
    {
        return true;
    }
    else
    {
        return false;
    }
}


// void ScalingLayer4D::check_range(const Tensor<type, 1>& inputs) const
// {
//     const Index inputs_number = get_neurons_number();

//     // Check inputs

//     if(display)
//     {
//         for(Index i = 0; i < inputs_number; i++)
//         {
//             if(inputs(i) < descriptives(i).minimum)
//             {
//                 cout << "OpenNN Warning: ScalingLayer4D class.\n"
//                      << "void check_range(const Tensor<type, 1>&) const method.\n"
//                      << "Input value " << i << " is less than corresponding minimum.\n";
//             }

//             if(inputs(i) > descriptives(i).maximum)
//             {
//                 cout << "OpenNN Warning: ScalingLayer4D class.\n"
//                      << "void check_range(const Tensor<type, 1>&) const method.\n"
//                      << "Input value " << i << " is greater than corresponding maximum.\n";
//             }
//         }
//     }
// }


void ScalingLayer4D::forward_propagate(const Tensor<pair<type*, dimensions>, 1>& inputs_pair,
                                     LayerForwardPropagation* forward_propagation,
                                     const bool& is_training)
{
    ScalingLayer4DForwardPropagation* scaling_layer_forward_propagation
            = static_cast<ScalingLayer4DForwardPropagation*>(forward_propagation);

    const TensorMap<Tensor<type, 4>> inputs(inputs_pair(0).first,
                                            inputs_pair(0).second[0],
                                            inputs_pair(0).second[1],
                                            inputs_pair(0).second[2],
                                            inputs_pair(0).second[3]);

    Tensor<type, 4>& outputs = scaling_layer_forward_propagation->outputs;

    outputs.device(*thread_pool_device) = inputs/type(255); 
}


void ScalingLayer4D::print() const
{
    cout << "Scaling layer" << endl;

    const Index inputs_number = get_inputs_number();

    print_dimensions(input_dimensions);

    const Tensor<string, 1> scalers_text = write_scalers_text();

    for(Index i = 0; i < inputs_number; i++)
    {
        // cout << "Neuron " << i << endl;

        // cout << "Scaler " << scalers_text(i) << endl;

        // descriptives(i).print();
    }
}


void ScalingLayer4D::write_XML(tinyxml2::XMLPrinter& file_stream) const
{
    ostringstream buffer;

    const Index neurons_number = get_neurons_number();

    // Scaling layer

    file_stream.OpenElement("ScalingLayer4D");

    // Scaling neurons number

    file_stream.OpenElement("ScalingNeuronsNumber");

    buffer.str("");
    buffer << neurons_number;

    file_stream.PushText(buffer.str().c_str());

    file_stream.CloseElement();

    // Scaling neurons

    const Tensor<string, 1> scaling_methods_string = write_scalers();

    for(Index i = 0; i < neurons_number; i++)
    {
        // Scaling neuron

        file_stream.OpenElement("ScalingNeuron");

        file_stream.PushAttribute("Index", int(i+1));

        //Descriptives

        file_stream.OpenElement("Descriptives");

        buffer.str(""); buffer << descriptives(i).minimum;
        file_stream.PushText(buffer.str().c_str());
        file_stream.PushText("\\");

        buffer.str(""); buffer << descriptives(i).maximum;
        file_stream.PushText(buffer.str().c_str());
        file_stream.PushText("\\");

        buffer.str(""); buffer << descriptives(i).mean;
        file_stream.PushText(buffer.str().c_str());
        file_stream.PushText("\\");

        buffer.str(""); buffer << descriptives(i).standard_deviation;
        file_stream.PushText(buffer.str().c_str());

        file_stream.CloseElement();

        // Scaler

        file_stream.OpenElement("Scaler");

        buffer.str("");
        buffer << scaling_methods_string(i);

        file_stream.PushText(buffer.str().c_str());

        file_stream.CloseElement();

        // Scaling neuron (end tag)

        file_stream.CloseElement();
    }

    // Scaling layer (end tag)

    file_stream.CloseElement();
}


void ScalingLayer4D::from_XML(const tinyxml2::XMLDocument& document)
{
    ostringstream buffer;

    const tinyxml2::XMLElement* scaling_layer_element = document.FirstChildElement("ScalingLayer4D");

    if(!scaling_layer_element)
        throw runtime_error("Scaling layer element is nullptr.\n");

    // Scaling neurons number

    const tinyxml2::XMLElement* neurons_number_element = scaling_layer_element->FirstChildElement("ScalingNeuronsNumber");

    if(!neurons_number_element)
        throw runtime_error("Scaling neurons number element is nullptr.\n");

    const Index neurons_number = Index(atoi(neurons_number_element->GetText()));

    set(neurons_number);

    unsigned index = 0; // Index does not work

    const tinyxml2::XMLElement* start_element = neurons_number_element;

    for(Index i = 0; i < neurons_number; i++)
    {
        const tinyxml2::XMLElement* scaling_neuron_element = start_element->NextSiblingElement("ScalingNeuron");
        start_element = scaling_neuron_element;

        if(!scaling_neuron_element)
            throw runtime_error("Scaling neuron " + to_string(i+1) + " is nullptr.\n");

        scaling_neuron_element->QueryUnsignedAttribute("Index", &index);

        if(index != i+1)
            throw runtime_error("Index " + to_string(index) + " is not correct.\n");

        // Descriptives

        const tinyxml2::XMLElement* descriptives_element = scaling_neuron_element->FirstChildElement("Descriptives");

        if(!descriptives_element)
            throw runtime_error("Descriptives element " + to_string(i+1) + " is nullptr.\n");

        if(descriptives_element->GetText())
        {
            const char* new_descriptives_element = descriptives_element->GetText();
            Tensor<string,1> splitted_descriptives = get_tokens(new_descriptives_element, "\\");
            descriptives[i].minimum = type(stof(splitted_descriptives[0]));
            descriptives[i].maximum = type(stof(splitted_descriptives[1]));
            descriptives[i].mean = type(stof(splitted_descriptives[2]));
            descriptives[i].standard_deviation = type(stof(splitted_descriptives[3]));
        }

        // Scaling method

        const tinyxml2::XMLElement* scaling_method_element = scaling_neuron_element->FirstChildElement("Scaler");

        if(!scaling_method_element)
            throw runtime_error("Scaling method element " + to_string(i+1) + " is nullptr.\n");

        const string new_method = scaling_method_element->GetText();

        if(new_method == "None" || new_method == "No Scaling")
        {
            scalers[i] = Scaler::None;
        }
        else if(new_method == "MinimumMaximum" || new_method == "Minimum - Maximum")
        {
            scalers[i] = Scaler::MinimumMaximum;
        }
        else if(new_method == "MeanStandardDeviation" || new_method == "Mean - Standard deviation")
        {
            scalers[i] = Scaler::MeanStandardDeviation;
        }
        else if(new_method == "StandardDeviation")
        {
            scalers[i] = Scaler::StandardDeviation;
        }
        else if(new_method == "Logarithm")
        {
            scalers[i] = Scaler::Logarithm;
        }
        else
        {
            scalers[i] = Scaler::None;
        }
    }

    // Display
    {
        const tinyxml2::XMLElement* display_element = scaling_layer_element->FirstChildElement("Display");

        if(display_element)
        {
            string new_display_string = display_element->GetText();

            try
            {
                set_display(new_display_string != "0");
            }
            catch(const exception& e)
            {
                cerr << e.what() << endl;
            }
        }
    }
}

pair<type*, dimensions> ScalingLayer4DForwardPropagation::get_outputs_pair() const
{
    const Index neurons_number = layer->get_neurons_number();

    return pair<type*, dimensions>(outputs_data, { batch_samples_number, neurons_number, 1, 1 });
}


void ScalingLayer4DForwardPropagation::set(const Index& new_batch_samples_number, Layer* new_layer)
{
    layer = new_layer;

    const Index neurons_number = layer->get_neurons_number();

    batch_samples_number = new_batch_samples_number;

    outputs.resize(batch_samples_number, neurons_number,1,1);

    outputs_data = outputs.data();
}

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2024 Artificial Intelligence Techniques, SL.
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
