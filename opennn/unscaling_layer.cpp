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

namespace opennn
{

UnscalingLayer::UnscalingLayer() : Layer()
{
    set();
}


UnscalingLayer::UnscalingLayer(const dimensions& new_input_dimensions) : Layer()
{
    set(new_input_dimensions[0]);
}


Index UnscalingLayer::get_inputs_number() const
{
    return descriptives.size();
}


Index UnscalingLayer::get_neurons_number() const
{
    return descriptives.size();
}


dimensions UnscalingLayer::get_output_dimensions() const
{
    const Index neurons_number = get_neurons_number();

    return { neurons_number };
}


Tensor<Descriptives, 1> UnscalingLayer::get_descriptives() const
{
    return descriptives;
}


Tensor<type, 1> UnscalingLayer::get_minimums() const
{
    const Index neurons_number = get_neurons_number();

    Tensor<type, 1> minimums(neurons_number);

    #pragma omp parallel for
    for(Index i = 0; i < neurons_number; i++)
        minimums[i] = descriptives[i].minimum;

    return minimums;
}


Tensor<type, 1> UnscalingLayer::get_maximums() const
{
    const Index neurons_number = get_neurons_number();

    Tensor<type, 1> maximums(neurons_number);

    #pragma omp parallel for
    for(Index i = 0; i < neurons_number; i++)
        maximums[i] = descriptives[i].maximum;

    return maximums;
}


Tensor<Scaler, 1> UnscalingLayer::get_unscaling_method() const
{
    return scalers;
}


string UnscalingLayer::write_expression(const Tensor<string, 1>& input_names, 
                                        const Tensor<string, 1>& output_names) const
{
    const Index neurons_number = get_neurons_number();

    ostringstream buffer;

    buffer.precision(10);

    for(Index i = 0; i < neurons_number; i++)
    {
        const Scaler scaler = scalers(i);

        switch (scaler)
        { 

        case Scaler::None:
        
            buffer << output_names(i) << " = " << input_names(i) << ";\n";
            break;

        case Scaler::MinimumMaximum:
        
            if(abs(descriptives(i).minimum - descriptives(i).maximum) < type(NUMERIC_LIMITS_MIN))
            {
                buffer << output_names[i] << "=" << descriptives(i).minimum <<";\n";
            }
            else
            {
                const type slope = (descriptives(i).maximum-descriptives(i).minimum)/(max_range-min_range);

                const type intercept = descriptives(i).minimum - min_range*(descriptives(i).maximum-descriptives(i).minimum)/(max_range-min_range);

                buffer << output_names[i] << "=" << input_names[i] << "*" << slope << "+" << intercept<<";\n";
            }
            break;

        case Scaler::MeanStandardDeviation:
        
            buffer << output_names[i] << "=" << input_names[i] << "*" << descriptives(i).standard_deviation <<"+"<< descriptives(i).mean <<";\n";
        
            break;

        case  Scaler::StandardDeviation:
        
            buffer << output_names[i] << "=" <<  input_names(i) << "*" << descriptives(i).standard_deviation <<";\n";
        
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


Tensor<string, 1> UnscalingLayer::write_unscaling_methods() const
{
    const Index neurons_number = get_neurons_number();

    Tensor<string, 1> scaling_methods_strings(neurons_number);

    for(Index i = 0; i < neurons_number; i++)
        if(scalers[i] == Scaler::None)
            scaling_methods_strings[i] = "None";
        else if(scalers[i] == Scaler::MinimumMaximum)
            scaling_methods_strings[i] = "MinimumMaximum";
        else if(scalers[i] == Scaler::MeanStandardDeviation)
            scaling_methods_strings[i] = "MeanStandardDeviation";
        else if(scalers[i] == Scaler::StandardDeviation)
            scaling_methods_strings[i] = "StandardDeviation";
        else if(scalers[i] == Scaler::Logarithm)
            scaling_methods_strings[i] = "Logarithm";
        else
            throw runtime_error("Unknown unscaling method.\n");

    return scaling_methods_strings;
}


Tensor<string, 1> UnscalingLayer::write_unscaling_method_text() const
{
    const Index neurons_number = get_neurons_number();

    Tensor<string, 1> scaling_methods_strings(neurons_number);

    for(Index i = 0; i < neurons_number; i++)
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


const bool& UnscalingLayer::get_display() const
{
    return display;
}


void UnscalingLayer::set()
{
    descriptives.resize(0);

    scalers.resize(0);

    set_default();
}


void UnscalingLayer::set_inputs_number(const Index& new_inputs_number)
{
    descriptives.resize(new_inputs_number);
}


void UnscalingLayer::set_neurons_number(const Index& new_neurons_number)
{
    descriptives.resize(new_neurons_number);
}


void UnscalingLayer::set(const Index& new_neurons_number)
{
    descriptives.resize(new_neurons_number);

    scalers.resize(new_neurons_number);

    scalers.setConstant(Scaler::MinimumMaximum);

    set_default();
}


void UnscalingLayer::set(const Tensor<Descriptives, 1>& new_descriptives)
{
    descriptives = new_descriptives;

    scalers.resize(new_descriptives.size());

    scalers.setConstant(Scaler::MinimumMaximum);

    set_default();
}


void UnscalingLayer::set(const Tensor<Descriptives, 1>& new_descriptives, const Tensor<Scaler, 1>& new_scalers)
{
    descriptives = new_descriptives;

    scalers = new_scalers;
}


void UnscalingLayer::set(const tinyxml2::XMLDocument& new_unscaling_layer_document)
{
    set_default();

    from_XML(new_unscaling_layer_document);
}


void UnscalingLayer::set(const UnscalingLayer& new_unscaling_layer)
{
    descriptives = new_unscaling_layer.descriptives;

    scalers = new_unscaling_layer.scalers;

    display = new_unscaling_layer.display;
}


void UnscalingLayer::set_default()
{
    name = "unscaling_layer";

    set_scalers(Scaler::MinimumMaximum);

    set_min_max_range(type(-1), type(1));

    set_display(true);

    layer_type = Type::Unscaling;
}


void UnscalingLayer::set_min_max_range(const type min, const type max)
{
    min_range = min;
    max_range = max;
}


void UnscalingLayer::set_descriptives(const Tensor<Descriptives, 1>& new_descriptives)
{
    descriptives = new_descriptives;
}


void UnscalingLayer::set_item_descriptives(const Index& i, const Descriptives& item_descriptives)
{
    descriptives[i] = item_descriptives;
}


void UnscalingLayer::set_scalers(const Tensor<Scaler,1>& new_unscaling_method)
{
    scalers = new_unscaling_method;
}


void UnscalingLayer::set_scalers(const string& new_scaling_methods_string)
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


void UnscalingLayer::set_scalers(const Tensor<string, 1>& new_scalers)
{
    const Index neurons_number = get_neurons_number();

    for(Index i = 0; i < neurons_number; i++)
        set_scaler(i, new_scalers(i));
}


void UnscalingLayer::set_scalers(const Scaler& new_unscaling_method)
{
    const Index neurons_number = get_neurons_number();

    for(Index i = 0; i < neurons_number; i++)
        scalers(i) = new_unscaling_method;
}


void UnscalingLayer::set_scaler(const Index& variable_index, const string& new_scaler)
{
    if(new_scaler == "None")
        scalers(variable_index) = Scaler::None;
    else if(new_scaler == "MeanStandardDeviation")
        scalers(variable_index) = Scaler::MeanStandardDeviation;
    else if(new_scaler == "StandardDeviation")
        scalers(variable_index) = Scaler::StandardDeviation;
    else if(new_scaler == "MinimumMaximum")
        scalers(variable_index) = Scaler::MinimumMaximum;
    else if(new_scaler == "Logarithm")
        scalers(variable_index) = Scaler::Logarithm;
    else
        throw runtime_error("Unknown scaling method: " + new_scaler + ".\n");
}


void UnscalingLayer::set_display(const bool& new_display)
{
    display = new_display;
}


bool UnscalingLayer::is_empty() const
{
    return get_neurons_number() == 0;
}


void UnscalingLayer::forward_propagate(const vector<pair<type*, dimensions>>& input_pairs,
                                       unique_ptr<LayerForwardPropagation>& forward_propagation,
                                       const bool& is_training)
{
    const Index samples_number = input_pairs[0].second[0];
    const Index neurons_number = get_neurons_number();

    UnscalingLayerForwardPropagation* unscaling_layer_forward_propagation =
            static_cast<UnscalingLayerForwardPropagation*>(forward_propagation.get());

    const TensorMap<Tensor<type,2>> inputs = tensor_map_2(input_pairs[0]);

    Tensor<type,2>& outputs = unscaling_layer_forward_propagation->outputs;

    for(Index i = 0; i < neurons_number; i++)
    {
        const Scaler& scaler = scalers(i);

        const TensorMap<Tensor<type, 1>> input_column(inputs.data() + i * samples_number, samples_number);

        TensorMap<Tensor<type, 1>> output_column(outputs.data() + i * samples_number, samples_number);

        if(abs(descriptives(i).standard_deviation) < type(NUMERIC_LIMITS_MIN))
        {
            if(display)
                cout << "OpenNN Warning: ScalingLayer2D class.\n"
                     << "void forward_propagate\n"
                     << "Standard deviation of variable " << i << " is zero.\n"
                     << "Those variables won't be scaled.\n";
            
            continue;
        }

        type slope, intercept, standard_deviation;

        switch(scaler)
        {
        case Scaler::None:
            break;

        case Scaler::MinimumMaximum:
            slope = (descriptives(i).maximum-descriptives(i).minimum)/(max_range-min_range);
            intercept = -(min_range*descriptives(i).maximum-max_range*descriptives(i).minimum)/(max_range-min_range);
            output_column.device(*thread_pool_device) = intercept + slope * input_column;
            break;

        case Scaler::MeanStandardDeviation:
            slope = descriptives(i).standard_deviation;
            intercept = descriptives(i).mean;
            output_column.device(*thread_pool_device) = intercept + slope*input_column;
            break;

        case Scaler::StandardDeviation:
            standard_deviation = descriptives(i).standard_deviation;
            output_column.device(*thread_pool_device) = standard_deviation*input_column;
            break;

        case Scaler::Logarithm:
            output_column.device(*thread_pool_device) = input_column.exp();
            break;

        default:
            throw runtime_error("Unknown scaling method.\n");
        }
    }
}


void UnscalingLayer::to_XML(tinyxml2::XMLPrinter& printer) const
{
    printer.OpenElement("UnscalingLayer");

    const Index neurons_number = get_neurons_number();
    add_xml_element(printer, "UnscalingNeuronsNumber", to_string(neurons_number));

    const Tensor<string, 1> scalers = write_unscaling_methods();

    for (Index i = 0; i < neurons_number; i++) {
        printer.OpenElement("UnscalingNeuron");
        printer.PushAttribute("Index", int(i + 1));

        add_xml_element(printer, "Descriptives", tensor_to_string(descriptives(i).to_tensor()));
        add_xml_element(printer, "Scaler", scalers(i));

        printer.CloseElement();  
    }

    printer.CloseElement();
}


Tensor<string, 1> UnscalingLayer::write_scalers_text() const
{
    const Index neurons_number = get_neurons_number();

    Tensor<string, 1> scaling_methods_strings(neurons_number);

    for(Index i = 0; i < neurons_number; i++)
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


void UnscalingLayer::print() const
{
    cout << "Unscaling layer" << endl;

    const Index inputs_number = get_inputs_number();

    const Tensor<string, 1> scalers_text = write_scalers_text();

    for(Index i = 0; i < inputs_number; i++)
    {
        cout << "Neuron " << i << endl;
        cout << "Scaler " << scalers_text(i) << endl;

        descriptives(i).print();
    }
}


void UnscalingLayer::from_XML(const tinyxml2::XMLDocument& document)
{
    const tinyxml2::XMLElement* root_element = document.FirstChildElement("UnscalingLayer");

    if(!root_element)
        throw runtime_error("Unscaling layer element is nullptr.\n");

    // Unscaling neurons number

    const tinyxml2::XMLElement* neurons_number_element = root_element->FirstChildElement("UnscalingNeuronsNumber");

    if(!neurons_number_element)
        throw runtime_error("Unscaling neurons number element is nullptr.\n");

    const Index neurons_number = Index(atoi(neurons_number_element->GetText()));

    set(neurons_number);

    unsigned index = 0; // Index does not work

    const tinyxml2::XMLElement* start_element = neurons_number_element;

    for(Index i = 0; i < neurons_number; i++)
    {
        const tinyxml2::XMLElement* unscaling_neuron_element = start_element->NextSiblingElement("UnscalingNeuron");
        start_element = unscaling_neuron_element;

        if(!unscaling_neuron_element)
            throw runtime_error("Unscaling neuron " + to_string(i+1) + " is nullptr.\n");

        unscaling_neuron_element->QueryUnsignedAttribute("Index", &index);

        if(index != i+1)
            throw runtime_error("Index " + to_string(index) + " is not correct.\n");

        // Minimum

        const tinyxml2::XMLElement* descriptives_element = unscaling_neuron_element->FirstChildElement("Descriptives");

        if(descriptives_element->GetText())
        {
            const char* new_descriptives_element = descriptives_element->GetText();

            const Tensor<string,1> splitted_descriptives = get_tokens(new_descriptives_element, " ");

            descriptives[i].set(type(stof(splitted_descriptives[0])),
                                type(stof(splitted_descriptives[1])),
                                type(stof(splitted_descriptives[2])),
                                type(stof(splitted_descriptives[3])));
        }

        // Unscaling method

        const tinyxml2::XMLElement* unscaling_method_element = unscaling_neuron_element->FirstChildElement("Scaler");

        if(!unscaling_method_element)
            throw runtime_error("Unscaling method element " + to_string(i+1) + " is nullptr.\n");

        const string new_method = unscaling_method_element->GetText();

        set_scaler(i, new_method);
    }

    // Display

    const tinyxml2::XMLElement* display_element = root_element->FirstChildElement("Display");

    if(display_element)
        set_display(display_element->GetText() != string("0"));
}


pair<type*, dimensions> UnscalingLayerForwardPropagation::get_outputs_pair() const
{
    const Index neurons_number = layer->get_neurons_number();

    return { outputs_data, { batch_samples_number, neurons_number } };
}


void UnscalingLayerForwardPropagation::set(const Index& new_batch_samples_number, Layer* new_layer)
{
    layer = new_layer;

    const Index neurons_number = static_cast<UnscalingLayer*>(layer)->get_neurons_number();

    batch_samples_number = new_batch_samples_number;

    outputs.resize(batch_samples_number, neurons_number);

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
