//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   S C A L I N G   L A Y E R   2 D   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "scaling_layer_2d.h"
#include "strings_utilities.h"
#include "tensors.h"
#include "descriptives.h"
#include "statistics.h"

namespace opennn
{

ScalingLayer2D::ScalingLayer2D(const dimensions& new_input_dimensions) : Layer()
{
    set(new_input_dimensions);
}


dimensions ScalingLayer2D::get_input_dimensions() const
{
    return dimensions{Index(scalers.size())};
}


dimensions ScalingLayer2D::get_output_dimensions() const
{
    return dimensions{Index(scalers.size())};
}


vector<Descriptives> ScalingLayer2D::get_descriptives() const
{
    return descriptives;
}


Descriptives ScalingLayer2D::get_descriptives(const Index& index) const
{
    return descriptives[index];
}


Tensor<type, 1> ScalingLayer2D::get_minimums() const
{
    const Index outputs_number = get_outputs_number();

    Tensor<type, 1> minimums(outputs_number);

    #pragma omp parallel for
    for(Index i = 0; i < outputs_number; i++)
        minimums[i] = descriptives[i].minimum;

    return minimums;
}


Tensor<type, 1> ScalingLayer2D::get_maximums() const
{
    const Index outputs_number = get_outputs_number();

    Tensor<type, 1> maximums(outputs_number);

    #pragma omp parallel for
    for(Index i = 0; i < outputs_number; i++)
        maximums[i] = descriptives[i].maximum;

    return maximums;
}


Tensor<type, 1> ScalingLayer2D::get_means() const
{
    const Index outputs_number = get_outputs_number();

    Tensor<type, 1> means(outputs_number);

    #pragma omp parallel for
    for(Index i = 0; i < outputs_number; i++)
        means[i] = descriptives[i].mean;

    return means;
}


Tensor<type, 1> ScalingLayer2D::get_standard_deviations() const
{
    const Index outputs_number = get_outputs_number();

    Tensor<type, 1> standard_deviations(outputs_number);

    #pragma omp parallel for
    for(Index i = 0; i < outputs_number; i++)
        standard_deviations[i] = descriptives[i].standard_deviation;

    return standard_deviations;
}


vector<Scaler> ScalingLayer2D::get_scaling_methods() const
{
    return scalers;
}


vector<string> ScalingLayer2D::write_scalers() const
{
    const Index outputs_number = get_outputs_number();

    vector<string> scaling_methods_strings(outputs_number);

    #pragma omp parallel for

    for (Index i = 0; i < outputs_number; i++)
        scaling_methods_strings[i] = scaler_to_string(scalers[i]);

    return scaling_methods_strings;
}


vector<string> ScalingLayer2D::write_scalers_text() const
{
    const Index outputs_number = get_outputs_number();

    vector<string> scaling_methods_strings(outputs_number);

    #pragma omp parallel for
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


void ScalingLayer2D::set(const dimensions& new_input_dimensions)
{
    if (new_input_dimensions.size() != 1)
        throw runtime_error("Input dimensions rank is not 1");

    const Index new_inputs_number = accumulate(new_input_dimensions.begin(), new_input_dimensions.end(), 1, multiplies<Index>());

    descriptives.resize(new_inputs_number);

    //new:
    // for(Index i = 0; i < new_inputs_number; i++){
    //     set_minimum(i,type(-1.0));
    //     set_maximum(i,type(1));
    //     set_mean(i,type(0));
    //     set_standard_deviation(i,type(1));
    // }
    //end new

    scalers.resize(new_inputs_number, Scaler::MeanStandardDeviation);

    name = "scaling_layer";

    set_scalers(Scaler::MeanStandardDeviation);

    set_min_max_range(type(-1), type(1));

    layer_type = Type::Scaling2D;

}


void ScalingLayer2D::set_input_dimensions(const dimensions& new_input_dimensions)
{
    descriptives.resize(new_input_dimensions[0]);

    scalers.resize(new_input_dimensions[0], Scaler::MeanStandardDeviation);
}


void ScalingLayer2D::set_output_dimensions(const dimensions& new_output_dimensions)
{
    set_input_dimensions(new_output_dimensions);
}


void ScalingLayer2D::set_min_max_range(const type& min, const type& max)
{
    min_range = min;
    max_range = max;
}


void ScalingLayer2D::set_descriptives(const vector<Descriptives>& new_descriptives)
{
    descriptives = new_descriptives;
}


void ScalingLayer2D::set_scalers(const vector<Scaler>& new_scaling_methods)
{
    scalers = new_scaling_methods;
}


void ScalingLayer2D::set_scalers(const vector<string>& new_scaling_methods_string)
{
    const Index outputs_number = get_outputs_number();

    vector<Scaler> new_scaling_methods(outputs_number);

    #pragma omp parallel for
    for(Index i = 0; i < outputs_number; i++)
        new_scaling_methods[i] = string_to_scaler(new_scaling_methods_string[i]);

    set_scalers(new_scaling_methods);
}


void ScalingLayer2D::set_scalers(const string& new_scaling_methods_string)
{
    const Index outputs_number = get_outputs_number();

    #pragma omp parallel for
    for(Index i = 0; i < outputs_number; i++)
        scalers[i] = string_to_scaler(new_scaling_methods_string);
}


void ScalingLayer2D::set_scalers(const Scaler& new_scaling_method)
{
    const Index outputs_number = get_outputs_number();

    #pragma omp parallel for
    for(Index i = 0; i < outputs_number; i++)
        scalers[i] = new_scaling_method;
}


bool ScalingLayer2D::is_empty() const
{
    return get_output_dimensions()[0] == 0;
}


void ScalingLayer2D::forward_propagate(const vector<pair<type*, dimensions>>& input_pairs,
                                       unique_ptr<LayerForwardPropagation>& forward_propagation,
                                       const bool&)
{
    const Index outputs_number = get_outputs_number();

    ScalingLayer2DForwardPropagation* scaling_layer_forward_propagation =
        static_cast<ScalingLayer2DForwardPropagation*>(forward_propagation.get());

    const TensorMap<Tensor<type, 2>> inputs = tensor_map_2(input_pairs[0]);

    Tensor<type, 2>& outputs = scaling_layer_forward_propagation->outputs;
    outputs = inputs;

    // for(Index i = 0; i < outputs_number; i++){
    //     type mean = opennn::mean(outputs,i);
    //     Tensor<type,1> col(outputs.dimension(0));
    //     for(Index j=0;j<outputs.dimension(0);j++){
    //         col(j)=outputs(j,i);
    //     }
    //     type std_dev = opennn::standard_deviation(col);
    //     descriptives[i]={min_range, max_range,mean,std_dev};
    // }

    for(Index i = 0; i < outputs_number; i++)
    {
        const Scaler& scaler = scalers[i];
        switch(scaler)
        {
        case Scaler::None:
            continue;
        break;

        case Scaler::MinimumMaximum:
            scale_minimum_maximum(outputs, i, descriptives[i], min_range, max_range);
        break;
        case Scaler::MeanStandardDeviation:
            scale_mean_standard_deviation(outputs, i, descriptives[i]);
        break;
        case Scaler::StandardDeviation:
            scale_standard_deviation(outputs, i, descriptives[i]);
        break;
        case Scaler::Logarithm:
            scale_logarithmic(outputs, i);
        break;
        case Scaler::ImageMinMax:
            outputs.chip(i,1).device(*thread_pool_device) =  outputs.chip(i,1) / type(255);
        break;
        default:
            throw runtime_error("Unknown scaling method.\n");
        }
    }

}


string ScalingLayer2D::write_no_scaling_expression(const vector<string>& input_names, const vector<string>& output_names) const
{
    const Index inputs_number = get_output_dimensions()[0];

    ostringstream buffer;

    buffer.precision(10);

    for(Index i = 0; i < inputs_number; i++)
        buffer << output_names[i] << " = " << input_names[i] << ";\n";

    return buffer.str();
}


string ScalingLayer2D::write_minimum_maximum_expression(const vector<string>& input_names, const vector<string>& output_names) const
{
    const Index inputs_number = get_output_dimensions()[0];

    ostringstream buffer;

    buffer.precision(10);

    for(Index i = 0; i < inputs_number; i++)
        buffer << output_names[i] << " = 2*(" << input_names[i] << "-(" << descriptives[i].minimum << "))/(" << descriptives[i].maximum << "-(" << descriptives[i].minimum << "))-1;\n";

    return buffer.str();
}


string ScalingLayer2D::write_mean_standard_deviation_expression(const vector<string>& input_names, const vector<string>& output_names) const
{
    const Index inputs_number = get_inputs_number();

    ostringstream buffer;

    buffer.precision(10);

    for(Index i = 0; i < inputs_number; i++)
        buffer << output_names[i] << " = (" << input_names[i] << "-(" << descriptives[i].mean << "))/" << descriptives[i].standard_deviation << ";\n";

    return buffer.str();
}


string ScalingLayer2D::write_standard_deviation_expression(const vector<string>& input_names, const vector<string>& output_names) const
{
    const Index inputs_number = get_output_dimensions()[0];

    ostringstream buffer;

    buffer.precision(10);

    for(Index i = 0; i < inputs_number; i++)
        buffer << output_names[i] << " = " << input_names[i] << "/(" << descriptives[i].standard_deviation << ");\n";

    return buffer.str();
}


string ScalingLayer2D::get_expression(const vector<string>& new_input_names, const vector<string>&) const
{
    const vector<string> input_names = new_input_names.empty()
    ? get_default_input_names()
    : new_input_names;

    const Index outputs_number = get_outputs_number();

    ostringstream buffer;

    buffer.precision(10);

    for(Index i = 0; i < outputs_number; i++)
    {
        switch(scalers[i])
        { 
        case Scaler::None:
            buffer << "scaled_" << input_names[i] << " = " << input_names[i] << ";\n";
            break;
        case Scaler::MinimumMaximum:
            buffer << "scaled_" << input_names[i]
                   << " = " << input_names[i] << "*(" << max_range << "-" << min_range << ")/("
                   << descriptives[i].maximum << "-(" << descriptives[i].minimum << "))-" << descriptives[i].minimum << "*("
                   << max_range << "-" << min_range << ")/("
                   << descriptives[i].maximum << "-" << descriptives[i].minimum << ")+" << min_range << ";\n";
            break;
        case Scaler::MeanStandardDeviation:
            buffer << "scaled_" << input_names[i] << " = (" << input_names[i] << "-" << descriptives[i].mean << ")/" << descriptives[i].standard_deviation << ";\n";
            break;
        case Scaler::StandardDeviation:
            buffer << "scaled_" << input_names[i] << " = " << input_names[i] << "/(" << descriptives[i].standard_deviation << ");\n";
            break;
        case Scaler::Logarithm:
            buffer << "scaled_" << input_names[i] << " = log(" << input_names[i] << ");\n";
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


void ScalingLayer2D::print() const
{
    cout << "Scaling layer" << endl;

    const Index inputs_number = get_inputs_number();

    const vector<string> scalers_text = write_scalers_text();

    for(Index i = 0; i < inputs_number; i++)
    {
        cout << "Neuron " << i << endl
             << "Scaler " << scalers_text[i] << endl;

        descriptives[i].print();
    }
}


void ScalingLayer2D::to_XML(XMLPrinter& printer) const
{
    printer.OpenElement("Scaling2D");

    add_xml_element(printer, "NeuronsNumber", to_string(get_output_dimensions()[0]));

    const Index outputs_number = get_outputs_number();
    const vector<string> scaling_methods_string = write_scalers();

    for (Index i = 0; i < outputs_number; i++)
    {
        printer.OpenElement("ScalingNeuron");
        printer.PushAttribute("Index", int(i + 1));
        add_xml_element(printer, "Descriptives", tensor_to_string(descriptives[i].to_tensor()));
        add_xml_element(printer, "Scaler", scaling_methods_string[i]);

        printer.CloseElement();  
    }

    printer.CloseElement();
}


void ScalingLayer2D::from_XML(const XMLDocument& document)
{
    const XMLElement* scaling_layer_element = document.FirstChildElement("Scaling2D");

    if(!scaling_layer_element)
        throw runtime_error("Scaling2D element is nullptr.\n");

    const Index neurons_number = read_xml_index(scaling_layer_element, "NeuronsNumber");
    set({ neurons_number });

    const XMLElement* start_element = scaling_layer_element->FirstChildElement("NeuronsNumber");

    for (Index i = 0; i < neurons_number; i++) {
        const XMLElement* scaling_neuron_element = start_element->NextSiblingElement("ScalingNeuron");
        if (!scaling_neuron_element) {
            throw runtime_error("Scaling neuron " + to_string(i + 1) + " is nullptr.\n");
        }

        unsigned index = 0;
        scaling_neuron_element->QueryUnsignedAttribute("Index", &index);
        if (index != i + 1) {
            throw runtime_error("Index " + to_string(index) + " is not correct.\n");
        }

        const XMLElement* descriptives_element = scaling_neuron_element->FirstChildElement("Descriptives");

        if (!descriptives_element)
            throw runtime_error("Descriptives element " + to_string(i + 1) + " is nullptr.\n");

        if (descriptives_element->GetText()) {
            const vector<string> descriptives_string = get_tokens(descriptives_element->GetText(), " ");
            descriptives[i].set(
                type(stof(descriptives_string[0])),
                type(stof(descriptives_string[1])),
                type(stof(descriptives_string[2])),
                type(stof(descriptives_string[3]))
            );
        }

        scalers[i] = string_to_scaler(read_xml_string(scaling_neuron_element, "Scaler"));

        start_element = scaling_neuron_element;
    }
}


ScalingLayer2DForwardPropagation::ScalingLayer2DForwardPropagation(const Index& new_batch_samples_number, Layer* new_layer)
    : LayerForwardPropagation()
{
    set(new_batch_samples_number, new_layer);
}


pair<type*, dimensions> ScalingLayer2DForwardPropagation::get_outputs_pair() const
{
    const dimensions output_dimensions = layer->get_output_dimensions();

    return {(type*)outputs.data(), {batch_size, output_dimensions[0]}};
}


void ScalingLayer2DForwardPropagation::set(const Index& new_batch_size, Layer* new_layer)
{
    layer = new_layer;

    const Index outputs_number = layer->get_outputs_number();

    batch_size = new_batch_size;

    outputs.resize(batch_size, outputs_number);
}


void ScalingLayer2DForwardPropagation::print() const
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
