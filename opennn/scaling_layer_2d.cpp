//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   S C A L I N G   L A Y E R   2 D   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include <numeric>

#include "scaling_layer_2d.h"
#include "strings_utilities.h"
#include "tensors.h"

namespace opennn
{

ScalingLayer2D::ScalingLayer2D(const dimensions& new_input_dimensions) : Layer()
{
    set(new_input_dimensions);
}


dimensions ScalingLayer2D::get_input_dimensions() const
{
    return dimensions{scalers.size()};
}


dimensions ScalingLayer2D::get_output_dimensions() const
{
    return dimensions{scalers.size()};
}


Tensor<Descriptives, 1> ScalingLayer2D::get_descriptives() const
{
    return descriptives;
}


Descriptives ScalingLayer2D::get_descriptives(const Index& index) const
{
    return descriptives(index);
}


Tensor<type, 1> ScalingLayer2D::get_minimums() const
{
    const Index neurons_number = get_output_dimensions()[0];

    Tensor<type, 1> minimums(neurons_number);

    #pragma omp parallel for
    for(Index i = 0; i < neurons_number; i++)
        minimums[i] = descriptives[i].minimum;

    return minimums;
}


Tensor<type, 1> ScalingLayer2D::get_maximums() const
{
    const Index neurons_number = get_output_dimensions()[0];

    Tensor<type, 1> maximums(neurons_number);

    #pragma omp parallel for
    for(Index i = 0; i < neurons_number; i++)
        maximums[i] = descriptives[i].maximum;

    return maximums;
}


Tensor<type, 1> ScalingLayer2D::get_means() const
{
    const Index neurons_number = get_output_dimensions()[0];

    Tensor<type, 1> means(neurons_number);

    #pragma omp parallel for
    for(Index i = 0; i < neurons_number; i++)
        means[i] = descriptives[i].mean;

    return means;
}


Tensor<type, 1> ScalingLayer2D::get_standard_deviations() const
{
    const Index neurons_number = get_output_dimensions()[0];

    Tensor<type, 1> standard_deviations(neurons_number);

    #pragma omp parallel for
    for(Index i = 0; i < neurons_number; i++)
        standard_deviations[i] = descriptives[i].standard_deviation;

    return standard_deviations;
}


Tensor<Scaler, 1> ScalingLayer2D::get_scaling_methods() const
{
    return scalers;
}


Tensor<string, 1> ScalingLayer2D::write_scalers() const
{
    const Index neurons_number = get_output_dimensions()[0];

    Tensor<string, 1> scaling_methods_strings(neurons_number);

    #pragma omp parallel for
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
            throw runtime_error("Unknown " + to_string(i) + " scaling method.\n");

    return scaling_methods_strings;
}


Tensor<string, 1> ScalingLayer2D::write_scalers_text() const
{
    const Index neurons_number = get_output_dimensions()[0];

    Tensor<string, 1> scaling_methods_strings(neurons_number);

    #pragma omp parallel for
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


void ScalingLayer2D::set(const dimensions& new_input_dimensions)
{
    if (new_input_dimensions.size() != 1)
        throw runtime_error("Input dimensions rank is not 1");

    const Index new_inputs_number = accumulate(new_input_dimensions.begin(), new_input_dimensions.end(), 1, multiplies<Index>());

    descriptives.resize(new_inputs_number);

    scalers.resize(new_inputs_number);
    scalers.setConstant(Scaler::MeanStandardDeviation);

    name = "scaling_layer";

    set_scalers(Scaler::MeanStandardDeviation);

    set_min_max_range(type(-1), type(1));

    layer_type = Type::Scaling2D;
}


void ScalingLayer2D::set_input_dimensions(const dimensions& new_input_dimensions)
{
    descriptives.resize(new_input_dimensions[0]);

    scalers.resize(new_input_dimensions[0]);

    scalers.setConstant(Scaler::MeanStandardDeviation);
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


void ScalingLayer2D::set_descriptives(const Tensor<Descriptives, 1>& new_descriptives)
{
    descriptives = new_descriptives;
}


void ScalingLayer2D::set_item_descriptives(const Index& i, const Descriptives& item_descriptives)
{
    descriptives(i) = item_descriptives;
}


void ScalingLayer2D::set_minimum(const Index& i, const type& new_minimum)
{
    descriptives(i).set_minimum(new_minimum);
}


void ScalingLayer2D::set_maximum(const Index& i, const type& new_maximum)
{
    descriptives(i).set_maximum(new_maximum);
}


void ScalingLayer2D::set_mean(const Index& i, const type& new_mean)
{
    descriptives(i).set_mean(new_mean);
}


void ScalingLayer2D::set_standard_deviation(const Index& i, const type& new_standard_deviation)
{
    descriptives(i).set_standard_deviation(new_standard_deviation);
}


void ScalingLayer2D::set_scalers(const Tensor<Scaler, 1>& new_scaling_methods)
{
    scalers = new_scaling_methods;
}


void ScalingLayer2D::set_scalers(const Tensor<string, 1>& new_scaling_methods_string)
{
    const Index neurons_number = get_output_dimensions()[0];

    Tensor<Scaler, 1> new_scaling_methods(neurons_number);

    #pragma omp parallel for
    for(Index i = 0; i < neurons_number; i++)
        if(new_scaling_methods_string(i) == "None")
            new_scaling_methods(i) = Scaler::None;
        else if(new_scaling_methods_string(i) == "MinimumMaximum")
            new_scaling_methods(i) = Scaler::MinimumMaximum;
        else if(new_scaling_methods_string(i) == "MeanStandardDeviation")
            new_scaling_methods(i) = Scaler::MeanStandardDeviation;
        else if(new_scaling_methods_string(i) == "StandardDeviation")
            new_scaling_methods(i) = Scaler::StandardDeviation;
        else if(new_scaling_methods_string(i) == "Logarithm")
            new_scaling_methods(i) = Scaler::Logarithm;
        else
            throw runtime_error("Unknown scaling method: " + new_scaling_methods_string[i] + ".\n");

    set_scalers(new_scaling_methods);
}


void ScalingLayer2D::set_scaler(const Index& variable_index, const Scaler& new_scaler)
{
    scalers(variable_index) = new_scaler;
}


void ScalingLayer2D::set_scaler(const Index& variable_index, const string& new_scaler_string)
{
    if(new_scaler_string == "None")
        scalers(variable_index) = Scaler::None;
    else if(new_scaler_string == "MeanStandardDeviation")
        scalers(variable_index) = Scaler::MeanStandardDeviation;
    else if(new_scaler_string == "MinimumMaximum")
        scalers(variable_index) = Scaler::MinimumMaximum;
    else if(new_scaler_string == "StandardDeviation")
        scalers(variable_index) = Scaler::StandardDeviation;
    else if(new_scaler_string == "Logarithm")
        scalers(variable_index) = Scaler::Logarithm;
    else
        throw runtime_error("Unknown scaling method: " + new_scaler_string + ".\n");
}


void ScalingLayer2D::set_scalers(const string& new_scaling_methods_string)
{
    const Index neurons_number = get_output_dimensions()[0];

    #pragma omp parallel for
    for(Index i = 0; i < neurons_number; i++)
        set_scaler(i, new_scaling_methods_string);
}


void ScalingLayer2D::set_scalers(const Scaler& new_scaling_method)
{
    const Index neurons_number = get_output_dimensions()[0];

    #pragma omp parallel for
    for(Index i = 0; i < neurons_number; i++)
        scalers(i) = new_scaling_method;
}


bool ScalingLayer2D::is_empty() const
{
    return get_output_dimensions()[0] == 0;
}


void ScalingLayer2D::forward_propagate(const vector<pair<type*, dimensions>>& input_pairs,
                                       unique_ptr<LayerForwardPropagation>& forward_propagation,
                                       const bool& is_training)
{
    const Index neurons_number = get_output_dimensions()[0];

    ScalingLayer2DForwardPropagation* scaling_layer_forward_propagation =
        static_cast<ScalingLayer2DForwardPropagation*>(forward_propagation.get());

    const TensorMap<Tensor<type, 2>> inputs = tensor_map_2(input_pairs[0]);

    Tensor<type, 2>& outputs = scaling_layer_forward_propagation->outputs;

    for(Index i = 0; i < neurons_number; i++)
    {
        const Scaler& scaler = scalers(i);

        // @todo What's going on with this?

        //const TensorMap<Tensor<type, 1>> input_column = tensor_map(inputs, i);

        const TensorMap<Tensor<type, 1>> input_column((type*) inputs.data() + i * inputs.dimension(0),
                                                      inputs.dimension(0));

        TensorMap<Tensor<type, 1>> output_column = tensor_map(outputs, i);
        
        if(abs(descriptives(i).standard_deviation) < type(NUMERIC_LIMITS_MIN))
        {
            if(display)
                cout << "OpenNN Warning: ScalingLayer2D class.\n"
                     << "forward_propagate method.\n"
                     << "Standard deviation of variable " << i << " is zero.\n"
                     << "Those variables won't be scaled.\n";

            continue;
        }

        switch(scaler)
        {
        case Scaler::None:
            output_column.device(*thread_pool_device) = input_column;
        break;
        case Scaler::MinimumMaximum:
        {
            const type slope =
                    (max_range-min_range)/(descriptives(i).maximum-descriptives(i).minimum);

            const type intercept =
                    (min_range*descriptives(i).maximum-max_range*descriptives(i).minimum)/(descriptives(i).maximum-descriptives(i).minimum);

            output_column.device(*thread_pool_device) = intercept + slope * input_column;
        }
        break;
        case Scaler::MeanStandardDeviation:
        {
            const type slope = type(1)/descriptives(i).standard_deviation;

            const type intercept = -descriptives(i).mean/descriptives(i).standard_deviation;

            output_column.device(*thread_pool_device) = intercept + slope*input_column;
        }
        break;
        case Scaler::StandardDeviation:
            output_column.device(*thread_pool_device) = type(1/descriptives(i).standard_deviation)*input_column;
            break;
        case Scaler::Logarithm:
            output_column.device(*thread_pool_device) = input_column.log();
            break;
        case Scaler::ImageMinMax:
            output_column.device(*thread_pool_device) = input_column/type(255);
            break;
        default:
            throw runtime_error("Unknown scaling method.\n");
        }
    }
}


string ScalingLayer2D::write_no_scaling_expression(const Tensor<string, 1>& input_names, const Tensor<string, 1>& output_names) const
{
    const Index inputs_number = get_output_dimensions()[0];

    ostringstream buffer;

    buffer.precision(10);

    for(Index i = 0; i < inputs_number; i++)
        buffer << output_names(i) << " = " << input_names(i) << ";\n";

    return buffer.str();
}


string ScalingLayer2D::write_minimum_maximum_expression(const Tensor<string, 1>& input_names, const Tensor<string, 1>& output_names) const
{
    const Index inputs_number = get_output_dimensions()[0];

    ostringstream buffer;

    buffer.precision(10);

    for(Index i = 0; i < inputs_number; i++)
        buffer << output_names(i) << " = 2*(" << input_names(i) << "-(" << descriptives(i).minimum << "))/(" << descriptives(i).maximum << "-(" << descriptives(i).minimum << "))-1;\n";

    return buffer.str();
}


string ScalingLayer2D::write_mean_standard_deviation_expression(const Tensor<string, 1>& input_names, const Tensor<string, 1>& output_names) const
{
    const Index inputs_number = get_input_dimensions()[0];

    ostringstream buffer;

    buffer.precision(10);

    for(Index i = 0; i < inputs_number; i++)
        buffer << output_names(i) << " = (" << input_names(i) << "-(" << descriptives(i).mean << "))/" << descriptives(i).standard_deviation << ";\n";

    return buffer.str();
}


string ScalingLayer2D::write_standard_deviation_expression(const Tensor<string, 1>& input_names, const Tensor<string, 1>& output_names) const
{
    const Index inputs_number = get_output_dimensions()[0];

    ostringstream buffer;

    buffer.precision(10);

    for(Index i = 0; i < inputs_number; i++)
        buffer << output_names(i) << " = " << input_names(i) << "/(" << descriptives(i).standard_deviation << ");\n";

    return buffer.str();
}


string ScalingLayer2D::get_expression(const Tensor<string, 1>& input_names, const Tensor<string, 1>&) const
{
    const Index neurons_number = get_output_dimensions()[0];

    ostringstream buffer;

    buffer.precision(10);

    for(Index i = 0; i < neurons_number; i++)
    {
        switch(scalers(i))
        { 
        case Scaler::None:
            buffer << "scaled_" << input_names(i) << " = " << input_names(i) << ";\n";
            break;
        case Scaler::MinimumMaximum:
            buffer << "scaled_" << input_names(i) 
                   << " = " << input_names(i) << "*(" << max_range << "-" << min_range << ")/("
                   << descriptives(i).maximum << "-(" << descriptives(i).minimum << "))-" << descriptives(i).minimum << "*("
                   << max_range << "-" << min_range << ")/("
                   << descriptives(i).maximum << "-" << descriptives(i).minimum << ")+" << min_range << ";\n";
            break;
        case Scaler::MeanStandardDeviation:
            buffer << "scaled_" << input_names(i) << " = (" << input_names(i) << "-" << descriptives(i).mean << ")/" << descriptives(i).standard_deviation << ";\n";
            break;
        case Scaler::StandardDeviation:
            buffer << "scaled_" << input_names(i) << " = " << input_names(i) << "/(" << descriptives(i).standard_deviation << ");\n";
            break;
        case Scaler::Logarithm:
            buffer << "scaled_" << input_names(i) << " = log(" << input_names(i) << ");\n";
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

    const Index inputs_number = get_input_dimensions()[0];

    const Tensor<string, 1> scalers_text = write_scalers_text();

    for(Index i = 0; i < inputs_number; i++)
    {
        cout << "Neuron " << i << endl
             << "Scaler " << scalers_text(i) << endl;

        descriptives(i).print();
    }
}


void ScalingLayer2D::to_XML(tinyxml2::XMLPrinter& printer) const
{
    printer.OpenElement("Scaling2D");

    add_xml_element(printer, "Name", name);
    add_xml_element(printer, "NeuronsNumber", to_string(get_output_dimensions()[0]));

    const Index neurons_number = get_output_dimensions()[0];
    const Tensor<string, 1> scaling_methods_string = write_scalers();

    for (Index i = 0; i < neurons_number; i++) 
    {
        printer.OpenElement("ScalingNeuron");
        printer.PushAttribute("Index", int(i + 1));
        add_xml_element(printer, "Descriptives", tensor_to_string(descriptives(i).to_tensor()));
        add_xml_element(printer, "Scaler", scaling_methods_string(i));

        printer.CloseElement();  
    }

    printer.CloseElement();
}


void ScalingLayer2D::from_XML(const tinyxml2::XMLDocument& document)
{
    const tinyxml2::XMLElement* scaling_layer_element = document.FirstChildElement("Scaling2D");

    if(!scaling_layer_element)
        throw runtime_error("Scaling2D element is nullptr.\n");

    set_name(read_xml_string(scaling_layer_element, "Name"));

    const Index neurons_number = read_xml_index(scaling_layer_element, "NeuronsNumber");
    set({ neurons_number });

    const tinyxml2::XMLElement* start_element = scaling_layer_element->FirstChildElement("NeuronsNumber");

    for (Index i = 0; i < neurons_number; i++) {
        const tinyxml2::XMLElement* scaling_neuron_element = start_element->NextSiblingElement("ScalingNeuron");
        if (!scaling_neuron_element) {
            throw runtime_error("Scaling neuron " + std::to_string(i + 1) + " is nullptr.\n");
        }

        // Verify neuron index
        unsigned index = 0;
        scaling_neuron_element->QueryUnsignedAttribute("Index", &index);
        if (index != i + 1) {
            throw runtime_error("Index " + std::to_string(index) + " is not correct.\n");
        }

        // Descriptives
        const tinyxml2::XMLElement* descriptives_element = scaling_neuron_element->FirstChildElement("Descriptives");
        if (!descriptives_element) {
            throw runtime_error("Descriptives element " + std::to_string(i + 1) + " is nullptr.\n");
        }
        if (descriptives_element->GetText()) {
            const Tensor<string, 1> descriptives_string = get_tokens(descriptives_element->GetText(), " ");
            descriptives[i].set(
                type(stof(descriptives_string[0])),
                type(stof(descriptives_string[1])),
                type(stof(descriptives_string[2])),
                type(stof(descriptives_string[3]))
            );
        }

        const tinyxml2::XMLElement* scaling_method_element = scaling_neuron_element->FirstChildElement("Scaler");
        if (!scaling_method_element) {
            throw runtime_error("Scaling method element " + std::to_string(i + 1) + " is nullptr.\n");
        }
        set_scaler(i, scaling_method_element->GetText());

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

    return {(type*)outputs.data(), {batch_samples_number, output_dimensions[0]}};
}


void ScalingLayer2DForwardPropagation::set(const Index& new_batch_samples_number, Layer* new_layer)
{
    layer = new_layer;

    const Index neurons_number = layer->get_output_dimensions()[0];

    batch_samples_number = new_batch_samples_number;

    outputs.resize(batch_samples_number, neurons_number);
}


void ScalingLayer2DForwardPropagation::print() const
{
    cout << "Outputs:" << endl
         << outputs << endl;
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
