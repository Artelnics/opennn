//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   A U T O   A S S O C I A T I V E   N E U R A L   N E T W O R K   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>

#include "tensors.h"
#include "strings_utilities.h"
#include "auto_associative_neural_network.h"

namespace opennn
{

AutoAssociativeNeuralNetwork::AutoAssociativeNeuralNetwork()
{
}


BoxPlot AutoAssociativeNeuralNetwork::get_auto_associative_distances_box_plot() const
{
    return auto_associative_distances_box_plot;
}

Descriptives AutoAssociativeNeuralNetwork::get_distances_descriptives() const
{
    return distances_descriptives;
}

type AutoAssociativeNeuralNetwork::get_box_plot_minimum() const
{
    return auto_associative_distances_box_plot.minimum;
}

type AutoAssociativeNeuralNetwork::get_box_plot_first_quartile() const
{
    return auto_associative_distances_box_plot.first_quartile;
}

type AutoAssociativeNeuralNetwork::get_box_plot_median() const
{
    return auto_associative_distances_box_plot.median;
}


type AutoAssociativeNeuralNetwork::get_box_plot_third_quartile() const
{
    return auto_associative_distances_box_plot.third_quartile;
}


type AutoAssociativeNeuralNetwork::get_box_plot_maximum() const
{
    return auto_associative_distances_box_plot.maximum;
}


Tensor<BoxPlot, 1> AutoAssociativeNeuralNetwork::get_multivariate_distances_box_plot() const
{
    return multivariate_distances_box_plot;
}


Tensor<type, 1> AutoAssociativeNeuralNetwork::get_multivariate_distances_box_plot_minimums() const
{
    Tensor<type, 1> minimum_distances(multivariate_distances_box_plot.size());

    for(Index i = 0; i < multivariate_distances_box_plot.size(); i++)
        minimum_distances(i) = multivariate_distances_box_plot(i).minimum;

    return minimum_distances;
}


Tensor<type, 1> AutoAssociativeNeuralNetwork::get_multivariate_distances_box_plot_first_quartile() const
{
    Tensor<type, 1> first_quartile_distances(multivariate_distances_box_plot.size());

    for(Index i = 0; i < multivariate_distances_box_plot.size(); i++)
        first_quartile_distances(i) = multivariate_distances_box_plot(i).first_quartile;

    return first_quartile_distances;
}


Tensor<type, 1> AutoAssociativeNeuralNetwork::get_multivariate_distances_box_plot_median() const
{
    Tensor<type, 1> median_distances(multivariate_distances_box_plot.size());

    for(Index i = 0; i < multivariate_distances_box_plot.size(); i++)
        median_distances(i) = multivariate_distances_box_plot(i).median;

    return median_distances;
}


Tensor<type, 1> AutoAssociativeNeuralNetwork::get_multivariate_distances_box_plot_third_quartile() const
{
    Tensor<type, 1> third_quartile_distances(multivariate_distances_box_plot.size());

    for(Index i = 0; i < multivariate_distances_box_plot.size(); i++)
        third_quartile_distances(i) = multivariate_distances_box_plot(i).third_quartile;

    return third_quartile_distances;
}


Tensor<type, 1> AutoAssociativeNeuralNetwork::get_multivariate_distances_box_plot_maximums() const
{
    Tensor<type, 1> maximum_distances(multivariate_distances_box_plot.size());

    for(Index i = 0; i < multivariate_distances_box_plot.size(); i++)
        maximum_distances(i) = multivariate_distances_box_plot(i).maximum;

    return maximum_distances;
}


void AutoAssociativeNeuralNetwork::set_box_plot_minimum(const type& new_box_plot_minimum)
{
    auto_associative_distances_box_plot.minimum = new_box_plot_minimum;
}


void AutoAssociativeNeuralNetwork::set_box_plot_first_quartile(const type& new_box_plot_first_quartile)
{
    auto_associative_distances_box_plot.first_quartile = new_box_plot_first_quartile;
}


void AutoAssociativeNeuralNetwork::set_box_plot_median(const type& new_box_plot_median)
{
    auto_associative_distances_box_plot.median = new_box_plot_median;
}


void AutoAssociativeNeuralNetwork::set_box_plot_third_quartile(const type& new_box_plot_third_quartile)
{
    auto_associative_distances_box_plot.third_quartile = new_box_plot_third_quartile;
}


void AutoAssociativeNeuralNetwork::set_box_plot_maximum(const type& new_box_plot_maximum)
{
    auto_associative_distances_box_plot.maximum = new_box_plot_maximum;
}


void AutoAssociativeNeuralNetwork::set_distances_box_plot(BoxPlot& new_auto_associative_distances_box_plot)
{
    auto_associative_distances_box_plot = new_auto_associative_distances_box_plot;
}


void AutoAssociativeNeuralNetwork::set_multivariate_distances_box_plot(Tensor<BoxPlot, 1>& new_multivariate_distances_box_plot)
{
    multivariate_distances_box_plot = new_multivariate_distances_box_plot;
}


void AutoAssociativeNeuralNetwork::set_distances_descriptives(Descriptives& new_distances_descriptives)
{
    distances_descriptives = new_distances_descriptives;
}


void AutoAssociativeNeuralNetwork::set_variable_distance_names(const vector<string>& new_variable_distance_names)
{
    variable_distance_names = new_variable_distance_names;
}


void AutoAssociativeNeuralNetwork::box_plot_from_XML(const XMLDocument& document)
{
    const XMLElement* root_element = document.FirstChildElement("BoxPlotDistances");

    if(!root_element)
        throw runtime_error("BoxPlotDistances element is nullptr.\n");

    // Minimum

    const XMLElement* minimum_element = root_element->FirstChildElement("Minimum");

    if(!minimum_element)
        throw runtime_error("Minimum element is nullptr.\n");

    type new_minimum = type(0);

    if(minimum_element->GetText())
        set_box_plot_minimum(type(stod(minimum_element->GetText())));

    // First Quartile

    const XMLElement* first_quartile_element = root_element->FirstChildElement("FirstQuartile");

    if(!first_quartile_element)
        throw runtime_error("FirstQuartile element is nullptr.\n");

    if(first_quartile_element->GetText())
        set_box_plot_first_quartile(type(stod(first_quartile_element->GetText())));

    // Median

    const XMLElement* median_element = root_element->FirstChildElement("Median");

    if(!median_element)
        throw runtime_error("Median element is nullptr.\n");

    if(median_element->GetText())
        set_box_plot_median(type(stod(median_element->GetText())));

    // ThirdQuartile

    const XMLElement* third_quartile_element = root_element->FirstChildElement("ThirdQuartile");

    if(!third_quartile_element)
        throw runtime_error("ThirdQuartile element is nullptr.\n");

    if(third_quartile_element->GetText())
        set_box_plot_third_quartile(type(stod(third_quartile_element->GetText())));

    // Maximum

    const XMLElement* maximum_element = root_element->FirstChildElement("Maximum");

    if(!maximum_element)
        throw runtime_error("Maximum element is nullptr.\n");

    if(maximum_element->GetText())
        set_box_plot_maximum(type(stod(maximum_element->GetText())));
}


void AutoAssociativeNeuralNetwork::distances_descriptives_from_XML(const XMLDocument& document)
{
    ostringstream buffer;

    const XMLElement* root_element = document.FirstChildElement("DistancesDescriptives");

    if(!root_element)
        throw runtime_error("DistancesDescriptives element is nullptr.\n");

    // Minimum

    const XMLElement* minimum_element = root_element->FirstChildElement("Minimum");

    if(!minimum_element)
        throw runtime_error("Minimum element is nullptr.\n");

    if(minimum_element->GetText())
        set_box_plot_minimum(type(stod(minimum_element->GetText())));

    // First Quartile

    const XMLElement* maximum_element = root_element->FirstChildElement("Maximum");

    if(!maximum_element)
        throw runtime_error("Maximum element is nullptr.\n");

    type new_maximum = type(0);

    if(maximum_element->GetText())
        set_box_plot_maximum(type(stod(maximum_element->GetText())));

    // Median

    const XMLElement* mean_element = root_element->FirstChildElement("Mean");

    if(!mean_element)
        throw runtime_error("Mean element is nullptr.\n");

    type new_mean = type(0);

    if(mean_element->GetText())
    {
        new_mean = type(stod(mean_element->GetText()));
    }

    // ThirdQuartile

    const XMLElement* standard_deviation_element = root_element->FirstChildElement("StandardDeviation");

    if(!standard_deviation_element)
        throw runtime_error("StandardDeviation element is nullptr.\n");

    type new_standard_deviation = type(0);

    if(standard_deviation_element->GetText())
        new_standard_deviation = type(stod(standard_deviation_element->GetText()));
/*
    Descriptives distances_descriptives(new_minimum, new_maximum, new_mean, new_standard_deviation);

    set_distances_descriptives(distances_descriptives);
*/
}


void AutoAssociativeNeuralNetwork::multivariate_box_plot_from_XML(const XMLDocument& document)
{
    ostringstream buffer;

    const XMLElement* multivariate_distances_element = document.FirstChildElement("MultivariateDistancesBoxPlot");

    if(!multivariate_distances_element)
        throw runtime_error("MultivariateDistancesBoxPlot element is nullptr.\n");

    const XMLElement* variables_number_element = multivariate_distances_element->FirstChildElement("VariablesNumber");

    if(!variables_number_element)
        throw runtime_error("Variables Number element is nullptr.\n");

    const Index variables_number = Index(atoi(variables_number_element->GetText()));

    multivariate_distances_box_plot.resize(variables_number);
    variable_distance_names.resize(variables_number);

    const XMLElement* start_element = variables_number_element;

    for(Index i = 0; i < variables_number; i++)
    {
        const XMLElement* variable_box_plot_element = start_element->NextSiblingElement("VariableBoxPlot");
        start_element = variable_box_plot_element;

        if(!variable_box_plot_element)
            throw runtime_error("Variable boxPlot element is nullptr.\n");

        if(variable_box_plot_element->GetText())
        {
            const char* new_box_plot_parameters_element = variable_box_plot_element->GetText();
            vector<string> splitted_box_plot_parameters_element = get_tokens(new_box_plot_parameters_element, "\\");
            variable_distance_names[i] = static_cast<string>(splitted_box_plot_parameters_element[0]);
            multivariate_distances_box_plot[i].minimum = type(stof(splitted_box_plot_parameters_element[1]));
            multivariate_distances_box_plot[i].first_quartile = type(stof(splitted_box_plot_parameters_element[2]));
            multivariate_distances_box_plot[i].median = type(stof(splitted_box_plot_parameters_element[3]));
            multivariate_distances_box_plot[i].third_quartile = type(stof(splitted_box_plot_parameters_element[4]));
            multivariate_distances_box_plot[i].maximum = type(stof(splitted_box_plot_parameters_element[5]));
        }
    }
}


string AutoAssociativeNeuralNetwork::get_expression_autoassociation_distances(string& input_variables_names, string& output_variables_names) const
{
    ostringstream buffer;

    buffer << "sample_autoassociation_distance = calculate_distances(" + input_variables_names + "," + output_variables_names + ")\n";

    string expression = buffer.str();

    replace(expression, "+-", "-");
    replace(expression, "--", "+");

    return expression;
}


string AutoAssociativeNeuralNetwork::get_expression_autoassociation_variables_distances(string& input_variables_names, string& output_variables_names) const
{
    ostringstream buffer;

    buffer << "sample_autoassociation_variables_distance = calculate_variables_distances(" + input_variables_names + "," + output_variables_names + ")\n";

    string expression = buffer.str();

    replace(expression, "+-", "-");
    replace(expression, "--", "+");

    return expression;
}


Tensor<type, 2> AutoAssociativeNeuralNetwork::calculate_multivariate_distances(type* & new_inputs_data, Tensor<Index,1>& input_dimensions,
                                                                type* & new_outputs_data, Tensor<Index,1>& output_dimensions)
{
    const Index samples_number = input_dimensions(0);
    const Index inputs_number = input_dimensions(1);

    TensorMap<Tensor<type, 2>> inputs(new_inputs_data, samples_number, inputs_number);
    TensorMap<Tensor<type, 2>> outputs(new_outputs_data, output_dimensions[0], output_dimensions(1));

    Tensor<type, 2> testing_samples_distances(samples_number, inputs_number);

    for(Index i = 0; i < samples_number; i++)
    {
        const Tensor<type, 1> input_row = inputs.chip(i, 0);
        const Tensor<type, 1> output_row = outputs.chip(i, 0);

        for(Index j = 0; j < input_row.size(); j++)
        {
            const type variable_input_value = input_row(j);
            //const TensorMap<Tensor<type, 0>> input_variable(&variable_input_value);

            const type variable_output_value = output_row(j);
            //const TensorMap<Tensor<type, 0>> output_variable(&variable_output_value);

            const type distance = l2_distance(variable_input_value, variable_output_value);

            if(!isnan(distance))
                testing_samples_distances(i, j) = distance;
        }
    }

    return testing_samples_distances;
}


Tensor<type, 1> AutoAssociativeNeuralNetwork::calculate_samples_distances(type* & new_inputs_data, Tensor<Index,1>& input_dimensions,
                                                            type* & new_outputs_data, Tensor<Index,1>& output_dimensions)
{
    const Index samples_number = input_dimensions(0);
    const Index inputs_number = input_dimensions(1);

    TensorMap<Tensor<type, 2>> inputs(new_inputs_data, samples_number, inputs_number);
    TensorMap<Tensor<type, 2>> outputs(new_outputs_data, output_dimensions[0], output_dimensions(1));

    Tensor<type, 1> distances(samples_number);
    Index distance_index = 0;

    for(Index i = 0; i < samples_number; i++)
    {
        Tensor<type, 1> input_row = inputs.chip(i, 0);
        Tensor<type, 1> output_row = outputs.chip(i, 0);

        const type distance = l2_distance(input_row, output_row)/inputs_number;

        if(!isnan(distance))
            distances(distance_index++) = l2_distance(input_row, output_row)/inputs_number;
    }

    return distances;
}


void AutoAssociativeNeuralNetwork::save_autoassociation_outputs(const Tensor<type, 1>& distances_vector,const vector<string>& types_vector, const string& file_name) const
{
    ofstream file(file_name.c_str());

    if(distances_vector.size() != types_vector.size())
        throw runtime_error("Distances and types vectors must have the same dimensions.\n");

    if(!file.is_open())
        throw runtime_error("Cannot open " + file_name + " file.\n");

    const Index samples_number = distances_vector.dimension(0);

    file << "Sample distance" << ";" << "Sample type" << "\n";

    for(Index i = 0; i < samples_number; i++)
        file << distances_vector(i) << ";" << types_vector[i] << "\n";

    file.close();
}

void AutoAssociativeNeuralNetwork::to_XML(XMLPrinter& file_stream) const
{
    ostringstream buffer;

    file_stream.OpenElement("NeuralNetwork");

    // Inputs

    file_stream.OpenElement("Inputs");

    // Inputs number

    file_stream.OpenElement("InputsNumber");
    file_stream.PushText(to_string(input_names.size()).c_str());
    file_stream.CloseElement();

    // Inputs names

    for(Index i = 0; i < input_names.size(); i++)
    {
        file_stream.OpenElement("Input");

        file_stream.PushAttribute("Index", to_string(i+1).c_str());

        file_stream.PushText(input_names[i].c_str());

        file_stream.CloseElement();
    }

    // Inputs (end tag)

    file_stream.CloseElement();

    // Layers

    file_stream.OpenElement("Layers");

    // Layers number

    file_stream.OpenElement("LayerTypes");

    buffer.str("");

    for(Index i = 0; i < Index(layers.size()); i++)
    {
        buffer << layers[i]->get_type_string();

        if(i != layers.size()-1)
            buffer << " ";
    }

    file_stream.PushText(buffer.str().c_str());

    file_stream.CloseElement();

    // Layers information

    for(Index i = 0; i < Index(layers.size()); i++)
        layers[i]->to_XML(file_stream);

    // Layers (end tag)

    file_stream.CloseElement();

    // Ouputs

    file_stream.OpenElement("Outputs");

    // Outputs number

    const Index outputs_number = output_names.size();

    file_stream.OpenElement("OutputsNumber");

    buffer.str("");
    buffer << outputs_number;

    file_stream.PushText(buffer.str().c_str());

    file_stream.CloseElement();

    // Outputs names

    for(Index i = 0; i < outputs_number; i++)
    {
        file_stream.OpenElement("Output");

        file_stream.PushAttribute("Index", to_string(i+1).c_str());

        file_stream.PushText(output_names[i].c_str());

        file_stream.CloseElement();
    }

    //Outputs (end tag)

    file_stream.CloseElement();

    // BoxPlot

    file_stream.OpenElement("BoxPlotDistances");

    // Minimum

    file_stream.OpenElement("Minimum");

    buffer.str("");
    buffer << get_box_plot_minimum();

    file_stream.PushText(buffer.str().c_str());

    file_stream.CloseElement();

    // First quartile

    file_stream.OpenElement("FirstQuartile");
    file_stream.PushText(to_string(get_box_plot_first_quartile()).c_str());
    file_stream.CloseElement();

    // Median

    file_stream.OpenElement("Median");
    file_stream.PushText(to_string(get_box_plot_median()).c_str());
    file_stream.CloseElement();

    // Third Quartile

    file_stream.OpenElement("ThirdQuartile");
    file_stream.PushText(to_string(get_box_plot_third_quartile()).c_str());
    file_stream.CloseElement();

    // Maximum

    file_stream.OpenElement("Maximum");
    file_stream.PushText(to_string(get_box_plot_maximum()).c_str());
    file_stream.CloseElement();

    //BoxPlotDistances (end tag)

    file_stream.CloseElement();

    // DistancesDescriptives

    Descriptives distances_descriptives = get_distances_descriptives();

    file_stream.OpenElement("DistancesDescriptives");

    // Minimum

    file_stream.OpenElement("Minimum");

    buffer.str("");
    buffer << distances_descriptives.minimum;

    file_stream.PushText(buffer.str().c_str());

    file_stream.CloseElement();

    // First quartile

    file_stream.OpenElement("Maximum");

    buffer.str("");
    buffer << distances_descriptives.maximum;

    file_stream.PushText(buffer.str().c_str());

    file_stream.CloseElement();

    // Median

    file_stream.OpenElement("Mean");

    buffer.str("");
    buffer << distances_descriptives.mean;

    file_stream.PushText(buffer.str().c_str());

    file_stream.CloseElement();

    // Third Quartile

    file_stream.OpenElement("StandardDeviation");

    buffer.str("");
    buffer << distances_descriptives.standard_deviation;

    file_stream.PushText(buffer.str().c_str());

    file_stream.CloseElement();

    //DistancesDescriptives (end tag)

    file_stream.CloseElement();

    // Multivariate BoxPlot

    file_stream.OpenElement("MultivariateDistancesBoxPlot");

    // Variables Number

    file_stream.OpenElement("VariablesNumber");

    buffer.str("");
    buffer << variable_distance_names.size();

    file_stream.PushText(buffer.str().c_str());

    file_stream.CloseElement();

    for(Index i = 0; i < variable_distance_names.size(); i++)
    {
        // Scaling neuron

        file_stream.OpenElement("VariableBoxPlot");

        buffer.str(""); buffer << variable_distance_names[i].c_str();
        file_stream.PushText(buffer.str().c_str());
        file_stream.PushText("\\");

        buffer.str(""); buffer << multivariate_distances_box_plot(i).minimum;
        file_stream.PushText(buffer.str().c_str());
        file_stream.PushText("\\");

        buffer.str(""); buffer << multivariate_distances_box_plot(i).first_quartile;
        file_stream.PushText(buffer.str().c_str());
        file_stream.PushText("\\");

        buffer.str(""); buffer << multivariate_distances_box_plot(i).median;
        file_stream.PushText(buffer.str().c_str());
        file_stream.PushText("\\");

        buffer.str(""); buffer << multivariate_distances_box_plot(i).third_quartile;
        file_stream.PushText(buffer.str().c_str());
        file_stream.PushText("\\");

        buffer.str(""); buffer << multivariate_distances_box_plot(i).maximum;
        file_stream.PushText(buffer.str().c_str());

        // VariableBoxPlot (end tag)

        file_stream.CloseElement();
    }

    // MultivariateDistancesBoxPlot (end tag)

    file_stream.CloseElement();

    // Neural network (end tag)

    file_stream.CloseElement();
}


void AutoAssociativeNeuralNetwork::from_XML(const XMLDocument& document)
{
    ostringstream buffer;

    const XMLElement* root_element = document.FirstChildElement("NeuralNetwork");

    if(!root_element)
        throw runtime_error("Neural network element is nullptr.\n");

    // Inputs
    {
        const XMLElement* element = root_element->FirstChildElement("Inputs");

        if(element)
        {
            XMLDocument inputs_document;
            inputs_document.InsertFirstChild(element->DeepClone(&inputs_document));
            inputs_from_XML(inputs_document);
        }
    }

    // Layers
    {
        const XMLElement* element = root_element->FirstChildElement("Layers");

        if(element)
        {
            XMLDocument layers_document;
            layers_document.InsertFirstChild(element->DeepClone(&layers_document));
            layers_from_XML(layers_document);
        }
    }

    // Outputs
    {
        const XMLElement* element = root_element->FirstChildElement("Outputs");

        if(element)
        {
            XMLDocument outputs_document;
            outputs_document.InsertFirstChild(element->DeepClone(&outputs_document));
            outputs_from_XML(outputs_document);
        }
    }


    {
        const XMLElement* element = root_element->FirstChildElement("BoxPlotDistances");

        if(element)
        {
            XMLDocument box_plot_document;
            box_plot_document.InsertFirstChild(element->DeepClone(&box_plot_document));
            box_plot_from_XML(box_plot_document);
        }
    }

    {
        const XMLElement* element = root_element->FirstChildElement("DistancesDescriptives");

        if(element)
        {
            XMLDocument distances_descriptives_document;
            distances_descriptives_document.InsertFirstChild(element->DeepClone(&distances_descriptives_document));
            distances_descriptives_from_XML(distances_descriptives_document);
        }
    }

    const XMLElement* element = root_element->FirstChildElement("MultivariateDistancesBoxPlot");

    if(element)
    {
        XMLDocument multivariate_box_plot_document;
        multivariate_box_plot_document.InsertFirstChild(element->DeepClone(&multivariate_box_plot_document));

        multivariate_box_plot_from_XML(multivariate_box_plot_document);
    }

    // Display

    const XMLElement* display_element = root_element->FirstChildElement("Display");

    if(display_element)
        set_display(display_element->GetText() != string("0"));
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
