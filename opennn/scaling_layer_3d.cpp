//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   S C A L I N G   L A Y E R   3 D   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "registry.h"
#include "strings_utilities.h"
#include "tensors.h"
#include "statistics.h"
#include "scaling_layer_3d.h"

namespace opennn
{

Scaling3d::Scaling3d(const dimensions& new_input_dimensions) : Layer()
{
    set(new_input_dimensions);
}


dimensions Scaling3d::get_input_dimensions() const
{
    return input_dimensions;
}


dimensions Scaling3d::get_output_dimensions() const
{
    return input_dimensions;
}


vector<Descriptives> Scaling3d::get_descriptives() const
{
    return descriptives;
}


Descriptives Scaling3d::get_descriptives(const Index& index) const
{
    return descriptives[index];
}


Tensor<type, 1> Scaling3d::get_minimums() const
{
    const Index inputs_number = get_output_dimensions()[1];

    Tensor<type, 1> minimums(inputs_number);

#pragma omp parallel for
    for(Index i = 0; i < inputs_number; i++)
        minimums[i] = descriptives[i].minimum;

    return minimums;
}


Tensor<type, 1> Scaling3d::get_maximums() const
{
    const Index inputs_number = get_output_dimensions()[1];

    Tensor<type, 1> maximums(inputs_number);

#pragma omp parallel for
    for(Index i = 0; i < inputs_number; i++)
        maximums[i] = descriptives[i].maximum;

    return maximums;
}


Tensor<type, 1> Scaling3d::get_means() const
{
    const Index inputs_number = get_output_dimensions()[1];

    Tensor<type, 1> means(inputs_number);

#pragma omp parallel for
    for(Index i = 0; i < inputs_number; i++)
        means[i] = descriptives[i].mean;

    return means;
}


Tensor<type, 1> Scaling3d::get_standard_deviations() const
{
    const Index inputs_number = get_output_dimensions()[1];

    Tensor<type, 1> standard_deviations(inputs_number);

#pragma omp parallel for
    for(Index i = 0; i < inputs_number; i++)
        standard_deviations[i] = descriptives[i].standard_deviation;

    return standard_deviations;
}


vector<Scaler> Scaling3d::get_scaling_methods() const
{
    return scalers;
}


vector<string> Scaling3d::write_scalers() const
{
    const Index inputs_number = get_output_dimensions()[1];

    vector<string> scaling_methods_strings(inputs_number);

#pragma omp parallel for
    for (Index i = 0; i < inputs_number; i++)
        scaling_methods_strings[i] = scaler_to_string(scalers[i]);

    return scaling_methods_strings;
}


vector<string> Scaling3d::write_scalers_text() const
{
    const Index inputs_number = get_output_dimensions()[1];

    vector<string> scaling_methods_strings(inputs_number);

#pragma omp parallel for
    for(Index i = 0; i < inputs_number; i++)
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


void Scaling3d::set(const dimensions& new_input_dimensions)
{
    if (new_input_dimensions.size() != 2)
        throw runtime_error("Input dimensions rank is not 2 for Scaling3d (time_steps, inputs).");

    input_dimensions = new_input_dimensions;
    const Index inputs_number = new_input_dimensions[1];

    descriptives.resize(inputs_number);

    for(Index i = 0; i < inputs_number; i++)
        descriptives[i].set(type(-1.0), type(1), type(0), type(1));

    scalers.resize(inputs_number, Scaler::MeanStandardDeviation);

    label = "scaling_layer_3d";

    set_scalers(Scaler::MeanStandardDeviation);

    set_min_max_range(type(-1), type(1));

    name = "Scaling3d";

    is_trainable = false;
}


void Scaling3d::set_input_dimensions(const dimensions& new_input_dimensions)
{
    input_dimensions = new_input_dimensions;
    const Index inputs_number = new_input_dimensions[1];
    descriptives.resize(inputs_number);
    scalers.resize(inputs_number, Scaler::MeanStandardDeviation);
}


void Scaling3d::set_output_dimensions(const dimensions& new_output_dimensions)
{
    set_input_dimensions(new_output_dimensions);
}


void Scaling3d::set_min_max_range(const type& min, const type& max)
{
    min_range = min;
    max_range = max;
}


void Scaling3d::set_descriptives(const vector<Descriptives>& new_descriptives)
{
    descriptives = new_descriptives;
}


void Scaling3d::set_scalers(const vector<Scaler>& new_scaling_methods)
{
    scalers = new_scaling_methods;
}


void Scaling3d::set_scalers(const vector<string>& new_scaling_methods_string)
{
    const Index inputs_number = get_output_dimensions()[1];
    vector<Scaler> new_scaling_methods(inputs_number);

#pragma omp parallel for
    for(Index i = 0; i < inputs_number; i++)
        new_scaling_methods[i] = string_to_scaler(new_scaling_methods_string[i]);

    set_scalers(new_scaling_methods);
}


void Scaling3d::set_scalers(const string& new_scaling_methods_string)
{
    const Index inputs_number = get_output_dimensions()[1];
#pragma omp parallel for
    for(Index i = 0; i < inputs_number; i++)
        scalers[i] = string_to_scaler(new_scaling_methods_string);
}


void Scaling3d::set_scalers(const Scaler& new_scaling_method)
{
    const Index inputs_number = get_output_dimensions()[1];
#pragma omp parallel for
    for(Index i = 0; i < inputs_number; i++)
        scalers[i] = new_scaling_method;
}


void Scaling3d::forward_propagate(const vector<pair<type*, dimensions>>& input_pairs,
                                  unique_ptr<LayerForwardPropagation>& forward_propagation,
                                  const bool&)
{
    const dimensions current_input_dimensions = input_pairs[0].second;
    const Index inputs_number = current_input_dimensions[2];

    Scaling3dForwardPropagation* scaling_layer_forward_propagation =
        static_cast<Scaling3dForwardPropagation*>(forward_propagation.get());

    const TensorMap<Tensor<type, 3>> inputs = tensor_map<3>(input_pairs[0]);

    Tensor<type, 3>& outputs = scaling_layer_forward_propagation->outputs;
    outputs = inputs;

    #pragma omp parallel for
    for(Index i = 0; i < inputs_number; i++)
    {
        const Scaler& scaler = scalers[i];
        switch(scaler)
        {
        case Scaler::None:
            continue;
            break;

        case Scaler::MinimumMaximum:
            scale_minimum_maximum_3d(outputs, i, descriptives[i], min_range, max_range);
            break;
        case Scaler::MeanStandardDeviation:
            scale_mean_standard_deviation_3d(outputs, i, descriptives[i]);
            break;
        case Scaler::StandardDeviation:
            scale_standard_deviation_3d(outputs, i, descriptives[i]);
            break;
        case Scaler::Logarithm:
            scale_logarithmic_3d(outputs, i);
            break;
        default:
            throw runtime_error("Unknown scaling method.\n");
        }
    }

}


string Scaling3d::get_expression(const vector<string>&, const vector<string>&) const
{
    // This method might need a more complex implementation depending on how expressions
    // are handled for time series data. For now, returning an informative message.
    return "Expression generation for Scaling3d is not implemented yet.\n";
}


void Scaling3d::print() const
{
    cout << "Scaling 3D Layer" << endl;
    cout << "Input Dimensions (Time Steps, Inputs): (" << input_dimensions[0] << ", " << input_dimensions[1] << ")" << endl;

    const Index inputs_number = get_output_dimensions()[1];
    const vector<string> scalers_text = write_scalers_text();

    for(Index i = 0; i < inputs_number; i++)
    {
        cout << "Input Feature " << i << endl
             << "Scaler: " << scalers_text[i] << endl;

        descriptives[i].print();
    }
}


void Scaling3d::to_XML(XMLPrinter& printer) const
{
    printer.OpenElement("Scaling3d");

    add_xml_element(printer, "InputDimensions", dimensions_to_string(input_dimensions));

    const Index inputs_number = get_output_dimensions()[1];
    const vector<string> scaling_methods_string = write_scalers();

    for (Index i = 0; i < inputs_number; i++)
    {
        printer.OpenElement("ScalingFeature");
        printer.PushAttribute("Index", int(i));
        add_xml_element(printer, "Descriptives", tensor_to_string<type, 1>(descriptives[i].to_tensor()));
        add_xml_element(printer, "Scaler", scaling_methods_string[i]);
        printer.CloseElement();
    }

    printer.CloseElement();
}


void Scaling3d::from_XML(const XMLDocument& document)
{
    const XMLElement* scaling_layer_element = document.FirstChildElement("Scaling3d");

    if(!scaling_layer_element)
        throw runtime_error("Scaling3d element is nullptr.\n");

    const dimensions dims = string_to_dimensions(read_xml_string(scaling_layer_element, "InputDimensions"));
    set(dims);

    const Index inputs_number = dims[1];
    const XMLElement* start_element = scaling_layer_element->FirstChildElement("InputDimensions");

    for (Index i = 0; i < inputs_number; i++) {
        const XMLElement* scaling_feature_element = start_element->NextSiblingElement("ScalingFeature");
        if (!scaling_feature_element) {
            throw runtime_error("Scaling feature " + to_string(i) + " is nullptr.\n");
        }

        unsigned index = 0;
        scaling_feature_element->QueryUnsignedAttribute("Index", &index);
        if (index != i) {
            throw runtime_error("Index " + to_string(index) + " is not correct.\n");
        }

        const XMLElement* descriptives_element = scaling_feature_element->FirstChildElement("Descriptives");

        if (!descriptives_element)
            throw runtime_error("Descriptives element " + to_string(i) + " is nullptr.\n");

        if (descriptives_element->GetText()) {
            const vector<string> descriptives_string = get_tokens(descriptives_element->GetText(), " ");
            descriptives[i].set(
                type(stof(descriptives_string[0])),
                type(stof(descriptives_string[1])),
                type(stof(descriptives_string[2])),
                type(stof(descriptives_string[3]))
                );
        }

        scalers[i] = string_to_scaler(read_xml_string(scaling_feature_element, "Scaler"));

        start_element = scaling_feature_element;
    }
}


Scaling3dForwardPropagation::Scaling3dForwardPropagation(const Index& new_batch_size, Layer* new_layer)
    : LayerForwardPropagation()
{
    set(new_batch_size, new_layer);
}


pair<type*, dimensions> Scaling3dForwardPropagation::get_output_pair() const
{
    const dimensions output_dims = layer->get_output_dimensions();
    return {(type*)outputs.data(), {batch_size, output_dims[0], output_dims[1]}};
}


void Scaling3dForwardPropagation::set(const Index& new_batch_size, Layer* new_layer)
{
    if (!new_layer) return;

    layer = new_layer;
    batch_size = new_batch_size;

    const dimensions output_dims = layer->get_output_dimensions();
    outputs.resize(batch_size, output_dims[0], output_dims[1]);
}


void Scaling3dForwardPropagation::print() const
{
    cout << "Outputs (dimensions: " << outputs.dimension(0) << "x" << outputs.dimension(1) << "x" << outputs.dimension(2) << "):" << endl
         << outputs << endl;
}

REGISTER(Layer, Scaling3d, "Scaling3d")
REGISTER(LayerForwardPropagation, Scaling3dForwardPropagation, "Scaling3d")

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
