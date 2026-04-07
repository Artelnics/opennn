//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   S C A L I N G   L A Y E R   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#pragma once

#include "statistics.h"
//#include "scaling.h"
#include "string_utilities.h"
#include "math_utilities.h"
#include "neural_network.h"

namespace opennn
{

template<int Rank> struct ScalingForwardPropagationCuda;

template<int Rank>
class Scaling final : public Layer
{

public:

    Scaling(const Shape& new_input_shape = Shape(Rank - 1, 0))
    {
        set(new_input_shape);
    }


    Shape get_output_shape() const override
    {
        return input_shape;
    }


    vector<Shape> get_forward_shapes(Index batch_size) const override
    {
        return {Shape{batch_size}.append(input_shape)}; // slot 1: Output
    }


    VectorR get_minimums() const
    {
        return minimums;
    }


    VectorR get_maximums() const
    {
        return maximums;
    }


    VectorR get_means() const
    {
        return means;
    }


    VectorR get_standard_deviations() const
    {
        return standard_deviations;
    }


    vector<string> get_scalers() const
    {
        return scalers;
    }


    void set(const Shape& new_input_shape = {})
    {
        if (new_input_shape.size() != Rank -1)
        {
           ostringstream buffer;
           buffer << "OpenNN Exception: Scaling Layer.\n"
                  << "void set(const Shape& new_input_shape) method.\n"
                  << "Input shape size must be " << Rank - 1 << ", but is " << new_input_shape.size() << ".\n";
           throw logic_error(buffer.str());
        }

        input_shape = new_input_shape;

        const Index new_inputs_number = new_input_shape.count();

        means.resize(new_inputs_number);
        means.setZero();
        standard_deviations.resize(new_inputs_number);
        standard_deviations.setOnes();
        minimums.resize(new_inputs_number);
        minimums.setConstant(type(-1.0));
        maximums.resize(new_inputs_number);
        maximums.setOnes();
        multipliers.resize(new_inputs_number);
        offsets.resize(new_inputs_number);

        scalers.resize(new_inputs_number, "MeanStandardDeviation");

        label = "scaling_layer";

        set_min_max_range(type(-1), type(1));

        calculate_coefficients();

        name = "Scaling" + to_string(Rank) + "d";

        is_trainable = false;
    }


    void set_input_shape(const Shape& new_input_shape) override
    {
        set(new_input_shape);
    }


    void set_output_shape(const Shape& new_output_shape) override
    {
        set_input_shape(new_output_shape);
    }


    void set_descriptives(const vector<Descriptives>& desc)
    {
        const Index n = desc.size();
        means.resize(n);
        standard_deviations.resize(n);
        minimums.resize(n);
        maximums.resize(n);
        multipliers.resize(n);
        offsets.resize(n);

        for(Index i = 0; i < n; ++i) {
            means[i] = desc[i].mean;
            standard_deviations[i] = desc[i].standard_deviation;
            minimums[i] = desc[i].minimum;
            maximums[i] = desc[i].maximum;
        }
        calculate_coefficients();
    }

    void set_min_max_range(const type min, type max)
    {
        min_range = min;
        max_range = max;
    }


    void set_scalers(const vector<string>& new_scalers)
    {
        scalers = new_scalers;
    }


    void set_scalers(const string& new_scaler)
    {
        for(string& scaler : scalers)
            scaler = new_scaler;
    }


    void forward_propagate(ForwardPropagation& forward_propagation, size_t layer, bool) override
    {
/*
        auto inputs = as_matrix(get_input_view(forward_propagation, layer));
        auto outputs = as_matrix(get_output_view(forward_propagation, layer, 1));

        outputs.array() = (inputs.array().rowwise() * multipliers.transpose().array()).rowwise() + offsets.transpose().array();

        // Logarithm is the only non-linear one, handle separately if needed
        // but 99% of use cases are now covered by the line above.
*/
    }


    string write_no_scaling_expression(const vector<string>& input_names, const vector<string>& output_names) const
    {
        const Index inputs_number = get_output_shape().empty() ? 0 : get_output_shape().count();

        ostringstream buffer;

        buffer.precision(10);

        for(Index i = 0; i < inputs_number; i++)
            buffer << output_names[i] << " = " << input_names[i] << ";\n";

        return buffer.str();
    }


    string write_minimum_maximum_expression(const vector<string>& input_names, const vector<string>& output_names) const
    {
        const Index inputs_number = get_output_shape().empty() ? 0 : get_output_shape().count();

        ostringstream buffer;

        buffer.precision(10);
/*
        for(Index i = 0; i < inputs_number; i++)
            buffer << output_names[i] << " = 2*(" << input_names[i] << "-(" << descriptives[i].minimum << "))/(" << descriptives[i].maximum << "-(" << descriptives[i].minimum << "))-1;\n";
*/
        return buffer.str();
    }


    string write_mean_standard_deviation_expression(const vector<string>& input_names, const vector<string>& output_names) const
    {
        const Index inputs_number = get_inputs_number();

        ostringstream buffer;

        buffer.precision(10);
/*
        for(Index i = 0; i < inputs_number; i++)
            buffer << output_names[i] << " = (" << input_names[i] << "-(" << descriptives[i].mean << "))/" << descriptives[i].standard_deviation << ";\n";
*/
        return buffer.str();
    }


    string write_standard_deviation_expression(const vector<string>& input_names, const vector<string>& output_names) const
    {
        const Index inputs_number = get_output_shape().empty() ? 0 : get_output_shape().count();

        ostringstream buffer;

        buffer.precision(10);
/*
        for(Index i = 0; i < inputs_number; i++)
            buffer << output_names[i] << " = " << input_names[i] << "/(" << descriptives[i].standard_deviation << ");\n";
*/
        return buffer.str();
    }


    string get_expression(const vector<string>& new_input_names = vector<string>(), const vector<string>& = vector<string>()) const override
    {
        const vector<string> input_names = new_input_names.empty()
                                               ? get_default_feature_names()
                                               : new_input_names;

        const Index outputs_number = get_outputs_number();

        ostringstream buffer;

        buffer.precision(10);
/*
        for(Index i = 0; i < outputs_number; i++)
        {
            const string& scaler = scalers[i];

            if(scaler == "None")
                buffer << "scaled_" << input_names[i] << " = " << input_names[i] << ";\n";
            else if(scaler == "MinimumMaximum")
                buffer << "scaled_" << input_names[i]
                       << " = " << input_names[i] << "*(" << max_range << "-" << min_range << ")/("
                       << descriptives[i].maximum << "-(" << descriptives[i].minimum << "))-" << descriptives[i].minimum << "*("
                       << max_range << "-" << min_range << ")/("
                       << descriptives[i].maximum << "-" << descriptives[i].minimum << ")+" << min_range << ";\n";
            else if(scaler == "MeanStandardDeviation")
                buffer << "scaled_" << input_names[i] << " = (" << input_names[i] << "-" << descriptives[i].mean << ")/" << descriptives[i].standard_deviation << ";\n";
            else if(scaler == "StandardDeviation")
                buffer << "scaled_" << input_names[i] << " = " << input_names[i] << "/(" << descriptives[i].standard_deviation << ");\n";
            else if(scaler == "Logarithm")
                buffer << "scaled_" << input_names[i] << " = log(" << input_names[i] << ");\n";
            else
                throw runtime_error("Unknown inputs scaling method.\n");
        }
*/
        string expression = buffer.str();

        expression = regex_replace(expression, regex("\\+-"), "-");
        expression = regex_replace(expression, regex("--"), "+");

        return expression;
    }


    void print() const override
    {
        cout << "Scaling layer" << endl;

        const Index inputs_number = get_inputs_number();

        for(Index i = 0; i < inputs_number; i++)
        {
            cout << "Neuron " << i << endl
                 << "string " << scalers[i] << endl;

            //descriptives[i].print();
        }
    }


    void from_XML(const XMLDocument& document) override
    {
        const XMLElement* scaling_layer_element = get_xml_root(document, name);

        const Index neurons_number = read_xml_index(scaling_layer_element, "NeuronsNumber");
        
        if constexpr (Rank == 2)
            set({ neurons_number });
        else
            set(Shape(Rank - 1, neurons_number));

        const XMLElement* start_element = scaling_layer_element->FirstChildElement("NeuronsNumber");
/*
        for(Index i = 0; i < neurons_number; i++)
        {
            const XMLElement* scaling_neuron_element = start_element->NextSiblingElement("ScalingNeuron");
            if(!scaling_neuron_element)
                throw runtime_error("Scaling neuron " + to_string(i + 1) + " is nullptr.\n");

            const XMLElement* descriptives_element = scaling_neuron_element->FirstChildElement("Descriptives");
            if (descriptives_element && descriptives_element->GetText()) {
                const vector<string> tokens = get_tokens(descriptives_element->GetText(), " ");
                descriptives[i].set(
                    type(stof(tokens[0])),
                    type(stof(tokens[1])),
                    type(stof(tokens[2])),
                    type(stof(tokens[3]))
                    );

            }

            scalers[i] = read_xml_string(scaling_neuron_element, "Scaler");
            start_element = scaling_neuron_element;
        }
*/
    }


    void to_XML(XMLPrinter& printer) const override
    {
        printer.OpenElement(name.c_str());

        const Index outputs_number = get_outputs_number();

        add_xml_element(printer, "NeuronsNumber", to_string(outputs_number));

        for(Index i = 0; i < outputs_number; i++)
        {
/*
            printer.OpenElement("ScalingNeuron");
            printer.PushAttribute("Index", int(i + 1));
            add_xml_element(printer, "Descriptives", vector_to_string(descriptives[i].to_tensor()));
            add_xml_element(printer, "Scaler", scalers[i]);
            printer.CloseElement();
*/
        }

        printer.CloseElement();
    }

    void calculate_coefficients()
    {
        const Index n = scalers.size();
        for(Index i = 0; i < n; ++i)
        {
            const string& method = scalers[i];
            if(method == "MeanStandardDeviation") {
                multipliers[i] = 1.0f / (standard_deviations[i] + EPSILON);
                offsets[i] = -means[i] * multipliers[i];
            }
            else if(method == "MinimumMaximum") {
                multipliers[i] = (max_range - min_range) / ((maximums[i] - minimums[i]) + EPSILON);
                offsets[i] = min_range - (minimums[i] * multipliers[i]);
            }
            else if(method == "ImageMinMax") {
                multipliers[i] = 1.0f / 255.0f;
                offsets[i] = 0.0f;
            }
            else { // None
                multipliers[i] = 1.0f;
                offsets[i] = 0.0f;
            }
        }
    }

private:

    enum Forward {Output = 1}; // slot 0 = wired input (implicit for all layers)

    VectorR means;
    VectorR standard_deviations;
    VectorR minimums;
    VectorR maximums;

    VectorR multipliers;
    VectorR offsets;

    vector<string> scalers;

    type min_range;
    type max_range;
};

void reference_scaling_layer();

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
