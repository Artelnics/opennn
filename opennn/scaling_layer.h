//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   S C A L I N G   L A Y E R   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#pragma once

#include <regex>
#include "statistics.h"
#include "layer.h"
#include "variable.h"
#include "math_utilities.h"
#include "forward_propagation.h"
#include "string_utilities.h"

namespace opennn
{


template<int Rank>
class Scaling final : public Layer
{

public:

    Scaling(const Shape& new_input_shape = {})
    {
        set(new_input_shape);
    }

    Shape get_input_shape() const override { return input_shape; }

    Shape get_output_shape() const override { return input_shape; }

    vector<Shape> get_forward_shapes(Index batch_size) const override
    {
        return {Shape{batch_size}.append(input_shape)}; // Output
    }

    const VectorR& get_minimums() const
    {
        return minimums;
    }

    const VectorR& get_maximums() const
    {
        return maximums;
    }

    const VectorR& get_means() const
    {
        return means;
    }

    const VectorR& get_standard_deviations() const
    {
        return standard_deviations;
    }

    const vector<ScalerMethod>& get_scalers() const
    {
        return scalers;
    }

    type get_min_range() const { return min_range; }
    type get_max_range() const { return max_range; }

    void set(const Shape& new_input_shape = {})
    {
        if (new_input_shape.rank() != Rank -1)
        {
           ostringstream buffer;
           buffer << "OpenNN Exception: Scaling Layer.\n"
                  << "void set(const Shape& new_input_shape) method.\n"
                  << "Input shape size must be " << Rank - 1 << ", but is " << new_input_shape.rank() << ".\n";
           throw logic_error(buffer.str());
        }

        input_shape = new_input_shape;

        const Index new_inputs_number = new_input_shape.size();

        means = VectorR::Zero(new_inputs_number);
        standard_deviations.resize(new_inputs_number);
        standard_deviations.setOnes();
        minimums.resize(new_inputs_number);
        minimums.setConstant(type(-1.0));
        maximums.resize(new_inputs_number);
        maximums.setOnes();
        multipliers.resize(new_inputs_number);
        offsets.resize(new_inputs_number);

        scalers.resize(new_inputs_number, ScalerMethod::MeanStandardDeviation);

        label = "scaling_layer";

        set_min_max_range(type(-1), type(1));

        calculate_coefficients();

        name = "Scaling" + to_string(Rank) + "d";
        if constexpr (Rank == 2) layer_type = LayerType::Scaling2d;
        else if constexpr (Rank == 3) layer_type = LayerType::Scaling3d;
        else layer_type = LayerType::Scaling4d;

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
        scalers.resize(new_scalers.size());
        for(size_t i = 0; i < new_scalers.size(); ++i)
            scalers[i] = string_to_scaler_method(new_scalers[i]);
    }

    void set_scalers(const string& new_scaler)
    {
        const ScalerMethod method = string_to_scaler_method(new_scaler);
        for(auto& scaler : scalers)
            scaler = method;
    }

    void forward_propagate(ForwardPropagation& forward_propagation, size_t layer, bool) noexcept override
    {
        auto& forward_views = forward_propagation.views[layer];

        copy(forward_views[Input][0], forward_views[Output][0]);
    }

    string write_no_scaling_expression(const vector<string>& input_names, const vector<string>& output_names) const
    {
        const Index inputs_number = get_output_shape().size();

        ostringstream buffer;

        buffer.precision(10);

        for(Index i = 0; i < inputs_number; ++i)
            buffer << output_names[i] << " = " << input_names[i] << ";\n";

        return buffer.str();
    }

    string write_minimum_maximum_expression(const vector<string>& input_names, const vector<string>& output_names) const
    {
        ostringstream buffer;
        buffer.precision(10);

        const Index inputs_number = get_output_shape().size();
        for(Index i = 0; i < inputs_number; ++i)
            buffer << output_names[i] << " = 2*(" << input_names[i] << "-(" << minimums[i]
                   << "))/(" << maximums[i] << "-(" << minimums[i] << "))-1;\n";

        return buffer.str();
    }

    string write_mean_standard_deviation_expression(const vector<string>& input_names, const vector<string>& output_names) const
    {
        ostringstream buffer;
        buffer.precision(10);

        const Index inputs_number = get_output_shape().size();
        for(Index i = 0; i < inputs_number; ++i)
            buffer << output_names[i] << " = (" << input_names[i] << "-(" << means[i]
                   << "))/" << standard_deviations[i] << ";\n";

        return buffer.str();
    }

    string write_standard_deviation_expression(const vector<string>& input_names, const vector<string>& output_names) const
    {
        ostringstream buffer;
        buffer.precision(10);

        const Index inputs_number = get_output_shape().size();
        for(Index i = 0; i < inputs_number; ++i)
            buffer << output_names[i] << " = " << input_names[i] << "/(" << standard_deviations[i] << ");\n";

        return buffer.str();
    }

    void from_XML(const XmlDocument& document) override
    {
        const XmlElement* scaling_layer_element = get_xml_root(document, name);

        const Index neurons_number = read_xml_index(scaling_layer_element, "NeuronsNumber");

        if constexpr (Rank == 2)
            set({ neurons_number });
        else
            set({ neurons_number });

        string_to_vector(read_xml_string(scaling_layer_element, "Means"), means);
        string_to_vector(read_xml_string(scaling_layer_element, "StandardDeviations"), standard_deviations);
        string_to_vector(read_xml_string(scaling_layer_element, "Minimums"), minimums);
        string_to_vector(read_xml_string(scaling_layer_element, "Maximums"), maximums);

        const vector<string> scaler_names = get_tokens(read_xml_string(scaling_layer_element, "Scalers"), " ");

        scalers.resize(scaler_names.size());

        for(size_t i = 0; i < scaler_names.size(); ++i)
            scalers[i] = string_to_scaler_method(scaler_names[i]);

        min_range = type(stof(read_xml_string(scaling_layer_element, "MinRange")));
        max_range = type(stof(read_xml_string(scaling_layer_element, "MaxRange")));

        calculate_coefficients();
    }

    void to_XML(XmlPrinter& printer) const override
    {
        printer.open_element(name.c_str());

        vector<string> scaler_names(scalers.size());
        for(size_t i = 0; i < scalers.size(); ++i)
            scaler_names[i] = scaler_method_to_string(scalers[i]);

        write_xml_properties(printer, {
            {"NeuronsNumber", to_string(get_outputs_number())},
            {"Means", vector_to_string(means)},
            {"StandardDeviations", vector_to_string(standard_deviations)},
            {"Minimums", vector_to_string(minimums)},
            {"Maximums", vector_to_string(maximums)},
            {"Scalers", vector_to_string(scaler_names)},
            {"MinRange", to_string(min_range)},
            {"MaxRange", to_string(max_range)}
        });

        printer.close_element();
    }

    void calculate_coefficients()
    {
        const Index n = scalers.size();
        for(Index i = 0; i < n; ++i)
        {
            switch(scalers[i])
            {
            case ScalerMethod::MeanStandardDeviation:
                multipliers[i] = 1.0f / (standard_deviations[i] + EPSILON);
                offsets[i] = -means[i] * multipliers[i];
                break;
            case ScalerMethod::MinimumMaximum:
                multipliers[i] = (max_range - min_range) / ((maximums[i] - minimums[i]) + EPSILON);
                offsets[i] = min_range - (minimums[i] * multipliers[i]);
                break;
            case ScalerMethod::ImageMinMax:
                multipliers[i] = 1.0f / 255.0f;
                offsets[i] = 0.0f;
                break;
            default: // None
                multipliers[i] = 1.0f;
                offsets[i] = 0.0f;
                break;
            }
        }
    }

private:

    Shape input_shape;

    enum Forward {Input, Output};

    VectorR means;
    VectorR standard_deviations;
    VectorR minimums;
    VectorR maximums;

    vector<ScalerMethod> scalers;

    type min_range;
    type max_range;

    VectorR multipliers;
    VectorR offsets;
};

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
