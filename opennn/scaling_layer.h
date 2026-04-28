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

    enum States {Minimums, Maximums, Means, StandardDeviations, Scalers};

    vector<Shape> get_state_shapes() const override
    {
        const Index features = input_shape.size();
        if (features == 0) return {};
        return {Shape{features}, Shape{features}, Shape{features}, Shape{features}, Shape{features}};
    }

    // Getters return by value (zero-copy via states[].as_vector() + Eigen move on return).
    VectorR get_minimums() const
    {
        return (ssize(states) > Minimums && states[Minimums].data) ? states[Minimums].as_vector() : VectorR();
    }

    VectorR get_maximums() const
    {
        return (ssize(states) > Maximums && states[Maximums].data) ? states[Maximums].as_vector() : VectorR();
    }

    VectorR get_means() const
    {
        return (ssize(states) > Means && states[Means].data) ? states[Means].as_vector() : VectorR();
    }

    VectorR get_standard_deviations() const
    {
        return (ssize(states) > StandardDeviations && states[StandardDeviations].data) ? states[StandardDeviations].as_vector() : VectorR();
    }

    const vector<ScalerMethod>& get_scalers() const
    {
        return scalers;
    }

    type get_min_range() const { return min_range; }
    type get_max_range() const { return max_range; }

    void set(const Shape& new_input_shape = {})
    {
        if (!new_input_shape.empty() && new_input_shape.rank() != Rank -1)
        {
           ostringstream buffer;
           buffer << "OpenNN Exception: Scaling Layer.\n"
                  << "void set(const Shape& new_input_shape) method.\n"
                  << "Input shape size must be " << Rank - 1 << ", but is " << new_input_shape.rank() << ".\n";
           throw logic_error(buffer.str());
        }

        if (new_input_shape.empty())
        {
            input_shape = {};
            name = "Scaling" + to_string(Rank) + "d";
            if constexpr (Rank == 2) layer_type = LayerType::Scaling2d;
            else if constexpr (Rank == 3) layer_type = LayerType::Scaling3d;
            else layer_type = LayerType::Scaling4d;
            is_trainable = false;
            return;
        }

        input_shape = new_input_shape;

        const Index new_inputs_number = new_input_shape.size();

        // Scaler methods are enum-valued, not float, so they can't live solely in the arena.
        // Keep as member; link_states() will write-through the float cast into states[Scalers].
        scalers.assign(new_inputs_number, ScalerMethod::MeanStandardDeviation);

        label = "scaling_layer";

        set_min_max_range(type(-1), type(1));

        name = "Scaling" + to_string(Rank) + "d";
        if constexpr (Rank == 2) layer_type = LayerType::Scaling2d;
        else if constexpr (Rank == 3) layer_type = LayerType::Scaling3d;
        else layer_type = LayerType::Scaling4d;

        is_trainable = false;
    }

    // Runs after NN::compile() allocates the states arena. Initializes descriptive
    // defaults (means=0, std=1, min=-1, max=1) and writes scaler enums as float.
    type* link_states(type* pointer) override
    {
        type* next = Layer::link_states(pointer);

        if(ssize(states) < 5) return next;

        if(states[Means].data)
            VectorMap(states[Means].data, states[Means].size()).setZero();
        if(states[StandardDeviations].data)
            VectorMap(states[StandardDeviations].data, states[StandardDeviations].size()).setOnes();
        if(states[Minimums].data)
            VectorMap(states[Minimums].data, states[Minimums].size()).setConstant(type(-1));
        if(states[Maximums].data)
            VectorMap(states[Maximums].data, states[Maximums].size()).setOnes();
        if(states[Scalers].data && ssize(scalers) == states[Scalers].size())
            for(size_t i = 0; i < scalers.size(); ++i)
                states[Scalers].data[i] = static_cast<type>(scalers[i]);

        return next;
    }

    void set_input_shape(const Shape& new_input_shape) override
    {
        set(new_input_shape);
    }

    void set_output_shape(const Shape& new_output_shape) override
    {
        set_input_shape(new_output_shape);
    }

    // Requires NN::compile() first — writes directly into the states arena.
    void set_descriptives(const vector<Descriptives>& desc)
    {
        if(ssize(states) < 5 || !states[Means].data)
            throw runtime_error("Scaling::set_descriptives: layer not compiled yet.");

        const Index n = desc.size();
        if(n != states[Means].size())
            throw runtime_error("Scaling::set_descriptives: size mismatch.");

        for(Index i = 0; i < n; ++i)
        {
            states[Means].data[i]              = desc[i].mean;
            states[StandardDeviations].data[i] = desc[i].standard_deviation;
            states[Minimums].data[i]           = desc[i].minimum;
            states[Maximums].data[i]           = desc[i].maximum;
        }
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
        flush_scalers_to_states();
    }

    void set_scalers(const string& new_scaler)
    {
        const ScalerMethod method = string_to_scaler_method(new_scaler);
        for(auto& scaler : scalers)
            scaler = method;
        flush_scalers_to_states();
    }

    void forward_propagate(ForwardPropagation& forward_propagation, size_t layer, bool) noexcept override
    {
        auto& forward_views = forward_propagation.views[layer];

        if (states.size() < 5)
        {
            copy(forward_views[Input][0], forward_views[Output][0]);
            return;
        }

        scale(forward_views[Input][0],
              states[Minimums], states[Maximums],
              states[Means], states[StandardDeviations],
              states[Scalers],
              min_range, max_range,
              forward_views[Output][0]);
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
        const type* mins = states[Minimums].data;
        const type* maxs = states[Maximums].data;
        for(Index i = 0; i < inputs_number; ++i)
            buffer << output_names[i] << " = 2*(" << input_names[i] << "-(" << mins[i]
                   << "))/(" << maxs[i] << "-(" << mins[i] << "))-1;\n";

        return buffer.str();
    }

    string write_mean_standard_deviation_expression(const vector<string>& input_names, const vector<string>& output_names) const
    {
        ostringstream buffer;
        buffer.precision(10);

        const Index inputs_number = get_output_shape().size();
        const type* mns = states[Means].data;
        const type* sds = states[StandardDeviations].data;
        for(Index i = 0; i < inputs_number; ++i)
            buffer << output_names[i] << " = (" << input_names[i] << "-(" << mns[i]
                   << "))/" << sds[i] << ";\n";

        return buffer.str();
    }

    string write_standard_deviation_expression(const vector<string>& input_names, const vector<string>& output_names) const
    {
        ostringstream buffer;
        buffer.precision(10);

        const Index inputs_number = get_output_shape().size();
        const type* sds = states[StandardDeviations].data;
        for(Index i = 0; i < inputs_number; ++i)
            buffer << output_names[i] << " = " << input_names[i] << "/(" << sds[i] << ");\n";

        return buffer.str();
    }

    // Phase 1: config only (neurons_number, scalers, min/max range).
    void from_XML(const XmlDocument& document) override
    {
        const XmlElement* scaling_layer_element = get_xml_root(document, name);

        const Index neurons_number = read_xml_index(scaling_layer_element, "NeuronsNumber");

        set({ neurons_number });

        const vector<string> scaler_names = get_tokens(read_xml_string(scaling_layer_element, "Scalers"), " ");
        scalers.resize(scaler_names.size());
        for(size_t i = 0; i < scaler_names.size(); ++i)
            scalers[i] = string_to_scaler_method(scaler_names[i]);

        min_range = type(stof(read_xml_string(scaling_layer_element, "MinRange")));
        max_range = type(stof(read_xml_string(scaling_layer_element, "MaxRange")));
    }

    // Phase 2: descriptives parsed directly into the states arena.
    void load_state_from_XML(const XmlDocument& document) override
    {
        if(ssize(states) < 5 || !states[Means].data) return;

        const XmlElement* scaling_layer_element = get_xml_root(document, name);

        VectorR tmp;
        string_to_vector(read_xml_string(scaling_layer_element, "Means"), tmp);
        if(tmp.size() == states[Means].size())
            VectorMap(states[Means].data, states[Means].size()) = tmp;

        string_to_vector(read_xml_string(scaling_layer_element, "StandardDeviations"), tmp);
        if(tmp.size() == states[StandardDeviations].size())
            VectorMap(states[StandardDeviations].data, states[StandardDeviations].size()) = tmp;

        string_to_vector(read_xml_string(scaling_layer_element, "Minimums"), tmp);
        if(tmp.size() == states[Minimums].size())
            VectorMap(states[Minimums].data, states[Minimums].size()) = tmp;

        string_to_vector(read_xml_string(scaling_layer_element, "Maximums"), tmp);
        if(tmp.size() == states[Maximums].size())
            VectorMap(states[Maximums].data, states[Maximums].size()) = tmp;
    }

    void to_XML(XmlPrinter& printer) const override
    {
        printer.open_element(name.c_str());

        vector<string> scaler_names(scalers.size());
        for(size_t i = 0; i < scalers.size(); ++i)
            scaler_names[i] = scaler_method_to_string(scalers[i]);

        write_xml(printer, {
            {"NeuronsNumber", to_string(get_outputs_number())},
            {"Means", vector_to_string(states[Means].as_vector())},
            {"StandardDeviations", vector_to_string(states[StandardDeviations].as_vector())},
            {"Minimums", vector_to_string(states[Minimums].as_vector())},
            {"Maximums", vector_to_string(states[Maximums].as_vector())},
            {"Scalers", vector_to_string(scaler_names)},
            {"MinRange", to_string(min_range)},
            {"MaxRange", to_string(max_range)}
        });

        printer.close_element();
    }

private:

    // Helper: writes the current scaler enum values into the arena (as floats).
    // Needed because ScalerMethod is non-float; setters maintain the enum member
    // and mirror it into states[Scalers] for the forward kernel.
    void flush_scalers_to_states()
    {
        if(ssize(states) <= Scalers || !states[Scalers].data) return;
        if(ssize(scalers) != states[Scalers].size()) return;
        for(size_t i = 0; i < scalers.size(); ++i)
            states[Scalers].data[i] = static_cast<type>(scalers[i]);
    }

    Shape input_shape;

    enum Forward {Input, Output};

    // Scaler method is enum, not float — can't live in the arena directly.
    vector<ScalerMethod> scalers;

    type min_range;
    type max_range;
};

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
