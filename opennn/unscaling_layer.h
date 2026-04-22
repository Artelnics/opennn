//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   U N S C A L I N G   L A Y E R   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#pragma once

#include "layer.h"
#include "scaling.h"
#include "variable.h"

namespace opennn
{

class Unscaling final : public Layer
{
private:

    enum Forward {Input, Output};

    // Scaler method is enum, not float — can't live in the arena directly.
    vector<ScalerMethod> scalers;

    type min_range = -1.0f;
    type max_range = 1.0f;

    vector<Shape> get_forward_shapes(const Index batch_size) const override
    {
        return {Shape{batch_size}.append(get_output_shape())};
    }

    void flush_scalers_to_states();

public:

    Unscaling(const Shape& = {0}, const string& = "unscaling_layer");

    // Getters

    Shape get_input_shape() const override;
    Shape get_output_shape() const override;

    // Return by value — zero-copy via states[].as_vector() + Eigen move on return.
    VectorR get_minimums() const;
    VectorR get_maximums() const;
    VectorR get_means() const;
    VectorR get_standard_deviations() const;
    const vector<ScalerMethod>& get_scalers() const { return scalers; }
    type get_min_range() const { return min_range; }
    type get_max_range() const { return max_range; }

    enum States {Minimums, Maximums, Means, StandardDeviations, Scalers};

    vector<Shape> get_state_shapes() const override
    {
        const Index features = ssize(scalers);
        if (features == 0) return {};
        return {Shape{features}, Shape{features}, Shape{features}, Shape{features}, Shape{features}};
    }

    type* link_states(type* pointer) override;

    // Setters

    void set(const Index = 0, const string& = "unscaling_layer");

    void set_input_shape(const Shape&) override;
    void set_output_shape(const Shape&) override;

    // Requires NN::compile() first — writes directly into the states arena.
    void set_descriptives(const vector<Descriptives>&);

    void set_min_max_range(const type, const type);

    void set_scalers(const vector<string>&);
    void set_scalers(const string&);

    // Forward propagation

    void forward_propagate(ForwardPropagation&, size_t, bool) noexcept override;

    // Serialization (two-phase: from_XML parses config; load_state_from_XML parses descriptives after compile)

    void print() const override;

    void from_XML(const XmlDocument&) override;
    void load_state_from_XML(const XmlDocument&) override;
    void to_XML(XmlPrinter&) const override;
};

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
