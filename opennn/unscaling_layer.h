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

    vector<ScalerMethod> scalers;

    float min_range = -1.0f;
    float max_range = 1.0f;

    vector<pair<Shape, Type>> get_forward_specs(const Index batch_size) const override
    {
        return {{Shape{batch_size}.append(get_output_shape()), Type::FP32}};
    }

    void flush_scalers_to_states();

public:

    Unscaling(const Shape& = {0}, const string& = "unscaling_layer");

    // Getters

    Shape get_input_shape() const override;
    Shape get_output_shape() const override;

    VectorR get_minimums() const;
    VectorR get_maximums() const;
    VectorR get_means() const;
    VectorR get_standard_deviations() const;
    const vector<ScalerMethod>& get_scalers() const { return scalers; }
    float get_min_range() const { return min_range; }
    float get_max_range() const { return max_range; }

    enum States {Minimums, Maximums, Means, StandardDeviations, Scalers};

    vector<pair<Shape, Type>> get_state_specs() const override
    {
        const Index features = ssize(scalers);
        if (features == 0) return {};
        return {
            {Shape{features}, Type::FP32}, // Minimums
            {Shape{features}, Type::FP32}, // Maximums
            {Shape{features}, Type::FP32}, // Means
            {Shape{features}, Type::FP32}, // StandardDeviations
            {Shape{features}, Type::FP32}, // Scalers
        };
    }

    float* link_states(float* pointer) override;

    // Setters

    void set(const Index = 0, const string& = "unscaling_layer");

    void set_input_shape(const Shape&) override;
    void set_output_shape(const Shape&) override;

    void set_descriptives(const vector<Descriptives>&);

    void set_min_max_range(const float, const float);

    void set_scalers(const vector<string>&);
    void set_scalers(const string&);

    // Forward propagation

    void forward_propagate(ForwardPropagation&, size_t, bool) noexcept override;

    // Serialization

    void print() const override;

    void from_JSON(const JsonDocument&) override;
    void load_state_from_JSON(const JsonDocument&) override;
    void to_JSON(JsonWriter&) const override;
};

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
