//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   U N S C A L I N G   L A Y E R   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#pragma once

#include "layer.h"
#include "operators.h"
#include "scaling.h"
#include "statistics.h"
#include "variable.h"

namespace opennn
{

/// @brief Output unscaling layer that reverts normalization back to the original feature ranges.
class Unscaling final : public Layer
{
public:

    /// @brief Constructs an unscaling layer for the given number of outputs and label.
    Unscaling(const Shape& = {0}, const string& = "unscaling_layer");

    Shape get_input_shape()  const override { return { Index(scalers.size()) }; }
    Shape get_output_shape() const override { return { Index(scalers.size()) }; }

    const vector<Descriptives>& get_descriptives() const { return descriptives; }
    const vector<ScalerMethod>& get_scalers()      const { return scalers; }

    /// @brief Returns the per-variable minimum values from the stored descriptive statistics.
    VectorR get_minimums()            const;
    /// @brief Returns the per-variable maximum values from the stored descriptive statistics.
    VectorR get_maximums()            const;
    /// @brief Returns the per-variable means from the stored descriptive statistics.
    VectorR get_means()               const;
    /// @brief Returns the per-variable standard deviations from the stored descriptive statistics.
    VectorR get_standard_deviations() const;

    float get_min_range() const { return min_range; }
    float get_max_range() const { return max_range; }

    /// @brief Reconfigures the layer with a new number of outputs and label.
    void set(Index = 0, const string& = "unscaling_layer");

    /// @copydoc Layer::set_input_shape
    void set_input_shape(const Shape&) override;
    /// @copydoc Layer::set_output_shape
    void set_output_shape(const Shape&) override;

    /// @brief Sets the descriptive statistics (min, max, mean, stddev) used for unscaling each variable.
    void set_descriptives(const vector<Descriptives>&);
    /// @brief Sets the source range expected by min-max unscaling methods.
    void set_min_max_range(float min, float max);
    /// @brief Sets the unscaling method for each output variable from a vector of method names.
    void set_scalers(const vector<string>&);
    /// @brief Sets the same unscaling method for all output variables from its name.
    void set_scalers(const string&);

    /// @copydoc Layer::link_states
    float* link_states(float*) override;

    /// @copydoc Layer::read_JSON_body
    void read_JSON_body(const Json*) override;
    /// @copydoc Layer::write_JSON_body
    void write_JSON_body(JsonWriter&) const override;

    /// @copydoc Layer::write_expression
    string write_expression(const vector<string>& input_names,
                            const vector<string>& output_names) const override;

private:

    vector<Descriptives> descriptives;
    vector<ScalerMethod> scalers;
    float min_range = -1.0f;
    float max_range = 1.0f;

    Buffer op_storage;
    bool   op_storage_dirty = true;

    UnscaleOp unscale_op;

    void refresh_op_storage(Device device);
};

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
