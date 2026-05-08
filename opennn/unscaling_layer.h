//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   U N S C A L I N G   L A Y E R   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

/**
 * @file unscaling_layer.h
 * @brief Declares the Unscaling layer: inverse of the Scaling layer,
 *        applied to network outputs to convert them back to physical units.
 */

#pragma once

#include "layer.h"
#include "operators.h"
#include "scaling.h"
#include "variable.h"

namespace opennn
{

/**
 * @class Unscaling
 * @brief Per-output inverse normalization layer.
 *
 * Mirror image of Scaling: applies the inverse of the configured scaler
 * method to each output feature so that the network's predictions are
 * returned in the original units. Holds one ScalerMethod per output and
 * the corresponding descriptive statistics.
 *
 * The layer has no trainable parameters.
 */
class Unscaling final : public Layer
{
public:

    /**
     * @brief Constructs an Unscaling layer.
     * @param input_shape Per-sample input shape (also the output shape).
     * @param label Human-readable label assigned to this layer.
     */
    Unscaling(const Shape& input_shape = {0},
              const string& label = "unscaling_layer");

    /** @brief Returns the per-sample input shape. */
    Shape get_input_shape() const override;
    /** @brief Returns the per-sample output shape (same as input). */
    Shape get_output_shape() const override;

    /** @brief Returns the per-output minimum statistics. */
    VectorR get_minimums() const;
    /** @brief Returns the per-output maximum statistics. */
    VectorR get_maximums() const;
    /** @brief Returns the per-output mean statistics. */
    VectorR get_means() const;
    /** @brief Returns the per-output standard-deviation statistics. */
    VectorR get_standard_deviations() const;

    /** @brief Read-only access to the per-output scaler method list. */
    const vector<ScalerMethod>& get_scalers() const { return scalers; }
    /** @brief Lower bound of the source range used by min-max unscalers. */
    float get_min_range() const { return unscale_op.min_range; }
    /** @brief Upper bound of the source range used by min-max unscalers. */
    float get_max_range() const { return unscale_op.max_range; }

    /** @brief Returns the single Unscale operator that implements this layer. */
    vector<Operator*> get_operators() override { return {&unscale_op}; }

    /**
     * @brief Re-initializes the layer.
     * @param outputs_number Number of output features.
     * @param label Human-readable label.
     */
    void set(Index outputs_number = 0,
             const string& label = "unscaling_layer");

    /** @brief Updates the input shape and resizes the scaler vector. */
    void set_input_shape(const Shape&) override;
    /** @brief Updates the output shape (kept equal to the input shape). */
    void set_output_shape(const Shape&) override;

    /**
     * @brief Sets the per-output descriptive statistics used by the unscalers.
     *
     * Receives a vector of Descriptives (minimum, maximum, mean, standard
     * deviation) with one entry per output.
     */
    void set_descriptives(const vector<Descriptives>&);

    /**
     * @brief Sets the source range used by min-max unscalers.
     *
     * Receives the lower bound followed by the upper bound (i.e. the
     * range that the network produced and that should be inverted).
     */
    void set_min_max_range(float, float);

    /**
     * @brief Sets the unscaler method per output from a vector of names.
     *
     * Receives one method name per output; see ScalerMethod for the
     * supported set.
     */
    void set_scalers(const vector<string>&);
    /**
     * @brief Sets the same unscaler method for every output.
     *
     * Receives a single method name applied to all outputs.
     */
    void set_scalers(const string&);

    /**
     * @brief Forward pass: applies the inverse scaler to each output.
     *
     * Receives the ForwardPropagation buffer slice, this layer's index and
     * the training flag.
     */
    void forward_propagate(ForwardPropagation&, size_t, bool) noexcept override;

    /** @brief Prints the per-output scaler methods and descriptives to stdout. */
    void print() const override;

    /**
     * @brief Reads the layer-specific JSON body (descriptives, scaler
     *        methods, source range).
     */
    void read_JSON_body(const Json*) override;
    /**
     * @brief Writes the layer-specific JSON body (descriptives, scaler
     *        methods, source range).
     */
    void write_JSON_body(JsonWriter&) const override;

private:

    /** @brief Per-output scaler method (one entry per output). */
    vector<ScalerMethod> scalers;

    /** @brief Underlying unscaling operator (holds descriptives and source range). */
    Unscale unscale_op;

    /** @brief Slot ordering in ForwardPropagation::views[layer]. */
    enum Forward {Input, Output};

    /** @brief Copies the current scaler vector into the operator's state buffer. */
    void flush_scalers_to_states();
};

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
