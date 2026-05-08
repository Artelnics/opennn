//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   S C A L I N G   L A Y E R   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

/**
 * @file scaling_layer.h
 * @brief Declares the Scaling layer: applies per-feature input
 *        normalization (min-max, mean-stddev, ...) before training.
 */

#pragma once

#include "statistics.h"
#include "layer.h"
#include "operators.h"
#include "variable.h"

namespace opennn
{

/**
 * @class Scaling
 * @brief Per-feature input normalization layer.
 *
 * Holds a vector of ScalerMethod values (one per feature) and the
 * corresponding descriptive statistics (minimum, maximum, mean, standard
 * deviation). The forward pass applies the configured scaler to each
 * feature so that downstream layers see normalized inputs.
 *
 * The layer has no trainable parameters; the scaler statistics are stored
 * as state and are typically derived from the training dataset.
 */
class Scaling final : public Layer
{
public:

    /**
     * @brief Constructs a Scaling layer.
     * @param input_shape Per-sample input shape.
     */
    Scaling(const Shape& input_shape = {});

    /** @brief Returns the per-sample input shape. */
    Shape get_input_shape() const override { return input_shape; }
    /** @brief Returns the per-sample output shape (same as input). */
    Shape get_output_shape() const override { return input_shape; }

    /** @brief Returns the per-feature minimum statistics. */
    VectorR get_minimums() const;
    /** @brief Returns the per-feature maximum statistics. */
    VectorR get_maximums() const;
    /** @brief Returns the per-feature mean statistics. */
    VectorR get_means() const;
    /** @brief Returns the per-feature standard-deviation statistics. */
    VectorR get_standard_deviations() const;

    /** @brief Read-only access to the per-feature scaler method list. */
    const vector<ScalerMethod>& get_scalers() const { return scalers; }

    /** @brief Lower bound of the target range used by min-max scalers. */
    float get_min_range() const { return scale_op.min_range; }
    /** @brief Upper bound of the target range used by min-max scalers. */
    float get_max_range() const { return scale_op.max_range; }

    /** @brief Returns the single Scale operator that implements this layer. */
    vector<Operator*> get_operators() override { return {&scale_op}; }

    /**
     * @brief Re-initializes the layer.
     * @param input_shape Per-sample input shape.
     */
    void set(const Shape& input_shape = {});

    /** @brief Updates the input shape and resizes the scaler vector. */
    void set_input_shape(const Shape&) override;
    /** @brief Updates the output shape (kept equal to the input shape). */
    void set_output_shape(const Shape&) override;

    /**
     * @brief Sets the per-feature descriptive statistics used by the scalers.
     *
     * Receives a vector of Descriptives (minimum, maximum, mean, standard
     * deviation) with one entry per input feature.
     */
    void set_descriptives(const vector<Descriptives>&);

    /**
     * @brief Sets the target range used by min-max scalers.
     * @param min Lower bound of the target range.
     * @param max Upper bound of the target range.
     */
    void set_min_max_range(float min, float max) { scale_op.min_range = min; scale_op.max_range = max; }

    /**
     * @brief Sets the scaler method per feature from a vector of names.
     *
     * Receives one method name per feature; see ScalerMethod for the
     * supported set.
     */
    void set_scalers(const vector<string>&);
    /**
     * @brief Sets the same scaler method for every feature.
     *
     * Receives a single method name applied to all features.
     */
    void set_scalers(const string&);

    /**
     * @brief Forward pass: applies the configured scaler to each feature.
     *
     * Receives the ForwardPropagation buffer slice, this layer's index and
     * the training flag.
     */
    void forward_propagate(ForwardPropagation&, size_t, bool) noexcept override;

    /**
     * @brief Generates a symbolic expression that performs no scaling.
     * @param inputs_names Variable names used in the expression for each input.
     * @param outputs_names Variable names used for each output.
     * @return Multi-line expression string suitable for code export.
     */
    string write_no_scaling_expression(const vector<string>& inputs_names,
                                       const vector<string>& outputs_names) const;
    /**
     * @brief Generates a symbolic expression for min-max scaling.
     * @param inputs_names Variable names used in the expression for each input.
     * @param outputs_names Variable names used for each output.
     * @return Multi-line expression string suitable for code export.
     */
    string write_minimum_maximum_expression(const vector<string>& inputs_names,
                                            const vector<string>& outputs_names) const;
    /**
     * @brief Generates a symbolic expression for mean / standard-deviation scaling.
     * @param inputs_names Variable names used in the expression for each input.
     * @param outputs_names Variable names used for each output.
     * @return Multi-line expression string suitable for code export.
     */
    string write_mean_standard_deviation_expression(const vector<string>& inputs_names,
                                                    const vector<string>& outputs_names) const;
    /**
     * @brief Generates a symbolic expression for standard-deviation scaling.
     * @param inputs_names Variable names used in the expression for each input.
     * @param outputs_names Variable names used for each output.
     * @return Multi-line expression string suitable for code export.
     */
    string write_standard_deviation_expression(const vector<string>& inputs_names,
                                                const vector<string>& outputs_names) const;

    /**
     * @brief Reads the layer-specific JSON body (input shape, descriptives,
     *        scaler methods, target range).
     */
    void read_JSON_body(const Json*) override;
    /**
     * @brief Writes the layer-specific JSON body (input shape, descriptives,
     *        scaler methods, target range).
     */
    void write_JSON_body(JsonWriter&) const override;

private:

    /** @brief Per-sample input shape (same as output). */
    Shape input_shape;

    /** @brief Per-feature scaler method (one entry per input feature). */
    vector<ScalerMethod> scalers;

    /** @brief Underlying scaling operator (holds descriptives and target range). */
    Scale scale_op;

    /** @brief Slot ordering in ForwardPropagation::views[layer]. */
    enum Forward {Input, Output};

    /** @brief Copies the current scaler vector into the operator's state buffer. */
    void flush_scalers_to_states();
};

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
