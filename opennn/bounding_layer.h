//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   B O U N D I N G   L A Y E R   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

/**
 * @file bounding_layer.h
 * @brief Declares the Bounding layer: clamps each output to a configurable
 *        per-feature [lower, upper] interval.
 */

#pragma once

#include "layer.h"
#include "operators.h"
#include "forward_propagation.h"
#include "back_propagation.h"

namespace opennn
{

/**
 * @class Bounding
 * @brief Per-feature output-clamping layer.
 *
 * Holds a lower bound and an upper bound for every feature and clamps the
 * incoming values to that interval. Used as the last layer of regression
 * networks when outputs are known to lie inside a physical range.
 *
 * The layer has no trainable parameters.
 */
class Bounding final : public Layer
{
public:

    /** @brief Alias for the underlying Bound::Method enumeration. */
    using BoundingMethod = Bound::Method;

    /**
     * @brief Constructs a Bounding layer.
     * @param output_shape Per-sample output (and input) shape.
     * @param label Human-readable label assigned to this layer.
     */
    Bounding(const Shape& output_shape = {0},
             const string& label = "bounding_layer");

    /** @brief Returns the per-sample input shape (same as output). */
    Shape get_input_shape() const override { return output_shape; }
    /** @brief Returns the per-sample output shape. */
    Shape get_output_shape() const override;

    /** @brief Returns the configured bounding method. */
    const BoundingMethod& get_bounding_method() const { return bound.method; }

    /** @brief Returns the per-feature lower bounds. */
    VectorR get_lower_bounds() const;
    /** @brief Returns the per-feature upper bounds. */
    VectorR get_upper_bounds() const;

    /** @brief Returns the single Bound operator that implements this layer. */
    vector<Operator*> get_operators() override { return {&bound}; }

    /**
     * @brief Re-initializes the layer.
     * @param output_shape Per-sample output shape.
     * @param label Human-readable label.
     */
    void set(const Shape& output_shape = {0},
             const string& label = "bounding_layer");

    /** @brief Updates the input shape (kept equal to the output shape). */
    void set_input_shape(const Shape&) override;
    /** @brief Updates the output shape and resizes the bound vectors. */
    void set_output_shape(const Shape&) override;

    /**
     * @brief Sets the bounding method directly.
     *
     * Receives a Bound::Method value (e.g. NoBounding, Clamp).
     */
    void set_bounding_method(const BoundingMethod&);
    /**
     * @brief Sets the bounding method by name.
     *
     * Receives the canonical method name; throws if it is unrecognized.
     */
    void set_bounding_method(const string&);

    /**
     * @brief Sets every per-feature lower bound at once.
     *
     * Receives a vector with as many entries as the layer has features.
     */
    void set_lower_bounds(const VectorR&);
    /**
     * @brief Sets the lower bound of a single feature.
     * @param feature_index Index of the feature whose lower bound is updated.
     * @param value New lower bound for that feature.
     */
    void set_lower_bound(Index feature_index, float value);

    /**
     * @brief Sets every per-feature upper bound at once.
     *
     * Receives a vector with as many entries as the layer has features.
     */
    void set_upper_bounds(const VectorR&);
    /**
     * @brief Sets the upper bound of a single feature.
     * @param feature_index Index of the feature whose upper bound is updated.
     * @param value New upper bound for that feature.
     */
    void set_upper_bound(Index feature_index, float value);

    /**
     * @brief Reads the layer-specific JSON body (method, lower and upper bounds).
     */
    void read_JSON_body(const Json*) override;
    /**
     * @brief Writes the layer-specific JSON body (method, lower and upper bounds).
     */
    void write_JSON_body(JsonWriter&) const override;

private:

    /** @brief Per-sample output (and input) shape. */
    Shape output_shape;

    /** @brief Underlying clamping operator. */
    Bound bound;

    /** @brief Slot ordering in ForwardPropagation::views[layer]. */
    enum Forward {Input, Output};

    /**
     * @brief Returns the singleton string<->enum mapping for BoundingMethod.
     * @return Reference to a process-wide EnumMap initialized on first call.
     */
    static const EnumMap<BoundingMethod>& bounding_method_map();
};

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
