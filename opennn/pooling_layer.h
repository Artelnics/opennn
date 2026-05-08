//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   P O O L I N G   L A Y E R   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

/**
 * @file pooling_layer.h
 * @brief Declares the Pooling layer for 2D max- or average-pooling, plus the
 *        PoolingMethod enumeration and its string conversion helpers.
 */

#pragma once

#include "layer.h"
#include "operators.h"
#include "forward_propagation.h"
#include "back_propagation.h"

namespace opennn
{

/**
 * @enum PoolingMethod
 * @brief Reduction strategy applied within each pooling window.
 */
enum class PoolingMethod
{
    MaxPooling,    ///< Take the maximum value of the window.
    AveragePooling ///< Take the arithmetic mean of the window.
};

/**
 * @brief Converts a PoolingMethod to its canonical string name.
 * @param method Pooling method value.
 * @return Reference to the canonical string ("MaxPooling" or "AveragePooling").
 */
inline const string& pooling_method_to_string(PoolingMethod method)
{
    static const string max_str = "MaxPooling";
    static const string avg_str = "AveragePooling";
    return method == PoolingMethod::MaxPooling ? max_str : avg_str;
}

/**
 * @brief Parses a PoolingMethod from its canonical string name.
 * @param name String to parse ("MaxPooling" or "AveragePooling").
 * @return Matching PoolingMethod; throws if the string is unrecognized.
 */
inline PoolingMethod string_to_pooling_method(const string& name)
{
    if (name == "MaxPooling")     return PoolingMethod::MaxPooling;
    if (name == "AveragePooling") return PoolingMethod::AveragePooling;
    throw runtime_error("Unknown pooling method: " + name);
}

/**
 * @class Pooling
 * @brief 2D pooling layer (max or average).
 *
 * Takes rank-3 inputs (height, width, channels) and produces outputs with
 * the same channel count and reduced spatial dimensions, following the
 * configured pool size, strides and padding. The layer has no trainable
 * parameters.
 *
 * For max-pooling, the index of the maximum within each window is cached
 * in the ForwardPropagation buffer to enable correct gradient routing in
 * the backward pass.
 */
class Pooling final : public Layer
{
public:

    /**
     * @brief Constructs a Pooling layer.
     * @param input_shape Per-sample input shape (height, width, channels).
     * @param pool_size Pool window size (height, width).
     * @param strides Row and column strides (one entry each).
     * @param padding Padding (height, width) added to each side.
     * @param pooling_method "MaxPooling" or "AveragePooling".
     * @param label Human-readable label assigned to this layer.
     */
    Pooling(const Shape& input_shape = {2, 2, 1},
            const Shape& pool_size = { 2, 2 },
            const Shape& strides = { 2, 2 },
            const Shape& padding = { 0, 0 },
            const string& pooling_method = "MaxPooling",
            const string& label = "pooling_layer");

    /** @brief Returns the per-sample input shape (height, width, channels). */
    Shape get_input_shape() const override { return {input_height, input_width, input_channels}; }

    /**
     * @brief Returns the per-sample output shape.
     * @return (output_height, output_width, input_channels).
     */
    Shape get_output_shape() const override;

    /** @brief Output spatial height after applying stride and padding. */
    Index get_output_height() const;
    /** @brief Output spatial width after applying stride and padding. */
    Index get_output_width() const;

    /** @brief Configured input height. */
    Index get_input_height() const { return input_height; }
    /** @brief Configured input width. */
    Index get_input_width() const { return input_width; }
    /** @brief Configured input (and output) channel count. */
    Index get_input_channels() const { return input_channels; }

    /** @brief Pool window height in pixels. */
    Index get_pool_height() const { return pool_height; }
    /** @brief Pool window width in pixels. */
    Index get_pool_width() const { return pool_width; }

    /** @brief Vertical stride in pixels. */
    Index get_row_stride() const { return row_stride; }
    /** @brief Horizontal stride in pixels. */
    Index get_column_stride() const { return column_stride; }

    /** @brief Padding rows added on each side. */
    Index get_padding_height() const { return padding_height; }
    /** @brief Padding columns added on each side. */
    Index get_padding_width() const { return padding_width; }

    /** @brief Configured pooling reduction method. */
    PoolingMethod get_pooling_method() const { return pooling_method; }

    /** @brief Returns the single Pool operator that implements this layer. */
    vector<Operator*> get_operators() override { return {&pool}; }

    /**
     * @brief Specifications of the forward intermediate buffers.
     * @param batch_size Batch size used for sizing.
     * @return Specs for Input, MaximalIndices and Output slots.
     */
    vector<pair<Shape, Type>> get_forward_specs(Index batch_size) const override;

    /**
     * @brief Re-initializes the layer; same arguments as the constructor.
     * @param input_shape Per-sample input shape.
     * @param pool_size Pool window size.
     * @param strides Row and column strides.
     * @param padding Padding sizes.
     * @param pooling_method Reduction method name.
     * @param label Human-readable label.
     */
    void set(const Shape& input_shape = { 0, 0, 0 },
             const Shape& pool_size = { 1, 1 },
             const Shape& strides = { 1, 1 },
             const Shape& padding = { 0, 0 },
             const string& pooling_method = "MaxPooling",
             const string& label = "pooling_layer");

    /** @brief Updates the input shape and re-derives the output shape. */
    void set_input_shape(const Shape&) override;
    /**
     * @brief Sets the pool window size.
     * @param new_pool_height New window height in pixels.
     * @param new_pool_width New window width in pixels.
     */
    void set_pool_size(Index new_pool_height, Index new_pool_width);
    /**
     * @brief Sets the vertical stride.
     * @param new_row_stride New row stride in pixels.
     */
    void set_row_stride(Index new_row_stride);
    /**
     * @brief Sets the horizontal stride.
     * @param new_column_stride New column stride in pixels.
     */
    void set_column_stride(Index new_column_stride);
    /**
     * @brief Sets the vertical padding.
     * @param new_padding_height New padding height in pixels (each side).
     */
    void set_padding_height(Index new_padding_height);
    /**
     * @brief Sets the horizontal padding.
     * @param new_padding_width New padding width in pixels (each side).
     */
    void set_padding_width(Index new_padding_width);
    /**
     * @brief Sets the pooling reduction method by name.
     *
     * Receives "MaxPooling" or "AveragePooling".
     */
    void set_pooling_method(const string&);

    /**
     * @brief Routes output-side gradients back to the corresponding input
     *        positions (max-index or average) in InputDelta.
     *
     * Receives the forward intermediates, the BackPropagation buffer and
     * this layer's index inside the network.
     */
    void back_propagate(ForwardPropagation&, BackPropagation&, size_t) const noexcept override;

    /**
     * @brief Reads the layer-specific JSON body (pool size, strides,
     *        padding, method) from the given JSON node.
     */
    void read_JSON_body(const Json*) override;
    /**
     * @brief Writes the layer-specific JSON body (pool size, strides,
     *        padding, method) to the given JSON writer.
     */
    void write_JSON_body(JsonWriter&) const override;

private:

    /** @brief Input height (rows) per sample. */
    Index input_height = 0;
    /** @brief Input width (columns) per sample. */
    Index input_width = 0;
    /** @brief Input (and output) channel count. */
    Index input_channels = 0;

    /** @brief Pool window height in pixels. */
    Index pool_height = 1;
    /** @brief Pool window width in pixels. */
    Index pool_width = 1;

    /** @brief Padding rows added on each side. */
    Index padding_height = 0;
    /** @brief Padding columns added on each side. */
    Index padding_width = 0;

    /** @brief Vertical stride in pixels. */
    Index row_stride = 1;
    /** @brief Horizontal stride in pixels. */
    Index column_stride = 1;

    /** @brief Selected reduction method (max or average). */
    PoolingMethod pooling_method = PoolingMethod::MaxPooling;

    /** @brief Underlying pooling operator. */
    Pool pool;

    /** @brief Slot ordering in ForwardPropagation::views[layer]. */
    enum Forward {Input, MaximalIndices, Output};
    /** @brief Slot ordering in BackPropagation::backward_views[layer]. */
    enum Backward {OutputDelta, InputDelta};

    /** @brief Refreshes the cached pool operator after a config change. */
    void update_pool_operator();
};

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
