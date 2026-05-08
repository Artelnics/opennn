//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   C O N V O L U T I O N A L   L A Y E R   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

/**
 * @file convolutional_layer.h
 * @brief Declares the Convolutional layer: 2D convolution with optional
 *        batch normalization and a configurable activation function.
 */

#pragma once

#include "layer.h"
#include "operators.h"

namespace opennn
{

/**
 * @class Convolutional
 * @brief 2D convolutional layer: y = activation(BN(conv(x, kernels) + bias)).
 *
 * Inputs are rank-3 tensors (height, width, channels). Outputs preserve the
 * spatial layout while replacing the channel dimension with the number of
 * kernels. The layer wraps three reusable Operators: Convolution (the
 * spatial linear part with kernels and bias), an optional BatchNorm and a
 * pointwise Activation.
 *
 * Padding is selected by name ("Valid" for no padding, "Same" for output
 * size equal to input size); strides default to 1 in each spatial axis.
 */
class Convolutional final : public Layer
{
public:

    /**
     * @brief Constructs a Convolutional layer.
     * @param input_shape Per-sample input shape (height, width, channels).
     * @param kernel_shape Kernel shape (height, width, channels, count).
     * @param activation_name Name of the activation function ("Identity"
     *                        by default; see Activation::Function).
     * @param strides Row and column strides (one entry each).
     * @param convolution_type Padding mode ("Valid" or "Same").
     * @param batch_normalization Enables BatchNorm before the activation.
     * @param label Human-readable label assigned to this layer.
     */
    Convolutional(const Shape& input_shape = {3, 3, 1},
                  const Shape& kernel_shape = {3, 3, 1, 1},
                  const string& activation_name = "Identity",
                  const Shape& strides = {1, 1},
                  const string& convolution_type = "Valid",
                  bool batch_normalization = false,
                  const string& label = "convolutional_layer");

    /** @brief Returns the per-sample input shape (height, width, channels). */
    Shape get_input_shape() const override { return {input_height, input_width, input_channels}; }

    /**
     * @brief Returns the per-sample output shape.
     * @return (output_height, output_width, kernels_number).
     */
    Shape get_output_shape() const override;

    /** @brief Output spatial height after applying stride and padding. */
    Index get_output_height() const;
    /** @brief Output spatial width after applying stride and padding. */
    Index get_output_width() const;

    /** @brief Configured input height. */
    Index get_input_height() const;
    /** @brief Configured input width. */
    Index get_input_width() const;
    /** @brief Configured input channel count. */
    Index get_input_channels() const;

    /** @brief Kernel height in pixels. */
    Index get_kernel_height() const { return kernel_height; }
    /** @brief Kernel width in pixels. */
    Index get_kernel_width() const { return kernel_width; }
    /** @brief Kernel channel count (must equal input channels). */
    Index get_kernel_channels() const { return kernel_channels; }
    /** @brief Number of kernels (output channel count). */
    Index get_kernels_number() const { return kernels_number; }

    /** @brief Vertical stride in pixels. */
    Index get_row_stride() const { return row_stride; }
    /** @brief Horizontal stride in pixels. */
    Index get_column_stride() const { return column_stride; }

    /**
     * @brief Returns padding sizes for the configured convolution type.
     * @return Pair (padding_height, padding_width) in pixels.
     */
    pair<Index, Index> get_padding() const;
    /** @brief Padding rows added on each side. */
    Index get_padding_height() const;
    /** @brief Padding columns added on each side. */
    Index get_padding_width() const;

    /** @brief True if padding is applied (i.e. convolution type is not "Valid"). */
    bool get_use_padding() const { return use_padding; }

    /** @brief Activation function applied at the layer's output. */
    Activation::Function get_activation_function() const { return activation.function; }
    /** @brief Activation function fused at the end of this layer. */
    Activation::Function get_output_activation() const override { return activation.function; }

    /** @brief Whether batch normalization is enabled in the pipeline. */
    bool get_batch_normalization() const { return batch_norm.active(); }

    /**
     * @brief Returns the active operators in pipeline order.
     * @return Pointers to Convolution, then BatchNorm if enabled, then Activation.
     */
    vector<Operator*> get_operators() override;

    /**
     * @brief Specifications of the forward intermediate buffers.
     * @param batch_size Batch size used for sizing.
     * @return One spec per slot in the Forward enum.
     */
    vector<pair<Shape, Type>> get_forward_specs(Index batch_size) const override;

    /**
     * @brief Re-initializes the layer; same arguments as the constructor.
     * @param input_shape Per-sample input shape.
     * @param kernel_shape Kernel shape (height, width, channels, count).
     * @param activation_name Activation function name.
     * @param strides Row and column strides.
     * @param convolution_type Padding mode ("Valid" or "Same").
     * @param batch_normalization Enables BatchNorm.
     * @param label Human-readable label.
     */
    void set(const Shape& input_shape = {0, 0, 0},
             const Shape& kernel_shape = {3, 3, 1, 1},
             const string& activation_name = "Identity",
             const Shape& strides = {1, 1},
             const string& convolution_type = "Valid",
             bool batch_normalization = false,
             const string& label = "convolutional_layer");

    /** @brief Updates the input shape and re-shapes kernel tensors accordingly. */
    void set_input_shape(const Shape&) override;

    /** @brief Recreates the convolution operator descriptor when the dtype changes. */
    void on_compute_dtype_changed() override { update_convolution_operator(); }

    /**
     * @brief Sets the vertical (row) stride.
     * @param new_row_stride New row stride in pixels.
     */
    void set_row_stride(const Index new_row_stride);
    /**
     * @brief Sets the horizontal (column) stride.
     * @param new_column_stride New column stride in pixels.
     */
    void set_column_stride(const Index new_column_stride);
    /**
     * @brief Sets the padding mode by name.
     *
     * Receives the convolution type, "Valid" (no padding) or "Same"
     * (output spatial dimensions equal to input spatial dimensions).
     */
    void set_convolution_type(const string&);
    /**
     * @brief Sets the activation function by name.
     *
     * Receives the activation name; see Activation::Function for the
     * supported set.
     */
    void set_activation_function(const string&);
    /**
     * @brief Enables or disables BatchNorm between convolution and activation.
     */
    void set_batch_normalization(bool);

    /**
     * @brief Backward pass through the Activation, BatchNorm and Convolution
     *        operators in reverse order.
     *
     * Receives the forward intermediates, the BackPropagation buffer in
     * which to accumulate gradients, and this layer's index inside the
     * network.
     */
    void back_propagate(ForwardPropagation&, BackPropagation&, size_t) const noexcept override;

    /**
     * @brief Reads the layer-specific JSON body (kernel shape, strides,
     *        padding, activation, BN flag) from the given JSON node.
     */
    void read_JSON_body(const Json*) override;
    /**
     * @brief Writes the layer-specific JSON body (kernel shape, strides,
     *        padding, activation, BN flag) to the given JSON writer.
     */
    void write_JSON_body(JsonWriter&) const override;

private:

    /** @brief Input height (rows) per sample. */
    Index input_height = 0;
    /** @brief Input width (columns) per sample. */
    Index input_width = 0;
    /** @brief Input channel count per sample. */
    Index input_channels = 0;

    /** @brief Number of convolution kernels (= output channels). */
    Index kernels_number = 0;
    /** @brief Kernel height in pixels. */
    Index kernel_height = 0;
    /** @brief Kernel width in pixels. */
    Index kernel_width = 0;
    /** @brief Kernel channel count (= input channels). */
    Index kernel_channels = 0;

    /** @brief Vertical stride in pixels. */
    Index row_stride = 1;
    /** @brief Horizontal stride in pixels. */
    Index column_stride = 1;

    /** @brief True when padding is applied (i.e. type is not "Valid"). */
    bool use_padding = false;

    /** @brief Spatial linear projection (kernels + bias). */
    Convolution convolution;
    /** @brief Pointwise activation function. */
    Activation  activation;
    /** @brief Optional batch normalization between convolution and activation. */
    BatchNorm   batch_norm;

    /** @brief Slot ordering in ForwardPropagation::views[layer]. */
    enum Forward {Input, ConvolutionView, BatchNormMean, BatchNormInverseVariance, Output};
    /** @brief Slot ordering in BackPropagation::backward_views[layer]. */
    enum Backward {OutputDelta, InputDelta};

    /** @brief Refreshes the cached convolution operator descriptor. */
    void update_convolution_operator();
};

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
