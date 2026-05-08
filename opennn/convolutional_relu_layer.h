//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   C O N V O L U T I O N A L   R E L U   L A Y E R   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

/**
 * @file convolutional_relu_layer.h
 * @brief Declares ConvolutionalRelu, a fused 2D convolution + ReLU layer
 *        optimized for cudnnConvolutionBiasActivationForward and CUDA
 *        Graph capture.
 */

#pragma once

#include "layer.h"
#include "operators.h"

namespace opennn
{

/**
 * @class ConvolutionalRelu
 * @brief 2D convolution + ReLU fused into a single forward op on GPU.
 *
 * Calls cudnnConvolutionBiasActivationForward with CUDNN_ACTIVATION_RELU.
 * On CPU the activation runs as a separate step. No batch normalization,
 * activation hard-wired to ReLU — keeps forward_propagate() branch-free
 * for CUDA Graph capture.
 *
 * Use this layer instead of Convolutional when ReLU is the desired
 * activation and the runtime cost of branching matters.
 */
class ConvolutionalRelu final : public Layer
{
public:

    /**
     * @brief Constructs a ConvolutionalRelu layer.
     * @param input_shape Per-sample input shape (height, width, channels).
     * @param kernel_shape Kernel shape (height, width, channels, count).
     * @param strides Row and column strides (one entry each).
     * @param convolution_type Padding mode ("Valid" or "Same").
     * @param label Human-readable label assigned to this layer.
     */
    ConvolutionalRelu(const Shape& input_shape = {3, 3, 1},
                      const Shape& kernel_shape = {3, 3, 1, 1},
                      const Shape& strides = {1, 1},
                      const string& convolution_type = "Valid",
                      const string& label = "convolutional_relu_layer");

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
    Index get_input_height() const { return input_height; }
    /** @brief Configured input width. */
    Index get_input_width() const { return input_width; }
    /** @brief Configured input channel count. */
    Index get_input_channels() const { return input_channels; }

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
    pair<Index, Index> get_padding() const { return {get_padding_height(), get_padding_width()}; }
    /** @brief Padding rows added on each side. */
    Index get_padding_height() const;
    /** @brief Padding columns added on each side. */
    Index get_padding_width() const;

    /** @brief True if padding is applied (i.e. convolution type is not "Valid"). */
    bool get_use_padding() const { return use_padding; }

    /** @brief Activation function fused at the end of this layer (always ReLU). */
    Activation::Function get_output_activation() const override { return Activation::Function::ReLU; }

    /** @brief Returns the single operator (Convolution with fused ReLU on GPU). */
    vector<Operator*> get_operators() override { return {&convolution}; }

    /**
     * @brief Specifications of the forward intermediate buffers.
     * @param batch_size Batch size used for sizing.
     * @return Specs for the Input and Output slots in the Forward enum.
     */
    vector<pair<Shape, Type>> get_forward_specs(Index batch_size) const override;

    /**
     * @brief Re-initializes the layer; same arguments as the constructor.
     * @param input_shape Per-sample input shape.
     * @param kernel_shape Kernel shape (height, width, channels, count).
     * @param strides Row and column strides.
     * @param convolution_type Padding mode ("Valid" or "Same").
     * @param label Human-readable label.
     */
    void set(const Shape& input_shape = {0, 0, 0},
             const Shape& kernel_shape = {3, 3, 1, 1},
             const Shape& strides = {1, 1},
             const string& convolution_type = "Valid",
             const string& label = "convolutional_relu_layer");

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
     * @brief Forward pass: convolution + ReLU fused on GPU, sequential on CPU.
     *
     * Receives the ForwardPropagation buffer slice, this layer's index and
     * the training flag.
     */
    void forward_propagate(ForwardPropagation&, size_t, bool) noexcept override;
    /**
     * @brief Backward pass through the fused ReLU and Convolution operators.
     *
     * Receives the forward intermediates, the BackPropagation buffer in
     * which to accumulate gradients, and this layer's index inside the
     * network.
     */
    void back_propagate(ForwardPropagation&, BackPropagation&, size_t) const noexcept override;

    /**
     * @brief Reads the layer-specific JSON body (kernel shape, strides,
     *        padding) from the given JSON node.
     */
    void read_JSON_body(const Json*) override;
    /**
     * @brief Writes the layer-specific JSON body (kernel shape, strides,
     *        padding) to the given JSON writer.
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

    /** @brief Spatial linear projection with fused ReLU epilogue (GPU). */
    Convolution convolution;
    /** @brief Activation handle kept for backward-pass derivative dispatch. */
    Activation  activation;

    /** @brief Slot ordering in ForwardPropagation::views[layer]. */
    enum Forward {Input, Output};
    /** @brief Slot ordering in BackPropagation::backward_views[layer]. */
    enum Backward {OutputDelta, InputDelta};

    /** @brief Refreshes the cached convolution operator descriptor. */
    void update_convolution_operator();
};

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
