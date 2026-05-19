//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   C O N V O L U T I O N A L   R E L U   L A Y E R   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#pragma once

#include "layer.h"
#include "operators.h"

namespace opennn
{

// Convolutional + ReLU fused into a single forward op on GPU
// (cudnnConvolutionBiasActivationForward with CUDNN_ACTIVATION_RELU). On CPU
// the activation runs as a separate step. No batch-norm, activation hard-wired
// to ReLU — keeps forward_propagate branch-free for CUDA Graph capture.

/// @brief Fused convolution + ReLU layer; runs as a single GPU op (cudnn) and is CUDA-Graph friendly.
class ConvolutionalRelu final : public Layer
{
public:

    /// @brief Constructs a fused convolution+ReLU layer.
    /// @param input_shape Shape of the input tensor (height, width, channels).
    /// @param kernel_shape Shape of the convolution kernel (height, width, channels, kernels).
    /// @param stride_shape Stride along height and width.
    /// @param convolution_type "Valid" or "Same" padding.
    /// @param name Layer name used for serialization.
    ConvolutionalRelu(const Shape& = {3, 3, 1},
                      const Shape& = {3, 3, 1, 1},
                      const Shape& = {1, 1},
                      const string& = "Valid",
                      const string& = "convolutional_relu_layer");

    /// @brief Returns the input tensor shape (height, width, channels).
    Shape get_input_shape() const override { return {input_height, input_width, input_channels}; }

    /// @brief Returns the output tensor shape after the fused conv+ReLU.
    Shape get_output_shape() const override;

    /// @brief Returns the output feature map height.
    Index get_output_height() const;

    /// @brief Returns the output feature map width.
    Index get_output_width() const;

    Index get_input_height() const { return input_height; }
    Index get_input_width() const { return input_width; }
    Index get_input_channels() const { return input_channels; }

    Index get_kernel_height() const { return kernel_height; }
    Index get_kernel_width() const { return kernel_width; }
    Index get_kernel_channels() const { return kernel_channels; }
    Index get_kernels_number() const { return kernels_number; }

    Index get_row_stride() const { return row_stride; }
    Index get_column_stride() const { return column_stride; }

    pair<Index, Index> get_padding() const { return {get_padding_height(), get_padding_width()}; }

    /// @brief Returns the padding applied along the height axis.
    Index get_padding_height() const;

    /// @brief Returns the padding applied along the width axis.
    Index get_padding_width() const;

    bool get_use_padding() const { return use_padding; }

    ActivationOp::Function get_output_activation() const override { return ActivationOp::Function::ReLU; }

    /// @brief Reconfigures the layer with new shapes and convolution settings.
    void set(const Shape& = {0, 0, 0},
             const Shape& = {3, 3, 1, 1},
             const Shape& = {1, 1},
             const string& = "Valid",
             const string& = "convolutional_relu_layer");

    /// @brief Updates the layer for a new input shape and reinitializes derived sizes.
    void set_input_shape(const Shape&) override;

    /// @brief Rebuilds the fused convolution operator when the compute dtype changes.
    void on_compute_dtype_changed() override { update_convolution_operator(); }

    void set_row_stride(const Index);
    void set_column_stride(const Index);

    /// @brief Sets convolution type ("Valid" or "Same") and updates padding.
    void set_convolution_type(const string&);

    /// @brief Reads the layer configuration from a JSON node.
    void read_JSON_body(const Json*) override;

    /// @brief Writes the layer configuration to a JSON writer.
    void write_JSON_body(JsonWriter&) const override;

private:

    Index input_height = 0;
    Index input_width = 0;
    Index input_channels = 0;

    Index kernels_number = 0;
    Index kernel_height = 0;
    Index kernel_width = 0;
    Index kernel_channels = 0;

    Index row_stride = 1;
    Index column_stride = 1;

    bool use_padding = false;

    ConvolutionReluOp convolution_relu;

    void update_convolution_operator();
};

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
