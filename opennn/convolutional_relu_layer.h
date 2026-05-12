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
class ConvolutionalRelu final : public Layer
{
public:

    ConvolutionalRelu(const Shape& = {3, 3, 1},
                      const Shape& = {3, 3, 1, 1},
                      const Shape& = {1, 1},
                      const string& = "Valid",
                      const string& = "convolutional_relu_layer");

    Shape get_input_shape() const override { return {input_height, input_width, input_channels}; }
    Shape get_output_shape() const override;

    Index get_output_height() const;
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
    Index get_padding_height() const;
    Index get_padding_width() const;

    bool get_use_padding() const { return use_padding; }

    ActivationOp::Function get_output_activation() const override { return ActivationOp::Function::ReLU; }

    vector<pair<Shape, Type>> get_forward_specs(Index batch_size) const override;

    void set(const Shape& = {0, 0, 0},
             const Shape& = {3, 3, 1, 1},
             const Shape& = {1, 1},
             const string& = "Valid",
             const string& = "convolutional_relu_layer");

    void set_input_shape(const Shape&) override;
    void on_compute_dtype_changed() override { update_convolution_operator(); }

    void set_row_stride(const Index);
    void set_column_stride(const Index);
    void set_convolution_type(const string&);

    void read_JSON_body(const Json*) override;
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
