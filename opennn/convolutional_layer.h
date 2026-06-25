//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   C O N V O L U T I O N A L   L A Y E R   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#pragma once

#include <cstdio>
#include "layer.h"
#include "operators.h"

namespace opennn
{

class Convolutional final : public Layer
{
public:

    Convolutional(const Shape& = {3, 3, 1},
                  const Shape& = {3, 3, 1, 1},
                  const string& = "Identity",
                  const Shape& = {1, 1},
                  const string& = "Valid",
                  bool = false,
                  const string& = "convolutional_layer");

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

    Index get_padding_height() const;
    Index get_padding_width() const;

    bool get_use_padding() const { return use_padding; }

    ActivationFunction get_activation_function() const { return activation_operator.activation_function; }
    ActivationFunction get_output_activation() const override { return activation_operator.activation_function; }

    bool get_batch_normalization() const { return batch_norm.active(); }

    bool get_residual() const { return residual; }
    void set_residual(bool);

    vector<TensorSpec> get_forward_specs(Index) const override;
    vector<TensorSpec> get_backward_specs(Index) const override;

    void set(const Shape& = {0, 0, 0},
             const Shape& = {3, 3, 1, 1},
             const string& = "Identity",
             const Shape& = {1, 1},
             const string& = "Valid",
             bool = false,
             const string& = "convolutional_layer");

    void set_input_shape(const Shape&) override;
    void set_output_shape(const Shape&) override {}

    void on_compute_dtype_changed() override { update_convolution_operator(); }

    void set_row_stride(const Index);
    void set_column_stride(const Index);
    void set_convolution_type(const string&);
    void set_activation_function(const string&);
    void set_batch_normalization(bool);

    // Load weights (and BN) for this layer from an open Darknet
    // binary weights file.  The file position must be placed immediately at
    // the start of this layer's data (i.e. after the Darknet file header and
    // all preceding layers).  With BN: reads beta, gamma, running_mean,
    // running_var then conv weights.  Without BN: reads bias then conv weights.
    // Conv weights are transposed from Darknet [O,I,H,W] → OpenNN [O,H,W,I].
    void load_darknet_weights(FILE*);

    void read_JSON_body(const Json*) override;
    void write_JSON_body(JsonWriter&) const override;
    void from_JSON(const JsonDocument&) override;

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
    bool residual = false;

    ConvolutionOperator convolution;
    ActivationOperator  activation;
    BatchNormOperator   batch_norm;

    enum Forward {Input, ConvolutionView, BatchNormMean, BatchNormInverseVariance, Output};

    void update_convolution_operator();
};

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
