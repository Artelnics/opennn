//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   C O N V O L U T I O N A L   L A Y E R   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#pragma once

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

    Index get_input_height() const;
    Index get_input_width() const;
    Index get_input_channels() const;

    Index get_kernel_height() const { return kernel_height; }
    Index get_kernel_width() const { return kernel_width; }
    Index get_kernel_channels() const { return kernel_channels; }
    Index get_kernels_number() const { return kernels_number; }

    Index get_row_stride() const { return row_stride; }
    Index get_column_stride() const { return column_stride; }

    pair<Index, Index> get_padding() const;
    Index get_padding_height() const;
    Index get_padding_width() const;

    bool get_use_padding() const { return use_padding; }

    Activation::Function get_activation_function() const { return activation.function; }
    Activation::Function get_output_activation() const override { return activation.function; }

    bool get_batch_normalization() const { return batch_normalization; }

    vector<Operator*> get_operators() override;
    vector<pair<Shape, Type>> get_forward_specs(Index batch_size) const override;
    vector<pair<Shape, Type>> get_backward_specs(Index batch_size) const override;

    void set(const Shape& = {0, 0, 0},
             const Shape& = {3, 3, 1, 1},
             const string& = "Identity",
             const Shape& = {1, 1},
             const string& = "Valid",
             bool = false,
             const string& = "convolutional_layer");

    void set_input_shape(const Shape&) override;
    void set_activation_dtype(Type new_activation_dtype) override
    {
        Layer::set_activation_dtype(new_activation_dtype);
        update_convolution_operator();
    }

    void set_row_stride(const Index);
    void set_column_stride(const Index);
    void set_convolution_type(const string&);
    void set_activation_function(const string&);
    void set_batch_normalization(bool);

    void set_parameters_glorot() override;
    void set_parameters_random() override;

    void forward_propagate(ForwardPropagation&, size_t, bool) noexcept override;
    void back_propagate(ForwardPropagation&, BackPropagation&, size_t) const noexcept override;

    void from_JSON(const JsonDocument&) override;
    void load_state_from_JSON(const JsonDocument&) override;
    void to_JSON(JsonWriter&) const override;

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

    bool batch_normalization = false;
    float momentum = 0.9f;

    Convolution convolution;
    Activation  activation;
    BatchNorm   batch_norm;

    enum Parameters {Bias, Weight, Gamma, Beta};
    enum States {RunningMean, RunningVariance};
    enum Forward {Input, PaddedInput, ConvolutionView, BatchNormMean, BatchNormInverseVariance, Output};
    enum Backward {OutputDelta, InputDelta};

    void update_convolution_operator();
};

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
