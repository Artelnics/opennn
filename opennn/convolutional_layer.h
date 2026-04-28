//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   C O N V O L U T I O N A L   L A Y E R   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#pragma once

#include "layer.h"

namespace opennn
{

enum class ConvolutionType
{
    Valid,
    Same
};

inline const EnumMap<ConvolutionType>& convolution_type_map()
{
    static const vector<pair<ConvolutionType, string>> entries = {
        {ConvolutionType::Valid, "Valid"},
        {ConvolutionType::Same,  "Same"}
    };
    static const EnumMap<ConvolutionType> map{entries};
    return map;
}

inline const string& convolution_type_to_string(ConvolutionType type)
{
    return convolution_type_map().to_string(type);
}

inline ConvolutionType string_to_convolution_type(const string& name)
{
    return convolution_type_map().from_string(name);
}

class Convolutional final : public Layer
{
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

    ConvolutionType convolution_type = ConvolutionType::Valid;
    bool use_padding = false;

    bool batch_normalization = false;
    type momentum = type(0.9);

    ActivationArguments activation_arguments;
    ConvolutionArguments convolution_arguments;

    cudnnFilterDescriptor_t kernel_descriptor = nullptr;
    cudnnConvolutionDescriptor_t convolution_descriptor = nullptr;

    cudnnConvolutionFwdAlgo_t convolution_algorithm = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
    cudnnConvolutionBwdDataAlgo_t algo_data = CUDNN_CONVOLUTION_BWD_DATA_ALGO_0;
    cudnnConvolutionBwdFilterAlgo_t algo_filter = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0;

    void* cuda_workspace = nullptr;
    size_t cuda_workspace_size = 0;
    void* cuda_backward_filter_workspace = nullptr;
    size_t cuda_backward_filter_workspace_size = 0;

    enum Parameters {Bias, Weight, Gamma, Beta};

    vector<Shape> get_parameter_shapes() const override
    {
        return {{kernels_number},                                               // Bias
                {kernels_number, kernel_height, kernel_width, kernel_channels}, // Weight
                {batch_normalization ? kernels_number : 0},                     // Gamma
                {batch_normalization ? kernels_number : 0}};                    // Beta
    }

    // Convolutional weight gradient (cudnnConvolutionBackwardFilter) shares its
    // dtype with the filter via the same cudnnFilterDescriptor. Our FP32
    // gradient buffer in BackPropagation can't accept BF16 writes, so we keep
    // the filter slot in FP32 even when activations are BF16. Activations
    // still go BF16 via the input/output tensor descriptors. Net: convolution
    // input × FP32 filter → BF16 output is a mixed-dtype path that cuDNN
    // supports out of the box.
    vector<cudnnDataType_t> get_parameter_dtypes() const override
    {
        return vector<cudnnDataType_t>(get_parameter_shapes().size(), CUDNN_DATA_FLOAT);
    }

    enum States {RunningMean, RunningVariance};

    vector<Shape> get_state_shapes() const override
    {
        if (!batch_normalization) return {};
        return {{kernels_number},   // RunningMean
                {kernels_number}};  // RunningVariance
    }

    enum Forward {Input, PaddedInput, Convolution, BatchNormMean, BatchNormInverseVariance, Output};

    vector<Shape> get_forward_shapes(const Index batch_size) const override
    {
        const Shape output_shape = {batch_size, get_output_height(), get_output_width(), kernels_number};
        const Shape padded_shape = {batch_size,
                                    input_height + 2 * get_padding_height(),
                                    input_width + 2 * get_padding_width(),
                                    input_channels};

        if (batch_normalization)
            return {padded_shape,             // PaddedInputs
                    output_shape,             // Convolution
                    Shape{kernels_number},    // BatchNormMean
                    Shape{kernels_number},    // BatchNormInverseVariance
                    output_shape};            // Output

        return {padded_shape,                 // PaddedInputs
                Shape{},                      // Convolution (unused)
                Shape{},                      // BatchNormMean (unused)
                Shape{},                      // BatchNormInverseVariance (unused)
                output_shape};                // Output
    }

    vector<cudnnDataType_t> get_forward_dtypes(Index) const override
    {
        return {CUDNN_ACTIVATION_DTYPE,  // PaddedInputs
                CUDNN_ACTIVATION_DTYPE,  // Convolution
                CUDNN_DATA_FLOAT,        // BatchNormMean
                CUDNN_DATA_FLOAT,        // BatchNormInverseVariance
                CUDNN_ACTIVATION_DTYPE}; // Output
    }

    enum Backward {OutputDelta, InputDelta};

    vector<Shape> get_backward_shapes(Index batch_size) const override
    {
        return {{batch_size, input_height, input_width, input_channels},
                {kernels_number, kernel_height, kernel_width, kernel_channels}};
    }

public:

    Convolutional(const Shape& = {3, 3, 1},
                  const Shape& = {3, 3, 1, 1},
                  const string& = "Linear",
                  const Shape& = {1, 1},
                  const string& = "Valid",
                  bool = false,
                  const string& = "convolutional_layer");

    ~Convolutional() override
    {
#ifdef OPENNN_WITH_CUDA
        destroy_cuda();
#endif
    }

    // Getters

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

    ConvolutionType get_convolution_type() const { return convolution_type; }

    ActivationFunction get_activation_function() const { return activation_arguments.activation_function; }
    ActivationFunction get_output_activation() const override { return activation_arguments.activation_function; }

    bool get_batch_normalization() const { return batch_normalization; }

    cudnnFilterDescriptor_t get_kernel_descriptor() const { return kernel_descriptor; }
    cudnnConvolutionDescriptor_t get_convolution_descriptor() const { return convolution_descriptor; }

    // Setters

    void set(const Shape& = {0, 0, 0},
             const Shape& = {3, 3, 1, 1},
             const string& = "Linear",
             const Shape& = {1, 1},
             const string& = "Valid",
             bool = false,
             const string& = "convolutional_layer");

    void set_input_shape(const Shape&) override;

    void load_state_from_XML(const XmlDocument&) override;

    void set_row_stride(const Index);
    void set_column_stride(const Index);

    void set_convolution_type(const string&);

    void set_activation_function(const string&);

    void set_batch_normalization(bool);

    // Parameter initialization

    void set_parameters_glorot() override;
    void set_parameters_random() override;

private:
    void init_conv_norm_defaults();
public:

    // Device setup

    void init_cuda(Index);
    void destroy_cuda();

    // Forward / back propagation

    void forward_propagate(ForwardPropagation&, size_t, bool) noexcept override;

    void back_propagate(ForwardPropagation&, BackPropagation&, size_t) const noexcept override;

    // Serialization

    void from_XML(const XmlDocument&) override;
    void to_XML(XmlPrinter&) const override;
};

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
