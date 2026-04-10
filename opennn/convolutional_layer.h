//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   C O N V O L U T I O N A L   L A Y E R   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#pragma once

#include "layer.h"
#include "math_utilities.h"

namespace opennn
{

class Convolutional final : public Layer
{

public:

    Convolutional(const Shape& = {3, 3, 1},                    // Input shape {height,width,channels}
                  const Shape& = {3, 3, 1, 1},                 // Kernel shape {kernel_height,kernel_width,channels,kernels_number}
                  const string& = "Linear",
                  const Shape& = { 1, 1 },                     // Stride shape {row_stride,column_stride}
                  const string& = "Valid",                          // Convolution type (Valid || Same)
                  bool = false,                              // Batch Normalization)
                  const string& = "convolutional_layer");

    bool get_batch_normalization() const { return batch_normalization; }

    ActivationFunction get_activation_function() const { return activation_function; }

    Shape get_output_shape() const override;

    pair<Index, Index> get_padding() const;

    Index get_output_height() const;
    Index get_output_width() const;

    string get_convolution_type() const { return convolution_type; }

    Index get_column_stride() const { return column_stride; }

    Index get_row_stride() const { return row_stride; }

    Index get_kernel_height() const { return kernel_height; }
    Index get_kernel_width() const { return kernel_width; }
    Index get_kernel_channels() const { return kernel_channels; }
    Index get_kernels_number() const { return kernels_number; }

    Index get_padding_height() const;
    Index get_padding_width() const;

    Index get_input_channels() const;
    Index get_input_height() const;
    Index get_input_width() const;

    vector<Shape> get_parameter_shapes() const override
    {
        if (batch_normalization)
            return {{kernels_number},
                    {kernels_number, kernel_height, kernel_width, kernel_channels},
                    {kernels_number}, {kernels_number} };

        return {{kernels_number},
                {kernels_number, kernel_height, kernel_width, kernel_channels}};
    }

    vector<Shape> get_forward_shapes(const Index) const override;

    vector<Shape> get_backward_shapes(Index batch_size) const override
    {
        const Index input_height = get_input_height();
        const Index input_width = get_input_width();
        const Index input_channels = get_input_channels();

        return {{batch_size, input_height, input_width, input_channels},
                {kernels_number, kernel_height, kernel_width, kernel_channels}};
    }

    bool use_convolutions() const
    {
        /*
        return activation_function == "ScaledExponentialLinear"
            || activation_function == "ClippedRelu"
            || activation_function == "Swish";
*/
        return false;
    }

    // Set

    void set(const Shape& = {0, 0, 0},
             const Shape& = {3, 3, 1, 1},
             const string& = "Linear",
             const Shape& = {1, 1},
             const string& = "Valid",
             bool = false,
             const string& = "convolutional_layer");

    void set_activation_function(const string&);

    void set_batch_normalization(bool);

    void set_convolution_type(const string&);

    void set_row_stride(const Index);

    void set_column_stride(const Index);

    void set_input_shape(const Shape&) override;

    void set_parameters_glorot() override;
    void set_parameters_random() override;

    void calculate_convolutions(const Tensor4&, TensorMap4) const;

    void forward_propagate(ForwardPropagation&, size_t index, bool) override;

    // Back propagation

    void back_propagate(ForwardPropagation&, BackPropagation&, size_t) const override;

    void from_XML(const XMLDocument&) override;
    void to_XML(XMLPrinter&) const override;

#ifdef CUDA

public:

    cudnnFilterDescriptor_t get_kernel_descriptor() const
    {
        return kernel_descriptor;
    }

    cudnnConvolutionDescriptor_t get_convolution_descriptor() const
    {
        return convolution_descriptor;
    }

protected:

    // Activations

    cudnnActivationDescriptor_t activation_descriptor = nullptr;

    cudnnFilterDescriptor_t kernel_descriptor = nullptr;

    cudnnConvolutionDescriptor_t convolution_descriptor = nullptr;

#endif

private:

    Index kernels_number = 0;
    Index kernel_height = 0;
    Index kernel_width = 0;
    Index kernel_channels = 0;

    enum Parameters {Biases, Weights, Gammas, Betas};
    enum Forward {Inputs, PaddedInputs};
    enum Backward {OutputGradients, InputGradients};

    // @todo Backward: Rotated weights

    Index row_stride = 1;
    Index column_stride = 1;

    string convolution_type = "Valid";

    ActivationFunction activation_function = ActivationFunction::Linear;

    // Batch normalization

    bool batch_normalization = false;

    // @todo here or in forward propagate?
    VectorR running_means;
    VectorR running_variances;

    type momentum = type(0.9);

#ifdef CUDA
public:
    void init_cuda_workspace(Index batch_size);

private:
    cudnnConvolutionFwdAlgo_t convolution_algorithm = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
    cudnnConvolutionBwdDataAlgo_t algo_data = CUDNN_CONVOLUTION_BWD_DATA_ALGO_0;
    cudnnConvolutionBwdFilterAlgo_t algo_filter = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0;

    void* cuda_workspace = nullptr;
    size_t cuda_workspace_size = 0;
    void* cuda_backward_filter_workspace = nullptr;
    size_t cuda_backward_filter_workspace_size = 0;
#endif
};

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
