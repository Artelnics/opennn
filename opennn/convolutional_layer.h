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

    bool get_batch_normalization() const;

    ActivationFunction get_activation_function() const;

    Shape get_output_shape() const override;

    pair<Index, Index> get_padding() const;

    Index get_output_height() const;
    Index get_output_width() const;

    string get_convolution_type() const;

    Index get_column_stride() const;

    Index get_row_stride() const;

    Index get_kernel_height() const;
    Index get_kernel_width() const;
    Index get_kernel_channels() const;
    Index get_kernels_number() const;

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
                 // Rotated Weights (Aux): {kernels, k_h, k_w, k_c}
                {kernels_number, kernel_height, kernel_width, kernel_channels}};
/*
        rotated_weights.resize(kernels_number, kernel_height, kernel_width, kernel_channels);

        input_gradients = {{nullptr, {batch_size, input_height, input_width, channels}}};

        // Batch Normalization

        if (batch_normalization)
        {
            gamma_gradients.shape = {kernels_number};
            beta_gradients.shape = {kernels_number};
        }

    const Index input_height = convolutional_layer->get_input_height();
    const Index input_width = convolutional_layer->get_input_width();
    const Index channels = convolutional_layer->get_input_channels();

    const Index output_height = convolutional_layer->get_output_height();
    const Index output_width = convolutional_layer->get_output_width();

    // Input Deltas

    input_gradients = {TensorView({batch_size, input_height, input_width, channels})};

    // Deltas

    cudnnSetTensor4dDescriptor(gradients_tensor_descriptor,
                               CUDNN_TENSOR_NHWC,
                               CUDNN_DATA_FLOAT,
                               batch_size,
                               kernels_number,
                               output_height,
                               output_width);

    // Biases derivatives

    bias_gradients.set_descriptor({ kernels_number });

    // Weight derivatives

    weight_gradients.set_descriptor({ kernels_number, kernel_height, kernel_width, channels });

    // Workspace

    int returned_algo_count;
    cudnnConvolutionBwdDataAlgoPerf_t data_perf;
    cudnnConvolutionBwdFilterAlgoPerf_t filter_perf;

    CHECK_CUDNN(cudnnFindConvolutionBackwardDataAlgorithm(
        get_cudnn_handle(),
        convolutional_layer->get_kernel_descriptor(),
        gradients_tensor_descriptor,
        convolutional_layer->get_convolution_descriptor(),
        input_gradients[0].get_descriptor(),
        1, &returned_algo_count, &data_perf));

    algo_data = data_perf.algo;

    CHECK_CUDNN(cudnnFindConvolutionBackwardFilterAlgorithm(
        get_cudnn_handle(),
        input_gradients[0].get_descriptor(),
        gradients_tensor_descriptor,
        convolutional_layer->get_convolution_descriptor(),
        convolutional_layer->get_kernel_descriptor(),
        1, &returned_algo_count, &filter_perf));

    algo_filter = filter_perf.algo;

    cudnnGetConvolutionBackwardDataWorkspaceSize(get_cudnn_handle(),
                                                 convolutional_layer->get_kernel_descriptor(),
                                                 gradients_tensor_descriptor,
                                                 convolutional_layer->get_convolution_descriptor(),
                                                 input_gradients[0].get_descriptor(),
                                                 algo_data,
                                                 &workspace_size);

    cudnnGetConvolutionBackwardFilterWorkspaceSize(get_cudnn_handle(),
                                                   input_gradients[0].get_descriptor(),
                                                   gradients_tensor_descriptor,
                                                   convolutional_layer->get_convolution_descriptor(),
                                                   convolutional_layer->get_kernel_descriptor(),
                                                   algo_filter,
                                                   &backward_filter_workspace_bytes);

    // Workspace memory

    CHECK_CUDA(cudaMalloc(&workspace, workspace_size));

    CHECK_CUDA(cudaMalloc(&backward_filter_workspace, backward_filter_workspace_bytes));

    // Batch Normalization

    if (convolutional_layer->get_batch_normalization())
    {
        beta_gradients.set_descriptor({ kernels_number });
        gamma_gradients.set_descriptor({ kernels_number });
    }

*/
        return {};
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
    enum Forward {Inputs, PaddedInputs, Outputs, ActivationDerivatives};
    enum Backward {OutputGradients, InputGradients};

    // @todo Forward TensorCuda inverse_variance;
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
    cudnnConvolutionFwdAlgo_t convolution_algorithm;
    cudnnConvolutionBwdDataAlgo_t algo_data;
    cudnnConvolutionBwdFilterAlgo_t algo_filter;
#endif
};

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
