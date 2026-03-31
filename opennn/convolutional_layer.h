//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   C O N V O L U T I O N A L   L A Y E R   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#pragma once

#include "layer.h"
#include "tensor_utilities.h"

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

    const string& get_activation_function() const;

    Shape get_input_shape() const override;
    Shape get_output_shape() const override;

    pair<Index, Index> get_padding() const;

    array<pair<Index, Index>, 4> get_paddings() const;

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
        vector<Shape> shapes = {
            //biases,
            //weights}
        };
/*
        if (batch_normalization)
            shapes.insert(shapes.end(), {&gammas, &betas});
*/
        return shapes;
    }

    vector<Shape> get_forward_shapes() const override;

    vector<Shape> get_backward_shapes() const override
    {
        const Index input_height = get_input_height();
        const Index input_width = get_input_width();
        const Index channels = get_input_channels();

        const Index kernel_height = get_kernel_height();
        const Index kernel_width = get_kernel_width();
        const Index kernel_channels = get_kernel_channels();
        const Index kernels_number = get_kernels_number();
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

    const Index kernels_number = convolutional_layer->get_kernels_number();
    const Index kernel_height = convolutional_layer->get_kernel_height();
    const Index kernel_width = convolutional_layer->get_kernel_width();

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
        return activation_function == "ScaledExponentialLinear"
            || activation_function == "ClippedRelu"
            || activation_function == "Swish";
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

    void pad_inputs(const Tensor4&, TensorMap4&) const;

    void calculate_convolutions(const Tensor4&, TensorMap4) const;

    void forward_propagate(ForwardPropagation&, size_t index, bool) override;

    // Back propagation

    void back_propagate(ForwardPropagation&, BackPropagation&, size_t) const override;

    void from_XML(const XMLDocument&) override;
    void to_XML(XMLPrinter&) const override;

    void print() const override;

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

    TensorView biases_device;
    TensorView weights_device;

    // Batch Normalization

    TensorView gammas_device;
    TensorView betas_device;

    TensorCuda running_means_device;
    TensorCuda running_variances_device;

    // Activations

    cudnnActivationDescriptor_t activation_descriptor = nullptr;

    cudnnFilterDescriptor_t kernel_descriptor = nullptr;

    cudnnConvolutionDescriptor_t convolution_descriptor = nullptr;

#endif

private:

    enum Parameters {Biases, Weights, Gammas, Betas};
    enum Forward {Inputs, PaddedInputs, Outputs, ActivationDerivatives};
    enum Backward {OutputGradients, InputGradients};

    // Forward TensorCuda inverse_variance;

    // Backward: Rotated weights

    Index row_stride = 1;
    Index column_stride = 1;

    Shape input_shape;

    string convolution_type = "Valid";

    string activation_function = "Linear";

    // Batch normalization

    bool batch_normalization = false;

    TensorView gammas;
    TensorView betas;

    // @todo here or in forward propagate?
    VectorR running_means;
    VectorR running_standard_deviations;

    type momentum = type(0.9);

#ifdef CUDA
    cudnnConvolutionFwdAlgo_t convolution_algorithm;
    cudnnConvolutionBwdDataAlgo_t algo_data;
    cudnnConvolutionBwdFilterAlgo_t algo_filter;
#endif
};


#ifdef CUDA

struct ConvolutionalBackPropagationCuda : public LayerBackPropagationCuda
{
    void initialize() override;

    vector<TensorView*> get_gradient_views() override;

    void print() const override;

    void free() override;

    TensorView bias_gradients;
    TensorView weight_gradients;

    TensorView gamma_gradients;
    TensorView beta_gradients;

    void* backward_filter_workspace = nullptr;   
    size_t backward_filter_workspace_bytes = 0;
};

#endif

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// This library is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 2.1 of the License, or any later version.
// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.
// You should have received a copy of the GNU Lesser General Public
// License along with this library; if not, write to the Free Software
// Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
