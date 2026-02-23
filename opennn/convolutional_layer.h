//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   C O N V O L U T I O N A L   L A Y E R   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#pragma once

#include "layer.h"
#include "tensors.h"

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

    void reorder_weights_for_cudnn();

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

    vector<TensorView*> get_parameter_views() override;

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

    // Forward propagation

    void preprocess_inputs(const Tensor4&, Tensor4&) const;

    void calculate_convolutions(const Tensor4&, TensorMap4) const;

    void forward_propagate(const vector<TensorView>&,
                           unique_ptr<LayerForwardPropagation>&,
                           bool) override;

    // Back propagation

    void back_propagate(const vector<TensorView>&,
                        const vector<TensorView>&,
                        unique_ptr<LayerForwardPropagation>&,
                        unique_ptr<LayerBackPropagation>&) const override;

    void from_XML(const XMLDocument&) override;
    void to_XML(XMLPrinter&) const override;

    void print() const override;

#ifdef OPENNN_CUDA

public:

    void forward_propagate(const vector<TensorViewCuda>&,
                                unique_ptr<LayerForwardPropagationCuda>&,
                                bool) override;

    void back_propagate(const vector<TensorViewCuda>&,
                        const vector<TensorViewCuda>&,
                        unique_ptr<LayerForwardPropagationCuda>&,
                        unique_ptr<LayerBackPropagationCuda>&) const override;

    vector<TensorViewCuda*> get_parameter_views_device() override;

    void copy_parameters_device();

protected:

    TensorViewCuda biases_device;
    TensorViewCuda weights_device;

    // Batch Normalization

    TensorViewCuda gammas_device;
    TensorViewCuda betas_device;

    TensorCuda running_means_device;
    TensorCuda running_variances_device;

    // Activations

    cudnnActivationDescriptor_t activation_descriptor = nullptr;

#endif

private:

    TensorView weights;
    TensorView biases;

    Index row_stride = 1;
    Index column_stride = 1;

    Shape input_shape;

    string convolution_type = "Valid";

    string activation_function = "Linear";

    // Batch normalization

    bool batch_normalization = false;

    TensorView gammas;
    TensorView betas;

    VectorR running_means;
    VectorR running_standard_deviations;

    type momentum = type(0.9);

};


struct ConvolutionalForwardPropagation final : LayerForwardPropagation
{
    ConvolutionalForwardPropagation(const Index = 0, Layer* = nullptr);

    void initialize() override;

    vector<TensorView*> get_workspace_views() override;

    void print() const override;

    Tensor4 preprocessed_inputs;

    TensorView means;
    TensorView standard_deviations;

    TensorView activation_derivatives;
};


struct ConvolutionalBackPropagation final : LayerBackPropagation
{
    ConvolutionalBackPropagation(const Index = 0, Layer* = nullptr);

    void initialize() override;

    vector<TensorView*> get_gradient_views() override;

    void print() const override;

    TensorView bias_gradients;
    TensorView weight_gradients;

    TensorView gamma_gradients;
    TensorView beta_gradients;

    Tensor4 rotated_weights;
};


#ifdef OPENNN_CUDA

struct ConvolutionalForwardPropagationCuda : public LayerForwardPropagationCuda
{
    ConvolutionalForwardPropagationCuda(const Index = 0, Layer* = nullptr);

    void initialize() override;

    void print() const override;

    void free() override;

    int output_batch_size, output_channels, output_height, output_width = 0;

    TensorCuda reordered_inputs_device;

    TensorCuda convolutions;

    cudnnTensorDescriptor_t input_tensor_descriptor = nullptr;

    cudnnFilterDescriptor_t kernel_descriptor = nullptr;

    cudnnConvolutionDescriptor_t convolution_descriptor = nullptr;
    cudnnConvolutionFwdAlgo_t convolution_algorithm;

    void* workspace = nullptr;
    size_t workspace_bytes = 0;

    bool is_first_layer = false;

    TensorCuda batch_means;
    TensorCuda bn_saved_inv_variance;
};


struct ConvolutionalBackPropagationCuda : public LayerBackPropagationCuda
{
    ConvolutionalBackPropagationCuda(const Index = 0, Layer* = nullptr);

    void initialize() override;

    vector<TensorViewCuda*> get_workspace_views() override;

    void print() const override;

    void free() override;

    TensorViewCuda bias_gradients;
    TensorViewCuda weight_gradients;

    void* backward_data_workspace = nullptr;
    void* backward_filter_workspace = nullptr;
    size_t backward_data_workspace_bytes = 0;
    size_t backward_filter_workspace_bytes = 0;

    cudnnTensorDescriptor_t gradients_tensor_descriptor = nullptr;

    cudnnFilterDescriptor_t kernel_descriptor = nullptr;
    cudnnFilterDescriptor_t weight_gradients_filter_descriptor = nullptr;

    cudnnConvolutionDescriptor_t convolution_descriptor = nullptr;

    cudnnConvolutionBwdDataAlgo_t algo_data;
    cudnnConvolutionBwdFilterAlgo_t algo_filter;

    TensorViewCuda gamma_gradients;
    TensorViewCuda beta_gradients;
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
