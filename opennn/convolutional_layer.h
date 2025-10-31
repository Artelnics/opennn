//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   C O N V O L U T I O N A L   L A Y E R   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#ifndef CONVOLUTIONALLAYER_H
#define CONVOLUTIONALLAYER_H

#include "layer.h"

namespace opennn
{

class Convolutional final : public Layer
{

public:
    //enum class Convolution{Valid, Same};

    Convolutional(const dimensions& = {3, 3, 1},                    // Input dimensions {height,width,channels}
                  const dimensions& = {3, 3, 1, 1},                 // Kernel dimensions {kernel_height,kernel_width,channels,kernels_number}
                  const string& = "Linear",
                  const dimensions& = { 1, 1 },                     // Stride dimensions {row_stride,column_stride}
                  const string& = "Valid",          // Convolution type (Valid || Same)
                  const bool& = false,                              // Batch Normalization)
                  const string& = "convolutional_layer");

    bool get_batch_normalization() const;

    Tensor<type, 1> get_scales() const;
    Tensor<type, 1> get_offsets() const;

    const string& get_activation_function() const;

    dimensions get_input_dimensions() const override;
    dimensions get_output_dimensions() const override;

    pair<Index, Index> get_padding() const;

    Eigen::array<pair<Index, Index>, 4> get_paddings() const;

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

    vector<ParameterView> get_parameter_views() const override;

    // Set

    void set(const dimensions& = {0, 0, 0},
             const dimensions& = {3, 3, 1, 1},
             const string& = "Linear",
             const dimensions& = {1, 1},
             const string& = "Valid",
             const bool& = false,
             const string& = "convolutional_layer");

    void set_activation_function(const string&);

    void set_batch_normalization(const bool&);

    void set_convolution_type(const string&);

    void set_row_stride(const Index&);

    void set_column_stride(const Index&);

    void set_input_dimensions(const dimensions&) override;

    // Forward propagation

    void preprocess_inputs(const Tensor<type, 4>&,
                           Tensor<type, 4>&) const;

    void calculate_convolutions(const Tensor<type, 4>&,
                                Tensor<type, 4>&) const;

    void apply_batch_normalization(unique_ptr<LayerForwardPropagation>&, const bool&);

    void forward_propagate(const vector<TensorView>&,
                           unique_ptr<LayerForwardPropagation>&,
                           const bool&) override;

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

    void forward_propagate_cuda(const vector<float*>&,
                                unique_ptr<LayerForwardPropagationCuda>&,
                                const bool&) override;

    void back_propagate_cuda(const vector<float*>&,
                             const vector<float*>&,
                             unique_ptr<LayerForwardPropagationCuda>&,
                             unique_ptr<LayerBackPropagationCuda>&) const override;

    vector<ParameterView> get_parameter_views_device() const override;

    void copy_parameters_host();

    void copy_parameters_device();

    void allocate_parameters_device();

    void free_parameters_device();

    bool use_convolutions = true;

protected:

    float* biases_device = nullptr;
    float* weights_device = nullptr;

    cudnnTensorDescriptor_t biases_tensor_descriptor = nullptr;

    // Batch Normalization

    float* bn_scale_device = nullptr;
    float* bn_offset_device = nullptr;
    float* bn_running_mean_device = nullptr;
    float* bn_running_variance_device = nullptr;

    cudnnTensorDescriptor_t bn_tensor_descriptor = nullptr;

    // Activations

    cudnnActivationDescriptor_t activation_descriptor = nullptr;

#endif

private:

    Tensor<type, 4> weights;

    Tensor<type, 1> biases;

    Index row_stride = 1;

    Index column_stride = 1;

    dimensions input_dimensions;

    string convolution_type = "Valid";

    string activation_function = "Linear";

    // Batch normalization

    bool batch_normalization = false;

    Tensor<type, 1> moving_means;
    Tensor<type, 1> moving_standard_deviations;

    type momentum = type(0.9);

    const type epsilon = type(1e-5);

    Tensor<type, 1> scales;
    Tensor<type, 1> offsets;
};


struct ConvolutionalForwardPropagation final : LayerForwardPropagation
{
    ConvolutionalForwardPropagation(const Index& = 0, Layer* = nullptr);

    TensorView get_output_pair() const override;

    void initialize() override;

    void print() const override;

    Tensor<type, 4> outputs;

    Tensor<type, 4> preprocessed_inputs;

    Tensor<type, 1> means;
    Tensor<type, 1> standard_deviations;

    Tensor<type, 4> activation_derivatives;
};


struct ConvolutionalBackPropagation final : LayerBackPropagation
{
    ConvolutionalBackPropagation(const Index& = 0, Layer* = nullptr);

    vector<TensorView> get_input_derivative_views() const override;

    vector<ParameterView> get_parameter_delta_views() const override;

    void initialize() override;

    void print() const override;

    Tensor<type, 1> bias_deltas;
    Tensor<type, 4> weight_deltas;
    Tensor<type, 4> input_deltas;

    Tensor<type, 4> rotated_weights;

    Tensor<type, 1> bn_scale_deltas;
    Tensor<type, 1> bn_offset_deltas;
};


#ifdef OPENNN_CUDA

struct ConvolutionalForwardPropagationCuda : public LayerForwardPropagationCuda
{
    ConvolutionalForwardPropagationCuda(const Index& = 0, Layer* = nullptr);

    void set(const Index& = 0, Layer* = nullptr) override;

    void print() const override;

    void free() override;

    int output_batch_size, output_channels, output_height, output_width = 0;

    float* convolutions = nullptr;

    cudnnTensorDescriptor_t input_tensor_descriptor = nullptr;
    cudnnTensorDescriptor_t biases_tensor_descriptor = nullptr;
    cudnnTensorDescriptor_t inputs_tensor_descriptor = nullptr;

    cudnnFilterDescriptor_t kernel_descriptor = nullptr;
    cudnnConvolutionDescriptor_t convolution_descriptor = nullptr;

    cudnnConvolutionFwdAlgo_t convolution_algorithm;

    void* workspace = nullptr;
    size_t workspace_bytes = 0;

    bool is_first_layer = false;

    float* reordered_inputs_device = nullptr;

    // Batch Normalizarion
    float* bn_saved_mean = nullptr;
    float* bn_saved_inv_variance = nullptr;
};


struct ConvolutionalBackPropagationCuda : public LayerBackPropagationCuda
{
    ConvolutionalBackPropagationCuda(const Index& = 0, Layer* = nullptr);

    vector<ParameterView> get_parameter_delta_views_device() const override;

    void set(const Index& = 0, Layer* = nullptr) override;

    void print() const override;

    void free() override;

    type* bias_deltas_device = nullptr;
    type* weight_deltas_device = nullptr;

    void* backward_data_workspace = nullptr;
    void* backward_filter_workspace = nullptr;
    size_t backward_data_workspace_bytes = 0;
    size_t backward_filter_workspace_bytes = 0;

    cudnnTensorDescriptor_t input_tensor_descriptor = nullptr;
    cudnnTensorDescriptor_t deltas_tensor_descriptor = nullptr;

    cudnnFilterDescriptor_t kernel_descriptor = nullptr;
    cudnnFilterDescriptor_t weight_deltas_tensor_descriptor = nullptr;

    cudnnConvolutionDescriptor_t convolution_descriptor = nullptr;

    // Batch Normalizarion
    float* bn_scale_deltas_device = nullptr;
    float* bn_offset_deltas_device = nullptr;
};

#endif

}

#endif

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2025 Artificial Intelligence Techniques, SL.
//
// This library is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 2.1 of the License, or any later version.
//
// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.

// You should have received a copy of the GNU Lesser General Public
// License along with this library; if not, write to the Free Software

// Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
