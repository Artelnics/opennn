//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   C O N V O L U T I O N A L   L A Y E R   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#ifndef CONVOLUTIONALLAYER_H
#define CONVOLUTIONALLAYER_H

#define EIGEN_USE_THREADS

#include "layer.h"

namespace opennn
{

//#ifdef OPENNN_CUDA
//struct ConvolutionalForwardPropagationCuda;
//struct ConvolutionalBackPropagationCuda;
//#endif


class Convolutional : public Layer
{

public:
    enum class Convolution{Valid, 
                           Same};

    enum class Activation{Logistic,
                          HyperbolicTangent,
                          Linear,
                          RectifiedLinear,
                          ExponentialLinear,
                          ScaledExponentialLinear,
                          SoftPlus,
                          SoftSign,
                          HardSigmoid};

    Convolutional(const dimensions& = {3, 3, 1},                    // Input dimensions {height,width,channels}
                  const dimensions& = {3, 3, 1, 1},                 // Kernel dimensions {kernel_height,kernel_width,channels,kernels_number}
                  const Activation& = Activation::Linear,
                  const dimensions& = { 1, 1 },                     // Stride dimensions {row_stride,column_stride}
                  const Convolution& = Convolution::Valid,  // Convolution type (Valid || Same)
                  const string = "convolutional_layer");

    bool get_batch_normalization() const;

    Activation get_activation_function() const;

    string get_activation_function_string() const;

    dimensions get_input_dimensions() const override;
    dimensions get_output_dimensions() const override;

    pair<Index, Index> get_padding() const;

    array<pair<Index, Index>, 4> get_paddings() const;

    array<Index, 4> get_strides() const;

    Index get_output_height() const;
    Index get_output_width() const;

    Convolution get_convolution_type() const;
    string write_convolution_type() const;

    Index get_column_stride() const;

    Index get_row_stride() const;

    Index get_kernel_height() const;
    Index get_kernel_width() const;
    Index get_kernel_channels() const;
    Index get_kernels_number() const;

    Index get_padding_width() const;
    Index get_padding_height() const;

    Index get_input_channels() const;
    Index get_input_height() const;
    Index get_input_width() const;

    Tensor<type, 1> get_parameters() const override;
    Index get_parameters_number() const override;

    // Set

    void set(const dimensions& = {0, 0, 0},
             const dimensions& = {3, 3, 1, 1},
             const Activation& = Activation::Linear,
             const dimensions& = {1, 1},
             const Convolution& = Convolution::Valid,
             const string = "convolutional_layer");

    void set_activation_function(const Activation&);
    void set_activation_function(const string&);

    void set_batch_normalization(const bool&);

    void set_convolution_type(const Convolution&);
    void set_convolution_type(const string&);

    void set_parameters(const Tensor<type, 1>&, Index&) override;

    void set_row_stride(const Index&);

    void set_column_stride(const Index&);

    void set_input_dimensions(const dimensions&) override;

    // Initialization

    void set_parameters_constant(const type&) override;

    void set_parameters_random() override;

    // Forward propagation

    void preprocess_inputs(const Tensor<type, 4>&,
                           Tensor<type, 4>&) const;

    void calculate_convolutions(const Tensor<type, 4>&,
                                Tensor<type, 4>&) const;

    void normalize(unique_ptr<LayerForwardPropagation>&, const bool&);

    void shift(unique_ptr<LayerForwardPropagation>&);

    void calculate_activations(Tensor<type, 4>&, Tensor<type, 4>&) const;

    void forward_propagate(const vector<pair<type*, dimensions>>&,
                           unique_ptr<LayerForwardPropagation>&,
                           const bool&) override;

   // Back propagation

   void back_propagate(const vector<pair<type*, dimensions>>&,
                       const vector<pair<type*, dimensions>>&,
                       unique_ptr<LayerForwardPropagation>&,
                       unique_ptr<LayerBackPropagation>&) const override;

   void insert_gradient(unique_ptr<LayerBackPropagation>&,
                        Index&,
                        Tensor<type, 1>&) const override;

   void from_XML(const XMLDocument&) override;
   void to_XML(XMLPrinter&) const override;

   void print() const override;

    #ifdef OPENNN_CUDA

    public:

    void forward_propagate_cuda(const Tensor<pair<type*, dimensions>, 1>&,
                                LayerForwardPropagationCuda*,
                                const bool&) final;

    void back_propagate_cuda(const Tensor<pair<type*, dimensions>, 1>&,
                             const Tensor<pair<type*, dimensions>, 1>&,
                             LayerForwardPropagationCuda*,
                             LayerBackPropagationCuda*) const final;

    void insert_gradient_cuda(LayerBackPropagationCuda*, const Index&, float*) const;

    void set_parameters_cuda(const float*, const Index&);

    void get_parameters_cuda(const Tensor<type, 1>&, const Index&);

    void allocate_parameters_device();
    void free_parameters_device();
    void copy_parameters_device();
    void copy_parameters_host();

    float* get_weights_device() const;
    float* get_biases_device() const;

    void print_cuda_parameters();

    void reverse_cuda(Index, Index, Index, float*);

    private:

    float* biases_device = nullptr;
    float* weights_device = nullptr;

    cudnnHandle_t cudnn_handle;
    cublasHandle_t cublas_handle;

#endif

private:

   Tensor<type, 4> weights;

   Tensor<type, 1> biases;

   Index row_stride = 1;

   Index column_stride = 1;

   dimensions input_dimensions;

   Convolution convolution_type = Convolution::Valid;

   Activation activation_function = Activation::Linear;

   // Batch normalization

   bool batch_normalization = false;

   Tensor<type, 1> moving_means;
   Tensor<type, 1> moving_standard_deviations;

   type momentum = type(0.9);
   const type epsilon = type(1.0e-5);

   Tensor<type, 1> scales;
   Tensor<type, 1> offsets;
};


struct ConvolutionalForwardPropagation : LayerForwardPropagation
{
   
   ConvolutionalForwardPropagation(const Index& = 0, Layer* = nullptr);
      
   pair<type*, dimensions> get_outputs_pair() const override;

   void set(const Index& = 0, Layer* = nullptr);

   void print() const override;

   Tensor<type, 4> outputs;

   Tensor<type, 4> preprocessed_inputs;

   Tensor<type, 1> means;
   Tensor<type, 1> standard_deviations;

   Tensor<type, 4> activation_derivatives;
};


struct ConvolutionalBackPropagation : LayerBackPropagation
{
   ConvolutionalBackPropagation(const Index& = 0, Layer* = nullptr);

   vector<pair<type*, dimensions>> get_input_derivative_pairs() const override;

   void set(const Index& = 0, Layer* = nullptr);

   void print() const override;

   Tensor<type, 1> bias_derivatives;
   Tensor<type, 4> weight_derivatives;
   Tensor<type, 4> input_derivatives;

   Tensor<type, 4> rotated_weights;

};


#ifdef OPENNN_CUDA

struct ConvolutionalLayerForwardPropagationCuda : public LayerForwardPropagationCuda
{
    explicit ConvolutionalLayerForwardPropagationCuda(const Index&, Layer*);

    void set(const Index&, Layer*) override;

    void print() const override;

    void free() override;

    pair<type*, dimensions> get_outputs_pair() const;

    cudnnTensorDescriptor_t inputs_tensor_descriptor = nullptr;
    cudnnTensorDescriptor_t biases_tensor_descriptor = nullptr;
    cudnnFilterDescriptor_t kernel_descriptor = nullptr;
    cudnnConvolutionDescriptor_t convolution_descriptor = nullptr;
    cudnnActivationDescriptor_t activation_descriptor = nullptr;

    cudnnConvolutionFwdAlgo_t convolution_algorithm;
    vector<cudnnConvolutionFwdAlgoPerf_t> perfResults;
    int returnedAlgoCount = 0;

    type* convolutions = nullptr;
    void* workspace = nullptr;
    size_t workspace_bytes = 0;

    int output_batch_size = 0, output_channels = 0, output_height = 0, output_width = 0;
};


struct ConvolutionalLayerBackPropagationCuda : public LayerBackPropagationCuda
{
    explicit ConvolutionalLayerBackPropagationCuda(const Index&, Layer*);

    void set(const Index&, Layer*) override;

    void print() const override;

    void free() override;

    cudnnTensorDescriptor_t deltas_device_tensor_descriptor = nullptr;
    cudnnTensorDescriptor_t error_combinations_derivatives_tensor_descriptor = nullptr;
    cudnnTensorDescriptor_t inputs_tensor_descriptor = nullptr;
    cudnnFilterDescriptor_t kernel_descriptor = nullptr;
    cudnnFilterDescriptor_t kernel_weights_derivatives_tensor_descriptor = nullptr;
    cudnnConvolutionDescriptor_t convolution_descriptor = nullptr;

    type* error_combinations_derivatives = nullptr;
    type* biases_derivatives = nullptr;
    type* kernel_weights_derivatives = nullptr;
    void* backward_data_workspace = nullptr;
    void* backward_filter_workspace = nullptr;
    size_t backward_data_workspace_bytes = 0;
    size_t backward_filter_workspace_bytes = 0;
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
