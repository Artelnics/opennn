//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   C O N V O L U T I O N A L  P O O L I N G   L A Y E R   T E S T   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "convolutional_pooling_layer_test.h"

static constexpr type NUMERIC_LIMITS_MIN_4DIGIT = static_cast<type>(1e-4);

template<size_t DIM>
static bool is_equal(const Tensor<type, DIM>& expected_output, const Tensor<type, DIM>& output)
{
    Eigen::array<Index, DIM> dims;
    std::iota(begin(dims), end(dims), 0U);
    Tensor<type, DIM> abs_diff = (output - expected_output).abs();
    Tensor<bool, DIM> cmp_res = abs_diff < NUMERIC_LIMITS_MIN_4DIGIT;
    Tensor<bool, 0> ouput_equals_expected_output = cmp_res.reduce(dims, Eigen::internal::AndReducer());
    return ouput_equals_expected_output();
}

template<size_t DIM>
static constexpr Eigen::array<Index, DIM> t1d2array(const Tensor<Index, 1>& oned_tensor)
{
    return [&]<typename T, T...ints>(std::integer_sequence<T, ints...> int_seq)
    {
        Eigen::array<Index, DIM> ret({oned_tensor(ints)...});
        return ret;
    }(std::make_index_sequence<DIM>());
}

ConvolutionalPoolingLayerTest::ConvolutionalPoolingLayerTest() 
{
}


ConvolutionalPoolingLayerTest::~ConvolutionalPoolingLayerTest()
{
}


void ConvolutionalPoolingLayerTest::test_convolution_pooling_forward_propagate()
{
    cout << "test_convolution_pooling_forward_propagate\n";
    
    bool switch_train = true;

    const Index numb_of_input_rows = 5;
    const Index numb_of_input_cols = 5;
    const Index numb_of_input_chs = 2;
    const Index numb_of_input_images = 2;

    Tensor<Index, 1> conv_input_dimension(4);
    conv_input_dimension.setValues({
        numb_of_input_rows,
        numb_of_input_cols,
        numb_of_input_chs,
        numb_of_input_images
    });

    const Index numb_of_kernel_rows = 2;
    const Index numb_of_kernel_cols = 2;
    const Index numb_of_kernels = 2;

    Tensor<Index, 1> kernel_dimension(4);
    kernel_dimension.setValues({
        numb_of_kernel_rows,
        numb_of_kernel_cols,
        numb_of_input_chs,
        numb_of_kernels
    });

    ConvolutionalLayer conv_layer(
        conv_input_dimension, 
        kernel_dimension);
    conv_layer.set_activation_function(ConvolutionalLayer::ActivationFunction::RectifiedLinear);
    
    Tensor<type, 4> kernels(t1d2array<4>(kernel_dimension));
    kernels(0, 0, 0, 0) = type(1);
    kernels(0, 0, 1, 0) = type(2);
    kernels(0, 1, 0, 0) = type(3);
    kernels(0, 1, 1, 0) = type(4);
    kernels(1, 0, 0, 0) = type(5);
    kernels(1, 0, 1, 0) = type(6);
    kernels(1, 1, 0, 0) = type(7);
    kernels(1, 1, 1, 0) = type(8);

    //negate first kernel
    kernels.chip(1, 3) = -kernels.chip(0, 3);

    Tensor<type, 1> bias(numb_of_kernels);
    bias.setConstant(type(0));

    conv_layer.set_synaptic_weights(kernels);
    conv_layer.set_biases(bias);

    Tensor<Index, 1> pooling_layer_input_dimension = conv_layer.get_outputs_dimensions();

    const Index pooling_layer_row_size = 2;
    const Index pooling_layer_col_size = 2;
    const Index stride = 2;
    
    Tensor<Index, 1> pool_dimension(2);
    pool_dimension.setValues({
        pooling_layer_row_size,
        pooling_layer_col_size,
    });

    PoolingLayer pooling_layer(pooling_layer_input_dimension, pool_dimension);
    pooling_layer.set_row_stride(stride);
    pooling_layer.set_column_stride(stride);
    pooling_layer.set_pooling_method(PoolingLayer::PoolingMethod::MaxPooling);

    ConvolutionalLayerForwardPropagation conv_forward_propagation(numb_of_input_images, &conv_layer);
    PoolingLayerForwardPropagation pooling_layer_forward_propagation(numb_of_input_images, &pooling_layer);

    Tensor<type, 4> conv_input(t1d2array<4>(conv_input_dimension));
    conv_input.setConstant(type(1));

    conv_layer.forward_propagate(
        conv_input.data(), 
        conv_input_dimension, 
        static_cast<LayerForwardPropagation*>(&conv_forward_propagation), 
        switch_train);
    pooling_layer.forward_propagate(
        conv_forward_propagation.outputs_data, 
        conv_forward_propagation.outputs_dimensions, 
        static_cast<LayerForwardPropagation*>(&pooling_layer_forward_propagation), 
        switch_train);

    TensorMap<Tensor<type, 4>> output(
        pooling_layer_forward_propagation.outputs_data, 
        t1d2array<4>(pooling_layer_forward_propagation.outputs_dimensions));
    //test
    Tensor<type, 4> expected_output(2, 2, 2, 2);
    expected_output.setZero();
    expected_output.chip(0, 2).setConstant(type(36));

    assert_true(is_equal<4>(expected_output, output), LOG);
}

void ConvolutionalPoolingLayerTest::test_pooling_convolution_forward_propagate()
{
    cout << "test_pooling_convolution_forward_propagate\n";

    bool switch_train = true;

    const Index numb_of_input_rows = 4;
    const Index numb_of_input_cols = 4;
    const Index numb_of_input_chs = 2;
    const Index numb_of_input_images = 2;

    Tensor<Index, 1> pooling_input_dimension(4);
    pooling_input_dimension.setValues({
        numb_of_input_rows,
        numb_of_input_cols,
        numb_of_input_chs,
        numb_of_input_images
    });

    const Index pool_row_size = 2;
    const Index pool_col_size = 2;
    Tensor<Index, 1> pool_dimension(2);
    pool_dimension.setValues({
        pool_row_size,
        pool_col_size
    });

    const Index stride = 1;

    PoolingLayer pooling_layer(pooling_input_dimension, pool_dimension);
    pooling_layer.set_row_stride(stride);
    pooling_layer.set_column_stride(stride);
    pooling_layer.set_pooling_method(PoolingLayer::PoolingMethod::MaxPooling);

    const Index numb_of_kernel_rows = 2;
    const Index numb_of_kernel_cols = 2;
    const Index numb_of_kernels = 1;

    Tensor<Index, 1> kernel_dimension(4);
    kernel_dimension.setValues({
        numb_of_kernel_rows,
        numb_of_kernel_cols,
        numb_of_input_chs,
        numb_of_kernels
    });

    Tensor<type, 4> kernel(t1d2array<4>(kernel_dimension));
    kernel.setConstant(type(1));

    Tensor<type, 1> bias(numb_of_kernels);
    bias.setConstant(type(0));

    Tensor<Index, 1> conv_input_dimension = pooling_layer.get_outputs_dimensions();

    ConvolutionalLayer conv_layer(conv_input_dimension, kernel_dimension);
    conv_layer.set_synaptic_weights(kernel);
    conv_layer.set_biases(bias);
    conv_layer.set_activation_function(ConvolutionalLayer::ActivationFunction::RectifiedLinear);

    PoolingLayerForwardPropagation pooling_layer_forward_propagation(numb_of_input_images, &pooling_layer);
    ConvolutionalLayerForwardPropagation conv_layer_forward_propagation(numb_of_input_images, &conv_layer);

    Tensor<type, 4> input(t1d2array<4>(pooling_input_dimension));
    input(0, 0, 0, 0) = type(1);
    input(0, 0, 1, 0) = type(2);
    input(0, 1, 0, 0) = type(3);
    input(0, 1, 1, 0) = type(4);
    input(0, 2, 0, 0) = type(5);
    input(0, 2, 1, 0) = type(6);
    input(0, 3, 0, 0) = type(7);
    input(0, 3, 1, 0) = type(8);
    input.chip(1, 3).chip(0, 0) = input.chip(0, 3).chip(0, 0);
    input.chip(1, 0) = input.chip(0, 0) + type(8);
    input.chip(2, 0) = input.chip(1, 0) + type(8);
    input.chip(3, 0) = input.chip(2, 0) + type(8);

    pooling_layer.forward_propagate(
        input.data(), 
        pooling_input_dimension, 
        static_cast<LayerForwardPropagation*>(&pooling_layer_forward_propagation), 
        switch_train);
    conv_layer.forward_propagate(
        pooling_layer_forward_propagation.outputs_data,
        pooling_layer.get_outputs_dimensions(),
        static_cast<LayerForwardPropagation*>(&conv_layer_forward_propagation),
        switch_train
    );

    //test
    Tensor<type, 4> expected_output(2, 2, 1, 2);
    expected_output(0, 0, 0, 0) = type(132);
    expected_output(0, 1, 0, 0) = type(148);
    expected_output(1, 0, 0, 0) = type(196);
    expected_output(1, 1, 0, 0) = type(212);
    expected_output.chip(1, 3) = expected_output.chip(0, 3);

    TensorMap<Tensor<type, 4>> output(
        conv_layer_forward_propagation.outputs_data,
        t1d2array<4>(conv_layer_forward_propagation.outputs_dimensions));

    assert_true(is_equal<4>(expected_output, output), LOG);
}

void ConvolutionalPoolingLayerTest::test_convolution_pooling_backward_pass()
{
    cout << "test_convolution_pooling_backward_pass\n";
    const Index numb_of_input_rows = 5;
    const Index numb_of_input_cols = 5;
    const Index numb_of_input_chs = 2;
    const Index numb_of_input_images = 2;

    Tensor<Index, 1> conv_input_dimension(4);
    conv_input_dimension.setValues({
        numb_of_input_rows,
        numb_of_input_cols,
        numb_of_input_chs,
        numb_of_input_images
    });

    const Index numb_of_kernel_rows = 2;
    const Index numb_of_kernel_cols = 2;
    const Index numb_of_kernels = 2;

    Tensor<Index, 1> kernel_dimension(4);
    kernel_dimension.setValues({
        numb_of_kernel_rows,
        numb_of_kernel_cols,
        numb_of_input_chs,
        numb_of_kernels
    });

    ConvolutionalLayer conv_layer(
        conv_input_dimension, 
        kernel_dimension);
    conv_layer.set_activation_function(ConvolutionalLayer::ActivationFunction::RectifiedLinear);

    Tensor<Index, 1> pooling_layer_input_dimension = conv_layer.get_outputs_dimensions();

    const Index pooling_layer_row_size = 2;
    const Index pooling_layer_col_size = 2;
    const Index stride = 2;
    
    Tensor<Index, 1> pool_dimension(2);
    pool_dimension.setValues({
        pooling_layer_row_size,
        pooling_layer_col_size,
    });

    PoolingLayer pooling_layer(pooling_layer_input_dimension, pool_dimension);
    pooling_layer.set_row_stride(stride);
    pooling_layer.set_column_stride(stride);
    pooling_layer.set_pooling_method(PoolingLayer::PoolingMethod::MaxPooling);

    PoolingLayerForwardPropagation pooling_layer_forward_propagation(
        numb_of_input_images,
        &pooling_layer
    );
    pooling_layer_forward_propagation.switches.resize(t1d2array<4>(pooling_layer.get_outputs_dimensions()));
    //image 1
    pooling_layer_forward_propagation.switches(0, 0, 0, 0) = make_tuple(0, 0);
    pooling_layer_forward_propagation.switches(0, 0, 1, 0) = make_tuple(0, 1);
    pooling_layer_forward_propagation.switches(0, 1, 0, 0) = make_tuple(1, 3);
    pooling_layer_forward_propagation.switches(0, 1, 1, 0) = make_tuple(0, 2);
    pooling_layer_forward_propagation.switches(1, 0, 0, 0) = make_tuple(3, 0);
    pooling_layer_forward_propagation.switches(1, 0, 1, 0) = make_tuple(2, 1);
    pooling_layer_forward_propagation.switches(1, 1, 0, 0) = make_tuple(2, 3);
    pooling_layer_forward_propagation.switches(1, 1, 1, 0) = make_tuple(3, 2);
    //image 2
    pooling_layer_forward_propagation.switches(0, 0, 0, 0) = make_tuple(1, 1);
    pooling_layer_forward_propagation.switches(0, 0, 1, 0) = make_tuple(1, 0);
    pooling_layer_forward_propagation.switches(0, 1, 0, 0) = make_tuple(0, 2);
    pooling_layer_forward_propagation.switches(0, 1, 1, 0) = make_tuple(1, 2);
    pooling_layer_forward_propagation.switches(1, 0, 0, 0) = make_tuple(2, 0);
    pooling_layer_forward_propagation.switches(1, 0, 1, 0) = make_tuple(3, 1);
    pooling_layer_forward_propagation.switches(1, 1, 0, 0) = make_tuple(3, 3);
    pooling_layer_forward_propagation.switches(1, 1, 1, 0) = make_tuple(2, 2);

    PoolingLayerBackPropagation pooling_layer_back_propagation(
        numb_of_input_images,
        &pooling_layer
    );
    TensorMap<Tensor<type, 4>> next_delta(
        pooling_layer_back_propagation.deltas_data, 
        t1d2array<4>(pooling_layer_back_propagation.deltas_dimensions));
    next_delta(0, 0, 0, 0) = type(1);
    next_delta(0, 0, 1, 0) = type(2);
    next_delta(0, 1, 0, 0) = type(3);
    next_delta(0, 1, 1, 0) = type(4);
    next_delta(1, 0, 0, 0) = type(5);
    next_delta(1, 0, 1, 0) = type(6);
    next_delta(1, 1, 0, 0) = type(7);
    next_delta(1, 1, 1, 0) = type(8);
    next_delta.chip(1, 3) = next_delta.chip(0, 3) + type(8);
    
    ConvolutionalLayerBackPropagation conv_layer_back_propagation(numb_of_input_images, &conv_layer);

    conv_layer.calculate_hidden_delta(
        static_cast<LayerForwardPropagation*>(&pooling_layer_forward_propagation), 
        static_cast<LayerBackPropagation*>(&pooling_layer_back_propagation),
        static_cast<LayerBackPropagation*>(&conv_layer_back_propagation));

    //test
    Tensor<type, 4> expected_output(4, 4, 2, 2);
    expected_output.setZero();
    //image 1
    expected_output(0, 0, 0, 0) = type(1);
    expected_output(0, 1, 1, 0) = type(2);
    expected_output(1, 3, 0, 0) = type(3);
    expected_output(0, 2, 1, 0) = type(4);
    expected_output(3, 0, 0, 0) = type(5);
    expected_output(2, 1, 1, 0) = type(6);
    expected_output(2, 3, 0, 0) = type(7);
    expected_output(3, 2, 1, 0) = type(8);
    //image 2
    expected_output(1, 1, 0, 1) = type(9);
    expected_output(1, 0, 1, 1) = type(10);
    expected_output(0, 2, 0, 1) = type(11);
    expected_output(1, 2, 1, 1) = type(12);
    expected_output(2, 0, 0, 1) = type(13);
    expected_output(3, 1, 1, 1) = type(14);
    expected_output(3, 3, 0, 1) = type(15);
    expected_output(2, 2, 1, 1) = type(16);

    TensorMap<Tensor<type, 4>> output(
        conv_layer_back_propagation.deltas_data,
        t1d2array<4>(conv_layer_back_propagation.deltas_dimensions));

    assert_true(is_equal<4>(expected_output, output), LOG);
}


void ConvolutionalPoolingLayerTest::run_test_case()
{
   cout << "Running convolutional pooling layer test case...\n";
   test_convolution_pooling_forward_propagate();
   test_pooling_convolution_forward_propagate();
   test_convolution_pooling_backward_pass();
   
   cout << "End of convolutional layer test case.\n\n";
}


// OpenNN: Open Neural Networks Library.
// Copyright (C) 2005-2021 Artificial Intelligence Techniques, SL.
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
;
