//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   F L A T T E N   P E R C E P T R O N    L A Y E R   T E S T   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "flatten_perceptron_layer_test.h"

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

FlattenPerceptronLayerTest::FlattenPerceptronLayerTest() 
{
}


FlattenPerceptronLayerTest::~FlattenPerceptronLayerTest()
{
}

void FlattenPerceptronLayerTest::test_flatten_perceptron_forward_propagate()
{
    cout << "test_flatten_perceptron_forward_propagate\n";

    bool switch_train = true;

    const Index numb_of_input_rows = 2;
    const Index numb_of_input_cols = 2;
    const Index numb_of_input_chs = 2;
    const Index numb_of_input_images = 2;

    Tensor<Index, 1> flatten_input_dimension(4);

    flatten_input_dimension[Convolutional4dDimensions::row_index] = numb_of_input_rows;
    flatten_input_dimension[Convolutional4dDimensions::column_index] = numb_of_input_cols;
    flatten_input_dimension[Convolutional4dDimensions::channel_index] = numb_of_input_chs;
    flatten_input_dimension[Convolutional4dDimensions::sample_index] = numb_of_input_images;

    FlattenLayer flatten_layer(flatten_input_dimension);

    const Index perceptron_layer_input_numbers = flatten_layer.get_outputs_dimensions()(1);
    const Index neurons_number = 4;

    PerceptronLayer perceptron_layer(
        perceptron_layer_input_numbers, 
        neurons_number, 
        PerceptronLayer::ActivationFunction::RectifiedLinear);

    perceptron_layer.set_synaptic_weights_constant(type(1));
    perceptron_layer.set_biases_constant(type(1));

    Tensor<type, 4> flatten_layer_input(t1d2array<4>(flatten_input_dimension));
    //image 1
    flatten_layer_input(0, 0, 0, 0) = type(1);
    flatten_layer_input(0, 1, 0, 0) = type(2);
    flatten_layer_input(0, 0, 1, 0) = type(3);
    flatten_layer_input(0, 1, 1, 0) = type(4);
    flatten_layer_input.chip(0, Convolutional4dDimensions::sample_index).
        chip(1, Convolutional4dDimensions::row_index - 1) = 
        flatten_layer_input.chip(0, Convolutional4dDimensions::sample_index).
        chip(0, Convolutional4dDimensions::row_index - 1) + type(4);
    //image 2
    flatten_layer_input.chip(1, Convolutional4dDimensions::sample_index).
        chip(0, Convolutional4dDimensions::row_index - 1) = 
        flatten_layer_input.chip(0, Convolutional4dDimensions::sample_index).chip(1, Convolutional4dDimensions::row_index - 1) + type(4);
    flatten_layer_input.chip(1, Convolutional4dDimensions::sample_index).
        chip(1, Convolutional4dDimensions::row_index - 1) = 
        flatten_layer_input.chip(1, Convolutional4dDimensions::sample_index).chip(0, Convolutional4dDimensions::row_index - 1) + type(4);

    FlattenLayerForwardPropagation flatten_layer_forward_propagation(numb_of_input_images, &flatten_layer);
    flatten_layer.forward_propagate(
        flatten_layer_input.data(), 
        flatten_input_dimension,
        &flatten_layer_forward_propagation, 
        switch_train);
    PerceptronLayerForwardPropagation perceptron_layer_forward_propagation(
        numb_of_input_images, 
        &perceptron_layer);

    flatten_layer.forward_propagate(
        flatten_layer_input.data(),
        flatten_input_dimension,
        static_cast<LayerForwardPropagation*>(&flatten_layer_forward_propagation),
        switch_train
    );

    perceptron_layer.forward_propagate(
        flatten_layer_forward_propagation.outputs_data,
        flatten_layer_forward_propagation.outputs_dimensions,
        static_cast<LayerForwardPropagation*>(&perceptron_layer_forward_propagation),
        switch_train
    );
    
    //test
    Tensor<type, 2> expected_output(2, 4);
    //image 1
    expected_output.chip(0, 0).setConstant(type(37));
    //image 2
    expected_output.chip(1, 0).setConstant(type(101));

    TensorMap<Tensor<type, 2>> output(
        perceptron_layer_forward_propagation.outputs_data,
        t1d2array<2>(perceptron_layer_forward_propagation.outputs_dimensions));

    assert_true(is_equal<2>(expected_output, output), LOG);
}

void FlattenPerceptronLayerTest::test_flatten_perceptron_backward_pass()
{
    cout << "test_flatten_perceptron_backward_pass\n";

    const Index numb_of_input_rows = 2;
    const Index numb_of_input_cols = 2;
    const Index numb_of_input_chs = 1;
    const Index numb_of_input_images = 3;

    Tensor<Index, 1> flatten_input_dimension(4);
    flatten_input_dimension[Convolutional4dDimensions::row_index] = numb_of_input_rows;
    flatten_input_dimension[Convolutional4dDimensions::column_index] = numb_of_input_cols;
    flatten_input_dimension[Convolutional4dDimensions::channel_index] = numb_of_input_chs;
    flatten_input_dimension[Convolutional4dDimensions::sample_index] = numb_of_input_images;


    
    FlattenLayer flatten_layer(flatten_input_dimension);

    const Index perceptron_layer_input_numbers = flatten_layer.get_outputs_dimensions()(1);
    const Index neurons_number = 2;

    PerceptronLayer perceptron_layer(
        perceptron_layer_input_numbers, 
        neurons_number, 
        PerceptronLayer::ActivationFunction::RectifiedLinear);

    Tensor<type, 2> synaptic_weights(
        perceptron_layer_input_numbers,
        neurons_number);
    synaptic_weights(0, 0) = type(1);
    synaptic_weights(0, 1) = type(2);
    synaptic_weights(1, 0) = type(3);
    synaptic_weights(1, 1) = type(4);
    synaptic_weights(2, 0) = type(5);
    synaptic_weights(2, 1) = type(6);
    synaptic_weights(3, 0) = type(7);
    synaptic_weights(3, 1) = type(8);
    perceptron_layer.set_synaptic_weights(synaptic_weights);
    
    PerceptronLayerBackPropagation perceptron_layer_back_propagation(numb_of_input_images, &perceptron_layer);
    TensorMap<Tensor<type, 2>> perceptron_layer_delta(
        perceptron_layer_back_propagation.deltas_data,
        t1d2array<2>(perceptron_layer_back_propagation.deltas_dimensions));
    perceptron_layer_delta(0, 0) = type(2);
    perceptron_layer_delta(0, 1) = type(4);
    perceptron_layer_delta(1, 0) = type(8);
    perceptron_layer_delta(1, 1) = type(16);
    perceptron_layer_delta(2, 0) = type(32);
    perceptron_layer_delta(2, 1) = type(64);

    PerceptronLayerForwardPropagation perceptron_layer_forward_propagation(numb_of_input_images, &perceptron_layer);
    perceptron_layer_forward_propagation.activations_derivatives.setConstant(type(0.5));

    FlattenLayerBackPropagation flatten_layer_back_propagation(numb_of_input_images, &flatten_layer);
    
    flatten_layer.calculate_hidden_delta(
        static_cast<LayerForwardPropagation*>(&perceptron_layer_forward_propagation),
        static_cast<LayerBackPropagation*>(&perceptron_layer_back_propagation),
        static_cast<LayerBackPropagation*>(&flatten_layer_back_propagation)
    );
    
    //test
    Tensor<type, 2> expected_output(3, 4);
    //image 1
    expected_output(0, 0) = type(5);
    expected_output(0, 1) = type(11);
    expected_output(0, 2) = type(17);
    expected_output(0, 3) = type(23);
    //image 2
    expected_output(1, 0) = type(20);
    expected_output(1, 1) = type(44);
    expected_output(1, 2) = type(68);
    expected_output(1, 3) = type(92);
    //image 3
    expected_output(2, 0) = type(80);
    expected_output(2, 1) = type(176);
    expected_output(2, 2) = type(272);
    expected_output(2, 3) = type(368);

    TensorMap<Tensor<type, 2>> output(
        flatten_layer_back_propagation.deltas_data,
        t1d2array<2>(flatten_layer_back_propagation.deltas_dimensions));

    assert_true(is_equal<2>(expected_output, output), LOG);
}

void FlattenPerceptronLayerTest::run_test_case()
{
   cout << "Running flatten pooling layer test case...\n";

   test_flatten_perceptron_forward_propagate();
   test_flatten_perceptron_backward_pass();
   
   cout << "End of flatten pooling layer test case.\n\n";
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
