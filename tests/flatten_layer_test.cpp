//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   F L A T T E N   L A Y E R   T E S T   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "flatten_layer_test.h"
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

FlattenLayerTest::FlattenLayerTest() : UnitTesting()
{
}


FlattenLayerTest::~FlattenLayerTest()
{
}


void FlattenLayerTest::test_constructor()
{
    cout << "test_constructor\n";

    const Index row_inputs_number = 4;
    const Index column_inputs_number = 4;
    const Index channel_inputs_number = 2;
    const Index image_inputs_number = 2;

    Tensor<Index, 1> input_dimension(4);
    input_dimension.setValues({
        row_inputs_number,
        column_inputs_number,
        channel_inputs_number,
        image_inputs_number
    });

    FlattenLayer flatten_layer(input_dimension);

    assert_true(
        flatten_layer.get_input_height() == row_inputs_number &&
        flatten_layer.get_input_width() == column_inputs_number &&
        flatten_layer.get_inputs_channels_number() == channel_inputs_number &&
        flatten_layer.get_inputs_batch_number() == image_inputs_number, LOG);
}

void FlattenLayerTest::test_destructor()
{
   cout << "test_destructor\n";

   FlattenLayer* flatten_layer_1 = new FlattenLayer;

   delete flatten_layer_1;

}

void FlattenLayerTest::test_output_dimension()
{
    cout << "test_output_dimension\n";

    const Index row_inputs_number = 4;
    const Index column_inputs_number = 4;
    const Index channel_inputs_number = 2;
    const Index image_inputs_number = 2;

    Tensor<Index, 1> input_dimension(4);
    input_dimension.setValues({
        row_inputs_number,
        column_inputs_number,
        channel_inputs_number,
        image_inputs_number
    });

    FlattenLayer flatten_layer(input_dimension);
    const Tensor<Index, 1> output_dimension = flatten_layer.get_outputs_dimensions();

    Tensor<Index, 1> expected_output_dimension(2);
    expected_output_dimension.setValues({2, 32});

    assert_true(output_dimension(0) == expected_output_dimension(0) &&
                output_dimension(1) == expected_output_dimension(1), LOG);
}

void FlattenLayerTest::test_calculate_outputs()
{
    cout << "test_calculate_outputs\n";

    const Index row_inputs_number = 2;
    const Index column_inputs_number = 2;
    const Index channel_inputs_number = 2;
    const Index image_inputs_number = 2;

    Tensor<Index, 1> input_dimension(4);
    input_dimension.setValues({
        row_inputs_number,
        column_inputs_number,
        channel_inputs_number,
        image_inputs_number
    });

    FlattenLayer flatten_layer(input_dimension);

    Tensor<type, 4> inputs(t1d2array<4>(input_dimension));
    inputs(0, 0, 0, 0) = type(1);
    inputs(0, 0, 1, 0) = type(2);
    inputs(0, 1, 0, 0) = type(3);
    inputs(0, 1, 1, 0) = type(4);
    inputs(1, 0, 0, 0) = type(5);
    inputs(1, 0, 1, 0) = type(6);
    inputs(1, 1, 0, 0) = type(7);
    inputs(1, 1, 1, 0) = type(8);
    
    inputs(0, 0, 0, 1) = type(9);
    inputs(0, 0, 1, 1) = type(10);
    inputs(0, 1, 0, 1) = type(11);
    inputs(0, 1, 1, 1) = type(12);
    inputs(1, 0, 0, 1) = type(13);
    inputs(1, 0, 1, 1) = type(14);
    inputs(1, 1, 0, 1) = type(15);
    inputs(1, 1, 1, 1) = type(16);

    Tensor<Index, 1> output_dimension(2);
    output_dimension.setValues({image_inputs_number, 
        row_inputs_number * 
        column_inputs_number *
        channel_inputs_number});

    Tensor<type, 2> outputs(t1d2array<2>(output_dimension));
    
    flatten_layer.calculate_outputs(
        inputs.data(), 
        input_dimension, 
        outputs.data(),
        output_dimension);
    
    Tensor<type, 2> expected_output(2, 8);

    expected_output(0, 0) = type(1);
    expected_output(0, 1) = type(2);
    expected_output(0, 2) = type(3);
    expected_output(0, 3) = type(4);
    expected_output(0, 4) = type(5);
    expected_output(0, 5) = type(6);
    expected_output(0, 6) = type(7);
    expected_output(0, 7) = type(8);
    
    expected_output(1, 0) = type(9);
    expected_output(1, 1) = type(10);
    expected_output(1, 2) = type(11);
    expected_output(1, 3) = type(12);
    expected_output(1, 4) = type(13);
    expected_output(1, 5) = type(14);
    expected_output(1, 6) = type(15);
    expected_output(1, 7) = type(16);

    assert_true(
        is_equal<2>(expected_output, outputs), LOG);
}

void FlattenLayerTest::test_forward_propagate()
{    
    cout << "test_calculate_forward_propagate\n";

    const Index image_height = 2;
    const Index image_width = 2;
    const Index image_channels_number= 2;
    const Index images_number = 2;
    const Index pixels_number = image_height * image_width * image_channels_number;
    bool switch_train = true;

    Tensor<type, 4> inputs(image_height, image_width, image_channels_number, images_number);
    inputs(0, 0, 0, 0) = type(1);
    inputs(0, 0, 1, 0) = type(2);
    inputs(0, 1, 0, 0) = type(3);
    inputs(0, 1, 1, 0) = type(4);
    inputs(1, 0, 0, 0) = type(5);
    inputs(1, 0, 1, 0) = type(6);
    inputs(1, 1, 0, 0) = type(7);
    inputs(1, 1, 1, 0) = type(8);

    inputs(0, 0, 0, 1) = type(9);
    inputs(0, 0, 1, 1) = type(10);
    inputs(0, 1, 0, 1) = type(11);
    inputs(0, 1, 1, 1) = type(12);
    inputs(1, 0, 0, 1) = type(13);
    inputs(1, 0, 1, 1) = type(14);
    inputs(1, 1, 0, 1) = type(15);
    inputs(1, 1, 1, 1) = type(16);

    Tensor<Index, 1> inputs_dimensions(4);
    inputs_dimensions(0) = image_height;
    inputs_dimensions(1) = image_width;
    inputs_dimensions(2) = image_channels_number;
    inputs_dimensions(3) = images_number;

    FlattenLayer flatten_layer(inputs_dimensions);

    Tensor<type, 2> outputs;

    FlattenLayerForwardPropagation flatten_layer_forward_propagation(images_number, &flatten_layer);

    flatten_layer.forward_propagate(inputs.data(), inputs_dimensions, &flatten_layer_forward_propagation, switch_train);

    outputs = TensorMap<Tensor<type, 2>>(flatten_layer_forward_propagation.outputs_data,
                                         flatten_layer_forward_propagation.outputs_dimensions(0),
                                         flatten_layer_forward_propagation.outputs_dimensions(1));

    Tensor<type, 2> expected_output(2, 8);

    expected_output(0, 0) = type(1);
    expected_output(0, 1) = type(2);
    expected_output(0, 2) = type(3);
    expected_output(0, 3) = type(4);
    expected_output(0, 4) = type(5);
    expected_output(0, 5) = type(6);
    expected_output(0, 6) = type(7);
    expected_output(0, 7) = type(8);
    
    expected_output(1, 0) = type(9);
    expected_output(1, 1) = type(10);
    expected_output(1, 2) = type(11);
    expected_output(1, 3) = type(12);
    expected_output(1, 4) = type(13);
    expected_output(1, 5) = type(14);
    expected_output(1, 6) = type(15);
    expected_output(1, 7) = type(16);

    // Test
    assert_true(inputs.size() == outputs.size(), LOG);
    assert_true(
        is_equal<2>(expected_output, outputs), LOG);   
}

void FlattenLayerTest::test_calculate_hidden_delta()
{
    cout << "test_calculate_hidden_delta\n";
    const Index row_inputs_number = 2;
    const Index column_inputs_number = 2;
    const Index channel_inputs_number = 2;
    const Index image_inputs_number = 2;
    const Index input_pixel_numbers = 
        row_inputs_number * 
        column_inputs_number * 
        channel_inputs_number;

    Tensor<Index, 1> input_dimension(4);
    input_dimension.setValues({
        row_inputs_number,
        column_inputs_number,
        channel_inputs_number,
        image_inputs_number
    });

    FlattenLayer flatten_layer(input_dimension);

    LayerBackPropagation back_propagation;
    back_propagation.deltas_dimensions.resize(4);
    back_propagation.deltas_dimensions.setValues({
        row_inputs_number,
        column_inputs_number,
        channel_inputs_number,
        image_inputs_number});
    back_propagation.deltas_data = static_cast<type*>(malloc(
        row_inputs_number *
        column_inputs_number *
        channel_inputs_number *
        image_inputs_number * sizeof(type)));

    FlattenLayerForwardPropagation next_layer_forward_propagation(image_inputs_number, &flatten_layer);

    FlattenLayerBackPropagation next_layer_back_propagation(image_inputs_number, &flatten_layer);
    
    TensorMap<Tensor<type, 2>> next_deltas(
        next_layer_back_propagation.deltas_data, 
        image_inputs_number, 
        input_pixel_numbers);

    next_deltas(0, 0) = type(1);
    next_deltas(0, 1) = type(2);
    next_deltas(0, 2) = type(3);
    next_deltas(0, 3) = type(4);
    next_deltas(0, 4) = type(5);
    next_deltas(0, 5) = type(6);
    next_deltas(0, 6) = type(7);
    next_deltas(0, 7) = type(8);
    
    next_deltas(1, 0) = type(9);
    next_deltas(1, 1) = type(10);
    next_deltas(1, 2) = type(11);
    next_deltas(1, 3) = type(12);
    next_deltas(1, 4) = type(13);
    next_deltas(1, 5) = type(14);
    next_deltas(1, 6) = type(15);
    next_deltas(1, 7) = type(16);

    flatten_layer.calculate_hidden_delta(
        static_cast<LayerForwardPropagation*>(&next_layer_forward_propagation), 
        static_cast<LayerBackPropagation*>(&next_layer_back_propagation),
        &back_propagation);

    Tensor<type, 4> expected_delta(
        row_inputs_number,
        column_inputs_number,
        channel_inputs_number,
        image_inputs_number
    );

    expected_delta(0, 0, 0, 0) = type(1);
    expected_delta(0, 0, 1, 0) = type(2);
    expected_delta(0, 1, 0, 0) = type(3);
    expected_delta(0, 1, 1, 0) = type(4);
    expected_delta(1, 0, 0, 0) = type(5);
    expected_delta(1, 0, 1, 0) = type(6);
    expected_delta(1, 1, 0, 0) = type(7);
    expected_delta(1, 1, 1, 0) = type(8);
    
    expected_delta(0, 0, 0, 1) = type(9);
    expected_delta(0, 0, 1, 1) = type(10);
    expected_delta(0, 1, 0, 1) = type(11);
    expected_delta(0, 1, 1, 1) = type(12);
    expected_delta(1, 0, 0, 1) = type(13);
    expected_delta(1, 0, 1, 1) = type(14);
    expected_delta(1, 1, 0, 1) = type(15);
    expected_delta(1, 1, 1, 1) = type(16);

    TensorMap<Tensor<type, 4>> delta(
        back_propagation.deltas_data, 
        t1d2array<4>(back_propagation.deltas_dimensions));

    //test
    assert_true(
        back_propagation.deltas_dimensions(0) == row_inputs_number &&
        back_propagation.deltas_dimensions(1) == column_inputs_number &&
        back_propagation.deltas_dimensions(2) == channel_inputs_number &&
        back_propagation.deltas_dimensions(3) == image_inputs_number, LOG);

    assert_true(is_equal<4>(expected_delta, delta), LOG);
}

void FlattenLayerTest::run_test_case()
{
   cout << "Running flatten layer test case...\n";

   // Constructor and destructor

    test_constructor();
    test_destructor();
    test_output_dimension();
    test_calculate_outputs();
    test_calculate_hidden_delta();
    // Outputs

    test_forward_propagate();

   cout << "End of flatten layer test case.\n\n";
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
