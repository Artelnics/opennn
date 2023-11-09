//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   P O O L I N G   L A Y E R   T E S T   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "pooling_layer_test.h"


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

PoolingLayerTest::PoolingLayerTest() : UnitTesting()
{
}


PoolingLayerTest::~PoolingLayerTest()
{
}


void PoolingLayerTest::test_constructor()
{
    cout << "test_constructor\n";
    Tensor<Index, 1> inputs_dimension(4);

    const Index numb_of_rows = 3;
    const Index numb_of_cols = 3;
    const Index numb_of_channels = 3;
    const Index numb_of_images = 1;
    inputs_dimension.setValues({numb_of_rows, numb_of_cols, numb_of_channels, numb_of_images});

    PoolingLayer pooling_layer(inputs_dimension);

    assert_true(pooling_layer.get_inputs_rows_number() == numb_of_rows &&
                pooling_layer.get_inputs_columns_number() == numb_of_cols &&
                pooling_layer.get_inputs_channels_number() == numb_of_channels &&
                pooling_layer.get_inputs_images_number() == numb_of_images, LOG);

}


void PoolingLayerTest::test_constructor1()
{
    cout << "test_constructor1\n";
    Tensor<Index, 1> inputs_dimension(4);

    const Index numb_of_rows = 3;
    const Index numb_of_cols = 3;
    const Index numb_of_channels = 3;
    const Index numb_of_images = 1;
    inputs_dimension.setValues({numb_of_rows, numb_of_cols, numb_of_channels, numb_of_images});

    const Index numb_of_pooling_rows = 2;
    const Index numb_of_pooling_cols = 3;

    Tensor<Index, 1> pooling_dimension(2);
    pooling_dimension.setValues({numb_of_pooling_rows, numb_of_pooling_cols});

    PoolingLayer pooling_layer(inputs_dimension, pooling_dimension);

    assert_true(pooling_layer.get_inputs_rows_number() == numb_of_rows &&
                pooling_layer.get_inputs_columns_number() == numb_of_cols &&
                pooling_layer.get_inputs_channels_number() == numb_of_channels &&
                pooling_layer.get_inputs_images_number() == numb_of_images, LOG);


    assert_true(pooling_layer.get_pool_rows_number() == numb_of_pooling_rows &&
                pooling_layer.get_pool_columns_number() == numb_of_pooling_cols, LOG);

}

void PoolingLayerTest::test_destructor()
{
   cout << "test_destructor\n";
   PoolingLayer* pooling_layer = new PoolingLayer();
   delete pooling_layer;

}


void PoolingLayerTest::test_set_column_stride()
{
    cout << "test_set_column_stride\n";

    PoolingLayer pooling_layer;

    const Index column_stride = 2;
    pooling_layer.set_column_stride(column_stride);

    assert_true(pooling_layer.get_column_stride() == column_stride, LOG);
}

void PoolingLayerTest::test_set_row_stride()
{
    cout << "test_set_row_stride\n";

    PoolingLayer pooling_layer;

    const Index row_stride = 2;
    pooling_layer.set_row_stride(row_stride);

    assert_true(pooling_layer.get_row_stride() == row_stride, LOG);
}

void PoolingLayerTest::test_set_pooling_method()
{
    cout << "test_set_pooling_method\n";

    PoolingLayer pooling_layer;

    PoolingLayer::PoolingMethod pooling_method = PoolingLayer::PoolingMethod::MaxPooling;

    pooling_layer.set_pooling_method(pooling_method);

    assert_true(pooling_layer.get_pooling_method() == pooling_method, LOG);
}

void PoolingLayerTest::test_output_dimension()
{
    cout << "test_output_dimension\n";
    Tensor<Index, 1> inputs_dimension(4);

    const Index numb_of_rows = 4;
    const Index numb_of_cols = 4;
    const Index numb_of_channels = 3;
    const Index numb_of_images = 1;
    inputs_dimension.setValues({numb_of_rows, numb_of_cols, numb_of_channels, numb_of_images});

    const Index numb_of_pooling_rows = 2;
    const Index numb_of_pooling_cols = 2;

    Tensor<Index, 1> pooling_dimension(2);
    pooling_dimension.setValues({numb_of_pooling_rows, numb_of_pooling_cols});

    PoolingLayer pooling_layer(inputs_dimension, pooling_dimension);

    PoolingLayer::PoolingMethod pooling_method = PoolingLayer::PoolingMethod::MaxPooling;

    pooling_layer.set_pooling_method(pooling_method);

    const Index row_stride = 2;
    const Index column_stride = 2;

    pooling_layer.set_row_stride(row_stride);
    pooling_layer.set_column_stride(column_stride);

    const Index expected_output_rows_number = (numb_of_rows - numb_of_pooling_rows + row_stride) / row_stride;
    const Index expected_output_columns_number = (numb_of_rows - numb_of_pooling_cols + column_stride) / column_stride;

    assert_true(pooling_layer.get_outputs_rows_number() == expected_output_rows_number &&
                pooling_layer.get_outputs_columns_number() == expected_output_columns_number, LOG);

    assert_true(pooling_layer.get_outputs_dimensions()(0) == expected_output_rows_number &&
                pooling_layer.get_outputs_dimensions()(1) == expected_output_columns_number &&
                pooling_layer.get_outputs_dimensions()(2) == numb_of_channels &&
                pooling_layer.get_outputs_dimensions()(3) == numb_of_images, LOG);
}

void PoolingLayerTest::test_calculate_no_pooling_outputs()
{
    cout << "test_calculate_no_pooling_outputs\n";
    Tensor<Index, 1> inputs_dimension(4);

    const Index numb_of_rows = 4;
    const Index numb_of_cols = 4;
    const Index numb_of_channels = 2;
    const Index numb_of_images = 2;
    inputs_dimension.setValues({numb_of_rows, numb_of_cols, numb_of_channels, numb_of_images});

    const Index numb_of_pooling_rows = 2;
    const Index numb_of_pooling_cols = 2;
    
    Tensor<Index, 1> pooling_dimension(2);
    pooling_dimension.setValues({numb_of_pooling_rows, numb_of_pooling_cols});

    PoolingLayer pooling_layer(inputs_dimension, pooling_dimension);

    Tensor<type, 4> inputs(t1d2array<4>(inputs_dimension));

    inputs.chip(0, 1).setConstant(type(1));
    inputs.chip(1, 1).setConstant(type(2));
    inputs.chip(2, 1).setConstant(type(3));
    inputs.chip(3, 1).setConstant(type(4));

    Tensor<type, 4> output = pooling_layer.calculate_no_pooling_outputs(inputs);

    assert_true(is_equal<4>(inputs, output), LOG);
}

void PoolingLayerTest::test_calculate_average_pooling_outputs()
{
    cout << "test_calculate_average_pooling_outputs\n";
    Tensor<Index, 1> inputs_dimension(4);

    const Index numb_of_rows = 4;
    const Index numb_of_cols = 4;
    const Index numb_of_channels = 2;
    const Index numb_of_images = 2;
    inputs_dimension.setValues({numb_of_rows, numb_of_cols, numb_of_channels, numb_of_images});

    const Index numb_of_pooling_rows = 2;
    const Index numb_of_pooling_cols = 2;
    
    Tensor<Index, 1> pooling_dimension(2);
    pooling_dimension.setValues({numb_of_pooling_rows, numb_of_pooling_cols});

    PoolingLayer pooling_layer(inputs_dimension, pooling_dimension);
    
    const Index row_stride = 2;
    const Index column_stride = 2;
    
    pooling_layer.set_row_stride(row_stride);
    pooling_layer.set_column_stride(column_stride);

    Tensor<type, 4> inputs(t1d2array<4>(inputs_dimension));

    inputs.chip(0, 0).setConstant(type(1));
    inputs.chip(1, 0).setConstant(type(2));
    inputs.chip(2, 0).setConstant(type(3));
    inputs.chip(3, 0).setConstant(type(4));

    Tensor<type, 4> outputs = pooling_layer.calculate_average_pooling_outputs(inputs);
    assert_true(outputs.dimension(0) == 2 && 
                outputs.dimension(1) == 2 &&
                outputs.dimension(2) == 2 &&
                outputs.dimension(3) == 2, LOG);

    Tensor<type, 4> expected_outputs(2, 2, 2, 2);

    expected_outputs.chip(0, 0).setConstant(type(3) / type(2));
    expected_outputs.chip(1, 0).setConstant(type(7) / type(2));

    assert_true(is_equal<4>(expected_outputs, outputs), LOG);


//

//    Tensor<type, 2> inputs;
//    Tensor<type, 2> outputs;

    // Test

//    inputs.resize(({6,6,6,6}));

//    pooling_layer.set_pool_size(1,1);
//    pooling_layer.set_row_stride(1);
//    pooling_layer.set_column_stride(1);

//    outputs = pooling_layer.calculate_average_pooling_outputs(inputs);

//    assert_true(outputs.dimension(0) == 6 &&
//                outputs.dimension(1) == 6 &&
//                outputs.dimension(2) == 6 &&
//                outputs.dimension(3) == 6, LOG);

    // Test

//    inputs.resize(({10,3,20,20}));

//    pooling_layer.set_pool_size(2,2);
//    pooling_layer.set_row_stride(1);
//    pooling_layer.set_column_stride(1);

//    outputs = pooling_layer.calculate_average_pooling_outputs(inputs);

//    assert_true(outputs.dimension(0) == 10 &&
//                outputs.dimension(1) == 3 &&
//                outputs.dimension(2) == 19 &&
//                outputs.dimension(3) == 19, LOG);

    // Test

//    inputs.resize(({1,1,4,4}));
//    inputs(0,0,0,0) = type(1);
//    inputs(0,0,0,1) = 2.0;
//    inputs(0,0,0,2) = 3.0;
//    inputs(0,0,0,3) = 4.0;
//    inputs(0,0,1,0) = 16.0;
//    inputs(0,0,1,1) = 9.0;
//    inputs(0,0,1,2) = 4.0;
//    inputs(0,0,1,3) = type(1);
//    inputs(0,0,2,0) = type(1);
//    inputs(0,0,2,1) = 8.0;
//    inputs(0,0,2,2) = 27.0;
//    inputs(0,0,2,3) = 64.0;
//    inputs(0,0,3,0) = 256.0;
//    inputs(0,0,3,1) = 81.0;
//    inputs(0,0,3,2) = 16.0;
//    inputs(0,0,3,3) = type(1);

//    pooling_layer.set_pool_size(2, 2);
//    pooling_layer.set_row_stride(1);
//    pooling_layer.set_column_stride(1);

//    outputs = pooling_layer.calculate_average_pooling_outputs(inputs);

//    assert_true(outputs.dimension(0) == 1 &&
//                outputs.dimension(1) == 1 &&
//                outputs.dimension(2) == 3 &&
//                outputs.dimension(3) == 3 &&
//                outputs(0,0,0,0) == 7.0 &&
//                outputs(0,0,0,1) == 4.5 &&
//                outputs(0,0,0,2) == 3.0 &&
//                outputs(0,0,1,0) == 8.5 &&
//                outputs(0,0,1,1) == 12.0 &&
//                outputs(0,0,1,2) == 24.0 &&
//                outputs(0,0,2,0) == 86.5 &&
//                outputs(0,0,2,1) == 33.0 &&
//                outputs(0,0,2,2) == 27.0, LOG);

    // Test

//    inputs.resize(({1,1,4,4}));
//    inputs(0,0,0,0) = type(1);
//    inputs(0,0,0,1) = 2.0;
//    inputs(0,0,0,2) = 3.0;
//    inputs(0,0,0,3) = 4.0;
//    inputs(0,0,1,0) = 16.0;
//    inputs(0,0,1,1) = 9.0;
//    inputs(0,0,1,2) = 4.0;
//    inputs(0,0,1,3) = type(1);
//    inputs(0,0,2,0) = type(1);
//    inputs(0,0,2,1) = 8.0;
//    inputs(0,0,2,2) = 27.0;
//    inputs(0,0,2,3) = 64.0;
//    inputs(0,0,3,0) = 256.0;
//    inputs(0,0,3,1) = 81.0;
//    inputs(0,0,3,2) = 16.0;
//    inputs(0,0,3,3) = type(1);

//    pooling_layer.set_pool_size(3, 3);
//    pooling_layer.set_row_stride(1);
//    pooling_layer.set_column_stride(1);

//    outputs = pooling_layer.calculate_average_pooling_outputs(inputs);

//    assert_true(outputs.dimension(0) == 1 &&
//                outputs.dimension(1) == 1 &&
//                outputs.dimension(2) == 2 &&
//                outputs.dimension(3) == 2 &&
//                outputs(0,0,0,0) - 7.8888 < 0.001 &&
//                outputs(0,0,0,1) - 13.5555 < 0.001 &&
//                outputs(0,0,1,0) - 46.4444 < 0.001 &&
//                outputs(0,0,1,1) - 23.4444 < 0.001, LOG);
}

void PoolingLayerTest::test_calculate_max_pooling_outputs()
{
    cout << "test_calculate_max_pooling_outputs\n";
    Tensor<Index, 1> inputs_dimension(4);

    const Index numb_of_rows = 4;
    const Index numb_of_cols = 4;
    const Index numb_of_channels = 2;
    const Index numb_of_images = 2;
    inputs_dimension.setValues({numb_of_rows, numb_of_cols, numb_of_channels, numb_of_images});

    const Index numb_of_pooling_rows = 2;
    const Index numb_of_pooling_cols = 2;
    
    Tensor<Index, 1> pooling_dimension(2);
    pooling_dimension.setValues({numb_of_pooling_rows, numb_of_pooling_cols});

    PoolingLayer pooling_layer(inputs_dimension, pooling_dimension);

    const Index row_stride = 2;
    const Index column_stride = 2;
    
    pooling_layer.set_row_stride(row_stride);
    pooling_layer.set_column_stride(column_stride);

    Tensor<type, 4> inputs(t1d2array<4>(inputs_dimension));

    inputs.chip(0, 0).setConstant(type(1));
    inputs.chip(1, 0).setConstant(type(2));
    inputs.chip(2, 0).setConstant(type(3));
    inputs.chip(3, 0).setConstant(type(4));

    Tensor<type, 4> outputs = pooling_layer.calculate_max_pooling_outputs(inputs);

    assert_true(outputs.dimension(0) == 2 && 
                outputs.dimension(1) == 2 &&
                outputs.dimension(2) == 2 &&
                outputs.dimension(3) == 2, LOG);

    Tensor<type, 4> expected_outputs(2, 2, 2, 2);

    expected_outputs.chip(0, 0).setConstant(2);
    expected_outputs.chip(1, 0).setConstant(4);
 
    assert_true(is_equal<4>(expected_outputs, outputs), LOG);
    
    Tensor<tuple<Index, Index>, 4> switches(pooling_layer.get_outputs_rows_number(), pooling_layer.get_outputs_columns_number(), numb_of_channels, numb_of_images);

    outputs = pooling_layer.calculate_max_pooling_outputs(inputs, switches);

    assert_true(is_equal<4>(expected_outputs, outputs), LOG);

    Tensor<tuple<Index, Index>, 4> expected_switches(2, 2, numb_of_channels, numb_of_images);
    expected_switches(0, 0, 0, 0) = make_tuple(1, 0);
    expected_switches(0, 0, 1, 0) = make_tuple(1, 0);
    expected_switches(0, 1, 0, 0) = make_tuple(1, 2);
    expected_switches(0, 1, 1, 0) = make_tuple(1, 2);
    expected_switches(1, 0, 0, 0) = make_tuple(3, 0);
    expected_switches(1, 0, 1, 0) = make_tuple(3, 0);
    expected_switches(1, 1, 0, 0) = make_tuple(3, 2);
    expected_switches(1, 1, 1, 0) = make_tuple(3, 2);

    expected_switches.chip(1, 3) = expected_switches.chip(0, 3);

    Tensor<bool, 0> is_switches_same = (switches == expected_switches).all();
    assert_true(is_switches_same(), LOG);
//

//    Tensor<type, 2> inputs;
//    Tensor<type, 2> outputs;

    // Test

//    inputs.resize(({6,6,6,6}));

//    pooling_layer.set_pool_size(1,1);
//    pooling_layer.set_row_stride(1);
//    pooling_layer.set_column_stride(1);

//    outputs = pooling_layer.calculate_max_pooling_outputs(inputs);

//    assert_true(outputs.dimension(0) == 6 &&
//                outputs.dimension(1) == 6 &&
//                outputs.dimension(2) == 6 &&
//                outputs.dimension(3) == 6, LOG);

    // Test

//    inputs.resize(({10,3,20,20}));

//    pooling_layer.set_pool_size(2,2);
//    pooling_layer.set_row_stride(1);
//    pooling_layer.set_column_stride(1);

//    outputs = pooling_layer.calculate_max_pooling_outputs(inputs);

//    assert_true(outputs.dimension(0) == 10 &&
//                outputs.dimension(1) == 3 &&
//                outputs.dimension(2) == 19 &&
//                outputs.dimension(3) == 19, LOG);

    // Test

//    inputs.resize(({1,1,4,4}));
//    inputs(0,0,0,0) = type(1);
//    inputs(0,0,0,1) = 2.0;
//    inputs(0,0,0,2) = 3.0;
//    inputs(0,0,0,3) = 4.0;
//    inputs(0,0,1,0) = 16.0;
//    inputs(0,0,1,1) = 9.0;
//    inputs(0,0,1,2) = 4.0;
//    inputs(0,0,1,3) = type(1);
//    inputs(0,0,2,0) = type(1);
//    inputs(0,0,2,1) = 8.0;
//    inputs(0,0,2,2) = 27.0;
//    inputs(0,0,2,3) = 64.0;
//    inputs(0,0,3,0) = 256.0;
//    inputs(0,0,3,1) = 81.0;
//    inputs(0,0,3,2) = 16.0;
//    inputs(0,0,3,3) = type(1);

//    pooling_layer.set_pool_size(2, 2);
//    pooling_layer.set_row_stride(1);
//    pooling_layer.set_column_stride(1);

//    outputs = pooling_layer.calculate_max_pooling_outputs(inputs);

//    assert_true(outputs.dimension(0) == 1 &&
//                outputs.dimension(1) == 1 &&
//                outputs.dimension(2) == 3 &&
//                outputs.dimension(3) == 3 &&
//                outputs(0,0,0,0) == 16.0 &&
//                outputs(0,0,0,1) == 9.0 &&
//                outputs(0,0,0,2) == 4.0 &&
//                outputs(0,0,1,0) == 16.0 &&
//                outputs(0,0,1,1) == 27.0 &&
//                outputs(0,0,1,2) == 64.0 &&
//                outputs(0,0,2,0) == 256.0 &&
//                outputs(0,0,2,1) == 81.0 &&
//                outputs(0,0,2,2) == 64.0, LOG);

    // Test

//    inputs.resize(({1,1,4,4}));
//    inputs(0,0,0,0) = type(1);
//    inputs(0,0,0,1) = 2.0;
//    inputs(0,0,0,2) = 3.0;
//    inputs(0,0,0,3) = 4.0;
//    inputs(0,0,1,0) = -16.0;
//    inputs(0,0,1,1) = -9.0;
//    inputs(0,0,1,2) = -4.0;
//    inputs(0,0,1,3) = -1.0;
//    inputs(0,0,2,0) = type(1);
//    inputs(0,0,2,1) = 8.0;
//    inputs(0,0,2,2) = 27.0;
//    inputs(0,0,2,3) = 64.0;
//    inputs(0,0,3,0) = -256.0;
//    inputs(0,0,3,1) = -81.0;
//    inputs(0,0,3,2) = -16.0;
//    inputs(0,0,3,3) = -1.0;

//    pooling_layer.set_pool_size(3, 3);
//    pooling_layer.set_row_stride(1);
//    pooling_layer.set_column_stride(1);

//    outputs = pooling_layer.calculate_max_pooling_outputs(inputs);

//    assert_true(outputs.dimension(0) == 1 &&
//                outputs.dimension(1) == 1 &&
//                outputs.dimension(2) == 2 &&
//                outputs.dimension(3) == 2 &&
//                outputs(0,0,0,0) == 27.0 &&
//                outputs(0,0,0,1) == 64.0 &&
//                outputs(0,0,1,0) == 27.0 &&
//                outputs(0,0,1,1) == 64.0, LOG);
}

void PoolingLayerTest::test_forward_propagate()
{
    cout << "test_forward_propagate\n";

    Tensor<Index, 1> inputs_dimension(4);

    const Index numb_of_rows = 4;
    const Index numb_of_cols = 4;
    const Index numb_of_channels = 2;
    const Index numb_of_images = 2;
    inputs_dimension.setValues({numb_of_rows, numb_of_cols, numb_of_channels, numb_of_images});

    const Index numb_of_pooling_rows = 2;
    const Index numb_of_pooling_cols = 2;
    
    Tensor<Index, 1> pooling_dimension(2);
    pooling_dimension.setValues({numb_of_pooling_rows, numb_of_pooling_cols});

    PoolingLayer pooling_layer(inputs_dimension, pooling_dimension);

    const Index row_stride = 2;
    const Index column_stride = 2;
    
    pooling_layer.set_row_stride(row_stride);
    pooling_layer.set_column_stride(column_stride);

    Tensor<type, 4> inputs(t1d2array<4>(inputs_dimension));

    inputs.chip(0, 0).setConstant(type(1));
    inputs.chip(1, 0).setConstant(type(2));
    inputs.chip(2, 0).setConstant(type(3));
    inputs.chip(3, 0).setConstant(type(4));

    PoolingLayer::PoolingMethod pooling_method = PoolingLayer::PoolingMethod::AveragePooling;

    pooling_layer.set_pooling_method(pooling_method);

    PoolingLayerForwardPropagation layer_forward_propagation(numb_of_images, &pooling_layer);
    bool switch_train = true;

    pooling_layer.forward_propagate(inputs.data(), inputs_dimension, &layer_forward_propagation, switch_train);

    Tensor<type, 4> expected_outputs(2, 2, 2, 2);

    expected_outputs.chip(0, 0).setConstant(type(3) / type(2));
    expected_outputs.chip(1, 0).setConstant(type(7) / type(2));

    TensorMap<Tensor<type, 4>> outputs(layer_forward_propagation.outputs_data, 
                                        t1d2array<4>(layer_forward_propagation.outputs_dimensions));

    assert_true(is_equal<4>(expected_outputs, outputs), LOG);

    pooling_layer.set_pooling_method(PoolingLayer::PoolingMethod::MaxPooling);

    pooling_layer.forward_propagate(inputs.data(), inputs_dimension, &layer_forward_propagation, switch_train);

    expected_outputs.chip(0,0).setConstant(type(2));
    expected_outputs.chip(1,0).setConstant(type(4));

    assert_true(is_equal<4>(expected_outputs, outputs), LOG);

    Tensor<tuple<Index, Index>, 4> expected_switches(2, 2, numb_of_channels, numb_of_images);
    expected_switches(0, 0, 0, 0) = make_tuple(1, 0);
    expected_switches(0, 0, 1, 0) = make_tuple(1, 0);
    expected_switches(0, 1, 0, 0) = make_tuple(1, 2);
    expected_switches(0, 1, 1, 0) = make_tuple(1, 2);
    expected_switches(1, 0, 0, 0) = make_tuple(3, 0);
    expected_switches(1, 0, 1, 0) = make_tuple(3, 0);
    expected_switches(1, 1, 0, 0) = make_tuple(3, 2);
    expected_switches(1, 1, 1, 0) = make_tuple(3, 2);

    expected_switches.chip(1, 3) = expected_switches.chip(0, 3);

    Tensor<bool, 0> is_switches_same = (layer_forward_propagation.switches == expected_switches).all();
    assert_true(is_switches_same(), LOG);
}

void PoolingLayerTest::test_calculate_hidden_delta_average_pooling()
{
    cout << "test_calculate_hidden_delta_average_pooling\n";

    Tensor<Index, 1> inputs_dimension(4);

    const Index numb_of_rows = 4;
    const Index numb_of_cols = 4;
    const Index numb_of_channels = 2;
    const Index numb_of_images = 2;
    inputs_dimension.setValues({numb_of_rows, numb_of_cols, numb_of_channels, numb_of_images});

    const Index numb_of_pooling_rows = 1;
    const Index numb_of_pooling_cols = 1;
    
    Tensor<Index, 1> pooling_dimension(2);
    pooling_dimension.setValues({numb_of_pooling_rows, numb_of_pooling_cols});

    PoolingLayer pooling_layer(inputs_dimension, pooling_dimension);

    const Index row_stride = 1;
    const Index column_stride = 1;
    
    pooling_layer.set_row_stride(row_stride);
    pooling_layer.set_column_stride(column_stride);

    Tensor<type, 4> inputs(t1d2array<4>(inputs_dimension));

    inputs.chip(0, 0).setConstant(type(1));
    inputs.chip(1, 0).setConstant(type(2));
    inputs.chip(2, 0).setConstant(type(3));
    inputs.chip(3, 0).setConstant(type(4));

    PoolingLayer::PoolingMethod pooling_method = PoolingLayer::PoolingMethod::NoPooling;

    pooling_layer.set_pooling_method(pooling_method);

    PoolingLayer next_pooling_layer(pooling_layer.get_outputs_dimensions());
    next_pooling_layer.set_column_stride(2);
    next_pooling_layer.set_row_stride(2);
    next_pooling_layer.set_pooling_method(PoolingLayer::PoolingMethod::AveragePooling);
    
    PoolingLayerBackPropagation next_back_propagation(numb_of_images, &next_pooling_layer);
    PoolingLayerForwardPropagation next_forward_propagation(numb_of_images, &next_pooling_layer);

    TensorMap<Tensor<type, 4>> next_delta(next_back_propagation.deltas_data, t1d2array<4>(next_back_propagation.deltas_dimensions));

    next_delta(0, 0, 0, 0) = type(3);
    next_delta(0, 0, 1, 0) = type(4);
    next_delta(0, 1, 0, 0) = type(-2);
    next_delta(0, 1, 1, 0) = type(2);
    next_delta(1, 0, 0, 0) = type(5);
    next_delta(1, 0, 1, 0) = type(2);
    next_delta(1, 1, 0, 0) = type(1);
    next_delta(1, 1, 1, 0) = type(3);

    next_delta(0, 0, 0, 1) = type(5);
    next_delta(0, 0, 1, 1) = type(2);
    next_delta(0, 1, 0, 1) = type(1);
    next_delta(0, 1, 1, 1) = type(3);
    next_delta(1, 0, 0, 1) = type(3);
    next_delta(1, 0, 1, 1) = type(4);
    next_delta(1, 1, 0, 1) = type(-2);
    next_delta(1, 1, 1, 1) = type(2);

    PoolingLayerBackPropagation back_propagation(
        numb_of_images, &pooling_layer);

    pooling_layer.calculate_hidden_delta(
        static_cast<PoolingLayerForwardPropagation*>(&next_forward_propagation), 
        static_cast<PoolingLayerBackPropagation*>(&next_back_propagation), 
        static_cast<PoolingLayerBackPropagation*>(&back_propagation));

    //Test
    TensorMap<Tensor<type, 4>> delta(back_propagation.deltas_data, t1d2array<4>(back_propagation.deltas_dimensions));

    Tensor<type, 4> expected_delta(4, 4, 2, 2);

    expected_delta(0, 0, 0, 0) = type(0.75);
    expected_delta(0, 0, 1, 0) = type(1);
    expected_delta(0, 1, 0, 0) = type(0.75);
    expected_delta(0, 1, 1, 0) = type(1);
    expected_delta(1, 0, 0, 0) = type(0.75);
    expected_delta(1, 0, 1, 0) = type(1);
    expected_delta(1, 1, 0, 0) = type(0.75);
    expected_delta(1, 1, 1, 0) = type(1);

    expected_delta(0, 2, 0, 0) = type(-0.5);
    expected_delta(0, 2, 1, 0) = type(0.5);
    expected_delta(0, 3, 0, 0) = type(-0.5);
    expected_delta(0, 3, 1, 0) = type(0.5);
    expected_delta(1, 2, 0, 0) = type(-0.5);
    expected_delta(1, 2, 1, 0) = type(0.5);
    expected_delta(1, 3, 0, 0) = type(-0.5);
    expected_delta(1, 3, 1, 0) = type(0.5);

    expected_delta(2, 0, 0, 0) = type(1.25);
    expected_delta(2, 0, 1, 0) = type(0.5);
    expected_delta(2, 1, 0, 0) = type(1.25);
    expected_delta(2, 1, 1, 0) = type(0.5);
    expected_delta(3, 0, 0, 0) = type(1.25);
    expected_delta(3, 0, 1, 0) = type(0.5);
    expected_delta(3, 1, 0, 0) = type(1.25);
    expected_delta(3, 1, 1, 0) = type(0.5);
    
    expected_delta(2, 2, 0, 0) = type(0.25);
    expected_delta(2, 2, 1, 0) = type(0.75);
    expected_delta(2, 3, 0, 0) = type(0.25);
    expected_delta(2, 3, 1, 0) = type(0.75);
    expected_delta(3, 2, 0, 0) = type(0.25);
    expected_delta(3, 2, 1, 0) = type(0.75);
    expected_delta(3, 3, 0, 0) = type(0.25);
    expected_delta(3, 3, 1, 0) = type(0.75);

    expected_delta.chip(1, 3).chip(0, 0) = expected_delta.chip(0, 3).chip(2, 0);
    expected_delta.chip(1, 3).chip(1, 0) = expected_delta.chip(0, 3).chip(2, 0);

    expected_delta.chip(1, 3).chip(2,0) = expected_delta.chip(0, 3).chip(0, 0);
    expected_delta.chip(1, 3).chip(3,0) = expected_delta.chip(0, 3).chip(0, 0);

    assert_true(is_equal<4>(expected_delta, delta), LOG);
}

void PoolingLayerTest::test_calculate_hidden_delta_max_pooling()
{
    cout << "test_calculate_hidden_delta_max_pooling\n";

    Tensor<Index, 1> inputs_dimension(4);

    const Index numb_of_rows = 4;
    const Index numb_of_cols = 4;
    const Index numb_of_channels = 2;
    const Index numb_of_images = 2;
    inputs_dimension.setValues({numb_of_rows, numb_of_cols, numb_of_channels, numb_of_images});

    const Index numb_of_pooling_rows = 1;
    const Index numb_of_pooling_cols = 1;
    
    Tensor<Index, 1> pooling_dimension(2);
    pooling_dimension.setValues({numb_of_pooling_rows, numb_of_pooling_cols});

    PoolingLayer pooling_layer(inputs_dimension, pooling_dimension);

    PoolingLayer::PoolingMethod pooling_method = PoolingLayer::PoolingMethod::NoPooling;

    pooling_layer.set_row_stride(1);
    pooling_layer.set_column_stride(1);
    
    pooling_layer.set_pooling_method(pooling_method);

    Tensor<Index, 1> next_pooling_dimension(2);
    next_pooling_dimension.setValues({2, 2});

    PoolingLayer next_pooling_layer(pooling_layer.get_outputs_dimensions(), next_pooling_dimension);

    next_pooling_layer.set_pooling_method(PoolingLayer::PoolingMethod::MaxPooling);

    const Index next_layer_row_stride = 2;
    const Index next_layer_column_stride = 2;
    next_pooling_layer.set_column_stride(next_layer_row_stride);
    next_pooling_layer.set_row_stride(next_layer_column_stride);
    
    PoolingLayerBackPropagation next_back_propagation(numb_of_images, &next_pooling_layer);
    PoolingLayerForwardPropagation next_forward_propagation(numb_of_images, &next_pooling_layer);
    next_forward_propagation.switches.resize(2, 2, 2, 2);
    next_forward_propagation.switches(0, 0, 0, 0) = make_tuple(1, 0);
    next_forward_propagation.switches(0, 0, 1, 0) = make_tuple(1, 0);
    next_forward_propagation.switches(0, 1, 0, 0) = make_tuple(1, 2);
    next_forward_propagation.switches(0, 1, 1, 0) = make_tuple(1, 2);
    next_forward_propagation.switches(1, 0, 0, 0) = make_tuple(3, 0);
    next_forward_propagation.switches(1, 0, 1, 0) = make_tuple(3, 0);
    next_forward_propagation.switches(1, 1, 0, 0) = make_tuple(3, 2);
    next_forward_propagation.switches(1, 1, 1, 0) = make_tuple(3, 2);
    next_forward_propagation.switches.chip(1, 3) = next_forward_propagation.switches.chip(0, 3);

    TensorMap<Tensor<type, 4>> next_delta(next_back_propagation.deltas_data, t1d2array<4>(next_back_propagation.deltas_dimensions));

    next_delta(0, 0, 0, 0) = type(3);
    next_delta(0, 0, 1, 0) = type(4);
    next_delta(0, 1, 0, 0) = type(-2);
    next_delta(0, 1, 1, 0) = type(2);
    next_delta(1, 0, 0, 0) = type(5);
    next_delta(1, 0, 1, 0) = type(2);
    next_delta(1, 1, 0, 0) = type(1);
    next_delta(1, 1, 1, 0) = type(3);

    next_delta(0, 0, 0, 1) = type(5);
    next_delta(0, 0, 1, 1) = type(2);
    next_delta(0, 1, 0, 1) = type(1);
    next_delta(0, 1, 1, 1) = type(3);
    next_delta(1, 0, 0, 1) = type(3);
    next_delta(1, 0, 1, 1) = type(4);
    next_delta(1, 1, 0, 1) = type(-2);
    next_delta(1, 1, 1, 1) = type(2);

    PoolingLayerBackPropagation back_propagation(
        numb_of_images, &pooling_layer);
    
    pooling_layer.calculate_hidden_delta(
        static_cast<LayerForwardPropagation*>(&next_forward_propagation), 
        static_cast<LayerBackPropagation*>(&next_back_propagation), 
        static_cast<LayerBackPropagation*>(&back_propagation));

    TensorMap<Tensor<type, 4>> delta(back_propagation.deltas_data, t1d2array<4>(back_propagation.deltas_dimensions));

    Tensor<type, 4> expected_delta(4, 4, 2, 2);

    expected_delta.setZero();

    expected_delta(1, 0, 0, 0) = type(3);
    expected_delta(1, 0, 1, 0) = type(4);
    expected_delta(1, 2, 0, 0) = type(-2);
    expected_delta(1, 2, 1, 0) = type(2);
    expected_delta(3, 0, 0, 0) = type(5);
    expected_delta(3, 0, 1, 0) = type(2);
    expected_delta(3, 2, 0, 0) = type(1);
    expected_delta(3, 2, 1, 0) = type(3);

    expected_delta(1, 0, 0, 1) = type(5);
    expected_delta(1, 0, 1, 1) = type(2);
    expected_delta(1, 2, 0, 1) = type(1);
    expected_delta(1, 2, 1, 1) = type(3);
    expected_delta(3, 0, 0, 1) = type(3);
    expected_delta(3, 0, 1, 1) = type(4);
    expected_delta(3, 2, 0, 1) = type(-2);
    expected_delta(3, 2, 1, 1) = type(2);

    assert_true(is_equal<4>(expected_delta, delta), LOG);
}

void PoolingLayerTest::run_test_case()
{
   cout << "Running pooling layer test case...\n";

   // Constructor and destructor

    test_constructor();
    test_constructor1();
    test_destructor();

    //Setters
    test_set_pooling_method();
    test_set_column_stride();
    test_set_row_stride();

    //Output
    test_output_dimension();

    // Calculating
    test_calculate_average_pooling_outputs();
    test_calculate_max_pooling_outputs();
    test_calculate_no_pooling_outputs();
    test_forward_propagate();

    //Backpropagation
    test_calculate_hidden_delta_max_pooling();
    test_calculate_hidden_delta_average_pooling();

   cout << "End of pooling layer test case.\n\n";
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
