//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   C O N V O L U T I O N A L   L A Y E R   T E S T   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "convolutional_layer_test.h"

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

ConvolutionalLayerTest::ConvolutionalLayerTest() : UnitTesting()
{
}


ConvolutionalLayerTest::~ConvolutionalLayerTest()
{
}


void ConvolutionalLayerTest::test_set()
{
    cout << "test_set\n";
}


void ConvolutionalLayerTest::test_eigen_convolution()
{
    cout << "test_eigen_convolution\n";

    // Convolution 2D, 1 channel
    Tensor<type, 2> input(3, 3);
    Tensor<type, 2> kernel(2, 2);
    Tensor<type, 2> output;
    input.setRandom();
    kernel.setRandom();

    Eigen::array<ptrdiff_t, 2> dims = {0, 1};

    output = input.convolve(kernel, dims);

    assert_true(output.dimension(0) == 2, LOG);
    assert_true(output.dimension(1) == 2, LOG);

    // Convolution 2D, 3 channels
    Tensor<type, 3> input_2(5, 5, 3);
    Tensor<type, 3> kernel_2(2, 2, 3);
    Tensor<type, 3> output_2;
    input_2.setRandom();
    kernel_2.setRandom();

    Eigen::array<ptrdiff_t, 3> dims_2 = {0, 1, 2};

    output_2 = input_2.convolve(kernel_2, dims_2);

    assert_true(output_2.dimension(0) == 4, LOG);
    assert_true(output_2.dimension(1) == 4, LOG);
    assert_true(output_2.dimension(2) == 1, LOG);


    // Convolution 2D, 3 channels, multiple images, 1 kernel
    Tensor<type, 4> input_3(10, 3, 5, 5);
    Tensor<type, 3> kernel_3(3, 2, 2);
    Tensor<type, 4> output_3;
    input_3.setConstant(type(1));
    input_3.chip(1, 0).setConstant(type(2));
    input_3.chip(2, 0).setConstant(type(3));

    kernel_3.setConstant(type(1.0/12.0));

    Eigen::array<ptrdiff_t, 3> dims_3 = {1, 2, 3};

    output_3 = input_3.convolve(kernel_3, dims_3);

    assert_true(output_3.dimension(0) == 10, LOG);
    assert_true(output_3.dimension(1) == 1, LOG);
    assert_true(output_3.dimension(2) == 4, LOG);
    assert_true(output_3.dimension(3) == 4, LOG);

    assert_true(abs(output_3(0, 0, 0, 0) - type(1)) < type(NUMERIC_LIMITS_MIN) &&
                abs(output_3(0, 0, 0, 1) - type(1)) < type(NUMERIC_LIMITS_MIN) &&
                abs(output_3(0, 0, 0, 2) - type(1)) < type(NUMERIC_LIMITS_MIN) &&
                abs(output_3(0, 0, 0, 3) - type(1)) < type(NUMERIC_LIMITS_MIN) &&
                abs(output_3(0, 0, 1, 0) - type(1)) < type(NUMERIC_LIMITS_MIN) &&
                abs(output_3(0, 0, 1, 1) - type(1)) < type(NUMERIC_LIMITS_MIN) &&
                abs(output_3(0, 0, 1, 2) - type(1)) < type(NUMERIC_LIMITS_MIN) &&
                abs(output_3(0, 0, 1, 3) - type(1)) < type(NUMERIC_LIMITS_MIN) &&
                abs(output_3(0, 0, 2, 0) - type(1)) < type(NUMERIC_LIMITS_MIN) &&
                abs(output_3(0, 0, 2, 1) - type(1)) < type(NUMERIC_LIMITS_MIN) &&
                abs(output_3(0, 0, 2, 2) - type(1)) < type(NUMERIC_LIMITS_MIN) &&
                abs(output_3(0, 0, 2, 3) - type(1)) < type(NUMERIC_LIMITS_MIN) &&
                abs(output_3(0, 0, 3, 0) - type(1)) < type(NUMERIC_LIMITS_MIN) &&
                abs(output_3(0, 0, 3, 1) - type(1)) < type(NUMERIC_LIMITS_MIN) &&
                abs(output_3(0, 0, 3, 2) - type(1)) < type(NUMERIC_LIMITS_MIN) &&
                abs(output_3(0, 0, 3, 3) - type(1)) < type(NUMERIC_LIMITS_MIN) &&
                abs(output_3(1, 0, 0, 0) - type(2)) < type(NUMERIC_LIMITS_MIN) &&
                abs(output_3(1, 0, 0, 1) - type(2)) < type(NUMERIC_LIMITS_MIN) &&
                abs(output_3(1, 0, 0, 2) - type(2)) < type(NUMERIC_LIMITS_MIN) &&
                abs(output_3(1, 0, 0, 3) - type(2)) < type(NUMERIC_LIMITS_MIN) &&
                abs(output_3(1, 0, 1, 0) - type(2)) < type(NUMERIC_LIMITS_MIN) &&
                abs(output_3(1, 0, 1, 1) - type(2)) < type(NUMERIC_LIMITS_MIN) &&
                abs(output_3(1, 0, 1, 2) - type(2)) < type(NUMERIC_LIMITS_MIN) &&
                abs(output_3(1, 0, 1, 3) - type(2)) < type(NUMERIC_LIMITS_MIN) &&
                abs(output_3(1, 0, 2, 0) - type(2)) < type(NUMERIC_LIMITS_MIN) &&
                abs(output_3(1, 0, 2, 1) - type(2)) < type(NUMERIC_LIMITS_MIN) &&
                abs(output_3(1, 0, 2, 2) - type(2)) < type(NUMERIC_LIMITS_MIN) &&
                abs(output_3(1, 0, 2, 3) - type(2)) < type(NUMERIC_LIMITS_MIN) &&
                abs(output_3(1, 0, 3, 0) - type(2)) < type(NUMERIC_LIMITS_MIN) &&
                abs(output_3(1, 0, 3, 1) - type(2)) < type(NUMERIC_LIMITS_MIN) &&
                abs(output_3(1, 0, 3, 2) - type(2)) < type(NUMERIC_LIMITS_MIN) &&
                abs(output_3(1, 0, 3, 3) - type(2)) < type(NUMERIC_LIMITS_MIN) &&
                abs(output_3(2, 0, 0, 0) - type(3)) < type(NUMERIC_LIMITS_MIN) &&
                abs(output_3(2, 0, 0, 1) - type(3)) < type(NUMERIC_LIMITS_MIN) &&
                abs(output_3(2, 0, 0, 2) - type(3)) < type(NUMERIC_LIMITS_MIN) &&
                abs(output_3(2, 0, 0, 3) - type(3)) < type(NUMERIC_LIMITS_MIN) &&
                abs(output_3(2, 0, 1, 0) - type(3)) < type(NUMERIC_LIMITS_MIN) &&
                abs(output_3(2, 0, 1, 1) - type(3)) < type(NUMERIC_LIMITS_MIN) &&
                abs(output_3(2, 0, 1, 2) - type(3)) < type(NUMERIC_LIMITS_MIN) &&
                abs(output_3(2, 0, 1, 3) - type(3)) < type(NUMERIC_LIMITS_MIN) &&
                abs(output_3(2, 0, 2, 0) - type(3)) < type(NUMERIC_LIMITS_MIN) &&
                abs(output_3(2, 0, 2, 1) - type(3)) < type(NUMERIC_LIMITS_MIN) &&
                abs(output_3(2, 0, 2, 2) - type(3)) < type(NUMERIC_LIMITS_MIN) &&
                abs(output_3(2, 0, 2, 3) - type(3)) < type(NUMERIC_LIMITS_MIN) &&
                abs(output_3(2, 0, 3, 0) - type(3)) < type(NUMERIC_LIMITS_MIN) &&
                abs(output_3(2, 0, 3, 1) - type(3)) < type(NUMERIC_LIMITS_MIN) &&
                abs(output_3(2, 0, 3, 2) - type(3)) < type(NUMERIC_LIMITS_MIN) &&
                abs(output_3(2, 0, 3, 3) - type(3)) <= type(NUMERIC_LIMITS_MIN), LOG);

}

void ConvolutionalLayerTest::test_eigen_convolution_3d()
{
    cout << "test_eigen_3d_convolution\n";

    // Convolution 3D, 2 channels

    Tensor<type, 3> input(3, 3, 2);
    Tensor<type, 3> kernel(2, 2, 2);
    Tensor<type, 3> output(2, 2, 1);

    for(int i=0;i<3*3*2;i++) *(input.data() + i) = i;
    for(int i=0;i<2*2*2;i++) *(kernel.data() + i) = i+1;

    Eigen::array<ptrdiff_t, 3> dims = {0,1,2};

    output = input.convolve(kernel, dims);

    assert_true(fabs(output(0,0,0) - 320)<type(NUMERIC_LIMITS_MIN), LOG);
    assert_true(fabs(output(1,0,0) - 356)<type(NUMERIC_LIMITS_MIN), LOG);
    assert_true(fabs(output(0,1,0) - 428)<type(NUMERIC_LIMITS_MIN), LOG);
    assert_true(fabs(output(1,1,0) - 464)<type(NUMERIC_LIMITS_MIN), LOG);
}


void ConvolutionalLayerTest::test_read_bmp()
{
    DataSet data_set;

    data_set.set_data_file_name("C:/Users/alvaromartin/Documents/Dataset for read_bmp()/");

    data_set.read_bmp();

    Tensor<type, 2> data = data_set.get_data();
}


void ConvolutionalLayerTest::test_constructor()
{
    cout << "test_constructor\n";

    Tensor<Index, 1> new_inputs_dimensions(4);
    Tensor<Index, 1> new_kernels_dimensions(4);

    new_inputs_dimensions.setValues({23, 64, 3, 1});
    new_kernels_dimensions.setValues({3, 2, 1, 1});

    ConvolutionalLayer convolutional_layer(new_inputs_dimensions, new_kernels_dimensions);

    assert_true(convolutional_layer.get_inputs_channels_number() == 3 &&
                convolutional_layer.get_inputs_rows_number() == 23 &&
                convolutional_layer.get_inputs_columns_number() == 64 &&
                convolutional_layer.get_inputs_images_number() == 1, LOG);
    assert_true(convolutional_layer.get_kernels_channels_number() == 1 &&
                convolutional_layer.get_kernels_rows_number() == 3 &&
                convolutional_layer.get_kernels_columns_number() == 2 &&
                convolutional_layer.get_kernels_number() == 1, LOG);
}


void ConvolutionalLayerTest::test_destructor()
{
   cout << "test_destructor\n";
}


void ConvolutionalLayerTest::test_set_parameters()
{
    cout << "test_set_parameters\n";
    ConvolutionalLayer convolutional_layer;

    Tensor<type, 4> new_synaptic_weights(2, 2, 2, 2);
    Tensor<type, 1> new_biases(2);
    Tensor<type, 1> parameters(18);

    convolutional_layer.set_biases(new_biases);
    convolutional_layer.set_synaptic_weights(new_synaptic_weights);
    parameters(0) = new_biases(0) = type(100);
    parameters(1) = new_biases(1) = type(200);
    //First kernel
    parameters(2) = new_synaptic_weights(0,0,0,0) = type(1);
    parameters(3) = new_synaptic_weights(1,0,0,0) = type(2);
    parameters(4) = new_synaptic_weights(0,1,0,0) = type(3);
    parameters(5) = new_synaptic_weights(1,1,0,0) = type(4);
    parameters(6) = new_synaptic_weights(0,0,1,0) = type(5);
    parameters(7) = new_synaptic_weights(1,0,1,0) = type(6);
    parameters(8) = new_synaptic_weights(0,1,1,0) = type(7);
    parameters(9) = new_synaptic_weights(1,1,1,0) = type(8);
    //Second kernel
    parameters(10) = new_synaptic_weights(0,0,0,1) = type(9);
    parameters(11) = new_synaptic_weights(1,0,0,1) = type(10);
    parameters(12) = new_synaptic_weights(0,1,0,1) = type(11);
    parameters(13) = new_synaptic_weights(1,1,0,1) = type(12);
    parameters(14) = new_synaptic_weights(0,0,1,1) = type(13);
    parameters(15) = new_synaptic_weights(1,0,1,1) = type(14);
    parameters(16) = new_synaptic_weights(0,1,1,1) = type(15);
    parameters(17) = new_synaptic_weights(1,1,1,1) = type(16);

    convolutional_layer.set_parameters(parameters, 0);

    const Tensor<type, 4> synaptic_weight = convolutional_layer.get_synaptic_weights();
    const Tensor<type, 1> biases = convolutional_layer.get_biases();

    assert_true(is_equal<1>(biases, new_biases),LOG);

    assert_true(is_equal<4>(synaptic_weight, new_synaptic_weights), LOG);


//    assert_true(!convolutional_layer.is_empty() &&
//                convolutional_layer.get_parameters_number() == 18 &&
//                convolutional_layer.get_synaptic_weights() == new_synaptic_weights &&
//                convolutional_layer.get_biases() == new_biases, LOG);
}

void ConvolutionalLayerTest::test_calculate_convolutions()
{
    cout << "test_calculate_convolutions\n";

    Tensor<Index, 1> input_dimension(4);
    const Index number_of_input_rows = 5U;
    const Index number_of_input_columns = 5U;
    const Index number_of_input_channels = 3U;
    const Index number_of_input_images = 1U;
    input_dimension.setValues({number_of_input_rows, number_of_input_columns, number_of_input_channels, number_of_input_images});

    Tensor<Index, 1> kernel_dimension(4);
    const Index number_of_kernel_rows = 3U;
    const Index number_of_kernel_columns = 3U;
    const Index number_of_kernels = 3U;
    kernel_dimension.setValues({number_of_kernel_rows, number_of_kernel_columns, number_of_input_channels, number_of_kernels});

    ConvolutionalLayer convolution_layer(input_dimension, kernel_dimension);

    Tensor<type, 1> biases(3);
    biases.setValues({type(0), type(1), type(2)});
    convolution_layer.set_biases(biases);

    Tensor<type, 4> kernel(t1d2array<4>(kernel_dimension));
    kernel.setConstant(type(1)/type(9));
    convolution_layer.set_synaptic_weights(kernel);

    Tensor<type, 4> input(t1d2array<4>(input_dimension));
    input.setConstant(type(1));

    Tensor<type, 4> output(
        number_of_input_rows - number_of_kernel_rows + 1, 
        number_of_input_columns - number_of_kernel_columns + 1, 
        number_of_kernels,
        number_of_input_images);
    convolutional_layer.calculate_convolutions(input, output.data());

    Tensor<type, 4> expected_output(3, 3, 3, 1);
    expected_output.setConstant(type(3));
    expected_output.chip(1, 2).setConstant(type(4));
    expected_output.chip(2, 2).setConstant(type(5));

    assert_true(expected_output.dimension(0) == output.dimension(0) &&
    expected_output.dimension(1) == output.dimension(1) &&
    expected_output.dimension(2) == output.dimension(2) &&
    expected_output.dimension(3) == output.dimension(3), LOG);

    assert_true(is_equal<4>(expected_output, output), LOG);
}

void ConvolutionalLayerTest::test_calculate_combinations()
{
    cout << "test_calculate_combinations\n";

    const Index input_images = 1;
    const Index input_kernels = 3;

    const Index channels = 3;

    const Index rows_input = 5;
    const Index cols_input = 5;
    const Index rows_kernel = 3;
    const Index cols_kernel = 3;

    Tensor<type, 4> inputs(rows_input,cols_input,channels,input_images);
    Tensor<type, 4> kernels(rows_kernel,cols_kernel,channels,input_kernels);

    Tensor<type, 4> combinations((rows_input - rows_kernel) + 1,
                                 (cols_input - cols_kernel) + 1,
                                 input_kernels,
                                 input_images);

    Tensor<type, 1> biases(input_kernels);

    inputs.setConstant(type(1.));
    kernels.setConstant(type(1./12.));

    biases(0) = type(0.);
    biases(1) = type(1.);
    biases(2) = type(2.);

    Tensor<Index, 1> new_inputs_dimension(4);
    Tensor<Index, 1> new_kernels_dimensions(4);

    new_inputs_dimension.setValues({rows_input, cols_input, channels,input_images});
    new_kernels_dimensions.setValues({rows_kernel, cols_kernel,channels,input_kernels});

    ConvolutionalLayer convolutional_layer(new_inputs_dimension, new_kernels_dimensions);

    convolutional_layer.set_biases(biases);
    convolutional_layer.set_synaptic_weights(kernels);

    convolutional_layer.calculate_convolutions(inputs, combinations.data());
    
    Tensor<type, 4> expected_output(3, 3, 3, 1);
    expected_output.chip(0, 2).setConstant(type(2.25));
    expected_output.chip(1, 2).setConstant(type(3.25));
    expected_output.chip(2, 2).setConstant(type(4.25));

    assert_true(is_equal<4>(expected_output, combinations), LOG);

    inputs.resize(5, 5, 2, 2);
    kernels.resize(2, 2, 2, 2);
    combinations.resize(4, 4, 2, 2);
    biases.resize(2);

    inputs.chip(0, 3).setConstant(type(1.));
    inputs.chip(1, 3).setConstant(type(2.));
    kernels.chip(0, 3).setConstant(type(1./8.));
    kernels.chip(1, 3).setConstant(type(1./4.));

    biases(0) = type(0);
    biases(1) = type(1);

    convolutional_layer.set_biases(biases);
    convolutional_layer.set_synaptic_weights(kernels);
    convolutional_layer.calculate_convolutions(inputs, combinations.data());

    expected_output.resize(4, 4, 2, 2);
    expected_output.chip(0, 3).chip(0, 2).setConstant(type(1));
    expected_output.chip(0, 3).chip(1, 2).setConstant(type(3));
    expected_output.chip(1, 3).chip(0, 2).setConstant(type(2));
    expected_output.chip(1, 3).chip(1, 2).setConstant(type(5));

    assert_true(is_equal<4>(expected_output, combinations), LOG);         
}

///@todo include this in pooling

void ConvolutionalLayerTest::test_calculate_average_pooling_outputs()
{
    cout << "test_calculate_max_pooling_outputs\n";

    //input_dims
    const Index input_images = 1;
    const Index channels = 1;

    const Index rows_input = 4;
    const Index cols_input = 4;

    //pooling dims
    const Index rows_polling = 2;
    const Index cols_polling = 2;

    //stride
    const Index rows_stride=1;
    const Index cols_stride=1;

    //output dims
    const Index output_rows_number = (rows_input - rows_polling)/rows_stride + 1;
    const Index output_cols_number = (cols_input - cols_polling)/cols_stride +1;

    Tensor<type, 4> inputs(rows_input, cols_input, channels, input_images);
    Tensor<type, 4> outputs(output_rows_number, output_cols_number, channels, input_images);

    inputs.setRandom();

    //pooling average

    Index col = 0;
    Index row = 0;

    for(int i=0; i<input_images; i++)
    {
        for(int c=0; c<channels; c++)
        {
            for(int k=0; k<output_cols_number; k++)
            {
                for(int l=0; l<output_rows_number; l++)
                {
                    float tmp_result = 0;

                    for(int m=0; m<cols_polling; m++)
                    {
                        col = m*cols_stride + k;

                        for(int n=0; n<rows_polling; n++)
                        {
                            row = n*rows_stride + l;

                            tmp_result += inputs(row,col,c,i);
                        }
                    }
                    outputs(l,k,c,i) = tmp_result/(cols_polling*rows_polling);
                }
            }
        }
    }
}


void ConvolutionalLayerTest::test_calculate_max_pooling_outputs()
{
    cout << "test_calculate_max_pooling_outputs\n";

    //input_dims
    const Index input_images = 1;
    const Index channels = 1;

    const Index rows_input = 4;
    const Index cols_input = 4;

    //pooling dims
    const Index rows_polling = 2;
    const Index cols_polling = 2;

    //stride
    const Index rows_stride=1;
    const Index cols_stride=1;

    //output dims
    const Index output_rows_number = (rows_input - rows_polling)/rows_stride + 1;
    const Index output_cols_number = (cols_input - cols_polling)/cols_stride +1;

    Tensor<type, 4> inputs(rows_input, cols_input, channels, input_images);
    Tensor<type, 4> outputs(output_rows_number, output_cols_number, channels, input_images);

    inputs.setRandom();

    //pooling average

    Index col = 0;
    Index row = 0;

    for(int i=0; i<input_images; i++)
    {
        for(int c=0; c<channels; c++)
        {
            for(int k=0; k<output_cols_number; k++)
            {
                for(int l=0; l<output_rows_number; l++)
                {
                    float tmp_result = 0;

                    float final_result = 0;

                    for(int m=0; m<cols_polling; m++)
                    {
                        col = m*cols_stride + k;

                        for(int n=0; n<rows_polling; n++)
                        {
                            row = n*rows_stride + l;

                            tmp_result = inputs(row,col,c,i);

                            if(tmp_result > final_result) final_result = tmp_result;
                        }
                    }
                    outputs(l,k,c,i) = final_result;
                }
            }
        }
    }
}




void ConvolutionalLayerTest::test_calculate_activations()
{
    cout << "test_calculate_activations\n";

    Tensor<type, 4> inputs;
    Tensor<type, 4> activations_4d;
    Tensor<type, 4> result;

    result.resize(2,2,2,2);
    inputs.resize(2,2,2,2);

    Tensor<Index, 1> new_inputs_dimensions(4);
    Tensor<Index, 1> new_kernels_dimensions(4);

    new_inputs_dimensions.setValues({2, 2, 2, 2});
    new_kernels_dimensions.setValues({2, 2, 2, 2});

    ConvolutionalLayer convolutional_layer(new_inputs_dimensions, new_kernels_dimensions);

    // Test

    inputs(0,0,0,0) = type(type(-1.111f));
    inputs(0,0,0,1) = type(type(-1.112));
    inputs(0,0,1,0) = type(type(-1.121));
    inputs(0,0,1,1) = type(-1.122f);
    inputs(0,1,0,0) = type(1.211f);
    inputs(0,1,0,1) = type(1.212f);
    inputs(0,1,1,0) = type(1.221f);
    inputs(0,1,1,1) = type(1.222f);
    inputs(1,0,0,0) = type(-2.111f);
    inputs(1,0,0,1) = type(-2.112f);
    inputs(1,0,1,0) = type(-2.121f);
    inputs(1,0,1,1) = type(-2.122f);
    inputs(1,1,0,0) = type(2.211f);
    inputs(1,1,0,1) = type(2.212f);
    inputs(1,1,1,0) = type(2.221f);
    inputs(1,1,1,1) = type(2.222f);

    activations_4d.resize(new_inputs_dimensions);
    convolutional_layer.set_activation_function(opennn::ConvolutionalLayer::ActivationFunction::Threshold);
    convolutional_layer.calculate_activations(inputs.data(), new_inputs_dimensions, activations_4d.data(), new_inputs_dimensions);

    Tensor<type, 4> expected_output(t1d2array<4>(new_inputs_dimensions));
    expected_output = inputs.unaryExpr([](const type& val){
        return val < type(0) ? type(0) : type(1);
    });

    assert_true(is_equal<4>(expected_output, activations_4d), LOG);


    convolutional_layer.set_activation_function(opennn::ConvolutionalLayer::ActivationFunction::SymmetricThreshold);

    convolutional_layer.calculate_activations(inputs.data(), new_inputs_dimensions, activations_4d.data(), new_inputs_dimensions);

    expected_output = inputs.unaryExpr([](const type& val){
        return val < type(0) ? type(-1) : type(1);
    });

    //    assert_true(activations == result, LOG);
    assert_true(is_equal<4>(expected_output, activations_4d), LOG);

    // Test

    convolutional_layer.set_activation_function(opennn::ConvolutionalLayer::ActivationFunction::HyperbolicTangent);
    convolutional_layer.calculate_activations(inputs.data(), new_inputs_dimensions, activations_4d.data(), new_inputs_dimensions);

    expected_output = inputs.tanh();

    assert_true(is_equal<4>(expected_output, activations_4d), LOG);

    // Test

    convolutional_layer.set_activation_function(opennn::ConvolutionalLayer::ActivationFunction::RectifiedLinear);
    convolutional_layer.calculate_activations(inputs.data(), new_inputs_dimensions, activations_4d.data(), new_inputs_dimensions);

    expected_output = inputs.unaryExpr([](const type& val){
        return val < type(0) ? type(0) : val;
    });

    assert_true(is_equal<4>(expected_output, activations_4d), LOG);
}


void ConvolutionalLayerTest::test_calculate_activations_derivatives()
{
    cout << "test_calculate_activations_derivatives\n";
    Tensor<Index, 1> dimension(4);
    dimension.setValues({2, 2, 2, 2});
    Tensor<type, 4> inputs(t1d2array<4>(dimension));
    Tensor<type, 4> activations_derivatives(t1d2array<4>(dimension));
    Tensor<type, 4> activations(t1d2array<4>(dimension));
    Tensor<type, 4> result(t1d2array<4>(dimension));

    ConvolutionalLayer convolutional_layer;

    // Test

    inputs(0,0,0,0) = type(-1.111f);
    inputs(0,0,0,1) = type(-1.112);
    inputs(0,0,1,0) = type(-1.121);
    inputs(0,0,1,1) = type(-1.122f);
    inputs(0,1,0,0) = type(1.211f);
    inputs(0,1,0,1) = type(1.212f);
    inputs(0,1,1,0) = type(1.221f);
    inputs(0,1,1,1) = type(1.222f);
    inputs(1,0,0,0) = type(-2.111f);
    inputs(1,0,0,1) = type(-2.112f);
    inputs(1,0,1,0) = type(-2.121f);
    inputs(1,0,1,1) = type(-2.122f);
    inputs(1,1,0,0) = type(2.211f);
    inputs(1,1,0,1) = type(2.212f);
    inputs(1,1,1,0) = type(2.221f);
    inputs(1,1,1,1) = type(2.222f);

    convolutional_layer.set_activation_function(opennn::ConvolutionalLayer::ActivationFunction::Threshold);

    activations = inputs.unaryExpr([](const type& val){
        return val < type(0) ? type(0) : type(1);
    });

    convolutional_layer.calculate_activations_derivatives(inputs.data(), dimension, activations.data(), dimension, activations_derivatives.data(), dimension);

    result = activations.constant(type(0));

    assert_true(is_equal<4>(result, activations_derivatives), LOG);

    inputs(0,0,0,0) = type(-1.111f);
    inputs(0,0,0,1) = type(-1.112);
    inputs(0,0,1,0) = type(-1.121);
    inputs(0,0,1,1) = type(-1.122f);
    inputs(0,1,0,0) = type(1.211f);
    inputs(0,1,0,1) = type(1.212f);
    inputs(0,1,1,0) = type(1.221f);
    inputs(0,1,1,1) = type(1.222f);
    inputs(1,0,0,0) = type(-2.111f);
    inputs(1,0,0,1) = type(-2.112f);
    inputs(1,0,1,0) = type(-2.121f);
    inputs(1,0,1,1) = type(-2.122f);
    inputs(1,1,0,0) = type(2.211f);
    inputs(1,1,0,1) = type(2.212f);
    inputs(1,1,1,0) = type(2.221f);
    inputs(1,1,1,1) = type(2.222f);

    convolutional_layer.set_activation_function(opennn::ConvolutionalLayer::ActivationFunction::SymmetricThreshold);

    activations = inputs.unaryExpr([](const type& val){
        return val < type(0) ? type(-1) : type(1);
    });

    convolutional_layer.calculate_activations_derivatives(inputs.data(), dimension, activations.data(), dimension, activations_derivatives.data(), dimension);

    result = activations.constant(type(0));

    assert_true(is_equal<4>(result, activations_derivatives), LOG);

    convolutional_layer.set_activation_function(opennn::ConvolutionalLayer::ActivationFunction::HyperbolicTangent);

    activations = inputs.tanh();

    convolutional_layer.calculate_activations_derivatives(inputs.data(), dimension, activations.data(), dimension, activations_derivatives.data(), dimension);

    result = type(4) / ((type(-1) * inputs).exp() + inputs.exp()).square();

    //result(0,0,0,0) = type(0.352916f);
    //result(0,0,0,1) = type(0.352348f);
    //result(0,0,1,0) = type(0.347271f);
    //result(0,0,1,1) = type(0.346710f);
    //result(0,1,0,0) = type(0.299466f);
    //result(0,1,0,1) = type(0.298965f);
    //result(0,1,1,0) = type(0.294486f);
    //result(0,1,1,1) = type(0.293991f);
    //result(1,0,0,0) = type(0.056993f);
    //result(1,0,0,1) = type(0.056882f);
    //result(1,0,1,0) = type(0.055896f);
    //result(1,0,1,1) = type(0.055788f);
    //result(1,1,0,0) = type(0.046907f);
    //result(1,1,0,1) = type(0.046816f);
    //result(1,1,1,0) = type(0.046000f);
    //result(1,1,1,1) = type(0.045910f);

    assert_true(is_equal<4>(result, activations_derivatives), LOG);

    inputs(0,0,0,0) = type(-1.111f);
    inputs(0,0,0,1) = type(-1.112);
    inputs(0,0,1,0) = type(-1.121);
    inputs(0,0,1,1) = type(-1.122f);
    inputs(0,1,0,0) = type(1.211f);
    inputs(0,1,0,1) = type(1.212f);
    inputs(0,1,1,0) = type(1.221f);
    inputs(0,1,1,1) = type(1.222f);
    inputs(1,0,0,0) = type(-2.111f);
    inputs(1,0,0,1) = type(-2.112f);
    inputs(1,0,1,0) = type(-2.121f);
    inputs(1,0,1,1) = type(-2.122f);
    inputs(1,1,0,0) = type(2.211f);
    inputs(1,1,0,1) = type(2.212f);
    inputs(1,1,1,0) = type(2.221f);
    inputs(1,1,1,1) = type(2.222f);

    convolutional_layer.set_activation_function(opennn::ConvolutionalLayer::ActivationFunction::RectifiedLinear);

    activations = inputs.unaryExpr([](const type& val){
        return val < type(0) ? 0 : val;
    });

    convolutional_layer.calculate_activations_derivatives(inputs.data(), dimension, activations.data(), dimension, activations_derivatives.data(), dimension);

    result = inputs.unaryExpr([](const type& val){
        return val < type(0) ? 0 : type(1);
    });

    assert_true(is_equal<4>(result, activations_derivatives), LOG);

    convolutional_layer.set_activation_function(opennn::ConvolutionalLayer::ActivationFunction::SoftPlus);

    activations = (type(1) + inputs).log();
    convolutional_layer.calculate_activations_derivatives(inputs.data(), dimension, activations.data(), dimension, activations_derivatives.data(), dimension);

    result = inputs.exp() / (type(1) + inputs.exp());
    //result(0,0,0,0) = type(0.247685f);
    //result(0,0,0,1) = type(0.247498f);
    //result(0,0,1,0) = type(0.245826f);
    //result(0,0,1,1) = type(0.245640f);
    //result(0,1,0,0) = type(0.770476f);
    //result(0,1,0,1) = type(0.770653f);
    //result(0,1,1,0) = type(0.772239f);
    //result(0,1,1,1) = type(0.772415f);
    //result(1,0,0,0) = type(0.108032f);
    //result(1,0,0,1) = type(0.107936f);
    //result(1,0,1,0) = type(0.107072f);
    //result(1,0,1,1) = type(0.106977f);
    //result(1,1,0,0) = type(0.901233f);
    //result(1,1,0,1) = type(0.901322f);
    //result(1,1,1,0) = type(0.902120f);
    //result(1,1,1,1) = type(0.902208f);

    assert_true(is_equal<4>(result, activations_derivatives), LOG);
}


void ConvolutionalLayerTest::test_forward_propagate_training()
{
    cout << "test_forward_propagate_training\n";

    const Index input_images = 2;
    const Index input_kernels = 3;

    const Index channels = 3;

    const Index rows_input = 4;
    const Index cols_input = 4;
    const Index rows_kernel = 3;
    const Index cols_kernel = 3;

    Tensor<Index, 1> input_dimension(4);
    input_dimension.setValues({rows_input, cols_input, channels, input_images});
    Tensor<type,4> inputs(t1d2array<4>(input_dimension));

    Tensor<Index, 1> kernel_dimension(4);
    kernel_dimension.setValues({rows_kernel, cols_kernel, channels, input_kernels});
    Tensor<type,4> kernel(t1d2array<4>(kernel_dimension));

    Tensor<type,1> bias(input_kernels);
    bias.setValues({0,1,2});

    bool switch_train = true;

    inputs.setConstant(1.);
    inputs.chip(0,3).chip(0,2).setConstant(2.);
    inputs.chip(0,3).chip(1,2).setConstant(3.);
    inputs.chip(0,3).chip(2,2).setConstant(4.);

    kernel.chip(0,3).setConstant(type(1./3.));
    kernel.chip(1,3).setConstant(type(1./9.));
    kernel.chip(2,3).setConstant(type(1./27.));
    
    ConvolutionalLayer convolutional_layer(input_dimension, kernel_dimension);
    convolutional_layer.set(inputs, kernel, bias);

    ConvolutionalLayerForwardPropagation forward_propagation;
    forward_propagation.set(input_images, &convolutional_layer);

    convolutional_layer.set_activation_function(ConvolutionalLayer::ActivationFunction::HyperbolicTangent);

    convolutional_layer.forward_propagate(inputs.data(), input_dimension, &forward_propagation, switch_train);

    assert_true(forward_propagation.outputs.dimension(0) == convolutional_layer.get_outputs_dimensions()(0)&&
                forward_propagation.outputs.dimension(1) == convolutional_layer.get_outputs_dimensions()(1)&&
                forward_propagation.outputs.dimension(2) == convolutional_layer.get_outputs_dimensions()(2)&&
                forward_propagation.outputs.dimension(3) == convolutional_layer.get_outputs_dimensions()(3), LOG);
    
    assert_true(forward_propagation.combinations.dimension(0) == (rows_input - rows_kernel + 1) &&
                forward_propagation.combinations.dimension(1) == (cols_input - cols_kernel + 1) &&
                forward_propagation.combinations.dimension(2) == (input_kernels) &&
                forward_propagation.combinations.dimension(3) == (input_images), LOG);

    assert_true(forward_propagation.activations_derivatives.dimension(0) == convolutional_layer.get_outputs_dimensions()(0)&&
                forward_propagation.activations_derivatives.dimension(1) == convolutional_layer.get_outputs_dimensions()(1)&&
                forward_propagation.activations_derivatives.dimension(2) == convolutional_layer.get_outputs_dimensions()(2)&&
                forward_propagation.activations_derivatives.dimension(3) == convolutional_layer.get_outputs_dimensions()(3), LOG);


    Tensor<type, 4> expected_combination_output((rows_input - rows_kernel + 1), (cols_input - cols_kernel + 1), (input_kernels), input_images);
    expected_combination_output.chip(0, 3).chip(0, 2).setConstant(type(27));
    expected_combination_output.chip(0, 3).chip(1, 2).setConstant(type(10));
    expected_combination_output.chip(0, 3).chip(2, 2).setConstant(type(5));
    expected_combination_output.chip(1, 3).chip(0, 2).setConstant(type(9));
    expected_combination_output.chip(1, 3).chip(1, 2).setConstant(type(4));
    expected_combination_output.chip(1, 3).chip(2, 2).setConstant(type(3));

    assert_true(is_equal<4>(expected_combination_output, forward_propagation.combinations), LOG);

    Tensor<type, 4> expected_activation_output = expected_combination_output.tanh();

    assert_true(is_equal<4>(expected_activation_output, forward_propagation.outputs), LOG);

    Tensor<type, 4> expected_activation_derivatives = type(4) / ((type(-1) * expected_combination_output).exp() + expected_combination_output.exp()).square();

    assert_true(is_equal<4>(expected_activation_derivatives, forward_propagation.activations_derivatives), LOG);
}


void ConvolutionalLayerTest::test_forward_propagate_not_training()
{
    cout << "test_forward_propagate_not_training\n";

    const Index input_images = 2;
    const Index input_kernels = 3;

    const Index channels = 3;

    const Index rows_input = 4;
    const Index cols_input = 4;
    const Index rows_kernel = 3;
    const Index cols_kernel = 3;

    Tensor<Index, 1> input_dimension(4);
    input_dimension.setValues({rows_input, cols_input, channels, input_images});
    Tensor<type,4> inputs(t1d2array<4>(input_dimension));

    Tensor<Index, 1> kernel_dimension(4);
    kernel_dimension.setValues({rows_kernel, cols_kernel, channels, input_kernels});
    Tensor<type,4> kernel(t1d2array<4>(kernel_dimension));

    Tensor<type,1> bias(input_kernels);
    bias.setValues({0,1,2});

    bool switch_train = false;

    inputs.setConstant(1.);
    inputs.chip(0,3).chip(0,2).setConstant(2.);
    inputs.chip(0,3).chip(1,2).setConstant(3.);
    inputs.chip(0,3).chip(2,2).setConstant(4.);

    kernel.chip(0,3).setConstant(type(1./3.));
    kernel.chip(1,3).setConstant(type(1./9.));
    kernel.chip(2,3).setConstant(type(1./27.));
    
    ConvolutionalLayer convolutional_layer(input_dimension, kernel_dimension);
    convolutional_layer.set(inputs, kernel, bias);

    ConvolutionalLayerForwardPropagation forward_propagation;
    forward_propagation.set(input_images, &convolutional_layer);

    convolutional_layer.set_activation_function(ConvolutionalLayer::ActivationFunction::HyperbolicTangent);

    convolutional_layer.forward_propagate(inputs.data(), input_dimension, &forward_propagation, switch_train);

    assert_true(forward_propagation.outputs.dimension(0) == convolutional_layer.get_outputs_dimensions()(0)&&
                forward_propagation.outputs.dimension(1) == convolutional_layer.get_outputs_dimensions()(1)&&
                forward_propagation.outputs.dimension(2) == convolutional_layer.get_outputs_dimensions()(2)&&
                forward_propagation.outputs.dimension(3) == convolutional_layer.get_outputs_dimensions()(3), LOG);
    
    assert_true(forward_propagation.combinations.dimension(0) == (rows_input - rows_kernel + 1) &&
                forward_propagation.combinations.dimension(1) == (cols_input - cols_kernel + 1) &&
                forward_propagation.combinations.dimension(2) == (input_kernels) &&
                forward_propagation.combinations.dimension(3) == (input_images), LOG);


    Tensor<type, 4> expected_combination_output((rows_input - rows_kernel + 1), (cols_input - cols_kernel + 1), (input_kernels), input_images);
    expected_combination_output.chip(0, 3).chip(0, 2).setConstant(type(27));
    expected_combination_output.chip(0, 3).chip(1, 2).setConstant(type(10));
    expected_combination_output.chip(0, 3).chip(2, 2).setConstant(type(5));
    expected_combination_output.chip(1, 3).chip(0, 2).setConstant(type(9));
    expected_combination_output.chip(1, 3).chip(1, 2).setConstant(type(4));
    expected_combination_output.chip(1, 3).chip(2, 2).setConstant(type(3));

    assert_true(is_equal<4>(expected_combination_output, forward_propagation.combinations), LOG);

    Tensor<type, 4> expected_activation_output = expected_combination_output.tanh();

    assert_true(is_equal<4>(expected_activation_output, forward_propagation.outputs), LOG);

}


void ConvolutionalLayerTest::test_insert_padding()
{
    cout << "test_insert_padding\n";

    const Index input_images = 2;
    const Index input_kernels = 3;

    const Index channels = 3;

    const Index rows_input = 4;
    const Index cols_input = 4;
    const Index rows_kernel = 3;
    const Index cols_kernel = 3;

    Tensor<type,4> inputs(rows_input, cols_input, channels, input_images);
    Tensor<type,4> kernels(rows_kernel, cols_kernel, channels, input_kernels);
    Tensor<type,4> padded(rows_input, cols_input, channels, input_images);

    inputs.setConstant(type(1));

    Tensor<Index, 1> inputs_dimensions(4);
    inputs_dimensions.setValues({rows_input, cols_input, channels, input_images});

    Tensor<Index, 1> kernels_dimensions(4);
    kernels_dimensions.setValues({rows_kernel, cols_kernel, channels, input_kernels});

    ConvolutionalLayer convolutional_layer(inputs_dimensions, kernels_dimensions);

    convolutional_layer.set_convolution_type(opennn::ConvolutionalLayer::ConvolutionType::Same);
    convolutional_layer.set(inputs_dimensions, kernels_dimensions);

    convolutional_layer.insert_padding(inputs, padded);

    assert_true(padded.dimension(0) == 6 &&
                padded.dimension(1) == 6,LOG);

    assert_true((padded(0, 0, 0, 0) - type(0)) < type(NUMERIC_LIMITS_MIN) &&
                (padded(0, 1, 0, 0) - type(0)) < type(NUMERIC_LIMITS_MIN) &&
                (padded(0, 2, 0, 0) - type(0)) < type(NUMERIC_LIMITS_MIN), LOG);
}


void ConvolutionalLayerTest::test_calculate_hidden_delta_perceptron_test()
{
    cout<< "calculate_hidden_delta_perceptron_test"<<endl;

    // Current layer's values

    const Index images_number = 2;
    const Index kernels_number = 2;
    const Index output_rows_number = 4;
    const Index output_columns_number = 4;


    // Next layer's values

    const Index neurons_perceptron = 3;

    PerceptronLayer perceptronlayer(kernels_number*output_rows_number*output_columns_number,
                                    neurons_perceptron, PerceptronLayer::ActivationFunction::Linear);

    convolutional_layer.set(Tensor<type, 4>(5,5,3,1), Tensor<type, 4>(2, 2, 3, kernels_number), Tensor<type, 1>());

    PerceptronLayerForwardPropagation perceptron_layer_forward_propagate(images_number, &perceptronlayer);
    PerceptronLayerBackPropagation perceptron_layer_backpropagate(images_number, &perceptronlayer);
    ConvolutionalLayerBackPropagation convolutional_layer_backpropagate(images_number, &convolutional_layer);

    // initialize

    Tensor<float,2> synaptic_weights_perceptron(kernels_number * output_rows_number * output_columns_number,
                                                neurons_perceptron);

    for(int i=0; i<kernels_number * output_rows_number * output_columns_number*neurons_perceptron; i++)
    {
        Index neuron_value = i / (kernels_number * output_rows_number * output_columns_number);

        *(synaptic_weights_perceptron.data() + i) = 1.0 * neuron_value;
    }
/*
    perceptron_layer_backpropagate.deltas.setValues({{1,1,1},
                                                    {2,2,2}});

    perceptronlayer.set_synaptic_weights(synaptic_weights_perceptron);


    convolutional_layer.calculate_hidden_delta_perceptron(&perceptron_layer_forward_propagate,
                                                          &perceptron_layer_backpropagate,
                                                          &convolutional_layer_backpropagate);

    cout << convolutional_layer_backpropagate.deltas << endl;
*/

}

void ConvolutionalLayerTest::test_calculate_hidden_delta()
{
    cout << "test_calculate_hidden_delta\n";
    
    Tensor<Index, 1> input_dimension(4);
    input_dimension.setValues({4, 4, 1, 1});
    
    Tensor<Index, 1> kernel_dimension(4);
    kernel_dimension.setValues({2, 2, 1, 1});
    Tensor<type, 4> kernel(t1d2array<4>(kernel_dimension));
    kernel.setConstant(type(1));
    
    ConvolutionalLayer convolutional_layer(input_dimension, kernel_dimension);

    Tensor<Index, 1> next_input_dimension(4);
    next_input_dimension.setValues({3, 3, 1, 1});
    
    ConvolutionalLayer next_convolutional_layer(next_input_dimension, kernel_dimension);
    next_convolutional_layer.set_synaptic_weights(kernel);

    ConvolutionalLayerForwardPropagation next_layer_forward_propagation(1, &next_convolutional_layer);
    assert_true(next_convolutional_layer.get_outputs_rows_number() == next_layer_forward_propagation.activations_derivatives.dimension(0) &&
                next_convolutional_layer.get_outputs_columns_number() == next_layer_forward_propagation.activations_derivatives.dimension(1) &&
                next_convolutional_layer.get_kernels_number() == next_layer_forward_propagation.activations_derivatives.dimension(2) &&
                next_convolutional_layer.get_outputs_dimensions()(3) == next_layer_forward_propagation.activations_derivatives.dimension(3),
                LOG);
    //next_layer_forward_propagation.activations_derivatives.resize(2, 2, 1, 1);
    next_layer_forward_propagation.activations_derivatives.setConstant(type(0.00542));

    ConvolutionalLayerBackPropagation next_layer_back_propagation(1, &next_convolutional_layer);
    assert_true(next_layer_back_propagation.deltas_dimensions.size() == next_convolutional_layer.get_outputs_dimensions().size(), LOG);

    assert_true(next_convolutional_layer.get_outputs_dimensions()(0) == next_layer_back_propagation.deltas_dimensions(0) &&
                next_convolutional_layer.get_outputs_dimensions()(1) == next_layer_back_propagation.deltas_dimensions(1) &&
                next_convolutional_layer.get_outputs_dimensions()(2) == next_layer_back_propagation.deltas_dimensions(2) &&
                next_convolutional_layer.get_outputs_dimensions()(3) == next_layer_back_propagation.deltas_dimensions(3), LOG);
    //next_layer_back_propagation.convolutional_delta.resize(2, 2, 1, 1);
                                              
    TensorMap<Tensor<type, 4>> next_layer_delta(next_layer_back_propagation.deltas_data, t1d2array<4>(next_layer_back_propagation.deltas_dimensions));
    next_layer_delta(0, 0, 0, 0) = type(0.25);
    next_layer_delta(0, 1, 0, 0) = type(-0.1);
    next_layer_delta(1, 0, 0, 0) = type(0.6);
    next_layer_delta(1, 1, 0, 0) = type(1.1);

    ConvolutionalLayerBackPropagation back_propagation(1, &convolutional_layer);



    convolutional_layer.calculate_hidden_delta(&next_layer_forward_propagation, &next_layer_back_propagation, &back_propagation);

    Tensor<type, 4> expected_delta(3, 3, 1, 1);
    expected_delta(0, 0, 0, 0) = type(0.001355);
    expected_delta(0, 1, 0, 0) = type(0.000813);
    expected_delta(0, 2, 0, 0) = type(-0.000542);
    expected_delta(1, 0, 0, 0) = type(0.004611018);
    expected_delta(1, 1, 0, 0) = type(0.010038385);
    expected_delta(1, 2, 0, 0) = type(0.005427367);
    expected_delta(2, 0, 0, 0) = type(0.003256018);
    expected_delta(2, 1, 0, 0) = type(0.009225385);
    expected_delta(2, 2, 0, 0) = type(0.005969367);
   
    TensorMap<Tensor<type, 4>> output_delta(back_propagation.deltas_data, t1d2array<4>(back_propagation.deltas_dimensions));

    assert_true(is_equal<4>(expected_delta, output_delta), LOG);
}


void ConvolutionalLayerTest::test_calculate_error_gradient()
{
    cout << "test_calculate_error_gradient\n";

    Tensor<Index, 1> input_dimension(4);
    input_dimension.setValues({4, 4, 1, 1});
    
    Tensor<Index, 1> kernel_dimension(4);
    kernel_dimension.setValues({2, 2, 1, 1});
    Tensor<type, 4> kernel(t1d2array<4>(kernel_dimension));
    kernel.setConstant(type(1));
    
    ConvolutionalLayer convolutional_layer(input_dimension, kernel_dimension);

    ConvolutionalLayerBackPropagation back_propagation(1, &convolutional_layer);
    assert_true(back_propagation.deltas_dimensions.size() == convolutional_layer.get_outputs_dimensions().size(), LOG);

    assert_true(back_propagation.deltas_dimensions(0) == convolutional_layer.get_outputs_dimensions()(0) &&
                back_propagation.deltas_dimensions(1) == convolutional_layer.get_outputs_dimensions()(1) &&
                back_propagation.deltas_dimensions(2) == convolutional_layer.get_outputs_dimensions()(2) &&
                back_propagation.deltas_dimensions(3) == convolutional_layer.get_outputs_dimensions()(3), 
                LOG);
                

    TensorMap<Tensor<type, 4>> delta(back_propagation.deltas_data, t1d2array<4>(back_propagation.deltas_dimensions));
    delta(0, 0, 0, 0) = type(0.001355);
    delta(0, 1, 0, 0) = type(0.000813);
    delta(0, 2, 0, 0) = type(-0.000542);
    delta(1, 0, 0, 0) = type(0.004611018);
    delta(1, 1, 0, 0) = type(0.010038385);
    delta(1, 2, 0, 0) = type(0.005427367);
    delta(2, 0, 0, 0) = type(0.003256018);
    delta(2, 1, 0, 0) = type(0.009225385);
    delta(2, 2, 0, 0) = type(0.005969367);
    
    ConvolutionalLayerForwardPropagation forward_propagation(1, &convolutional_layer);
    
    assert_true(forward_propagation.activations_derivatives.dimension(0) == convolutional_layer.get_outputs_dimensions()(0) &&
                forward_propagation.activations_derivatives.dimension(1) == convolutional_layer.get_outputs_dimensions()(1) &&
                forward_propagation.activations_derivatives.dimension(2) == convolutional_layer.get_outputs_dimensions()(2) &&
                forward_propagation.activations_derivatives.dimension(3) == convolutional_layer.get_outputs_dimensions()(3), 
                LOG);
    
    forward_propagation.activations_derivatives.setConstant(type(0));
    forward_propagation.activations_derivatives.chip(1, 0).setConstant(type(1));

    Tensor<type, 4> inputs(t1d2array<4>(input_dimension));
    inputs.chip(0, 0).setConstant(type(1));
    inputs.chip(1, 0).setConstant(type(-2));
    inputs.chip(2, 0).setConstant(type(3));
    inputs.chip(3, 0).setConstant(type(-4));

    //Test
    convolutional_layer.calculate_error_gradient(inputs.data(), &forward_propagation, &back_propagation);

    Tensor<type, 4> expected_weight_gradient(t1d2array<4>(kernel_dimension));
    expected_weight_gradient(0, 0, 0, 0) = type(-0.04015354);
    expected_weight_gradient(0, 1, 0, 0) = type(-0.04015354);
    expected_weight_gradient(1, 0, 0, 0) = type(0.06023031);
    expected_weight_gradient(1, 1, 0, 0) = type(0.06023031);

    Tensor<type, 1> expected_bias_gradient(1);
    expected_bias_gradient(0) = type(0.02007677);

    assert_true(is_equal<4>(expected_weight_gradient, back_propagation.synaptic_weights_derivatives), LOG);

    assert_true(is_equal<1>(expected_bias_gradient, back_propagation.biases_derivatives), LOG);
}

void ConvolutionalLayerTest::test_memcpy_approach()
{

    const int images_number = 1;
    const int kernel_number = 1;

    const int channel = 3;

    const int rows_input = 4;
    const int cols_input = 4;

    const int kernel_rows = 2;
    const int kernel_cols = 2;

    Tensor<type, 4> input(rows_input, cols_input, channel, images_number);
    Tensor<type, 4> kernel(kernel_rows, kernel_cols, channel, kernel_number);
    Tensor<type, 4> result((rows_input-kernel_rows)+1,
                           (cols_input-kernel_cols)+1,
                            kernel_number,
                            images_number);

    Tensor<type, 3> tmp_result((rows_input-kernel_rows)+1,
                               (cols_input-kernel_cols)+1,
                               1);

    const Index output_size_rows_cols = ((rows_input-kernel_rows)+1)*((cols_input-kernel_cols)+1);

    float* ptr_result = (float*) malloc(static_cast<size_t>(output_size_rows_cols*kernel_number*images_number*sizeof(type)));

    input.setConstant(1.0);

    input.chip(0,2).setConstant(1.);
    input.chip(1,2).setConstant(2.);
    input.chip(2,2).setConstant(3.);

    kernel.setConstant(type(1./12.));
    kernel.chip(1,3).setConstant(type(1./6.));

    time_t beginning_time;
    time_t current_time;
    time(&beginning_time);
    type elapsed_time = type(0);

    const Eigen::array<ptrdiff_t, 3> dims = {0, 1, 2};

    #pragma omp parallel for
    for(int i =0; i<images_number ;i++)
    {
        const Index next_image = input.dimension(0)*input.dimension(1)*input.dimension(2);

        const TensorMap<Tensor<type, 3>> single_image(input.data()+i*next_image, input.dimension(0), input.dimension(1), input.dimension(2));

        for(int j =0; j<kernel_number; j++)
        {
            const Index next_kernel = kernel.dimension(0)*kernel.dimension(1)*kernel.dimension(2);

            const TensorMap<Tensor<type, 3>> single_kernel(kernel.data()+j*next_kernel , kernel.dimension(0), kernel.dimension(1), kernel.dimension(2));

            Tensor<type, 3> tmp_result = single_image.convolve(single_kernel, dims);

            memcpy(result.data() +j*output_size_rows_cols +i*output_size_rows_cols*kernel_number,
                   tmp_result.data(), output_size_rows_cols*sizeof(type));
         }
    }
}


void ConvolutionalLayerTest::test_calculate_hidden_delta1()
{
    cout << "test_calculate_hidden_delta1\n";

    Tensor<Index, 1> inputs_dimension(4);
    inputs_dimension.setValues({4, 4, 2, 2});

    Tensor<Index, 1> kernels_dimension(4);
    kernels_dimension.setValues({2, 2, 2, 2});
    Tensor<type, 4> kernels(t1d2array<4>(kernels_dimension));
    kernels.chip(0, 3).chip(0, 2).setConstant(-0.5);
    kernels.chip(0, 3).chip(1, 2).setConstant(0.25);
    kernels.chip(1, 3).chip(0, 2).setConstant(0.125);
    kernels.chip(1, 3).chip(1, 2).setConstant(0.0625);

    ConvolutionalLayer convolutional_layer(inputs_dimension, kernels_dimension);

    Tensor<Index, 1> next_layer_inputs_dimension(4);
    next_layer_inputs_dimension.setValues({3, 3, 2, 2});
    ConvolutionalLayer next_convolutional_layer(next_layer_inputs_dimension, kernels_dimension);
    next_convolutional_layer.set_synaptic_weights(kernels);

    Tensor<type, 1> biases(2);
    biases.setValues({0, 1});
    next_convolutional_layer.set_biases(biases);

    ConvolutionalLayerForwardPropagation next_forward_propagation(2, &next_convolutional_layer);
    ConvolutionalLayerBackPropagation next_backward_propagation(2, &next_convolutional_layer);

    Eigen::array<Index, 4> next_backward_propagation_deltas_dimensions({
        next_backward_propagation.deltas_dimensions(0),
        next_backward_propagation.deltas_dimensions(1),
        next_backward_propagation.deltas_dimensions(2),
        next_backward_propagation.deltas_dimensions(3)
    });

    TensorMap<Tensor<type, 4>> delta(next_backward_propagation.deltas_data, next_backward_propagation_deltas_dimensions);
    delta.chip(0, 3).chip(0, 2).setConstant(1);
    delta.chip(0, 3).chip(1, 2).setConstant(2);
    delta.chip(1, 3).chip(0, 2).setConstant(4);
    delta.chip(1, 3).chip(1, 2).setConstant(8);

    next_forward_propagation.activations_derivatives.chip(0, 2).setConstant(0);
    next_forward_propagation.activations_derivatives.chip(1, 2).setConstant(1);

    ConvolutionalLayerBackPropagation backward_propagation(2, &convolutional_layer);

    convolutional_layer.calculate_hidden_delta(&next_forward_propagation, &next_backward_propagation, &backward_propagation);

    Eigen::array<Index, 4> backward_propagation_deltas_dimensions({
        backward_propagation.deltas_dimensions(0),
        backward_propagation.deltas_dimensions(1),
        backward_propagation.deltas_dimensions(2),
        backward_propagation.deltas_dimensions(3)
    });

    TensorMap<Tensor<type, 4>> delta_res(backward_propagation.deltas_data, backward_propagation_deltas_dimensions);

    Tensor<type, 4> expected_result(3, 3, 2, 2);
    expected_result(0, 0, 0, 0) = type(0.5);
    expected_result(0, 0, 1, 0) = type(0.125);
    expected_result(0, 1, 0, 0) = type(1);
    expected_result(0, 1, 1, 0) = type(0.25);
    expected_result(0, 2, 0, 0) = type(0.5);
    expected_result(0, 2, 1, 0) = type(0.125);
    expected_result(1, 0, 0, 0) = type(1);
    expected_result(1, 0, 1, 0) = type(0.25);
    expected_result(1, 1, 0, 0) = type(2);
    expected_result(1, 1, 1, 0) = type(0.5);
    expected_result(1, 2, 0, 0) = type(1);
    expected_result(1, 2, 1, 0) = type(0.25);
    expected_result(2, 0, 0, 0) = type(0.5);
    expected_result(2, 0, 1, 0) = type(0.125);
    expected_result(2, 1, 0, 0) = type(1);
    expected_result(2, 1, 1, 0) = type(0.25);
    expected_result(2, 2, 0, 0) = type(0.5);
    expected_result(2, 2, 1, 0) = type(0.125);

    expected_result(0, 0, 0, 1) = type(2);
    expected_result(0, 0, 1, 1) = type(0.5);
    expected_result(0, 1, 0, 1) = type(4);
    expected_result(0, 1, 1, 1) = type(1);
    expected_result(0, 2, 0, 1) = type(2);
    expected_result(0, 2, 1, 1) = type(0.5);
    expected_result(1, 0, 0, 1) = type(4);
    expected_result(1, 0, 1, 1) = type(1);
    expected_result(1, 1, 0, 1) = type(8);
    expected_result(1, 1, 1, 1) = type(2);
    expected_result(1, 2, 0, 1) = type(4);
    expected_result(1, 2, 1, 1) = type(1);
    expected_result(2, 0, 0, 1) = type(2);
    expected_result(2, 0, 1, 1) = type(0.5);
    expected_result(2, 1, 0, 1) = type(4);
    expected_result(2, 1, 1, 1) = type(1);
    expected_result(2, 2, 0, 1) = type(2);
    expected_result(2, 2, 1, 1) = type(0.5);

    assert_true(
        is_equal<4>(expected_result, delta_res),
        LOG);
   
}

void ConvolutionalLayerTest::test_calculate_error_gradient1()
{
    cout << "test_calculate_error_gradient1\n";
    
    Tensor<Index, 1> inputs_dimension(4);
    inputs_dimension.setValues({3, 3, 2, 2});

    Tensor<type, 4> input(t1d2array<4>(inputs_dimension));
    input.chip(0, 3).chip(0, 2).setConstant(1);
    input.chip(0, 3).chip(1, 2).setConstant(2);
    input.chip(1, 3).chip(0, 2).setConstant(4);
    input.chip(1, 3).chip(1, 2).setConstant(8);

    Tensor<Index, 1> kernels_dimension(4);
    kernels_dimension.setValues({2, 2, 2, 2});
    
    ConvolutionalLayer convolutional_layer(inputs_dimension, kernels_dimension);

    ConvolutionalLayerForwardPropagation forward_propagation(2, &convolutional_layer);

    forward_propagation.activations_derivatives.chip(0, 2).setConstant(0);
    forward_propagation.activations_derivatives.chip(1, 2).setConstant(1);

    ConvolutionalLayerBackPropagation backward_propagation(2, &convolutional_layer);

    TensorMap<Tensor<type, 4>> delta(backward_propagation.deltas_data, t1d2array<4>(backward_propagation.deltas_dimensions));
    delta.chip(0, 3).chip(0, 2).setConstant(type(1) / type(2));
    delta.chip(0, 3).chip(1, 2).setConstant(type(1) / type(4));
    delta.chip(1, 3).chip(0, 2).setConstant(type(1) / type(8));
    delta.chip(1, 3).chip(1, 2).setConstant(type(1) / type(16));

    convolutional_layer.calculate_error_gradient(input.data(), &forward_propagation, &backward_propagation);

    Tensor<type, 4> expected_synaptic_weights_derivatives(2, 2, 2, 2);
    expected_synaptic_weights_derivatives.chip(0, 3).chip(0, 2).setConstant(type(2));
    expected_synaptic_weights_derivatives.chip(0, 3).chip(1, 2).setConstant(type(0.5));
    expected_synaptic_weights_derivatives.chip(1, 3).chip(0, 2).setConstant(type(8));
    expected_synaptic_weights_derivatives.chip(1, 3).chip(1, 2).setConstant(type(2));

    assert_true(is_equal<4>(expected_synaptic_weights_derivatives, backward_propagation.synaptic_weights_derivatives), LOG);

    Tensor<type, 1> expected_biases_derivatives(2);
    expected_biases_derivatives.setValues({type(1), type(0.25)});

    assert_true(is_equal<1>(expected_biases_derivatives, backward_propagation.biases_derivatives), LOG);
}

void ConvolutionalLayerTest::run_test_case()
{
   cout << "Running convolutional layer test case...\n";

   // Constructor and destructor

   test_constructor();
   test_destructor();

   // Set methods

   test_set();
   test_set_parameters();

   // Convolutions

   test_eigen_convolution();
   test_eigen_convolution_3d();
   //test_read_bmp();

   // Combinations

   test_calculate_combinations();
   //test_calculate_average_pooling_outputs();
   //test_calculate_max_pooling_outputs();

   // Activation

   test_calculate_activations();
   test_calculate_activations_derivatives();

   // Outputs

//   test_calculate_outputs();

   // Padding

   test_insert_padding();

   // Forward propagate

   test_forward_propagate_training();
   test_forward_propagate_not_training();

   // Back_propagate

   //test_calculate_hidden_delta_perceptron_test();
   test_calculate_hidden_delta();
   test_calculate_error_gradient();
   test_calculate_hidden_delta1();
   test_calculate_error_gradient1();
    
   //Utils
   test_memcpy_approach();

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
