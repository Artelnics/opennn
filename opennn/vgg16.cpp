#include "vgg16.h"

VGG16::VGG16() : NeuralNetwork()
{
//        set_project_type(ProjectType::ImageClassification);

    Tensor<Index, 1> input_variables_dimensions(3);
    input_variables_dimensions.setValues({224,224,3});

    Tensor<Index, 1> kernels_dimensions(4);

    Tensor<Index, 1> pooling_dimensions(2);
    pooling_dimensions.setValues({2, 2});

    // Block 1

    kernels_dimensions.setValues({3,3,3,64});
    convolutional_layer_1.set(input_variables_dimensions, kernels_dimensions);
    convolutional_layer_1.set_activation_function(ConvolutionalLayer::ActivationFunction::RectifiedLinear);
    convolutional_layer_1.set_name("convolutional_layer_1");
    convolutional_layer_1.set_convolution_type(ConvolutionalLayer::ConvolutionType::Same);
    add_layer(&convolutional_layer_1);

    kernels_dimensions.setValues({3,3,64,64});
    convolutional_layer_2.set(convolutional_layer_1.get_outputs_dimensions(), kernels_dimensions);
    convolutional_layer_2.set_activation_function(ConvolutionalLayer::ActivationFunction::RectifiedLinear);
    convolutional_layer_2.set_name("convolutional_layer_2");
    convolutional_layer_2.set_convolution_type(ConvolutionalLayer::ConvolutionType::Same);
    add_layer(&convolutional_layer_2);

    pooling_layer_1.set(convolutional_layer_2.get_outputs_dimensions(), pooling_dimensions);
    pooling_layer_1.set_pooling_method(PoolingLayer::PoolingMethod::MaxPooling);
    pooling_layer_1.set_column_stride(2);
    pooling_layer_1.set_row_stride(2);
    pooling_layer_1.set_name("pooling_layer_1");
    add_layer(&pooling_layer_1);

   // Block 2

    kernels_dimensions.setValues({3,3,64,128});
    convolutional_layer_3.set(pooling_layer_1.get_outputs_dimensions(), kernels_dimensions);
    convolutional_layer_3.set_activation_function(ConvolutionalLayer::ActivationFunction::RectifiedLinear);
    convolutional_layer_3.set_convolution_type(ConvolutionalLayer::ConvolutionType::Same);
    convolutional_layer_3.set_name("convolutional_layer_3");
    add_layer(&convolutional_layer_3);

    kernels_dimensions.setValues({3,3,128,128});
    convolutional_layer_4.set(convolutional_layer_3.get_outputs_dimensions(), kernels_dimensions);
    convolutional_layer_4.set_activation_function(ConvolutionalLayer::ActivationFunction::RectifiedLinear);
    convolutional_layer_4.set_convolution_type(ConvolutionalLayer::ConvolutionType::Same);
    convolutional_layer_4.set_name("convolutional_layer_4");
    add_layer(&convolutional_layer_4);

    pooling_layer_2.set(convolutional_layer_4.get_outputs_dimensions(), pooling_dimensions);
    pooling_layer_2.set_pooling_method(PoolingLayer::PoolingMethod::MaxPooling);
    pooling_layer_2.set_column_stride(2);
    pooling_layer_2.set_row_stride(2);
    pooling_layer_2.set_name("pooling_layer_2");
    add_layer(&pooling_layer_2);

    // Block 3

    kernels_dimensions.setValues({3,3,128,256});
    convolutional_layer_5.set(pooling_layer_2.get_outputs_dimensions(), kernels_dimensions);
    convolutional_layer_5.set_activation_function(ConvolutionalLayer::ActivationFunction::RectifiedLinear);
    convolutional_layer_5.set_convolution_type(ConvolutionalLayer::ConvolutionType::Same);
    convolutional_layer_5.set_name("convolutional_layer_5");
    add_layer(&convolutional_layer_5);

    kernels_dimensions.setValues({3,3,256,256});
    convolutional_layer_6.set(convolutional_layer_5.get_outputs_dimensions(), kernels_dimensions);
    convolutional_layer_6.set_activation_function(ConvolutionalLayer::ActivationFunction::RectifiedLinear);
    convolutional_layer_6.set_convolution_type(ConvolutionalLayer::ConvolutionType::Same);
    convolutional_layer_6.set_name("convolutional_layer_6");
    add_layer(&convolutional_layer_6);

    kernels_dimensions.setValues({3,3,256,256});
    convolutional_layer_7.set(convolutional_layer_6.get_outputs_dimensions(), kernels_dimensions);
    convolutional_layer_7.set_activation_function(ConvolutionalLayer::ActivationFunction::RectifiedLinear);
    convolutional_layer_7.set_convolution_type(ConvolutionalLayer::ConvolutionType::Same);
    convolutional_layer_7.set_name("convolutional_layer_7");
    add_layer(&convolutional_layer_7);

    pooling_layer_3.set(convolutional_layer_7.get_outputs_dimensions(), pooling_dimensions);
    pooling_layer_3.set_pooling_method(PoolingLayer::PoolingMethod::MaxPooling);
    pooling_layer_3.set_column_stride(2);
    pooling_layer_3.set_row_stride(2);
    pooling_layer_3.set_name("pooling_layer_3");
    add_layer(&pooling_layer_3);

    // Block 4

    kernels_dimensions.setValues({3,3,256,512});
    convolutional_layer_8.set(pooling_layer_3.get_outputs_dimensions(), kernels_dimensions);
    convolutional_layer_8.set_activation_function(ConvolutionalLayer::ActivationFunction::RectifiedLinear);
    convolutional_layer_8.set_convolution_type(ConvolutionalLayer::ConvolutionType::Same);
    convolutional_layer_8.set_name("convolutional_layer_8");
    add_layer(&convolutional_layer_8);

    kernels_dimensions.setValues({3,3,512,512});
    convolutional_layer_9.set(convolutional_layer_8.get_outputs_dimensions(), kernels_dimensions);
    convolutional_layer_9.set_activation_function(ConvolutionalLayer::ActivationFunction::RectifiedLinear);
    convolutional_layer_9.set_convolution_type(ConvolutionalLayer::ConvolutionType::Same);
    convolutional_layer_9.set_name("convolutional_layer_9");
    add_layer(&convolutional_layer_9);

    kernels_dimensions.setValues({3,3,512,512});
    convolutional_layer_10.set(convolutional_layer_9.get_outputs_dimensions(), kernels_dimensions);
    convolutional_layer_10.set_activation_function(ConvolutionalLayer::ActivationFunction::RectifiedLinear);
    convolutional_layer_10.set_convolution_type(ConvolutionalLayer::ConvolutionType::Same);
    convolutional_layer_10.set_name("convolutional_layer_10");
    add_layer(&convolutional_layer_10);

    pooling_layer_4.set(convolutional_layer_10.get_outputs_dimensions(), pooling_dimensions);
    pooling_layer_4.set_pooling_method(PoolingLayer::PoolingMethod::MaxPooling);
    pooling_layer_4.set_column_stride(2);
    pooling_layer_4.set_row_stride(2);
    pooling_layer_4.set_name("pooling_layer_4");
    add_layer(&pooling_layer_4);

    // Block 5

    kernels_dimensions.setValues({3,3,512,512});
    convolutional_layer_11.set(pooling_layer_4.get_outputs_dimensions(), kernels_dimensions);
    convolutional_layer_11.set_activation_function(ConvolutionalLayer::ActivationFunction::RectifiedLinear);
    convolutional_layer_11.set_convolution_type(ConvolutionalLayer::ConvolutionType::Same);
    convolutional_layer_11.set_name("convolutional_layer_11");
    add_layer(&convolutional_layer_11);

    kernels_dimensions.setValues({3,3,512,512});
    convolutional_layer_12.set(convolutional_layer_11.get_outputs_dimensions(), kernels_dimensions);
    convolutional_layer_12.set_activation_function(ConvolutionalLayer::ActivationFunction::RectifiedLinear);
    convolutional_layer_12.set_convolution_type(ConvolutionalLayer::ConvolutionType::Same);
    convolutional_layer_12.set_name("convolutional_layer_12");
    add_layer(&convolutional_layer_12);

    kernels_dimensions.setValues({3,3,512,512});
    convolutional_layer_13.set(convolutional_layer_12.get_outputs_dimensions(), kernels_dimensions);
    convolutional_layer_13.set_activation_function(ConvolutionalLayer::ActivationFunction::RectifiedLinear);
    convolutional_layer_13.set_convolution_type(ConvolutionalLayer::ConvolutionType::Same);
    convolutional_layer_13.set_name("convolutional_layer_13");
    add_layer(&convolutional_layer_13);

    pooling_layer_5.set(convolutional_layer_13.get_outputs_dimensions(), pooling_dimensions);
    pooling_layer_5.set_pooling_method(PoolingLayer::PoolingMethod::MaxPooling);
    pooling_layer_5.set_column_stride(2);
    pooling_layer_5.set_row_stride(2);
    pooling_layer_5.set_name("pooling_layer_5");
    add_layer(&pooling_layer_5);

    flatten_layer.set(pooling_layer_5.get_outputs_dimensions());
    flatten_layer.set_name("flatten_layer");
    add_layer(&flatten_layer);

    perceptron_layer_1.set(flatten_layer.get_outputs_dimensions()(0), 4096);
    perceptron_layer_1.set_activation_function(PerceptronLayer::ActivationFunction::RectifiedLinear);
    perceptron_layer_1.set_name("perceptron_layer_1");
    add_layer(&perceptron_layer_1);

    perceptron_layer_2.set(4096, 4096);
    perceptron_layer_2.set_activation_function(PerceptronLayer::ActivationFunction::RectifiedLinear);
    perceptron_layer_2.set_name("perceptron_layer_2");
    add_layer(&perceptron_layer_2);

    const Index classes_number = 1000;

    probabilistic_layer.set(perceptron_layer_2.get_neurons_number(), classes_number);
    add_layer(&probabilistic_layer);

    summary();

};

VGG16::~VGG16()
{

}
