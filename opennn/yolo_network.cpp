#include <iostream>

#include "yolo_network.h"
#include "yolo_dataset.h"
#include "tensors.h"
#include "pooling_layer.h"
#include "convolutional_layer.h"
#include "addition_layer_3d.h"
#include "strings_utilities.h"

namespace opennn
{

YoloNetwork::YoloNetwork()
{
    NeuralNetwork::set();
}

YoloNetwork::YoloNetwork(const dimensions& input_dimensions)
{
    set(input_dimensions);
}

void YoloNetwork::set(const dimensions& input_dimensions)
{
    if(input_dimensions.size() != 3)
        throw runtime_error("Input dimension must be 3");

    image_height = input_dimensions[0];
    image_width = input_dimensions[1];
    image_channels = input_dimensions[2];

    set(image_height, image_width, image_channels);
}

void YoloNetwork::set(const Index& height, const Index& width, const Index& channels)
{
    const dimensions convolution_stride_dimensions = {1, 1};
    const ConvolutionalLayer::ConvolutionType convolution_type = ConvolutionalLayer::ConvolutionType::Same;

    const dimensions pool_dimensions = {2, 2};
    const dimensions pooling_stride_dimensions = { 2, 2 };
    const dimensions padding_dimensions = { 0, 0 };
    const PoolingLayer::PoolingMethod pooling_method = PoolingLayer::PoolingMethod::MaxPooling;

    add_layer(make_unique<ScalingLayer4D>((dimensions){height, width, channels}));

    // cout<<get_output_dimensions()[0]<<","<<get_output_dimensions()[1]<<","<<get_output_dimensions()[2]<<","<<get_output_dimensions()[3]<<endl;

    add_layer(make_unique<ConvolutionalLayer> (get_output_dimensions(),
                                                 (dimensions){3, 3, channels, 32},
                                                 ConvolutionalLayer::ActivationFunction::RectifiedLinear,
                                                 convolution_stride_dimensions,
                                                 convolution_type,
                                                 "Convolutional layer 1"));

    add_layer(make_unique<PoolingLayer>(get_output_dimensions(),
                                           pool_dimensions,
                                           pooling_stride_dimensions,
                                           padding_dimensions,
                                           pooling_method,
                                           "Pooling layer 1"));

    add_layer(make_unique<ConvolutionalLayer>(get_output_dimensions(),
                                                 (dimensions){3, 3, get_output_dimensions()[2], 64},
                                                 ConvolutionalLayer::ActivationFunction::RectifiedLinear,
                                                 convolution_stride_dimensions,
                                                 convolution_type,
                                                 "Convolutional layer 2"));

    add_layer(make_unique<PoolingLayer>(get_output_dimensions(),
                                           pool_dimensions,
                                           pooling_stride_dimensions,
                                           padding_dimensions,
                                           pooling_method,
                                           "Pooling layer 2"));

    add_layer(make_unique<ConvolutionalLayer>(get_output_dimensions(),
                                                 (dimensions){3, 3, get_output_dimensions()[2], 128},
                                                 ConvolutionalLayer::ActivationFunction::RectifiedLinear,
                                                 convolution_stride_dimensions,
                                                 convolution_type,
                                                 "Convolutional layer 3"));

    add_layer(make_unique<ConvolutionalLayer>(get_output_dimensions(),
                                                 (dimensions){1, 1, get_output_dimensions()[2], 64},
                                                 ConvolutionalLayer::ActivationFunction::RectifiedLinear,
                                                 convolution_stride_dimensions,
                                                 convolution_type,
                                                 "Convolutional layer 4"));

    add_layer(make_unique<ConvolutionalLayer>(get_output_dimensions(),
                                                 (dimensions){3, 3, get_output_dimensions()[2], 128},
                                                 ConvolutionalLayer::ActivationFunction::RectifiedLinear,
                                                 convolution_stride_dimensions,
                                                 convolution_type,
                                                 "Convolutional layer 5"));

    add_layer(make_unique<PoolingLayer>(get_output_dimensions(),
                                           pool_dimensions,
                                           pooling_stride_dimensions,
                                           padding_dimensions,
                                           pooling_method,
                                           "Pooling layer 3"));

    add_layer(make_unique<ConvolutionalLayer>(get_output_dimensions(),
                                                 (dimensions){3, 3, get_output_dimensions()[2], 256},
                                                 ConvolutionalLayer::ActivationFunction::RectifiedLinear,
                                                 convolution_stride_dimensions,
                                                 convolution_type,
                                                 "Convolutional layer 6"));

    add_layer(make_unique<ConvolutionalLayer>(get_output_dimensions(),
                                                 (dimensions){1, 1, get_output_dimensions()[2], 128},
                                                 ConvolutionalLayer::ActivationFunction::RectifiedLinear,
                                                 convolution_stride_dimensions,
                                                 convolution_type,
                                                 "Convolutional layer 7"));

    add_layer(make_unique<ConvolutionalLayer>(get_output_dimensions(),
                                                 (dimensions){3, 3, get_output_dimensions()[2], 256},
                                                 ConvolutionalLayer::ActivationFunction::RectifiedLinear,
                                                 convolution_stride_dimensions,
                                                 convolution_type,
                                                 "Convolutional layer 8"));

    add_layer(make_unique<PoolingLayer>(get_output_dimensions(),
                                           pool_dimensions,
                                           pooling_stride_dimensions,
                                           padding_dimensions,
                                           pooling_method,
                                           "Pooling layer 4"));


    add_layer(make_unique<ConvolutionalLayer>(get_output_dimensions(),
                                                 (dimensions){3, 3, get_output_dimensions()[2], 512},
                                                 ConvolutionalLayer::ActivationFunction::RectifiedLinear,
                                                 convolution_stride_dimensions,
                                                 convolution_type,
                                                 "Convolutional layer 9"));

    add_layer(make_unique<ConvolutionalLayer>(get_output_dimensions(),
                                                 (dimensions){1, 1, get_output_dimensions()[2], 256},
                                                 ConvolutionalLayer::ActivationFunction::RectifiedLinear,
                                                 convolution_stride_dimensions,
                                                 convolution_type,
                                                 "Convolutional layer 10"));

    add_layer(make_unique<ConvolutionalLayer>(get_output_dimensions(),
                                                 (dimensions){3, 3, get_output_dimensions()[2], 512},
                                                 ConvolutionalLayer::ActivationFunction::RectifiedLinear,
                                                 convolution_stride_dimensions,
                                                 convolution_type,
                                                 "Convolutional layer 11"));

    add_layer(make_unique<ConvolutionalLayer>(get_output_dimensions(),
                                                 (dimensions){1, 1, get_output_dimensions()[2], 256},
                                                 ConvolutionalLayer::ActivationFunction::RectifiedLinear,
                                                 convolution_stride_dimensions,
                                                 convolution_type,
                                                 "Convolutional layer 12"));

    add_layer(make_unique<ConvolutionalLayer>(get_output_dimensions(),
                                                 (dimensions){3, 3, get_output_dimensions()[2], 512},
                                                 ConvolutionalLayer::ActivationFunction::RectifiedLinear,
                                                 convolution_stride_dimensions,
                                                 convolution_type,
                                                 "Convolutional layer 13"));

    add_layer(make_unique<PoolingLayer>(get_output_dimensions(),
                                           pool_dimensions,
                                           pooling_stride_dimensions,
                                           padding_dimensions,
                                           pooling_method,
                                           "Pooling layer 5"));

    add_layer(make_unique<ConvolutionalLayer>(get_output_dimensions(),
                                                 (dimensions){3, 3, get_output_dimensions()[2], 1024},
                                                 ConvolutionalLayer::ActivationFunction::RectifiedLinear,
                                                 convolution_stride_dimensions,
                                                 convolution_type,
                                                 "Convolutional layer 14"));

    add_layer(make_unique<ConvolutionalLayer>(get_output_dimensions(),
                                                 (dimensions){1, 1, get_output_dimensions()[2], 512},
                                                 ConvolutionalLayer::ActivationFunction::RectifiedLinear,
                                                 convolution_stride_dimensions,
                                                 convolution_type,
                                                 "Convolutional layer 15"));

    add_layer(make_unique<ConvolutionalLayer>(get_output_dimensions(),
                                                 (dimensions){3, 3, get_output_dimensions()[2], 1024},
                                                 ConvolutionalLayer::ActivationFunction::RectifiedLinear,
                                                 convolution_stride_dimensions,
                                                 convolution_type,
                                                 "Convolutional layer 16"));

    add_layer(make_unique<ConvolutionalLayer>(get_output_dimensions(),
                                                 (dimensions){1, 1, get_output_dimensions()[2], 512},
                                                 ConvolutionalLayer::ActivationFunction::RectifiedLinear,
                                                 convolution_stride_dimensions,
                                                 convolution_type,
                                                 "Convolutional layer 17"));

    add_layer(make_unique<ConvolutionalLayer>(get_output_dimensions(),
                                                 (dimensions){3, 3, get_output_dimensions()[2], 1024},
                                                 ConvolutionalLayer::ActivationFunction::RectifiedLinear,
                                                 convolution_stride_dimensions,
                                                 convolution_type,
                                                 "Convolutional layer 18"));

    add_layer(make_unique<ConvolutionalLayer>(get_output_dimensions(),
                                                 (dimensions){1, 1, get_output_dimensions()[2], 125},
                                                 ConvolutionalLayer::ActivationFunction::RectifiedLinear,
                                                 convolution_stride_dimensions,
                                                 convolution_type,
                                                 "Convolutional layer 19"));

}

Tensor<type, 4> YoloNetwork::calculate_outputs(const Tensor<type, 4>& input)
{
    const pair<type*, dimensions> input_pair((type*)input.data(), { input.dimension(0), input.dimension(1), input.dimension(2), input.dimension(3)});

    ForwardPropagation forward_propagation(input.dimension(0), this);

    const pair<type*, dimensions> outputs_pair = forward_propagation.get_last_trainable_layer_outputs_pair();

    return tensor_map_4(outputs_pair);

}

void YoloNetwork::load_yolo(const string& path)
{
    cout << "Loading YOLO model..." << endl;

    load(path);
}

}
