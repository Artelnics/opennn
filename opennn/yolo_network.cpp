#include <iostream>

#include "yolo_network.h"
#include "detection_layer.h"
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

YoloNetwork::YoloNetwork(const dimensions& input_dimensions, const vector<Tensor<type, 1>>& new_anchors)
{

    set(input_dimensions, new_anchors);
}

void YoloNetwork::set(const dimensions& input_dimensions, const vector<Tensor<type, 1>>& new_anchors)
{
    if(input_dimensions.size() != 3)
        throw runtime_error("Input dimension must be 3");

    image_height = input_dimensions[0];
    image_width = input_dimensions[1];
    image_channels = input_dimensions[2];

    anchors = new_anchors;

    model_type = NeuralNetwork::ModelType::YoloV2;

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
    const ConvolutionalLayer::ActivationFunction activation_function = ConvolutionalLayer::ActivationFunction::LeakyRectifiedLinear;

    // add_layer(make_unique<ScalingLayer4D>((dimensions){height, width, channels}));

    // cout<<get_output_dimensions()[0]<<","<<get_output_dimensions()[1]<<","<<get_output_dimensions()[2]<<","<<get_output_dimensions()[3]<<endl;

    add_layer(make_unique<ConvolutionalLayer> (/*get_output_dimensions(),*/
                                              (dimensions){height, width, channels},
                                              (dimensions){3, 3, channels, 32},
                                              activation_function,
                                              convolution_stride_dimensions,
                                              convolution_type,
                                              "Convolutional layer 1"));
    // layers[1]->set_parameters_constant(1);

    add_layer(make_unique<PoolingLayer>(get_output_dimensions(),
                                           pool_dimensions,
                                           pooling_stride_dimensions,
                                           padding_dimensions,
                                           pooling_method,
                                           "Pooling layer 1"));

    add_layer(make_unique<ConvolutionalLayer>(get_output_dimensions(),
                                                 (dimensions){3, 3, get_output_dimensions()[2], 64},
                                                 activation_function,
                                                 convolution_stride_dimensions,
                                                 convolution_type,
                                                 "Convolutional layer 2"));

    // layers[3]->set_parameters_constant(1);

    add_layer(make_unique<PoolingLayer>(get_output_dimensions(),
                                           pool_dimensions,
                                           pooling_stride_dimensions,
                                           padding_dimensions,
                                           pooling_method,
                                           "Pooling layer 2"));

    // add_layer(make_unique<ConvolutionalLayer>(get_output_dimensions(),
    //                                              (dimensions){3, 3, get_output_dimensions()[2], 128},
    //                                              activation_function,
    //                                              convolution_stride_dimensions,
    //                                              convolution_type,
    //                                              "Convolutional layer 3"));

    // // // layers[5]->set_parameters_constant(1);

    // add_layer(make_unique<ConvolutionalLayer>(get_output_dimensions(),
    //                                              (dimensions){1, 1, get_output_dimensions()[2], 64},
    //                                              activation_function,
    //                                              convolution_stride_dimensions,
    //                                              convolution_type,
    //                                              "Convolutional layer 4"));

    // // // layers[6]->set_parameters_constant(1);

    add_layer(make_unique<ConvolutionalLayer>(get_output_dimensions(),
                                                 (dimensions){3, 3, get_output_dimensions()[2], 128},
                                                 activation_function,
                                                 convolution_stride_dimensions,
                                                 convolution_type,
                                                 "Convolutional layer 5"));

    // layers[7]->set_parameters_constant(1);

    add_layer(make_unique<PoolingLayer>(get_output_dimensions(),
                                           pool_dimensions,
                                           pooling_stride_dimensions,
                                           padding_dimensions,
                                           pooling_method,
                                           "Pooling layer 3"));

    // add_layer(make_unique<ConvolutionalLayer>(get_output_dimensions(),
    //                                              (dimensions){3, 3, get_output_dimensions()[2], 256},
    //                                              activation_function,
    //                                              convolution_stride_dimensions,
    //                                              convolution_type,
    //                                              "Convolutional layer 6"));

    // // // layers[9]->set_parameters_constant(1);

    // add_layer(make_unique<ConvolutionalLayer>(get_output_dimensions(),
    //                                              (dimensions){1, 1, get_output_dimensions()[2], 128},
    //                                              activation_function,
    //                                              convolution_stride_dimensions,
    //                                              convolution_type,
    //                                              "Convolutional layer 7"));

    // // // layers[10]->set_parameters_constant(1);

    add_layer(make_unique<ConvolutionalLayer>(get_output_dimensions(),
                                                 (dimensions){3, 3, get_output_dimensions()[2], 256},
                                                 activation_function,
                                                 convolution_stride_dimensions,
                                                 convolution_type,
                                                 "Convolutional layer 8"));

    // layers[11]->set_parameters_constant(1);

    add_layer(make_unique<PoolingLayer>(get_output_dimensions(),
                                           pool_dimensions,
                                           pooling_stride_dimensions,
                                           padding_dimensions,
                                           pooling_method,
                                           "Pooling layer 4"));


    add_layer(make_unique<ConvolutionalLayer>(get_output_dimensions(),
                                                 (dimensions){3, 3, get_output_dimensions()[2], 512},
                                                 activation_function,
                                                 convolution_stride_dimensions,
                                                 convolution_type,
                                                 "Convolutional layer 9"));

    // layers[13]->set_parameters_constant(1);

    // add_layer(make_unique<ConvolutionalLayer>(get_output_dimensions(),
    //                                              (dimensions){1, 1, get_output_dimensions()[2], 256},
    //                                              activation_function,
    //                                              convolution_stride_dimensions,
    //                                              convolution_type,
    //                                              "Convolutional layer 10"));

    // // // layers[14]->set_parameters_constant(1);

    // add_layer(make_unique<ConvolutionalLayer>(get_output_dimensions(),
    //                                              (dimensions){3, 3, get_output_dimensions()[2], 512},
    //                                              activation_function,
    //                                              convolution_stride_dimensions,
    //                                              convolution_type,
    //                                              "Convolutional layer 11"));

    // // // layers[15]->set_parameters_constant(1);

    // add_layer(make_unique<ConvolutionalLayer>(get_output_dimensions(),
    //                                              (dimensions){1, 1, get_output_dimensions()[2], 256},
    //                                              activation_function,
    //                                              convolution_stride_dimensions,
    //                                              convolution_type,
    //                                              "Convolutional layer 12"));

    // // // layers[16]->set_parameters_constant(1);

    // add_layer(make_unique<ConvolutionalLayer>(get_output_dimensions(),
    //                                              (dimensions){3, 3, get_output_dimensions()[2], 512},
    //                                              activation_function,
    //                                              convolution_stride_dimensions,
    //                                              convolution_type,
    //                                              "Convolutional layer 13"));

    // // // layers[17]->set_parameters_constant(1);

    add_layer(make_unique<PoolingLayer>(get_output_dimensions(),
                                           pool_dimensions,
                                           pooling_stride_dimensions,
                                           padding_dimensions,
                                           pooling_method,
                                           "Pooling layer 5"));

    add_layer(make_unique<ConvolutionalLayer>(get_output_dimensions(),
                                                 (dimensions){3, 3, get_output_dimensions()[2], 1024},
                                                 activation_function,
                                                 convolution_stride_dimensions,
                                                 convolution_type,
                                                 "Convolutional layer 14"));

    // // layers[19]->set_parameters_constant(1);

    // add_layer(make_unique<ConvolutionalLayer>(get_output_dimensions(),
    //                                              (dimensions){1, 1, get_output_dimensions()[2], 512},
    //                                              activation_function,
    //                                              convolution_stride_dimensions,
    //                                              convolution_type,
    //                                              "Convolutional layer 15"));

    // // // layers[20]->set_parameters_constant(1);

    // add_layer(make_unique<ConvolutionalLayer>(get_output_dimensions(),
    //                                              (dimensions){3, 3, get_output_dimensions()[2], 1024},
    //                                              activation_function,
    //                                              convolution_stride_dimensions,
    //                                              convolution_type,
    //                                              "Convolutional layer 16"));

    // // // layers[21]->set_parameters_constant(1);

    // add_layer(make_unique<ConvolutionalLayer>(get_output_dimensions(),
    //                                              (dimensions){1, 1, get_output_dimensions()[2], 512},
    //                                              activation_function,
    //                                              convolution_stride_dimensions,
    //                                              convolution_type,
    //                                              "Convolutional layer 17"));

    // // // layers[22]->set_parameters_constant(1);

    // add_layer(make_unique<ConvolutionalLayer>(get_output_dimensions(),
    //                                              (dimensions){3, 3, get_output_dimensions()[2], 1024},
    //                                              activation_function,
    //                                              convolution_stride_dimensions,
    //                                              convolution_type,
    //                                              "Convolutional layer 18"));

    // // // layers[23]->set_parameters_constant(1);

    // add_layer(make_unique<ConvolutionalLayer>(get_output_dimensions(),
    //                                           (dimensions){3, 3, get_output_dimensions()[2], 1024},
    //                                           activation_function,
    //                                           convolution_stride_dimensions,
    //                                           convolution_type,
    //                                           "Convolutional layer 19"));

    // // // layers[24]->set_parameters_constant(1)

    // add_layer(make_unique<ConvolutionalLayer>(get_output_dimensions(),
    //                                           (dimensions){3, 3, get_output_dimensions()[2], 1024},
    //                                           activation_function,
    //                                           convolution_stride_dimensions,
    //                                           convolution_type,
    //                                           "Convolutional layer 20"));

    // // // layers[25]->set_parameters_constant(1)

    // add_layer(make_unique<ConvolutionalLayer>(get_output_dimensions(),
    //                                           (dimensions){3, 3, get_output_dimensions()[2], 1024},
    //                                           activation_function,
    //                                           convolution_stride_dimensions,
    //                                           convolution_type,
    //                                           "Convolutional layer 21"));

    // // // layers[26]->set_parameters_constant(1)

    add_layer(make_unique<ConvolutionalLayer>(get_output_dimensions(),
                                                 (dimensions){1, 1, get_output_dimensions()[2], 125},
                                                 ConvolutionalLayer::ActivationFunction::RectifiedLinear,
                                                 convolution_stride_dimensions,
                                                 convolution_type,
                                                 "Convolutional layer 22"));

    // layers[27]->set_parameters_constant(1);

    add_layer(make_unique<DetectionLayer>(get_output_dimensions(),
                                          anchors,
                                          "Detection layer"));

}

void YoloNetwork::set_dropout_rate(const type& new_dropout_rate)
{
    dropout_rate = new_dropout_rate;
}

Index YoloNetwork::get_classes_number()
{
    return classes_number = get_output_dimensions()[2] / anchors.size() - 5;
}

Tensor<type, 4> YoloNetwork::calculate_outputs(const Tensor<type, 4>& inputs)
{
    const Index layers_number = get_layers_number();

    if (layers_number == 0)
        return Tensor<type, 4>();

    const Index batch_samples_number = inputs.dimension(0);
    //const Index inputs_number = inputs.dimension(1);

    ForwardPropagation forward_propagation(batch_samples_number, this);

    const pair<type*, dimensions> input_pair((type*)inputs.data(), {{batch_samples_number, inputs.dimension(1), inputs.dimension(2), inputs.dimension(3)}});

    forward_propagate({input_pair}, forward_propagation);

    // forward_propagation.print();

    const pair<type*, dimensions> outputs_pair
        = forward_propagation.layers[layers_number - 1]->get_outputs_pair();

    return tensor_map_4(outputs_pair);

}

void YoloNetwork::load_yolo(const string& path)
{
    cout << "Loading YOLO model..." << endl;

    load(path);
}

}
