#ifndef VGG16_H
#define VGG16_H

#include "neural_network.h"

using namespace opennn;
using namespace std;


class VGG16 : public NeuralNetwork
{
public:
    VGG16();

    ~VGG16();

private:

    // Block 1

    ConvolutionalLayer convolutional_layer_1;
    ConvolutionalLayer convolutional_layer_2;
    PoolingLayer pooling_layer_1;

    // Block 2

    ConvolutionalLayer convolutional_layer_3;
    ConvolutionalLayer convolutional_layer_4;
    PoolingLayer pooling_layer_2;

    // Block 3

    ConvolutionalLayer convolutional_layer_5;
    ConvolutionalLayer convolutional_layer_6;
    ConvolutionalLayer convolutional_layer_7;
    PoolingLayer pooling_layer_3;

    // Block 4

    ConvolutionalLayer convolutional_layer_8;
    ConvolutionalLayer convolutional_layer_9;
    ConvolutionalLayer convolutional_layer_10;
    PoolingLayer pooling_layer_4;

    // Block 5

    ConvolutionalLayer convolutional_layer_11;
    ConvolutionalLayer convolutional_layer_12;
    ConvolutionalLayer convolutional_layer_13;
    PoolingLayer pooling_layer_5;
    FlattenLayer flatten_layer;
    PerceptronLayer perceptron_layer_1;
    PerceptronLayer perceptron_layer_2;

    ProbabilisticLayer probabilistic_layer;

};

#endif
