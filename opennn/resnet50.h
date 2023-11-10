#ifndef RESNET50_H
#define RESNET50_H

#include "neural_network.h"

using namespace opennn;
using namespace std;


class Resnet50 : public NeuralNetwork
{
public:

    Resnet50();

    ~Resnet50();

private:


    ConvolutionalLayer convolutional_layer_0;
    PoolingLayer pooling_layer_0;

    // Residual block 1

    ConvolutionalLayer convolutional_layer_1_1_a;
    PoolingLayer pooling_layer_1_a;

    ConvolutionalLayer convolutional_layer_1_2_a;
    PoolingLayer pooling_layer_1_2_a;

    ConvolutionalLayer convolutional_layer_1_3_a;
    PoolingLayer pooling_layer_1_3_a;

    ConvolutionalLayer convolutional_layer_1_1_b;
    PoolingLayer pooling_layer_1_1_b;


/*
    // Residual block 2

    ConvolutionalLayer convolutional_layer_5;
    PoolingLayer pooling_layer_3;

    // Residual block 1

    ConvolutionalLayer convolutional_layer_8;
    PoolingLayer pooling_layer_4;

    // Block 5

    ConvolutionalLayer convolutional_layer_11;
    PoolingLayer pooling_layer_5;
*/
    FlattenLayer flatten_layer;

    ProbabilisticLayer probabilistic_layer;

};

#endif
