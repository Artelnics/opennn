#ifndef DETECTION_LAYER_H
#define DETECTION_LAYER_H

#include <iostream>
#include <string>

#include "config.h"
#include "layer.h"
#include "layer_forward_propagation.h"
#include "layer_back_propagation.h"
#include "layer_back_propagation_lm.h"

namespace opennn
{

//@TODO ask about the dimensions of the activation derivatives and the rest of dimensions I don't know about

struct DetectionLayerForwardPropagation : LayerForwardPropagation
{
    explicit DetectionLayerForwardPropagation(const Index& = 0, Layer* = nullptr);

    pair<type *, dimensions> get_outputs_pair() const final;

    void set(const Index& = 0, Layer* = nullptr) final;

    void print() const;

    Tensor<type, 4> outputs;
    // Tensor<type, 2> activation_derivatives;
};

class DetectionLayer : public Layer
{
    explicit DetectionLayer(const dimensions&,                      // Input dimensions (output of the last convolutional)
                            const Index& = 5,                       // Number of boxes per grid cell
                            const string = "detection_layer");

    void set(const dimensions&,
             const Index&,
             const string = "detection_layer");

};
}

#endif // DETECTION_LAYER_H
