#ifndef DETECTION_LAYER_H
#define DETECTION_LAYER_H

#include <iostream>
#include <string>

#include "config.h"
#include "layer.h"
#include "layer_forward_propagation.h"

namespace opennn
{

class DetectionLayer : public Layer
{
public:
    explicit DetectionLayer(const dimensions&,                      // Input dimensions (output of the last convolutional)
                            const Index& = 5,                       // Number of boxes per grid cell
                            const string = "detection_layer");

    void set(const dimensions&,
             const Index&,
             const string = "detection_layer");

    void forward_propagate(const vector<pair<type*, dimensions>>&,
                           unique_ptr<LayerForwardPropagation>&,
                           const bool&) final;


    dimensions get_output_dimensions() const;

    Index get_height() const;
    Index get_width() const;
    Index get_channels() const;

    void apply_detection(const Tensor<type, 4>&, Tensor<type, 4>&, const Index&);

    void print() const;

protected:
    dimensions input_dimensions;
    Index boxes_per_cell;
    Index box_info = 5;         //x_center, y_center, width, height and object_confidence
    Index classes_number = 20;  //for the VOC2007 dataset
    Index grid_size;
};

struct DetectionLayerForwardPropagation : LayerForwardPropagation
{
    explicit DetectionLayerForwardPropagation(const Index& = 0, Layer* = nullptr);

    pair<type *, dimensions> get_outputs_pair() const final;

    void set(const Index& = 0, Layer* = nullptr) final;

    void print() const;

    Tensor<type, 4> outputs;
};

}

#endif // DETECTION_LAYER_H
