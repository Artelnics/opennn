#ifndef YOLO_NETWORK_H
#define YOLO_NETWORK_H

#include <string>
#include <sstream>

#include "neural_network.h"
#include "forward_propagation.h"

namespace opennn
{

class YoloNetwork : public NeuralNetwork
{
public:

    // Constructors

    explicit YoloNetwork();

    explicit YoloNetwork(const dimensions&, const vector<Tensor<type, 1>>&);

    void set(const dimensions&, const vector<Tensor<type, 1>>&);

    void set(const Index&, const Index&, const Index&);

    void set_dropout_rate(const type&);

    Index get_classes_number();

    Tensor<type, 4> calculate_outputs(const Tensor<type, 4>&);

    void load_yolo(const string&);

protected:

    string name = "yolov2";

    Index input_dimensions;

    Index image_width;

    Index image_height;

    Index image_channels;

    vector<Tensor<type, 1>> anchors;

    Index classes_number;

    // Index batch_size;

    // Index layers_number;

    type dropout_rate = 0;

};

};

#endif // YOLO_NETWORK_H
