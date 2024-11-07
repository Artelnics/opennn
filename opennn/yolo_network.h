#ifndef YOLO_NETWORK_H
#define YOLO_NETWORK_H

#include <string>
#include <sstream>

#include "neural_network.h"
#include "neural_network_forward_propagation.h"

namespace opennn
{

struct YoloForwardPropagation;
struct YoloBackPropagation;

class YoloNetwork : public NeuralNetwork
{
public:

    // Constructors

    explicit YoloNetwork();

    explicit YoloNetwork(const dimensions&);

    void set(const dimensions&);

    void set(/*const Index& batch_size,*/ const Index& height, const Index& width, const Index& channels);

    // void set_dropout_rate(const type&);

    Tensor<type, 4> calculate_outputs(const Tensor<type, 4>&);

    void load_yolo(const string&);

protected:

    string name = "yolov2";

    Index input_dimensions;

    Index image_width;

    Index image_height;

    Index image_channels;

    // Index batch_size;

    // Index layers_number;

    type dropout_rate = 0;

};


struct YoloForwardPropagation : ForwardPropagation
{
    // Constructors

    YoloForwardPropagation() {}

    YoloForwardPropagation(const Index& new_batch_samples, NeuralNetwork* new_neural_network)
    {
        set(new_batch_samples, new_neural_network);
    }

    void set(const Index& new_batch_samples, NeuralNetwork* new_neural_network);

    void print() const;

    Index batch_samples_number = 0;

    Tensor<unique_ptr<LayerForwardPropagation>, 1> layers;
};
};

#endif // YOLO_NETWORK_H
