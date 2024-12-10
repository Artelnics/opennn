#include <iostream>

#include "strings_utilities.h"
#include "tensors.h"
#include "detection_layer.h"

namespace opennn
{
DetectionLayerForwardPropagation::DetectionLayerForwardPropagation(
    const Index& new_batch_samples_number, Layer *new_layer)
    : LayerForwardPropagation()
{
    set(new_batch_samples_number, new_layer);
}

DetectionLayer::DetectionLayer(const dimensions& new_input_dimensions,
                               const vector<Tensor<type, 1>>& new_anchors,
                               const string name)
{
    layer_type = Layer::Type::Detection;

    // cout<<"==============="<<endl;

    set(new_input_dimensions, new_anchors, name);
}

void DetectionLayer::set(const dimensions& new_input_dimensions, const vector<Tensor<type, 1>>& new_anchors, const string name)
{
    if(new_input_dimensions.size() != 3)
        throw runtime_error("Dimensions must be 3");

    input_dimensions = new_input_dimensions;
    anchors = new_anchors;
    boxes_per_cell = new_anchors.size();
    classes_number = (new_input_dimensions[2] / boxes_per_cell) - box_info;
    grid_size = new_input_dimensions[0];


    // cout<<"==============="<<endl;

    set_name(name);
    // cout<<"==============="<<endl;
}

void DetectionLayer::forward_propagate(const vector<pair<type*, dimensions>>& input_pairs,
                                       unique_ptr<LayerForwardPropagation>& layer_forward_propagation,
                                       const bool&)
{
    const TensorMap<Tensor<type, 4>> inputs = tensor_map_4(input_pairs[0]);

    const Index batch_size = inputs.dimension(0);

    DetectionLayerForwardPropagation* detection_layer_forward_propagation =
        static_cast<DetectionLayerForwardPropagation*>(layer_forward_propagation.get());

    Tensor<type, 4>& outputs = detection_layer_forward_propagation->outputs;

    apply_detection(inputs, outputs, batch_size);
}

dimensions DetectionLayer::get_input_dimensions() const
{
    const Index rows_number = get_height();
    const Index columns_number = get_width();
    const Index channels_number = get_channels();

    return { rows_number, columns_number, channels_number };
}

dimensions DetectionLayer::get_output_dimensions() const
{
    const Index rows_number = get_height();
    const Index columns_number = get_width();
    const Index channels_number = get_channels();

    return { rows_number, columns_number, channels_number };
}

Index DetectionLayer::get_height() const
{
    return input_dimensions[0];
}

Index DetectionLayer::get_width() const
{
    return input_dimensions[1];
}

Index DetectionLayer::get_channels() const
{
    return input_dimensions[2];
}

void DetectionLayer::apply_detection(const Tensor<type, 4>& inputs, Tensor<type, 4>& detections, const Index& batch_size)
{
    const Eigen::array<Index, 1> softmax_dimension{{3}};

    const Index box_data_size = box_info + classes_number;

    const Eigen::array<Index, 4> range_4{ { batch_size, grid_size, grid_size, 1 }};
    const Eigen::array<Index, 4> expand_softmax_dim{ { 1, 1, 1, classes_number} };

    for(Index box = 0; box < boxes_per_cell; box++)
    {

        // Softmax

        Index class_start = box * box_data_size + box_info;

        Tensor<type, 4> class_probabilities =  inputs.slice(Eigen::array<Index, 4>{0, 0, 0, class_start},
                                                    Eigen::array<Index, 4>{batch_size, grid_size, grid_size, classes_number});


        // cout<<"class probabilities before softmax:\n"<<class_probabilities<<endl;

        class_probabilities.device(*thread_pool_device) = class_probabilities - (class_probabilities.maximum(softmax_dimension)
                                                                                                   .eval()
                                                                                                   .reshape(range_4)
                                                                                                   .broadcast(expand_softmax_dim));

        // cout<<"class probabilities before softmax exponential:\n"<<class_probabilities<<endl;

        class_probabilities.device(*thread_pool_device) = class_probabilities.exp();

        // cout<<"class probabilities after softmax exponential:\n"<<class_probabilities<<endl;

        class_probabilities.device(*thread_pool_device) = class_probabilities / (class_probabilities.sum(softmax_dimension)
                                                                                                   .eval()
                                                                                                   .reshape(range_4)
                                                                                                   .broadcast(expand_softmax_dim));

        // cout<<"class probabilities after softmax :\n"<<class_probabilities<<endl;

        detections.slice(Eigen::array<Index, 4>{0, 0, 0, class_start},
                         Eigen::array<Index, 4>{batch_size, grid_size, grid_size, classes_number}) = class_probabilities;


        // Logistic

        Tensor<type, 4> normalized_centers =  inputs.slice(Eigen::array<Index, 4>{0, 0, 0, box * box_data_size + 0},
                                                           Eigen::array<Index, 4>{batch_size, grid_size, grid_size, 2});

        normalized_centers/*.device(*thread_pool_device)*/ = (type(1) + (-normalized_centers).exp()).inverse();


        detections.slice(Eigen::array<Index, 4>{0, 0, 0,  box * box_data_size + 0},
                         Eigen::array<Index, 4>{batch_size, grid_size, grid_size, 2}) = normalized_centers;


        Tensor<type, 4> object_probability = inputs.slice(Eigen::array<Index, 4>{0, 0, 0, box * box_data_size +  4},
                                                          Eigen::array<Index, 4>{batch_size, grid_size, grid_size, 1});

        object_probability/*.device(*thread_pool_device)*/ = (type(1) + (-object_probability).exp()).inverse();

        detections.slice(Eigen::array<Index, 4>{0, 0, 0,  box * box_data_size + 4},
                         Eigen::array<Index, 4>{batch_size, grid_size, grid_size, 1}) = object_probability;


        // Exponential

        detections.slice(Eigen::array<Index, 4>{0, 0, 0, box * box_data_size + 2},
                         Eigen::array<Index, 4>{batch_size, grid_size, grid_size, 2})
                  // .device(*thread_pool_device)
            =  (inputs.slice(Eigen::array<Index, 4>{0, 0, 0, box * box_data_size + 2},
                            Eigen::array<Index, 4>{batch_size, grid_size, grid_size, 2})).exp();


        // cout<<"Detection layer outputs:\n"<<detections<<endl;

        // cout<<"Anchors:\n"<<anchors[0]<<endl<<anchors[1]<<endl<<anchors[2]<<endl<<anchors[3]<<endl<<anchors[4]<<endl;

        // throw runtime_error("n");


      //  Box dimensions detection

        Tensor<type, 4> box_detections = detections.slice(Eigen::array<Index, 4>{0, 0, 0, box * box_data_size + 0},
                                                          Eigen::array<Index, 4>{batch_size, grid_size, grid_size, 4});

        for(Index i = 0; i < batch_size; i++)
        {
            for(Index j = 0; j < grid_size; j++)
            {
                for(Index k = 0; k < grid_size; k++)
                {
                    box_detections(i,j,k,2) *= anchors[box](0);
                    box_detections(i,j,k,3) *= anchors[box](1);
                }
            }
        }
        detections.slice(Eigen::array<Index, 4>{0, 0, 0, box * box_data_size + 0},
                         Eigen::array<Index, 4>{batch_size, grid_size, grid_size, 4}) = box_detections;
    }

    // cout<<"Detection layer outputs:\n"<<detections<<endl;
    // throw runtime_error("n");
    // cout<<detections.slice(Eigen::array<Index, 4>{0, 0, 0, 5}, Eigen::array<Index, 4>{batch_size, grid_size, grid_size, classes_number})<<endl<<endl<<endl<<endl;
}

void DetectionLayer::print() const
{
    cout << "Detection layer" << endl;
    cout << "Input dimensions: " << endl;
    print_vector(input_dimensions);
    cout << "Output dimensions: " << endl;
    print_vector(get_output_dimensions());
}

void DetectionLayerForwardPropagation::set(const Index& new_batch_samples_number, Layer* new_layer)
{
    batch_samples_number = new_batch_samples_number;

    layer = new_layer;

    const DetectionLayer* detection_layer = static_cast<DetectionLayer*>(layer);

    const Index output_height = detection_layer->get_height();
    const Index output_width = detection_layer->get_width();
    const Index channels = detection_layer->get_channels();

    outputs.resize(batch_samples_number,
                   output_height,
                   output_width,
                   channels);
}

pair<type *, dimensions> DetectionLayerForwardPropagation::get_outputs_pair() const
{
    const DetectionLayer* detection_layer = static_cast<DetectionLayer*>(layer);

    const Index output_height = detection_layer->get_height();
    const Index output_width = detection_layer->get_width();
    const Index channels = detection_layer->get_channels();

    return {(type*)outputs.data(), {batch_samples_number, output_height, output_width, channels}};
}

void DetectionLayerForwardPropagation::print() const
{
    cout<<"Detection layer" << endl
         << "Outputs:" << endl
         << outputs << endl;
}

}
