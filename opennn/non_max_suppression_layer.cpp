//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   N O N   M A X   S U P R E S S I O N   L A Y E R   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "non_max_suppression_layer.h"
#include "tensors.h"
#include "yolo_dataset.h"

namespace opennn
{

/// Default constructor.
/// It creates a empty layer object.
/// This constructor also initializes the rest of the class members to their default values.

NonMaxSuppressionLayer::NonMaxSuppressionLayer(const dimensions& new_input_dimensions,
                                               const Index& new_boxes_per_cell,
                                               const string name) : Layer()
{
    layer_type = Type::NonMaxSuppression;

    set(new_input_dimensions, new_boxes_per_cell, name);
}

void NonMaxSuppressionLayer::set(const dimensions& new_input_dimensions, const Index& new_boxes_per_cell, const string name)
{
    if(new_input_dimensions.size() != 3)
        throw runtime_error("Dimensions must be 3");

    input_dimensions = new_input_dimensions;
    boxes_per_cell = new_boxes_per_cell;
    classes_number = (new_input_dimensions[2] / boxes_per_cell) - output_box_info;
    grid_size = new_input_dimensions[0];

    set_name(name);
}

void NonMaxSuppressionLayer::forward_propagate(const vector<pair<type*, dimensions>>& input_pairs,
                                               unique_ptr<LayerForwardPropagation>& layer_forward_propagation,
                                               const bool&)
{

    const TensorMap<Tensor<type, 4>> inputs = tensor_map_4(input_pairs[0]);

    const Index batch_size = inputs.dimension(0);

    NonMaxSuppressionLayerForwardPropagation* non_max_suppression_layer_forward_propagation
            = static_cast<NonMaxSuppressionLayerForwardPropagation*>(layer_forward_propagation.get());

    Tensor<type, 3>& outputs = non_max_suppression_layer_forward_propagation->outputs;

    Tensor<type, 0>& maximum_box_number = non_max_suppression_layer_forward_propagation->maximum_box_number;

    calculate_boxes(inputs, outputs, batch_size, maximum_box_number);
}

void NonMaxSuppressionLayer::calculate_boxes(const Tensor<type, 4>& network_output, Tensor<type, 3>& outputs, const Index& batch_size, Tensor<type, 0>& maximum_box_number)
{
    Tensor<type, 4> converted_network_output(batch_size, grid_size, grid_size, boxes_per_cell * final_box_info);    // I change p = object_prob to p' = p * maximum_class_probability and (c_0,...,c_19) to c = class_number

    vector<vector<Tensor<type, 1>>> input_bounding_boxes(batch_size);
    vector<vector<Tensor<type, 1>>> final_boxes(batch_size);
    Tensor<type, 3> tensor_final_boxes(batch_size, boxes_per_cell * grid_size * grid_size, 1 + final_box_info);
    vector<vector<vector<Tensor<type, 1>>>> classified_boxes(batch_size);
    Tensor<type, 1> box(final_box_info);

    Tensor<type, 1> boxes_number(batch_size);
    boxes_number.setZero();

    for(Index i = 0; i < batch_size; i++)
    {
        classified_boxes[i].resize(classes_number);

        for(Index j = 0; j < grid_size; j++)
        {
            for(Index k = 0; k < grid_size; k++)
            {
                for(Index b = 0; b < boxes_per_cell; b++)
                {
                    Index object_class = 0;
                    const Tensor<type, 0> max_class_probability = network_output.slice(Eigen::array<Index, 4>{0, 0, 0, b * (output_box_info + classes_number) + 5},
                                                                                 Eigen::array<Index, 4>{1, 1, 1, classes_number}).maximum().eval();

                    while(network_output(i, j, k, b * (output_box_info + classes_number) + 5 + object_class) != max_class_probability() && object_class < 19)
                        object_class++;

                    if(network_output(i, j, k, b * (output_box_info + classes_number) + 5 + object_class) != max_class_probability())
                        cerr<<"Error calculating the class probabilities";

                    converted_network_output(i, j, k, b * final_box_info + 0) = network_output(i, j, k, b * (output_box_info + classes_number) + 0);
                    converted_network_output(i, j, k, b * final_box_info + 1) = network_output(i, j, k, b * (output_box_info + classes_number) + 1);
                    converted_network_output(i, j, k, b * final_box_info + 2) = network_output(i, j, k, b * (output_box_info + classes_number) + 2);
                    converted_network_output(i, j, k, b * final_box_info + 3) = network_output(i, j, k, b * (output_box_info + classes_number) + 3);
                    converted_network_output(i, j, k, b * final_box_info + 4) = network_output(i, j, k, b * (output_box_info + classes_number) + 4) * max_class_probability();
                    converted_network_output(i, j, k, b * final_box_info + 5) = object_class;


                    if(network_output(i, j, k, b * (output_box_info + classes_number) + 4) > 0.5)
                    {
                        for(Index l = 0; l < final_box_info; l++)
                            box(l) = converted_network_output(i, j, k, b * (final_box_info) + l);

                        input_bounding_boxes[i].push_back(box);
                    }
                }
            }
        }

        for(size_t j = 0; j < input_bounding_boxes[i].size(); j++)
        {
            classified_boxes[i][input_bounding_boxes[i][j](5)].push_back(input_bounding_boxes[i][j]);
        }

        for(Index k = 0; k < classes_number; k++)
        {
            if(classified_boxes[i][k].empty())
                continue;

            sort(classified_boxes[i][k].begin(), classified_boxes[i][k].end(), [](const Tensor<type, 1>& a, const Tensor<type, 1>& b){
                return a(4) > b(4);
            });


            while(!classified_boxes[i][k].empty())
            {
                const Tensor<type, 1> box = classified_boxes[i][k].front();
                classified_boxes[i][k].erase(classified_boxes[i][k].begin());

                for(auto it = classified_boxes[i][k].begin(); it != classified_boxes[i][k].end();)
                    if(calculate_intersection_over_union(box, *it) > overlap_threshold)
                        it = classified_boxes[i][k].erase(it);
                    else
                        it++;

                final_boxes[i].push_back(box);
            }
        }

        // @todo use memcpy

        // memcpy(outputs[i].data(), final_boxes[i].data(), tensor_final_boxes[i].size()*sizeof(type));     // @TODO check if I'm doing it right




        for(size_t box = 0; box < final_boxes[i].size(); box++)
        {
            for(Index element = 0; element < final_box_info; element++)
            {
                tensor_final_boxes(i, boxes_number(i), element) = final_boxes[i][box](element);
            }

            boxes_number(i)++;
        }

    }



    maximum_box_number = boxes_number.maximum();

    outputs.resize(batch_size, (Index) maximum_box_number(), 7);

    outputs = tensor_final_boxes.slice(Eigen::array<Index, 3> {0,0},
                                       Eigen::array<Index, 3> {batch_size,  (Index) maximum_box_number(), final_box_info});
}

dimensions NonMaxSuppressionLayer::get_input_dimensions() const
{
    return input_dimensions;
}

dimensions NonMaxSuppressionLayer::get_output_dimensions() const
{
    return {grid_size * grid_size * boxes_per_cell, final_box_info};
}

void NonMaxSuppressionLayer::print() const
{
    cout << "NonMaxSuppression layer" << endl;
    cout << "Input dimensions: " << endl;
    print_vector(input_dimensions);
    cout << "Maximum output dimensions: " << endl;
    print_vector(get_output_dimensions());
}


NonMaxSuppressionLayerForwardPropagation::NonMaxSuppressionLayerForwardPropagation(
    const Index& new_batch_samples_number, Layer* new_layer)
    : LayerForwardPropagation()
{
    set(new_batch_samples_number, new_layer);
}

void NonMaxSuppressionLayerForwardPropagation::set(const Index& new_batch_samples_number, Layer* new_layer)
{
    batch_samples_number = new_batch_samples_number;

    layer = new_layer;

    // outputs.resize(batch_samples_number, 7);
}


pair<type*, dimensions> NonMaxSuppressionLayerForwardPropagation::get_outputs_pair() const
{
    return {(type*)outputs.data(), {batch_samples_number, (Index) maximum_box_number(), 6}};
}

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2023 Artificial Intelligence Techniques, SL.
//
// This library is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 2.1 of the License, or any later version.
//
// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.

// You should have received a copy of the GNU Lesser General Public
// License along with this library; if not, write to the Free Software
// Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
