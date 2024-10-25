//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   N O N   M A X   S U P R E S S I O N   L A Y E R   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "non_max_suppression_layer.h"

namespace opennn
{

/// Default constructor.
/// It creates a empty layer object.
/// This constructor also initializes the rest of the class members to their default values.

NonMaxSuppressionLayer::NonMaxSuppressionLayer() : Layer()
{
    layer_type = Type::NonMaxSuppression;

    name = "non_max_suppression_layer";
}


void NonMaxSuppressionLayer::calculate_regions(type* inputs_data, const Tensor<Index, 1>& inputs_dimensions,
                                               type* outputs_data, const Tensor<Index, 1>& outputs_dimensions)
{
    // inputs_data -> Score of each input image bounding box
    //             -> 4 parameters defining the bbox
    // outputs_data -> Bounding box that surpasses our criteria

    const type overlap_threshold = type(0.65);
    const type confidence_score_threshold = type(0.4);

    const Index regions_number = inputs_dimensions(0);
    TensorMap<Tensor<type, 2>> inputs(inputs_data, inputs_dimensions(0), inputs_dimensions(1));

    // INPUTS MATRIX

    Tensor<Tensor<type, 1>, 1> region_inputs(regions_number); // x_center, y_center, width, height ¿(from region_proposal_layer)?
    Tensor<type, 1> inputs_confidence_score; // confidence_score

    Tensor<bool, 1> mask_regions(regions_number);

    Index higher_score_regions_number = 0;

    for(Index i = 0; i < regions_number; i++)
    {
        if(inputs_confidence_score(i) > confidence_score_threshold)
        {
            mask_regions(i) = true;
            higher_score_regions_number++;
        }
        else
        {
            mask_regions(i) = false;
        }
    }

    // Fill filtered input tensor

    Tensor<Tensor<type, 1>, 1> filtered_proposals(higher_score_regions_number);
    Tensor<type, 1> filtered_proposals_score(higher_score_regions_number);

    Index filtered_proposals_index = 0;

    for(Index i = 0; i < regions_number; i++)
    {
        if(mask_regions(i))
        {
            filtered_proposals(filtered_proposals_index) = region_inputs(i);
            filtered_proposals_score(filtered_proposals_index) = inputs_confidence_score(i);
            filtered_proposals_index++;
        }
    }

    // Calculate IoU between regions

    Tensor<bool, 1> final_detections_boolean(higher_score_regions_number);
    Index final_detections_number = 0;

    for(Index i = 0; i < higher_score_regions_number; i++)
    {
        const Index x_center_box_1 = filtered_proposals(i)(0);
        const Index y_center_box_1 = filtered_proposals(i)(1);
        const Index width_box_1 = filtered_proposals(i)(2);
        const Index height_right_box_1 = filtered_proposals(i)(3);


        for(Index j = i; j < higher_score_regions_number; j++)
        {
            if(i < j)
            {
                const Index x_center_box_2 = filtered_proposals(j)(0);
                const Index y_center_box_2 = filtered_proposals(j)(1);
                const Index width_box_2 = filtered_proposals(j)(2);
                const Index height_box_2 = filtered_proposals(j)(3);
/*
                const type intersection_over_union_between_boxes = intersection_over_union(x_top_left_box_1, y_top_left_box_1,
                                                                                           x_bottom_right_box_1, y_bottom_right_box_1,
                                                                                           x_top_left_box_2, y_top_left_box_2,
                                                                                           x_bottom_right_box_2, y_bottom_right_box_2);
                if(intersection_over_union_between_boxes > overlap_threshold)
                {
                    final_detections_boolean(i) = true;
                    final_detections_number++;
                }
                else
                {
                    final_detections_boolean(i) = false;
                }
*/
            }
        }
    }

    Tensor<Tensor<type, 1>, 1> outputs(final_detections_number);

    Tensor<type, 1> outputs_score(final_detections_number);

    Index final_detections_index = 0;

    for(Index i = 0; i < higher_score_regions_number; i++)
    {
        if(final_detections_boolean(i))
        {
            outputs(final_detections_index) = filtered_proposals(i);
            outputs_score(final_detections_index) = filtered_proposals_score(i);
            final_detections_index++;
        }
    }
}


void NonMaxSuppressionLayer::forward_propagate(Tensor<type*, 1> inputs_data,
                                               const Tensor<Tensor<Index,1>, 1>& inputs_dimensions,
                                               LayerForwardPropagation* forward_propagation,
                                               const bool& is_training)
{
    NonMaxSuppressionLayerForwardPropagation* non_max_suppression_layer_forward_propagation
            = static_cast<NonMaxSuppressionLayerForwardPropagation*>(forward_propagation);

    // Propose random region for each image (NON NECESSARY FOR YOLO)

//    Tensor<type, 2> outputs(regions_number, channels_number * region_rows * region_columns);

//    const Tensor<Index, 1> outputs_dimensions = get_dimensions(non_max_suppression_layer_forward_propagation->outputs);

//    calculate_regions(inputs_data,
//                      inputs_dimensions,
//                      non_max_suppression_layer_forward_propagation->outputs_data(0),
//                      outputs_dimensions);
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
