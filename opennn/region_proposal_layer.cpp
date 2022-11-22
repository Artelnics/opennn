//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   R E G I O N   P R O P O S A L   L A Y E R   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "region_proposal_layer.h"
#include "opennn_images.h"


namespace opennn
{

/// Default constructor.
/// It creates a empty layer object.
/// This constructor also initializes the rest of the class members to their default values.

RegionProposalLayer::RegionProposalLayer() : Layer()
{
}

/*
BoundingBox RegionProposalLayer::propose_random_region(const Tensor<unsigned char, 1>& image, const string& filename)
{
    const Index channels_number = get_channels_number();
    const Index image_height = get_image_height();
    const Index image_width = get_image_width();

    Index x_center = rand() % image_width;
    Index y_center = rand() % image_height;

    Index x_top_left;
    Index y_top_left;

    if(x_center == 0){x_top_left = 0;}else{x_top_left = rand() % x_center;}
    if(y_center == 0){y_top_left = 0;} else{y_top_left = rand() % y_center;}

    Index x_bottom_right;

    if(x_top_left == 0){x_bottom_right = rand()%(image_width - (x_center + 1) + 1) + (x_center + 1);}
    else{x_bottom_right = rand()%(image_width - x_center + 1) + x_center;}

    Index y_bottom_right;

    if(y_top_left == 0){y_bottom_right = rand()%(image_height - (y_center + 1) + 1) + (y_center + 1);}
    else{y_bottom_right = rand() % (image_height - y_center + 1) + y_center;}

    BoundingBox random_region(channels_number, x_top_left, y_top_left, x_bottom_right, y_bottom_right);

    random_region.data = get_bounding_box(image, x_top_left, y_top_left, x_bottom_right, y_bottom_right);

    return random_region;
}
*/

void RegionProposalLayer::forward_propagate(type* inputs_data,
                          const Tensor<Index,1>& inputs_dimensions,
                          LayerForwardPropagation* forward_propagation)
{

    const Index channels_number = 2000;

    const Index images_number = 1;
    const Index input_rows = 0;
    const Index input_columns = 0;


    const Index regions_number = 2000;
    const Index region_rows = 227;
    const Index region_columns = 227;

    Tensor<type, 4> inputs(images_number, channels_number, region_rows, region_columns);

    // Propose random region for each image

    Tensor<type, 4> outputs(regions_number, channels_number, region_rows, region_columns);


//    BatchNormalizationLayerForwardPropagation* batch_norm_layer_forward_propagation
//            = static_cast<BatchNormalizationLayerForwardPropagation*>(forward_propagation);

//    const TensorMap<Tensor<type, 2>> inputs(inputs_data, inputs_dimensions(0), inputs_dimensions(1));

//    Tensor<type, 2> inputs_normalized = perform_inputs_normalization(inputs, batch_norm_layer_forward_propagation);

//    calculate_combinations(inputs_normalized,
//                           synaptic_weights,
//                           batch_norm_layer_forward_propagation->outputs_data);

}

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2022 Artificial Intelligence Techniques, SL.
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
