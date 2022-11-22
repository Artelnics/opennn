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

void RegionProposalLayer::set_filename(const string& new_filename)
{
    filename = new_filename;
}

const Tensor<type, 4> RegionProposalLayer::get_input_regions()
{
    Tensor<type, 4> input_regions;

//    const Tensor<Tensor<type, 1>, 1> image_data = read_bmp_image_data(filename);

//    for (Index i = 0; i < regions_number; i ++)
//    {
//        const Tensor<type, 1> proposed_region = propose_random_region(image_data);
//    }

    return input_regions;
}


void RegionProposalLayer::forward_propagate(type* inputs_data,
                          const Tensor<Index,1>& inputs_dimensions,
                          LayerForwardPropagation* forward_propagation)
{

    const Index channels_number = 3;

    const Index images_number = 1;
    const Index input_rows = 0;
    const Index input_columns = 0;

//    Tensor<type, 4> inputs(images_number, channels_number, region_rows, region_columns);

//    const TensorMap<Tensor<type, 4>> inputs(inputs_data, inputs_dimensions(0), inputs_dimensions(1), inputs_dimensions(2), inputs_dimensions(3));

    const Tensor<Tensor<type, 1>, 1> input_image = read_bmp_image_data(filename);
    // Propose random region for each image

    Tensor<type, 4> outputs(regions_number, channels_number, region_rows, region_columns);

    Index image_index = 0;

    for (Index region_index = 0; region_index < regions_number; region_index ++)
    {
        Tensor<type, 1> proposed_region = propose_single_random_region(input_image, region_columns, region_rows);

        for (Index channel_index = 0; channel_index < channels_number; channel_index ++)
        {
            for (Index rows_index = 0; rows_index < region_rows; rows_index ++)
            {
                for (Index columns_index = 0; columns_index < region_columns; columns_index ++)
                {
                    outputs(region_index, channel_index, rows_index, columns_index) = proposed_region(image_index);
                    image_index++;
                }
            }
        }
    }





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
