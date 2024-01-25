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
    /*
/// Default constructor.
/// It creates a empty layer object.
/// This constructor also initializes the rest of the class members to their default values.

RegionProposalLayer::RegionProposalLayer(const Tensor<Index, 1>& new_inputs_dimensions) : Layer()
{
    inputs_dimensions = new_inputs_dimensions;

    layer_type = Type::RegionProposal;

    layer_name = "region_proposal_layer";
}


Tensor<Index, 1> RegionProposalLayer::get_inputs_dimensions() const
{
    return inputs_dimensions;
}


Tensor<Index, 1> RegionProposalLayer::get_outputs_dimensions() const
{
    Tensor<Index, 1> outputs_dimensions(3);

    outputs_dimensions[0] = region_rows;
    outputs_dimensions(1) = region_columns;
    outputs_dimensions(2) = channels_number;

    return outputs_dimensions;
}


Index RegionProposalLayer::get_regions_number() const
{
    return regions_number;
}


Index RegionProposalLayer::get_region_rows() const
{
    return region_rows;
}


Index RegionProposalLayer::get_region_columns() const
{
    return region_columns;
}


Index RegionProposalLayer::get_channels_number() const
{
    return channels_number;
}


void RegionProposalLayer::set_regions_number(const Index& new_regions_number)
{
    regions_number = new_regions_number;
}


void RegionProposalLayer::set_region_rows(const Index& new_region_rows)
{
    region_rows = new_region_rows;
}


void RegionProposalLayer::set_region_columns(const Index& new_region_columns)
{
    region_columns = new_region_columns;
}


void RegionProposalLayer::set_channels_number(const Index& new_channels_number)
{
    channels_number = new_channels_number;
}


void RegionProposalLayer::calculate_regions(type* inputs_data, const Tensor<Index, 1>& inputs_dimensions,
                                            type* regions_data, const Tensor<Index, 1>& regions_dimensions,
                                            type* outputs_data, const Tensor<Index, 1>& outputs_dimensions)
{
    if(inputs_dimensions.size() != 2)
    {
        ostringstream buffer;
        buffer << "OpenNN Exception: RegionProposalLayer class.\n"
               << "void RegionProposalLayer::calculate_regions(type*, const Tensor<Index, 1>&, type*, Tensor<Index, 1>&)"
               << "Inputs dimensions must be equal to 2.\n";
        throw invalid_argument(buffer.str());
    }

    // Propose random region for each image

    TensorMap<Tensor<type, 2>> inputs(inputs_data, inputs_dimensions(0), inputs_dimensions(1));

    Tensor<Tensor<type, 1>, 1> input_image(2);

    Index pixel_values_index = 0;
    Index image_dimensions_index = 0;

    input_image(0).resize(inputs_dimensions(1) - 3);
    input_image(1).resize(3);

    for(Index i = 0; i < inputs_dimensions(1); i++)
    {
        if(i < inputs_dimensions(1) - 3)
        {
            input_image(0)(pixel_values_index) = inputs(0,i);
            pixel_values_index++;
        }
        else
        {
            input_image(1)(image_dimensions_index) = inputs(0,i);
            image_dimensions_index++;
        }
    }

    const Index image_channels_number = input_image(1)(2);

    TensorMap<Tensor<type, 2>> outputs(outputs_data, outputs_dimensions[0], outputs_dimensions(1));
    TensorMap<Tensor<type, 2>> regions(regions_data, regions_dimensions(0), regions_dimensions(1));

    for(Index region_index = 0; region_index < regions_number; region_index++)
    {
        Index image_pixel_index = 0;

        Tensor<Tensor<type, 1>, 1> proposed_region = propose_single_random_region(input_image, region_columns, region_rows);

        regions(region_index, 0) = proposed_region(1)(0); // x_top_left
        regions(region_index, 1) = proposed_region(1)(1); // y_top_left
        regions(region_index, 2) = proposed_region(1)(2); // x_bottom_right
        regions(region_index, 3) = proposed_region(1)(3); // y_bottom_right

        for(Index channel_index = 0; channel_index < image_channels_number; channel_index++)
        {
            for(Index rows_index = 0; rows_index < region_rows; rows_index++)
            {
                for(Index columns_index = 0; columns_index < region_columns; columns_index++)
                {
                    outputs(region_index, image_pixel_index) = proposed_region(0)(image_pixel_index);
                    image_pixel_index++;
                }
            }
        }
    }
}


void RegionProposalLayer::forward_propagate(Tensor<type*, 1> inputs_data,
                          const Tensor<Tensor<Index, 1>, 1>& inputs_dimensions,
                          LayerForwardPropagation* forward_propagation,
                          const bool& is_training)
{

    RegionProposalLayerForwardPropagation* region_proposal_layer_forward_propagation
            = static_cast<RegionProposalLayerForwardPropagation*>(forward_propagation);

    TensorMap<Tensor<type, 4>> inputs(inputs_data, inputs_dimensions(0), inputs_dimensions(1));

    // Propose random region for each image

//    Tensor<type, 2> outputs(regions_number, channels_number * region_rows * region_columns);

//    const Tensor<Index, 1> outputs_dimensions = get_dimensions(region_proposal_layer_forward_propagation->outputs);
//    const Tensor<Index, 1> regions_dimensions = get_dimensions(region_proposal_layer_forward_propagation->outputs_data);

    // Propose random region for each image

//    TensorMap<Tensor<type, 2>> inputs(inputs_data, inputs_dimensions(0), inputs_dimensions(1));

    Tensor<Tensor<type, 1>, 1> input_image(2);

    Index pixel_values_index = 0;
    Index image_dimensions_index = 0;

    input_image(0).resize(inputs_dimensions(1) - 3);
    input_image(1).resize(3);

    for(Index i = 0; i < inputs_dimensions(1); i++)
    {
        if(i < inputs_dimensions(1) - 3)
        {
            input_image(0)(pixel_values_index) = inputs(0,i);
            pixel_values_index++;
        }
        else
        {
            input_image(1)(image_dimensions_index) = inputs(0,i);
            image_dimensions_index++;
        }
    }

    const Index image_channels_number = input_image(1)(2);

    TensorMap<Tensor<type, 2>> outputs(forward_propagation->outputs_data(0), outputs_dimensions[0], outputs_dimensions(1));
    TensorMap<Tensor<type, 2>> regions(region_proposal_layer_forward_propagation->outputs_regions.data(), regions_dimensions(0), regions_dimensions(1));

    Tensor<Tensor<type, 1>, 1> proposed_region;

    for(Index region_index = 0; region_index < regions_number; region_index++)
    {
        Index image_pixel_index = 0;

        proposed_region = propose_single_random_region(input_image, region_columns, region_rows);

        regions(region_index, 0) = proposed_region(1)(0); // x_top_left
        regions(region_index, 1) = proposed_region(1)(1); // y_top_left
        regions(region_index, 2) = proposed_region(1)(2); // x_bottom_right
        regions(region_index, 3) = proposed_region(1)(3); // y_bottom_right

        // Do it with Eigen

        for(Index channel_index = 0; channel_index < image_channels_number; channel_index++)
        {
            for(Index rows_index = 0; rows_index < region_rows; rows_index++)
            {
                for(Index columns_index = 0; columns_index < region_columns; columns_index++)
                {
                    outputs(region_index, image_pixel_index) = proposed_region(0)(image_pixel_index);

                    image_pixel_index++;
                }
            }
        }
    }
*/
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
