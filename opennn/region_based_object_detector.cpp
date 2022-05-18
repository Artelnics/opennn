//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   R E G I O N   B A S E D   O B J E C T   D E C T E C T O R   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "region_based_object_detector.h"

namespace opennn
{

/// Default constructor.

RegionBasedObjectDetector::RegionBasedObjectDetector()
{
}


/// Training strategy constructor.
/// @param new_training_strategy_pointer Pointer to a training strategy object.

RegionBasedObjectDetector::RegionBasedObjectDetector(NeuralNetwork* new_neural_network_pointer)
{
    neural_network_pointer = new_neural_network_pointer;
}


Tensor<BoundingBox, 1> RegionBasedObjectDetector::detect_objects(const Tensor<Index, 1>& image) const
{
    Tensor<BoundingBox, 1> proposed_regions = propose_regions(image);
    Tensor<BoundingBox, 1> warped_regions = warp_regions(proposed_regions);
    Tensor<Index, 2> regions_predicted = calculate_region_outputs(warped_regions);
    Tensor<BoundingBox, 1> image_objects = select_strongest(regions_predicted);

    return image_objects;
}

void RegionBasedObjectDetector::segment_image()
{
    Index cont = 0;
    while(true)
    {
        Index stop = 0;

        if(stop == 1) break;
        cont ++;
    }

    regions_number = cont;
}

Tensor<BoundingBox, 1> RegionBasedObjectDetector::propose_regions(const Tensor<unsigned char, 1>& image) const
{
    Tensor<BoundingBox, 1> proposed_regions(regions_number);

    DataSet data_set;

    const Index height = data_set.get_image_height();
    const Index width = data_set.get_image_width();
    const Index channels_number = data_set.get_channels_number();
    const Index size = data_set.get_image_size();

    /**/

    int half_width = image_width/2;
    int half_height = image_height/2;

    int bounding_box_x;// = x;
    int bounding_box_y;// = image_width * y + (x - image_width);
    int bounding_box_width;// =
    int bounding_box_height;// =

    for(Index i = 0; i < data.dimension(0); i++)
    {
        for(Index j = 0; j < data.dimension(1); j++)
        {

        }
    }

    return proposed_regions;
}


Tensor<BoundingBox, 1> RegionBasedObjectDetector::warp_regions(const Tensor<BoundingBox, 1>& proposed_regions) const
{
    Tensor<BoundingBox, 1> warped_regions(regions_number);

    DataSet data_set;

    const Index channels_number = data_set.get_channels_number();
    Index newWidth;
    Index newHeight;

    // #pragma

    for(Index i = 0; i < regions_number; i++)
    {
        const Index height = proposed_regions(i).height;
        const Index width = proposed_regions(i).width;

        //  if(region_data == NULL) return false;

        // Get a new buffer to interpolate into

        warped_regions(i).data(newWidth * newHeight * channels_number);

        const type scaleWidth =  (type)newWidth / (type)width;
        const type scaleHeight = (type)newHeight / (type)height;

        for(Index h = 0; h < newHeight; h++)
        {
            for(Index w = 0; w < newWidth; w++)
            {
                const int pixel = (h * (newWidth *channels_number)) + (w*channels_number);
                const int nearestMatch =  (((int)(h / scaleHeight) * (width *channels_number)) + ((int)(w / scaleWidth) * channels_number));

                if(channels_number == 3)
                {
                    warped_regions(i).data[pixel] =  proposed_regions(i).data[nearestMatch];
                    warped_regions(i).data[pixel + 1] =  proposed_regions(i).data[nearestMatch + 1];
                    warped_regions(i).data[pixel + 2] =  proposed_regions(i).data[nearestMatch + 2];
                }
                else
                {
                    warped_regions(i).data[pixel] =  proposed_regions(i).data[nearestMatch];
                }
            }
        }

        warped_regions(i).width = newWidth;
        warped_regions(i).height = newHeight;
    }

    return warped_regions;
}


Tensor<Index, 2> RegionBasedObjectDetector::calculate_region_outputs(const Tensor<BoundingBox, 1>& warped_regions) const
{
    Tensor<Index, 2> region_outputs;

    for(Index i = 0; i < regions_number; i++)
    {
        Tensor<type, 2> inputs;
        Tensor<type, 2> outputs = neural_network_pointer->calculate_outputs(inputs);
    }

    return region_outputs;
}


Tensor<BoundingBox, 1> RegionBasedObjectDetector::select_strongest(Tensor<Index, 2>& region_outputs) const
{
    Tensor<BoundingBox, 1> final_objects;

    return final_objects;
}


type RegionBasedObjectDetector::calculate_intersection_over_union(const BoundingBox& data_set_bouding_box, const BoundingBox& neural_network_bounding_box)
{
    return 0;
}


void RegionBasedObjectDetector::generate_data_set(const GroundTruth& ground_truth)
{
    DataSet data_set;
}


void RegionBasedObjectDetector::perform_training()
{
    DataSet data_set;

    NeuralNetwork neural_network;

    TrainingStrategy training_strategy(&neural_network, &data_set);

    training_strategy.perform_training();
}

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2021 Artificial Intelligence Techniques, SL.
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
