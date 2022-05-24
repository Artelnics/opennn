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


Tensor<BoundingBox, 1> RegionBasedObjectDetector::detect_objects(Tensor<Index, 1>& image) const
{
//    Tensor<BoundingBox, 1> proposed_regions = propose_regions(image);
//    Tensor<BoundingBox, 1> warped_regions = warp_regions(proposed_regions);
//    Tensor<Index, 2> regions_predicted = calculate_region_outputs(warped_regions);
//    Tensor<BoundingBox, 1> image_objects;// = select_strongest(regions_predicted);

//    return image_objects;

//    Tensor<BoundingBox, 1> proposed_regions = propose_regions(image);
    Tensor<BoundingBox, 1> image_objects;
    return image_objects;

}


void RegionBasedObjectDetector::segment_image()
{
//    Index cont = 0;
//    while(true)
//    {
//        Index stop = 0;

//        if(stop == 1) break;
//        cont ++;
//    }

//    regions_number = cont;

}


BoundingBox RegionBasedObjectDetector::get_unique_bounding_box(const Tensor<unsigned char, 1>& image,
                                                                             const Index& x_top_left, const Index& y_top_left,
                                                                             const Index& x_bottom_right, const Index& y_bottom_right) const
{
    DataSet data_set;
    BoundingBox proposed_region;

    proposed_region.x_top_left = x_top_left;
    proposed_region.y_top_left = y_top_left;
    proposed_region.x_bottom_right = x_bottom_right;
    proposed_region.y_bottom_right = y_bottom_right;

    cout << "x top left: " << x_top_left << endl;
    cout << "y top left: " << y_top_left << endl;
    cout << "x bottom right: " << x_bottom_right << endl;
    cout << "y bottom right: " <<  y_bottom_right<< endl;

    const Index width = data_set.get_image_width();
    const Index channels_number = data_set.get_channels_number();

    cout << "width: " <<  width << endl;
    cout << "channels number: " <<  channels_number<< endl;

    Index data_index = 0;

    for(Index i = channels_number * width * (y_bottom_right - 1); i < channels_number * width * (y_top_left - 1) ; i++)
    {
        if((i > (x_top_left + i * width) * channels_number) && (i < (x_bottom_right + i * width) * channels_number))
        {
            proposed_region.data(data_index) = static_cast<type>(image[i]);
            data_index++;
        }
    }

    return proposed_region;
}


Tensor<BoundingBox, 1> RegionBasedObjectDetector::propose_regions(const Tensor<unsigned char, 1>& image) const
{
    const Index temporal_regions_number = 50; // We select random number of regions
    Tensor<BoundingBox, 1> proposed_regions(temporal_regions_number);

    DataSet data_set;

    const Index width = data_set.get_image_width();
    const Index channels_number = data_set.get_channels_number();

    for(Index l = 0; l < temporal_regions_number; l++) // regions_number; l++)
    {
        // Pick the next four values from the neuralLabeler
        Index bounding_box_x_top_left = l;
        Index bounding_box_y_top_left = l;
        Index bounding_box_x_bottom_right = l + 28 ;
        Index bounding_box_y_bottom_right = l + 28;

        proposed_regions(l).x_top_left = bounding_box_x_top_left;
        proposed_regions(l).y_top_left = bounding_box_y_top_left;
        proposed_regions(l).x_bottom_right = bounding_box_x_bottom_right;
        proposed_regions(l).y_bottom_right = bounding_box_y_bottom_right;

        const Index bounding_box_width = bounding_box_x_top_left - bounding_box_x_bottom_right;
        const Index bounding_box_height = bounding_box_y_top_left - bounding_box_y_bottom_right;

        Index data_index = 0;
        for(Index i = channels_number * width * (bounding_box_y_bottom_right - 1); i < channels_number * width * (bounding_box_y_top_left - 1) ; i++)
        {
            if((i > (bounding_box_x_top_left + i * width) * channels_number) && (i < (bounding_box_x_bottom_right + i * width) * channels_number))
            {
                proposed_regions(l).data(data_index) = static_cast<type>(image[i]);
                data_index++;
            }
        }

        cout << "Bounding box size: " << data_index << endl;

        proposed_regions(l).width = bounding_box_width;
        proposed_regions(l).height = bounding_box_height;
        proposed_regions(l).x_center = bounding_box_x_top_left + proposed_regions(l).width / 2;
        proposed_regions(l).y_center = bounding_box_y_bottom_right + proposed_regions(l).height / 2;
    }

    return proposed_regions;
}


BoundingBox RegionBasedObjectDetector::warp_single_region(const BoundingBox& region, const Index& newWidth, const Index& newHeight) const
{
    DataSet data_set;

    BoundingBox warped_region;

    const Index channels_number = data_set.get_channels_number();

    const Index height = region.height;
    const Index width = region.width;

    //  if(region_data == NULL) return false;

    // Get a new buffer to interpolate into

    warped_region.data(newWidth * newHeight * channels_number);

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
                warped_region.data[pixel] =  region.data[nearestMatch];
                warped_region.data[pixel + 1] =  region.data[nearestMatch + 1];
                warped_region.data[pixel + 2] =  region.data[nearestMatch + 2];
            }
            else
            {
                warped_region.data[pixel] =  region.data[nearestMatch];
            }
        }
    }

    warped_region.width = newWidth;
    warped_region.height = newHeight;
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


Tensor<BoundingBox, 1> RegionBasedObjectDetector::select_strongest(Tensor<BoundingBox, 1>& region_outputs) const
{
    Tensor<BoundingBox, 1> final_objects;

    return final_objects;
}


type RegionBasedObjectDetector::calculate_intersection_over_union(const BoundingBox& data_set_bouding_box, const BoundingBox& neural_network_bounding_box)
{
    int intersection_width;
    int intersection_height;

    if((data_set_bouding_box.y_top_left > neural_network_bounding_box.y_top_left) && (data_set_bouding_box.x_bottom_right < neural_network_bounding_box.x_bottom_right))
    {
        intersection_width = abs((int) data_set_bouding_box.x_bottom_right - (int) neural_network_bounding_box.x_top_left);
        intersection_height = abs((int) data_set_bouding_box.y_bottom_right - (int) neural_network_bounding_box.y_top_left);
    }
    else if((data_set_bouding_box.y_top_left > neural_network_bounding_box.y_top_left) && (data_set_bouding_box.x_bottom_right > neural_network_bounding_box.x_bottom_right))
    {
        intersection_width = abs((int) data_set_bouding_box.x_top_left - (int) neural_network_bounding_box.x_bottom_right);
        intersection_height = abs((int) data_set_bouding_box.y_bottom_right - (int) neural_network_bounding_box.y_top_left);
    }
    else if((data_set_bouding_box.y_top_left < neural_network_bounding_box.y_top_left)&& (data_set_bouding_box.x_bottom_right < neural_network_bounding_box.x_bottom_right))
    {
        intersection_width = abs((int) data_set_bouding_box.x_bottom_right - (int) neural_network_bounding_box.x_top_left);
        intersection_height = abs((int) data_set_bouding_box.y_top_left - (int) neural_network_bounding_box.y_bottom_right);
    }
    else// if((data_set_bouding_box.y_top_left < neural_network_bounding_box.y_top_left)&& (data_set_bouding_box.x_bottom_right > neural_network_bounding_box.x_bottom_right))
    {
        intersection_width = abs((int) data_set_bouding_box.x_top_left - (int) neural_network_bounding_box.x_bottom_right);
        intersection_height = abs((int) data_set_bouding_box.y_top_left - (int) neural_network_bounding_box.y_bottom_right);
    }

    const int intersection_area = intersection_height * intersection_width;
    const int union_area = data_set_bouding_box.width * data_set_bouding_box.height + neural_network_bounding_box.width * neural_network_bounding_box.height - intersection_area;

    const type intersection_over_union = static_cast<type>(intersection_area / union_area);

    return intersection_over_union;

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

/// Default constructor.

BoundingBox::BoundingBox()
{
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
