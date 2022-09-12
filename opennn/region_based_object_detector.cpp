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


///// Destructor.

RegionBasedObjectDetector::~RegionBasedObjectDetector()
{
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


Tensor<type, 1> RegionBasedObjectDetector::get_unique_bounding_box(const Tensor<unsigned char, 1>& image,
                                                             const Index& x_top_left, const Index& y_top_left,
                                                             const Index& x_bottom_right, const Index& y_bottom_right) const
{
    const Index channels_number = 1;// = data_set_pointer->get_channels_number();
    const Index height = 28;//data_set_pointer->get_image_height();
    const Index width = 28;// = data_set_pointer->get_image_width();

    const Index bounding_box_width = abs(x_top_left - x_bottom_right);
    const Index bounding_box_height = abs(y_top_left - y_bottom_right);

    Tensor<type, 1> data;
    data.resize(channels_number * bounding_box_width * bounding_box_height);

    const Index pixel_loop_start = channels_number * (width * (y_bottom_right - 1) + (x_top_left - 1));
    const Index pixel_loop_end = channels_number * (width * (y_top_left - 2) + (x_bottom_right - 1));

    Index data_index = 0;

    for(Index i = pixel_loop_start; i <= pixel_loop_end - 1; i++)
    {
        const int height_number = (int)(i/height);

        const Index left_margin = (height_number * width + (x_top_left - 1)) * channels_number;
        const Index right_margin = (height_number * width + (x_bottom_right - 1)) * channels_number;

        if(i >= left_margin && i < right_margin)
        {
            data(data_index) = static_cast<type>(image[i]);
            data_index++;
        }
    }

    return data;
}

// Object detector deploy pipeline

// 1. Propose random regions

BoundingBox RegionBasedObjectDetector::propose_random_region(const Tensor<unsigned char, 1>& image) const
{
    const Index channels_number = 1; // = data_set_pointer->get_channels_number();
    const Index image_height = 28; //data_set_pointer->get_image_height();
    const Index image_width = 28; // = data_set_pointer->get_image_width();

    const Index half_height =  image_height / 2;
    const Index half_width = image_width / 2;

    /* initialize random seed: */

    srand(time(NULL));

    Index x_center = rand() % image_width;
    Index y_center = rand() % image_height;

    Index x_top_left;
    Index y_bottom_right;

    if(x_center == 0){x_top_left = 0;} else{x_top_left = rand() % x_center;}
    if(x_center == 0){y_bottom_right = 1;} else{y_bottom_right = rand() % y_center;}

    Index y_top_left = rand() % image_height + y_center;
    Index x_bottom_right = rand() % image_width + x_center;

    BoundingBox random_region(channels_number, x_top_left, y_top_left, x_bottom_right, y_bottom_right);

    random_region.data = get_unique_bounding_box(image, x_top_left, y_top_left, x_bottom_right, y_bottom_right);

    return random_region;
}

// 2. Propose random regions

Tensor<BoundingBox, 1> RegionBasedObjectDetector::propose_regions(const Tensor<unsigned char, 1>& image) const
{
    const Index regions_number = 50; // We select random number of regions
    Tensor<BoundingBox, 1> proposed_regions(regions_number);

    for(Index i = 0; i < regions_number; i++)
    {
        proposed_regions(i) = propose_random_region(image);
    }

    return proposed_regions;
}


Tensor<BoundingBox, 1> RegionBasedObjectDetector::regress_regions(Tensor<BoundingBox, 1>& proposed_regions) const
{
    Tensor<BoundingBox, 1> regressed_regions(regions_number);

    for(Index i = 0; i < regions_number; i++)
    {
        regressed_regions(i) = proposed_regions(i).regression();
    }

    return regressed_regions;
}

// 3. Warp proposed regions

Tensor<BoundingBox, 1> RegionBasedObjectDetector::warp_regions(const Tensor<BoundingBox, 1>& proposed_regions) const
{
    Tensor<BoundingBox, 1> warped_regions(regions_number);

    // #pragma

    const Index channels_number = proposed_regions(0).channels_number;

    for(Index i = 0; i < regions_number; i++)
    {
        warped_regions(i) = proposed_regions(i).resize(channels_number, 224, 224); // Default size for the input images of the cnn (224,224)
    }

    return warped_regions;
}


Tensor<Index, 2> RegionBasedObjectDetector::calculate_region_outputs(const Tensor<BoundingBox, 1>& warped_regions) const
{
    Tensor<Index, 2> region_outputs;

//    for(Index i = 0; i < regions_number; i++)
//    {
//        Tensor<type, 2> inputs;
//        Tensor<type, 2> outputs = neural_network_pointer->calculate_outputs(inputs);
//    }

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
    else
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
