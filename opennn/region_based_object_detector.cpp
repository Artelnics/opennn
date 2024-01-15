//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   R E G I O N   B A S E D   O B J E C T   D E C T E C T O R   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com
/*
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


/// Destructor.

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


// Object detector deploy pipeline

// 1. Propose random regions

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



//void RegionBasedObjectDetector::generate_data_set(const GroundTruth& ground_truth)
//{
//    DataSet data_set;
//}


void RegionBasedObjectDetector::perform_training()
{
    DataSet data_set;

    NeuralNetwork neural_network;

    TrainingStrategy training_strategy(&neural_network, &data_set);

    training_strategy.perform_training();
}

}
*/
// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2024 Artificial Intelligence Techniques, SL.
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
