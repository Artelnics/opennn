//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   R E G I O N   B A S E D   O B J E C T   D E C T E C T O R   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#ifndef REGIONBASEDOBJECTDETECTOR_H
#define REGIONBASEDOBJECTDETECTOR_H

// System includes

#include <iostream>
#include <fstream>
#include <algorithm>
#include <functional>
#include <limits>
#include <cmath>
#include <ctime>

// OpenNN includes

#include "opennn.h"

using namespace opennn;

namespace opennn
{

struct BoundingBox;
struct GroundTruth;
struct Image;

class RegionBasedObjectDetector
{

public:

    // Constructors

    explicit RegionBasedObjectDetector();

    explicit RegionBasedObjectDetector(NeuralNetwork*);

    virtual ~RegionBasedObjectDetector();


    Tensor<BoundingBox, 1> detect_objects(Tensor<Index, 1>&) const;

    Tensor<type, 1> get_unique_bounding_box(const Tensor<unsigned char, 1>&,
                                                      const Index&, const Index&,
                                                      const Index&, const Index&) const;

    BoundingBox propose_random_region(const Tensor<unsigned char, 1>&) const;

    Tensor<BoundingBox, 1> propose_regions(const Tensor<unsigned char, 1>&) const;

    BoundingBox warp_region(const BoundingBox&, const Index&, const Index&) const;

    Tensor<BoundingBox, 1> regress_regions(Tensor<BoundingBox, 1>&) const;

    Tensor<BoundingBox, 1> warp_regions(const Tensor<BoundingBox, 1>&) const;

    Tensor<Index, 2> calculate_region_outputs(const Tensor<BoundingBox, 1>&) const;

    Tensor<BoundingBox, 1> select_strongest(Tensor<BoundingBox, 1>&) const;

    type calculate_intersection_over_union(const BoundingBox&, const BoundingBox&);

    void generate_data_set(const GroundTruth&);

    void perform_training();

private:

    /// Maximum number of inputs in the neural network.

    NeuralNetwork* neural_network_pointer;

    DataSet* data_set_pointer;

    Index regions_number = 2000;

    type confidence_value = 0.2;

    // Utilities

    void resize_image(const Index&, const Index&);

    void segment_image();

};


struct BoundingBox
{
    /// Default constructor.

    explicit BoundingBox();

    explicit BoundingBox(const Index& new_channels_number, const Index& new_width, const Index& new_height)
    {
        channels_number = new_channels_number;
        width = new_width;
        height = new_height;

        data.resize(channels_number*width*height);
    };


    explicit BoundingBox(const Index& new_channels_number, const Tensor<Index, 1>& new_center, const Index& new_width, const Index& new_height)
    {
        channels_number = new_channels_number;

        x_center = new_center(0);
        y_center = new_center(1);
        width = new_width;
        height = new_height;

        data.resize(channels_number*width*height);
    };


    explicit BoundingBox(const Index& new_channels_number, const Index& new_x_top_left, const Index& new_y_top_left,
                         const Index& new_x_bottom_right, const Index& new_y_bottom_right)
    {
        channels_number = new_channels_number;

        x_top_left = new_x_top_left;
        y_top_left = new_y_top_left;
        x_bottom_right = new_x_bottom_right;
        y_bottom_right = new_y_bottom_right;

        width = abs(new_x_top_left - new_x_bottom_right);
        height = abs(new_y_top_left - new_y_bottom_right);

        data.resize(channels_number*width*height);
    };

    BoundingBox regression()
    {
        /// todo
        BoundingBox regressed_bounging_box;
        return regressed_bounging_box;
    };

    Index size(const BoundingBox& bounding_box) const
    {
        const Index size = bounding_box.data.size();
        return size;
    };

    BoundingBox resize(const Index& new_channels_number, const Index& new_width, const Index& new_height) const
    {
        BoundingBox new_bounding_box(new_channels_number, new_width, new_height);

        const type scaleWidth =  (type)new_width / (type)width;
        const type scaleHeight = (type)new_height / (type)height;

        for(Index i = 0; i < new_height; i++)
        {
            for(Index j = 0; j < new_width; j++)
            {
                const int pixel = (i * (new_width * channels_number)) + (j * channels_number);
                const int nearestMatch =  (((int)(i / scaleHeight) * (width * channels_number)) + ((int)(j / scaleWidth) * channels_number));

                if(channels_number == 3)
                {
                    new_bounding_box.data[pixel] =  data[nearestMatch];
                    new_bounding_box.data[pixel + 1] =  data[nearestMatch + 1];
                    new_bounding_box.data[pixel + 2] =  data[nearestMatch + 2];
                }
                else
                {
                    new_bounding_box.data[pixel] =  data[nearestMatch];
                }
            }
        }

        return new_bounding_box;
    };

    virtual void print() const
    {
        cout << "Showing the values from the bounding box of size " << width << " x " << height << " x " << channels_number << ": " << endl;
        cout << data << endl;
        cout << "Total size of the bounding box data: " << data.size() << endl;
    };

    Tensor<type, 1> data;

    Index x_center;
    Index y_center;
    Index channels_number;
    Index width;
    Index height;

    Index x_top_left;
    Index y_top_left;
    Index x_bottom_right;
    Index y_bottom_right;

    string label; // ????
    Index score; // ????
};


struct Image
{
    explicit Image();

    Tensor<Index, 1> image;
};


struct GroundTruth
{
    explicit GroundTruth();

    virtual void print() const;

    void load(const string& filename);

    Tensor<BoundingBox, 1> objects;
};

}

#endif

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
