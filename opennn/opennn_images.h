//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   O P E N N N   I M A G E S   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#ifndef OPENNN_IMAGES_H
#define OPENNN_IMAGES_H

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

// Eigen includes

#include "config.h"

using namespace std;
using namespace Eigen;

namespace opennn
{

    Tensor<Tensor<type, 1>, 1> read_bmp_image_data(const string& filename);

    Tensor<type, 1> resize_image(Tensor<type, 1>&);

    Tensor<type, 1> get_ground_truth_values(Tensor<unsigned char, 1>&, Index&, Index&, Index&, Index&);

    Tensor<type, 1> get_bounding_box(const Tensor<unsigned char, 1>&, const Index&, const Index&,
                                     const Index&, const Index&);

    void sort_channel(Tensor<unsigned char,1>&, Tensor<unsigned char,1>&, const int&);

    Tensor<unsigned char, 1> remove_padding(Tensor<unsigned char, 1>&, const int&,const int&, const int&);

    /*
    Index channels_number = 0;
    Index image_width = 0;
    Index image_height = 0;
    Index padding = 0;
    */

    /*
    struct BoundingBox
    {
        /// Default constructor.

        explicit BoundingBox() {}

        explicit BoundingBox(const Index& new_channels_number, const Index& new_width, const Index& new_height)
        {
            channels_number = new_channels_number;
            width = new_width;
            height = new_height;

            data.resize(channels_number*width*height);
        }

        explicit BoundingBox(const Index& new_channels_number, const Tensor<Index, 1>& new_center,
                             const Index& new_width, const Index& new_height)
        {
           channels_number = new_channels_number;

           x_center = new_center(0);
           y_center = new_center(1);
           width = new_width;
           height = new_height;

           data.resize(channels_number*width*height);
        }

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
        }


        /// Destructor.

        virtual ~BoundingBox() {}


        Index get_bounding_box_size(const BoundingBox& bounding_box) const
        {
            return bounding_box.data.size();
        }


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
        }

        void print() const
        {
            cout << "Showing the values from the bounding box of size " << width << " x " << height << " x " << channels_number << ": " << endl;
            cout << data << endl;
            cout << "Total size of the bounding box data: " << data.size() << endl;
        }


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
    */
}

#endif // OPENNN_IMAGES_H
