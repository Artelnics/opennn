#include "bounding_box.h"

namespace opennn
{

// BoundingBox constructor

BoundingBox::BoundingBox(const Index& new_channels_number, const Index& new_width, const Index& new_height)
{
    channels_number = new_channels_number;
    width = new_width;
    height = new_height;

    data.resize(channels_number*width*height);
}


// BoundingBox constructor

BoundingBox::BoundingBox(const Index& new_channels_number,
                                  const Tensor<Index, 1>& new_center,
                                  const Index& new_width,
                                  const Index& new_height)
{
   channels_number = new_channels_number;

   x_center = new_center(0);
   y_center = new_center(1);
   width = new_width;
   height = new_height;

   data.resize(channels_number*width*height);
}


// BoundingBox constructor

BoundingBox::BoundingBox(const Index& new_channels_number,
                                  const Index& new_x_top_left,
                                  const Index& new_y_top_left,
                                  const Index& new_x_bottom_right,
                                  const Index& new_y_bottom_right)
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


Index BoundingBox::get_size() const
{
    return data.size();
}


BoundingBox BoundingBox::resize(const Index& new_channels_number, const Index& new_width, const Index& new_height) const
{
    BoundingBox new_bounding_box(new_channels_number, new_width, new_height);

    const type scaleWidth =  (type)new_width / (type)width;
    const type scaleHeight = (type)new_height / (type)height;

    for(Index i = 0; i < new_height; i++)
    {
        for(Index j = 0; j < new_width; j++)
        {
            const int pixel = i * new_width * channels_number + j * channels_number;
            const int nearest_match =  ((int)(i / scaleHeight) * (width * channels_number)) + ((int)(j / scaleWidth) * channels_number);

            if(channels_number == 3)
            {
                new_bounding_box.data[pixel] =  data[nearest_match];
                new_bounding_box.data[pixel + 1] =  data[nearest_match + 1];
                new_bounding_box.data[pixel + 2] =  data[nearest_match + 2];
            }
            else
            {
                new_bounding_box.data[pixel] =  data[nearest_match];
            }
        }
    }

    return new_bounding_box;
}


void BoundingBox::print() const
{
/*
    cout << "Showing the values from the bounding box of size " << width << " x " << height << " x " << channels_number << ": " << endl;

    cout << data << endl;

    cout << "Total size of the bounding box data: " << data.size() << endl;
*/
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
