//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   I M A G E   D A T A S E T   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#ifndef IMAGEDATASET_H
#define IMAGEDATASET_H

#include "data_set.h"

namespace opennn
{

class ImageDataSet : public DataSet
{

public:

    ImageDataSet(const Index& = 0, const dimensions& = {0, 0, 0}, const dimensions& = {0});

    Index get_channels_number() const;
    Index get_image_width() const;
    Index get_image_height() const;
    Index get_image_padding() const;
    Index get_image_size() const;

    bool get_augmentation() const;
    bool get_random_reflection_axis_x() const;
    bool get_random_reflection_axis_y() const;
    type get_random_rotation_minimum() const;
    type get_random_rotation_maximum() const;
    type get_random_horizontal_translation_minimum() const;
    type get_random_horizontal_translation_maximum() const;
    type get_random_vertical_translation_minimum() const;
    type get_random_vertical_translation_maximum() const;

    void set_image_data_random();

    void set_input_dimensions(const dimensions&);
    void set_channels_number(const int&);
    void set_image_width(const int&);
    void set_image_height(const int&);
    void set_image_padding(const int&);

    void set_augmentation(const bool&);
    void set_random_reflection_axis_x(const bool&);
    void set_random_reflection_axis_y(const bool&);
    void set_random_rotation_minimum(const type&);
    void set_random_rotation_maximum(const type&);
    void set_random_horizontal_translation_minimum(const type&);
    void set_random_horizontal_translation_maximum(const type&);
    void set_random_vertical_translation_minimum(const type&);
    void set_random_vertical_translation_maximum(const type&);

    vector<Descriptives> scale_variables(const VariableUse&) override;

    void read_bmp();

    void from_XML(const XMLDocument&) override;
    void to_XML(XMLPrinter&) const override;

private:

    Index padding = 0;

    bool augmentation = false;
    bool random_reflection_axis_x = false;
    bool random_reflection_axis_y = false;
    type random_rotation_minimum = type(0);
    type random_rotation_maximum = type(0);
    type random_horizontal_translation_minimum = type(0);
    type random_horizontal_translation_maximum = type(0);
    type random_vertical_translation_minimum = type(0);
    type random_vertical_translation_maximum = type(0);

    // Object detection

    vector<string> labels_tokens;

    Index width_no_padding = 0;

    Index regions_number = 1000; // Number of region proposals per image
    Index region_rows = 6; // Final region width to warp
    Index region_raw_variables = 6; // Final region height to warp

};

}

#endif


// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2024 Artificial Intelligence Techniques, SL.
//
// This library is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation either
// version 2.1 of the License, or any later version.
//
// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.

// You should have received a copy of the GNU Lesser General Public
// License along with this library; if not, write to the Free Software
// Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
