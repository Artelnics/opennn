//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   I M A G E   D A T A S E T   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#pragma once

#include "dataset.h"

namespace opennn
{

class ImageDataset : public Dataset
{

public:

    ImageDataset(const Index = 0, const Shape& = {0, 0, 0}, const Shape& = {0});

    ImageDataset(const filesystem::path&);

    Index get_channels_number() const;
    Index get_image_width() const;
    Index get_image_height() const;
    Index get_image_padding() const;
    Index get_image_size() const;

    bool get_random_reflection_axis_x() const;
    bool get_random_reflection_axis_y() const;
    type get_random_rotation_minimum() const;
    type get_random_rotation_maximum() const;
    type get_random_horizontal_translation_minimum() const;
    type get_random_horizontal_translation_maximum() const;
    type get_random_vertical_translation_minimum() const;
    type get_random_vertical_translation_maximum() const;

    void set_data_random() override;

    void set_channels_number(const int&);
    void set_image_width(const int&);
    void set_image_height(const int&);
    void set_image_padding(const int&);

    void set_augmentation(bool);
    void set_random_reflection_axis_x(bool);
    void set_random_reflection_axis_y(bool);
    void set_random_rotation_minimum(const type);
    void set_random_rotation_maximum(const type);
    void set_random_horizontal_translation_minimum(const type);
    void set_random_horizontal_translation_maximum(const type);
    void set_random_vertical_translation_minimum(const type);
    void set_random_vertical_translation_maximum(const type);

    vector<Descriptives> scale_features(const string&) override;
    void unscale_features(const string&);

    void read_bmp(const Shape& new_input_shape = { 0, 0, 0 });

    void from_XML(const XMLDocument&) override;
    void to_XML(XMLPrinter&) const override;

    void print() const override;

    void fill_inputs(const vector<Index>&,
                     const vector<Index>&,
                     type*,
                     bool = true) const override;

    void perform_augmentation(type*) const;

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
    Index region_variables = 6; // Final region height to warp

};

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
