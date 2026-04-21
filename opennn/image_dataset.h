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

struct AugmentationSettings
{
    bool enabled = false;
    bool reflection_axis_x = false;
    bool reflection_axis_y = false;
    type rotation_minimum = type(0);
    type rotation_maximum = type(0);
    type horizontal_translation_minimum = type(0);
    type horizontal_translation_maximum = type(0);
    type vertical_translation_minimum = type(0);
    type vertical_translation_maximum = type(0);
};

class ImageDataset : public Dataset
{

public:

    ImageDataset(const Index = 0, const Shape& = {0, 0, 0}, const Shape& = {0});

    ImageDataset(const filesystem::path&);

    Index get_channels_number() const;

    const AugmentationSettings& get_augmentation() const { return augmentation; }
    void set_augmentation(const AugmentationSettings& a) { augmentation = a; }

    void set_data_random() override;

    void set_image_padding(const int& p) { padding = p; }

    vector<Descriptives> scale_features(const string&) override;
    void unscale_features(const string&);

    void read_bmp(const Shape& new_input_shape = { 0, 0, 0 });

    void from_XML(const XmlDocument&) override;
    void to_XML(XmlPrinter&) const override;

    void augment_inputs(type*, Index) const override;

private:

    Index padding = 0;

    AugmentationSettings augmentation;

    // Object detection

    vector<string> labels_tokens;

    Index width_no_padding = 0;

    Index regions_number = 1000;
    Index region_rows = 6;
    Index region_variables = 6;

};

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
