//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   I M A G E   D A T A S E T   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "image_dataset.h"
#include "images.h"
#include "tensors.h"
#include "strings_utilities.h"
#include "random_utilities.h"

namespace opennn
{

ImageDataset::ImageDataset(const Index new_samples_number,
                           const Shape& new_input_shape,
                           const Shape& new_target_shape)
{
    if (new_input_shape.size() != 3)
        throw runtime_error("Input shape is not 3");

    if (new_target_shape.size() != 1)
        throw runtime_error("Target shape is not 1");

    set(new_samples_number, new_input_shape, new_target_shape);
}


ImageDataset::ImageDataset(const filesystem::path& new_data_path) : Dataset()
{
    data_path = new_data_path;

    read_bmp();
}


Index ImageDataset::get_channels_number() const
{
    return input_shape[2];
}


Index ImageDataset::get_image_width() const
{
    return input_shape[1];
}


Index ImageDataset::get_image_height() const
{
    return input_shape[0];
}


Index ImageDataset::get_image_size() const
{
    return input_shape[0] * input_shape[1] * input_shape[2];
}


Index ImageDataset::get_image_padding() const
{
    return padding;
}


bool ImageDataset::get_augmentation() const
{
    return augmentation;
}


bool ImageDataset::get_random_reflection_axis_x() const
{
    return random_reflection_axis_x;
}


bool ImageDataset::get_random_reflection_axis_y() const
{
    return random_reflection_axis_y;
}


type ImageDataset::get_random_rotation_minimum() const
{
    return random_rotation_minimum;
}


type ImageDataset::get_random_rotation_maximum() const
{
    return random_rotation_maximum;
}


type ImageDataset::get_random_horizontal_translation_minimum() const
{
    return random_horizontal_translation_minimum;
}


type ImageDataset::get_random_horizontal_translation_maximum() const
{
    return random_horizontal_translation_maximum;
}


type ImageDataset::get_random_vertical_translation_maximum() const
{
    return random_vertical_translation_maximum;
}


type ImageDataset::get_random_vertical_translation_minimum() const
{
    return random_vertical_translation_minimum;
}


void ImageDataset::set_data_random()
{
    const Index height = input_shape[0];
    const Index width = input_shape[1];
    const Index channels = input_shape[2];

    const Index targets_number = target_shape[0];
    const Index inputs_number = height * width * channels;
    const Index samples_number = data.rows();

    data.setZero();

    if (targets_number == 1)
    {
        const Index half_samples = samples_number / 2;

        for(Index i = 0; i < samples_number; i++)
        {
            for(Index j = 0; j < inputs_number; j++)
                data(i, j) = random_integer(0, 255);

            data(i, inputs_number) = (i < half_samples) ? 0 : 1;
        }
    }
    else
    {
        VectorI images_number(targets_number);
        images_number.setZero();

        const Index images_per_category = samples_number / targets_number;
        Index remainder = samples_number % targets_number;

        for(Index i = 0; i < targets_number; i++)
        {
            images_number[i] = images_per_category + (remainder > 0 ? 1 : 0);

            if (remainder > 0)
                remainder--;
        }

        Index current_sample = 0;

        for(Index k = 0; k < targets_number; k++)
        {
            for(Index i = 0; i < images_number[k]; i++)
            {
                for(Index j = 0; j < inputs_number; j++)
                    data(current_sample, j) = random_integer(0, 255);

                data(current_sample, k + inputs_number) = 1;

                current_sample++;
            }
        }
    }
}


void ImageDataset::set_channels_number(const int& new_channels)
{
    input_shape[2] = new_channels;
}


void ImageDataset::set_image_width(const int& new_width)
{
    input_shape[1] = new_width;
}


void ImageDataset::set_image_height(const int& new_height)
{
    input_shape[0] = new_height;
}


void ImageDataset::set_image_padding(const int& new_padding)
{
    padding = new_padding;
}


void ImageDataset::set_augmentation(bool new_augmentation)
{
    augmentation = new_augmentation;
}


void ImageDataset::set_random_reflection_axis_x(bool new_random_reflection_axis_x)
{
    random_reflection_axis_x = new_random_reflection_axis_x;
}


void ImageDataset::set_random_reflection_axis_y(bool new_random_reflection_axis_y)
{
    random_reflection_axis_y = new_random_reflection_axis_y;
}


void ImageDataset::set_random_rotation_minimum(const type new_random_rotation_minimum)
{
    random_rotation_minimum = new_random_rotation_minimum;
}


void ImageDataset::set_random_rotation_maximum(const type new_random_rotation_maximum)
{
    random_rotation_maximum = new_random_rotation_maximum;
}


void ImageDataset::set_random_horizontal_translation_maximum(const type new_random_horizontal_translation_maximum)
{
    random_horizontal_translation_maximum = new_random_horizontal_translation_maximum;
}


void ImageDataset::set_random_horizontal_translation_minimum(const type new_random_horizontal_translation_minimum)
{
    random_horizontal_translation_minimum = new_random_horizontal_translation_minimum;
}


void ImageDataset::set_random_vertical_translation_minimum(const type new_random_vertical_translation_minimum)
{
    random_vertical_translation_minimum = new_random_vertical_translation_minimum;
}


void ImageDataset::set_random_vertical_translation_maximum(const type new_random_vertical_translation_maximum)
{
    random_vertical_translation_maximum = new_random_vertical_translation_maximum;
}


void ImageDataset::to_XML(XMLPrinter& printer) const
{
    printer.OpenElement("ImageDataset");

    printer.OpenElement("DataSource");

    add_xml_element(printer, "FileType", "bmp");
    add_xml_element(printer, "Path", data_path.string());
    add_xml_element(printer, "HasSamplesId", to_string(has_sample_ids));
    add_xml_element(printer, "Channels", to_string(get_channels_number()));
    add_xml_element(printer, "Width", to_string(get_image_width()));
    add_xml_element(printer, "Height", to_string(get_image_height()));
    add_xml_element(printer, "Padding", to_string(get_image_padding()));
    add_xml_element(printer, "RandomReflectionAxisX", to_string(get_random_reflection_axis_x()));
    add_xml_element(printer, "RandomReflectionAxisY", to_string(get_random_reflection_axis_y()));
    add_xml_element(printer, "RandomRotationMinimum", to_string(get_random_rotation_minimum()));
    add_xml_element(printer, "RandomRotationMaximum", to_string(get_random_rotation_maximum()));
    add_xml_element(printer, "RandomHorizontalTranslationMinimum", to_string(get_random_horizontal_translation_minimum()));
    add_xml_element(printer, "RandomHorizontalTranslationMaximum", to_string(get_random_horizontal_translation_maximum()));
    add_xml_element(printer, "RandomVerticalTranslationMinimum", to_string(get_random_vertical_translation_minimum()));
    add_xml_element(printer, "RandomVerticalTranslationMaximum", to_string(get_random_vertical_translation_maximum()));
    add_xml_element(printer, "Codification", get_codification_string());

    printer.CloseElement();

    variables_to_XML(printer);

    samples_to_XML(printer);

    add_xml_element(printer, "Display", to_string(display));

    printer.CloseElement();
}


void ImageDataset::perform_augmentation(type* input_data) const
{
    throw runtime_error("Image Augmentation is not yet implemented. Please check back in a future version.");

    const Shape input_shape = get_shape("Input");

    const Index samples_number = input_shape[0];
    const Index input_height = input_shape[0];
    const Index input_width = input_shape[1];
    const Index channels = input_shape[2];

    TensorMap4 inputs(input_data,
                      samples_number,
                      input_height,
                      input_width,
                      channels);

    for(Index batch_index = 0; batch_index < samples_number; batch_index++)
    {
        Tensor3 image = inputs.chip(batch_index, 0);

        if(random_reflection_axis_x)
            reflect_image_x(image);

        if(random_reflection_axis_y)
            reflect_image_y(image);

        if(random_rotation_minimum != 0 && random_rotation_maximum != 0)
            rotate_image(image, image, random_uniform(random_rotation_minimum, random_rotation_maximum));

        if(random_horizontal_translation_minimum != 0 && random_horizontal_translation_maximum != 0)
            translate_image_x(image, image, random_uniform(random_horizontal_translation_minimum, random_horizontal_translation_maximum));

        if(random_vertical_translation_minimum != 0 && random_vertical_translation_maximum != 0)
            translate_image_y(image, image, random_uniform(random_vertical_translation_minimum, random_vertical_translation_maximum));
    }
}


void ImageDataset::fill_input_tensor(const vector<Index>& sample_indices, const vector<Index>& input_indices, type* input_data) const
{
    fill_tensor_data(data, sample_indices, input_indices, input_data);

    if (augmentation)
        perform_augmentation(input_data);

}


void ImageDataset::fill_input_tensor_row_major(const vector<Index>& sample_indices, const vector<Index>& input_indices, type* input_data) const
{
    fill_tensor_data_row_major(data, sample_indices, input_indices, input_data);

    if (augmentation)
        perform_augmentation(input_data);
}


void ImageDataset::from_XML(const XMLDocument& data_set_document)
{
    const XMLElement* image_dataset_element = data_set_document.FirstChildElement("ImageDataset");

    if(!image_dataset_element)
        throw runtime_error("ImageDataset element is nullptr.\n");

    // Data Source

    const XMLElement* data_source_element = image_dataset_element->FirstChildElement("DataSource");

    if(!data_source_element)
        throw runtime_error("Element is nullptr: DataSource");

    set_data_path(read_xml_string(data_source_element, "Path"));
    set_has_ids(read_xml_bool(data_source_element, "HasSamplesId"));

    set_shape("Input", { read_xml_index(data_source_element, "Height"),
                         read_xml_index(data_source_element, "Width"),
                         read_xml_index(data_source_element, "Channels") });

    set_image_padding(read_xml_index(data_source_element, "Padding"));

    set_codification(read_xml_string(data_source_element, "Codification"));

    set_random_reflection_axis_x(read_xml_index(data_source_element, "RandomReflectionAxisX"));
    set_random_reflection_axis_y(read_xml_index(data_source_element, "RandomReflectionAxisY"));
    set_random_rotation_minimum(type(atof(read_xml_string(data_source_element, "RandomRotationMinimum").c_str())));
    set_random_rotation_maximum(type(atof(read_xml_string(data_source_element, "RandomRotationMaximum").c_str())));
    set_random_horizontal_translation_minimum(type(atof(read_xml_string(data_source_element, "RandomHorizontalTranslationMinimum").c_str())));
    set_random_horizontal_translation_maximum(type(atof(read_xml_string(data_source_element, "RandomHorizontalTranslationMaximum").c_str())));
    set_random_vertical_translation_minimum(type(atof(read_xml_string(data_source_element, "RandomVerticalTranslationMinimum").c_str())));
    set_random_vertical_translation_maximum(type(atof(read_xml_string(data_source_element, "RandomVerticalTranslationMaximum").c_str())));

    // Variables

    const XMLElement* variables_element = image_dataset_element->FirstChildElement("Variables");

    variables_from_XML(variables_element);

    // Samples

    const XMLElement* samples_element = image_dataset_element->FirstChildElement("Samples");

    samples_from_XML(samples_element);
}


vector<Descriptives> ImageDataset::scale_features(const string&)
{
    TensorMap4 inputs_data(data.data(),
                           get_samples_number(),
                           input_shape[0],
                           input_shape[1],
                           input_shape[2]);

    inputs_data.device(get_device()) = inputs_data / type(255);

    return vector<Descriptives>();
}


void ImageDataset::unscale_features(const string&)
{
    TensorMap4 inputs_data(data.data(),
                           get_samples_number(),
                           input_shape[0],
                           input_shape[1],
                           input_shape[2]);

    inputs_data.device(get_device()) = inputs_data * type(255);
}


void ImageDataset::read_bmp(const Shape& new_input_shape)
{
    chrono::high_resolution_clock::time_point start_time = chrono::high_resolution_clock::now();
    
    vector<filesystem::path> directory_path;

    for(const filesystem::directory_entry& current_directory : filesystem::directory_iterator(data_path))
        if(current_directory.is_directory())
            directory_path.emplace_back(current_directory.path());

    const Index folders_number = directory_path.size();

    VectorI images_number(folders_number + 1);
    images_number.setZero();
    
    Index samples_number = 0;

    vector<string> image_path;

    for(Index i = 0; i < folders_number; i++)
    {
        for(const filesystem::directory_entry& current_directory : filesystem::directory_iterator(directory_path[i]))
        {
            if(current_directory.is_regular_file() && current_directory.path().extension() == ".bmp")
            {
                image_path.emplace_back(current_directory.path().string());
                samples_number++;
            }
        }

        images_number[i+1] = samples_number;
    }

    if (samples_number == 0)
        throw runtime_error("No images in folder \n");

    const Tensor3 image = load_image(image_path[0]);

    Index height = image.dimension(0);
    Index width = image.dimension(1);
    const Index channels = image.dimension(2);

    if (new_input_shape[2] != channels && new_input_shape[2] != 0)
        throw runtime_error("Different number of channels in new_input_shape \n");

    if (new_input_shape[0] != 0 && new_input_shape[1] != 0)
    {
        height = new_input_shape[0];
        width = new_input_shape[1];
    }

    const Index inputs_number = height * width * channels;
    const Index variables_number = inputs_number + 1;

    const Index pixels_number = height * width * channels;

    const Index targets_number = (folders_number == 2) 
        ? folders_number -1 
        : folders_number;

    set(samples_number, { height, width, channels }, { targets_number });

    vector<string> categories(targets_number);

    for(Index i = 0; i < targets_number; i++)
        categories[i] = directory_path[i].filename().string();

    variables[variables_number-1].set_categories(categories);

    data.setZero();

    Index progress_counter = 0;

    #pragma omp parallel for
    for(Index i = 0; i < samples_number; i++)
    {
        Tensor3 image = load_image(image_path[i]);

        const Index current_height = image.dimension(0);
        const Index current_width = image.dimension(1);
        const Index current_channels = image.dimension(2);

        if (current_channels != channels)
            throw runtime_error("Different number of channels in image: " + image_path[i] + "\n");

        if (current_height != height || current_width != width)
            image = resize_image(image, height, width);

        #pragma omp parallel for
        for(Index j = 0; j < pixels_number; j++)
            data(i, j) = image(j);

        if (targets_number == 1)
        {
            data(i, pixels_number) = (i >= images_number(0) && i < images_number(1)) ? 0 : 1;
        }
        else
        {
            for(Index k = 0; k < targets_number; k++)
            {
                if (i >= images_number(k) && i < images_number(k + 1))
                {
                    data(i, k + pixels_number) = 1;
                    break;
                }
            }
        }

        #pragma omp atomic
        progress_counter++;

        if (omp_get_thread_num() == 0)
            display_progress_bar(progress_counter, samples_number);
    }

    if (display)
    {
        chrono::high_resolution_clock::time_point end_time = chrono::high_resolution_clock::now();
        chrono::milliseconds duration = chrono::duration_cast<chrono::milliseconds>(end_time - start_time);

        const long long total_milliseconds = duration.count();
        const long long minutes = total_milliseconds / 60000;
        const long long seconds = (total_milliseconds % 60000) / 1000;
        const long long milliseconds = total_milliseconds % 1000;

        cout << "\nImage dataset loaded in: "
             << minutes << " minutes, "
             << seconds << " seconds, "
             << milliseconds << " milliseconds." << endl;
    }

    shuffle_rows(data);
}

} // opennn namespace


// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// This library is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 2.1 of the License, or any later version.
// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.
// You should have received a copy of the GNU Lesser General Public
// License along with this library; if not, write to the Free Software
// Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
