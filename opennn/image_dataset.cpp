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

namespace opennn
{

ImageDataset::ImageDataset(const Index& new_samples_number,
                           const dimensions& new_input_dimensions,
                           const dimensions& new_target_dimensions)
{
    if (new_input_dimensions.size() != 3)
        throw runtime_error("Input dimensions is not 3");

    if (new_target_dimensions.size() != 1)
        throw runtime_error("Target dimensions is not 1");

    set(new_samples_number, new_input_dimensions, new_target_dimensions);
}


ImageDataset::ImageDataset(const filesystem::path& new_data_path) : Dataset()
{
    data_path = new_data_path;

    read_bmp();
}


Index ImageDataset::get_channels_number() const
{
    return input_dimensions[2];
}


Index ImageDataset::get_image_width() const
{
    return input_dimensions[1];
}


Index ImageDataset::get_image_height() const
{
    return input_dimensions[0];
}


Index ImageDataset::get_image_size() const
{
    return input_dimensions[0] * input_dimensions[1] * input_dimensions[2];
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
    const Index height = input_dimensions[0];
    const Index width = input_dimensions[1];
    const Index channels = input_dimensions[2];

    const Index targets_number = target_dimensions[0];
    const Index inputs_number = height * width * channels;
    const Index samples_number = data.dimension(0);

    data.setZero();

    if (targets_number == 1)
    {
        const Index half_samples = samples_number / 2;

        for (Index i = 0; i < samples_number; i++)
        {
            for (Index j = 0; j < inputs_number; j++)
                data(i, j) = rand() % 255;

            data(i, inputs_number) = (i < half_samples) ? 0 : 1;
        }
    }
    else
    {
        Tensor<Index, 1> images_number(targets_number);
        images_number.setZero();

        const Index images_per_category = samples_number / targets_number;
        Index remainder = samples_number % targets_number;

        for (Index i = 0; i < targets_number; i++)
        {
            images_number[i] = images_per_category + (remainder > 0 ? 1 : 0);

            if (remainder > 0)
                remainder--;
        }

        Index current_sample = 0;

        for (Index k = 0; k < targets_number; k++)
        {
            for (Index i = 0; i < images_number[k]; i++)
            {
                for (Index j = 0; j < inputs_number; j++)
                    data(current_sample, j) = rand() % 255;

                data(current_sample, k + inputs_number) = 1;

                current_sample++;
            }
        }
    }
}


void ImageDataset::set_channels_number(const int& new_channels)
{
    input_dimensions[2] = new_channels;
}


void ImageDataset::set_image_width(const int& new_width)
{
    input_dimensions[1] = new_width;
}


void ImageDataset::set_image_height(const int& new_height)
{
    input_dimensions[0] = new_height;
}


void ImageDataset::set_image_padding(const int& new_padding)
{
    padding = new_padding;
}


void ImageDataset::set_augmentation(const bool& new_augmentation)
{
    augmentation = new_augmentation;
}


void ImageDataset::set_random_reflection_axis_x(const bool& new_random_reflection_axis_x)
{
    random_reflection_axis_x = new_random_reflection_axis_x;
}


void ImageDataset::set_random_reflection_axis_y(const bool& new_random_reflection_axis_y)
{
    random_reflection_axis_y = new_random_reflection_axis_y;
}


void ImageDataset::set_random_rotation_minimum(const type& new_random_rotation_minimum)
{
    random_rotation_minimum = new_random_rotation_minimum;
}


void ImageDataset::set_random_rotation_maximum(const type& new_random_rotation_maximum)
{
    random_rotation_maximum = new_random_rotation_maximum;
}


void ImageDataset::set_random_horizontal_translation_maximum(const type& new_random_horizontal_translation_maximum)
{
    random_horizontal_translation_maximum = new_random_horizontal_translation_maximum;
}


void ImageDataset::set_random_horizontal_translation_minimum(const type& new_random_horizontal_translation_minimum)
{
    random_horizontal_translation_minimum = new_random_horizontal_translation_minimum;
}


void ImageDataset::set_random_vertical_translation_minimum(const type& new_random_vertical_translation_minimum)
{
    random_vertical_translation_minimum = new_random_vertical_translation_minimum;
}


void ImageDataset::set_random_vertical_translation_maximum(const type& new_random_vertical_translation_maximum)
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

    printer.OpenElement("RawVariables");

    add_xml_element(printer, "RawVariablesNumber", to_string(get_raw_variables_number()));

    // Raw variables items

    const Index raw_variables_number = get_raw_variables_number();

    for(Index i = 0; i < raw_variables_number; i++)
    {
        printer.OpenElement("RawVariable");
        printer.PushAttribute("Item", to_string(i+1).c_str());
        raw_variables[i].to_XML(printer);
        printer.CloseElement();
    }

    printer.CloseElement();

    if(has_sample_ids)
        add_xml_element(printer, "Ids", vector_to_string(sample_ids));

    printer.OpenElement("Samples");

    add_xml_element(printer, "SamplesNumber", to_string(get_samples_number()));
    add_xml_element(printer, "SampleUses", vector_to_string(get_sample_uses_vector()));

    printer.CloseElement();

    printer.CloseElement();
}


Tensor<type, 2> ImageDataset::perform_augmentation(const Tensor<type, 2>& input_tensor)
{
    throw runtime_error("Image Augmentation is not yet implemented. Please check back in a future version.");
/*
    const dimensions input_dimensions = get_dimensions("Input");

    const Index samples_number = input_dimensions[0];
//    const Index input_height = input_dimensions[0];
//    const Index input_width = input_dimensions[1];
//    const Index channels = input_dimensions[2];

//    TensorMap<Tensor<type, 4>> inputs(input_tensor.data(),
//                                      samples_number,
//                                      input_height,
//                                      input_width,
//                                      channels);

//    const bool random_reflection_axis_x = image_dataset->get_random_reflection_axis_x();
//    const bool random_reflection_axis_y = image_dataset->get_random_reflection_axis_y();
//    const type random_rotation_minimum = image_dataset->get_random_rotation_minimum();
//    const type random_rotation_maximum = image_dataset->get_random_rotation_maximum();
//    const type random_horizontal_translation_minimum = image_dataset->get_random_horizontal_translation_minimum();
//    const type random_horizontal_translation_maximum = image_dataset->get_random_horizontal_translation_maximum();
//    const type random_vertical_translation_minimum = image_dataset->get_random_vertical_translation_minimum();
//    const type random_vertical_translation_maximum = image_dataset->get_random_vertical_translation_maximum();

    for(Index batch_index = 0; batch_index < samples_number; batch_index++)
    {
        Tensor<type, 3> image = inputs.chip(batch_index, 0);

        if(random_reflection_axis_x)
            reflect_image_x(thread_pool_device.get(),
                            image);

        if(random_reflection_axis_y)
            reflect_image_y(thread_pool_device.get(),
                            image);

        if(random_rotation_minimum != 0 && random_rotation_maximum != 0)
            rotate_image(thread_pool_device.get(),
                         image,
                         image,
                         get_random_type(random_rotation_minimum, random_rotation_maximum));

        if(random_horizontal_translation_minimum != 0 && random_horizontal_translation_maximum != 0)
            translate_image_x(thread_pool_device.get(),
                              image,
                              image,
                              get_random_type(random_horizontal_translation_minimum, random_horizontal_translation_maximum));

        if(random_vertical_translation_minimum != 0 && random_vertical_translation_maximum != 0)
            translate_image_y(thread_pool_device.get(),
                              image,
                              image,
                              get_random_type(random_vertical_translation_minimum, random_vertical_translation_maximum));
    }
*/
    return Tensor<type, 2>();
}


void ImageDataset::fill_input_tensor(const vector<Index>& sample_indices, const vector<Index>& input_indices, type* input_tensor_data) const
{
    if (get_augmentation())
    {
        // Optional: apply augmentation here
        // Tensor<type, 2> augmented_data = perform_augmentation(data);
        // fill_tensor_data(augmented_data, sample_indices, input_indices, destination_data);
    }
    else
    {
        fill_tensor_data(data, sample_indices, input_indices, input_tensor_data);
    }
}


void ImageDataset::fill_input_tensor_row_major(const vector<Index>& sample_indices, const vector<Index>& input_indices, type* input_tensor_data) const
{
    if (get_augmentation())
    {
        // Optional: apply augmentation here
        // Tensor<type, 2> augmented_data = perform_augmentation(data);
        // fill_tensor_data_row_major(augmented_data, sample_indices, input_indices, destination_data);
    }
    else
    {
        fill_tensor_data_row_major(data, sample_indices, input_indices, input_tensor_data);
    } 
}


void ImageDataset::from_XML(const XMLDocument& data_set_document)
{
    const XMLElement* image_dataset_element = data_set_document.FirstChildElement("ImageDataset");

    if (!image_dataset_element)
        throw runtime_error("ImageDataset element is nullptr.\n");

    // Data Source

    const XMLElement* data_source_element = image_dataset_element->FirstChildElement("DataSource");

    if (!data_source_element)
        throw runtime_error("Element is nullptr: DataSource");

    set_data_path(read_xml_string(data_source_element, "Path"));
    set_has_ids(read_xml_bool(data_source_element, "HasSamplesId"));

    set_dimensions("Input", { read_xml_index(data_source_element, "Height"),
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

    // Raw Variables

    const XMLElement* raw_variables_element = image_dataset_element->FirstChildElement("RawVariables");

    if (!raw_variables_element)
        throw runtime_error("RawVariables element is nullptr.\n");

    const Index raw_variables_number = read_xml_index(raw_variables_element, "RawVariablesNumber");

    set_raw_variables_number(raw_variables_number);

    const XMLElement* start_element = raw_variables_element->FirstChildElement("RawVariablesNumber");

    Index target_count = 0;

    for (auto& raw_variable : raw_variables)
    {
        const XMLElement* raw_variable_element = start_element->NextSiblingElement("RawVariable");
        start_element = raw_variable_element;

        raw_variable.name = read_xml_string(start_element, "Name");
        raw_variable.set_scaler(read_xml_string(start_element, "string"));
        raw_variable.set_use(read_xml_string(start_element, "Use"));
        raw_variable.set_type(read_xml_string(start_element, "Type"));

        if (raw_variable.type == RawVariableType::Categorical || raw_variable.type == RawVariableType::Binary)
        {
            raw_variable.categories = get_tokens(read_xml_string(start_element, "Categories"), ";");
            target_count++;
        }
    }

    const Index targets_number = (target_count == 2) ? 1 : target_count;

    target_dimensions = { targets_number };

    // Samples

    if (has_sample_ids)
        sample_ids = get_tokens(read_xml_string(image_dataset_element, "Ids"), ",");

    const XMLElement* samples_element = image_dataset_element->FirstChildElement("Samples");

    if (!samples_element)
        throw runtime_error("Samples element is nullptr.\n");

    const Index samples_number = read_xml_index(samples_element, "SamplesNumber");

    sample_uses.resize(samples_number);
    data.resize(samples_number, (Index)raw_variables.size());
    set_sample_uses(get_tokens(read_xml_string(samples_element, "SampleUses"), " "));
}


vector<Descriptives> ImageDataset::scale_variables(const string&)
{
    TensorMap<Tensor<type, 4>> inputs_data(data.data(),
                                           get_samples_number(),
                                           input_dimensions[0],
                                           input_dimensions[1],
                                           input_dimensions[2]);

    inputs_data.device(*thread_pool_device) = inputs_data / type(255);

    return vector<Descriptives>();
}


void ImageDataset::unscale_variables(const string&)
{
    TensorMap<Tensor<type, 4>> inputs_data(data.data(),
                                           get_samples_number(),
                                           input_dimensions[0],
                                           input_dimensions[1],
                                           input_dimensions[2]);

    inputs_data.device(*thread_pool_device) = inputs_data * type(255);
}


void ImageDataset::read_bmp(const dimensions& new_input_dimensions)
{
    chrono::high_resolution_clock::time_point start_time = chrono::high_resolution_clock::now();
    
    vector<filesystem::path> directory_path;
    vector<string> image_path;

    const filesystem::path path = data_path;

    for(const filesystem::directory_entry& current_directory : filesystem::directory_iterator(path))
        if(is_directory(current_directory))
            directory_path.emplace_back(current_directory.path().string());

    const Index folders_number = directory_path.size();

    Tensor<Index, 1> images_number(folders_number + 1);
    images_number.setZero();
    
    Index samples_number = 0;

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

    Index height, width, image_channels;

    const Tensor<type, 3> image_data = read_bmp_image(image_path[0]);

    height = image_data.dimension(0);
    width = image_data.dimension(1);
    image_channels = image_data.dimension(2);

    if (new_input_dimensions[2] != image_channels && new_input_dimensions[2] != 0)
        throw runtime_error("Different number of channels in new_input_dimensions \n");

    if (new_input_dimensions[0] != 0 && new_input_dimensions[1] != 0)
    {
        height = new_input_dimensions[0];
        width = new_input_dimensions[1];
    }

    const Index inputs_number = height * width * image_channels;
    const Index raw_variables_number = inputs_number + 1;

    const Index pixels_number = height * width * image_channels;

    const Index targets_number = (folders_number == 2) 
        ? folders_number -1 
        : folders_number;

    set(samples_number, { height, width, image_channels }, { targets_number });

    vector<string> categories(targets_number);

    for(Index i = 0; i < targets_number; i++)
        categories[i] = directory_path[i].filename().string();

    raw_variables[raw_variables_number-1].set_categories(categories);

    data.setZero();

    #pragma omp parallel for
    for (Index i = 0; i < samples_number; i++)
    {
        Tensor<type, 3> image = read_bmp_image(image_path[i]);

        const Index current_height = image.dimension(0);
        const Index current_width = image.dimension(1);
        const Index current_channels = image.dimension(2);

        if (current_channels != image_channels)
            throw runtime_error("Different number of channels in image: " + image_path[i] + "\n");

        if (current_height != height || current_width != width)
            image = resize_image(image, height, width);

        #pragma omp parallel for
        for (Index j = 0; j < pixels_number; j++)
            data(i, j) = image(j);

        if (targets_number == 1)
        {
            data(i, pixels_number) = (i >= images_number(0) && i < images_number(1)) ? 0 : 1;
        }
        else
        {
            for (Index k = 0; k < targets_number; k++)
            {
                if (i >= images_number(k) && i < images_number(k + 1))
                {
                    data(i, k + pixels_number) = 1;
                    break;
                }
            }
        }

        if (display && i % 1000 == 0)
            display_progress_bar(i, samples_number - 1000);
    }

    if (display)
    {
        chrono::high_resolution_clock::time_point end_time = chrono::high_resolution_clock::now();
        chrono::milliseconds duration = chrono::duration_cast<chrono::milliseconds>(end_time - start_time);

        long long total_milliseconds = duration.count();
        long long minutes = total_milliseconds / 60000;
        long long seconds = (total_milliseconds % 60000) / 1000;
        long long milliseconds = total_milliseconds % 1000;

        cout << "\nImage data set loaded in: "
             << minutes << " minutes, "
             << seconds << " seconds, "
             << milliseconds << " milliseconds." << endl;
    }

    // cerr << "Included scale_variables function in the image_dataset::read_bmp" << endl;
    // scale_variables("Input");
}

} // opennn namespace


// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2025 Artificial Intelligence Techniques, SL.
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
