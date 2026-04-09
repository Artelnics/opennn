//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   I M A G E   D A T A S E T   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "image_dataset.h"
#include "image_utilities.h"
#include "tensor_utilities.h"
#include "string_utilities.h"
#include "random_utilities.h"

namespace opennn
{

ImageDataset::ImageDataset(const Index new_samples_number,
                           const Shape& new_input_shape,
                           const Shape& new_target_shape)
{
    if (new_input_shape.rank != 3)
        throw runtime_error("Input shape is not 3");

    if (new_target_shape.rank != 1)
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

void ImageDataset::set_data_random()
{
    const Index height = input_shape[0];
    const Index width = input_shape[1];
    const Index channels = input_shape[2];

    const Index targets_number = target_shape[0];
    const Index inputs_number = height * width * channels;
    const Index samples_number = data.rows();

    data.setZero();

    const Index images_per_category = samples_number / targets_number;
    Index remainder = samples_number % targets_number;

    VectorI images_number(targets_number);
    for(Index i = 0; i < targets_number; i++)
    {
        images_number[i] = images_per_category + (remainder > 0 ? 1 : 0);
        if (remainder > 0) remainder--;
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

void ImageDataset::to_XML(XMLPrinter& printer) const
{
    printer.OpenElement("ImageDataset");

    printer.OpenElement("DataSource");

    write_xml_properties(printer, {
        {"FileType", "bmp"},
        {"Path", data_path.string()},
        {"HasSamplesId", to_string(has_sample_ids)},
        {"Channels", to_string(input_shape[2])},
        {"Width", to_string(input_shape[1])},
        {"Height", to_string(input_shape[0])},
        {"Padding", to_string(padding)},
        {"RandomReflectionAxisX", to_string(augmentation.reflection_axis_x)},
        {"RandomReflectionAxisY", to_string(augmentation.reflection_axis_y)},
        {"RandomRotationMinimum", to_string(augmentation.rotation_minimum)},
        {"RandomRotationMaximum", to_string(augmentation.rotation_maximum)},
        {"RandomHorizontalTranslationMinimum", to_string(augmentation.horizontal_translation_minimum)},
        {"RandomHorizontalTranslationMaximum", to_string(augmentation.horizontal_translation_maximum)},
        {"RandomVerticalTranslationMinimum", to_string(augmentation.vertical_translation_minimum)},
        {"RandomVerticalTranslationMaximum", to_string(augmentation.vertical_translation_maximum)},
        {"Codification", get_codification_string()}
    });

    printer.CloseElement();

    variables_to_XML(printer);

    samples_to_XML(printer);

    add_xml_element(printer, "Display", to_string(display));

    printer.CloseElement();
}

void ImageDataset::augment_inputs(type* input_data, Index batch_size) const
{
    if(!augmentation.enabled) return;

    const Index height = input_shape[0];
    const Index width = input_shape[1];
    const Index channels = input_shape[2];

    TensorMap4 inputs(input_data,
                      batch_size,
                      height,
                      width,
                      channels);

    for(Index i = 0; i < batch_size; i++)
    {
        Tensor3 image = inputs.chip(i, 0);

        if(augmentation.reflection_axis_x)
            reflect_image_horizontal(image);

        if(augmentation.reflection_axis_y)
            reflect_image_vertical(image);

        if(augmentation.rotation_minimum != 0 && augmentation.rotation_maximum != 0)
            rotate_image(image, image, random_uniform(augmentation.rotation_minimum, augmentation.rotation_maximum));

        if(augmentation.horizontal_translation_minimum != 0 && augmentation.horizontal_translation_maximum != 0)
            translate_image_x(image, image, Index(random_uniform(augmentation.horizontal_translation_minimum, augmentation.horizontal_translation_maximum)));

        if(augmentation.vertical_translation_minimum != 0 && augmentation.vertical_translation_maximum != 0)
            translate_image_y(image, image, Index(random_uniform(augmentation.vertical_translation_minimum, augmentation.vertical_translation_maximum)));
    }
}

void ImageDataset::from_XML(const XMLDocument& data_set_document)
{
    const XMLElement* image_dataset_element = get_xml_root(data_set_document, "ImageDataset");

    // Data Source

    const XMLElement* data_source_element = require_xml_element(image_dataset_element, "DataSource");

    set_data_path(read_xml_string(data_source_element, "Path"));
    set_has_ids(read_xml_bool(data_source_element, "HasSamplesId"));

    set_shape("Input", { read_xml_index(data_source_element, "Height"),
                         read_xml_index(data_source_element, "Width"),
                         read_xml_index(data_source_element, "Channels") });

    set_image_padding(read_xml_index(data_source_element, "Padding"));

    set_codification(read_xml_string(data_source_element, "Codification"));

    augmentation.reflection_axis_x = read_xml_index(data_source_element, "RandomReflectionAxisX");
    augmentation.reflection_axis_y = read_xml_index(data_source_element, "RandomReflectionAxisY");
    augmentation.rotation_minimum = read_xml_type(data_source_element, "RandomRotationMinimum");
    augmentation.rotation_maximum = read_xml_type(data_source_element, "RandomRotationMaximum");
    augmentation.horizontal_translation_minimum = read_xml_type(data_source_element, "RandomHorizontalTranslationMinimum");
    augmentation.horizontal_translation_maximum = read_xml_type(data_source_element, "RandomHorizontalTranslationMaximum");
    augmentation.vertical_translation_minimum = read_xml_type(data_source_element, "RandomVerticalTranslationMinimum");
    augmentation.vertical_translation_maximum = read_xml_type(data_source_element, "RandomVerticalTranslationMaximum");

    variables_from_XML(require_xml_element(image_dataset_element, "Variables"));
    samples_from_XML(require_xml_element(image_dataset_element, "Samples"));
}

vector<Descriptives> ImageDataset::scale_features(const string&)
{
    data.leftCols(get_features_number("Input")) /= type(255);

    return {};
}

void ImageDataset::unscale_features(const string&)
{
    data.leftCols(get_features_number("Input")) *= type(255);
}

void ImageDataset::read_bmp(const Shape& new_input_shape)
{
    const chrono::high_resolution_clock::time_point start_time = chrono::high_resolution_clock::now();
    
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

    const Index pixels_number = height * width * channels;

    const Index targets_number = (folders_number == 2) 
        ? folders_number - 1 
        : folders_number;

    set(samples_number, { height, width, channels }, { targets_number });

    vector<string> categories(folders_number);
    for(Index i = 0; i < folders_number; i++)
        categories[i] = directory_path[i].filename().string();

    if(targets_number == 1)
    {
        variables[pixels_number].name = categories[0] + "_" + categories[1];
        variables[pixels_number].role = "Target";
        variables[pixels_number].type = VariableType::Binary;
        variables[pixels_number].set_categories(categories);
        variables[pixels_number].scaler = "None";
    }
    else
    {
        variables.resize(pixels_number + 1);

        variables[pixels_number].name = "Class";
        variables[pixels_number].role = "Target";
        variables[pixels_number].type = VariableType::Categorical;
        variables[pixels_number].set_categories(categories);
        variables[pixels_number].scaler = "None";
    }

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

        copy(image.data(), image.data() + pixels_number, &data(i, 0));

        for(Index k = 0; k < folders_number; k++)
        {
            if (i >= images_number(k) && i < images_number(k + 1))
            {
                if (targets_number == 1)
                    data(i, pixels_number) = k;
                else
                    data(i, k + pixels_number) = 1;
                break;
            }
        }

        #pragma omp atomic
        progress_counter++;

        if (omp_get_thread_num() == 0)
            display_progress_bar(progress_counter, samples_number);
    }

    if (display)
    {
        const chrono::high_resolution_clock::time_point end_time = chrono::high_resolution_clock::now();
        const chrono::milliseconds duration = chrono::duration_cast<chrono::milliseconds>(end_time - start_time);

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
// Licensed under the GNU Lesser General Public License v2.1 or later.
