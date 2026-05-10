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

ImageDataset::ImageDataset(const filesystem::path& new_data_path, bool new_streaming) : MaterializedDataset()
{
    streaming = new_streaming;
    data_path = new_data_path;

    read_bmp();
}

Index ImageDataset::get_samples_number() const
{
    return streaming ? Index(image_paths.size()) : data.rows();
}

Index ImageDataset::get_channels_number() const
{
    return input_shape[2];
}

void ImageDataset::set_data_random()
{
    if (streaming)
        throw runtime_error("set_data_random is not supported in streaming mode.");

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
    for (Index i = 0; i < targets_number; ++i)
    {
        images_number[i] = images_per_category + (remainder > 0 ? 1 : 0);
        if (remainder > 0) remainder--;
    }

    Index current_sample = 0;

    for (Index k = 0; k < targets_number; ++k)
    {
        for (Index i = 0; i < images_number[k]; ++i)
        {
            for (Index j = 0; j < inputs_number; ++j)
                data(current_sample, j) = random_integer(0, 255);

            data(current_sample, k + inputs_number) = 1;

            ++current_sample;
        }
    }
}

void ImageDataset::to_JSON(JsonWriter& printer) const
{
    printer.open_element("ImageDataset");

    printer.open_element("DataSource");

    write_json(printer, {
        {"FileType", "bmp"},
        {"Path", data_path.string()},
        {"Streaming", to_string(streaming)},
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

    printer.close_element();

    variables_to_JSON(printer);

    samples_to_JSON(printer);

    add_json_field(printer, "Display", to_string(display));

    printer.close_element();
}

void ImageDataset::augment_inputs(float* input_data, Index batch_size) const
{
    if (!augmentation.enabled) return;

    const Index height = input_shape[0];
    const Index width = input_shape[1];
    const Index channels = input_shape[2];

    TensorMap4 inputs(input_data,
                      batch_size,
                      height,
                      width,
                      channels);

    for (Index i = 0; i < batch_size; ++i)
    {
        Tensor3 image = inputs.chip(i, 0);

        if (augmentation.reflection_axis_x)
            reflect_image_horizontal(image);

        if (augmentation.reflection_axis_y)
            reflect_image_vertical(image);

        if (augmentation.rotation_minimum != 0 && augmentation.rotation_maximum != 0)
            rotate_image(image, image, random_uniform(augmentation.rotation_minimum, augmentation.rotation_maximum));

        if (augmentation.horizontal_translation_minimum != 0 && augmentation.horizontal_translation_maximum != 0)
            translate_image_x(image, image, Index(random_uniform(augmentation.horizontal_translation_minimum, augmentation.horizontal_translation_maximum)));

        if (augmentation.vertical_translation_minimum != 0 && augmentation.vertical_translation_maximum != 0)
            translate_image_y(image, image, Index(random_uniform(augmentation.vertical_translation_minimum, augmentation.vertical_translation_maximum)));

        inputs.chip(i, 0) = image;
    }
}

void ImageDataset::from_JSON(const JsonDocument& data_set_document)
{
    const Json* image_dataset_element = get_json_root(data_set_document, "ImageDataset");

    const Json* data_source_element = require_json_field(image_dataset_element, "DataSource");

    set_data_path(read_json_string(data_source_element, "Path"));

    streaming = data_source_element->has("Streaming")
        ? read_json_bool(data_source_element, "Streaming")
        : false;

    set_has_ids(read_json_bool(data_source_element, "HasSamplesId"));

    set_shape("Input", { read_json_index(data_source_element, "Height"),
                         read_json_index(data_source_element, "Width"),
                         read_json_index(data_source_element, "Channels") });

    set_image_padding(read_json_index(data_source_element, "Padding"));

    set_codification(read_json_string(data_source_element, "Codification"));

    augmentation.reflection_axis_x = read_json_index(data_source_element, "RandomReflectionAxisX");
    augmentation.reflection_axis_y = read_json_index(data_source_element, "RandomReflectionAxisY");
    augmentation.rotation_minimum = read_json_type(data_source_element, "RandomRotationMinimum");
    augmentation.rotation_maximum = read_json_type(data_source_element, "RandomRotationMaximum");
    augmentation.horizontal_translation_minimum = read_json_type(data_source_element, "RandomHorizontalTranslationMinimum");
    augmentation.horizontal_translation_maximum = read_json_type(data_source_element, "RandomHorizontalTranslationMaximum");
    augmentation.vertical_translation_minimum = read_json_type(data_source_element, "RandomVerticalTranslationMinimum");
    augmentation.vertical_translation_maximum = read_json_type(data_source_element, "RandomVerticalTranslationMaximum");

    if (streaming)
    {
        read_bmp();
    }
    else
    {
        variables_from_JSON(require_json_field(image_dataset_element, "Variables"));
        samples_from_JSON(require_json_field(image_dataset_element, "Samples"));
    }
}

vector<Descriptives> ImageDataset::scale_features(const string&)
{
    if (streaming) return {};

    const Index samples_number = get_samples_number();
    const Index input_features_number = get_features_number("Input");

    #pragma omp parallel for
    for (Index i = 0; i < samples_number; ++i)
        for (Index j = 0; j < input_features_number; ++j)
            data(i, j) /= 255.0f;

    return {};
}

void ImageDataset::unscale_features(const string&)
{
    if (streaming) return;

    data.leftCols(get_features_number("Input")) *= 255.0f;
}

void ImageDataset::read_bmp(const Shape& new_input_shape)
{
    const chrono::high_resolution_clock::time_point start_time = chrono::high_resolution_clock::now();

    vector<filesystem::path> directory_path;

    for (const filesystem::directory_entry& current_directory : filesystem::directory_iterator(data_path))
        if (current_directory.is_directory())
            directory_path.emplace_back(current_directory.path());

    const Index folders_number = directory_path.size();

    Index samples_number = 0;

    vector<filesystem::path> paths;
    vector<Index> labels;

    for (Index i = 0; i < folders_number; ++i)
    {
        for (const filesystem::directory_entry& current_directory : filesystem::directory_iterator(directory_path[i]))
        {
            if (current_directory.is_regular_file() && current_directory.path().extension() == ".bmp")
            {
                paths.emplace_back(current_directory.path());
                labels.push_back(i);
                ++samples_number;
            }
        }
    }

    if (samples_number == 0)
        throw runtime_error("No images in folder \n");

    const Tensor3 first_image = load_image(paths[0]);

    Index height = first_image.dimension(0);
    Index width = first_image.dimension(1);
    const Index channels = first_image.dimension(2);

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

    if (streaming)
    {
        input_shape = { height, width, channels };
        target_shape = { targets_number };

        variables.resize(pixels_number + 1);

        for (Index i = 0; i < pixels_number; ++i)
        {
            variables[i].type = VariableType::Numeric;
            variables[i].name = "variable_" + to_string(i + 1);
            variables[i].role = VariableRole::Input;
        }

        sample_roles.resize(samples_number, SampleRole::Training);
    }
    else
    {
        set(samples_number, { height, width, channels }, { targets_number });
    }

    vector<string> categories(folders_number);
    for (Index i = 0; i < folders_number; ++i)
        categories[i] = directory_path[i].filename().string();

    if (targets_number == 1)
    {
        variables[pixels_number].name = categories[0] + "_" + categories[1];
        variables[pixels_number].role = VariableRole::Target;
        variables[pixels_number].type = VariableType::Binary;
        variables[pixels_number].set_categories(categories);
        variables[pixels_number].scaler = ScalerMethod::None;
    }
    else
    {
        variables.resize(pixels_number + 1);

        variables[pixels_number].name = "Class";
        variables[pixels_number].role = VariableRole::Target;
        variables[pixels_number].type = VariableType::Categorical;
        variables[pixels_number].set_categories(categories);
        variables[pixels_number].scaler = ScalerMethod::None;
    }

    if (streaming)
    {
        image_paths = std::move(paths);
        sample_labels = std::move(labels);
        split_samples_random();
    }
    else
    {
        Index progress_counter = 0;
        string omp_error;

        #pragma omp parallel for
        for (Index i = 0; i < samples_number; ++i)
        {
            try
            {
                load_image(paths[i], &data(i, 0), height, width, channels, false);
            }
            catch (const std::exception& e)
            {
                #pragma omp critical
                { omp_error = e.what(); }
                continue;
            }

            const Index label = labels[i];
            if (targets_number == 1)
                data(i, pixels_number) = label;
            else
                data(i, label + pixels_number) = 1;

            #pragma omp atomic
            ++progress_counter;

            if (omp_get_thread_num() == 0)
                display_progress_bar(progress_counter, samples_number);
        }

        if (!omp_error.empty())
            throw runtime_error(omp_error);
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
             << milliseconds << " milliseconds." << "\n";
    }
}

void ImageDataset::fill_inputs(const vector<Index>& sample_indices,
                               const vector<Index>& input_indices,
                               float* input_data,
                               bool parallelize,
                               int contiguous) const
{
    if (!streaming)
    {
        MaterializedDataset::fill_inputs(sample_indices, input_indices, input_data, parallelize, contiguous);
        return;
    }

    const Index batch_size = sample_indices.size();
    const Index height = input_shape[0];
    const Index width = input_shape[1];
    const Index channels = input_shape[2];
    const Index pixels_per_image = height * width * channels;

    string omp_error;

    #pragma omp parallel for schedule(dynamic) if (parallelize)
    for (Index i = 0; i < batch_size; ++i)
    {
        try
        {
            load_image(image_paths[sample_indices[i]],
                       input_data + i * pixels_per_image,
                       height, width, channels,
                       /*divide_by_255*/ true);
        }
        catch (const std::exception& e)
        {
            #pragma omp critical
            { omp_error = e.what(); }
            continue;
        }
    }

    if (!omp_error.empty())
        throw runtime_error(omp_error);
}

void ImageDataset::fill_targets(const vector<Index>& sample_indices,
                                const vector<Index>& target_indices,
                                float* target_data,
                                bool parallelize,
                                int contiguous) const
{
    if (!streaming)
    {
        MaterializedDataset::fill_targets(sample_indices, target_indices, target_data, parallelize, contiguous);
        return;
    }

    const Index batch_size = sample_indices.size();
    const Index targets_number = target_indices.size();

    if (targets_number == 1)
    {
        #pragma omp parallel for if (parallelize)
        for (Index i = 0; i < batch_size; ++i)
            target_data[i] = float(sample_labels[sample_indices[i]]);
    }
    else
    {
        std::fill_n(target_data, batch_size * targets_number, 0.0f);

        #pragma omp parallel for if (parallelize)
        for (Index i = 0; i < batch_size; ++i)
        {
            const Index label = sample_labels[sample_indices[i]];
            target_data[i * targets_number + label] = 1.0f;
        }
    }
}

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
