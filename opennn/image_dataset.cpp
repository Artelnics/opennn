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
#include "io_utilities.h"

namespace opennn
{

namespace {

bool has_augmentation_transform(const AugmentationSettings& augmentation)
{
    return augmentation.reflection_axis_x
        || augmentation.reflection_axis_y
        || augmentation.rotation_minimum != 0.0f
        || augmentation.rotation_maximum != 0.0f
        || augmentation.horizontal_translation_minimum != 0.0f
        || augmentation.horizontal_translation_maximum != 0.0f
        || augmentation.vertical_translation_minimum != 0.0f
        || augmentation.vertical_translation_maximum != 0.0f;
}

float sample_augmentation_value(float minimum, float maximum)
{
    if (minimum == maximum)
        return minimum;

    return random_uniform(min(minimum, maximum), max(minimum, maximum));
}

Index sample_augmentation_shift(float minimum, float maximum)
{
    return static_cast<Index>(lround(sample_augmentation_value(minimum, maximum)));
}

pair<float, float> scaling_affine(ScalerMethod scaler,
                                  const Descriptives& descriptives,
                                  float min_range,
                                  float max_range)
{
    switch (scaler)
    {
    case ScalerMethod::MinimumMaximum:
    {
        const float scale = (max_range - min_range)
                          / ((descriptives.maximum - descriptives.minimum) + EPSILON);
        return {scale, min_range - descriptives.minimum * scale};
    }
    case ScalerMethod::MeanStandardDeviation:
    {
        const float scale = 1.0f / (descriptives.standard_deviation + EPSILON);
        return {scale, -descriptives.mean * scale};
    }
    case ScalerMethod::StandardDeviation:
        return {1.0f / (descriptives.standard_deviation + EPSILON), 0.0f};
    case ScalerMethod::ImageMinMax:
        return {1.0f / 255.0f, 0.0f};
    case ScalerMethod::None:
    case ScalerMethod::Logarithm:
        return {1.0f, 0.0f};
    default:
        return {1.0f, 0.0f};
    }
}

}  // namespace

void ImageDataset::set_storage_mode(StorageMode new_storage_mode)
{
    Dataset::set_storage_mode(new_storage_mode);

    if (new_storage_mode == StorageMode::BinaryFile)
        data.resize(0, 0);
}

ImageDataset::ImageDataset(const filesystem::path& new_data_path) : Dataset()
{
    data_path = new_data_path;
    storage_mode = StorageMode::BinaryFile;

    read_images();
}

Index ImageDataset::get_channels_number() const
{
    return input_shape[2];
}

void ImageDataset::set_input_scaling(const vector<Descriptives>& descriptives,
                                     const vector<ScalerMethod>& scalers,
                                     float min_range,
                                     float max_range)
{
    const Index channels = get_channels_number();
    throw_if(ssize(descriptives) != channels || ssize(scalers) != channels,
             "ImageDataset::set_input_scaling: channel count mismatch.");

    input_scale.resize(size_t(channels));
    input_offset.resize(size_t(channels));

    for (Index i = 0; i < channels; ++i)
    {
        const auto [scale, offset] = scaling_affine(scalers[size_t(i)],
                                                    descriptives[size_t(i)],
                                                    min_range,
                                                    max_range);
        input_scale[size_t(i)] = scale;
        input_offset[size_t(i)] = offset;
    }
}

void ImageDataset::to_JSON(JsonWriter& printer) const
{
    printer.open_element("ImageDataset");

    printer.open_element("DataSource");

    write_json(printer, {
        {"FileType", "image"},
        {"Path", data_path.string()},
        {"HasSamplesId", to_string(has_sample_ids)},
        {"Channels", to_string(input_shape[2])},
        {"Width", to_string(input_shape[1])},
        {"Height", to_string(input_shape[0])},
        {"RandomAugmentation", to_string(augmentation.enabled)},
        {"RandomReflectionAxisX", to_string(augmentation.reflection_axis_x)},
        {"RandomReflectionAxisY", to_string(augmentation.reflection_axis_y)},
        {"RandomRotationMinimum", to_string(augmentation.rotation_minimum)},
        {"RandomRotationMaximum", to_string(augmentation.rotation_maximum)},
        {"RandomHorizontalTranslationMinimum", to_string(augmentation.horizontal_translation_minimum)},
        {"RandomHorizontalTranslationMaximum", to_string(augmentation.horizontal_translation_maximum)},
        {"RandomVerticalTranslationMinimum", to_string(augmentation.vertical_translation_minimum)},
        {"RandomVerticalTranslationMaximum", to_string(augmentation.vertical_translation_maximum)},
        {"Codification", get_codification_string()},
        {"StorageMode", get_storage_mode_string()}
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
    if (batch_size <= 0) return;

    const Index height = input_shape[0];
    const Index width = input_shape[1];
    const Index channels = input_shape[2];
    const Index pixels = height * width * channels;

    const bool use_rotation = augmentation.rotation_minimum != 0.0f
                           || augmentation.rotation_maximum != 0.0f;
    const bool use_horizontal_translation = augmentation.horizontal_translation_minimum != 0.0f
                                         || augmentation.horizontal_translation_maximum != 0.0f;
    const bool use_vertical_translation = augmentation.vertical_translation_minimum != 0.0f
                                       || augmentation.vertical_translation_maximum != 0.0f;

    const auto augment_sample = [&](Index i, Tensor3* scratch_storage)
    {
        float* sample = input_data + i * pixels;
        TensorMap3 image(sample, height, width, channels);

        if (augmentation.reflection_axis_x && random_bool(0.5f))
            reflect_image_horizontal(image);

        if (augmentation.reflection_axis_y && random_bool(0.5f))
            reflect_image_vertical(image);

        if (use_rotation)
        {
            copy_n(sample, pixels, scratch_storage->data());
            TensorMap3 scratch(scratch_storage->data(), height, width, channels);
            rotate_image(scratch, image, sample_augmentation_value(augmentation.rotation_minimum,
                                                                   augmentation.rotation_maximum));
        }

        if (use_horizontal_translation)
            translate_image_x(image, sample_augmentation_shift(augmentation.horizontal_translation_minimum,
                                                               augmentation.horizontal_translation_maximum));

        if (use_vertical_translation)
            translate_image_y(image, sample_augmentation_shift(augmentation.vertical_translation_minimum,
                                                               augmentation.vertical_translation_maximum));
    };

    #pragma omp parallel
    {
        unique_ptr<Tensor3> scratch_storage;
        if (use_rotation)
            scratch_storage = make_unique<Tensor3>(height, width, channels);

        #pragma omp for schedule(static)
        for (Index i = 0; i < batch_size; ++i)
            augment_sample(i, scratch_storage.get());
    }
}

void ImageDataset::from_JSON(const JsonDocument& data_set_document)
{
    const Json* image_dataset_element = get_json_root(data_set_document, "ImageDataset");

    const Json* data_source_element = require_json_field(image_dataset_element, "DataSource");

    set_data_path(read_json_string(data_source_element, "Path"));

    if (data_source_element->has("Streaming"))
        (void)read_json_bool(data_source_element, "Streaming");

    set_has_ids(read_json_bool(data_source_element, "HasSamplesId"));

    set_shape("Input", { read_json_index(data_source_element, "Height"),
                         read_json_index(data_source_element, "Width"),
                         read_json_index(data_source_element, "Channels") });

    set_codification(read_json_string(data_source_element, "Codification"));
    set_storage_mode(data_source_element->has("StorageMode")
                   ? read_json_string(data_source_element, "StorageMode")
                   : "BinaryFile");

    augmentation.reflection_axis_x = read_json_bool(data_source_element, "RandomReflectionAxisX");
    augmentation.reflection_axis_y = read_json_bool(data_source_element, "RandomReflectionAxisY");
    augmentation.rotation_minimum = read_json_type(data_source_element, "RandomRotationMinimum");
    augmentation.rotation_maximum = read_json_type(data_source_element, "RandomRotationMaximum");
    augmentation.horizontal_translation_minimum = read_json_type(data_source_element, "RandomHorizontalTranslationMinimum");
    augmentation.horizontal_translation_maximum = read_json_type(data_source_element, "RandomHorizontalTranslationMaximum");
    augmentation.vertical_translation_minimum = read_json_type(data_source_element, "RandomVerticalTranslationMinimum");
    augmentation.vertical_translation_maximum = read_json_type(data_source_element, "RandomVerticalTranslationMaximum");
    augmentation.enabled = data_source_element->has("RandomAugmentation")
                         ? read_json_bool(data_source_element, "RandomAugmentation")
                         : has_augmentation_transform(augmentation);

    read_images();
}

VectorI ImageDataset::calculate_target_distribution() const
{
    VectorI distribution = VectorI::Zero(Index(classes_number));

    for (int32_t label : sample_labels)
        if (label >= 0 && label < distribution.size())
            distribution(label)++;

    return distribution;
}

void ImageDataset::read_images()
{
    const chrono::high_resolution_clock::time_point start_time = chrono::high_resolution_clock::now();

    data.resize(0, 0);
    cache_reader.close();

    vector<filesystem::path> directory_path;

    for (const filesystem::directory_entry& current_directory : filesystem::directory_iterator(data_path))
        if (current_directory.is_directory()
            && !current_directory.path().filename().string().starts_with('.'))
            directory_path.emplace_back(current_directory.path());

    ranges::sort(directory_path);

    const Index folders_number = directory_path.size();

    throw_if(folders_number < 2, 
        "ImageDataset: image classification requires at least two class folders.");

    vector<filesystem::path> paths;
    vector<Index> labels;

    for (Index i = 0; i < folders_number; ++i)
    {
        vector<filesystem::path> folder_files;
        for (const filesystem::directory_entry& current_directory : filesystem::directory_iterator(directory_path[i]))
            if (current_directory.is_regular_file() && is_supported_image_file(current_directory.path()))
                folder_files.emplace_back(current_directory.path());

        ranges::sort(folder_files);
        for (auto& p : folder_files)
        {
            paths.emplace_back(std::move(p));
            labels.push_back(i);
        }
    }

    const Index samples_number = paths.size();

    throw_if(samples_number == 0, "No images in folder.");

    const Tensor3 first_image = load_image(paths[0]);

    const Index height = first_image.dimension(0);
    const Index width = first_image.dimension(1);
    const Index channels = first_image.dimension(2);

    const Index pixels_number = height * width * channels;

    const Index targets_number = (folders_number == 2) ? 1 : folders_number;

    input_shape  = { height, width, channels };
    target_shape = { targets_number };
    pixel_number = uint64_t(pixels_number);
    classes_number = uint32_t(folders_number);

    variables.resize(pixels_number + 1);
    for (Index i = 0; i < pixels_number; ++i)
    {
        variables[i].type = VariableType::Numeric;
        variables[i].name = format("variable_{}", i + 1);
        variables[i].role = VariableRole::Input;
    }

    vector<string> categories(folders_number);
    ranges::transform(directory_path, categories.begin(),
                      [](const filesystem::path& p) { return p.filename().string(); });

    const bool binary_target = (targets_number == 1);

    Variable& target_variable = variables[pixels_number];
    target_variable.name = binary_target ? categories[0] + "_" + categories[1] : "Class";
    target_variable.role = VariableRole::Target;
    target_variable.type = binary_target ? VariableType::Binary : VariableType::Categorical;
    target_variable.set_categories(categories);
    target_variable.scaler = ScalerMethod::None;

    sample_labels.resize(size_t(samples_number));
    for (Index i = 0; i < samples_number; ++i)
        sample_labels[size_t(i)] = int32_t(labels[size_t(i)]);

    sample_roles.assign(samples_number, SampleRole::Training);

    string load_kind;

    if (storage_mode == StorageMode::Matrix)
    {
        // Decode every image straight into the base data matrix (samples x pixels).
        data.resize(samples_number, pixels_number);

        for (Index i = 0; i < samples_number; ++i)
        {
            load_image(paths[i], &data(i, 0), height, width, channels);

            if (display && (i % 1000 == 0 || i + 1 == samples_number))
                display_progress_bar(i + 1, samples_number);
        }

        load_kind = "loaded into memory";
    }
    else
    {
        // BinaryFile: keep raw uint8 pixels in a .bin and read them lazily per batch.
        cache_path = data_path / ".cache" / "images.bin";

        const bool cache_valid = filesystem::exists(cache_path)
            && filesystem::file_size(cache_path) == uint64_t(samples_number) * pixel_number;

        if (!cache_valid)
            write_image_cache(paths);

        cache_reader.open(cache_path);

        load_kind = cache_valid ? "loaded from cache" : "cache built";
    }

    split_samples_random();

    if (display)
    {
        const long long total_milliseconds = chrono::duration_cast<chrono::milliseconds>(
            chrono::high_resolution_clock::now() - start_time).count();

        const long long minutes = total_milliseconds / 60000;
        const long long seconds = (total_milliseconds % 60000) / 1000;
        const long long milliseconds = total_milliseconds % 1000;

        cout << "\nImage dataset " << load_kind
             << " in: " << minutes << " minutes, "
             << seconds << " seconds, "
             << milliseconds << " milliseconds.\n";
    }
}

void ImageDataset::write_image_cache(const vector<filesystem::path>& paths)
{
    const Index samples_number = ssize(paths);
    const Index height = input_shape[0];
    const Index width = input_shape[1];
    const Index channels = input_shape[2];
    const Index pixels_number = Index(pixel_number);

    filesystem::create_directories(cache_path.parent_path());
    const filesystem::path tmp_path = cache_path.string() + ".tmp";

    FileWriter writer;
    writer.open(tmp_path);

    vector<float> tmp(static_cast<size_t>(pixels_number));
    vector<uint8_t> pixels(static_cast<size_t>(pixels_number));

    for (Index i = 0; i < samples_number; ++i)
    {
        load_image(paths[i], tmp.data(), height, width, channels);

        Map<Array<uint8_t, Dynamic, 1>>(pixels.data(), pixels_number) =
            (Map<const Array<float, Dynamic, 1>>(tmp.data(), pixels_number)
                .max(0.0f).min(255.0f) + 0.5f).cast<uint8_t>();

        writer.write(pixels.data(), pixels.size());

        if (display && (i % 1000 == 0 || i + 1 == samples_number))
            display_progress_bar(i + 1, samples_number);
    }

    writer.finish_with_rename(cache_path);
}

void ImageDataset::fill_inputs(const vector<Index>& sample_indices,
                               const vector<Index>& input_indices,
                               float* input_data,
                               bool is_training,
                               int contiguous) const
{
    const Index batch_size = ssize(sample_indices);
    const Index channels = input_shape[2];
    const Index pixels_per_image = Index(pixel_number);
    const bool apply_scaling = is_training;
    const bool has_scaling = ssize(input_scale) == channels
                          && ssize(input_offset) == channels;
    const bool use_default_scaling = apply_scaling && !has_scaling;
    const bool apply_augmentation = is_training && augmentation.enabled;

    if (storage_mode == StorageMode::Matrix)
    {
        // Pixels live in the base data matrix; copy them like a tabular dataset.
        fill_tensor_data(data, sample_indices, input_indices, input_data, contiguous);
    }
    else
    {
        // BinaryFile: read each image's raw uint8 pixels lazily from the .bin.
        string omp_error;

        #pragma omp parallel for schedule(dynamic)
        for (Index i = 0; i < batch_size; ++i)
        {
            try
            {
                thread_local vector<uint8_t> buf;
                buf.resize(size_t(pixels_per_image));

                const Index sample_index = sample_indices[size_t(i)];
                throw_if(sample_index < 0 || sample_index >= ssize(sample_labels),
                         "ImageDataset input sample index is out of range.");

                const uint64_t off = uint64_t(sample_index) * pixel_number;
                cache_reader.read_at(buf.data(), size_t(pixels_per_image), off);

                float* dst = input_data + i * pixels_per_image;
                Map<Array<float, Dynamic, 1>>(dst, pixels_per_image) =
                    Map<const Array<uint8_t, Dynamic, 1>>(buf.data(), pixels_per_image).cast<float>();
            }
            catch (const exception& e)
            {
                #pragma omp critical
                { omp_error = e.what(); }
                continue;
            }
        }

        throw_if(!omp_error.empty(),
                 omp_error);
    }

    if (apply_augmentation)
        augment_inputs(input_data, batch_size);

    if (apply_scaling && has_scaling)
    {
        const Index pixels_per_channel = pixels_per_image / channels;
        const Map<const Array<float, 1, Dynamic>> scale_row(input_scale.data(), 1, channels);
        const Map<const Array<float, 1, Dynamic>> offset_row(input_offset.data(), 1, channels);

        #pragma omp parallel for schedule(static)
        for (Index i = 0; i < batch_size; ++i)
        {
            Map<MatrixR> image_pixels(input_data + i * pixels_per_image, pixels_per_channel, channels);
            image_pixels.array().rowwise() *= scale_row;
            image_pixels.array().rowwise() += offset_row;
        }
    }
    else if (use_default_scaling)
    {
        #pragma omp parallel for schedule(static)
        for (Index i = 0; i < batch_size; ++i)
            Map<Array<float, Dynamic, 1>>(input_data + i * pixels_per_image, pixels_per_image) *= 1.0f / 255.0f;
    }
}

void ImageDataset::fill_targets(const vector<Index>& sample_indices,
                                const vector<Index>& target_indices,
                                float* target_data,
                                bool /*is_training*/,
                                int /*contiguous*/) const
{
    const Index batch_size = ssize(sample_indices);
    const Index targets_number = ssize(target_indices);

    if (targets_number == 0) return;

    for (const Index sample_index : sample_indices)
        throw_if(sample_index < 0 || sample_index >= ssize(sample_labels),
                 "ImageDataset target sample index is out of range.");

    if (targets_number > 1)
        for (const Index sample_index : sample_indices)
        {
            const int32_t label = sample_labels[size_t(sample_index)];
            throw_if(label < 0 || label >= targets_number,
                     "ImageDataset target label is out of range.");
        }

    if (targets_number == 1)
    {
        auto label_of = [&](Index s) { return float(sample_labels[size_t(s)]); };
        transform(execution::par, sample_indices.begin(), sample_indices.begin() + batch_size, target_data, label_of);
    }
    else
    {
        fill_n(target_data, batch_size * targets_number, 0.0f);

        #pragma omp parallel for
        for (Index i = 0; i < batch_size; ++i)
        {
            const int32_t label = sample_labels[size_t(sample_indices[i])];
            target_data[i * targets_number + label] = 1.0f;
        }
    }
}

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
