//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   I M A G E   D A T A S E T   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "opennn/dataset/image_dataset.h"
#include "opennn/core/device_backend.h"
#include "opennn/dataset/image_processing.h"
#include "opennn/core/scaling.h"
#include "opennn/core/tensor_types.h"
#include "opennn/core/string_utilities.h"
#include "opennn/core/random_utilities.h"
#include "opennn/core/io_utilities.h"

namespace opennn
{

static bool has_augmentation_transform(const AugmentationSettings& augmentation)
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

static float sample_augmentation_value(float minimum, float maximum)
{
    if (minimum == maximum)
        return minimum;

    return random_uniform(min(minimum, maximum), max(minimum, maximum));
}

static Index sample_augmentation_shift(float minimum, float maximum)
{
    return static_cast<Index>(lround(sample_augmentation_value(minimum, maximum)));
}

static string image_cache_signature(Index samples, Index height, Index width, Index channels,
                                    const vector<filesystem::path>& class_folders,
                                    const filesystem::file_time_type& newest_write_time)
{
    string signature = to_string(samples) + "|"
        + to_string(height) + "x" + to_string(width) + "x" + to_string(channels) + "|"
        + to_string(static_cast<long long>(newest_write_time.time_since_epoch().count())) + "|";
    for (const filesystem::path& folder : class_folders)
        signature += folder.filename().string() + ",";
    return signature;
}

ImageDataset::ImageDataset(const filesystem::path& new_data_path) : Dataset()
{
    data_path = new_data_path;
    storage_mode = StorageMode::BinaryFile;

    read_images();
}

ImageDataset::ImageDataset(const filesystem::path& new_data_path,
                           const Shape& new_input_shape) : Dataset()
{
    data_path = new_data_path;
    storage_mode = StorageMode::BinaryFile;
    requested_input_shape = new_input_shape;

    read_images();
}

Index ImageDataset::get_channels_number() const
{
    return input_shape[2];
}

void ImageDataset::enable_device_residency()
{
    if (!device::is_cuda_build()) return;
    if (is_device_resident()) return;
    if (augmentation.enabled) return;
    if (get_samples_number() == 0) return;

    const Index samples_number = get_samples_number();
    const vector<Index> input_indices = get_feature_indices(VariableRole::Input);
    const vector<Index> target_indices = get_feature_indices(VariableRole::Target);
    const Index inputs_number = ssize(input_indices);
    const Index targets_number = ssize(target_indices);

    vector<Index> all_samples(samples_number);
    iota(all_samples.begin(), all_samples.end(), 0);

    MatrixR inputs(samples_number, inputs_number);
    fill_inputs(all_samples, input_indices, inputs.data(), FillMode::Training, 1);

    MatrixR targets(samples_number, targets_number);
    fill_targets(all_samples, target_indices, targets.data(), FillMode::Training, 1);

    MatrixR staged(samples_number, inputs_number + targets_number);
    staged.leftCols(inputs_number) = inputs;
    staged.rightCols(targets_number) = targets;

    upload_device_matrix(staged);
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
    write_json_header(printer, {
        {"FileType", "image"},
        {"Path", data_path.string()},
        {"HasSamplesId", has_sample_ids},
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

    write_json_footer(printer);
}

void ImageDataset::augment_inputs(const span<float> input_data, Index batch_size) const
{
    if (!augmentation.enabled || batch_size <= 0) return;

    const Index height = input_shape[0];
    const Index width = input_shape[1];
    const Index channels = input_shape[2];
    const Index pixels = height * width * channels;

    throw_if(ssize(input_data) < batch_size * pixels,
             "ImageDataset::augment_inputs: buffer holds {} values but {} samples x {} pixels = {} are required.",
             ssize(input_data), batch_size, pixels, batch_size * pixels);

    float* const input_values = input_data.data();

    const bool use_rotation = augmentation.rotation_minimum != 0.0f
                           || augmentation.rotation_maximum != 0.0f;
    const bool use_horizontal_translation = augmentation.horizontal_translation_minimum != 0.0f
                                         || augmentation.horizontal_translation_maximum != 0.0f;
    const bool use_vertical_translation = augmentation.vertical_translation_minimum != 0.0f
                                       || augmentation.vertical_translation_maximum != 0.0f;

    const auto augment_sample = [&](Index i, Tensor3* scratch_storage)
    {
        float* sample = input_values + i * pixels;
        TensorMap3 image(sample, height, width, channels);

        if (augmentation.reflection_axis_x && random_bool(0.5f))
            reflect_image_horizontal(image);

        if (augmentation.reflection_axis_y && random_bool(0.5f))
            reflect_image_vertical(image);

        if (use_rotation)
        {
            copy_n(sample, pixels, scratch_storage->data());
            const TensorMap3 scratch(scratch_storage->data(), height, width, channels);
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
    const Json* const image_dataset_element = get_json_root(data_set_document, "Dataset");

    const Json* const data_source_element = require_json_field(image_dataset_element, "DataSource");

    set_data_path(read_json_string(data_source_element, "Path"));

    set_has_ids(read_json_bool(data_source_element, "HasSamplesId"));

    const Index requested_height   = read_json_index(data_source_element, "Height");
    const Index requested_width    = read_json_index(data_source_element, "Width");
    const Index requested_channels = read_json_index(data_source_element, "Channels");

    if (requested_height > 0 || requested_width > 0 || requested_channels > 0)
    {
        requested_input_shape = { requested_height, requested_width, requested_channels };

        if (requested_height > 0 && requested_width > 0 && requested_channels > 0)
            set_shape(VariableRole::Input, requested_input_shape);
    }
    else
        requested_input_shape.clear();

    set_codification(read_json_string(data_source_element, "Codification"));
    set_storage_mode(data_source_element->has("StorageMode")
                   ? read_json_string(data_source_element, "StorageMode")
                   : "BinaryFile");

    augmentation.reflection_axis_x = read_json_bool(data_source_element, "RandomReflectionAxisX");
    augmentation.reflection_axis_y = read_json_bool(data_source_element, "RandomReflectionAxisY");
    augmentation.rotation_minimum = read_json_float(data_source_element, "RandomRotationMinimum");
    augmentation.rotation_maximum = read_json_float(data_source_element, "RandomRotationMaximum");
    augmentation.horizontal_translation_minimum = read_json_float(data_source_element, "RandomHorizontalTranslationMinimum");
    augmentation.horizontal_translation_maximum = read_json_float(data_source_element, "RandomHorizontalTranslationMaximum");
    augmentation.vertical_translation_minimum = read_json_float(data_source_element, "RandomVerticalTranslationMinimum");
    augmentation.vertical_translation_maximum = read_json_float(data_source_element, "RandomVerticalTranslationMaximum");
    augmentation.enabled = data_source_element->has("RandomAugmentation")
                         ? read_json_bool(data_source_element, "RandomAugmentation")
                         : has_augmentation_transform(augmentation);

    read_images();
}

VectorI ImageDataset::calculate_target_distribution() const
{
    VectorI distribution = VectorI::Zero(Index(classes_number));

    for (const int32_t label : sample_labels)
        if (label >= 0 && label < distribution.size())
            distribution(label)++;

    return distribution;
}

void ImageDataset::read_images()
{
    const chrono::high_resolution_clock::time_point start_time = chrono::high_resolution_clock::now();

    data.resize(0, 0);
    cache_reader.close();

    const vector<filesystem::path> candidate_folders =
        list_directories(data_path, [](const filesystem::path& folder)
                         { return !folder.filename().string().starts_with('.'); });

    vector<filesystem::path> directory_path;
    vector<filesystem::path> paths;
    vector<int32_t> labels;
    filesystem::file_time_type newest_write_time = filesystem::file_time_type::min();

    for (const filesystem::path& folder : candidate_folders)
    {
        // Not list_files: this pass also folds newest_write_time, and splitting it
        // would cost a second last_write_time() stat per image.
        vector<filesystem::path> folder_files;
        for (const filesystem::directory_entry& current_directory : filesystem::directory_iterator(folder))
            if (current_directory.is_regular_file() && is_supported_image_file(current_directory.path()))
            {
                folder_files.emplace_back(current_directory.path());
                newest_write_time = max(newest_write_time, current_directory.last_write_time());
            }

        if (folder_files.empty())
            continue;

        ranges::sort(folder_files);
        const int32_t class_index = int32_t(directory_path.size());
        directory_path.push_back(folder);
        for (auto& p : folder_files)
        {
            paths.emplace_back(move(p));
            labels.push_back(class_index);
        }
    }

    const Index folders_number = directory_path.size();

    throw_if(folders_number < 2,
        "ImageDataset: image classification requires at least two non-empty class folders.");

    const Index samples_number = paths.size();

    throw_if(samples_number == 0, "No images in folder.");

    const Tensor3 first_image = load_image(paths[0]);

    Index height = first_image.dimension(0);
    Index width = first_image.dimension(1);
    Index channels = first_image.dimension(2);

    if (!requested_input_shape.empty())
    {
        throw_if(requested_input_shape.rank != 3,
                 "ImageDataset: requested input shape must be {height, width, channels}.");

        if (requested_input_shape[0] > 0) height   = requested_input_shape[0];
        if (requested_input_shape[1] > 0) width    = requested_input_shape[1];
        if (requested_input_shape[2] > 0) channels = requested_input_shape[2];

        throw_if(height <= 0 || width <= 0 || channels <= 0,
                 "ImageDataset: image dimensions must be positive.");
    }

    const Index pixels_number = height * width * channels;

    const Index targets_number = (folders_number == 2) ? 1 : folders_number;

    input_shape  = { height, width, channels };
    target_shape = { targets_number };
    pixel_number = uint64_t(pixels_number);
    classes_number = uint32_t(folders_number);

    variables.assign(2, Variable());

    Variable& image_variable = variables[0];
    image_variable.name = "image";
    image_variable.type = VariableType::Numeric;
    image_variable.role = VariableRole::Input;
    image_variable.features = pixels_number;

    vector<string> categories(folders_number);
    ranges::transform(directory_path, categories.begin(),
                      [](const filesystem::path& p) { return p.filename().string(); });

    const bool binary_target = (targets_number == 1);

    Variable& target_variable = variables[1];
    target_variable.name = binary_target ? categories[0] + "_" + categories[1] : "Class";
    target_variable.role = VariableRole::Target;
    target_variable.type = binary_target ? VariableType::Binary : VariableType::Categorical;
    target_variable.set_categories(categories);
    target_variable.scaler = ScalerMethod::None;

    sample_labels = move(labels);

    sample_roles.assign(samples_number, SampleRole::Training);

    string load_kind;

    if (storage_mode == StorageMode::Matrix)
    {
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
        cache_path = cache_directory.empty()
            ? data_path / ".cache" / "images.bin"
            : cache_directory / (data_path.filename().string() + ".cache") / "images.bin";

        const string signature = image_cache_signature(samples_number, height, width, channels,
                                                       directory_path, newest_write_time);
        const uint64_t pixel_bytes = uint64_t(samples_number) * pixel_number;

        bool cache_valid = false;

        if (filesystem::exists(cache_path))
        {
            cache_reader.open(cache_path);
            const uint64_t total_bytes = cache_reader.file_size();

            if (total_bytes > pixel_bytes && total_bytes - pixel_bytes == signature.size())
            {
                string trailer(signature.size(), '\0');
                cache_reader.read_at(span(trailer), pixel_bytes);
                cache_valid = (trailer == signature);
            }

            if (!cache_valid)
                cache_reader.close();
        }

        if (!cache_valid)
        {
            write_image_cache(paths, signature);
            cache_reader.open(cache_path);
        }

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

void ImageDataset::write_image_cache(const vector<filesystem::path>& paths, const string& trailer) const
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

        writer.write(span(pixels));

        if (display && (i % 1000 == 0 || i + 1 == samples_number))
            display_progress_bar(i + 1, samples_number);
    }

    writer.write(span(trailer));

    writer.finish_with_rename(cache_path);
}

void ImageDataset::fill_inputs(const vector<Index>& sample_indices,
                               const vector<Index>& input_indices,
                               float* input_data,
                               FillMode mode,
                               int contiguous) const
{
    const Index batch_size = ssize(sample_indices);
    const Index channels = input_shape[2];
    const Index pixels_per_image = Index(pixel_number);
    const Index pixels_per_channel = pixels_per_image / channels;

    const span<float> input_span(input_data, size_t(batch_size * pixels_per_image));

    const bool apply_scaling = mode != FillMode::Inference;
    const bool has_scaling = ssize(input_scale) == channels
                          && ssize(input_offset) == channels;
    const bool apply_augmentation = mode == FillMode::Training && augmentation.enabled;

    const auto scale_sample = [&](float* sample)
    {
        if (!apply_scaling) return;

        if (!has_scaling)
        {
            Map<Array<float, Dynamic, 1>>(sample, pixels_per_image) *= 1.0f / 255.0f;
            return;
        }

        const Map<const Array<float, 1, Dynamic>> scale_row(input_scale.data(), 1, channels);
        const Map<const Array<float, 1, Dynamic>> offset_row(input_offset.data(), 1, channels);

        Map<MatrixR> image_pixels(sample, pixels_per_channel, channels);
        image_pixels.array().rowwise() *= scale_row;
        image_pixels.array().rowwise() += offset_row;
    };

    const bool scale_in_fill = !apply_augmentation && storage_mode != StorageMode::Matrix;

    if (storage_mode == StorageMode::Matrix)
    {
        fill_tensor_data(data, sample_indices, input_indices, input_span, contiguous);
    }
    else
    {
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
                cache_reader.read_at(span(buf), off);

                float* dst = input_data + i * pixels_per_image;
                Map<Array<float, Dynamic, 1>>(dst, pixels_per_image) =
                    Map<const Array<uint8_t, Dynamic, 1>>(buf.data(), pixels_per_image).cast<float>();

                if (scale_in_fill)
                    scale_sample(dst);
            }
            catch (const exception& e)
            {
                #pragma omp critical
                { omp_error = e.what(); }
            }
        }

        throw_if(!omp_error.empty(),
                 omp_error);
    }

    if (apply_augmentation)
        augment_inputs(input_span, batch_size);

    if (!scale_in_fill)
    {
        #pragma omp parallel for schedule(static)
        for (Index i = 0; i < batch_size; ++i)
            scale_sample(input_data + i * pixels_per_image);
    }
}

void ImageDataset::fill_targets(const vector<Index>& sample_indices,
                                const vector<Index>& target_indices,
                                float* target_data,
                                FillMode,
                                int) const
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
        ranges::transform(sample_indices, target_data,
                          [this](Index sample_index) { return float(sample_labels[size_t(sample_index)]); });
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
