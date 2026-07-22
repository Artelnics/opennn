//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   I M A G E   D A T A S E T   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "image_dataset.h"
#include "device_backend.h"
#include "image_processing.h"
#include "tensor_types.h"
#include "string_utilities.h"
#include "random_utilities.h"
#include "io_utilities.h"

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

static filesystem::path image_cache_path(const filesystem::path& data_path)
{
    return data_path / ".cache" / "images.bin";
}

// Identity trailer appended after the pixel blob in images.bin (see read_images):
// sample count, image shape, the exact class list and the newest source mtime.
// Re-derived from the current folders on load and matched against the file's
// trailer, so a cache built for a different dataset — or for images since
// edited in place — is rebuilt instead of read as stale pixels.
static string image_cache_signature(Index samples, Index height, Index width, Index channels,
                                    const vector<filesystem::path>& class_folders,
                                    const filesystem::file_time_type& newest_write_time)
{
    string signature = to_string(samples) + "|"
        + to_string(height) + "x" + to_string(width) + "x" + to_string(channels) + "|"
        + to_string(newest_write_time.time_since_epoch().count()) + "|";
    for (const filesystem::path& folder : class_folders)
        signature += folder.filename().string() + ",";
    return signature;
}

static pair<float, float> scaling_affine(ScalerMethod scaler,
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
    }

    throw runtime_error("ImageDataset: invalid scaler method.");
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
    const vector<Index> input_indices = get_feature_indices("Input");
    const vector<Index> target_indices = get_feature_indices("Target");
    const Index inputs_number = ssize(input_indices);
    const Index targets_number = ssize(target_indices);

    vector<Index> all_samples(samples_number);
    iota(all_samples.begin(), all_samples.end(), 0);

    // Stage the rows the host fill_* path would produce for training batches
    // (scaled pixels followed by the one-hot targets), so the device gather
    // can replace the per-batch decode entirely.
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
    // "Dataset", not "ImageDataset": every dataset kind serializes under the
    // same root tag (see TabularDataset/LanguageDataset) — the model JSON's
    // section name does not vary per type, ModelType drives the dispatch.
    printer.open_element("Dataset");

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
    if (!augmentation.enabled || batch_size <= 0) return;

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
    const Json* image_dataset_element = get_json_root(data_set_document, "Dataset");

    const Json* data_source_element = require_json_field(image_dataset_element, "DataSource");

    set_data_path(read_json_string(data_source_element, "Path"));


    set_has_ids(read_json_bool(data_source_element, "HasSamplesId"));

    // A fresh editor-written model carries 0x0x0 here: the editor cannot know
    // the image geometry before the first import (the engine discovers it by
    // scanning the folder). And a resize request from the editor's large-image
    // popup arrives as {224, 224, 0} (channels unknown). So each positive
    // component is a request and each non-positive one means auto-detect; an
    // all-zero shape is no request at all.
    const Index requested_height   = read_json_index(data_source_element, "Height");
    const Index requested_width    = read_json_index(data_source_element, "Width");
    const Index requested_channels = read_json_index(data_source_element, "Channels");

    if (requested_height > 0 || requested_width > 0 || requested_channels > 0)
    {
        requested_input_shape = { requested_height, requested_width, requested_channels };

        if (requested_height > 0 && requested_width > 0 && requested_channels > 0)
            set_shape("Input", requested_input_shape);
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

    vector<filesystem::path> candidate_folders;

    for (const filesystem::directory_entry& current_directory : filesystem::directory_iterator(data_path))
        if (current_directory.is_directory()
            && !current_directory.path().filename().string().starts_with('.'))
            candidate_folders.emplace_back(current_directory.path());

    ranges::sort(candidate_folders);

    // Only directories that actually contain supported images count as classes.
    // Skipping empty/non-image folders keeps a stray sibling directory (e.g. a
    // text dataset's *.cache folder that landed in a shared data dir) from
    // inflating the class count and corrupting the target encoding.
    vector<filesystem::path> directory_path;
    vector<filesystem::path> paths;
    vector<int32_t> labels;
    // file_clock's epoch can sit in the future (libstdc++ uses 2174), making
    // real mtimes negative: min() is the only safe identity for the max-fold.
    filesystem::file_time_type newest_write_time = filesystem::file_time_type::min();

    for (const filesystem::path& folder : candidate_folders)
    {
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

        // Positive components are explicit requests (e.g. the editor's resize
        // to 224x224); non-positive ones fall back to the detected geometry.
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
        // Host-provided cache directory when set (e.g. the model's working dir)
        // so the user's image folder is not polluted; standalone keeps the cache
        // inside the dataset folder as before.
        cache_path = cache_directory.empty()
            ? image_cache_path(data_path)
            : cache_directory / (data_path.filename().string() + ".cache") / "images.bin";

        // images.bin layout: [pixels: samples×pixel_number bytes][identity trailer].
        // The pixel region stays at offset 0 (sample reads are unaffected); the
        // trailer is validated against the signature re-derived from the current
        // folders. A mismatch forces a rebuild instead of reading foreign pixels.
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
                cache_reader.read_at(trailer.data(), signature.size(), pixel_bytes);
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

void ImageDataset::write_image_cache(const vector<filesystem::path>& paths, const string& trailer)
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

    // Identity trailer, right after the pixel blob (see read_images cache check).
    writer.write(trailer.data(), trailer.size());

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

    // Contract with Optimizer::mark_validation_propagation: every Training AND
    // Validation batch leaves here scaled, because both forward passes skip the
    // network's Scaling layers. Inference fills stay raw for the Scaling layer.
    const bool apply_scaling = mode != FillMode::Inference;
    const bool has_scaling = ssize(input_scale) == channels
                          && ssize(input_offset) == channels;
    const bool use_default_scaling = apply_scaling && !has_scaling;
    const bool apply_augmentation = mode == FillMode::Training && augmentation.enabled;

    const auto scale_sample = [&](float* sample)
    {
        if (apply_scaling && has_scaling)
        {
            const Map<const Array<float, 1, Dynamic>> scale_row(input_scale.data(), 1, channels);
            const Map<const Array<float, 1, Dynamic>> offset_row(input_offset.data(), 1, channels);

            Map<MatrixR> image_pixels(sample, pixels_per_channel, channels);
            image_pixels.array().rowwise() *= scale_row;
            image_pixels.array().rowwise() += offset_row;
        }
        else if (use_default_scaling)
            Map<Array<float, Dynamic, 1>>(sample, pixels_per_image) *= 1.0f / 255.0f;
    };

    // Augmentation interpolates and zero-fills borders, so it must see raw
    // pixel values: only then does the affine wait for a separate pass.
    const bool scale_in_fill = !apply_augmentation && storage_mode != StorageMode::Matrix;

    if (storage_mode == StorageMode::Matrix)
    {
        fill_tensor_data(data, sample_indices, input_indices, input_data, contiguous);
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
                cache_reader.read_at(buf.data(), size_t(pixels_per_image), off);

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
                continue;
            }
        }

        throw_if(!omp_error.empty(),
                 omp_error);
    }

    if (apply_augmentation)
        augment_inputs(input_data, batch_size);

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
        for (Index i = 0; i < batch_size; ++i)
            target_data[i] = float(sample_labels[size_t(sample_indices[size_t(i)])]);
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
