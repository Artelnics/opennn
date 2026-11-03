// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Artificial Intelligence Techniques, S.L. and Roberto López González

#include "opennn/dataset/image_dataset.h"
#include "opennn/dataset/batch.h"
#include "opennn/core/log.h"

#include <utility>

#include "opennn/core/device_backend.h"
#include "opennn/core/io_utilities.h"
#include "opennn/core/random_utilities.h"
#include "opennn/core/scaling.h"
#include "opennn/core/string_utilities.h"
#include "opennn/core/tensor_types.h"
#include "opennn/dataset/image_processing.h"

namespace opennn
{

static bool has_configured_augmentation(const ImageAugmentationPolicy& policy)
{
    return policy.reflection_axis_x
        || policy.reflection_axis_y
        || policy.rotation_minimum != 0.0f
        || policy.rotation_maximum != 0.0f
        || policy.horizontal_translation_minimum != 0.0f
        || policy.horizontal_translation_maximum != 0.0f
        || policy.vertical_translation_minimum != 0.0f
        || policy.vertical_translation_maximum != 0.0f;
}

static void validate_augmentation_range(string_view name, float minimum, float maximum,
                                       double exclusive_limit = numeric_limits<double>::infinity())
{
    throw_if(!isfinite(minimum) || !isfinite(maximum),
             "Image augmentation {} bounds must be finite.", name);
    throw_if(minimum > maximum,
             "Image augmentation {} minimum must not exceed maximum.", name);
    throw_if(double(minimum) <= -exclusive_limit || double(maximum) >= exclusive_limit,
             "Image augmentation {} bounds must fit the signed pixel-index range (-{}, {}).",
             name, exclusive_limit, exclusive_limit);
}

static float sample_augmentation_value(RandomStream& random, float minimum, float maximum)
{
    if (minimum == maximum)
        return minimum;

    return float(lerp(double(minimum), double(maximum), double(random.uniform())));
}

static Index sample_augmentation_shift(RandomStream& random, float minimum, float maximum)
{
    return static_cast<Index>(llround(sample_augmentation_value(random, minimum, maximum)));
}

static ImageAffineTransform sample_image_transform(uint64_t seed, Index sample,
                                                   const ImageAugmentationPolicy& policy)
{
    RandomStream random(derive_random_seed(seed, uint64_t(sample)));
    ImageAffineTransform transform;
    transform.flip_x = policy.reflection_axis_x && random.uniform() < 0.5f;
    transform.flip_y = policy.reflection_axis_y && random.uniform() < 0.5f;
    if (policy.rotation_minimum != 0.0f || policy.rotation_maximum != 0.0f)
    {
        transform.angle = remainder(sample_augmentation_value(random, policy.rotation_minimum,
                                                               policy.rotation_maximum), 360.0f);
        const float radians = -transform.angle * numbers::pi_v<float> / 180.0f;
        transform.cosine = cos(radians);
        transform.sine = sin(radians);
    }
    if (policy.horizontal_translation_minimum != 0.0f || policy.horizontal_translation_maximum != 0.0f)
        transform.shift_x = sample_augmentation_shift(random, policy.horizontal_translation_minimum,
                                                       policy.horizontal_translation_maximum);
    if (policy.vertical_translation_minimum != 0.0f || policy.vertical_translation_maximum != 0.0f)
        transform.shift_y = sample_augmentation_shift(random, policy.vertical_translation_minimum,
                                                       policy.vertical_translation_maximum);
    return transform;
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

ImageDataset::ImageDataset(const filesystem::path& new_data_path)
    : ImageDataset(new_data_path, {})
{
}

ImageDataset::ImageDataset(const filesystem::path& new_data_path,
                           const Shape& new_input_shape) : Dataset()
{
    data_path = new_data_path;
    storage_mode = StorageMode::BinaryFile;
    requested_input_shape = new_input_shape;

    read_images();
}

void ImageDataset::set_augmentation_policy(const ImageAugmentationPolicy& new_policy)
{
    validate_augmentation_range("rotation", new_policy.rotation_minimum, new_policy.rotation_maximum);
    const double shift_limit = -double(numeric_limits<Index>::min());
    validate_augmentation_range("horizontal translation", new_policy.horizontal_translation_minimum,
                                new_policy.horizontal_translation_maximum, shift_limit);
    validate_augmentation_range("vertical translation", new_policy.vertical_translation_minimum,
                                new_policy.vertical_translation_maximum, shift_limit);
    if (is_device_resident()) disable_device_residency();

    augmentation_policy = new_policy;
}

void ImageDataset::enable_device_residency()
{
    if (!device::is_cuda_build()) return;
    if (augmentation_policy.enabled)
    {
        if (is_device_resident()) disable_device_residency();
        return;
    }
    upload_device_samples();
}

void ImageDataset::set_input_scaling(const vector<Descriptives>& descriptives,
                                     const vector<ScalerMethod>& scalers,
                                     float min_range,
                                     float max_range)
{
    const Index channels = get_channels_number();
    throw_if(ssize(descriptives) != channels || ssize(scalers) != channels,
             "ImageDataset::set_input_scaling: channel count mismatch.");

    disable_device_residency();
    input_scale.resize(size_t(channels));
    input_offset.resize(size_t(channels));

    for (Index i = 0; i < channels; ++i)
    {
        const AffineMap affine = scaling_affine(scalers[size_t(i)],
                                                    descriptives[size_t(i)],
                                                    min_range,
                                                    max_range);
        input_scale[size_t(i)] = affine.slope;
        input_offset[size_t(i)] = affine.offset;
    }
}

FeatureScaling ImageDataset::prepare_training_scaling(
    VariableRole role,
    const FeatureScaling& requested,
    Index expected_features)
{
    throw_if(role != VariableRole::Input,
             "ImageDataset supports training scaling only for inputs.");
    throw_if(expected_features != get_channels_number(),
             "ImageDataset training scaling expects {} channels, got {}.",
             get_channels_number(), expected_features);

    set_input_scaling(requested.descriptives,
                      requested.scalers,
                      requested.min_range,
                      requested.max_range);
    return requested;
}

void ImageDataset::clear_training_scaling() noexcept
{
    input_scale.clear();
    input_offset.clear();

    if (is_device_resident()) disable_device_residency();
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
        {"RandomAugmentation", to_string(augmentation_policy.enabled)},
        {"RandomReflectionAxisX", to_string(augmentation_policy.reflection_axis_x)},
        {"RandomReflectionAxisY", to_string(augmentation_policy.reflection_axis_y)},
        {"RandomRotationMinimum", to_string(augmentation_policy.rotation_minimum)},
        {"RandomRotationMaximum", to_string(augmentation_policy.rotation_maximum)},
        {"RandomHorizontalTranslationMinimum", to_string(augmentation_policy.horizontal_translation_minimum)},
        {"RandomHorizontalTranslationMaximum", to_string(augmentation_policy.horizontal_translation_maximum)},
        {"RandomVerticalTranslationMinimum", to_string(augmentation_policy.vertical_translation_minimum)},
        {"RandomVerticalTranslationMaximum", to_string(augmentation_policy.vertical_translation_maximum)},
        {"Codification", get_codification_string()},
        {"StorageMode", get_storage_mode_string()}
    });

    write_json_footer(printer);
}

void ImageDataset::augment_inputs(const span<float> input_data, Index batch_size) const
{
    augment_inputs(input_data, batch_size, {});
}

void ImageDataset::augment_inputs(span<float> input_data, Index batch_size, span<const Index> indices) const
{
    if (!augmentation_policy.enabled || batch_size <= 0) return;

    const Index height = input_shape[0];
    const Index width = input_shape[1];
    const Index channels = input_shape[2];
    const Index pixels = height * width * channels;

    throw_if(ssize(input_data) < batch_size * pixels,
             "ImageDataset::augment_inputs: buffer holds {} values but {} samples x {} pixels = {} are required.",
             ssize(input_data), batch_size, pixels, batch_size * pixels);

    float* const input_values = input_data.data();
    const uint64_t epoch_seed = get_augmentation_seed();

    const bool use_rotation = augmentation_policy.rotation_minimum != 0.0f
                           || augmentation_policy.rotation_maximum != 0.0f;
    const auto augment_sample = [&](Index i, Tensor3& scratch_storage)
    {
        const Index sample_index = indices.empty() ? i : indices[size_t(i)];
        const auto transform = sample_image_transform(epoch_seed, sample_index, augmentation_policy);
        float* sample = input_values + i * pixels;
        TensorMap3 image(sample, height, width, channels);

        if (transform.flip_x)
            reflect_image_horizontal(image);

        if (transform.flip_y)
            reflect_image_vertical(image);

        if (use_rotation)
        {
            copy_n(sample, pixels, scratch_storage.data());
            const TensorMap3 scratch(scratch_storage.data(), height, width, channels);
            rotate_image(scratch, image, transform.angle);
        }

        translate_image_x(image, transform.shift_x);
        translate_image_y(image, transform.shift_y);
    };

    #pragma omp parallel
    {
        Tensor3 scratch_storage;
        if (use_rotation)
            scratch_storage.resize(height, width, channels);

        #pragma omp for schedule(static)
        for (Index i = 0; i < batch_size; ++i)
            augment_sample(i, scratch_storage);
    }
}

void ImageDataset::from_JSON(const JsonDocument& data_set_document)
{
    const Json* const image_dataset_element = get_json_root(data_set_document, "Dataset");

    const Json* const data_source_element = require_json_field(image_dataset_element, "DataSource");

    set_data_path(read_json_string(data_source_element, "Path"));

    set_has_ids(read_json_bool(data_source_element, "HasSamplesId"));

    const Index requested_height = read_json_index(data_source_element, "Height");
    const Index requested_width = read_json_index(data_source_element, "Width");
    const Index requested_channels = read_json_index(data_source_element, "Channels");

    if (max({ requested_height, requested_width, requested_channels }) > 0)
        requested_input_shape = { requested_height, requested_width, requested_channels };
    else
        requested_input_shape.clear();

    if (min({ requested_height, requested_width, requested_channels }) > 0)
        set_shape(VariableRole::Input, requested_input_shape);

    set_codification(read_json_string(data_source_element, "Codification"));
    set_storage_mode(data_source_element->has("StorageMode")
                   ? read_json_string(data_source_element, "StorageMode")
                   : "BinaryFile");

    ImageAugmentationPolicy parsed_policy;
    parsed_policy.reflection_axis_x = read_json_bool(data_source_element, "RandomReflectionAxisX");
    parsed_policy.reflection_axis_y = read_json_bool(data_source_element, "RandomReflectionAxisY");
    parsed_policy.rotation_minimum = read_json_float(data_source_element, "RandomRotationMinimum");
    parsed_policy.rotation_maximum = read_json_float(data_source_element, "RandomRotationMaximum");
    parsed_policy.horizontal_translation_minimum = read_json_float(data_source_element, "RandomHorizontalTranslationMinimum");
    parsed_policy.horizontal_translation_maximum = read_json_float(data_source_element, "RandomHorizontalTranslationMaximum");
    parsed_policy.vertical_translation_minimum = read_json_float(data_source_element, "RandomVerticalTranslationMinimum");
    parsed_policy.vertical_translation_maximum = read_json_float(data_source_element, "RandomVerticalTranslationMaximum");
    parsed_policy.enabled = data_source_element->has("RandomAugmentation")
                          ? read_json_bool(data_source_element, "RandomAugmentation")
                          : has_configured_augmentation(parsed_policy);
    set_augmentation_policy(parsed_policy);

    const auto restore_without_samples = [this, image_dataset_element]()
    {
        read_json_blocks(image_dataset_element);

        sample_roles.clear();
        sample_labels.clear();
        data.resize(0, 0);

        set_shape(VariableRole::Target, { get_features_number(VariableRole::Target) });

        classes_number = 0;
        if (const vector<Variable> targets = get_variables(VariableRole::Target); !targets.empty())
            classes_number = uint32_t(targets.front().get_categories_number());

        pixel_number = 1;
        for (size_t axis = 0; axis < 3; ++axis)
            pixel_number *= uint64_t(input_shape[axis]);
    };

    const bool images_missing = requested_input_shape.get_rank() == 3
        && image_dataset_element->has("Variables")
        && !data_path.empty()
        && !filesystem::exists(data_path);

    if (images_missing)
    {
        logging::warning() << "ImageDataset: image directory \"" << data_path.string()
                           << "\" is missing; loading the dataset without samples for deployment use.\n";
        restore_without_samples();
        return;
    }

    read_images();

    read_json_blocks(image_dataset_element);
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
    invalidate_data();
    const chrono::high_resolution_clock::time_point start_time = chrono::high_resolution_clock::now();

    data.resize(0, 0);
    cache_reader.close();

    if (data_path.empty())
    {
        sample_labels.clear();
        sample_roles.clear();
        return;
    }

    const vector<filesystem::path> candidate_folders =
        list_directories(data_path, [](const filesystem::path& folder)
                         { return !folder.filename().string().starts_with('.'); });

    vector<filesystem::path> directory_path;
    vector<filesystem::path> paths;
    vector<int32_t> labels;
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
            paths.emplace_back(std::move(p));
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
        throw_if(requested_input_shape.get_rank() != 3,
                 "ImageDataset: requested input shape must be {height, width, channels}.");

        Index* const geometry[] = { &height, &width, &channels };

        for (size_t axis = 0; axis < 3; ++axis)
            if (const Index requested = requested_input_shape[axis]; requested > 0)
                *geometry[axis] = requested;

        throw_if(min({ height, width, channels }) <= 0,
                 "ImageDataset: image dimensions must be positive, got {}x{}x{}.",
                 height, width, channels);
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

    sample_labels = std::move(labels);

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
        const filesystem::path cache_folder = cache_directory.empty()
            ? data_path / ".cache"
            : cache_directory / (data_path.filename().string() + ".cache");

        cache_path = cache_folder / "images.bin";

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

        logging::info() << "\nImage dataset " << load_kind
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

optional<ImageTransformBatch> ImageDataset::get_gpu_image_transform(Index samples) const
{
    if (!augmentation_policy.enabled) return nullopt;
    ImageTransformBatch transform;
    transform.samples = transform.sources = samples;
    transform.height = transform.output_height = input_shape[0];
    transform.width = transform.output_width = input_shape[1];
    transform.channels = input_shape[2];
    transform.float_input = storage_mode == StorageMode::Matrix;
    transform.enabled = true;
    return transform;
}

void ImageDataset::fill_augmented_batch(Batch& batch, const vector<Index>& sample_indices,
                                        const FeatureSelection& features) const
{
    throw_if(ssize(sample_indices) != batch.batch_size,
             "fill_batch sample count does not match the batch size.");
    validate_augmentation_samples(sample_indices);
    const auto transform = *get_gpu_image_transform(batch.batch_size);
    const auto parameters = batch.prepare_image_transform(transform);
    const auto scales = batch.image_channel_scales();
    if (ssize(input_scale) == transform.channels && ssize(input_offset) == transform.channels)
        for (Index c = 0; c < transform.channels; ++c)
            scales[size_t(c)] = {input_scale[size_t(c)], input_offset[size_t(c)]};
    const uint64_t seed = get_augmentation_seed();
    for (Index i = 0; i < batch.batch_size; ++i)
    {
        auto& affine = parameters[size_t(i)].affine;
        affine = sample_image_transform(seed, sample_indices[size_t(i)], augmentation_policy);
        affine.shift_x = clamp(affine.shift_x, -transform.width, transform.width);
        affine.shift_y = clamp(affine.shift_y, -transform.height, transform.height);
        if (!transform.float_input)
            cache_reader.read_at(span(batch.input.host.as<uint8_t>() + size_t(i) * pixel_number,
                                      size_t(pixel_number)), uint64_t(sample_indices[size_t(i)]) * pixel_number);
    }
    if (transform.float_input)
        fill_tensor_data(data, sample_indices, features.inputs,
                          span(batch.input.host.as<float>(), size_t(batch.batch_size * input_shape.size())),
                          classify_column_contiguity(features.inputs));
    fill_targets(sample_indices, features.targets, batch.target.host.as<float>(), FillMode::Training);
}

void ImageDataset::fill_batch(Batch& batch,
                               const vector<Index>& sample_indices,
                               const FeatureSelection& features,
                               FillMode mode) const
{
    batch.image_transform.reset();
    batch.packed_image_input.reset();
    if (batch.uses_cuda() && mode == FillMode::Training && augmentation_policy.enabled)
        return fill_augmented_batch(batch, sample_indices, features);
    if (!batch.pack_image_input || !batch.uses_cuda() || !supports_byte_inputs()
        || mode != FillMode::Training)
        return Dataset::fill_batch(batch, sample_indices, features, mode);

    throw_if(ssize(sample_indices) != batch.batch_size,
             "fill_batch sample count does not match the batch size.");
    batch.device_gather.reset();
    PackedImageInput& packed = batch.packed_image_input.emplace();
    packed.channels = int(get_channels_number());
    // Training always scales: the configured affine map, or bytes to [0, 1].
    const bool configured = ssize(input_scale) == packed.channels
                         && ssize(input_offset) == packed.channels;
    for (int channel = 0; channel < packed.channels; ++channel)
    {
        packed.scale[size_t(channel)] = configured ? input_scale[size_t(channel)] : 1.0f / 255.0f;
        packed.offset[size_t(channel)] = configured ? input_offset[size_t(channel)] : 0.0f;
    }
    for (Index i = 0; i < batch.batch_size; ++i)
    {
        const Index index = sample_indices[size_t(i)];
        throw_if(index < 0 || index >= ssize(sample_labels),
                 "ImageDataset input sample index is out of range.");
        cache_reader.read_at(span(batch.input.host.as<uint8_t>() + size_t(i) * pixel_number,
                                  size_t(pixel_number)), uint64_t(index) * pixel_number);
    }
    fill_targets(sample_indices, features.targets, batch.target.host.as<float>(), mode);
}

void ImageDataset::fill_inputs(const vector<Index>& sample_indices,
                               const vector<Index>& input_indices,
                               float* input_data,
                               FillMode mode,
                               ColumnContiguity column_contiguity) const
{
    const Index batch_size = ssize(sample_indices);
    const Index channels = input_shape[2];
    const Index pixels_per_image = Index(pixel_number);
    const Index pixels_per_channel = pixels_per_image / channels;

    const span<float> input_span(input_data, size_t(batch_size * pixels_per_image));

    const bool apply_scaling = mode != FillMode::Inference;
    const bool has_scaling = ssize(input_scale) == channels
                          && ssize(input_offset) == channels;
    const bool apply_augmentation = mode == FillMode::Training && augmentation_policy.enabled;
    if (apply_augmentation) validate_augmentation_samples(sample_indices);

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
        fill_tensor_data(data, sample_indices, input_indices, input_span, column_contiguity);
    }
    else
    {
        string omp_error;

        const int workers = max(1, min(omp_get_max_threads(), to_int(batch_size)));

        #pragma omp parallel num_threads(workers)
        {
            vector<uint8_t> buffer(static_cast<size_t>(pixels_per_image));

            #pragma omp for schedule(dynamic)
            for (Index i = 0; i < batch_size; ++i)
            {
                try
                {
                    const Index sample_index = sample_indices[size_t(i)];
                    throw_if(sample_index < 0 || sample_index >= ssize(sample_labels),
                             "ImageDataset input sample index is out of range.");

                    const uint64_t off = uint64_t(sample_index) * pixel_number;
                    cache_reader.read_at(span(buffer), off);

                    float* dst = input_data + i * pixels_per_image;
                    Map<Array<float, Dynamic, 1>>(dst, pixels_per_image) =
                        Map<const Array<uint8_t, Dynamic, 1>>(
                            buffer.data(), pixels_per_image).cast<float>();

                    if (scale_in_fill)
                        scale_sample(dst);
                }
                catch (const exception& e)
                {
                    #pragma omp critical
                    { if (omp_error.empty()) omp_error = e.what(); }
                }
            }
        }

        throw_if(!omp_error.empty(),
                 omp_error);
    }

    if (apply_augmentation)
        augment_inputs(input_span, batch_size, sample_indices);

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
                                ColumnContiguity) const
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
