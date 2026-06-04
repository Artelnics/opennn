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

#pragma pack(push, 1)
struct ImageCacheHeader
{
    char     magic[8];        //  "OPENNNIM"
    uint32_t version;
    uint32_t height;
    uint32_t width;
    uint32_t channels;
    uint64_t num_samples;
    uint64_t record_bytes;    // = height * width * channels
    uint64_t labels_off;      // = sizeof(header) + num_samples * record_bytes
    uint32_t num_classes;
    uint8_t  pad[12];         // pad to 64
};
#pragma pack(pop)
static_assert(sizeof(ImageCacheHeader) == 64, "ImageCacheHeader must be 64 bytes");

constexpr uint32_t IMAGE_CACHE_VERSION = 1;
constexpr const char IMAGE_CACHE_MAGIC[8] = {'O','P','E','N','N','N','I','M'};

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

bool read_header(FileReader& reader, ImageCacheHeader& header)
{
    if (reader.file_size() < sizeof(ImageCacheHeader)) return false;
    reader.read_at(&header, sizeof(header), 0);
    if (memcmp(header.magic, IMAGE_CACHE_MAGIC, 8) != 0) return false;
    if (header.version != IMAGE_CACHE_VERSION) return false;
    return true;
}

bool header_matches_request(const ImageCacheHeader& header,
                            uint32_t expected_height,
                            uint32_t expected_width,
                            uint32_t expected_channels)
{
    if (expected_height   != 0 && header.height   != expected_height)   return false;
    if (expected_width    != 0 && header.width    != expected_width)    return false;
    if (expected_channels != 0 && header.channels != expected_channels) return false;
    return true;
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

ImageDataset::ImageDataset(const Index new_samples_number,
                           const Shape& new_input_shape,
                           const Shape& new_target_shape)
{
    if (new_samples_number == 0)
        return;

    throw_if(new_input_shape.rank != 3,
             "Input shape is not 3");

    throw_if(new_target_shape.rank != 1,
             "Target shape is not 1");

    input_shape = new_input_shape;
    target_shape = new_target_shape;
    record_bytes = uint64_t(input_shape.size());
    classes_number = uint32_t(target_shape[0] == 1 ? 2 : target_shape[0]);

    variables.resize(input_shape.size() + 1);
    for (Index i = 0; i < input_shape.size(); ++i)
    {
        variables[i].type = VariableType::Numeric;
        variables[i].name = format("variable_{}", i + 1);
        variables[i].role = VariableRole::Input;
    }

    Variable& target_variable = variables[input_shape.size()];
    target_variable.name = "Class";
    target_variable.role = VariableRole::Target;
    target_variable.type = target_shape[0] == 1 ? VariableType::Binary : VariableType::Categorical;
    target_variable.scaler = ScalerMethod::None;

    vector<string> categories{size_t(classes_number)};
    for (size_t i = 0; i < categories.size(); ++i)
        categories[i] = to_string(i);
    target_variable.set_categories(categories);

    labels_ram.resize(size_t(new_samples_number));
    for (Index i = 0; i < new_samples_number; ++i)
        labels_ram[size_t(i)] = int32_t(i % classes_number);

    sample_roles.assign(size_t(new_samples_number), SampleRole::Training);
    split_samples_random();
}

void ImageDataset::set_data_random()
{
    const Index samples_number = ssize(labels_ram);
    throw_if(samples_number == 0,
             "ImageDataset::set_data_random: dataset has no samples; use the (samples, input_shape, target_shape) constructor first.");
    throw_if(record_bytes == 0 || input_shape.rank != 3,
             "ImageDataset::set_data_random: input_shape is not set.");

    cache_reader.close();

    const filesystem::path temp_dir = filesystem::temp_directory_path();
    cache_path = temp_dir / format("opennn_imagedataset_random_{}.bin",
                                   random_integer(0, 1000000000));
    const filesystem::path tmp_path = cache_path.string() + ".tmp";

    ImageCacheHeader header{};
    memcpy(header.magic, IMAGE_CACHE_MAGIC, 8);
    header.version = IMAGE_CACHE_VERSION;
    header.height = uint32_t(input_shape[0]);
    header.width = uint32_t(input_shape[1]);
    header.channels = uint32_t(input_shape[2]);
    header.num_samples = uint64_t(samples_number);
    header.record_bytes = record_bytes;
    header.labels_off = uint64_t(sizeof(ImageCacheHeader))
                      + uint64_t(samples_number) * record_bytes;
    header.num_classes = classes_number;
    labels_offset = header.labels_off;

    FileWriter writer;
    writer.open(tmp_path);
    writer.write(&header, sizeof(header));

    vector<uint8_t> pixels;
    pixels.resize(size_t(record_bytes));
    for (Index i = 0; i < samples_number; ++i)
    {
        for (auto& byte : pixels)
            byte = uint8_t(random_integer(0, 255));
        writer.write(pixels.data(), pixels.size());
    }
    writer.write(labels_ram.data(), labels_ram.size() * sizeof(int32_t));

    writer.finish_with_rename(cache_path);
    cache_reader.open(cache_path);
}

ImageDataset::ImageDataset(const filesystem::path& new_data_path) : Dataset()
{
    data_path = new_data_path;

    read_bmp();
}

Index ImageDataset::get_samples_number() const
{
    return Index(labels_ram.size());
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
        {"Padding", to_string(padding)},
        {"RandomAugmentation", to_string(augmentation.enabled)},
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

    set_image_padding(read_json_index(data_source_element, "Padding"));

    set_codification(read_json_string(data_source_element, "Codification"));

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

    read_bmp();
}

VectorI ImageDataset::calculate_target_distribution() const
{
    VectorI distribution = VectorI::Zero(Index(classes_number));

    for (int32_t label : labels_ram)
        if (label >= 0 && label < distribution.size())
            distribution(label)++;

    return distribution;
}

bool ImageDataset::try_load_image_cache(const Shape& new_input_shape,
                                        const chrono::high_resolution_clock::time_point& start_time)
{
    if (!filesystem::exists(cache_path)) return false;

    try
    {
        cache_reader.open(cache_path);
        ImageCacheHeader header{};
        const uint32_t expected_h = uint32_t(new_input_shape[0]);
        const uint32_t expected_w = uint32_t(new_input_shape[1]);
        const uint32_t expected_c = uint32_t(new_input_shape[2]);

        if (read_header(cache_reader, header)
            && header_matches_request(header, expected_h, expected_w, expected_c)
            && cache_reader.file_size() == header.labels_off
                                          + header.num_samples * sizeof(int32_t))
        {
            input_shape  = { Index(header.height), Index(header.width), Index(header.channels) };
            target_shape = { Index(header.num_classes == 2 ? 1 : header.num_classes) };
            record_bytes = header.record_bytes;
            labels_offset   = header.labels_off;
            classes_number  = header.num_classes;

            const Index pixels_number = Index(header.record_bytes);
            const bool single_target = (header.num_classes == 2);

            variables.resize(pixels_number + 1);
            for (Index i = 0; i < pixels_number; ++i)
            {
                variables[i].type = VariableType::Numeric;
                variables[i].name = format("variable_{}", i + 1);
                variables[i].role = VariableRole::Input;
            }
            Variable& target_variable = variables[pixels_number];
            target_variable.role = VariableRole::Target;
            target_variable.type = single_target ? VariableType::Binary : VariableType::Categorical;
            target_variable.scaler = ScalerMethod::None;
            target_variable.name = "Class";

            vector<string> placeholder_categories(size_t(header.num_classes));
            for (size_t i = 0; i < placeholder_categories.size(); ++i)
                placeholder_categories[i] = to_string(i);
            target_variable.set_categories(placeholder_categories);

            labels_ram.resize(size_t(header.num_samples));
            cache_reader.read_at(labels_ram.data(),
                                 size_t(header.num_samples) * sizeof(int32_t),
                                 labels_offset);

            sample_roles.assign(size_t(header.num_samples), SampleRole::Training);
            split_samples_random();

            if (display)
            {
                const long long ms = chrono::duration_cast<chrono::milliseconds>(
                    chrono::high_resolution_clock::now() - start_time).count();
                cout << "Image cache loaded in " << ms << " ms ("
                     << header.num_samples << " samples).\n";
            }
            return true;
        }
        cache_reader.close();
    }
    catch (const exception&)
    {
        cache_reader.close();
    }

    return false;
}


void ImageDataset::read_bmp(const Shape& new_input_shape)
{
    const chrono::high_resolution_clock::time_point start_time = chrono::high_resolution_clock::now();

    cache_path = data_path / ".cache" / "images.bin";

    if (try_load_image_cache(new_input_shape, start_time))
        return;

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

    throw_if(samples_number == 0,
             "No images in folder.");

    const Tensor3 first_image = load_image(paths[0]);

    Index height = first_image.dimension(0);
    Index width = first_image.dimension(1);
    const Index channels = first_image.dimension(2);

    throw_if(new_input_shape[2] != channels && new_input_shape[2] != 0,
             "Different number of channels in new_input_shape.");

    if (new_input_shape[0] != 0 && new_input_shape[1] != 0)
    {
        height = new_input_shape[0];
        width = new_input_shape[1];
    }

    const Index pixels_number = height * width * channels;

    const Index targets_number = (folders_number == 2)
        ? folders_number - 1
        : folders_number;

    input_shape  = { height, width, channels };
    target_shape = { targets_number };

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

    const bool single_target = (targets_number == 1);

    Variable& target_variable = variables[pixels_number];
    target_variable.name = single_target ? categories[0] + "_" + categories[1] : "Class";
    target_variable.role = VariableRole::Target;
    target_variable.type = single_target ? VariableType::Binary : VariableType::Categorical;
    target_variable.set_categories(categories);
    target_variable.scaler = ScalerMethod::None;

    sample_roles.assign(samples_number, SampleRole::Training);

    record_bytes = uint64_t(pixels_number);
    labels_offset   = uint64_t(sizeof(ImageCacheHeader)) + uint64_t(samples_number) * record_bytes;
    classes_number  = uint32_t(folders_number);

    ImageCacheHeader header{};
    memcpy(header.magic, IMAGE_CACHE_MAGIC, 8);
    header.version      = IMAGE_CACHE_VERSION;
    header.height       = uint32_t(height);
    header.width        = uint32_t(width);
    header.channels     = uint32_t(channels);
    header.num_samples  = uint64_t(samples_number);
    header.record_bytes = record_bytes;
    header.labels_off   = labels_offset;
    header.num_classes  = classes_number;

    filesystem::create_directories(cache_path.parent_path());
    const filesystem::path tmp_path = cache_path.string() + ".tmp";

    vector<int32_t> labels_out(static_cast<size_t>(samples_number));

    FileWriter writer;
    writer.open(tmp_path);
    writer.write(&header, sizeof(header));

    vector<float> tmp(static_cast<size_t>(pixels_number));
    vector<uint8_t> pixels(static_cast<size_t>(pixels_number));

    for (Index i = 0; i < samples_number; ++i)
    {
        load_image(paths[i], tmp.data(), height, width, channels, /*divide_by_255=*/false);

        Map<Array<uint8_t, Dynamic, 1>>(pixels.data(), pixels_number) =
            (Map<const Array<float, Dynamic, 1>>(tmp.data(), pixels_number)
                .max(0.0f).min(255.0f) + 0.5f).cast<uint8_t>();

        writer.write(pixels.data(), pixels.size());
        labels_out[size_t(i)] = int32_t(labels[i]);

        if (display && (i % 1000 == 0 || i + 1 == samples_number))
            display_progress_bar(i + 1, samples_number);
    }

    writer.write(labels_out.data(), labels_out.size() * sizeof(int32_t));
    writer.finish_with_rename(cache_path);

    labels_ram = std::move(labels_out);
    cache_reader.open(cache_path);

    split_samples_random();

    if (display)
    {
        const long long total_milliseconds = chrono::duration_cast<chrono::milliseconds>(
            chrono::high_resolution_clock::now() - start_time).count();

        const long long minutes = total_milliseconds / 60000;
        const long long seconds = (total_milliseconds % 60000) / 1000;
        const long long milliseconds = total_milliseconds % 1000;

        cout << "\nImage dataset cache built in: "
             << minutes << " minutes, "
             << seconds << " seconds, "
             << milliseconds << " milliseconds.\n";
    }
}

void ImageDataset::fill_inputs(const vector<Index>& sample_indices,
                               const vector<Index>& /*input_indices*/,
                               float* input_data,
                               bool is_training,
                               int /*contiguous*/) const
{
    const Index batch_size = ssize(sample_indices);
    const Index channels = input_shape[2];
    const Index pixels_per_image = Index(record_bytes);
    const bool apply_scaling = is_training;
    const bool has_scaling = ssize(input_scale) == channels
                          && ssize(input_offset) == channels;
    const bool use_default_scaling = apply_scaling && !has_scaling;
    const bool apply_augmentation = is_training && augmentation.enabled;

    string omp_error;

    #pragma omp parallel for schedule(dynamic)
    for (Index i = 0; i < batch_size; ++i)
    {
        try
        {
            thread_local vector<uint8_t> buf;
            buf.resize(size_t(pixels_per_image));

            const Index sample_index = sample_indices[size_t(i)];
            throw_if(sample_index < 0 || sample_index >= ssize(labels_ram),
                     "ImageDataset input sample index is out of range.");

            const uint64_t off = uint64_t(sizeof(ImageCacheHeader))
                                + uint64_t(sample_index) * record_bytes;
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
        throw_if(sample_index < 0 || sample_index >= ssize(labels_ram),
                 "ImageDataset target sample index is out of range.");

    if (targets_number > 1)
        for (const Index sample_index : sample_indices)
        {
            const int32_t label = labels_ram[size_t(sample_index)];
            throw_if(label < 0 || label >= targets_number,
                     "ImageDataset target label is out of range.");
        }

    if (targets_number == 1)
    {
        auto label_of = [&](Index s) { return float(labels_ram[size_t(s)]); };
        transform(execution::par, sample_indices.begin(), sample_indices.begin() + batch_size, target_data, label_of);
    }
    else
    {
        fill_n(target_data, batch_size * targets_number, 0.0f);

        #pragma omp parallel for
        for (Index i = 0; i < batch_size; ++i)
        {
            const int32_t label = labels_ram[size_t(sample_indices[i])];
            target_data[i * targets_number + label] = 1.0f;
        }
    }
}

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
