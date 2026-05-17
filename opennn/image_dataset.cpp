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

}  // namespace

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

Index ImageDataset::get_samples_number() const
{
    return Index(labels_ram.size());
}

Index ImageDataset::get_channels_number() const
{
    return input_shape[2];
}

void ImageDataset::set_data_random()
{
    // Synthetic random images: used by tests that build ImageDataset via the
    // (samples_number, input_shape, target_shape) constructor without a real
    // image folder. In that mode there's no binary cache; we keep `data` as
    // the source of truth.
    const Index height = input_shape[0];
    const Index width = input_shape[1];
    const Index channels = input_shape[2];

    const Index targets_number = target_shape[0];
    const Index inputs_number = height * width * channels;
    const Index samples_number = data.rows();

    data.setZero();

    const Index images_per_category = samples_number / targets_number;
    const Index extra = samples_number % targets_number;

    VectorI images_number = VectorI::Constant(targets_number, images_per_category);
    images_number.head(extra).array() += 1;

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

#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Warray-bounds"
#endif
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
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic pop
#endif

void ImageDataset::from_JSON(const JsonDocument& data_set_document)
{
    const Json* image_dataset_element = get_json_root(data_set_document, "ImageDataset");

    const Json* data_source_element = require_json_field(image_dataset_element, "DataSource");

    set_data_path(read_json_string(data_source_element, "Path"));

    // Legacy: "Streaming" key may exist in old JSONs — read and discard.
    if (data_source_element->has("Streaming"))
        (void)read_json_bool(data_source_element, "Streaming");

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

    read_bmp();
}

vector<Descriptives> ImageDataset::scale_features(const string&)
{
    // Streaming binary stores pixels as uint8 0..255; the model's Scaling
    // layer (ImageMinMax) handles normalization. Nothing to do here.
    return {};
}

void ImageDataset::unscale_features(const string&)
{
    // Symmetric: dataset bytes never get scaled, so nothing to undo.
}

void ImageDataset::read_bmp(const Shape& new_input_shape)
{
    const chrono::high_resolution_clock::time_point start_time = chrono::high_resolution_clock::now();

    cache_path = data_path / ".cache" / "images.bin";

    // 1) Try to use existing cache.
    if (filesystem::exists(cache_path))
    {
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
                record_bytes_ = header.record_bytes;
                labels_off_   = header.labels_off;
                num_classes_  = header.num_classes;

                const Index pixels_number = Index(header.record_bytes);
                const bool single_target = (header.num_classes == 2);

                variables.resize(pixels_number + 1);
                for (Index i = 0; i < pixels_number; ++i)
                {
                    variables[i].type = VariableType::Numeric;
                    variables[i].name = "variable_" + to_string(i + 1);
                    variables[i].role = VariableRole::Input;
                }
                Variable& target_variable = variables[pixels_number];
                target_variable.role = VariableRole::Target;
                target_variable.type = single_target ? VariableType::Binary : VariableType::Categorical;
                target_variable.scaler = ScalerMethod::None;
                target_variable.name = "Class";
                // Placeholder category names; the binary cache stores only
                // labels as int32 indices, not the original folder names.
                // The training/loss/optimizer use the count, not the names.
                vector<string> placeholder_categories(size_t(header.num_classes));
                for (size_t i = 0; i < placeholder_categories.size(); ++i)
                    placeholder_categories[i] = to_string(i);
                target_variable.set_categories(placeholder_categories);

                labels_ram.resize(size_t(header.num_samples));
                cache_reader.read_at(labels_ram.data(),
                                     size_t(header.num_samples) * sizeof(int32_t),
                                     labels_off_);

                sample_roles.assign(size_t(header.num_samples), SampleRole::Training);
                split_samples_random();

                if (display)
                {
                    const long long ms = chrono::duration_cast<chrono::milliseconds>(
                        chrono::high_resolution_clock::now() - start_time).count();
                    cout << "Image cache loaded in " << ms << " ms ("
                         << header.num_samples << " samples).\n";
                }
                return;
            }
            // Header doesn't match — close and regenerate.
            cache_reader.close();
        }
        catch (const exception&)
        {
            cache_reader.close();
        }
    }

    // 2) No valid cache — scan the BMP directory and build it.
    vector<filesystem::path> directory_path;

    for (const filesystem::directory_entry& current_directory : filesystem::directory_iterator(data_path))
        if (current_directory.is_directory())
            directory_path.emplace_back(current_directory.path());

    // Sort for reproducibility of the cache content.
    ranges::sort(directory_path);

    const Index folders_number = directory_path.size();

    vector<filesystem::path> paths;
    vector<Index> labels;

    for (Index i = 0; i < folders_number; ++i)
    {
        vector<filesystem::path> folder_files;
        for (const filesystem::directory_entry& current_directory : filesystem::directory_iterator(directory_path[i]))
            if (current_directory.is_regular_file() && current_directory.path().extension() == ".bmp")
                folder_files.emplace_back(current_directory.path());

        ranges::sort(folder_files);
        for (auto& p : folder_files)
        {
            paths.emplace_back(std::move(p));
            labels.push_back(i);
        }
    }

    const Index samples_number = paths.size();

    if (samples_number == 0)
        throw runtime_error("No images in folder.");

    const Tensor3 first_image = load_image(paths[0]);

    Index height = first_image.dimension(0);
    Index width = first_image.dimension(1);
    const Index channels = first_image.dimension(2);

    if (new_input_shape[2] != channels && new_input_shape[2] != 0)
        throw runtime_error("Different number of channels in new_input_shape.");

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
        variables[i].name = "variable_" + to_string(i + 1);
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

    // Write header (with placeholder, will rewrite after writing pixels) then
    // pixels then labels into a tmp file, rename atomically at the end.
    record_bytes_ = uint64_t(pixels_number);
    labels_off_   = uint64_t(sizeof(ImageCacheHeader)) + uint64_t(samples_number) * record_bytes_;
    num_classes_  = uint32_t(folders_number);

    ImageCacheHeader header{};
    memcpy(header.magic, IMAGE_CACHE_MAGIC, 8);
    header.version      = IMAGE_CACHE_VERSION;
    header.height       = uint32_t(height);
    header.width        = uint32_t(width);
    header.channels     = uint32_t(channels);
    header.num_samples  = uint64_t(samples_number);
    header.record_bytes = record_bytes_;
    header.labels_off   = labels_off_;
    header.num_classes  = num_classes_;

    filesystem::create_directories(cache_path.parent_path());
    const filesystem::path tmp_path = cache_path.string() + ".tmp";

    // We can't trivially do this with FileWriter + OMP (multi-writer single-stream
    // is not thread-safe), so we materialize the binary in memory in two pieces:
    // (a) a contiguous pixel buffer filled in parallel by OMP, and (b) labels.
    // For typical melanoma datasets (~10k × 32 KB) this is ~320 MB peak — fits
    // in RAM; the cache file is the same size. For larger datasets the limit
    // grows linearly, which is the trade-off for parallel decode.
    vector<uint8_t> pixels(size_t(samples_number) * size_t(pixels_number));
    vector<int32_t> labels_out;
    labels_out.resize(size_t(samples_number));

    Index progress_counter = 0;
    string omp_error;

    #pragma omp parallel for schedule(dynamic)
    for (Index i = 0; i < samples_number; ++i)
    {
        try
        {
            vector<float> tmp;
            tmp.resize(size_t(pixels_number));
            load_image(paths[i], tmp.data(), height, width, channels, /*divide_by_255=*/false);
            uint8_t* dst = pixels.data() + size_t(i) * size_t(pixels_number);
            for (Index p = 0; p < pixels_number; ++p)
            {
                const float v = tmp[size_t(p)];
                const int iv = int(v < 0.0f ? 0.0f : (v > 255.0f ? 255.0f : v) + 0.5f);
                dst[p] = uint8_t(iv);
            }
            labels_out[size_t(i)] = int32_t(labels[i]);
        }
        catch (const exception& e)
        {
            #pragma omp critical
            { omp_error = e.what(); }
            continue;
        }

        #pragma omp atomic
        ++progress_counter;

        if (omp_get_thread_num() == 0)
            display_progress_bar(progress_counter, samples_number);
    }

    if (!omp_error.empty()) throw runtime_error(omp_error);

    FileWriter writer;
    writer.open(tmp_path);
    writer.write(&header, sizeof(header));
    writer.write(pixels.data(), pixels.size());
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
                               bool parallelize,
                               int /*contiguous*/) const
{
    const Index batch_size = ssize(sample_indices);
    const Index pixels_per_image = Index(record_bytes_);
    const float scale = is_training ? (1.0f / 255.0f) : 1.0f;

    string omp_error;

    #pragma omp parallel for schedule(dynamic) if (parallelize)
    for (Index i = 0; i < batch_size; ++i)
    {
        try
        {
            // Thread-local staging buffer to avoid heap churn per call.
            thread_local vector<uint8_t> buf;
            buf.resize(size_t(pixels_per_image));

            const uint64_t off = uint64_t(sizeof(ImageCacheHeader))
                                + uint64_t(sample_indices[i]) * record_bytes_;
            cache_reader.read_at(buf.data(), size_t(pixels_per_image), off);

            float* dst = input_data + i * pixels_per_image;
            for (Index p = 0; p < pixels_per_image; ++p)
                dst[p] = float(buf[size_t(p)]) * scale;
        }
        catch (const exception& e)
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
                                bool /*is_training*/,
                                bool parallelize,
                                int /*contiguous*/) const
{
    const Index batch_size = ssize(sample_indices);
    const Index targets_number = ssize(target_indices);

    if (targets_number == 1)
    {
        auto label_of = [&](Index s) { return float(labels_ram[size_t(s)]); };
        if (parallelize)
            transform(execution::par, sample_indices.begin(), sample_indices.begin() + batch_size, target_data, label_of);
        else
            transform(sample_indices.begin(), sample_indices.begin() + batch_size, target_data, label_of);
    }
    else
    {
        fill_n(target_data, batch_size * targets_number, 0.0f);

        #pragma omp parallel for if (parallelize)
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
