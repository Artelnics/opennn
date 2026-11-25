// SPDX-License-Identifier: LGPL-2.1-or-later
// Copyright (C) 2005-2026 Artificial Intelligence Techniques, SL.

#include "opennn/dataset/segmentation_dataset.h"

#include "opennn/core/io_utilities.h"
#include "opennn/core/log.h"
#include "opennn/core/string_utilities.h"
#include "opennn/core/tensor_types.h"
#include "opennn/dataset/image_processing.h"

namespace opennn
{

namespace
{

// Nearest-neighbour resample of a single-channel mask, then threshold at the
// midpoint. load_image() would bilinear-resample it, which turns every class
// boundary into a grey ramp and shifts the mask by up to half a pixel on each
// side -- an error the segmentation metric charges for as wrong pixels.
void load_mask(const filesystem::path& path,
               float* dst,
               Index height,
               Index width)
{
    const Tensor3 mask = load_image(path);

    throw_if(mask.dimension(2) != 1,
             "SegmentationDataset: mask is not single-channel: {}", path.string());

    const Index mask_height = mask.dimension(0);
    const Index mask_width  = mask.dimension(1);

    const float* const src = mask.data();

    #pragma omp parallel for
    for (Index y = 0; y < height; ++y)
    {
        const Index source_y = min(mask_height - 1,
                                   static_cast<Index>(Index(y) * mask_height / height));

        for (Index x = 0; x < width; ++x)
        {
            const Index source_x = min(mask_width - 1,
                                       static_cast<Index>(Index(x) * mask_width / width));

            dst[y * width + x] = src[source_y * mask_width + source_x] >= 127.5f ? 1.0f : 0.0f;
        }
    }
}


// A mask file is paired with its image by stem, so the two directories may use
// different extensions: Oxford Pets ships .jpg images beside .png trimaps.
filesystem::path find_mask(const filesystem::path& image_path,
                           const vector<filesystem::path>& mask_files)
{
    const string stem = image_path.stem().string();

    for (const filesystem::path& mask : mask_files)
        if (mask.stem().string() == stem)
            return mask;

    return filesystem::path();
}

}

SegmentationDataset::SegmentationDataset(const filesystem::path& new_images_directory,
                                         const filesystem::path& new_masks_directory,
                                         const Shape& new_input_shape)
{
    data_path = new_images_directory;
    masks_directory = new_masks_directory;
    requested_input_shape = new_input_shape;

    read_images();
}

void SegmentationDataset::read_images()
{
    invalidate_data();

    data.resize(0, 0);
    image_paths.clear();
    mask_paths.clear();

    throw_if(data_path.empty(),
             "SegmentationDataset: no images directory was given.");

    const vector<filesystem::path> mask_files =
        list_files(masks_directory, is_supported_image_file);

    image_paths = list_files(data_path, is_supported_image_file);
    ranges::sort(image_paths);

    throw_if(image_paths.empty(),
             "SegmentationDataset: no images in folder {}.",
             data_path.string());

    for (const filesystem::path& image_path : image_paths)
    {
        const filesystem::path mask_path = find_mask(image_path, mask_files);

        throw_if(mask_path.empty(),
                 "SegmentationDataset: no mask found for image {}.",
                 image_path.string());

        mask_paths.push_back(mask_path);
    }

    Index height   = 0;
    Index width    = 0;
    Index channels = 0;

    if (requested_input_shape.get_rank() == 3)
    {
        height   = requested_input_shape[0];
        width    = requested_input_shape[1];
        channels = requested_input_shape[2];
    }
    else
    {
        const Tensor3 first_image = load_image(image_paths.front());

        height   = first_image.dimension(0);
        width    = first_image.dimension(1);
        channels = first_image.dimension(2);
    }

    throw_if(height <= 0 || width <= 0 || channels <= 0,
             "SegmentationDataset: image dimensions must be positive.");

    const Index samples_number = ssize(image_paths);
    const Index image_pixels = height * width * channels;
    const Index mask_pixels  = height * width;

    input_shape  = { height, width, channels };
    target_shape = { height, width, 1 };

    variables.assign(2, Variable());

    Variable& image_variable = variables[0];
    image_variable.name = "image";
    image_variable.type = VariableType::Numeric;
    image_variable.role = VariableRole::Input;
    image_variable.features = image_pixels;

    Variable& mask_variable = variables[1];
    mask_variable.name = "mask";
    mask_variable.type = VariableType::Binary;
    mask_variable.role = VariableRole::Target;
    mask_variable.features = mask_pixels;
    mask_variable.scaler = ScalerMethod::None;

    data.resize(samples_number, image_pixels + mask_pixels);
    sample_roles.assign(size_t(samples_number), SampleRole::Training);

    string load_error;

    #pragma omp parallel
    {
        vector<float> mask_pixels_scratch(static_cast<size_t>(mask_pixels));

        #pragma omp for schedule(dynamic)
        for (Index i = 0; i < samples_number; ++i)
        {
            try
            {
                float* const sample = &data(i, 0);

                load_image(image_paths[size_t(i)], sample, height, width, channels);

                // Images reach the model normalized to [0, 1]; masks stay binary.
                Map<Array<float, Dynamic, 1>>(sample, image_pixels) /= 255.0f;

                load_mask(mask_paths[size_t(i)], mask_pixels_scratch.data(), height, width);

                copy_n(mask_pixels_scratch.data(), mask_pixels, sample + image_pixels);
            }
            catch (const exception& e)
            {
                #pragma omp critical
                { if (load_error.empty()) load_error = e.what(); }
            }
        }
    }

    throw_if(!load_error.empty(),
             load_error);

    split_samples_random();

    if (display)
        logging::info() << "\nSegmentation dataset: " << samples_number
             << " samples of " << height << "x" << width << "x" << channels << " loaded.\n";
}

void SegmentationDataset::fill_inputs(const vector<Index>& sample_indices,
                                      const vector<Index>& input_indices,
                                      float* input_data,
                                      FillMode,
                                      ColumnContiguity column_contiguity) const
{
    fill_tensor_data(data, sample_indices, input_indices,
                     span<float>(input_data, size_t(input_indices.size() * ssize(sample_indices))),
                     column_contiguity);
}

void SegmentationDataset::fill_targets(const vector<Index>& sample_indices,
                                       const vector<Index>& target_indices,
                                       float* target_data,
                                       FillMode,
                                       ColumnContiguity column_contiguity) const
{
    fill_tensor_data(data, sample_indices, target_indices,
                     span<float>(target_data, size_t(target_indices.size() * ssize(sample_indices))),
                     column_contiguity);
}

namespace
{

// Accumulate per-channel min/max/mean/std over `samples` rows of `mat`.
// Layout: each row has H*W*C image values (HWC order) followed by mask values
// (which are ignored here). Returns C descriptives.
FeatureScaling channel_stats(const MatrixR& mat,
                             const vector<Index>& samples,
                             Index H, Index W, Index C,
                             const FeatureScaling& reference)
{
    const Index pixels = H * W;
    const Index n      = ssize(samples) * pixels;

    vector<double> sum(size_t(C), 0.0), sum_sq(size_t(C), 0.0);
    vector<float>  mn(size_t(C),  std::numeric_limits<float>::max());
    vector<float>  mx(size_t(C), -std::numeric_limits<float>::max());

    for (Index si : samples)
    {
        const float* row = mat.row(si).data();
        for (Index p = 0; p < pixels; ++p)
            for (Index c = 0; c < C; ++c)
            {
                const float v = row[p * C + c];
                sum[size_t(c)]    += double(v);
                sum_sq[size_t(c)] += double(v) * v;
                if (v < mn[size_t(c)]) mn[size_t(c)] = v;
                if (v > mx[size_t(c)]) mx[size_t(c)] = v;
            }
    }

    const vector<ScalerMethod>& scalers_src =
        (reference.scalers.size() == size_t(C))
        ? reference.scalers
        : vector<ScalerMethod>(size_t(C), ScalerMethod::None);

    FeatureScaling fs;
    fs.descriptives.resize(size_t(C));
    fs.scalers      = scalers_src;
    fs.min_range    = reference.min_range;
    fs.max_range    = reference.max_range;

    for (Index c = 0; c < C; ++c)
    {
        const double mean = (n > 0) ? sum[size_t(c)] / n : 0.0;
        const double var  = (n > 0) ? sum_sq[size_t(c)] / n - mean * mean : 0.0;
        fs.descriptives[size_t(c)] = Descriptives(
            mn[size_t(c)],
            mx[size_t(c)],
            float(mean),
            float(sqrt(max(0.0, var))));
    }

    return fs;
}

} // anonymous namespace

FeatureScaling SegmentationDataset::prepare_training_scaling(
    VariableRole role, const FeatureScaling& requested, Index expected_features)
{
    if (role == VariableRole::Target)
        return requested;

    throw_if(role != VariableRole::Input,
             "SegmentationDataset::prepare_training_scaling: unsupported role.");
    throw_if(expected_features != input_shape[2],
             "SegmentationDataset::prepare_training_scaling: "
             "expected {} feature descriptives (one per channel), got {}.",
             input_shape[2], expected_features);

    vector<Index> samples = get_sample_indices(SampleRole::Training);
    if (samples.empty())
        samples = get_used_sample_indices();

    return channel_stats(data, samples,
                         input_shape[0], input_shape[1], input_shape[2],
                         requested);
}

FeatureScaling SegmentationDataset::calculate_used_feature_scaling(VariableRole role) const
{
    if (role == VariableRole::Target)
    {
        FeatureScaling fs;
        fs.descriptives.assign(size_t(target_shape[2]), Descriptives(0.0f, 1.0f, 0.5f, 0.5f));
        fs.scalers.assign(size_t(target_shape[2]), ScalerMethod::None);
        return fs;
    }

    throw_if(role != VariableRole::Input,
             "SegmentationDataset::calculate_used_feature_scaling: unsupported role.");

    const vector<Index> samples = get_used_sample_indices();
    FeatureScaling reference;
    reference.scalers.assign(size_t(input_shape[2]),
                             variables.empty() ? ScalerMethod::None
                                               : variables[0].get_scaler_type());
    return channel_stats(data, samples,
                         input_shape[0], input_shape[1], input_shape[2],
                         reference);
}

void SegmentationDataset::to_JSON(JsonWriter& printer) const
{
    write_json_header(printer, {
        {"FileType", "segmentation"},
        {"Path", data_path.string()},
        {"MasksPath", masks_directory.string()},
        {"Channels", to_string(input_shape[2])},
        {"Width", to_string(input_shape[1])},
        {"Height", to_string(input_shape[0])},
        {"Codification", get_codification_string()}
    });

    write_json_footer(printer);
}

void SegmentationDataset::from_JSON(const JsonDocument& data_set_document)
{
    const Json* const dataset_element = get_json_root(data_set_document, "Dataset");
    const Json* const data_source_element = require_json_field(dataset_element, "DataSource");

    set_data_path(read_json_string(data_source_element, "Path"));
    masks_directory = read_json_string(data_source_element, "MasksPath");
    set_codification(read_json_string(data_source_element, "Codification"));

    requested_input_shape = { read_json_index(data_source_element, "Height"),
                              read_json_index(data_source_element, "Width"),
                              read_json_index(data_source_element, "Channels") };

    read_images();

    read_json_blocks(dataset_element);
}

}