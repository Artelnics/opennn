//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   Y O L O   D A T A S E T   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "yolo_dataset.h"
#include "image_utilities.h"
#include "io_utilities.h"
#include "json.h"

namespace opennn
{

namespace
{

#pragma pack(push, 1)
struct YoloImageCacheHeader
{
    char magic[8];
    uint32_t version;
    uint32_t height;
    uint32_t width;
    uint32_t channels;
    uint64_t samples;
    uint64_t record_bytes;
    uint8_t pad[24];
};

struct YoloTargetCacheHeader
{
    char magic[8];
    uint32_t version;
    uint32_t grid_size;
    uint32_t boxes_per_cell;
    uint32_t classes_number;
    uint64_t samples;
    uint64_t target_floats;
    uint64_t anchors_hash;
    uint64_t anchors_offset;
    uint64_t targets_offset;
};
#pragma pack(pop)

static_assert(sizeof(YoloImageCacheHeader) == 64);
static_assert(sizeof(YoloTargetCacheHeader) == 64);

constexpr uint32_t YOLO_CACHE_VERSION = 1;
constexpr char YOLO_IMAGE_MAGIC[8] = {'O','P','E','N','N','Y','I','M'};
constexpr char YOLO_TARGET_MAGIC[8] = {'O','P','E','N','N','Y','T','G'};

bool has_bmp_extension(const filesystem::path& path)
{
    return path.extension() == ".bmp";
}

vector<filesystem::path> list_files(const filesystem::path& directory,
                                    bool (*predicate)(const filesystem::path&))
{
    vector<filesystem::path> paths;
    for (const auto& entry : filesystem::directory_iterator(directory))
        if (entry.is_regular_file() && predicate(entry.path()))
            paths.push_back(entry.path());

    ranges::sort(paths);
    return paths;
}

vector<string> read_yolo_classes(const filesystem::path& labels_directory)
{
    for (const auto& entry : filesystem::directory_iterator(labels_directory))
    {
        if (!entry.is_regular_file() || entry.path().extension() != ".names")
            continue;

        ifstream file(entry.path());
        if (!file)
            throw runtime_error(format("Cannot open YOLO classes file: {}", entry.path().string()));

        vector<string> classes;
        string line;
        while (getline(file, line))
            if (!line.empty())
                classes.push_back(line);

        return classes;
    }

    return {};
}

vector<YoloDataset::Box> read_yolo_boxes(const filesystem::path& label_path)
{
    ifstream file(label_path);
    if (!file)
        return {};

    vector<YoloDataset::Box> boxes;
    string line;

    while (getline(file, line))
    {
        if (line.empty()) continue;

        istringstream stream(line);
        YoloDataset::Box box;
        if (!(stream >> box.class_id >> box.x >> box.y >> box.w >> box.h))
            throw runtime_error(format("Invalid YOLO label line in {}: {}", label_path.string(), line));

        if (box.x < 0.0f || box.x > 1.0f || box.y < 0.0f || box.y > 1.0f
        ||  box.w < 0.0f || box.w > 1.0f || box.h < 0.0f || box.h > 1.0f)
            throw runtime_error(format("YOLO label values out of range in {}: {}", label_path.string(), line));

        boxes.push_back(box);
    }

    return boxes;
}

uint64_t hash_anchors(const vector<array<float, 2>>& anchors)
{
    uint64_t h = 1469598103934665603ull;

    auto mix = [&](uint32_t value)
    {
        h ^= value;
        h *= 1099511628211ull;
    };

    mix(uint32_t(anchors.size()));

    for (const auto& anchor : anchors)
    {
        uint32_t bits = 0;
        memcpy(&bits, &anchor[0], sizeof(uint32_t));
        mix(bits);
        memcpy(&bits, &anchor[1], sizeof(uint32_t));
        mix(bits);
    }

    return h;
}

float yolo_iou_wh(const array<float, 2>& box, const array<float, 2>& anchor)
{
    const float inter = min(box[0], anchor[0]) * min(box[1], anchor[1]);
    const float area = box[0] * box[1] + anchor[0] * anchor[1] - inter;
    return area > 0.0f ? inter / area : 0.0f;
}

vector<array<float, 2>> calculate_yolo_anchors(const vector<vector<YoloDataset::Box>>& labels,
                                               Index boxes_per_cell)
{
    vector<array<float, 2>> boxes;

    for (const auto& sample : labels)
        for (const auto& box : sample)
            if (box.w > 0.0f && box.h > 0.0f)
                boxes.push_back({box.w, box.h});

    if (boxes.empty())
    {
        vector<array<float, 2>> fallback(size_t(boxes_per_cell), {0.1f, 0.1f});
        return fallback;
    }

    vector<array<float, 2>> anchors(static_cast<size_t>(boxes_per_cell));
    for (Index i = 0; i < boxes_per_cell; ++i)
        anchors[size_t(i)] = boxes[size_t(i % ssize(boxes))];

    vector<Index> assignments(boxes.size());

    for (Index iteration = 0; iteration < 100; ++iteration)
    {
        bool changed = false;

        for (size_t i = 0; i < boxes.size(); ++i)
        {
            float best_iou = -1.0f;
            Index best_anchor = 0;

            for (Index a = 0; a < boxes_per_cell; ++a)
            {
                const float iou = yolo_iou_wh(boxes[i], anchors[size_t(a)]);
                if (iou > best_iou)
                {
                    best_iou = iou;
                    best_anchor = a;
                }
            }

            changed = changed || assignments[i] != best_anchor;
            assignments[i] = best_anchor;
        }

        vector<array<float, 2>> sums(size_t(boxes_per_cell), {0.0f, 0.0f});
        vector<Index> counts(size_t(boxes_per_cell), 0);

        for (size_t i = 0; i < boxes.size(); ++i)
        {
            const Index a = assignments[i];
            sums[size_t(a)][0] += boxes[i][0];
            sums[size_t(a)][1] += boxes[i][1];
            counts[size_t(a)]++;
        }

        for (Index a = 0; a < boxes_per_cell; ++a)
            if (counts[size_t(a)] > 0)
                anchors[size_t(a)] = {
                    sums[size_t(a)][0] / float(counts[size_t(a)]),
                    sums[size_t(a)][1] / float(counts[size_t(a)])
                };

        if (!changed) break;
    }

    return anchors;
}

Tensor3 letterbox_image(const Tensor3& image,
                        Index target_height,
                        Index target_width,
                        float& scale,
                        Index& offset_x,
                        Index& offset_y)
{
    const Index original_height = image.dimension(0);
    const Index original_width = image.dimension(1);
    const Index channels = image.dimension(2);

    scale = min(float(target_width) / float(original_width),
                float(target_height) / float(original_height));

    const Index scaled_width = max<Index>(1, Index(round(float(original_width) * scale)));
    const Index scaled_height = max<Index>(1, Index(round(float(original_height) * scale)));

    offset_x = (target_width - scaled_width) / 2;
    offset_y = (target_height - scaled_height) / 2;

    Tensor3 output(target_height, target_width, channels);
    output.setZero();

    const Tensor3 resized = resize_image(image, scaled_height, scaled_width);

    for (Index y = 0; y < scaled_height; ++y)
        for (Index x = 0; x < scaled_width; ++x)
            for (Index c = 0; c < channels; ++c)
                output(y + offset_y, x + offset_x, c) = resized(y, x, c);

    return output;
}

void adjust_boxes_to_letterbox(vector<YoloDataset::Box>& boxes,
                               Index original_height,
                               Index original_width,
                               Index target_height,
                               Index target_width,
                               float scale,
                               Index offset_x,
                               Index offset_y)
{
    for (auto& box : boxes)
    {
        const float x_abs = box.x * float(original_width);
        const float y_abs = box.y * float(original_height);
        const float w_abs = box.w * float(original_width);
        const float h_abs = box.h * float(original_height);

        box.x = (x_abs * scale + float(offset_x)) / float(target_width);
        box.y = (y_abs * scale + float(offset_y)) / float(target_height);
        box.w = w_abs * scale / float(target_width);
        box.h = h_abs * scale / float(target_height);
    }
}

Index best_anchor_for_box(const YoloDataset::Box& box,
                          const vector<array<float, 2>>& anchors)
{
    float best_iou = -1.0f;
    Index best = 0;

    for (Index a = 0; a < ssize(anchors); ++a)
    {
        const float iou = yolo_iou_wh({box.w, box.h}, anchors[size_t(a)]);
        if (iou > best_iou)
        {
            best_iou = iou;
            best = a;
        }
    }

    return best;
}

void make_target(const vector<YoloDataset::Box>& boxes,
                 const vector<array<float, 2>>& anchors,
                 Index grid_size,
                 Index boxes_per_cell,
                 Index classes_number,
                 float* target)
{
    const Index values_per_box = 5 + classes_number;
    const Index channels = boxes_per_cell * values_per_box;
    fill_n(target, grid_size * grid_size * channels, 0.0f);

    for (const auto& box : boxes)
    {
        if (box.class_id < 0 || box.class_id >= classes_number)
            continue;

        const Index col = min<Index>(grid_size - 1, max<Index>(0, Index(floor(box.x * grid_size))));
        const Index row = min<Index>(grid_size - 1, max<Index>(0, Index(floor(box.y * grid_size))));
        const Index anchor = best_anchor_for_box(box, anchors);
        const Index base = (row * grid_size + col) * channels + anchor * values_per_box;

        target[base + 0] = box.x * grid_size - float(col);
        target[base + 1] = box.y * grid_size - float(row);
        target[base + 2] = box.w;
        target[base + 3] = box.h;
        target[base + 4] = 1.0f;
        target[base + 5 + box.class_id] = 1.0f;
    }
}

} // namespace

vector<YoloDetection> decode_yolo_detections(const float* nms_output,
                                             Index max_boxes,
                                             Index original_height,
                                             Index original_width,
                                             Index network_height,
                                             Index network_width)
{
    if (original_height <= 0 || original_width <= 0
    ||  network_height  <= 0 || network_width  <= 0)
        throw runtime_error("decode_yolo_detections: dimensions must be positive.");

    const float scale = min(float(network_width)  / float(original_width),
                            float(network_height) / float(original_height));
    const float scaled_width  = float(original_width)  * scale;
    const float scaled_height = float(original_height) * scale;
    const float offset_x = (float(network_width)  - scaled_width)  * 0.5f;
    const float offset_y = (float(network_height) - scaled_height) * 0.5f;

    vector<YoloDetection> detections;

    for (Index k = 0; k < max_boxes; ++k)
    {
        const float* row = nms_output + k * 6;
        const float score = row[4];

        if (score <= 0.0f) break;  // NMS writes kept boxes first; first zero terminates

        const float cx_net_px = row[0] * float(network_width);
        const float cy_net_px = row[1] * float(network_height);
        const float w_net_px  = row[2] * float(network_width);
        const float h_net_px  = row[3] * float(network_height);

        YoloDetection detection;
        detection.center_x = (cx_net_px - offset_x) / scale;
        detection.center_y = (cy_net_px - offset_y) / scale;
        detection.width    = w_net_px / scale;
        detection.height   = h_net_px / scale;
        detection.score    = score;
        detection.class_id = Index(row[5]);

        detection.center_x = clamp(detection.center_x, 0.0f, float(original_width));
        detection.center_y = clamp(detection.center_y, 0.0f, float(original_height));

        detections.push_back(detection);
    }

    return detections;
}

YoloDataset::YoloDataset(const filesystem::path& new_images_dir,
                         const filesystem::path& new_labels_dir,
                         const Shape& new_input_shape,
                         Index new_grid_size,
                         Index new_boxes_per_cell,
                         const vector<array<float, 2>>& new_anchors)
{
    set(new_images_dir, new_labels_dir, new_input_shape, new_grid_size,
        new_boxes_per_cell, new_anchors);
}

void YoloDataset::set(const filesystem::path& new_images_dir,
                      const filesystem::path& new_labels_dir,
                      const Shape& new_input_shape,
                      Index new_grid_size,
                      Index new_boxes_per_cell,
                      const vector<array<float, 2>>& new_anchors)
{
    if (new_input_shape.rank != 3)
        throw runtime_error("YoloDataset: input_shape must be rank 3.");
    if (new_grid_size <= 0 || new_boxes_per_cell <= 0)
        throw runtime_error("YoloDataset: grid_size and boxes_per_cell must be positive.");

    images_directory = new_images_dir;
    labels_directory = new_labels_dir;
    data_path = images_directory;
    input_shape = new_input_shape;
    grid_size = new_grid_size;
    boxes_per_cell = new_boxes_per_cell;
    image_record_bytes = input_shape.size();
    class_names = read_yolo_classes(labels_directory);

    image_cache_path = images_directory / ".cache" / "yolo_images.bin";
    target_cache_path = images_directory / ".cache" / "yolo_targets.bin";

    open_or_build_cache(new_anchors);
}

void YoloDataset::open_or_build_cache(const vector<array<float, 2>>& requested_anchors)
{
    if (try_open_cache(requested_anchors))
        return;

    image_cache_reader.close();
    target_cache_reader.close();
    build_cache(requested_anchors);
}

bool YoloDataset::try_open_cache(const vector<array<float, 2>>& requested_anchors)
{
    if (!filesystem::exists(image_cache_path) || !filesystem::exists(target_cache_path))
        return false;

    try
    {
        image_cache_reader.open(image_cache_path);
        target_cache_reader.open(target_cache_path);

        YoloImageCacheHeader image_header{};
        YoloTargetCacheHeader target_header{};
        image_cache_reader.read_at(&image_header, sizeof(image_header), 0);
        target_cache_reader.read_at(&target_header, sizeof(target_header), 0);

        if (memcmp(image_header.magic, YOLO_IMAGE_MAGIC, 8) != 0
        ||  memcmp(target_header.magic, YOLO_TARGET_MAGIC, 8) != 0
        ||  image_header.version != YOLO_CACHE_VERSION
        ||  target_header.version != YOLO_CACHE_VERSION)
            return false;

        if (Index(image_header.height) != input_shape[0]
        ||  Index(image_header.width) != input_shape[1]
        ||  Index(image_header.channels) != input_shape[2]
        ||  Index(target_header.grid_size) != grid_size
        ||  Index(target_header.boxes_per_cell) != boxes_per_cell
        ||  image_header.samples != target_header.samples)
            return false;

        vector<array<float, 2>> cached_anchors(static_cast<size_t>(boxes_per_cell));
        target_cache_reader.read_at(cached_anchors.data(),
                                    cached_anchors.size() * sizeof(array<float, 2>),
                                    target_header.anchors_offset);

        const uint64_t cached_hash = hash_anchors(cached_anchors);
        if (cached_hash != target_header.anchors_hash)
            return false;

        if (!requested_anchors.empty() && hash_anchors(requested_anchors) != cached_hash)
            return false;

        const uint64_t expected_image_size = sizeof(YoloImageCacheHeader)
            + image_header.samples * image_header.record_bytes;
        const uint64_t expected_target_size = target_header.targets_offset
            + target_header.samples * target_header.target_floats * sizeof(float);

        if (image_cache_reader.file_size() != expected_image_size
        ||  target_cache_reader.file_size() != expected_target_size)
            return false;

        anchors = std::move(cached_anchors);
        classes_number = Index(target_header.classes_number);
        if (class_names.empty())
        {
            class_names.resize(size_t(classes_number));
            for (Index i = 0; i < classes_number; ++i)
                class_names[size_t(i)] = to_string(i);
        }

        target_record_floats = Index(target_header.target_floats);
        target_data_offset = target_header.targets_offset;
        setup_metadata(Index(image_header.samples));
        return true;
    }
    catch (const exception&)
    {
        image_cache_reader.close();
        target_cache_reader.close();
        return false;
    }
}

void YoloDataset::build_cache(const vector<array<float, 2>>& requested_anchors)
{
    vector<filesystem::path> image_paths = list_files(images_directory, has_bmp_extension);
    if (image_paths.empty())
        throw runtime_error(format("YoloDataset: no BMP files found in {}", images_directory.string()));

    vector<vector<Box>> labels(image_paths.size());
    Index max_class_id = -1;

    filesystem::create_directories(image_cache_path.parent_path());

    const filesystem::path image_tmp_path = image_cache_path.string() + ".tmp";
    FileWriter image_writer;
    image_writer.open(image_tmp_path);

    YoloImageCacheHeader image_header{};
    memcpy(image_header.magic, YOLO_IMAGE_MAGIC, 8);
    image_header.version = YOLO_CACHE_VERSION;
    image_header.height = uint32_t(input_shape[0]);
    image_header.width = uint32_t(input_shape[1]);
    image_header.channels = uint32_t(input_shape[2]);
    image_header.samples = uint64_t(image_paths.size());
    image_header.record_bytes = uint64_t(image_record_bytes);
    image_writer.write(&image_header, sizeof(image_header));

    vector<uint8_t> pixels(static_cast<size_t>(image_record_bytes));

    for (size_t i = 0; i < image_paths.size(); ++i)
    {
        Tensor3 image = load_image(image_paths[i]);
        if (image.dimension(2) != input_shape[2])
            throw runtime_error(format("YoloDataset: channel mismatch in {}", image_paths[i].string()));

        float scale = 1.0f;
        Index offset_x = 0;
        Index offset_y = 0;
        const Tensor3 prepared = letterbox_image(image, input_shape[0], input_shape[1],
                                                 scale, offset_x, offset_y);

        filesystem::path label_path = labels_directory / image_paths[i].filename();
        label_path.replace_extension(".txt");
        labels[i] = read_yolo_boxes(label_path);
        adjust_boxes_to_letterbox(labels[i], image.dimension(0), image.dimension(1),
                                  input_shape[0], input_shape[1], scale, offset_x, offset_y);

        for (const Box& box : labels[i])
            max_class_id = max(max_class_id, box.class_id);

        for (Index p = 0; p < image_record_bytes; ++p)
        {
            const float v = prepared.data()[p];
            const int iv = int(v < 0.0f ? 0.0f : (v > 255.0f ? 255.0f : v) + 0.5f);
            pixels[size_t(p)] = uint8_t(iv);
        }

        image_writer.write(pixels.data(), pixels.size());

        if (display && (i % 1000 == 0 || i + 1 == image_paths.size()))
            display_progress_bar(Index(i + 1), Index(image_paths.size()));
    }

    image_writer.finish_with_rename(image_cache_path);

    classes_number = class_names.empty() ? max_class_id + 1 : ssize(class_names);
    if (classes_number <= 0)
        throw runtime_error("YoloDataset: cannot infer classes_number.");

    if (class_names.empty())
    {
        class_names.resize(size_t(classes_number));
        for (Index i = 0; i < classes_number; ++i)
            class_names[size_t(i)] = to_string(i);
    }

    anchors = requested_anchors.empty()
        ? calculate_yolo_anchors(labels, boxes_per_cell)
        : requested_anchors;

    if (ssize(anchors) != boxes_per_cell)
        throw runtime_error("YoloDataset: anchors size must equal boxes_per_cell.");

    target_record_floats = grid_size * grid_size * boxes_per_cell * (5 + classes_number);

    const filesystem::path target_tmp_path = target_cache_path.string() + ".tmp";
    FileWriter target_writer;
    target_writer.open(target_tmp_path);

    YoloTargetCacheHeader target_header{};
    memcpy(target_header.magic, YOLO_TARGET_MAGIC, 8);
    target_header.version = YOLO_CACHE_VERSION;
    target_header.grid_size = uint32_t(grid_size);
    target_header.boxes_per_cell = uint32_t(boxes_per_cell);
    target_header.classes_number = uint32_t(classes_number);
    target_header.samples = uint64_t(image_paths.size());
    target_header.target_floats = uint64_t(target_record_floats);
    target_header.anchors_hash = hash_anchors(anchors);
    target_header.anchors_offset = sizeof(YoloTargetCacheHeader);
    target_header.targets_offset = target_header.anchors_offset
        + uint64_t(anchors.size() * sizeof(array<float, 2>));

    target_writer.write(&target_header, sizeof(target_header));
    target_writer.write(anchors.data(), anchors.size() * sizeof(array<float, 2>));

    vector<float> target(static_cast<size_t>(target_record_floats));
    for (const auto& sample_boxes : labels)
    {
        make_target(sample_boxes, anchors, grid_size, boxes_per_cell, classes_number, target.data());
        target_writer.write(target.data(), target.size() * sizeof(float));
    }

    target_writer.finish_with_rename(target_cache_path);

    image_cache_reader.open(image_cache_path);
    target_cache_reader.open(target_cache_path);
    target_data_offset = target_header.targets_offset;

    setup_metadata(Index(image_paths.size()));

    if (display)
        cout << "\nYOLO cache built (" << samples_number << " samples).\n";
}

void YoloDataset::setup_metadata(Index new_samples_number)
{
    samples_number = new_samples_number;
    target_shape = {grid_size, grid_size, boxes_per_cell * (5 + classes_number)};

    variables.clear();
    variables.resize(size_t(input_shape.size() + target_shape.size()));

    for (Index i = 0; i < input_shape.size(); ++i)
    {
        Variable& variable = variables[size_t(i)];
        variable.name = format("pixel_{}", i + 1);
        variable.role = VariableRole::Input;
        variable.type = VariableType::Numeric;
        variable.scaler = ScalerMethod::None;
    }

    for (Index i = 0; i < target_shape.size(); ++i)
    {
        Variable& variable = variables[size_t(input_shape.size() + i)];
        variable.name = format("yolo_target_{}", i + 1);
        variable.role = VariableRole::Target;
        variable.type = VariableType::Numeric;
        variable.scaler = ScalerMethod::None;
    }

    sample_roles.assign(size_t(samples_number), SampleRole::Training);
    split_samples_random();
}

void YoloDataset::fill_inputs(const vector<Index>& sample_indices,
                              const vector<Index>&,
                              float* input_data,
                              bool,
                              bool parallelize,
                              int) const
{
    const Index batch_size = ssize(sample_indices);
    const float scale = 1.0f / 255.0f;

    string omp_error;

    #pragma omp parallel for schedule(dynamic) if (parallelize)
    for (Index i = 0; i < batch_size; ++i)
    {
        try
        {
            thread_local vector<uint8_t> pixels;
            pixels.resize(size_t(image_record_bytes));

            const uint64_t offset = sizeof(YoloImageCacheHeader)
                + uint64_t(sample_indices[size_t(i)]) * uint64_t(image_record_bytes);

            image_cache_reader.read_at(pixels.data(), pixels.size(), offset);

            float* dst = input_data + i * image_record_bytes;
            for (Index p = 0; p < image_record_bytes; ++p)
                dst[p] = float(pixels[size_t(p)]) * scale;
        }
        catch (const exception& e)
        {
            #pragma omp critical
            { omp_error = e.what(); }
        }
    }

    if (!omp_error.empty())
        throw runtime_error(omp_error);
}

void YoloDataset::fill_targets(const vector<Index>& sample_indices,
                               const vector<Index>&,
                               float* target_data,
                               bool,
                               bool parallelize,
                               int) const
{
    const Index batch_size = ssize(sample_indices);

    string omp_error;

    #pragma omp parallel for if (parallelize)
    for (Index i = 0; i < batch_size; ++i)
    {
        try
        {
            const uint64_t offset = target_data_offset
                + uint64_t(sample_indices[size_t(i)]) * uint64_t(target_record_floats) * sizeof(float);

            target_cache_reader.read_at(target_data + i * target_record_floats,
                                        size_t(target_record_floats) * sizeof(float),
                                        offset);
        }
        catch (const exception& e)
        {
            #pragma omp critical
            { omp_error = e.what(); }
        }
    }

    if (!omp_error.empty())
        throw runtime_error(omp_error);
}

void YoloDataset::to_JSON(JsonWriter& printer) const
{
    printer.open_element("YoloDataset");
    printer.open_element("DataSource");
    write_json(printer, {
        {"ImagesPath", images_directory.string()},
        {"LabelsPath", labels_directory.string()},
        {"Height", to_string(input_shape[0])},
        {"Width", to_string(input_shape[1])},
        {"Channels", to_string(input_shape[2])},
        {"GridSize", to_string(grid_size)},
        {"BoxesPerCell", to_string(boxes_per_cell)}
    });
    printer.close_element();
    variables_to_JSON(printer);
    samples_to_JSON(printer);
    printer.close_element();
}

void YoloDataset::from_JSON(const JsonDocument& document)
{
    const Json* yolo_element = get_json_root(document, "YoloDataset");
    const Json* source = require_json_field(yolo_element, "DataSource");

    set(read_json_string(source, "ImagesPath"),
        read_json_string(source, "LabelsPath"),
        {read_json_index(source, "Height"),
         read_json_index(source, "Width"),
         read_json_index(source, "Channels")},
        read_json_index(source, "GridSize"),
        read_json_index(source, "BoxesPerCell"));
}

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
