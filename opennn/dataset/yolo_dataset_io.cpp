//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   Y O L O   D A T A S E T   I M P O R T
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "opennn/dataset/yolo_dataset.h"
#include "opennn/dataset/yolo_dataset_internal.h"

#include <utility>

#include "opennn/core/io_utilities.h"
#include "opennn/core/json.h"
#include "opennn/core/string_utilities.h"
#include "opennn/dataset/image_processing.h"

namespace opennn
{

using namespace yolo_detail;

namespace
{

vector<string> read_yolo_classes(const filesystem::path& labels_directory)
{
    vector<filesystem::path> search_dirs = { labels_directory };
    if (labels_directory.has_parent_path())
        search_dirs.push_back(labels_directory.parent_path());

    for (const auto& dir : search_dirs)
    {
        if (!filesystem::is_directory(dir)) continue;
        for (const auto& entry : filesystem::directory_iterator(dir))
        {
            if (!entry.is_regular_file() || entry.path().extension() != ".names")
                continue;

            ifstream file(entry.path());
            throw_if(!file,
                     "Cannot open YOLO classes file: {}", entry.path().string());

            vector<string> classes;
            string line;
            while (getline(file, line))
                if (!line.empty())
                    classes.push_back(line);

            if (!classes.empty())
                return classes;
        }
    }

    return {};
}

void assign_default_class_names(vector<string>& class_names, Index classes_number)
{
    if (!class_names.empty()) return;

    class_names.resize(size_t(classes_number));
    for (Index i = 0; i < classes_number; ++i)
        class_names[size_t(i)] = to_string(i);
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

        const vector<float> values = parse_number_list<float>(line, "YOLO label");
        throw_if(values.size() != 5 || float(Index(values[0])) != values[0],
                 "Invalid YOLO label line in {}: {}", label_path.string(), line);

        YoloDataset::Box box;
        box.class_id = Index(values[0]);
        box.x = values[1];
        box.y = values[2];
        box.w = values[3];
        box.h = values[4];

        if (box.x < 0.0f || box.x > 1.0f || box.y < 0.0f || box.y > 1.0f
        ||  box.w < 0.0f || box.w > 1.0f || box.h < 0.0f || box.h > 1.0f)
            throw runtime_error(format("YOLO label values out of range in {}: {}", label_path.string(), line));

        boxes.push_back(box);
    }

    return boxes;
}

uint64_t hash_anchors(const vector<array<float, 2>>& anchors)
{
    uint64_t hash_value = 1469598103934665603ull;

    auto mix = [&](uint32_t value)
    {
        hash_value ^= value;
        hash_value *= 1099511628211ull;
    };

    mix(uint32_t(anchors.size()));

    for (const auto& anchor : anchors)
    {
        mix(bit_cast<uint32_t>(anchor[0]));
        mix(bit_cast<uint32_t>(anchor[1]));
    }

    return hash_value;
}

// Both target-cache writers stamp the same header; they differed only in where
// the sample count and the anchor set came from, and the two copies had already
// drifted apart in spacing.
YoloTargetCacheHeader make_target_cache_header(Index grid_size, Index boxes_per_cell,
                                               Index classes_number, uint64_t samples,
                                               Index target_record_floats,
                                               const vector<array<float, 2>>& anchors)
{
    YoloTargetCacheHeader header{};
    memcpy(header.magic, YOLO_TARGET_MAGIC, 8);
    header.version        = YOLO_CACHE_VERSION;
    header.grid_size      = uint32_t(grid_size);
    header.boxes_per_cell = uint32_t(boxes_per_cell);
    header.classes_number = uint32_t(classes_number);
    header.samples        = samples;
    header.target_floats  = uint64_t(target_record_floats);
    header.anchors_hash   = hash_anchors(anchors);
    header.anchors_offset = sizeof(YoloTargetCacheHeader);
    header.targets_offset = header.anchors_offset
        + uint64_t(anchors.size() * sizeof(array<float, 2>));
    return header;
}

uint64_t hash_sources(const filesystem::path& images_dir,
                      const filesystem::path& labels_dir)
{
    uint64_t hash_value = 1469598103934665603ull;

    auto mix_u64 = [&](uint64_t value)
    {
        for (int i = 0; i < 8; ++i)
        {
            hash_value ^= (value >> (i * 8)) & 0xff;
            hash_value *= 1099511628211ull;
        }
    };

    auto mix_bytes = [&](const void* data, size_t n)
    {
        const uint8_t* byte_pointer = static_cast<const uint8_t*>(data);
        for (size_t i = 0; i < n; ++i)
        {
            hash_value ^= byte_pointer[i];
            hash_value *= 1099511628211ull;
        }
    };

    auto mix_path = [&](const filesystem::path& path)
    {
        const string s = path.filename().string();
        mix_u64(uint64_t(s.size()));
        mix_bytes(s.data(), s.size());

        error_code ec;
        const auto size = filesystem::file_size(path, ec);
        mix_u64(ec ? 0ull : uint64_t(size));

        const auto mtime = filesystem::last_write_time(path, ec);
        if (ec)
            mix_u64(0ull);
        else
            mix_u64(uint64_t(mtime.time_since_epoch().count()));
    };

    const vector<filesystem::path> image_paths = list_files(images_dir, is_supported_image_file);
    mix_u64(uint64_t(image_paths.size()));

    for (const auto& image_path : image_paths)
    {
        mix_path(image_path);

        filesystem::path label_path = labels_dir / image_path.filename();
        label_path.replace_extension(".txt");
        if (filesystem::exists(label_path))
            mix_path(label_path);
        else
            mix_u64(0ull);
    }

    return hash_value;
}
vector<array<float, 2>> calculate_yolo_anchors(const vector<vector<YoloDataset::Box>>& labels,
                                               Index boxes_per_cell)
{
    if (boxes_per_cell <= 0)
        return {};

    vector<array<float, 2>> boxes;

    for (const auto& sample : labels)
        for (const auto& box : sample)
            if (box.w > 0.0f && box.h > 0.0f)
                boxes.push_back({box.w, box.h});

    if (boxes.empty())
        return vector<array<float, 2>>(size_t(boxes_per_cell), {0.1f, 0.1f});

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

            for (Index j = 0; j < boxes_per_cell; ++j)
            {
                const float iou = yolo_iou_wh(boxes[i], anchors[size_t(j)]);
                if (iou > best_iou)
                {
                    best_iou = iou;
                    best_anchor = j;
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

        for (Index i = 0; i < boxes_per_cell; ++i)
            if (counts[size_t(i)] > 0)
                anchors[size_t(i)] = {
                    sums[size_t(i)][0] / float(counts[size_t(i)]),
                    sums[size_t(i)][1] / float(counts[size_t(i)])
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

    output.slice(array<Index, 3>{offset_y, offset_x, 0},
                 array<Index, 3>{scaled_height, scaled_width, channels}) = resized;

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
string_view read_voc_element(string_view xml,
                             string_view open_tag,
                             string_view close_tag,
                             size_t from = 0)
{
    const size_t a = xml.find(open_tag, from);
    if (a == string::npos) return {};
    const size_t value_start = a + open_tag.size();
    const size_t b = xml.find(close_tag, value_start);
    if (b == string::npos) return {};
    return trim_view(xml.substr(value_start, b - value_start));
}

struct VocBox
{
    string class_name;
    float xmin, ymin, xmax, ymax;
};

struct VocAnnotation
{
    float width = 0.0f;
    float height = 0.0f;
    vector<VocBox> boxes;
};

VocAnnotation parse_voc_xml(const filesystem::path& xml_path)
{
    const string xml = read_text_file(xml_path);

    VocAnnotation annotation;
    const string_view size_block = read_voc_element(xml, "<size>", "</size>");
    annotation.width = parse_float(
        read_voc_element(size_block, "<width>", "</width>"), "VOC annotation: width");
    annotation.height = parse_float(
        read_voc_element(size_block, "<height>", "</height>"), "VOC annotation: height");

    throw_if(annotation.width <= 0.0f || annotation.height <= 0.0f,
             "VOC annotation has invalid size: {}", xml_path.string());

    size_t cursor = 0;
    constexpr string_view obj_open = "<object>";
    constexpr string_view obj_close = "</object>";
    while ((cursor = xml.find(obj_open, cursor)) != string::npos)
    {
        const size_t obj_end = xml.find(obj_close, cursor);
        throw_if(obj_end == string::npos,
                 "Unterminated <object> in {}", xml_path.string());

        const string_view object = string_view(xml).substr(cursor, obj_end - cursor);
        const string_view bbox = read_voc_element(object, "<bndbox>", "</bndbox>");
        VocBox box;
        box.class_name = read_voc_element(object, "<name>", "</name>");
        box.xmin = parse_float(read_voc_element(bbox, "<xmin>", "</xmin>"), "VOC bndbox: xmin");
        box.ymin = parse_float(read_voc_element(bbox, "<ymin>", "</ymin>"), "VOC bndbox: ymin");
        box.xmax = parse_float(read_voc_element(bbox, "<xmax>", "</xmax>"), "VOC bndbox: xmax");
        box.ymax = parse_float(read_voc_element(bbox, "<ymax>", "</ymax>"), "VOC bndbox: ymax");
        annotation.boxes.push_back(box);

        cursor = obj_end + obj_close.size();
    }

    return annotation;
}

const vector<string>& voc_class_names()
{
    static const vector<string> names = {
        "aeroplane", "bicycle", "bird", "boat", "bottle",
        "bus", "car", "cat", "chair", "cow",
        "diningtable", "dog", "horse", "motorbike", "person",
        "pottedplant", "sheep", "sofa", "train", "tvmonitor"
    };
    return names;
}

}

Index YoloDataset::convert_voc_to_yolo(const filesystem::path& voc_root,
                                       const string& image_set,
                                       const filesystem::path& output_labels_dir,
                                       const vector<string>& class_filter)
{
    throw_if(!filesystem::is_directory(voc_root),
             "VOC root is not a directory: {}", voc_root.string());

    const filesystem::path image_set_path =
        voc_root / "ImageSets" / "Main" / (image_set + ".txt");
    throw_if(!filesystem::is_regular_file(image_set_path),
             "VOC image-set file not found: {}",
                    image_set_path.string());

    const filesystem::path annotations_dir = voc_root / "Annotations";
    throw_if(!filesystem::is_directory(annotations_dir),
             "VOC Annotations dir not found: {}",
                    annotations_dir.string());

    filesystem::create_directories(output_labels_dir);

    const vector<string>& active_classes = class_filter.empty() ? voc_class_names() : class_filter;
    unordered_map<string, Index> class_index;
    for (Index i = 0; i < ssize(active_classes); ++i)
        class_index[active_classes[size_t(i)]] = i;

    ofstream names_file(output_labels_dir / "voc.names");
    throw_if(!names_file,
             "Cannot write VOC names file in {}",
                    output_labels_dir.string());
    for (const string& name : active_classes)
        names_file << name << '\n';
    names_file.close();

    ifstream id_file(image_set_path);
    string image_id;
    Index converted = 0;

    while (id_file >> image_id)
    {
        if (image_id.empty()) continue;

        const filesystem::path xml_path = annotations_dir / (image_id + ".xml");
        if (!filesystem::is_regular_file(xml_path))
            continue;

        const VocAnnotation ann = parse_voc_xml(xml_path);

        vector<pair<Index, array<float,4>>> kept_boxes;
        for (const VocBox& box : ann.boxes)
        {
            const auto it = class_index.find(box.class_name);
            if (it == class_index.end())
                continue;
            const float cx = clamp_unit(0.5f * (box.xmin + box.xmax) / ann.width);
            const float cy = clamp_unit(0.5f * (box.ymin + box.ymax) / ann.height);
            const float bw = clamp_unit((box.xmax - box.xmin) / ann.width);
            const float bh = clamp_unit((box.ymax - box.ymin) / ann.height);
            kept_boxes.push_back({it->second, {cx, cy, bw, bh}});
        }

        if (!class_filter.empty() && kept_boxes.empty())
            continue;

        const filesystem::path out_path = output_labels_dir / (image_id + ".txt");
        ofstream out(out_path);
        throw_if(!out, "Cannot write YOLO label: {}", out_path.string());
        for (const auto& [id, b] : kept_boxes)
            out << id << ' ' << b[0] << ' ' << b[1] << ' ' << b[2] << ' ' << b[3] << '\n';
        ++converted;
    }

    return converted;
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
    throw_if(new_input_shape.get_rank() != 3,
             "YoloDataset: input_shape must be rank 3.");
    throw_if(new_grid_size <= 0,
             "YoloDataset: grid_size must be positive.");
    throw_if(new_boxes_per_cell < 0,
             "YoloDataset: boxes_per_cell must be non-negative (0 = v8 anchor-free mode).");

    images_directory = new_images_dir;
    labels_directory = new_labels_dir;
    data_path = images_directory;
    storage_mode = StorageMode::BinaryFile;
    input_shape = new_input_shape;
    grid_size = new_grid_size;
    boxes_per_cell = new_boxes_per_cell;
    image_record_bytes = input_shape.size();
    class_names = read_yolo_classes(labels_directory);

    image_cache_path = images_directory / ".cache" / "yolo_images.bin";
    target_cache_path = images_directory / ".cache" / "yolo_targets.bin";
    boxes_cache_path = images_directory / ".cache" / "yolo_boxes.bin";
    images_ram.clear();
    targets_ram.clear();

    image_filenames = list_files(images_directory, is_supported_image_file);

    open_or_build_cache(new_anchors);
}

void YoloDataset::set_storage_mode(StorageMode new_storage_mode)
{
    if (new_storage_mode == StorageMode::Matrix)
        load_cache_to_ram();

    Dataset::set_storage_mode(new_storage_mode);

    if (new_storage_mode == StorageMode::BinaryFile)
    {
        images_ram.clear();
        targets_ram.clear();
    }
}
void YoloDataset::open_or_build_cache(const vector<array<float, 2>>& requested_anchors)
{
    if (try_open_cache(requested_anchors))
        return;

    image_cache_reader.close();
    target_cache_reader.close();
    boxes_cache_reader.close();

    if (try_rebuild_target_from_boxes(requested_anchors))
        return;

    build_cache(requested_anchors);
}

bool YoloDataset::try_rebuild_target_from_boxes(const vector<array<float, 2>>& requested_anchors)
{
    if (!filesystem::exists(image_cache_path) || !filesystem::exists(boxes_cache_path))
        return false;

    try
    {
        image_cache_reader.open(image_cache_path);
        boxes_cache_reader.open(boxes_cache_path);

        YoloImageCacheHeader image_header{};
        image_cache_reader.read_at(span(&image_header, 1), 0);

        if (memcmp(image_header.magic, YOLO_IMAGE_MAGIC, 8) != 0
        ||  image_header.version != YOLO_CACHE_VERSION
        ||  Index(image_header.height)   != input_shape[0]
        ||  Index(image_header.width)    != input_shape[1]
        ||  Index(image_header.channels) != input_shape[2]
        ||  image_header.sources_hash   != hash_sources(images_directory, labels_directory))
        {
            image_cache_reader.close();
            boxes_cache_reader.close();
            return false;
        }

        YoloBoxesCacheHeader boxes_header{};
        boxes_cache_reader.read_at(span(&boxes_header, 1), 0);

        if (memcmp(boxes_header.magic, YOLO_BOXES_MAGIC, 8) != 0
        ||  boxes_header.version != YOLO_CACHE_VERSION
        ||  boxes_header.samples != image_header.samples)
        {
            image_cache_reader.close();
            boxes_cache_reader.close();
            return false;
        }

        const size_t n_samples = static_cast<size_t>(boxes_header.samples);

        vector<uint64_t> offsets(n_samples + 1);
        boxes_cache_reader.read_at(span(offsets), boxes_header.offsets_byte_offset);

        vector<YoloBoxRecord> raw_boxes(static_cast<size_t>(boxes_header.total_boxes));
        if (boxes_header.total_boxes > 0)
            boxes_cache_reader.read_at(span(raw_boxes), boxes_header.boxes_byte_offset);

        vector<vector<Box>> labels(n_samples);
        Index max_class_id = -1;
        for (size_t i = 0; i < n_samples; ++i)
        {
            const size_t start = static_cast<size_t>(offsets[i]);
            const size_t end   = static_cast<size_t>(offsets[i + 1]);
            labels[i].reserve(end - start);
            for (size_t j = start; j < end; ++j)
            {
                const auto& r = raw_boxes[j];
                labels[i].push_back({Index(r.class_id), r.x, r.y, r.w, r.h});
                max_class_id = max(max_class_id, Index(r.class_id));
            }
        }

        classes_number = class_names.empty() ? max_class_id + 1 : ssize(class_names);
        if (classes_number <= 0)
        {
            image_cache_reader.close();
            boxes_cache_reader.close();
            return false;
        }
        assign_default_class_names(class_names, classes_number);

        const vector<array<float, 2>> new_anchors = requested_anchors.empty()
            ? calculate_yolo_anchors(labels, boxes_per_cell)
            : requested_anchors;

        if (ssize(new_anchors) != boxes_per_cell)
        {
            image_cache_reader.close();
            boxes_cache_reader.close();
            return false;
        }

        target_record_floats = v8_mode
            ? MAX_GT_BOXES * 5
            : grid_size * grid_size * boxes_per_cell * (5 + classes_number);

        filesystem::create_directories(target_cache_path.parent_path());
        const filesystem::path target_tmp_path = target_cache_path.string() + ".tmp";
        FileWriter target_writer;
        target_writer.open(target_tmp_path);

        const YoloTargetCacheHeader target_header = make_target_cache_header(
            grid_size, boxes_per_cell, classes_number, uint64_t(n_samples),
            target_record_floats, new_anchors);

        target_writer.write(span(&target_header, 1));
        target_writer.write(span(new_anchors));

        vector<float> target_buf(static_cast<size_t>(target_record_floats), 0.f);
        for (const auto& sample_boxes : labels)
        {
            fill(target_buf.begin(), target_buf.end(), 0.f);
            if (v8_mode)
                make_target_v8_gtlist(sample_boxes, classes_number, target_buf.data());
            else
                make_target(sample_boxes, new_anchors, grid_size, boxes_per_cell,
                            classes_number, target_buf.data());
            target_writer.write(span(target_buf));
        }
        target_writer.finish_with_rename(target_cache_path);

        target_cache_reader.open(target_cache_path);
        anchors            = new_anchors;
        target_data_offset = target_header.targets_offset;
        boxes_offsets      = move(offsets);
        boxes_data_offset  = boxes_header.boxes_byte_offset;

        setup_metadata(Index(n_samples));

        if (display)
            cout << "\nYOLO target cache rebuilt (" << n_samples
                 << " samples, grid=" << grid_size
                 << ", bpc=" << boxes_per_cell << ").\n";

        return true;
    }
    catch (const exception&)
    {
        image_cache_reader.close();
        target_cache_reader.close();
        boxes_cache_reader.close();
        return false;
    }
}

bool YoloDataset::try_open_cache(const vector<array<float, 2>>& requested_anchors)
{
    if (!filesystem::exists(image_cache_path)
    ||  !filesystem::exists(target_cache_path)
    ||  !filesystem::exists(boxes_cache_path))
        return false;

    try
    {
        image_cache_reader.open(image_cache_path);
        target_cache_reader.open(target_cache_path);
        boxes_cache_reader.open(boxes_cache_path);

        YoloImageCacheHeader image_header{};
        YoloTargetCacheHeader target_header{};
        image_cache_reader.read_at(span(&image_header, 1), 0);
        target_cache_reader.read_at(span(&target_header, 1), 0);

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

        if (image_header.sources_hash != hash_sources(images_directory, labels_directory))
            return false;

        vector<array<float, 2>> cached_anchors(static_cast<size_t>(boxes_per_cell));
        target_cache_reader.read_at(span(cached_anchors),
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

        if (!class_names.empty() && Index(target_header.classes_number) != ssize(class_names))
            return false;

        anchors = std::move(cached_anchors);
        classes_number = Index(target_header.classes_number);
        assign_default_class_names(class_names, classes_number);

        target_record_floats = Index(target_header.target_floats);
        target_data_offset = target_header.targets_offset;

        YoloBoxesCacheHeader boxes_header{};
        boxes_cache_reader.read_at(span(&boxes_header, 1), 0);
        if (memcmp(boxes_header.magic, YOLO_BOXES_MAGIC, 8) != 0
        ||  boxes_header.version != YOLO_CACHE_VERSION
        ||  boxes_header.samples != image_header.samples)
            return false;

        boxes_offsets.assign(size_t(boxes_header.samples + 1), 0);
        boxes_cache_reader.read_at(span(boxes_offsets),
                                   boxes_header.offsets_byte_offset);
        if (boxes_offsets.empty()
        ||  boxes_offsets.front() != 0
        ||  boxes_offsets.back() != boxes_header.total_boxes)
            return false;

        for (size_t i = 1; i < boxes_offsets.size(); ++i)
            if (boxes_offsets[i] < boxes_offsets[i - 1])
                return false;

        boxes_data_offset = boxes_header.boxes_byte_offset;

        const uint64_t expected_boxes_size = boxes_header.boxes_byte_offset
            + boxes_header.total_boxes * sizeof(YoloBoxRecord);
        if (boxes_cache_reader.file_size() != expected_boxes_size)
            return false;

        setup_metadata(Index(image_header.samples));
        return true;
    }
    catch (const exception&)
    {
        image_cache_reader.close();
        target_cache_reader.close();
        boxes_cache_reader.close();
        return false;
    }
}

void YoloDataset::build_cache(const vector<array<float, 2>>& requested_anchors)
{
    const vector<filesystem::path>& image_paths = image_filenames;
    throw_if(image_paths.empty(),
             "YoloDataset: no images found in {}", images_directory.string());

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
    image_header.sources_hash = hash_sources(images_directory, labels_directory);
    image_writer.write(span(&image_header, 1));

    vector<uint8_t> pixels(static_cast<size_t>(image_record_bytes));

    for (size_t i = 0; i < image_paths.size(); ++i)
    {
        Tensor3 image = load_image(image_paths[i]);

        if (image.dimension(2) == 1 && input_shape[2] == 3)
        {
            Tensor3 rgb(image.dimension(0), image.dimension(1), 3);
            rgb.chip(0, 2) = image.chip(0, 2);
            rgb.chip(1, 2) = image.chip(0, 2);
            rgb.chip(2, 2) = image.chip(0, 2);
            image = std::move(rgb);
        }
        throw_if(image.dimension(2) != input_shape[2],
                 "YoloDataset: channel mismatch in {} (got {} channels, expected {})",
                        image_paths[i].string(), image.dimension(2), input_shape[2]);

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

        Map<Array<uint8_t, Dynamic, 1>>(pixels.data(), image_record_bytes) =
            (Map<const Array<float, Dynamic, 1>>(prepared.data(), image_record_bytes)
                .max(0.0f).min(255.0f) + 0.5f).cast<uint8_t>();

        image_writer.write(span(pixels));

        if (display && (i % 1000 == 0 || i + 1 == image_paths.size()))
            display_progress_bar(Index(i + 1), ssize(image_paths));
    }

    image_writer.finish_with_rename(image_cache_path);

    classes_number = class_names.empty() ? max_class_id + 1 : ssize(class_names);
    throw_if(classes_number <= 0,
             "YoloDataset: cannot infer classes_number.");

    assign_default_class_names(class_names, classes_number);

    anchors = requested_anchors.empty()
        ? calculate_yolo_anchors(labels, boxes_per_cell)
        : requested_anchors;

    throw_if(ssize(anchors) != boxes_per_cell,
             "YoloDataset: anchors size must equal boxes_per_cell.");

    target_record_floats = v8_mode
        ? MAX_GT_BOXES * 5
        : grid_size * grid_size * boxes_per_cell * (5 + classes_number);

    const filesystem::path target_tmp_path = target_cache_path.string() + ".tmp";
    FileWriter target_writer;
    target_writer.open(target_tmp_path);

    const YoloTargetCacheHeader target_header = make_target_cache_header(
        grid_size, boxes_per_cell, classes_number, uint64_t(image_paths.size()),
        target_record_floats, anchors);

    target_writer.write(span(&target_header, 1));
    target_writer.write(span(anchors));

    vector<float> target(static_cast<size_t>(target_record_floats));
    for (const auto& sample_boxes : labels)
    {
        make_target(sample_boxes, anchors, grid_size, boxes_per_cell, classes_number, target.data());
        target_writer.write(span(target));
    }

    target_writer.finish_with_rename(target_cache_path);

    const filesystem::path boxes_tmp_path = boxes_cache_path.string() + ".tmp";
    FileWriter boxes_writer;
    boxes_writer.open(boxes_tmp_path);

    uint64_t total_boxes = 0;
    for (const auto& sample_boxes : labels) total_boxes += sample_boxes.size();

    YoloBoxesCacheHeader boxes_header{};
    memcpy(boxes_header.magic, YOLO_BOXES_MAGIC, 8);
    boxes_header.version = YOLO_CACHE_VERSION;
    boxes_header.samples = uint64_t(image_paths.size());
    boxes_header.total_boxes = total_boxes;
    boxes_header.offsets_byte_offset = sizeof(YoloBoxesCacheHeader);
    boxes_header.boxes_byte_offset = boxes_header.offsets_byte_offset
        + (boxes_header.samples + 1) * sizeof(uint64_t);

    boxes_writer.write(span(&boxes_header, 1));

    vector<uint64_t> offsets(image_paths.size() + 1, 0);
    for (size_t i = 0; i < image_paths.size(); ++i)
        offsets[i + 1] = offsets[i] + labels[i].size();
    boxes_writer.write(span(offsets));

    for (const auto& sample_boxes : labels)
        for (const auto& box : sample_boxes)
        {
            const YoloBoxRecord rec{int32_t(box.class_id), box.x, box.y, box.w, box.h};
            boxes_writer.write(span(&rec, 1));
        }

    boxes_writer.finish_with_rename(boxes_cache_path);

    image_cache_reader.open(image_cache_path);
    target_cache_reader.open(target_cache_path);
    boxes_cache_reader.open(boxes_cache_path);
    target_data_offset = target_header.targets_offset;
    boxes_data_offset = boxes_header.boxes_byte_offset;
    boxes_offsets = std::move(offsets);

    setup_metadata(Index(image_paths.size()));

    if (display)
        cout << "\nYOLO cache built (" << samples_number << " samples).\n";
}

void YoloDataset::setup_metadata(Index new_samples_number)
{
    samples_number = new_samples_number;

    cache_input_shape = input_shape;
    cache_grid_size = grid_size;
    cache_image_record_bytes = image_record_bytes;
    cache_target_record_floats = target_record_floats;

    target_shape = {grid_size, grid_size, boxes_per_cell * (5 + classes_number)};

    variables.assign(2, Variable());

    Variable& image_variable = variables[0];
    image_variable.name = "image";
    image_variable.role = VariableRole::Input;
    image_variable.type = VariableType::Numeric;
    image_variable.scaler = ScalerMethod::None;
    image_variable.features = input_shape.size();

    Variable& target_variable = variables[1];
    target_variable.name = "yolo_target";
    target_variable.role = VariableRole::Target;
    target_variable.type = VariableType::Numeric;
    target_variable.scaler = ScalerMethod::None;
    target_variable.features = target_shape.size();

    sample_roles.assign(size_t(samples_number), SampleRole::Training);
    split_samples_random();
}
void YoloDataset::load_cache_to_ram()
{
    if (samples_number == 0) return;

    const size_t image_bytes = size_t(samples_number) * size_t(cache_image_record_bytes);
    const size_t target_values = size_t(samples_number) * size_t(cache_target_record_floats);
    if (images_ram.size() == image_bytes && targets_ram.size() == target_values)
        return;

    throw_if(!image_cache_reader.is_open(),
             "YoloDataset::load_cache_to_ram: image cache is not open.");
    throw_if(!target_cache_reader.is_open(),
             "YoloDataset::load_cache_to_ram: target cache is not open.");

    vector<uint8_t> loaded_images(image_bytes);
    vector<float> loaded_targets(target_values);

    image_cache_reader.read_at(span(loaded_images), sizeof(YoloImageCacheHeader));
    target_cache_reader.read_at(span(loaded_targets), target_data_offset);

    images_ram = std::move(loaded_images);
    targets_ram = std::move(loaded_targets);
}
void YoloDataset::to_JSON(JsonWriter& printer) const
{
    printer.open_element("YoloDataset");
    printer.open_element("DataSource");
    write_json(printer, {
        {"ImagesPath", images_directory.string()},
        {"LabelsPath", labels_directory.string()},
        {"StorageMode", get_storage_mode_string()},
        {"Height", input_shape[0]},
        {"Width", input_shape[1]},
        {"Channels", input_shape[2]},
        {"GridSize", grid_size},
        {"BoxesPerCell", boxes_per_cell},
        {"DisplayConfidenceThreshold", display_confidence_threshold},
        {"AugEnabled",    augmentation_policy.enabled    ? 1 : 0},
        {"AugJitter",     augmentation_policy.jitter},
        {"AugExposure",   augmentation_policy.exposure},
        {"AugSaturation", augmentation_policy.saturation},
        {"AugHue",        augmentation_policy.hue},
        {"AugFlip",       augmentation_policy.flip    ? 1 : 0},
        {"AugMosaic",     augmentation_policy.mosaic  ? 1 : 0}
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

    set_storage_mode(source->has("StorageMode")
                   ? read_json_string(source, "StorageMode")
                   : "BinaryFile");

    if (source->has("DisplayConfidenceThreshold"))
        display_confidence_threshold = read_json_float(source, "DisplayConfidenceThreshold");

    AugmentationPolicy parsed_policy;
    parsed_policy.enabled    = source->has("AugEnabled")    ? (read_json_index(source, "AugEnabled")    != 0) : true;
    parsed_policy.jitter     = source->has("AugJitter")     ? read_json_float(source, "AugJitter")     : 0.2f;
    parsed_policy.exposure   = source->has("AugExposure")   ? read_json_float(source, "AugExposure")   : 1.5f;
    parsed_policy.saturation = source->has("AugSaturation") ? read_json_float(source, "AugSaturation") : 1.5f;
    parsed_policy.hue        = source->has("AugHue")        ? read_json_float(source, "AugHue")        : 0.1f;
    parsed_policy.flip       = source->has("AugFlip")       ? (read_json_index(source, "AugFlip")      != 0) : true;
    parsed_policy.mosaic     = source->has("AugMosaic")     ? (read_json_index(source, "AugMosaic")    != 0) : false;
    set_augmentation_policy(parsed_policy);

    const Json* samples_element = yolo_element->find("Samples");
    if (samples_element)
        samples_from_JSON(samples_element);
}

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
