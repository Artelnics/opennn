#include "yolo_dataset.h"
#include "images.h"
#include "tensors.h"
#include "strings_utilities.h"

using namespace fs;

namespace opennn
{

YOLODataset::YOLODataset(const string& images_directory, const string& labels_directory)
{

    model_type = ModelType::ObjectDetection;

    for (const auto& entry : fs::directory_iterator(images_directory))
    {
        if (entry.is_regular_file() && entry.path().extension() == ".bmp")
        {
            images_files.push_back(entry.path().string());
        }
    }

    sort(images_files.begin(), images_files.end());

    images.resize(images_files.size());
    tensor_images.resize(images_files.size(), 416, 416, 3);
    offsets.resize(images_files.size());

    #pragma omp parallel for
    for(size_t i = 0; i < images_files.size(); i++)
    {
        Tensor<type, 1> image_offsets;

        images[i] = /*normalize_tensor(*/resize_image_416x416(read_bmp_image(images_files[i]).cast<type>(), image_offsets)/*, false)*/;
        offsets[i] = image_offsets;
        tensor_images.chip(i,0) = images[i];

    }

    for (const auto& entry : fs::directory_iterator(labels_directory))
    {

        if (entry.is_regular_file() && entry.path().extension() == ".txt")
        {
            labels_files.push_back(entry.path().string());
        }
        if(entry.is_regular_file() && entry.path().extension() == ".names")
        {
            classes = read_classes(entry.path().string());
        }
    }

    sort(labels_files.begin(),labels_files.end());

    for(size_t j = 0; j < labels_files.size(); j++)
    {
        raw_labels.push_back(read_bounding_boxes(labels_files[j]));
    }


    labels.resize(raw_labels.size());

    #pragma omp parallel for
    for(size_t i = 0; i < raw_labels.size(); i++)
    {

        Index scaled_width = 416 - 2 * offsets[i](0);
        Index scaled_height = 416 - 2 * offsets[i](1);

        Index offset_x = offsets[i](0);
        Index offset_y = offsets[i](1);

        Tensor<type, 2> modified_label(raw_labels[i].dimension(0), 5);

        for(Index j = 0; j < raw_labels[i].dimension(0); j++)
        {
            type x_center = raw_labels[i](j, 1) * scaled_width;
            type y_center = raw_labels[i](j, 2) * scaled_height;
            type width = raw_labels[i](j, 3) * scaled_width;
            type height = raw_labels[i](j, 4) * scaled_height;

            modified_label(j, 0) = raw_labels[i](j, 0);
            modified_label(j, 1) = (x_center  + offset_x) / 416;
            modified_label(j, 2) = 1 - (y_center + offset_y) / 416;         //I do the 1 - (...) because the y coordinate is given reversed (we are working with y = 0 being the top part of the image and the label is given as if it started from the bottom part)
            modified_label(j, 3) = width / 416;
            modified_label(j, 4) = height / 416;
        }
        labels[i] = modified_label;
    }

    anchors = calculate_anchors(labels, anchor_number);

    if (images.size() != labels.size()) {
        // cerr << "Images and labels file number do not match!" << endl;
        throw runtime_error("Images and labels file number do not match!");
    }

    targets.resize(images.size());
    tensor_targets.resize(images.size(), 13, 13, 125);

    for(size_t img = 0; img < images.size(); img++)
    {
        targets[img] = convert_to_YOLO_grid_data(labels[img], anchors, grid_size, anchor_number, classes.size());
        tensor_targets.chip(img, 0) = targets[img];
    }


    target_dimensions = {{grid_size, grid_size, (Index) (anchor_number * (5 + classes.size()))}};

    input_dimensions = {{416, 416, 3}};

    set(images.size(), {{416, 416, 3}}, {{grid_size, grid_size, (Index) (anchor_number * (5 + classes.size()))}});

    for(Index i = 0; i < data.dimension(0); i++)
    {
        for(Index j = 0; j < images[i].size(); j++)
        {
            data(i, j) = images[i](j);
        }
        for(Index k = 0; k < targets[i].size(); k++)
        {
            data(i, k + input_dimensions[0]*input_dimensions[1]*input_dimensions[2]) = targets[i](k);
        }
    }

}

YOLODataset::YOLODataset(const string& images_directory)
{
    for (const auto& entry : fs::directory_iterator(images_directory))
    {
        if (entry.is_regular_file() && entry.path().extension() == ".bmp")
        {
            images_files.push_back(entry.path().string());
        }
    }
    sort(images_files.begin(), images_files.end());

    images.resize(images_files.size());

#pragma omp parallel for
    for(size_t i = 0; i < images_files.size(); i++)
    {
        images[i] = normalize_tensor(resize_image_416x416(read_bmp_image(images_files[i]).cast<type>()), false);
    }

    model_type = ModelType::ObjectDetection;
}

vector<Descriptives> YOLODataset::scale_variables(const VariableUse&)
{
    TensorMap<Tensor<type, 4>> inputs_data(data.data(),
                                           get_samples_number(),
                                           input_dimensions[0],
                                           input_dimensions[1],
                                           input_dimensions[2]);

    inputs_data.device(*thread_pool_device) = inputs_data / type(255);

    return vector<Descriptives>();
}


size_t YOLODataset::size() const
{
    return images.size();
}

Tensor<type, 3> YOLODataset::get_image(const Index& index) const
{
    return images[index];
}

Tensor<type, 2> YOLODataset::get_label(const Index& index) const
{
    return labels[index];
}

Tensor<type, 4> YOLODataset::get_images() const
{    
    return tensor_images;
}

Tensor<type, 4> YOLODataset::get_targets() const
{
    return tensor_targets;
}

vector<Tensor<type, 1>> YOLODataset::get_anchors() const
{
    return anchors;
}

string YOLODataset::get_class(const Index& index) const
{
    return classes[index];
}

void YOLODataset::rotate_90_degrees(Tensor<type, 3>& image, Tensor<type, 2>& labels)
{
    Eigen::array<Index, 3> rotation = {1,0,2};
    Tensor<type, 3> rotated_image = image.shuffle(rotation);
    image = rotated_image;
    for(Index i = 0; i < labels.dimension(0); i++)
    {
        type old_x, temp_w;

        old_x = labels(i, 1);
        labels(i, 1) = labels(i, 2);
        labels(i, 2) = old_x;
        temp_w = labels(i, 3);
        labels(i, 3) = labels(i, 4);
        labels(i, 4) = temp_w;
    }
}

void YOLODataset::flip_image_horizontally(Tensor<type, 3>& image, Tensor<type, 2>& labels)
{
    Tensor<type, 3> flipped_image(image.dimension(0),image.dimension(1),image.dimension(2));

    for(Index i = 0; i < image.dimension(0); i++)
    {
        for(Index j = 0; j < image.dimension(1); j++)
        {
            flipped_image(i,j,0) = image(image.dimension(0) - (i + 1), j, 0);
            flipped_image(i,j,1) = image(image.dimension(0) - (i + 1), j, 1);
            flipped_image(i,j,2) = image(image.dimension(0) - (i + 1), j, 2);
        }
    }

    for(Index k = 0; k < labels.dimension(0); k++)
    {
        labels(k,1) = labels(k,1);
        labels(k,2) = 1 - labels(k,2);
    }

    image = flipped_image;
}

void YOLODataset::adjust_brightness_contrast(Tensor<type, 3>& image, const type& brightness_factor, const type& contrast_factor)
{
    for(Index i = 0; i < image.dimension(0); i++)
    {
        for(Index j = 0; j < image.dimension(1); j++)
        {
            for(Index k = 0; k < image.dimension(2); k++)
            {
                image(i,j,k) = (type) min((Index)(image(i,j,k) * contrast_factor + brightness_factor), (Index) 1.0);
            }
        }
    }
}

void YOLODataset::apply_data_augmentation(Tensor<type, 3>& image, Tensor<type, 2>& labels)
{
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<> dist(0.0, 1.0);

    if(dist(gen) > 0.5) flip_image_horizontally(image, labels);
    if(dist(gen) > 0.5) rotate_90_degrees(image, labels);
    if(dist(gen) > 0.5) adjust_brightness_contrast(image, dist(gen) * 0.1 + 0.55, dist(gen) * 0.2 + 0.7);
}


Tensor<type, 2> read_bounding_boxes(const string& label_path)
{

    ifstream infile(label_path);
    if (!infile.is_open()) {
        cerr << "Unable to open the labels file: " << label_path << endl;
        return Tensor<type, 2>(0, 5);                   // Returns an empty vector if unable to read the file
    }

    vector<YOLODataset::YOLO_bounding_box> bounding_boxes;
    string line;

    // cout << "Reading labels file: " << label_path << endl;

    // Read each line of the labels file
    while (getline(infile, line))
    {
        stringstream ss(line);
        YOLODataset::YOLO_bounding_box box;

        // Read each value on the line and check if they are valid
        if (!(ss >> box.class_id >> box.x_center >> box.y_center >> box.width >> box.height)) {
            cerr << "Wrong format in the file: " << label_path << " in line: " << line << endl;
            continue;  // Skip the lines with wrong format
        }

        /* cout << "Read: class_id=" << box.class_id
             << ", x_center=" << box.x_center
             << ", y_center=" << box.y_center
             << ", width=" << box.width
             << ", height=" << box.height << endl;
*/
        if (box.x_center < 0.0f || box.x_center > 1.0f ||
            box.y_center < 0.0f || box.y_center > 1.0f ||
            box.width < 0.0f || box.width > 1.0f ||
            box.height < 0.0f || box.height > 1.0f) {
            cerr << "values of the bounding box out of the range in the file: " << label_path << " on the line: " << line << endl;
            continue;
        }

        bounding_boxes.push_back(box);
    }

    //cout << "Total read labels: " << bounding_boxes.size() << endl;

    Index n_labels = bounding_boxes.size();
    Tensor<type, 2> label_tensor(n_labels, 5);

#pragma omp parallel for
    for (Index i = 0; i < n_labels; ++i) {
        label_tensor(i, 0) = bounding_boxes[i].class_id;
        label_tensor(i, 1) = bounding_boxes[i].x_center;
        label_tensor(i, 2) = bounding_boxes[i].y_center;
        label_tensor(i, 3) = bounding_boxes[i].width;
        label_tensor(i, 4) = bounding_boxes[i].height;
    }

    //cout << "Labels converted to a tensor of dimensions: [" << label_tensor.dimension(0)
    //   << ", " << label_tensor.dimension(1) << "]" << endl;

    return label_tensor;
}

vector<string> read_classes(const string& classes_path)
{
    ifstream infile(classes_path);
    if (!infile.is_open()) {
        throw runtime_error("Unable to open the labels file: " );
    }
    vector<string> classes;
    string line;
    while (getline(infile, line))
    {
        stringstream ss(line);

        classes.push_back(line);
    }
    return classes;
}

Tensor<type, 3> scale_image(const Tensor<type, 3>& image, const Index& new_height, const Index& new_width)
{

    const Index original_height = image.dimension(0);
    const Index original_width = image.dimension(1);
    const Index channels = image.dimension(2);


    Tensor<type, 3> scaled_image(new_height, new_width, channels);

    // Calculate the scale factors
    type scale_y = static_cast<type>(original_height) / new_height;
    type scale_x = static_cast<type>(original_width) / new_width;


    for (Index y = 0; y < new_height; ++y)
    {
        for (Index x = 0; x < new_width; ++x)
        {
            for (Index c = 0; c < channels; ++c)
            {

                type original_y = y * scale_y;
                type original_x = x * scale_x;

                Index y0 = floor(original_y);
                Index x0 = floor(original_x);
                Index y1 = min(y0 + 1, original_height - 1);
                Index x1 = min(x0 + 1, original_width - 1);

                // Pixel values in the original image
                type top_left = image(y0, x0, c);
                type top_right = image(y0, x1, c);
                type bottom_left = image(y1, x0, c);
                type bottom_right = image(y1, x1, c);

                // Bilineal interpolation
                type value = top_left * (x1 - original_x) * (y1 - original_y) +
                             top_right * (original_x - x0) * (y1 - original_y) +
                             bottom_left * (x1 - original_x) * (original_y - y0) +
                             bottom_right * (original_x - x0) * (original_y - y0);

                scaled_image(y, x, c) = round(value);
            }
        }
    }

    return scaled_image;
}

Tensor<type, 3> apply_zero_padding(const Tensor<type, 3>& image, const Index& target_height, const Index& target_width)
{

    const Index original_height = image.dimension(0);
    const Index original_width = image.dimension(1);
    const Index channels = image.dimension(2);


    Tensor<type, 3> padded_image(target_height, target_width, channels);
    padded_image.setZero();

    // Calculate the offset to center the image
    Index offset_y = (target_height - original_height) / 2;
    Index offset_x = (target_width - original_width) / 2;

    // Copy the original image in the center of the new tensor with padding
    for (Index y = 0; y < original_height; ++y) {
        for (Index x = 0; x < original_width; ++x) {
            for (Index c = 0; c < channels; ++c) {
                padded_image(y + offset_y, x + offset_x, c) = image(y, x, c);
            }
        }
    }

    return padded_image;
}

Tensor<type, 3> resize_image_416x416(const Tensor<type, 3>& image)
{

    if(image.dimension(0) == 416 && image.dimension(1) == 416)
    {
        return image;
    }
    else
    {
        const Index original_height = image.dimension(0);
        const Index original_width = image.dimension(1);
        //Index channels = image.dimension(2);

        // Calculate the scaling factor to preserve the proportions of the image
        type scale = 416.0f / max(original_height, original_width);


        Index new_height = static_cast<Index>(original_height * scale);
        Index new_width = static_cast<Index>(original_width * scale);


        Tensor<type, 3> scaled_image = scale_image(image, new_height, new_width);

        Tensor<type, 3> final_image = apply_zero_padding(scaled_image, 416, 416);

        return final_image.abs();
    }
}

Tensor<type, 3> resize_image_416x416(const Tensor<type, 3>& image, Tensor<type, 1>& offsets)
{

    offsets.resize(2);

    if(image.dimension(0) == 416 && image.dimension(1) == 416)
    {
        offsets.setZero();
        return image;
    }
    else
    {
        const Index original_height = image.dimension(0);
        const Index original_width = image.dimension(1);
        //Index channels = image.dimension(2);

        // Calculate the scaling factor to preserve the proportions of the image
        type scale = 416.0f / max(original_height, original_width);


        Index new_height = static_cast<Index>(original_height * scale);
        Index new_width = static_cast<Index>(original_width * scale);


        Tensor<type, 3> scaled_image = scale_image(image, new_height, new_width);

        offsets(0) = (416 - scaled_image.dimension(1)) / 2;     //Offset_x (the dimensions are stored (height, width, channels))

        offsets(1) = (416 - scaled_image.dimension(0)) / 2;     //Offset_y (the dimensions are stored (height, width, channels))

        Tensor<type, 3> final_image = apply_zero_padding(scaled_image, 416, 416);

        return final_image.abs();
    }
}

Tensor<type, 3>  normalize_tensor(const Tensor<type, 3>& image, const bool& is_normal)
{
    Tensor<type, 3> normalized_image;
    if(is_normal)
    {
        normalized_image = image * (type)255;
    }
    else
    {
        normalized_image = image / (type)255;
    }

    return normalized_image;
}


Tensor<type, 2> extract_boxes_width_height_data(const vector<Tensor<type, 2> >&labels)
{
    vector<Tensor<type, 1>> box_dimensions;
    // const Index image_size = 416;
    Tensor<type, 1> dimension(2);

    for(size_t i = 0; i < labels.size(); i++)
    {
        for(Index j = 0; j < labels[i].dimension(0); j++)
        {
            dimension(0) = labels[i](j, 3)/* * image_size*/;
            dimension(1) = labels[i](j, 4)/* * image_size*/;
            box_dimensions.push_back(dimension);
        }
    }

    size_t boxes_number = box_dimensions.size();

    Tensor<type, 2> tensor_box_dimensions(boxes_number, 2);

    for(size_t k = 0; k < boxes_number; k++)
    {
        tensor_box_dimensions(k,0) = box_dimensions[k](0);
        tensor_box_dimensions(k,1) = box_dimensions[k](1);
        // cout<<tensor_box_dimensions(k,0)<<", "<<tensor_box_dimensions(k,1)<<endl;
    }

    return tensor_box_dimensions;

}

type calculate_intersection_over_union(const Tensor<type, 1>& box_1, const Tensor<type, 1>& box_2)
{
    if(box_1.dimension(0) < 4 || box_2.dimension(0) < 4)
        throw runtime_error("Boxes must have 4 coordinates");

    type x_left = max(box_1(0) - box_1(2) / 2, box_2(0) - box_2(2) / 2);
    type y_top = max(box_1(1) - box_1(3) / 2, box_2(1) - box_2(3) / 2);
    type x_right = min(box_1(0) + box_1(2) / 2, box_2(0) + box_2(2) / 2);
    type y_bottom = min(box_1(1) + box_1(3) / 2, box_2(1) + box_2(3) / 2);

    // cout << "x_left: " << x_left << endl;
    // cout << "x_right: " << x_right << endl;
    // cout << "y_top: " << y_top << endl;
    // cout << "y_bottom: " << y_bottom << endl;

    // throw runtime_error("Checking boxes");

    type intersection_area = max(0.0f, x_right - x_left) * max(0.0f, y_bottom - y_top);

    type box_1_area = box_1(2) * box_1(3);
    type box_2_area = box_2(2) * box_2(3);

    type union_area = box_1_area + box_2_area - intersection_area;

    return intersection_area / union_area;
}

type calculate_intersection_over_union_anchors(const Tensor<type, 1>& box, const Tensor<type, 1>& anchor)
{
    type min_w = min(box(0), anchor(0));
    type min_h = min(box(1), anchor(1));

    type intersection_area = min_w * min_h;
    type box_area = box(1) * box(0);
    type anchor_area = anchor(1) * anchor(0);

    type union_area = box_area + anchor_area - intersection_area;

    type iou = intersection_area / union_area;

    return iou;
}

Tensor<type, 1> compute_distance(const Tensor<type, 1>& box, const vector<Tensor<type, 1>>& anchors)
{
    const Index num_anchor = anchors.size();

    Tensor<type, 1> distances(num_anchor);
    for(Index i = 0; i < num_anchor; i++)
    {
        distances(i) = 1 - calculate_intersection_over_union_anchors(box, anchors[i]);
    }
    return distances;
}

Tensor<Index, 1> assign_boxes_to_anchors(const Tensor<type, 2>& boxes, const vector<Tensor<type, 1>>& anchors)
{
    const Index num_boxes = boxes.dimension(0);
    Tensor<Index, 1> assignments(num_boxes);

    for(Index i = 0; i < num_boxes; i++)
    {
        Tensor<type, 1> box(2);
        box(0) = boxes(i, 0);
        box(1) = boxes(i, 1);

        Tensor<type, 1> distances = compute_distance(box, anchors);

        type min_distance = 1;
        Index best_anchor = 0;

        for(Index j = 0; j < distances.size(); j++)
        {
            if(distances(j) < min_distance)
            {
                min_distance = distances(j);
                best_anchor = j;
            }
        }
        assignments(i) = best_anchor;
    }
    return assignments;
}

vector<Tensor<type, 1>> update_anchors(const Tensor<type, 2>& boxes, const Tensor<Index, 1>& assignments, const Index& k)
{
    vector<Tensor<type, 1>> new_anchors(k);

    for(Index i = 0; i < k; i++)
    {
        new_anchors[i].resize(2);
    }

    vector<Index> counts(k, 0);

    for (Index i = 0; i < boxes.dimension(0); i++)
    {
        new_anchors[assignments(i)](0) += boxes(i, 0);
        new_anchors[assignments(i)](1) += boxes(i, 1);
        counts[assignments(i)]++;
    }

    for (Index j = 0; j < k; j++)
    {
        if (counts[j] > 0)
        {
            new_anchors[j](0) /= counts[j];
            new_anchors[j](1) /= counts[j];
        }
    }

    return new_anchors;
}

vector<Tensor<type, 1>> calculate_anchors(const vector<Tensor<type, 2>>& labels, const Index& k, const Index& iterations , const Index& seed)
{

    Tensor<type, 2> boxes = extract_boxes_width_height_data(labels);
    mt19937 urng(seed);
    uniform_int_distribution<> dist(0, boxes.dimension(0) - 1);


    vector<Tensor<type, 1>> anchors;
    anchors.resize(k);


    for(Index i = 0; i < k; i++)
    {
        anchors[i].resize(2);
        anchors[i](0) = boxes(dist(urng), 0);
        anchors[i](1) = boxes(dist(urng), 1);
    }

    for(Index iter = 0; iter < iterations; iter++)
    {
        Tensor<Index, 1> assignments = assign_boxes_to_anchors(boxes, anchors);

        vector<Tensor<type, 1>> new_anchors = update_anchors(boxes, assignments, k);

        bool converged = true;

        for(Index j = 0; j < k; j++)
        {
            if(new_anchors[j](0) != anchors[j](0) || new_anchors[j](1) != anchors[j](1))
            {
                converged = false;
                break;
            }
        }
        if(converged) break;


        anchors = new_anchors;

    }
    return anchors;
}



Tensor<type, 3> convert_to_YOLO_grid_data(const Tensor<type, 2>& labels, const vector<Tensor<type, 1>>& anchors, const Index& S, const Index& B, const Index& C)
{
    Tensor<type, 3> target(S, S, B * (5 + C));
    target.setZero();

    Tensor<Index, 1> box_assignment = assign_boxes_to_anchors(labels, anchors);

    #pragma omp parallel for
    for(Index j = 0; j < labels.dimension(0); j++)
    {
        Index cell_x = static_cast<Index>(floor(labels(j,1) * S));
        Index cell_y = static_cast<Index>(floor(labels(j,2) * S));

        type relative_x = labels(j,1) * S - cell_x;
        type relative_y = labels(j,2) * S - cell_y;

        target(cell_x, cell_y, 0 + box_assignment(j) * (5 + C)) = relative_x;
        target(cell_x, cell_y, 1 + box_assignment(j) * (5 + C)) = relative_y;
        target(cell_x, cell_y, 2 + box_assignment(j) * (5 + C)) = labels(j, 3);
        target(cell_x, cell_y, 3 + box_assignment(j) * (5 + C)) = labels(j, 4);
        target(cell_x, cell_y, 4 + box_assignment(j) * (5 + C)) = 1.0;
        target(cell_x, cell_y, 5 + static_cast<Index>(labels(j,0)) + box_assignment(j) * (5 + C)) = 1.0;
    }

    return target;
}



}
