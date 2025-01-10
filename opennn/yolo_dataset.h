#ifndef YOLO_DATASET_H
#define YOLO_DATASET_H

#include <string>
#include <chrono>
#include <fstream>

#include "config.h"
#include "data_set.h"
#include <filesystem>


namespace fs = std::filesystem;



namespace opennn
{

class YOLODataset : public DataSet
{
public:
    struct YOLO_bounding_box
    {
        Index class_id;
        type x_center;
        type y_center;
        type width;
        type height;
    };

    explicit YOLODataset(const string&);

    explicit YOLODataset(const string&, const string&);

    vector<Descriptives> scale_variables(const VariableUse&);

    size_t size() const;

    Tensor<type, 3> get_image(const Index&) const;

    Tensor<type, 2> get_label(const Index&) const;

    Tensor<type, 4> get_images() const;

    Tensor<type, 4> get_targets()const;

    vector<Tensor<type, 1>> get_anchors() const;

    string get_class(const Index&) const;

    Index get_grid_size() const
    {
        return grid_size;
    }

    void rotate_90_degrees(Tensor<type, 3>&, Tensor<type, 2>&);

    void flip_image_horizontally(Tensor<type, 3>&, Tensor<type, 2>&);

    void adjust_brightness_contrast(Tensor<type, 3>&, const type&, const type&);

    void apply_data_augmentation(Tensor<type, 3>&, Tensor<type, 2>&);



protected:
    vector<Tensor<type, 1>> offsets;    
    vector<Tensor<type, 2>> raw_labels;
    vector<Tensor<type, 2>> labels;
    vector<string> labels_files;
    vector<string> images_files;
    vector<Tensor<type, 1>> anchors;
    vector<string> classes;

    vector<Tensor<type, 3>> images;
    vector<Tensor<type, 3>> targets;

    Index grid_size = 13;
    Index anchor_number = 5;

    Tensor<type, 4> tensor_targets;
    Tensor<type, 4> tensor_images;

    // Index yolo_scale = 416;
};

Tensor<type, 2> read_bounding_boxes(const string&);

vector<string> read_classes(const string&);

Tensor<type, 3> scale_image(const Tensor<type, 3>&, const Index&, const Index&);

Tensor<type, 3> apply_zero_padding(const Tensor<type, 3>&, const Index&, const Index&);

Tensor<type, 3> resize_image_416x416(const Tensor<type, 3>&);

Tensor<type, 3> resize_image_416x416(const Tensor<type, 3>&, Tensor<type, 1>&);

Tensor<type, 3> normalize_tensor(const Tensor<type, 3>&, const bool&);

Tensor<type, 2> extract_boxes_width_height_data(const vector<Tensor<type, 2>>&);

type calculate_intersection_over_union(const Tensor<type, 1>&, const Tensor<type, 1>&);

type calculate_intersection_over_union_anchors(const Tensor<type, 1>&, const Tensor<type, 1>&);

Tensor<type, 1> compute_distance(const Tensor<type, 1>&, const vector<Tensor<type, 1>>&);

Tensor<Index, 1> assign_boxes_to_anchors(const Tensor<type, 2>&, const vector<Tensor<type, 1>>&);

vector<Tensor<type, 1>> update_anchors(const Tensor<type, 2>&, const Tensor<Index, 1>&, const Index&);

vector<Tensor<type, 1>> calculate_anchors(const vector<Tensor<type, 2>>&, const Index&, const Index& iterations = 1000, const Index& seed = 42);


Tensor<type, 3> convert_to_YOLO_grid_data(const Tensor<type, 2>&, const vector<Tensor<type, 1>>&, const Index&, const Index&, const Index&);
}

#endif // YOLO_DATASET_H
