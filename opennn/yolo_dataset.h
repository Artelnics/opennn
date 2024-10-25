#ifndef YOLO_DATASET_H
#define YOLO_DATASET_H

// System includes
#include <iostream>
#include <string>

// OpenNN includes
#include "data_set.h"


namespace opennn
{

struct YOLO_bounding_box;
struct YOLO_anchor_box;

Tensor<type, 2> read_bounding_boxes(const string&);

vector<string> read_classes(const string&);

Tensor<type, 3> scale_image(const Tensor<type, 3>&, const Index&, const Index&);

Tensor<type, 3> apply_zero_padding(const Tensor<type, 3>&, const Index&, const Index&);

Tensor<type, 3> resize_image_416x416(const Tensor<type, 3>&);

Tensor<type, 3> resize_image_416x416(const Tensor<type, 3>&, Tensor<type, 1>&);

Tensor<type, 3> normalize_tensor(const Tensor<type, 3>&, const bool&);

Tensor<YOLO_anchor_box, 1> extract_boxes_width_height_data(const vector<Tensor<type, 2>>&);

type calculate_intersection_over_union(const Tensor<type, 1>&, const Tensor<type, 1>&);

type calculate_intersection_over_union_anchors(const YOLO_anchor_box&, const YOLO_anchor_box&);

Tensor<type, 1> compute_distance(const YOLO_anchor_box&, const vector<YOLO_anchor_box>&);

Tensor<Index, 1> assign_boxes_to_anchors(const Tensor<YOLO_anchor_box, 1>&, const vector<YOLO_anchor_box>&);

vector<YOLO_anchor_box> update_anchors(const Tensor<YOLO_anchor_box, 1>&, const Tensor<Index, 1>&, const Index&);

vector<YOLO_anchor_box> calculate_anchors(const vector<Tensor<type, 2>>&, const Index&, const Index& iterations = 1000, const Index& seed = 42);

class YOLODataset : public DataSet
{
public:
    vector<Tensor<type, 1>> offsets;
    vector<Tensor<type, 3>> images;
    vector<Tensor<type, 2>> raw_labels;
    vector<Tensor<type, 2>> labels;
    vector<string> labels_files;
    vector<string> images_files;
    // vector<string> classes;
    Tensor<type, 2> anchors;

    Index anchor_number = 5;
    // int yolo_scale = 416;

    explicit YOLODataset(const string&, const string&);
    explicit YOLODataset(const string&);
};

}
#endif // YOLO_DATASET_H
