//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   B L A N K   A P P L I C A T I O N
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

// System includes

//#include <stdio.h>
//#include <cstring>
#include <iostream>
//#include <fstream>
//#include <sstream>
#include <string>
//#include <ctime>
//#include <chrono>
//#include <algorithm>
//#include <execution>
#include <cstdlib>
//#include <batch.h>

// OpenNN includes

#include "../opennn/opennn.h"

using namespace std;
using namespace opennn;
using namespace std::chrono;
using namespace Eigen;

//namespace fs = std::filesystem;

struct YOLO_bounding_box
{
    Index class_id;
    type x_center;
    type y_center;
    type width;
    type height;
};

struct YOLO_anchor_box
{
    type width;
    type height;

    YOLO_anchor_box() : width(0), height(0){}
    YOLO_anchor_box(type w, type h) : width(w), height(h){}
};

Tensor<type, 2> read_bounding_boxes(const string& label_path);

vector<string> read_classes(const string& classes_path);

Tensor<type, 3> scale_image(const Tensor<type, 3>& image, const Index& new_height, const Index& new_width);

Tensor<type, 3> apply_zero_padding(const Tensor<type, 3>& image, const Index& target_height, const Index& target_width);

Tensor<type, 3> resize_image_416x416(const Tensor<type, 3>& image);

Tensor<type, 3> resize_image_416x416(const Tensor<type, 3>& image, Tensor<type, 1>& offsets);

Tensor<type, 3> normalize_tensor(const Tensor<type, 3> &image, const bool &is_normal);

Tensor<YOLO_anchor_box, 1> extract_boxes_width_height_data(const vector<Tensor<type, 2>>& labels);

type calculate_intersection_over_union(const Tensor<type, 1>& box_1, const Tensor<type, 1>& box_2);

type calculate_intersection_over_union_anchors(const YOLO_anchor_box& box, const YOLO_anchor_box& anchor);

Tensor<type, 1> compute_distance(const YOLO_anchor_box& box, const vector<YOLO_anchor_box>& anchors);

Tensor<Index, 1> assign_boxes_to_anchors(const Tensor<YOLO_anchor_box, 1>& boxes, const vector<YOLO_anchor_box>& anchors);

vector<YOLO_anchor_box> update_anchors(const Tensor<YOLO_anchor_box, 1>& boxes, const Tensor<Index, 1>& assignments, const Index& k);

vector<YOLO_anchor_box> calculate_anchors(const vector<Tensor<type, 2>>& labels, const Index& k, const Index& iterations = 1000, const Index& seed = 42);

class YOLODataset : public DataSet  //Tengo que ver cómo hago lo del tensor data (si es que puedo meterlo en un Tensor<type, 2> y no necesito tenerlo como ya lo tengo guardado)
{
public:

    vector<Tensor<type, 1>> offsets;
    vector<Tensor<type, 3>> images;
    vector<Tensor<type, 2>> raw_labels;
    vector<Tensor<type, 2>> labels;
    vector<string> labels_files;
    vector<string> images_files;
    vector<string> classes;
    Tensor<type, 2> anchors;

    Index anchor_number = 5;
    // int yolo_scale = 416;         //Lo pondré si cambio el código de resize_image_416x416 para que me haga resize a un tamaño cuadrado variable

   YOLODataset(const string& images_directory, const string& labels_directory) //Existe la posibilidad de poner todos los datos en la misma carpeta pero creo que tardaría más
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
        offsets.resize(images_files.size());

        #pragma omp parallel for
        for(size_t i = 0; i < images_files.size(); i++)
        {
            Tensor<type, 1> image_offsets;

            images[i] = normalize_tensor(resize_image_416x416(read_bmp_image(images_files[i]).cast<type>(), image_offsets), false);
            offsets[i] = image_offsets;
            // for(Index j = 0; j < yolo_scale * yolo_scale * 3; j++)
            // {
            //     data(i,j) = images[i](j);
            // }
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
                modified_label(j, 2) = 1 - (y_center + offset_y) / 416;         //I do the 1 - (...) because the y coord. is given reversed (we are working that y = 0 is the top part of the image and the label is given as if it started from the bottom part)
                modified_label(j, 3) = width / 416;
                modified_label(j, 4) = height / 416;
            }
            labels[i] = modified_label;
        }

        //cout<<images.size()<<endl<<labels.size()<<endl;

        if (images.size() != labels.size()) {
            cerr << "Images and labels file number do not match!" << endl;
            //throw runtime_error("Images and labels file number do not match!");
        }

        vector<YOLO_anchor_box> anchor_boxes = calculate_anchors(labels, anchor_number);
        anchors.resize(anchor_number, 2);

        for (Index i = 0; i < anchor_number; ++i) {
            anchors(i, 0) = anchor_boxes[i].width;
            anchors(i, 1) = anchor_boxes[i].height;
        }


        model_type = opennn::DataSet::ModelType::ObjectDetection;
    }

    YOLODataset(const string& images_directory)
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

        model_type = opennn::DataSet::ModelType::ObjectDetection;
    }

    size_t size() const
    {
        return images.size();
    }

    Tensor<type, 3> getImage(size_t index) const
    {
        return images[index];
    }

    Tensor<type, 2> getLabel(size_t index) const
    {
        return labels[index];
    }

    Tensor<type, 2> getAnchors() const
    {
        return anchors;
    }

    string getClass(size_t index) const
    {
        return classes[index];
    }

    void rotate_90_degrees(Tensor<type, 3>& image, Tensor<type, 2>& labels)
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

    void flip_image_horizontally(Tensor<type, 3>& image, Tensor<type, 2>& labels)
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

    void adjust_brightness_contrast(Tensor<type, 3>& image, const type& brightness_factor, const type& contrast_factor)
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

    void apply_data_augmentation(Tensor<type, 3>& image, Tensor<type, 2>& labels)
    {
        random_device rd;
        mt19937 gen(rd());
        uniform_real_distribution<> dist(0.0, 1.0);

        if(dist(gen) > 0.5) flip_image_horizontally(image, labels);
        if(dist(gen) > 0.5) rotate_90_degrees(image, labels);
        if(dist(gen) > 0.5) adjust_brightness_contrast(image, dist(gen) * 0.1 + 0.55, dist(gen) * 0.2 + 0.7);
    }

};


//falta meter una función que extraiga los batches

/*
NeuralNetwork set_YOLO_network(const Index& input_height, const Index& input_width, const Index& input_channels);
*/


Tensor<type, 3> convert_to_YOLO_grid_data(const Tensor<type, 2>& labels, const Index& S, const Index& B, const Index& C);

type loss_function(const Tensor<type, 2>& labels, const vector<YOLO_anchor_box> anchors, Tensor<type, 3> network_output);




Tensor<type, 2> non_maximum_suppression(const vector<Tensor<type, 1>>& input_bounding_boxes, const type& overlap_treshold, const Index& classes_number);





Tensor<type, 3> draw_boxes(const Tensor<type, 3>& image, const Tensor<type, 2>& final_boxes);

void save_tensor_as_BMP(const Tensor<type, 3>& tensor, const string& filename);






int main()
{
    try
    {
        // NeuralNetwork yolo_network = set_YOLO_network(416, 416, 3);


        //cout<<yolo_network.get_output_dimensions()[0]<<", "<<yolo_network.get_output_dimensions()[1]<<endl;

        //read_classes("/Users/artelnics/Desktop/Labels/Classes.names");

        YOLODataset train_dataset("/Users/artelnics/Desktop/Testing_dataset/VOCdevkit/VOC2007/BMPImages_debug","/Users/artelnics/Desktop/Testing_dataset/VOCdevkit/VOC2007/Labels_debug");     //PARA DEBUGGEAR

        // YOLODataset train_dataset("/Users/artelnics/Desktop/Testing_dataset/VOCdevkit/VOC2007/BMPImages","/Users/artelnics/Desktop/Testing_dataset/VOCdevkit/VOC2007/Labels");

        // cout<<train_dataset.size()<<endl;

        train_dataset.set(train_dataset.size(), 416 * 416 * 3);
        //cout<<train_dataset.get_data()<<endl;

        // cout<<"========"<<endl;

        Tensor<Index, 1> training_indices = train_dataset.get_training_samples_indices();

        // cout<<"========"<<endl;

        // Tensor<Index, 1> validation_indices = train_dataset.get_selection_samples_indices();

        // cout<<"========"<<endl;

        // Tensor<Index, 1> testing_indices = train_dataset.get_testing_samples_indices();

        // cout<<"========"<<endl;

        // Tensor<Index, 2> training_batches = train_dataset.get_batches(training_indices, 200, true);
        // Tensor<Index, 2> validation_batches = train_dataset.get_batches(validation_indices, 200, true);
        // Tensor<Index, 2> testing_batches = train_dataset.get_batches(testing_indices, 200, true);

        cout<<training_indices.dimension(0)<<endl;

        Index image_number = 160;

        cout<<train_dataset.getClass(0)<<endl;

        // for(Index i = 0; i < training_indices.dimension(0); i++)
        for(Index i = 0; i< 4; i++)
        {
            string complete = "output_" + to_string(i) + ".bmp";

            string filename = "/Users/artelnics/Desktop/image_test/" + complete ;    //Emplear este proceso para guardar las imágenes con las bounding_boxes ya metidas en una carpeta

            /*cout<<train_dataset.offsets[image_number - 1 + i]<<endl*/;

            train_dataset.apply_data_augmentation(train_dataset.images[image_number - 1 + i], train_dataset.labels[image_number - 1 + i]);

            // cout<<train_dataset.labels[image_number - 1 + i]<<endl;

            // train_dataset.apply_data_augmentation(train_dataset.images[training_indices(i)], train_dataset.labels[training_indices(i)]);

            // save_tensor_as_BMP(draw_boxes(normalize_tensor(train_dataset.getImage(training_indices(i)), true), train_dataset.getLabel(training_indices(i))), filename);

            save_tensor_as_BMP(draw_boxes(normalize_tensor(train_dataset.getImage(image_number - 1 + i), true), train_dataset.getLabel(image_number - 1 + i)), filename);
        }

        cout<<"works properly"<<endl;

        return 0;
    }
    catch (const exception& e)
    {
        cerr << e.what() << endl;

        return 1;
    }
}







Tensor<type, 2> read_bounding_boxes(const string& label_path)
{

    ifstream infile(label_path);
    if (!infile.is_open()) {
        cerr << "Unable to open the labels file: " << label_path << endl;
        return Tensor<type, 2>(0, 5);                   // Returns an empty vector if unable to read the file
    }

    vector<YOLO_bounding_box> bounding_boxes;
    string line;

   // cout << "Reading labels file: " << label_path << endl;

    // Read each line of the labels file
    while (getline(infile, line))
    {
        stringstream ss(line);
        YOLO_bounding_box box;

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

    Index original_height = image.dimension(0);
    Index original_width = image.dimension(1);
    Index channels = image.dimension(2);


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

    Index original_height = image.dimension(0);
    Index original_width = image.dimension(1);
    Index channels = image.dimension(2);


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
        Index original_height = image.dimension(0);
        Index original_width = image.dimension(1);
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
        Index original_height = image.dimension(0);
        Index original_width = image.dimension(1);
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


Tensor<YOLO_anchor_box, 1> extract_boxes_width_height_data(const vector<Tensor<type, 2>>& labels)
{
    vector<YOLO_anchor_box> box_dimensions;
    Index boxes_number = 0;
    Index image_size = 416;

    for(size_t i = 0; i < labels.size(); i++)
    {
        for(Index j = 0; j < labels[i].dimension(0); j++)
        {

            box_dimensions.emplace_back(labels[i](j, 3) * image_size, labels[i](j, 4) * image_size);
            //box_dimensions[boxes_number].height = data.labels[i](j, 4) * image_size;
            boxes_number++;
        }
    }

    Tensor<YOLO_anchor_box, 1> tensor_box_dimensions(boxes_number);

    for(Index k = 0; k < boxes_number; k++)
    {
        tensor_box_dimensions(k) = box_dimensions[k];
        //cout<<tensor_box_dimensions(k).width<<", "<<tensor_box_dimensions(k).height<<endl;
    }

    return tensor_box_dimensions;

}

type calculate_intersection_over_union(const Tensor<type, 1>& box_1, const Tensor<type, 1>& box_2)
{
    type x_left = max(box_1(0) - box_1(2) / 2, box_2(0) - box_2(2) / 2);
    type y_top = max(box_1(1) - box_1(3) / 2, box_2(1) - box_2(3) / 2);
    type x_right = min(box_1(0) + box_1(2) / 2, box_2(0) + box_2(2) / 2);
    type y_bottom = min(box_1(1) + box_1(3) / 2, box_2(1) + box_2(3) / 2);


    type intersection_area = max(0.0f, x_right - x_left) * max(0.0f, y_bottom - y_top);

    type box_1_area = box_1(2) * box_1(3);
    type box_2_area = box_2(2) * box_2(3);

    type union_area = box_1_area + box_2_area - intersection_area;

    return intersection_area / union_area;
}

type calculate_intersection_over_union_anchors(const YOLO_anchor_box& box, const YOLO_anchor_box& anchor)
{
    type min_w = min(box.width, anchor.width);
    type min_h = min(box.height, anchor.height);

    type intersection_area = min_w * min_h;
    type box_area = box.height * box.width;
    type anchor_area = anchor.height * anchor.width;

    type union_area = box_area + anchor_area - intersection_area;

    type iou = intersection_area / union_area;

    return iou;
}

Tensor<type, 1> compute_distance(const YOLO_anchor_box& box, const vector<YOLO_anchor_box>& anchors)
{
    Index num_anchor = anchors.size();

    Tensor<type, 1> distances(num_anchor);
    for(Index i = 0; i < num_anchor; i++)
    {
        distances(i) = 1 - calculate_intersection_over_union_anchors(box, anchors[i]);
    }
    return distances;
}

Tensor<Index, 1> assign_boxes_to_anchors(const Tensor<YOLO_anchor_box, 1>& boxes, const vector<YOLO_anchor_box>& anchors)
{
    Index num_boxes = boxes.dimension(0);
    Tensor<Index, 1> assignments(num_boxes);

    for(Index i = 0; i < num_boxes; i++)
    {
        YOLO_anchor_box box(boxes(i).width, boxes(i).height);

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

vector<YOLO_anchor_box> update_anchors(const Tensor<YOLO_anchor_box, 1>& boxes, const Tensor<Index, 1>& assignments, const Index& k)
{
    vector<YOLO_anchor_box> new_anchors(k, YOLO_anchor_box(0, 0));
    vector<Index> counts(k, 0);


    for (Index i = 0; i < boxes.dimension(0); i++)
    {
        new_anchors[assignments(i)].width += boxes(i).width;
        new_anchors[assignments(i)].height += boxes(i).height;
        counts[assignments(i)]++;
    }


    for (Index j = 0; j < k; j++)
    {
        if (counts[j] > 0)
        {
            new_anchors[j].width /= counts[j];
            new_anchors[j].height /= counts[j];
        }
    }

    return new_anchors;
}

vector<YOLO_anchor_box> calculate_anchors(const vector<Tensor<type, 2>>& labels, const Index& k, const Index& iterations , const Index& seed)
{

    Tensor<YOLO_anchor_box, 1> boxes = extract_boxes_width_height_data(labels);
    mt19937 urng(seed);
    uniform_int_distribution<> dist(0, boxes.dimension(0) - 1);


    vector<YOLO_anchor_box> anchors;

    for(Index i = 0; i < k; i++)
    {
        anchors.emplace_back(boxes(dist(urng)).width, boxes(dist(urng)).height);
    }

    for(Index iter = 0; iter < iterations; iter++)
    {
        Tensor<Index, 1> assignments = assign_boxes_to_anchors(boxes, anchors);

        vector<YOLO_anchor_box> new_anchors = update_anchors(boxes, assignments, k);

        bool converged = true;

        for(Index j = 0; j < k; j++)
        {
            if(new_anchors[j].width != anchors[j].width || new_anchors[j].height != anchors[j].height)
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



/*NeuralNetwork set_YOLO_network(const Index& input_height, const Index& input_width, const Index& input_channels)
{
    NeuralNetwork yolov2_net; //Seguiré la estructura de Darknet-19

    dimensions input_dimensions_1 = {input_height, input_width, input_channels};

    ConvolutionalLayer conv1(input_dimensions_1, {3, 3, input_channels, 32});

    conv1.set_activation_function(ConvolutionalLayer::ActivationFunction::RectifiedLinear);
    conv1.set_row_stride(1);
    conv1.set_column_stride(1);
    conv1.set_convolution_type(ConvolutionalLayer::ConvolutionType::Same);
    conv1.set_batch_normalization(true);

    dimensions pooling_kernel_dimensions = {2, 2};

    PoolingLayer pool1(conv1.get_output_dimensions(), pooling_kernel_dimensions);

    pool1.set_column_stride(2);
    pool1.set_row_stride(2);
    pool1.set_pooling_method(PoolingLayer::PoolingMethod::MaxPooling);

    ConvolutionalLayer conv2(pool1.get_output_dimensions(), {3, 3, 32, 64});

    conv2.set_activation_function(ConvolutionalLayer::ActivationFunction::RectifiedLinear);
    conv2.set_row_stride(1);
    conv2.set_column_stride(1);
    conv2.set_convolution_type(ConvolutionalLayer::ConvolutionType::Same);
    conv2.set_batch_normalization(true);

    PoolingLayer pool2(conv2.get_output_dimensions(), pooling_kernel_dimensions);

    pool2.set_column_stride(2);
    pool2.set_row_stride(2);
    pool2.set_pooling_method(PoolingLayer::PoolingMethod::MaxPooling);

    ConvolutionalLayer conv3(pool2.get_output_dimensions(), {3, 3, 64, 128});

    conv3.set_activation_function(ConvolutionalLayer::ActivationFunction::RectifiedLinear);
    conv3.set_row_stride(1);
    conv3.set_column_stride(1);
    conv3.set_convolution_type(ConvolutionalLayer::ConvolutionType::Same);
    conv3.set_batch_normalization(true);

    ConvolutionalLayer conv4(conv3.get_output_dimensions(), {1, 1, 128, 64});

    conv4.set_activation_function(ConvolutionalLayer::ActivationFunction::RectifiedLinear);
    conv4.set_row_stride(1);
    conv4.set_column_stride(1);
    conv4.set_convolution_type(ConvolutionalLayer::ConvolutionType::Same);
    conv4.set_batch_normalization(true);

    ConvolutionalLayer conv5(conv4.get_output_dimensions(), {3, 3, 64, 128});

    conv5.set_activation_function(ConvolutionalLayer::ActivationFunction::RectifiedLinear);
    conv5.set_row_stride(1);
    conv5.set_column_stride(1);
    conv5.set_convolution_type(ConvolutionalLayer::ConvolutionType::Same);
    conv5.set_batch_normalization(true);

    PoolingLayer pool3(conv5.get_output_dimensions(), pooling_kernel_dimensions);

    pool3.set_column_stride(2);
    pool3.set_row_stride(2);
    pool3.set_pooling_method(PoolingLayer::PoolingMethod::MaxPooling);

    ConvolutionalLayer conv6(pool3.get_output_dimensions(), {3, 3, 128, 256});

    conv6.set_activation_function(ConvolutionalLayer::ActivationFunction::RectifiedLinear);
    conv6.set_row_stride(1);
    conv6.set_column_stride(1);
    conv6.set_convolution_type(ConvolutionalLayer::ConvolutionType::Same);
    conv6.set_batch_normalization(true);

    ConvolutionalLayer conv7(conv6.get_output_dimensions(), {1, 1, 256, 128});

    conv7.set_activation_function(ConvolutionalLayer::ActivationFunction::RectifiedLinear);
    conv7.set_row_stride(1);
    conv7.set_column_stride(1);
    conv7.set_convolution_type(ConvolutionalLayer::ConvolutionType::Same);
    conv7.set_batch_normalization(true);

    ConvolutionalLayer conv8(conv7.get_output_dimensions(), {3, 3, 128, 256});

    conv8.set_activation_function(ConvolutionalLayer::ActivationFunction::RectifiedLinear);
    conv8.set_row_stride(1);
    conv8.set_column_stride(1);
    conv8.set_convolution_type(ConvolutionalLayer::ConvolutionType::Same);
    conv8.set_batch_normalization(true);

    PoolingLayer pool4(conv8.get_output_dimensions(), pooling_kernel_dimensions);

    pool4.set_column_stride(2);
    pool4.set_row_stride(2);
    pool4.set_pooling_method(PoolingLayer::PoolingMethod::MaxPooling);

    ConvolutionalLayer conv9(pool4.get_output_dimensions(), {3, 3, 256, 512});

    conv9.set_activation_function(ConvolutionalLayer::ActivationFunction::RectifiedLinear);
    conv9.set_row_stride(1);
    conv9.set_column_stride(1);
    conv9.set_convolution_type(ConvolutionalLayer::ConvolutionType::Same);
    conv9.set_batch_normalization(true);

    ConvolutionalLayer conv10(conv9.get_output_dimensions(), {1, 1, 512, 256});

    conv10.set_activation_function(ConvolutionalLayer::ActivationFunction::RectifiedLinear);
    conv10.set_row_stride(1);
    conv10.set_column_stride(1);
    conv10.set_convolution_type(ConvolutionalLayer::ConvolutionType::Same);
    conv10.set_batch_normalization(true);

    ConvolutionalLayer conv11(conv10.get_output_dimensions(), {3, 3, 256, 512});

    conv11.set_activation_function(ConvolutionalLayer::ActivationFunction::RectifiedLinear);
    conv11.set_row_stride(1);
    conv11.set_column_stride(1);
    conv11.set_convolution_type(ConvolutionalLayer::ConvolutionType::Same);
    conv11.set_batch_normalization(true);

    ConvolutionalLayer conv12(conv11.get_output_dimensions(), {1, 1, 512, 256});

    conv12.set_activation_function(ConvolutionalLayer::ActivationFunction::RectifiedLinear);
    conv12.set_row_stride(1);
    conv12.set_column_stride(1);
    conv12.set_convolution_type(ConvolutionalLayer::ConvolutionType::Same);
    conv12.set_batch_normalization(true);

    ConvolutionalLayer conv13(conv12.get_output_dimensions(), {3, 3, 256, 512});

    conv13.set_activation_function(ConvolutionalLayer::ActivationFunction::RectifiedLinear);
    conv13.set_row_stride(1);
    conv13.set_column_stride(1);
    conv13.set_convolution_type(ConvolutionalLayer::ConvolutionType::Same);
    conv13.set_batch_normalization(true);

    PoolingLayer pool5(conv13.get_output_dimensions(), pooling_kernel_dimensions);

    pool5.set_column_stride(2);
    pool5.set_row_stride(2);
    pool5.set_pooling_method(PoolingLayer::PoolingMethod::MaxPooling);

    ConvolutionalLayer conv14(pool5.get_output_dimensions(), {3, 3, 512, 1024});

    conv14.set_activation_function(ConvolutionalLayer::ActivationFunction::RectifiedLinear);
    conv14.set_row_stride(1);
    conv14.set_column_stride(1);
    conv14.set_convolution_type(ConvolutionalLayer::ConvolutionType::Same);
    conv14.set_batch_normalization(true);

    ConvolutionalLayer conv15(conv14.get_output_dimensions(), {1, 1, 1024, 512});

    conv15.set_activation_function(ConvolutionalLayer::ActivationFunction::RectifiedLinear);
    conv15.set_row_stride(1);
    conv15.set_column_stride(1);
    conv15.set_convolution_type(ConvolutionalLayer::ConvolutionType::Same);
    conv15.set_batch_normalization(true);

    ConvolutionalLayer conv16(conv15.get_output_dimensions(), {3, 3, 512, 1024});

    conv16.set_activation_function(ConvolutionalLayer::ActivationFunction::RectifiedLinear);
    conv16.set_row_stride(1);
    conv16.set_column_stride(1);
    conv16.set_convolution_type(ConvolutionalLayer::ConvolutionType::Same);
    conv16.set_batch_normalization(true);

    ConvolutionalLayer conv17(conv16.get_output_dimensions(), {1, 1, 1024, 512});

    conv17.set_activation_function(ConvolutionalLayer::ActivationFunction::RectifiedLinear);
    conv17.set_row_stride(1);
    conv17.set_column_stride(1);
    conv17.set_convolution_type(ConvolutionalLayer::ConvolutionType::Same);
    conv17.set_batch_normalization(true);

    ConvolutionalLayer conv18(conv17.get_output_dimensions(), {3, 3, 512, 1024});

    conv18.set_activation_function(ConvolutionalLayer::ActivationFunction::RectifiedLinear);
    conv18.set_row_stride(1);
    conv18.set_column_stride(1);
    conv18.set_convolution_type(ConvolutionalLayer::ConvolutionType::Same);
    conv18.set_batch_normalization(true);

    ConvolutionalLayer conv19(conv18.get_output_dimensions(), {1, 1, 1024, 125});

    conv19.set_activation_function(ConvolutionalLayer::ActivationFunction::Linear);
    conv19.set_row_stride(1);
    conv19.set_column_stride(1);
    conv19.set_convolution_type(ConvolutionalLayer::ConvolutionType::Same);
    conv19.set_batch_normalization(true);

    cout<<conv19.get_output_dimensions()[0]<<", "<<conv19.get_output_dimensions()[1]<<", "<<conv19.get_output_dimensions()[2]<<endl;

    // FlattenLayer flat(conv19.get_output_dimensions());

    // cout<<flat.get_output_dimensions()[0]<<endl;

    // ProbabilisticLayer prob(conv19.get_output_dimensions(), {20});

    // cout<<prob.get_output_dimensions()[0]<<endl;

    yolov2_net.add_layer(&conv1, "Convolutional");
    yolov2_net.add_layer(&pool1, "Maxpool");
    yolov2_net.add_layer(&conv2, "Convolutional");
    yolov2_net.add_layer(&pool2, "Maxpool");
    yolov2_net.add_layer(&conv3, "Convolutional");
    yolov2_net.add_layer(&conv4, "Convolutional");
    yolov2_net.add_layer(&conv5, "Convolutional");
    yolov2_net.add_layer(&pool3, "Maxpool");
    yolov2_net.add_layer(&conv6, "Convolutional");
    yolov2_net.add_layer(&conv7, "Convolutional");
    yolov2_net.add_layer(&conv8, "Convolutional");
    yolov2_net.add_layer(&pool4, "Maxpool");
    yolov2_net.add_layer(&conv9, "Convolutional");
    yolov2_net.add_layer(&conv10, "Convolutional");
    yolov2_net.add_layer(&conv11, "Convolutional");
    yolov2_net.add_layer(&conv12, "Convolutional");
    yolov2_net.add_layer(&conv13, "Convolutional");
    yolov2_net.add_layer(&pool5, "Maxpool");
    yolov2_net.add_layer(&conv14, "Convolutional");
    yolov2_net.add_layer(&conv15, "Convolutional");
    yolov2_net.add_layer(&conv16, "Convolutional");
    yolov2_net.add_layer(&conv17, "Convolutional");
    yolov2_net.add_layer(&conv18, "Convolutional");

    yolov2_net.add_layer(&conv19, "Convolutional");
    // yolov2_net.add_layer(&flat, "Flatten");
    // yolov2_net.add_layer(&prob, "Softmax");

    return yolov2_net;
}
*/



Tensor<type, 3> convert_to_YOLO_grid_data(const Tensor<type, 2>& labels, const Index& S, const Index& B, const Index& C)
{
    Tensor<type, 3> output(S, S, B * (5 + C));

    //for(size_t i = 0; i < labels.size(); i++)
    //{
        for(Index j = 0; j < labels.dimension(0); j++)
        {
            Index cell_x = static_cast<Index>(floor(labels(j,1) * S));
            Index cell_y = static_cast<Index>(floor(labels(j,2) * S));

            type relative_x = labels(j,1) * S - cell_x;
            type relative_y = labels(j,2) * S - cell_y;

            for(Index b = 0; b < B; b++)
            {
                if(output(cell_x, cell_y, 2 + b * (5 + C)) == 0)
                {
                    output(cell_x, cell_y, 0 + b * (5 + C)) = relative_x;
                    output(cell_x, cell_y, 1 + b * (5 + C)) = relative_y;
                    output(cell_x, cell_y, 2 + b * (5 + C)) = labels(j, 3);
                    output(cell_x, cell_y, 3 + b * (5 + C)) = labels(j, 4);
                    output(cell_x, cell_y, 4 + b * (5 + C)) = 1.0;
                    output(cell_x, cell_y, 5 + static_cast<Index>(labels(j,0)) + b * (5 + C)) = 1.0;
                    break;
                }
                continue;
            }
        }

    //}
    return output;
}

type loss_function(const Tensor<type, 2>& labels, const vector<YOLO_anchor_box> anchors, Tensor<type, 3> network_output)
{
    Index S = network_output.dimension(0);
    Index B = anchors.size();
    Index C = (network_output.dimension(3) / anchors.size()) - 5;
    type total_loss = 0, coord_loss = 0, conf_loss_object = 0, conf_loss_noobject = 0, conf_loss = 0, class_loss = 0;
    type lambda_coord = 5.0;
    type lambda_noobject = 0.5;


    Tensor<type, 3> ground_truth = convert_to_YOLO_grid_data(labels, S, B, C);

    for(Index i = 0; i < S; i++)
    {
        for(Index j = 0; j < S; j++)
        {
            for(Index k = 0; k < B; k++)
            {

                if(ground_truth(i,j, k * (5 + C) + 4) == 1)
                {
                    coord_loss += pow(ground_truth(i, j, k * (5 + C) + 0) - network_output(i, j, k * (5 + C) + 0), 2) +
                                  pow(ground_truth(i, j, k * (5 + C) + 1) - network_output(i, j, k * (5 + C) + 1), 2) +
                                  pow(sqrt(ground_truth(i, j, k * (5 + C) + 2)) - sqrt(network_output(i, j, k * (5 + C) + 2)), 2) +
                                  pow(sqrt(ground_truth(i, j, k * (5 + C) + 3)) - sqrt(network_output(i, j, k * (5 + C) + 3)), 2);

                    conf_loss_object += pow(1 - network_output(i,j, k * (5 + C) + 4), 2);
                    for(Index c = 0; c < C; c++)
                    {
                        class_loss += ground_truth(i,j, k * (5 + C) + c) * log(network_output(i,j, k * (5 + C) + c));
                    }
                }
                else conf_loss_noobject = pow(network_output(i,j, k * (5 + C) + 4), 2);
            }

        }
        coord_loss = lambda_coord * coord_loss;
        class_loss = -class_loss;
        conf_loss = conf_loss_object + lambda_noobject * conf_loss_noobject;
    }

    total_loss = coord_loss + class_loss + conf_loss;

    return total_loss;
}

Tensor<type, 2> non_maximum_suppression(const vector<Tensor<type, 1>>& input_bounding_boxes, const type& overlap_treshold, const Index& classes_number) //Las input_bboxes son las bboxes que el modelo cree que son las definitivas y de las que hay que quitar las redundantes, es decir, no pasas todos los datos de la grid
{
    vector<Tensor<type, 1>> final_boxes;
    vector<vector<Tensor<type, 1>>> classified_boxes(classes_number);

    for(size_t i = 0; i < input_bounding_boxes.size(); i++)
    {
        for(Index j = 0; j < classes_number; j++)
        {
            if(final_boxes[i](5+j) != 0)
            {
                classified_boxes[j].push_back(final_boxes[i]);
                break;
            }
        }
    }


    for(Index k = 0; k < classes_number; k++)
    {
        if(classified_boxes[k].empty()) continue;

        sort(classified_boxes[k].begin(), classified_boxes[k].end(), [](const Tensor<type, 1>& a, const Tensor<type, 1>& b){
            return a(4) > b(4);
        });


        while(!classified_boxes[k].empty())
        {
            Tensor<type, 1> box = classified_boxes[k].front();
            classified_boxes[k].erase(classified_boxes[k].begin());

            for(auto it = classified_boxes[k].begin(); it != classified_boxes[k].end();)
            {
                if(calculate_intersection_over_union(box, *it) > overlap_treshold)
                {
                    it = classified_boxes[k].erase(it);
                }else
                {
                    it++;
                }
            }
            final_boxes.push_back(box);
        }
    }



    Tensor<type, 2> tensor_final_boxes;

    for(size_t box = 0; box < final_boxes.size(); box++)
    {
        for(Index element = 0; element < final_boxes[box].dimension(0); element++)
        {
            tensor_final_boxes(box, element) = final_boxes[box](element);
        }
    }

    return tensor_final_boxes;
}



Tensor<type, 3> draw_boxes(const Tensor<type, 3>& images, const Tensor<type, 2>& final_boxes)
{

    Tensor<type, 3> image = images;
    Index image_height = image.dimension(0);
    Index image_width = image.dimension(1);
    Index channels = image.dimension(2);

    if (channels != 3) {
        // std::cerr << "Error: Image must have 3 channels (RGB)." << std::endl;
        throw runtime_error("Error: Image must have 3 channels (RGB)");
    }

    for (Index i = 0; i < final_boxes.dimension(0); i++)
    {
        type x_center = final_boxes(i, 1);
        type y_center = final_boxes(i, 2);
        type width = final_boxes(i, 3);
        type height = final_boxes(i, 4);


        // type x_center = final_boxes[i](1);
        // type y_center = final_boxes[i](2);
        // type width = final_boxes[i](3);
        // type height = final_boxes[i](4);

        Index box_x_min = static_cast<Index>((x_center - width / 2) * image_width);
        Index box_y_min = static_cast<Index>((y_center - height / 2) * image_height);
        Index box_x_max = static_cast<Index>((x_center + width / 2) * image_width);
        Index box_y_max = static_cast<Index>((y_center + height / 2) * image_height);

        // Check that the coordinates are inside of the image
        box_x_min = max((Index)0, min(image_width - 1, box_x_min));
        box_y_min = max((Index)0, min(image_height - 1, box_y_min));
        box_x_max = max((Index)0, min(image_width - 1, box_x_max));
        box_y_max = max((Index)0, min(image_height - 1, box_y_max));

        // Set the color of the box borders (For example, red = (255, 0, 0))
        type red = 255, green = 0, blue = 255;  // red color for the box


        for (Index x = box_x_min; x <= box_x_max; ++x) {

            image(box_y_min, x, 2) = red;
            image(box_y_min, x, 1) = green;
            image(box_y_min, x, 0) = blue;

            image(box_y_max, x, 2) = red;
            image(box_y_max, x, 1) = green;
            image(box_y_max, x, 0) = blue;
        }


        for (Index y = box_y_min; y <= box_y_max; ++y) {

            image(y, box_x_min, 2) = red;
            image(y, box_x_min, 1) = green;
            image(y, box_x_min, 0) = blue;

            image(y, box_x_max, 2) = red;
            image(y, box_x_max, 1) = green;
            image(y, box_x_max, 0) = blue;
        }
    }
    return image;
}

void save_tensor_as_BMP(const Tensor<type, 3>& tensor, const string& filename)
{

    const Index height = tensor.dimension(0);
    const Index width = tensor.dimension(1);
    const Index channels = tensor.dimension(2);

    if (channels != 3) {
        std::cerr << " The tensor must have 3 channels to represent an RGB image." << std::endl;
        return;
    }

    // Definition of the size of the image and each row size (4 bytes alignment)
    const Index rowSize = (3 * width + 3) & (~3);  // Each row of the image must be aligned to 4 bytes
    const Index dataSize = rowSize * height;
    const Index fileSize = 54 + dataSize;  // Total size of the file (BMP header + image data)

    // Creation of the BMP header
    unsigned char bmpHeader[54] = {
        0x42, 0x4D,                      // File type 'BM'
        0, 0, 0, 0,                      // Total size of the file (will be updated later)
        0, 0,                            // Reserved
        0, 0,                            // Reserved
        54, 0, 0, 0,                     // Moving to the image data (54 bytes)
        40, 0, 0, 0,                     // Size of the DIB header (40 bytes)
        0, 0, 0, 0,                      // Image width (will be updated later)
        0, 0, 0, 0,                      // Image height (will be updated later)
        1, 0,                            // Planes (must be 1)
        24, 0,                           // Bits per pixel (24 for RGB)
        0, 0, 0, 0,                      // Compression (0 in order not to compress)
        0, 0, 0, 0,                      // Image size (can be 0 if you don't compress the file)
        0, 0, 0, 0,                      // Color number in the palette (0 for the maximum)
        0, 0, 0, 0                       // Important colors (0 for all of them)
    };

    // Update of specific fields on the header
    bmpHeader[2] = (unsigned char)(fileSize);
    bmpHeader[3] = (unsigned char)(fileSize >> 8);
    bmpHeader[4] = (unsigned char)(fileSize >> 16);
    bmpHeader[5] = (unsigned char)(fileSize >> 24);
    bmpHeader[18] = (unsigned char)(width);
    bmpHeader[19] = (unsigned char)(width >> 8);
    bmpHeader[20] = (unsigned char)(width >> 16);
    bmpHeader[21] = (unsigned char)(width >> 24);
    bmpHeader[22] = (unsigned char)(height);
    bmpHeader[23] = (unsigned char)(height >> 8);
    bmpHeader[24] = (unsigned char)(height >> 16);
    bmpHeader[25] = (unsigned char)(height >> 24);

    // Creation of the BMP file, making sure that it creates a new file
    std::ofstream file(filename, std::ios::out | std::ios::binary | std::ios::trunc);
    if (!file)
    {
        std::cerr << "Error while opening the file to save the image." << std::endl;
        return;
    }

    // Writing of the BMP header
    file.write(reinterpret_cast<char*>(bmpHeader), 54);

    // Writing of the image data (RGB format)
    #pragma omp parallel for
    for (Index h = 0; h < height; h++)
    {
        for (Index w = 0; w < width; w++)
        {
            file.put(tensor(h, w, 0));  // Red channel
            file.put(tensor(h, w, 1));  // Green channel
            file.put(tensor(h, w, 2));  // Blue channel
        }
        // Adding filling bytes if it's necessary to align 4 bytes per row
        for (Index pad = 0; pad < rowSize - (width * 3); ++pad) {
            file.put(0);
        }
    }

    file.close();
    std::cout << "Image saved as " << filename << std::endl;
}



// OpenNN: Open Neural Networks Library.
// Copyright (C) Artificial Intelligence Techniques SL.
//
// This library is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 2.1 of the License, or any later version.
//
// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.
// You should have received a copy of the GNU Lesser General Public
// License along with this library; if not, write to the Free Software
// Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
