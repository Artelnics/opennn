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


// OpenNN includes

#include "../opennn/opennn.h"
#include "../opennn/data_set.h"
#include "yolo_dataset.h"
#include "neural_network.h"
#include "yolo_network.h"
#include "layer.h"
#include "detection_layer.h"


using namespace std;
using namespace opennn;
using namespace std::chrono;
using namespace Eigen;

//namespace fs = std::filesystem;




type loss_function(const Tensor<type, 4> &, const vector<Tensor<type, 1> > &, Tensor<type, 4> &);


vector<Tensor<type, 2>> non_maximum_suppression(const Tensor<type, 4>&, const type&, const Index&);


Tensor<type, 3> draw_boxes(const Tensor<type, 3>&, const Tensor<type, 2>&);

void save_tensor_as_BMP(const Tensor<type, 3>&, const string&);



int main()
{
    try
    {

        YOLODataset train_dataset("/Users/artelnics/Desktop/Testing_dataset/VOCdevkit/VOC2007/BMPImages_fixing","/Users/artelnics/Desktop/Testing_dataset/VOCdevkit/VOC2007/Labels_fixing");     //PARA COMPROBAR DONDE FALLA

        // YOLODataset train_dataset("/Users/artelnics/Desktop/Testing_dataset/VOCdevkit/VOC2007/BMPImages_debug","/Users/artelnics/Desktop/Testing_dataset/VOCdevkit/VOC2007/Labels_debug");     //PARA DEBUGGEAR

        // YOLODataset train_dataset("/Users/artelnics/Desktop/Testing_dataset/VOCdevkit/VOC2007/BMPImages","/Users/artelnics/Desktop/Testing_dataset/VOCdevkit/VOC2007/Labels");

        YoloNetwork yolo({416, 416, 3}, train_dataset.get_anchors());

        // cout<<train_dataset.get_anchors()[1]<<endl;
        // cout<<train_dataset.get_label(1)<<endl;

        // yolo.print();
        // yolo.calculate_outputs(train_dataset.get_images());

        // cout<<yolo.calculate_outputs(train_dataset.get_images())<<endl;



        // cout<<yolo.calculate_outputs(train_dataset.get_images()).dimensions()<<endl;

        // cout<<train_dataset.get_targets()<<endl;



        TrainingStrategy training_strategy(&yolo, &train_dataset);

        training_strategy.set_loss_method(TrainingStrategy::LossMethod::YOLO_ERROR);
        training_strategy.set_optimization_method(TrainingStrategy::OptimizationMethod::ADAPTIVE_MOMENT_ESTIMATION);
        training_strategy.get_loss_index()->set_regularization_method(LossIndex::RegularizationMethod::NoRegularization);
        training_strategy.get_adaptive_moment_estimation()->set_batch_samples_number(32);
        training_strategy.get_adaptive_moment_estimation()->set_maximum_epochs_number(3);
        training_strategy.set_display_period(1);


        training_strategy.perform_training();








        // // Softmax

        // Tensor<type, 2> test_data(2, 20);
        // test_data.setConstant(1);
        // for(Index j = 0; j < test_data.dimension(0); j++)
        //     for(Index i = 0 ; i < 15; i++)
        //         test_data(j, i) = i * (j + 1);

        // cout<<test_data<<endl<<endl;

        // const Eigen::array<Index, 1> dimension{{1}};

        // const Eigen::array<Index, 2> range_2{ {2, 1} };
        // const Eigen::array<Index, 2> expand_softmax_dim{ {1, 5}};
        // // Tensor<type, 1> softmax_results(20);
        // for(Index i = 0; i < 2; i++)
        // {
        //     Tensor<type, 2> class_probabilities =  test_data.slice(Eigen::array<Index, 2>{0, i * 10 + 5},
        //                                                        Eigen::array<Index, 2>{2, 5});

        //     // test_data/*.device(*thread_pool_device) */= test_data - test_data.maximum(dimension)
        //     //                                                              .eval()
        //     //                                                              .reshape(range_2)
        //     //                                                              .broadcast(expand_softmax_dim);
        //     // test_data/*.device(*thread_pool_device)*/ = test_data.exp();
        //     // test_data/*.device(*thread_pool_device)*/ = test_data / test_data.sum(dimension)
        //     //                                                              .eval()
        //     //                                                              .reshape(range_2)
        //     //                                                              .broadcast(expand_softmax_dim);

        //     class_probabilities = class_probabilities - class_probabilities.maximum(dimension)
        //                                                                    .eval()
        //                                                                    .reshape(range_2)
        //                                                                    .broadcast(expand_softmax_dim);
        //     class_probabilities = class_probabilities.exp();
        //     class_probabilities = class_probabilities / class_probabilities.sum(dimension)
        //                                                                    .eval()
        //                                                                    .reshape(range_2)
        //                                                                    .broadcast(expand_softmax_dim);
        //     test_data.slice(Eigen::array<Index, 2>{0, i * 10 + 5},
        //                     Eigen::array<Index, 2>{2, 5}) = class_probabilities;

        // }
        // cout<<test_data<<endl<<endl;















/*
        Tensor<Index, 1> training_indices = train_dataset.get_samples_uses_tensor();

        cout<<training_indices<<endl;

        cout<<"========"<<endl;

        Tensor<Index, 1> validation_indices = train_dataset.get_selection_samples_indices();

        cout<<"========"<<endl;

        Tensor<Index, 1> testing_indices = train_dataset.get_testing_samples_indices();

        cout<<"========"<<endl;

        Tensor<Index, 2> training_batches = train_dataset.get_batches(training_indices, 32, true);
        Tensor<Index, 2> validation_batches = train_dataset.get_batches(validation_indices, 32, true);
        Tensor<Index, 2> testing_batches = train_dataset.get_batches(testing_indices, 32, true);
*/



/*
        cout<<training_indices.dimension(0)<<endl;

        Index image_number = 160;

        cout<<train_dataset.getClass(0)<<endl;

        // for(Index i = 0; i < training_indices.dimension(0); i++)
        for(Index i = 0; i< 4; i++)
        {
            string complete = "output_" + to_string(i) + ".bmp";

            string filename = "/Users/artelnics/Desktop/image_test/" + complete ;    //Emplear este proceso para guardar las imágenes con las bounding_boxes ya metidas en una carpeta

            cout<<train_dataset.offsets[image_number - 1 + i]<<endl;

            train_dataset.apply_data_augmentation(train_dataset.images[image_number - 1 + i], train_dataset.labels[image_number - 1 + i]);

            // cout<<train_dataset.labels[image_number - 1 + i]<<endl;

            // train_dataset.apply_data_augmentation(train_dataset.images[training_indices(i)], train_dataset.labels[training_indices(i)]);

            // save_tensor_as_BMP(draw_boxes(normalize_tensor(train_dataset.getImage(training_indices(i)), true), train_dataset.getLabel(training_indices(i))), filename);

            save_tensor_as_BMP(draw_boxes(normalize_tensor(train_dataset.getImage(image_number - 1 + i), true), train_dataset.getLabel(image_number - 1 + i)), filename);
        }
*/

        cout<<"works properly"<<endl;

        return 0;
    }
    catch (const exception& e)
    {
        cerr << e.what() << endl;

        return 1;
    }
}

// @TODO check if the loss is calculated for each image or for each batch

type loss_function(const Tensor<type, 4>& targets, const vector<Tensor<type, 1>>& anchors, Tensor<type, 4>& outputs)
{
    const Index batch_size = outputs.dimension(0);
    const Index grid_size = outputs.dimension(1);
    const Index boxes_per_cell = anchors.size();
    const Index classes_number = (outputs.dimension(3) / anchors.size()) - 5;
    const Index box_data_size = 5 + classes_number;
    type coordinate_loss = 0, confidence_loss_object = 0, confidence_loss_noobject = 0, confidence_loss = 0, class_loss = 0;
    const type lambda_coord = 5.0;
    const type lambda_noobject = 0.5;


    for(Index i = 0; i < batch_size; i++)
    {
        for(Index j = 0; j < grid_size; j++)
        {
            for(Index k = 0; k < grid_size; k++)
            {
                for(Index l = 0; l < boxes_per_cell; l++)
                {
                    if(targets(i,j, k, l * box_data_size + 4) == 1)
                    {


                        coordinate_loss += pow(targets(i, j, k, l * box_data_size + 0) - outputs(i, j, k, l * box_data_size + 0), 2)
                                    + pow(targets(i, j, k, l * box_data_size + 1) - outputs(i, j, k, l * box_data_size + 1), 2) +
                                      pow(sqrt(targets(i, j, k, l * box_data_size + 2)) - sqrt(outputs(i, j, k, l * box_data_size + 2) * anchors[l](0)), 2) +
                                      pow(sqrt(targets(i, j, k, l * box_data_size + 3)) - sqrt(outputs(i, j, k, l * box_data_size + 3) * anchors[l](1)), 2);

                        confidence_loss_object += pow(1 - outputs(i, j, k, l * box_data_size + 4), 2);

                        for(Index c = 0; c < classes_number; c++)
                            class_loss += targets(i,j, k, l * box_data_size + (5 + c)) * log(outputs(i,j, k, l * box_data_size + (5 + c)));
                    }
                    else

                        confidence_loss_noobject += pow(outputs(i, j, k, l * box_data_size + 4), 2);
                }

            }
        }
    }

    coordinate_loss = lambda_coord * coordinate_loss;
    // class_loss = -class_loss;
    confidence_loss = confidence_loss_object + lambda_noobject * confidence_loss_noobject;

    // total_loss = coord_loss + class_loss + conf_loss;
    const type total_loss = coordinate_loss - class_loss + confidence_loss;

    return total_loss;
}


vector<Tensor<type, 2>> non_maximum_suppression(const Tensor<type, 4>& network_output, const type& overlap_threshold, const Index& classes_number)
{
    const Index batch_size = network_output.dimension(0);

    // @TODO check if I need to pre-convert the boxes format to a 6-data format (x,y,w,h,p,c) where p = object_prob * class_prob and c = class (bc if I do it, the output of the NMS will be the standard one)

    vector<vector<Tensor<type, 1>>> input_bounding_boxes(batch_size);
    vector<vector<Tensor<type, 1>>> final_boxes(batch_size);
    vector<vector<vector<Tensor<type, 1>>>> classified_boxes(batch_size);
    Tensor<type, 1> box(25);

    vector<Tensor<type, 2>> tensor_final_boxes(batch_size);

    for(Index i = 0; i < batch_size; i++)
    {
        classified_boxes[i].resize(classes_number);

        for(Index j = 0; j < network_output.dimension(1); j++)
        {
            for(Index k = 0; k < network_output.dimension(2); k++)
            {
                if(network_output(i,j,k,4) > 0.5)
                {
                    for(Index l = 0; l < 25; l++)
                        box(l) = network_output(i,j,k,l);

                    input_bounding_boxes[i].push_back(box);
                }

                if(network_output(i,j,k,29) > 0.5)
                {
                    for(Index l = 0; l < 25; l++)
                        box(l) = network_output(i,j,k,25 + l);

                    input_bounding_boxes[i].push_back(box);
                }

                if (network_output(i,j,k, 54) > 0.5)
                {
                    for(Index l = 0; l < 25; l++)
                        box(l) = network_output(i,j,k,50 + l);

                    input_bounding_boxes[i].push_back(box);
                }

                if (network_output(i,j,k, 79) > 0.5)
                {
                    for(Index l = 0; l < 25; l++)
                        box(l) = network_output(i,j,k,75 + l);

                    input_bounding_boxes[i].push_back(box);
                }

                if (network_output(i,j,k, 104) > 0.5)
                {
                    for(Index l = 0; l < 25; l++)
                        box(l) = network_output(i,j,k,100 + l);

                    input_bounding_boxes[i].push_back(box);
                }
            }
        }

        for(size_t j = 0; j < input_bounding_boxes[i].size(); j++)
        {
            for(Index k = 0; k < classes_number; k++)
            {
                if(input_bounding_boxes[i][j](5+k) > 0.4)
                {
                    classified_boxes[i][k].push_back(input_bounding_boxes[i][j]);
                    break;
                }
            }
        }

        for(Index k = 0; k < classes_number; k++)
        {
            if(classified_boxes[i][k].empty())
                continue;

            sort(classified_boxes[i][k].begin(), classified_boxes[i][k].end(), [](const Tensor<type, 1>& a, const Tensor<type, 1>& b){
                return a(4) > b(4);
            });


            while(!classified_boxes[i][k].empty())
            {
                const Tensor<type, 1> box = classified_boxes[i][k].front();
                classified_boxes[i][k].erase(classified_boxes[i][k].begin());

                for(auto it = classified_boxes[i][k].begin(); it != classified_boxes[i][k].end();)
                    if(calculate_intersection_over_union(box, *it) > overlap_threshold)
                        it = classified_boxes[i][k].erase(it);
                    else
                        it++;

                final_boxes[i].push_back(box);
            }
        }

        // @todo use memcpy

        for(size_t box = 0; box < final_boxes.size(); box++)
            for(Index element = 0; element < final_boxes[i][box].dimension(0); element++)
                tensor_final_boxes[i](box, element) = final_boxes[i][box](element);
    }

    return tensor_final_boxes;
}


Tensor<type, 3> draw_boxes(const Tensor<type, 3>& images, const Tensor<type, 2>& final_boxes)
{

    Tensor<type, 3> image = images;
    const Index image_height = image.dimension(0);
    const Index image_width = image.dimension(1);
    const Index channels = image.dimension(2);

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

void save_tensor_as_BMP(const Tensor<type, 3> &tensor, const string& filename)
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
// Copyright (C) 2005-2024 Artificial Intelligence Techniques SL
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
