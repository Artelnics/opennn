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
#include "yolo_dataset.h"


// OpenNN includes

#include "../opennn/opennn.h"
#include "../opennn/data_set.h"


using namespace std;
using namespace opennn;
using namespace std::chrono;
using namespace Eigen;

//namespace fs = std::filesystem;

//falta meter una función que extraiga los batches


type loss_function(const Tensor<type, 3>&, const vector<Tensor<type, 1>>&, Tensor<type, 3>&);


// MIRAR SI TENGO QUE METER UNA FUNCIÓN QUE CONSTRUYA LAS CAJAS A PARTIR DE LOS DATOS DADOS POR LA RED (la red me da un resize de las anchors y tengo que aplicarlo al anchor correspondiente para tener la supuesta caja final)


Tensor<type, 2> non_maximum_suppression(const Tensor<type, 2>&, const type&, const Index&);


Tensor<type, 3> draw_boxes(const Tensor<type, 3>&, const Tensor<type, 2>&);

void save_tensor_as_BMP(const Tensor<type, 3>&, const string&);






int main()
{
    try
    {

        // Tensor<type, 3> test(2,2,3);
        // test(0,0,0)=0;
        // test(0,0,1)=4;
        // test(0,0,2)=8;
        // test(0,1,0)=2;
        // test(0,1,1)=6;
        // test(0,1,2)=10;
        // test(1,0,0)=1;
        // test(1,0,1)=5;
        // test(1,0,2)=9;
        // test(1,1,0)=3;
        // test(1,1,1)=7;
        // test(1,1,2)=11;

        // for(Index i = 0; i < 2*2*3; i++)
        // {
        //     cout<<test(i)<<endl;
        // }

        YOLODataset train_dataset("/Users/artelnics/Desktop/Testing_dataset/VOCdevkit/VOC2007/BMPImages_fixing","/Users/artelnics/Desktop/Testing_dataset/VOCdevkit/VOC2007/Labels_fixing");     //PARA COMPROBAR DONDE FALLA

        // YOLODataset train_dataset("/Users/artelnics/Desktop/Testing_dataset/VOCdevkit/VOC2007/BMPImages_debug","/Users/artelnics/Desktop/Testing_dataset/VOCdevkit/VOC2007/Labels_debug");     //PARA DEBUGGEAR

        // YOLODataset train_dataset("/Users/artelnics/Desktop/Testing_dataset/VOCdevkit/VOC2007/BMPImages","/Users/artelnics/Desktop/Testing_dataset/VOCdevkit/VOC2007/Labels");

        // cout<<convert_to_YOLO_grid_data(train_dataset.getLabel(0), 13, 5, 20)<<endl<<endl;

        // NeuralNetwork neural_network(NeuralNetwork::ModelType::YoloV2,
        //                              train_dataset.get_input_dimensions(),
        //                              { 1 },
        //                              train_dataset.get_target_dimensions());

        // neural_network.print();

        // Tensor<Index, 1> training_indices = train_dataset.get_training_samples_indices();

        // TrainingStrategy training_strategy(&neural_network, &train_dataset);

        // training_strategy.set_loss_method(TrainingStrategy::LossMethod::CROSS_ENTROPY_ERROR); //TENGO QUE CREAR UN TIPO DE ERROR LLAMADO YOLOV2ERROR QUE ME DE LA LOSS FUNCTION DE YOLO Y USAR ESE
        // training_strategy.set_optimization_method(TrainingStrategy::OptimizationMethod::ADAPTIVE_MOMENT_ESTIMATION);
        // training_strategy.get_loss_index()->set_regularization_method(LossIndex::RegularizationMethod::NoRegularization);
        // training_strategy.get_adaptive_moment_estimation()->set_batch_samples_number(70);
        // training_strategy.get_adaptive_moment_estimation()->set_maximum_epochs_number(0);
        // training_strategy.set_display_period(1);


        // training_strategy.perform_training();

        // cout<<"========"<<endl;

        // Tensor<Index, 1> validation_indices = train_dataset.get_selection_samples_indices();

        // cout<<"========"<<endl;

        // Tensor<Index, 1> testing_indices = train_dataset.get_testing_samples_indices();

        // cout<<"========"<<endl;

        // Tensor<Index, 2> training_batches = train_dataset.get_batches(training_indices, 200, true);
        // Tensor<Index, 2> validation_batches = train_dataset.get_batches(validation_indices, 200, true);
        // Tensor<Index, 2> testing_batches = train_dataset.get_batches(testing_indices, 200, true);





        // cout<<training_indices.dimension(0)<<endl;

        // Index image_number = 160;

        // cout<<train_dataset.getClass(0)<<endl;

        // // for(Index i = 0; i < training_indices.dimension(0); i++)
        // for(Index i = 0; i< 4; i++)
        // {
        //     string complete = "output_" + to_string(i) + ".bmp";

        //     string filename = "/Users/artelnics/Desktop/image_test/" + complete ;    //Emplear este proceso para guardar las imágenes con las bounding_boxes ya metidas en una carpeta

        //     /*cout<<train_dataset.offsets[image_number - 1 + i]<<endl*/;

        //     train_dataset.apply_data_augmentation(train_dataset.images[image_number - 1 + i], train_dataset.labels[image_number - 1 + i]);

        //     // cout<<train_dataset.labels[image_number - 1 + i]<<endl;

        //     // train_dataset.apply_data_augmentation(train_dataset.images[training_indices(i)], train_dataset.labels[training_indices(i)]);

        //     // save_tensor_as_BMP(draw_boxes(normalize_tensor(train_dataset.getImage(training_indices(i)), true), train_dataset.getLabel(training_indices(i))), filename);

        //     save_tensor_as_BMP(draw_boxes(normalize_tensor(train_dataset.getImage(image_number - 1 + i), true), train_dataset.getLabel(image_number - 1 + i)), filename);
        // }

        cout<<"works properly"<<endl;

        return 0;
    }
    catch (const exception& e)
    {
        cerr << e.what() << endl;

        return 1;
    }
}


type loss_function(const Tensor<type, 3>& target_data, const vector<Tensor<type, 1>>& anchors, Tensor<type, 3>& network_output)
{
    Index S = network_output.dimension(0);
    Index B = anchors.size();
    Index C = (network_output.dimension(3) / anchors.size()) - 5;
    type total_loss = 0, coord_loss = 0, conf_loss_object = 0, conf_loss_noobject = 0, conf_loss = 0, class_loss = 0;
    type lambda_coord = 5.0;
    type lambda_noobject = 0.5;

    for(Index i = 0; i < S; i++)
    {
        for(Index j = 0; j < S; j++)
        {
            for(Index k = 0; k < B; k++)
            {

                if(target_data(i,j, k * (5 + C) + 4) == 1)
                {
                    coord_loss += pow(target_data(i, j, k * (5 + C) + 0) - network_output(i, j, k * (5 + C) + 0), 2) +
                                  pow(target_data(i, j, k * (5 + C) + 1) - network_output(i, j, k * (5 + C) + 1), 2) +
                                  pow(sqrt(target_data(i, j, k * (5 + C) + 2)) - sqrt(exp(network_output(i, j, k * (5 + C) + 2)) * anchors[k](0)), 2) +
                                  pow(sqrt(target_data(i, j, k * (5 + C) + 3)) - sqrt(exp(network_output(i, j, k * (5 + C) + 3)) * anchors[k](1)), 2);

                    conf_loss_object += pow(1 - network_output(i,j, k * (5 + C) + 4), 2);
                    for(Index c = 0; c < C; c++)
                    {
                        class_loss += target_data(i,j, k * (5 + C) + c) * log(network_output(i,j, k * (5 + C) + c));
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

Tensor<type, 2> non_maximum_suppression(const Tensor<type, 2>& network_output, const type& overlap_treshold, const Index& classes_number)
{
    Tensor<type, 3> converted_network_output(/*network_output.dimension(0),*/13,13,125); //Should be Tensor<type, 4> bc I need to add the batch_size dimension

    // pair<type*, Tensor;



    vector<Tensor<type, 1>> input_bounding_boxes;
    vector<Tensor<type, 1>> final_boxes;
    vector<vector<Tensor<type, 1>>> classified_boxes(classes_number);
    Tensor<type, 1> box(25);

    for(Index i = 0; i < converted_network_output.dimension(0); i++)
    {
        for(Index j = 0; j < converted_network_output.dimension(1); j++)
        {
            for(Index k = 0; k < converted_network_output.dimension(2); k++)
            {
                if(k % 25 == 0 && converted_network_output(i,j,k) == 0)
                {
                    break;
                } else
                {
                    box (k % 25) = converted_network_output(i,j,k);
                    if(k % 25 == 24) input_bounding_boxes.push_back(box);
                }
            }
        }
    }


    for(size_t i = 0; i < input_bounding_boxes.size(); i++)
    {
        for(Index j = 0; j < classes_number; j++)
        {
            if(input_bounding_boxes[i](5+j) > 0.3)
            {
                classified_boxes[j].push_back(input_bounding_boxes[i]);
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



    Tensor<type, 2> tensor_final_boxes(/*network_output.dimension(0),*/ final_boxes.size(), final_boxes[0].dimension(0));    //mirar si puedo hacer un tensor map

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
