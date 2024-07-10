//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   B L A N K   A P P L I C A T I O N
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

// System includes

#include <stdio.h>
#include <cstring>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <time.h>
#include <chrono>
#include <algorithm>
#include <execution>

// OpenNN includes

#include "../opennn/opennn.h"
#include <iostream>

// OneDNN

using namespace std;
using namespace opennn;
using namespace std::chrono;
using namespace Eigen;

int main()
{
   try
   {
        cout << "Blank\n";

        // Data Set

        const Index samples_number = 1000;
        const Index inputs_number = 10;
        const Index outputs_number = 2;
        const Index hidden_neurons_number = 10;

        DataSet data_set(samples_number, inputs_number, outputs_number);

        data_set.set_data_random();
        data_set.set_training();

        // Neural network
        
        NeuralNetwork neural_network;

        PerceptronLayer* perceptron_layer_1 = new PerceptronLayer(inputs_number, hidden_neurons_number, PerceptronLayer::ActivationFunction::HyperbolicTangent);
        neural_network.add_layer(perceptron_layer_1);
        
        PerceptronLayer* perceptron_layer_2 = new PerceptronLayer(hidden_neurons_number, outputs_number, PerceptronLayer::ActivationFunction::Linear);
        neural_network.add_layer(perceptron_layer_2);

        // Training strategy

        TrainingStrategy training_strategy(&neural_network, &data_set);

        training_strategy.set_loss_method(TrainingStrategy::LossMethod::MEAN_SQUARED_ERROR);
        training_strategy.set_optimization_method(TrainingStrategy::OptimizationMethod::LEVENBERG_MARQUARDT_ALGORITHM);

        training_strategy.get_loss_index()->set_regularization_method(LossIndex::RegularizationMethod::NoRegularization);
        //training_strategy.get_loss_index()->set_regularization_weight(0.01);

        training_strategy.set_maximum_epochs_number(5);
        training_strategy.set_display_period(1);
        training_strategy.set_maximum_time(86400);

        training_strategy.perform_training();

        cout << "Bye!" << endl;

        return 0;
   }
   catch (const exception& e)
   {
       cerr << e.what() << endl;

       return 1;
   }
}

/*
#include "../opennn/opennn.h"

#include <iostream>
#include <fstream>
#include <vector>

using namespace Eigen;
using namespace std;

// Simple BMP file header structure
#pragma pack(push, 1)
struct BMPHeader {
    uint16_t signature; // BM
    uint32_t fileSize;
    uint32_t reserved;
    uint32_t dataOffset;
    uint32_t headerSize;
    int32_t width;
    int32_t height;
    uint16_t planes;
    uint16_t bitsPerPixel;
    uint32_t compression;
    uint32_t dataSize;
    int32_t horizontalResolution;
    int32_t verticalResolution;
    uint32_t colors;
    uint32_t importantColors;
};
#pragma pack(pop)

// Function to read BMP image and store it in an Eigen Tensor
Tensor<unsigned char, 3> readBMPImage(const char* filename) {
    ifstream file(filename, ios::in | ios::binary);
    if (!file.is_open()) {
        cerr << "Error opening file" << endl;
        exit(EXIT_FAILURE);
    }

    BMPHeader header;

    file.read(reinterpret_cast<char*>(&header), sizeof(BMPHeader));

    if (header.signature != 0x4D42) {
        cerr << "Not a BMP file" << endl;
        exit(EXIT_FAILURE);
    }

    if (header.bitsPerPixel != 24) {
        cerr << "Only 24-bit BMP images are supported" << endl;
        exit(EXIT_FAILURE);
    }

    Tensor<unsigned char, 3> tensor(3, header.height, header.width);

    const int padding = (4 - (header.width * 3 % 4)) % 4; // BMP padding

    unsigned char pixel;

    for (int y = header.height - 1; y >= 0; --y) {
        for (int x = 0; x < header.width; ++x) {
            for (int c = 0; c < 3; ++c) {
                file.read(reinterpret_cast<char*>(&pixel), sizeof(unsigned char));
                tensor(c, header.height - 1 - y, x) = pixel;
            }
        }
        file.seekg(padding, ios::cur);
    }

    file.close();

    cout << tensor << endl;
    return tensor;
}

int main() {
    const char* filename = "C:/8.bmp";
    Tensor<unsigned char, 3> imageTensor = readBMPImage(filename);

    // Now you can use the 'imageTensor' for further processing or analysis

    cout << "Image dimensions: " << imageTensor.dimension(0) << "x" << imageTensor.dimension(1) << "x" << imageTensor.dimension(2) << endl;

    return 0;
}
*/
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
