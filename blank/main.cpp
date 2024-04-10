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
//using namespace Eigen;

int main()
{
   try
   {
        cout << "Blank\n";
        /*
        LanguageDataSet language_data_set;

        //language_data_set.set_data_source_path("data/example2.txt");
        //language_data_set.set_data_source_path("data/PTtoEN_dataset.txt");
        language_data_set.set_data_source_path("data/three_letter_combinations_with_spaces.txt");
        language_data_set.set_text_separator(DataSet::Separator::Tab);

        language_data_set.read_txt_language_model();
        
        language_data_set.set_training();

        Index input_length = language_data_set.get_completion_length();
        Index context_length = language_data_set.get_context_length();
        Index inputs_dimension = language_data_set.get_completion_vocabulary_size();
        Index context_dimension = language_data_set.get_context_vocabulary_size();
        
        Index number_of_layers = 1;
        Index depth = 4;
        Index perceptron_depth = 12;
        Index heads_number = 2;

        Transformer transformer({ input_length, context_length, inputs_dimension, context_dimension,
                          depth, perceptron_depth, heads_number, number_of_layers });

        Tensor<string, 1>& completion_vocabulary = language_data_set.get_completion_vocabulary();
        Tensor<string, 1>& context_vocabulary = language_data_set.get_context_vocabulary();

        transformer.set_input_vocabulary(completion_vocabulary);
        transformer.set_context_vocabulary(context_vocabulary);

        //type training_loss_goal = type(0.05);

        CrossEntropyError3D cross_entropy_error_3d(&transformer, &language_data_set);

        StochasticGradientDescent stochastic_gradient_descent;
        stochastic_gradient_descent.set_loss_index(&cross_entropy_error_3d);

        stochastic_gradient_descent.set_display(true);
        stochastic_gradient_descent.set_display_period(1);

        //stochastic_gradient_descent.set_loss_goal(training_loss_goal);
        stochastic_gradient_descent.set_maximum_epochs_number(19);
        stochastic_gradient_descent.set_maximum_time(86400);
        stochastic_gradient_descent.set_batch_samples_number(32);

        Index parameter_number = 0;
        for (Index i = 0; i < transformer.get_layers().size(); i++)
        {
            cout << transformer.get_layer(i)->get_name() << " from parameter " << parameter_number << " to " << parameter_number + transformer.get_layer(i)->get_parameters_number() - 1 << endl;
            parameter_number += transformer.get_layer(i)->get_parameters_number();
        }

        TrainingResults training_results = stochastic_gradient_descent.perform_training();

        //transformer.calculate_outputs();
        */


        const Index samples_number = 1000000;
        const Index inputs_number = 1000;
        const Index outputs_number = 1;
        const Index hidden_neurons_number = 1000;

        DataSet data_set;// ("C:/R_100000_samples_11_variables.csv", ',', true);

        data_set.generate_Rosenbrock_data(samples_number, inputs_number + outputs_number);

        data_set.set_training();

        const Index rows =  8*1000000;
        const Index columns = 1000;
        type mean = 0.0;

        Tensor<float, 2> matrix(rows, columns);

        for(Index i = 0; i < rows; i++)
        {
            for(Index j = 0; j < columns; j++)
            {
                matrix(i,j) = arc4random();
            }
        }
        for(Index i = 0; i < rows; i++)
        {
            for(Index j = 0; j < columns; j++)
            {
                mean = mean + matrix(i,j);
            }
        }
        mean = mean / (rows*columns);

        cout << "Matrix mean: " << mean << endl;

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
