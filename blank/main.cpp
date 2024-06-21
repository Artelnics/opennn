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
        /**/
        _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
        _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
        
        LanguageDataSet language_data_set;
        
        language_data_set.set_data_source_path("data/language/ENtoES_large_shuffled.txt");

        language_data_set.set_text_separator(DataSet::Separator::Tab);

        language_data_set.set_context_vocabulary_path("data/language/EN_mac_vocabulary.txt");
        language_data_set.set_completion_vocabulary_path("data/language/ES_mac_vocabulary.txt");

        language_data_set.read_txt_language_model();

        const Tensor<string, 1>& completion_vocabulary = language_data_set.get_completion_vocabulary();
        const Tensor<string, 1>& context_vocabulary = language_data_set.get_context_vocabulary();

        Index input_length = language_data_set.get_completion_length();
        Index context_length = language_data_set.get_context_length();
        Index inputs_dimension = language_data_set.get_completion_vocabulary_size();
        Index context_dimension = language_data_set.get_context_vocabulary_size();

        // Testing Analysis
        {
            Transformer transformer2;

            transformer2.load_transformer("data/language/ENtoES_model.xml");

            transformer2.set_input_vocabulary(completion_vocabulary);
            transformer2.set_context_vocabulary(context_vocabulary);

            const TestingAnalysis testing_analysis2(&transformer2, &language_data_set);

            pair<type, type> transformer_error_accuracy2 = testing_analysis2.test_transformer();

            cout << "TESTING ANALYSIS:" << endl;
            cout << "Testing error: " << transformer_error_accuracy2.first << endl;
            cout << "Testing accuracy: " << transformer_error_accuracy2.second << endl;
        }
        
        //const bool imported_vocabulary = true;
        /*
        cout << endl << "DEPLOYMENT:" << endl;
        string input;
        string output;
        
        input = "He is playing soccer.";
        output = transformer2.calculate_outputs(input);
        
        cout << "Input: " << input << endl;
        cout << "Output: " << output << endl;
        cout << endl;
        
        input = "She is studying for her history test.";
        output = transformer2.calculate_outputs(input);

        cout << "Input: " << input << endl;
        cout << "Output: " << output << endl;
        cout << endl;

        input = "The dog is barking.";
        output = transformer2.calculate_outputs(input);

        cout << "Input: " << input << endl;
        cout << "Output: " << output << endl;
        cout << endl;

        input = "The cat likes to sleep.";
        output = transformer2.calculate_outputs(input);

        cout << "Input: " << input << endl;
        cout << "Output: " << output << endl;
        cout << endl;

        input = "Tom took his friends out on a party Saturday night.";
        output = transformer2.calculate_outputs(input);

        cout << "Input: " << input << endl;
        cout << "Output: " << output << endl;
        cout << endl;
        /*
        while (true)
        {
            cout << "Input: ";
            getline(cin, input);

            if (input == "-Q")
                break;

            output = transformer2.calculate_outputs(input);

            cout << "Output: " << output << endl;
            cout << endl;
        }
        /**/        

        /*
        const Index samples_number = 200;
        const Index inputs_number = 1000;
        const Index inputs_dimension = 1000;
        const Index depth = 1000;

        Tensor<Index, 1> inputs_dimensions(2);
        inputs_dimensions.setValues({ inputs_number, depth });

        DataSet dataset(samples_number, inputs_number * depth, inputs_number);

        dataset.set_input_variables_dimensions(inputs_dimensions);

        Tensor<type, 2> data(samples_number, inputs_number * depth + inputs_number);
        data.setRandom();

        for (Index i = 0; i < samples_number; i++)
        {
            for (Index j = inputs_number * depth; j < inputs_number * depth + inputs_number; j++)
            {
                data(i, j) = type(Index(data(i, j) * 1000) % inputs_dimension);
            }
        }

        Tensor<Index, 1> input_variables(inputs_number * depth);
        for (Index i = 0; i < input_variables.size(); i++)    input_variables(i) = i;

        Tensor<Index, 1> target_variables(inputs_number);
        for (Index i = 0; i < target_variables.size(); i++)    target_variables(i) = inputs_number * depth + i;

        dataset.set_data(data);
        dataset.set_input_target_raw_variables(input_variables, target_variables);
        dataset.set_training();

        NeuralNetwork neural_network;

        ProbabilisticLayer3D* probabilistic_layer = new ProbabilisticLayer3D(inputs_number, depth, inputs_dimension);

        neural_network.add_layer(probabilistic_layer);

        CrossEntropyError3D cross_entropy_error_3d(&neural_network, &dataset);
        cross_entropy_error_3d.set_regularization_method(LossIndex::RegularizationMethod::NoRegularization);

        AdaptiveMomentEstimation optimization_algorithm;
        optimization_algorithm.set_loss_index(&cross_entropy_error_3d);

        optimization_algorithm.set_display(true);
        optimization_algorithm.set_display_period(1);

        type loss_goal = type(0.99);
        //type training_accuracy_goal = type(0.80);

        optimization_algorithm.set_loss_goal(loss_goal);
        //optimization_algorithm.set_accuracy_goal(training_accuracy_goal);
        optimization_algorithm.set_maximum_epochs_number(0);
        optimization_algorithm.set_maximum_time(3 * 86400);
        optimization_algorithm.set_batch_samples_number(200);

        TrainingResults training_results = optimization_algorithm.perform_training();
        */

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
