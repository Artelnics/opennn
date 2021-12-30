//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   M N I S T    A P P L I C A T I O N
//
//   Artificial Intelligence Techniques SL (Artelnics)
//   artelnics@artelnics.com
#define _SILENCE_EXPERIMENTAL_FILESYSTEM_DEPRECATION_WARNING

#include <iostream>
#include <vector>
#include <filesystem>
#include <experimental/filesystem>
using namespace std;

vector<char> read_bmp(const string& filename)
{
    int i;
    FILE* f = fopen(filename.data(), "rb");

    unsigned char info[54];

    // read the 54-byte header
    fread(info, sizeof(unsigned char), 54, f);

    // extract image height and width from header
    const int width = *(int*)&info[18];
    const int height = *(int*)&info[22];
    const int image_size = *(int*)&info[34];
    //const short int channels1 = *(int*)&info[26];
    const short int channels3 = image_size/(width*height);
    short int channels;
    int size;

    if (channels3 != 3){
         size =  width * height;
         channels = 1;
    }else {
         size = channels3 * width * height;
         channels=channels3;
    }

    cout<<"Channels: "<<channels<<endl;
    //unsigned char* data = new unsigned char[size];

    vector<char> data(size);

    // read the rest of the data at once
    fread(data.data(), sizeof(unsigned char), size, f);
    fclose(f);

    for(i = 0; i < size; i += channels)
    {
        if(channels==3){
            // flip the order of every 3 bytes
            unsigned char tmp = data[i];
            data[i] = data[i+2];
            data[i+2] = tmp;

            cout << "R: " << int(data[i] & 0xff) << " G: " << int(data[i+1] & 0xff) << " B: " << int(data[i+2] & 0xff) << endl;
        }else if (channels==1) {
            unsigned char tmp = data[i];
            cout << "GrayScale1: " << int(data[i] & 0xff)<<endl;        }
    }



//    std::vector<char> v(data, data + sizeof data / sizeof data[0]);

    return data;
}

const int number_of_files_in_directory(std::experimental::filesystem::path path)
{
    using std::experimental::filesystem::directory_iterator;
    return std::distance(directory_iterator(path), directory_iterator{});
}


void read_images()
{
    int images_number;
    int classes_number;
    int image_size = 28*28;
    string path;

    path = "C:/Users/Artelnics/Desktop/mnist/data/";

    // Miramos cuantas carpetas contiene. Ese es el número de clases.
    //classes_number = number_of_files_in_directory(std::experimental::filesystem::current_path());
    classes_number = number_of_files_in_directory(path);
    cout << "El numero de clases es: " << classes_number << endl;

    // Vamos a cada subdirectorio y Contamos cuantos archivos tiene
    images_number = number_of_files_in_directory("C:/Users/Artelnics/Desktop/mnist/data/zero/")+
                    number_of_files_in_directory("C:/Users/Artelnics/Desktop/mnist/data/one/")+
                    number_of_files_in_directory("C:/Users/Artelnics/Desktop/mnist/data/two/")+
                    number_of_files_in_directory("C:/Users/Artelnics/Desktop/mnist/data/three/");

    cout << "El numero de imagenes es: " << images_number << endl;

    // El número total es el número de filas.

    // Cambiamos el tamaño de las matrices y vectores

    //data.resize(images_number, image_size + classes_number);

    //rows_labels.resize(9);

    // Rellenamos las matrices leyendo las imagenes.

    //v=readBMP("C:/Users/Artelnics/Desktop/mnist/data/two/2_345.bmp");

}

int main()
{
    vector<char> v = read_bmp("C:/Users/Artelnics/Desktop/mnist/data/101.bmp");

    //for(int i = 0; i < v.size(); i++) cout << v[i] << " ";

    cout << v.size() << endl;
    read_images();
    return 0;
}


/*
// This is an approximation application.

// System includes

#include <iostream>
#include <sstream>
#include <time.h>
#include <stdexcept>

// OpenNN includes

#include "../../opennn/opennn.h"
#include "../../opennn/opennn_strings.h"

using namespace std;
using namespace OpenNN;

int main()
{
    try
    {
        cout << "OpenNN. National Institute of Standards and Techonology (MNIST) Example." << endl;

        srand(static_cast<unsigned>(time(nullptr)));

        // Data set

        DataSet data_set;//("../data/mnist_train.csv", ',', true);

        data_set.set_data_file_name("c:/mnsit");

        data_set.read_images();

        data_set.set_input();

        data_set.set_column_use(0, DataSet::VariableUse::Target);

        const Index input_variables_number = data_set.get_input_variables_number();
        const Index target_variables_number = data_set.get_target_variables_number();

        const Tensor<DataSet::Column, 1> columns = data_set.get_columns();

        for(Index i = 0; i < columns.size(); i++)
        {
            cout << "Column " << i << ": " << endl;
            cout << "   Name: " << columns(i).name << endl;

            if(columns(i).column_use == OpenNN::DataSet::VariableUse::Input) cout << "   Use: input" << endl;
            else if(columns(i).column_use == OpenNN::DataSet::VariableUse::Target) cout << "   Use: target" << endl;
            else if(columns(i).column_use == OpenNN::DataSet::VariableUse::UnusedVariable) cout << "   Use: unused" << endl;

            if(columns(i).type == OpenNN::DataSet::ColumnType::Categorical) cout << "   Categories: " << columns(i).categories << endl;

            cout << endl;
        }

        cout << "Input variables number: " << data_set.get_target_variables_number() << endl;
        cout << "Target variables number: " << data_set.get_target_variables_number() << endl;

        // Neural network

        Index hidden_neurons_number = 50;

        NeuralNetwork neural_network(NeuralNetwork::ProjectType::Classification, {input_variables_number, hidden_neurons_number, target_variables_number});

        PerceptronLayer* perceptron_layer_pointer = neural_network.get_first_perceptron_layer_pointer();
        perceptron_layer_pointer->set_activation_function("RectifiedLinear");

        Tensor<Layer*, 1> layers_pointers = neural_network.get_trainable_layers_pointers();

        for(Index i = 0; i < layers_pointers.size(); i++)
        {
            cout << "Layer " << i << ": " << endl;
            cout << "   Type: " << layers_pointers(i)->get_type_string() << endl;

            if(layers_pointers(i)->get_type_string() == "Perceptron") cout << "   Activation: " << static_cast<PerceptronLayer*>(layers_pointers(i))->write_activation_function() << endl;
            if(layers_pointers(i)->get_type_string() == "Probabilistic") cout << "   Activation: " << static_cast<ProbabilisticLayer*>(layers_pointers(i))->write_activation_function() << endl;
        }

        // Training strategy

        TrainingStrategy training_strategy(&neural_network, &data_set);

        training_strategy.set_loss_method(TrainingStrategy::LossMethod::CROSS_ENTROPY_ERROR);
        training_strategy.set_optimization_method(TrainingStrategy::OptimizationMethod::ADAPTIVE_MOMENT_ESTIMATION);
        training_strategy.set_display_period(100);
        training_strategy.set_maximum_epochs_number(1000);

        training_strategy.perform_training();

        // Testing analysis

        const TestingAnalysis testing_analysis(&neural_network, &data_set);

        const Tensor<Index, 2> confusion = testing_analysis.calculate_confusion();
        const Tensor<type, 1> multiple_classification_tests = testing_analysis.calculate_multiple_classification_tests();

        cout << "Confusion matrix: " << endl;
        cout << confusion << endl;

        cout << "Accuracy: " << multiple_classification_tests(0)*type(100) << "%" << endl;
        cout << "Error: " << multiple_classification_tests(1)*type(100) << "%" << endl;

        return 0;
    }
    catch(exception& e)
    {
        cerr << e.what() << endl;

        return 1;
    }
}
*/


// OpenNN: Open Neural Networks Library.
// Copyright (C) 2005-2021 Artificial Intelligence Techniques SL
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
