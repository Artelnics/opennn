//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   B L A N K   C U D A   A P P L I C A T I O N
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <cstring>
#include <ctime>

#include "../opennn/pch.h"
#include "../opennn/dataset.h"
#include "../opennn/neural_network.h"
#include "../opennn/vgg16.h"
#include "../opennn/training_strategy.h"
#include "../opennn/testing_analysis.h"
#include "../opennn/image_dataset.h"
#include "../opennn/scaling_layer_4d.h"
#include "../opennn/convolutional_layer.h"
#include "../opennn/pooling_layer.h"
#include "../opennn/flatten_layer.h"
#include "../opennn/perceptron_layer.h"

using namespace std;
using namespace chrono;
using namespace Eigen;
using namespace opennn;


int main()
{
    try
    {
        cout << "OpenNN. Blank Cuda." << endl;

        #ifdef OPENNN_CUDA

        // Data set
        /*
        const Index samples_number = 2;

        const Index image_height = 3;
        const Index image_width = 3;
        const Index channels = 3;
        const Index targets = 2;

        ImageDataset dataset(samples_number, {image_height, image_width, channels}, {targets});

        dataset.set_data_random();
        dataset.set_data_ascending();

        dataset.set(Dataset::SampleUse::Training);

        dataset.print_data();
        */
        ImageDataset dataset;

        //dataset.set_data_path("C:/melanoma_dataset_bmp_medium");
        dataset.set_data_path("/mnt/c/melanoma_dataset_bmp_medium"); // WSL
        //dataset.set_data_path("../examples/mnist/data_bin");

        dimensions data_dimensions = { 224, 224, 3 };

        dataset.read_bmp(data_dimensions);
        //dataset.read_bmp();

        dataset.split_samples_random(0.8, 0.0, 0.2);

        const dimensions input_dimensions  = dataset.get_dimensions(Dataset::VariableUse::Input);
        const dimensions output_dimensions = dataset.get_dimensions(Dataset::VariableUse::Target);
        
        // Neural network

        NeuralNetwork neural_network(NeuralNetwork::ModelType::ImageClassification,
            input_dimensions,
            { 64, 64, 128, 128, 32 },
            output_dimensions);
        
        //VGG16 neural_network(input_dimensions, output_dimensions);

        // Training strategy
        
        CrossEntropyError2d cross_entropy_error(&neural_network, &dataset);
        cross_entropy_error.set_regularization_method(LossIndex::RegularizationMethod::NoRegularization);

        AdaptiveMomentEstimation adaptive_moment_estimation(&cross_entropy_error);
        adaptive_moment_estimation.set_batch_size(8);
        adaptive_moment_estimation.set_maximum_epochs_number(10);
        adaptive_moment_estimation.set_display_period(1);

        TrainingStrategy training_strategy(&neural_network, &dataset);

        //training_strategy.perform_training();
        training_strategy.perform_training_cuda();

        // Testing analysis

        TestingAnalysis testing_analysis(&neural_network, &dataset);
        testing_analysis.set_batch_size(8);

        cout << "Calculating confusion...." << endl;
        Tensor<Index, 2> confusion = testing_analysis.calculate_confusion_cuda();
        cout << "\nConfusion matrix CUDA:\n" << confusion << endl;
        
        #endif

        cout << "Bye!" << endl;
        
        return 0;
    }
    catch (const exception& e)
    {
        cerr << e.what() << endl;

        return 1;
    }
}
