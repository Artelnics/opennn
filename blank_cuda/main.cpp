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

        ImageDataset data_set(samples_number, {image_height, image_width, channels}, {targets});

        data_set.set_data_random();
        data_set.set_data_ascending();

        data_set.set(Dataset::SampleUse::Training);

        data_set.print_data();
        */
        
        ImageDataset data_set;

        data_set.set_data_path("../examples/mnist/data");

        data_set.read_bmp();

        data_set.split_samples_random(0.8, 0.0, 0.2);

        const dimensions input_dimensions = data_set.get_dimensions(Dataset::VariableUse::Input);
        const dimensions target_dimensions = data_set.get_dimensions(Dataset::VariableUse::Target);
        
        // Neural network

        NeuralNetwork neural_network(NeuralNetwork::ModelType::ImageClassification,
            data_set.get_dimensions(Dataset::VariableUse::Input),
            { 32 },
            data_set.get_dimensions(Dataset::VariableUse::Target));
        
        //VGG16 neural_network(input_dimensions, target_dimensions);

        // Training strategy

        TrainingStrategy training_strategy(&neural_network, &data_set);

        training_strategy.set_loss_method(TrainingStrategy::LossMethod::CROSS_ENTROPY_ERROR_2D);
        training_strategy.set_optimization_method(TrainingStrategy::OptimizationMethod::ADAPTIVE_MOMENT_ESTIMATION);
        training_strategy.get_loss_index()->set_regularization_method(LossIndex::RegularizationMethod::NoRegularization);
        training_strategy.get_adaptive_moment_estimation()->set_batch_samples_number(128);
        training_strategy.get_adaptive_moment_estimation()->set_maximum_epochs_number(10);
        training_strategy.set_display_period(1);

        //training_strategy.perform_training();
        training_strategy.perform_training_cuda();

        // Testing analysis

        const TestingAnalysis testing_analysis(&neural_network, &data_set);

        cout << "Calculating confusion...." << endl;
        const Tensor<Index, 2> confusion = testing_analysis.calculate_confusion();
        cout << "\nConfusion matrix:\n" << confusion << endl;

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