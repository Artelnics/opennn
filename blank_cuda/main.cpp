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
#include <time.h>

#include "../opennn/opennn.h"

using namespace std;
using namespace opennn;
using namespace chrono;
using namespace Eigen;


int main()
{
    try
    {
        cout << "OpenNN. Blank Cuda." << endl;

        #ifdef OPENNN_CUDA

        // Data set
        
        const Index samples_number = 2;

        const Index image_height = 4;
        const Index image_width = 4;
        const Index channels = 1;
        const Index targets = 2;

        ImageDataSet data_set(samples_number, {image_height, image_width, channels}, {targets});

        data_set.set_data_random();
 
        /*
        ImageDataSet data_set;

        data_set.set_data_path("../examples/mnist/data");

        data_set.read_bmp();
        */
       
        // Neural network
        
        NeuralNetwork neural_network(NeuralNetwork::ModelType::ImageClassification,
            data_set.get_dimensions(DataSet::VariableUse::Input),
            { 1 },
            data_set.get_dimensions(DataSet::VariableUse::Target));

        // Training strategy

        TrainingStrategy training_strategy(&neural_network, &data_set);

        training_strategy.set_loss_method(TrainingStrategy::LossMethod::CROSS_ENTROPY_ERROR);
        training_strategy.set_optimization_method(TrainingStrategy::OptimizationMethod::ADAPTIVE_MOMENT_ESTIMATION);
        training_strategy.get_loss_index()->set_regularization_method(LossIndex::RegularizationMethod::NoRegularization);
        training_strategy.get_adaptive_moment_estimation()->set_batch_samples_number(512);
        training_strategy.get_adaptive_moment_estimation()->set_maximum_epochs_number(10);
        training_strategy.set_display_period(1);

        training_strategy.perform_training();
        //training_strategy.perform_training_cuda();

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