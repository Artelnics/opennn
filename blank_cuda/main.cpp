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

        #ifdef OPENNN_CUDA_test

        // Data set
        
        DataSet data_set("data/iris_plant_original.csv", ";", true, false);

        const Index input_variables_number = data_set.get_variables_number(DataSet::VariableUse::Input);
        const Index target_variables_number = data_set.get_variables_number(DataSet::VariableUse::Target);

        // Neural network

        const Index neurons_number = 1;

        NeuralNetwork neural_network(NeuralNetwork::ModelType::Classification,
            { input_variables_number }, { neurons_number }, { target_variables_number });

        // Training strategy
        
        TrainingStrategy training_strategy(&neural_network, &data_set);
        
        training_strategy.set_loss_method(TrainingStrategy::LossMethod::CROSS_ENTROPY_ERROR);
        training_strategy.set_optimization_method(TrainingStrategy::OptimizationMethod::ADAPTIVE_MOMENT_ESTIMATION);
        training_strategy.get_loss_index()->set_regularization_method(LossIndex::RegularizationMethod::NoRegularization);
        
        //TrainingResults results = training_strategy.perform_training();
        TrainingResults results = training_strategy.perform_training_cuda();

        // Testing analysis

        const TestingAnalysis testing_analysis(&neural_network, &data_set);

        cout << "Confusion matrix:\n" << testing_analysis.calculate_confusion() << endl;
        
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