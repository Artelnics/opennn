//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   B L A N K   C U D A
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "../opennn/opennn.h"
#include <iostream>

using namespace opennn;


int main()
{
    try
    {
        cout << "OpenNN. Blank Cuda." << endl;
        
#ifdef OPENNN_CUDA

        cout << "OpenNN. Breast Cancer Example." << endl;

        // Dataset

        Dataset dataset("/home/artelnics/Documents/breast_cancer.csv", ";", true, false);

        const Index neurons_number = 3;

        // Neural Network

        ClassificationNetwork classification_network(dataset.get_input_shape(), { neurons_number}, dataset.get_target_shape());

        // Training Strategy

        WeightedSquaredError loss(&classification_network, &dataset);
        loss.set_regularization_method("L1");

        TrainingStrategy training_strategy(&classification_network, &dataset);

        training_strategy.get_loss()->set_regularization_method("None");
        training_strategy.set_optimization_algorithm("AdaptiveMomentEstimation");
        AdaptiveMomentEstimation* adam = dynamic_cast<AdaptiveMomentEstimation*>(training_strategy.get_optimization_algorithm());
        adam->set_maximum_epochs(1000);

        training_strategy.train_cuda();

        // Testing Analysis

        TestingAnalysis testing_analysis(&classification_network, &dataset);

        cout << testing_analysis.calculate_confusion_cuda() << endl;

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
