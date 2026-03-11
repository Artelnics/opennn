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
        
        cout << "OpenNN. Melanoma Cancer CUDA Example." << endl;

#ifdef OPENNN_CUDA

        // Data set
/*
        ImageDataset image_dataset("/home/davidgonzalez/opennn/melanoma_dataset_bmp");

        image_dataset.split_samples_random(0.6, 0.2, 0.2);

        // Neural network
        
        ImageClassificationNetwork image_classification_network(
            image_dataset.get_shape("Input"),
            { 32, 64, 16 },
            image_dataset.get_shape("Target"));
        
        // Training strategy

        TrainingStrategy training_strategy(&image_classification_network, &image_dataset);

        training_strategy.set_loss("CrossEntropyError2d");
        training_strategy.set_optimization_algorithm("AdaptiveMomentEstimation");
        training_strategy.get_loss()->set_regularization_method("None");

        AdaptiveMomentEstimation* adam = dynamic_cast<AdaptiveMomentEstimation*>(training_strategy.get_optimization_algorithm());
        adam->set_display_period(1);
        adam->set_batch_size(64);
        adam->set_maximum_epochs(5);

        training_strategy.train_cuda();

        // Testing analysis

        TestingAnalysis testing_analysis(&image_classification_network, &image_dataset);

        testing_analysis.set_batch_size(64);
        cout << "Calculating confusion...." << endl;
        const MatrixI confusion = testing_analysis.calculate_confusion_cuda();
        cout << "\nConfusion matrix:\n" << confusion << endl;
        */

#endif

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

        training_strategy.train();

        // Testing Analysis

        TestingAnalysis testing_analysis(&classification_network, &dataset);

        testing_analysis.print_binary_classification_tests();

        TestingAnalysis::RocAnalysis roc = testing_analysis.perform_roc_analysis();

        cout << "Good bye!" << endl;

   
#ifndef OPENNN_CUDA
        cout << "Enable CUDA in pch.h" << endl;
#endif
        cout << "Completed." << endl;

        return 0;
    }
    catch (const exception& e)
    {
        cerr << e.what() << endl;

        return 1;
    }
}
