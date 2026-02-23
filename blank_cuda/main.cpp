//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   B L A N K   C U D A
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "../opennn/opennn.h"

using namespace opennn;


int main()
{
    try
    {
        cout << "OpenNN. Blank Cuda." << endl;
        
        cout << "OpenNN. Melanoma Cancer CUDA Example." << endl;

#ifdef OPENNN_CUDA

        // Data set

        ImageDataset image_dataset("/home/davidgonzalez/opennn/melanoma_dataset_bmp");

        image_dataset.split_samples_random(0.6, 0.2, 0.2);

        // Neural network

        ImageClassificationNetwork image_classification_network(
            image_dataset.get_shape("Input"),
            { 32, 64, 16 },
            image_dataset.get_shape("Target"));

        // Training strategy

        TrainingStrategy training_strategy(&image_classification_network, &image_dataset);

        training_strategy.set_loss_index("CrossEntropyError2d");
        training_strategy.set_optimization_algorithm("AdaptiveMomentEstimation");
        training_strategy.get_loss_index()->set_regularization_method("None");

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

#endif
/*
cout << "OpenNN. National Institute of Standards and Techonology (MNIST) Example." << endl;

        // Dataset

        ImageDataset image_dataset("/mnt/c/Users/davidgonzalez/Documents/mnist_data");

        // Neural network

        ImageClassificationNetwork image_classification_network(image_dataset.get_shape("Input"),
            {4},
            image_dataset.get_shape("Target"));

        // Training strategy

        TrainingStrategy training_strategy(&image_classification_network, &image_dataset);

        training_strategy.set_loss_index("CrossEntropyError2d");
        training_strategy.set_optimization_algorithm("AdaptiveMomentEstimation");
        training_strategy.get_loss_index()->set_regularization_method("None");

        AdaptiveMomentEstimation* adam = dynamic_cast<AdaptiveMomentEstimation*>(training_strategy.get_optimization_algorithm());
        adam->set_maximum_epochs(100);
        adam->set_display_period(10);

#ifdef OPENNN_CUDA
    training_strategy.train_cuda();
#else
    training_strategy.train();
#endif

        // Testing analysis

        TestingAnalysis testing_analysis(&image_classification_network, &image_dataset);

        testing_analysis.set_batch_size(64);
        cout << "Calculating confusion...." << endl;
        const MatrixI confusion = testing_analysis.calculate_confusion_cuda();
        cout << "\nConfusion matrix:\n" << confusion << endl;
*/
        cout << "Bye!" << endl;
        

   
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
