//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   B L A N K   C U D A
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
#include "../opennn/standard_networks.h"
#include "../opennn/vgg16.h"
#include "../opennn/training_strategy.h"
#include "../opennn/adaptive_moment_estimation.h"
#include "../opennn/stochastic_gradient_descent.h"
#include "../opennn/mean_squared_error.h"
#include "../opennn/testing_analysis.h"
#include "../opennn/image_dataset.h"
#include "../opennn/scaling_layer_4d.h"
#include "../opennn/convolutional_layer.h"
#include "../opennn/pooling_layer.h"
#include "../opennn/flatten_layer.h"
#include "../opennn/dense_layer.h"

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
        cout << "Enable CUDA in pch.h" << endl;
#endif

        const Index neurons_number = 3;

        // DataSet

        Dataset dataset("/mnt/c/Users/davidgonzalez/Documents/breast_cancer.csv", ";", true, false);

        dataset.split_samples_random(type(0.8), type(0), type(0.2));

        // Neural Network

        ApproximationNetwork approximation_network(dataset.get_input_dimensions(), { neurons_number }, dataset.get_target_dimensions());

        // Training strategy

        MeanSquaredError loss(&approximation_network, &dataset);
        loss.set_regularization_method("none");

        TrainingStrategy training_strategy(&approximation_network, &dataset);
        training_strategy.set_optimization_algorithm("AdaptiveMomentEstimation");
        //training_strategy.set_loss_index("MeanSquaredError");
        training_strategy.set_loss_index("CrossEntropyError2d");

        AdaptiveMomentEstimation* adam = dynamic_cast<AdaptiveMomentEstimation*>(training_strategy.get_optimization_algorithm());
        adam->set_maximum_epochs_number(4000);

        TrainingResults training_results = training_strategy.train_cuda();

        // Testing analysis

        TestingAnalysis testing_analysis(&approximation_network, &dataset);
        //testing_analysis.print_goodness_of_fit_analysis();
        cout << testing_analysis.calculate_confusion() << endl;


        cout << "Bye!" << endl;

        return 0;
    }
    catch (const exception& e)
    {
        cerr << e.what() << endl;

        return 1;
    }
}
