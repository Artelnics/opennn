//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   A I R F O I L   S E L F   N O I S E   A P P L I C A T I O N
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include <iostream>
#include <string>

#include "../../opennn/tabular_dataset.h"
#include "../../opennn/standard_networks.h"
#include "../../opennn/bounding_layer.h"
#include "../../opennn/training_strategy.h"
#include "../../opennn/testing_analysis.h"
#include "../../opennn/model_selection.h"
#include "../../opennn/testing_analysis.h"
#include "../../opennn/optimizer.h"
#include "../../opennn/stochastic_gradient_descent.h"
#include "../../opennn/random_utilities.h"

using namespace opennn;

int main()
{
    try
    {
        cout << "Airfoil self noise" << endl;

        set_seed(42);

        Configuration::instance().set(Device::Auto, Type::FP32);

        const Index neurons_number = 12;
        const float regularization_weight = float(0.001);

        // DataSet

        TabularDataset dataset("../data/airfoil_self_noise.csv", ";", true, false);

        dataset.split_samples_random(float(0.8), float(0.0), float(0.2));

        // Neural Network

        ApproximationNetwork approximation_network(dataset.get_input_shape(), {neurons_number}, dataset.get_target_shape());

        Bounding* bounding_layer = (Bounding*)approximation_network.get_first("Bounding");

        if(bounding_layer)
            bounding_layer->set_bounding_method("NoBounding");

        // Training strategy

        TrainingStrategy training_strategy(&approximation_network, &dataset);

        training_strategy.set_loss("MeanSquaredError");
        training_strategy.get_loss()->set_regularization("L2");
        training_strategy.get_loss()->set_regularization_weight(regularization_weight);

        training_strategy.set_optimization_algorithm("StochasticGradientDescent");
        StochasticGradientDescent* sgd = dynamic_cast<StochasticGradientDescent*>(training_strategy.get_optimization_algorithm());
        sgd->set_initial_learning_rate(float(0.3));
        sgd->set_display_period(50);


        TrainingResult training_results = training_strategy.train();

        // Testing analysis

        TestingAnalysis testing_analysis(&approximation_network, &dataset);
        testing_analysis.print_goodness_of_fit_analysis();

        cout << "Good bye!" << endl;

        return 0;
    }
    catch(exception& e)
    {
        cerr << e.what() << endl;

        return 1;
    }
}


// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
