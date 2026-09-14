//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   Y A C H T   H Y D R O D Y N A M I C S   A P P L I C A T I O N
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

// Residuary resistance of sailing yachts from the Delft series towing-tank
// experiments: select the hidden-layer width, train it with the quasi-Newton
// method and assess the test samples with a linear regression.

#include <iomanip>
#include <iostream>
#include <limits>

#include "opennn/core/configuration.h"
#include "opennn/core/device_backend.h"
#include "opennn/core/random_utilities.h"
#include "opennn/dataset/correlations.h"
#include "opennn/dataset/tabular_dataset.h"
#include "opennn/evaluation/evaluation.h"
#include "opennn/model_selection/growing_neurons.h"
#include "opennn/models/models.h"
#include "opennn/training/quasi_newton.h"
#include "opennn/training/training.h"

using namespace opennn;

namespace
{

// Normalized squared error without regularization, minimized with the quasi-Newton
// method until the loss decrease falls below 1e-12. There is no early stopping,
// so the trained parameters are those of the last epoch.

void configure_training(Training& training)
{
    training.set_loss("NormalizedSquaredError");
    training.get_loss()->set_regularization(Loss::Regularization::NoRegularization);

    training.set_optimization_algorithm("QuasiNewton");

    auto* quasi_newton = dynamic_cast<QuasiNewton*>(training.get_optimization_algorithm());

    quasi_newton->set_minimum_loss_decrease(1.0e-12f);
    quasi_newton->set_loss_goal(0.0f);
    quasi_newton->set_maximum_epochs(10000);
    quasi_newton->set_maximum_time(3600.0f);
    quasi_newton->set_maximum_validation_failures(numeric_limits<Index>::max());
    quasi_newton->set_restore_best(false);
    quasi_newton->set_display(false);
}

}

int main()
{
    try
    {
        cout << "OpenNN. Yacht Hydrodynamics Example." << endl;

        // The quasi-Newton method runs on the CPU. One thread keeps the run repeatable.

        Configuration::instance().set(Device::CPU, Type::FP32);
        Configuration::instance().set_blas(Blas::Eigen);
        set_threads_number(1);
        set_seed(1);

        // 308 experiments: six hull-shape and speed variables and the residuary resistance.
        // Every variable is standardized with the training statistics.

        TabularDataset dataset("../data/yacht_hydrodynamics/yacht_hydrodynamics.csv", ";", true, false);
        dataset.set_display(false);
        dataset.set_variable_scalers("MeanStandardDeviation");
        dataset.split_samples_random(0.50f, 0.25f, 0.25f);

        // Model selection: 6, 9 and 12 hidden neurons, ranked by the validation error.

        ApproximationNetwork selection_network(dataset.get_input_shape(), {6}, dataset.get_target_shape());

        Training selection_training(&selection_network, &dataset);
        configure_training(selection_training);

        GrowingNeurons growing_neurons(&selection_training);
        growing_neurons.set_minimum_neurons(6);
        growing_neurons.set_neurons_increment(3);
        growing_neurons.set_maximum_neurons(12);
        growing_neurons.set_trials_number(1);
        growing_neurons.set_warm_start(false);
        growing_neurons.set_display(false);

        const NeuronsSelectionResult selection = growing_neurons.perform_neurons_selection();

        cout << "Hidden neurons  Training error  Validation error" << endl;

        for (Index i = 0; i < selection.neurons_number_history.size(); ++i)
            cout << setw(14) << selection.neurons_number_history(i)
                 << setw(16) << selection.training_error_history(i)
                 << setw(18) << selection.validation_error_history(i) << endl;

        const Index hidden_neurons = selection.optimal_neurons_number;

        cout << "Selected hidden neurons: " << hidden_neurons << endl;

        // Final training of the selected network: tanh hidden layer, linear output.

        ApproximationNetwork network(dataset.get_input_shape(), {hidden_neurons}, dataset.get_target_shape());
        network.set_parameters_random();

        Training training(&network, &dataset);
        configure_training(training);

        const TrainingResult training_result = training.train();

        cout << "Epochs: " << training_result.get_epochs_number() << endl
             << "Stopping condition: " << training_result.write_stopping_condition() << endl;

        // Testing: linear regression of the outputs on the targets.

        Evaluation evaluation(&network, &dataset);

        const Evaluation::GoodnessOfFitAnalysis goodness_of_fit = evaluation.perform_goodness_of_fit_analysis()(0);
        const Correlation regression = linear_correlation(goodness_of_fit.targets, goodness_of_fit.outputs);

        cout << "Test R2: " << goodness_of_fit.determination << endl
             << "Intercept: " << regression.intercept << endl
             << "Slope: " << regression.slope << endl;

        cout << "Good bye!" << endl;

        return 0;
    }
    catch (const exception& e)
    {
        cerr << e.what() << endl;

        return 1;
    }
}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
