//   OpenNN convergence-gate benchmark on the Rosenbrock dataset (10 inputs).
//
//   MLPerf-style metric: WALL-CLOCK TIME TO REACH A FIXED QUALITY TARGET, not
//   throughput at a fixed epoch count. Trains the shared MLP with a training-loss
//   goal (set_loss_goal); when the goal is hit, OpenNN stops and we report the
//   wall-clock time, the epochs taken, AND the held-out TEST MSE -- so a low
//   training loss that did not generalize cannot pass the gate.
//
//   This answers the reviewer question "are you fast because you do not actually
//   learn?": every engine must reach the same held-out MSE, and we time how long
//   that takes. Same data, arch, optimizer, and target as the PyTorch/TF drivers.
//
//   usage: opennn_convergence [seed] [target_mse] [max_epochs] [lr]

#include <chrono>
#include <iostream>

#include "opennn/tabular_dataset.h"
#include "opennn/standard_networks.h"
#include "opennn/scaling_layer.h"
#include "opennn/unscaling_layer.h"
#include "opennn/bounding_layer.h"
#include "opennn/training_strategy.h"
#include "opennn/adaptive_moment_estimation.h"
#include "opennn/random_utilities.h"
#include "opennn/configuration.h"

using namespace opennn;

int main(int argc, char* argv[])
{
    const unsigned seed       = argc > 1 ? unsigned(stoul(argv[1])) : 0;
    const float    target_mse = argc > 2 ? stof(argv[2])           : 0.05f;
    const Index    max_epochs = argc > 3 ? Index(stoul(argv[3]))   : 5000;
    const float    lr         = argc > 4 ? stof(argv[4])           : 1.0e-3f;

    set_seed(seed);
    Configuration::instance().set(Device::Auto, Type::FP32);

    // Shared normalized data (generate_rosenbrock.py): 10 inputs + 1 target.
    TabularDataset dataset("rosenbrock_train.csv", ",", false, false);
    dataset.set_sample_roles("Training");

    ApproximationNetwork network(dataset.get_input_shape(), {50, 50}, dataset.get_target_shape());
    static_cast<Scaling*>(network.get_first(LayerType::Scaling))->set_scalers("None");
    static_cast<Unscaling*>(network.get_first(LayerType::Unscaling))->set_scalers("None");
    static_cast<Bounding*>(network.get_first(LayerType::Bounding))->set_bounding_method("NoBounding");

    TrainingStrategy training_strategy(&network, &dataset);
    training_strategy.set_loss("MeanSquaredError");
    training_strategy.get_loss()->set_regularization("NoRegularization");
    training_strategy.set_optimization_algorithm("AdaptiveMomentEstimation");

    auto* adam = dynamic_cast<AdaptiveMomentEstimation*>(training_strategy.get_optimization_algorithm());
    adam->set_batch_size(64);
    adam->set_display(false);
    adam->set_gradient_clip_norm(1.0e9f);
    adam->set_learning_rate(lr);
    adam->set_loss_goal(0.0f);   // never stop on training loss; we gate on test MSE

    // The convergence gate is the HELD-OUT test MSE, not the training loss -- a
    // model that overfits the train split must not pass. Train in short chunks,
    // evaluate the test set after each, and stop the clock when the held-out MSE
    // reaches the target. The chunk's epochs are added to the running clock so
    // only training time is counted (evaluation is excluded).
    TabularDataset test("rosenbrock_test.csv", ",", false, false);
    const Index inputs_number = dataset.get_input_shape()[0];
    const MatrixR inputs  = test.get_data().leftCols(inputs_number);
    const MatrixR targets = test.get_data().rightCols(test.get_data().cols() - inputs_number);

    const Index chunk = 5;
    adam->set_maximum_epochs(chunk);

    bool reached = false;
    Index epochs = 0;
    float final_train_mse = NAN;
    double test_mse = NAN;
    double train_s = 0.0;

    while (epochs < max_epochs)
    {
        const auto t0 = chrono::steady_clock::now();
        const TrainingResult results = training_strategy.train();  // resumes from current params
        train_s += chrono::duration<double>(chrono::steady_clock::now() - t0).count();
        epochs += chunk;
        final_train_mse = results.get_training_error();

        const MatrixR outputs = network.calculate_outputs(inputs);
        test_mse = (outputs - targets).array().square().mean();
        if (test_mse <= target_mse) { reached = true; break; }
    }

    cout.precision(10);
    cout << "target_mse=" << target_mse << "\n";
    cout << "reached_goal=" << (reached ? 1 : 0) << "\n";
    cout << "epochs_to_target=" << epochs << "\n";
    cout << "final_train_mse=" << final_train_mse << "\n";
    cout << "test_mse=" << test_mse << "\n";
    cout << "time_to_target_s=" << train_s << "\n";
    cout << "RESULT=" << (reached ? "OK" : "DID_NOT_CONVERGE") << "\n";

    return reached ? 0 : 1;
}
