//   OpenNN precision benchmark on the Rosenbrock dataset (10 inputs), after
//   the Neural Designer blog protocol: 10 -> 10 (tanh) -> 1 (linear), weights
//   initialized U(-1, 1), MSE loss, train on all 10,000 samples, no split.
//   The optimizer is selectable so the blog's two configurations can both run:
//   second-order full-batch (QuasiNewtonMethod, 1000 epochs) and Adam
//   (batch 1000, 10000 epochs). Prints the training wall time and writes
//   full-dataset predictions for the neutral scorer.
//
//   usage: opennn_precision <seed> <optimizer> <epochs>

#include <chrono>
#include <fstream>
#include <iostream>

#include "../../../opennn/tabular_dataset.h"
#include "../../../opennn/standard_networks.h"
#include "../../../opennn/scaling_layer.h"
#include "../../../opennn/unscaling_layer.h"
#include "../../../opennn/bounding_layer.h"
#include "../../../opennn/training_strategy.h"
#include "../../../opennn/adaptive_moment_estimation.h"
#include "../../../opennn/random_utilities.h"
#include "../../../opennn/configuration.h"

using namespace opennn;

int main(int argc, char* argv[])
{
    const unsigned seed = argc > 1 ? unsigned(stoul(argv[1])) : 0;
    const string optimizer_name = argc > 2 ? argv[2] : "QuasiNewtonMethod";
    const Index epochs = argc > 3 ? Index(stoul(argv[3])) : 1000;

    set_seed(seed);
    Configuration::instance().set(Device::CPU, Type::FP32);

    // Shared normalized data: 10 inputs + 1 target, comma-separated, no header.
    TabularDataset dataset("rosenbrock.csv", ",", false, false);

    // The blog trains on the full dataset: no validation or test hold-out.
    dataset.set_sample_roles("Training");

    ApproximationNetwork network(dataset.get_input_shape(), {10}, dataset.get_target_shape());

    // Neutralize scaling/unscaling/bounding: data is already normalized.
    static_cast<Scaling*>(network.get_first(LayerType::Scaling))->set_scalers("None");
    static_cast<Unscaling*>(network.get_first(LayerType::Unscaling))->set_scalers("None");
    static_cast<Bounding*>(network.get_first(LayerType::Bounding))->set_bounding_method("NoBounding");

    // Blog initialization: all parameters U(-1, 1).
    VectorMap parameters(network.get_parameters_data(), network.get_parameters_size());
    set_random_uniform(parameters, -1.0f, 1.0f);

    TrainingStrategy training_strategy(&network, &dataset);
    training_strategy.set_loss("MeanSquaredError");
    training_strategy.get_loss()->set_regularization("NoRegularization");
    training_strategy.set_optimization_algorithm(optimizer_name);

    Optimizer* optimizer = training_strategy.get_optimization_algorithm();
    optimizer->set_maximum_epochs(epochs);
    optimizer->set_display(false);

    if (auto* adam = dynamic_cast<AdaptiveMomentEstimation*>(optimizer))
    {
        // Blog Adam configuration (same as the PyTorch/TensorFlow scripts).
        adam->set_batch_size(1000);
        adam->set_learning_rate(0.001f);
        adam->set_gradient_clip_norm(1.0e9f);
    }

    const auto start = chrono::steady_clock::now();
    const TrainingResult results = training_strategy.train();
    const auto stop = chrono::steady_clock::now();

    cout << "train_time=" << chrono::duration<double>(stop - start).count() << "\n";
    cout << "epochs_run=" << results.get_epochs_number() << "\n";
    cerr << "final_training_error " << results.get_training_error() << "\n";

    const Index inputs_number = dataset.get_input_shape()[0];
    const MatrixR inputs = dataset.get_data().leftCols(inputs_number);

    const MatrixR outputs = network.calculate_outputs(inputs);

    ofstream out("pred_opennn.txt");
    out.precision(10);
    for (Index i = 0; i < outputs.rows(); ++i)
        out << outputs(i, 0) << "\n";

    return 0;
}
