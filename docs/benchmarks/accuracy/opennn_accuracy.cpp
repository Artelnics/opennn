//   OpenNN accuracy-parity benchmark on the Rosenbrock dataset (10 inputs).
//   Trains an MLP on the shared normalized train split and writes its test-set
//   predictions for the neutral scorer. Scaling/unscaling/bounding layers are
//   neutralized so the pre-normalized data is used as-is, matching PyTorch/TF.

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
    set_seed(seed);
    Configuration::instance().set(Device::Auto, Type::FP32);

    // Shared normalized data: 10 inputs + 1 target, comma-separated, no header.
    TabularDataset dataset("rosenbrock_train.csv", ",", false, false);

    // The constructor auto-splits 60/20/20 into Training/Validation/Testing.
    // PyTorch and TensorFlow train on the whole train file, so put every sample
    // in the Training role here too (no validation hold-out, no early stopping)
    // for an apples-to-apples comparison.
    dataset.set_sample_roles("Training");

    ApproximationNetwork network(dataset.get_input_shape(), {50, 50}, dataset.get_target_shape());

    // Neutralize scaling/unscaling/bounding: data is already normalized.
    static_cast<Scaling*>(network.get_first(LayerType::Scaling))->set_scalers("None");
    static_cast<Unscaling*>(network.get_first(LayerType::Unscaling))->set_scalers("None");
    static_cast<Bounding*>(network.get_first(LayerType::Bounding))->set_bounding_method("NoBounding");

    TrainingStrategy training_strategy(&network, &dataset);
    training_strategy.set_loss("MeanSquaredError");
    // OpenNN defaults to L2 regularization (weight 0.001); PyTorch/TF use none.
    // Disable it so all three minimize the same pure-MSE objective.
    training_strategy.get_loss()->set_regularization("NoRegularization");
    training_strategy.set_optimization_algorithm("AdaptiveMomentEstimation");

    auto* adam = dynamic_cast<AdaptiveMomentEstimation*>(training_strategy.get_optimization_algorithm());
    adam->set_maximum_epochs(argc > 2 ? Index(stoul(argv[2])) : 200);
    adam->set_batch_size(64);
    adam->set_display(false);

    // Match PyTorch/TF Adam: no gradient-norm clipping (theirs is off by default).
    adam->set_gradient_clip_norm(1.0e9f);
    if (argc > 3) adam->set_learning_rate(stof(argv[3]));

    const auto results = training_strategy.train();
    cerr << "final_training_error " << results.get_training_error() << "\n";

    // Predict on the shared test split and write one prediction per line.
    TabularDataset test("rosenbrock_test.csv", ",", false, false);
    const Index inputs_number = dataset.get_input_shape()[0];
    const MatrixR inputs = test.get_data().leftCols(inputs_number);

    const MatrixR outputs = network.calculate_outputs(inputs);

    ofstream out("pred_opennn.txt");
    out.precision(10);
    for (Index i = 0; i < outputs.rows(); ++i)
        out << outputs(i, 0) << "\n";

    return 0;
}
