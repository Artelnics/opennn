//   Single max-batch TRIAL for the shallow Rosenbrock MLP in OpenNN.
//
//   One invocation = one batch attempt, in its own process so a CUDA fault
//   (out-of-memory can leave the context corrupted with a sticky error 700)
//   cannot contaminate later trials. The bash driver (run_maxbatch.sh) does the
//   exponential-grow + binary-search by spawning this repeatedly.
//
//   Net: inputs -> hidden (tanh) -> 1, MSE, Adam, fp32.
//   Exit 0 = the batch fit. Exit 1 = it failed (OOM / CUDA error).
//
//   usage: opennn_rosenbrock_trial <inference|train> <batch> [inputs] [hidden]

#include <iostream>
#include <string>

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

static void neutralize(ApproximationNetwork& network)
{
    static_cast<Scaling*>(network.get_first(LayerType::Scaling))->set_scalers("None");
    static_cast<Unscaling*>(network.get_first(LayerType::Unscaling))->set_scalers("None");
    static_cast<Bounding*>(network.get_first(LayerType::Bounding))->set_bounding_method("NoBounding");
}

int main(int argc, char* argv[])
{
    std::cout << std::unitbuf;
    std::cerr << std::unitbuf;

    const std::string mode = argc > 1 ? argv[1] : "train";
    const Index batch = argc > 2 ? Index(std::stoll(argv[2])) : 1024;
    const Index inputs = argc > 3 ? Index(std::stoll(argv[3])) : 1000;
    const Index hidden = argc > 4 ? Index(std::stoll(argv[4])) : 1000;

    try
    {
        set_seed(0);
        Configuration::instance().set(Device::CUDA, Type::FP32);

        if (mode == "inference")
        {
            ApproximationNetwork network(Shape{inputs}, {hidden}, Shape{1});
            neutralize(network);
            const MatrixR in = MatrixR::Random(batch, inputs);
            const MatrixR out = network.calculate_outputs(in);
            if (out.rows() != batch) { std::cerr << "bad output rows\n"; return 1; }
        }
        else
        {
            TabularDataset dataset(batch, Shape{inputs}, Shape{1});
            dataset.set_data_random();
            dataset.set_sample_roles("Training");

            ApproximationNetwork network(Shape{inputs}, {hidden}, Shape{1});
            neutralize(network);

            TrainingStrategy training_strategy(&network, &dataset);
            training_strategy.set_loss("MeanSquaredError");
            training_strategy.get_loss()->set_regularization("NoRegularization");
            training_strategy.set_optimization_algorithm("AdaptiveMomentEstimation");

            auto* adam = dynamic_cast<AdaptiveMomentEstimation*>(
                training_strategy.get_optimization_algorithm());
            adam->set_batch_size(batch);
            adam->set_maximum_epochs(1);
            adam->set_display(false);
            adam->set_gradient_clip_norm(0.0f);

            training_strategy.train();
        }

        std::cout << "FIT batch=" << batch << "\n";
        return 0;
    }
    catch (const std::exception& e)
    {
        std::cerr << "FAIL batch=" << batch << " : " << e.what() << "\n";
        return 1;
    }
}
