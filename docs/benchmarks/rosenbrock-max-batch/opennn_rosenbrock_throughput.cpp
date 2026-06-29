//   Throughput vs prefetch-pool depth for the shallow Rosenbrock MLP (OpenNN).
//
//   Unlike the max-batch trial (one batch == whole set, no overlap to measure),
//   this uses a dataset MUCH larger than the batch so there are many steps per
//   epoch -- the regime where the prefetch pool's overlap of batch-prep with GPU
//   compute actually matters. Times only the training loop (CUDA init excluded)
//   and reports samples/sec, so pool=1 vs 2 vs 3 can be compared fairly.
//
//   Train data is GPU-resident (synthetic in-memory dataset) -- "preparing a batch"
//   is just an index gather, the cheap case. Residency is enabled in this code (no
//   environment switch). Vary the prefetch-pool depth with the [batch_pool] argument.
//
//   The CUDA mega-graph is on by default and can be toggled with the optional
//   [cuda_graph 0|1] argument, so a run can compare graph on/off without env vars.
//
//   usage: opennn_rosenbrock_throughput [mode] [samples] [batch] [iters] [inputs] [hidden] [cuda_graph 0|1] [batch_pool]
//          mode = train | inference

#include <chrono>
#include <cstdlib>
#include <iostream>
#include <string>

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
    std::cout << std::unitbuf;

    const std::string mode = argc > 1 ? argv[1] : "train";
    const Index samples = argc > 2 ? Index(std::stoll(argv[2])) : 200000;
    const Index batch   = argc > 3 ? Index(std::stoll(argv[3])) : 2000;
    const Index iters   = argc > 4 ? Index(std::stoll(argv[4])) : 10;
    const Index inputs  = argc > 5 ? Index(std::stoll(argv[5])) : 1000;
    const Index hidden  = argc > 6 ? Index(std::stoll(argv[6])) : 1000;
    const bool cuda_graph = argc > 7 ? (std::stoi(argv[7]) != 0) : true;
    const int  batch_pool = argc > 8 ? std::stoi(argv[8]) : 0;   // 0 = library default

    try
    {
        set_seed(0);
        const bool use_bf16 = std::getenv("OPENNN_BF16") != nullptr;
        Configuration::instance().set(Device::CUDA, use_bf16 ? Type::BF16 : Type::FP32);

        ApproximationNetwork network(Shape{inputs}, {hidden}, Shape{1});
        static_cast<Scaling*>(network.get_first(LayerType::Scaling))->set_scalers("None");
        static_cast<Unscaling*>(network.get_first(LayerType::Unscaling))->set_scalers("None");
        static_cast<Bounding*>(network.get_first(LayerType::Bounding))->set_bounding_method("NoBounding");

        if (mode == "inference")
        {
            // Steady-state inference throughput: repeat one forward at fixed batch.
            const MatrixR in = MatrixR::Random(batch, inputs);
            network.calculate_outputs(in);                       // warmup
            const auto t0 = std::chrono::steady_clock::now();
            for (Index it = 0; it < iters; ++it)
                network.calculate_outputs(in);
            const auto t1 = std::chrono::steady_clock::now();
            const double per = std::chrono::duration<double>(t1 - t0).count() / double(iters);
            std::cout << "mode=inference batch=" << batch << "\n";
            std::cout << "step_s=" << per << "\n";
            std::cout << "samples_per_sec=" << long(double(batch) / per) << "\n";
            return 0;
        }

        TabularDataset dataset(samples, Shape{inputs}, Shape{1});
        dataset.set_data_random();
        dataset.set_sample_roles("Training");
        // Keep the synthetic set GPU-resident so a batch is just a device-side
        // gather (the regime this throughput benchmark measures). Enabled in code.
        dataset.set_storage_mode(Dataset::StorageMode::GPUPersistantData);

        TrainingStrategy training_strategy(&network, &dataset);
        training_strategy.set_loss("MeanSquaredError");
        training_strategy.get_loss()->set_regularization("NoRegularization");
        training_strategy.set_optimization_algorithm("AdaptiveMomentEstimation");

        auto* adam = dynamic_cast<AdaptiveMomentEstimation*>(
            training_strategy.get_optimization_algorithm());
        adam->set_batch_size(batch);
        adam->set_display(false);
        adam->set_gradient_clip_norm(0.0f);
        adam->set_cuda_graph(cuda_graph);
        adam->set_batch_pool_size(batch_pool);   // 0 = library default depth

        // Warmup epoch (autotune, allocations) excluded from timing.
        adam->set_maximum_epochs(1);
        training_strategy.train();

        adam->set_maximum_epochs(iters);
        const auto t0 = std::chrono::steady_clock::now();
        training_strategy.train();
        const auto t1 = std::chrono::steady_clock::now();

        const double s = std::chrono::duration<double>(t1 - t0).count();
        const double per_epoch = s / double(iters);
        std::cout << "mode=train samples=" << samples << " batch=" << batch
                  << " precision=" << (use_bf16 ? "bf16" : "fp32")
                  << " cuda_graph=" << cuda_graph
                  << " steps_per_epoch=" << (samples / batch) << "\n";
        std::cout << "epoch_s=" << per_epoch << "\n";
        std::cout << "samples_per_sec=" << long(double(samples) / per_epoch) << "\n";
        return 0;
    }
    catch (const std::exception& e)
    {
        std::cerr << "FAIL: " << e.what() << "\n";
        return 1;
    }
}
