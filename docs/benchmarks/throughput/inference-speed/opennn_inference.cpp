//   OpenNN CPU inference-speed benchmark.
//
//   Mirrors the training-speed benchmark's model so the two read together: a
//   2-layer MLP (F -> F -> 1, tanh then linear) on the Rosenbrock dataset. Here
//   the network only does inference: forward_propagate(..., is_training=false)
//   runs a pure forward pass (dropout is skipped and no gradients are built).
//
//   Reports median seconds per full pass over the dataset, samples/second
//   (throughput), and milliseconds per batch (latency).
//
//   The optional [fast_vml 0|1] argument turns on MKL's enhanced-performance VML
//   tanh mode (set in code via cpu_math::set_mkl_fast_vml; no environment var).
//
//   usage:  opennn_inference <csv_path> <features> [batch] [reps] [fast_vml 0|1]

#include <algorithm>
#include <chrono>
#include <iostream>
#include <string>
#include <vector>

#include "opennn/tabular_dataset.h"
#include "opennn/neural_network.h"
#include "opennn/dense_layer.h"
#include "opennn/forward_propagation.h"
#include "opennn/configuration.h"
#include "opennn/cpu_math_backend.h"
#include "opennn/random_utilities.h"

using namespace opennn;
using clock_type = std::chrono::steady_clock;

int main(int argc, char* argv[])
{
    try
    {
        if (argc < 3)
        {
            std::cerr << "usage: opennn_inference <csv_path> <features> [batch] [reps] [fast_vml 0|1]\n";
            return 2;
        }

        const std::string csv_path = argv[1];
        const Index features = Index(std::stoll(argv[2]));
        const Index batch = (argc > 3) ? Index(std::stoll(argv[3])) : 1000;
        const Index reps = (argc > 4) ? Index(std::stoll(argv[4])) : 30;
        const bool fast_vml = (argc > 5) ? (std::stoi(argv[5]) != 0) : false;

        set_seed(42);
        Configuration::instance().set(Device::CPU, Type::FP32);

        // MKL enhanced-performance VML tanh (optional), set in code (no env var).
        if (fast_vml) set_mkl_fast_vml(true);

        TabularDataset dataset(csv_path, ",", false, false);
        dataset.split_samples_random(1.0f, 0.0f, 0.0f);
        const Index samples = dataset.get_samples_number();

        const Index inputs_number = dataset.get_input_shape()[0];
        const MatrixR data = dataset.get_data().leftCols(inputs_number);

        std::cout << "samples=" << samples << " features=" << features
                  << " batch=" << batch << " reps=" << reps << "\n";

        NeuralNetwork network;
        network.add_layer(make_unique<opennn::Dense>(dataset.get_input_shape(),
                                                     Shape{features},
                                                     "Tanh",
                                                     false,
                                                     "hidden_layer"));
        network.add_layer(make_unique<opennn::Dense>(Shape{features},
                                                     dataset.get_target_shape(),
                                                     "Identity",
                                                     false,
                                                     "output_layer"));
        network.compile();
        network.set_parameters_glorot();

        ForwardPropagation forward_propagation(batch, &network);

        const std::vector<TensorView> batches = [&]
        {
            std::vector<TensorView> views;
            views.reserve(size_t(samples / batch));
            for (Index i = 0; i + batch <= samples; i += batch)
            {
                float* batch_data = const_cast<float*>(data.data()) + i * inputs_number;
                views.emplace_back(batch_data, Shape{batch, inputs_number}, Type::FP32);
            }
            return views;
        }();

        double sink = 0.0;
        auto run_pass = [&]
        {
            for (const TensorView& inputs : batches)
            {
                network.forward_propagate({inputs}, forward_propagation, false);
                const MatrixMap outputs = forward_propagation.get_outputs().as_matrix();
                sink += double(outputs(0, 0));  // consume the result so it isn't optimized away
            }
        };

        // Warmup: first passes absorb one-time buffer allocation.
        run_pass();
        run_pass();

        const Index batched_samples = Index(batches.size()) * batch;
        std::vector<double> times;
        times.reserve(reps);
        for (Index r = 0; r < reps; ++r)
        {
            const auto t0 = clock_type::now();
            run_pass();
            const auto t1 = clock_type::now();
            times.push_back(std::chrono::duration<double>(t1 - t0).count());
        }

        std::sort(times.begin(), times.end());
        const double median = times[times.size() / 2];
        const double samples_per_sec = double(batched_samples) / median;
        const double ms_per_batch = median / double(batches.size()) * 1000.0;

        std::cerr << "checksum=" << sink << "\n";  // keep run_pass from being elided

        std::cout << "median_pass_s=" << median << "\n";
        std::cout << "samples_per_sec=" << long(samples_per_sec) << "\n";
        std::cout << "ms_per_batch=" << ms_per_batch << "\n";
        std::cout << "RESULT=OK\n";
        return 0;
    }
    catch (const std::exception& e)
    {
        std::cerr << e.what() << "\n";
        std::cout << "RESULT=ERROR\n";
        return 1;
    }
}
