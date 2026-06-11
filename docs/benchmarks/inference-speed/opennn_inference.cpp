//   OpenNN CPU inference-speed benchmark.
//
//   Mirrors the training-speed benchmark's model so the two read together: a
//   2-layer MLP (F -> F -> 1, tanh then linear) on the Rosenbrock dataset. Here
//   the network only does inference: calculate_outputs() runs a pure forward
//   pass (is_training=false, so dropout is skipped and no gradients are built).
//
//   Reports median seconds per full pass over the dataset, samples/second
//   (throughput), and milliseconds per batch (latency).
//
//   usage:  opennn_inference <csv_path> <features> [batch] [reps]

#include <algorithm>
#include <chrono>
#include <iostream>
#include <string>
#include <vector>

#include "../../../opennn/tabular_dataset.h"
#include "../../../opennn/standard_networks.h"
#include "../../../opennn/configuration.h"
#include "../../../opennn/random_utilities.h"

using namespace opennn;
using clock_type = std::chrono::steady_clock;

int main(int argc, char* argv[])
{
    try
    {
        if (argc < 3)
        {
            std::cerr << "usage: opennn_inference <csv_path> <features> [batch] [reps]\n";
            return 2;
        }

        const std::string csv_path = argv[1];
        const Index features = Index(std::stoll(argv[2]));
        const Index batch = (argc > 3) ? Index(std::stoll(argv[3])) : 1000;
        const Index reps = (argc > 4) ? Index(std::stoll(argv[4])) : 30;

        set_seed(42);
        Configuration::instance().set(Device::CPU, Type::FP32);

        TabularDataset dataset(csv_path, ",", false, false);
        dataset.split_samples_random(1.0f, 0.0f, 0.0f);
        const Index samples = dataset.get_samples_number();

        const Index inputs_number = dataset.get_input_shape()[0];
        const MatrixR data = dataset.get_data().leftCols(inputs_number);

        std::cout << "samples=" << samples << " features=" << features
                  << " batch=" << batch << " reps=" << reps << "\n";

        ApproximationNetwork network(dataset.get_input_shape(),
                                     {features},
                                     dataset.get_target_shape());

        const std::vector<Index> starts = [&]
        {
            std::vector<Index> s;
            for (Index i = 0; i + batch <= samples; i += batch) s.push_back(i);
            return s;
        }();

        double sink = 0.0;
        auto run_pass = [&]
        {
            for (const Index s : starts)
            {
                const MatrixR inputs = data.middleRows(s, batch);
                const MatrixR outputs = network.calculate_outputs(inputs);
                sink += double(outputs(0, 0));  // consume the result so it isn't optimized away
            }
        };

        // Warmup: first passes absorb one-time buffer allocation.
        run_pass();
        run_pass();

        const Index batched_samples = Index(starts.size()) * batch;
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
        const double ms_per_batch = median / double(starts.size()) * 1000.0;

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
