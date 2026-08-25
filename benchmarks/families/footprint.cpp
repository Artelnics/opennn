// What OpenNN costs before it does any work.
//
// PLAN.md. The one part of the suite that cannot ride along on a training run:
// speed, peak memory and energy are all readings of a run in progress, whereas
// these three ask what the framework costs merely by existing.
//
//   footprint memory    resident set and GPU-ready VRAM after empty objects
//   footprint startup   time to first prediction
//   footprint export    train a small model, write it as dependency-free source
//
// Each mode is one process that does its thing and exits, because that is the
// only honest way to measure a cost paid at startup: anything sharing a
// process with another mode has already paid it.
//
// Note what `startup` can and cannot see. It times from main(), so it counts
// library initialisation, construction and the forward pass, but not the
// dynamic loader -- and for an engine whose shared objects run to hundreds of
// megabytes the loader is most of the answer. run.py records whole-process
// wall time alongside, which is the figure to compare.

#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

#ifdef __linux__
#include <unistd.h>
#endif

#include "opennn/core/configuration.h"
#include "opennn/core/random_utilities.h"
#include "opennn/dataset/tabular_dataset.h"
#include "opennn/neural_network/model_expression.h"
#include "opennn/neural_network/neural_network.h"
#include "opennn/neural_network/standard_networks.h"
#include "opennn/training_strategy/adaptive_moment_estimation.h"
#include "opennn/training_strategy/training_strategy.h"

using namespace opennn;
using clock_type = chrono::steady_clock;

namespace
{

double resident_mb()
{
#ifdef __linux__
    // statm reports pages; the second field is resident.
    ifstream statm("/proc/self/statm");
    long total = 0, resident = 0;
    if (statm >> total >> resident)
        return double(resident) * double(sysconf(_SC_PAGESIZE)) / (1024.0 * 1024.0);
#endif
    return 0.0;
}

int usage()
{
    cerr << "usage: footprint memory | startup | export\n";
    return 2;
}

}   // namespace

int main(int argc, char* argv[])
{
    const string mode = argc > 1 ? argv[1] : "";
    const auto entered = clock_type::now();

    cout << "engine=opennn\nmode=" << mode << "\n";

    if (mode == "memory")
    {
        // Empty objects only: a network with no layers, a dataset with no
        // data, a training strategy over both. This is the floor -- what an
        // application pays for linking the library and declaring intent,
        // before a single sample is loaded.
        Configuration::instance().set(Device::Auto, Type::FP32);

        NeuralNetwork network;
        TabularDataset dataset;
        TrainingStrategy strategy(&network, &dataset);

        cout << "baseline_ram_mb=" << resident_mb() << "\n";
    }
    else if (mode == "startup")
    {
        Configuration::instance().set(Device::Auto, Type::FP32);

        ApproximationNetwork network({10}, {64}, {1});

        MatrixR input(1, 10);
        input.setOnes();

        const MatrixR output = network.calculate_outputs(input);

        const double seconds =
            chrono::duration<double>(clock_type::now() - entered).count();

        cout << "prediction=" << output(0, 0) << "\n"
             << "first_prediction_s=" << seconds << "\n";
    }
    else if (mode == "export")
    {
        set_seed(42);
        Configuration::instance().set(Device::CPU, Type::FP32);

        // Its own data, so footprint needs no dataset prepared for it: three
        // inputs whose sum is the target, which is enough to train something
        // worth exporting. Everything lands in a temporary directory: these
        // are run outputs, and a benchmark that litters the tree it is run
        // from will eventually have one of those files committed.
        const filesystem::path scratch =
            filesystem::temp_directory_path() / "opennn_footprint";
        filesystem::create_directories(scratch);

        const string csv_path = (scratch / "sum.csv").string();
        const string c_path = (scratch / "model.c").string();
        const string python_path = (scratch / "model.py").string();

        {
            ofstream csv(csv_path);
            for (int row = 0; row < 512; ++row)
            {
                const float a = float(row % 17) / 17.0f;
                const float b = float(row % 7) / 7.0f;
                const float c = float(row % 3) / 3.0f;
                csv << a << ";" << b << ";" << c << ";" << (a + b + c) << "\n";
            }
        }

        TabularDataset dataset(csv_path, ";", false, false);

        ApproximationNetwork network(dataset.get_input_shape(), {64},
                                     dataset.get_target_shape());

        TrainingStrategy strategy(&network, &dataset);
        strategy.set_loss("MeanSquaredError");
        strategy.set_optimization_algorithm("AdaptiveMomentEstimation");

        auto* adam = dynamic_cast<AdaptiveMomentEstimation*>(
            strategy.get_optimization_algorithm());
        adam->set_maximum_epochs(50);
        adam->set_batch_size(32);
        adam->set_display(false);

        strategy.train();

        ModelExpression expression(&network);
        expression.save(c_path, ModelExpression::ProgrammingLanguage::C);
        expression.save(python_path, ModelExpression::ProgrammingLanguage::Python);

        const auto size_of = [](const string& path) -> long
        {
            ifstream file(path, ios::binary | ios::ate);
            return file ? long(file.tellg()) : -1;
        };

        // The exported file is the deliverable: source with no runtime
        // dependency, which is the whole claim being measured.
        cout << "export_c_bytes=" << size_of(c_path) << "\n"
             << "export_python_bytes=" << size_of(python_path) << "\n";
    }
    else
    {
        return usage();
    }

    cout << "RESULT=OK\n";

    return 0;
}
