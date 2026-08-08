//   OpenNN data-capacity benchmark (HIGGS rows).
//
//   Loads a headerless HIGGS CSV into a TabularDataset (the default, in-RAM
//   float32 matrix), builds a small dense net (28 -> hidden -> 1), and runs a
//   short Adam training so the batch buffers are actually allocated. Prints the
//   process peak working set (resident memory).
//
//   The CSV is the prepared HIGGS training file (28 features + 1 label per row)
//   tiled up to the sweep's target sample count by tile_higgs.exe, so every
//   value is a real HIGGS row rather than synthetic data. See
//   ../../throughput/higgs/README.md for the dataset contract.
//
//   A "successful" run loads the file and trains; if the dataset does not fit
//   in RAM the allocation throws bad_alloc (caught -> exit 1) or the OS
//   terminates the process. The driver script sweeps sample counts to find the
//   largest one that still succeeds.
//
//   usage:  opennn_capacity <csv_path> [hidden_neurons]

#include <iostream>
#include <string>

#include "opennn/tabular_dataset.h"
#include "opennn/standard_networks.h"
#include "opennn/training_strategy.h"
#include "opennn/optimizer.h"
#include "opennn/adaptive_moment_estimation.h"
#include "opennn/random_utilities.h"

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <psapi.h>

using namespace opennn;

static double peak_working_set_mb()
{
    PROCESS_MEMORY_COUNTERS pmc;
    if (GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc)))
        return double(pmc.PeakWorkingSetSize) / (1024.0 * 1024.0);
    return -1.0;
}

static double current_working_set_mb()
{
    PROCESS_MEMORY_COUNTERS pmc;
    if (GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc)))
        return double(pmc.WorkingSetSize) / (1024.0 * 1024.0);
    return -1.0;
}

int main(int argc, char* argv[])
{
    try
    {
        if (argc < 2)
        {
            cerr << "usage: opennn_capacity <csv_path> [hidden_neurons]\n";
            return 2;
        }

        const string csv_path = argv[1];
        const Index hidden_neurons = (argc > 2) ? Index(stoll(argv[2])) : Index(1024);

        set_seed(42);
        Configuration::instance().set(Device::Auto, Type::FP32);

        TabularDataset dataset(csv_path, ",", false, false);

        const Index samples = dataset.get_samples_number();
        cout << "loaded_samples=" << samples << "\n";
        cout << "sustained_after_load_mb=" << current_working_set_mb() << "\n";
        cout << "after_load_peak_mb=" << peak_working_set_mb() << "\n";

        dataset.split_samples_random(1.0f, 0.0f, 0.0f);

        ApproximationNetwork network(dataset.get_input_shape(),
                                     {hidden_neurons},
                                     dataset.get_target_shape());

        TrainingStrategy training_strategy(&network, &dataset);
        training_strategy.set_loss("MeanSquaredError");
        training_strategy.set_optimization_algorithm("AdaptiveMomentEstimation");

        auto* adam = dynamic_cast<AdaptiveMomentEstimation*>(
            training_strategy.get_optimization_algorithm());
        adam->set_batch_size(1000);
        adam->set_maximum_epochs(1);
        adam->set_display_period(1);

        training_strategy.train();

        cout << "trained=1\n";
        cout << "peak_mb=" << peak_working_set_mb() << "\n";
        cout << "RESULT=OK\n";
        return 0;
    }
    catch (const bad_alloc&)
    {
        cout << "peak_mb=" << peak_working_set_mb() << "\n";
        cout << "RESULT=OOM\n";
        return 1;
    }
    catch (const exception& e)
    {
        cerr << e.what() << "\n";
        cout << "peak_mb=" << peak_working_set_mb() << "\n";
        cout << "RESULT=ERROR\n";
        return 1;
    }
}
