//   OpenNN data-capacity benchmark.
//
//   Loads a headerless Rosenbrock CSV into a TabularDataset (the default,
//   in-RAM float32 matrix), builds the article's network (N -> N -> 1, tanh),
//   and runs a short Adam training so the batch buffers are actually
//   allocated. Prints the process peak working set (resident memory).
//
//   A "successful" run loads the file and trains; if the dataset does not fit
//   in RAM the allocation throws std::bad_alloc (caught -> exit 1) or the OS
//   terminates the process. The driver script sweeps file sizes to find the
//   largest one that still succeeds.
//
//   usage:  opennn_capacity <csv_path> <input_variables>

#include <iostream>
#include <string>

#include "opennn/tabular_dataset.h"
#include "opennn/standard_networks.h"
#include "opennn/training_strategy.h"
#include "opennn/optimizer.h"
#include "opennn/adaptive_moment_estimation.h"
#include "opennn/random_utilities.h"

// windows.h must come after the OpenNN/Eigen headers and with NOMINMAX so its
// min/max macros do not clobber std::min / std::max inside those headers.
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

// Live resident set (not the monotonic peak): lets us separate the transient
// load-time spike from the sustained footprint once the parser buffers free.
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
        if (argc < 3)
        {
            std::cerr << "usage: opennn_capacity <csv_path> <input_variables>\n";
            return 2;
        }

        const std::string csv_path = argv[1];
        const Index input_variables = Index(std::stoll(argv[2]));

        set_seed(42);
        Configuration::instance().set(Device::Auto, Type::FP32);

        // Load the CSV: comma-separated, no header, no id column.
        // This is where a too-large dataset runs out of memory.
        TabularDataset dataset(csv_path, ",", false, false);

        const Index samples = dataset.get_samples_number();
        std::cout << "loaded_samples=" << samples << "\n";
        // Live resident set right after load (parser buffers already freed when
        // read_csv returned) vs the peak reached during parsing.
        std::cout << "sustained_after_load_mb=" << current_working_set_mb() << "\n";
        std::cout << "after_load_peak_mb=" << peak_working_set_mb() << "\n";

        dataset.split_samples_random(1.0f, 0.0f, 0.0f);

        // Network: input_variables -> input_variables -> 1, tanh then linear.
        ApproximationNetwork network(dataset.get_input_shape(),
                                     {input_variables},
                                     dataset.get_target_shape());

        TrainingStrategy training_strategy(&network, &dataset);
        training_strategy.set_loss("MeanSquaredError");
        training_strategy.set_optimization_algorithm("AdaptiveMomentEstimation");

        auto* adam = dynamic_cast<AdaptiveMomentEstimation*>(
            training_strategy.get_optimization_algorithm());
        adam->set_batch_size(1000);
        adam->set_maximum_epochs(1);          // one pass is enough to allocate batch buffers
        adam->set_display_period(1);

        training_strategy.train();

        std::cout << "trained=1\n";
        std::cout << "peak_mb=" << peak_working_set_mb() << "\n";
        std::cout << "RESULT=OK\n";
        return 0;
    }
    catch (const std::bad_alloc&)
    {
        std::cout << "peak_mb=" << peak_working_set_mb() << "\n";
        std::cout << "RESULT=OOM\n";
        return 1;
    }
    catch (const std::exception& e)
    {
        std::cerr << e.what() << "\n";
        std::cout << "peak_mb=" << peak_working_set_mb() << "\n";
        std::cout << "RESULT=ERROR\n";
        return 1;
    }
}
