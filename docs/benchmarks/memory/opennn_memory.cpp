//   OpenNN peak-memory benchmark: load sum.csv (1000 x 101, regression),
//   build an MLP, train, and report resident-set-size (RSS) at two points:
//   baseline (framework loaded + model built, before training) and peak.

#include <cstdio>
#include <sys/resource.h>

#include "../../../opennn/tabular_dataset.h"
#include "../../../opennn/standard_networks.h"
#include "../../../opennn/training_strategy.h"
#include "../../../opennn/adaptive_moment_estimation.h"
#include "../../../opennn/random_utilities.h"
#include "../../../opennn/configuration.h"

using namespace opennn;

// Peak RSS so far, in MB (ru_maxrss is kilobytes on Linux).
static double peak_rss_mb()
{
    rusage usage;
    getrusage(RUSAGE_SELF, &usage);
    return usage.ru_maxrss / 1024.0;
}

int main()
{
    set_seed(42);
    Configuration::instance().set(Device::Auto, Type::FP32);

    // sum.csv: 100 inputs + 1 target, ';'-separated, no header.
    TabularDataset dataset("sum.csv", ";", false, false);

    ApproximationNetwork network(dataset.get_input_shape(), {64}, dataset.get_target_shape());

    printf("baseline_rss_mb %.1f\n", peak_rss_mb());

    TrainingStrategy training_strategy(&network, &dataset);
    training_strategy.set_loss("MeanSquaredError");
    training_strategy.set_optimization_algorithm("AdaptiveMomentEstimation");

    auto* adam = dynamic_cast<AdaptiveMomentEstimation*>(training_strategy.get_optimization_algorithm());
    adam->set_maximum_epochs(50);
    adam->set_batch_size(32);
    adam->set_display(false);

    training_strategy.train();

    printf("peak_rss_mb %.1f\n", peak_rss_mb());
    return 0;
}
