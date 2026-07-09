//   OpenNN standalone-code-export benchmark: train a small MLP on sum.csv, then
//   export the trained model to dependency-free C and Python source.

#include "opennn/tabular_dataset.h"
#include "opennn/standard_networks.h"
#include "opennn/training_strategy.h"
#include "opennn/adaptive_moment_estimation.h"
#include "opennn/model_expression.h"
#include "opennn/random_utilities.h"
#include "opennn/configuration.h"

using namespace opennn;

int main()
{
    set_seed(42);
    Configuration::instance().set(Device::Auto, Type::FP32);

    // sum.csv: 100 inputs + 1 target, ';'-separated, no header.
    TabularDataset dataset("sum.csv", ";", false, false);

    ApproximationNetwork network(dataset.get_input_shape(), {64}, dataset.get_target_shape());

    TrainingStrategy training_strategy(&network, &dataset);
    training_strategy.set_loss("MeanSquaredError");
    training_strategy.set_optimization_algorithm("AdaptiveMomentEstimation");

    auto* adam = dynamic_cast<AdaptiveMomentEstimation*>(training_strategy.get_optimization_algorithm());
    adam->set_maximum_epochs(50);
    adam->set_batch_size(32);
    adam->set_display(false);

    training_strategy.train();

    // Export the trained model to standalone source code.
    ModelExpression expression(&network);
    expression.save("model.c", ModelExpression::ProgrammingLanguage::C);
    expression.save("model.py", ModelExpression::ProgrammingLanguage::Python);

    cout << "exported model.c and model.py" << endl;
    return 0;
}
