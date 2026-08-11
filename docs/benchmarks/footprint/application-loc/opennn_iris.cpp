#include "opennn/dataset/tabular_dataset.h"
#include "opennn/neural_network/standard_networks.h"
#include "opennn/training_strategy/training_strategy.h"
#include "opennn/testing_analysis/testing_analysis.h"

using namespace opennn;

int main()
{
    TabularDataset dataset("../data/iris_plant_original.csv", ";", true, false);

    ClassificationNetwork classification_network(
        {dataset.get_features_number("Input")}, {16}, {dataset.get_features_number("Target")});

    TrainingStrategy(&classification_network, &dataset).train();

    cout << "Confusion matrix:\n"
         << TestingAnalysis(&classification_network, &dataset).calculate_confusion()
         << endl;

    MatrixR input_vector(1, 4);
    input_vector << 5.1, 3.5, 1.4, 0.2;

    cout << "Class probabilities: "
         << classification_network.calculate_outputs(input_vector)
         << endl;

    classification_network.save("iris_model.json");
}
