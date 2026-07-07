#include "../../opennn/dataset.h"
#include "../../opennn/standard_networks.h"
#include "../../opennn/training_strategy.h"
#include "../../opennn/testing_analysis.h"

using namespace opennn;

int main()
{
    Dataset dataset("../data/iris_plant_original.csv", ";", true, false);

    ClassificationNetwork classification_network(
        {dataset.get_variables_number("Input")}, {16}, {dataset.get_variables_number("Target")});

    TrainingStrategy(&classification_network, &dataset).train();

    cout << "Confusion matrix:\n"
         << TestingAnalysis(&classification_network, &dataset).calculate_confusion()
         << endl;

    Tensor<type, 2> input_tensor(1, 4);
    input_tensor.setValues({{5.1, 3.5, 1.4, 0.2}});

    cout << "Class probabilities: "
         << classification_network.calculate_outputs<2, 2>(input_tensor)
         << endl;

    classification_network.save("iris_model.xml");
}
