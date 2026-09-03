//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   I R I S   P L A N T   A P P L I C A T I O N
//
//   Artificial Intelligence Techniques SL (Artelnics)
//   artelnics@artelnics.com

#include <fstream>
#include <iostream>

#include "opennn/core/configuration.h"
#include "opennn/dataset/tabular_dataset.h"
#include "opennn/models/models.h"
#include "opennn/neural_network/model_expression.h"
#include "opennn/testing_analysis/testing_analysis.h"
#include "opennn/training_strategy/training_strategy.h"

using namespace opennn;

namespace
{

void export_tinyml_artifacts(ClassificationNetwork& network)
{
    network.save("iris_model.json");

    const ModelExpression model_expression(&network);
    model_expression.save("iris_model.c", ModelExpression::ProgrammingLanguage::C);
    model_expression.save("iris_model_tables.c", ModelExpression::ProgrammingLanguage::CEmbedded);
    model_expression.save("iris_model.py", ModelExpression::ProgrammingLanguage::Python);

    MatrixR inputs(9, 4);
    inputs << 5.1, 3.5, 1.4, 0.2,
              4.9, 3.0, 1.4, 0.2,
              5.0, 3.4, 1.5, 0.2,
              6.4, 3.2, 4.5, 1.5,
              5.7, 2.8, 4.1, 1.3,
              6.0, 2.9, 4.5, 1.5,
              6.3, 3.3, 6.0, 2.5,
              5.8, 2.7, 5.1, 1.9,
              7.7, 3.8, 6.7, 2.2;

    const MatrixR outputs = network.calculate_outputs(inputs);

    ofstream reference_file("iris_reference.csv");
    reference_file.precision(9);

    for (Index i = 0; i < inputs.rows(); ++i)
    {
        for (Index j = 0; j < inputs.cols(); ++j)
            reference_file << inputs(i, j) << ";";

        for (Index j = 0; j < outputs.cols(); ++j)
            reference_file << outputs(i, j) << (j + 1 < outputs.cols() ? ";" : "\n");
    }

    cout << "Exported TinyML artifacts." << endl;
}

}

int main()
{
    try
    {
        cout << "OpenNN. Iris Plant Example." << endl;

        Configuration::instance().set(Device::CPU, Type::FP32);

        TabularDataset dataset("../data/iris_plant/iris_plant_original.csv", ";", true, false);

        ClassificationNetwork network(dataset.get_input_shape(), {16}, dataset.get_target_shape());

        TrainingStrategy training_strategy(&network, &dataset);
        training_strategy.train();

        TestingAnalysis testing_analysis(&network, &dataset);
        testing_analysis.print_multiple_classification_tests();

        export_tinyml_artifacts(network);

        return 0;
    }
    catch(const exception& e)
    {
        cout << e.what() << endl;

        return 1;
    }
}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
