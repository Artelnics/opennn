// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Artificial Intelligence Techniques, S.L. and Roberto López González

#include <iostream>

#include "opennn/core/configuration.h"
#include "opennn/dataset/tabular_dataset.h"
#include "opennn/models/models.h"
#include "opennn/network/model_expression.h"
#include "opennn/evaluation/evaluation.h"
#include "opennn/training/training.h"

using namespace opennn;

namespace
{

void export_model(ClassificationNetwork& network)
{
    network.save("iris_model.json");

    const ModelExpression model_expression(&network);
    model_expression.save("iris_model.c", ModelExpression::ProgrammingLanguage::C);
    model_expression.save("iris_model_tables.c", ModelExpression::ProgrammingLanguage::CEmbedded);
    model_expression.save("iris_model.py", ModelExpression::ProgrammingLanguage::Python);

    cout << "Exported the model as JSON, C and Python." << endl;
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

        Training training(&network, &dataset);
        training.train();

        Evaluation evaluation(&network, &dataset);
        evaluation.print_multiple_classification_tests();

        export_model(network);

        return 0;
    }
    catch(const exception& e)
    {
        cout << e.what() << endl;

        return 1;
    }
}
