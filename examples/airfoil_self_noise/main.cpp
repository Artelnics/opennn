// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Artificial Intelligence Techniques, S.L. and Roberto López González

#include <iostream>

#include "opennn/core/random_utilities.h"
#include "opennn/dataset/tabular_dataset.h"
#include "opennn/models/models.h"
#include "opennn/evaluation/evaluation.h"
#include "opennn/training/training.h"

using namespace opennn;

int main()
{
    try
    {
        cout << "OpenNN. Airfoil self noise Example." << endl;

        set_seed(0);

        TabularDataset dataset("../data/airfoil_self_noise/airfoil_self_noise.csv", ";", true, false);

        ApproximationNetwork network(dataset.get_input_shape(),
                                     {12},
                                     dataset.get_target_shape());

        Training training(&network, &dataset);

        training.train();

        Evaluation evaluation(&network, &dataset);

        evaluation.print_goodness_of_fit_analysis();

        cout << "Good bye!" << endl;

        return 0;
    }
    catch(exception& e)
    {
        cerr << e.what() << endl;

        return 1;
    }
}
