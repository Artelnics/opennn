//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   I R I S   P L A N T   A P P L I C A T I O N
//
//   Artificial Intelligence Techniques SL (Artelnics)
//   artelnics@artelnics.com

#include "../../opennn/dataset.h"
#include "../../opennn/standard_networks.h"
#include "../../opennn/training_strategy.h"
#include "../../opennn/testing_analysis.h"
#include "../../opennn/adaptive_moment_estimation.h"
#include "../../opennn/random_utilities.h"

using namespace opennn;

int main()
{
    try
    {
        cout << "OpenNN. Iris Plant Example." << endl;

        // Dataset

        Dataset dataset("../data/iris_plant_original.csv", ";", true, false);

        const Index inputs_number = dataset.get_features_number("Input");
        const Index targets_number = dataset.get_features_number("Target");

        // Neural network

        const Index neurons_number = 16;

        ClassificationNetwork classification_network({inputs_number}, {neurons_number}, {targets_number});

        // Training Strategy

        TrainingStrategy training_strategy(&classification_network, &dataset);

        training_strategy.train();

        // Testing Analysis

        TestingAnalysis testing_analysis(&classification_network, &dataset);

        cout << "Confusion matrix:\n" << testing_analysis.calculate_confusion() << endl;

        // Deployment

        MatrixR input_vector(1, 4);
        input_vector << 5.1, 3.5, 1.4, 0.2;

        const MatrixR output_tensor = classification_network.calculate_outputs(input_vector);

        cout << "Class probabilities: " << output_tensor << endl;

        // Export

        classification_network.save("iris_model.xml");

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
