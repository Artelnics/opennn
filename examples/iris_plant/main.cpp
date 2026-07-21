//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   I R I S   P L A N T   A P P L I C A T I O N
//
//   Artificial Intelligence Techniques SL (Artelnics)
//   artelnics@artelnics.com

#include "opennn/tabular_dataset.h"
#include "opennn/standard_networks.h"
#include "opennn/training_strategy.h"
#include "opennn/testing_analysis.h"
#include "opennn/adaptive_moment_estimation.h"
#include "opennn/random_utilities.h"
#include "opennn/configuration.h"
#include "opennn/model_expression.h"

using namespace opennn;

int main()
{
    try
    {
        cout << "OpenNN. Iris Plant Example." << endl;

        Configuration::instance().set(Device::CPU, Type::FP32);

        // Dataset

        TabularDataset dataset("../data/iris_plant/iris_plant_original.csv", ";", true, false);

        const Index inputs_number = dataset.get_features_number("Input");
        const Index targets_number = dataset.get_features_number("Target");

        // Neural network

        const Index neurons_number = 16;

        ClassificationNetwork classification_network({inputs_number}, {neurons_number}, {targets_number});

        // Training Strategy

        TrainingStrategy training_strategy(&classification_network, &dataset);

        // QuasiNewton (TrainingStrategy default for classification) has no GPU
        // backend; switch to Adam to exercise the CUDA path.
        training_strategy.set_optimization_algorithm("AdaptiveMomentEstimation");

        AdaptiveMomentEstimation* adam = dynamic_cast<AdaptiveMomentEstimation*>(training_strategy.get_optimization_algorithm());
        adam->set_maximum_epochs(500);

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

        classification_network.save("iris_model.json");

        const ModelExpression model_expression(&classification_network);
        model_expression.save("iris_model.c", ModelExpression::ProgrammingLanguage::C);
        model_expression.save("iris_model_tables.c", ModelExpression::ProgrammingLanguage::CEmbedded);
        model_expression.save("iris_model.py", ModelExpression::ProgrammingLanguage::Python);

        // Reference vectors (inputs;outputs) to check parity of the exported model
        // on other targets (e.g. microcontrollers)

        MatrixR reference_inputs(9, 4);
        reference_inputs << 5.1, 3.5, 1.4, 0.2,
                            4.9, 3.0, 1.4, 0.2,
                            5.0, 3.4, 1.5, 0.2,
                            6.4, 3.2, 4.5, 1.5,
                            5.7, 2.8, 4.1, 1.3,
                            6.0, 2.9, 4.5, 1.5,
                            6.3, 3.3, 6.0, 2.5,
                            5.8, 2.7, 5.1, 1.9,
                            7.7, 3.8, 6.7, 2.2;

        const MatrixR reference_outputs = classification_network.calculate_outputs(reference_inputs);

        ofstream reference_file("iris_reference.csv");
        reference_file.precision(9);

        for (Index i = 0; i < reference_inputs.rows(); ++i)
        {
            for (Index j = 0; j < reference_inputs.cols(); ++j)
                reference_file << reference_inputs(i, j) << ";";

            for (Index j = 0; j < reference_outputs.cols(); ++j)
                reference_file << reference_outputs(i, j) << (j + 1 < reference_outputs.cols() ? ";" : "\n");
        }

        cout << "Exported iris_model.c and iris_reference.csv" << endl;

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
