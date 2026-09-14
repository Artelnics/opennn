//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   E C G 5 0 0 0   A N O M A L Y   D E T E C T I O N
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

// An autoencoder learns to reconstruct normal heartbeats. A heartbeat raises an
// alert when its reconstruction error reaches the mean plus one standard
// deviation of the errors on the normal training heartbeats.

#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>

#include "opennn/core/configuration.h"
#include "opennn/core/random_utilities.h"
#include "opennn/dataset/tabular_dataset.h"
#include "opennn/evaluation/evaluation.h"
#include "opennn/models/models.h"
#include "opennn/training/adam.h"
#include "opennn/training/training.h"

using namespace opennn;

int main()
{
    try
    {
        cout << "OpenNN. ECG5000 anomaly detection example." << endl;

        set_seed(21);

        Configuration::instance().set(Device::CPU, Type::FP32);

        // Each row holds 140 ECG values and a label: 1 normal, 0 anomalous.

        const Index signal_size = 140;

        TabularDataset dataset("../data/ecg5000_anomaly_detection/ecg.csv", ",", false, false);

        const Index samples_number = dataset.get_samples_number();
        const MatrixR raw_data = dataset.get_data();
        const VectorR labels = raw_data.col(signal_size);

        // The 1,000 testing rows are listed in a file; the other rows are for development.

        ifstream test_indices_file("../data/ecg5000_anomaly_detection/test_indices.csv");
        if (!test_indices_file) throw runtime_error("Cannot open test_indices.csv.");

        vector<bool> is_testing(size_t(samples_number), false);

        for (Index index; test_indices_file >> index;)
            is_testing[size_t(index)] = true;

        // One minimum and one maximum over all development values map every value to [0, 1].

        float minimum = numeric_limits<float>::max();
        float maximum = numeric_limits<float>::lowest();

        for (Index sample = 0; sample < samples_number; ++sample)
        {
            if (is_testing[size_t(sample)]) continue;

            minimum = min(minimum, raw_data.row(sample).head(signal_size).minCoeff());
            maximum = max(maximum, raw_data.row(sample).head(signal_size).maxCoeff());
        }

        MatrixR scaled_data = raw_data;
        scaled_data.leftCols(signal_size).array() =
            (raw_data.leftCols(signal_size).array() - minimum) / (maximum - minimum);

        dataset.set_data(scaled_data);
        dataset.set_variable_scalers("None");

        // The 140 values are both inputs and targets. Only normal development
        // signals are used for training; anomalous development signals stay unused.

        vector<Index> signal_indices(signal_size);
        iota(signal_indices.begin(), signal_indices.end(), Index(0));
        dataset.set_variable_indices(signal_indices, signal_indices);

        dataset.set_sample_roles(SampleRole::None);

        for (Index sample = 0; sample < samples_number; ++sample)
            if (is_testing[size_t(sample)])
                dataset.set_sample_role(sample, SampleRole::Testing);
            else if (labels(sample) >= 0.5f)
                dataset.set_sample_role(sample, SampleRole::Training);

        // Autoencoder 140-32-16-8-16-32-140 with ReLU hidden layers and a sigmoid output,
        // trained for 20 epochs with Adam on the mean absolute error.

        Autoencoder autoencoder(dataset.get_input_shape(), {32, 16, 8}, "ReLU", "Sigmoid");

        Training training(&autoencoder, &dataset);
        training.set_loss("MeanAbsoluteError");
        training.set_optimization_algorithm("Adam");

        Adam* adam = dynamic_cast<Adam*>(training.get_optimization_algorithm());
        adam->set_learning_rate(0.001f);
        adam->set_beta_1(0.9f);
        adam->set_beta_2(0.999f);
        adam->set_batch_size(512);
        adam->set_maximum_epochs(20);
        adam->set_shuffle(true);
        adam->set_bf16_first_moment(false);

        training.train();

        // The threshold uses only the errors of the normal training signals.

        Evaluation evaluation(&autoencoder, &dataset);

        const VectorR training_errors = evaluation.calculate_reconstruction_errors("Training");
        const Evaluation::ReconstructionErrorStatistics statistics =
            evaluation.calculate_reconstruction_error_statistics(training_errors);
        const float threshold = evaluation.calculate_anomaly_threshold(statistics, 1.0f);

        // Testing, with anomaly as the positive class.

        const vector<Index> testing_indices = dataset.get_sample_indices(SampleRole::Testing);
        const VectorR testing_errors = evaluation.calculate_reconstruction_errors("Testing");
        const VectorI alerts = evaluation.calculate_anomaly_predictions(testing_errors, threshold);

        const Index testing_samples = testing_errors.size();

        const MatrixR scores = testing_errors;
        const MatrixR predictions = alerts.cast<float>();
        MatrixR targets(testing_samples, 1);

        for (Index i = 0; i < testing_samples; ++i)
            targets(i, 0) = labels(testing_indices[size_t(i)]) >= 0.5f ? 0.0f : 1.0f;

        const MatrixI confusion = evaluation.calculate_confusion(targets, predictions);
        const VectorR tests = evaluation.calculate_binary_classification_tests(targets, predictions);
        const float area_under_curve = evaluation.calculate_area_under_curve(evaluation.calculate_roc_curve(targets, scores));

        cout << setprecision(7)
             << "Training error mean: " << statistics.mean
             << " +/- " << statistics.population_standard_deviation << endl
             << "Threshold: " << threshold << endl
             << "Confusion matrix (rows: actual anomaly, normal; columns: predicted anomaly, normal):" << endl
             << confusion(0, 0) << ' ' << confusion(0, 1) << endl
             << confusion(1, 0) << ' ' << confusion(1, 1) << endl
             << "Accuracy    : " << tests(0) << endl
             << "Precision   : " << tests(4) << endl
             << "Sensitivity : " << tests(2) << endl
             << "Specificity : " << tests(3) << endl
             << "F1 score    : " << tests(7) << endl
             << "ROC AUC     : " << area_under_curve << endl;

        cout << "Good bye!" << endl;

        return 0;
    }
    catch (const exception& e)
    {
        cerr << e.what() << endl;

        return 1;
    }
}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
