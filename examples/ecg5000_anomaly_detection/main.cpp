//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   E C G 5 0 0 0   A N O M A L Y   D E T E C T I O N
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include <iostream>
#include <numeric>
#include <vector>

#include "opennn/core/configuration.h"
#include "opennn/core/random_utilities.h"
#include "opennn/dataset/tabular_dataset.h"
#include "opennn/models/models.h"
#include "opennn/testing_analysis/testing_analysis.h"
#include "opennn/training_strategy/training_strategy.h"

using namespace opennn;

int main()
{
    try
    {
        cout << "OpenNN. ECG5000 anomaly detection example." << endl;

        set_seed(21);
        Configuration::instance().set(Device::Auto, Type::FP32);

        TabularDataset dataset("../data/ecg5000_anomaly_detection/ecg.csv",
                               ",", false, false);

        const Index signal_size = dataset.get_variables_number() - 1;
        const Index label_index = signal_size;

        dataset.split_samples_random(0.8f, 0.0f, 0.2f);

        for(const Index sample : dataset.get_sample_indices(SampleRole::Training))
            if(dataset.get_data()(sample, label_index) < 0.5f)
                dataset.set_sample_role(sample, SampleRole::None);

        vector<Index> signal_indices(signal_size);
        iota(signal_indices.begin(), signal_indices.end(), Index(0));
        dataset.set_variable_indices(signal_indices, signal_indices);

        AutoAssociationNetwork autoencoder(dataset.get_input_shape(),
                                            {32, 16, 8},
                                            "ReLU",
                                            "Identity");

        TrainingStrategy training_strategy(&autoencoder, &dataset);
        training_strategy.set_loss("MeanAbsoluteError");
        training_strategy.get_optimization_algorithm()->set_batch_size(512);
        training_strategy.get_optimization_algorithm()->set_maximum_epochs(20);

        training_strategy.train();

        TestingAnalysis testing_analysis(&autoencoder, &dataset);

        const VectorR training_errors =
            testing_analysis.calculate_reconstruction_errors("Training");
        const auto error_statistics =
            testing_analysis.calculate_reconstruction_error_statistics(training_errors);
        const float anomaly_threshold =
            testing_analysis.calculate_anomaly_threshold(error_statistics);
        const VectorI anomalies = testing_analysis.calculate_anomaly_predictions(
            testing_analysis.calculate_reconstruction_errors("Testing"),
            anomaly_threshold);

        cout << "Anomaly threshold: " << anomaly_threshold << endl
             << "Detected " << anomalies.sum() << " anomalies in "
             << anomalies.size() << " testing samples." << endl
             << "Good bye!" << endl;

        return 0;
    }
    catch(const exception& error)
    {
        cerr << error.what() << endl;
        return 1;
    }
}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
