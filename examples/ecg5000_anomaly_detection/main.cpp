//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   E C G 5 0 0 0   A N O M A L Y   D E T E C T I O N
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <utility>
#include <vector>

#include "opennn/core/configuration.h"
#include "opennn/core/random_utilities.h"
#include "opennn/dataset/tabular_dataset.h"
#include "opennn/models/models.h"
#include "opennn/testing_analysis/testing_analysis.h"
#include "opennn/training_strategy/adaptive_moment_estimation.h"
#include "opennn/training_strategy/training_strategy.h"

using namespace opennn;

namespace
{

constexpr Index signal_size = 140;

vector<bool> load_test_mask(const filesystem::path& path, const Index samples_number)
{
    ifstream file(path);
    throw_if(!file, "Cannot open test split file: {}", path.string());

    vector<bool> test_mask(size_t(samples_number), false);
    Index test_samples = 0;
    Index sample_index = -1;

    while(file >> sample_index)
    {
        throw_if(sample_index < 0 || sample_index >= samples_number,
                 "Test sample index {} is out of range.", sample_index);
        throw_if(test_mask[size_t(sample_index)],
                 "Test sample index {} is duplicated.", sample_index);

        test_mask[size_t(sample_index)] = true;
        ++test_samples;
    }

    throw_if(test_samples != 1000,
             "Expected 1000 test sample indices, found {}.", test_samples);

    return test_mask;
}

pair<float, float> calculate_training_range(const MatrixR& data,
                                            const vector<bool>& test_mask)
{
    float minimum = numeric_limits<float>::max();
    float maximum = numeric_limits<float>::lowest();

    for(Index sample = 0; sample < data.rows(); ++sample)
    {
        if(test_mask[size_t(sample)]) continue;

        for(Index point = 0; point < signal_size; ++point)
        {
            minimum = min(minimum, data(sample, point));
            maximum = max(maximum, data(sample, point));
        }
    }

    throw_if(maximum - minimum < EPSILON,
             "The ECG training range is degenerate.");

    return {minimum, maximum};
}

void save_reconstruction_errors(const filesystem::path& path,
                                const vector<Index>& training_indices,
                                const VectorR& training_errors,
                                const vector<Index>& testing_indices,
                                const VectorR& testing_errors,
                                const VectorR& labels)
{
    ofstream file(path);
    throw_if(!file, "Cannot create reconstruction-error file: {}", path.string());

    file << "sample,error,label,subset\n" << setprecision(9);

    for(Index i = 0; i < ssize(training_indices); ++i)
        file << training_indices[size_t(i)] << ',' << training_errors(i)
             << ",normal,training\n";

    for(Index i = 0; i < ssize(testing_indices); ++i)
        file << testing_indices[size_t(i)] << ',' << testing_errors(i) << ','
             << (labels(testing_indices[size_t(i)]) >= 0.5f ? "normal" : "anomalous")
             << ",testing\n";
}

void save_signal_reconstruction(const filesystem::path& path,
                                const MatrixR& normalized_data,
                                const Index sample_index,
                                AutoAssociationNetwork& network)
{
    MatrixR input(1, signal_size);
    input.row(0) = normalized_data.row(sample_index).head(signal_size);
    const MatrixR reconstruction = network.calculate_outputs(input);

    ofstream file(path);
    throw_if(!file, "Cannot create signal reconstruction file: {}", path.string());

    file << "time_index,input,reconstruction,absolute_error\n" << setprecision(9);
    for(Index i = 0; i < signal_size; ++i)
        file << i << ',' << input(0, i) << ',' << reconstruction(0, i) << ','
             << abs(input(0, i) - reconstruction(0, i)) << '\n';
}

void save_detector_metadata(const filesystem::path& path,
                            const float minimum,
                            const float maximum,
                            const float threshold)
{
    ofstream file(path);
    throw_if(!file, "Cannot create detector metadata file: {}", path.string());

    file << setprecision(9)
         << "{\n"
         << "  \"model\": \"ecg5000_autoencoder.json\",\n"
         << "  \"normalization\": {\"method\": \"global_minimum_maximum\", "
         << "\"minimum\": " << minimum << ", \"maximum\": " << maximum << "},\n"
         << "  \"reconstruction_error\": \"mean_absolute_error\",\n"
         << "  \"threshold\": " << threshold << ",\n"
         << "  \"anomaly_rule\": \"error >= threshold\"\n"
         << "}\n";
}

MatrixR make_labels(const vector<Index>& sample_indices,
                    const VectorR& labels,
                    const bool anomaly_positive)
{
    MatrixR result(sample_indices.size(), 1);

    for(Index i = 0; i < ssize(sample_indices); ++i)
    {
        const float normal = labels(sample_indices[size_t(i)]) >= 0.5f ? 1.0f : 0.0f;
        result(i, 0) = anomaly_positive ? 1.0f - normal : normal;
    }

    return result;
}

MatrixR make_predictions(const VectorI& anomaly_predictions,
                         const bool anomaly_positive)
{
    MatrixR result(anomaly_predictions.size(), 1);

    for(Index i = 0; i < anomaly_predictions.size(); ++i)
    {
        const float anomaly = float(anomaly_predictions(i));
        result(i, 0) = anomaly_positive ? anomaly : 1.0f - anomaly;
    }

    return result;
}

void print_metrics(const string& title,
                   const MatrixI& confusion,
                   const VectorR& metrics)
{
    cout << "\n" << title << "\n"
         << "Confusion matrix (rows: actual positive/negative; "
            "columns: predicted positive/negative):\n"
         << confusion(0, 0) << ' ' << confusion(0, 1) << '\n'
         << confusion(1, 0) << ' ' << confusion(1, 1) << '\n'
         << "Accuracy:  " << metrics(0) << '\n'
         << "Precision: " << metrics(4) << '\n'
         << "Recall:    " << metrics(2) << '\n'
         << "Specificity: " << metrics(3) << '\n'
         << "F1 score:  " << metrics(7) << '\n';
}

}

int main()
{
    try
    {
        cout << "OpenNN. ECG5000 anomaly detection.\n";

        set_seed(21);
        Configuration::instance().set(Device::Auto, Type::FP32);

        TabularDataset dataset("../data/ecg5000_anomaly_detection/ecg.csv",
                               ",", false, false);

        throw_if(dataset.get_variables_number() != signal_size + 1,
                 "Expected 140 ECG values and one label, found {} columns.",
                 dataset.get_variables_number());

        const Index samples_number = dataset.get_samples_number();
        const Index label_column = signal_size;
        const MatrixR raw_data = dataset.get_data();
        const VectorR labels = raw_data.col(label_column);
        const vector<bool> test_mask = load_test_mask(
            "../data/ecg5000_anomaly_detection/test_indices.csv", samples_number);

        const auto [minimum, maximum] = calculate_training_range(raw_data, test_mask);
        MatrixR normalized_data = raw_data;
        normalized_data.leftCols(signal_size).array() =
            (normalized_data.leftCols(signal_size).array() - minimum) / (maximum - minimum);
        dataset.set_data(normalized_data);

        vector<Index> signal_indices(signal_size);
        iota(signal_indices.begin(), signal_indices.end(), Index(0));
        dataset.set_variable_indices(signal_indices, signal_indices);
        dataset.set_variable_scalers("None");

        dataset.set_sample_roles(SampleRole::None);
        Index excluded_anomalies = 0;
        for(Index sample = 0; sample < samples_number; ++sample)
        {
            if(test_mask[size_t(sample)])
                dataset.set_sample_role(sample, SampleRole::Testing);
            else if(labels(sample) >= 0.5f)
                dataset.set_sample_role(sample, SampleRole::Training);
            else
                ++excluded_anomalies;
        }

        const Index training_samples =
            ssize(dataset.get_sample_indices(SampleRole::Training));
        const Index testing_samples =
            ssize(dataset.get_sample_indices(SampleRole::Testing));

        cout << "\nTraining samples\n"
             << "Normal: " << training_samples << '\n'
             << "Anomalous excluded from fitting: " << excluded_anomalies << '\n'
             << "Global normalization range: [" << minimum << ", " << maximum << "]\n"
             << "\nNetwork\n140-32-16-8-16-32-140\n";

        AutoAssociationNetwork network({signal_size}, {32, 16, 8}, "ReLU", "Sigmoid");

        TrainingStrategy training_strategy(&network, &dataset);
        training_strategy.set_loss("MeanAbsoluteError");
        training_strategy.set_optimization_algorithm("AdaptiveMomentEstimation");

        auto* adam = dynamic_cast<AdaptiveMomentEstimation*>(
            training_strategy.get_optimization_algorithm());
        throw_if(!adam, "Expected the Adam optimizer.");

        adam->set_learning_rate(0.001f);
        adam->set_beta_1(0.9f);
        adam->set_beta_2(0.999f);
        adam->set_batch_size(512);
        adam->set_maximum_epochs(20);
        adam->set_shuffle(true);
        adam->set_display_period(1);

        cout << "\nTraining\n"
             << "Loss: Mean absolute error\n"
             << "Optimizer: Adam\n"
             << "Epochs: 20\n"
             << "Batch size: 512\n";

        training_strategy.train();

        TestingAnalysis testing_analysis(&network, &dataset);
        const VectorR training_errors =
            testing_analysis.calculate_reconstruction_errors("Training");
        const auto statistics =
            testing_analysis.calculate_reconstruction_error_statistics(training_errors);
        const float threshold =
            testing_analysis.calculate_anomaly_threshold(statistics);

        const VectorR testing_errors =
            testing_analysis.calculate_reconstruction_errors("Testing");
        const VectorI anomaly_predictions =
            testing_analysis.calculate_anomaly_predictions(testing_errors, threshold);

        const vector<Index> training_indices =
            dataset.get_sample_indices(SampleRole::Training);
        const vector<Index> testing_indices =
            dataset.get_sample_indices(SampleRole::Testing);

        const MatrixR normal_targets = make_labels(testing_indices, labels, false);
        const MatrixR normal_predictions = make_predictions(anomaly_predictions, false);
        const MatrixI normal_confusion =
            testing_analysis.calculate_confusion(normal_targets, normal_predictions);
        const VectorR normal_metrics =
            testing_analysis.calculate_binary_classification_tests(normal_targets,
                                                                   normal_predictions);

        const MatrixR anomaly_targets = make_labels(testing_indices, labels, true);
        const MatrixR anomaly_outputs = make_predictions(anomaly_predictions, true);
        const MatrixI anomaly_confusion =
            testing_analysis.calculate_confusion(anomaly_targets, anomaly_outputs);
        const VectorR anomaly_metrics =
            testing_analysis.calculate_binary_classification_tests(anomaly_targets,
                                                                   anomaly_outputs);

        Index normal_test_samples = 0;
        for(const Index sample : testing_indices)
            normal_test_samples += labels(sample) >= 0.5f ? 1 : 0;

        cout << "\nReconstruction analysis\n"
             << "Normal training error minimum: " << statistics.minimum << '\n'
             << "Normal training error maximum: " << statistics.maximum << '\n'
             << "Normal training error mean: " << statistics.mean << '\n'
             << "Normal training error population standard deviation: "
             << statistics.population_standard_deviation << '\n'
             << "Threshold (mean + standard deviation): " << threshold << '\n'
             << "\nTesting samples\n"
             << "Normal: " << normal_test_samples << '\n'
             << "Anomalous: " << testing_samples - normal_test_samples << '\n';

        print_metrics("TensorFlow-compatible metrics (positive = normal)",
                      normal_confusion, normal_metrics);
        print_metrics("Anomaly-centric metrics (positive = anomaly)",
                      anomaly_confusion, anomaly_metrics);

        network.save("ecg5000_autoencoder.json");
        save_detector_metadata("ecg5000_anomaly_detector.json",
                               minimum, maximum, threshold);
        save_reconstruction_errors("ecg5000_reconstruction_errors.csv",
                                   training_indices, training_errors,
                                   testing_indices, testing_errors, labels);

        const auto normal_sample = ranges::find_if(testing_indices, [&](const Index sample)
        {
            return labels(sample) >= 0.5f;
        });
        const auto anomalous_sample = ranges::find_if(testing_indices, [&](const Index sample)
        {
            return labels(sample) < 0.5f;
        });

        if(normal_sample != testing_indices.end())
            save_signal_reconstruction("ecg5000_normal_reconstruction.csv",
                                       normalized_data, *normal_sample, network);
        if(anomalous_sample != testing_indices.end())
            save_signal_reconstruction("ecg5000_anomalous_reconstruction.csv",
                                       normalized_data, *anomalous_sample, network);

        cout << "\nSaved model, detector metadata, and analysis CSV files.\n";

        return 0;
    }
    catch(const exception& error)
    {
        cerr << error.what() << '\n';
        return 1;
    }
}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
