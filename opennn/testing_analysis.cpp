//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   T E S T I N G   A N A L Y S I S   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "testing_analysis.h"
#include "correlations.h"
#include "language_dataset.h"
#include "time_series_dataset.h"
#include "standard_networks.h"
#include "statistics.h"
#include "unscaling_layer.h"
#include "error_utilities.h"
#include "forward_propagation.h"
#include "back_propagation.h"

namespace opennn
{

TestingAnalysis::TestingAnalysis(NeuralNetwork* new_neural_network, Dataset* new_dataset)
{
    neural_network = new_neural_network;
    dataset = new_dataset;
}

void TestingAnalysis::check() const
{
    if (!neural_network)
        throw runtime_error("neural network is not set.");

    if (!dataset)
        throw runtime_error("dataset is not set.");
}

Tensor<Correlation, 1> TestingAnalysis::linear_correlation(const MatrixR& target, const MatrixR& output) const
{
    const Index outputs_number = dataset->get_features_number("Target");

    Tensor<Correlation, 1> linear_correlation(outputs_number);

    for (Index i = 0; i < outputs_number; ++i)
        linear_correlation(i) = opennn::linear_correlation(output.col(i), target.col(i));

    return linear_correlation;
}

void TestingAnalysis::print_linear_correlations() const
{
    const auto [targets, outputs] = get_targets_and_outputs("Testing");
    const Tensor<Correlation, 1> linear_correlations = linear_correlation(targets, outputs);

    const vector<string> targets_name = dataset->get_feature_names("Target");

    const Index targets_number = linear_correlations.size();

    for (Index i = 0; i < targets_number; ++i)
        cout << targets_name[i] << " correlation: " << linear_correlations[i].r << "\n";
}

Tensor<TestingAnalysis::GoodnessOfFitAnalysis, 1> TestingAnalysis::perform_goodness_of_fit_analysis() const
{
    check();

    // Dataset

    const Index testing_samples_number = dataset->get_samples_number("Testing");

    if (testing_samples_number == 0)
        throw runtime_error("Number of testing samples is zero.\n");

    // Neural network

    const Index outputs_number = neural_network->get_outputs_number();

    const auto [all_targets, all_outputs] = get_targets_and_outputs("Testing");
    Tensor<GoodnessOfFitAnalysis, 1> goodness_of_fit_results(outputs_number);

    for (Index i = 0; i < outputs_number; ++i)
    {
        const VectorMap targets = vector_map(all_targets, i);
        const VectorMap outputs = vector_map(all_outputs, i);

        const float determination = calculate_determination(outputs, targets);

        goodness_of_fit_results[i].set(targets, outputs, determination);
    }

    return goodness_of_fit_results;
}

void TestingAnalysis::print_goodness_of_fit_analysis() const
{
    const Tensor<GoodnessOfFitAnalysis, 1> goodness_of_fit_analysis = perform_goodness_of_fit_analysis();

    for (Index i = 0; i < goodness_of_fit_analysis.size(); ++i)
        goodness_of_fit_analysis(i).print();
}

pair<MatrixR, MatrixR> TestingAnalysis::get_targets_and_outputs(const string& sample_role) const
{
    check();

    // Dataset

    const Index samples_number = dataset->get_samples_number(sample_role);

    if (samples_number == 0)
        throw runtime_error("Number of samples is zero.\n");

    MatrixR output_data;
    MatrixR target_data;

    const vector<Index> sample_indices = dataset->get_sample_indices(sample_role);
    const vector<Index> input_feature_indices  = dataset->get_feature_indices("Input");
    const vector<Index> target_feature_indices = dataset->get_feature_indices("Target");

    target_data.resize(ssize(sample_indices), ssize(target_feature_indices));
    dataset->fill_targets(sample_indices, target_feature_indices,
                          target_data.data(), /*is_training=*/false, /*parallelize=*/true);

    if (const TimeSeriesDataset* time_series_dataset = dynamic_cast<TimeSeriesDataset*>(dataset))
    {
        const Tensor3 input_data = time_series_dataset->get_data(sample_role, "Input");
        output_data = neural_network->calculate_outputs(input_data);
    }
    else
    {
        const Shape input_shape = dataset->get_shape("Input");

        if (input_shape.rank == 1)
        {
            MatrixR input_data(samples_number, input_shape[0]);
            dataset->fill_inputs(sample_indices, input_feature_indices,
                                 input_data.data(), /*is_training=*/false, /*parallelize=*/true);
            output_data = neural_network->calculate_outputs(input_data);
        }
        else if (input_shape.rank == 3)
        {
            Tensor4 inputs_4d(samples_number, input_shape[0], input_shape[1], input_shape[2]);
            dataset->fill_inputs(sample_indices, input_feature_indices,
                                 inputs_4d.data(), /*is_training=*/false, /*parallelize=*/true);
            output_data = neural_network->calculate_outputs(inputs_4d);
        }
        else
        {
            throw runtime_error("Unsupported input rank " + to_string(input_shape.rank));
        }
    }

    return {target_data, output_data};
}

MatrixR TestingAnalysis::calculate_error() const
{
    const auto [targets, outputs] = get_targets_and_outputs("Testing");

    return targets - outputs;
}

Tensor3 TestingAnalysis::calculate_error_data() const
{
    check();

    const Index testing_samples_number = dataset->get_samples_number("Testing");

    if (testing_samples_number == 0)
        throw runtime_error("Number of testing samples is zero.\n");

    const Index outputs_number = neural_network->get_outputs_number();

    const auto [targets, outputs] = get_targets_and_outputs("Testing");

    const auto* unscaling_layer = dynamic_cast<const Unscaling*>(neural_network->get_first("Unscaling"));

    if (!unscaling_layer)
        throw runtime_error("Unscaling layer not found.\n");

    const VectorR& output_minimums = unscaling_layer->get_minimums();
    const VectorR& output_maximums = unscaling_layer->get_maximums();

    Tensor3 error_data(testing_samples_number, 3, outputs_number);

    const MatrixR absolute_errors = (targets - outputs).array().abs();

#pragma omp parallel for
    for (Index i = 0; i < outputs_number; ++i)
    {
        const float range = abs(output_maximums(i) - output_minimums(i));

        for (Index j = 0; j < testing_samples_number; ++j)
        {
            const float abs_err = absolute_errors(j, i);
            const float scaled = abs_err / range;
            error_data(j, 0, i) = abs_err;
            error_data(j, 1, i) = scaled;
            error_data(j, 2, i) = scaled * 100.0f;
        }
    }

    return error_data;
}

MatrixR TestingAnalysis::calculate_percentage_error_data() const
{
    check();

    // Dataset

    const Index testing_samples_number = dataset->get_samples_number("Testing");

    if (testing_samples_number == 0)
        throw runtime_error("Number of testing samples is zero.\n");

    // Neural network

    const Index outputs_number = neural_network->get_outputs_number();

    const auto [targets, outputs] = get_targets_and_outputs("Testing");

    const auto* unscaling_layer = dynamic_cast<const Unscaling*>(neural_network->get_first("Unscaling"));

    if (!unscaling_layer)
        throw runtime_error("Unscaling layer not found.\n");

    const VectorR& output_minimums = unscaling_layer->get_minimums();
    const VectorR& output_maximums = unscaling_layer->get_maximums();

    const VectorR ranges = (output_maximums - output_minimums).cwiseAbs();
    const MatrixR errors = targets - outputs;
    MatrixR error_data(testing_samples_number, outputs_number);

#pragma omp parallel for
    for (Index i = 0; i < testing_samples_number; ++i)
        for (Index j = 0; j < outputs_number; ++j)
            error_data(i, j) = errors(i, j) * 100.0f / ranges(j);

    return error_data;
}

vector<Descriptives> TestingAnalysis::calculate_absolute_errors_descriptives() const
{
    const auto [targets, outputs] = get_targets_and_outputs("Testing");

    return calculate_absolute_errors_descriptives(targets, outputs);
}

vector<Descriptives> TestingAnalysis::calculate_absolute_errors_descriptives(const MatrixR& targets,
                                                                             const MatrixR& outputs) const
{
    const MatrixR difference = (targets-outputs).array().abs();

    return descriptives(difference);
}

vector<Descriptives> TestingAnalysis::calculate_percentage_errors_descriptives() const
{
    const auto [targets, outputs] = get_targets_and_outputs("Testing");

    return calculate_percentage_errors_descriptives(targets, outputs);
}

vector<Descriptives> TestingAnalysis::calculate_percentage_errors_descriptives(const MatrixR& targets,
                                                                               const MatrixR& outputs) const
{
    const MatrixR difference = 100.0f*(targets-outputs).array().abs()/targets.array();

    return descriptives(difference);
}

vector<vector<Descriptives>> TestingAnalysis::calculate_error_data_descriptives() const
{
    // Neural network

    const Index outputs_number = neural_network->get_outputs_number();

    const Index testing_samples_number = dataset->get_samples_number("Testing");
    vector<vector<Descriptives>> descriptives(outputs_number);

    Tensor3 error_data = calculate_error_data();

    const Index stride = testing_samples_number * 3;

    for (Index i = 0; i < outputs_number; ++i)
    {
        const MatrixMap matrix_error(error_data.data() + i * stride, testing_samples_number, 3);
        descriptives[i] = opennn::descriptives(MatrixR(matrix_error));
    }

    return descriptives;
}

void TestingAnalysis::print_error_data_descriptives() const
{
    const Index targets_number = dataset->get_features_number("Target");

    const vector<string> targets_name = dataset->get_feature_names("Target");

    const vector<vector<Descriptives>> error_data_statistics = calculate_error_data_descriptives();

    for (Index i = 0; i < targets_number; ++i)
        cout << targets_name[i] << "\n"
             << "Minimum error: " << error_data_statistics[i][0].minimum << "\n"
             << "Maximum error: " << error_data_statistics[i][0].maximum << "\n"
             << "Mean error: " << error_data_statistics[i][0].mean << " " << "\n"
             << "Standard deviation error: " << error_data_statistics[i][0].standard_deviation << " " << "\n"
             << "Minimum percentage error: " << error_data_statistics[i][2].minimum << " %" << "\n"
             << "Maximum percentage error: " << error_data_statistics[i][2].maximum << " %" << "\n"
             << "Mean percentage error: " << error_data_statistics[i][2].mean << " %" << "\n"
             << "Standard deviation percentage error: " << error_data_statistics[i][2].standard_deviation << " %" << "\n"
             << "\n";
}

vector<Histogram> TestingAnalysis::calculate_error_data_histograms(const Index bins_number) const
{
    const MatrixR error_data = calculate_percentage_error_data();

    const Index outputs_number = error_data.cols();

    vector<Histogram> histograms(outputs_number);

    for (Index i = 0; i < outputs_number; ++i)
        histograms[i] = histogram_centered(error_data.col(i), 0.0f, bins_number);

    return histograms;
}

Tensor<VectorI, 1> TestingAnalysis::calculate_maximal_errors(const Index samples_number) const
{
    Tensor3 error_data = calculate_error_data();

    const Index outputs_number = error_data.dimension(2);
    const Index testing_samples_number = error_data.dimension(0);

    Tensor<VectorI, 1> maximal_errors(samples_number);

    const Index stride = testing_samples_number * 3;

    for (Index i = 0; i < outputs_number; ++i)
    {
        const MatrixMap matrix_error(error_data.data() + i * stride, testing_samples_number, 3);
        maximal_errors[i] = maximal_indices(matrix_error.col(0), samples_number);
    }

    return maximal_errors;
}

MatrixR TestingAnalysis::calculate_errors() const
{
    MatrixR errors(5, 3);

    errors.col(0) = calculate_errors("Training");
    errors.col(1) = calculate_errors("Validation");
    errors.col(2) = calculate_errors("Testing");

    return errors;
}

MatrixR TestingAnalysis::calculate_binary_classification_errors() const
{
    MatrixR errors(6, 3);

    errors.col(0) = calculate_binary_classification_errors("Training");
    errors.col(1) = calculate_binary_classification_errors("Validation");
    errors.col(2) = calculate_binary_classification_errors("Testing");

    return errors;
}

MatrixR TestingAnalysis::calculate_multiple_classification_errors() const
{
    MatrixR errors(5, 3);

    errors.col(0) = calculate_multiple_classification_errors("Training");
    errors.col(1) = calculate_multiple_classification_errors("Validation");
    errors.col(2) = calculate_multiple_classification_errors("Testing");

    return errors;
}

VectorR TestingAnalysis::calculate_errors(const MatrixR& targets,
                                          const MatrixR& outputs) const
{
    const TensorView outputs_view(const_cast<float*>(outputs.data()), {outputs.rows(), outputs.cols()});
    const TensorView targets_view(const_cast<float*>(targets.data()), {targets.rows(), targets.cols()});

    VectorR errors(5);

    // 1. Mean Squared Error
    mean_squared_error(outputs_view, targets_view, errors(1), nullptr);

    // 0. Sum Squared Error
    errors(0) = errors(1) * static_cast<float>(targets.size());

    // 2. Root Mean Squared Error
    errors(2) = sqrt(errors(1));

    // 3. Normalized Squared Error
    const VectorR targets_mean = mean(targets);
    const float normalization_coefficient = (targets.rowwise() - targets_mean.transpose()).squaredNorm();
    normalized_squared_error(outputs_view, targets_view, normalization_coefficient, errors(3), nullptr);

    // 4. Minkowski Error
    minkowski_error(outputs_view, targets_view, 1.5f, errors(4), nullptr);

    return errors;
}

VectorR TestingAnalysis::calculate_errors(const string& sample_role) const
{
    const auto [targets, outputs] = get_targets_and_outputs(sample_role);

    return calculate_errors(targets, outputs);
}

VectorR TestingAnalysis::calculate_binary_classification_errors(const string& sample_role) const
{
    const auto [targets, outputs] = get_targets_and_outputs(sample_role);

    const TensorView outputs_view(const_cast<float*>(outputs.data()), {outputs.rows(), outputs.cols()});
    const TensorView targets_view(const_cast<float*>(targets.data()), {targets.rows(), targets.cols()});

    VectorR errors(6);

    const VectorR std_errors = calculate_errors(targets, outputs);
    errors.head(4) = std_errors.head(4);

    // 4. Binary Cross Entropy
    binary_cross_entropy(outputs_view, targets_view, errors(4), nullptr);

    // 5. Weighted Squared Error
    const VectorI target_distribution = dataset->calculate_target_distribution();
    const float neg_w = 1.0f;
    const float pos_w = (target_distribution[0] == 0 || target_distribution[1] == 0)
                           ? 1.0f
                           : static_cast<float>(target_distribution[0]) / target_distribution[1];

    weighted_squared_error(outputs_view, targets_view, pos_w, neg_w, errors(5), nullptr);

    return errors;
}

VectorR TestingAnalysis::calculate_multiple_classification_errors(const string& sample_role) const
{
    const auto [targets, outputs] = get_targets_and_outputs(sample_role);

    const TensorView outputs_view(const_cast<float*>(outputs.data()), {outputs.rows(), outputs.cols()});
    const TensorView targets_view(const_cast<float*>(targets.data()), {targets.rows(), targets.cols()});

    VectorR errors(5);

    const VectorR std_errors = calculate_errors(targets, outputs);
    errors.head(4) = std_errors.head(4);

    // 4. Categorical Cross Entropy
    categorical_cross_entropy(outputs_view, targets_view, errors(4), nullptr);

    return errors;
}

float TestingAnalysis::calculate_masked_accuracy(const Tensor3& /*outputs*/, const MatrixR& /*targets*/) const
{
/*
    const Index batch_size = outputs.rows();
    const Index outputs_number = outputs.cols();

    MatrixR predictions(batch_size, outputs_number);
    MatrixB matches(batch_size, outputs_number);
    MatrixB mask(batch_size, outputs_number);

    Tensor0 accuracy;

    mask = targets != targets.constant(0);

    const Tensor0 mask_sum = mask.cast<float>().sum();

    predictions = outputs.argmax(2).cast<float>();

    matches = predictions == targets;

    matches = matches && mask;

    accuracy = matches.cast<float>().sum() / mask_sum(0);

    return accuracy(0);
*/
    return 0;
}

float TestingAnalysis::calculate_determination(const VectorR& outputs, const VectorR& targets) const
{
    const auto targets_centered = targets.array() - targets.mean();
    const auto outputs_centered = outputs.array() - outputs.mean();

    const float numerator = (targets_centered * outputs_centered).sum();

    const float targets_ss = targets_centered.square().sum();
    const float outputs_ss = outputs_centered.square().sum();

    const float denominator = sqrt(targets_ss * outputs_ss);

    if (denominator < EPSILON)
        return 0.0f;

    const float r = numerator / denominator;

    return r * r;
}

vector<MatrixI> TestingAnalysis::calculate_multilabel_confusion(const float decision_threshold) const
{
    check();

    const auto [targets, outputs] = get_targets_and_outputs("Testing");
    const Index outputs_number = neural_network->get_outputs_number();

    vector<MatrixI> confusion_matrices(outputs_number);

    for (Index j = 0; j < outputs_number; ++j)
        confusion_matrices[j] = calculate_confusion(targets.col(j), outputs.col(j), decision_threshold);

    return confusion_matrices;
}

VectorI TestingAnalysis::calculate_positives_negatives_rate(const MatrixR& targets, const MatrixR& outputs) const
{
    const MatrixI confusion = calculate_confusion(targets, outputs, 0.5f);

    VectorI positives_negatives_rate(2);

    positives_negatives_rate << (confusion(0,0) + confusion(0,1)),
                                (confusion(1,0) + confusion(1,1));

    return positives_negatives_rate;
}

MatrixI TestingAnalysis::calculate_confusion(const float decision_threshold) const
{
    check();

    const vector<Index> testing_indices = dataset->get_sample_indices("Testing");

    const Index current_batch_size = (batch_size <= 0 || batch_size > ssize(testing_indices))
                                         ? (testing_indices.empty() ? 1 : testing_indices.size())
                                         : batch_size;

    vector<vector<Index>> testing_batches;
    dataset->get_batches(testing_indices, current_batch_size, false, testing_batches);

    const vector<Index> input_feature_indices = dataset->get_feature_indices("Input");
    const vector<Index> target_feature_indices = dataset->get_feature_indices("Target");

    const Shape input_shape = dataset->get_shape("Input");

    const Index outputs_number = neural_network->get_outputs_number();
    const Index confusion_matrix_size = (outputs_number == 1) ? 3 : (outputs_number + 1);

    MatrixI total_confusion_matrix = MatrixI::Zero(confusion_matrix_size, confusion_matrix_size);

    Index input_elem_count = 1;
    for (size_t d = 0; d < input_shape.rank; ++d) input_elem_count *= input_shape[d];

    const Index targets_number = ssize(target_feature_indices);

    for (const vector<Index>& current_batch_indices : testing_batches)
    {
        if (current_batch_indices.empty()) continue;
        const Index batch_n = current_batch_indices.size();

        MatrixR batch_targets(batch_n, targets_number);
        dataset->fill_targets(current_batch_indices, target_feature_indices,
                              batch_targets.data(), /*is_training=*/false, /*parallelize=*/true);

        MatrixR batch_outputs;
        if (input_shape.rank == 1)
        {
            MatrixR batch_inputs(batch_n, input_shape[0]);
            dataset->fill_inputs(current_batch_indices, input_feature_indices,
                                 batch_inputs.data(), /*is_training=*/false, /*parallelize=*/true);
            batch_outputs = neural_network->calculate_outputs(batch_inputs);
        }
        else if (input_shape.rank == 3)
        {
            Tensor4 inputs_4d(batch_n, input_shape[0], input_shape[1], input_shape[2]);
            dataset->fill_inputs(current_batch_indices, input_feature_indices,
                                 inputs_4d.data(), /*is_training=*/false, /*parallelize=*/true);
            batch_outputs = neural_network->calculate_outputs(inputs_4d);
        }
        else
            return {};

        total_confusion_matrix += calculate_confusion(batch_targets, batch_outputs, decision_threshold);
    }

    if (!testing_indices.empty())
        total_confusion_matrix(confusion_matrix_size - 1, confusion_matrix_size - 1) = testing_indices.size();

    return total_confusion_matrix;
}

MatrixI TestingAnalysis::calculate_confusion(const MatrixR& targets,
                                             const MatrixR& outputs,
                                             float decision_threshold) const
{
    const Index samples = targets.rows();
    const Index outputs_number = outputs.cols();
    const Index num_classes = (outputs_number == 1) ? 2 : outputs_number;

    MatrixI confusion = MatrixI::Zero(num_classes + 1, num_classes + 1);

    for (Index i = 0; i < samples; ++i)
    {
        Index target_class, output_class;

        if (outputs_number == 1)
        {
            target_class = targets(i, 0) >= decision_threshold ? 0 : 1;
            output_class = outputs(i, 0) >= decision_threshold ? 0 : 1;
        }
        else
        {
            target_class = maximal_index(targets.row(i));
            output_class = maximal_index(outputs.row(i));
        }

        confusion(target_class, output_class)++;
        confusion(target_class, num_classes)++;
        confusion(num_classes, output_class)++;
    }

    confusion(num_classes, num_classes) = samples;

    return confusion;
}

TestingAnalysis::RocAnalysis TestingAnalysis::perform_roc_analysis() const
{
    const auto [targets, outputs] = get_targets_and_outputs("Testing");

    RocAnalysis roc_analysis;
    roc_analysis.roc_curve = calculate_roc_curve(targets, outputs);
    roc_analysis.area_under_curve = calculate_area_under_curve(roc_analysis.roc_curve);
    roc_analysis.confidence_limit = calculate_area_under_curve_confidence_limit(targets, outputs);
    roc_analysis.optimal_threshold = calculate_optimal_threshold(roc_analysis.roc_curve);

    return roc_analysis;
}

MatrixR TestingAnalysis::calculate_roc_curve(const MatrixR& targets, const MatrixR& outputs) const
{
    const VectorI positives_negatives_rate = calculate_positives_negatives_rate(targets, outputs);

    const Index total_positives = positives_negatives_rate(0);
    const Index total_negatives = positives_negatives_rate(1);

    if (total_positives == 0)
        throw runtime_error("Number of positive samples (" + to_string(total_positives) + ") must be greater than zero.\n");

    if (total_negatives == 0)
        throw runtime_error("Number of negative samples (" + to_string(total_negatives) + ") must be greater than zero.\n");

    const Index points_number = 100;

    if (targets.cols() != 1)
        throw runtime_error("Number of of target variables (" +  to_string(targets.cols()) + ") must be one.\n");

    if (outputs.cols() != 1)
        throw runtime_error("Number of of output variables (" + to_string(outputs.cols()) + ") must be one.\n");

    MatrixR roc_curve = MatrixR::Zero(points_number + 1, 3);

#pragma omp parallel for schedule(dynamic)

    for (Index i = 1; i < Index(points_number); ++i)
    {
        const float threshold = float(i) * (1.0f/float(points_number));

        Index true_positive = 0;
        Index false_negative = 0;
        Index false_positive = 0;
        Index true_negative = 0;

        for (Index j = 0; j < targets.size(); ++j)
        {
            const bool target_positive = targets(j, 0) >= threshold;
            const bool output_positive = outputs(j, 0) >= threshold;

            if      (target_positive && output_positive) ++true_positive;
            else if (target_positive)                    ++false_negative;
            else if (output_positive)                    ++false_positive;
            else                                         ++true_negative;
        }

        roc_curve(i,0) = 1.0f - float(true_positive)/float(true_positive + false_negative);
        roc_curve(i,1) = float(true_negative)/float(true_negative + false_positive);
        roc_curve(i,2) = threshold;

        if (isnan(roc_curve(i,0)))
            roc_curve(i,0) = 1.0f;

        if (isnan(roc_curve(i,1)))
            roc_curve(i,1) = 0.0f;
    }

    roc_curve.row(0).setZero();
    roc_curve.row(points_number).setOnes();

    return roc_curve;
}

float TestingAnalysis::calculate_area_under_curve(const MatrixR& roc_curve) const
{
    float area_under_curve = 0.0f;

    for (Index i = 1; i < roc_curve.rows(); ++i)
        area_under_curve += (roc_curve(i,0) - roc_curve(i-1,0))*(roc_curve(i,1) + roc_curve(i-1,1));

    return area_under_curve/ 2.0f;
}

float TestingAnalysis::calculate_area_under_curve_confidence_limit(const MatrixR& targets, const MatrixR& outputs) const
{
    const VectorI positives_negatives_rate = calculate_positives_negatives_rate(targets, outputs);

    const Index total_positives = positives_negatives_rate[0];
    const Index total_negatives = positives_negatives_rate[1];

    if (total_positives == 0)
        throw runtime_error("Number of positive samples(" + to_string(total_positives) + ") must be greater than zero.\n");

    if (total_negatives == 0)
        throw runtime_error("Number of negative samples(" + to_string(total_negatives) + ") must be greater than zero.\n");

    const MatrixR roc_curve = calculate_roc_curve(targets, outputs);

    const float area_under_curve = calculate_area_under_curve(roc_curve);

    const float Q_1 = area_under_curve/(2.0f - area_under_curve);
    const float Q_2 = (2.0f * area_under_curve * area_under_curve) / (1.0f + area_under_curve);

    constexpr float z_95 = 1.64485f;  // one-sided 95% confidence Z-score
    const float auc_squared = area_under_curve * area_under_curve;
    return z_95 * sqrt((area_under_curve * (1.0f - area_under_curve)
                        + (float(total_positives) - 1.0f) * (Q_1 - auc_squared)
                        + (float(total_negatives) - 1.0f) * (Q_2 - auc_squared))
                       / float(total_positives * total_negatives));
}

float TestingAnalysis::calculate_optimal_threshold(const MatrixR& roc_curve) const
{
    const Index points_number = roc_curve.rows();

    float optimal_threshold = 0.5f;

    float minimum_distance = MAX;

    for (Index i = 0; i < points_number; ++i)
    {
        const float distance = hypot(roc_curve(i, 0), roc_curve(i, 1) - 1.0f);

        if (distance < minimum_distance)
        {
            optimal_threshold = roc_curve(i,2);

            minimum_distance = distance;
        }
    }

    return optimal_threshold;
}

MatrixR TestingAnalysis::perform_cumulative_gain_analysis() const
{
    const auto [targets, outputs] = get_targets_and_outputs("Testing");

    return calculate_cumulative_gain(targets, outputs);
}

MatrixR TestingAnalysis::calculate_cumulative_gain_impl(const MatrixR& targets, const MatrixR& outputs, bool positive) const
{
    const VectorI rates = calculate_positives_negatives_rate(targets, outputs);
    const Index total = positive ? rates[0] : rates[1];
    const string label = positive ? "positive" : "negative";

    if (total == 0)
        throw runtime_error("Number of " + label + " samples(" + to_string(total) + ") must be greater than zero.\n");

    const Index testing_samples_number = targets.rows();

    VectorI sorted_indices(outputs.rows());
    iota(sorted_indices.data(), sorted_indices.data() + sorted_indices.size(), 0);

    stable_sort(sorted_indices.data(),
                sorted_indices.data() + sorted_indices.size(),
                [&outputs](Index i1, Index i2) { return outputs(i1, 0) > outputs(i2, 0); });

    VectorR sorted_targets(testing_samples_number);

    for (Index i = 0; i < testing_samples_number; ++i)
        sorted_targets(i) = targets(sorted_indices(i), 0);

    const Index points_number = 21;
    const float percentage_increment = 0.05f;

    MatrixR cumulative_gain = MatrixR::Zero(points_number, 2);

    for (Index i = 0; i < points_number - 1; ++i)
    {
        const float percentage = float(i + 1) * percentage_increment;

        const Index maximum_index = Index(percentage * float(testing_samples_number));

        const Index count = count_if(sorted_targets.data(), sorted_targets.data() + maximum_index,
            [&](float t) { return positive ? double(t) == 1.0 : t < EPSILON; });

        cumulative_gain(i + 1, 0) = percentage;
        cumulative_gain(i + 1, 1) = float(count) / float(total);
    }

    return cumulative_gain;
}

MatrixR TestingAnalysis::calculate_cumulative_gain(const MatrixR& targets, const MatrixR& outputs) const
{
    return calculate_cumulative_gain_impl(targets, outputs, true);
}

MatrixR TestingAnalysis::calculate_negative_cumulative_gain(const MatrixR& targets, const MatrixR& outputs) const
{
    return calculate_cumulative_gain_impl(targets, outputs, false);
}

MatrixR TestingAnalysis::perform_lift_chart_analysis() const
{
    const auto [targets, outputs] = get_targets_and_outputs("Testing");

    const MatrixR cumulative_gain = calculate_cumulative_gain(targets, outputs);

    return calculate_lift_chart(cumulative_gain);
}

MatrixR TestingAnalysis::calculate_lift_chart(const MatrixR& cumulative_gain) const
{
    const Index rows_number = cumulative_gain.rows();
    const Index variables_number = cumulative_gain.cols();

    MatrixR lift_chart(rows_number, variables_number);

    lift_chart(0,0) = 0.0f;
    lift_chart(0,1) = 1.0f;

#pragma omp parallel for

    for (Index i = 1; i < rows_number; ++i)
    {
        const float gain_x = cumulative_gain(i, 0);
        lift_chart(i, 0) = gain_x;
        lift_chart(i, 1) = cumulative_gain(i, 1) / gain_x;
    }

    return lift_chart;
}

TestingAnalysis::KolmogorovSmirnovResults TestingAnalysis::perform_Kolmogorov_Smirnov_analysis() const
{
    const auto [targets, outputs] = get_targets_and_outputs("Testing");

    TestingAnalysis::KolmogorovSmirnovResults Kolmogorov_Smirnov_results;

    Kolmogorov_Smirnov_results.positive_cumulative_gain = calculate_cumulative_gain(targets, outputs);
    Kolmogorov_Smirnov_results.negative_cumulative_gain = calculate_negative_cumulative_gain(targets, outputs);
    Kolmogorov_Smirnov_results.maximum_gain =
        calculate_maximum_gain(Kolmogorov_Smirnov_results.positive_cumulative_gain, Kolmogorov_Smirnov_results.negative_cumulative_gain);

    return Kolmogorov_Smirnov_results;
}

VectorR TestingAnalysis::calculate_maximum_gain(const MatrixR& positive_cumulative_gain,
                                                const MatrixR& negative_cumulative_gain) const
{
    const Index points_number = positive_cumulative_gain.rows();

    VectorR maximum_gain = VectorR::Zero(2);

    const float percentage_increment = 0.05f;

    for (Index i = 0; i < points_number - 1; ++i)
    {
        const float percentage = float(i + 1) * percentage_increment;

        const float gain_diff = positive_cumulative_gain(i+1,1) - negative_cumulative_gain(i+1,1);

        if (gain_diff > maximum_gain[1] && gain_diff > 0.0f)
        {
            maximum_gain(1) = gain_diff;
            maximum_gain(0) = percentage;
        }
    }

    return maximum_gain;
}

vector<Histogram> TestingAnalysis::calculate_output_histogram(const MatrixR& outputs,
                                                              Index bins_number) const
{
    return { histogram(VectorR(outputs.col(0)), bins_number) };
}

TestingAnalysis::BinaryClassificationRates TestingAnalysis::calculate_binary_classification_rates(const float decision_threshold) const
{
    const auto [targets, outputs] = get_targets_and_outputs("Testing");

    const vector<Index> testing_indices = dataset->get_sample_indices("Testing");

    BinaryClassificationRates binary_classification_rates;

    binary_classification_rates.true_positives_indices = calculate_true_positive_samples(targets, outputs, testing_indices, decision_threshold);
    binary_classification_rates.false_positives_indices = calculate_false_positive_samples(targets, outputs, testing_indices, decision_threshold);
    binary_classification_rates.false_negatives_indices = calculate_false_negative_samples(targets, outputs, testing_indices, decision_threshold);
    binary_classification_rates.true_negatives_indices = calculate_true_negative_samples(targets, outputs, testing_indices, decision_threshold);

    return binary_classification_rates;
}

vector<Index> TestingAnalysis::filter_classification_samples(const MatrixR& targets,
                                                              const MatrixR& outputs,
                                                              const vector<Index>& testing_indices,
                                                              float decision_threshold,
                                                              bool target_positive,
                                                              bool output_positive) const
{
    const Index rows_number = targets.rows();

    vector<Index> result;
    result.reserve(rows_number);

    for (Index i = 0; i < rows_number; ++i)
    {
        const bool t_pos = targets(i, 0) >= decision_threshold;
        const bool o_pos = outputs(i, 0) >= decision_threshold;

        if (t_pos == target_positive && o_pos == output_positive)
            result.push_back(testing_indices[i]);
    }

    return result;
}

vector<Index> TestingAnalysis::calculate_true_positive_samples(const MatrixR& targets, const MatrixR& outputs,
                                                               const vector<Index>& testing_indices, float threshold) const
{
    return filter_classification_samples(targets, outputs, testing_indices, threshold, true, true);
}

vector<Index> TestingAnalysis::calculate_false_positive_samples(const MatrixR& targets, const MatrixR& outputs,
                                                                const vector<Index>& testing_indices, float threshold) const
{
    return filter_classification_samples(targets, outputs, testing_indices, threshold, false, true);
}

vector<Index> TestingAnalysis::calculate_false_negative_samples(const MatrixR& targets, const MatrixR& outputs,
                                                                const vector<Index>& testing_indices, float threshold) const
{
    return filter_classification_samples(targets, outputs, testing_indices, threshold, true, false);
}

vector<Index> TestingAnalysis::calculate_true_negative_samples(const MatrixR& targets, const MatrixR& outputs,
                                                               const vector<Index>& testing_indices, float threshold) const
{
    return filter_classification_samples(targets, outputs, testing_indices, threshold, false, false);
}

VectorR TestingAnalysis::calculate_multiple_classification_precision() const
{
    const auto [targets, outputs] = get_targets_and_outputs("Testing");

    VectorR multiple_classification_tests(2);

    const MatrixI confusion_matrix = calculate_confusion(targets, outputs);

    const Index classes_number = confusion_matrix.rows() - 1;
    const float total = static_cast<float>(confusion_matrix(classes_number, classes_number));

    const float diagonal_sum = static_cast<float>(confusion_matrix.topLeftCorner(classes_number, classes_number).diagonal().sum());

    multiple_classification_tests(0) = diagonal_sum / total;
    multiple_classification_tests(1) = (total - diagonal_sum) / total;

    return multiple_classification_tests;
}

void TestingAnalysis::save_confusion(const filesystem::path& file_name) const
{
    const MatrixI confusion = calculate_confusion();

    const Index classes_number = confusion.rows() - 1;

    ofstream file(file_name);

    const vector<string> target_variable_names = dataset->get_feature_names("Target");

    file << ",";

    for (Index i = 0; i < classes_number; ++i)
        file << target_variable_names[i] << (i == classes_number - 1 ? "\n" : ",");

    for (Index i = 0; i < classes_number; ++i)
    {
        file << target_variable_names[i] << ",";

        for (Index j = 0; j < classes_number; ++j)
            file << confusion(i, j) << (j == classes_number - 1 ? "\n" : ",");
    }

    file.close();
}

void TestingAnalysis::save_multiple_classification_tests(const filesystem::path& file_name) const
{
    const VectorR multiple_classification_tests = calculate_multiple_classification_precision();

    ofstream file(file_name);

    file << "accuracy,error" << "\n";
    file << multiple_classification_tests(0)* 100.0f << "," << multiple_classification_tests(1)* 100.0f << "\n";

    file.close();
}

Tensor<VectorI, 2> TestingAnalysis::calculate_multiple_classification_rates() const
{
    const auto [targets, outputs] = get_targets_and_outputs("Testing");

    const vector<Index> testing_indices = dataset->get_sample_indices("Testing");

    return calculate_multiple_classification_rates(targets, outputs, testing_indices);
}

Tensor<VectorI, 2> TestingAnalysis::calculate_multiple_classification_rates(const MatrixR& targets,
                                                                                    const MatrixR& outputs,
                                                                                    const vector<Index>& testing_indices) const
{
    const Index samples_number = targets.rows();
    const Index targets_number = targets.cols();

    Tensor< VectorI, 2> multiple_classification_rates(targets_number, targets_number);

    const MatrixI confusion = calculate_confusion(targets, outputs);

    for (Index i = 0; i < targets_number; ++i)
        for (Index j = 0; j < targets_number; ++j)
            multiple_classification_rates(i, j).resize(confusion(i, j));

    // Save indices

    MatrixI indices = MatrixI::Zero(targets_number, targets_number);

    for (Index i = 0; i < samples_number; ++i)
    {
        const Index target_index = maximal_index(targets.row(i));
        const Index output_index = maximal_index(outputs.row(i));

        multiple_classification_rates(target_index, output_index)(indices(target_index, output_index))
            = testing_indices[i];

        indices(target_index, output_index)++;
    }

    return multiple_classification_rates;
}

Tensor<string, 2> TestingAnalysis::classify_samples(const MatrixR& targets,
                                                    const MatrixR& outputs,
                                                    const vector<string>& labels,
                                                    bool match) const
{
    const Index samples_number = targets.rows();
    const vector<string> target_names = dataset->get_feature_names("Target");

    Tensor<string, 2> result(samples_number, 4);
    Index count = 0;

    for (Index i = 0; i < samples_number; ++i)
    {
        const Index actual = maximal_index(targets.row(i));
        const Index predicted = maximal_index(outputs.row(i));

        if ((actual == predicted) != match) continue;

        result(count, 0) = labels[i];
        result(count, 1) = target_names[actual];
        result(count, 2) = target_names[predicted];
        result(count, 3) = to_string(double(outputs(i, predicted)));
        ++count;
    }

    return result.slice(array<Index, 2>{0, 0}, array<Index, 2>{count, 4});
}

Tensor<string, 2> TestingAnalysis::calculate_well_classified_samples(const MatrixR& targets,
                                                                     const MatrixR& outputs,
                                                                     const vector<string>& labels) const
{
    return classify_samples(targets, outputs, labels, true);
}

Tensor<string, 2> TestingAnalysis::calculate_misclassified_samples(const MatrixR& targets,
                                                                   const MatrixR& outputs,
                                                                   const vector<string>& labels) const
{
    return classify_samples(targets, outputs, labels, false);
}

void TestingAnalysis::save_classified_samples_csv(const Tensor<string, 2>& samples, const filesystem::path& file_name) const
{
    ofstream file(file_name);

    file << "sample_name,actual_class,predicted_class,probability" << "\n";

    for (Index i = 0; i < samples.dimension(0); ++i)
        file << samples(i, 0) << "," << samples(i, 1) << ","
             << samples(i, 2) << "," << samples(i, 3) << "\n";

    file.close();
}

void TestingAnalysis::save_classified_samples_statistics_csv(const Tensor<string, 2>& samples, const filesystem::path& file_name) const
{
    const VectorR probabilities = extract_probabilities(samples);

    ofstream file(file_name);

    file << "minimum,maximum,mean,std" << "\n"
         << probabilities.minCoeff() << ","
         << probabilities.maxCoeff() << ","
         << probabilities.mean() << ","
         << standard_deviation(probabilities);
}

void TestingAnalysis::save_classified_samples_probability_histogram(const Tensor<string, 2>& samples, const filesystem::path& file_name) const
{
    const Histogram h(extract_probabilities(samples));
    h.save(file_name);
}

VectorR TestingAnalysis::extract_probabilities(const Tensor<string, 2>& samples)
{
    VectorR probabilities(samples.dimension(0));

    for (Index i = 0; i < probabilities.size(); ++i)
        probabilities(i) = float(::atof(samples(i, 3).c_str()));

    return probabilities;
}

void TestingAnalysis::save_well_classified_samples(const MatrixR& targets, const MatrixR& outputs,
                                                   const vector<string>& labels, const filesystem::path& file_name) const
{
    save_classified_samples_csv(calculate_well_classified_samples(targets, outputs, labels), file_name);
}

void TestingAnalysis::save_misclassified_samples(const MatrixR& targets, const MatrixR& outputs,
                                                 const vector<string>& labels, const filesystem::path& file_name) const
{
    save_classified_samples_csv(calculate_misclassified_samples(targets, outputs, labels), file_name);
}

void TestingAnalysis::save_well_classified_samples_statistics(const MatrixR& targets, const MatrixR& outputs,
                                                              const vector<string>& labels, const filesystem::path& file_name) const
{
    save_classified_samples_statistics_csv(calculate_well_classified_samples(targets, outputs, labels), file_name);
}

void TestingAnalysis::save_misclassified_samples_statistics(const MatrixR& targets, const MatrixR& outputs,
                                                            const vector<string>& labels, const filesystem::path& file_name) const
{
    save_classified_samples_statistics_csv(calculate_misclassified_samples(targets, outputs, labels), file_name);
}

vector<VectorR> TestingAnalysis::calculate_error_autocorrelation(const Index maximum_past_time_steps) const
{
    const auto [targets, outputs] = get_targets_and_outputs("Testing");

    const Index targets_number = dataset->get_features_number("Target");

    const MatrixR error = outputs - targets;

    vector<VectorR> error_autocorrelations(targets_number);

    for (Index i = 0; i < targets_number; ++i)
        error_autocorrelations[i] = autocorrelations(error.col(i), maximum_past_time_steps);

    return error_autocorrelations;
}

vector<VectorR> TestingAnalysis::calculate_inputs_errors_cross_correlation(const Index past_time_steps) const
{
    const Index targets_number = dataset->get_features_number("Target");

    const vector<Index> sample_indices         = dataset->get_sample_indices("Testing");
    const vector<Index> input_feature_indices  = dataset->get_feature_indices("Input");
    const vector<Index> target_feature_indices = dataset->get_feature_indices("Target");
    const Index samples_n = ssize(sample_indices);

    MatrixR inputs(samples_n, ssize(input_feature_indices));
    MatrixR targets(samples_n, ssize(target_feature_indices));
    dataset->fill_inputs(sample_indices, input_feature_indices,
                         inputs.data(), /*is_training=*/false, /*parallelize=*/true);
    dataset->fill_targets(sample_indices, target_feature_indices,
                          targets.data(), /*is_training=*/false, /*parallelize=*/true);

    const MatrixR outputs = neural_network->calculate_outputs(inputs);

    const MatrixR errors = outputs - targets;

    vector<VectorR> inputs_errors_cross_correlation(targets_number);

    for (Index i = 0; i < targets_number; ++i)
        inputs_errors_cross_correlation[i] = cross_correlations(inputs.col(i), errors.col(i), past_time_steps);

    return inputs_errors_cross_correlation;
}

pair<float, float> TestingAnalysis::test_transformer() const
{
    cout << "Testing transformer..." << "\n";

    const auto* transformer = dynamic_cast<Transformer*>(neural_network);
    throw_if(!transformer, "Expected Transformer neural network.");
    const auto* language_dataset = dynamic_cast<LanguageDataset*>(dataset);
    throw_if(!language_dataset, "Expected LanguageDataset.");

    const vector<Index> sample_indices   = language_dataset->get_sample_indices("Testing");
    const vector<Index> input_features   = language_dataset->get_feature_indices("Input");
    const vector<Index> decoder_features = language_dataset->get_feature_indices("Decoder");
    const vector<Index> target_features  = language_dataset->get_feature_indices("Target");
    const Index n = ssize(sample_indices);

    MatrixR context(n, ssize(input_features));
    MatrixR input(n, ssize(decoder_features));
    MatrixR target(n, ssize(target_features));
    language_dataset->fill_inputs(sample_indices, input_features,
                                  context.data(), /*is_training=*/false, /*parallelize=*/true);
    language_dataset->fill_decoder(sample_indices, decoder_features,
                                   input.data(), /*is_training=*/false, /*parallelize=*/true);
    language_dataset->fill_targets(sample_indices, target_features,
                                   target.data(), /*is_training=*/false, /*parallelize=*/true);

    const Index testing_batch_size = min(static_cast<Index>(2000), input.rows());

    MatrixR testing_input = input.topRows(testing_batch_size);
    MatrixR testing_context = context.topRows(testing_batch_size);
    MatrixR testing_target = target.topRows(testing_batch_size);

    //const Tensor3 outputs = transformer->calculate_outputs(testing_input, testing_context);

    // cout<<"English:"<<endl;
    // cout<<testing_context.chip(10,0)<<endl;
    // for (Index i = 0; i < testing_context.dimension(1); ++i)
    //     cout<<language_dataset->get_context_vocabulary()[Index(testing_context(10,i))]<<" ";
    // cout<<endl;
    // cout<<endl;
    // cout<<"Spanish:"<<endl;
    // cout<<testing_input.chip(10,0)<<endl;
    // for (Index i = 0; i < testing_input.dimension(1); ++i)
    //     cout<<language_dataset->get_completion_vocabulary()[Index(testing_input(10,i))]<<" ";
    // cout<<endl;
    // cout<<endl;
    // cout<<"Prediction:"<<endl;

    // for (Index j = 0; j < outputs.dimension(1); ++j) {
    //     float max = outputs(10, j, 0);
    //     Index index = 0;
    //     for (Index i = 1; i < outputs.dimension(2); ++i) {
    //         if (max < outputs(10,j,i)) {
    //             index = i;
    //             max = outputs(10,j,i);
    //         } else {continue;}
    //     }
    //     cout<<index<<" ";
    // }
    // cout<<endl;
    // for (Index j = 0; j < outputs.dimension(1); ++j) {
    //     float max = outputs(10, j, 0);
    //     Index index = 0;
    //     for (Index i = 1; i < outputs.dimension(2); ++i) {
    //         if (max < outputs(10,j,i)) {
    //             index = i;
    //             max = outputs(10,j,i);
    //         } else {continue;}
    //     }
    //     cout<<language_dataset->get_completion_vocabulary()[index]<<" ";
    // }
/*
    const float error = calculate_cross_entropy_error_3d(outputs, testing_target);

    const float accuracy = calculate_masked_accuracy(outputs, testing_target);

    return pair<float, float>(error, accuracy);
*/
    return {};
}

string TestingAnalysis::test_transformer(const vector<string>& /*context_string*/, bool /*imported_vocabulary*/) const
{
    cout << "Testing transformer..." << endl;
/*
    Transformer* transformer = static_cast<Transformer*>(neural_network);

    return transformer->calculate_outputs(context_string);
*/
    return string();
}

VectorR TestingAnalysis::calculate_binary_classification_tests(const float decision_threshold) const
{
    const MatrixI confusion = calculate_confusion(decision_threshold);

    const Index true_positive = confusion(0,0);
    const Index false_positive = confusion(1,0);
    const Index false_negative = confusion(0,1);
    const Index true_negative = confusion(1,1);

    const Index total = true_positive + true_negative + false_positive + false_negative;

    const float classification_accuracy = (total == 0)
                                             ? 0.0f
                                             : float(true_positive + true_negative) / float(total);

    const float error_rate = (total == 0)
                                ? 0.0f
                                : float(false_positive + false_negative) / float(total);

    const Index tp_plus_fn = true_positive + false_negative;
    const Index fp_plus_tn = false_positive + true_negative;
    const Index tp_plus_fp = true_positive + false_positive;

    const float sensitivity = (tp_plus_fn == 0) ? 0.0f : float(true_positive) / float(tp_plus_fn);

    const float false_positive_rate = (fp_plus_tn == 0) ? 0.0f : float(false_positive) / float(fp_plus_tn);

    const float specificity = (fp_plus_tn == 0) ? 0.0f : float(true_negative) / float(fp_plus_tn);

    const float precision = (tp_plus_fp == 0) ? 0.0f : float(true_positive) / float(tp_plus_fp);

    const bool accuracy_is_one = abs(classification_accuracy - 1.0f) < EPSILON;

    float positive_likelihood;

    if (accuracy_is_one)
        positive_likelihood = 1.0f;
    else if (abs(1.0f - specificity) < EPSILON)
        positive_likelihood = 0.0f;
    else
        positive_likelihood = sensitivity/(1.0f - specificity);

    float negative_likelihood;

    if (accuracy_is_one)
        negative_likelihood = 1.0f;
    else if (abs(1.0f - sensitivity) < EPSILON)
        negative_likelihood = 0.0f;
    else
        negative_likelihood = specificity/(1.0f - sensitivity);

    const Index f1_denominator = 2 * true_positive + false_positive + false_negative;
    const float f1_score = (f1_denominator == 0)
                              ? 0.0f
                              : 2.0f * float(true_positive) / float(f1_denominator);

    const float false_discovery_rate = (tp_plus_fp == 0) ? 0.0f : float(false_positive) / float(tp_plus_fp);

    const float false_negative_rate = (tp_plus_fn == 0) ? 0.0f : float(false_negative) / float(tp_plus_fn);

    const Index tn_plus_fn = true_negative + false_negative;

    const float negative_predictive_value = (tn_plus_fn == 0) ? 0.0f : float(true_negative) / float(tn_plus_fn);

    const Index matthews_denominator_squared = tp_plus_fp * tp_plus_fn * fp_plus_tn * tn_plus_fn;

    const float Matthews_correlation_coefficient = (matthews_denominator_squared == 0)
                                                      ? 0.0f
                                                      : float(true_positive * true_negative - false_positive * false_negative) / float(sqrt(matthews_denominator_squared));

    const float informedness = sensitivity + specificity - 1.0f;

    const float markedness = (fp_plus_tn == 0)
                                ? precision - 1.0f
                                : precision + negative_predictive_value - 1.0f;

    VectorR binary_classification_test(15);

    binary_classification_test << classification_accuracy,
                                  error_rate,
                                  sensitivity,
                                  specificity,
                                  precision,
                                  positive_likelihood,
                                  negative_likelihood,
                                  f1_score,
                                  false_positive_rate,
                                  false_discovery_rate,
                                  false_negative_rate,
                                  negative_predictive_value,
                                  Matthews_correlation_coefficient,
                                  informedness,
                                  markedness;

    return binary_classification_test;
}

void TestingAnalysis::print_binary_classification_tests() const
{
    const VectorR binary_classification_tests = calculate_binary_classification_tests();

    cout << "Binary classification tests: " << "\n"
         << "Classification accuracy : " << binary_classification_tests[0] << "\n"
         << "Error rate              : " << binary_classification_tests[1] << "\n"
         << "Sensitivity             : " << binary_classification_tests[2] << "\n"
         << "Specificity             : " << binary_classification_tests[3] << "\n";
}

MatrixR TestingAnalysis::calculate_multiple_classification_tests() const
{
    const Index targets_number = dataset->get_features_number("Target");

    MatrixR multiple_classification_tests(targets_number + 2, 3);

    const MatrixI confusion = calculate_confusion();

    float total_precision = 0.0f;
    float total_recall = 0.0f;
    float total_f1_score= 0.0f;

    float total_weighted_precision = 0.0f;
    float total_weighted_recall = 0.0f;
    float total_weighted_f1_score= 0.0f;

    Index total_samples = 0;

    for (Index target_index = 0; target_index < targets_number; ++target_index)
    {
        const Index true_positives = confusion(target_index, target_index);

        const Index row_sum = confusion(target_index, targets_number);
        const Index column_sum = confusion(targets_number, target_index);

        const Index false_negatives = row_sum - true_positives;
        const Index false_positives = column_sum - true_positives;

        const Index tp_plus_fp = true_positives + false_positives;
        const Index tp_plus_fn = true_positives + false_negatives;

        const float precision = (tp_plus_fp == 0) ? 1.0f : float(true_positives) / float(tp_plus_fp);
        const float recall    = (tp_plus_fn == 0) ? 1.0f : float(true_positives) / float(tp_plus_fn);

        const float f1_score = (precision + recall == 0)
                                  ? 0.0f
                                  : 2 * precision * recall / (precision + recall);

        multiple_classification_tests(target_index, 0) = precision;
        multiple_classification_tests(target_index, 1) = recall;
        multiple_classification_tests(target_index, 2) = f1_score;

        total_precision += precision;
        total_recall += recall;
        total_f1_score += f1_score;

        total_weighted_precision += precision * float(row_sum);
        total_weighted_recall += recall * float(row_sum);
        total_weighted_f1_score += f1_score * float(row_sum);

        total_samples += row_sum;
    }

    // Averages

    if (targets_number > 0)
    {
        multiple_classification_tests(targets_number, 0) = total_precision / targets_number;
        multiple_classification_tests(targets_number, 1) = total_recall / targets_number;
        multiple_classification_tests(targets_number, 2) = total_f1_score / targets_number;
    }

    if (total_samples > 0)
    {
        multiple_classification_tests(targets_number + 1, 0) = total_weighted_precision / total_samples;
        multiple_classification_tests(targets_number + 1, 1) = total_weighted_recall / total_samples;
        multiple_classification_tests(targets_number + 1, 2) = total_weighted_f1_score / total_samples;
    }

    return multiple_classification_tests;
}

void TestingAnalysis::to_JSON(JsonWriter& printer) const
{
    printer.open_element("TestingAnalysis");

    printer.close_element();
}

void TestingAnalysis::from_JSON(const JsonDocument& /*document*/)
{
}

void TestingAnalysis::save(const filesystem::path& file_name) const
{
    ofstream file(file_name);

    if (!file.is_open())
        throw runtime_error("Cannot open file: " + file_name.string());

    JsonWriter printer;

    to_JSON(printer);

    file << printer.c_str();
}

void TestingAnalysis::load(const filesystem::path& file_name)
{
    from_JSON(load_json_file(file_name));
}

void TestingAnalysis::GoodnessOfFitAnalysis::set(const VectorR& new_targets,
                                                 const VectorR& new_outputs,
                                                 float new_determination)
{
    targets = new_targets;
    outputs = new_outputs;
    determination = new_determination;
}

void TestingAnalysis::GoodnessOfFitAnalysis::save(const filesystem::path& file_name) const
{
    ofstream file(file_name);

    file << "Goodness-of-fit analysis\n"
         << "Determination: " << determination << "\n";

    file.close();
}

void TestingAnalysis::GoodnessOfFitAnalysis::print() const
{
    cout << "Goodness-of-fit analysis" << "\n"
         << "Determination: " << determination << "\n";

    // cout << "Targets:" << "\n";
    // cout << targets << "\n";
    // cout << "Outputs:" << "\n";
    // cout << outputs << "\n";
}

void TestingAnalysis::RocAnalysis::print() const
{
    cout << "ROC Curve analysis" << "\n";

//    cout << "Roc Curve:\n" << roc_curve << "\n";
    cout << "Area Under Curve: " << area_under_curve << "\n";
    cout << "Confidence Limit: " << confidence_limit << "\n";
    cout << "Optimal Threshold: " << optimal_threshold << "\n";
}

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
