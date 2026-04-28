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
    if(!neural_network)
        throw runtime_error("TestingAnalysis error: neural network is not set.");

    if(!dataset)
        throw runtime_error("TestingAnalysis error: dataset is not set.");
}

Tensor<Correlation, 1> TestingAnalysis::linear_correlation(const MatrixR& target, const MatrixR& output) const
{
    const Index outputs_number = dataset->get_features_number("Target");

    Tensor<Correlation, 1> linear_correlation(outputs_number);

    for(Index i = 0; i < outputs_number; ++i)
        linear_correlation(i) = opennn::linear_correlation(output.col(i), target.col(i));

    return linear_correlation;
}

void TestingAnalysis::print_linear_correlations() const
{
    const auto [targets, outputs] = get_targets_and_outputs("Testing");
    const Tensor<Correlation, 1> linear_correlations = linear_correlation(targets, outputs);

    const vector<string> targets_name = dataset->get_feature_names("Target");

    const Index targets_number = linear_correlations.size();

    for(Index i = 0; i < targets_number; ++i)
        cout << targets_name[i] << " correlation: " << linear_correlations[i].r << "\n";
}

Tensor<TestingAnalysis::GoodnessOfFitAnalysis, 1> TestingAnalysis::perform_goodness_of_fit_analysis() const
{
    check();

    // Dataset

    const Index testing_samples_number = dataset->get_samples_number("Testing");

    if(testing_samples_number == Index(0))
        throw runtime_error("Number of testing samples is zero.\n");

    // Neural network

    const Index outputs_number = neural_network->get_outputs_number();

    const pair<MatrixR, MatrixR> targets_outputs = get_targets_and_outputs("Testing");

    // Testing analysis

    Tensor<GoodnessOfFitAnalysis, 1> goodness_of_fit_results(outputs_number);

    for(Index i = 0;  i < outputs_number; ++i)
    {
        const VectorMap targets = vector_map(targets_outputs.first, i);
        const VectorMap outputs = vector_map(targets_outputs.second, i);

        const type determination = calculate_determination(outputs, targets);

        goodness_of_fit_results[i].set(targets, outputs, determination);
    }

    return goodness_of_fit_results;
}

void TestingAnalysis::print_goodness_of_fit_analysis() const
{
    const Tensor<GoodnessOfFitAnalysis, 1> goodness_of_fit_analysis = perform_goodness_of_fit_analysis();

    for(Index i = 0; i < goodness_of_fit_analysis.size(); ++i)
        goodness_of_fit_analysis(i).print();
}

pair<MatrixR, MatrixR> TestingAnalysis::get_targets_and_outputs(const string& sample_role) const
{
    check();

    // Dataset

    const Index samples_number = dataset->get_samples_number(sample_role);

    if(samples_number == Index(0))
        throw runtime_error("Number of samples is zero.\n");

    MatrixR output_data;
    MatrixR target_data;

    if (const TimeSeriesDataset* time_series_dataset = dynamic_cast<TimeSeriesDataset*>(dataset))
    {
        const Tensor3 input_data = time_series_dataset->get_data(sample_role, "Input");
        output_data = neural_network->calculate_outputs(input_data);

        const vector<Index> sample_indices = time_series_dataset->get_sample_indices(sample_role);
        const vector<Index> feature_indices = time_series_dataset->get_feature_indices("Target");
        target_data.resize(ssize(sample_indices), ssize(feature_indices));
        time_series_dataset->fill_targets(sample_indices, feature_indices, target_data.data());
    }
    else
    {
        target_data = dataset->get_data(sample_role, "Target");
        const MatrixR input_data = dataset->get_data(sample_role, "Input");
        output_data = neural_network->calculate_outputs(input_data);
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

    if(testing_samples_number == Index(0))
        throw runtime_error("Number of testing samples is zero.\n");

    const Index outputs_number = neural_network->get_outputs_number();

    const auto [targets, outputs] = get_targets_and_outputs("Testing");

    const auto* unscaling_layer = dynamic_cast<const Unscaling*>(neural_network->get_first("Unscaling"));

    if(!unscaling_layer)
        throw runtime_error("Unscaling layer not found.\n");

    const VectorR& output_minimums = unscaling_layer->get_minimums();
    const VectorR& output_maximums = unscaling_layer->get_maximums();

    Tensor3 error_data(testing_samples_number, 3, outputs_number);

    const MatrixR absolute_errors = (targets - outputs).array().abs();

#pragma omp parallel for
    for(Index i = 0; i < outputs_number; ++i)
    {
        const type range = abs(output_maximums(i) - output_minimums(i));

        for(Index j = 0; j < testing_samples_number; ++j)
        {
            error_data(j, 0, i) = absolute_errors(j,i);
            error_data(j, 1, i) = absolute_errors(j,i) / range;
            error_data(j, 2, i) = error_data(j, 1, i) * type(100.0);
        }
    }

    return error_data;
}

MatrixR TestingAnalysis::calculate_percentage_error_data() const
{
    check();

    // Dataset

    const Index testing_samples_number = dataset->get_samples_number("Testing");

    if(testing_samples_number == Index(0))
        throw runtime_error("Number of testing samples is zero.\n");

    // Neural network

    const Index outputs_number = neural_network->get_outputs_number();

    const auto [targets, outputs] = get_targets_and_outputs("Testing");

    const auto* unscaling_layer = dynamic_cast<const Unscaling*>(neural_network->get_first("Unscaling"));

    if(!unscaling_layer)
        throw runtime_error("Unscaling layer not found.\n");

    const VectorR& output_minimums = unscaling_layer->get_minimums();
    const VectorR& output_maximums = unscaling_layer->get_maximums();

    const MatrixR errors = (targets - outputs);

    // Error data

    MatrixR error_data(testing_samples_number, outputs_number);

#pragma omp parallel for

    for(Index i = 0; i < testing_samples_number; ++i)
        for(Index j = 0; j < outputs_number; ++j)
            error_data(i, j) = errors(i, j)*type(100.0)/abs(output_maximums(j) - output_minimums(j));

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
    const MatrixR difference = type(100)*(targets-outputs).array().abs()/targets.array();

    return descriptives(difference);
}

vector<vector<Descriptives>> TestingAnalysis::calculate_error_data_descriptives() const
{
    // Neural network

    const Index outputs_number = neural_network->get_outputs_number();

    const Index testing_samples_number = dataset->get_samples_number("Testing");

    // Testing analysis stuff

    vector<vector<Descriptives>> descriptives(outputs_number);

    Tensor3 error_data = calculate_error_data();

    Index index = 0;

    for(Index i = 0; i < outputs_number; ++i)
    {
        const MatrixMap matrix_error(error_data.data() + index, testing_samples_number, 3);

        const MatrixR matrix(matrix_error);

        descriptives[i] = opennn::descriptives(matrix);

        index += testing_samples_number*3;
    }

    return descriptives;
}

void TestingAnalysis::print_error_data_descriptives() const
{
    const Index targets_number = dataset->get_features_number("Target");

    const vector<string> targets_name = dataset->get_feature_names("Target");

    const vector<vector<Descriptives>> error_data_statistics = calculate_error_data_descriptives();

    for(Index i = 0; i < targets_number; ++i)
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

    for(Index i = 0; i < outputs_number; ++i)
        histograms[i] = histogram_centered(error_data.col(i), type(0), bins_number);

    return histograms;
}

Tensor<VectorI, 1> TestingAnalysis::calculate_maximal_errors(const Index samples_number) const
{
    Tensor3 error_data = calculate_error_data();

    const Index outputs_number = error_data.dimension(2);
    const Index testing_samples_number = error_data.dimension(0);

    Tensor<VectorI, 1> maximal_errors(samples_number);

    Index index = 0;

    for(Index i = 0; i < outputs_number; ++i)
    {
        const MatrixMap matrix_error(error_data.data()+index, testing_samples_number, 3);

        maximal_errors[i] = maximal_indices(matrix_error.col(0), samples_number);

        index += testing_samples_number*3;
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
    const VectorR training_errors = calculate_binary_classification_errors("Training");
    const VectorR validation_errors = calculate_binary_classification_errors("Validation");
    const VectorR testing_errors = calculate_binary_classification_errors("Testing");

    MatrixR errors(6, 3);

    errors.col(0) = training_errors;
    errors.col(1) = validation_errors;
    errors.col(2) = testing_errors;

    return errors;
}

MatrixR TestingAnalysis::calculate_multiple_classification_errors() const
{
    const VectorR training_errors = calculate_multiple_classification_errors("Training");
    const VectorR validation_errors = calculate_multiple_classification_errors("Validation");
    const VectorR testing_errors = calculate_multiple_classification_errors("Testing");

    MatrixR errors(5, 3);

    errors.col(0) = training_errors;
    errors.col(1) = validation_errors;
    errors.col(2) = testing_errors;

    return errors;
}

VectorR TestingAnalysis::calculate_errors(const MatrixR& targets,
                                          const MatrixR& outputs) const
{
    const TensorView outputs_view(const_cast<type*>(outputs.data()), {outputs.rows(), outputs.cols()});
    const TensorView targets_view(const_cast<type*>(targets.data()), {targets.rows(), targets.cols()});

    VectorR errors(5);

    // 1. Mean Squared Error
    mean_squared_error(outputs_view, targets_view, errors(1), nullptr);

    // 0. Sum Squared Error
    errors(0) = errors(1) * static_cast<type>(targets.size());

    // 2. Root Mean Squared Error
    errors(2) = sqrt(errors(1));

    // 3. Normalized Squared Error
    const VectorR targets_mean = mean(targets);
    const type normalization_coefficient = (targets.rowwise() - targets_mean.transpose()).squaredNorm();
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

    const TensorView outputs_view(const_cast<type*>(outputs.data()), {outputs.rows(), outputs.cols()});
    const TensorView targets_view(const_cast<type*>(targets.data()), {targets.rows(), targets.cols()});

    VectorR errors(6);

    const VectorR std_errors = calculate_errors(targets, outputs);
    errors.head(4) = std_errors.head(4);

    // 4. Binary Cross Entropy
    binary_cross_entropy(outputs_view, targets_view, errors(4), nullptr);

    // 5. Weighted Squared Error
    const VectorI target_distribution = dataset->calculate_target_distribution();
    const type neg_w = 1.0f;
    const type pos_w = (target_distribution[0] == 0 || target_distribution[1] == 0)
                           ? 1.0f
                           : static_cast<type>(target_distribution[0]) / target_distribution[1];

    weighted_squared_error(outputs_view, targets_view, pos_w, neg_w, errors(5), nullptr);

    return errors;
}

VectorR TestingAnalysis::calculate_multiple_classification_errors(const string& sample_role) const
{
    const auto [targets, outputs] = get_targets_and_outputs(sample_role);

    const TensorView outputs_view(const_cast<type*>(outputs.data()), {outputs.rows(), outputs.cols()});
    const TensorView targets_view(const_cast<type*>(targets.data()), {targets.rows(), targets.cols()});

    VectorR errors(5);

    const VectorR std_errors = calculate_errors(targets, outputs);
    errors.head(4) = std_errors.head(4);

    // 4. Categorical Cross Entropy
    categorical_cross_entropy(outputs_view, targets_view, errors(4), nullptr);

    return errors;
}

type TestingAnalysis::calculate_masked_accuracy(const Tensor3& /*outputs*/, const MatrixR& /*targets*/) const
{
/*
    const Index batch_size = outputs.rows();
    const Index outputs_number = outputs.cols();

    MatrixR predictions(batch_size, outputs_number);
    MatrixB matches(batch_size, outputs_number);
    MatrixB mask(batch_size, outputs_number);

    Tensor0 accuracy;

    mask = targets != targets.constant(0);

    const Tensor0 mask_sum = mask.cast<type>().sum();

    predictions = outputs.argmax(2).cast<type>();

    matches = predictions == targets;

    matches = matches && mask;

    accuracy = matches.cast<type>().sum() / mask_sum(0);

    return accuracy(0);
*/
    return 0;
}

type TestingAnalysis::calculate_determination(const VectorR& outputs, const VectorR& targets) const
{
    const type targets_mean = targets.mean();
    const type outputs_mean = outputs.mean();

    const type numerator = ((targets.array() - targets_mean) * (outputs.array() - outputs_mean)).sum();

    const type targets_ss = (targets.array() - targets_mean).square().sum();
    const type outputs_ss = (outputs.array() - outputs_mean).square().sum();

    const type denominator = sqrt(targets_ss * outputs_ss);

    if(denominator < EPSILON)
        return type(0);

    const type r = numerator / denominator;

    return r * r;
}

vector<MatrixI> TestingAnalysis::calculate_multilabel_confusion(const type decision_threshold) const
{
    check();

    const auto [targets, outputs] = get_targets_and_outputs("Testing");
    const Index outputs_number = neural_network->get_outputs_number();

    vector<MatrixI> confusion_matrices(static_cast<size_t>(outputs_number));

    for(Index j = 0; j < outputs_number; ++j)
        confusion_matrices[static_cast<size_t>(j)] = calculate_confusion(targets.col(j), outputs.col(j), decision_threshold);

    return confusion_matrices;
}

VectorI TestingAnalysis::calculate_positives_negatives_rate(const MatrixR& targets, const MatrixR& outputs) const
{
    const MatrixI confusion = calculate_confusion(targets, outputs, type(0.5));

    VectorI positives_negatives_rate(2);

    positives_negatives_rate << (confusion(0,0) + confusion(0,1)),
                                (confusion(1,0) + confusion(1,1));

    return positives_negatives_rate;
}

MatrixI TestingAnalysis::calculate_confusion(const type decision_threshold) const
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

    for(const vector<Index>& current_batch_indices : testing_batches)
    {
        const Index current_batch_size = current_batch_indices.size();
        if (current_batch_size == 0) continue;

        MatrixR batch_inputs_flat = dataset->get_data_from_indices(current_batch_indices, input_feature_indices);
        const MatrixR batch_targets = dataset->get_data_from_indices(current_batch_indices, target_feature_indices);

        MatrixR batch_outputs;

        if(input_shape.rank == 1)
            batch_outputs = neural_network->calculate_outputs(batch_inputs_flat);
        else if(input_shape.rank == 3)
        {
            Tensor4 inputs_4d(current_batch_size,
                              input_shape[0],
                              input_shape[1],
                              input_shape[2]);

            memcpy(inputs_4d.data(), batch_inputs_flat.data(),
                   current_batch_size * batch_inputs_flat.cols() * sizeof(type));

            batch_outputs = neural_network->calculate_outputs(inputs_4d);
        }
        else
            return {};

        const MatrixI batch_confusion = calculate_confusion(batch_targets, batch_outputs, decision_threshold);
        total_confusion_matrix += batch_confusion;
    }

    if(testing_indices.size() > 0)
        total_confusion_matrix(confusion_matrix_size - 1, confusion_matrix_size - 1) = testing_indices.size();

    return total_confusion_matrix;
}

MatrixI TestingAnalysis::calculate_confusion(const MatrixR& targets,
                                             const MatrixR& outputs,
                                             type decision_threshold) const
{
    const Index samples = targets.rows();
    const Index outputs_number = outputs.cols();
    const Index num_classes = (outputs_number == 1) ? 2 : outputs_number;

    MatrixI confusion = MatrixI::Zero(num_classes + 1, num_classes + 1);

    for(Index i = 0; i < samples; ++i)
    {
        Index target_class, output_class;

        if(outputs_number == 1)
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

    if(total_positives == 0)
        throw runtime_error("Number of positive samples (" + to_string(total_positives) + ") must be greater than zero.\n");

    if(total_negatives == 0)
        throw runtime_error("Number of negative samples (" + to_string(total_negatives) + ") must be greater than zero.\n");

    const Index points_number = 100;

    if(targets.cols() != 1)
        throw runtime_error("Number of of target variables (" +  to_string(targets.cols()) + ") must be one.\n");

    if(outputs.cols() != 1)
        throw runtime_error("Number of of output variables (" + to_string(outputs.cols()) + ") must be one.\n");

    MatrixR roc_curve = MatrixR::Zero(points_number + 1, 3);

#pragma omp parallel for schedule(dynamic)

    for(Index i = 1; i < Index(points_number); ++i)
    {
        const type threshold = type(i) * (type(1)/type(points_number));

        Index true_positive = 0;
        Index false_negative = 0;
        Index false_positive = 0;
        Index true_negative = 0;

        type target;
        type output;

        for(Index j = 0; j < targets.size(); ++j)
        {
            target = targets(j, 0);
            output = outputs(j, 0);

            if(target >= threshold && output >= threshold)
                ++true_positive;
            else if(target >= threshold && output < threshold)
                ++false_negative;
            else if(target < threshold && output >= threshold)
                ++false_positive;
            else if(target < threshold && output < threshold)
                ++true_negative;
        }

        roc_curve(i,0) = type(1) - type(true_positive)/type(true_positive + false_negative);
        roc_curve(i,1) = type(true_negative)/type(true_negative + false_positive);
        roc_curve(i,2) = type(threshold);

        if(isnan(roc_curve(i,0)))
            roc_curve(i,0) = type(1);

        if(isnan(roc_curve(i,1)))
            roc_curve(i,1) = type(0);
    }

    roc_curve(0,0) = type(0);
    roc_curve(0,1) = type(0);
    roc_curve(0,2) = type(0);

    roc_curve(points_number,0) = type(1);
    roc_curve(points_number,1) = type(1);
    roc_curve(points_number,2) = type(1);

    return roc_curve;
}

type TestingAnalysis::calculate_area_under_curve(const MatrixR& roc_curve) const
{
    type area_under_curve = type(0);

    for(Index i = 1; i < roc_curve.rows(); ++i)
        area_under_curve += (roc_curve(i,0) - roc_curve(i-1,0))*(roc_curve(i,1) + roc_curve(i-1,1));

    return area_under_curve/ type(2);
}

type TestingAnalysis::calculate_area_under_curve_confidence_limit(const MatrixR& targets, const MatrixR& outputs) const
{
    const VectorI positives_negatives_rate = calculate_positives_negatives_rate(targets, outputs);

    const Index total_positives = positives_negatives_rate[0];
    const Index total_negatives = positives_negatives_rate[1];

    if(total_positives == 0)
        throw runtime_error("Number of positive samples(" + to_string(total_positives) + ") must be greater than zero.\n");

    if(total_negatives == 0)
        throw runtime_error("Number of negative samples(" + to_string(total_negatives) + ") must be greater than zero.\n");

    const MatrixR roc_curve = calculate_roc_curve(targets, outputs);

    const type area_under_curve = calculate_area_under_curve(roc_curve);

    const type Q_1 = area_under_curve/(type(2.0) - area_under_curve);
    const type Q_2 = (type(2.0) * area_under_curve * area_under_curve) / (type(1) + area_under_curve);

    const type confidence_limit = type(type(1.64485)*sqrt((area_under_curve*(type(1) - area_under_curve)
                                                             + (type(total_positives) - type(1))*(Q_1-area_under_curve*area_under_curve)
                                                             + (type(total_negatives) - type(1))*(Q_2-area_under_curve*area_under_curve))/(type(total_positives*total_negatives))));

    return confidence_limit;
}

type TestingAnalysis::calculate_optimal_threshold(const MatrixR& roc_curve) const
{
    const Index points_number = roc_curve.rows();

    type optimal_threshold = type(0.5);

    type minimun_distance = MAX;

    for(Index i = 0; i < points_number; ++i)
    {
        const type distance = hypot(roc_curve(i, 0), roc_curve(i, 1) - type(1));

        if(distance < minimun_distance)
        {
            optimal_threshold = roc_curve(i,2);

            minimun_distance = distance;
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

    if(total == 0)
        throw runtime_error("Number of " + label + " samples(" + to_string(total) + ") must be greater than zero.\n");

    const Index testing_samples_number = targets.rows();

    VectorI sorted_indices(outputs.rows());
    iota(sorted_indices.data(), sorted_indices.data() + sorted_indices.size(), 0);

    stable_sort(sorted_indices.data(),
                sorted_indices.data() + sorted_indices.size(),
                [outputs](Index i1, Index i2) { return outputs(i1, 0) > outputs(i2, 0); });

    VectorR sorted_targets(testing_samples_number);

    for(Index i = 0; i < testing_samples_number; ++i)
        sorted_targets(i) = targets(sorted_indices(i), 0);

    const Index points_number = 21;
    const type percentage_increment = type(0.05);

    MatrixR cumulative_gain(points_number, 2);
    cumulative_gain(0, 0) = type(0);
    cumulative_gain(0, 1) = type(0);

    type percentage = type(0);

    for(Index i = 0; i < points_number - 1; ++i)
    {
        percentage += percentage_increment;

        Index count = 0;
        const Index maximum_index = Index(percentage * type(testing_samples_number));

        for(Index j = 0; j < maximum_index; ++j)
            if(positive ? double(sorted_targets(j)) == 1.0 : sorted_targets(j) < EPSILON)
                ++count;

        cumulative_gain(i + 1, 0) = percentage;
        cumulative_gain(i + 1, 1) = type(count) / type(total);
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

    lift_chart(0,0) = type(0);
    lift_chart(0,1) = type(1);

#pragma omp parallel for

    for(Index i = 1; i < rows_number; ++i)
    {
        lift_chart(i, 0) = type(cumulative_gain(i, 0));
        lift_chart(i, 1) = type(cumulative_gain(i, 1))/type(cumulative_gain(i, 0));
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

    const type percentage_increment = type(0.05);

    type percentage = type(0);

    for(Index i = 0; i < points_number - 1; ++i)
    {
        percentage += percentage_increment;

        if(positive_cumulative_gain(i+1,1)-negative_cumulative_gain(i+1,1) > maximum_gain[1]
        && positive_cumulative_gain(i+1,1)-negative_cumulative_gain(i+1,1) > type(0))
        {
            maximum_gain(1) = positive_cumulative_gain(i+1,1)-negative_cumulative_gain(i+1,1);
            maximum_gain(0) = percentage;
        }
    }

    return maximum_gain;
}

vector<Histogram> TestingAnalysis::calculate_output_histogram(const MatrixR& outputs,
                                                              Index bins_number) const
{
    const VectorR output_column = outputs.col(0);

    vector<Histogram> output_histogram(1);
    output_histogram[0] = histogram(output_column, bins_number);

    return output_histogram;
}

TestingAnalysis::BinaryClassificationRates TestingAnalysis::calculate_binary_classification_rates(const type decision_threshold) const
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
                                                              type decision_threshold,
                                                              bool target_positive,
                                                              bool output_positive) const
{
    const Index rows_number = targets.rows();

    vector<Index> result;
    result.reserve(rows_number);

    for(Index i = 0; i < rows_number; ++i)
    {
        const bool t_pos = targets(i, 0) >= decision_threshold;
        const bool o_pos = outputs(i, 0) >= decision_threshold;

        if(t_pos == target_positive && o_pos == output_positive)
            result.push_back(testing_indices[i]);
    }

    return result;
}

vector<Index> TestingAnalysis::calculate_true_positive_samples(const MatrixR& targets, const MatrixR& outputs,
                                                               const vector<Index>& testing_indices, type threshold) const
{
    return filter_classification_samples(targets, outputs, testing_indices, threshold, true, true);
}

vector<Index> TestingAnalysis::calculate_false_positive_samples(const MatrixR& targets, const MatrixR& outputs,
                                                                const vector<Index>& testing_indices, type threshold) const
{
    return filter_classification_samples(targets, outputs, testing_indices, threshold, false, true);
}

vector<Index> TestingAnalysis::calculate_false_negative_samples(const MatrixR& targets, const MatrixR& outputs,
                                                                const vector<Index>& testing_indices, type threshold) const
{
    return filter_classification_samples(targets, outputs, testing_indices, threshold, true, false);
}

vector<Index> TestingAnalysis::calculate_true_negative_samples(const MatrixR& targets, const MatrixR& outputs,
                                                               const vector<Index>& testing_indices, type threshold) const
{
    return filter_classification_samples(targets, outputs, testing_indices, threshold, false, false);
}

VectorR TestingAnalysis::calculate_multiple_classification_precision() const
{
    const auto [targets, outputs] = get_targets_and_outputs("Testing");

    VectorR multiple_classification_tests(2);

    const MatrixI confusion_matrix = calculate_confusion(targets, outputs);

    const Index n = confusion_matrix.rows() - 1;
    const type total = static_cast<type>(confusion_matrix(n, n));

    const type diagonal_sum = static_cast<type>(confusion_matrix.topLeftCorner(n, n).diagonal().sum());

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

    for(Index i = 0; i < classes_number; ++i)
    {
        file << target_variable_names[i];

        if(i != classes_number - 1)
            file << ",";
    }

    file << "\n";

    for(Index i = 0; i < classes_number; ++i)
    {
        file << target_variable_names[i] << ",";

        for(Index j = 0; j < classes_number; ++j)
            j == classes_number - 1
                ? file << confusion(i, j) << "\n"
                : file << confusion(i, j) << ",";
    }

    file.close();
}

void TestingAnalysis::save_multiple_classification_tests(const filesystem::path& file_name) const
{
    const VectorR multiple_classification_tests = calculate_multiple_classification_precision();

    ofstream file(file_name);

    file << "accuracy,error" << "\n";
    file << multiple_classification_tests(0)* type(100) << "," << multiple_classification_tests(1)* type(100) << "\n";

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

    for(Index i = 0; i < targets_number; ++i)
        for(Index j = 0; j < targets_number; ++j)
            multiple_classification_rates(i, j).resize(confusion(i, j));

    // Save indices

    MatrixI indices = MatrixI::Zero(targets_number, targets_number);

    for(Index i = 0; i < samples_number; ++i)
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

    for(Index i = 0; i < samples_number; ++i)
    {
        const Index actual = maximal_index(targets.row(i));
        const Index predicted = maximal_index(outputs.row(i));

        if((actual == predicted) != match) continue;

        result(count, 0) = labels[i];
        result(count, 1) = target_names[actual];
        result(count, 2) = target_names[predicted];
        result(count, 3) = to_string(double(outputs(i, predicted)));
        ++count;
    }

    return result.slice(Eigen::array<Index, 2>{0, 0}, Eigen::array<Index, 2>{count, 4});
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

    for(Index i = 0; i < samples.dimension(0); ++i)
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

    for(Index i = 0; i < probabilities.size(); ++i)
        probabilities(i) = type(::atof(samples(i, 3).c_str()));

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

    for(Index i = 0; i < targets_number; ++i)
        error_autocorrelations[i] = autocorrelations(error.col(i), maximum_past_time_steps);

    return error_autocorrelations;
}

vector<VectorR> TestingAnalysis::calculate_inputs_errors_cross_correlation(const Index past_time_steps) const
{
    const Index targets_number = dataset->get_features_number("Target");

    const MatrixR inputs = dataset->get_data("Testing", "Input");

    const MatrixR targets = dataset->get_data("Testing", "Target");

    const MatrixR outputs = neural_network->calculate_outputs(inputs);

    const MatrixR errors = outputs - targets;

    vector<VectorR> inputs_errors_cross_correlation(targets_number);

    for(Index i = 0; i < targets_number; ++i)
        inputs_errors_cross_correlation[i] = cross_correlations(inputs.col(i), errors.col(i), past_time_steps);

    return inputs_errors_cross_correlation;
}

pair<type, type> TestingAnalysis::test_transformer() const
{
    cout << "Testing transformer..." << "\n";

    const auto* transformer = dynamic_cast<Transformer*>(neural_network);
    if(!transformer) throw runtime_error("Expected Transformer neural network.");
    const auto* language_dataset = dynamic_cast<LanguageDataset*>(dataset);
    if(!language_dataset) throw runtime_error("Expected LanguageDataset.");

    const MatrixR context = language_dataset->get_data("Testing", "Input");
    const MatrixR input = language_dataset->get_data("Testing", "Decoder");
    const MatrixR target = language_dataset->get_data("Testing", "Target");

    const Index testing_batch_size = min(static_cast<Index>(2000), input.rows());

    MatrixR testing_input = input.topRows(testing_batch_size);
    MatrixR testing_context = context.topRows(testing_batch_size);
    MatrixR testing_target = target.topRows(testing_batch_size);

    //const Tensor3 outputs = transformer->calculate_outputs(testing_input, testing_context);

    // cout<<"English:"<<endl;
    // cout<<testing_context.chip(10,0)<<endl;
    // for(Index i = 0; i < testing_context.dimension(1); ++i)
    //     cout<<language_dataset->get_context_vocabulary()[Index(testing_context(10,i))]<<" ";
    // cout<<endl;
    // cout<<endl;
    // cout<<"Spanish:"<<endl;
    // cout<<testing_input.chip(10,0)<<endl;
    // for(Index i = 0; i < testing_input.dimension(1); ++i)
    //     cout<<language_dataset->get_completion_vocabulary()[Index(testing_input(10,i))]<<" ";
    // cout<<endl;
    // cout<<endl;
    // cout<<"Prediction:"<<endl;

    // for(Index j = 0; j < outputs.dimension(1); ++j){
    //     type max = outputs(10, j, 0);
    //     Index index = 0;
    //     for(Index i = 1; i < outputs.dimension(2); ++i){
    //         if(max < outputs(10,j,i)){
    //             index = i;
    //             max = outputs(10,j,i);
    //         }else{continue;}
    //     }
    //     cout<<index<<" ";
    // }
    // cout<<endl;
    // for(Index j = 0; j < outputs.dimension(1); ++j){
    //     type max = outputs(10, j, 0);
    //     Index index = 0;
    //     for(Index i = 1; i < outputs.dimension(2); ++i){
    //         if(max < outputs(10,j,i)){
    //             index = i;
    //             max = outputs(10,j,i);
    //         }else{continue;}
    //     }
    //     cout<<language_dataset->get_completion_vocabulary()[index]<<" ";
    // }
/*
    const type error = calculate_cross_entropy_error_3d(outputs, testing_target);

    const type accuracy = calculate_masked_accuracy(outputs, testing_target);

    return pair<type, type>(error, accuracy);
*/
    return {};
}

string TestingAnalysis::test_transformer(const vector<string>& /*context_string*/, bool /*imported_vocabulary*/) const
{
    cout<<"Testing transformer..."<<endl;
/*
    Transformer* transformer = static_cast<Transformer*>(neural_network);

    return transformer->calculate_outputs(context_string);
*/
    return string();
}

VectorR TestingAnalysis::calculate_binary_classification_tests(const type decision_threshold) const
{
    const MatrixI confusion = calculate_confusion(decision_threshold);

    const Index true_positive = confusion(0,0);
    const Index false_positive = confusion(1,0);
    const Index false_negative = confusion(0,1);
    const Index true_negative = confusion(1,1);

    const type classification_accuracy = (true_positive + true_negative + false_positive + false_negative == 0)
                                             ? type(0)
                                             : type(true_positive + true_negative) / type(true_positive + true_negative + false_positive + false_negative);

    const type error_rate = (true_positive + true_negative + false_positive + false_negative == 0)
                                ? type(0)
                                : type(false_positive + false_negative) / type(true_positive + true_negative + false_positive + false_negative);

    const type sensitivity = (true_positive + false_negative == 0)
                                 ? type(0)
                                 : type(true_positive) / type(true_positive + false_negative);

    const type false_positive_rate = (false_positive + true_negative == 0)
                                         ? type(0)
                                         : type(false_positive) / type(false_positive + true_negative);

    const type specificity = (false_positive + true_negative == 0)
                                 ? type(0)
                                 : type(true_negative) / type(true_negative + false_positive);

    const type precision = (true_positive + false_positive == 0)
                               ? type(0)
                               : type(true_positive) / type(true_positive + false_positive);

    type positive_likelihood;

    if(abs(classification_accuracy - type(1)) < EPSILON)
        positive_likelihood = type(1);
    else if(abs(type(1) - specificity) < EPSILON)
        positive_likelihood = type(0);
    else
        positive_likelihood = sensitivity/(type(1) - specificity);

    type negative_likelihood;

    if(abs(classification_accuracy - type(1)) < EPSILON)
        negative_likelihood = type(1);
    else if(abs(type(1) - sensitivity) < EPSILON)
        negative_likelihood = type(0);
    else
        negative_likelihood = specificity/(type(1) - sensitivity);

    const type f1_score = (2 * true_positive + false_positive + false_negative == 0)
                              ? type(0)
                              : type(2.0) * type(true_positive) / (type(2.0) * type(true_positive) + type(false_positive) + type(false_negative));

    const type false_discovery_rate = (false_positive + true_positive == 0)
                                          ? type(0)
                                          : type(false_positive) / type(false_positive + true_positive);

    const type false_negative_rate = (false_negative + true_positive == 0)
                                         ? type(0)
                                         : type(false_negative) / type(false_negative + true_positive);

    const type negative_predictive_value = (true_negative + false_negative == 0)
                                               ? type(0)
                                               : type(true_negative) / type(true_negative + false_negative);

    const type Matthews_correlation_coefficient = ((true_positive + false_positive) * (true_positive + false_negative) * (true_negative + false_positive) * (true_negative + false_negative) == 0)
                                                      ? type(0)
                                                      : type(true_positive * true_negative - false_positive * false_negative) / type(sqrt((true_positive + false_positive) * (true_positive + false_negative) * (true_negative + false_positive) * (true_negative + false_negative)));

    const type informedness = sensitivity + specificity - type(1);

    const type markedness = (true_negative + false_positive == 0)
                                ? precision - type(1)
                                : precision + negative_predictive_value - type(1);

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

    type total_precision = type(0);
    type total_recall = type(0);
    type total_f1_score= type(0);

    type total_weighted_precision = type(0);
    type total_weighted_recall = type(0);
    type total_weighted_f1_score= type(0);

    Index total_samples = 0;

    for(Index target_index = 0; target_index < targets_number; ++target_index)
    {
        const Index true_positives = confusion(target_index, target_index);

        const Index row_sum = confusion(target_index, targets_number);
        const Index column_sum = confusion(targets_number, target_index);

        const Index false_negatives = row_sum - true_positives;
        const Index false_positives = column_sum - true_positives;

        const type precision = (true_positives + false_positives == 0)
                                   ? type(1.0)
                                   : type(true_positives) / type(true_positives + false_positives);

        const type recall = (true_positives + false_negatives == 0)
                                ? type(1.0)
                                : type(true_positives) / type(true_positives + false_negatives);

        const type f1_score = (precision + recall == 0)
                                  ? type(0)
                                  : type(2 * precision * recall) / type(precision + recall);

        multiple_classification_tests(target_index, 0) = precision;
        multiple_classification_tests(target_index, 1) = recall;
        multiple_classification_tests(target_index, 2) = f1_score;

        total_precision += precision;
        total_recall += recall;
        total_f1_score += f1_score;

        total_weighted_precision += precision * type(row_sum);
        total_weighted_recall += recall * type(row_sum);
        total_weighted_f1_score += f1_score * type(row_sum);

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

void TestingAnalysis::to_XML(XmlPrinter& printer) const
{
    printer.open_element("TestingAnalysis");

    printer.close_element();
}

void TestingAnalysis::from_XML(const XmlDocument& /*document*/)
{
}

void TestingAnalysis::save(const filesystem::path& file_name) const
{
    ofstream file(file_name);

    if(!file.is_open())
        throw runtime_error("Cannot open file: " + file_name.string());

    XmlPrinter printer;

    to_XML(printer);

    file << printer.c_str();
}

void TestingAnalysis::load(const filesystem::path& file_name)
{
    from_XML(load_xml_file(file_name));
}

void TestingAnalysis::GoodnessOfFitAnalysis::set(const VectorR& new_targets,
                                                 const VectorR& new_outputs,
                                                 type new_determination)
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
