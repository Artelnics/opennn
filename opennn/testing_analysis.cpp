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
#include "error_functions.h"
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
    throw_if(!neural_network,
             "neural network is not set.");

    throw_if(!dataset,
             "dataset is not set.");
}

Tensor<Correlation, 1> TestingAnalysis::linear_correlation(const MatrixR& target, const MatrixR& output) const
{
    const Index outputs_number = dataset->get_features_number("Target");

    Tensor<Correlation, 1> linear_correlation(outputs_number);

    for (Index i = 0; i < outputs_number; ++i)
        linear_correlation(i) = opennn::linear_correlation(output.col(i), target.col(i));

    return linear_correlation;
}

Tensor<TestingAnalysis::GoodnessOfFitAnalysis, 1> TestingAnalysis::perform_goodness_of_fit_analysis() const
{
    check();


    const Index testing_samples_number = dataset->get_samples_number("Testing");

    throw_if(testing_samples_number == 0,
             "Number of testing samples is zero.\n");


    const Index outputs_number = neural_network->get_outputs_number();

    const auto [all_targets, all_outputs] = get_targets_and_outputs("Testing");
    Tensor<GoodnessOfFitAnalysis, 1> goodness_of_fit_results(outputs_number);

    for (Index i = 0; i < outputs_number; ++i)
    {
        const VectorR targets = all_targets.col(i);
        const VectorR outputs = all_outputs.col(i);

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

    const Index samples_number = dataset->get_samples_number(sample_role);

    throw_if(samples_number == 0,
             "Number of samples is zero.\n");

    const vector<Index> sample_indices = dataset->get_sample_indices(sample_role);
    const vector<Index> input_feature_indices  = dataset->get_feature_indices("Input");
    const vector<Index> target_feature_indices = dataset->get_feature_indices("Target");

    const Index target_width = dataset->get_target_shape().size();

    const Index default_batch_size =
        neural_network->is_gpu() ? Index(256) : samples_number;
    const Index current_batch_size =
        (batch_size <= 0) ? default_batch_size
                          : min<Index>(batch_size, samples_number);

    vector<vector<Index>> testing_batches;
    for (Index start = 0; start < samples_number; start += current_batch_size)
    {
        const Index end = min(start + current_batch_size, samples_number);
        testing_batches.emplace_back(sample_indices.begin() + start,
                                     sample_indices.begin() + end);
    }

    const Shape input_shape = dataset->get_shape("Input");

    MatrixR target_data(samples_number, target_width);
    MatrixR output_data;

    Index row_cursor = 0;
    for (const vector<Index>& batch_indices : testing_batches)
    {
        if (batch_indices.empty()) continue;
        const Index n = batch_indices.size();

        dataset->fill_targets(batch_indices, target_feature_indices,
                              target_data.data() + row_cursor * target_width,
                              /*is_training=*/false);

        MatrixR batch_outputs;
        if (auto* tsd = dynamic_cast<TimeSeriesDataset*>(dataset))
        {
            Tensor3 batch_inputs(n, input_shape[0], input_shape[1]);
            tsd->fill_inputs(batch_indices, input_feature_indices,
                             batch_inputs.data(), /*is_training=*/false);
            batch_outputs = neural_network->calculate_outputs(batch_inputs);
        }
        else if (input_shape.rank == 1)
        {
            MatrixR batch_inputs(n, input_shape[0]);
            dataset->fill_inputs(batch_indices, input_feature_indices,
                                 batch_inputs.data(), /*is_training=*/false);
            batch_outputs = neural_network->calculate_outputs(batch_inputs);
        }
        else if (input_shape.rank == 3)
        {
            Tensor4 batch_inputs(n, input_shape[0], input_shape[1], input_shape[2]);
            dataset->fill_inputs(batch_indices, input_feature_indices,
                                 batch_inputs.data(), /*is_training=*/false);
            batch_outputs = neural_network->calculate_outputs(batch_inputs);
        }
        else
        {
            throw runtime_error(format("Unsupported input rank {}", input_shape.rank));
        }

        if (output_data.size() == 0)
            output_data.resize(samples_number, batch_outputs.cols());

        output_data.middleRows(row_cursor, n) = batch_outputs;
        row_cursor += n;
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

    throw_if(testing_samples_number == 0,
             "Number of testing samples is zero.\n");

    const Index outputs_number = neural_network->get_outputs_number();

    const auto [targets, outputs] = get_targets_and_outputs("Testing");

    const auto* unscaling_layer = dynamic_cast<const Unscaling*>(neural_network->get_first("Unscaling"));

    throw_if(!unscaling_layer,
             "Unscaling layer not found.\n");

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


    const Index testing_samples_number = dataset->get_samples_number("Testing");

    throw_if(testing_samples_number == 0,
             "Number of testing samples is zero.\n");


    const Index outputs_number = neural_network->get_outputs_number();

    const auto [targets, outputs] = get_targets_and_outputs("Testing");

    const auto* unscaling_layer = dynamic_cast<const Unscaling*>(neural_network->get_first("Unscaling"));

    throw_if(!unscaling_layer,
             "Unscaling layer not found.\n");

    const VectorR& output_minimums = unscaling_layer->get_minimums();
    const VectorR& output_maximums = unscaling_layer->get_maximums();

    const VectorR ranges = (output_maximums - output_minimums).cwiseAbs();
    const MatrixR errors = targets - outputs;
    MatrixR error_data(testing_samples_number, outputs_number);

    error_data = ((errors.array() * 100.0f).rowwise() / ranges.transpose().array()).matrix();
    error_data = error_data.array().isFinite().select(error_data.array(), 0.0f).matrix();

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
    MatrixR difference = 100.0f*(targets-outputs).array().abs()/targets.array();
    difference = difference.array().isFinite().select(difference.array(), 0.0f).matrix();

    return descriptives(difference);
}

vector<vector<Descriptives>> TestingAnalysis::calculate_error_data_descriptives() const
{

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

    Tensor<VectorI, 1> maximal_errors(outputs_number);

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

    const Index batch_size = targets.rows();

    VectorR errors(5);

    const float sum_squared = (outputs.array() - targets.array()).square().sum();

    errors(0) = sum_squared;

    errors(1) = sum_squared / (2.0f * float(batch_size));

    errors(2) = std::sqrt(errors(1));

    const VectorR targets_mean = mean(targets);
    const float normalization_coefficient =
        (targets.rowwise() - targets_mean.transpose()).squaredNorm();
    errors(3) = sum_squared / (2.0f * (normalization_coefficient + EPSILON));

    const float p = 1.5f;
    errors(4) = (outputs.array() - targets.array())
                    .abs()
                    .pow(p)
                    .sum() / static_cast<float>(batch_size);

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

    binary_cross_entropy(outputs_view, targets_view, errors(4), nullptr);

    const VectorI target_distribution = dataset->calculate_target_distribution();
    const float negative_weight = 1.0f;
    const float positive_weight = (target_distribution[0] == 0 || target_distribution[1] == 0)
                           ? 1.0f
                           : static_cast<float>(target_distribution[0]) / target_distribution[1];

    weighted_squared_error(outputs_view, targets_view, positive_weight, negative_weight, errors(5), nullptr);

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

    categorical_cross_entropy(outputs_view, targets_view, errors(4), nullptr);

    return errors;
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
    const auto [targets, outputs] = get_targets_and_outputs("Testing");

    return calculate_confusion(targets, outputs, decision_threshold);
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

    throw_if(total_positives == 0,
             format("Number of positive samples ({}) must be greater than zero.\n", total_positives));

    throw_if(total_negatives == 0,
             format("Number of negative samples ({}) must be greater than zero.\n", total_negatives));

    const Index points_number = 100;

    throw_if(targets.cols() != 1,
             format("Number of of target variables ({}) must be one.\n", targets.cols()));

    throw_if(outputs.cols() != 1,
             format("Number of of output variables ({}) must be one.\n", outputs.cols()));

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

    throw_if(total_positives == 0,
             format("Number of positive samples({}) must be greater than zero.\n", total_positives));

    throw_if(total_negatives == 0,
             format("Number of negative samples({}) must be greater than zero.\n", total_negatives));

    const MatrixR roc_curve = calculate_roc_curve(targets, outputs);

    const float area_under_curve = calculate_area_under_curve(roc_curve);

    const float Q_1 = area_under_curve/(2.0f - area_under_curve);
    const float Q_2 = (2.0f * area_under_curve * area_under_curve) / (1.0f + area_under_curve);

    constexpr float z_95 = 1.64485f;
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

    const float positive_likelihood = accuracy_is_one ? 1.0f
        : (abs(1.0f - specificity) < EPSILON) ? 0.0f
        : sensitivity / (1.0f - specificity);

    const float negative_likelihood = accuracy_is_one ? 1.0f
        : (specificity < EPSILON) ? 0.0f
        : (1.0f - sensitivity) / specificity;

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

}

void TestingAnalysis::RocAnalysis::print() const
{
    cout << "ROC Curve analysis" << "\n";

    cout << "Area Under Curve: " << area_under_curve << "\n";
    cout << "Confidence Limit: " << confidence_limit << "\n";
    cout << "Optimal Threshold: " << optimal_threshold << "\n";
}

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
