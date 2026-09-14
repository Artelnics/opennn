// SPDX-License-Identifier: LGPL-2.1-or-later
// Copyright (C) 2005-2026 Artificial Intelligence Techniques, SL.

#include "opennn/evaluation/evaluation.h"

#include <sstream>
#include <numeric>
#include <limits>
#include <utility>

#include "opennn/dataset/dataset.h"
#include "opennn/core/parallel_algorithms.h"
#include "opennn/core/scaling.h"
#include "opennn/network/network.h"
#include "opennn/core/statistics.h"
#include "opennn/training/error_functions.h"
#include "opennn/network/forward_propagation.h"
#include "opennn/dataset/batch.h"

namespace opennn
{

namespace
{

FeatureScaling get_output_scaling(const Network& network)
{
    for (const unique_ptr<Layer>& layer : network.get_layers())
        if (const auto* endpoint = dynamic_cast<const FeatureScalingEndpoint*>(layer.get());
            endpoint && endpoint->get_scaling_role() == VariableRole::Target)
            return endpoint->get_feature_scaling();

    throw runtime_error("Output scaling endpoint not found.\n");
}

VectorR get_scaling_ranges(const FeatureScaling& scaling, Index outputs_number)
{
    throw_if(ssize(scaling.descriptives) != outputs_number,
             "Output scaling expects {} features, got {}.",
             outputs_number, scaling.descriptives.size());

    VectorR ranges(outputs_number);
    for (Index i = 0; i < outputs_number; ++i)
    {
        const Descriptives& descriptives = scaling.descriptives[size_t(i)];
        ranges(i) = abs(descriptives.maximum - descriptives.minimum);
    }

    return ranges;
}

void check_classification_data(const MatrixR& targets, const MatrixR& outputs, bool binary = false)
{
    throw_if(targets.rows() != outputs.rows() || targets.cols() != outputs.cols()
             || targets.cols() == 0 || (binary && targets.cols() != 1),
             "Classification requires equally shaped targets and outputs with {}.",
             binary ? "one column" : "at least one column");
}

template<typename Visit>
void for_each_classification(const MatrixR& targets, const MatrixR& outputs, float threshold, Visit&& visit,
                             bool binary_only = false)
{
    check_classification_data(targets, outputs, binary_only);
    const bool binary = outputs.cols() == 1;
    for (Index row = 0; row < targets.rows(); ++row)
    {
        const Index target = binary ? (targets(row, 0) >= threshold ? 0 : 1)
                                    : maximal_index(targets.row(row));
        const Index output = binary ? (outputs(row, 0) >= threshold ? 0 : 1)
                                    : maximal_index(outputs.row(row));
        visit(row, target, output);
    }
}

Index count_binary_positives(const MatrixR& targets, const MatrixR& outputs)
{
    check_classification_data(targets, outputs, true);
    return (targets.array() >= 0.5f).count();
}

float classification_ratio(float numerator, Index denominator)
{
    return denominator == 0 ? 0.0f : numerator / float(denominator);
}

}

Evaluation::Evaluation(Network* new_network, Dataset* new_dataset)
    : network(new_network), dataset(new_dataset)
{
}

void Evaluation::check() const
{
    throw_if(!network,
             "neural network is not set.");

    throw_if(!dataset,
             "dataset is not set.");
}

Tensor<Evaluation::GoodnessOfFitAnalysis, 1> Evaluation::perform_goodness_of_fit_analysis() const
{
    const auto [all_targets, all_outputs] = get_targets_and_outputs("Testing");
    const Index outputs_number = all_outputs.cols();
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

void Evaluation::print_goodness_of_fit_analysis() const
{
    const Tensor<GoodnessOfFitAnalysis, 1> goodness_of_fit_analysis = perform_goodness_of_fit_analysis();

    for (Index i = 0; i < goodness_of_fit_analysis.size(); ++i)
        goodness_of_fit_analysis(i).print();
}

pair<MatrixR, MatrixR> Evaluation::get_targets_and_outputs(const string& sample_role) const
{
    check();

    const vector<Index> sample_indices = dataset->get_sample_indices(sample_role);
    const Index samples_number = ssize(sample_indices);

    throw_if(samples_number == 0,
             "Number of samples is zero.\n");

    const FeatureSelection features = dataset->get_feature_selection();
    const Index target_width = dataset->get_target_shape().size();

    // Bound inference memory independently of the number of evaluated samples.
    constexpr Index maximum_cpu_batch_size = 4096;
    const Index default_batch_size =
        network->is_gpu() ? Index(256) : maximum_cpu_batch_size;
    const Index current_batch_size =
        min(batch_size <= 0 ? default_batch_size : batch_size, samples_number);

    MatrixR target_data(samples_number, target_width);
    MatrixR output_data;
    vector<Index> batch_indices;
    batch_indices.reserve(size_t(current_batch_size));

    EffectiveConfig host_config;
    host_config.device = Device::CPU;
    host_config.training_type = Type::FP32;

    // Reuse the CPU arena until the batch size changes. GPU residency and
    // graph capture remain owned by Network::calculate_outputs.
    const bool reuse_arena = !network->is_gpu();

    unique_ptr<Batch> batch;
    unique_ptr<ForwardPropagation> propagation;

    for (Index start = 0; start < samples_number; start += current_batch_size)
    {
        const Index n = min(current_batch_size, samples_number - start);
        batch_indices.assign(sample_indices.begin() + start, sample_indices.begin() + start + n);

        if (!batch || batch->get_batch_size() != n)
        {
            batch = make_unique<Batch>(n, dataset, host_config);
            if (reuse_arena)
                propagation = make_unique<ForwardPropagation>(
                    n, network, ForwardPropagationMode::Inference);
        }

        batch->fill(batch_indices, features, FillMode::Inference);
        if (target_width > 0)
            target_data.middleRows(start, n) = batch->get_targets().as_matrix();

        MatrixR batch_outputs;

        if (reuse_arena)
        {
            network->forward_propagate(batch->get_inputs(), *propagation,
                                       ForwardPropagationMode::Inference);
            batch_outputs = propagation->get_outputs().as_matrix();
        }
        else
        {
            batch_outputs = network->calculate_outputs(batch->get_inputs());
        }

        if (output_data.size() == 0)
            output_data.resize(samples_number, batch_outputs.cols());

        output_data.middleRows(start, n) = batch_outputs;
    }

    return {std::move(target_data), std::move(output_data)};
}

Tensor3 Evaluation::calculate_error_data() const
{
    const auto [targets, outputs] = get_targets_and_outputs("Testing");
    const Index testing_samples_number = targets.rows();
    const Index outputs_number = outputs.cols();

    const VectorR ranges = get_scaling_ranges(
        get_output_scaling(*network), outputs_number);

    Tensor3 error_data(testing_samples_number, 3, outputs_number);

    const MatrixR absolute_errors = (targets - outputs).array().abs();

#pragma omp parallel for
    for (Index i = 0; i < outputs_number; ++i)
    {
        const float range = ranges(i);

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

MatrixR Evaluation::calculate_percentage_error_data() const
{
    const auto [targets, outputs] = get_targets_and_outputs("Testing");

    const VectorR ranges = get_scaling_ranges(
        get_output_scaling(*network), outputs.cols());
    const MatrixR errors = targets - outputs;
    MatrixR error_data = ((errors.array() * 100.0f).rowwise() / ranges.transpose().array()).matrix();
    error_data = error_data.array().isFinite().select(error_data.array(), 0.0f).matrix();

    return error_data;
}

vector<vector<Descriptives>> Evaluation::calculate_error_data_descriptives() const
{

    const Tensor3 error_data = calculate_error_data();

    const Index testing_samples_number = error_data.dimension(0);
    const Index outputs_number = error_data.dimension(2);

    vector<vector<Descriptives>> descriptives(outputs_number);

    for (Index i = 0; i < outputs_number; ++i)
    {
        MatrixR matrix_error(testing_samples_number, 3);

        for (Index sample = 0; sample < testing_samples_number; ++sample)
            for (Index column = 0; column < 3; ++column)
                matrix_error(sample, column) = error_data(sample, column, i);

        descriptives[i] = opennn::descriptives(matrix_error);
    }

    return descriptives;
}

vector<Histogram> Evaluation::calculate_error_data_histograms(const Index bins_number) const
{
    const MatrixR error_data = calculate_percentage_error_data();

    const Index outputs_number = error_data.cols();

    vector<Histogram> histograms(outputs_number);

    for (Index i = 0; i < outputs_number; ++i)
        histograms[i] = histogram_centered(error_data.col(i), 0.0f, bins_number);

    return histograms;
}

VectorR Evaluation::calculate_errors(const MatrixR& targets,
                                          const MatrixR& outputs) const
{

    const Index samples_number = targets.rows();

    VectorR errors(5);

    const float sum_squared = (outputs.array() - targets.array()).square().sum();

    errors(0) = sum_squared;

    errors(1) = sum_squared / (2.0f * float(samples_number));

    errors(2) = sqrt(errors(1));

    const VectorR targets_mean = mean(targets);
    const float normalization_coefficient =
        (targets.rowwise() - targets_mean.transpose()).squaredNorm();
    errors(3) = sum_squared / (2.0f * (normalization_coefficient + EPSILON));

    const float p = 1.5f;
    errors(4) = (outputs.array() - targets.array())
                    .abs()
                    .pow(p)
                    .sum() / float(samples_number);

    return errors;
}

VectorR Evaluation::calculate_errors(const string& sample_role) const
{
    const auto [targets, outputs] = get_targets_and_outputs(sample_role);

    return calculate_errors(targets, outputs);
}

VectorR Evaluation::calculate_reconstruction_errors(const MatrixR& targets,
                                                         const MatrixR& reconstructions) const
{
    throw_if(targets.rows() == 0 || targets.cols() == 0,
             "Evaluation::calculate_reconstruction_errors: matrices cannot be empty.");
    throw_if(targets.rows() != reconstructions.rows() || targets.cols() != reconstructions.cols(),
             "Evaluation::calculate_reconstruction_errors: target and reconstruction shapes must match.");
    throw_if(!targets.array().isFinite().all() || !reconstructions.array().isFinite().all(),
             "Evaluation::calculate_reconstruction_errors: matrices must contain finite values.");

    return (targets - reconstructions).array().abs().rowwise().mean();
}

VectorR Evaluation::calculate_reconstruction_errors(const string& sample_role) const
{
    const auto [targets, reconstructions] = get_targets_and_outputs(sample_role);
    return calculate_reconstruction_errors(targets, reconstructions);
}

Evaluation::ReconstructionErrorStatistics
Evaluation::calculate_reconstruction_error_statistics(const VectorR& errors) const
{
    throw_if(errors.size() == 0,
             "Evaluation::calculate_reconstruction_error_statistics: errors cannot be empty.");
    throw_if(!errors.array().isFinite().all(),
             "Evaluation::calculate_reconstruction_error_statistics: errors must be finite.");

    const auto errors_double = errors.cast<double>();
    const double mean = errors_double.mean();
    const double variance = (errors_double.array() - mean).square().mean();

    return {errors.minCoeff(), errors.maxCoeff(), static_cast<float>(mean),
            static_cast<float>(sqrt(variance))};
}

float Evaluation::calculate_anomaly_threshold(
    const ReconstructionErrorStatistics& statistics,
    const float standard_deviations) const
{
    throw_if(!isfinite(standard_deviations) || standard_deviations < 0.0f,
             "Evaluation::calculate_anomaly_threshold: standard deviation multiplier must be finite and nonnegative.");
    throw_if(!isfinite(statistics.mean)
             || !isfinite(statistics.population_standard_deviation),
             "Evaluation::calculate_anomaly_threshold: statistics must be finite.");
    throw_if(statistics.population_standard_deviation < 0.0f,
             "Evaluation::calculate_anomaly_threshold: standard deviation cannot be negative.");

    const float threshold = statistics.mean
                          + standard_deviations * statistics.population_standard_deviation;
    throw_if(!isfinite(threshold),
             "Evaluation::calculate_anomaly_threshold: calculated threshold must be finite.");
    return threshold == 0.0f ? nextafter(0.0f, 1.0f) : threshold;
}

VectorI Evaluation::calculate_anomaly_predictions(const VectorR& errors,
                                                       const float threshold) const
{
    throw_if(!isfinite(threshold),
             "Evaluation::calculate_anomaly_predictions: threshold must be finite.");
    throw_if(!errors.array().isFinite().all(),
             "Evaluation::calculate_anomaly_predictions: errors must be finite.");

    return (errors.array() >= threshold).cast<Index>().matrix();
}

VectorR Evaluation::calculate_classification_errors(const string& sample_role, const bool binary) const
{
    const auto [targets, outputs] = get_targets_and_outputs(sample_role);

    const TensorView outputs_view(const_cast<float*>(outputs.data()), {outputs.rows(), outputs.cols()});
    const TensorView targets_view(const_cast<float*>(targets.data()), {targets.rows(), targets.cols()});

    VectorR errors(binary ? 6 : 5);

    const VectorR std_errors = calculate_errors(targets, outputs);
    errors.head(4) = std_errors.head(4);

    if (binary)
    {
        binary_cross_entropy(outputs_view, targets_view, errors(4), nullptr);

        const auto [negatives, positives] = dataset->count_binary_targets("Training");
        const float negative_weight = 1.0f;
        const float positive_weight = (negatives == 0 || positives == 0)
                               ? 1.0f
                               : static_cast<float>(negatives) / positives;

        weighted_squared_error(outputs_view, targets_view, positive_weight, negative_weight, errors(5), nullptr);
    }
    else
        categorical_cross_entropy(outputs_view, targets_view, errors(4), nullptr);

    return errors;
}

float Evaluation::calculate_determination(const VectorR& outputs, const VectorR& targets) const
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

VectorI Evaluation::calculate_positives_negatives_rate(const MatrixR& targets, const MatrixR& outputs) const
{
    const Index positives = count_binary_positives(targets, outputs);
    VectorI positives_negatives_rate(2);
    positives_negatives_rate << positives, targets.rows() - positives;
    return positives_negatives_rate;
}

MatrixI Evaluation::calculate_confusion(const float decision_threshold) const
{
    const auto [targets, outputs] = get_targets_and_outputs("Testing");

    return calculate_confusion(targets, outputs, decision_threshold);
}

MatrixI Evaluation::calculate_confusion(const MatrixR& targets,
                                             const MatrixR& outputs,
                                             float decision_threshold) const
{
    const Index outputs_number = outputs.cols();
    const Index num_classes = (outputs_number == 1) ? 2 : outputs_number;

    MatrixI confusion = MatrixI::Zero(num_classes + 1, num_classes + 1);

    for_each_classification(targets, outputs, decision_threshold, [&](Index, Index target_class, Index output_class)
    {
        confusion(target_class, output_class)++;
        confusion(target_class, num_classes)++;
        confusion(num_classes, output_class)++;
    });

    confusion(num_classes, num_classes) = targets.rows();

    return confusion;
}

Evaluation::RocAnalysis Evaluation::perform_roc_analysis() const
{
    const auto [targets, outputs] = get_targets_and_outputs("Testing");

    const VectorI positives_negatives_rate = calculate_positives_negatives_rate(targets, outputs);

    RocAnalysis roc_analysis;
    roc_analysis.roc_curve = calculate_roc_curve(targets, outputs);
    roc_analysis.area_under_curve = calculate_area_under_curve(roc_analysis.roc_curve);
    roc_analysis.confidence_limit = calculate_area_under_curve_confidence_limit(roc_analysis.area_under_curve,
                                                                                positives_negatives_rate(0),
                                                                                positives_negatives_rate(1));
    roc_analysis.optimal_threshold = calculate_optimal_threshold(roc_analysis.roc_curve);

    return roc_analysis;
}

MatrixR Evaluation::calculate_roc_curve(const MatrixR& targets, const MatrixR& outputs) const
{
    const Index total_positives = count_binary_positives(targets, outputs);
    throw_if(!targets.array().isFinite().all() || !outputs.array().isFinite().all(),
             "ROC targets and scores must be finite.");

    const Index total_negatives = targets.rows() - total_positives;
    throw_if(total_positives == 0 || total_negatives == 0,
             "ROC requires both positive and negative samples.");

    vector<Index> order(size_t(outputs.rows()));
    iota(order.begin(), order.end(), Index(0));
    sort(order.begin(), order.end(), [&](Index a, Index b)
         { return outputs(a, 0) < outputs(b, 0); });

    MatrixR roc_curve(outputs.rows() + 1, 3);
    Index true_positive = total_positives;
    Index false_positive = total_negatives;
    Index row = 0;
    for (const Index index : order)
    {
        const float threshold = outputs(index, 0);
        if (row == 0 || threshold != roc_curve(row - 1, 2))
        {
            roc_curve(row, 0) = float(false_positive) / float(total_negatives);
            roc_curve(row, 1) = float(true_positive) / float(total_positives);
            roc_curve(row++, 2) = threshold;
        }
        if (targets(index, 0) >= 0.5f) --true_positive;
        else --false_positive;
    }
    roc_curve(row, 0) = 0.0f;
    roc_curve(row, 1) = 0.0f;
    roc_curve(row++, 2) = nextafter(outputs(order.back(), 0),
                                   numeric_limits<float>::infinity());
    roc_curve.conservativeResize(row, 3);

    return roc_curve;
}

float Evaluation::calculate_area_under_curve(const MatrixR& roc_curve) const
{
    double area_under_curve = 0.0;

    for (Index i = 1; i < roc_curve.rows(); ++i)
        area_under_curve += (double(roc_curve(i,0)) - double(roc_curve(i-1,0)))
                          * (double(roc_curve(i,1)) + double(roc_curve(i-1,1)));

    return float(fabs(area_under_curve) / 2.0);
}

float Evaluation::calculate_area_under_curve_confidence_limit(const MatrixR& targets, const MatrixR& outputs) const
{
    const VectorI positives_negatives_rate = calculate_positives_negatives_rate(targets, outputs);

    const MatrixR roc_curve = calculate_roc_curve(targets, outputs);

    return calculate_area_under_curve_confidence_limit(calculate_area_under_curve(roc_curve),
                                                       positives_negatives_rate(0),
                                                       positives_negatives_rate(1));
}

float Evaluation::calculate_area_under_curve_confidence_limit(float area_under_curve,
                                                                   Index total_positives,
                                                                   Index total_negatives) const
{
    throw_if(total_positives == 0,
             "Number of positive samples({}) must be greater than zero.\n", total_positives);

    throw_if(total_negatives == 0,
             "Number of negative samples({}) must be greater than zero.\n", total_negatives);

    const float Q_1 = area_under_curve/(2.0f - area_under_curve);
    const float Q_2 = (2.0f * area_under_curve * area_under_curve) / (1.0f + area_under_curve);

    constexpr float z_95 = 1.64485f;
    const float auc_squared = area_under_curve * area_under_curve;
    return z_95 * sqrt((area_under_curve * (1.0f - area_under_curve)
                        + (float(total_positives) - 1.0f) * (Q_1 - auc_squared)
                        + (float(total_negatives) - 1.0f) * (Q_2 - auc_squared))
                       / float(total_positives * total_negatives));
}

float Evaluation::calculate_optimal_threshold(const MatrixR& roc_curve) const
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

MatrixR Evaluation::perform_lift_chart_analysis() const
{
    const auto [targets, outputs] = get_targets_and_outputs("Testing");

    return calculate_lift_chart(calculate_cumulative_gain(targets, outputs));
}

MatrixR Evaluation::calculate_cumulative_gain(const MatrixR& targets, const MatrixR& outputs) const
{
    const Index total_positives = count_binary_positives(targets, outputs);

    throw_if(total_positives == 0,
             "Number of positive samples ({}) must be greater than zero.\n", total_positives);

    const Index testing_samples_number = targets.rows();

    vector<Index> sorted_indices(static_cast<size_t>(testing_samples_number));
    iota(sorted_indices.begin(), sorted_indices.end(), Index(0));

    stable_sort_parallel_if_large(
        sorted_indices.begin(), sorted_indices.end(),
        [&outputs](Index i, Index j) { return outputs(i, 0) > outputs(j, 0); });

    const Index points_number = 21;
    const Index buckets_number = points_number - 1;

    MatrixR cumulative_gain(points_number, 2);

    cumulative_gain(0, 0) = 0.0f;
    cumulative_gain(0, 1) = 0.0f;

    Index positives = 0;
    Index next_row = 0;

    for (Index i = 0; i < buckets_number; ++i)
    {
        const Index maximum_index = min((i + 1) * testing_samples_number / buckets_number,
                                        testing_samples_number);

        for (; next_row < maximum_index; ++next_row)
            if (targets(sorted_indices[size_t(next_row)], 0) >= 0.5f)
                ++positives;

        cumulative_gain(i + 1, 0) = float(i + 1) / float(buckets_number);
        cumulative_gain(i + 1, 1) = float(positives) / float(total_positives);
    }

    return cumulative_gain;
}

MatrixR Evaluation::calculate_lift_chart(const MatrixR& cumulative_gain) const
{
    throw_if(cumulative_gain.rows() == 0 || cumulative_gain.cols() != 2,
             "Lift chart requires a nonempty two-column cumulative gain curve.");
    MatrixR lift_chart = cumulative_gain;
    lift_chart.row(0) << 0.0f, 1.0f;
    for (Index i = 1; i < lift_chart.rows(); ++i)
        lift_chart(i, 1) /= lift_chart(i, 0);
    return lift_chart;
}

Evaluation::BinaryClassificationRates Evaluation::calculate_binary_classification_rates(const float decision_threshold) const
{
    const auto [targets, outputs] = get_targets_and_outputs("Testing");

    const vector<Index> testing_indices = dataset->get_sample_indices(SampleRole::Testing);

    throw_if(ssize(testing_indices) != targets.rows(), "Classification sample-index count must match the data rows.");
    BinaryClassificationRates binary_classification_rates;
    const std::array<vector<Index>*, 4> cells{
        &binary_classification_rates.true_positives_indices,
        &binary_classification_rates.false_negatives_indices,
        &binary_classification_rates.false_positives_indices,
        &binary_classification_rates.true_negatives_indices};
    for_each_classification(targets, outputs, decision_threshold, [&](Index row, Index target, Index output)
    {
        cells[size_t(2 * target + output)]->push_back(testing_indices[size_t(row)]);
    }, true);
    return binary_classification_rates;
}

vector<Index> Evaluation::filter_classification_samples(const MatrixR& targets,
                                                              const MatrixR& outputs,
                                                              const vector<Index>& testing_indices,
                                                              float decision_threshold,
                                                              ConfusionCell cell) const
{
    throw_if(ssize(testing_indices) != targets.rows(), "Classification sample-index count must match the data rows.");
    const Index target_class = cell == ConfusionCell::TruePositive || cell == ConfusionCell::FalseNegative ? 0 : 1;
    const Index output_class = cell == ConfusionCell::TruePositive || cell == ConfusionCell::FalsePositive ? 0 : 1;

    vector<Index> result;
    result.reserve(targets.rows());
    for_each_classification(targets, outputs, decision_threshold, [&](Index row, Index target, Index output)
    {
        if (target == target_class && output == output_class)
            result.push_back(testing_indices[size_t(row)]);
    }, true);
    return result;
}

Tensor<VectorI, 2> Evaluation::calculate_multiple_classification_rates() const
{
    const auto [targets, outputs] = get_targets_and_outputs("Testing");

    const vector<Index> testing_indices = dataset->get_sample_indices(SampleRole::Testing);

    return calculate_multiple_classification_rates(targets, outputs, testing_indices);
}

Tensor<VectorI, 2> Evaluation::calculate_multiple_classification_rates(const MatrixR& targets,
                                                                                    const MatrixR& outputs,
                                                                                    const vector<Index>& testing_indices) const
{
    const Index targets_number = targets.cols();

    throw_if(targets_number < 2 || outputs.cols() != targets_number,
             "Evaluation::calculate_multiple_classification_rates requires one column per class "
             "(got {} target and {} output columns); use calculate_binary_classification_rates for a single output.",
             targets_number, outputs.cols());
    throw_if(ssize(testing_indices) != targets.rows(), "Classification sample-index count must match the data rows.");

    Tensor< VectorI, 2> multiple_classification_rates(targets_number, targets_number);

    MatrixI positions = calculate_confusion(targets, outputs);

    for (Index i = 0; i < targets_number; ++i)
        for (Index j = 0; j < targets_number; ++j)
            multiple_classification_rates(i, j).resize(positions(i, j));

    positions.setZero();
    for_each_classification(targets, outputs, 0.5f, [&](Index row, Index target_index, Index output_index)
    {
        VectorI& samples = multiple_classification_rates(target_index, output_index);
        samples(positions(target_index, output_index)++) = testing_indices[size_t(row)];
    });

    return multiple_classification_rates;
}

VectorR Evaluation::calculate_binary_classification_tests(const float decision_threshold) const
{
    const auto [targets, outputs] = get_targets_and_outputs("Testing");
    return calculate_binary_classification_tests(targets, outputs, decision_threshold);
}

VectorR Evaluation::calculate_binary_classification_tests(const MatrixR& targets,
                                                               const MatrixR& outputs,
                                                               const float decision_threshold) const
{
    check_classification_data(targets, outputs, true);
    const MatrixI confusion = calculate_confusion(targets, outputs, decision_threshold);

    const Index true_positive = confusion(0,0);
    const Index false_positive = confusion(1,0);
    const Index false_negative = confusion(0,1);
    const Index true_negative = confusion(1,1);

    const Index total = true_positive + true_negative + false_positive + false_negative;

    const float classification_accuracy = classification_ratio(float(true_positive + true_negative), total);
    const Index tp_plus_fn = true_positive + false_negative;
    const Index fp_plus_tn = false_positive + true_negative;
    const Index tp_plus_fp = true_positive + false_positive;

    const float sensitivity = classification_ratio(float(true_positive), tp_plus_fn);
    const float specificity = classification_ratio(float(true_negative), fp_plus_tn);
    const float precision = classification_ratio(float(true_positive), tp_plus_fp);

    const bool accuracy_is_one = abs(classification_accuracy - 1.0f) < EPSILON;

    const float positive_likelihood = accuracy_is_one ? 1.0f
        : (abs(1.0f - specificity) < EPSILON) ? 0.0f
        : sensitivity / (1.0f - specificity);

    const float negative_likelihood = accuracy_is_one ? 1.0f
        : (specificity < EPSILON) ? 0.0f
        : (1.0f - sensitivity) / specificity;

    const Index f1_denominator = 2 * true_positive + false_positive + false_negative;
    const Index tn_plus_fn = true_negative + false_negative;
    const float negative_predictive_value = classification_ratio(float(true_negative), tn_plus_fn);

    const double matthews_denominator = sqrt(double(tp_plus_fp) * double(tp_plus_fn)
                                           * double(fp_plus_tn) * double(tn_plus_fn));

    const float Matthews_correlation_coefficient = (matthews_denominator == 0.0)
                                                      ? 0.0f
                                                      : float((double(true_positive) * double(true_negative)
                                                             - double(false_positive) * double(false_negative))
                                                              / matthews_denominator);

    const float markedness = (fp_plus_tn == 0)
                                ? precision - 1.0f
                                : precision + negative_predictive_value - 1.0f;

    VectorR binary_classification_test(15);

    binary_classification_test << classification_accuracy,
                                  classification_ratio(float(false_positive + false_negative), total), // Error rate
                                  sensitivity,
                                  specificity,
                                  precision,
                                  positive_likelihood,
                                  negative_likelihood,
                                  classification_ratio(2.0f * float(true_positive), f1_denominator), // F1 score
                                  classification_ratio(float(false_positive), fp_plus_tn), // False positive rate
                                  classification_ratio(float(false_positive), tp_plus_fp), // False discovery rate
                                  classification_ratio(float(false_negative), tp_plus_fn), // False negative rate
                                  negative_predictive_value,
                                  Matthews_correlation_coefficient,
                                  sensitivity + specificity - 1.0f, // Informedness
                                  markedness;

    return binary_classification_test;
}

void Evaluation::print_binary_classification_tests() const
{
    const VectorR binary_classification_tests = calculate_binary_classification_tests();

    logging::info() << "Binary classification tests: " << "\n"
         << "Classification accuracy : " << binary_classification_tests[0] << "\n"
         << "Error rate              : " << binary_classification_tests[1] << "\n"
         << "Sensitivity             : " << binary_classification_tests[2] << "\n"
         << "Specificity             : " << binary_classification_tests[3] << "\n";
}

void Evaluation::print_multiple_classification_tests() const
{
    const auto [targets, outputs] = get_targets_and_outputs("Testing");
    const Index classes_number = targets.cols();

    throw_if(classes_number < 2 || outputs.cols() != classes_number,
             "Evaluation::print_multiple_classification_tests requires one column per class "
             "(got {} target and {} output columns); use print_binary_classification_tests for a single output.",
             classes_number, outputs.cols());

    const MatrixI confusion = calculate_confusion(targets, outputs);
    const Index samples_number = confusion(classes_number, classes_number);

    const Index correct = confusion.topLeftCorner(classes_number, classes_number).trace();
    const float accuracy = classification_ratio(float(correct), samples_number);

    ostringstream report;
    report << "Multiple classification tests: \n"
           << "Classification accuracy : " << accuracy << "\n"
           << "Confusion matrix:\n" << confusion << "\n";
    logging::info() << report.str();
}

void Evaluation::GoodnessOfFitAnalysis::set(const VectorR& new_targets,
                                                 const VectorR& new_outputs,
                                                 float new_determination)
{
    targets = new_targets;
    outputs = new_outputs;
    determination = new_determination;
}

void Evaluation::GoodnessOfFitAnalysis::save(const filesystem::path& file_name) const
{
    ofstream file(file_name);

    throw_if(!file.is_open(), "Cannot open file {}.", file_name.string());

    file << "Goodness-of-fit analysis\n"
         << "Determination: " << determination << "\n";
}

void Evaluation::GoodnessOfFitAnalysis::print() const
{
    logging::info() << "Goodness-of-fit analysis" << "\n"
         << "Determination: " << determination << "\n";

}

void Evaluation::RocAnalysis::print() const
{
    logging::info() << "ROC Curve analysis" << "\n";

    logging::info() << "Area Under Curve: " << area_under_curve << "\n";
    logging::info() << "Confidence Limit: " << confidence_limit << "\n";
    logging::info() << "Optimal Threshold: " << optimal_threshold << "\n";
}

}
