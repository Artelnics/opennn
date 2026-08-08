//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   C O R R E L A T I O N S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "tensor_types.h"
#include "correlations.h"
#include "parallel_algorithms.h"
#include "tabular_dataset.h"
#include "scaling_layer.h"
#include "dense_layer.h"
#include "neural_network.h"
#include "quasi_newton_method.h"
#include "levenberg_marquardt_algorithm.h"
#include "standard_networks.h"

namespace opennn
{

namespace
{

void set_confidence_interval(Correlation& correlation, Index sample_count)
{
    const float z = r_correlation_to_z_correlation(correlation.coefficient);
    const auto [ci_lower, ci_upper] = confidence_interval_z_correlation(z, sample_count);
    correlation.lower_confidence = z_correlation_to_r_correlation(ci_lower);
    correlation.upper_confidence = z_correlation_to_r_correlation(ci_upper);
}

float output_target_correlation(NeuralNetwork& neural_network, TabularDataset& dataset, Index& sample_count)
{
    const MatrixR inputs = dataset.get_feature_data("Input");
    const MatrixR targets = dataset.get_feature_data("Target");
    const MatrixR outputs = neural_network.calculate_outputs(inputs);

    sample_count = inputs.rows();

    return linear_correlation(outputs.reshaped(), targets.reshaped()).coefficient;
}

struct SoftmaxFitOptions
{
    Shape hidden_shape;
    bool set_binary_and_default_scalers = false;
    bool set_feature_shapes = false;
    const char* error_type = "MeanSquaredError";
    Index maximum_epochs = -1;
    Index display_period = -1;
    Index confidence_sample_count = -1;
};

Correlation fit_softmax_correlation(const MatrixR& x_filter,
                                    const MatrixR& y_filter,
                                    Correlation correlation,
                                    const SoftmaxFitOptions& options)
{
    MatrixR data(x_filter.rows(), x_filter.cols() + y_filter.cols());
    data << x_filter, y_filter;

    vector<Index> input_columns_indices(x_filter.cols());
    iota(input_columns_indices.begin(), input_columns_indices.end(), 0);

    vector<Index> target_columns_indices(y_filter.cols());
    iota(target_columns_indices.begin(), target_columns_indices.end(), x_filter.cols());

    TabularDataset dataset(x_filter.rows(), { x_filter.cols() }, { y_filter.cols() });

    dataset.set_data(data);
    dataset.set_variable_indices(input_columns_indices, target_columns_indices);

    if (options.set_binary_and_default_scalers)
    {
        dataset.set_binary_variables();
        dataset.set_default_variable_scalers();
    }

    dataset.set_sample_roles(SampleRole::Training);

    const Index input_features_number = dataset.get_features_number(VariableRole::Input);
    const Index target_features_number = dataset.get_features_number(VariableRole::Target);

    if (options.set_feature_shapes)
    {
        dataset.set_shape(VariableRole::Input, { input_features_number });
        dataset.set_shape(VariableRole::Target, { target_features_number });
    }

    ClassificationNetwork neural_network({ input_features_number }, options.hidden_shape, { target_features_number });

    neural_network.compile(Device::CPU);
    neural_network.set_parameters_glorot();

    auto* const dense_2d = dynamic_cast<Dense*>(neural_network.get_first(LayerType::Dense));
    throw_if(!dense_2d, "Expected Dense layer.");

    dense_2d->set_activation_function("Softmax");

    Loss loss(&neural_network, &dataset);
    loss.set_error(options.error_type);
    loss.set_regularization("None");

    QuasiNewtonMethod quasi_newton_method(&loss);
    if (options.maximum_epochs >= 0) quasi_newton_method.set_maximum_epochs(options.maximum_epochs);
    quasi_newton_method.set_display(false);
    if (options.display_period >= 0) quasi_newton_method.set_display_period(options.display_period);

    try
    {
        quasi_newton_method.train();
    }
    catch (const exception&)
    {
        correlation.coefficient = 0.0f;
        return correlation;
    }

    Index sample_count = 0;
    correlation.coefficient = output_target_correlation(neural_network, dataset, sample_count);

    set_confidence_interval(correlation,
                            options.confidence_sample_count >= 0 ? options.confidence_sample_count : sample_count);

    return correlation;
}

}

VectorR autocorrelations(const VectorR& x, Index past_time_steps)
{
    VectorR autocorrelation(past_time_steps);
    const Index this_size = x.size();

    for (Index i = 0; i < past_time_steps; ++i)
        autocorrelation(i) = linear_correlation(x.head(this_size - i), x.tail(this_size - i)).coefficient;

    return autocorrelation;
}

Correlation correlation(const MatrixR& x, const MatrixR& y)
{
    if (is_constant(x) || is_constant(y))
        return Correlation();

    const Index x_columns = x.cols();
    const Index y_columns = y.cols();

    const bool x_binary = is_binary(x);
    const bool y_binary = is_binary(y);

    if (x_columns == 1 && y_columns == 1)
    {
        const VectorR x_vector = x.col(0);
        const VectorR y_vector = y.col(0);

        if (!x_binary && !y_binary)
        {
            const Correlation linear_correlation
                = opennn::linear_correlation(x_vector, y_vector);

            const Correlation exponential_correlation
                = opennn::exponential_correlation(x_vector, y_vector);

            const Correlation logarithmic_correlation
                = opennn::logarithmic_correlation(x_vector, y_vector);

            const Correlation power_correlation
                = opennn::power_correlation(x_vector, y_vector);

            return max({linear_correlation, exponential_correlation, logarithmic_correlation, power_correlation},
                       [](const Correlation& a, const Correlation& b) {
                           return abs(a.coefficient) < abs(b.coefficient);
                       });
        }

        if (x_binary && y_binary)
            return opennn::linear_correlation(x_vector, y_vector);

        if (y_binary)
            return opennn::logistic_correlation(x_vector, y_vector);

        return opennn::logistic_correlation(y_vector, x_vector);
    }

    return logistic_correlation(x, y);
}

Correlation correlation_spearman(const MatrixR& x, const MatrixR& y)
{
    if (is_constant(x) || is_constant(y))
        return Correlation();

    if (x.cols() == 1 && y.cols() == 1)
    {
        const VectorR x_vector = x.col(0);
        const VectorR y_vector = y.col(0);

        const bool x_binary = is_binary(x);
        const bool y_binary = is_binary(y);

        Correlation result;

        if (x_binary != y_binary)
            result = y_binary ? logistic_correlation_spearman(x_vector, y_vector)
                              : logistic_correlation_spearman(y_vector, x_vector);
        else
            result = linear_correlation_spearman(x_vector, y_vector);

        result.method = Correlation::Method::Spearman;
        return result;
    }

    return correlation(x, y);
}

VectorR cross_correlations(const VectorR& x,
                           const VectorR& y,
                           Index maximum_past_time_steps)
{
    throw_if(y.size() != x.size(),
             "Both vectors must have the same size.\n");

    VectorR cross_correlation(maximum_past_time_steps);

    const Index this_size = x.size();

    for (Index i = 0; i < maximum_past_time_steps; ++i)
        cross_correlation[i] = linear_correlation(x.head(this_size - i), y.segment(i, this_size - i)).coefficient;

    return cross_correlation;
}

Correlation exponential_correlation(const VectorR& x, const VectorR& y)
{
    if ((y.array() <= 0.0f).any())
        return {.coefficient = QUIET_NAN};

    const VectorR log_y = y.array().log().matrix();

    Correlation result = linear_correlation(x, log_y);
    result.form = Correlation::Form::Exponential;
    result.intercept = exp(result.intercept);
    return result;
}

MatrixR get_correlation_values(const Tensor<Correlation, 2>& correlations)
{
    const Index rows_number = correlations.dimension(0);
    const Index columns_number = correlations.dimension(1);

    MatrixR values(rows_number, columns_number);

    for (Index i = 0; i < rows_number; ++i)
        for (Index j = 0; j < columns_number; ++j)
            values(i, j) = correlations(i, j).coefficient;

    return values;
}

Correlation linear_correlation(const VectorR& x,
                               const VectorR& y)
{
    throw_if(x.size() != y.size(),
             "Y size must be equal to X size.\n");

    if (is_constant(x) || is_constant(y))
        return Correlation();

    const auto [x_filter, y_filter] = filter_missing_values(x, y);

    const Index sample_count = x_filter.size();

    if (sample_count == 0)
        return Correlation();

    const auto x_double = x_filter.cast<double>();
    const auto y_double = y_filter.cast<double>();

    const double s_x = x_double.sum();
    const double s_y = y_double.sum();
    const double s_xx = x_double.squaredNorm();
    const double s_yy = y_double.squaredNorm();
    const double s_xy = x_double.dot(y_double);

    const double n = double(sample_count);
    const double sx_term = n * s_xx - s_x * s_x;
    const double sy_term = n * s_yy - s_y * s_y;
    const double xy_term = n * s_xy - s_x * s_y;

    const double denominator = sqrt(sx_term * sy_term);

    if (denominator < static_cast<double>(EPSILON))
        return Correlation();

    Correlation linear_correlation;
    linear_correlation.form = Correlation::Form::Identity;
    linear_correlation.intercept = static_cast<float>((s_y * s_xx - s_x * s_xy) / sx_term);
    linear_correlation.slope = static_cast<float>(xy_term / sx_term);
    linear_correlation.coefficient = static_cast<float>(xy_term / denominator);

    set_confidence_interval(linear_correlation, sample_count);
    linear_correlation.coefficient = clamp(linear_correlation.coefficient, -1.0f, 1.0f);

    return linear_correlation;
}

float r_correlation_to_z_correlation(const float r_correlation)
{
    const float r_clamped = clamp(r_correlation, -0.9999f, 0.9999f);

    return 0.5f * log((1 + r_clamped) / (1 - r_clamped));
}

float z_correlation_to_r_correlation(const float z_correlation)
{
    return tanh(z_correlation);
}

pair<float, float> confidence_interval_z_correlation(const float z_correlation, Index sample_count)
{
    if (sample_count <= 3)
        return { z_correlation, z_correlation };

    const float margin = 1.959964f / float(sqrt(sample_count - 3));

    return { z_correlation - margin, z_correlation + margin };
}

VectorR calculate_spearman_ranks(const VectorR& x)
{
    const Index size = x.size();

    if (size == 0) return {};

    VectorI sorted_indices(size);

    iota(sorted_indices.data(), sorted_indices.data() + size, 0);

    sort_parallel_if_large(
        sorted_indices.data(), sorted_indices.data() + size,
        [&](Index i, Index j) { return x(i) < x(j); });

    VectorR ranks(size);

    Index tie_start = 0;
    while (tie_start < size)
    {
        Index tie_end = tie_start;
        while (tie_end + 1 < size && x(sorted_indices(tie_end + 1)) == x(sorted_indices(tie_start)))
            ++tie_end;

        const float average_rank = float(tie_start + tie_end + 2) / 2.0f;

        for (Index i = tie_start; i <= tie_end; ++i)
            ranks(sorted_indices(i)) = average_rank;

        tie_start = tie_end + 1;
    }

    return ranks;
}

Correlation linear_correlation_spearman(const VectorR& x, const VectorR& y)
{
    const auto [x_filter, y_filter] = filter_missing_values(x, y);

    const VectorR x_rank = calculate_spearman_ranks(x_filter);
    const VectorR y_rank = calculate_spearman_ranks(y_filter);

    return linear_correlation(x_rank, y_rank);
}

Correlation logarithmic_correlation(const VectorR& x,
                                    const VectorR& y)
{
    if ((x.array() <= 0.0f).any())
        return {.coefficient = QUIET_NAN};

    const VectorR log_x = x.array().log().matrix();

    Correlation result = linear_correlation(log_x, y);
    result.form = Correlation::Form::Logarithmic;
    return result;
}

static constexpr Index maximum_levenberg_marquardt_samples = 10000;

static Correlation fit_logistic_correlation(const VectorR& input, const VectorR& target, const string& scaler)
{
    Correlation correlation;
    correlation.form = Correlation::Form::Sigmoid;

    MatrixR data(input.size(), 2);
    data.col(0) = input;
    data.col(1) = target;

    TabularDataset dataset(input.size(), {1}, {1});
    dataset.set_data(data);
    dataset.set_sample_roles(SampleRole::Training);
    dataset.set_variable_scalers(scaler);
    dataset.set_shape(VariableRole::Input, {1});
    dataset.set_shape(VariableRole::Target, {1});
    dataset.set_display(false);

    NeuralNetwork neural_network;
    const Shape dimensions = { 1 };
    neural_network.add_layer(make_unique<Scaling>(dimensions));
    neural_network.add_layer(make_unique<Dense>(dimensions, dimensions, "Sigmoid"));

    neural_network.compile(Device::CPU);

    Loss loss(&neural_network, &dataset);
    loss.set_error("MeanSquaredError");
    loss.set_regularization("None");

    try
    {
        if (input.size() > maximum_levenberg_marquardt_samples)
        {
            QuasiNewtonMethod quasi_newton(&loss);
            quasi_newton.set_display(false);

            quasi_newton.set_minimum_loss_decrease(1.0e-6f);

            quasi_newton.train();
        }
        else
        {
            LevenbergMarquardtAlgorithm levenberg_marquardt(&loss);
            levenberg_marquardt.set_display(false);
            levenberg_marquardt.train();
        }
    }
    catch (const exception&)
    {
        correlation.coefficient = 0.0f;
        return correlation;
    }

    Index sample_count = 0;
    correlation.coefficient = output_target_correlation(neural_network, dataset, sample_count);

    if (!isfinite(correlation.coefficient))
    {
        correlation.coefficient = 0.0f;
        return correlation;
    }

    set_confidence_interval(correlation, sample_count);

    const VectorR coefficients = Map<const VectorR, AlignedMax>(
        neural_network.get_parameters_data(), neural_network.get_parameters_buffer_size());
    correlation.intercept = coefficients(0);
    correlation.slope = coefficients(1);

    if (correlation.slope < 0.0f)
    {
        correlation.coefficient *= -1.0f;
        const float old_lower = correlation.lower_confidence;
        correlation.lower_confidence = -correlation.upper_confidence;
        correlation.upper_confidence = -old_lower;
    }

    return correlation;
}

Correlation logistic_correlation(const VectorR& x, const VectorR& y)
{
    const auto [x_filter, y_filter] = filter_missing_values(x, y);

    if (x_filter.size() < 2 || is_constant(x_filter) || is_constant(y_filter))
    {
        Correlation correlation;
        correlation.coefficient = QUIET_NAN;
        correlation.form = Correlation::Form::Sigmoid;
        return correlation;
    }

    return fit_logistic_correlation(x_filter, y_filter, "MeanStandardDeviation");
}

Correlation logistic_correlation_spearman(const VectorR& x, const VectorR& y)
{
    const auto [x_filter, y_filter] = filter_missing_values(x, y);

    if (x_filter.size() < 2)
    {
        Correlation correlation;
        correlation.coefficient = QUIET_NAN;
        correlation.form = Correlation::Form::Sigmoid;
        return correlation;
    }

    return fit_logistic_correlation(calculate_spearman_ranks(x_filter), y_filter, "MinimumMaximum");
}

Correlation logistic_correlation(const MatrixR& x, const MatrixR& y)
{
    Correlation correlation;
    correlation.form = Correlation::Form::Sigmoid;

    const auto [x_filter, y_filter] = filter_missing_values(x, y);

    if (x_filter.rows() == y_filter.rows()
        && x_filter.cols() == y_filter.cols()
        && (x_filter.array() == y_filter.array()).all())
    {
        correlation.coefficient = 1.0f;
        return correlation;
    }

    if (x.cols() > 50 || y.cols() > 50)
    {
        cerr << "Warning: One variable has too many categories.\n";

        correlation.coefficient = QUIET_NAN;
        return correlation;
    }

    if (x_filter.size() == 0 && y_filter.size() == 0)
    {
        correlation.coefficient = QUIET_NAN;
        return correlation;
    }

    return fit_softmax_correlation(x_filter, y_filter, correlation,
                                   {.error_type = "MeanSquaredError",
                                    .maximum_epochs = 500});
}

Correlation power_correlation(const VectorR& x, const VectorR& y)
{
    if ((x.array() <= 0.0f).any() || (y.array() <= 0.0f).any())
        return {.coefficient = QUIET_NAN};

    const VectorR log_x = x.array().log().matrix();
    const VectorR log_y = y.array().log().matrix();

    Correlation result = linear_correlation(log_x, log_y);
    result.form = Correlation::Form::Power;
    result.intercept = exp(result.intercept);
    return result;
}

void Correlation::set_perfect()
{
    coefficient = 1.0f;
    intercept = 0.0f;
    slope = 1.0f;

    upper_confidence = 1.0f;
    lower_confidence = 1.0f;
    form = Correlation::Form::Identity;
}

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
