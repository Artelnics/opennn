//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   C O R R E L A T I O N S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "tensor_utilities.h"
#include "correlations.h"
#include "tabular_dataset.h"
#include "scaling_layer.h"
#include "dense_layer.h"
#include "neural_network.h"
#include "quasi_newton_method.h"
#include "levenberg_marquardt_algorithm.h"
#include "standard_networks.h"

namespace opennn
{

VectorR autocorrelations(const VectorR& x, Index past_time_steps)
{
    VectorR autocorrelation(past_time_steps);
    const Index this_size = x.size();

    for (Index i = 0; i < past_time_steps; ++i)
        autocorrelation(i) = linear_correlation(x.head(this_size - i), x.tail(this_size - i)).r;

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
        const auto x_vector = x.col(0);
        const auto y_vector = y.col(0);

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
                           return abs(a.r) < abs(b.r);
                       });
        }

        if (y_binary && !x_binary)
            return opennn::point_biserial_correlation(x_vector, y_vector);

        if (x_binary && !y_binary)
            return opennn::point_biserial_correlation(y_vector, x_vector);

        return opennn::linear_correlation(x_vector, y_vector);
    }

    if (x_columns != 1 && y_columns == 1)
        return eta_squared_correlation(y.col(0), x);

    if (x_columns == 1 && y_columns != 1)
        return eta_squared_correlation(x.col(0), y);

    return logistic_correlation(x, y);
}

Correlation correlation_spearman(const MatrixR& x, const MatrixR& y)
{
    const Index x_columns = x.cols();
    const Index y_columns = y.cols();

    const bool x_binary = is_binary(x);
    const bool y_binary = is_binary(y);

    if (x_columns == 1 && y_columns == 1)
    {
        const auto x_vector = x.col(0);
        const auto y_vector = y.col(0);

        if (x_binary == y_binary)
            return linear_correlation_spearman(x_vector, y_vector);

        return x_binary
            ? logistic_correlation_spearman(y_vector, x_vector)
            : logistic_correlation_spearman(x_vector, y_vector);
    }

    if (x_columns == 1 && y_columns != 1)
        return logistic_correlation(VectorR(x.col(0)), y);

    if (x_columns != 1 && y_columns == 1)
        return logistic_correlation(x, VectorR(y.col(0)));

    return logistic_correlation(x, y);
}

VectorR cross_correlations(const VectorR& x,
                           const VectorR& y,
                           Index maximum_past_time_steps)
{
    if (y.size() != x.size())
        throw runtime_error("Both vectors must have the same size.\n");

    VectorR cross_correlation(maximum_past_time_steps);

    const Index this_size = x.size();

    for (Index i = 0; i < maximum_past_time_steps; ++i)
        cross_correlation[i] = linear_correlation(x.head(this_size - i), y.segment(i, this_size - i)).r;

    return cross_correlation;
}

Correlation exponential_correlation(const VectorR& x, const VectorR& y)
{
    Correlation exponential_correlation;

    for(Index i = 0; i < y.size(); ++i)
        if(!isnan(y(i)) && y(i) <= 0.0f)
        {
            exponential_correlation.r = NAN;
            return exponential_correlation;
        }

    const VectorR log_y = y.array().log().matrix();

    exponential_correlation = linear_correlation(x, log_y);

    exponential_correlation.form = Correlation::Form::Exponential;
    exponential_correlation.a = exp(exponential_correlation.a);
    return exponential_correlation;
}

MatrixR get_correlation_values(const Tensor<Correlation, 2>& correlations)
{
    const Index rows_number = correlations.dimension(0);
    const Index columns_number = correlations.dimension(1);

    MatrixR values(rows_number, columns_number);

    for (Index i = 0; i < rows_number; ++i)
        for (Index j = 0; j < columns_number; ++j)
            values(i, j) = correlations(i, j).r;

    return values;
}

Correlation linear_correlation(const VectorR& x,
                               const VectorR& y)
{
    if (x.size() != y.size())
        throw runtime_error("Y size must be equal to X size.\n");

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
    linear_correlation.a = static_cast<float>((s_y * s_xx - s_x * s_xy) / sx_term);
    linear_correlation.b = static_cast<float>(xy_term / sx_term);
    linear_correlation.r = static_cast<float>(xy_term / denominator);

    const float z_correlation = r_correlation_to_z_correlation(linear_correlation.r);

    const auto [ci_lower, ci_upper] = confidence_interval_z_correlation(z_correlation, sample_count);

    linear_correlation.lower_confidence = clamp(z_correlation_to_r_correlation(ci_lower), -1.0f, 1.0f);
    linear_correlation.upper_confidence = clamp(z_correlation_to_r_correlation(ci_upper), -1.0f, 1.0f);
    linear_correlation.r = clamp(linear_correlation.r, -1.0f, 1.0f);

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
    const float margin = 1.959964f / float(sqrt(sample_count - 3));

    return { z_correlation - margin, z_correlation + margin };
}

VectorR calculate_spearman_ranks(const VectorR& x)
{
    const Index size = x.size();

    if (size == 0) return {};

    VectorI sorted_indices(size);

    iota(sorted_indices.data(), sorted_indices.data() + size, 0);

    sort(sorted_indices.data(), sorted_indices.data() + size,
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
    Correlation logarithmic_correlation;

    for(Index i = 0; i < x.size(); ++i)
        if(!isnan(x(i)) && x(i) <= 0.0f)
        {
            logarithmic_correlation.r = NAN;
            return logarithmic_correlation;
        }

    const VectorR log_x = x.array().log().matrix();

    logarithmic_correlation = linear_correlation(log_x, y);

    logarithmic_correlation.form = Correlation::Form::Logarithmic;

    return logarithmic_correlation;
}

static Correlation fit_logistic_correlation(const VectorR& input, const VectorR& target, const string& scaler)
{
    Correlation correlation;
    correlation.form = Correlation::Form::Sigmoid;

    MatrixR data(input.size(), 2);
    data.col(0) = input;
    data.col(1) = target;

    TabularDataset dataset(input.size(), {1}, {1});
    dataset.set_data(data);
    dataset.set_sample_roles("Training");
    dataset.set_variable_scalers(scaler);
    dataset.set_shape("Input", {1});
    dataset.set_shape("Target", {1});
    dataset.set_display(false);

    NeuralNetwork neural_network;
    const Shape dimensions = { 1 };
    neural_network.add_layer(make_unique<Scaling>(dimensions));
    neural_network.add_layer(make_unique<Dense>(dimensions, dimensions, "Sigmoid"));
    neural_network.compile();

    Loss loss(&neural_network, &dataset);
    loss.set_error("MeanSquaredError");
    loss.set_regularization("None");

    LevenbergMarquardtAlgorithm levenberg_marquardt(&loss);
    levenberg_marquardt.set_display(false);

    try
    {
        levenberg_marquardt.train();
    }
    catch (const exception&)
    {
        correlation.r = 0.0f;
        return correlation;
    }

    const MatrixR inputs = dataset.get_feature_data("Input");
    const MatrixR targets = dataset.get_feature_data("Target");
    const MatrixR outputs = neural_network.calculate_outputs(inputs);

    correlation.r = linear_correlation(outputs.reshaped(), targets.reshaped()).r;

    if (!isfinite(correlation.r))
    {
        correlation.r = 0.0f;
        return correlation;
    }

    const float z_correlation = r_correlation_to_z_correlation(correlation.r);
    const auto [ci_lower, ci_upper] = confidence_interval_z_correlation(z_correlation, inputs.rows());

    correlation.lower_confidence = z_correlation_to_r_correlation(ci_lower);
    correlation.upper_confidence = z_correlation_to_r_correlation(ci_upper);

    const VectorR coefficients = Map<const VectorR, AlignedMax>(
        neural_network.get_parameters_data(), neural_network.get_parameters_size());
    correlation.a = coefficients(0);
    correlation.b = coefficients(1);

    if (correlation.b < 0.0f)
    {
        correlation.r *= -1.0f;
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
        correlation.r = NAN;
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
        correlation.r = NAN;
        correlation.form = Correlation::Form::Sigmoid;
        return correlation;
    }

    return fit_logistic_correlation(calculate_spearman_ranks(x_filter), y_filter, "MinimumMaximum");
}

Correlation logistic_correlation(const VectorR& x, const MatrixR& y)
{
    Correlation correlation;
    correlation.form = Correlation::Form::Sigmoid;

    const auto [x_filter, y_filter] = filter_missing_values(x, y);

    if (y_filter.cols() > 50)
    {
        cout << "Warning: Y variable has too many categories." << "\n";

        correlation.r = NAN;
        return correlation;
    }

    if (x_filter.size() == 0)
    {
        correlation.r = NAN;
        return correlation;
    }

    MatrixR data(x_filter.rows(), 1 + y_filter.cols());
    data << x_filter, y_filter;

    vector<Index> input_columns_indices = {0};

    vector<Index> target_columns_indices(y_filter.cols());
    iota(target_columns_indices.begin(), target_columns_indices.end(), 1);

    TabularDataset dataset(x_filter.size(), {1}, {y_filter.cols()});

    dataset.set_data(data);
    dataset.set_variable_indices(input_columns_indices, target_columns_indices);
    dataset.set_binary_variables();
    dataset.set_default_variable_scalers();

    dataset.set_sample_roles("Training");

    const Index input_features_number = dataset.get_features_number("Input");
    const Index target_features_number = dataset.get_features_number("Target");

    dataset.set_shape("Input", { input_features_number });
    dataset.set_shape("Target", { target_features_number });

    ClassificationNetwork neural_network({ input_features_number }, {1}, {target_features_number});

    auto* dense_2d = dynamic_cast<Dense*>(neural_network.get_first(LayerType::Dense));
    throw_if(!dense_2d, "Expected Dense layer.");

    dense_2d->set_activation_function("Softmax");

    Loss loss(&neural_network, &dataset);
    loss.set_error("CrossEntropy");
    loss.set_regularization("None");

    QuasiNewtonMethod quasi_newton_method(&loss);
    quasi_newton_method.set_display(false);
    quasi_newton_method.set_display_period(1000);

    try
    {
        quasi_newton_method.train();
    }
    catch (const exception&)
    {
        correlation.r = 0.0f;
        return correlation;
    }

    const MatrixR inputs = dataset.get_feature_data("Input");
    const MatrixR targets = dataset.get_feature_data("Target");
    const MatrixR outputs = neural_network.calculate_outputs(inputs);

    correlation.r = linear_correlation(outputs.reshaped(), targets.reshaped()).r;

    const float z_correlation = r_correlation_to_z_correlation(correlation.r);

    const auto [ci_lower, ci_upper] = confidence_interval_z_correlation(z_correlation, x_filter.size());

    correlation.lower_confidence = z_correlation_to_r_correlation(ci_lower);
    correlation.upper_confidence = z_correlation_to_r_correlation(ci_upper);

    return correlation;
}

Correlation logistic_correlation(const MatrixR& y, const VectorR& x)
{
    return logistic_correlation(x, y);
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
        correlation.r = 1.0f;
        return correlation;
    }

    if (x.cols() > 50 || y.cols() > 50)
    {
        cout << "Warning: One variable has too many categories." << "\n";

        correlation.r = NAN;
        return correlation;
    }

    if (x_filter.size() == 0 && y_filter.size() == 0)
    {
        correlation.r = NAN;
        return correlation;
    }

    MatrixR data(x_filter.rows(), x_filter.cols() + y_filter.cols());
    data << x_filter, y_filter;

    vector<Index> input_columns_indices(x_filter.cols());

    iota(input_columns_indices.begin(), input_columns_indices.end(), 0);

    vector<Index> target_columns_indices(y_filter.cols());
    iota(target_columns_indices.begin(), target_columns_indices.end(), x_filter.cols());

    TabularDataset dataset(x_filter.rows(), { x_filter.cols() }, { y_filter.cols() });

    dataset.set_data(data);

    dataset.set_variable_indices(input_columns_indices, target_columns_indices);

    dataset.set_sample_roles("Training");

    const Index input_features_number = dataset.get_features_number("Input");
    const Index target_features_number = dataset.get_features_number("Target");

    ClassificationNetwork neural_network({input_features_number }, {}, {target_features_number});

    auto* dense_2d = dynamic_cast<Dense*>(neural_network.get_first(LayerType::Dense));
    throw_if(!dense_2d, "Expected Dense layer.");

    dense_2d->set_activation_function("Softmax");

    Loss loss(&neural_network, &dataset);
    loss.set_error("MeanSquaredError");
    loss.set_regularization("None");

    QuasiNewtonMethod quasi_newton_method(&loss);
    quasi_newton_method.set_maximum_epochs(500);
    quasi_newton_method.set_display(false);

    try
    {
        quasi_newton_method.train();
    }
    catch (const exception&)
    {
        correlation.r = 0.0f;
        return correlation;
    }

    const MatrixR inputs = dataset.get_feature_data("Input");

    const MatrixR targets = dataset.get_feature_data("Target");

    const MatrixR outputs = neural_network.calculate_outputs(inputs);

    correlation.r = linear_correlation(outputs.reshaped(), targets.reshaped()).r;

    const float z_correlation = r_correlation_to_z_correlation(correlation.r);

    const auto [ci_lower, ci_upper] = confidence_interval_z_correlation(z_correlation, inputs.rows());

    correlation.lower_confidence = z_correlation_to_r_correlation(ci_lower);
    correlation.upper_confidence = z_correlation_to_r_correlation(ci_upper);

    correlation.form = Correlation::Form::Sigmoid;

    return correlation;
}

Correlation point_biserial_correlation(const VectorR& continuous,
                                       const VectorR& binary)
{
    Correlation result;
    result.form   = Correlation::Form::Identity;
    result.method = Correlation::Method::Pearson;

    const auto [x_filter, y_filter] = filter_missing_values(continuous, binary);

    if (x_filter.size() < 2 || is_constant(x_filter) || is_constant(y_filter))
    {
        result.r = 0.0f;
        return result;
    }

    const Index sample_count = x_filter.size();

    const auto x_double = x_filter.cast<double>();
    const double sum_all = x_double.sum();
    const double sum_sq = x_double.squaredNorm();

    const auto positive_mask = (y_filter.array() > 0.5f);
    const Index positive_count = positive_mask.count();
    const Index negative_count = sample_count - positive_count;
    const double positive_sum = positive_mask.select(x_double.array(), 0.0).sum();
    const double negative_sum = sum_all - positive_sum;

    if (positive_count == 0 || negative_count == 0)
    {
        result.r = 0.0f;
        return result;
    }

    const double mean_all = sum_all / double(sample_count);
    const double variance  = (sum_sq / double(sample_count)) - (mean_all * mean_all);

    if (variance <= 0)
    {
        result.r = 0.0f;
        return result;
    }

    const double s_x  = sqrt(variance);
    const double group_one_mean   = positive_sum / double(positive_count);
    const double group_zero_mean   = negative_sum / double(negative_count);
    const double point_biserial_r = (group_one_mean - group_zero_mean) / s_x
                        * sqrt(double(positive_count) * double(negative_count) / (double(sample_count) * double(sample_count)));

    result.r = float(clamp(point_biserial_r, -1.0, 1.0));

    const float z = r_correlation_to_z_correlation(result.r);
    const auto [ci_lower, ci_upper] = confidence_interval_z_correlation(z, sample_count);
    result.lower_confidence = z_correlation_to_r_correlation(ci_lower);
    result.upper_confidence = z_correlation_to_r_correlation(ci_upper);

    return result;
}

Correlation eta_squared_correlation(const VectorR& continuous,
                                    const MatrixR& categorical)
{
    Correlation result;
    result.form   = Correlation::Form::Identity;
    result.method = Correlation::Method::Pearson;

    const auto [x_filter, y_filter] = filter_missing_values(continuous, categorical);

    if (x_filter.size() < 2 || is_constant(x_filter))
    {
        result.r = 0.0f;
        return result;
    }

    const Index sample_count = x_filter.size();
    const Index categories_number     = y_filter.cols();

    const auto x_double = x_filter.cast<double>();
    const double grand_mean = x_double.mean();
    const double ss_total = (x_double.array() - grand_mean).square().sum();

    if (ss_total <= 0)
    {
        result.r = 0.0f;
        return result;
    }

    double ss_between = 0;

    for (Index i = 0; i < categories_number; ++i)
    {
        const auto mask = (y_filter.col(i).array() > 0.5f);
        const double group_sum = mask.select(x_double.array(), 0.0).sum();
        const Index group_count = mask.count();

        if (group_count == 0) continue;

        const double group_mean = group_sum / double(group_count);
        const double diff       = group_mean - grand_mean;
        ss_between += double(group_count) * diff * diff;
    }

    const double eta_sq = ss_between / ss_total;

    result.r = float(clamp(sqrt(eta_sq), 0.0, 1.0));

    const float z = r_correlation_to_z_correlation(result.r);
    const auto [ci_lower, ci_upper] = confidence_interval_z_correlation(z, sample_count);
    result.lower_confidence = z_correlation_to_r_correlation(ci_lower);
    result.upper_confidence = z_correlation_to_r_correlation(ci_upper);

    return result;
}

Correlation power_correlation(const VectorR& x, const VectorR& y)
{
    Correlation power_correlation;

    for(Index i = 0; i < x.size(); ++i)
        if(!isnan(x(i)) && x(i) <= 0.0f)
        {
            power_correlation.r = NAN;
            return power_correlation;
        }

    for(Index i = 0; i < y.size(); ++i)
        if(!isnan(y(i)) && y(i) <= 0.0f)
        {
            power_correlation.r = NAN;
            return power_correlation;
        }

    const VectorR log_x = x.array().log().matrix();
    const VectorR log_y = y.array().log().matrix();

    power_correlation = linear_correlation(log_x, log_y);

    power_correlation.form = Correlation::Form::Power;
    power_correlation.a = exp(power_correlation.a);

    return power_correlation;
}

void Correlation::set_perfect()
{
    r = 1.0f;
    a = 0.0f;
    b = 1.0f;

    upper_confidence = 1.0f;
    lower_confidence = 1.0f;
    form = Correlation::Form::Identity;
}

static const char* form_to_string(Correlation::Form form)
{
    using enum Correlation::Form;
    switch (form)
    {
    case Identity:      return "linear";
    case Sigmoid:     return "logistic";
    case Logarithmic: return "logarithmic";
    case Exponential: return "exponential";
    case Power:       return "power";
    default:                             return "";
    }
}

void Correlation::print() const
{
    cout << "Correlation" << "\n"
         << "Type: " << form_to_string(form) << "\n"
         << "a: " << a << "\n"
         << "b: " << b << "\n"
         << "r: " << r << "\n"
         << "Lower confidence: " << lower_confidence << "\n"
         << "Upper confidence: " << upper_confidence << "\n";
}

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
