//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   C O R R E L A T I O N S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "tensor_utilities.h"
#include "correlations.h"
#include "dataset.h"
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

    for(Index i = 0; i < past_time_steps; ++i)
        autocorrelation(i) = linear_correlation(x.head(this_size - i), x.tail(this_size - i)).r;

    return autocorrelation;
}

Correlation correlation(const MatrixR& x, const MatrixR& y)
{
    if(is_constant(x) || is_constant(y))
        return Correlation();

    const Index x_columns = x.cols();
    const Index y_columns = y.cols();

    const bool x_binary = is_binary(x);
    const bool y_binary = is_binary(y);

    if(x_columns == 1 && y_columns == 1)
    {
        const auto x_vector = x.col(0);
        const auto y_vector = y.col(0);

        if(!x_binary && !y_binary)
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

        if(y_binary && !x_binary)
            return opennn::point_biserial_correlation(x_vector, y_vector);

        if(x_binary && !y_binary)
            return opennn::point_biserial_correlation(y_vector, x_vector);

        return opennn::linear_correlation(x_vector, y_vector);
    }

    if(x_columns != 1 && y_columns == 1)
        return eta_squared_correlation(y.col(0), x);

    if(x_columns == 1 && y_columns != 1)
        return eta_squared_correlation(x.col(0), y);

    if(x_columns != 1 && y_columns != 1)
        return logistic_correlation(x, y);

    throw runtime_error("Correlations Exception: Unknown case.");
}

Correlation correlation_spearman(const MatrixR& x, const MatrixR& y)
{
    const Index x_columns = x.cols();
    const Index y_columns = y.cols();

    const bool x_binary = is_binary(x);
    const bool y_binary = is_binary(y);

    if(x_columns == 1 && y_columns == 1)
    {
        const auto x_vector = x.col(0);
        const auto y_vector = y.col(0);

        if(!x_binary && !y_binary)
            return linear_correlation_spearman(x_vector, y_vector);

        if(y_binary && !x_binary)
            return logistic_correlation_spearman(x_vector, y_vector);

        if(x_binary && !y_binary)
            return logistic_correlation_spearman(y_vector, x_vector);

        return linear_correlation_spearman(x_vector, y_vector);
    }

    if(x_columns == 1 && y_columns != 1)
        return logistic_correlation(VectorR(x.col(0)), y);

    if(x_columns != 1 && y_columns == 1)
        return logistic_correlation(x, VectorR(y.col(0)));

    if(x_columns != 1 && y_columns != 1)
        return logistic_correlation(x, y);

    throw runtime_error("Correlations Exception: Unknown case.");
}

VectorR cross_correlations(const VectorR& x,
                           const VectorR& y,
                           Index maximum_past_time_steps)
{
    if(y.size() != x.size())
        throw runtime_error("Both vectors must have the same size.\n");

    VectorR cross_correlation(maximum_past_time_steps);

    const Index this_size = x.size();

    for(Index i = 0; i < maximum_past_time_steps; ++i)
        cross_correlation[i] = linear_correlation(x.head(this_size - i), y.segment(i, this_size - i)).r;

    return cross_correlation;
}

Correlation exponential_correlation(const VectorR& x, const VectorR& y)
{
    Correlation exponential_correlation;

    if((!y.array().isNaN() && y.array() <= 0.0f).any())
    {
        exponential_correlation.r = static_cast<type>(NAN);
        return exponential_correlation;
    }

    exponential_correlation = linear_correlation(x, y.array().log().matrix());

    exponential_correlation.form = Correlation::Form::Exponential;
    exponential_correlation.a = exp(exponential_correlation.a);
    return exponential_correlation;
}

MatrixR get_correlation_values(const Tensor<Correlation, 2>& correlations)
{
    const Index rows_number = correlations.dimension(0);
    const Index columns_number = correlations.dimension(1);

    MatrixR values(rows_number, columns_number);

    for(Index i = 0; i < rows_number; ++i)
        for(Index j = 0; j < columns_number; ++j)
            values(i, j) = correlations(i, j).r;

    return values;
}

Correlation linear_correlation(const VectorR& x,
                               const VectorR& y)
{
    if(x.size() != y.size())
        throw runtime_error("Y size must be equal to X size.\n");

    if(is_constant(x) || is_constant(y))
        return Correlation();

    const auto [x_filter, y_filter] = filter_missing_values(x, y);

    const Index n = x_filter.size();

    if(x_filter.size() == 0)
        return Correlation();

    const auto x_double = x_filter.cast<double>();
    const auto y_double = y_filter.cast<double>();

    const double s_x = x_double.sum();
    const double s_y = y_double.sum();
    const double s_xx = x_double.squaredNorm();
    const double s_yy = y_double.squaredNorm();
    const double s_xy = x_double.dot(y_double);

    const double denominator = sqrt((double(n) * s_xx - s_x * s_x) * (double(n) * s_yy - s_y * s_y));

    if (denominator < static_cast<double>(EPSILON))
        return Correlation();

    Correlation linear_correlation;
    linear_correlation.form = Correlation::Form::Linear;
    linear_correlation.a = static_cast<type>((s_y * s_xx - s_x * s_xy) / (double(n) * s_xx - s_x * s_x));
    linear_correlation.b = static_cast<type>((double(n) * s_xy - s_x * s_y) / (double(n) * s_xx - s_x * s_x));
    linear_correlation.r = static_cast<type>((double(n) * s_xy - s_x * s_y) / denominator);

    const type z_correlation = r_correlation_to_z_correlation(linear_correlation.r);

    const auto [ci_lower, ci_upper] = confidence_interval_z_correlation(z_correlation, n);

    linear_correlation.lower_confidence = clamp(z_correlation_to_r_correlation(ci_lower), type(-1), type(1));
    linear_correlation.upper_confidence = clamp(z_correlation_to_r_correlation(ci_upper), type(-1), type(1));
    linear_correlation.r = clamp(linear_correlation.r, type(-1), type(1));

    return linear_correlation;
}

type r_correlation_to_z_correlation(const type r_correlation)
{
    const type r_clamped = clamp(r_correlation, type(-0.9999), type(0.9999));

    return type(0.5 * log((1 + r_clamped) / (1 - r_clamped)));
}

type z_correlation_to_r_correlation (const type z_correlation)
{
    return tanh(z_correlation);
}

pair<type, type> confidence_interval_z_correlation(const type z_correlation, Index n)
{
    const type margin = type(1.959964) * type(1/sqrt(n - 3));

    return { z_correlation - margin, z_correlation + margin };
}

VectorR calculate_spearman_ranks(const VectorR& x)
{
    const Index n = x.size();

    if (n == 0) return {};

    VectorI sorted_indices(n);

    iota(sorted_indices.data(), sorted_indices.data() + n, 0);

    sort(sorted_indices.data(), sorted_indices.data() + n,
         [&](Index i, Index j) { return x(i) < x(j); });

    VectorR ranks(n);

    Index i = 0;
    while (i < n)
    {
        Index j = i;
        while (j + 1 < n && x(sorted_indices(j + 1)) == x(sorted_indices(i)))
            ++j;

        const type average_rank = (static_cast<type>(i + 1) + static_cast<type>(j + 1)) / type(2.0);

        for(Index k = i; k <= j; ++k)
            ranks(sorted_indices(k)) = average_rank;

        i = j + 1;
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

    if((!x.array().isNaN() && x.array() <= type(0)).any())
    {
        logarithmic_correlation.r = type(NAN);
        return logarithmic_correlation;
    }

    logarithmic_correlation = linear_correlation(x.array().log(), y);

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

    Dataset dataset(input.size(), {1}, {1});
    dataset.set_data(data);
    dataset.set_sample_roles("Training");
    dataset.set_variable_scalers(scaler);
    dataset.set_shape("Input", {1});
    dataset.set_shape("Target", {1});
    dataset.set_display(false);

    NeuralNetwork neural_network;
    const Shape dim = { 1 };
    neural_network.add_layer(make_unique<Scaling<2>>(dim));
    neural_network.add_layer(make_unique<Dense<2>>(dim, dim, "Sigmoid"));
    neural_network.compile();

    Loss loss(&neural_network, &dataset);
    loss.set_error("MeanSquaredError");
    loss.set_regularization("None");

    LevenbergMarquardtAlgorithm lm(&loss);
    lm.set_display(false);

    try
    {
        lm.train();
    }
    catch(const exception&)
    {
        correlation.r = type(0);
        return correlation;
    }

    const MatrixR inputs = dataset.get_feature_data("Input");
    const MatrixR targets = dataset.get_feature_data("Target");
    const MatrixR outputs = neural_network.calculate_outputs(inputs);

    correlation.r = linear_correlation(outputs.reshaped(), targets.reshaped()).r;

    if(!isfinite(correlation.r))
    {
        correlation.r = type(0);
        return correlation;
    }

    const type z_correlation = r_correlation_to_z_correlation(correlation.r);
    const auto [ci_lower, ci_upper] = confidence_interval_z_correlation(z_correlation, inputs.rows());

    correlation.lower_confidence = z_correlation_to_r_correlation(ci_lower);
    correlation.upper_confidence = z_correlation_to_r_correlation(ci_upper);

    const VectorR coefficients = Map<const VectorR, AlignedMax>(
        neural_network.get_parameters_data(), neural_network.get_parameters_size());
    correlation.a = coefficients(0);
    correlation.b = coefficients(1);

    if(correlation.b < type(0))
    {
        correlation.r *= type(-1);
        const type old_lower = correlation.lower_confidence;
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
        Correlation c;
        c.r = type(NAN);
        c.form = Correlation::Form::Sigmoid;
        return c;
    }

    return fit_logistic_correlation(x_filter, y_filter, "MeanStandardDeviation");
}

Correlation logistic_correlation_spearman(const VectorR& x, const VectorR& y)
{
    const auto [x_filter, y_filter] = filter_missing_values(x, y);

    if(x_filter.size() < 2)
    {
        Correlation c;
        c.r = type(NAN);
        c.form = Correlation::Form::Sigmoid;
        return c;
    }

    return fit_logistic_correlation(calculate_spearman_ranks(x_filter), y_filter, "MinimumMaximum");
}

Correlation logistic_correlation(const VectorR& x, const MatrixR& y)
{
    Correlation correlation;
    correlation.form = Correlation::Form::Sigmoid;

    const auto [x_filter, y_filter] = filter_missing_values(x, y);

    if(y_filter.cols() > 50)
    {
        cout << "Warning: Y variable has too many categories." << "\n";

        correlation.r = type(NAN);

        return correlation;
    }

    if(x_filter.size() == 0)
    {
        correlation.r = type(NAN);

        return correlation;
    }

    MatrixR data(x_filter.rows(), 1 + y_filter.cols());
    data << x_filter, y_filter;

    vector<Index> input_columns_indices(1);
    input_columns_indices[0] = type(0);

    vector<Index> target_columns_indices(y_filter.cols());
    iota(target_columns_indices.begin(), target_columns_indices.end(), 1);

    Dataset dataset(x_filter.size(), {1}, {y_filter.cols()});

    dataset.set_data(data);
    dataset.set_variable_indices(input_columns_indices, target_columns_indices);
    dataset.set_binary_variables();
    dataset.set_default_variable_scalers();

    dataset.set_sample_roles("Training");
    dataset.set_shape("Input", {dataset.get_features_number("Input")});
    dataset.set_shape("Target", {dataset.get_features_number("Target")});

    const Index input_features_number = dataset.get_features_number("Input");
    const Index target_features_number = dataset.get_features_number("Target");

    ClassificationNetwork neural_network({ input_features_number }, {1}, {target_features_number});

    auto* dense_2d = dynamic_cast<Dense<2>*>(neural_network.get_first("Dense2d"));
    if(!dense_2d) throw runtime_error("Expected Dense<2> layer.");

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
    catch(const exception&)
    {
        correlation.r = type(0);
        return correlation;
    }

    const MatrixR inputs = dataset.get_feature_data("Input");
    const MatrixR targets = dataset.get_feature_data("Target");
    const MatrixR outputs = neural_network.calculate_outputs(inputs);

    correlation.r = linear_correlation(outputs.reshaped(), targets.reshaped()).r;

    const type z_correlation = r_correlation_to_z_correlation(correlation.r);

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

    if(x_filter.rows() == y_filter.rows() && x_filter.cols() == y_filter.cols())
        if((x_filter.array() == y_filter.array()).all())
        {
            correlation.r = static_cast<type>(1);
            return correlation;
        }

    if(x.cols() > 50 || y.cols() > 50)
    {
        cout << "Warning: One variable has too many categories." << "\n";

        correlation.r = type(NAN);

        return correlation;
    }

    if(x_filter.size() == 0 && y_filter.size() == 0)
    {
        correlation.r = type(NAN);

        return correlation;
    }

    MatrixR data(x_filter.rows(), x_filter.cols() + y_filter.cols());
    data << x_filter, y_filter;

    vector<Index> input_columns_indices(x_filter.cols());

    iota(input_columns_indices.begin(), input_columns_indices.end(), 0);

    vector<Index> target_columns_indices(y_filter.cols());
    iota(target_columns_indices.begin(), target_columns_indices.end(), x_filter.cols());

    Dataset dataset(x_filter.rows(), { x_filter.cols() }, { y_filter.cols() });

    dataset.set_data(data);

    dataset.set_variable_indices(input_columns_indices, target_columns_indices);

    dataset.set_sample_roles("Training");

    const Index input_features_number = dataset.get_features_number("Input");
    const Index target_features_number = dataset.get_features_number("Target");

    ClassificationNetwork neural_network({input_features_number }, {}, {target_features_number});

    auto* dense_2d = dynamic_cast<Dense<2>*>(neural_network.get_first("Dense2d"));
    if(!dense_2d) throw runtime_error("Expected Dense<2> layer.");

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
    catch(const exception&)
    {
        correlation.r = type(0);
        return correlation;
    }

    const MatrixR inputs = dataset.get_feature_data("Input");

    const MatrixR targets = dataset.get_feature_data("Target");

    const MatrixR outputs = neural_network.calculate_outputs(inputs);

    correlation.r = linear_correlation(outputs.reshaped(), targets.reshaped()).r;

    const type z_correlation = r_correlation_to_z_correlation(correlation.r);

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
    result.form   = Correlation::Form::Linear;
    result.method = Correlation::Method::Pearson;

    const auto [x_filter, y_filter] = filter_missing_values(continuous, binary);

    if(x_filter.size() < 2 || is_constant(x_filter) || is_constant(y_filter))
    {
        result.r = type(0);
        return result;
    }

    const Index n = x_filter.size();

    const auto x_dbl = x_filter.cast<double>();
    const double sum_all = x_dbl.sum();
    const double sum_sq = x_dbl.squaredNorm();

    const auto mask1 = (y_filter.array() > type(0.5));
    const Index n1 = mask1.count();
    const Index n0 = n - n1;
    const double sum1 = mask1.select(x_dbl.array(), 0.0).sum();
    const double sum0 = sum_all - sum1;

    if(n1 == 0 || n0 == 0)
    {
        result.r = type(0);
        return result;
    }

    const double mean_all = sum_all / double(n);
    const double variance  = (sum_sq / double(n)) - (mean_all * mean_all);

    if(variance <= 0)
    {
        result.r = type(0);
        return result;
    }

    const double s_x  = sqrt(variance);
    const double M1   = sum1 / double(n1);
    const double M0   = sum0 / double(n0);
    const double r_pb = (M1 - M0) / s_x
                        * sqrt(double(n1) * double(n0) / (double(n) * double(n)));

    result.r = type(clamp(r_pb, -1.0, 1.0));

    const type z = r_correlation_to_z_correlation(result.r);
    const auto [ci_lower, ci_upper] = confidence_interval_z_correlation(z, n);
    result.lower_confidence = z_correlation_to_r_correlation(ci_lower);
    result.upper_confidence = z_correlation_to_r_correlation(ci_upper);

    return result;
}

Correlation eta_squared_correlation(const VectorR& continuous,
                                    const MatrixR& categorical)
{
    Correlation result;
    result.form   = Correlation::Form::Linear;
    result.method = Correlation::Method::Pearson;

    const auto [x_filter, y_filter] = filter_missing_values(continuous, categorical);

    if(x_filter.size() < 2 || is_constant(x_filter))
    {
        result.r = type(0);
        return result;
    }

    const Index n          = x_filter.size();
    const Index n_cats     = y_filter.cols();

    const double grand_mean = x_filter.cast<double>().mean();

    const double ss_total = (x_filter.cast<double>().array() - grand_mean).square().sum();

    if(ss_total <= 0)
    {
        result.r = type(0);
        return result;
    }

    double ss_between = 0;

    for(Index cat = 0; cat < n_cats; ++cat)
    {
        const auto mask = (y_filter.col(cat).array() > type(0.5));
        const double group_sum = mask.select(x_filter.cast<double>().array(), 0.0).sum();
        const Index group_count = mask.count();

        if(group_count == 0) continue;

        const double group_mean = group_sum / double(group_count);
        const double diff       = group_mean - grand_mean;
        ss_between += double(group_count) * diff * diff;
    }

    const double eta_sq = ss_between / ss_total;

    result.r = type(clamp(sqrt(eta_sq), 0.0, 1.0));

    const type z = r_correlation_to_z_correlation(result.r);
    const auto [ci_lower, ci_upper] = confidence_interval_z_correlation(z, n);
    result.lower_confidence = z_correlation_to_r_correlation(ci_lower);
    result.upper_confidence = z_correlation_to_r_correlation(ci_upper);

    return result;
}

Correlation power_correlation(const VectorR& x, const VectorR& y)
{
    Correlation power_correlation;

    if ((x.array() <= 0.0f).any() || (y.array() <= 0.0f).any())
    {
        power_correlation.r = static_cast<type>(NAN);
        return power_correlation;
    }

    power_correlation = linear_correlation(x.array().log().matrix(), y.array().log().matrix());

    power_correlation.form = Correlation::Form::Power;
    power_correlation.a = exp(power_correlation.a);

    return power_correlation;
}

void Correlation::set_perfect()
{
    r = type(1);
    a = type(0);
    b = type(1);

    upper_confidence = type(1);
    lower_confidence = type(1);
    form = Correlation::Form::Linear;
}

static const char* form_to_string(Correlation::Form form)
{
    switch(form)
    {
    case Correlation::Form::Linear:      return "linear";
    case Correlation::Form::Sigmoid:     return "logistic";
    case Correlation::Form::Logarithmic: return "logarithmic";
    case Correlation::Form::Exponential: return "exponential";
    case Correlation::Form::Power:       return "power";
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
