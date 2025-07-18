//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   C O R R E L A T I O N S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "tensors.h"
#include "correlations.h"
#include "dataset.h"
#include "scaling_layer_2d.h"
#include "perceptron_layer.h"
#include "neural_network.h"
#include "standard_networks.h"
#include "mean_squared_error.h"
#include "cross_entropy_error.h"
#include "cross_entropy_error_3d.h"
#include "minkowski_error.h"
#include "normalized_squared_error.h"
#include "weighted_squared_error.h"
#include "stochastic_gradient_descent.h"
#include "adaptive_moment_estimation.h"
#include "quasi_newton_method.h"
#include "levenberg_marquardt_algorithm.h"

namespace opennn
{

Tensor<type, 1> autocorrelations(const ThreadPoolDevice* thread_pool_device,
                                 const Tensor<type, 1>& x,
                                 const Index& lags_number)
{
    Tensor<type, 1> autocorrelation(lags_number);

    const Index this_size = x.size();
    for(Index i = 0; i < lags_number; i++)
    {
        Tensor<type, 1> column_x(this_size-i);
        Tensor<type, 1> column_y(this_size-i);

        for(Index j = 0; j < this_size - i; j++)
        {
            column_x(j) = x(j);
            column_y(j) = x(j + i);
        }
        autocorrelation(i) = linear_correlation(thread_pool_device, column_x, column_y).r;
    }

    return autocorrelation;
}


Correlation correlation(const ThreadPoolDevice* thread_pool_device,
                        const Tensor<type, 2>& x,
                        const Tensor<type, 2>& y)
{
    if(is_constant(x) || is_constant(y))
        return Correlation();

    const Index x_rows = x.dimension(0);
    const Index x_columns = x.dimension(1);
    const Index y_columns = y.dimension(1);

    const bool x_binary = is_binary(x);
    const bool y_binary = is_binary(y);

    const array<Index, 1> vector{{x_rows}};

    if(x_columns == 1 && y_columns == 1)
    {
        const Tensor<type, 1> x_vector = x.reshape(vector);
        const Tensor<type, 1> y_vector = y.reshape(vector);

        if(!x_binary && !y_binary)
        {
            const Correlation linear_correlation
                    = opennn::linear_correlation(thread_pool_device, x_vector, y_vector);

            const Correlation exponential_correlation
                    = opennn::exponential_correlation(thread_pool_device, x_vector, y_vector);

            const Correlation logarithmic_correlation
                    = opennn::logarithmic_correlation(thread_pool_device, x_vector, y_vector);

            const Correlation power_correlation
                    = opennn::power_correlation(thread_pool_device, x_vector, y_vector);

            return max({linear_correlation, exponential_correlation, logarithmic_correlation, power_correlation},
                [](const Correlation& a, const Correlation& b) {
                    return abs(a.r) < abs(b.r);
                });
        }

        if(!x_binary && y_binary)
            return logistic_correlation_vector_vector(thread_pool_device, x_vector, y_vector);

        if(x_binary && !y_binary)
            return logistic_correlation_vector_vector(thread_pool_device, y_vector, x_vector);

        if(x_binary && y_binary)
            return opennn::linear_correlation(thread_pool_device, x_vector, y_vector);
    }

    if(x_columns != 1 && y_columns == 1)
        return logistic_correlation_matrix_vector(thread_pool_device, x, y.reshape(vector));

    if(x_columns == 1 && y_columns != 1)
        return logistic_correlation_vector_matrix(thread_pool_device, x.reshape(vector), y);

    if(x_columns != 1 && y_columns != 1)
        return logistic_correlation_matrix_matrix(thread_pool_device, x, y);

    throw runtime_error("Correlations Exception: Unknown case.");

//    return Correlation();
}


Correlation correlation_spearman(const ThreadPoolDevice* thread_pool_device,
                                 const Tensor<type, 2>& x,
                                 const Tensor<type, 2>& y)
{
    const Index x_rows = x.dimension(0);
    const Index x_columns = x.dimension(1);
    const Index y_columns = y.dimension(1);

    const bool x_binary = is_binary(x);
    const bool y_binary = is_binary(y);

    const array<Index, 1> vector{{x_rows}};

    if(x_columns == 1 && y_columns == 1)
    {
        const Tensor<type, 1> x_vector = x.reshape(vector);
        const Tensor<type, 1> y_vector = y.reshape(vector);

        if(!x_binary && !y_binary)
            return linear_correlation_spearman(thread_pool_device, x_vector, y_vector);
        else if(!x_binary && y_binary)
            return logistic_correlation_vector_vector_spearman(thread_pool_device, x_vector, y_vector);
        else if(x_binary && !y_binary)
            return logistic_correlation_vector_vector_spearman(thread_pool_device, y_vector, x_vector);
        else if(x_binary && y_binary)
            return linear_correlation_spearman(thread_pool_device, x_vector, y_vector);
    }

    if(x_columns == 1 && y_columns != 1)
        return logistic_correlation_vector_matrix(thread_pool_device, x.reshape(vector), y);

    if(x_columns != 1 && y_columns == 1)
        return logistic_correlation_matrix_vector(thread_pool_device, x, y.reshape(vector));

    if(x_columns != 1 && y_columns != 1)
        return logistic_correlation_matrix_matrix(thread_pool_device, x, y);

    throw runtime_error("Correlations Exception: Unknown case.");
}


Tensor<type, 1> cross_correlations(const ThreadPoolDevice* thread_pool_device,
                                   const Tensor<type, 1>& x,
                                   const Tensor<type, 1>& y,
                                   const Index& maximum_lags_number)
{
    if(y.size() != x.size())
        throw runtime_error("Both vectors must have the same size.\n");

    Tensor<type, 1> cross_correlation(maximum_lags_number);

    const Index this_size = x.size();

    for(Index i = 0; i < maximum_lags_number; i++)
    {
        Tensor<type, 1> column_x(this_size-i);
        Tensor<type, 1> column_y(this_size-i);

        for(Index j = 0; j < this_size - i; j++)
        {
            column_x(j) = x(j);
            column_y(j) = y(j + i);
        }

        cross_correlation[i] = linear_correlation(thread_pool_device, column_x, column_y).r;
    }

    return cross_correlation;
}


Correlation exponential_correlation(const ThreadPoolDevice* thread_pool_device,
                                    const Tensor<type, 1>& x,
                                    const Tensor<type, 1>& y)
{
    Correlation exponential_correlation;

    for(Index i = 0; i < y.dimension(0); i++)
    {
        if(!isnan(y(i)) && y(i) <= type(0))
        {
            exponential_correlation.r = type(NAN);

            return exponential_correlation;
        }
    }

    exponential_correlation = linear_correlation(thread_pool_device, x, y.log());

    exponential_correlation.form = Correlation::Form::Exponential;
    exponential_correlation.a = exp(exponential_correlation.a);
    exponential_correlation.b = exponential_correlation.b;

    return exponential_correlation;
}


pair<Tensor<type, 1>, Tensor<type, 1>> filter_missing_values_vector_vector(const Tensor<type, 1>& x,
                                                                           const Tensor<type, 1>& y)
{
    Index new_size = 0;

    for(Index i = 0; i < x.size(); i++)
        if(!isnan(x(i)) && !isnan(y(i))) 
            new_size++;

    if(new_size == x.size())
        return make_pair(x, y);

    Tensor<type, 1> new_x(new_size);
    Tensor<type, 1> new_y(new_size);

    Index index = 0;

    for(Index i = 0; i < x.size(); i++)
    {
        if(!isnan(x(i)) && !isnan(y(i)))
        {
            new_x(index) = x(i);
            new_y(index) = y(i);

            index++;
        }
    }

    return {new_x, new_y};
}


pair<Tensor<type, 1>, Tensor<type, 2>> filter_missing_values_vector_matrix(const Tensor<type, 1>& x,
                                                                           const Tensor<type, 2>& y)
{
    const Index rows_number = x.size();
    const Index y_columns_number = y.dimension(1);

    Index new_rows_number = 0;

    Tensor<bool, 1> not_NAN_row(rows_number);

    for(Index i = 0; i < rows_number; i++)
    {
        not_NAN_row(i) = true;

        if(isnan(x(i)) || isnan(y(i)))
            not_NAN_row(i) = false;

        if(not_NAN_row(i))
            new_rows_number++;
    }

    Tensor<type, 1> new_x(new_rows_number);
    Tensor<type, 2> new_y(new_rows_number, y_columns_number);

    Index index = 0;

    for(Index i = 0; i < rows_number; i++)
    {
        if(not_NAN_row(i))
        {
            for(Index j = 0; j < y_columns_number; j++)
                new_y(index, j) = y(i, j);

            new_x(index++) = x(i);
        }
    }

    return {new_x, new_y};
}


pair<Tensor<type, 2>, Tensor<type, 2>> filter_missing_values_matrix_matrix(const Tensor<type, 2>& x,
                                                                           const Tensor<type, 2>& y)
{
    const Index rows_number = x.dimension(0);
    const Index x_columns_number = x.dimension(1);
    const Index y_columns_number = y.dimension(1);

    Index new_rows_number = 0;

    Tensor<bool, 1> not_NAN_row(rows_number);

    for(Index i = 0; i < rows_number; i++)
    {
        not_NAN_row(i) = true;

        if(isnan(y(i)))
        {
            not_NAN_row(i) = false;
        }
        else
        {
            for(Index j = 0; j < x_columns_number; j++)
            {
                if(isnan(x(i, j)))
                {
                    not_NAN_row(i) = false;
                    break;
                }
            }
        }

        if(not_NAN_row(i)) 
            new_rows_number++;
    }

    Tensor<type, 2> new_x(new_rows_number, x_columns_number);

    Tensor<type, 2> new_y(new_rows_number, y_columns_number);

    Index index = 0;

    for(Index i = 0; i < rows_number; i++)
    {
        if(not_NAN_row(i))
        {
            for(Index j = 0; j < y_columns_number; j++)
                new_y(index, j) = y(i, j);

            for(Index j = 0; j < x_columns_number; j++)
                new_x(index, j) = x(i, j);

            index++;
        }
    }

    return {new_x, new_y};
}


Tensor<type, 2> get_correlation_values(const Tensor<Correlation, 2>& correlations)
{
    const Index rows_number = correlations.dimension(0);
    const Index columns_number = correlations.dimension(1);
    Tensor<type, 2> values(rows_number, columns_number);

    for(Index i = 0; i < rows_number; i++)
        for(Index j = 0; j < columns_number; j++)
            values(i, j) = correlations(i, j).r;

    return values;
}


Correlation linear_correlation(const ThreadPoolDevice* thread_pool_device,
                               const Tensor<type, 1>& x,
                               const Tensor<type, 1>& y)
{
    if(x.size() != y.size())
        throw runtime_error("Y size must be equal to X size.\n");

    if(is_constant(x) || is_constant(y))
        return Correlation();

    const pair<Tensor<type, 1>, Tensor<type, 1>> filter_vectors = filter_missing_values_vector_vector(x,y);

    const Tensor<double, 1> x_filter = filter_vectors.first.cast<double>();
    const Tensor<double, 1> y_filter = filter_vectors.second.cast<double>();

    const Index n = x_filter.size();

    if(x_filter.size() == 0)
        return Correlation();

    Tensor<double, 0> s_x;
    s_x.device(*thread_pool_device) = x_filter.sum();

    Tensor<double, 0> s_y;
    s_y.device(*thread_pool_device) = y_filter.sum();

    Tensor<double, 0> s_xx;
    s_xx.device(*thread_pool_device) = x_filter.square().sum();

    Tensor<double, 0> s_yy;
    s_yy.device(*thread_pool_device) = y_filter.square().sum();

    Tensor<double, 0> s_xy;
    s_xy.device(*thread_pool_device) = (y_filter*x_filter).sum();

    const double denominator = sqrt((double(n) * s_xx() - s_x() * s_x()) * (double(n) * s_yy() - s_y() * s_y()));

    if (denominator < NUMERIC_LIMITS_MIN)
        return Correlation();

    Correlation linear_correlation;
    linear_correlation.form = Correlation::Form::Linear;
    linear_correlation.a = type(s_y() * s_xx() - s_x() * s_xy()) / type(double(n) * s_xx() - s_x() * s_x());
    linear_correlation.b = type(double(n) * s_xy() - s_x() * s_y()) / type(double(n) * s_xx() - s_x() * s_x());
    linear_correlation.r = type(double(n) * s_xy() - s_x() * s_y()) / type(denominator);

    const type z_correlation = r_correlation_to_z_correlation(linear_correlation.r);

    const Tensor<type, 1> confidence_interval_z = confidence_interval_z_correlation(z_correlation, n);

    linear_correlation.lower_confidence = bound(z_correlation_to_r_correlation(confidence_interval_z(0)), type(-1), type(1));
    linear_correlation.upper_confidence = bound(z_correlation_to_r_correlation(confidence_interval_z(1)), type(-1), type(1));
    linear_correlation.r = bound(linear_correlation.r, type(-1), type(1));

    return linear_correlation;
}


type r_correlation_to_z_correlation(const type& r_correlation)
{
    return type(0.5 * log((1 + r_correlation) / (1 - r_correlation)));
}


type z_correlation_to_r_correlation (const type& z_correlation)
{
    return type((exp(2 * z_correlation) - 1) / (exp(2 * z_correlation) + 1));
}


Tensor<type, 1> confidence_interval_z_correlation(const type& z_correlation, const Index& n)
{
    Tensor<type, 1> confidence_interval(2);

    const type z_standard_error = type(1.959964);

    confidence_interval(0) = z_correlation - z_standard_error * type(1/sqrt(n - 3));
    confidence_interval(1) = z_correlation + z_standard_error * type(1/sqrt(n - 3));

    return confidence_interval;
}


Tensor<type, 1> calculate_spearman_ranks(const Tensor<type, 1> & x)
{
    const Index n = x.size();

    if (n == 0)
    {
        return Tensor<type, 1>();
    }

    Tensor<Index, 1> sorted_indices(n);

    iota(sorted_indices.data(), sorted_indices.data() + n, 0);

    sort(sorted_indices.data(), sorted_indices.data() + n,
         [&](Index i, Index j) { return x(i) < x(j); });

    Tensor<type, 1> ranks(n);

    Index i = 0;
    while (i < n)
    {
        Index j = i;
        while (j + 1 < n && x(sorted_indices(j + 1)) == x(sorted_indices(i)))
            j++;

        const type average_rank = (static_cast<type>(i + 1) + static_cast<type>(j + 1)) / type(2.0);

        for (Index k = i; k <= j; k++)
            ranks(sorted_indices(k)) = average_rank;

        i = j + 1;
    }

    return ranks;
}


Correlation linear_correlation_spearman(const ThreadPoolDevice* thread_pool_device, const Tensor<type, 1>& x, const Tensor<type, 1>& y)
{
    
    const pair<Tensor<type, 1>, Tensor<type, 1>> filter_vectors = filter_missing_values_vector_vector(x, y);

    const Tensor<type, 1> x_filter = filter_vectors.first.cast<type>();
    const Tensor<type, 1> y_filter = filter_vectors.second.cast<type>();

    const Tensor<type, 1> x_rank = calculate_spearman_ranks(x_filter);
    const Tensor<type, 1> y_rank = calculate_spearman_ranks(y_filter);

    Correlation result = linear_correlation(thread_pool_device, x_rank, y_rank);

    return result;
}


Correlation logarithmic_correlation(const ThreadPoolDevice* thread_pool_device,
                                    const Tensor<type, 1>& x,
                                    const Tensor<type, 1>& y)
{
    Correlation logarithmic_correlation;

    for(Index i = 0; i < x.dimension(0); i++)
    {
        if(!isnan(x(i)) && x(i) <= type(0))
        {
            logarithmic_correlation.r = type(NAN);

            return logarithmic_correlation;
        }
    }

    logarithmic_correlation = linear_correlation(thread_pool_device, x.log(), y);

    logarithmic_correlation.form = Correlation::Form::Logarithmic;

    return logarithmic_correlation;
}




Correlation logistic_correlation_vector_vector(const ThreadPoolDevice* thread_pool_device,
                                               const Tensor<type, 1>& x,
                                               const Tensor<type, 1>& y)
{
    Correlation correlation;

    const pair<Tensor<type, 1>, Tensor<type, 1>> filtered_elements = filter_missing_values_vector_vector(x,y);

    const Tensor<type, 1> x_filtered = filtered_elements.first;
    const Tensor<type, 1> y_filtered = filtered_elements.second;

    if (x_filtered.size() == 0
    || is_constant(x_filtered)
    || is_constant(y_filtered))
    {
        correlation.r = type(NAN);
        correlation.form = Correlation::Form::Logistic;
        return correlation;
    }
    const Tensor<type, 2> data = assemble_vector_vector(x_filtered, y_filtered);

    Dataset dataset(x_filtered.size(), {1}, {1});
    dataset.set_data(data);
    dataset.set("Training");
    dataset.set_raw_variable_scalers(Scaler::MinimumMaximum);

    NeuralNetwork neural_network;
    dimensions dim1 = { 1 };
    dimensions dim2 = { 1 };
    neural_network.add_layer(make_unique<Scaling2d>(dim1));
    neural_network.add_layer(make_unique<Dense2d>(dim1, dim2, "Logistic"));

    neural_network.set_parameters_random();

    MeanSquaredError mean_squared_error(&neural_network, &dataset);
    mean_squared_error.set_regularization_method("NoRegularization");

    LevenbergMarquardtAlgorithm levenberg_marquardt_algorithm(&mean_squared_error);
    levenberg_marquardt_algorithm.set_display(false);
    levenberg_marquardt_algorithm.perform_training();

    const Tensor<type, 2> inputs = dataset.get_data_variables("Input");
    const Tensor<type, 2> targets = dataset.get_data_variables("Target");
    const Tensor<type, 2> outputs = neural_network.calculate_outputs<2,2>(inputs);

    // Logistic correlation
    const array<Index, 1> vector{{x_filtered.size()}};

    correlation.r = linear_correlation(thread_pool_device, outputs.reshape(vector), targets.reshape(vector)).r;

    const type z_correlation = r_correlation_to_z_correlation(correlation.r);

    const Tensor<type, 1> confidence_interval_z = confidence_interval_z_correlation(z_correlation, inputs.dimensions()[0]);

    correlation.lower_confidence = z_correlation_to_r_correlation(confidence_interval_z(0));

    correlation.upper_confidence = z_correlation_to_r_correlation(confidence_interval_z(1));

    correlation.form = Correlation::Form::Logistic;

    Tensor<type, 1> coefficients;
    neural_network.get_parameters(coefficients);

    correlation.a = coefficients(1);
    correlation.b = coefficients(0);

    if(correlation.b < type(0))
        correlation.r *= type(-1);

    return correlation;
}


Correlation logistic_correlation_vector_vector_spearman(const ThreadPoolDevice* thread_pool_device,
                                                        const Tensor<type, 1>& x,
                                                        const Tensor<type, 1>& y)
{
    Correlation correlation;

    const pair<Tensor<type, 1>, Tensor<type, 1>> filtered_elements = filter_missing_values_vector_vector(x,y);

    const Tensor<type, 1> x_filtered = filtered_elements.first;
    const Tensor<type, 1> y_filtered = filtered_elements.second;

    if(x_filtered.size() == 0)
    {
        correlation.r = type(NAN);

        correlation.form = Correlation::Form::Logistic;

        return correlation;
    }

    const Tensor<type, 1> x_rank = calculate_spearman_ranks(x_filtered);

    const Tensor<type, 2> data = assemble_vector_vector(x_rank, y_filtered);

    Dataset dataset(x_filtered.size(), {1}, {1});
    dataset.set_data(data);
    dataset.set_sample_uses("Training");
    dataset.set_raw_variable_scalers(Scaler::MinimumMaximum);

    NeuralNetwork neural_network;
    dimensions dim1 = { 1 };
    dimensions dim2 = { 1 };
    neural_network.add_layer(make_unique<Scaling2d>(dim1));
    neural_network.add_layer(make_unique<Dense2d>(dim1, dim2, "Logistic"));

    MeanSquaredError mean_squared_error(&neural_network, &dataset);
    mean_squared_error.set_regularization_method("NoRegularization");

    LevenbergMarquardtAlgorithm levenberg_marquardt_algorithm(&mean_squared_error);
    levenberg_marquardt_algorithm.set_display(false);
    levenberg_marquardt_algorithm.perform_training();

    const Tensor<type, 2> inputs = dataset.get_data_variables("Input");
    const Tensor<type, 2> targets = dataset.get_data_variables("Target");
    const Tensor<type, 2> outputs = neural_network.calculate_outputs<2,2>(inputs);

    // Logistic correlation

    const array<Index, 1> vector{{x_filtered.size()}};

    correlation.r = linear_correlation(thread_pool_device, outputs.reshape(vector), targets.reshape(vector)).r;

    const type z_correlation = r_correlation_to_z_correlation(correlation.r);

    const Tensor<type, 1> confidence_interval_z = confidence_interval_z_correlation(z_correlation, x_rank.size());

    correlation.lower_confidence = z_correlation_to_r_correlation(confidence_interval_z(0));

    correlation.upper_confidence = z_correlation_to_r_correlation(confidence_interval_z(1));

    correlation.form = Correlation::Form::Logistic;

    Tensor<type, 1> coefficients;
    neural_network.get_parameters(coefficients);

    correlation.a = coefficients(1);
    correlation.b = coefficients(0);

    if(correlation.b < type(0)) correlation.r *= type(-1);

    return correlation;
}


Correlation logistic_correlation_vector_matrix(const ThreadPoolDevice* thread_pool_device,
                                               const Tensor<type, 1>& x,
                                               const Tensor<type, 2>& y)
{
    Correlation correlation;
    correlation.form = Correlation::Form::Logistic;

    const pair<Tensor<type, 1>, Tensor<type,2>> filtered_elements = opennn::filter_missing_values_vector_matrix(x, y);

    const Tensor<type, 1> x_filtered = filtered_elements.first;
    const Tensor<type,2> y_filtered = filtered_elements.second;

    if(y_filtered.dimension(1) > 50)
    {
        cout << "Warning: Y variable has too many categories." << endl;

        correlation.r = type(NAN);

        return correlation;
    }

    if(x_filtered.size() == 0)
    {
        correlation.r = type(NAN);

        return correlation;
    }

    const Tensor<type, 2> data = opennn::assemble_vector_matrix(x_filtered, y_filtered);

    vector<Index> input_columns_indices(1);
    input_columns_indices[0] = type(0);

    vector<Index> target_columns_indices(y_filtered.dimension(1));

    for(Index i = 0; i < y_filtered.dimension(1); i++)
        target_columns_indices[i] = i + 1;
 
    Dataset dataset(x_filtered.size(), {1}, {y_filtered.dimension(1)});

    dataset.set_data(data);
    dataset.set_raw_variable_indices(input_columns_indices, target_columns_indices);
    dataset.set_binary_raw_variables();
    dataset.set_default_raw_variables_scalers();

    // Dataset.print();

    dataset.set_sample_uses("Training");
    dataset.set_dimensions("Input", {dataset.get_variables_number("Input")});
    dataset.set_dimensions("Target", {dataset.get_variables_number("Target")});

    const Index input_variables_number = dataset.get_variables_number("Input");
    const Index target_variables_number = dataset.get_variables_number("Target");

    ClassificationNetwork neural_network({ input_variables_number }, {1}, {target_variables_number});

    Scaling2d* scaling_layer_2d = static_cast<Scaling2d*>(neural_network.get_first("Scaling2d"));

    Dense2d* dense_2d = static_cast<Dense2d*>(neural_network.get_first("Dense2d"));

    dense_2d->set_activation_function("Softmax");
    scaling_layer_2d->set_display(false);

    CrossEntropyError2d cross_entropy_error_2d(&neural_network, &dataset);
    cross_entropy_error_2d.set_regularization_method("NoRegularization");

    QuasiNewtonMethod quasi_newton_method(&cross_entropy_error_2d);
    quasi_newton_method.set_display(false);
    quasi_newton_method.set_display_period(1000);
    quasi_newton_method.perform_training();

    // Logistic correlation

    const Tensor<type, 2> inputs = dataset.get_data_variables("Input");
    const Tensor<type, 2> targets = dataset.get_data_variables("Target");

    const Tensor<type, 2> outputs = neural_network.calculate_outputs<2,2>(inputs);

    const array<Index, 1> vector{{targets.size()}};

    correlation.r = linear_correlation(thread_pool_device, outputs.reshape(vector), targets.reshape(vector)).r;

    const type z_correlation = r_correlation_to_z_correlation(correlation.r);

    const Tensor<type, 1> confidence_interval_z = confidence_interval_z_correlation(z_correlation, x_filtered.size());

    correlation.lower_confidence = z_correlation_to_r_correlation(confidence_interval_z(0));

    correlation.upper_confidence = z_correlation_to_r_correlation(confidence_interval_z(1));

    return correlation;
}


Correlation logistic_correlation_matrix_vector(const ThreadPoolDevice* thread_pool_device,
                                               const Tensor<type, 2>& y,
                                               const Tensor<type, 1>& x)
{
    return logistic_correlation_vector_matrix(thread_pool_device, x,y);
}


Correlation logistic_correlation_matrix_matrix(const ThreadPoolDevice* thread_pool_device,
                                               const Tensor<type, 2>& x,
                                               const Tensor<type, 2>& y)
{
    Correlation correlation;
    correlation.form = Correlation::Form::Logistic;

    // Scrub missing values

    const pair<Tensor<type,2>, Tensor<type,2>> filtered_matrixes = filter_missing_values_matrix_matrix(x,y);

    const Tensor<type,2> x_filtered = filtered_matrixes.first;
    const Tensor<type,2> y_filtered = filtered_matrixes.second;

    if(x.dimension(0)  == y.dimension(0) && x.dimension(1)  == y.dimension(1))
    {
        const Tensor<bool, 0> are_equal = (x_filtered == y_filtered).all();

        if(are_equal(0))
        {
            correlation.r = type(1);

            return correlation;
        }
    }

    if(x.dimension(1) > 50 || y.dimension(1) > 50)
    {
        cout << "Warning: One variable has too many categories." << endl;

        correlation.r = type(NAN);

        return correlation;
    }

    if(x_filtered.size() == 0 && y_filtered.size() == 0)
    {
        correlation.r = type(NAN);

        return correlation;
    }

    const Tensor<type, 2> data = opennn::assemble_matrix_matrix(x_filtered, y_filtered);

    vector<Index> input_columns_indices(x_filtered.dimension(1));

    iota(input_columns_indices.begin(), input_columns_indices.end(), 0);

    vector<Index> target_columns_indices(y_filtered.dimension(1));

    for(Index i = 0; i < y_filtered.dimension(1); i++)
        target_columns_indices[i] = x_filtered.dimension(1)+i;

    Dataset Dataset(x_filtered.dimension(0), { x_filtered.dimension(1) }, { y_filtered.dimension(1) });

    Dataset.set_data(data);

    Dataset.set_raw_variable_indices(input_columns_indices, target_columns_indices);

    Dataset.set_sample_uses("Training");

    const Index input_variables_number = Dataset.get_variables_number("Input");
    const Index target_variables_number = Dataset.get_variables_number("Target");

    ClassificationNetwork neural_network({input_variables_number }, {}, {target_variables_number});

    Scaling2d* scaling_layer_2d = static_cast<Scaling2d*>(neural_network.get_first("Scaling2d"));

    Dense2d* dense_2d = static_cast<Dense2d*>(neural_network.get_first("Dense2d"));

    dense_2d->set_activation_function("Softmax");

    scaling_layer_2d->set_display(false);

    MeanSquaredError mean_squared_error(&neural_network, &Dataset);
    mean_squared_error.set_regularization_method("NoRegularization");

    QuasiNewtonMethod quasi_newton_method(&mean_squared_error);
    quasi_newton_method.set_maximum_epochs_number(500);
    quasi_newton_method.set_display(false);
    quasi_newton_method.perform_training();

    // Logistic correlation

    const Tensor<type, 2> inputs = Dataset.get_data_variables("Input");

    const Tensor<type, 2> targets = Dataset.get_data_variables("Target");

    const Tensor<type, 2> outputs = neural_network.calculate_outputs<2,2>(inputs);

    const array<Index, 1> vector{{targets.size()}};

    correlation.r = linear_correlation(thread_pool_device, outputs.reshape(vector), targets.reshape(vector)).r;

    const type z_correlation = r_correlation_to_z_correlation(correlation.r);

    const Tensor<type, 1> confidence_interval_z = confidence_interval_z_correlation(z_correlation, inputs.dimension(0));

    correlation.lower_confidence = z_correlation_to_r_correlation(confidence_interval_z(0));

    correlation.upper_confidence = z_correlation_to_r_correlation(confidence_interval_z(1));

    correlation.form = Correlation::Form::Logistic;

    return correlation;
}


Correlation power_correlation(const ThreadPoolDevice* thread_pool_device,
                              const Tensor<type, 1>& x,
                              const Tensor<type, 1>& y)
{
    Correlation power_correlation;

    for(Index i = 0; i < x.dimension(0); i++)
    {
        if((!isnan(x(i)) && x(i) <= type(0))
        || (!isnan(y(i)) && y(i) <= type(0)))
        {
            power_correlation.r = type(NAN);

            return power_correlation;
        }
    }

    power_correlation = linear_correlation(thread_pool_device, x.log(), y.log());

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


string Correlation::write_type() const
{
    switch(form)
    {
    case Form::Linear: return "linear";
    case Form::Logistic: return "logistic";
    case Form::Logarithmic: return "logarithmic";
    case Form::Exponential: return "exponential";
    case Form::Power: return "power";
    default:
        return string();
    }
}


void Correlation::print() const
{
    cout << "Correlation" << endl
         << "Type: " << write_type() << endl
         << "a: " << a << endl
         << "b: " << b << endl
         << "r: " << r << endl
         << "Lower confidence: " << lower_confidence << endl
         << "Upper confidence: " << upper_confidence << endl;
}


void register_layers()
{
<<<<<<< HEAD

    Bounding bounding_layer;//bounding_layer.print();
=======
    Bounding bounding_layer;bounding_layer.print();
    MultiHeadAttention multi_head_attention; multi_head_attention.print();
>>>>>>> 19a758faa721d9bff5d8e9f8e147ff8e4ba560fb
    Recurrent recurrent_layer; // recurrent_layer.print();
    MultiHeadAttention multihead_layer;
}


void register_loss_indices()
{
    CrossEntropyError2d cross_entropy_error_2d;//cross_entropy_error_2d.print();
    MeanSquaredError mean_squared_error;//mean_squared_error.print();

}

void register_optimization_algorithms()
{
    AdaptiveMomentEstimation adaptive_moment_estimation; //adaptive_moment_estimation.print();
    StochasticGradientDescent stochastic_gradient_descent; //stochastic_gradient_descent.print();
    QuasiNewtonMethod quasi_newton_method; //quasi_newton_method.print();
    LevenbergMarquardtAlgorithm levenberg_marquardt_algorithm; //levenberg_marquardt_algorithm.print();
}

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2025 Artificial Intelligence Techniques, SL.
//
// This library is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 2.1 of the License, or any later version.
//
// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.

// You should have received a copy of the GNU Lesser General Public
// License along with this library; if not, write to the Free Software
// Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
