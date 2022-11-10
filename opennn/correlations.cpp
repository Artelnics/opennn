//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   C O R R E L A T I O N S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "correlations.h"
#include "data_set.h"
#include "neural_network.h"
#include "training_strategy.h"

namespace opennn
{

/// Calculates autocorrelation for a given number of maximum lags.
/// @param x Vector containing the data.
/// @param lags_number Maximum lags number.

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


/// Calculates the correlation between two vectors.
/// @param x Vector containing data.
/// @param y Vector for computing the correlation with this vector.

Correlation correlation(const ThreadPoolDevice* thread_pool_device,
                        const Tensor<type, 2>& x,
                        const Tensor<type, 2>& y)
{
    Correlation correlation;

    const Index x_rows = x.dimension(0);
    const Index x_columns = x.dimension(1);
    const Index y_columns = y.dimension(1);

    const bool x_binary = is_binary(x);
    const bool y_binary = is_binary(y);

    const Eigen::array<Index, 1> vector{{x_rows}};

    if(x_columns == 1 && y_columns == 1)
    {
        if(!x_binary && !y_binary)
        {
            const Correlation linear_correlation
                    = opennn::linear_correlation(thread_pool_device, x.reshape(vector), y.reshape(vector));

            const Correlation exponential_correlation
                    = opennn::exponential_correlation(thread_pool_device, x.reshape(vector), y.reshape(vector));

            const Correlation logarithmic_correlation
                    = opennn::logarithmic_correlation(thread_pool_device, x.reshape(vector), y.reshape(vector));

            const Correlation power_correlation
                    = opennn::power_correlation(thread_pool_device, x.reshape(vector), y.reshape(vector));

            Correlation strongest_correlation = linear_correlation;

            if(abs(exponential_correlation.r) > abs(strongest_correlation.r))
                strongest_correlation = exponential_correlation;

            if(abs(logarithmic_correlation.r) > abs(strongest_correlation.r))
                strongest_correlation = logarithmic_correlation;

            if(abs(power_correlation.r) > abs(strongest_correlation.r))
                strongest_correlation = power_correlation;

            return strongest_correlation;
        }
        else if(!x_binary && y_binary)
        {
            return opennn::logistic_correlation_vector_vector(thread_pool_device, x.reshape(vector), y.reshape(vector));
        }
        else if(x_binary && !y_binary)
        {
            return opennn::logistic_correlation_vector_vector(thread_pool_device, y.reshape(vector), x.reshape(vector));
        }
        else if(x_binary && y_binary)
        {
            return opennn::linear_correlation(thread_pool_device, x.reshape(vector), y.reshape(vector));
        }
    }
    else if(x_columns != 1 && y_columns == 1)
    {
        return opennn::logistic_correlation_matrix_vector(thread_pool_device, x, y.reshape(vector));
    }
    else if(x_columns == 1 && y_columns != 1)
    {
        return opennn::logistic_correlation_vector_matrix(thread_pool_device, x.reshape(vector), y);
    }
    else if(x_columns != 1 && y_columns != 1)
    {
        return opennn::logistic_correlation_matrix_matrix(thread_pool_device, x, y);
    }
    else
    {
        throw invalid_argument("Correlations Exception: Unknown case.");
    }

    return correlation;
}


Correlation correlation_spearman(const ThreadPoolDevice* thread_pool_device,
                        const Tensor<type, 2>& x,
                        const Tensor<type, 2>& y)
{
    Correlation correlation;


    const Index x_rows = x.dimension(0);
    const Index x_columns = x.dimension(1);
    const Index y_columns = y.dimension(1);

    const bool x_binary = is_binary(x);
    const bool y_binary = is_binary(y);

    const Eigen::array<Index, 1> vector{{x_rows}};

    if(x_columns == 1 && y_columns == 1)
    {
        if(!x_binary && !y_binary)
        {
            const Correlation linear_correlation
                    = opennn::linear_correlation_spearman(thread_pool_device, x.reshape(vector), y.reshape(vector));

            return linear_correlation;
        }
        else if(!x_binary && y_binary)
        {
            return opennn::logistic_correlation_vector_vector_spearman(thread_pool_device, x.reshape(vector), y.reshape(vector));
        }
        else if(x_binary && !y_binary)
        {
            return opennn::logistic_correlation_vector_vector_spearman(thread_pool_device, y.reshape(vector), x.reshape(vector));
        }
        else if(x_binary && y_binary)
        {
            return opennn::linear_correlation_spearman(thread_pool_device, x.reshape(vector), y.reshape(vector));
        }
    }
    else if(x_columns != 1 && y_columns == 1)
    {
        return opennn::logistic_correlation_matrix_vector(thread_pool_device, x, y.reshape(vector));
    }
    else if(x_columns == 1 && y_columns != 1)
    {
        return opennn::logistic_correlation_vector_matrix(thread_pool_device, x.reshape(vector), y);
    }
    else if(x_columns != 1 && y_columns != 1)
    {
        return opennn::logistic_correlation_matrix_matrix(thread_pool_device, x, y);
    }
    else
    {
        throw invalid_argument("Correlations Exception: Unknown case.");
    }

    return correlation;
}


/// Calculates the cross-correlation between two vectors.
/// @param x Vector containing data.
/// @param y Vector for computing the linear correlation with this vector.
/// @param maximum_lags_number Maximum lags for which cross-correlation is calculated.

Tensor<type, 1> cross_correlations(const ThreadPoolDevice* thread_pool_device,
                                   const Tensor<type, 1>& x,
                                   const Tensor<type, 1>& y,
                                   const Index& maximum_lags_number)
{
    if(y.size() != x.size())
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: Correlations.\n"
               << "Tensor<type, 1> calculate_cross_correlation(const Tensor<type, 1>&) method.\n"
               << "Both vectors must have the same size.\n";

        throw invalid_argument(buffer.str());
    }

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


/// Calculate the coefficients of a exponential regression (a, b) and the correlation among the variables
/// @param x Vector of the independent variable.
/// @param y Vector of the dependent variable.

Correlation exponential_correlation(const ThreadPoolDevice* thread_pool_device,
                                    const Tensor<type, 1>& x,
                                    const Tensor<type, 1>& y)
{
#ifdef OPENNN_DEBUG

    ostringstream buffer;

    if(x.size() != y.size())
    {
        buffer << "OpenNN Exception: Vector Template.\n"
               << "Correlation "
               "exponential_correlation(const Tensor<type, 1>&, const Tensor<type, 1>&) const method.\n"
               << "Y size must be equal to X size.\n";

        throw invalid_argument(buffer.str());
    }

#endif

    // Check negative values from y

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

    exponential_correlation.correlation_type = CorrelationType::Exponential;

    exponential_correlation.a = exp(exponential_correlation.a);
    exponential_correlation.b = exponential_correlation.b;

    return exponential_correlation;
}


/// Filter the missing values of two vectors.
/// @param x First vector.
/// @param y Second vector.

pair<Tensor<type, 1>, Tensor<type, 1>> filter_missing_values_vector_vector(const Tensor<type, 1>& x,
                                                                           const Tensor<type, 1>& y)
{
    Index new_size = 0;

    for(Index i = 0; i < x.size(); i++)
    {
        if(!isnan(x(i)) && !isnan(y(i))) new_size++;
    }

    if(new_size == x.size())
    {
        return make_pair(x, y);
    }

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

    return make_pair(new_x, new_y);
}

/// Filter the missing values of two vectors.
/// @param x First vector.
/// @param y Second vector.

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

        if(isnan(y(i)))
        {
            not_NAN_row(i) = false;
        }
        else if(isnan(x(i)))
        {
            not_NAN_row(i) = false;
        }

        if(not_NAN_row(i)) new_rows_number++;
    }

    Tensor<type, 1> new_x(new_rows_number);

    Tensor<type, 2> new_y(new_rows_number,y_columns_number);

    Index index = 0;

    for(Index i = 0; i < rows_number; i++)
    {
        if(not_NAN_row(i))
        {
            for(Index j = 0; j < y_columns_number; j++)
            {
                new_y(index, j) = y(i,j);
            }

            new_x(index) = x(i);

            index++;
        }
    }

    return make_pair(new_x, new_y);
}


pair<Tensor<type, 1>, Tensor<type, 2>> filter_missing_values_matrix_vector(const Tensor<type, 2>&x,
                                                                           const Tensor<type, 1>y)
{
    return filter_missing_values_vector_matrix(y,x);
};



/// Filter the missing values of two matrix.
/// @param x First matrix.
/// @param y Second matrix.

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
                if(isnan(x(i,j)))
                {
                    not_NAN_row(i) = false;
                    break;
                }
            }
        }

        if(not_NAN_row(i)) new_rows_number++;
    }

    Tensor<type, 2> new_x(new_rows_number, x_columns_number);

    Tensor<type, 2> new_y(new_rows_number,y_columns_number);

    Index index = 0;

    for(Index i = 0; i < rows_number; i++)
    {
        if(not_NAN_row(i))
        {
            for(Index j = 0; j < y_columns_number; j++)
            {
                new_y(index, j) = y(i,j);
            }

            for(Index j = 0; j < x_columns_number; j++)
            {
                new_x(index, j) = x(i, j);
            }

            index++;
        }
    }

    return make_pair(new_x, new_y);
}


/// Get correlation values from a Correlation matrix.
/// @param correlations Correlation matrix.

Tensor<type, 2> get_correlation_values(const Tensor<Correlation, 2>& correlations)
{
    const Index rows_number = correlations.dimension(0);
    const Index columns_number = correlations.dimension(1);

    Tensor<type, 2> values(rows_number, columns_number);

    for(Index i = 0; i < rows_number; i++)
    {
        for(Index j = 0; j < columns_number; j++)
        {
            values(i,j) = correlations(i,j).r;
        }
    }

    return values;
}


/// Calculate the coefficients of a goodness-of-fit (a, b) and the correlation among the variables.
/// @param x Vector of the independent variable.
/// @param y Vector of the dependent variable.

Correlation linear_correlation(const ThreadPoolDevice* thread_pool_device,
                               const Tensor<type, 1>& x,
                               const Tensor<type, 1>& y)
{
#ifdef OPENNN_DEBUG

    const Index x_size = x.size();

    ostringstream buffer;

    if(x_size != y.size())
    {
        buffer << "OpenNN Exception: Vector Template.\n"
               << "Correlation linear_correlation(const Tensor<type, 1>&) const method.\n"
               << "Y size must be equal to X size.\n";

        throw invalid_argument(buffer.str());
    }

#endif

    Correlation linear_correlation;

    linear_correlation.correlation_type = CorrelationType::Linear;

    if(is_constant(x) && !is_constant(y))
    {
        cout << "Warning: Column X is constant." << endl;
        linear_correlation.a = NAN;
        linear_correlation.b = NAN;
        linear_correlation.r = NAN;

        linear_correlation.lower_confidence = NAN;
        linear_correlation.upper_confidence = NAN;

        return linear_correlation;
    }
    else if(!is_constant(x) && is_constant(y))
    {
        cout << "Warning: Column Y is constant." << endl;

        linear_correlation.a = y(0);
        linear_correlation.b = type(0);
        linear_correlation.r = NAN;


        linear_correlation.lower_confidence = NAN;
        linear_correlation.upper_confidence = NAN;

        return linear_correlation;
    }
    else if(is_constant(x) && is_constant(y))
    {
        cout << "Warning: Column X and column Y are constant." << endl;

        linear_correlation.a = NAN;
        linear_correlation.b = NAN;
        linear_correlation.r = NAN;

        linear_correlation.lower_confidence = NAN;
        linear_correlation.upper_confidence = NAN;

        return linear_correlation;
    }

    pair<Tensor<type, 1>, Tensor<type, 1>> filter_vectors = filter_missing_values_vector_vector(x,y);

    const Tensor<double, 1> x_filter = filter_vectors.first.cast<double>();
    const Tensor<double, 1> y_filter = filter_vectors.second.cast<double>();

    const Index n = x_filter.size();

    if(x_filter.size() == 0 )
    {
        cout << "Warning: Column X and Y hasn't common rows." << endl;

        linear_correlation.a = NAN;
        linear_correlation.b = NAN;
        linear_correlation.r = NAN;

        linear_correlation.lower_confidence = NAN;
        linear_correlation.upper_confidence = NAN;

        return linear_correlation;
    }

    Tensor<double, 0> s_x;
    Tensor<double, 0> s_y;

    Tensor<double, 0> s_xx;
    Tensor<double, 0> s_yy;

    Tensor<double, 0> s_xy;

    s_x.device(*thread_pool_device) = x_filter.sum();
    s_y.device(*thread_pool_device) = y_filter.sum();
    s_xx.device(*thread_pool_device) = x_filter.square().sum();
    s_yy.device(*thread_pool_device) = y_filter.square().sum();
    s_xy.device(*thread_pool_device) = (y_filter*x_filter).sum();

    if(abs(s_x()) < NUMERIC_LIMITS_MIN
    && abs(s_y()) < NUMERIC_LIMITS_MIN
    && abs(s_xx()) < NUMERIC_LIMITS_MIN
    && abs(s_yy()) < NUMERIC_LIMITS_MIN
    && abs(s_xy()) < NUMERIC_LIMITS_MIN)
    {
        linear_correlation.a = type(0);

        linear_correlation.b = type(0);

        linear_correlation.r = type(1);

        linear_correlation.lower_confidence = type(1);

        linear_correlation.upper_confidence = type(1);
    }
    else
    {

        linear_correlation.a =
            type((s_y() * s_xx() - s_x() * s_xy())/(static_cast<double>(n) * s_xx() - s_x() * s_x()));

        linear_correlation.b =
            type(((static_cast<double>(n) * s_xy()) - (s_x() * s_y())) /((static_cast<double>(n) * s_xx()) - (s_x() * s_x())));



        if(sqrt((static_cast<double>(n) * s_xx() - s_x() * s_x()) *(static_cast<double>(n) * s_yy() - s_y() * s_y())) < NUMERIC_LIMITS_MIN)
        {
            linear_correlation.r = NAN;
            linear_correlation.lower_confidence = NAN;
            linear_correlation.upper_confidence = NAN;
        }
        else
        {
            linear_correlation.r =
                type((static_cast<double>(n) * s_xy() - s_x() * s_y()) /
                sqrt((static_cast<double>(n) * s_xx() - s_x() * s_x()) *(static_cast<double>(n) * s_yy() - s_y() * s_y())));

            // Confidence intervals, with transformation of coefficients to Z distribution and back

            const type z_correlation = r_correlation_to_z_correlation(linear_correlation.r);

            const Tensor<type, 1> confidence_interval_z = confidence_interval_z_correlation(z_correlation, n);

            linear_correlation.lower_confidence = z_correlation_to_r_correlation(confidence_interval_z(0));

            linear_correlation.upper_confidence = z_correlation_to_r_correlation(confidence_interval_z(1));
        }
    }

    return linear_correlation;
}


type r_correlation_to_z_correlation(const type& r_correlation)
{
    const type z_correlation = 0.5*log((1+r_correlation)/(1 - r_correlation));

    return z_correlation;
}


type z_correlation_to_r_correlation (const type& z_correlation)
{
    const type r_correlation = (exp(2*z_correlation)-1) / (exp(2*z_correlation)+1);

    return r_correlation;
}



Tensor<type,1> confidence_interval_z_correlation(const type& z_correlation, const Index& n)
{
    Tensor<type, 1> confidence_interval(2);

    const type z_standard_error = 1.959964;

    confidence_interval(0) = z_correlation - z_standard_error * 1/sqrt(n - 3);

    confidence_interval(1) = z_correlation + z_standard_error * 1/sqrt(n - 3);

    return confidence_interval;
}


Tensor<type,1> calculate_spearman_ranks(const Tensor<type,1> & x)
{
    const int n = x.size();

    Tensor<type,1> rank_x(n);

    for(int i = 0; i < n; i++)
    {
        int r = 1, s = 1;

        // Count no of smaller elements in 0 to i-1

        for(int j = 0; j < i; j++)
        {
            if (x[j] < x[i] ) r++;
            if (x[j] == x[i] ) s++;
        }

        // Count no of smaller elements in i+1 to N-1

        for(int j = i+1; j < n; j++)
        {
            if (x[j] < x[i] ) r++;
            if (x[j] == x[i] ) s++;
        }

        // Use Fractional Rank formula fractional_rank = r + (n-1)/2

        rank_x[i] = r + (s-1) * 0.5;
    }

    return rank_x;
}


Correlation linear_correlation_spearman(const ThreadPoolDevice* thread_pool_device, const Tensor<type, 1>& x, const Tensor<type, 1>& y)
{
    pair<Tensor<type, 1>, Tensor<type, 1>> filter_vectors = filter_missing_values_vector_vector(x,y);

    Tensor<type, 1> x_filter = filter_vectors.first.cast<type>();
    Tensor<type, 1> y_filter = filter_vectors.second.cast<type>();

    const Tensor<type, 1> x_rank = calculate_spearman_ranks(x_filter);
    const Tensor<type, 1> y_rank = calculate_spearman_ranks(y_filter);

    return linear_correlation(thread_pool_device, x_rank, y_rank);
}


/// Calculate the coefficients of a logarithmic regression (a, b) and the correlation among the variables
/// @param x Vector of the independent variable.
/// @param y Vector of the dependent variable.

Correlation logarithmic_correlation(const ThreadPoolDevice* thread_pool_device,
                                    const Tensor<type, 1>& x,
                                    const Tensor<type, 1>& y)
{
#ifdef OPENNN_DEBUG

    Index n = y.size();

    const Index x_size = x.size();

    ostringstream buffer;

    if(x_size != n)
    {
        buffer << "OpenNN Exception: Vector Template.\n"
               << "Correlation "
               "logarithmic_correlation(const Tensor<type, 1>&) const "
               "method.\n"
               << "Y size must be equal to X size.\n";

        throw invalid_argument(buffer.str());
    }

#endif

    // Check negative values from x

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

    logarithmic_correlation.correlation_type = CorrelationType::Logarithmic;

    return logarithmic_correlation;
}


/// Calculate the coefficients of a logistic regression (a, b) and the correlation among the variables
/// @param x Vector of the independent variable.
/// @param y Vector of the dependent variable.

Correlation logistic_correlation_vector_vector(const ThreadPoolDevice* thread_pool_device,
                                               const Tensor<type, 1>& x,
                                               const Tensor<type, 1>& y)
{
    Correlation correlation;

    pair<Tensor<type,1>, Tensor<type,1>> filtered_elements = filter_missing_values_vector_vector(x,y);

    Tensor<type,1> x_filtered = filtered_elements.first;
    Tensor<type,1> y_filtered = filtered_elements.second;

    if(x_filtered.size() == 0)
    {
        correlation.r = static_cast<type>(NAN);

        correlation.correlation_type = CorrelationType::Logistic;

        return correlation;
    }

    const Tensor<type, 2> data = opennn::assemble_vector_vector(x_filtered, y_filtered);

    DataSet data_set(data);
    data_set.set_training();

    data_set.set_columns_scalers(Scaler::MinimumMaximum);

    NeuralNetwork neural_network(NeuralNetwork::ProjectType::Classification, {1,1});
    neural_network.get_scaling_layer_pointer()->set_display(false);

    neural_network.get_probabilistic_layer_pointer()->set_activation_function(ProbabilisticLayer::ActivationFunction::Logistic);

    TrainingStrategy training_strategy(&neural_network, &data_set);
    training_strategy.set_display(false);

    training_strategy.set_loss_method(TrainingStrategy::LossMethod::MEAN_SQUARED_ERROR);

    training_strategy.set_optimization_method(TrainingStrategy::OptimizationMethod::LEVENBERG_MARQUARDT_ALGORITHM);

    training_strategy.get_loss_index_pointer()->set_regularization_method(LossIndex::RegularizationMethod::NoRegularization);

    training_strategy.perform_training();

    Tensor<type, 2> inputs = data_set.get_input_data();
    Tensor<Index, 1> inputs_dimensions = get_dimensions(inputs);

    const Tensor<type, 2> targets = data_set.get_target_data();

    Tensor<type, 2> outputs;

    outputs = neural_network.calculate_outputs(inputs.data(), inputs_dimensions);

    // Logistic correlation

    const Eigen::array<Index, 1> vector{{x_filtered.size()}};

    correlation.r = linear_correlation(thread_pool_device, outputs.reshape(vector), targets.reshape(vector)).r;

    const type z_correlation = r_correlation_to_z_correlation(correlation.r);

    const Tensor<type, 1> confidence_interval_z = confidence_interval_z_correlation(z_correlation, inputs_dimensions(0));

    correlation.lower_confidence = z_correlation_to_r_correlation(confidence_interval_z(0));

    correlation.upper_confidence = z_correlation_to_r_correlation(confidence_interval_z(1));

    correlation.correlation_type = CorrelationType::Logistic;

    const Tensor<type, 1> coefficients = neural_network.get_parameters();

    correlation.a = coefficients(0);
    correlation.b = coefficients(1);
    // no r correlation here

    if(correlation.b < type(0)) correlation.r *= type(-1);

    return correlation;
}



Correlation logistic_correlation_vector_vector_spearman(const ThreadPoolDevice* thread_pool_device,
                                                        const Tensor<type, 1>& x,
                                                        const Tensor<type, 1>& y)
{
    Correlation correlation;

    pair<Tensor<type,1>, Tensor<type,1>> filtered_elements = filter_missing_values_vector_vector(x,y);

    Tensor<type,1> x_filtered = filtered_elements.first;
    Tensor<type,1> y_filtered = filtered_elements.second;

    if(x_filtered.size() == 0)
    {
        correlation.r = static_cast<type>(NAN);

        correlation.correlation_type = CorrelationType::Logistic;

        return correlation;
    }

    Tensor<type,1> x_rank = calculate_spearman_ranks(x_filtered);

    const Tensor<type, 2> data = opennn::assemble_vector_vector(x_rank, y_filtered);

    DataSet data_set(data);
    data_set.set_training();

    data_set.set_columns_scalers(Scaler::MinimumMaximum);

    NeuralNetwork neural_network(NeuralNetwork::ProjectType::Classification, {1,1});
    neural_network.get_scaling_layer_pointer()->set_display(false);

    neural_network.get_probabilistic_layer_pointer()->set_activation_function(ProbabilisticLayer::ActivationFunction::Logistic);

    TrainingStrategy training_strategy(&neural_network, &data_set);
    training_strategy.set_display(false);

    training_strategy.set_loss_method(TrainingStrategy::LossMethod::MEAN_SQUARED_ERROR);

    training_strategy.set_optimization_method(TrainingStrategy::OptimizationMethod::LEVENBERG_MARQUARDT_ALGORITHM);

    training_strategy.get_loss_index_pointer()->set_regularization_method(LossIndex::RegularizationMethod::NoRegularization);

    training_strategy.perform_training();

    Tensor<type, 2> inputs = data_set.get_input_data();
    Tensor<Index, 1> inputs_dimensions = get_dimensions(inputs);

    const Tensor<type, 2> targets = data_set.get_target_data();

    Tensor<type, 2> outputs;

    outputs = neural_network.calculate_outputs(inputs.data(), inputs_dimensions);

    // Logistic correlation

    const Eigen::array<Index, 1> vector{{x_filtered.size()}};

    correlation.r = linear_correlation(thread_pool_device, outputs.reshape(vector), targets.reshape(vector)).r;

    const type z_correlation = r_correlation_to_z_correlation(correlation.r);

    const Tensor<type, 1> confidence_interval_z = confidence_interval_z_correlation(z_correlation, inputs_dimensions(0));

    correlation.lower_confidence = z_correlation_to_r_correlation(confidence_interval_z(0));

    correlation.upper_confidence = z_correlation_to_r_correlation(confidence_interval_z(1));

    correlation.correlation_type = CorrelationType::Logistic;

    const Tensor<type, 1> coefficients = neural_network.get_parameters();

    correlation.a = coefficients(0);
    correlation.b = coefficients(1);

    if(correlation.b < type(0)) correlation.r *= type(-1);

    return correlation;
}



Correlation logistic_correlation_vector_matrix(const ThreadPoolDevice* thread_pool_device,
                                               const Tensor<type, 1>& x,
                                               const Tensor<type, 2>& y)
{
    Correlation correlation;

    pair<Tensor<type,1>, Tensor<type,2>> filtered_elements = opennn::filter_missing_values_vector_matrix(x, y);

    Tensor<type,1> x_filtered = filtered_elements.first;
    Tensor<type,2> y_filtered = filtered_elements.second;

    if( y_filtered.dimension(1) > 50)
    {
        cout << "Warning: Y variable has too many categories." << endl;

        correlation.r = static_cast<type>(NAN);

        correlation.correlation_type = CorrelationType::Logistic;

        return correlation;
    }

    if(x_filtered.size() == 0)
    {
        correlation.r = static_cast<type>(NAN);

        correlation.correlation_type = CorrelationType::Logistic;

        return correlation;
    }

    const Tensor<type, 2> data = opennn::assemble_vector_matrix(x_filtered, y_filtered);

    Tensor<Index, 1> input_columns_indices(1);
    input_columns_indices(0) = 0;

    Tensor<Index, 1> target_columns_indices(y_filtered.dimension(1));
    for(Index i = 0; i < y_filtered.dimension(1); i++) target_columns_indices(i) = i + 1;

    DataSet data_set(data);

    data_set.set_input_target_columns(input_columns_indices, target_columns_indices);

    data_set.set_training();

    const Index input_variables_number = data_set.get_input_variables_number();
    const Index target_variables_number = data_set.get_target_variables_number();

    NeuralNetwork neural_network(NeuralNetwork::ProjectType::Classification, {input_variables_number, target_variables_number});
    neural_network.get_probabilistic_layer_pointer()->set_activation_function(ProbabilisticLayer::ActivationFunction::Logistic);
    neural_network.get_scaling_layer_pointer()->set_display(false);

    TrainingStrategy training_strategy(&neural_network, &data_set);

    training_strategy.set_optimization_method(TrainingStrategy::OptimizationMethod::LEVENBERG_MARQUARDT_ALGORITHM);

    training_strategy.set_loss_method(TrainingStrategy::LossMethod::MEAN_SQUARED_ERROR);

    training_strategy.get_loss_index_pointer()->set_regularization_method(LossIndex::RegularizationMethod::NoRegularization);

    training_strategy.set_display(false);

    training_strategy.set_display_period(1000);

    training_strategy.perform_training();

    // Logistic correlation

    Tensor<type, 2> inputs = data_set.get_input_data();
    Tensor<Index, 1> inputs_dimensions = get_dimensions(inputs);

    Tensor<type, 2> targets = data_set.get_target_data();

    Tensor<type, 2> outputs;

    outputs = neural_network.calculate_outputs(inputs.data(), inputs_dimensions);

    const Eigen::array<Index, 1> vector{{targets.size()}};

    correlation.r = linear_correlation(thread_pool_device, outputs.reshape(vector), targets.reshape(vector)).r;

    const type z_correlation = r_correlation_to_z_correlation(correlation.r);

    const Tensor<type, 1> confidence_interval_z = confidence_interval_z_correlation(z_correlation, inputs_dimensions(0));

    correlation.lower_confidence = z_correlation_to_r_correlation(confidence_interval_z(0));

    correlation.upper_confidence = z_correlation_to_r_correlation(confidence_interval_z(1));

    correlation.correlation_type = CorrelationType::Logistic;

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

    pair<Tensor<type,2>, Tensor<type,2>> filtered_matrixes = filter_missing_values_matrix_matrix(x,y);

    Tensor<type,2> x_filtered = filtered_matrixes.first;
    Tensor<type,2> y_filtered = filtered_matrixes.second;

    if(x.dimension(0)  == y.dimension(0) && x.dimension(1)  == y.dimension(1))
    {
        Tensor<bool, 0> are_equal = ( x_filtered == y_filtered).all();

        if(are_equal(0))
        {
            correlation.r = static_cast<type>(1);

            correlation.correlation_type = CorrelationType::Logistic;

            return correlation;
        }
    }

    if(x.dimension(1) > 50 || y.dimension(1) > 50)
    {
        cout << "Warning: One variable has too many categories." << endl;

        correlation.r = static_cast<type>(NAN);

        correlation.correlation_type = CorrelationType::Logistic;

        return correlation;
    }


    if(x_filtered.size() == 0 && y_filtered.size() == 0)
    {
        correlation.r = static_cast<type>(NAN);

        correlation.correlation_type = CorrelationType::Logistic;

        return correlation;
    }

    const Tensor<type, 2> data = opennn::assemble_matrix_matrix(x_filtered, y_filtered);

    Tensor<Index, 1> input_columns_indices(x_filtered.dimension(1));
    for(Index i = 0; i < x_filtered.dimension(1); i++)
    {
            input_columns_indices(i) = i;
    }

    Tensor<Index, 1> target_columns_indices(y_filtered.dimension(1));
    for(Index i = 0; i < y_filtered.dimension(1); i++)
    {
            target_columns_indices(i) = x_filtered.dimension(1)+i;
    }

    DataSet data_set(data);

    data_set.set_input_target_columns(input_columns_indices, target_columns_indices);

    data_set.set_training();

    const Index input_variables_number = data_set.get_input_variables_number();
    const Index target_variables_number = data_set.get_target_variables_number();

    NeuralNetwork neural_network(NeuralNetwork::ProjectType::Classification, {input_variables_number, target_variables_number});
    neural_network.get_probabilistic_layer_pointer()->set_activation_function(ProbabilisticLayer::ActivationFunction::Softmax);
    neural_network.get_scaling_layer_pointer()->set_display(false);

    TrainingStrategy training_strategy(&neural_network, &data_set);

    training_strategy.get_loss_index_pointer()->set_regularization_method(LossIndex::RegularizationMethod::NoRegularization);

    training_strategy.set_optimization_method(TrainingStrategy::OptimizationMethod::LEVENBERG_MARQUARDT_ALGORITHM);

    training_strategy.set_loss_method(TrainingStrategy::LossMethod::MEAN_SQUARED_ERROR);

    training_strategy.set_maximum_epochs_number(500);

    training_strategy.set_display(false);

    training_strategy.perform_training();

    // Logistic correlation

    Tensor<type, 2> inputs = data_set.get_input_data();
    Tensor<Index, 1> inputs_dimensions = get_dimensions(inputs);

    Tensor<type, 2> targets = data_set.get_target_data();

    Tensor<type, 2> outputs;

    outputs = neural_network.calculate_outputs(inputs.data(), inputs_dimensions);

    const Eigen::array<Index, 1> vector{{targets.size()}};

    correlation.r = linear_correlation(thread_pool_device, outputs.reshape(vector), targets.reshape(vector)).r;

    const type z_correlation = r_correlation_to_z_correlation(correlation.r);

    const Tensor<type, 1> confidence_interval_z = confidence_interval_z_correlation(z_correlation, inputs_dimensions(0));

    correlation.lower_confidence = z_correlation_to_r_correlation(confidence_interval_z(0));

    correlation.upper_confidence = z_correlation_to_r_correlation(confidence_interval_z(1));

    correlation.correlation_type = CorrelationType::Logistic;

    return correlation;
}


/// Calculate the coefficients of a power regression (a, b) and the correlation among the variables
/// @param x Vector of the independent variable.
/// @param y Vector of the dependent variable.

Correlation power_correlation(const ThreadPoolDevice* thread_pool_device,
                              const Tensor<type, 1>& x,
                              const Tensor<type, 1>& y)
{
#ifdef OPENNN_DEBUG

    ostringstream buffer;

    if(x.size() != y.size())
    {
        buffer << "OpenNN Exception: Vector Template.\n"
               << "Correlation "
                  "power_correlation(const Tensor<type, 1>&) const "
                  "method.\n"
               << "Y size must be equal to X size.\n";

        throw invalid_argument(buffer.str());
    }

#endif

    // Check negative values from x and y

    Correlation power_correlation;

    for(Index i = 0; i < x.dimension(0); i++)
    {
        if(!isnan(x(i)) && x(i) <= type(0))
        {
            power_correlation.r = type(NAN);

            return power_correlation;
        }

        if(!isnan(y(i)) && y(i) <= type(0))
        {
            power_correlation.r = type(NAN);

            return power_correlation;
        }
    }

    power_correlation = linear_correlation(thread_pool_device, x.log(), y.log());

    power_correlation.correlation_type = CorrelationType::Power;

    power_correlation.a = exp(power_correlation.a);

    return power_correlation;
}

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2021 Artificial Intelligence Techniques, SL.
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
