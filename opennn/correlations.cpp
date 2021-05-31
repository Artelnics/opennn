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

namespace OpenNN
{

/// Calculates autocorrelation for a given number of maximum lags.
/// @param x Vector containing the data.
/// @param lags_number Maximum lags number.

Tensor<type, 1> autocorrelations(const ThreadPoolDevice* thread_pool_device, const Tensor<type, 1>& x, const Index& lags_number)
{
    Tensor<type, 1> autocorrelation(lags_number);

    const Index this_size = x.size();

    for(Index i = 0; i < lags_number; i++)
    {
        Tensor<type, 1> column_x(this_size-i);
        Tensor<type, 1> column_y(this_size-i);

        for(Index j = 0; j < this_size - i; j++)
        {
            column_x[j] = x(j);
            column_y[j] = x[j + i];
        }

        autocorrelation[i] = linear_correlation(thread_pool_device, column_x, column_y, false).r;
    }

    return autocorrelation;
}


/// Calculates the cross-correlation between two vectors.
/// @param x Vector containing data.
/// @param y Vector for computing the linear correlation with this vector.
/// @param maximum_lags_number Maximum lags for which cross-correlation is calculated.

Tensor<type, 1> cross_correlations(const ThreadPoolDevice* thread_pool_device,
                                   const Tensor<type, 1>& x, const Tensor<type, 1>& y, const Index& maximum_lags_number)
{
    if(y.size() != x.size())
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: Correlations.\n"
               << "Tensor<type, 1> calculate_cross_correlation(const Tensor<type, 1>&) method.\n"
               << "Both vectors must have the same size.\n";

        throw logic_error(buffer.str());
    }

    Tensor<type, 1> cross_correlation(maximum_lags_number);

    const Tensor<type, 0> this_mean = x.mean();
    const Tensor<type, 0> y_mean = y.mean();

    const Index this_size = x.size();

    for(Index i = 0; i < maximum_lags_number; i++)
    {
        Tensor<type, 1> column_x(this_size-i);
        Tensor<type, 1> column_y(this_size-i);

        for(Index j = 0; j < this_size - i; j++)
        {
            column_x[j] = x(j);
            column_y[j] = y[j + i];
        }

        cross_correlation[i] = linear_correlation(thread_pool_device, column_x, column_y, false).r;
    }

    return cross_correlation;
}


/// Calculate the logistic function with specific parameters 'a' and 'b'.
/// @param a Parameter a.
/// @param b Parameter b.

Tensor<type, 1> logistic(const type& a, const type& b, const Tensor<type, 1>& x)
{      
    const Tensor<type, 1> combination = b*x+a;

    return (1 + combination.exp().inverse()).inverse();
}


/// Calculate the coefficients of a linear regression (a, b) and the correlation among the variables.
/// @param x Vector of the independent variable.
/// @param y Vector of the dependent variable.

Correlation linear_correlation(const ThreadPoolDevice* thread_pool_device,
                            const Tensor<type, 1>& x,
                            const Tensor<type, 1>& y,
                            const bool& scale_data)
{
#ifdef OPENNN_DEBUG

    const Index x_size = x.size();

    ostringstream buffer;

    if(x_size != y.size())
    {
        buffer << "OpenNN Exception: Vector Template.\n"
               << "Correlation linear_correlation(const Tensor<type, 1>&) const method.\n"
               << "Y size must be equal to X size.\n";

        throw logic_error(buffer.str());
    }

#endif

    pair<Tensor<type, 1>, Tensor<type, 1>> filter_vectors = filter_missing_values(x,y);

    const Tensor<type, 1> new_x = scale_data ? scale_minimum_maximum(filter_vectors.first) : filter_vectors.first;
    const Tensor<type, 1> new_y = scale_data ? scale_minimum_maximum(filter_vectors.second) : filter_vectors.second;

    const Index new_size = new_x.size();

    Tensor<type, 0> s_x;
    Tensor<type, 0> s_y;

    Tensor<type, 0> s_xx;
    Tensor<type, 0> s_yy;

    Tensor<type, 0> s_xy;

    s_x.device(*thread_pool_device) = new_x.sum();
    s_y.device(*thread_pool_device) = new_y.sum();
    s_xx.device(*thread_pool_device) = new_x.square().sum();
    s_yy.device(*thread_pool_device) = new_y.square().sum();
    s_xy.device(*thread_pool_device) = (new_y*new_x).sum();

    Correlation linear_correlation;

    linear_correlation.correlation_type = Linear;

    if(abs(s_x()) < numeric_limits<type>::min()
    && abs(s_y()) < numeric_limits<type>::min()
    && abs(s_xx()) < numeric_limits<type>::min()
    && abs(s_yy()) < numeric_limits<type>::min()
    && abs(s_xy()) < numeric_limits<type>::min())
    {
        linear_correlation.a = 0;

        linear_correlation.b = 0;

        linear_correlation.r = 1.0;
    }
    else
    {
        linear_correlation.a =
            (s_y() * s_xx() - s_x() * s_xy())/(static_cast<type>(new_size) * s_xx() - s_x() * s_x());

        linear_correlation.b =
            ((static_cast<type>(new_size) * s_xy()) - (s_x() * s_y())) /((static_cast<type>(new_size) * s_xx()) - (s_x() * s_x()));

        if(sqrt((static_cast<type>(new_size) * s_xx() - s_x() * s_x()) *(static_cast<type>(new_size) * s_yy() - s_y() * s_y())) < numeric_limits<type>::min())
        {
            linear_correlation.r = 1.0;
        }
        else
        {
            linear_correlation.r =
                (static_cast<type>(new_size) * s_xy() - s_x() * s_y()) /
                sqrt((static_cast<type>(new_size) * s_xx() - s_x() * s_x()) *(static_cast<type>(new_size) * s_yy() - s_y() * s_y()));
        }
    }

    return linear_correlation;
}


/// Calculate the coefficients of a logarithmic regression (a, b) and the correlation among the variables
/// @param x Vector of the independent variable.
/// @param y Vector of the dependent variable.
/// @todo check

Correlation logarithmic_correlation(const ThreadPoolDevice* thread_pool_device, const Tensor<type, 1>& x, const Tensor<type, 1>& y)
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

        throw logic_error(buffer.str());
    }

#endif

    // Check negative values from x

    Correlation logarithmic_correlation;
    logarithmic_correlation.correlation_type = Logarithmic;

    for(Index i = 0; i < x.dimension(0); i++)
    {
        if(!::isnan(x(i)) && x(i) <= 0)
        {
            logarithmic_correlation.r = NAN;

            return logarithmic_correlation;
        }
    }

    logarithmic_correlation = linear_correlation(thread_pool_device, x.log(), y, false);

    return logarithmic_correlation;
}


/// Calculate the coefficients of a exponential regression (a, b) and the correlation among the variables
/// @param x Vector of the independent variable.
/// @param y Vector of the dependent variable.

Correlation exponential_correlation(const ThreadPoolDevice* thread_pool_device, const Tensor<type, 1>& x, const Tensor<type, 1>& y)
{
#ifdef OPENNN_DEBUG

    ostringstream buffer;

    if(x.size() != y.size())
    {
        buffer << "OpenNN Exception: Vector Template.\n"
               << "Correlation "
               "exponential_correlation(const Tensor<type, 1>&, const Tensor<type, 1>&) const method.\n"
               << "Y size must be equal to X size.\n";

        throw logic_error(buffer.str());
    }

#endif

    // Check negative values from y

    Correlation exponential_correlation;

    for(Index i = 0; i < y.dimension(0); i++)
    {
        if(!::isnan(y(i)) && y(i) <= 0)
        {
            exponential_correlation.correlation_type = Exponential;
            exponential_correlation.r = NAN;

            return exponential_correlation;
        }
    }

    exponential_correlation = linear_correlation(thread_pool_device, x, y.log(), false);

    exponential_correlation.correlation_type = Exponential;
    exponential_correlation.a = exp(exponential_correlation.a);
    exponential_correlation.b = exp(exponential_correlation.b);

    return exponential_correlation;
}


/// Calculate the coefficients of a power regression (a, b) and the correlation among the variables
/// @param x Vector of the independent variable.
/// @param y Vector of the dependent variable.

Correlation power_correlation(const ThreadPoolDevice* thread_pool_device, const Tensor<type, 1>& x, const Tensor<type, 1>& y)
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

        throw logic_error(buffer.str());
    }

#endif

    // Check negative values from x and y

    Correlation power_correlation;

    for(Index i = 0; i < x.dimension(0); i++)
    {
        if(!::isnan(x(i)) && x(i) <= 0)
        {
            power_correlation.correlation_type = Exponential;
            power_correlation.r = NAN;

            return power_correlation;
        }

        if(!::isnan(y(i)) && y(i) <= 0)
        {
            power_correlation.correlation_type = Exponential;
            power_correlation.r = NAN;

            return power_correlation;
        }
    }

    power_correlation = linear_correlation(thread_pool_device, x.log(), y.log(), false);

    power_correlation.correlation_type = Power;

    power_correlation.a = exp(power_correlation.a);

    return power_correlation;
}


/// Calculate the coefficients of a logistic regression (a, b) and the correlation among the variables
/// @param x Vector of the independent variable.
/// @param y Vector of the dependent variable.

Correlation logistic_correlation(const ThreadPoolDevice* thread_pool_device, const Tensor<type, 1>& x, const Tensor<type, 1>& y)
{
#ifdef OPENNN_DEBUG

    ostringstream buffer;

    if(y.size() != x.size())
    {
        buffer << "OpenNN Exception: Correlations.\n"
               << "type logistic_correlation(const Tensor<type, 1>&, const Tensor<type, 1>&) method.\n"
               << "Y size(" <<y.size()<<") must be equal to X size("<<x.size()<<").\n";

        throw logic_error(buffer.str());
    }

#endif

    // Filter missing values

    pair<Tensor<type, 1>, Tensor<type, 1>> filter_vectors = filter_missing_values(x,y);

    const Tensor<type, 1>& new_x = filter_vectors.first;
    const Tensor<type, 1>& new_y = filter_vectors.second;

    // Scale data

    // Inputs: scaled_x; Targets: sorted_y

    const Index input_variables_number = 1;
    const Index target_variables_number = 1;
    const Index samples_number = new_x.dimension(0);

    Tensor<type, 2> data(samples_number, input_variables_number+target_variables_number);

    for(Index j = 0; j < input_variables_number+target_variables_number; j++)
    {
        if(j < input_variables_number)
            for(Index i = 0; i < samples_number; i++)
                data(i,j) = new_x(i);
        else
            for(Index i = 0; i < samples_number; i++)
                data(i,j) = new_y(i);
    }

    DataSet data_set(data);
    data_set.set_training();

    NeuralNetwork neural_network(NeuralNetwork::Classification, {1,1});
    neural_network.set_parameters_random();

    TrainingStrategy training_strategy(&neural_network, &data_set);

    training_strategy.set_optimization_method(TrainingStrategy::OptimizationMethod::LEVENBERG_MARQUARDT_ALGORITHM);

    training_strategy.set_loss_method(TrainingStrategy::LossMethod::WEIGHTED_SQUARED_ERROR);

    training_strategy.get_loss_index_pointer()->set_regularization_method("NO_REGULARIZATION");

    training_strategy.set_display(false);

    training_strategy.perform_training();

    // Logistic correlation

    const Tensor<type, 1> coefficients = neural_network.get_parameters();

    // Regression results

    Correlation correlation;

    correlation.a = coefficients(0);
    correlation.b = coefficients(1);

    const Tensor<type, 1> logistic_y = logistic(correlation.a, correlation.b, new_x);

    correlation.r = linear_correlation(thread_pool_device, logistic_y, new_y, false).r;

    if(correlation.b < 0) correlation.r *= (-1);

    correlation.correlation_type = Logistic;

    return correlation;
}


vector<int> get_indices_sorted(Tensor<type,1>& x)
{
    vector<type> y(x.size());

    vector<int> index;

    size_t n(0);

    generate(begin(y), end(y), [&]{ return n++; });

    sort(begin(y), end(y), [&](int i1, int i2) { return x[i1] < x[i2]; } );

    for(auto v : y) index.push_back(v);

    return index;
}


Correlation multiple_logistic_correlation(const ThreadPoolDevice* thread_pool_device,
                                                  const Tensor<type, 2>& x,
                                                  const Tensor<type, 2>& y)
{
#ifdef OPENNN_DEBUG

    ostringstream buffer;

    if(y.size() != x.dimension(0))
    {
        buffer << "OpenNN Exception: Correlations.\n"
               << "type logistic_correlations(const Tensor<type, 1>&, const Tensor<type, 1>&) method.\n"
               << "Y size(" <<y.size()<<") must be equal to X size("<<x.size()<<").\n";

        throw logic_error(buffer.str());
    }

#endif

    // Filter missing values

    pair<Tensor<type, 2>, Tensor<type, 2>> filter_vectors = filter_missing_values(x,y);

    const Tensor<type, 2>& new_x = filter_vectors.first;
    const Tensor<type, 2>& new_y = filter_vectors.second;

    // Scale data

    const Tensor<type, 2> scaled_x = scale_minimum_maximum(new_x);
    const Tensor<type, 2> scaled_y = scale_minimum_maximum(new_y);

    const Index input_variables_number = scaled_x.dimension(1);
    const Index target_variables_number = new_y.dimension(1);
    const Index samples_number = scaled_x.dimension(0);

    Tensor<type, 2> data(samples_number, input_variables_number+target_variables_number);

    for(Index j = 0; j <input_variables_number+target_variables_number; j++)
    {
        if(j < input_variables_number)
        {
            for(Index i = 0; i < samples_number; i++)
            {
                data(i,j) = scaled_x(i,j);
            }
        }
        else
        {
            for(Index i = 0; i < samples_number; i++)
            {
                data(i,j) = new_y(i,j-input_variables_number);
            }
        }
    }

    DataSet data_set(data);
    data_set.set_training();

    NeuralNetwork neural_network;

    PerceptronLayer* perceptron_layer = new PerceptronLayer(input_variables_number, target_variables_number, PerceptronLayer::Logistic);

    neural_network.add_layer(perceptron_layer);

    neural_network.set_parameters_random();

    TrainingStrategy training_strategy(&neural_network, &data_set);

    training_strategy.set_optimization_method(TrainingStrategy::OptimizationMethod::LEVENBERG_MARQUARDT_ALGORITHM);
    training_strategy.set_loss_method(TrainingStrategy::LossMethod::NORMALIZED_SQUARED_ERROR);
    training_strategy.get_normalized_squared_error_pointer()->set_normalization_coefficient();

    training_strategy.get_loss_index_pointer()->set_regularization_method("L2_NORM");
    training_strategy.get_loss_index_pointer()->set_regularization_weight(static_cast<type>(0.01));

    training_strategy.set_display(false);
    training_strategy.get_optimization_algorithm_pointer()->set_display(false);

    training_strategy.perform_training();

    // Logistic correlation

    Correlation logistic_correlations;

    const Tensor<type, 1> bias = perceptron_layer->get_biases().chip(0,1);
    const Tensor<type, 2> weights = perceptron_layer->get_synaptic_weights();

    /// @todo
//    const Tensor<type, 2> logistic_y = logistic(thread_pool_device, bias, weights, scaled_x);

//    logistic_correlations.r = linear_correlation(thread_pool_device, logistic_y.chip(0,1), scaled_y.chip(0,1), false).r;

    logistic_correlations.correlation_type = Logistic;

    return logistic_correlations;
}


/// Calculate the Karl Pearson correlation between two variables.
/// @param x Matrix of the variable X.
/// @param y Matrix of the variable Y.

Correlation karl_pearson_correlation(const ThreadPoolDevice*, const Tensor<type, 2>& x, const Tensor<type, 2>& y)
{
#ifdef  OPENNN_DEBUG

    if(x.dimension(1) == 0)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: Correlation class."
               << "type karl_pearson_correlation(const Tensor<type, 2>&, const Tensor<type, 2>&) method.\n"
               << "Number of columns("<< x.dimension(1) <<") must be greater than zero.\n";

        throw logic_error(buffer.str());
    }

    if(y.dimension(1) == 0)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: Correlation class."
               << "type karl_pearson_correlation(const Tensor<type, 2>&, const Tensor<type, 2>&) method.\n"
               << "Number of columns("<< y.dimension(1) <<") must be greater than zero.\n";

        throw logic_error(buffer.str());
    }

    if(x.dimension(0) != y.dimension(0))
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: Correlation class."
               << "type karl_pearson_correlation(const Tensor<type, 2>&, const Tensor<type, 2>&) method.\n"
               << "Number of rows of the two variables must be equal\t"<< x.dimension(0) <<"!=" << y.dimension(0) << ".\n";

        throw logic_error(buffer.str());
    }

#endif

    const Index rows_number = x.dimension(0);
    const Index x_columns_number = x.dimension(1);
    const Index y_columns_number = y.dimension(1);

    Index x_NAN = 0;
    Index y_NAN = 0;

    for(Index i = 0; i < rows_number; i++)
    {
        if(isnan(x(i, 0))) x_NAN++;

        if(isnan(x(i, 0))) y_NAN++;
    }

    Tensor<type, 2> new_x;
    Tensor<type, 2> new_y;

    Index new_rows_number;
    x_NAN >= y_NAN ? new_rows_number = rows_number-x_NAN : new_rows_number = rows_number-y_NAN;

    if(x_NAN > 0 || y_NAN > 0)
    {
        new_x.resize(new_rows_number, x_columns_number);
        new_y.resize(new_rows_number, y_columns_number);

        Index row_index = 0;
        Index x_column_index = 0;
        Index y_column_index = 0;

        for(Index i = 0; i < rows_number; i++)
        {
            if(!::isnan(x(i,0)) && !::isnan(y(i,0)))
            {
                for(Index j = 0; j < x_columns_number; j++)
                {
                    new_x(row_index, x_column_index) = x(i,j);
                    x_column_index++;
                }

                for(Index j = 0; j < x_columns_number; j++)
                {
                    new_x(row_index, y_column_index) = y(i,j);
                    y_column_index++;
                }

                row_index++;
                x_column_index = 0;
                y_column_index = 0;
            }
        }
    }
    else
    {
        new_x = x;
        new_y = y;
    }

    const Index new_size = new_x.dimension(0);

    Tensor<Index, 2> contingency_table(new_x.dimension(1),new_y.dimension(1));

    for(Index i = 0; i < new_x.dimension(1); i ++)
    {
        for(Index j = 0; j < new_y.dimension(1); j ++)
        {
            Index count = 0;

            for(Index k = 0; k < new_size; k ++)
            {
                if(abs(new_x(k,i) + new_y(k,j) - static_cast<type>(2.0)) <= static_cast<type>(0.0001))
                {
                    count ++;

                    contingency_table(i,j) = count;
                }
            }
        }
    }

    Index k;

    if(x.dimension(1) <= y.dimension(1)) k = x.dimension(1);
    else k = y.dimension(1);

    const type chi_squared = chi_square_test(contingency_table.cast<type>());

    Correlation karl_pearson;

    karl_pearson.correlation_type = KarlPearson;

    const Tensor<type, 0> contingency_table_sum = contingency_table.cast<type>().sum();

    karl_pearson.r
            = sqrt(static_cast<type>(k) / static_cast<type>(k - 1.0)) * sqrt(chi_squared/(chi_squared + contingency_table_sum(0)));

    return karl_pearson;
}


/// Calculate the chi squared test statistic of a contingency table
/// @param matrix Matrix that represent the contingency table

type chi_square_test(const Tensor<type, 2>& matrix)
{
    // Eigen stuff

    const Eigen::array<int, 1> rows = {Eigen::array<int, 1>({1})};
    const Eigen::array<int, 1> columns = {Eigen::array<int, 1>({0})};

    const Tensor<type, 1> sum_columns = matrix.sum(columns);
    const Tensor<type, 1> sum_rows = matrix.sum(rows);

    const Tensor<type, 0> total = sum_columns.sum();

    Tensor<type, 2> row_colum(sum_rows.size(),sum_columns.size());

    for(Index i = 0; i < sum_columns.size(); i++)
    {
        for(Index j = 0; j < sum_rows.size(); j++)
        {
            row_colum(j,i) = sum_columns[i]*sum_rows[j];
        }
    }

    const Tensor<type, 2> ei = row_colum/total(0);

    const Tensor<type, 2> squared = (matrix.cast<type>()-ei)*(matrix.cast<type>()-ei);

    Tensor<type, 2> chi(sum_rows.size(),sum_columns.size());

    for(Index i = 0; i < sum_rows.size(); i++)
    {
        for(Index j = 0; j < sum_columns.size(); j ++)
        {
            chi(i,j) = squared(i,j)/ei(i,j);
        }
    }

    const Tensor<type, 0> chi_sum = chi.sum();

    return chi_sum(0);
}


/// Filter the missing values of two vectors

pair<Tensor<type, 1>, Tensor<type, 1>> filter_missing_values (const Tensor<type, 1>& x, const Tensor<type, 1>& y)
{
    Index new_size = 0;

    for(Index i = 0; i < x.size(); i++)
    {
        if(!::isnan(x(i)) && !::isnan(y(i))) new_size++;
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
        if(!::isnan(x(i)) && !::isnan(y(i)))
        {
            new_x(index) = x(i);
            new_y(index) = y(i);

            index++;
        }
    }

    return make_pair(new_x, new_y);
}


pair<Tensor<type, 2>, Tensor<type, 2>> filter_missing_values(const Tensor<type, 2>& x, const Tensor<type, 2>& y)
{
    Index rows_number = x.dimension(0);
    Index x_columns_number = x.dimension(1);
    Index y_columns_number = y.dimension(1);

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


Index count_NAN(const Tensor<type, 1>& x)
{
    Index NAN_number = 0;

    for(Index i = 0; i < x.size(); i++)
    {
        if(::isnan(x(i))) NAN_number++;
    }

    return NAN_number;
}


Tensor<type, 1> scale_minimum_maximum(const Tensor<type, 1>& x)
{
    const Tensor<type, 0> minimum = x.minimum();
    const Tensor<type, 0> maximum = x.maximum();

    const type min_range = -1;
    const type max_range = 1;

    const type slope = (max_range-min_range)/(maximum()-minimum());
    const type intercept = (min_range*maximum()-max_range*minimum())/(maximum()-minimum());

    Tensor<type, 1> scaled_x(x.size());

    for(Index i = 0; i < scaled_x.size(); i++)
    {
        scaled_x(i) = slope*x(i)+intercept;
    }

    return scaled_x;
}


Tensor<type, 2> scale_minimum_maximum(const Tensor<type, 2>& x)
{
    const Index rows_number = x.dimension(0);
    const Index columns_number = x.dimension(1);

    Tensor<type, 2> scaled_x(rows_number, columns_number);

    const Tensor<type, 1> columns_minimums = OpenNN::columns_minimums(x);

    const Tensor<type, 1> columns_maximums = OpenNN::columns_maximums(x);

    const type min_range = -1;
    const type max_range = 1;

    for(Index j = 0; j < columns_number; j++)
    {
        const type minimum = columns_minimums(j);
        const type maximum = columns_maximums(j);

        const type slope = (max_range-min_range)/(maximum-minimum);
        const type intercept = (min_range*maximum-max_range*minimum)/(maximum-minimum);

        for(Index i = 0; i < rows_number; i++)
        {
            scaled_x(i,j) = slope*x(i,j)+intercept;
        }
    }

    return scaled_x;
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
