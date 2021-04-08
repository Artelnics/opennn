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

/// Calculates the linear correlation coefficient(Spearman method)(R-value) between two vectors.
/// @param x Vector containing data.
/// @param y Vector for computing the linear correlation with the x vector.

type linear_correlation(const ThreadPoolDevice* thread_pool_device,
                        const Tensor<type, 1>& x, const Tensor<type, 1>& y, const bool & scale_vectors)
{
    return linear_regression(thread_pool_device, x, y, scale_vectors).correlation;
}


/// Calculates the Rank-Order correlation coefficient (Spearman method) between two vectors.
/// @param x Vector containing data.
/// @param y Vector for computing the linear correlation with the x vector.
/// @todo Missing values

type rank_linear_correlation(const ThreadPoolDevice* thread_pool_device, const Tensor<type, 1>& x, const Tensor<type, 1>& y)
{

#ifdef OPENNN_DEBUG

    ostringstream buffer;

    if(y.size() != x.size())
    {
        buffer << "OpenNN Exception: Correlations.\n"
               << "type rank_linear_correlation(const Tensor<type, 1>&, const Tensor<type, 1>&) method.\n"
               << "Y size must be equal to X size.\n";

        throw logic_error(buffer.str());
    }

#endif

    const Tensor<type, 1> ranks_x = less_rank_with_ties(x);
    const Tensor<type, 1> ranks_y = less_rank_with_ties(y);

    return linear_correlation(thread_pool_device, ranks_x, ranks_y);
}


/// Calculates the correlation between Y and exp(A*X + B).
/// @param x Vector containing the input values.
/// @param y Vector containing the target values.
/// @param missing Vector with the missing samples indices.

type exponential_correlation(const ThreadPoolDevice* thread_pool_device, const Tensor<type, 1>& x, const Tensor<type, 1>& y)
{
    return exponential_regression(thread_pool_device, x, y).correlation;
}


/// Calculates the correlation between Y and ln(A*X+B).
/// @param x Vector containing the input values.
/// @param y Vector containing the target values.

type logarithmic_correlation(const ThreadPoolDevice* thread_pool_device,
                             const Tensor<type, 1>& x, const Tensor<type, 1>& y)
{
    return logarithmic_regression(thread_pool_device, x, y).correlation;
}


/// Calculates the logistic correlation coefficient between two vectors.
/// It uses non-parametric Spearman Rank-Order method.
/// It uses a Neural Network to compute the logistic function approximation.
/// @param x Vector containing data.
/// @param y Vector for computing the linear correlation with the x vector.

type rank_logistic_correlation(const ThreadPoolDevice* thread_pool_device, const Tensor<type, 1>& x, const Tensor<type, 1>& y)
{
    const Tensor<type, 1> x_new = less_rank_with_ties(x);

    return logistic_correlations(thread_pool_device, x_new, y).correlation;
}


/// Calculates the power correlation between two variables.
/// @param x Vector of the independent variable
/// @param y Vector of the dependent variable

type power_correlation(const ThreadPoolDevice* thread_pool_device, const Tensor<type, 1>& x, const Tensor<type, 1>& y)
{
    return power_regression(thread_pool_device, x, y).correlation;
}


///Calculate the Karl Pearson correlation between two variables.
/// @param x Matrix of the variable X.
/// @param y Matrix of the variable Y.

type karl_pearson_correlation(const ThreadPoolDevice*, const Tensor<type, 2>& x, const Tensor<type, 2>& y)
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

        if(isnan(y(i, 0))) y_NAN++;
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

    Tensor<Index, 2> contingency_table(new_x.dimension(1), new_y.dimension(1));

    for(Index i = 0; i < new_x.dimension(1); i ++)
    {
        for(Index j = 0; j < new_y.dimension(1); j ++)
        {
            Index count = 0;

            for(Index k = 0; k < new_size; k ++)
            {
                if(fabsf(new_x(k,i) + new_y(k,j) - static_cast<type>(2.0)) <= static_cast<type>(1.0e-4))
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

    const Tensor<type, 0> contingency_table_sum = contingency_table.cast<type>().sum();

    return sqrt(static_cast<type>(k) / static_cast<type>(k - 1.0)) * sqrt(chi_squared/(chi_squared + contingency_table_sum(0)));
}


/// Calculates autocorrelation for a given number of maximum lags.
/// @param x Vector containing the data.
/// @param lags_number Maximum lags number.

Tensor<type, 1> autocorrelations(const ThreadPoolDevice* thread_pool_device, const Tensor<type, 1>& x, const Index& lags_number)
{
    Tensor<type, 1> autocorrelation(lags_number);

    const Tensor<type, 0> mean = x.mean();

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

        autocorrelation[i] = linear_correlation(thread_pool_device, column_x, column_y, false);
    }

    return autocorrelation;
}


/// Calculates the cross-correlation between two vectors.
/// @param x Vector containing data.
/// @param y Vector for computing the linear correlation with this vector.
/// @param maximum_lags_number Maximum lags for which cross-correlation is calculated.

Tensor<type, 1> cross_correlations(const ThreadPoolDevice* thread_pool_device, const Tensor<type, 1>& x, const Tensor<type, 1>& y, const Index& maximum_lags_number)
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

        cross_correlation[i] = linear_correlation(thread_pool_device, column_x, column_y, false);
    }

    return cross_correlation;
}


/// Returns a vector with the logistic error gradient.
/// @param coeffients.
/// @param x Independent data.
/// @param y Dependent data.

Tensor<type, 1> logistic_error_gradient(const type& a, const type& b, const Tensor<type, 1>& x, const Tensor<type, 1>& y)
{

#ifdef OPENNN_DEBUG

    const Index n = y.size();
    const Index x_size = x.size();

    ostringstream buffer;

    if(x_size != n)
    {
        buffer << "OpenNN Exception: type.\n"
               << "logistic error(const type&, const type&, const Tensor<type, 1>&, const Tensor<type, 1>& "
               "method.\n"
               << "Y size must be equal to X size.\n";

        throw logic_error(buffer.str());
    }

#endif

    Tensor<type, 1> error_gradient(2);

    const Tensor<type, 1> activation = logistic(a, b, x);
    const Tensor<type, 1> error = activation - y;

    Tensor<type, 0> sum_a = (2*error*activation*(-1+activation)).sum();
    Tensor<type, 0> sum_b = (2*error*activation*(-1+activation)*(-x)).sum();

    error_gradient(0) = sum_a();
    error_gradient(1) = sum_b();

    return error_gradient;
}


/// Calculate the logistic function with specific parameters 'a' and 'b'.
/// @param a Parameter a.
/// @param b Parameter b.

type logistic(const type& a, const type& b, const type& x)
{
    return static_cast<type>(1.0)/(static_cast<type>(1.0) + exp(-(a+b*x)));
}


/// Calculate the logistic function with specific parameters 'a' and 'b'.
/// @param a Parameter a.
/// @param b Parameter b.

Tensor<type, 1> logistic(const type& a, const type& b, const Tensor<type, 1>& x)
{      
    const Tensor<type, 1> combination = b*x+a;

    return (1 + combination.exp().inverse()).inverse();
}


/// Calculate the gaussian function with specific parameters 'a' and 'b'.
/// @param a Parameter a.
/// @param b Parameter b.

Tensor<type, 1> gaussian (const type& a, const type& b, const Tensor<type, 1>& x)
{
    const Tensor<type, 1> combination =(-0.5*((x-a)/b)*((x-a)/b)).exp();

    return combination;
}


/// Calculate the logistic function with specific parameters 'a' and 'b'.
/// @param a Parameter a.
/// @param b Parameter b.

Tensor<type, 2> logistic(const ThreadPoolDevice* thread_pool_device, const Tensor<type, 1>& a, const Tensor<type, 2>& b, const Tensor<type, 2>& x)
{
    const Index samples_number = x.dimension(0);
    const Index biases_number = a.dimension(0);

    Tensor<type, 2> combinations(samples_number, biases_number);

    for(Index i = 0; i < biases_number; i++)
    {
        fill_n(combinations.data() + i*samples_number, samples_number, a(i));
    }

    const Eigen::array<IndexPair<Index>, 1> A_B = {IndexPair<Index>(1, 0)};

    combinations.device(*thread_pool_device) += x.contract(b, A_B);

    return (1 + combinations.exp().inverse()).inverse();
}


///Calculate the mean square error of the logistic function.

type logistic_error(const type& a, const type& b, const Tensor<type, 1>& x, const Tensor<type, 1>& y)
{
    const Index n = y.size();

#ifdef OPENNN_DEBUG

    const Index x_size = x.size();

    ostringstream buffer;

    if(x_size != n)
    {
        buffer << "OpenNN Exception: type.\n"
               << "logistic error(const type&, const type&, const Tensor<type, 1>&, const Tensor<type, 1>& "
               "method.\n"
               << "Y size must be equal to X size.\n";

        throw logic_error(buffer.str());
    }

#endif

    Tensor<type, 0> error = (logistic(a, b, x) - y).square().sum();

    return error()/static_cast<type>(n);
}


///Calculate the coefficients of a linear regression (a, b) and the correlation among the variables.
/// @param x Vector of the independent variable.
/// @param y Vector of the dependent variable.

RegressionResults linear_regression(const ThreadPoolDevice* thread_pool_device,
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
               << "RegressionResults linear_regression(const Tensor<type, 1>&) const method.\n"
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
                        ;
    Tensor<type, 0> s_xy;

    s_x.device(*thread_pool_device) = new_x.sum();
    s_y.device(*thread_pool_device) = new_y.sum();
    s_xx.device(*thread_pool_device) = new_x.square().sum();
    s_yy.device(*thread_pool_device) = new_y.square().sum();
    s_xy.device(*thread_pool_device) = (new_y*new_x).sum();

    RegressionResults linear_regression;

    linear_regression.regression_type = Linear;

    if(abs(s_x()) < numeric_limits<type>::min()
            && abs(s_y()) < numeric_limits<type>::min()
            && abs(s_xx()) < numeric_limits<type>::min()
            && abs(s_yy()) < numeric_limits<type>::min()
            && abs(s_xy()) < numeric_limits<type>::min())
    {
        linear_regression.a = 0;

        linear_regression.b = 0;

        linear_regression.correlation = 1.0;
    }
    else
    {
        linear_regression.a =
            (s_y() * s_xx() - s_x() * s_xy()) /(static_cast<type>(new_size) * s_xx() - s_x() * s_x());

        linear_regression.b =
            ((static_cast<type>(new_size) * s_xy()) - (s_x() * s_y())) /((static_cast<type>(new_size) * s_xx()) - (s_x() * s_x()));

        if(sqrt((static_cast<type>(new_size) * s_xx() - s_x() * s_x()) *(static_cast<type>(new_size) * s_yy() - s_y() * s_y())) < numeric_limits<type>::min())
        {
            linear_regression.correlation = 1.0;
        }
        else
        {
            linear_regression.correlation =
                (static_cast<type>(new_size) * s_xy() - s_x() * s_y()) /
                sqrt((static_cast<type>(new_size) * s_xx() - s_x() * s_x()) *(static_cast<type>(new_size) * s_yy() - s_y() * s_y()));
        }
    }

    return linear_regression;
}


///Calculate the coefficients of a logarithmic regression (a, b) and the correlation among the variables
/// @param x Vector of the independent variable.
/// @param y Vector of the dependent variable.
/// @todo check

RegressionResults logarithmic_regression(const ThreadPoolDevice* thread_pool_device, const Tensor<type, 1>& x, const Tensor<type, 1>& y)
{
#ifdef OPENNN_DEBUG

    Index n = y.size();

    const Index x_size = x.size();

    ostringstream buffer;

    if(x_size != n)
    {
        buffer << "OpenNN Exception: Vector Template.\n"
               << "RegressionResults "
               "logarithmic_regression(const Tensor<type, 1>&) const "
               "method.\n"
               << "Y size must be equal to X size.\n";

        throw logic_error(buffer.str());
    }

#endif

    // Check negative values from x

    RegressionResults logarithmic_regression;

    for(Index i = 0; i < x.dimension(0); i++)
    {
        if(!::isnan(x(i)) && x(i) <= 0)
        {
            logarithmic_regression.regression_type = Logarithmic;
            logarithmic_regression.correlation = NAN;

            return logarithmic_regression;
        }
    }

    logarithmic_regression = linear_regression(thread_pool_device, x.log(), y, false);

    logarithmic_regression.regression_type = Logarithmic;

    return logarithmic_regression;
}


///Calculate the coefficients of a exponential regression (a, b) and the correlation among the variables
/// @param x Vector of the independent variable.
/// @param y Vector of the dependent variable.

RegressionResults exponential_regression(const ThreadPoolDevice* thread_pool_device, const Tensor<type, 1>& x, const Tensor<type, 1>& y)
{
#ifdef OPENNN_DEBUG

    ostringstream buffer;

    if(x.size() != y.size())
    {
        buffer << "OpenNN Exception: Vector Template.\n"
               << "RegressionResults "
               "exponential_regression(const Tensor<type, 1>&, const Tensor<type, 1>&) const method.\n"
               << "Y size must be equal to X size.\n";

        throw logic_error(buffer.str());
    }

#endif

    // Check negative values from y

    RegressionResults exponential_regression;

    for(Index i = 0; i < y.dimension(0); i++)
    {
        if(!::isnan(y(i)) && y(i) <= 0)
        {
            exponential_regression.regression_type = Exponential;
            exponential_regression.correlation = NAN;

            return exponential_regression;
        }
    }

    exponential_regression = linear_regression(thread_pool_device, x, y.log(), false);

    exponential_regression.regression_type = Exponential;
    exponential_regression.a = exp(exponential_regression.a);
    exponential_regression.b = exp(exponential_regression.b);

    return exponential_regression;
}


///Calculate the coefficients of a power regression (a, b) and the correlation among the variables
/// @param x Vector of the independent variable.
/// @param y Vector of the dependent variable.

RegressionResults power_regression(const ThreadPoolDevice* thread_pool_device, const Tensor<type, 1>& x, const Tensor<type, 1>& y)
{
#ifdef OPENNN_DEBUG

    ostringstream buffer;

    if(x.size() != y.size())
    {
        buffer << "OpenNN Exception: Vector Template.\n"
               << "RegressionResults "
                  "power_regression(const Tensor<type, 1>&) const "
                  "method.\n"
               << "Y size must be equal to X size.\n";

        throw logic_error(buffer.str());
    }

#endif

    // Check negative values from x and y

    RegressionResults power_regression;

    for(Index i = 0; i < x.dimension(0); i++)
    {
        if(!::isnan(x(i)) && x(i) <= 0)
        {
            power_regression.regression_type = Exponential;
            power_regression.correlation = NAN;

            return power_regression;
        }

        if(!::isnan(y(i)) && y(i) <= 0)
        {
            power_regression.regression_type = Exponential;
            power_regression.correlation = NAN;

            return power_regression;
        }
    }

    power_regression = linear_regression(thread_pool_device, x.log(), y.log(), false);

    power_regression.regression_type = Power;

    power_regression.a = exp(power_regression.a);

    return power_regression;
}


///Calculate the coefficients of a logistic regression (a, b) and the correlation among the variables
/// @param x Vector of the independent variable.
/// @param y Vector of the dependent variable.

RegressionResults logistic_regression(const ThreadPoolDevice* thread_pool_device, const Tensor<type, 1>& x, const Tensor<type, 1>& y)
{
#ifdef OPENNN_DEBUG

    ostringstream buffer;

    if(y.size() != x.size())
    {
        buffer << "OpenNN Exception: Correlations.\n"
               << "type logistic_regression(const Tensor<type, 1>&, const Tensor<type, 1>&) method.\n"
               << "Y size(" <<y.size()<<") must be equal to X size("<<x.size()<<").\n";

        throw logic_error(buffer.str());
    }

#endif

    // Filter missing values

    pair<Tensor<type, 1>, Tensor<type, 1>> filter_vectors = filter_missing_values(x,y);

    const Tensor<type, 1>& new_x = filter_vectors.first;
    const Tensor<type, 1>& new_y = filter_vectors.second;

    // Scale data

    const Tensor<type, 1> scaled_x = scale_minimum_maximum(new_x);

    // Inputs: scaled_x; Targets: sorted_y

    const Index input_variables_number = 1;
    const Index target_variables_number = 1;
    const Index samples_number = scaled_x.dimension(0);

    Tensor<type, 2> data(samples_number, input_variables_number+target_variables_number);

    for(Index j = 0; j < input_variables_number+target_variables_number; j++)
    {
        if(j < input_variables_number)
        {
            for(Index i = 0; i < samples_number; i++)
            {
                data(i,j) = scaled_x(i);
            }
        }
        else
        {
            for(Index i = 0; i < samples_number; i++)
            {
                data(i,j) = new_y(i);
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

    const Tensor<type, 1> coefficients = neural_network.get_parameters();

    // Regression results

    RegressionResults regression_results;

    regression_results.a = coefficients(0);
    regression_results.b = coefficients(1);

    const Tensor<type, 1> logistic_y = logistic(regression_results.a,regression_results.b, scaled_x);

    regression_results.correlation = linear_correlation(thread_pool_device, logistic_y, new_y, false);

    if(regression_results.b < 0) regression_results.correlation *= (-1);

    regression_results.regression_type = Logistic;

    return regression_results;
}


///Calculate the linear correlation between two variables.
/// @param x Vector of the independent variable.
/// @param y Vector of the dependent variable.

CorrelationResults linear_correlations(const ThreadPoolDevice*, const Tensor<type, 1>& x, const Tensor<type, 1>& y)
{
    Index n = y.size();

#ifdef OPENNN_DEBUG

    const Index x_size = x.size();

    ostringstream buffer;

    if(x_size != n)
    {
        buffer << "OpenNN Exception: Vector Template.\n"
               << "RegressionResults linear_correlations(const Tensor<type, 1>&) const method.\n"
               << "Y size must be equal to X size.\n";

        throw logic_error(buffer.str());
    }

#endif

    pair<Tensor<type, 1>, Tensor<type, 1>> filter_vectors = filter_missing_values(x,y);

    const Tensor<type, 1> new_vector_x = scale_minimum_maximum(filter_vectors.first);
    const Tensor<type, 1> new_vector_y = scale_minimum_maximum(filter_vectors.second);

    n = new_vector_x.size();

    Tensor<type, 0> s_x = new_vector_x.sum();
    Tensor<type, 0> s_y = new_vector_y.sum();
    Tensor<type, 0> s_xx = (new_vector_x*new_vector_x).sum();
    Tensor<type, 0> s_yy = (new_vector_y*new_vector_y).sum();
    Tensor<type, 0> s_xy = (new_vector_x*new_vector_y).sum();

    CorrelationResults linear_correlations;

    linear_correlations.correlation_type = Linear_correlation;

    if(abs(s_x()) < numeric_limits<type>::min() && abs(s_y()) < numeric_limits<type>::min()
            && abs(s_xx()) < numeric_limits<type>::min() && abs(s_yy()) < numeric_limits<type>::min()
            && abs(s_xy()) < numeric_limits<type>::min())
    {
        linear_correlations.correlation = 1.0;
    }
    else
    {
        if(sqrt((n * s_xx() - s_x() * s_x()) *(n * s_yy() - s_y() * s_y())) < numeric_limits<type>::min())
        {
            linear_correlations.correlation = 1.0;
        }
        else
        {
            linear_correlations.correlation =
                (n * s_xy() - s_x() * s_y()) /
                sqrt((n * s_xx() - s_x() * s_x()) *(n * s_yy() - s_y() * s_y()));
        }
    }

    return linear_correlations;
}


///Calculate the logarithmic correlation between two variables.
/// @param x Vector of the independent variable.
/// @param y Vector of the dependent variable.

CorrelationResults logarithmic_correlations(const ThreadPoolDevice* thread_pool_device,
                                            const Tensor<type, 1>& x, const Tensor<type, 1>& y)
{
#ifdef OPENNN_DEBUG

    const Index x_size = x.size();

    ostringstream buffer;

    if(x_size != y.size())
    {
        buffer << "OpenNN Exception: Vector Template.\n"
               << "RegressionResults "
               "logarithmic_correlations(const Tensor<type, 1>&) const "
               "method.\n"
               << "Y size must be equal to X size.\n";

        throw logic_error(buffer.str());
    }

#endif

    CorrelationResults logarithmic_correlation;

    logarithmic_correlation.correlation = OpenNN::logarithmic_correlation(thread_pool_device, x, y);

    logarithmic_correlation.correlation_type = Logarithmic_correlation;

    return logarithmic_correlation;
}


///Calculate the exponential correlation between two variables.
/// @param x Vector of the independent variable.
/// @param y Vector of the dependent variable.

CorrelationResults exponential_correlations(const ThreadPoolDevice* thread_pool_device,
                                            const Tensor<type, 1>& x, const Tensor<type, 1>& y)
{

#ifdef OPENNN_DEBUG
    Index n = y.size();

    const Index x_size = x.size();

    ostringstream buffer;

    if(x_size != n)
    {
        buffer << "OpenNN Exception: Vector Template.\n"
               << "RegressionResults "
               "exponential_correlations(const Tensor<type, 1>&) const "
               "method.\n"
               << "Y size must be equal to X size.\n";

        throw logic_error(buffer.str());
    }

#endif

    CorrelationResults exponential_correlation;

    exponential_correlation.correlation = OpenNN::exponential_correlation(thread_pool_device, x, y);

    exponential_correlation.correlation_type = Exponential_correlation;

    return exponential_correlation;
}


///Calculate the power correlation between two variables.
/// @param x Vector of the independent variable.
/// @param y Vector of the dependent variable.

CorrelationResults power_correlations(const ThreadPoolDevice* thread_pool_device,
                                      const Tensor<type, 1>& x, const Tensor<type, 1>& y)
{
#ifdef OPENNN_DEBUG

    Index n = y.size();

    const Index x_size = x.size();

    ostringstream buffer;

    if(x_size != n)
    {
        buffer << "OpenNN Exception: Vector Template.\n"
               << "RegressionResults "
               "power_correlations(const Tensor<type, 1>&) const "
               "method.\n"
               << "Y size must be equal to X size.\n";

        throw logic_error(buffer.str());
    }

#endif

    CorrelationResults power_correlation;

    power_correlation.correlation = OpenNN::power_correlation(thread_pool_device, x, y);

    power_correlation.correlation_type = Power_correlation;

    return power_correlation;
}


///Calculate the logistic correlation between two variables.
/// @param x Vector of the independent variable.
/// @param y Vector of the dependent variable.

CorrelationResults logistic_correlations(const ThreadPoolDevice* thread_pool_device,
                                         const Tensor<type, 1>& x, const Tensor<type, 1>& y)
{
#ifdef OPENNN_DEBUG

    ostringstream buffer;

    if(y.size() != x.size())
    {
        buffer << "OpenNN Exception: Correlations.\n"
               << "type logistic_correlations(const Tensor<type, 1>&, const Tensor<type, 1>&) method.\n"
               << "Y size(" <<y.size()<<") must be equal to X size("<<x.size()<<").\n";

        throw logic_error(buffer.str());
    }

#endif

    // Filter missing values

    pair<Tensor<type, 1>, Tensor<type, 1>> filter_vectors = filter_missing_values(x,y);

    const Tensor<type, 1>& new_x = filter_vectors.first;
    Tensor<type, 1>& new_y = filter_vectors.second;

    // Scale data

    Tensor<type, 1> scaled_x = scale_minimum_maximum(new_x);

    // Check ideal correlation

    vector<int> sorted_index = get_indices_sorted(scaled_x);

    Tensor<type,1> y_sorted(y.dimension(0));

    for(Index i = 0; i < scaled_x.dimension(0); i++)
    {
        y_sorted(i) = new_y(sorted_index[i]);
    }

    Index counter = 0;

    for(Index i=0; i< scaled_x.dimension(0)-1; i++)
    {
        if((y_sorted(i) - y_sorted(i+1)) > numeric_limits<type>::min())
        {
            counter++;
        }
    }

    if(counter == 1 && (new_y(sorted_index[0]) - 0) < numeric_limits<type>::min())
    {
        CorrelationResults logistic_correlations;

        logistic_correlations.correlation = 1;

        logistic_correlations.correlation_type = Logistic_correlation;

        return logistic_correlations;
    }

    if(counter == 1 && (new_y(sorted_index[0]) - 1) < numeric_limits<type>::min())
    {
        CorrelationResults logistic_correlations;

        logistic_correlations.correlation = -1;

        logistic_correlations.correlation_type = Logistic_correlation;

        return logistic_correlations;
    }

    // Inputs: scaled_x; Targets: sorted_y

    const Index input_variables_number = 1;
    const Index target_variables_number = 1;
    const Index samples_number = scaled_x.dimension(0);

    Tensor<type, 2> data(samples_number, input_variables_number+target_variables_number);

    for(Index j = 0; j <input_variables_number+target_variables_number; j++)
    {
        if(j < input_variables_number)
        {
            for(Index i = 0; i < samples_number; i++)
            {
                data(i,j) = scaled_x(i);
            }
        }
        else
        {
            for(Index i = 0; i < samples_number; i++)
            {
                data(i,j) = new_y(i);
            }
        }
    }

    DataSet data_set(data);
    data_set.set_training();

    Tensor<Index, 1> architecture(2);
    architecture.setValues({1, 1});

    NeuralNetwork neural_network(NeuralNetwork::Classification, architecture);

    neural_network.set_parameters_random();

    NormalizedSquaredError normalized_squared_error(&neural_network, &data_set);
    normalized_squared_error.set_normalization_coefficient();
    normalized_squared_error.set_regularization_method("L2_NORM");
    normalized_squared_error.set_regularization_weight(static_cast<type>(0.01));

    LevenbergMarquardtAlgorithm levenberg_marquardt_algorithm(&normalized_squared_error);
    levenberg_marquardt_algorithm.set_display(false);
    levenberg_marquardt_algorithm.perform_training();

    // Logistic correlation

    CorrelationResults logistic_correlations;

    const Tensor<type, 1> coefficients = neural_network.get_parameters();

    const Tensor<type, 1> logistic_y = logistic(coefficients(0), coefficients(1), scaled_x);

    logistic_correlations.correlation = linear_correlation(thread_pool_device, logistic_y, new_y, false);

    if(coefficients(1) < 0) logistic_correlations.correlation *= (-1);

    logistic_correlations.correlation_type = Logistic_correlation;

    return logistic_correlations;    
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


CorrelationResults multiple_logistic_correlations(const ThreadPoolDevice* thread_pool_device,
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

    // Inputs: scaled_x; Targets: sorted_y

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

    CorrelationResults logistic_correlations;

    const Tensor<type, 1> bias = perceptron_layer->get_biases().chip(0,1);
    const Tensor<type, 2> weights = perceptron_layer->get_synaptic_weights();

    const Tensor<type, 2> logistic_y = logistic(thread_pool_device,bias, weights, scaled_x);

    logistic_correlations.correlation = linear_correlation(thread_pool_device, logistic_y.chip(0,1), scaled_y.chip(0,1), false);

    logistic_correlations.correlation_type = Logistic_correlation;

    return logistic_correlations;

}


///Calculate the Karl Pearson correlation between two variables.
/// @param x Matrix of the variable X.
/// @param y Matrix of the variable Y.

CorrelationResults karl_pearson_correlations(const ThreadPoolDevice*, const Tensor<type, 2>& x, const Tensor<type, 2>& y)
{
#ifdef  OPENNN_DEBUG

    if(x.dimension(1) == 0)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: Correlation class."
               << "type karl_pearson_correlations(const Tensor<type, 2>&, const Tensor<type, 2>&) method.\n"
               << "Number of columns("<< x.dimension(1) <<") must be greater than zero.\n";

        throw logic_error(buffer.str());
    }

    if(y.dimension(1) == 0)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: Correlation class."
               << "type karl_pearson_correlations(const Tensor<type, 2>&, const Tensor<type, 2>&) method.\n"
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
        if(isnan(x(i, 0)))
        {
            x_NAN++;
        }
        if(isnan(x(i, 0)))
        {
            y_NAN++;
        }
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

    CorrelationResults karl_pearson;

    karl_pearson.correlation_type = KarlPearson_correlation;

    const Tensor<type, 0> contingency_table_sum = contingency_table.cast<type>().sum();

    karl_pearson.correlation
            = sqrt(static_cast<type>(k) / static_cast<type>(k - 1.0)) * sqrt(chi_squared/(chi_squared + contingency_table_sum(0)));

    return karl_pearson;
}


CorrelationResults gauss_correlations(const ThreadPoolDevice* thread_pool_device, const Tensor<type, 1>& x, const Tensor<type, 1>& y)
{
#ifdef OPENNN_DEBUG

    ostringstream buffer;

    if(y.size() != x.size())
    {
        buffer << "OpenNN Exception: Correlations.\n"
               << "type gauss_correlations(const Tensor<type, 1>&, const Tensor<type, 1>&) method.\n"
               << "Y size(" <<y.size()<<") must be equal to X size("<<x.size()<<").\n";

        throw logic_error(buffer.str());
    }

#endif

    // Filter missing values

    pair<Tensor<type, 1>, Tensor<type, 1>> filter_vectors = filter_missing_values(x,y);

    const Tensor<type, 1>& new_x = filter_vectors.first;
    const Tensor<type, 1>& new_y = filter_vectors.second;

    const Index new_size = new_x.size();

    // Scale data

    Tensor<type, 1> scaled_x = scale_minimum_maximum(new_x);

    // Calculate coefficients

    Tensor<type, 1> coefficients(2);
    Tensor<type, 1> last_coefficients(2);
    coefficients.setRandom();
    last_coefficients.setConstant(0.0);

    const Index epochs_number = 10000;
    const type step_size = static_cast<type>(-0.01);

    const type error_goal = static_cast<type>(1.0e-3);
    const type gradient_norm_goal = 0;

    Tensor<type, 0> mean_squared_error;
    Tensor<type, 0> gradient_norm;

    Tensor<type, 1> gradient(2);

    Tensor<type, 1> combination(new_size);
    Tensor<type, 1> error(new_size);

    for(Index i = 0; i < epochs_number; i++)
    {

        combination.device(*thread_pool_device) = static_cast<type>(-0.5)*
                                                  ((scaled_x - coefficients(0))/ coefficients(1))*
                                                  ((scaled_x - coefficients(0))/ coefficients(1));

        combination.device(*thread_pool_device) = combination.exp();

        error.device(*thread_pool_device) = combination - new_y;

        mean_squared_error.device(*thread_pool_device) = error.square().sum();

        if(mean_squared_error() < error_goal) break;

        Tensor<type, 0> sum_a;
        sum_a.device(*thread_pool_device) = ((2*error*combination*(scaled_x-coefficients(0)))/
                (coefficients(1)*coefficients(1))).sum();

        Tensor<type, 0> sum_b;
        sum_b.device(*thread_pool_device) = ((2*error*combination*(scaled_x-coefficients(0))*(scaled_x-coefficients(0)))/
                (coefficients(1)*coefficients(1)*coefficients(1))).sum();


        gradient(0) = sum_a();
        gradient(1) = sum_b();

        gradient_norm = gradient.square().sum().sqrt();

        if(gradient_norm() < gradient_norm_goal) break;

        coefficients += gradient*step_size;

    }

    // Gaussian correlation

    CorrelationResults gaussian_correlations;

    const Tensor<type, 1> gaussian_y = gaussian(coefficients(0), coefficients(1), scaled_x);

    gaussian_correlations.correlation = abs(linear_correlation(thread_pool_device, gaussian_y, new_y, false));

    gaussian_correlations.correlation_type = Gauss_correlation;

    return gaussian_correlations;
}



/// Returns the covariance of this vector and other vector
/// @param vector_1 data
/// @param vector_2 data

type covariance(const Tensor<type, 1>& vector_1, const Tensor<type, 1>& vector_2)
{
    const Index size_1 = vector_1.size();
    const Index size_2 = vector_2.size();

#ifdef OPENNN_DEBUG

    if(size_1 == 0)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: Vector Template.\n"
               << "type covariance(const Tensor<type, 1>&) const method.\n"
               << "Size must be greater than zero.\n";

        throw logic_error(buffer.str());
    }

    if(size_1 != size_2)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: Vector Template.\n"
               << "type covariance(const Tensor<type, 1>&) const method.\n"
               << "Size of this vectro must be equal to size of other vector.\n";

        throw logic_error(buffer.str());
    }

#endif

    if(size_1 == 1)
    {
        return 0.0;
    }

    const Tensor<type, 0> mean_1 = vector_1.mean();
    const Tensor<type, 0> mean_2 = vector_2.mean();

    type numerator = 0;
    type denominator = static_cast<type>(size_2-1);

    for(Index i = 0; i < size_1; i++)
    {
        numerator += (vector_1(i) - mean_1(0))*(vector_2(i)-mean_2(0));
    }

    return numerator/denominator;
}


/// Returns the covariance of this vector and other vector
/// @param vector_1 data
/// @param vector_2 data

type covariance_missing_values(const Tensor<type, 1>& x, const Tensor<type, 1>& y)
{
    pair<Tensor<type, 1>, Tensor<type, 1>> filter_vectors = filter_missing_values(x,y);

    const Tensor<type, 1> new_x = filter_vectors.first;
    const Tensor<type, 1> new_y = filter_vectors.second;

    const Index size_1 = new_x.size();
    const Index size_2 = new_y.size();

#ifdef OPENNN_DEBUG

    if(size_1 == 0)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: Vector Template.\n"
               << "type covariance(const Tensor<type, 1>&) const method.\n"
               << "Size must be greater than zero.\n";

        throw logic_error(buffer.str());
    }

    if(size_1 != size_2)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: Vector Template.\n"
               << "type covariance(const Tensor<type, 1>&) const method.\n"
               << "Size of this vectro must be equal to size of other vector.\n";

        throw logic_error(buffer.str());
    }

#endif

    if(size_1 == 1)
    {
        return 0.0;
    }

    const Tensor<type, 0> mean_1 = new_x.mean();
    const Tensor<type, 0> mean_2 = new_y.mean();

    type numerator = 0;
    type denominator = static_cast<type>(size_2-1);

    for(Index i = 0; i < size_1; i++)
    {
        numerator += (new_x(i) - mean_1(0))*(new_y(i)-mean_2(0));
    }

    return numerator/denominator;
}


/// Retruns the covariance matrix of this matrix.
/// The number of columns and rows of the matrix is equal to the number of columns of this matrix.
/// Covariance matrix is symmetric

Tensor<type, 2> covariance_matrix(const Tensor<type, 2>& matrix)
{
    const Index columns_number = matrix.dimension(1);

    const Index size = columns_number;

#ifdef OPENNN_DEBUG

    if(size == 0)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: Matrix template."
               << "void covariance_matrix() const method.\n"
               << "Number of columns("<< columns_number <<") must be greater than zero.\n";

        throw logic_error(buffer.str());
    }

#endif

    Tensor<type, 2> covariance_matrix(size, size);

    Tensor<type, 1> first_column;
    Tensor<type, 1> second_column;

    for(Index i = 0; i < size; i++)
    {
        first_column = matrix.chip(i,1);

        for(Index j = i; j < size; j++)
        {
            second_column = matrix.chip(j,1);

            covariance_matrix(i,j) = covariance(first_column, second_column);
            covariance_matrix(j,i) = covariance_matrix(i,j);
        }
    }

    return covariance_matrix;
}


/// Returns a vector with the rank of the elements of this vector.
/// The smallest element will have rank 1, and the greatest element will have
/// size.
/// That is, small values correspond with small ranks.
/// Ties are assigned the mean of the ranks.
/// @todo

Tensor<type, 1> less_rank_with_ties(const Tensor<type, 1>& vector)
{
//    Tensor<type, 1> indices_this = vector.calculate_less_rank().cast<type>();

//    const Tensor<type, 1> this_unique = vector.get_unique_elements();
//  const Tensor<Index, 1> this_unique_frecuency = vector.count_unique();

    const Index n = vector.size();

//    for(Index  i = 0; i < static_cast<Index>(this_unique.size()); i++)
    {
//        if(this_unique_frecuency[i] > 1)
        {
//             const type unique = this_unique[i];

//             Tensor<type, 1> indices(this_unique_frecuency[i]);

             for(Index j = 0; j < n; j++)
             {
//                 if(abs(vector[j] - unique) < numeric_limits<type>::min())
                 {
//                     indices.push_back(indices_this[j]);
                 }
             }

//             const Tensor<type, 0> mean_index = indices.mean();

             for(Index j = 0; j < n; j++)
             {
//                 if(abs(vector[j] - unique) < numeric_limits<type>::min())
                 {
//                     indices_this[j] = mean_index(0);
                 }
             }
        }
    }

//    return indices_this + 1;

    return Tensor<type, 1>();
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

    const Tensor<type, 2> squared = ((matrix.cast<type>()-ei)*(matrix.cast<type>()-ei));

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


///Filter the missing values of two vectors

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
