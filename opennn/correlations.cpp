//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   C O R R E L A T I O N S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "correlations.h"

namespace OpenNN
{


/// Calculates the linear correlation coefficient(Spearman method)(R-value) between two vectors.
/// @param x Vector containing data.
/// @param y Vector for computing the linear correlation with the x vector.

double linear_correlation(const Vector<double>& x, const Vector<double>& y)
{
  const size_t n = x.size();

  if(x.is_constant() || y.is_constant()) return 1.0;

  #ifdef __OPENNN_DEBUG__

    const size_t y_size = y.size();

    ostringstream buffer;

    if(y_size != n)
    {
      buffer << "OpenNN Exception: Correlations.\n"
             << "static double linear_correlation(const Vector<double>&, const Vector<double>&) method.\n"
             << "Y size must be equal to X size.\n";

      throw logic_error(buffer.str());
    }

#endif

  double s_x = 0.0;
  double s_y = 0.0;

  double s_xx = 0.0;
  double s_yy = 0.0;

  double s_xy = 0.0;

  for(size_t i = 0; i < n; i++)
  {
    s_x += x[i];
    s_y += y[i];

    s_xx += x[i] * x[i];
    s_yy += y[i] * y[i];

    s_xy += y[i] * x[i];
  }

  double linear_correlation;

  if(abs(s_x - 0) < numeric_limits<double>::epsilon() && abs(s_y - 0) < numeric_limits<double>::epsilon() && abs(s_xx - 0) < numeric_limits<double>::epsilon()
  && abs(s_yy - 0) < numeric_limits<double>::epsilon() && abs(s_xy - 0) < numeric_limits<double>::epsilon())
  {
    linear_correlation = 1.0;
  }
  else
  {
    const double numerator = (n * s_xy - s_x * s_y);

    const double radicand = (n * s_xx - s_x * s_x) *(n * s_yy - s_y * s_y);

    if(radicand <= 0.0)
    {
      return 1;
    }

    const double denominator = sqrt(radicand);

    if(denominator < numeric_limits<double>::epsilon())
    {
      linear_correlation = 0.0;
    }
    else
    {
      linear_correlation = numerator / denominator;
    }
  }

  return linear_correlation;
}


/// Calculates the linear correlation coefficient(R-value) between two
/// when there are missing values in the data.
/// @param x Vector containing data.
/// @param y Vector for computing the linear correlation with this vector.

double linear_correlation_missing_values(const Vector<double>& x, const Vector<double>& y)
{
    pair <Vector<double>, Vector<double>> filter_vectors = filter_missing_values(x,y);

    const Vector<double> new_x = filter_vectors.first;
    const Vector<double> new_y = filter_vectors.second;

    return linear_correlation(new_x, new_y);
}


/// Calculates the Rank-Order correlation coefficient(Spearman method) between two vectors.
/// @param x Vector containing data.
/// @param y Vector for computing the linear correlation with the x vector.

double rank_linear_correlation(const Vector<double>& x, const Vector<double>& y)
{
    #ifdef __OPENNN_DEBUG__

    ostringstream buffer;

    if(y.size() != x.size()) {
      buffer << "OpenNN Exception: Correlations.\n"
             << "static double rank_linear_correlation(const Vector<double>&, const Vector<double>&) method.\n"
             << "Y size must be equal to X size.\n";

      throw logic_error(buffer.str());
    }

    #endif

    if(x.is_constant() || y.is_constant()) return 1;

    const Vector<double> ranks_x = less_rank_with_ties(x);
    const Vector<double> ranks_y = less_rank_with_ties(y);

    return linear_correlation(ranks_x, ranks_y);
}


/// Calculates the Rank-Order correlation coefficient(Spearman method)(R-value) between two vectors.
/// Takes into account possible missing values.
/// @param x Vector containing input values.
/// @param y Vector for computing the linear correlation with the x vector.
/// @param mising Vector with the missing instances idices.

double rank_linear_correlation_missing_values(const Vector<double>& x, const Vector<double>& y)
{
    pair <Vector<double>, Vector<double>> filter_vectors = filter_missing_values(x,y);

    const Vector<double> new_x = filter_vectors.first;
    const Vector<double> new_y = filter_vectors.second;

    return rank_linear_correlation(new_x, new_y);
}


/// Calculates the correlation between Y and exp(A*X+B).
/// @param x Vector containing the input values.
/// @param y Vector containing the target values.

double exponential_correlation(const Vector<double>& x, const Vector<double>& y)
{
  #ifdef __OPENNN_DEBUG__

    ostringstream buffer;

      if(y.size() != x.size()) {
        buffer << "OpenNN Exception: Correlations.\n"
               << "static double exponential_correlation(const Vector<double>&, const Vector<double>&) method.\n"
               << "Y size must be equal to X size.\n";

        throw logic_error(buffer.str());
      }

#endif

    const Vector<size_t> negative_indices = y.get_indices_less_equal_to(0.0);
    const Vector<double> y_valid = y.delete_indices(negative_indices);

    Vector<double> log_y(y_valid.size());

    for(int i = 0; i < static_cast<int>(log_y.size()); i++)
    {
        log_y[static_cast<size_t>(i)] = log(y_valid[static_cast<size_t>(i)]);
    }

    return linear_correlation(x.delete_indices(negative_indices), log_y);
}


/// Calculates the correlation between Y and exp(A*X + B).
/// @param x Vector containing the input values.
/// @param y Vector containing the target values.
/// @param missing Vector with the missing instances indices.

double exponential_correlation_missing_values(const Vector<double>& x, const Vector<double>& y)
{
    #ifdef __OPENNN_DEBUG__

        ostringstream buffer;

        if(y.size() != x.size())
        {
          buffer << "OpenNN Exception: Correlations.\n"
                 << "static double logistic_correlation(const Vector<double>&, const Vector<double>&) method.\n"
                 << "Size of Y (" << y.size() << ") must be equal to size of X (" << x.size() << ").\n";

          throw logic_error(buffer.str());
        }

    #endif

    pair <Vector<double>, Vector<double>> filter_vector = filter_missing_values(x,y);

    const Vector<double> new_x = filter_vector.first;
    const Vector<double> new_y = filter_vector.second;

    return exponential_correlation(new_x, new_y);
}


/// Calculates the correlation between Y and ln(A*X+B).
/// @param x Vector containing the input values.
/// @param y Vector containing the target values.

double logarithmic_correlation(const Vector<double>& x, const Vector<double>& y)
{
#ifdef __OPENNN_DEBUG__

    ostringstream buffer;

     if(y.size() != x.size()) {
       buffer << "OpenNN Exception: Correlations.\n"
              << "static double logarithmic_correlation(const Vector<double>&, const Vector<double>&) method.\n"
              << "Y size must be equal to X size.\n";

       throw logic_error(buffer.str());
     }

#endif

    return linear_correlation(logarithm(x), y);
}


/// Calculates the correlation between Y and ln(A*X+B).
/// Takes into account possible missing values.
/// @param x Vector containing the input values.
/// @param y Vector containing the target values.
/// @param missing Vector with the missing instances indices.

double logarithmic_correlation_missing_values(const Vector<double>& x, const Vector<double>& y)
    {
    #ifdef __OPENNN_DEBUG__

        ostringstream buffer;

        if(y.size() != x.size())
        {
          buffer << "OpenNN Exception: Correlations.\n"
                 << "static double logarithmic_correlation_missing_values(const Vector<double>&, const Vector<double>&) method.\n"
                 << "Size of Y (" << y.size() << ") must be equal to size of X (" << x.size() << ").\n";

          throw logic_error(buffer.str());
        }

    #endif

        pair <Vector<double>, Vector<double>> filter_vectors = filter_missing_values(x,y);

        const Vector<double> new_x = filter_vectors.first;
        const Vector<double> new_y = filter_vectors.second;

    return logarithmic_correlation(new_x, new_y);
}


/// Calculates the logistic correlation coefficient between inputs and target.
/// It uses SGD method.
/// @param x Matrix containing inputs data.
/// @param y Vector for computing the linear correlation with the x vector.

double logistic_correlation(const Vector<double>& x, const Vector<double>& y)
{
#ifdef __OPENNN_DEBUG__

    ostringstream buffer;

    if(y.size() != x.size())
    {
      buffer << "OpenNN Exception: Correlations.\n"
             << "static double logistic_correlation(const Vector<double>&, const Vector<double>&) method.\n"
             << "Y size(" <<y.size()<<") must be equal to X size("<<x.size()<<").\n";

      throw logic_error(buffer.str());
    }

#endif

    Vector<double> scaled_x(x);
    scale_minimum_maximum(scaled_x);


    const size_t epochs_number = 50000;
    const double learning_rate = 0.01;
    const double momentum = 0.9;

    const double error_goal = 1.0e-8;
    const double gradient_norm_goal = 1.0e-8;

    double error;

    Vector<double> coefficients({0.0, 0.0});
    Vector<double> gradient(2, 0.0);
    Vector<double> increment(2, 0.0);

    Vector<double> last_increment(2, 0.0);

    for(size_t i = 0; i < epochs_number; i++)
    {
        error = logistic_error(coefficients[0], coefficients[1], x, y);

        gradient = logistic_error_gradient(coefficients[0], coefficients[1], x, y);

        increment = gradient*(-learning_rate);

        increment += last_increment*momentum;

        coefficients += increment;

        if(error < error_goal) break;

        if(l2_norm(gradient) < gradient_norm_goal) break;

        last_increment = increment;

    }

    RegressionResults regression_results;
    regression_results.regression_type = Logistic;
    regression_results.a = coefficients[0];
    regression_results.b = coefficients[1];
    regression_results.correlation = linear_correlation(logistic_function(x,regression_results.a,regression_results.b), y);

    return regression_results.correlation;
}


/// Calculates the logistic correlation coefficient between inputs and target.
/// It uses SGD method.
/// @param x Matrix containing inputs data.
/// @param y Vector for computing the linear correlation with the x vector.

double logistic_correlation_missing_values(const Vector<double>& x, const Vector<double>& y)
{
#ifdef __OPENNN_DEBUG__

    ostringstream buffer;

    if(y.size() != x.size())
    {
      buffer << "OpenNN Exception: Correlations.\n"
             << "static double logistic_correlation(const Vector<double>&, const Vector<double>&) method.\n"
             << "Size of Y (" << y.size() << ") must be equal to size of X (" << x.size() << ").\n";

      throw logic_error(buffer.str());
    }

#endif

    pair <Vector<double>, Vector<double>> filter_vectors = filter_missing_values(x,y);

    const Vector<double> new_x = filter_vectors.first;
    const Vector<double> new_y = filter_vectors.second;

    Vector<double> scaled_x(new_x);
    scale_minimum_maximum(scaled_x);


    const size_t epochs_number = 50000;
    const double learning_rate = 0.01;
    const double momentum = 0.9;

    const double error_goal = 1.0e-8;
    const double gradient_norm_goal = 1.0e-8;

    double error;

    Vector<double> coefficients({0.0, 0.0});
    Vector<double> gradient(2, 0.0);
    Vector<double> increment(2, 0.0);

    Vector<double> last_increment(2, 0.0);

    for(size_t i = 0; i < epochs_number; i++)
    {
        error = logistic_error(coefficients[0], coefficients[1], new_x, new_y);

        gradient = logistic_error_gradient(coefficients[0], coefficients[1], new_x, new_y);

        increment = gradient*(-learning_rate);

        increment += last_increment*momentum;

        coefficients += increment;

        if(error < error_goal) break;

        if(l2_norm(gradient) < gradient_norm_goal) break;

        last_increment = increment;

    }

    RegressionResults regression_results;
    regression_results.regression_type = Logistic;
    regression_results.a = coefficients[0];
    regression_results.b = coefficients[1];
    regression_results.correlation = linear_correlation(logistic_function(new_x,regression_results.a,regression_results.b), new_y);

    return regression_results.correlation;
}


/// Calculates the logistic correlation coefficient between two vectors.
/// It uses non-parametric Spearman Rank-Order method.
/// It uses a Neural Network to compute the logistic function approximation.
/// @param x Vector containing data.
/// @param y Vector for computing the linear correlation with the x vector.

double rank_logistic_correlation(const Vector<double>& x, const Vector<double>& y)
{
    const Vector<double> x_new = less_rank_with_ties(x);

    return logistic_correlation_missing_values(x_new, y);
}


///Calculate the power correlation between two variables.
/// @param x Vector of the independent variable
/// @param y Vector of the dependent variable

double power_correlation(const Vector<double>& x, const Vector<double>& y)
{
#ifdef __OPENNN_DEBUG__

    ostringstream buffer;

    if(y.size() != x.size())
    {
      buffer << "OpenNN Exception: Correlations.\n"
             << "static double logistic_correlation(const Vector<double>&, const Vector<double>&) method.\n"
             << "Size of Y (" << y.size() << ") must be equal to size of X (" << x.size() << ").\n";

      throw logic_error(buffer.str());
    }

#endif
    return linear_correlation(logarithm(x), logarithm(y));
}


///Calculate the power correlation between two variables.
/// @param x Vector of the independent variable
/// @param y Vector of the dependent variable

double power_correlation_missing_values(const Vector<double>&x, const Vector<double>&y)
{
#ifdef __OPENNN_DEBUG__

    ostringstream buffer;

    if(y.size() != x.size())
    {
      buffer << "OpenNN Exception: Correlations.\n"
             << "static double logistic_correlation(const Vector<double>&, const Vector<double>&) method.\n"
             << "Size of Y (" << y.size() << ") must be equal to size of X (" << x.size() << ").\n";

      throw logic_error(buffer.str());
    }

#endif
    pair <Vector<double>, Vector<double>> filter_vectors = filter_missing_values(x,y);

    const Vector<double> new_x = filter_vectors.first;
    const Vector<double> new_y = filter_vectors.second;

    return linear_correlation(logarithm(new_x), logarithm(new_y));
}


/// Calculates autocorrelation for a given number of maximum lags.
/// @param x Vector containing the data.
/// @param lags_number Maximum lags number.

Vector<double> autocorrelations(const Vector<double>& x, const size_t &lags_number)
{
  Vector<double> autocorrelation(lags_number);

  const double mean = OpenNN::mean(x);

  const size_t this_size = x.size();

  double numerator = 0;
  double denominator = 0;

  for(size_t i = 0; i < lags_number; i++)
  {
    for(size_t j = 0; j < this_size - i; j++)
    {
      numerator += ((x[j] - mean) *(x[j + i] - mean)) /static_cast<double>(this_size - i);
    }
    for(size_t j = 0; j < this_size; j++)
    {
      denominator += ((x[j] - mean) *(x[j] - mean)) /static_cast<double>(this_size);
    }

    if(denominator == 0.0)
    {
     autocorrelation[i] = 1.0;
    }
    else
    {
      autocorrelation[i] = numerator / denominator;
    }

    numerator = 0;
    denominator = 0;
  }

  return autocorrelation;
}


/// Calculates the cross-correlation between two vectors.
/// @param x Vector containing data.
/// @param y Vector for computing the linear correlation with this vector.
/// @param maximum_lags_number Maximum lags for which cross-correlation is calculated.

Vector<double> cross_correlations(const Vector<double>& x, const Vector<double>& y, const size_t &maximum_lags_number)
{
  if(y.size() != x.size())
  {
    ostringstream buffer;

    buffer << "OpenNN Exception: Correlations.\n"
           << "Vector<double calculate_cross_correlation(const "
           << "Vector<double>&) method.\n"
           << "Both vectors must have the same size.\n";

    throw logic_error(buffer.str());
  }

  Vector<double> cross_correlation(maximum_lags_number);

  const double this_mean = mean(x);
  const double y_mean = mean(y);

  const size_t this_size = x.size();

  double numerator = 0;

  double this_denominator = 0;
  double y_denominator = 0;
  double denominator = 0;

  for(size_t i = 0; i < maximum_lags_number; i++)
  {
    numerator = 0;
    this_denominator = 0;
    y_denominator = 0;

    for(size_t j = 0; j < this_size - i; j++)
    {
      numerator += (x[j] - this_mean) *(y[j + i] - y_mean);
    }

    for(size_t j = 0; j < this_size; j++)
    {
      this_denominator += (x[j] - this_mean) *(x[j] - this_mean);
      y_denominator += (y[j] - y_mean) *(y[j] - y_mean);
    }

    denominator = sqrt(this_denominator * y_denominator);

    if(denominator == 0.0) {
      cross_correlation[i] = 0.0;
    } else {
      cross_correlation[i] = numerator / denominator;
    }
  }

  return cross_correlation;
}


/// Returns a vector with the logistic error gradient.
/// @param coeffients.
/// @param x Independent data.
/// @param y Dependent data.

Vector<double> logistic_error_gradient(const double& a, const double& b, const Vector<double>& x, const Vector<double>& y)
{
   const size_t n = y.size();

#ifdef __OPENNN_DEBUG__

  const size_t x_size = x.size();

  ostringstream buffer;

  if(x_size != n) {
    buffer << "OpenNN Exception: double.\n"
           << "logistic error(const double&, const double&, const Vector<double>&, const Vector<double>& "
              "method.\n"
           << "Y size must be equal to X size.\n";

    throw logic_error(buffer.str());
  }

#endif

    Vector<double> error_gradient(2);

    double sum_a = 0.0;
    double sum_b = 0.0;

    double exponential;

    #pragma omp parallel for

    for(size_t i = 0; i < n; i ++)
    {
        exponential = exp(-a - b * x[i]);

        sum_a += (2 * exponential) /  (1 + exponential)/(1 + exponential)/(1 + exponential) - 2 * exponential * y[i] / (1 + exponential)/(1 +exponential);

        sum_b += (2 * exponential * x[i]) /  (1 + exponential)/(1 + exponential)/(1 + exponential) - 2 * exponential * x[i] * y[i] / (1 + exponential)/(1 +exponential);
    }

    error_gradient[0] = sum_a / n;
    error_gradient[1] = sum_b / n;

    return error_gradient;
}


///Calculate the gradient of the logistic error function.
/// @param a Coefficient a of the logistic function.
/// @param b Coefficient b of the logistic function.
/// @param x Input vector.
/// @param y Target vector.

Vector<double> logistic_error_gradient_missing_values(const double& a, const double& b, const Vector<double>& x, const Vector<double>& y)
{
   const size_t n = y.size();

#ifdef __OPENNN_DEBUG__

  const size_t x_size = x.size();

  ostringstream buffer;

  if(x_size != n) {
    buffer << "OpenNN Exception: double.\n"
           << "logistic error(const double&, const double&, const Vector<double>&, const Vector<double>& "
              "method.\n"
           << "Y size must be equal to X size.\n";

    throw logic_error(buffer.str());
  }

#endif

    pair <Vector<double>, Vector<double>> filter_vectors = filter_missing_values(x,y);

    const Vector<double> new_vector_x = filter_vectors.first;
    const Vector<double> new_vector_y = filter_vectors.second;

    Vector<double> f_x(new_vector_y.size());

    const double new_n = static_cast<double>(new_vector_x.size());

    Vector<double> error_gradient(2);

    double sum_a = 0.0;
    double sum_b = 0.0;

    double exponential;

    for(size_t i = 0; i < new_n; i ++)
    {
        exponential = exp(-a - b * new_vector_x[i]);

        sum_a += (2 * exponential) /  (1 + exponential)/(1 + exponential)/(1 + exponential) - 2 * exponential * new_vector_y[i] / (1 + exponential)/(1 +exponential);

        sum_b += (2 * exponential * new_vector_x[i]) /  (1 + exponential)/(1 + exponential)/(1 + exponential) - 2 * exponential * new_vector_x[i] * new_vector_y[i] / (1 + exponential)/(1 +exponential);
    }

    error_gradient[0] = sum_a / new_n;
    error_gradient[1] = sum_b / new_n;

    return error_gradient;
}


/// Calculate the logistic function with specifics parameters 'a' and 'b'.
/// @param a Parameter a.
/// @param b Parameter b.

double logistic(const double& a, const double& b, const double& x)
{   
    return 1.0/(1.0+exp(-(a+b*x)));
}


///Calculate the mean square error of the logistic function.

double logistic_error(const double& a, const double& b, const Vector<double>& x, const Vector<double>& y)
{
    const size_t n = y.size();

#ifdef __OPENNN_DEBUG__

  const size_t x_size = x.size();

  ostringstream buffer;

  if(x_size != n) {
    buffer << "OpenNN Exception: double.\n"
           << "logistic error(const double&, const double&, const Vector<double>&, const Vector<double>& "
              "method.\n"
           << "Y size must be equal to X size.\n";

    throw logic_error(buffer.str());
  }

#endif

    double error;

    double sum_squared_error = 0.0;

    for(size_t i = 0; i < x.size(); i ++)
    {
        error = logistic(a, b, x[i]) - y[i];

        sum_squared_error += error*error;
    }

    return sum_squared_error/n;
}


///Calculate the error of the logistic function when there are missing values.

double  logistic_error_missing_values
(const double& a, const double& b, const Vector<double>& x, const Vector<double>& y)
{
    const size_t n = y.size();

#ifdef __OPENNN_DEBUG__

  const size_t x_size = x.size();

  ostringstream buffer;

  if(x_size != n) {
    buffer << "OpenNN Exception: double.\n"
           << "logistic error(const double&, const double&, const Vector<double>&, const Vector<double>& "
              "method.\n"
           << "Y size must be equal to X size.\n";

    throw logic_error(buffer.str());
  }

#endif

    pair <Vector<double>, Vector<double>> filter_vectors = filter_missing_values(x,y);

    const Vector<double> new_vector_x = filter_vectors.first;
    const Vector<double> new_vector_y = filter_vectors.second;

    Vector<double> f_x(new_vector_y.size());

    const double new_n = static_cast<double>(new_vector_x.size());

    double difference = 0.0;

    double error = 0.0;

    for(size_t i = 0; i < new_n; i ++)
    {
        f_x[i] = logistic(a, b, new_vector_x[i]);

        difference += pow(f_x[i] - new_vector_y[i],2);
    }

    return error = difference/new_n;
}


///Calculate the coefficients of a linear regression (a, b) and the correlation among the variables even when there are missing values.
/// @param x Vector of the independent variable.
/// @param y Vector of the dependent variable.

RegressionResults linear_regression(const Vector<double>& x, const Vector<double>& y)
{
    const size_t n = y.size();

  #ifdef __OPENNN_DEBUG__

    const size_t x_size = x.size();

    ostringstream buffer;

    if(x_size != n) {
      buffer << "OpenNN Exception: Vector Template.\n"
             << "RegressionResults "
                "linear_regression(const Vector<T>&) const "
                "method.\n"
             << "Y size must be equal to X size.\n";

      throw logic_error(buffer.str());
    }

  #endif

    double s_x = 0;
    double s_y = 0;

    double s_xx = 0;
    double s_yy = 0;

    double s_xy = 0;

    for(size_t i = 0; i < n; i++) {
      s_x += x[i];
      s_y += y[i];

      s_xx += x[i] * x[i];
      s_yy += y[i] * y[i];

      s_xy += x[i] * y[i];
    }

    RegressionResults linear_regression;

    linear_regression.regression_type = Linear;

    if(s_x == 0.0 && s_y == 0.0 && s_xx == 0.0 && s_yy == 0.0 && s_xy == 0.0) {
      linear_regression.a = 0.0;

      linear_regression.b = 0.0;

      linear_regression.correlation = 1.0;
    } else {
      linear_regression.a =
         (s_y * s_xx - s_x * s_xy) /(n * s_xx - s_x * s_x);

      linear_regression.b =
         ((n * s_xy) - (s_x * s_y)) /((n * s_xx) - (s_x * s_x));

      if(sqrt((n * s_xx - s_x * s_x) *(n * s_yy - s_y * s_y)) < 1.0e-12)
      {
          linear_regression.correlation = 1.0;
      }
      else
      {
          linear_regression.correlation =
             (n * s_xy - s_x * s_y) /
              sqrt((n * s_xx - s_x * s_x) *(n * s_yy - s_y * s_y));
      }
    }

    return linear_regression;
}


///Calculate the coefficients of a linear regression (a, b) and the correlation among the variables
/// @param x Vector of the independent variable.
/// @param y Vector of the dependent variable.

RegressionResults linear_regression_missing_values(const Vector<double>& x, const Vector<double>& y)
{
   size_t n = y.size();

  #ifdef __OPENNN_DEBUG__

    const size_t x_size = x.size();

    ostringstream buffer;

    if(x_size != n) {
      buffer << "OpenNN Exception: Vector Template.\n"
             << "RegressionResults linear_regression(const Vector<T>&) const method.\n"
             << "Y size must be equal to X size.\n";

      throw logic_error(buffer.str());
    }

  #endif

    pair <Vector<double>, Vector<double>> filter_vectors = filter_missing_values(x,y);

    const Vector<double> new_vector_x = filter_vectors.first;
    const Vector<double> new_vector_y = filter_vectors.second;

    n = new_vector_x.size();

    double s_x = 0;
    double s_y = 0;

    double s_xx = 0;
    double s_yy = 0;

    double s_xy = 0;

    for(size_t i = 0; i < new_vector_x.size(); i++) {
      s_x += new_vector_x[i];
      s_y += new_vector_y[i];

      s_xx += new_vector_x[i] * new_vector_x[i];
      s_yy += new_vector_y[i] * new_vector_y[i];

      s_xy += new_vector_x[i] * new_vector_y[i];
    }

    RegressionResults linear_regression;

    linear_regression.regression_type = Linear;

    if(s_x == 0.0 && s_y == 0.0 && s_xx == 0.0 && s_yy == 0.0 && s_xy == 0.0)
    {
      linear_regression.a = 0.0;

      linear_regression.b = 0.0;

      linear_regression.correlation = 1.0;
    }
    else
    {
      linear_regression.a =
         (s_y * s_xx - s_x * s_xy) /(n * s_xx - s_x * s_x);

      linear_regression.b =
         ((n * s_xy) - (s_x * s_y)) /((n * s_xx) - (s_x * s_x));

      if(sqrt((n * s_xx - s_x * s_x) *(n * s_yy - s_y * s_y)) < 1.0e-12)
      {
          linear_regression.correlation = 1.0;
      }
      else
      {
          linear_regression.correlation =
             (n * s_xy - s_x * s_y) /
              sqrt((n * s_xx - s_x * s_x) *(n * s_yy - s_y * s_y));
      }
    }

    return linear_regression;
}


///Calculate the coefficients of a logarithmic regression (a, b) and the correlation among the variables
/// @param x Vector of the independent variable.
/// @param y Vector of the dependent variable.

RegressionResults logarithmic_regression_missing_values(const Vector<double>& x, const Vector<double>& y)
{
   size_t n = y.size();

  #ifdef __OPENNN_DEBUG__

    const size_t x_size = x.size();

    ostringstream buffer;

    if(x_size != n) {
      buffer << "OpenNN Exception: Vector Template.\n"
             << "RegressionResults "
                "logarithmic_regression(const Vector<T>&) const "
                "method.\n"
             << "Y size must be equal to X size.\n";

      throw logic_error(buffer.str());
    }

  #endif
    pair <Vector<double>, Vector<double>> filter_vectors = filter_missing_values(x,y);

    const Vector<double> new_vector_x = filter_vectors.first;
    const Vector<double> new_vector_y = filter_vectors.second;

    n = new_vector_x.size();

    double s_x = 0;
    double s_y = 0;

    double s_xx = 0;
    double s_yy = 0;

    double s_xy = 0;

    Vector<double> x1(new_vector_x.size());

    double y1 = 0;

    for(size_t i = 0; i < new_vector_x.size(); i++)
    {
      s_x += new_vector_x[i];
      s_y += new_vector_y[i];

      x1[i]= log(new_vector_x[i]);
      y1 += new_vector_y[i];
    }

     for(size_t i = 0; i < new_vector_x.size(); i++)
     {
         s_xx += pow((x1[i] - x1.calculate_sum() / new_vector_x.size()),2);

         s_yy += pow(new_vector_y[i] - y1 / new_vector_y.size(),2);

         s_xy += (x1[i] - x1.calculate_sum() / new_vector_x.size()) * (new_vector_y[i] - y1/new_vector_y.size());
     }

     RegressionResults logarithmic_regression;

     logarithmic_regression.regression_type = Logarithmic;

     if(s_x == 0.0 && s_y == 0.0 && s_xx == 0.0 && s_yy == 0.0 && s_xy == 0.0)
     {
       logarithmic_regression.a = 0.0;

       logarithmic_regression.b = 0.0;

       logarithmic_regression.correlation = 1.0;
     } else
     {
       logarithmic_regression.b = s_xy/s_xx;

       logarithmic_regression.a = new_vector_y.calculate_sum()/new_vector_y.size() - logarithmic_regression.b * x1.calculate_sum()/new_vector_x.size();

       logarithmic_regression.correlation = linear_correlation(logarithm(new_vector_x), new_vector_y);
    }

     return logarithmic_regression;
}


///Calculate the coefficients of a logarithmic regression (a, b) and the correlation among the variables
/// @param x Vector of the independent variable.
/// @param y Vector of the dependent variable.

RegressionResults logarithmic_regression(const Vector<double>& x, const Vector<double>& y)
{
    const size_t n = y.size();

  #ifdef __OPENNN_DEBUG__

    const size_t x_size = x.size();

    ostringstream buffer;

    if(x_size != n) {
      buffer << "OpenNN Exception: Vector Template.\n"
             << "RegressionResults "
                "logarithmic_regression(const Vector<T>&) const "
                "method.\n"
             << "Y size must be equal to X size.\n";

      throw logic_error(buffer.str());
    }

  #endif

    double s_x = 0;
    double s_y = 0;

    double s_xx = 0;
    double s_yy = 0;

    double s_xy = 0;

    Vector<double> x1(x.size());

    double y1 = 0;

    for(size_t i = 0; i < n; i++)
    {
      s_x += x[i];
      s_y += y[i];

      x1[i]= log(x[i]);
      y1 += y[i];
    }

     for(size_t i = 0; i < n; i++)
     {
         s_xx += pow((x1[i] - x1.calculate_sum()/x.size()),2);

         s_yy += pow(y[i] - y1/y.size(),2);

         s_xy += (x1[i] - x1.calculate_sum()/x.size())*(y[i] - y1/y.size());
     }

     RegressionResults logarithmic_regression;

     logarithmic_regression.regression_type = Logarithmic;

     if(s_x == 0.0 && s_y == 0.0 && s_xx == 0.0 && s_yy == 0.0 && s_xy == 0.0) {
       logarithmic_regression.a = 0.0;

       logarithmic_regression.b = 0.0;

       logarithmic_regression.correlation = 1.0;
     } else {

       logarithmic_regression.b = s_xy/s_xx;

       logarithmic_regression.a = y.calculate_sum()/y.size() - logarithmic_regression.b * x1.calculate_sum()/x.size();

       logarithmic_regression.correlation = linear_correlation(logarithm(x), y);
    }

     return logarithmic_regression;
}


///Calculate the coefficients of a exponential regression (a, b) and the correlation among the variables
/// @param x Vector of the independent variable.
/// @param y Vector of the dependent variable.

RegressionResults exponential_regression(const Vector<double>& x, const Vector<double>& y)
{
    const size_t n = y.size();

  #ifdef __OPENNN_DEBUG__

    const size_t x_size = x.size();

    ostringstream buffer;

    if(x_size != n) {
      buffer << "OpenNN Exception: Vector Template.\n"
             << "RegressionResults "
                "exponential_regression(const Vector<T>&) const "
                "method.\n"
             << "Y size must be equal to X size.\n";

      throw logic_error(buffer.str());
    }

  #endif

    double s_x = 0;
    double s_y = 0;

    double s_xx = 0;

    double s_xy = 0;

    for(size_t i = 0; i < n; i++) {
      s_x += x[i];

      s_y += log(y[i]);

      s_xx += x[i] * x[i];

      s_xy += x[i] * log(y[i]);
    }

    RegressionResults exponential_regression;

    exponential_regression.regression_type = Exponential;

    if(s_x == 0.0 && s_y == 0.0 && s_xx == 0.0 &&  s_xy == 0.0)
    {
      exponential_regression.a = 0.0;

      exponential_regression.b = 0.0;

      exponential_regression.correlation = 1.0;
     } else {
       exponential_regression.b =
         ((n * s_xy) - (s_x * s_y)) /((n * s_xx) - (s_x*s_x));
       exponential_regression.a =
          exp(s_y/y.size() - exponential_regression.b * s_x/x.size());



     const Vector<size_t> negative_indices = y.get_indices_less_equal_to(0.0);
     const Vector<double> y_valid = y.delete_indices(negative_indices);

     Vector<double> log_y(y_valid.size());

     for(int i = 0; i < static_cast<int>(log_y.size()); i++)
     {
         log_y[static_cast<size_t>(i)] = log(y_valid[static_cast<size_t>(i)]);
     }

     exponential_regression.correlation = exponential_correlation(x.delete_indices(negative_indices), log_y);

    }

    return exponential_regression;
}


///Calculate the coefficients of a exponential regression (a, b) and the correlation among the variables
/// @param x Vector of the independent variable.
/// @param y Vector of the dependent variable.

RegressionResults exponential_regression_missing_values(const Vector<double>& x, const Vector<double>& y)
{
    size_t n = y.size();

  #ifdef __OPENNN_DEBUG__

    const size_t x_size = x.size();

    ostringstream buffer;

    if(x_size != n) {
      buffer << "OpenNN Exception: Vector Template.\n"
             << "RegressionResults "
                "exponential_regression(const Vector<T>&) const "
                "method.\n"
             << "Y size must be equal to X size.\n";

      throw logic_error(buffer.str());
    }

  #endif

    pair <Vector<double>, Vector<double>> filter_vectors = filter_missing_values(x,y);

    const Vector<double> new_vector_x = filter_vectors.first;
    const Vector<double> new_vector_y = filter_vectors.second;

    const  double new_size = static_cast<double>(new_vector_x.size());

    double s_x = 0;
    double s_y = 0;

    double s_xx = 0;

    double s_xy = 0;

    for(size_t i = 0; i < new_size; i++)
    {
      s_x += new_vector_x[i];

      s_y += log(new_vector_y[i]);

      s_xx += new_vector_x[i] * new_vector_x[i];

      s_xy += new_vector_x[i] * log(new_vector_y[i]);
    }

    RegressionResults exponential_regression;

    exponential_regression.regression_type = Exponential;

    if(s_x == 0.0 && s_y == 0.0 && s_xx == 0.0 &&  s_xy == 0.0)
    {
      exponential_regression.a = 0.0;

      exponential_regression.b = 0.0;

      exponential_regression.correlation = 1.0;
     } else {
       exponential_regression.b =
         ((new_size * s_xy) - (s_x * s_y)) /((new_size * s_xx) - (s_x*s_x));
       exponential_regression.a =
          exp(s_y/new_size - exponential_regression.b * s_x/new_size);

     const Vector<size_t> negative_indices = new_vector_y.get_indices_less_equal_to(0.0);
     const Vector<double> y_valid = new_vector_y.delete_indices(negative_indices);

     Vector<double> log_y(y_valid.size());

     for(int i = 0; i < static_cast<int>(log_y.size()); i++)
     {
         log_y[static_cast<size_t>(i)] = log(y_valid[static_cast<size_t>(i)]);
     }

     exponential_regression.correlation = exponential_correlation(new_vector_x.delete_indices(negative_indices), log_y);

    }

    return exponential_regression;
}


///Calculate the coefficients of a power regression (a, b) and the correlation among the variables
/// @param x Vector of the independent variable.
/// @param y Vector of the dependent variable.

RegressionResults power_regression(const Vector<double>& x, const Vector<double>& y)
{
    const size_t n = y.size();

  #ifdef __OPENNN_DEBUG__

    const size_t x_size = x.size();

    ostringstream buffer;

    if(x_size != n) {
      buffer << "OpenNN Exception: Vector Template.\n"
             << "RegressionResults "
                "power_regression(const Vector<T>&) const "
                "method.\n"
             << "Y size must be equal to X size.\n";

      throw logic_error(buffer.str());
    }

  #endif

    double s_x = 0;
    double s_y = 0;

    double s_xx = 0;

    double s_xy = 0;

    for(size_t i = 0; i < n; i++) {
      s_x += log(x[i]);
      s_y += log(y[i]);

      s_xx += log(x[i]) * log(x[i]);

      s_xy += log(x[i]) * log(y[i]);
    }

    RegressionResults power_regression;

    power_regression.regression_type = Power;

    if(s_x == 0.0 && s_y == 0.0 && s_xx == 0.0 && s_xy == 0.0) {
      power_regression.a = 0.0;

      power_regression.b = 0.0;

      power_regression.correlation = 1.0;
    } else {

        power_regression.b =

                ((n * s_xy) - (s_x * s_y)) /((n * s_xx) - (s_x * s_x));

        power_regression.a =

                exp(s_y/y.size() - power_regression.b * s_x/x.size());

        power_regression.correlation = linear_correlation(logarithm(x), logarithm(y));
    }

    return power_regression;
}


///Calculate the coefficients of a power regression (a, b) and the correlation among the variables
/// @param x Vector of the independent variable.
/// @param y Vector of the dependent variable.

RegressionResults power_regression_missing_values(const Vector<double>& x, const Vector<double>& y)
{
    size_t n = y.size();

  #ifdef __OPENNN_DEBUG__

    const size_t x_size = x.size();

    ostringstream buffer;

    if(x_size != n) {
      buffer << "OpenNN Exception: Vector Template.\n"
             << "RegressionResults "
                "power_regression(const Vector<T>&) const "
                "method.\n"
             << "Y size must be equal to X size.\n";

      throw logic_error(buffer.str());
    }

  #endif

    pair <Vector<double>, Vector<double>> filter_vectors = filter_missing_values(x,y);

    const Vector<double> new_vector_x = filter_vectors.first;
    const Vector<double> new_vector_y = filter_vectors.second;

    n = new_vector_x.size();

    double s_x = 0;
    double s_y = 0;

    double s_xx = 0;

    double s_xy = 0;

    for(size_t i = 0; i < n; i++) {
      s_x += log(new_vector_x[i]);
      s_y += log(new_vector_y[i]);

      s_xx += log(new_vector_x[i]) * log(new_vector_x[i]);

      s_xy += log(new_vector_x[i]) * log(new_vector_y[i]);
    }

    RegressionResults power_regression;

    power_regression.regression_type = Power;

    if(s_x == 0.0 && s_y == 0.0 && s_xx == 0.0 && s_xy == 0.0) {
      power_regression.a = 0.0;

      power_regression.b = 0.0;

      power_regression.correlation = 1.0;
    } else {

        power_regression.b =

                ((n * s_xy) - (s_x * s_y)) /((n * s_xx) - (s_x * s_x));

        power_regression.a =

                exp(s_y/new_vector_y.size() - power_regression.b * s_x/new_vector_x.size());

        power_regression.correlation = power_regression.correlation = linear_correlation(logarithm(new_vector_x), logarithm(new_vector_y));

    }

    return power_regression;
}


///Calculate the coefficients of a logistic regression (a, b) and the correlation among the variables
/// @param x Vector of the independent variable.
/// @param y Vector of the dependent variable.

RegressionResults logistic_regression(const Vector<double>& x, const Vector<double>& y)
{
#ifdef __OPENNN_DEBUG__

    ostringstream buffer;

    if(y.size() != x.size())
    {
      buffer << "OpenNN Exception: Correlations.\n"
             << "static double logistic_correlation(const Vector<double>&, const Vector<double>&) method.\n"
             << "Y size(" <<y.size()<<") must be equal to X size("<<x.size()<<").\n";

      throw logic_error(buffer.str());
    }

#endif

    Vector<double> scaled_x(x);
    scale_minimum_maximum(scaled_x);

    Vector<double> coefficients({0.0, 0.0});

    const size_t epochs_number = 100000;
    double step_size = 0.01;

    const double error_goal = 1.0e-8;
    const double gradient_norm_goal = 1.0e-8;

    double error;

    Vector<double> gradient(2);

    for(size_t i = 0; i < epochs_number; i++)
    {
        error = logistic_error(coefficients[0], coefficients[1], x, y);

        if(error < error_goal) break;

        gradient = logistic_error_gradient(coefficients[0], coefficients[1], x, y);

        if(l2_norm(gradient) < gradient_norm_goal) break;

        coefficients -= gradient*step_size;

    }

    RegressionResults regression_results;
    regression_results.regression_type = Logistic;
    regression_results.a = coefficients[0];
    regression_results.b = coefficients[1];    
    regression_results.correlation = linear_correlation(logistic_function(x,regression_results.a,regression_results.b), y);

    return regression_results;
}


///Calculate the coefficients of a logistic regression (a, b) and the correlation among the variables
/// @param x Vector of the independent variable.
/// @param y Vector of the dependent variable.

RegressionResults logistic_regression_missing_values(const Vector<double>& x, const Vector<double>& y)
{
#ifdef __OPENNN_DEBUG__

    ostringstream buffer;

    if(y.size() != x.size())
    {
      buffer << "OpenNN Exception: Correlations.\n"
             << "static double logistic_correlation(const Vector<double>&, const Vector<double>&) method.\n"
             << "Y size(" <<y.size()<<") must be equal to X size("<<x.size()<<").\n";

      throw logic_error(buffer.str());
    }

#endif

    pair <Vector<double>, Vector<double>> filter_vectors = filter_missing_values(x,y);

    const Vector<double> new_vector_x = filter_vectors.first;
    const Vector<double> new_vector_y = filter_vectors.second;

    Vector<double> scaled_x(new_vector_x);
    scale_minimum_maximum(scaled_x);

    Vector<double> coefficients({0.0, 0.0});

    const size_t epochs_number = 100000;
    double step_size = 0.01;

    const double error_goal = 1.0e-8;
    const double gradient_norm_goal = 1.0e-8;

    double error;

    Vector<double> gradient(2);

    for(size_t i = 0; i < epochs_number; i++)
    {
        error = logistic_error(coefficients[0], coefficients[1], new_vector_x, new_vector_y);

        if(error < error_goal) break;

        gradient = logistic_error_gradient(coefficients[0], coefficients[1], new_vector_x, new_vector_y);

        if(l2_norm(gradient) < gradient_norm_goal) break;

        coefficients -= gradient*step_size;

    }

    RegressionResults regression_results;
    regression_results.regression_type = Logistic;
    regression_results.a = coefficients[0];
    regression_results.b = coefficients[1];
    regression_results.correlation = linear_correlation(logistic_function(new_vector_x,regression_results.a,regression_results.b), new_vector_y);

    return regression_results;
}


///Calculate the linear correlation between two variables.
/// @param x Vector of the independent variable.
/// @param y Vector of the dependent variable.

CorrelationResults linear_correlations(const Vector<double>& x, const Vector<double>& y)
{
    const size_t n = y.size();

  #ifdef __OPENNN_DEBUG__

    const size_t x_size = x.size();

    ostringstream buffer;

    if(x_size != n) {
      buffer << "OpenNN Exception: Vector Template.\n"
             << "RegressionResults "
                "linear_correlation(const Vector<T>&) const "
                "method.\n"
             << "Y size must be equal to X size.\n";

      throw logic_error(buffer.str());
    }

  #endif

    double s_x = 0;
    double s_y = 0;

    double s_xx = 0;
    double s_yy = 0;

    double s_xy = 0;

    for(size_t i = 0; i < n; i++) {
      s_x += x[i];
      s_y += y[i];

      s_xx += x[i] * x[i];
      s_yy += y[i] * y[i];

      s_xy += x[i] * y[i];
    }

    CorrelationResults linear_correlation;

    linear_correlation.correlation_type = Linear_correlation;

    if(s_x == 0.0 && s_y == 0.0 && s_xx == 0.0 && s_yy == 0.0 && s_xy == 0.0) {

      linear_correlation.correlation = 1.0;

    } else {

      if(sqrt((n * s_xx - s_x * s_x) *(n * s_yy - s_y * s_y)) < 1.0e-12)
      {
          linear_correlation.correlation = 1.0;
      }
      else
      {
          linear_correlation.correlation =
             (n * s_xy - s_x * s_y) /
              sqrt((n * s_xx - s_x * s_x) *(n * s_yy - s_y * s_y));
      }
    }

    return linear_correlation;
}


///Calculate the linear correlation between two variables whrn there are missing values.
/// @param x Vector of the independent variable.
/// @param y Vector of the dependent variable.

CorrelationResults linear_correlations_missing_values(const Vector<double>& x, const Vector<double>& y)
{
    size_t n = y.size();

   #ifdef __OPENNN_DEBUG__

     const size_t x_size = x.size();

     ostringstream buffer;

     if(x_size != n) {
       buffer << "OpenNN Exception: Vector Template.\n"
              << "RegressionResults linear_regression(const Vector<T>&) const method.\n"
              << "Y size must be equal to X size.\n";

       throw logic_error(buffer.str());
     }

   #endif

     pair <Vector<double>, Vector<double>> filter_vectors = filter_missing_values(x,y);

     const Vector<double> new_vector_x = filter_vectors.first;
     const Vector<double> new_vector_y = filter_vectors.second;

     n = new_vector_x.size();

     double s_x = 0;
     double s_y = 0;

     double s_xx = 0;
     double s_yy = 0;

     double s_xy = 0;

     for(size_t i = 0; i < new_vector_x.size(); i++) {
       s_x += new_vector_x[i];
       s_y += new_vector_y[i];

       s_xx += new_vector_x[i] * new_vector_x[i];
       s_yy += new_vector_y[i] * new_vector_y[i];

       s_xy += new_vector_x[i] * new_vector_y[i];
     }

     CorrelationResults linear_correlations;

     linear_correlations.correlation_type = Linear_correlation;

     if(s_x == 0.0 && s_y == 0.0 && s_xx == 0.0 && s_yy == 0.0 && s_xy == 0.0)
     {
       linear_correlations.correlation = 1.0;
     }
     else
     {
       if(sqrt((n * s_xx - s_x * s_x) *(n * s_yy - s_y * s_y)) < 1.0e-12)
       {
           linear_correlations.correlation = 1.0;
       }
       else
       {
           linear_correlations.correlation =
              (n * s_xy - s_x * s_y) /
               sqrt((n * s_xx - s_x * s_x) *(n * s_yy - s_y * s_y));
       }
     }

     return linear_correlations;
}


///Calculate the logarithmic correlation between two variables.
/// @param x Vector of the independent variable.
/// @param y Vector of the dependent variable.

CorrelationResults logarithmic_correlations(const Vector<double>& x, const Vector<double>& y)
{
    const size_t n = y.size();

  #ifdef __OPENNN_DEBUG__

    const size_t x_size = x.size();

    ostringstream buffer;

    if(x_size != n) {
      buffer << "OpenNN Exception: Vector Template.\n"
             << "RegressionResults "
                "logarithmic_regression(const Vector<T>&) const "
                "method.\n"
             << "Y size must be equal to X size.\n";

      throw logic_error(buffer.str());
    }

  #endif

    double s_x = 0;
    double s_y = 0;

    double s_xx = 0;
    double s_yy = 0;

    double s_xy = 0;

    Vector<double> x1(x.size());

    double y1 = 0;

    for(size_t i = 0; i < n; i++)
    {
      s_x += x[i];
      s_y += y[i];

      x1[i]= log(x[i]);
      y1 += y[i];
    }

     for(size_t i = 0; i < n; i++)
     {
         s_xx += pow((x1[i] - x1.calculate_sum()/x.size()),2);

         s_yy += pow(y[i] - y1/y.size(),2);

         s_xy += (x1[i] - x1.calculate_sum()/x.size())*(y[i] - y1/y.size());
     }

     CorrelationResults logarithmic_correlation;

     logarithmic_correlation.correlation_type = Logarithmic_correlation;

     if(s_x == 0.0 && s_y == 0.0 && s_xx == 0.0 && s_yy == 0.0 && s_xy == 0.0) {

       logarithmic_correlation.correlation = 1.0;

     } else {

       logarithmic_correlation.correlation = linear_correlation(logarithm(x), y);
    }

     return logarithmic_correlation;
}


///Calculate the logarithmic correlation between two variables when there are missing values.
/// @param x Vector of the independent variable.
/// @param y Vector of the dependent variable.

CorrelationResults logarithmic_correlations_missing_values(const Vector<double>& x, const Vector<double>& y)
{
   size_t n = y.size();

  #ifdef __OPENNN_DEBUG__

    const size_t x_size = x.size();

    ostringstream buffer;

    if(x_size != n) {
      buffer << "OpenNN Exception: Vector Template.\n"
             << "RegressionResults "
                "logarithmic_regression(const Vector<T>&) const "
                "method.\n"
             << "Y size must be equal to X size.\n";

      throw logic_error(buffer.str());
    }

  #endif
    pair <Vector<double>, Vector<double>> filter_vectors = filter_missing_values(x,y);

    const Vector<double> new_vector_x = filter_vectors.first;
    const Vector<double> new_vector_y = filter_vectors.second;

    n = new_vector_x.size();

    double s_x = 0;
    double s_y = 0;

    double s_xx = 0;
    double s_yy = 0;

    double s_xy = 0;

    Vector<double> x1(new_vector_x.size());

    double y1 = 0;

    for(size_t i = 0; i < new_vector_x.size(); i++)
    {
      s_x += new_vector_x[i];
      s_y += new_vector_y[i];

      x1[i]= log(new_vector_x[i]);
      y1 += new_vector_y[i];
    }

     for(size_t i = 0; i < new_vector_x.size(); i++)
     {
         s_xx += pow((x1[i] - x1.calculate_sum() / new_vector_x.size()),2);

         s_yy += pow(new_vector_y[i] - y1 / new_vector_y.size(),2);

         s_xy += (x1[i] - x1.calculate_sum() / new_vector_x.size()) * (new_vector_y[i] - y1/new_vector_y.size());
     }

     CorrelationResults logarithmic_correlation;

     logarithmic_correlation.correlation_type = Logarithmic_correlation;

     if(s_x == 0.0 && s_y == 0.0 && s_xx == 0.0 && s_yy == 0.0 && s_xy == 0.0)
     {

       logarithmic_correlation.correlation = 1.0;

     } else
     {

       logarithmic_correlation.correlation = linear_correlation_missing_values(logarithm(new_vector_x), new_vector_y);

     }

     return logarithmic_correlation;
}


///Calculate the exponential correlation between two variables.
/// @param x Vector of the independent variable.
/// @param y Vector of the dependent variable.

CorrelationResults exponential_correlations(const Vector<double>& x, const Vector<double>& y)
{
    const size_t n = y.size();

  #ifdef __OPENNN_DEBUG__

    const size_t x_size = x.size();

    ostringstream buffer;

    if(x_size != n) {
      buffer << "OpenNN Exception: Vector Template.\n"
             << "RegressionResults "
                "exponential_regression(const Vector<T>&) const "
                "method.\n"
             << "Y size must be equal to X size.\n";

      throw logic_error(buffer.str());
    }

  #endif

    double s_x = 0;
    double s_y = 0;

    double s_xx = 0;

    double s_xy = 0;

    for(size_t i = 0; i < n; i++) {
      s_x += x[i];

      s_y += log(y[i]);

      s_xx += x[i] * x[i];

      s_xy += x[i] * log(y[i]);
    }

    CorrelationResults exponential_correlation;

    exponential_correlation.correlation_type = Exponential_correlation;

    if(s_x == 0.0 && s_y == 0.0 && s_xx == 0.0 &&  s_xy == 0.0)
    {
        exponential_correlation.correlation = 1.0;

     } else {

        const Vector<size_t> negative_indices = y.get_indices_less_equal_to(0.0);

        const Vector<double> y_valid = y.delete_indices(negative_indices);

        Vector<double> log_y(y_valid.size());

        for(int i = 0; i < static_cast<int>(log_y.size()); i++)
        {
             log_y[static_cast<size_t>(i)] = log(y_valid[static_cast<size_t>(i)]);
        }

        exponential_correlation.correlation = OpenNN::linear_correlation(x.delete_indices(negative_indices), log_y);
    }

    return exponential_correlation;
}


///Calculate the exponential correlation between two variables when there are missing values.
/// @param x Vector of the independent variable.
/// @param y Vector of the dependent variable.

CorrelationResults exponential_correlations_missing_values(const Vector<double>& x, const Vector<double>& y)
{
    size_t n = y.size();

  #ifdef __OPENNN_DEBUG__

    const size_t x_size = x.size();

    ostringstream buffer;

    if(x_size != n) {
      buffer << "OpenNN Exception: Vector Template.\n"
             << "RegressionResults "
                "exponential_regression(const Vector<T>&) const "
                "method.\n"
             << "Y size must be equal to X size.\n";

      throw logic_error(buffer.str());
    }

  #endif

    pair <Vector<double>, Vector<double>> filter_vectors = filter_missing_values(x,y);

    const Vector<double> new_vector_x = filter_vectors.first;
    const Vector<double> new_vector_y = filter_vectors.second;

    const  double new_size = static_cast<double>(new_vector_x.size());

    double s_x = 0;
    double s_y = 0;

    double s_xx = 0;

    double s_xy = 0;

    for(size_t i = 0; i < new_size; i++)
    {
      s_x += new_vector_x[i];

      s_y += log(new_vector_y[i]);

      s_xx += new_vector_x[i] * new_vector_x[i];

      s_xy += new_vector_x[i] * log(new_vector_y[i]);
    }

    CorrelationResults exponential_correlation;

    exponential_correlation.correlation_type = Exponential_correlation;

    if(s_x == 0.0 && s_y == 0.0 && s_xx == 0.0 &&  s_xy == 0.0)
    {

      exponential_correlation.correlation = 1.0;

    } else {

     const Vector<size_t> negative_indices = new_vector_y.get_indices_less_equal_to(0.0);
     const Vector<double> y_valid = new_vector_y.delete_indices(negative_indices);

     Vector<double> log_y(y_valid.size());

     for(int i = 0; i < static_cast<int>(log_y.size()); i++)
     {
         log_y[static_cast<size_t>(i)] = log(y_valid[static_cast<size_t>(i)]);
     }

     exponential_correlation.correlation = exponential_correlation_missing_values(new_vector_x.delete_indices(negative_indices), log_y);

    }

    return exponential_correlation;
}


///Calculate the power correlation between two variables.
/// @param x Vector of the independent variable.
/// @param y Vector of the dependent variable.

CorrelationResults power_correlations(const Vector<double>& x, const Vector<double>& y)
{
    const size_t n = y.size();

  #ifdef __OPENNN_DEBUG__

    const size_t x_size = x.size();

    ostringstream buffer;

    if(x_size != n) {
      buffer << "OpenNN Exception: Vector Template.\n"
             << "RegressionResults "
                "power_regression(const Vector<T>&) const "
                "method.\n"
             << "Y size must be equal to X size.\n";

      throw logic_error(buffer.str());
    }

  #endif

    double s_x = 0;
    double s_y = 0;

    double s_xx = 0;

    double s_xy = 0;

    for(size_t i = 0; i < n; i++) {
      s_x += log(x[i]);
      s_y += log(y[i]);

      s_xx += log(x[i]) * log(x[i]);

      s_xy += log(x[i]) * log(y[i]);
    }

    CorrelationResults power_correlation;

    power_correlation.correlation_type = Power_correlation;

    if(s_x == 0.0 && s_y == 0.0 && s_xx == 0.0 && s_xy == 0.0)
    {
      power_correlation.correlation = 1.0;

    } else {

        power_correlation.correlation = linear_correlation(logarithm(x), logarithm(y));
    }

    return power_correlation;
}


///Calculate the power correlation between two variables when there are missing values.
/// @param x Vector of the independent variable.
/// @param y Vector of the dependent variable.

CorrelationResults power_correlations_missing_values(const Vector<double>& x, const Vector<double>& y)
{
    size_t n = y.size();

  #ifdef __OPENNN_DEBUG__

    const size_t x_size = x.size();

    ostringstream buffer;

    if(x_size != n) {
      buffer << "OpenNN Exception: Vector Template.\n"
             << "RegressionResults "
                "power_regression(const Vector<T>&) const "
                "method.\n"
             << "Y size must be equal to X size.\n";

      throw logic_error(buffer.str());
    }

  #endif

    pair <Vector<double>, Vector<double>> filter_vectors = filter_missing_values(x,y);

    const Vector<double> new_vector_x = filter_vectors.first;
    const Vector<double> new_vector_y = filter_vectors.second;

    n = new_vector_x.size();

    double s_x = 0;
    double s_y = 0;

    double s_xx = 0;

    double s_xy = 0;

    for(size_t i = 0; i < n; i++) {
      s_x += log(new_vector_x[i]);
      s_y += log(new_vector_y[i]);

      s_xx += log(new_vector_x[i]) * log(new_vector_x[i]);

      s_xy += log(new_vector_x[i]) * log(new_vector_y[i]);
    }

    CorrelationResults power_correlation;

    power_correlation.correlation_type = Power_correlation;

    if(s_x == 0.0 && s_y == 0.0 && s_xx == 0.0 && s_xy == 0.0)
    {
            power_correlation.correlation = 1.0;

    } else {

        power_correlation.correlation = linear_correlation_missing_values(logarithm(new_vector_x), logarithm(new_vector_y));
    }

    return power_correlation;
}


///Calculate the logistic correlation between two variables.
/// @param x Vector of the independent variable.
/// @param y Vector of the dependent variable.

CorrelationResults logistic_correlations(const Vector<double>& x, const Vector<double>& y)
{
#ifdef __OPENNN_DEBUG__

    ostringstream buffer;

    if(y.size() != x.size())
    {
      buffer << "OpenNN Exception: Correlations.\n"
             << "static double logistic_correlation(const Vector<double>&, const Vector<double>&) method.\n"
             << "Y size(" <<y.size()<<") must be equal to X size("<<x.size()<<").\n";

      throw logic_error(buffer.str());
    }

#endif

    Vector<double> scaled_x(x);
    scale_minimum_maximum(scaled_x);


    const size_t epochs_number = 50000;
    const double learning_rate = 0.01;
    const double momentum = 0.9;

    const double error_goal = 1.0e-8;
    const double gradient_norm_goal = 1.0e-8;

    double error;

    Vector<double> coefficients({0.0, 0.0});
    Vector<double> gradient(2, 0.0);
    Vector<double> increment(2, 0.0);

    Vector<double> last_increment(2, 0.0);

    for(size_t i = 0; i < epochs_number; i++)
    {               
        error = logistic_error(coefficients[0], coefficients[1], x, y);

        gradient = logistic_error_gradient(coefficients[0], coefficients[1], x, y);

        increment = gradient*(-learning_rate);

        increment += last_increment*momentum;

        coefficients += increment;

        if(error < error_goal) break;

        if(l2_norm(gradient) < gradient_norm_goal) break;

        last_increment = increment;

    }

    RegressionResults regression_results;
    regression_results.regression_type = Logistic;
    regression_results.a = coefficients[0];
    regression_results.b = coefficients[1];
    regression_results.correlation = linear_correlation(logistic_function(x,regression_results.a,regression_results.b), y);

    CorrelationResults correlation_results;

    correlation_results.correlation_type = Logistic_correlation;
    correlation_results.correlation = regression_results.correlation;

    return correlation_results;
}


///Calculate the logistic correlation between two variables when there are missing values.
/// @param x Vector of the independent variable.
/// @param y Vector of the dependent variable.

CorrelationResults logistic_correlations_missing_values(const Vector<double>& x, const Vector<double>& y)
{
#ifdef __OPENNN_DEBUG__

    ostringstream buffer;

    if(y.size() != x.size())
    {
      buffer << "OpenNN Exception: Correlations.\n"
             << "static double logistic_correlation(const Vector<double>&, const Vector<double>&) method.\n"
             << "Y size(" <<y.size()<<") must be equal to X size("<<x.size()<<").\n";

      throw logic_error(buffer.str());
    }

#endif

    pair <Vector<double>, Vector<double>> filter_vectors = filter_missing_values(x,y);

    const Vector<double> new_vector_x = filter_vectors.first;
    const Vector<double> new_vector_y = filter_vectors.second;

    Vector<double> scaled_x(new_vector_x);
    scale_minimum_maximum(scaled_x);

    Vector<double> coefficients({0.0, 0.0});

    const size_t epochs_number = 100000;
    double step_size = 0.01;

    const double error_goal = 1.0e-8;
    const double gradient_norm_goal = 1.0e-8;

    double error;

    Vector<double> gradient(2);

    for(size_t i = 0; i < epochs_number; i++)
    {
        error = logistic_error(coefficients[0], coefficients[1], new_vector_x, new_vector_y);

        if(error < error_goal) break;

        gradient = logistic_error_gradient(coefficients[0], coefficients[1], new_vector_x, new_vector_y);

        if(l2_norm(gradient) < gradient_norm_goal) break;

        coefficients -= gradient*step_size;
    }

    RegressionResults regression_results;

    regression_results.regression_type = Logistic;
    regression_results.a = coefficients[0];
    regression_results.b = coefficients[1];
    regression_results.correlation = linear_correlation(logistic_function(new_vector_x,regression_results.a,regression_results.b), new_vector_y);

    CorrelationResults correlation_results;

    correlation_results.correlation_type = Logistic_correlation;
    correlation_results.correlation = regression_results.correlation;

    return correlation_results;
}


///Calculate the Karl Pearson correlation between two variables.
/// @param x Matrix of the variable X.
/// @param y Matrix of the variable Y.

CorrelationResults karl_pearson_correlations(const Matrix<double>& x, const Matrix<double>& y)
{
    const size_t n = x.get_rows_number();

    #ifdef  __OPENNN_DEBUG__

    if(x.get_columns_number() == 0)
    {
       ostringstream buffer;

       buffer << "OpenNN Exception: Matrix template."
              << "size_t karl_pearson_correlation_missing_values const method.\n"
              << "Number of columns("<< x.get_columns_number() <<") must be greater than zero.\n";

       throw logic_error(buffer.str());
    }

    if(y.get_columns_number() == 0)
    {
       ostringstream buffer;

       buffer << "OpenNN Exception: Matrix template."
              << "size_t karl_pearson_correlation_missing_values const method.\n"
              << "Number of columns("<< y.get_columns_number() <<") must be greater than zero.\n";

       throw logic_error(buffer.str());
    }

    if(n != y.get_rows_number())
    {
       ostringstream buffer;

       buffer << "OpenNN Exception: Matrix template."
              << "size_t karl_pearson_correlation const method.\n"
              << "Number of rows of the two variables must be equal\t"<< x.get_rows_number() <<"!=" << y.get_rows_number() << ".\n";

       throw logic_error(buffer.str());
    }

    #endif

    Matrix<size_t> contingency_table(x.get_columns_number(),y.get_columns_number());

    for(size_t i = 0; i < x.get_columns_number(); i ++ )
    {
        for(size_t j = 0; j < y.get_columns_number(); j ++)
        {
            size_t count = 0;

            for(size_t k = 0; k < n; k ++)
            {
                if(abs(x(k,i) + y(k,j) - 2) <= 1e-4)
                {
                    count ++;

                    contingency_table(i,j) = count;
                }
            }
        }
    }

    size_t k;

    if(x.get_columns_number() <= y.get_columns_number()) k = x.get_columns_number();
     else k = y.get_columns_number();

    const double chi_squared = chi_square_test(contingency_table.to_double_matrix());

    CorrelationResults karl_pearson;

    karl_pearson.correlation_type = KarlPearson_correlation;

    karl_pearson.correlation = sqrt(k / (k - 1.0)) * sqrt(chi_squared/(chi_squared + contingency_table.calculate_sum()));

    return karl_pearson;
}


///Calculate the Karl Pearson correlation between two variables when there are missing values.
/// @param x Matrix of the variable X.
/// @param y Matrix of the variable Y.

CorrelationResults karl_pearson_correlations_missing_values(const Matrix<double>& x, const Matrix<double>& y)
{
    size_t n = x.get_rows_number();

    #ifdef  __OPENNN_DEBUG__

    if(x.get_columns_number() == 0)
    {
       ostringstream buffer;

       buffer << "OpenNN Exception: Correlation class."
              << "double karl_pearson_correlation_missing_values(const Matrix<double>&, const Matrix<double>& ) method.\n"
              << "Number of columns("<< x.get_columns_number() <<") must be greater than zero.\n";

       throw logic_error(buffer.str());
    }

    if(y.get_columns_number() == 0)
    {
       ostringstream buffer;

       buffer << "OpenNN Exception: Correlation class."
              << "double karl_pearson_correlation_missing_values(const Matrix<double>&, const Matrix<double>& ) method.\n"
              << "Number of columns("<< y.get_columns_number() <<") must be greater than zero.\n";

       throw logic_error(buffer.str());
    }

    if(n != y.get_rows_number())
    {
       ostringstream buffer;

       buffer << "OpenNN Exception: Correlation class."
              << "double karl_pearson_correlation_missing_values(const Matrix<double>&, const Matrix<double>& ) method.\n"
              << "Number of rows of the two variables must be equal\t"<< x.get_rows_number() <<"!=" << y.get_rows_number() << ".\n";

       throw logic_error(buffer.str());
    }

    #endif

    const size_t NAN_x = x.count_rows_with_nan();

    const size_t NAN_y = y.count_rows_with_nan();

    size_t new_size;

    if(NAN_x <= NAN_y )
    {
        new_size = n - NAN_y;
    }
    else
    {
        new_size = n - NAN_x;
    }

    Matrix<double> new_x(new_size,x.get_columns_number());

    Matrix<double> new_y(new_size,y.get_columns_number());

    const Vector<size_t> nan_indices_x = x.get_nan_indices();

    const Vector<size_t> nan_indices_y = y.get_nan_indices();

    for(size_t i = 0; i < nan_indices_x.size(); i++)
    {

        new_x = x.delete_rows(nan_indices_x);
        new_y = y.delete_rows(nan_indices_x);
    }

    for(size_t j = 0; j < nan_indices_y.size(); j++)
    {
        new_x = x.delete_rows(nan_indices_y);
        new_y = y.delete_rows(nan_indices_y);
    }

    n = new_x.get_rows_number();

    Matrix<size_t> contingency_table(new_x.get_columns_number(),new_y.get_columns_number());

    for(size_t i = 0; i < new_x.get_columns_number(); i ++ )
    {
        for(size_t j = 0; j < new_y.get_columns_number(); j ++)
        {
            size_t count = 0;

            for(size_t k = 0; k < n; k ++)
            {
                if(abs(new_x(k,i) + new_y(k,j) - 2) <= 0.0001)
                {
                    count ++;

                    contingency_table(i,j) = count;
                }
            }
        }
    }

    size_t k;

    if(x.get_columns_number() <= y.get_columns_number()) k = x.get_columns_number();
    else k = y.get_columns_number();

    const double chi_squared = chi_square_test(contingency_table.to_double_matrix());

    CorrelationResults karl_pearson;

    karl_pearson.correlation_type = KarlPearson_correlation;

    karl_pearson.correlation = sqrt(k / (k - 1.0)) * sqrt(chi_squared/(chi_squared + contingency_table.calculate_sum()));

    return karl_pearson;
}


///Calculate the one way anova correlation between two variables.
/// @param x Matrix of the categorical variable.
/// @param y Vector of the variable numeric variable.

CorrelationResults one_way_anova_correlations(const Matrix<double>& matrix, const Vector<double>& vector)
{
#ifdef __OPENNN_DEBUG__

    ostringstream buffer;

    if(matrix.get_rows_number() != vector.size())
    {
      buffer << "OpenNN Exception: Correlations.\n"
             << "one_way_anova_correlation(const Matrix<double>&, const Vector<double>& ) method.\n"
             << "Rows of the matrix (" << matrix.get_rows_number() << ") must be equal to size of vector (" << vector.size() << ").\n";

      throw logic_error(buffer.str());
    }

#endif
    const double n = static_cast<double>(matrix.get_rows_number());

    Matrix<double> new_matrix(matrix.get_rows_number(),matrix.get_columns_number());

    Vector<double> number_elements(matrix.get_columns_number());

    Vector<double> groups_average(matrix.get_columns_number());

    const double total_average = vector.calculate_sum() / n;

    double total_sum_of_squares = 0.0;
    double treatment_sum_of_squares = 0.0;

    for(size_t i = 0; i < n; i ++)
    {
        for(size_t j = 0; j < matrix.get_columns_number(); j++)
        {
           new_matrix(i,j) = matrix(i,j) * vector[i];

           number_elements[j] = matrix.calculate_column_sum(j);

           groups_average[j] = new_matrix.calculate_column_sum(j) / number_elements[j];
        }

        total_sum_of_squares += pow(vector[i] - total_average,2);

    }

    for(size_t i = 0; i < matrix.get_columns_number(); i ++)
    {
        treatment_sum_of_squares += number_elements[i] * pow(groups_average[i] - total_average,2);
    }

    CorrelationResults one_way_anova;

    one_way_anova.correlation_type = OneWayAnova_correlation;

    one_way_anova.correlation = sqrt(treatment_sum_of_squares / total_sum_of_squares);

    return one_way_anova;
}

///Calculate the one way anova correlation between two variables when there are missing values.
/// @param x Matrix of the categorical variable.
/// @param y Vector of the variable numeric variable.

CorrelationResults one_way_anova_correlations_missing_values(const Matrix<double>& matrix, const Vector<double>& vector)
{
#ifdef __OPENNN_DEBUG__

    ostringstream buffer;

    if(matrix.get_rows_number() != vector.size())
    {
      buffer << "OpenNN Exception: Correlations.\n"
             << "one_way_anova_correlation(const Matrix<double>& matrix, const Vector<double>& vector) method.\n"
             << "Rows of the matrix (" << matrix.get_rows_number() << ") must be equal to size of vector (" << vector.size() << ").\n";

      throw logic_error(buffer.str());
    }

#endif
    const size_t this_size = matrix.get_rows_number();

    const size_t not_NAN_x = this_size - matrix.count_rows_with_nan();

    const size_t not_NAN_y = vector.count_not_NAN();

    size_t new_size;

    if(not_NAN_x <= not_NAN_y )
    {
        new_size = not_NAN_x;
    }
    else
    {
        new_size = not_NAN_y;
    }

    Matrix<double> matrix1(new_size,matrix.get_columns_number());

    Vector<double> new_y(new_size);

    size_t index = 0;

    for(size_t i = 0; i < this_size ; i++)
    {
        if(!::isnan(vector[i]))
        {
            new_y[index] = vector[i];

            for(size_t j = 0; j < matrix1.get_columns_number(); j++)
            {
                if(!::isnan(matrix[i]))
                {
                    matrix1(index,j) = matrix(i,j);
                }
            }

            index++;
         }
    }

    const double n = static_cast<double>(matrix1.get_rows_number());

    Matrix<double> new_matrix(matrix1.get_rows_number(),matrix1.get_columns_number());

    Vector<double> number_elements(matrix1.get_columns_number());

    Vector<double> groups_average(matrix1.get_columns_number());

    const double total_average = new_y.calculate_sum() / n;

    double total_sum_of_squares = 0.0;
    double treatment_sum_of_squares = 0.0;

    for(size_t i = 0; i < n; i ++)
    {
        for(size_t j = 0; j < matrix1.get_columns_number(); j++)
        {
           new_matrix(i,j) = matrix1(i,j) * new_y[i];

           number_elements[j] = matrix1.calculate_column_sum(j);

           groups_average[j] = new_matrix.calculate_column_sum(j) / number_elements[j];
        }

        total_sum_of_squares += pow(new_y[i] - total_average,2);
    }

    for(size_t i = 0; i < matrix1.get_columns_number(); i ++)
    {
        treatment_sum_of_squares += number_elements[i] * pow(groups_average[i] - total_average,2);
    }

    CorrelationResults one_way_anova;

    one_way_anova.correlation_type = OneWayAnova_correlation;

    one_way_anova.correlation = sqrt(treatment_sum_of_squares / total_sum_of_squares);

    return one_way_anova;
}


/// Returns the covariance of this vector and other vector
/// @param vector_1 data
/// @param vector_2 data

double covariance(const Vector<double>& vector_1, const Vector<double>& vector_2)
{
    const size_t size_1 = vector_1.size();
    const size_t size_2 = vector_2.size();

   #ifdef __OPENNN_DEBUG__

     if(size_1 == 0) {
       ostringstream buffer;

       buffer << "OpenNN Exception: Vector Template.\n"
              << "double covariance(const Vector<double>&) const method.\n"
              << "Size must be greater than zero.\n";

       throw logic_error(buffer.str());
     }

         if(size_1 != size_2) {
             ostringstream buffer;

             buffer << "OpenNN Exception: Vector Template.\n"
                    << "double covariance(const Vector<double>&) const method.\n"
                    << "Size of this vectro must be equal to size of other vector.\n";

             throw logic_error(buffer.str());
         }

    #endif

     if(size_1 == 1)
     {
         return 0.0;
     }

     const double mean_1 = mean(vector_1);
     const double mean_2 = mean(vector_2);

     double numerator = 0.0;
     double denominator = static_cast<double>(size_2-1);

     for(size_t i = 0; i < size_1; i++)
     {
         numerator += (vector_1[i] - mean_1)*(vector_2[i]-mean_2);
     }

     return numerator/denominator;
}


/// Returns the covariance of this vector and other vector
/// @param vector_1 data
/// @param vector_2 data

double covariance_missing_values(const Vector<double>& x, const Vector<double>& y)
{
    pair <Vector<double>, Vector<double>> filter_vectors = filter_missing_values(x,y);

    const Vector<double> new_x = filter_vectors.first;
    const Vector<double> new_y = filter_vectors.second;

    const size_t size_1 = new_x.size();
    const size_t size_2 = new_y.size();

   #ifdef __OPENNN_DEBUG__

     if(size_1 == 0) {
       ostringstream buffer;

       buffer << "OpenNN Exception: Vector Template.\n"
              << "double covariance(const Vector<double>&) const method.\n"
              << "Size must be greater than zero.\n";

       throw logic_error(buffer.str());
     }

         if(size_1 != size_2) {
             ostringstream buffer;

             buffer << "OpenNN Exception: Vector Template.\n"
                    << "double covariance(const Vector<double>&) const method.\n"
                    << "Size of this vectro must be equal to size of other vector.\n";

             throw logic_error(buffer.str());
         }

    #endif

     if(size_1 == 1)
     {
         return 0.0;
     }

     const double mean_1 = mean(new_x);
     const double mean_2 = mean(new_y);

     double numerator = 0.0;
     double denominator = static_cast<double>(size_2-1);

     for(size_t i = 0; i < size_1; i++)
     {
         numerator += (new_x[i] - mean_1)*(new_y[i]-mean_2);
     }

     return numerator/denominator;
}


/// Retruns the covariance matrix of this matrix.
/// The number of columns and rows of the matrix is equal to the number of columns of this matrix.
/// Covariance matrix is symmetric

Matrix<double> covariance_matrix(const Matrix<double>& matrix)
{
    const size_t columns_number = matrix.get_columns_number();

    const size_t size = columns_number;

    #ifdef __OPENNN_DEBUG__

    if(size == 0)
    {
       ostringstream buffer;

       buffer << "OpenNN Exception: Matrix template."
              << "void covariance_matrix() const method.\n"
              << "Number of columns("<< columns_number <<") must be greater than zero.\n";

       throw logic_error(buffer.str());
    }

    #endif

    Matrix<double> covariance_matrix(size, size, 0.0);

    Vector<double> first_column;
    Vector<double> second_column;

    for(size_t i = 0; i < size; i++)
    {
        first_column = matrix.get_column(i);

        for(size_t j = i; j < size; j++)
        {
            second_column = matrix.get_column(j);

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

Vector<double> less_rank_with_ties(const Vector<double>& vector)
{

    Vector<double> indices_this = vector.calculate_less_rank().to_double_vector();

    const Vector<double> this_unique = vector.get_unique_elements();
    const Vector<size_t> this_unique_frecuency = vector.count_unique();

    const size_t n = vector.size();

    for(int  i = 0; i < static_cast<int>(this_unique.size()); i++)
    {
        if(this_unique_frecuency[static_cast<size_t>(i)] > 1)
        {
             const double unique = this_unique[static_cast<size_t>(i)];

             Vector<double> indices(this_unique_frecuency[static_cast<size_t>(i)]);

             for(size_t j = 0; j < n; j++)
             {
                 if(abs(vector[j] - unique) < numeric_limits<double>::min())
                 {
                     indices.push_back(indices_this[j]);
                 }
             }

             const double mean_index = mean(indices);

             for(size_t j = 0; j < n; j++)
             {
                 if(abs(vector[j] - unique) < numeric_limits<double>::min())
                 {
                     indices_this[j] = mean_index;
                 }
             }
        }
    }
    return indices_this + 1;
}


/// Calculate the contingency table of two cualitatives variables given to the function.
/// @param vector1 First variable.
/// @param vector2 Second variable.

Matrix<size_t> contingency_table(const Vector<string>& vector1, const Vector<string>& vector2)
{
    Matrix<string> data_set = {vector1, vector2};

    data_set.set_header(Vector<string>({"variable1","variable2"}));

    const Vector<string> categories_vector1 = vector1.get_unique_elements();
    const Vector<string> categories_vector2 = vector2.get_unique_elements();

    const size_t rows_number = categories_vector1.size();
    const size_t columns_number = categories_vector2.size();

    Matrix<size_t> contingency_table(rows_number,columns_number);

    for(size_t i = 0 ; i < rows_number; i ++)
    {
        for(size_t j = 0 ; j < columns_number; j ++)
        {
            contingency_table(i,j) = data_set.count_equal_to("variable1",categories_vector1[i],"variable2",categories_vector2[j]);
        }
    }

    return contingency_table;
}


/// Calculate the contingency table of two cualitatives variables given to the function
/// @param contingency_table Data set to perform the test.

Matrix<size_t> contingency_table(Matrix<string>& matrix)
{
    #ifdef __OPENNN_DEBUG__

    if(matrix.get_columns_number() == 0)
    {
       ostringstream buffer;

       buffer << "OpenNN Exception: Matrix template."
              << "size_t contingency_table const method.\n"
              << "Number of columns("<< matrix.get_columns_number() <<") must be greater than zero.\n";

       throw logic_error(buffer.str());
    }

    if(matrix.get_columns_number() > 2)
    {
       ostringstream buffer;

       buffer << "OpenNN Exception: Matrix template."
              << "size_t contingency_table const method.\n"
              << "Number of columns("<< matrix.get_columns_number() <<") must be lower than two.\n";

       throw logic_error(buffer.str());
    }

    #endif


    matrix.set_header({"variable1", "variable2"});

    const Vector<string> vector1 = matrix.get_column(0);
    const Vector<string> vector2 = matrix.get_column(1);

    const Vector<string> categories_vector1 = vector1.get_unique_elements();
    const Vector<string> categories_vector2 = vector2.get_unique_elements();

    const size_t rows_number = categories_vector1.size();
    const size_t columns_number = categories_vector2.size();

    Matrix<size_t> contingency_table(rows_number,columns_number);

    for(size_t i = 0 ; i < rows_number; i ++)
    {
        for(size_t j = 0 ; j < columns_number; j ++)
        {
            contingency_table(i,j) = matrix.count_equal_to("variable1",categories_vector1[i],"variable2",categories_vector2[j]);
        }
    }

    return contingency_table;
}


/// Calculate the contingency table of two cualitatives variables given to the function

Matrix<size_t> contingency_table(const Matrix<double>& matrix, const Vector<size_t>& indices1, const Vector<size_t>& indices2)
{
    Matrix<size_t> contingency_table(indices1.size(), indices2.size());

    for(size_t i = 0; i < indices1.size(); i ++)
    {

        size_t count = 0;

        size_t i_2;

        i_2 = indices1[i];

        for(size_t j = 0; j < indices2.size(); j ++)
        {
            size_t j_2;

            j_2 = indices2[j];

            for(size_t k = 0; k < matrix.get_rows_number(); k ++)
            {
                if(matrix(k,i_2) + matrix(k,j_2) - 2 <= 0.0001)
                {
                    count ++;
                    contingency_table(i,j) = count;
                }
            }
        }
    }

    return contingency_table;
}


/// Calculate the chi squared test statistic of a contingency table
/// @param matrix Matrix that represent the contingency table

double chi_square_test(const Matrix<double>& matrix)
{
    const Vector<double> sum_columns = matrix.calculate_columns_sum();
    const Vector<double> sum_rows = matrix.calculate_rows_sum();

    const double total = sum_columns.calculate_sum();

    Matrix<double> row_colum(sum_rows.size(),sum_columns.size());

    for(size_t i = 0; i < sum_columns.size(); i++)
    {
      for(size_t j = 0; j < sum_rows.size(); j++)
      {
            row_colum(j,i) = sum_columns[i]*sum_rows[j];
      }
    }

    const Matrix<double> ei = row_colum/total;

    const Matrix<double> squared = ((matrix.to_double_matrix()-ei)*(matrix.to_double_matrix()-ei));

    Matrix<double> chi (sum_rows.size(),sum_columns.size());

    for(size_t i = 0; i < sum_rows.size(); i++)
    {
          for(size_t j = 0; j < sum_columns.size(); j ++)
          {
                chi(i,j) = squared(i,j)/ei(i,j);
          }
    }

    return chi.calculate_sum();
}


///Calaculate the chi squared critical point of a contingency table
/// @param alpha significance of the test
/// @param degree_freedom Degree of freedom of the contingency table

double chi_square_critical_point(const double& alpha, const double& degrees_of_freedom)
{   
    const double zeta = degrees_of_freedom/2.0;

    const double gamma = pow((zeta+1),zeta-0.5)/exp(zeta+1)*(sqrt(2*3.14159265)+(pow(1,0.5)*exp(1)/zeta));

    const double step = 1.0e-5;

    double p_0 = 0.0;
    double p_1 = 0.0;

    double x = 0.0;

    while(p_1 < 1.0 - alpha)
    {
        x += step;

        const double f_x = pow(x, (zeta-1))/((exp(x/2))*pow(2, zeta)*gamma);

        p_1 = p_0 + step * f_x;

        p_0 = p_1;
    }

    return x;
}


///Calculate the karl_pearson_coefficient between two cualitative variable. It shows the realtionship between the two varibles.
/// @param x First variable
/// @param y Second variable

double karl_pearson_correlation(const Vector<string>& x, const Vector<string>& y)
{
    const Matrix<size_t> contingency_table = OpenNN::contingency_table(x,y);

    const double chi_squared_exp = chi_square_test(contingency_table.to_double_matrix());

    Vector<size_t> categories(2);
    categories[0] = x.get_unique_elements().size();
    categories[1] = y.get_unique_elements().size();

    const size_t k = minimum(categories);

    return sqrt(k/(k-1))*sqrt(chi_squared_exp/(x.size() + chi_squared_exp));
}


///Calculate the karl_pearson_coefficient between two categorical variables.
/// It shows the realtionship between the two varibles.
/// @param x First variable
/// @param y Second variable

double karl_pearson_correlation(const Matrix<double>& x, const Matrix<double>& y)
{
    const size_t n = x.get_rows_number();

    #ifdef  __OPENNN_DEBUG__

    if(x.get_columns_number() == 0)
    {
       ostringstream buffer;

       buffer << "OpenNN Exception: Matrix template."
              << "size_t karl_pearson_correlation_missing_values const method.\n"
              << "Number of columns("<< x.get_columns_number() <<") must be greater than zero.\n";

       throw logic_error(buffer.str());
    }

    if(y.get_columns_number() == 0)
    {
       ostringstream buffer;

       buffer << "OpenNN Exception: Matrix template."
              << "size_t karl_pearson_correlation_missing_values const method.\n"
              << "Number of columns("<< y.get_columns_number() <<") must be greater than zero.\n";

       throw logic_error(buffer.str());
    }

    if(n != y.get_rows_number())
    {
       ostringstream buffer;

       buffer << "OpenNN Exception: Matrix template."
              << "size_t karl_pearson_correlation const method.\n"
              << "Number of rows of the two variables must be equal\t"<< x.get_rows_number() <<"!=" << y.get_rows_number() << ".\n";

       throw logic_error(buffer.str());
    }

    #endif

    Matrix<size_t> contingency_table(x.get_columns_number(),y.get_columns_number());

    for(size_t i = 0; i < x.get_columns_number(); i ++ )
    {
        for(size_t j = 0; j < y.get_columns_number(); j ++)
        {
            size_t count = 0;

            for(size_t k = 0; k < n; k ++)
            {
                if(abs(x(k,i) + y(k,j) - 2) <= 1e-4)
                {
                    count ++;

                    contingency_table(i,j) = count;
                }
            }
        }
    }

    size_t k;

    if(x.get_columns_number() <= y.get_columns_number()) k = x.get_columns_number();
     else k = y.get_columns_number();

    const double chi_squared = chi_square_test(contingency_table.to_double_matrix());

    const double karl_pearson_correlation = sqrt(k / (k - 1.0)) * sqrt(chi_squared/(chi_squared + contingency_table.calculate_sum()));

    return karl_pearson_correlation;
}


///Calculate the karl_pearson_coefficient between two cualitative variable. It shows the realtionship between the two varibles.
/// @param x First variable
/// @param y Second variable

double karl_pearson_correlation_missing_values(const Matrix<double>& x, const Matrix<double>& y)
{
    size_t n = x.get_rows_number();

    #ifdef  __OPENNN_DEBUG__

    if(x.get_columns_number() == 0)
    {
       ostringstream buffer;

       buffer << "OpenNN Exception: Correlation class."
              << "double karl_pearson_correlation_missing_values(const Matrix<double>&, const Matrix<double>& ) method.\n"
              << "Number of columns("<< x.get_columns_number() <<") must be greater than zero.\n";

       throw logic_error(buffer.str());
    }

    if(y.get_columns_number() == 0)
    {
       ostringstream buffer;

       buffer << "OpenNN Exception: Correlation class."
              << "double karl_pearson_correlation_missing_values(const Matrix<double>&, const Matrix<double>& ) method.\n"
              << "Number of columns("<< y.get_columns_number() <<") must be greater than zero.\n";

       throw logic_error(buffer.str());
    }

    if(n != y.get_rows_number())
    {
       ostringstream buffer;

       buffer << "OpenNN Exception: Correlation class."
              << "double karl_pearson_correlation_missing_values(const Matrix<double>&, const Matrix<double>& ) method.\n"
              << "Number of rows of the two variables must be equal\t"<< x.get_rows_number() <<"!=" << y.get_rows_number() << ".\n";

       throw logic_error(buffer.str());
    }

    #endif

    const size_t NAN_x = x.count_rows_with_nan();

    const size_t NAN_y = y.count_rows_with_nan();

    size_t new_size;

    if(NAN_x <= NAN_y )
    {
        new_size = n - NAN_y;
    }
    else
    {
        new_size = n - NAN_x;
    }

    Matrix<double> new_x(new_size,x.get_columns_number());

    Matrix<double> new_y(new_size,y.get_columns_number());

    const Vector<size_t> nan_indices_x = x.get_nan_indices();

    const Vector<size_t> nan_indices_y = y.get_nan_indices();

    for(size_t i = 0; i < nan_indices_x.size(); i++)
    {

        new_x = x.delete_rows(nan_indices_x);
        new_y = y.delete_rows(nan_indices_x);
    }

    for(size_t j = 0; j < nan_indices_y.size(); j++)
    {
        new_x = x.delete_rows(nan_indices_y);
        new_y = y.delete_rows(nan_indices_y);
    }

    n = new_x.get_rows_number();

    Matrix<size_t> contingency_table(new_x.get_columns_number(),new_y.get_columns_number());

    for(size_t i = 0; i < new_x.get_columns_number(); i ++ )
    {
        for(size_t j = 0; j < new_y.get_columns_number(); j ++)
        {
            size_t count = 0;

            for(size_t k = 0; k < n; k ++)
            {
                if(abs(new_x(k,i) + new_y(k,j) - 2) <= 0.0001)
                {
                    count ++;

                    contingency_table(i,j) = count;
                }
            }
        }
    }

    size_t k;

    if(x.get_columns_number() <= y.get_columns_number()) k = x.get_columns_number();
    else k = y.get_columns_number();

    const double chi_squared = chi_square_test(contingency_table.to_double_matrix());

    const double karl_pearson_correlation = sqrt(k / (k - 1.0)) * sqrt(chi_squared/(chi_squared + contingency_table.calculate_sum()));

    return karl_pearson_correlation;
}


/// Returns the F test stadistic obteined of the performance of the one way anova
/// @param x Matrix of the categorical variable.
/// @param y Vector of the variable numeric variable.

double one_way_anova(const Matrix<double>& matrix, const Vector<double>& vector)
{
    const size_t n = matrix.get_rows_number();

    Matrix<double> new_matrix(matrix.get_rows_number(),matrix.get_columns_number());

    Vector<double> number_elements(matrix.get_columns_number());

    Vector<double> groups_average(matrix.get_columns_number());

    const double total_average = vector.calculate_sum() / n;

    double total_sum_of_squares = 0.0;
    double treatment_sum_of_squares = 0.0;

    for(size_t i = 0; i < n; i ++)
    {
        for(size_t j = 0; j < matrix.get_columns_number(); j++)
        {
           new_matrix(i,j) = matrix(i,j) * vector[i];

           number_elements[j] = matrix.calculate_column_sum(j);

           groups_average[j] = new_matrix.calculate_column_sum(j) / number_elements[j];

        }

        total_sum_of_squares += pow(vector[i] - total_average,2);

    }

    for(size_t i = 0; i < matrix.get_columns_number(); i ++)
    {
        treatment_sum_of_squares += number_elements[i] * pow(groups_average[i] - total_average,2);
    }

    const double error_sum_of_squares = total_sum_of_squares - treatment_sum_of_squares;

    const double treatment_mean_of_squares = treatment_sum_of_squares / (new_matrix.get_columns_number() - 1);

    const double error_mean_of_squares = error_sum_of_squares / (new_matrix.get_rows_number() - new_matrix.get_columns_number());

    const double f_statistic_test = treatment_mean_of_squares / error_mean_of_squares;

    return f_statistic_test;
}


/// Returns the F test stadistic obteined of the performance of the one way anova
/// @param matrix Data set to perform the test
/// @param index Index of the input variable
/// @param indices Vector of indices of the target variables

double one_way_anova(const Matrix<double>& matrix,const size_t& index, const Vector<size_t>& indices)
{
    const double total_average = matrix.calculate_sum() / matrix.get_rows_number();

    const Matrix<double> new_matrix = matrix.get_submatrix_columns(indices).assemble_columns(matrix.get_submatrix_columns({index}));

    Vector<double> treatments_element(new_matrix.get_columns_number()-1);
    Vector<double> average_of_targets(new_matrix.get_columns_number()-1);
    Vector<double> average(new_matrix.get_columns_number()-1);
    Vector<double> total_sum_of_squares(new_matrix.get_columns_number() - 1,0.0);
    Vector<double> treatment_sum_of_squares(new_matrix.get_columns_number() - 1);
    Vector<double> error_sum_of_squares(new_matrix.get_columns_number() - 1);

    for(size_t i = 0; i < new_matrix.get_columns_number() - 1; i++)
    {
        for(size_t j  = 0; j < new_matrix.get_rows_number(); j++)
        {
            treatments_element[i] = new_matrix.calculate_column_sum(i);

            average_of_targets[i] = (average_of_targets[i] + new_matrix(j,i) * new_matrix(j,new_matrix.get_columns_number()-1));

            average[i] = average_of_targets[i]/ new_matrix.calculate_column_sum(i);

            if (new_matrix(j,i) > 0 )
            {
                total_sum_of_squares[i] = total_sum_of_squares[i] + pow(new_matrix(j,i) * new_matrix(j,new_matrix.get_columns_number()-1) - total_average,2);
            }

            treatment_sum_of_squares[i] = new_matrix.calculate_column_sum(i) * pow(average[i] - total_average,2);

            error_sum_of_squares[i] = total_sum_of_squares[i] - treatment_sum_of_squares[i];
        }
    }

    const double mean_square_treatment = treatment_sum_of_squares.calculate_sum() / (indices.size() - 1);

    const double mean_square_error = error_sum_of_squares.calculate_sum() / (new_matrix.get_rows_number() - indices.size());

    const double f_statistic_test = mean_square_treatment / mean_square_error;

    return f_statistic_test;
}


///Returns the correlation between one categorical varaible and one numeric variable.
///@param matrix Binary matrix of the categorical variable.
///@param vector Contains the numerical variable.

double one_way_anova_correlation(const Matrix<double>& matrix, const Vector<double>& vector)
{
#ifdef __OPENNN_DEBUG__

    ostringstream buffer;

    if(matrix.get_rows_number() != vector.size())
    {
      buffer << "OpenNN Exception: Correlations.\n"
             << "one_way_anova_correlation(const Matrix<double>&, const Vector<double>& ) method.\n"
             << "Rows of the matrix (" << matrix.get_rows_number() << ") must be equal to size of vector (" << vector.size() << ").\n";

      throw logic_error(buffer.str());
    }

#endif
    const double n = static_cast<double>(matrix.get_rows_number());

    Matrix<double> new_matrix(matrix.get_rows_number(),matrix.get_columns_number());

    Vector<double> number_elements(matrix.get_columns_number());

    Vector<double> groups_average(matrix.get_columns_number());

    const double total_average = vector.calculate_sum() / n;

    double total_sum_of_squares = 0.0;
    double treatment_sum_of_squares = 0.0;

    for(size_t i = 0; i < n; i ++)
    {
        for(size_t j = 0; j < matrix.get_columns_number(); j++)
        {
           new_matrix(i,j) = matrix(i,j) * vector[i];

           number_elements[j] = matrix.calculate_column_sum(j);

           groups_average[j] = new_matrix.calculate_column_sum(j) / number_elements[j];
        }

        total_sum_of_squares += pow(vector[i] - total_average,2);

    }

    for(size_t i = 0; i < matrix.get_columns_number(); i ++)
    {
        treatment_sum_of_squares += number_elements[i] * pow(groups_average[i] - total_average,2);
    }

    const double correlation = sqrt(treatment_sum_of_squares / total_sum_of_squares);

    return correlation;
}


///Returns the correlation between one categorical varaible and one numeric variable.
///@param matrix Binary matrix of the categorical variable.
///@param vector Contains the numerical variable.

double one_way_anova_correlation_missing_values(const Matrix<double>& matrix, const Vector<double>& vector)
{
#ifdef __OPENNN_DEBUG__

    ostringstream buffer;

    if(matrix.get_rows_number() != vector.size())
    {
      buffer << "OpenNN Exception: Correlations.\n"
             << "one_way_anova_correlation(const Matrix<double>& matrix, const Vector<double>& vector) method.\n"
             << "Rows of the matrix (" << matrix.get_rows_number() << ") must be equal to size of vector (" << vector.size() << ").\n";

      throw logic_error(buffer.str());
    }

#endif
    const size_t this_size = matrix.get_rows_number();

    const size_t not_NAN_x = this_size - matrix.count_rows_with_nan();

    const size_t not_NAN_y = vector.count_not_NAN();

    size_t new_size;

    if(not_NAN_x <= not_NAN_y )
    {
        new_size = not_NAN_x;
    }
    else
    {
        new_size = not_NAN_y;
    }

    Matrix<double> matrix1(new_size,matrix.get_columns_number());

    Vector<double> new_y(new_size);

    size_t index = 0;

    for(size_t i = 0; i < this_size ; i++)
    {
        if(!::isnan(vector[i]))
        {
            new_y[index] = vector[i];

            for(size_t j = 0; j < matrix1.get_columns_number(); j++)
            {
                if(!::isnan(matrix[i]))
                {
                    matrix1(index,j) = matrix(i,j);
                }
            }

            index++;
         }
    }

    const double n = static_cast<double>(matrix1.get_rows_number());

    Matrix<double> new_matrix(matrix1.get_rows_number(),matrix1.get_columns_number());

    Vector<double> number_elements(matrix1.get_columns_number());

    Vector<double> groups_average(matrix1.get_columns_number());

    const double total_average = new_y.calculate_sum() / n;

    double total_sum_of_squares = 0.0;
    double treatment_sum_of_squares = 0.0;

    for(size_t i = 0; i < n; i ++)
    {
        for(size_t j = 0; j < matrix1.get_columns_number(); j++)
        {
           new_matrix(i,j) = matrix1(i,j) * new_y[i];

           number_elements[j] = matrix1.calculate_column_sum(j);

           groups_average[j] = new_matrix.calculate_column_sum(j) / number_elements[j];
        }

        total_sum_of_squares += pow(new_y[i] - total_average,2);
    }

    for(size_t i = 0; i < matrix1.get_columns_number(); i ++)
    {
        treatment_sum_of_squares += number_elements[i] * pow(groups_average[i] - total_average,2);
    }

    const double correlation = sqrt(treatment_sum_of_squares / total_sum_of_squares);

    return correlation;
}


///Calaculate the F squared critical point of a one way anova
/// @param matrix Data set to perform the test
/// @param alpha Significance of the test

double f_snedecor_critical_point(const Matrix<double>& matrix, const double& alpha)
{
    const size_t degrees_of_freedom1 = matrix.get_columns_number() - 1;
    const size_t degrees_of_freedom2 = matrix.get_columns_number() * matrix.get_rows_number() - matrix.count_equal_to(0) - matrix.get_columns_number();

    const double zeta1 = degrees_of_freedom1/2.0;
    const double zeta2 = degrees_of_freedom2/2.0;
    const double zeta3 = zeta1 + zeta2;

    const double gamma1 = pow((zeta1+1),zeta1-0.5)/exp(zeta1+1)*(sqrt(2*3.14159265)+(pow(1,0.5)*exp(1)/zeta1));
    const double gamma2 = pow((zeta2+1),zeta2-0.5)/exp(zeta2+1)*(sqrt(2*3.14159265)+(pow(1,0.5)*exp(1)/zeta2));
    const double gamma3 = pow((zeta3+1),zeta3-0.5)/exp(zeta3+1)*(sqrt(2*3.14159265)+(pow(1,0.5)*exp(1)/zeta3));

    const double beta = gamma1 * gamma2 / gamma3;

    double x = 0.0;
    double step = 0.00001;

    double p_0 = 0.0;
    double p_1 = 0.0;
    double f_x = 0.0;

    while (p_1 < 1- alpha){

        x += step;

        f_x = pow(pow(degrees_of_freedom1 * x,degrees_of_freedom1) * pow(degrees_of_freedom2, degrees_of_freedom2) / pow(degrees_of_freedom1 * x + degrees_of_freedom2, (degrees_of_freedom1 + degrees_of_freedom2)), 0.5) / (x * beta);

        p_1 = p_0 + step * static_cast<double>(f_x);

        p_0 = p_1;
    }

    return x;
}


///Calaculate the F squared critical point of a one way anova
/// @param matrix Data set to perform the test
/// @param alpha Significance of the test

double f_snedecor_critical_point(const Matrix<string>& matrix, const double& alpha)
{
    const size_t degrees_of_freedom1 = matrix.get_columns_number() - 2;
    const size_t degrees_of_freedom2 = matrix.get_rows_number() - matrix.get_columns_number() + 1;

    const double zeta1 = degrees_of_freedom1/2.0;
    const double zeta2 = degrees_of_freedom2/2.0;
    const double zeta3 = zeta1 + zeta2;

    const double gamma1 = pow((zeta1+1),zeta1-0.5)/exp(zeta1+1)*(sqrt(2*3.14159265)+(pow(1,0.5)*exp(1)/zeta1));
    const double gamma2 = pow((zeta2+1),zeta2-0.5)/exp(zeta2+1)*(sqrt(2*3.14159265)+(pow(1,0.5)*exp(1)/zeta2));
    const double gamma3 = pow((zeta3+1),zeta3-0.5)/exp(zeta3+1)*(sqrt(2*3.14159265)+(pow(1,0.5)*exp(1)/zeta3));

    const double beta = gamma1 * gamma2 / gamma3;

    double x = 0.0;
    double step = 0.0001;

    double p_0 = 0.0;
    double p_1 = 0.0;
    double f_x = 0.0;

    while (p_1 < 1- alpha){

        x += step;

        f_x = pow(pow(degrees_of_freedom1 * x,degrees_of_freedom1) * pow(degrees_of_freedom2, degrees_of_freedom2) / pow(degrees_of_freedom1 * x + degrees_of_freedom2, (degrees_of_freedom1 + degrees_of_freedom2)), 0.5) / (x * beta);

        p_1 = p_0 + step * static_cast<double>(f_x);

        p_0 = p_1;
    }

    return x;
}


///Calaculate the F squared critical point of a one way anova
/// @param matrix Data set to perform the test
/// @param alpha Significance of the test

double f_snedecor_critical_point(const Matrix<double>& matrix)
{
    const size_t degrees_of_freedom1 = matrix.get_columns_number() - 1;
    const size_t degrees_of_freedom2 = matrix.get_rows_number() - matrix.get_columns_number();

    const double zeta1 = degrees_of_freedom1/2.0;
    const double zeta2 = degrees_of_freedom2/2.0;
    const double zeta3 = zeta1 + zeta2;

    const double gamma1 = pow((zeta1+1),zeta1-0.5)/exp(zeta1+1)*(sqrt(2*3.14159265)+(pow(1,0.5)*exp(1)/zeta1));
    const double gamma2 = pow((zeta2+1),zeta2-0.5)/exp(zeta2+1)*(sqrt(2*3.14159265)+(pow(1,0.5)*exp(1)/zeta2));
    const double gamma3 = pow((zeta3+1),zeta3-0.5)/exp(zeta3+1)*(sqrt(2*3.14159265)+(pow(1,0.5)*exp(1)/zeta3));

    const double beta = gamma1 * gamma2 / gamma3;

    double x = 0.0;
    double step = 0.0001;

    double p_0 = 0.0;
    double p_1 = 0.0;
    double f_x = 0.0;

    while (p_1 < 1- 0.01){

        x += step;

        f_x = pow(pow(degrees_of_freedom1 * x,degrees_of_freedom1) * pow(degrees_of_freedom2, degrees_of_freedom2) / pow(degrees_of_freedom1 * x + degrees_of_freedom2, (degrees_of_freedom1 + degrees_of_freedom2)), 0.5) / (x * beta);

        p_1 = p_0 + step * static_cast<double>(f_x);

        p_0 = p_1;
    }

    return x;
}


///Calaculate the F squared critical point of a one way anova
/// @param matrix Data set to perform the test
/// @param alpha Significance of the test

double f_snedecor_critical_point_missing_values(const Matrix<double>& matrix)
{
    Vector<size_t> nan_indices = matrix.get_nan_indices();
    const size_t new_size = matrix.get_rows_number() - nan_indices.size();

    Matrix<double> new_matrix(new_size,matrix.get_columns_number());

    new_matrix = matrix.delete_rows(nan_indices);

    const size_t degrees_of_freedom1 = new_matrix.get_columns_number() - 1;
    const size_t degrees_of_freedom2 = new_matrix.get_rows_number() - new_matrix.get_columns_number();

    const double zeta1 = degrees_of_freedom1/2.0;
    const double zeta2 = degrees_of_freedom2/2.0;
    const double zeta3 = zeta1 + zeta2;

    const double gamma1 = pow((zeta1+1),zeta1-0.5)/exp(zeta1+1)*(sqrt(2*3.14159265)+(pow(1,0.5)*exp(1)/zeta1));
    const double gamma2 = pow((zeta2+1),zeta2-0.5)/exp(zeta2+1)*(sqrt(2*3.14159265)+(pow(1,0.5)*exp(1)/zeta2));
    const double gamma3 = pow((zeta3+1),zeta3-0.5)/exp(zeta3+1)*(sqrt(2*3.14159265)+(pow(1,0.5)*exp(1)/zeta3));

    const double beta = gamma1 * gamma2 / gamma3;

    double x = 0.0;
    double step = 0.0001;

    double p_0 = 0.0;
    double p_1 = 0.0;
    double f_x = 0.0;

    while (p_1 < 1- 0.99){

        x += step;

        f_x = pow(pow(degrees_of_freedom1 * x,degrees_of_freedom1) * pow(degrees_of_freedom2, degrees_of_freedom2) / pow(degrees_of_freedom1 * x + degrees_of_freedom2, (degrees_of_freedom1 + degrees_of_freedom2)), 0.5) / (x * beta);

        p_1 = p_0 + step * static_cast<double>(f_x);

        p_0 = p_1;
    }

    return x;
}


///Calculate the correlation between two variable. One of them numerical and the other one categorical.
/// @param matrix Data set to perform the test
/// @param index Index of the input variable
/// @param indices Vector of indices of the target variables

double one_way_anova_correlation(const Matrix<double>& matrix,const size_t& index, const Vector<size_t>& indices)
{
    const double total_average = matrix.calculate_sum() / matrix.get_rows_number();

    const Matrix<double> new_matrix = matrix.get_submatrix_columns(indices).assemble_columns(matrix.get_submatrix_columns({index}));

    Vector<double> treatments_element(new_matrix.get_columns_number()-1);
    Vector<double> average_of_targets(new_matrix.get_columns_number()-1);
    Vector<double> average(new_matrix.get_columns_number()-1);
    Vector<double> total_sum_of_squares(new_matrix.get_columns_number() - 1,0.0);
    Vector<double> treatment_sum_of_squares(new_matrix.get_columns_number() - 1);
    Vector<double> error_sum_of_squares(new_matrix.get_columns_number() - 1);

    for(size_t i = 0; i < new_matrix.get_columns_number() - 1; i++)
    {
        for(size_t j  = 0; j < new_matrix.get_rows_number(); j++)
        {

            treatments_element[i] = new_matrix.calculate_column_sum(i);

            average_of_targets[i] = (average_of_targets[i] + new_matrix(j,i) * new_matrix(j,new_matrix.get_columns_number()-1));

            average[i] = average_of_targets[i]/ new_matrix.calculate_column_sum(i);

            if (new_matrix(j,i) > 0 )
            {
                total_sum_of_squares[i] = total_sum_of_squares[i] + pow(new_matrix(j,i) * new_matrix(j,new_matrix.get_columns_number()-1) - total_average,2);
            }

            treatment_sum_of_squares[i] = new_matrix.calculate_column_sum(i) * pow(average[i] - total_average,2);

            error_sum_of_squares[i] = total_sum_of_squares[i] - treatment_sum_of_squares[i];
        }
    }

    const double one_way_anova_correlation = sqrt(treatment_sum_of_squares.calculate_sum()/ total_sum_of_squares.calculate_sum());

    return one_way_anova_correlation;
}


///Filter the missing values of two vectors

pair <Vector<double>, Vector<double>> filter_missing_values (const Vector<double>& x, const Vector<double>& y)
{
    const size_t not_NAN_x = x.count_not_NAN();

    const size_t not_NAN_y = y.count_not_NAN();

    size_t new_size;

    if(not_NAN_x <= not_NAN_y )
    {
        new_size = not_NAN_x;
    }
    else
    {
        new_size = not_NAN_y;
    }

    Vector<double> new_vector_x(new_size);

    Vector<double> new_vector_y(new_size);

    size_t index = 0;

    for(size_t i = 0; i < new_vector_x.size() ; i++)
    {
        if(!::isnan(x[i]) && !::isnan(y[i]))
        {
            new_vector_x[index] = x[i];
            new_vector_y[index] = y[i];

            index++;
        }
    }

    return make_pair(new_vector_x, new_vector_y);
}

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2019 Artificial Intelligence Techniques, SL.
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
