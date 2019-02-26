/****************************************************************************************************************/
/*                                                                                                              */
/*   OpenNN: Open Neural Networks Library                                                                       */
/*   www.opennn.net                                                                                             */
/*                                                                                                              */
/*   C O R R E L A T I O N   A N A L Y S I S                                                                    */
/*                                                                                                              */
/*   Javier Sanchez                                                                                             */
/*   Artificial Intelligence Techniques SL                                                                      */
/*   javiersanchez@artelnics.com                                                                                */
/*                                                                                                              */
/****************************************************************************************************************/

#include "correlation_analysis.h"


namespace OpenNN
{

// CONSTRUCTOR

// Default constructor

CorrelationAnalysis::CorrelationAnalysis()
{

}


// DESTRUCTOR

CorrelationAnalysis::~CorrelationAnalysis()
{

}


// METHODS


/// Calculates the linear correlation coefficient(Spearman method)(R-value) between two vectors.
/// @param x Vector containing data.
/// @param y Vector for computing the linear correlation with the x vector.

double CorrelationAnalysis::calculate_linear_correlation(const Vector<double>& x, const Vector<double>& y)
{
  const size_t n = x.size();

  if(x.is_constant() || y.is_constant())
  {
      return 1;
  }

// Control sentence(if debug)

  #ifdef __OPENNN_DEBUG__

    const size_t y_size = y.size();

    ostringstream buffer;

    if(y_size != n) {
      buffer << "OpenNN Exception: Correlation Analysis.\n"
             << "static double calculate_linear_correlation(const Vector<double>&, const Vector<double>&) method.\n"
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

    s_xy += y[i] * x[i];
  }

#ifdef __OPENNN_MPI__

  int n_send = (int)n;
  int n_mpi;

  double s_x_mpi = s_x;
  double s_y_mpi = s_y;

  double s_xx_mpi = s_xx;
  double s_yy_mpi = s_yy;

  double s_xy_mpi = s_xy;

  MPI_Allreduce(&n_send,&n_mpi,1,MPI_INT,MPI_SUM,MPI_COMM_WORLD);

  MPI_Allreduce(&s_x_mpi,&s_x,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
  MPI_Allreduce(&s_y_mpi,&s_y,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);

  MPI_Allreduce(&s_xx_mpi,&s_xx,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
  MPI_Allreduce(&s_yy_mpi,&s_yy,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);

  MPI_Allreduce(&s_xy_mpi,&s_xy,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);

  n = n_mpi;

#endif

  double linear_correlation;

  if(fabs(s_x - 0) < numeric_limits<double>::epsilon() && fabs(s_y - 0) < numeric_limits<double>::epsilon() && fabs(s_xx - 0) < numeric_limits<double>::epsilon()
          && fabs(s_yy - 0) < numeric_limits<double>::epsilon() && fabs(s_xy - 0) < numeric_limits<double>::epsilon()) {
    linear_correlation = 1;
  } else {
    const double numerator = (n * s_xy - s_x * s_y);

    const double radicand = (n * s_xx - s_x * s_x) *(n * s_yy - s_y * s_y);

    if(radicand <= 0.0) {
      return(1);
    }

    const double denominator = sqrt(radicand);

    if(denominator < 1.0e-50) {
      linear_correlation = 0;
    } else {
      linear_correlation = numerator / denominator;
    }
  }

  return(linear_correlation);
}


/// Calculates the linear correlation coefficient(R-value) between two
/// when there are missing values in the data.
/// @param x Vector containing data.
/// @param y Vector for computing the linear correlation with this vector.
/// @param missing_indices Vector with the indices of the missing values.

double CorrelationAnalysis::calculate_linear_correlation_missing_values(const Vector<double>& x, const Vector<double>& y, const Vector<size_t> &missing_indices)
{

  if(missing_indices.size() >= x.size())
  {
    return 0;
  }

  const Vector<double> x_missing = x.delete_indices(missing_indices);
  const Vector<double> y_missing = y.delete_indices(missing_indices);

  return calculate_linear_correlation(x_missing, y_missing);
}


/// Calculates the Rank-Order correlation coefficient(Spearman method) between two vectors.
/// @param x Vector containing data.
/// @param y Vector for computing the linear correlation with the x vector.

double CorrelationAnalysis::calculate_rank_linear_correlation(const Vector<double>& x, const Vector<double>& y)
{
    // Control sentence(if debug)

    #ifdef __OPENNN_DEBUG__

        ostringstream buffer;

        if(y.size() != x.size()) {
          buffer << "OpenNN Exception: Correlation Analysis.\n"
                 << "static double calculate_rank_linear_correlation(const Vector<double>&, const Vector<double>&) method.\n"
                 << "Y size must be equal to X size.\n";

          throw logic_error(buffer.str());
        }

    #endif

    if(x.is_constant() || y.is_constant()) return 1;

    const Vector<double> ranks_x = x.calculate_less_rank_with_ties();
    const Vector<double> ranks_y = y.calculate_less_rank_with_ties();

    return calculate_linear_correlation(ranks_x, ranks_y);
}


/// Calculates the Rank-Order correlation coefficient(Spearman method)(R-value) between two vectors.
/// Takes into account possible missing values.
/// @param x Vector containing input values.
/// @param y Vector for computing the linear correlation with the x vector.
/// @param mising Vector with the missing instances idices.

double CorrelationAnalysis::calculate_rank_linear_correlation_missing_values(const Vector<double>& x, const Vector<double>& y, const Vector<size_t>& missing_indices)
{
    const Vector<double> x_new = x.delete_indices(missing_indices);
    const Vector<double> y_new = y.delete_indices(missing_indices);

    return calculate_rank_linear_correlation(x_new, y_new);
}


/// Calculates the Point-Biserial correlation.
/// This is the Pearson correlation when 1 variable is binary.
/// @param x Vector containing the input vlaues.
/// @param y Vector containing the target values.

double CorrelationAnalysis::calculate_point_biserial_correlation(const Vector<double>& x, const Vector<double>& y)
{
#ifdef __OPENNN_DEBUG__

    ostringstream buffer;

    if(y.size() != x.size()) {
      buffer << "OpenNN Exception: Correlation Analysis.\n"
             << "static double calculate_point_biserial_correlation(const Vector<double>&, const Vector<double>&) method.\n"
             << "Y size must be equal to X size.\n";

      throw logic_error(buffer.str());
   }

#endif
    double pb_correlation;

    const size_t data_size = x.size();

    double x_std_dev = x.calculate_standard_deviation();

    double values_y1 = 0.0;
    double values_y0 = 0.0;
    double rate_1;
    int number_k1 = 0;
    int number_k0 = 0;

    for(size_t i = 0; i < data_size; i++)
    {

        if(fabs(y[i] - 1) < numeric_limits<double>::epsilon())
        {
            values_y1 += x[i];
            number_k1 +=1;
        }
        else if(fabs(y[i] - 0) < numeric_limits<double>::epsilon())
        {
            values_y0 += x[i];
            number_k0 += 1;
        }
    }

    values_y1 = values_y1 / number_k1;
    values_y0 = values_y0 / number_k0;
    rate_1 = double(number_k1)/data_size;

    pb_correlation = (values_y1-values_y0) * sqrt(rate_1*(1-rate_1)) / x_std_dev;

    return pb_correlation;
}


/// Calculates the Point-Biserial correlation.
/// This is the Pearson correlation when 1 variable is binary.
/// Takes into account possible missing values.
/// @param x Vector containing the input vlaues.
/// @param y Vector containing the target values.
/// @param mising Vector with the missing instances idices.

double CorrelationAnalysis::calculate_point_biserial_correlation_missing_values(const Vector<double>& x, const Vector<double>& y, const Vector<size_t>& missing_indices)
{
    if(missing_indices.size() >= x.size())
    {
      return 0;
    }

    const Vector<double> x_new = x.delete_indices(missing_indices);
    const Vector<double> y_new = y.delete_indices(missing_indices);

    return calculate_point_biserial_correlation(x_new, y_new);
}


/// Calculates the correlation between  Y and exp(A*X+B).
/// @param x Vector containing the input vlaues.
/// @param y Vector containing the target values.

double CorrelationAnalysis::calculate_exponential_correlation(const Vector<double>& x, const Vector<double>& y)
{
  #ifdef __OPENNN_DEBUG__

    ostringstream buffer;

      if(y.size() != x.size()) {
        buffer << "OpenNN Exception: Correlation Analysis.\n"
               << "static double calculate_exponential_correlation(const Vector<double>&, const Vector<double>&) method.\n"
               << "Y size must be equal to X size.\n";

        throw logic_error(buffer.str());
      }

#endif

    const Vector<size_t> negative_indices = y.calculate_less_equal_to_indices(0.0);
    const Vector<double> y_valid = y.delete_indices(negative_indices);

    Vector<double> log_y(y_valid.size());
    for(int i = 0; i < static_cast<int>(log_y.size()); i++) log_y[static_cast<size_t>(i)] = log(y_valid[static_cast<size_t>(i)]);

    return calculate_linear_correlation(x.delete_indices(negative_indices), log_y);
}


/// Calculates the correlation between  Y and exp(A*X + B).
/// @param x Vector containing the input vlaues.
/// @param y Vector containing the target values.
/// @param missing Vector with the missing instances indices.

double CorrelationAnalysis::calculate_exponential_correlation_missing_values(const Vector<double>& x, const Vector<double>& y, const Vector<size_t>& missing_indices)
{
    const Vector<double> x_new = x.delete_indices(missing_indices);
    const Vector<double> y_new = y.delete_indices(missing_indices);

    return calculate_exponential_correlation(x_new, y_new);
}


/// Calculates the correlation between  Y and ln(A*X + B).
/// @param x Vector containing the input vlaues.
/// @param y Vector containing the target values.

double CorrelationAnalysis::calculate_logarithmic_correlation(const Vector<double>& x, const Vector<double>& y)
{
#ifdef __OPENNN_DEBUG__

    ostringstream buffer;

     if(y.size() != x.size()) {
       buffer << "OpenNN Exception: Correlation Analysis.\n"
              << "static double calculate_logarithmic_correlation(const Vector<double>&, const Vector<double>&) method.\n"
              << "Y size must be equal to X size.\n";

       throw logic_error(buffer.str());
     }

#endif

    Vector<double> exp_y(y.size());
    for(int i = 0; i < static_cast<int>(exp_y.size()); i++) exp_y[static_cast<size_t>(i)] = exp(y[static_cast<size_t>(i)]);

    return calculate_linear_correlation(x, exp_y);
}


/// Calculates the correlation between  Y and ln(A*X+B).
/// Takes into account possible missing values.
/// @param x Vector containing the input vlaues.
/// @param y Vector containing the target values.
/// @param missing Vector with the missing instances indices.

double CorrelationAnalysis::calculate_logarithmic_correlation_missing_values(const Vector<double>& x, const Vector<double>& y, const Vector<size_t>& missing_indices)
{
    const Vector<double> x_new = x.delete_indices(missing_indices);
    const Vector<double> y_new = y.delete_indices(missing_indices);

    return calculate_logarithmic_correlation(x_new, y_new);
}


/// Calculates the logistic correlation coefficient between two vectors.
/// @param x Vector containing data.
/// @param y Vector for computing the linear correlation with the x vector.

double CorrelationAnalysis::calculate_logistic_correlation(const Vector<double>& x, const Vector<double>& y)
{
#ifdef __OPENNN_DEBUG__

    ostringstream buffer;

    if(y.size() != x.size()) {
      buffer << "OpenNN Exception: Correlation Analysis.\n"
             << "static double calculate_logistic_correlation(const Vector<double>&, const Vector<double>&) method.\n"
             << "Y size must be equal to X size.\n";

      throw logic_error(buffer.str());
    }

#endif

    Matrix<double> data(x.size(),2);

    data.set_column(0, x);
    data.set_column(1, y);

    DataSet data_set(data);

    data_set.get_instances_pointer()->set_training();

    const Vector<Statistics<double>> inputs_statistics = data_set.scale_inputs_minimum_maximum();

    NeuralNetwork neural_network(1,1);

    neural_network.construct_scaling_layer();

    ScalingLayer* scaling_layer_pointer = neural_network.get_scaling_layer_pointer();
    scaling_layer_pointer->set_statistics(inputs_statistics);
    scaling_layer_pointer->set_scaling_methods(ScalingLayer::MeanStandardDeviation);

    MultilayerPerceptron* multilayer_perceptron_pointer = neural_network.get_multilayer_perceptron_pointer();

    multilayer_perceptron_pointer->set_layer_activation_function(0, PerceptronLayer::Logistic);

    neural_network.construct_probabilistic_layer();
    ProbabilisticLayer* plp = neural_network.get_probabilistic_layer_pointer();
    plp->set_probabilistic_method(ProbabilisticLayer::Probability);

    TrainingStrategy training_strategy(&neural_network, &data_set);

    training_strategy.set_loss_method(TrainingStrategy::MEAN_SQUARED_ERROR);

//    WeightedSquaredError* weighted_squared_error = training_strategy.get_weighted_squared_error_pointer();

//    weighted_squared_error->set_weights();

//    weighted_squared_error->set_normalization_coefficient();

//    training_strategy.get_loss_index_pointer()->set_regularization_method(LossIndex::L2);

    training_strategy.set_training_method(TrainingStrategy::LEVENBERG_MARQUARDT_ALGORITHM);
//    training_strategy.get_conjugate_gradient_pointer()->set_training_batch_size(data_set.get_instances().get_training_instances_number());
//    training_strategy.get_conjugate_gradient_pointer()->set_selection_batch_size(data_set.get_instances().get_selection_instances_number());
//    training_strategy.get_conjugate_gradient_pointer()->get_learning_rate_algorithm_pointer()->set_training_rate_method(LearningRateAlgorithm::BrentMethod);


    training_strategy.get_Levenberg_Marquardt_algorithm_pointer()->set_training_batch_size(data_set.get_instances().get_training_instances_number());
    training_strategy.get_Levenberg_Marquardt_algorithm_pointer()->set_selection_batch_size(data_set.get_instances().get_selection_instances_number());

//    training_strategy.get_quasi_Newton_method_pointer()->set_training_batch_size(data_set.get_instances().get_training_instances_number());
//    training_strategy.get_quasi_Newton_method_pointer()->set_selection_batch_size(data_set.get_instances().get_selection_instances_number());
//    training_strategy.get_quasi_Newton_method_pointer()->get_learning_rate_algorithm_pointer()->set_training_rate_method(LearningRateAlgorithm::BrentMethod);

//    training_strategy.get_gradient_descent_pointer()->set_training_batch_size(data_set.get_instances().get_training_instances_number());
//    training_strategy.get_gradient_descent_pointer()->set_selection_batch_size(data_set.get_instances().get_selection_instances_number());
//    training_strategy.get_gradient_descent_pointer()->get_learning_rate_algorithm_pointer()->set_training_rate_method(LearningRateAlgorithm::BrentMethod);

    training_strategy.set_display(false);

    training_strategy.perform_training();

    Vector<double> value(1);
    Vector<double> output(x.size());

    for(size_t i = 0; i <x.size(); i++)
    {
        value[0] = x[i];
        output[i] = neural_network.calculate_outputs(value.to_column_matrix())[0];
    }

//    cout << "target: " << y << endl;
//    cout << "output: " << output << endl;

    return calculate_linear_correlation(y,output);

//    if(calculate_linear_correlation(x,output) < 0)
//    {
//        return  -1 * calculate_linear_correlation(y,output);
//    }
//    else
//    {
//        return calculate_linear_correlation(y,output);
//    }

//    return 0.0;
}


/// Calculates the logistic correlation coefficient between two vectors.
/// @param x Vector containing data.
/// @param y Vector for computing the linear correlation with the x vector.
/// @param missing_indices Vector with the missing indices.

double CorrelationAnalysis::calculate_logistic_correlation_missing_values(const Vector<double>& x, const Vector<double>& y, const Vector<size_t> & missing_indices)
{
    const size_t n = x.size();

    if(missing_indices.size() >= n)
    {
      return 0;
    }

    const Vector<double> this_valid = x.delete_indices(missing_indices);
    const Vector<double> y_valid = y.delete_indices(missing_indices);

    return calculate_logistic_correlation(this_valid, y_valid);
}

/// Calculates the logistic correlation coefficient between two vectors.
/// It uses non-parametric Spearman Rank-Order method.
/// It uses a Neural Network to compute the logistic function approximation.
/// @param x Vector containing data.
/// @param y Vector for computing the linear correlation with the x vector.

double CorrelationAnalysis::calculate_rank_logistic_correlation(const Vector<double>& x, const Vector<double>& y)
{
    const Vector<double> x_new = x.calculate_less_rank_with_ties();

    return calculate_logistic_correlation(x_new,y);
}


/// Returns the linear correlation betwen x^order and y
/// @param x Vector containing input data.
/// @param y Vector containing the target data.
/// @param order Order of the

double CorrelationAnalysis::calculate_polynomial_correlation(const Vector<double>& x, const Vector<double>& y, const size_t& order)
{
    Matrix<double> inputs_matrix = x.get_power_matrix(order);

    return calculate_multivariate_linear_correlation(inputs_matrix,y);
}


/// Finds the best order for the linear regression.
/// It test the correlation betwen x^n(n=1,2,...,max_order) and y.
/// Useful for finding non linear correlations between 1 input and 1 target.
/// x Vector containing input data.
/// y Vector containing the target data.
/// max_order Size_t maximun order to try.

size_t CorrelationAnalysis::calculate_best_correlation_order(const Vector<double>& x, const Vector<double>& y, const size_t& max_order)
{
#ifdef __OPENNN_DEBUG__

  ostringstream buffer;

   if(y.size() != x.size()) {
     buffer << "OpenNN Exception: Correlation Analysis.\n"
            << "static size_t calculate_best_correlation_order(const Vector<double>&, const Vector<double>&) method.\n"
            << "Y size must be equal to X size.\n";

     throw logic_error(buffer.str());
   }

#endif
    Matrix<double> inputs = x.get_power_matrix(max_order);

    Vector<double> correlations(max_order, 0.0);
    Vector<size_t> indexes(3,0);
    indexes.initialize_sequential();

    for(size_t i = 0; i <max_order; i++)
    {
        if(i == 0)
        {
            correlations[i] = abs(calculate_linear_correlation(x,y));
            if(correlations[i] > 0.9999999) return i+1;
        }
        else
        {
            indexes.resize(i);
            indexes.initialize_sequential();
            correlations[i] = calculate_multivariate_linear_correlation(inputs.get_submatrix_columns(indexes),y);
            if(correlations[i] > 0.9999999){return i+1;}
        }
    }

    return correlations.calculate_maximal_index()+1;
}


/// Returns the multivariate correlation of a set of variables and a vector.
/// @param x Matrix containing the variables.
/// @param y Target vector.

double CorrelationAnalysis::calculate_multivariate_linear_correlation(const Matrix<double>& x, const Vector<double>& y)
{
#ifdef __OPENNN_DEBUG__

  ostringstream buffer;

   if(y.size() != x.get_rows_number()) {
     buffer << "OpenNN Exception: Correlation Analysis.\n"
            << "static double calculate_multivariate_linear_correlation(const Matrix<double>&, const Vector<double>&) method.\n"
            << "Y size must be equal to X size.\n";

     throw logic_error(buffer.str());
   }

#endif

    const size_t inputs_number = x.get_columns_number();
    Vector<double> correlations_vector(inputs_number);

    double R_squared = 0.0;

    for(size_t i = 0; i < inputs_number; i++)
    {
        correlations_vector[i] = CorrelationAnalysis::calculate_linear_correlation(x.get_column(i), y);
    }

    R_squared = correlations_vector.dot(CorrelationAnalysis::calculate_correlations(x).calculate_inverse().dot(correlations_vector));

    return R_squared;
}


/// Returns the multivariate correlation of a set of variables and a vector.
/// Takes into account possible missing values.
/// @param x Matrix containing the variables.
/// @param y Target vector.
/// @param missin Vector containing the missing instances indexes.

double CorrelationAnalysis::calculate_multivariate_linear_correlation_missing_values(const Matrix<double>& x, const Vector<double>& y, const Vector<size_t>& missing_indices)
{
    if(missing_indices.size() >= x.get_rows_number())
    {
      return 0;
    }

    const Matrix<double> x_missing = x.delete_rows(missing_indices);
    const Vector<double> y_missing = y.delete_indices(missing_indices);

    return calculate_multivariate_linear_correlation(x_missing,y_missing);
}


/// Returns the paramater for logistic regression.
/// Based on Iterative Reweighted Least Squares(IRLS).
/// Valid for multiple input variables.
/// @param x Vector, continous variable.
/// @param y Vector, binary target.
/// @param tolerance Double, convergency goal parameter.

Vector<double> CorrelationAnalysis::calculate_logistic_regression(const Matrix<double>& x, const Vector<double>& y, const double& tolerance)
{
#ifdef __OPENNN_DEBUG__

   ostringstream buffer;

   if(y.size() != x.size()) {
     buffer << "OpenNN Exception: Correlation Analysis.\n"
            << "static Vector<double> calculate_logistic_regression(const Vector<double>&, const Vector<double>&) method.\n"
            << "Y size must be equal to X size.\n";

     throw logic_error(buffer.str());
   }

#endif

    const size_t vector_size = x.get_rows_number();
    const size_t parameters_number = x.get_columns_number()+1;

    Vector<double> old_parameters(parameters_number, 0.0);
    Vector<double> new_parameters(parameters_number);
    Vector<double> change_rate(old_parameters.size(), 1.0);

    Matrix<double> x_matrix(vector_size, parameters_number, 1.0);

    for(size_t i=1; i < parameters_number; i++)
    {
            x_matrix.set_column(i,x.get_column(i-1));
    }

    while(change_rate.calculate_maximum() > tolerance)
    {
        Vector<double> mu_vector(vector_size);
        for(size_t i = 0; i < vector_size; i++)
        {
            double factor = old_parameters.dot(x_matrix.get_row(i));
            mu_vector[i] = 1 /(1+exp(-factor));
        }

        Matrix<double> s_matrix(vector_size, vector_size, 0.0);
        for(size_t i = 0; i < vector_size; i++)
        {
            s_matrix(i,i) = mu_vector[i]*(1-mu_vector[i]);
        }

        new_parameters = x_matrix.dot(old_parameters);
        new_parameters = s_matrix.dot(new_parameters);
        new_parameters = new_parameters + y - mu_vector;
        new_parameters = x_matrix.calculate_transpose().dot(new_parameters);
        new_parameters = x_matrix.calculate_transpose().dot(s_matrix.dot(x_matrix)).calculate_inverse().dot(new_parameters);

        for(size_t i = 0; i < new_parameters.size(); i++)
        {
            change_rate[i] = abs(new_parameters[i] - old_parameters[i]);
            old_parameters[i] = new_parameters[i];
        }
    }

    return new_parameters;
}


Vector<double> CorrelationAnalysis::calculate_logistic_regression(const Vector<double>& x, const Vector<double>& y, const double& tolerance)
{
    Matrix<double> input_matrix(x.size(), 1, 0.0);
    input_matrix.set_column(0,x);

    return calculate_logistic_regression(input_matrix, y, tolerance);
}


/// Calculates autocorrelation for a given number of maximum lags.
/// @param x Vector containing the data.
/// @param lags_number Maximum lags number.

Vector<double> CorrelationAnalysis::calculate_autocorrelations(const Vector<double>& x, const size_t &lags_number)
{
  Vector<double> autocorrelation(lags_number);

  const double mean = x.calculate_mean();

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

Vector<double> CorrelationAnalysis::calculate_cross_correlations(const Vector<double>& x, const Vector<double>& y, const size_t &maximum_lags_number)
{
  if(y.size() != x.size()) {
    ostringstream buffer;

    buffer << "OpenNN Exception: Correlation Analysis.\n"
           << "Vector<double calculate_cross_correlation(const "
           << "Vector<double>&) method.\n"
           << "Both vectors must have the same size.\n";

    throw logic_error(buffer.str());
  }

  Vector<double> cross_correlation(maximum_lags_number);

  const double this_mean = x.calculate_mean();
  const double y_mean = y.calculate_mean();

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

  return(cross_correlation);
}


/// Calculate the corresponding correlation between two vectors.
/// It automatically select the right correlation type.
/// @param x Vector containing data.
/// @param y Vector for computing the linear correlation with this vector.

double CorrelationAnalysis::calculate_correlation(const Vector<double>& x, const Vector<double>& y)
{
    if (x.is_constant()) return 0;
    else if (y.is_constant()) return 0;

    if (x == y) return 1;

    Vector<double> new_x = x;
    Vector<double> new_y = y;
    const bool this_binary = x.is_binary();
    const bool other_binary = y.is_binary();

    if (this_binary)
    {
        Vector<double> unique_elements = new_x.get_unique_elements();

        for (size_t i = 0; i < new_x.size(); i++)
        {
            if (fabs(new_x[i] - unique_elements[0]) < numeric_limits<double>::epsilon())
            {
                new_x[i] = 0;
            }
            else
            {
                new_x[i] = 1;
            }
        }
    }

    if (other_binary)
    {
        Vector<double> unique_elements = new_y.get_unique_elements();

        for (size_t i = 0; i < new_y.size(); i++)
        {
            if (fabs(new_y[i] - unique_elements[0]) < numeric_limits<double>::epsilon())
            {
                new_y[i] = 0;
            }
            else
            {
                new_y[i] = 1;
            }
        }
    }

//    cout << "size: " << new_x.size() << " / " << new_y.size() << endl;
//    cout << "x: " << new_x << endl;
//    cout << "y: " << new_y << endl;

    if(!this_binary && !other_binary)
    {
        return calculate_linear_correlation(new_x, new_y);
    }
    else if(this_binary && other_binary)
    {
        return calculate_logistic_correlation(new_x, new_y);
    }
    else if(this_binary && !other_binary)
    {
        return calculate_logistic_correlation(new_y, new_x);
    }
    else if(!this_binary && other_binary)
    {
        return calculate_logistic_correlation(new_x, new_y);
    }
    else
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: Correlation Analysis.\n"
               << "double calculate_correlation(const Vector<double>&) method.\n"
               << "Unknown case.\n";

        throw logic_error(buffer.str());
    }
}


/// Calculate the correlation between all columns in a matrix.
/// It selects Linear or Logistic automatically.
/// @param input_matrix Matrix containing data.

Matrix<double> CorrelationAnalysis::calculate_correlations(const Matrix<double>& input_matrix)
{
    const size_t columns_number = input_matrix.get_columns_number();


    Matrix<double> correlations_matrix(columns_number, columns_number, 0);

#pragma omp parallel for

    for(int i = 0; i < static_cast<int>(columns_number); i++)
    {
        const Vector<double> x = input_matrix.get_column(static_cast<size_t>(i));
        for(size_t j=static_cast<size_t>(i); j<columns_number; j++)
        {
            const Vector<double> y = input_matrix.get_column(j);
            correlations_matrix(static_cast<size_t>(i),j) = calculate_correlation(x,y);
            correlations_matrix(j,static_cast<size_t>(i)) = correlations_matrix(static_cast<size_t>(i),j);
        }
    }


    return correlations_matrix;
}


/// Calculate the correlation between the specified column and the rest of columns
/// in a matrix.
/// It selects Linear or Logistic automatically.
/// @param input_matrix Matrix containing data.
/// @param index Index of the desired column.

Vector<double> CorrelationAnalysis::calculate_correlations(const Matrix<double>& input_matrix, const size_t& index)
{
    const size_t columns_number = input_matrix.get_columns_number();

    const Vector<double> y = input_matrix.get_column(index);

    Vector<double> correlations(columns_number);

    #pragma omp parallel for
    for(int i = 0; i < static_cast<int>(columns_number); i++)
    {
        Vector<double> x = input_matrix.get_column(static_cast<size_t>(i));

        correlations[static_cast<size_t>(i)] = calculate_correlation(x,y);
    }

    return correlations;
}


/// Calculate the linear correlation between specified columns in a matrix.
/// It selects Linear or Logistic automatically.
/// @param input_matrix Matrix containing data.
/// @param indexs Vector containing the desired columns indexes.

Matrix<double> CorrelationAnalysis::calculate_correlations(const Matrix<double>& input_matrix, const Vector<size_t>& indexes)
{
    const size_t indexes_number = indexes.size();

    Matrix<double> correlations_matrix(indexes_number, indexes_number, 0);

//#pragma omp parallel for
    for(int i = 0; i < static_cast<int>(indexes_number); i++)
    {
        const Vector<double> x =input_matrix.get_column(indexes[static_cast<size_t>(i)]);
        for(size_t j=static_cast<size_t>(i); j<indexes_number; j++)
        {
            const Vector<double> y = input_matrix.get_column(indexes[j]);
            correlations_matrix(static_cast<size_t>(i),j) = calculate_correlation(x,y);
            correlations_matrix(j,static_cast<size_t>(i)) = correlations_matrix(static_cast<size_t>(i),j);
        }
    }

    return correlations_matrix;
}


/// Calculates the linear correlations between each input and each target variable.
/// For nominal input variables it calculates the multiple linear correlation between all the classes of
/// the input variable and the target.
/// @param nominal_variables Vector containing the classes of each nominal variable.

Matrix<double> CorrelationAnalysis::calculate_multiple_linear_correlations(const DataSet& data_set, const Vector<size_t> & nominal_variables)
{
    const Variables& variables = data_set.get_variables();
    const Instances& instances = data_set.get_instances();
    const MissingValues& missing_values = data_set.get_missing_values();

    const Vector<size_t> targets_indices = variables.get_targets_indices();
    const size_t targets_number = variables.get_targets_number();

    const Vector<size_t> used_instances_indices = instances.get_used_indices();

    const size_t inputs_number = data_set.calculate_input_variables_number(nominal_variables);
    const Vector< Vector<size_t> > new_input_indices = data_set.get_inputs_indices(inputs_number, nominal_variables);

    // Calculate correlations

    Matrix<double> multiple_linear_correlations(inputs_number, targets_number);

    Matrix<double> input_variables;
    Vector<size_t> input_indices;

    Vector<double> target_variable;
    size_t target_index;

    for(size_t i = 0; i < inputs_number; i++)
    {
        for(size_t j = 0; j < targets_number; j++)
        {
            input_indices = new_input_indices[i];

            target_index = targets_indices[j];

            Vector<size_t> current_missing_values = missing_values.get_missing_instances(input_indices);

            current_missing_values = current_missing_values.get_union(missing_values.get_missing_instances(target_index));

            const Vector<size_t> current_used_indices = used_instances_indices.get_difference(current_missing_values);

            input_variables = data_set.get_data().get_submatrix(current_used_indices, input_indices);

            target_variable = data_set.get_data().get_column(target_index, current_used_indices);

            if(input_variables.get_columns_number() == 1)
            {
                multiple_linear_correlations(i,j) = calculate_linear_correlation(input_variables.to_vector(), target_variable);
            }
            else
            {
                multiple_linear_correlations(i,j) = calculate_multiple_linear_correlation(input_variables,target_variable);
            }
        }
    }

    return multiple_linear_correlations;
}


/// Calculates the multiple linear correlation coefficient.
/// @param x Vector
/// @param y Independent vector

double CorrelationAnalysis::calculate_multiple_linear_correlation(const Matrix<double>& x, const Vector<double>& y)
{
    // x: input variables
    // y: target variables

    if(x.get_columns_number() == 1) // Simple linear correlation
    {
        return calculate_linear_correlation(x.get_column(0), y);
    }

    const Vector<double> multiple_linear_regression_parameters = x.calculate_multiple_linear_regression_parameters(y);

    cout << "Multiple linear regression parameters: " << multiple_linear_regression_parameters << endl;

    const Vector<double> other_approximation = x.dot(multiple_linear_regression_parameters);

    cout << "other_approximation: " << other_approximation << endl;

    return CorrelationAnalysis::calculate_linear_correlation(y, other_approximation);
}


Vector<double> CorrelationAnalysis::calculate_logistic_error_gradient(const Vector<double>& coefficients, const Vector<double>& x, const Vector<double>& y)
{
    const size_t n = x.size();

    const size_t other_size = x.size();

    Vector<double> error_gradient(3, 0.0);

    size_t negatives_number = 0;
    size_t positives_number = 0;

    for(size_t i = 0; i < other_size; i++)
    {
        if(fabs(y[i] - 1) < numeric_limits<double>::epsilon())
        {
            positives_number++;
        }
        else if(fabs(y[i] - 0) < numeric_limits<double>::epsilon())
        {
            negatives_number++;
        }
    }

    double negatives_weight = 1.0;

    double positives_weight = 1.0;

    if(positives_number == 0)
    {
        positives_weight = 1.0;
        negatives_weight = 1.0;
    }
    else if(negatives_number == 0)
    {
        positives_weight = 1.0;
        negatives_weight = 1.0;

        negatives_number = 1;
    }
    else
    {
        positives_weight = static_cast<double>(negatives_number)/static_cast<double>(positives_number);
    }

#pragma omp parallel for

    for(int i = 0; i < static_cast<int>(n); i++)
    {
        Vector<double> x_corr(1, x[static_cast<size_t>(i)]);

        const double current_logistic_function = calculate_logistic_function(coefficients, x_corr);

        const double gradient_multiply = exp(-(coefficients[0]+coefficients[1]*x_corr[0]))*(y[static_cast<size_t>(i)] - current_logistic_function)*current_logistic_function*current_logistic_function;

        Vector<double> this_error_gradient(3, 0.0);

        this_error_gradient[0] += (y[static_cast<size_t>(i)]*positives_weight + (1-y[static_cast<size_t>(i)])*negatives_weight)*(y[static_cast<size_t>(i)]- current_logistic_function)*(y[static_cast<size_t>(i)] - current_logistic_function)/2;
        this_error_gradient[1] -= (y[static_cast<size_t>(i)]*positives_weight + (1-y[static_cast<size_t>(i)])*negatives_weight)*gradient_multiply;
        this_error_gradient[2] -= (y[static_cast<size_t>(i)]*positives_weight + (1-y[static_cast<size_t>(i)])*negatives_weight)*x_corr[0]*gradient_multiply;

#pragma omp critical
        {
            error_gradient += this_error_gradient;
        }
    }

    return error_gradient/static_cast<double>(negatives_weight*negatives_number);
}


double CorrelationAnalysis::calculate_logistic_function(const Vector<double>& coefficients, const Vector<double>& x)
{
    const size_t coefficients_size = coefficients.size();

    double exponential = coefficients[0];

    for(size_t i = 1; i < coefficients_size; i++)
    {
        exponential += coefficients[i]*x[i-1];
    }

    return(1.0/(1.0+exp(-exponential)));
}


Matrix<double> CorrelationAnalysis::remove_correlations(const Matrix<double>& data, const size_t& index, const double& minimum_correlation)
{
    double new_minimum_correlation = minimum_correlation;

    const Vector<double> correlations = calculate_correlations(data,index).calculate_absolute_value();

    size_t columns_to_keep_number = correlations.count_greater_equal_to(new_minimum_correlation);

    do
    {
        new_minimum_correlation = new_minimum_correlation - 0.05;

        columns_to_keep_number = correlations.count_greater_equal_to(new_minimum_correlation);
    }
    while(columns_to_keep_number < 5);

    const Vector<size_t> indices_to_remove = correlations.get_indices_less_than(new_minimum_correlation);

    return data.delete_columns(indices_to_remove);
}

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2018 Artificial Intelligence Techniques, SL.
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
