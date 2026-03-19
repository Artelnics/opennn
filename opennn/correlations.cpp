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
#include "standard_networks.h"
#include "bounding_layer.h"
#include "multihead_attention_layer.h"
#include "recurrent_layer.h"
#include "mean_squared_error.h"
#include "cross_entropy_error.h"
#include "stochastic_gradient_descent.h"
#include "adaptive_moment_estimation.h"
#include "quasi_newton_method.h"
#include "levenberg_marquardt_algorithm.h"
#include "standard_networks.h"

namespace opennn
{

VectorR autocorrelations(const VectorR& x, Index past_time_steps)
{
    VectorR autocorrelation(past_time_steps);
    const Index this_size = x.size();

    for(Index i = 0; i < past_time_steps; i++)
        autocorrelation(i) = linear_correlation(x.head(this_size - i), x.tail(this_size - i)).r;

    return autocorrelation;
}


Correlation correlation(const MatrixR& x,
                        const MatrixR& y)
{
    if(is_constant(x) || is_constant(y))
        return Correlation();

    const Index x_rows = x.rows();
    const Index x_columns = x.cols();
    const Index y_columns = y.cols();

    const bool x_binary = is_binary(x);
    const bool y_binary = is_binary(y);

    if(x_columns == 1 && y_columns == 1)
    {
        const VectorR x_vector = x.reshaped(x_rows, 1);
        const VectorR y_vector = y.reshaped(x_rows, 1);

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

        if(!x_binary && y_binary)
            return opennn::point_biserial_correlation(x_vector, y_vector);

        if(x_binary && !y_binary)
            return opennn::point_biserial_correlation(y_vector, x_vector);

        if(x_binary && y_binary)
            return opennn::linear_correlation(x_vector, y_vector);
    }

    if(x_columns != 1 && y_columns == 1)
        return eta_squared_correlation(y.reshaped(x_rows, 1), x);

    if(x_columns == 1 && y_columns != 1)
        return eta_squared_correlation(x.reshaped(x_rows, 1), y);

    if(x_columns != 1 && y_columns != 1)
        return logistic_correlation_matrix_matrix(x, y);

    throw runtime_error("Correlations Exception: Unknown case.");
}


Correlation correlation_spearman(const MatrixR& x, const MatrixR& y)
{
    const Index x_rows = x.rows();
    const Index x_columns = x.cols();
    const Index y_columns = y.cols();

    const bool x_binary = is_binary(x);
    const bool y_binary = is_binary(y);

    if(x_columns == 1 && y_columns == 1)
    {
        const VectorR x_vector = x.reshaped(x_rows, 1);
        const VectorR y_vector = y.reshaped(x_rows, 1);

        if(!x_binary && !y_binary)
            return linear_correlation_spearman(x_vector, y_vector);
        else if(!x_binary && y_binary)
            return logistic_correlation_vector_vector_spearman(x_vector, y_vector);
        else if(x_binary && !y_binary)
            return logistic_correlation_vector_vector_spearman(y_vector, x_vector);
        else if(x_binary && y_binary)
            return linear_correlation_spearman(x_vector, y_vector);
    }

    if(x_columns == 1 && y_columns != 1)
        return logistic_correlation_vector_matrix(x.reshaped(x_rows, 1), y);

    if(x_columns != 1 && y_columns == 1)
        return logistic_correlation_matrix_vector(x, y.reshaped(x_rows, 1));

    if(x_columns != 1 && y_columns != 1)
        return logistic_correlation_matrix_matrix(x, y);

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

    for(Index i = 0; i < maximum_past_time_steps; i++)
    {
        VectorR column_x(this_size-i);
        VectorR column_y(this_size-i);

        for(Index j = 0; j < this_size - i; j++)
        {
            column_x(j) = x(j);
            column_y(j) = y(j + i);
        }

        cross_correlation[i] = linear_correlation(column_x, column_y).r;
    }

    return cross_correlation;
}


Correlation exponential_correlation(const VectorR& x, const VectorR& y)
{
    Correlation exponential_correlation;

    for(Index i = 0; i < y.rows(); i++)
    {
        if(!isnan(y(i)) && y(i) <= 0.0f)
        {
            exponential_correlation.r = static_cast<type>(NAN);
            return exponential_correlation;
        }
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

    for(Index i = 0; i < rows_number; i++)
        for(Index j = 0; j < columns_number; j++)
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

    const double s_x = x_filter.cast<double>().sum();
    const double s_y = y_filter.cast<double>().sum();
    const double s_xx = x_filter.cast<double>().squaredNorm();
    const double s_yy = y_filter.cast<double>().squaredNorm();
    const double s_xy = x_filter.cast<double>().dot(y_filter.cast<double>());

    const double denominator = sqrt((double(n) * s_xx - s_x * s_x) * (double(n) * s_yy - s_y * s_y));

    if (denominator < static_cast<double>(EPSILON))
        return Correlation();

    Correlation linear_correlation;
    linear_correlation.form = Correlation::Form::Linear;
    linear_correlation.a = static_cast<type>((s_y * s_xx - s_x * s_xy) / (double(n) * s_xx - s_x * s_x));
    linear_correlation.b = static_cast<type>((double(n) * s_xy - s_x * s_y) / (double(n) * s_xx - s_x * s_x));
    linear_correlation.r = static_cast<type>((double(n) * s_xy - s_x * s_y) / denominator);

    const type z_correlation = r_correlation_to_z_correlation(linear_correlation.r);

    const VectorR confidence_interval_z = confidence_interval_z_correlation(z_correlation, n);

    linear_correlation.lower_confidence = clamp(z_correlation_to_r_correlation(confidence_interval_z(0)), type(-1), type(1));
    linear_correlation.upper_confidence = clamp(z_correlation_to_r_correlation(confidence_interval_z(1)), type(-1), type(1));
    linear_correlation.r = clamp(linear_correlation.r, type(-1), type(1));

    return linear_correlation;
}

/*
type r_correlation_to_z_correlation(const type r_correlation)
{
    return type(0.5 * log((1 + r_correlation) / (1 - r_correlation)));
}*/
// ANTES
// DESPUÉS
type r_correlation_to_z_correlation(const type r_correlation)
{
    cout << "[Z-CONV] r_correlation entrada = " << r_correlation << endl;

    const type r_clamped = clamp(r_correlation, type(-0.9999), type(0.9999));

    if(r_clamped != r_correlation)
        cout << "[Z-CONV] CLAMP aplicado: " << r_correlation << " -> " << r_clamped << endl;

    const type result = type(0.5 * log((1 + r_clamped) / (1 - r_clamped)));
    cout << "[Z-CONV] z resultado = " << result << endl;

    return result;
}


type z_correlation_to_r_correlation (const type z_correlation)
{
    return type((exp(2 * z_correlation) - 1) / (exp(2 * z_correlation) + 1));
}


VectorR confidence_interval_z_correlation(const type z_correlation, Index n)
{
    VectorR confidence_interval(2);

    const type z_standard_error = type(1.959964);

    confidence_interval(0) = z_correlation - z_standard_error * type(1/sqrt(n - 3));
    confidence_interval(1) = z_correlation + z_standard_error * type(1/sqrt(n - 3));

    return confidence_interval;
}


VectorR calculate_spearman_ranks(const VectorR& x)
{
    const Index n = x.size();

    if (n == 0)
        return VectorR();

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
            j++;

        const type average_rank = (static_cast<type>(i + 1) + static_cast<type>(j + 1)) / type(2.0);

        for(Index k = i; k <= j; k++)
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

    for(Index i = 0; i < x.rows(); i++)
    {
        if(!isnan(x(i)) && x(i) <= type(0))
        {
            logarithmic_correlation.r = type(NAN);

            return logarithmic_correlation;
        }
    }

    logarithmic_correlation = linear_correlation(x.array().log(), y);

    logarithmic_correlation.form = Correlation::Form::Logarithmic;

    return logarithmic_correlation;
}



Correlation logistic_correlation_vector_vector(const VectorR& x,
                                               const VectorR& y)
{
    Correlation correlation;

    const auto [x_filter, y_filter] = filter_missing_values(x,y);

     cout<<"x_filter.size()"<<x_filter.size()<<endl;
     cout<<"is_constant(x_filter)"<<is_constant(x_filter)<<endl;
     cout<<"is_constant(y_filter)"<<is_constant(y_filter)<<endl;
   //  cout<<"y_filter"<<y_filter<<endl;
   //  cout<<"x_filter"<<x_filter<<endl;

    cout<<"001"<<endl;

    if (x_filter.size() < 2
        || is_constant(x_filter)
        || is_constant(y_filter))
    {
        cout<<"002"<<endl;
       // cout<<"x_filter.size()"<<x_filter.size()<<endl;
       // cout<<"is_constant(x_filter)"<<is_constant(x_filter)<<endl;
       // cout<<"is_constant(y_filter)"<<is_constant(y_filter)<<endl;

        correlation.r = type(NAN);
        correlation.form = Correlation::Form::Sigmoid;
        return correlation;
    }
    cout<<"003"<<endl;


    MatrixR data(x_filter.size(), 2);
    data.col(0) = x_filter;
    data.col(1) = y_filter;

    Dataset dataset(x_filter.size(), {1}, {1});
    dataset.set_data(data);
    dataset.set_sample_roles("Training");
    dataset.set_variable_scalers("MeanStandardDeviation");
    dataset.set_shape("Input", {1});
    dataset.set_shape("Target", {1});
    dataset.set_display(false);
    cout<<"005"<<endl;

    NeuralNetwork neural_network;
    Shape dim1 = { 1 };
    Shape dim2 = { 1 };
    neural_network.add_layer(make_unique<Scaling<2>>(dim1));
    neural_network.add_layer(make_unique<Dense<2>>(dim1, dim2, "Sigmoid"));
    cout<<"006"<<endl;

    MeanSquaredError mean_squared_error(&neural_network, &dataset);
    mean_squared_error.set_regularization_method("None");

    LevenbergMarquardtAlgorithm levenberg_marquardt_algorithm(&mean_squared_error);
    levenberg_marquardt_algorithm.set_display(false);
    cout<<"007"<<endl;

    try
    {
        levenberg_marquardt_algorithm.train();
    }
    catch(const exception& e)
    {
        cout << "[LOGISTIC] train() excepcion capturada: " << e.what() << endl;
        cout << "[LOGISTIC] devolviendo r=0 para este par" << endl;
        correlation.r = type(0);
        correlation.form = Correlation::Form::Sigmoid;
        return correlation;
    }
    cout<<"008"<<endl;

    cout<<"[POST-008-A] get_feature_data Input..."<<endl;
    const MatrixR inputs = dataset.get_feature_data("Input");

    cout<<"[POST-008-B] get_feature_data Target..."<<endl;
    const MatrixR targets = dataset.get_feature_data("Target");

    cout<<"[POST-008-C] calculate_outputs..."<<endl;
    const MatrixR outputs = neural_network.calculate_outputs(inputs);

    cout<<"[POST-008-D] outputs.rows="<<outputs.rows()<<" cols="<<outputs.cols()<<endl;
    cout<<"[POST-008-D] outputs tiene NaN="<<outputs.hasNaN()<<endl;
    cout<<"[POST-008-D] targets tiene NaN="<<targets.hasNaN()<<endl;

    cout<<"[POST-008-E] linear_correlation..."<<endl;
    correlation.r = linear_correlation(outputs.reshaped(), targets.reshaped()).r;
    cout<<"[POST-008-F] r = "<<correlation.r<<endl;
    cout << "[LOGISTIC] correlation.r tras linear_correlation = " << correlation.r << endl;

    if(isnan(correlation.r) || !isfinite(correlation.r))
    {
        cout << "[LOGISTIC] GUARD ACTIVADO -> r era NaN/Inf, devolviendo 0" << endl;
        correlation.r = type(0);
        correlation.form = Correlation::Form::Sigmoid;
        return correlation;
    }
    cout << "[LOGISTIC] guard no activado, continuando..." << endl;
    const type z_correlation = r_correlation_to_z_correlation(correlation.r);

    const VectorR confidence_interval_z = confidence_interval_z_correlation(z_correlation, inputs.rows());

    correlation.lower_confidence = z_correlation_to_r_correlation(confidence_interval_z(0));

    correlation.upper_confidence = z_correlation_to_r_correlation(confidence_interval_z(1));

    correlation.form = Correlation::Form::Sigmoid;

    const VectorR coefficients = neural_network.get_parameters();

    correlation.a = coefficients(0);
    correlation.b = coefficients(1);

    if(correlation.b < type(0))
    {
        correlation.r *= type(-1);
        type old_lower = correlation.lower_confidence;
        correlation.lower_confidence = -correlation.upper_confidence;
        correlation.upper_confidence = -old_lower;
    }

    return correlation;
}


Correlation logistic_correlation_vector_vector_spearman(const VectorR& x,
                                                        const VectorR& y)
{
    Correlation correlation;

    const auto [x_filter, y_filter] = filter_missing_values(x, y);

    if(x_filter.size() == 0)
    {
        correlation.r = type(NAN);

        correlation.form = Correlation::Form::Sigmoid;

        return correlation;
    }

    const VectorR x_rank = calculate_spearman_ranks(x_filter);

    MatrixR data;
    data << x_rank, y_filter;

    Dataset dataset(x_filter.size(), {1}, {1});
    dataset.set_data(data);
    dataset.set_sample_roles("Training");
    dataset.set_variable_scalers("MinimumMaximum");

    NeuralNetwork neural_network;
    Shape dim1 = { 1 };
    Shape dim2 = { 1 };
    neural_network.add_layer(make_unique<Scaling<2>>(dim1));
    neural_network.add_layer(make_unique<Dense<2>>(dim1, dim2, "Sigmoid"));

    MeanSquaredError mean_squared_error(&neural_network, &dataset);
    mean_squared_error.set_regularization_method("None");

    LevenbergMarquardtAlgorithm levenberg_marquardt_algorithm(&mean_squared_error);
    levenberg_marquardt_algorithm.set_display(false);

    try
    {
        levenberg_marquardt_algorithm.train();
    }
    catch(const exception& e)
    {
        cout << "[LOGISTIC-SPEARMAN] train() excepcion: " << e.what() << endl;
        correlation.r = type(0);
        correlation.form = Correlation::Form::Sigmoid;
        return correlation;
    }
    const MatrixR inputs = dataset.get_feature_data("Input");
    const MatrixR targets = dataset.get_feature_data("Target");
    const MatrixR outputs = neural_network.calculate_outputs(inputs);

    // Sigmoid correlation

    correlation.r = linear_correlation(outputs.reshaped(), targets.reshaped()).r;

    const type z_correlation = r_correlation_to_z_correlation(correlation.r);

    const VectorR confidence_interval_z = confidence_interval_z_correlation(z_correlation, x_rank.size());

    correlation.lower_confidence = z_correlation_to_r_correlation(confidence_interval_z(0));

    correlation.upper_confidence = z_correlation_to_r_correlation(confidence_interval_z(1));

    correlation.form = Correlation::Form::Sigmoid;

    const VectorR& coefficients = neural_network.get_parameters();

    correlation.a = coefficients(0);
    correlation.b = coefficients(1);

    if(correlation.b < type(0))
    {
        correlation.r *= type(-1);
        type old_lower = correlation.lower_confidence;
        correlation.lower_confidence = -correlation.upper_confidence;
        correlation.upper_confidence = -old_lower;
    }

    return correlation;
}


Correlation logistic_correlation_vector_matrix(const VectorR& x, const MatrixR& y)
{
    Correlation correlation;
    correlation.form = Correlation::Form::Sigmoid;

    const auto [x_filter, y_filter] = filter_missing_values(x, y);

    if(y_filter.cols() > 50)
    {
        cout << "Warning: Y variable has too many categories." << endl;

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

    // Dataset.print();

    dataset.set_sample_roles("Training");
    dataset.set_shape("Input", {dataset.get_features_number("Input")});
    dataset.set_shape("Target", {dataset.get_features_number("Target")});

    const Index input_features_number = dataset.    get_features_number("Input");
    const Index target_features_number = dataset.get_features_number("Target");

    ClassificationNetwork neural_network({ input_features_number }, {1}, {target_features_number});

    Scaling<2>* scaling_layer = static_cast<Scaling<2>*>(neural_network.get_first("Scaling2d"));

    Dense<2>* dense_2d = static_cast<Dense<2>*>(neural_network.get_first("Dense2d"));

    dense_2d->set_activation_function("Softmax");
    scaling_layer->set_display(false);

    CrossEntropyError2d cross_entropy_error_2d(&neural_network, &dataset);
    cross_entropy_error_2d.set_regularization_method("None");

    QuasiNewtonMethod quasi_newton_method(&cross_entropy_error_2d);
    quasi_newton_method.set_display(false);
    quasi_newton_method.set_display_period(1000);

    try
    {
        quasi_newton_method.train();
    }
    catch(const exception& e)
    {
        cout << "[LOGISTIC-VEC-MAT] train() excepcion: " << e.what() << endl;
        correlation.r = type(0);
        correlation.form = Correlation::Form::Sigmoid;
        return correlation;
    }
    // Sigmoid correlation

    const MatrixR inputs = dataset.get_feature_data("Input");
    const MatrixR targets = dataset.get_feature_data("Target");

    const MatrixR outputs = neural_network.calculate_outputs(inputs);

    correlation.r = linear_correlation(outputs.reshaped(), targets.reshaped()).r;

    const type z_correlation = r_correlation_to_z_correlation(correlation.r);

    const VectorR confidence_interval_z = confidence_interval_z_correlation(z_correlation, x_filter.size());

    correlation.lower_confidence = z_correlation_to_r_correlation(confidence_interval_z(0));

    correlation.upper_confidence = z_correlation_to_r_correlation(confidence_interval_z(1));

    return correlation;
}


Correlation logistic_correlation_matrix_vector(const MatrixR& y, const VectorR& x)
{
    return logistic_correlation_vector_matrix(x, y);
}


Correlation logistic_correlation_matrix_matrix(const MatrixR& x, const MatrixR& y)
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
        cout << "Warning: One variable has too many categories." << endl;

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

    Scaling<2>* scaling_layer = static_cast<Scaling<2>*>(neural_network.get_first("Scaling2d"));

    Dense<2>* dense_2d = static_cast<Dense<2>*>(neural_network.get_first("Dense2d"));

    dense_2d->set_activation_function("Softmax");

    scaling_layer->set_display(false);

    MeanSquaredError mean_squared_error(&neural_network, &dataset);
    mean_squared_error.set_regularization_method("None");

    QuasiNewtonMethod quasi_newton_method(&mean_squared_error);
    quasi_newton_method.set_maximum_epochs(500);
    quasi_newton_method.set_display(false);
    try
    {
        quasi_newton_method.train();
    }
    catch(const exception& e)
    {
        cout << "[LOGISTIC-MAT-MAT] train() excepcion: " << e.what() << endl;
        correlation.r = type(0);
        correlation.form = Correlation::Form::Sigmoid;
        return correlation;
    }
    // Sigmoid correlation

    const MatrixR inputs = dataset.get_feature_data("Input");

    const MatrixR targets = dataset.get_feature_data("Target");

    const MatrixR outputs = neural_network.calculate_outputs(inputs);

    correlation.r = linear_correlation(outputs.reshaped(), targets.reshaped()).r;

    const type z_correlation = r_correlation_to_z_correlation(correlation.r);

    const VectorR confidence_interval_z = confidence_interval_z_correlation(z_correlation, inputs.rows());

    correlation.lower_confidence = z_correlation_to_r_correlation(confidence_interval_z(0));

    correlation.upper_confidence = z_correlation_to_r_correlation(confidence_interval_z(1));

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

    double sum_all = 0, sum_sq = 0;
    double sum1 = 0, sum0 = 0;
    Index  n1 = 0,   n0 = 0;

    for(Index i = 0; i < n; i++)
    {
        const double xi = double(x_filter(i));
        sum_all += xi;
        sum_sq  += xi * xi;

        if(y_filter(i) > type(0.5)) { sum1 += xi; n1++; }
        else                        { sum0 += xi; n0++; }
    }

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

    const type z  = r_correlation_to_z_correlation(result.r);
    const VectorR ci = confidence_interval_z_correlation(z, n);
    result.lower_confidence = z_correlation_to_r_correlation(ci(0));
    result.upper_confidence = z_correlation_to_r_correlation(ci(1));

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

    // Media global
    const double grand_mean = x_filter.cast<double>().mean();

    // SS_total
    double ss_total = 0;
    for(Index i = 0; i < n; i++)
    {
        const double diff = double(x_filter(i)) - grand_mean;
        ss_total += diff * diff;
    }

    if(ss_total <= 0)
    {
        result.r = type(0);
        return result;
    }

    // SS_between — suma de cuadrados entre grupos
    double ss_between = 0;

    for(Index cat = 0; cat < n_cats; cat++)
    {
        double group_sum   = 0;
        Index  group_count = 0;

        for(Index i = 0; i < n; i++)
        {
            if(y_filter(i, cat) > type(0.5))
            {
                group_sum += double(x_filter(i));
                group_count++;
            }
        }

        if(group_count == 0) continue;

        const double group_mean = group_sum / double(group_count);
        const double diff       = group_mean - grand_mean;
        ss_between += double(group_count) * diff * diff;
    }

    // eta² = SS_between / SS_total
    const double eta_sq = ss_between / ss_total;

    // Convertimos a r equivalente: r = sqrt(eta²)
    // con signo siempre positivo (magnitud de asociación)
    result.r = type(clamp(sqrt(eta_sq), 0.0, 1.0));

    // Intervalo de confianza via Fisher z
    const type z      = r_correlation_to_z_correlation(result.r);
    const VectorR ci  = confidence_interval_z_correlation(z, n);
    result.lower_confidence = z_correlation_to_r_correlation(ci(0));
    result.upper_confidence = z_correlation_to_r_correlation(ci(1));

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


string Correlation::write_type() const
{
    switch(form)
    {
    case Form::Linear: return "linear";
    case Form::Sigmoid: return "logistic";
    case Form::Logarithmic: return "logarithmic";
    case Form::Exponential: return "exponential";
    case Form::Power: return "power";
    default: return string();
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
    Bounding bounding_layer;//bounding_layer.print();
    MultiHeadAttention multi_head_attention; multi_head_attention.print();
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
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// This library is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 2.1 of the License, or any later version.
// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.
// You should have received a copy of the GNU Lesser General Public
// License along with this library; if not, write to the Free Software
// Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
