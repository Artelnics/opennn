//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   C O R R E L A T I O N S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "perceptron_layer.h"
#include "tensors.h"
#include "correlations.h"
#include "data_set.h"
#include "neural_network.h"
#include "training_strategy.h"
#include "scaling_layer_2d.h"
#include "probabilistic_layer.h"

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

    if(x_columns == 1)
        return logistic_correlation_matrix_vector(thread_pool_device, x, y.reshape(vector));

    if(y_columns == 1)
        return logistic_correlation_vector_matrix(thread_pool_device, x.reshape(vector), y);

    if(x_columns != 1 && y_columns != 1)
        return logistic_correlation_matrix_matrix(thread_pool_device, x, y);

    throw runtime_error("Correlations Exception: Unknown case.");
}


Tensor<type, 1> cross_correlations(const ThreadPoolDevice* thread_pool_device,
                                   const Tensor<type, 1>& x,
                                   const Tensor<type, 1>& y,
                                   const Index& maximum_lags_number)
{
    cout << "a" << endl;
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
    cout << "cross correlatio calculate" << endl;
    cout << cross_correlation << endl;
    cout << "-----------------------" << endl;

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
    // @todo Improve this method to be more similar to the other code.

    const int n = x.size();

    vector<pair<type, size_t> > sorted_vector(n);

    for(size_t i = 0U; i < n; i++)
        sorted_vector[i] = make_pair(x[i], i);

    sort(sorted_vector.begin(), sorted_vector.end());

    vector<type> x_rank_vector(n);

    type rank = type(1);

    for(size_t i = 0; i < n; i++)
    {
        size_t repeated = 1U;

        for(size_t j = i + 1U; j < sorted_vector.size() && sorted_vector[j].first == sorted_vector[i].first; j++, repeated++);

        for(size_t k = 0; k < repeated; k++)
            x_rank_vector[sorted_vector[i + k].second] = rank + type(repeated - 1) / type(2);

        i += repeated - 1;

        rank += type(repeated);
    }

    TensorMap<Tensor<type, 1>> x_rank(x_rank_vector.data(), x_rank_vector.size());

    return x_rank;
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

    const Tensor<type, 2> data = opennn::assemble_vector_vector(x_filtered, y_filtered);

    DataSet data_set(x_filtered.size(), {1}, {1});
    data_set.set_data(data);
    data_set.set(DataSet::SampleUse::Training);
    data_set.set_raw_variable_scalers(Scaler::MinimumMaximum);

    NeuralNetwork neural_network;
    dimensions dim1 = { 1 };
    dimensions dim2 = { 1 };
    neural_network.add_layer(make_unique<Scaling2d>(dim1));
    neural_network.add_layer(make_unique<Perceptron>(dim1, dim2, Perceptron::Activation::Logistic));

    neural_network.set_parameters_constant(type(0.001));

    TrainingStrategy training_strategy(&neural_network, &data_set);
    training_strategy.set_display(false);

    training_strategy.set_loss_method(TrainingStrategy::LossMethod::MEAN_SQUARED_ERROR);

    training_strategy.set_optimization_method(TrainingStrategy::OptimizationMethod::LEVENBERG_MARQUARDT_ALGORITHM);

    training_strategy.get_loss_index()->set_regularization_method(LossIndex::RegularizationMethod::NoRegularization);

    training_strategy.set_maximum_epochs_number(1000);

    training_strategy.perform_training();

    const Tensor<type, 2> inputs = data_set.get_data(DataSet::VariableUse::Input);

    const Tensor<type, 2> targets = data_set.get_data(DataSet::VariableUse::Target);

    const Tensor<type, 2> outputs = neural_network.calculate_outputs(inputs);

    // Logistic correlation

    const array<Index, 1> vector{{x_filtered.size()}};

    correlation.r = linear_correlation(thread_pool_device, outputs.reshape(vector), targets.reshape(vector)).r;

    const type z_correlation = r_correlation_to_z_correlation(correlation.r);

    const Tensor<type, 1> confidence_interval_z = confidence_interval_z_correlation(z_correlation, inputs.dimensions()[0]);

    correlation.lower_confidence = z_correlation_to_r_correlation(confidence_interval_z(0));

    correlation.upper_confidence = z_correlation_to_r_correlation(confidence_interval_z(1));

    correlation.form = Correlation::Form::Logistic;

    const Tensor<type, 1> coefficients = neural_network.get_parameters();

    correlation.a = coefficients(1);
    correlation.b = coefficients(0);

    // if(correlation.a < type(0))
    //     correlation.r *= type(-1);

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

    DataSet data_set(x_filtered.size(), {1}, {1});

    data_set.set_data(data);

    data_set.set(DataSet::SampleUse::Training);

    data_set.set_raw_variable_scalers(Scaler::MinimumMaximum);

    // NeuralNetwork neural_network(NeuralNetwork::ModelType::Classification, {1}, {}, {1});

    // Scaling2d* scaling_layer_2d = static_cast<Scaling2d*>(neural_network.get_first(Layer::Type::Scaling2d));

    // scaling_layer_2d->set_display(false);

    // Probabilistic* probabilistic_layer = static_cast<Probabilistic*>(neural_network.get_first(Layer::Type::Probabilistic));

    // probabilistic_layer->set_activation_function(Probabilistic::Activation::Logistic);

    NeuralNetwork neural_network;
    dimensions dim1 = { 1 };
    dimensions dim2 = { 1 };
    neural_network.add_layer(make_unique<Scaling2d>(dim1));
    neural_network.add_layer(make_unique<Perceptron>(dim1, dim2, Perceptron::Activation::Logistic));

    TrainingStrategy training_strategy(&neural_network, &data_set);
    training_strategy.set_display(false);

    training_strategy.set_loss_method(TrainingStrategy::LossMethod::MEAN_SQUARED_ERROR);

    training_strategy.set_optimization_method(TrainingStrategy::OptimizationMethod::LEVENBERG_MARQUARDT_ALGORITHM);

    training_strategy.get_loss_index()->set_regularization_method(LossIndex::RegularizationMethod::NoRegularization);

    training_strategy.perform_training();

    const Tensor<type, 2> inputs = data_set.get_data(DataSet::VariableUse::Input);

    const Tensor<type, 2> targets = data_set.get_data(DataSet::VariableUse::Target);

    const Tensor<type, 2> outputs = neural_network.calculate_outputs(inputs);

    // Logistic correlation

    const array<Index, 1> vector{{x_filtered.size()}};

    correlation.r = linear_correlation(thread_pool_device, outputs.reshape(vector), targets.reshape(vector)).r;

    const type z_correlation = r_correlation_to_z_correlation(correlation.r);

    const Tensor<type, 1> confidence_interval_z = confidence_interval_z_correlation(z_correlation, x_rank.size());

    correlation.lower_confidence = z_correlation_to_r_correlation(confidence_interval_z(0));

    correlation.upper_confidence = z_correlation_to_r_correlation(confidence_interval_z(1));

    correlation.form = Correlation::Form::Logistic;

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
 
    DataSet data_set(x_filtered.size(), {1}, {y_filtered.dimension(1)});

    data_set.set_data(data);
    // data_set.set_raw_variable_indices(input_columns_indices, target_columns_indices);
    data_set.set_binary_raw_variables();
    data_set.set_default_raw_variables_scalers();


    // data_set.print();

    data_set.set(DataSet::SampleUse::Training);

    const Index input_variables_number = data_set.get_variables_number(DataSet::VariableUse::Input);
    const Index target_variables_number = data_set.get_variables_number(DataSet::VariableUse::Target);

    NeuralNetwork neural_network(NeuralNetwork::ModelType::Classification,
                                 { input_variables_number }, {1}, {target_variables_number});

    // Scaling2d* scaling_layer_2d = static_cast<Scaling2d*>(neural_network.get_first(Layer::Type::Scaling2d));

    // Probabilistic* probabilistic_layer = static_cast<Probabilistic*>(neural_network.get_first(Layer::Type::Probabilistic));

    // probabilistic_layer->set_activation_function(Probabilistic::Activation::Softmax);
    // scaling_layer_2d->set_display(false);

    TrainingStrategy training_strategy(&neural_network, &data_set);

    training_strategy.set_optimization_method(TrainingStrategy::OptimizationMethod::ADAPTIVE_MOMENT_ESTIMATION);

    training_strategy.set_loss_method(TrainingStrategy::LossMethod::CROSS_ENTROPY_ERROR);

    training_strategy.get_loss_index()->set_regularization_method(LossIndex::RegularizationMethod::NoRegularization);

    training_strategy.set_display(false);

    training_strategy.set_display_period(1000);

    training_strategy.perform_training();

    // Logistic correlation

    const Tensor<type, 2> inputs = data_set.get_data(DataSet::VariableUse::Input);

    const Tensor<type, 2> targets = data_set.get_data(DataSet::VariableUse::Target);

    const Tensor<type, 2> outputs = neural_network.calculate_outputs(inputs);

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

    DataSet data_set(x_filtered.dimension(0), { x_filtered.dimension(1) }, { x_filtered.dimension(1) });

    data_set.set_data(data);

    data_set.set_raw_variable_indices(input_columns_indices, target_columns_indices);

    data_set.set(DataSet::SampleUse::Training);

    const Index input_variables_number = data_set.get_variables_number(DataSet::VariableUse::Input);
    const Index target_variables_number = data_set.get_variables_number(DataSet::VariableUse::Target);

    NeuralNetwork neural_network(NeuralNetwork::ModelType::Classification,
                                 {input_variables_number }, {}, {target_variables_number});

    Scaling2d* scaling_layer_2d = static_cast<Scaling2d*>(neural_network.get_first(Layer::Type::Scaling2d));

    Probabilistic* probabilistic_layer = static_cast<Probabilistic*>(neural_network.get_first(Layer::Type::Probabilistic));

    probabilistic_layer->set_activation_function(Probabilistic::Activation::Softmax);

    scaling_layer_2d->set_display(false);

    TrainingStrategy training_strategy(&neural_network, &data_set);

    training_strategy.get_loss_index()->set_regularization_method(LossIndex::RegularizationMethod::NoRegularization);

    training_strategy.set_optimization_method(TrainingStrategy::OptimizationMethod::LEVENBERG_MARQUARDT_ALGORITHM);

    training_strategy.set_loss_method(TrainingStrategy::LossMethod::MEAN_SQUARED_ERROR);

    training_strategy.set_maximum_epochs_number(500);

    training_strategy.set_display(false);

    training_strategy.perform_training();

    // Logistic correlation

    const Tensor<type, 2> inputs = data_set.get_data(DataSet::VariableUse::Input);

    const Tensor<type, 2> targets = data_set.get_data(DataSet::VariableUse::Target);

    const Tensor<type, 2> outputs = neural_network.calculate_outputs(inputs);

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
