//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   T E S T I N G   A N A L Y S I S   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "testing_analysis.h"
#include "tensors.h"
#include "correlations.h"
#include "language_data_set.h"
#include "transformer.h"
#include "statistics.h"
#include "unscaling_layer.h"
#include "probabilistic_layer.h"

namespace opennn
{

TestingAnalysis::TestingAnalysis(NeuralNetwork* new_neural_network, DataSet* new_data_set)
{
    neural_network = new_neural_network;
    data_set = new_data_set;

    const unsigned int threads_number = thread::hardware_concurrency();

    thread_pool = make_unique<ThreadPool>(threads_number);
    thread_pool_device = make_unique<ThreadPoolDevice>(thread_pool.get(), threads_number);

}


NeuralNetwork* TestingAnalysis::get_neural_network() const
{
    return neural_network;
}


DataSet* TestingAnalysis::get_data_set() const
{
    return data_set;
}


const bool& TestingAnalysis::get_display() const
{
    return display;
}


void TestingAnalysis::set_threads_number(const int& new_threads_number)
{
    thread_pool = make_unique<ThreadPool>(new_threads_number);
    thread_pool_device = make_unique<ThreadPoolDevice>(thread_pool.get(), new_threads_number);
}


void TestingAnalysis::set_neural_network(NeuralNetwork* new_neural_network)
{
    neural_network = new_neural_network;
}


void TestingAnalysis::set_data_set(DataSet* new_data_set)
{
    data_set = new_data_set;
}


void TestingAnalysis::set_display(const bool& new_display)
{
    display = new_display;
}


void TestingAnalysis::check() const
{
    if(!neural_network)
        throw runtime_error("Neural network pointer is nullptr.\n");

    if(!data_set)
        throw runtime_error("Data set pointer is nullptr.\n");
}


Tensor<Correlation, 1> TestingAnalysis::linear_correlation() const
{
    const Tensor<type, 2> inputs = data_set->get_data(DataSet::SampleUse::Testing, DataSet::VariableUse::Input);

    const Tensor<type, 2> targets = data_set->get_data(DataSet::SampleUse::Testing, DataSet::VariableUse::Target);

    const Tensor<type, 2> outputs = neural_network->calculate_outputs(inputs);

    return linear_correlation(targets, outputs);
}


Tensor<Correlation, 1> TestingAnalysis::linear_correlation(const Tensor<type, 2>& target, const Tensor<type, 2>& output) const
{
    const Index outputs_number = data_set->get_variables_number(DataSet::VariableUse::Target);

    Tensor<Correlation, 1> linear_correlation(outputs_number);

    for(Index i = 0; i < outputs_number; i++)
        linear_correlation(i) = opennn::linear_correlation(thread_pool_device.get(), output.chip(i,1), target.chip(i,1));

    return linear_correlation;
}


void TestingAnalysis::print_linear_correlations() const
{
    const Tensor<Correlation, 1> linear_correlations = linear_correlation();

    const vector<string> targets_name = data_set->get_variable_names(DataSet::VariableUse::Target);

    const Index targets_number = linear_correlations.size();

    for(Index i = 0; i < targets_number; i++)
        cout << targets_name[i] << " correlation: " << linear_correlations[i].r << endl;
}


Tensor<TestingAnalysis::GoodnessOfFitAnalysis, 1> TestingAnalysis::perform_goodness_of_fit_analysis() const
{

    check();

    // Data set

    const Index testing_samples_number = data_set->get_samples_number(DataSet::SampleUse::Testing);

    if(testing_samples_number == 0)
        throw runtime_error("Number of testing samples is zero.\n");

    const Tensor<type, 2> testing_input_data = data_set->get_data(DataSet::SampleUse::Testing, DataSet::VariableUse::Input);

    const Tensor<type, 2> testing_target_data = data_set->get_data(DataSet::SampleUse::Testing, DataSet::VariableUse::Target);

    // Neural network

    const Index outputs_number = neural_network->get_outputs_number();

    const Tensor<type, 2> testing_output_data = neural_network->calculate_outputs(testing_input_data);

    // Testing analysis

    Tensor<GoodnessOfFitAnalysis, 1> goodness_of_fit_results(outputs_number);

    for(Index i = 0;  i < outputs_number; i++)
    {
        const TensorMap<Tensor<type, 1>> targets = tensor_map(testing_target_data, i);
        const TensorMap<Tensor<type, 1>> outputs = tensor_map(testing_output_data, i);

        const type determination = calculate_determination(outputs, targets);

        goodness_of_fit_results[i].set(targets, outputs, determination);
    }

    return goodness_of_fit_results;
}


void TestingAnalysis::print_goodness_of_fit_analysis() const
{
    const Tensor<GoodnessOfFitAnalysis, 1> linear_regression_analysis = perform_goodness_of_fit_analysis();

    for(Index i = 0; i < linear_regression_analysis.size(); i++)
        linear_regression_analysis(i).print();
}


Tensor<type, 3> TestingAnalysis::calculate_error_data() const
{
    const Index testing_samples_number = data_set->get_samples_number(DataSet::SampleUse::Testing);

    const Tensor<type, 2> inputs = data_set->get_data(DataSet::SampleUse::Testing, DataSet::VariableUse::Input);

    const Tensor<type, 2> targets = data_set->get_data(DataSet::SampleUse::Testing, DataSet::VariableUse::Target);

    // Neural network

    const Index outputs_number = neural_network->get_outputs_number();

    const Tensor<type, 2> outputs  = neural_network->calculate_outputs(inputs);

    UnscalingLayer* unscaling_layer = static_cast<UnscalingLayer*>(neural_network->get_first(Layer::Type::Unscaling));

    const Tensor<type, 1>& outputs_minimum = unscaling_layer->get_minimums();

    const Tensor<type, 1>& outputs_maximum = unscaling_layer->get_maximums();

    // Error data

    Tensor<type, 3> error_data(testing_samples_number, 3, outputs_number);

    // Absolute error

   const Tensor<type, 2> difference_absolute_value = (outputs - targets).abs();

   #pragma omp parallel for

   for(Index i = 0; i < outputs_number; i++)
   {
       for(Index j = 0; j < testing_samples_number; j++)
       {
           error_data(j, 0, i) = difference_absolute_value(j,i);
           error_data(j, 1, i) = difference_absolute_value(j,i)/abs(outputs_maximum(i)-outputs_minimum(i));
           error_data(j, 2, i) = difference_absolute_value(j,i)*type(100.0)/abs(outputs_maximum(i)-outputs_minimum(i));
       }
   }

    return error_data;
}


Tensor<type, 2> TestingAnalysis::calculate_percentage_error_data() const
{
    // Data set

    const Index testing_samples_number = data_set->get_samples_number(DataSet::SampleUse::Testing);

    const Tensor<type, 2> inputs = data_set->get_data(DataSet::SampleUse::Testing, DataSet::VariableUse::Input);

    const Tensor<type, 2> targets = data_set->get_data(DataSet::SampleUse::Testing, DataSet::VariableUse::Target);

    // Neural network

    const Index outputs_number = neural_network->get_outputs_number();

    const Tensor<type, 2> outputs = neural_network->calculate_outputs(inputs);

    UnscalingLayer* unscaling_layer = static_cast<UnscalingLayer*>(neural_network->get_first(Layer::Type::Unscaling));

    const Tensor<type, 1>& outputs_minimum = unscaling_layer->get_minimums();
    const Tensor<type, 1>& outputs_maximum = unscaling_layer->get_maximums();

    const Tensor<type, 2> difference_value = outputs - targets;

    // Error data

    Tensor<type, 2> error_data(testing_samples_number, outputs_number);

    #pragma omp parallel for

    for(Index i = 0; i < testing_samples_number; i++)
       for(Index j = 0; j < outputs_number; j++)
           error_data(i, j) = difference_value(i, j)*type(100.0)/abs(outputs_maximum(j) - outputs_minimum(j));

    return error_data;
}


vector<Descriptives> TestingAnalysis::calculate_absolute_errors_descriptives() const
{
    const Tensor<type, 2> inputs = data_set->get_data(DataSet::SampleUse::Testing, DataSet::VariableUse::Input);

    const Tensor<type, 2> targets = data_set->get_data(DataSet::SampleUse::Testing, DataSet::VariableUse::Target);

    const Tensor<type, 2> outputs = neural_network->calculate_outputs(inputs);

    return calculate_absolute_errors_descriptives(targets, outputs);
}


vector<Descriptives> TestingAnalysis::calculate_absolute_errors_descriptives(const Tensor<type, 2>& targets,
                                                                                const Tensor<type, 2>& outputs) const
{
    const Tensor<type, 2> difference = (targets-outputs).abs();

    return descriptives(difference);
}


vector<Descriptives> TestingAnalysis::calculate_percentage_errors_descriptives() const
{
    const Tensor<type, 2> inputs = data_set->get_data(DataSet::SampleUse::Testing, DataSet::VariableUse::Input);

    const Tensor<type, 2> targets = data_set->get_data(DataSet::SampleUse::Testing, DataSet::VariableUse::Target);

    const Tensor<type, 2> outputs = neural_network->calculate_outputs(inputs);

    return calculate_percentage_errors_descriptives(targets,outputs);
}


vector<Descriptives> TestingAnalysis::calculate_percentage_errors_descriptives(const Tensor<type, 2>& targets,
                                                                                  const Tensor<type, 2>& outputs) const
{
    const Tensor<type, 2> difference = type(100)*(targets-outputs).abs()/targets;

    return descriptives(difference);
}


vector<vector<Descriptives>> TestingAnalysis::calculate_error_data_descriptives() const
{
    // Neural network

    const Index outputs_number = neural_network->get_outputs_number();

    const Index testing_samples_number = data_set->get_samples_number(DataSet::SampleUse::Testing);

    // Testing analysis stuff

    vector<vector<Descriptives>> descriptives(outputs_number);

    Tensor<type, 3> error_data = calculate_error_data();

    Index index = 0;

    for(Index i = 0; i < outputs_number; i++)
    {
        const TensorMap<Tensor<type, 2>> matrix_error(error_data.data() + index, testing_samples_number, 3);

        const Tensor<type, 2> matrix(matrix_error);

        descriptives[i] = opennn::descriptives(matrix);

        index += testing_samples_number*3;
    }

    return descriptives;
}


void TestingAnalysis::print_error_data_descriptives() const
{
    const Index targets_number = data_set->get_variables_number(DataSet::VariableUse::Target);

    const vector<string> targets_name = data_set->get_variable_names(DataSet::VariableUse::Target);

    const vector<vector<Descriptives>> error_data_statistics = calculate_error_data_descriptives();

    for(Index i = 0; i < targets_number; i++)
        cout << targets_name[i] << endl
             << "Minimum error: " << error_data_statistics[i][0].minimum << endl
             << "Maximum error: " << error_data_statistics[i][0].maximum << endl
             << "Mean error: " << error_data_statistics[i][0].mean << " " << endl
             << "Standard deviation error: " << error_data_statistics[i][0].standard_deviation << " " << endl
             << "Minimum percentage error: " << error_data_statistics[i][2].minimum << " %" << endl
             << "Maximum percentage error: " << error_data_statistics[i][2].maximum << " %" << endl
             << "Mean percentage error: " << error_data_statistics[i][2].mean << " %" << endl
             << "Standard deviation percentage error: " << error_data_statistics[i][2].standard_deviation << " %" << endl
             << endl;
}


vector<Histogram> TestingAnalysis::calculate_error_data_histograms(const Index& bins_number) const
{
    const Tensor<type, 2> error_data = calculate_percentage_error_data();

    const Index outputs_number = error_data.dimension(1);

    vector<Histogram> histograms(outputs_number);

    for(Index i = 0; i < outputs_number; i++)
        histograms[i] = histogram_centered(error_data.chip(i,1), type(0), bins_number);

    return histograms;
}


Tensor<Tensor<Index, 1>, 1> TestingAnalysis::calculate_maximal_errors(const Index& samples_number) const
{
    Tensor<type, 3> error_data = calculate_error_data();

    const Index outputs_number = error_data.dimension(2);
    const Index testing_samples_number = error_data.dimension(0);

    Tensor<Tensor<Index, 1>, 1> maximal_errors(samples_number);

    Index index = 0;

    for(Index i = 0; i < outputs_number; i++)
    {
        const TensorMap<Tensor<type, 2>> matrix_error(error_data.data()+index, testing_samples_number, 3);

        maximal_errors[i] = maximal_indices(matrix_error.chip(0,1), samples_number);

        index += testing_samples_number*3;
    }

    return maximal_errors;
}


Tensor<type, 2> TestingAnalysis::calculate_errors() const
{
    Tensor<type, 2> errors(5, 3);

    const Tensor<type, 1> training_errors = calculate_errors(DataSet::SampleUse::Training);
    const Tensor<type, 1> selection_errors = calculate_errors(DataSet::SampleUse::Selection);
    const Tensor<type, 1> testing_errors = calculate_errors(DataSet::SampleUse::Testing);

    errors(0,0) = training_errors(0);
    errors(1,0) = training_errors(1);
    errors(2,0) = training_errors(2);
    errors(3,0) = training_errors(3);
    errors(4,0) = training_errors(4);

    errors(0,1) = selection_errors(0);
    errors(1,1) = selection_errors(1);
    errors(2,1) = selection_errors(2);
    errors(3,1) = selection_errors(3);
    errors(4,1) = selection_errors(4);

    errors(0,2) = testing_errors(0);
    errors(1,2) = testing_errors(1);
    errors(2,2) = testing_errors(2);
    errors(3,2) = testing_errors(3);
    errors(4,2) = testing_errors(4);

    return errors;
}


Tensor<type, 2> TestingAnalysis::calculate_binary_classification_errors() const
{
    Tensor<type, 2> errors(7, 3);

    const Tensor<type, 1> training_errors = calculate_binary_classification_errors(DataSet::SampleUse::Training);
    const Tensor<type, 1> selection_errors = calculate_binary_classification_errors(DataSet::SampleUse::Selection);
    const Tensor<type, 1> testing_errors = calculate_binary_classification_errors(DataSet::SampleUse::Testing);

    errors(0,0) = training_errors(0);
    errors(1,0) = training_errors(1);
    errors(2,0) = training_errors(2);
    errors(3,0) = training_errors(3);
    errors(4,0) = training_errors(4);
    errors(5,0) = training_errors(5);
    errors(6,0) = training_errors(6);

    errors(0,1) = selection_errors(0);
    errors(1,1) = selection_errors(1);
    errors(2,1) = selection_errors(2);
    errors(3,1) = selection_errors(3);
    errors(4,1) = selection_errors(4);
    errors(5,1) = selection_errors(5);
    errors(6,1) = selection_errors(6);

    errors(0,2) = testing_errors(0);
    errors(1,2) = testing_errors(1);
    errors(2,2) = testing_errors(2);
    errors(3,2) = testing_errors(3);
    errors(4,2) = testing_errors(4);
    errors(5,2) = testing_errors(5);
    errors(6,2) = testing_errors(6);

    return errors;
}


Tensor<type, 2> TestingAnalysis::calculate_multiple_classification_errors() const
{
    Tensor<type, 2> errors(6,3);

    const Tensor<type, 1> training_errors = calculate_multiple_classification_errors(DataSet::SampleUse::Training);
    const Tensor<type, 1> selection_errors = calculate_multiple_classification_errors(DataSet::SampleUse::Training);
    const Tensor<type, 1> testing_errors = calculate_multiple_classification_errors(DataSet::SampleUse::Training);

    errors(0,0) = training_errors(0);
    errors(1,0) = training_errors(1);
    errors(2,0) = training_errors(2);
    errors(3,0) = training_errors(3);
    errors(4,0) = training_errors(4);
    errors(5,0) = training_errors(5);

    errors(0,1) = selection_errors(0);
    errors(1,1) = selection_errors(1);
    errors(2,1) = selection_errors(2);
    errors(3,1) = selection_errors(3);
    errors(4,1) = selection_errors(4);
    errors(5,1) = selection_errors(5);

    errors(0,2) = testing_errors(0);
    errors(1,2) = testing_errors(1);
    errors(2,2) = testing_errors(2);
    errors(3,2) = testing_errors(3);
    errors(4,2) = testing_errors(4);
    errors(5,2) = testing_errors(5);

    return errors;
}


Tensor<type, 1> TestingAnalysis::calculate_errors(const Tensor<type, 2>& targets,
                                                  const Tensor<type, 2>& outputs) const
{
    const Index samples_number = outputs.dimension(0);

    Tensor<type, 1> errors(4);

    Tensor<type, 0> mean_squared_error;
    mean_squared_error.device(*thread_pool_device) = (outputs - targets).square().sum().sqrt();

    errors.setValues({mean_squared_error(0),
                      errors(0) / type(samples_number),
                      sqrt(errors(1)),
                      calculate_normalized_squared_error(targets, outputs)});;

    return errors;
}


Tensor<type, 1> TestingAnalysis::calculate_errors(const DataSet::SampleUse& sample_use) const
{
    const Tensor<type, 2> inputs = data_set->get_data(sample_use, DataSet::VariableUse::Input);

    const Tensor<type, 2> targets = data_set->get_data(sample_use, DataSet::VariableUse::Target);

    const Tensor<type, 2> outputs = neural_network->calculate_outputs(inputs);

    return calculate_errors(targets, outputs);
}


Tensor<type, 1> TestingAnalysis::calculate_binary_classification_errors(const DataSet::SampleUse& sample_use) const
{
    // Data set

    const Index training_samples_number = data_set->get_samples_number(sample_use);

    const Tensor<type, 2> inputs = data_set->get_data(sample_use, DataSet::VariableUse::Input);

    const Tensor<type, 2> targets = data_set->get_data(sample_use, DataSet::VariableUse::Target);

    // Neural network

    const Tensor<type, 2> outputs = neural_network->calculate_outputs(inputs);

    Tensor<type, 1> errors(6);

    // Results

    Tensor<type, 0> mean_squared_error;
    mean_squared_error.device(*thread_pool_device) = (outputs-targets).square().sum().sqrt();

    errors(0) = mean_squared_error(0);
    errors(1) = errors(0)/type(training_samples_number);
    errors(2) = sqrt(errors(1));
    errors(3) = calculate_normalized_squared_error(targets, outputs);
    errors(4) = calculate_cross_entropy_error(targets, outputs);
    errors(5) = calculate_weighted_squared_error(targets, outputs);

    return errors;
}


Tensor<type, 1> TestingAnalysis::calculate_multiple_classification_errors(const DataSet::SampleUse& sample_use) const
{
    // Data set

    const Index training_samples_number = data_set->get_samples_number(sample_use);

    const Tensor<type, 2> inputs = data_set->get_data(sample_use, DataSet::VariableUse::Input);

    const Tensor<type, 2> targets = data_set->get_data(sample_use, DataSet::VariableUse::Target);

    // Neural network

    const Tensor<type, 2> outputs = neural_network->calculate_outputs(inputs);

    Tensor<type, 1> errors(5);

    // Results

    Tensor<type, 0> mean_squared_error;
    mean_squared_error.device(*thread_pool_device) = (outputs-targets).square().sum().sqrt();

    errors(0) = mean_squared_error(0);
    errors(1) = errors(0)/type(training_samples_number);
    errors(2) = sqrt(errors(1));
    errors(3) = calculate_normalized_squared_error(targets, outputs);
    errors(4) = calculate_cross_entropy_error(targets, outputs); // NO

    return errors;
}


type TestingAnalysis::calculate_normalized_squared_error(const Tensor<type, 2>& targets, const Tensor<type, 2>& outputs) const
{
    const Index samples_number = targets.dimension(0);

    const Tensor<type, 1> targets_mean = mean(targets);

    Tensor<type, 0> mean_squared_error;
    mean_squared_error.device(*thread_pool_device) = (outputs - targets).square().sum();

    type normalization_coefficient = type(0);

    Tensor<type, 0> norm;

    for(Index i = 0; i < samples_number; i++)
    {
        norm.device(*thread_pool_device) = (targets.chip(i, 0) - targets_mean).square().sum();

        normalization_coefficient += norm(0);
    }

    return mean_squared_error()/normalization_coefficient;
}


type TestingAnalysis::calculate_cross_entropy_error(const Tensor<type, 2>& targets,
                                                    const Tensor<type, 2>& outputs) const
{
    const Index testing_samples_number = targets.dimension(0);
    const Index outputs_number = targets.dimension(1);

    Tensor<type, 1> targets_row(outputs_number);
    Tensor<type, 1> outputs_row(outputs_number);

    type cross_entropy_error = type(0);

#pragma omp parallel for reduction(+:cross_entropy_error)

    for(Index i = 0; i < testing_samples_number; i++)
    {
        outputs_row = outputs.chip(i, 0);
        targets_row = targets.chip(i, 0);

        for(Index j = 0; j < outputs_number; j++)
        {
            if(outputs_row(j) < NUMERIC_LIMITS_MIN)
                outputs_row(j) = type(1.0e-6);
            else if(double(outputs_row(j)) == 1.0)
                outputs_row(j) = numeric_limits<type>::max();

            cross_entropy_error -=
                    targets_row(j)*log(outputs_row(j)) + (type(1) - targets_row(j))*log(type(1) - outputs_row(j));
        }
    }

    return cross_entropy_error/type(testing_samples_number);
}


type TestingAnalysis::calculate_cross_entropy_error_3d(const Tensor<type, 3>& outputs, const Tensor<type, 2>& targets) const
{
    const Index batch_samples_number = outputs.dimension(0);
    const Index outputs_number = outputs.dimension(1);

    Tensor<type, 2> errors(batch_samples_number, outputs_number);
    Tensor<type, 2> predictions(batch_samples_number, outputs_number);
    Tensor<bool, 2> matches(batch_samples_number, outputs_number);
    Tensor<bool, 2> mask(batch_samples_number, outputs_number);

    Tensor<type, 0> cross_entropy_error;
    mask = targets != targets.constant(0);

    Tensor<type, 0> mask_sum;
    mask_sum = mask.cast<type>().sum();

    for(Index i = 0; i < batch_samples_number; i++)
        for(Index j = 0; j < outputs_number; j++)
            errors(i, j) = -log(outputs(i, j, Index(targets(i, j))));

    errors = errors * mask.cast<type>();

    cross_entropy_error = errors.sum();

    return cross_entropy_error(0) / mask_sum(0);
}


type TestingAnalysis::calculate_weighted_squared_error(const Tensor<type, 2>& targets,
                                                       const Tensor<type, 2>& outputs,
                                                       const Tensor<type, 1>& weights) const
{
    type negatives_weight;
    type positives_weight;

    if(weights.size() != 2)
    {
        const Tensor<Index, 1> target_distribution = data_set->calculate_target_distribution();

        const Index negatives_number = target_distribution[0];
        const Index positives_number = target_distribution[1];

        negatives_weight = type(1);
        positives_weight = type(negatives_number/positives_number);
    }
    else
    {
        positives_weight = weights[0];
        negatives_weight = weights[1];
    }

    const Tensor<bool, 2> if_sentence = elements_are_equal(targets, targets.constant(type(1)));
    const Tensor<bool, 2> else_sentence = elements_are_equal(targets, targets.constant(type(0)));

    Tensor<type, 2> f_1(targets.dimension(0), targets.dimension(1));

    Tensor<type, 2> f_2(targets.dimension(0), targets.dimension(1));

    Tensor<type, 2> f_3(targets.dimension(0), targets.dimension(1));

    f_1.device(*thread_pool_device) = (targets - outputs).square() * positives_weight;

    f_2.device(*thread_pool_device) = (targets - outputs).square()*negatives_weight;

    f_3.device(*thread_pool_device) = targets.constant(type(0));

    Tensor<type, 0> mean_squared_error;
    mean_squared_error.device(*thread_pool_device) = (if_sentence.select(f_1, else_sentence.select(f_2, f_3))).sum();

    Index negatives = 0;

    Tensor<type, 1> target_column = targets.chip(0,1);

    for(Index i = 0; i < target_column.size(); i++)
        if(double(target_column(i)) == 0.0)
            negatives++;

    const type normalization_coefficient = type(negatives)*negatives_weight*type(0.5);

    return mean_squared_error(0)/normalization_coefficient;
}


type TestingAnalysis::calculate_Minkowski_error(const Tensor<type, 2>& targets,
                                                const Tensor<type, 2>& outputs,
                                                const type minkowski_parameter) const
{
    Tensor<type, 0> minkowski_error;
    minkowski_error.device(*thread_pool_device) = (outputs - targets).abs().pow(minkowski_parameter).sum().pow(type(1)/minkowski_parameter);

    return minkowski_error();
}


type TestingAnalysis::calculate_masked_accuracy(const Tensor<type, 3>& outputs, const Tensor<type, 2>& targets) const
{
    const Index batch_samples_number = outputs.dimension(0);
    const Index outputs_number = outputs.dimension(1);

    Tensor<type, 2> predictions(batch_samples_number, outputs_number);
    Tensor<bool, 2> matches(batch_samples_number, outputs_number);
    Tensor<bool, 2> mask(batch_samples_number, outputs_number);

    Tensor<type, 0> accuracy;

    mask = targets != targets.constant(0);

    const Tensor<type, 0> mask_sum = mask.cast<type>().sum();

    predictions = outputs.argmax(2).cast<type>();

    matches = predictions == targets;

    matches = matches && mask;

    accuracy = matches.cast<type>().sum() / mask_sum(0);

    return accuracy(0);
}


type TestingAnalysis::calculate_determination(const Tensor<type, 1>& outputs, const Tensor<type, 1>& targets) const
{
    Tensor<type, 0> targets_mean;
    targets_mean.device(*thread_pool_device) = targets.mean();

    Tensor<type, 0> outputs_mean;
    outputs_mean.device(*thread_pool_device) = outputs.mean();

    Tensor<type,0> numerator;
    numerator.device(*thread_pool_device) = ((-targets_mean(0) + targets)*(-outputs_mean(0) + outputs)).sum();

    Tensor<type,0> denominator;
    denominator.device(*thread_pool_device) = ((-targets_mean(0) + targets).square().sum()*(-outputs_mean(0) + outputs).square().sum()).sqrt();

    if(denominator(0) == type(0))
        denominator(0) = 1;

    return (numerator(0)*numerator(0))/(denominator(0)*denominator(0));
}


Tensor<Index, 2> TestingAnalysis::calculate_confusion_binary_classification(const Tensor<type, 2>& targets,
                                                                            const Tensor<type, 2>& outputs,
                                                                            const type& decision_threshold) const
{
    const Index testing_samples_number = targets.dimension(0);

    Tensor<Index, 2> confusion(3, 3);

    Index true_positive = 0;
    Index false_negative = 0;
    Index false_positive = 0;
    Index true_negative = 0;

    type target = type(0);
    type output = type(0);

    for(Index i = 0; i < testing_samples_number; i++)
    {
        target = targets(i,0);
        output = outputs(i,0);

        if(target >= decision_threshold && output >= decision_threshold)
            true_positive++;
        else if(target >= decision_threshold && output < decision_threshold)
            false_negative++;
        else if(target < decision_threshold && output >= decision_threshold)
            false_positive++;
        else if(target < decision_threshold && output < decision_threshold)
            true_negative++;
        else
            throw runtime_error("calculate_confusion_binary_classification Unknown case.\n");
    }

    confusion(0,0) = true_positive;
    confusion(0,1) = false_negative;
    confusion(1,0) = false_positive;
    confusion(1,1) = true_negative;

    confusion(0,2) = true_positive + false_negative;
    confusion(1,2) = false_positive + true_negative;
    confusion(2,0) = true_positive + false_positive;
    confusion(2,1) = true_negative + false_negative;
    confusion(2,2) = testing_samples_number;

    const Index confusion_sum = true_positive + false_negative + false_positive + true_negative;

    if(confusion_sum != testing_samples_number)
        throw runtime_error("Number of elements in confusion matrix (" + to_string(confusion_sum) + ") "
                            "must be equal to number of testing samples (" + to_string(testing_samples_number) + ").\n");

    return confusion;
}


Tensor<Index, 2> TestingAnalysis::calculate_confusion_multiple_classification(const Tensor<type, 2>& targets,
                                                                              const Tensor<type, 2>& outputs) const
{
    const Index samples_number = targets.dimension(0);
    const Index targets_number = targets.dimension(1);

    if(targets_number != outputs.dimension(1))
        throw runtime_error("Number of targets (" + to_string(targets_number) + ") "
                            "must be equal to number of outputs (" + to_string(outputs.dimension(1)) + ").\n");

    Tensor<Index, 2> confusion(targets_number + 1, targets_number + 1);
    confusion.setZero();
    confusion(targets_number, targets_number) = samples_number;

    Index target_index = 0;
    Index output_index = 0;

    for(Index i = 0; i < samples_number; i++)
    {
        target_index = maximal_index(targets.chip(i, 0));
        output_index = maximal_index(outputs.chip(i, 0));

        confusion(target_index, output_index)++;
        confusion(target_index, targets_number)++;
        confusion(targets_number, output_index)++;
    }

    return confusion;
}


Tensor<Index, 1> TestingAnalysis::calculate_positives_negatives_rate(const Tensor<type, 2>& targets, const Tensor<type, 2>& outputs) const
{
    const Tensor<Index, 2> confusion = calculate_confusion_binary_classification(targets, outputs, type(0.5));

    Tensor<Index, 1> positives_negatives_rate;
    positives_negatives_rate.setValues({confusion(0,0) + confusion(0,1), confusion(1,0) + confusion(1,1)});

    return positives_negatives_rate;
}


Tensor<Index, 2> TestingAnalysis::calculate_confusion() const
{
    check();

    //const Index outputs_number = neural_network->get_outputs_number();

    Tensor<type, 2> inputs = data_set->get_data(DataSet::SampleUse::Testing, DataSet::VariableUse::Input);

    const Tensor<type, 2> targets = data_set->get_data(DataSet::SampleUse::Testing, DataSet::VariableUse::Target);

    const Index samples_number = targets.dimension(0);

    const dimensions input_dimensions = data_set->get_dimensions(DataSet::VariableUse::Input);

    if(input_dimensions.size() == 1)
    {
        const Tensor<type, 2> outputs = neural_network->calculate_outputs(inputs);

        return calculate_confusion(outputs, targets);
    }
    else if(input_dimensions.size() == 3)
    {
        type* input_data = inputs.data();

        Tensor<type, 4> inputs_4d(samples_number,
                                  input_dimensions[0],
                                  input_dimensions[1],
                                  input_dimensions[2]);
        
        memcpy(inputs_4d.data(), input_data, samples_number * inputs.dimension(1)*sizeof(type));

        const Tensor<type, 2> outputs = neural_network->calculate_outputs(inputs_4d);

        return calculate_confusion(outputs, targets);
    }

    return Tensor<Index, 2>();
}


Tensor<Index, 2> TestingAnalysis::calculate_sentimental_analysis_transformer_confusion() const
{
    Transformer* transformer = static_cast<Transformer*>(neural_network);
    LanguageDataSet* language_data_set = static_cast<LanguageDataSet*>(data_set);

    const Tensor<type, 2> inputs = language_data_set->get_data(DataSet::SampleUse::Testing, DataSet::VariableUse::Input);
    const Tensor<type, 2> context = language_data_set->get_data(DataSet::SampleUse::Testing, DataSet::VariableUse::Decoder);
    const Tensor<type, 2> targets = language_data_set->get_data(DataSet::SampleUse::Testing, DataSet::VariableUse::Target);

    const dimensions input_dimensions = data_set->get_dimensions(DataSet::VariableUse::Input);

    const Index testing_batch_size = inputs.dimension(0) > 2000 ? 2000 : inputs.dimension(0);

    Tensor<type, 2> testing_input(testing_batch_size, inputs.dimension(1));
    Tensor<type, 2> testing_context(testing_batch_size, context.dimension(1));
    Tensor<type, 2> testing_target(testing_batch_size, targets.dimension(1));

    for(Index i = 0; i < testing_batch_size; i++)
    {
        testing_input.chip(i, 0) = inputs.chip(i, 0);
        testing_context.chip(i, 0) = context.chip(i, 0);
        testing_target.chip(i, 0) = targets.chip(i, 0);
    }


    if(input_dimensions.size() == 1)
    {
        Tensor<type, 3> outputs = transformer->calculate_outputs(testing_input, testing_context);
        cout<<outputs.dimensions()<<endl;
        Tensor<type, 2> reduced_outputs(outputs.dimension(0), targets.dimension(1));

        for (Index i = 0; i < outputs.dimension(0); i++) {
            reduced_outputs(i,0) = outputs(i,1,9);
            reduced_outputs(i,1) = outputs(i,1,10);
        }

        // Tensor<type, 2> reduced_outputs(outputs.dimension(0), outputs.dimension(1));
        // type index;
        // type max;

        // for (Index i = 0; i < outputs.dimension(0); i++) {
        //     for (Index j = 0; j < outputs.dimension(1); j++) {
        //         index = 0;
        //         max = outputs(i,j,0);
        //         for(Index k = 1; k < outputs.dimension(2); k++)
        //             if(max < outputs(i,j,k)){
        //                 index = type(k);
        //                 max = outputs(i,j,k);
        //             }
        //         reduced_outputs(i,j) = index;
        //     }
        // }
        // cout<<reduced_outputs.dimensions()<<endl;

        return calculate_confusion(reduced_outputs, testing_target);
    }

    return Tensor<Index, 2>();
}


Tensor<Index, 2> TestingAnalysis::calculate_confusion(const Tensor<type, 2>& outputs,
                                                      const Tensor<type, 2>& targets) const
{
    const Index outputs_number = neural_network->get_outputs_number();

    if (outputs_number == 1)
    {
        ProbabilisticLayer* probabilistic_layer = static_cast<ProbabilisticLayer*>(neural_network->get_first(Layer::Type::Probabilistic));

        const type decision_threshold = probabilistic_layer
                                        ? probabilistic_layer->get_decision_threshold()
                                        : type(0.5);

        return calculate_confusion_binary_classification(targets, outputs, decision_threshold);
    }
    else
    {
        return calculate_confusion_multiple_classification(targets, outputs);
    }
}


Tensor<Index, 2> TestingAnalysis::calculate_confusion(const Tensor<type, 3>& outputs,
                                                      const Tensor<type, 3>& targets) const
{
    return Tensor<Index, 2>();
}


TestingAnalysis::RocAnalysis TestingAnalysis::perform_roc_analysis() const
{
    const Tensor<type, 2> inputs = data_set->get_data(DataSet::SampleUse::Testing, DataSet::VariableUse::Input);

    const Tensor<type, 2> targets = data_set->get_data(DataSet::SampleUse::Testing, DataSet::VariableUse::Target);

    const Tensor<type, 2> outputs = neural_network->calculate_outputs(inputs);

    RocAnalysis roc_analysis;

    roc_analysis.roc_curve = calculate_roc_curve(targets, outputs);

    roc_analysis.area_under_curve = calculate_area_under_curve(roc_analysis.roc_curve);

    roc_analysis.confidence_limit = calculate_area_under_curve_confidence_limit(targets, outputs);

    roc_analysis.optimal_threshold = calculate_optimal_threshold(roc_analysis.roc_curve);

    return roc_analysis;
}


Tensor<type, 2> TestingAnalysis::calculate_roc_curve(const Tensor<type, 2>& targets, const Tensor<type, 2>& outputs) const
{
    const Tensor<Index, 1> positives_negatives_rate = calculate_positives_negatives_rate(targets, outputs);

    const Index total_positives = positives_negatives_rate(0);
    const Index total_negatives = positives_negatives_rate(1);

    if(total_positives == 0)
        throw runtime_error("Number of positive samples (" + to_string(total_positives) + ") must be greater than zero.\n");

    if(total_negatives == 0)
        throw runtime_error("Number of negative samples (" + to_string(total_negatives) + ") must be greater than zero.\n");

    const Index maximum_points_number = 200;

    Index points_number;

    points_number = maximum_points_number;

   if(targets.dimension(1) != 1)
        throw runtime_error("Number of of target variables (" +  to_string(targets.dimension(1)) + ") must be one.\n");

    if(outputs.dimension(1) != 1)
        throw runtime_error("Number of of output variables (" + to_string(targets.dimension(1)) + ") must be one.\n");

    // Sort by ascending values of outputs vector

    Tensor<Index, 1> sorted_indices(outputs.dimension(0));

    Index* sorted_indices_data = sorted_indices.data();

    iota(sorted_indices_data, sorted_indices_data + sorted_indices.size(), 0);

    stable_sort(sorted_indices.data(),
                sorted_indices_data + sorted_indices.size(),
                [outputs](Index i1, Index i2) {return outputs(i1,0) < outputs(i2,0);});

    Tensor<type, 2> roc_curve(points_number + 1, 3);
    roc_curve.setZero();

#pragma omp parallel for schedule(dynamic)

    for(Index i = 1; i < Index(points_number); i++)
    {
        const type threshold = type(i) * (type(1)/type(points_number));

        Index true_positive = 0;
        Index false_negative = 0;
        Index false_positive = 0;
        Index true_negative = 0;

        type target;
        type output;

        for(Index j = 0; j < targets.size(); j++)
        {
            target = targets(j, 0);
            output = outputs(j, 0);

            if(target >= threshold && output >= threshold)
                true_positive++;
            else if(target >= threshold && output < threshold)
                false_negative++;
            else if(target < threshold && output >= threshold)
                false_positive++;
            else if(target < threshold && output < threshold)
                true_negative++;
        }

        roc_curve(i,0) = type(1 - true_positive)/type(true_positive + false_negative);
        roc_curve(i,1) = type(true_negative)/type(true_negative + false_positive);
        roc_curve(i,2) = type(threshold);

        if(isnan(roc_curve(i,0)))
            roc_curve(i,0) = type(1);

        if(isnan(roc_curve(i,1)))
            roc_curve(i,1) = type(0);
    }

    roc_curve(0,0) = type(0);
    roc_curve(0,1) = type(0);
    roc_curve(0,2) = type(0);

    roc_curve(points_number,0) = type(1);
    roc_curve(points_number,1) = type(1);
    roc_curve(points_number,2) = type(1);

    return roc_curve;
}


type TestingAnalysis::calculate_area_under_curve(const Tensor<type, 2>& roc_curve) const
{
    type area_under_curve = type(0);        

    for(Index i = 1; i < roc_curve.dimension(0); i++)
        area_under_curve += (roc_curve(i,0) - roc_curve(i-1,0))*(roc_curve(i,1) + roc_curve(i-1,1));

    return area_under_curve/ type(2);
}


type TestingAnalysis::calculate_area_under_curve_confidence_limit(const Tensor<type, 2>& targets, const Tensor<type, 2>& outputs) const
{
    const Tensor<Index, 1> positives_negatives_rate = calculate_positives_negatives_rate(targets, outputs);

    const Index total_positives = positives_negatives_rate[0];
    const Index total_negatives = positives_negatives_rate[1];

    if(total_positives == 0)
        throw runtime_error("Number of positive samples(" + to_string(total_positives) + ") must be greater than zero.\n");

    if(total_negatives == 0)
        throw runtime_error("Number of negative samples(" + to_string(total_negatives) + ") must be greater than zero.\n");

    const Tensor<type, 2> roc_curve = calculate_roc_curve(targets, outputs);

    const type area_under_curve = calculate_area_under_curve(roc_curve);

    const type Q_1 = area_under_curve/(type(2.0) - area_under_curve);
    const type Q_2 = (type(2.0) *area_under_curve*area_under_curve)/(type(1) *area_under_curve);

    const type confidence_limit = type(type(1.64485)*sqrt((area_under_curve*(type(1) - area_under_curve)
                                  + (type(total_positives) - type(1))*(Q_1-area_under_curve*area_under_curve)
                                  + (type(total_negatives) - type(1))*(Q_2-area_under_curve*area_under_curve))/(type(total_positives*total_negatives))));

    return confidence_limit;
}


type TestingAnalysis::calculate_optimal_threshold(const Tensor<type, 2>& roc_curve) const
{
    const Index points_number = roc_curve.dimension(0);

    type optimal_threshold = type(0.5);

    type minimun_distance = numeric_limits<type>::max();

    for(Index i = 0; i < points_number; i++)
    {
        //const type distance = sqrt(roc_curve(i,0)*roc_curve(i,0) + (roc_curve(i,1) - type(1))*(roc_curve(i,1) - type(1)));

        const type distance = hypot(roc_curve(i, 0), roc_curve(i, 1) - type(1));

        if(distance < minimun_distance)
        {
            optimal_threshold = roc_curve(i,2);

            minimun_distance = distance;
        }
    }

    return optimal_threshold;
}


Tensor<type, 2> TestingAnalysis::perform_cumulative_gain_analysis() const
{
    const Tensor<type, 2> inputs = data_set->get_data(DataSet::SampleUse::Testing, DataSet::VariableUse::Input);

    const Tensor<type, 2> targets = data_set->get_data(DataSet::SampleUse::Testing, DataSet::VariableUse::Target);

    const Tensor<type, 2> outputs = neural_network->calculate_outputs(inputs);

    return calculate_cumulative_gain(targets, outputs);
}


Tensor<type, 2> TestingAnalysis::calculate_cumulative_gain(const Tensor<type, 2>& targets, const Tensor<type, 2>& outputs) const
{
    const Index total_positives = calculate_positives_negatives_rate(targets, outputs)[0];

    if(total_positives == 0)
        throw runtime_error("Number of positive samples(" + to_string(total_positives) + ") must be greater than zero.\n");

    const Index testing_samples_number = targets.dimension(0);

    // Sort by ascending values of outputs vector

    Tensor<Index, 1> sorted_indices(outputs.dimension(0));
    iota(sorted_indices.data(), sorted_indices.data() + sorted_indices.size(), 0);

    stable_sort(sorted_indices.data(),
                sorted_indices.data()+sorted_indices.size(),
                [outputs](Index i1, Index i2) {return outputs(i1,0) > outputs(i2,0);});

    Tensor<type, 1> sorted_targets(testing_samples_number);

    for(Index i = 0; i < testing_samples_number; i++)
        sorted_targets(i) = targets(sorted_indices(i),0);

    const Index points_number = 21;
    const type percentage_increment = type(0.05);

    Tensor<type, 2> cumulative_gain(points_number, 2);

    cumulative_gain(0,0) = type(0);
    cumulative_gain(0,1) = type(0);

    Index positives = 0;

    type percentage = type(0);

    Index maximum_index;

    for(Index i = 0; i < points_number - 1; i++)
    {
        percentage += percentage_increment;

        positives = 0;

        maximum_index = Index(percentage*testing_samples_number);

        for(Index j = 0; j < maximum_index; j++)
            if(double(sorted_targets(j)) == 1.0)
                 positives++;

        cumulative_gain(i + 1, 0) = percentage;
        cumulative_gain(i + 1, 1) = type(positives)/type(total_positives);
    }

    return cumulative_gain;
}


Tensor<type, 2> TestingAnalysis::calculate_negative_cumulative_gain(const Tensor<type, 2>& targets, const Tensor<type, 2>& outputs) const
{
    const Index total_negatives = calculate_positives_negatives_rate(targets, outputs)[1];

    if(total_negatives == 0)
        throw runtime_error("Number of negative samples(" + to_string(total_negatives) + ") must be greater than zero.\n");

    const Index testing_samples_number = targets.dimension(0);

    // Sort by ascending values of outputs vector

    Tensor<Index, 1> sorted_indices(outputs.dimension(0));
    iota(sorted_indices.data(), sorted_indices.data() + sorted_indices.size(), 0);

    stable_sort(sorted_indices.data(), sorted_indices.data()+sorted_indices.size(), [outputs](Index i1, Index i2) {return outputs(i1,0) > outputs(i2,0);});

    Tensor<type, 1> sorted_targets(testing_samples_number);

    for(Index i = 0; i < testing_samples_number; i++)
        sorted_targets(i) = targets(sorted_indices(i),0);

    const Index points_number = 21;
    const type percentage_increment = type(0.05);

    Tensor<type, 2> negative_cumulative_gain(points_number, 2);

    negative_cumulative_gain(0,0) = type(0);
    negative_cumulative_gain(0,1) = type(0);

    Index negatives = 0;

    type percentage = type(0);

    Index maximum_index;

    for(Index i = 0; i < points_number - 1; i++)
    {
        percentage += percentage_increment;

        negatives = 0;

        maximum_index = Index(percentage* type(testing_samples_number));

        for(Index j = 0; j < maximum_index; j++)
            if(sorted_targets(j) < NUMERIC_LIMITS_MIN)
                 negatives++;

        negative_cumulative_gain(i + 1, 0) = percentage;
        negative_cumulative_gain(i + 1, 1) = type(negatives)/type(total_negatives);
    }

    return negative_cumulative_gain;
}


Tensor<type, 2> TestingAnalysis::perform_lift_chart_analysis() const
{
    const Tensor<type, 2> inputs = data_set->get_data(DataSet::SampleUse::Testing, DataSet::VariableUse::Input);

    const Tensor<type, 2> targets = data_set->get_data(DataSet::SampleUse::Testing, DataSet::VariableUse::Target);

    const Tensor<type, 2> outputs = neural_network->calculate_outputs(inputs);

    const Tensor<type, 2> cumulative_gain = calculate_cumulative_gain(targets, outputs);

    return calculate_lift_chart(cumulative_gain);
}


Tensor<type, 2> TestingAnalysis::calculate_lift_chart(const Tensor<type, 2>& cumulative_gain) const
{
    const Index rows_number = cumulative_gain.dimension(0);
    const Index raw_variables_number = cumulative_gain.dimension(1);

    Tensor<type, 2> lift_chart(rows_number, raw_variables_number);

    lift_chart(0,0) = type(0);
    lift_chart(0,1) = type(1);

    #pragma omp parallel for

    for(Index i = 1; i < rows_number; i++)
    {
        lift_chart(i, 0) = type(cumulative_gain(i, 0));
        lift_chart(i, 1) = type(cumulative_gain(i, 1))/type(cumulative_gain(i, 0));
    }

    return lift_chart;
}


Tensor<type, 1> TestingAnalysis::calculate_maximum_gain(const Tensor<type, 2>& positive_cumulative_gain,
                                                        const Tensor<type, 2>& negative_cumulative_gain) const
{
    const Index points_number = positive_cumulative_gain.dimension(0);

    Tensor<type, 1> maximum_gain(2);

    const type percentage_increment = type(0.05);

    type percentage = type(0);

    for(Index i = 0; i < points_number - 1; i++)
    {
        percentage += percentage_increment;

        if(positive_cumulative_gain(i+1,1)-negative_cumulative_gain(i+1,1) > maximum_gain[1]
        && positive_cumulative_gain(i+1,1)-negative_cumulative_gain(i+1,1) > type(0))
        {
            maximum_gain(1) = positive_cumulative_gain(i+1,1)-negative_cumulative_gain(i+1,1);
            maximum_gain(0) = percentage;
        }
    }

    return maximum_gain;
}


Tensor<type, 2> TestingAnalysis::perform_calibration_plot_analysis() const
{
    const Tensor<type, 2> inputs = data_set->get_data(DataSet::SampleUse::Testing, DataSet::VariableUse::Input);

    const Tensor<type, 2> targets = data_set->get_data(DataSet::SampleUse::Testing, DataSet::VariableUse::Target);

    const Tensor<type, 2> outputs = neural_network->calculate_outputs(inputs);

    return calculate_calibration_plot(targets, outputs);
}


Tensor<type, 2> TestingAnalysis::calculate_calibration_plot(const Tensor<type, 2>& targets, const Tensor<type, 2>& outputs) const
{
    const Index rows_number = targets.dimension(0);

    const Index points_number = 10;

    Tensor<type, 2> calibration_plot(points_number+2, 2);

    // First point

    calibration_plot(0,0) = type(0);
    calibration_plot(0,1) = type(0);

    Index positives = 0;

    Index count = 0;

    type probability = type(0);

    long double sum = 0.0;

    for(Index i = 1; i < points_number+1; i++)
    {
        count = 0;

        positives = 0;
        sum = type(0);

        probability += type(0.1);

        for(Index j = 0; j < rows_number; j++)
        {
            if(outputs(j, 0) >= (probability - type(0.1)) && outputs(j, 0) < probability)
            {
                count++;

                sum += outputs(j, 0);

                if(Index(targets(j, 0)) == 1)
                    positives++;
            }
        }

        if(count == 0)
        {
            calibration_plot(i, 0) = type(-1);
            calibration_plot(i, 1) = type(-1);
        }
        else
        {
            calibration_plot(i, 0) = type(sum)/type(count);
            calibration_plot(i, 1) = type(positives)/type(count);
        }
    }

    // Last point

    calibration_plot(points_number+1,0) = type(1);
    calibration_plot(points_number+1,1) = type(1);

    // Subtracts calibration plot rows with value -1

    Index points_number_subtracted = 0;
/*
    while(contains(calibration_plot.chip(0,1), type(-1)))
     {
         for(Index i = 1; i < points_number - points_number_subtracted + 1; i++)
         {
             if(abs(calibration_plot(i, 0) + type(1)) < NUMERIC_LIMITS_MIN)
             {
                 calibration_plot = delete_row(calibration_plot, i);

                 points_number_subtracted++;
             }
         }
     }
*/
    return calibration_plot;
}


vector<Histogram> TestingAnalysis::calculate_output_histogram(const Tensor<type, 2>& outputs,
                                                              const Index& bins_number) const
{

    const Tensor<type, 1> output_column = outputs.chip(0,1);

    vector<Histogram> output_histogram (1);

    output_histogram[0] = histogram(output_column, bins_number);

    return output_histogram;
}


TestingAnalysis::BinaryClassificationRates TestingAnalysis::calculate_binary_classification_rates() const
{
    const Tensor<type, 2> inputs = data_set->get_data(DataSet::SampleUse::Testing, DataSet::VariableUse::Input);

    const Tensor<type, 2> targets = data_set->get_data(DataSet::SampleUse::Testing, DataSet::VariableUse::Target);

    const Tensor<type, 2> outputs = neural_network->calculate_outputs(inputs);

    const vector<Index> testing_indices = data_set->get_sample_indices(DataSet::SampleUse::Testing);

    ProbabilisticLayer* probabilistic_layer = static_cast<ProbabilisticLayer*>(neural_network->get_first(Layer::Type::Probabilistic));

    const type decision_threshold = probabilistic_layer
                                  ? probabilistic_layer->get_decision_threshold()
                                  : type(0.5);

    BinaryClassificationRates binary_classification_rates;

    binary_classification_rates.true_positives_indices = calculate_true_positive_samples(targets, outputs, testing_indices, decision_threshold);
    binary_classification_rates.false_positives_indices = calculate_false_positive_samples(targets, outputs, testing_indices, decision_threshold);
    binary_classification_rates.false_negatives_indices = calculate_false_negative_samples(targets, outputs, testing_indices, decision_threshold);
    binary_classification_rates.true_negatives_indices = calculate_true_negative_samples(targets, outputs, testing_indices, decision_threshold);

    return binary_classification_rates;
}


vector<Index> TestingAnalysis::calculate_true_positive_samples(const Tensor<type, 2>& targets,
                                                               const Tensor<type, 2>& outputs,
                                                               const vector<Index>& testing_indices,
                                                               const type& decision_threshold) const
{
    const Index rows_number = targets.dimension(0);

    Tensor<Index, 1> true_positives_indices_copy(rows_number);

    Index index = 0;

    for(Index i = 0; i < rows_number; i++)
        if(targets(i,0) >= decision_threshold && outputs(i,0) >= decision_threshold)
            true_positives_indices_copy(index++) = testing_indices[i];

    vector<Index> true_positives_indices(index);

    copy(true_positives_indices_copy.data(),
         true_positives_indices_copy.data() + index,
         true_positives_indices.data());

    return true_positives_indices;
}


vector<Index> TestingAnalysis::calculate_false_positive_samples(const Tensor<type, 2>& targets,
                                                                const Tensor<type, 2>& outputs,
                                                                const vector<Index>& testing_indices,
                                                                const type& decision_threshold) const
{
    const Index rows_number = targets.dimension(0);

    vector<Index> false_positives_indices_copy(rows_number);

    Index index = 0;

    for(Index i = 0; i < rows_number; i++)
        if(targets(i,0) < decision_threshold && outputs(i,0) >= decision_threshold)
            false_positives_indices_copy[index++] = testing_indices[i];

    vector<Index> false_positives_indices(index);

    copy(false_positives_indices_copy.data(),
         false_positives_indices_copy.data() + index,
         false_positives_indices.data());

    return false_positives_indices;
}


vector<Index> TestingAnalysis::calculate_false_negative_samples(const Tensor<type, 2>& targets,
                                                                   const Tensor<type, 2>& outputs,
                                                                   const vector<Index>& testing_indices,
                                                                   const type& decision_threshold) const
{
    const Index rows_number = targets.dimension(0);

    vector<Index> false_negatives_indices_copy(rows_number);

    Index index = 0;

    for(Index i = 0; i < rows_number; i++)
        if(targets(i,0) > decision_threshold && outputs(i,0) < decision_threshold)
            false_negatives_indices_copy[index++] = testing_indices[i];

    vector<Index> false_negatives_indices(index);

    copy(false_negatives_indices_copy.data(),
         false_negatives_indices_copy.data() + index,
         false_negatives_indices.data());

    return false_negatives_indices;
}


vector<Index> TestingAnalysis::calculate_true_negative_samples(const Tensor<type, 2>& targets,
                                                               const Tensor<type, 2>& outputs,
                                                               const vector<Index>& testing_indices,
                                                               const type& decision_threshold) const
{
    const Index rows_number = targets.dimension(0);

    Tensor<Index, 1> true_negatives_indices_copy(rows_number);

    Index index = 0;

    for(Index i = 0; i < rows_number; i++)
        if(targets(i,0) < decision_threshold && outputs(i,0) < decision_threshold)
            true_negatives_indices_copy(index++) = testing_indices[i];

    vector<Index> true_negatives_indices(index);

    copy(true_negatives_indices_copy.data(),
         true_negatives_indices_copy.data() + index,
         true_negatives_indices.data());

    return true_negatives_indices;
}


Tensor<type, 1> TestingAnalysis::calculate_multiple_classification_precision() const
{
    Tensor<type, 1> multiple_classification_tests(2);

    const Tensor<type, 2> inputs = data_set->get_data(DataSet::SampleUse::Testing, DataSet::VariableUse::Input);

    const Tensor<type, 2> targets = data_set->get_data(DataSet::SampleUse::Testing, DataSet::VariableUse::Target);

    const Tensor<type, 2> outputs = neural_network->calculate_outputs(inputs);

    const Tensor<Index, 2> confusion_matrix = calculate_confusion_multiple_classification(targets, outputs);

    type diagonal_sum = type(0);
    type off_diagonal_sum = type(0);

    const Tensor<Index, 0> total_sum = confusion_matrix.sum();

    #pragma omp parallel for
    for(Index i = 0; i < confusion_matrix.dimension(0); i++)
        for(Index j = 0; j < confusion_matrix.dimension(1); j++)
            i == j
            ? diagonal_sum += type(confusion_matrix(i, j))
            : off_diagonal_sum += type(confusion_matrix(i, j));

    multiple_classification_tests(0) = diagonal_sum/type(total_sum());
    multiple_classification_tests(1) = off_diagonal_sum/type(total_sum());

    return multiple_classification_tests;
}


void TestingAnalysis::save_confusion(const filesystem::path& file_name) const
{
    const Tensor<Index, 2> confusion = calculate_confusion();

    const Index raw_variables_number = confusion.dimension(0);

    ofstream file(file_name);

    const vector<string> target_variable_names = data_set->get_variable_names(DataSet::VariableUse::Target);

    file << ",";

    for(Index i = 0; i < confusion.dimension(0); i++)
    {
        file << target_variable_names[i];

        if(i != Index(target_variable_names.size()) - 1)
            file << ",";
    }

    file << endl;

    for(Index i = 0; i < raw_variables_number; i++)
    {
        file << target_variable_names[i] << ",";

        for(Index j = 0; j < raw_variables_number; j++)
            if(j == raw_variables_number - 1)
                file << confusion(i, j) << endl;
            else
                file << confusion(i, j) << ",";
    }

    file.close();
}


void TestingAnalysis::save_multiple_classification_tests(const filesystem::path& file_name) const
{
    const Tensor<type, 1> multiple_classification_tests = calculate_multiple_classification_precision();

    ofstream file(file_name);

    file << "accuracy,error" << endl;
    file << multiple_classification_tests(0)* type(100) << "," << multiple_classification_tests(1)* type(100) << endl;

    file.close();
}


Tensor<Tensor<Index,1>, 2> TestingAnalysis::calculate_multiple_classification_rates() const
{
    const Tensor<type, 2> inputs = data_set->get_data(DataSet::SampleUse::Testing, DataSet::VariableUse::Input);

    const Tensor<type, 2> targets = data_set->get_data(DataSet::SampleUse::Testing, DataSet::VariableUse::Target);

    const Tensor<type, 2> outputs = neural_network->calculate_outputs(inputs);

    const vector<Index> testing_indices = data_set->get_sample_indices(DataSet::SampleUse::Testing);

    return calculate_multiple_classification_rates(targets, outputs, testing_indices);
}


Tensor<Tensor<Index,1>, 2> TestingAnalysis::calculate_multiple_classification_rates(const Tensor<type, 2>& targets,
                                                                                    const Tensor<type, 2>& outputs,
                                                                                    const vector<Index>& testing_indices) const
{
    const Index samples_number = targets.dimension(0);
    const Index targets_number = targets.dimension(1);

    Tensor< Tensor<Index, 1>, 2> multiple_classification_rates(targets_number, targets_number);

    // Count instances per class

    const Tensor<Index, 2> confusion = calculate_confusion_multiple_classification(targets, outputs);

    for(Index i = 0; i < targets_number; i++)
        for(Index j = 0; j < targets_number; j++)
            multiple_classification_rates(i, j).resize(confusion(i, j));

    // Save indices

    Index target_index;
    Index output_index;

    Tensor<Index, 2> indices(targets_number, targets_number);
    indices.setZero();

    for(Index i = 0; i < samples_number; i++)
    {
        target_index = maximal_index(targets.chip(i, 0));
        output_index = maximal_index(outputs.chip(i, 0));

        multiple_classification_rates(target_index, output_index)(indices(target_index, output_index)) 
            = testing_indices[i];

        indices(target_index, output_index)++;
    }

    return multiple_classification_rates;
}


Tensor<string, 2> TestingAnalysis::calculate_well_classified_samples(const Tensor<type, 2>& targets,
                                                                      const Tensor<type, 2>& outputs,
                                                                      const vector<string>& labels) const
{
    const Index samples_number = targets.dimension(0);

    Tensor<string, 2> well_lassified_samples(samples_number, 4);

    Index predicted_class;
    Index actual_class;
    Index number_of_well_classified = 0;
    string class_name;

    const vector<string> target_variables_names = data_set->get_variable_names(DataSet::VariableUse::Target);

    for(Index i = 0; i < samples_number; i++)
    {
        predicted_class = maximal_index(outputs.chip(i, 0));
        actual_class = maximal_index(targets.chip(i, 0));

        if(actual_class != predicted_class) continue;

        well_lassified_samples(number_of_well_classified, 0) = labels[i];
        class_name = target_variables_names[actual_class];
        well_lassified_samples(number_of_well_classified, 1) = class_name;
        class_name = target_variables_names[predicted_class];
        well_lassified_samples(number_of_well_classified, 2) = class_name;
        well_lassified_samples(number_of_well_classified, 3) = to_string(double(outputs(i, predicted_class)));

        number_of_well_classified++;
    }

    const Eigen::array<Index, 2> offsets = {0, 0};
    const Eigen::array<Index, 2> extents = {number_of_well_classified, 4};

    return well_lassified_samples.slice(offsets, extents);
}


Tensor<string, 2> TestingAnalysis::calculate_misclassified_samples(const Tensor<type, 2>& targets,
                                                                   const Tensor<type, 2>& outputs,
                                                                   const vector<string>& labels) const
{
    const Index samples_number = targets.dimension(0);

    Index predicted_class;
    Index actual_class;
    string class_name;

    const vector<string> target_variables_names = neural_network->get_output_names();

    Index count_misclassified = 0;

    for(Index i = 0; i < samples_number; i++)
    {
        predicted_class = maximal_index(outputs.chip(i, 0));
        actual_class = maximal_index(targets.chip(i, 0));

        if(actual_class != predicted_class)
            count_misclassified++;
    }

    Tensor<string, 2> misclassified_samples(count_misclassified, 4);

    Index j = 0;

    for(Index i = 0; i < samples_number; i++)
    {
        predicted_class = maximal_index(outputs.chip(i, 0));
        actual_class = maximal_index(targets.chip(i, 0));

        if(actual_class == predicted_class) continue;

        misclassified_samples(j, 0) = labels[i];
        class_name = target_variables_names[actual_class];
        misclassified_samples(j, 1) = class_name;
        class_name = target_variables_names[predicted_class];
        misclassified_samples(j, 2) = class_name;
        misclassified_samples(j, 3) = to_string(double(outputs(i, predicted_class)));
        j++;
    }

//    Eigen::array<Index, 2> offsets = {0, 0};
//    Eigen::array<Index, 2> extents = {number_of_misclassified, 4};

//    return misclassified_samples.slice(offsets, extents);
    return misclassified_samples;
}


void TestingAnalysis::save_well_classified_samples(const Tensor<type, 2>& targets,
                                                    const Tensor<type, 2>& outputs,
                                                    const vector<string>& labels,
                                                    const filesystem::path& file_name) const
{
    const Tensor<string,2> well_classified_samples = calculate_well_classified_samples(targets,
                                                                                       outputs,
                                                                                       labels);

    ofstream file(file_name);

    file << "sample_name,actual_class,predicted_class,probability" << endl;

    for(Index i = 0; i < well_classified_samples.dimension(0); i++)
        file << well_classified_samples(i, 0) << ","
             << well_classified_samples(i, 1) << ","
             << well_classified_samples(i, 2) << ","
             << well_classified_samples(i, 3) << endl;

    file.close();
}


void TestingAnalysis::save_misclassified_samples(const Tensor<type, 2>& targets,
                                                 const Tensor<type, 2>& outputs,
                                                 const vector<string>& labels,
                                                 const filesystem::path& file_name) const
{
    const Tensor<string,2> misclassified_samples = calculate_misclassified_samples(targets,
                                                                                   outputs,
                                                                                   labels);

    ofstream file(file_name);

    file << "sample_name,actual_class,predicted_class,probability" << endl;

    for(Index i = 0; i < misclassified_samples.dimension(0); i++)
        file << misclassified_samples(i, 0) << ","
             << misclassified_samples(i, 1) << ","
             << misclassified_samples(i, 2) << ","
             << misclassified_samples(i, 3) << endl;

    file.close();
}


void TestingAnalysis::save_well_classified_samples_statistics(const Tensor<type, 2>& targets,
                                                              const Tensor<type, 2>& outputs,
                                                              const vector<string>& labels,
                                                              const filesystem::path& file_name) const
{
    const Tensor<string, 2> well_classified_samples = calculate_well_classified_samples(targets,
                                                                                        outputs,
                                                                                        labels);

    Tensor<type, 1> well_classified_numerical_probabilities(well_classified_samples.dimension(0));

    for(Index i = 0; i < well_classified_numerical_probabilities.size(); i++)
        well_classified_numerical_probabilities(i) = type(::atof(well_classified_samples(i, 3).c_str()));

    ofstream file(file_name);

    file << "minimum,maximum,mean,std" << endl
         << well_classified_numerical_probabilities.minimum() << ","
         << well_classified_numerical_probabilities.maximum() << ","
         << well_classified_numerical_probabilities.mean() << ","
         << standard_deviation(well_classified_numerical_probabilities);
}


void TestingAnalysis::save_misclassified_samples_statistics(const Tensor<type, 2>& targets,
                                                            const Tensor<type, 2>& outputs,
                                                            const vector<string>& labels,
                                                            const filesystem::path& statistics_file_name) const
{
    const Tensor<string, 2> misclassified_samples = calculate_misclassified_samples(targets,
                                                                                    outputs,
                                                                                    labels);

    Tensor<type, 1> misclassified_numerical_probabilities(misclassified_samples.dimension(0));

    for(Index i = 0; i < misclassified_numerical_probabilities.size(); i++)
        misclassified_numerical_probabilities(i) = type(::atof(misclassified_samples(i, 3).c_str()));

    ofstream classification_statistics_file(statistics_file_name);
    classification_statistics_file << "minimum,maximum,mean,std" << endl;
    classification_statistics_file << misclassified_numerical_probabilities.minimum() << ",";
    classification_statistics_file << misclassified_numerical_probabilities.maximum() << ",";
    classification_statistics_file << misclassified_numerical_probabilities.mean() << ",";
    classification_statistics_file << standard_deviation(misclassified_numerical_probabilities);
}


void TestingAnalysis::save_well_classified_samples_probability_histogram(const Tensor<type, 2>& targets,
                                                                         const Tensor<type, 2>& outputs,
                                                                         const vector<string>& labels,
                                                                         const filesystem::path& histogram_file_name) const
{
    const Tensor<string, 2> well_classified_samples = calculate_well_classified_samples(targets,
                                                                                        outputs,
                                                                                        labels);

    Tensor<type, 1> well_classified_numerical_probabilities(well_classified_samples.dimension(0));

    for(Index i = 0; i < well_classified_numerical_probabilities.size(); i++)
        well_classified_numerical_probabilities(i) = type(::atof(well_classified_samples(i, 3).c_str()));

    const Histogram misclassified_samples_histogram(well_classified_numerical_probabilities);

    misclassified_samples_histogram.save(histogram_file_name);
}


void TestingAnalysis::save_well_classified_samples_probability_histogram(const Tensor<string, 2>& well_classified_samples,
                                                                         const filesystem::path& histogram_file_name) const
{
    Tensor<type, 1> well_classified_numerical_probabilities(well_classified_samples.dimension(0));

    for(Index i = 0; i < well_classified_numerical_probabilities.size(); i++)
        well_classified_numerical_probabilities(i) = type(::atof(well_classified_samples(i, 3).c_str()));

    const Histogram misclassified_samples_histogram(well_classified_numerical_probabilities);

    misclassified_samples_histogram.save(histogram_file_name);
}


void TestingAnalysis::save_misclassified_samples_probability_histogram(const Tensor<type, 2>& targets,
                                                                       const Tensor<type, 2>& outputs,
                                                                       const vector<string>& labels,
                                                                       const filesystem::path& histogram_file_name) const
{
    const Tensor<string, 2> misclassified_samples = calculate_misclassified_samples(targets,
                                                                                    outputs,
                                                                                    labels);

    Tensor<type, 1> misclassified_numerical_probabilities(misclassified_samples.dimension(0));

    for(Index i = 0; i < misclassified_numerical_probabilities.size(); i++)
        misclassified_numerical_probabilities(i) = type(::atof(misclassified_samples(i, 3).c_str()));

    const Histogram misclassified_samples_histogram(misclassified_numerical_probabilities);

    misclassified_samples_histogram.save(histogram_file_name);
}


void TestingAnalysis::save_misclassified_samples_probability_histogram(const Tensor<string, 2>& misclassified_samples,
                                                                       const filesystem::path& histogram_file_name) const
{
    Tensor<type, 1> misclassified_numerical_probabilities(misclassified_samples.dimension(0));

    for(Index i = 0; i < misclassified_numerical_probabilities.size(); i++)
        misclassified_numerical_probabilities(i) = type(::atof(misclassified_samples(i, 3).c_str()));

    const Histogram misclassified_samples_histogram(misclassified_numerical_probabilities);

    misclassified_samples_histogram.save(histogram_file_name);
}


Tensor<Tensor<type, 1>, 1> TestingAnalysis::calculate_error_autocorrelation(const Index& maximum_lags_number) const
{
    const Tensor<type, 2> inputs = data_set->get_data(DataSet::SampleUse::Testing, DataSet::VariableUse::Input);

    const Tensor<type, 2> targets = data_set->get_data(DataSet::SampleUse::Testing, DataSet::VariableUse::Target);

    const Tensor<type, 2> outputs = neural_network->calculate_outputs(inputs);

    const Index targets_number = data_set->get_variables_number(DataSet::VariableUse::Target);

    const Tensor<type, 2> error = outputs - targets;

    Tensor<Tensor<type, 1>, 1> error_autocorrelations(targets_number);

    for(Index i = 0; i < targets_number; i++)
        error_autocorrelations[i] = autocorrelations(thread_pool_device.get(), error.chip(i,1), maximum_lags_number);

    return error_autocorrelations;
}


Tensor<Tensor<type, 1>, 1> TestingAnalysis::calculate_inputs_errors_cross_correlation(const Index& lags_number) const
{
    const Index targets_number = data_set->get_variables_number(DataSet::VariableUse::Target);

    const Tensor<type, 2> inputs = data_set->get_data(DataSet::SampleUse::Testing, DataSet::VariableUse::Input);

    const Tensor<type, 2> targets = data_set->get_data(DataSet::SampleUse::Testing, DataSet::VariableUse::Target);

    const Tensor<type, 2> outputs = neural_network->calculate_outputs(inputs);

    const Tensor<type, 2> errors = outputs - targets;

    Tensor<Tensor<type, 1>, 1> inputs_errors_cross_correlation(targets_number);

    for(Index i = 0; i < targets_number; i++)
        inputs_errors_cross_correlation[i] = cross_correlations(thread_pool_device.get(), 
            inputs.chip(i,1), errors.chip(i,1), lags_number);

    return inputs_errors_cross_correlation;
}


pair<type, type> TestingAnalysis::test_transformer() const
{
    cout << "Testing transformer..." << endl;

    Transformer* transformer = static_cast<Transformer*>(neural_network);
    LanguageDataSet* language_data_set = static_cast<LanguageDataSet*>(data_set);

    const Tensor<type, 2> input = language_data_set->get_data(DataSet::SampleUse::Testing, DataSet::VariableUse::Input);
    const Tensor<type, 2> context = language_data_set->get_data(DataSet::SampleUse::Testing, DataSet::VariableUse::Decoder);
    const Tensor<type, 2> target = language_data_set->get_data(DataSet::SampleUse::Testing, DataSet::VariableUse::Target);

    const Index testing_batch_size = input.dimension(0) > 2000 ? 2000 : input.dimension(0);

    Tensor<type, 2> testing_input(testing_batch_size, input.dimension(1));

    for(Index i = 0; i < testing_batch_size; i++)
        testing_input.chip(i, 0) = input.chip(i, 0);

    Tensor<type, 2> testing_context(testing_batch_size, context.dimension(1));

    for(Index i = 0; i < testing_batch_size; i++)
        testing_context.chip(i, 0) = context.chip(i, 0);

    Tensor<type, 2> testing_target(testing_batch_size, target.dimension(1));

    for(Index i = 0; i < testing_batch_size; i++)
        testing_target.chip(i, 0) = target.chip(i, 0);

    const Tensor<type, 3> outputs = transformer->calculate_outputs(testing_input, testing_context);


    // cout<<"English:"<<endl;
    // cout<<testing_context.chip(10,0)<<endl;
    // for(Index i = 0; i < testing_context.dimension(1); i++){
    //     cout<<language_data_set->get_context_vocabulary()[Index(testing_context(10,i))]<<" ";
    // }
    // cout<<endl;
    // cout<<endl;
    // cout<<"Spanish:"<<endl;
    // cout<<testing_input.chip(10,0)<<endl;
    // for(Index i = 0; i < testing_input.dimension(1); i++){
    //     cout<<language_data_set->get_completion_vocabulary()[Index(testing_input(10,i))]<<" ";
    // }
    // cout<<endl;
    // cout<<endl;
    // cout<<"Prediction:"<<endl;

    // for(Index j = 0; j < outputs.dimension(1); j++){
    //     type max = outputs(10, j, 0);
    //     Index index = 0;
    //     for(Index i = 1; i < outputs.dimension(2); i++){
    //         if(max < outputs(10,j,i)){
    //             index = i;
    //             max = outputs(10,j,i);
    //         }else{continue;}
    //     }
    //     cout<<index<<" ";
    // }
    // cout<<endl;
    // for(Index j = 0; j < outputs.dimension(1); j++){
    //     type max = outputs(10, j, 0);
    //     Index index = 0;
    //     for(Index i = 1; i < outputs.dimension(2); i++){
    //         if(max < outputs(10,j,i)){
    //             index = i;
    //             max = outputs(10,j,i);
    //         }else{continue;}
    //     }
    //     cout<<language_data_set->get_completion_vocabulary()[index]<<" ";
    // }


    const type error = calculate_cross_entropy_error_3d(outputs, testing_target);

    const type accuracy = calculate_masked_accuracy(outputs, testing_target);

    return pair<type, type>(error, accuracy);
}


string TestingAnalysis::test_transformer(const vector<string>& context_string, const bool& imported_vocabulary) const
{   cout<<endl;
    cout<<"Testing transformer..."<<endl;

    Transformer* transformer = static_cast<Transformer*>(neural_network);

    return transformer->calculate_outputs(context_string);
}


Tensor<type, 1> TestingAnalysis::calculate_binary_classification_tests() const
{
    const Tensor<Index, 2> confusion = calculate_confusion();

    const Index true_positive = confusion(0,0);
    const Index false_positive = confusion(1,0);
    const Index false_negative = confusion(0,1);
    const Index true_negative = confusion(1,1);

    const type classification_accuracy = (true_positive + true_negative + false_positive + false_negative == 0)
    ? type(0)
    : type(true_positive + true_negative) / type(true_positive + true_negative + false_positive + false_negative);

    const type error_rate = (true_positive + true_negative + false_positive + false_negative == 0)
    ? type(0)
    : type(false_positive + false_negative) / type(true_positive + true_negative + false_positive + false_negative);

    const type sensitivity = (true_positive + false_negative == 0)
    ? type(0)
    : type(true_positive) / type(true_positive + false_negative);

    const type false_positive_rate = (false_positive + true_negative == 0)
    ? type(0)
    : type(false_positive) / type(false_positive + true_negative);

    const type specificity = (false_positive + true_negative == 0)
    ? type(0)
    : type(true_negative) / type(true_negative + false_positive);

    const type precision = (true_positive + false_positive == 0)
    ? type(0)
    : type(true_positive) / type(true_positive + false_positive);

    type positive_likelihood;

    if(abs(classification_accuracy - type(1)) < NUMERIC_LIMITS_MIN)
        positive_likelihood = type(1);
    else if(abs(type(1) - specificity) < NUMERIC_LIMITS_MIN)
        positive_likelihood = type(0);
    else
        positive_likelihood = sensitivity/(type(1) - specificity);

    type negative_likelihood;

    if(Index(classification_accuracy) == 1)
        negative_likelihood = type(1);
    else if(abs(type(1) - sensitivity) < NUMERIC_LIMITS_MIN)
        negative_likelihood = type(0);
    else
        negative_likelihood = specificity/(type(1) - sensitivity);

    const type f1_score = (2 * true_positive + false_positive + false_negative == 0)
    ? type(0)
    : type(2.0) * type(true_positive) / (type(2.0) * type(true_positive) + type(false_positive) + type(false_negative));

    const type false_discovery_rate = (false_positive + true_positive == 0)
    ? type(0)
    : type(false_positive) / type(false_positive + true_positive);

    const type false_negative_rate = (false_negative + true_positive == 0)
    ? type(0)
    : type(false_negative) / type(false_negative + true_positive);

    const type negative_predictive_value = (true_negative + false_negative == 0)
    ? type(0)
    : type(true_negative) / type(true_negative + false_negative);

    const type Matthews_correlation_coefficient = ((true_positive + false_positive) * (true_positive + false_negative) * (true_negative + false_positive) * (true_negative + false_negative) == 0)
    ? type(0)
    : type(true_positive * true_negative - false_positive * false_negative) / type(sqrt((true_positive + false_positive) * (true_positive + false_negative) * (true_negative + false_positive) * (true_negative + false_negative)));

    const type informedness = sensitivity + specificity - type(1);

    const type markedness = (true_negative + false_positive == 0)
                          ? precision - type(1)
                          : precision + type(true_negative) / type(true_negative + false_positive) - type(1);

    Tensor<type, 1> binary_classification_test(15);

    binary_classification_test.setValues(
    {classification_accuracy,
    error_rate,
    sensitivity,
    specificity,
    precision,
    positive_likelihood,
    negative_likelihood,
    f1_score,
    false_positive_rate,
    false_discovery_rate,
    false_negative_rate,
    negative_predictive_value,
    Matthews_correlation_coefficient,
    informedness,
    markedness});

    return binary_classification_test;
}


void TestingAnalysis::print_binary_classification_tests() const
{
    const Tensor<type, 1> binary_classification_tests = calculate_binary_classification_tests();

    cout << "Binary classification tests: " << endl
         << "Classification accuracy : " << binary_classification_tests[0] << endl
         << "Error rate              : " << binary_classification_tests[1] << endl
         << "Sensitivity             : " << binary_classification_tests[2] << endl
         << "Specificity             : " << binary_classification_tests[3] << endl;
}


Tensor<type, 2> TestingAnalysis::calculate_multiple_classification_tests() const
{
    //const Index inputs_number = neural_network->get_inputs_number();

    const Index targets_number = data_set->get_variables_number(DataSet::VariableUse::Target);

    //const Index outputs_number = neural_network->get_outputs_number();

    Tensor<type,2> multiple_classification_tests(targets_number + 2, 3);

    const Tensor<Index, 2> confusion = calculate_confusion();

    Index true_positives = 0;
    //Index true_negatives = 0;
    Index false_positives = 0;
    Index false_negatives = 0;

    type total_precision = type(0);
    type total_recall = type(0);
    type total_f1_score= type(0);

    type total_weighted_precision = type(0);
    type total_weighted_recall = type(0);
    type total_weighted_f1_score= type(0);

    Index total_samples = 0;

    for(Index target_index = 0; target_index < targets_number; target_index++)
    {
        true_positives = confusion(target_index, target_index);

        const Tensor<Index,0> row_sum = confusion.chip(target_index,0).sum();
        const Tensor<Index,0> column_sum = confusion.chip(target_index,1).sum();

        false_negatives = row_sum(0) - true_positives;
        false_positives= column_sum(0) - true_positives;

        const type precision = (true_positives + false_positives == 0)
                             ? type(0)
                             : type(true_positives) / type(true_positives + false_positives);

        const type recall = (true_positives + false_negatives == 0)
                          ? type(0)
                          : type(true_positives) / type(true_positives + false_negatives);

        const type f1_score = (precision + recall == 0)
                            ? type(0)
                            : type(2 * precision * recall) / type(precision + recall);

        // Save results

        multiple_classification_tests(target_index, 0) = precision;
        multiple_classification_tests(target_index, 1) = recall;
        multiple_classification_tests(target_index, 2) = f1_score;

        total_precision += precision;
        total_recall += recall;
        total_f1_score += f1_score;

        total_weighted_precision += precision * type(row_sum(0));
        total_weighted_recall += recall * type(row_sum(0));
        total_weighted_f1_score += f1_score * type(row_sum(0));

        total_samples += row_sum(0);
    }

    // Averages

    multiple_classification_tests(targets_number, 0) = total_precision/targets_number;
    multiple_classification_tests(targets_number, 1) = total_recall/targets_number;
    multiple_classification_tests(targets_number, 2) = total_f1_score/targets_number;

    multiple_classification_tests(targets_number + 1, 0) = total_weighted_precision/total_samples;
    multiple_classification_tests(targets_number + 1, 1) = total_weighted_recall/total_samples;
    multiple_classification_tests(targets_number + 1, 2) = total_weighted_f1_score/total_samples;

    return multiple_classification_tests;
}


type TestingAnalysis::calculate_logloss() const
{
    const Tensor<type, 2> inputs = data_set->get_data(DataSet::SampleUse::Testing, DataSet::VariableUse::Input);

    const Tensor<type, 2> targets = data_set->get_data(DataSet::SampleUse::Testing, DataSet::VariableUse::Target);

    const Tensor<type, 2> outputs = neural_network->calculate_outputs(inputs);

    const Index testing_samples_number = data_set->get_samples_number(DataSet::SampleUse::Testing);

    type logloss = type(0);

    for(Index i = 0; i < testing_samples_number; i++)
        logloss += targets(i,0)*log(outputs(i,0)) + (type(1) - targets(i,0))*log(type(1) - outputs(i,0));

    return -logloss/type(testing_samples_number);
}


void TestingAnalysis::to_XML(XMLPrinter& printer) const
{
    printer.OpenElement("TestingAnalysis");

    add_xml_element(printer, "Display", to_string(display));

    printer.CloseElement();
}


void TestingAnalysis::from_XML(const XMLDocument& document)
{
    const XMLElement* root_element = document.FirstChildElement("TestingAnalysis");

    if(!root_element)
        throw runtime_error("Testing analysis element is nullptr.\n");

    set_display(read_xml_bool(root_element, "Display"));
}


void TestingAnalysis::save(const filesystem::path& file_name) const
{
    ofstream file(file_name);

    if (!file.is_open())
        return;

    XMLPrinter printer;

    to_XML(printer);

    file << printer.CStr();
}


void TestingAnalysis::load(const filesystem::path& file_name)
{
    XMLDocument document;

    if (document.LoadFile(file_name.string().c_str()))
        throw runtime_error("Cannot load XML file " + file_name.string() + ".\n");

    from_XML(document);
}


void TestingAnalysis::GoodnessOfFitAnalysis::set(const Tensor<type, 1> &new_targets,
                                                 const Tensor<type, 1> &new_outputs,
                                                 const type &new_determination)
{
    targets = new_targets;
    outputs = new_outputs;
    determination = new_determination;
}


void TestingAnalysis::GoodnessOfFitAnalysis::save(const filesystem::path& file_name) const
{
    ofstream file(file_name);

    file << "Goodness-of-fit analysis\n"
         << "Determination: " << determination << endl;

    file.close();
}


void TestingAnalysis::GoodnessOfFitAnalysis::print() const
{
    cout << "Goodness-of-fit analysis" << endl
         << "Determination: " << determination << endl;

    // cout << targets << endl;
    // cout << outputs << endl;
}

}


// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2024 Artificial Intelligence Techniques, SL.
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
