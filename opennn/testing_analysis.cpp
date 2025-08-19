//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   T E S T I N G   A N A L Y S I S   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "testing_analysis.h"
#include "correlations.h"
#include "language_dataset.h"
#include "transformer.h"
#include "statistics.h"
#include "unscaling_layer.h"

namespace opennn
{

TestingAnalysis::TestingAnalysis(const NeuralNetwork* new_neural_network, const Dataset* new_dataset)
{
    neural_network = const_cast<NeuralNetwork*>(new_neural_network);
    dataset = const_cast<Dataset*>(new_dataset);

    const unsigned int threads_number = thread::hardware_concurrency();

    thread_pool = make_unique<ThreadPool>(threads_number);
    thread_pool_device = make_unique<ThreadPoolDevice>(thread_pool.get(), threads_number);
}


NeuralNetwork* TestingAnalysis::get_neural_network() const
{
    return neural_network;
}


Dataset* TestingAnalysis::get_dataset() const
{
    return dataset;
}


const bool& TestingAnalysis::get_display() const
{
    return display;
}

void TestingAnalysis::set_threads_number(const int& new_threads_number)
{
    thread_pool.reset();
    thread_pool_device.reset();

    thread_pool = make_unique<ThreadPool>(new_threads_number);
    thread_pool_device = make_unique<ThreadPoolDevice>(thread_pool.get(), new_threads_number);
}


void TestingAnalysis::set_neural_network(NeuralNetwork* new_neural_network)
{
    neural_network = new_neural_network;
}


void TestingAnalysis::set_dataset(Dataset* new_dataset)
{
    dataset = new_dataset;
}


void TestingAnalysis::set_display(const bool& new_display)
{
    display = new_display;
}


void TestingAnalysis::check() const
{
    if(!neural_network)
        throw runtime_error("Neural network pointer is nullptr.\n");

    if(!dataset)
        throw runtime_error("Data set pointer is nullptr.\n");
}


Tensor<Correlation, 1> TestingAnalysis::linear_correlation() const
{
    const Tensor<type, 2> inputs = dataset->get_data("Testing", "Input");

    const Tensor<type, 2> targets = dataset->get_data("Testing", "Target");

    const Tensor<type, 2> outputs = neural_network->calculate_outputs<2,2>(inputs);

    return linear_correlation(targets, outputs);
}


Tensor<Correlation, 1> TestingAnalysis::linear_correlation(const Tensor<type, 2>& target, const Tensor<type, 2>& output) const
{
    const Index outputs_number = dataset->get_variables_number("Target");

    Tensor<Correlation, 1> linear_correlation(outputs_number);

    for(Index i = 0; i < outputs_number; i++)
        linear_correlation(i) = opennn::linear_correlation(thread_pool_device.get(), output.chip(i,1), target.chip(i,1));

    return linear_correlation;
}


void TestingAnalysis::print_linear_correlations() const
{
    const Tensor<Correlation, 1> linear_correlations = linear_correlation();

    const vector<string> targets_name = dataset->get_variable_names("Target");

    const Index targets_number = linear_correlations.size();

    for(Index i = 0; i < targets_number; i++)
        cout << targets_name[i] << " correlation: " << linear_correlations[i].r << endl;
}


Tensor<TestingAnalysis::GoodnessOfFitAnalysis, 1> TestingAnalysis::perform_goodness_of_fit_analysis() const
{
    check();

    // Data set

    const Index testing_samples_number = dataset->get_samples_number("Testing");

    if(testing_samples_number == Index(0))
        throw runtime_error("Number of testing samples is zero.\n");

    const Tensor<type, 2> testing_input_data = dataset->get_data("Testing", "Input");

    const Tensor<type, 2> testing_target_data = dataset->get_data("Testing", "Target");

    // Neural network

    const Index outputs_number = neural_network->get_outputs_number();

    const Tensor<type, 2> testing_output_data = neural_network->calculate_outputs<2,2>(testing_input_data);

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


Tensor<type, 2> TestingAnalysis::calculate_error() const
{
    const Tensor<type, 2> inputs = dataset->get_data("Testing", "Input");

    const Tensor<type, 2> targets = dataset->get_data("Testing", "Target");

    const Tensor<type, 2> outputs = neural_network->calculate_outputs<2,2>(inputs);

    const Tensor<type, 2> error = (targets - outputs);

    return error;
}

Tensor<type, 3> TestingAnalysis::calculate_error_data() const
{
    check();

    // Data set

    const Index testing_samples_number = dataset->get_samples_number("Testing");

    if(testing_samples_number == Index(0))
        throw runtime_error("Number of testing samples is zero.\n");

    const Tensor<type, 2> testing_input_data = dataset->get_data("Testing", "Input");

    const Tensor<type, 2> testing_target_data = dataset->get_data("Testing", "Target");

    // Neural network

    const Index outputs_number = neural_network->get_outputs_number();

    const Tensor<type, 2> testing_output_data = neural_network->calculate_outputs<2,2>(testing_input_data);

    Unscaling* unscaling_layer = static_cast<Unscaling*>(neural_network->get_first("Unscaling"));
    unscaling_layer->set_scalers(Scaler::MinimumMaximum);

    Descriptives desc;
    vector<Descriptives> descriptives = unscaling_layer->get_descriptives();

    const Tensor<type, 1>& output_minimums = unscaling_layer->get_minimums();
    const Tensor<type, 1>& output_maximums = unscaling_layer->get_maximums();

    for(Index i = 0; i < outputs_number; i++){
        type min_range=output_minimums[i];
        type max_range=output_maximums[i];
        desc.set(min_range, max_range);
        descriptives[i] = desc;
    }

    unscaling_layer->set_descriptives(descriptives);

    Tensor<type, 3> error_data(testing_samples_number, 3, outputs_number);

    const Tensor<type, 2> absolute_errors = (testing_target_data - testing_output_data).abs();

#pragma omp parallel for
    for(Index i = 0; i < outputs_number; i++)
    {
        for(Index j = 0; j < testing_samples_number; j++)
        {
            error_data(j, 0, i) = absolute_errors(j,i);
            error_data(j, 1, i) = absolute_errors(j,i)/abs(output_maximums(i)-output_minimums(i));
            error_data(j, 2, i) = absolute_errors(j,i)*type(100.0)/abs(output_maximums(i)-output_minimums(i));
        }
    }
    return error_data;
}


Tensor<type, 2> TestingAnalysis::calculate_percentage_error_data() const
{

    check();

    // Data set

    const Index testing_samples_number = dataset->get_samples_number("Testing");

    if(testing_samples_number == Index(0))
        throw runtime_error("Number of testing samples is zero.\n");

    const Tensor<type, 2> testing_input_data = dataset->get_data("Testing", "Input");

    const Tensor<type, 2> testing_target_data = dataset->get_data("Testing", "Target");

    // Neural network

    const Index outputs_number = neural_network->get_outputs_number();

    const Tensor<type, 2> testing_output_data = neural_network->calculate_outputs<2,2>(testing_input_data);

    Unscaling* unscaling_layer = static_cast<Unscaling*>(neural_network->get_first("Unscaling"));
    unscaling_layer->set_scalers(Scaler::MinimumMaximum);

    Descriptives desc;
    vector<Descriptives> descriptives = unscaling_layer->get_descriptives();

    const Tensor<type, 1>& output_minimums = unscaling_layer->get_minimums();
    const Tensor<type, 1>& output_maximums = unscaling_layer->get_maximums();

    for(Index i = 0; i < outputs_number; i++){
        type min_range=output_minimums[i];
        type max_range=output_maximums[i];
        desc.set(min_range, max_range);
        descriptives[i] = desc;
    }

    unscaling_layer->set_descriptives(descriptives);

    const Tensor<type, 2> errors = (testing_target_data - testing_output_data);

    // Error data

    Tensor<type, 2> error_data(testing_samples_number, outputs_number);

#pragma omp parallel for

    for(Index i = 0; i < testing_samples_number; i++)
    {
        for(Index j = 0; j < outputs_number; j++)
        {
            error_data(i, j) = errors(i, j)*type(100.0)/abs(output_maximums(j) - output_minimums(j));
        }
    }

    return error_data;
}


vector<Descriptives> TestingAnalysis::calculate_absolute_errors_descriptives() const
{
    const Tensor<type, 2> inputs = dataset->get_data("Testing", "Input");

    const Tensor<type, 2> targets = dataset->get_data("Testing", "Target");

    const Tensor<type, 2> outputs = neural_network->calculate_outputs<2,2>(inputs);

    dataset->set_raw_variable_scalers(Scaler::MinimumMaximum);
    dataset->scale_data();
    dataset->calculate_variable_descriptives();

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
    const Tensor<type, 2> inputs = dataset->get_data("Testing", "Input");

    const Tensor<type, 2> targets = dataset->get_data("Testing", "Target");

    const Tensor<type, 2> outputs = neural_network->calculate_outputs<2,2>(inputs);

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

    const Index testing_samples_number = dataset->get_samples_number("Testing");

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
    const Index targets_number = dataset->get_variables_number("Target");

    const vector<string> targets_name = dataset->get_variable_names("Target");

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

    const Tensor<type, 1> training_errors = calculate_errors("Training");
    const Tensor<type, 1> selection_errors = calculate_errors("Selection");
    const Tensor<type, 1> testing_errors = calculate_errors("Testing");

    Tensor<type, 2> errors(5, 3);

    errors.setValues({
        {training_errors(0), selection_errors(0), testing_errors(0)},
        {training_errors(1), selection_errors(1), testing_errors(1)},
        {training_errors(2), selection_errors(2), testing_errors(2)},
        {training_errors(3), selection_errors(3), testing_errors(3)},
        {training_errors(4), selection_errors(4), testing_errors(4)}
    });

    return errors;
}


Tensor<type, 2> TestingAnalysis::calculate_binary_classification_errors() const
{
    const Tensor<type, 1> training_errors = calculate_binary_classification_errors("Training");
    const Tensor<type, 1> selection_errors = calculate_binary_classification_errors("Selection");
    const Tensor<type, 1> testing_errors = calculate_binary_classification_errors("Testing");

    Tensor<type, 2> errors(7, 3);

    errors.setValues({
        {training_errors(0), selection_errors(0), testing_errors(0)},
        {training_errors(1), selection_errors(1), testing_errors(1)},
        {training_errors(2), selection_errors(2), testing_errors(2)},
        {training_errors(3), selection_errors(3), testing_errors(3)},
        {training_errors(4), selection_errors(4), testing_errors(4)},
        {training_errors(5), selection_errors(5), testing_errors(5)},
        {training_errors(6), selection_errors(6), testing_errors(6)}
    });

    return errors;
}


Tensor<type, 2> TestingAnalysis::calculate_multiple_classification_errors() const
{
    const Tensor<type, 1> training_errors = calculate_multiple_classification_errors("Training");
    const Tensor<type, 1> selection_errors = calculate_multiple_classification_errors("Training");
    const Tensor<type, 1> testing_errors = calculate_multiple_classification_errors("Training");

    Tensor<type, 2> errors(6, 3);

    errors.setValues({
        {training_errors(0), selection_errors(0), testing_errors(0)},
        {training_errors(1), selection_errors(1), testing_errors(1)},
        {training_errors(2), selection_errors(2), testing_errors(2)},
        {training_errors(3), selection_errors(3), testing_errors(3)},
        {training_errors(4), selection_errors(4), testing_errors(4)},
        {training_errors(5), selection_errors(5), testing_errors(5)}
    });
    return errors;
}


Tensor<type, 1> TestingAnalysis::calculate_errors(const Tensor<type, 2>& targets,
                                                  const Tensor<type, 2>& outputs) const
{
    const Index batch_size = outputs.dimension(0);

    Tensor<type, 1> errors(4);

    Tensor<type, 0> mean_squared_error;
    mean_squared_error.device(*thread_pool_device) = (outputs - targets).square().sum().sqrt();

    errors(0) = mean_squared_error(0);
    errors(1) = errors(0)/type(batch_size);
    errors(2) = sqrt(errors(1));
    errors(3) = calculate_normalized_squared_error(targets, outputs);

    return errors;
}


Tensor<type, 1> TestingAnalysis::calculate_errors(const string& sample_use) const
{
    const Tensor<type, 2> inputs = dataset->get_data(sample_use, "Input");

    const Tensor<type, 2> targets = dataset->get_data(sample_use, "Target");

    const Tensor<type, 2> outputs = neural_network->calculate_outputs<2,2>(inputs);

    return calculate_errors(targets, outputs);
}


Tensor<type, 1> TestingAnalysis::calculate_binary_classification_errors(const string& sample_use) const
{
    // Data set

    const Index training_samples_number = dataset->get_samples_number(sample_use);

    const Tensor<type, 2> inputs = dataset->get_data(sample_use, "Input");

    const Tensor<type, 2> targets = dataset->get_data(sample_use, "Target");

    // Neural network

    const Tensor<type, 2> outputs = neural_network->calculate_outputs<2,2>(inputs);

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


Tensor<type, 1> TestingAnalysis::calculate_multiple_classification_errors(const string& sample_use) const
{
    // Data set

    const Index training_samples_number = dataset->get_samples_number(sample_use);

    const Tensor<type, 2> inputs = dataset->get_data(sample_use, "Input");

    const Tensor<type, 2> targets = dataset->get_data(sample_use, "Target");

    // Neural network

    const Tensor<type, 2> outputs = neural_network->calculate_outputs<2,2>(inputs);

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

    type cross_entropy_error_2d = type(0);

#pragma omp parallel for reduction(+:cross_entropy_error_2d)

    for(Index i = 0; i < testing_samples_number; i++)
    {
        outputs_row = outputs.chip(i, 0);
        targets_row = targets.chip(i, 0);

        for(Index j = 0; j < outputs_number; j++)
        {
            outputs_row(j) = clamp(outputs_row(j), type(1.0e-6), numeric_limits<type>::max());

            cross_entropy_error_2d -=
                targets_row(j)*log(outputs_row(j)) + (type(1) - targets_row(j))*log(type(1) - outputs_row(j));
        }
    }

    return cross_entropy_error_2d/type(testing_samples_number);
}


type TestingAnalysis::calculate_cross_entropy_error_3d(const Tensor<type, 3>& outputs, const Tensor<type, 2>& targets) const
{
    const Index batch_size = outputs.dimension(0);
    const Index outputs_number = outputs.dimension(1);

    Tensor<type, 2> errors(batch_size, outputs_number);
    Tensor<type, 2> predictions(batch_size, outputs_number);
    Tensor<bool, 2> matches(batch_size, outputs_number);
    Tensor<bool, 2> mask(batch_size, outputs_number);

    Tensor<type, 0> cross_entropy_error_2d;
    mask = targets != targets.constant(0);

    Tensor<type, 0> mask_sum;
    mask_sum = mask.cast<type>().sum();

    for(Index i = 0; i < batch_size; i++)
        for(Index j = 0; j < outputs_number; j++)
            errors(i, j) = -log(outputs(i, j, Index(targets(i, j))));

    errors = errors * mask.cast<type>();

    cross_entropy_error_2d = errors.sum();

    return cross_entropy_error_2d(0) / mask_sum(0);
}


type TestingAnalysis::calculate_weighted_squared_error(const Tensor<type, 2>& targets,
                                                       const Tensor<type, 2>& outputs,
                                                       const Tensor<type, 1>& weights) const
{
    type negatives_weight;
    type positives_weight;

    if(weights.size() != 2)
    {
        const Tensor<Index, 1> target_distribution = dataset->calculate_target_distribution();

        const Index negatives_number = target_distribution[0];
        const Index positives_number = target_distribution[1];

        negatives_weight = type(1);

        if(negatives_number == 0 || positives_number == 0)
            positives_weight = type(0);
        else
            positives_weight = type(negatives_number/positives_number);

    }
    else
    {
        positives_weight = weights[0];
        negatives_weight = weights[1];
    }

    const Tensor<bool, 2> if_sentence = (targets == targets.constant(type(1))).cast<bool>();
    const Tensor<bool, 2> else_sentence = (targets == targets.constant(type(0))).cast<bool>();

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
    const Index batch_size = outputs.dimension(0);
    const Index outputs_number = outputs.dimension(1);

    Tensor<type, 2> predictions(batch_size, outputs_number);
    Tensor<bool, 2> matches(batch_size, outputs_number);
    Tensor<bool, 2> mask(batch_size, outputs_number);

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

    for(Index i = 0; i < testing_samples_number; i++)
    {
        const bool is_target_positive = targets(i, 0) >= decision_threshold;
        const bool is_output_positive = outputs(i, 0) >= decision_threshold;

        if (is_target_positive && is_output_positive)
            true_positive++;
        else if (is_target_positive && !is_output_positive)
            false_negative++;
        else if (!is_target_positive && is_output_positive)
            false_positive++;
        else  // !is_target_positive && !is_output_positive
            true_negative++;
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

    Tensor<Index, 1> positives_negatives_rate(2);
    positives_negatives_rate.setValues({confusion(0,0) + confusion(0,1), confusion(1,0) + confusion(1,1)});

    return positives_negatives_rate;
}


Tensor<Index, 2> TestingAnalysis::calculate_confusion(const type& decision_threshold) const
{
    check();

    Tensor<type, 2> inputs = dataset->get_data("Testing", "Input");

    const Tensor<type, 2> targets = dataset->get_data("Testing", "Target");

    const Index samples_number = targets.dimension(0);

    const dimensions input_dimensions = dataset->get_dimensions("Input");

    if(input_dimensions.size() == 1)
    {
        const Tensor<type, 2> outputs = neural_network->calculate_outputs<2,2>(inputs);

        return calculate_confusion(outputs, targets, decision_threshold);
    }
    else if(input_dimensions.size() == 3)
    {
        type* input_data = inputs.data();

        Tensor<type, 4> inputs_4d(samples_number,
                                  input_dimensions[0],
                                  input_dimensions[1],
                                  input_dimensions[2]);

        memcpy(inputs_4d.data(), input_data, samples_number * inputs.dimension(1)*sizeof(type));

        const Tensor<type, 2> outputs = neural_network->calculate_outputs<4,2>(inputs_4d);

        return calculate_confusion(outputs, targets, decision_threshold);
    }

    return Tensor<Index, 2>();
}


Tensor<Index, 2> TestingAnalysis::calculate_sentimental_analysis_transformer_confusion() const
{
    // Transformer* transformer = static_cast<Transformer*>(neural_network);
    // LanguageDataset* language_dataset = static_cast<LanguageDataset*>(dataset);

    // const Tensor<type, 2> inputs = language_dataset->get_data("Testing", "Input");
    // const Tensor<type, 2> context = language_dataset->get_data("Testing", "Decoder");
    // const Tensor<type, 2> targets = language_dataset->get_data("Testing", "Target");

    // const dimensions input_dimensions = dataset->get_dimensions("Input");

    // const Index testing_batch_size = inputs.dimension(0) > 2000 ? 2000 : inputs.dimension(0);

    // Tensor<type, 2> testing_input(testing_batch_size, inputs.dimension(1));    // Transformer* transformer = static_cast<Transformer*>(neural_network);
    // LanguageDataset* language_dataset = static_cast<LanguageDataset*>(dataset);

    // const Tensor<type, 2> inputs = language_dataset->get_data("Testing", "Input");
    // const Tensor<type, 2> context = language_dataset->get_data("Testing", "Decoder");
    // const Tensor<type, 2> targets = language_dataset->get_data("Testing", "Target");

    // const dimensions input_dimensions = dataset->get_dimensions("Input");

    // const Index testing_batch_size = inputs.dimension(0) > 2000 ? 2000 : inputs.dimension(0);

    // Tensor<type, 2> testing_input(testing_batch_size, inputs.dimension(1));
    // Tensor<type, 2> testing_context(testing_batch_size, context.dimension(1));
    // Tensor<type, 2> testing_target(testing_batch_size, targets.dimension(1));

    // for(Index i = 0; i < testing_batch_size; i++)
    // {
    //     testing_input.chip(i, 0) = inputs.chip(i, 0);
    //     testing_context.chip(i, 0) = context.chip(i, 0);
    //     testing_target.chip(i, 0) = targets.chip(i, 0);
    // }

    // if(input_dimensions.size() == 1)
    // {
    //     const Tensor<type, 3> outputs = transformer->calculate_outputs(testing_input, testing_context);

    //     Tensor<type, 2> reduced_outputs(outputs.dimension(0), targets.dimension(1));

    //     for (Index i = 0; i < outputs.dimension(0); i++)
    //     {
    //         reduced_outputs(i,0) = outputs(i,1,9);
    //         reduced_outputs(i,1) = outputs(i,1,10);
    //     }

    //     // Tensor<type, 2> reduced_outputs(outputs.dimension(0), outputs.dimension(1));
    //     // type index;
    //     // type max;

    //     // for (Index i = 0; i < outputs.dimension(0); i++) {
    //     //     for (Index j = 0; j < outputs.dimension(1); j++) {
    //     //         index = 0;
    //     //         max = outputs(i,j,0);
    //     //         for(Index k = 1; k < outputs.dimension(2); k++)
    //     //             if(max < outputs(i,j,k)){
    //     //                 index = type(k);
    //     //                 max = outputs(i,j,k);
    //     //             }
    //     //         reduced_outputs(i,j) = index;
    //     //     }
    //     // }

    //     return calculate_confusion(reduced_outputs, testing_target);
    // }

    // Tensor<type, 2> testing_context(testing_batch_size, context.dimension(1));
    // Tensor<type, 2> testing_target(testing_batch_size, targets.dimension(1));

    // for(Index i = 0; i < testing_batch_size; i++)
    // {
    //     testing_input.chip(i, 0) = inputs.chip(i, 0);
    //     testing_context.chip(i, 0) = context.chip(i, 0);
    //     testing_target.chip(i, 0) = targets.chip(i, 0);
    // }

    // if(input_dimensions.size() == 1)
    // {
    //     const Tensor<type, 3> outputs = transformer->calculate_outputs(testing_input, testing_context);

    //     Tensor<type, 2> reduced_outputs(outputs.dimension(0), targets.dimension(1));

    //     for (Index i = 0; i < outputs.dimension(0); i++)
    //     {
    //         reduced_outputs(i,0) = outputs(i,1,9);
    //         reduced_outputs(i,1) = outputs(i,1,10);
    //     }

    //     // Tensor<type, 2> reduced_outputs(outputs.dimension(0), outputs.dimension(1));
    //     // type index;
    //     // type max;

    //     // for (Index i = 0; i < outputs.dimension(0); i++) {
    //     //     for (Index j = 0; j < outputs.dimension(1); j++) {
    //     //         index = 0;
    //     //         max = outputs(i,j,0);
    //     //         for(Index k = 1; k < outputs.dimension(2); k++)
    //     //             if(max < outputs(i,j,k)){
    //     //                 index = type(k);
    //     //                 max = outputs(i,j,k);
    //     //             }
    //     //         reduced_outputs(i,j) = index;
    //     //     }
    //     // }

    //     return calculate_confusion(reduced_outputs, testing_target);
    // }

    return Tensor<Index, 2>();
}


Tensor<Index, 2> TestingAnalysis::calculate_confusion(const Tensor<type, 2>& outputs,
                                                      const Tensor<type, 2>& targets,
                                                      const type& decision_threshold) const
{
    const Index outputs_number = neural_network->get_outputs_number();

    if (outputs_number == 1)
        return calculate_confusion_binary_classification(targets, outputs, decision_threshold);
    else
        return calculate_confusion_multiple_classification(targets, outputs);

    return Tensor<Index, 2>();
}


TestingAnalysis::RocAnalysis TestingAnalysis::perform_roc_analysis() const
{
    const Tensor<type, 2> inputs = dataset->get_data("Testing", "Input");
    const Tensor<type, 2> targets = dataset->get_data("Testing", "Target");

    const Tensor<type, 2> outputs = neural_network->calculate_outputs<2,2>(inputs);

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

        roc_curve(i,0) = type(1) - type(true_positive)/type(true_positive + false_negative);
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
    const Tensor<type, 2> inputs = dataset->get_data("Testing", "Input");

    const Tensor<type, 2> targets = dataset->get_data("Testing", "Target");

    const Tensor<type, 2> outputs = neural_network->calculate_outputs<2,2>(inputs);

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
    const Tensor<type, 2> inputs = dataset->get_data("Testing", "Input");

    const Tensor<type, 2> targets = dataset->get_data("Testing", "Target");

    const Tensor<type, 2> outputs = neural_network->calculate_outputs<2,2>(inputs);

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


TestingAnalysis::KolmogorovSmirnovResults TestingAnalysis::perform_Kolmogorov_Smirnov_analysis() const
{
    Tensor<type, 2> inputs = dataset->get_data("Testing", "Input");

    Tensor<type, 2> targets = dataset->get_data("Testing", "Target");

    Tensor<type, 2> outputs = neural_network->calculate_outputs<2,2>(inputs);

    TestingAnalysis::KolmogorovSmirnovResults Kolmogorov_Smirnov_results;

    Kolmogorov_Smirnov_results.positive_cumulative_gain = calculate_cumulative_gain(targets, outputs);
    Kolmogorov_Smirnov_results.negative_cumulative_gain = calculate_negative_cumulative_gain(targets, outputs);
    Kolmogorov_Smirnov_results.maximum_gain =
        calculate_maximum_gain(Kolmogorov_Smirnov_results.positive_cumulative_gain, Kolmogorov_Smirnov_results.negative_cumulative_gain);

    return Kolmogorov_Smirnov_results;
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


vector<Histogram> TestingAnalysis::calculate_output_histogram(const Tensor<type, 2>& outputs,
                                                              const Index& bins_number) const
{
    const Tensor<type, 1> output_column = outputs.chip(0,1);

    vector<Histogram> output_histogram(1);
    output_histogram[0] = histogram(output_column, bins_number);

    return output_histogram;
}


TestingAnalysis::BinaryClassificationRates TestingAnalysis::calculate_binary_classification_rates(const type& decision_threshold) const
{
    const Tensor<type, 2> inputs = dataset->get_data("Testing", "Input");

    const Tensor<type, 2> targets = dataset->get_data("Testing", "Target");

    const Tensor<type, 2> outputs = neural_network->calculate_outputs<2,2>(inputs);

    const vector<Index> testing_indices = dataset->get_sample_indices("Testing");

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

    const vector<Index> false_positives_indices(false_positives_indices_copy.begin(),
                                                false_positives_indices_copy.begin() + index);

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

    const vector<Index> false_negatives_indices(false_negatives_indices_copy.begin(),
                                                false_negatives_indices_copy.begin() + index);

    return false_negatives_indices;
}


vector<Index> TestingAnalysis::calculate_true_negative_samples(const Tensor<type, 2>& targets,
                                                               const Tensor<type, 2>& outputs,
                                                               const vector<Index>& testing_indices,
                                                               const type& decision_threshold) const
{
    const Index rows_number = targets.dimension(0);

    vector<Index> true_negatives_indices_copy(rows_number);

    Index index = 0;

    for(Index i = 0; i < rows_number; i++)
        if(targets(i,0) < decision_threshold && outputs(i,0) < decision_threshold)
            true_negatives_indices_copy[index++] = testing_indices[i];

    vector<Index> true_negatives_indices(true_negatives_indices_copy.begin(),
                                         true_negatives_indices_copy.begin() + index);

    return true_negatives_indices;
}


Tensor<type, 1> TestingAnalysis::calculate_multiple_classification_precision() const
{
    Tensor<type, 1> multiple_classification_tests(2);

    const Tensor<type, 2> inputs = dataset->get_data("Testing", "Input");

    const Tensor<type, 2> targets = dataset->get_data("Testing", "Target");

    const Tensor<type, 2> outputs = neural_network->calculate_outputs<2,2>(inputs);

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

    const vector<string> target_variable_names = dataset->get_variable_names("Target");

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
            j == raw_variables_number - 1
                ? file << confusion(i, j) << endl
                : file << confusion(i, j) << ",";
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
    const Tensor<type, 2> inputs = dataset->get_data("Testing", "Input");

    const Tensor<type, 2> targets = dataset->get_data("Testing", "Target");

    const Tensor<type, 2> outputs = neural_network->calculate_outputs<2,2>(inputs);

    const vector<Index> testing_indices = dataset->get_sample_indices("Testing");

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

    const vector<string> target_variables_names = dataset->get_variable_names("Target");

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

    return well_lassified_samples.slice(array_2(0, 0),
                                        array_2(number_of_well_classified, 4));
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

    //    array<Index, 2> offsets = {0, 0};
    //    array<Index, 2> extents = {number_of_misclassified, 4};

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

    classification_statistics_file << "minimum,maximum,mean,std" << endl
                                   << misclassified_numerical_probabilities.minimum() << ","
                                   << misclassified_numerical_probabilities.maximum() << ","
                                   << misclassified_numerical_probabilities.mean() << ","
                                   << standard_deviation(misclassified_numerical_probabilities);
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


Tensor<Tensor<type, 1>, 1> TestingAnalysis::calculate_error_autocorrelation(const Index& maximum_past_time_steps) const
{
    const Tensor<type, 2> inputs = dataset->get_data("Testing", "Input");

    const Tensor<type, 2> targets = dataset->get_data("Testing", "Target");

    const Tensor<type, 2> outputs = neural_network->calculate_outputs<2,2>(inputs);

    const Index targets_number = dataset->get_variables_number("Target");

    const Tensor<type, 2> error = outputs - targets;

    Tensor<Tensor<type, 1>, 1> error_autocorrelations(targets_number);

    for(Index i = 0; i < targets_number; i++)
        error_autocorrelations[i] = autocorrelations(thread_pool_device.get(), error.chip(i,1), maximum_past_time_steps);

    return error_autocorrelations;
}


Tensor<Tensor<type, 1>, 1> TestingAnalysis::calculate_inputs_errors_cross_correlation(const Index& past_time_steps) const
{
    const Index targets_number = dataset->get_variables_number("Target");

    const Tensor<type, 2> inputs = dataset->get_data("Testing", "Input");

    const Tensor<type, 2> targets = dataset->get_data("Testing", "Target");

    const Tensor<type, 2> outputs = neural_network->calculate_outputs<2,2>(inputs);

    const Tensor<type, 2> errors = outputs - targets;

    Tensor<Tensor<type, 1>, 1> inputs_errors_cross_correlation(targets_number);

    for(Index i = 0; i < targets_number; i++)
        inputs_errors_cross_correlation[i] = cross_correlations(thread_pool_device.get(),
                                                                inputs.chip(i,1), errors.chip(i,1), past_time_steps);

    return inputs_errors_cross_correlation;
}


pair<type, type> TestingAnalysis::test_transformer() const
{
    cout << "Testing transformer..." << endl;

    Transformer* transformer = static_cast<Transformer*>(neural_network);
    LanguageDataset* language_dataset = static_cast<LanguageDataset*>(dataset);

    const Tensor<type, 2> context = language_dataset->get_data("Testing", "Input");
    const Tensor<type, 2> input = language_dataset->get_data("Testing", "Decoder");
    const Tensor<type, 2> target = language_dataset->get_data("Testing", "Target");

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
    // for(Index i = 0; i < testing_context.dimension(1); i++)
    //     cout<<language_dataset->get_context_vocabulary()[Index(testing_context(10,i))]<<" ";
    // cout<<endl;
    // cout<<endl;
    // cout<<"Spanish:"<<endl;
    // cout<<testing_input.chip(10,0)<<endl;
    // for(Index i = 0; i < testing_input.dimension(1); i++)
    //     cout<<language_dataset->get_completion_vocabulary()[Index(testing_input(10,i))]<<" ";
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
    //     cout<<language_dataset->get_completion_vocabulary()[index]<<" ";
    // }

    const type error = calculate_cross_entropy_error_3d(outputs, testing_target);

    const type accuracy = calculate_masked_accuracy(outputs, testing_target);

    return pair<type, type>(error, accuracy);
}


string TestingAnalysis::test_transformer(const vector<string>& context_string, const bool& imported_vocabulary) const
{
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
                                : precision + negative_predictive_value - type(1);

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

    const Index targets_number = dataset->get_variables_number("Target");

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


void TestingAnalysis::GoodnessOfFitAnalysis::set(const Tensor<type, 1>& new_targets,
                                                 const Tensor<type, 1>& new_outputs,
                                                 const type& new_determination)
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

    // cout << "Targets:" << endl;
    // cout << targets << endl;
    // cout << "Outputs:" << endl;
    // cout << outputs << endl;
}


void TestingAnalysis::RocAnalysis::print() const
{
    cout << "Roc Curve analysis" << endl;

    cout << "Roc Curve:\n" << roc_curve << endl;
    cout << "Area Under Curve: " << area_under_curve << endl;
    cout << "Confidence Limit: " << confidence_limit << endl;
    cout << "Optimal Threshold: " << optimal_threshold << endl;
}


#ifdef OPENNN_CUDA

void TestingAnalysis::set_batch_size(const Index& new_batch_size)
{
    batch_size = new_batch_size;
}

Index TestingAnalysis::get_batch_size()
{
    return batch_size;
}


Tensor<Index, 2> TestingAnalysis::calculate_confusion_cuda(const type& decision_threshold) const
{
    check();

    const vector<Index> testing_indices = dataset->get_sample_indices("Testing");
    const vector<vector<Index>> testing_batches = dataset->get_batches(testing_indices, batch_size, false);

    const vector<Index> input_variable_indices = dataset->get_variable_indices("Input");
    const vector<Index> target_variable_indices = dataset->get_variable_indices("Target");
    //const vector<Index> decoder_variable_indices = dataset->get_variable_indices("Decoder");

    const Index outputs_number = neural_network->get_outputs_number();

    const Index confusion_matrix_size = (outputs_number == 1) ? 3 : (outputs_number + 1);

    Tensor<Index, 2> total_confusion_matrix(confusion_matrix_size, confusion_matrix_size);

    total_confusion_matrix.setZero();

    BatchCuda testing_batch_cuda(batch_size, dataset);
    ForwardPropagationCuda testing_forward_propagation_cuda(batch_size, neural_network);

    neural_network->allocate_parameters_device();
    neural_network->copy_parameters_device();

    for (const auto& current_batch_indices : testing_batches)
    {
        const Index current_batch_size = current_batch_indices.size();
        if (current_batch_size == 0) continue;

        if (current_batch_size != batch_size) {
            testing_batch_cuda.free();
            testing_forward_propagation_cuda.free();
            testing_batch_cuda.set(current_batch_size, dataset);
            testing_forward_propagation_cuda.set(current_batch_size, neural_network);
        }

        testing_batch_cuda.fill(current_batch_indices,
                                input_variable_indices,
                                //decoder_variable_indices,
                                target_variable_indices);

        neural_network->forward_propagate_cuda(testing_batch_cuda.get_input_device(),
                                               testing_forward_propagation_cuda,
                                               false);

        const float* outputs_device = testing_forward_propagation_cuda.get_last_trainable_layer_outputs_device();

        Tensor<type, 2> batch_outputs(current_batch_size, outputs_number);
        cudaMemcpy(batch_outputs.data(), outputs_device, current_batch_size * outputs_number * sizeof(type), cudaMemcpyDeviceToHost);

        const Tensor<type, 2> batch_targets = dataset->get_data_from_indices(current_batch_indices, target_variable_indices);
        Tensor<Index, 2> batch_confusion = calculate_confusion(batch_outputs, batch_targets, decision_threshold);
        total_confusion_matrix += batch_confusion;
    }

    neural_network->free_parameters_device();

    total_confusion_matrix(confusion_matrix_size - 1, confusion_matrix_size - 1) = testing_indices.size();

    return total_confusion_matrix;
}

#endif

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
