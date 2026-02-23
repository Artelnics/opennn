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
#include "time_series_dataset.h"
#include "standard_networks.h"
#include "statistics.h"
#include "unscaling_layer.h"

namespace opennn
{

TestingAnalysis::TestingAnalysis(const NeuralNetwork* new_neural_network, const Dataset* new_dataset)
{
    neural_network = const_cast<NeuralNetwork*>(new_neural_network);
    dataset = const_cast<Dataset*>(new_dataset);
}


NeuralNetwork* TestingAnalysis::get_neural_network() const
{
    return neural_network;
}


Dataset* TestingAnalysis::get_dataset() const
{
    return dataset;
}


bool TestingAnalysis::get_display() const
{
    return display;
}


Index TestingAnalysis::get_batch_size()
{
    return batch_size;
}


void TestingAnalysis::set_neural_network(NeuralNetwork* new_neural_network)
{
    neural_network = new_neural_network;
}


void TestingAnalysis::set_dataset(Dataset* new_dataset)
{
    dataset = new_dataset;
}


void TestingAnalysis::set_display(bool new_display)
{
    display = new_display;
}


void TestingAnalysis::set_batch_size(const Index new_batch_size)
{
    batch_size = new_batch_size;
}


void TestingAnalysis::check() const
{
    if(!neural_network)
        throw runtime_error("Neural network pointer is nullptr.\n");

    if(!dataset)
        throw runtime_error("Dataset pointer is nullptr.\n");
}


Tensor<Correlation, 1> TestingAnalysis::linear_correlation() const
{
    const auto [targets, outputs] = get_targets_and_outputs("Testing");

    return linear_correlation(targets, outputs);
}


Tensor<Correlation, 1> TestingAnalysis::linear_correlation(const MatrixR& target, const MatrixR& output) const
{
    const Index outputs_number = dataset->get_features_number("Target");

    Tensor<Correlation, 1> linear_correlation(outputs_number);

    for(Index i = 0; i < outputs_number; i++)
        linear_correlation(i) = opennn::linear_correlation(output.col(i), target.col(i));

    return linear_correlation;
}


void TestingAnalysis::print_linear_correlations() const
{
    const Tensor<Correlation, 1> linear_correlations = linear_correlation();

    const vector<string> targets_name = dataset->get_feature_names("Target");

    const Index targets_number = linear_correlations.size();

    for(Index i = 0; i < targets_number; i++)
        cout << targets_name[i] << " correlation: " << linear_correlations[i].r << endl;
}


Tensor<TestingAnalysis::GoodnessOfFitAnalysis, 1> TestingAnalysis::perform_goodness_of_fit_analysis() const
{
    check();

    // Dataset

    const Index testing_samples_number = dataset->get_samples_number("Testing");

    if(testing_samples_number == Index(0))
        throw runtime_error("Number of testing samples is zero.\n");

    // Neural network

    const Index outputs_number = neural_network->get_outputs_number();

    const pair<MatrixR, MatrixR> targets_outputs = get_targets_and_outputs("Testing");

    // Testing analysis

    Tensor<GoodnessOfFitAnalysis, 1> goodness_of_fit_results(outputs_number);

    for(Index i = 0;  i < outputs_number; i++)
    {
        const VectorMap targets = vector_map(targets_outputs.first, i);
        const VectorMap outputs = vector_map(targets_outputs.second, i);

        const type determination = calculate_determination(outputs, targets);

        goodness_of_fit_results[i].set(targets, outputs, determination);
    }

    return goodness_of_fit_results;
}


void TestingAnalysis::print_goodness_of_fit_analysis() const
{
    const Tensor<GoodnessOfFitAnalysis, 1> goodness_of_fit_analysis = perform_goodness_of_fit_analysis();

    for(Index i = 0; i < goodness_of_fit_analysis.size(); i++)
        goodness_of_fit_analysis(i).print();
}


pair<MatrixR, MatrixR> TestingAnalysis::get_targets_and_outputs(const string& sample_role) const
{
    check();

    // Dataset

    const Index samples_number = dataset->get_samples_number(sample_role);

    if(samples_number == Index(0))
        throw runtime_error("Number of samples is zero.\n");

    MatrixR output_data;
    MatrixR target_data;

    if (TimeSeriesDataset* time_series_dataset = dynamic_cast<TimeSeriesDataset*>(dataset))
    {
        const Tensor3 input_data = time_series_dataset->get_data(sample_role, "Input");
        output_data = neural_network->calculate_outputs(input_data);

        const vector<Index> sample_indices = time_series_dataset->get_sample_indices(sample_role);
        const vector<Index> feature_indices = time_series_dataset->get_feature_indices("Target");
        target_data.resize(static_cast<Index>(sample_indices.size()), static_cast<Index>(feature_indices.size()));
        time_series_dataset->fill_target_tensor(sample_indices, feature_indices, target_data.data());
    }
    else
    {
        target_data = dataset->get_data(sample_role, "Target");
        const MatrixR input_data = dataset->get_data(sample_role, "Input");
        output_data = neural_network->calculate_outputs(input_data);
    }

    return {target_data, output_data};
}


MatrixR TestingAnalysis::calculate_error() const
{
    const auto [targets, outputs] = get_targets_and_outputs("Testing");

    return targets - outputs;
}


Tensor3 TestingAnalysis::calculate_error_data() const
{
    check();

    const Index testing_samples_number = dataset->get_samples_number("Testing");

    if(testing_samples_number == Index(0))
        throw runtime_error("Number of testing samples is zero.\n");

    const Index outputs_number = neural_network->get_outputs_number();

    const auto [targets, outputs] = get_targets_and_outputs("Testing");

    Unscaling* unscaling_layer = static_cast<Unscaling*>(neural_network->get_first("Unscaling"));

    if(!unscaling_layer)
        throw runtime_error("Unscaling layer not found.\n");

    const VectorR& output_minimums = unscaling_layer->get_minimums();
    const VectorR& output_maximums = unscaling_layer->get_maximums();

    Tensor3 error_data(testing_samples_number, 3, outputs_number);

    const MatrixR absolute_errors = (targets - outputs).array().abs();

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


MatrixR TestingAnalysis::calculate_percentage_error_data() const
{
    check();

    // Dataset

    const Index testing_samples_number = dataset->get_samples_number("Testing");

    if(testing_samples_number == Index(0))
        throw runtime_error("Number of testing samples is zero.\n");

    // Neural network

    const Index outputs_number = neural_network->get_outputs_number();

    const auto [targets, outputs] = get_targets_and_outputs("Testing");

    Unscaling* unscaling_layer = static_cast<Unscaling*>(neural_network->get_first("Unscaling"));

    if(!unscaling_layer)
        throw runtime_error("Unscaling layer not found.\n");

    const VectorR& output_minimums = unscaling_layer->get_minimums();
    const VectorR& output_maximums = unscaling_layer->get_maximums();

    const MatrixR errors = (targets - outputs);

    // Error data

    MatrixR error_data(testing_samples_number, outputs_number);

#pragma omp parallel for

    for(Index i = 0; i < testing_samples_number; i++)
        for(Index j = 0; j < outputs_number; j++)
            error_data(i, j) = errors(i, j)*type(100.0)/abs(output_maximums(j) - output_minimums(j));

    return error_data;
}


vector<Descriptives> TestingAnalysis::calculate_absolute_errors_descriptives() const
{    
    const auto [targets, outputs] = get_targets_and_outputs("Testing");

    return calculate_absolute_errors_descriptives(targets, outputs);
}


vector<Descriptives> TestingAnalysis::calculate_absolute_errors_descriptives(const MatrixR& targets,
                                                                             const MatrixR& outputs) const
{
    const MatrixR difference = (targets-outputs).array().abs();

    return descriptives(difference);
}


vector<Descriptives> TestingAnalysis::calculate_percentage_errors_descriptives() const
{
    const auto [targets, outputs] = get_targets_and_outputs("Testing");

    return calculate_percentage_errors_descriptives(targets, outputs);
}


vector<Descriptives> TestingAnalysis::calculate_percentage_errors_descriptives(const MatrixR& targets,
                                                                               const MatrixR& outputs) const
{
    const MatrixR difference = type(100)*(targets-outputs).array().abs()/targets.array();

    return descriptives(difference);
}


vector<vector<Descriptives>> TestingAnalysis::calculate_error_data_descriptives() const
{
    // Neural network

    const Index outputs_number = neural_network->get_outputs_number();

    const Index testing_samples_number = dataset->get_samples_number("Testing");

    // Testing analysis stuff

    vector<vector<Descriptives>> descriptives(outputs_number);

    Tensor3 error_data = calculate_error_data();

    Index index = 0;

    for(Index i = 0; i < outputs_number; i++)
    {
        const MatrixMap matrix_error(error_data.data() + index, testing_samples_number, 3);

        const MatrixR matrix(matrix_error);

        descriptives[i] = opennn::descriptives(matrix);

        index += testing_samples_number*3;
    }

    return descriptives;
}


void TestingAnalysis::print_error_data_descriptives() const
{
    const Index targets_number = dataset->get_features_number("Target");

    const vector<string> targets_name = dataset->get_feature_names("Target");

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


vector<Histogram> TestingAnalysis::calculate_error_data_histograms(const Index bins_number) const
{
    const MatrixR error_data = calculate_percentage_error_data();

    const Index outputs_number = error_data.cols();

    vector<Histogram> histograms(outputs_number);

    for(Index i = 0; i < outputs_number; i++)
        histograms[i] = histogram_centered(error_data.col(i), type(0), bins_number);

    return histograms;
}


Tensor<VectorI, 1> TestingAnalysis::calculate_maximal_errors(const Index samples_number) const
{
    Tensor3 error_data = calculate_error_data();

    const Index outputs_number = error_data.dimension(2);
    const Index testing_samples_number = error_data.dimension(0);

    Tensor<VectorI, 1> maximal_errors(samples_number);

    Index index = 0;

    for(Index i = 0; i < outputs_number; i++)
    {
        const MatrixMap matrix_error(error_data.data()+index, testing_samples_number, 3);

        maximal_errors[i] = maximal_indices(matrix_error.col(0), samples_number);

        index += testing_samples_number*3;
    }

    return maximal_errors;
}


MatrixR TestingAnalysis::calculate_errors() const
{
    MatrixR errors(5, 3);

    errors.col(0) = calculate_errors("Training");
    errors.col(1) = calculate_errors("Validation");
    errors.col(2) = calculate_errors("Testing");

    return errors;
}


MatrixR TestingAnalysis::calculate_binary_classification_errors() const
{
    const VectorR training_errors = calculate_binary_classification_errors("Training");
    const VectorR validation_errors = calculate_binary_classification_errors("Validation");
    const VectorR testing_errors = calculate_binary_classification_errors("Testing");

    MatrixR errors(7, 3);

    errors <<
        training_errors(0), validation_errors(0), testing_errors(0),
        training_errors(1), validation_errors(1), testing_errors(1),
        training_errors(2), validation_errors(2), testing_errors(2),
        training_errors(3), validation_errors(3), testing_errors(3),
        training_errors(4), validation_errors(4), testing_errors(4),
        training_errors(5), validation_errors(5), testing_errors(5),
        training_errors(6), validation_errors(6), testing_errors(6);

    return errors;
}


MatrixR TestingAnalysis::calculate_multiple_classification_errors() const
{
    const VectorR training_errors = calculate_multiple_classification_errors("Training");
    const VectorR validation_errors = calculate_multiple_classification_errors("Validation");
    const VectorR testing_errors = calculate_multiple_classification_errors("Testing");

    MatrixR errors(6, 3);

    errors <<
        training_errors(0), validation_errors(0), testing_errors(0),
        training_errors(1), validation_errors(1), testing_errors(1),
        training_errors(2), validation_errors(2), testing_errors(2),
        training_errors(3), validation_errors(3), testing_errors(3),
        training_errors(4), validation_errors(4), testing_errors(4),
        training_errors(5), validation_errors(5), testing_errors(5);

    return errors;
}


VectorR TestingAnalysis::calculate_errors(const MatrixR& targets,
                                          const MatrixR& outputs) const
{
    const type predictions_number = static_cast<type>(targets.size());

    const type mean_squared_error = (outputs - targets).squaredNorm();

    VectorR errors(5);
    errors(0) = mean_squared_error;
    errors(1) = errors(0)/type(predictions_number);
    errors(2) = sqrt(errors(1));
    errors(3) = calculate_normalized_squared_error(targets, outputs);
    errors(4) = calculate_Minkowski_error(targets, outputs);

    return errors;
}


VectorR TestingAnalysis::calculate_errors(const string& sample_role) const
{
    const auto [targets, outputs] = get_targets_and_outputs(sample_role);

    return calculate_errors(targets, outputs);
}


VectorR TestingAnalysis::calculate_binary_classification_errors(const string& sample_role) const
{
    // Dataset

    const Index training_samples_number = dataset->get_samples_number(sample_role);

    const auto [targets, outputs] = get_targets_and_outputs("Testing");

    const type mean_squared_error = (outputs - targets).squaredNorm();

    VectorR errors(6);

    errors(0) = mean_squared_error;
    errors(1) = errors(0)/type(training_samples_number);
    errors(2) = sqrt(errors(1));
    errors(3) = calculate_normalized_squared_error(targets, outputs);
    errors(4) = calculate_cross_entropy_error(targets, outputs);
    errors(5) = calculate_weighted_squared_error(targets, outputs);

    return errors;
}


VectorR TestingAnalysis::calculate_multiple_classification_errors(const string& sample_role) const
{
    const Index training_samples_number = dataset->get_samples_number(sample_role);

    const auto [targets, outputs] = get_targets_and_outputs("Testing");

    const type mean_squared_error = (outputs - targets).squaredNorm();

    VectorR errors(5);
    errors(0) = mean_squared_error;
    errors(1) = errors(0)/type(training_samples_number);
    errors(2) = sqrt(errors(1));
    errors(3) = calculate_normalized_squared_error(targets, outputs);
    errors(4) = calculate_cross_entropy_error(targets, outputs); // NO

    return errors;
}


type TestingAnalysis::calculate_normalized_squared_error(const MatrixR& targets, const MatrixR& outputs) const
{
    const VectorR targets_mean = mean(targets);

    const type mean_squared_error = (outputs - targets).squaredNorm();

    const type normalization_coefficient = (targets.rowwise() - targets_mean.transpose()).squaredNorm();

    return mean_squared_error/normalization_coefficient;
}


type TestingAnalysis::calculate_cross_entropy_error(const MatrixR& targets,
                                                    const MatrixR& outputs) const
{
    const Index testing_samples_number = targets.rows();
    const Index outputs_number = targets.cols();

    VectorR targets_row(outputs_number);
    VectorR outputs_row(outputs_number);

    type cross_entropy_error_2d = type(0);

#pragma omp parallel for reduction(+:cross_entropy_error_2d)

    for(Index i = 0; i < testing_samples_number; i++)
    {
        outputs_row = outputs.row(i);
        targets_row = targets.row(i);

        for(Index j = 0; j < outputs_number; j++)
        {
            outputs_row(j) = clamp(outputs_row(j), type(1.0e-6), numeric_limits<type>::max());

            cross_entropy_error_2d -=
                targets_row(j)*log(outputs_row(j)) + (type(1) - targets_row(j))*log(type(1) - outputs_row(j));
        }
    }

    return cross_entropy_error_2d/type(testing_samples_number);
}


type TestingAnalysis::calculate_cross_entropy_error_3d(const Tensor3& outputs, const MatrixR& targets) const
{
/*
    const Index batch_size = outputs.rows();
    const Index outputs_number = outputs.cols();

    MatrixR errors(batch_size, outputs_number);
    MatrixR predictions(batch_size, outputs_number);
    MatrixB matches(batch_size, outputs_number);
    MatrixB mask(batch_size, outputs_number);

    Tensor0 cross_entropy_error_2d;
    mask = targets != targets.constant(0);

    Tensor0 mask_sum;
    mask_sum = mask.cast<type>().sum();

    for(Index i = 0; i < batch_size; i++)
        for(Index j = 0; j < outputs_number; j++)
            errors(i, j) = -log(outputs(i, j, Index(targets(i, j))));

    errors = errors * mask.cast<type>();

    cross_entropy_error_2d = errors.sum();

    return cross_entropy_error_2d(0) / mask_sum(0);
*/
    return 0;
}


type TestingAnalysis::calculate_weighted_squared_error(const MatrixR& targets,
                                                       const MatrixR& outputs,
                                                       const VectorR& weights) const
{
/*
    type negatives_weight;
    type positives_weight;

    if(weights.size() != 2)
    {
        const VectorI target_distribution = dataset->calculate_target_distribution();

        const Index negatives_number = target_distribution[0];
        const Index positives_number = target_distribution[1];

        negatives_weight = type(1);

        positives_weight = (negatives_number == 0 || positives_number == 0)
           ? type(0)
           : type(negatives_number / positives_number);
    }
    else
    {
        positives_weight = weights[0];
        negatives_weight = weights[1];
    }

    const MatrixB if_sentence = (targets == targets.constant(type(1))).cast<bool>();
    const MatrixB else_sentence = (targets == targets.constant(type(0))).cast<bool>();

    MatrixR f_1(targets.rows(), targets.cols());
    MatrixR f_2(targets.rows(), targets.cols());
    MatrixR f_3(targets.rows(), targets.cols());

    f_1.device(get_device()) = (targets - outputs).square() * positives_weight;

    f_2.device(get_device()) = (targets - outputs).square()*negatives_weight;

    f_3.device(get_device()) = targets.constant(type(0));

    Tensor0 mean_squared_error;
    mean_squared_error.device(get_device()) = (if_sentence.select(f_1, else_sentence.select(f_2, f_3))).sum();

    Index negatives = 0;

    VectorR target_column = targets.col(0);

    for(Index i = 0; i < target_column.size(); i++)
        if(double(target_column(i)) == 0.0)
            negatives++;

    const type normalization_coefficient = type(negatives)*negatives_weight*type(0.5);

    return mean_squared_error(0)/normalization_coefficient;
*/
    return 0;
}


type TestingAnalysis::calculate_Minkowski_error(const MatrixR& targets,
                                                const MatrixR& outputs,
                                                const type minkowski_parameter) const
{
/*
    const type predictions_number = static_cast<type>(targets.size());

    if (predictions_number == 0)
        return type(0);

    Tensor0 minkowski_error;

    minkowski_error.device(get_device()) =
        (((outputs - targets).array().abs().pow(minkowski_parameter).sum()) / predictions_number).pow(type(1.0) / minkowski_parameter);

    return minkowski_error();
*/
    return 0;
}


type TestingAnalysis::calculate_masked_accuracy(const Tensor3& outputs, const MatrixR& targets) const
{
/*
    const Index batch_size = outputs.rows();
    const Index outputs_number = outputs.cols();

    MatrixR predictions(batch_size, outputs_number);
    MatrixB matches(batch_size, outputs_number);
    MatrixB mask(batch_size, outputs_number);

    Tensor0 accuracy;

    mask = targets != targets.constant(0);

    const Tensor0 mask_sum = mask.cast<type>().sum();

    predictions = outputs.argmax(2).cast<type>();

    matches = predictions == targets;

    matches = matches && mask;

    accuracy = matches.cast<type>().sum() / mask_sum(0);

    return accuracy(0);
*/
    return 0;
}


type TestingAnalysis::calculate_determination(const VectorR& outputs, const VectorR& targets) const
{
    const type targets_mean = targets.mean();
    const type outputs_mean = outputs.mean();

    const type numerator = ((targets.array() - targets_mean) * (outputs.array() - outputs_mean)).sum();

    const type targets_ss = (targets.array() - targets_mean).square().sum();
    const type outputs_ss = (outputs.array() - outputs_mean).square().sum();

    const type denominator = sqrt(targets_ss * outputs_ss);

    if(denominator < NUMERIC_LIMITS_MIN)
        return type(0);

    const type r = numerator / denominator;

    return r * r;
}


MatrixI TestingAnalysis::calculate_confusion_binary_classification(const MatrixR& targets,
                                                                            const MatrixR& outputs,
                                                                            type decision_threshold) const
{
    const Index testing_samples_number = targets.rows();

    MatrixI confusion(3, 3);

    auto t_pos = targets.col(0).array() >= decision_threshold;
    auto o_pos = outputs.col(0).array() >= decision_threshold;

    const Index true_positive = (t_pos && o_pos).count();
    const Index false_negative = (t_pos && !o_pos).count();
    const Index false_positive = (!t_pos && o_pos).count();
    const Index true_negative = (!t_pos && !o_pos).count();

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


MatrixI TestingAnalysis::calculate_confusion_multiple_classification(const MatrixR& targets,
                                                                              const MatrixR& outputs) const
{
    const Index samples_number = targets.rows();
    const Index targets_number = targets.cols();

    if(targets_number != outputs.cols())
        throw runtime_error("Number of targets (" + to_string(targets_number) + ") "
                            "must be equal to number of outputs (" + to_string(outputs.cols()) + ").\n");

    MatrixI confusion(targets_number + 1, targets_number + 1);
    confusion.setZero();
    confusion(targets_number, targets_number) = samples_number;

    Index target_index = 0;
    Index output_index = 0;

    for(Index i = 0; i < samples_number; i++)
    {
        target_index = maximal_index(targets.row(i));
        output_index = maximal_index(outputs.row(i));

        confusion(target_index, output_index)++;
        confusion(target_index, targets_number)++;
        confusion(targets_number, output_index)++;
    }

    return confusion;
}


vector<MatrixI> TestingAnalysis::calculate_multilabel_confusion(const type decision_threshold) const
{
    check();

    const auto [targets, outputs] = get_targets_and_outputs("Testing");
    const Index outputs_number = neural_network->get_outputs_number();

    vector<MatrixI> confusion_matrices(static_cast<size_t>(outputs_number));

    for(Index j = 0; j < outputs_number; j++)
    {
        const MatrixR target_col = targets.col(j);
        const MatrixR output_col = outputs.col(j);

        confusion_matrices[static_cast<size_t>(j)] = calculate_confusion_binary_classification(target_col, output_col, decision_threshold);
    }

    return confusion_matrices;
}


VectorI TestingAnalysis::calculate_positives_negatives_rate(const MatrixR& targets, const MatrixR& outputs) const
{
    const MatrixI confusion = calculate_confusion_binary_classification(targets, outputs, type(0.5));

    VectorI positives_negatives_rate(2);

    positives_negatives_rate << (confusion(0,0) + confusion(0,1)),
                                (confusion(1,0) + confusion(1,1));

    return positives_negatives_rate;
}


MatrixI TestingAnalysis::calculate_confusion(const type decision_threshold) const
{
    check();

    const vector<Index> testing_indices = dataset->get_sample_indices("Testing");
    const vector<vector<Index>> testing_batches = dataset->get_batches(testing_indices, batch_size, false);

    const vector<Index> input_feature_indices = dataset->get_feature_indices("Input");
    const vector<Index> target_feature_indices = dataset->get_feature_indices("Target");

    const Shape input_shape = dataset->get_shape("Input");

    const Index outputs_number = neural_network->get_outputs_number();
    const Index confusion_matrix_size = (outputs_number == 1) ? 3 : (outputs_number + 1);

    MatrixI total_confusion_matrix(confusion_matrix_size, confusion_matrix_size);
    total_confusion_matrix.setZero();

    for(const vector<Index>& current_batch_indices : testing_batches)
    {
        const Index current_batch_size = current_batch_indices.size();
        if (current_batch_size == 0) continue;

        MatrixR batch_inputs_flat = dataset->get_data_from_indices(current_batch_indices, input_feature_indices);
        const MatrixR batch_targets = dataset->get_data_from_indices(current_batch_indices, target_feature_indices);

        MatrixR batch_outputs;

        if(input_shape.size() == 1)
            batch_outputs = neural_network->calculate_outputs(batch_inputs_flat);
        else if(input_shape.size() == 3)
        {
            Tensor4 inputs_4d(current_batch_size,
                              input_shape[0],
                              input_shape[1],
                              input_shape[2]);

            memcpy(inputs_4d.data(), batch_inputs_flat.data(), 
                   current_batch_size * batch_inputs_flat.cols() * sizeof(type));

            batch_outputs = neural_network->calculate_outputs(inputs_4d);
        }
        else
            return MatrixI();

        const MatrixI batch_confusion = calculate_confusion(batch_outputs, batch_targets, decision_threshold);
        total_confusion_matrix += batch_confusion;
    }

    total_confusion_matrix(confusion_matrix_size - 1, confusion_matrix_size - 1) = testing_indices.size();

    return total_confusion_matrix;
}


MatrixI TestingAnalysis::calculate_confusion(const MatrixR& outputs,
                                             const MatrixR& targets,
                                             type decision_threshold) const
{
    const Index outputs_number = neural_network->get_outputs_number();

    if(outputs_number == 1)
        return calculate_confusion_binary_classification(targets, outputs, decision_threshold);
    else
        return calculate_confusion_multiple_classification(targets, outputs);

    return MatrixI();
}


TestingAnalysis::RocAnalysis TestingAnalysis::perform_roc_analysis() const
{
    const auto [targets, outputs] = get_targets_and_outputs("Testing");

    RocAnalysis roc_analysis;
    roc_analysis.roc_curve = calculate_roc_curve(targets, outputs);
    roc_analysis.area_under_curve = calculate_area_under_curve(roc_analysis.roc_curve);
    roc_analysis.confidence_limit = calculate_area_under_curve_confidence_limit(targets, outputs);
    roc_analysis.optimal_threshold = calculate_optimal_threshold(roc_analysis.roc_curve);

    return roc_analysis;
}


MatrixR TestingAnalysis::calculate_roc_curve(const MatrixR& targets, const MatrixR& outputs) const
{
    const VectorI positives_negatives_rate = calculate_positives_negatives_rate(targets, outputs);

    const Index total_positives = positives_negatives_rate(0);
    const Index total_negatives = positives_negatives_rate(1);

    if(total_positives == 0)
        throw runtime_error("Number of positive samples (" + to_string(total_positives) + ") must be greater than zero.\n");

    if(total_negatives == 0)
        throw runtime_error("Number of negative samples (" + to_string(total_negatives) + ") must be greater than zero.\n");

    const Index maximum_points_number = 100;

    Index points_number = maximum_points_number;

    if(targets.cols() != 1)
        throw runtime_error("Number of of target variables (" +  to_string(targets.cols()) + ") must be one.\n");

    if(outputs.cols() != 1)
        throw runtime_error("Number of of output variables (" + to_string(targets.cols()) + ") must be one.\n");

    // Sort by ascending values of outputs vector

    VectorI sorted_indices(outputs.rows());

    Index* sorted_indices_data = sorted_indices.data();

    iota(sorted_indices_data, sorted_indices_data + sorted_indices.size(), 0);

    stable_sort(sorted_indices.data(),
                sorted_indices_data + sorted_indices.size(),
                [outputs](Index i1, Index i2) {return outputs(i1,0) < outputs(i2,0);});

    MatrixR roc_curve(points_number + 1, 3);
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


type TestingAnalysis::calculate_area_under_curve(const MatrixR& roc_curve) const
{
    type area_under_curve = type(0);

    for(Index i = 1; i < roc_curve.rows(); i++)
        area_under_curve += (roc_curve(i,0) - roc_curve(i-1,0))*(roc_curve(i,1) + roc_curve(i-1,1));

    return area_under_curve/ type(2);
}


type TestingAnalysis::calculate_area_under_curve_confidence_limit(const MatrixR& targets, const MatrixR& outputs) const
{
    const VectorI positives_negatives_rate = calculate_positives_negatives_rate(targets, outputs);

    const Index total_positives = positives_negatives_rate[0];
    const Index total_negatives = positives_negatives_rate[1];

    if(total_positives == 0)
        throw runtime_error("Number of positive samples(" + to_string(total_positives) + ") must be greater than zero.\n");

    if(total_negatives == 0)
        throw runtime_error("Number of negative samples(" + to_string(total_negatives) + ") must be greater than zero.\n");

    const MatrixR roc_curve = calculate_roc_curve(targets, outputs);

    const type area_under_curve = calculate_area_under_curve(roc_curve);

    const type Q_1 = area_under_curve/(type(2.0) - area_under_curve);
    const type Q_2 = (type(2.0) *area_under_curve*area_under_curve)/(type(1) *area_under_curve);

    const type confidence_limit = type(type(1.64485)*sqrt((area_under_curve*(type(1) - area_under_curve)
                                                             + (type(total_positives) - type(1))*(Q_1-area_under_curve*area_under_curve)
                                                             + (type(total_negatives) - type(1))*(Q_2-area_under_curve*area_under_curve))/(type(total_positives*total_negatives))));

    return confidence_limit;
}


type TestingAnalysis::calculate_optimal_threshold(const MatrixR& roc_curve) const
{
    const Index points_number = roc_curve.rows();

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


MatrixR TestingAnalysis::perform_cumulative_gain_analysis() const
{
    const auto [targets, outputs] = get_targets_and_outputs("Testing");

    return calculate_cumulative_gain(targets, outputs);
}


MatrixR TestingAnalysis::calculate_cumulative_gain(const MatrixR& targets, const MatrixR& outputs) const
{
    const Index total_positives = calculate_positives_negatives_rate(targets, outputs)[0];

    if(total_positives == 0)
        throw runtime_error("Number of positive samples(" + to_string(total_positives) + ") must be greater than zero.\n");

    const Index testing_samples_number = targets.rows();

    // Sort by ascending values of outputs vector

    VectorI sorted_indices(outputs.rows());
    iota(sorted_indices.data(), sorted_indices.data() + sorted_indices.size(), 0);

    stable_sort(sorted_indices.data(),
                sorted_indices.data()+sorted_indices.size(),
                [outputs](Index i1, Index i2) {return outputs(i1,0) > outputs(i2,0);});

    VectorR sorted_targets(testing_samples_number);

    for(Index i = 0; i < testing_samples_number; i++)
        sorted_targets(i) = targets(sorted_indices(i),0);

    const Index points_number = 21;
    const type percentage_increment = type(0.05);

    MatrixR cumulative_gain(points_number, 2);

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


MatrixR TestingAnalysis::calculate_negative_cumulative_gain(const MatrixR& targets, const MatrixR& outputs) const
{
    const Index total_negatives = calculate_positives_negatives_rate(targets, outputs)[1];

    if(total_negatives == 0)
        throw runtime_error("Number of negative samples(" + to_string(total_negatives) + ") must be greater than zero.\n");

    const Index testing_samples_number = targets.rows();

    // Sort by ascending values of outputs vector

    VectorI sorted_indices(outputs.rows());
    iota(sorted_indices.data(), sorted_indices.data() + sorted_indices.size(), 0);

    stable_sort(sorted_indices.data(), sorted_indices.data()+sorted_indices.size(), [outputs](Index i1, Index i2) {return outputs(i1,0) > outputs(i2,0);});

    VectorR sorted_targets(testing_samples_number);

    for(Index i = 0; i < testing_samples_number; i++)
        sorted_targets(i) = targets(sorted_indices(i),0);

    const Index points_number = 21;
    const type percentage_increment = type(0.05);

    MatrixR negative_cumulative_gain(points_number, 2);

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


MatrixR TestingAnalysis::perform_lift_chart_analysis() const
{
    const auto [targets, outputs] = get_targets_and_outputs("Testing");

    const MatrixR cumulative_gain = calculate_cumulative_gain(targets, outputs);

    return calculate_lift_chart(cumulative_gain);
}


MatrixR TestingAnalysis::calculate_lift_chart(const MatrixR& cumulative_gain) const
{
    const Index rows_number = cumulative_gain.rows();
    const Index variables_number = cumulative_gain.cols();

    MatrixR lift_chart(rows_number, variables_number);

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
    const auto [targets, outputs] = get_targets_and_outputs("Testing");

    TestingAnalysis::KolmogorovSmirnovResults Kolmogorov_Smirnov_results;

    Kolmogorov_Smirnov_results.positive_cumulative_gain = calculate_cumulative_gain(targets, outputs);
    Kolmogorov_Smirnov_results.negative_cumulative_gain = calculate_negative_cumulative_gain(targets, outputs);
    Kolmogorov_Smirnov_results.maximum_gain =
        calculate_maximum_gain(Kolmogorov_Smirnov_results.positive_cumulative_gain, Kolmogorov_Smirnov_results.negative_cumulative_gain);

    return Kolmogorov_Smirnov_results;
}


VectorR TestingAnalysis::calculate_maximum_gain(const MatrixR& positive_cumulative_gain,
                                                const MatrixR& negative_cumulative_gain) const
{
    const Index points_number = positive_cumulative_gain.rows();

    VectorR maximum_gain(2);

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


vector<Histogram> TestingAnalysis::calculate_output_histogram(const MatrixR& outputs,
                                                              Index bins_number) const
{
    const VectorR output_column = outputs.col(0);

    vector<Histogram> output_histogram(1);
    output_histogram[0] = histogram(output_column, bins_number);

    return output_histogram;
}


TestingAnalysis::BinaryClassificationRates TestingAnalysis::calculate_binary_classification_rates(const type decision_threshold) const
{
    const auto [targets, outputs] = get_targets_and_outputs("Testing");

    const vector<Index> testing_indices = dataset->get_sample_indices("Testing");

    BinaryClassificationRates binary_classification_rates;

    binary_classification_rates.true_positives_indices = calculate_true_positive_samples(targets, outputs, testing_indices, decision_threshold);
    binary_classification_rates.false_positives_indices = calculate_false_positive_samples(targets, outputs, testing_indices, decision_threshold);
    binary_classification_rates.false_negatives_indices = calculate_false_negative_samples(targets, outputs, testing_indices, decision_threshold);
    binary_classification_rates.true_negatives_indices = calculate_true_negative_samples(targets, outputs, testing_indices, decision_threshold);

    return binary_classification_rates;
}


vector<Index> TestingAnalysis::calculate_true_positive_samples(const MatrixR& targets,
                                                               const MatrixR& outputs,
                                                               const vector<Index>& testing_indices,
                                                               type decision_threshold) const
{
    const Index rows_number = targets.rows();

    VectorI true_positives_indices_copy(rows_number);

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


vector<Index> TestingAnalysis::calculate_false_positive_samples(const MatrixR& targets,
                                                                const MatrixR& outputs,
                                                                const vector<Index>& testing_indices,
                                                                type decision_threshold) const
{
    const Index rows_number = targets.rows();

    vector<Index> false_positives_indices_copy(rows_number);

    Index index = 0;

    for(Index i = 0; i < rows_number; i++)
        if(targets(i,0) < decision_threshold && outputs(i,0) >= decision_threshold)
            false_positives_indices_copy[index++] = testing_indices[i];

    const vector<Index> false_positives_indices(false_positives_indices_copy.begin(),
                                                false_positives_indices_copy.begin() + index);

    return false_positives_indices;
}


vector<Index> TestingAnalysis::calculate_false_negative_samples(const MatrixR& targets,
                                                                const MatrixR& outputs,
                                                                const vector<Index>& testing_indices,
                                                                type decision_threshold) const
{
    const Index rows_number = targets.rows();

    vector<Index> false_negatives_indices_copy(rows_number);

    Index index = 0;

    for(Index i = 0; i < rows_number; i++)
        if(targets(i,0) > decision_threshold && outputs(i,0) < decision_threshold)
            false_negatives_indices_copy[index++] = testing_indices[i];

    const vector<Index> false_negatives_indices(false_negatives_indices_copy.begin(),
                                                false_negatives_indices_copy.begin() + index);

    return false_negatives_indices;
}


vector<Index> TestingAnalysis::calculate_true_negative_samples(const MatrixR& targets,
                                                               const MatrixR& outputs,
                                                               const vector<Index>& testing_indices,
                                                               type decision_threshold) const
{
    const Index rows_number = targets.rows();

    vector<Index> true_negatives_indices_copy(rows_number);

    Index index = 0;

    for(Index i = 0; i < rows_number; i++)
        if(targets(i,0) < decision_threshold && outputs(i,0) < decision_threshold)
            true_negatives_indices_copy[index++] = testing_indices[i];

    vector<Index> true_negatives_indices(true_negatives_indices_copy.begin(),
                                         true_negatives_indices_copy.begin() + index);

    return true_negatives_indices;
}


VectorR TestingAnalysis::calculate_multiple_classification_precision() const
{
    const auto [targets, outputs] = get_targets_and_outputs("Testing");

    VectorR multiple_classification_tests(2);

    const MatrixI confusion_matrix = calculate_confusion_multiple_classification(targets, outputs);

    const type total_sum = static_cast<type>(confusion_matrix.sum());

    const type diagonal_sum = static_cast<type>(confusion_matrix.diagonal().sum());

    const type off_diagonal_sum = total_sum - diagonal_sum;

    multiple_classification_tests(0) = diagonal_sum/type(total_sum);
    multiple_classification_tests(1) = off_diagonal_sum/type(total_sum);

    return multiple_classification_tests;
}


void TestingAnalysis::save_confusion(const filesystem::path& file_name) const
{
    const MatrixI confusion = calculate_confusion();

    const Index variables_number = confusion.rows();

    ofstream file(file_name);

    const vector<string> target_variable_names = dataset->get_feature_names("Target");

    file << ",";

    for(Index i = 0; i < confusion.rows(); i++)
    {
        file << target_variable_names[i];

        if(i != Index(target_variable_names.size()) - 1)
            file << ",";
    }

    file << endl;

    for(Index i = 0; i < variables_number; i++)
    {
        file << target_variable_names[i] << ",";

        for(Index j = 0; j < variables_number; j++)
            j == variables_number - 1
                ? file << confusion(i, j) << endl
                : file << confusion(i, j) << ",";
    }

    file.close();
}


void TestingAnalysis::save_multiple_classification_tests(const filesystem::path& file_name) const
{
    const VectorR multiple_classification_tests = calculate_multiple_classification_precision();

    ofstream file(file_name);

    file << "accuracy,error" << endl;
    file << multiple_classification_tests(0)* type(100) << "," << multiple_classification_tests(1)* type(100) << endl;

    file.close();
}


Tensor<VectorI, 2> TestingAnalysis::calculate_multiple_classification_rates() const
{
    const auto [targets, outputs] = get_targets_and_outputs("Testing");

    const vector<Index> testing_indices = dataset->get_sample_indices("Testing");

    return calculate_multiple_classification_rates(targets, outputs, testing_indices);
}


Tensor<VectorI, 2> TestingAnalysis::calculate_multiple_classification_rates(const MatrixR& targets,
                                                                                    const MatrixR& outputs,
                                                                                    const vector<Index>& testing_indices) const
{
    const Index samples_number = targets.rows();
    const Index targets_number = targets.cols();

    Tensor< VectorI, 2> multiple_classification_rates(targets_number, targets_number);

    // Count instances per class

    const MatrixI confusion = calculate_confusion_multiple_classification(targets, outputs);

    for(Index i = 0; i < targets_number; i++)
        for(Index j = 0; j < targets_number; j++)
            multiple_classification_rates(i, j).resize(confusion(i, j));

    // Save indices

    Index target_index;
    Index output_index;

    MatrixI indices(targets_number, targets_number);
    indices.setZero();

    for(Index i = 0; i < samples_number; i++)
    {
        target_index = maximal_index(targets.row(i));
        output_index = maximal_index(outputs.row(i));

        multiple_classification_rates(target_index, output_index)(indices(target_index, output_index))
            = testing_indices[i];

        indices(target_index, output_index)++;
    }

    return multiple_classification_rates;
}


Tensor<string, 2> TestingAnalysis::calculate_well_classified_samples(const MatrixR& targets,
                                                                     const MatrixR& outputs,
                                                                     const vector<string>& labels) const
{
    const Index samples_number = targets.rows();

    Tensor<string, 2> well_lassified_samples(samples_number, 4);

    Index predicted_class;
    Index actual_class;
    Index number_of_well_classified = 0;
    string class_name;

    const vector<string> target_variables_names = dataset->get_feature_names("Target");

    for(Index i = 0; i < samples_number; i++)
    {
        predicted_class = maximal_index(outputs.row(i));
        actual_class = maximal_index(targets.row(i));

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


Tensor<string, 2> TestingAnalysis::calculate_misclassified_samples(const MatrixR& targets,
                                                                   const MatrixR& outputs,
                                                                   const vector<string>& labels) const
{
    const Index samples_number = targets.rows();

    Index predicted_class;
    Index actual_class;
    string class_name;

    const vector<string> target_variables_names = neural_network->get_output_names();

    Index count_misclassified = 0;

    for(Index i = 0; i < samples_number; i++)
    {
        predicted_class = maximal_index(outputs.row(i));
        actual_class = maximal_index(targets.row(i));

        if(actual_class != predicted_class)
            count_misclassified++;
    }

    Tensor<string, 2> misclassified_samples(count_misclassified, 4);

    Index j = 0;

    for(Index i = 0; i < samples_number; i++)
    {
        predicted_class = maximal_index(outputs.row(i));
        actual_class = maximal_index(targets.row(i));

        if(actual_class == predicted_class) continue;

        misclassified_samples(j, 0) = labels[i];
        class_name = target_variables_names[actual_class];
        misclassified_samples(j, 1) = class_name;
        class_name = target_variables_names[predicted_class];
        misclassified_samples(j, 2) = class_name;
        misclassified_samples(j, 3) = to_string(double(outputs(i, predicted_class)));
        j++;
    }

    return misclassified_samples;
}


void TestingAnalysis::save_well_classified_samples(const MatrixR& targets,
                                                   const MatrixR& outputs,
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


void TestingAnalysis::save_misclassified_samples(const MatrixR& targets,
                                                 const MatrixR& outputs,
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


void TestingAnalysis::save_well_classified_samples_statistics(const MatrixR& targets,
                                                              const MatrixR& outputs,
                                                              const vector<string>& labels,
                                                              const filesystem::path& file_name) const
{
    const Tensor<string, 2> well_classified_samples = calculate_well_classified_samples(targets,
                                                                                        outputs,
                                                                                        labels);

    VectorR well_classified_numerical_probabilities(well_classified_samples.dimension(0));

    for(Index i = 0; i < well_classified_numerical_probabilities.size(); i++)
        well_classified_numerical_probabilities(i) = type(::atof(well_classified_samples(i, 3).c_str()));

    ofstream file(file_name);

    file << "minimum,maximum,mean,std" << endl
         << well_classified_numerical_probabilities.minCoeff() << ","
         << well_classified_numerical_probabilities.maxCoeff() << ","
         << well_classified_numerical_probabilities.mean() << ","
         << standard_deviation(well_classified_numerical_probabilities);
}


void TestingAnalysis::save_misclassified_samples_statistics(const MatrixR& targets,
                                                            const MatrixR& outputs,
                                                            const vector<string>& labels,
                                                            const filesystem::path& statistics_file_name) const
{
    const Tensor<string, 2> misclassified_samples = calculate_misclassified_samples(targets,
                                                                                    outputs,
                                                                                    labels);

    VectorR misclassified_numerical_probabilities(misclassified_samples.dimension(0));

    for(Index i = 0; i < misclassified_numerical_probabilities.size(); i++)
        misclassified_numerical_probabilities(i) = type(::atof(misclassified_samples(i, 3).c_str()));

    ofstream classification_statistics_file(statistics_file_name);

    classification_statistics_file << "minimum,maximum,mean,std" << endl
                                   << misclassified_numerical_probabilities.minCoeff() << ","
                                   << misclassified_numerical_probabilities.maxCoeff() << ","
                                   << misclassified_numerical_probabilities.mean() << ","
                                   << standard_deviation(misclassified_numerical_probabilities);
}


void TestingAnalysis::save_well_classified_samples_probability_histogram(const MatrixR& targets,
                                                                         const MatrixR& outputs,
                                                                         const vector<string>& labels,
                                                                         const filesystem::path& histogram_file_name) const
{
    const Tensor<string, 2> well_classified_samples = calculate_well_classified_samples(targets,
                                                                                        outputs,
                                                                                        labels);

    VectorR well_classified_numerical_probabilities(well_classified_samples.dimension(0));

    for(Index i = 0; i < well_classified_numerical_probabilities.size(); i++)
        well_classified_numerical_probabilities(i) = type(::atof(well_classified_samples(i, 3).c_str()));

    const Histogram misclassified_samples_histogram(well_classified_numerical_probabilities);

    misclassified_samples_histogram.save(histogram_file_name);
}


void TestingAnalysis::save_well_classified_samples_probability_histogram(const Tensor<string, 2>& well_classified_samples,
                                                                         const filesystem::path& histogram_file_name) const
{
    VectorR well_classified_numerical_probabilities(well_classified_samples.dimension(0));

    for(Index i = 0; i < well_classified_numerical_probabilities.size(); i++)
        well_classified_numerical_probabilities(i) = type(::atof(well_classified_samples(i, 3).c_str()));

    const Histogram misclassified_samples_histogram(well_classified_numerical_probabilities);

    misclassified_samples_histogram.save(histogram_file_name);
}


void TestingAnalysis::save_misclassified_samples_probability_histogram(const MatrixR& targets,
                                                                       const MatrixR& outputs,
                                                                       const vector<string>& labels,
                                                                       const filesystem::path& histogram_file_name) const
{
    const Tensor<string, 2> misclassified_samples = calculate_misclassified_samples(targets,
                                                                                    outputs,
                                                                                    labels);

    VectorR misclassified_numerical_probabilities(misclassified_samples.dimension(0));

    for(Index i = 0; i < misclassified_numerical_probabilities.size(); i++)
        misclassified_numerical_probabilities(i) = type(::atof(misclassified_samples(i, 3).c_str()));

    const Histogram misclassified_samples_histogram(misclassified_numerical_probabilities);

    misclassified_samples_histogram.save(histogram_file_name);
}


void TestingAnalysis::save_misclassified_samples_probability_histogram(const Tensor<string, 2>& misclassified_samples,
                                                                       const filesystem::path& histogram_file_name) const
{
    VectorR misclassified_numerical_probabilities(misclassified_samples.dimension(0));

    for(Index i = 0; i < misclassified_numerical_probabilities.size(); i++)
        misclassified_numerical_probabilities(i) = type(::atof(misclassified_samples(i, 3).c_str()));

    const Histogram misclassified_samples_histogram(misclassified_numerical_probabilities);

    misclassified_samples_histogram.save(histogram_file_name);
}


vector<VectorR> TestingAnalysis::calculate_error_autocorrelation(const Index maximum_past_time_steps) const
{
    const auto [targets, outputs] = get_targets_and_outputs("Testing");

    const Index targets_number = dataset->get_features_number("Target");

    const MatrixR error = outputs - targets;

    vector<VectorR> error_autocorrelations(targets_number);

    for(Index i = 0; i < targets_number; i++)
        error_autocorrelations[i] = autocorrelations(error.col(i), maximum_past_time_steps);

    return error_autocorrelations;
}


vector<VectorR> TestingAnalysis::calculate_inputs_errors_cross_correlation(const Index past_time_steps) const
{
    const Index targets_number = dataset->get_features_number("Target");

    const MatrixR inputs = dataset->get_data("Testing", "Input");

    const MatrixR targets = dataset->get_data("Testing", "Target");

    const MatrixR outputs = neural_network->calculate_outputs(inputs);

    const MatrixR errors = outputs - targets;

    vector<VectorR> inputs_errors_cross_correlation(targets_number);

    for(Index i = 0; i < targets_number; i++)
        inputs_errors_cross_correlation[i] = cross_correlations(inputs.col(i), errors.col(i), past_time_steps);

    return inputs_errors_cross_correlation;
}


pair<type, type> TestingAnalysis::test_transformer() const
{
    cout << "Testing transformer..." << endl;

    Transformer* transformer = static_cast<Transformer*>(neural_network);
    LanguageDataset* language_dataset = static_cast<LanguageDataset*>(dataset);

    const MatrixR context = language_dataset->get_data("Testing", "Input");
    const MatrixR input = language_dataset->get_data("Testing", "Decoder");
    const MatrixR target = language_dataset->get_data("Testing", "Target");

    const Index testing_batch_size = input.rows() > 2000 ? 2000 : input.rows();

    MatrixR testing_input(testing_batch_size, input.cols());

    for(Index i = 0; i < testing_batch_size; i++)
        testing_input.row(i) = input.row(i);

    MatrixR testing_context(testing_batch_size, context.cols());
    MatrixR testing_target(testing_batch_size, target.cols());

    for(Index i = 0; i < testing_batch_size; i++)
    {
        testing_context.row(i) = context.row(i);
        testing_target.row(i) = target.row(i);
    }

    //const Tensor3 outputs = transformer->calculate_outputs(testing_input, testing_context);

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
/*
    const type error = calculate_cross_entropy_error_3d(outputs, testing_target);

    const type accuracy = calculate_masked_accuracy(outputs, testing_target);

    return pair<type, type>(error, accuracy);
*/
    return pair<type, type>();
}


string TestingAnalysis::test_transformer(const vector<string>& context_string, bool imported_vocabulary) const
{
    cout<<"Testing transformer..."<<endl;
/*
    Transformer* transformer = static_cast<Transformer*>(neural_network);

    return transformer->calculate_outputs(context_string);
*/
    return string();
}


VectorR TestingAnalysis::calculate_binary_classification_tests(const type decision_threshold) const
{
    const MatrixI confusion = calculate_confusion(decision_threshold);

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

    VectorR binary_classification_test(15);

    binary_classification_test << classification_accuracy,
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
                                  markedness;

    return binary_classification_test;
}


void TestingAnalysis::print_binary_classification_tests() const
{
    const VectorR binary_classification_tests = calculate_binary_classification_tests();

    cout << "Binary classification tests: " << endl
         << "Classification accuracy : " << binary_classification_tests[0] << endl
         << "Error rate              : " << binary_classification_tests[1] << endl
         << "Sensitivity             : " << binary_classification_tests[2] << endl
         << "Specificity             : " << binary_classification_tests[3] << endl;
}


MatrixR TestingAnalysis::calculate_multiple_classification_tests() const
{
    const Index targets_number = dataset->get_features_number("Target");

    MatrixR multiple_classification_tests(targets_number + 2, 3);

    const MatrixI confusion = calculate_confusion();

    type total_precision = type(0);
    type total_recall = type(0);
    type total_f1_score= type(0);

    type total_weighted_precision = type(0);
    type total_weighted_recall = type(0);
    type total_weighted_f1_score= type(0);

    Index total_samples = 0;

    for(Index target_index = 0; target_index < targets_number; target_index++)
    {
        const Index true_positives = confusion(target_index, target_index);

        const Index row_sum = confusion(target_index, targets_number);
        const Index column_sum = confusion(targets_number, target_index);

        const Index false_negatives = row_sum - true_positives;
        const Index false_positives = column_sum - true_positives;

        const type precision = (true_positives + false_positives == 0)
                                   ? type(1.0)
                                   : type(true_positives) / type(true_positives + false_positives);

        const type recall = (true_positives + false_negatives == 0)
                                ? type(1.0)
                                : type(true_positives) / type(true_positives + false_negatives);

        const type f1_score = (precision + recall == 0)
                                  ? type(0)
                                  : type(2 * precision * recall) / type(precision + recall);

        multiple_classification_tests(target_index, 0) = precision;
        multiple_classification_tests(target_index, 1) = recall;
        multiple_classification_tests(target_index, 2) = f1_score;

        total_precision += precision;
        total_recall += recall;
        total_f1_score += f1_score;

        total_weighted_precision += precision * type(row_sum);
        total_weighted_recall += recall * type(row_sum);
        total_weighted_f1_score += f1_score * type(row_sum);

        total_samples += row_sum;
    }

    // Averages

    if (targets_number > 0)
    {
        multiple_classification_tests(targets_number, 0) = total_precision / targets_number;
        multiple_classification_tests(targets_number, 1) = total_recall / targets_number;
        multiple_classification_tests(targets_number, 2) = total_f1_score / targets_number;
    }

    if (total_samples > 0)
    {
        multiple_classification_tests(targets_number + 1, 0) = total_weighted_precision / total_samples;
        multiple_classification_tests(targets_number + 1, 1) = total_weighted_recall / total_samples;
        multiple_classification_tests(targets_number + 1, 2) = total_weighted_f1_score / total_samples;
    }

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

    if(!file.is_open())
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


void TestingAnalysis::GoodnessOfFitAnalysis::set(const VectorR& new_targets,
                                                 const VectorR& new_outputs,
                                                 type new_determination)
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
    cout << "ROC Curve analysis" << endl;

//    cout << "Roc Curve:\n" << roc_curve << endl;
    cout << "Area Under Curve: " << area_under_curve << endl;
    cout << "Confidence Limit: " << confidence_limit << endl;
    cout << "Optimal Threshold: " << optimal_threshold << endl;
}


#ifdef OPENNN_CUDA

MatrixI TestingAnalysis::calculate_confusion_cuda(const type decision_threshold) const
{
    check();

    const vector<Index> testing_indices = dataset->get_sample_indices("Testing");
    const vector<vector<Index>> testing_batches = dataset->get_batches(testing_indices, batch_size, false);

    const vector<Index> input_feature_indices = dataset->get_feature_indices("Input");
    const vector<Index> target_feature_indices = dataset->get_feature_indices("Target");
    //const vector<Index> decoder_feature_indices = dataset->get_feature_indices("Decoder");

    const Index outputs_number = neural_network->get_outputs_number();

    const Index confusion_matrix_size = (outputs_number == 1) ? 3 : (outputs_number + 1);

    MatrixI total_confusion_matrix(confusion_matrix_size, confusion_matrix_size);
    total_confusion_matrix.setZero();

    BatchCuda testing_batch(batch_size, dataset);
    ForwardPropagationCuda testing_forward_propagation(batch_size, neural_network);

    neural_network->allocate_parameters_device();
    neural_network->copy_parameters_device();

    for(const auto& current_batch_indices : testing_batches)
    {
        const Index current_batch_size = current_batch_indices.size();
        if (current_batch_size == 0) continue;

        if (current_batch_size != batch_size)
        {
            testing_forward_propagation.free();
            testing_batch.set(current_batch_size, dataset);
            testing_forward_propagation.set(current_batch_size, neural_network);
        }

        testing_batch.fill(current_batch_indices,
                           input_feature_indices,
                           //decoder_feature_indices,
                           target_feature_indices);

        neural_network->forward_propagate(testing_batch.get_inputs_device(),
                                          testing_forward_propagation,
                                          false);

        const float* outputs_device = testing_forward_propagation.get_last_trainable_layer_outputs_device().data;

        MatrixR batch_outputs(current_batch_size, outputs_number);
        cudaMemcpy(batch_outputs.data(), outputs_device, current_batch_size * outputs_number * sizeof(type), cudaMemcpyDeviceToHost);

        const MatrixR batch_targets = dataset->get_data_from_indices(current_batch_indices, target_feature_indices);
        MatrixI batch_confusion = calculate_confusion(batch_outputs, batch_targets, decision_threshold);
        total_confusion_matrix += batch_confusion;
    }

    total_confusion_matrix(confusion_matrix_size - 1, confusion_matrix_size - 1) = testing_indices.size();

    return total_confusion_matrix;
}

#endif

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
