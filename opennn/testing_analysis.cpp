//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   T E S T I N G   A N A L Y S I S   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "testing_analysis.h"
#include "tensor_utilities.h"

namespace opennn
{

/// Default constructor.
/// It creates a testing analysis object neither associated with a neural network nor to a mathematical model or a data set.
/// By default, it constructs the function regression testing object.

TestingAnalysis::TestingAnalysis()
{
    set_default();
}


/// Neural network and data set constructor.
/// It creates a testing analysis object associated with a neural network and to a data set.
/// By default, it constructs the function regression testing object.
/// @param new_neural_network_pointer Pointer to a neural network object.
/// @param new_data_set_pointer Pointer to a data set object.

TestingAnalysis::TestingAnalysis(NeuralNetwork* new_neural_network_pointer, DataSet* new_data_set_pointer)
    : neural_network_pointer(new_neural_network_pointer),
      data_set_pointer(new_data_set_pointer)
{
    set_default();
}


/// Destructor.
/// It deletes the function regression testing, classification testing, time series prediction testing and inverse problem testing objects.

TestingAnalysis::~TestingAnalysis()
{
    delete thread_pool;
    delete thread_pool_device;
}


/// Returns a pointer to the neural network object which is to be tested.

NeuralNetwork* TestingAnalysis::get_neural_network_pointer() const
{
#ifdef OPENNN_DEBUG

    if(!neural_network_pointer)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: TestingAnalysis class.\n"
               << "NeuralNetwork* get_neural_network_pointer() const method.\n"
               << "Neural network pointer is nullptr.\n";

        throw invalid_argument(buffer.str());
    }

#endif

    return neural_network_pointer;
}


/// Returns a pointer to the data set object on which the neural network is tested.

DataSet* TestingAnalysis::get_data_set_pointer() const
{
#ifdef OPENNN_DEBUG

    if(!data_set_pointer)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: TestingAnalysis class.\n"
               << "DataSet* get_data_set_pointer() const method.\n"
               << "Data set pointer is nullptr.\n";

        throw invalid_argument(buffer.str());
    }

#endif

    return data_set_pointer;
}


/// Returns true if messages from this class can be displayed on the screen,
/// or false if messages from this class can't be displayed on the screen.

const bool& TestingAnalysis::get_display() const
{
    return display;
}


/// Sets some default values to the testing analysis object:
/// <ul>
/// <li> Display: True.
/// </ul>

void TestingAnalysis::set_default()
{
    delete thread_pool;
    delete thread_pool_device;

    const int n = omp_get_max_threads();
    thread_pool = new ThreadPool(n);
    thread_pool_device = new ThreadPoolDevice(thread_pool, n);
}


void TestingAnalysis::set_threads_number(const int& new_threads_number)
{
    if(thread_pool != nullptr) delete this->thread_pool;
    if(thread_pool_device != nullptr) delete this->thread_pool_device;

    thread_pool = new ThreadPool(new_threads_number);
    thread_pool_device = new ThreadPoolDevice(thread_pool, new_threads_number);
}


/// Sets a new neural network object to be tested.
/// @param new_neural_network_pointer Pointer to a neural network object.

void TestingAnalysis::set_neural_network_pointer(NeuralNetwork* new_neural_network_pointer)
{
    neural_network_pointer = new_neural_network_pointer;
}


/// Sets a new data set to be used for validating the quality of a trained neural network.
/// @param new_data_set_pointer Pointer to a data set object.

void TestingAnalysis::set_data_set_pointer(DataSet* new_data_set_pointer)
{
    data_set_pointer = new_data_set_pointer;
}


/// Sets a new display value.
/// If it is set to true messages from this class are displayed on the screen;
/// if it is set to false messages from this class are not displayed on the screen.
/// @param new_display Display value.

void TestingAnalysis::set_display(const bool& new_display)
{
    display = new_display;
}


/// Checks that:
/// <ul>
/// <li> The neural network pointer is not nullptr.
/// <li> The data set pointer is not nullptr.
/// </ul>

void TestingAnalysis::check() const
{
    ostringstream buffer;

    if(!neural_network_pointer)
    {
        buffer << "OpenNN Exception: TestingAnalysis class.\n"
               << "void check() const method.\n"
               << "Neural network pointer is nullptr.\n";

        throw invalid_argument(buffer.str());
    }

    if(!data_set_pointer)
    {
        buffer << "OpenNN Exception: TestingAnalysis class.\n"
               << "void check() const method.\n"
               << "Data set pointer is nullptr.\n";

        throw invalid_argument(buffer.str());
    }
}


/// Performs a goodness-of-fit analysis between the testing samples in the data set and
/// the corresponding neural network outputs.
/// It returns all the provided parameters in a vector of vectors.
/// The number of elements in the vector is equal to the number of output variables.
/// The size of each element is equal to the number of regression parameters(2).
/// In this way, each subvector contains the regression parameters intercept and slope of an output variable.

Tensor<Correlation, 1> TestingAnalysis::linear_correlation() const
{
#ifdef OPENNN_DEBUG

    check();

    const Index testing_samples_number = data_set_pointer->get_testing_samples_number();

    ostringstream buffer;

    if(testing_samples_number == 0)
    {
        buffer << "OpenNN Exception: TestingAnalysis class.\n"
               << "Tensor<Correlation, 1> linear_correlation() const method.\n"
               << "Number of testing samples is zero.\n";

        throw invalid_argument(buffer.str());
    }

#endif

    // Calculate regression parameters

    Tensor<type, 2> inputs = data_set_pointer->get_testing_input_data();

    Tensor<Index, 1> inputs_dimensions = get_dimensions(inputs);

    Tensor<type, 2> targets = data_set_pointer->get_testing_target_data();

    Tensor<type, 2> outputs;

    outputs = neural_network_pointer->calculate_outputs(inputs.data(), inputs_dimensions);

    return linear_correlation(targets,outputs);
}


Tensor<Correlation, 1> TestingAnalysis::linear_correlation(const Tensor<type, 2>& target, const Tensor<type, 2>& output) const
{
#ifdef OPENNN_DEBUG

    if(target.dimension(0) != output.dimension(0) || target.dimension(1) != output.dimension(1))
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: " << LOG << endl
               << "Targets and outputs dimensions must be the same.\n";

        throw invalid_argument(buffer.str());
    }

#endif

    const Index outputs_number = data_set_pointer->get_target_variables_number();

    Tensor<Correlation, 1> linear_correlation(outputs_number);

    for(Index i = 0; i < outputs_number; i++)
    {
        linear_correlation[i] = opennn::linear_correlation(thread_pool_device, output.chip(i,1), target.chip(i,1));
    }

    return linear_correlation;
}


void TestingAnalysis::print_linear_regression_correlations() const
{
    const Tensor<Correlation, 1> linear_correlation = this->linear_correlation();

    const Tensor<string, 1> targets_name = data_set_pointer->get_target_variables_names();

    const Index targets_number = linear_correlation.size();

    for(Index i = 0; i < targets_number; i++)
    {
        cout << targets_name[i] << " correlation: " << linear_correlation[i].r << endl;
    }
}


/// Performs a goodness of fit analysis of a neural network on the testing indices of a data set.
/// It returns a goodness of fit analysis results structure, which consists of:
/// <ul>
/// <li> Goodness of fit parameters.
/// <li> Scaled target and output data.
/// </ul>

Tensor<TestingAnalysis::GoodnessOfFitAnalysis, 1> TestingAnalysis::perform_goodness_of_fit_analysis() const
{
    check();

    const Index outputs_number = neural_network_pointer->get_outputs_number();

    const Index testing_samples_number = data_set_pointer->get_testing_samples_number();

    if(testing_samples_number == 0)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: TestingAnalysis class.\n"
               << "GoodnessOfFit perform_linear_regression_analysis() const method.\n"
               << "Number of testing samples is zero.\n";

        throw invalid_argument(buffer.str());
    }

    Tensor<type, 2> testing_inputs = data_set_pointer->get_testing_input_data();

    Tensor<Index, 1> testing_inputs_dimensions = get_dimensions(testing_inputs);

    Tensor<type, 2> testing_targets = data_set_pointer->get_testing_target_data();

    // Neural network

    Tensor<type, 2> testing_outputs;

    testing_outputs = neural_network_pointer->calculate_outputs(testing_inputs.data(), testing_inputs_dimensions);

    // Approximation testing stuff

    Tensor<GoodnessOfFitAnalysis, 1> goodness_of_fit_results(outputs_number);

    for(Index i = 0;  i < outputs_number; i++)
    {        
        const Tensor<type,1> targets = testing_targets.chip(i,1);
        const Tensor<type,1> outputs = testing_outputs.chip(i,1);

        type determination_coefficient = calculate_determination_coefficient(outputs, targets);

        goodness_of_fit_results[i].targets = targets;
        goodness_of_fit_results[i].outputs = outputs;

        goodness_of_fit_results[i].determination = determination_coefficient;

    }

    return goodness_of_fit_results;
}


void TestingAnalysis::print_goodness_of_fit_analysis() const
{
    const Tensor<GoodnessOfFitAnalysis, 1> linear_regression_analysis = perform_goodness_of_fit_analysis();

    for(Index i = 0; i < linear_regression_analysis.size(); i++)
    {
        linear_regression_analysis(i).print();
    }
}


/// Calculates the errors between the outputs from a neural network and the testing samples in a data set.
/// It returns a vector of tree matrices:
/// <ul>
/// <li> Absolute error.
/// <li> Relative error.
/// <li> Percentage error.
/// </ul>
/// The number of rows in each matrix is the number of testing samples in the data set.
/// The number of columns is the number of outputs in the neural network.

Tensor<type, 3> TestingAnalysis::calculate_error_data() const
{
    // Data set

#ifdef OPENNN_DEBUG

    check();

#endif

    const Index testing_samples_number = data_set_pointer->get_testing_samples_number();

#ifdef OPENNN_DEBUG

    ostringstream buffer;

    if(testing_samples_number == 0)
    {
        buffer << "OpenNN Exception: TestingAnalysis class.\n"
               << "Tensor<Tensor<type, 2>, 1> calculate_error_data() const.\n"
               << "Number of testing samples is zero.\n";

        throw invalid_argument(buffer.str());
    }

#endif

    Tensor<type, 2> inputs = data_set_pointer->get_testing_input_data();

    Tensor<Index, 1> inputs_dimensions = get_dimensions(inputs);

    Tensor<type, 2> targets = data_set_pointer->get_testing_target_data();

    // Neural network

    Index outputs_number = neural_network_pointer->get_outputs_number();

    Tensor<type, 2> outputs;

    outputs = neural_network_pointer->calculate_outputs(inputs.data(), inputs_dimensions);

    const UnscalingLayer* unscaling_layer_pointer = neural_network_pointer->get_unscaling_layer_pointer();

#ifdef OPENNN_DEBUG

    if(!unscaling_layer_pointer)
    {
        buffer << "OpenNN Exception: TestingAnalysis class.\n"
               << "Tensor<Tensor<type, 2>, 1> calculate_error_data() const.\n"
               << "Unscaling layer is nullptr.\n";

        throw invalid_argument(buffer.str());
    }

#endif

    const Tensor<type, 1>& outputs_minimum = unscaling_layer_pointer->get_minimums();

    const Tensor<type, 1>& outputs_maximum = unscaling_layer_pointer->get_maximums();

    // Error data

    Tensor<type, 3> error_data(testing_samples_number, 3, outputs_number);

    // Absolute error

   Tensor<type, 2> difference_absolute_value = (targets - outputs).abs();

   for(Index i = 0; i < outputs_number; i++)
   {
       for(Index j = 0; j < testing_samples_number; j++)
       {
           // Absolute error
           error_data(j,0,i) = difference_absolute_value(j,i);
           // Relative error
           error_data(j,1,i) = difference_absolute_value(j,i)/abs(outputs_maximum(i)-outputs_minimum(i));
           // Percentage error
           error_data(j,2,i) = (difference_absolute_value(j,i)*static_cast<type>(100.0))/abs(outputs_maximum(i)-outputs_minimum(i));
       }
   }

    return error_data;
}


/// Calculates the percentege errors between the outputs from a neural network and the testing samples in a data set.
/// The number of rows in each matrix is the number of testing samples in the data set.

Tensor<type, 2> TestingAnalysis::calculate_percentage_error_data() const
{
    // Data set

#ifdef OPENNN_DEBUG

    check();

#endif

    const Index testing_samples_number = data_set_pointer->get_testing_samples_number();

#ifdef OPENNN_DEBUG

    ostringstream buffer;

    if(testing_samples_number == 0)
    {
        buffer << "OpenNN Exception: TestingAnalysis class.\n"
               << "Tensor<Tensor<type, 2>, 1> calculate_error_data() const.\n"
               << "Number of testing samples is zero.\n";

        throw invalid_argument(buffer.str());
    }

#endif

    Tensor<type, 2> inputs = data_set_pointer->get_testing_input_data();

    Tensor<Index, 1> inputs_dimensions = get_dimensions(inputs);

    Tensor<type, 2> targets = data_set_pointer->get_testing_target_data();

    // Neural network

    Index outputs_number = neural_network_pointer->get_outputs_number();

    Tensor<type, 2> outputs;

    outputs = neural_network_pointer->calculate_outputs(inputs.data(), inputs_dimensions);

    const UnscalingLayer* unscaling_layer_pointer = neural_network_pointer->get_unscaling_layer_pointer();

#ifdef OPENNN_DEBUG

    if(!unscaling_layer_pointer)
    {
        buffer << "OpenNN Exception: TestingAnalysis class.\n"
               << "Tensor<Tensor<type, 1>, 1> calculate_percentage_error_data() const.\n"
               << "Unscaling layer is nullptr.\n";

        throw invalid_argument(buffer.str());
    }

#endif

    const Tensor<type, 1>& outputs_minimum = unscaling_layer_pointer->get_minimums();
    const Tensor<type, 1>& outputs_maximum = unscaling_layer_pointer->get_maximums();

    // Error data

    Tensor<type, 2> error_data(testing_samples_number, outputs_number);

    Tensor<type, 2> difference_value = (targets - outputs);

    for(Index i = 0; i < testing_samples_number; i++)
    {
       for(Index j = 0; j < outputs_number; j++)
       {
           error_data(i,j) = (difference_value(i,j)*static_cast<type>(100.0))/abs(outputs_maximum(j)-outputs_minimum(j));
       }
    }

    return error_data;
}


Tensor<Descriptives, 1> TestingAnalysis::calculate_absolute_errors_descriptives() const
{
    // Data set

    Tensor<type, 2> inputs = data_set_pointer->get_testing_input_data();

    Tensor<Index, 1> inputs_dimensions = get_dimensions(inputs);

    Tensor<type, 2> targets = data_set_pointer->get_testing_target_data();

    // Neural network

    Tensor<type, 2> outputs;

    outputs = neural_network_pointer->calculate_outputs(inputs.data(), inputs_dimensions);

    // Error descriptives

    return calculate_absolute_errors_descriptives(targets, outputs);
}


Tensor<Descriptives, 1> TestingAnalysis::calculate_absolute_errors_descriptives(const Tensor<type, 2>& targets,
                                                                                const Tensor<type, 2>& outputs) const
{
    const Tensor<type, 2> diff = (targets-outputs).abs();

    return descriptives(diff);
}


Tensor<Descriptives, 1> TestingAnalysis::calculate_percentage_errors_descriptives() const
{
    // Data set

    Tensor<type, 2> inputs = data_set_pointer->get_testing_input_data();

    Tensor<Index, 1> inputs_dimensions = get_dimensions(inputs);

    Tensor<type, 2> targets = data_set_pointer->get_testing_target_data();

    // Neural network

    Tensor<type, 2> outputs;

    outputs = neural_network_pointer->calculate_outputs(inputs.data(), inputs_dimensions);

    // Error descriptives

    return calculate_percentage_errors_descriptives(targets,outputs);
}


Tensor<Descriptives, 1> TestingAnalysis::calculate_percentage_errors_descriptives(const Tensor<type, 2>& targets,
        const Tensor<type, 2>& outputs) const
{
    const Tensor<type, 2> diff = (static_cast<type>(100)*(targets-outputs).abs())/targets;

    return descriptives(diff);
}


/// Calculates the basic descriptives on the error data.
/// <ul>
/// <li> Minimum.
/// <li> Maximum.
/// <li> Mean.
/// <li> Standard deviation
/// </ul>

Tensor<Tensor<Descriptives, 1>, 1> TestingAnalysis::calculate_error_data_descriptives() const
{
    // Neural network

    const Index outputs_number = neural_network_pointer->get_outputs_number();

    const Index testing_samples_number = data_set_pointer->get_testing_samples_number();

    // Testing analysis stuff

    Tensor<Tensor<Descriptives, 1>, 1> descriptives(outputs_number);

    Tensor<type, 3> error_data = calculate_error_data();

    Index index = 0;

    for(Index i = 0; i < outputs_number; i++)
    {
        TensorMap< Tensor<type, 2> > matrix_error(error_data.data()+index, testing_samples_number, 3);

        Tensor<type, 2> matrix(matrix_error);

        descriptives[i] = opennn::descriptives(matrix);

        index += testing_samples_number*3;
    }

    return descriptives;
}


void TestingAnalysis::print_error_data_descriptives() const
{
    const Index targets_number = data_set_pointer->get_target_variables_number();

    const Tensor<string, 1> targets_name = data_set_pointer->get_target_variables_names();

    const Tensor<Tensor<Descriptives, 1>, 1> error_data_statistics = calculate_error_data_descriptives();

    for(Index i = 0; i < targets_number; i++)
    {
        cout << targets_name[i] << endl;
        cout << "Minimum error: " << error_data_statistics[i][0].minimum << endl;
        cout << "Maximum error: " << error_data_statistics[i][0].maximum << endl;
        cout << "Mean error: " << error_data_statistics[i][0].mean << " " << endl;
        cout << "Standard deviation error: " << error_data_statistics[i][0].standard_deviation << " " << endl;

        cout << "Minimum percentage error: " << error_data_statistics[i][2].minimum << " %" << endl;
        cout << "Maximum percentage error: " << error_data_statistics[i][2].maximum << " %" << endl;
        cout << "Mean percentage error: " << error_data_statistics[i][2].mean << " %" << endl;
        cout << "Standard deviation percentage error: " << error_data_statistics[i][2].standard_deviation << " %" << endl;
        cout << endl;
    }
}


/// Calculates histograms for the relative errors of all the output variables.
/// The number of bins is set by the user.
/// @param bins_number Number of bins in the histograms.

Tensor<Histogram, 1> TestingAnalysis::calculate_error_data_histograms(const Index& bins_number) const
{
    const Tensor<type, 2> error_data = calculate_percentage_error_data();

    const Index outputs_number = error_data.dimension(1);

    Tensor<Histogram, 1> histograms(outputs_number);

    for(Index i = 0; i < outputs_number; i++)
    {
        histograms(i) = histogram_centered(error_data.chip(i,1), type(0), bins_number);
    }

    return histograms;
}


/// Returns a vector with the indices of the samples which have the greatest error.
/// @param samples_number Number of maximal indices to be computed.

Tensor<Tensor<Index, 1>, 1> TestingAnalysis::calculate_maximal_errors(const Index& samples_number) const
{
    Tensor<type, 3> error_data = calculate_error_data();

    const Index outputs_number = error_data.dimension(2);
    const Index testing_samples_number = error_data.dimension(0);

    Tensor<Tensor<Index, 1>, 1> maximal_errors(samples_number);

    Index index = 0;

    for(Index i = 0; i < outputs_number; i++)
    {
        TensorMap< Tensor<type, 2> > matrix_error(error_data.data()+index, testing_samples_number, 3);

        maximal_errors[i] = maximal_indices(matrix_error.chip(0,1), samples_number);

        index += testing_samples_number*3;
    }

    return maximal_errors;
}


/// This method calculates the training, selection and testing errors.
/// Returns a matrix with the differents errors.

Tensor<type, 2> TestingAnalysis::calculate_errors() const
{
    Tensor<type, 2> errors(5,3);

    const Tensor<type, 1> training_errors = calculate_training_errors();
    const Tensor<type, 1> selection_errors = calculate_selection_errors();
    const Tensor<type, 1> testing_errors = calculate_testing_errors();

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

    const Tensor<type, 1> training_errors = calculate_binary_classification_training_errors();
    const Tensor<type, 1> selection_errors = calculate_binary_classification_selection_errors();
    const Tensor<type, 1> testing_errors = calculate_binary_classification_testing_errors();

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

    const Tensor<type, 1> training_errors = calculate_multiple_classification_training_errors();
    const Tensor<type, 1> selection_errors = calculate_multiple_classification_selection_errors();
    const Tensor<type, 1> testing_errors = calculate_multiple_classification_testing_errors();

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


Tensor<type, 1> TestingAnalysis::calculate_training_errors() const
{
    // Data set

#ifdef OPENNN_DEBUG

    check();

#endif

    const Index training_samples_number = data_set_pointer->get_training_samples_number();

#ifdef OPENNN_DEBUG

    ostringstream buffer;

    if(training_samples_number == 0)
    {
        buffer << "OpenNN Exception: TestingAnalysis class.\n"
               << "Tensor<type, 1> calculate_training_errors() const.\n"
               << "Number of training samples is zero.\n";

        throw invalid_argument(buffer.str());
    }

#endif

    Tensor<type, 2> inputs = data_set_pointer->get_training_input_data();

    Tensor<Index, 1> inputs_dimensions = get_dimensions(inputs);

    Tensor<type, 2> targets = data_set_pointer->get_training_target_data();

    // Neural network

    Tensor<type, 2> outputs;

    outputs = neural_network_pointer->calculate_outputs(inputs.data(), inputs_dimensions);

    Tensor<type, 1> errors(4);

    // Results

    const Tensor<type, 0> sum_squared_error = (outputs-targets).square().sum().sqrt();

    errors(0) = sum_squared_error(0);
    errors(1) = errors(0)/type(training_samples_number);
    errors(2) = sqrt(errors(1));
    errors(3) = calculate_normalized_squared_error(targets, outputs);

    return errors;
}


Tensor<type, 1> TestingAnalysis::calculate_binary_classification_training_errors() const
{
    // Data set

#ifdef OPENNN_DEBUG

    check();

#endif

    const Index training_samples_number = data_set_pointer->get_training_samples_number();

#ifdef OPENNN_DEBUG

    ostringstream buffer;

    if(training_samples_number == 0)
    {
        buffer << "OpenNN Exception: TestingAnalysis class.\n"
               << "Tensor<type, 1> calculate_binary_classification_training_errors() const.\n"
               << "Number of training samples is zero.\n";

        throw invalid_argument(buffer.str());
    }

#endif

    Tensor<type, 2> inputs = data_set_pointer->get_training_input_data();

    Tensor<Index, 1> inputs_dimensions = get_dimensions(inputs);

    Tensor<type, 2> targets = data_set_pointer->get_training_target_data();

    // Neural network

    Tensor<type, 2> outputs;

    outputs = neural_network_pointer->calculate_outputs(inputs.data(), inputs_dimensions);

    Tensor<type, 1> errors(6);

    // Results
    const Tensor<type, 0> sum_squared_error = (outputs-targets).square().sum().sqrt();

    // SSE
    errors(0) = sum_squared_error(0);

    // MSE
    errors(1) = errors(0)/type(training_samples_number);

    // RMSE
    errors(2) = sqrt(errors(1));

    // NSE
    errors(3) = calculate_normalized_squared_error(targets, outputs);

    // CE
    errors(4) = calculate_cross_entropy_error(targets, outputs);

    // WSE
    errors(5) = calculate_weighted_squared_error(targets, outputs);

    return errors;
}

Tensor<type, 1> TestingAnalysis::calculate_multiple_classification_training_errors() const
{
    // Data set

#ifdef OPENNN_DEBUG

    check();

#endif

    const Index training_samples_number = data_set_pointer->get_training_samples_number();

#ifdef OPENNN_DEBUG

    ostringstream buffer;

    if(training_samples_number == 0)
    {
        buffer << "OpenNN Exception: TestingAnalysis class.\n"
               << "Tensor<type, 1> calculate_multiple_classification_training_errors() const.\n"
               << "Number of training samples is zero.\n";

        throw invalid_argument(buffer.str());
    }

#endif

    Tensor<type, 2> inputs = data_set_pointer->get_training_input_data();

    Tensor<Index, 1> inputs_dimensions = get_dimensions(inputs);

    Tensor<type, 2> targets = data_set_pointer->get_training_target_data();

    // Neural network

    Tensor<type, 2> outputs;

    outputs = neural_network_pointer->calculate_outputs(inputs.data(), inputs_dimensions);

    Tensor<type, 1> errors(5);

    // Results

    const Tensor<type, 0> sum_squared_error = (outputs-targets).square().sum().sqrt();

    errors(0) = sum_squared_error(0);
    errors(1) = errors(0)/type(training_samples_number);
    errors(2) = sqrt(errors(1));
    errors(3) = calculate_normalized_squared_error(targets, outputs);
    errors(4) = calculate_cross_entropy_error(targets, outputs); // NO

    return errors;
}

Tensor<type, 1> TestingAnalysis::calculate_selection_errors() const
{
    // Data set

#ifdef OPENNN_DEBUG

    check();

#endif

    const Index selection_samples_number = data_set_pointer->get_selection_samples_number();

#ifdef OPENNN_DEBUG

    ostringstream buffer;

    if(selection_samples_number == 0)
    {
        buffer << "OpenNN Exception: TestingAnalysis class.\n"
               << "Tensor<type, 1> calculate_selection_errors() const.\n"
               << "Number of selection samples is zero.\n";

        throw invalid_argument(buffer.str());
    }

#endif

    Tensor<type, 2> inputs = data_set_pointer->get_selection_input_data();

    Tensor<Index, 1> inputs_dimensions = get_dimensions(inputs);

    Tensor<type, 2> targets = data_set_pointer->get_selection_target_data();

    // Neural network

    Tensor<type, 2> outputs;

    outputs = neural_network_pointer->calculate_outputs(inputs.data(), inputs_dimensions);

    Tensor<type, 1> errors(4);

    // Results

    const Tensor<type, 0> sum_squared_error = (outputs-targets).square().sum().sqrt();

    errors(0) = sum_squared_error(0);
    errors(1) = errors(0)/type(selection_samples_number);
    errors(2) = sqrt(errors(1));
    errors(3) = calculate_normalized_squared_error(targets, outputs);

    return errors;
}


Tensor<type, 1> TestingAnalysis::calculate_binary_classification_selection_errors() const
{
    // Data set

#ifdef OPENNN_DEBUG

    check();

#endif

    const Index selection_samples_number = data_set_pointer->get_selection_samples_number();

#ifdef OPENNN_DEBUG

    ostringstream buffer;

    if(selection_samples_number == 0)
    {
        buffer << "OpenNN Exception: TestingAnalysis class.\n"
               << "Tensor<type, 1> calculate_binary_classification_selection_errors() const.\n"
               << "Number of selection samples is zero.\n";

        throw invalid_argument(buffer.str());
    }

#endif

    Tensor<type, 2> inputs = data_set_pointer->get_selection_input_data();

    Tensor<Index, 1> inputs_dimensions = get_dimensions(inputs);

    Tensor<type, 2> targets = data_set_pointer->get_selection_target_data();

    // Neural network

    Tensor<type, 2> outputs;

    outputs = neural_network_pointer->calculate_outputs(inputs.data(), inputs_dimensions);

    Tensor<type, 1> errors(6);

    // Results

    const Tensor<type, 0> sum_squared_error = (outputs-targets).square().sum().sqrt();

    errors(0) = sum_squared_error(0);
    errors(1) = errors(0)/type(selection_samples_number);
    errors(2) = sqrt(errors(1));
    errors(3) = calculate_normalized_squared_error(targets, outputs);
    errors(4) = calculate_cross_entropy_error(targets, outputs);
    errors(5) = calculate_weighted_squared_error(targets, outputs);

    return errors;
}


Tensor<type, 1> TestingAnalysis::calculate_multiple_classification_selection_errors() const
{
    // Data set

#ifdef OPENNN_DEBUG

    check();

#endif

    const Index selection_samples_number = data_set_pointer->get_selection_samples_number();

#ifdef OPENNN_DEBUG

    ostringstream buffer;

    if(selection_samples_number == 0)
    {
        buffer << "OpenNN Exception: TestingAnalysis class.\n"
               << "Tensor<type, 1> calculate_multiple_classification_selection_errors() const.\n"
               << "Number of selection samples is zero.\n";

        throw invalid_argument(buffer.str());
    }

#endif

    Tensor<type, 2> inputs = data_set_pointer->get_selection_input_data();

    Tensor<Index, 1> inputs_dimensions = get_dimensions(inputs);

    Tensor<type, 2> targets = data_set_pointer->get_selection_target_data();

    // Neural network

    Tensor<type, 2> outputs;

    outputs = neural_network_pointer->calculate_outputs(inputs.data(), inputs_dimensions);

    Tensor<type, 1> errors(5);

    // Results

    const Tensor<type, 0> sum_squared_error = (outputs-targets).square().sum().sqrt();

    errors(0) = sum_squared_error(0);
    errors(1) = errors(0)/type(selection_samples_number);
    errors(2) = sqrt(errors(1));
    errors(3) = calculate_normalized_squared_error(targets, outputs);
    errors(4) = calculate_cross_entropy_error(targets, outputs);

    return errors;
}


/// Returns a vector containing the values of the errors between the outputs of the neural network
/// and the targets. The vector consists of:
/// <ul>
/// <li> Sum squared error.
/// <li> Mean squared error.
/// <li> Root mean squared error.
/// <li> Normalized squared error.
/// </ul>

Tensor<type, 1> TestingAnalysis::calculate_testing_errors() const
{
    // Data set

#ifdef OPENNN_DEBUG

    check();

#endif

    const Index testing_samples_number = data_set_pointer->get_testing_samples_number();

#ifdef OPENNN_DEBUG

    ostringstream buffer;

    if(testing_samples_number == 0)
    {
        buffer << "OpenNN Exception: TestingAnalysis class.\n"
               << "Tensor<Tensor<type, 2>, 1> calculate_testing_errors() const.\n"
               << "Number of testing samples is zero.\n";

        throw invalid_argument(buffer.str());
    }

#endif

    Tensor<type, 2> inputs = data_set_pointer->get_testing_input_data();

    Tensor<Index, 1> inputs_dimensions = get_dimensions(inputs);

    Tensor<type, 2> targets = data_set_pointer->get_testing_target_data();

    // Neural network

    Tensor<type, 2> outputs;

    outputs = neural_network_pointer->calculate_outputs(inputs.data(), inputs_dimensions);

    Tensor<type, 1> errors(4);

    // Results

    const Tensor<type, 0> sum_squared_error = (outputs-targets).square().sum().sqrt();

    errors(0) = sum_squared_error(0);
    errors(1) = errors(0)/type(testing_samples_number);
    errors(2) = sqrt(errors(1));
    errors(3) = calculate_normalized_squared_error(targets, outputs);

    return errors;
}


/// Returns a vector containing the values of the errors between the outputs of the neural network
/// and the targets for a binary classification problem. The vector consists of:
/// <ul>
/// <li> Sum squared error.
/// <li> Mean squared error.
/// <li> Root mean squared error.
/// <li> Normalized squared error.
/// <li> Cross-entropy error.
/// <li> Weighted squared error.
/// </ul>

Tensor<type, 1> TestingAnalysis::calculate_binary_classification_testing_errors() const
{
    // Data set

#ifdef OPENNN_DEBUG

    check();

#endif

    const Index testing_samples_number = data_set_pointer->get_testing_samples_number();

#ifdef OPENNN_DEBUG

    ostringstream buffer;

    if(testing_samples_number == 0)
    {
        buffer << "OpenNN Exception: TestingAnalysis class.\n"
               << "Tensor<type, 1> calculate_binary_classification_testing_errors() const.\n"
               << "Number of testing samples is zero.\n";

        throw invalid_argument(buffer.str());
    }

#endif

    Tensor<type, 2> inputs = data_set_pointer->get_testing_input_data();

    Tensor<Index, 1> inputs_dimensions = get_dimensions(inputs);

    Tensor<type, 2> targets = data_set_pointer->get_testing_target_data();

    // Neural network

    Tensor<type, 2> outputs;

    outputs = neural_network_pointer->calculate_outputs(inputs.data(), inputs_dimensions);

    Tensor<type, 1> errors(6);

    // Results

    const Tensor<type, 0> sum_squared_error = (outputs-targets).square().sum().sqrt();

    errors(0) = sum_squared_error(0);
    errors(1) = errors(0)/type(testing_samples_number);
    errors(2) = sqrt(errors(1));
    errors(3) = calculate_normalized_squared_error(targets, outputs);
    errors(4) = calculate_cross_entropy_error(targets, outputs);
    errors(5) = calculate_weighted_squared_error(targets, outputs);

    return errors;
}


/// Returns a vector containing the values of the errors between the outputs of the neural network
/// and the targets for a multiple classification problem. The vector consists of:
/// <ul>
/// <li> Sum squared error.
/// <li> Mean squared error.
/// <li> Root mean squared error.
/// <li> Normalized squared error.
/// <li> Cross-entropy error.
/// </ul>

Tensor<type, 1> TestingAnalysis::calculate_multiple_classification_testing_errors() const
{
    // Data set

#ifdef OPENNN_DEBUG

    check();

#endif

    const Index testing_samples_number = data_set_pointer->get_testing_samples_number();

#ifdef OPENNN_DEBUG

    ostringstream buffer;

    if(testing_samples_number == 0)
    {
        buffer << "OpenNN Exception: TestingAnalysis class.\n"
               << "Tensor<type, 1> calculate_multiple_classification_testing_errors() const.\n"
               << "Number of testing samples is zero.\n";

        throw invalid_argument(buffer.str());
    }

#endif

    Tensor<type, 2> inputs = data_set_pointer->get_testing_input_data();

    Tensor<Index, 1> inputs_dimensions = get_dimensions(inputs);

    Tensor<type, 2> targets = data_set_pointer->get_testing_target_data();

    // Neural network

    Tensor<type, 2> outputs;

    outputs = neural_network_pointer->calculate_outputs(inputs.data(), inputs_dimensions);

    Tensor<type, 1> errors(4);

    // Results

    const Tensor<type, 0> sum_squared_error = (outputs-targets).square().sum().sqrt();

    errors(0) = sum_squared_error(0);
    errors(1) = errors(0)/type(testing_samples_number);
    errors(2) = sqrt(errors(1));
    errors(3) = calculate_normalized_squared_error(targets, outputs);

    return errors;
}


/// Returns the normalized squared error between the targets and the outputs of the neural network.
/// @param targets Testing target data.
/// @param outputs Testing output data.

type TestingAnalysis::calculate_normalized_squared_error(const Tensor<type, 2>& targets, const Tensor<type, 2>& outputs) const
{
    const Index samples_number = targets.dimension(0);

    const Tensor<type, 1> targets_mean = mean(targets);

    Tensor<type, 0> sum_squared_error = (outputs - targets).square().sum();

    type normalization_coefficient = type(0);

#pragma omp parallel for reduction(+: normalization_coefficient)

    for(Index i = 0; i < samples_number; i++)
    {
        const Tensor<type, 0> norm_1 = (targets.chip(i,0) - targets_mean).square().sum();

        normalization_coefficient += norm_1(0);
    }

    return sum_squared_error()/normalization_coefficient;
}


/// Returns the cross-entropy error between the targets and the outputs of the neural network.
/// It can only be computed for classification problems.
/// @param targets Testing target data.
/// @param outputs Testing output data.

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
            if(outputs_row(j) < type(NUMERIC_LIMITS_MIN))
            {
                outputs_row(j) = static_cast<type>(1.0e-6);
            }
            else if(static_cast<double>(outputs_row(j)) == 1.0)
            {
                outputs_row(j) = numeric_limits<type>::max();
            }

            cross_entropy_error -=
                    targets_row(j)*log(outputs_row(j)) + (static_cast<type>(1) - targets_row(j))*log(static_cast<type>(1) - outputs_row(j));
        }
    }

    return cross_entropy_error/static_cast<type>(testing_samples_number);
}


/// Returns the weighted squared error between the targets and the outputs of the neural network. It can only be computed for
/// binary classification problems.
/// @param targets Testing target data.
/// @param outputs Testing output data.
/// @param weights Weights of the postitives and negatives to evaluate.

type TestingAnalysis::calculate_weighted_squared_error(const Tensor<type, 2>& targets,
                                                       const Tensor<type, 2>& outputs,
                                                       const Tensor<type, 1>& weights) const
{
#ifdef OPENNN_DEBUG

    const Index outputs_number = outputs.dimension(1);

    ostringstream buffer;

    if(outputs_number != 1)
    {
        buffer << "OpenNN Exception: TestingAnalysis class.\n"
               << "type calculate_testing_weighted_squared_error(const Tensor<type, 2>&, const Tensor<type, 2>&, const Tensor<type, 1>&) const.\n"
               << "Number of outputs must be one.\n";

        throw invalid_argument(buffer.str());
    }

#endif

    type negatives_weight;
    type positives_weight;

    if(weights.size() != 2)
    {
        const Tensor<Index, 1> target_distribution = data_set_pointer->calculate_target_distribution();

        const Index negatives_number = target_distribution[0];
        const Index positives_number = target_distribution[1];

        negatives_weight = type(1);
        positives_weight = static_cast<type>(negatives_number/positives_number);
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

    f_1 = (targets - outputs).square() * positives_weight;

    f_2 = (targets - outputs).square()*negatives_weight;

    f_3 = targets.constant(type(0));

    Tensor<type, 0> sum_squared_error = (if_sentence.select(f_1, else_sentence.select(f_2, f_3))).sum();

    Index negatives = 0;

    Tensor<type, 1> target_column = targets.chip(0,1);

    for(Index i = 0; i < target_column.size(); i++)
    {
        if(static_cast<double>(target_column(i)) == 0.0) negatives++;
    }

    const type normalization_coefficient = type(negatives)*negatives_weight*static_cast<type>(0.5);

    return sum_squared_error(0)/normalization_coefficient;
}


type TestingAnalysis::calculate_Minkowski_error(const Tensor<type, 2>& targets, const Tensor<type, 2>& outputs, const type minkowski_parameter) const
{
    Tensor<type, 0> Minkoski_error = (outputs - targets).abs().pow(minkowski_parameter).sum().pow(static_cast<type>(1.0)/minkowski_parameter);

    return Minkoski_error();
}


type TestingAnalysis::calculate_determination_coefficient(const Tensor<type,1>& outputs, const Tensor<type,1>& targets) const
{
#ifdef OPENNN_DEBUG

    const Index outputs_number = outputs.dimension(0);

    ostringstream buffer;

    if(targets.size() != outputs_number)
    {
        buffer << "OpenNN Exception: TestingAnalysis class.\n"
               << "type calculate_determination_coefficient(const Tensor<type, 1>&, const Tensor<type, 1>&) const.\n"
               << "Outputs and targets dimensions must be equal.\n";

        throw invalid_argument(buffer.str());
    }

#endif

    type numerador = 0;
    const Tensor<type, 0> targets_mean = targets.mean();
    type  denominador = 0;

    // @todo Implementation with tensor operations
    for(Index i = 0; i < outputs.size(); i++)
    {
        numerador += (targets(i) - outputs(i))*(targets(i) - outputs(i));
        denominador += (targets(i) - targets_mean(0))*(targets(i) - targets_mean(0));
    }

    type determination_coefficient = type(1) - (numerador/denominador);

    return determination_coefficient;
};


/// Returns the confusion matrix for a binary classification problem.
/// @param targets Testing target data.
/// @param outputs Testing output data.
/// @param decision_threshold Decision threshold.

Tensor<Index, 2> TestingAnalysis::calculate_confusion_binary_classification(const Tensor<type, 2>& targets,
                                                                            const Tensor<type, 2>& outputs,
                                                                            const type& decision_threshold) const
{
    const Index testing_samples_number = targets.dimension(0);

    Tensor<Index, 2> confusion(2, 2);

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

        if(target > decision_threshold && output > decision_threshold)
        {
            true_positive++;
        }
        else if(target >= decision_threshold && output < decision_threshold)
        {
            false_negative++;
        }
        else if(target <= decision_threshold && output > decision_threshold)
        {
            false_positive++;
        }
        else if(target < decision_threshold && output < decision_threshold)
        {
            true_negative++;
        }
        else
        {
            ostringstream buffer;

            buffer << "OpenNN Exception: TestingAnalysis class.\n"
                   << "Tensor<Index, 2> calculate_confusion_binary_classification(const Tensor<type, 2>&, const Tensor<type, 2>&, const type&) const method.\n"
                   << "Unknown case.\n";

            throw invalid_argument(buffer.str());
        }
    }

    confusion(0,0) = true_positive;
    confusion(0,1) = false_negative;
    confusion(1,0) = false_positive;
    confusion(1,1) = true_negative;

    const Index confusion_sum = true_positive + false_negative + false_positive + true_negative;

    if(confusion_sum != testing_samples_number)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: TestingAnalysis class.\n"
               << "Tensor<Index, 2> calculate_confusion_binary_classification(const Tensor<type, 2>&, const Tensor<type, 2>&, const type&) const method.\n"
               << "Number of elements in confusion matrix (" << confusion_sum << ") must be equal to number of testing samples (" << testing_samples_number << ").\n";

        throw invalid_argument(buffer.str());
    }

    return confusion;
}


/// Returns the confusion matrix for a binary classification problem.
/// @param targets Testing target data.
/// @param outputs Testing output data.

Tensor<Index, 2> TestingAnalysis::calculate_confusion_multiple_classification(const Tensor<type, 2>& targets, const Tensor<type, 2>& outputs) const
{
    const Index samples_number = targets.dimension(0);
    const Index targets_number = targets.dimension(1);

    if(targets_number != outputs.dimension(1))
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: TestingAnalysis class.\n"
               << "Tensor<Index, 2> calculate_confusion_multiple_classification(const Tensor<type, 2>&, const Tensor<type, 2>&) const method.\n"
               << "Number of targets (" << targets_number << ") must be equal to number of outputs (" << outputs.dimension(1) << ").\n";

        throw invalid_argument(buffer.str());
    }

    Tensor<Index, 2> confusion(targets_number, targets_number);
    confusion.setZero();

    Index target_index = 0;
    Index output_index = 0;

    for(Index i = 0; i < samples_number; i++)
    {
        target_index = maximal_index(targets.chip(i, 0));
        output_index = maximal_index(outputs.chip(i, 0));

        confusion(target_index,output_index)++;
    }

    return confusion;
}


/// Returns a vector containing the number of total positives and the number of total negatives
/// samples of a data set.
/// The size of the vector is two and consists of:
/// <ul>
/// <li> Total positives.
/// <li> Total negatives.
/// </ul>

Tensor<Index, 1> TestingAnalysis::calculate_positives_negatives_rate(const Tensor<type, 2>& targets, const Tensor<type, 2>& outputs) const
{
    const Tensor<Index, 2> confusion = calculate_confusion_binary_classification(targets, outputs, type(0.5));
    Tensor<Index, 1> positives_negatives_rate(2);

    positives_negatives_rate[0] = confusion(0,0) + confusion(0,1);
    positives_negatives_rate[1] = confusion(1,0) + confusion(1,1);

    return positives_negatives_rate;
}


/// Returns the confusion matrix of a neural network on the testing samples of a data set.
/// If the number of outputs is one, the size of the confusion matrix is two.
/// If the number of outputs is greater than one, the size of the confusion matrix is the number of outputs.

Tensor<Index, 2> TestingAnalysis::calculate_confusion() const
{
    const Index outputs_number = neural_network_pointer->get_outputs_number();

#ifdef OPENNN_DEBUG

    check();

    if(!neural_network_pointer)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: TestingAnalysis class.\n"
               << "Tensor<Index, 2> calculate_confusion() const method.\n"
               << "Pointer to neural network in neural network is nullptr.\n";

        throw invalid_argument(buffer.str());
    }

    const Index inputs_number = neural_network_pointer->get_inputs_number();

    // Control sentence

    if(inputs_number != data_set_pointer->get_input_variables_number())
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: TestingAnalysis class." << endl
               << "Tensor<Index, 2> calculate_confusion() const method." << endl
               << "Number of inputs in neural network must be equal to number of inputs in data set." << endl;

        throw invalid_argument(buffer.str());
    }

    if(outputs_number != data_set_pointer->get_target_variables_number())
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: TestingAnalysis class." << endl
               << "Tensor<Index, 2> calculate_confusion() const method." << endl
               << "Number of outputs in neural network must be equal to number of targets in data set." << endl;

        throw invalid_argument(buffer.str());
    }

#endif

    Tensor<type, 2> inputs = data_set_pointer->get_testing_input_data();

    Tensor<Index, 1> inputs_dimensions = get_dimensions(inputs);

    Tensor<type, 2> targets = data_set_pointer->get_testing_target_data();

    // Neural network

    Tensor<type, 2> outputs;

    outputs = neural_network_pointer->calculate_outputs(inputs.data(), inputs_dimensions);

    if(outputs_number == 1)
    {
        type decision_threshold;

        if(neural_network_pointer->get_probabilistic_layer_pointer() != nullptr)
        {
            decision_threshold = neural_network_pointer->get_probabilistic_layer_pointer()->get_decision_threshold();
        }
        else
        {
            decision_threshold = type(0.5);
        }

        return calculate_confusion_binary_classification(targets, outputs, decision_threshold);
    }
    else
    {
        return calculate_confusion_multiple_classification(targets, outputs);
    }
}


/// Performs a ROC curve analysis.
/// It returns a ROC curve analysis results structure, which consists of:
/// <ul>
/// <li> ROC curve
/// <li> Area under the ROC curve
/// <li> Optimal threshold
/// </ul>

TestingAnalysis::RocAnalysisResults TestingAnalysis::perform_roc_analysis() const
{
#ifdef OPENNN_DEBUG

    check();

    if(!neural_network_pointer)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: TestingAnalysis class.\n"
               << "RocCurveResults perform_roc_analysis() const method.\n"
               << "Pointer to neural network in neural network is nullptr.\n";

        throw invalid_argument(buffer.str());
    }

    const Index inputs_number = neural_network_pointer->get_inputs_number();

    // Control sentence

    if(inputs_number != data_set_pointer->get_input_variables_number())
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: TestingAnalysis class." << endl
               << "RocCurveResults perform_roc_analysis() const method." << endl
               << "Number of inputs in neural network must be equal to number of inputs in data set." << endl;

        throw invalid_argument(buffer.str());
    }

    const Index outputs_number = neural_network_pointer->get_outputs_number();

    if(outputs_number != data_set_pointer->get_target_variables_number())
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: TestingAnalysis class." << endl
               << "RocCurveResults perform_roc_analysis() const method." << endl
               << "Number of outputs in neural network must be equal to number of targets in data set." << endl;

        throw invalid_argument(buffer.str());
    }

#endif

    Tensor<type, 2> inputs = data_set_pointer->get_testing_input_data();

    Tensor<Index, 1> inputs_dimensions = get_dimensions(inputs);

    Tensor<type, 2> targets = data_set_pointer->get_testing_target_data();

    Tensor<type, 2> outputs;

    outputs = neural_network_pointer->calculate_outputs(inputs.data(), inputs_dimensions);

    RocAnalysisResults roc_analysis_results;

    cout << "Calculating ROC curve..." << endl;

    roc_analysis_results.roc_curve = calculate_roc_curve(targets, outputs);

    cout << "Calculating area under curve..." << endl;

    roc_analysis_results.area_under_curve = calculate_area_under_curve(roc_analysis_results.roc_curve);

    cout << "Calculating confidence limits..." << endl;

    roc_analysis_results.confidence_limit = calculate_area_under_curve_confidence_limit(targets, outputs);

    cout << "Calculating optimal threshold..." << endl;

    roc_analysis_results.optimal_threshold = calculate_optimal_threshold(roc_analysis_results.roc_curve);

    return roc_analysis_results;
}


/// Calculates the Wilcoxon parameter, which is used for calculating the area under a ROC curve.
/// Returns 1 if first value is greater than second one, 0 if second value is greater than first one or 0.5 in other case.
/// @param x Target data value.
/// @param y Target data value.

type TestingAnalysis::calculate_Wilcoxon_parameter(const type& x, const type& y) const
{
    if(x > y)
    {
        return type(1);
    }
    else if(x < y)
    {
        return type(0);
    }
    else
    {
        return type(0.5);
    }
}


/// Returns a matrix with the values of a ROC curve for a binary classification problem.
/// The number of columns is three. The third column contains the decision threshold.
/// The number of rows is one more than the number of outputs if the number of outputs is lower than 100
/// or 50 in other case.
/// @param targets Testing target data.
/// @param outputs Testing output data.

Tensor<type, 2> TestingAnalysis::calculate_roc_curve(const Tensor<type, 2>& targets, const Tensor<type, 2>& outputs) const
{
    const Tensor<Index, 1> positives_negatives_rate = calculate_positives_negatives_rate(targets, outputs);

    const Index total_positives = positives_negatives_rate(0);
    const Index total_negatives = positives_negatives_rate(1);

    if(total_positives == 0)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: TestingAnalysis class.\n"
               << "Tensor<type, 2> calculate_roc_curve(const Tensor<type, 2>&, const Tensor<type, 2>&) const.\n"
               << "Number of positive samples ("<< total_positives <<") must be greater than zero.\n";

        throw invalid_argument(buffer.str());
    }

    if(total_negatives == 0)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: TestingAnalysis class.\n"
               << "Tensor<type, 2> calculate_roc_curve(const Tensor<type, 2>&, const Tensor<type, 2>&) const.\n"
               << "Number of negative samples ("<< total_negatives <<") must be greater than zero.\n";

        throw invalid_argument(buffer.str());
    }

    const Index maximum_points_number = 200;

    Index points_number;

    points_number = maximum_points_number;

   if(targets.dimension(1) != 1)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: TestingAnalysis class.\n"
               << "Tensor<type, 2> calculate_roc_curve(const Tensor<type, 2>&, const Tensor<type, 2>&) const.\n"
               << "Number of of target variables ("<< targets.dimension(1) <<") must be one.\n";

        throw invalid_argument(buffer.str());
    }

    if(outputs.dimension(1) != 1)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: TestingAnalysis class.\n"
               << "Tensor<type, 2> calculate_roc_curve(const Tensor<type, 2>&, const Tensor<type, 2>&) const.\n"
               << "Number of of output variables ("<< targets.dimension(1) <<") must be one.\n";

        throw invalid_argument(buffer.str());
    }

    // Sort by ascending values of outputs vector

    Tensor<Index, 1> sorted_indices(outputs.dimension(0));
    iota(sorted_indices.data(), sorted_indices.data() + sorted_indices.size(), 0);

    stable_sort(sorted_indices.data(), sorted_indices.data()+sorted_indices.size(), [outputs](Index i1, Index i2) {return outputs(i1,0) < outputs(i2,0);});

    Tensor<type, 2> roc_curve(points_number + 1, 3);
    roc_curve.setZero();

#pragma omp parallel for schedule(dynamic)

    for(Index i = 1; i < static_cast<Index>(points_number); i++)
    {
        const type threshold = i * (1/static_cast<type>(points_number));

        Index true_positive = 0;
        Index false_negative = 0;
        Index false_positive = 0;
        Index true_negative = 0;

        type target;
        type output;

        for(Index j = 0; j <= targets.size(); j++)
        {
            target = targets(j,0);
            output = outputs(j,0);

            if(target > threshold && output > threshold)
            {
                true_positive++;
            }
            else if(target >= threshold && output < threshold)
            {
                false_negative++;
            }
            else if(target <= threshold && output > threshold)
            {
                false_positive++;
            }
            else if(target < threshold && output < threshold)
            {
                true_negative++;
            }
        }

        roc_curve(i,0) = 1 - static_cast<type>(true_positive)/(static_cast<type>(true_positive + false_negative));
        roc_curve(i,1) = static_cast<type>(true_negative)/(static_cast<type>(true_negative + false_positive));
        roc_curve(i,2) = static_cast<type>(threshold);

        if(  isnan(roc_curve(i,0)) )
        {
            roc_curve(i,0) = 1;
        }
        if( isnan(roc_curve(i,1)))
        {
            roc_curve(i,1) = 0;
        }
    }

    roc_curve(0,0) = 0;
    roc_curve(0,1) = 0;
    roc_curve(0,2) = 0;

    roc_curve(points_number,0) = 1;
    roc_curve(points_number,1) = 1;
    roc_curve(points_number,2) = 1;

    return roc_curve;
}

/*
/// Returns the area under a ROC curve using Wilcoxon parameter test.
/// @param targets Testing target data.
/// @param outputs Testing output data.

type TestingAnalysis::calculate_area_under_curve(const Tensor<type, 2>& targets, const Tensor<type, 2>& outputs) const
{
    const Tensor<Index, 1> positives_negatives_rate = calculate_positives_negatives_rate(targets, outputs);

    const Index total_positives = positives_negatives_rate[0];
    const Index total_negatives = positives_negatives_rate[1];

    if(total_positives == 0)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: TestingAnalysis class.\n"
               << "Tensor<type, 2> calculate_roc_curve(const Tensor<type, 2>&, const Tensor<type, 2>&) const.\n"
               << "Number of positive samples("<< total_positives <<") must be greater than zero.\n";

        throw invalid_argument(buffer.str());
    }

    if(total_negatives == 0)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: TestingAnalysis class.\n"
               << "Tensor<type, 2> calculate_roc_curve(const Tensor<type, 2>&, const Tensor<type, 2>&) const.\n"
               << "Number of negative samples("<< total_negatives <<") must be greater than zero.\n";

        throw invalid_argument(buffer.str());
    }

    Index testing_samples_number = targets.dimension(0);

    long double sum = 0.0;

    type area_under_curve;

    #pragma omp parallel for reduction(+ : sum) schedule(dynamic)

    for(Index i = 0; i < testing_samples_number; i++)
    {
        if(abs(targets(i,0) - static_cast<type>(1.0)) < type(NUMERIC_LIMITS_MIN))
        {
            for(Index j = 0; j < testing_samples_number; j++)
            {
                if(abs(targets(j,0)) < type(NUMERIC_LIMITS_MIN))
                {
                    sum += calculate_Wilcoxon_parameter(outputs(i,0),outputs(j,0));
                }
            }
        }
    }

    area_under_curve = static_cast<type>(sum)/static_cast<type>(total_positives*total_negatives);

    return area_under_curve;
}
*/

/// Returns the area under a ROC curve using trapezoidal integration.
/// @param roc_curve ROC curve.

type TestingAnalysis::calculate_area_under_curve(const Tensor<type, 2>& roc_curve) const
{
    type area_under_curve = type(0);

    for(Index i = 1; i < roc_curve.dimension(0); i++)
    {
        area_under_curve += (roc_curve(i,0)-roc_curve(i-1,0))*(roc_curve(i,1)+roc_curve(i-1,1));
    }

    return area_under_curve/ type(2);
}


/// Returns the confidence limit for the area under a roc curve.
/// @param targets Testing target data.
/// @param outputs Testing output data.

type TestingAnalysis::calculate_area_under_curve_confidence_limit(const Tensor<type, 2>& targets, const Tensor<type, 2>& outputs) const
{
    const Tensor<Index, 1> positives_negatives_rate = calculate_positives_negatives_rate(targets, outputs);

    const Index total_positives = positives_negatives_rate[0];
    const Index total_negatives = positives_negatives_rate[1];

    if(total_positives == 0)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: TestingAnalysis class.\n"
               << "Tensor<type, 2> calculate_roc_curve_confidence_limit(const Tensor<type, 2>&, const Tensor<type, 2>&) const.\n"
               << "Number of positive samples("<< total_positives <<") must be greater than zero.\n";

        throw invalid_argument(buffer.str());
    }

    if(total_negatives == 0)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: TestingAnalysis class.\n"
               << "Tensor<type, 2> calculate_roc_curve_confidence_limit(const Tensor<type, 2>&, const Tensor<type, 2>&) const.\n"
               << "Number of negative samples("<< total_negatives <<") must be greater than zero.\n";

        throw invalid_argument(buffer.str());
    }

    const Tensor<type, 2> roc_curve = calculate_roc_curve(targets, outputs);

    const type area_under_curve = calculate_area_under_curve(roc_curve);

    const type Q_1 = area_under_curve/(static_cast<type>(2.0) - area_under_curve);
    const type Q_2 = (static_cast<type>(2.0) *area_under_curve*area_under_curve)/(static_cast<type>(1.0) *area_under_curve);

    const type confidence_limit = type(type(1.64485)*sqrt((area_under_curve*(type(1) - area_under_curve)
                                  + (type(total_positives) - type(1))*(Q_1-area_under_curve*area_under_curve)
                                  + (type(total_negatives) - type(1))*(Q_2-area_under_curve*area_under_curve))/(type(total_positives*total_negatives))));

    return confidence_limit;
}


/// Returns the point of optimal classification accuracy, which is the nearest ROC curve point to the upper left corner(0,1).
/// @param targets Testing target data.
/// @param outputs Testing output data.
/// @param roc_curve ROC curve.

type TestingAnalysis::calculate_optimal_threshold(const Tensor<type, 2>& roc_curve) const
{
    const Index points_number = roc_curve.dimension(0);

    type optimal_threshold = type(0.5);

    type minimun_distance = numeric_limits<type>::max();

    type distance;

    for(Index i = 0; i < points_number; i++)
    {
        distance = sqrt(roc_curve(i,0)*roc_curve(i,0) + (roc_curve(i,1) - static_cast<type>(1))*(roc_curve(i,1) - static_cast<type>(1)));

        if(distance < minimun_distance)
        {
            optimal_threshold = roc_curve(i,2);

            minimun_distance = distance;
        }
    }

    return optimal_threshold;
}


/// Performs a cumulative gain analysis.
/// Returns a matrix with the values of a cumulative gain chart.

Tensor<type, 2> TestingAnalysis::perform_cumulative_gain_analysis() const
{
#ifdef OPENNN_DEBUG

    check();

#endif

#ifdef OPENNN_DEBUG

    if(!neural_network_pointer)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: TestingAnalysis class.\n"
               << "Tensor<type, 2> perform_cumulative_gain_analysis() const method.\n"
               << "Pointer to neural network in neural network is nullptr.\n";

        throw invalid_argument(buffer.str());
    }

#endif

#ifdef OPENNN_DEBUG

    const Index inputs_number = neural_network_pointer->get_inputs_number();

    // Control sentence

    if(inputs_number != data_set_pointer->get_input_variables_number())
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: TestingAnalysis class." << endl
               << "Tensor<type, 2> perform_cumulative_gain_analysis() const method." << endl
               << "Number of inputs in neural network must be equal to number of inputs in data set." << endl;

        throw invalid_argument(buffer.str());
    }

    const Index outputs_number = neural_network_pointer->get_outputs_number();

    if(outputs_number != data_set_pointer->get_target_variables_number())
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: TestingAnalysis class." << endl
               << "Tensor<type, 2> perform_cumulative_gain_analysis() const method." << endl
               << "Number of outputs in neural network must be equal to number of targets in data set." << endl;

        throw invalid_argument(buffer.str());
    }

#endif

    Tensor<type, 2> inputs = data_set_pointer->get_testing_input_data();

    Tensor<Index, 1> inputs_dimensions = get_dimensions(inputs);

    Tensor<type, 2> targets = data_set_pointer->get_testing_target_data();

    Tensor<type, 2> outputs;

    outputs = neural_network_pointer->calculate_outputs(inputs.data(), inputs_dimensions);


    const Tensor<type, 2> cumulative_gain = calculate_cumulative_gain(targets, outputs);

    return cumulative_gain;
}


/// Returns a matrix with the values of a cumulative gain chart.
/// The number of columns is two, the number of rows is 20.
/// @param targets Testing target data.
/// @param outputs Testing output data.

Tensor<type, 2> TestingAnalysis::calculate_cumulative_gain(const Tensor<type, 2>& targets, const Tensor<type, 2>& outputs) const
{
    const Index total_positives = calculate_positives_negatives_rate(targets, outputs)[0];

    if(total_positives == 0)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: TestingAnalysis class.\n"
               << "Tensor<type, 2> calculate_cumulative_gain(const Tensor<type, 2>&, const Tensor<type, 2>&) const.\n"
               << "Number of positive samples(" << total_positives << ") must be greater than zero.\n";

        throw invalid_argument(buffer.str());
    }

    const Index testing_samples_number = targets.dimension(0);

    // Sort by ascending values of outputs vector

    Tensor<Index, 1> sorted_indices(outputs.dimension(0));
    iota(sorted_indices.data(), sorted_indices.data() + sorted_indices.size(), 0);

    stable_sort(sorted_indices.data(),
                sorted_indices.data()+sorted_indices.size(),
                [outputs](Index i1, Index i2) {return outputs(i1,0) > outputs(i2,0);});

    Tensor<type, 1> sorted_targets(testing_samples_number);

    for(Index i = 0; i < testing_samples_number; i++)
    {
        sorted_targets(i) = targets(sorted_indices(i),0);
    }

    const Index points_number = 21;
    const type percentage_increment = static_cast<type>(0.05);

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

        maximum_index = static_cast<Index>(percentage* type(testing_samples_number));

        for(Index j = 0; j < maximum_index; j++)
        {
            if(static_cast<double>(sorted_targets(j)) == 1.0)
            {
                 positives++;
            }
        }

        cumulative_gain(i + 1, 0) = percentage;
        cumulative_gain(i + 1, 1) = static_cast<type>(positives)/static_cast<type>(total_positives);
    }

    return cumulative_gain;
}


/// Returns a matrix with the values of a cumulative gain chart for the negative samples.
/// The number of columns is two, the number of rows is 20.
/// @param targets Testing target data.
/// @param outputs Testing output data.

Tensor<type, 2> TestingAnalysis::calculate_negative_cumulative_gain(const Tensor<type, 2>& targets, const Tensor<type, 2>& outputs) const
{
    const Index total_negatives = calculate_positives_negatives_rate(targets, outputs)[1];

    if(total_negatives == 0)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: TestingAnalysis class.\n"
               << "Tensor<type, 2> calculate_negative_cumulative_gain(const Tensor<type, 2>&, const Tensor<type, 2>&) const.\n"
               << "Number of negative samples(" << total_negatives << ") must be greater than zero.\n";

        throw invalid_argument(buffer.str());
    }

    const Index testing_samples_number = targets.dimension(0);

    // Sort by ascending values of outputs vector

    Tensor<Index, 1> sorted_indices(outputs.dimension(0));
    iota(sorted_indices.data(), sorted_indices.data() + sorted_indices.size(), 0);

    stable_sort(sorted_indices.data(), sorted_indices.data()+sorted_indices.size(), [outputs](Index i1, Index i2) {return outputs(i1,0) > outputs(i2,0);});

    Tensor<type, 1> sorted_targets(testing_samples_number);

    for(Index i = 0; i < testing_samples_number; i++)
    {
        sorted_targets(i) = targets(sorted_indices(i),0);
    }

    const Index points_number = 21;
    const type percentage_increment = static_cast<type>(0.05);

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

        maximum_index = static_cast<Index>(percentage* type(testing_samples_number));

        for(Index j = 0; j < maximum_index; j++)
        {
            if(sorted_targets(j) < type(NUMERIC_LIMITS_MIN))
            {
                 negatives++;
            }
        }

        negative_cumulative_gain(i + 1, 0) = percentage;

        negative_cumulative_gain(i + 1, 1) = static_cast<type>(negatives)/static_cast<type>(total_negatives);
    }

    return negative_cumulative_gain;
}


/// Performs a lift chart analysis.
/// Returns a matrix with the values of a lift chart.

Tensor<type, 2> TestingAnalysis::perform_lift_chart_analysis() const
{
#ifdef OPENNN_DEBUG

    check();

    if(!neural_network_pointer)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: TestingAnalysis class.\n"
               << "Tensor<type, 2> perform_lift_chart_analysis() const method.\n"
               << "Pointer to neural network in neural network is nullptr.\n";

        throw invalid_argument(buffer.str());
    }

    const Index inputs_number = neural_network_pointer->get_inputs_number();

    // Control sentence

    if(inputs_number != data_set_pointer->get_input_variables_number())
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: TestingAnalysis class." << endl
               << "Tensor<type, 2> perform_lift_chart_analysis() const method." << endl
               << "Number of inputs in neural network must be equal to number of inputs in data set." << endl;

        throw invalid_argument(buffer.str());
    }

    const Index outputs_number = neural_network_pointer->get_outputs_number();

    if(outputs_number != data_set_pointer->get_target_variables_number())
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: TestingAnalysis class." << endl
               << "Tensor<type, 2> perform_lift_chart_analysis() const method." << endl
               << "Number of outputs in neural network must be equal to number of targets in data set." << endl;

        throw invalid_argument(buffer.str());
    }

#endif

    Tensor<type, 2> inputs = data_set_pointer->get_testing_input_data();

    Tensor<Index, 1> inputs_dimensions = get_dimensions(inputs);

    Tensor<type, 2> targets = data_set_pointer->get_testing_target_data();

    Tensor<type, 2> outputs;

    outputs = neural_network_pointer->calculate_outputs(inputs.data(), inputs_dimensions);

    const Tensor<type, 2> cumulative_gain = calculate_cumulative_gain(targets, outputs);
    const Tensor<type, 2> lift_chart = calculate_lift_chart(cumulative_gain);

    return lift_chart;
}


/// Returns a matrix with the values of lift chart for a given cumulative gain chart.
/// Size is the same as the cumulative lift chart one.
/// @param cumulative_gain A cumulative gain chart.

Tensor<type, 2> TestingAnalysis::calculate_lift_chart(const Tensor<type, 2>& cumulative_gain) const
{
    const Index rows_number = cumulative_gain.dimension(0);
    const Index columns_number = cumulative_gain.dimension(1);

    Tensor<type, 2> lift_chart(rows_number, columns_number);

    lift_chart(0,0) = type(0);
    lift_chart(0,1) = type(1);

    #pragma omp parallel for

    for(Index i = 1; i < rows_number; i++)
    {
        lift_chart(i, 0) = static_cast<type>(cumulative_gain(i, 0));
        lift_chart(i, 1) = static_cast<type>(cumulative_gain(i, 1))/static_cast<type>(cumulative_gain(i, 0));
    }

    return lift_chart;
}


/// Performs a Kolmogorov-Smirnov analysis, which consists of the cumulative gain for the positive samples and the cumulative
/// gain for the negative samples.
/// It returns a Kolmogorov-Smirnov results structure, which consists of:
/// <ul>
/// <li> Positive cumulative gain
/// <li> Negative cumulative gain
/// <li> Maximum gain
/// </ul>

TestingAnalysis::KolmogorovSmirnovResults TestingAnalysis::perform_Kolmogorov_Smirnov_analysis() const
{
#ifdef OPENNN_DEBUG

    check();

    if(!neural_network_pointer)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: TestingAnalysis class.\n"
               << "Tensor<type, 2> perform_Kolmogorov_Smirnov_analysis() const method.\n"
               << "Pointer to neural network in neural network is nullptr.\n";

        throw invalid_argument(buffer.str());
    }

#endif

#ifdef OPENNN_DEBUG

    const Index inputs_number = neural_network_pointer->get_inputs_number();

    // Control sentence

    if(inputs_number != data_set_pointer->get_input_variables_number())
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: TestingAnalysis class." << endl
               << "Tensor<type, 2> perform_Kolmogorov_Smirnov_analysis() const method." << endl
               << "Number of inputs in neural network must be equal to number of inputs in data set." << endl;

        throw invalid_argument(buffer.str());
    }

    const Index outputs_number = neural_network_pointer->get_outputs_number();

    if(outputs_number != data_set_pointer->get_target_variables_number())
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: TestingAnalysis class." << endl
               << "Tensor<type, 2> perform_Kolmogorov_Smirnov_analysis() const method." << endl
               << "Number of outputs in neural network must be equal to number of targets in data set." << endl;

        throw invalid_argument(buffer.str());
    }

#endif

    Tensor<type, 2> inputs = data_set_pointer->get_testing_input_data();

    Tensor<Index, 1> inputs_dimensions = get_dimensions(inputs);

    Tensor<type, 2> targets = data_set_pointer->get_testing_target_data();

    Tensor<type, 2> outputs;

    outputs = neural_network_pointer->calculate_outputs(inputs.data(), inputs_dimensions);

    TestingAnalysis::KolmogorovSmirnovResults Kolmogorov_Smirnov_results;

    Kolmogorov_Smirnov_results.positive_cumulative_gain = calculate_cumulative_gain(targets, outputs);
    Kolmogorov_Smirnov_results.negative_cumulative_gain = calculate_negative_cumulative_gain(targets, outputs);
    Kolmogorov_Smirnov_results.maximum_gain =
        calculate_maximum_gain(Kolmogorov_Smirnov_results.positive_cumulative_gain,Kolmogorov_Smirnov_results.negative_cumulative_gain);

    return Kolmogorov_Smirnov_results;
}


/// Returns the score of the the maximum gain, which is the point of major separation between the positive and
/// the negative cumulative gain charts, and the samples ratio for which it occurs.
/// @param positive_cumulative_gain Cumulative gain fo the positive samples.
/// @param negative_cumulative_gain Cumulative gain fo the negative samples.

Tensor<type, 1> TestingAnalysis::calculate_maximum_gain(const Tensor<type, 2>& positive_cumulative_gain, const Tensor<type, 2>& negative_cumulative_gain) const
{
    const Index points_number = positive_cumulative_gain.dimension(0);

#ifdef OPENNN_DEBUG

    if(points_number != negative_cumulative_gain.dimension(0))
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: TestingAnalysis class.\n"
               << "Tensor<type, 2> calculate_maximum_gain() const method.\n"
               << "Positive and negative cumulative gain matrix must have the same rows number.\n";

        throw invalid_argument(buffer.str());
    }

#endif

    Tensor<type, 1> maximum_gain(2);

    const type percentage_increment = static_cast<type>(0.05);

    type percentage = type(0);

    for(Index i = 0; i < points_number - 1; i++)
    {
        percentage += percentage_increment;

        if(positive_cumulative_gain(i+1,1)-negative_cumulative_gain(i+1,1) > maximum_gain[1]
                && positive_cumulative_gain(i+1,1)-negative_cumulative_gain(i+1,1) > static_cast<type>(0.0))
        {
            maximum_gain(1) = positive_cumulative_gain(i+1,1)-negative_cumulative_gain(i+1,1);
            maximum_gain(0) = percentage;
        }
    }

    return maximum_gain;
}


/// Performs a calibration plot analysis.

Tensor<type, 2> TestingAnalysis::perform_calibration_plot_analysis() const
{
#ifdef OPENNN_DEBUG

    check();

    if(!neural_network_pointer)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: TestingAnalysis class.\n"
               << "Tensor<type, 2> perform_calibration_plot_analysis() const method.\n"
               << "Pointer to neural network in neural network is nullptr.\n";

        throw invalid_argument(buffer.str());
    }

    const Index inputs_number = neural_network_pointer->get_inputs_number();

    // Control sentence

    if(inputs_number != data_set_pointer->get_input_variables_number())
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: TestingAnalysis class." << endl
               << "Tensor<type, 2> perform_calibration_plot_analysis() const method." << endl
               << "Number of inputs in neural network must be equal to number of inputs in data set." << endl;

        throw invalid_argument(buffer.str());
    }

    const Index outputs_number = neural_network_pointer->get_outputs_number();

    if(outputs_number != data_set_pointer->get_target_variables_number())
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: TestingAnalysis class." << endl
               << "Tensor<type, 2> perform_calibration_plot_analysis() const method." << endl
               << "Number of outputs in neural network must be equal to number of targets in data set." << endl;

        throw invalid_argument(buffer.str());
    }

#endif

    Tensor<type, 2> inputs = data_set_pointer->get_testing_input_data();

    Tensor<Index, 1> inputs_dimensions = get_dimensions(inputs);

    Tensor<type, 2> targets = data_set_pointer->get_testing_target_data();

    Tensor<type, 2> outputs;

    outputs = neural_network_pointer->calculate_outputs(inputs.data(), inputs_dimensions);

    const Tensor<type, 2> calibration_plot = calculate_calibration_plot(targets, outputs);

    return calibration_plot;

}


/// Returns a matix with the values of a calibration plot.
/// Number of columns is two. Number of rows varies depending on the distribution of positive targets.
/// @param targets Testing target data.
/// @param outputs Testing output data.

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

        probability += static_cast<type>(0.1);

        for(Index j = 0; j < rows_number; j++)
        {
            if(outputs(j, 0) >= (probability - static_cast<type>(0.1)) && outputs(j, 0) < probability)
            {
                count++;

                sum += outputs(j, 0);

                if(static_cast<Index>(targets(j, 0)) == 1)
                {
                    positives++;
                }
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

    while(contains(calibration_plot.chip(0,1), type(-1)))
     {
         for(Index i = 1; i < (points_number - points_number_subtracted + 1); i++)
         {
             if(abs(calibration_plot(i, 0) + type(1)) < type(NUMERIC_LIMITS_MIN))
             {
                 calibration_plot = delete_row(calibration_plot, i);

                 points_number_subtracted++;
             }
         }
     }

    return calibration_plot;
}


/// Returns the histogram of the outputs.
/// @param outputs Testing output data.
/// @param bins_number Number of bins of the histogram.

Tensor<Histogram, 1> TestingAnalysis::calculate_output_histogram(const Tensor<type, 2>& outputs, const Index& bins_number) const
{
    Tensor<Histogram, 1> output_histogram(1);

    Tensor<type, 1> output_column = outputs.chip(0,1);

    output_histogram (0) = histogram(output_column, bins_number);

    return output_histogram;
}


/// Returns a structure with the binary classification rates, which has the indices of the true positive, false negative, false positive and true negative samples.
/// <ul>
/// <li> True positive samples
/// <li> False positive samples
/// <li> False negative samples
/// <li> True negative samples
/// </ul>

TestingAnalysis::BinaryClassificationRates TestingAnalysis::calculate_binary_classification_rates() const
{
#ifdef OPENNN_DEBUG

    check();

    if(!neural_network_pointer)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: TestingAnalysis class.\n"
               << "BinaryClassificationRates calculate_binary_classification_rates() const method.\n"
               << "Pointer to neural network in neural network is nullptr.\n";

        throw invalid_argument(buffer.str());
    }

    const Index inputs_number = neural_network_pointer->get_inputs_number();

    // Control sentence

    if(inputs_number != data_set_pointer->get_input_variables_number())
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: TestingAnalysis class." << endl
               << "BinaryClassificationRates calculate_binary_classification_rates() const method." << endl
               << "Number of inputs in neural network must be equal to number of inputs in data set." << endl;

        throw invalid_argument(buffer.str());
    }

    const Index outputs_number = neural_network_pointer->get_outputs_number();

    if(outputs_number != data_set_pointer->get_target_variables_number())
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: TestingAnalysis class." << endl
               << "BinaryClassificationRates calculate_binary_classification_rates() const method." << endl
               << "Number of outputs in neural network must be equal to number of targets in data set." << endl;

        throw invalid_argument(buffer.str());
    }

#endif

    Tensor<type, 2> inputs = data_set_pointer->get_testing_input_data();

    Tensor<Index, 1> inputs_dimensions = get_dimensions(inputs);

    Tensor<type, 2> targets = data_set_pointer->get_testing_target_data();

    Tensor<type, 2> outputs;

    outputs = neural_network_pointer->calculate_outputs(inputs.data(), inputs_dimensions);

    const Tensor<Index, 1> testing_indices = data_set_pointer->get_testing_samples_indices();

    type decision_threshold;

    if(neural_network_pointer->get_probabilistic_layer_pointer() != nullptr)
    {
        decision_threshold = neural_network_pointer->get_probabilistic_layer_pointer()->get_decision_threshold();
    }
    else
    {
        decision_threshold = type(0.5);
    }

    BinaryClassificationRates binary_classification_rates;

    binary_classification_rates.true_positives_indices = calculate_true_positive_samples(targets, outputs, testing_indices, decision_threshold);
    binary_classification_rates.false_positives_indices = calculate_false_positive_samples(targets, outputs, testing_indices, decision_threshold);
    binary_classification_rates.false_negatives_indices = calculate_false_negative_samples(targets, outputs, testing_indices, decision_threshold);
    binary_classification_rates.true_negatives_indices = calculate_true_negative_samples(targets, outputs, testing_indices, decision_threshold);

    return binary_classification_rates;
}


/// Returns a vector with the indices of the true positive samples.
/// The size of the vector is the number of true positive samples.
/// @param targets Testing target data.
/// @param outputs Testing outputs.
/// @param testing_indices Indices of testing data.
/// @param decision_threshold Decision threshold.

Tensor<Index, 1> TestingAnalysis::calculate_true_positive_samples(const Tensor<type, 2>& targets, const Tensor<type, 2>& outputs,
        const Tensor<Index, 1>& testing_indices, const type& decision_threshold) const
{
    const Index rows_number = targets.dimension(0);

    Tensor<Index, 1> true_positives_indices_copy(rows_number);

    Index index = 0;

    for(Index i = 0; i < rows_number; i++)
    {
        Tensor<Index, 1> copy;
        if(targets(i,0) >= decision_threshold && outputs(i,0) >= decision_threshold)
        {
            true_positives_indices_copy(index) = testing_indices(i);
            index++;
        }
    }

    Tensor<Index, 1> true_positives_indices(index);

    copy( true_positives_indices_copy.data(),
          true_positives_indices_copy.data() + index,
          true_positives_indices.data());

    return true_positives_indices;
}


/// Returns a vector with the indices of the false positive samples.
/// The size of the vector is the number of false positive samples.
/// @param targets Testing target data.
/// @param outputs Testing output data.
/// @param testing_indices Indices of the testing data
/// @param decision_threshold Decision threshold.

Tensor<Index, 1> TestingAnalysis::calculate_false_positive_samples(const Tensor<type, 2>& targets, const Tensor<type, 2>& outputs,
        const Tensor<Index, 1>& testing_indices, const type& decision_threshold) const
{
    const Index rows_number = targets.dimension(0);

    Tensor<Index, 1> false_positives_indices_copy(rows_number);

    Index index = 0;

    for(Index i = 0; i < rows_number; i++)
    {
        if(targets(i,0) < decision_threshold && outputs(i,0) >= decision_threshold)
        {
            false_positives_indices_copy(index) = testing_indices(i);
            index++;
        }
    }

    Tensor<Index, 1> false_positives_indices(index);

    copy(false_positives_indices_copy.data(),
         false_positives_indices_copy.data() + index,
         false_positives_indices.data());

    return false_positives_indices;
}


/// Returns a vector with the indices of the false negative samples.
/// The size of the vector is the number of false negative samples.
/// @param targets Testing target data.
/// @param outputs Testing output data.
/// @param testing_indices Indices of the testing data
/// @param decision_threshold Decision threshold.

Tensor<Index, 1> TestingAnalysis::calculate_false_negative_samples(const Tensor<type, 2>& targets, const Tensor<type, 2>& outputs,
        const Tensor<Index, 1>& testing_indices, const type& decision_threshold) const
{
    const Index rows_number = targets.dimension(0);

    Tensor<Index, 1> false_negatives_indices_copy(rows_number);

    Index index = 0;

    for(Index i = 0; i < rows_number; i++)
    {
        if(targets(i,0) > decision_threshold && outputs(i,0) < decision_threshold)
        {
            false_negatives_indices_copy(index) = testing_indices(i);
            index++;
        }
    }

    Tensor<Index, 1> false_negatives_indices(index);

    copy(false_negatives_indices_copy.data(),
         false_negatives_indices_copy.data() + index,
         false_negatives_indices.data());

    return false_negatives_indices;
}


/// Returns a vector with the indices of the true negative samples.
/// The size of the vector is the number of true negative samples.
/// @param targets Testing target data.
/// @param outputs Testinga output data.
/// @param testing_indices Indices of the testing data
/// @param decision_threshold Decision threshold.

Tensor<Index, 1> TestingAnalysis::calculate_true_negative_samples(const Tensor<type, 2>& targets, const Tensor<type, 2>& outputs,
        const Tensor<Index, 1>& testing_indices, const type& decision_threshold) const
{
    const Index rows_number = targets.dimension(0);

    Tensor<Index, 1> true_negatives_indices_copy(rows_number);

    Index index = 0;

    for(Index i = 0; i < rows_number; i++)
    {
        if(targets(i,0) < decision_threshold && outputs(i,0) < decision_threshold)
        {
            true_negatives_indices_copy(index) = testing_indices(i);
            index++;
        }
    }

    Tensor<Index, 1> true_negatives_indices(index);

    copy(true_negatives_indices_copy.data(),
         true_negatives_indices_copy.data() + index,
         true_negatives_indices.data());

    return true_negatives_indices;
}


Tensor<type, 1> TestingAnalysis::calculate_multiple_classification_precision() const
{
    Tensor<type, 1> multiple_classification_tests(2);

    Tensor<type, 2> inputs = data_set_pointer->get_testing_input_data();

    Tensor<Index, 1> inputs_dimensions = get_dimensions(inputs);

    Tensor<type, 2> targets = data_set_pointer->get_testing_target_data();

    Tensor<type, 2> outputs;

    outputs = neural_network_pointer->calculate_outputs(inputs.data(), inputs_dimensions);

    const Tensor<Index, 2> confusion_matrix = calculate_confusion_multiple_classification(targets, outputs);

    type diagonal_sum = type(0);
    type off_diagonal_sum = type(0);
    const Tensor<Index, 0> total_sum = confusion_matrix.sum();

    for(Index i = 0; i < confusion_matrix.dimension(0); i++)
    {
        for(Index j = 0; j < confusion_matrix.dimension(1); j++)
        {
            i == j ? diagonal_sum += static_cast<type>(confusion_matrix(i,j)) : off_diagonal_sum += static_cast<type>(confusion_matrix(i,j));
        }
    }

    multiple_classification_tests(0) = diagonal_sum/static_cast<type>(total_sum());
    multiple_classification_tests(1) = off_diagonal_sum/static_cast<type>(total_sum());

    return multiple_classification_tests;
}


void TestingAnalysis::save_confusion(const string& confusion_file_name) const
{
    const Tensor<Index, 2> confusion = calculate_confusion();

    const Index columns_number = confusion.dimension(0);

    std::ofstream confusion_file(confusion_file_name);

    Tensor<string, 1> target_variable_names = data_set_pointer->get_target_variables_names();

    confusion_file << ",";

    for(Index i = 0; i < confusion.dimension(0); i++)
    {
        confusion_file << target_variable_names(i);

        if(i != target_variable_names.dimension(0) -1)
        {
            confusion_file << ",";
        }
    }

    confusion_file << endl;

    for(Index i = 0; i < columns_number; i++)
    {
        confusion_file << target_variable_names(i) << ",";

        for(Index j = 0; j < columns_number; j++)
        {
            if(j == columns_number - 1)
            {
                confusion_file << confusion(i,j) << endl;
            }
            else
            {
                confusion_file << confusion(i,j) << ",";
            }
        }
    }
    confusion_file.close();
}


void TestingAnalysis::save_multiple_classification_tests(const string& classification_tests_file_name) const
{
    const Tensor<type, 1> multiple_classification_tests = calculate_multiple_classification_precision();

    std::ofstream multiple_classifiaction_tests_file(classification_tests_file_name);

    multiple_classifiaction_tests_file << "accuracy,error" << endl;
    multiple_classifiaction_tests_file << multiple_classification_tests(0)* type(100) << "," << multiple_classification_tests(1)* type(100) << endl;

    multiple_classifiaction_tests_file.close();
}


/// Returns a matrix of subvectors which have the rates for a multiple classification problem.

Tensor<Tensor<Index,1>, 2> TestingAnalysis::calculate_multiple_classification_rates() const
{
#ifdef OPENNN_DEBUG

    check();

    if(!neural_network_pointer)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: TestingAnalysis class.\n"
               << "BinaryClassificationRates calculate_binary_classification_rates() const method.\n"
               << "Pointer to neural network in neural network is nullptr.\n";

        throw invalid_argument(buffer.str());
    }

    const Index inputs_number = neural_network_pointer->get_inputs_number();

    // Control sentence

    if(inputs_number != data_set_pointer->get_input_variables_number())
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: TestingAnalysis class." << endl
               << "BinaryClassificationRates calculate_binary_classification_rates() const method." << endl
               << "Number of inputs in neural network must be equal to number of inputs in data set." << endl;

        throw invalid_argument(buffer.str());
    }

    const Index outputs_number = neural_network_pointer->get_outputs_number();

    if(outputs_number != data_set_pointer->get_target_variables_number())
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: TestingAnalysis class." << endl
               << "BinaryClassificationRates calculate_binary_classification_rates() const method." << endl
               << "Number of outputs in neural network must be equal to number of targets in data set." << endl;

        throw invalid_argument(buffer.str());
    }

#endif

    Tensor<type, 2> inputs = data_set_pointer->get_testing_input_data();

    Tensor<Index, 1> inputs_dimensions = get_dimensions(inputs);

    Tensor<type, 2> targets = data_set_pointer->get_testing_target_data();

    Tensor<type, 2> outputs;

    outputs = neural_network_pointer->calculate_outputs(inputs.data(), inputs_dimensions);

    const Tensor<Index, 1> testing_indices = data_set_pointer->get_testing_samples_indices();

    return calculate_multiple_classification_rates(targets, outputs, testing_indices);
}


/// Returns a matrix of subvectors which have the rates for a multiple classification problem.
/// The number of rows of the matrix is the number targets.
/// The number of columns of the matrix is the number of columns of the target data.

Tensor<Tensor<Index,1>, 2> TestingAnalysis::calculate_multiple_classification_rates(const Tensor<type, 2>& targets,
                                                                                    const Tensor<type, 2>& outputs,
                                                                                    const Tensor<Index, 1>& testing_indices) const
{
    const Index samples_number = targets.dimension(0);
    const Index targets_number = targets.dimension(1);

    Tensor< Tensor<Index, 1>, 2> multiple_classification_rates(targets_number, targets_number);

    // Count instances per class

    const Tensor<Index, 2> confusion = calculate_confusion_multiple_classification(targets, outputs);

    for(Index i = 0; i < targets_number; i++)
    {
        for(Index j = 0; j < targets_number; j++)
        {
            multiple_classification_rates(i,j).resize(confusion(i,j));
        }
    }

    // Save indices

    Index target_index;
    Index output_index;

    Tensor<Index, 2> indices(targets_number, targets_number);
    indices.setZero();

    for(Index i = 0; i < samples_number; i++)
    {
        target_index = maximal_index(targets.chip(i, 0));
        output_index = maximal_index(outputs.chip(i, 0));

        multiple_classification_rates(target_index, output_index)(indices(target_index, output_index)) = testing_indices(i);

        indices(target_index, output_index)++;
    }

    return multiple_classification_rates;
}


Tensor<string, 2> TestingAnalysis::calculate_well_classified_samples(const Tensor<type, 2>& targets,
                                                                      const Tensor<type, 2>& outputs,
                                                                      const Tensor<string, 1>& labels) const
{
    const Index samples_number = targets.dimension(0);

    Tensor<string, 2> well_lassified_samples(samples_number, 4);

    Index predicted_class;
    Index actual_class;
    Index number_of_well_classified = 0;
    string class_name;

    const Tensor<string, 1> target_variables_names = data_set_pointer->get_target_variables_names();

    for(Index i = 0; i < samples_number; i++)
    {
        predicted_class = maximal_index(outputs.chip(i, 0));
        actual_class = maximal_index(targets.chip(i, 0));

        if(actual_class != predicted_class)
        {
            continue;
        }
        else
        {
            well_lassified_samples(number_of_well_classified, 0) = labels(i);
            class_name = target_variables_names(actual_class);
            well_lassified_samples(number_of_well_classified, 1) = class_name;
            class_name = target_variables_names(predicted_class);
            well_lassified_samples(number_of_well_classified, 2) = class_name;
            well_lassified_samples(number_of_well_classified, 3) = to_string(double(outputs(i, predicted_class)));

            number_of_well_classified ++;
        }
    }

    Eigen::array<Index, 2> offsets = {0, 0};
    Eigen::array<Index, 2> extents = {number_of_well_classified, 4};

    return well_lassified_samples.slice(offsets, extents);
}


Tensor<string, 2> TestingAnalysis::calculate_misclassified_samples(const Tensor<type, 2>& targets,
                                                                      const Tensor<type, 2>& outputs,
                                                                      const Tensor<string, 1>& labels) const
{
    const Index samples_number = targets.dimension(0);

    Index predicted_class;
    Index actual_class;
    string class_name;

    const Tensor<string, 1> target_variables_names = neural_network_pointer->get_outputs_names();

    Index count_misclassified = 0;

    for(Index i = 0; i < samples_number; i++)
    {
        predicted_class = maximal_index(outputs.chip(i, 0));
        actual_class = maximal_index(targets.chip(i, 0));

        if(actual_class != predicted_class) count_misclassified++;
    }

    Tensor<string, 2> misclassified_samples(count_misclassified, 4);

    Index j = 0;

    for(Index i = 0; i < samples_number; i++)
    {
        predicted_class = maximal_index(outputs.chip(i, 0));
        actual_class = maximal_index(targets.chip(i, 0));

        if(actual_class == predicted_class)
        {
            continue;
        }
        else
        {
            misclassified_samples(j, 0) = labels(i);
            class_name = target_variables_names(actual_class);
            misclassified_samples(j, 1) = class_name;
            class_name = target_variables_names(predicted_class);
            misclassified_samples(j, 2) = class_name;
            misclassified_samples(j, 3) = to_string(double(outputs(i, predicted_class)));
            j++;
        }
    }

//    Eigen::array<Index, 2> offsets = {0, 0};
//    Eigen::array<Index, 2> extents = {number_of_misclassified, 4};

//    return misclassified_samples.slice(offsets, extents);
    return misclassified_samples;
}


void TestingAnalysis::save_well_classified_samples(const Tensor<type, 2>& targets,
                                                    const Tensor<type, 2>& outputs,
                                                    const Tensor<string, 1>& labels,
                                                    const string& well_classified_samples_file_name)
{
    const Tensor<string,2> well_classified_samples = calculate_well_classified_samples(targets,
                                                                                           outputs,
                                                                                           labels);

    std::ofstream well_classified_samples_file(well_classified_samples_file_name);
    well_classified_samples_file << "sample_name,actual_class,predicted_class,probability" << endl;
    for(Index i = 0; i < well_classified_samples.dimension(0); i++)
    {
        well_classified_samples_file << well_classified_samples(i, 0) << ",";
        well_classified_samples_file << well_classified_samples(i, 1) << ",";
        well_classified_samples_file << well_classified_samples(i, 2) << ",";
        well_classified_samples_file << well_classified_samples(i, 3) << endl;
    }
    well_classified_samples_file.close();
}


void TestingAnalysis::save_misclassified_samples(const Tensor<type, 2>& targets,
                                                    const Tensor<type, 2>& outputs,
                                                    const Tensor<string, 1>& labels,
                                                    const string& misclassified_samples_file_name)
{
    const Tensor<string,2> misclassified_samples = calculate_misclassified_samples(targets,
                                                                                         outputs,
                                                                                         labels);

    std::ofstream misclassified_samples_file(misclassified_samples_file_name);
    misclassified_samples_file << "sample_name,actual_class,predicted_class,probability" << endl;
    for(Index i = 0; i < misclassified_samples.dimension(0); i++)
    {
        misclassified_samples_file << misclassified_samples(i, 0) << ",";
        misclassified_samples_file << misclassified_samples(i, 1) << ",";
        misclassified_samples_file << misclassified_samples(i, 2) << ",";
        misclassified_samples_file << misclassified_samples(i, 3) << endl;
    }
    misclassified_samples_file.close();
}


void TestingAnalysis::save_well_classified_samples_statistics(const Tensor<type, 2>& targets,
                                                                const Tensor<type, 2>& outputs,
                                                                const Tensor<string, 1>& labels,
                                                                const string& statistics_file_name)
{
    const Tensor<string, 2> well_classified_samples = calculate_well_classified_samples(targets,
                                                                                            outputs,
                                                                                            labels);

    Tensor<type, 1> well_classified_numerical_probabilities(well_classified_samples.dimension(0));

    for(Index i = 0; i < well_classified_numerical_probabilities.size(); i++)
    {
        well_classified_numerical_probabilities(i) = type(::atof(well_classified_samples(i, 3).c_str()));
    }

    std::ofstream classification_statistics_file(statistics_file_name);
    classification_statistics_file << "minimum,maximum,mean,std" << endl;
    classification_statistics_file << well_classified_numerical_probabilities.minimum() << ",";
    classification_statistics_file << well_classified_numerical_probabilities.maximum() << ",";

    classification_statistics_file << well_classified_numerical_probabilities.mean() << ",";

    classification_statistics_file << standard_deviation(well_classified_numerical_probabilities);

}


void TestingAnalysis::save_misclassified_samples_statistics(const Tensor<type, 2>& targets,
                                                               const Tensor<type, 2>& outputs,
                                                               const Tensor<string, 1>& labels,
                                                               const string& statistics_file_name)
{
    const Tensor<string, 2> misclassified_samples = calculate_misclassified_samples(targets,
                                                                                        outputs,
                                                                                        labels);

    Tensor<type, 1> misclassified_numerical_probabilities(misclassified_samples.dimension(0));

    for(Index i = 0; i < misclassified_numerical_probabilities.size(); i++)
    {
        misclassified_numerical_probabilities(i) = type(::atof(misclassified_samples(i, 3).c_str()));
    }
    std::ofstream classification_statistics_file(statistics_file_name);
    classification_statistics_file << "minimum,maximum,mean,std" << endl;
    classification_statistics_file << misclassified_numerical_probabilities.minimum() << ",";
    classification_statistics_file << misclassified_numerical_probabilities.maximum() << ",";
/*    
    classification_statistics_file << misclassified_numerical_probabilities.mean() << ",";
*/    
    classification_statistics_file << standard_deviation(misclassified_numerical_probabilities);
}


void TestingAnalysis::save_well_classified_samples_probability_histogram(const Tensor<type, 2>& targets,
                                                                           const Tensor<type, 2>& outputs,
                                                                           const Tensor<string, 1>& labels,
                                                                           const string& histogram_file_name) const
{
    const Tensor<string, 2> well_classified_samples = calculate_well_classified_samples(targets,
                                                                                            outputs,
                                                                                            labels);

    Tensor<type, 1> well_classified_numerical_probabilities(well_classified_samples.dimension(0));

    for(Index i = 0; i < well_classified_numerical_probabilities.size(); i++)
    {
        well_classified_numerical_probabilities(i) = type(::atof(well_classified_samples(i, 3).c_str()));
    }

    Histogram misclassified_samples_histogram(well_classified_numerical_probabilities);
    misclassified_samples_histogram.save(histogram_file_name);
}


void TestingAnalysis::save_well_classified_samples_probability_histogram(const Tensor<string, 2>& well_classified_samples,
                                                                           const string& histogram_file_name) const
{

    Tensor<type, 1> well_classified_numerical_probabilities(well_classified_samples.dimension(0));

    for(Index i = 0; i < well_classified_numerical_probabilities.size(); i++)
    {
        well_classified_numerical_probabilities(i) = type(::atof(well_classified_samples(i, 3).c_str()));
    }

    Histogram misclassified_samples_histogram(well_classified_numerical_probabilities);
    misclassified_samples_histogram.save(histogram_file_name);
}


void TestingAnalysis::save_misclassified_samples_probability_histogram(const Tensor<type, 2>& targets,
                                                                          const Tensor<type, 2>& outputs,
                                                                          const Tensor<string, 1>& labels,
                                                                          const string& histogram_file_name)
{
    const Tensor<string, 2> misclassified_samples = calculate_misclassified_samples(targets,
                                                                                          outputs,
                                                                                          labels);

    Tensor<type, 1> misclassified_numerical_probabilities(misclassified_samples.dimension(0));

    for(Index i = 0; i < misclassified_numerical_probabilities.size(); i++)
    {
        misclassified_numerical_probabilities(i) = type(::atof(misclassified_samples(i, 3).c_str()));
    }

    Histogram misclassified_samples_histogram(misclassified_numerical_probabilities);
    misclassified_samples_histogram.save(histogram_file_name);
}


void TestingAnalysis::save_misclassified_samples_probability_histogram(const Tensor<string, 2>& misclassified_samples,
                                                                          const string& histogram_file_name)
{

    Tensor<type, 1> misclassified_numerical_probabilities(misclassified_samples.dimension(0));

    for(Index i = 0; i < misclassified_numerical_probabilities.size(); i++)
    {
        misclassified_numerical_probabilities(i) = type(::atof(misclassified_samples(i, 3).c_str()));
    }

    Histogram misclassified_samples_histogram(misclassified_numerical_probabilities);
    misclassified_samples_histogram.save(histogram_file_name);
}


/// Calculates error autocorrelation across varying lags.
/// Returns a vector of subvectors.
/// The size of the vector is the number of targets.
/// The size of the subvectors is the number of lags for which autocorrelation is calculated.
/// @param maximum_lags_number Number of lags for which error autocorrelation is to be calculated.

Tensor<Tensor<type, 1>, 1> TestingAnalysis::calculate_error_autocorrelation(const Index& maximum_lags_number) const
{
#ifdef OPENNN_DEBUG

    check();

    if(!neural_network_pointer)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: TestingAnalysis class.\n"
               << "Tensor<type, 1> calculate_error_autocorrelation() const method.\n"
               << "Pointer to neural network in neural network is nullptr.\n";

        throw invalid_argument(buffer.str());
    }

    const Index inputs_number = neural_network_pointer->get_inputs_number();

    // Control sentence

    if(inputs_number != data_set_pointer->get_input_variables_number())
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: TestingAnalysis class." << endl
               << "Tensor<type, 1> calculate_error_autocorrelation() const method." << endl
               << "Number of inputs in neural network must be equal to number of inputs in data set." << endl;

        throw invalid_argument(buffer.str());
    }

    const Index outputs_number = neural_network_pointer->get_outputs_number();

    if(outputs_number != data_set_pointer->get_target_variables_number())
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: TestingAnalysis class." << endl
               << "Tensor<type, 1> calculate_error_autocorrelation() const method." << endl
               << "Number of outputs in neural network must be equal to number of targets in data set." << endl;

        throw invalid_argument(buffer.str());
    }

#endif

    Tensor<type, 2> inputs = data_set_pointer->get_testing_input_data();

    Tensor<Index, 1> inputs_dimensions = get_dimensions(inputs);

    Tensor<type, 2> targets = data_set_pointer->get_testing_target_data();

    Tensor<type, 2> outputs;

    outputs = neural_network_pointer->calculate_outputs(inputs.data(), inputs_dimensions);

    const Index targets_number = data_set_pointer->get_target_variables_number();

    const Tensor<type, 2> error = targets - outputs;

    Tensor<Tensor<type, 1>, 1> error_autocorrelations(targets_number);

    for(Index i = 0; i < targets_number; i++)
    {
        error_autocorrelations[i] = autocorrelations(this->thread_pool_device, error.chip(i,1), maximum_lags_number);
    }

    return error_autocorrelations;
}


/// Calculates the correlation between input and error.
/// Returns a vector of subvectors.
/// The size of the vector is the number of targets.
/// The size of the subvectors is the number of lags for which cross-correlation is calculated.
/// @param maximum_lags_number Number of lags for which cross-correlation is calculated.

Tensor<Tensor<type, 1>, 1> TestingAnalysis::calculate_inputs_errors_cross_correlation(const Index& lags_number) const
{
#ifdef OPENNN_DEBUG

    check();

    if(!neural_network_pointer)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: TestingAnalysis class.\n"
               << "Tensor<type, 1> calculate_inputs_errors_cross_correlation() const method.\n"
               << "Pointer to neural network in neural network is nullptr.\n";

        throw invalid_argument(buffer.str());
    }

    const Index inputs_number = neural_network_pointer->get_inputs_number();

    // Control sentence


    if(inputs_number != data_set_pointer->get_input_variables_number())
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: TestingAnalysis class." << endl
               << "Tensor<type, 1> calculate_inputs_errors_cross_correlation() const method." << endl
               << "Number of inputs in neural network must be equal to number of inputs in data set." << endl;

        throw invalid_argument(buffer.str());
    }

    const Index outputs_number = neural_network_pointer->get_outputs_number();

    if(outputs_number != data_set_pointer->get_target_variables_number())
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: TestingAnalysis class." << endl
               << "Tensor<type, 1> calculate_inputs_errors_cross_correlation() const method." << endl
               << "Number of outputs in neural network must be equal to number of targets in data set." << endl;

        throw invalid_argument(buffer.str());
    }

#endif

    const Index targets_number = data_set_pointer->get_target_variables_number();

    Tensor<type, 2> inputs = data_set_pointer->get_testing_input_data();

    Tensor<Index, 1> inputs_dimensions = get_dimensions(inputs);

    Tensor<type, 2> targets = data_set_pointer->get_testing_target_data();

    Tensor<type, 2> outputs;

    outputs = neural_network_pointer->calculate_outputs(inputs.data(), inputs_dimensions);

    const Tensor<type, 2> errors = targets - outputs;

    Tensor<Tensor<type, 1>, 1> inputs_errors_cross_correlation(targets_number);

    for(Index i = 0; i < targets_number; i++)
    {
        inputs_errors_cross_correlation[i] = cross_correlations(this->thread_pool_device, inputs.chip(i,1), errors.chip(i,1), lags_number);
    }

    return inputs_errors_cross_correlation;
}


/// Returns the results of a binary classification test in a single vector.
/// The size of that vector is fifteen.
/// The elements are:
/// <ul>
/// <li> Classification accuracy
/// <li> Error rate
/// <li> Sensitivity
/// <li> Specificity
/// <li> Precision
/// <li> Positive likelihood
/// <li> Negative likelihood
/// <li> F1 score
/// <li> False positive rate
/// <li> False discovery rate
/// <li> False negative rate
/// <li> Negative predictive value
/// <li> Matthews correlation coefficient
/// <li> Informedness
/// <li> Markedness
/// </ul>

Tensor<type, 1> TestingAnalysis::calculate_binary_classification_tests() const
{
#ifdef OPENNN_DEBUG

    const Index inputs_number = neural_network_pointer->get_inputs_number();

    if(!data_set_pointer)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: TestingAnalysis class." << endl
               << "Tensor<type, 1> calculate_binary_classification_tests() const." << endl
               << "Data set is nullptr." << endl;

        throw invalid_argument(buffer.str());
    }

    const Index targets_number = data_set_pointer->get_target_variables_number();

    const Index outputs_number = neural_network_pointer->get_outputs_number();

    // Control sentence

    if(inputs_number != data_set_pointer->get_input_variables_number())
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: TestingAnalysis class." << endl
               << "Tensor<type, 1> calculate_binary_classification_tests() const." << endl
               << "Number of inputs in neural network is not equal to number of inputs in data set." << endl;

        throw invalid_argument(buffer.str());
    }
    else if(outputs_number != 1)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: TestingAnalysis class." << endl
               << "Tensor<type, 1> calculate_binary_classification_tests() const." << endl
               << "Number of outputs in neural network must be one." << endl;

        throw invalid_argument(buffer.str());
    }
    else if(targets_number != 1)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: TestingAnalysis class." << endl
               << "Tensor<type, 1> calculate_binary_classification_tests() const." << endl
               << "Number of targets in data set must be one." << endl;

        throw invalid_argument(buffer.str());
    }

#endif

    // Confusion matrix

    const Tensor<Index, 2> confusion = calculate_confusion();

    const Index true_positive = confusion(0,0);
    const Index false_positive = confusion(1,0);
    const Index false_negative = confusion(0,1);
    const Index true_negative = confusion(1,1);

    // Classification accuracy

    type classification_accuracy;

    if(true_positive + true_negative + false_positive + false_negative == 0)
    {
        classification_accuracy = type(0);
    }
    else
    {
        classification_accuracy = static_cast<type>(true_positive + true_negative)/static_cast<type>(true_positive + true_negative + false_positive + false_negative);
    }

    // Error rate

    type error_rate;

    if(true_positive + true_negative + false_positive + false_negative == 0)
    {
        error_rate = type(0);
    }
    else
    {
        error_rate = static_cast<type>(false_positive + false_negative)/static_cast<type>(true_positive + true_negative + false_positive + false_negative);
    }

    // Sensitivity

    type sensitivity;

    if(true_positive + false_negative == 0)
    {
        sensitivity = type(0);
    }
    else
    {
        sensitivity = static_cast<type>(true_positive)/static_cast<type>(true_positive + false_negative);
    }

    // False positive rate

    type false_positive_rate;

    if(false_positive + true_negative == 0)
    {
        false_positive_rate = type(0);
    }
    else
    {
        false_positive_rate = static_cast<type>(false_positive)/static_cast<type>(false_positive + true_negative);
    }

    // Specificity

    type specificity;

    if(false_positive + true_negative== 0)
    {
        specificity = type(0);
    }
    else
    {
        specificity = static_cast<type>(true_negative)/static_cast<type>(true_negative + false_positive);
    }

    // Precision

    type precision;

    if(true_positive + false_positive == 0)
    {
        precision = type(0);
    }
    else
    {
        precision = static_cast<type>(true_positive) /static_cast<type>(true_positive + false_positive);
    }

    // Positive likelihood

    type positive_likelihood;

    if(abs(classification_accuracy - static_cast<type>(1.0)) < type(NUMERIC_LIMITS_MIN))
    {
        positive_likelihood = type(1);
    }
    else if(abs(static_cast<type>(1.0) - specificity) < type(NUMERIC_LIMITS_MIN))
    {
        positive_likelihood = type(0);
    }
    else
    {
        positive_likelihood = sensitivity/(static_cast<type>(1.0) - specificity);
    }

    // Negative likelihood

    type negative_likelihood;

    if(static_cast<Index>(classification_accuracy) == 1)
    {
        negative_likelihood = type(1);
    }
    else if(abs(type(1) - sensitivity) < type(NUMERIC_LIMITS_MIN))
    {
        negative_likelihood = type(0);
    }
    else
    {
        negative_likelihood = specificity/(static_cast<type>(1.0) - sensitivity);
    }

    // F1 score

    type f1_score;

    if(2*true_positive + false_positive + false_negative == 0)
    {
        f1_score = type(0);
    }
    else
    {
        f1_score = static_cast<type>(2.0)* type(true_positive)/(static_cast<type>(2.0)* type(true_positive) + type(false_positive) + type(false_negative));
    }

    // False discovery rate

    type false_discovery_rate;

    if(false_positive + true_positive == 0)
    {
        false_discovery_rate = type(0);
    }
    else
    {
        false_discovery_rate = static_cast<type>(false_positive) /static_cast<type>(false_positive + true_positive);
    }

    // False negative rate

    type false_negative_rate;

    if(false_negative + true_positive == 0)
    {
        false_negative_rate = type(0);
    }
    else
    {
        false_negative_rate = static_cast<type>(false_negative)/static_cast<type>(false_negative + true_positive);
    }

    // Negative predictive value

    type negative_predictive_value;

    if(true_negative + false_negative == 0)
    {
        negative_predictive_value = type(0);
    }
    else
    {
        negative_predictive_value = static_cast<type>(true_negative)/static_cast<type>(true_negative + false_negative);
    }

    // Matthews correlation coefficient

    type Matthews_correlation_coefficient;

    if((true_positive + false_positive) *(true_positive + false_negative) *(true_negative + false_positive) *(true_negative + false_negative) == 0)
    {
        Matthews_correlation_coefficient = type(0);
    }
    else
    {
        Matthews_correlation_coefficient = static_cast<type>(true_positive * true_negative - false_positive * false_negative) / static_cast<type>(sqrt((true_positive + false_positive) *(true_positive + false_negative) *(true_negative + false_positive) *(true_negative + false_negative)))
                                           ;
    }

    //Informedness

    type informedness = sensitivity + specificity - type(1);

    //Markedness

    type markedness;

    if(true_negative + false_positive == 0)
    {
        markedness = precision - type(1);
    }
    else
    {
        markedness = precision + static_cast<type>(true_negative)/static_cast<type>(true_negative + false_positive) - static_cast<type>(1.0);
    }

    //Arrange vector

    Tensor<type, 1> binary_classification_test(15);

    binary_classification_test[0] = classification_accuracy;
    binary_classification_test[1] = error_rate;
    binary_classification_test[2] = sensitivity;
    binary_classification_test[3] = specificity;
    binary_classification_test[4] = precision;
    binary_classification_test[5] = positive_likelihood;
    binary_classification_test[6] = negative_likelihood;
    binary_classification_test[7] = f1_score;
    binary_classification_test[8] = false_positive_rate;
    binary_classification_test[9] = false_discovery_rate;
    binary_classification_test[10] = false_negative_rate;
    binary_classification_test[11] = negative_predictive_value;
    binary_classification_test[12] = Matthews_correlation_coefficient;
    binary_classification_test[13] = informedness;
    binary_classification_test[14] = markedness;

    return binary_classification_test;
}




void TestingAnalysis::print_binary_classification_tests() const
{
    const Tensor<type, 1> binary_classification_tests = calculate_binary_classification_tests();

    cout << "Binary classification tests: " << endl;
    cout << "Classification accuracy : " << binary_classification_tests[0] << endl;
    cout << "Error rate              : " << binary_classification_tests[1] << endl;
    cout << "Sensitivity             : " << binary_classification_tests[2] << endl;
    cout << "Specificity             : " << binary_classification_tests[3] << endl;
}


/// Returns the results of a multiple classification test in a single matrix.
/// The size of that vector is 3 *(number of target variables + 2).
/// The columns are:
/// <ul>
/// <li> Precision
/// <li> Recall
/// <li> F1-score
/// </ul>
/// The 2 lasts row represents the normal and weighted average of each test.

Tensor<type, 2> TestingAnalysis::calculate_multiple_classification_tests() const
{

    const Index inputs_number = neural_network_pointer->get_inputs_number();

    const Index targets_number = data_set_pointer->get_target_variables_number();

    const Index outputs_number = neural_network_pointer->get_outputs_number();

#ifdef OPENNN_DEBUG

    if(!data_set_pointer)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: TestingAnalysis class." << endl
               << "Tensor<type, 1> calculate_binary_classification_tests() const." << endl
               << "Data set is nullptr." << endl;

        throw invalid_argument(buffer.str());
    }

    if(inputs_number != data_set_pointer->get_input_variables_number())
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: TestingAnalysis class." << endl
               << "Tensor<type, 1> calculate_binary_classification_tests() const." << endl
               << "Number of inputs in neural network is not equal to number of inputs in data set." << endl;

        throw invalid_argument(buffer.str());
    }

#endif

    Tensor<type,2> multiple_classification_tests(targets_number + 2,3);

    const Tensor<Index, 2> confusion = calculate_confusion();

    Index true_positives = 0;
    Index true_negatives = 0;
    Index false_positives = 0;
    Index false_negatives = 0;

    type total_precision = 0;
    type total_recall = 0;
    type total_f1_score= 0;

    type total_weighted_precision = 0;
    type total_weighted_recall = 0;
    type total_weighted_f1_score= 0;

    Index total_samples = 0;

    for(Index target_index = 0; target_index < targets_number; target_index++)
    {
        true_positives = confusion(target_index, target_index);

        Tensor<Index,0> row_sum = confusion.chip(target_index,0).sum();
        Tensor<Index,0> column_sum = confusion.chip(target_index,1).sum();

        false_negatives = row_sum(0) - true_positives;
        false_positives= column_sum(0) - true_positives;

        // Precision

        type precision;
        if(true_positives + false_positives == 0)
            precision = type(0);
        else
            precision = static_cast<type>(true_positives) /static_cast<type>(true_positives + false_positives);

        // Recall

        type recall;

        if(true_positives + false_negatives == 0)
            recall = type(0);
        else
            recall = static_cast<type>(true_positives)/static_cast<type>(true_positives + false_negatives);

        // F1-Score

        type f1_score;

        if(precision + recall == 0)
            f1_score = type(0);
        else
            f1_score = static_cast<type>(2*precision*recall)/static_cast<type>(precision + recall);

        // Save results

        multiple_classification_tests(target_index, 0) = precision;
        multiple_classification_tests(target_index, 1) = recall;
        multiple_classification_tests(target_index, 2) = f1_score;

        total_precision += precision;
        total_recall += recall;
        total_f1_score += f1_score;

        total_weighted_precision += precision * row_sum(0);
        total_weighted_recall += recall * row_sum(0);
        total_weighted_f1_score += f1_score * row_sum(0);

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


/// Returns the logloss for a binary classification problem.

type TestingAnalysis::calculate_logloss() const
{
#ifdef OPENNN_DEBUG

    check();

    if(!neural_network_pointer)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: TestingAnalysis class.\n"
               << "Tensor<type, 1> calculate_inputs_errors_cross_correlation() const method.\n"
               << "Pointer to neural network in neural network is nullptr.\n";

        throw invalid_argument(buffer.str());
    }

    const Index inputs_number = neural_network_pointer->get_inputs_number();

    // Control sentence

    if(inputs_number != data_set_pointer->get_input_variables_number())
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: TestingAnalysis class." << endl
               << "Tensor<type, 1> calculate_inputs_errors_cross_correlation() const method." << endl
               << "Number of inputs in neural network must be equal to number of inputs in data set." << endl;

        throw invalid_argument(buffer.str());
    }

    const Index outputs_number = neural_network_pointer->get_outputs_number();

    if(outputs_number != data_set_pointer->get_target_variables_number())
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: TestingAnalysis class." << endl
               << "Tensor<type, 1> calculate_inputs_errors_cross_correlation() const method." << endl
               << "Number of outputs in neural network must be equal to number of targets in data set." << endl;

        throw invalid_argument(buffer.str());
    }

#endif
    Tensor<type, 2> inputs = data_set_pointer->get_testing_input_data();

    Tensor<Index, 1> inputs_dimensions = get_dimensions(inputs);

    Tensor<type, 2> targets = data_set_pointer->get_testing_target_data();

    Tensor<type, 2> outputs;

    outputs = neural_network_pointer->calculate_outputs(inputs.data(), inputs_dimensions);

    const Index testing_samples_number = data_set_pointer->get_testing_samples_number();

    type logloss = type(0);

    for(Index i = 0; i < testing_samples_number; i++)
    {
        logloss += targets(i,0)*log(outputs(i,0)) + (type(1) - targets(i,0))*log(type(1) - outputs(i,0));
    }

    return -logloss/type(testing_samples_number);
}


/// Prints to the standard output the string representation of this testing analysis object.

void TestingAnalysis::print() const
{
}


/// Serializes the testing analysis object into an XML document of the TinyXML library without keeping the DOM tree in memory.
/// See the OpenNN manual for more information about the format of this document.

void TestingAnalysis::write_XML(tinyxml2::XMLPrinter& file_stream) const
{
    ostringstream buffer;
    file_stream.OpenElement("TestingAnalysis");

    // Display

    file_stream.OpenElement("Display");

    buffer.str("");
    buffer << display;

    file_stream.PushText(buffer.str().c_str());

    file_stream.CloseElement();


    file_stream.CloseElement();
}


/// Deserializes a TinyXML document into this testing analysis object.
/// @param document XML document containing the member data.

void TestingAnalysis::from_XML(const tinyxml2::XMLDocument& document)
{
    ostringstream buffer;

    const tinyxml2::XMLElement* root_element = document.FirstChildElement("TestingAnalysis");

    if(!root_element)
    {
        buffer << "OpenNN Exception: TestingAnalysis class.\n"
               << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
               << "Testing analysis element is nullptr.\n";

        throw invalid_argument(buffer.str());
    }

    // Display

    const tinyxml2::XMLElement* element = root_element->FirstChildElement("Display");

    if(element)
    {
        string new_display_string = element->GetText();

        try
        {
            set_display(new_display_string != "0");
        }
        catch(const invalid_argument& e)
        {
            cerr << e.what() << endl;
        }
    }
}


/// Saves to an XML file the members of this testing analysis object.
/// @param file_name Name of testing analysis XML file.

void TestingAnalysis::save(const string& file_name) const
{
    FILE *pFile;

    pFile = fopen(file_name.c_str(), "w");

    if(pFile)
    {
        tinyxml2::XMLPrinter printer(pFile);
        write_XML(printer);
        fclose(pFile);
    }
}


/// Loads from an XML file the members for this testing analysis object.
/// @param file_name Name of testing analysis XML file.

void TestingAnalysis::load(const string& file_name)
{
    set_default();

    tinyxml2::XMLDocument document;

    if(document.LoadFile(file_name.c_str()))
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: Testing analysis class.\n"
               << "void load(const string&) method.\n"
               << "Cannot load XML file " << file_name << ".\n";

        throw invalid_argument(buffer.str());
    }

    from_XML(document);
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
