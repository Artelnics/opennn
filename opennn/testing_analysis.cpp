/****************************************************************************************************************/
/*                                                                                                              */
/*   OpenNN: Open Neural Networks Library                                                                       */
/*   www.opennn.net                                                                                             */
/*                                                                                                              */
/*   T E S T I N G   A N A L Y S I S   C L A S S                                                                */
/*                                                                                                              */ 
/*   Artificial Intelligence Techniques SL                                                                      */
/*   artelnics@artelnics.com                                                                                    */
/*                                                                                                              */
/****************************************************************************************************************/

// OpenNN includes

#include "testing_analysis.h"
#include "correlation_analysis.h"

namespace OpenNN
{

// DEFAULT CONSTRUCTOR

/// Default constructor.
/// It creates a testing analysis object neither associated to a neural network nor to a mathematical model or a data set.
/// By default, it constructs the function regression testing object.

TestingAnalysis::TestingAnalysis()
 : neural_network_pointer(nullptr),
   data_set_pointer(nullptr)
{
   set_default();
}


// NEURAL NETWORK CONSTRUCTOR

/// Neural network constructor. 
/// It creates a testing analysis object associated to a neural network but not to a mathematical model or a data set.
/// By default, it constructs the function regression testing object. 
/// @param new_neural_network_pointer Pointer to a neural network object.

TestingAnalysis::TestingAnalysis(NeuralNetwork* new_neural_network_pointer)
: neural_network_pointer(new_neural_network_pointer),
   data_set_pointer(nullptr)
{
   set_default();
}


/// Data set constructor. 
/// It creates a testing analysis object not associated to a neural network, associated to a data set and not associated to a mathematical model. 
/// By default, it constructs the function regression testing object. 
/// @param new_data_set_pointer Pointer to a data set object.

TestingAnalysis::TestingAnalysis(DataSet* new_data_set_pointer)
: neural_network_pointer(nullptr),
   data_set_pointer(new_data_set_pointer)
{
   set_default();
}


/// Neural network and data set constructor.
/// It creates a testing analysis object associated to a neural network and to a data set.
/// By default, it constructs the function regression testing object.
/// @param new_neural_network_pointer Pointer to a neural network object.
/// @param new_data_set_pointer Pointer to a data set object.

TestingAnalysis::TestingAnalysis(NeuralNetwork* new_neural_network_pointer, DataSet* new_data_set_pointer)
 : neural_network_pointer(new_neural_network_pointer),
   data_set_pointer(new_data_set_pointer)
{
   set_default();
}


// XML CONSTRUCTOR

/// XML constructor. 
/// It creates a testing analysis object neither associated to a neural network nor to a mathematical model or a data set. 
/// It also loads the members of this object from a TinyXML document.
/// @param testing_analysis_document XML document containing the member data.

TestingAnalysis::TestingAnalysis(const tinyxml2::XMLDocument& testing_analysis_document)
 : neural_network_pointer(nullptr),
   data_set_pointer(nullptr)
{
   set_default();

   from_XML(testing_analysis_document);
}


// FILE CONSTRUCTOR

/// File constructor.
/// It creates a testing analysis object neither associated to a neural network nor to a mathematical model or a data set.
/// It also loads the members of this object from XML file.
/// @param file_name Name of testing analysis XML file.

TestingAnalysis::TestingAnalysis(const string& file_name)
 : neural_network_pointer(nullptr),
   data_set_pointer(nullptr)
{
   set_default();

   load(file_name);
}

// DESTRUCTOR

/// Destructor. 
/// It deletes the function regression testing, classification testing, time series prediction testing and inverse problem testing objects. 

TestingAnalysis::~TestingAnalysis()
{
}


// METHODS

// NeuralNetwork* get_neural_network_pointer() const method

/// Returns a pointer to the neural network object which is to be tested.

NeuralNetwork* TestingAnalysis::get_neural_network_pointer() const
{
    #ifdef __OPENNN_DEBUG__

    if(!neural_network_pointer)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: TestingAnalysis class.\n"
               << "NeuralNetwork* get_neural_network_pointer() const method.\n"
               << "Neural network pointer is nullptr.\n";

        throw logic_error(buffer.str());
    }

    #endif

   return(neural_network_pointer);   
}


// DataSet* get_data_set_pointer() const method

/// Returns a pointer to the data set object on which the neural network is tested. 

DataSet* TestingAnalysis::get_data_set_pointer() const
{
    #ifdef __OPENNN_DEBUG__

    if(!data_set_pointer)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: TestingAnalysis class.\n"
               << "DataSet* get_data_set_pointer() const method.\n"
               << "Data set pointer is nullptr.\n";

        throw logic_error(buffer.str());
    }

    #endif

   return(data_set_pointer);
}


/// Returns true if messages from this class can be displayed on the screen,
/// or false if messages from this class can't be displayed on the screen.

const bool& TestingAnalysis::get_display() const
{
   return(display);     
}


/// Sets some default values to the testing analysis object:
/// <ul>
/// <li> Display: True.
/// </ul>

void TestingAnalysis::set_default()
{
   display = true;
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
/// If it is set to true messages from this class are to be displayed on the screen;
/// if it is set to false messages from this class are not to be displayed on the screen.
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

      throw logic_error(buffer.str());
   }

   if(!data_set_pointer)
   {
      buffer << "OpenNN Exception: TestingAnalysis class.\n"
             << "void check() const method.\n"
             << "Data set pointer is nullptr.\n";

      throw logic_error(buffer.str());
   }
}


// Vector< Matrix<double> > calculate_target_outputs() const method

/// Returns a vector of matrices with number of rows equal to number of testing instances and
/// number of columns equal to two(the targets value and the outputs value).

Vector< Matrix<double> > TestingAnalysis::calculate_target_outputs() const
{
   // Control sentence(if debug)

   #ifdef __OPENNN_DEBUG__

   check();

   #endif

   // Data set stuff

   const Instances& instances = data_set_pointer->get_instances();

   const size_t testing_instances_number = instances.get_testing_instances_number();

   const Matrix<double> testing_inputs = data_set_pointer->get_testing_inputs();

   const Matrix<double> testing_targets = data_set_pointer->get_testing_targets();

   // Neural network stuff

   const MultilayerPerceptron* multilayer_perceptron_pointer = neural_network_pointer->get_multilayer_perceptron_pointer();

   const size_t outputs_number = multilayer_perceptron_pointer->get_outputs_number();

   const Matrix<double> testing_outputs = neural_network_pointer->calculate_outputs(testing_inputs);

   // Approximation testing stuff

   Vector< Matrix<double> > target_outputs(outputs_number);

   for(size_t i = 0; i < outputs_number; i++)
   {
      target_outputs[i].set(testing_instances_number, 2);

      target_outputs[i].set_column(0, testing_targets.get_column(i), "");
      target_outputs[i].set_column(1, testing_outputs.get_column(i), "");
   }

   return(target_outputs);
}


/// Returns a vector of matrices with number of rows equal to number of testing instances and
/// number of columns equal to two(the targets value and the outputs value).

Vector< Matrix<double> > TestingAnalysis::calculate_forecasting_target_outputs() const
{
   // Control sentence(if debug)

   #ifdef __OPENNN_DEBUG__

   check();

   #endif
/*
   // Data set stuff

   const Instances& instances = data_set_pointer->get_instances();

   const size_t testing_instances_number = instances.get_testing_instances_number();

   const Matrix<double> testing_inputs = data_set_pointer->get_testing_inputs();

   const Matrix<double> targets = data_set_pointer->get_testing_targets();

   const Vector<double> time = data_set_pointer->get_testing_time();

   // Neural network stuff

   const MultilayerPerceptron* multilayer_perceptron_pointer = neural_network_pointer->get_multilayer_perceptron_pointer();

   const size_t outputs_number = multilayer_perceptron_pointer->get_outputs_number();

   const Matrix<double> outputs = neural_network_pointer->calculate_outputs(testing_inputs, time);

   // Approximation testing stuff

   Vector< Matrix<double> > target_outputs(outputs_number);

   for(size_t i = 0; i < outputs_number; i++)
   {
      target_outputs[i].set(testing_instances_number, 2);

      target_outputs[i].set_column(0, targets.get_column(i), "name");
      target_outputs[i].set_column(1, outputs.get_column(i), "value");
   }

   return(target_outputs);
*/

   return Vector< Matrix<double> >();
}


/// Performs a linear regression analysis between the testing instances in the data set and
/// the corresponding neural network outputs.
/// It returns all the provided parameters in a vector of vectors.
/// The number of elements in the vector is equal to the number of output variables.
/// The size of each element is equal to the number of regression parameters(2).
/// In this way, each subvector contains the regression parameters intercept and slope of an output variable.

Vector< LinearRegressionParameters<double> > TestingAnalysis::calculate_linear_regression_parameters() const
{
   // Control sentence(if debug)

   #ifdef __OPENNN_DEBUG__

   check();

   #endif

   // Data set stuff

   const Instances& instances = data_set_pointer->get_instances();

   const size_t testing_instances_number = instances.get_testing_instances_number();

   // Neural network stuff

   const MultilayerPerceptron* multilayer_perceptron_pointer = neural_network_pointer->get_multilayer_perceptron_pointer();

   const size_t outputs_number = multilayer_perceptron_pointer->get_outputs_number();

   #ifdef __OPENNN_DEBUG__

   ostringstream buffer;

   if(testing_instances_number == 0)
   {
      buffer << "OpenNN Exception: TestingAnalysis class.\n"
             << "Vector< LinearRegressionParameters<double> > calculate_linear_regression_parameters() const method.\n"
             << "Number of testing instances is zero.\n";

      throw logic_error(buffer.str());
   }

   #endif

   // Calculate regression parameters

   const Matrix<double> inputs = data_set_pointer->get_testing_inputs();
   const Matrix<double> targets = data_set_pointer->get_testing_targets();
   const Matrix<double> outputs = neural_network_pointer->get_multilayer_perceptron_pointer()->calculate_outputs(inputs);

   const Vector< LinearRegressionParameters<double> > linear_regression_parameters = calculate_linear_regression_parameters(targets,outputs);

//   Matrix<double> data({targets, outputs});

//   cout << data;

   return(linear_regression_parameters);
}


Vector< LinearRegressionParameters<double> > TestingAnalysis::calculate_linear_regression_parameters(const Matrix<double>& target, const Matrix<double>& output) const
{
    const size_t outputs_number = data_set_pointer->get_variables_pointer()->get_targets_number();

   Vector< LinearRegressionParameters<double> > linear_regression_parameters(outputs_number);

   for(size_t i = 0; i < outputs_number; i++)
   {
       linear_regression_parameters[i] = output.get_column(i).calculate_linear_regression_parameters(target.get_column(i));
   }

   return(linear_regression_parameters);
}


void TestingAnalysis::print_linear_regression_correlations() const
{
    const Vector< LinearRegressionParameters<double> > linear_regression_parameters = calculate_linear_regression_parameters();

    const Vector<string> targets_name = data_set_pointer->get_variables().get_targets_name();

    const size_t targets_number = linear_regression_parameters.size();

    for(size_t i = 0; i < targets_number; i++)
    {
        cout << targets_name[i] << " correlation: " << linear_regression_parameters[i].correlation << endl;
    }
}

std::vector<double> TestingAnalysis::get_linear_regression_correlations_std() const
{
    const Vector< LinearRegressionParameters<double> > linear_regression_parameters = calculate_linear_regression_parameters();

    const Vector<string> targets_name = data_set_pointer->get_variables().get_targets_name();

    const size_t targets_number = linear_regression_parameters.size();

    std::vector<double> std_correlations(targets_number);

    for(size_t i = 0; i < targets_number; i++)
    {
        std_correlations[i] = linear_regression_parameters[i].correlation;
    }

    return(std_correlations);
}



// Vector< LinearRegressionParameters<double> > calculate_forecasting_linear_regression_parameters() const method

/// Performs a linear regression analysis between the testing instances in the data set and
/// the corresponding neural network outputs for a forecasting problem.
/// It returns all the provided parameters in a vector of vectors.
/// The number of elements in the vector is equal to the number of output variables.
/// The size of each element is equal to the number of regression parameters(2).
/// In this way, each subvector contains the regression parameters intercept and slope of an output variable.

Vector< LinearRegressionParameters<double> > TestingAnalysis::calculate_forecasting_linear_regression_parameters() const
{
   // Control sentence(if debug)

   #ifdef __OPENNN_DEBUG__

   check();

   #endif
/*
   // Data set stuff

   const Instances& instances = data_set_pointer->get_instances();

   const size_t testing_instances_number = instances.get_testing_instances_number();

   // Neural network stuff

   const MultilayerPerceptron* multilayer_perceptron_pointer = neural_network_pointer->get_multilayer_perceptron_pointer();

   const size_t outputs_number = multilayer_perceptron_pointer->get_outputs_number();

   #ifdef __OPENNN_DEBUG__

   ostringstream buffer;

   if(testing_instances_number == 0)
   {
      buffer << "OpenNN Exception: TestingAnalysis class.\n"
             << "Vector< LinearRegressionParameters<double> > calculate_linear_regression_parameters() const method.\n"
             << "Number of testing instances is zero.\n";

      throw logic_error(buffer.str());
   }

   #endif

   // Calculate regression parameters

   const Matrix<double> inputs = data_set_pointer->get_testing_inputs();
   const Matrix<double> targets = data_set_pointer->get_testing_targets();
   const Vector<double> time = data_set_pointer->get_testing_time();
   const Matrix<double> outputs = neural_network_pointer->calculate_outputs(inputs, time);

   Vector<double> target_variable(testing_instances_number);
   Vector<double> output_variable(testing_instances_number);

   Vector< LinearRegressionParameters<double> > linear_regression_parameters(outputs_number);

   for(size_t i = 0; i < outputs_number; i++)
   {
       target_variable = targets.get_column(i);
       output_variable = outputs.get_column(i);

       linear_regression_parameters[i] = output_variable.calculate_linear_regression_parameters(target_variable);
   }

   return(linear_regression_parameters);
*/
    return Vector< LinearRegressionParameters<double> >();
}



// TestingAnalysis::LinearRegressionResults perform_linear_regression_analysis() const method

/// Performs a linear regression analysis of a neural network on the testing indices of a data set.
/// It returns a linear regression analysis results structure, which consists of:
/// <ul>
/// <li> Linear regression parameters.
/// <li> Scaled target and output data.
/// </ul>

Vector<TestingAnalysis::LinearRegressionAnalysis> TestingAnalysis::perform_linear_regression_analysis() const
{
    check();

    const MultilayerPerceptron* multilayer_perceptron_pointer = neural_network_pointer->get_multilayer_perceptron_pointer();

    const size_t outputs_number = multilayer_perceptron_pointer->get_outputs_number();

    const Instances& instances = data_set_pointer->get_instances();

    const size_t testing_instances_number = instances.get_testing_instances_number();

    if(testing_instances_number == 0)
    {
        ostringstream buffer;

       buffer << "OpenNN Exception: TestingAnalysis class.\n"
              << "LinearRegressionResults perform_linear_regression_analysis() const method.\n"
              << "Number of testing instances is zero.\n";

       throw logic_error(buffer.str());
    }


   const Matrix<double> testing_inputs = data_set_pointer->get_testing_inputs();

   const Matrix<double> testing_targets = data_set_pointer->get_testing_targets();

   // Neural network stuff

   const Matrix<double> testing_outputs = neural_network_pointer->calculate_outputs(testing_inputs);

   // Approximation testing stuff

   Vector<LinearRegressionAnalysis> linear_regression_results(outputs_number);

   for(size_t i = 0;  i < outputs_number; i++)
   {
       const Vector<double> targets = testing_targets.get_column(i);
       const Vector<double> outputs = testing_outputs.get_column(i);

       const LinearRegressionParameters<double> linear_regression_parameters = outputs.calculate_linear_regression_parameters(targets);

       linear_regression_results[i].targets = targets;
       linear_regression_results[i].outputs = outputs;

       linear_regression_results[i].intercept = linear_regression_parameters.intercept;
       linear_regression_results[i].slope = linear_regression_parameters.slope;
       linear_regression_results[i].correlation = linear_regression_parameters.correlation;
   }

   return(linear_regression_results);
}


void TestingAnalysis::perform_linear_regression_analysis_void() const
{
    perform_linear_regression_analysis();
}


// TestingAnalysis::LinearRegressionResults perform_forecasting_linear_regression_analysis() const method

/// Performs a linear regression analysis of a neural network on the testing indices of a data set for a forecasting problem.
/// It returns a linear regression analysis results structure, which consists of:
/// <ul>
/// <li> Linear regression parameters.
/// <li> Target and output data.
/// </ul>

TestingAnalysis::LinearRegressionAnalysis TestingAnalysis::perform_forecasting_linear_regression_analysis() const
{
    check();

    const Instances& instances = data_set_pointer->get_instances();

    const size_t testing_instances_number = instances.get_testing_instances_number();

    if(testing_instances_number == 0)
    {
        ostringstream buffer;

       buffer << "OpenNN Exception: TestingAnalysis class.\n"
              << "LinearRegressionResults perform_forecasting_linear_regression_analysis() const method.\n"
              << "Number of testing instances is zero.\n";

       throw logic_error(buffer.str());
    }

   LinearRegressionAnalysis linear_regression_results;

//   linear_regression_results.target_outputs = calculate_forecasting_target_outputs();
//   linear_regression_results.linear_regression_parameters = calculate_forecasting_linear_regression_parameters();

   return(linear_regression_results);
}

/// Calculates the errors between the outputs from a neural network and the testing instances in a data set.
/// It returns a vector of tree matrices:
/// <ul>
/// <li> Absolute error.
/// <li> Relative error.
/// <li> Percentage error.
/// </ul>
/// The number of rows in each matrix is the number of testing instances in the data set.
/// The number of columns is the number of outputs in the neural network.

Vector< Matrix<double> > TestingAnalysis::calculate_error_data() const
{
   // Data set stuff

   #ifdef __OPENNN_DEBUG__

   check();

   #endif

   const size_t testing_instances_number = data_set_pointer->get_instances().get_testing_instances_number();

    #ifdef __OPENNN_DEBUG__

    ostringstream buffer;

    if(testing_instances_number == 0)
    {
       buffer << "OpenNN Exception: TestingAnalysis class.\n"
              << "Vector< Matrix<double> > calculate_error_data() const.\n"
              << "Number of testing instances is zero.\n";

       throw logic_error(buffer.str());
    }

    #endif

   const Matrix<double> inputs = data_set_pointer->get_testing_inputs();

   const Matrix<double> targets = data_set_pointer->get_testing_targets();

   // Neural network stuff

   const Matrix<double> outputs = neural_network_pointer->calculate_outputs(inputs);

   const UnscalingLayer* unscaling_layer_pointer = neural_network_pointer->get_unscaling_layer_pointer();

   #ifdef __OPENNN_DEBUG__

   if(!unscaling_layer_pointer)
   {
      buffer << "OpenNN Exception: TestingAnalysis class.\n"
             << "Vector< Matrix<double> > calculate_error_data() const.\n"
             << "Unscaling layer is nullptr.\n";

      throw logic_error(buffer.str());
   }

   #endif

   const Vector<double>& outputs_minimum = unscaling_layer_pointer->get_minimums();
   const Vector<double>& outputs_maximum = unscaling_layer_pointer->get_maximums();

   const size_t outputs_number = unscaling_layer_pointer->get_unscaling_neurons_number();

   // Error data

   Vector< Matrix<double> > error_data(outputs_number);

   Vector<double> targets_vector(testing_instances_number);
   Vector<double> outputs_vector(testing_instances_number);

   Vector<double> difference_absolute_value(testing_instances_number);

   for(size_t i = 0; i < outputs_number; i++)
   {
       error_data[i].set(testing_instances_number, 3, 0.0);

       // Absolute error

       targets_vector = targets.get_column(i);
       outputs_vector = outputs.get_column(i);

       difference_absolute_value = (targets - outputs).calculate_absolute_value();

       error_data[i].set_column(0, difference_absolute_value, "");

       // Relative error

       error_data[i].set_column(1, difference_absolute_value/abs(outputs_maximum[i]-outputs_minimum[i]), "");

       // Percentage error

       error_data[i].set_column(2, difference_absolute_value*100.0/abs(outputs_maximum[i]-outputs_minimum[i]), "");
    }

   return(error_data);
}


/// Calculates the percentege errors between the outputs from a neural network and the testing instances in a data set.
/// The number of rows in each matrix is the number of testing instances in the data set.

Vector< Vector<double> > TestingAnalysis::calculate_percentage_error_data() const
{
   // Data set stuff

   #ifdef __OPENNN_DEBUG__

   check();

   #endif

   const size_t testing_instances_number = data_set_pointer->get_instances().get_testing_instances_number();

    #ifdef __OPENNN_DEBUG__

    ostringstream buffer;

    if(testing_instances_number == 0)
    {
       buffer << "OpenNN Exception: TestingAnalysis class.\n"
              << "Vector< Matrix<double> > calculate_error_data() const.\n"
              << "Number of testing instances is zero.\n";

       throw logic_error(buffer.str());
    }

    #endif

   const Matrix<double> inputs = data_set_pointer->get_testing_inputs();

   const Matrix<double> targets = data_set_pointer->get_testing_targets();

   // Neural network stuff

   const Matrix<double> outputs = neural_network_pointer->calculate_outputs(inputs);

   const UnscalingLayer* unscaling_layer_pointer = neural_network_pointer->get_unscaling_layer_pointer();

   #ifdef __OPENNN_DEBUG__

   if(!unscaling_layer_pointer)
   {
      buffer << "OpenNN Exception: TestingAnalysis class.\n"
             << "Vector< Vector<double> > calculate_percentage_error_data() const.\n"
             << "Unscaling layer is nullptr.\n";

      throw logic_error(buffer.str());
   }

   #endif

   const Vector<double>& outputs_minimum = unscaling_layer_pointer->get_minimums();
   const Vector<double>& outputs_maximum = unscaling_layer_pointer->get_maximums();

   const size_t outputs_number = unscaling_layer_pointer->get_unscaling_neurons_number();

   // Error data

   Vector< Vector<double> > error_data(outputs_number);

   Vector<double> targets_vector(testing_instances_number);
   Vector<double> outputs_vector(testing_instances_number);

   Vector<double> difference_absolute_value(testing_instances_number);

   for(size_t i = 0; i < outputs_number; i++)
   {
       error_data[i].set(testing_instances_number, 0.0);

       // Absolute error

       targets_vector = targets.get_column(i);
       outputs_vector = outputs.get_column(i);

       difference_absolute_value = (targets - outputs);

       // Percentage error

       error_data[i].set(difference_absolute_value*100.0/abs(outputs_maximum[i]-outputs_minimum[i]));
    }

   return(error_data);
}



/// Calculates the errors between the outputs from a neural network and the testing instances
/// in a data set for a forecasting problem.
/// It returns a vector of tree matrices:
/// <ul>
/// <li> Absolute error.
/// <li> Relative error.
/// <li> Percentage error.
/// </ul>
/// The number of rows in each matrix is the number of testing instances in the data set.
/// The number of columns is the number of outputs in the neural network.

Vector< Matrix<double> > TestingAnalysis::calculate_forecasting_error_data() const
{
   // Data set stuff

   #ifdef __OPENNN_DEBUG__

   check();

   #endif

   const size_t testing_instances_number = data_set_pointer->get_instances().get_testing_instances_number();

    #ifdef __OPENNN_DEBUG__

    ostringstream buffer;

    if(testing_instances_number == 0)
    {
       buffer << "OpenNN Exception: TestingAnalysis class.\n"
              << "Vector< Matrix<double> > calculate_error_data() const.\n"
              << "Number of testing instances is zero.\n";

       throw logic_error(buffer.str());
    }

    #endif
/*

   const Matrix<double> inputs = data_set_pointer->get_testing_inputs();

   const Matrix<double> targets = data_set_pointer->get_testing_targets();

   const Vector<double> time = data_set_pointer->get_testing_time();

   // Neural network stuff

   const Matrix<double> outputs = neural_network_pointer->calculate_outputs(inputs, time);

   const UnscalingLayer* unscaling_layer_pointer = neural_network_pointer->get_unscaling_layer_pointer();

   #ifdef __OPENNN_DEBUG__

   if(!unscaling_layer_pointer)
   {
      buffer << "OpenNN Exception: TestingAnalysis class.\n"
             << "Vector< Matrix<double> > calculate_forecasting_error_data() const.\n"
             << "Unscaling layer is nullptr.\n";

      throw logic_error(buffer.str());
   }

   #endif

   const Vector<double>& outputs_minimum = unscaling_layer_pointer->get_minimums();
   const Vector<double>& outputs_maximum = unscaling_layer_pointer->get_maximums();

   const size_t outputs_number = unscaling_layer_pointer->get_unscaling_neurons_number();

   // Error data

   Vector< Matrix<double> > error_data(outputs_number);

   Vector<double> targets(testing_instances_number);
   Vector<double> outputs(testing_instances_number);

   Vector<double> difference_absolute_value(testing_instances_number);

   for(size_t i = 0; i < outputs_number; i++)
   {
       error_data[i].set(testing_instances_number, 3, 0.0);

       // Absolute error

       targets = targets.get_column(i);
       outputs = outputs.get_column(i);

       difference_absolute_value = (targets - outputs).calculate_absolute_value();

       error_data[i].set_column(0, difference_absolute_value, "");

       // Relative error

       error_data[i].set_column(1, difference_absolute_value/abs(outputs_maximum[i]-outputs_minimum[i]), "");

       // Percentage error

       error_data[i].set_column(2, difference_absolute_value*100.0/abs(outputs_maximum[i]-outputs_minimum[i]), "");
    }

   return(error_data);
*/

    return Vector< Matrix<double> >();
}


Vector< Statistics<double> > TestingAnalysis::calculate_absolute_errors_statistics() const
{
    // Data set

    const Matrix<double> inputs = data_set_pointer->get_testing_inputs();
    const Matrix<double> targets = data_set_pointer->get_testing_targets();

    // Neural network

    const Matrix<double> outputs = neural_network_pointer->calculate_outputs(inputs);

    // Error statistics

    Vector< Statistics<double> > statistics = calculate_absolute_errors_statistics(targets,outputs);

    return statistics;
}


Vector< Statistics<double> > TestingAnalysis::calculate_absolute_errors_statistics(const Matrix<double>& targets,
                                                                                   const Matrix<double>& outputs) const
{
    return (outputs - targets).calculate_absolute_value().calculate_statistics();
}


Vector< Statistics<double> > TestingAnalysis::calculate_percentage_errors_statistics() const
{
    // Data set

    const Matrix<double> inputs = data_set_pointer->get_testing_inputs();
    const Matrix<double> targets = data_set_pointer->get_testing_targets();

    // Neural network

    const Matrix<double> outputs = neural_network_pointer->calculate_outputs(inputs);

    // Error statistics

    Vector< Statistics<double> > statistics = calculate_percentage_errors_statistics(targets,outputs);

    return statistics;
}


Vector< Statistics<double> > TestingAnalysis::calculate_percentage_errors_statistics(const Matrix<double>& targets,
                                                                                     const Matrix<double>& outputs) const
{
    return ((outputs - targets).calculate_absolute_value()*100.0/targets).calculate_statistics();
}


/// Calculates the basic statistics on the error data.
/// <ul>
/// <li> Minimum.
/// <li> Maximum.
/// <li> Mean.
/// <li> Standard deviation
/// </ul>

Vector< Vector< Statistics<double> > > TestingAnalysis::calculate_error_data_statistics() const
{
    // Neural network stuff

    const MultilayerPerceptron* multilayer_perceptron_pointer = neural_network_pointer->get_multilayer_perceptron_pointer();

    const size_t outputs_number = multilayer_perceptron_pointer->get_outputs_number();

    // Testing analysis stuff

    Vector< Vector< Statistics<double> > > statistics(outputs_number);

    const Vector< Matrix<double> > error_data = calculate_error_data();

    for(size_t i = 0; i < outputs_number; i++)
    {
        statistics[i] = error_data[i].calculate_statistics();
    }

    return(statistics);
}


void TestingAnalysis::print_error_data_statistics() const
{
    const Variables& variables = data_set_pointer->get_variables();

    const size_t targets_number = variables.get_targets_number();

    const Vector<string> targets_name = variables.get_targets_name();
    const Vector<string> targets_units = variables.get_targets_units();

    const Vector< Vector< Statistics<double> > > error_data_statistics = calculate_error_data_statistics();

    for(size_t i = 0; i < targets_number; i++)
    {
        cout << targets_name[i] << endl;
        cout << "Minimum error: " << error_data_statistics[i][0].minimum << " " << targets_units[i] << endl;
        cout << "Maximum error: " << error_data_statistics[i][0].maximum << " " << targets_units[i] << endl;
        cout << "Mean error: " << error_data_statistics[i][0].mean << " " << targets_units[i] << endl;
        cout << "Standard deviation error: " << error_data_statistics[i][0].standard_deviation << " " << targets_units[i] << endl;

        cout << "Minimum percentage error: " << error_data_statistics[i][2].minimum << " %" <<  endl;
        cout << "Maximum percentage error: " << error_data_statistics[i][2].maximum << " %" <<  endl;
        cout << "Mean percentage error: " << error_data_statistics[i][2].mean << " %" <<  endl;
        cout << "Standard deviation percentage error: " << error_data_statistics[i][2].standard_deviation << " %" <<  endl;

        cout << endl;
    }
}


// Vector< Vector< Statistics<double> > > calculate_forecasting_error_data_statistics() const method

/// Calculates the basic statistics on the error data for a forecasting problem.
/// <ul>
/// <li> Minimum.
/// <li> Maximum.
/// <li> Mean.
/// <li> Standard deviation
/// </ul>

Vector< Vector< Statistics<double> > > TestingAnalysis::calculate_forecasting_error_data_statistics() const
{
    // Neural network stuff

    const MultilayerPerceptron* multilayer_perceptron_pointer = neural_network_pointer->get_multilayer_perceptron_pointer();

    const size_t outputs_number = multilayer_perceptron_pointer->get_outputs_number();

    // Testing analysis stuff

    Vector< Vector< Statistics<double> > > statistics(outputs_number);

   const Vector< Matrix<double> > error_data = calculate_forecasting_error_data();

   for(size_t i = 0; i < outputs_number; i++)
   {
       statistics[i] = error_data[i].calculate_statistics();
   }

   return(statistics);
}


// Vector< Matrix<double> > calculate_error_data_statistics_matrices() const method

/// Returns a vector of matrices with the statistics of the errors between the neural network outputs and the testing targets in the data set.
/// The size of the vector is the number of output variables.
/// The number of rows in each submatrix is three(absolute, relative and percentage errors).
/// The number of columns in each submatrix is four(minimum, maximum, mean and standard deviation).

Vector< Matrix<double> > TestingAnalysis::calculate_error_data_statistics_matrices() const
{
    const Vector< Vector< Statistics<double> > > error_data_statistics = calculate_error_data_statistics();

    const size_t outputs_number = error_data_statistics.size();

    Vector< Matrix<double> > statistics(outputs_number);

    for(size_t i = 0; i < outputs_number; i++)
    {
        statistics[i].set(3, 4);
        statistics[i].set_row(0, error_data_statistics[i][0].to_vector());
        statistics[i].set_row(1, error_data_statistics[i][1].to_vector());
        statistics[i].set_row(2, error_data_statistics[i][2].to_vector());
    }

    return(statistics);
}


// Vector< Matrix<double> > calculate_forecasting_error_data_statistics_matrices() const method

/// Returns a vector of matrices with the statistics of the errors between the neural network outputs
/// and the testing targets in the data set for a forecasting problem.
/// The size of the vector is the number of output variables.
/// The number of rows in each submatrix is three(absolute, relative and percentage errors).
/// The number of columns in each submatrix is four(minimum, maximum, mean and standard deviation).

Vector< Matrix<double> > TestingAnalysis::calculate_forecasting_error_data_statistics_matrices() const
{
    const Vector< Vector< Statistics<double> > > error_data_statistics = calculate_forecasting_error_data_statistics();

    const size_t outputs_number = error_data_statistics.size();

    Vector< Matrix<double> > statistics(outputs_number);

    for(size_t i = 0; i < outputs_number; i++)
    {
        statistics[i].set(3, 4);
        statistics[i].set_row(0, error_data_statistics[i][0].to_vector());
        statistics[i].set_row(1, error_data_statistics[i][1].to_vector());
        statistics[i].set_row(2, error_data_statistics[i][2].to_vector());
    }

    return(statistics);
}


// Vector< Histogram<double> > calculate_error_data_histograms(const size_t&) const method

/// Calculates histograms for the relative errors of all the output variables.
/// The number of bins is set by the user.
/// @param bins_number Number of bins in the histograms.

Vector< Histogram<double> > TestingAnalysis::calculate_error_data_histograms(const size_t& bins_number) const
{
   const Vector< Vector<double> > error_data = calculate_percentage_error_data();

   const size_t outputs_number = error_data.size();

   Vector< Histogram<double> > histograms(outputs_number);

   for(size_t i = 0; i < outputs_number; i++)
   {
       histograms[i] = error_data[i].calculate_histogram_centered(0.0, bins_number);
   }

   return(histograms);
}


// Vector< Histogram<double> > calculate_forecasting_error_data_histograms(const size_t&) const method

/// Calculates histograms for the relative errors of all the output variables for a forecasting problem.
/// The number of bins is set by the user.
/// @param bins_number Number of bins in the histograms.

Vector< Histogram<double> > TestingAnalysis::calculate_forecasting_error_data_histograms(const size_t& bins_number) const
{
   const Vector< Matrix<double> > error_data = calculate_forecasting_error_data();

   const size_t outputs_number = error_data.size();

   Vector< Histogram<double> > histograms(outputs_number);

   for(size_t i = 0; i < outputs_number; i++)
   {
       histograms[i] = error_data[i].get_column(0).calculate_histogram(bins_number);
   }

   return(histograms);
}



// Vector< Vector<size_t> > calculate_maximal_errors(const size_t&) const method

/// Returns a vector with the indices of the instances which have the greatest error.
/// @param instances_number Size of the vector to be returned.
/// @todo Finish the method.

Vector< Vector<size_t> > TestingAnalysis::calculate_maximal_errors(const size_t& instances_number) const
{
    const Vector< Matrix<double> > error_data = calculate_error_data();

    const size_t outputs_number = error_data.size();

    Vector< Vector<size_t> > maximal_errors(outputs_number);

    for(size_t i = 0; i < outputs_number; i++)
    {
        maximal_errors[i] = error_data[i].get_column(0).calculate_maximal_indices(instances_number);
    }

    return(maximal_errors);
}


// Vector< Vector<size_t> > calculate_forecasting_maximal_errors(const size_t&) const method

/// Returns a vector with the indices of the instances which have the greatest error for a forecasting problem.
/// @param instances_number Size of the vector to be returned.
/// @todo Finish the method.

Vector< Vector<size_t> > TestingAnalysis::calculate_forecasting_maximal_errors(const size_t& instances_number) const
{
    const Vector< Matrix<double> > error_data = calculate_forecasting_error_data();

    const size_t outputs_number = error_data.size();

    Vector< Vector<size_t> > maximal_errors(outputs_number);

    for(size_t i = 0; i < outputs_number; i++)
    {
        maximal_errors[i] = error_data[i].get_column(0).calculate_maximal_indices(instances_number);
    }

    return(maximal_errors);
}


Matrix<double> TestingAnalysis::calculate_errors() const
{
    Matrix<double> errors(4,3,0.0);

    errors.set_column(0, calculate_training_errors(), "training_errors");
    errors.set_column(1, calculate_selection_errors(), "selection_errors");
    errors.set_column(2, calculate_testing_errors(), "testing_errors");

    return errors;
}


Matrix<double> TestingAnalysis::calculate_binary_classification_errors() const
{
    Matrix<double> errors(5,3,0.0);

    errors.set_column(0, calculate_binary_classification_training_errors(), "");
    errors.set_column(1, calculate_binary_classification_selection_errors(), "");
    errors.set_column(2, calculate_binary_classification_testing_errors(), "");

    return errors;
}

Matrix<double> TestingAnalysis::calculate_multiple_classification_errors() const
{
    Matrix<double> errors(5,3,0.0);

    errors.set_column(0, calculate_multiple_classification_training_errors(), "");
    errors.set_column(1, calculate_multiple_classification_selection_errors(), "");
    errors.set_column(2, calculate_multiple_classification_testing_errors(), "");

    return errors;
}

Vector<double> TestingAnalysis::calculate_training_errors() const
{
    // Data set stuff

    #ifdef __OPENNN_DEBUG__

    check();

    #endif

    const size_t training_instances_number = data_set_pointer->get_instances().get_training_instances_number();

     #ifdef __OPENNN_DEBUG__

     ostringstream buffer;

     if(training_instances_number == 0)
     {
        buffer << "OpenNN Exception: TestingAnalysis class.\n"
               << "Vector<double> calculate_training_errors() const.\n"
               << "Number of training instances is zero.\n";

        throw logic_error(buffer.str());
     }

     #endif

    const Matrix<double> inputs = data_set_pointer->get_training_inputs();

    const Matrix<double> targets = data_set_pointer->get_training_targets();

    // Neural network stuff

    const Matrix<double> outputs = neural_network_pointer->calculate_outputs(inputs);

    Vector<double> errors(4,0.0);

    // Results

    errors[0] = outputs.calculate_sum_squared_error(targets);
    errors[1] = errors[0]/training_instances_number;
    errors[2] = sqrt(errors[1]);
    errors[3] = calculate_testing_normalized_squared_error(targets, outputs);

    return errors;
}

Vector<double> TestingAnalysis::calculate_binary_classification_training_errors() const
{
    // Data set stuff

    #ifdef __OPENNN_DEBUG__

    check();

    #endif

    const size_t training_instances_number = data_set_pointer->get_instances().get_training_instances_number();

     #ifdef __OPENNN_DEBUG__

     ostringstream buffer;

     if(training_instances_number == 0)
     {
        buffer << "OpenNN Exception: TestingAnalysis class.\n"
               << "Vector<double> calculate_binary_classification_training_errors() const.\n"
               << "Number of training instances is zero.\n";

        throw logic_error(buffer.str());
     }

     #endif

    const Matrix<double> inputs = data_set_pointer->get_training_inputs();

    const Matrix<double> targets = data_set_pointer->get_training_targets();

    // Neural network stuff

    const Matrix<double> outputs = neural_network_pointer->calculate_outputs(inputs);

    Vector<double> errors(5,0.0);

    // Results

    errors[0] = outputs.calculate_sum_squared_error(targets);
    errors[1] = errors[0]/training_instances_number;
    errors[2] = sqrt(errors[1]);
    errors[3] = calculate_testing_normalized_squared_error(targets, outputs);
    errors[4] = calculate_testing_cross_entropy_error(targets, outputs);
//    errors[5] = calculate_testing_weighted_squared_error(targets, outputs);

    return errors;
}

Vector<double> TestingAnalysis::calculate_multiple_classification_training_errors() const
{
    // Data set stuff

    #ifdef __OPENNN_DEBUG__

    check();

    #endif

    const size_t training_instances_number = data_set_pointer->get_instances().get_training_instances_number();

     #ifdef __OPENNN_DEBUG__

     ostringstream buffer;

     if(training_instances_number == 0)
     {
        buffer << "OpenNN Exception: TestingAnalysis class.\n"
               << "Vector<double> calculate_multiple_classification_training_errors() const.\n"
               << "Number of training instances is zero.\n";

        throw logic_error(buffer.str());
     }

     #endif

    const Matrix<double> inputs = data_set_pointer->get_training_inputs();

    const Matrix<double> targets = data_set_pointer->get_training_targets();

    // Neural network stuff

    const Matrix<double> outputs = neural_network_pointer->calculate_outputs(inputs);

    Vector<double> errors(5,0.0);

    // Results

    errors[0] = outputs.calculate_sum_squared_error(targets);
    errors[1] = errors[0]/training_instances_number;
    errors[2] = sqrt(errors[1]);
    errors[3] = calculate_testing_normalized_squared_error(targets, outputs);
    errors[4] = calculate_testing_cross_entropy_error(targets, outputs);

    return errors;
}

Vector<double> TestingAnalysis::calculate_selection_errors() const
{
    // Data set stuff

    #ifdef __OPENNN_DEBUG__

    check();

    #endif

    const size_t selection_instances_number = data_set_pointer->get_instances().get_selection_instances_number();

     #ifdef __OPENNN_DEBUG__

     ostringstream buffer;

     if(selection_instances_number == 0)
     {
        buffer << "OpenNN Exception: TestingAnalysis class.\n"
               << "Vector<double> calculate_selection_errors() const.\n"
               << "Number of selection instances is zero.\n";

        throw logic_error(buffer.str());
     }

     #endif

    const Matrix<double> inputs = data_set_pointer->get_selection_inputs();

    const Matrix<double> targets = data_set_pointer->get_selection_targets();

    // Neural network stuff

    const Matrix<double> outputs = neural_network_pointer->calculate_outputs(inputs);

    Vector<double> errors(4,0.0);

    // Results

    errors[0] = outputs.calculate_sum_squared_error(targets);
    errors[1] = errors[0]/selection_instances_number;
    errors[2] = sqrt(errors[1]);
    errors[3] = calculate_testing_normalized_squared_error(targets, outputs);

    return errors;
}

Vector<double> TestingAnalysis::calculate_binary_classification_selection_errors() const
{
    // Data set stuff

    #ifdef __OPENNN_DEBUG__

    check();

    #endif

    const size_t selection_instances_number = data_set_pointer->get_instances().get_selection_instances_number();

     #ifdef __OPENNN_DEBUG__

     ostringstream buffer;

     if(selection_instances_number == 0)
     {
        buffer << "OpenNN Exception: TestingAnalysis class.\n"
               << "Vector<double> calculate_binary_classification_selection_errors() const.\n"
               << "Number of selection instances is zero.\n";

        throw logic_error(buffer.str());
     }

     #endif

    const Matrix<double> inputs = data_set_pointer->get_selection_inputs();

    const Matrix<double> targets = data_set_pointer->get_selection_targets();

    // Neural network stuff

    const Matrix<double> outputs = neural_network_pointer->calculate_outputs(inputs);

    Vector<double> errors(5,0.0);

    // Results

    errors[0] = outputs.calculate_sum_squared_error(targets);
    errors[1] = errors[0]/selection_instances_number;
    errors[2] = sqrt(errors[1]);
    errors[3] = calculate_testing_normalized_squared_error(targets, outputs);
    errors[4] = calculate_testing_cross_entropy_error(targets, outputs);
//    errors[5] = calculate_testing_weighted_squared_error(targets, outputs);

    return errors;
}

Vector<double> TestingAnalysis::calculate_multiple_classification_selection_errors() const
{
    // Data set stuff

    #ifdef __OPENNN_DEBUG__

    check();

    #endif

    const size_t selection_instances_number = data_set_pointer->get_instances().get_selection_instances_number();

     #ifdef __OPENNN_DEBUG__

     ostringstream buffer;

     if(selection_instances_number == 0)
     {
        buffer << "OpenNN Exception: TestingAnalysis class.\n"
               << "Vector<double> calculate_multiple_classification_selection_errors() const.\n"
               << "Number of selection instances is zero.\n";

        throw logic_error(buffer.str());
     }

     #endif

    const Matrix<double> inputs = data_set_pointer->get_selection_inputs();

    const Matrix<double> targets = data_set_pointer->get_selection_targets();

    // Neural network stuff

    const Matrix<double> outputs = neural_network_pointer->calculate_outputs(inputs);

    Vector<double> errors(5,0.0);

    // Results

    errors[0] = outputs.calculate_sum_squared_error(targets);
    errors[1] = errors[0]/selection_instances_number;
    errors[2] = sqrt(errors[1]);
    errors[3] = calculate_testing_normalized_squared_error(targets, outputs);
    errors[4] = calculate_testing_cross_entropy_error(targets, outputs);

    return errors;
}

// Vector<double> calculate_testing_errors() const method

/// Returns a vector containing the values of the errors between the outputs of the neural network
/// and the targets. The vector consists of:
/// <ul>
/// <li> Sum squared error.
/// <li> Mean squared error.
/// <li> Root mean squared error.
/// <li> Normalized squared error.
/// </ul>

Vector<double> TestingAnalysis::calculate_testing_errors() const
{
    // Data set stuff

    #ifdef __OPENNN_DEBUG__

    check();

    #endif

    const size_t testing_instances_number = data_set_pointer->get_instances().get_testing_instances_number();

     #ifdef __OPENNN_DEBUG__

     ostringstream buffer;

     if(testing_instances_number == 0)
     {
        buffer << "OpenNN Exception: TestingAnalysis class.\n"
               << "Vector< Matrix<double> > calculate_testing_errors() const.\n"
               << "Number of testing instances is zero.\n";

        throw logic_error(buffer.str());
     }

     #endif

    const Matrix<double> inputs = data_set_pointer->get_testing_inputs();

    const Matrix<double> targets = data_set_pointer->get_testing_targets();

    // Neural network stuff

    const Matrix<double> outputs = neural_network_pointer->calculate_outputs(inputs);

    #ifdef __OPENNN_DEBUG__   

//    if(!unscaling_layer_pointer)
//    {
//       buffer << "OpenNN Exception: TestingAnalysis class.\n"
//              << "Vector<double> calculate_testing_errors() const.\n"
//              << "Unscaling layer is nullptr.\n";

//       throw logic_error(buffer.str());
//    }

    #endif

    Vector<double> errors(4,0.0);

    // Results

    errors[0] = outputs.calculate_sum_squared_error(targets);
    errors[1] = errors[0]/testing_instances_number;
    errors[2] = sqrt(errors[1]);
    errors[3] = calculate_testing_normalized_squared_error(targets, outputs);

    return errors;
}


// Vector<double> calculate_forecasting_testing_errors() const method

/// Returns a vector containing the values of the errors between the outputs of the neural network
/// and the targets for a forecasting problem. The vector consists of:
/// <ul>
/// <li> Sum squared error.
/// <li> Mean squared error.
/// <li> Root mean squared error.
/// <li> Normalized squared error.
/// </ul>

Vector<double> TestingAnalysis::calculate_forecasting_testing_errors() const
{
    // Data set stuff

    #ifdef __OPENNN_DEBUG__

    check();

    #endif

    const size_t testing_instances_number = data_set_pointer->get_instances().get_testing_instances_number();

     #ifdef __OPENNN_DEBUG__

     ostringstream buffer;

     if(testing_instances_number == 0)
     {
        buffer << "OpenNN Exception: TestingAnalysis class.\n"
               << "Vector< Matrix<double> > calculate_testing_errors() const.\n"
               << "Number of testing instances is zero.\n";

        throw logic_error(buffer.str());
     }

     #endif
/*
    const Matrix<double> inputs = data_set_pointer->get_testing_inputs();

    const Matrix<double> targets = data_set_pointer->get_testing_targets();

    const Vector<double> time = data_set_pointer->get_testing_time();

    // Neural network stuff

    const Matrix<double> outputs = neural_network_pointer->calculate_outputs(inputs, time);

    #ifdef __OPENNN_DEBUG__

//    if(!unscaling_layer_pointer)
//    {
//       buffer << "OpenNN Exception: TestingAnalysis class.\n"
//              << "Vector<double> calculate_testing_errors() const.\n"
//              << "Unscaling layer is nullptr.\n";

//       throw logic_error(buffer.str());
//    }

    #endif

    Vector<double> errors(4,0.0);

    // Results

    errors[0] = outputs.calculate_sum_squared_error(targets);
    errors[1] = errors[0]/testing_instances_number;
    errors[2] = sqrt(errors[1]);
    errors[3] = calculate_testing_normalized_squared_error(targets, outputs);

    return errors;
*/

     return Vector<double>();
}


// Vector<double> calculate_binary_classification_testing_errors() const method

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

Vector<double> TestingAnalysis::calculate_binary_classification_testing_errors() const
{
    // Data set stuff

    #ifdef __OPENNN_DEBUG__

    check();

    #endif

    const size_t testing_instances_number = data_set_pointer->get_instances().get_testing_instances_number();

     #ifdef __OPENNN_DEBUG__

     ostringstream buffer;

     if(testing_instances_number == 0)
     {
        buffer << "OpenNN Exception: TestingAnalysis class.\n"
               << "Vector<double> calculate_binary_classification_testing_errors() const.\n"
               << "Number of testing instances is zero.\n";

        throw logic_error(buffer.str());
     }

     #endif

    const Matrix<double> inputs = data_set_pointer->get_testing_inputs();

    const Matrix<double> targets = data_set_pointer->get_testing_targets();

    // Neural network stuff

    const Matrix<double> outputs = neural_network_pointer->calculate_outputs(inputs);

    Vector<double> errors(5,0.0);

    // Results

    errors[0] = outputs.calculate_sum_squared_error(targets);
    errors[1] = errors[0]/testing_instances_number;
    errors[2] = sqrt(errors[1]);
    errors[3] = calculate_testing_normalized_squared_error(targets, outputs);
    errors[4] = calculate_testing_cross_entropy_error(targets, outputs);
//    errors[5] = calculate_testing_weighted_squared_error(targets, outputs);

    return errors;
}


// Vector<double> calculate_multiple_classification_testing_errors() const method

/// Returns a vector containing the values of the errors between the outputs of the neural network
/// and the targets for a multiple classification problem. The vector consists of:
/// <ul>
/// <li> Sum squared error.
/// <li> Mean squared error.
/// <li> Root mean squared error.
/// <li> Normalized squared error.
/// <li> Cross-entropy error.
/// </ul>

Vector<double> TestingAnalysis::calculate_multiple_classification_testing_errors() const
{
    // Data set stuff

    #ifdef __OPENNN_DEBUG__

    check();

    #endif

    const size_t testing_instances_number = data_set_pointer->get_instances().get_testing_instances_number();

     #ifdef __OPENNN_DEBUG__

     ostringstream buffer;

     if(testing_instances_number == 0)
     {
        buffer << "OpenNN Exception: TestingAnalysis class.\n"
               << "Vector<double> calculate_multiple_classification_testing_errors() const.\n"
               << "Number of testing instances is zero.\n";

        throw logic_error(buffer.str());
     }

     #endif

    const Matrix<double> inputs = data_set_pointer->get_testing_inputs();

    const Matrix<double> targets = data_set_pointer->get_testing_targets();

    // Neural network stuff

    const Matrix<double> outputs = neural_network_pointer->calculate_outputs(inputs);

    Vector<double> errors(5,0.0);

    // Results

    errors[0] = outputs.calculate_sum_squared_error(targets);
    errors[1] = errors[0]/testing_instances_number;
    errors[2] = sqrt(errors[1]);
    errors[3] = calculate_testing_normalized_squared_error(targets, outputs);
    errors[4] = calculate_testing_cross_entropy_error(targets, outputs);

    return errors;
}


// double calculate_testing_normalized_squared_error(const Matrix<double>&, const Matrix<double>&) const method

/// Returns the normalized squared error between the targets and the outputs of the neural network.
/// @param targets Testing target data.
/// @param outputs Testing output data.

double TestingAnalysis::calculate_testing_normalized_squared_error(const Matrix<double>& targets, const Matrix<double>& outputs) const
{
    const size_t testing_instances_number = targets.get_rows_number();

    const Vector<double> testing_targets_mean = targets.calculate_mean();

    double normalization_coefficient = 0.0;
    double sum_squared_error = 0.0;

//#pragma omp parallel for reduction(+:sum_squared_error, normalization_coefficient)

    for(int i = 0; i < (int)testing_instances_number; i++)
    {
        sum_squared_error += outputs.get_row(i).calculate_sum_squared_error(targets.get_row(i));

        normalization_coefficient += targets.get_row(i).calculate_sum_squared_error(testing_targets_mean);
    }

    return sum_squared_error/normalization_coefficient;
}


/// Returns the cross-entropy error between the targets and the outputs of the neural network.
/// It can only be computed for classification problems.
/// @param targets Testing target data.
/// @param outputs Testing output data.
/// @todo

double TestingAnalysis::calculate_testing_cross_entropy_error(const Matrix<double>& targets, const Matrix<double>& outputs) const
{
/*
    const size_t testing_instances_number = targets.get_rows_number();
    const size_t outputs_number = targets.get_columns_number();

    Vector<double> targets(outputs_number);
    Vector<double> outputs(outputs_number);

    double cross_entropy_error = 0.0;

//#pragma omp parallel for reduction(+:cross_entropy_error)

    for(int i = 0; i < (int)testing_instances_number; i++)
    {
        outputs = outputs.get_row(i);
        targets = targets.get_row(i);

        for(size_t j = 0; j < outputs_number; j++)
        {
            if(outputs[j] == 0.0)
            {
                outputs[j] = 1.0e-6;
            }
            else if(outputs[j] == 1.0)
            {
                outputs[j] = 0.999999;
            }

            cross_entropy_error -= targets[j]*log(outputs[j]) + (1.0 - targets[j])*log(1.0 - outputs[j]);
        }
    }

    return cross_entropy_error/(double)testing_instances_number;
*/
    return 0.0;
}


// double calculate_testing_weighted_squared_error(const Matrix<double>&, const Matrix<double>&) const method

/// Returns the weighted squared error between the targets and the outputs of the neural network. It can only be computed for
/// binary classification problems.
/// @param targets Testing target data.
/// @param outputs Testing output data.
/// @param weights Weights of the postitives and negatives to evaluate.

double TestingAnalysis::calculate_testing_weighted_squared_error(const Matrix<double>& targets, const Matrix<double>& outputs,
                                                                 const Vector<double>& weights) const
{
    const size_t testing_instances_number = targets.get_rows_number();

    #ifdef __OPENNN_DEBUG__

//    ostringstream buffer;

//    if(outputs_number != 1)
//    {
//       buffer << "OpenNN Exception: TestingAnalysis class.\n"
//              << "double calculate_testing_weighted_squared_error(const Matrix<double>&, const Matrix<double>&) const.\n"
//              << "Number of outputs must be one.\n";

//       throw logic_error(buffer.str());
//    }

    #endif

    double negatives_weight;
    double positives_weight;

    if(weights.size() != 2)
    {
        const Vector<size_t> target_distribution = data_set_pointer->calculate_target_distribution();

        const size_t negatives_number = target_distribution[0];
        const size_t positives_number = target_distribution[1];

        negatives_weight = 1.0;
        positives_weight = (double)negatives_number/positives_number;
    }
    else
    {
        positives_weight = weights[0];
        negatives_weight = weights[1];
    }

    double error = 0.0;
    double sum_squared_error = 0.0;

//#pragma omp parallel for private(error) reduction(+:sum_squared_error)

    for(int i = 0; i < (int)testing_instances_number; i++)
    {
        if(targets(i,0) == 1.0)
        {
            error = positives_weight*outputs.get_row(i).calculate_sum_squared_error(targets.get_row(i));
        }
        else if(targets(i,0) == 0.0)
        {
            error = negatives_weight*outputs.get_row(i).calculate_sum_squared_error(targets.get_row(i));
        }
        else
        {
            ostringstream buffer;

            buffer << "OpenNN Exception: TestingAnalysis class.\n"
                   << "double calculate_testing_weighted_squared_error(const Matrix<double>&, const Matrix<double>&) const method.\n"
                   << "Target is neither a positive nor a negative.\n";

            throw logic_error(buffer.str());
        }

        sum_squared_error += error;
    }

//    const Vector<size_t> targets_indices = data_set_pointer->get_variables().get_targets_indices();

//    const size_t negatives = data_set_pointer->calculate_training_negatives(targets_indices[0]);

    const size_t negatives = targets.get_column(0).count_equal_to(0.0);

    const double normalization_coefficient = negatives*negatives_weight*0.5;

    return(sum_squared_error/normalization_coefficient);
}

// Matrix<size_t> calculate_confusion_binary_classification(const Matrix<double>&, const Matrix<double>&, const double&) const method

/// Returns the confusion matrix for a binary classification problem.
/// @param targets Testing target data.
/// @param outputs Testing output data.
/// @param decision_threshold Decision threshold.

Matrix<size_t> TestingAnalysis::calculate_confusion_binary_classification(const Matrix<double>& targets, const Matrix<double>& outputs, const double& decision_threshold) const
{
    const size_t rows_number = targets.get_rows_number();

    Matrix<size_t> confusion(2, 2);

    size_t true_positive = 0;
    size_t false_negative = 0;
    size_t false_positive = 0;
    size_t true_negative = 0;        

    for(size_t i = 0; i < rows_number; i++)
    {
        if(decision_threshold == 0.0 && targets(i,0) == 0.0 )
        {
            false_positive++;

        }
        else if(decision_threshold == 0.0 && targets(i,0) == 1.0)
        {
            true_positive++;

        }
        else if(targets(i,0) >= decision_threshold && outputs(i,0) >= decision_threshold)
        {
            // True positive

            true_positive++;

        }
        else if(targets(i,0) >= decision_threshold && outputs(i,0) < decision_threshold)
        {
            // False negative

            false_negative++;

        }
        else if(targets(i,0) < decision_threshold && outputs(i,0) >= decision_threshold)
        {
            // False positive

            false_positive++;

        }
        else if(targets(i,0) < decision_threshold && outputs(i,0) < decision_threshold)
        {
            // True negative

            true_negative++;
        }
    }

    confusion(0,0) = true_positive;
    confusion(0,1) = false_negative;
    confusion(1,0) = false_positive;
    confusion(1,1) = true_negative;

    if(confusion.calculate_sum() != rows_number)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: TestingAnalysis class.\n"
               << "Matrix<size_t> calculate_confusion_binary_classification(const Matrix<double>&, const Matrix<double>&, const double&) const method.\n"
               << "Number of elements in confusion matrix must be equal to number of testing instances.\n";

        throw logic_error(buffer.str());
    }

    return(confusion);
}


// Matrix<size_t> calculate_confusion_multiple_classification(const Matrix<double>&, const Matrix<double>&) const method

/// Returns the confusion matrix for a binary classification problem.
/// @param targets Testing target data.
/// @param outputs Testing output data.

Matrix<size_t> TestingAnalysis::calculate_confusion_multiple_classification(const Matrix<double>& targets, const Matrix<double>& outputs) const
{
    const size_t rows_number = targets.get_rows_number();
    const size_t columns_number = targets.get_columns_number();

    Matrix<size_t> confusion(columns_number, columns_number, 0);

    size_t target_index = 0;
    size_t output_index = 0;

    for(size_t i = 0; i < rows_number; i++)
    {
        target_index = targets.get_row(i).calculate_maximal_index();
        output_index = outputs.get_row(i).calculate_maximal_index();

        confusion(target_index,output_index)++;
    }

    return(confusion);
}


// Vector<size_t> calculate_positives_negatives_rate(const Matrix<double>&, const Matrix<double>&) const

/// Returns a vector containing the number of total positives and the number of total negatives
/// instances of a data set.
/// The size of the vector is two and consists of:
/// <ul>
/// <li> Total positives
/// <li> Total negatives
/// </ul>

Vector<size_t> TestingAnalysis::calculate_positives_negatives_rate(const Matrix<double>& targets, const Matrix<double>& outputs) const
{
    const Matrix<size_t> confusion = calculate_confusion_binary_classification(targets, outputs, 0.5);
    Vector<size_t> positives_negatives_rate(2);

    positives_negatives_rate[0] = confusion(0,0) + confusion(0,1);
    positives_negatives_rate[1] = confusion(1,0) + confusion(1,1);

    return(positives_negatives_rate);
}


// Matrix<size_t> calculate_confusion() const method

/// Returns the confusion matrix of a neural network on the testing instances of a data set.
/// If the number of outputs is one, the size of the confusion matrix is two.
/// If the number of outputs is greater than one, the size of the confusion matrix is the number of outputs.

Matrix<size_t> TestingAnalysis::calculate_confusion() const
{
   #ifdef __OPENNN_DEBUG__

    check();

   #endif

   const MultilayerPerceptron* multilayer_perceptron_pointer = neural_network_pointer->get_multilayer_perceptron_pointer();

   #ifdef __OPENNN_DEBUG__

   if(!multilayer_perceptron_pointer)
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: TestingAnalysis class.\n"
             << "Matrix<size_t> calculate_confusion() const method.\n"
             << "Pointer to multilayer perceptron in neural network is nullptr.\n";

      throw logic_error(buffer.str());
   }

   #endif

   const size_t outputs_number = multilayer_perceptron_pointer->get_outputs_number();

   #ifdef __OPENNN_DEBUG__

   const size_t inputs_number = multilayer_perceptron_pointer->get_inputs_number();

   // Control sentence

   const Variables& variables = data_set_pointer->get_variables();

   if(inputs_number != variables.get_inputs_number())
   {      
       ostringstream buffer;

       buffer << "OpenNN Exception: TestingAnalysis class." << endl
              << "Matrix<size_t> calculate_confusion() const method." << endl
              << "Number of inputs in neural network must be equal to number of inputs in data set." << endl;

      throw logic_error(buffer.str());
   }

   if(outputs_number != variables.get_targets_number())
   {
       ostringstream buffer;

      buffer << "OpenNN Exception: TestingAnalysis class." << endl
             << "Matrix<size_t> calculate_confusion() const method." << endl
             << "Number of outputs in neural network must be equal to number of targets in data set." << endl;

      throw logic_error(buffer.str());
   }

   #endif

    const Matrix<double> inputs = data_set_pointer->get_testing_inputs();
    const Matrix<double> targets = data_set_pointer->get_testing_targets();    

    const Matrix<double> outputs = neural_network_pointer->calculate_outputs(inputs);

//    outputs.save("../data/outputs.dat");

    if(outputs_number == 1)
    {
        double decision_threshold;

        if(neural_network_pointer->get_probabilistic_layer_pointer() != nullptr)
        {
            decision_threshold = neural_network_pointer->get_probabilistic_layer_pointer()->get_decision_threshold();
        }
        else
        {
            decision_threshold = 0.5;
        }

        return(calculate_confusion_binary_classification(targets, outputs, decision_threshold));
    }
    else
    {
        return(calculate_confusion_multiple_classification(targets, outputs));
    }
}


//TestingAnalysis::RocCurveResults perform_roc_analysis() const

/// Performs a ROC curve analysis.
/// It returns a ROC curve analysis results structure, which consists of:
/// <ul>
/// <li> ROC curve
/// <li> Area under the ROC curve
/// <li> Optimal threshold
/// </ul>

TestingAnalysis::RocAnalysisResults TestingAnalysis::perform_roc_analysis() const
{
    #ifdef __OPENNN_DEBUG__

    check();

    #endif

    #ifdef __OPENNN_DEBUG__

    const MultilayerPerceptron* multilayer_perceptron_pointer = neural_network_pointer->get_multilayer_perceptron_pointer();

    if(!multilayer_perceptron_pointer)
    {
        ostringstream buffer;

       buffer << "OpenNN Exception: TestingAnalysis class.\n"
              << "RocCurveResults perform_roc_analysis() const method.\n"
              << "Pointer to multilayer perceptron in neural network is nullptr.\n";

      throw logic_error(buffer.str());
    }

    #endif

    #ifdef __OPENNN_DEBUG__

    const size_t inputs_number = multilayer_perceptron_pointer->get_inputs_number();

    // Control sentence

    const Variables& variables = data_set_pointer->get_variables();

    if(inputs_number != variables.get_inputs_number())
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: TestingAnalysis class." << endl
               << "RocCurveResults perform_roc_analysis() const method." << endl
               << "Number of inputs in neural network must be equal to number of inputs in data set." << endl;

       throw logic_error(buffer.str());
    }

    const size_t outputs_number = multilayer_perceptron_pointer->get_outputs_number();

    if(outputs_number != variables.get_targets_number())
    {
        ostringstream buffer;

       buffer << "OpenNN Exception: TestingAnalysis class." << endl
              << "RocCurveResults perform_roc_analysis() const method." << endl
              << "Number of outputs in neural network must be equal to number of targets in data set." << endl;

       throw logic_error(buffer.str());
    }

    #endif

     const Matrix<double> inputs = data_set_pointer->get_testing_inputs();
     const Matrix<double> targets = data_set_pointer->get_testing_targets();

     const Matrix<double> outputs = neural_network_pointer->calculate_outputs(inputs);

     RocAnalysisResults roc_analysis_results;

     roc_analysis_results.roc_curve = calculate_roc_curve(targets, outputs);
     roc_analysis_results.area_under_curve = calculate_area_under_curve(targets, outputs);
     roc_analysis_results.confidence_limit = calculate_area_under_curve_confidence_limit(targets, outputs, roc_analysis_results.area_under_curve);
     roc_analysis_results.optimal_threshold = calculate_optimal_threshold(targets, outputs, roc_analysis_results.roc_curve);

     return(roc_analysis_results);
}


//double calculate_Wilcoxon_parameter(const double& ,const double& ) const

/// Calculates the Wilcoxon parameter, which is used for calculating the area under a ROC curve.
/// Returns 1 if first value is greater than second one, 0 if second value is greater than first one or 0.5 in other case.
/// @param x Target data value.
/// @param y Target data value.

double TestingAnalysis::calculate_Wilcoxon_parameter(const double& x, const double& y) const
{
    if(x > y)
    {
        return(1);
    }
    else if(x < y)
    {
        return(0);
    }
    else
    {
        return(0.5);
    }
}


// Matrix<double> calculate_roc_curve(const Matrix<double>&, const Matrix<double>& outputs) const

/// Returns a matrix with the values of a ROC curve for a binary classification problem.
/// The number of columns is three. The third column contains the decision threshold.
/// The number of rows is one more than the number of outputs if the number of outputs is lower than 100
/// or 50 in other case.
/// @param targets Testing target data.
/// @param outputs Testing output data.

Matrix<double> TestingAnalysis::calculate_roc_curve(const Matrix<double>& targets, const Matrix<double>& outputs) const
{
    const Vector<size_t> positives_negatives_rate = calculate_positives_negatives_rate(targets, outputs);

    const size_t total_positives = positives_negatives_rate[0];
    const size_t total_negatives = positives_negatives_rate[1];

    if(total_positives == 0)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: TestingAnalysis class.\n"
               << "Matrix<double> calculate_roc_curve(const Matrix<double>&, const Matrix<double>&) const.\n"
               << "Number of positive instances("<< total_positives <<") must be greater than zero.\n";

        throw logic_error(buffer.str());
     }

    if(total_negatives == 0)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: TestingAnalysis class.\n"
               << "Matrix<double> calculate_roc_curve(const Matrix<double>&, const Matrix<double>&) const.\n"
               << "Number of negative instances("<< total_negatives <<") must be greater than zero.\n";

        throw logic_error(buffer.str());
     }

    const size_t maximum_points_number = 1000;

    size_t step_size;

    const size_t testing_instances_number = targets.get_rows_number();
    size_t points_number;

    if(testing_instances_number > maximum_points_number)
    {
        step_size = (size_t)((double)testing_instances_number/(double)maximum_points_number);
        points_number = (size_t)((double)testing_instances_number/(double)step_size);
    }
    else
    {
        points_number = testing_instances_number;
        step_size = 1;
    }

    Matrix<double> target_outputs = outputs.assemble_columns(targets);

    Matrix<double> sorted_target_outputs = target_outputs.sort_ascending(0);

    const Vector<size_t> columns_output_indices(1,0);
    const Vector<size_t> columns_target_indices(1,1);

    const Matrix<double> sorted_targets = sorted_target_outputs.get_submatrix_columns(columns_target_indices);
    const Matrix<double> sorted_outputs = sorted_target_outputs.get_submatrix_columns(columns_output_indices);

    Matrix<double> roc_curve(points_number+1, 3, 0.0);

    const size_t step_s = step_size;

    #pragma omp parallel for schedule(dynamic)

    for(int i = 0; i < (int)points_number; i++)
    {
        size_t positives = 0;
        size_t negatives = 0;

        const size_t current_index = i*step_s;

        const double threshold = sorted_outputs(current_index, 0);

        for(int j = 0; j < (int)current_index; j++)
        {
             if(sorted_outputs(j,0) < threshold && sorted_targets(j,0) == 1.0)
             {
                 positives++;
             }
             if(sorted_outputs(j,0) < threshold && sorted_targets(j,0) == 0.0)
             {
                 negatives++;
             }
        }

        roc_curve(i,0) = (double)positives/(double)(total_positives);
        roc_curve(i,1) = (double)negatives/(double)(total_negatives);
        roc_curve(i,2) = (double)threshold;
    }

    roc_curve(points_number, 0) = 1.0;
    roc_curve(points_number, 1) = 1.0;
    roc_curve(points_number, 2) = 1.0;

    return(roc_curve);
}


// double calculate_area_under_curve(const Matrix<double>&, const Matrix<double>& ) const

/// Returns the area under a ROC curve.
/// @param targets Testing target data.
/// @param outputs Testing output data.

double TestingAnalysis::calculate_area_under_curve(const Matrix<double>& targets, const Matrix<double>& outputs) const
{
    const Vector<size_t> positives_negatives_rate = calculate_positives_negatives_rate(targets, outputs);

    const size_t total_positives = positives_negatives_rate[0];
    const size_t total_negatives = positives_negatives_rate[1];

    if(total_positives == 0)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: TestingAnalysis class.\n"
               << "Matrix<double> calculate_roc_curve(const Matrix<double>&, const Matrix<double>&) const.\n"
               << "Number of positive instances("<< total_positives <<") must be greater than zero.\n";

        throw logic_error(buffer.str());
     }

    if(total_negatives == 0)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: TestingAnalysis class.\n"
               << "Matrix<double> calculate_roc_curve(const Matrix<double>&, const Matrix<double>&) const.\n"
               << "Number of negative instances("<< total_negatives <<") must be greater than zero.\n";

        throw logic_error(buffer.str());
     }

    size_t testing_instances_number = targets.get_rows_number();

    double sum = 0.0;

    double area_under_curve;

    #pragma omp parallel for reduction(+ : sum) schedule(dynamic)

    for(int i = 0; i < (int)testing_instances_number; i++)
    {
        if(targets(i,0) == 1)
        {
            for(int j = 0; j < (int)testing_instances_number; j++)
            {
                if(targets(j,0) == 0)
                {
                   sum += calculate_Wilcoxon_parameter(outputs(i,0),outputs(j,0));
                }
            }
        }
    }

    area_under_curve = static_cast<double>(sum)/(double)(total_positives*total_negatives);

    return(area_under_curve);
}



// double calculate_area_under_curve_confidence_limit(const Matrix<double>&, const Matrix<double>& ) const

/// Returns the confidence limit for the area under a roc curve.
/// @param targets Testing target data.
/// @param outputs Testing output data.

double TestingAnalysis::calculate_area_under_curve_confidence_limit(const Matrix<double>& targets, const Matrix<double>& outputs) const
{
    const Vector<size_t> positives_negatives_rate = calculate_positives_negatives_rate(targets, outputs);

    const size_t total_positives = positives_negatives_rate[0];
    const size_t total_negatives = positives_negatives_rate[1];

    if(total_positives == 0)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: TestingAnalysis class.\n"
               << "Matrix<double> calculate_roc_curve_confidence_limit(const Matrix<double>&, const Matrix<double>&) const.\n"
               << "Number of positive instances("<< total_positives <<") must be greater than zero.\n";

        throw logic_error(buffer.str());
     }

    if(total_negatives == 0)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: TestingAnalysis class.\n"
               << "Matrix<double> calculate_roc_curve_confidence_limit(const Matrix<double>&, const Matrix<double>&) const.\n"
               << "Number of negative instances("<< total_negatives <<") must be greater than zero.\n";

        throw logic_error(buffer.str());
     }

    const double area_under_curve = calculate_area_under_curve(targets, outputs);

    const double Q_1 = area_under_curve/(2.0-area_under_curve);
    const double Q_2 = (2.0*area_under_curve*area_under_curve)/(1.0*area_under_curve);

    const double confidence_limit = 1.64485*sqrt((area_under_curve*(1.0-area_under_curve)
                                                + (total_positives-1.0)*(Q_1-area_under_curve*area_under_curve)
                                                + (total_negatives-1.0)*(Q_2-area_under_curve*area_under_curve))/(total_positives*total_negatives));

    return(confidence_limit);
}


// double calculate_area_under_curve_confidence_limit(const Matrix<double>&, const Matrix<double>&, const double&) const

/// Returns the confidence limit for the area under a roc curve.
/// @param targets Testing target data.
/// @param outputs Testing output data.
/// @param area_under_curve Area under curve.

double TestingAnalysis::calculate_area_under_curve_confidence_limit(const Matrix<double>& targets, const Matrix<double>& outputs, const double& area_under_curve) const
{
    const Vector<size_t> positives_negatives_rate = calculate_positives_negatives_rate(targets, outputs);

    const size_t total_positives = positives_negatives_rate[0];
    const size_t total_negatives = positives_negatives_rate[1];

    if(total_positives == 0)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: TestingAnalysis class.\n"
               << "Matrix<double> calculate_roc_curve_confidence_limit(const Matrix<double>&, const Matrix<double>&, const double&) const.\n"
               << "Number of positive instances("<< total_positives <<") must be greater than zero.\n";

        throw logic_error(buffer.str());
     }

    if(total_negatives == 0)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: TestingAnalysis class.\n"
               << "Matrix<double> calculate_roc_curve_confidence_limit(const Matrix<double>&, const Matrix<double>&, const double&) const.\n"
               << "Number of negative instances("<< total_negatives <<") must be greater than zero.\n";

        throw logic_error(buffer.str());
     }

    const double Q_1 = area_under_curve/(2.0-area_under_curve);
    const double Q_2 = (2.0*area_under_curve*area_under_curve)/(1.0*area_under_curve);

    const double confidence_limit = 1.64485*sqrt((area_under_curve*(1.0-area_under_curve)
                                                + (total_positives-1.0)*(Q_1-area_under_curve*area_under_curve)
                                                + (total_negatives-1.0)*(Q_2-area_under_curve*area_under_curve))/(total_positives*total_negatives));

    return(confidence_limit);
}


// double calculate_optimal_threshold(const Matrix<double>&, const Matrix<double>& ) const

/// Returns the point of optimal classification accuracy, which is the nearest ROC curve point to the upper left corner(0,1).
/// @param targets Testing target data.
/// @param outputs Testing output data.

double TestingAnalysis::calculate_optimal_threshold(const Matrix<double>& targets, const Matrix<double>& outputs ) const
{
    const size_t columns_number = targets.get_columns_number();

    const size_t maximum_points_number = 1000;

    size_t step_size;

    const size_t testing_instances_number = targets.get_rows_number();
    size_t points_number;

    if(testing_instances_number > maximum_points_number)
    {
        step_size = (size_t)((double)testing_instances_number/(double)maximum_points_number);
        points_number = (size_t)((double)testing_instances_number/(double)step_size);
    }
    else
    {
        points_number = testing_instances_number;
        step_size = 1;
    }

    Matrix<double> target_outputs = outputs.assemble_columns(targets);

    Matrix<double> sorted_target_outputs = target_outputs.sort_ascending(0);

    const Vector<size_t> columns_output_indices(0, 1, columns_number - 1);
    const Vector<size_t> columns_target_indices(columns_number, 1, 2*columns_number - 1);

    const Matrix<double> sorted_targets = sorted_target_outputs.get_submatrix_columns(columns_target_indices);
    const Matrix<double> sorted_outputs = sorted_target_outputs.get_submatrix_columns(columns_output_indices);

    const Matrix<double> roc_curve = calculate_roc_curve(sorted_targets, sorted_outputs);

    double threshold = 0.0;
    double optimal_threshold = 0.5;

    double minimun_distance = numeric_limits<double>::max();
    double distance;

    size_t current_index;

    for(size_t i = 0; i < points_number; i++)
    {
        current_index = i*step_size;

        threshold = sorted_outputs(current_index, 0);

        distance = sqrt(roc_curve(i,0)*roc_curve(i,0) + (roc_curve(i,1) - 1.0)*(roc_curve(i,1) - 1.0));

        if(distance < minimun_distance)
        {
            optimal_threshold = threshold;

            minimun_distance = distance;
        }
    }

    return(optimal_threshold);
}

// double calculate_optimal_threshold(const Matrix<double>&, const Matrix<double>&, const Matrix<double>&) const

/// Returns the point of optimal classification accuracy, which is the nearest ROC curve point to the upper left corner(0,1).
/// @param targets Testing target data.
/// @param outputs Testing output data.
/// @param roc_curve ROC curve.

double TestingAnalysis::calculate_optimal_threshold(const Matrix<double>& targets, const Matrix<double>& outputs, const Matrix<double>& roc_curve) const
{
    const size_t columns_number = targets.get_columns_number();

    const size_t maximum_points_number = 1000;

    size_t step_size;

    const size_t testing_instances_number = targets.get_rows_number();
    size_t points_number;

    if(testing_instances_number > maximum_points_number)
    {
        step_size = (size_t)((double)testing_instances_number/(double)maximum_points_number);
        points_number = (size_t)((double)testing_instances_number/(double)step_size);
    }
    else
    {
        points_number = testing_instances_number;
        step_size = 1;
    }

    const Matrix<double> target_outputs = outputs.assemble_columns(targets);

    const Matrix<double> sorted_target_outputs = target_outputs.sort_ascending(0);

    const Vector<size_t> columns_output_indices(0, 1, columns_number - 1);

    const Matrix<double> sorted_outputs = sorted_target_outputs.get_submatrix_columns(columns_output_indices);

    double threshold = 0.0;
    double optimal_threshold = 0.5;

    double minimun_distance = numeric_limits<double>::max();
    double distance;

    size_t current_index;

    for(size_t i = 0; i < points_number; i++)
    {
        current_index = i*step_size;

        threshold = sorted_outputs(current_index, 0);

        distance = sqrt(roc_curve(i,0)*roc_curve(i,0) + (roc_curve(i,1) - 1.0)*(roc_curve(i,1) - 1.0));

        if(distance < minimun_distance)
        {
            optimal_threshold = threshold;

            minimun_distance = distance;
        }
    }

    return(optimal_threshold);
}

// Matrix<double> perform_cumulative_gain_analysis() const

/// Performs a cumulative gain analysis.
/// Returns a matrix with the values of a cumulative gain chart.

Matrix<double> TestingAnalysis::perform_cumulative_gain_analysis() const
{
    #ifdef __OPENNN_DEBUG__

    check();

    #endif

    #ifdef __OPENNN_DEBUG__

    const MultilayerPerceptron* multilayer_perceptron_pointer = neural_network_pointer->get_multilayer_perceptron_pointer();

    if(!multilayer_perceptron_pointer)
    {
        ostringstream buffer;

       buffer << "OpenNN Exception: TestingAnalysis class.\n"
              << "Matrix<double> perform_cumulative_gain_analysis() const method.\n"
              << "Pointer to multilayer perceptron in neural network is nullptr.\n";

      throw logic_error(buffer.str());
    }

    #endif

    #ifdef __OPENNN_DEBUG__

    const size_t inputs_number = multilayer_perceptron_pointer->get_inputs_number();

    // Control sentence

    const Variables& variables = data_set_pointer->get_variables();

    if(inputs_number != variables.get_inputs_number())
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: TestingAnalysis class." << endl
               << "Matrix<double> perform_cumulative_gain_analysis() const method." << endl
               << "Number of inputs in neural network must be equal to number of inputs in data set." << endl;

       throw logic_error(buffer.str());
    }

    const size_t outputs_number = multilayer_perceptron_pointer->get_outputs_number();

    if(outputs_number != variables.get_targets_number())
    {
        ostringstream buffer;

       buffer << "OpenNN Exception: TestingAnalysis class." << endl
              << "Matrix<double> perform_cumulative_gain_analysis() const method." << endl
              << "Number of outputs in neural network must be equal to number of targets in data set." << endl;

       throw logic_error(buffer.str());
    }

    #endif

     const Matrix<double> inputs = data_set_pointer->get_testing_inputs();
     const Matrix<double> targets = data_set_pointer->get_testing_targets();

     const Matrix<double> outputs = neural_network_pointer->calculate_outputs(inputs);

     const Matrix<double> cumulative_gain = calculate_cumulative_gain(targets, outputs);

     return(cumulative_gain);
}


// Matrix<double> calculate_cumulative_gain(const Matrix<double>&, const Matrix<double>&) const

/// Returns a matrix with the values of a cumulative gain chart.
/// The number of columns is two, the number of rows is 20.
/// @param targets Testing target data.
/// @param outputs Testing output data.

Matrix<double> TestingAnalysis::calculate_cumulative_gain(const Matrix<double>& targets, const Matrix<double>& outputs) const
{
    const size_t total_positives = calculate_positives_negatives_rate(targets, outputs)[0];

    if(total_positives == 0)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: TestingAnalysis class.\n"
               << "Matrix<double> calculate_cumulative_gain(const Matrix<double>&, const Matrix<double>&) const.\n"
               << "Number of positive instances(" << total_positives << ") must be greater than zero.\n";

        throw logic_error(buffer.str());
     }

    const size_t rows_number = targets.get_rows_number();

    const Matrix<double> target_outputs = outputs.assemble_columns(targets);

    const Matrix<double> sorted_target_outputs = target_outputs.sort_descending(0);

    const Vector<size_t> target_indices(1,1);

    const Matrix<double> sorted_targets = sorted_target_outputs.get_submatrix_columns(target_indices);

    const size_t points_number = 21;
    const double percentage_increment = 0.05;

    Matrix<double> cumulative_gain(points_number, 2);

    cumulative_gain(0,0) = 0.0;
    cumulative_gain(0,1) = 0.0;

    size_t positives = 0;
    size_t negatives = 0;

    double percentage = 0.0;

    size_t maximum_index;

    for(size_t i = 0; i < points_number - 1; i++)
    {
        percentage += percentage_increment;
        positives = 0;
        negatives = 0;
        maximum_index = (size_t)(percentage*rows_number);

        for(size_t j = 0; j < maximum_index; j++)
        {
            if(sorted_targets(j, 0) == 1.0)
            {
                 positives++;
            }
        }

        cumulative_gain(i + 1, 0) = (double) percentage;
        cumulative_gain(i + 1, 1) = (double) positives/(double)(total_positives);
    }

    return(cumulative_gain);
}


// Matrix<double> calculate_cumulative_gain(const Matrix<double>&, const Matrix<double>&) const

/// Returns a matrix with the values of a cumulative gain chart for the negative instances.
/// The number of columns is two, the number of rows is 20.
/// @param targets Testing target data.
/// @param outputs Testing output data.

Matrix<double> TestingAnalysis::calculate_negative_cumulative_gain(const Matrix<double>& targets, const Matrix<double>& outputs) const
{
    const size_t total_negatives = calculate_positives_negatives_rate(targets, outputs)[1];

    if(total_negatives == 0)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: TestingAnalysis class.\n"
               << "Matrix<double> calculate_negative_cumulative_gain(const Matrix<double>&, const Matrix<double>&) const.\n"
               << "Number of negative instances(" << total_negatives << ") must be greater than zero.\n";

        throw logic_error(buffer.str());
     }

    const size_t rows_number = targets.get_rows_number();

    const Matrix<double> target_outputs = outputs.assemble_columns(targets);

    const Matrix<double> sorted_target_outputs = target_outputs.sort_descending(0);

    const Vector<size_t> target_indices(1,1);

    const Matrix<double> sorted_targets = sorted_target_outputs.get_submatrix_columns(target_indices);

    const size_t points_number = 21;
    const double percentage_increment = 0.05;

    Matrix<double> negative_cumulative_gain(points_number, 2);

    negative_cumulative_gain(0,0) = 0.0;
    negative_cumulative_gain(0,1) = 0.0;

    size_t negatives = 0;

    double percentage = 0.0;

    size_t maximum_index;

    for(size_t i = 0; i < points_number - 1; i++)
    {
        percentage += percentage_increment;
        negatives = 0;
        maximum_index = (size_t)(percentage*rows_number);

        for(size_t j = 0; j < maximum_index; j++)
        {
            if(sorted_targets(j, 0) == 0.0)
            {
                 negatives++;
            }
        }

        negative_cumulative_gain(i + 1, 0) = (double) percentage;
        negative_cumulative_gain(i + 1, 1) = (double) negatives/(double)(total_negatives);
    }

    return(negative_cumulative_gain);
}


// Matrix<double> perform_lift_chart_analysis() const

/// Performs a lift chart analysis.
/// Returns a matrix with the values of a lift chart.

Matrix<double> TestingAnalysis::perform_lift_chart_analysis() const
{
    #ifdef __OPENNN_DEBUG__

    check();

    #endif

    #ifdef __OPENNN_DEBUG__

    const MultilayerPerceptron* multilayer_perceptron_pointer = neural_network_pointer->get_multilayer_perceptron_pointer();

    if(!multilayer_perceptron_pointer)
    {
        ostringstream buffer;

       buffer << "OpenNN Exception: TestingAnalysis class.\n"
              << "Matrix<double> perform_lift_chart_analysis() const method.\n"
              << "Pointer to multilayer perceptron in neural network is nullptr.\n";

      throw logic_error(buffer.str());
    }

    #endif

    #ifdef __OPENNN_DEBUG__

    const size_t inputs_number = multilayer_perceptron_pointer->get_inputs_number();

    // Control sentence

    const Variables& variables = data_set_pointer->get_variables();

    if(inputs_number != variables.get_inputs_number())
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: TestingAnalysis class." << endl
               << "Matrix<double> perform_lift_chart_analysis() const method." << endl
               << "Number of inputs in neural network must be equal to number of inputs in data set." << endl;

       throw logic_error(buffer.str());
    }

    const size_t outputs_number = multilayer_perceptron_pointer->get_outputs_number();

    if(outputs_number != variables.get_targets_number())
    {
        ostringstream buffer;

       buffer << "OpenNN Exception: TestingAnalysis class." << endl
              << "Matrix<double> perform_lift_chart_analysis() const method." << endl
              << "Number of outputs in neural network must be equal to number of targets in data set." << endl;

       throw logic_error(buffer.str());
    }

    #endif

     const Matrix<double> inputs = data_set_pointer->get_testing_inputs();
     const Matrix<double> targets = data_set_pointer->get_testing_targets();

     const Matrix<double> outputs = neural_network_pointer->calculate_outputs(inputs);

     const Matrix<double> cumulative_gain = calculate_cumulative_gain(targets, outputs);
     const Matrix<double> lift_chart = calculate_lift_chart(cumulative_gain);

     return(lift_chart);
}


// Matrix<double> calculate_lift_chart(const Matrix<double>&, const Matrix<double>&) const

/// Returns a matrix with the values of lift chart for a given cumulative gain chart.
/// Size is the same as the cumulative lift chart one.
/// @param cumulative_gain A cumulative gain chart.

Matrix<double> TestingAnalysis::calculate_lift_chart(const Matrix<double>& cumulative_gain) const
{
    const size_t rows_number = cumulative_gain.get_rows_number();
    const size_t columns_number = cumulative_gain.get_columns_number();

    Matrix<double> lift_chart(rows_number, columns_number);

    lift_chart(0,0) = 0.0;
    lift_chart(0,1) = 1.0;

#pragma omp parallel for

    for(int i = 1; i < static_cast<int>(rows_number); i++)
    {
        lift_chart(i, 0) = cumulative_gain(i, 0);
        lift_chart(i, 1) = (double) cumulative_gain(i, 1)/(double)cumulative_gain(i, 0);
    }

    return(lift_chart);
}


// Matrix<double> perform_Kolmogorov_Smirnov_analysis() const

/// Performs a Kolmogorov-Smirnov analysis, which consists of the cumulative gain for the positive instances and the cumulative
/// gain for the negative instances. It returns a Kolmogorov-Smirnov results structure, which consists of:
/// <ul>
/// <li> Positive cumulative gain
/// <li> Negative cumulative gain
/// <li> Maximum gain
/// </ul>

TestingAnalysis::KolmogorovSmirnovResults TestingAnalysis::perform_Kolmogorov_Smirnov_analysis() const
{
    #ifdef __OPENNN_DEBUG__

    check();

    #endif

    #ifdef __OPENNN_DEBUG__

    const MultilayerPerceptron* multilayer_perceptron_pointer = neural_network_pointer->get_multilayer_perceptron_pointer();

    if(!multilayer_perceptron_pointer)
    {
        ostringstream buffer;

       buffer << "OpenNN Exception: TestingAnalysis class.\n"
              << "Matrix<double> perform_Kolmogorov_Smirnov_analysis() const method.\n"
              << "Pointer to multilayer perceptron in neural network is nullptr.\n";

      throw logic_error(buffer.str());
    }

    #endif

    #ifdef __OPENNN_DEBUG__

    const size_t inputs_number = multilayer_perceptron_pointer->get_inputs_number();

    // Control sentence

    const Variables& variables = data_set_pointer->get_variables();

    if(inputs_number != variables.get_inputs_number())
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: TestingAnalysis class." << endl
               << "Matrix<double> perform_Kolmogorov_Smirnov_analysis() const method." << endl
               << "Number of inputs in neural network must be equal to number of inputs in data set." << endl;

       throw logic_error(buffer.str());
    }

    const size_t outputs_number = multilayer_perceptron_pointer->get_outputs_number();

    if(outputs_number != variables.get_targets_number())
    {
        ostringstream buffer;

       buffer << "OpenNN Exception: TestingAnalysis class." << endl
              << "Matrix<double> perform_Kolmogorov_Smirnov_analysis() const method." << endl
              << "Number of outputs in neural network must be equal to number of targets in data set." << endl;

       throw logic_error(buffer.str());
    }

    #endif

     const Matrix<double> inputs = data_set_pointer->get_testing_inputs();
     const Matrix<double> targets = data_set_pointer->get_testing_targets();

     const Matrix<double> outputs = neural_network_pointer->calculate_outputs(inputs);

     TestingAnalysis::KolmogorovSmirnovResults Kolmogorov_Smirnov_results;

     Kolmogorov_Smirnov_results.positive_cumulative_gain = calculate_cumulative_gain(targets, outputs);
     Kolmogorov_Smirnov_results.negative_cumulative_gain = calculate_negative_cumulative_gain(targets, outputs);
     Kolmogorov_Smirnov_results.maximum_gain =
     calculate_maximum_gain(Kolmogorov_Smirnov_results.positive_cumulative_gain,Kolmogorov_Smirnov_results.negative_cumulative_gain);

     return(Kolmogorov_Smirnov_results);
}


// double calculate_Komogorov_Smirnov_score(const Matrix<double>&, const Matrix<double>&) const

/// Returns the score of the the maximum gain, which is the point of major separation between the positive and
/// the negative cumulative gain charts, and the instances ratio for which it occurs.
/// @param positive_cumulative_gain Cumulative gain fo the positive instances.
/// @param negative_cumulative_gain Cumulative gain fo the negative instances.

Vector<double> TestingAnalysis::calculate_maximum_gain(const Matrix<double>& positive_cumulative_gain, const Matrix<double>& negative_cumulative_gain) const
{
    const size_t points_number = positive_cumulative_gain.get_rows_number();

    #ifdef __OPENNN_DEBUG__

    if(points_number != negative_cumulative_gain.get_rows_number())
    {
        ostringstream buffer;

       buffer << "OpenNN Exception: TestingAnalysis class.\n"
              << "Matrix<double> calculate_maximum_gain() const method.\n"
              << "Positive and negative cumulative gain matrix must have the same rows number.\n";

      throw logic_error(buffer.str());
    }

    #endif

    Vector<double> maximum_gain(2, 0.0);

    const double percentage_increment = 0.05;

    double percentage = 0.0;

    for(size_t i = 0; i < points_number - 1; i++)
    {
        percentage += percentage_increment;

        if(positive_cumulative_gain(i+1,1)-negative_cumulative_gain(i+1,1) > maximum_gain[1]
        && positive_cumulative_gain(i+1,1)-negative_cumulative_gain(i+1,1) > 0.0)
        {
            maximum_gain[1] = positive_cumulative_gain(i+1,1)-negative_cumulative_gain(i+1,1);
            maximum_gain[0] = percentage;
        }
    }

    return maximum_gain;
}


// Matrix<double> perform_calibration_plot_analysis() const

/// Performs a calibration plot analysis.

Matrix<double> TestingAnalysis::perform_calibration_plot_analysis() const
{
    #ifdef __OPENNN_DEBUG__

    check();

    #endif

    #ifdef __OPENNN_DEBUG__

    const MultilayerPerceptron* multilayer_perceptron_pointer = neural_network_pointer->get_multilayer_perceptron_pointer();

    if(!multilayer_perceptron_pointer)
    {
        ostringstream buffer;

       buffer << "OpenNN Exception: TestingAnalysis class.\n"
              << "Matrix<double> perform_calibration_plot_analysis() const method.\n"
              << "Pointer to multilayer perceptron in neural network is nullptr.\n";

      throw logic_error(buffer.str());
    }

    #endif

    #ifdef __OPENNN_DEBUG__

    const size_t inputs_number = multilayer_perceptron_pointer->get_inputs_number();

    // Control sentence

    const Variables& variables = data_set_pointer->get_variables();

    if(inputs_number != variables.get_inputs_number())
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: TestingAnalysis class." << endl
               << "Matrix<double> perform_calibration_plot_analysis() const method." << endl
               << "Number of inputs in neural network must be equal to number of inputs in data set." << endl;

       throw logic_error(buffer.str());
    }

    const size_t outputs_number = multilayer_perceptron_pointer->get_outputs_number();

    if(outputs_number != variables.get_targets_number())
    {
        ostringstream buffer;

       buffer << "OpenNN Exception: TestingAnalysis class." << endl
              << "Matrix<double> perform_calibration_plot_analysis() const method." << endl
              << "Number of outputs in neural network must be equal to number of targets in data set." << endl;

       throw logic_error(buffer.str());
    }

    #endif

    const Matrix<double> inputs = data_set_pointer->get_testing_inputs();
    const Matrix<double> targets = data_set_pointer->get_testing_targets();

    const Matrix<double> outputs = neural_network_pointer->calculate_outputs(inputs);

    const Matrix<double> calibration_plot = calculate_calibration_plot(targets, outputs);

    return(calibration_plot);

}

//Matrix<double> calculate_calibration_plot(const Matrix<double>&, const Matrix<double>&) const

/// Returns a matix with the values of a calibration plot.
/// Number of columns is two. Number of rows varies depending on the distribution of positive targets.
/// @param targets Testing target data.
/// @param outputs Testing output data.

Matrix<double> TestingAnalysis::calculate_calibration_plot(const Matrix<double>& targets, const Matrix<double>& outputs) const
{
    cout << "Calibration plot" << endl;

    const size_t rows_number = targets.get_rows_number();   

    cout << "Rows number: " << rows_number << endl;

    const size_t points_number = 10;

    Matrix<double> calibration_plot(points_number+2, 2);

    // First point

    calibration_plot(0,0) = 0.0;
    calibration_plot(0,1) = 0.0;

    size_t positives = 0;

    size_t count = 0;

    double probability = 0.0;

    double sum = 0.0;

    for(size_t i = 1; i < points_number+1; i++)
    {
        count = 0;
        positives = 0;
        sum = 0.0;
        probability += 0.1;

        for(size_t j = 0; j < rows_number; j++)
        {
            if(outputs(j, 0) >= (probability - 0.1) && outputs(j, 0) < probability)
            {
                count++;

                sum += outputs(j, 0);

                if(targets(j, 0) == 1.0)
                {
                    positives++;
                }
            }
        }

        if(count == 0)
        {
            calibration_plot(i, 0) = -1;
            calibration_plot(i, 1) = -1;
        }
        else
        {
            calibration_plot(i, 0) = static_cast<double>(sum)/(double)count;
            calibration_plot(i, 1) = (double)positives/(double)count;
        }
     }

    // Last point

    calibration_plot(points_number+1,0) = 1.0;
    calibration_plot(points_number+1,1) = 1.0;

   // Subtracts calibration plot rows with value -1

    size_t points_number_subtracted = 0;

    size_t current_rows_number;

    while(calibration_plot.get_column(0).contains(-1))
     {
        current_rows_number = calibration_plot.get_rows_number();

         for(size_t i = 1; i < points_number - points_number_subtracted+1; i++)
         {
             if(calibration_plot(i, 0) == -1)
             {
                 calibration_plot = calibration_plot.delete_row(i);

                 points_number_subtracted++;
             }
         }
     }

    return(calibration_plot);
}


// Vector< Histogram <double> > calculate_output_histogram(const Matrix<double>&, const size_t&) const;

/// Returns the histogram of the outputs.
/// @param outputs Testing output data.
/// @param bins_number Number of bins of the histogram.

Vector< Histogram <double> > TestingAnalysis::calculate_output_histogram(const Matrix<double>& outputs, const size_t& bins_number) const
{
    Vector< Histogram <double> > output_histogram(1);

    output_histogram [0] = outputs.get_column(0).calculate_histogram(bins_number);

    return(output_histogram);
}


// TestingAnalysis::BinaryClassificationRates TestingAnalysis::calculate_binary_classification_rates() const

/// Returns a structure with the binary classification rates, which has the indices of the true positive, false negative, false positive and true negative instances.
/// <ul>
/// <li> True positive instances
/// <li> False positive instances
/// <li> False negative instances
/// <li> True negative instances
/// </ul>

TestingAnalysis::BinaryClassifcationRates TestingAnalysis::calculate_binary_classification_rates() const
{
    #ifdef __OPENNN_DEBUG__

     check();

    #endif

    #ifdef __OPENNN_DEBUG__

    const MultilayerPerceptron* multilayer_perceptron_pointer = neural_network_pointer->get_multilayer_perceptron_pointer();

    if(!multilayer_perceptron_pointer)
    {
        ostringstream buffer;

       buffer << "OpenNN Exception: TestingAnalysis class.\n"
              << "BinaryClassificationRates calculate_binary_classification_rates() const method.\n"
              << "Pointer to multilayer perceptron in neural network is nullptr.\n";

       throw logic_error(buffer.str());
    }

    #endif

    #ifdef __OPENNN_DEBUG__

    const size_t inputs_number = multilayer_perceptron_pointer->get_inputs_number();

    // Control sentence

    const Variables& variables = data_set_pointer->get_variables();

    if(inputs_number != variables.get_inputs_number())
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: TestingAnalysis class." << endl
               << "BinaryClassificationRates calculate_binary_classification_rates() const method." << endl
               << "Number of inputs in neural network must be equal to number of inputs in data set." << endl;

       throw logic_error(buffer.str());
    }

    const size_t outputs_number = multilayer_perceptron_pointer->get_outputs_number();

    if(outputs_number != variables.get_targets_number())
    {
        ostringstream buffer;

       buffer << "OpenNN Exception: TestingAnalysis class." << endl
              << "BinaryClassificationRates calculate_binary_classification_rates() const method." << endl
              << "Number of outputs in neural network must be equal to number of targets in data set." << endl;

       throw logic_error(buffer.str());
    }

    #endif

    const Matrix<double> inputs = data_set_pointer->get_testing_inputs();
    const Matrix<double> targets = data_set_pointer->get_testing_targets();

    const Matrix<double> outputs = neural_network_pointer->calculate_outputs(inputs);

    const Vector<size_t> testing_indices = data_set_pointer->get_instances().get_testing_indices();

    double decision_threshold;

    if(neural_network_pointer->get_probabilistic_layer_pointer() != nullptr)
    {
        decision_threshold = neural_network_pointer->get_probabilistic_layer_pointer()->get_decision_threshold();
    }
    else
    {
        decision_threshold = 0.5;
    }

    BinaryClassifcationRates binary_classification_rates;

    binary_classification_rates.true_positive_instances = calculate_true_positive_instances(targets, outputs, testing_indices, decision_threshold);
    binary_classification_rates.false_positive_instances = calculate_false_positive_instances(targets, outputs, testing_indices, decision_threshold);
    binary_classification_rates.false_negative_instances = calculate_false_negative_instances(targets, outputs, testing_indices, decision_threshold);
    binary_classification_rates.true_negative_instances = calculate_true_negative_instances(targets, outputs, testing_indices, decision_threshold);

    return(binary_classification_rates);
}


// Vector<size_t> calculate_true_positive_instances(const Matrix<double>&, const Matrix<double>&, const Vector<size_t>&, const double&) const

/// Returns a vector with the indices of the true positive instances.
/// The size of the vector is the number of true positive instances.
/// @param targets Testing target data.
/// @param outputs Testing outputs.
/// @param testing_indices Indices of testing data.
/// @param decision_threshold Decision threshold.

Vector<size_t> TestingAnalysis::calculate_true_positive_instances(const Matrix<double>& targets, const Matrix<double>& outputs, const Vector<size_t>& testing_indices, const double& decision_threshold) const
{
    const size_t rows_number = targets.get_rows_number();

    Vector<size_t> true_positive_instances;

    for(size_t i = 0; i < rows_number; i++)
    {
        if(targets(i,0) >= decision_threshold && outputs(i,0) >= decision_threshold)
        {
            true_positive_instances.push_back(testing_indices[i]);
        }
    }

    return(true_positive_instances);
}


// Vector<size_t> calculate_false_positive_instances(const Matrix<double>&, const Matrix<double>&, const Vector<size_t>&, const double&) const

/// Returns a vector with the indices of the false positive instances.
/// The size of the vector is the number of false positive instances.
/// @param targets Testing target data.
/// @param outputs Testing output data.
/// @param testing_indices Indices of the testing data
/// @param decision_threshold Decision threshold.

Vector<size_t> TestingAnalysis::calculate_false_positive_instances(const Matrix<double>& targets, const Matrix<double>& outputs, const Vector<size_t>& testing_indices, const double& decision_threshold) const
{
    const size_t rows_number = targets.get_rows_number();

    Vector<size_t> false_positive_instances;

    for(size_t i = 0; i < rows_number; i++)
    {
        if(targets(i,0) < decision_threshold && outputs(i,0) >= decision_threshold)
        {
            false_positive_instances.push_back(testing_indices[i]);
        }
    }

    return(false_positive_instances);
}


//Vector<size_t> calculate_false_negative_instances(const Matrix<double>&, const Matrix<double>&, const Vector<size_t>&, const double&)

/// Returns a vector with the indices of the false negative instances.
/// The size of the vector is the number of false negative instances.
/// @param targets Testing target data.
/// @param outputs Testing output data.
/// @param testing_indices Indices of the testing data
/// @param decision_threshold Decision threshold.

Vector<size_t> TestingAnalysis::calculate_false_negative_instances(const Matrix<double>& targets, const Matrix<double>& outputs, const Vector<size_t>& testing_indices, const double& decision_threshold) const
{
    const size_t rows_number = targets.get_rows_number();

    Vector<size_t> false_negative_instances;

    for(size_t i = 0; i < rows_number; i++)
    {
        if(targets(i,0) > decision_threshold && outputs(i,0) < decision_threshold)
        {
            false_negative_instances.push_back(testing_indices[i]);
        }
    }

    return(false_negative_instances);
}


// Vector<size_t> calculate_true_negative_instances(const Matrix<double>&, const Matrix<double>&, const Vector<size_t>&, const double&) const

/// Returns a vector with the indices of the true negative instances.
/// The size of the vector is the number of true negative instances.
/// @param targets Testing target data.
/// @param outputs Testinga output data.
/// @param testing_indices Indices of the testing data
/// @param decision_threshold Decision threshold.

Vector<size_t> TestingAnalysis::calculate_true_negative_instances(const Matrix<double>& targets, const Matrix<double>& outputs, const Vector<size_t>& testing_indices, const double& decision_threshold) const
{
    Vector<size_t> true_negative_instances;

    const size_t rows_number = targets.get_rows_number();

    for(size_t i = 0; i < rows_number; i++)
    {
        if(targets(i,0) < decision_threshold && outputs(i,0) < decision_threshold)
        {
            true_negative_instances.push_back(testing_indices[i]);
        }
    }

    return(true_negative_instances);
}


// Matrix< Vector<size_t> > calculate_multiple_classification_rates() const

/// Returns a matrix of subvectors which have the rates for a multiple classification problem.

Matrix< Vector<size_t> > TestingAnalysis::calculate_multiple_classification_rates() const
{
    #ifdef __OPENNN_DEBUG__

     check();

    #endif

    #ifdef __OPENNN_DEBUG__

    const MultilayerPerceptron* multilayer_perceptron_pointer = neural_network_pointer->get_multilayer_perceptron_pointer();

    if(!multilayer_perceptron_pointer)
    {
    ostringstream buffer;

    buffer << "OpenNN Exception: TestingAnalysis class.\n"
          << "BinaryClassificationRates calculate_binary_classification_rates() const method.\n"
          << "Pointer to multilayer perceptron in neural network is nullptr.\n";

    throw logic_error(buffer.str());
    }

    #endif

    #ifdef __OPENNN_DEBUG__

    const size_t inputs_number = multilayer_perceptron_pointer->get_inputs_number();

    // Control sentence

    const Variables& variables = data_set_pointer->get_variables();

    if(inputs_number != variables.get_inputs_number())
    {
    ostringstream buffer;

    buffer << "OpenNN Exception: TestingAnalysis class." << endl
           << "BinaryClassificationRates calculate_binary_classification_rates() const method." << endl
           << "Number of inputs in neural network must be equal to number of inputs in data set." << endl;

    throw logic_error(buffer.str());
    }

    const size_t outputs_number = multilayer_perceptron_pointer->get_outputs_number();

    if(outputs_number != variables.get_targets_number())
    {
    ostringstream buffer;

    buffer << "OpenNN Exception: TestingAnalysis class." << endl
          << "BinaryClassificationRates calculate_binary_classification_rates() const method." << endl
          << "Number of outputs in neural network must be equal to number of targets in data set." << endl;

    throw logic_error(buffer.str());
    }

    #endif

    const Matrix<double> inputs = data_set_pointer->get_testing_inputs();
    const Matrix<double> targets = data_set_pointer->get_testing_targets();

    const Matrix<double> outputs = neural_network_pointer->calculate_outputs(inputs);

    const Vector<size_t> testing_indices = data_set_pointer->get_instances().get_testing_indices();

    return(calculate_multiple_classification_rates(targets, outputs, testing_indices));
}


// Matrix< Vector<size_t> > calculate_multiple_classification_rates() method

/// Returns a matrix of subvectors which have the rates for a multiple classification problem.
/// The number of rows of the matrix is the number targets.
/// The number of columns of the matrix is the number of columns of the target data.

Matrix< Vector<size_t> > TestingAnalysis::calculate_multiple_classification_rates(const Matrix<double>& targets, const Matrix<double>& outputs, const Vector<size_t>& testing_indices) const
{
    const size_t rows_number = targets.get_rows_number();
    const size_t columns_number = outputs.get_columns_number();

    Matrix< Vector<size_t> > multiple_classification_rates(rows_number, columns_number);

    size_t target_index;
    size_t output_index;

    for(size_t i = 0; i < rows_number; i++)
    {
        target_index = targets.get_row(i).calculate_maximal_index();
        output_index = outputs.get_row(i).calculate_maximal_index();

        multiple_classification_rates(target_index, output_index).push_back(testing_indices[i]);
    }

    return(multiple_classification_rates);
}


// Vector< Vector<double> > calculate_error_autocorrelation(const size_t&) const

/// Calculates error autocorrelation across varying lags.
/// Returns a vector of subvectors.
/// The size of the vector is the number of targets.
/// The size of the subvectors is the number of lags for which autocorrelation is calculated.
/// @param maximum_lags_number Number of lags for which error autocorrelation is to be calculated.

Vector< Vector<double> > TestingAnalysis::calculate_error_autocorrelation(const size_t& maximum_lags_number) const
{
    #ifdef __OPENNN_DEBUG__

     check();

    #endif

    #ifdef __OPENNN_DEBUG__

    const MultilayerPerceptron* multilayer_perceptron_pointer = neural_network_pointer->get_multilayer_perceptron_pointer();

    if(!multilayer_perceptron_pointer)
    {
    ostringstream buffer;

    buffer << "OpenNN Exception: TestingAnalysis class.\n"
          << "Vector<double> calculate_error_autocorrelation() const method.\n"
          << "Pointer to multilayer perceptron in neural network is nullptr.\n";

    throw logic_error(buffer.str());
    }

    #endif

    #ifdef __OPENNN_DEBUG__

    const size_t inputs_number = multilayer_perceptron_pointer->get_inputs_number();

    // Control sentence

    const Variables& variables = data_set_pointer->get_variables();

    if(inputs_number != variables.get_inputs_number())
    {
    ostringstream buffer;

    buffer << "OpenNN Exception: TestingAnalysis class." << endl
           << "Vector<double> calculate_error_autocorrelation() const method." << endl
           << "Number of inputs in neural network must be equal to number of inputs in data set." << endl;

    throw logic_error(buffer.str());
    }

    const size_t outputs_number = multilayer_perceptron_pointer->get_outputs_number();

    if(outputs_number != variables.get_targets_number())
    {
    ostringstream buffer;

    buffer << "OpenNN Exception: TestingAnalysis class." << endl
          << "Vector<double> calculate_error_autocorrelation() const method." << endl
          << "Number of outputs in neural network must be equal to number of targets in data set." << endl;

    throw logic_error(buffer.str());
    }

    #endif

    const Matrix<double> inputs = data_set_pointer->get_testing_inputs();
    const Matrix<double> targets = data_set_pointer->get_testing_targets();

    const Matrix<double> outputs = neural_network_pointer->calculate_outputs(inputs);

    const size_t targets_number = targets.get_columns_number();

    Vector< Vector<double> > error_autocorrelations(targets_number);

    Matrix<double> error = targets - outputs;

    for(size_t i = 0; i < targets_number; i++)
    {
        error_autocorrelations[i] = CorrelationAnalysis::calculate_autocorrelations(error.get_column(i), maximum_lags_number);
    }

    return error_autocorrelations;
}


/// Calculates error autocorrelation across varying lags.
/// Returns a vector of subvectors.
/// The size of the vector is the number of targets.
/// The size of the subvectors is the number of lags for which autocorrelation is calculated.
/// @param maximum_lags_number Number of lags for which error autocorrelation is to be calculated.

Vector< Vector<double> > TestingAnalysis::calculate_forecasting_error_autocorrelation(const size_t& maximum_lags_number) const
{
    #ifdef __OPENNN_DEBUG__

     check();

    #endif

    #ifdef __OPENNN_DEBUG__

    const MultilayerPerceptron* multilayer_perceptron_pointer = neural_network_pointer->get_multilayer_perceptron_pointer();

    if(!multilayer_perceptron_pointer)
    {
    ostringstream buffer;

    buffer << "OpenNN Exception: TestingAnalysis class.\n"
          << "Vector<double> calculate_forecasting_error_autocorrelation() const method.\n"
          << "Pointer to multilayer perceptron in neural network is nullptr.\n";

    throw logic_error(buffer.str());
    }

    #endif

    #ifdef __OPENNN_DEBUG__

    const size_t inputs_number = multilayer_perceptron_pointer->get_inputs_number();

    // Control sentence

    const Variables& variables = data_set_pointer->get_variables();

    if(inputs_number != variables.get_inputs_number())
    {
    ostringstream buffer;

    buffer << "OpenNN Exception: TestingAnalysis class." << endl
           << "Vector<double> calculate_forecasting_error_autocorrelation() const method." << endl
           << "Number of inputs in neural network must be equal to number of inputs in data set." << endl;

    throw logic_error(buffer.str());
    }

    const size_t outputs_number = multilayer_perceptron_pointer->get_outputs_number();

    if(outputs_number != variables.get_targets_number())
    {
    ostringstream buffer;

    buffer << "OpenNN Exception: TestingAnalysis class." << endl
          << "Vector<double> calculate_forecasting_error_autocorrelation() const method." << endl
          << "Number of outputs in neural network must be equal to number of targets in data set." << endl;

    throw logic_error(buffer.str());
    }

    #endif

    const Matrix<double> inputs = data_set_pointer->get_testing_inputs();
    const Matrix<double> targets = data_set_pointer->get_testing_targets();
    //const Vector<double> time = data_set_pointer->get_testing_time();

    const Matrix<double> outputs = neural_network_pointer->calculate_outputs(inputs);

    const size_t targets_number = targets.get_columns_number();

    Vector< Vector<double> > error_autocorrelations(targets_number);

    const Matrix<double> error = targets - outputs;

    for(size_t i = 0; i < targets_number; i++)
    {
        error_autocorrelations[i] = CorrelationAnalysis::calculate_autocorrelations(error.get_column(i), maximum_lags_number);
    }

    return error_autocorrelations;
}


/// Calculates the correlation between input and error.
/// Returns a vector of subvectors.
/// The size of the vector is the number of targets.
/// The size of the subvectors is the number of lags for which cross-correlation is calculated.
/// @param maximum_lags_number Number of lags for which cross-correlation is calculated.

Vector< Vector<double> > TestingAnalysis::calculate_input_error_cross_correlation(const size_t& maximum_lags_number) const
{
    #ifdef __OPENNN_DEBUG__

     check();

    #endif

    #ifdef __OPENNN_DEBUG__

    const MultilayerPerceptron* multilayer_perceptron_pointer = neural_network_pointer->get_multilayer_perceptron_pointer();

    if(!multilayer_perceptron_pointer)
    {
    ostringstream buffer;

    buffer << "OpenNN Exception: TestingAnalysis class.\n"
          << "Vector<double> calculate_input_error_cross_correlation() const method.\n"
          << "Pointer to multilayer perceptron in neural network is nullptr.\n";

    throw logic_error(buffer.str());
    }

    #endif

    #ifdef __OPENNN_DEBUG__

    const size_t inputs_number = multilayer_perceptron_pointer->get_inputs_number();

    // Control sentence

    const Variables& variables = data_set_pointer->get_variables();

    if(inputs_number != variables.get_inputs_number())
    {
    ostringstream buffer;

    buffer << "OpenNN Exception: TestingAnalysis class." << endl
           << "Vector<double> calculate_input_error_cross_correlation() const method." << endl
           << "Number of inputs in neural network must be equal to number of inputs in data set." << endl;

    throw logic_error(buffer.str());
    }

    const size_t outputs_number = multilayer_perceptron_pointer->get_outputs_number();

    if(outputs_number != variables.get_targets_number())
    {
    ostringstream buffer;

    buffer << "OpenNN Exception: TestingAnalysis class." << endl
          << "Vector<double> calculate_input_error_cross_correlation() const method." << endl
          << "Number of outputs in neural network must be equal to number of targets in data set." << endl;

    throw logic_error(buffer.str());
    }

    #endif

    const Matrix<double> inputs = data_set_pointer->get_testing_inputs();
    const Matrix<double> targets = data_set_pointer->get_testing_targets();

    const Matrix<double> outputs = neural_network_pointer->calculate_outputs(inputs);

    const size_t targets_number = targets.get_columns_number();

    const Matrix<double> error = targets - outputs;

    Vector<double> input_column;

    Vector< Vector<double> > input_error_cross_correlation(targets_number);

    for(size_t i = 0; i < targets_number; i++)
    {
        input_column = inputs.get_column(i);

//        input_error_cross_correlation[i] = input_column.calculate_cross_correlation(error.get_column(i), maximum_lags_number);
        input_error_cross_correlation[i] = CorrelationAnalysis::calculate_cross_correlations(input_column, error.get_column(i), maximum_lags_number);
    }

    return(input_error_cross_correlation);
}


/// Calculates the correlation between input and error.
/// Returns a vector of subvectors.
/// The size of the vector is the number of targets.
/// The size of the subvectors is the number of lags for which cross-correlation is calculated.
/// @param maximum_lags_number Number of lags for which cross-correlation is calculated.

Vector< Vector<double> > TestingAnalysis::calculate_forecasting_input_error_cross_correlation(const size_t& maximum_lags_number) const
{
    #ifdef __OPENNN_DEBUG__

     check();

    #endif

    #ifdef __OPENNN_DEBUG__

    const MultilayerPerceptron* multilayer_perceptron_pointer = neural_network_pointer->get_multilayer_perceptron_pointer();

    if(!multilayer_perceptron_pointer)
    {
    ostringstream buffer;

    buffer << "OpenNN Exception: TestingAnalysis class.\n"
          << "Vector<double> calculate_forecasting_input_error_cross_correlation() const method.\n"
          << "Pointer to multilayer perceptron in neural network is nullptr.\n";

    throw logic_error(buffer.str());
    }

    #endif

    #ifdef __OPENNN_DEBUG__

    const size_t inputs_number = multilayer_perceptron_pointer->get_inputs_number();

    // Control sentence

    const Variables& variables = data_set_pointer->get_variables();

    if(inputs_number != variables.get_inputs_number())
    {
    ostringstream buffer;

    buffer << "OpenNN Exception: TestingAnalysis class." << endl
           << "Vector<double> calculate_forecasting_input_error_cross_correlation() const method." << endl
           << "Number of inputs in neural network must be equal to number of inputs in data set." << endl;

    throw logic_error(buffer.str());
    }

    const size_t outputs_number = multilayer_perceptron_pointer->get_outputs_number();

    if(outputs_number != variables.get_targets_number())
    {
    ostringstream buffer;

    buffer << "OpenNN Exception: TestingAnalysis class." << endl
          << "Vector<double> calculate_forecasting_input_error_cross_correlation() const method." << endl
          << "Number of outputs in neural network must be equal to number of targets in data set." << endl;

    throw logic_error(buffer.str());
    }

    #endif

    const Matrix<double> inputs = data_set_pointer->get_testing_inputs();
    const Matrix<double> targets = data_set_pointer->get_testing_targets();
    const Vector<double> time = data_set_pointer->get_testing_time();

    const Matrix<double> outputs = neural_network_pointer->calculate_outputs(inputs, time[0]);

    const size_t targets_number = targets.get_columns_number();

    const Matrix<double> error = targets - outputs;

    Vector<double> input_column;

    Vector< Vector<double> > input_error_cross_correlation(targets_number);

    for(size_t i = 0; i < targets_number; i++)
    {
        input_column = inputs.get_column(i);

//        input_error_cross_correlation[i] = input_column.calculate_cross_correlation(error.get_column(i), maximum_lags_number);
        input_error_cross_correlation[i] = CorrelationAnalysis::calculate_cross_correlations(input_column, error.get_column(i), maximum_lags_number);
    }

    return(input_error_cross_correlation);


//    return Vector< Vector<double> >();
}



Vector< Vector<double> > TestingAnalysis::calculate_forecasting_time_series() const
{
/*
    #ifdef __OPENNN_DEBUG__

     check();

    #endif

    #ifdef __OPENNN_DEBUG__

    const MultilayerPerceptron* multilayer_perceptron_pointer = neural_network_pointer->get_multilayer_perceptron_pointer();

    if(!multilayer_perceptron_pointer)
    {
    ostringstream buffer;

    buffer << "OpenNN Exception: TestingAnalysis class.\n"
          << "Vector<double> calculate_forecasting_input_error_cross_correlation() const method.\n"
          << "Pointer to multilayer perceptron in neural network is nullptr.\n";

    throw logic_error(buffer.str());
    }

    #endif

    #ifdef __OPENNN_DEBUG__

    const size_t inputs_number = multilayer_perceptron_pointer->get_inputs_number();

    // Control sentence

    const Variables& variables = data_set_pointer->get_variables();

    if(inputs_number != variables.get_inputs_number())
    {
    ostringstream buffer;

    buffer << "OpenNN Exception: TestingAnalysis class." << endl
           << "Vector<double> calculate_forecasting_input_error_cross_correlation() const method." << endl
           << "Number of inputs in neural network must be equal to number of inputs in data set." << endl;

    throw logic_error(buffer.str());
    }

    const size_t outputs_number = multilayer_perceptron_pointer->get_outputs_number();

    if(outputs_number != variables.get_targets_number())
    {
    ostringstream buffer;

    buffer << "OpenNN Exception: TestingAnalysis class." << endl
          << "Vector<double> calculate_forecasting_input_error_cross_correlation() const method." << endl
          << "Number of outputs in neural network must be equal to number of targets in data set." << endl;

    throw logic_error(buffer.str());
    }

    #endif

    const Matrix<double> inputs = data_set_pointer->get_testing_inputs();
    const Matrix<double> targets = data_set_pointer->get_testing_targets();
    const Vector<double> time = data_set_pointer->get_testing_time();

    const Matrix<double> outputs = neural_network_pointer->calculate_outputs(inputs, time);

    const size_t targets_number = targets.get_columns_number();

    const Matrix<double> error = targets - outputs;

    Vector<double> input_column;

    Vector< Vector<double> > input_error_cross_correlation(targets_number);

    for(size_t i = 0; i < targets_number; i++)
    {
        input_column = inputs.get_column(i);

        input_error_cross_correlation[i] = input_column.calculate_cross_correlation(error.get_column(i), maximum_lags_number);
    }

    return(input_error_cross_correlation);
*/

    Vector< Vector<double> > time_series;

    return time_series;

}



// Vector<double> calculate_binary_classification_tests() method

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

Vector<double> TestingAnalysis::calculate_binary_classification_tests() const
{
   // Control sentence(if debug)

   #ifdef __OPENNN_DEBUG__

   const MultilayerPerceptron* multilayer_perceptron_pointer = neural_network_pointer->get_multilayer_perceptron_pointer();

   const size_t inputs_number = multilayer_perceptron_pointer->get_inputs_number();

   if(!data_set_pointer)
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: TestingAnalysis class." << endl
             << "Vector<double> calculate_binary_classification_tests() const." << endl
             << "Data set is nullptr." << endl;

      throw logic_error(buffer.str());
   }

   const Variables& variables = data_set_pointer->get_variables();

   const size_t targets_number = variables.get_targets_number();

   const size_t outputs_number = multilayer_perceptron_pointer->get_outputs_number();

   // Control sentence

   if(inputs_number != variables.get_inputs_number())
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: TestingAnalysis class." << endl
             << "Vector<double> calculate_binary_classification_tests() const." << endl
             << "Number of inputs in neural network is not equal to number of inputs in data set." << endl;

      throw logic_error(buffer.str());
   }
   else if(outputs_number != 1)
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: TestingAnalysis class." << endl
             << "Vector<double> calculate_binary_classification_tests() const." << endl
             << "Number of outputs in neural network must be one." << endl;

      throw logic_error(buffer.str());
   }
   else if(targets_number != 1)
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: TestingAnalysis class." << endl
             << "Vector<double> calculate_binary_classification_tests() const." << endl
             << "Number of targets in data set must be one." << endl;

      throw logic_error(buffer.str());
   }

   #endif

   // Confusion matrix

   const Matrix<size_t> confusion = calculate_confusion();

   const size_t true_positive = confusion(0,0);
   const size_t false_positive = confusion(1,0);
   const size_t false_negative = confusion(0,1);
   const size_t true_negative = confusion(1,1);

   // Classification accuracy

   double classification_accuracy;

   if(true_positive + true_negative + false_positive + false_negative == 0)
   {
       classification_accuracy = 0.0;
   }
   else
   {
       classification_accuracy = (double)(true_positive + true_negative)/(double)(true_positive + true_negative + false_positive + false_negative);
   }

   // Error rate

   double error_rate;

   if(true_positive + true_negative + false_positive + false_negative == 0)
   {
       error_rate = 0.0;
   }
   else
   {
       error_rate = (double)(false_positive + false_negative)/(double)(true_positive + true_negative + false_positive + false_negative);
   }

   // Sensitivity

   double sensitivity;

   if(true_positive + false_negative == 0)
   {
       sensitivity = 0.0;
   }
   else
   {
       sensitivity = (double)true_positive/(double)(true_positive + false_negative);
   }

   // Specificity

   double specificity;

   if(true_negative + false_positive == 0)
   {
       specificity = 0.0;
   }
   else
   {
       specificity = (double)true_negative/(double)(true_negative + false_positive);
   }

   // Precision

   double precision;

   if(true_positive + false_positive == 0)
   {
       precision = 0.0;
   }
   else
   {
      precision = (double) true_positive /(double)(true_positive + false_positive);
   }

   // Positive likelihood

   double positive_likelihood;

   if(classification_accuracy == 1.0)
   {
       positive_likelihood = 1.0;
   }
   else if(1.0 - specificity == 0.0)
   {
       positive_likelihood = 0.0;
   }
   else
   {
       positive_likelihood = sensitivity/(1.0 - specificity);
   }

   // Negative likelihood

   double negative_likelihood;

   if(classification_accuracy == 1.0)
   {
       negative_likelihood = 1.0;
   }
   else if(1.0 - sensitivity == 0.0)
   {
       negative_likelihood = 0.0;
   }
   else
   {
       negative_likelihood = specificity/(1.0 - sensitivity);
   }

   // F1 score

   double F1_score;

   if(2*true_positive + false_positive + false_negative == 0)
   {
       F1_score = 0.0;
   }
   else
   {
       F1_score = (double) 2*true_positive/(double)(2*true_positive + false_positive + false_negative);
   }

   // False positive rate

   double false_positive_rate;

   if(false_positive + true_negative == 0)
   {
       false_positive_rate = 0.0;
   }
   else
   {
       false_positive_rate = (double) false_positive/(double)(false_positive + true_negative);
   }

   // False discovery rate

   double false_discovery_rate;

   if(false_positive + true_positive == 0)
   {
       false_discovery_rate = 0.0;
   }
   else
   {
       false_discovery_rate = (double) false_positive /(double)(false_positive + true_positive);
   }

   // False negative rate

   double false_negative_rate;

   if(false_negative + true_positive == 0)
   {
       false_negative_rate = 0.0;
   }
   else
   {
       false_negative_rate = (double) false_negative /(double)(false_negative + true_positive);
   }

   // Negative predictive value

   double negative_predictive_value;

   if(true_negative + false_negative == 0)
   {
       negative_predictive_value = 0.0;
   }
   else
   {
       negative_predictive_value = (double) true_negative/(double)(true_negative + false_negative);
   }

   //Matthews correlation coefficient

   double Matthews_correlation_coefficient;

   if((true_positive + false_positive) *(true_positive + false_negative) *(true_negative + false_positive) *(true_negative + false_negative) == 0)
   {
       Matthews_correlation_coefficient = 0.0;
   }
   else
   {
       Matthews_correlation_coefficient = (double)(true_positive * true_negative - false_positive * false_negative) /(double) sqrt((double)((true_positive + false_positive) *(true_positive + false_negative) *(true_negative + false_positive) *(true_negative + false_negative)));
   }

   //Informedness

   double informedness = sensitivity + specificity - 1;

   //Markedness

   double markedness;

   if(true_negative + false_positive == 0)
   {
       markedness = precision - 1;
   }
   else
   {
       markedness = precision + (double) true_negative/(double)(true_negative + false_positive) - 1;
   }

   //Arrange vector

   Vector<double> binary_classification_test(15);

   binary_classification_test[0] = classification_accuracy;
   binary_classification_test[1] = error_rate;
   binary_classification_test[2] = sensitivity;
   binary_classification_test[3] = specificity;
   binary_classification_test[4] = precision;
   binary_classification_test[5] = positive_likelihood;
   binary_classification_test[6] = negative_likelihood;
   binary_classification_test[7] = F1_score;
   binary_classification_test[8] = false_positive_rate;
   binary_classification_test[9] = false_discovery_rate;
   binary_classification_test[10] = false_negative_rate;
   binary_classification_test[11] = negative_predictive_value;
   binary_classification_test[12] = Matthews_correlation_coefficient;
   binary_classification_test[13] = informedness;
   binary_classification_test[14] = markedness;

   return(binary_classification_test);
}


/// Returns the logloss for a binary classification problem.

double TestingAnalysis::calculate_logloss() const
{
    #ifdef __OPENNN_DEBUG__

     check();

    #endif

    #ifdef __OPENNN_DEBUG__

    const MultilayerPerceptron* multilayer_perceptron_pointer = neural_network_pointer->get_multilayer_perceptron_pointer();

    if(!multilayer_perceptron_pointer)
    {
    ostringstream buffer;

    buffer << "OpenNN Exception: TestingAnalysis class.\n"
          << "Vector<double> calculate_input_error_cross_correlation() const method.\n"
          << "Pointer to multilayer perceptron in neural network is nullptr.\n";

    throw logic_error(buffer.str());
    }

    #endif

    #ifdef __OPENNN_DEBUG__

    const size_t inputs_number = multilayer_perceptron_pointer->get_inputs_number();

    // Control sentence

    const Variables& variables = data_set_pointer->get_variables();

    if(inputs_number != variables.get_inputs_number())
    {
    ostringstream buffer;

    buffer << "OpenNN Exception: TestingAnalysis class." << endl
           << "Vector<double> calculate_input_error_cross_correlation() const method." << endl
           << "Number of inputs in neural network must be equal to number of inputs in data set." << endl;

    throw logic_error(buffer.str());
    }

    const size_t outputs_number = multilayer_perceptron_pointer->get_outputs_number();

    if(outputs_number != variables.get_targets_number())
    {
    ostringstream buffer;

    buffer << "OpenNN Exception: TestingAnalysis class." << endl
          << "Vector<double> calculate_input_error_cross_correlation() const method." << endl
          << "Number of outputs in neural network must be equal to number of targets in data set." << endl;

    throw logic_error(buffer.str());
    }

    #endif

    const Matrix<double> inputs = data_set_pointer->get_testing_inputs();
    const Matrix<double> targets = data_set_pointer->get_testing_targets();

    const Matrix<double> outputs = neural_network_pointer->calculate_outputs(inputs);

    const size_t testing_instances_number = targets.get_rows_number();

    double logloss = 0.0;

    for(size_t i = 0; i < testing_instances_number; i++)
    {
        logloss += targets(i,0)*log(outputs(i,0)) + (1-targets(i,0))*log(1-outputs(i,0));
    }

    return(-logloss/testing_instances_number);
}


// string object_to_string() const method

/// Returns a string representation of the testing analysis object. 

string TestingAnalysis::object_to_string() const
{
   ostringstream buffer;

   buffer << "Testing analysis\n"
          << "Display: " << display << "\n";

   return(buffer.str());
}


// void print() const method

/// Prints to the standard output the string representation of this testing analysis object. 

void TestingAnalysis::print() const
{
   cout << object_to_string();
}


// tinyxml2::XMLDocument* to_XML() const method

/// Serializes the testing analysis object into a XML document of the TinyXML library. 
/// See the OpenNN manual for more information about the format of this element. 

tinyxml2::XMLDocument* TestingAnalysis::to_XML() const
{
    tinyxml2::XMLDocument* document = new tinyxml2::XMLDocument;

    ostringstream buffer;

    // Root element

    tinyxml2::XMLElement* testing_analysis_element = document->NewElement("TestingAnalysis");

    document->InsertFirstChild(testing_analysis_element);

    // Display

    tinyxml2::XMLElement* display_element = document->NewElement("Display");
    testing_analysis_element->LinkEndChild(display_element);

    buffer.str("");
    buffer << display;

    tinyxml2::XMLText* display_text = document->NewText(buffer.str().c_str());
    display_element->LinkEndChild(display_text);

    return(document);
}


// void write_XML(tinyxml2::XMLPrinter&) const method

/// Serializes the testing analysis object into a XML document of the TinyXML library without keep the DOM tree in memory.
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


// void from_XML(const tinyxml2::XMLDocument&) method

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

        throw logic_error(buffer.str());
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
       catch(const logic_error& e)
       {
          cerr << e.what() << endl;
       }
    }
}


// void save(const string&) const method

/// Saves to a XML file the members of this testing analysis object.
/// @param file_name Name of testing analysis XML file.

void TestingAnalysis::save(const string& file_name) const
{
    tinyxml2::XMLDocument* document = to_XML();

    document->SaveFile(file_name.c_str());

    delete document;
}


// void load(const string&) method

/// Loads from a XML file the members for this testing analysis object.
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

      throw logic_error(buffer.str());
   }

   from_XML(document);
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
