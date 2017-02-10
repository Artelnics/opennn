/****************************************************************************************************************/
/*                                                                                                              */
/*   OpenNN: Open Neural Networks Library                                                                       */
/*   www.opennn.net                                                                                             */
/*                                                                                                              */
/*   T E S T I N G   A N A L Y S I S   C L A S S                                                                */
/*                                                                                                              */ 
/*   Roberto Lopez                                                                                              */ 
/*   Artelnics - Making intelligent use of data                                                                 */
/*   robertolopez@artelnics.com                                                                                 */
/*                                                                                                              */
/****************************************************************************************************************/

// OpenNN includes

#include "testing_analysis.h"

namespace OpenNN
{

// DEFAULT CONSTRUCTOR

/// Default constructor. 
/// It creates a testing analysis object neither associated to a neural network nor to a mathematical model or a data set.
/// By default, it constructs the function regression testing object. 

TestingAnalysis::TestingAnalysis(void)
 : neural_network_pointer(NULL),
   data_set_pointer(NULL),
   mathematical_model_pointer(NULL)
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
   data_set_pointer(NULL),
   mathematical_model_pointer(NULL)
{
   set_default();
}


// MATHEMATICAL MODEL CONSTRUCTOR

/// Mathematical mmodel constructor. 
/// It creates a testing analysis object not associated to a neural network, not associated to a data set, and associated to a mathematical model. 
/// By default, it constructs the inverse problem testing object. 
/// @param new_mathematical_model_pointer Pointer to a mathematical model object.

TestingAnalysis::TestingAnalysis(MathematicalModel* new_mathematical_model_pointer)
: neural_network_pointer(NULL),
   data_set_pointer(NULL),
   mathematical_model_pointer(new_mathematical_model_pointer)
{
   set_default();
}


// DATA SET CONSTRUCTOR

/// Data set constructor. 
/// It creates a testing analysis object not associated to a neural network, associated to a data set and not associated to a mathematical model. 
/// By default, it constructs the function regression testing object. 
/// @param new_data_set_pointer Pointer to a data set object.

TestingAnalysis::TestingAnalysis(DataSet* new_data_set_pointer)
: neural_network_pointer(NULL),
   data_set_pointer(new_data_set_pointer),
   mathematical_model_pointer(NULL)
{
   set_default();
}


// NEURAL NETWORK AND MATHEMATICAL MODEL CONSTRUCTOR

/// Neural network and mathematical model constructor. 
/// It creates a testing analysis object associated to a neural network and to a mathematical model, but not to a data set.
/// By default, it constructs the inverse problem testing object. 
/// @param new_neural_network_pointer Pointer to a neural network object.
/// @param new_mathematical_model_pointer Pointer to a mathematical model object.

TestingAnalysis::TestingAnalysis(NeuralNetwork* new_neural_network_pointer, MathematicalModel* new_mathematical_model_pointer)
 : neural_network_pointer(new_neural_network_pointer),
   data_set_pointer(NULL),
   mathematical_model_pointer(new_mathematical_model_pointer)
{
   set_default();
}


// NEURAL NETWORK AND DATA SET CONSTRUCTOR

/// Neural network and data set constructor. 
/// It creates a testing analysis object associated to a neural network and to a data set.
/// By default, it constructs the function regression testing object. 
/// @param new_neural_network_pointer Pointer to a neural network object.
/// @param new_data_set_pointer Pointer to a data set object.

TestingAnalysis::TestingAnalysis(NeuralNetwork* new_neural_network_pointer, DataSet* new_data_set_pointer)
 : neural_network_pointer(new_neural_network_pointer),
   data_set_pointer(new_data_set_pointer),
   mathematical_model_pointer(NULL)
{
   set_default();
}


// NEURAL NETWORK, MATHEMATICAL MODEL AND DATA SET CONSTRUCTOR

/// Neural network, mathematical model and data set constructor. 
/// It creates a testing analysis object associated to a neural network, a mathematical model and a data set.
/// By default, it constructs the inverse problem testing object. 
/// @param new_neural_network_pointer Pointer to a neural network object.
/// @param new_mathematical_model_pointer Pointer to a mathematical model object.
/// @param new_data_set_pointer Pointer to a data set object.

TestingAnalysis::TestingAnalysis(NeuralNetwork* new_neural_network_pointer, DataSet* new_data_set_pointer, MathematicalModel* new_mathematical_model_pointer)
 : neural_network_pointer(new_neural_network_pointer),
   data_set_pointer(new_data_set_pointer),
   mathematical_model_pointer(new_mathematical_model_pointer)
{
   set_default();
}


// XML CONSTRUCTOR

/// XML constructor. 
/// It creates a testing analysis object neither associated to a neural network nor to a mathematical model or a data set. 
/// It also loads the members of this object from a TinyXML document.
/// @param testing_analysis_document XML document containing the member data.

TestingAnalysis::TestingAnalysis(const tinyxml2::XMLDocument& testing_analysis_document)
 : neural_network_pointer(NULL),
   data_set_pointer(NULL),
   mathematical_model_pointer(NULL)
{
   set_default();

   from_XML(testing_analysis_document);
}


// FILE CONSTRUCTOR

/// File constructor. 
/// It creates a testing analysis object neither associated to a neural network nor to a mathematical model or a data set. 
/// It also loads the members of this object from XML file. 
/// @param file_name Name of testing analysis XML file.  

TestingAnalysis::TestingAnalysis(const std::string& file_name)
 : neural_network_pointer(NULL),
   data_set_pointer(NULL),
   mathematical_model_pointer(NULL)
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

// NeuralNetwork* get_neural_network_pointer(void) const method

/// Returns a pointer to the neural network object which is to be tested.

NeuralNetwork* TestingAnalysis::get_neural_network_pointer(void) const
{
    #ifdef __OPENNN_DEBUG__

    if(!neural_network_pointer)
    {
        std::ostringstream buffer;

        buffer << "OpenNN Exception: TestingAnalysis class.\n"
               << "NeuralNetwork* get_neural_network_pointer(void) const method.\n"
               << "Neural network pointer is NULL.\n";

        throw std::logic_error(buffer.str());
    }

    #endif

   return(neural_network_pointer);   
}


// DataSet* get_data_set_pointer(void) const method

/// Returns a pointer to the data set object on which the neural network is tested. 

DataSet* TestingAnalysis::get_data_set_pointer(void) const
{
    #ifdef __OPENNN_DEBUG__

    if(!data_set_pointer)
    {
        std::ostringstream buffer;

        buffer << "OpenNN Exception: TestingAnalysis class.\n"
               << "DataSet* get_data_set_pointer(void) const method.\n"
               << "Data set pointer is NULL.\n";

        throw std::logic_error(buffer.str());
    }

    #endif

   return(data_set_pointer);
}


// MathematicalModel* get_mathematical_model_pointer(void) const method

/// Returns a pointer to the mathematical model object on which the neural network is tested. 

MathematicalModel* TestingAnalysis::get_mathematical_model_pointer(void) const
{
    #ifdef __OPENNN_DEBUG__

    if(!mathematical_model_pointer)
    {
        std::ostringstream buffer;

        buffer << "OpenNN Exception: TestingAnalysis class.\n"
               << "MathematicalModel* get_mathematical_model_pointer(void) const method.\n"
               << "Mathematical model pointer is NULL.\n";

        throw std::logic_error(buffer.str());
    }

    #endif

   return(mathematical_model_pointer);
}


// const bool& get_display(void) const method

/// Returns true if messages from this class can be displayed on the screen,
/// or false if messages from this class can't be displayed on the screen.

const bool& TestingAnalysis::get_display(void) const
{
   return(display);     
}


// void set_default(void) method

/// Sets some default values to the testing analysis object:
/// <ul>
/// <li> Display: True.
/// </ul>

void TestingAnalysis::set_default(void)
{
   display = true;
}


// void set_neural_network_pointer(NeuralNetwork*) method

/// Sets a new neural network object to be tested.
/// @param new_neural_network_pointer Pointer to a neural network object.

void TestingAnalysis::set_neural_network_pointer(NeuralNetwork* new_neural_network_pointer)
{
   neural_network_pointer = new_neural_network_pointer;   
}


// void set_mathematical_model_pointer(MathematicalModel*) method

/// Sets a mathematical model to be used for validating the quality of a trained neural network.
/// @param new_mathematical_model_pointer Pointer to a mathematical model object.

void TestingAnalysis::set_mathematical_model_pointer(MathematicalModel* new_mathematical_model_pointer)
{
   mathematical_model_pointer = new_mathematical_model_pointer;   
}


// void set_data_set_pointer(DataSet*) method

/// Sets a new data set to be used for validating the quality of a trained neural network.
/// @param new_data_set_pointer Pointer to a data set object.

void TestingAnalysis::set_data_set_pointer(DataSet* new_data_set_pointer)
{
   data_set_pointer = new_data_set_pointer;   
}


// void set_display(const bool&) method

/// Sets a new display value. 
/// If it is set to true messages from this class are to be displayed on the screen;
/// if it is set to false messages from this class are not to be displayed on the screen.
/// @param new_display Display value.

void TestingAnalysis::set_display(const bool& new_display)
{
   display = new_display;
}


// void check(void) const method

/// Checks that:
/// <ul>
/// <li> The neural network pointer is not NULL.
/// <li> The data set pointer is not NULL.
/// </ul>

void TestingAnalysis::check(void) const
{
   std::ostringstream buffer;

   if(!neural_network_pointer)
   {
      buffer << "OpenNN Exception: TestingAnalysis class.\n"
             << "void check(void) const method.\n"
             << "Neural network pointer is NULL.\n";

      throw std::logic_error(buffer.str());
   }

   if(!data_set_pointer)
   {
      buffer << "OpenNN Exception: TestingAnalysis class.\n"
             << "void check(void) const method.\n"
             << "Data set pointer is NULL.\n";

      throw std::logic_error(buffer.str());
   }
}


// Vector< Matrix<double> > calculate_target_output_data(void) const method

/// Returns a vector of matrices with number of rows equal to number of testing instances and
/// number of columns equal to two (the targets value and the outputs value).

Vector< Matrix<double> > TestingAnalysis::calculate_target_output_data(void) const
{
   // Control sentence (if debug)

   #ifdef __OPENNN_DEBUG__

   check();

   #endif

   // Data set stuff

   const Instances& instances = data_set_pointer->get_instances();

   const size_t testing_instances_number = instances.count_testing_instances_number();

   const Matrix<double> testing_input_data = data_set_pointer->arrange_testing_input_data();

   const Matrix<double> target_data = data_set_pointer->arrange_testing_target_data();

   // Neural network stuff

   const MultilayerPerceptron* multilayer_perceptron_pointer = neural_network_pointer->get_multilayer_perceptron_pointer();

   const size_t outputs_number = multilayer_perceptron_pointer->get_outputs_number();

   const Matrix<double> output_data = neural_network_pointer->calculate_output_data(testing_input_data);

   // Approximation testing stuff

   Vector< Matrix<double> > target_output_data(outputs_number);

   for(size_t i = 0; i < outputs_number; i++)
   {
      target_output_data[i].set(testing_instances_number, 2);

      target_output_data[i].set_column(0, target_data.arrange_column(i));
      target_output_data[i].set_column(1, output_data.arrange_column(i));
   }

   return(target_output_data);
}



// Vector< LinearRegressionParameters<double> > calculate_linear_regression_parameters(void) const method

/// Performs a linear regression analysis between the testing instances in the data set and
/// the corresponding neural network outputs.
/// It returns all the provided parameters in a vector of vectors.
/// The number of elements in the vector is equal to the number of output variables.
/// The size of each element is equal to the number of regression parameters (2).
/// In this way, each subvector contains the regression parameters intercept and slope of an output variable.

Vector< LinearRegressionParameters<double> > TestingAnalysis::calculate_linear_regression_parameters(void) const
{
   // Control sentence (if debug)

   #ifdef __OPENNN_DEBUG__

   check();

   #endif

   // Data set stuff

   const Instances& instances = data_set_pointer->get_instances();

   const size_t testing_instances_number = instances.count_testing_instances_number();

   // Neural network stuff

   const MultilayerPerceptron* multilayer_perceptron_pointer = neural_network_pointer->get_multilayer_perceptron_pointer();

   const size_t outputs_number = multilayer_perceptron_pointer->get_outputs_number();

   #ifdef __OPENNN_DEBUG__

   std::ostringstream buffer;

   if(testing_instances_number == 0)
   {
      buffer << "OpenNN Exception: TestingAnalysis class.\n"
             << "Vector< LinearRegressionParameters<double> > calculate_linear_regression_parameters(void) const method.\n"
             << "Number of testing instances is zero.\n";

      throw std::logic_error(buffer.str());
   }

   #endif

   // Calculate regression parameters

   const Matrix<double> input_data = data_set_pointer->arrange_testing_input_data();
   const Matrix<double> target_data = data_set_pointer->arrange_testing_target_data();
   const Matrix<double> output_data = neural_network_pointer->calculate_output_data(input_data);

   Vector<double> target_variable(testing_instances_number);
   Vector<double> output_variable(testing_instances_number);

   Vector< LinearRegressionParameters<double> > linear_regression_parameters(outputs_number);

   for(size_t i = 0; i < outputs_number; i++)
   {
       target_variable = target_data.arrange_column(i);
       output_variable = output_data.arrange_column(i);

       linear_regression_parameters[i] = output_variable.calculate_linear_regression_parameters(target_variable);
   }

   return(linear_regression_parameters);
}


// TestingAnalysis::LinearRegressionResults TestingAnalysis::perform_linear_regression_analysis(void) const

/// Performs a linear regression analysis of a neural network on the testing indices of a data set.
/// It returns a linear regression analysis results structure, which consists of:
/// <ul>
/// <li> Linear regression parameters.
/// <li> Scaled target and output data.
/// </ul>

TestingAnalysis::LinearRegressionResults TestingAnalysis::perform_linear_regression_analysis(void) const
{
    check();

    const Instances& instances = data_set_pointer->get_instances();

    const size_t testing_instances_number = instances.count_testing_instances_number();

    if(testing_instances_number == 0)
    {
        std::ostringstream buffer;

       buffer << "OpenNN Exception: TestingAnalysis class.\n"
              << "LinearRegressionResults perform_linear_regression_analysis(void) const method.\n"
              << "Number of testing instances is zero.\n";

       throw std::logic_error(buffer.str());
    }

   LinearRegressionResults linear_regression_results;

   linear_regression_results.target_output_data = calculate_target_output_data();
   linear_regression_results.linear_regression_parameters = calculate_linear_regression_parameters();

   return(linear_regression_results);
}


// void LinearRegressionResults::save(const std::string&) const method

/// Saves a linear regression analysis results structure to a data file.
/// @param file_name Name of results data file.

void TestingAnalysis::LinearRegressionResults::save(const std::string& file_name) const
{
   std::ofstream file(file_name.c_str());

   file << linear_regression_parameters
        << "Target-output data:\n"
        << target_output_data;

   file.close();
}


// Matrix<double> calculate_error_data(void) const method

/// Calculates the errors between the outputs from a neural network and the testing instances in a data set.
/// It returns a vector of tree matrices:
/// <ul>
/// <li> Absolute error.
/// <li> Relative error.
/// <li> Percentage error.
/// </ul>
/// The number of rows in each matrix is the number of testing instances in the data set.
/// The number of columns is the number of outputs in the neural network.

Vector< Matrix<double> > TestingAnalysis::calculate_error_data(void) const
{
   // Data set stuff

   #ifdef __OPENNN_DEBUG__

   check();

   #endif

   const size_t testing_instances_number = data_set_pointer->get_instances().count_testing_instances_number();

    #ifdef __OPENNN_DEBUG__

    std::ostringstream buffer;

    if(testing_instances_number == 0)
    {
       buffer << "OpenNN Exception: TestingAnalysis class.\n"
              << "Vector< Matrix<double> > calculate_error_data(void) const.\n"
              << "Number of testing instances is zero.\n";

       throw std::logic_error(buffer.str());
    }

    #endif


   const Matrix<double> input_data = data_set_pointer->arrange_testing_input_data();

   const Matrix<double> target_data = data_set_pointer->arrange_testing_target_data();

   // Neural network stuff

   const Matrix<double> output_data = neural_network_pointer->calculate_output_data(input_data);

   const UnscalingLayer* unscaling_layer_pointer = neural_network_pointer->get_unscaling_layer_pointer();

   #ifdef __OPENNN_DEBUG__

   if(!unscaling_layer_pointer)
   {
      buffer << "OpenNN Exception: TestingAnalysis class.\n"
             << "Vector< Matrix<double> > calculate_error_data(void) const.\n"
             << "Unscaling layer is NULL.\n";

      throw std::logic_error(buffer.str());
   }

   #endif

   const Vector<double>& outputs_minimum = unscaling_layer_pointer->arrange_minimums();
   const Vector<double>& outputs_maximum = unscaling_layer_pointer->arrange_maximums();

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

       targets = target_data.arrange_column(i);
       outputs = output_data.arrange_column(i);

       difference_absolute_value = (targets - outputs).calculate_absolute_value();

       error_data[i].set_column(0, difference_absolute_value);

       // Relative error

       error_data[i].set_column(1, difference_absolute_value/std::abs(outputs_maximum[i]-outputs_minimum[i]));

       // Percentage error

       error_data[i].set_column(2, difference_absolute_value*100.0/std::abs(outputs_maximum[i]-outputs_minimum[i]));
    }

   return(error_data);
}


// Vector< Vector< Statistics<double> > > calculate_error_data_statistics(void) const method

/// Calculates the basic statistics on the error data.
/// <ul>
/// <li> Minimum.
/// <li> Maximum.
/// <li> Mean.
/// <li> Standard deviation
/// </ul>

Vector< Vector< Statistics<double> > > TestingAnalysis::calculate_error_data_statistics(void) const
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


// Vector< Matrix<double> > calculate_error_data_statistics_matrices(void) const method

/// Returns a vector of matrices with the statistics of the errors between the neural network outputs and the testing targets in the data set.
/// The size of the vector is the number of output variables.
/// The number of rows in each submatrix is three (absolute, relative and percentage errors).
/// The number of columns in each submatrix is four (minimum, maximum, mean and standard deviation).

Vector< Matrix<double> > TestingAnalysis::calculate_error_data_statistics_matrices(void) const
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


// Vector< Histogram<double> > calculate_error_data_histograms(const size_t&) const method

/// Calculates histograms for the relative errors of all the output variables.
/// The number of bins is set by the user.
/// @param bins_number Number of bins in the histograms.

Vector< Histogram<double> > TestingAnalysis::calculate_error_data_histograms(const size_t& bins_number) const
{
   const Vector< Matrix<double> > error_data = calculate_error_data();

   const size_t outputs_number = error_data.size();

   Vector< Histogram<double> > histograms(outputs_number);

   for(size_t i = 0; i < outputs_number; i++)
   {
       histograms[i] = error_data[i].arrange_column(0).calculate_histogram(bins_number);
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
        maximal_errors[i] = error_data[i].arrange_column(0).calculate_maximal_indices(instances_number);
    }

    return(maximal_errors);
}


// Vector<double> calculate_testing_errors(void) const method

/// Returns a vector containing the values of the errors between the outputs of the neural network
/// and the targets. The vector consists of:
/// <ul>
/// <li> Sum squared error.
/// <li> Mean squared error.
/// <li> Root mean squared error.
/// <li> Normalized squared error.
/// </ul>

Vector<double> TestingAnalysis::calculate_testing_errors(void) const
{
    // Data set stuff

    #ifdef __OPENNN_DEBUG__

    check();

    #endif

    const size_t testing_instances_number = data_set_pointer->get_instances().count_testing_instances_number();

     #ifdef __OPENNN_DEBUG__

     std::ostringstream buffer;

     if(testing_instances_number == 0)
     {
        buffer << "OpenNN Exception: TestingAnalysis class.\n"
               << "Vector< Matrix<double> > calculate_errors(void) const.\n"
               << "Number of testing instances is zero.\n";

        throw std::logic_error(buffer.str());
     }

     #endif

    const Matrix<double> input_data = data_set_pointer->arrange_testing_input_data();

    const Matrix<double> target_data = data_set_pointer->arrange_testing_target_data();

    // Neural network stuff

    const Matrix<double> output_data = neural_network_pointer->calculate_output_data(input_data);

    #ifdef __OPENNN_DEBUG__   

//    if(!unscaling_layer_pointer)
//    {
//       buffer << "OpenNN Exception: TestingAnalysis class.\n"
//              << "Vector< Matrix<double> > calculate_errors(void) const.\n"
//              << "Unscaling layer is NULL.\n";

//       throw std::logic_error(buffer.str());
//    }

    #endif

    Vector<double> errors(4,0.0);

    // Results

    errors[0] = output_data.calculate_sum_squared_error(target_data);
    errors[1] = output_data.calculate_sum_squared_error(target_data)/testing_instances_number;
    errors[2] = sqrt(errors[1]);
    errors[3] = calculate_testing_normalized_squared_error(target_data, output_data);

    return errors;
}


// Vector<double> calculate_classification_testing_errors(void) const method

/// Returns a vector containing the values of the errors between the outputs of the neural network
/// and the targets for a classification problem. The vector consists of:
/// <ul>
/// <li> Sum squared error.
/// <li> Mean squared error.
/// <li> Root mean squared error.
/// <li> Normalized squared error.
/// <li> Cross-entropy error.
/// </ul>

Vector<double> TestingAnalysis::calculate_classification_testing_errors(void) const
{
    // Data set stuff

    #ifdef __OPENNN_DEBUG__

    check();

    #endif

    const size_t testing_instances_number = data_set_pointer->get_instances().count_testing_instances_number();

     #ifdef __OPENNN_DEBUG__

     std::ostringstream buffer;

     if(testing_instances_number == 0)
     {
        buffer << "OpenNN Exception: TestingAnalysis class.\n"
               << "Vector< Matrix<double> > calculate_errors(void) const.\n"
               << "Number of testing instances is zero.\n";

        throw std::logic_error(buffer.str());
     }

     #endif

    const Matrix<double> input_data = data_set_pointer->arrange_testing_input_data();

    const Matrix<double> target_data = data_set_pointer->arrange_testing_target_data();

    // Neural network stuff

    const Matrix<double> output_data = neural_network_pointer->calculate_output_data(input_data);

    Vector<double> errors(5,0.0);

    // Results

    errors[0] = output_data.calculate_sum_squared_error(target_data);
    errors[1] = output_data.calculate_sum_squared_error(target_data)/testing_instances_number;
    errors[2] = sqrt(errors[1]);
    errors[3] = calculate_testing_normalized_squared_error(target_data, output_data);
    errors[4] = calculate_testing_cross_entropy_error(target_data, output_data);

    return errors;
}


// double calculate_testing_normalized_squared_error(const Matrix<double>&, const Matrix<double>&) const method

/// Returns the normalized squared error between the targets and the outputs of the neural network.
/// @param target_data Testing target data.
/// @param output_data Testing output data.

double TestingAnalysis::calculate_testing_normalized_squared_error(const Matrix<double>& target_data, const Matrix<double>& output_data) const
{
    const size_t testing_instances_number = target_data.get_rows_number();

    const Vector<double> testing_target_data_mean = data_set_pointer->calculate_testing_target_data_mean();

    double normalization_coefficient = 0.0;
    double sum_squared_error = 0.0;

    for(size_t i = 0; i < testing_instances_number; i++)
    {
        sum_squared_error += output_data.arrange_row(i).calculate_sum_squared_error(target_data.arrange_row(i));

        normalization_coefficient += target_data.arrange_row(i).calculate_sum_squared_error(testing_target_data_mean);
    }

    return sum_squared_error/normalization_coefficient;
}


// double calculate_testing_cross_entropy_error(const Matrix<double>&, const Matrix<double>&) const method

/// Returns the cross-entropy error between the targets and the outputs of the neural network.
/// It can only be computed for classification problems.
/// @param target_data Testing target data.
/// @param output_data Testing output data.

double TestingAnalysis::calculate_testing_cross_entropy_error(const Matrix<double>& target_data, const Matrix<double>& output_data) const
{
    const size_t testing_instances_number = target_data.get_rows_number();
    const size_t outputs_number = target_data.get_columns_number();

    Vector<double> targets(outputs_number);
    Vector<double> outputs(outputs_number);

    double cross_entropy_error = 0.0;

    for(size_t i = 0; i < testing_instances_number; i++)
    {
        outputs = output_data.arrange_row(i);
        targets = target_data.arrange_row(i);

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

    return cross_entropy_error;
}


// double calculate_testing_weighted_squared_error(const Matrix<double>&, const Matrix<double>&) const method

/// Returns the weighted squared error between the targets and the outputs of the neural network. It can only be computed for
/// binary classification problems.
/// @param target_data Testing target data.
/// @param output_data Testing output data.

double TestingAnalysis::calculate_testing_weighted_squared_error(const Matrix<double>& target_data, const Matrix<double>& output_data) const
{
    const size_t testing_instances_number = target_data.get_rows_number();

    #ifdef __OPENNN_DEBUG__

//    std::ostringstream buffer;

//    if(outputs_number != 1)
//    {
//       buffer << "OpenNN Exception: TestingAnalysis class.\n"
//              << "double calculate_testing_weighted_squared_error(const Matrix<double>&, const Matrix<double>&) const.\n"
//              << "Number of outputs must be one.\n";

//       throw std::logic_error(buffer.str());
//    }

    #endif

    const Vector<size_t> target_distribution = data_set_pointer->calculate_target_distribution();

    const size_t negatives_number = target_distribution[0];
    const size_t positives_number = target_distribution[1];

    const double negatives_weight = 1.0;
    const double positives_weight = (double)negatives_number/positives_number;

    double error = 0.0;
    double sum_squared_error = 0.0;

    for(size_t i = 0; i < testing_instances_number; i++)
    {
        if(target_data(0,i) == 1.0)
        {
            error = positives_weight*output_data.arrange_column(i).calculate_sum_squared_error(target_data.arrange_column(i));
        }
        else if(target_data(0,i) == 0.0)
        {
            error = negatives_weight*output_data.arrange_column(i).calculate_sum_squared_error(target_data.arrange_column(i));
        }
        else
        {
            std::ostringstream buffer;

            buffer << "OpenNN Exception: TestingAnalysis class.\n"
                   << "double calculate_testing_weighted_squared_error(const Matrix<double>&, const Matrix<double>&) const method.\n"
                   << "Target is neither a positive nor a negative.\n";

            throw std::logic_error(buffer.str());
        }

        sum_squared_error += error;
    }

    const Vector<size_t> targets_indices = data_set_pointer->get_variables().arrange_targets_indices();

    const size_t negatives = data_set_pointer->calculate_training_negatives(targets_indices[0]);

    const double normalization_coefficient = negatives*negatives_weight*0.5;

    return(sum_squared_error/normalization_coefficient);
}


// Matrix<size_t> calculate_confusion_binary_classification(const Matrix<double>&, const Matrix<double>&, const double&) const method

/// Returns the confusion matrix for a binary classification problem.
/// @param target_data Testing target data.
/// @param output_data Testing output data.
/// @param decision_threshold Decision threshold.

Matrix<size_t> TestingAnalysis::calculate_confusion_binary_classification(const Matrix<double>& target_data, const Matrix<double>& output_data, const double& decision_threshold) const
{
    const size_t rows_number = target_data.get_rows_number();

    Matrix<size_t> confusion(2, 2);

    size_t true_positive = 0;
    size_t false_negative = 0;
    size_t false_positive = 0;
    size_t true_negative = 0;        

    for(size_t i = 0; i < rows_number; i++)
    {

        if(decision_threshold == 0.0 && target_data(i,0) == 0.0 )
        {
            false_positive++;

        }
        else if (decision_threshold == 0.0 && target_data(i,0) == 1.0)
        {
            true_positive++;

        }
        else if(target_data(i,0) >= decision_threshold && output_data(i,0) >= decision_threshold)
        {
            // True positive

            true_positive++;

        }
        else if(target_data(i,0) >= decision_threshold && output_data(i,0) < decision_threshold)
        {
            // False negative

            false_negative++;

        }
        else if(target_data(i,0) < decision_threshold && output_data(i,0) >= decision_threshold)
        {
            // False positive

            false_positive++;

        }
        else if(target_data(i,0) < decision_threshold && output_data(i,0) < decision_threshold)
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
        std::ostringstream buffer;

        buffer << "OpenNN Exception: TestingAnalysis class.\n"
               << "Matrix<size_t> calculate_confusion_binary_classification(const Matrix<double>&, const Matrix<double>&, const double&) const method.\n"
               << "Number of elements in confusion matrix must be equal to number of testing instances.\n";

        throw std::logic_error(buffer.str());
    }

    return(confusion);
}


// Matrix<size_t> calculate_confusion_multiple_classification(const Matrix<double>&, const Matrix<double>&) const method

/// Returns the confusion matrix for a binary classification problem.
/// @param target_data Testing target data.
/// @param output_data Testing output data.

Matrix<size_t> TestingAnalysis::calculate_confusion_multiple_classification(const Matrix<double>& target_data, const Matrix<double>& output_data) const
{
    const size_t rows_number = target_data.get_rows_number();
    const size_t columns_number = target_data.get_columns_number();

    Matrix<size_t> confusion(columns_number, columns_number, 0);

    size_t target_index = 0;
    size_t output_index = 0;

    for(size_t i = 0; i < rows_number; i++)
    {
        target_index = target_data.arrange_row(i).calculate_maximal_index();
        output_index = output_data.arrange_row(i).calculate_maximal_index();

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

Vector<size_t> TestingAnalysis::calculate_positives_negatives_rate(const Matrix<double>& target_data, const Matrix<double>& output_data) const
{
    const Matrix<size_t> confusion = calculate_confusion_binary_classification(target_data, output_data, 0.5);
    Vector<size_t> positives_negatives_rate(2);

    positives_negatives_rate[0] = confusion(0,0) + confusion(0,1);
    positives_negatives_rate[1] = confusion(1,0) + confusion(1,1);

    return(positives_negatives_rate);
}


// Matrix<size_t> calculate_confusion(void) const method

/// Returns the confusion matrix of a neural network on the testing instances of a data set.
/// If the number of outputs is one, the size of the confusion matrix is two.
/// If the number of outputs is greater than one, the size of the confusion matrix is the number of outputs.

Matrix<size_t> TestingAnalysis::calculate_confusion(void) const
{
   #ifdef __OPENNN_DEBUG__

    check();

   #endif

   const MultilayerPerceptron* multilayer_perceptron_pointer = neural_network_pointer->get_multilayer_perceptron_pointer();

   #ifdef __OPENNN_DEBUG__

   if(!multilayer_perceptron_pointer)
   {
      std::ostringstream buffer;

      buffer << "OpenNN Exception: TestingAnalysis class.\n"
             << "Matrix<size_t> calculate_confusion(void) const method.\n"
             << "Pointer to multilayer perceptron in neural network is NULL.\n";

      throw std::logic_error(buffer.str());
   }

   #endif

   const size_t outputs_number = multilayer_perceptron_pointer->get_outputs_number();

   #ifdef __OPENNN_DEBUG__

   const size_t inputs_number = multilayer_perceptron_pointer->get_inputs_number();

   // Control sentence

   const Variables& variables = data_set_pointer->get_variables();

   if(inputs_number != variables.count_inputs_number())
   {      
       std::ostringstream buffer;

       buffer << "OpenNN Exception: TestingAnalysis class." << std::endl
              << "Matrix<size_t> calculate_confusion(void) const method." << std::endl
              << "Number of inputs in neural network must be equal to number of inputs in data set." << std::endl;

      throw std::logic_error(buffer.str());
   }

   if(outputs_number != variables.count_targets_number())
   {
       std::ostringstream buffer;

      buffer << "OpenNN Exception: TestingAnalysis class." << std::endl
             << "Matrix<size_t> calculate_confusion(void) const method." << std::endl
             << "Number of outputs in neural network must be equal to number of targets in data set." << std::endl;

      throw std::logic_error(buffer.str());
   }

   #endif

    const Matrix<double> input_data = data_set_pointer->arrange_testing_input_data();
    const Matrix<double> target_data = data_set_pointer->arrange_testing_target_data();    

    const Matrix<double> output_data = neural_network_pointer->calculate_output_data(input_data);

//    output_data.save("../data/output_data.dat");

    if(outputs_number == 1)
    {
        double decision_threshold;

        if(neural_network_pointer->get_probabilistic_layer_pointer() != NULL)
        {
            decision_threshold = neural_network_pointer->get_probabilistic_layer_pointer()->get_decision_threshold();
        }
        else
        {
            decision_threshold = 0.5;
        }

        return(calculate_confusion_binary_classification(target_data, output_data, decision_threshold));
    }
    else
    {
        return(calculate_confusion_multiple_classification(target_data, output_data));
    }
}


//TestingAnalysis::RocCurveResults perform_roc_analysis (void) const

/// Performs a ROC curve analysis.
/// It returns a ROC curve analysis results structure, which consists of:
/// <ul>
/// <li> ROC curve
/// <li> Area under the ROC curve
/// <li> Optimal threshold
/// </ul>

TestingAnalysis::RocAnalysisResults TestingAnalysis::perform_roc_analysis (void) const
{
    #ifdef __OPENNN_DEBUG__

    check();

    #endif

    #ifdef __OPENNN_DEBUG__

    const MultilayerPerceptron* multilayer_perceptron_pointer = neural_network_pointer->get_multilayer_perceptron_pointer();

    if(!multilayer_perceptron_pointer)
    {
        std::ostringstream buffer;

       buffer << "OpenNN Exception: TestingAnalysis class.\n"
              << "RocCurveResults perform_roc_analysis(void) const method.\n"
              << "Pointer to multilayer perceptron in neural network is NULL.\n";

      throw std::logic_error(buffer.str());
    }

    #endif

    #ifdef __OPENNN_DEBUG__

    const size_t inputs_number = multilayer_perceptron_pointer->get_inputs_number();

    // Control sentence

    const Variables& variables = data_set_pointer->get_variables();

    if(inputs_number != variables.count_inputs_number())
    {
        std::ostringstream buffer;

        buffer << "OpenNN Exception: TestingAnalysis class." << std::endl
               << "RocCurveResults perform_roc_analysis(void) const method." << std::endl
               << "Number of inputs in neural network must be equal to number of inputs in data set." << std::endl;

       throw std::logic_error(buffer.str());
    }

    const size_t outputs_number = multilayer_perceptron_pointer->get_outputs_number();

    if(outputs_number != variables.count_targets_number())
    {
        std::ostringstream buffer;

       buffer << "OpenNN Exception: TestingAnalysis class." << std::endl
              << "RocCurveResults perform_roc_analysis(void) const method." << std::endl
              << "Number of outputs in neural network must be equal to number of targets in data set." << std::endl;

       throw std::logic_error(buffer.str());
    }

    #endif

     const Matrix<double> input_data = data_set_pointer->arrange_testing_input_data();
     const Matrix<double> target_data = data_set_pointer->arrange_testing_target_data();

     const Matrix<double> output_data = neural_network_pointer->calculate_output_data(input_data);

     RocAnalysisResults roc_analysis_results;

     roc_analysis_results.roc_curve = calculate_roc_curve(target_data, output_data);
     roc_analysis_results.area_under_curve = calculate_area_under_curve(target_data, output_data);
     roc_analysis_results.optimal_threshold = calculate_optimal_threshold(target_data, output_data, roc_analysis_results.roc_curve);

     return(roc_analysis_results);
}


//double calculate_Wilcoxon_parameter(const double& ,const double& ) const

/// Calculates the Wilcoxon parameter, which is used for calculating the area under a ROC curve.
/// Returns 1 if first value is greater than second one, 0 if second value is greater than first one or 0.5 in other case.
/// @param x Target data value.
/// @param y Target data value.

double TestingAnalysis::calculate_Wilcoxon_parameter (const double& x, const double& y) const
{
    if(x > y)
    {
        return (1);
    }
    else if(x < y)
    {
        return (0);
    }
    else
    {
        return (0.5);
    }
}


// Matrix<double> calculate_roc_curve (const Matrix<double>& , const Matrix<double>& output_data) const

/// Returns a matrix with the values of a ROC curve for a binary classification problem.
/// The number of columns is three. The third column contains the decision threshold.
/// The number of rows is one more than the number of outputs if the number of outputs is lower than 100
/// or 50 in other case.
/// @param target_data Testing target data.
/// @param output_data Testing output data.

Matrix<double> TestingAnalysis::calculate_roc_curve(const Matrix<double>& target_data, const Matrix<double>& output_data) const
{
    const Vector<size_t> positives_negatives_rate = calculate_positives_negatives_rate(target_data, output_data);

    const size_t total_positives = positives_negatives_rate[0];
    const size_t total_negatives = positives_negatives_rate[1];

    if(total_positives == 0)
    {
        std::ostringstream buffer;

        buffer << "OpenNN Exception: TestingAnalysis class.\n"
               << "Matrix<double> calculate_roc_curve(const Matrix<double>&, const Matrix<double>&) const.\n"
               << "Number of positive instances ("<< total_positives <<") must be greater than zero.\n";

        throw std::logic_error(buffer.str());
     }

    if(total_negatives == 0)
    {
        std::ostringstream buffer;

        buffer << "OpenNN Exception: TestingAnalysis class.\n"
               << "Matrix<double> calculate_roc_curve(const Matrix<double>&, const Matrix<double>&) const.\n"
               << "Number of negative instances ("<< total_negatives <<") must be greater than zero.\n";

        throw std::logic_error(buffer.str());
     }

    const size_t maximum_points_number = 1000;

    size_t step_size;

    const size_t testing_instances_number = target_data.get_rows_number();
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

    Matrix<double> target_output_data = output_data.assemble_columns(target_data);

    Matrix<double> sorted_target_output_data = target_output_data.sort_less_rows(0);

    const Vector<size_t> columns_output_indices(1,0);
    const Vector<size_t> columns_target_indices(1,1);

    const Matrix<double> sorted_target_data = sorted_target_output_data.arrange_submatrix_columns(columns_target_indices);
    const Matrix<double> sorted_output_data = sorted_target_output_data.arrange_submatrix_columns(columns_output_indices);

    Matrix<double> roc_curve(points_number+1, 3, 0.0);

    double threshold = 0;

    size_t positives;
    size_t negatives;

    size_t current_index;

    int j = 0;
    int i = 0;

    const size_t step_s = step_size;

#pragma omp parallel for private(i, j, positives, negatives, threshold, current_index) schedule(dynamic)

    for(i = 0; i < (int)points_number; i++)
    {
        positives = 0;
        negatives = 0;

        current_index = i*step_s;

        threshold = sorted_output_data(current_index, 0);

        for(j = 0; j < (int)current_index; j++)
        {
             if(sorted_output_data(j,0) < threshold && sorted_target_data(j,0) == 1.0)
             {
                 positives++;
             }
             if(sorted_output_data(j,0) < threshold && sorted_target_data(j,0) == 0.0)
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

    return (roc_curve);
}


// double calculate_area_under_curve(const Matrix<double>& , const Matrix<double>& ) const

/// Returns the area under a ROC curve.
/// @param target_data Testing target data.
/// @param output_data Testing output data.

double TestingAnalysis::calculate_area_under_curve (const Matrix<double>& target_data, const Matrix<double>& output_data) const
{
    const Vector<size_t> positives_negatives_rate = calculate_positives_negatives_rate(target_data, output_data);

    const size_t total_positives = positives_negatives_rate[0];
    const size_t total_negatives = positives_negatives_rate[1];

    if(total_positives == 0)
    {
        std::ostringstream buffer;

        buffer << "OpenNN Exception: TestingAnalysis class.\n"
               << "Matrix<double> calculate_roc_curve(const Matrix<double>&, const Matrix<double>&) const.\n"
               << "Number of positive instances ("<< total_positives <<") must be greater than zero.\n";

        throw std::logic_error(buffer.str());
     }

    if(total_negatives == 0)
    {
        std::ostringstream buffer;

        buffer << "OpenNN Exception: TestingAnalysis class.\n"
               << "Matrix<double> calculate_roc_curve(const Matrix<double>&, const Matrix<double>&) const.\n"
               << "Number of negative instances ("<< total_negatives <<") must be greater than zero.\n";

        throw std::logic_error(buffer.str());
     }

    size_t testing_instances_number = target_data.get_rows_number();

    double sum = 0.0;

    double area_under_curve;

    int i,j;

#pragma omp parallel for private(i, j) reduction(+ : sum) schedule(dynamic)

    for(i = 0; i < testing_instances_number; i++)
    {
        if(target_data(i,0) == 1)
        {
            for(j = 0; j < testing_instances_number; j++)
            {
                if(target_data(j,0) == 0)
                {
                   sum += calculate_Wilcoxon_parameter(output_data(i,0),output_data(j,0));
                }
            }
        }
    }

    area_under_curve = (double)sum/(double)(total_positives*total_negatives);

    return (area_under_curve);
}


// double calculate_optimal_threshold (const Matrix<double>& , const Matrix<double>& ) const

/// Returns the point of optimal classification accuracy, which is the nearest ROC curve point to the upper left corner (0,1).
/// @param target_data Testing target data.
/// @param output_data Testing output data.

double TestingAnalysis::calculate_optimal_threshold (const Matrix<double>& target_data, const Matrix<double>& output_data ) const
{
    const size_t rows_number = target_data.get_rows_number();
    const size_t columns_number = target_data.get_columns_number();

    const size_t maximum_points_number = 1000;

    size_t step_size;
    size_t points_number;

    if(rows_number > maximum_points_number)
    {
        step_size = rows_number/maximum_points_number;
        points_number = rows_number/step_size;
    }
    else
    {
        points_number = rows_number;
        step_size = 1;
    }

    Matrix<double> target_output_data = output_data.assemble_columns(target_data);

    Matrix<double> sorted_target_output_data = target_output_data.sort_less_rows(0);

    const Vector<size_t> columns_output_indices(0, 1, columns_number - 1);
    const Vector<size_t> columns_target_indices(columns_number, 1, 2*columns_number - 1);

    const Matrix<double> sorted_target_data = sorted_target_output_data.arrange_submatrix_columns(columns_target_indices);
    const Matrix<double> sorted_output_data = sorted_target_output_data.arrange_submatrix_columns(columns_output_indices);

    const Matrix<double> roc_curve = calculate_roc_curve(sorted_target_data, sorted_output_data);

    double threshold = 0.0;
    double optimal_threshold = 0.5;

    double minimun_distance = std::numeric_limits<double>::max();
    double distance;

    size_t current_index;

    for(size_t i = 0; i < points_number; i++)
    {
        current_index = i*step_size;

        threshold = sorted_output_data(current_index, 0);

        distance = sqrt(roc_curve(i,0)*roc_curve(i,0) + (roc_curve(i,1) - 1.0)*(roc_curve(i,1) - 1.0));

        if(distance < minimun_distance)
        {
            optimal_threshold = threshold;

            minimun_distance = distance;
        }
    }

    return (optimal_threshold);
}

// double calculate_optimal_threshold (const Matrix<double>& , const Matrix<double>&, const Matrix<double>&) const

/// Returns the point of optimal classification accuracy, which is the nearest ROC curve point to the upper left corner (0,1).
/// @param target_data Testing target data.
/// @param output_data Testing output data.
/// @param roc_curve ROC curve.

double TestingAnalysis::calculate_optimal_threshold (const Matrix<double>& target_data, const Matrix<double>& output_data, const Matrix<double>& roc_curve) const
{
    const size_t rows_number = target_data.get_rows_number();
    const size_t columns_number = target_data.get_columns_number();

    size_t step_size;
    const size_t points_number = roc_curve.get_rows_number();

    if(rows_number > points_number)
    {
        step_size = rows_number/points_number;
    }
    else
    {
        step_size = 1;
    }

    Matrix<double> target_output_data = output_data.assemble_columns(target_data);

    Matrix<double> sorted_target_output_data = target_output_data.sort_less_rows(0);

    const Vector<size_t> columns_output_indices(0, 1, columns_number - 1);

    const Matrix<double> sorted_output_data = sorted_target_output_data.arrange_submatrix_columns(columns_output_indices);

    double threshold = 0.0;
    double optimal_threshold = 0.5;

    double minimun_distance = std::numeric_limits<double>::max();
    double distance;

    size_t current_index;

    for(size_t i = 0; i < points_number; i++)
    {
        current_index = i*step_size;

        threshold = sorted_output_data(current_index, 0);

        distance = sqrt(roc_curve(i,0)*roc_curve(i,0) + (roc_curve(i,1) - 1.0)*(roc_curve(i,1) - 1.0));

        if(distance < minimun_distance)
        {
            optimal_threshold = threshold;

            minimun_distance = distance;
        }
    }

    return (optimal_threshold);
}

// Matrix<double> perform_cumulative_gain_analysis(void) const

/// Performs a cumulative gain analysis.
/// Returns a matrix with the values of a cumulative gain chart.

Matrix<double> TestingAnalysis::perform_cumulative_gain_analysis(void) const
{
    #ifdef __OPENNN_DEBUG__

    check();

    #endif

    #ifdef __OPENNN_DEBUG__

    const MultilayerPerceptron* multilayer_perceptron_pointer = neural_network_pointer->get_multilayer_perceptron_pointer();

    if(!multilayer_perceptron_pointer)
    {
        std::ostringstream buffer;

       buffer << "OpenNN Exception: TestingAnalysis class.\n"
              << "Matrix<double> perform_cumulative_gain_analysis(void) const method.\n"
              << "Pointer to multilayer perceptron in neural network is NULL.\n";

      throw std::logic_error(buffer.str());
    }

    #endif

    #ifdef __OPENNN_DEBUG__

    const size_t inputs_number = multilayer_perceptron_pointer->get_inputs_number();

    // Control sentence

    const Variables& variables = data_set_pointer->get_variables();

    if(inputs_number != variables.count_inputs_number())
    {
        std::ostringstream buffer;

        buffer << "OpenNN Exception: TestingAnalysis class." << std::endl
               << "Matrix<double> perform_cumulative_gain_analysis(void) const method." << std::endl
               << "Number of inputs in neural network must be equal to number of inputs in data set." << std::endl;

       throw std::logic_error(buffer.str());
    }

    const size_t outputs_number = multilayer_perceptron_pointer->get_outputs_number();

    if(outputs_number != variables.count_targets_number())
    {
        std::ostringstream buffer;

       buffer << "OpenNN Exception: TestingAnalysis class." << std::endl
              << "Matrix<double> perform_cumulative_gain_analysis(void) const method." << std::endl
              << "Number of outputs in neural network must be equal to number of targets in data set." << std::endl;

       throw std::logic_error(buffer.str());
    }

    #endif

     const Matrix<double> input_data = data_set_pointer->arrange_testing_input_data();
     const Matrix<double> target_data = data_set_pointer->arrange_testing_target_data();

     const Matrix<double> output_data = neural_network_pointer->calculate_output_data(input_data);

     const Matrix<double> cumulative_gain = calculate_cumulative_gain(target_data, output_data);

     return(cumulative_gain);
}


// Matrix<double> calculate_cumulative_gain(const Matrix<double>& , const Matrix<double>&) const

/// Returns a matrix with the values of a cumulative gain chart.
/// The number of columns is two, the number of rows is 20.
/// @param target_data Testing target data.
/// @param output_data Testing output data.

Matrix<double> TestingAnalysis::calculate_cumulative_gain(const Matrix<double>& target_data, const Matrix<double>& output_data) const
{
    const size_t total_positives = calculate_positives_negatives_rate(target_data, output_data)[0];

    if(total_positives == 0)
    {
        std::ostringstream buffer;

        buffer << "OpenNN Exception: TestingAnalysis class.\n"
               << "Matrix<double> calculate_cumulative_gain(const Matrix<double>&, const Matrix<double>&) const.\n"
               << "Number of positive instances (" << total_positives << ") must be greater than zero.\n";

        throw std::logic_error(buffer.str());
     }

    const size_t rows_number = target_data.get_rows_number();

    const Matrix<double> target_output_data = output_data.assemble_columns(target_data);

    const Matrix<double> sorted_target_output_data = target_output_data.sort_greater_rows(0);

    const Vector<size_t> target_indices(1,1);

    const Matrix<double> sorted_target_data = sorted_target_output_data.arrange_submatrix_columns(target_indices);

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
            if(sorted_target_data(j, 0) == 1.0)
            {
                 positives++;
            }
        }

        cumulative_gain(i + 1, 0) = (double) percentage;
        cumulative_gain(i + 1, 1) = (double) positives/(double)(total_positives);
    }

    return(cumulative_gain);
}


// Matrix<double> calculate_cumulative_gain(const Matrix<double>& , const Matrix<double>&) const

/// Returns a matrix with the values of a cumulative gain chart for the negative instances.
/// The number of columns is two, the number of rows is 20.
/// @param target_data Testing target data.
/// @param output_data Testing output data.

Matrix<double> TestingAnalysis::calculate_negative_cumulative_gain(const Matrix<double>& target_data, const Matrix<double>& output_data) const
{
    const size_t total_negatives = calculate_positives_negatives_rate(target_data, output_data)[1];

    if(total_negatives == 0)
    {
        std::ostringstream buffer;

        buffer << "OpenNN Exception: TestingAnalysis class.\n"
               << "Matrix<double> calculate_negative_cumulative_gain(const Matrix<double>&, const Matrix<double>&) const.\n"
               << "Number of negative instances (" << total_negatives << ") must be greater than zero.\n";

        throw std::logic_error(buffer.str());
     }

    const size_t rows_number = target_data.get_rows_number();

    const Matrix<double> target_output_data = output_data.assemble_columns(target_data);

    const Matrix<double> sorted_target_output_data = target_output_data.sort_greater_rows(0);

    const Vector<size_t> target_indices(1,1);

    const Matrix<double> sorted_target_data = sorted_target_output_data.arrange_submatrix_columns(target_indices);

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
            if(sorted_target_data(j, 0) == 0.0)
            {
                 negatives++;
            }
        }

        negative_cumulative_gain(i + 1, 0) = (double) percentage;
        negative_cumulative_gain(i + 1, 1) = (double) negatives/(double)(total_negatives);
    }

    return(negative_cumulative_gain);
}


// Matrix<double> perform_lift_chart_analysis(void) const

/// Performs a lift chart analysis.
/// Returns a matrix with the values of a lift chart.

Matrix<double> TestingAnalysis::perform_lift_chart_analysis(void) const
{
    #ifdef __OPENNN_DEBUG__

    check();

    #endif

    #ifdef __OPENNN_DEBUG__

    const MultilayerPerceptron* multilayer_perceptron_pointer = neural_network_pointer->get_multilayer_perceptron_pointer();

    if(!multilayer_perceptron_pointer)
    {
        std::ostringstream buffer;

       buffer << "OpenNN Exception: TestingAnalysis class.\n"
              << "Matrix<double> perform_lift_chart_analysis(void) const method.\n"
              << "Pointer to multilayer perceptron in neural network is NULL.\n";

      throw std::logic_error(buffer.str());
    }

    #endif

    #ifdef __OPENNN_DEBUG__

    const size_t inputs_number = multilayer_perceptron_pointer->get_inputs_number();

    // Control sentence

    const Variables& variables = data_set_pointer->get_variables();

    if(inputs_number != variables.count_inputs_number())
    {
        std::ostringstream buffer;

        buffer << "OpenNN Exception: TestingAnalysis class." << std::endl
               << "Matrix<double> perform_lift_chart_analysis(void) const method." << std::endl
               << "Number of inputs in neural network must be equal to number of inputs in data set." << std::endl;

       throw std::logic_error(buffer.str());
    }

    const size_t outputs_number = multilayer_perceptron_pointer->get_outputs_number();

    if(outputs_number != variables.count_targets_number())
    {
        std::ostringstream buffer;

       buffer << "OpenNN Exception: TestingAnalysis class." << std::endl
              << "Matrix<double> perform_lift_chart_analysis(void) const method." << std::endl
              << "Number of outputs in neural network must be equal to number of targets in data set." << std::endl;

       throw std::logic_error(buffer.str());
    }

    #endif

     const Matrix<double> input_data = data_set_pointer->arrange_testing_input_data();
     const Matrix<double> target_data = data_set_pointer->arrange_testing_target_data();

     const Matrix<double> output_data = neural_network_pointer->calculate_output_data(input_data);

     const Matrix<double> cumulative_gain = calculate_cumulative_gain(target_data, output_data);
     const Matrix<double> lift_chart = calculate_lift_chart(cumulative_gain);

     return(lift_chart);
}


// Matrix<double> calculate_lift_chart(const Matrix<double>& , const Matrix<double>&) const

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

    for(int i = 1; i < rows_number; i++)
    {
        lift_chart(i, 0) = cumulative_gain(i, 0);
        lift_chart(i, 1) = (double) cumulative_gain(i, 1)/(double)cumulative_gain(i, 0);
    }

    return(lift_chart);
}


// Matrix<double> perform_Kolmogorov_Smirnov_analysis(void) const

/// Performs a Kolmogorov-Smirnov analysis, which consists of the cumulative gain for the positive instances and the cumulative
/// gain for the negative instances. It returns a Kolmogorov-Smirnov results structure, which consists of:
/// <ul>
/// <li> Positive cumulative gain
/// <li> Negative cumulative gain
/// <li> Maximum gain
/// </ul>

TestingAnalysis::KolmogorovSmirnovResults TestingAnalysis::perform_Kolmogorov_Smirnov_analysis(void) const
{
    #ifdef __OPENNN_DEBUG__

    check();

    #endif

    #ifdef __OPENNN_DEBUG__

    const MultilayerPerceptron* multilayer_perceptron_pointer = neural_network_pointer->get_multilayer_perceptron_pointer();

    if(!multilayer_perceptron_pointer)
    {
        std::ostringstream buffer;

       buffer << "OpenNN Exception: TestingAnalysis class.\n"
              << "Matrix<double> perform_Kolmogorov_Smirnov_analysis(void) const method.\n"
              << "Pointer to multilayer perceptron in neural network is NULL.\n";

      throw std::logic_error(buffer.str());
    }

    #endif

    #ifdef __OPENNN_DEBUG__

    const size_t inputs_number = multilayer_perceptron_pointer->get_inputs_number();

    // Control sentence

    const Variables& variables = data_set_pointer->get_variables();

    if(inputs_number != variables.count_inputs_number())
    {
        std::ostringstream buffer;

        buffer << "OpenNN Exception: TestingAnalysis class." << std::endl
               << "Matrix<double> perform_Kolmogorov_Smirnov_analysis(void) const method." << std::endl
               << "Number of inputs in neural network must be equal to number of inputs in data set." << std::endl;

       throw std::logic_error(buffer.str());
    }

    const size_t outputs_number = multilayer_perceptron_pointer->get_outputs_number();

    if(outputs_number != variables.count_targets_number())
    {
        std::ostringstream buffer;

       buffer << "OpenNN Exception: TestingAnalysis class." << std::endl
              << "Matrix<double> perform_Kolmogorov_Smirnov_analysis(void) const method." << std::endl
              << "Number of outputs in neural network must be equal to number of targets in data set." << std::endl;

       throw std::logic_error(buffer.str());
    }

    #endif

     const Matrix<double> input_data = data_set_pointer->arrange_testing_input_data();
     const Matrix<double> target_data = data_set_pointer->arrange_testing_target_data();

     const Matrix<double> output_data = neural_network_pointer->calculate_output_data(input_data);

     TestingAnalysis::KolmogorovSmirnovResults Kolmogorov_Smirnov_results;

     Kolmogorov_Smirnov_results.positive_cumulative_gain = calculate_cumulative_gain(target_data, output_data);
     Kolmogorov_Smirnov_results.negative_cumulative_gain = calculate_negative_cumulative_gain(target_data, output_data);
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
        std::ostringstream buffer;

       buffer << "OpenNN Exception: TestingAnalysis class.\n"
              << "Matrix<double> calculate_maximum_gain(void) const method.\n"
              << "Positive and negative cumulative gain matrix must have the same rows number.\n";

      throw std::logic_error(buffer.str());
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


// Matrix<double> perform_calibration_plot_analysis(void) const

/// Performs a calibration plot analysis.

Matrix<double> TestingAnalysis::perform_calibration_plot_analysis(void) const
{
    #ifdef __OPENNN_DEBUG__

    check();

    #endif

    #ifdef __OPENNN_DEBUG__

    const MultilayerPerceptron* multilayer_perceptron_pointer = neural_network_pointer->get_multilayer_perceptron_pointer();

    if(!multilayer_perceptron_pointer)
    {
        std::ostringstream buffer;

       buffer << "OpenNN Exception: TestingAnalysis class.\n"
              << "Matrix<double> perform_calibration_plot_analysis(void) const method.\n"
              << "Pointer to multilayer perceptron in neural network is NULL.\n";

      throw std::logic_error(buffer.str());
    }

    #endif

    #ifdef __OPENNN_DEBUG__

    const size_t inputs_number = multilayer_perceptron_pointer->get_inputs_number();

    // Control sentence

    const Variables& variables = data_set_pointer->get_variables();

    if(inputs_number != variables.count_inputs_number())
    {
        std::ostringstream buffer;

        buffer << "OpenNN Exception: TestingAnalysis class." << std::endl
               << "Matrix<double> perform_calibration_plot_analysis(void) const method." << std::endl
               << "Number of inputs in neural network must be equal to number of inputs in data set." << std::endl;

       throw std::logic_error(buffer.str());
    }

    const size_t outputs_number = multilayer_perceptron_pointer->get_outputs_number();

    if(outputs_number != variables.count_targets_number())
    {
        std::ostringstream buffer;

       buffer << "OpenNN Exception: TestingAnalysis class." << std::endl
              << "Matrix<double> perform_calibration_plot_analysis(void) const method." << std::endl
              << "Number of outputs in neural network must be equal to number of targets in data set." << std::endl;

       throw std::logic_error(buffer.str());
    }

    #endif

    const Matrix<double> input_data = data_set_pointer->arrange_testing_input_data();
    const Matrix<double> target_data = data_set_pointer->arrange_testing_target_data();

    const Matrix<double> output_data = neural_network_pointer->calculate_output_data(input_data);

    const Matrix<double> calibration_plot = calculate_calibration_plot(target_data, output_data);

    return(calibration_plot);

}

//Matrix<double> calculate_calibration_plot(const Matrix<double>&, const Matrix<double>&) const

/// Returns a matix with the values of a calibration plot.
/// Number of columns is two. Number of rows varies depending on the distribution of positive targets.
/// @param target_data Testing target data.
/// @param output_data Testing output data.

Matrix<double> TestingAnalysis::calculate_calibration_plot(const Matrix<double>& target_data, const Matrix<double>& output_data) const
{
    std::cout << "Calibration plot" << std::endl;

    const size_t rows_number = target_data.get_rows_number();   

    std::cout << "Rows number: " << rows_number << std::endl;

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
            if(output_data(j, 0) >= (probability - 0.1) && output_data(j, 0) < probability)
            {
                count++;

                sum += output_data(j, 0);

                if(target_data(j, 0) == 1.0)
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
            calibration_plot(i, 0) = (double)sum/(double)count;
            calibration_plot(i, 1) = (double)positives/(double)count;
        }
     }

    // Last point

    calibration_plot(points_number+1,0) = 1.0;
    calibration_plot(points_number+1,1) = 1.0;

   // Subtracts calibration plot rows with value -1

    size_t points_number_subtracted = 0;

    size_t current_rows_number;

    while(calibration_plot.arrange_column(0).contains(-1))
     {
        current_rows_number = calibration_plot.get_rows_number();

         for(size_t i = 1; i < points_number - points_number_subtracted+1; i++)
         {
             if(calibration_plot(i, 0) == -1)
             {
                 calibration_plot.subtract_row(i);

                 points_number_subtracted++;
             }
         }
     }

    return(calibration_plot);
}


// Vector< Histogram <double> > calculate_output_histogram(const Matrix<double>&, const size_t&) const;

/// Returns the histogram of the outputs.
/// @param output_data Testing output data.
/// @param bins_number Number of bins of the histogram.

Vector< Histogram <double> > TestingAnalysis::calculate_output_histogram(const Matrix<double>& output_data, const size_t& bins_number) const
{
    Vector< Histogram <double> > output_histogram (1);

    output_histogram [0] = output_data.arrange_column(0).calculate_histogram(bins_number);

    return(output_histogram);
}


// TestingAnalysis::BinaryClassificationRates TestingAnalysis::calculate_binary_classification_rates(void) const

/// Returns a structure with the binary classification rates, which has the indices of the true positive, false negative, false positive and true negative instances.
/// <ul>
/// <li> True positive instances
/// <li> False positive instances
/// <li> False negative instances
/// <li> True negative instances
/// </ul>

TestingAnalysis::BinaryClassifcationRates TestingAnalysis::calculate_binary_classification_rates(void) const
{
    #ifdef __OPENNN_DEBUG__

     check();

    #endif

    #ifdef __OPENNN_DEBUG__

    const MultilayerPerceptron* multilayer_perceptron_pointer = neural_network_pointer->get_multilayer_perceptron_pointer();

    if(!multilayer_perceptron_pointer)
    {
        std::ostringstream buffer;

       buffer << "OpenNN Exception: TestingAnalysis class.\n"
              << "BinaryClassificationRates calculate_binary_classification_rates(void) const method.\n"
              << "Pointer to multilayer perceptron in neural network is NULL.\n";

       throw std::logic_error(buffer.str());
    }

    #endif

    #ifdef __OPENNN_DEBUG__

    const size_t inputs_number = multilayer_perceptron_pointer->get_inputs_number();

    // Control sentence

    const Variables& variables = data_set_pointer->get_variables();

    if(inputs_number != variables.count_inputs_number())
    {
        std::ostringstream buffer;

        buffer << "OpenNN Exception: TestingAnalysis class." << std::endl
               << "BinaryClassificationRates calculate_binary_classification_rates(void) const method." << std::endl
               << "Number of inputs in neural network must be equal to number of inputs in data set." << std::endl;

       throw std::logic_error(buffer.str());
    }

    const size_t outputs_number = multilayer_perceptron_pointer->get_outputs_number();

    if(outputs_number != variables.count_targets_number())
    {
        std::ostringstream buffer;

       buffer << "OpenNN Exception: TestingAnalysis class." << std::endl
              << "BinaryClassificationRates calculate_binary_classification_rates(void) const method." << std::endl
              << "Number of outputs in neural network must be equal to number of targets in data set." << std::endl;

       throw std::logic_error(buffer.str());
    }

    #endif

    const Matrix<double> input_data = data_set_pointer->arrange_testing_input_data();
    const Matrix<double> target_data = data_set_pointer->arrange_testing_target_data();

    const Matrix<double> output_data = neural_network_pointer->calculate_output_data(input_data);

    const Vector<size_t> testing_indices = data_set_pointer->get_instances().arrange_testing_indices();

    double decision_threshold;

    if(neural_network_pointer->get_probabilistic_layer_pointer() != NULL)
    {
        decision_threshold = neural_network_pointer->get_probabilistic_layer_pointer()->get_decision_threshold();
    }
    else
    {
        decision_threshold = 0.5;
    }

    BinaryClassifcationRates binary_classification_rates;

    binary_classification_rates.true_positive_instances = calculate_true_positive_instances(target_data, output_data, testing_indices, decision_threshold);
    binary_classification_rates.false_positive_instances = calculate_false_positive_instances(target_data, output_data, testing_indices, decision_threshold);
    binary_classification_rates.false_negative_instances = calculate_false_negative_instances(target_data, output_data, testing_indices, decision_threshold);
    binary_classification_rates.true_negative_instances = calculate_true_negative_instances(target_data, output_data, testing_indices, decision_threshold);

    return(binary_classification_rates);
}


// Vector<size_t> calculate_true_positive_instances (const Matrix<double>&, const Matrix<double>&, const Vector<size_t>&, const double&) const

/// Returns a vector with the indices of the true positive instances.
/// The size of the vector is the number of true positive instances.
/// @param target_data Testing target data.
/// @param output_data Testing output_data.
/// @param testing_indices Indices of testing data.
/// @param decision_threshold Decision threshold.

Vector<size_t> TestingAnalysis::calculate_true_positive_instances(const Matrix<double>& target_data, const Matrix<double>& output_data, const Vector<size_t>& testing_indices, const double& decision_threshold) const
{
    const size_t rows_number = target_data.get_rows_number();

    Vector<size_t> true_positive_instances;

    for(size_t i = 0; i < rows_number; i++)
    {
        if(target_data(i,0) >= decision_threshold && output_data(i,0) >= decision_threshold)
        {
            true_positive_instances.push_back(testing_indices[i]);
        }
    }

    return(true_positive_instances);
}


// Vector<size_t> calculate_false_positive_instances(const Matrix<double>& , const Matrix<double>&, const Vector<size_t>&, const double&) const

/// Returns a vector with the indices of the false positive instances.
/// The size of the vector is the number of false positive instances.
/// @param target_data Testing target data.
/// @param output_data Testing output data.
/// @param testing_indices Indices of the testing data
/// @param decision_threshold Decision threshold.

Vector<size_t> TestingAnalysis::calculate_false_positive_instances(const Matrix<double>& target_data, const Matrix<double>& output_data, const Vector<size_t>& testing_indices, const double& decision_threshold) const
{
    const size_t rows_number = target_data.get_rows_number();

    Vector<size_t> false_positive_instances;

    for(size_t i = 0; i < rows_number; i++)
    {
        if(target_data(i,0) < decision_threshold && output_data(i,0) >= decision_threshold)
        {
            false_positive_instances.push_back(testing_indices[i]);
        }
    }

    return(false_positive_instances);
}


//Vector<size_t> calculate_false_negative_instances(const Matrix<double>& , const Matrix<double>&, const Vector<size_t>&, const double&)

/// Returns a vector with the indices of the false negative instances.
/// The size of the vector is the number of false negative instances.
/// @param target_data Testing target data.
/// @param output_data Testing output data.
/// @param testing_indices Indices of the testing data
/// @param decision_threshold Decision threshold.

Vector<size_t> TestingAnalysis::calculate_false_negative_instances(const Matrix<double>& target_data, const Matrix<double>& output_data, const Vector<size_t>& testing_indices, const double& decision_threshold) const
{
    const size_t rows_number = target_data.get_rows_number();

    Vector<size_t> false_negative_instances;

    for(size_t i = 0; i < rows_number; i++)
    {
        if(target_data(i,0) > decision_threshold && output_data(i,0) < decision_threshold)
        {
            false_negative_instances.push_back(testing_indices[i]);
        }
    }

    return(false_negative_instances);
}


// Vector<size_t> calculate_true_negative_instances(const Matrix<double>& , const Matrix<double>&, const Vector<size_t>&, const double&) const

/// Returns a vector with the indices of the true negative instances.
/// The size of the vector is the number of true negative instances.
/// @param target_data Testing target data.
/// @param output_data Testinga output data.
/// @param testing_indices Indices of the testing data
/// @param decision_threshold Decision threshold.

Vector<size_t> TestingAnalysis::calculate_true_negative_instances(const Matrix<double>& target_data, const Matrix<double>& output_data, const Vector<size_t>& testing_indices, const double& decision_threshold) const
{
    Vector<size_t> true_negative_instances;

    const size_t rows_number = target_data.get_rows_number();

    for(size_t i = 0; i < rows_number; i++)
    {
        if(target_data(i,0) < decision_threshold && output_data(i,0) < decision_threshold)
        {
            true_negative_instances.push_back(testing_indices[i]);
        }
    }

    return(true_negative_instances);
}


// Matrix< Vector<size_t> > calculate_multiple_classification_rates(void) const

/// Returns a matrix of subvectors which have the rates for a multiple classification problem.

Matrix< Vector<size_t> > TestingAnalysis::calculate_multiple_classification_rates(void) const
{
    #ifdef __OPENNN_DEBUG__

     check();

    #endif

    #ifdef __OPENNN_DEBUG__

    const MultilayerPerceptron* multilayer_perceptron_pointer = neural_network_pointer->get_multilayer_perceptron_pointer();

    if(!multilayer_perceptron_pointer)
    {
    std::ostringstream buffer;

    buffer << "OpenNN Exception: TestingAnalysis class.\n"
          << "BinaryClassificationRates calculate_binary_classification_rates(void) const method.\n"
          << "Pointer to multilayer perceptron in neural network is NULL.\n";

    throw std::logic_error(buffer.str());
    }

    #endif

    #ifdef __OPENNN_DEBUG__

    const size_t inputs_number = multilayer_perceptron_pointer->get_inputs_number();

    // Control sentence

    const Variables& variables = data_set_pointer->get_variables();

    if(inputs_number != variables.count_inputs_number())
    {
    std::ostringstream buffer;

    buffer << "OpenNN Exception: TestingAnalysis class." << std::endl
           << "BinaryClassificationRates calculate_binary_classification_rates(void) const method." << std::endl
           << "Number of inputs in neural network must be equal to number of inputs in data set." << std::endl;

    throw std::logic_error(buffer.str());
    }

    const size_t outputs_number = multilayer_perceptron_pointer->get_outputs_number();

    if(outputs_number != variables.count_targets_number())
    {
    std::ostringstream buffer;

    buffer << "OpenNN Exception: TestingAnalysis class." << std::endl
          << "BinaryClassificationRates calculate_binary_classification_rates(void) const method." << std::endl
          << "Number of outputs in neural network must be equal to number of targets in data set." << std::endl;

    throw std::logic_error(buffer.str());
    }

    #endif

    const Matrix<double> input_data = data_set_pointer->arrange_testing_input_data();
    const Matrix<double> target_data = data_set_pointer->arrange_testing_target_data();

    const Matrix<double> output_data = neural_network_pointer->calculate_output_data(input_data);

    const Vector<size_t> testing_indices = data_set_pointer->get_instances().arrange_testing_indices();

    return(calculate_multiple_classification_rates(target_data, output_data, testing_indices));
}


// Matrix< Vector<size_t> > calculate_multiple_classification_rates(void) method

/// Returns a matrix of subvectors which have the rates for a multiple classification problem.
/// The number of rows of the matrix is the number targets.
/// The number of columns of the matrix is the number of columns of the target data.

Matrix< Vector<size_t> > TestingAnalysis::calculate_multiple_classification_rates(const Matrix<double>& target_data, const Matrix<double>& output_data, const Vector<size_t>& testing_indices) const
{
    const size_t rows_number = target_data.get_rows_number();
    const size_t columns_number = output_data.get_columns_number();

    Matrix< Vector<size_t> > multiple_classification_rates(rows_number, columns_number);

    size_t target_index;
    size_t output_index;

    for(size_t i = 0; i < rows_number; i++)
    {
        target_index = target_data.arrange_row(i).calculate_maximal_index();
        output_index = output_data.arrange_row(i).calculate_maximal_index();

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
    std::ostringstream buffer;

    buffer << "OpenNN Exception: TestingAnalysis class.\n"
          << "Vector<double> calculate_error_autocorrelation(void) const method.\n"
          << "Pointer to multilayer perceptron in neural network is NULL.\n";

    throw std::logic_error(buffer.str());
    }

    #endif

    #ifdef __OPENNN_DEBUG__

    const size_t inputs_number = multilayer_perceptron_pointer->get_inputs_number();

    // Control sentence

    const Variables& variables = data_set_pointer->get_variables();

    if(inputs_number != variables.count_inputs_number())
    {
    std::ostringstream buffer;

    buffer << "OpenNN Exception: TestingAnalysis class." << std::endl
           << "Vector<double> calculate_error_autocorrelation(void) const method." << std::endl
           << "Number of inputs in neural network must be equal to number of inputs in data set." << std::endl;

    throw std::logic_error(buffer.str());
    }

    const size_t outputs_number = multilayer_perceptron_pointer->get_outputs_number();

    if(outputs_number != variables.count_targets_number())
    {
    std::ostringstream buffer;

    buffer << "OpenNN Exception: TestingAnalysis class." << std::endl
          << "Vector<double> calculate_error_autocorrelation(void) const method." << std::endl
          << "Number of outputs in neural network must be equal to number of targets in data set." << std::endl;

    throw std::logic_error(buffer.str());
    }

    #endif

    const Matrix<double> input_data = data_set_pointer->arrange_testing_input_data();
    const Matrix<double> target_data = data_set_pointer->arrange_testing_target_data();

    const Matrix<double> output_data = neural_network_pointer->calculate_output_data(input_data);

    const size_t targets_number = target_data.get_columns_number();

    Vector< Vector<double> > error_autocorrelation(targets_number);

    Matrix<double> error = target_data - output_data;

    for(size_t i = 0; i < targets_number; i++)
    {
        error_autocorrelation[i] = error.arrange_column(i).calculate_autocorrelation(maximum_lags_number);
    }

    return error_autocorrelation;
}


// Vector< Vector<double> > calculate_input_error_cross_correlation(const size_t&) const

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
    std::ostringstream buffer;

    buffer << "OpenNN Exception: TestingAnalysis class.\n"
          << "Vector<double> calculate_input_error_cross_correlation(void) const method.\n"
          << "Pointer to multilayer perceptron in neural network is NULL.\n";

    throw std::logic_error(buffer.str());
    }

    #endif

    #ifdef __OPENNN_DEBUG__

    const size_t inputs_number = multilayer_perceptron_pointer->get_inputs_number();

    // Control sentence

    const Variables& variables = data_set_pointer->get_variables();

    if(inputs_number != variables.count_inputs_number())
    {
    std::ostringstream buffer;

    buffer << "OpenNN Exception: TestingAnalysis class." << std::endl
           << "Vector<double> calculate_input_error_cross_correlation(void) const method." << std::endl
           << "Number of inputs in neural network must be equal to number of inputs in data set." << std::endl;

    throw std::logic_error(buffer.str());
    }

    const size_t outputs_number = multilayer_perceptron_pointer->get_outputs_number();

    if(outputs_number != variables.count_targets_number())
    {
    std::ostringstream buffer;

    buffer << "OpenNN Exception: TestingAnalysis class." << std::endl
          << "Vector<double> calculate_input_error_cross_correlation(void) const method." << std::endl
          << "Number of outputs in neural network must be equal to number of targets in data set." << std::endl;

    throw std::logic_error(buffer.str());
    }

    #endif

    const Matrix<double> input_data = data_set_pointer->arrange_testing_input_data();
    const Matrix<double> target_data = data_set_pointer->arrange_testing_target_data();

    const Matrix<double> output_data = neural_network_pointer->calculate_output_data(input_data);

    const size_t targets_number = target_data.get_columns_number();

    const Matrix<double> error = target_data - output_data;

    Vector<double> input_column;

    Vector< Vector<double> > input_error_cross_correlation(targets_number);

    for(size_t i = 0; i < targets_number; i++)
    {
        input_column = input_data.arrange_column(i);

        input_error_cross_correlation[i] = input_column.calculate_cross_correlation(error.arrange_column(i), maximum_lags_number);
    }

    return (input_error_cross_correlation);
}


// Vector<double> calculate_binary_classification_tests(void) method

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

Vector<double> TestingAnalysis::calculate_binary_classification_tests(void) const
{
   // Control sentence (if debug)

   #ifdef __OPENNN_DEBUG__

   const MultilayerPerceptron* multilayer_perceptron_pointer = neural_network_pointer->get_multilayer_perceptron_pointer();

   const size_t inputs_number = multilayer_perceptron_pointer->get_inputs_number();

   if(!data_set_pointer)
   {
      std::ostringstream buffer;

      buffer << "OpenNN Exception: TestingAnalysis class." << std::endl
             << "Vector<double> calculate_binary_classification_tests(void) const." << std::endl
             << "Data set is NULL." << std::endl;

      throw std::logic_error(buffer.str());
   }

   const Variables& variables = data_set_pointer->get_variables();

   const size_t targets_number = variables.count_targets_number();

   const size_t outputs_number = multilayer_perceptron_pointer->get_outputs_number();

   // Control sentence

   if(inputs_number != variables.count_inputs_number())
   {
      std::ostringstream buffer;

      buffer << "OpenNN Exception: TestingAnalysis class." << std::endl
             << "Vector<double> calculate_binary_classification_tests(void) const." << std::endl
             << "Number of inputs in neural network is not equal to number of inputs in data set." << std::endl;

      throw std::logic_error(buffer.str());
   }
   else if(outputs_number != 1)
   {
      std::ostringstream buffer;

      buffer << "OpenNN Exception: TestingAnalysis class." << std::endl
             << "Vector<double> calculate_binary_classification_tests(void) const." << std::endl
             << "Number of outputs in neural network must be one." << std::endl;

      throw std::logic_error(buffer.str());
   }
   else if(targets_number != 1)
   {
      std::ostringstream buffer;

      buffer << "OpenNN Exception: TestingAnalysis class." << std::endl
             << "Vector<double> calculate_binary_classification_tests(void) const." << std::endl
             << "Number of targets in data set must be one." << std::endl;

      throw std::logic_error(buffer.str());
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
      precision = (double) true_positive / (double)(true_positive + false_positive);
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
       F1_score = (double) 2*true_positive/(double) (2*true_positive + false_positive + false_negative);
   }

   // False positive rate

   double false_positive_rate;

   if(false_positive + true_negative == 0)
   {
       false_positive_rate = 0.0;
   }
   else
   {
       false_positive_rate = (double) false_positive/(double) (false_positive + true_negative);
   }

   // False discovery rate

   double false_discovery_rate;

   if(false_positive + true_positive == 0)
   {
       false_discovery_rate = 0.0;
   }
   else
   {
       false_discovery_rate = (double) false_positive /(double) (false_positive + true_positive);
   }

   // False negative rate

   double false_negative_rate;

   if(false_negative + true_positive == 0)
   {
       false_negative_rate = 0.0;
   }
   else
   {
       false_negative_rate = (double) false_negative /(double) (false_negative + true_positive);
   }

   // Negative predictive value

   double negative_predictive_value;

   if(true_negative + false_negative == 0)
   {
       negative_predictive_value = 0.0;
   }
   else
   {
       negative_predictive_value = (double) true_negative/(double) (true_negative + false_negative);
   }

   //Matthews correlation coefficient

   double Matthews_correlation_coefficient;

   if((true_positive + false_positive) * (true_positive + false_negative) * (true_negative + false_positive) * (true_negative + false_negative) == 0)
   {
       Matthews_correlation_coefficient = 0.0;
   }
   else
   {
       Matthews_correlation_coefficient = (double) (true_positive * true_negative - false_positive * false_negative) /(double) sqrt((double)((true_positive + false_positive) * (true_positive + false_negative) * (true_negative + false_positive) * (true_negative + false_negative)));
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
       markedness = precision + (double) true_negative/(double) (true_negative + false_positive) - 1;
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


// double calculate_logloss(void) const method

/// Returns the logloss for a binary classification problem

double TestingAnalysis::calculate_logloss(void) const
{
    #ifdef __OPENNN_DEBUG__

     check();

    #endif

    #ifdef __OPENNN_DEBUG__

    const MultilayerPerceptron* multilayer_perceptron_pointer = neural_network_pointer->get_multilayer_perceptron_pointer();

    if(!multilayer_perceptron_pointer)
    {
    std::ostringstream buffer;

    buffer << "OpenNN Exception: TestingAnalysis class.\n"
          << "Vector<double> calculate_input_error_cross_correlation(void) const method.\n"
          << "Pointer to multilayer perceptron in neural network is NULL.\n";

    throw std::logic_error(buffer.str());
    }

    #endif

    #ifdef __OPENNN_DEBUG__

    const size_t inputs_number = multilayer_perceptron_pointer->get_inputs_number();

    // Control sentence

    const Variables& variables = data_set_pointer->get_variables();

    if(inputs_number != variables.count_inputs_number())
    {
    std::ostringstream buffer;

    buffer << "OpenNN Exception: TestingAnalysis class." << std::endl
           << "Vector<double> calculate_input_error_cross_correlation(void) const method." << std::endl
           << "Number of inputs in neural network must be equal to number of inputs in data set." << std::endl;

    throw std::logic_error(buffer.str());
    }

    const size_t outputs_number = multilayer_perceptron_pointer->get_outputs_number();

    if(outputs_number != variables.count_targets_number())
    {
    std::ostringstream buffer;

    buffer << "OpenNN Exception: TestingAnalysis class." << std::endl
          << "Vector<double> calculate_input_error_cross_correlation(void) const method." << std::endl
          << "Number of outputs in neural network must be equal to number of targets in data set." << std::endl;

    throw std::logic_error(buffer.str());
    }

    #endif

    const Matrix<double> input_data = data_set_pointer->arrange_testing_input_data();
    const Matrix<double> target_data = data_set_pointer->arrange_testing_target_data();

    const Matrix<double> output_data = neural_network_pointer->calculate_output_data(input_data);

    const size_t testing_instances_number = target_data.get_rows_number();

    double logloss = 0.0;

    for(size_t i = 0; i < testing_instances_number; i++)
    {
        logloss += target_data(i,0)*log(output_data(i,0)) + (1-target_data(i,0))*log(1-output_data(i,0));
    }

    return(-logloss/testing_instances_number);
}


// std::string to_string(void) const method

/// Returns a string representation of the testing analysis object. 

std::string TestingAnalysis::to_string(void) const
{
   std::ostringstream buffer;

   buffer << "Testing analysis\n"
          << "Display: " << display << "\n";

   return(buffer.str());
}


// void print(void) const method

/// Prints to the standard output the string representation of this testing analysis object. 

void TestingAnalysis::print(void) const
{
   std::cout << to_string();
}


// tinyxml2::XMLDocument* to_XML(void) const method

/// Serializes the testing analysis object into a XML document of the TinyXML library. 
/// See the OpenNN manual for more information about the format of this element. 

tinyxml2::XMLDocument* TestingAnalysis::to_XML(void) const
{
    tinyxml2::XMLDocument* document = new tinyxml2::XMLDocument;

    std::ostringstream buffer;

    // Root element

    tinyxml2::XMLElement* testing_analysis_element = document->NewElement("TestingAnalysis");

    document->InsertFirstChild(testing_analysis_element);

    // Display

    tinyxml2::XMLElement* display_element = document->NewElement("Display");
    testing_analysis_element->LinkEndChild(display_element);

    buffer.str("");
    buffer << display;

    tinyxml2::XMLText* display_text = document->NewText(buffer.str().c_str());
    testing_analysis_element->LinkEndChild(display_text);

    return(document);
}


// void write_XML(tinyxml2::XMLPrinter&) const method

/// Serializes the testing analysis object into a XML document of the TinyXML library without keep the DOM tree in memory.
/// See the OpenNN manual for more information about the format of this document.

void TestingAnalysis::write_XML(tinyxml2::XMLPrinter& file_stream) const
{
    std::ostringstream buffer;
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
    std::ostringstream buffer;

    const tinyxml2::XMLElement* root_element = document.FirstChildElement("TestingAnalysis");

    if(!root_element)
    {
        buffer << "OpenNN Exception: TestingAnalysis class.\n"
               << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
               << "Testing analysis element is NULL.\n";

        throw std::logic_error(buffer.str());
    }

    // Display

    const tinyxml2::XMLElement* element = root_element->FirstChildElement("Display");

    if(element)
    {
       std::string new_display_string = element->GetText();

       try
       {
          set_display(new_display_string != "0");
       }
       catch(const std::logic_error& e)
       {
          std::cout << e.what() << std::endl;
       }
    }
}


// void save(const std::string&) const method

/// Saves to a XML file the members of this testing analysis object.
/// @param file_name Name of testing analysis XML file.

void TestingAnalysis::save(const std::string& file_name) const
{
    tinyxml2::XMLDocument* document = to_XML();

    document->SaveFile(file_name.c_str());

    delete document;
}


// void load(const std::string&) method

/// Loads from a XML file the members for this testing analysis object.
/// @param file_name Name of testing analysis XML file.

void TestingAnalysis::load(const std::string& file_name)
{
    set_default();

   tinyxml2::XMLDocument document;

   if(document.LoadFile(file_name.c_str()))
   {
       std::ostringstream buffer;

      buffer << "OpenNN Exception: Testing analysis class.\n"
             << "void load(const std::string&) method.\n"
             << "Cannot load XML file " << file_name << ".\n";

      throw std::logic_error(buffer.str());
   }

   from_XML(document);
}

}


// OpenNN: Open Neural Networks Library.
// Copyright (c) 2005-2016 Roberto Lopez.
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
