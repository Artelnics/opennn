/****************************************************************************************************************/
/*                                                                                                              */
/*   OpenNN: Open Neural Networks Library                                                                       */
/*   www.opennn.net                                                                                             */
/*                                                                                                              */
/*   D A T A   S E T   C L A S S                                                                                */
/*                                                                                                              */
/*   Artificial Intelligence Techniques SL                                                                      */
/*   artelnics@artelnics.com                                                                                    */
/*                                                                                                              */
/****************************************************************************************************************/

// OpenNN includes

#include "data_set.h"
#include "correlation_analysis.h"

#ifdef __OPENNN_CUDA__
#include <cuda_runtime.h>
#include <cublas_v2.h>

int mallocCUDA(double** A_d, int nBytes);
int memcpyCUDA(double* A_d, const double* A_h, int nBytes);
void freeCUDA(double* A_d);

#endif

namespace OpenNN
{

// DEFAULT CONSTRUCTOR

/// Default constructor. It creates a data set object with zero instances and zero inputs and target variables.
/// It also initializes the rest of class members to their default values.

DataSet::DataSet()
{
   set();

   set_default();
}


// DATA CONSTRUCTOR

DataSet::DataSet(const Eigen::MatrixXd& data)
{
    set(data);

    set_default();

}


/// Data constructor. It creates a data set object from a data matrix.
/// It also initializes the rest of class members to their default values.
/// @param data Data matrix.

DataSet::DataSet(const Matrix<double>& data)
{
   set(data);

   set_default();
}


// GENERAL CONSTRUCTOR

/// Instances and variables number constructor.
/// It creates a data set object with given instances and variables numbers.
/// All the variables are set as inputs.
/// It also initializes the rest of class members to their default values.
/// @param new_instances_number Number of instances in the data set.
/// @param new_variables_number Number of variables.

DataSet::DataSet(const size_t& new_instances_number, const size_t& new_variables_number)
{
   set(new_instances_number, new_variables_number);

   set_default();
}


// INSTANCES, INPUTS AND TARGETS NUMBERS CONSTRUCTORS

/// Instances number, input variables number and target variables number constructor.
/// It creates a data set object with given instances and inputs and target variables numbers.
/// It also initializes the rest of class members to their default values.
/// @param new_instances_number Number of instances in the data set.
/// @param new_inputs_number Number of input variables.
/// @param new_targets_number Number of target variables.

DataSet::DataSet(const size_t& new_instances_number, const size_t& new_inputs_number, const size_t& new_targets_number)
{
   set(new_instances_number, new_inputs_number, new_targets_number);

   set_default();
}


// XML CONSTRUCTOR

/// Sets the data set members from a XML document.
/// @param data_set_document TinyXML document containing the member data.

DataSet::DataSet(const tinyxml2::XMLDocument& data_set_document)
{
   set_default();

   from_XML(data_set_document);
}


// FILE CONSTRUCTOR

/// File constructor. It creates a data set object by loading the object members from a data file.
/// Please mind about the file format. This is specified in the User's Guide.
/// @param data_file_name Data file file name.

DataSet::DataSet(const string& data_file_name)
{
   set();

   set_default();

   set_data_file_name(data_file_name);

   load_data();
}


// FILE AND SEPARATOR CONSTRUCTOR

/// File and separator constructor. It creates a data set object by loading the object members from a data file.
/// It also sets a separator.
/// Please mind about the file format. This is specified in the User's Guide.
/// @param data_file_name Data file file name.
/// @param separator Data file file name.

DataSet::DataSet(const string& data_file_name, const string& separator)
{
   set();

   set_default();

   set_data_file_name(data_file_name);

   set_separator(separator);

   load_data();
}


// COPY CONSTRUCTOR

/// Copy constructor.
/// It creates a copy of an existing inputs targets data set object.
/// @param other_data_set Data set object to be copied.

DataSet::DataSet(const DataSet& other_data_set)
{
   set_default();

   set(other_data_set);
}


// DESTRUCTOR

/// Destructor.

DataSet::~DataSet()
{
}


// ASSIGNMENT OPERATOR

/// Assignment operator.
/// It assigns to the current object the members of an existing data set object.
/// @param other_data_set Data set object to be assigned.

DataSet& DataSet::operator = (const DataSet& other_data_set)
{
   if(this != &other_data_set)
   {
      data_file_name = other_data_set.data_file_name;

      // Data matrix

      data = other_data_set.data;

      // Variables

      variables = other_data_set.variables;

      // Instances

      instances = other_data_set.instances;

      // Utilities

      display = other_data_set.display;
   }

   return(*this);
}


// EQUAL TO OPERATOR


/// Equal to operator.
/// It compares this object with another object of the same class.
/// It returns true if the members of the two objects have the same values, and false otherwise.
/// @ param other_data_set Data set object to be compared with.

bool DataSet::operator == (const DataSet& other_data_set) const
{
   if(data_file_name == other_data_set.data_file_name
   && data == other_data_set.data
   && variables == other_data_set.variables
   && instances == other_data_set.instances
   && display == other_data_set.display)
   {
      return(true);
   }
   else
   {
      return(false);
   }
}


// METHODS


/// Returns a constant reference to the variables object composing this data set object.

const Variables& DataSet::get_variables() const
{
   return(variables);
}

/// Returns a pointer to the variables object composing this data set object.

Variables* DataSet::get_variables_pointer()
{
   return(&variables);
}


/// Returns a constant reference to the instances object composing this data set object.

const Instances& DataSet::get_instances() const
{
   return(instances);
}


/// Returns a pointer to the variables object composing this data set object.

Instances* DataSet::get_instances_pointer()
{
   return(&instances);
}


/// Returns true if messages from this class can be displayed on the screen,
/// or false if messages from this class can't be displayed on the screen.

const bool& DataSet::get_display() const
{
   return(display);
}


/// Returns true if the data set is a binary classification problem, false otherwise.

bool DataSet::is_binary_classification() const
{
    if(variables.get_targets_number() != 1)
    {
        return(false);
    }

    if(!get_targets().is_binary())
    {
        return(false);
    }

    return(true);
}


/// Returns true if the data set is a multiple classification problem, false otherwise.

bool DataSet::is_multiple_classification() const
{
    const Matrix<double> targets = get_targets();

    if(!targets.is_binary())
    {
        return(false);
    }

    for(size_t i = 0; i < targets.get_rows_number(); i++)
    {
        if(targets.get_row(i).calculate_sum() == 0.0)
        {
            return(false);
        }
    }

    return(true);
}


/// Returns true if the given variable is binary and false in other case.
/// @param variable_index Index of the variable that is going to be checked.

bool DataSet::is_binary_variable(const size_t& variable_index) const
{
    const Vector<size_t> used_instances = instances.get_used_indices();

    const Vector<size_t> missing_instances = missing_values.get_missing_instances(variable_index);

    const Vector<size_t> indices_remaining = used_instances.get_difference(missing_instances);

    const size_t instances_number = indices_remaining.size();

    for(size_t i = 0; i < instances_number; i++)
    {
        if(data(indices_remaining[i],variable_index) == 0.0 || data(indices_remaining[i],variable_index) == 1.0)
        {
            continue;
        }
        else
        {
            return false;
        }

    }

//    const size_t instances_number = instances.get_instances_number();

//    cout << "unique_elements: " << data.get_c.get_column(variable_index).get_unique_elements().vector_to_string('/') << endl;

//    for(size_t i = 0; i < instances_number; i++)
//    {
//        if(data(i,variable_index) == 0.0 || data(i,variable_index) == 1.0)
//        {
//            continue;
//        }
//        else
//        {
//            return false;
//        }

//    }

    return true;
}


/// Returns true if the given variable is binary and false in other case.
/// @param variable_name Name of the variable that is going to be checked.

bool DataSet::is_binary_variable(const string& variable_name) const
{
    const Vector<string> variable_names = variables.get_names();

    const Vector<size_t> variable_index = variable_names.calculate_equal_to_indices(variable_name);

#ifdef __OPENNN_DEBUG__

    const size_t variables_size = variable_index.size();

    if(variables_size == 0)
    {
       ostringstream buffer;

       buffer << "OpenNN Exception: DataSet class.\n"
              << "Vector<double> get_variable(const string&) const method.\n"
              << "Variable: " << variable_name << " does not exist.\n";

       throw logic_error(buffer.str());
    }

    if(variables_size > 1)
    {
       ostringstream buffer;

       buffer << "OpenNN Exception: DataSet class.\n"
              << "Vector<double> get_variable(const string&) const method.\n"
              << "Variable: " << variable_name << " appears more than once in the data set.\n";

       throw logic_error(buffer.str());
    }

#endif

    return is_binary_variable(variable_index[0]);
}


/// Returns true if the data matrix is empty, and false otherwise.

bool DataSet::empty() const
{
   return(data.empty());
}


/// Returns a reference to the data matrix in the data set.
/// The number of rows is equal to the number of instances.
/// The number of columns is equal to the number of variables.

const Matrix<double>& DataSet::get_data() const
{
   return(data);
}

const Eigen::MatrixXd DataSet::get_data_eigen() const
{
    const size_t variables_number = data.get_columns_number();
    const size_t instances_number = data.get_rows_number();

    Matrix<double> copy(data);

    const Eigen::Map<Eigen::MatrixXd> matrix_eigen((double *)copy.data(), instances_number, variables_number);

    return(matrix_eigen);
}


/// Returns a reference to the time series data matrix in the data set.
/// Only for time series problems.

const Matrix<double>& DataSet::get_time_series_data() const
{
   return(time_series_data);
}


/// Returns the submatrix of the data with the asked instances.
/// @param instances_indices Indices of the instances to return.

Matrix<double> DataSet::get_instances_submatrix_data(const Vector<size_t>& instances_indices) const
{
    return(data.get_submatrix_rows(instances_indices));
}


/// Returns the file type.

DataSet::FileType DataSet::get_file_type() const
{
    return(file_type);
}


/// Returns a string with the name of the file type.

string DataSet::write_file_type() const
{
    switch(file_type)
    {
        case TXT:
        return "txt";

        case DAT:
        return "dat";

        case CSV:
        return "csv";

        case ODS:
        return "ods";

        case XLSX:
        return "xlsx";

        case ARFF:
        return "arff";

        case JSON:
        return "json";

        default:
        {
           ostringstream buffer;

           buffer << "OpenNN Exception: DataSet class.\n"
                  << "string write_file_type() const method.\n"
                  << "Unknown file type.\n";

           throw logic_error(buffer.str());
        }
    }
}


/// Returns a string with the first cell for excel files.

string DataSet::write_first_cell() const
{
   return first_cell;
}


/// Returns a string with the last cell for excel files.

string DataSet::write_last_cell() const
{
   return last_cell;
}


/// Returns a string with the sheet number for excel files.

size_t DataSet::write_sheet_number() const
{
   return sheet_number;
}


/// Returns the project type.

DataSet::ProjectType DataSet::get_learning_task() const
{
    return learning_task;
}


/// Returns a string with the name of the project type.

string DataSet::write_learning_task() const
{
    switch(learning_task)
    {
        case Approximation:
        return "Approximation";

        case Classification:
        return "Classification";

        case Forecasting:
        return "Forecasting";

        case Association:
        return "Association";
    }

    return string();
}


/// Returns a reference to the missing values object in the data set.

const MissingValues& DataSet::get_missing_values() const
{
   return(missing_values);
}


/// Returns a pointer to the missing values object in the data set.

MissingValues* DataSet::get_missing_values_pointer()
{
   return(&missing_values);
}


/// Returns the name of the data file.

const string& DataSet::get_data_file_name() const
{
   return(data_file_name);
}


/// Returns true if the first line of the data file has a header with the names of the variables, and false otherwise.

const bool& DataSet::get_header_line() const
{
    return(header_line);
}


/// Returns true if the data file has rows label, and false otherwise.

const bool& DataSet::get_rows_label() const
{
    return(rows_label);
}


/// Returns the separator to be used in the data file.

const DataSet::Separator& DataSet::get_separator() const
{
    return(separator);
}


/// Returns the string which will be used as separator in the data file.

char DataSet::get_separator_char() const
{
    switch(separator)
    {
       case Space:
       {
          return(' ');
       }

       case Tab:
       {
          return('\t');
       }

        case Comma:
        {
           return(',');
        }

        case Semicolon:
        {
           return(';');
        }
    }

    return char();
}


/// Returns the string which will be used as separator in the data file.

string DataSet::write_separator() const
{
    switch(separator)
    {
       case Space:
       {
          return("Space");
       }

       case Tab:
       {
          return("Tab");
       }

        case Comma:
        {
           return("Comma");
        }

        case Semicolon:
        {
           return("Semicolon");
        }
    }

    return string();
}


/// Returns the string which will be used as label for the missing values in the data file.

const string& DataSet::get_missing_values_label() const
{
    return(missing_values_label);
}


/// Returns the value of the grouping factor which will be used for grouping the nominal variables.

const double& DataSet::get_grouping_factor() const
{
    return(grouping_factor);
}


/// Returns the number of lags to be used in a time series prediction application.

const size_t& DataSet::get_lags_number() const
{
    return(lags_number);
}


/// Returns the number of steps ahead to be used in a time series prediction application.

const size_t& DataSet::get_steps_ahead() const
{
    return(steps_ahead);
}


const size_t& DataSet::get_time_index() const
{
    return(time_index);
}


/// Returns true if the data set will be used for an association application, and false otherwise.
/// In an association problem the target data is equal to the input data.

const bool& DataSet::get_autoassociation() const
{
    return(association);
}


/// Returns the indices of the angular variables in the data set.
/// When loading a data set with angular variables,
/// a transformation of the data will be performed in order to avoid discontinuities(from 359 degrees to 1 degree).

const Vector<size_t>& DataSet::get_angular_variables() const
{
    return(angular_variables);
}


/// Returns the units used for the angular variables(Radians or Degrees).

const DataSet::AngularUnits& DataSet::get_angular_units() const
{
    return(angular_units);
}


/// Returns a value of the scaling-unscaling method enumeration from a string containing the name of that method.
/// @param scaling_unscaling_method String with the name of the scaling and unscaling method.

DataSet::ScalingUnscalingMethod DataSet::get_scaling_unscaling_method(const string& scaling_unscaling_method)
{
    if(scaling_unscaling_method == "NoScaling")
    {
        return(NoScaling);
    }
    else if(scaling_unscaling_method == "NoUnscaling")
    {
        return(NoUnscaling);
    }
    else if(scaling_unscaling_method == "MinimumMaximum")
    {
        return(MinimumMaximum);
    }
    else if(scaling_unscaling_method == "Logarithmic")
    {
        return(Logarithmic);
    }
    else if(scaling_unscaling_method == "MeanStandardDeviation")
    {
        return(MeanStandardDeviation);
    }
    else if(scaling_unscaling_method == "StandardDeviation")
    {
        return(StandardDeviation);
    }
    else
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: DataSet class.\n"
               << "static ScalingUnscalingMethod get_scaling_unscaling_method(const string).\n"
               << "Unknown scaling-unscaling method: " << scaling_unscaling_method << ".\n";

        throw logic_error(buffer.str());
    }
}


/// Returns a matrix with the training instances in the data set.
/// The number of rows is the number of training instances.
/// The number of columns is the number of variables.

Matrix<double> DataSet::get_training_data() const
{
   const size_t variables_number = variables.get_variables_number();

   Vector<size_t> variables_indices(0, 1,variables_number-1);

   const Vector<size_t> training_indices = instances.get_training_indices();

   return(data.get_submatrix(training_indices, variables_indices));
}

Eigen::MatrixXd DataSet::get_training_data_eigen() const
{
    Matrix<double> data = get_training_data();
    const size_t rows_number = data.get_rows_number();
    const size_t columns_number = data.get_columns_number();

    const Eigen::Map<Eigen::MatrixXd> data_eigen((double*)data.data(), rows_number, columns_number);

    return(data_eigen);
}


/// Returns a matrix with the selection instances in the data set.
/// The number of rows is the number of selection instances.
/// The number of columns is the number of variables.

Matrix<double> DataSet::get_selection_data() const
{
   const size_t variables_number = variables.get_variables_number();

   const Vector<size_t> selection_indices = instances.get_selection_indices();

   Vector<size_t> variables_indices(0, 1,variables_number-1);

   return(data.get_submatrix(selection_indices, variables_indices));
}


Eigen::MatrixXd DataSet::get_selection_data_eigen() const
{
    Matrix<double> data = get_selection_data();
    const size_t rows_number = data.get_rows_number();
    const size_t columns_number = data.get_columns_number();

    const Eigen::Map<Eigen::MatrixXd> data_eigen((double*)data.data(), rows_number, columns_number);

    return(data_eigen);
}


/// Returns a matrix with the testing instances in the data set.
/// The number of rows is the number of testing instances.
/// The number of columns is the number of variables.

Matrix<double> DataSet::get_testing_data() const
{
   const size_t variables_number = variables.get_variables_number();
   Vector<size_t> variables_indices(0, 1,variables_number-1);

   const Vector<size_t> testing_indices = instances.get_testing_indices();

   return(data.get_submatrix(testing_indices, variables_indices));
}


Eigen::MatrixXd DataSet::get_testing_data_eigen() const
{
    Matrix<double> data = get_testing_data();
    const size_t rows_number = data.get_rows_number();
    const size_t columns_number = data.get_columns_number();

    const Eigen::Map<Eigen::MatrixXd> data_eigen((double*)data.data(), rows_number, columns_number);

    return(data_eigen);
}


/// Returns a matrix with the input variables in the data set.
/// The number of rows is the number of instances.
/// The number of columns is the number of input variables.

Matrix<double> DataSet::get_inputs() const
{
   const size_t instances_number = instances.get_instances_number();
   const Vector<size_t> indices(0, 1,instances_number-1);

   const Vector<size_t> input_indices = variables.get_inputs_indices();

   return(data.get_submatrix(indices, input_indices));
}


Eigen::MatrixXd DataSet::get_inputs_eigen() const
{
    Matrix<double> data = get_inputs();
    const size_t rows_number = data.get_rows_number();
    const size_t columns_number = data.get_columns_number();

    const Eigen::Map<Eigen::MatrixXd> data_eigen((double*)data.data(), rows_number, columns_number);

    return(data_eigen);
}


/// Returns a matrix with the target variables in the data set.
/// The number of rows is the number of instances.
/// The number of columns is the number of target variables.

Matrix<double> DataSet::get_targets() const
{
   const size_t instances_number = instances.get_instances_number();
   const Vector<size_t> indices(0, 1, instances_number-1);

   const Vector<size_t> targets_indices = variables.get_targets_indices();

   return(data.get_submatrix(indices, targets_indices));
}


Eigen::MatrixXd DataSet::get_targets_eigen() const
{
    Matrix<double> data = get_targets();
    const size_t rows_number = data.get_rows_number();
    const size_t columns_number = data.get_columns_number();

    const Eigen::Map<Eigen::MatrixXd> data_eigen((double*)data.data(), rows_number, columns_number);

    return(data_eigen);
}


Matrix<double> DataSet::get_inputs(const Vector<size_t>& instances_indices) const
{
    const Vector<size_t> input_indices = variables.get_inputs_indices();

    return data.get_submatrix(instances_indices, input_indices);
}


Matrix<double> DataSet::get_targets(const Vector<size_t>& instances_indices) const
{
    const Vector<size_t> target_indices = variables.get_targets_indices();

    return data.get_submatrix(instances_indices, target_indices);
}

Matrix<double> DataSet::get_used_data() const
{
   const Vector<size_t> instances_indices = instances.get_used_indices();

   const Vector<size_t> variables_indices = variables.get_used_indices();

   return(data.get_submatrix(instances_indices, variables_indices));
}


/// Returns a matrix with the input variables of the used instances in the data set.
/// The number of rows is the number of used instances.
/// The number of columns is the number of input variables.

Matrix<double> DataSet::get_used_inputs() const
{
   const Vector<size_t> indices = instances.get_used_indices();

   const Vector<size_t> input_indices = variables.get_inputs_indices();

   return(data.get_submatrix(indices, input_indices));
}


/// Returns a matrix with the target variables of the used instances in the data set.
/// The number of rows is the number of used instances.
/// The number of columns is the number of target variables.

Matrix<double> DataSet::get_used_targets() const
{
   const Vector<size_t> indices = instances.get_used_indices();

   const Vector<size_t> targets_indices = variables.get_targets_indices();

   return(data.get_submatrix(indices, targets_indices));
}


/// Returns a matrix with training instances and input variables.
/// The number of rows is the number of training instances.
/// The number of columns is the number of input variables.

Matrix<double> DataSet::get_training_inputs() const
{
   const Vector<size_t> inputs_indices = variables.get_inputs_indices();

   const Vector<size_t> training_indices = instances.get_training_indices();

   return(data.get_submatrix(training_indices, inputs_indices));
}


Eigen::MatrixXd DataSet::get_training_inputs_eigen() const
{
    Matrix<double> data = get_training_inputs();
    const size_t rows_number = data.get_rows_number();
    const size_t columns_number = data.get_columns_number();

    const Eigen::Map<Eigen::MatrixXd> data_eigen((double*)data.data(), rows_number, columns_number);

    return(data_eigen);
}


/// Returns a matrix with training instances and target variables.
/// The number of rows is the number of training instances.
/// The number of columns is the number of target variables.

Matrix<double> DataSet::get_training_targets() const
{
   const Vector<size_t> training_indices = instances.get_training_indices();

   const Vector<size_t> targets_indices = variables.get_targets_indices();

   return(data.get_submatrix(training_indices, targets_indices));
}



Eigen::MatrixXd DataSet::get_training_targets_eigen() const
{
    Matrix<double> data = get_training_targets();
    const size_t rows_number = data.get_rows_number();
    const size_t columns_number = data.get_columns_number();

    const Eigen::Map<Eigen::MatrixXd> data_eigen((double*)data.data(), rows_number, columns_number);

    return(data_eigen);
}


/// Returns a matrix with selection instances and input variables.
/// The number of rows is the number of selection instances.
/// The number of columns is the number of input variables.

Matrix<double> DataSet::get_selection_inputs() const
{
   const Vector<size_t> selection_indices = instances.get_selection_indices();

   const Vector<size_t> inputs_indices = variables.get_inputs_indices();

   return(data.get_submatrix(selection_indices, inputs_indices));
}


Eigen::MatrixXd DataSet::get_selection_inputs_eigen() const
{
    Matrix<double> data = get_selection_inputs();
    const size_t rows_number = data.get_rows_number();
    const size_t columns_number = data.get_columns_number();

    const Eigen::Map<Eigen::MatrixXd> data_eigen((double*)data.data(), rows_number, columns_number);

    return(data_eigen);
}


/// Returns a matrix with selection instances and target variables.
/// The number of rows is the number of selection instances.
/// The number of columns is the number of target variables.

Matrix<double> DataSet::get_selection_targets() const
{
   const Vector<size_t> selection_indices = instances.get_selection_indices();

   const Vector<size_t> targets_indices = variables.get_targets_indices();

   return(data.get_submatrix(selection_indices, targets_indices));
}


Eigen::MatrixXd DataSet::get_selection_targets_eigen() const
{
    Matrix<double> data = get_selection_targets();
    const size_t rows_number = data.get_rows_number();
    const size_t columns_number = data.get_columns_number();

    const Eigen::Map<Eigen::MatrixXd> data_eigen((double*)data.data(), rows_number, columns_number);

    return(data_eigen);
}


/// Returns a matrix with testing instances and input variables.
/// The number of rows is the number of testing instances.
/// The number of columns is the number of input variables.

Matrix<double> DataSet::get_testing_inputs() const
{
   const Vector<size_t> inputs_indices = variables.get_inputs_indices();

   const Vector<size_t> testing_indices = instances.get_testing_indices();

   return(data.get_submatrix(testing_indices, inputs_indices));
}


Eigen::MatrixXd DataSet::get_testing_inputs_eigen() const
{
    Matrix<double> data = get_testing_inputs();
    const size_t rows_number = data.get_rows_number();
    const size_t columns_number = data.get_columns_number();

    const Eigen::Map<Eigen::MatrixXd> data_eigen((double*)data.data(), rows_number, columns_number);

    return(data_eigen);
}


/// Returns a matrix with testing instances and target variables.
/// The number of rows is the number of testing instances.
/// The number of columns is the number of target variables.

Matrix<double> DataSet::get_testing_targets() const
{
   const Vector<size_t> targets_indices = variables.get_targets_indices();

   const Vector<size_t> testing_indices = instances.get_testing_indices();

   return(data.get_submatrix(testing_indices, targets_indices));
}


Eigen::MatrixXd DataSet::get_testing_targets_eigen() const
{
    Matrix<double> data = get_testing_targets();
    const size_t rows_number = data.get_rows_number();
    const size_t columns_number = data.get_columns_number();

    const Eigen::Map<Eigen::MatrixXd> data_eigen((double*)data.data(), rows_number, columns_number);

    return(data_eigen);
}


/// Returns a column with the testing instances and the time variable.
/// The number of rows is the number of testing instances.

Vector<double> DataSet::get_testing_time() const
{
   const Vector<size_t> testing_indices = instances.get_testing_indices();

   Matrix<double> matrix = data.get_submatrix_rows(testing_indices);

   const size_t time_index = variables.get_time_index();

   return(matrix.get_column(time_index));
}


/// Returns a Dataset containing only with the training instances of the original

DataSet DataSet::get_training_data_set() const
{
    const Vector<size_t> training_indices = instances.get_training_indices();


    const Matrix<double> training_data = data.get_submatrix_rows(instances.get_training_indices());

    DataSet training_data_set(training_data);


//    const Instances* training_instances = training_data_set.get_instances_pointer();

    training_data_set.instances.set_training();

//    const Variables* training_variables = training_data_set.get_variables_pointer();

//    &training_variables = variables;

//    training_variables->set_target_indices(variables.get_targets_indices());
    training_data_set.variables.set_target_indices(variables.get_targets_indices());

    return(training_data_set);

}


/// Returns a Dataset containing only with the testing insatances of the original
/// This DataSet does not contain the target data.

DataSet DataSet::get_testing_data_set() const
{
    const Vector<size_t> testing_indices = instances.get_testing_indices();

    Matrix<double> testing_data = data.get_submatrix_rows(instances.get_testing_indices()).delete_columns(variables.get_targets_indices());

    DataSet testing_data_set(testing_data);

    testing_data_set.instances.set_testing();

    testing_data_set.variables.set_input();

    return(testing_data_set);

}


/// Returns a Dataset containing only with the selection insatances of the original

DataSet DataSet::get_selection_data_set() const
{
    const Vector<size_t> selection_indices = instances.get_selection_indices();


    Matrix<double> selection_data = data.get_submatrix_rows(instances.get_selection_indices());

    selection_data.delete_columns(variables.get_targets_indices());

    DataSet selection_data_set(selection_data);


    selection_data_set.instances.set_selection();

    selection_data_set.variables.set_target_indices(variables.get_targets_indices());


    return(selection_data_set);
}


/// Returns the inputs and target values of a single instance in the data set.
/// @param i Index of the instance.

Vector<double> DataSet::get_instance(const size_t& i) const
{
   // Control sentence(if debug)

   #ifdef __OPENNN_DEBUG__

   const size_t instances_number = instances.get_instances_number();

   if(i >= instances_number)
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: DataSet class.\n"
             << "Vector<double> get_instance(const size_t&) const method.\n"
             << "Index of instance must be less than number of instances.\n";

      throw logic_error(buffer.str());
   }

   #endif

   // Get instance

   return(data.get_row(i));
}


/// Returns the inputs and target values of a single instance in the data set.
/// @param instance_index Index of the instance.
/// @param variables_indices Indices of the variables.

Vector<double> DataSet::get_instance(const size_t& instance_index, const Vector<size_t>& variables_indices) const
{
   // Control sentence(if debug)

   #ifdef __OPENNN_DEBUG__

   const size_t instances_number = instances.get_instances_number();

   if(instance_index >= instances_number)
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: DataSet class.\n"
             << "Vector<double> get_instance(const size_t&, const Vector<size_t>&) const method.\n"
             << "Index of instance must be less than number of instances.\n";

      throw logic_error(buffer.str());
   }

   #endif

   // Get instance

   return(data.get_row(instance_index, variables_indices));
}


/// Returns all the instances of a single variable in the data set.
/// @param i Index of the variable.

Vector<double> DataSet::get_variable(const size_t& i) const
{
   // Control sentence(if debug)

   #ifdef __OPENNN_DEBUG__

   const size_t variables_number = variables.get_variables_number();

   if(i >= variables_number)
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: DataSet class.\n"
             << "Vector<double> get_variable(const size_t&) const method.\n"
             << "Index of variable must be less than number of instances.\n";

      throw logic_error(buffer.str());
   }

   #endif

   // Get variable

   return(data.get_column(i));
}


/// Returns all the instances of a single variable in the data set.
/// @param variable_name Name of the variable.

Vector<double> DataSet::get_variable(const string& variable_name) const
{
    const Vector<string> variable_names = variables.get_names();

    const Vector<size_t> variable_index = variable_names.calculate_equal_to_indices(variable_name);

#ifdef __OPENNN_DEBUG__

    const size_t variables_size = variable_index.size();

    if(variables_size == 0)
    {
       ostringstream buffer;

       buffer << "OpenNN Exception: DataSet class.\n"
              << "Vector<double> get_variable(const string&) const method.\n"
              << "Variable: " << variable_name << " does not exist.\n";

       throw logic_error(buffer.str());
    }

    if(variables_size > 1)
    {
       ostringstream buffer;

       buffer << "OpenNN Exception: DataSet class.\n"
              << "Vector<double> get_variable(const string&) const method.\n"
              << "Variable: " << variable_name << " appears more than once in the data set.\n";

       throw logic_error(buffer.str());
    }

#endif

    return(data.get_column(variable_index[0]));
}


/// Returns a given set of instances of a single variable in the data set.
/// @param variable_index Index of the variable.
/// @param instances_indices Indices of the instances.

Vector<double> DataSet::get_variable(const size_t& variable_index, const Vector<size_t>& instances_indices) const
{
   // Control sentence(if debug)

   #ifdef __OPENNN_DEBUG__

   const size_t variables_number = variables.get_variables_number();

   if(variable_index >= variables_number)
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: DataSet class.\n"
             << "Vector<double> get_variable(const size_t&, const Vector<size_t>&) const method.\n"
             << "Index of variable must be less than number of instances.\n";

      throw logic_error(buffer.str());
   }

   #endif

   // Get variable

   return(data.get_column(variable_index, instances_indices));
}


/// Returns a given set of instances of a single variable in the data set.
/// @param variable_name Name of the variable.
/// @param instances_indices Indices of the instances.

Vector<double> DataSet::get_variable(const string& variable_name, const Vector<size_t>& instances_indices) const
{
    const Vector<string> variable_names = variables.get_names();

    const Vector<size_t> variable_index = variable_names.calculate_equal_to_indices(variable_name);

#ifdef __OPENNN_DEBUG__

    const size_t variables_size = variable_index.size();

    if(variables_size == 0)
    {
       ostringstream buffer;

       buffer << "OpenNN Exception: DataSet class.\n"
              << "Vector<double> get_variable(const string&) const method.\n"
              << "Variable: " << variable_name << " does not exist.\n";

       throw logic_error(buffer.str());
    }

    if(variables_size > 1)
    {
       ostringstream buffer;

       buffer << "OpenNN Exception: DataSet class.\n"
              << "Vector<double> get_variable(const string&, const Vector<size_t>&) const method.\n"
              << "Variable: " << variable_name << " appears more than once in the data set.\n";

       throw logic_error(buffer.str());
    }

#endif

    return(data.get_column(variable_index[0], instances_indices));
}


/// Returns a given set of instances a given set of variablesin the data set.
/// @param variables_indices Indices of the variables.
/// @param instances_indices Indices of the instances.

Matrix<double> DataSet::get_variables(const Vector<size_t>& variables_indices, const Vector<size_t>& instances_indices) const
{
   // Control sentence(if debug)

   #ifdef __OPENNN_DEBUG__

   const size_t variables_number = variables.get_variables_number();

   if(variables_indices.calculate_maximum() >= variables_number)
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: DataSet class.\n"
             << "Matrix<double> get_variablesconst Vector<size_t>&, const Vector<size_t>&) const method.\n"
             << "One or more indices are greater than the number of variables in the data set.\n";

      throw logic_error(buffer.str());
   }

   if(variables_indices.size() == 0)
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: DataSet class.\n"
             << "Matrix<double> get_variablesconst Vector<size_t>&, const Vector<size_t>&) const method.\n"
             << "Variables indices vector size must be greater than 0.\n";

      throw logic_error(buffer.str());
   }

   if(instances_indices.size() == 0)
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: DataSet class.\n"
             << "Matrix<double> get_variables(const Vector<size_t>&, const Vector<size_t>&) const method.\n"
             << "Indices indices vector size must be greater than 0.\n";

      throw logic_error(buffer.str());
   }

   #endif

   // Get variables

   return(data.get_submatrix(instances_indices, variables_indices));
}


/// Sets zero instances and zero variables in the data set.

void DataSet::set()
{
   data_file_name = "";

   first_cell = "";
   last_cell = "";

   data.set();

   variables.set();
   instances.set();

   missing_values.set();

   display = true;

   file_type = DAT;
}


/// Sets all variables from a data matrix.
/// @param new_data Data matrix.

void DataSet::set(const Matrix<double>& new_data)
{
   data_file_name = "";

   const size_t variables_number = new_data.get_columns_number();
   const size_t instances_number = new_data.get_rows_number();

   set(instances_number, variables_number);

   data = new_data;

   if(!data.get_header().empty()) variables.set_names(data.get_header());

   display = true;

   file_type = DAT;
}


void DataSet::set(const Eigen::MatrixXd& new_data)
{
   data_file_name = "";

   const size_t variables_number = static_cast<size_t>(new_data.cols());
   const size_t instances_number = static_cast<size_t>(new_data.rows());

   set(instances_number, variables_number);

   data.set(instances_number, variables_number);

   Eigen::Map<Eigen::MatrixXd> auxiliar_eigen(data.data(), static_cast<unsigned>(instances_number), static_cast<unsigned>(variables_number));

   auxiliar_eigen = new_data;

   if(!data.get_header().empty()) variables.set_names(data.get_header());

   display = true;

   file_type = DAT;
}


/// Sets new numbers of instances and variables in the inputs targets data set.
/// All the instances are set for training.
/// All the variables are set as inputs.
/// @param new_instances_number Number of instances.
/// @param new_variables_number Number of variables.

void DataSet::set(const size_t& new_instances_number, const size_t& new_variables_number)
{
    // Control sentence(if debug)

    #ifdef __OPENNN_DEBUG__

    if(new_instances_number == 0)
    {
       ostringstream buffer;

       buffer << "OpenNN Exception: DataSet class.\n"
              << "void set(const size_t&, const size_t&) method.\n"
              << "Number of instances must be greater than zero.\n";

       throw logic_error(buffer.str());
    }

    if(new_variables_number == 0)
    {
       ostringstream buffer;

       buffer << "OpenNN Exception: DataSet class.\n"
              << "void set(const size_t&, const size_t&) method.\n"
              << "Number of variables must be greater than zero.\n";

       throw logic_error(buffer.str());
    }

    #endif

   data.set(new_instances_number, new_variables_number);

   instances.set(new_instances_number);

   variables.set(new_variables_number);

   missing_values.set(new_instances_number, new_variables_number);

   display = true;

   file_type = DAT;
}


/// Sets new numbers of instances and inputs and target variables in the data set.
/// The variables in the data set are the number of inputs plus the number of targets.
/// @param new_instances_number Number of instances.
/// @param new_inputs_number Number of input variables.
/// @param new_targets_number Number of target variables.

void DataSet::set(const size_t& new_instances_number, const size_t& new_inputs_number, const size_t& new_targets_number)
{
   data_file_name = "";

   const size_t new_variables_number = new_inputs_number + new_targets_number;

   data.set(new_instances_number, new_variables_number);

   variables.set(new_inputs_number, new_targets_number);

   instances.set(new_instances_number);

   missing_values.set(new_instances_number, new_variables_number);

   display = true;

   file_type = DAT;
}


/// Sets the members of this data set object with those from another data set object.
/// @param other_data_set Data set object to be copied.

void DataSet::set(const DataSet& other_data_set)
{
   data_file_name = other_data_set.data_file_name;

   header_line = other_data_set.header_line;

   separator = other_data_set.separator;

   missing_values_label = other_data_set.missing_values_label;

   data = other_data_set.data;

   variables = other_data_set.variables;

   instances = other_data_set.instances;

   missing_values = other_data_set.missing_values;

   display = other_data_set.display;

   file_type = other_data_set.file_type;
}


/// Sets the data set members from a XML document.
/// @param data_set_document TinyXML document containing the member data.

void DataSet::set(const tinyxml2::XMLDocument& data_set_document)
{
    set_default();

   from_XML(data_set_document);
}


/// Sets the data set members by loading them from a XML file.
/// @param file_name Data set XML file_name.

void DataSet::set(const string& file_name)
{
   load(file_name);
}


/// Sets a new display value.
/// If it is set to true messages from this class are to be displayed on the screen;
/// if it is set to false messages from this class are not to be displayed on the screen.
/// @param new_display Display value.

void DataSet::set_display(const bool& new_display)
{
   display = new_display;
}


/// Sets the default member values:
/// <ul>
/// <li> Display: True.
/// </ul>

void DataSet::set_default()
{
    header_line = false;

    separator = Space;

    missing_values_label = "?";

    lags_number = 0;

    steps_ahead = 0;

    association = false;

    angular_units = Degrees;

    display = true;

    file_type = DAT;

    sheet_number = 1;
}


/// Send the DataSet to all the processors using MPI.
/// @param data_set Original DataSet object, initialized by processor 0.

void DataSet::set_MPI(const DataSet* data_set)
{
#ifdef __OPENNN_MPI__

    int size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int training_instances_number;
    int selection_instances_number;

    int inputs_number;
    int outputs_number;

    Vector<int> inputs_indices;
    Vector<int> targets_indices;

    Vector<size_t> training_indices;
    Vector<size_t> selection_indices;

    if(rank == 0)
    {
        // Variables to send initialization

        const Instances& instances = data_set->get_instances();

        training_instances_number = (int)instances.get_training_instances_number();
        selection_instances_number = (int)instances.get_selection_instances_number();

        training_indices = instances.get_training_indices();
        selection_indices = instances.get_selection_indices();

        const Variables& variables = data_set->get_variables();

        inputs_indices = variables.get_inputs_indices_int();
        targets_indices = variables.get_targets_indices_int();

        inputs_number = (int)variables.get_inputs_number();
        outputs_number = (int)variables.get_targets_number();
    }

    // Send variables

    MPI_Barrier(MPI_COMM_WORLD);

    if(rank > 0)
    {
        MPI_Request req[4];

        MPI_Irecv(&inputs_number, 1, MPI_INT, rank-1, 1, MPI_COMM_WORLD, &req[0]);
        MPI_Irecv(&outputs_number, 1, MPI_INT, rank-1, 2, MPI_COMM_WORLD, &req[1]);

        MPI_Waitall(2, req, MPI_STATUS_IGNORE);

        inputs_indices.set(inputs_number);
        targets_indices.set(outputs_number);

        MPI_Irecv(&training_instances_number, 1, MPI_INT, rank-1, 5, MPI_COMM_WORLD, &req[0]);
        MPI_Irecv(&selection_instances_number, 1, MPI_INT, rank-1, 6, MPI_COMM_WORLD, &req[1]);

        MPI_Irecv(inputs_indices.data(),(int)inputs_number, MPI_INT, rank-1, 7, MPI_COMM_WORLD, &req[2]);
        MPI_Irecv(targets_indices.data(),(int)outputs_number, MPI_INT, rank-1, 8, MPI_COMM_WORLD, &req[3]);

        MPI_Waitall(4, req, MPI_STATUS_IGNORE);
    }

    if(rank < size-1)
    {
        MPI_Request req[6];

        MPI_Isend(&inputs_number, 1, MPI_INT, rank+1, 1, MPI_COMM_WORLD, &req[0]);
        MPI_Isend(&outputs_number, 1, MPI_INT, rank+1, 2, MPI_COMM_WORLD, &req[1]);

        MPI_Isend(&training_instances_number, 1, MPI_INT, rank+1, 5, MPI_COMM_WORLD, &req[2]);
        MPI_Isend(&selection_instances_number, 1, MPI_INT, rank+1, 6, MPI_COMM_WORLD, &req[3]);

        MPI_Isend(inputs_indices.data(),(int)inputs_number, MPI_INT, rank+1, 7, MPI_COMM_WORLD, &req[4]);
        MPI_Isend(targets_indices.data(),(int)outputs_number, MPI_INT, rank+1, 8, MPI_COMM_WORLD, &req[5]);

        MPI_Waitall(6, req, MPI_STATUS_IGNORE);
    }

    size = min(size,static_cast<int>(training_instances_number));

    // Get the group of processes in MPI_COMM_WORLD
    MPI_Group world_group;
    MPI_Comm_group(MPI_COMM_WORLD, &world_group);

    int* ranks = (int*)malloc(size*sizeof(int));

    for(int i = 0; i < size; i++)
    {
        ranks[i] = i;
    }

    // Construct a group containing all of the prime ranks in world_group
    MPI_Group error_group;
    MPI_Group_incl(world_group, size, ranks, &error_group);

    // Create a new communicator based on the group
    MPI_Comm error_comm;
    MPI_Comm_create(MPI_COMM_WORLD, error_group, &error_comm);

    int i = 0;

    Vector<size_t> training_instances_per_processor(size);
    Vector<size_t> selection_instances_per_processor(size);

    for(i = 0; i < size; i++)
    {
        training_instances_per_processor[i] = (size_t)(training_instances_number/size);
        selection_instances_per_processor[i] = (size_t)(selection_instances_number/size);

        if(i < training_instances_number%size)
        {
            training_instances_per_processor[i]++;
        }
        if(i < selection_instances_number%size)
        {
            selection_instances_per_processor[i]++;
        }
    }

    Matrix<double> processor_data;

    // Append training instances

    if(rank != 0 && rank < size)
    {
        processor_data.set(inputs_number+outputs_number, training_instances_per_processor[rank]+selection_instances_per_processor[rank]);

        for(int j = 0; j < training_instances_per_processor[rank]; j++)
        {
            MPI_Recv(processor_data.data()+ (j*(inputs_number+outputs_number)),(int)(inputs_number+outputs_number), MPI_DOUBLE, 0, j, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

        for(int j = 0; j < selection_instances_per_processor[rank]; j++)
        {
            MPI_Recv(processor_data.data()+ ((j+training_instances_per_processor[rank])*(inputs_number+outputs_number)),
                     (int)(inputs_number+outputs_number), MPI_DOUBLE, 0, j+ (int)training_instances_per_processor[rank], MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

        processor_data = processor_data.calculate_transpose();
    }
    else if(rank == 0)
    {
        size_t training_instances_sent = training_instances_per_processor[rank];
        size_t selection_instances_sent = selection_instances_per_processor[rank];

        for(i = 1; i < size; i++)
        {
            for(int j = 0; j < training_instances_per_processor[i]; j++)
            {
                const Vector<double> instance_to_send = data_set->get_instance(training_indices[training_instances_sent]);

                MPI_Send(instance_to_send.data(),(int)(inputs_number+outputs_number), MPI_DOUBLE, i, j, MPI_COMM_WORLD);

                training_instances_sent++;
            }
            for(int j = 0; j < selection_instances_per_processor[i]; j++)
            {
                const Vector<double> instance_to_send = data_set->get_instance(selection_indices[selection_instances_sent]);

                MPI_Send(instance_to_send.data(),(int)(inputs_number+outputs_number), MPI_DOUBLE, i, j+ (int)training_instances_per_processor[i], MPI_COMM_WORLD);

                selection_instances_sent++;
            }
        }

        processor_data.set(training_instances_per_processor[rank]+selection_instances_per_processor[rank],inputs_number+outputs_number);

        for(i = 0; i < training_instances_per_processor[rank]; i++)
        {
            const Vector<double> instance_to_append = data_set->get_instance(training_indices[i]);

            processor_data.set_row(i, instance_to_append);
        }
        for(i = 0; i < selection_instances_per_processor[rank]; i++)
        {
            const Vector<double> instance_to_append = data_set->get_instance(selection_indices[i]);

            processor_data.set_row(i+training_instances_per_processor[rank], instance_to_append);
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);

    if(rank < size)
    {
        set_data(processor_data);
        get_variables_pointer()->set_unuse();
        get_variables_pointer()->set_input_indices(inputs_indices);
        get_variables_pointer()->set_target_indices(targets_indices);
        get_instances_pointer()->split_sequential_indices(static_cast<double>(training_instances_per_processor[rank]),static_cast<double>(selection_instances_per_processor[rank]), 0.0);
    }

    free(ranks);

#else

    set(*data_set);

#endif
}


/// Sets a new data matrix.
/// The number of rows must be equal to the number of instances.
/// The number of columns must be equal to the number of variables.
/// Indices of all training, selection and testing instances and inputs and target variables do not change.
/// @param new_data Data matrix.

void DataSet::set_data(const Matrix<double>& new_data)
{
   // Control sentence(if debug)
/*
   #ifdef __OPENNN_DEBUG__

   const size_t rows_number = new_data.get_rows_number();
   const size_t instances_number = instances.get_instances_number();

   if(rows_number != instances_number)
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: DataSet class.\n"
             << "void set_data(const Matrix<double>&) method.\n"
             << "Number of rows(" << rows_number << ") must be equal to number of instances(" << instances_number << ").\n";

      throw logic_error(buffer.str());
   }

   const size_t columns_number = new_data.get_columns_number();
   const size_t variables_number = variables.get_variables_number();

   if(columns_number != variables_number)
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: DataSet class.\n"
             << "void set_data(const Matrix<double>&) method.\n"
             << "Number of columns(" << columns_number << ") must be equal to number of variables(" << variables_number << ").\n";

      throw logic_error(buffer.str());
   }

   #endif
*/
   // Set data

   data = new_data;

   instances.set_instances_number(data.get_rows_number());
   variables.set_variables_number(data.get_columns_number());
}


/// Sets the name of the data file.
/// It also loads the data from that file.
/// Moreover, it sets the variables and instances objects.
/// @param new_data_file_name Name of the file containing the data.

void DataSet::set_data_file_name(const string& new_data_file_name)
{
   data_file_name = new_data_file_name;
}


/// Sets the file type.

void DataSet::set_file_type(const DataSet::FileType& new_file_type)
{
    file_type = new_file_type;
}


/// Sets the file type from a string

void DataSet::set_file_type(const string& new_file_type)
{
    if(new_file_type == "txt")
    {
        file_type = TXT;
    }
    else if(new_file_type == "dat")
    {
        file_type = DAT;
    }
    else if(new_file_type == "csv")
    {
        file_type = CSV;
    }
    else if(new_file_type == "ods")
    {
        file_type = ODS;
    }
    else if(new_file_type == "xlsx")
    {
        file_type = XLSX;
    }
    else if(new_file_type == "arff")
    {
        file_type = ARFF;
    }
    else if(new_file_type == "json")
    {
        file_type = JSON;
    }
    else
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: DataSet class.\n"
               << "void set_file_type(const string&) method.\n"
               << "Unknown file type.";

        throw logic_error(buffer.str());
    }
}


/// Sets if the data file contains a header with the names of the variables.

void DataSet::set_header_line(const bool& new_header_line)
{
    header_line = new_header_line;
}


/// Sets if the data file contains rows label.

void DataSet::set_rows_label(const bool& new_rows_label)
{
    rows_label = new_rows_label;
}


/// Sets a new separator.
/// @param new_separator Separator value.

void DataSet::set_separator(const Separator& new_separator)
{
    separator = new_separator;
}


/// Sets a new separator from a string.
/// @param new_separator String with the separator value.

void DataSet::set_separator(const string& new_separator)
{
    if(new_separator == "Space")
    {
        separator = Space;
    }
    else if(new_separator == "Tab")
    {
        separator = Tab;
    }
    else if(new_separator == "Comma")
    {
        separator = Comma;
    }
    else if(new_separator == "Semicolon")
    {
        separator = Semicolon;
    }
    else
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: DataSet class.\n"
               << "void set_separator(const string&) method.\n"
               << "Unknown separator: " << new_separator << ".\n";

        throw logic_error(buffer.str());
    }
}


/// Sets a new label for the missing values.
/// @param new_missing_values_label Label for the missing values.

void DataSet::set_missing_values_label(const string& new_missing_values_label)
{
    // Control sentence(if debug)

    #ifdef __OPENNN_DEBUG__

    if(get_trimmed(new_missing_values_label).empty())
    {
       ostringstream buffer;

       buffer << "OpenNN Exception: DataSet class.\n"
              << "void set_missing_values_label(const string&) method.\n"
              << "Missing values label cannot be empty.\n";

       throw logic_error(buffer.str());
    }

    #endif


    missing_values_label = new_missing_values_label;
}


/// Sets a new grouping factor value.
/// @param new_grouping_factor Value for the grouping factor.

void DataSet::set_grouping_factor(const double& new_grouping_factor)
{
    grouping_factor = new_grouping_factor;
}


/// Sets a new number of lags to be defined for a time series prediction application.
/// When loading the data file, the time series data will be modified according to this number.
/// @param new_lags_number Number of lags(x-1, ..., x-l) to be used.

void DataSet::set_lags_number(const size_t& new_lags_number)
{
    lags_number = new_lags_number;
}


/// Sets a new number of steps ahead to be defined for a time series prediction application.
/// When loading the data file, the time series data will be modified according to this number.
/// @param new_steps_ahead_number Number of steps ahead to be used.

void DataSet::set_steps_ahead_number(const size_t& new_steps_ahead_number)
{
    steps_ahead = new_steps_ahead_number;
}


void DataSet::set_time_index(const size_t& new_time_index)
{
    time_index = new_time_index;
}


/// Sets a new autoasociation flag.
/// If the new value is true, the data will be processed for association when loading.
/// That is, the data file will contain the input data. The target data will be created as being equal to the input data.
/// If the association value is set to false, the data from the file will not be processed.
/// @param new_autoassociation Association value.

void DataSet::set_autoassociation(const bool& new_autoassociation)
{
    association = new_autoassociation;
}


/// Sets a new project type.
/// @param new_learning_task New project type.

void DataSet::set_learning_task(const DataSet::ProjectType& new_learning_task)
{
    learning_task = new_learning_task;
}


/// Sets a new project type from a string.
/// @param new_learning_task New project type.

void DataSet::set_learning_task(const string& new_learning_task)
{
    if(new_learning_task == "Approximation")
    {
        learning_task = Approximation;
    }
    else if(new_learning_task == "Classification")
    {
        learning_task = Classification;
    }
    else if(new_learning_task == "Forecasting")
    {
        learning_task = Forecasting;
    }
    else if(new_learning_task == "Association")
    {
        learning_task = Association;
    }
    else
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: DataSet class.\n"
               << "void set_learning_task(const string&) method.\n"
               << "Not known project type.\n";

        throw logic_error(buffer.str());
    }
}


/// Sets the indices of those variables which represent angles.
/// @param new_angular_variables Indices of angular variables.

void DataSet::set_angular_variables(const Vector<size_t>& new_angular_variables)
{
    angular_variables = new_angular_variables;
}


/// Sets the units of the angular variables(Radians or Degrees).

void DataSet::set_angular_units(AngularUnits& new_angular_units)
{
    angular_units = new_angular_units;
}


/// Sets a new number of instances in the data set.
/// All instances are also set for training.
/// The indices of the inputs and target variables do not change.
/// @param new_instances_number Number of instances.

void DataSet::set_instances_number(const size_t& new_instances_number)
{
   const size_t variables_number = variables.get_variables_number();

   data.set(new_instances_number, variables_number);

   instances.set(new_instances_number);
}


/// Sets a new number of input variables in the data set.
/// The indices of the training, selection and testing instances do not change.
/// All variables are set as inputs.
/// @param new_variables_number Number of variables.

void DataSet::set_variables_number(const size_t& new_variables_number)
{
   const size_t instances_number = instances.get_instances_number();

   data.set(instances_number, new_variables_number);

   variables.set(new_variables_number);
}


/// Sets new inputs and target values of a single instance in the data set.
/// @param instance_index Index of the instance.
/// @param instance New inputs and target values of the instance.

void DataSet::set_instance(const size_t& instance_index, const Vector<double>& instance)
{
   // Control sentence(if debug)

   #ifdef __OPENNN_DEBUG__

   const size_t instances_number = instances.get_instances_number();

   if(instance_index >= instances_number)
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: DataSet class.\n"
             << "void set_instance(const size_t&, const Vector<double>&) method.\n"
             << "Index of instance must be less than number of instances.\n";

      throw logic_error(buffer.str());
   }

   const size_t size = instance.size();
   const size_t variables_number = variables.get_variables_number();

   if(size != variables_number)
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: DataSet class.\n"
             << "void set_instance(const size_t&, const Vector<double>&) method.\n"
             << "Size(" << size << ") must be equal to number of variables(" << variables_number << ").\n";

      throw logic_error(buffer.str());
   }

   #endif

   // Set instance

   data.set_row(instance_index, instance);
}


/// Adds a new instance to the data matrix from a vector of real numbers.
/// The size of that vector must be equal to the number of variables.
/// Note that resizing is here necessary and therefore computationally expensive.
/// All instances are also set for training.
/// @param instance Input and target values of the instance to be added.

void DataSet::add_instance(const Vector<double>& instance)
{
   // Control sentence(if debug)

   #ifdef __OPENNN_DEBUG__

   const size_t size = instance.size();
   const size_t variables_number = variables.get_variables_number();

   if(size != variables_number)
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: DataSet class.\n"
             << "void add_instance(const Vector<double>&) method.\n"
             << "Size of instance must be equal to number of variables.\n";

      throw logic_error(buffer.str());
   }

   #endif

   const size_t instances_number = instances.get_instances_number();

   data.append_row(instance);

   instances.set(instances_number+1);
}


/// Substracts the inputs-targets instance with a given index from the data set.
/// All instances are also set for training.
/// Note that resizing is here necessary and therefore computationally expensive.
/// @param instance_index Index of instance to be removed.

void DataSet::remove_instance(const size_t& instance_index)
{
    const size_t instances_number = instances.get_instances_number();

   // Control sentence(if debug)

   #ifdef __OPENNN_DEBUG__

   if(instance_index >= instances_number)
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: DataSet class.\n"
             << "void remove_instance(size_t) method.\n"
             << "Index of instance must be less than number of instances.\n";

      throw logic_error(buffer.str());
   }

   #endif

   data.delete_row(instance_index);

   instances.set_instances_number(instances_number-1);

}


/// Appends a variable with given values to the data matrix.
/// @param variable Vector of values. The size must be equal to the number of instances.

void DataSet::append_variable(const Vector<double>& variable, const string& variable_name)
{
   // Control sentence(if debug)

   #ifdef __OPENNN_DEBUG__

   const size_t size = variable.size();
   const size_t instances_number = instances.get_instances_number();

   if(size != instances_number)
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: DataSet class.\n"
             << "void append_variable(const Vector<double>&) method.\n"
             << "Size of variable must be equal to number of instances.\n";

      throw logic_error(buffer.str());
   }

   #endif

   const size_t variables_number = variables.get_variables_number();

   data = data.append_column(variable, "");

   Matrix<double> new_data(data);

   const size_t new_variables_number = variables_number + 1;

   Vector<Variables::Item> items = variables.get_items();

   set_variables_number(new_variables_number);

   Variables::Item new_item;
   new_item.name = variable_name;

   items.push_back(new_item);

   variables.set_items(items);

   set_data(new_data);
}


/// Removes a variable with given index from the data matrix.
/// @param variable_index Index of variable to be subtracted.

void DataSet::remove_variable(const size_t& variable_index)
{
   const size_t variables_number = variables.get_variables_number();

   // Control sentence(if debug)

   #ifdef __OPENNN_DEBUG__

   if(variable_index >= variables_number)
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: DataSet class.\n"
             << "void remove_variable(size_t) method.\n"
             << "Index of variable must be less than number of variables.\n";

      throw logic_error(buffer.str());
   }

   #endif

   data = data.delete_column(variable_index);

   const size_t new_variables_number = variables_number - 1;

   Vector<Variables::Item> items = variables.get_items();

   variables.set(new_variables_number);

   items = items.delete_index(variable_index);

   variables.set_items(items);
}


/// Removes a variable with given name from the data matrix.
/// @param variable_name Name of variable to be subtracted.

void DataSet::remove_variable(const string& variable_name)
{
    const Vector<string> variable_names = variables.get_names();

    const Vector<size_t> variable_index = variable_names.calculate_equal_to_indices(variable_name);

#ifdef __OPENNN_DEBUG__

    const size_t variables_size = variable_index.size();

    if(variables_size == 0)
    {
       ostringstream buffer;

       buffer << "OpenNN Exception: DataSet class.\n"
              << "Vector<double> get_variable(const string&) const method.\n"
              << "Variable: " << variable_name << " does not exist.\n";

       throw logic_error(buffer.str());
    }

    if(variables_size > 1)
    {
       ostringstream buffer;

       buffer << "OpenNN Exception: DataSet class.\n"
              << "Vector<double> get_variable(const string&) const method.\n"
              << "Variable: " << variable_name << " appears more than once in the data set.\n";

       throw logic_error(buffer.str());
    }

#endif

    remove_variable(variable_index[0]);
}


/// Removes the input of target indices of that variables with zero standard deviation.
/// It might change the size of the vectors containing the inputs and targets indices.

Vector<string> DataSet::unuse_constant_variables()
{
   const size_t variables_number = variables.get_variables_number();

   // Control sentence(if debug)

   #ifdef __OPENNN_DEBUG__

   if(variables_number == 0)
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: DataSet class.\n"
             << "Vector<string> unuse_constant_variables() method.\n"
             << "Number of variables is zero.\n";

      throw logic_error(buffer.str());
   }

   #endif

   Vector<size_t> constant_variables;

   for(size_t i = 0; i < variables_number; i++)
   {
      if(variables.get_use(i) ==  Variables::Input && data.is_column_constant(i))
      {
         variables.set_use(i, Variables::Unused);

         constant_variables.push_back(i);
      }
   }

   return variables.get_names().get_subvector(constant_variables);
}


/// Removes the training, selection and testing indices of that instances which are repeated in the data matrix.
/// It might change the size of the vectors containing the training, selection and testing indices.

Vector<size_t> DataSet::unuse_repeated_instances()
{
    const size_t instances_number = instances.get_instances_number();

    // Control sentence(if debug)

    #ifdef __OPENNN_DEBUG__

    if(instances_number == 0)
    {
       ostringstream buffer;

       buffer << "OpenNN Exception: DataSet class.\n"
              << "Vector<size_t> unuse_repeated_indices() method.\n"
              << "Number of instances is zero.\n";

       throw logic_error(buffer.str());
    }

    #endif

    Vector<size_t> repeated_instances;

    Vector<double> instance_i;
    Vector<double> instance_j;

    int i = 0;

    #pragma omp parallel for private(i, instance_i, instance_j) schedule(dynamic)

    for(i = 0; i < static_cast<int>(instances_number); i++)
    {
       instance_i = get_instance(static_cast<size_t>(i));

       for(size_t j = static_cast<size_t>(i+1); j < instances_number; j++)
       {
          instance_j = get_instance(j);

          if(instances.get_use(j) != Instances::Unused
          && instance_j == instance_i)
          {
              instances.set_use(j, Instances::Unused);
              repeated_instances.push_back(j);
          }
       }
    }

    return(repeated_instances);
}


/// Unuses those binary inputs whose positives does not correspond to any positive in the target variables.

Vector<size_t> DataSet::unuse_non_significant_inputs()
{
    const Vector<size_t> inputs_indices = get_variables_pointer()->get_inputs_indices();
    const size_t inputs_number = inputs_indices.size();

    const size_t target_index = get_variables_pointer()->get_targets_indices()[0];

    const size_t instances_number = get_instances_pointer()->get_used_instances_number();

    Vector<size_t> non_significant_variables;

    if(!is_binary_classification())
    {
        return non_significant_variables;
    }

    size_t positives = 0;

    size_t current_input_index;

    for(size_t i = 0; i < inputs_number; i++)
    {
        positives = 0;

        current_input_index = inputs_indices[i];

        if(!is_binary_variable(current_input_index))
        {
            continue;
        }

        for(size_t j = 0; j < instances_number; j++)
        {
            if(data(j, current_input_index) == 1.0 && data(j, target_index) == 1.0)
            {
                positives++;
            }
        }

        if(positives == 0)
        {
            variables.set_use(current_input_index, Variables::Unused);
            non_significant_variables.push_back(current_input_index);
        }
    }

    return non_significant_variables;
}


Vector<string> DataSet::unuse_variables_missing_values(const double& missing_ratio)
{
    const size_t variables_number = variables.get_variables_number();

    const size_t instances_number = instances.get_instances_number();

    const Vector<size_t> variables_missing_values = missing_values.count_variables_missing_indices();

    const Vector<double> variables_missing_ratios = variables_missing_values.to_double_vector()/(static_cast<double>(instances_number)-1.0);

    Vector<string> unused_variables;

    for(size_t i = 0; i < variables_number; i++)
    {
        if(variables.is_used(i) && variables_missing_ratios[i] >= missing_ratio)
        {
            variables.set_use(i, Variables::Unused);

            unused_variables.push_back(variables.get_name(i));
        }
    }

    return unused_variables;
}


Vector<size_t> DataSet::unuse_uncorrelated_variables(const double& minimum_correlation, const Vector<size_t>& nominal_variables)
{
    Vector<double> correlations;

    Vector<size_t> unused_variables;

    if(nominal_variables.empty())
    {
        correlations = calculate_total_input_correlations();

        const Vector<size_t> inputs_indices = variables.get_inputs_indices();

        const size_t inputs_number = inputs_indices.size();

        for(size_t i = 0; i < inputs_number; i++)
        {
            const size_t input_index = inputs_indices[i];

            if(correlations[input_index] < minimum_correlation)
            {
                variables.set_use(input_index, Variables::Unused);

                unused_variables.push_back(input_index);
            }
        }
    }
    else
    {
        correlations = calculate_multiple_total_linear_correlations(nominal_variables);

        const size_t inputs_number = calculate_input_variables_number(nominal_variables);
        const Vector< Vector<size_t> > new_input_indices = get_inputs_indices(inputs_number, nominal_variables);

        for(size_t i = 0; i < inputs_number; i++)
        {
            if(correlations[i] < minimum_correlation)
            {
                for(size_t j = 0; j < new_input_indices[i].size(); j++)
                {
                    variables.set_use(new_input_indices[i][j], Variables::Unused);
                }

                unused_variables.push_back(i);
            }
        }
    }

    return unused_variables;
}


/// Returns a histogram for each variable with a given number of bins.
/// The default number of bins is 10.
/// The format is a vector of subvectors of subsubvectors.
/// The size of the vector is the number of variables.
/// The size of the subvectors is 2(centers and frequencies).
/// The size of the subsubvectors is the number of bins.
/// @param bins_number Number of bins.

Vector< Histogram<double> > DataSet::calculate_data_histograms(const size_t& bins_number) const
{
   const size_t used_variables_number = variables.count_used_variables_number();
   const Vector<size_t> used_variables_indices = variables.get_used_indices();
   const size_t used_instances_number = instances.get_used_instances_number();
   const Vector<size_t> used_instances_indices = instances.get_used_indices();

   const Vector< Vector<size_t> > missing_indices = missing_values.get_missing_indices(used_variables_indices);

   Vector< Histogram<double> > histograms(used_variables_number);

   Vector<double> column(used_instances_number);

   Vector< Vector<size_t> > used_indices(used_variables_number);

   int i = 0;

    #pragma omp parallel for schedule(dynamic)

   for(int i = 0; i < static_cast<int>(used_variables_number); i++)
   {
//       used_indices[static_cast<size_t>(i)] = used_instances_indices.get_difference(missing_values.get_missing_instances(used_variables_indices[static_cast<size_t>(i)]));
       used_indices[static_cast<size_t>(i)] = used_instances_indices.get_difference(missing_indices[static_cast<size_t>(i)]);
   }

   const Vector<string> used_variables_names = variables.get_used_names();

   #pragma omp parallel for private(i, column) shared(histograms)

   for(i = 0; i < static_cast<int>(used_variables_number); i++)
   {
       column = data.get_column(used_variables_indices[static_cast<size_t>(i)], used_indices[static_cast<size_t>(i)]);

       if(column.is_binary_0_1())
       {
           histograms[static_cast<size_t>(i)] = column.calculate_histogram_binary();
       }
       else if(column.get_unique_elements().size() < 10)
       {
           histograms[static_cast<size_t>(i)] = column.calculate_histogram_doubles();
       }
       else
       {
           histograms[static_cast<size_t>(i)] = column.calculate_histogram(bins_number);
       }
   }

   return(histograms);
}


/// Returns a histogram for each target variable with a given number of bins.
/// The default number of bins is 10.
/// The format is a vector of subvectors of subsubvectors.
/// The size of the vector is the number of variables.
/// The size of the subvectors is 2(centers and frequencies).
/// The size of the subsubvectors is the number of bins.
/// @param bins_number Number of bins.

Vector< Histogram<double> > DataSet::calculate_targets_histograms(const size_t& bins_number) const
{
   const size_t targets_number = variables.get_targets_number();

   const Vector<size_t> targets_indices = variables.get_targets_indices();

   const size_t used_instances_number = instances.get_used_instances_number();
   const Vector<size_t> used_instances_indices = instances.get_used_indices();

   const Vector< Vector<size_t> > missing_indices = missing_values.get_missing_indices();

   Vector< Histogram<double> > histograms(targets_number);

   Vector<double> column(used_instances_number);

   for(size_t i = 0; i < targets_number; i++)
   {
       column = data.get_column(targets_indices[i], used_instances_indices);

       histograms[i] = column.calculate_histogram_missing_values(missing_indices[i], bins_number);
   }

   return(histograms);
}


/// Returns a vector of subvectors with the values of a box and whiskers plot.
/// The size of the vector is equal to the number of used variables.
/// The size of the subvectors is 5 and they consist on:
/// <ul>
/// <li> Minimum
/// <li> First quartile
/// <li> Second quartile
/// <li> Third quartile
/// <li> Maximum
/// </ul>

Vector< Vector<double> > DataSet::calculate_box_plots() const
{
    const size_t variables_number = variables.count_used_variables_number();

    const Vector<size_t> used_variables_indices = variables.get_used_indices();

    const Vector<size_t> used_instances_indices = instances.get_used_indices();

    Vector< Vector<size_t> > used_indices(variables_number);

#pragma omp parallel for schedule(dynamic)

    for(int i = 0; i < static_cast<int>(variables_number); i++)
    {
        used_indices[static_cast<size_t>(i)] = used_instances_indices.get_difference(missing_values.get_missing_instances(used_variables_indices[static_cast<size_t>(i)]));
    }

    const Vector< Vector<double> > box_plots = data.calculate_box_plots(used_indices, used_variables_indices);

    return(box_plots);
}


/// Counts the number of negatives of the selected target in the training data.
/// @param target_index Index of the target to evaluate.

size_t DataSet::calculate_training_negatives(const size_t& target_index) const
{
    size_t negatives = 0;

    const Vector<size_t> training_indices = instances.get_training_indices();

    const size_t training_instances_number = training_indices.size();

    #pragma omp parallel for reduction(+: negatives)

    for(int i = 0; i < static_cast<int>(training_instances_number); i++)
    {
        const size_t training_index = training_indices[static_cast<size_t>(i)];

        if(data(training_index, target_index) == 0.0)
        {
            negatives++;
        }
        else if(data(training_index, target_index) != 1.0)
        {
            ostringstream buffer;

           buffer << "OpenNN Exception: DataSet class.\n"
                  << "size_t calculate_training_negatives(const size_t&) const method.\n"
                  << "Training instance is neither a positive nor a negative: " << data(training_index, target_index) << endl;

           throw logic_error(buffer.str());
        }
    }

    return(negatives);
}


/// Counts the number of negatives of the selected target in the selection data.
/// @param target_index Index of the target to evaluate.

size_t DataSet::calculate_selection_negatives(const size_t& target_index) const
{
    size_t negatives = 0;

    const size_t selection_instances_number = instances.get_selection_instances_number();

    const Vector<size_t> selection_indices = instances.get_selection_indices();

    #pragma omp parallel for reduction(+: negatives)

    for(int i = 0; i < static_cast<int>(selection_instances_number); i++)
    {
        const size_t selection_index = selection_indices[static_cast<size_t>(i)];

        if(data(selection_index, target_index) == 0.0)
        {
            negatives++;
        }
        else if(data(selection_index, target_index) != 1.0)
        {
            ostringstream buffer;

           buffer << "OpenNN Exception: DataSet class.\n"
                  << "size_t calculate_selection_negatives(const size_t&) const method.\n"
                  << "Selection instance is neither a positive nor a negative: " << data(selection_index, target_index) << endl;

           throw logic_error(buffer.str());
        }
    }

    return(negatives);
}


/// Counts the number of negatives of the selected target in the testing data.
/// @param target_index Index of the target to evaluate.

size_t DataSet::calculate_testing_negatives(const size_t& target_index) const
{
    size_t negatives = 0;

    const size_t testing_instances_number = instances.get_testing_instances_number();

    const Vector<size_t> testing_indices = instances.get_testing_indices();

    #pragma omp parallel for reduction(+: negatives)

    for(int i = 0; i < static_cast<int>(testing_instances_number); i++)
    {
        const size_t testing_index = testing_indices[static_cast<size_t>(i)];

        if(data(testing_index, target_index) == 0.0)
        {
            negatives++;
        }
        else if(data(testing_index, target_index) != 1.0)
        {
            ostringstream buffer;

           buffer << "OpenNN Exception: DataSet class.\n"
                  << "size_t calculate_selection_negatives(const size_t&) const method.\n"
                  << "Testing instance is neither a positive nor a negative: " << data(testing_index, target_index) << endl;

           throw logic_error(buffer.str());
        }
    }

    return(negatives);
}


/// Returns a vector of vectors containing some basic statistics of all the variables in the data set.
/// The size of this vector is four. The subvectors are:
/// <ul>
/// <li> Minimum.
/// <li> Maximum.
/// <li> Mean.
/// <li> Standard deviation.
/// </ul>

Vector< Statistics<double> > DataSet::calculate_data_statistics() const
{
    const Vector< Vector<size_t> > missing_indices = missing_values.get_missing_indices();

    return(data.calculate_statistics_missing_values(missing_indices));
}


/// Returns a vector fo subvectors containing the shape parameters for all the variables in the data set.
/// The size of this vector is 2. The subvectors are:
/// <ul>
/// <li> Asymmetry.
/// <li> Kurtosis.
/// </ul>

Vector< Vector<double> > DataSet::calculate_data_shape_parameters() const
{
    const Vector< Vector<size_t> > missing_indices = missing_values.get_missing_indices();

    return(data.calculate_shape_parameters_missing_values(missing_indices));
}


/// Returns all the variables statistics from a single matrix.
/// The number of rows is the number of used variables.
/// The number of columns is five(minimum, maximum, mean and standard deviation).

Matrix<double> DataSet::calculate_data_statistics_matrix() const
{
    const size_t variables_number = variables.count_used_variables_number();

    const Vector<size_t> used_variables_indices = variables.get_used_indices();

    const Vector<size_t> used_instances_indices = instances.get_used_indices();

    Vector< Vector<size_t> > used_indices(variables_number);

    Matrix<double> data_statistics_matrix(variables_number, 4);

#pragma omp parallel for schedule(dynamic)

    for(int i = 0; i < static_cast<int>(variables_number); i++)
    {
        used_indices[static_cast<size_t>(i)] = used_instances_indices.get_difference(missing_values.get_missing_instances(used_variables_indices[static_cast<size_t>(i)]));
    }

    const Vector<Statistics<double>> data_statistics_vector = data.calculate_statistics(used_indices, used_variables_indices);

    for(int i = 0; i < static_cast<int>(variables_number); i++)
    {
        data_statistics_matrix.set_row(static_cast<size_t>(i), data_statistics_vector[static_cast<size_t>(i)].to_vector());
    }

    return data_statistics_matrix;
}



Eigen::MatrixXd DataSet::calculate_data_statistics_eigen_matrix() const
{
//    const Matrix<double> statistics = calculate_data_statistics();

    Eigen::MatrixXd eigen_statistics;

    return eigen_statistics;
}


/// Calculate the statistics of the instances with positive targets in binary classification problems.

Matrix<double> DataSet::calculate_positives_data_statistics_matrix() const
{
#ifdef __OPENNN_DEBUG__

    const size_t targets_number = variables.get_targets_number();

    if(targets_number != 1)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: DataSet class.\n"
               << "Matrix<double> calculate_positives_data_statistics_matrix() const method.\n"
               << "Number of targets muste be 1.\n";

        throw logic_error(buffer.str());
    }
#endif

    const size_t target_index = variables.get_targets_indices()[0];

    const Vector<size_t> used_instances_indices = instances.get_used_indices();

    const Vector<double> targets = data.get_column(target_index, used_instances_indices);

#ifdef __OPENNN_DEBUG__

    if(!targets.is_binary())
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: DataSet class.\n"
               << "Matrix<double> calculate_positives_data_statistics_matrix() const method.\n"
               << "Targets vector must be binary.\n";

        throw logic_error(buffer.str());
    }
#endif

    const Vector<size_t> inputs_variables_indices = variables.get_inputs_indices();

    const Vector< Vector<size_t> > missing_indices = missing_values.get_missing_indices();

    const size_t inputs_number = inputs_variables_indices.size();

    const Vector<size_t> positives_used_instances_indices = used_instances_indices.get_subvector(targets.calculate_equal_to_indices(1.0));

    Matrix<double> data_statistics_matrix(inputs_number, 4);

    for(size_t i = 0; i < inputs_number; i++)
    {
        const size_t variable_index = inputs_variables_indices[i];

        const Vector<size_t> current_variable_positives_instances_index = positives_used_instances_indices.get_difference(missing_indices[variable_index]);

        const Vector<double> variable_data = data.get_column(variable_index, current_variable_positives_instances_index);

        const Statistics<double> data_statistics = variable_data.calculate_statistics();

        data_statistics_matrix.set_row(i, data_statistics.to_vector());
    }
    return data_statistics_matrix;
}


/// Calculate the statistics of the instances with neagtive targets in binary classification problems.

Matrix<double> DataSet::calculate_negatives_data_statistics_matrix() const
{
#ifdef __OPENNN_DEBUG__

    const size_t targets_number = variables.get_targets_number();

    if(targets_number != 1)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: DataSet class.\n"
               << "Matrix<double> calculate_positives_data_statistics_matrix() const method.\n"
               << "Number of targets muste be 1.\n";

        throw logic_error(buffer.str());
    }
#endif

    const size_t target_index = variables.get_targets_indices()[0];

    const Vector<size_t> used_instances_indices = instances.get_used_indices();

    const Vector< Vector<size_t> > missing_indices = missing_values.get_missing_indices();

    const Vector<double> targets = data.get_column(target_index, used_instances_indices);

#ifdef __OPENNN_DEBUG__

    if(!targets.is_binary())
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: DataSet class.\n"
               << "Matrix<double> calculate_positives_data_statistics_matrix() const method.\n"
               << "Targets vector must be binary.\n";

        throw logic_error(buffer.str());
    }
#endif

    const Vector<size_t> inputs_variables_indices = variables.get_inputs_indices();

    const size_t inputs_number = inputs_variables_indices.size();

    const Vector<size_t> negatives_used_instances_indices = used_instances_indices.get_subvector(targets.calculate_equal_to_indices(0.0));

    Matrix<double> data_statistics_matrix(inputs_number, 4);

    for(size_t i = 0; i < inputs_number; i++)
    {
        const size_t variable_index = inputs_variables_indices[i];

        const Vector<size_t> current_variable_negatives_instances_index = negatives_used_instances_indices.get_difference(missing_indices[variable_index]);

        const Vector<double> variable_data = data.get_column(variable_index, current_variable_negatives_instances_index);

        const Statistics<double> data_statistics = variable_data.calculate_statistics();

        data_statistics_matrix.set_row(i, data_statistics.to_vector());
    }
    return data_statistics_matrix;
}


/// Returns all the variables shape parameters from a single matrix.
/// The number of rows is the number of used variables.
/// The number of columns is two(asymmetry, kurtosis).

Matrix<double> DataSet::calculate_data_shape_parameters_matrix() const
{
    const Vector< Vector<size_t> > missing_indices = missing_values.get_missing_indices();

    const Vector<size_t> used_variables_indices = variables.get_used_indices();
    const Vector<size_t> used_instances_indices = instances.get_used_indices();

    const Vector< Vector<size_t> > used_missing_indices = missing_indices.get_subvector(used_variables_indices);

    const size_t variables_number = variables.count_used_variables_number();

    Matrix<double> data_shape_parameters_matrix(variables_number, 2);

    for(size_t i = 0; i < variables_number; i++)
    {
        const size_t variable_index = used_variables_indices[i];

        const Vector<double> variable_data = data.get_column(variable_index, used_instances_indices);

        const Vector<double> shape_parameters = variable_data.calculate_shape_parameters_missing_values(used_missing_indices[variable_index]);

        data_shape_parameters_matrix.set_row(i, shape_parameters);
    }

    return(data_shape_parameters_matrix);
}


/// Returns a vector of vectors containing some basic statistics of all variables on the training instances.
/// The size of this vector is two. The subvectors are:
/// <ul>
/// <li> Training data minimum.
/// <li> Training data maximum.
/// <li> Training data mean.
/// <li> Training data standard deviation.
/// </ul>

Vector< Statistics<double> > DataSet::calculate_training_instances_statistics() const
{
   const Vector<size_t> training_indices = instances.get_training_indices();

   const Vector< Vector<size_t> > missing_indices = missing_values.get_missing_indices();

   return(data.calculate_rows_statistics_missing_values(training_indices, missing_indices));
}


/// Returns a vector of vectors containing some basic statistics of all variables on the selection instances.
/// The size of this vector is two. The subvectors are:
/// <ul>
/// <li> Selection data minimum.
/// <li> Selection data maximum.
/// <li> Selection data mean.
/// <li> Selection data standard deviation.
/// </ul>

Vector< Statistics<double> > DataSet::calculate_selection_instances_statistics() const
{
    const Vector<size_t> selection_indices = instances.get_selection_indices();

    const Vector< Vector<size_t> > missing_indices = missing_values.get_missing_indices();

   return(data.calculate_rows_statistics_missing_values(selection_indices, missing_indices));
}


/// Returns a vector of vectors containing some basic statistics of all variables on the testing instances.
/// The size of this vector is five. The subvectors are:
/// <ul>
/// <li> Testing data minimum.
/// <li> Testing data maximum.
/// <li> Testing data mean.
/// <li> Testing data standard deviation.
/// </ul>

Vector< Statistics<double> > DataSet::calculate_testing_instances_statistics() const
{
    const Vector<size_t> testing_indices = instances.get_testing_indices();

    const Vector< Vector<size_t> > missing_indices = missing_values.get_missing_indices();

   return(data.calculate_rows_statistics_missing_values(testing_indices, missing_indices));
}


/// Returns a vector of vectors containing some shape parameters of all variables on the training instances.
/// The size of this vector is two. The subvectors are:
/// <ul>
/// <li> Training data asymmetry.
/// <li> Training data kurtosis.
/// </ul>

Vector< Vector<double> > DataSet::calculate_training_instances_shape_parameters() const
{
   const Vector<size_t> training_indices = instances.get_training_indices();

   const Vector< Vector<size_t> > missing_indices = missing_values.get_missing_indices();

   return(data.calculate_rows_shape_parameters_missing_values(training_indices, missing_indices));
}


/// Returns a vector of vectors containing some shape parameters of all variables on the selection instances.
/// The size of this vector is five. The subvectors are:
/// <ul>
/// <li> Selection data asymmetry.
/// <li> Selection data kurtosis.
/// </ul>

Vector< Vector<double> > DataSet::calculate_selection_instances_shape_parameters() const
{
    const Vector<size_t> selection_indices = instances.get_selection_indices();

    const Vector< Vector<size_t> > missing_indices = missing_values.get_missing_indices();

   return(data.calculate_rows_shape_parameters_missing_values(selection_indices, missing_indices));
}


/// Returns a vector of vectors containing some shape parameters of all variables on the testing instances.
/// The size of this vector is five. The subvectors are:
/// <ul>
/// <li> Testing data asymmetry.
/// <li> Testing data kurtosis.
/// </ul>

Vector< Vector<double> > DataSet::calculate_testing_instances_shape_parameters() const
{
    const Vector<size_t> testing_indices = instances.get_testing_indices();

    const Vector< Vector<size_t> > missing_indices = missing_values.get_missing_indices();

   return(data.calculate_rows_shape_parameters_missing_values(testing_indices, missing_indices));
}


/// Returns a vector of vectors with some basic statistics of the input variables on all instances.
/// The size of this vector is four. The subvectors are:
/// <ul>
/// <li> Input variables minimum.
/// <li> Input variables maximum.
/// <li> Input variables mean.
/// <li> Input variables standard deviation.
/// </ul>

Vector< Statistics<double> > DataSet::calculate_inputs_statistics() const
{
    const size_t inputs_number = variables.get_inputs_number();

    const Vector<size_t> inputs_indices = variables.get_inputs_indices();

    Vector<size_t> used_instances = instances.get_used_indices();

    Vector< Vector<size_t> > used_indices(inputs_number);

#pragma omp parallel for schedule(dynamic)

    for(int i = 0; i < static_cast<int>(inputs_number); i++)
    {
        used_indices[static_cast<size_t>(i)] = used_instances.get_difference(missing_values.get_missing_instances(inputs_indices[static_cast<size_t>(i)]));
    }

    data.calculate_statistics(used_indices, inputs_indices);

    return(data.calculate_statistics(used_indices, inputs_indices));
}


/// Returns a vector of vectors with some basic statistics of the target variables on all instances.
/// The size of this vector is four. The subvectors are:
/// <ul>
/// <li> Target variables minimum.
/// <li> Target variables maximum.
/// <li> Target variables mean.
/// <li> Target variables standard deviation.
/// </ul>

Vector< Statistics<double> > DataSet::calculate_targets_statistics() const
{
   const size_t targets_number = variables.get_targets_number();

   const Vector<size_t> targets_indices = variables.get_targets_indices();

   Vector<size_t> used_instances = instances.get_used_indices();

   Vector< Vector<size_t> > used_indices(targets_number);

#pragma omp parallel for schedule(dynamic)

   for(int i = 0; i < static_cast<int>(targets_number); i++)
   {
       used_indices[static_cast<size_t>(i)] = used_instances.get_difference(missing_values.get_missing_instances(targets_indices[static_cast<size_t>(i)]));
   }

   return(data.calculate_statistics(used_indices, targets_indices));
}


/// Returns a vector of vectors with minimums and maximums of the input variables on all instances with no missing values.
/// The size of this vector is two. The subvectors are:
/// <ul>
/// <li> Input variables minimum.
/// <li> Input variables maximum.
/// </ul>

Vector< Vector<double> > DataSet::calculate_inputs_minimums_maximums() const
{
    const size_t inputs_number = variables.get_inputs_number();

    const Vector<size_t> inputs_indices = variables.get_inputs_indices();

    Vector<size_t> used_instances = instances.get_used_indices();

    Vector< Vector<size_t> > used_indices(inputs_number);

#pragma omp parallel for schedule(dynamic)

   for(int i = 0; i < static_cast<int>(inputs_number); i++)
   {
       used_indices[static_cast<size_t>(i)] = used_instances.get_difference(missing_values.get_missing_instances(inputs_indices[static_cast<size_t>(i)]));
   }

    return data.calculate_columns_minimums_maximums(used_indices, inputs_indices);
}


/// Returns a vector of vectors with minimums and maximums of the targets variables on all instances with no missing values.
/// The size of this vector is two. The subvectors are:
/// <ul>
/// <li> Target variables minimum.
/// <li> Target variables maximum.
/// </ul>

Vector< Vector<double> > DataSet::calculate_targets_minimums_maximums() const
{
    const size_t targets_number = variables.get_targets_number();

    const Vector<size_t> targets_indices = variables.get_targets_indices();

    Vector<size_t> used_instances = instances.get_used_indices();

    Vector< Vector<size_t> > used_indices(targets_number);

#pragma omp parallel for schedule(dynamic)

   for(int i = 0; i < static_cast<int>(targets_number); i++)
   {
       used_indices[static_cast<size_t>(i)] = used_instances.get_difference(missing_values.get_missing_instances(targets_indices[static_cast<size_t>(i)]));
   }

    return data.calculate_columns_minimums_maximums(used_indices, targets_indices);
}



/// Returns a vector containing the means of a set of given variables.
/// @param variables_indices Indices of the variables.

Vector<double> DataSet::calculate_variables_means(const Vector<size_t>& variables_indices) const
{
    const size_t variables_number = variables_indices.size();

    Vector<double> means(variables_number, 0.0);

#pragma omp parallel for

    for(int i = 0; i < static_cast<int>(variables_number); i++)
    {
        const size_t variable_index = variables_indices[static_cast<size_t>(i)];

        means[static_cast<size_t>(i)] = data.get_column(variable_index).calculate_mean();
    }

    return means;
}


/// Returns a vector with some basic statistics of the given input variable on all instances.
/// The size of this vector is four:
/// <ul>
/// <li> Input variable minimum.
/// <li> Input variable maximum.
/// <li> Input variable mean.
/// <li> Input variable standard deviation.
/// </ul>

Statistics<double> DataSet::calculate_input_statistics(const size_t& input_index) const
{
    Vector<size_t> missing_indices = missing_values.get_missing_indices()[input_index];

    const Vector<size_t> unused_instances_indices = instances.get_unused_indices();

    for(size_t i = 0; i < unused_instances_indices.size(); i++)
    {
        missing_indices.push_back(unused_instances_indices[i]);
    }

   return(data.get_column(input_index).calculate_statistics_missing_values(missing_indices));
}


/// Returns the mean values of the target variables on the training instances.

Vector<double> DataSet::calculate_training_targets_mean() const
{
    const Vector<size_t> training_indices = instances.get_training_indices();

    const Vector<size_t> targets_indices = variables.get_targets_indices();

    const Vector< Vector<size_t> > missing_indices = missing_values.get_missing_indices(targets_indices);

    return(data.calculate_mean_missing_values(training_indices, targets_indices, missing_indices));
}


/// Returns the mean values of the target variables on the selection instances.

Vector<double> DataSet::calculate_selection_targets_mean() const
{
    const Vector<size_t> selection_indices = instances.get_selection_indices();

    const Vector<size_t> targets_indices = variables.get_targets_indices();

    const Vector< Vector<size_t> > missing_indices = missing_values.get_missing_indices(targets_indices);

    return(data.calculate_mean_missing_values(selection_indices, targets_indices, missing_indices));
}


/// Returns the mean values of the target variables on the testing instances.

Vector<double> DataSet::calculate_testing_targets_mean() const
{
   const Vector<size_t> testing_indices = instances.get_testing_indices();

   const Vector<size_t> targets_indices = variables.get_targets_indices();

   const Vector< Vector<size_t> > missing_indices = missing_values.get_missing_indices(targets_indices);

   return(data.calculate_mean_missing_values(testing_indices, targets_indices, missing_indices));
}


/// @todo

Vector< Vector< Vector<double> > > DataSet::calculate_means_columns() const
{
    const size_t inputs_number = variables.get_inputs_number();
    const size_t targets_number = variables.get_targets_number();
    const Vector<size_t> inputs_indices = variables.get_inputs_indices();

    const Vector<size_t> targets_indices = variables.get_targets_indices();

    Vector< Vector< Vector<double> > > means(inputs_number);

    Vector<size_t> input_target_indices(2);

    for(size_t i = 0; i < inputs_number; i++)
    {
        input_target_indices[0] = inputs_indices[i];

        for(size_t j = 0; j < targets_number; j++)
        {
            input_target_indices[1] = targets_indices[j];
        }
    }

    return means;
}


/// Calculates the linear correlations between all outputs and all inputs.
/// It returns a matrix with number of rows the targets number and number of columns the inputs number.
/// Each element contains the linear correlation between a single target and a single output.

Matrix<double> DataSet::calculate_input_target_correlations() const
{
   const size_t inputs_number = variables.get_inputs_number();
   const size_t targets_number = variables.get_targets_number();

   const Vector<size_t> input_indices = variables.get_inputs_indices();
   const Vector<size_t> target_indices = variables.get_targets_indices();

   const Vector<size_t> unused_instances_indices = instances.get_unused_indices();

   Matrix<double> correlations(inputs_number, targets_number);

    #ifndef __OPENNN_MPI__
    #pragma omp parallel for
    #endif

   for(int i = 0; i < static_cast<int>(inputs_number); i++)
   {
       const size_t input_index = input_indices[static_cast<size_t>(i)];

       Vector<double> input_variable = data.get_column(input_index);

       const bool binary_input = input_variable.is_binary();

       if(binary_input && input_variable.contains(-1))
       {
           input_variable = input_variable.replace_value(-1,0);
       }

       const Vector<size_t> input_missing_instances = missing_values.get_missing_instances(input_index);

       for(size_t j = 0; j < targets_number; j++)
       {
           const size_t target_index = target_indices[j];

           Vector<double> target_variable = data.get_column(target_index);

           const bool binary_target = target_variable.is_binary();

           if(binary_target && target_variable.contains(-1))
           {
               target_variable = target_variable.replace_value(-1,0);
           }

           const Vector<size_t> target_missing_instances = missing_values.get_missing_instances(target_index);

           const Vector<size_t> missing_instances = input_missing_instances.get_union(target_missing_instances);

           const Vector<size_t> this_unused_instances = missing_instances.get_union(unused_instances_indices);

           if(!binary_input && !binary_target)
           {
               correlations(static_cast<size_t>(i),j) = CorrelationAnalysis::calculate_linear_correlation_missing_values(input_variable,target_variable, this_unused_instances);
           }
           else if(!binary_input && binary_target)
           {
               correlations(static_cast<size_t>(i),j) = CorrelationAnalysis::calculate_logistic_correlation_missing_values(input_variable,target_variable, this_unused_instances);
           }
           else if(binary_input && !binary_target)
           {// asda
               correlations(static_cast<size_t>(i),j) = CorrelationAnalysis::calculate_logistic_correlation_missing_values(target_variable,input_variable, this_unused_instances);
           }
           else if(binary_input && binary_target)
           {
               correlations(static_cast<size_t>(i),j) = CorrelationAnalysis::calculate_linear_correlation_missing_values(target_variable,target_variable, this_unused_instances);
           }
           else
           {
               cout << "Unknown case" << endl;
           }
       }
   }

   return correlations;
}


Eigen::MatrixXd DataSet::calculate_input_target_correlations_eigen() const
{
    const size_t target_number = variables.get_targets_number();
    const size_t input_number = variables.get_inputs_number();

    const Eigen::Map<Eigen::MatrixXd> correlations((double*)calculate_input_target_correlations().data(), target_number, input_number);

    return(correlations);
}


Vector<double> DataSet::calculate_total_input_correlations() const
{
    const Matrix<double> correlations = calculate_input_target_correlations();

    return correlations.calculate_absolute_value().calculate_rows_sum();
}


void DataSet::print_missing_values_information() const
{
    const size_t variables_number = variables.get_variables_number();

    const size_t instances_number = instances.get_instances_number();

    const Vector<size_t> variables_missing_values = missing_values.count_variables_missing_indices();


    cout << "Missing values: " << endl;

    for(size_t i = 0; i < variables_number; i++)
    {
        if(variables.is_used(i))
        {
            cout << variables.get_name(i) << ": " << variables_missing_values[i]*100.0/(static_cast<double>(instances_number)-1.0) << "%" << endl;
        }
    }
}


void DataSet::print_input_target_correlations() const
{
    const size_t inputs_number = variables.get_inputs_number();
    const size_t targets_number = variables.get_targets_number();

    const Vector<string> inputs_name = variables.get_inputs_name();
    const Vector<string> targets_name = variables.get_targets_name();

    const Matrix<double> linear_correlations = calculate_input_target_correlations();

    for(size_t j = 0; j < targets_number; j++)
    {
        for(size_t i = 0; i < inputs_number; i++)
        {
            cout << targets_name[j] << " - " << inputs_name[i] << ": " << linear_correlations(i,j) << endl;
        }
    }
}


void DataSet::print_top_input_target_correlations(const size_t& number) const
{
    const size_t inputs_number = variables.get_inputs_number();
    const size_t targets_number = variables.get_targets_number();

    const Vector<string> inputs_name = variables.get_inputs_name();
    const Vector<string> targets_name = variables.get_targets_name();

    const Matrix<double> linear_correlations = calculate_input_target_correlations();

    Vector<double> target_correlations(inputs_number);

    Matrix<string> top_correlations(inputs_number, 2);

    const size_t end = (number < inputs_number) ? number : inputs_number;

    for(size_t j = 0; j < targets_number; j++)
    {
        cout << targets_name[j] << endl;

        target_correlations = linear_correlations.get_column(j);

        top_correlations.set_column(0, inputs_name, "input");
        top_correlations.set_column(1, target_correlations.to_string_vector(), "correlation");

        top_correlations = top_correlations.sort_descending_strings_absolute_value(1);

        for(size_t i = 0; i < end; i++)
        {
            cout << i+1 << ": " << targets_name[j] << " - " << top_correlations(i,0) << ": " << top_correlations(i,1) << endl;
        }

        cout << endl;
    }
}


Matrix<double> DataSet::calculate_variables_correlations() const
{
    const size_t variables_number = variables.get_variables_number();

    const Vector< Vector<size_t> > missing_indices = missing_values.get_missing_indices();

    Matrix<double> correlations(variables_number, variables_number, -99.9);

    correlations.initialize_identity();

    Vector<double> variable_i;
    Vector<double> variable_j;

    Vector<size_t> missing_indices_i;
    Vector<size_t> missing_indices_j;

    for(size_t i = 0; i < variables_number; i++)
    {
        variable_i = get_variable(i);

        missing_indices_i = missing_indices[i];

        for(size_t j = i; j < variables_number; j++)
        {
            variable_j = get_variable(j);

            missing_indices_j = missing_indices[j];

            correlations(i,j) = CorrelationAnalysis::calculate_linear_correlation_missing_values(variable_i,variable_j, missing_indices_i.get_union(missing_indices_j));
        }
    }

    for(size_t i = 0; i < variables_number; i++)
    {
        for(size_t j = 0; j < i; j++)
        {
            correlations(i,j) = correlations(j,i);
        }
    }

    return(correlations);
}


void DataSet::print_variables_correlations() const
{
    const Vector<string> variables_names = variables.get_names();

    const Matrix<double> variables_correlations = calculate_variables_correlations();

    cout << variables_names << endl
         << variables_correlations << endl
         << endl;
}


void DataSet::print_top_variables_correlations(const size_t& number) const
{
    const size_t variables_number = variables.get_variables_number();

    const Vector<string> variables_name = variables.get_names();

    const Matrix<double> variables_correlations = calculate_variables_correlations();

    const size_t correlations_number = variables_number*(variables_number-1)/2;

    Matrix<string> top_correlations(correlations_number, 3);

    size_t row = 0;

    for(size_t i = 0; i < variables_number; i++)
    {
        for(size_t j = i; j < variables_number; j++)
        {
            if(i == j) continue;

            top_correlations(row, 0) = variables_name[i];
            top_correlations(row, 1) = variables_name[j];
            top_correlations(row, 2) = to_string(variables_correlations(i,j));

            row++;
        }
    }

    top_correlations = top_correlations.sort_descending_strings_absolute_value(2);

    const size_t end = (number < correlations_number) ? number : correlations_number;

    for(size_t i = 0; i < end; i++)
    {
        cout << top_correlations(i,0) << " - " << top_correlations(i,1) << ": " << top_correlations(i,2) << endl;
    }

    cout << endl;
}


/// Calculates the linear correlations between each input and each target variable.
/// For nominal input variables it calculates the multiple linear correlation between all the classes of
/// the input variable and the target.
/// @param nominal_variables Vector containing the classes of each nominal variable.

Matrix<double> DataSet::calculate_multiple_linear_correlations(const Vector<size_t> & nominal_variables) const
{
    const Vector<size_t> targets_indices = variables.get_targets_indices();
    const size_t targets_number = variables.get_targets_number();

    const Vector<size_t> used_instances_indices = instances.get_used_indices();

    const size_t inputs_number = calculate_input_variables_number(nominal_variables);
    const Vector< Vector<size_t> > new_input_indices = get_inputs_indices(inputs_number, nominal_variables);

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

            const Vector<size_t> current_missing_values = missing_values.get_missing_instances(input_indices)
                                          .get_union(missing_values.get_missing_instances(target_index));

            const Vector<size_t> current_used_indices = used_instances_indices.get_difference(current_missing_values);

            input_variables = data.get_submatrix(current_used_indices, input_indices);

            target_variable = data.get_column(target_index, current_used_indices);

            if(input_variables.get_columns_number() == 1)
            {
                multiple_linear_correlations(i,j) = CorrelationAnalysis::calculate_linear_correlation(input_variables.to_vector(), target_variable);
            }
            else
            {
                multiple_linear_correlations(i,j) = CorrelationAnalysis::calculate_multiple_linear_correlation(input_variables,target_variable);
            }
        }
    }

    return multiple_linear_correlations;
}


Vector<double> DataSet::calculate_multiple_total_linear_correlations(const Vector<size_t>& nominal_variables) const
{
    const Matrix<double> correlations = calculate_multiple_linear_correlations(nominal_variables);

    return correlations.calculate_absolute_value().calculate_rows_sum();
}

/*
Matrix<double> DataSet::calculate_multiple_logistic_correlations(const Vector<size_t>& nominal_variables) const
{
    const Vector<size_t> targets_indices = variables.get_targets_indices();
    const size_t targets_number = variables.get_targets_number();

    const Vector<size_t> used_instances_indices = instances.get_used_indices();

    const size_t inputs_number = calculate_input_variables_number(nominal_variables);
    const Vector< Vector<size_t> > new_input_indices = get_inputs_indices(inputs_number, nominal_variables);

    // Calculate correlations

    Matrix<double> multiple_logistic_correlations(inputs_number, targets_number);

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

            input_variables = data.get_submatrix(current_used_indices, input_indices);

            target_variable = data.get_column(target_index, current_used_indices);

            if(input_variables.get_columns_number() == 1)
            {
//                multiple_logistic_correlations(i,j) = input_variables.to_vector().calculate_logistic_correlation(target_variable);
                multiple_logistic_correlations(i,j) = CorrelationAnalysis::calculate_logistic_correlation(input_variables.to_vector(), target_variable);
            }
            else
            {
                multiple_logistic_correlations(i,j) = input_variables.calculate_multiple_logistic_correlation(target_variable);
            }
        }
    }

    return multiple_logistic_correlations;
}
*/


/// Calculates the number of inputs taking nominal inputs as just one variable.

size_t DataSet::calculate_input_variables_number(const Vector<size_t>& nominal_variables) const
{
    const Vector<size_t> inputs_indices = variables.get_inputs_indices();

    size_t current_nominal_classes;
    size_t current_variable_index = 0;
    size_t inputs_number = 0;

    const size_t variables_number = nominal_variables.size();

    for(size_t i = 0; i < variables_number; i++)
    {
        current_nominal_classes = nominal_variables[i];

        if(inputs_indices.contains(current_variable_index))
        {
            inputs_number++;
        }

        current_variable_index += current_nominal_classes;
    }

    return inputs_number;
}


/// Returns the inputs indices taking into account the nominal variables.
/// @param inputs_number New inputs number
/// @param nominal_variables Vector containing the classes of each nominal variable.

Vector< Vector<size_t> > DataSet::get_inputs_indices(const size_t& inputs_number, const Vector<size_t>& nominal_variables) const
{
    const Vector<size_t> inputs_indices = variables.get_inputs_indices();

    const size_t variables_number = nominal_variables.size();

    // Count input variables

    size_t current_nominal_classes;

    size_t current_variable_index = 0;


    Vector< Vector<size_t> > new_input_indices(inputs_number);
    size_t new_input_index = 0;
    size_t old_input_index = 0;

    for(size_t i = 0; i < variables_number; i++)
    {
        current_nominal_classes = nominal_variables[i];

        if(inputs_indices.contains(current_variable_index))
        {
            Vector<size_t> new_indices(current_nominal_classes);

           for(size_t j = 0; j < current_nominal_classes; j++)
           {
               new_indices[j] = inputs_indices[old_input_index];
               old_input_index++;
           }

           new_input_indices[new_input_index] = new_indices;
           new_input_index++;
        }

        current_variable_index += current_nominal_classes;

        if(old_input_index > inputs_indices.size())
        {
            new_input_indices.set(inputs_indices.size());

            for(size_t j = 0; j < inputs_indices.size(); j++)
            {
                new_input_indices[j].set(1, inputs_indices[j]);
            }

            break;
        }
    }

    return new_input_indices;
}


/// Returns the covariance matrix for the input data set.
/// The number of rows of the matrix is the number of inputs.
/// The number of columns of the matrix is the number of inputs.

Matrix<double> DataSet::calculate_covariance_matrix() const
{
    const Vector<size_t> inputs_indices = variables.get_inputs_indices();
    const Vector<size_t> used_instances_indices = instances.get_used_indices();

    const size_t inputs_number = variables.get_inputs_number();

    Matrix<double> covariance_matrix(inputs_number, inputs_number, 0.0);

#pragma omp parallel for schedule(dynamic)

    for(int i = 0; i < static_cast<int>(inputs_number); i++)
    {
        const size_t first_input_index = inputs_indices[static_cast<size_t>(i)];

        const Vector<double> first_inputs = data.get_column(first_input_index, used_instances_indices);

        for(size_t j = static_cast<size_t>(i); j < inputs_number; j++)
        {
            const size_t second_input_index = inputs_indices[j];

            const Vector<double> second_inputs = data.get_column(second_input_index, used_instances_indices);

            covariance_matrix(static_cast<size_t>(i),j) = first_inputs.calculate_covariance(second_inputs);
            covariance_matrix(j,static_cast<size_t>(i)) = covariance_matrix(static_cast<size_t>(i),j);
        }
    }

    return(covariance_matrix);
}


/// Performs the principal components analysis of the inputs.
/// It returns a matrix containing the principal components getd in rows.
/// This method deletes the unused instances of the original data set.
/// @param minimum_explained_variance Minimum percentage of variance used to select a principal component.

Matrix<double> DataSet::perform_principal_components_analysis(const double& minimum_explained_variance)
{
    cout << "hola" << endl;
    system("pause");
    //const Vector<size_t> inputs_indices = variables.get_inputs_indices();

    // Subtract off the mean

    remove_inputs_mean();

    // Calculate covariance matrix

    const Matrix<double> covariance_matrix = calculate_covariance_matrix();

    // Calculate eigenvectors

    const Matrix<double> eigenvectors = covariance_matrix.calculate_eigenvectors();

    // Calculate eigenvalues

    const Matrix<double> eigenvalues = covariance_matrix.calculate_eigenvalues();

    // Calculate explained variance

    const Vector<double> explained_variance = eigenvalues.get_column(0).calculate_explained_variance();

    // Sort principal components

    const Vector<size_t> sorted_principal_components_indices = explained_variance.sort_descending_indices();

    // Choose eigenvectors

    const size_t inputs_number = covariance_matrix.get_columns_number();

    Vector<size_t> principal_components_indices;

    size_t index;

    for(size_t i = 0; i < inputs_number; i++)
    {
        index = sorted_principal_components_indices[i];

        if(explained_variance[index] >= minimum_explained_variance)
        {
            principal_components_indices.push_back(i);
        }
        else
        {
            continue;
        }
    }

    const size_t principal_components_number = principal_components_indices.size();

    // Arrange principal components matrix

    Matrix<double> principal_components;

    if(principal_components_number == 0)
    {
        return principal_components;
    }
    else
    {
        principal_components.set(principal_components_number, inputs_number);
    }

    for(size_t i = 0; i < principal_components_number; i++)
    {
        index = sorted_principal_components_indices[i];

        principal_components.set_row(i, eigenvectors.get_column(index));
    }

    // Return feature matrix

    return principal_components.get_submatrix_rows(principal_components_indices);
}


/// Performs the principal components analysis of the inputs.
/// It returns a matrix containing the principal components arranged in rows.
/// This method deletes the unused instances of the original data set.
/// @param covariance_matrix Matrix of covariances.
/// @param explained_variance Vector of the explained variances of the variables.
/// @param minimum_explained_variance Minimum percentage of variance used to select a principal component.

Matrix<double> DataSet::perform_principal_components_analysis(const Matrix<double>& covariance_matrix,
                                                              const Vector<double>& explained_variance,
                                                              const double& minimum_explained_variance)
{
    // Subtract off the mean

    remove_inputs_mean();

    // Calculate eigenvectors

    const Matrix<double> eigenvectors = covariance_matrix.calculate_eigenvectors();

    // Sort principal components

    const Vector<size_t> sorted_principal_components_indices = explained_variance.sort_descending_indices();

    // Choose eigenvectors

    const size_t inputs_number = covariance_matrix.get_columns_number();

    Vector<size_t> principal_components_indices;

    size_t index;

    for(size_t i = 0; i < inputs_number; i++)
    {
        index = sorted_principal_components_indices[i];

        if(explained_variance[index] >= minimum_explained_variance)
        {
            principal_components_indices.push_back(i);
        }
        else
        {
            continue;
        }
    }

    const size_t principal_components_number = principal_components_indices.size();

    // Arrange principal components matrix

    Matrix<double> principal_components;

    if(principal_components_number == 0)
    {
        return principal_components;
    }
    else
    {
        principal_components.set(principal_components_number, inputs_number);
    }

    for(size_t i = 0; i < principal_components_number; i++)
    {
        index = sorted_principal_components_indices[i];

        principal_components.set_row(i, eigenvectors.get_column(index));
    }

    // Return feature matrix

    return principal_components.get_submatrix_rows(principal_components_indices);
}


/// Transforms the data according to the principal components.
/// @param principal_components Matrix containing the principal components.

void DataSet::transform_principal_components_data(const Matrix<double>& principal_components)
{
    const Matrix<double> targets = get_targets();

    remove_inputs_mean();

    const size_t principal_components_number = principal_components.get_rows_number();

    // Transform data

    const Vector<size_t> used_instances = instances.get_used_indices();

    const size_t new_instances_number = used_instances.size();

    const Matrix<double> inputs = get_inputs();

    Matrix<double> new_data(new_instances_number, principal_components_number, 0.0);

    size_t instance_index;

    for(size_t i = 0; i < new_instances_number; i++)
    {
        instance_index = used_instances[i];

        for(size_t j = 0; j < principal_components_number; j++)
        {
            new_data(i,j) = inputs.get_row(instance_index).dot(principal_components.get_row(j));
        }
    }

    data = new_data.assemble_columns(targets);
}


/// Scales the data matrix with given mean and standard deviation values.
/// It updates the data matrix.
/// @param data_statistics Vector of statistics structures for all the variables in the data set.
/// The size of that vector must be equal to the number of variables.

void DataSet::scale_data_mean_standard_deviation(const Vector< Statistics<double> >& data_statistics)
{
   // Control sentence(if debug)

   #ifdef __OPENNN_DEBUG__

   ostringstream buffer;

   const size_t columns_number = data.get_columns_number();

   const size_t statistics_size = data_statistics.size();

   if(statistics_size != columns_number)
   {
      buffer << "OpenNN Exception: DataSet class.\n"
             << "void scale_data_mean_standard_deviation(const Vector< Statistics<double> >&) method.\n"
             << "Size of statistics must be equal to number of columns.\n";

      throw logic_error(buffer.str());
   }

   #endif

   const size_t variables_number = variables.get_variables_number();

   for(size_t i = 0; i < variables_number; i++)
   {
       if(display && data_statistics[i].standard_deviation < numeric_limits<double>::min())
       {
          cout << "OpenNN Warning: DataSet class.\n"
                    << "void scale_data_mean_standard_deviation(const Vector< Statistics<Type> >&) method.\n"
                    << "Standard deviation of variable " <<  i << " is zero.\n"
                    << "That variable won't be scaled.\n";
        }
    }

   data.scale_mean_standard_deviation(data_statistics);
}


/// Scales the data using the minimum and maximum method,
/// and the minimum and maximum values calculated from the data matrix.
/// It also returns the statistics from all columns.

Vector< Statistics<double> > DataSet::scale_data_minimum_maximum()
{
    const Vector< Statistics<double> > data_statistics = calculate_data_statistics();

    scale_data_minimum_maximum(data_statistics);

    return(data_statistics);
}


/// Scales the data using the mean and standard deviation method,
/// and the mean and standard deviation values calculated from the data matrix.
/// It also returns the statistics from all columns.

Vector< Statistics<double> > DataSet::scale_data_mean_standard_deviation()
{
    const Vector< Statistics<double> > data_statistics = calculate_data_statistics();

    scale_data_mean_standard_deviation(data_statistics);

    return(data_statistics);
}


/// Subtracts off the mean to every of the input variables.

void DataSet::remove_inputs_mean()
{
    Vector< Statistics<double> > input_statistics = calculate_inputs_statistics();

    Vector<size_t> inputs_indices = variables.get_inputs_indices();
    Vector<size_t> used_instances_indices = instances.get_used_indices();

    size_t input_index;
    size_t instance_index;

    double input_mean;

    for(size_t i = 0; i < inputs_indices.size(); i++)
    {
        input_index = inputs_indices[i];

        input_mean = input_statistics[i].mean;

        for(size_t j = 0; j < used_instances_indices.size(); j++)
        {
            instance_index = used_instances_indices[j];

            data(instance_index,input_index) = data(instance_index,input_index) - input_mean;
        }
    }
}


/// Returns a vector of strings containing the scaling method that best fits each
/// of the input variables.

Vector<string> DataSet::calculate_default_scaling_methods() const
{
    const Vector<size_t> used_inputs_indices = variables.get_inputs_indices();
    const size_t used_inputs_number = used_inputs_indices.size();

    //const Vector< Vector<size_t> > missing_indices = missing_values.get_missing_indices();

    size_t current_distribution;
    Vector<string> scaling_methods(used_inputs_number);

//#pragma omp parallel for private(current_distribution)

    for(int i = 0; i < static_cast<int>(used_inputs_number); i++)
    {
//        current_distribution = data.get_column(used_inputs_indices[i]).perform_distribution_distance_analysis_missing_values(missing_indices[i]);
        current_distribution = data.get_column(used_inputs_indices[static_cast<size_t>(i)]).perform_distribution_distance_analysis();

        if(current_distribution == 0) // Normal distribution
        {
            scaling_methods[static_cast<size_t>(i)] = "MeanStandardDeviation";
        }
//        else if(current_distribution == 1) // Half-normal distribution
//        {
//            scaling_methods[static_cast<size_t>(i)] = "StandardDeviation";
//        }
        else if(current_distribution == 1) // Uniform distribution
        {
            scaling_methods[static_cast<size_t>(i)] = "MinimumMaximum";
        }
        else // Default
        {
            scaling_methods[static_cast<size_t>(i)] = "MinimumMaximum";
        }
    }

    return scaling_methods;
}


/// Scales the data matrix with given minimum and maximum values.
/// It updates the data matrix.
/// @param data_statistics Vector of statistics structures for all the variables in the data set.
/// The size of that vector must be equal to the number of variables.

void DataSet::scale_data_minimum_maximum(const Vector< Statistics<double> >& data_statistics)
{
    const size_t variables_number = variables.get_variables_number();

   // Control sentence(if debug)

   #ifdef __OPENNN_DEBUG__

   ostringstream buffer;

   const size_t statistics_size = data_statistics.size();

   if(statistics_size != variables_number)
   {
      buffer << "OpenNN Exception: DataSet class.\n"
             << "void scale_data_minimum_maximum(const Vector< Statistics<double> >&) method.\n"
             << "Size of data statistics must be equal to number of variables.\n";

      throw logic_error(buffer.str());
   }

   #endif

   for(size_t i = 0; i < variables_number; i++)
   {
       if(display && data_statistics[i].maximum-data_statistics[i].minimum < numeric_limits<double>::min())
       {
          cout << "OpenNN Warning: DataSet class.\n"
                    << "void scale_data_minimum_maximum(const Vector< Statistics<Type> >&) method.\n"
                    << "Range of variable " <<  i << " is zero.\n"
                    << "That variable won't be scaled.\n";
        }
    }


   data.scale_minimum_maximum(data_statistics);
}


/// Scales the input variables with given mean and standard deviation values.
/// It updates the input variables of the data matrix.
/// @param inputs_statistics Vector of statistics structures for the input variables.
/// The size of that vector must be equal to the number of inputs.

void DataSet::scale_inputs_mean_standard_deviation(const Vector< Statistics<double> >& inputs_statistics)
{
    const Vector<size_t> inputs_indices = variables.get_inputs_indices();

    data.scale_columns_mean_standard_deviation(inputs_statistics, inputs_indices);
}


/// Scales the input variables with the calculated mean and standard deviation values from the data matrix.
/// It updates the input variables of the data matrix.
/// It also returns a vector of vectors with the variables statistics.

Vector< Statistics<double> > DataSet::scale_inputs_mean_standard_deviation()
{
    // Control sentence(if debug)

    #ifdef __OPENNN_DEBUG__

    if(data.empty())
    {
       ostringstream buffer;

       buffer << "OpenNN Exception: DataSet class.\n"
              << "Vector< Statistics<double> > scale_inputs_mean_standard_deviation() method.\n"
              << "Data file is not loaded.\n";

       throw logic_error(buffer.str());
    }

    #endif

   const Vector< Statistics<double> > inputs_statistics = calculate_inputs_statistics();

   scale_inputs_mean_standard_deviation(inputs_statistics);

   return(inputs_statistics);
}


/// Scales the given input variables with given mean and standard deviation values.
/// It updates the input variable of the data matrix.
/// @param inputs_statistics Vector of statistics structures for the input variables.
/// @param input_index Index of the input to be scaled.

void DataSet::scale_input_mean_standard_deviation(const Statistics<double>& input_statistics, const size_t& input_index)
{
    Vector<double> column = data.get_column(input_index);
    column.scale_mean_standard_deviation(input_statistics);

    data.set_column(input_index, column, "");
}


/// Scales the given input variables with the calculated mean and standard deviation values from the data matrix.
/// It updates the input variables of the data matrix.
/// It also returns a vector with the variables statistics.
/// @param input_index Index of the input to be scaled.

Statistics<double> DataSet::scale_input_mean_standard_deviation(const size_t& input_index)
{
    // Control sentence(if debug)

    #ifdef __OPENNN_DEBUG__

    if(data.empty())
    {
       ostringstream buffer;

       buffer << "OpenNN Exception: DataSet class.\n"
              << "Statistics<double> scale_input_mean_standard_deviation(const size_t&) method.\n"
              << "Data file is not loaded.\n";

       throw logic_error(buffer.str());
    }

    #endif

   const Statistics<double> input_statistics = calculate_input_statistics(input_index);

   scale_input_mean_standard_deviation(input_statistics, input_index);

   return(input_statistics);
}


/// Scales the given input variables with given standard deviation values.
/// It updates the input variable of the data matrix.
/// @param inputs_statistics Vector of statistics structures for the input variables.
/// @param input_index Index of the input to be scaled.

void DataSet::scale_input_standard_deviation(const Statistics<double>& input_statistics, const size_t& input_index)
{
    Vector<double> column = data.get_column(input_index);
    column.scale_standard_deviation(input_statistics);

    data.set_column(input_index, column, "");
}


/// Scales the given input variables with the calculated standard deviation values from the data matrix.
/// It updates the input variables of the data matrix.
/// It also returns a vector with the variables statistics.
/// @param input_index Index of the input to be scaled.

Statistics<double> DataSet::scale_input_standard_deviation(const size_t& input_index)
{
    // Control sentence(if debug)

    #ifdef __OPENNN_DEBUG__

    if(data.empty())
    {
       ostringstream buffer;

       buffer << "OpenNN Exception: DataSet class.\n"
              << "Statistics<double> scale_input_standard_deviation(const size_t&) method.\n"
              << "Data file is not loaded.\n";

       throw logic_error(buffer.str());
    }

    #endif

   const Statistics<double> input_statistics = calculate_input_statistics(input_index);

   scale_input_standard_deviation(input_statistics, input_index);

   return(input_statistics);
}


/// Scales the input variables with given minimum and maximum values.
/// It updates the input variables of the data matrix.
/// @param inputs_statistics Vector of statistics structures for all the inputs in the data set.
/// The size of that vector must be equal to the number of input variables.

void DataSet::scale_inputs_minimum_maximum(const Vector< Statistics<double> >& inputs_statistics)
{
    const Vector<size_t> inputs_indices = variables.get_inputs_indices();

    data.scale_columns_minimum_maximum(inputs_statistics, inputs_indices);
}


/// Scales the input variables with the calculated minimum and maximum values from the data matrix.
/// It updates the input variables of the data matrix.
/// It also returns a vector of vectors with the minimum and maximum values of the input variables.

Vector< Statistics<double> > DataSet::scale_inputs_minimum_maximum()
{
    // Control sentence(if debug)

    #ifdef __OPENNN_DEBUG__

    if(data.empty())
    {
       ostringstream buffer;

       buffer << "OpenNN Exception: DataSet class.\n"
              << "Vector< Statistics<double> > scale_inputs_minimum_maximum() method.\n"
              << "Data file is not loaded.\n";

       throw logic_error(buffer.str());
    }

    #endif

   const Vector< Statistics<double> > inputs_statistics = calculate_inputs_statistics();

   scale_inputs_minimum_maximum(inputs_statistics);

   return(inputs_statistics);
}


Eigen::MatrixXd DataSet::scale_inputs_minimum_maximum_eigen()
{
    const Vector< Statistics<double> > inputs_statistics = scale_inputs_minimum_maximum();

    const size_t inputs_number = inputs_statistics.size();

    Eigen::MatrixXd eigen_matrix(inputs_number, 4);

    for(size_t i = 0; i < inputs_number; i++)
    {
        eigen_matrix(i,0) = inputs_statistics[i].minimum;
        eigen_matrix(i,1) = inputs_statistics[i].maximum;
        eigen_matrix(i,2) = inputs_statistics[i].mean;
        eigen_matrix(i,3) = inputs_statistics[i].standard_deviation;
    }

    return eigen_matrix;
}



/// Scales the given input variable with given minimum and maximum values.
/// It updates the input variables of the data matrix.
/// @param input_statistics Vector with the statistics of the input variable.
/// @param input_index Index of the input to be scaled.

void DataSet::scale_input_minimum_maximum(const Statistics<double>& input_statistics, const size_t & input_index)
{
    Vector<double> column = data.get_column(input_index);
    column.scale_minimum_maximum(input_statistics);

    data.set_column(input_index, column, "");
}


/// Scales the given input variable with the calculated minimum and maximum values from the data matrix.
/// It updates the input variable of the data matrix.
/// It also returns a vector with the minimum and maximum values of the input variables.

Statistics<double> DataSet::scale_input_minimum_maximum(const size_t& input_index)
{
    // Control sentence(if debug)

    #ifdef __OPENNN_DEBUG__

    if(data.empty())
    {
       ostringstream buffer;

       buffer << "OpenNN Exception: DataSet class.\n"
              << "Statistics<double> scale_input_minimum_maximum(const size_t&) method.\n"
              << "Data file is not loaded.\n";

       throw logic_error(buffer.str());
    }

    #endif

    const Statistics<double> input_statistics = calculate_input_statistics(input_index);

    scale_input_minimum_maximum(input_statistics, input_index);

    return(input_statistics);
}


/// Calculates the input and target variables statistics.
/// Then it scales the input variables with that values.
/// The method to be used is that in the scaling and unscaling method variable.
/// Finally, it returns the statistics.

Vector< Statistics<double> > DataSet::scale_inputs(const string& scaling_unscaling_method)
{
    switch(get_scaling_unscaling_method(scaling_unscaling_method))
    {
    case NoScaling:
    {
        return(calculate_inputs_statistics());
    }

    case MinimumMaximum:
    {
        return(scale_inputs_minimum_maximum());
    }

    case MeanStandardDeviation:
    {
        return(scale_inputs_mean_standard_deviation());
    }

    case StandardDeviation:
    {
        return(scale_inputs_mean_standard_deviation());
    }

    default:
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: DataSet class\n"
               << "Vector< Statistics<double> > scale_inputs() method.\n"
               << "Unknown scaling and unscaling method.\n";

        throw logic_error(buffer.str());
    }
    }
}


/// Calculates the input and target variables statistics.
/// Then it scales the input variables with that values.
/// The method to be used is that in the scaling and unscaling method variable.

void DataSet::scale_inputs(const string& scaling_unscaling_method, const Vector< Statistics<double> >& inputs_statistics)
{
   switch(get_scaling_unscaling_method(scaling_unscaling_method))
   {
      case NoScaling:
      {
          // Do nothing
      }
      break;

      case MinimumMaximum:
      {
         scale_inputs_minimum_maximum(inputs_statistics);
      }
      break;

      case MeanStandardDeviation:
      {
         scale_inputs_mean_standard_deviation(inputs_statistics);
      }
      break;

      default:
      {
         ostringstream buffer;

         buffer << "OpenNN Exception: DataSet class\n"
                << "void scale_inputs(const string&, const Vector< Statistics<double> >&) method.\n"
                << "Unknown scaling and unscaling method.\n";

         throw logic_error(buffer.str());
      }
   }
}


/// It scales every input variable with the given method.
/// The method to be used is that in the scaling and unscaling method variable.

void DataSet::scale_inputs(const Vector<string>& scaling_unscaling_methods, const Vector< Statistics<double> >& inputs_statistics)
{
    const Vector<size_t> inputs_indices = variables.get_inputs_indices();

   for(size_t i = 0; i < scaling_unscaling_methods.size(); i++)
   {
       switch(get_scaling_unscaling_method(scaling_unscaling_methods[i]))
       {
          case NoScaling:
          {
              // Do nothing
          }
          break;

          case MinimumMaximum:
          {
             scale_input_minimum_maximum(inputs_statistics[i], inputs_indices[i]);
          }
          break;

          case MeanStandardDeviation:
          {
              scale_input_mean_standard_deviation(inputs_statistics[i], inputs_indices[i]);
          }
          break;

          case StandardDeviation:
          {
               scale_input_standard_deviation(inputs_statistics[i], inputs_indices[i]);
          }
          break;

          default:
          {
             ostringstream buffer;

             buffer << "OpenNN Exception: DataSet class\n"
                    << "void scale_inputs(const Vector<string>&, const Vector< Statistics<double> >&) method.\n"
                    << "Unknown scaling and unscaling method: " << scaling_unscaling_methods[i] << "\n";

             throw logic_error(buffer.str());
          }
       }
   }
}


/// Scales the target variables with given mean and standard deviation values.
/// It updates the target variables of the data matrix.
/// @param targets_statistics Vector of statistics structures for all the targets in the data set.
/// The size of that vector must be equal to the number of target variables.

void DataSet::scale_targets_mean_standard_deviation(const Vector< Statistics<double> >& targets_statistics)
{
    const Vector<size_t> targets_indices = variables.get_targets_indices();

    data.scale_columns_mean_standard_deviation(targets_statistics, targets_indices);
}


/// Scales the target variables with the calculated mean and standard deviation values from the data matrix.
/// It updates the target variables of the data matrix.
/// It also returns a vector of statistics structures with the basic statistics of all the variables.

Vector< Statistics<double> > DataSet::scale_targets_mean_standard_deviation()
{
    // Control sentence(if debug)

    #ifdef __OPENNN_DEBUG__

    if(data.empty())
    {
       ostringstream buffer;

       buffer << "OpenNN Exception: DataSet class.\n"
              << "Vector< Statistics<double> > scale_targets_mean_standard_deviation() method.\n"
              << "Data file is not loaded.\n";

       throw logic_error(buffer.str());
    }

    #endif

   const Vector< Statistics<double> > targets_statistics = calculate_targets_statistics();

   scale_targets_mean_standard_deviation(targets_statistics);

   return(targets_statistics);
}


/// Scales the target variables with given minimum and maximum values.
/// It updates the target variables of the data matrix.
/// @param targets_statistics Vector of statistics structures for all the targets in the data set.
/// The size of that vector must be equal to the number of target variables.

void DataSet::scale_targets_minimum_maximum(const Vector< Statistics<double> >& targets_statistics)
{
    // Control sentence(if debug)

    #ifdef __OPENNN_DEBUG__

    if(data.empty())
    {
       ostringstream buffer;

       buffer << "OpenNN Exception: DataSet class.\n"
              << "Vector< Statistics<double> > scale_targets_minimum_maximum() method.\n"
              << "Data file is not loaded.\n";

       throw logic_error(buffer.str());
    }

    #endif

    const Vector<size_t> targets_indices = variables.get_targets_indices();

    data.scale_columns_minimum_maximum(targets_statistics, targets_indices);
}


/// Scales the target variables with the calculated minimum and maximum values from the data matrix.
/// It updates the target variables of the data matrix.
/// It also returns a vector of vectors with the statistics of the input target variables.

Vector< Statistics<double> > DataSet::scale_targets_minimum_maximum()
{
   const Vector< Statistics<double> > targets_statistics = calculate_targets_statistics();

   scale_targets_minimum_maximum(targets_statistics);

   return(targets_statistics);
}


Eigen::MatrixXd DataSet::scale_targets_minimum_maximum_eigen()
{
    const Vector< Statistics<double> > targets_statistics = scale_targets_minimum_maximum();

    const size_t inputs_number = targets_statistics.size();

    Eigen::MatrixXd statistics_eigen(inputs_number, 4);

    for(size_t i = 0; i < inputs_number; i++)
    {
        statistics_eigen(i,0) = targets_statistics[i].minimum;
        statistics_eigen(i,1) = targets_statistics[i].maximum;
        statistics_eigen(i,2) = targets_statistics[i].mean;
        statistics_eigen(i,3) = targets_statistics[i].standard_deviation;
    }

    return statistics_eigen;
}


/// Scales the target variables with the logarithmic scale using the given minimum and maximum values.
/// It updates the target variables of the data matrix.
/// @param targets_statistics Vector of statistics structures for all the targets in the data set.
/// The size of that vector must be equal to the number of target variables.

void DataSet::scale_targets_logarithmic(const Vector< Statistics<double> >& targets_statistics)
{
    // Control sentence(if debug)

    #ifdef __OPENNN_DEBUG__

    if(data.empty())
    {
       ostringstream buffer;

       buffer << "OpenNN Exception: DataSet class.\n"
              << "Vector< Statistics<double> > scale_targets_logarithmic() method.\n"
              << "Data file is not loaded.\n";

       throw logic_error(buffer.str());
    }

    #endif

    const Vector<size_t> targets_indices = variables.get_targets_indices();

    data.scale_columns_logarithmic(targets_statistics, targets_indices);
}


/// Scales the target variables with the logarithmic scale using the calculated minimum and maximum values
/// from the data matrix.
/// It updates the target variables of the data matrix.
/// It also returns a vector of vectors with the statistics of the input target variables.

Vector< Statistics<double> > DataSet::scale_targets_logarithmic()
{
   const Vector< Statistics<double> > targets_statistics = calculate_targets_statistics();

   scale_targets_logarithmic(targets_statistics);

   return(targets_statistics);
}


/// Calculates the input and target variables statistics.
/// Then it scales the target variables with those values.
/// The method to be used is that in the scaling and unscaling method variable.
/// Finally, it returns the statistics.

Vector< Statistics<double> > DataSet::scale_targets(const string& scaling_unscaling_method)
{
    switch(get_scaling_unscaling_method(scaling_unscaling_method))
   {
    case NoUnscaling:
    {
        return(calculate_targets_statistics());
    }

    case MinimumMaximum:
    {
        return(scale_targets_minimum_maximum());
    }

    case Logarithmic:
    {
        return(scale_targets_logarithmic());
    }

    case MeanStandardDeviation:
    {
        return(scale_targets_mean_standard_deviation());
    }

    default:
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: DataSet class\n"
               << "Vector< Statistics<double> > scale_targets(const string&) method.\n"
               << "Unknown scaling and unscaling method.\n";

        throw logic_error(buffer.str());
    }
   }
}


/// It scales the input variables with that values.
/// The method to be used is that in the scaling and unscaling method variable.

void DataSet::scale_targets(const string& scaling_unscaling_method, const Vector< Statistics<double> >& targets_statistics)
{
    switch(get_scaling_unscaling_method(scaling_unscaling_method))
   {
    case NoUnscaling:
    {
        // Do nothing
    }
        break;
    case MinimumMaximum:
    {
        scale_targets_minimum_maximum(targets_statistics);
    }
        break;

    case MeanStandardDeviation:
    {
        scale_targets_mean_standard_deviation(targets_statistics);
    }
        break;

    case Logarithmic:
    {
        scale_targets_logarithmic(targets_statistics);
    }
        break;

    default:
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: DataSet class\n"
               << "void scale_targets(const string&, const Vector< Statistics<double> >&) method.\n"
               << "Unknown scaling and unscaling method.\n";

        throw logic_error(buffer.str());
    }
   }
}


/// Unscales the data matrix with given mean and standard deviation values.
/// It updates the data matrix.
/// @param data_statistics Vector of statistics structures for all the variables in the data set.
/// The size of that vector must be equal to the number of variables.

void DataSet::unscale_data_mean_standard_deviation(const Vector< Statistics<double> >& data_statistics)
{
   data.unscale_mean_standard_deviation(data_statistics);
}


/// Unscales the data matrix with given minimum and maximum values.
/// It updates the data matrix.
/// @param data_statistics Vector of statistics structures for all the variables in the data set.
/// The size of that vector must be equal to the number of variables.

void DataSet::unscale_data_minimum_maximum(const Vector< Statistics<double> >& data_statistics)
{
   data.unscale_minimum_maximum(data_statistics);
}


/// Unscales the input variables with given mean and standard deviation values.
/// It updates the input variables of the data matrix.
/// @param data_statistics Vector of statistics structures for all the variables in the data set.
/// The size of that vector must be equal to the number of variables.

void DataSet::unscale_inputs_mean_standard_deviation(const Vector< Statistics<double> >& data_statistics)
{
    const Vector<size_t> inputs_indices = variables.get_inputs_indices();

    data.unscale_columns_mean_standard_deviation(data_statistics, inputs_indices);
}


/// Unscales the input variables with given minimum and maximum values.
/// It updates the input variables of the data matrix.
/// @param data_statistics Vector of statistics structures for all the data in the data set.
/// The size of that vector must be equal to the number of variables.

void DataSet::unscale_inputs_minimum_maximum(const Vector< Statistics<double> >& data_statistics)
{
    const Vector<size_t> inputs_indices = variables.get_inputs_indices();

    data.unscale_columns_minimum_maximum(data_statistics, inputs_indices);
}


/// Unscales the target variables with given mean and standard deviation values.
/// It updates the target variables of the data matrix.
/// @param data_statistics Vector of statistics structures for all the variables in the data set.
/// The size of that vector must be equal to the number of variables.

void DataSet::unscale_targets_mean_standard_deviation(const Vector< Statistics<double> >& targets_statistics)
{    
    const Vector<size_t> targets_indices = variables.get_targets_indices();

    data.unscale_columns_mean_standard_deviation(targets_statistics, targets_indices);
}


/// Unscales the target variables with given minimum and maximum values.
/// It updates the target variables of the data matrix.
/// @param data_statistics Vector of statistics structures for all the variables.
/// The size of that vector must be equal to the number of variables.

void DataSet::unscale_targets_minimum_maximum(const Vector< Statistics<double> >& targets_statistics)
{
    const Vector<size_t> targets_indices = variables.get_targets_indices();

    data.unscale_columns_minimum_maximum(targets_statistics, targets_indices);
}


/// Initializes the data matrix with a given value.
/// @param new_value Initialization value.

void DataSet::initialize_data(const double& new_value)
{
   data.initialize(new_value);
}


/// Initializes the data matrix with random values chosen from a uniform distribution
/// with given minimum and maximum.

void DataSet::randomize_data_uniform(const double& minimum, const double& maximum)
{
   data.randomize_uniform(minimum, maximum);
}


/// Initializes the data matrix with random values chosen from a normal distribution
/// with given mean and standard deviation.

void DataSet::randomize_data_normal(const double& mean, const double& standard_deviation)
{
   data.randomize_normal(mean, standard_deviation);
}


/// Serializes the data set object into a XML document of the TinyXML library.

tinyxml2::XMLDocument* DataSet::to_XML() const
{
   tinyxml2::XMLDocument* document = new tinyxml2::XMLDocument;

   ostringstream buffer;

   // Data set

   tinyxml2::XMLElement* data_set_element = document->NewElement("DataSet");
   document->InsertFirstChild(data_set_element);

   tinyxml2::XMLElement* element = nullptr;
   tinyxml2::XMLText* text = nullptr;

   // Data file

   tinyxml2::XMLElement* data_file_element = document->NewElement("DataFile");

   data_set_element->InsertFirstChild(data_file_element);

   // File type
   {
       element = document->NewElement("FileType");
       data_file_element->LinkEndChild(element);

       text = document->NewText(write_file_type().c_str());
       element->LinkEndChild(text);
   }

   //First cell
   {
       element = document->NewElement("FirstCell");
       data_file_element->LinkEndChild(element);

       text = document->NewText(write_first_cell().c_str());
       element->LinkEndChild(text);
   }

   //Last cell
   {
       element = document->NewElement("LastCell");
       data_file_element->LinkEndChild(element);

       text = document->NewText(write_last_cell().c_str());
       element->LinkEndChild(text);
   }

   //Sheet number
   {
       element = document->NewElement("SheetNumber");
       data_file_element->LinkEndChild(element);

       buffer.str("");
       buffer << write_sheet_number();

       text = document->NewText(buffer.str().c_str());
       element->LinkEndChild(text);
   }

   // Lags number
   {
       element = document->NewElement("LagsNumber");
       data_file_element->LinkEndChild(element);

       const size_t lags_number = get_lags_number();

       buffer.str("");
       buffer << lags_number;

       text = document->NewText(buffer.str().c_str());
       element->LinkEndChild(text);
   }

   // Steps ahead
   {
       element = document->NewElement("StepsAhead");
       data_file_element->LinkEndChild(element);

       const size_t steps_ahead = get_steps_ahead();

       buffer.str("");
       buffer << steps_ahead;

       text = document->NewText(buffer.str().c_str());
       element->LinkEndChild(text);
   }

   // Time index
   {
       element = document->NewElement("TimeIndex");
       data_file_element->LinkEndChild(element);

       const size_t time_index = get_time_index();

       buffer.str("");
       buffer << time_index;

       text = document->NewText(buffer.str().c_str());
       element->LinkEndChild(text);
   }

   // Header line
   {
      element = document->NewElement("ColumnsName");
      data_file_element->LinkEndChild(element);

      buffer.str("");
      buffer << header_line;

      text = document->NewText(buffer.str().c_str());
      element->LinkEndChild(text);
   }

   // Rows label
   {
      element = document->NewElement("RowsLabel");
      data_file_element->LinkEndChild(element);

      buffer.str("");
      buffer << rows_label;

      text = document->NewText(buffer.str().c_str());
      element->LinkEndChild(text);
   }

   // Separator
   {
      element = document->NewElement("Separator");
      data_file_element->LinkEndChild(element);

      text = document->NewText(write_separator().c_str());
      element->LinkEndChild(text);
   }

   // Missing values label
   {
      element = document->NewElement("MissingValuesLabel");
      data_file_element->LinkEndChild(element);

      text = document->NewText(missing_values_label.c_str());
      element->LinkEndChild(text);
   }

   // Grouping factor
   {
       element = document->NewElement("GroupingFactor");
       data_file_element->LinkEndChild(element);

       const double grouping_factor = get_grouping_factor();

       buffer.str("");
       buffer << grouping_factor;

       text = document->NewText(buffer.str().c_str());
       element->LinkEndChild(text);
   }

   // Data file name
   {
      element = document->NewElement("DataFileName");
      data_file_element->LinkEndChild(element);

      text = document->NewText(data_file_name.c_str());
      element->LinkEndChild(text);
   }

   // Variables
   {
      const tinyxml2::XMLDocument* variables_document = variables.to_XML();

      const tinyxml2::XMLElement* inputs_element = variables_document->FirstChildElement("Variables");

      tinyxml2::XMLNode* node = inputs_element->DeepClone(document);

      data_set_element->InsertEndChild(node);

      delete variables_document;
   }

   // Instances
   {
       const tinyxml2::XMLDocument* instances_document = instances.to_XML();

       const tinyxml2::XMLElement* inputs_element = instances_document->FirstChildElement("Instances");

       tinyxml2::XMLNode* node = inputs_element->DeepClone(document);

       data_set_element->InsertEndChild(node);

       delete instances_document;
   }

   // Missing values
   {
       const tinyxml2::XMLDocument* missing_values_document = missing_values.to_XML();

       const tinyxml2::XMLElement* inputs_element = missing_values_document->FirstChildElement("MissingValues");

       tinyxml2::XMLNode* node = inputs_element->DeepClone(document);

       data_set_element->InsertEndChild(node);

       delete missing_values_document;
   }

   // Display
//   {
//      element = document->NewElement("Display");
//      data_set_element->LinkEndChild(element);

//      buffer.str("");
//      buffer << display;

//      text = document->NewText(buffer.str().c_str());
//      element->LinkEndChild(text);
//   }

   return(document);
}


/// Serializes the data set object into a XML document of the TinyXML library without keep the DOM tree in memory.

void DataSet::write_XML(tinyxml2::XMLPrinter& file_stream) const
{
    ostringstream buffer;

    file_stream.OpenElement("DataSet");

    // Data file

    file_stream.OpenElement("DataFile");

    // File type

    {
        file_stream.OpenElement("FileType");

        file_stream.PushText(write_file_type().c_str());

        file_stream.CloseElement();
    }

    // First cell

    {
        file_stream.OpenElement("FirstCell");

        file_stream.PushText(write_first_cell().c_str());

        file_stream.CloseElement();
    }

    // Last cell

    {
        file_stream.OpenElement("LastCell");

        file_stream.PushText(write_last_cell().c_str());

        file_stream.CloseElement();
    }

    // Sheet number

    {
        file_stream.OpenElement("SheetNumber");

        buffer.str("");
        buffer << write_sheet_number();

        file_stream.PushText(buffer.str().c_str());

        file_stream.CloseElement();
    }

    // Lags number

    {
        file_stream.OpenElement("LagsNumber");

        buffer.str("");
        buffer << get_lags_number();

        file_stream.PushText(buffer.str().c_str());

        file_stream.CloseElement();
    }

    // Steps Ahead

    {
        file_stream.OpenElement("StepsAhead");

        buffer.str("");
        buffer << get_steps_ahead();

        file_stream.PushText(buffer.str().c_str());

        file_stream.CloseElement();
    }

    // Time Index

    {
        file_stream.OpenElement("TimeIndex");

        buffer.str("");
        buffer << get_time_index();

        file_stream.PushText(buffer.str().c_str());

        file_stream.CloseElement();
    }

    // Header line

    {
        file_stream.OpenElement("ColumnsName");

        buffer.str("");
        buffer << header_line;

        file_stream.PushText(buffer.str().c_str());

        file_stream.CloseElement();
    }

    // Rows label

    {
        file_stream.OpenElement("RowsLabel");

        buffer.str("");
        buffer << rows_label;

        file_stream.PushText(buffer.str().c_str());

        file_stream.CloseElement();
    }

    // Separator

    {
        file_stream.OpenElement("Separator");

        file_stream.PushText(write_separator().c_str());

        file_stream.CloseElement();
    }

    // Missing values label

    {
        file_stream.OpenElement("MissingValuesLabel");

        file_stream.PushText(missing_values_label.c_str());

        file_stream.CloseElement();
    }

    // Grouping factor

    {
        file_stream.OpenElement("GroupingFactor");

        buffer.str("");
        buffer << get_grouping_factor();

        file_stream.PushText(buffer.str().c_str());

        file_stream.CloseElement();
    }


    // Data file name

    {
        file_stream.OpenElement("DataFileName");

        file_stream.PushText(data_file_name.c_str());

        file_stream.CloseElement();
    }

    file_stream.CloseElement();

    // Variables

    variables.write_XML(file_stream);

    // Instances

    instances.write_XML(file_stream);

    // Missing values

    missing_values.write_XML(file_stream);


    file_stream.CloseElement();
}


/// Deserializes a TinyXML document into this data set object.
/// @param data_set_document XML document containing the member data.

void DataSet::from_XML(const tinyxml2::XMLDocument& data_set_document)
{
   ostringstream buffer;

   // Data set element

   const tinyxml2::XMLElement* data_set_element = data_set_document.FirstChildElement("DataSet");

   if(!data_set_element)
   {
       buffer << "OpenNN Exception: DataSet class.\n"
              << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
              << "Data set element is nullptr.\n";

       throw logic_error(buffer.str());
   }

    // Data file

    const tinyxml2::XMLElement* data_file_element = data_set_element->FirstChildElement("DataFile");

    if(!data_file_element)
    {
       buffer << "OpenNN Exception: DataSet class.\n"
              << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
              << "Data file element is nullptr.\n";

       throw logic_error(buffer.str());
    }

    // Data file name
    {
       const tinyxml2::XMLElement* data_file_name_element = data_file_element->FirstChildElement("DataFileName");

       if(!data_file_name_element)
       {
           buffer << "OpenNN Exception: DataSet class.\n"
                  << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
                  << "Data file name element is nullptr.\n";

           throw logic_error(buffer.str());
       }

       if(data_file_name_element->GetText())
       {
            const string new_data_file_name = data_file_name_element->GetText();

            set_data_file_name(new_data_file_name);
       }
    }

    // Lags number
    {
        const tinyxml2::XMLElement* lags_number_element = data_file_element->FirstChildElement("LagsNumber");

        if(!lags_number_element)
        {
            buffer << "OpenNN Exception: DataSet class.\n"
                   << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
                   << "Lags number element is nullptr.\n";

            throw logic_error(buffer.str());
        }

        if(lags_number_element->GetText())
        {
             const size_t new_lags_number = static_cast<size_t>(atoi(lags_number_element->GetText()));

             set_lags_number(new_lags_number);
        }
    }

    // Steps ahead
    {
        const tinyxml2::XMLElement* steps_ahead_element = data_file_element->FirstChildElement("StepsAhead");

        if(!steps_ahead_element)
        {
            buffer << "OpenNN Exception: DataSet class.\n"
                   << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
                   << "Steps ahead element is nullptr.\n";

            throw logic_error(buffer.str());
        }

        if(steps_ahead_element->GetText())
        {
             const size_t new_steps_ahead = static_cast<size_t>(atoi(steps_ahead_element->GetText()));

             set_steps_ahead_number(new_steps_ahead);
        }
    }

    // File Type
    {
       const tinyxml2::XMLElement* file_type_element = data_file_element->FirstChildElement("FileType");

       if(!file_type_element)
       {
           buffer << "OpenNN Exception: DataSet class.\n"
                  << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
                  << "File type element is nullptr.\n";

           throw logic_error(buffer.str());
       }

       if(file_type_element->GetText())
       {
            const string new_file_type = file_type_element->GetText();

            set_file_type(new_file_type);
       }
    }

    // First Cell
    {
       const tinyxml2::XMLElement* first_cell_element = data_file_element->FirstChildElement("FirstCell");

       if(!first_cell_element)
       {
           buffer << "OpenNN Exception: DataSet class.\n"
                  << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
                  << "First Cell element is nullptr.\n";

           throw logic_error(buffer.str());
       }

       if(first_cell_element->GetText())
       {
            const string new_first_cell = first_cell_element->GetText();

            first_cell = new_first_cell;
       }
    }

    // Last Cell
    {
       const tinyxml2::XMLElement* last_cell_element = data_file_element->FirstChildElement("LastCell");

       if(!last_cell_element)
       {
           buffer << "OpenNN Exception: DataSet class.\n"
                  << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
                  << "Last Cell element is nullptr.\n";

           throw logic_error(buffer.str());
       }

       if(last_cell_element->GetText())
       {
            const string new_last_cell = last_cell_element->GetText();

            last_cell = new_last_cell;
       }
    }

    // Sheet number
    {
       const tinyxml2::XMLElement* sheet_number_element = data_file_element->FirstChildElement("SheetNumber");

       if(!sheet_number_element)
       {
           buffer << "OpenNN Exception: DataSet class.\n"
                  << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
                  << "Sheet Number element is nullptr.\n";

           throw logic_error(buffer.str());
       }

       if(sheet_number_element->GetText())
       {
            const size_t new_sheet_number = static_cast<size_t>(atoi(sheet_number_element->GetText()));

            sheet_number = new_sheet_number;
       }
    }

      // Header line
      {
         const tinyxml2::XMLElement* header_element = data_file_element->FirstChildElement("ColumnsName");

         if(header_element)
         {
            const string new_header_string = header_element->GetText();

            try
            {
               set_header_line(new_header_string != "0");
            }
            catch(const logic_error& e)
            {
               cerr << e.what() << endl;
            }
         }
      }

        // Rows label
        {
           const tinyxml2::XMLElement* rows_label_element = data_file_element->FirstChildElement("RowsLabel");

           if(rows_label_element)
           {
              const string new_rows_label_string = rows_label_element->GetText();

              try
              {
                 set_rows_label(new_rows_label_string != "0");
              }
              catch(const logic_error& e)
              {
                 cerr << e.what() << endl;
              }
           }
        }

      // Separator
    {
      const tinyxml2::XMLElement* separator_element = data_file_element->FirstChildElement("Separator");

      if(separator_element)
     {
          if(separator_element->GetText())
          {
            const string new_separator = separator_element->GetText();

            set_separator(new_separator);
          }
          else
          {
              set_separator("Space");
          }
      }
      else
      {
          set_separator("Space");
      }
    }

    // Missing values label
    {
         const tinyxml2::XMLElement* missing_values_label_element = data_file_element->FirstChildElement("MissingValuesLabel");

         if(missing_values_label_element)
         {
             if(missing_values_label_element->GetText())
             {
                const string new_missing_values_label = missing_values_label_element->GetText();

                set_missing_values_label(new_missing_values_label);
             }
         }
    }


    // Grouping factor
    {
         const tinyxml2::XMLElement* grouping_factor_element = data_file_element->FirstChildElement("GroupingFactor");

         if(grouping_factor_element)
         {
             if(grouping_factor_element->GetText())
             {
                const double new_grouping_factor = atof(grouping_factor_element->GetText());

                set_grouping_factor(new_grouping_factor);
             }
         }
    }

    // Variables
    {
       const tinyxml2::XMLElement* variables_element = data_set_element->FirstChildElement("Variables");

       if(variables_element)
       {
           tinyxml2::XMLDocument variables_document;
           tinyxml2::XMLNode* element_clone;

           element_clone = variables_element->DeepClone(&variables_document);

           variables_document.InsertFirstChild(element_clone);

           variables.from_XML(variables_document);
       }
    }

    // Instances
    {
       const tinyxml2::XMLElement* instances_element = data_set_element->FirstChildElement("Instances");

       if(instances_element)
       {
           tinyxml2::XMLDocument instances_document;
           tinyxml2::XMLNode* element_clone;

           element_clone = instances_element->DeepClone(&instances_document);

           instances_document.InsertFirstChild(element_clone);

           instances.from_XML(instances_document);
       }
    }

    // Missing values
    {
       const tinyxml2::XMLElement* missing_values_element = data_set_element->FirstChildElement("MissingValues");

       if(missing_values_element)
       {
           tinyxml2::XMLDocument missing_values_document;
           tinyxml2::XMLNode* element_clone;

           element_clone = missing_values_element->DeepClone(&missing_values_document);

           missing_values_document.InsertFirstChild(element_clone);

           missing_values.from_XML(missing_values_document);
       }
    }

   // Display
   {
      const tinyxml2::XMLElement* display_element = data_set_element->FirstChildElement("Display");

      if(display_element)
      {
         const string new_display_string = display_element->GetText();

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

}


/// Returns a string representation of the current data set object.

string DataSet::object_to_string() const
{
   ostringstream buffer;

   buffer << "Data set object\n"
          << "Data file name: " << data_file_name << "\n"
          << "Header line: " << header_line << "\n"
          << "Separator: " << separator << "\n"
          << "Missing values label: " << missing_values_label << "\n"
          << "Data:\n" << data << "\n"
          << "Display: " << display << "\n"
          << variables.object_to_string()
          << instances.object_to_string()
          << missing_values.object_to_string();

   return(buffer.str());
}


/// Prints to the screen in text format the members of the data set object.

void DataSet::print() const
{
   if(display)
   {
      cout << object_to_string();
   }
}


/// Prints to the screen in text format the main numbers from the data set object.

void DataSet::print_summary() const
{
    if(display)
    {
        const size_t variables_number = variables.get_variables_number();
        const size_t instances_number = instances.get_instances_number();
        const size_t missing_values_number = missing_values.get_missing_values_number();

       cout << "Data set object summary:\n"
                 << "Number of variables: " << variables_number << "\n"
                 << "Number of instances: " << instances_number << "\n"
                 << "Number of missing values: " << missing_values_number << endl;
    }
}


/// Saves the members of a data set object to a XML-type file in an XML-type format.
/// @param file_name Name of data set XML-type file.

void DataSet::save(const string& file_name) const
{
   tinyxml2::XMLDocument* document = to_XML();

   document->SaveFile(file_name.c_str());

   delete document;
}

Vector<double*> DataSet::host_to_device(const Vector<size_t>& batch_indices) const
{
    Vector<double*> batch_pointers(2);

#ifdef __OPENNN_CUDA__

    const Matrix<double> inputs_matrix = get_inputs(batch_indices);
    const double* input_data = inputs_matrix.data();
    const size_t input_rows = inputs_matrix.get_rows_number();
    const size_t input_columns = inputs_matrix.get_columns_number();

    const Matrix<double> targets_matrix = get_targets(batch_indices);
    const double* target_data = targets_matrix.data();
    const size_t target_rows = targets_matrix.get_rows_number();
    const size_t target_columns = targets_matrix.get_columns_number();

    mallocCUDA(&batch_pointers[0], input_rows*input_columns*sizeof(double));
    mallocCUDA(&batch_pointers[1], target_rows*target_columns*sizeof(double));

    memcpyCUDA(batch_pointers[0], input_data, input_rows*input_columns*sizeof(double));
    memcpyCUDA(batch_pointers[1], target_data, target_rows*target_columns*sizeof(double));

#endif

    return batch_pointers;
}


/// Loads the members of a data set object from a XML-type file:
/// <ul>
/// <li> Instances number.
/// <li> Training instances number.
/// <li> Training instances indices.
/// <li> Selection instances number.
/// <li> Selection instances indices.
/// <li> Testing instances number.
/// <li> Testing instances indices.
/// <li> Input variables number.
/// <li> Input variables indices.
/// <li> Target variables number.
/// <li> Target variables indices.
/// <li> Input variables name.
/// <li> Target variables name.
/// <li> Input variables description.
/// <li> Target variables description.
/// <li> Display.
/// <li> Data.
/// </ul>
/// Please mind about the file format. This is specified in the User's Guide.
/// @param file_name Name of data set XML-type file.

void DataSet::load(const string& file_name)
{
   tinyxml2::XMLDocument document;

   if(document.LoadFile(file_name.c_str()))
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: DataSet class.\n"
             << "void load(const string&) method.\n"
             << "Cannot load XML file " << file_name << ".\n";

      throw logic_error(buffer.str());
   }

   from_XML(document);
}


/// Prints to the sceen the values of the data matrix.

void DataSet::print_data() const
{
   if(display)
   {
      cout << data << endl;
   }
}


/// Prints to the sceen a preview of the data matrix,
/// i.e., the first, second and last instances

void DataSet::print_data_preview() const
{
   if(display)
   {
       const size_t instances_number = instances.get_instances_number();

       if(instances_number > 0)
       {
          const Vector<double> first_instance = data.get_row(0);

          cout << "First instance:\n"
                    << first_instance << endl;
       }

       if(instances_number > 1)
       {
          const Vector<double> second_instance = data.get_row(1);

          cout << "Second instance:\n"
                    << second_instance << endl;
       }

       if(instances_number > 2)
       {
          const Vector<double> last_instance = data.get_row(instances_number-1);

          cout << "Instance " << instances_number << ":\n"
                    << last_instance << endl;
       }
    }
}


/// Saves to the data file the values of the data matrix.

void DataSet::save_data() const
{
   ofstream file(data_file_name.c_str());

   if(!file.is_open())
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: DataSet class.\n"
             << "void save_data() const method.\n"
             << "Cannot open data file: " << data_file_name << "\n";

      throw logic_error(buffer.str());
   }

   const char separator_char = get_separator_char();

   if(header_line)
   {
       const Vector<string> variables_name = variables.get_names();

       file << variables_name.vector_to_string(separator_char) << endl;
   }

   // Write data

   const size_t rows_number = data.get_rows_number();
   const size_t columns_number = data.get_columns_number();

   for(size_t i = 0; i < rows_number; i++)
   {
       for(size_t j = 0; j < columns_number; j++)
       {
           if(fabs(data(i,j) - -99.9) < numeric_limits<double>::epsilon())
           {
               file << missing_values_label;
           }
           else
           {
               file << data(i,j);
           }

           if(j != columns_number-1)
           {
               file << separator_char;
           }
       }

      file << endl;
   }

   // Close file

   file.close();
}


/// Returns the index of a variable when reading the data file.
/// @param nominal_labels Values of all nominal variables in the data file.
/// @param column_index Index of column.

size_t DataSet::get_column_index(const Vector< Vector<string> >& nominal_labels, const size_t column_index) const
{
    size_t variable_index = 0;

    for(size_t i = 0; i < column_index; i++)
    {
        if(nominal_labels[i].size() <= 2)
        {
            variable_index++;
        }
        else
        {
            variable_index += nominal_labels[i].size();
        }
    }

    return variable_index;
}


/// Verifies that a given line in the data file contains the separator characer.
/// If the line does not contain the separator, this method throws an exception.
/// @param line Data file line.

void DataSet::check_separator(const string& line) const
{
    if(line.empty())
    {
        return;
    }

//    const char separator_char = get_separator_char();

//    if(line.find(separator_string) == string::npos)
//    {
//        ostringstream buffer;

//        buffer << "OpenNN Exception: DataSet class.\n"
//               << "void check_separator(const string&) method.\n"
//               << "Separator '" << write_separator() << "' not found in data file " << data_file_name << ".\n";

//        throw logic_error(buffer.str());
//    }
}


/// Returns the number of tokens in the first line of the data file.
/// That will be interpreted as the number of columns in the data file.

size_t DataSet::count_data_file_columns_number() const
{
    ifstream file(data_file_name.c_str());

    string line;

    size_t columns_number = 0;

    while(file.good())
    {
        getline(file, line);

        if(separator != Tab)
        {
            replace(line.begin(), line.end(), '\t', ' ');
        }

        trim(line);

        if(line.empty())
        {
            continue;
        }

        check_separator(line);

        columns_number = count_tokens(line);

        break;
    }

    file.close();

    return columns_number;

}


/// Verifies that the data file has a header line.
/// All elements in a header line must be strings.
/// This method can change the value of the header line member.
/// It throws an exception if some inconsistencies are found.

void DataSet::check_header_line()
{
    ifstream file(data_file_name.c_str());

    string line;
    Vector<string> tokens;

    while(file.good())
    {
        getline(file, line);

        if(separator != Tab)
        {
            replace(line.begin(), line.end(), '\t', ' ');
        }

        trim(line);

        if(line.empty())
        {
            continue;
        }

        break;
    }

    file.close();

    check_separator(line);

    tokens = get_tokens(line);

    if(header_line && is_not_numeric(tokens))
    {
        return;
    }
    if(header_line && is_numeric(tokens))
    {
        if(display)
        {
            cout << "OpenNN Warning: DataSet class.\n"
                      << "void check_header_line() method.\n"
                      << "First line of data file interpreted as not header.\n";
        }

        header_line = false;
    }
    else if(header_line && is_mixed(tokens))
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: DataSet class.\n"
               << "void check_header_line() method.\n"
               << "Header line contains numeric values: \n"
               << line << "\n";

        throw logic_error(buffer.str());
    }
    else if(!header_line && is_not_numeric(tokens))
    {
        if(display)
        {
            cout << "OpenNN Warning: DataSet class.\n"
                      << "void check_header_line() method.\n"
                      << "First line of data file interpreted as header.\n";
        }

        header_line = true;
    }
}


/// Returns the name of the columns in the data set as a list of strings.

Vector<string> DataSet::read_header_line() const
{
    Vector<string> header_line;

    string line;

    ifstream file(data_file_name.c_str());

    // First line

    while(file.good())
    {
        getline(file, line);

        if(separator != Tab)
        {
            replace(line.begin(), line.end(), '\t', ' ');
        }

        trim(line);

        if(line.empty())
        {
            continue;
        }

        check_separator(line);

        header_line = get_tokens(line);

        break;
    }

    file.close();

    return header_line;
}


/// Sets the values of a single instance in the data matrix from a line in the data file.
/// @param line Data file line.
/// @param nominal_labels Values of all nominal variables in the data file.
/// @param instance_index Index of instance.

void DataSet::read_instance(const string& line, const Vector< Vector<string> >& nominal_labels, const size_t& instance_index)
{
    // Control sentence(if debug)

    #ifdef __OPENNN_DEBUG__

    const size_t instances_number = instances.get_instances_number();

    if(instance_index >= instances_number)
    {
       ostringstream buffer;

       buffer << "OpenNN Exception: DataSet class.\n"
              << "void read_instance(const string&, const Vector< Vector<string> >&, const size_t&) method.\n"
              << "Index of instance(" << instance_index << ") must be less than number of instances(" << instances_number << ").\n";

       throw logic_error(buffer.str());
    }

    #endif

    const Vector<string> tokens = get_tokens(line);

    #ifdef __OPENNN_DEBUG__

    if(tokens.size() != nominal_labels.size())
    {
       ostringstream buffer;

       buffer << "OpenNN Exception: DataSet class.\n"
              << "void read_instance(const string&, const Vector< Vector<string> >&, const size_t&) method.\n"
              << "Size of tokens (" << tokens.size() << ") must be equal to size of names (" << nominal_labels.size() << ").\n";

       throw logic_error(buffer.str());
    }

    #endif

    size_t column_index;

    for(size_t j = 0; j < tokens.size(); j++)
    {
        column_index = get_column_index(nominal_labels, j);

        if(nominal_labels[j].size() == 0) // Numeric variable
        {
            if(tokens[j] != missing_values_label) // No missing values
            {
                data(instance_index, column_index) = atof(tokens[j].c_str());
            }
            else // Missing values
            {
                data(instance_index, column_index) = -99.9;

                missing_values.append(instance_index, column_index);
            }
        }

        else if(nominal_labels[j].size() == 2) // Binary variable
        {
            if(tokens[j] != missing_values_label) // No missing values
            {
                if(tokens[j] == "false" || tokens[j] == "False"||  tokens[j] == "FALSE" || tokens[j] == "F"
                || tokens[j] == "negative"|| tokens[j] == "Negative"|| tokens[j] == "NEGATIVE")
                {
                    data(instance_index, column_index) = 0.0;
                }
                else if(tokens[j] == "true" || tokens[j] == "True"||  tokens[j] == "TRUE" || tokens[j] == "T"
                || tokens[j] == "positive"|| tokens[j] == "Positive"|| tokens[j] == "POSITIVE")
                {
                    data(instance_index, column_index) = 1.0;
                }
                else if(tokens[j] == nominal_labels[j][0])
                {
                    data(instance_index, column_index) = 0.0;
                }
                else if(tokens[j] == nominal_labels[j][1])
                {
                    data(instance_index, column_index) = 1.0;
                }
                else
                {
                    ostringstream buffer;

                    buffer << "OpenNN Exception: DataSet class.\n"
                           << "void read_instance(const string&, const Vector< Vector<string> >&, const size_t&) method.\n"
                           << "Unknown token binary value.\n";

                    throw logic_error(buffer.str());
                }
            }
            else // Missing values
            {
                data(instance_index, column_index) = -99.9;

                missing_values.append(instance_index, column_index);
            }
        }

        else // Nominal variable
        {
            if(tokens[j] != missing_values_label)
            {
                for(size_t k = 0; k < nominal_labels[j].size(); k++)
                {
                    if(tokens[j] == nominal_labels[j][k])
                    {
                        data(instance_index, column_index+k) = 1.0;
                    }
                    else
                    {
                        data(instance_index, column_index+k) = 0.0;
                    }
                }
            }
            else // Missing values
            {
                for(size_t k = 0; k < nominal_labels[j].size(); k++)
                {
                    data(instance_index, column_index+k) = -99.9;

                    missing_values.append(instance_index, column_index+k);
                }
            }
        }
    }
}


/// Performs a first data file read in which the format is checked,
/// and the numbers of variables, instances and missing values are set.

Vector< Vector<string> > DataSet::set_from_data_file()
{
    const size_t columns_number = count_data_file_columns_number();

    Vector< Vector<string> > nominal_labels(columns_number);

    string line;
    Vector<string> tokens;

    check_header_line();

    int instances_count;

    if(header_line)
    {
        instances_count = -1;
    }
    else
    {
        instances_count = 0;
    }

    ifstream file(data_file_name.c_str());

    // Rest of lines

    while(file.good())
    {
        getline(file, line);

        if(separator != Tab)
        {
            replace(line.begin(), line.end(), '\t', ' ');
        }

        trim(line);

        if(line.empty())
        {
            continue;
        }

        if(header_line && instances_count == -1)
        {
            instances_count = 0;

            continue;
        }

        check_separator(line);

        tokens = get_tokens(line);
// caixa
  /*      if(tokens.size() != columns_number)
        {
            ostringstream buffer;

            buffer << "OpenNN Exception: DataSet class.\n"
                   << "Vector< Vector<string> > DataSet::set_from_data_file().\n"
                   << "Row " << instances_count << ": Size of tokens (" << tokens.size() << ") is not equal to "
                   << "number of columns (" << columns_number << ").\n";

            throw logic_error(buffer.str());
        }
*/
        instances_count++;

        /*
        if(tokens.size() == columns_number)
        {
            instances_count++;
            for(size_t j = 0; j < columns_number; j++)
            {
                numeric = is_numeric(tokens[j]);

                if(!numeric
                && tokens[j] != missing_values_label
                && !nominal_labels[j].contains(tokens[j]))
                {
                    nominal_labels[j].push_back(tokens[j]);
                }
            }
        }*/


    }

    file.close();

    size_t variables_count = 0;

    for(size_t i = 0; i < columns_number; i++)
    {
        if(nominal_labels[i].size() == 0 || nominal_labels[i].size() == 2)
        {
            variables_count++;
        }
        else
        {
            variables_count += nominal_labels[i].size();
        }
    }

    // Fix label case

    for(size_t i = 0; i < columns_number; i++)
    {
        if(nominal_labels[i].size() == static_cast<size_t>(instances_count))
        {
            ostringstream buffer;

            buffer << "OpenNN Exception: DataSet class.\n"
                   << "Vector< Vector<string> > DataSet::set_from_data_file().\n"
                   << "Column " << i << ": All elements are nominal and different. It contains meaningless data.\n";

            throw logic_error(buffer.str());
        }
    }

    // Set instances and variables number

    if(instances_count == 0 || variables_count == 0)
    {
        set();

        return(nominal_labels);
    }

    data.set(static_cast<size_t>(instances_count), variables_count);

    if(variables.get_variables_number() != variables_count)
    {
        variables.set(variables_count);

        if(nominal_labels[columns_number-1].size() > 2)
        {
            for(size_t i = variables_count-1; i >= variables_count - nominal_labels[columns_number-1].size(); i--)
            {
                variables.set_use(i, Variables::Target);
            }
        }
    }

    if(instances.get_instances_number() != static_cast<size_t>(instances_count))
    {
        instances.set(static_cast<size_t>(instances_count));
    }

    missing_values.set(instances.get_instances_number(), variables.get_variables_number());

    return(nominal_labels);
}


/// Performs a second data file read in which the data is set.

void DataSet::read_from_data_file(const Vector< Vector<string> >& nominal_labels)
{
    ifstream file(data_file_name.c_str());

    file.clear();
    file.seekg(0, ios::beg);

    string line;

    if(header_line)
    {
        while(file.good())
        {
            getline(file, line);

            if(separator != Tab)
            {
                replace(line.begin(), line.end(), '\t', ' ');
            }

            trim(line);

            if(line.empty())
            {
                continue;
            }

            break;
        }
    }

    size_t i = 0;

    while(file.good())
    {
        getline(file, line);

        if(separator != Tab)
        {
            replace(line.begin(), line.end(), '\t', ' ');
        }

        const Vector<string> tokens = get_tokens(line);

        trim(line);

        if(line.empty())
        {
            continue;
        }

        // #pragma omp task
/*
        if(tokens.size() == nominal_labels.size())
        {
            read_instance(line, nominal_labels, i);
            i++;
        }*/
// caixa
        read_instance(line, nominal_labels, i);
        i++;

    }

    file.close();
}


/// Returns a vector with the names getd for time series prediction, according to the number of lags.
/// @todo

Vector<string> DataSet::get_time_series_names(const Vector<string>&) const
{
    Vector<string> time_series_prediction_names;
/*
    Vector< Vector<string> > new_names((1+columns_number)*lags_number);

    for(size_t i = 0; i < 1+lags_number; i++)
    {
        for(size_t j = 0; j < names.size(); j++)
        {
            new_names[i+j] = names[j];

            if(i != lags_number)
            {
                for(size_t k = 0; k < names[j].size();k++)
                {
                    new_names[i+j][k].append("_lag_").append(string::from_size_t(lags_number-i).c_str());
                }
            }
        }
    }
*/
    return(time_series_prediction_names);
}


/// Returns a vector with the names getd for association.
/// @todo

Vector<string> DataSet::get_association_names(const Vector<string>&) const
{
    Vector<string> association_names;

    return(association_names);
}


/// Arranges an input-target matrix from a time series matrix, according to the number of lags.

void DataSet::convert_time_series()
{
    if(lags_number == 0)
    {
        return;
    }

    time_series_data = data;

    data.convert_time_series(time_index, lags_number, steps_ahead);

    variables.convert_time_series(lags_number, steps_ahead, time_index);

    instances.convert_time_series(lags_number, steps_ahead);

    missing_values.convert_time_series(data);
}


/// Arranges the data set for association.
/// @todo

void DataSet::convert_association()
{
    data.convert_association();

    variables.convert_association();

    missing_values.convert_association();
}


/// This method loads the data file.

void DataSet::load_data()
{
    if(data_file_name.empty())
    {
       ostringstream buffer;

       buffer << "OpenNN Exception: DataSet class.\n"
              << "void load_data() method.\n"
              << "Data file name has not been set.\n";

       throw logic_error(buffer.str());
    }

    ifstream file(data_file_name.c_str());

    if(!file.is_open())
    {
       ostringstream buffer;

       buffer << "OpenNN Exception: DataSet class.\n"
              << "void load_data() method.\n"
              << "Cannot open data file: " << data_file_name << "\n";

       throw logic_error(buffer.str());
    }

    file.close();

    const Vector< Vector<string> > nominal_labels = set_from_data_file();

    read_from_data_file(nominal_labels);

    // Variables name

    Vector<string> columns_name;

    if(header_line)
    {
        columns_name = read_header_line();
    }
    else
    {
        for(unsigned i = 0; i < nominal_labels.size(); i++)
        {
            ostringstream buffer;

            buffer << "variable_" << i;

            columns_name.push_back(buffer.str());
        }
    }

    variables.set_names(columns_name, nominal_labels);

    // Angular variables

    if(!angular_variables.empty())
    {
        convert_angular_variables();
    }

    // Time series

    if(lags_number != 0)
    {
        convert_time_series();
    }

    // Association

    if(association)
    {
        convert_association();
    }
}


/// This method loads the data from a binary data file.

void DataSet::load_data_binary()
{
    ifstream file;

    file.open(data_file_name.c_str(), ios::binary);

    if(!file.is_open())
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: DataSet class.\n"
               << "void load_data_binary() method.\n"
               << "Cannot open data file: " << data_file_name << "\n";

        throw logic_error(buffer.str());
    }

    streamsize size = sizeof(size_t);

    size_t variables_number;
    size_t instances_number;

    file.read(reinterpret_cast<char*>(&variables_number), size);
    file.read(reinterpret_cast<char*>(&instances_number), size);

    size = sizeof(double);

    double value;

    data.set(instances_number, variables_number);

    for(size_t i = 0; i < variables_number*instances_number; i++)
    {
        file.read(reinterpret_cast<char*>(&value), size);

        data[i] = value;
    }

    file.close();
}


/// This method loads data from a binary data file for time series prediction methods.

void DataSet::load_time_series_data_binary()
{
    ifstream file;

    file.open(data_file_name.c_str(), ios::binary);

    if(!file.is_open())
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: DataSet class.\n"
               << "void load_time_series_data_binary() method.\n"
               << "Cannot open data file: " << data_file_name << "\n";

        throw logic_error(buffer.str());
    }

    streamsize size = sizeof(size_t);

    size_t variables_number;
    size_t instances_number;

    file.read(reinterpret_cast<char*>(&variables_number), size);
    file.read(reinterpret_cast<char*>(&instances_number), size);

    size = sizeof(double);

    double value;

    data.set(instances_number, variables_number);

    for(size_t i = 0; i < variables_number*instances_number; i++)
    {
        file.read(reinterpret_cast<char*>(&value), size);

        data[i] = value;
    }

//    if(get_missing_values().has_missing_values())
//    {
//        scrub_missing_values();
//    }

    size = sizeof(size_t);

    size_t time_series_instances_number;
    size_t time_series_variables_number;

    file.read(reinterpret_cast<char*>(&time_series_instances_number), size);
    file.read(reinterpret_cast<char*>(&time_series_variables_number), size);

    time_series_variables_number = time_series_variables_number + 1;

    time_series_data.set(time_series_instances_number, time_series_variables_number);

    size = sizeof(double);

    for(size_t i = 0; i < time_series_instances_number*time_series_variables_number; i++)
    {
        file.read(reinterpret_cast<char*>(&value), size);

        time_series_data[i] = value;
    }

    file.close();
}


/// @todo This method is not implemented.
/*
void DataSet::load_time_series_data()
{
    if(lags_number <= 0)
    {
       ostringstream buffer;

       buffer << "OpenNN Exception: DataSet class.\n"
              << "void load_time_series_data() const method.\n"
              << "Number of lags(" << lags_number << ") must be greater than zero.\n";

       throw logic_error(buffer.str());
    }


    if(header)
    {
//        Vector<string> columns_name;

//        variables.set_names(names);
    }


    const Matrix<double> time_series_data(data_file_name);

    const size_t rows_number = time_series_data.get_rows_number();
    const size_t columns_number = time_series_data.get_columns_number();

    const size_t instances_number = rows_number - lags_number;
    const size_t variables_number = columns_number*(1 + lags_number);

    set(variables_number, instances_number);

    Vector<double> row(rows_number);

    for(size_t i = 0; i < instances_number; i++)
    {
        row = time_series_data.get_row(i);

        for(size_t j = 1; j <= lags_number; j++)
        {
            row = row.assemble(time_series_data.get_row(i+j));
        }

        data.set_row(i, row);
    }

    // Variables

    Vector<Variables::Use> uses(variables_number);

    fill(uses.begin(), uses.begin()+lags_number*variables_number/(lags_number+1)-1, Variables::Use::Input);
    fill(uses.begin()+lags_number*variables_number/(lags_number+1), uses.end(), Variables::Use::Target);

    variables.set_uses(uses);

}
*/


/// Returns a vector containing the number of instances of each class in the data set.
/// If the number of target variables is one then the number of classes is two.
/// If the number of target variables is greater than one then the number of classes is equal to the number
/// of target variables.

Vector<size_t> DataSet::calculate_target_distribution() const
{
   // Control sentence(if debug)

   const size_t instances_number = instances.get_instances_number();
   const size_t targets_number = variables.get_targets_number();
   const Vector<size_t> targets_indices = variables.get_targets_indices();

   Vector<size_t> class_distribution;

   if(targets_number == 1) // Two classes
   {
      class_distribution.set(2, 0);

      size_t target_index = targets_indices[0];

      size_t positives = 0;
      size_t negatives = 0;

      const Vector<size_t> target_missing_indices = missing_values.get_missing_indices(target_index);

       Vector<size_t> used_instances = instances.get_used_indices();

       used_instances = used_instances.get_difference(target_missing_indices);

       const size_t used_instances_number = used_instances.size();

//#pragma omp parallel for reduction(+ : positives, negatives)

      for(int instance_index = 0; instance_index < static_cast<int>(used_instances_number); instance_index++)
      {
          if(data(used_instances[static_cast<size_t>(instance_index)],target_index) < 0.5)
          {
              negatives++;
          }
          else
          {
              positives++;
          }
      }

      class_distribution[0] = negatives;
      class_distribution[1] = positives;
   }
   else // More than two classes
   {
      class_distribution.set(targets_number, 0);

      for(size_t i = 0; i < instances_number; i++)
      {
          if(instances.get_use(i) != Instances::Unused)
          {
             for(size_t j = 0; j < targets_number; j++)
             {
                 if(fabs(data(i,targets_indices[j]) - -99.9) < numeric_limits<double>::epsilon())
                 {
                    continue;
                 }

                if(data(i,targets_indices[j]) > 0.5)
                {
                   class_distribution[j]++;
                }
             }
          }
      }
   }

   // Check data consistency

//   const size_t used_instances_number = instances.get_used_instances_number();

//   if(class_distribution.calculate_sum() != used_instances_number)
//   {
//      ostringstream buffer;

//      buffer << "OpenNN Exception: DataSet class.\n"
//             << "Vector<size_t> calculate_target_distribution() const method.\n"
//             << "Sum of class distributions(" << class_distribution << ") is not equal to "
//             << "number of used instances(" << used_instances_number << ")." << endl;

//      throw logic_error(buffer.str());
//   }

   return(class_distribution);
}


/// Returns a normalized distance between each instance and the mean instance.
/// The size of this vector is the number of instances.

Vector<double> DataSet::calculate_distances() const
{
    const Matrix<double> data_statistics_matrix = calculate_data_statistics_matrix();

    const Vector<double> means = data_statistics_matrix.get_column(2);
    const Vector<double> standard_deviations = data_statistics_matrix.get_column(3);

    const size_t instances_number = instances.get_instances_number();
    Vector<double> distances(instances_number);

    const size_t variables_number = variables.get_variables_number();
    Vector<double> instance(variables_number);

    int i = 0;

    #pragma omp parallel for private(i, instance)

    for(i = 0; i < static_cast<int>(instances_number); i++)
    {
        instance = data.get_row(static_cast<size_t>(i));

        distances[static_cast<size_t>(i)] = (instance-means/standard_deviations).calculate_L2_norm();
    }

    return(distances);
}


/// This method balances the targets ditribution of a data set with only one target variable by unusing
/// instances whose target variable belongs to the most populated target class.
/// It returns a vector with the indices of the instances set unused.
/// @param percentage Percentage of instances to be unused.

Vector<size_t> DataSet::balance_binary_targets_distribution(const double& percentage)
{
    Vector<size_t> unused_instances;

    const size_t instances_number = instances.get_used_instances_number();

    const Vector<size_t> target_class_distribution = calculate_target_distribution();

    const Vector<size_t> maximal_indices = target_class_distribution.calculate_maximal_indices(2);

    const size_t maximal_target_class_index = maximal_indices[0];
    const size_t minimal_target_class_index = maximal_indices[1];

    size_t total_unbalanced_instances_number = static_cast<size_t>((percentage/100.0)*(target_class_distribution[maximal_target_class_index] - target_class_distribution[minimal_target_class_index]));

    size_t actual_unused_instances_number;

    size_t unbalanced_instances_number = total_unbalanced_instances_number/10;

    Vector<size_t> actual_unused_instances;

//    cout << "Instances to unuse: " << total_unbalanced_instances_number << endl;

    while(total_unbalanced_instances_number != 0)
    {
        if(total_unbalanced_instances_number < instances_number/10)
        {
           unbalanced_instances_number = total_unbalanced_instances_number;
        }
        else if(total_unbalanced_instances_number > 0 && unbalanced_instances_number < 1)
        {
            unbalanced_instances_number = total_unbalanced_instances_number;
        }

        actual_unused_instances = unuse_most_populated_target(unbalanced_instances_number);

        actual_unused_instances_number = actual_unused_instances.size();

        unused_instances = unused_instances.assemble(actual_unused_instances);

        total_unbalanced_instances_number = total_unbalanced_instances_number - actual_unused_instances_number;

        actual_unused_instances.clear();
    }

    return(unused_instances);
}


/// This method balances the targets ditribution of a data set with more than one target variable by unusing
/// instances whose target variable belongs to the most populated target class.
/// It returns a vector with the indices of the instances set unused.

Vector<size_t> DataSet::balance_multiple_targets_distribution()
{
    Vector<size_t> unused_instances;

    const size_t bins_number = 10;

    const Vector<size_t> target_class_distribution = calculate_target_distribution();

    const size_t targets_number = variables.get_targets_number();

    const Vector<size_t> inputs_variables_indices = variables.get_inputs_indices();
    const Vector<size_t> targets_variables_indices = variables.get_targets_indices();

    const Vector<size_t> maximal_target_class_indices = target_class_distribution.calculate_maximal_indices(targets_number);

    const size_t minimal_target_class_index = maximal_target_class_indices[targets_number - 1];

    // Target class differences

    Vector<size_t> target_class_differences(targets_number);

    for(size_t i = 0; i < targets_number; i++)
    {
        target_class_differences[i] = target_class_distribution[i] - target_class_distribution[minimal_target_class_index];
    }

    Vector<double> instance;

    size_t count_instances = 0;

    size_t unbalanced_instances_number;

    size_t instances_number;
    Vector<size_t> instances_indices;

    Vector< Histogram<double> > data_histograms;

    Matrix<size_t> total_frequencies;
    Vector<size_t> instance_frequencies;

    size_t maximal_difference_index;
    size_t instance_index;
    size_t instance_target_index;

    Vector<size_t> unbalanced_instances_indices;

    while(!target_class_differences.is_in(0, 0))
    {
        unbalanced_instances_indices.clear();
        instances_indices.clear();

        instances_indices = instances.get_used_indices();

        instances_number = instances_indices.size();


        maximal_difference_index = target_class_differences.calculate_maximal_index();

        unbalanced_instances_number = static_cast<size_t>(target_class_differences[maximal_difference_index]/10);

        if(unbalanced_instances_number < 1)
        {
            unbalanced_instances_number = 1;
        }

        data_histograms = calculate_data_histograms(bins_number);

        total_frequencies.clear();

        total_frequencies.set(instances_number, 2);

        count_instances = 0;

        for(size_t i = 0; i < instances_number; i++)
        {
            instance_index = instances_indices[i];

            instance = get_instance(instance_index);

            instance_target_index = targets_variables_indices[maximal_difference_index];

            if(instance[instance_target_index] == 1.0)
            {
                instance_frequencies = instance.calculate_total_frequencies(data_histograms);

                total_frequencies(count_instances, 0) = instance_frequencies.calculate_partial_sum(inputs_variables_indices);
                total_frequencies(count_instances, 1) = instance_index;

                count_instances++;
            }
        }

        unbalanced_instances_indices = total_frequencies.sort_descending(0).get_column(1).get_first(unbalanced_instances_number);

        unused_instances = unused_instances.assemble(unbalanced_instances_indices);

        instances.set_unused(unbalanced_instances_indices);

        target_class_differences[maximal_difference_index] = target_class_differences[maximal_difference_index] - unbalanced_instances_number;
    }

    return(unused_instances);
}


/// This method unuses a given number of instances of the most populated target.
/// If the given number is greater than the number of used instances which belongs to that target,
/// it unuses all the instances in that target.
/// If the given number is lower than 1, it unuses 1 instance.
/// @param instances_to_unuse Number of instances to set unused.

Vector<size_t> DataSet::unuse_most_populated_target(const size_t& instances_to_unuse)
{
    Vector<size_t> most_populated_instances(instances_to_unuse);

    if(instances_to_unuse == 0)
    {
        return(most_populated_instances);
    }

    const size_t bins_number = 10;

    // Variables

    const size_t targets_number = variables.get_targets_number();

    const Vector<size_t> inputs = variables.get_inputs_indices();
    const Vector<size_t> targets = variables.get_targets_indices();

    const Vector<size_t> unused_variables = variables.get_unused_indices();

    // Instances

    const Vector<size_t> used_instances = instances.get_used_indices();

    const size_t used_instances_number = instances.get_used_instances_number();

    // Most populated target

    const Vector< Histogram<double> > data_histograms = calculate_data_histograms(bins_number);

    size_t most_populated_target = 0;
    size_t most_populated_bin = 0;

    size_t frequency;
    size_t maximum_frequency = 0;

    size_t unused = 0;

    for(size_t i = 0; i < targets_number; i++)
    {
        frequency = data_histograms[targets[i] - unused_variables.count_less_than(targets[i])].calculate_maximum_frequency();

        if(frequency > maximum_frequency)
        {
            unused = unused_variables.count_less_than(targets[i]);

            maximum_frequency = frequency;

            most_populated_target = targets[i];

            most_populated_bin = data_histograms[targets[i] - unused].calculate_most_populated_bin();
        }
    }

    // Calculates frequencies of the instances which belong to the most populated target

    size_t index;
    size_t bin;
    double value;
    Vector<double> instance;

    Vector<size_t> instance_frequencies;

    Matrix<size_t> total_instances_frequencies(maximum_frequency, 2);

    size_t count_instances = 0;

    int i = 0;

//    #pragma omp parallel for private(i, instance_index, instance, instance_value, instance_bin, instance_frequencies) shared(count_instances, total_instances_frequencies, maximal_frequency_target_index)

    for(i = 0; i < static_cast<int>(used_instances_number); i++)
    {
        index = used_instances[static_cast<size_t>(i)];

        instance = get_instance(index);

        value = instance[most_populated_target];

        bin = data_histograms[most_populated_target - unused].calculate_bin(value);

        if(bin == most_populated_bin)
        {
            instance_frequencies = instance.calculate_total_frequencies(data_histograms);

            total_instances_frequencies(count_instances, 0) = instance_frequencies.calculate_partial_sum(inputs);
            total_instances_frequencies(count_instances, 1) = used_instances[static_cast<size_t>(i)];

            count_instances++;
        }
    }

    // Unuses instances

    if(instances_to_unuse > maximum_frequency)
    {
        most_populated_instances = total_instances_frequencies.sort_descending(0).get_column(1).get_first(maximum_frequency);
    }
    else
    {
        most_populated_instances = total_instances_frequencies.sort_descending(0).get_column(1).get_first(instances_to_unuse);
    }

    instances.set_unused(most_populated_instances);

    return(most_populated_instances);
}


/// This method balances the target ditribution of a data set for a function regression problem.
/// It returns a vector with the indices of the instances set unused.
/// It unuses a given percentage of the instances.
/// @param percentage Percentage of the instances to be unused.

Vector<size_t> DataSet::balance_approximation_targets_distribution(const double& percentage)
{
    Vector<size_t> unused_instances;

    const size_t instances_number = instances.get_used_instances_number();

    const size_t instances_to_unuse = static_cast<size_t>(instances_number*percentage/100.0);

    size_t count;

    while(unused_instances.size() < instances_to_unuse)
    {
        if(instances_to_unuse - unused_instances.size() < instances_to_unuse/10)
        {
            count = instances_to_unuse - unused_instances.size();
        }
        else
        {
            count = instances_to_unuse/10;
        }

        if(count == 0)
        {
            count = 1;
        }

        unused_instances = unused_instances.assemble(unuse_most_populated_target(count));
    }

    return(unused_instances);
}


/// Returns a vector with the indices of the inputs that are binary.

Vector<size_t> DataSet::get_binary_inputs_indices() const
{
    const size_t inputs_number = variables.get_inputs_number();

    const Vector<size_t> inputs_indices = variables.get_inputs_indices();

    Vector<size_t> binary_inputs_indices;

    for(size_t i = 0; i < inputs_number; i++)
    {
        if(is_binary_variable(inputs_indices[i]))
        {
            binary_inputs_indices.push_back(inputs_indices[i]);
        }
    }

    return(binary_inputs_indices);
}


/// Returns a vector with the indices of the inputs that are real.

Vector<size_t> DataSet::get_real_inputs_indices() const
{
    const size_t inputs_number = variables.get_inputs_number();

    const Vector<size_t> inputs_indices = variables.get_inputs_indices();

    Vector<size_t> real_inputs_indices;

    for(size_t i = 0; i < inputs_number; i++)
    {
        if(!is_binary_variable(inputs_indices[i]))
        {
            real_inputs_indices.push_back(inputs_indices[i]);
        }
    }

    return(real_inputs_indices);
}


/// @todo

void DataSet::sum_binary_inputs()
{
//    const size_t inputs_number = variables.get_inputs_number();

//    const size_t instances_number = instances.get_instances_number();

//    const Vector<size_t> binary_inputs_indices = get_binary_inputs_indices();

//    const size_t binary_inputs_number = binary_inputs_indices.size();

//    Vector<double> binary_variable(instances_number, 0.0);

//    for(size_t i = 0; i < binary_inputs_number; i++)
//    {
//        binary_variable += data.get_column(binary_inputs_indices[i]);
//    }

//    const Vector<size_t> real_inputs_indices = get_real_inputs_indices();

//    Matrix<double> new_data = data.get_submatrix_columns(real_inputs_indices);

//    new_data.append_column(binary_variable);

//    new_data = new_data.assemble_columns(get_targets());
}

/*

// Vector<double> calculate_local_outlier_factor(const size_t&) const

/// Returns a vector with the local outlier factors for every used instance.
/// @param nearest_neighbours_number Number of neighbors to be calculated.

Vector<double> DataSet::calculate_local_outlier_factor(const size_t& nearest_neighbours_number) const
{
    const size_t instances_number = instances.get_used_instances_number();

    const Matrix<double> distances = calculate_instances_distances(nearest_neighbours_number);
    const Vector<double> reachability_density = calculate_reachability_density(distances, nearest_neighbours_number);

    const Matrix<size_t> nearest_neighbors = calculate_k_nearest_neighbors(distances, nearest_neighbours_number);

    Vector<size_t> instance_nearest_neighbors(nearest_neighbours_number);

    Vector<double> local_outlier_factor(instances_number);

    for(size_t i = 0; i < instances_number; i++)
    {
        instance_nearest_neighbors = nearest_neighbors.get_row(i);

        local_outlier_factor[i] = (reachability_density.calculate_partial_sum(instance_nearest_neighbors))/(nearest_neighbours_number*reachability_density[i]);
    }

    return(local_outlier_factor);
}


// Vector<size_t> clean_local_outlier_factor(const size_t&)

/// Removes the outliers from the data set using the local outlier factor method.
/// @param nearest_neighbours_number Number of nearest neighbours to calculate
/// @todo

Vector<size_t> DataSet::clean_local_outlier_factor(const size_t& nearest_neighbours_number)
{
    Vector<size_t> unused_instances;

    const Vector<double> local_outlier_factor = calculate_local_outlier_factor(nearest_neighbours_number);

    const size_t instances_number = instances.get_used_instances_number();
    const Vector<size_t> instances_indices = instances.get_used_indices();

    for(size_t i = 0; i < instances_number; i++)
    {
        if(local_outlier_factor[i] > 1.6001)
        {
            instances.set_use(instances_indices[i], Instances::Unused);

            unused_instances.push_back(instances_indices[i]);
        }
    }

    return(unused_instances);
}*/


/// Calculate the outliers from the data set using the Tukey's test for a single variable.
/// @param variable_index Index of the variable to calculate the outliers.
/// @param cleaning_parameter Parameter used to detect outliers.

Vector<size_t> DataSet::calculate_Tukey_outliers(const size_t& variable_index, const double& cleaning_parameter) const
{
    const size_t instances_number = instances.get_used_instances_number();
    const Vector<size_t> instances_indices = instances.get_used_indices();

    double interquartile_range;

    Vector<size_t> unused_instances_indices;

    if(is_binary_variable(variable_index))
    {
        return(unused_instances_indices);
    }

    const Vector<double> box_plot = data.get_column(variable_index).calculate_box_plot();

    if(fabs(box_plot[3] - box_plot[1]) < numeric_limits<double>::epsilon())
    {
        return(unused_instances_indices);
    }
    else
    {
        interquartile_range = abs((box_plot[3] - box_plot[1]));
    }

    for(size_t j = 0; j < instances_number; j++)
    {
        const Vector<double> instance = get_instance(instances_indices[j]);

        if(instance[variable_index] <(box_plot[1] - cleaning_parameter*interquartile_range))
        {
            unused_instances_indices.push_back(instances_indices[j]);
        }
        else if(instance[variable_index] >(box_plot[3] + cleaning_parameter*interquartile_range))
        {
            unused_instances_indices.push_back(instances_indices[j]);
        }
    }

    return(unused_instances_indices);
}


/// Calculate the outliers from the data set using the Tukey's test.
/// @param cleaning_parameter Parameter used to detect outliers.

Vector< Vector<size_t> > DataSet::calculate_Tukey_outliers(const double& cleaning_parameter) const
{
    const size_t instances_number = instances.get_used_instances_number();
    const Vector<size_t> instances_indices = instances.get_used_indices();

    const size_t variables_number = variables.count_used_variables_number();
    const Vector<size_t> used_variables_indices = variables.get_used_indices();

    double interquartile_range;

    Vector< Vector<size_t> > return_values(2);
    return_values[0] = Vector<size_t>(instances_number, 0);
    return_values[1] = Vector<size_t>(variables_number, 0);

    size_t variable_index;

    Vector< Vector<double> > box_plots(variables_number);

#pragma omp parallel for private(variable_index) schedule(dynamic)

    for(int i = 0; i < static_cast<int>(variables_number); i++)
    {
        variable_index = used_variables_indices[static_cast<size_t>(i)];

        if(is_binary_variable(variable_index))
        {
            continue;
        }

        box_plots[static_cast<size_t>(i)] = data.get_column(variable_index).calculate_box_plot();
    }

    for(int i = 0; i < static_cast<int>(variables_number); i++)
    {
        variable_index = used_variables_indices[static_cast<size_t>(i)];

        if(is_binary_variable(variable_index))
        {
            continue;
        }

        const Vector<double> variable_box_plot = box_plots[static_cast<size_t>(i)];

        if(fabs(variable_box_plot[3] - variable_box_plot[1]) < numeric_limits<double>::epsilon())
        {
            continue;
        }
        else
        {
            interquartile_range = abs((variable_box_plot[3] - variable_box_plot[1]));
        }

        size_t variables_outliers = 0;

#pragma omp parallel for schedule(dynamic) reduction(+ : variables_outliers)

        for(int j = 0; j < static_cast<int>(instances_number); j++)
        {
            const Vector<double> instance = get_instance(instances_indices[static_cast<size_t>(j)]);

            if(instance[variable_index] <(variable_box_plot[1] - cleaning_parameter*interquartile_range) ||
               instance[variable_index] >(variable_box_plot[3] + cleaning_parameter*interquartile_range))
            {
                    return_values[0][static_cast<size_t>(j)] = 1;

                    variables_outliers++;
            }
        }

        return_values[1][static_cast<size_t>(i)] = variables_outliers;
    }

    return(return_values);
}


void DataSet::unuse_Tukey_outliers(const double& cleaning_parameter)
{
    const Vector< Vector<size_t> > outliers_indices = calculate_Tukey_outliers(cleaning_parameter);

    const Vector<size_t> outliers_instances = outliers_indices[0].calculate_greater_than_indices(0);

    instances.set_unused(outliers_instances);
}


/// Returns a matrix with the values of autocorrelation for every variable in the data set.
/// The number of rows is equal to the number of instances.
/// The number of columns is the maximum lags number.
/// @param maximum_lags_number Maximum lags number for which autocorrelation is calculated.

Matrix<double> DataSet::calculate_autocorrelations(const size_t& maximum_lags_number) const
{
    if(lags_number > instances.get_used_instances_number())
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: DataSet class.\n"
               << "Matrix<double> calculate_autocorrelations(const size_t&) method.\n"
               << "Maximum lags number(" << maximum_lags_number << ") is greater than the number of instances("
               << instances.get_used_instances_number() <<") \n";

        throw logic_error(buffer.str());
    }

//    if(time_series_data.empty())
//    {
//        ostringstream buffer;

//        buffer << "OpenNN Exception: DataSet class.\n"
//               << "Matrix<double> calculate_autocorrelation(const size_t&) method.\n"
//               << "Time series data is empty.\n";

//        throw logic_error(buffer.str());
//    }

    const size_t variables_number = data.get_columns_number();

    Matrix<double> autocorrelations(variables_number, maximum_lags_number);

    for(size_t i = 0; i < maximum_lags_number; i++)
    {
        for(size_t j = 0; j < variables_number; j++)
        {
            autocorrelations.set_row(j, CorrelationAnalysis::calculate_autocorrelations(data.get_column(j), maximum_lags_number));
        }
    }

    return autocorrelations;
}


/// Calculates the cross-correlation between all the variables in the data set.

Matrix< Vector<double> > DataSet::calculate_cross_correlations(const size_t& lags_number) const
{
    const size_t variables_number = variables.get_variables_number();

    Matrix< Vector<double> > cross_correlations(variables_number, variables_number);

    Vector<double> actual_column;

    for(size_t i = 0; i < variables_number; i++)
    {
        actual_column = data.get_column(i);

        for(size_t j = 0; j < variables_number; j++)
        {
            cross_correlations(i , j) = CorrelationAnalysis::calculate_cross_correlations(actual_column, data.get_column(j), lags_number);
        }
    }

    return(cross_correlations);
}


/// @todo

Matrix<double> DataSet::calculate_lag_plot() const
{
    const size_t instances_number = instances.get_used_instances_number();

    const size_t columns_number = data.get_columns_number() - 1;

    Matrix<double> lag_plot(instances_number, columns_number);

    Vector<size_t> columns_indices(1, 1, columns_number);

    lag_plot = data.get_submatrix_columns(columns_indices);

    return lag_plot;
}


/// @todo

Matrix<double> DataSet::calculate_lag_plot(const size_t& maximum_lags_number)
{
    const size_t instances_number = instances.get_used_instances_number();

    if(maximum_lags_number > instances_number)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: DataSet class.\n"
               << "Matrix<double> calculate_lag_plot(const size_t&) method.\n"
               << "Maximum lags number(" << maximum_lags_number
               << ") is greater than the number of instances("
               << instances_number << ") \n";

        throw logic_error(buffer.str());
    }

    const Matrix<double> lag_plot = time_series_data.calculate_lag_plot_matrix(maximum_lags_number, time_index);

    return lag_plot;
}


/// Returns a vector with the linear regression parameters for each used variable.

Vector< LinearRegressionParameters<double> > DataSet::perform_trends_transformation()
{
    const Vector<size_t> used_instances_indices = instances.get_used_indices();
    const size_t used_instances_number = used_instances_indices.size();

    const Vector< Vector<size_t> > missing_indices = missing_values.get_missing_indices();

    const size_t used_variables_number = variables.count_used_variables_number();
    const Vector<size_t> used_variables_indices = variables.get_used_indices();

    Vector< LinearRegressionParameters<double> > trends(used_variables_number - 1);

    Vector<double> independent_variable = get_variable(time_index);
    Vector<double> dependent_variable(used_instances_number - 1);

    size_t used_variable_index;
    size_t trends_index = 0;

    for(size_t i = 0; i < used_variables_number; i++)
    {
        used_variable_index = used_variables_indices[i];

        if(used_variable_index != time_index)
        {
            const Vector<size_t> current_used_instances = used_instances_indices.get_difference(missing_indices[i]);

            dependent_variable = get_variable(used_variable_index, current_used_instances);

            independent_variable = get_variable(time_index, current_used_instances);

            trends[trends_index] = dependent_variable.calculate_linear_regression_parameters(independent_variable);

            trends_index++;
        }
    }

    data.delete_trends_missing_values(trends, time_index, missing_indices);

    return(trends);
}


/// Returns a vector with the linear regression parameters for each input variable.

Vector< LinearRegressionParameters<double> > DataSet::perform_inputs_trends_transformation()
{
    const Vector<size_t> used_instances_indices = instances.get_used_indices();
    const size_t used_instances_number = used_instances_indices.size();

    const Vector< Vector<size_t> > missing_indices = missing_values.get_missing_indices();

    const size_t inputs_number = variables.get_inputs_number();
    const Vector<size_t> inputs_indices = variables.get_inputs_indices();

    Vector< LinearRegressionParameters<double> > inputs_trends(inputs_number);

    Vector<double> independent_variable = get_variable(time_index);
    Vector<double> dependent_variable(used_instances_number - 1);

    size_t input_index;
    size_t inputs_trends_index = 0;

    for(size_t i = 0; i < inputs_number; i++)
    {
        input_index = inputs_indices[i];

        Vector<size_t> current_missing_indices = missing_indices[input_index];

        Vector<size_t> current_used_instances = used_instances_indices.get_difference(current_missing_indices);

        dependent_variable = get_variable(input_index, current_used_instances);

        independent_variable = get_variable(time_index, current_used_instances);

        inputs_trends[inputs_trends_index] = dependent_variable.calculate_linear_regression_parameters(independent_variable);

        inputs_trends_index++;
    }

    data.delete_inputs_trends_missing_values(inputs_trends, time_index, inputs_indices, missing_indices);

    return(inputs_trends);
}


/// Returns a vector with the linear regression parameters for each target variable.

Vector< LinearRegressionParameters<double> > DataSet::perform_outputs_trends_transformation()
{
    const Vector<size_t> used_instances_indices = instances.get_used_indices();
    const size_t used_instances_number = used_instances_indices.size();

    const Vector< Vector<size_t> > missing_indices = missing_values.get_missing_indices();

    const size_t targets_number = variables.get_targets_number();
    const Vector<size_t> targets_indices = variables.get_targets_indices();

    Vector< LinearRegressionParameters<double> > outputs_trends(targets_number);

    Vector<double> independent_variable = get_variable(time_index);
    Vector<double> dependent_variable(used_instances_number - 1);

    size_t target_index;
    size_t outputs_trends_index = 0;

    for(size_t i = 0; i < targets_number; i++)
    {
        target_index = targets_indices[i];

        Vector<size_t> current_missing_indices = missing_indices[target_index];

        Vector<size_t> current_used_instances = used_instances_indices.get_difference(current_missing_indices);

        dependent_variable = get_variable(target_index, current_used_instances);

        independent_variable = get_variable(time_index, current_used_instances);

        outputs_trends[outputs_trends_index] = dependent_variable.calculate_linear_regression_parameters(independent_variable);

        outputs_trends_index++;
    }

    data.delete_outputs_trends_missing_values(outputs_trends, time_index, targets_indices, missing_indices);

    return(outputs_trends);
}


void DataSet::generate_constant_data(const size_t& instances_number, const size_t& variables_number)
{
    set(instances_number, variables_number);

    data.randomize_uniform(-5.12, 5.12);

    for(size_t i = 0; i < instances_number; i++)
    {
        data(i, variables_number-1) = 0.0;
    }

    data.scale_minimum_maximum();
}


void DataSet::generate_random_data(const size_t& instances_number, const size_t& variables_number)
{
    set(instances_number, variables_number);

    data.randomize_uniform(0.0, 1.0);
}


void DataSet::generate_paraboloid_data(const size_t& instances_number, const size_t& variables_number)
{
    const size_t inputs_number = variables_number-1;

    set(instances_number, variables_number);

    data.randomize_uniform(-5.12, 5.12);

    for(size_t i = 0; i < instances_number; i++)
    {
        const double norm = data.get_row(i).delete_last(1).calculate_L2_norm();

        data(i, inputs_number) = norm*norm;
    }

    data.scale_minimum_maximum();
}


/// Generates an artificial dataset with a given number of instances and number of variables
/// using the Rosenbrock function.
/// @param instances_number Number of instances in the dataset.
/// @param variables_number Number of variables in the dataset.

void DataSet::generate_Rosenbrock_data(const size_t& instances_number, const size_t& variables_number)
{
    const size_t inputs_number = variables_number-1;

    set(instances_number, variables_number);

    data.randomize_uniform(-2.048, 2.048);

    double rosenbrock;

    for(size_t i = 0; i < instances_number; i++)
    {
        rosenbrock = 0.0;

        for(size_t j = 0; j < inputs_number-1; j++)
        {
            rosenbrock +=
           (1.0 - data(i,j))*(1.0 - data(i,j))
            + 100.0*(data(i,j+1)-data(i,j)*data(i,j))*(data(i,j+1)-data(i,j)*data(i,j));
        }

        data(i, inputs_number) = rosenbrock;
    }

    data.scale_range(0.0, 1.0);

//    cout << data.calculate_statistics() << endl;


}


/// Generate artificial data for a binary classification problem with a given number of instances and inputs.
/// @param instances_number Number of the instances to generate.
/// @param inputs_number Number of the variables that the data set will have.

void DataSet::generate_data_binary_classification(const size_t& instances_number, const size_t& inputs_number)
{
    const size_t negatives = instances_number/2;
    const size_t positives = instances_number - negatives;

    // Negatives data

    Vector<double> target_0(negatives, 0.0);

    Matrix<double> class_0(negatives, inputs_number);

    class_0.randomize_normal(-0.5, 1.0);

    class_0 = class_0.append_column(target_0, "");

    // Positives data

    Vector<double> target_1(positives, 1.0);

    Matrix<double> class_1(positives, inputs_number);

    class_1.randomize_normal(0.5, 1.0);

    class_1 = class_1.append_column(target_1, "");

    // Assemble

    set(class_0.assemble_rows(class_1));

}


/// @todo

void DataSet::generate_data_multiple_classification(const size_t&, const size_t&)
{

}


/// Returns true if the data matrix is not empty(it has not been loaded),
/// and false otherwise.

bool DataSet::has_data() const
{
    if(data.empty())
    {
        return(false);
    }
    else
    {
        return(true);
    }
}


/// Unuses those instances with values outside a defined range.
/// @param minimums Vector of minimum values in the range.
/// The size must be equal to the number of variables.
/// @param maximums Vector of maximum values in the range.
/// The size must be equal to the number of variables.

Vector<size_t> DataSet::filter_data(const Vector<double>& minimums, const Vector<double>& maximums)
{
    const Vector<size_t> used_variables_indices = variables.get_used_indices();

    const size_t used_variables_number = used_variables_indices.size();//variables.get_variables_number();

    // Control sentence(if debug)

    #ifdef __OPENNN_DEBUG__

    if(minimums.size() != used_variables_number)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: DataSet class.\n"
               << "Vector<size_t> filter_data(const Vector<double>&, const Vector<double>&) method.\n"
               << "Size of minimums(" << minimums.size() << ") is not equal to number of variables(" << used_variables_number << ").\n";

        throw logic_error(buffer.str());
    }

    if(maximums.size() != used_variables_number)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: DataSet class.\n"
               << "Vector<size_t> filter_data(const Vector<double>&, const Vector<double>&) method.\n"
               << "Size of maximums(" << maximums.size() << ") is not equal to number of variables(" << used_variables_number << ").\n";

        throw logic_error(buffer.str());
    }

    #endif

    const size_t instances_number = instances.get_instances_number();

    Vector<double> filtered_indices(instances_number, 0.0);

    const Vector<size_t> used_instances_indices = instances.get_used_indices();

    for(size_t j = 0; j < used_variables_number; j++)
    {
        const size_t current_variable_index = used_variables_indices[j];

        const Vector<size_t> current_missing_instances_indices = missing_values.get_missing_instances(current_variable_index);

        const Vector<size_t> current_instances_indices = used_instances_indices.get_difference(current_missing_instances_indices);

        const size_t current_instances_number = current_instances_indices.size();

        for(size_t i = 0; i < current_instances_number; i++)
        {
            const size_t current_instance_index = current_instances_indices[i];

            if(data(current_instance_index,current_variable_index) < minimums[j]
            || data(current_instance_index,current_variable_index) > maximums[j])
            {
                filtered_indices[current_instance_index] = 1.0;

                instances.set_use(current_instance_index, Instances::Unused);
            }
        }
    }

    return(filtered_indices.calculate_greater_than_indices(0.5));
}


Vector<size_t> DataSet::filter_variable(const size_t& variable_index, const double& minimum, const double& maximum)
{
    const size_t instances_number = instances.get_instances_number();

    Vector<double> filtered_indices(instances_number, 0.0);

    const Vector<size_t> used_instances_indices = instances.get_used_indices();

    const Vector<size_t> current_missing_instances_indices = missing_values.get_missing_instances(variable_index);

    const Vector<size_t> current_instances_indices = used_instances_indices.get_difference(current_missing_instances_indices);

    const size_t current_instances_number = current_instances_indices.size();

    for(size_t i = 0; i < current_instances_number; i++)
    {
        const size_t index = current_instances_indices[i];

        if(data(index,variable_index) < minimum || data(index,variable_index) > maximum)
        {
            filtered_indices[index] = 1.0;

            instances.set_use(index, Instances::Unused);
        }
    }

    return(filtered_indices.calculate_greater_than_indices(0.5));
}


Vector<size_t> DataSet::filter_variable(const string& variable_name, const double& minimum, const double& maximum)
{
    const size_t variable_index = variables.get_variable_index(variable_name);

    const size_t instances_number = instances.get_instances_number();

    Vector<double> filtered_indices(instances_number, 0.0);

    const Vector<size_t> used_instances_indices = instances.get_used_indices();

    const Vector<size_t> current_missing_instances_indices = missing_values.get_missing_instances(variable_index);

    const Vector<size_t> current_instances_indices = used_instances_indices.get_difference(current_missing_instances_indices);

    const size_t current_instances_number = current_instances_indices.size();

    for(size_t i = 0; i < current_instances_number; i++)
    {
        const size_t index = current_instances_indices[i];

        if(data(index,variable_index) < minimum || data(index,variable_index) > maximum)
        {
            filtered_indices[index] = 1.0;

            instances.set_use(index, Instances::Unused);
        }
    }

    return(filtered_indices.calculate_greater_than_indices(0.5));
}


/// Replaces a given angular variable expressed in degrees by the sinus and cosinus of that variable.
/// This solves the discontinuity associated with angular variables.
/// Note that this method modifies the number of variables.
/// @param variable_index Index of angular variable.

void DataSet::convert_angular_variable_degrees(const size_t& variable_index)
{
    // Control sentence(if debug)

    #ifdef __OPENNN_DEBUG__

    const size_t variables_number = variables.get_variables_number();

    if(variable_index >= variables_number)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: DataSet class.\n"
               << "void convert_angular_variable_degrees(const size_t&) method.\n"
               << "Index of variable(" << variable_index << ") must be less than number of variables(" << variables_number << ").\n";

        throw logic_error(buffer.str());
    }

    #endif

    Vector<Variables::Item> items = variables.get_items();

    Variables::Item sin_item = items[variable_index];
    prepend("sin_", sin_item.name);

    Variables::Item cos_item = items[variable_index];
    prepend("cos_", cos_item.name);

    items[variable_index] = sin_item;
    items = items.insert_element(variable_index, cos_item);

    variables.set_items(items);

    data.convert_angular_variables_degrees(variable_index);

}


/// Replaces a given angular variable expressed in radians by the sinus and cosinus of that variable.
/// This solves the discontinuity associated with angular variables.
/// Note that this method modifies the number of variables.
/// @param variable_index Index of angular variable.

void DataSet::convert_angular_variable_radians(const size_t& variable_index)
{
    // Control sentence(if debug)

    #ifdef __OPENNN_DEBUG__

    const size_t variables_number = variables.get_variables_number();

    if(variable_index >= variables_number)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: DataSet class.\n"
               << "void convert_angular_variable_radians(const size_t&) method.\n"
               << "Index of variable(" << variable_index << ") must be less than number of variables(" << variables_number << ").\n";

        throw logic_error(buffer.str());
    }

    #endif

    Vector<Variables::Item> items = variables.get_items();

    Variables::Item sin_item = items[variable_index];
    prepend("sin_", sin_item.name);

    Variables::Item cos_item = items[variable_index];
    prepend("cos_", cos_item.name);

    items[variable_index] = sin_item;
    items = items.insert_element(variable_index, cos_item);

    variables.set_items(items);

    data.convert_angular_variables_radians(variable_index);

}


/// Replaces a given set of angular variables expressed in degrees by the sinus and cosinus of that variable.
/// This solves the discontinuity associated with angular variables.
/// Note that this method modifies the number of variables.
/// @param indices Indices of angular variables.

void DataSet::convert_angular_variables_degrees(const Vector<size_t>& indices)
{
    // Control sentence(if debug)

    #ifdef __OPENNN_DEBUG__

    const size_t variables_number = variables.get_variables_number();

    for(size_t i = 0; i < indices.size(); i++)
    {
        if(indices[i] >= variables_number)
        {
            ostringstream buffer;

            buffer << "OpenNN Exception: DataSet class.\n"
                   << "void convert_angular_variables_degrees(const Vector<size_t>&) method.\n"
                   << "Index(" << i << ") must be less than number of variables(" << variables_number << ").\n";

            throw logic_error(buffer.str());
        }
    }

    #endif

    size_t size = indices.size();

    unsigned count = 0;

    size_t index;

    for(size_t i = 0; i < size; i++)
    {
        index = indices[i]+count;

        convert_angular_variable_degrees(index);

        count++;
    }
}


/// Replaces a given set of angular variables expressed in radians by the sinus and cosinus of that variable.
/// This solves the discontinuity associated with angular variables.
/// Note that this method modifies the number of variables.
/// @param indices Indices of angular variables.

void DataSet::convert_angular_variables_radians(const Vector<size_t>& indices)
{
    // Control sentence(if debug)

    #ifdef __OPENNN_DEBUG__

    const size_t variables_number = variables.get_variables_number();

    for(size_t i = 0; i < indices.size(); i++)
    {
        if(indices[i] >= variables_number)
        {
            ostringstream buffer;

            buffer << "OpenNN Exception: DataSet class.\n"
                   << "void convert_angular_variables_radians(const Vector<size_t>&) method.\n"
                   << "Index(" << i << ") must be less than number of variables(" << variables_number << ").\n";

            throw logic_error(buffer.str());
        }
    }

    #endif

    size_t size = indices.size();

    unsigned count = 0;

    size_t index;

    for(size_t i = 0; i < size; i++)
    {
        index = indices[i]+count;

        convert_angular_variable_radians(index);

        count++;
    }
}


/// Replaces a given set of angular variables by the sinus and cosinus of that variable, according to the angular units used.
/// This solves the discontinuity associated with angular variables.
/// Note that this method modifies the number of variables.

void DataSet::convert_angular_variables()
{
    switch(angular_units)
    {
       case DataSet::Radians:
       {
            convert_angular_variables_radians(angular_variables);
       }
       break;

       case DataSet::Degrees:
       {
            convert_angular_variables_degrees(angular_variables);
       }
       break;
    }
}


/// Sets all the instances with missing values to "Unused".

void DataSet::impute_missing_values_unuse()
{
    const Vector<size_t> used_variables_indices = get_variables_pointer()->get_used_indices();

    const Vector<size_t> missing_instances = missing_values.get_missing_instances(used_variables_indices);

    const size_t missing_instances_size = missing_instances.size();

#pragma omp parallel for

    for(int i = 0; i < static_cast<int>(missing_instances_size); i++)
    {
        instances.set_use(missing_instances[static_cast<size_t>(i)], Instances::Unused);
    }
}


/// Substitutes all the missing values by the mean of the corresponding variable.

void DataSet::impute_missing_values_mean()
{
    const Vector<size_t> used_variables_indices = get_variables_pointer()->get_used_indices();

    const Vector< Vector<size_t> > missing_indices = missing_values.get_missing_indices(used_variables_indices);

    const Vector<double> means = data.calculate_mean_missing_values(Vector<size_t>(0,1,data.get_rows_number()-1),used_variables_indices,missing_indices);

//    const size_t variables_number = variables.get_variables_number();
    const size_t variables_number = used_variables_indices.size();

    #pragma omp parallel for schedule(dynamic)

    for(int i = 0; i < static_cast<int>(variables_number); i++)
    {
        for(size_t j = 0; j < missing_indices[static_cast<size_t>(i)].size(); j++)
        {
            const size_t instance_index = missing_indices[static_cast<size_t>(i)][j];

            data(instance_index, used_variables_indices[static_cast<size_t>(i)]) = means[static_cast<size_t>(i)];
        }
    }
}


/// @todo

void DataSet::impute_missing_values_time_series_mean()
{
    const Vector< Vector<size_t> > missing_indices = missing_values.get_missing_indices();

    const Matrix<double> means = data.calculate_time_series_mean_missing_values(missing_indices);

    const size_t variables_number = variables.get_variables_number();

    #pragma omp parallel for schedule(dynamic)

    for(int i = 0; i < static_cast<int>(variables_number); i++)
    {
        for(size_t j = 0; j < missing_indices[static_cast<size_t>(i)].size(); j++)
        {
            const size_t instance_index = missing_indices[static_cast<size_t>(i)][j];

            data(instance_index, static_cast<size_t>(i)) = means(instance_index,static_cast<size_t>(i));
        }
    }
}


void DataSet::impute_missing_values_time_series_regression()
{
//    const Vector< Vector<size_t> > missing_indices = missing_values.get_missing_indices();

//    const size_t variables_number = variables.get_variables_number();

//    const size_t instances_number = instances.get_instances_number();

//    const Vector<double> variables_means = data.calculate_mean_missing_values(missing_indices);

//    size_t previous_good_value_index;
//    size_t next_good_value_index;
//    Vector<size_t> items_to_scrub;

//    for(size_t i = 0; i < variables_number; i++)
//    {
//        const Vector<size_t> current_missing_indices = missing_indices[i];

//        items_to_scrub.clear();
//        previous_good_value_index = -1;
//        next_good_value_index = -1;

//        for(size_t j = 0; j < instances_number; j++)
//        {
//            if(current_missing_indices.contains(j))
//            {
//                items_to_scrub.push_back(j);
//            }
//            else if(items_to_scrub.empty())
//            {
//                previous_good_value_index = j;
//            }
//            else if(previous_good_value_index == -1 && !items_to_scrub.empty())
//            {
//                for(size_t k = 0; k < items_to_scrub.size(); k++)
//                {
//                    data(items_to_scrub[k],i) = variables_means[i];
//                }

//                items_to_scrub.clear();
//                previous_good_value_index = next_good_value_index;
//                next_good_value_index = -1;
//            }
//            else if(previous_good_value_index == -1)
//            {
//                previous_good_value_index = j;
//            }
//            else
//            {
//                next_good_value_index = j;

//                for(size_t k = 0; k < items_to_scrub.size(); k++)
//                {
//                    data(items_to_scrub[k],i) = data(previous_good_value_index,i) +
//                           (data(next_good_value_index,i) - data(previous_good_value_index,i))*(static_cast<double>(items_to_scrub[k]) - previous_good_value_index)/static_cast<double>(next_good_value_index - previous_good_value_index);
//                }

//                items_to_scrub.clear();
//                previous_good_value_index = next_good_value_index;
//                next_good_value_index = -1;
//            }
//        }

//        if(!items_to_scrub.empty() && previous_good_value_index != -1)
//        {
//            for(size_t k = 0; k < items_to_scrub.size(); k++)
//            {
//                data(items_to_scrub[k],i) = variables_means[i];

////                data(items_to_scrub[k],i) = data(previous_good_value_index,i) +
////                       (data(next_good_value_index,i) - data(previous_good_value_index,i))*(static_cast<double>(items_to_scrub[k]) - previous_good_value_index)/static_cast<double>(next_good_value_index - previous_good_value_index);
//            }
//        }
//    }
}


/// Substitutes all the missing values by the median of the corresponding variable.

void DataSet::impute_missing_values_median()
{
    const Vector<size_t> used_variables_indices = get_variables_pointer()->get_used_indices();

    const Vector< Vector<size_t> > missing_indices = missing_values.get_missing_indices(used_variables_indices);

    const Vector<double> medians = data.calculate_median_missing_values(Vector<size_t>(0,1,data.get_rows_number()-1),used_variables_indices,missing_indices);

//    const size_t variables_number = variables.get_variables_number();
    const size_t variables_number = used_variables_indices.size();

    #pragma omp parallel for schedule(dynamic)

    for(int i = 0; i < static_cast<int>(variables_number); i++)
    {
        for(size_t j = 0; j < missing_indices[static_cast<size_t>(i)].size(); j++)
        {
            const size_t instance_index = missing_indices[static_cast<size_t>(i)][j];

            data(instance_index, used_variables_indices[static_cast<size_t>(i)]) = medians[static_cast<size_t>(i)];
        }
    }
}


/// General method for dealing with missing values.
/// It switches among the different scrubbing methods available,
/// according to the corresponding value in the missing values object.

void DataSet::scrub_missing_values()
{
    const MissingValues::ScrubbingMethod scrubbing_method = missing_values.get_scrubbing_method();

    switch(scrubbing_method)
    {
       case MissingValues::Unuse:
       {
            impute_missing_values_unuse();
       }
       break;

       case MissingValues::Mean:
       {
            impute_missing_values_mean();
       }
       break;

       case MissingValues::Median:
       {
            impute_missing_values_median();
       }
       break;

       default:
       {
          ostringstream buffer;

          buffer << "OpenNN Exception: DataSet class\n"
                 << "void scrub_missing_values() method.\n"
                 << "Unknown scrubbing method.\n";

          throw logic_error(buffer.str());
       }
    }
}


/// Returns the number of strings delimited by separator.
/// If separator does not match anywhere in the string, this method returns 0.
/// @param str String to be tokenized.

size_t DataSet::count_tokens(string& str) const
{
//    if(!(this->find(separator) != string::npos))
//    {
//        ostringstream buffer;
//
//        buffer << "OpenNN Exception:\n"
//               << "string class.\n"
//               << "inline size_t count_tokens(const string&) const method.\n"
//               << "Separator not found in string: \"" << separator << "\".\n";
//
//        throw logic_error(buffer.str());
//    }

    trim(str);

    size_t tokens_count = 0;

    // Skip delimiters at beginning.

    const char separator_char = get_separator_char();

    string::size_type last_pos = str.find_first_not_of(separator_char, 0);

    // Find first "non-delimiter".

    string::size_type pos = str.find_first_of(separator_char, last_pos);

    while(string::npos != pos || string::npos != last_pos)
    {
        // Found a token, add it to the vector

        tokens_count++;

        // Skip delimiters.  Note the "not_of"

        last_pos = str.find_first_not_of(separator_char, pos);

        // Find next "non-delimiter"

        pos = str.find_first_of(separator_char, last_pos);
    }

    return(tokens_count);
}


/// Splits the string into substrings(tokens) wherever separator occurs, and returns a vector with those strings.
/// If separator does not match anywhere in the string, this method returns a single-element list containing this string.
/// @param str String to be tokenized.

Vector<string> DataSet::get_tokens(const string& str) const
{
    const string new_string = get_trimmed(str);

    Vector<string> tokens;

    const char separator_char = get_separator_char();

    // Skip delimiters at beginning.

    string::size_type lastPos = new_string.find_first_not_of(separator_char, 0);

    // Find first "non-delimiter"

    string::size_type pos = new_string.find_first_of(separator_char, lastPos);

    while(string::npos != pos || string::npos != lastPos)
    {
        // Found a token, add it to the vector

        tokens.push_back(new_string.substr(lastPos, pos - lastPos));

        // Skip delimiters. Note the "not_of"

        lastPos = new_string.find_first_not_of(separator_char, pos);

        // Find next "non-delimiter"

        pos = new_string.find_first_of(separator_char, lastPos);
    }

    for(size_t i = 0; i < tokens.size(); i++)
    {
        trim(tokens[i]);
    }

    return(tokens);
}


/// Returns true if the string passed as argument represents a number, and false otherwise.
/// @param str String to be checked.

bool DataSet::is_numeric(const string& str) const
{
    istringstream iss(str.data());

    double dTestSink;

    iss >> dTestSink;

    // was any input successfully consumed/converted?

    if(!iss)
    {
        return false;
    }

    // was all the input successfully consumed/converted?

    return(iss.rdbuf()->in_avail() == 0);
}


/// Removes whitespaces from the start and the end of the string passed as argument.
/// This includes the ASCII characters "\t", "\n", "\v", "\f", "\r", and " ".
/// @param str String to be checked.

void DataSet::trim(string& str) const
{
    //prefixing spaces

    str.erase(0, str.find_first_not_of(' '));

    //surfixing spaces

    str.erase(str.find_last_not_of(' ') + 1);
}


/// Returns a string that has whitespace removed from the start and the end.
/// This includes the ASCII characters "\t", "\n", "\v", "\f", "\r", and " ".
/// @param str String to be checked.

string DataSet::get_trimmed(const string& str) const
{
    string output(str);

    //prefixing spaces

    output.erase(0, output.find_first_not_of(' '));

    //surfixing spaces

    output.erase(output.find_last_not_of(' ') + 1);

    return(output);
}


/// Prepends the string pre to the beginning of the string str and returns the whole string.
/// @param pre String to be prepended.
/// @param str original string.

string DataSet::prepend(const string& pre, const string& str) const
{
    ostringstream buffer;

    buffer << pre << str;

    return(buffer.str());
}


/// Returns true if all the elements in a string list are numeric, and false otherwise.
/// @param v String list to be checked.

bool DataSet::is_numeric(const Vector<string>& v) const
{
    for(size_t i = 0; i < v.size(); i++)
    {
        if(!is_numeric(v[i]))
        {
            return false;
        }
    }

    return true;
}


/// Returns true if none element in a string list is numeric, and false otherwise.
/// @param v String list to be checked.

bool DataSet::is_not_numeric(const Vector<string>& v) const
{
    for(size_t i = 0; i < v.size(); i++)
    {
        if(is_numeric(v[i]))
        {
            return false;
        }
    }

    return true;
}


/// Returns true if some the elements in a string list are numeric and some others are not numeric.
/// @param v String list to be checked.

bool DataSet::is_mixed(const Vector<string>& v) const
{
    unsigned count_numeric = 0;
    unsigned count_not_numeric = 0;

    for(size_t i = 0; i < v.size(); i++)
    {
        if(is_numeric(v[i]))
        {
            count_numeric++;
        }
        else
        {
            count_not_numeric++;
        }
    }

    if(count_numeric > 0 && count_not_numeric > 0)
    {
        return true;
    }
    else
    {
        return false;
    }
}


void DataSet::set_variable_use(const size_t& index, const string& use)
{
    variables.set_use(index, use);
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
