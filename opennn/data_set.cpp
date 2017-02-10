/****************************************************************************************************************/
/*                                                                                                              */
/*   OpenNN: Open Neural Networks Library                                                                       */
/*   www.opennn.net                                                                                             */
/*                                                                                                              */
/*   D A T A   S E T   C L A S S                                                                                */
/*                                                                                                              */ 
/*   Roberto Lopez                                                                                              */ 
/*   Artelnics - Making intelligent use of data                                                                 */
/*   robertolopez@artelnics.com                                                                                 */
/*                                                                                                              */
/****************************************************************************************************************/

// OpenNN includes

#include "data_set.h"


namespace OpenNN
{

// DEFAULT CONSTRUCTOR

/// Default constructor. It creates a data set object with zero instances and zero inputs and target variables. 
/// It also initializes the rest of class members to their default values.

DataSet::DataSet(void)
{
   set();  

   set_default();
}


// DATA CONSTRUCTOR

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

/// File constructor. It creates a data set object by loading the object members from a XML-type file. 
/// Please mind about the file format. This is specified in the User's Guide.
/// @param file_name Data set file name.

DataSet::DataSet(const std::string& file_name)
{
   set();

   set_default();

   load(file_name);
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

DataSet::~DataSet(void)
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

// bool operator == (const DataSet&) const method

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

// const Variables& get_variables(void) const

/// Returns a constant reference to the variables object composing this data set object. 

const Variables& DataSet::get_variables(void) const
{
   return(variables);
}


// Variables* get_variables_pointer(void) const

/// Returns a pointer to the variables object composing this data set object. 

Variables* DataSet::get_variables_pointer(void) 
{
   return(&variables);
}


// const Instances& get_instances(void) const

/// Returns a constant reference to the instances object composing this data set object. 

const Instances& DataSet::get_instances(void) const
{
   return(instances);
}


// Instances* get_instances_pointer(void)

/// Returns a pointer to the variables object composing this data set object. 

Instances* DataSet::get_instances_pointer(void) 
{
   return(&instances);
}


// const bool& get_display(void) const method

/// Returns true if messages from this class can be displayed on the screen,
/// or false if messages from this class can't be displayed on the screen.

const bool& DataSet::get_display(void) const
{
   return(display);   
}


// bool is_binary_classification(void) const method

/// Returns true if the data set is a binary classification problem, false otherwise.

bool DataSet::is_binary_classification(void) const
{
    if(variables.count_targets_number() != 1)
    {
        return(false);
    }

    if(!arrange_target_data().is_binary())
    {
        return(false);
    }

    return(true);
}


// bool is_multiple_classification(void) const method

/// Returns true if the data set is a multiple classification problem, false otherwise.

bool DataSet::is_multiple_classification(void) const
{
    const Matrix<double> target_data = arrange_target_data();

    if(!target_data.is_binary())
    {
        return(false);
    }

    for(size_t i = 0; i < target_data.get_rows_number(); i++)
    {
        if(target_data.arrange_row(i).calculate_sum() == 0.0)
        {
            return(false);
        }
    }

    return(true);
}


// bool is_binary_variable(const size_t&) const method

/// Returns true if the given variable is binary and false in other case.
/// @param variable_index Index of the variable that is going to be checked.

bool DataSet::is_binary_variable(const size_t& variable_index) const
{
    const size_t instances_number = instances.get_instances_number();

    for(size_t i = 0; i < instances_number; i++)
    {
        if(data(i,variable_index) == 0.0 || data(i,variable_index) == 1.0)
        {
            continue;
        }
        else
        {
            return false;
        }

    }

    return true;
}


// bool empty(void) const method

/// Returns true if the data matrix is empty, and false otherwise.

bool DataSet::empty(void) const
{
   return(data.empty());
}


// const Matrix<double>& get_data(void) const method

/// Returns a reference to the data matrix in the data set. 
/// The number of rows is equal to the number of instances.
/// The number of columns is equal to the number of variables. 

const Matrix<double>& DataSet::get_data(void) const
{
   return(data);
}


// const Matrix<double>& get_time_series_data(void) const method

/// Returns a reference to the time series data matrix in the data set.
/// Only for time series problems.

const Matrix<double>& DataSet::get_time_series_data(void) const
{
   return(time_series_data);
}

// Matrix<double> get_instances_submatrix_data(const Vector<size_t>&) const method

/// Returns the submatrix of the data with the asked instances.
/// @param instances_indices Indices of the instances to return.

Matrix<double> DataSet::get_instances_submatrix_data(const Vector<size_t>& instances_indices) const
{
    return(data.arrange_submatrix_rows(instances_indices));
}


// FileType get_file_type(void) const method

/// Returns the file type.

DataSet::FileType DataSet::get_file_type(void) const
{
    return(file_type);
}


// std::string write_file_type(void) const

/// Returns a string with the name of the file type.

std::string DataSet::write_file_type(void) const
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

        default:
        {
           std::ostringstream buffer;

           buffer << "OpenNN Exception: DataSet class.\n"
                  << "std::string write_file_type(void) const method.\n"
                  << "Unknown file type.\n";

           throw std::logic_error(buffer.str());
        }
    }
}


// std::string write_first_cell(void) const

/// Returns a string with the first cell for excel files.

std::string DataSet::write_first_cell(void) const
{
   return first_cell;
}


// std::string write_last_cell(void) const

/// Returns a string with the last cell for excel files.

std::string DataSet::write_last_cell(void) const
{
   return last_cell;
}


// int write_sheet_number(void) const

/// Returns a string with the sheet number for excel files.

size_t DataSet::write_sheet_number(void) const
{
   return sheet_number;
}


// ProjectType get_learning_task(void) const

/// Returns the project type.

DataSet::ProjectType DataSet::get_learning_task(void) const
{
    return learning_task;
}


// string write_learning_task(void) const

/// Returns a string with the name of the project type.

std::string DataSet::write_learning_task(void) const
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

        default:
        {
            std::ostringstream buffer;

            buffer << "OpenNN Exception: DataSet class.\n"
                   << "std::string write_learning_task(void) const method.\n"
                   << "Unknown project type.\n";

            throw std::logic_error(buffer.str());
        }
    }
}


// const MissingValues& get_missing_values(void) const method

/// Returns a reference to the missing values object in the data set.

const MissingValues& DataSet::get_missing_values(void) const
{
   return(missing_values);
}


// MissingValues* get_missing_values_pointer(void) method

/// Returns a pointer to the missing values object in the data set.

MissingValues* DataSet::get_missing_values_pointer(void)
{
   return(&missing_values);
}


// const std::string& get_data_file_name(void) const method

/// Returns the name of the data file. 

const std::string& DataSet::get_data_file_name(void) const
{
   return(data_file_name);
}


// const bool& get_header(void) const

/// Returns true if the first line of the data file has a header with the names of the variables, and false otherwise.

const bool& DataSet::get_header_line(void) const
{
    return(header_line);
}


// const bool& get_rows_label(void) const

/// Returns true if the data file has rows label, and false otherwise.

const bool& DataSet::get_rows_label(void) const
{
    return(rows_label);
}


// const Separator& get_separator(void) const

/// Returns the separator to be used in the data file.

const DataSet::Separator& DataSet::get_separator(void) const
{
    return(separator);
}


// std::string get_separator_string(void) const

/// Returns the string which will be used as separator in the data file.

std::string DataSet::get_separator_string(void) const
{
    switch(separator)
    {
       case Space:
       {
          return(" ");
       }
       break;

       case Tab:
       {
          return("\t");
       }
       break;

        case Comma:
        {
           return(",");
        }
        break;

        case Semicolon:
        {
           return(";");
        }
        break;

       default:
       {
          std::ostringstream buffer;

          buffer << "OpenNN Exception: DataSet class.\n"
                 << "std::string get_separator_string(void) const method.\n"
                 << "Unknown separator.\n";

          throw std::logic_error(buffer.str());
       }
       break;
    }
}


// std::string write_separator(void) const

/// Returns the string which will be used as separator in the data file.

std::string DataSet::write_separator(void) const
{
    switch(separator)
    {
       case Space:
       {
          return("Space");
       }
       break;

       case Tab:
       {
          return("Tab");
       }
       break;

        case Comma:
        {
           return("Comma");
        }
        break;

        case Semicolon:
        {
           return("Semicolon");
        }
        break;

       default:
       {
          std::ostringstream buffer;

          buffer << "OpenNN Exception: DataSet class.\n"
                 << "std::string write_separator(void) const method.\n"
                 << "Unknown separator.\n";

          throw std::logic_error(buffer.str());
       }
       break;
    }
}


// const std::string& get_missing_values_label(void) const

/// Returns the string which will be used as label for the missing values in the data file.

const std::string& DataSet::get_missing_values_label(void) const
{
    return(missing_values_label);
}


// const size_t& get_lags_number(void) const

/// Returns the number of lags to be used in a time series prediction application.

const size_t& DataSet::get_lags_number(void) const
{
    return(lags_number);
}


// const size_t& get_steps_ahead(void) const

/// Returns the number of steps ahead to be used in a time series prediction application.

const size_t& DataSet::get_steps_ahead(void) const
{
    return(steps_ahead);
}


// const bool& get_autoassociation(void) const

/// Returns true if the data set will be used for an association application, and false otherwise.
/// In an association problem the target data is equal to the input data.

const bool& DataSet::get_autoassociation(void) const
{
    return(association);
}


// const Vector<size_t>& get_angular_variables(void) const method

/// Returns the indices of the angular variables in the data set.
/// When loading a data set with angular variables,
/// a transformation of the data will be performed in order to avoid discontinuities (from 359 degrees to 1 degree).

const Vector<size_t>& DataSet::get_angular_variables(void) const
{
    return(angular_variables);
}


// const AngularUnits& get_angular_units(void) const method

/// Returns the units used for the angular variables (Radians or Degrees).

const DataSet::AngularUnits& DataSet::get_angular_units(void) const
{
    return(angular_units);
}


// static ScalingUnscalingMethod get_scaling_unscaling_method(const std::string&) method

/// Returns a value of the scaling-unscaling method enumeration from a string containing the name of that method.
/// @param scaling_unscaling_method String with the name of the scaling and unscaling method.

DataSet::ScalingUnscalingMethod DataSet::get_scaling_unscaling_method(const std::string& scaling_unscaling_method)
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
    else if(scaling_unscaling_method == "MeanStandardDeviation")
    {
        return(MeanStandardDeviation);
    }
    else
    {
        std::ostringstream buffer;

        buffer << "OpenNN Exception: DataSet class.\n"
               << "static ScalingUnscalingMethod get_scaling_unscaling_method(const std::string).\n"
               << "Unknown scaling-unscaling method: " << scaling_unscaling_method << ".\n";

        throw std::logic_error(buffer.str());
    }
}


// Matrix<double> arrange_training_data(void) const method

/// Returns a matrix with the training instances in the data set. 
/// The number of rows is the number of training instances.
/// The number of columns is the number of variables. 

Matrix<double> DataSet::arrange_training_data(void) const
{
   const size_t variables_number = variables.get_variables_number();
   
   Vector<size_t> variables_indices(0, 1, (int)variables_number-1);

   const Vector<size_t> training_indices = instances.arrange_training_indices();

   return(data.arrange_submatrix(training_indices, variables_indices));
}


// Matrix<double> arrange_selection_data(void) const method

/// Returns a matrix with the selection instances in the data set. 
/// The number of rows is the number of selection instances.
/// The number of columns is the number of variables. 

Matrix<double> DataSet::arrange_selection_data(void) const
{
   const size_t variables_number = variables.get_variables_number();

   const Vector<size_t> selection_indices = instances.arrange_selection_indices();

   Vector<size_t> variables_indices(0, 1, (int)variables_number-1);

   return(data.arrange_submatrix(selection_indices, variables_indices));
}


// Matrix<double> arrange_testing_data(void) const method

/// Returns a matrix with the testing instances in the data set. 
/// The number of rows is the number of testing instances.
/// The number of columns is the number of variables. 

Matrix<double> DataSet::arrange_testing_data(void) const
{
   const size_t variables_number = variables.get_variables_number();
   Vector<size_t> variables_indices(0, 1, (int)variables_number-1);

   const Vector<size_t> testing_indices = instances.arrange_testing_indices();

   return(data.arrange_submatrix(testing_indices, variables_indices));
}


// Matrix<double> arrange_input_data(void) const method

/// Returns a matrix with the input variables in the data set.
/// The number of rows is the number of instances.
/// The number of columns is the number of input variables.

Matrix<double> DataSet::arrange_input_data(void) const
{
   const size_t instances_number = instances.get_instances_number();
   Vector<size_t> indices(0, 1, (size_t)instances_number-1);

   const Vector<size_t> input_indices = variables.arrange_inputs_indices();

   return(data.arrange_submatrix(indices, input_indices));
}


// Matrix<double> arrange_target_data(void) const method

/// Returns a matrix with the target variables in the data set.
/// The number of rows is the number of instances.
/// The number of columns is the number of target variables. 

Matrix<double> DataSet::arrange_target_data(void) const
{
   const size_t instances_number = instances.get_instances_number();
   Vector<size_t> indices(0, 1, (size_t)instances_number-1);

   const Vector<size_t> targets_indices = variables.arrange_targets_indices();

   return(data.arrange_submatrix(indices, targets_indices));
}

// Matrix<double> arrange_used_input_data(void) const method

/// Returns a matrix with the input variables of the used instances in the data set.
/// The number of rows is the number of used instances.
/// The number of columns is the number of input variables.

Matrix<double> DataSet::arrange_used_input_data(void) const
{
   const Vector<size_t> indices = instances.arrange_used_indices();

   const Vector<size_t> input_indices = variables.arrange_inputs_indices();

   return(data.arrange_submatrix(indices, input_indices));
}


// Matrix<double> arrange_used_target_data(void) const method

/// Returns a matrix with the target variables of the used instances in the data set.
/// The number of rows is the number of used instances.
/// The number of columns is the number of target variables.

Matrix<double> DataSet::arrange_used_target_data(void) const
{
   const Vector<size_t> indices = instances.arrange_used_indices();

   const Vector<size_t> targets_indices = variables.arrange_targets_indices();

   return(data.arrange_submatrix(indices, targets_indices));
}

// Matrix<double> arrange_training_input_data(void) const method

/// Returns a matrix with training instances and input variables.
/// The number of rows is the number of training instances.
/// The number of columns is the number of input variables. 

Matrix<double> DataSet::arrange_training_input_data(void) const
{
   const Vector<size_t> inputs_indices = variables.arrange_inputs_indices();

   const Vector<size_t> training_indices = instances.arrange_training_indices();

   return(data.arrange_submatrix(training_indices, inputs_indices));
}


// Matrix<double> arrange_training_target_data(void) const method

/// Returns a matrix with training instances and target variables.
/// The number of rows is the number of training instances.
/// The number of columns is the number of target variables. 

Matrix<double> DataSet::arrange_training_target_data(void) const 
{
   const Vector<size_t> training_indices = instances.arrange_training_indices();

   const Vector<size_t> targets_indices = variables.arrange_targets_indices();

   return(data.arrange_submatrix(training_indices, targets_indices));
}


// Matrix<double> arrange_selection_input_data(void) const method

/// Returns a matrix with selection instances and input variables.
/// The number of rows is the number of selection instances.
/// The number of columns is the number of input variables. 

Matrix<double> DataSet::arrange_selection_input_data(void) const
{
   const Vector<size_t> selection_indices = instances.arrange_selection_indices();

   const Vector<size_t> inputs_indices = variables.arrange_inputs_indices();

   return(data.arrange_submatrix(selection_indices, inputs_indices));
}


// Matrix<double> arrange_selection_target_data(void) const method

/// Returns a matrix with selection instances and target variables.
/// The number of rows is the number of selection instances.
/// The number of columns is the number of target variables. 

Matrix<double> DataSet::arrange_selection_target_data(void) const
{
   const Vector<size_t> selection_indices = instances.arrange_selection_indices();

   const Vector<size_t> targets_indices = variables.arrange_targets_indices();

   return(data.arrange_submatrix(selection_indices, targets_indices));
}


// Matrix<double> arrange_testing_input_data(void) const method

/// Returns a matrix with testing instances and input variables.
/// The number of rows is the number of testing instances.
/// The number of columns is the number of input variables. 

Matrix<double> DataSet::arrange_testing_input_data(void) const
{
   const Vector<size_t> inputs_indices = variables.arrange_inputs_indices();

   const Vector<size_t> testing_indices = instances.arrange_testing_indices();

   return(data.arrange_submatrix(testing_indices, inputs_indices));
}


// Matrix<double> arrange_testing_target_data(void) const method

/// Returns a matrix with testing instances and target variables.
/// The number of rows is the number of testing instances.
/// The number of columns is the number of target variables. 

Matrix<double> DataSet::arrange_testing_target_data(void) const
{
   const Vector<size_t> targets_indices = variables.arrange_targets_indices();

   const Vector<size_t> testing_indices = instances.arrange_testing_indices();

   return(data.arrange_submatrix(testing_indices, targets_indices));
}


// Vector<double> get_instance(const size_t&) const method

/// Returns the inputs and target values of a single instance in the data set. 
/// @param i Index of the instance. 

Vector<double> DataSet::get_instance(const size_t& i) const
{
   // Control sentence (if debug)

   #ifdef __OPENNN_DEBUG__

   const size_t instances_number = instances.get_instances_number();

   if(i >= instances_number)
   {
      std::ostringstream buffer;

      buffer << "OpenNN Exception: DataSet class.\n"
             << "Vector<double> get_instance(const size_t&) const method.\n"
             << "Index of instance must be less than number of instances.\n";

	  throw std::logic_error(buffer.str());
   }

   #endif

   // Get instance

   return(data.arrange_row(i));
}


// Vector<double> get_instance(const size_t&, const Vector<size_t>&) const method

/// Returns the inputs and target values of a single instance in the data set.
/// @param instance_index Index of the instance.
/// @param variables_indices Indices of the variables.

Vector<double> DataSet::get_instance(const size_t& instance_index, const Vector<size_t>& variables_indices) const
{
   // Control sentence (if debug)

   #ifdef __OPENNN_DEBUG__

   const size_t instances_number = instances.get_instances_number();

   if(instance_index >= instances_number)
   {
      std::ostringstream buffer;

      buffer << "OpenNN Exception: DataSet class.\n"
             << "Vector<double> get_instance(const size_t&, const Vector<size_t>&) const method.\n"
             << "Index of instance must be less than number of instances.\n";

      throw std::logic_error(buffer.str());
   }

   #endif

   // Get instance

   return(data.arrange_row(instance_index, variables_indices));
}


// Vector<double> get_variable(const size_t&) const method

/// Returns all the instances of a single variable in the data set. 
/// @param i Index of the variable. 

Vector<double> DataSet::get_variable(const size_t& i) const
{
   // Control sentence (if debug)

   #ifdef __OPENNN_DEBUG__

   const size_t variables_number = variables.get_variables_number();

   if(i >= variables_number)
   {
      std::ostringstream buffer;

      buffer << "OpenNN Exception: DataSet class.\n"
             << "Vector<double> get_variable(const size_t&) const method.\n"
             << "Index of variable must be less than number of instances.\n";

	  throw std::logic_error(buffer.str());
   }

   #endif

   // Get variable

   return(data.arrange_column(i));
}


// Vector<double> get_variable(const size_t&, const Vector<size_t>&) const method

/// Returns a given set of instances of a single variable in the data set.
/// @param variable_index Index of the variable.
/// @param instances_indices Indices of the instances.

Vector<double> DataSet::get_variable(const size_t& variable_index, const Vector<size_t>& instances_indices) const
{
   // Control sentence (if debug)

   #ifdef __OPENNN_DEBUG__

   const size_t variables_number = variables.get_variables_number();

   if(variable_index >= variables_number)
   {
      std::ostringstream buffer;

      buffer << "OpenNN Exception: DataSet class.\n"
             << "Vector<double> get_variable(const size_t&, const Vector<double>&) const method.\n"
             << "Index of variable must be less than number of instances.\n";

      throw std::logic_error(buffer.str());
   }

   #endif

   // Get variable

   return(data.arrange_column(variable_index, instances_indices));
}


// void set(void) method

/// Sets zero instances and zero variables in the data set. 

void DataSet::set(void)
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


// void set(const Matrix<double>&) method

/// Sets all variables from a data matrix.
/// @param new_data Data matrix.

void DataSet::set(const Matrix<double>& new_data)
{
   data_file_name = "";

   const size_t variables_number = new_data.get_columns_number();
   const size_t instances_number = new_data.get_rows_number();

   set(instances_number, variables_number);

   data = new_data;

   display = true;

   file_type = DAT;
}


// void set(const size_t&, const size_t&) method

/// Sets new numbers of instances and variables in the inputs targets data set. 
/// All the instances are set for training. 
/// All the variables are set as inputs. 
/// @param new_instances_number Number of instances.
/// @param new_variables_number Number of variables.

void DataSet::set(const size_t& new_instances_number, const size_t& new_variables_number)
{
    // Control sentence (if debug)

    #ifdef __OPENNN_DEBUG__

    if(new_instances_number == 0)
    {
       std::ostringstream buffer;

       buffer << "OpenNN Exception: DataSet class.\n"
              << "void set(const size_t&, const size_t&) method.\n"
              << "Number of instances must be greater than zero.\n";

       throw std::logic_error(buffer.str());
    }

    if(new_variables_number == 0)
    {
       std::ostringstream buffer;

       buffer << "OpenNN Exception: DataSet class.\n"
              << "void set(const size_t&, const size_t&) method.\n"
              << "Number of variables must be greater than zero.\n";

       throw std::logic_error(buffer.str());
    }

    #endif

   data.set(new_instances_number, new_variables_number);

   instances.set(new_instances_number);

   variables.set(new_variables_number);

   missing_values.set(new_instances_number, new_variables_number);

   display = true;

   file_type = DAT;
}


// void set(const size_t&, const size_t&, const size_t&) method

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


// void set(const DataSet& other_data_set)

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


// void set(const tinyxml2::XMLDocument&) method

/// Sets the data set members from a XML document.
/// @param data_set_document TinyXML document containing the member data.

void DataSet::set(const tinyxml2::XMLDocument& data_set_document)
{
    set_default();

   from_XML(data_set_document);
}


// void set(const std::string&) method

/// Sets the data set members by loading them from a XML file. 
/// @param file_name Data set XML file_name.

void DataSet::set(const std::string& file_name)
{
   load(file_name);
}


// void set_display(const bool&) method

/// Sets a new display value. 
/// If it is set to true messages from this class are to be displayed on the screen;
/// if it is set to false messages from this class are not to be displayed on the screen.
/// @param new_display Display value.

void DataSet::set_display(const bool& new_display)
{
   display = new_display;
}


// void set_default(void) method

/// Sets the default member values:
/// <ul>
/// <li> Display: True.
/// </ul>

void DataSet::set_default(void)
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

// void set_MPI(const DataSet*) method

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

        training_instances_number = (int)instances.count_training_instances_number();
        selection_instances_number = (int)instances.count_selection_instances_number();

        training_indices = instances.arrange_training_indices();
        selection_indices = instances.arrange_selection_indices();

        const Variables& variables = data_set->get_variables();

        inputs_indices = variables.arrange_inputs_indices_int();
        targets_indices = variables.arrange_targets_indices_int();

        inputs_number = (int)variables.count_inputs_number();
        outputs_number = (int)variables.count_targets_number();
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

        MPI_Irecv(inputs_indices.data(), (int)inputs_number, MPI_INT, rank-1, 7, MPI_COMM_WORLD, &req[2]);
        MPI_Irecv(targets_indices.data(), (int)outputs_number, MPI_INT, rank-1, 8, MPI_COMM_WORLD, &req[3]);

        MPI_Waitall(4, req, MPI_STATUS_IGNORE);
    }

    if(rank < size-1)
    {
        MPI_Request req[6];

        MPI_Isend(&inputs_number, 1, MPI_INT, rank+1, 1, MPI_COMM_WORLD, &req[0]);
        MPI_Isend(&outputs_number, 1, MPI_INT, rank+1, 2, MPI_COMM_WORLD, &req[1]);

        MPI_Isend(&training_instances_number, 1, MPI_INT, rank+1, 5, MPI_COMM_WORLD, &req[2]);
        MPI_Isend(&selection_instances_number, 1, MPI_INT, rank+1, 6, MPI_COMM_WORLD, &req[3]);

        MPI_Isend(inputs_indices.data(), (int)inputs_number, MPI_INT, rank+1, 7, MPI_COMM_WORLD, &req[4]);
        MPI_Isend(targets_indices.data(), (int)outputs_number, MPI_INT, rank+1, 8, MPI_COMM_WORLD, &req[5]);

        MPI_Waitall(6, req, MPI_STATUS_IGNORE);
    }

    size = std::min(size,(int)training_instances_number);

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
            MPI_Recv(processor_data.data()+(j*(inputs_number+outputs_number)), (int)(inputs_number+outputs_number), MPI_DOUBLE, 0, j, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

        for(int j = 0; j < selection_instances_per_processor[rank]; j++)
        {
            MPI_Recv(processor_data.data()+((j+training_instances_per_processor[rank])*(inputs_number+outputs_number)),
                      (int)(inputs_number+outputs_number), MPI_DOUBLE, 0, j+(int)training_instances_per_processor[rank], MPI_COMM_WORLD, MPI_STATUS_IGNORE);
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

                MPI_Send(instance_to_send.data(), (int)(inputs_number+outputs_number), MPI_DOUBLE, i, j, MPI_COMM_WORLD);

                training_instances_sent++;
            }
            for(int j = 0; j < selection_instances_per_processor[i]; j++)
            {
                const Vector<double> instance_to_send = data_set->get_instance(selection_indices[selection_instances_sent]);

                MPI_Send(instance_to_send.data(), (int)(inputs_number+outputs_number), MPI_DOUBLE, i, j+(int)training_instances_per_processor[i], MPI_COMM_WORLD);

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
        get_instances_pointer()->split_sequential_indices((double)training_instances_per_processor[rank], (double)selection_instances_per_processor[rank], 0.0);
    }

    free(ranks);

#else

    set(*data_set);

#endif
}


// void set_data(const Matrix<double>&) method

/// Sets a new data matrix. 
/// The number of rows must be equal to the number of instances.
/// The number of columns must be equal to the number of variables. 
/// Indices of all training, selection and testing instances and inputs and target variables do not change.
/// @param new_data Data matrix.

void DataSet::set_data(const Matrix<double>& new_data)
{
   // Control sentence (if debug)

   #ifdef __OPENNN_DEBUG__

   const size_t rows_number = new_data.get_rows_number();
   const size_t instances_number = instances.get_instances_number();

   if(rows_number != instances_number)
   {
      std::ostringstream buffer;

      buffer << "OpenNN Exception: DataSet class.\n"
             << "void set_data(const Matrix<double>&) method.\n"
             << "Number of rows (" << rows_number << ") must be equal to number of instances (" << instances_number << ").\n";

	  throw std::logic_error(buffer.str());
   }

   const size_t columns_number = new_data.get_columns_number();
   const size_t variables_number = variables.get_variables_number();

   if(columns_number != variables_number)
   {
      std::ostringstream buffer;

      buffer << "OpenNN Exception: DataSet class.\n"
             << "void set_data(const Matrix<double>&) method.\n"
             << "Number of columns (" << columns_number << ") must be equal to number of variables (" << variables_number << ").\n";

	  throw std::logic_error(buffer.str());
   }

   #endif

   // Set data
   
   data = new_data;   

   instances.set_instances_number(data.get_rows_number());
   variables.set_variables_number(data.get_columns_number());

}


// void set_data_file_name(const std::string&) method

/// Sets the name of the data file.
/// It also loads the data from that file. 
/// Moreover, it sets the variables and instances objects. 
/// @param new_data_file_name Name of the file containing the data.

void DataSet::set_data_file_name(const std::string& new_data_file_name)
{   
   data_file_name = new_data_file_name;
}


// void set_file_type(const FileType&) method

/// Sets the file type.

void DataSet::set_file_type(const DataSet::FileType& new_file_type)
{
    file_type = new_file_type;
}


// void set_file_type(const std::string&) method

/// Sets the file type from a string

void DataSet::set_file_type(const std::string& new_file_type)
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
    else
    {
        std::ostringstream buffer;

        buffer << "OpenNN Exception: DataSet class.\n"
               << "void set_file_type(const std::string&) method.\n"
               << "Unknown file type.";

        throw std::logic_error(buffer.str());
    }
}


// void set_header_line(const bool&) method

/// Sets if the data file contains a header with the names of the variables.

void DataSet::set_header_line(const bool& new_header_line)
{
    header_line = new_header_line;
}


// void set_rows_label(const bool&) method

/// Sets if the data file contains rows label.

void DataSet::set_rows_label(const bool& new_rows_label)
{
    rows_label = new_rows_label;
}


// void set_separator(const Separator&) method

/// Sets a new separator.
/// @param new_separator Separator value.

void DataSet::set_separator(const Separator& new_separator)
{
    separator = new_separator;
}


// void set_separator(const std::string&) method

/// Sets a new separator from a string.
/// @param new_separator String with the separator value.

void DataSet::set_separator(const std::string& new_separator)
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
        std::ostringstream buffer;

        buffer << "OpenNN Exception: DataSet class.\n"
               << "void set_separator(const std::string&) method.\n"
               << "Unknown separator: " << new_separator << ".\n";

        throw std::logic_error(buffer.str());
    }
}


// void set_missing_values_label(const std::string&) method

/// Sets a new label for the missing values.
/// @param new_missing_values_label Label for the missing values.

void DataSet::set_missing_values_label(const std::string& new_missing_values_label)
{
    // Control sentence (if debug)

    #ifdef __OPENNN_DEBUG__

    if(get_trimmed(new_missing_values_label).empty())
    {
       std::ostringstream buffer;

       buffer << "OpenNN Exception: DataSet class.\n"
              << "void set_missing_values_label(const std::string&) method.\n"
              << "Missing values label cannot be empty.\n";

       throw std::logic_error(buffer.str());
    }

    #endif


    missing_values_label = new_missing_values_label;
}


// void set_lags_number(const size_t&)

/// Sets a new number of lags to be defined for a time series prediction application.
/// When loading the data file, the time series data will be modified according to this number.
/// @param new_lags_number Number of lags (x-1, ..., x-l) to be used.

void DataSet::set_lags_number(const size_t& new_lags_number)
{
    lags_number = new_lags_number;
}


// void set_steps_ahead_number(const size_t&)

/// Sets a new number of steps ahead to be defined for a time series prediction application.
/// When loading the data file, the time series data will be modified according to this number.
/// @param new_steps_ahead_number Number of steps ahead to be used.

void DataSet::set_steps_ahead_number(const size_t& new_steps_ahead_number)
{
    steps_ahead = new_steps_ahead_number;
}


// void set_autoassociation(const size_t&)

/// Sets a new autoasociation flag.
/// If the new value is true, the data will be processed for association when loading.
/// That is, the data file will contain the input data. The target data will be created as being equal to the input data.
/// If the association value is set to false, the data from the file will not be processed.
/// @param new_autoassociation Association value.

void DataSet::set_autoassociation(const bool& new_autoassociation)
{
    association = new_autoassociation;
}


// void set_learning_task(const ProjectType&)

/// Sets a new project type.
/// @param new_learning_task New project type.

void DataSet::set_learning_task(const DataSet::ProjectType& new_learning_task)
{
    learning_task = new_learning_task;
}


// void set_learning_task(const std::string&)

/// Sets a new project type from a string.
/// @param new_learning_task New project type.

void DataSet::set_learning_task(const std::string& new_learning_task)
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
        std::ostringstream buffer;

        buffer << "OpenNN Exception: DataSet class.\n"
               << "void set_learning_task(const std::string&) method.\n"
               << "Not known project type.\n";

        throw std::logic_error(buffer.str());
    }
}


// void set_angular_variables(const Vector<size_t>&)

/// Sets the indices of those variables which represent angles.
/// @param new_angular_variables Indices of angular variables.

void DataSet::set_angular_variables(const Vector<size_t>& new_angular_variables)
{
    angular_variables = new_angular_variables;
}


// void set_angular_units(AngularUnits&)

/// Sets the units of the angular variables (Radians or Degrees).

void DataSet::set_angular_units(AngularUnits& new_angular_units)
{
    angular_units = new_angular_units;
}



// void set_instances_number(const size_t&) method

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


// void set_variables_number(const size_t&) method

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


// void set_instance(const size_t&, const Vector<double>&)

/// Sets new inputs and target values of a single instance in the data set. 
/// @param instance_index Index of the instance. 
/// @param instance New inputs and target values of the instance.

void DataSet::set_instance(const size_t& instance_index, const Vector<double>& instance)
{
   // Control sentence (if debug)

   #ifdef __OPENNN_DEBUG__

   const size_t instances_number = instances.get_instances_number();

   if(instance_index >= instances_number)
   {
      std::ostringstream buffer;

      buffer << "OpenNN Exception: DataSet class.\n"
             << "void set_instance(const size_t&, const Vector<double>&) method.\n"
             << "Index of instance must be less than number of instances.\n";

	  throw std::logic_error(buffer.str());
   }

   const size_t size = instance.size();
   const size_t variables_number = variables.get_variables_number();

   if(size != variables_number)
   {
      std::ostringstream buffer;

      buffer << "OpenNN Exception: DataSet class.\n"
             << "void set_instance(const size_t&, const Vector<double>&) method.\n"
             << "Size (" << size << ") must be equal to number of variables (" << variables_number << ").\n";

	  throw std::logic_error(buffer.str());
   } 

   #endif

   // Set instance

   data.set_row(instance_index, instance);
}


// void add_instance(const Vector<double>&) method

/// Adds a new instance to the data matrix from a vector of real numbers.
/// The size of that vector must be equal to the number of variables. 
/// Note that resizing is here necessary and therefore computationally expensive. 
/// All instances are also set for training. 
/// @param instance Input and target values of the instance to be added. 

void DataSet::add_instance(const Vector<double>& instance)
{
   // Control sentence (if debug)

   #ifdef __OPENNN_DEBUG__

   const size_t size = instance.size();
   const size_t variables_number = variables.get_variables_number();

   if(size != variables_number)
   {
      std::ostringstream buffer;

      buffer << "OpenNN Exception: DataSet class.\n"
             << "void add_instance(const Vector<double>&) method.\n"
             << "Size of instance must be equal to number of variables.\n";

	  throw std::logic_error(buffer.str());
   }

   #endif

   const size_t instances_number = instances.get_instances_number();

   data.append_row(instance);

   instances.set(instances_number+1);
}


// void subtract_instance(size_t) method

/// Substracts the inputs-targets instance with a given index from the data set.
/// All instances are also set for training. 
/// Note that resizing is here necessary and therefore computationally expensive. 
/// @param instance_index Index of instance to be removed. 

void DataSet::subtract_instance(const size_t& instance_index)
{
    const size_t instances_number = instances.get_instances_number();

   // Control sentence (if debug)

   #ifdef __OPENNN_DEBUG__

   if(instance_index >= instances_number)
   {
      std::ostringstream buffer;

      buffer << "OpenNN Exception: DataSet class.\n"
             << "void subtract_instance(size_t) method.\n"
             << "Index of instance must be less than number of instances.\n";

	  throw std::logic_error(buffer.str());
   }

   #endif

   data.subtract_row(instance_index);

   instances.set_instances_number(instances_number-1);

}


// void append_variable(const Vector<double>&) method

/// Appends a variable with given values to the data matrix.
/// @param variable Vector of values. The size must be equal to the number of instances. 

void DataSet::append_variable(const Vector<double>& variable)
{
   // Control sentence (if debug)

   #ifdef __OPENNN_DEBUG__

   const size_t size = variable.size();
   const size_t instances_number = instances.get_instances_number();

   if(size != instances_number)
   {
      std::ostringstream buffer;

      buffer << "OpenNN Exception: DataSet class.\n"
             << "void append_variable(const Vector<double>&) method.\n"
             << "Size of variable must be equal to number of instances.\n";

	  throw std::logic_error(buffer.str());
   }

   #endif

   const size_t variables_number = variables.get_variables_number();

   data.append_column(variable);

   Matrix<double> new_data(data);

   const size_t new_variables_number = variables_number + 1;

   set_variables_number(new_variables_number);

   set_data(new_data);
}


// void subtract_variable(size_t) method

/// Removes a variable with given index from the data matrix.
/// @param variable_index Index of variable to be subtracted. 

void DataSet::subtract_variable(const size_t& variable_index)
{
   const size_t variables_number = variables.get_variables_number();

   // Control sentence (if debug)

   #ifdef __OPENNN_DEBUG__

   if(variable_index >= variables_number)
   {
      std::ostringstream buffer;

      buffer << "OpenNN Exception: DataSet class.\n"
             << "void subtract_variable(size_t) method.\n"
             << "Index of variable must be less than number of variables.\n";

	  throw std::logic_error(buffer.str());
   }

   #endif

   data.subtract_column(variable_index);

   Matrix<double> new_data(data);

   const size_t new_variables_number = variables_number - 1;

   set_variables_number(new_variables_number);

   set_data(new_data);
}


// Vector<size_t> unuse_constant_variables(void) method

/// Removes the input of target indices of that variables with zero standard deviation.
/// It might change the size of the vectors containing the inputs and targets indices. 

Vector<size_t> DataSet::unuse_constant_variables(void)
{
   const size_t variables_number = variables.get_variables_number();

   // Control sentence (if debug)

   #ifdef __OPENNN_DEBUG__

   if(variables_number == 0)
   {
      std::ostringstream buffer;

      buffer << "OpenNN Exception: DataSet class.\n"
             << "Vector<size_t> unuse_constant_variables(void) method.\n"
             << "Number of variables is zero.\n";

      throw std::logic_error(buffer.str());
   }

   #endif

   const Vector< Statistics<double> > statistics = data.calculate_statistics();

   Vector<size_t> constant_variables;

   for(size_t i = 0; i < variables_number; i++)
   {
      if(variables.get_use(i) ==  Variables::Input && statistics[i].standard_deviation < 1.0e-6)
      {
         variables.set_use(i, Variables::Unused);
         constant_variables.push_back(i);
	  }
   }

   return(constant_variables);
}


// Vector<size_t> unuse_repeated_instances(void) method

/// Removes the training, selection and testing indices of that instances which are repeated in the data matrix.
/// It might change the size of the vectors containing the training, selection and testing indices. 

Vector<size_t> DataSet::unuse_repeated_instances(void)
{
    const size_t instances_number = instances.get_instances_number();

    // Control sentence (if debug)

    #ifdef __OPENNN_DEBUG__

    if(instances_number == 0)
    {
       std::ostringstream buffer;

       buffer << "OpenNN Exception: DataSet class.\n"
              << "Vector<size_t> unuse_repeated_indices(void) method.\n"
              << "Number of instances is zero.\n";

       throw std::logic_error(buffer.str());
    }

    #endif

    Vector<size_t> repeated_instances;

	Vector<double> instance_i;
	Vector<double> instance_j;    

    int i = 0;

    #pragma omp parallel for private(i, instance_i, instance_j) schedule(dynamic)

    for(i = 0; i < (int)instances_number; i++)
	{
	   instance_i = get_instance(i);

       for(size_t j = i+1; j < instances_number; j++)
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


// Vector<size_t> unuse_non_significant_inputs(void)

/// Unuses those binary inputs whose positives does not correspond to any positive in the target variables.

Vector<size_t> DataSet::unuse_non_significant_inputs(void)
{
    const Vector<size_t> inputs_indices = get_variables_pointer()->arrange_inputs_indices();
    const size_t inputs_number = inputs_indices.size();

    const size_t target_index = get_variables_pointer()->arrange_targets_indices()[0];

    const size_t instances_number = get_instances_pointer()->count_used_instances_number();

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



// Vector<Histogram> calculate_data_histograms(const size_t&) const method

/// Returns a histogram for each variable with a given number of bins. 
/// The default number of bins is 10.
/// The format is a vector of subvectors of subsubvectors.
/// The size of the vector is the number of variables. 
/// The size of the subvectors is 2 (centers and frequencies).
/// The size of the subsubvectors is the number of bins.
/// @param bins_number Number of bins.

Vector< Histogram<double> > DataSet::calculate_data_histograms(const size_t& bins_number) const
{
   const size_t used_variables_number = variables.count_used_variables_number();
   const Vector<size_t> used_variables_indices = variables.arrange_used_indices();
   const size_t used_instances_number = instances.count_used_instances_number();
   const Vector<size_t> used_instances_indices = instances.arrange_used_indices();

   const Vector< Vector<size_t> > missing_indices = missing_values.arrange_missing_indices();

   Vector< Histogram<double> > histograms(used_variables_number);

   Vector<double> column(used_instances_number);

   int i = 0;

   #pragma omp parallel for private(i, column) shared(histograms)

   for(i = 0; i < (int)used_variables_number; i++)
   {
       column = data.arrange_column(used_variables_indices[i], used_instances_indices);

       if (column.is_binary())
       {
           histograms[i] = column.calculate_histogram_binary();
       }
       else
       {
           histograms[i] = column.calculate_histogram(bins_number);
//           histograms[i] = column.calculate_histogram_missing_values(missing_indices[i], bins_number);
       }
   }

   return(histograms);
}


// Vector<Histogram> calculate_targets_histograms(const size_t&) const method

/// Returns a histogram for each target variable with a given number of bins.
/// The default number of bins is 10.
/// The format is a vector of subvectors of subsubvectors.
/// The size of the vector is the number of variables.
/// The size of the subvectors is 2 (centers and frequencies).
/// The size of the subsubvectors is the number of bins.
/// @param bins_number Number of bins.

Vector< Histogram<double> > DataSet::calculate_targets_histograms(const size_t& bins_number) const
{
   const size_t targets_number = variables.count_targets_number();

   const Vector<size_t> targets_indices = variables.arrange_targets_indices();

   const size_t used_instances_number = instances.count_used_instances_number();
   const Vector<size_t> used_instances_indices = instances.arrange_used_indices();

   const Vector< Vector<size_t> > missing_indices = missing_values.arrange_missing_indices();

   Vector< Histogram<double> > histograms(targets_number);

   Vector<double> column(used_instances_number);

   for(size_t i = 0; i < targets_number; i++)
   {
       column = data.arrange_column(targets_indices[i], used_instances_indices);

       histograms[i] = column.calculate_histogram_missing_values(missing_indices[i], bins_number);
   }

   return(histograms);
}


// Vector<double> calculate_box_and_whiskers(void) const

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

Vector< Vector<double> > DataSet::calculate_box_plots(void) const
{
    const size_t variables_number = variables.count_used_variables_number();
    const Vector<size_t> variables_indices = variables.arrange_used_indices();

    const size_t instances_number = instances.count_used_instances_number();
    const Vector<size_t> instances_indices = instances.arrange_used_indices();

    const Vector< Vector<size_t> > missing_indices = missing_values.arrange_missing_indices();

    Vector< Vector<double> > box_plots;
    box_plots.set(variables_number);

    Vector<double> column(instances_number);

#pragma omp parallel for private(column)
    for(int i = 0; i < (int)variables_number; i++)
    {
        column = data.arrange_column(variables_indices[i], instances_indices);

        box_plots[i] = column.calculate_box_plots_missing_values(missing_indices[i]);
    }

    return(box_plots);
}


// size_t calculate_training_negatives(const size_t&) const method

/// Counts the number of negatives of the selected target in the training data.
/// @param target_index Index of the target to evaluate.

size_t DataSet::calculate_training_negatives(const size_t& target_index) const
{
    size_t negatives = 0;

    const size_t training_instances_number = instances.count_training_instances_number();

    Vector<size_t> training_indices = instances.arrange_training_indices();

    size_t training_index;

    for(size_t i = 0; i < training_instances_number; i++)
    {
        training_index = training_indices[i];

        if(data(training_index, target_index) == 0.0)
        {
            negatives++;
        }
        else if(data(training_index, target_index) != 1.0)
        {
            std::ostringstream buffer;

           buffer << "OpenNN Exception: DataSet class.\n"
                  << "size_t calculate_training_negatives(const size_t&) const method.\n"
                  << "Training instance is neither a positive nor a negative: " << data(training_index, target_index) << std::endl;

           throw std::logic_error(buffer.str());
        }
    }

    return(negatives);
}


// size_t calculate_selection_negatives(const size_t&) const method

/// Counts the number of negatives of the selected target in the selection data.
/// @param target_index Index of the target to evaluate.

size_t DataSet::calculate_selection_negatives(const size_t& target_index) const
{
    size_t negatives = 0;

    const size_t selection_instances_number = instances.count_selection_instances_number();

    Vector<size_t> selection_indices = instances.arrange_selection_indices();

    size_t selection_index;

    for(size_t i = 0; i < selection_instances_number; i++)
    {
        selection_index = selection_indices[i];

        if(data(selection_index, target_index) == 0.0)
        {
            negatives++;
        }
        else if(data(selection_index, target_index) != 1.0)
        {
            std::ostringstream buffer;

           buffer << "OpenNN Exception: DataSet class.\n"
                  << "size_t calculate_selection_negatives(const size_t&) const method.\n"
                  << "Selection instance is neither a positive nor a negative: " << data(selection_index, target_index) << std::endl;

           throw std::logic_error(buffer.str());
        }
    }

    return(negatives);
}


// size_t calculate_testing_negatives(const size_t&) const method

/// Counts the number of negatives of the selected target in the testing data.
/// @param target_index Index of the target to evaluate.

size_t DataSet::calculate_testing_negatives(const size_t& target_index) const
{
    size_t negatives = 0;

    const size_t testing_instances_number = instances.count_testing_instances_number();

    Vector<size_t> testing_indices = instances.arrange_testing_indices();

    size_t testing_index;

    for(size_t i = 0; i < testing_instances_number; i++)
    {
        testing_index = testing_indices[i];

        if(data(testing_index, target_index) == 0.0)
        {
            negatives++;
        }
        else if(data(testing_index, target_index) != 1.0)
        {
            std::ostringstream buffer;

           buffer << "OpenNN Exception: DataSet class.\n"
                  << "size_t calculate_selection_negatives(const size_t&) const method.\n"
                  << "Testing instance is neither a positive nor a negative: " << data(testing_index, target_index) << std::endl;

           throw std::logic_error(buffer.str());
        }
    }

    return(negatives);
}


// Vector< Vector<double> > calculate_data_statistics(void) const method

/// Returns a vector of vectors containing some basic statistics of all the variables in the data set.
/// The size of this vector is four. The subvectors are:
/// <ul>
/// <li> Minimum.
/// <li> Maximum.
/// <li> Mean.
/// <li> Standard deviation.
/// </ul> 

Vector< Statistics<double> > DataSet::calculate_data_statistics(void) const
{
    const Vector< Vector<size_t> > missing_indices = missing_values.arrange_missing_indices();

    return(data.calculate_statistics_missing_values(missing_indices));
}


// Vector < Vector<double> > calculate_data_shape_parameters(void) const method

/// Returns a vector fo subvectors containing the shape parameters for all the variables in the data set.
/// The size of this vector is 2. The subvectors are:
/// <ul>
/// <li> Asymmetry.
/// <li> Kurtosis.
/// </ul>

Vector< Vector<double> > DataSet::calculate_data_shape_parameters(void) const
{
    const Vector< Vector<size_t> > missing_indices = missing_values.arrange_missing_indices();

    return(data.calculate_shape_parameters_missing_values(missing_indices));
}


// Matrix<double> calculate_data_statistics_matrix(void) const method

/// Returns all the variables statistics from a single matrix.
/// The number of rows is the number of used variables.
/// The number of columns is five (minimum, maximum, mean and standard deviation).

Matrix<double> DataSet::calculate_data_statistics_matrix(void) const
{
    const Vector< Vector<size_t> > missing_indices = missing_values.arrange_missing_indices();

    const Vector<size_t> used_variables_indices = variables.arrange_used_indices();
    const Vector<size_t> used_instances_indices = instances.arrange_used_indices();

    const size_t variables_number = used_variables_indices.size();//variables.count_used_variables_number();

    Matrix<double> data_statistics_matrix(variables_number, 4);

    for(size_t i = 0; i < variables_number; i++)
    {
        const size_t variable_index = used_variables_indices[i];

        const Vector<double> variable_data = data.arrange_column(variable_index, used_instances_indices);

        const Statistics<double> data_statistics = variable_data.calculate_statistics_missing_values(missing_indices[variable_index]);

        data_statistics_matrix.set_row(i, data_statistics.to_vector());
    }

    return(data_statistics_matrix);
}

// Matrix<double> calculate_positives_data_statistics_matrix(void) const method

/// Calculate the statistics of the instances with positive targets in binary classification problems.

Matrix<double> DataSet::calculate_positives_data_statistics_matrix(void) const
{
#ifdef __OPENNN_DEBUG__

    const size_t targets_number = variables.count_targets_number();

    if(targets_number != 1)
    {
        std::ostringstream buffer;

        buffer << "OpenNN Exception: DataSet class.\n"
               << "Matrix<double> calculate_positives_data_statistics_matrix(void) const method.\n"
               << "Number of targets muste be 1.\n";

        throw std::logic_error(buffer.str());
    }
#endif

    const size_t target_index = variables.arrange_targets_indices()[0];

    const Vector<size_t> used_instances_indices = instances.arrange_used_indices();

    const Vector<double> targets = data.arrange_column(target_index, used_instances_indices);

#ifdef __OPENNN_DEBUG__

    if(!targets.is_binary())
    {
        std::ostringstream buffer;

        buffer << "OpenNN Exception: DataSet class.\n"
               << "Matrix<double> calculate_positives_data_statistics_matrix(void) const method.\n"
               << "Targets vector must be binary.\n";

        throw std::logic_error(buffer.str());
    }
#endif

    const Vector<size_t> inputs_variables_indices = variables.arrange_inputs_indices();

    const Vector< Vector<size_t> > missing_indices = missing_values.arrange_missing_indices();

    const size_t inputs_number = inputs_variables_indices.size();

    const Vector<size_t> positives_used_instances_indices = used_instances_indices.arrange_subvector(targets.calculate_equal_than_indices(1.0));

    Matrix<double> data_statistics_matrix(inputs_number, 4);

    for(size_t i = 0; i < inputs_number; i++)
    {
        const size_t variable_index = inputs_variables_indices[i];

        const Vector<double> variable_data = data.arrange_column(variable_index, positives_used_instances_indices);

        const Statistics<double> data_statistics = variable_data.calculate_statistics_missing_values(missing_indices[variable_index]);

        data_statistics_matrix.set_row(i, data_statistics.to_vector());
    }
    return data_statistics_matrix;
}

// Matrix<double> calculate_negatives_data_statistics_matrix(void) const method

/// Calculate the statistics of the instances with neagtive targets in binary classification problems.

Matrix<double> DataSet::calculate_negatives_data_statistics_matrix(void) const
{
#ifdef __OPENNN_DEBUG__

    const size_t targets_number = variables.count_targets_number();

    if(targets_number != 1)
    {
        std::ostringstream buffer;

        buffer << "OpenNN Exception: DataSet class.\n"
               << "Matrix<double> calculate_positives_data_statistics_matrix(void) const method.\n"
               << "Number of targets muste be 1.\n";

        throw std::logic_error(buffer.str());
    }
#endif

    const size_t target_index = variables.arrange_targets_indices()[0];

    const Vector<size_t> used_instances_indices = instances.arrange_used_indices();

    const Vector< Vector<size_t> > missing_indices = missing_values.arrange_missing_indices();

    const Vector<double> targets = data.arrange_column(target_index, used_instances_indices);

#ifdef __OPENNN_DEBUG__

    if(!targets.is_binary())
    {
        std::ostringstream buffer;

        buffer << "OpenNN Exception: DataSet class.\n"
               << "Matrix<double> calculate_positives_data_statistics_matrix(void) const method.\n"
               << "Targets vector must be binary.\n";

        throw std::logic_error(buffer.str());
    }
#endif

    const Vector<size_t> inputs_variables_indices = variables.arrange_inputs_indices();

    const size_t inputs_number = inputs_variables_indices.size();

    const Vector<size_t> negatives_used_instances_indices = used_instances_indices.arrange_subvector(targets.calculate_equal_than_indices(0.0));

    Matrix<double> data_statistics_matrix(inputs_number, 4);

    for(size_t i = 0; i < inputs_number; i++)
    {
        const size_t variable_index = inputs_variables_indices[i];

        const Vector<double> variable_data = data.arrange_column(variable_index, negatives_used_instances_indices);

        const Statistics<double> data_statistics = variable_data.calculate_statistics_missing_values(missing_indices[variable_index]);

        data_statistics_matrix.set_row(i, data_statistics.to_vector());
    }
    return data_statistics_matrix;
}

// Matrix<double> calculate_data_shape_parameters_matrix(void) const method

/// Returns all the variables shape parameters from a single matrix.
/// The number of rows is the number of used variables.
/// The number of columns is two (asymmetry, kurtosis).

Matrix<double> DataSet::calculate_data_shape_parameters_matrix(void) const
{
    const Vector< Vector<size_t> > missing_indices = missing_values.arrange_missing_indices();

    const Vector<size_t> used_variables_indices = variables.arrange_used_indices();
    const Vector<size_t> used_instances_indices = instances.arrange_used_indices();

    const Vector< Vector<size_t> > used_missing_indices = missing_indices.arrange_subvector(used_variables_indices);

    const size_t variables_number = variables.count_used_variables_number();

    Matrix<double> data_shape_parameters_matrix(variables_number, 2);

    for(size_t i = 0; i < variables_number; i++)
    {
        const size_t variable_index = used_variables_indices[i];

        const Vector<double> variable_data = data.arrange_column(variable_index, used_instances_indices);

        const Vector<double> shape_parameters = variable_data.calculate_shape_parameters_missing_values(used_missing_indices[variable_index]);

        data_shape_parameters_matrix.set_row(i, shape_parameters);
    }

    return(data_shape_parameters_matrix);
}


// Vector< Vector<double> > calculate_training_instances_statistics(void) const method

/// Returns a vector of vectors containing some basic statistics of all variables on the training instances.
/// The size of this vector is two. The subvectors are:
/// <ul>
/// <li> Training data minimum.
/// <li> Training data maximum.
/// <li> Training data mean.
/// <li> Training data standard deviation.
/// </ul> 

Vector< Statistics<double> > DataSet::calculate_training_instances_statistics(void) const
{
   const Vector<size_t> training_indices = instances.arrange_training_indices();

   const Vector< Vector<size_t> > missing_indices = missing_values.arrange_missing_indices();

   return(data.calculate_rows_statistics_missing_values(training_indices, missing_indices));
}


// Vector< Vector<double> > calculate_selection_instances_statistics(void) const method

/// Returns a vector of vectors containing some basic statistics of all variables on the selection instances.
/// The size of this vector is two. The subvectors are:
/// <ul>
/// <li> Selection data minimum.
/// <li> Selection data maximum.
/// <li> Selection data mean.
/// <li> Selection data standard deviation.
/// </ul>

Vector< Statistics<double> > DataSet::calculate_selection_instances_statistics(void) const
{
    const Vector<size_t> selection_indices = instances.arrange_selection_indices();

    const Vector< Vector<size_t> > missing_indices = missing_values.arrange_missing_indices();

   return(data.calculate_rows_statistics_missing_values(selection_indices, missing_indices));
}


// Vector< Vector<double> > calculate_testing_instances_statistics(void) const method

/// Returns a vector of vectors containing some basic statistics of all variables on the testing instances.
/// The size of this vector is five. The subvectors are:
/// <ul>
/// <li> Testing data minimum.
/// <li> Testing data maximum.
/// <li> Testing data mean.
/// <li> Testing data standard deviation.
/// </ul>

Vector< Statistics<double> > DataSet::calculate_testing_instances_statistics(void) const
{
    const Vector<size_t> testing_indices = instances.arrange_testing_indices();

    const Vector< Vector<size_t> > missing_indices = missing_values.arrange_missing_indices();

   return(data.calculate_rows_statistics_missing_values(testing_indices, missing_indices));
}


// Vector< Vector<double> > calculate_training_instances_shape_parameters(void) const method

/// Returns a vector of vectors containing some shape parameters of all variables on the training instances.
/// The size of this vector is two. The subvectors are:
/// <ul>
/// <li> Training data asymmetry.
/// <li> Training data kurtosis.
/// </ul>

Vector< Vector<double> > DataSet::calculate_training_instances_shape_parameters(void) const
{
   const Vector<size_t> training_indices = instances.arrange_training_indices();

   const Vector< Vector<size_t> > missing_indices = missing_values.arrange_missing_indices();

   return(data.calculate_rows_shape_parameters_missing_values(training_indices, missing_indices));
}


// Vector< Vector<double> > calculate_selection_instances_shape_parameters(void) const method

/// Returns a vector of vectors containing some shape parameters of all variables on the selection instances.
/// The size of this vector is five. The subvectors are:
/// <ul>
/// <li> Selection data asymmetry.
/// <li> Selection data kurtosis.
/// </ul>

Vector< Vector<double> > DataSet::calculate_selection_instances_shape_parameters(void) const
{
    const Vector<size_t> selection_indices = instances.arrange_selection_indices();

    const Vector< Vector<size_t> > missing_indices = missing_values.arrange_missing_indices();

   return(data.calculate_rows_shape_parameters_missing_values(selection_indices, missing_indices));
}


// Vector< Vector<double> > calculate_testing_instances_shape_parameters(void) const method

/// Returns a vector of vectors containing some shape parameters of all variables on the testing instances.
/// The size of this vector is five. The subvectors are:
/// <ul>
/// <li> Testing data asymmetry.
/// <li> Testing data kurtosis.
/// </ul>

Vector< Vector<double> > DataSet::calculate_testing_instances_shape_parameters(void) const
{
    const Vector<size_t> testing_indices = instances.arrange_testing_indices();

    const Vector< Vector<size_t> > missing_indices = missing_values.arrange_missing_indices();

   return(data.calculate_rows_shape_parameters_missing_values(testing_indices, missing_indices));
}



// Vector< Statistics<double> > calculate_inputs_statistics(void) const method

/// Returns a vector of vectors with some basic statistics of the input variables on all instances. 
/// The size of this vector is five. The subvectors are:
/// <ul>
/// <li> Input variables minimum.
/// <li> Input variables maximum.
/// <li> Input variables mean.
/// <li> Input variables standard deviation.
/// </ul> 

Vector< Statistics<double> > DataSet::calculate_inputs_statistics(void) const
{
    const Vector<size_t> inputs_indices = variables.arrange_inputs_indices();

    Vector< Vector<size_t> > missing_indices = missing_values.arrange_missing_indices();

    const Vector<size_t> unused_instances_indices = instances.arrange_unused_indices();

    for (size_t i = 0; i < unused_instances_indices.size(); i++)
    {
        for (size_t j = 0; j < missing_indices.size(); j++)
        {
            missing_indices[j].push_back(unused_instances_indices[i]);
        }
    }

   return(data.calculate_columns_statistics_missing_values(inputs_indices, missing_indices));
}


// Vector< Statistics<double> > calculate_targets_statistics(void) const method

/// Returns a vector of vectors with some basic statistics of the target variables on all instances. 
/// The size of this vector is five. The subvectors are:
/// <ul>
/// <li> Target variables minimum.
/// <li> Target variables maximum.
/// <li> Target variables mean.
/// <li> Target variables standard deviation.
/// </ul> 

Vector< Statistics<double> > DataSet::calculate_targets_statistics(void) const
{
   const Vector<size_t> targets_indices = variables.arrange_targets_indices();

   Vector< Vector<size_t> > missing_indices = missing_values.arrange_missing_indices();

   const Vector<size_t> unused_instances_indices = instances.arrange_unused_indices();

   for (size_t i = 0; i < unused_instances_indices.size(); i++)
   {
       for (size_t j = 0; j < missing_indices.size(); j++)
       {
           missing_indices[j].push_back(unused_instances_indices[i]);
       }
   }

   return(data.calculate_columns_statistics_missing_values(targets_indices, missing_indices));
}


// Vector<double> calculate_training_target_data_mean(void) const method

/// Returns the mean values of the target variables on the training instances. 

Vector<double> DataSet::calculate_training_target_data_mean(void) const
{
   const Vector<size_t> targets_indices = variables.arrange_targets_indices();

   const Vector<size_t> training_indices = instances.arrange_training_indices();

   const Vector< Vector<size_t> > missing_indices = missing_values.arrange_missing_indices();

   return(data.calculate_mean_missing_values(training_indices, targets_indices, missing_indices));
}


// Vector<double> calculate_selection_target_data_mean(void) const method

/// Returns the mean values of the target variables on the selection instances. 

Vector<double> DataSet::calculate_selection_target_data_mean(void) const
{
    const Vector<size_t> targets_indices = variables.arrange_targets_indices();

   const Vector<size_t> selection_indices = instances.arrange_selection_indices();

   const Vector< Vector<size_t> > missing_indices = missing_values.arrange_missing_indices();

   return(data.calculate_mean_missing_values(selection_indices, targets_indices, missing_indices));
}


// Vector<double> calculate_testing_target_data_mean(void) const method

/// Returns the mean values of the target variables on the testing instances. 

Vector<double> DataSet::calculate_testing_target_data_mean(void) const
{
   const Vector<size_t> testing_indices = instances.arrange_testing_indices();

   const Vector<size_t> targets_indices = variables.arrange_targets_indices();

   const Vector< Vector<size_t> > missing_indices = missing_values.arrange_missing_indices();

   return(data.calculate_mean_missing_values(testing_indices, targets_indices, missing_indices));
}


// Matrix<double> calculate_linear_correlations(void) const method

/// Calculates the linear correlations between all outputs and all inputs.
/// It returns a matrix with number of rows the targets number and number of columns the inputs number.
/// Each element contains the linear correlation between a single target and a single output.

Matrix<double> DataSet::calculate_linear_correlations(void) const
{
   const size_t inputs_number = variables.count_inputs_number();
   const size_t targets_number = variables.count_targets_number();

   const Vector<size_t> input_indices = variables.arrange_inputs_indices();
   const Vector<size_t> target_indices = variables.arrange_targets_indices();

   size_t input_index;
   size_t target_index;

   const size_t instances_number = instances.get_instances_number();

   Vector<double> input_variable(instances_number);
   Vector<double> target_variable(instances_number);

   Matrix<double> linear_correlations(inputs_number, targets_number);

#ifndef __OPENNN_MPI__
#pragma omp parallel for private(input_index, input_variable, target_index, target_variable)
#endif
   for(int i = 0; i < (int)inputs_number; i++)
   {
       for(int j = 0; j < (int)targets_number; j++)
       {
           input_index = input_indices[i];

           input_variable = data.arrange_column(input_index);

           target_index = target_indices[j];

           target_variable = data.arrange_column(target_index);

           linear_correlations(i,j) = input_variable.calculate_linear_correlation(target_variable);
       }
   }

   return(linear_correlations);
}


// Matrix<double> calculate_covariance_matrix(void) const method

/// Returns the covariance matrix for the input data set.
/// The number of rows of the matrix is the number of inputs.
/// The number of columns of the matrix is the number of inputs.

Matrix<double> DataSet::calculate_covariance_matrix(void) const
{
    const Vector<size_t> inputs_indices = variables.arrange_inputs_indices();
    const Vector<size_t> used_instances_indices = instances.arrange_used_indices();

    const size_t inputs_number = variables.count_inputs_number();

    Matrix<double> covariance_matrix(inputs_number, inputs_number, 0.0);

#pragma omp parallel for

    for(int i = 0; i < (int)inputs_number; i++)
    {
        const size_t first_input_index = inputs_indices[i];

        const Vector<double> first_input_data = data.arrange_column(first_input_index, used_instances_indices);

        for(size_t j = 0; j < inputs_number; j++)
        {
            const size_t second_input_index = inputs_indices[j];

            const Vector<double> second_input_data = data.arrange_column(second_input_index, used_instances_indices);

            covariance_matrix(i,j) = first_input_data.calculate_covariance(second_input_data);
            covariance_matrix(j,i) = covariance_matrix(i,j);
        }
    }

    return (covariance_matrix);
}


// Matrix<double> perform_principal_components_analysis(const double& = 0.0) mehtod

/// Performs the principal components analysis of the inputs.
/// It returns a matrix containing the principal components arranged in rows.
/// This method deletes the unused instances of the original data set.
/// @param minimum_explained_variance Minimum percentage of variance used to select a principal component.

Matrix<double> DataSet::perform_principal_components_analysis(const double& minimum_explained_variance)
{
    //const Vector<size_t> inputs_indices = variables.arrange_inputs_indices();

    // Subtract off the mean

    subtract_input_data_mean();

    // Calculate covariance matrix

    const Matrix<double> covariance_matrix = calculate_covariance_matrix();

    // Calculate eigenvectors

    const Matrix<double> eigenvectors = covariance_matrix.calculate_eigenvectors();

    // Calculate eigenvalues

    const Matrix<double> eigenvalues = covariance_matrix.calculate_eigenvalues();

    // Calculate explained variance

    const Vector<double> explained_variance = eigenvalues.arrange_column(0).calculate_explained_variance();

    // Sort principal components

    const Vector<size_t> sorted_principal_components_indices = explained_variance.sort_greater_indices();

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

        principal_components.set_row(i, eigenvectors.arrange_column(index));
    }

    // Return feature matrix

    return principal_components.arrange_submatrix_rows(principal_components_indices);
}


// Matrix<double> perform_principal_components_analysis(const Matrix<double>&, const Matrix<double>&, const Vector<double>&, const double& = 0.0) mehtod

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

    subtract_input_data_mean();

    // Calculate eigenvectors

    const Matrix<double> eigenvectors = covariance_matrix.calculate_eigenvectors();

    // Sort principal components

    Vector<size_t> sorted_principal_components_indices = explained_variance.sort_greater_indices();

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

        principal_components.set_row(i, eigenvectors.arrange_column(index));
    }

    // Return feature matrix

    return principal_components.arrange_submatrix_rows(principal_components_indices);
}


// void transform_principal_components_data(const Matrix<double>&);

/// Transforms the data according to the principal components.
/// @param principal_components Matrix containing the principal components.

void DataSet::transform_principal_components_data(const Matrix<double>& principal_components)
{
    const Vector<size_t> targets_indices = variables.arrange_targets_indices();

    const Matrix<double> target_data = arrange_target_data();

    subtract_input_data_mean();

    const size_t principal_components_number = principal_components.get_rows_number();

    // Transform data

    const Vector<size_t> used_instances = instances.arrange_used_indices();

    const size_t new_instances_number = used_instances.size();

    const Matrix<double> input_data = arrange_input_data();

    Matrix<double> new_data(new_instances_number, principal_components_number, 0.0);

    size_t instance_index;

    for(size_t i = 0; i < new_instances_number; i++)
    {
        instance_index = used_instances[i];

        for(size_t j = 0; j < principal_components_number; j++)
        {   
            new_data(i,j) = input_data.arrange_row(instance_index).dot(principal_components.arrange_row(j));
        }
    }

    data = new_data.assemble_columns(target_data);
}


// void scale_data_mean_standard_deviation(const Vector< Statistics<double> >&) const method

/// Scales the data matrix with given mean and standard deviation values.
/// It updates the data matrix.
/// @param data_statistics Vector of statistics structures for all the variables in the data set.
/// The size of that vector must be equal to the number of variables.

void DataSet::scale_data_mean_standard_deviation(const Vector< Statistics<double> >& data_statistics)
{
   // Control sentence (if debug)

   #ifdef __OPENNN_DEBUG__

   std::ostringstream buffer;

   const size_t columns_number = data.get_columns_number();

   const size_t statistics_size = data_statistics.size();

   if(statistics_size != columns_number)
   {
      buffer << "OpenNN Exception: DataSet class.\n"
             << "void scale_data_mean_standard_deviation(const Vector< Statistics<double> >&) method.\n"
             << "Size of statistics must be equal to number of columns.\n";

	  throw std::logic_error(buffer.str());
   }

   #endif

   const size_t variables_number = variables.get_variables_number();

   for(size_t i = 0; i < variables_number; i++)
   {
       if(display && data_statistics[i].standard_deviation < 1.0e-99)
       {
          std::cout << "OpenNN Warning: DataSet class.\n"
                    << "void scale_data_mean_standard_deviation(const Vector< Statistics<Type> >&) method.\n"
                    << "Standard deviation of variable " <<  i << " is zero.\n"
                    << "That variable won't be scaled.\n";
        }
    }

   data.scale_mean_standard_deviation(data_statistics);
}


// Vector< Statistics<double> > scale_data_minimum_maximum(void) method

/// Scales the data using the minimum and maximum method,
/// and the minimum and maximum values calculated from the data matrix.
/// It also returns the statistics from all columns.

Vector< Statistics<double> > DataSet::scale_data_minimum_maximum(void)
{
    const Vector< Statistics<double> > data_statistics = calculate_data_statistics();

    scale_data_minimum_maximum(data_statistics);

    return(data_statistics);
}


// Vector< Statistics<double> > scale_data_mean_standard_deviation(void) method

/// Scales the data using the mean and standard deviation method,
/// and the mean and standard deviation values calculated from the data matrix.
/// It also returns the statistics from all columns.

Vector< Statistics<double> > DataSet::scale_data_mean_standard_deviation(void)
{
    const Vector< Statistics<double> > data_statistics = calculate_data_statistics();

    scale_data_mean_standard_deviation(data_statistics);

    return(data_statistics);
}


// void subtract_input_data_mean(void) method

/// Subtracts off the mean to every of the input variables.

void DataSet::subtract_input_data_mean(void)
{
    Vector< Statistics<double> > input_statistics = calculate_inputs_statistics();

    Vector<size_t> inputs_indices = variables.arrange_inputs_indices();
    Vector<size_t> used_instances_indices = instances.arrange_used_indices();

    size_t input_index;
    size_t instance_index;

    double input_mean;

    for(size_t i = 0; i < inputs_indices.size(); i++)
    {
        input_index = inputs_indices[i];

        input_mean = input_statistics[i/*input_index*/].mean;

        for(size_t j = 0; j < used_instances_indices.size(); j++)
        {
            instance_index = used_instances_indices[j];

            data(instance_index,input_index) = data(instance_index,input_index) - input_mean;
        }
    }
}


// void scale_data_minimum_maximum(const Vector< Statistics<double> >&) method

/// Scales the data matrix with given minimum and maximum values.
/// It updates the data matrix.
/// @param data_statistics Vector of statistics structures for all the variables in the data set.
/// The size of that vector must be equal to the number of variables.

void DataSet::scale_data_minimum_maximum(const Vector< Statistics<double> >& data_statistics)
{
    const size_t variables_number = variables.get_variables_number();

   // Control sentence (if debug)

   #ifdef __OPENNN_DEBUG__

   std::ostringstream buffer;

   const size_t statistics_size = data_statistics.size();

   if(statistics_size != variables_number)
   {
      buffer << "OpenNN Exception: DataSet class.\n"
             << "void scale_data_minimum_maximum(const Vector< Statistics<double> >&) method.\n"
             << "Size of data statistics must be equal to number of variables.\n";

	  throw std::logic_error(buffer.str());
   }

   #endif

   for(size_t i = 0; i < variables_number; i++)
   {
       if(display && data_statistics[i].maximum-data_statistics[i].minimum < 1.0e-99)
       {
          std::cout << "OpenNN Warning: DataSet class.\n"
                    << "void scale_data_minimum_maximum(const Vector< Statistics<Type> >&) method.\n"
                    << "Range of variable " <<  i << " is zero.\n"
                    << "That variable won't be scaled.\n";
        }
    }


   data.scale_minimum_maximum(data_statistics);
}

/*
// void scale_data(const std::string&, const Vector< Statistics<double> >&) method

/// Scales the data matrix.
/// The method to be used is that in the scaling and unscaling method variable. 
/// @param scaling_unscaling_method_string String with the name of the scaling-unscaling method
/// (MinimumMaximum or MeanStandardDeviation).
/// @param data_statistics Vector of statistics structures for all the variables in the data set.
/// The size of that vector must be equal to the number of variables.

void DataSet::scale_data(const std::string& scaling_unscaling_method_string, const Vector< Statistics<double> >& data_statistics)
{
   switch(get_scaling_unscaling_method(scaling_unscaling_method_string))
   {
      case MinimumMaximum:
      {
         scale_data_minimum_maximum(data_statistics);
      }            
      break;

      case MeanStandardDeviation:
      {
         scale_data_mean_standard_deviation(data_statistics);
      }
      break;

      default:
      {
         std::ostringstream buffer;

         buffer << "OpenNN Exception: DataSet class\n"
                << "void scale_data(const std::string&, const Vector< Vector<double> >&) method.\n"
                << "Unknown data scaling and unscaling method.\n";

	     throw std::logic_error(buffer.str());
      }
      break;
   }
}


// Vector< Statistics<double> > scale_data(void) method

/// Calculates the data statistics, scales the data with that values and returns the statistics. 
/// The method to be used is that in the scaling and unscaling method variable. 

Vector< Statistics<double> > DataSet::scale_data(const std::string& scaling_unscaling_method)
{
   const Vector< Statistics<double> > statistics = data.calculate_statistics();

   switch(get_scaling_unscaling_method(scaling_unscaling_method))
   {
      case MinimumMaximum:
      {
         scale_data_minimum_maximum(statistics);
      }            
      break;

      case MeanStandardDeviation:
      {
         scale_data_mean_standard_deviation(statistics);
      }
      break;

      default:
      {
         std::ostringstream buffer;

         buffer << "OpenNN Exception: DataSet class\n"
                << "Vector< Statistics<double> > scale_data(const std::string&) method.\n"
                << "Unknown scaling and unscaling method.\n";

	     throw std::logic_error(buffer.str());
      }
      break;
   }

   return(statistics);
}
*/

// void scale_inputs_mean_standard_deviation(const Vector< Statistics<double> >&) method

/// Scales the input variables with given mean and standard deviation values.
/// It updates the input variables of the data matrix.
/// @param inputs_statistics Vector of statistics structures for the input variables.
/// The size of that vector must be equal to the number of inputs.

void DataSet::scale_inputs_mean_standard_deviation(const Vector< Statistics<double> >& inputs_statistics)
{
    const Vector<size_t> inputs_indices = variables.arrange_inputs_indices();

    data.scale_columns_mean_standard_deviation(inputs_statistics, inputs_indices);
}


// Vector< Statistics<double> > scale_inputs_mean_standard_deviation(void) method

/// Scales the input variables with the calculated mean and standard deviation values from the data matrix.
/// It updates the input variables of the data matrix.
/// It also returns a vector of vectors with the variables statistics. 

Vector< Statistics<double> > DataSet::scale_inputs_mean_standard_deviation(void)
{
    // Control sentence (if debug)

    #ifdef __OPENNN_DEBUG__

    if(data.empty())
    {
       std::ostringstream buffer;

       buffer << "OpenNN Exception: DataSet class.\n"
              << "Vector< Statistics<double> > scale_inputs_mean_standard_deviation(void) method.\n"
              << "Data file is not loaded.\n";

       throw std::logic_error(buffer.str());
    }

    #endif

   const Vector< Statistics<double> > inputs_statistics = calculate_inputs_statistics();

   scale_inputs_mean_standard_deviation(inputs_statistics);

   return(inputs_statistics);
}


// void scale_inputs_minimum_maximum(const Vector< Statistics<double> >&) method

/// Scales the input variables with given minimum and maximum values.
/// It updates the input variables of the data matrix.
/// @param inputs_statistics Vector of statistics structures for all the inputs in the data set.
/// The size of that vector must be equal to the number of input variables.

void DataSet::scale_inputs_minimum_maximum(const Vector< Statistics<double> >& inputs_statistics)
{
    const Vector<size_t> inputs_indices = variables.arrange_inputs_indices();

    data.scale_columns_minimum_maximum(inputs_statistics, inputs_indices);
}


// Vector< Statistics<double> > scale_inputs_minimum_maximum(void) method

/// Scales the input variables with the calculated minimum and maximum values from the data matrix.
/// It updates the input variables of the data matrix.
/// It also returns a vector of vectors with the minimum and maximum values of the input variables. 

Vector< Statistics<double> > DataSet::scale_inputs_minimum_maximum(void)
{
    // Control sentence (if debug)

    #ifdef __OPENNN_DEBUG__

    if(data.empty())
    {
       std::ostringstream buffer;

       buffer << "OpenNN Exception: DataSet class.\n"
              << "Vector< Statistics<double> > scale_inputs_minimum_maximum(void) method.\n"
              << "Data file is not loaded.\n";

       throw std::logic_error(buffer.str());
    }

    #endif

   const Vector< Statistics<double> > inputs_statistics = calculate_inputs_statistics();

   scale_inputs_minimum_maximum(inputs_statistics);

   return(inputs_statistics);
}


// Vector< Vector<double> > scale_inputs(const std::string&) method

/// Calculates the input and target variables statistics. 
/// Then it scales the input variables with that values.
/// The method to be used is that in the scaling and unscaling method variable. 
/// Finally, it returns the statistics. 

Vector< Statistics<double> > DataSet::scale_inputs(const std::string& scaling_unscaling_method)
{
    switch(get_scaling_unscaling_method(scaling_unscaling_method))
    {
    case NoScaling:
    {
        return(calculate_inputs_statistics());
    }
        break;

    case MinimumMaximum:
    {
        return(scale_inputs_minimum_maximum());
    }
        break;

    case MeanStandardDeviation:
    {
        return(scale_inputs_mean_standard_deviation());
    }
        break;

    default:
    {
        std::ostringstream buffer;

        buffer << "OpenNN Exception: DataSet class\n"
               << "Vector< Statistics<double> > scale_inputs(void) method.\n"
               << "Unknown scaling and unscaling method.\n";

        throw std::logic_error(buffer.str());
    }
        break;
    }
}


// void scale_inputs(const std::string&, const Vector< Statistics<double> >&) method

/// Calculates the input and target variables statistics.
/// Then it scales the input variables with that values.
/// The method to be used is that in the scaling and unscaling method variable.

void DataSet::scale_inputs(const std::string& scaling_unscaling_method, const Vector< Statistics<double> >& inputs_statistics)
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
         std::ostringstream buffer;

         buffer << "OpenNN Exception: DataSet class\n"
                << "void scale_inputs(const std::string&, const Vector< Statistics<double> >&) method.\n"
                << "Unknown scaling and unscaling method.\n";

         throw std::logic_error(buffer.str());
      }
      break;
   }
}


// void scale_targets_mean_standard_deviation(const Vector< Statistics<double> >&)

/// Scales the target variables with given mean and standard deviation values.
/// It updates the target variables of the data matrix.
/// @param targets_statistics Vector of statistics structures for all the targets in the data set.
/// The size of that vector must be equal to the number of target variables.

void DataSet::scale_targets_mean_standard_deviation(const Vector< Statistics<double> >& targets_statistics)
{
    const Vector<size_t> targets_indices = variables.arrange_targets_indices();

    data.scale_columns_mean_standard_deviation(targets_statistics, targets_indices);
}


// Vector< Statistics<double> > scale_targets_mean_standard_deviation(void) method

/// Scales the target variables with the calculated mean and standard deviation values from the data matrix.
/// It updates the target variables of the data matrix.
/// It also returns a vector of statistics structures with the basic statistics of all the variables.

Vector< Statistics<double> > DataSet::scale_targets_mean_standard_deviation(void)
{    
    // Control sentence (if debug)

    #ifdef __OPENNN_DEBUG__

    if(data.empty())
    {
       std::ostringstream buffer;

       buffer << "OpenNN Exception: DataSet class.\n"
              << "Vector< Statistics<double> > scale_targets_mean_standard_deviation(void) method.\n"
              << "Data file is not loaded.\n";

       throw std::logic_error(buffer.str());
    }

    #endif

   const Vector< Statistics<double> > targets_statistics = calculate_targets_statistics();

   scale_targets_mean_standard_deviation(targets_statistics);

   return(targets_statistics);
}


// void scale_targets_minimum_maximum(const Vector< Statistics<double> >&) method

/// Scales the target variables with given minimum and maximum values.
/// It updates the target variables of the data matrix.
/// @param targets_statistics Vector of statistics structures for all the targets in the data set.
/// The size of that vector must be equal to the number of target variables.

void DataSet::scale_targets_minimum_maximum(const Vector< Statistics<double> >& targets_statistics)
{
    // Control sentence (if debug)

    #ifdef __OPENNN_DEBUG__

    if(data.empty())
    {
       std::ostringstream buffer;

       buffer << "OpenNN Exception: DataSet class.\n"
              << "Vector< Statistics<double> > scale_targets_minimum_maximum(void) method.\n"
              << "Data file is not loaded.\n";

       throw std::logic_error(buffer.str());
    }

    #endif

    const Vector<size_t> targets_indices = variables.arrange_targets_indices();

    data.scale_columns_minimum_maximum(targets_statistics, targets_indices);
}


// Vector< Statistics<double> > scale_targets_minimum_maximum(void) method

/// Scales the target variables with the calculated minimum and maximum values from the data matrix.
/// It updates the target variables of the data matrix.
/// It also returns a vector of vectors with the statistics of the input target variables. 

Vector< Statistics<double> > DataSet::scale_targets_minimum_maximum(void)
{
   const Vector< Statistics<double> > targets_statistics = calculate_targets_statistics();

   scale_targets_minimum_maximum(targets_statistics);

   return(targets_statistics);
}


// Vector< Statistics<double> > scale_targets(const std::string&) method

/// Calculates the input and target variables statistics. 
/// Then it scales the target variables with that values.
/// The method to be used is that in the scaling and unscaling method variable. 
/// Finally, it returns the statistics. 

Vector< Statistics<double> > DataSet::scale_targets(const std::string& scaling_unscaling_method)
{
    switch(get_scaling_unscaling_method(scaling_unscaling_method))
    {

    case NoUnscaling:
    {
        return(calculate_targets_statistics());
    }
        break;
    case MinimumMaximum:
    {
        return(scale_targets_minimum_maximum());
    }
        break;

    case MeanStandardDeviation:
    {
        return(scale_targets_mean_standard_deviation());
    }
        break;

    default:
    {
        std::ostringstream buffer;

        buffer << "OpenNN Exception: DataSet class\n"
               << "Vector< Statistics<double> > scale_targets(const std::string&) method.\n"
               << "Unknown scaling and unscaling method.\n";

        throw std::logic_error(buffer.str());
    }
        break;
    }
}


// void scale_targets(const std::string&, const Vector< Statistics<double> >&) method

/// It scales the input variables with that values.
/// The method to be used is that in the scaling and unscaling method variable.

void DataSet::scale_targets(const std::string& scaling_unscaling_method, const Vector< Statistics<double> >& targets_statistics)
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

    default:
    {
        std::ostringstream buffer;

        buffer << "OpenNN Exception: DataSet class\n"
               << "void scale_targets(const std::string&, const Vector< Statistics<double> >&) method.\n"
               << "Unknown scaling and unscaling method.\n";

        throw std::logic_error(buffer.str());
    }
        break;
    }
}


// void unscale_data_mean_standard_deviation(const Vector< Statistics<double> >&) method

/// Unscales the data matrix with given mean and standard deviation values.
/// It updates the data matrix.
/// @param data_statistics Vector of statistics structures for all the variables in the data set.
/// The size of that vector must be equal to the number of variables.

void DataSet::unscale_data_mean_standard_deviation(const Vector< Statistics<double> >& data_statistics)
{
   data.unscale_mean_standard_deviation(data_statistics);
}


// void unscale_data_minimum_maximum(const Vector< Statistics<double> >&) method

/// Unscales the data matrix with given minimum and maximum values.
/// It updates the data matrix.
/// @param data_statistics Vector of statistics structures for all the variables in the data set.
/// The size of that vector must be equal to the number of variables.

void DataSet::unscale_data_minimum_maximum(const Vector< Statistics<double> >& data_statistics)
{
   data.unscale_minimum_maximum(data_statistics);
}


// void unscale_inputs_mean_standard_deviation(const Vector< Statistics<double> >&) method

/// Unscales the input variables with given mean and standard deviation values.
/// It updates the input variables of the data matrix.
/// @param data_statistics Vector of statistics structures for all the variables in the data set.
/// The size of that vector must be equal to the number of variables.

void DataSet::unscale_inputs_mean_standard_deviation(const Vector< Statistics<double> >& data_statistics)
{
    const Vector<size_t> inputs_indices = variables.arrange_inputs_indices();

    data.unscale_columns_mean_standard_deviation(data_statistics, inputs_indices);
}


// void unscale_inputs_minimum_maximum(const Vector< Statistics<double> >&) method

/// Unscales the input variables with given minimum and maximum values.
/// It updates the input variables of the data matrix.
/// @param data_statistics Vector of statistics structures for all the data in the data set.
/// The size of that vector must be equal to the number of variables.

void DataSet::unscale_inputs_minimum_maximum(const Vector< Statistics<double> >& data_statistics)
{
    const Vector<size_t> inputs_indices = variables.arrange_inputs_indices();

    data.unscale_columns_minimum_maximum(data_statistics, inputs_indices);
}


// void unscale_targets_mean_standard_deviation(const Vector< Statistics<double> >&) method

/// Unscales the target variables with given mean and standard deviation values.
/// It updates the target variables of the data matrix.
/// @param data_statistics Vector of statistics structures for all the variables in the data set.
/// The size of that vector must be equal to the number of variables.

void DataSet::unscale_targets_mean_standard_deviation(const Vector< Statistics<double> >& data_statistics)
{
    const Vector<size_t> targets_indices = variables.arrange_targets_indices();

    data.unscale_columns_mean_standard_deviation(data_statistics, targets_indices);
}


// void unscale_targets_minimum_maximum(const Vector< Statistics<double> >&) method

/// Unscales the target variables with given minimum and maximum values.
/// It updates the target variables of the data matrix.
/// @param data_statistics Vector of statistics structures for all the variables.
/// The size of that vector must be equal to the number of variables.

void DataSet::unscale_targets_minimum_maximum(const Vector< Statistics<double> >& data_statistics)
{
    const Vector<size_t> targets_indices = variables.arrange_targets_indices();

    data.unscale_columns_minimum_maximum(data_statistics, targets_indices);
}


// void initialize_data(const double& value) method

/// Initializes the data matrix with a given value.
/// @param new_value Initialization value. 

void DataSet::initialize_data(const double& new_value)
{
   data.initialize(new_value);
}


// void randomize_data_uniform(const double&, const double&) method

/// Initializes the data matrix with random values chosen from a uniform distribution
/// with given minimum and maximum.

void DataSet::randomize_data_uniform(const double& minimum, const double& maximum)
{
   data.randomize_uniform(minimum, maximum);
}


// void randomize_data_normal(const double&, const double&) method

/// Initializes the data matrix with random values chosen from a normal distribution
/// with given mean and standard deviation.

void DataSet::randomize_data_normal(const double& mean, const double& standard_deviation)
{
   data.randomize_normal(mean, standard_deviation);
}


// tinyxml2::XMLDocument* to_XML(void) const method

/// Serializes the data set object into a XML document of the TinyXML library. 

tinyxml2::XMLDocument* DataSet::to_XML(void) const
{
   tinyxml2::XMLDocument* document = new tinyxml2::XMLDocument;

   std::ostringstream buffer;

   // Data set

   tinyxml2::XMLElement* data_set_element = document->NewElement("DataSet");
   document->InsertFirstChild(data_set_element);

   tinyxml2::XMLElement* element = NULL;
   tinyxml2::XMLText* text = NULL;

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

   // Lags Number
   {
       element = document->NewElement("LagsNumber");
       data_file_element->LinkEndChild(element);

       const size_t lags_number = get_lags_number();

       buffer.str("");
       buffer << lags_number;

       text = document->NewText(buffer.str().c_str());
       element->LinkEndChild(text);
   }

   // Steps Ahead
   {
       element = document->NewElement("StepsAhead");
       data_file_element->LinkEndChild(element);

       const size_t steps_ahead = get_steps_ahead();

       buffer.str("");
       buffer << steps_ahead;

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

   // Data file name
   {
      element = document->NewElement("DataFileName");
      data_file_element->LinkEndChild(element);

      text = document->NewText(data_file_name.c_str());
      element->LinkEndChild(text);
   }

   // Variables 
   {
      element = document->NewElement("Variables");
      data_set_element->LinkEndChild(element);

      const tinyxml2::XMLDocument* variables_document = variables.to_XML();

      const tinyxml2::XMLElement* variables_element = variables_document->FirstChildElement("Variables");

      DeepClone(element, variables_element, document, NULL);

      delete variables_document;
   }

   // Instances
   {
       element = document->NewElement("Instances");
       data_set_element->LinkEndChild(element);

       const tinyxml2::XMLDocument* instances_document = instances.to_XML();

       const tinyxml2::XMLElement* instances_element = instances_document->FirstChildElement("Instances");

       DeepClone(element, instances_element, document, NULL);

       delete instances_document;
   }

   // Missing values
   {
       element = document->NewElement("MissingValues");
       data_set_element->LinkEndChild(element);

       const tinyxml2::XMLDocument* missing_values_document = missing_values.to_XML();

       const tinyxml2::XMLElement* missing_values_element = missing_values_document->FirstChildElement("MissingValues");

       DeepClone(element, missing_values_element, document, NULL);

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


// void write_XML(tinyxml2::XMLPrinter&) const method

/// Serializes the data set object into a XML document of the TinyXML library without keep the DOM tree in memory.

void DataSet::write_XML(tinyxml2::XMLPrinter& file_stream) const
{
    std::ostringstream buffer;

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


// void from_XML(const tinyxml2::XMLDocument&) method

/// Deserializes a TinyXML document into this data set object.
/// @param data_set_document XML document containing the member data.

void DataSet::from_XML(const tinyxml2::XMLDocument& data_set_document)
{
   std::ostringstream buffer;

   // Data set element

   const tinyxml2::XMLElement* data_set_element = data_set_document.FirstChildElement("DataSet");

   if(!data_set_element)
   {
       buffer << "OpenNN Exception: DataSet class.\n"
              << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
              << "Data set element is NULL.\n";

       throw std::logic_error(buffer.str());
   }

    // Data file

    const tinyxml2::XMLElement* data_file_element = data_set_element->FirstChildElement("DataFile");

    if(!data_file_element)
    {
       buffer << "OpenNN Exception: DataSet class.\n"
              << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
              << "Data file element is NULL.\n";

       throw std::logic_error(buffer.str());
    }

    // Data file name
    {
       const tinyxml2::XMLElement* data_file_name_element = data_file_element->FirstChildElement("DataFileName");

       if(!data_file_name_element)
       {
           buffer << "OpenNN Exception: DataSet class.\n"
                  << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
                  << "Data file name element is NULL.\n";

           throw std::logic_error(buffer.str());
       }

       if(data_file_name_element->GetText())
       {
            const std::string new_data_file_name = data_file_name_element->GetText();

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
                   << "Lags number element is NULL.\n";

            throw std::logic_error(buffer.str());
        }

        if(lags_number_element->GetText())
        {
             const size_t new_lags_number = atoi(lags_number_element->GetText());

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
                   << "Steps ahead element is NULL.\n";

            throw std::logic_error(buffer.str());
        }

        if(steps_ahead_element->GetText())
        {
             const size_t new_steps_ahead = atoi(steps_ahead_element->GetText());

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
                  << "File type element is NULL.\n";

           throw std::logic_error(buffer.str());
       }

       if(file_type_element->GetText())
       {
            const std::string new_file_type = file_type_element->GetText();

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
                  << "First Cell element is NULL.\n";

           throw std::logic_error(buffer.str());
       }

       if(first_cell_element->GetText())
       {
            const std::string new_first_cell = first_cell_element->GetText();

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
                  << "Last Cell element is NULL.\n";

           throw std::logic_error(buffer.str());
       }

       if(last_cell_element->GetText())
       {
            const std::string new_last_cell = last_cell_element->GetText();

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
                  << "Sheet Number element is NULL.\n";

           throw std::logic_error(buffer.str());
       }

       if(sheet_number_element->GetText())
       {
            const size_t new_sheet_number = atoi(sheet_number_element->GetText());

            sheet_number = new_sheet_number;
       }
    }

      // Header line
      {
         const tinyxml2::XMLElement* header_element = data_file_element->FirstChildElement("ColumnsName");

         if(header_element)
         {
            const std::string new_header_string = header_element->GetText();

            try
            {
               set_header_line(new_header_string != "0");
            }
            catch(const std::logic_error& e)
            {
               std::cout << e.what() << std::endl;
            }
         }
      }

        // Rows label
        {
           const tinyxml2::XMLElement* rows_label_element = data_file_element->FirstChildElement("RowsLabel");

           if(rows_label_element)
           {
              const std::string new_rows_label_string = rows_label_element->GetText();

              try
              {
                 set_rows_label(new_rows_label_string != "0");
              }
              catch(const std::logic_error& e)
              {
                 std::cout << e.what() << std::endl;
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
            const std::string new_separator = separator_element->GetText();

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
                const std::string new_missing_values_label = missing_values_label_element->GetText();

                set_missing_values_label(new_missing_values_label);
             }
         }
    }

    // Variables
    {
       const tinyxml2::XMLElement* variables_element = data_set_element->FirstChildElement("Variables");

       if(variables_element)
       {
           tinyxml2::XMLDocument variables_document;

           tinyxml2::XMLElement* variables_element_clone = variables_document.NewElement("Variables");
           variables_document.InsertFirstChild(variables_element_clone);

           DeepClone(variables_element_clone, variables_element, &variables_document, NULL);

           variables.from_XML(variables_document);
       }
    }

    // Instances
    {
       const tinyxml2::XMLElement* instances_element = data_set_element->FirstChildElement("Instances");

       if(instances_element)
       {
           tinyxml2::XMLDocument instances_document;

           tinyxml2::XMLElement* instances_element_clone = instances_document.NewElement("Instances");
           instances_document.InsertFirstChild(instances_element_clone);

           DeepClone(instances_element_clone, instances_element, &instances_document, NULL);

           instances.from_XML(instances_document);
       }
    }

    // Missing values
    {
       const tinyxml2::XMLElement* missing_values_element = data_set_element->FirstChildElement("MissingValues");

       if(missing_values_element)
       {
           tinyxml2::XMLDocument missing_values_document;

           tinyxml2::XMLElement* missing_values_element_clone = missing_values_document.NewElement("MissingValues");
           missing_values_document.InsertFirstChild(missing_values_element_clone);

           DeepClone(missing_values_element_clone, missing_values_element, &missing_values_document, NULL);

           missing_values.from_XML(missing_values_document);
       }
    }

   // Display
   {
      const tinyxml2::XMLElement* display_element = data_set_element->FirstChildElement("Display");

      if(display_element)
      {
         const std::string new_display_string = display_element->GetText();

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

}


// std::string to_string(void) const method

/// Returns a string representation of the current data set object. 

std::string DataSet::to_string(void) const
{
   std::ostringstream buffer;

   buffer << "Data set object\n"
          << "Data file name: " << data_file_name << "\n"
          << "Header line: " << header_line << "\n"
          << "Separator: " << separator << "\n"
          << "Missing values label: " << missing_values_label << "\n"
          << "Data:\n" << data << "\n"
          << "Display: " << display << "\n"
          << variables.to_string()
          << instances.to_string()
          << missing_values.to_string();

   return(buffer.str());
}


// void print(void) const method

/// Prints to the screen in text format the members of the data set object.

void DataSet::print(void) const
{
   if(display)
   {
      std::cout << to_string();
   }
}


// void print_summary(void) const method

/// Prints to the screen in text format the main numbers from the data set object.

void DataSet::print_summary(void) const
{
    if(display)
    {
        const size_t variables_number = variables.get_variables_number();
        const size_t instances_number = instances.get_instances_number();
        const size_t missing_values_number = missing_values.get_missing_values_number();

       std::cout << "Data set object summary:\n"
                 << "Number of variables: " << variables_number << "\n"
                 << "Number of instances: " << instances_number << "\n"
                 << "Number of missing values: " << missing_values_number << std::endl;
    }
}


// void save(const std::string&) const method

/// Saves the members of a data set object to a XML-type file in an XML-type format.
/// @param file_name Name of data set XML-type file.

void DataSet::save(const std::string& file_name) const
{
   tinyxml2::XMLDocument* document = to_XML();

   document->SaveFile(file_name.c_str());

   delete document;
}


// void load(const std::string&) method

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

void DataSet::load(const std::string& file_name)
{
   tinyxml2::XMLDocument document;

   if(document.LoadFile(file_name.c_str()))
   {
      std::ostringstream buffer;

      buffer << "OpenNN Exception: DataSet class.\n"
             << "void load(const std::string&) method.\n"
             << "Cannot load XML file " << file_name << ".\n";

      throw std::logic_error(buffer.str());
   }

   from_XML(document);
}


// void print_data(void) const method

/// Prints to the sceen the values of the data matrix.

void DataSet::print_data(void) const
{
   if(display)
   {
      std::cout << data << std::endl;
   }
}


// void print_data_preview(void) const method

/// Prints to the sceen a preview of the data matrix,
/// i.e., the first, second and last instances

void DataSet::print_data_preview(void) const
{
   if(display)
   {
       const size_t instances_number = instances.get_instances_number();

       if(instances_number > 0)
       {
          const Vector<double> first_instance = data.arrange_row(0);

          std::cout << "First instance:\n"
                    << first_instance << std::endl;
       }

       if(instances_number > 1)
       {
          const Vector<double> second_instance = data.arrange_row(1);

          std::cout << "Second instance:\n"
                    << second_instance << std::endl;
       }

       if(instances_number > 2)
       {
          const Vector<double> last_instance = data.arrange_row(instances_number-1);

          std::cout << "Instance " << instances_number << ":\n"
                    << last_instance << std::endl;
       }
    }
}


// void save_data(void) const method

/// Saves to the data file the values of the data matrix.

void DataSet::save_data(void) const
{
   std::ofstream file(data_file_name.c_str());

   if(!file.is_open())
   {
      std::ostringstream buffer;

      buffer << "OpenNN Exception: DataSet class.\n" 
             << "void save_data(void) const method.\n"
             << "Cannot open data file.\n";

      throw std::logic_error(buffer.str());	  
   }
  
   const std::string separator_string = get_separator_string();

   if(header_line)
   {
       const Vector<std::string> variables_name = variables.arrange_names();

       file << variables_name.to_string(separator_string) << std::endl;
   }

   // Write data

   const size_t rows_number = data.get_rows_number();
   const size_t columns_number = data.get_columns_number();

   for(size_t i = 0; i < rows_number; i++)
   {
       for(size_t j = 0; j < columns_number; j++)
       {
           file << data(i,j);

           if(j != columns_number-1)
           {
               file << separator_string;
           }
       }

      file << std::endl;
   }

   // Close file

   file.close();
}


// size_t get_column_index(const Vector< Vector<std::string> >&, const size_t) const method

/// Returns the index of a variable when reading the data file.
/// @param nominal_labels Values of all nominal variables in the data file.
/// @param column_index Index of column.

size_t DataSet::get_column_index(const Vector< Vector<std::string> >& nominal_labels, const size_t column_index) const
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


// void check_separator(const std::string&) method

/// Verifies that a given line in the data file contains the separator characer.
/// If the line does not contain the separator, this method throws an exception.
/// @param line Data file line.

void DataSet::check_separator(const std::string& line) const
{
    if(line.empty())
    {
        return;
    }

    const std::string separator_string = get_separator_string();

    if(line.find(separator_string) == std::string::npos)
    {
        std::ostringstream buffer;

        buffer << "OpenNN Exception: DataSet class.\n"
               << "void check_separator(const std::string&) method.\n"
               << "Separator '" << write_separator() << "' not found in data file " << data_file_name << ".\n";

        throw std::logic_error(buffer.str());
    }
}


// size_t count_data_file_columns_number(void) const method

/// Returns the number of tokens in the first line of the data file.
/// That will be interpreted as the number of columns in the data file.

size_t DataSet::count_data_file_columns_number(void) const
{
    std::ifstream file(data_file_name.c_str());

    std::string line;

    size_t columns_number = 0;

    while(file.good())
    {
        getline(file, line);

        if(separator != Tab)
        {
            std::replace(line.begin(), line.end(), '\t', ' ');
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


// void check_header_line(void) method

/// Verifies that the data file has a header line.
/// All elements in a header line must be strings.
/// This method can change the value of the header line member.
/// It throws an exception if some inconsistencies are found.

void DataSet::check_header_line(void)
{
    std::ifstream file(data_file_name.c_str());

    std::string line;
    Vector<std::string> tokens;

    while(file.good())
    {
        getline(file, line);

        if(separator != Tab)
        {
            std::replace(line.begin(), line.end(), '\t', ' ');
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
            std::cout << "OpenNN Warning: DataSet class.\n"
                      << "void check_header_line(void) method.\n"
                      << "First line of data file interpreted as not header.\n";
        }

        header_line = false;
    }
    else if(header_line && is_mixed(tokens))
    {
        std::ostringstream buffer;

        buffer << "OpenNN Exception: DataSet class.\n"
               << "void check_header_line(void) method.\n"
               << "Header line contains numeric values: \n"
               << line << "\n";

        throw std::logic_error(buffer.str());
    }
    else if(!header_line && is_not_numeric(tokens))
    {
        if(display)
        {
            std::cout << "OpenNN Warning: DataSet class.\n"
                      << "void check_header_line(void) method.\n"
                      << "First line of data file interpreted as header.\n";
        }

        header_line = true;
    }
}


// Vector<std::string> read_header_line(void) const method

/// Returns the name of the columns in the data set as a list of strings.

Vector<std::string> DataSet::read_header_line(void) const
{
    Vector<std::string> header_line;

    std::string line;

    std::ifstream file(data_file_name.c_str());

    // First line

    while(file.good())
    {
        getline(file, line);

        if(separator != Tab)
        {
            std::replace(line.begin(), line.end(), '\t', ' ');
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


// void read_instance(const std::string&, const Vector< Vector<std::string> >&, const size_t&) method

/// Sets the values of a single instance in the data matrix from a line in the data file.
/// @param line Data file line.
/// @param nominal_labels Values of all nominal variables in the data file.
/// @param instance_index Index of instance.

void DataSet::read_instance(const std::string& line, const Vector< Vector<std::string> >& nominal_labels, const size_t& instance_index)
{
    // Control sentence (if debug)

    #ifdef __OPENNN_DEBUG__

    const size_t instances_number = instances.get_instances_number();

    if(instance_index >= instances_number)
    {
       std::ostringstream buffer;

       buffer << "OpenNN Exception: DataSet class.\n"
              << "void read_instance(const std::string&, const Vector< Vector<std::string> >&, const size_t&) method.\n"
              << "Index of instance (" << instance_index << ") must be less than number of instances (" << instances_number << ").\n";

       throw std::logic_error(buffer.str());
    }

    #endif

    const Vector<std::string> tokens = get_tokens(line);

    #ifdef __OPENNN_DEBUG__

    if(tokens.size() != nominal_labels.size())
    {
       std::ostringstream buffer;

       buffer << "OpenNN Exception: DataSet class.\n"
              << "void read_instance(const std::string&, const Vector< Vector<std::string> >&, const size_t&) method.\n"
              << "Size of tokens (" << tokens.size() << ") must be equal to size of names (" << nominal_labels.size() << ").\n";

       throw std::logic_error(buffer.str());
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
                    std::ostringstream buffer;

                    buffer << "OpenNN Exception: DataSet class.\n"
                           << "void read_instance(const std::string&, const Vector< Vector<std::string> >&, const size_t&) method.\n"
                           << "Unknown token binary value.\n";

                    throw std::logic_error(buffer.str());
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


// Vector< Vector<std::string> > set_from_data_file(void) method

/// Performs a first data file read in which the format is checked,
/// and the numbers of variables, instances and missing values are set.

Vector< Vector<std::string> > DataSet::set_from_data_file(void)
{
    const size_t columns_number = count_data_file_columns_number();

    Vector< Vector<std::string> > nominal_labels(columns_number);

    std::string line;
    Vector<std::string> tokens;

//    bool numeric;

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

    std::ifstream file(data_file_name.c_str());   

    // Rest of lines

    while(file.good())
    {
        getline(file, line);

        if(separator != Tab)
        {
            std::replace(line.begin(), line.end(), '\t', ' ');
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
            std::ostringstream buffer;

            buffer << "OpenNN Exception: DataSet class.\n"
                   << "Vector< Vector<std::string> > DataSet::set_from_data_file(void).\n"
                   << "Row " << instances_count << ": Size of tokens (" << tokens.size() << ") is not equal to "
                   << "number of columns (" << columns_number << ").\n";

            throw std::logic_error(buffer.str());
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
        if(nominal_labels[i].size() == (size_t)instances_count)
        {
            std::ostringstream buffer;

            buffer << "OpenNN Exception: DataSet class.\n"
                   << "Vector< Vector<std::string> > DataSet::set_from_data_file(void).\n"
                   << "Column " << i << ": All elements are nominal and different. It contains meaningless data.\n";

            throw std::logic_error(buffer.str());
        }
    }

    // Set instances and variables number

    if(instances_count == 0 || variables_count == 0)
    {
        set();

        return(nominal_labels);
    }

    data.set(instances_count, variables_count);

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

    if(instances.get_instances_number() != (size_t)instances_count)
    {
        instances.set(instances_count);
    }

    missing_values.set(instances.get_instances_number(), variables.get_variables_number());

    return(nominal_labels);
}


// void read_from_data_file(Vector< Vector<std::string> >&) method

/// Performs a second data file read in which the data is set.

void DataSet::read_from_data_file(const Vector< Vector<std::string> >& nominal_labels)
{
    std::ifstream file(data_file_name.c_str());

    file.clear();
    file.seekg(0, std::ios::beg);

    std::string line;

    if(header_line)
    {
        while(file.good())
        {
            getline(file, line);

            if(separator != Tab)
            {
                std::replace(line.begin(), line.end(), '\t', ' ');
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

//    #pragma omp parallel for private(i, line)

    while(file.good())
    {
        getline(file, line);

        if(separator != Tab)
        {
            std::replace(line.begin(), line.end(), '\t', ' ');
        }

        const Vector<std::string> tokens = get_tokens(line);

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


// Vector<std::string> arrange_time_series_prediction_names(const Vector<std::string>&) const method

/// Returns a vector with the names arranged for time series prediction, according to the number of lags.
/// @todo

Vector<std::string> DataSet::arrange_time_series_names(const Vector<std::string>&) const
{        
    Vector<std::string> time_series_prediction_names;
/*
    Vector< Vector<std::string> > new_names((1+columns_number)*lags_number);

    for(size_t i = 0; i < 1+lags_number; i++)
    {
        for(size_t j = 0; j < names.size(); j++)
        {
            new_names[i+j] = names[j];

            if(i != lags_number)
            {
                for(size_t k = 0; k < names[j].size();k++)
                {
                    new_names[i+j][k].append("_lag_").append(std::string::from_size_t(lags_number-i).c_str());
                }
            }
        }
    }
*/
    return(time_series_prediction_names);
}


// Vector<std::string> DataSet::arrange_association_names(const Vector<std::string>& names) const method

/// Returns a vector with the names arranged for association.
/// @todo

Vector<std::string> DataSet::arrange_association_names(const Vector<std::string>&) const
{
    Vector<std::string> association_names;

    return(association_names);
}


// void convert_time_series(void) method

/// Arranges an input-target matrix from a time series matrix, according to the number of lags.
/// @todo

void DataSet::convert_time_series(void)
{
    if(lags_number == 0)
    {
        return;
    }

    data.convert_time_series(lags_number);

    variables.convert_time_series(lags_number);

    instances.convert_time_series(lags_number);

    missing_values.convert_time_series(lags_number);
}


// void convert_association(void) method

/// Arranges the data set for association.
/// @todo

void DataSet::convert_association(void)
{
    data.convert_association();

    variables.convert_association();

    missing_values.convert_association();
}


// void load_data(void) method

/// This method loads the data file.

void DataSet::load_data(void)
{
    if(data_file_name.empty())
    {
       std::ostringstream buffer;

       buffer << "OpenNN Exception: DataSet class.\n"
              << "void load_data(void) method.\n"
              << "Data file name has not been set.\n";

       throw std::logic_error(buffer.str());
    }

    std::ifstream file(data_file_name.c_str());

    if(!file.is_open())
    {
       std::ostringstream buffer;

       buffer << "OpenNN Exception: DataSet class.\n"
              << "void load_data(void) method.\n"
              << "Cannot open data file: " << data_file_name << "\n";

       throw std::logic_error(buffer.str());
    }

    file.close();

    const Vector< Vector<std::string> > nominal_labels = set_from_data_file();

    read_from_data_file(nominal_labels);

    // Variables name

    Vector<std::string> columns_name;

    if(header_line)
    {
        columns_name = read_header_line();
    }
    else
    {
        for(unsigned i = 0; i < nominal_labels.size(); i++)
        {
            std::ostringstream buffer;

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

void DataSet::load_data_binary(void)
{
    std::ifstream file;

    file.open(data_file_name.c_str(), std::ios::binary);

    if(!file.is_open())
    {
        std::ostringstream buffer;

        buffer << "OpenNN Exception: DataSet class.\n"
               << "void load_data_binary(void) method.\n"
               << "Cannot open data file: " << data_file_name << "\n";

        throw std::logic_error(buffer.str());
    }

    std::streamsize size = sizeof(size_t);

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

void DataSet::load_time_series_data_binary(void)
{
    std::ifstream file;

    file.open(data_file_name.c_str(), std::ios::binary);

    if(!file.is_open())
    {
        std::ostringstream buffer;

        buffer << "OpenNN Exception: DataSet class.\n"
               << "void load_data_binary(void) method.\n"
               << "Cannot open data file: " << data_file_name << "\n";

        throw std::logic_error(buffer.str());
    }

    std::streamsize size = sizeof(size_t);

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

    time_series_data.set(time_series_instances_number, time_series_variables_number);

    size = sizeof(double);

    for(size_t i = 0; i < time_series_instances_number*time_series_variables_number; i++)
    {
        file.read(reinterpret_cast<char*>(&value), size);

        time_series_data[i] = value;
    }    

    file.close();
}


// void load_time_series_data(void) method

/// @todo This method is not implemented.
/*
void DataSet::load_time_series_data(void)
{
    if(lags_number <= 0)
    {
       std::ostringstream buffer;

       buffer << "OpenNN Exception: DataSet class.\n"
              << "void load_time_series_data(void) const method.\n"
              << "Number of lags (" << lags_number << ") must be greater than zero.\n";

       throw std::logic_error(buffer.str());
    }


    if(header)
    {
//        Vector<std::string> columns_name;

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
        row = time_series_data.arrange_row(i);

        for(size_t j = 1; j <= lags_number; j++)
        {
            row = row.assemble(time_series_data.arrange_row(i+j));
        }

        data.set_row(i, row);
    }

    // Variables

    Vector<Variables::Use> uses(variables_number);

    std::fill(uses.begin(), uses.begin()+lags_number*variables_number/(lags_number+1)-1, Variables::Use::Input);
    std::fill(uses.begin()+lags_number*variables_number/(lags_number+1), uses.end(), Variables::Use::Target);

    variables.set_uses(uses);

}
*/

// Vector<size_t> calculate_target_distribution(void) const method

/// Returns a vector containing the number of instances of each class in the data set.
/// If the number of target variables is one then the number of classes is two.
/// If the number of target variables is greater than one then the number of classes is equal to the number 
/// of target variables.

Vector<size_t> DataSet::calculate_target_distribution(void) const
{ 
   // Control sentence (if debug)

   const size_t instances_number = instances.get_instances_number();
   const size_t targets_number = variables.count_targets_number();
   const Vector<size_t> targets_indices = variables.arrange_targets_indices();

   Vector<size_t> class_distribution;

   if(targets_number == 1) // Two classes
   {
      class_distribution.set(2, 0);

      size_t target_index = targets_indices[0];

      for(size_t instance_index = 0; instance_index < instances_number; instance_index++)
      {
          if(missing_values.arrange_missing_indices()[target_index].contains(instance_index))
          {
              continue;
          }
          if(instances.get_use(instance_index) != Instances::Unused)
          {
             if(data(instance_index,target_index) < 0.5)
             {
                class_distribution[0]++;
             }
             else
             {
                class_distribution[1]++;
             }
          }
      }
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
                 if(data(i,targets_indices[j]) == -123.456)
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

//   const size_t used_instances_number = instances.count_used_instances_number();

//   if(class_distribution.calculate_sum() != used_instances_number)
//   {
//      std::ostringstream buffer;

//      buffer << "OpenNN Exception: DataSet class.\n"
//             << "Vector<size_t> calculate_target_distribution(void) const method.\n"
//             << "Sum of class distributions (" << class_distribution << ") is not equal to "
//             << "number of used instances (" << used_instances_number << ")." << std::endl;

//      throw std::logic_error(buffer.str());
//   }

   return(class_distribution);
}


// Vector<double> calculate_distances(void) const method

/// Returns a normalized distance between each instance and the mean instance.
/// The size of this vector is the number of instances.

Vector<double> DataSet::calculate_distances(void) const
{
    const Matrix<double> data_statistics_matrix = calculate_data_statistics_matrix();

    const Vector<double> means = data_statistics_matrix.arrange_column(2);
    const Vector<double> standard_deviations = data_statistics_matrix.arrange_column(3);

    const size_t instances_number = instances.get_instances_number();
    Vector<double> distances(instances_number);

    const size_t variables_number = variables.get_variables_number();
    Vector<double> instance(variables_number);

    int i = 0;

    #pragma omp parallel for private(i, instance)

    for(i = 0; i < (int)instances_number; i++)
    {
        instance = data.arrange_row(i);

        distances[i] = (instance-means/standard_deviations).calculate_norm();
    }

    return(distances);
}


// Vector<size_t> balance_binary_targets_class_distribution(void) method

/// This method balances the targets ditribution of a data set with only one target variable by unusing
/// instances whose target variable belongs to the most populated target class.
/// It returns a vector with the indices of the instances set unused.
/// @param percentage Percentage of instances to be unused.

Vector<size_t> DataSet::balance_binary_targets_distribution(const double& percentage)
{
    Vector<size_t> unused_instances;

    const size_t instances_number = instances.count_used_instances_number();

    const Vector<size_t> target_class_distribution = calculate_target_distribution();

    const Vector<size_t> maximal_indices = target_class_distribution.calculate_maximal_indices(2);

    const size_t maximal_target_class_index = maximal_indices[0];
    const size_t minimal_target_class_index = maximal_indices[1];

    size_t total_unbalanced_instances_number = (size_t)((percentage/100.0)*(target_class_distribution[maximal_target_class_index] - target_class_distribution[minimal_target_class_index]));

    size_t actual_unused_instances_number;

    size_t unbalanced_instances_number = total_unbalanced_instances_number/10;

    Vector<size_t> actual_unused_instances;

//    std::cout << "Instances to unuse: " << total_unbalanced_instances_number << std::endl;

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

    return (unused_instances);
}


// Vector<size_t> balance_multiple_targets_distribution(void) method

/// This method balances the targets ditribution of a data set with more than one target variable by unusing
/// instances whose target variable belongs to the most populated target class.
/// It returns a vector with the indices of the instances set unused.

Vector<size_t> DataSet::balance_multiple_targets_distribution(void)
{
    Vector<size_t> unused_instances;

    const size_t bins_number = 10;

    const Vector<size_t> target_class_distribution = calculate_target_distribution();

    const size_t targets_number = variables.count_targets_number();

    const Vector<size_t> inputs_variables_indices = variables.arrange_inputs_indices();
    const Vector<size_t> targets_variables_indices = variables.arrange_targets_indices();

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

        instances_indices = instances.arrange_used_indices();

        instances_number = instances_indices.size();


        maximal_difference_index = target_class_differences.calculate_maximal_index();

        unbalanced_instances_number = (size_t) (target_class_differences[maximal_difference_index]/10);

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

        unbalanced_instances_indices = total_frequencies.sort_greater_rows(0).arrange_column(1).arrange_subvector_first(unbalanced_instances_number);

        unused_instances = unused_instances.assemble(unbalanced_instances_indices);

        instances.set_unused(unbalanced_instances_indices);

        target_class_differences[maximal_difference_index] = target_class_differences[maximal_difference_index] - unbalanced_instances_number;
    }

    return(unused_instances);
}


// Vector<size_t> unuse_most_populated_target(const size_t&)

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

    const size_t targets_number = variables.count_targets_number();

    const Vector<size_t> inputs = variables.arrange_inputs_indices();
    const Vector<size_t> targets = variables.arrange_targets_indices();

    const Vector<size_t> unused_variables = variables.arrange_unused_indices();

    // Instances

    const Vector<size_t> used_instances = instances.arrange_used_indices();

    const size_t used_instances_number = instances.count_used_instances_number();

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

    for(i = 0; i < (int)used_instances_number; i++)
    {                
        index = used_instances[i];

        instance = get_instance(index);

        value = instance[most_populated_target];

        bin = data_histograms[most_populated_target - unused].calculate_bin(value);

        if(bin == most_populated_bin)
        {
            instance_frequencies = instance.calculate_total_frequencies(data_histograms);

            total_instances_frequencies(count_instances, 0) = instance_frequencies.calculate_partial_sum(inputs);
            total_instances_frequencies(count_instances, 1) = used_instances[i];

            count_instances++;
        }
    }

    // Unuses instances

    if(instances_to_unuse > maximum_frequency)
    {
        most_populated_instances = total_instances_frequencies.sort_greater_rows(0).arrange_column(1).arrange_subvector_first(maximum_frequency);
    }
    else
    {
        most_populated_instances = total_instances_frequencies.sort_greater_rows(0).arrange_column(1).arrange_subvector_first(instances_to_unuse);
    }

    instances.set_unused(most_populated_instances);

    return(most_populated_instances);
}


// Vector<size_t> balance_approximation_targets_distribution(void)

/// This method balances the target ditribution of a data set for a function regression problem.
/// It returns a vector with the indices of the instances set unused.
/// It unuses a given percentage of the instances.
/// @param percentage Percentage of the instances to be unused.

Vector<size_t> DataSet::balance_approximation_targets_distribution(const double& percentage)
{
    Vector<size_t> unused_instances;

    const size_t instances_number = instances.count_used_instances_number();

    const size_t instances_to_unuse = (size_t)(instances_number*percentage/100.0);

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


// Vector<size_t> arrange_binary_inputs_indices(void) const method

/// Returns a vector with the indices of the inputs that are binary.

Vector<size_t> DataSet::arrange_binary_inputs_indices(void) const
{
    const size_t inputs_number = variables.count_inputs_number();

    const Vector<size_t> inputs_indices = variables.arrange_inputs_indices();

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


// Vector<size_t> arrange_real_inputs_indices(void) const method

/// Returns a vector with the indices of the inputs that are real.

Vector<size_t> DataSet::arrange_real_inputs_indices(void) const
{
    const size_t inputs_number = variables.count_inputs_number();

    const Vector<size_t> inputs_indices = variables.arrange_inputs_indices();

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


// void sum_binary_inputs(void) method

/// @todo

void DataSet::sum_binary_inputs(void)
{
//    const size_t inputs_number = variables.count_inputs_number();

//    const size_t instances_number = instances.get_instances_number();

//    const Vector<size_t> binary_inputs_indices = arrange_binary_inputs_indices();

//    const size_t binary_inputs_number = binary_inputs_indices.size();

//    Vector<double> binary_variable(instances_number, 0.0);

//    for(size_t i = 0; i < binary_inputs_number; i++)
//    {
//        binary_variable += data.arrange_column(binary_inputs_indices[i]);
//    }

//    const Vector<size_t> real_inputs_indices = arrange_real_inputs_indices();

//    Matrix<double> new_data = data.arrange_submatrix_columns(real_inputs_indices);

//    new_data.append_column(binary_variable);

//    new_data = new_data.assemble_columns(arrange_target_data());
}


// Vector<double> calculate_distances(const size_t&) const

/// Returns a matrix with the distances between every instance and the rest of the instances.
/// The number of rows is the number of instances in the data set.
/// The number of columns is the number of instances in the data set.
/// @param nearest_neighbours_number Nearest neighbors number.

Matrix<double> DataSet::calculate_instances_distances(const size_t& nearest_neighbours_number) const
{
    const size_t instances_number = instances.count_used_instances_number();
    const Vector<size_t> instances_indices = instances.arrange_used_indices();

    //Matrix<double> distances(instances_number, instances_number, 0.0);

    Vector<double> distances(nearest_neighbours_number , -1.0);

    Vector<double> instance;
    Vector<double> other_instance;

    size_t maximal_index;

    double distance;

// #pragma omp parallel for private(maximal_index, distance) collapse(2)

    for(size_t i = 0; i < instances_number; i++)
    {
        for(size_t j = 0; j < instances_number; j++)
         {

              distance = data.calculate_distance(instances_indices[i], instances_indices[j]);

              if(distances.count_greater_than(distance) != 0)
              {
                   maximal_index = distances.calculate_maximal_index();

                   distances[maximal_index] = distance;
              }
          }
    }

    Matrix<double> distances1(instances_number, instances_number, 0.0);
    return(distances1);
}


// Matrix<size_t> calculate_nearest_neighbors(const Matrix<double>&, const size_t&) const

/// Returns a matrix with the k-nearest neighbors to every used instance in the data set.
/// Number of rows is the number of isntances in the data set.
/// Number of columns is the number of nearest neighbors to calculate.
/// @param distances Distances between every instance and the rest of them.
/// @param nearest_neighbours_number Number of nearest neighbors to be calculated.

Matrix<size_t> DataSet::calculate_nearest_neighbors(const Matrix<double>& distances, const size_t& nearest_neighbours_number) const
{
    const size_t instances_number = instances.count_used_instances_number();

    Matrix<size_t> nearest_neighbors(instances_number, nearest_neighbours_number);

    Vector<double> instance_distances;
    Vector<size_t> minimal_distances_indices;

    for(size_t i = 0; i < instances_number; i++)
    {
        instance_distances = distances.arrange_row(i);
        minimal_distances_indices = instance_distances.calculate_minimal_indices(nearest_neighbours_number + 1);

        for(size_t j = 0; j < nearest_neighbours_number; j++)
        {
            nearest_neighbors(i, j) = minimal_distances_indices[j + 1];
        }
    }

    return(nearest_neighbors);
}


// Vector<double> calculate_k_distances(const Matrix<double>&) const

/// Returns a vector with the k-distance of every instance in the data set, which is the distance between every
/// instance and k-th nearest neighbor.
/// @param distances Distances between every instance in the data set.
/// @param nearest_neighbours_number Number of nearest neighbors to be calculated.

Vector<double> DataSet::calculate_k_distances(const Matrix<double>& distances, const size_t& nearest_neighbours_number) const
{
    const size_t instances_number = instances.count_used_instances_number();

    const Matrix<size_t> nearest_neighbors = calculate_nearest_neighbors(distances, nearest_neighbours_number);

    size_t maximal_index;

    Vector<double> k_distances(instances_number);

    for(size_t i = 0; i < instances_number; i++)
    {
        maximal_index = nearest_neighbors(i, nearest_neighbours_number - 1);

        k_distances[i] = distances.arrange_row(i)[maximal_index];
    }

    return(k_distances);
}


// Matrix<double> calculate_reachability_distance(const Matrix<double>&, Vector<double>&) const

/// Calculates the reachability distances for the instances in the data set.
/// @param distances Distances between every instance.
/// @param k_distances Distances of the k-th nearest neighbors.

Matrix<double> DataSet::calculate_reachability_distances(const Matrix<double>& distances, const Vector<double>& k_distances) const
{
    const size_t instances_number = instances.count_used_instances_number();

    Matrix<double> reachability_distances(instances_number, instances_number);

    for(size_t i = 0; i < instances_number; i++)
    {
        for(size_t j = i; j < instances_number; j++)
        {
            if(distances(i, j) <= k_distances[i])
            {
                reachability_distances(i, j) = k_distances[i];
                reachability_distances(j, i) = k_distances[i];
            }
            else if(distances(i, j) > k_distances[i])
            {
                reachability_distances(i, j) = distances(i, j);
                reachability_distances(j, i) = distances(j, i);
            }
         }
    }

    return (reachability_distances);
}


// Vector<double> calculate_reachability_density(const Matrix<double>&, const size_t&) const

/// Calculates reachability density for every element of the data set.
/// @param distances Distances between every instance in the data set.
/// @param nearest_neighbours_number Number of nearest neighbors to be calculated.

Vector<double> DataSet::calculate_reachability_density(const Matrix<double>& distances, const size_t& nearest_neighbours_number) const
{
   const size_t instances_number = instances.count_used_instances_number();

   const Vector<double> k_distances = calculate_k_distances(distances, nearest_neighbours_number);

   const Matrix<double> reachability_distances = calculate_reachability_distances(distances, k_distances);

   const Matrix<size_t> nearest_neighbors_indices = calculate_nearest_neighbors(distances, nearest_neighbours_number);

   Vector<double> reachability_density(instances_number);

   Vector<size_t> nearest_neighbors_instance;

   for(size_t i = 0; i < instances_number; i++)
   {
       nearest_neighbors_instance = nearest_neighbors_indices.arrange_row(i);

       reachability_density[i] = nearest_neighbours_number/ reachability_distances.arrange_row(i).calculate_partial_sum(nearest_neighbors_instance);
   }

   return (reachability_density);

}


// Vector<double> calculate_local_outlier_factor(const size_t&) const

/// Returns a vector with the local outlier factors for every used instance.
/// @param nearest_neighbours_number Number of neighbors to be calculated.

Vector<double> DataSet::calculate_local_outlier_factor(const size_t& nearest_neighbours_number) const
{
    const size_t instances_number = instances.count_used_instances_number();

    const Matrix<double> distances = calculate_instances_distances(nearest_neighbours_number);
    const Vector<double> reachability_density = calculate_reachability_density(distances, nearest_neighbours_number);

    const Matrix<size_t> nearest_neighbors = calculate_nearest_neighbors(distances, nearest_neighbours_number);

    Vector<size_t> instance_nearest_neighbors(nearest_neighbours_number);

    Vector<double> local_outlier_factor(instances_number);

    for(size_t i = 0; i < instances_number; i++)
    {
        instance_nearest_neighbors = nearest_neighbors.arrange_row(i);

        local_outlier_factor[i] = (reachability_density.calculate_partial_sum(instance_nearest_neighbors))/(nearest_neighbours_number*reachability_density[i]);
    }

    return (local_outlier_factor);
}


// Vector<size_t> clean_local_outlier_factor(const size_t&)

/// Removes the outliers from the data set using the local outlier factor method.
/// @param nearest_neighbours_number Number of nearest neighbros to calculate
/// @todo

Vector<size_t> DataSet::clean_local_outlier_factor(const size_t& nearest_neighbours_number)
{
    Vector<size_t> unused_instances;

    const Vector<double> local_outlier_factor = calculate_local_outlier_factor(nearest_neighbours_number);

    const size_t instances_number = instances.count_used_instances_number();
    const Vector<size_t> instances_indices = instances.arrange_used_indices();

    for(size_t i = 0; i < instances_number; i++)
    {
        if(local_outlier_factor[i] > 1.6001)
        {
            instances.set_use(instances_indices[i], Instances::Unused);

            unused_instances.push_back(instances_indices[i]);
        }
    }

    return(unused_instances);
}

// Vector<size_t> calculate_Tukey_outliers(const size_t&, const double&) const method

/// Calculate the outliers from the data set using the Tukey's test for a single variable.
/// @param variable_index Index of the variable to calculate the outliers.
/// @param cleaning_parameter Parameter used to detect outliers.

Vector<size_t> DataSet::calculate_Tukey_outliers(const size_t& variable_index, const double& cleaning_parameter) const
{
    const size_t instances_number = instances.count_used_instances_number();
    const Vector<size_t> instances_indices = instances.arrange_used_indices();

    double interquartile_range;

    Vector<size_t> unused_instances_indices;

    if(is_binary_variable(variable_index))
    {
        return(unused_instances_indices);
    }

    const Vector<double> box_plot = data.arrange_column(variable_index).calculate_box_plots();

    if(box_plot[3] == box_plot[1])
    {
        return(unused_instances_indices);
    }
    else
    {
        interquartile_range = std::abs((box_plot[3] - box_plot[1]));
    }

    for(size_t j = 0; j < instances_number; j++)
    {
        const Vector<double> instance = get_instance(instances_indices[j]);

        if(instance[variable_index] < (box_plot[1] - cleaning_parameter*interquartile_range))
        {
            unused_instances_indices.push_back(instances_indices[j]);
        }
        else if(instance[variable_index] > (box_plot[3] + cleaning_parameter*interquartile_range))
        {
            unused_instances_indices.push_back(instances_indices[j]);
        }
    }

    return(unused_instances_indices);
}


// Vector< Vector<size_t> > calculate_Tukey_outliers(const double&) const

/// Calculate the outliers from the data set using the Tukey's test.
/// @param cleaning_parameter Parameter used to detect outliers.

Vector< Vector<size_t> > DataSet::calculate_Tukey_outliers(const double& cleaning_parameter) const
{
    const size_t instances_number = instances.count_used_instances_number();
    const Vector<size_t> instances_indices = instances.arrange_used_indices();

    const size_t variables_number = variables.count_used_variables_number();
    const Vector<size_t> used_variables_indices = variables.arrange_used_indices();

    double interquartile_range;

    Vector< Vector<size_t> > return_values(2);
    return_values[0] = Vector<size_t>(instances_number, 0);
    return_values[1] = Vector<size_t>(variables_number, 0);

    size_t variable_index;

    Vector< Vector<double> > box_plots(variables_number);

#pragma omp parallel for private(variable_index) schedule(dynamic)

    for(int i = 0; i < (int)variables_number; i++)
    {
        variable_index = used_variables_indices[i];

        if(is_binary_variable(variable_index))
        {
            continue;
        }

        box_plots[i] = data.arrange_column(variable_index).calculate_box_plots();
    }

    for(int i = 0; i < (int)variables_number; i++)
    {
        variable_index = used_variables_indices[i];

        if(is_binary_variable(variable_index))
        {
            continue;
        }

        const Vector<double> variable_box_plot = box_plots[i];

        if(variable_box_plot[3] == variable_box_plot[1])
        {
            continue;
        }
        else
        {
            interquartile_range = std::abs((variable_box_plot[3] - variable_box_plot[1]));
        }

        size_t variables_outliers = 0;

#pragma omp parallel for schedule(dynamic) reduction(+ : variables_outliers)

        for(int j = 0; j < instances_number; j++)
        {
            const Vector<double> instance = get_instance(instances_indices[j]);

            if(instance[variable_index] < (variable_box_plot[1] - cleaning_parameter*interquartile_range) ||
               instance[variable_index] > (variable_box_plot[3] + cleaning_parameter*interquartile_range))
            {
                    return_values[0][j] = 1;

                    variables_outliers++;
            }
        }

        return_values[1][i] = variables_outliers;
    }

    return(return_values);
}


// Matrix<double> calculate_autocorrelation(const size_t&) const method

/// Returns a matrix with the values of autocorrelation for every variable in the data set.
/// The number of rows is equal to the number of instances.
/// The number of columns is the maximum lags number.
/// @param maximum_lags_number Maximum lags number for which autocorrelation is calculated.

Matrix<double> DataSet::calculate_autocorrelation(const size_t& maximum_lags_number) const
{
    if(lags_number > instances.count_used_instances_number())
    {
        std::ostringstream buffer;

        buffer << "OpenNN Exception: DataSet class.\n"
               << "Matrix<double> calculate_autocorrelation(const size_t&) method.\n"
               << "Maximum lags number (" << maximum_lags_number << ") is greater than the number of instances (" << instances.count_used_instances_number() <<") \n";

        throw std::logic_error(buffer.str());
    }

    const size_t variables_number = time_series_data.get_columns_number();

    Matrix<double> autocorrelation(variables_number, maximum_lags_number);

    for(size_t i = 0; i < maximum_lags_number; i++)
    {
        for(size_t j = 0; j < variables_number; j++)
        {
            autocorrelation.set_row(j, time_series_data.arrange_column(j).calculate_autocorrelation(maximum_lags_number));
        }
    }

    return autocorrelation;
}


// Matrix< Vector<double> > calculate_cross_correlation(void)

/// Calculates the cross-correlation between all the variables in the data set.

Matrix< Vector<double> > DataSet::calculate_cross_correlation(void) const
{
    const size_t variables_number = variables.count_used_variables_number()/(lags_number + steps_ahead);

    Matrix< Vector<double> > cross_correlation(variables_number, variables_number);

    Vector<double> actual_column;

    for(size_t i = 0; i < variables_number; i++)
    {
        actual_column = time_series_data.arrange_column(i);

        for(size_t j = 0; j < variables_number; j++)
        {
            cross_correlation(i , j) = actual_column.calculate_cross_correlation(time_series_data.arrange_column(j));
        }
    }

    return(cross_correlation);
}


// void generate_data_approximation(const size_t&, const size_t&) method

/// Generates an artificial dataset with a given number of instances and number of variables
/// using the Rosenbrock function.
/// @param instances_number Number of instances in the dataset.
/// @param variables_number Number of variables in the dataset.

void DataSet::generate_data_approximation(const size_t& instances_number, const size_t& variables_number)
{
    const size_t inputs_number = variables_number-1;
    const size_t targets_number = 1;

//    Matrix<double> input_data(instances_number, inputs_number);
//    input_data.randomize_uniform(-2.048, 2.048);

//    Matrix<double> target_data(instances_number, targets_number);

//    Matrix<double> new_data(instances_number, inputs_number+targets_number);

    data.set(instances_number, variables_number);

    data.randomize_uniform(-2.048, 2.048);

    double rosenbrock;

    for(size_t i = 0; i < instances_number; i++)
    {
        data(i, inputs_number) = data.arrange_row(i, Vector<size_t>(0,1,inputs_number-1)).calculate_norm();

        rosenbrock = 0.0;

        for(size_t j = 0; j < inputs_number-1; j++)
        {
            rosenbrock +=
            (1.0 - data(i,j))*(1.0 - data(i,j))
            + 100.0*(data(i,j+1)-data(i,j)*data(i,j))*(data(i,j+1)-data(i,j)*data(i,j));
        }

        data(i, inputs_number) = rosenbrock;
//        target_data(i, 0) = input_data.arrange_row(i).calculate_norm();

//        rosenbrock = 0.0;

//        for(size_t j = 0; j < inputs_number-1; j++)
//        {
//            rosenbrock +=
//            (1.0 - input_data(i,j))*(1.0 - input_data(i,j))
//            + 100.0*(input_data(i,j+1)-input_data(i,j)*input_data(i,j))*(input_data(i,j+1)-input_data(i,j)*input_data(i,j));
//        }

//        target_data(i, 0) = rosenbrock;
    }

//    set(input_data.assemble_columns(target_data));

//    set(new_data);

    data.scale_minimum_maximum();
}


// void generate_data_binary_classification(const size_t&, const size_t&) method

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

    class_0.append_column(target_0);

    // Positives data

    Vector<double> target_1(positives, 1.0);

    Matrix<double> class_1(positives, inputs_number);

    class_1.randomize_normal(0.5, 1.0);

    class_1.append_column(target_1);

    // Assemble

    set(class_0.assemble_rows(class_1));

}


// void generate_data_multiple_classification(const size_t&, const size_t&) method

/// @todo

void DataSet::generate_data_multiple_classification(const size_t&, const size_t&)
{

}


// bool has_data(void) const method

/// Returns true if the data matrix is not empty (it has not been loaded),
/// and false otherwise.

bool DataSet::has_data(void) const
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


// Vector<size_t> filter_data(const Vector<double>&, const Vector<double>&) method

/// Unuses those instances with values outside a defined range.
/// @param minimums Vector of minimum values in the range.
/// The size must be equal to the number of variables.
/// @param maximums Vector of maximum values in the range.
/// The size must be equal to the number of variables.

Vector<size_t> DataSet::filter_data(const Vector<double>& minimums, const Vector<double>& maximums)
{
    const size_t variables_number = variables.get_variables_number();

    // Control sentence (if debug)

    #ifdef __OPENNN_DEBUG__

    if(minimums.size() != variables_number)
    {
        std::ostringstream buffer;

        buffer << "OpenNN Exception: DataSet class.\n"
               << "Vector<size_t> filter_data(const Vector<double>&, const Vector<double>&) method.\n"
               << "Size of minimums (" << minimums.size() << ") is not equal to number of variables (" << variables_number << ").\n";

        throw std::logic_error(buffer.str());
    }

    if(maximums.size() != variables_number)
    {
        std::ostringstream buffer;

        buffer << "OpenNN Exception: DataSet class.\n"
               << "Vector<size_t> filter_data(const Vector<double>&, const Vector<double>&) method.\n"
               << "Size of maximums (" << maximums.size() << ") is not equal to number of variables (" << variables_number << ").\n";

        throw std::logic_error(buffer.str());
    }

    #endif

//    const Vector< Vector<size_t> > missing_indices = missing_values.arrange_missing_indices();

    Vector<size_t> filtered_indices;

    const size_t instances_number = instances.get_instances_number();

#pragma omp parallel for

    for(int i = 0; i < (int)instances_number; i++)
    {
        if(instances.is_unused(i))
        {
            continue;
        }

        for(size_t j = 0; j < variables_number; j++)
        {
            if(missing_values.is_missing_value(i, j))
            {
                continue;
            }

            if(data(i,j) < minimums[j] || data(i,j) > maximums[j])
            {
                #pragma omp critical
                {
                filtered_indices.push_back(i);
                }
                instances.set_use(i, Instances::Unused);

                break;
            }
        }
    }

    return(filtered_indices);
}


// void convert_angular_variable_degrees(const size_t&) method

/// Replaces a given angular variable expressed in degrees by the sinus and cosinus of that variable.
/// This solves the discontinuity associated with angular variables.
/// Note that this method modifies the number of variables.
/// @param variable_index Index of angular variable.

void DataSet::convert_angular_variable_degrees(const size_t& variable_index)
{
    // Control sentence (if debug)

    #ifdef __OPENNN_DEBUG__

    const size_t variables_number = variables.get_variables_number();

    if(variable_index >= variables_number)
    {
        std::ostringstream buffer;

        buffer << "OpenNN Exception: DataSet class.\n"
               << "void convert_angular_variable_degrees(const size_t&) method.\n"
               << "Index of variable (" << variable_index << ") must be less than number of variables (" << variables_number << ").\n";

        throw std::logic_error(buffer.str());
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


// void convert_angular_variable_radians(const size_t&) method

/// Replaces a given angular variable expressed in radians by the sinus and cosinus of that variable.
/// This solves the discontinuity associated with angular variables.
/// Note that this method modifies the number of variables.
/// @param variable_index Index of angular variable.

void DataSet::convert_angular_variable_radians(const size_t& variable_index)
{
    // Control sentence (if debug)

    #ifdef __OPENNN_DEBUG__

    const size_t variables_number = variables.get_variables_number();

    if(variable_index >= variables_number)
    {
        std::ostringstream buffer;

        buffer << "OpenNN Exception: DataSet class.\n"
               << "void convert_angular_variable_radians(const size_t&) method.\n"
               << "Index of variable (" << variable_index << ") must be less than number of variables (" << variables_number << ").\n";

        throw std::logic_error(buffer.str());
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


// void convert_angular_variables_degrees(const Vector<size_t>&)

/// Replaces a given set of angular variables expressed in degrees by the sinus and cosinus of that variable.
/// This solves the discontinuity associated with angular variables.
/// Note that this method modifies the number of variables.
/// @param indices Indices of angular variables.

void DataSet::convert_angular_variables_degrees(const Vector<size_t>& indices)
{
    // Control sentence (if debug)

    #ifdef __OPENNN_DEBUG__

    const size_t variables_number = variables.get_variables_number();

    for(size_t i = 0; i < indices.size(); i++)
    {
        if(indices[i] >= variables_number)
        {
            std::ostringstream buffer;

            buffer << "OpenNN Exception: DataSet class.\n"
                   << "void convert_angular_variables_degrees(const Vector<size_t>&) method.\n"
                   << "Index (" << i << ") must be less than number of variables (" << variables_number << ").\n";

            throw std::logic_error(buffer.str());
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


// void convert_angular_variables_radians(const Vector<size_t>&)

/// Replaces a given set of angular variables expressed in radians by the sinus and cosinus of that variable.
/// This solves the discontinuity associated with angular variables.
/// Note that this method modifies the number of variables.
/// @param indices Indices of angular variables.

void DataSet::convert_angular_variables_radians(const Vector<size_t>& indices)
{
    // Control sentence (if debug)

    #ifdef __OPENNN_DEBUG__

    const size_t variables_number = variables.get_variables_number();

    for(size_t i = 0; i < indices.size(); i++)
    {
        if(indices[i] >= variables_number)
        {
            std::ostringstream buffer;

            buffer << "OpenNN Exception: DataSet class.\n"
                   << "void convert_angular_variables_radians(const Vector<size_t>&) method.\n"
                   << "Index (" << i << ") must be less than number of variables (" << variables_number << ").\n";

            throw std::logic_error(buffer.str());
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


// void convert_angular_variables(void) method

/// Replaces a given set of angular variables by the sinus and cosinus of that variable, according to the angular units used.
/// This solves the discontinuity associated with angular variables.
/// Note that this method modifies the number of variables.

void DataSet::convert_angular_variables(void)
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

       default:
       {
          std::ostringstream buffer;

          buffer << "OpenNN Exception: DataSet class.\n"
                 << "void convert_angular_variables(void) method.\n"
                 << "Unknown angular units.\n";

          throw std::logic_error(buffer.str());
       }
       break;
    }

}


// void scrub_missing_values_unuse(void) method

/// Sets all the instances with missing values to "Unused".

void DataSet::scrub_missing_values_unuse(void)
{
    const Vector<size_t> missing_instances = missing_values.arrange_missing_instances();

    for(size_t i = 0; i < missing_instances.size(); i++)
    {
        instances.set_use(missing_instances[i], Instances::Unused);
    }
}


// void scrub_missing_values_mean(void) method

/// Substitutes all the missing values by the mean of the corresponding variable.

void DataSet::scrub_missing_values_mean(void)
{
    const Vector< Vector<size_t> > missing_indices = missing_values.arrange_missing_indices();

    const Vector<double> means = data.calculate_mean_missing_values(missing_indices);

    const size_t variables_number = variables.get_variables_number();

    const Vector<size_t> targets_indices = variables.arrange_targets_indices();

    size_t instance_index;

    for(size_t i = 0; i < variables_number; i++)
    {
        for(size_t j = 0; j < missing_indices[i].size(); j++)
        {
            instance_index = missing_indices[i][j];

            data(instance_index, i) = means[i];
        }
    }
}


// void scrub_missing_values(void) method

/// General method for dealing with missing values.
/// It switches among the different scrubbing methods available,
/// according to the corresponding value in the missing values object.

void DataSet::scrub_missing_values(void)
{
    const MissingValues::ScrubbingMethod scrubbing_method = missing_values.get_scrubbing_method();

    switch(scrubbing_method)
    {
       case MissingValues::Unuse:
       {
            scrub_missing_values_unuse();
       }
       break;

       case MissingValues::Mean:
       {
            scrub_missing_values_mean();
       }
       break;

       default:
       {
          std::ostringstream buffer;

          buffer << "OpenNN Exception: DataSet class\n"
                 << "void scrub_missing_values(void) method.\n"
                 << "Unknown scrubbing method.\n";

          throw std::logic_error(buffer.str());
       }
       break;
    }
}


// size_t count_tokens(std::string& str) const method

/// Returns the number of strings delimited by separator.
/// If separator does not match anywhere in the string, this method returns 0.
/// @param str String to be tokenized.

size_t DataSet::count_tokens(std::string& str) const
{
//    if(!(this->find(separator) != std::string::npos))
//    {
//        std::ostringstream buffer;
//
//        buffer << "OpenNN Exception:\n"
//               << "std::string class.\n"
//               << "inline size_t count_tokens(const std::string&) const method.\n"
//               << "Separator not found in string: \"" << separator << "\".\n";
//
//        throw std::logic_error(buffer.str());
//    }

    trim(str);

    size_t tokens_count = 0;

    // Skip delimiters at beginning.

    const std::string separator_string = get_separator_string();

    std::string::size_type last_pos = str.find_first_not_of(separator_string, 0);

    // Find first "non-delimiter".

    std::string::size_type pos = str.find_first_of(separator_string, last_pos);

    while (std::string::npos != pos || std::string::npos != last_pos)
    {
        // Found a token, add it to the vector

        tokens_count++;

        // Skip delimiters.  Note the "not_of"

        last_pos = str.find_first_not_of(separator_string, pos);

        // Find next "non-delimiter"

        pos = str.find_first_of(separator_string, last_pos);
    }

    return(tokens_count);
}


/// Splits the string into substrings (tokens) wherever separator occurs, and returns a vector with those strings.
/// If separator does not match anywhere in the string, this method returns a single-element list containing this string.
/// @param str String to be tokenized.

Vector<std::string> DataSet::get_tokens(const std::string& str) const
{   
    const std::string new_string = get_trimmed(str);

    Vector<std::string> tokens;

    const std::string separator_string = get_separator_string();

    // Skip delimiters at beginning.

    std::string::size_type lastPos = new_string.find_first_not_of(separator_string, 0);

    // Find first "non-delimiter"

    std::string::size_type pos = new_string.find_first_of(separator_string, lastPos);

    while(std::string::npos != pos || std::string::npos != lastPos)
    {
        // Found a token, add it to the vector

        tokens.push_back(new_string.substr(lastPos, pos - lastPos));

        // Skip delimiters. Note the "not_of"

        lastPos = new_string.find_first_not_of(separator_string, pos);

        // Find next "non-delimiter"

        pos = new_string.find_first_of(separator_string, lastPos);
    }

    for(size_t i = 0; i < tokens.size(); i++)
    {
        trim(tokens[i]);
    }

    return(tokens);
}


// bool is_numeric(const std::string&) const method

/// Returns true if the string passed as argument represents a number, and false otherwise.
/// @param str String to be checked.

bool DataSet::is_numeric(const std::string& str) const
{
    std::istringstream iss(str.data());

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


// void DataSet::trim(std::string&) const method

/// Removes whitespaces from the start and the end of the string passed as argument.
/// This includes the ASCII characters "\t", "\n", "\v", "\f", "\r", and " ".
/// @param str String to be checked.

void DataSet::trim(std::string& str) const
{
    //prefixing spaces

    str.erase(0, str.find_first_not_of(' '));

    //surfixing spaces

    str.erase(str.find_last_not_of(' ') + 1);
}


/// Returns a string that has whitespace removed from the start and the end.
/// This includes the ASCII characters "\t", "\n", "\v", "\f", "\r", and " ".
/// @param str String to be checked.

std::string DataSet::get_trimmed(const std::string& str) const
{
    std::string output(str);

    //prefixing spaces

    output.erase(0, output.find_first_not_of(' '));

    //surfixing spaces

    output.erase(output.find_last_not_of(' ') + 1);

    return(output);
}


// std::string prepend(const std::string&, const std::string&) const method

/// Prepends the string pre to the beginning of the string str and returns the whole string.
/// @param pre String to be prepended.
/// @param str original string.

std::string DataSet::prepend(const std::string& pre, const std::string& str) const
{
    std::ostringstream buffer;

    buffer << pre << str;

    return(buffer.str());
}


// bool is_numeric(const Vector<std::string>&) const

/// Returns true if all the elements in a string list are numeric, and false otherwise.
/// @param v String list to be checked.

bool DataSet::is_numeric(const Vector<std::string>& v) const
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


// bool is_not_numeric(const Vector<std::string>&) const

/// Returns true if none element in a string list is numeric, and false otherwise.
/// @param v String list to be checked.

bool DataSet::is_not_numeric(const Vector<std::string>& v) const
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


// bool is_mixed(const Vector<std::string>&) const

/// Returns true if some the elements in a string list are numeric and some others are not numeric.
/// @param v String list to be checked.

bool DataSet::is_mixed(const Vector<std::string>& v) const
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
