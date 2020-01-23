//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   D A T A   S E T   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#ifndef DATASET_H
#define DATASET_H

// System includes

#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <cmath>
#include <algorithm>
#include <cstdlib>
#include <stdexcept>
#include <ctime>
#include <exception>
#include <regex>
//#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <limits.h>

// OpenNN includes

#include "metrics.h"
#include "statistics.h"
#include "transformations.h"
#include "correlations.h"
#include "opennn_strings.h"
#include "tinyxml2.h"
#include "config.h"

// Eigen includes

#include "../eigen/Eigen/Eigen"
#include "../eigen/unsupported/Eigen/CXX11/Tensor"

#ifdef __OPENNN_CUDA__
    #include "D:/artelnics/opennn_cuda/opennn_cuda/kernels.h"
#endif

using namespace std;
using namespace Eigen;

namespace OpenNN
{

/// This class represents the concept of data set for data modelling problems, such as function regression, classification, time series prediction, images approximation and images classification.

///
/// It basically consists of a data Matrix separated by columns.
/// These columns can take different categories depending on the data hosted in them.
///
/// With OpenNN DataSet class you can edit the data to prepare your model, such as eliminating missing values,
/// calculating correlations between variables (inputs and targets), not using certain variables or instances, etc \dots.

class DataSet
{

public:

   // Constructors

   explicit DataSet();

   explicit DataSet(const MatrixXd&);

   explicit DataSet(const Tensor<type, 2>&);

   explicit DataSet(const int&, const int&);

   explicit DataSet(const int&, const int&, const int&);

   explicit DataSet(const tinyxml2::XMLDocument&);

   explicit DataSet(const string&, const char&, const bool&);

   DataSet(const DataSet&);

   // Destructor

   virtual ~DataSet();

   // Enumerations

   /// Enumeration of available separators for the data file.

   enum Separator{Space, Tab, Comma, Semicolon};

   /// Enumeration of available methods for missing values in the data.

   enum MissingValuesMethod{Unuse, Mean, Median};

   /// Enumeration of available methods for scaling and unscaling the data.

   enum ScalingUnscalingMethod{NoScaling, NoUnscaling, MinimumMaximum, MeanStandardDeviation, StandardDeviation, Logarithmic};

   /// Enumeration of the learning tasks.

   enum ProjectType{Approximation, Classification, Forecasting, ImageApproximation, ImageClassification};

   /// This enumeration represents the possible uses of an instance
   /// (training, selection, testing or unused).

   enum InstanceUse{Training, Selection, Testing, UnusedInstance};

   /// This enumeration represents the possible uses of an variable
   /// (input, target, time or unused).

   enum VariableUse{Input, Target, Time, UnusedVariable};

   /// This enumeration represents the data type of a column
   /// (numeric, binary, categorical or time).

   enum ColumnType{Numeric, Binary, Categorical, DateTime};

   // Structs

   /// This structure represents the columns of the DataSet.

   struct Column
   {
       /// Default constructor.

       Column();

       /// Values constructor

       Column(const string&, const VariableUse&, const ColumnType& = Numeric, const Tensor<string, 1>& = Tensor<string, 1>(), const Tensor<VariableUse, 1>& = Tensor<VariableUse, 1>());

       /// Destructor.

       virtual ~Column();

       /// Column name.

       string name;

       /// Column use.

       VariableUse column_use;

       /// Column type.

       ColumnType type;

       /// Categories within the column.

       Tensor<string, 1> categories;

       /// Categories use.

       Tensor<VariableUse, 1> categories_uses;

       // Methods

       int get_categories_number() const;

       Tensor<string, 1> get_used_variables_names() const;

       void set_use(const VariableUse&);
       void set_use(const string&);

       void set_type(const string&);

       void set_categories_uses(const Tensor<string, 1>&);

       bool is_used();
       bool is_unused();

       void from_XML(const tinyxml2::XMLDocument&);
       void write_XML(tinyxml2::XMLPrinter&) const;
   };


   struct Batch
   {
       /// Default constructor.

       Batch() {}

       Batch(DataSet* new_data_set_pointer)
       {
           data_set_pointer = new_data_set_pointer;

           allocate();
       }

       /// Destructor.

       virtual ~Batch() {}

       void allocate()
       {
           const Index batch_instances_number = data_set_pointer->get_batch_instances_number();
           const int input_variables_number = data_set_pointer->get_input_variables_number();
           const int target_variables_number = data_set_pointer->get_target_variables_number();

           const Tensor<int, 1> input_variables_dimensions = data_set_pointer->get_input_variables_dimensions();
           const Tensor<int, 1> target_variables_dimensions = data_set_pointer->get_target_variables_dimensions();

           inputs_2d = Tensor<type, 2>(batch_instances_number, input_variables_number);
           targets_2d = Tensor<type, 2>(batch_instances_number, target_variables_number);
       }

       void print()
       {
           cout << "Batch structure" << endl;

           cout << "Inputs:" << endl;
           cout << inputs_2d << endl;

           cout << "Targets:" << endl;
           cout << targets_2d << endl;
       }

       void fill(const Tensor<int, 1>& rows, const Tensor<int, 1>& inputs, const Tensor<int, 1>& targets)
       {
           inputs_2d.setRandom();
           targets_2d.setRandom();

/*
           const size_t rows_number = rows.size();
           const size_t inputs_number = inputs.size();
           const size_t targets_number = targets.size();

           const Tensor<type, 2>& data = data_set_pointer->get_data();

           for(size_t i = 0; i < rows_number; i++)
           {
               for(size_t j = 0; j < inputs_number; j++)
               {
                   inputs_2d(i,j) = data(rows[i], inputs[j]);
               }

               for(size_t j = 0; j < targets_number; j++)
               {
                   targets_2d(i,j) = data(rows[i], targets[j]);
               }
           }
*/
       }

       DataSet* data_set_pointer = nullptr;

       Tensor<type, 2> inputs_2d;
       Tensor<type, 2> targets_2d;
   };

   // Instances get methods

   inline int get_instances_number() const {return instances_uses.size();}

   int get_training_instances_number() const;
   int get_selection_instances_number() const;
   int get_testing_instances_number() const;

   int get_used_instances_number() const;
   int get_unused_instances_number() const;

   Tensor<int, 1> get_training_instances_indices() const;
   Tensor<int, 1> get_selection_instances_indices() const;
   Tensor<int, 1> get_testing_instances_indices() const;

   Tensor<int, 1> get_used_instances_indices() const;
   Tensor<int, 1> get_unused_instances_indices() const;

   InstanceUse get_instance_use(const int&) const;
   const vector<InstanceUse>& get_instances_uses() const;

   Tensor<int, 1> get_instances_uses_numbers() const;
   Tensor<type, 1> get_instances_uses_percentages() const;

   int get_batch_instances_number() const {return batch_instances_number;}

   // Columns get methods

   vector<Column> get_columns() const;
   vector<Column> get_used_columns() const;

   int get_columns_number() const;

   int get_input_columns_number() const;
   int get_target_columns_number() const;
   int get_time_columns_number() const;
   int get_unused_columns_number() const;
   int get_used_columns_number() const;

   int get_column_index(const string&) const;

   Tensor<int, 1> get_input_columns_indices() const;
   Tensor<int, 1> get_target_columns_indices() const;
   Tensor<int, 1> get_unused_columns_indices() const;
   Tensor<int, 1> get_used_columns_indices() const;

   Tensor<string, 1> get_columns_names() const;

   Tensor<string, 1> get_input_columns_names() const;
   Tensor<string, 1> get_target_columns_names() const;
   Tensor<string, 1> get_used_columns_names() const;

   ColumnType get_column_type(const int& index) const {return columns[index].type;}

   VariableUse get_column_use(const int &) const;
   Tensor<VariableUse, 1> get_columns_uses() const;

   // Variables get methods

   int get_variables_number() const;

   int get_input_variables_number() const;
   int get_target_variables_number() const;
   int get_unused_variables_number() const;
   int get_used_variables_number() const;

   string get_variable_name(const int&) const;
   Tensor<string, 1> get_variables_names() const;

   Tensor<string, 1> get_input_variables_names() const;
   Tensor<string, 1> get_target_variables_names() const;

   int get_variable_index(const string&) const;

   Tensor<int, 1> get_variable_indices(const int&) const;
   Tensor<int, 1> get_unused_variables_indices() const;
   Tensor<int, 1> get_input_variables_indices() const;
   Tensor<int, 1> get_target_variables_indices() const;

   VariableUse get_variable_use(const int&) const;
   Tensor<VariableUse, 1> get_variables_uses() const;

   const Tensor<int, 1>& get_input_variables_dimensions() const;
   const Tensor<int, 1>& get_target_variables_dimensions() const;

   // Batches get methods

   inline int get_batch_instances_number() {return batch_instances_number;}

   Tensor<Index, 2> get_training_batches(const bool& = true) const;
   Tensor<Index, 2> get_selection_batches(const bool& = true) const;
   Tensor<Index, 2> get_testing_batches(const bool& = true) const;

   // Data get methods

   const Tensor<type, 2>& get_data() const;

   const Tensor<type, 2>& get_time_series_data() const;

   Tensor<type, 2> get_training_data() const;
   Tensor<type, 2> get_selection_data() const;
   Tensor<type, 2> get_testing_data() const;

   Tensor<type, 2> get_input_data() const;
   Tensor<type, 2> get_target_data() const;

   Tensor<type, 2> get_input_data(const Tensor<int, 1>&) const;
   Tensor<type, 2> get_target_data(const Tensor<int, 1>&) const;

   Matrix<float, Dynamic, Dynamic> get_input_data_float(const Tensor<int, 1>&) const;
   Matrix<float, Dynamic, Dynamic> get_target_data_float(const Tensor<int, 1>&) const;

   Tensor<type, 2> get_training_input_data() const;
   Tensor<type, 2> get_training_target_data() const;

   Tensor<type, 2> get_selection_input_data() const;
   Tensor<type, 2> get_selection_target_data() const;

   Tensor<type, 2> get_testing_input_data() const;
   Tensor<type, 2> get_testing_target_data() const;

   Tensor<type, 1> get_instance_data(const int&) const;
   Tensor<type, 1> get_instance_data(const int&, const Tensor<int, 1>&) const;
   Tensor<type, 2> get_instance_input_data(const int&) const;
   Tensor<type, 2> get_instance_target_data(const int&) const;

   Tensor<type, 2> get_column_data(const int&) const;
   Tensor<type, 2> get_column_data(const Tensor<int, 1>&) const;
   Tensor<type, 2> get_column_data(const string&) const;

   Tensor<type, 1> get_variable_data(const int&) const;
   Tensor<type, 1> get_variable_data(const string&) const;

   Tensor<type, 1> get_variable_data(const int&, const Tensor<int, 1>&) const;
   Tensor<type, 1> get_variable_data(const string&, const Tensor<int, 1>&) const;

   Tensor<type, 2> get_data_subtensor(const Tensor<int, 1>&, const Tensor<int, 1>&) const;

   // Members get methods

   MissingValuesMethod get_missing_values_method() const;

   const string& get_data_file_name() const;

   const bool& get_header_line() const;
   const bool& get_rows_label() const;

   const Separator& get_separator() const;
   char get_separator_char() const;
   string get_separator_string() const;

   const string& get_missing_values_label() const;

   const int& get_lags_number() const;
   const int& get_steps_ahead() const;
   const int& get_time_index() const;

   static Tensor<string, 1> get_default_columns_names(const int&);

   static ScalingUnscalingMethod get_scaling_unscaling_method(const string&);

   int get_gmt() const;

   const bool& get_display() const;

   // Set methods

   void set();
   void set(const Tensor<type, 2>&);
   void set(const MatrixXd&);
   void set(const int&, const int&);
   void set(const int&, const int&, const int&);
   void set(const DataSet&);
   void set(const tinyxml2::XMLDocument&);
   void set(const string&);

   void set_default();

   // Instances set methods

   void set_instances_number(const int&);

   void set_training();
   void set_selection();
   void set_testing();

   void set_training(const Tensor<int, 1>&);
   void set_selection(const Tensor<int, 1>&);
   void set_testing(const Tensor<int, 1>&);

   void set_instances_unused();
   void set_instances_unused(const Tensor<int, 1>&);

   void set_instance_use(const int&, const InstanceUse&);
   void set_instance_use(const int&, const string&);

   void set_instances_uses(const vector<InstanceUse>&);
   void set_instances_uses(const Tensor<string, 1>&);

   void set_testing_to_selection_instances();
   void set_selection_to_testing_instances();

   void set_batch_instances_number(const int&);

   void set_k_fold_cross_validation_instances_uses(const int&, const int&);

   // Columns set methods

   void set_default_columns_uses();

   void set_default_columns_names();

   void set_column_name(const int&, const string&);

   void set_columns_uses(const Tensor<string, 1>&);
   void set_columns_uses(const Tensor<VariableUse, 1>&);
   void set_columns_unused();
   void set_input_columns_unused();

   void set_column_use(const int&, const VariableUse&);
   void set_column_use(const string&, const VariableUse&);

   void set_columns_names(const Tensor<string, 1>&);

   void set_columns_number(const int&);

   void set_binary_simple_columns();

   // Variables set methods

   void set_variables_names(const Tensor<string, 1>&);
   void set_variable_name(const int&, const string&);

   void set_input();
   void set_target();
   void set_variables_unused();

   void set_input_variables_dimensions(const Tensor<int, 1>&);
   void set_target_variables_dimensions(const Tensor<int, 1>&);

   // Data set methods

   void set_data(const Tensor<type, 2>&);

   void set_instance(const int&, const Tensor<type, 1>&);

   // Members set methods

   void set_data_file_name(const string&);

   void set_has_columns_names(const bool&);
   void set_has_rows_label(const bool&);

   void set_separator(const Separator&);
   void set_separator(const string&);
   void set_separator(const char&);

   void set_missing_values_label(const string&);
   void set_missing_values_method(const MissingValuesMethod&);
   void set_missing_values_method(const string&);

   void set_lags_number(const int&);
   void set_steps_ahead_number(const int&);
   void set_time_index(const int&);

   void set_gmt(int&);

   void set_display(const bool&);

   // Check methods

   bool is_binary_classification() const;
   bool is_multiple_classification() const;

   bool is_empty() const;

   bool is_instance_used(const int&) const;
   bool is_instance_unused(const int&) const;

   bool has_data() const;

   bool has_categorical_variables() const;
   bool has_time_variables() const;

   // Splitting methods

   void split_instances_sequential(const double& training_ratio = 0.6, const double& selection_ratio = 0.2, const double& testing_ratio = 0.2);

   void split_instances_random(const double& training_ratio = 0.6, const double& selection_ratio = 0.2, const double& testing_ratio = 0.2);

   // Unusing methods

   Tensor<string, 1> unuse_constant_columns();

   Tensor<int, 1> unuse_repeated_instances();

   Tensor<int, 1> unuse_non_significant_input_columns();

   Tensor<int, 1> unuse_uncorrelated_columns(const double& = 0.25);

   Tensor<int, 1> unuse_most_populated_target(const int&);

   // Initialization methods

   void initialize_data(const double&);

   void randomize_data_uniform(const double& minimum = -1.0, const double& maximum = 1.0);
   void randomize_data_normal(const double& mean = 0.0, const double& standard_deviation = 1.0);

   // Descriptives methods

   vector<Descriptives> calculate_columns_descriptives() const;

   Tensor<type, 2> calculate_columns_descriptives_matrix() const;

   vector<Descriptives> calculate_columns_descriptives_positive_instances() const;
   vector<Descriptives> calculate_columns_descriptives_negative_instances() const;
   vector<Descriptives> calculate_columns_descriptives_classes(const int&) const;

   vector<Descriptives> calculate_columns_descriptives_training_instances() const;
   vector<Descriptives> calculate_columns_descriptives_selection_instances() const;
   vector<Descriptives> calculate_columns_descriptives_testing_instances() const;

   vector<Descriptives> calculate_input_variables_descriptives() const;
   vector<Descriptives> calculate_target_variables_descriptives() const;

   Tensor<type, 1> calculate_variables_means(const Tensor<int, 1>&) const;

   Descriptives calculate_inputs_descriptives(const int&) const;

   Tensor<type, 1> calculate_training_targets_mean() const;
   Tensor<type, 1> calculate_selection_targets_mean() const;
   Tensor<type, 1> calculate_testing_targets_mean() const;

   int calculate_training_negatives(const int&) const;
   int calculate_selection_negatives(const int&) const;
   int calculate_testing_negatives(const int&) const;

   // Histrogram methods

   vector<Histogram> calculate_columns_histograms(const int& = 10) const;

   // Box and whiskers

   vector<BoxPlot> calculate_columns_box_plots() const;

   // Inputs correlations

   Tensor<type, 2> calculate_inputs_correlations() const;

   void print_inputs_correlations() const;

   void print_top_inputs_correlations(const int& = 10) const;

   // Inputs-targets correlations

   Matrix<CorrelationResults, Dynamic, Dynamic> calculate_input_target_columns_correlations() const;
   Tensor<type, 2> calculate_input_target_columns_correlations_double() const;

   void print_input_target_columns_correlations() const;

   void print_top_input_target_columns_correlations(const int& = 10) const;

   // Principal components

   Tensor<type, 2> calculate_covariance_matrix() const;

   Tensor<type, 2> perform_principal_components_analysis(const double& = 0.0);

   Tensor<type, 2> perform_principal_components_analysis(const Tensor<type, 2>&, const Tensor<type, 1>&, const double& = 0.0);

   void transform_principal_components_data(const Tensor<type, 2>&);

   void subtract_inputs_mean();

   // Filtering methods

   Tensor<int, 1> filter_column(const int&, const double&, const double&);
   Tensor<int, 1> filter_column(const string&, const double&, const double&);

   Tensor<int, 1> filter_data(const Tensor<type, 1>&, const Tensor<type, 1>&);

   // Data scaling

   Tensor<string, 1> calculate_default_scaling_methods() const;
   void scale_data_minimum_maximum(const vector<Descriptives>&);
   void scale_data_mean_standard_deviation(const vector<Descriptives>&);
   vector<Descriptives> scale_data_minimum_maximum();
   vector<Descriptives> scale_data_mean_standard_deviation();

   // Input variables scaling

   void scale_inputs_mean_standard_deviation(const vector<Descriptives>&);
   vector<Descriptives> scale_inputs_mean_standard_deviation();

   void scale_input_mean_standard_deviation(const Descriptives&, const int&);
   Descriptives scale_input_mean_standard_deviation(const int&);

   void scale_input_standard_deviation(const Descriptives&, const int&);
   Descriptives scale_input_standard_deviation(const int&);

   void scale_inputs_minimum_maximum(const vector<Descriptives>&);
   vector<Descriptives> scale_inputs_minimum_maximum();

   void scale_input_minimum_maximum(const Descriptives&, const int&);
   Descriptives scale_input_minimum_maximum(const int&);

   vector<Descriptives> scale_inputs(const string&);
   void scale_inputs(const string&, const vector<Descriptives>&);
   void scale_inputs(const Tensor<string, 1>&, const vector<Descriptives>&);

   // Target variables scaling

   void scale_targets_minimum_maximum(const vector<Descriptives>&);
   vector<Descriptives> scale_targets_minimum_maximum();

   void scale_targets_mean_standard_deviation(const vector<Descriptives>&);
   vector<Descriptives> scale_targets_mean_standard_deviation();

   void scale_targets_logarithmic(const vector<Descriptives>&);
   vector<Descriptives> scale_targets_logarithmic();

   vector<Descriptives> scale_targets(const string&);
   void scale_targets(const string&, const vector<Descriptives>&);

   // Data unscaling

   void unscale_data_minimum_maximum(const vector<Descriptives>&);
   void unscale_data_mean_standard_deviation(const vector<Descriptives>&);

   // Input variables unscaling

   void unscale_inputs_minimum_maximum(const vector<Descriptives>&);
   void unscale_inputs_mean_standard_deviation(const vector<Descriptives>&);

   // Target variables unscaling

   void unscale_targets_minimum_maximum(const vector<Descriptives>&);
   void unscale_targets_mean_standard_deviation(const vector<Descriptives>&);

   // Classification methods

   Tensor<int, 1> calculate_target_distribution() const;

   Tensor<int, 1> balance_binary_targets_distribution(const double& = 100.0);
   Tensor<int, 1> balance_multiple_targets_distribution();


   Tensor<int, 1> balance_approximation_targets_distribution(const double& = 10.0);

   // Outlier detection

   Tensor<int, 1> calculate_Tukey_outliers(const int&, const double& = 1.5) const;

   vector<Tensor<int, 1>> calculate_Tukey_outliers(const double& = 1.5) const;

   void unuse_Tukey_outliers(const double& = 1.5);

   // Time series methods

   void transform_columns_time_series();

   Tensor<type, 2> calculate_autocorrelations(const int& = 10) const;
   Matrix<Tensor<type, 1>, Dynamic, Dynamic> calculate_cross_correlations(const int& = 10) const;

   Tensor<type, 2> calculate_lag_plot() const;
   Tensor<type, 2> calculate_lag_plot(const int&);

   // Data generation

   void generate_constant_data(const int&, const int&);
   void generate_random_data(const int&, const int&);
   void generate_sequential_data(const int&, const int&);
   void generate_paraboloid_data(const int&, const int&);
   void generate_Rosenbrock_data(const int&, const int&);
   void generate_inputs_selection_data(const int&, const int&);
   void generate_sum_data(const int&, const int&);

   void generate_data_binary_classification(const int&, const int&);
   void generate_data_multiple_classification(const int&, const int&, const int&);

   // Serialization methods

   string object_to_string() const;

   void print() const;
   void print_summary() const;

   tinyxml2::XMLDocument* to_XML() const;

   void from_XML(const tinyxml2::XMLDocument&);
   void write_XML(tinyxml2::XMLPrinter&) const;

   void save(const string&) const;
   void load(const string&);

   void print_columns_types() const;

   void print_data() const;
   void print_data_preview() const;

   void print_data_file_preview() const;

   void save_data() const;

   void save_data_binary(const string&) const;

   // Data load methods

   void read_csv();

   void load_data_binary();
   void load_time_series_data_binary();

   // Trasform methods

   void transform_time_series();
   void transform_association();

   void fill_time_series(const int&);

   void delete_unused_instances();

   void numeric_to_categorical(const int&);

   // Missing values

   void print_missing_values_information() const;

   void impute_missing_values_unuse();
   void impute_missing_values_mean();
   void impute_missing_values_median();

   void scrub_missing_values();

   Tensor<string, 1> unuse_columns_missing_values(const double&);

   void get_tensor_2_d(const Tensor<int, 1>&, const Tensor<int, 1>&, Tensor<type, 2>&);

   Tensor<int, 1> count_nan_columns() const;
   int count_rows_with_nan() const;

   // Eigen methods

   void intialize_sequential_eigen_tensor(Tensor<int, 1>&, const int&, const int&, const int&) const;

private:

   /// Data file name.

   string data_file_name;

   /// Separator character.

   Separator separator = Comma;

   /// Missing values label.

   string missing_values_label = "NA";

   /// Number of lags.

   int lags_number;

   /// Number of steps ahead.

   int steps_ahead;

   /// Data Matrix.
   /// The number of rows is the number of instances.
   /// The number of columns is the number of variables.

   Tensor<type, 2> data;

   /// Time series data matrix.
   /// The number of rows is the number of instances before time series transfomration.
   /// The number of columns is the number of variables before time series transformation.

   Tensor<type, 2> time_series_data;

   vector<Column> time_series_columns;

   /// Display messages to screen.

   bool display = true;

   /// Index where time variable is located for forecasting applications.

   int time_index;

   /// Missing values method object.

   MissingValuesMethod missing_values_method = Unuse;

   // Instances

   vector<InstanceUse> instances_uses;

   /// Number of batch instances. It is used to optimized the training strategy.

   int batch_instances_number = 32;

   // Variables

   // Reader

   void read_csv_1();

   void read_csv_2_simple();
   void read_csv_3_simple();

   void read_csv_2_complete();
   void read_csv_3_complete();

   void check_separators(const string&) const;

   /// Header which contains variables name.

   bool has_columns_names = false;

   Tensor<int, 1> input_variables_dimensions;

   Tensor<int, 1> target_variables_dimensions;

   vector<Column> columns;

   /// Header wihch contains the rows label.

   bool has_rows_labels = false;

   Tensor<string, 1> rows_labels;

   int gmt = 0;

   vector<Tensor<string, 1>> data_file_preview;

#ifdef __OPENNN_CUDA__
    #include "../../artelnics/opennn_cuda/opennn_cuda/data_set_cuda.h"
#endif

};

}

#endif

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2020 Artificial Intelligence Techniques, SL.
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
