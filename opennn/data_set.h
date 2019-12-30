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

// OpenNN includes

#include "metrics.h"
#include "matrix.h"
#include "statistics.h"
#include "transformations.h"
#include "vector.h"
#include "correlations.h"
#include "tinyxml2.h"

// Eigen includes

#include "../eigen/Eigen"

using namespace std;

namespace OpenNN
{

/// This class represents the concept of data set for data modelling problems, such as function regression, classification, time series prediction, images approximation and images classification.

///
/// It basically consists of a data Matrix separated by columns.
/// These columns can take different categories depending on the data hosted in them.
///
/// With OpenNN DataSet class you can edit the data to prepare your model, such as remove missing values,
/// calculate correlations between variables (inputs and targets), select particular variables or instances,
/// transform human date into timestamp,... .

class DataSet
{

public:

   // Constructors

   explicit DataSet();

   explicit DataSet(const Eigen::MatrixXd&);

   explicit DataSet(const Matrix<double>&);

   explicit DataSet(const size_t&, const size_t&);

   explicit DataSet(const size_t&, const size_t&, const size_t&);

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

       Column(const string&, const VariableUse&, const ColumnType& = Numeric, const Vector<string>& = Vector<string>(), const Vector<VariableUse>& = Vector<VariableUse>());

       /// Destructor.

       virtual ~Column();

       /// Column name.

       string name;

       /// Column use.

       VariableUse column_use;

       /// Column type.

       ColumnType type;

       /// Categories within the column.

       Vector<string> categories;

       /// Categories use.

       Vector<VariableUse> categories_uses;

       // Methods

       size_t get_categories_number() const;

       Vector<string> get_used_variables_names() const;

       void set_use(const VariableUse&);
       void set_use(const string&);

       void set_type(const ColumnType&);
       void set_type(const string&);

       void set_categories_uses(const Vector<VariableUse>&);
       void set_categories_uses(const Vector<string>&);

       bool is_used();
       bool is_unused();

       void from_XML(const tinyxml2::XMLDocument&);
       void write_XML(tinyxml2::XMLPrinter&) const;
   };

   // Instances get methods

   inline size_t get_instances_number() const {return instances_uses.size();}

   size_t get_training_instances_number() const;
   size_t get_selection_instances_number() const;
   size_t get_testing_instances_number() const;

   size_t get_used_instances_number() const;
   size_t get_unused_instances_number() const;

   Vector<size_t> get_training_instances_indices() const;
   Vector<size_t> get_selection_instances_indices() const;
   Vector<size_t> get_testing_instances_indices() const;

   Vector<size_t> get_used_instances_indices() const;
   Vector<size_t> get_unused_instances_indices() const;

   InstanceUse get_instance_use(const size_t&) const;
   const Vector<InstanceUse>& get_instances_uses() const;

   Vector<size_t> get_instances_uses_numbers() const;
   Vector<double> get_instances_uses_percentages() const;

   // Columns get methods

   Vector<Column> get_columns() const;
   Vector<Column> get_used_columns() const;

   size_t get_columns_number() const;

   size_t get_input_columns_number() const;
   size_t get_target_columns_number() const;
   size_t get_time_columns_number() const;
   size_t get_unused_columns_number() const;
   size_t get_used_columns_number() const;

   size_t get_column_index(const string&) const;

   Vector<size_t> get_input_columns_indices() const;
   Vector<size_t> get_target_columns_indices() const;
   Vector<size_t> get_unused_columns_indices() const;
   Vector<size_t> get_used_columns_indices() const;

   Vector<string> get_columns_names() const;

   Vector<string> get_input_columns_names() const;
   Vector<string> get_target_columns_names() const;
   Vector<string> get_used_columns_names() const;

   ColumnType get_column_type(const size_t& index) const {return columns[index].type;}

   VariableUse get_column_use(const size_t &) const;
   Vector<VariableUse> get_columns_uses() const;

   // Variables get methods

   size_t get_variables_number() const;

   size_t get_input_variables_number() const;
   size_t get_target_variables_number() const;
   size_t get_unused_variables_number() const;
   size_t get_used_variables_number() const;

   string get_variable_name(const size_t&) const;
   Vector<string> get_variables_names() const;

   Vector<string> get_input_variables_names() const;
   Vector<string> get_target_variables_names() const;

   size_t get_variable_index(const string&) const;

   Vector<size_t> get_variable_indices(const size_t&) const;
   Vector<size_t> get_unused_variables_indices() const;
   Vector<size_t> get_input_variables_indices() const;
   Vector<size_t> get_target_variables_indices() const;

   VariableUse get_variable_use(const size_t&) const;
   Vector<VariableUse> get_variables_uses() const;

   Vector<size_t> get_input_variables_dimensions() const;
   Vector<size_t> get_target_variables_dimensions() const;

   // Batches get methods

   inline size_t get_batch_instances_number() {return batch_instances_number;}

   Vector<Vector<size_t>> get_training_batches(const bool& = true) const;
   Vector<Vector<size_t>> get_selection_batches(const bool& = true) const;
   Vector<Vector<size_t>> get_testing_batches(const bool& = true) const;

   // Data get methods

   const Matrix<double>& get_data() const;
   const Eigen::MatrixXd get_data_eigen() const;

   const Matrix<double>& get_time_series_data() const;

   Matrix<double> get_training_data() const;
   Eigen::MatrixXd get_training_data_eigen() const;
   Matrix<double> get_selection_data() const;
   Eigen::MatrixXd get_selection_data_eigen() const;
   Matrix<double> get_testing_data() const;
   Eigen::MatrixXd get_testing_data_eigen() const;

   Matrix<double> get_input_data() const;
   Eigen::MatrixXd get_input_data_eigen() const;
   Matrix<double> get_target_data() const;
   Eigen::MatrixXd get_target_data_eigen() const;

   Tensor<double> get_input_data(const Vector<size_t>&) const;
   Tensor<double> get_target_data(const Vector<size_t>&) const;

   Matrix<float> get_input_data_float(const Vector<size_t>&) const;
   Matrix<float> get_target_data_float(const Vector<size_t>&) const;

   Tensor<double> get_training_input_data() const;
   Eigen::MatrixXd get_training_input_data_eigen() const;
   Tensor<double> get_training_target_data() const;
   Eigen::MatrixXd get_training_target_data_eigen() const;

   Tensor<double> get_selection_input_data() const;
   Eigen::MatrixXd get_selection_input_data_eigen() const;
   Tensor<double> get_selection_target_data() const;
   Eigen::MatrixXd get_selection_target_data_eigen() const;

   Tensor<double> get_testing_input_data() const;
   Eigen::MatrixXd get_testing_input_data_eigen() const;
   Tensor<double> get_testing_target_data() const;
   Eigen::MatrixXd get_testing_target_data_eigen() const;

   Vector<double> get_instance_data(const size_t&) const;
   Vector<double> get_instance_data(const size_t&, const Vector<size_t>&) const;
   Tensor<double> get_instance_input_data(const size_t&) const;
   Tensor<double> get_instance_target_data(const size_t&) const;

   Matrix<double> get_column_data(const size_t&) const;
   Matrix<double> get_column_data(const Vector<size_t>&) const;
   Matrix<double> get_column_data(const string&) const;

   Vector<double> get_variable_data(const size_t&) const;
   Vector<double> get_variable_data(const string&) const;

   Vector<double> get_variable_data(const size_t&, const Vector<size_t>&) const;
   Vector<double> get_variable_data(const string&, const Vector<size_t>&) const;

   // Members get methods

   MissingValuesMethod get_missing_values_method() const;

   const string& get_data_file_name() const;

   const bool& get_header_line() const;
   const bool& get_rows_label() const;

   const Separator& get_separator() const;
   char get_separator_char() const;
   string get_separator_string() const;

   const string& get_missing_values_label() const;

   const size_t& get_lags_number() const;
   const size_t& get_steps_ahead() const;
   const size_t& get_time_index() const;

   static Vector<string> get_default_columns_names(const size_t&);

   static ScalingUnscalingMethod get_scaling_unscaling_method(const string&);

   int get_gmt() const;

   const bool& get_display() const;

   // Set methods

   void set();
   void set(const Matrix<double>&);
   void set(const Eigen::MatrixXd&);
   void set(const size_t&, const size_t&);
   void set(const size_t&, const size_t&, const size_t&);
   void set(const DataSet&);
   void set(const tinyxml2::XMLDocument&);
   void set(const string&);

   void set_default();

   // Instances set methods

   void set_instances_number(const size_t&);

   void set_training();
   void set_selection();
   void set_testing();

   void set_training(const Vector<size_t>&);
   void set_selection(const Vector<size_t>&);
   void set_testing(const Vector<size_t>&);

   void set_instances_unused();
   void set_instances_unused(const Vector<size_t>&);

   void set_instance_use(const size_t&, const InstanceUse&);
   void set_instance_use(const size_t&, const string&);

   void set_instances_uses(const Vector<InstanceUse>&);
   void set_instances_uses(const Vector<string>&);

   void set_testing_to_selection_instances();
   void set_selection_to_testing_instances();

   void set_batch_instances_number(const size_t&);

   void set_k_fold_cross_validation_instances_uses(const size_t&, const size_t&);

   // Columns set methods

   void set_default_columns_uses();

   void set_default_columns_names();

   void set_columns_uses(const Vector<string>&);
   void set_columns_uses(const Vector<VariableUse>&);
   void set_columns_unused();
   void set_input_columns_unused();


   void set_column_use(const size_t&, const VariableUse&);
   void set_column_use(const string&, const VariableUse&);

   void set_columns_names(const Vector<string>&);

   void set_columns_number(const size_t&);

   // Variables set methods

   void set_variables_names(const Vector<string>&);
   void set_variable_name(const size_t&, const string&);

   void set_input();
   void set_target();
   void set_variables_unused();

   void set_input_variables_dimensions(const Vector<size_t>& );
   void set_target_variables_dimensions(const Vector<size_t>& );

   // Data set methods

   void set_data(const Matrix<double>&);

   void set_instance(const size_t&, const Vector<double>&);

   // Batch set methods

//   void set_shufffle_batches_instances(const bool&);

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

   void set_lags_number(const size_t&);
   void set_steps_ahead_number(const size_t&);
   void set_time_index(const size_t&);

   void set_gmt(int&);

   void set_display(const bool&);

   // Check methods

   bool is_binary_classification() const;
   bool is_multiple_classification() const;

   bool is_empty() const;

   bool is_instance_used(const size_t&) const;
   bool is_instance_unused(const size_t&) const;

   bool has_data() const;

   bool has_categorical_variables() const;
   bool has_time_variables() const;

   // Splitting methods

   void split_instances_sequential(const double& training_ratio = 0.6, const double& selection_ratio = 0.2, const double& testing_ratio = 0.2);

   void split_instances_random(const double& training_ratio = 0.6, const double& selection_ratio = 0.2, const double& testing_ratio = 0.2);

   // Unusing methods

   Vector<string> unuse_constant_columns();

   Vector<size_t> unuse_repeated_instances();

   Vector<size_t> unuse_non_significant_input_columns();

   Vector<size_t> unuse_uncorrelated_columns(const double& = 0.25);

   Vector<size_t> unuse_most_populated_target(const size_t&);

   // Initialization methods

   void initialize_data(const double&);

   void randomize_data_uniform(const double& minimum = -1.0, const double& maximum = 1.0);
   void randomize_data_normal(const double& mean = 0.0, const double& standard_deviation = 1.0);

   // Descriptives methods

   Vector<Descriptives> calculate_columns_descriptives() const;

   Matrix<double> calculate_columns_descriptives_matrix() const;

   Eigen::MatrixXd calculate_columns_descriptives_eigen() const;

   Vector<Descriptives> calculate_columns_descriptives_positive_instances() const;
   Vector<Descriptives> calculate_columns_descriptives_negative_instances() const;
   Vector<Descriptives> calculate_columns_descriptives_classes(const size_t&) const;

   Vector<Descriptives> calculate_columns_descriptives_training_instances() const;
   Vector<Descriptives> calculate_columns_descriptives_selection_instances() const;
   Vector<Descriptives> calculate_columns_descriptives_testing_instances() const;

   Vector<Descriptives> calculate_input_variables_descriptives() const;
   Vector<Descriptives> calculate_target_variables_descriptives() const;

   Vector<double> calculate_variables_means(const Vector<size_t>&) const;

   Descriptives calculate_inputs_descriptives(const size_t&) const;

   Vector<double> calculate_training_targets_mean() const;
   Vector<double> calculate_selection_targets_mean() const;
   Vector<double> calculate_testing_targets_mean() const;

   size_t calculate_training_negatives(const size_t&) const;
   size_t calculate_selection_negatives(const size_t&) const;
   size_t calculate_testing_negatives(const size_t&) const;

   // Histrogram methods

   Vector<Histogram> calculate_columns_histograms(const size_t& = 10) const;

   // Box and whiskers

   Vector<BoxPlot> calculate_columns_box_plots() const;

   // Inputs correlations

   Matrix<double> calculate_inputs_correlations() const;

   void print_inputs_correlations() const;

   void print_top_inputs_correlations(const size_t& = 10) const;

   // Inputs-targets correlations

   Matrix<CorrelationResults> calculate_input_target_columns_correlations() const;
   Matrix<double> calculate_input_target_columns_correlations_double() const;

   Eigen::MatrixXd calculate_input_target_columns_correlations_eigen() const;

   void print_input_target_columns_correlations() const;

   void print_top_input_target_columns_correlations(const size_t& = 10) const;

   // Principal components

   Matrix<double> calculate_covariance_matrix() const;

   Matrix<double> perform_principal_components_analysis(const double& = 0.0);

   Matrix<double> perform_principal_components_analysis(const Matrix<double>&, const Vector<double>&, const double& = 0.0);

   void transform_principal_components_data(const Matrix<double>&);

   void subtract_inputs_mean();

   // Filtering methods

   Vector<size_t> filter_column(const size_t&, const double&, const double&);
   Vector<size_t> filter_column(const string&, const double&, const double&);

   Vector<size_t> filter_data(const Vector<double>&, const Vector<double>&);

   // Data scaling

   Vector<string> calculate_default_scaling_methods() const;
   void scale_data_minimum_maximum(const Vector<Descriptives>&);
   void scale_data_mean_standard_deviation(const Vector<Descriptives>&);
   Vector<Descriptives> scale_data_minimum_maximum();
   Vector<Descriptives> scale_data_mean_standard_deviation();

   // Input variables scaling

   void scale_inputs_mean_standard_deviation(const Vector<Descriptives>&);
   Vector<Descriptives> scale_inputs_mean_standard_deviation();

   void scale_input_mean_standard_deviation(const Descriptives&, const size_t&);
   Descriptives scale_input_mean_standard_deviation(const size_t&);

   void scale_input_standard_deviation(const Descriptives&, const size_t&);
   Descriptives scale_input_standard_deviation(const size_t&);

   void scale_inputs_minimum_maximum(const Vector<Descriptives>&);
   Vector<Descriptives> scale_inputs_minimum_maximum();

   Eigen::MatrixXd scale_inputs_minimum_maximum_eigen();
   Eigen::MatrixXd scale_targets_minimum_maximum_eigen();

   void scale_input_minimum_maximum(const Descriptives&, const size_t&);
   Descriptives scale_input_minimum_maximum(const size_t&);

   Vector<Descriptives> scale_inputs(const string&);
   void scale_inputs(const string&, const Vector<Descriptives>&);
   void scale_inputs(const Vector<string>&, const Vector<Descriptives>&);

   // Target variables scaling

   void scale_targets_minimum_maximum(const Vector<Descriptives>&);
   Vector<Descriptives> scale_targets_minimum_maximum();

   void scale_targets_mean_standard_deviation(const Vector<Descriptives>&);
   Vector<Descriptives> scale_targets_mean_standard_deviation();

   void scale_targets_logarithmic(const Vector<Descriptives>&);
   Vector<Descriptives> scale_targets_logarithmic();

   Vector<Descriptives> scale_targets(const string&);
   void scale_targets(const string&, const Vector<Descriptives>&);

   // Data unscaling

   void unscale_data_minimum_maximum(const Vector<Descriptives>&);
   void unscale_data_mean_standard_deviation(const Vector<Descriptives>&);

   // Input variables unscaling

   void unscale_inputs_minimum_maximum(const Vector<Descriptives>&);
   void unscale_inputs_mean_standard_deviation(const Vector<Descriptives>&);

   // Target variables unscaling

   void unscale_targets_minimum_maximum(const Vector<Descriptives>&);
   void unscale_targets_mean_standard_deviation(const Vector<Descriptives>&);

   // Classification methods

   Vector<size_t> calculate_target_distribution() const;

   Vector<size_t> balance_binary_targets_distribution(const double& = 100.0);
   Vector<size_t> balance_multiple_targets_distribution();


   Vector<size_t> balance_approximation_targets_distribution(const double& = 10.0);

   // Outlier detection

   Vector<size_t> calculate_Tukey_outliers(const size_t&, const double& = 1.5) const;

   Vector<Vector<size_t>> calculate_Tukey_outliers(const double& = 1.5) const;

   void unuse_Tukey_outliers(const double& = 1.5);

   // Time series methods

   void transform_columns_time_series();

   Matrix<double> calculate_autocorrelations(const size_t& = 10) const;
   Matrix<Vector<double>> calculate_cross_correlations(const size_t& = 10) const;

   Matrix<double> calculate_lag_plot() const;
   Matrix<double> calculate_lag_plot(const size_t&);

   // Data generation

   void generate_constant_data(const size_t&, const size_t&);
   void generate_random_data(const size_t&, const size_t&);
   void generate_sequential_data(const size_t&, const size_t&);
   void generate_paraboloid_data(const size_t&, const size_t&);
   void generate_Rosenbrock_data(const size_t&, const size_t&);
   void generate_inputs_selection_data(const size_t&, const size_t&);
   void generate_sum_data(const size_t&, const size_t&);

   void generate_data_binary_classification(const size_t&, const size_t&);
   void generate_data_multiple_classification(const size_t&, const size_t&, const size_t&);

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

   // Data load methods

   void read_csv();

   void load_data_binary();
   void load_time_series_data_binary();

   // Trasform methods

   void transform_time_series();
   void transform_association();

   void fill_time_series(const size_t&);

   void delete_unused_instances();

   void numeric_to_categorical(const size_t&);

   // Missing values

   void print_missing_values_information() const;

   void impute_missing_values_unuse();
   void impute_missing_values_mean();
   void impute_missing_values_median();

   void scrub_missing_values();

   Vector<string> unuse_columns_missing_values(const double&);

private:

   /// Data file name.

   string data_file_name;

   /// Separator character.

   Separator separator = Comma;

   /// Missing values label.

   string missing_values_label = "NA";

   /// Number of lags.

   size_t lags_number;

   /// Number of steps ahead.

   size_t steps_ahead;

   /// Data Matrix.
   /// The number of rows is the number of instances.
   /// The number of columns is the number of variables.

   Matrix<double> data;

   /// Time series data matrix.
   /// The number of rows is the number of instances before time series transfomration.
   /// The number of columns is the number of variables before time series transformation.

   Matrix<double> time_series_data;

   Vector<Column> time_series_columns;

   /// Display messages to screen.

   bool display = true;

   /// Index where time variable is located for forecasting applications.

   size_t time_index;

   /// Missing values method object.

   MissingValuesMethod missing_values_method = Unuse;

   // Instances

   Vector<InstanceUse> instances_uses;

   /// Number of batch instances. It is used to optimized the training strategy.

   size_t batch_instances_number = 1000;

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

   /// Dimensions of the tensor input.

   Vector<size_t> inputs_dimensions;

   /// Dimensions of the tensor target.

   Vector<size_t> targets_dimensions;

   /// Vector which contains the columns of the dataset.

   Vector<Column> columns;

   /// Header wihch contains the rows label.

   bool has_rows_labels = false;

   /// Vector which contains the labels of the rows.

   Vector<string> rows_labels;

   /// Greenwich Mean Time, to transform human date into timpestamp.

   int gmt = 0;

   Vector<Vector<string>> data_file_preview;

//   bool shuffle_batches_instances = false;
};

}

#endif

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2019 Artificial Intelligence Techniques, SL.
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
