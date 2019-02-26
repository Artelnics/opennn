/****************************************************************************************************************/
/*                                                                                                              */
/*   OpenNN: Open Neural Networks Library                                                                       */
/*   www.opennn.net                                                                                             */
/*                                                                                                              */
/*   D A T A   S E T   C L A S S   H E A D E R                                                                  */
/*                                                                                                              */
/*   Artificial Intelligence Techniques SL                                                                      */
/*   artelnics@artelnics.com                                                                                    */
/*                                                                                                              */
/****************************************************************************************************************/

#ifndef __DATASET_H__
#define __DATASET_H__

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

#ifdef __OPENNN_MPI__
#include <mpi.h>
#endif
// OpenNN includes

#include "vector.h"
#include "matrix.h"

#include "missing_values.h"
#include "variables.h"
#include "instances.h"

// TinyXml includes

#include "tinyxml2.h"

using namespace std;

namespace OpenNN
{

// Forward declaration

/// This class represents the concept of data set for data modelling problems,
/// such as function regression, classification and time series prediction.
/// It basically consists of a data matrix plus a variables and an instances objects.

class DataSet
{

public:

   // DEFAULT CONSTRUCTOR

   explicit DataSet();

   // DATA CONSTRUCTOR

   explicit DataSet(const Eigen::MatrixXd&);

   explicit DataSet(const Matrix<double>&);

   // INSTANCES AND VARIABLES CONSTRUCTOR

   explicit DataSet(const size_t&, const size_t&);

   // INSTANCES, INPUTS AND TARGETS CONSTRUCTOR

   explicit DataSet(const size_t&, const size_t&, const size_t&);

   // XML CONSTRUCTOR

   explicit DataSet(const tinyxml2::XMLDocument&);

   // DATA FILE CONSTRUCTOR

   explicit DataSet(const string&);

   // DATA FILE AND SEPARATOR CONSTRUCTOR

   explicit DataSet(const string&, const string&);

   // COPY CONSTRUCTOR

   DataSet(const DataSet&);

   // DESTRUCTOR

   virtual ~DataSet();

   // ASSIGNMENT OPERATOR

   DataSet& operator = (const DataSet&);

   // EQUAL TO OPERATOR

   bool operator == (const DataSet&) const;

   // ENUMERATIONS

   /// Enumeration of available separators for the data file.

   enum Separator{Space, Tab, Comma, Semicolon};

   /// Enumeration of available methods for scaling and unscaling the data.

   enum ScalingUnscalingMethod{NoScaling, NoUnscaling, MinimumMaximum, MeanStandardDeviation, StandardDeviation, Logarithmic};

   /// Enumeration of the units used for angular variables.

   enum AngularUnits{Radians, Degrees};

   /// Enumeration of the file types

   enum FileType{TXT, DAT, DATA, CSV, ODS, XLSX, ARFF, JSON};

   /// Enumeration of the learning tasks

   enum ProjectType{Approximation, Classification, Forecasting, Association};

   // METHODS

   // Get methods

   FileType get_file_type() const;
   string write_file_type() const;

   string write_first_cell() const;
   string write_last_cell() const;
   size_t write_sheet_number() const;

   ProjectType get_learning_task() const;
   string write_learning_task() const;

   const string& get_data_file_name() const;

   const bool& get_header_line() const;
   const bool& get_rows_label() const;

   const Separator& get_separator() const;
   char get_separator_char() const;
   string write_separator() const;

   const string& get_missing_values_label() const;

   const double& get_grouping_factor() const;

   const size_t& get_lags_number() const;
   const size_t& get_steps_ahead() const;
   const size_t& get_time_index() const;

   const bool& get_autoassociation() const;

   const Vector<size_t>& get_angular_variables() const;
   const AngularUnits& get_angular_units() const;

   static ScalingUnscalingMethod get_scaling_unscaling_method(const string&);

   const MissingValues& get_missing_values() const;
   MissingValues* get_missing_values_pointer();

   const Variables& get_variables() const;
   Variables* get_variables_pointer();

   const Instances& get_instances() const;
   Instances* get_instances_pointer();

   const bool& get_display() const;

   bool is_binary_classification() const;
   bool is_multiple_classification() const;

   bool is_binary_variable(const size_t&) const;
   bool is_binary_variable(const string&) const;

   // Data methods

   bool empty() const;

   const Matrix<double>& get_data() const;
   const Eigen::MatrixXd get_data_eigen() const;

   const Matrix<double>& get_time_series_data() const;

   Matrix<double> get_instances_submatrix_data(const Vector<size_t>&) const;

   Matrix<double> get_training_data() const;
   Eigen::MatrixXd get_training_data_eigen() const;
   Matrix<double> get_selection_data() const;
   Eigen::MatrixXd get_selection_data_eigen() const;
   Matrix<double> get_testing_data() const;
   Eigen::MatrixXd get_testing_data_eigen() const;

   Matrix<double> get_inputs() const;
   Eigen::MatrixXd get_inputs_eigen() const;
   Matrix<double> get_targets() const;
   Eigen::MatrixXd get_targets_eigen() const;

   Matrix<double> get_inputs(const Vector<size_t>&) const;
   Matrix<double> get_targets(const Vector<size_t>&) const;

   Matrix<double> get_used_data() const;
   Matrix<double> get_used_inputs() const;
   Matrix<double> get_used_targets() const;

   Matrix<double> get_training_inputs() const;
   Eigen::MatrixXd get_training_inputs_eigen() const;
   Matrix<double> get_training_targets() const;
   Eigen::MatrixXd get_training_targets_eigen() const;

   Matrix<double> get_selection_inputs() const;
   Eigen::MatrixXd get_selection_inputs_eigen() const;
   Matrix<double> get_selection_targets() const;
   Eigen::MatrixXd get_selection_targets_eigen() const;

   Matrix<double> get_testing_inputs() const;
   Eigen::MatrixXd get_testing_inputs_eigen() const;
   Matrix<double> get_testing_targets() const;
   Eigen::MatrixXd get_testing_targets_eigen() const;
   Vector<double> get_testing_time() const;

   DataSet get_training_data_set() const;
   DataSet get_testing_data_set() const;
   DataSet get_selection_data_set() const;


   // Instance methods

   Vector<double> get_instance(const size_t&) const;
   Vector<double> get_instance(const size_t&, const Vector<size_t>&) const;

   // Variable methods

   Vector<double> get_variable(const size_t&) const;
   Vector<double> get_variable(const string&) const;

   Vector<double> get_variable(const size_t&, const Vector<size_t>&) const;
   Vector<double> get_variable(const string&, const Vector<size_t>&) const;

   Matrix<double> get_variables(const Vector<size_t>&, const Vector<size_t>&) const;

   // Set methods

   void set();
   void set(const Matrix<double>&);
   void set(const Eigen::MatrixXd&);
   void set(const size_t&, const size_t&);
   void set(const size_t&, const size_t&, const size_t&);
   void set(const DataSet&);
   void set(const tinyxml2::XMLDocument&);
   void set(const string&);

   // Data methods

   void set_data(const Matrix<double>&);

   void set_instances_number(const size_t&);
   void set_variables_number(const size_t&);

   void set_data_file_name(const string&);

   void set_file_type(const FileType&);
   void set_file_type(const string&);

   void set_header_line(const bool&);
   void set_rows_label(const bool&);

   void set_separator(const Separator&);
   void set_separator(const string&);

   void set_missing_values_label(const string&);

   void set_grouping_factor(const double&);

   void set_lags_number(const size_t&);
   void set_steps_ahead_number(const size_t&);
   void set_time_index(const size_t&);

   void set_autoassociation(const bool&);

   void set_learning_task(const ProjectType&);
   void set_learning_task(const string&);

   void set_angular_variables(const Vector<size_t>&);
   void set_angular_units(AngularUnits&);

   // Utilities

   void set_display(const bool&);

   void set_default();

   void set_MPI(const DataSet*);

   // Instance methods

   void set_instance(const size_t&, const Vector<double>&);

   // Data resizing methods

   void add_instance(const Vector<double>&);
   void remove_instance(const size_t&);

   void append_variable(const Vector<double>&, const string& = "");

   void remove_variable(const size_t&);
   void remove_variable(const string&);

   Vector<string> unuse_constant_variables();
   Vector<string> unuse_binary_inputs(const double&);

   Vector<size_t> unuse_repeated_instances();

   Vector<size_t> unuse_non_significant_inputs();

   Vector<string> unuse_variables_missing_values(const double&);

   Vector<size_t> unuse_uncorrelated_variables(const double& = 0.25, const Vector<size_t>& = Vector<size_t>());

   // Initialization methods

   void initialize_data(const double&);

   void randomize_data_uniform(const double& minimum = -1.0, const double& maximum = 1.0);
   void randomize_data_normal(const double& mean = 0.0, const double& standard_deviation = 1.0);

   // Statistics methods

   Vector< Statistics<double> > calculate_data_statistics() const;

   Vector< Vector<double> > calculate_data_shape_parameters() const;

   Matrix<double> calculate_data_statistics_matrix() const;

   Eigen::MatrixXd calculate_data_statistics_eigen_matrix() const;


   Matrix<double> calculate_positives_data_statistics_matrix() const;
   Matrix<double> calculate_negatives_data_statistics_matrix() const;

   Matrix<double> calculate_data_shape_parameters_matrix() const;

   Vector< Statistics<double> > calculate_training_instances_statistics() const;
   Vector< Statistics<double> > calculate_selection_instances_statistics() const;
   Vector< Statistics<double> > calculate_testing_instances_statistics() const;

   Vector< Vector<double> > calculate_training_instances_shape_parameters() const;
   Vector< Vector<double> > calculate_selection_instances_shape_parameters() const;
   Vector< Vector<double> > calculate_testing_instances_shape_parameters() const;

   Vector< Statistics<double> > calculate_inputs_statistics() const;
   Vector< Statistics<double> > calculate_targets_statistics() const;

   Vector< Vector<double> > calculate_inputs_minimums_maximums() const;
   Vector< Vector<double> > calculate_targets_minimums_maximums() const;

   Vector<double> calculate_variables_means(const Vector<size_t>&) const;

   Statistics<double> calculate_input_statistics(const size_t&) const;

   Vector<double> calculate_training_targets_mean() const;
   Vector<double> calculate_selection_targets_mean() const;
   Vector<double> calculate_testing_targets_mean() const;

   Vector< Vector< Vector<double> > > calculate_means_columns() const;

   // Correlation methods

   Matrix<double> calculate_input_target_correlations() const;
   Eigen::MatrixXd calculate_input_target_correlations_eigen() const;

   Vector<double> calculate_total_input_correlations() const;

   // Information methods

   void print_missing_values_information() const;

   void print_input_target_correlations() const;

   void print_top_input_target_correlations(const size_t& = 10) const;

   Matrix<double> calculate_variables_correlations() const;
   void print_variables_correlations() const;
   void print_top_variables_correlations(const size_t& = 10) const;

   // Nominal variables

   Matrix<double> calculate_multiple_linear_correlations(const Vector<size_t>&) const;
   Vector<double> calculate_multiple_total_linear_correlations(const Vector<size_t>&) const;

//   Matrix<double> calculate_multiple_logistic_correlations(const Vector<size_t>&) const;

   size_t calculate_input_variables_number(const Vector<size_t>&) const;
   Vector< Vector<size_t> > get_inputs_indices(const size_t&, const Vector<size_t>&) const;

   // Principal components mehtod

   Matrix<double> calculate_covariance_matrix() const;

   Matrix<double> perform_principal_components_analysis(const double& = 0.0);
   Matrix<double> perform_principal_components_analysis(const Matrix<double>&, const Vector<double>&, const double& = 0.0);
   void transform_principal_components_data(const Matrix<double>&);

   void remove_inputs_mean();

   // Histrogram methods

   Vector< Histogram<double> > calculate_data_histograms(const size_t& = 10) const;

   Vector< Histogram<double> > calculate_targets_histograms(const size_t& = 10) const;

   // Box and whiskers

   Vector< Vector<double> > calculate_box_plots() const;

   size_t calculate_training_negatives(const size_t&) const;
   size_t calculate_selection_negatives(const size_t&) const;
   size_t calculate_testing_negatives(const size_t&) const;

   // Filtering methods

   Vector<size_t> filter_data(const Vector<double>&, const Vector<double>&);

   Vector<size_t> filter_variable(const size_t&, const double&, const double&);
   Vector<size_t> filter_variable(const string&, const double&, const double&);

   // Data scaling

   Vector<string> calculate_default_scaling_methods() const;

   void scale_data_minimum_maximum(const Vector< Statistics<double> >&);
   void scale_data_mean_standard_deviation(const Vector< Statistics<double> >&);

   Vector< Statistics<double> > scale_data_minimum_maximum();
   Vector< Statistics<double> > scale_data_mean_standard_deviation();

   Vector< Statistics<double> > scale_data_range(const double&, const double&);

   // Input variables scaling

   void scale_inputs_mean_standard_deviation(const Vector< Statistics<double> >&);
   Vector< Statistics<double> > scale_inputs_mean_standard_deviation();

   void scale_input_mean_standard_deviation(const Statistics<double>&, const size_t&);
   Statistics<double> scale_input_mean_standard_deviation(const size_t&);

   void scale_input_standard_deviation(const Statistics<double>&, const size_t&);
   Statistics<double> scale_input_standard_deviation(const size_t&);

   void scale_inputs_minimum_maximum(const Vector< Statistics<double> >&);
   Vector< Statistics<double> > scale_inputs_minimum_maximum();

   Eigen::MatrixXd scale_inputs_minimum_maximum_eigen();
   Eigen::MatrixXd scale_targets_minimum_maximum_eigen();

   void scale_input_minimum_maximum(const Statistics<double>&, const size_t&);
   Statistics<double> scale_input_minimum_maximum(const size_t&);

   Vector< Statistics<double> > scale_inputs(const string&);
   void scale_inputs(const string&, const Vector< Statistics<double> >&);
   void scale_inputs(const Vector<string>&, const Vector< Statistics<double> > &);

   // Target variables scaling

   void scale_targets_minimum_maximum(const Vector< Statistics<double> >&);
   Vector< Statistics<double> > scale_targets_minimum_maximum();

   void scale_targets_mean_standard_deviation(const Vector< Statistics<double> >&);
   Vector< Statistics<double> > scale_targets_mean_standard_deviation();

   void scale_targets_logarithmic(const Vector< Statistics<double> >&);
   Vector< Statistics<double> > scale_targets_logarithmic();

   Vector< Statistics<double> > scale_targets(const string&);
   void scale_targets(const string&, const Vector< Statistics<double> >&);

   // Data unscaling

   void unscale_data_minimum_maximum(const Vector< Statistics<double> >&);
   void unscale_data_mean_standard_deviation(const Vector< Statistics<double> >&);

   // Input variables unscaling

   void unscale_inputs_minimum_maximum(const Vector< Statistics<double> >&);
   void unscale_inputs_mean_standard_deviation(const Vector< Statistics<double> >&);

   // Target variables unscaling

   void unscale_targets_minimum_maximum(const Vector< Statistics<double> >&);
   void unscale_targets_mean_standard_deviation(const Vector< Statistics<double> >&);

   // Classification methods

   Vector<size_t> calculate_target_distribution() const;

   Vector<double> calculate_distances() const;

   Vector<size_t> balance_binary_targets_distribution(const double& = 100.0);
   Vector<size_t> balance_multiple_targets_distribution();

   Vector<size_t> unuse_most_populated_target(const size_t&);

   Vector<size_t> balance_approximation_targets_distribution(const double& = 10.0);

   Vector<size_t> get_binary_inputs_indices() const;
   Vector<size_t> get_real_inputs_indices() const;

   void sum_binary_inputs();

   // Outlier detection

   Matrix<double> calculate_instances_distances(const size_t&) const;
   Matrix<size_t> calculate_k_nearest_neighbors(const Matrix<double>&, const size_t&) const;
   Vector<double> calculate_k_distances(const Matrix<double>&, const size_t&) const;
   Matrix<double> calculate_reachability_distances(const Matrix<double>&, const Vector<double>&) const;
   Vector<double> calculate_reachability_density(const Matrix<double>&, const size_t&) const;
   Vector<double> calculate_local_outlier_factor(const size_t& = 5) const;

   Vector<size_t> clean_local_outlier_factor(const size_t& = 5);

   Vector<size_t> calculate_Tukey_outliers(const size_t&, const double& = 1.5) const;

   Vector< Vector<size_t> > calculate_Tukey_outliers(const double& = 1.5) const;

   void unuse_Tukey_outliers(const double& = 1.5);

   // Time series methods

   Matrix<double> calculate_autocorrelations(const size_t& = 10) const;
   Matrix< Vector<double> > calculate_cross_correlations(const size_t& = 10) const;

   Matrix<double> calculate_lag_plot() const;
   Matrix<double> calculate_lag_plot(const size_t&);

   // Trending methods

   Vector< LinearRegressionParameters<double> > perform_trends_transformation();

   Vector< LinearRegressionParameters<double> > perform_inputs_trends_transformation();

   Vector< LinearRegressionParameters<double> > perform_outputs_trends_transformation();

   // Data generation

   void generate_constant_data(const size_t&, const size_t&);
   void generate_random_data(const size_t&, const size_t&);
   void generate_paraboloid_data(const size_t&, const size_t&);
   void generate_Rosenbrock_data(const size_t&, const size_t&);

   void generate_data_binary_classification(const size_t&, const size_t&);
   void generate_data_multiple_classification(const size_t&, const size_t&);

   // Serialization methods

   Vector<double*> host_to_device(const Vector<size_t>&) const;

   string object_to_string() const;

   void print() const;
   void print_summary() const;

   tinyxml2::XMLDocument* to_XML() const;
   void from_XML(const tinyxml2::XMLDocument&);

   void write_XML(tinyxml2::XMLPrinter&) const;

   void save(const string&) const;
   void load(const string&);

   void print_data() const;
   void print_data_preview() const;

   void save_data() const;

   bool has_data() const;

   // Data load methods

   void load_data();
   void load_data_binary();
   void load_time_series_data_binary();

   Vector<string> get_time_series_names(const Vector<string>&) const;

   Vector<string> get_association_names(const Vector<string>&) const;

   void convert_time_series();
   void convert_association();

   void convert_angular_variable_degrees(const size_t&);
   void convert_angular_variable_radians(const size_t&);

   void convert_angular_variables_degrees(const Vector<size_t>&);
   void convert_angular_variables_radians(const Vector<size_t>&);

   void convert_angular_variables();

   // Missing values

   void impute_missing_values_unuse();
   void impute_missing_values_mean();
   void impute_missing_values_time_series_mean();
   void impute_missing_values_time_series_regression();
   void impute_missing_values_median();

   void scrub_missing_values();

   // String utilities

   size_t count_tokens(string&) const;

   Vector<string> get_tokens(const string&) const;

   bool is_numeric(const string&) const;

   void trim(string&) const;

   string get_trimmed(const string&) const;

   string prepend(const string&, const string&) const;

   // Vector string utilities

   bool is_numeric(const Vector<string>&) const;
   bool is_not_numeric(const Vector<string>&) const;
   bool is_mixed(const Vector<string>&) const;

   // Utility methods

   void set_variable_use(const size_t& index, const string& use);


private:

   // MEMBERS

   /// File type.

   FileType file_type;

   /// First cell.

   string first_cell;

   /// Last cell.

   string last_cell;

   /// Sheet number.

   size_t sheet_number;

   /// Data file name.

   string data_file_name;

   /// Header which contains variables name.

   bool header_line;

   /// Header wihch contains the rows label.

   bool rows_label;

   /// Separator character.

   Separator separator;

   /// Missing values label.

   string missing_values_label;

   /// Grouping factor

   double grouping_factor;

   /// Number of lags.

   size_t lags_number;

   /// Number of steps ahead

   size_t steps_ahead;

   /// Association flag.

    bool association;

   /// Project type

    ProjectType learning_task;

   /// Indices of angular variables.

    Vector<size_t> angular_variables;

   /// Units of angular variables.

    AngularUnits angular_units;

   /// Data Matrix.
   /// The number of rows is the number of instances.
   /// The number of columns is the number of variables.

   Matrix<double> data;

   /// Time series data matrix.
   /// The number of rows is the number of instances before time series changes.
   /// The number of columns is the number of variables before time series changes.

   Matrix<double> time_series_data;

   /// Variables object(inputs and target variables).

   Variables variables;

   /// Instances  object(training, selection and testing instances).

   Instances instances;

   /// Missing values object.

   MissingValues missing_values;

   /// Display messages to screen.

   bool display;

   size_t time_index;

   // METHODS

   size_t get_column_index(const Vector< Vector<string> >&, const size_t) const;

   void check_separator(const string&) const;

   size_t count_data_file_columns_number() const;
   void check_header_line();
   Vector<string> read_header_line() const;

   void read_instance(const string&, const Vector< Vector<string> >&, const size_t&);

   Vector< Vector<string> > set_from_data_file();
   void read_from_data_file(const Vector< Vector<string> >&);

};

}

#endif

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
