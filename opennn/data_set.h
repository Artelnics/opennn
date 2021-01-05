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
#include <random>
#include <regex>
#include <map>
#include <stdlib.h>
#include <stdio.h>
#include <limits.h>

// OpenNN includes

#include "config.h"
#include "statistics.h"
#include "correlations.h"
#include "opennn_strings.h"

using namespace std;
using namespace Eigen;

namespace OpenNN
{

/// This class represents the concept of data set for data modelling problems, such as approximation, classification or forecasting.

///
/// It basically consists of a data Matrix separated by columns.
/// These columns can take different categories depending on the data hosted in them.
///
/// With OpenNN DataSet class you can edit the data to prepare your model, such as eliminating missing values,
/// calculating correlations between variables (inputs and targets), not using certain variables or samples, etc \dots.

class DataSet
{

public:

   // Constructors

   explicit DataSet();

   explicit DataSet(const Tensor<type, 2>&);

   explicit DataSet(const Index&, const Index&);

   explicit DataSet(const Index&, const Index&, const Index&);

   explicit DataSet(const string&, const char&, const bool&);

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

   /// This enumeration represents the possible uses of an sample
   /// (training, selection, testing or unused).

   enum SampleUse{Training, Selection, Testing, UnusedSample};

   /// This enumeration represents the possible uses of an variable
   /// (input, target, time or unused).

   enum VariableUse{Id, Input, Target, Time, UnusedVariable};

   /// This enumeration represents the data type of a column
   /// (numeric, binary, categorical or time).

   enum ColumnType{Numeric, Binary, Categorical, DateTime, Constant};

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

       Index get_categories_number() const;
       Index get_used_categories_number() const;

       Tensor<string, 1> get_used_variables_names() const;

       void set_use(const VariableUse&);
       void set_use(const string&);

       void set_type(const string&);

       void add_category(const string&);

       void set_categories_uses(const Tensor<string, 1>&);
       void set_categories_uses(const VariableUse&);

       bool is_used();
       bool is_unused();

       void from_XML(const tinyxml2::XMLDocument&);
       void write_XML(tinyxml2::XMLPrinter&) const;
   };


   struct Batch
   {
       /// Default constructor.

       Batch() {}

       Batch(const Index& new_samples_number, DataSet* new_data_set_pointer);

       /// Destructor.

       virtual ~Batch() {}

       Index get_samples_number() const;

       void print();

       void fill(const Tensor<Index, 1>& samples, const Tensor<Index, 1>& inputs, const Tensor<Index, 1>& targets);

//       void fill_submatrix(const Tensor<type, 2>& matrix,
//                 const Tensor<Index, 1>& rows_indices,
//                 const Tensor<Index, 1>& columns_indices, Tensor<type, 2>& submatrix);


       Index samples_number = 0;

       DataSet* data_set_pointer = nullptr;

       Tensor<type, 2> inputs_2d;
       Tensor<type, 4> inputs_4d;

       Tensor<type, 2> targets_2d;
   };


   // Samples get methods

   inline Index get_samples_number() const {return samples_uses.size();}

   Index get_training_samples_number() const;
   Index get_selection_samples_number() const;
   Index get_testing_samples_number() const;

   Index get_used_samples_number() const;
   Index get_unused_samples_number() const;

   Tensor<Index, 1> get_training_samples_indices() const;
   Tensor<Index, 1> get_selection_samples_indices() const;
   Tensor<Index, 1> get_testing_samples_indices() const;

   Tensor<Index, 1> get_used_samples_indices() const;
   Tensor<Index, 1> get_unused_samples_indices() const;

   SampleUse get_sample_use(const Index&) const;
   const Tensor<SampleUse, 1>& get_samples_uses() const;

   Tensor<Index, 1> get_samples_uses_numbers() const;
   Tensor<type, 1> get_samples_uses_percentages() const;

   string get_sample_string(const Index&, const string& = ",") const;

   // Columns get methods

   Tensor<Column, 1> get_columns() const;
   Tensor<Column, 1> get_time_series_columns() const;
   Tensor<Column, 1> get_input_columns() const;
   Tensor<Column, 1> get_target_columns() const;
   Tensor<Column, 1> get_used_columns() const;

   Index get_columns_number() const;

   Index get_input_columns_number() const;
   Index get_target_columns_number() const;
   Index get_time_columns_number() const;
   Index get_unused_columns_number() const;
   Index get_used_columns_number() const;

   Index get_column_index(const string&) const;
   Index get_column_index(const Index&) const;

   Tensor<Index, 1> get_input_columns_indices() const;
   Tensor<Index, 1> get_target_columns_indices() const;
   Tensor<Index, 1> get_unused_columns_indices() const;
   Tensor<Index, 1> get_used_columns_indices() const;

   Tensor<string, 1> get_columns_names() const;

   Tensor<string, 1> get_input_columns_names() const;
   Tensor<string, 1> get_target_columns_names() const;
   Tensor<string, 1> get_used_columns_names() const;

   ColumnType get_column_type(const Index& index) const {return columns[index].type;}

   VariableUse get_column_use(const Index &) const;
   Tensor<VariableUse, 1> get_columns_uses() const;

   // Variables get methods

   Index get_variables_number() const;

   Index get_input_variables_number() const;
   Index get_target_variables_number() const;
   Index get_unused_variables_number() const;
   Index get_used_variables_number() const;

   string get_variable_name(const Index&) const;
   Tensor<string, 1> get_variables_names() const;

   Tensor<string, 1> get_input_variables_names() const;
   Tensor<string, 1> get_target_variables_names() const;

   Index get_variable_index(const string&name) const;

   Tensor<Index, 1> get_variable_indices(const Index&) const;
   Tensor<Index, 1> get_unused_variables_indices() const;
   Tensor<Index, 1> get_used_variables_indices() const;
   Tensor<Index, 1> get_input_variables_indices() const;
   Tensor<Index, 1> get_target_variables_indices() const;

   VariableUse get_variable_use(const Index&) const;
   Tensor<VariableUse, 1> get_variables_uses() const;

   const Tensor<Index, 1>& get_input_variables_dimensions() const;

   // Batches get methods

   Tensor<Index, 2> get_batches(const Tensor<Index,1>&, const Index&, const bool&, const Index& buffer_size= 100) const;

   // Data get methods

   const Tensor<type, 2>& get_data() const;
   Tensor<type, 2>* get_data_pointer();

   const Tensor<type, 2>& get_time_series_data() const;

   Tensor<type, 2> get_training_data() const;
   Tensor<type, 2> get_selection_data() const;
   Tensor<type, 2> get_testing_data() const;
   Tensor<string, 1> get_time_series_columns_names() const;
   Index get_time_series_columns_number() const;

   Tensor<type, 2> get_input_data() const;
   Tensor<type, 2> get_target_data() const;

   Tensor<type, 2> get_input_data(const Tensor<Index, 1>&) const;
   Tensor<type, 2> get_target_data(const Tensor<Index, 1>&) const;

   Tensor<type, 2> get_training_input_data() const;
   Tensor<type, 2> get_training_target_data() const;

   Tensor<type, 2> get_selection_input_data() const;
   Tensor<type, 2> get_selection_target_data() const;

   Tensor<type, 2> get_testing_input_data() const;
   Tensor<type, 2> get_testing_target_data() const;

   Tensor<type, 1> get_sample_data(const Index&) const;
   Tensor<type, 1> get_sample_data(const Index&, const Tensor<Index, 1>&) const;
   Tensor<type, 2> get_sample_input_data(const Index&) const;
   Tensor<type, 2> get_sample_target_data(const Index&) const;

   Tensor<type, 2> get_column_data(const Index&) const;
   Tensor<type, 2> get_column_data(const Index&, Tensor<Index, 1>&) const;
   Tensor<type, 2> get_column_data(const Tensor<Index, 1>&) const;
   Tensor<type, 2> get_column_data(const string&) const;

   Tensor<type, 1> get_variable_data(const Index&) const;
   Tensor<type, 1> get_variable_data(const string&) const;

   Tensor<type, 1> get_variable_data(const Index&, const Tensor<Index, 1>&) const;
   Tensor<type, 1> get_variable_data(const string&, const Tensor<Index, 1>&) const;

   Tensor<Tensor<string, 1>, 1> get_data_file_preview() const;

   Tensor<type, 2> get_subtensor_data(const Tensor<Index, 1>&, const Tensor<Index, 1>&) const;

   // Members get methods

   MissingValuesMethod get_missing_values_method() const;

   const string& get_data_file_name() const;

   const bool& get_header_line() const;
   const bool& get_rows_label() const;

   Tensor<string, 1> get_rows_label_tensor() const;
   Tensor<string, 1> get_selection_rows_label_tensor();
   Tensor<string, 1> get_testing_rows_label_tensor();

   const Separator& get_separator() const;
   char get_separator_char() const;
   string get_separator_string() const;

   const string& get_missing_values_label() const;

   const Index& get_lags_number() const;
   const Index& get_steps_ahead() const;
   const Index& get_time_index() const;

   static Tensor<string, 1> get_default_columns_names(const Index&);

   static ScalingUnscalingMethod get_scaling_unscaling_method(const string&);

   Index get_gmt() const;

   const bool& get_display() const;

   // Set methods

   void set();
   void set(const Tensor<type, 2>&);
   void set(const Index&, const Index&);
   void set(const Index&, const Index&, const Index&);
   void set(const DataSet&);
   void set(const tinyxml2::XMLDocument&);
   void set(const string&);

   void set_default();

   void set_threads_number(const int&);

   // Samples set methods

   void set_samples_number(const Index&);

   void set_training();
   void set_selection();
   void set_testing();

   void set_training(const Tensor<Index, 1>&);
   void set_selection(const Tensor<Index, 1>&);
   void set_testing(const Tensor<Index, 1>&);

   void set_samples_unused();
   void set_samples_unused(const Tensor<Index, 1>&);

   void set_sample_use(const Index&, const SampleUse&);
   void set_sample_use(const Index&, const string&);

   void set_samples_uses(const Tensor<SampleUse, 1>&);
   void set_samples_uses(const Tensor<string, 1>&);

   void set_k_fold_cross_validation_samples_uses(const Index&, const Index&);

   // Columns set methods

   void set_default_columns_uses();
   void set_default_classification_columns_uses();

   void set_default_columns_names();

   void set_column_name(const Index&, const string&);

   void set_columns_uses(const Tensor<string, 1>&);
   void set_columns_uses(const Tensor<VariableUse, 1>&);
   void set_columns_unused();
   void set_input_columns_unused();

   void set_column_use(const Index&, const VariableUse&);
   void set_column_use(const string&, const VariableUse&);

   void set_columns_names(const Tensor<string, 1>&);

   void set_columns_number(const Index&);

   void set_binary_simple_columns();

   void binarize_input_data(const type&);

   // Columns other methods

   Tensor<type,2> transform_binary_column(const Tensor<type,1>&) const;

   // Variables set methods

   void set_variables_names(const Tensor<string, 1>&);
   void set_variable_name(const Index&, const string&);

   void set_input();
   void set_target();
   void set_variables_unused();

   void set_input_variables_dimensions(const Tensor<Index, 1>&);

   // Data set methods

   void set_data(const Tensor<type, 2>&);

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

   void set_lags_number(const Index&);
   void set_steps_ahead_number(const Index&);
   void set_time_index(const Index&);

   void set_gmt(Index&);

   void set_display(const bool&);

   // Check methods

   bool is_binary_classification() const;
   bool is_multiple_classification() const;

   bool is_empty() const;

   bool is_less_than(const Tensor<type, 1>&, const type&) const;

   bool is_sample_used(const Index&) const;
   bool is_sample_unused(const Index&) const;

   bool has_data() const;

   bool has_binary_columns() const;
   bool has_categorical_columns() const;
   bool has_time_columns() const;

   bool has_selection() const;

   // Splitting methods

   void split_samples_sequential(const type& training_ratio = static_cast<type>(0.6),
                                   const type& selection_ratio = static_cast<type>(0.2),
                                   const type& testing_ratio = static_cast<type>(0.2));

   void split_samples_random(const type& training_ratio = static_cast<type>(0.6),
                               const type& selection_ratio = static_cast<type>(0.2),
                               const type& testing_ratio = static_cast<type>(0.2));

   // Unusing methods

   Tensor<string, 1> unuse_constant_columns();

   Tensor<Index, 1> unuse_repeated_samples();

   Tensor<string, 1> unuse_uncorrelated_columns(const type& = 0.25);

   // Initialization methods

   void initialize_data(const type&);

   void set_data_random();
   void set_data_binary_random();

   // Descriptives methods

   Tensor<Descriptives, 1> calculate_variables_descriptives() const;
   Tensor<Descriptives, 1> calculate_used_variables_descriptives() const;

   Tensor<Descriptives, 1> calculate_columns_descriptives_positive_samples() const;
   Tensor<Descriptives, 1> calculate_columns_descriptives_negative_samples() const;
   Tensor<Descriptives, 1> calculate_columns_descriptives_categories(const Index&) const;

   Tensor<Descriptives, 1> calculate_columns_descriptives_training_samples() const;
   Tensor<Descriptives, 1> calculate_columns_descriptives_selection_samples() const;

   Tensor<Descriptives, 1> calculate_input_variables_descriptives() const;
   Tensor<Descriptives, 1> calculate_target_variables_descriptives() const;

   Tensor<type, 1> calculate_input_variables_minimums() const;
   Tensor<type, 1> calculate_target_variables_minimums() const;
   Tensor<type, 1> calculate_input_variables_maximums() const;
   Tensor<type, 1> calculate_target_variables_maximums() const;

   Tensor<type, 1> calculate_variables_means(const Tensor<Index, 1>&) const;
   Tensor<type, 1> calculate_used_variables_minimums() const;

   Descriptives calculate_input_descriptives(const Index&) const;

   Tensor<type, 1> calculate_used_targets_mean() const;
   Tensor<type, 1> calculate_selection_targets_mean() const;

   Index calculate_used_negatives(const Index&) const;
   Index calculate_training_negatives(const Index&) const;
   Index calculate_selection_negatives(const Index&) const;
   Index calculate_testing_negatives(const Index&) const;

   // Distribution methods

   Tensor<Histogram, 1> calculate_columns_distribution(const Index& = 10) const;

   // Box and whiskers

   Tensor<BoxPlot, 1> calculate_columns_box_plots() const;

   // Inputs correlations

   Tensor<type, 2> calculate_input_columns_correlations() const;

   void print_inputs_correlations() const;

   void print_top_inputs_correlations(const Index& = 10) const;

   // Inputs-targets correlations

   Tensor<CorrelationResults, 2> calculate_input_target_columns_correlations() const;
   Tensor<type, 2> calculate_input_target_columns_correlations_values() const;

   void print_input_target_columns_correlations() const;

   void print_top_input_target_columns_correlations(const Index& = 10) const;

   // Inputs-targets regressions

   Tensor<RegressionResults, 2> calculate_input_target_columns_regressions() const;

   // Principal components

   Tensor<type, 2> calculate_covariance_matrix() const;

   Tensor<type, 2> perform_principal_components_analysis(const type& = 0.0);

   Tensor<type, 2> perform_principal_components_analysis(const Tensor<type, 2>&, const Tensor<type, 1>&, const type& = 0.0);

   void transform_principal_components_data(const Tensor<type, 2>&);

   void subtract_inputs_mean();

   // Filtering methods

   Tensor<Index, 1> filter_column(const Index&, const type&, const type&);
   Tensor<Index, 1> filter_column(const string&, const type&, const type&);

   Tensor<Index, 1> filter_data(const Tensor<type, 1>&, const Tensor<type, 1>&);

   // Data scaling

   Tensor<string, 1> calculate_default_scaling_methods() const;
   Tensor<string, 1> calculate_default_unscaling_methods() const;
   void scale_data_minimum_maximum(const Tensor<Descriptives, 1>&);
   void scale_minimum_maximum_binary(const type&, const type&, const Index&);
   void scale_data_mean_standard_deviation(const Tensor<Descriptives, 1>&);
   Tensor<Descriptives, 1> scale_data_minimum_maximum();
   Tensor<Descriptives, 1> scale_data_mean_standard_deviation();

   // Input variables scaling

   void scale_input_mean_standard_deviation(const Descriptives&, const Index&);
   Descriptives scale_input_mean_standard_deviation(const Index&);

   void scale_input_standard_deviation(const Descriptives&, const Index&);
   Descriptives scale_input_standard_deviation(const Index&);

   void scale_input_minimum_maximum(const Descriptives&, const Index&);
   Descriptives scale_input_minimum_maximum(const Index&);

   void scale_input_variables_minimum_maximum(const Tensor<Descriptives, 1>&);
   Tensor<Descriptives, 1> scale_input_variables_minimum_maximum();

   void unscale_input_variables_minimum_maximum(const Tensor<Descriptives, 1>&);

   Tensor<Descriptives, 1> scale_input_variables(const Tensor<string, 1>&);

   // Target variables scaling

   void scale_target_minimum_maximum(const Descriptives&, const Index&);
   void scale_target_mean_standard_deviation(const Descriptives&, const Index&);
   void scale_target_logarithmic(const Descriptives&, const Index&);

   void scale_target_variables_minimum_maximum(const Tensor<Descriptives, 1>&);
   Tensor<Descriptives, 1> scale_target_variables_minimum_maximum();

   void scale_target_variables_mean_standard_deviation(const Tensor<Descriptives, 1>&);
   Tensor<Descriptives, 1> scale_target_variables_mean_standard_deviation();

   void scale_target_variables_logarithm(const Tensor<Descriptives, 1>&);
   Tensor<Descriptives, 1> scale_target_variables_logarithm();

   Tensor<Descriptives, 1> scale_target_variables(const string&);
   Tensor<Descriptives, 1> scale_target_variables(const Tensor<string, 1>&);

   // Data unscaling

   void unscale_input_variable_minimum_maximum(const Descriptives&, const Index&);
   void unscale_input_mean_standard_deviation(const Descriptives&, const Index&);
   void unscale_input_variable_standard_deviation(const Descriptives&, const Index&);
   void unscale_input_variables(const Tensor<string,1>&, const Tensor<Descriptives, 1>&);

   void unscale_target_minimum_maximum(const Descriptives&, const Index&);
   void unscale_target_mean_standard_deviation(const Descriptives&, const Index&);
   void unscale_target_logarithmic(const Descriptives&, const Index&);
   void unscale_target_variables(const Tensor<string,1>&, const Tensor<Descriptives, 1>&);

   // Classification methods

   Tensor<Index, 1> calculate_target_distribution() const;

   // Outlier detection

   Tensor<Tensor<Index, 1>, 1> calculate_Tukey_outliers(const type& = 1.5) const;

   void unuse_Tukey_outliers(const type& = 1.5);

   // Time series methods

   void transform_time_series_columns();
   void transform_time_series_data();
   void get_time_series_columns_number(const Index&);
   void set_time_series_data(const Tensor<type, 2>&);

   Tensor<type, 2> get_time_series_column_data(const Index&) const;
   Tensor<type, 2> calculate_autocorrelations(const Index& = 10) const;
   Tensor<Tensor<type, 1>, 2> calculate_cross_correlations(const Index& = 10) const;
   Tensor<type, 2> calculate_lag_plot() const;
   Tensor<type, 2> calculate_lag_plot(const Index&);

   // Data generation

   void generate_constant_data(const Index&, const Index&);
   void generate_random_data(const Index&, const Index&);
   void generate_sequential_data(const Index&, const Index&);
   void generate_paraboloid_data(const Index&, const Index&);
   void generate_Rosenbrock_data(const Index&, const Index&);
   void generate_inputs_selection_data(const Index&, const Index&);
   void generate_sum_data(const Index&, const Index&);

   void generate_data_binary_classification(const Index&, const Index&);
   void generate_data_multiple_classification(const Index&, const Index&, const Index&);

   // Serialization methods

   void print() const;
   void print_summary() const;

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

   void check_input_csv(const string&, const char&) const;
   Tensor<type, 2> read_input_csv(const string&, const char&, const string&, const bool&, const bool&) const;

   // Trasform methods

   void transform_time_series();
   void transform_association();

   void fill_time_series(const Index&);

   void numeric_to_categorical(const Index&);

   // Missing values

   bool has_nan() const;

   bool has_nan_row(const Index&) const;

   void print_missing_values_information() const;

   void impute_missing_values_unuse();
   void impute_missing_values_mean();
   void impute_missing_values_median();

   void scrub_missing_values();

   Tensor<Index, 1> count_nan_columns() const;
   Index count_rows_with_nan() const;
   Index count_nan() const;

   void set_missing_values_number(const Index&);
   void set_missing_values_number();

   void set_columns_missing_values_number(const Tensor<Index, 1>&);
   void set_columns_missing_values_number();

   void set_rows_missing_values_number(const Index&);
   void set_rows_missing_values_number();

   // Other methods

   void fix_repeated_names();

   // scaling

   void set_min_max_range(const type min, const type max);

   // Eigen methods

   Tensor<Index, 1> push_back(const Tensor<Index, 1>&, const Index&) const;
   Tensor<string, 1> push_back(const Tensor<string, 1>&, const string&) const;

   void initialize_sequential_eigen_tensor(Tensor<Index, 1>&, const Index&, const Index&, const Index&) const;
   void intialize_sequential_eigen_type_tensor(Tensor<type, 1>&, const type&, const type&, const type&) const;

   Tensor<Index, 2> split_samples(const Tensor<Index, 1>&, const Index&) const;

   void fill_submatrix(const Tensor<type, 2>& matrix,
             const Tensor<Index, 1>& rows_indices,
             const Tensor<Index, 1>& columns_indices, type*submatrix);

   bool get_has_rows_labels() const;

   void shuffle();

private:

   NonBlockingThreadPool* non_blocking_thread_pool = nullptr;
   ThreadPoolDevice* thread_pool_device = nullptr;

   /// Data file name.

   string data_file_name;

   /// Separator character.

   Separator separator = Comma;

   /// Missing values label.

   string missing_values_label = "NA";

   /// Number of lags.

   Index lags_number;

   /// Number of steps ahead.

   Index steps_ahead;

   /// Min Max Range Scaling

   type min_range = -1;
   type max_range = 1;

   /// Data Matrix.
   /// The number of rows is the number of samples.
   /// The number of columns is the number of variables.

   Tensor<type, 2> data;

   /// Time series data matrix.
   /// The number of rows is the number of samples before time series transfomration.
   /// The number of columns is the number of variables before time series transformation.

   Tensor<type, 2> time_series_data;

   Tensor<Column, 1> time_series_columns;

   /// Display messages to screen.

   bool display = true;

   /// Index where time variable is located for forecasting applications.

   Index time_index;

   /// Missing values method object.

   MissingValuesMethod missing_values_method = Unuse;

   // Samples

   Tensor<SampleUse, 1> samples_uses;

   // Variables

   // Reader

   void read_csv_1();

   void read_csv_2_simple();
   void read_csv_3_simple();

   void read_csv_2_complete();
   void read_csv_3_complete();

   void check_separators(const string&) const;

   void check_special_characters(const string&) const;

   /// Header which contains variables name.

   bool has_columns_names = false;

   Tensor<Index, 1> input_variables_dimensions;

   Tensor<Column, 1> columns;

   /// Header wihch contains the rows label.

   bool has_rows_labels = false;

   Tensor<string, 1> rows_labels;

   Index gmt = 0;

   Tensor<Tensor<string, 1>, 1> data_file_preview;

   Eigen::array<IndexPair<Index>, 1> product_vector_vector = {IndexPair<Index>(0, 0)}; // Vector product, (0,0) first vector is transpose

   /// Missing values

   Index missing_values_number;

   Tensor<Index, 1> columns_missing_values_number;

   Index rows_missing_values_number;

#ifdef OPENNN_CUDA
    #include "../../opennn-cuda/opennn_cuda/data_set_cuda.h"
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
