//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   D A T A   S E T   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#ifndef DATASET_H
#define DATASET_H
#define _SILENCE_EXPERIMENTAL_FILESYSTEM_DEPRECATION_WARNING

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
#include <list>
#include <filesystem>
#include <experimental/filesystem>

// OpenNN includes

#include "config.h"
#include "statistics.h"
#include "scaling.h"
#include "correlations.h"
#include "opennn_strings.h"
#include "tensor_utilities.h"
#include "text_analytics.h"

// Filesystem namespace

#ifdef __APPLE__
#include <Availability.h> // for deployment target to support pre-catalina targets without std::fs
#endif
#if ((defined(_MSVC_LANG) && _MSVC_LANG >= 201703L) || (defined(__cplusplus) && __cplusplus >= 201703L)) && defined(__has_include)
#if __has_include(<filesystem>) && (!defined(__MAC_OS_X_VERSION_MIN_REQUIRED) || __MAC_OS_X_VERSION_MIN_REQUIRED >= 101500)
#define GHC_USE_STD_FS
#include <filesystem>
namespace fs = std::filesystem;
#endif
#endif
#ifndef GHC_USE_STD_FS
#include "filesystem.h"
namespace fs = ghc::filesystem;
#endif

using namespace std;
using namespace Eigen;
using namespace fs;

namespace opennn
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

    /// This enumeration represents the data file string codification
    /// (utf8, shift_jis)

    enum class Codification{UTF8, SHIFT_JIS};

    explicit DataSet(const string&, const char&, const bool&, const Codification& = Codification::UTF8);

    // Destructor

    virtual ~DataSet();

    // Enumerations

    /// Enumeration of available separators for the data file.

    enum class Separator{None, Space, Tab, Comma, Semicolon};

    /// Enumeration of available methods for missing values in the data.

    enum class MissingValuesMethod{Unuse, Mean, Median};

    /// Enumeration of the learning tasks.

    enum class ProjectType{Approximation, Classification, Forecasting, ImageClassification, TextClassification};

    /// This enumeration represents the possible uses of an sample
    /// (training, selection, testing or unused).

    enum class SampleUse{Training, Selection, Testing, Unused};

    /// This enumeration represents the possible uses of an variable
    /// (input, target, time or unused).

    enum class VariableUse{Id, Input, Target, Time, Unused};

    /// This enumeration represents the data type of a column
    /// (numeric, binary, categorical or time).

    enum class ColumnType{Numeric, Binary, Categorical, DateTime, Constant};

    // Structs

    /// This structure represents the columns of the DataSet.

    struct Column
    {
        /// Default constructor.

        Column();

        /// Values constructor

        Column(const string&,
               const VariableUse&,
               const ColumnType& = ColumnType::Numeric,
               const Scaler& = Scaler::MeanStandardDeviation,
               const Tensor<string, 1>& = Tensor<string, 1>(),
               const Tensor<VariableUse, 1>& = Tensor<VariableUse, 1>());

        /// Column name.

        string name = "";

        /// Column use.

        VariableUse column_use = VariableUse::Input;

        /// Column type.

        ColumnType type = ColumnType::Numeric;

        /// Categories within the column.

        Tensor<string, 1> categories;

        /// Categories use.

        Tensor<VariableUse, 1> categories_uses;

        Scaler scaler= Scaler::MeanStandardDeviation;


        // Methods

        Index get_variables_number() const;

        Index get_categories_number() const;
        Index get_used_categories_number() const;

        Tensor<string, 1> get_used_variables_names() const;

        void set_scaler(const Scaler&);
        void set_scaler(const string&);

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

        void print() const;
    };

    // Project type

    ProjectType get_project_type() const;

    string get_project_type_string(const DataSet::ProjectType&) const;

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
    Index get_time_series_data_rows_number() const;
    Tensor<Column, 1> get_input_columns() const;
    Tensor<bool, 1> get_input_columns_binary() const;
    Tensor<Column, 1> get_target_columns() const;
    Tensor<Column, 1> get_used_columns() const;

    Index get_columns_number() const;

    Index get_input_columns_number() const;
    Index get_input_time_series_columns_number() const;
    Index get_target_columns_number() const;
    Index get_target_time_series_columns_number() const;
    Index get_time_columns_number() const;
    Index get_unused_columns_number() const;
    Index get_used_columns_number() const;

    Index get_column_index(const string&) const;
    Index get_column_index(const Index&) const;

    Tensor<Index, 1> get_input_columns_indices() const;
    Tensor<Index, 1> get_input_time_series_columns_indices() const;
    Tensor<Index, 1> get_target_columns_indices() const;
    Tensor<Index, 1> get_target_time_series_columns_indices() const;
    Tensor<Index, 1> get_unused_columns_indices() const;
    Tensor<Index, 1> get_used_columns_indices() const;

    Tensor<string, 1> get_columns_names() const;

    Tensor<string, 1> get_input_columns_names() const;
    Tensor<string, 1> get_target_columns_names() const;
    Tensor<string, 1> get_used_columns_names() const;

    ColumnType get_column_type(const Index& index) const {return columns[index].type;}

    VariableUse get_column_use(const Index& ) const;
    Tensor<VariableUse, 1> get_columns_uses() const;

    // Variables get methods

    Index get_variables_number() const;
    Index get_time_series_variables_number() const;

    Index get_input_variables_number() const;
    Index get_target_variables_number() const;
    Index get_unused_variables_number() const;
    Index get_used_variables_number() const;

    string get_variable_name(const Index&) const;
    Tensor<string, 1> get_variables_names() const;
    Tensor<string, 1> get_time_series_variables_names() const;

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
    Index get_input_variables_rank() const;

    // Scalers get methods

    Tensor<Scaler, 1> get_columns_scalers() const;

    Tensor<Scaler, 1> get_input_variables_scalers() const;
    Tensor<Scaler, 1> get_target_variables_scalers() const;

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
    Tensor<type, 2> get_column_data(const Index&, const Tensor<Index, 1>&) const;
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
    string get_text_separator_string() const;

    const Codification get_codification() const;
    const string get_codification_string() const;

    const string& get_missing_values_label() const;

    const Index& get_lags_number() const;
    const Index& get_steps_ahead() const;
    const string& get_time_column() const;
    Index get_time_series_time_column_index() const;

    const Index& get_short_words_length() const;
    const Index& get_long_words_length() const;
    const Tensor<Index,1>& get_words_frequencies() const;

    static Tensor<string, 1> get_default_columns_names(const Index&);

    static Scaler get_scaling_unscaling_method(const string&);

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
    void set(const string&, const char&, const bool&);
    void set(const string&, const char&, const bool&, const DataSet::Codification&);

    void set_default();

    void set_project_type_string(const string&);
    void set_project_type(const ProjectType&);

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

    // Columns set methods

    void set_columns(const Tensor<Column, 1>&);

    void set_default_columns_uses();

    void set_default_columns_names();

    void set_column_name(const Index&, const string&);

    void set_columns_uses(const Tensor<string, 1>&);
    void set_columns_uses(const Tensor<VariableUse, 1>&);
    void set_columns_unused();
    void set_input_target_columns(const Tensor<Index, 1>&, const Tensor<Index, 1>&);
    void set_input_columns_unused();

    void set_input_columns(const Tensor<Index, 1>&, const Tensor<bool, 1>&);

    void set_column_use(const Index&, const VariableUse&);
    void set_column_use(const string&, const VariableUse&);

    void set_column_type(const Index&, const ColumnType&);
    void set_column_type(const string&, const ColumnType&);

    void set_columns_names(const Tensor<string, 1>&);

    void set_columns_number(const Index&);

    void set_columns_scalers(const Scaler&);

    void set_binary_simple_columns();

    // Columns other methods

    void check_constant_columns();

    Tensor<type, 2> transform_binary_column(const Tensor<type, 1>&) const;

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

    void set_has_text_data(const bool&);

    void set_separator(const Separator&);
    void set_separator(const string&);
    void set_separator(const char&);
    void set_text_separator(const Separator&);
    void set_text_separator(const string&);

    void set_codification(const Codification&);
    void set_codification(const string&);

    void set_missing_values_label(const string&);
    void set_missing_values_method(const MissingValuesMethod&);
    void set_missing_values_method(const string&);

    void set_lags_number(const Index&);
    void set_steps_ahead_number(const Index&);
    void set_time_column(const string&);

    void set_short_words_length(const Index&);
    void set_long_words_length(const Index&);
    void set_words_frequencies(const Tensor<Index,1>&);

    void set_gmt(Index&);

    void set_display(const bool&);

    // Check methods

    bool is_empty() const;

    bool is_sample_used(const Index&) const;
    bool is_sample_unused(const Index&) const;

    bool has_binary_columns() const;
    bool has_categorical_columns() const;
    bool has_time_columns() const;
    bool has_time_time_series_columns() const;

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

    Tensor<string, 1> unuse_uncorrelated_columns(const type& = type(0.25));

    // Initialization methods

    void set_data_constant(const type&);

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

    Tensor<Descriptives, 1> calculate_testing_target_variables_descriptives() const;

    Tensor<type, 1> calculate_input_variables_minimums() const;
    Tensor<type, 1> calculate_target_variables_minimums() const;
    Tensor<type, 1> calculate_input_variables_maximums() const;
    Tensor<type, 1> calculate_target_variables_maximums() const;

    Tensor<type, 1> calculate_variables_means(const Tensor<Index, 1>&) const;
    Tensor<type, 1> calculate_used_variables_minimums() const;

    Tensor<type, 1> calculate_used_targets_mean() const;
    Tensor<type, 1> calculate_selection_targets_mean() const;

    Index calculate_used_negatives(const Index&);
    Index calculate_training_negatives(const Index&) const;
    Index calculate_selection_negatives(const Index&) const;
    Index calculate_testing_negatives(const Index&) const;

    // Distribution methods

    Tensor<Histogram, 1> calculate_columns_distribution(const Index& = 10) const;

    // Box and whiskers

    Tensor<BoxPlot, 1> calculate_columns_box_plots() const;

    // Inputs correlations

    Tensor<Correlation, 2> calculate_input_columns_correlations() const;

    void print_inputs_correlations() const;

    void print_top_inputs_correlations() const;

    // Inputs-targets correlations

    Tensor<Correlation, 2> calculate_input_target_columns_correlations() const;

    void print_input_target_columns_correlations() const;

    void print_top_input_target_columns_correlations() const;

    // Filtering methods

    Tensor<Index, 1> filter_data(const Tensor<type, 1>&, const Tensor<type, 1>&);

    // Scaling methods

    void set_default_columns_scalers();

    // Data scaling

    Tensor<Descriptives, 1> scale_data();

    Tensor<Descriptives, 1> scale_input_variables();
    Tensor<Descriptives, 1> scale_target_variables();

    // Data unscaling

    void unscale_data(const Tensor<Descriptives, 1>&);

    void unscale_input_variables(const Tensor<Descriptives, 1>&);
    void unscale_target_variables(const Tensor<Descriptives, 1>&);

    // Classification methods

    Tensor<Index, 1> calculate_target_distribution() const;

    // Tuckey outlier detection

    Tensor<Tensor<Index, 1>, 1> calculate_Tukey_outliers(const type& = type(1.5)) const;

    void unuse_Tukey_outliers(const type& = type(1.5)) const;

    // Local outlier factor

    Tensor<Index, 1> calculate_local_outlier_factor_outliers(const Index& = 20, const Index& = 0, const type& = type(0)) const;

    void unuse_local_outlier_factor_outliers(const Index& = 20, const type& = type(1.5));

    // Isolation Forest outlier

    Tensor<Index, 1> calculate_isolation_forest_outliers(const Index& = 100, const Index& = 256, const type& = type(0)) const;

    void unuse_isolation_forest_outliers(const Index& = 20, const type& = type(1.5));

    // Time series methods

    void transform_time_series();

    void transform_time_series_columns();
    void transform_time_series_data();
    void get_time_series_columns_number(const Index&);
    void set_time_series_data(const Tensor<type, 2>&);
    void set_time_series_columns_number(const Index&);

    Tensor<type, 2> get_time_series_column_data(const Index&) const;
    Tensor<type, 2> calculate_autocorrelations(const Index& = 10) const;
    Tensor<type, 3> calculate_cross_correlations(const Index& = 10) const;

    // Image classification methods

    Index get_channels_number() const;
    Index get_image_width() const;
    Index get_image_height() const;
    Index get_image_padding() const;
    Index get_image_size() const;

    void set_channels_number(const int&);
    void set_image_width(const int&);
    void set_image_height(const int&);
    void set_image_padding(const int&);


    // Text classification methods

    Tensor<type,1> sentence_to_data(const string&) const;

    // Data generation

    void generate_constant_data(const Index&, const Index&, const type&);
    void generate_random_data(const Index&, const Index&);
    void generate_sequential_data(const Index&, const Index&);
    void generate_Rosenbrock_data(const Index&, const Index&);
    void generate_sum_data(const Index&, const Index&);

    // Serialization methods

    void print() const;

    void from_XML(const tinyxml2::XMLDocument&);
    void write_XML(tinyxml2::XMLPrinter&) const;

    void save(const string&) const;
    void load(const string&);

    void print_columns() const;
    void print_columns_types() const;
    void print_columns_uses() const;

    void print_data() const;
    void print_data_preview() const;

    void print_data_file_preview() const;

    void save_data() const;

    void save_data_binary(const string&) const;

    void save_time_series_data_binary(const string&) const;

    void load_data_binary();

    void load_time_series_data_binary(const string&);

    void check_input_csv(const string&, const char&) const;

    Tensor<type, 2> read_input_csv(const string&, const char&, const string&, const bool&, const bool&) const;

    string decode(const string&) const;

    // Data load methods

    void read_csv();

    Tensor<unsigned char, 1> read_bmp_image(const string&);

    void read_bmp();

    void read_ground_truth();

    void read_txt();

    // Image methods

    void sort_channel(Tensor<unsigned char,1>&, Tensor<unsigned char,1>&, const int& );

    Tensor<unsigned char, 1> remove_padding(Tensor<unsigned char, 1>&, const int&,const int&, const int& );

    Tensor<unsigned char, 1> resize_image(Tensor<unsigned char, 1> &, const Index &, const Index &, const Index &);

    Index get_bounding_boxes_number_from_XML(const string&);

    Index get_label_classes_number_from_XML(const string&);

    // Trasform methods

    void fill_time_series(const Index&);

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

    // Eigen methods

    void initialize_sequential(Tensor<Index, 1>&, const Index&, const Index&, const Index&) const;
    void intialize_sequential(Tensor<type, 1>&, const type&, const type&, const type&) const;

    Tensor<Index, 2> split_samples(const Tensor<Index, 1>&, const Index&) const;

    bool get_has_rows_labels() const;
    bool get_has_text_data() const;

    void shuffle();

    // Reader

    void read_csv_1();

    void read_csv_2_simple();
    void read_csv_3_simple();

    void read_csv_2_complete();
    void read_csv_3_complete();

    void check_separators(const string&) const;

    void check_special_characters(const string&) const;

private:

    DataSet::ProjectType project_type;

    ThreadPool* thread_pool = nullptr;
    ThreadPoolDevice* thread_pool_device = nullptr;

    // DATA

    /// Data Matrix.
    /// The number of rows is the number of samples.
    /// The number of columns is the number of variables.

    Tensor<type, 2> data;

    // Samples

    Tensor<SampleUse, 1> samples_uses;

    Tensor<string, 1> rows_labels;

    // Columns

    Tensor<Column, 1> columns;

    Tensor<Index, 1> input_variables_dimensions;

    // DATA FILE

    /// Data file name.

    string data_file_name;

    /// Separator character.

    Separator separator = Separator::Comma;

    /// Missing values label.

    string missing_values_label = "NA";

    /// Header which contains variables name.

    bool has_columns_names = false;

    /// Header which contains the rows label.

    bool has_rows_labels = false;

    /// Image classification model

    bool convolutional_model = false;

    /// Class containing file string codification

    Codification codification = Codification::UTF8;


    Tensor<Tensor<string, 1>, 1> data_file_preview;

    // TIME SERIES

    /// Index where time variable is located for forecasting applications.

    string time_column;

    /// Number of lags.

    Index lags_number = 0;

    /// Number of steps ahead.

    Index steps_ahead = 0;

    /// Time series data matrix.
    /// The number of rows is the number of samples before time series transformation.
    /// The number of columns is the number of variables before time series transformation.

    Tensor<type, 2> time_series_data;

    Tensor<Column, 1> time_series_columns;

    Index gmt = 0;

    // TEXT CLASSIFICATION

    Separator text_separator = Separator::Tab;

    Index short_words_length = 2;

    Index long_words_length = 15;

    Tensor<Index, 1> words_frequencies;

    TextAnalytics text_analytics;

    Tensor<string, 1> stop_words = text_analytics.get_stop_words();

    Tensor<string, 2> text_data_file_preview;

    // MISSING VALUES

    /// Missing values method.

    MissingValuesMethod missing_values_method = MissingValuesMethod::Unuse;

    /// Missing values

    Index missing_values_number;

    Tensor<Index, 1> columns_missing_values_number;

    Index rows_missing_values_number;

    /// Display messages to screen.

    bool display = true;

    const Eigen::array<IndexPair<Index>, 1> product_vector_vector = {IndexPair<Index>(0, 0)}; // Vector product, (0,0) first vector is transpose

    // Image treatment

    static size_t number_of_elements_in_directory(const fs::path& path);

    Index images_number = 0;
    Index channels_number = 0;
    Index image_width = 0;
    Index image_height = 0;
    Index padding = 0;

    Tensor<string, 1> labels_tokens;

    Index width_no_padding;
    // Local Outlier Factor

    Tensor<Index, 1> select_outliers_via_standard_deviation(const Tensor<type, 1>&, const type & = type(2.0), bool = true) const;

    Tensor<Index, 1> select_outliers_via_contamination(const Tensor<type, 1>&, const type & = type(0.05), bool = true) const;

    type calculate_euclidean_distance(const Tensor<Index, 1>&, const Index&, const Index&) const;

    Tensor<type, 2> calculate_distance_matrix(const Tensor<Index, 1>&) const;

    Tensor<list<Index>, 1> calculate_k_nearest_neighbors(const Tensor<type, 2>&, const Index& = 20) const;

    Tensor<Tensor<type, 1>, 1> get_kd_tree_data() const;

    Tensor<Tensor<Index, 1>, 1> create_bounding_limits_kd_tree(const Index&) const;

    void create_kd_tree(Tensor<Tensor<type, 1>, 1>&, const Tensor<Tensor<Index, 1>, 1>&) const;

    Tensor<list<Index>, 1> calculate_bounding_boxes_neighbors(const Tensor<Tensor<type, 1>, 1>&,
                                                              const Tensor<Index, 1>&,
                                                              const Index&, const Index&) const;

    Tensor<list<Index>, 1> calculate_kd_tree_neighbors(const Index& = 20, const Index& = 40) const;

    Tensor<type, 1> calculate_average_reachability(Tensor<list<Index>, 1>&, const Index&) const;

    Tensor<type, 1> calculate_local_outlier_factor(Tensor<list<Index>, 1>&, const Tensor<type, 1>&, const Index &) const;

    // Isolation Forest

    void calculate_min_max_indices_list(list<Index>&, const Index&, type&, type&) const;

    Index split_isolation_tree(Tensor<type, 2>&, list<list<Index>>&, list<Index>&) const;

    Tensor<type, 2> create_isolation_tree(const Tensor<Index, 1>&, const Index&) const;

    Tensor<Tensor<type, 2>, 1> create_isolation_forest(const Index&, const Index&, const Index&) const;

    type calculate_tree_path(const Tensor<type, 2>&, const Index&, const Index&) const;

    Tensor<type, 1> calculate_average_forest_paths(const Tensor<Tensor<type, 2>, 1>&, const Index&) const;

};


#ifdef OPENNN_CUDA
#include "../../opennn-cuda/opennn-cuda/data_set_cuda.h"
#endif

struct DataSetBatch
{
    /// Default constructor.

    DataSetBatch() {}

    DataSetBatch(const Index&, DataSet*);

    /// Destructor.

    virtual ~DataSetBatch() {}

    Index get_batch_size() const;

    void set(const Index&, DataSet*);

    void fill(const Tensor<Index, 1>&, const Tensor<Index, 1>&, const Tensor<Index, 1>&);

    void print() const;

    Index batch_size = 0;

    DataSet* data_set_pointer = nullptr;

    type* inputs_data;

    Tensor<Index, 1> inputs_dimensions;

    type* targets_data;

    Tensor<Index, 1> targets_dimensions;

};


}

#endif

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2022 Artificial Intelligence Techniques, SL.
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
