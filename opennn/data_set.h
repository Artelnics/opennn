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

#include <string>

// OpenNN includes

#include "tinyxml2.h"
#include "histogram.h"
#include "box_plot.h"
#include "config.h"
#include "correlation.h"
#include "scaling.h"

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <Windows.h>
#endif

using namespace Eigen;

// Filesystem namespace

// #ifdef __APPLE__
// #include <Availability.h> // for deployment target to support pre-catalina targets without fs
// #endif
// #if((defined(_MSVC_LANG) && _MSVC_LANG >= 201703L) || (defined(__cplusplus) && __cplusplus >= 201703L)) && defined(__has_include)
// #if __has_include(<filesystem>) && (!defined(__MAC_OS_X_VERSION_MIN_REQUIRED) || __MAC_OS_X_VERSION_MIN_REQUIRED >= 101500)
// #define GHC_USE_STD_FS
// #include <filesystem>
// namespace fs = filesystem;
// #endif
// #endif
// #ifndef GHC_USE_STD_FS
// #include "filesystem.h"
// namespace fs = ghc::filesystem;
// #endif

// using namespace fs;

namespace opennn
{

#ifdef _WIN32
    inline std::wstring string_to_wide_string(const std::string& string)
    {
        if (string.empty())
        {
            return L"";
        }

        const auto size_needed = MultiByteToWideChar(CP_UTF8, 0, string.data(), (int)string.size(), nullptr, 0);
        if (size_needed <= 0)
        {
            throw std::runtime_error("MultiByteToWideChar() failed: " + std::to_string(size_needed));
        }

        std::wstring result(size_needed, 0);
        MultiByteToWideChar(CP_UTF8, 0, string.data(), (int)string.size(), const_cast<wchar_t*>(result.data()), size_needed);
        return result;
    }
#endif

/// This class represents the concept of data set for data modelling problems, such as approximation, classification or forecasting.

///
/// It basically consists of a data Matrix separated by raw_variables.
/// These raw_variables can take different categories depending on the data hosted in them.
///
/// With OpenNN DataSet class you can edit the data to prepare your model, such as eliminating missing values,
/// calculating correlations between variables (inputs and targets), not using certain variables or samples, etc \dots.

class DataSet
{

public:

    // Constructors

    explicit DataSet();

    explicit DataSet(const Tensor<type, 2>&);

    explicit DataSet(const Tensor<type, 2>&, const Index&, const Tensor<string, 1>&, bool&);

    explicit DataSet(const Index&, const Index&);

    explicit DataSet(const Index&, const Index&, const Index&);

    explicit DataSet(const Tensor<type, 1>&, const Index&);

    void set_default_columns_scalers();
    void set_indra_columns(const Tensor<string, 1>&);

    /// This enumeration represents the data file string codification (utf8, shift_jis)

    enum class Codification{UTF8, SHIFT_JIS};

    explicit DataSet(const string&, const char&, const bool&, const Codification& = Codification::UTF8);

    // Destructor

    virtual ~DataSet();

    // Enumerations

    /// Enumeration of available separators for the data file.

    enum class Separator{None, Space, Tab, Comma, Semicolon};

    /// Enumeration of available methods for missing values in the data.

    enum class MissingValuesMethod{Unuse, Mean, Median, Interpolation};

    /// Enumeration of the learning tasks.

    enum class ModelType{Approximation, Classification, Forecasting, AutoAssociation, TextClassification, ImageClassification};

    /// This enumeration represents the possible uses of an sample
    /// (training, selection, testing or unused).

    enum class SampleUse{Training, Selection, Testing, Unused};

    /// This enumeration represents the possible uses of an variable
    /// (input, target, time or unused).

    enum class VariableUse{Id, Input, Target, Time, Unused, Context};

    /// This enumeration represents the data type of a raw_variable
    /// (numeric, binary, categorical or time).

    enum class RawVariableType{Numeric, Binary, Categorical, DateTime, Constant};

    // Structs

    /// This structure represents the raw variables of the DataSet.

    struct RawVariable
    {
        /// Default constructor.

        RawVariable();

        /// Values constructor

        RawVariable(const string&,
               const DataSet::VariableUse&,
               const DataSet::RawVariableType& = DataSet::RawVariableType::Numeric,
               const Scaler& = Scaler::MeanStandardDeviation,
               const Tensor<string, 1>& = Tensor<string, 1>(),
               const Tensor<DataSet::VariableUse, 1>& = Tensor<DataSet::VariableUse, 1>());

        /// Raw variable name.

        string name = "";

        /// Raw variable use.

        DataSet::VariableUse raw_variable_use = DataSet::VariableUse::Input;

        /// Raw variable type.

        DataSet::RawVariableType type = DataSet::RawVariableType::Numeric;

        /// Categories within the raw variable.

        Tensor<string, 1> categories;

        /// Categories use.

        Tensor<DataSet::VariableUse, 1> categories_uses;

        Scaler scaler = Scaler::MeanStandardDeviation;

        // Methods

        Index get_variables_number() const;

        Index get_categories_number() const;
        Index get_used_categories_number() const;

        Tensor<string, 1> get_used_variables_names() const;

        void set_scaler(const Scaler&);
        void set_scaler(const string&);

        void set_use(const DataSet::VariableUse&);
        void set_use(const string&);

        void set_type(const string&);

        void add_category(const string&);

        void set_categories(const Tensor<string, 1>&);
        void set_categories_uses(const Tensor<string, 1>&);
        void set_categories_uses(const DataSet::VariableUse&);

        virtual void from_XML(const tinyxml2::XMLDocument&);
        virtual void write_XML(tinyxml2::XMLPrinter&) const;

        void print() const;
    };

    // Project type

    ModelType get_model_type() const;

    string get_model_type_string(const DataSet::ModelType&) const;

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

    // Create Box plot from histogram

    Tensor<type, 1> box_plot_from_histogram(const Histogram&, const Index&) const;

    // Raw variables get methods

    Tensor<RawVariable, 1> get_raw_variables() const;
    Tensor<RawVariable, 1> get_input_raw_variables() const;
    //Tensor<bool, 1> get_input_raw_variables_binary() const;
    Tensor<RawVariable, 1> get_target_raw_variables() const;
    Tensor<RawVariable, 1> get_used_raw_variables() const;

    Index get_raw_variables_number() const;
    Index get_constant_raw_variables_number() const;

    Index get_input_raw_variables_number() const;
    Index get_target_raw_variables_number() const;
    Index get_time_raw_variables_number() const;
    Index get_unused_raw_variables_number() const;
    Index get_used_raw_variables_number() const;

    Index get_input_and_unused_variables_number() const;

    Tensor<Index, 1> get_raw_variables_index(const Tensor<string, 1>&) const;

    Index get_raw_variable_index(const string&) const;
    Index get_raw_variable_index(const Index&) const;

    Tensor<Index, 1> get_input_raw_variables_indices() const;
    Tensor<Index, 1> get_target_raw_variables_indices() const;
    Tensor<Index, 1> get_unused_raw_variables_indices() const;
    Tensor<Index, 1> get_used_raw_variables_indices() const;
    Tensor<Index, 1> get_numeric_input_raw_variables() const;

    Tensor<string, 1> get_raw_variables_names() const;

    Tensor<string, 1> get_input_raw_variables_names() const;
    Tensor<string, 1> get_target_raw_variables_names() const;
    Tensor<string, 1> get_used_raw_variables_names() const;

    RawVariableType get_raw_variable_type(const Index& index) const {return raw_variables[index].type;}

    VariableUse get_raw_variable_use(const Index&) const;
    Tensor<VariableUse, 1> get_raw_variables_uses() const;

    // Variables get methods

    Index get_variables_number() const;

    Index get_input_variables_number() const;
    Index get_target_variables_number() const;
    Index get_unused_variables_number() const;
    Index get_used_variables_number() const;
    Index get_numerical_input_raw_variables_number() const;

    string get_numeric_variable_name(const Index&) const;
    Tensor<string, 1> get_variables_names() const;

    Tensor<string, 1> get_input_variables_names() const;
    Tensor<string, 1> get_target_variables_names() const;

    Index get_variable_index(const string&name) const;

    Tensor<Index, 1> get_numeric_variable_indices(const Index&) const;
    Tensor<Index, 1> get_categorical_to_indices(const Index&) const;
    Tensor<Index, 1> get_unused_variables_indices() const;
    Tensor<Index, 1> get_used_variables_indices() const;
    Tensor<Index, 1> get_input_variables_indices() const;
    Tensor<Index, 1> get_numeric_input_variables_indices() const;
    Tensor<Index, 1> get_target_variables_indices() const;

    VariableUse get_numeric_variable_use(const Index&) const;
    Tensor<VariableUse, 1> get_variables_uses() const;

    const dimensions& get_input_dimensions() const;
    Index get_input_variables_rank() const;

    const dimensions& get_target_dimensions() const;

    // Scalers get methods

    Tensor<Scaler, 1> get_raw_variables_scalers() const;

    Tensor<Scaler, 1> get_input_variables_scalers() const;
    Tensor<Scaler, 1> get_target_variables_scalers() const;

    // Batches get methods

    Tensor<Index, 2> get_batches(const Tensor<Index,1>&, const Index&, const bool&, const Index& buffer_size= 100) const;

    // Data get methods

    const Tensor<type, 2>& get_data() const;
    Tensor<type, 2>* get_data_p();

    Tensor<type, 2> get_training_data() const;
    Tensor<type, 2> get_selection_data() const;
    Tensor<type, 2> get_testing_data() const;

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

    Tensor<type, 2> get_raw_variables_data(const Tensor<Index, 1>&) const;

    Tensor<type, 2> get_raw_variable_data(const Index&) const;
    Tensor<type, 2> get_raw_variable_data(const Index&, const Tensor<Index, 1>&) const;
    Tensor<type, 2> get_raw_variable_data(const Tensor<Index, 1>&) const;
    Tensor<type, 2> get_raw_variable_data(const string&) const;

    string get_sample_category(const Index&, const Index&) const;
    Tensor<type, 1> get_sample(const Index&) const;
    void add_sample(const Tensor<type, 1>&);

    Tensor<type, 1> get_variable_data(const Index&) const;
    Tensor<type, 1> get_variable_data(const string&) const;

    Tensor<type, 1> get_variable_data(const Index&, const Tensor<Index, 1>&) const;
    Tensor<type, 1> get_variable_data(const string&, const Tensor<Index, 1>&) const;

    Tensor<Tensor<string, 1>, 1> get_data_file_preview() const;

    // Members get methods

    MissingValuesMethod get_missing_values_method() const;

    const string& get_data_source_path() const;

    const bool& get_header_line() const;
    const bool& get_rows_label() const;

    Tensor<string, 1> get_rows_label_tensor() const;
    Tensor<string, 1> get_selection_rows_label_tensor();
    Tensor<string, 1> get_testing_rows_label_tensor();

    const Separator& get_separator() const;
    string get_separator_string() const;
    string get_separator_name() const;

    const Codification get_codification() const;
    const string get_codification_string() const;

    const string& get_missing_values_label() const;

    static Tensor<string, 1> get_default_raw_variables_names(const Index&);
    static string get_raw_variable_type_string(const RawVariableType&);
    static string get_raw_variable_use_string(const VariableUse&);

    static Scaler get_scaling_unscaling_method(const string&);

    Index get_gmt() const;

    const bool& get_display() const;

    bool get_augmentation() const;

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
    void set(const Tensor<type, 1>&, const Index&);
    void set_default();

    void set_model_type_string(const string&);
    void set_model_type(const ModelType&);

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
    void set_samples_uses(const Tensor<Index, 1>&, const SampleUse);

    // raw_variables set methods

//    void set_indra_columns(const Tensor<string, 1>&);

    void set_raw_variables(const Tensor<RawVariable, 1>&);
    void set_default_raw_variables_uses();

    void set_default_raw_variables_names();

    void set_raw_variable_name(const Index&, const string&);

    void set_raw_variables_uses(const Tensor<string, 1>&);
    void set_raw_variables_uses(const Tensor<VariableUse, 1>&);
    void set_raw_variables_unused();
    void set_raw_variables_types(const Tensor<string, 1>&);
    void set_input_target_raw_variables(const Tensor<Index, 1>&, const Tensor<Index, 1>&);
    void set_input_target_raw_variables(const Tensor<string, 1>&, const Tensor<string, 1>&);
    void set_input_raw_variables_unused();

    void set_raw_variables_unused(const Tensor<Index, 1>&);

    void set_input_raw_variables(const Tensor<Index, 1>&, const Tensor<bool, 1>&);

    void set_raw_variable_use(const Index&, const VariableUse&);
    void set_raw_variable_use(const string&, const VariableUse&);

    void set_raw_variable_type(const Index&, const RawVariableType&);
    void set_raw_variable_type(const string&, const RawVariableType&);

    void set_all_raw_variables_type(const RawVariableType& new_type);

    void set_raw_variables_names(const Tensor<string, 1>&);

    void set_raw_variables_number(const Index&);

    void set_raw_variables_scalers(const Scaler&);

    void set_raw_variables_scalers(const Tensor<Scaler, 1>&);

    void set_binary_simple_raw_variables();

    // raw_variables other methods

    void check_constant_raw_variables();

    Tensor<type, 2> transform_binary_column(const Tensor<type, 1>&) const;

    // Variables set methods

    void set_variables_names(const Tensor<string, 1>&);
    void set_variables_names_from_raw_variables(const Tensor<string, 1>&, const Tensor<DataSet::RawVariable, 1>&);
    void set_variable_name(const Index&, const string&);

    void set_input();
    void set_target();
    void set_variables_unused();

    void set_input_variables_dimensions(const dimensions&);
    void set_target_dimensions(const dimensions&);

    // Data set methods

    void set_data(const Tensor<type, 2>&);
    void set_data(const Tensor<type, 1>&);
    void set_data(const Tensor<type, 2>&, const bool&);

    // Members set methods

    void set_data_source_path(const string&);

    void set_has_header(const bool&);
    void set_has_rows_label(const bool&);

    void set_has_text_data(const bool&);

    void set_separator(const Separator&);
    void set_separator(const string&);
    void set_separator(const char&);

    void set_codification(const Codification&);
    void set_codification(const string&);

    void set_missing_values_label(const string&);
    void set_missing_values_method(const MissingValuesMethod&);
    void set_missing_values_method(const string&);

    void set_gmt(Index&);

    void set_display(const bool&);

    // Check methods

    bool is_empty() const;

    bool is_sample_used(const Index&) const;
    bool is_sample_unused(const Index&) const;

    bool has_binary_raw_variables() const;
    bool has_categorical_raw_variables() const;

    //bool has_time_time_series_raw_variables() const;

    bool has_selection() const;

    // Splitting methods

    void split_samples_sequential(const type& training_ratio = type(0.6),
                                  const type& selection_ratio = type(0.2),
                                  const type& testing_ratio = type(0.2));

    void split_samples_random(const type& training_ratio = type(0.6),
                              const type& selection_ratio = type(0.2),
                              const type& testing_ratio = type(0.2));

    // Unusing methods

    Tensor<string, 1> unuse_constant_raw_variables();

    Tensor<Index, 1> unuse_repeated_samples();
    Tensor<string, 1> get_raw_variables_types() const;

    Tensor<string, 1> unuse_uncorrelated_raw_variables(const type& = type(0.25));
    Tensor<string, 1> unuse_multicollinear_raw_variables(Tensor<Index, 1>&, Tensor<Index, 1>&);

    // Initialization methods

    void set_data_constant(const type&);

    static type round_to_precision(type, const int&);
    static Tensor<type,2> round_to_precision_matrix(Tensor<type,2>, const int&);
    static Tensor<type, 1> round_to_precision_tensor(Tensor<type, 1> tensor, const int& precision);

    void set_data_random();
    void set_data_binary_random();

    // Descriptives methods

    Tensor<Descriptives, 1> calculate_variables_descriptives() const;
    Tensor<Descriptives, 1> calculate_used_variables_descriptives() const;

    Tensor<Descriptives, 1> calculate_raw_variables_descriptives_positive_samples() const;
    Tensor<Descriptives, 1> calculate_raw_variables_descriptives_negative_samples() const;
    Tensor<Descriptives, 1> calculate_raw_variables_descriptives_categories(const Index&) const;

    Tensor<Descriptives, 1> calculate_raw_variables_descriptives_training_samples() const;
    Tensor<Descriptives, 1> calculate_raw_variables_descriptives_selection_samples() const;

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

    Tensor<Histogram, 1> calculate_raw_variables_distribution(const Index& = 10) const;

    // Box and whiskers

    //BoxPlot calculate_single_box_plot(Tensor<type,1>&) const;
    Tensor<BoxPlot, 1> calculate_raw_variables_box_plots() const;
    Tensor<BoxPlot, 1> calculate_data_raw_variables_box_plot(Tensor<type,2>&) const;

    // Inputs correlations

    Tensor<Tensor<Correlation, 2>, 1> calculate_input_raw_variables_correlations(const bool& = true,
                                                                                 const bool& = false) const;

    void print_inputs_correlations() const;

    void print_top_inputs_correlations() const;

    // Inputs-targets correlations

    Tensor<Correlation, 2> calculate_input_target_raw_variables_correlations() const;
    Tensor<Correlation, 2> calculate_input_target_raw_variables_correlations_spearman() const;
    Tensor<Correlation, 2> calculate_relevant_input_target_raw_variables_correlations(const Tensor<Index, 1>&, const Tensor<Index, 1>&) const;

    void print_input_target_raw_variables_correlations() const;

    void print_top_input_target_raw_variables_correlations() const;

    // Filtering methods

    Tensor<Index, 1> filter_data(const Tensor<type, 1>&, const Tensor<type, 1>&);

    // Scaling methods

    void set_default_raw_variables_scalers();

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

    Tensor<Tensor<Index, 1>, 1> replace_Tukey_outliers_with_NaN(const type& cleaning_parameter);

    void unuse_Tukey_outliers(const type& = type(1.5));

    // Data generation

    void generate_constant_data(const Index&, const Index&, const type&);
    void generate_random_data(const Index&, const Index&);
    void generate_sequential_data(const Index&, const Index&);
    void generate_Rosenbrock_data(const Index&, const Index&);
    void generate_sum_data(const Index&, const Index&);
    void generate_classification_data(const Index&, const Index&, const Index&);

    // Serialization methods

    virtual void print() const;

    void from_XML(const tinyxml2::XMLDocument&);
    void write_XML(tinyxml2::XMLPrinter&) const;

    void save(const string&) const;
    void load(const string&);

    void print_raw_variables() const;
    void print_raw_variables_types() const;
    void print_raw_variables_uses() const;
    void print_raw_variables_scalers() const;

    void print_data() const;
    void print_data_preview() const;

    void print_data_file_preview() const;

    void save_data() const;

    void save_data_binary(const string&) const;

    void load_data_binary();

    // Data load methods

    virtual void load_data();

    // Missing values

    bool has_nan() const;

    bool has_nan_row(const Index&) const;

    void print_missing_values_information() const;

    void impute_missing_values_unuse();
    void impute_missing_values_mean();
    void impute_missing_values_median();
    void impute_missing_values_interpolate();

    void scrub_missing_values();

    Tensor<Index, 1> count_raw_variables_with_nan() const;
    Index count_rows_with_nan() const;
    Index count_nan() const;

    void set_missing_values_number(const Index&);
    void set_missing_values_number();

    void set_raw_variables_missing_values_number(const Tensor<Index, 1>&);
    void set_raw_variables_missing_values_number();

    void set_rows_missing_values_number(const Index&);
    void set_rows_missing_values_number();

    // Other methods

    void fix_repeated_names();

    // Eigen methods

    Tensor<Index, 2> split_samples(const Tensor<Index, 1>&, const Index&) const;

    bool get_has_rows_labels() const;
    bool get_has_text_data() const;

    void shuffle();

    // Reader

    string decode(const string&) const;

    void read_csv();
    void read_csv_1();

    void read_csv_2_simple();
    void read_csv_3_simple();

    void read_csv_2_complete();
    void read_csv_3_complete();

    void check_separators(const string&) const;

    void check_special_characters(const string&) const;

    void check_input_csv(const string&, const char&) const;

    Tensor<type, 2> read_input_csv(const string&, const string&, const string&, const bool&, const bool&) const;

    //Virtual functions

    //Image Models
    virtual void fill_image_data(const int&, const int&, const int&, const Tensor<type, 2>&);

    //Languaje Models
    virtual void read_txt_language_model();

    //AutoAssociation Models

    virtual void transform_associative_dataset();
    virtual void save_auto_associative_data_binary(const string&) const;

protected:

    DataSet::ModelType model_type;

    ThreadPool* thread_pool = nullptr;
    ThreadPoolDevice* thread_pool_device = nullptr;

    // DATA

    /// Data Matrix.
    /// The number of rows is the number of samples.
    /// The number of raw_variables is the number of variables.

    Tensor<type, 2> data;

    // Samples

    Tensor<SampleUse, 1> samples_uses;

    Tensor<string, 1> rows_labels;

    // Raw variables

    Tensor<RawVariable, 1> raw_variables;

    dimensions input_dimensions;

    dimensions target_dimensions;

    // DATA FILE

    /// Data file name.

    string data_source_path;

    /// Separator character.

    Separator separator = Separator::Comma;

    /// Missing values label.

    string missing_values_label = "NA";

    Tensor<bool, 1> nans_raw_variables;

    /// Header which contains variables name.

    bool has_header = false;

    /// Header which contains the rows label.

    bool has_rows_labels = false;

    /// Class containing file string codification

    Codification codification = Codification::UTF8;

    Tensor<Tensor<string, 1>, 1> data_file_preview;

    Index gmt = 0;

    // MISSING VALUES

    /// Missing values method.

    MissingValuesMethod missing_values_method = MissingValuesMethod::Unuse;

    /// Missing values

    Index missing_values_number;

    Tensor<Index, 1> raw_variables_missing_values_number;

    Index rows_missing_values_number;

    bool augmentation = false;

    /// Display messages to screen.        

    bool display = true;     
};

}

#define STRINGIFY_ENUM(x) #x

#define ENUM_TO_STRING(x) STRINGIFY_ENUM(x)

#endif

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2024 Artificial Intelligence Techniques, SL.
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
