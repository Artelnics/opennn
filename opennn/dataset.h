//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   D A T A   S E T   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#ifndef DATASET_H
#define DATASET_H

#include "tinyxml2.h"
#include "correlations.h"
#include "scaling.h"
#include "tensors.h"

using namespace tinyxml2;

namespace opennn
{

class Dataset
{

public:

    enum class Codification { UTF8, SHIFT_JIS };

    Dataset(const Index& = 0,
            const dimensions& = {0},
            const dimensions& = {0});

    Dataset(const filesystem::path&,
            const string&,
            const bool& = true,
            const bool& = false,
            const Codification& = Codification::UTF8);

    // Enumerations

    enum class Separator{None, Space, Tab, Comma, Semicolon};

    enum class MissingValuesMethod{Unuse, Mean, Median, Interpolation};

    enum class RawVariableType{None, Numeric, Binary, Categorical, DateTime, Constant};

    // Structs

    struct RawVariable
    {
        RawVariable(const string& = string(),
                    const string& = "None",
                    const Dataset::RawVariableType& = Dataset::RawVariableType::Numeric,
                    const Scaler& = Scaler::MeanStandardDeviation,
                    const vector<string>& = vector<string>());

        void set(const string& = string(),
                 const string& = "None",
                 const Dataset::RawVariableType& = Dataset::RawVariableType::Numeric,
                 const Scaler& = Scaler::MeanStandardDeviation,
                 const vector<string>& = vector<string>());

        string name;

        string use = "None";

        Dataset::RawVariableType type = Dataset::RawVariableType::None;

        vector<string> categories;

        Scaler scaler = Scaler::None;

        // Methods

        string get_use() const;
        string get_type_string() const;

        Index get_categories_number() const;

        void set_scaler(const Scaler&);
        void set_scaler(const string&);

        void set_use(const string&);

        void set_type(const string&);

        void set_categories(const vector<string>&);

        virtual void from_XML(const XMLDocument&);
        virtual void to_XML(XMLPrinter&) const;

        void print() const;
    };

    // Samples get

    inline Index get_samples_number() const {return data.dimension(0);}

    Index get_samples_number(const string&) const;

    Index get_used_samples_number() const;

    vector<Index> get_sample_indices(const string&) const;

    vector<Index> get_used_sample_indices() const;

    string get_sample_use(const Index&) const;
    const vector<string>& get_sample_uses() const;

    vector<Index> get_sample_uses_vector() const;

    Tensor<Index, 1> get_sample_use_numbers() const;

    inline Index get_raw_variables_number() const { return raw_variables.size(); }
    Index get_raw_variables_number(const string&) const;
    Index get_used_raw_variables_number() const;

    const vector<RawVariable>& get_raw_variables() const;
    vector<RawVariable> get_raw_variables(const string&) const;

    Index get_raw_variable_index(const string&) const;
    Index get_raw_variable_index(const Index&) const;

    vector<Index> get_raw_variable_indices(const string&) const;
    vector<Index> get_used_raw_variables_indices() const;

    vector<string> get_raw_variable_names() const;
    vector<string> get_raw_variable_names(const string&) const;

    RawVariableType get_raw_variable_type(const Index& index) const {return raw_variables[index].type;}

    // Variables get

    Index get_variables_number() const;
    Index get_variables_number(const string&) const;
    Index get_used_variables_number() const;

    vector<string> get_variable_names() const;
    vector<string> get_variable_names(const string&) const;

    vector<vector<Index>> get_variable_indices() const;
    vector<Index> get_variable_indices(const Index&) const;
    vector<Index> get_variable_indices(const string&) const;
    vector<Index> get_used_variable_indices() const;

    dimensions get_dimensions(const string&) const;

    vector<Scaler> get_variable_scalers(const string&) const;

    virtual vector<vector<Index>> get_batches(const vector<Index>&, const Index&, const bool&) const;

    const Tensor<type, 2>& get_data() const;
    Tensor<type, 2>* get_data_p();
    Tensor<type, 2> get_data_samples(const string&) const;
    Tensor<type, 2> get_data_variables(const string&) const;
    Tensor<type, 2> get_data(const string&, const string&) const;
    Tensor<type, 2> get_data_from_indices(const vector<Index>&, const vector<Index>&) const;

    Tensor<type, 1> get_sample_data(const Index&) const;
    Tensor<type, 1> get_sample_data(const Index&, const vector<Index>&) const;
    Tensor<type, 2> get_sample_input_data(const Index&) const;
    Tensor<type, 2> get_sample_target_data(const Index&) const;

    Tensor<type, 2> get_raw_variable_data(const Index&) const;
    Tensor<type, 2> get_raw_variable_data(const Index&, const vector<Index>&) const;
    //Tensor<type, 2> get_raw_variable_data(const Tensor<Index, 1>&) const;
    Tensor<type, 2> get_raw_variable_data(const string&) const;

    string get_sample_category(const Index&, const Index&) const;
    Tensor<type, 1> get_sample(const Index&) const;

    const vector<vector<string>>& get_data_file_preview() const;

    const vector<string>& get_positive_words() const { return positive_words; }
    const vector<string>& get_negative_words() const { return negative_words; }

    // Members get

    MissingValuesMethod get_missing_values_method() const;
    string get_missing_values_method_string() const;

    const filesystem::path& get_data_path() const;

    const bool& get_header_line() const;
    const bool& get_has_sample_ids() const;

    vector<string> get_sample_ids() const;

    const Separator& get_separator() const;
    string get_separator_string() const;
    string get_separator_name() const;

    const Codification& get_codification() const;
    const string get_codification_string() const;

    const string& get_missing_values_label() const;

    Index get_gmt() const;

    const bool& get_display() const;

    bool is_empty() const;

    dimensions get_input_dimensions() const;
    dimensions get_target_dimensions() const;

    // Set

    void set(const Index& = 0, const dimensions& = {}, const dimensions& = {});

    void set(const filesystem::path&,
             const string&,
             const bool& = true,
             const bool& = false,
             const Dataset::Codification& = Codification::UTF8);

    void set(const filesystem::path&);

    void set_default();

    void set_threads_number(const int&);

    // Samples set

    void set_sample_uses(const string&);

    void set_sample_use(const Index&, const string&);

    void set_sample_uses(const vector<string>&);
    void set_sample_uses(const vector<Index>&, const string&);

    // Raw variables set

    void set_raw_variables(const vector<RawVariable>&);

    void set_default_raw_variable_names();

    void set_default_raw_variables_uses();
    void set_default_raw_variables_uses_forecasting();
    virtual void set_raw_variable_uses(const vector<string>&);

    void set_raw_variables(const string&);
    void set_raw_variable_indices(const vector<Index>&, const vector<Index>&);
    void set_input_raw_variables_unused();

    void set_raw_variable_use(const Index&, const string&);
    void set_raw_variable_use(const string&, const string&);

    void set_raw_variable_type(const Index&, const RawVariableType&);
    void set_raw_variable_type(const string&, const RawVariableType&);

    void set_raw_variable_types(const RawVariableType&);

    void set_raw_variable_names(const vector<string>&);

    void set_raw_variables_number(const Index&);

    void set_raw_variable_scalers(const Scaler&);

    void set_raw_variable_scalers(const vector<Scaler>&);

    void set_binary_raw_variables();
    void unuse_constant_raw_variables();

    // Variables set

    void set_variable_names(const vector<string>&);

    void set_variable_uses(const string&);

    void set_dimensions(const string&, const dimensions&);

    // Data set

    void set_data(const Tensor<type, 2>&);

    // Members set

    void set_data_path(const filesystem::path&);

    void set_has_header(const bool&);
    void set_has_ids(const bool&);

    void set_separator(const Separator&);
    void set_separator_string(const string&);
    void set_separator_name(const string&);

    void set_codification(const Codification&);
    void set_codification(const string&);

    void set_missing_values_label(const string&);
    void set_missing_values_method(const MissingValuesMethod&);
    void set_missing_values_method(const string&);

    void set_gmt(const Index&);

    void set_display(const bool&);

    bool is_sample_used(const Index&) const;

    bool has_binary_raw_variables() const;
    bool has_categorical_raw_variables() const;
    bool has_binary_or_categorical_raw_variables() const;
    bool has_time_raw_variable() const;

    bool has_selection() const;

    bool has_missing_values(const vector<string>&) const;

    // Splitting

    void split_samples_sequential(const type& training_ratio = type(0.6),
                                  const type& selection_ratio = type(0.2),
                                  const type& testing_ratio = type(0.2));

    void split_samples_random(const type& training_ratio = type(0.6),
                              const type& selection_ratio = type(0.2),
                              const type& testing_ratio = type(0.2));

    // Unusing

    //Tensor<Index, 1> unuse_repeated_samples();

    vector<string> unuse_uncorrelated_raw_variables(const type& = type(0.25));
    vector<string> unuse_collinear_raw_variables(const type& = type(0.95));

    // Initialization

    void set_data_constant(const type&);

    // Descriptives

    vector<Descriptives> calculate_variable_descriptives() const;
    vector<Descriptives> calculate_used_variable_descriptives() const;

    vector<Descriptives> calculate_raw_variable_descriptives_positive_samples() const;
    vector<Descriptives> calculate_raw_variable_descriptives_negative_samples() const;
    vector<Descriptives> calculate_raw_variable_descriptives_categories(const Index&) const;

    vector<Descriptives> calculate_variable_descriptives(const string&) const;

    vector<Descriptives> calculate_testing_target_variable_descriptives() const;

    //Tensor<type, 1> calculate_used_variables_minimums() const;

    Tensor<type, 1> calculate_means(const string& , const string&) const;

    Index calculate_used_negatives(const Index&);
    Index calculate_negatives(const Index&, const string&) const;

    // Distribution

    vector<Histogram> calculate_raw_variable_distributions(const Index& = 10) const;

    // Box and whiskers

    vector<BoxPlot> calculate_raw_variables_box_plots() const;
    //Tensor<BoxPlot, 1> calculate_data_raw_variables_box_plot(Tensor<type,2>&) const;

    // Inputs correlations

    Tensor<Correlation, 2> calculate_input_raw_variable_pearson_correlations() const;

    Tensor<Correlation, 2> calculate_input_raw_variable_spearman_correlations() const;

    void print_inputs_correlations() const;

    void print_top_inputs_correlations() const;

    // Inputs-targets correlations

    Tensor<Correlation, 2> calculate_input_target_raw_variable_pearson_correlations() const;
    Tensor<Correlation, 2> calculate_input_target_raw_variable_spearman_correlations() const;

    void print_input_target_raw_variables_correlations() const;

    void print_top_input_target_raw_variables_correlations() const;

    // Filtering

    Tensor<Index, 1> filter_data(const Tensor<type, 1>&, const Tensor<type, 1>&);

    // Scaling2d

    void set_default_raw_variables_scalers();

    // Data scaling

    vector<Descriptives> scale_data();

    virtual vector<Descriptives> scale_variables(const string&);

    // Data unscaling

    void unscale_variables(const string&, const vector<Descriptives>&);

    // Classification

    Tensor<Index, 1> calculate_target_distribution() const;

    // Tuckey outlier detection

    vector<vector<Index>> calculate_Tukey_outliers(const type& = type(1.5)) const;

    vector<vector<Index>> replace_Tukey_outliers_with_NaN(const type& = type(1.5));

    void unuse_Tukey_outliers(const type& = type(1.5));

    // Data generation

    virtual void set_data_random();
    void set_data_rosenbrock();
    void set_data_binary_classification();
    void set_data_ascending();

    // Serialization

    virtual void print() const;

    virtual void from_XML(const XMLDocument&);
    virtual void to_XML(XMLPrinter&) const;

    void save(const filesystem::path&) const;
    void load(const filesystem::path&);

    void print_raw_variables() const;

    void print_data() const;
    void print_data_preview() const;

    void print_data_file_preview() const;

    void save_data() const;

    void save_data_binary(const filesystem::path&) const;

    void load_data_binary();

    // Missing values

    inline Index get_missing_values_number() const { return missing_values_number; }

    bool has_nan() const;

    bool has_nan_row(const Index&) const;

    void print_missing_values_information() const;

    virtual void impute_missing_values_unuse();
    void impute_missing_values_mean();
    void impute_missing_values_median();
    virtual void impute_missing_values_interpolate();

    void scrub_missing_values();

    Tensor<Index, 1> count_nans_per_raw_variable() const;
    Index count_raw_variables_with_nan() const;
    Index count_rows_with_nan() const;
    Index count_nan() const;

    // Other

    void fix_repeated_names();

    // Eigen

    vector<vector<Index>> split_samples(const vector<Index>&, const Index&) const;

    bool get_has_rows_labels() const;
    //bool get_has_text_data() const;

    // Reader

    void decode(string&) const;

    virtual void read_csv();

    void prepare_line(string&) const;
    void infer_column_types(const vector<vector<string>>&);

    void read_data_file_preview(const vector<vector<string>>&);

    void check_separators(const string&) const;

    virtual void fill_input_tensor(const vector<Index>&,
                                   const vector<Index>&,
                                   type*) const;

    virtual void fill_input_tensor_row_major(const vector<Index>&,
                                             const vector<Index>&,
                                             type*) const;

    virtual void fill_target_tensor(const vector<Index>&,
                                    const vector<Index>&,
                                    type*) const;

    // virtual void fill_decoder_tensor(const vector<Index>&,
    //                                  const vector<Index>&,
    //                                  type*) const;


protected:

    unique_ptr<ThreadPool> thread_pool = nullptr;
    unique_ptr<ThreadPoolDevice> thread_pool_device = nullptr;

    // DATA

    Tensor<type, 2> data;

    // Dimensions

    dimensions input_dimensions;
    dimensions target_dimensions;
    dimensions decoder_dimensions;

    // Samples

    vector<string> sample_uses;

    vector<string> sample_ids;

    // Raw variables

    vector<RawVariable> raw_variables;

    // Data File

    filesystem::path data_path;

    Separator separator = Separator::Comma;

    string missing_values_label = "NA";

    Tensor<bool, 1> nans_raw_variables;

    bool has_header = false;

    bool has_sample_ids = false;

    Codification codification = Codification::UTF8;

    vector<vector<string>> data_file_preview;

    Index gmt = 0;

    // Missing Values

    MissingValuesMethod missing_values_method = MissingValuesMethod::Mean;

    Index missing_values_number = 0;

    Tensor<Index, 1> raw_variables_missing_values_number;

    Index rows_missing_values_number = 0;

    // Display

    bool display = true;

    const vector<string> positive_words = {"1", "yes", "positive", "+", "true", "good", "si", "sí", "Sí"};
    const vector<string> negative_words = {"0", "no", "negative", "-", "false", "bad", "not", "No"};
};


struct Batch
{
    Batch(const Index& = 0, const Dataset* = nullptr);

    vector<TensorView> get_input_pairs() const;
    TensorView get_target_pair() const;

    Index get_samples_number() const;

    void set(const Index& = 0, const Dataset* = nullptr);

    void fill(const vector<Index>&,
              const vector<Index>&,
              // const vector<Index>&,
              const vector<Index>& = vector<Index>());

    Tensor<type, 2> perform_augmentation(const Tensor<type, 2>&);

    void print() const;

    bool is_empty() const;

    Index samples_number = 0;

    Dataset* dataset = nullptr;

    dimensions input_dimensions;
    Tensor<type, 1> input_tensor;

    dimensions decoder_dimensions;
    Tensor<type, 1> decoder_tensor;

    dimensions target_dimensions;
    Tensor<type, 1> target_tensor;

    unique_ptr<ThreadPool> thread_pool = nullptr;
    unique_ptr<ThreadPoolDevice> thread_pool_device = nullptr;
};

#ifdef OPENNN_CUDA

struct BatchCuda
{
    BatchCuda(const Index& = 0, Dataset* = nullptr);

    ~BatchCuda() { free(); }

    BatchCuda(const BatchCuda&) = delete;
    BatchCuda& operator=(const BatchCuda&) = delete;

    vector<float*> get_input_device() const;
    TensorView get_target_pair_device() const;

    Index get_samples_number() const;

    Tensor<type, 2> get_inputs_device() const;
    Tensor<type, 2> get_decoder_device() const;
    Tensor<type, 2> get_targets_device() const;

    void set(const Index&, Dataset*);

    void copy_device(const Index&);

    void free();

    void fill(const vector<Index>&,
              const vector<Index>&,
              //const vector<Index>&,
              const vector<Index> & = vector<Index>());

    void print() const;

    bool is_empty() const;

    Index samples_number = 0;
    Index num_input_features = 0;
    Index num_target_features = 0;

    Dataset* dataset = nullptr;

    dimensions input_dimensions;
    dimensions decoder_dimensions;
    dimensions target_dimensions;

    float* inputs_host = nullptr;
    float* decoder_host = nullptr;
    float* targets_host = nullptr;

    float* inputs_device = nullptr;
    float* decoder_device = nullptr;
    float* targets_device = nullptr;
};

#endif

}

#define STRINGIFY_ENUM(x) #x

#define ENUM_TO_STRING(x) STRINGIFY_ENUM(x)

#endif

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2025 Artificial Intelligence Techniques, SL.
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
