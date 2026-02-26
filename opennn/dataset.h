//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   D A T A   S E T   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#pragma once

#include "correlations.h"
#include "statistics.h"
#include "tensors.h"
#include "strings_utilities.h"


namespace opennn
{

class Dataset
{

public:

    enum class Codification { UTF8, SHIFT_JIS };

    Dataset(const Index = 0,
            const Shape& = {0},
            const Shape& = {0});

    Dataset(const filesystem::path&,
            const string&,
            bool = true,
            bool = false,
            const Codification& = Codification::UTF8);

    // Enumerations

    enum class Separator{None, Space, Tab, Comma, Semicolon};

    enum class MissingValuesMethod{Unuse, Mean, Median, Interpolation};

    enum class VariableType{None, Numeric, Binary, Categorical, DateTime, Constant};

    // Structs

    struct Variable
    {
        Variable(const string& = string(),
                    const string& = "None",
                    const Dataset::VariableType& = Dataset::VariableType::Numeric,
                    const string& = "MeanStandardDeviation",
                    const vector<string>& = vector<string>());

        void set(const string& = string(),
                 const string& = "None",
                 const Dataset::VariableType& = Dataset::VariableType::Numeric,
                 const string& = "MeanStandardDeviation",
                 const vector<string>& = vector<string>());

        string name;

        string role = "None";

        Dataset::VariableType type = Dataset::VariableType::None;

        vector<string> categories;

        string scaler = "None";

        // Methods

        string get_role() const;
        string get_type_string() const;

        Index get_categories_number() const;

        void set_scaler(const string&);

        void set_role(const string&);

        void set_type(const string&);

        void set_categories(const vector<string>&);

        virtual void from_XML(const XMLDocument&);

        virtual void to_XML(XMLPrinter&) const;

        bool is_binary() const
        {
            if(type == Dataset::VariableType::Binary)
                return true;
            else
                return false;
        }

        bool is_used() const
        {
            if(role == "None" || role == "Time")
                return false;

            return true;
        }


        bool is_categorical() const
        {
            if(type == Dataset::VariableType::Categorical)
                return true;
            else
                return false;
        }

        void print() const;
    };

    // Samples get

    inline Index get_samples_number() const {return data.rows();}

    Index get_samples_number(const string&) const;

    Index get_used_samples_number() const;

    vector<Index> get_sample_indices(const string&) const;

    vector<Index> get_used_sample_indices() const;

    string get_sample_role(const Index) const;
    const vector<string>& get_sample_roles() const;

    vector<Index> get_sample_roles_vector() const;

    VectorI get_sample_role_numbers() const;

    inline Index get_variables_number() const { return variables.size(); }
    Index get_variables_number(const string&) const;
    Index get_used_variables_number() const;

    const vector<Variable>& get_variables() const;
    vector<Variable> get_variables(const string&) const;

    Index get_variable_index(const string&) const;
    Index get_variable_index(const Index) const;

    vector<Index> get_variable_indices(const string&) const;
    vector<Index> get_used_variables_indices() const;

    vector<string> get_variable_names() const;
    vector<string> get_variable_names(const string&) const;

    VariableType get_variable_type(const Index index) const
    {
        return variables[index].type;
    }
    vector<VariableType> get_variable_types(const vector<Index> indices) const;

    // Variables get

    Index get_features_number() const;
    Index get_features_number(const string&) const;
    Index get_used_features_number() const;

    vector<string> get_feature_names() const;
    vector<string> get_feature_names(const string&) const;

    vector<vector<Index>> get_feature_indices() const;
    vector<Index> get_feature_indices(const Index) const;
    vector<Index> get_feature_indices(const string&) const;
    vector<Index> get_used_feature_indices() const;

    vector<Index> get_feature_dimensions() const;
//@simone forse input e aoutput

    Shape get_shape(const string&) const;

    vector<string> get_feature_scalers(const string&) const;

    virtual vector<vector<Index>> get_batches(const vector<Index>&, Index, bool) const;

    const MatrixR& get_data() const;
    MatrixR get_data_samples(const string&) const;
    MatrixR get_feature_data(const string&) const;
    MatrixR get_data(const string&, const string&) const;
    MatrixR get_data_from_indices(const vector<Index>&, const vector<Index>&) const;

    VectorR get_sample_data(const Index) const;
    VectorR get_sample_data(const Index, const vector<Index>&) const;
    MatrixR get_sample_input_data(const Index) const;
    MatrixR get_sample_target_data(const Index) const;

    MatrixR get_variable_data(const Index) const;
    MatrixR get_variable_data(const Index, const vector<Index>&) const;
    //Tensor2 get_variable_data(const VectorI&) const;
    MatrixR get_variable_data(const string&) const;

    string get_sample_category(const Index, Index) const;
    VectorR get_sample(const Index) const;

    const vector<vector<string>>& get_data_file_preview() const;

    const vector<string>& get_positive_words() const { return positive_words; }
    const vector<string>& get_negative_words() const { return negative_words; }

    // Members get

    MissingValuesMethod get_missing_values_method() const;
    string get_missing_values_method_string() const;

    const filesystem::path& get_data_path() const;

    bool get_header_line() const;
    bool get_has_sample_ids() const;

    vector<string> get_sample_ids() const;

    const Separator& get_separator() const;
    string get_separator_string() const;
    string get_separator_name() const;

    const Codification& get_codification() const;
    const string get_codification_string() const;

    const string& get_missing_values_label() const;

    Index get_gmt() const;

    bool get_display() const;

    bool is_empty() const;

    Shape get_input_shape() const;
    Shape get_target_shape() const;

    void get_categorical_info(const string&, vector<Index>&, vector<Index>&) const;

    // Set

    void set(const Index = 0, const Shape& = {}, const Shape& = {});

    void set(const filesystem::path&,
             const string&,
             bool = true,
             bool = false,
             const Dataset::Codification& = Codification::UTF8);

    void set(const filesystem::path&);

    void set_default();

    // Samples set

    void set_sample_roles(const string&);

    void set_sample_role(const Index, const string&);

    void set_sample_roles(const vector<string>&);
    void set_sample_roles(const vector<Index>&, const string&);

    // Variables set

    void set_variables(const vector<Variable>&);

    void set_default_variable_names();

    void set_default_variables_roles();
    void set_default_variables_roles_forecasting();
    virtual void set_variable_roles(const vector<string>&);

    void set_variables(const string&);
    void set_variable_indices(const vector<Index>&, const vector<Index>&);
    void set_input_variables_unused();

    void set_variable_role(const Index, const string&);
    void set_variable_role(const string&, const string&);

    void set_variable_type(const Index, const VariableType&);
    void set_variable_type(const string&, const VariableType&);

    void set_variable_types(const VariableType&);

    void set_variable_names(const vector<string>&);

    void set_variables_number(const Index);

    void set_variable_scalers(const string&);

    void set_variable_scalers(const vector<string>&);

    void set_binary_variables();
    void unuse_constant_variables();

    // Variables set

    void set_feature_names(const vector<string>&);

    void set_feature_roles(const string&);

    void set_shape(const string&, const Shape&);

    // Dataset

    void set_data(const MatrixR&);

    // Members set

    void set_data_path(const filesystem::path&);

    void set_has_header(bool);
    void set_has_ids(bool);

    void set_separator(const Separator&);
    void set_separator_string(const string&);
    void set_separator_name(const string&);

    void set_codification(const Codification&);
    void set_codification(const string&);

    void set_missing_values_label(const string&);
    void set_missing_values_method(const MissingValuesMethod&);
    void set_missing_values_method(const string&);

    void set_gmt(const Index);

    void set_display(bool);

    bool is_sample_used(const Index) const;

    bool has_binary_variables() const;
    bool has_categorical_variables() const;
    bool has_binary_or_categorical_variables() const;
    bool has_time_variable() const;

    bool has_validation() const;

    bool has_missing_values(const vector<string>&) const;

    // Splitting

    void split_samples_sequential(const type training_ratio = type(0.6),
                                  type selection_ratio = type(0.2),
                                  type testing_ratio = type(0.2));

    void split_samples_random(const type training_ratio = type(0.6),
                              type selection_ratio = type(0.2),
                              type testing_ratio = type(0.2));

    // Unusing

    //VectorI unuse_repeated_samples();

    vector<string> unuse_uncorrelated_variables(const type = type(0.25));
    vector<string> unuse_collinear_variables(const type = type(0.95));

    // Initialization

    void set_data_constant(const type);

    // Descriptives

    vector<Descriptives> calculate_feature_descriptives() const;
    //vector<Descriptives> calculate_used_variable_descriptives() const;

    vector<Descriptives> calculate_variable_descriptives_positive_samples() const;
    vector<Descriptives> calculate_variable_descriptives_negative_samples() const;
    vector<Descriptives> calculate_variable_descriptives_categories(const Index) const;

    vector<Descriptives> calculate_feature_descriptives(const string&) const;

    vector<Descriptives> calculate_testing_target_variable_descriptives() const;

    //VectorR calculate_used_variables_minimums() const;

    VectorR calculate_means(const string& , const string&) const;

    Index calculate_used_negatives(const Index) const;
    Index calculate_negatives(const Index, const string&) const;

    // Distribution

    vector<Histogram> calculate_variable_distributions(const Index = 10) const;

    // Box plots

    vector<BoxPlot> calculate_variables_box_plots() const;

    // Inputs correlations

    Tensor<Correlation, 2> calculate_input_variable_pearson_correlations() const;

    Tensor<Correlation, 2> calculate_input_variable_spearman_correlations() const;

    void print_inputs_correlations() const;

    void print_top_inputs_correlations() const;

    // Input-target correlations

    Tensor<Correlation, 2> calculate_input_target_variable_pearson_correlations() const;
    Tensor<Correlation, 2> calculate_input_target_variable_spearman_correlations() const;

    VectorI calculate_correlations_rank() const;

    void print_input_target_variables_correlations() const;

    void print_top_input_target_variables_correlations() const;

    // Filtering

    VectorI filter_data(const VectorR&, const VectorR&);

    // Scaling2d

    void set_default_variables_scalers();

    // Data scaling

    vector<Descriptives> scale_data();

    virtual vector<Descriptives> scale_features(const string&);

    // Data unscaling

    void unscale_features(const string&, const vector<Descriptives>&);

    // Classification

    VectorI calculate_target_distribution() const;

    // Tuckey outlier detection

    vector<vector<Index>> calculate_Tukey_outliers(const type = type(1.5)) const;

    vector<vector<Index>> replace_Tukey_outliers_with_NaN(const type = type(1.5));

    void unuse_Tukey_outliers(const type = type(1.5));

    // Data generation

    virtual void set_data_random();
    virtual void set_data_integer(const Index vocabulary_size);
    void set_data_rosenbrock();
    void set_data_binary_classification();

    // Serialization

    virtual void print() const;

    virtual void from_XML(const XMLDocument&);
    virtual void to_XML(XMLPrinter&) const;


    void save(const filesystem::path&) const;
    void load(const filesystem::path&);

    void print_variables() const;

    void print_data() const;
    void print_data_preview() const;

    void print_data_file_preview() const;

    void save_data() const;

    void save_data_binary(const filesystem::path&) const;

    void load_data_binary();

    // Missing values

    inline Index get_missing_values_number() const { return missing_values_number; }

    bool has_nan() const;

    bool has_nan_row(const Index) const;

    void print_missing_values_information() const;

    virtual void impute_missing_values_unuse();
    void impute_missing_values_mean();
    void impute_missing_values_median();
    virtual void impute_missing_values_interpolate();

    void scrub_missing_values();
    void calculate_missing_values_statistics();

    VectorI count_nans_per_variable() const;
    Index count_variables_with_nan() const;
    Index count_rows_with_nan() const;
    Index count_nan() const;

    // Other

    void fix_repeated_names();

    // Eigen

    vector<vector<Index>> split_samples(const vector<Index>&, Index) const;

    bool get_has_rows_labels() const;
    //bool get_has_text_data() const;

    // Reader

    //void decode(string&) const;

    virtual void read_csv();

    void infer_column_types(const vector<vector<string>>&);
    DateFormat infer_dataset_date_format(const vector<Dataset::Variable>&, const vector<vector<string>>&, bool, const string&);

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

    // DATA

    MatrixR data;

    // Dimensions

    Shape input_shape;
    Shape target_shape;
    Shape decoder_shape;

    // Samples

    vector<string> sample_roles;

    vector<string> sample_ids;

    // Variables

    vector<Variable> variables;

    // Data File

    filesystem::path data_path;

    Separator separator = Separator::Comma;

    string missing_values_label = "NA";

    VectorB nans_variables;

    bool has_header = false;

    bool has_sample_ids = false;

    Codification codification = Codification::UTF8;

    vector<vector<string>> data_file_preview;

    Index gmt = 0;

    // Missing Values

    MissingValuesMethod missing_values_method = MissingValuesMethod::Mean;

    Index missing_values_number = 0;

    VectorI variables_missing_values_number;

    Index rows_missing_values_number = 0;

    // Display

    bool display = true;

    const vector<string> positive_words = {"1", "yes", "positive", "+", "true", "good", "si", "sí", "Sí"};
    const vector<string> negative_words = {"0", "no", "negative", "-", "false", "bad", "not", "No"};

    void variables_to_XML(XMLPrinter&) const;
    void samples_to_XML(XMLPrinter&) const;
    void missing_values_to_XML(XMLPrinter&) const;
    void preview_data_to_XML(XMLPrinter&) const;

    void variables_from_XML(const XMLElement*);
    void samples_from_XML(const XMLElement*);
    void missing_values_from_XML(const XMLElement*);
    void preview_data_from_XML(const XMLElement*);

};


struct Batch
{
    Batch(const Index = 0, const Dataset* = nullptr);

    void set(const Index = 0, const Dataset* = nullptr);

    void fill(const vector<Index>&,
              const vector<Index>&,
              // const vector<Index>&,
              const vector<Index>& = vector<Index>());

    vector<TensorView> get_inputs() const;
    TensorView get_targets() const;

    Index get_samples_number() const;

    Tensor2 perform_augmentation(const Tensor2&);

    void print() const;

    bool is_empty() const;

    Index samples_number = 0;

    Dataset* dataset = nullptr;

    Shape input_shape;
    VectorR input_tensor;

    Shape decoder_shape;
    VectorR decoder_tensor;

    Shape target_shape;
    VectorR target_tensor;
};


#ifdef OPENNN_CUDA

struct BatchCuda
{
    BatchCuda(const Index = 0, Dataset* = nullptr);
    ~BatchCuda();

    void set(const Index, Dataset*);

    void fill(const vector<Index>&,
              const vector<Index>&,
              //const vector<Index>&,
              const vector<Index>& = vector<Index>());

    void fill_host(const vector<Index>&,
                   const vector<Index>&,
                   //const vector<Index>&,
                   const vector<Index>& = vector<Index>());

    vector<TensorViewCuda> get_inputs_device() const;
    TensorViewCuda get_targets_device() const;

    Index get_samples_number() const;

    Tensor2 get_inputs_from_device() const;
    Tensor2 get_decoder_from_device() const;
    Tensor2 get_targets_from_device() const;

    void copy_device(const Index);
    void copy_device_async(const Index, cudaStream_t);

    void print() const;

    bool is_empty() const;

    Index samples_number = 0;
    Index num_input_features = 0;
    Index num_target_features = 0;

    Dataset* dataset = nullptr;

    Shape input_shape;
    Shape decoder_shape;
    Shape target_shape;

    float* inputs_host = nullptr;
    float* decoder_host = nullptr;
    float* targets_host = nullptr;

    Index inputs_host_allocated_size = 0;
    Index decoder_host_allocated_size = 0;
    Index targets_host_allocated_size = 0;

    TensorCuda inputs_device;
    TensorCuda decoder_device;
    TensorCuda targets_device;
};


#endif

}

#define STRINGIFY_ENUM(x) #x

#define ENUM_TO_STRING(x) STRINGIFY_ENUM(x)

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// This library is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 2.1 of the License, or any later version.
// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.
// You should have received a copy of the GNU Lesser General Public
// License along with this library; if not, write to the Free Software
// Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
