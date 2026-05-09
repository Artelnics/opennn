//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   M A T E R I A L I Z E D   D A T A S E T   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "materialized_dataset.h"
#include "scaling.h"
#include "random_utilities.h"
#include <regex>

namespace opennn
{

MaterializedDataset::MaterializedDataset(const Index new_samples_number,
                                         const Shape& new_input_shape,
                                         const Shape& new_target_shape)
{
    set(new_samples_number, new_input_shape, new_target_shape);
}

MaterializedDataset::MaterializedDataset(const filesystem::path& data_path,
                                         const string& separator,
                                         bool has_header,
                                         bool has_sample_ids,
                                         const Codification& data_codification)
{
    set(data_path, separator, has_header, has_sample_ids, data_codification);
}

void MaterializedDataset::set(const filesystem::path& new_data_path,
                              const string& new_separator,
                              bool new_has_header,
                              bool new_has_ids,
                              const Codification& new_codification)
{
    set_default();

    set_data_path(new_data_path);

    set_separator_string(new_separator);

    set_has_header(new_has_header);

    set_has_ids(new_has_ids);

    set_codification(new_codification);

    read_csv();

    set_default_variable_scalers();

    set_default_variable_roles();

    missing_values_method = MissingValuesMethod::Unuse;

    input_shape = { get_features_number("Input") };
    target_shape = { get_features_number("Target") };
}

void MaterializedDataset::set(const Index new_samples_number,
                              const Shape& new_input_shape,
                              const Shape& new_target_shape)
{
    if (new_samples_number == 0
        || new_input_shape.empty()
        || new_target_shape.empty())
        return;

    input_shape = new_input_shape;

    const Index new_inputs_number = new_input_shape.size();

    Index new_targets_number = new_target_shape.size();

    new_targets_number = (new_targets_number == 2) ? 1 : new_targets_number;

    target_shape = { new_targets_number };

    const Index new_features_number = new_inputs_number + new_targets_number;

    data.resize(new_samples_number, new_features_number);

    variables.resize(new_features_number);

    set_default();

    for (Index i = 0; i < new_features_number; ++i)
    {
        Variable& variable = variables[i];

        variable.type = VariableType::Numeric;
        variable.name = "variable_" + to_string(i + 1);

        variable.role = (i < new_inputs_number)
                               ? VariableRole::Input
                               : VariableRole::Target;
    }

    sample_roles.resize(new_samples_number, SampleRole::Training);

    split_samples_random();
}

void MaterializedDataset::set(const filesystem::path& file_name)
{
    load(file_name);
}

void MaterializedDataset::set_data(const MatrixR& new_data)
{
    if (new_data.rows() != get_samples_number())
        throw runtime_error("Rows number is not equal to samples number");

    if (new_data.cols() != get_features_number())
        throw runtime_error("Columns number is not equal to variables number");

    data = new_data;
}

MatrixR MaterializedDataset::get_feature_data(const string& variable_role) const
{
    const Index samples_number = get_samples_number();

    vector<Index> indices(samples_number);
    iota(indices.begin(), indices.end(), 0);

    return get_data_from_indices(indices, get_feature_indices(variable_role));
}

MatrixR MaterializedDataset::get_data(const string& sample_role, const string& variable_role) const
{
    return get_data_from_indices(get_sample_indices(sample_role), get_feature_indices(variable_role));
}

MatrixR MaterializedDataset::get_data_from_indices(const vector<Index>& sample_indices, const vector<Index>& feature_indices) const
{
    MatrixR this_data(sample_indices.size(), feature_indices.size());

    fill_tensor_data(data, sample_indices, feature_indices, this_data.data());

    return this_data;
}

VectorR MaterializedDataset::get_sample_data(const Index index) const
{
    return data.row(index);
}

MatrixR MaterializedDataset::get_variable_data(const Index variable_index) const
{
    Index variables_number = 1;
    const Index rows_number = data.rows();

    if (variables[variable_index].type == VariableType::Categorical)
        variables_number = variables[variable_index].get_categories_number();

    const Index start_column = get_feature_indices(variable_index)[0];

    return data.block(0, start_column, rows_number, variables_number);
}

MatrixR MaterializedDataset::get_variable_data(const Index variable_index, const vector<Index>& row_indices) const
{
    MatrixR variable_data(row_indices.size(), get_feature_indices(variable_index).size());

    fill_tensor_data(data, row_indices, get_feature_indices(variable_index), variable_data.data());

    return variable_data;
}

MatrixR MaterializedDataset::get_variable_data(const string& column_name) const
{
    const Index variable_index = get_variable_index(column_name);

    return get_variable_data(variable_index);
}

void MaterializedDataset::infer_variable_types_from_data()
{
    Index feature_index = 0;

    const Index variables_number = get_variables_number();

    for (Index variable_index = 0; variable_index < variables_number; ++variable_index)
    {
        Variable& variable = variables[variable_index];

        if (variable.type == VariableType::Numeric)
        {
            const VectorR data_column = data.col(feature_index);

            if (is_constant(data_column))
                variable.set(variable.name, "None", VariableType::Constant);
            else if (is_binary(data_column))
            {
                variable.type = VariableType::Binary;
                variable.categories = { "0", "1" };
            }

            ++feature_index;
        }
        else if (variable.type == VariableType::Binary)
        {
            if (variable.get_categories_number() == 1)
                variable.set(variable.name, "None", VariableType::Constant);

            ++feature_index;
        }
        else if (variable.type == VariableType::Categorical)
        {
            if (variable.get_categories_number() == 1)
                variable.set(variable.name, "None", VariableType::Constant);

            feature_index += variable.get_categories_number();
        }
        else
        {
            ++feature_index;
        }
    }
}

void MaterializedDataset::set_binary_variables()
{
    infer_variable_types_from_data();
}

vector<string> MaterializedDataset::unuse_uncorrelated_variables(const float minimum_correlation)
{
    vector<string> unused_variables;

    const Tensor<Correlation, 2> correlations = calculate_input_target_variable_pearson_correlations();

    const Index input_variables_number = get_variables_number("Input");
    const Index target_variables_number = get_variables_number("Target");

    const vector<Index> input_variable_indices = get_variable_indices("Input");

    for (Index i = 0; i < input_variables_number; ++i)
    {
        const Index input_variable_index = input_variable_indices[i];

        bool has_significant_correlation = false;

        for (Index j = 0; j < target_variables_number; ++j)
        {
            const float correlation_value = correlations(i, j).r;

            if (!isnan(correlation_value) && abs(correlation_value) >= minimum_correlation)
            {
                has_significant_correlation = true;
                break;
            }
        }

        if (!has_significant_correlation && variables[input_variable_index].role != VariableRole::None)
        {
            variables[input_variable_index].set_role("None");
            unused_variables.push_back(variables[input_variable_index].name);
        }
    }

    const Index new_input_variables_number = get_features_number("Input");
    const Index new_target_variables_number = get_features_number("Target");

    resize_input_shape(new_input_variables_number);
    set_shape("Target", {new_target_variables_number});

    return unused_variables;
}

vector<string> MaterializedDataset::unuse_collinear_variables(const float maximum_correlation)
{
    const Tensor<Correlation, 2> correlations = calculate_input_variable_pearson_correlations();
    const vector<Index> input_variable_indices = get_variable_indices("Input");
    const Index input_variables_number = input_variable_indices.size();

    vector<Index> high_corr_counts(input_variables_number, 0);
    vector<float> mean_abs_corr(input_variables_number, 0.0);
    vector<bool> to_be_removed(input_variables_number, false);

    for (Index i = 0; i < input_variables_number; ++i)
    {
        float sum_of_abs_corr = 0.0;
        for (Index j = 0; j < input_variables_number; ++j)
        {
            if (i == j) continue;

            const float abs_r = abs(correlations(i, j).r);
            if (!isnan(abs_r))
            {
                if (abs_r >= maximum_correlation)
                    high_corr_counts[i]++;

                sum_of_abs_corr += abs_r;
            }
        }

        if (input_variables_number > 1)
            mean_abs_corr[i] = sum_of_abs_corr / (input_variables_number - 1);
    }

    for (Index i = 0; i < input_variables_number; ++i)
    {
        for (Index j = i + 1; j < input_variables_number; ++j)
        {

            if (to_be_removed[i] || to_be_removed[j])
                continue;

            if (!isnan(correlations(i, j).r) && abs(correlations(i, j).r) >= maximum_correlation)
            {
                const Index index_to_flag_for_removal =
                    (high_corr_counts[i] > high_corr_counts[j]) ? i :
                        (high_corr_counts[j] > high_corr_counts[i]) ? j :
                        (mean_abs_corr[i] >= mean_abs_corr[j]) ? i : j;

                to_be_removed[index_to_flag_for_removal] = true;
            }
        }
    }

    vector<string> unused_variables;
    for (Index i = 0; i < input_variables_number; ++i)
    {
        if (to_be_removed[i])
        {
            const Index global_variable_index = input_variable_indices[i];

            if (variables[global_variable_index].role != VariableRole::None)
            {
                variables[global_variable_index].set_role("None");
                unused_variables.push_back(variables[global_variable_index].name);
            }
        }
    }

    return unused_variables;
}

vector<Histogram> MaterializedDataset::calculate_variable_distributions(const Index bins_number) const
{
    const Index used_variables_number = get_used_variables_number();
    const vector<Index> used_sample_indices = get_used_sample_indices();
    const Index used_samples_number = used_sample_indices.size();

    vector<Histogram> histograms(used_variables_number);

    Index feature_index = 0;
    Index used_variable_index = 0;

    for (const Variable& variable : variables)
    {
        if (variable.role == VariableRole::None)
        {
            feature_index += (variable.type == VariableType::Categorical)
            ? variable.get_categories_number()
            : 1;
            continue;
        }

        switch (variable.type)
        {

        case VariableType::Numeric:
        {
            VectorR variable_data(used_samples_number);

            for (Index j = 0; j < used_samples_number; ++j)
                variable_data(j) = data(used_sample_indices[j], feature_index);

            histograms[used_variable_index++] = histogram(variable_data, bins_number);

            ++feature_index;
        }
        break;

        case VariableType::Categorical:
        {
            const Index categories_number = variable.get_categories_number();

            VectorR categories_frequencies = VectorR::Zero(categories_number);
            VectorR centers(categories_number);

            for (Index j = 0; j < categories_number; ++j)
            {
                for (Index k = 0; k < used_samples_number; ++k)
                    if (abs(data(used_sample_indices[k], feature_index) - 1.0f) < EPSILON)
                        categories_frequencies(j)++;

                centers(j) = float(j);

                ++feature_index;
            }

            histograms[used_variable_index].frequencies = categories_frequencies;
            histograms[used_variable_index].centers = centers;

            ++used_variable_index;
        }
        break;

        case VariableType::Binary:
        {
            VectorR binary_frequencies = VectorR::Zero(2);

            for (Index j = 0; j < used_samples_number; ++j)
                binary_frequencies(abs(data(used_sample_indices[j], feature_index) - 1.0f) < EPSILON
                   ? 1
                   : 0)++;

            histograms[used_variable_index].frequencies = binary_frequencies;
            ++feature_index;
            ++used_variable_index;
        }
        break;

        case VariableType::DateTime:

            ++feature_index;

            break;

        default:

            throw runtime_error("Unknown variable type.");
        }
    }

    return histograms;
}

vector<BoxPlot> MaterializedDataset::calculate_variables_box_plots() const
{
    const Index variables_number = get_variables_number();

    const vector<Index> used_sample_indices = get_used_sample_indices();

    vector<BoxPlot> box_plots(variables_number);

    Index feature_index = 0;

    for (Index i = 0; i < variables_number; ++i)
    {
        const Variable& variable = variables[i];

        if (variable.type == VariableType::Numeric
        || variable.type == VariableType::Binary)
        {
            if (variable.role != VariableRole::None)
                box_plots[i] = box_plot(data.col(feature_index), used_sample_indices);

            ++feature_index;
        }
        else if (variable.type == VariableType::Categorical)
        {
            feature_index += variable.get_categories_number();
        }
        else
        {
            ++feature_index;
        }
    }

    return box_plots;
}

vector<Descriptives> MaterializedDataset::calculate_feature_descriptives() const
{
    return descriptives(data);
}

vector<Index> MaterializedDataset::filter_used_samples_by_column(Index column_index, bool positive) const
{
    const vector<Index> used_sample_indices = get_used_sample_indices();

    vector<Index> filtered;
    filtered.reserve(used_sample_indices.size());

    for (const Index sample_index : used_sample_indices)
    {
        const bool match = positive
            ? abs(data(sample_index, column_index) - 1.0f) < EPSILON
            : data(sample_index, column_index) < EPSILON;

        if (match)
            filtered.push_back(sample_index);
    }

    return filtered;
}

vector<Descriptives> MaterializedDataset::calculate_variable_descriptives_positive_samples() const
{
    const Index target_index = get_feature_indices("Target")[0];

    return descriptives(data, filter_used_samples_by_column(target_index, true), get_feature_indices("Input"));
}

vector<Descriptives> MaterializedDataset::calculate_variable_descriptives_negative_samples() const
{
    const Index target_index = get_feature_indices("Target")[0];

    return descriptives(data, filter_used_samples_by_column(target_index, false), get_feature_indices("Input"));
}

vector<Descriptives> MaterializedDataset::calculate_variable_descriptives_categories(const Index class_index) const
{
    return descriptives(data, filter_used_samples_by_column(class_index, true), get_feature_indices("Input"));
}

vector<Descriptives> MaterializedDataset::calculate_feature_descriptives(const string& variable_role) const
{
    const vector<Index> used_sample_indices = get_used_sample_indices();

    const vector<Index> input_feature_indices = get_feature_indices(variable_role);

    return descriptives(data, used_sample_indices, input_feature_indices);
}

Tensor<Correlation, 2> MaterializedDataset::calculate_input_target_variable_correlations(
    Correlation (*correlation_function)(const MatrixR&, const MatrixR&),
    const string& method_name) const
{
    if (display) cout << "Calculating " << method_name << " correlations..." << "\n";

    const Index input_variables_number = get_variables_number("Input");
    const Index target_variables_number = get_variables_number("Target");

    const vector<Index> input_variable_indices = get_variable_indices("Input");
    const vector<Index> target_variable_indices = get_variable_indices("Target");

    const vector<Index> used_sample_indices = get_used_sample_indices();

    Tensor<Correlation, 2> correlations(input_variables_number, target_variables_number);

    #pragma omp parallel for schedule(dynamic)
    for (Index i = 0; i < input_variables_number; ++i)
    {
        const Index input_variable_index = input_variable_indices[i];
        const MatrixR input_variable_data = get_variable_data(input_variable_index, used_sample_indices);

        for (Index j = 0; j < target_variables_number; ++j)
        {
            const Index target_variable_index = target_variable_indices[j];
            const MatrixR target_variable_data = get_variable_data(target_variable_index, used_sample_indices);
            correlations(i, j) = correlation_function(input_variable_data, target_variable_data);
        }
    }

    return correlations;
}

Tensor<Correlation, 2> MaterializedDataset::calculate_input_target_variable_pearson_correlations() const
{
    return calculate_input_target_variable_correlations(correlation, "pearson");
}

Tensor<Correlation, 2> MaterializedDataset::calculate_input_target_variable_spearman_correlations() const
{
    return calculate_input_target_variable_correlations(correlation_spearman, "spearman");
}

bool MaterializedDataset::has_nan() const
{
    const Index rows_number = data.rows();

    for (Index i = 0; i < rows_number; ++i)
        if (sample_roles[i] != SampleRole::None)
            if (has_nan_row(i))
                return true;

    return false;
}

bool MaterializedDataset::has_nan_row(const Index row_index) const
{
    const Index features_number = get_features_number();

    for (Index j = 0; j < features_number; ++j)
        if (isnan(data(row_index, j)))
            return true;

    return false;
}

Tensor<Correlation, 2> MaterializedDataset::calculate_input_variable_correlations(
    Correlation (*correlation_function)(const MatrixR&, const MatrixR&),
    Correlation::Method method,
    const string& method_name) const
{
    if (display) cout << "Calculating " << method_name << " inputs correlations..." << "\n";

    const vector<Index> input_variable_indices = get_variable_indices("Input");

    const Index input_variables_number = input_variable_indices.size();

    Tensor<Correlation, 2> correlations(input_variables_number, input_variables_number);

    for (Index i = 0; i < input_variables_number; ++i)
    {
        if (display) cout << "Correlation " << i + 1 << " of " << input_variables_number << "\n";

        const MatrixR input_i = get_variable_data(input_variable_indices[i]);

        if (is_constant(input_i)) continue;

        correlations(i, i).set_perfect();
        correlations(i, i).method = method;

        for (Index j = i + 1; j < input_variables_number; ++j)
        {
            const MatrixR input_j = get_variable_data(input_variable_indices[j]);

            correlations(i, j) = correlation_function(input_i, input_j);

            if (correlations(i, j).r > 1.0f - EPSILON)
                correlations(i, j).r = 1.0f;

            correlations(j, i) = correlations(i, j);
        }
    }

    return correlations;
}

Tensor<Correlation, 2> MaterializedDataset::calculate_input_variable_pearson_correlations() const
{
    return calculate_input_variable_correlations(correlation, Correlation::Method::Pearson, "pearson");
}

Tensor<Correlation, 2> MaterializedDataset::calculate_input_variable_spearman_correlations() const
{
    return calculate_input_variable_correlations(correlation_spearman, Correlation::Method::Spearman, "spearman");
}

VectorI MaterializedDataset::calculate_correlations_rank() const
{
    const Tensor<Correlation, 2> correlations = calculate_input_target_variable_pearson_correlations();

    const MatrixR absolute_correlations = get_correlation_values(correlations).array().abs();

    const VectorR absolute_mean_correlations = absolute_correlations.rowwise().mean();

    return calculate_rank(absolute_mean_correlations);
}

void MaterializedDataset::apply_scaler(Index feature_index, const string& scaler, const Descriptives& desc, bool unscale)
{
    const ScalerMethod method = string_to_scaler_method(scaler);

    if (method == ScalerMethod::None)
        return;

    MatrixMap map(data.data(), data.rows(), data.cols());

    switch (method)
    {
    case ScalerMethod::MinimumMaximum:
        if (unscale)
            unscale_minimum_maximum(map, feature_index, desc);
        else
            scale_minimum_maximum(map, feature_index, desc);
        break;
    case ScalerMethod::MeanStandardDeviation:
        if (unscale)
            unscale_mean_standard_deviation(map, feature_index, desc);
        else
            scale_mean_standard_deviation(map, feature_index, desc);
        break;
    case ScalerMethod::StandardDeviation:
        if (unscale)
            unscale_standard_deviation(map, feature_index, desc);
        else
            scale_standard_deviation(map, feature_index, desc);
        break;
    case ScalerMethod::Logarithm:
        if (unscale)
            unscale_logarithmic(map, feature_index);
        else
            scale_logarithmic(map, feature_index);
        break;
    case ScalerMethod::ImageMinMax:
        if (unscale) unscale_image_minimum_maximum(map, feature_index);
        break;
    default:
        break;
    }
}

vector<Descriptives> MaterializedDataset::scale_data()
{
    const Index features_number = get_features_number();

    const vector<Descriptives> feature_descriptives = calculate_feature_descriptives();

    for (Index i = 0; i < features_number; ++i)
        apply_scaler(i, scaler_method_to_string(variables[get_variable_index(i)].scaler), feature_descriptives[i], false);

    return feature_descriptives;
}

vector<Descriptives> MaterializedDataset::scale_features(const string& variable_role)
{
    const vector<Index> feature_indices = get_feature_indices(variable_role);
    const vector<string> scalers = get_feature_scalers(variable_role);
    const vector<Descriptives> feature_descriptives = calculate_feature_descriptives(variable_role);

    for (size_t i = 0; i < feature_indices.size(); ++i)
        apply_scaler(feature_indices[i], scalers[i], feature_descriptives[i], false);

    return feature_descriptives;
}

void MaterializedDataset::unscale_features(const string& variable_role,
                                            const vector<Descriptives>& feature_descriptives)
{
    const vector<Index> feature_indices = get_feature_indices(variable_role);
    const vector<string> scalers = get_feature_scalers(variable_role);

    for (size_t i = 0; i < feature_indices.size(); ++i)
        apply_scaler(feature_indices[i], scalers[i], feature_descriptives[i], true);
}

void MaterializedDataset::set_data_constant(const float new_value)
{
    data.setConstant(new_value);
}

void MaterializedDataset::set_data_random()
{
    set_random_uniform(data);
}

void MaterializedDataset::set_data_integer(const Index vocabulary_size)
{
    set_random_integer(data, 0, vocabulary_size - 1);
}

void MaterializedDataset::samples_from_JSON(const Json *samples_element)
{
    if (!samples_element)
        throw runtime_error("Samples element is nullptr.\n");

    const Index samples_number = read_json_index(samples_element, "SamplesNumber");

    if (has_sample_ids)
    {
        const string separator_string = get_separator_string();
        sample_ids = get_tokens(read_json_string(samples_element, "SamplesId"), separator_string);
    }

    if (!variables.empty())
    {
        const vector<vector<Index>> all_feature_indices = get_feature_indices();

        const auto& last_indices = all_feature_indices.back();
        data = MatrixR::Zero(samples_number, last_indices.back() + 1);

        sample_roles.resize(samples_number, SampleRole::Training);
        set_sample_roles(get_tokens(read_json_string(samples_element, "SampleRoles"), " "));
    }
    else
        data.resize(0, 0);
}

void MaterializedDataset::from_JSON(const JsonDocument& data_set_document)
{
    const Json* root = get_json_root(data_set_document, "Dataset");

    const Json* src = require_json_field(root, "DataSource");

    set_data_path(read_json_string(src, "Path"));
    set_separator_name(read_json_string(src, "Separator"));
    set_has_header(read_json_bool(src, "HasHeader"));
    set_has_ids(read_json_bool(src, "HasSamplesId"));
    set_missing_values_label(read_json_string(src, "MissingValuesLabel"));
    set_codification(read_json_string(src, "Codification"));

    variables_from_JSON(require_json_field(root, "Variables"));
    samples_from_JSON(require_json_field(root, "Samples"));
    missing_values_from_JSON(require_json_field(root, "MissingValues"));
    preview_data_from_JSON(require_json_field(root, "PreviewData"));

    set_display(read_json_bool(root, "Display"));

    input_shape = { get_features_number("Input") };
    target_shape = { get_features_number("Target") };
}

void MaterializedDataset::save_data() const
{
    ofstream file(data_path);

    if (!file.is_open())
        throw runtime_error("Cannot open matrix data file: " + data_path.string() + "\n");

    file.precision(20);

    const Index samples_number = get_samples_number();
    const Index features_number = get_features_number();

    const vector<string> feature_names = get_feature_names();

    const string separator_string = get_separator_string();

    if (has_sample_ids)
        file << "id" << separator_string;

    for (Index j = 0; j < features_number; ++j)
    {
        file << feature_names[j];

        if (j != features_number - 1)
            file << separator_string;
    }

    file << "\n";

    for (Index i = 0; i < samples_number; ++i)
    {
        if (has_sample_ids)
            file << sample_ids[i] << separator_string;

        for (Index j = 0; j < features_number; ++j)
        {
            file << data(i, j);

            if (j != features_number - 1)
                file << separator_string;
        }

        file << "\n";
    }

    file.close();
}

void MaterializedDataset::save_data_binary(const filesystem::path& binary_data_file_name) const
{
    ofstream file(binary_data_file_name, ios::binary);

    if (!file.is_open())
        throw runtime_error("Cannot open data binary file.");

    streamsize size = sizeof(Index);

    Index columns_number = data.cols();
    Index rows_number = data.rows();

    file.write(reinterpret_cast<char*>(&columns_number), size);
    file.write(reinterpret_cast<char*>(&rows_number), size);

    size = sizeof(float);

    const Index total_elements = columns_number * rows_number;

    file.write(reinterpret_cast<const char*>(data.data()), total_elements * size);

    file.close();
}

void MaterializedDataset::load_data_binary()
{
    ifstream file(data_path, ios::binary);

    if (!file.is_open())
        throw runtime_error("Failed to open file: " + data_path.string());

    streamsize size = sizeof(Index);

    Index columns_number = 0;
    Index rows_number = 0;

    file.read(reinterpret_cast<char*>(&columns_number), size);
    file.read(reinterpret_cast<char*>(&rows_number), size);

    size = sizeof(float);

    data.resize(rows_number, columns_number);

    const Index total_elements = rows_number * columns_number;

    file.read(reinterpret_cast<char*>(data.data()), total_elements * size);

    file.close();
}

VectorI MaterializedDataset::calculate_target_distribution() const
{
    const Index samples_number = get_samples_number();
    const Index targets_number = get_features_number("Target");
    const vector<Index> target_feature_indices = get_feature_indices("Target");

    VectorI class_distribution;

    if (targets_number == 1)
    {
        class_distribution.resize(2);

        const Index target_index = target_feature_indices[0];

        Index positives = 0;
        Index negatives = 0;

        for (Index sample_index = 0; sample_index < samples_number; ++sample_index) {
            if (!isnan(data(sample_index, target_index)))
                (data(sample_index, target_index) < 0.5f)
                    ? negatives++
                    : positives++;
        }

        class_distribution(0) = negatives;
        class_distribution(1) = positives;
    }
    else
    {
        class_distribution = VectorI::Zero(targets_number);

        for (Index i = 0; i < samples_number; ++i)
        {
            if (sample_roles[i] == SampleRole::None)
                continue;

            for (Index j = 0; j < targets_number; ++j)
            {
                if (isnan(data(i, target_feature_indices[j])))
                    continue;

                if (data(i, target_feature_indices[j]) > 0.5f)
                    class_distribution(j)++;
            }
        }
    }

    return class_distribution;
}

vector<vector<Index>> MaterializedDataset::calculate_Tukey_outliers(const float cleaning_parameter, bool replace_with_nan)
{
    const Index samples_number = get_used_samples_number();
    const vector<Index> sample_indices = get_used_sample_indices();

    const Index variables_number = get_variables_number();
    const Index used_variables_number = get_used_variables_number();

    vector<vector<Index>> return_values(2);

    return_values[0].resize(samples_number, 0);
    return_values[1].resize(used_variables_number, 0);

    const vector<BoxPlot> box_plots = calculate_variables_box_plots();

    Index feature_index = 0;
    Index used_feature_index = 0;

    for (Index i = 0; i < variables_number; ++i)
    {
        const Variable& variable = variables[i];

        if (variable.role == VariableRole::None
        && variable.type == VariableType::Categorical)
        {
            feature_index += variable.get_categories_number();
            continue;
        }
        else if (variable.role == VariableRole::None)
        {
            ++feature_index;
            continue;
        }

        if (variable.type == VariableType::Categorical)
        {
            feature_index += variable.get_categories_number();
            ++used_feature_index;
            continue;
        }
        else if (variable.type == VariableType::Binary
                 || variable.type == VariableType::DateTime)
        {
            ++feature_index;
            ++used_feature_index;
            continue;
        }
        else
        {
            const float interquartile_range = box_plots[i].third_quartile - box_plots[i].first_quartile;

            if (interquartile_range < EPSILON)
            {
                ++feature_index;
                ++used_feature_index;
                continue;
            }

            Index variables_outliers = 0;

            for (Index j = 0; j < samples_number; ++j)
            {
                const VectorR sample = get_sample_data(sample_indices[Index(j)]);

                if (sample(feature_index) < box_plots[i].first_quartile - cleaning_parameter * interquartile_range
                || sample(feature_index) > box_plots[i].third_quartile + cleaning_parameter * interquartile_range)
                {
                    return_values[0][j] = 1;

                    ++variables_outliers;

                    if (replace_with_nan)
                        data(sample_indices[Index(j)], feature_index) = QUIET_NAN;
                }
            }

            return_values[1][used_feature_index] = variables_outliers;

            ++feature_index;
            ++used_feature_index;
        }
    }

    return return_values;
}

vector<vector<Index>> MaterializedDataset::replace_Tukey_outliers_with_NaN(const float cleaning_parameter)
{
    return calculate_Tukey_outliers(cleaning_parameter, true);
}

void MaterializedDataset::unuse_Tukey_outliers(const float cleaning_parameter)
{
    const vector<vector<Index>> outliers_indices = calculate_Tukey_outliers(cleaning_parameter);

    vector<Index> flat_outliers;
    for (const auto& per_feature : outliers_indices)
        flat_outliers.insert(flat_outliers.end(), per_feature.begin(), per_feature.end());

    const vector<Index> outliers_samples = get_elements_greater_than(flat_outliers, 0);

    set_sample_roles(outliers_samples, "None");
}

void MaterializedDataset::set_data_rosenbrock()
{
    const Index samples_number = get_samples_number();
    const Index features_number = get_features_number();

    set_data_random();

#pragma omp parallel for

    for (Index i = 0; i < samples_number; ++i)
    {
        float rosenbrock(0);

        for (Index j = 0; j < features_number - 1; ++j)
        {
            const float value = data(i, j);
            const float next_value = data(i, j + 1);

            rosenbrock += (1.0f - value) * (1.0f - value) + 100.0f * (next_value - value * value) * (next_value - value * value);
        }

        data(i, features_number - 1) = rosenbrock;
    }
}

void MaterializedDataset::set_data_binary_classification()
{
    const Index samples_number = get_samples_number();
    const Index features_number = get_features_number();

    set_data_random();

#pragma omp parallel for
    for (Index i = 0; i < samples_number; ++i)
        data(i, features_number - 1) = float(random_bool());
}

void MaterializedDataset::impute_missing_values_unuse()
{
    const Index samples_number = get_samples_number();

#pragma omp parallel for

    for (Index i = 0; i < samples_number; ++i)
        if (has_nan_row(i))
            set_sample_role(i, "None");
}

void MaterializedDataset::impute_missing_values_statistic(const MissingValuesMethod& method)
{
    const vector<Index> used_sample_indices = get_used_sample_indices();
    const vector<Index> used_feature_indices = get_used_feature_indices();
    const vector<Index> input_feature_indices = get_feature_indices("Input");
    const vector<Index> target_feature_indices = get_feature_indices("Target");

    if (used_sample_indices.empty() || used_feature_indices.empty())
        return;

    const VectorR replacements = (method == MissingValuesMethod::Mean)
        ? mean(data, used_sample_indices, used_feature_indices)
        : median(data, used_sample_indices, used_feature_indices);

    const Index samples_number = used_sample_indices.size();
    const Index features_number = used_feature_indices.size();
    const Index target_features_number = target_feature_indices.size();

    for (Index j = 0; j < features_number - target_features_number; ++j)
    {
        const Index current_variable = input_feature_indices[j];

        for (Index i = 0; i < samples_number; ++i)
        {
            const Index current_sample = used_sample_indices[i];

            if (isnan(data(current_sample, current_variable)))
                data(current_sample, current_variable) = replacements(j);
        }
    }

    for (Index j = 0; j < target_features_number; ++j)
    {
        const Index current_variable = target_feature_indices[j];

        for (Index i = 0; i < samples_number; ++i)
        {
            const Index current_sample = used_sample_indices[i];

            if (isnan(data(current_sample, current_variable)))
                set_sample_role(i, "None");
        }
    }
}

void MaterializedDataset::impute_missing_values_interpolate()
{
    const vector<Index> used_sample_indices = get_used_sample_indices();
    const vector<Index> used_feature_indices = get_used_feature_indices();
    const vector<Index> input_feature_indices = get_feature_indices("Input");
    const vector<Index> target_feature_indices = get_feature_indices("Target");

    const Index samples_number = used_sample_indices.size();
    const Index features_number = used_feature_indices.size();
    const Index target_features_number = target_feature_indices.size();

    Index current_variable;
    Index current_sample;

    for (Index j = 0; j < features_number - target_features_number; ++j)
    {
        current_variable = input_feature_indices[j];

        for (Index i = 0; i < samples_number; ++i)
        {
            current_sample = used_sample_indices[i];

            if (isnan(data(current_sample, current_variable)))
            {
                Index x1 = 0;
                Index x2 = 0;
                Index y1 = 0;
                Index y2 = 0;
                float x = 0.0f;
                float y = 0.0f;

                for (Index k = i - 1; k >= 0; k--)
                {
                    if (isnan(data(used_sample_indices[k], current_variable))) continue;

                    x1 = used_sample_indices[k];
                    y1 = data(x1, current_variable);
                    break;
                }

                for (Index k = i + 1; k < samples_number; ++k)
                {
                    if (isnan(data(used_sample_indices[k], current_variable))) continue;

                    x2 = used_sample_indices[k];
                    y2 = data(x2, current_variable);
                    break;
                }

                if (x2 != x1)
                {
                    x = float(current_sample);
                    y = y1 + (x - x1) * (y2 - y1) / (x2 - x1);
                }
                else
                {
                    y = y1;
                }

                data(current_sample, current_variable) = y;
            }
        }
    }

    for (Index j = 0; j < target_features_number; ++j)
    {
        current_variable = target_feature_indices[j];

        for (Index i = 0; i < samples_number; ++i)
        {
            current_sample = used_sample_indices[i];

            if (isnan(data(current_sample, current_variable)))
                set_sample_role(i, "None");
        }
    }
}

void MaterializedDataset::scrub_missing_values()
{
    switch (missing_values_method)
    {
    case MissingValuesMethod::Unuse:
        impute_missing_values_unuse();
        break;

    case MissingValuesMethod::Mean:
    case MissingValuesMethod::Median:
        impute_missing_values_statistic(missing_values_method);
        break;

    case MissingValuesMethod::Interpolation:
        impute_missing_values_interpolate();
        break;
    }

    missing_values_number = count_nan();
}

void MaterializedDataset::calculate_missing_values_statistics()
{
    missing_values_number = count_nan();
    variables_missing_values_number = count_nans_per_variable();
    rows_missing_values_number = count_rows_with_nan();
}

void MaterializedDataset::read_csv()
{
    if (data_path.empty())
        throw runtime_error("Data path is empty.\n");

    ifstream file(data_path, ios::binary);

    if (!file.is_open())
        throw runtime_error("Cannot open file " + data_path.string() + "\n");

    // BOM

    char bom[3] = {0};
    file.read(bom, 3);

    if (static_cast<unsigned char>(bom[0]) != 0xEF
        || static_cast<unsigned char>(bom[1]) != 0xBB
        || static_cast<unsigned char>(bom[2]) != 0xBF)
        file.seekg(0);

    // Read file

    vector<vector<string>> raw_file_content;
    string line;
    const string separator_string = get_separator_string();

    while (getline(file, line))
    {
        if (!line.empty() && line.back() == '\r')
            line.pop_back();

        prepare_line(line);

        if (line.empty())
            continue;

        check_separators(line);

        raw_file_content.push_back(get_tokens(line, separator_string));
    }

    file.close();

    if (raw_file_content.empty())
        throw runtime_error("File " + data_path.string() + " is empty or contains no valid data rows.");

    read_data_file_preview(raw_file_content);

    vector<string> header_tokens = raw_file_content[0];
    if (has_header)
    {
        if (has_numbers(header_tokens))
            throw runtime_error("Some header names are numeric.");

        raw_file_content.erase(raw_file_content.begin());
    }

    if (raw_file_content.empty())
        throw runtime_error("Data file only contains a header.");

    // Id detection

    const Index samples_number = raw_file_content.size();

    if (!has_sample_ids && samples_number > 0)
    {
        std::unordered_set<string> unique_elements;

        bool possible_id = true;
        bool is_numeric_column = true;
        bool is_date_column = true;
        Index date_check_count = 0;
        const Index max_date_checks = 20;

        for (const vector<string>& row : raw_file_content)
        {
            if (row.empty())
                continue;

            const string& token = row[0];

            if (!unique_elements.insert(token).second)
            {
                possible_id = false;
                break;
            }

            if (is_numeric_column && !token.empty() && token != missing_values_label)
                if (!is_numeric_string(token))
                    is_numeric_column = false;

            if (is_date_column && date_check_count < max_date_checks && !token.empty() && token != missing_values_label)
            {
                if (!is_date_time_string(token))
                    is_date_column = false;

                ++date_check_count;
            }
        }

        if (is_date_column && date_check_count > 0)
            possible_id = false;

        if (possible_id && !is_numeric_column && unique_elements.size() == static_cast<size_t>(samples_number))
            has_sample_ids = true;
    }

    // Variables

    const size_t columns_number = header_tokens.size();
    const Index variables_number = has_sample_ids ? columns_number - 1 : columns_number;
    variables.resize(variables_number);

    if (has_header)
        if (has_sample_ids)
            for (Index i = 0; i < variables_number; ++i) variables[i].name = header_tokens[i + 1];
        else
            set_variable_names(header_tokens);
    else
        set_default_variable_names();

    infer_column_types(raw_file_content);

    const DateFormat date_format = infer_dataset_date_format(variables, raw_file_content, has_sample_ids, missing_values_label);

    for (Variable& variable : variables)
        if (variable.type == VariableType::Categorical && variable.get_categories_number() == 2)
            variable.type = VariableType::Binary;

    // Samples data

    sample_roles.resize(samples_number, SampleRole::Training);
    sample_ids.resize(samples_number);

    const vector<vector<Index>> all_feature_indices = get_feature_indices();
    const Index total_numeric_columns = all_feature_indices.empty() ? 0 : all_feature_indices.back().back() + 1;

    data = MatrixR::Zero(samples_number, total_numeric_columns);

    rows_missing_values_number = 0;
    missing_values_number = 0;

    variables_missing_values_number = VectorI::Zero(variables_number);

    vector<unordered_map<string, Index>> category_maps(variables_number);
    for (Index variable_index = 0; variable_index < variables_number; ++variable_index)
        if (variables[variable_index].type == VariableType::Categorical)
            for (Index category_index = 0; category_index < ssize(variables[variable_index].categories); ++category_index)
                category_maps[variable_index][variables[variable_index].categories[category_index]] = category_index;

    for (Index sample_index = 0; sample_index < samples_number; ++sample_index)
    {
        const vector<string>& tokens = raw_file_content[sample_index];

        if (has_missing_values(tokens))
        {
            ++rows_missing_values_number;
            for (size_t i = (has_sample_ids ? 1 : 0); i < tokens.size(); ++i)
            {
                if (tokens[i].empty() || tokens[i] == missing_values_label)
                {
                    ++missing_values_number;
                    variables_missing_values_number(has_sample_ids ? i - 1 : i)++;
                }
            }
        }

        if (has_sample_ids)
            sample_ids[sample_index] = tokens[0];

        for (Index variable_index = 0; variable_index < variables_number; ++variable_index)
        {
            const Variable& variable = variables[variable_index];
            const size_t token_index = has_sample_ids ? variable_index + 1 : variable_index;
            if (token_index >= tokens.size())
                throw runtime_error("Row " + to_string(sample_index) + " has fewer columns than expected (" + to_string(tokens.size()) + ").");
            const string& token = tokens[token_index];
            const vector<Index>& feature_indices = all_feature_indices[variable_index];

            switch (variable.type)
            {
            case VariableType::Numeric:
                data(sample_index, feature_indices[0]) = (token.empty() || token == missing_values_label) ? NAN : stof(token);
                break;
            case VariableType::DateTime:
                if (token.empty() || token == missing_values_label)
                    data(sample_index, feature_indices[0]) = NAN;
                else
                {
                    const time_t timestamp = date_to_timestamp(token, gmt, date_format);

                    if (timestamp == -1)
                        throw runtime_error("Date format is unsupported or date is prior to 1970.");
                    else
                        data(sample_index, feature_indices[0]) = timestamp;
                }
                break;
            case VariableType::Categorical:
                if (token.empty() || token == missing_values_label)
                    for (const Index cat_index : feature_indices)
                        data(sample_index, cat_index) = NAN;
                else
                {
                    auto it = category_maps[variable_index].find(token);
                    if (it != category_maps[variable_index].end())
                        data(sample_index, feature_indices[it->second]) = 1;
                }
                break;
            case VariableType::Binary:
                if (const bool is_positive = contains(positive_words, token); is_positive || contains(negative_words, token))
                    data(sample_index, feature_indices[0]) = is_positive ? 1 : 0;
                else
                {
                    const vector<string>& categories = variable.categories;

                    if (token.empty() || token == missing_values_label)
                        data(sample_index, feature_indices[0]) = NAN;
                    else if (!categories.empty() && token == categories[0])
                        data(sample_index, feature_indices[0]) = 0;
                    else if (categories.size() > 1 && token == categories[1])
                        data(sample_index, feature_indices[0]) = 1;
                    else
                        data(sample_index, feature_indices[0]) = stof(token);
                }
                break;
            default:
                break;
            }
        }
    }

    infer_variable_types_from_data();
    split_samples_random();
}

void MaterializedDataset::fill_inputs(const vector<Index>& sample_indices,
                                      const vector<Index>& input_indices,
                                      float* input_data, bool parallelize, int contiguous) const
{
    fill_tensor_data(data, sample_indices, input_indices, input_data, parallelize, contiguous);
}

void MaterializedDataset::fill_decoder(const vector<Index>& sample_indices,
                                       const vector<Index>& decoder_indices,
                                       float* decoder_data, bool parallelize, int contiguous) const
{
    fill_tensor_data(data, sample_indices, decoder_indices, decoder_data, parallelize, contiguous);
}

void MaterializedDataset::fill_targets(const vector<Index>& sample_indices,
                                       const vector<Index>& target_indices,
                                       float* target_data, bool parallelize, int contiguous) const
{
    fill_tensor_data(data, sample_indices, target_indices, target_data, parallelize, contiguous);
}

VectorI MaterializedDataset::count_nans_per_variable() const
{
    return data.array().isNaN().cast<Index>().colwise().sum();
}

Index MaterializedDataset::count_variables_with_nan() const
{
    return (count_nans_per_variable().array() > 0).count();
}

Index MaterializedDataset::count_rows_with_nan() const
{
    return data.array().isNaN().rowwise().any().count();
}

Index MaterializedDataset::count_nan() const
{
    return data.array().isNaN().count();
}

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
