//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   T A B U L A R   D A T A S E T   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "tabular_dataset.h"
#include "io_utilities.h"
#include "scaling.h"
#include "random_utilities.h"
#include <regex>

namespace opennn
{

TabularDataset::TabularDataset(const Index new_samples_number,
                                         const Shape& new_input_shape,
                                         const Shape& new_target_shape)
{
    set(new_samples_number, new_input_shape, new_target_shape);
}

TabularDataset::TabularDataset(const filesystem::path& data_path,
                                         const string& separator,
                                         bool has_header,
                                         bool has_sample_ids,
                                         const Codification& data_codification)
{
    set(data_path, separator, has_header, has_sample_ids, data_codification);
}

void TabularDataset::set(const filesystem::path& new_data_path,
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

void TabularDataset::set(const filesystem::path& file_name)
{
    load(file_name);
}

vector<string> TabularDataset::unuse_uncorrelated_variables(const float minimum_correlation)
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

        Variable& variable = variables[input_variable_index];

        if (!has_significant_correlation && variable.role != VariableRole::None)
        {
            variable.set_role("None");
            unused_variables.push_back(variable.name);
        }
    }

    resize_input_shape(get_features_number("Input"));
    set_shape("Target", { get_features_number("Target") });

    return unused_variables;
}

vector<string> TabularDataset::unuse_collinear_variables(const float maximum_correlation)
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

            const float r = correlations(i, j).r;

            if (!isnan(r) && abs(r) >= maximum_correlation)
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
        if (!to_be_removed[i]) continue;

        Variable& variable = variables[input_variable_indices[i]];

        if (variable.role != VariableRole::None)
        {
            variable.set_role("None");
            unused_variables.push_back(variable.name);
        }
    }

    return unused_variables;
}

vector<Histogram> TabularDataset::calculate_variable_distributions(const Index bins_number) const
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
            feature_index += variable.feature_count();
            continue;
        }

        switch (variable.type)
        {

        case VariableType::Numeric:
        case VariableType::Constant:
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

        case VariableType::None:
            throw runtime_error("Cannot calculate distributions for a variable with type None.");
        }
    }

    return histograms;
}

vector<BoxPlot> TabularDataset::calculate_variables_box_plots() const
{
    const Index variables_number = get_variables_number();

    const vector<Index> used_sample_indices = get_used_sample_indices();

    vector<BoxPlot> box_plots(variables_number);

    Index feature_index = 0;

    for (Index i = 0; i < variables_number; ++i)
    {
        const Variable& variable = variables[i];

        if ((variable.type == VariableType::Numeric || variable.type == VariableType::Binary)
            && variable.role != VariableRole::None)
            box_plots[i] = box_plot(data.col(feature_index), used_sample_indices);

        feature_index += variable.feature_count();
    }

    return box_plots;
}

vector<Descriptives> TabularDataset::calculate_feature_descriptives() const
{
    return descriptives(data);
}

vector<Index> TabularDataset::filter_used_samples_by_column(Index column_index, bool positive) const
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

vector<Descriptives> TabularDataset::calculate_variable_descriptives_positive_samples() const
{
    const Index target_index = get_feature_indices("Target")[0];

    return descriptives(data, filter_used_samples_by_column(target_index, true), get_feature_indices("Input"));
}

vector<Descriptives> TabularDataset::calculate_variable_descriptives_negative_samples() const
{
    const Index target_index = get_feature_indices("Target")[0];

    return descriptives(data, filter_used_samples_by_column(target_index, false), get_feature_indices("Input"));
}

vector<Descriptives> TabularDataset::calculate_variable_descriptives_categories(const Index class_index) const
{
    return descriptives(data, filter_used_samples_by_column(class_index, true), get_feature_indices("Input"));
}

vector<Descriptives> TabularDataset::calculate_feature_descriptives(const string& variable_role) const
{
    const vector<Index> used_sample_indices = get_used_sample_indices();

    const vector<Index> input_feature_indices = get_feature_indices(variable_role);

    return descriptives(data, used_sample_indices, input_feature_indices);
}

Tensor<Correlation, 2> TabularDataset::calculate_input_target_variable_correlations(
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

Tensor<Correlation, 2> TabularDataset::calculate_input_target_variable_pearson_correlations() const
{
    return calculate_input_target_variable_correlations(correlation, "pearson");
}

Tensor<Correlation, 2> TabularDataset::calculate_input_target_variable_spearman_correlations() const
{
    return calculate_input_target_variable_correlations(correlation_spearman, "spearman");
}

Tensor<Correlation, 2> TabularDataset::calculate_input_variable_correlations(
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

Tensor<Correlation, 2> TabularDataset::calculate_input_variable_pearson_correlations() const
{
    return calculate_input_variable_correlations(correlation, Correlation::Method::Pearson, "pearson");
}

Tensor<Correlation, 2> TabularDataset::calculate_input_variable_spearman_correlations() const
{
    return calculate_input_variable_correlations(correlation_spearman, Correlation::Method::Spearman, "spearman");
}

VectorI TabularDataset::calculate_correlations_rank() const
{
    const Tensor<Correlation, 2> correlations = calculate_input_target_variable_pearson_correlations();

    const MatrixR absolute_correlations = get_correlation_values(correlations).array().abs();

    const VectorR absolute_mean_correlations = absolute_correlations.rowwise().mean();

    return calculate_rank(absolute_mean_correlations);
}

void TabularDataset::apply_scaler(Index feature_index, const string& scaler, const Descriptives& desc, bool unscale)
{
    const ScalerMethod method = string_to_scaler_method(scaler);

    if (method == ScalerMethod::None)
        return;

    MatrixMap map(data.data(), data.rows(), data.cols());

    switch (method)
    {
    case ScalerMethod::None:
        break;
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
    }
}

vector<Descriptives> TabularDataset::scale_data()
{
    const Index features_number = get_features_number();

    const vector<Descriptives> feature_descriptives = calculate_feature_descriptives();

    for (Index i = 0; i < features_number; ++i)
        apply_scaler(i, scaler_method_to_string(variables[get_variable_index(i)].scaler), feature_descriptives[i], false);

    return feature_descriptives;
}

vector<Descriptives> TabularDataset::scale_features(const string& variable_role)
{
    const vector<Index> feature_indices = get_feature_indices(variable_role);
    const vector<string> scalers = get_feature_scalers(variable_role);
    const vector<Descriptives> feature_descriptives = calculate_feature_descriptives(variable_role);

    for (size_t i = 0; i < feature_indices.size(); ++i)
        apply_scaler(feature_indices[i], scalers[i], feature_descriptives[i], false);

    return feature_descriptives;
}

void TabularDataset::unscale_features(const string& variable_role,
                                            const vector<Descriptives>& feature_descriptives)
{
    const vector<Index> feature_indices = get_feature_indices(variable_role);
    const vector<string> scalers = get_feature_scalers(variable_role);

    for (size_t i = 0; i < feature_indices.size(); ++i)
        apply_scaler(feature_indices[i], scalers[i], feature_descriptives[i], true);
}

void TabularDataset::set_data_random()
{
    set_random_uniform(data);
}

void TabularDataset::set_data_integer(const Index vocabulary_size)
{
    set_random_integer(data, 0, vocabulary_size - 1);
}

void TabularDataset::from_JSON(const JsonDocument& data_set_document)
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

VectorI TabularDataset::calculate_target_distribution() const
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

        for (Index sample_index = 0; sample_index < samples_number; ++sample_index)
        {
            const float value = data(sample_index, target_index);
            if (!isnan(value))
                (value < 0.5f) ? negatives++ : positives++;
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
                const float value = data(i, target_feature_indices[j]);

                if (isnan(value)) continue;
                if (value > 0.5f) class_distribution(j)++;
            }
        }
    }

    return class_distribution;
}

vector<vector<Index>> TabularDataset::calculate_Tukey_outliers(const float cleaning_parameter, bool replace_with_nan)
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

        if (variable.role == VariableRole::None)
        {
            feature_index += variable.feature_count();
            continue;
        }

        if (variable.is_categorical() || variable.is_binary() || variable.type == VariableType::DateTime)
        {
            feature_index += variable.feature_count();
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

            const float lower = box_plots[i].first_quartile - cleaning_parameter * interquartile_range;
            const float upper = box_plots[i].third_quartile + cleaning_parameter * interquartile_range;

            Index variables_outliers = 0;

            for (Index j = 0; j < samples_number; ++j)
            {
                const Index sample_idx = sample_indices[j];
                const VectorR sample = get_sample_data(sample_idx);
                const float value = sample(feature_index);

                if (value < lower || value > upper)
                {
                    return_values[0][j] = 1;
                    ++variables_outliers;

                    if (replace_with_nan)
                        data(sample_idx, feature_index) = QUIET_NAN;
                }
            }

            return_values[1][used_feature_index] = variables_outliers;

            ++feature_index;
            ++used_feature_index;
        }
    }

    return return_values;
}

vector<vector<Index>> TabularDataset::replace_Tukey_outliers_with_NaN(const float cleaning_parameter)
{
    return calculate_Tukey_outliers(cleaning_parameter, true);
}

void TabularDataset::unuse_Tukey_outliers(const float cleaning_parameter)
{
    const vector<vector<Index>> outliers_indices = calculate_Tukey_outliers(cleaning_parameter);

    vector<Index> flat_outliers;
    for (const auto& per_feature : outliers_indices)
        flat_outliers.insert(flat_outliers.end(), per_feature.begin(), per_feature.end());

    const vector<Index> outliers_samples = get_elements_greater_than(flat_outliers, 0);

    set_sample_roles(outliers_samples, "None");
}

void TabularDataset::set_data_rosenbrock()
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
            const float a = 1.0f - value;
            const float b = data(i, j + 1) - value * value;

            rosenbrock += a * a + 100.0f * b * b;
        }

        data(i, features_number - 1) = rosenbrock;
    }
}

void TabularDataset::set_data_binary_classification()
{
    const Index samples_number = get_samples_number();
    const Index features_number = get_features_number();

    set_data_random();

#pragma omp parallel for
    for (Index i = 0; i < samples_number; ++i)
        data(i, features_number - 1) = float(random_bool());
}

namespace {

float parse_float_or_nan(string_view token)
{
    float value;
    auto [ptr, ec] = std::from_chars(token.data(), token.data() + token.size(), value);
    return (ec == std::errc{} && ptr == token.data() + token.size()) ? value : NAN;
}

}

void TabularDataset::read_csv()
{
    const string separator_string = get_separator_string();

    CsvReader::Config cfg;
    cfg.separator = separator_string.empty() ? ',' : separator_string[0];
    cfg.line_validator = [this](string_view line) { check_separators(line); };

    CsvReader::Result parsed = CsvReader(cfg).read(data_path);
    vector<vector<string_view>>& raw_file_content = parsed.rows;

    if (raw_file_content.empty())
        throw runtime_error("File " + data_path.string() + " is empty or contains no valid data rows.");

    read_data_file_preview(raw_file_content);

    vector<string_view> header_tokens = raw_file_content[0];
    if (has_header)
    {
        if (has_numbers(header_tokens))
            throw runtime_error("Some header names are numeric.");

        raw_file_content.erase(raw_file_content.begin());
    }

    if (raw_file_content.empty())
        throw runtime_error("Data file only contains a header.");

    const Index samples_number = raw_file_content.size();

    auto is_missing = [&](string_view t) { return t.empty() || t == missing_values_label; };

    if (!has_sample_ids && samples_number > 0)
    {
        std::unordered_set<string_view> unique_elements;

        bool possible_id = true;
        bool is_numeric_column = true;
        bool is_date_column = true;
        Index date_check_count = 0;
        const Index max_date_checks = 20;

        for (const vector<string_view>& row : raw_file_content)
        {
            if (row.empty())
                continue;

            const string_view token = row[0];

            if (!unique_elements.insert(token).second)
            {
                possible_id = false;
                break;
            }

            if (is_numeric_column && !is_missing(token))
                if (!is_numeric_string(token))
                    is_numeric_column = false;

            if (is_date_column && date_check_count < max_date_checks && !is_missing(token))
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

    const size_t columns_number = header_tokens.size();
    const size_t id_offset = has_sample_ids ? 1 : 0;
    const Index variables_number = columns_number - id_offset;
    variables.resize(variables_number);

    if (has_header)
    {
        vector<string> names;
        names.reserve(variables_number);
        for (Index i = 0; i < variables_number; ++i)
            names.emplace_back(header_tokens[i + id_offset]);
        set_variable_names(names);
    }
    else
        set_default_variable_names();

    infer_column_types(raw_file_content);

    const DateFormat date_format = infer_dataset_date_format(variables, raw_file_content, has_sample_ids, missing_values_label);

    for (Variable& variable : variables)
        if (variable.is_categorical() && variable.get_categories_number() == 2)
            variable.type = VariableType::Binary;

    sample_roles.resize(samples_number, SampleRole::Training);
    sample_ids.resize(samples_number);

    const vector<vector<Index>> all_feature_indices = get_feature_indices();
    const Index total_numeric_columns = all_feature_indices.empty() ? 0 : all_feature_indices.back().back() + 1;

    data = MatrixR::Zero(samples_number, total_numeric_columns);

    rows_missing_values_number = 0;
    missing_values_number = 0;

    variables_missing_values_number = VectorI::Zero(variables_number);

    vector<unordered_map<string_view, Index>> category_maps(variables_number);
    for (Index variable_index = 0; variable_index < variables_number; ++variable_index)
    {
        const Variable& variable = variables[variable_index];
        if (!variable.is_categorical()) continue;

        for (Index ci = 0; ci < ssize(variable.categories); ++ci)
            category_maps[variable_index].emplace(string_view(variable.categories[ci]), ci);
    }

    for (Index sample_index = 0; sample_index < samples_number; ++sample_index)
    {
        const vector<string_view>& tokens = raw_file_content[sample_index];

        if (has_missing_values(tokens))
        {
            ++rows_missing_values_number;
            for (size_t i = id_offset; i < tokens.size(); ++i)
            {
                if (is_missing(tokens[i]))
                {
                    ++missing_values_number;
                    variables_missing_values_number(i - id_offset)++;
                }
            }
        }

        if (has_sample_ids)
            sample_ids[sample_index] = string(tokens[0]);

        for (Index variable_index = 0; variable_index < variables_number; ++variable_index)
        {
            const Variable& variable = variables[variable_index];
            const size_t token_index = variable_index + id_offset;
            if (token_index >= tokens.size())
                throw runtime_error("Row " + to_string(sample_index) + " has fewer columns than expected (" + to_string(tokens.size()) + ").");
            const string_view token = tokens[token_index];
            const vector<Index>& feature_indices = all_feature_indices[variable_index];

            switch (variable.type)
            {
            case VariableType::None:
            case VariableType::Constant:
                break;
            case VariableType::Numeric:
                data(sample_index, feature_indices[0]) = is_missing(token) ? NAN : parse_float_or_nan(token);
                break;
            case VariableType::DateTime:
                if (is_missing(token))
                    data(sample_index, feature_indices[0]) = NAN;
                else
                {
                    const time_t timestamp = date_to_timestamp(string(token), gmt, date_format);

                    if (timestamp == -1)
                        throw runtime_error("Date format is unsupported or date is prior to 1970.");

                    data(sample_index, feature_indices[0]) = timestamp;
                }
                break;
            case VariableType::Categorical:
                if (is_missing(token))
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

                    if (is_missing(token))
                        data(sample_index, feature_indices[0]) = NAN;
                    else if (!categories.empty() && token == categories[0])
                        data(sample_index, feature_indices[0]) = 0;
                    else if (categories.size() > 1 && token == categories[1])
                        data(sample_index, feature_indices[0]) = 1;
                    else
                        data(sample_index, feature_indices[0]) = parse_float_or_nan(token);
                }
                break;
            }
        }
    }

    infer_variable_types_from_data();
    split_samples_random();
}

static const vector<pair<TabularDataset::MissingValuesMethod, string>> missing_values_method_map = {
    {TabularDataset::MissingValuesMethod::Unuse,         "Unuse"},
    {TabularDataset::MissingValuesMethod::Mean,          "Mean"},
    {TabularDataset::MissingValuesMethod::Median,        "Median"},
    {TabularDataset::MissingValuesMethod::Interpolation, "Interpolation"}
};

string TabularDataset::get_missing_values_method_string() const
{
    for (const auto& [method, name] : missing_values_method_map)
        if (method == missing_values_method) return name;

    throw runtime_error("Unknown missing values method");
}

void TabularDataset::set_missing_values_method(const string& new_missing_values_method)
{
    for (const auto& [method, name] : missing_values_method_map)
        if (name == new_missing_values_method) { missing_values_method = method; return; }

    throw runtime_error("Unknown method type.\n");
}

bool TabularDataset::has_missing_values(const vector<string_view>& row) const
{
    return any_of(row.begin(), row.end(),
                  [&](string_view t) { return t.empty() || t == missing_values_label; });
}

void TabularDataset::missing_values_to_JSON(JsonWriter &printer) const
{
    printer.open_element("MissingValues");

    if (missing_values_number > 0)
        write_json(printer, {
            {"MissingValuesNumber", to_string(missing_values_number)},
            {"MissingValuesMethod", get_missing_values_method_string()},
            {"VariablesMissingValuesNumber", vector_to_string(variables_missing_values_number)},
            {"SamplesMissingValuesNumber", to_string(rows_missing_values_number)}
        });
    else
        add_json_field(printer, "MissingValuesNumber", to_string(missing_values_number));

    printer.close_element();
}

void TabularDataset::missing_values_from_JSON(const Json *missing_values_element)
{
    if (!missing_values_element)
        throw runtime_error("Missing values element is nullptr.\n");

    missing_values_number = read_json_index(missing_values_element, "MissingValuesNumber");

    if (missing_values_number > 0)
    {
        set_missing_values_method(read_json_string(missing_values_element, "MissingValuesMethod"));

        const string variables_string = read_json_string_fallback(missing_values_element,
            {"VariablesMissingValuesNumber", "RawVariablesMissingValuesNumber"});

        const vector<string> tokens = get_tokens(variables_string, " ");

        variables_missing_values_number.resize(tokens.size());
        for (size_t i = 0; i < tokens.size(); ++i)
            if (!tokens[i].empty())
                variables_missing_values_number(i) = stoi(tokens[i]);

        rows_missing_values_number = stol(read_json_string_fallback(missing_values_element,
            {"SamplesMissingValuesNumber", "RowsMissingValuesNumber"}));
    }
}

void TabularDataset::impute_missing_values_unuse()
{
    const Index samples_number = get_samples_number();

#pragma omp parallel for

    for (Index i = 0; i < samples_number; ++i)
        if (has_nan_row(i))
            set_sample_role(i, "None");
}

void TabularDataset::impute_missing_values_statistic(const MissingValuesMethod& method)
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
                set_sample_role(current_sample, "None");
        }
    }
}

void TabularDataset::impute_missing_values_interpolate()
{
    const vector<Index> used_sample_indices = get_used_sample_indices();
    const vector<Index> input_feature_indices = get_feature_indices("Input");
    const vector<Index> target_feature_indices = get_feature_indices("Target");

    const Index samples_number = used_sample_indices.size();

    for (const Index current_variable : input_feature_indices)
    {
        for (Index i = 0; i < samples_number; ++i)
        {
            const Index current_sample = used_sample_indices[i];

            if (!isnan(data(current_sample, current_variable))) continue;

            Index x1 = 0, x2 = 0;
            float x = 0.0f, y = 0.0f, y1 = 0.0f, y2 = 0.0f;

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

    for (const Index current_variable : target_feature_indices)
    {
        for (Index i = 0; i < samples_number; ++i)
        {
            const Index current_sample = used_sample_indices[i];

            if (isnan(data(current_sample, current_variable)))
                set_sample_role(current_sample, "None");
        }
    }
}

void TabularDataset::scrub_missing_values()
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

void TabularDataset::calculate_missing_values_statistics()
{
    missing_values_number = count_nan();
    variables_missing_values_number = count_nans_per_variable();
    rows_missing_values_number = count_rows_with_nan();
}

void TabularDataset::infer_column_types(const vector<vector<string_view>>& sample_rows)
{
    const Index variables_number = variables.size();
    const size_t total_rows = sample_rows.size();

    if (total_rows == 0) return;

    vector<size_t> row_indices(total_rows);
    iota(row_indices.begin(), row_indices.end(), 0);

    shuffle_vector(row_indices);

    const size_t rows_to_check = min(size_t(100), total_rows);
    const size_t id_offset = has_sample_ids ? 1 : 0;

    for (Index col_index = 0; col_index < variables_number; ++col_index)
    {
        Variable& variable = variables[col_index];
        variable.type = VariableType::None;

        const size_t token_index = col_index + id_offset;

        for (size_t i = 0; i < rows_to_check; ++i)
        {
            const size_t row_index = row_indices[i];

            if (token_index >= sample_rows[row_index].size()) continue;

            const string_view token = sample_rows[row_index][token_index];

            if (token.empty() || token == missing_values_label) continue;

            if (variable.is_categorical()) break;

            if (is_numeric_string(token))
            {
                if (variable.type == VariableType::None)
                    variable.type = VariableType::Numeric;
            }
            else if (is_date_time_string(token))
            {
                if (variable.type == VariableType::None)
                    variable.type = VariableType::DateTime;
            }
            else
                variable.type = VariableType::Categorical;
        }

        if (variable.type == VariableType::None)
            variable.type = VariableType::Numeric;
    }

    for (Index col_index = 0; col_index < variables_number; ++col_index)
    {
        if (!variables[col_index].is_categorical()) continue;

        const size_t token_index = col_index + id_offset;

        std::set<string> unique_categories;
        for (const vector<string_view>& row : sample_rows)
            if (token_index < row.size() && !row[token_index].empty() && row[token_index] != missing_values_label)
                unique_categories.emplace(row[token_index]);

        variables[col_index].categories.assign(unique_categories.begin(), unique_categories.end());
    }
}

vector<string> TabularDataset::get_feature_scalers(const string& variable_role) const
{
    const vector<Variable> role_variables = get_variables(variable_role);

    vector<string> scalers;
    scalers.reserve(get_features_number(variable_role));

    for (const Variable& var : role_variables)
        scalers.insert(scalers.end(), var.feature_count(), scaler_method_to_string(var.scaler));

    return scalers;
}

void TabularDataset::set_variable_scalers(const string& scalers)
{
    const ScalerMethod method = string_to_scaler_method(scalers);
    for (Variable& variable : variables)
        variable.scaler = method;
}

void TabularDataset::set_variable_scalers(const vector<string>& new_scalers)
{
    const size_t variables_number = get_variables_number();

    if (new_scalers.size() != variables_number)
        throw runtime_error("Size of variable scalers(" + to_string(new_scalers.size()) + ") "
                            "has to be the same as variables numbers(" + to_string(variables_number) + ").\n");

    for (size_t i = 0; i < variables_number; ++i)
        variables[i].set_scaler(new_scalers[i]);
}

void TabularDataset::set_default_variable_scalers()
{
    for (Variable& variable : variables)
        variable.scaler = (variable.type == VariableType::Numeric)
                                  ? ScalerMethod::MeanStandardDeviation
                                  : ScalerMethod::MinimumMaximum;
}

DateFormat TabularDataset::infer_dataset_date_format(const vector<Variable>& variables,
                                                      const vector<vector<string_view>>& sample_rows,
                                                      bool has_sample_ids,
                                                      const string& missing_values_label)
{
    static const regex date_re(R"((\d{1,2})[-/.](\d{1,2})[-/.](\d{4}).*)");

    const size_t id_offset = has_sample_ids ? 1 : 0;

    for (size_t col_index = 0; col_index < variables.size(); ++col_index)
    {
        if (variables[col_index].type != VariableType::DateTime)
            continue;

        const size_t token_index = col_index + id_offset;

        for (const vector<string_view>& row : sample_rows)
        {
            if (token_index >= row.size())
                continue;

            const string_view token = row[token_index];

            if (token.empty() || token == missing_values_label)
                continue;

            cmatch date_parts;
            if (regex_match(token.data(), token.data() + token.size(), date_parts, date_re))
            {
                const int part1 = stoi(date_parts[1].str());
                const int part2 = stoi(date_parts[2].str());

                if (part1 > 12)
                    return DMY;
                if (part2 > 12)
                    return MDY;
            }
        }
    }

    return AUTO;
}

void TabularDataset::to_JSON(JsonWriter& printer) const
{
    printer.open_element("Dataset");

    printer.open_element("DataSource");
    write_json(printer, {
        {"FileType", "csv"},
        {"Path", data_path.string()},
        {"Separator", get_separator_name()},
        {"HasHeader", to_string(has_header)},
        {"HasSamplesId", to_string(has_sample_ids)},
        {"MissingValuesLabel", missing_values_label},
        {"Codification", get_codification_string()}
    });
    printer.close_element();

    variables_to_JSON(printer);
    samples_to_JSON(printer);
    missing_values_to_JSON(printer);
    preview_data_to_JSON(printer);

    add_json_field(printer, "Display", to_string(display));

    printer.close_element();
}

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
