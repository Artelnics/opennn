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
#include "tensor_types.h"
#include "random_utilities.h"

#include <set>

namespace opennn
{

void TabularDataset::set(const Index new_samples_number,
                         const Shape& new_input_shape,
                         const Shape& new_target_shape)
{
    if (new_samples_number == 0 || new_input_shape.empty() || new_target_shape.empty())
        return;

    input_shape = new_input_shape;

    const Index new_inputs_number = new_input_shape.size();
    const Index new_targets_number = new_target_shape.size();
    const Index new_features_number = new_inputs_number + new_targets_number;

    target_shape = { new_targets_number };
    data.resize(new_samples_number, new_features_number);
    set_storage_mode(StorageMode::Matrix);
    variables.resize(new_features_number);

    for (Index i = 0; i < new_features_number; ++i)
    {
        Variable& variable = variables[i];

        variable.type = VariableType::Numeric;
        variable.name = format("variable_{}", i + 1);
        variable.role = (i < new_inputs_number) ? VariableRole::Input : VariableRole::Target;
    }

    sample_roles.resize(new_samples_number, SampleRole::Training);
    split_samples_random();
}

MatrixR TabularDataset::get_feature_data(const string& variable_role) const
{
    vector<Index> indices(get_samples_number());
    iota(indices.begin(), indices.end(), 0);

    return get_data_from_indices(indices, get_feature_indices(variable_role));
}

MatrixR TabularDataset::get_data(const string& sample_role, const string& variable_role) const
{
    return get_data_from_indices(get_sample_indices(sample_role), get_feature_indices(variable_role));
}

MatrixR TabularDataset::get_data_from_indices(const vector<Index>& sample_indices, const vector<Index>& feature_indices) const
{
    MatrixR this_data(sample_indices.size(), feature_indices.size());
    fill_tensor_data(data, sample_indices, feature_indices, this_data.data());
    return this_data;
}

MatrixR TabularDataset::get_variable_data(Index variable_index) const
{
    const Index start_column = transform_reduce(variables.begin(), variables.begin() + variable_index,
        Index(0), plus<>{}, [](const Variable& v) { return v.get_feature_count(); });
    return data.block(0, start_column, data.rows(), variables[variable_index].get_feature_count());
}

MatrixR TabularDataset::get_variable_data(Index variable_index, const vector<Index>& row_indices) const
{
    MatrixR variable_data(row_indices.size(), get_feature_indices(variable_index).size());
    fill_tensor_data(data, row_indices, get_feature_indices(variable_index), variable_data.data());
    return variable_data;
}

MatrixR TabularDataset::get_variable_data(const string& column_name) const { return get_variable_data(get_variable_index(column_name)); }

bool TabularDataset::has_nan() const
{
    for (Index i = 0; i < data.rows(); ++i)
        if (sample_roles[i] != SampleRole::None && has_nan_row(i))
            return true;

    return false;
}

bool TabularDataset::has_nan_row(Index row_index) const { return data.row(row_index).array().isNaN().any(); }

VectorI TabularDataset::count_nans_per_variable() const { return data.array().isNaN().cast<Index>().colwise().sum(); }

Index TabularDataset::count_variables_with_nan() const { return (count_nans_per_variable().array() > 0).count(); }

Index TabularDataset::count_rows_with_nan() const { return data.array().isNaN().rowwise().any().count(); }

Index TabularDataset::count_nan() const { return data.array().isNaN().count(); }

void TabularDataset::save_data() const
{
    ofstream file(data_path);

    throw_if(!file.is_open(),
             format("Cannot open matrix data file: {}\n", data_path.string()));

    file.precision(20);

    const Index samples_number = get_samples_number();
    const Index features_number = get_features_number();

    const vector<string> feature_names = get_feature_names();

    const string separator_string = get_separator_string();

    if (has_sample_ids)
        file << "id" << separator_string;

    for (Index j = 0; j < features_number; ++j)
        file << feature_names[j] << (j == features_number - 1 ? "\n" : separator_string);

    for (Index i = 0; i < samples_number; ++i)
    {
        if (has_sample_ids)
            file << sample_ids[i] << separator_string;

        for (Index j = 0; j < features_number; ++j)
            file << data(i, j) << (j == features_number - 1 ? "\n" : separator_string);
    }

    throw_if(!file,
             format("Failed to write matrix data file: {}", data_path.string()));
}

void TabularDataset::set_storage_mode(StorageMode new_storage_mode)
{
    Dataset::set_storage_mode(new_storage_mode);

    if (new_storage_mode != StorageMode::BinaryFile)
    {
        cache_columns_number = 0;
        cache_feature_descriptives.clear();
        cache_feature_transforms.clear();
        cache_feature_replacement.clear();
        cache_reader.close();
    }
}

filesystem::path TabularDataset::cache_file_path() const
{
    // Embedding applications (e.g. Neural Designer) keep the cache next to the
    // model instead of polluting the data folder; they set an explicit path.
    if (!cache_path_override.empty())
        return cache_path_override;

    return data_path.parent_path() / ".cache" / (data_path.stem().string() + ".bin");
}


void TabularDataset::set_binary_cache_path(const filesystem::path& new_cache_path)
{
    cache_path_override = new_cache_path;
    cache_reader.close();
    cache_path = cache_file_path();

    if (storage_mode == StorageMode::BinaryFile && filesystem::exists(cache_path))
        cache_reader.open(cache_path);
}

// Element-wise equivalents of the column scalers in scaling.cpp, applied to
// batches read from the raw cache file.
static float scale_value(ScalerMethod method, const Descriptives& desc, float value)
{
    using enum ScalerMethod;
    switch (method)
    {
    case None:
    case ImageMinMax:
        return value;
    case MinimumMaximum:
    {
        const float range = desc.maximum - desc.minimum;
        return range < EPSILON ? 0.0f : (value - desc.minimum) / range * 2.0f - 1.0f;
    }
    case MeanStandardDeviation:
        return desc.standard_deviation > EPSILON ? (value - desc.mean) / desc.standard_deviation : 0.0f;
    case StandardDeviation:
        return desc.standard_deviation > EPSILON ? value / desc.standard_deviation : 0.0f;
    case Logarithm:
        return log(max(value, EPSILON));
    }

    return value;
}

void TabularDataset::compute_cache_replacement() const
{
    // Per-column value used to impute NaN when filling model batches, resolved from
    // the variable types: mean for continuous variables, majority value (mode) for
    // binary ones, and the most frequent category for categoricals (its one-hot
    // column -> 1, the rest -> 0). The cache stays raw; this only feeds batch fill.
    if (cache_feature_descriptives.empty()) compute_cache_descriptives();

    const Index columns_number = cache_columns_number;
    cache_feature_replacement.assign(static_cast<size_t>(columns_number), 0.0f);

    const vector<vector<Index>> variable_feature_indices = get_feature_indices();

    for (Index variable_index = 0; variable_index < ssize(variables); ++variable_index)
    {
        const vector<Index>& feature_indices = variable_feature_indices[size_t(variable_index)];

        using enum VariableType;
        const VariableType type = variables[variable_index].type;

        if (type == Categorical)
        {
            Index mode_column = -1;
            float best_mean = -1.0f;

            for (const Index column : feature_indices)
            {
                cache_feature_replacement[size_t(column)] = 0.0f;
                const float column_mean = cache_feature_descriptives[size_t(column)].mean;
                if (column_mean > best_mean) { best_mean = column_mean; mode_column = column; }
            }

            if (mode_column >= 0) cache_feature_replacement[size_t(mode_column)] = 1.0f;
        }
        else if (type == Binary)
        {
            for (const Index column : feature_indices)
                cache_feature_replacement[size_t(column)] =
                    cache_feature_descriptives[size_t(column)].mean >= 0.5f ? 1.0f : 0.0f;
        }
        else
        {
            for (const Index column : feature_indices)
                cache_feature_replacement[size_t(column)] = cache_feature_descriptives[size_t(column)].mean;
        }
    }
}

void TabularDataset::fill_from_binary_cache(const vector<Index>& sample_indices,
                                            const vector<Index>& feature_indices,
                                            float* output,
                                            int contiguous_hint) const
{
    if (sample_indices.empty() || feature_indices.empty()) return;

    const Index columns_number = cache_columns_number;
    const Index rows_number = get_samples_number();
    const Index features_number = ssize(feature_indices);
    const bool contiguous = contiguous_hint >= 0
                          ? static_cast<bool>(contiguous_hint)
                          : is_contiguous(feature_indices);

    // The on-disk cache is raw (missing cells are NaN). Impute them on the fly so the
    // model never sees NaN, while correlations keep reading the raw cache pairwise.
    if (cache_feature_replacement.empty()) compute_cache_replacement();

    const Index first_column = feature_indices.front();
    if (contiguous)
    {
        throw_if(first_column < 0 || first_column + features_number > columns_number,
                 "Binary data feature index is out of range.");
    }
    else
    {
        for (const Index feature_index : feature_indices)
            throw_if(feature_index < 0 || feature_index >= columns_number,
                     "Binary data feature index is out of range.");
    }

    for (const Index row : sample_indices)
        throw_if(row < 0 || row >= rows_number,
                 "Binary data row index is out of range.");

    for (Index i = 0; i < ssize(sample_indices); ++i)
    {
        const Index row = sample_indices[size_t(i)];
        float* const dst = output + i * features_number;

        if (contiguous)
        {
            const uint64_t offset =
                (uint64_t(row) * uint64_t(columns_number) + uint64_t(first_column)) * sizeof(float);
            cache_reader.read_at(dst, size_t(features_number) * sizeof(float), offset);

            for (Index j = 0; j < features_number; ++j)
                if (isnan(dst[j])) dst[j] = cache_feature_replacement[size_t(first_column + j)];
        }
        else
        {
            thread_local vector<float> row_buffer;
            row_buffer.resize(size_t(columns_number));

            const uint64_t offset = uint64_t(row) * uint64_t(columns_number) * sizeof(float);
            cache_reader.read_at(row_buffer.data(), size_t(columns_number) * sizeof(float), offset);

            for (Index j = 0; j < features_number; ++j)
            {
                const Index column = feature_indices[size_t(j)];
                const float value = row_buffer[size_t(column)];
                dst[j] = isnan(value) ? cache_feature_replacement[size_t(column)] : value;
            }
        }
    }

    if (cache_feature_transforms.empty()) return;

    const Index batch_size = ssize(sample_indices);

    for (Index j = 0; j < features_number; ++j)
    {
        const Index column = feature_indices[size_t(j)];
        const ScalerMethod method = cache_feature_transforms[size_t(column)];

        if (method == ScalerMethod::None) continue;

        const Descriptives& desc = cache_feature_descriptives[size_t(column)];

        for (Index i = 0; i < batch_size; ++i)
        {
            float& value = output[i * features_number + j];
            value = scale_value(method, desc, value);
        }
    }
}

void TabularDataset::compute_cache_descriptives() const
{
    const Index columns_number = cache_columns_number;
    const vector<Index> used_sample_indices = get_used_sample_indices();

    vector<float> minimums(size_t(columns_number), numeric_limits<float>::infinity());
    vector<float> maximums(size_t(columns_number), -numeric_limits<float>::infinity());
    vector<double> sums(size_t(columns_number), 0.0);
    vector<double> squared_sums(size_t(columns_number), 0.0);
    vector<Index> counts(size_t(columns_number), 0);

    vector<float> row(static_cast<size_t>(columns_number));

    for (const Index sample_index : used_sample_indices)
    {
        cache_reader.read_at(row.data(), size_t(columns_number) * sizeof(float),
                             uint64_t(sample_index) * uint64_t(columns_number) * sizeof(float));

        for (Index j = 0; j < columns_number; ++j)
        {
            const float value = row[size_t(j)];

            if (isnan(value)) continue;

            if (value < minimums[size_t(j)]) minimums[size_t(j)] = value;
            if (value > maximums[size_t(j)]) maximums[size_t(j)] = value;

            sums[size_t(j)] += value;
            squared_sums[size_t(j)] += double(value) * double(value);
            ++counts[size_t(j)];
        }
    }

    cache_feature_descriptives.assign(size_t(columns_number), Descriptives());

    for (Index j = 0; j < columns_number; ++j)
    {
        const Index count = counts[size_t(j)];

        if (count == 0)
        {
            minimums[size_t(j)] = 0;
            maximums[size_t(j)] = 0;
        }

        const double mean = count > 0 ? sums[size_t(j)] / double(count) : 0.0;

        double standard_deviation = 0.0;

        if (count > 1)
        {
            const double variance = (squared_sums[size_t(j)] - sums[size_t(j)] * sums[size_t(j)] / double(count))
                                  / (double(count) - 1.0);
            standard_deviation = sqrt(max(0.0, variance));
        }

        cache_feature_descriptives[size_t(j)].set(minimums[size_t(j)],
                                                  maximums[size_t(j)],
                                                  float(mean),
                                                  float(standard_deviation));
    }
}

void TabularDataset::fill_features(const vector<Index>& sample_indices, const vector<Index>& feature_indices,
                                   float* output, int contiguous) const
{
    if (storage_mode == StorageMode::BinaryFile)
        fill_from_binary_cache(sample_indices, feature_indices, output, contiguous);
    else
        fill_tensor_data(data, sample_indices, feature_indices, output, contiguous);
}

void TabularDataset::fill_inputs(const vector<Index>& sample_indices, const vector<Index>& input_indices,
                                 float* input_data, bool, int contiguous) const
{
    fill_features(sample_indices, input_indices, input_data, contiguous);
}

void TabularDataset::fill_decoder(const vector<Index>& sample_indices, const vector<Index>& decoder_indices,
                                  float* decoder_data, bool, int contiguous) const
{
    fill_features(sample_indices, decoder_indices, decoder_data, contiguous);
}

void TabularDataset::fill_targets(const vector<Index>& sample_indices, const vector<Index>& target_indices,
                                  float* target_data, bool, int contiguous) const
{
    fill_features(sample_indices, target_indices, target_data, contiguous);
}

void TabularDataset::infer_variable_types_from_data()
{
    Index feature_index = 0;

    const Index variables_number = get_variables_number();

    for (Index variable_index = 0; variable_index < variables_number; ++variable_index)
    {
        Variable& variable = variables[variable_index];
        const Index advance = variable.get_feature_count();

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
        }
        else if ((variable.type == VariableType::Binary || variable.type == VariableType::Categorical)
              && variable.get_categories_number() == 1)
            variable.set(variable.name, "None", VariableType::Constant);

        feature_index += advance;
    }
}

void TabularDataset::set_binary_variables() { infer_variable_types_from_data(); }

void TabularDataset::resize_data_from_JSON(Index samples_number)
{
    if (storage_mode == StorageMode::BinaryFile || variables.empty())
        data.resize(0, 0);
    else
        data = MatrixR::Zero(samples_number, get_features_number());
}

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
    set_data_path(new_data_path);

    set_separator_string(new_separator);

    set_has_header(new_has_header);

    set_has_ids(new_has_ids);

    set_codification(new_codification);

    read_csv();

    set_default_variable_scalers();

    set_default_variable_roles();

    missing_values_method = MissingValuesMethod::Mean;

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
            const float correlation_value = correlations(i, j).coefficient;

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

vector<string> TabularDataset::unuse_least_correlated_variables(const Index inputs_to_keep)
{
    vector<string> unused_variables;

    const Index input_variables_number = get_variables_number("Input");

    if (inputs_to_keep <= 0 || input_variables_number <= inputs_to_keep)
        return unused_variables;

    const Tensor<Correlation, 2> correlations = calculate_input_target_variable_pearson_correlations();

    const Index target_variables_number = get_variables_number("Target");

    const vector<Index> input_variable_indices = get_variable_indices("Input");

    // Rank each input by its strongest absolute correlation with any target;
    // inputs whose correlations are all NaN rank last.

    vector<pair<float, Index>> ranking(input_variables_number);

    for (Index i = 0; i < input_variables_number; ++i)
    {
        float best_correlation = -1.0f;

        for (Index j = 0; j < target_variables_number; ++j)
        {
            const float correlation_value = correlations(i, j).coefficient;

            if (!isnan(correlation_value))
                best_correlation = max(best_correlation, abs(correlation_value));
        }

        ranking[i] = { best_correlation, i };
    }

    stable_sort(ranking.begin(), ranking.end(),
                [](const pair<float, Index>& a, const pair<float, Index>& b)
                { return a.first > b.first; });

    for (Index rank = inputs_to_keep; rank < input_variables_number; ++rank)
    {
        Variable& variable = variables[input_variable_indices[ranking[rank].second]];

        if (variable.role == VariableRole::None) continue;

        variable.set_role("None");
        unused_variables.push_back(variable.name);
    }

    resize_input_shape(get_features_number("Input"));
    set_shape("Target", { get_features_number("Target") });

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
            feature_index += variable.get_feature_count();
            continue;
        }

        using enum VariableType;
        switch (variable.type)
        {

        case Numeric:
        case Integer:
        case Constant:
        {
            VectorR variable_data(used_samples_number);

            for (Index j = 0; j < used_samples_number; ++j)
                variable_data(j) = data(used_sample_indices[j], feature_index);

            histograms[used_variable_index++] = histogram(variable_data, bins_number);

            ++feature_index;
        }
        break;

        case Categorical:
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

        case Binary:
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

        case DateTime:

            ++feature_index;

            break;

        case None:
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

        if ((variable.type == VariableType::Numeric || variable.type == VariableType::Binary || variable.type == VariableType::Integer)
            && variable.role != VariableRole::None)
            box_plots[i] = box_plot(data.col(feature_index), used_sample_indices);

        feature_index += variable.get_feature_count();
    }

    return box_plots;
}

vector<Descriptives> TabularDataset::calculate_feature_descriptives() const
{
    if (storage_mode == StorageMode::BinaryFile)
    {
        if (cache_feature_descriptives.empty()) compute_cache_descriptives();

        return cache_feature_descriptives;
    }

    return descriptives(data);
}

vector<Descriptives> TabularDataset::calculate_feature_descriptives(const string& variable_role) const
{
    if (storage_mode == StorageMode::BinaryFile)
    {
        if (cache_feature_descriptives.empty()) compute_cache_descriptives();

        const vector<Index> feature_indices = get_feature_indices(variable_role);

        vector<Descriptives> result(feature_indices.size());

        for (size_t i = 0; i < feature_indices.size(); ++i)
            result[i] = cache_feature_descriptives[size_t(feature_indices[i])];

        return result;
    }

    return calculate_feature_descriptives(variable_role, get_used_sample_indices());
}

vector<Descriptives> TabularDataset::calculate_feature_descriptives(const string& variable_role,
                                                                    const vector<Index>& sample_indices) const
{
    const vector<Index> input_feature_indices = get_feature_indices(variable_role);

    return descriptives(data, sample_indices, input_feature_indices);
}

// Used samples whose value in feature `column` is on the requested side of the
// binary split (1 = positive, 0 = negative). NaN falls on neither side, so both
// comparisons exclude it. Operates on the in-memory `data` matrix -- callers load
// it first (loadMatrixFromBinary switches the dataset to Matrix storage).
vector<Index> TabularDataset::filter_used_samples_by_column(Index column, bool positive) const
{
    const vector<Index> used_sample_indices = get_used_sample_indices();

    vector<Index> filtered;
    filtered.reserve(used_sample_indices.size());

    for (const Index sample_index : used_sample_indices)
    {
        const float value = data(sample_index, column);

        if (positive ? (value > 0.5f) : (value < 0.5f))
            filtered.push_back(sample_index);
    }

    return filtered;
}

// Descriptives of every input feature over the samples where the (single, binary)
// target is positive. Size == number of input features, matching the order the
// engine walks input variables when building the positive/negative statistics table.
vector<Descriptives> TabularDataset::calculate_variable_descriptives_positive_samples() const
{
    const vector<Index> target_feature_indices = get_feature_indices("Target");
    if (target_feature_indices.empty()) return {};

    return descriptives(data,
                        filter_used_samples_by_column(target_feature_indices[0], true),
                        get_feature_indices("Input"));
}

vector<Descriptives> TabularDataset::calculate_variable_descriptives_negative_samples() const
{
    const vector<Index> target_feature_indices = get_feature_indices("Target");
    if (target_feature_indices.empty()) return {};

    return descriptives(data,
                        filter_used_samples_by_column(target_feature_indices[0], false),
                        get_feature_indices("Input"));
}

// Multi-class variant: `class_index` is the one-hot feature index of one target
// category; describe the input features over the samples belonging to that class.
vector<Descriptives> TabularDataset::calculate_variable_descriptives_categories(Index class_index) const
{
    return descriptives(data,
                        filter_used_samples_by_column(class_index, true),
                        get_feature_indices("Input"));
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

            if (correlations(i, j).coefficient > 1.0f - EPSILON)
                correlations(i, j).coefficient = 1.0f;

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

    using enum ScalerMethod;
    switch (method)
    {
    case None:
        break;
    case MinimumMaximum:
        if (unscale)
            unscale_minimum_maximum(map, feature_index, desc);
        else
            scale_minimum_maximum(map, feature_index, desc);
        break;
    case MeanStandardDeviation:
        if (unscale)
            unscale_mean_standard_deviation(map, feature_index, desc);
        else
            scale_mean_standard_deviation(map, feature_index, desc);
        break;
    case StandardDeviation:
        if (unscale)
            unscale_standard_deviation(map, feature_index, desc);
        else
            scale_standard_deviation(map, feature_index, desc);
        break;
    case Logarithm:
        if (unscale)
            unscale_logarithmic(map, feature_index);
        else
            scale_logarithmic(map, feature_index);
        break;
    case ImageMinMax:
        if (unscale) unscale_image_minimum_maximum(map, feature_index);
        break;
    }

}

vector<Descriptives> TabularDataset::scale_data()
{
    const Index features_number = get_features_number();

    const vector<Descriptives> feature_descriptives = calculate_feature_descriptives();

    #pragma omp parallel for
    for (Index i = 0; i < features_number; ++i)
        apply_scaler(i, scaler_method_to_string(variables[get_variable_index(i)].scaler), feature_descriptives[i], false);

    return feature_descriptives;
}

vector<Descriptives> TabularDataset::scale_features(const string& variable_role)
{
    const vector<Index> feature_indices = get_feature_indices(variable_role);
    const vector<string> scalers = get_feature_scalers(variable_role);

    if (storage_mode == StorageMode::BinaryFile)
    {
        // Descriptives from the raw cache (streaming). Must use the
        // BinaryFile-aware single-arg overload: the (role, sample_indices)
        // overload reads the data matrix, which is empty in BinaryFile mode
        // (null deref during training's set_scaling).
        const vector<Descriptives> feature_descriptives = calculate_feature_descriptives(variable_role);

        if (cache_feature_transforms.empty())
            cache_feature_transforms.assign(size_t(cache_columns_number), ScalerMethod::None);

        for (size_t i = 0; i < feature_indices.size(); ++i)
            cache_feature_transforms[size_t(feature_indices[i])] = string_to_scaler_method(scalers[i]);

        return feature_descriptives;
    }

    vector<Index> statistic_sample_indices = get_sample_indices("Training");
    if (statistic_sample_indices.empty())
        statistic_sample_indices = get_used_sample_indices();

    const vector<Descriptives> feature_descriptives =
        calculate_feature_descriptives(variable_role, statistic_sample_indices);

    #pragma omp parallel for
    for (Index i = 0; i < Index(feature_indices.size()); ++i)
        apply_scaler(feature_indices[i], scalers[i], feature_descriptives[i], false);

    return feature_descriptives;
}

void TabularDataset::unscale_features(const string& variable_role,
                                            const vector<Descriptives>& feature_descriptives)
{
    const vector<Index> feature_indices = get_feature_indices(variable_role);
    const vector<string> scalers = get_feature_scalers(variable_role);

    if (storage_mode == StorageMode::BinaryFile)
    {
        if (cache_feature_transforms.empty()) return;

        for (const Index feature_index : feature_indices)
            cache_feature_transforms[size_t(feature_index)] = ScalerMethod::None;

        return;
    }

    #pragma omp parallel for
    for (Index i = 0; i < Index(feature_indices.size()); ++i)
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
    if (src->has("StorageMode"))
        set_storage_mode(read_json_string(src, "StorageMode"));

    // A freshly-created model (before the first import) carries only a
    // DataSource: Variables/Samples/MissingValues/PreviewData are produced by
    // read_csv. Read them only when present so load() works pre-import.
    if (const Json* variables_element = root->find("Variables"))
        variables_from_JSON(variables_element);
    if (const Json* samples_element = root->find("Samples"))
        samples_from_JSON(samples_element);
    if (const Json* missing_values_element = root->find("MissingValues"))
        missing_values_from_JSON(missing_values_element);
    if (const Json* preview_data_element = root->find("PreviewData"))
        preview_data_from_JSON(preview_data_element);

    set_display(read_json_bool(root, "Display"));

    if (storage_mode == StorageMode::BinaryFile)
    {
        const vector<vector<Index>> feature_indices = get_feature_indices();
        cache_columns_number = feature_indices.empty() ? 0 : feature_indices.back().back() + 1;
        cache_feature_descriptives.clear();
        cache_feature_transforms.clear();
        cache_feature_replacement.clear();

        cache_path = cache_file_path();

        // Tolerate a missing cache: embedding applications may re-import the
        // data source (regenerating the cache) after loading the model.
        if (filesystem::exists(cache_path))
        {
            cache_reader.open(cache_path);

            const uint64_t expected_bytes =
                uint64_t(get_samples_number()) * uint64_t(cache_columns_number) * sizeof(float);

            throw_if(cache_reader.file_size() != expected_bytes,
                     format("Binary data cache size mismatch for {} (got {} bytes, expected {}).",
                            cache_path.string(), cache_reader.file_size(), expected_bytes));
        }
    }

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

        const auto target_column = data.col(target_feature_indices[0]).head(samples_number).array();

        class_distribution(0) = (target_column < 0.5f).count();
        class_distribution(1) = (target_column >= 0.5f).count();
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
            feature_index += variable.get_feature_count();
            continue;
        }

        if (variable.is_categorical() || variable.is_binary() || variable.type == VariableType::DateTime)
        {
            feature_index += variable.get_feature_count();
            ++used_feature_index;
            continue;
        }

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
            const Index sample_index = sample_indices[j];
            const float value = data(sample_index, feature_index);

            if (value < lower || value > upper)
            {
                return_values[0][j] = 1;
                ++variables_outliers;

                if (replace_with_nan)
                {
                    data(sample_index, feature_index) = QUIET_NAN;
                }
            }
        }

        return_values[1][used_feature_index] = variables_outliers;

        ++feature_index;
        ++used_feature_index;
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

    const vector<Index>& outliers_mask = outliers_indices[0];
    const vector<Index> sample_indices = get_used_sample_indices();

    vector<Index> outliers_samples;
    outliers_samples.reserve(outliers_mask.size());

    for (size_t j = 0; j < outliers_mask.size(); ++j)
        if (outliers_mask[j] > 0)
            outliers_samples.push_back(sample_indices[j]);

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
            const float first_term = 1.0f - value;
            const float second_term = data(i, j + 1) - value * value;

            rosenbrock += first_term * first_term + 100.0f * second_term * second_term;
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

static float parse_float_or_nan(string_view token)
{
    float value;
    auto [ptr, ec] = from_chars(token.data(), token.data() + token.size(), value);
    return (ec == errc{} && ptr == token.data() + token.size()) ? value : NAN;
}

static bool is_missing_token(string_view token, string_view missing_label)
{
    return token.empty() || token == missing_label;
}

static void parse_numeric_token(float* row, Index feature_index,
                         string_view token, string_view missing_label)
{
    row[feature_index] = is_missing_token(token, missing_label) ? NAN : parse_float_or_nan(token);
}

static void parse_datetime_token(float* row, Index feature_index,
                          string_view token, string_view missing_label,
                          Index gmt, const DateFormat& date_format)
{
    if (is_missing_token(token, missing_label))
    {
        row[feature_index] = NAN;
        return;
    }

    const time_t timestamp = date_to_timestamp(string(token), gmt, date_format);
    throw_if(timestamp == -1, "Date format is unsupported or date is prior to 1970.");
    row[feature_index] = timestamp;
}

static void parse_categorical_token(float* row, const vector<Index>& feature_indices,
                             string_view token, string_view missing_label,
                             const unordered_map<string_view, Index>& category_map)
{
    if (is_missing_token(token, missing_label))
        for (const Index cat_index : feature_indices)
            row[cat_index] = NAN;
    else
    {
        auto it = category_map.find(token);
        if (it != category_map.end())
            row[feature_indices[it->second]] = 1;
    }
}

static void parse_binary_token(float* row, Index feature_index,
                        string_view token, string_view missing_label,
                        const vector<string>& categories)
{
    if (const bool is_positive = contains(positive_words, token); is_positive || contains(negative_words, token))
        row[feature_index] = is_positive ? 1 : 0;
    else
    {
        if (is_missing_token(token, missing_label))
            row[feature_index] = NAN;
        else if (!categories.empty() && token == categories[0])
            row[feature_index] = 0;
        else if (categories.size() > 1 && token == categories[1])
            row[feature_index] = 1;
        else
            row[feature_index] = parse_float_or_nan(token);
    }
}

static DateFormat infer_dataset_date_format(const vector<Variable>& variables,
                                     const vector<string_view>& sample_lines,
                                     char file_separator,
                                     bool has_sample_ids,
                                     const string& missing_values_label,
                                     bool has_quotes)
{
    const bool any_datetime = ranges::any_of(variables,
        [](const Variable& v) { return v.type == VariableType::DateTime; });

    if (!any_datetime)
        return Auto;

    static const regex date_re(R"((\d{1,2})[-/.](\d{1,2})[-/.](\d{4}).*)");

    const size_t id_offset = has_sample_ids ? 1 : 0;

    string scratch;
    vector<string_view> row;
    for (const string_view line : sample_lines)
    {
        get_token_views_maybe_quoted(line, file_separator, has_quotes, scratch, row);

        for (size_t col_index = 0; col_index < variables.size(); ++col_index)
        {
            if (variables[col_index].type != VariableType::DateTime)
                continue;

            const size_t token_index = col_index + id_offset;
            if (token_index >= row.size())
                continue;

            const string_view token = row[token_index];

            if (is_missing_token(token, missing_values_label))
                continue;

            cmatch date_parts;
            if (regex_match(token.data(), token.data() + token.size(), date_parts, date_re))
            {
                const int part1 = stoi(date_parts[1].str());
                const int part2 = stoi(date_parts[2].str());

                if (part1 > 12)
                    return Dmy;
                if (part2 > 12)
                    return Mdy;
            }
        }
    }

    return Auto;
}

void TabularDataset::read_csv()
{
    const string separator_string = get_separator_string();

    CsvReader::Configuration configuration;
    configuration.separator = separator_string.empty() ? ',' : separator_string[0];
    configuration.line_validator = [this](string_view line) { check_separators(line); };

    CsvReader::Result parsed = CsvReader(configuration).read(data_path);
    const char file_separator = parsed.separator;
    const bool has_quotes = parsed.has_quotes;
    vector<string_view>& lines = parsed.lines;

    // Buffers reutilizables para la limpieza de comillas por linea (solo se usan
    // en ficheros con comillas; con mmap zero-copy sustituyen a la copia global).
    // Se usan buffers SEPARADOS por seccion porque las vistas devueltas apuntan al
    // scratch: header_tokens debe sobrevivir hasta set_variable_names, asi que no
    // puede compartir buffer con los bucles posteriores.
    string header_scratch;

    throw_if(lines.empty(),
             format("File {} is empty or contains no valid data rows.", data_path.string()));

    read_data_file_preview(lines, file_separator, has_quotes);

    const vector<string_view> header_tokens = get_token_views_maybe_quoted(lines[0], file_separator, has_quotes, header_scratch);
    if (has_header)
    {
        throw_if(has_numbers(header_tokens),
                 "Some header names are numeric.");

        lines.erase(lines.begin());
    }

    throw_if(lines.empty(),
             "Data file only contains a header.");

    const Index samples_number = lines.size();

    auto is_missing = [&](string_view t) { return is_missing_token(t, missing_values_label); };

    if (!has_sample_ids && samples_number > 0)
    {
        // unordered_set<string> (dueno) y no <string_view>: con comillas los tokens
        // apuntan a un scratch reutilizado, cuyas vistas no sobreviven a la
        // iteracion; guardamos el valor limpio de la columna 0 en propiedad.
        unordered_set<string> unique_elements;
        string id_scratch;

        bool possible_id = true;
        bool is_numeric_column = true;
        bool is_date_column = true;
        Index date_check_count = 0;
        const Index max_date_checks = 20;

        for (const string_view line : lines)
        {
            const string_view token = first_token_maybe_quoted(line, file_separator, has_quotes, id_scratch);

            if (!unique_elements.emplace(token).second)
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

    const Index columns_number = ssize(header_tokens);
    const Index id_offset = has_sample_ids ? 1 : 0;
    throw_if(columns_number <= id_offset,
             "Data file contains no variables.");

    const Index variables_number = columns_number - id_offset;
    variables.resize(variables_number);

    if (has_header)
        set_variable_names(vector<string>(header_tokens.begin() + id_offset, header_tokens.end()));
    else
        set_default_variable_names();

    infer_column_types(lines, file_separator, has_quotes);

    const DateFormat date_format = infer_dataset_date_format(variables, lines, file_separator, has_sample_ids, missing_values_label, has_quotes);

    for (Variable& variable : variables)
        if (variable.is_categorical() && variable.get_categories_number() == 2)
            variable.type = VariableType::Binary;

    sample_roles.resize(samples_number, SampleRole::Training);
    sample_ids.resize(samples_number);

    const vector<vector<Index>> all_feature_indices = get_feature_indices();
    const Index feature_columns_number = all_feature_indices.empty() ? 0 : all_feature_indices.back().back() + 1;

    const bool binary_storage = storage_mode == StorageMode::BinaryFile;

    FileWriter cache_writer;
    vector<float> row_values;

    if (binary_storage)
    {
        cache_path = cache_file_path();
        filesystem::create_directories(cache_path.parent_path());
        cache_writer.open(cache_path.string() + ".tmp");

        cache_columns_number = feature_columns_number;
        cache_feature_descriptives.clear();
        cache_feature_transforms.clear();
        cache_feature_replacement.clear();
        row_values.resize(size_t(feature_columns_number));
    }
    else
        data = MatrixR::Zero(samples_number, feature_columns_number);

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

    // Numeric type refinement is accumulated while parsing because in BinaryFile
    // mode there is no data matrix to re-scan afterwards.
    struct NumericColumnValues { bool has_value = false; float first_value = 0.0f; bool constant = true; bool zero_one = true; };
    vector<NumericColumnValues> numeric_column_values(variables_number);

    // ------------------------------------------------------------------
    // Bucle principal de escritura paralelizado con OpenMP.
    //
    // Diseno race-free y byte-identico al secuencial:
    //  - Se paraleliza SOLO el troceo (get_token_views) y el parseo de celdas,
    //    que es la parte cara. Cada fila escribe en su propio slot del buffer
    //    (offset distinto por fila) -> sin escrituras compartidas.
    //  - Los contadores de faltantes son sumas conmutativas -> reduccion por
    //    hilo y suma final en orden.
    //  - El refinamiento numerico (numeric_column_values) depende del ORDEN de
    //    las filas, asi que se hace SECUENCIAL, en orden global de filas, tras
    //    el parseo -> resultado identico al secuencial.
    //  - category_maps / variables / all_feature_indices son solo-lectura ->
    //    seguros de compartir entre hilos.
    // ------------------------------------------------------------------

    // Parseo de una fila ya troceada en `row_tokens` hacia el buffer `row`.
    // Solo lee estado compartido (const) -> seguro entre hilos.
    auto parse_row = [&](float* row, const vector<string_view>& row_tokens)
    {
        for (Index variable_index = 0; variable_index < variables_number; ++variable_index)
        {
            const Variable& variable = variables[variable_index];
            const string_view token = row_tokens[variable_index + id_offset];
            const vector<Index>& feature_indices = all_feature_indices[variable_index];

            using enum VariableType;
            switch (variable.type)
            {
            case None:
            case Constant:
                break;
            case Numeric:
            case Integer:
                parse_numeric_token(row, feature_indices[0], token, missing_values_label);
                break;
            case DateTime:
                parse_datetime_token(row, feature_indices[0], token, missing_values_label, gmt, date_format);
                break;
            case Categorical:
                parse_categorical_token(row, feature_indices, token, missing_values_label, category_maps[variable_index]);
                break;
            case Binary:
                parse_binary_token(row, feature_indices[0], token, missing_values_label, variable.categories);
                break;
            }
        }
    };

    // Refinamiento numerico secuencial de una fila ya escrita en `row`. DEBE
    // recorrerse en orden global de filas para ser identico al secuencial.
    auto refine_numeric = [&](const float* row)
    {
        for (Index variable_index = 0; variable_index < variables_number; ++variable_index)
        {
            if (variables[variable_index].type != VariableType::Numeric) continue;

            NumericColumnValues& column = numeric_column_values[variable_index];
            const float value = row[all_feature_indices[variable_index][0]];

            if (isnan(value)) continue;

            if (!column.has_value)
            {
                column.has_value = true;
                column.first_value = value;
            }
            else if (abs(value - column.first_value) > numeric_limits<float>::min())
                column.constant = false;

            if (value != 0.0f && value != 1.0f)
                column.zero_one = false;
        }
    };

    // Deteccion de faltantes de una fila -> acumula en contadores POR HILO.
    // Replica exactamente la logica secuencial: has_missing_values recorre
    // TODOS los tokens (incluido el id en la columna 0); el conteo por variable
    // arranca en id_offset. Devuelve si la fila tiene algun faltante.
    auto count_missing = [&](const vector<string_view>& row_tokens,
                             Index& th_rows_mv, Index& th_mv, vector<Index>& th_var_mv)
    {
        bool row_has_missing = false;
        for (const string_view t : row_tokens)
            if (is_missing(t)) { row_has_missing = true; break; }

        if (row_has_missing)
        {
            ++th_rows_mv;
            for (size_t k = id_offset; k < row_tokens.size(); ++k)
            {
                if (Index(k - id_offset) >= variables_number) break;

                if (is_missing(row_tokens[k]))
                {
                    ++th_mv;
                    th_var_mv[k - id_offset]++;
                }
            }
        }

        return row_has_missing;
    };

    // No se puede lanzar excepciones dentro de una region #pragma omp; se
    // registra la fila mala de MENOR indice y se lanza tras terminar todo.
    bool bad_row = false;
    Index bad_row_index = samples_number;
    Index bad_row_cols = 0;

    // Errores de parseo (p.ej. parse_datetime_token con una fecha malformada)
    // pueden lanzar. No se puede propagar una excepcion desde una region
    // #pragma omp (terminaria el proceso), asi que se captura y se difiere el
    // de MENOR indice de fila, igual que el error de numero de columnas.
    bool parse_error = false;
    Index parse_error_index = samples_number;
    string parse_error_msg;

    if (binary_storage)
    {
        const Index CHUNK = 16384;
        vector<float> chunk_buffer;

        for (Index base = 0; base < samples_number; base += CHUNK)
        {
            const Index end = (base + CHUNK < samples_number) ? base + CHUNK : samples_number;
            const Index n = end - base;

            chunk_buffer.assign(size_t(n) * feature_columns_number, 0.0f);

            Index chunk_rows_mv = 0, chunk_mv = 0;
            vector<Index> chunk_var_mv(variables_number, 0);

            #pragma omp parallel
            {
                string th_scratch;
                vector<string_view> th_tokens;
                vector<Index> th_var_mv(variables_number, 0);
                Index th_rows_mv = 0, th_mv = 0;

                #pragma omp for schedule(static) nowait
                for (Index i = base; i < end; ++i)
                {
                    get_token_views_maybe_quoted(lines[i], file_separator, has_quotes, th_scratch, th_tokens);

                    float* row = chunk_buffer.data() + size_t(i - base) * feature_columns_number;

                    const bool row_has_missing = count_missing(th_tokens, th_rows_mv, th_mv, th_var_mv);

                    if (has_sample_ids)
                        sample_ids[i] = string(th_tokens[0]);

                    if (row_has_missing)
                        sample_roles[i] = SampleRole::None;

                    if (Index(ssize(th_tokens)) < variables_number + id_offset)
                    {
                        #pragma omp critical
                        {
                            if (i < bad_row_index)
                            {
                                bad_row = true;
                                bad_row_index = i;
                                bad_row_cols = ssize(th_tokens);
                            }
                        }
                        continue;
                    }

                    try
                    {
                        parse_row(row, th_tokens);
                    }
                    catch (const exception& e)
                    {
                        #pragma omp critical
                        {
                            if (i < parse_error_index)
                            {
                                parse_error = true;
                                parse_error_index = i;
                                parse_error_msg = e.what();
                            }
                        }
                    }
                }

                #pragma omp critical
                {
                    chunk_rows_mv += th_rows_mv;
                    chunk_mv += th_mv;
                    for (Index v = 0; v < variables_number; ++v)
                        chunk_var_mv[v] += th_var_mv[v];
                }
            }

            rows_missing_values_number += chunk_rows_mv;
            missing_values_number += chunk_mv;
            for (Index v = 0; v < variables_number; ++v)
                variables_missing_values_number(v) += chunk_var_mv[v];

            for (Index i = base; i < end; ++i)
                refine_numeric(chunk_buffer.data() + size_t(i - base) * feature_columns_number);

            cache_writer.write(chunk_buffer.data(), size_t(n) * feature_columns_number * sizeof(float));
        }
    }
    else
    {
        Index total_rows_mv = 0, total_mv = 0;
        vector<Index> total_var_mv(variables_number, 0);

        #pragma omp parallel
        {
            string th_scratch;
            vector<string_view> th_tokens;
            vector<Index> th_var_mv(variables_number, 0);
            Index th_rows_mv = 0, th_mv = 0;

            #pragma omp for schedule(static) nowait
            for (Index i = 0; i < samples_number; ++i)
            {
                get_token_views_maybe_quoted(lines[i], file_separator, has_quotes, th_scratch, th_tokens);

                float* row = data.data() + i * feature_columns_number;

                count_missing(th_tokens, th_rows_mv, th_mv, th_var_mv);

                if (has_sample_ids)
                    sample_ids[i] = string(th_tokens[0]);

                if (Index(ssize(th_tokens)) < variables_number + id_offset)
                {
                    #pragma omp critical
                    {
                        if (i < bad_row_index)
                        {
                            bad_row = true;
                            bad_row_index = i;
                            bad_row_cols = ssize(th_tokens);
                        }
                    }
                    continue;
                }

                try
                {
                    parse_row(row, th_tokens);
                }
                catch (const exception& e)
                {
                    #pragma omp critical
                    {
                        if (i < parse_error_index)
                        {
                            parse_error = true;
                            parse_error_index = i;
                            parse_error_msg = e.what();
                        }
                    }
                }
            }

            #pragma omp critical
            {
                total_rows_mv += th_rows_mv;
                total_mv += th_mv;
                for (Index v = 0; v < variables_number; ++v)
                    total_var_mv[v] += th_var_mv[v];
            }
        }

        rows_missing_values_number += total_rows_mv;
        missing_values_number += total_mv;
        for (Index v = 0; v < variables_number; ++v)
            variables_missing_values_number(v) += total_var_mv[v];

        for (Index i = 0; i < samples_number; ++i)
            refine_numeric(data.data() + i * feature_columns_number);
    }

    // Lanzar (ya fuera de la region paralela) el error de MENOR indice de fila,
    // sea de numero de columnas o de parseo (p.ej. fecha malformada).
    if (bad_row && (!parse_error || bad_row_index <= parse_error_index))
        throw runtime_error(format("Row {} has fewer columns than expected ({}).", bad_row_index, bad_row_cols));

    if (parse_error)
        throw runtime_error(format("Row {}: {}", parse_error_index, parse_error_msg));

    if (binary_storage)
    {
        // Re-import: el lector puede tener abierta la cache anterior (se abre
        // al cargar el modelo); Windows no permite reemplazar un fichero
        // abierto (MoveFileEx -> ACCESS_DENIED), asi que cerrar antes.
        cache_reader.close();
        cache_writer.finish_with_rename(cache_path);
        cache_reader.open(cache_path);
    }

    for (Index variable_index = 0; variable_index < variables_number; ++variable_index)
    {
        Variable& variable = variables[variable_index];

        if (variable.type == VariableType::Numeric)
        {
            const NumericColumnValues& column = numeric_column_values[variable_index];

            if (column.constant)
                variable.set(variable.name, "None", VariableType::Constant);
            else if (column.zero_one)
            {
                variable.type = VariableType::Binary;
                variable.categories = { "0", "1" };
            }
        }
        else if ((variable.type == VariableType::Binary || variable.type == VariableType::Categorical)
              && variable.get_categories_number() == 1)
            variable.set(variable.name, "None", VariableType::Constant);
    }

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
    return ranges::any_of(row,
                          [&](string_view t) { return is_missing_token(t, missing_values_label); });
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
    throw_if(!missing_values_element,
             "Missing values element is nullptr.\n");

    missing_values_number = read_json_index(missing_values_element, "MissingValuesNumber");

    if (missing_values_number <= 0) return;

    set_missing_values_method(read_json_string(missing_values_element, "MissingValuesMethod"));

    const string variables_string = read_json_string_fallback(missing_values_element,
        {"VariablesMissingValuesNumber", "RawVariablesMissingValuesNumber"});

    const vector<string> tokens = get_tokens(variables_string, " ");

    variables_missing_values_number.resize(tokens.size());
    for (size_t i = 0; i < tokens.size(); ++i)
        if (!tokens[i].empty())
            variables_missing_values_number(i) = parse_long(tokens[i], "VariablesMissingValuesNumber");

    rows_missing_values_number = parse_long(read_json_string_fallback(missing_values_element,
        {"SamplesMissingValuesNumber", "RowsMissingValuesNumber"}), "SamplesMissingValuesNumber");
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
    const vector<Index> target_feature_indices = get_feature_indices("Target");

    if (used_sample_indices.empty() || used_feature_indices.empty())
        return;

    VectorR replacements = (method == MissingValuesMethod::Mean)
        ? mean(data, used_sample_indices, used_feature_indices)
        : median(data, used_sample_indices, used_feature_indices);

    // A feature that is entirely missing over the used samples has a NaN mean/median,
    // so imputing with it would leave NaN behind and poison every downstream forward
    // pass (this is what made genetic input selection on missing-heavy datasets report
    // NaN errors). Fall back to 0 -- the column carries no information but stays finite.
    for (Index j = 0; j < replacements.size(); ++j)
        if (!isfinite(replacements(j))) replacements(j) = 0.0f;

    const Index samples_number = used_sample_indices.size();
    const Index features_number = used_feature_indices.size();
    const Index target_features_number = target_feature_indices.size();

    for (Index j = 0; j < features_number; ++j)
    {
        const Index current_variable = used_feature_indices[j];

        if (ranges::find(target_feature_indices, current_variable) != target_feature_indices.end())
            continue;

        for (Index i = 0; i < samples_number; ++i)
        {
            const Index current_sample = used_sample_indices[i];

            if (isnan(data(current_sample, current_variable)))
            {
                data(current_sample, current_variable) = replacements(j);
            }
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

void TabularDataset::reuse_input_incomplete_rows_binary()
{
    // BinaryFile strategy: keep the on-disk cache RAW (NaN preserved) and only fix
    // sample roles. A row is usable as long as its target is present -- rows that are
    // missing only inputs are re-used (streaming had unused every incomplete row).
    // Their NaN is dropped pairwise by the correlation code (so correlations use each
    // column's available values and keep the variable type intact) and -- TODO, step
    // 4 -- imputed on the fly when filling training/testing batches. Rows whose target
    // is missing stay unused: a target value cannot be invented.
    if (storage_mode != StorageMode::BinaryFile) return;
    if (cache_columns_number == 0 || !cache_reader.is_open()) return;

    const vector<Index> target_feature_indices = get_feature_indices("Target");

    const Index columns_number = cache_columns_number;
    const Index samples_number = get_samples_number();

    vector<float> row(static_cast<size_t>(columns_number));

    for (Index sample_index = 0; sample_index < samples_number; ++sample_index)
    {
        cache_reader.read_at(row.data(),
                             size_t(columns_number) * sizeof(float),
                             uint64_t(sample_index) * uint64_t(columns_number) * sizeof(float));

        bool target_missing = false;
        for (const Index target_index : target_feature_indices)
            if (isnan(row[size_t(target_index)])) { target_missing = true; break; }

        if (target_missing)
            set_sample_role(sample_index, "None");
        else if (sample_roles[size_t(sample_index)] == SampleRole::None)
            set_sample_role(sample_index, "Training");
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

            Index left_sample_index = 0, right_sample_index = 0;
            float current_sample_position = 0.0f, interpolated_value = 0.0f, left_value = 0.0f, right_value = 0.0f;

            for (Index k = i - 1; k >= 0; k--)
            {
                if (isnan(data(used_sample_indices[k], current_variable))) continue;

                left_sample_index = used_sample_indices[k];
                left_value = data(left_sample_index, current_variable);
                break;
            }

            for (Index k = i + 1; k < samples_number; ++k)
            {
                if (isnan(data(used_sample_indices[k], current_variable))) continue;

                right_sample_index = used_sample_indices[k];
                right_value = data(right_sample_index, current_variable);
                break;
            }

            if (right_sample_index != left_sample_index)
            {
                current_sample_position = float(current_sample);
                interpolated_value = left_value + (current_sample_position - left_sample_index) * (right_value - left_value) / (right_sample_index - left_sample_index);
            }
            else
            {
                interpolated_value = left_value;
            }

            data(current_sample, current_variable) = interpolated_value;
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
    // BinaryFile storage keeps NaN in the on-disk cache and streaming marked every
    // incomplete row unused. There is no data matrix to impute, so operate on the
    // cache directly: Mean/Median fill missing inputs and re-use those rows; Unuse
    // (and Interpolation, unsupported while streaming) keep the streaming default.
    if (storage_mode == StorageMode::BinaryFile)
    {
        // Keep the cache raw (NaN preserved) so correlations do pairwise deletion and
        // variable types stay intact. Unless the method is Unuse, re-use the rows that
        // are missing only inputs (target present); their NaN is handled per consumer
        // (pairwise in correlations; imputed on the fly at batch fill -- step 4). Rows
        // missing the target stay unused.
        using enum MissingValuesMethod;
        if (missing_values_method != Unuse)
            reuse_input_incomplete_rows_binary();

        return;
    }

    using enum MissingValuesMethod;
    switch (missing_values_method)
    {
    case Unuse:
        impute_missing_values_unuse();
        break;

    case Mean:
    case Median:
        impute_missing_values_statistic(missing_values_method);
        break;

    case Interpolation:
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

void TabularDataset::infer_column_types(const vector<string_view>& sample_lines, char file_separator, bool has_quotes)
{
    const Index variables_number = variables.size();
    const size_t total_rows = sample_lines.size();

    if (total_rows == 0) return;

    vector<size_t> row_indices(total_rows);
    iota(row_indices.begin(), row_indices.end(), 0);

    shuffle_vector(row_indices);

    const size_t rows_to_check = min(size_t(100), total_rows);
    const size_t id_offset = has_sample_ids ? 1 : 0;

    vector<vector<string_view>> sampled_tokens(rows_to_check);
    // Backing por fila: sampled_tokens[i] se conserva y se usa despues, asi que
    // con comillas cada fila necesita su propio scratch (no uno reutilizado).
    vector<string> sampled_scratch(rows_to_check);
    for (size_t i = 0; i < rows_to_check; ++i)
        sampled_tokens[i] = get_token_views_maybe_quoted(sample_lines[row_indices[i]], file_separator, has_quotes, sampled_scratch[i]);

    for (Index col_index = 0; col_index < variables_number; ++col_index)
    {
        Variable& variable = variables[col_index];
        variable.type = VariableType::None;

        const size_t token_index = col_index + id_offset;

        for (size_t i = 0; i < rows_to_check; ++i)
        {
            const vector<string_view>& tokens = sampled_tokens[i];

            if (token_index >= tokens.size()) continue;

            const string_view token = tokens[token_index];

            if (is_missing_token(token, missing_values_label)) continue;

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

    const bool any_categorical = ranges::any_of(variables,
        [](const Variable& v) { return v.is_categorical(); });

    if (!any_categorical) return;

    vector<std::set<string>> unique_categories(variables_number);
    const Index n_lines = ssize(sample_lines);

    #pragma omp parallel
    {
        vector<std::set<string>> local(variables_number);
        string cat_scratch;
        vector<string_view> cat_tokens;

        #pragma omp for schedule(static) nowait
        for (Index r = 0; r < n_lines; ++r)
        {
            get_token_views_maybe_quoted(sample_lines[r], file_separator, has_quotes, cat_scratch, cat_tokens);
            for (Index col_index = 0; col_index < variables_number; ++col_index)
            {
                if (!variables[col_index].is_categorical()) continue;
                const size_t token_index = col_index + id_offset;
                if (token_index < cat_tokens.size() && !is_missing_token(cat_tokens[token_index], missing_values_label))
                    local[col_index].emplace(cat_tokens[token_index]);
            }
        }

        #pragma omp critical
        for (Index col_index = 0; col_index < variables_number; ++col_index)
            unique_categories[col_index].insert(local[col_index].begin(), local[col_index].end());
    }

    for (Index col_index = 0; col_index < variables_number; ++col_index)
        if (variables[col_index].is_categorical())
            variables[col_index].categories.assign(
                unique_categories[col_index].begin(), unique_categories[col_index].end());
}

vector<string> TabularDataset::get_feature_scalers(const string& variable_role) const
{
    const vector<Variable> role_variables = get_variables(variable_role);

    vector<string> scalers;
    scalers.reserve(get_features_number(variable_role));

    for (const Variable& var : role_variables)
        scalers.insert(scalers.end(), var.get_feature_count(), scaler_method_to_string(var.scaler));

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

    throw_if(new_scalers.size() != variables_number,
             format("Size of variable scalers({}) has to be the same as variables numbers({}).\n",
                    new_scalers.size(), variables_number));

    for (size_t i = 0; i < variables_number; ++i)
        variables[i].set_scaler(new_scalers[i]);
}

void TabularDataset::set_default_variable_scalers()
{
    for (Variable& variable : variables)
        variable.scaler = (variable.type == VariableType::Numeric || variable.type == VariableType::Integer)
                                  ? ScalerMethod::MeanStandardDeviation
                                  : ScalerMethod::MinimumMaximum;
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
        {"Codification", get_codification_string()},
        {"StorageMode", get_storage_mode_string()}
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
