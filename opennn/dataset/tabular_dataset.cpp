//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   T A B U L A R   D A T A S E T   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "opennn/dataset/tabular_dataset.h"
#include "opennn/core/io_utilities.h"
#include "opennn/core/string_utilities.h"
#include "opennn/core/scaling.h"
#include "opennn/core/tensor_types.h"
#include "opennn/core/random_utilities.h"

namespace opennn
{

namespace
{

constexpr size_t maximum_expanded_data_bytes = size_t(2) * 1024 * 1024 * 1024;

bool looks_like_id_variable(const Variable& variable, const Index samples_number)
{
    if (!variable.is_categorical() || samples_number == 0)
        return false;

    const Index categories_number = ssize(variable.categories);

    if (categories_number * 20 >= samples_number * 19)
        return true;

    string name = variable.name;
    ranges::transform(name, name.begin(), [](unsigned char c) { return char(tolower(c)); });

    const bool id_name = name == "id"
                      || name.ends_with(" id")
                      || name.ends_with("_id")
                      || name.ends_with("-id")
                      || name.ends_with(".id");

    return id_name && categories_number * 10 >= samples_number * 9;
}

}

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
    fill_tensor_data(data, sample_indices, feature_indices, span<float>(this_data.data(), static_cast<size_t>(this_data.size())));
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
    fill_tensor_data(data, row_indices, get_feature_indices(variable_index),
                     span<float>(variable_data.data(), size_t(variable_data.size())));
    return variable_data;
}

bool TabularDataset::has_nan() const
{
    for (Index i = 0; i < data.rows(); ++i)
        if (sample_roles[i] != SampleRole::None && has_nan_row(i))
            return true;

    return false;
}

bool TabularDataset::has_nan_row(Index row_index) const { return data.row(row_index).array().isNaN().any(); }

// One entry per variable, matching what read_csv counts and what
// missing_values_to_JSON writes. The per-feature-column sum this used to
// return gave a categorical variable one entry per one-hot column - so the same
// missing value was counted once per category, and the vector changed length
// depending on which of the two writers had run last.
VectorI TabularDataset::count_nans_per_variable() const
{
    const VectorI per_column = data.array().isNaN().cast<Index>().colwise().sum();

    VectorI per_variable = VectorI::Zero(ssize(variables));

    Index column = 0;

    for (Index i = 0; i < ssize(variables); ++i)
    {
        const Index feature_count = variables[size_t(i)].get_feature_count();

        // Every one-hot column of a categorical is NaN together, so the first
        // column of the variable already carries its count.
        if (feature_count > 0 && column < per_column.size())
            per_variable(i) = per_column(column);

        column += feature_count;
    }

    return per_variable;
}

Index TabularDataset::count_rows_with_nan() const { return data.array().isNaN().rowwise().any().count(); }

Index TabularDataset::count_nan() const { return data.array().isNaN().count(); }

void TabularDataset::set_storage_mode(StorageMode new_storage_mode)
{
    Dataset::set_storage_mode(new_storage_mode);

    if (new_storage_mode != StorageMode::BinaryFile)
    {
        cache_columns_number = 0;
        clear_cache_derived_state();
        cache_reader.close();
    }
}

filesystem::path TabularDataset::cache_file_path() const
{

    if (!cache_path_override.empty())
        return cache_path_override;

    return data_path.parent_path() / ".cache" / (data_path.stem().string() + ".bin");
}

void TabularDataset::set_binary_cache_path(const filesystem::path& new_cache_path)
{
    cache_path_override = new_cache_path;
    cache_reader.close();
    clear_cache_derived_state();
    cache_path = cache_file_path();

    if (storage_mode == StorageMode::BinaryFile && filesystem::exists(cache_path))
    {
        cache_reader.open(cache_path);
        if (cache_columns_number > 0 && get_samples_number() > 0)
            refresh_cache_statistics();
    }
}

void TabularDataset::clear_cache_derived_state()
{
    cache_feature_descriptives.clear();
    cache_transform_descriptives.clear();
    cache_feature_transforms.clear();
    cache_feature_replacement.clear();
}

void TabularDataset::refresh_cache_statistics()
{
    cache_feature_descriptives = compute_descriptives_streaming(get_used_sample_indices());

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

void TabularDataset::on_used_samples_changed()
{
    if (storage_mode == StorageMode::BinaryFile
        && cache_columns_number > 0
        && cache_reader.is_open())
        refresh_cache_statistics();
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

    throw_if(ssize(cache_feature_replacement) != columns_number,
             "TabularDataset: binary-cache statistics are not initialized.");

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

    vector<float> row_buffer(contiguous ? 0 : size_t(columns_number));

    for (Index i = 0; i < ssize(sample_indices); ++i)
    {
        const Index row = sample_indices[size_t(i)];
        float* const dst = output + i * features_number;

        if (!contiguous)
        {
            const uint64_t offset = uint64_t(row) * uint64_t(columns_number) * sizeof(float);
            cache_reader.read_at(span(row_buffer), offset);

            ranges::transform(feature_indices, dst, [&](const Index column)
            {
                const float value = row_buffer[size_t(column)];
                return isnan(value) ? cache_feature_replacement[size_t(column)] : value;
            });
            continue;
        }

        const uint64_t offset =
            (uint64_t(row) * uint64_t(columns_number) + uint64_t(first_column)) * sizeof(float);
        cache_reader.read_at(span(dst, size_t(features_number)), offset);

        for (Index j = 0; j < features_number; ++j)
            if (isnan(dst[j])) dst[j] = cache_feature_replacement[size_t(first_column + j)];
    }

    if (cache_feature_transforms.empty()) return;

    const Index batch_size = ssize(sample_indices);

    for (Index j = 0; j < features_number; ++j)
    {
        const Index column = feature_indices[size_t(j)];
        const ScalerMethod method = cache_feature_transforms[size_t(column)];

        if (method == ScalerMethod::None) continue;

        const Descriptives& desc = cache_transform_descriptives.empty()
                                 ? cache_feature_descriptives[size_t(column)]
                                 : cache_transform_descriptives[size_t(column)];

        for (Index i = 0; i < batch_size; ++i)
        {
            float& value = output[i * features_number + j];
            value = scale_value(method, desc, value);
        }
    }
}

vector<Descriptives> TabularDataset::compute_descriptives_streaming(const vector<Index>& sample_indices) const
{
    const Index columns_number = cache_columns_number;

    vector<float> minimums(size_t(columns_number), POS_INFINITY);
    vector<float> maximums(size_t(columns_number), NEG_INFINITY);
    vector<double> sums(size_t(columns_number), 0.0);
    vector<double> squared_sums(size_t(columns_number), 0.0);
    vector<Index> counts(size_t(columns_number), 0);

    vector<float> row(static_cast<size_t>(columns_number));

    for (const Index sample_index : sample_indices)
    {
        cache_reader.read_at(span(row),
                             uint64_t(sample_index) * uint64_t(columns_number) * sizeof(float));

        for (Index j = 0; j < columns_number; ++j)
        {
            const float value = row[size_t(j)];

            if (isnan(value)) continue;

            minimums[size_t(j)] = std::min(minimums[size_t(j)], value);
            maximums[size_t(j)] = std::max(maximums[size_t(j)], value);

            sums[size_t(j)] += value;
            squared_sums[size_t(j)] += double(value) * double(value);
            ++counts[size_t(j)];
        }
    }

    vector<Descriptives> feature_descriptives(static_cast<size_t>(columns_number));

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

        feature_descriptives[size_t(j)].set(minimums[size_t(j)],
                                            maximums[size_t(j)],
                                            float(mean),
                                            float(standard_deviation));
    }

    return feature_descriptives;
}

void TabularDataset::fill_features(const vector<Index>& sample_indices, const vector<Index>& feature_indices,
                                   float* output, int contiguous) const
{
    if (storage_mode == StorageMode::BinaryFile)
        fill_from_binary_cache(sample_indices, feature_indices, output, contiguous);
    else
        fill_tensor_data(data, sample_indices, feature_indices,
                         span<float>(output, size_t(ssize(sample_indices) * ssize(feature_indices))),
                         contiguous);

    apply_training_scaling(feature_indices, output, ssize(sample_indices));
}

float TabularDataset::apply_training_scaling(Index feature_index, float value) const
{
    if (feature_index < 0
        || size_t(feature_index) >= training_transforms.size())
        return value;

    return training_transforms[size_t(feature_index)].apply(value);
}

void TabularDataset::apply_training_scaling(const vector<Index>& feature_indices,
                                            float* output,
                                            Index samples_number) const
{
    if (training_transforms.empty() || !output) return;

    const Index features_number = ssize(feature_indices);
    for (Index j = 0; j < features_number; ++j)
    {
        const Index feature_index = feature_indices[size_t(j)];
        if (feature_index < 0
            || size_t(feature_index) >= training_transforms.size())
            continue;

        const TrainingTransform& transform = training_transforms[size_t(feature_index)];
        if (!transform.configured) continue;

        for (Index i = 0; i < samples_number; ++i)
        {
            float& value = output[i * features_number + j];
            value = transform.apply(value);
        }
    }
}

void TabularDataset::fill_inputs(const vector<Index>& sample_indices, const vector<Index>& input_indices,
                                 float* input_data, FillMode, int contiguous) const
{
    fill_features(sample_indices, input_indices, input_data, contiguous);
}

void TabularDataset::fill_decoder(const vector<Index>& sample_indices, const vector<Index>& decoder_indices,
                                  float* decoder_data, FillMode, int contiguous) const
{
    fill_features(sample_indices, decoder_indices, decoder_data, contiguous);
}

void TabularDataset::fill_targets(const vector<Index>& sample_indices, const vector<Index>& target_indices,
                                  float* target_data, FillMode, int contiguous) const
{
    fill_features(sample_indices, target_indices, target_data, contiguous);
}

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

    input_shape = { get_features_number(VariableRole::Input) };
    target_shape = { get_features_number(VariableRole::Target) };
}

void TabularDataset::set(const filesystem::path& file_name)
{
    load(file_name);
}

vector<string> TabularDataset::unuse_uncorrelated_variables(const float minimum_correlation)
{
    vector<string> unused_variables;

    const Tensor<Correlation, 2> correlations = calculate_input_target_variable_pearson_correlations();

    const Index input_variables_number = get_variables_number(VariableRole::Input);
    const Index target_variables_number = get_variables_number(VariableRole::Target);

    const vector<Index> input_variable_indices = get_variable_indices(VariableRole::Input);

    for (Index i = 0; i < input_variables_number; ++i)
    {
        const Index input_variable_index = input_variable_indices[i];

        const bool has_significant_correlation =
            ranges::any_of(views::iota(Index(0), target_variables_number), [&](Index j)
            {
                const float correlation_value = correlations(i, j).coefficient;
                return !isnan(correlation_value) && abs(correlation_value) >= minimum_correlation;
            });

        Variable& variable = variables[input_variable_index];

        if (!has_significant_correlation && variable.role != VariableRole::None)
        {
            variable.set_role("None");
            unused_variables.push_back(variable.name);
        }
    }

    resize_input_shape(get_features_number(VariableRole::Input));
    set_shape(VariableRole::Target, { get_features_number(VariableRole::Target) });

    return unused_variables;
}

vector<string> TabularDataset::unuse_least_correlated_variables(const Index inputs_to_keep)
{
    vector<string> unused_variables;

    const Index input_variables_number = get_variables_number(VariableRole::Input);

    if (inputs_to_keep <= 0 || input_variables_number <= inputs_to_keep)
        return unused_variables;

    const Tensor<Correlation, 2> correlations = calculate_input_target_variable_pearson_correlations();

    const Index target_variables_number = get_variables_number(VariableRole::Target);

    const vector<Index> input_variable_indices = get_variable_indices(VariableRole::Input);

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

    ranges::stable_sort(ranking, greater<>{},
                        [](const auto& item) { return item.first; });

    for (Index rank = inputs_to_keep; rank < input_variables_number; ++rank)
    {
        Variable& variable = variables[input_variable_indices[ranking[rank].second]];

        if (variable.role == VariableRole::None) continue;

        variable.set_role("None");
        unused_variables.push_back(variable.name);
    }

    resize_input_shape(get_features_number(VariableRole::Input));
    set_shape(VariableRole::Target, { get_features_number(VariableRole::Target) });

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

            const float abs_r = abs(correlations(i, j).coefficient);
            if (isnan(abs_r)) continue;

            if (abs_r >= maximum_correlation)
                high_corr_counts[i]++;

            sum_of_abs_corr += abs_r;
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

            const float r = correlations(i, j).coefficient;

            if (isnan(r) || !(abs(r) >= maximum_correlation)) continue;

            const Index index_to_remove =
                tie(high_corr_counts[i], mean_abs_corr[i])
                    >= tie(high_corr_counts[j], mean_abs_corr[j]) ? i : j;

            to_be_removed[index_to_remove] = true;
        }
    }

    vector<string> unused_variables;
    for (Index i = 0; i < input_variables_number; ++i)
    {
        if (!to_be_removed[i]) continue;

        Variable& variable = variables[input_variable_indices[i]];

        if (variable.role == VariableRole::None) continue;

        variable.set_role("None");
        unused_variables.push_back(variable.name);
    }

    // The cached input/target shapes are separate from the roles, and every
    // other unuse_* resyncs them here. Skipping it left get_shape("Input")
    // reporting more features than get_feature_indices returns, so Batch::set
    // sized the input slot for a width fill_inputs no longer writes.
    resize_input_shape(get_features_number(VariableRole::Input));
    set_shape(VariableRole::Target, { get_features_number(VariableRole::Target) });

    return unused_variables;
}

vector<Histogram> TabularDataset::calculate_variable_distributions(const Index bins_number) const
{
    require_in_memory_data("TabularDataset::calculate_variable_distributions");

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
            const VectorR variable_data = data(used_sample_indices, feature_index);

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
    require_in_memory_data("TabularDataset::calculate_variables_box_plots");

    const Index variables_number = get_variables_number();

    const vector<Index> used_sample_indices = get_used_sample_indices();

    vector<BoxPlot> box_plots(variables_number);

    Index feature_index = 0;

    for (Index i = 0; i < variables_number; ++i)
    {
        const Variable& variable = variables[i];

        if (is_one_of(variable.type, VariableType::Numeric, VariableType::Binary, VariableType::Integer)
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
        throw_if(ssize(cache_feature_descriptives) != cache_columns_number,
                 "TabularDataset: binary-cache descriptives are not initialized.");
        return cache_feature_descriptives;
    }

    return descriptives(data);
}

vector<Descriptives> TabularDataset::calculate_feature_descriptives(const string& variable_role) const
{
    if (storage_mode == StorageMode::BinaryFile)
    {
        throw_if(ssize(cache_feature_descriptives) != cache_columns_number,
                 "TabularDataset: binary-cache descriptives are not initialized.");
        const vector<Index> feature_indices = get_feature_indices(variable_role);

        vector<Descriptives> result(feature_indices.size());
        ranges::transform(feature_indices, result.begin(),
                          [this](Index feature_index) { return cache_feature_descriptives[size_t(feature_index)]; });

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

vector<Descriptives> TabularDataset::calculate_variable_descriptives_samples(bool positive) const
{
    const vector<Index> target_feature_indices = get_feature_indices(VariableRole::Target);
    if (target_feature_indices.empty()) return {};

    return descriptives(data,
                        filter_used_samples_by_column(target_feature_indices[0], positive),
                        get_feature_indices(VariableRole::Input));
}

vector<Descriptives> TabularDataset::calculate_variable_descriptives_positive_samples() const
{
    return calculate_variable_descriptives_samples(true);
}

vector<Descriptives> TabularDataset::calculate_variable_descriptives_negative_samples() const
{
    return calculate_variable_descriptives_samples(false);
}

vector<Descriptives> TabularDataset::calculate_variable_descriptives_categories(Index class_index) const
{
    return descriptives(data,
                        filter_used_samples_by_column(class_index, true),
                        get_feature_indices(VariableRole::Input));
}

Tensor<Correlation, 2> TabularDataset::calculate_input_target_variable_correlations(
    Correlation (*correlation_function)(const MatrixR&, const MatrixR&),
    const string& method_name) const
{
    if (display) cout << "Calculating " << method_name << " correlations..." << "\n";

    const Index input_variables_number = get_variables_number(VariableRole::Input);
    const Index target_variables_number = get_variables_number(VariableRole::Target);

    const vector<Index> input_variable_indices = get_variable_indices(VariableRole::Input);
    const vector<Index> target_variable_indices = get_variable_indices(VariableRole::Target);

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

MatrixR TabularDataset::calculate_input_target_correlation_values() const
{
    return get_correlation_values(calculate_input_target_variable_pearson_correlations());
}

Tensor<Correlation, 2> TabularDataset::calculate_input_variable_correlations(
    Correlation (*correlation_function)(const MatrixR&, const MatrixR&),
    Correlation::Method method,
    const string& method_name) const
{
    if (display) cout << "Calculating " << method_name << " inputs correlations..." << "\n";

    const vector<Index> input_variable_indices = get_variable_indices(VariableRole::Input);
    const vector<Index> used_sample_indices = get_used_sample_indices();

    const Index input_variables_number = input_variable_indices.size();

    Tensor<Correlation, 2> correlations(input_variables_number, input_variables_number);

    for (Index i = 0; i < input_variables_number; ++i)
    {
        if (display) cout << "Correlation " << i + 1 << " of " << input_variables_number << "\n";

        // Restricted to the used samples, as the input-target twin already is.
        // Reading every row meant samples excluded by filtering, Tukey cleaning
        // or the user still steered the collinearity analysis while being
        // absent from the relevance one.
        const MatrixR input_i = get_variable_data(input_variable_indices[i], used_sample_indices);

        if (is_constant(input_i)) continue;

        correlations(i, i).set_perfect();
        correlations(i, i).method = method;

        for (Index j = i + 1; j < input_variables_number; ++j)
        {
            const MatrixR input_j = get_variable_data(input_variable_indices[j], used_sample_indices);

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

FeatureScaling TabularDataset::calculate_used_feature_scaling(VariableRole role) const
{
    const string role_name = variable_role_to_string(role);
    FeatureScaling scaling;
    scaling.descriptives = calculate_feature_descriptives(role_name);

    const vector<string> scaler_names = get_feature_scalers(role_name);
    scaling.scalers.resize(scaler_names.size());
    ranges::transform(scaler_names, scaling.scalers.begin(), string_to_scaler_method);

    return scaling;
}

void TabularDataset::apply_scaler(Index feature_index, const string& scaler, const Descriptives& desc, bool unscale)
{
    const ScalerMethod method = string_to_scaler_method(scaler);

    if (method == ScalerMethod::None)
        return;

    auto column = data.col(feature_index);

    constexpr float min_range = -1.0f;
    constexpr float max_range = 1.0f;

    using enum ScalerMethod;
    switch (method)
    {
    case None:
        break;

    case MinimumMaximum:
        if (unscale)
        {
            column.array() = (column.array() - min_range) / (max_range - min_range)
                           * (desc.maximum - desc.minimum) + desc.minimum;
            break;
        }

        if (desc.maximum - desc.minimum < EPSILON)
            column.setZero();
        else
            column.array() = scale_minimum_maximum_formula(column.array(), desc, min_range, max_range);
        break;

    case MeanStandardDeviation:
        if (!unscale)
        {
            if (desc.standard_deviation > EPSILON)
                column.array() = scale_mean_standard_deviation_formula(column.array(), desc);
            else
                column.setZero();
            break;
        }

        if (desc.standard_deviation < EPSILON)
            column.setConstant(desc.mean);
        else
            column.array() = desc.mean + column.array() * desc.standard_deviation;
        break;

    case StandardDeviation:
        column *= unscale
            ? (abs(desc.standard_deviation) < EPSILON ? 1.0f : desc.standard_deviation)
            : (desc.standard_deviation > EPSILON ? 1.0f / desc.standard_deviation : 0.0f);
        break;

    case Logarithm:
        if (unscale)
        {
            column.array() = column.array().exp();
            break;
        }
        column.array() = column.array().max(EPSILON).log();
        break;

    case ImageMinMax:
        if (unscale) column *= 255.0f;
        else         column /= 255.0f;
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

    const auto statistic_sample_indices = [this]
    {
        vector<Index> indices = get_sample_indices(SampleRole::Training);
        if (indices.empty())
            indices = get_used_sample_indices();
        return indices;
    };

    if (storage_mode == StorageMode::BinaryFile)
    {

        if (cache_transform_descriptives.empty())
            cache_transform_descriptives = compute_descriptives_streaming(statistic_sample_indices());

        if (cache_feature_transforms.empty())
            cache_feature_transforms.assign(size_t(cache_columns_number), ScalerMethod::None);

        for (size_t i = 0; i < feature_indices.size(); ++i)
            cache_feature_transforms[size_t(feature_indices[i])] = string_to_scaler_method(scalers[i]);

        vector<Descriptives> feature_descriptives(feature_indices.size());
        ranges::transform(feature_indices, feature_descriptives.begin(),
                          [this](Index feature_index) { return cache_transform_descriptives[size_t(feature_index)]; });

        return feature_descriptives;
    }

    const vector<Descriptives> feature_descriptives =
        calculate_feature_descriptives(variable_role, statistic_sample_indices());

    #pragma omp parallel for
    for (Index i = 0; i < Index(feature_indices.size()); ++i)
        apply_scaler(feature_indices[i], scalers[i], feature_descriptives[i], false);

    return feature_descriptives;
}

FeatureScaling TabularDataset::prepare_training_scaling(
    VariableRole role,
    const FeatureScaling& requested,
    Index expected_features)
{
    throw_if(!is_one_of(role, VariableRole::Input, VariableRole::Target),
             "TabularDataset supports training scaling only for inputs and targets.");
    throw_if(!(requested.min_range < requested.max_range),
             "TabularDataset training scaling range is invalid.");

    const vector<Index> feature_indices = get_feature_indices(role);
    const vector<Index> input_feature_indices =
        role == VariableRole::Target
        ? get_feature_indices(VariableRole::Input)
        : vector<Index>{};
    throw_if(expected_features != ssize(feature_indices),
             "TabularDataset {} training scaling expects {} features, got {}.",
             variable_role_to_string(role), feature_indices.size(), expected_features);

    vector<Index> statistic_sample_indices = get_sample_indices(SampleRole::Training);
    if (statistic_sample_indices.empty())
        statistic_sample_indices = get_used_sample_indices();

    vector<Descriptives> feature_descriptives;
    if (storage_mode == StorageMode::BinaryFile)
    {
        const vector<Descriptives> all_descriptives =
            compute_descriptives_streaming(statistic_sample_indices);
        feature_descriptives.reserve(feature_indices.size());
        ranges::transform(feature_indices,
                          back_inserter(feature_descriptives),
                          [&](Index feature)
                          {
                              return all_descriptives[size_t(feature)];
                          });
    }
    else
    {
        feature_descriptives =
            calculate_feature_descriptives(variable_role_to_string(role),
                                            statistic_sample_indices);
    }

    const vector<string> scaler_names =
        get_feature_scalers(variable_role_to_string(role));
    vector<ScalerMethod> feature_scalers(scaler_names.size());
    ranges::transform(scaler_names, feature_scalers.begin(), string_to_scaler_method);

    const Index columns_number = storage_mode == StorageMode::BinaryFile
                               ? cache_columns_number
                               : data.cols();
    if (training_transforms.empty())
        training_transforms.resize(size_t(columns_number));

    FeatureScaling effective;
    effective.descriptives.reserve(feature_indices.size());
    effective.scalers.reserve(feature_indices.size());
    effective.min_range = requested.min_range;
    effective.max_range = requested.max_range;

    for (size_t i = 0; i < feature_indices.size(); ++i)
    {
        const size_t feature = size_t(feature_indices[i]);
        TrainingTransform& transform = training_transforms[feature];
        const bool shared_with_input = role == VariableRole::Target
                                    && ranges::find(input_feature_indices, feature_indices[i])
                                       != input_feature_indices.end();
        const bool reuse_input_transform = shared_with_input && transform.configured;

        throw_if(shared_with_input && !reuse_input_transform,
                 "Input-target feature {} requires an input Scaling layer.",
                 feature_indices[i]);

        throw_if(reuse_input_transform
                 && (transform.min_range != requested.min_range
                     || transform.max_range != requested.max_range),
                 "Input-target feature {} cannot use different input and output scaling ranges.",
                 feature_indices[i]);

        if (!reuse_input_transform)
        {
            transform = {feature_descriptives[i],
                         feature_scalers[i],
                         requested.min_range,
                         requested.max_range,
                         true};
        }

        effective.descriptives.push_back(transform.descriptives);
        effective.scalers.push_back(transform.scaler);

    }

    return effective;
}

void TabularDataset::clear_training_scaling() noexcept
{
    training_transforms.clear();

    if (is_device_resident()) disable_device_residency();
}

void TabularDataset::enable_device_residency()
{
    if (training_transforms.empty())
        return Dataset::enable_device_residency();

    MatrixR staged = data;
    for (Index row = 0; row < staged.rows(); ++row)
        for (Index column = 0; column < staged.cols(); ++column)
            staged(row, column) = apply_training_scaling(column, staged(row, column));

    upload_device_matrix(staged);
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

        cache_transform_descriptives.clear();

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
    cache_reader.close();
    cache_columns_number = 0;
    clear_cache_derived_state();

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

    const string decimal_separator_name =
        src->has("DecimalSeparator") ? read_json_string(src, "DecimalSeparator") : "Auto";

    if (decimal_separator_name == "Auto")
        set_number_format_auto();
    else
    {
        const string group_separator_name =
            src->has("ThousandsSeparator") ? read_json_string(src, "ThousandsSeparator") : "None";

        set_number_format(
            {number_format_separator(decimal_separator_name, "DecimalSeparator"),
             group_separator_name == "Auto"
                 ? '\0'
                 : number_format_separator(group_separator_name, "ThousandsSeparator")});
    }

    read_json_blocks(root);

    set_display(read_json_bool(root, "Display"));

    if (storage_mode == StorageMode::BinaryFile)
    {
        const vector<vector<Index>> feature_indices = get_feature_indices();
        cache_columns_number = feature_indices.empty() ? 0 : feature_indices.back().back() + 1;

        cache_path = cache_file_path();

        if (filesystem::exists(cache_path))
        {
            cache_reader.open(cache_path);

            const uint64_t expected_bytes =
                uint64_t(get_samples_number()) * uint64_t(cache_columns_number) * sizeof(float);

            throw_if(cache_reader.file_size() != expected_bytes,
                     "Binary data cache size mismatch for {} (got {} bytes, expected {}).",
                            cache_path.string(), cache_reader.file_size(), expected_bytes);

            refresh_cache_statistics();
        }
    }

    input_shape = { get_features_number(VariableRole::Input) };
    target_shape = { get_features_number(VariableRole::Target) };
}

VectorI TabularDataset::calculate_target_distribution() const
{
    require_in_memory_data("TabularDataset::calculate_target_distribution");

    const Index samples_number = get_samples_number();
    const Index targets_number = get_features_number(VariableRole::Target);
    const vector<Index> target_feature_indices = get_feature_indices(VariableRole::Target);

    VectorI class_distribution;

    if (targets_number == 1)
    {
        class_distribution = VectorI::Zero(2);

        const Index target_feature = target_feature_indices[0];

        for (Index i = 0; i < samples_number; ++i)
        {
            const float value = data(i, target_feature);

            if (isnan(value)) continue;

            ++class_distribution(value >= 0.5f ? 1 : 0);
        }
    }
    else
    {
        class_distribution = VectorI::Zero(targets_number);

        for (Index i = 0; i < samples_number; ++i)
        {
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
    require_in_memory_data("TabularDataset::calculate_Tukey_outliers");

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

        if (!variable.is_used())
        {
            feature_index += variable.get_feature_count();
            continue;
        }

        if (is_one_of(variable.type, VariableType::Categorical, VariableType::Binary,
                      VariableType::DateTime))
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

void TabularDataset::set_data_binary_classification()
{
    const Index samples_number = get_samples_number();
    const Index features_number = get_features_number();

    set_data_random();

    // Serial on purpose: random_bool() draws from one mutex-guarded generator,
    // so a parallel loop both contends on the lock and makes the draw order -
    // and therefore a seeded run - depend on thread scheduling.
    for (Index i = 0; i < samples_number; ++i)
        data(i, features_number - 1) = float(random_bool());

}

static float parse_float_or_nan(string_view token, const NumberFormat& number_format)
{
    float value;
    return parse_real(token, value, number_format) ? value : QUIET_NAN;
}

static bool is_missing_token(string_view token, string_view missing_label)
{
    return token.empty() || token == missing_label;
}

static void parse_numeric_token(float* row, Index feature_index,
                         string_view token, string_view missing_label,
                         const NumberFormat& number_format)
{
    row[feature_index] = is_missing_token(token, missing_label)
                       ? QUIET_NAN
                       : parse_float_or_nan(token, number_format);
}

static void parse_datetime_token(float* row, Index feature_index,
                          string_view token, string_view missing_label,
                          const DateFormat& date_format)
{
    if (is_missing_token(token, missing_label))
    {
        row[feature_index] = QUIET_NAN;
        return;
    }

    const time_t timestamp = date_to_timestamp(token, 0, date_format);
    throw_if(timestamp == -1, "Date format is unsupported or date is prior to 1970.");
    row[feature_index] = timestamp;
}

static void parse_categorical_token(float* row, const vector<Index>& feature_indices,
                             string_view token, string_view missing_label,
                             const unordered_map<string_view, Index>& category_map)
{
    if (is_missing_token(token, missing_label))
        for (const Index cat_index : feature_indices)
            row[cat_index] = QUIET_NAN;
    else
    {
        const auto it = category_map.find(token);
        if (it != category_map.end())
            row[feature_indices[it->second]] = 1;
    }
}

static void parse_binary_token(float* row, Index feature_index,
                        string_view token, string_view missing_label,
                        const vector<string>& categories,
                        const NumberFormat& number_format)
{
    row[feature_index] =
        contains(positive_words, token) ? 1.0f :
        contains(negative_words, token) ? 0.0f :
        is_missing_token(token, missing_label) ? QUIET_NAN :
        !categories.empty() && token == categories[0] ? 0.0f :
        categories.size() > 1 && token == categories[1] ? 1.0f :
        parse_float_or_nan(token, number_format);
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

            const DateFormat detected = detect_date_format(token);
            if (detected != Auto) return detected;
        }
    }

    return Auto;
}

static NumberFormat detect_number_format(const vector<string_view>& lines,
                                         const char file_separator,
                                         const bool has_quotes)
{
    constexpr size_t maximum_rows_to_check = 100;

    const size_t total_rows = lines.size();

    if (total_rows == 0) return {};

    const size_t rows_to_check = min(maximum_rows_to_check, total_rows);

    NumberFormatVotes votes;

    string scratch;
    vector<string_view> tokens;

    for (size_t i = 0; i < rows_to_check; ++i)
    {
        get_token_views_maybe_quoted(
            lines[i * total_rows / rows_to_check],
            file_separator,
            has_quotes,
            scratch,
            tokens);

        for (const string_view token : tokens)
            vote_number_format(token, votes);
    }

    return decide_number_format(votes);
}

void TabularDataset::read_csv()
{
    const string separator_string = get_separator_string();
    const char file_separator = separator_string.empty() ? ',' : separator_string[0];

    CsvReader::Result parsed =
        CsvReader(
            file_separator,
            [this](const string_view line)
            {
                check_separators(line);
            })
        .read(data_path);

    const bool has_quotes = parsed.has_quotes;
    vector<string_view>& lines = parsed.lines;

    string header_scratch;

    throw_if(lines.empty(),
             "File {} is empty or contains no valid data rows.",
             data_path.string());

    read_data_file_preview(lines, file_separator, has_quotes);

    const vector<string_view> header_tokens =
        get_token_views_maybe_quoted(
            lines[0],
            file_separator,
            has_quotes,
            header_scratch);

    if(has_header)
    {
        const auto is_number = [](const string_view token)
        {
            return is_numeric_string(token);
        };

        throw_if(ranges::any_of(header_tokens, is_number),
                 "Some header names are numeric.");

        lines.erase(lines.begin());
    }

    throw_if(lines.empty(),
             "Data file only contains a header.");

    const Index samples_number = ssize(lines);

    if(number_format_automatic)
        number_format = detect_number_format(lines, file_separator, has_quotes);

    if(display && !number_format.is_default())
    {
        cout << "Reading numbers in " << data_path.string()
             << " with decimal separator "
             << number_format_name(number_format.decimal_separator)
             << " and thousands separator "
             << number_format_name(number_format.group_separator)
             << ".\n";
    }

    if(!has_sample_ids)
    {
        unordered_set<string> unique_elements;
        string id_scratch;

        bool possible_id = true;
        bool is_numeric_column = true;
        bool is_date_column = true;

        Index date_check_count = 0;
        constexpr Index max_date_checks = 20;

        for(const string_view line : lines)
        {
            const string_view token =
                first_token_maybe_quoted(
                    line,
                    file_separator,
                    has_quotes,
                    id_scratch);

            if(!unique_elements.emplace(token).second)
            {
                possible_id = false;
                break;
            }

            if(is_numeric_column
               && !is_missing_token(token, missing_values_label)
               && !is_numeric_string(token, number_format))
            {
                is_numeric_column = false;
            }

            if(is_date_column
               && date_check_count < max_date_checks
               && !is_missing_token(token, missing_values_label))
            {
                if(!is_date_time_string(token))
                    is_date_column = false;

                ++date_check_count;
            }
        }

        if(is_date_column && date_check_count > 0)
            possible_id = false;

        has_sample_ids =
            possible_id
            && !is_numeric_column
            && unique_elements.size() == size_t(samples_number);
    }

    const Index columns_number = ssize(header_tokens);
    const Index id_offset = has_sample_ids ? 1 : 0;

    throw_if(columns_number <= id_offset,
             "Data file contains no variables.");

    const vector<Variable> previous_variables = variables;

    Index variables_number = columns_number - id_offset;

    variables.resize(size_t(variables_number));

    if(has_header)
    {
        set_variable_names(
            vector<string>(
                header_tokens.begin() + id_offset,
                header_tokens.end()));
    }
    else
    {
        set_default_variable_names();
    }

    const DateFormat date_format =
        infer_column_types(lines, file_separator, has_quotes);

    for(Variable& variable : variables)
    {
        if(variable.is_categorical()
           && variable.get_categories_number() == 2)
        {
            variable.type = VariableType::Binary;
        }
    }

    vector<Index> variable_token_indices(variables.size());

    iota(variable_token_indices.begin(),
         variable_token_indices.end(),
         id_offset);

    for(const size_t i :
        views::iota(size_t(0), variables.size()) | views::reverse)
    {
        if(!looks_like_id_variable(
               variables[i],
               samples_number))
        {
            continue;
        }

        cout << "Excluding identifier column: "
             << variables[i].name
             << endl;

        variables.erase(variables.begin() + Index(i));

        variable_token_indices.erase(
            variable_token_indices.begin() + Index(i));
    }

    throw_if(variables.empty(),
             "Data file contains no variables (all columns are identifiers).");

    variables_number = ssize(variables);

    const Index required_tokens =
        variable_token_indices.back() + 1;

    for(Variable& variable : variables)
    {
        const vector<Variable>::const_iterator previous =
            ranges::find_if(
                previous_variables,
                [&](const Variable& candidate)
                {
                    return candidate.name == variable.name;
                });

        if(previous == previous_variables.end())
            continue;

        variable.role = previous->role;
        variable.scaler = previous->scaler;
    }

    // assign, not resize: re-reading a file into a live dataset is supported
    // (the variable roles and scalers are restored by name just above), but
    // resize keeps the existing elements, so rows another file had marked None
    // stayed None here and never trained. Binary mode re-marks incomplete rows
    // below as it always did.
    sample_roles.assign(size_t(samples_number), SampleRole::Training);
    sample_ids.assign(size_t(samples_number), {});

    const vector<vector<Index>> all_feature_indices =
        get_feature_indices();

    const Index feature_columns_number =
        all_feature_indices.empty()
        ? 0
        : all_feature_indices.back().back() + 1;

    if(feature_columns_number > 0)
    {
        const size_t projected_bytes =
            size_t(samples_number)
            * size_t(feature_columns_number)
            * sizeof(float);

        if(projected_bytes > maximum_expanded_data_bytes)
        {
            Index worst_index = 0;
            Index worst_categories = 0;

            for(Index i = 0; i < variables_number; ++i)
            {
                if(variables[size_t(i)].is_categorical()
                   && ssize(variables[size_t(i)].categories) > worst_categories)
                {
                    worst_categories =
                        ssize(variables[size_t(i)].categories);

                    worst_index = i;
                }
            }

            throw runtime_error(
                format(
                    "Expanding the categorical variables of this file would produce {} feature "
                    "columns for {} samples ({:.1f} GB). The largest contributor is '{}' with {} "
                    "categories. If that column is meant to be numeric, check its number format; "
                    "if it is an identifier, remove it from the file.",
                    feature_columns_number,
                    samples_number,
                    double(projected_bytes)
                        / (1024.0 * 1024.0 * 1024.0),
                    variables[size_t(worst_index)].name,
                    worst_categories));
        }
    }

    const bool binary_storage =
        storage_mode == StorageMode::BinaryFile;

    FileWriter cache_writer;
    vector<float> row_values;

    if(binary_storage)
    {
        cache_reader.close();
        clear_cache_derived_state();
        cache_path = cache_file_path();

        filesystem::create_directories(
            cache_path.parent_path());

        cache_writer.open(
            cache_path.string() + ".tmp");

        cache_columns_number = feature_columns_number;

        row_values.resize(size_t(feature_columns_number));
    }
    else
    {
        data = MatrixR::Zero(
            samples_number,
            feature_columns_number);
    }

    rows_missing_values_number = 0;
    missing_values_number = 0;

    variables_missing_values_number =
        VectorI::Zero(variables_number);

    vector<unordered_map<string_view, Index>>
        category_maps(static_cast<size_t>(variables_number));

    for(Index variable_index = 0;
        variable_index < variables_number;
        ++variable_index)
    {
        const Variable& variable =
            variables[size_t(variable_index)];

        if(!variable.is_categorical())
            continue;

        unordered_map<string_view, Index>& category_map =
            category_maps[size_t(variable_index)];

        for(Index category = 0;
            category < ssize(variable.categories);
            ++category)
        {
            category_map.emplace(
                string_view(variable.categories[size_t(category)]),
                category);
        }
    }

    struct NumericColumnValues
    {
        bool has_value = false;
        float first_value = 0.0f;
        bool constant = true;
        bool zero_one = true;
    };

    vector<NumericColumnValues>
        numeric_column_values(static_cast<size_t>(variables_number));

    const auto parse_row =
        [&](float* row,
            const vector<string_view>& row_tokens)
    {
        for(Index variable_index = 0;
            variable_index < variables_number;
            ++variable_index)
        {
            const size_t index = size_t(variable_index);

            const Variable& variable = variables[index];

            const string_view token =
                row_tokens[size_t(variable_token_indices[index])];

            const vector<Index>& feature_indices =
                all_feature_indices[index];

            using enum VariableType;

            switch(variable.type)
            {
                case None:
                case Constant:
                    break;

                case Numeric:
                case Integer:
                    parse_numeric_token(
                        row,
                        feature_indices[0],
                        token,
                        missing_values_label,
                        number_format);
                    break;

                case DateTime:
                    parse_datetime_token(
                        row,
                        feature_indices[0],
                        token,
                        missing_values_label,
                        date_format);
                    break;

                case Categorical:
                    parse_categorical_token(
                        row,
                        feature_indices,
                        token,
                        missing_values_label,
                        category_maps[index]);
                    break;

                case Binary:
                    parse_binary_token(
                        row,
                        feature_indices[0],
                        token,
                        missing_values_label,
                        variable.categories,
                        number_format);
                    break;
            }
        }
    };

    const auto refine_numeric =
        [&](const float* row)
    {
        for(Index variable_index = 0;
            variable_index < variables_number;
            ++variable_index)
        {
            const size_t index = size_t(variable_index);

            if(variables[index].type != VariableType::Numeric)
                continue;

            NumericColumnValues& column =
                numeric_column_values[index];

            const float value =
                row[size_t(all_feature_indices[index][0])];

            if(isnan(value))
                continue;

            if(!column.has_value)
            {
                column.has_value = true;
                column.first_value = value;
            }
            else if(abs(value - column.first_value)
                    > numeric_limits<float>::min())
            {
                column.constant = false;
            }

            if(value != 0.0f && value != 1.0f)
                column.zero_one = false;
        }
    };

    const auto count_missing =
        [&](const vector<string_view>& row_tokens,
            Index& thread_rows_missing,
            Index& thread_missing,
            vector<Index>& thread_variables_missing)
    {
        bool row_has_missing = false;

        for(Index variable_index = 0;
            variable_index < variables_number;
            ++variable_index)
        {
            const size_t index = size_t(variable_index);

            const size_t token_index =
                size_t(variable_token_indices[index]);

            if(token_index >= row_tokens.size())
                break;

            if(!is_missing_token(row_tokens[token_index], missing_values_label))
                continue;

            row_has_missing = true;
            ++thread_missing;
            ++thread_variables_missing[index];
        }

        if(row_has_missing)
            ++thread_rows_missing;

        return row_has_missing;
    };

    bool bad_row = false;
    Index bad_row_index = samples_number;
    Index bad_row_columns = 0;

    bool parse_error = false;
    Index parse_error_index = samples_number;
    string parse_error_message;

    const auto parse_rows =
        [&](const Index base,
            const Index end,
            float* destination)
    {
        Index range_rows_missing = 0;
        Index range_missing = 0;

        vector<Index> range_variables_missing(
            size_t(variables_number),
            0);

#pragma omp parallel
        {
            string thread_scratch;
            vector<string_view> thread_tokens;

            vector<Index> thread_variables_missing(
                size_t(variables_number),
                0);

            Index thread_rows_missing = 0;
            Index thread_missing = 0;

#pragma omp for schedule(static) nowait
            for(Index i = base; i < end; ++i)
            {
                get_token_views_maybe_quoted(
                    lines[size_t(i)],
                    file_separator,
                    has_quotes,
                    thread_scratch,
                    thread_tokens);

                float* row =
                    destination
                    + size_t(i - base)
                        * size_t(feature_columns_number);

                const bool row_has_missing =
                    count_missing(
                        thread_tokens,
                        thread_rows_missing,
                        thread_missing,
                        thread_variables_missing);

                if(ssize(thread_tokens) < required_tokens)
                {
#pragma omp critical
                    {
                        if(i < bad_row_index)
                        {
                            bad_row = true;
                            bad_row_index = i;
                            bad_row_columns =
                                ssize(thread_tokens);
                        }
                    }

                    continue;
                }

                if(has_sample_ids)
                {
                    sample_ids[size_t(i)] =
                        string(thread_tokens[0]);
                }

                if(binary_storage && row_has_missing)
                {
                    sample_roles[size_t(i)] =
                        SampleRole::None;
                }

                try
                {
                    parse_row(row, thread_tokens);
                }
                catch(const exception& e)
                {
#pragma omp critical
                    {
                        if(i < parse_error_index)
                        {
                            parse_error = true;
                            parse_error_index = i;
                            parse_error_message = e.what();
                        }
                    }
                }
            }

#pragma omp critical
            {
                range_rows_missing += thread_rows_missing;
                range_missing += thread_missing;

                for(Index variable_index = 0;
                    variable_index < variables_number;
                    ++variable_index)
                {
                    range_variables_missing[
                        size_t(variable_index)]
                        += thread_variables_missing[
                            size_t(variable_index)];
                }
            }
        }

        rows_missing_values_number +=
            range_rows_missing;

        missing_values_number +=
            range_missing;

        for(Index variable_index = 0;
            variable_index < variables_number;
            ++variable_index)
        {
            variables_missing_values_number(variable_index)
                += range_variables_missing[
                    size_t(variable_index)];
        }

        for(Index i = base; i < end; ++i)
        {
            refine_numeric(
                destination
                + size_t(i - base)
                    * size_t(feature_columns_number));
        }
    };

    if(binary_storage)
    {
        constexpr Index chunk_size = 16384;

        vector<float> chunk_buffer;

        for(Index base = 0;
            base < samples_number;
            base += chunk_size)
        {
            const Index end =
                min(base + chunk_size, samples_number);

            const Index rows_number =
                end - base;

            chunk_buffer.assign(
                size_t(rows_number)
                    * size_t(feature_columns_number),
                0.0f);

            parse_rows(
                base,
                end,
                chunk_buffer.data());

            cache_writer.write(
                span(chunk_buffer));
        }
    }
    else
    {
        parse_rows(
            0,
            samples_number,
            data.data());
    }

    if(bad_row
       && (!parse_error
           || bad_row_index <= parse_error_index))
    {
        throw runtime_error(
            format(
                "Row {} has fewer columns than expected ({}).",
                bad_row_index,
                bad_row_columns));
    }

    if(parse_error)
    {
        throw runtime_error(
            format(
                "Row {}: {}",
                parse_error_index,
                parse_error_message));
    }

    if(binary_storage)
    {
        cache_writer.finish_with_rename(cache_path);
        cache_reader.open(cache_path);
    }

    for(Index variable_index = 0;
        variable_index < variables_number;
        ++variable_index)
    {
        Variable& variable =
            variables[size_t(variable_index)];

        if(variable.type == VariableType::Numeric)
        {
            const NumericColumnValues& column =
                numeric_column_values[
                    size_t(variable_index)];

            if(column.constant)
            {
                variable.set(
                    variable.name,
                    "None",
                    VariableType::Constant);
            }
            else if(column.zero_one)
            {
                variable.type = VariableType::Binary;
                variable.categories = {"0", "1"};
            }
        }
        else if(is_one_of(
                    variable.type,
                    VariableType::Binary,
                    VariableType::Categorical)
                && variable.get_categories_number() == 1)
        {
            variable.set(
                variable.name,
                "None",
                VariableType::Constant);
        }
    }

    split_samples_random();

    if (binary_storage)
        refresh_cache_statistics();
}

static const EnumMap<TabularDataset::MissingValuesMethod>& missing_values_method_map()
{
    static const EnumMap<TabularDataset::MissingValuesMethod> map{
        {TabularDataset::MissingValuesMethod::Unuse,         "Unuse"},
        {TabularDataset::MissingValuesMethod::Mean,          "Mean"},
        {TabularDataset::MissingValuesMethod::Median,        "Median"},
        {TabularDataset::MissingValuesMethod::Interpolation, "Interpolation"}
    };
    return map;
}

string TabularDataset::get_missing_values_method_string() const
{
    return missing_values_method_map().to_string(missing_values_method);
}

void TabularDataset::set_missing_values_method(const string& new_missing_values_method)
{
    missing_values_method = missing_values_method_map().from_string(new_missing_values_method);
}

void TabularDataset::missing_values_to_JSON(JsonWriter &printer) const
{
    printer.open_element("MissingValues");

    if (missing_values_number > 0)
        write_json(printer, {
            {"MissingValuesNumber", missing_values_number},
            {"MissingValuesMethod", get_missing_values_method_string()},
            {"VariablesMissingValuesNumber", vector_to_string(variables_missing_values_number)},
            {"SamplesMissingValuesNumber", rows_missing_values_number}
        });
    else
        add_json_field(printer, "MissingValuesNumber", missing_values_number);

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

    vector<Index> counts;
    counts.reserve(tokens.size());

    for (const string& token : tokens)
        if (!token.empty())
            counts.push_back(parse_number<Index>(token, "VariablesMissingValuesNumber", "integer"));

    variables_missing_values_number = VectorI::Zero(ssize(counts));

    for (size_t i = 0; i < counts.size(); ++i)
        variables_missing_values_number(Index(i)) = counts[i];

    if (variables_missing_values_number.sum() != missing_values_number)
        variables_missing_values_number.resize(0);

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

void TabularDataset::unuse_samples_with_missing_targets(const vector<Index>& sample_indices,
                                                        const vector<Index>& target_feature_indices)
{
    for (const Index current_variable : target_feature_indices)
        for (const Index current_sample : sample_indices)
            if (isnan(data(current_sample, current_variable)))
                set_sample_role(current_sample, "None");
}

void TabularDataset::impute_missing_values_statistic(const MissingValuesMethod& method)
{
    require_in_memory_data("TabularDataset::impute_missing_values_statistic");

    const vector<Index> used_sample_indices = get_used_sample_indices();
    const vector<Index> used_feature_indices = get_used_feature_indices();
    const vector<Index> target_feature_indices = get_feature_indices(VariableRole::Target);

    if (used_sample_indices.empty() || used_feature_indices.empty())
        return;

    VectorR replacements = (method == MissingValuesMethod::Mean)
        ? mean(data, used_sample_indices, used_feature_indices)
        : median(data, used_sample_indices, used_feature_indices);

    for (Index j = 0; j < replacements.size(); ++j)
        if (!isfinite(replacements(j))) replacements(j) = 0.0f;

    const Index samples_number = used_sample_indices.size();
    const Index features_number = used_feature_indices.size();

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

    unuse_samples_with_missing_targets(used_sample_indices, target_feature_indices);
}

void TabularDataset::reuse_input_incomplete_rows_binary()
{

    if (storage_mode != StorageMode::BinaryFile) return;
    if (cache_columns_number == 0 || !cache_reader.is_open()) return;

    const vector<Index> target_feature_indices = get_feature_indices(VariableRole::Target);

    const Index columns_number = cache_columns_number;
    const Index samples_number = get_samples_number();

    vector<float> row(static_cast<size_t>(columns_number));

    bool roles_changed = false;

    for (Index sample_index = 0; sample_index < samples_number; ++sample_index)
    {
        cache_reader.read_at(span(row),
                             uint64_t(sample_index) * uint64_t(columns_number) * sizeof(float));

        const bool target_missing = ranges::any_of(target_feature_indices,
            [&](const Index target_index) { return isnan(row[size_t(target_index)]); });

        if (target_missing)
        {
            roles_changed |= sample_roles[size_t(sample_index)] != SampleRole::None;
            sample_roles[size_t(sample_index)] = SampleRole::None;
        }
        else if (sample_roles[size_t(sample_index)] == SampleRole::None)
        {
            sample_roles[size_t(sample_index)] = SampleRole::Training;
            roles_changed = true;
        }
    }

    if (roles_changed) on_used_samples_changed();
}

void TabularDataset::impute_missing_values_interpolate()
{
    require_in_memory_data("TabularDataset::impute_missing_values_interpolate");

    const vector<Index> used_sample_indices = get_used_sample_indices();
    const vector<Index> input_feature_indices = get_feature_indices(VariableRole::Input);
    const vector<Index> target_feature_indices = get_feature_indices(VariableRole::Target);

    const Index samples_number = used_sample_indices.size();

    for (const Index current_variable : input_feature_indices)
    {
        for (Index i = 0; i < samples_number; ++i)
        {
            const Index current_sample = used_sample_indices[i];

            if (!isnan(data(current_sample, current_variable))) continue;

            // "No neighbour" and "the neighbour is sample 0 holding 0.0" used
            // to be the same state, so a leading NaN was interpolated against a
            // phantom point at the origin and a trailing one was extrapolated
            // towards zero. Optional keeps them apart: two neighbours
            // interpolate, one is carried across, none leaves the NaN for
            // unuse_samples_with_missing_targets to deal with.
            optional<pair<Index, float>> left;
            optional<pair<Index, float>> right;

            for (Index k = i - 1; k >= 0; k--)
            {
                if (isnan(data(used_sample_indices[k], current_variable))) continue;

                left = {used_sample_indices[k], data(used_sample_indices[k], current_variable)};
                break;
            }

            for (Index k = i + 1; k < samples_number; ++k)
            {
                if (isnan(data(used_sample_indices[k], current_variable))) continue;

                right = {used_sample_indices[k], data(used_sample_indices[k], current_variable)};
                break;
            }

            if (!left && !right) continue;

            float interpolated_value = 0.0f;

            if (left && right && right->first != left->first)
            {
                const float span = float(right->first - left->first);
                interpolated_value = left->second
                    + (float(current_sample) - float(left->first)) * (right->second - left->second) / span;
            }
            else
            {
                interpolated_value = left ? left->second : right->second;
            }

            data(current_sample, current_variable) = interpolated_value;
        }
    }

    unuse_samples_with_missing_targets(used_sample_indices, target_feature_indices);
}

void TabularDataset::scrub_missing_values()
{

    if (storage_mode == StorageMode::BinaryFile)
    {

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

DateFormat TabularDataset::infer_column_types(
    const vector<string_view>& sample_lines,
    const char file_separator,
    const bool has_quotes)
{
    const Index variables_number = ssize(variables);
    const size_t total_rows = sample_lines.size();

    if(total_rows == 0) return Auto;

    constexpr size_t max_rows_to_check = 100;

    const size_t rows_to_check = min(max_rows_to_check, total_rows);
    const size_t id_offset = has_sample_ids ? 1 : 0;

    vector<vector<string_view>> sampled_tokens(rows_to_check);
    vector<string> sampled_scratch(rows_to_check);

    for(size_t i = 0; i < rows_to_check; ++i)
    {
        const size_t row = i * total_rows / rows_to_check;

        sampled_tokens[i] = get_token_views_maybe_quoted(
            sample_lines[row],
            file_separator,
            has_quotes,
            sampled_scratch[i]);
    }

    for(Index col_index = 0; col_index < variables_number; ++col_index)
    {
        Variable& variable = variables[size_t(col_index)];
        variable.type = VariableType::None;

        const size_t token_index = size_t(col_index) + id_offset;

        size_t checked_tokens = 0;
        size_t numeric_tokens = 0;
        string first_unparseable;

        for(const vector<string_view>& tokens : sampled_tokens)
        {
            if(token_index >= tokens.size()) continue;

            const string_view token = tokens[token_index];

            if(is_missing_token(token, missing_values_label))
                continue;

            ++checked_tokens;

            const bool numeric = is_numeric_string(token, number_format);

            if(numeric)
            {
                ++numeric_tokens;
            }
            else if(first_unparseable.empty())
            {
                first_unparseable = token;
            }

            if(variable.is_categorical())
                continue;

            if(numeric)
            {
                if(variable.type == VariableType::None)
                    variable.type = VariableType::Numeric;

                continue;
            }

            if(is_date_time_string(token))
            {
                if(variable.type == VariableType::None)
                    variable.type = VariableType::DateTime;

                continue;
            }

            variable.type = VariableType::Categorical;
        }

        if(variable.type == VariableType::None)
            variable.type = VariableType::Numeric;

        if(variable.type == VariableType::Categorical
           && checked_tokens > 0
           && numeric_tokens * 10 >= checked_tokens * 9)
        {
            cout << "Warning: variable '" << variable.name
                 << "' was classified as categorical, but "
                 << numeric_tokens << " of its " << checked_tokens
                 << " sampled values are numeric. First value that failed to parse: '"
                 << first_unparseable
                 << "'. Check the number format (thousands separators, decimal commas); "
                    "otherwise this column expands into one column per distinct value.\n";
        }
    }

    DateFormat date_format = Auto;

    if(ranges::any_of(
           variables,
           [](const Variable& variable)
           {
               return variable.type == VariableType::DateTime;
           }))
    {
        size_t hit_row = rows_to_check;

        for(size_t i = 0;
            i < rows_to_check && date_format == Auto;
            ++i)
        {
            const vector<string_view>& tokens = sampled_tokens[i];

            for(Index col_index = 0;
                col_index < variables_number;
                ++col_index)
            {
                const Variable& variable =
                    variables[size_t(col_index)];

                if(variable.type != VariableType::DateTime)
                    continue;

                const size_t token_index =
                    size_t(col_index) + id_offset;

                if(token_index >= tokens.size())
                    continue;

                const string_view token =
                    tokens[token_index];

                if(is_missing_token(token, missing_values_label))
                    continue;

                date_format = detect_date_format(token);

                if(date_format != Auto)
                {
                    hit_row = i;
                    break;
                }
            }
        }

        if(rows_to_check != total_rows && hit_row != 0)
        {
            date_format = infer_dataset_date_format(
                variables,
                sample_lines,
                file_separator,
                has_sample_ids,
                missing_values_label,
                has_quotes);
        }
    }

    if(ranges::none_of(
           variables,
           [](const Variable& variable)
           {
               return variable.is_categorical();
           }))
    {
        return date_format;
    }

    vector<unordered_set<string>>
        unique_categories(static_cast<size_t>(variables_number));

    const Index lines_number = ssize(sample_lines);

#pragma omp parallel
    {
        vector<unordered_set<string>>
            local_categories(static_cast<size_t>(variables_number));

        string scratch;
        vector<string_view> tokens;

#pragma omp for schedule(static) nowait
        for(Index row = 0; row < lines_number; ++row)
        {
            get_token_views_maybe_quoted(
                sample_lines[size_t(row)],
                file_separator,
                has_quotes,
                scratch,
                tokens);

            for(Index col_index = 0;
                col_index < variables_number;
                ++col_index)
            {
                const size_t index = size_t(col_index);

                if(!variables[index].is_categorical())
                    continue;

                const size_t token_index =
                    index + id_offset;

                if(token_index >= tokens.size())
                    continue;

                const string_view token =
                    tokens[token_index];

                if(is_missing_token(token, missing_values_label))
                    continue;

                local_categories[index].emplace(token);
            }
        }

#pragma omp critical
        {
            for(Index col_index = 0;
                col_index < variables_number;
                ++col_index)
            {
                const size_t index = size_t(col_index);

                unique_categories[index].insert(
                    local_categories[index].begin(),
                    local_categories[index].end());
            }
        }
    }

    for(Index col_index = 0;
        col_index < variables_number;
        ++col_index)
    {
        Variable& variable =
            variables[size_t(col_index)];

        if(!variable.is_categorical())
            continue;

        const unordered_set<string>& unique =
            unique_categories[size_t(col_index)];

        variable.categories.assign(
            unique.begin(),
            unique.end());

        ranges::sort(variable.categories);
    }

    return date_format;
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
             "Size of variable scalers({}) has to be the same as variables numbers({}).\n",
                    new_scalers.size(), variables_number);

    for (size_t i = 0; i < variables_number; ++i)
        variables[i].set_scaler(new_scalers[i]);
}

void TabularDataset::set_default_variable_scalers()
{
    for (Variable& variable : variables)
        variable.scaler = is_one_of(variable.type, VariableType::Numeric, VariableType::Integer)
                                  ? ScalerMethod::MeanStandardDeviation
                                  : ScalerMethod::MinimumMaximum;
}

void TabularDataset::to_JSON(JsonWriter& printer) const
{
    const string decimal_separator_name =
        number_format_automatic
        ? "Auto"
        : number_format_name(number_format.decimal_separator);

    const string group_separator_name =
        number_format_automatic
        ? "Auto"
        : number_format_name(number_format.group_separator);

    write_json_header(printer, {
        {"FileType", "csv"},
        {"Path", data_path.string()},
        {"Separator", get_separator_name()},
        {"HasHeader", has_header},
        {"HasSamplesId", has_sample_ids},
        {"MissingValuesLabel", missing_values_label},
        {"DecimalSeparator", decimal_separator_name},
        {"ThousandsSeparator", group_separator_name},
        {"Codification", get_codification_string()},
        {"StorageMode", get_storage_mode_string()}
    });

    missing_values_to_JSON(printer);
    preview_data_to_JSON(printer);

    write_json_footer(printer);
}

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
