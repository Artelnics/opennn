//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   T I M E   S E R I E S   D A T A S E T   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "time_series_dataset.h"
#include "batch.h"
#include "statistics.h"
#include "correlations.h"
#include "tensor_types.h"
#include "random_utilities.h"

namespace opennn
{

namespace
{

Index interpolate_gap(MatrixR& data,
                      const vector<Index>& used_sample_indices,
                      const Index used_samples_number,
                      const Index feature_index,
                      const Index start_missing)
{
    Index prev_index = start_missing - 1;
    float prev_value = NAN;
    while (prev_index >= 0 && isnan(prev_value))
    {
        prev_value = data(used_sample_indices[prev_index], feature_index);
        if (isnan(prev_value)) prev_index--;
    }

    Index end_missing = start_missing;

    while (end_missing < used_samples_number && isnan(data(used_sample_indices[end_missing], feature_index)))
        ++end_missing;

    const Index n_missing = end_missing - start_missing;

    const float next_value = (end_missing < used_samples_number)
        ? data(used_sample_indices[end_missing], feature_index) : NAN;

    for (Index i = 0; i < n_missing; ++i)
    {
        const Index sample_i = used_sample_indices[start_missing + i];

        if (isnan(prev_value))
            data(sample_i, feature_index) = next_value;
        else if (isnan(next_value))
            data(sample_i, feature_index) = prev_value;
        else
        {
            const float fraction = float(i + 1) / float(n_missing + 1);
            data(sample_i, feature_index) = lerp(prev_value, next_value, fraction);
        }
    }

    return end_missing;
}

}

TimeSeriesDataset::TimeSeriesDataset(const Index new_samples_number,
                                     const Shape& new_input_shape,
                                     const Shape& new_target_shape)
    :TabularDataset(new_samples_number, new_input_shape, new_target_shape)
{
}

TimeSeriesDataset::TimeSeriesDataset(const filesystem::path& data_path,
                                     const string& separator,
                                     bool has_header,
                                     bool has_sample_ids,
                                     const Codification& data_codification)
    :TabularDataset(data_path, separator, has_header, has_sample_ids, data_codification)
{
    configure_forecasting();
}

Index TimeSeriesDataset::get_past_time_steps() const
{
    return past_time_steps;
}

Index TimeSeriesDataset::get_future_time_steps() const
{
    return future_time_steps;
}

bool TimeSeriesDataset::get_multi_target() const
{
    return multi_target;
}

Tensor3 TimeSeriesDataset::get_data(const string& sample_role, const string& feature_role) const
{
    const vector<Index> sample_indices = get_sample_indices(sample_role);
    const vector<Index> feature_indices = get_feature_indices(feature_role);

    if (sample_indices.empty() || feature_indices.empty())
        return {};

    const Index samples_number = sample_indices.size();
    const Index features_number = feature_indices.size();

    Tensor3 data_3d(samples_number, past_time_steps, features_number);
    data_3d.setZero();

    const VariableRole role_type = string_to_variable_role(feature_role);

    if (role_type == VariableRole::Input || role_type == VariableRole::InputTarget)
        fill_inputs(sample_indices, feature_indices, data_3d.data(), FillMode::Inference);
    else if (role_type == VariableRole::Target)
        throw runtime_error("get_data for 3D is only implemented for 'Input' variables in TimeSeriesDataset.");

    return data_3d;
}

void TimeSeriesDataset::set_past_time_steps(const Index new_past_time_steps)
{
    past_time_steps = new_past_time_steps;
    input_shape = { past_time_steps, get_features_number("Input") };
    refresh_forecasting_roles();
}

void TimeSeriesDataset::set_future_time_steps(const Index new_future_time_steps)
{
    future_time_steps = new_future_time_steps;
    if (multi_target)
    {
        const Index n_targets = get_features_number("Target");
        target_shape = { future_time_steps * (n_targets > 0 ? n_targets : 1) };
    }
    refresh_forecasting_roles();
}

void TimeSeriesDataset::refresh_forecasting_roles()
{
    const Index samples_number = get_samples_number();
    const Index window_span = past_time_steps + future_time_steps;

    if (samples_number == 0 || window_span <= 0) return;

    const Index a = Index(0.6f * float(samples_number));
    const Index b = a + Index(0.2f * float(samples_number));

    auto mark = [&](Index lo, Index hi, SampleRole role)
    {
        const Index last_valid_start = hi - window_span + 1;
        for (Index i = lo; i < hi; ++i)
            sample_roles[i] = (i < last_valid_start) ? role : SampleRole::None;
    };

    mark(0, a, SampleRole::Training);
    mark(a, b, SampleRole::Validation);
    mark(b, samples_number, SampleRole::Testing);
}

void TimeSeriesDataset::set_multi_target(bool new_multi_target)
{
    multi_target = new_multi_target;

    const Index n_targets = get_features_number("Target");
    target_shape = multi_target ? Shape{ future_time_steps * (n_targets > 0 ? n_targets : 1) }
                                : Shape{ n_targets > 0 ? n_targets : 1 };
}

void TimeSeriesDataset::resize_input_shape(Index input_features_count)
{
    set_shape("Input", {past_time_steps, input_features_count});
}

void TimeSeriesDataset::to_JSON(JsonWriter& printer) const
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
        {"LagsNumber", to_string(get_past_time_steps())},
        {"StepsAhead", to_string(get_future_time_steps())},
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

void TimeSeriesDataset::from_JSON(const JsonDocument& data_set_document)
{
    const Json* data_set_element = get_json_root(data_set_document, "Dataset");


    const Json* data_source_element = require_json_field(data_set_element, "DataSource");

    require_json_field(data_source_element, "FileType");

    set_data_path(read_json_string(data_source_element, "Path"));
    set_separator_name(read_json_string(data_source_element, "Separator"));
    set_has_header(read_json_bool(data_source_element, "HasHeader"));
    set_has_ids(read_json_bool(data_source_element, "HasSamplesId"));
    set_missing_values_label(read_json_string(data_source_element, "MissingValuesLabel"));
    set_past_time_steps(parse_int(read_json_string(data_source_element, "LagsNumber"), "LagsNumber"));
    set_future_time_steps(parse_int(read_json_string(data_source_element, "StepsAhead"), "StepsAhead"));
    set_codification(read_json_string(data_source_element, "Codification"));


    if (const Json* variables_element = data_set_element->find("Variables"))
        variables_from_JSON(variables_element);
    if (const Json* samples_element = data_set_element->find("Samples"))
        samples_from_JSON(samples_element);
    if (const Json* missing_values_element = data_set_element->find("MissingValues"))
        missing_values_from_JSON(missing_values_element);
    if (const Json* preview_data_element = data_set_element->find("PreviewData"))
        preview_data_from_JSON(preview_data_element);

    set_display(read_json_bool(data_set_element, "Display"));

    input_shape = { past_time_steps, get_features_number("Input") };
    target_shape = { get_features_number("Target") };
}

void TimeSeriesDataset::read_csv()
{
    TabularDataset::read_csv();

    configure_forecasting();
}

void TimeSeriesDataset::configure_forecasting()
{
    set_default_variable_roles_forecasting();

    if (get_features_number() == 1)
    {
        set_variable_role(0, "InputTarget");
    }
    else
    {
        const vector<Index> target_indices = get_feature_indices("Target");

        if (!target_indices.empty())
            set_variable_role(get_variable_index(target_indices[0]), "InputTarget");
    }

    input_shape = {past_time_steps, get_features_number("Input")};
    target_shape = {get_features_number("Target")};

    refresh_forecasting_roles();
}

void TimeSeriesDataset::impute_missing_values_unuse()
{
    const Index samples_number = get_samples_number();
    const Index lags = get_past_time_steps();

    vector<char> row_has_nan(samples_number);
    for (Index i = 0; i < samples_number; ++i)
        row_has_nan[i] = has_nan_row(i);

    const Index num_sequences = samples_number - lags;
    if (num_sequences < 0) return;

    #pragma omp parallel for
    for (Index i = 0; i < num_sequences; ++i)
    {
        const auto first = row_has_nan.begin() + i;

        if (any_of(first, first + lags + 1, [](char value) { return value != 0; }))
            set_sample_role(i, "None");
    }

    for (Index i = num_sequences; i < samples_number; ++i)
        set_sample_role(i, "None");
}

void TimeSeriesDataset::impute_missing_values_interpolate()
{
    const vector<Index> used_sample_indices = get_used_sample_indices();

    const Index used_samples_number = used_sample_indices.size();

    #pragma omp parallel for
    for (Index feature_index = 0; feature_index < get_features_number(); ++feature_index)
    {
        Index i = 0;

        while (i < used_samples_number)
        {
            if (!isnan(data(used_sample_indices[i], feature_index)))
            {
                ++i;
                continue;
            }

            i = interpolate_gap(data, used_sample_indices, used_samples_number, feature_index, i);
        }
    }

}

void TimeSeriesDataset::fill_inputs(const vector<Index>& sample_indices,
                                    const vector<Index>& input_indices,
                                    float* input_data,
                                    FillMode,
                                    int) const
{
    if (sample_indices.empty() || input_indices.empty()) return;

    const Index batch_size = sample_indices.size();
    const Index inputs_number = input_indices.size();
    const Index data_rows_number = data.rows();

    for (const Index sample_index : sample_indices)
        throw_if(sample_index < 0 || sample_index >= data_rows_number,
                 "TimeSeriesDataset input sample index is out of range.");

    for (const Index input_index : input_indices)
        throw_if(input_index < 0 || input_index >= data.cols(),
                 "TimeSeriesDataset input feature index is out of range.");

    TensorMap3 inputs(input_data, batch_size, past_time_steps, inputs_number);

    #pragma omp parallel for schedule(static)
    for (Index i = 0; i < batch_size; ++i)
    {
        const Index start_row = sample_indices[i];

        for (Index j = 0; j < past_time_steps; ++j)
        {
            const Index actual_row = start_row + j;
            const bool in_range = actual_row < data_rows_number;

            for (Index k = 0; k < inputs_number; ++k)
                inputs(i, j, k) = in_range ? data(actual_row, input_indices[k]) : 0.0f;
        }
    }
}

void TimeSeriesDataset::fill_targets(const vector<Index>& sample_indices,
                                     const vector<Index>& target_indices,
                                     float* target_data,
                                     FillMode,
                                     int) const
{
    if (sample_indices.empty() || target_indices.empty()) return;

    const Index batch_size = ssize(sample_indices);
    const Index targets_number = get_target_shape()[0];
    const Index total_rows_in_data = data.rows();
    const Index target_columns = ssize(target_indices);

    for (const Index sample_index : sample_indices)
        throw_if(sample_index < 0 || sample_index >= total_rows_in_data,
                 "TimeSeriesDataset target sample index is out of range.");

    for (const Index target_index : target_indices)
        throw_if(target_index < 0 || target_index >= data.cols(),
                 "TimeSeriesDataset target feature index is out of range.");

    MatrixMap targets(target_data, batch_size, targets_number);

    if (multi_target)
    {
        #pragma omp parallel for schedule(static)
        for (Index i = 0; i < batch_size; ++i)
            for (Index j = 0; j < target_columns; ++j)
                for (Index k = 0; k < future_time_steps; ++k)
                {
                    const Index target_row = sample_indices[i] + past_time_steps + k;
                    targets(i, j * future_time_steps + k) =
                        (target_row < total_rows_in_data)
                            ? data(target_row, target_indices[j])
                            : 0.0f;
                }
    }
    else
    {
        #pragma omp parallel for schedule(static)
        for (Index i = 0; i < batch_size; ++i)
        {
            const Index target_row = sample_indices[i] + past_time_steps + (future_time_steps - 1);
            for (Index j = 0; j < target_columns; ++j)
                targets(i, j) = (target_row < total_rows_in_data)
                    ? data(target_row, target_indices[j])
                    : 0.0f;
        }
    }
}

void TimeSeriesDataset::fill_batch(Batch& batch,
                                   const vector<Index>& sample_indices,
                                   const vector<Index>& input_indices,
                                   const vector<Index>& decoder_indices,
                                   const vector<Index>& target_indices,
                                   FillMode mode) const
{
    throw_if(Index(sample_indices.size()) != batch.samples_number,
             "fill_batch sample count does not match the batch size.");

    if (batch.uses_cuda() && is_device_resident() && !batch.input_is_bf16
        && batch.decoder.shape.empty()
        && is_contiguous(input_indices) && is_contiguous(target_indices))
    {
        batch.device_gather = true;
        batch.gather_row_indices.resize(sample_indices.size());
        for (size_t i = 0; i < sample_indices.size(); ++i)
            batch.gather_row_indices[i] = int(sample_indices[i]);
        batch.input_col_offset    = input_indices.empty()  ? 0 : input_indices.front();
        batch.target_col_offset   = target_indices.empty() ? 0 : target_indices.front();
        batch.window_past         = past_time_steps;
        batch.window_future       = future_time_steps;
        batch.window_features     = ssize(input_indices);
        batch.window_target_cols  = ssize(target_indices);
        batch.window_multi_target = multi_target;
        batch.window_matrix_rows  = data.rows();
        batch.needs_device_copy   = true;
        return;
    }

    fill_batch_host(batch, sample_indices, input_indices, decoder_indices,
                    target_indices, mode);
}

MatrixR TimeSeriesDataset::calculate_autocorrelations(const Index past_time_steps) const
{
    const Index samples_number = get_samples_number();

    throw_if(past_time_steps > samples_number,
             format("Past time steps ({}) is greater than samples number ({}) \n",
                    past_time_steps, samples_number));

    const Index variables_number = get_variables_number();

    const vector<Index> input_variable_indices = get_variable_indices("Input");
    const vector<Index> target_variable_indices = get_variable_indices("Target");

    const Index extra_targets = ranges::count_if(target_variable_indices,
        [&](Index index) { return variables[index].role != VariableRole::InputTarget; });

    const Index input_target_numeric_variables_number =
        ranges::count_if(input_variable_indices,
                         [&](Index index) { return variables[index].type == VariableType::Numeric; })
      + count_if(target_variable_indices.begin(),
                 target_variable_indices.begin() + extra_targets,
                 [&](Index index) { return variables[index].type == VariableType::Numeric; });

    const Index new_past_time_steps =
        ((samples_number <= past_time_steps) && past_time_steps > 2) ? past_time_steps - 2 :
         (samples_number == past_time_steps + 1 && past_time_steps > 1) ? past_time_steps - 1 :
         past_time_steps;

    MatrixR autocorrelations(input_target_numeric_variables_number, new_past_time_steps);
    Index counter_i = 0;

    for (Index i = 0; i < variables_number; ++i)
    {
        if (variables[i].role == VariableRole::None || variables[i].type != VariableType::Numeric)
            continue;

        MatrixR input_i = get_variable_data(i);
        cout << "Calculating " << variables[i].name << " autocorrelations" << "\n";

        const Map<const VectorR> current_input_i(input_i.data(), input_i.rows());

        autocorrelations.row(counter_i) = opennn::autocorrelations(current_input_i, new_past_time_steps).transpose();

        ++counter_i;
    }

    return autocorrelations;
}

Tensor3 TimeSeriesDataset::calculate_cross_correlations(const Index past_time_steps) const
{
    const Index samples_number = get_samples_number();

    throw_if(past_time_steps > samples_number,
             format("Past time steps ({}) is greater than samples number ({}) \n",
                    past_time_steps, samples_number));

    const Index variables_number = get_variables_number();

    const vector<Index> input_variable_indices = get_variable_indices("Input");
    const vector<Index> target_variable_indices = get_variable_indices("Target");

    const Index input_target_numeric_variables_number =
        ranges::count_if(input_variable_indices,
                         [&](Index index) { return variables[index].type == VariableType::Numeric; })
      + ranges::count_if(target_variable_indices,
                         [&](Index index) { return variables[index].type == VariableType::Numeric
                                                && variables[index].role != VariableRole::InputTarget; });

    const Index new_past_time_steps = (samples_number == past_time_steps) ? (past_time_steps - 2)
                                : (samples_number == past_time_steps + 1) ? (past_time_steps - 1)
                                : past_time_steps;

    vector<Index> numeric_variable_indices;

    for (Index i = 0; i < variables_number; ++i)
        if (variables[i].role != VariableRole::None && variables[i].type == VariableType::Numeric)
            numeric_variable_indices.push_back(i);

    const Index numeric_variables_number = numeric_variable_indices.size();

    Tensor3 cross_correlations(input_target_numeric_variables_number,
                               input_target_numeric_variables_number,
                               new_past_time_steps);

    VectorR cross_correlations_vector(new_past_time_steps);

    for (Index i = 0; i < numeric_variables_number; ++i)
    {
        const Index variable_i = numeric_variable_indices[i];

        MatrixR input_i = get_variable_data(variable_i);

        if (display) cout << "Calculating " << variables[variable_i].name << " cross correlations:" << "\n";

        for (Index j = 0; j < numeric_variables_number; ++j)
        {
            const Index variable_j = numeric_variable_indices[j];

            MatrixR input_j = get_variable_data(variable_j);

            if (display) cout << "  vs. " << variables[variable_j].name << "\n";

            const Map<const VectorR> current_input_i(input_i.data(), input_i.rows());
            const Map<const VectorR> current_input_j(input_j.data(), input_j.rows());

            cross_correlations_vector = opennn::cross_correlations(current_input_i, current_input_j, new_past_time_steps);

            for (Index k = 0; k < new_past_time_steps; ++k)
                cross_correlations(i, j, k) = cross_correlations_vector(k);
        }
    }

    return cross_correlations;
}

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
