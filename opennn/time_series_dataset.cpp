//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   T I M E   S E R I E S   D A T A S E T   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "time_series_dataset.h"
#include "statistics.h"
#include "correlations.h"
#include "tensors.h"
#include "random_utilities.h"

namespace opennn
{

TimeSeriesDataset::TimeSeriesDataset(const Index new_samples_number,
                                     const Shape& new_input_shape,
                                     const Shape& new_target_shape)
    :Dataset(new_samples_number, new_input_shape, new_target_shape)
{

}


TimeSeriesDataset::TimeSeriesDataset(const filesystem::path& data_path,
                                     const string& separator,
                                     bool has_header,
                                     bool has_sample_ids,
                                     const Codification& data_codification)
    :Dataset(data_path, separator, has_header, has_sample_ids, data_codification)
{
    const Index features_number = get_features_number();

    if(features_number == 1)
        set_variable_role(0, "InputTarget");
    else
    {
        const vector<Index> target_index = get_feature_indices("Target");
        set_variable_role(target_index[0], "InputTarget");
    }

    input_shape = {past_time_steps, get_features_number("Input")};
    target_shape = { get_features_number("Target") };

    split_samples_sequential(type(0.6), type(0.2), type(0.2));

}


Index TimeSeriesDataset::get_time_variable_index() const
{
    return time_variable_index;
}


Index TimeSeriesDataset::get_past_time_steps() const
{
    return past_time_steps;
}


Index TimeSeriesDataset::get_future_time_steps() const
{
    return future_time_steps;
}


Tensor3 TimeSeriesDataset::get_data(const string& sample_role, const string& feature_use) const
{
    const vector<Index> sample_indices = get_sample_indices(sample_role);
    const vector<Index> feature_indices = get_feature_indices(feature_use);

    if (sample_indices.empty() || feature_indices.empty()) {
        return Tensor3();
    }

    const Index samples_number = sample_indices.size();
    const Index features_number = feature_indices.size();

    Tensor3 data_3d(samples_number, past_time_steps, features_number);
    data_3d.setZero();

    if (feature_use == "Input" || feature_use == "InputTarget")
        fill_input_tensor(sample_indices, feature_indices, data_3d.data());
    else if (feature_use == "Target")
        throw runtime_error("get_data for 3D is only implemented for 'Input' variables in TimeSeriesDataset.");

    return data_3d;
}


TimeSeriesDataset::TimeSeriesData TimeSeriesDataset::get_data() const
{
    const Index total_samples = get_samples_number();
    if (total_samples == 0) {
        return TimeSeriesData();
    }

    vector<Index> all_sample_indices(total_samples);
    iota(all_sample_indices.begin(), all_sample_indices.end(), 0);

    TimeSeriesData ts_data;

    const vector<Index> input_feature_indices = get_feature_indices("Input");
    if(!input_feature_indices.empty())
    {
        const Index input_vars_number = input_feature_indices.size();
        ts_data.inputs.resize(total_samples, past_time_steps, input_vars_number);
        ts_data.inputs.setZero();
        fill_input_tensor(all_sample_indices, input_feature_indices, ts_data.inputs.data());
    }

    const vector<Index> target_feature_indices = get_feature_indices("Target");
    if(!target_feature_indices.empty())
    {
        const Index target_vars_number = target_feature_indices.size();
        ts_data.targets.resize(total_samples, target_vars_number);
        ts_data.targets.setZero();
        fill_target_tensor(all_sample_indices, target_feature_indices, ts_data.targets.data());
    }

    return ts_data;
}


void TimeSeriesDataset::set_past_time_steps(const Index new_past_time_steps)
{
    past_time_steps = new_past_time_steps;
}


void TimeSeriesDataset::set_future_time_steps(const Index new_future_time_steps)
{
    future_time_steps = new_future_time_steps;
}


void TimeSeriesDataset::set_time_variable_index(const Index new_time_variable_index)
{
    time_variable_index = new_time_variable_index;
}


void TimeSeriesDataset::print() const
{
    if(!display) return;

    const Index features_number = get_features_number();
    const Index input_features_number = get_features_number("Input");
    const Index samples_number = get_samples_number();
    const Index target_variables_number = get_features_number("Target");

    cout << "Time series dataset object summary:\n"
         << "Number of samples: " << samples_number << "\n"
         << "Number of variables: " << features_number << "\n"
         << "Number of input variables: " << input_features_number << "\n"
         << "Number of target variables: " << target_variables_number << "\n"
         << "Input variables shape: " << get_shape("Input")
         << "Target variables shape: " << get_shape("Target")
         << "Past time steps: " << get_past_time_steps() << endl
         << "Future time steps: " << get_future_time_steps() << endl;
}


void TimeSeriesDataset::to_XML(XMLPrinter& printer) const
{
    printer.OpenElement("Dataset");

    printer.OpenElement("DataSource");
    add_xml_element(printer, "FileType", "csv");
    add_xml_element(printer, "Path", data_path.string());
    add_xml_element(printer, "Separator", get_separator_name());
    add_xml_element(printer, "HasHeader", to_string(has_header));
    add_xml_element(printer, "HasSamplesId", to_string(has_sample_ids));
    add_xml_element(printer, "MissingValuesLabel", missing_values_label);
    add_xml_element(printer, "LagsNumber", to_string(get_past_time_steps()));
    add_xml_element(printer, "StepsAhead", to_string(get_future_time_steps()));
    add_xml_element(printer, "Codification", get_codification_string());
    printer.CloseElement();

    variables_to_XML(printer);

    samples_to_XML(printer);

    missing_values_to_XML(printer);

    preview_data_to_XML(printer);

    add_xml_element(printer, "Display", to_string(display));

    printer.CloseElement();
}


void TimeSeriesDataset::from_XML(const XMLDocument& data_set_document)
{
    const XMLElement* data_set_element = data_set_document.FirstChildElement("Dataset");
    if(!data_set_element)
        throw runtime_error("Dataset element is nullptr.\n");

    // Data file

    const XMLElement* data_source_element = data_set_element->FirstChildElement("DataSource");
    if(!data_source_element)
        throw runtime_error("Data file element is nullptr.\n");

    const XMLElement* file_type_element = data_source_element->FirstChildElement("FileType");
    if(!file_type_element)
        throw runtime_error("File Type element is nullptr.\n");

    set_data_path(read_xml_string(data_source_element, "Path"));
    set_separator_name(read_xml_string(data_source_element, "Separator"));
    set_has_header(read_xml_bool(data_source_element, "HasHeader"));
    set_has_ids(read_xml_bool(data_source_element, "HasSamplesId"));
    set_missing_values_label(read_xml_string(data_source_element, "MissingValuesLabel"));
    set_past_time_steps(stoi(read_xml_string(data_source_element, "LagsNumber")));
    set_future_time_steps(stoi(read_xml_string(data_source_element, "StepsAhead")));
    set_codification(read_xml_string(data_source_element, "Codification"));

    // Variables

    const XMLElement* variables_element = data_set_element->FirstChildElement("Variables");

    variables_from_XML(variables_element);

    // Samples

    const XMLElement* samples_element = data_set_element->FirstChildElement("Samples");

    samples_from_XML(samples_element);

    // Missing values

    const XMLElement* missing_values_element = data_set_element->FirstChildElement("MissingValues");

    missing_values_from_XML(missing_values_element);

    // Preview data

    const XMLElement* preview_data_element = data_set_element->FirstChildElement("PreviewData");

    preview_data_from_XML(preview_data_element);

    set_display(read_xml_bool(data_set_element, "Display"));

    input_shape = { past_time_steps, get_features_number("Input") };
    target_shape = { get_features_number("Target") };
}


void TimeSeriesDataset::read_csv()
{
    Dataset::read_csv();

    set_default_variables_roles_forecasting();

    const Index features_number = get_features_number();

    if (features_number == 1)
        set_variable_role(0, "InputTarget");
    else
    {
        const vector<Index> target_indices = get_feature_indices("Target");

        if(!target_indices.empty())
        {
            const Index variable_target_index = get_variable_index(target_indices[0]);
            set_variable_role(variable_target_index, "InputTarget");
        }
    }

    input_shape = {past_time_steps, get_features_number("Input")};
    target_shape = {get_features_number("Target")};

    const Index samples_number = get_samples_number();

    if(samples_number > past_time_steps)
        for(Index i = samples_number - past_time_steps; i < samples_number; i++)
            set_sample_role(i, "None");

    split_samples_sequential(type(0.6), type(0.2), type(0.2));
}


void TimeSeriesDataset::impute_missing_values_unuse()
{
    const Index samples_number = get_samples_number();
    const Index lags = get_past_time_steps();

    vector<bool> row_has_nan(samples_number, false);
    for(Index i = 0; i < samples_number; ++i)
        if (has_nan_row(i))
            row_has_nan[i] = true;

    const Index num_sequences = samples_number - lags;
    if (num_sequences < 0) return;

    #pragma omp parallel for
    for(Index i = 0; i < num_sequences; ++i)
    {
        bool sequence_is_invalid = false;

        for(Index j = 0; j <= lags; ++j)
        {
            const Index current_row = i + j;
            if (row_has_nan[current_row])
            {
                sequence_is_invalid = true;
                break;
            }
        }

        if (sequence_is_invalid)
            set_sample_role(i, "None");
    }

    for(Index i = num_sequences; i < samples_number; ++i)
        set_sample_role(i, "None");
}


void TimeSeriesDataset::impute_missing_values_interpolate()
{
    const vector<Index> used_sample_indices = get_used_sample_indices();
    const vector<Index> used_feature_indices = get_used_feature_indices();
    const vector<Index> input_feature_indices = get_feature_indices("Input");
    const vector<Index> target_feature_indices = get_feature_indices("Target");

    const VectorR means = mean(data, used_sample_indices, used_feature_indices);

    const Index used_samples_number = used_sample_indices.size();

    #pragma omp parallel for
    for(Index feature_index = 0; feature_index < get_features_number(); feature_index++)
    {
        for(Index i = 0; i < used_samples_number; i++)
        {
            const Index current_sample_index = used_sample_indices[i];

            if(!isnan(data(current_sample_index, feature_index)))
                continue;

            Index prev_index = i-1;
            type prev_value = NAN;
            while(prev_index >= 0 && isnan(prev_value))
            {
                prev_value = data(used_sample_indices[prev_index], feature_index);
                if(isnan(prev_value)) prev_index--;
            }

            Index start_missing = i;
            Index end_missing = i;
            while(end_missing < used_samples_number && isnan(data(used_sample_indices[end_missing], feature_index)))
            {
                end_missing++;
            }
            Index n_missing = end_missing - start_missing;

            type next_value = NAN;
            if(end_missing < used_samples_number)
                next_value = data(used_sample_indices[end_missing], feature_index);

            for(Index k = 0; k < n_missing; ++k)
            {
                Index sample_k = used_sample_indices[start_missing + k];

                if(isnan(prev_value))
                    data(sample_k, feature_index) = type(next_value);
                else if(isnan(next_value))
                    data(sample_k, feature_index) = type(prev_value);
                else if(!isnan(prev_value) && !isnan(next_value))
                {
                    type fraction = type(k + 1) / type(n_missing + 1);
                    type value_interpolated = prev_value + (next_value - prev_value) * fraction;

                    data(sample_k, feature_index) = value_interpolated;
                }
                else
                    throw runtime_error("The last " + to_string(sample_k-i+1) + " samples are all missing, delete them.\n");
            }
            i = end_missing;
        }
    }
}


void TimeSeriesDataset::fill_input_tensor(const vector<Index>& sample_indices,
                                          const vector<Index>& input_indices,
                                          type* input_data) const
{
    if(sample_indices.empty() || input_indices.empty()) return;

    const Index batch_size = sample_indices.size();
    const Index input_size = input_indices.size();
    const Index total_rows_in_data = data.rows();

    TensorMap3 batch(input_data, batch_size, past_time_steps, input_size);
    const type* const matrix_data = data.data();

#pragma omp parallel for
    for(Index i = 0; i < batch_size; ++i)
    {
        const Index start_row = sample_indices[i];

        for(Index j = 0; j < past_time_steps; ++j)
        {
            const Index actual_row = start_row + j;

            if(actual_row < total_rows_in_data)
            {
                const Index row_offset = actual_row;
                for(Index k = 0; k < input_size; ++k)
                {
                    const Index column_index = input_indices[k];
                    batch(i, j, k) = matrix_data[row_offset + total_rows_in_data * column_index];
                }
            }
            else
            {
                for(Index k = 0; k < input_size; ++k)
                    batch(i, j, k) = static_cast<type>(0);
            }
        }
    }

/*
    if (sample_indices.empty() || input_indices.empty())
        return;

    const Index batch_size = sample_indices.size();
    const Index input_size = input_indices.size();
    const Index total_rows_in_data = data.dimension(0);

    TensorMap3 batch(input_data, batch_size, past_time_steps, input_size);
    const type* const matrix_data = data.data();

#pragma omp parallel for
    for(Index i = 0; i < batch_size; ++i)
    {
        const Index start_row = sample_indices[i];
        for(Index j = 0; j < past_time_steps; ++j)
        {
            const Index actual_row = start_row + j;
            for(Index k = 0; k < input_size; ++k)
            {
                const Index column_index = input_indices[k];

                if (actual_row < total_rows_in_data)
                    batch(i, j, k) = matrix_data[actual_row + total_rows_in_data * column_index];
                else
                    batch(i, j, k) = static_cast<type>(0);
            }
        }
    }
*/
}


void TimeSeriesDataset::fill_target_tensor(const vector<Index>& sample_indices,
                                           const vector<Index>& target_indices,
                                           type* target_tensor_data) const
{
    if(sample_indices.empty() || target_indices.empty()) return;

    const Index batch_size = sample_indices.size();
    const Index target_size = target_indices.size();
    const Index total_rows_in_data = data.rows();

    MatrixMap targets(target_tensor_data, batch_size, target_size);
    const type* const matrix_data = data.data();

#pragma omp parallel for
    for(Index i = 0; i < batch_size; ++i)
    {
        const Index target_row = sample_indices[i] + past_time_steps;

        if(target_row < total_rows_in_data)
        {
            const Index row_offset = target_row;
            for(Index j = 0; j < target_size; ++j)
            {
                const Index column_index = target_indices[j];
                targets(i, j) = matrix_data[row_offset + total_rows_in_data * column_index];
            }
        }
        else
        {
            for(Index j = 0; j < target_size; ++j)
                targets(i, j) = static_cast<type>(0);
        }
    }

/*
    if (sample_indices.empty() || target_indices.empty())
        return;

    const Index batch_size = sample_indices.size();
    const Index target_size = target_indices.size();
    const Index total_rows_in_data = data.dimension(0);

    MatrixMap targets(target_tensor_data, batch_size, target_size);
    const type* const matrix_data = data.data();

#pragma omp parallel for
    for(Index i = 0; i < batch_size; ++i)
    {
        const Index target_row = sample_indices[i] + past_time_steps;
        for(Index j = 0; j < target_size; ++j)
        {
            const Index column_index = target_indices[j];

            if (target_row < total_rows_in_data)
                targets(i, j) = matrix_data[target_row + total_rows_in_data * column_index];
            else
                targets(i, j) = static_cast<type>(0);
        }
    }
*/
}


// @todo Complete method following the structure.

void TimeSeriesDataset::fill_gaps()
{   
    type start_time = 50;
    type end_time = 100;

    type period = 2;

    type new_samples_number = (end_time - start_time)/period;
    type new_features_number = get_features_number();

    Tensor2 new_data(new_samples_number,  new_features_number);

    type timestamp = 0;

    type new_timestamp = 0;

    Index row_index = 0;
    Index column_index = 0;

    for(Index i = 0; i < new_samples_number; i++)
    {
        new_timestamp = start_time + i*period;
        timestamp = new_data(row_index, column_index);

        if(new_timestamp == timestamp)
        {
            data.row(i) = data.row(row_index);

            row_index++;
        }
    }
}


MatrixR TimeSeriesDataset::calculate_autocorrelations(const Index past_time_steps) const
{
    const Index samples_number = get_samples_number();

    if(past_time_steps > samples_number)
        throw runtime_error("Past time steps (" + to_string(past_time_steps) + ") "
                            "is greater than samples number (" + to_string(samples_number) + ") \n");

    const Index variables_number = get_variables_number();

    const Index input_variables_number = get_variables_number("Input");
    const Index target_variables_number = get_variables_number("Target");

    Index input_target_variables_number = input_variables_number;

    const vector<Index> input_variable_indices = get_variable_indices("Input");
    const vector<Index> target_variable_indices = get_variable_indices("Target");

    for(Index i = 0; i < target_variables_number; i++)
        if(variables[target_variable_indices[i]].role != "InputTarget")
            input_target_variables_number++;

    Index input_target_numeric_variables_number = 0;

    int count = 0;

    for(Index i = 0; i < input_target_variables_number; i++)
    {
        if(i < input_variables_number)
        {
            const Index variable_index = input_variable_indices[i];

            const VariableType input_variable_type = variables[variable_index].type;

            if(input_variable_type == VariableType::Numeric)
                input_target_numeric_variables_number++;
        }
        else
        {
            const Index variable_index = target_variable_indices[count];

            const VariableType target_variable_type = variables[variable_index].type;

            if(target_variable_type == VariableType::Numeric)
                input_target_numeric_variables_number++;

            count++;
        }
    }

    const Index new_past_time_steps =
        ((samples_number <= past_time_steps) && past_time_steps > 2) ? past_time_steps - 2 :
         (samples_number == past_time_steps + 1 && past_time_steps > 1) ? past_time_steps - 1 :
         past_time_steps;

    MatrixR autocorrelations(input_target_numeric_variables_number, new_past_time_steps);
    VectorR autocorrelations_vector(new_past_time_steps);
    MatrixR input_i;
    Index counter_i = 0;

    for(Index i = 0; i < variables_number; i++)
    {
        if(variables[i].role == "None" || variables[i].type != VariableType::Numeric)
            continue;

        input_i = get_variable_data(i);
        cout << "Calculating " << variables[i].name << " autocorrelations" << endl;

        const VectorMap current_input_i(input_i.data(), input_i.rows());
        
        autocorrelations_vector = opennn::autocorrelations(current_input_i, new_past_time_steps);

        for(Index j = 0; j < new_past_time_steps; j++)
            autocorrelations (counter_i, j) = autocorrelations_vector(j) ;

        counter_i++;
    }

    return autocorrelations;
}


Tensor3 TimeSeriesDataset::calculate_cross_correlations(const Index past_time_steps) const
{
    const Index samples_number = get_samples_number();

    if(past_time_steps > samples_number)
        throw runtime_error("Past time steps (" + to_string(past_time_steps) + ") is greater than samples number (" + to_string(samples_number) + ") \n");

    const Index variables_number = get_variables_number();

    const Index input_variables_number = get_variables_number("Input");
    const Index target_variables_number = get_variables_number("Target");

    const Index input_target_variables_number = input_variables_number + target_variables_number;

    const vector<Index> input_variable_indices = get_variable_indices("Input");
    const vector<Index> target_variable_indices = get_variable_indices("Target");

    Index input_target_numeric_variables_number = 0;
    int count = 0;

    for(Index i = 0; i < input_target_variables_number; i++)
    {
        if(i < input_variables_number)
        {
            const Index variable_index = input_variable_indices[i];

            const VariableType input_variable_type = variables[variable_index].type;

            if(input_variable_type == VariableType::Numeric)
                input_target_numeric_variables_number++;
        }
        else
        {
            const Index variable_index = target_variable_indices[count];

            const VariableType target_variable_type = variables[variable_index].type;

            if(target_variable_type == VariableType::Numeric && variables[variable_index].role != "InputTarget")
                input_target_numeric_variables_number++;

            count++;
        }
    }

    const Index new_past_time_steps = (samples_number == past_time_steps) ? (past_time_steps - 2)
                                : (samples_number == past_time_steps + 1) ? (past_time_steps - 1)
                                : past_time_steps;

    Tensor3 cross_correlations(input_target_numeric_variables_number,
                                       input_target_numeric_variables_number,
                                       new_past_time_steps);

    VectorR cross_correlations_vector(new_past_time_steps);

    MatrixR input_i;
    MatrixR input_j;

    Index counter_i = 0;
    Index counter_j = 0;

    for(Index i = 0; i < variables_number; i++)
    {
        if(variables[i].role == "None" || variables[i].type != VariableType::Numeric)
            continue;

        input_i = get_variable_data(i);

        if (display) cout << "Calculating " << variables[i].name << " cross correlations:" << endl;

        counter_j = 0;

        for(Index j = 0; j < variables_number; j++)
        {
            if(variables[j].role == "None"
            || variables[j].type != VariableType::Numeric)
                continue;

            input_j = get_variable_data(j);

            if(display) cout << "  vs. " << variables[j].name << endl;
 
            const VectorMap current_input_i(input_i.data(), input_i.rows());
            const VectorMap current_input_j(input_j.data(), input_j.rows());

            cross_correlations_vector = opennn::cross_correlations(current_input_i, current_input_j, new_past_time_steps);

            for(Index k = 0; k < new_past_time_steps; k++)
                cross_correlations(counter_i, counter_j, k) = cross_correlations_vector(k) ;

            counter_j++;
        }

        counter_i++;
    }

    return cross_correlations;
}


Tensor3 TimeSeriesDataset::calculate_cross_correlations_spearman(const Index past_time_steps) const
{
    const Index samples_number = get_samples_number();

    if (past_time_steps > samples_number)
        throw runtime_error("Past time steps (" + to_string(past_time_steps) + ") is greater than samples number (" + to_string(samples_number) + ") \n");

    vector<Index> numeric_vars_indices;

    for(size_t i = 0; i < variables.size(); ++i)
        if(variables[i].role != "None" && variables[i].type == VariableType::Numeric)
            numeric_vars_indices.push_back(i);

    const Index numeric_vars_count = numeric_vars_indices.size();

    if (numeric_vars_count == 0)
        return Tensor3();

    map<Index, VectorR> ranked_series;

    for(Index global_idx : numeric_vars_indices)
    {
        const MatrixR var_data = get_variable_data(global_idx);
        ranked_series[global_idx] = calculate_spearman_ranks(var_data.col(0));
    }

    Tensor3 cross_correlations(numeric_vars_count, numeric_vars_count, past_time_steps);

    #pragma omp parallel for
    for(Index i = 0; i < numeric_vars_count; ++i)
    {
        const VectorR& ranked_series_i = ranked_series.at(numeric_vars_indices[i]);

        for(Index j = 0; j < numeric_vars_count; ++j)
        {
            const VectorR& ranked_series_j = ranked_series.at(numeric_vars_indices[j]);

            const VectorR ccf_vector = opennn::cross_correlations(ranked_series_i, ranked_series_j, past_time_steps);

            for(Index k = 0; k < past_time_steps; ++k)
                cross_correlations(i, j, k) = ccf_vector(k);
        }
    }

    return cross_correlations;
}


vector<vector<Index>> TimeSeriesDataset::get_batches(const vector<Index>& sample_indices,
                                                     Index batch_size,
                                                     bool shuffle) const
{
    // @todo copied from dataset

    if(!shuffle) return split_samples(sample_indices, batch_size);

    const Index samples_number = sample_indices.size();

    const Index batches_number = (samples_number + batch_size - 1) / batch_size;

    vector<vector<Index>> batches(batches_number);

    vector<Index> samples_copy(sample_indices);

    shuffle_vector(samples_copy);

#pragma omp parallel for
    for(Index i = 0; i < batches_number; i++)
    {
        const Index start_index = i * batch_size;

        const Index end_index = min(start_index + batch_size, samples_number);

        batches[i].assign(samples_copy.begin() + start_index,
                          samples_copy.begin() + end_index);
    }

    return batches;
}

}

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
