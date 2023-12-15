//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   T I M E   S E R I E S   D A T A S E T   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "time_series_data_set.h"
#include "correlations.h"
#include "tensor_utilities.h"

namespace opennn
{

TimeSeriesDataSet::TimeSeriesDataSet() : DataSet()
{
    time_series_data.resize(0,0);

    lags_number = 0;

    steps_ahead = 0;
}


/// File and separator constructor. It creates a data set object by loading the object members from a data file.
/// It also sets a separator.
/// Please mind about the file format. This is specified in the User's Guide.
/// @param data_source_path Data file name.
/// @param separator Data file separator between columns.
/// @param has_columns_names True if data file contains a row with columns names, False otherwise.
/// @param data_codification String codification of the input file

TimeSeriesDataSet::TimeSeriesDataSet(const string& data_source_path, const char& separator, const bool& has_columns_names, const Codification& data_codification)
{
    set(data_source_path, separator, has_columns_names, data_codification);
}


/// Returns the number of columns in the time series.

Index TimeSeriesDataSet::get_time_series_columns_number() const
{
    return time_series_columns.size();
}



Tensor<DataSet::Column, 1> TimeSeriesDataSet::get_time_series_columns() const
{
    return time_series_columns;
}


/// Returns the indices of the time variables in the data set.

const string& TimeSeriesDataSet::get_time_column() const
{
    return time_column;
}


const string& TimeSeriesDataSet::get_group_by_column() const
{
    return group_by_column;
}


/// Returns the number of variables in the time series data.

Index TimeSeriesDataSet::get_time_series_numeric_variables_number() const
{
    Index variables_number = 0;

    for(Index i = 0; i < time_series_columns.size(); i++)
    {
        if(columns(i).type == ColumnType::Categorical)
        {
            variables_number += time_series_columns(i).categories.size();
        }
        else
        {
            variables_number++;
        }
    }

    return variables_number;
}



/// Returns the number of lags to be used in a time series prediction application.

const Index& TimeSeriesDataSet::get_lags_number() const
{
    return lags_number;
}


/// Returns the number of steps ahead to be used in a time series prediction application.

const Index& TimeSeriesDataSet::get_steps_ahead() const
{
    return steps_ahead;
}


Index TimeSeriesDataSet::get_time_series_data_rows_number() const
{
    return time_series_data.dimension(0);
}


/// Returns the data from the time series column with a given index,
/// @param column_index Index of the column.

Tensor<type, 2> TimeSeriesDataSet::get_time_series_column_data(const Index& column_index) const
{
    Index columns_number = 1;

    const Index rows_number = time_series_data.dimension(0);

    if(time_series_columns(column_index).type == ColumnType::Categorical)
    {
        columns_number = time_series_columns(column_index).get_categories_number();
    }

    const Eigen::array<Index, 2> extents = {rows_number, columns_number};
    const Eigen::array<Index, 2> offsets = {0, get_numeric_variable_indices(column_index)(0)};

    return time_series_data.slice(offsets, extents);
}



Tensor<string, 1> TimeSeriesDataSet::get_time_series_columns_names() const
{
    const Index columns_number = get_time_series_columns_number();

    Tensor<string, 1> columns_names(columns_number);

    for(Index i = 0; i < columns_number; i++)
    {
        columns_names(i) = time_series_columns(i).name;
    }

    return columns_names;
}



Index TimeSeriesDataSet::get_input_time_series_columns_number() const
{
    Index input_columns_number = 0;

    for(Index i = 0; i < time_series_columns.size(); i++)
    {
        if(time_series_columns(i).column_use == VariableUse::Input)
        {
            input_columns_number++;
        }
    }

    return input_columns_number;
}



Index TimeSeriesDataSet::get_target_time_series_columns_number() const
{
    Index target_columns_number = 0;

    for(Index i = 0; i < time_series_columns.size(); i++)
    {
        if(time_series_columns(i).column_use == VariableUse::Target)
        {
            target_columns_number++;
        }
    }

    return target_columns_number;
}



Index TimeSeriesDataSet::get_time_series_time_column_index() const
{
    for(Index i = 0; i < time_series_columns.size(); i++)
    {
        if(time_series_columns(i).type == ColumnType::DateTime) return i;
    }

    return static_cast<Index>(NAN);
}



Tensor<Index, 1> TimeSeriesDataSet::get_target_time_series_columns_indices() const
{
    const Index target_columns_number = get_target_time_series_columns_number();

    Tensor<Index, 1> target_columns_indices(target_columns_number);

    Index index = 0;

    for(Index i = 0; i < time_series_columns.size(); i++)
    {
        if(time_series_columns(i).column_use == VariableUse::Target)
        {
            target_columns_indices(index) = i;
            index++;
        }
    }

    return target_columns_indices;
}



/// Returns a string vector with the names of all the variables in the time series data.
/// The size of the vector is the number of variables.

Tensor<string, 1> TimeSeriesDataSet::get_time_series_variables_names() const
{
    const Index variables_number = get_time_series_numeric_variables_number();

    Tensor<string, 1> variables_names(variables_number);

    Index index = 0;

    for(Index i = 0; i < time_series_columns.size(); i++)
    {
        if(time_series_columns(i).type == ColumnType::Categorical)
        {
            for(Index j = 0; j < time_series_columns(i).categories.size(); j++)
            {
                variables_names(index) = time_series_columns(i).categories(j);

                index++;
            }
        }
        else
        {
            variables_names(index) = time_series_columns(i).name;
            index++;
        }
    }

    return variables_names;
}



Tensor<Index, 1> TimeSeriesDataSet::get_input_time_series_columns_indices() const
{
    const Index input_columns_number = get_input_time_series_columns_number();

    Tensor<Index, 1> input_columns_indices(input_columns_number);

    Index index = 0;

    for(Index i = 0; i < time_series_columns.size(); i++)
    {
        if(time_series_columns(i).column_use == VariableUse::Input)
        {
            input_columns_indices(index) = i;
            index++;
        }
    }

    return input_columns_indices;
}



/// Sets a new number of lags to be defined for a time series prediction application.
/// When loading the data file, the time series data will be modified according to this number.
/// @param new_lags_number Number of lags(x-1, ..., x-l) to be used.

void TimeSeriesDataSet::set_lags_number(const Index& new_lags_number)
{
    lags_number = new_lags_number;
}


/// Sets a new number of steps ahead to be defined for a time series prediction application.
/// When loading the data file, the time series data will be modified according to this number.
/// @param new_steps_ahead_number Number of steps ahead to be used.

void TimeSeriesDataSet::set_steps_ahead_number(const Index& new_steps_ahead_number)
{
    steps_ahead = new_steps_ahead_number;
}


void TimeSeriesDataSet::set_time_series_columns_number(const Index& new_variables_number)
{
    time_series_columns.resize(new_variables_number);
}


void TimeSeriesDataSet::set_time_series_data(const Tensor<type, 2>& new_data)
{
    time_series_data = new_data;
}


/// Sets the new position where the time data is located in the data set.
/// @param new_time_index Position where the time data is located.

void TimeSeriesDataSet::set_time_column(const string& new_time_column)
{
    time_column = new_time_column;
}


void TimeSeriesDataSet::set_group_by_column(const string& new_group_by_column)
{
    group_by_column = new_group_by_column;
}


/// This method transforms the columns into time series for forecasting problems.

void TimeSeriesDataSet::

transform_time_series_columns()
{
    cout << "Transforming time series columns..." << endl;

    // Categorical columns?

    time_series_columns = columns;

    const Index columns_number = get_columns_number();

    Tensor<Column, 1> new_columns;

    if(has_time_columns())
    {
        // @todo check if there are more than one time column
        new_columns.resize((columns_number-1)*(lags_number+steps_ahead));
    }
    else
    {
        new_columns.resize(columns_number*(lags_number+steps_ahead));
    }

    Index lag_index = lags_number - 1;
    Index ahead_index = 0;
    Index column_index = 0;
    Index new_column_index = 0;

    for(Index i = 0; i < columns_number*(lags_number+steps_ahead); i++)
    {
        column_index = i%columns_number;

        if(time_series_columns(column_index).type == ColumnType::DateTime) continue;

        if(i < lags_number*columns_number)
        {
            new_columns(new_column_index).name = columns(column_index).name + "_lag_" + to_string(lag_index);

            new_columns(new_column_index).categories_uses.resize(columns(column_index).get_categories_number());
            new_columns(new_column_index).set_use(VariableUse::Input);

            new_columns(new_column_index).type = columns(column_index).type;
            new_columns(new_column_index).categories = columns(column_index).categories;

            new_column_index++;
        }
        else if(i == columns_number*(lags_number+steps_ahead) - 1)
        {
            new_columns(new_column_index).name = columns(column_index).name + "_ahead_" + to_string(ahead_index);

            new_columns(new_column_index).type = columns(column_index).type;
            new_columns(new_column_index).categories = columns(column_index).categories;

            new_columns(new_column_index).categories_uses.resize(columns(column_index).get_categories_number());
            new_columns(new_column_index).set_use(VariableUse::Target);

            new_column_index++;
        }
        else
        {
            new_columns(new_column_index).name = columns(column_index).name + "_ahead_" + to_string(ahead_index);

            new_columns(new_column_index).type = columns(column_index).type;
            new_columns(new_column_index).categories = columns(column_index).categories;

            new_columns(new_column_index).categories_uses.resize(columns(column_index).get_categories_number());
            new_columns(new_column_index).set_use(VariableUse::Unused);

            new_column_index++;
        }

        if(lag_index > 0 && column_index == columns_number - 1)
        {
            lag_index--;
        }
        else if(column_index == columns_number - 1)
        {
            ahead_index++;
        }
    }

    columns = new_columns;
}

/*
void TimeSeriesDataSet::transform_time_series_data()
{
    cout << "Transforming time series data..." << endl;

    // Categorical / Time columns?

    const Index old_samples_number = data.dimension(0);
    const Index old_variables_number = data.dimension(1);

    const Index new_samples_number = old_samples_number - (lags_number + steps_ahead - 1);
    const Index new_variables_number = has_time_columns() ? (old_variables_number-1) * (lags_number + steps_ahead) : old_variables_number * (lags_number + steps_ahead);

    time_series_data = data;

    Tensor<type, 2> no_time_data(new_samples_number, new_variables_number);
    data.resize(new_samples_number, new_variables_number + 1);

    Tensor<type, 2> timestamp_index_column(new_samples_number, 1);

    Index index = 0;

    for(Index j = 0; j < old_variables_number; j++)
    {
       if(columns(get_column_index(j)).type == ColumnType::DateTime)
       {
            time_variable_index = j;
            Tensor<type, 1> timestamp_raw(time_series_data.chip(j, 1));

            Index time_index = 0;

            for(Index i = lags_number; i < lags_number + new_samples_number; i++)
            {
                timestamp_index_column(time_index, 0) = timestamp_raw(i);
                time_index++;
            }

            index++;
            continue;
       }

        for(Index i = 0; i < lags_number+steps_ahead; i++)
        {
            memcpy(no_time_data.data() + i * (old_variables_number - index) * new_samples_number + (j - index) * new_samples_number,
                   time_series_data.data() + i + j * old_samples_number,
                   static_cast<size_t>(old_samples_number - lags_number - steps_ahead + 1) * sizeof(type));
        }
    }

    data = timestamp_index_column.concatenate(no_time_data, 1);

    samples_uses.resize(new_samples_number);
    split_samples_random();
}
*/

void TimeSeriesDataSet::transform_time_series_data()
{
    cout << "Transforming time series data..." << endl;

    // Categorical / Time columns?

    const Index old_samples_number = data.dimension(0);
    const Index old_variables_number = data.dimension(1);

    const Index new_samples_number = old_samples_number - (lags_number + steps_ahead - 1);
    const Index new_variables_number = has_time_columns() ? (old_variables_number-1) * (lags_number + steps_ahead) : old_variables_number * (lags_number + steps_ahead);

    time_series_data = data;

    data.resize(new_samples_number, new_variables_number);

    Index index = 0;

    for(Index j = 0; j < old_variables_number; j++)
    {
        if(columns(get_column_index(j)).type == ColumnType::DateTime)
        {
            index++;
            continue;
        }

        for(Index i = 0; i < lags_number+steps_ahead; i++)
        {
            memcpy(data.data() + i*(old_variables_number-index)*new_samples_number + (j-index)*new_samples_number,
                   time_series_data.data() + i + j*old_samples_number,
                   static_cast<size_t>(old_samples_number-lags_number-steps_ahead+1)*sizeof(type));
        }
    }

    samples_uses.resize(new_samples_number);
    split_samples_random();
}


/// Arranges an input-target DataSet from a time series matrix, according to the number of lags.

void TimeSeriesDataSet::transform_time_series()
{
    cout << "Transforming time series..." << endl;

    if(lags_number == 0 || steps_ahead == 0) return;

    transform_time_series_data();

    fill_time_series_gaps();

    transform_time_series_columns();

    split_samples_sequential();

    unuse_constant_columns();
}


/// Serializes the data set object into a XML document of the TinyXML library without keep the DOM tree in memory.

void TimeSeriesDataSet::write_XML(tinyxml2::XMLPrinter& file_stream) const
{
    ostringstream buffer;

    time_t start, finish;
    time(&start);

    file_stream.OpenElement("DataSet");

    // Data file

    file_stream.OpenElement("DataFile");

    if(model_type != ModelType::ImageClassification)
    {
        // File type
        {
            file_stream.OpenElement("FileType");

            file_stream.PushText("csv");

            file_stream.CloseElement();
        }
    }
    else
    {
        // File type
        {
            file_stream.OpenElement("FileType");

            file_stream.PushText("bmp");

            file_stream.CloseElement();
        }
    }

    // Data file name
    {
        file_stream.OpenElement("DataFileName");

        file_stream.PushText(data_source_path.c_str());

        file_stream.CloseElement();
    }

    // Separator
    {
        file_stream.OpenElement("Separator");

        file_stream.PushText(get_separator_string().c_str());

        file_stream.CloseElement();
    }

    // Columns names
    {
        file_stream.OpenElement("ColumnsNames");

        buffer.str("");
        buffer << has_columns_names;

        file_stream.PushText(buffer.str().c_str());

        file_stream.CloseElement();
    }

    // Rows labels
    {
        file_stream.OpenElement("RowsLabels");

        buffer.str("");

        buffer << has_rows_labels;

        file_stream.PushText(buffer.str().c_str());

        file_stream.CloseElement();
    }

    // Missing values label
    {
        file_stream.OpenElement("MissingValuesLabel");

        file_stream.PushText(missing_values_label.c_str());

        file_stream.CloseElement();
    }

    // Lags number
    {
        file_stream.OpenElement("LagsNumber");

        buffer.str("");
        buffer << get_lags_number();

        file_stream.PushText(buffer.str().c_str());

        file_stream.CloseElement();
    }

    // Steps Ahead
    {
        file_stream.OpenElement("StepsAhead");

        buffer.str("");
        buffer << get_steps_ahead();

        file_stream.PushText(buffer.str().c_str());

        file_stream.CloseElement();
    }

    // Time Column
    {
        file_stream.OpenElement("TimeColumn");

        buffer.str("");
        buffer << get_time_column();

        file_stream.PushText(buffer.str().c_str());

        file_stream.CloseElement();
    }

    // Group by Column
    {
        file_stream.OpenElement("GroupByColumn");

        buffer.str("");
        buffer << get_time_column();

        file_stream.PushText(buffer.str().c_str());

        file_stream.CloseElement();
    }

    // Codification

    file_stream.OpenElement("Codification");

    buffer.str("");
    buffer << get_codification_string();

    file_stream.PushText(buffer.str().c_str());

    file_stream.CloseElement();

    // Close DataFile

    file_stream.CloseElement();

    // Columns

    file_stream.OpenElement("Columns");

    // Columns number
    {
        file_stream.OpenElement("ColumnsNumber");

        buffer.str("");
        buffer << get_columns_number();

        file_stream.PushText(buffer.str().c_str());

        file_stream.CloseElement();
    }

    // Columns items

    const Index columns_number = get_columns_number();

    {
        for(Index i = 0; i < columns_number; i++)
        {
            file_stream.OpenElement("Column");

            file_stream.PushAttribute("Item", to_string(i+1).c_str());

            columns(i).write_XML(file_stream);

            file_stream.CloseElement();
        }
    }

    // Close columns

    file_stream.CloseElement();

    // Time series columns

    const Index time_series_columns_number = get_time_series_columns_number();

    if(time_series_columns_number != 0)
    {
        file_stream.OpenElement("TimeSeriesColumns");

        // Time series columns number
        {
            file_stream.OpenElement("TimeSeriesColumnsNumber");

            buffer.str("");
            buffer << get_time_series_columns_number();

            file_stream.PushText(buffer.str().c_str());

            file_stream.CloseElement();
        }

        // Time series columns items

        for(Index i = 0; i < time_series_columns_number; i++)
        {
            file_stream.OpenElement("TimeSeriesColumn");

            file_stream.PushAttribute("Item", to_string(i+1).c_str());

            time_series_columns(i).write_XML(file_stream);

            file_stream.CloseElement();
        }

        // Close time series columns

        file_stream.CloseElement();
    }

    // Rows labels

    if(has_rows_labels)
    {
        const Index rows_labels_number = rows_labels.size();

        file_stream.OpenElement("RowsLabels");

        buffer.str("");

        for(Index i = 0; i < rows_labels_number; i++)
        {
            buffer << rows_labels(i);

            if(i != rows_labels_number-1) buffer << ",";
        }

        file_stream.PushText(buffer.str().c_str());

        file_stream.CloseElement();
    }

    // Samples

    file_stream.OpenElement("Samples");

    // Samples number
    {
        file_stream.OpenElement("SamplesNumber");

        buffer.str("");
        buffer << get_samples_number();

        file_stream.PushText(buffer.str().c_str());

        file_stream.CloseElement();
    }

    // Samples uses

    {
        file_stream.OpenElement("SamplesUses");

        buffer.str("");

        const Index samples_number = get_samples_number();

        for(Index i = 0; i < samples_number; i++)
        {
            SampleUse sample_use = samples_uses(i);

            buffer << Index(sample_use);

            if(i < (samples_number-1)) buffer << " ";
        }

        file_stream.PushText(buffer.str().c_str());

        file_stream.CloseElement();
    }

    // Close samples

    file_stream.CloseElement();

    // Missing values

    file_stream.OpenElement("MissingValues");

    // Missing values method
    {
        file_stream.OpenElement("MissingValuesMethod");

        if(missing_values_method == MissingValuesMethod::Mean)
        {
            file_stream.PushText("Mean");
        }
        else if(missing_values_method == MissingValuesMethod::Median)
        {
            file_stream.PushText("Median");
        }
        else if(missing_values_method == MissingValuesMethod::Unuse)
        {
            file_stream.PushText("Unuse");
        }
        else if(missing_values_method == MissingValuesMethod::Interpolation)
        {
            file_stream.PushText("Interpolation");
        }

        file_stream.CloseElement();
    }

    // Missing values number
    {
        file_stream.OpenElement("MissingValuesNumber");

        buffer.str("");
        buffer << missing_values_number;

        file_stream.PushText(buffer.str().c_str());

        file_stream.CloseElement();
    }

    if(missing_values_number > 0)
    {
        // Columns missing values number
        {
            file_stream.OpenElement("ColumnsMissingValuesNumber");

            const Index columns_number = columns_missing_values_number.size();

            buffer.str("");

            for(Index i = 0; i < columns_number; i++)
            {
                buffer << columns_missing_values_number(i);

                if(i != (columns_number-1)) buffer << " ";
            }

            file_stream.PushText(buffer.str().c_str());

            file_stream.CloseElement();
        }

        // Rows missing values number
        {
            file_stream.OpenElement("RowsMissingValuesNumber");

            buffer.str("");
            buffer << rows_missing_values_number;

            file_stream.PushText(buffer.str().c_str());

            file_stream.CloseElement();
        }
    }

    // Missing values

    file_stream.CloseElement();

    // Preview data

    file_stream.OpenElement("PreviewData");

    file_stream.OpenElement("PreviewSize");

    buffer.str("");

        buffer << data_file_preview.size();

        file_stream.PushText(buffer.str().c_str());

        file_stream.CloseElement();

        for(Index i = 0; i < data_file_preview.size(); i++)
        {
            file_stream.OpenElement("Row");

            file_stream.PushAttribute("Item", to_string(i+1).c_str());

            for(Index j = 0; j < data_file_preview(i).size(); j++)
            {
                file_stream.PushText(data_file_preview(i)(j).c_str());

                if(j != data_file_preview(i).size()-1)
                {
                    file_stream.PushText(",");
                }
            }

            file_stream.CloseElement();
        }

    // Close preview data

    file_stream.CloseElement();

    // Close data set

    file_stream.CloseElement();

    time(&finish);
}


/// Substitutes all the missing values by the mean of the corresponding variable.

void TimeSeriesDataSet::impute_missing_values_mean()
{
    const Tensor<Index, 1> used_samples_indices = get_used_samples_indices();
    const Tensor<Index, 1> used_variables_indices = get_used_variables_indices();
    const Tensor<Index, 1> input_variables_indices = get_input_variables_indices();
    const Tensor<Index, 1> target_variables_indices = get_target_numeric_variables_indices();

    const Tensor<type, 1> means = mean(data, used_samples_indices, used_variables_indices);

    const Index samples_number = used_samples_indices.size();
    const Index variables_number = used_variables_indices.size();
    const Index target_variables_number = target_variables_indices.size();

    Index current_variable;
    Index current_sample;

    if(lags_number == 0 && steps_ahead == 0)
    {
#pragma omp parallel for schedule(dynamic)

        for(Index j = 0; j < variables_number - target_variables_number; j++)
        {
            current_variable = input_variables_indices(j);

            for(Index i = 0; i < samples_number; i++)
            {
                current_sample = used_samples_indices(i);

                if(isnan(data(current_sample, current_variable)))
                {
                    data(current_sample,current_variable) = means(j);
                }
            }
        }

        for(Index j = 0; j < target_variables_number; j++)
        {
            current_variable = target_variables_indices(j);
            for(Index i = 0; i < samples_number; i++)
            {
                current_sample = used_samples_indices(i);

                if(isnan(data(current_sample, current_variable)))
                {
                    set_sample_use(i, "Unused");
                }
            }
        }
    }
    else
    {
        for(Index j = 0; j < get_numeric_variables_number(); j++)
        {
            current_variable = j;

            for(Index i = 0; i < samples_number; i++)
            {
                current_sample = used_samples_indices(i);

                if(isnan(data(current_sample, current_variable)))
                {
                    if(i < lags_number || i > samples_number - steps_ahead)
                    {
                        data(current_sample, current_variable) = means(j);
                    }
                    else
                    {
                        Index k = i;
                        double preview_value = NAN, next_value = NAN;

                        while(isnan(preview_value) && k > 0)
                        {
                            k--;
                            preview_value = data(used_samples_indices(k), current_variable);
                        }

                        k = i;

                        while(isnan(next_value) && k < samples_number)
                        {
                            k++;
                            next_value = data(used_samples_indices(k), current_variable);
                        }

                        if(isnan(preview_value) && isnan(next_value))
                        {
                            ostringstream buffer;

                            buffer << "OpenNN Exception: DataSet class.\n"
                                   << "void DataSet::impute_missing_values_mean() const.\n"
                                   << "The last " << (samples_number - i) + 1 << " samples are all missing, delete them.\n";

                            throw invalid_argument(buffer.str());
                        }

                        if(isnan(preview_value))
                        {
                            data(current_sample, current_variable) = next_value;
                        }
                        else if(isnan(next_value))
                        {
                            data(current_sample, current_variable) = preview_value;
                        }
                        else
                        {
                            data(current_sample, current_variable) = (preview_value + next_value) / 2.0;
                        }
                    }
                }
            }
        }
    }
}


void TimeSeriesDataSet::quicksort_by_column(Index target_column)
{
    Tensor<type, 2>* data_matrix_ptr = get_data_pointer();
    const Index rows_number = data_matrix_ptr->dimension(0);

    quick_sort(*data_matrix_ptr, 0, rows_number - 1, target_column);
}


/// @todo bad quality code.

void TimeSeriesDataSet::fill_time_series_gaps()
{
/*
    const Index rows_number = data.dimension(0);
    const Index columns_number = data.dimension(1);

    quicksort_by_column(0);

    const Tensor<type, 1> time_data = data.chip(0, 1);

    const Tensor<type, 1> time_difference_data = calculate_delta(time_data);

    const Tensor<type, 1> time_step_mode = opennn::mode(time_difference_data);

    const Tensor<type, 1> time_series_filled = fill_gaps_by_value(time_data, time_difference_data, time_step_mode(0));

    Tensor<type, 2> NaN_data(time_series_filled.size(), columns_number);
    NaN_data.setConstant(type(NAN));

    NaN_data.chip(0, 1) = time_series_filled;

    Tensor<type, 2> final_data(data);

    data.resize(time_data.size() + time_series_filled.size(), columns_number);

    data = final_data.concatenate(NaN_data, 0);
*/
}


/// Returns a matrix with the values of autocorrelation for every variable in the data set.
/// The number of rows is equal to the number of
/// The number of columns is the maximum lags number.
/// @param maximum_lags_number Maximum lags number for which autocorrelation is calculated.

Tensor<type, 2> TimeSeriesDataSet::calculate_autocorrelations(const Index& lags_number) const
{
    const Index samples_number = time_series_data.dimension(0);

    if(lags_number > samples_number)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: DataSet class.\n"
               << "Tensor<type, 2> calculate_cross_correlations(const Index& lags_number) const method.\n"
               << "Lags number(" << lags_number
               << ") is greater than the number of samples("
               << samples_number << ") \n";

        throw invalid_argument(buffer.str());
    }

    const Index columns_number = get_time_series_columns_number();

    const Index input_columns_number = get_input_time_series_columns_number();
    const Index target_columns_number = get_target_time_series_columns_number();

    const Index input_target_columns_number = input_columns_number + target_columns_number;

    const Tensor<Index, 1> input_columns_indices = get_input_time_series_columns_indices();
    const Tensor<Index, 1> target_columns_indices = get_target_time_series_columns_indices();

    Index input_target_numeric_column_number = 0;

    int counter = 0;

    for(Index i = 0; i < input_target_columns_number; i++)
    {
        if(i < input_columns_number)
        {
            const Index column_index = input_columns_indices(i);

            const ColumnType input_column_type = time_series_columns(column_index).type;

            if(input_column_type == ColumnType::Numeric)
            {
                input_target_numeric_column_number++;
            }
        }
        else
        {
            const Index column_index = target_columns_indices(counter);

            const ColumnType target_column_type = time_series_columns(column_index).type;

            if(target_column_type == ColumnType::Numeric)
            {
                input_target_numeric_column_number++;
            }

            counter++;
        }
    }

    Index new_lags_number;

    if((samples_number == lags_number || samples_number < lags_number) && lags_number > 2)
    {
        new_lags_number = lags_number - 2;
    }
    else if(samples_number == lags_number + 1 && lags_number > 1)
    {
        new_lags_number = lags_number - 1;
    }
    else
    {
        new_lags_number = lags_number;
    }

    Tensor<type, 2> autocorrelations(input_target_numeric_column_number, new_lags_number);
    Tensor<type, 1> autocorrelations_vector(new_lags_number);
    Tensor<type, 2> input_i;
    Index counter_i = 0;

    for(Index i = 0; i < columns_number; i++)
    {
        if(time_series_columns(i).column_use != VariableUse::Unused && time_series_columns(i).type == ColumnType::Numeric)
        {
            input_i = get_time_series_column_data(i);
            cout << "Calculating " << time_series_columns(i).name << " autocorrelations" << endl;
        }
        else
        {
            continue;
        }

        const TensorMap<Tensor<type, 1>> current_input_i(input_i.data(), input_i.dimension(0));

        autocorrelations_vector = opennn::autocorrelations(thread_pool_device, current_input_i, new_lags_number);

        for(Index j = 0; j < new_lags_number; j++)
        {
            autocorrelations (counter_i, j) = autocorrelations_vector(j) ;
        }

        counter_i++;
    }

    return autocorrelations;
}


/// Calculates the cross-correlation between all the variables in the data set.

Tensor<type, 3> TimeSeriesDataSet::calculate_cross_correlations(const Index& lags_number) const
{
    const Index samples_number = time_series_data.dimension(0);

    if(lags_number > samples_number)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: DataSet class.\n"
               << "Tensor<type, 3> calculate_cross_correlations(const Index& lags_number) const method.\n"
               << "Lags number(" << lags_number
               << ") is greater than the number of samples("
               << samples_number << ") \n";

        throw invalid_argument(buffer.str());
    }

    const Index columns_number = get_time_series_columns_number();

    const Index input_columns_number = get_input_time_series_columns_number();
    const Index target_columns_number = get_target_time_series_columns_number();

    const Index input_target_columns_number = input_columns_number + target_columns_number;

    const Tensor<Index, 1> input_columns_indices = get_input_time_series_columns_indices();
    const Tensor<Index, 1> target_columns_indices = get_target_time_series_columns_indices();

    Index input_target_numeric_column_number = 0;
    int counter = 0;

    for(Index i = 0; i < input_target_columns_number; i++)
    {
        if(i < input_columns_number)
        {
            const Index column_index = input_columns_indices(i);

            const ColumnType input_column_type = time_series_columns(column_index).type;

            if(input_column_type == ColumnType::Numeric)
            {
                input_target_numeric_column_number++;
            }
        }
        else
        {
            const Index column_index = target_columns_indices(counter);

            ColumnType target_column_type = time_series_columns(column_index).type;

            if(target_column_type == ColumnType::Numeric)
            {
                input_target_numeric_column_number++;
            }
            counter++;
        }
    }

    Index new_lags_number;

    if(samples_number == lags_number)
    {
        new_lags_number = lags_number - 2;
    }
    else if(samples_number == lags_number + 1)
    {
        new_lags_number = lags_number - 1;
    }
    else
    {
        new_lags_number = lags_number;
    }

    Tensor<type, 3> cross_correlations(input_target_numeric_column_number, input_target_numeric_column_number, new_lags_number);
    Tensor<type, 1> cross_correlations_vector(new_lags_number);
    Tensor<type, 2> input_i;
    Tensor<type, 2> input_j;
    Index counter_i = 0;
    Index counter_j = 0;

    for(Index i = 0; i < columns_number; i++)
    {
        if(time_series_columns(i).column_use != VariableUse::Unused && time_series_columns(i).type == ColumnType::Numeric)
        {
            input_i = get_time_series_column_data(i);

            if(display) cout << "Calculating " << time_series_columns(i).name << " cross correlations:" << endl;

            counter_j = 0;
        }
        else
        {
            continue;
        }

        for(Index j = 0; j < columns_number; j++)
        {
            if(time_series_columns(j).column_use != VariableUse::Unused && time_series_columns(j).type == ColumnType::Numeric)
            {
                input_j = get_time_series_column_data(j);

                if(display) cout << "   vs. " << time_series_columns(j).name << endl;

            }
            else
            {
                continue;
            }

            const TensorMap<Tensor<type, 1>> current_input_i(input_i.data(), input_i.dimension(0));
            const TensorMap<Tensor<type, 1>> current_input_j(input_j.data(), input_j.dimension(0));

            cross_correlations_vector = opennn::cross_correlations(thread_pool_device, current_input_i, current_input_j, new_lags_number);

            for(Index k = 0; k < new_lags_number; k++)
            {
                cross_correlations (counter_i, counter_j, k) = cross_correlations_vector(k) ;
            }

            counter_j++;
        }

        counter_i++;

    }

    return cross_correlations;
}


/// This method loads time series data from a binary data file.

void TimeSeriesDataSet::load_time_series_data_binary(const string& time_series_data_file_name)
{
    std::ifstream file;

    file.open(time_series_data_file_name.c_str(), ios::binary);

    if(!file.is_open())
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: DataSet class.\n"
               << "void load_time_series_data_binary(const string&) method.\n"
               << "Cannot open binary file: " << time_series_data_file_name << "\n";

        throw invalid_argument(buffer.str());
    }

    streamsize size = sizeof(Index);

    Index columns_number;
    Index rows_number;

    file.read(reinterpret_cast<char*>(&columns_number), size);
    file.read(reinterpret_cast<char*>(&rows_number), size);

    size = sizeof(type);

    type value;

    time_series_data.resize(rows_number, columns_number);

    for(Index i = 0; i < rows_number*columns_number; i++)
    {
        file.read(reinterpret_cast<char*>(&value), size);

        time_series_data(i) = value;
    }

    file.close();
}


/// Saves to the data file the values of the time series data matrix in binary format.

void TimeSeriesDataSet::save_time_series_data_binary(const string& binary_data_file_name) const
{
    std::ofstream file(binary_data_file_name.c_str(), ios::binary);

    if(!file.is_open())
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: DataSet class." << endl
               << "void save_time_series_data_binary(const string) method." << endl
               << "Cannot open data binary file." << endl;

        throw invalid_argument(buffer.str());
    }

    // Write data

    streamsize size = sizeof(Index);

    Index columns_number = time_series_data.dimension(1);
    Index rows_number = time_series_data.dimension(0);

    cout << "Saving binary data file..." << endl;

    file.write(reinterpret_cast<char*>(&columns_number), size);
    file.write(reinterpret_cast<char*>(&rows_number), size);

    size = sizeof(type);

    type value;

    for(int i = 0; i < columns_number; i++)
    {
        for(int j = 0; j < rows_number; j++)
        {
            value = time_series_data(j,i);

            file.write(reinterpret_cast<char*>(&value), size);
        }
    }

    file.close();

    cout << "Binary data file saved." << endl;
}

}


// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2023 Artificial Intelligence Techniques, SL.
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
