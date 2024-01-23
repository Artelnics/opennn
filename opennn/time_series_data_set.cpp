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
    time_series_data.resize(0, 0);

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



Tensor<DataSet::RawVariable, 1> TimeSeriesDataSet::get_time_series_columns() const
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
        if(columns(i).type == RawVariableType::Categorical)
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

    if(time_series_columns(column_index).type == RawVariableType::Categorical)
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
        if(time_series_columns(i).type == RawVariableType::DateTime) return i;
    }

    return Index(NAN);
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
        if(time_series_columns(i).type == RawVariableType::Categorical)
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

void TimeSeriesDataSet::transform_time_series_columns()
{
    cout << "Transforming time series columns..." << endl;

    // Categorical columns?

    time_series_columns = columns;

    const Index columns_number = get_columns_number();

    Tensor<RawVariable, 1> new_columns;

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

        if(time_series_columns(column_index).type == RawVariableType::DateTime) continue;

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


void TimeSeriesDataSet::transform_time_series_data()
{
    cout << "Transforming time series data..." << endl;

    // Categorical / Time columns?

    const Index old_samples_number = data.dimension(0);
    const Index old_variables_number = data.dimension(1);


    // steps_ahead = 1;

    const Index new_samples_number = old_samples_number - (lags_number + steps_ahead - 1);
    const Index new_variables_number = has_time_columns() ? (old_variables_number-1) * (lags_number + steps_ahead) : old_variables_number * (lags_number + steps_ahead);

    time_series_data = data;

    data.resize(new_samples_number, new_variables_number);

    Index index = 0;

    for(Index j = 0; j < old_variables_number; j++)
    {
        if(columns(get_column_index(j)).type == RawVariableType::DateTime)
        {
            index++;
            continue;
        }

        for(Index i = 0; i < lags_number+steps_ahead; i++)
        {
            memcpy(data.data() + i*(old_variables_number-index)*new_samples_number + (j-index)*new_samples_number,
                   time_series_data.data() + i + j*old_samples_number,
                   size_t(old_samples_number-lags_number-steps_ahead+1)*sizeof(type));
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

    cout << "STEPS AHEAD TRANSFORM: " << endl;
    cout << steps_ahead << endl;

    transform_time_series_data();

    transform_time_series_columns();

    split_samples_sequential();

    unuse_constant_columns();
}


/// Serializes the data set object into a XML document of the TinyXML library without keep the DOM tree in memory.
///
void TimeSeriesDataSet::write_XML(tinyxml2::XMLPrinter& file_stream) const
{
    ostringstream buffer;

    time_t start, finish;
    time(&start);

    file_stream.OpenElement("DataSet");

    // Data file

    file_stream.OpenElement("DataFile");

    // File type
    {
        file_stream.OpenElement("FileType");

        file_stream.PushText("csv");

        file_stream.CloseElement();
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

    const Index columns_number = get_columns_number();

    // Columns items
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


void TimeSeriesDataSet::from_XML(const tinyxml2::XMLDocument& data_set_document)
{
    ostringstream buffer;

    // Data set element

    const tinyxml2::XMLElement* data_set_element = data_set_document.FirstChildElement("DataSet");

    if(!data_set_element)
    {
        buffer << "OpenNN Exception: DataSet class.\n"
               << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
               << "Data set element is nullptr.\n";

        throw invalid_argument(buffer.str());
    }

    // Data file

    const tinyxml2::XMLElement* data_file_element = data_set_element->FirstChildElement("DataFile");

    if(!data_file_element)
    {
        buffer << "OpenNN Exception: DataSet class.\n"
               << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
               << "Data file element is nullptr.\n";

        throw invalid_argument(buffer.str());
    }

    // Data file name

    const tinyxml2::XMLElement* data_file_name_element = data_file_element->FirstChildElement("DataFileName");

    if(!data_file_name_element)
    {
        buffer << "OpenNN Exception: DataSet class.\n"
               << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
               << "DataFileName element is nullptr.\n";

        throw invalid_argument(buffer.str());
    }

    if(data_file_name_element->GetText())
    {
        const string new_data_file_name = data_file_name_element->GetText();

        set_data_source_path(new_data_file_name);
    }

    // Separator

    const tinyxml2::XMLElement* separator_element = data_file_element->FirstChildElement("Separator");

    if(separator_element)
    {
        if(separator_element->GetText())
        {
            const string new_separator = separator_element->GetText();

            set_separator(new_separator);
        }
        else
        {
            set_separator("Comma");
        }
    }
    else
    {
        set_separator("Comma");
    }

    // Has columns names

    const tinyxml2::XMLElement* columns_names_element = data_file_element->FirstChildElement("ColumnsNames");

    if(columns_names_element)
    {
        const string new_columns_names_string = columns_names_element->GetText();

        try
        {
            set_has_columns_names(new_columns_names_string == "1");
        }
        catch(const invalid_argument& e)
        {
            cerr << e.what() << endl;
        }
    }

    // Rows labels

    const tinyxml2::XMLElement* rows_label_element = data_file_element->FirstChildElement("RowsLabels");

    if(rows_label_element)
    {
        const string new_rows_label_string = rows_label_element->GetText();

        try
        {
            set_has_rows_label(new_rows_label_string == "1");
        }
        catch(const invalid_argument& e)
        {
            cerr << e.what() << endl;
        }
    }

    // Missing values label

    const tinyxml2::XMLElement* missing_values_label_element = data_file_element->FirstChildElement("MissingValuesLabel");

    if(missing_values_label_element)
    {
        if(missing_values_label_element->GetText())
        {
            const string new_missing_values_label = missing_values_label_element->GetText();

            set_missing_values_label(new_missing_values_label);
        }
        else
        {
            set_missing_values_label("NA");
        }
    }
    else
    {
        set_missing_values_label("NA");
    }

    // Forecasting

    // Lags number

    const tinyxml2::XMLElement* lags_number_element = data_file_element->FirstChildElement("LagsNumber");

    if(!lags_number_element)
    {
        buffer << "OpenNN Exception: DataSet class.\n"
               << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
               << "Lags number element is nullptr.\n";

        throw invalid_argument(buffer.str());
    }

    if(lags_number_element->GetText())
    {
        const Index new_lags_number = Index(atoi(lags_number_element->GetText()));

        set_lags_number(new_lags_number);
    }

    // Steps ahead

    const tinyxml2::XMLElement* steps_ahead_element = data_file_element->FirstChildElement("StepsAhead");

    if(!steps_ahead_element)
    {
        buffer << "OpenNN Exception: DataSet class.\n"
               << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
               << "Steps ahead element is nullptr.\n";

        throw invalid_argument(buffer.str());
    }

    if(steps_ahead_element->GetText())
    {
        const Index new_steps_ahead = Index(atoi(steps_ahead_element->GetText()));

        set_steps_ahead_number(new_steps_ahead);
    }

    // Time column

    const tinyxml2::XMLElement* time_column_element = data_file_element->FirstChildElement("TimeColumn");

    if(!time_column_element)
    {
        buffer << "OpenNN Exception: DataSet class.\n"
               << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
               << "Time column element is nullptr.\n";

        throw invalid_argument(buffer.str());
    }

    if(time_column_element->GetText())
    {
        const string new_time_column = time_column_element->GetText();

        set_time_column(new_time_column);
    }

    // Group by column

    const tinyxml2::XMLElement* group_by_column_element = data_file_element->FirstChildElement("GroupByColumn");

    if(!group_by_column_element)
    {
        buffer << "OpenNN Exception: DataSet class.\n"
               << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
               << "Group by column element is nullptr.\n";

        throw invalid_argument(buffer.str());
    }

    if(group_by_column_element->GetText())
    {
        const string new_group_by_column = group_by_column_element->GetText();

        set_group_by_column(new_group_by_column);
    }

    // Codification

    const tinyxml2::XMLElement* codification_element = data_file_element->FirstChildElement("Codification");

    if(codification_element)
    {
        if(codification_element->GetText())
        {
            const string new_codification = codification_element->GetText();

            set_codification(new_codification);
        }
    }

    // Columns

    const tinyxml2::XMLElement* columns_element = data_set_element->FirstChildElement("Columns");

    if(!columns_element)
    {
        buffer << "OpenNN Exception: DataSet class.\n"
               << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
               << "Columns element is nullptr.\n";

        throw invalid_argument(buffer.str());
    }

    // Columns number

    const tinyxml2::XMLElement* columns_number_element = columns_element->FirstChildElement("ColumnsNumber");

    if(!columns_number_element)
    {
        buffer << "OpenNN Exception: DataSet class.\n"
               << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
               << "Columns number element is nullptr.\n";

        throw invalid_argument(buffer.str());
    }

    Index new_columns_number = 0;

    if(columns_number_element->GetText())
    {
        new_columns_number = Index(atoi(columns_number_element->GetText()));

        set_columns_number(new_columns_number);
    }

    // Columns

    const tinyxml2::XMLElement* start_element = columns_number_element;

    if(new_columns_number > 0)
    {
        for(Index i = 0; i < new_columns_number; i++)
        {
            const tinyxml2::XMLElement* column_element = start_element->NextSiblingElement("Column");
            start_element = column_element;

            if(column_element->Attribute("Item") != to_string(i+1))
            {
                buffer << "OpenNN Exception: DataSet class.\n"
                       << "void DataSet:from_XML(const tinyxml2::XMLDocument&) method.\n"
                       << "Column item number (" << i+1 << ") does not match (" << column_element->Attribute("Item") << ").\n";

                throw invalid_argument(buffer.str());
            }

            // Name

            const tinyxml2::XMLElement* name_element = column_element->FirstChildElement("Name");

            if(!name_element)
            {
                buffer << "OpenNN Exception: DataSet class.\n"
                       << "void Column::from_XML(const tinyxml2::XMLDocument&) method.\n"
                       << "Name element is nullptr.\n";

                throw invalid_argument(buffer.str());
            }

            if(name_element->GetText())
            {
                const string new_name = name_element->GetText();

                columns(i).name = new_name;
            }

            // Scaler

            const tinyxml2::XMLElement* scaler_element = column_element->FirstChildElement("Scaler");

            if(!scaler_element)
            {
                buffer << "OpenNN Exception: DataSet class.\n"
                       << "void DataSet::from_XML(const tinyxml2::XMLDocument&) method.\n"
                       << "Scaler element is nullptr.\n";

                throw invalid_argument(buffer.str());
            }

            if(scaler_element->GetText())
            {
                const string new_scaler = scaler_element->GetText();

                columns(i).set_scaler(new_scaler);
            }

            // Column use

            const tinyxml2::XMLElement* column_use_element = column_element->FirstChildElement("ColumnUse");

            if(!column_use_element)
            {
                buffer << "OpenNN Exception: DataSet class.\n"
                       << "void DataSet::from_XML(const tinyxml2::XMLDocument&) method.\n"
                       << "Column use element is nullptr.\n";

                throw invalid_argument(buffer.str());
            }

            if(column_use_element->GetText())
            {
                const string new_column_use = column_use_element->GetText();

                columns(i).set_use(new_column_use);
            }

            // Type

            const tinyxml2::XMLElement* type_element = column_element->FirstChildElement("Type");

            if(!type_element)
            {
                buffer << "OpenNN Exception: DataSet class.\n"
                       << "void Column::from_XML(const tinyxml2::XMLDocument&) method.\n"
                       << "Type element is nullptr.\n";

                throw invalid_argument(buffer.str());
            }

            if(type_element->GetText())
            {
                const string new_type = type_element->GetText();

                columns(i).set_type(new_type);
            }

            if(columns(i).type == RawVariableType::Categorical || columns(i).type == RawVariableType::Binary)
            {
                // Categories

                const tinyxml2::XMLElement* categories_element = column_element->FirstChildElement("Categories");

                if(!categories_element)
                {
                    buffer << "OpenNN Exception: DataSet class.\n"
                           << "void Column::from_XML(const tinyxml2::XMLDocument&) method.\n"
                           << "Categories element is nullptr.\n";

                    throw invalid_argument(buffer.str());
                }

                if(categories_element->GetText())
                {
                    const string new_categories = categories_element->GetText();

                    columns(i).categories = get_tokens(new_categories, ';');
                }

                // Categories uses

                const tinyxml2::XMLElement* categories_uses_element = column_element->FirstChildElement("CategoriesUses");

                if(!categories_uses_element)
                {
                    buffer << "OpenNN Exception: DataSet class.\n"
                           << "void Column::from_XML(const tinyxml2::XMLDocument&) method.\n"
                           << "Categories uses element is nullptr.\n";

                    throw invalid_argument(buffer.str());
                }

                if(categories_uses_element->GetText())
                {
                    const string new_categories_uses = categories_uses_element->GetText();

                    columns(i).set_categories_uses(get_tokens(new_categories_uses, ';'));
                }
            }
        }
    }

    // Time series columns

    const tinyxml2::XMLElement* time_series_columns_element = data_set_element->FirstChildElement("TimeSeriesColumns");

    // Time series columns number

    const tinyxml2::XMLElement* time_series_columns_number_element = time_series_columns_element->FirstChildElement("TimeSeriesColumnsNumber");

    if(!time_series_columns_number_element)
    {
        buffer << "OpenNN Exception: DataSet class.\n"
               << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
               << "Time seires columns number element is nullptr.\n";

        throw invalid_argument(buffer.str());
    }

    Index time_series_new_columns_number = 0;

    if(time_series_columns_number_element->GetText())
    {
        time_series_new_columns_number = Index(atoi(time_series_columns_number_element->GetText()));

        set_time_series_columns_number(time_series_new_columns_number);
    }

    // Time series columns

    const tinyxml2::XMLElement* time_series_start_element = time_series_columns_number_element;

    if(time_series_new_columns_number > 0)
    {
        for(Index i = 0; i < time_series_new_columns_number; i++)
        {
            const tinyxml2::XMLElement* time_series_column_element = time_series_start_element->NextSiblingElement("TimeSeriesColumn");
            time_series_start_element = time_series_column_element;

            if(time_series_column_element->Attribute("Item") != to_string(i+1))
            {
                buffer << "OpenNN Exception: DataSet class.\n"
                       << "void DataSet:from_XML(const tinyxml2::XMLDocument&) method.\n"
                       << "Time series column item number (" << i+1 << ") does not match (" << time_series_column_element->Attribute("Item") << ").\n";

                throw invalid_argument(buffer.str());
            }

            // Name

            const tinyxml2::XMLElement* time_series_name_element = time_series_column_element->FirstChildElement("Name");

            if(!time_series_name_element)
            {
                buffer << "OpenNN Exception: DataSet class.\n"
                       << "void Column::from_XML(const tinyxml2::XMLDocument&) method.\n"
                       << "Time series name element is nullptr.\n";

                throw invalid_argument(buffer.str());
            }

            if(time_series_name_element->GetText())
            {
                const string time_series_new_name = time_series_name_element->GetText();

                time_series_columns(i).name = time_series_new_name;
            }

            // Scaler

            const tinyxml2::XMLElement* time_series_scaler_element = time_series_column_element->FirstChildElement("Scaler");

            if(!time_series_scaler_element)
            {
                buffer << "OpenNN Exception: DataSet class.\n"
                       << "void DataSet::from_XML(const tinyxml2::XMLDocument&) method.\n"
                       << "Time series scaler element is nullptr.\n";

                throw invalid_argument(buffer.str());
            }

            if(time_series_scaler_element->GetText())
            {
                const string time_series_new_scaler = time_series_scaler_element->GetText();

                time_series_columns(i).set_scaler(time_series_new_scaler);
            }

            // Column use

            const tinyxml2::XMLElement* time_series_column_use_element = time_series_column_element->FirstChildElement("ColumnUse");

            if(!time_series_column_use_element)
            {
                buffer << "OpenNN Exception: DataSet class.\n"
                       << "void DataSet::from_XML(const tinyxml2::XMLDocument&) method.\n"
                       << "Time series column use element is nullptr.\n";

                throw invalid_argument(buffer.str());
            }

            if(time_series_column_use_element->GetText())
            {
                const string time_series_new_column_use = time_series_column_use_element->GetText();

                time_series_columns(i).set_use(time_series_new_column_use);
            }

            // Type

            const tinyxml2::XMLElement* time_series_type_element = time_series_column_element->FirstChildElement("Type");

            if(!time_series_type_element)
            {
                buffer << "OpenNN Exception: DataSet class.\n"
                       << "void Column::from_XML(const tinyxml2::XMLDocument&) method.\n"
                       << "Time series type element is nullptr.\n";

                throw invalid_argument(buffer.str());
            }

            if(time_series_type_element->GetText())
            {
                const string time_series_new_type = time_series_type_element->GetText();
                time_series_columns(i).set_type(time_series_new_type);
            }

            if(time_series_columns(i).type == RawVariableType::Categorical || time_series_columns(i).type == RawVariableType::Binary)
            {
                // Categories

                const tinyxml2::XMLElement* time_series_categories_element = time_series_column_element->FirstChildElement("Categories");

                if(!time_series_categories_element)
                {
                    buffer << "OpenNN Exception: DataSet class.\n"
                           << "void Column::from_XML(const tinyxml2::XMLDocument&) method.\n"
                           << "Time series categories element is nullptr.\n";

                    throw invalid_argument(buffer.str());
                }

                if(time_series_categories_element->GetText())
                {
                    const string time_series_new_categories = time_series_categories_element->GetText();

                    time_series_columns(i).categories = get_tokens(time_series_new_categories, ';');
                }

                // Categories uses

                const tinyxml2::XMLElement* time_series_categories_uses_element = time_series_column_element->FirstChildElement("CategoriesUses");

                if(!time_series_categories_uses_element)
                {
                    buffer << "OpenNN Exception: DataSet class.\n"
                           << "void Column::from_XML(const tinyxml2::XMLDocument&) method.\n"
                           << "Time series categories uses element is nullptr.\n";

                    throw invalid_argument(buffer.str());
                }

                if(time_series_categories_uses_element->GetText())
                {
                    const string time_series_new_categories_uses = time_series_categories_uses_element->GetText();

                    time_series_columns(i).set_categories_uses(get_tokens(time_series_new_categories_uses, ';'));
                }
            }
        }
    }

    // Rows label

    if(has_rows_labels)
    {
        // Rows labels begin tag

        const tinyxml2::XMLElement* rows_labels_element = data_set_element->FirstChildElement("RowsLabels");

        if(!rows_labels_element)
        {
            buffer << "OpenNN Exception: DataSet class.\n"
                   << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
                   << "Rows labels element is nullptr.\n";

            throw invalid_argument(buffer.str());
        }

        // Rows labels

        if(rows_labels_element->GetText())
        {
            const string new_rows_labels = rows_labels_element->GetText();

            char separator = ',';

            if(new_rows_labels.find(",") == string::npos
                    && new_rows_labels.find(";") != string::npos) {
                separator = ';';
            }

            rows_labels = get_tokens(new_rows_labels, separator);
        }
    }

    // Samples

    const tinyxml2::XMLElement* samples_element = data_set_element->FirstChildElement("Samples");

    if(!samples_element)
    {
        buffer << "OpenNN Exception: DataSet class.\n"
               << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
               << "Samples element is nullptr.\n";

        throw invalid_argument(buffer.str());
    }

    // Samples number

    const tinyxml2::XMLElement* samples_number_element = samples_element->FirstChildElement("SamplesNumber");

    if(!samples_number_element)
    {
        buffer << "OpenNN Exception: DataSet class.\n"
               << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
               << "Samples number element is nullptr.\n";

        throw invalid_argument(buffer.str());
    }

    if(samples_number_element->GetText())
    {
        const Index new_samples_number = Index(atoi(samples_number_element->GetText()));

        samples_uses.resize(new_samples_number);

        set_training();
    }

    // Samples uses

    const tinyxml2::XMLElement* samples_uses_element = samples_element->FirstChildElement("SamplesUses");

    if(!samples_uses_element)
    {
        buffer << "OpenNN Exception: DataSet class.\n"
               << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
               << "Samples uses element is nullptr.\n";

        throw invalid_argument(buffer.str());
    }

    if(samples_uses_element->GetText())
    {
        set_samples_uses(get_tokens(samples_uses_element->GetText(), ' '));
    }

    // Missing values

    const tinyxml2::XMLElement* missing_values_element = data_set_element->FirstChildElement("MissingValues");

    if(!missing_values_element)
    {
        buffer << "OpenNN Exception: DataSet class.\n"
               << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
               << "Missing values element is nullptr.\n";

        throw invalid_argument(buffer.str());
    }

    // Missing values method

    const tinyxml2::XMLElement* missing_values_method_element = missing_values_element->FirstChildElement("MissingValuesMethod");

    if(!missing_values_method_element)
    {
        buffer << "OpenNN Exception: DataSet class.\n"
               << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
               << "Missing values method element is nullptr.\n";

        throw invalid_argument(buffer.str());
    }

    if(missing_values_method_element->GetText())
    {
        set_missing_values_method(missing_values_method_element->GetText());
    }

    // Missing values number

    const tinyxml2::XMLElement* missing_values_number_element = missing_values_element->FirstChildElement("MissingValuesNumber");

    if(!missing_values_number_element)
    {
        buffer << "OpenNN Exception: DataSet class.\n"
               << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
               << "Missing values number element is nullptr.\n";

        throw invalid_argument(buffer.str());
    }

    if(missing_values_number_element->GetText())
    {
        missing_values_number = Index(atoi(missing_values_number_element->GetText()));
    }

    if(missing_values_number > 0)
    {
        // Columns Missing values number

        const tinyxml2::XMLElement* columns_missing_values_number_element = missing_values_element->FirstChildElement("ColumnsMissingValuesNumber");

        if(!columns_missing_values_number_element)
        {
            buffer << "OpenNN Exception: DataSet class.\n"
                   << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
                   << "Columns missing values number element is nullptr.\n";

            throw invalid_argument(buffer.str());
        }

        if(columns_missing_values_number_element->GetText())
        {
            Tensor<string, 1> new_columns_missing_values_number = get_tokens(columns_missing_values_number_element->GetText(), ' ');

            columns_missing_values_number.resize(new_columns_missing_values_number.size());

            for(Index i = 0; i < new_columns_missing_values_number.size(); i++)
            {
                columns_missing_values_number(i) = atoi(new_columns_missing_values_number(i).c_str());
            }
        }

        // Rows missing values number

        const tinyxml2::XMLElement* rows_missing_values_number_element = missing_values_element->FirstChildElement("RowsMissingValuesNumber");

        if(!rows_missing_values_number_element)
        {
            buffer << "OpenNN Exception: DataSet class.\n"
                   << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
                   << "Rows missing values number element is nullptr.\n";

            throw invalid_argument(buffer.str());
        }

        if(rows_missing_values_number_element->GetText())
        {
            rows_missing_values_number = Index(atoi(rows_missing_values_number_element->GetText()));
        }
    }

    // Preview data

    const tinyxml2::XMLElement* preview_data_element = data_set_element->FirstChildElement("PreviewData");

    if(!preview_data_element)
    {
        buffer << "OpenNN Exception: DataSet class.\n"
               << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
               << "Preview data element is nullptr.\n";

        throw invalid_argument(buffer.str());
    }

    // Preview size

    const tinyxml2::XMLElement* preview_size_element = preview_data_element->FirstChildElement("PreviewSize");

    if(!preview_size_element)
    {
        buffer << "OpenNN Exception: DataSet class.\n"
               << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
               << "Preview size element is nullptr.\n";

        throw invalid_argument(buffer.str());
    }

    Index new_preview_size = 0;

    if(preview_size_element->GetText())
    {
        new_preview_size = Index(atoi(preview_size_element->GetText()));

        if(new_preview_size > 0) data_file_preview.resize(new_preview_size);
    }

    // Preview data

    start_element = preview_size_element;

    for(Index i = 0; i < new_preview_size; i++)
    {
        const tinyxml2::XMLElement* row_element = start_element->NextSiblingElement("Row");
        start_element = row_element;

        if(row_element->Attribute("Item") != to_string(i+1))
        {
            buffer << "OpenNN Exception: DataSet class.\n"
                   << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
                   << "Row item number (" << i+1 << ") does not match (" << row_element->Attribute("Item") << ").\n";

            throw invalid_argument(buffer.str());
        }

        if(row_element->GetText())
        {
            data_file_preview(i) = get_tokens(row_element->GetText(), ',');
        }
    }

    // Display

    const tinyxml2::XMLElement* display_element = data_set_element->FirstChildElement("Display");

    if(display_element)
    {
        const string new_display_string = display_element->GetText();

        try
        {
            set_display(new_display_string != "0");
        }
        catch(const invalid_argument& e)
        {
            cerr << e.what() << endl;
        }
    }
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
                            data(current_sample, current_variable) = type(next_value);
                        }
                        else if(isnan(next_value))
                        {
                            data(current_sample, current_variable) = type(preview_value);
                        }
                        else
                        {
                            data(current_sample, current_variable) = type((preview_value + next_value)/2);
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

            const RawVariableType input_column_type = time_series_columns(column_index).type;

            if(input_column_type == RawVariableType::Numeric)
            {
                input_target_numeric_column_number++;
            }
        }
        else
        {
            const Index column_index = target_columns_indices(counter);

            const RawVariableType target_column_type = time_series_columns(column_index).type;

            if(target_column_type == RawVariableType::Numeric)
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
        if(time_series_columns(i).column_use != VariableUse::Unused && time_series_columns(i).type == RawVariableType::Numeric)
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

            const RawVariableType input_column_type = time_series_columns(column_index).type;

            if(input_column_type == RawVariableType::Numeric)
            {
                input_target_numeric_column_number++;
            }
        }
        else
        {
            const Index column_index = target_columns_indices(counter);

            RawVariableType target_column_type = time_series_columns(column_index).type;

            if(target_column_type == RawVariableType::Numeric)
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
        if(time_series_columns(i).column_use != VariableUse::Unused && time_series_columns(i).type == RawVariableType::Numeric)
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
            if(time_series_columns(j).column_use != VariableUse::Unused && time_series_columns(j).type == RawVariableType::Numeric)
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
