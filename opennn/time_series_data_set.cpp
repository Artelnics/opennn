//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   T I M E   S E R I E S   D A T A S E T   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "time_series_data_set.h"
#include "statistics.h"
#include "correlations.h"
#include "tensors.h"
#include "strings_utilities.h"

namespace opennn
{

TimeSeriesDataSet::TimeSeriesDataSet() : DataSet()
{
    time_series_data.resize(0, 0);

    time_series_raw_variables.resize(0);

    lags_number = 0;

    steps_ahead = 0;
}


TimeSeriesDataSet::TimeSeriesDataSet(const string& data_path,
                                     const string& separator,
                                     const bool& has_header,
                                     const bool& has_samples_id,
                                     const Index& new_lags_number,
                                     const Index& new_steps_ahead,
                                     const Codification& data_codification)
{
    set(data_path, separator, has_header, has_samples_id, data_codification);

    lags_number = new_lags_number;
    steps_ahead = new_steps_ahead;

    transform_time_series();
}


Index TimeSeriesDataSet::get_time_series_raw_variables_number() const
{
    return time_series_raw_variables.size();
}


Tensor<DataSet::RawVariable, 1> TimeSeriesDataSet::get_time_series_raw_variables() const
{
    return time_series_raw_variables;
}


const string& TimeSeriesDataSet::get_time_raw_variable() const
{
    //return time_column; @todo
    return "";
}


const string& TimeSeriesDataSet::get_group_by_column() const
{
    //return group_by_column; @todo
    return "";
}


Index TimeSeriesDataSet::get_time_series_variables_number() const
{
    const Index time_series_raw_variables_number = time_series_raw_variables.size();

    Index variables_number = 0;

    for(Index i = 0; i < time_series_raw_variables_number; i++)
    {
        if(raw_variables(i).type == RawVariableType::Categorical)
        {
            variables_number += time_series_raw_variables(i).categories.size();
        }
        else
        {
            variables_number++;
        }
    }

    return variables_number;
}


const Index& TimeSeriesDataSet::get_lags_number() const
{
    return lags_number;
}


const Index& TimeSeriesDataSet::get_steps_ahead() const
{
    return steps_ahead;
}


Index TimeSeriesDataSet::get_time_series_data_rows_number() const
{
    return time_series_data.dimension(0);
}


Tensor<type, 2> TimeSeriesDataSet::get_time_series_raw_variable_data(const Index& raw_variable_index) const
{
    Index raw_variables_number = 1;

    const Index rows_number = time_series_data.dimension(0);

    if(time_series_raw_variables(raw_variable_index).type == RawVariableType::Categorical)
    {
        raw_variables_number = time_series_raw_variables(raw_variable_index).get_categories_number();
    }

    const Eigen::array<Index, 2> extents = {rows_number, raw_variables_number};
    const Eigen::array<Index, 2> offsets = {0, get_variable_indices(raw_variable_index)(0)};

    return time_series_data.slice(offsets, extents);
}


Tensor<string, 1> TimeSeriesDataSet::get_time_series_raw_variables_names() const
{
    const Index raw_variables_number = get_time_series_raw_variables_number();

    Tensor<string, 1> raw_variables_names(raw_variables_number);

    for(Index i = 0; i < raw_variables_number; i++)
    {
        raw_variables_names(i) = time_series_raw_variables(i).name;
    }

    return raw_variables_names;
}


Index TimeSeriesDataSet::get_input_time_series_raw_variables_number() const
{
    Index input_raw_variables_number = 0;

    for(Index i = 0; i < time_series_raw_variables.size(); i++)
    {
        if(time_series_raw_variables(i).use == VariableUse::Input)
        {
            input_raw_variables_number++;
        }
    }

    return input_raw_variables_number;
}


Index TimeSeriesDataSet::get_target_time_series_raw_variables_number() const
{
    Index target_raw_variables_number = 0;

    for(Index i = 0; i < time_series_raw_variables.size(); i++)
    {
        if(time_series_raw_variables(i).use == VariableUse::Target)
        {
            target_raw_variables_number++;
        }
    }

    return target_raw_variables_number;
}


Index TimeSeriesDataSet::get_time_series_time_raw_variable_index() const
{
    for(Index i = 0; i < time_series_raw_variables.size(); i++)
    {
        if(time_series_raw_variables(i).type == RawVariableType::DateTime) return i;
    }

    return Index(NAN);
}


Tensor<Index, 1> TimeSeriesDataSet::get_target_time_series_raw_variables_indices() const
{
    const Index target_raw_variables_number = get_target_time_series_raw_variables_number();

    Tensor<Index, 1> target_raw_variables_indices(target_raw_variables_number);

    Index index = 0;

    for(Index i = 0; i < time_series_raw_variables.size(); i++)
    {
        if(time_series_raw_variables(i).use == VariableUse::Target)
        {
            target_raw_variables_indices(index) = i;
            index++;
        }
    }

    return target_raw_variables_indices;
}


Tensor<string, 1> TimeSeriesDataSet::get_time_series_variables_names() const
{
    const Index variables_number = get_time_series_variables_number();

    Tensor<string, 1> variables_names(variables_number);

    Index index = 0;

    for(Index i = 0; i < time_series_raw_variables.size(); i++)
    {
        if(time_series_raw_variables(i).type == RawVariableType::Categorical)
        {
            for(Index j = 0; j < time_series_raw_variables(i).categories.size(); j++)
            {
                variables_names(index) = time_series_raw_variables(i).categories(j);

                index++;
            }
        }
        else
        {
            variables_names(index) = time_series_raw_variables(i).name;
            index++;
        }
    }

    return move(variables_names);
}


Tensor<Index, 1> TimeSeriesDataSet::get_input_time_series_raw_variables_indices() const
{
    const Index input_raw_variables_number = get_input_time_series_raw_variables_number();

    Tensor<Index, 1> input_raw_variables_indices(input_raw_variables_number);

    Index index = 0;

    for(Index i = 0; i < time_series_raw_variables.size(); i++)
    {
        if(time_series_raw_variables(i).use == VariableUse::Input)
        {
            input_raw_variables_indices(index) = i;
            index++;
        }
    }

    return input_raw_variables_indices;
}


void TimeSeriesDataSet::set_lags_number(const Index& new_lags_number)
{
    lags_number = new_lags_number;
}


void TimeSeriesDataSet::set_steps_ahead_number(const Index& new_steps_ahead_number)
{
    steps_ahead = new_steps_ahead_number;
}


void TimeSeriesDataSet::set_time_series_raw_variables_number(const Index& new_variables_number)
{
    time_series_raw_variables.resize(new_variables_number);
}


void TimeSeriesDataSet::set_time_series_data(const Tensor<type, 2>& new_data)
{
    time_series_data = new_data;
}



void TimeSeriesDataSet::set_time_raw_variable(const string& new_time_column)
{
//    time_column = new_time_column;
}


void TimeSeriesDataSet::set_group_by_raw_variable(const string& new_group_by_column)
{
//    group_by_column = new_group_by_column;
}


void TimeSeriesDataSet::transform_time_series_raw_variables()
{
    cout << "Transforming time series raw variables..." << endl;

    // Categorical raw_variables?

    time_series_raw_variables = raw_variables;

    const Index raw_variables_number = get_raw_variables_number();

    Tensor<RawVariable, 1> new_raw_variables;

    const Index time_raw_variables_number = get_time_raw_variables_number();

    if(time_raw_variables_number == 0)
    {
        new_raw_variables.resize(raw_variables_number*(lags_number+steps_ahead));
    }
    else if(time_raw_variables_number == 1)
    {
        new_raw_variables.resize((raw_variables_number-1)*(lags_number+steps_ahead));
    }
    else
    {
        throw runtime_error("More than 1 time variable.");
    }

    Index lag_index = lags_number - 1;
    Index ahead_index = 0;
    Index raw_variable_index = 0;
    Index new_raw_variable_index = 0;

    for(Index i = 0; i < raw_variables_number*(lags_number+steps_ahead); i++)
    {
        raw_variable_index = i%raw_variables_number;

        if(time_series_raw_variables(raw_variable_index).type == RawVariableType::DateTime) continue;

        if(i < lags_number*raw_variables_number)
        {            
            new_raw_variables(new_raw_variable_index).name = raw_variables(raw_variable_index).name + "_lag_" + to_string(lag_index);

            new_raw_variables(new_raw_variable_index).set_use(VariableUse::Input);

            new_raw_variables(new_raw_variable_index).type = raw_variables(raw_variable_index).type;
            new_raw_variables(new_raw_variable_index).categories = raw_variables(raw_variable_index).categories;

            new_raw_variable_index++;
        }
        else if(i == raw_variables_number*(lags_number+steps_ahead) - 1)
        {            
            new_raw_variables(new_raw_variable_index).name = raw_variables(raw_variable_index).name + "_ahead_" + to_string(ahead_index);

            new_raw_variables(new_raw_variable_index).type = raw_variables(raw_variable_index).type;
            new_raw_variables(new_raw_variable_index).categories = raw_variables(raw_variable_index).categories;

            new_raw_variables(new_raw_variable_index).set_use(VariableUse::Target);

            new_raw_variable_index++;            
        }
        else
        {

            new_raw_variables(new_raw_variable_index).name = raw_variables(raw_variable_index).name + "_ahead_" + to_string(ahead_index);

            new_raw_variables(new_raw_variable_index).type = raw_variables(raw_variable_index).type;
            new_raw_variables(new_raw_variable_index).categories = raw_variables(raw_variable_index).categories;

            new_raw_variables(new_raw_variable_index).set_use(VariableUse::None);

            new_raw_variable_index++;
        }

        if(lag_index > 0 && raw_variable_index == raw_variables_number - 1)
        {
            lag_index--;
        }
        else if(raw_variable_index == raw_variables_number - 1)
        {
            ahead_index++;
        }
    }

    raw_variables = new_raw_variables;
}


void TimeSeriesDataSet::transform_time_series_data()
{
    cout << "Transforming time series data..." << endl;

    // Categorical / Time raw_variables?

    const Index old_samples_number = data.dimension(0);
    const Index old_variables_number = data.dimension(1);

    const Index time_raw_variables_number = get_time_raw_variables_number();

    const Index new_samples_number = old_samples_number - (lags_number + steps_ahead - 1);

    const Index new_variables_number = time_raw_variables_number == 1
                                     ? (old_variables_number - 1) * (lags_number + steps_ahead)
                                     : old_variables_number * (lags_number + steps_ahead);

    time_series_data = data;

    data.resize(new_samples_number, new_variables_number);

    Index index = 0;

    for(Index j = 0; j < old_variables_number; j++)
    {
        if(raw_variables(get_raw_variable_index(j)).type == RawVariableType::DateTime)
        {
            index++;
            continue;
        }

        for(Index i = 0; i < lags_number+steps_ahead; i++)
        {
            copy(time_series_data.data() + i + j * old_samples_number,
                 time_series_data.data() + i + j * old_samples_number + old_samples_number - lags_number - steps_ahead + 1,
                 data.data() + i * (old_variables_number - index) * new_samples_number + (j - index) * new_samples_number);
        }
    }

    samples_uses.resize(new_samples_number);

    split_samples_random();
}


void TimeSeriesDataSet::transform_time_series()
{
    cout << "Transforming time series..." << endl;

    if(lags_number == 0 || steps_ahead == 0) return;

    transform_time_series_data();

    transform_time_series_raw_variables();

    split_samples_sequential();

    unuse_constant_raw_variables();
}


void TimeSeriesDataSet::print() const
{
    if(!display) return;

    const Index variables_number = get_variables_number();
    const Index input_variables_number = get_input_variables_number();
    const Index samples_number = get_samples_number();
    const Index target_variables_bumber = get_target_variables_number();

    cout << "Time series data set object summary:\n"
         << "Number of samples: " << samples_number << "\n"
         << "Number of variables: " << variables_number << "\n"
         << "Number of input variables: " << input_variables_number << "\n"
         << "Number of targets: " << target_variables_bumber << "\n"
         << "Input variables dimensions: ";

    print_dimensions(input_dimensions);

    cout << "Target variables dimensions: ";

    print_dimensions(target_dimensions);
}


void TimeSeriesDataSet::to_XML(tinyxml2::XMLPrinter& file_stream) const
{
    file_stream.OpenElement("TimeSeriesDataSet");

    // Data file

    file_stream.OpenElement("DataSource");

    // File type

    file_stream.OpenElement("FileType");
    file_stream.PushText("csv");
    file_stream.CloseElement();

    // Data source path

    file_stream.OpenElement("Path");
    file_stream.PushText(data_path.c_str());
    file_stream.CloseElement();

    // Separator

    file_stream.OpenElement("Separator");
    file_stream.PushText(get_separator_string().c_str());
    file_stream.CloseElement();

    // Has header

    file_stream.OpenElement("HasHeader");
    file_stream.PushText(to_string(has_header).c_str());
    file_stream.CloseElement();

    // Samples id

    file_stream.OpenElement("HasSamplesId");
    file_stream.PushText(to_string(has_samples_id).c_str());
    file_stream.CloseElement();

    // Missing values label

    file_stream.OpenElement("MissingValuesLabel");
    file_stream.PushText(missing_values_label.c_str());
    file_stream.CloseElement();

    // Lags number

    file_stream.OpenElement("LagsNumber");
    file_stream.PushText(to_string(get_lags_number()).c_str());
    file_stream.CloseElement();

    // Steps Ahead

    file_stream.OpenElement("StepsAhead");
    file_stream.PushText(to_string(get_steps_ahead()).c_str());
    file_stream.CloseElement();

    // Time raw variable
/*
    file_stream.OpenElement("TimeRawVariable");
    file_stream.PushText(get_time_raw_variable().c_str());
    file_stream.CloseElement();

    // Group by raw_variable
    {
        file_stream.OpenElement("GroupByRawVariable");

        buffer.str("");
        buffer << get_time_raw_variable();

        file_stream.PushText(buffer.str().c_str());

        file_stream.CloseElement();
    }
*/
    // Codification

    file_stream.OpenElement("Codification");
    file_stream.PushText(get_codification_string().c_str());
    file_stream.CloseElement();

    // Close DataFile

    file_stream.CloseElement();

    // Raw variables

    file_stream.OpenElement("RawVariables");

    // Raw variables number

    file_stream.OpenElement("RawVariablesNumber");
    file_stream.PushText(to_string(get_raw_variables_number()).c_str());
    file_stream.CloseElement();

    const Index raw_variables_number = get_raw_variables_number();

    // Raw variables items

    for(Index i = 0; i < raw_variables_number; i++)
    {
        file_stream.OpenElement("RawVariable");
        file_stream.PushAttribute("Item", to_string(i+1).c_str());
        raw_variables(i).to_XML(file_stream);
        file_stream.CloseElement();
    }

    // Close raw_variables

    file_stream.CloseElement();

    // Time series raw_variables

    const Index time_series_raw_variables_number = get_time_series_raw_variables_number();

    if(time_series_raw_variables_number != 0)
    {
        file_stream.OpenElement("TimeSeriesRawVariables");

        // Time series raw_variables number

            file_stream.OpenElement("TimeSeriesRawVariablesNumber");
            file_stream.PushText(to_string(get_time_series_raw_variables_number()).c_str());
            file_stream.CloseElement();

        // Time series raw_variables items

        for(Index i = 0; i < time_series_raw_variables_number; i++)
        {
            file_stream.OpenElement("TimeSeriesRawVariable");
            file_stream.PushAttribute("Item", to_string(i+1).c_str());
            time_series_raw_variables(i).to_XML(file_stream);
            file_stream.CloseElement();
        }

        // Close time series raw variables

        file_stream.CloseElement();
    }

    // Ids

    if(has_samples_id)
    {
        file_stream.OpenElement("HasSamplesId");
        file_stream.PushText(string_tensor_to_string(get_ids()).c_str());
        file_stream.CloseElement();
    }

    // Samples

    file_stream.OpenElement("Samples");

    // Samples number

    file_stream.OpenElement("SamplesNumber");
    file_stream.PushText(to_string(get_samples_number()).c_str());
    file_stream.CloseElement();

    // Samples uses

    file_stream.OpenElement("SamplesUses");
    file_stream.PushText(tensor_to_string(get_samples_uses_tensor()).c_str());
    file_stream.CloseElement();

    // Close samples

    file_stream.CloseElement();

    // Missing values

    file_stream.OpenElement("MissingValues");
    file_stream.PushText(get_missing_values_method_string().c_str());
    file_stream.CloseElement();

    // Missing values number

    file_stream.OpenElement("MissingValuesNumber");
    file_stream.PushText(to_string(missing_values_number).c_str());
    file_stream.CloseElement();
/*
    if(missing_values_number > 0)
    {
        // Raw variables missing values number
        {
            file_stream.OpenElement("RawVariablesMissingValuesNumber");

            const Index raw_variables_number = raw_variables_missing_values_number.size();

            buffer.str("");

            for(Index i = 0; i < raw_variables_number; i++)
            {
                buffer << raw_variables_missing_values_number(i);

                if(i != (raw_variables_number-1)) buffer << " ";
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
*/
    // Missing values

    file_stream.CloseElement();

    // Preview data
/*
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
*/
    // Close data set

    file_stream.CloseElement();
}


void TimeSeriesDataSet::from_XML(const tinyxml2::XMLDocument& data_set_document)
{
    // Data set element

    const tinyxml2::XMLElement* data_set_element = data_set_document.FirstChildElement("DataSet");

    if(!data_set_element)
        throw runtime_error("Data set element is nullptr.\n");

    // Data file

    const tinyxml2::XMLElement* data_source_element = data_set_element->FirstChildElement("DataSource");

    if(!data_source_element)
        throw runtime_error("Data file element is nullptr.\n");

    // Data file name

    const tinyxml2::XMLElement* data_source_path_element = data_source_element->FirstChildElement("Path");

    if(!data_source_path_element)
        throw runtime_error("Path element is nullptr.\n");

    if(data_source_path_element->GetText())
        set_data_source_path(data_source_path_element->GetText());

    // Separator

    const tinyxml2::XMLElement* separator_element = data_source_element->FirstChildElement("Separator");

    if(separator_element)
        if(separator_element->GetText())
            set_separator_name(separator_element->GetText());

    // Has raw_variables names

    const tinyxml2::XMLElement* raw_variables_names_element = data_source_element->FirstChildElement("HasHeader");

    if(raw_variables_names_element)
        set_has_header(raw_variables_names_element->GetText() == string("1"));

    // Samples id

    const tinyxml2::XMLElement* rows_label_element = data_source_element->FirstChildElement("HasSamplesId");

    if(rows_label_element)
        set_has_ids(rows_label_element->GetText() == string("1"));

    // Missing values label

    const tinyxml2::XMLElement* missing_values_label_element = data_source_element->FirstChildElement("MissingValuesLabel");

    if(missing_values_label_element)
        if(missing_values_label_element->GetText())
            set_missing_values_label(missing_values_label_element->GetText());

    // Forecasting

    // Lags number

    const tinyxml2::XMLElement* lags_number_element = data_source_element->FirstChildElement("LagsNumber");

    if(!lags_number_element)
        throw runtime_error("Lags number element is nullptr.\n");

    if(lags_number_element->GetText())
        set_lags_number(Index(atoi(lags_number_element->GetText())));

    // Steps ahead

    const tinyxml2::XMLElement* steps_ahead_element = data_source_element->FirstChildElement("StepsAhead");

    if(!steps_ahead_element)
        throw runtime_error("Steps ahead element is nullptr.\n");

    if(steps_ahead_element->GetText())
        set_steps_ahead_number(Index(atoi(steps_ahead_element->GetText())));

    // Time raw_variable

    const tinyxml2::XMLElement* time_raw_variable_element = data_source_element->FirstChildElement("TimeRawVariable");

    if(!time_raw_variable_element)
        throw runtime_error("Time raw_variable element is nullptr.\n");

    if(time_raw_variable_element->GetText())
        set_time_raw_variable(time_raw_variable_element->GetText());

    // Group by raw_variable

    const tinyxml2::XMLElement* group_by_raw_variable_element = data_source_element->FirstChildElement("GroupByRawVariable");

    if(!group_by_raw_variable_element)
        throw runtime_error("Group by raw_variable element is nullptr.\n");

    if(group_by_raw_variable_element->GetText())
        set_group_by_raw_variable(group_by_raw_variable_element->GetText());

    // Codification

    const tinyxml2::XMLElement* codification_element = data_source_element->FirstChildElement("Codification");

    if(codification_element)
        if(codification_element->GetText())
            set_codification(codification_element->GetText());

    // Raw variables

    const tinyxml2::XMLElement* raw_variables_element = data_set_element->FirstChildElement("RawVariables");

    if(!raw_variables_element)
        throw runtime_error("RawVariables element is nullptr.\n");

    // Raw variables number

    const tinyxml2::XMLElement* raw_variables_number_element = raw_variables_element->FirstChildElement("RawVariablesNumber");

    if(!raw_variables_number_element)
        throw runtime_error("RawVariablesNumber element is nullptr.\n");

    if(raw_variables_number_element->GetText())
        set_raw_variables_number(Index(atoi(raw_variables_number_element->GetText())));

    // Raw variables

    const tinyxml2::XMLElement* start_element = raw_variables_number_element;

    for(Index i = 0; i < raw_variables.size(); i++)
    {
        const tinyxml2::XMLElement* raw_variable_element = start_element->NextSiblingElement("RawVariable");
        start_element = raw_variable_element;

        if(raw_variable_element->Attribute("Item") != to_string(i+1))
            throw runtime_error("Raw variable item number (" + to_string(i+1) + ") does not match (" + raw_variable_element->Attribute("Item") + ").\n");

        // Name

        const tinyxml2::XMLElement* name_element = raw_variable_element->FirstChildElement("Name");

        if(!name_element)
            throw runtime_error("Name element is nullptr.\n");

        if(name_element->GetText())
        {
            const string new_name = name_element->GetText();

            raw_variables(i).name = new_name;
        }

        // Scaler

        const tinyxml2::XMLElement* scaler_element = raw_variable_element->FirstChildElement("Scaler");

        if(!scaler_element)
            throw runtime_error("Scaler element is nullptr.\n");

        if(scaler_element->GetText())
        {
            const string new_scaler = scaler_element->GetText();

            raw_variables(i).set_scaler(new_scaler);
        }

        // raw_variable use

        const tinyxml2::XMLElement* use_element = raw_variable_element->FirstChildElement("Use");

        if(!use_element)
            throw runtime_error("Raw variable use element is nullptr.\n");

        if(use_element->GetText())
        {
            const string new_raw_variable_use = use_element->GetText();

            raw_variables(i).set_use(new_raw_variable_use);
        }

        // Type

        const tinyxml2::XMLElement* type_element = raw_variable_element->FirstChildElement("Type");

        if(!type_element)
            throw runtime_error("Type element is nullptr.\n");

        if(type_element->GetText())
        {
            const string new_type = type_element->GetText();

            raw_variables(i).set_type(new_type);
        }

        if(raw_variables(i).type == RawVariableType::Categorical || raw_variables(i).type == RawVariableType::Binary)
        {
            // Categories

            const tinyxml2::XMLElement* categories_element = raw_variable_element->FirstChildElement("Categories");

            if(!categories_element)
                throw runtime_error("Categories element is nullptr.\n");

            if(categories_element->GetText())
            {
                const string new_categories = categories_element->GetText();

                raw_variables(i).categories = get_tokens(new_categories, " ");
            }
        }
    }

    // Time series raw_variables

    const tinyxml2::XMLElement* time_series_raw_variables_element = data_set_element->FirstChildElement("TimeSeriesRawVariables");

    // Time series raw_variables number

    const tinyxml2::XMLElement* time_series_raw_variables_number_element = time_series_raw_variables_element->FirstChildElement("TimeSeriesRawVariablesNumber");

    if(!time_series_raw_variables_number_element)
        throw runtime_error("Time seires raw_variables number element is nullptr.\n");

    Index time_series_new_raw_variables_number = 0;

    if(time_series_raw_variables_number_element->GetText())
    {
        time_series_new_raw_variables_number = Index(atoi(time_series_raw_variables_number_element->GetText()));

        set_time_series_raw_variables_number(time_series_new_raw_variables_number);
    }

    // Time series raw_variables

    const tinyxml2::XMLElement* time_series_start_element = time_series_raw_variables_number_element;

    if(time_series_new_raw_variables_number > 0)
    {
        for(Index i = 0; i < time_series_new_raw_variables_number; i++)
        {
            const tinyxml2::XMLElement* time_series_raw_variable_element = time_series_start_element->NextSiblingElement("TimeSeriesRawVariable");
            time_series_start_element = time_series_raw_variable_element;

            if(time_series_raw_variable_element->Attribute("Item") != to_string(i+1))
                throw runtime_error("Time series raw_variable item number (" + to_string(i+1) + ") "
                                    "does not match (" + time_series_raw_variable_element->Attribute("Item") + ").\n");

            // Name

            const tinyxml2::XMLElement* time_series_name_element = time_series_raw_variable_element->FirstChildElement("Name");

            if(!time_series_name_element)
                throw runtime_error("Time series name element is nullptr.\n");

            if(time_series_name_element->GetText())
            {
                const string time_series_new_name = time_series_name_element->GetText();

                time_series_raw_variables(i).name = time_series_new_name;
            }

            // Scaler

            const tinyxml2::XMLElement* time_series_scaler_element = time_series_raw_variable_element->FirstChildElement("Scaler");

            if(!time_series_scaler_element)
                throw runtime_error("Time series scaler element is nullptr.\n");

            if(time_series_scaler_element->GetText())
            {
                const string time_series_new_scaler = time_series_scaler_element->GetText();

                time_series_raw_variables(i).set_scaler(time_series_new_scaler);
            }

            // raw_variable use

            const tinyxml2::XMLElement* time_series_raw_variable_use_element = time_series_raw_variable_element->FirstChildElement("Use");

            if(!time_series_raw_variable_use_element)
                throw runtime_error("Time series raw_variable use element is nullptr.\n");

            if(time_series_raw_variable_use_element->GetText())
            {
                const string time_series_new_raw_variable_use = time_series_raw_variable_use_element->GetText();

                time_series_raw_variables(i).set_use(time_series_new_raw_variable_use);
            }

            // Type

            const tinyxml2::XMLElement* time_series_type_element = time_series_raw_variable_element->FirstChildElement("Type");

            if(!time_series_type_element)
                throw runtime_error("Time series type element is nullptr.\n");

            if(time_series_type_element->GetText())
            {
                const string time_series_new_type = time_series_type_element->GetText();
                time_series_raw_variables(i).set_type(time_series_new_type);
            }

            if(time_series_raw_variables(i).type == RawVariableType::Categorical || time_series_raw_variables(i).type == RawVariableType::Binary)
            {
                // Categories

                const tinyxml2::XMLElement* time_series_categories_element = time_series_raw_variable_element->FirstChildElement("Categories");

                if(!time_series_categories_element)
                    throw runtime_error("Time series categories element is nullptr.\n");

                if(time_series_categories_element->GetText())
                {
                    const string time_series_new_categories = time_series_categories_element->GetText();

                    time_series_raw_variables(i).categories = get_tokens(time_series_new_categories, ";");
                }
            }
        }
    }

    // Rows label

    if(has_samples_id)
    {
        // Samples id begin tag

        const tinyxml2::XMLElement* has_ids_element = data_set_element->FirstChildElement("HasSamplesId");

        if(!has_ids_element)
            throw runtime_error("Rows labels element is nullptr.\n");

        // Samples id

        if(has_ids_element->GetText())
        {
            const string new_rows_labels = has_ids_element->GetText();

            string separator = ",";

            if(new_rows_labels.find(",") == string::npos
                    && new_rows_labels.find(";") != string::npos) {
                separator = ';';
            }

            samples_id = get_tokens(new_rows_labels, separator);
        }
    }

    // Samples

    const tinyxml2::XMLElement* samples_element = data_set_element->FirstChildElement("Samples");

    if(!samples_element)
        throw runtime_error("Samples element is nullptr.\n");

    // Samples number

    const tinyxml2::XMLElement* samples_number_element = samples_element->FirstChildElement("SamplesNumber");

    if(!samples_number_element)
        throw runtime_error("Samples number element is nullptr.\n");

    if(samples_number_element->GetText())
    {
        const Index new_samples_number = Index(atoi(samples_number_element->GetText()));

        samples_uses.resize(new_samples_number);

        set_training();
    }

    // Samples uses

    const tinyxml2::XMLElement* samples_uses_element = samples_element->FirstChildElement("SamplesUses");

    if(!samples_uses_element)
        throw runtime_error("Samples uses element is nullptr.\n");

    if(samples_uses_element->GetText())
        set_samples_uses(get_tokens(samples_uses_element->GetText(), " "));

    // Missing values

    const tinyxml2::XMLElement* missing_values_element = data_set_element->FirstChildElement("MissingValues");

    if(!missing_values_element)
        throw runtime_error("Missing values element is nullptr.\n");

    // Missing values method

    const tinyxml2::XMLElement* missing_values_method_element = missing_values_element->FirstChildElement("MissingValuesMethod");

    if(!missing_values_method_element)
        throw runtime_error("Missing values method element is nullptr.\n");

    if(missing_values_method_element->GetText())
        set_missing_values_method(missing_values_method_element->GetText());

    // Missing values number

    const tinyxml2::XMLElement* missing_values_number_element = missing_values_element->FirstChildElement("MissingValuesNumber");

    if(!missing_values_number_element)
        throw runtime_error("Missing values number element is nullptr.\n");

    if(missing_values_number_element->GetText())
        missing_values_number = Index(atoi(missing_values_number_element->GetText()));

    if(missing_values_number > 0)
    {
        // Raw variables Missing values number

        const tinyxml2::XMLElement* raw_variables_missing_values_number_element = missing_values_element->FirstChildElement("RawVariablesMissingValuesNumber");

        if(!raw_variables_missing_values_number_element)
            throw runtime_error("RawVariablesMissingValuesNumber element is nullptr.\n");

        if(raw_variables_missing_values_number_element->GetText())
        {
            const Tensor<string, 1> new_raw_variables_missing_values_number = get_tokens(raw_variables_missing_values_number_element->GetText(), " ");

            raw_variables_missing_values_number.resize(new_raw_variables_missing_values_number.size());

            for(Index i = 0; i < new_raw_variables_missing_values_number.size(); i++)
            {
                raw_variables_missing_values_number(i) = atoi(new_raw_variables_missing_values_number(i).c_str());
            }
        }

        // Rows missing values number

        const tinyxml2::XMLElement* rows_missing_values_number_element
            = missing_values_element->FirstChildElement("RowsMissingValuesNumber");

        if(!rows_missing_values_number_element)
            throw runtime_error("Rows missing values number element is nullptr.\n");

        if(rows_missing_values_number_element->GetText())
            rows_missing_values_number = Index(atoi(rows_missing_values_number_element->GetText()));
    }

    // Preview data

    const tinyxml2::XMLElement* preview_data_element = data_set_element->FirstChildElement("PreviewData");

    if(!preview_data_element)
        throw runtime_error("Preview data element is nullptr.\n");

    // Preview size

    const tinyxml2::XMLElement* preview_size_element = preview_data_element->FirstChildElement("PreviewSize");

    if(!preview_size_element)
        throw runtime_error("Preview size element is nullptr.\n");

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
            throw runtime_error("Row item number (" + to_string(i+1) + ") "
                                "does not match (" + row_element->Attribute("Item") + ").\n");

        if(row_element->GetText())
            data_file_preview(i) = get_tokens(row_element->GetText(), " ");
    }

    // Display

    const tinyxml2::XMLElement* display_element = data_set_element->FirstChildElement("Display");

    if(display_element)
        set_display(display_element->GetText() != string("0"));
}


void TimeSeriesDataSet::impute_missing_values_mean()
{
    const Tensor<Index, 1> used_samples_indices = get_used_samples_indices();
    const Tensor<Index, 1> used_variables_indices = get_used_variables_indices();
    const Tensor<Index, 1> input_variables_indices = get_input_variables_indices();
    const Tensor<Index, 1> target_variables_indices = get_target_variables_indices();

    const Tensor<type, 1> means = mean(data, used_samples_indices, used_variables_indices);

    const Index used_samples_number = used_samples_indices.size();
    const Index used_variables_number = used_variables_indices.size();
    const Index target_variables_number = target_variables_indices.size();

    //Index current_variable_index;
    //Index current_sample_index;

    if(lags_number == 0 && steps_ahead == 0)
    {
        #pragma omp parallel for 

        for(Index j = 0; j < used_variables_number - target_variables_number; j++)
        {
            const Index current_variable_index = input_variables_indices(j);

            for(Index i = 0; i < used_samples_number; i++)
            {
                const Index current_sample_index = used_samples_indices(i);

                if(isnan(data(current_sample_index, current_variable_index)))
                {
                    data(current_sample_index, current_variable_index) = means(j);
                }
            }
        }

        #pragma omp parallel for 

        for(Index j = 0; j < target_variables_number; j++)
        {
            const Index current_variable_index = target_variables_indices(j);

            for(Index i = 0; i < used_samples_number; i++)
            {
                const Index current_sample_index = used_samples_indices(i);

                if(isnan(data(current_sample_index, current_variable_index)))
                {
                    set_sample_use(i, "None");
                }
            }
        }
    }
    else
    {
        #pragma omp parallel for 

        for(Index j = 0; j < get_variables_number(); j++)
        {
            const Index current_variable_index = j;

            for(Index i = 0; i < used_samples_number; i++)
            {
                const Index current_sample_index = used_samples_indices(i);

                if(isnan(data(current_sample_index, current_variable_index)))
                {
                    if(i < lags_number || i > used_samples_number - steps_ahead)
                    {
                        data(current_sample_index, current_variable_index) = means(j);
                    }
                    else
                    {
                        Index k = i;
                        double previous_value = NAN, next_value = NAN;

                        while(isnan(previous_value) && k > 0)
                        {
                            k--;
                            previous_value = data(used_samples_indices(k), current_variable_index);
                        }

                        k = i;

                        while(isnan(next_value) && k < used_samples_number)
                        {
                            k++;
                            next_value = data(used_samples_indices(k), current_variable_index);
                        }

                        if(isnan(previous_value) && isnan(next_value))
                        {
                            ostringstream buffer;

                            buffer << "OpenNN Exception: DataSet class.\n"
                                   << "void DataSet::impute_missing_values_mean() const.\n"
                                   << "The last " << (used_samples_number - i) + 1 << " samples are all missing, delete them.\n";

                            throw runtime_error(buffer.str());
                        }

                        if(isnan(previous_value))
                        {
                            data(current_sample_index, current_variable_index) = type(next_value);
                        }
                        else if(isnan(next_value))
                        {
                            data(current_sample_index, current_variable_index) = type(previous_value);
                        }
                        else
                        {
                            data(current_sample_index, current_variable_index) = type((previous_value + next_value)/2);
                        }
                    }
                }
            }
        }
    }
}


// @todo Complete method following the structure.

void TimeSeriesDataSet::fill_gaps()
{    
    type start_time = 50;
    type end_time = 100;

    type period = 2;

    type new_time_series_samples_number = (end_time - start_time)/period;
    type new_time_series_variables_number = time_series_data.dimension(1);

    Tensor<type, 2> new_time_series_data(new_time_series_samples_number,  new_time_series_variables_number);

    type timestamp = 0;

    type new_timestamp = 0;

    Index row_index = 0;
    Index column_index = 0;

    Tensor<type, 1> sample;

    for(Index i = 0; i < new_time_series_samples_number; i++)
    {
        new_timestamp = start_time + i*period;
        timestamp = time_series_data(row_index, column_index);

        if(new_timestamp == timestamp)
        {
            sample = time_series_data.chip(row_index, 0);

            new_time_series_data.chip(i, 0) = sample;

            row_index++;
        }
    }

    if(true)
    {
        throw runtime_error("Specify here the error");
    }

}


Tensor<type, 2> TimeSeriesDataSet::calculate_autocorrelations(const Index& lags_number) const
{
    const Index samples_number = time_series_data.dimension(0);

    if(lags_number > samples_number)
    {
        throw runtime_error("Lags number (" + to_string(lags_number) + ") "
                            "is greater than samples number (" + to_string(samples_number) + ") \n");
    }

    const Index raw_variables_number = get_time_series_raw_variables_number();

    const Index input_raw_variables_number = get_input_time_series_raw_variables_number();
    const Index target_raw_variables_number = get_target_time_series_raw_variables_number();

    const Index input_target_raw_variables_number = input_raw_variables_number + target_raw_variables_number;

    const Tensor<Index, 1> input_raw_variables_indices = get_input_time_series_raw_variables_indices();
    const Tensor<Index, 1> target_raw_variables_indices = get_target_time_series_raw_variables_indices();

    Index input_target_numeric_raw_variables_number = 0;

    int counter = 0;

    for(Index i = 0; i < input_target_raw_variables_number; i++)
    {
        if(i < input_raw_variables_number)
        {
            const Index raw_variable_index = input_raw_variables_indices(i);

            const RawVariableType input_raw_variable_type = time_series_raw_variables(raw_variable_index).type;

            if(input_raw_variable_type == RawVariableType::Numeric)
            {
                input_target_numeric_raw_variables_number++;
            }
        }
        else
        {
            const Index raw_variable_index = target_raw_variables_indices(counter);

            const RawVariableType target_raw_variable_type = time_series_raw_variables(raw_variable_index).type;

            if(target_raw_variable_type == RawVariableType::Numeric)
            {
                input_target_numeric_raw_variables_number++;
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

    Tensor<type, 2> autocorrelations(input_target_numeric_raw_variables_number, new_lags_number);
    Tensor<type, 1> autocorrelations_vector(new_lags_number);
    Tensor<type, 2> input_i;
    Index counter_i = 0;

    for(Index i = 0; i < raw_variables_number; i++)
    {
        if(time_series_raw_variables(i).use != VariableUse::None
        && time_series_raw_variables(i).type == RawVariableType::Numeric)
        {
            input_i = get_time_series_raw_variable_data(i);
            cout << "Calculating " << time_series_raw_variables(i).name << " autocorrelations" << endl;
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


Tensor<type, 3> TimeSeriesDataSet::calculate_cross_correlations(const Index& lags_number) const
{
    const Index samples_number = time_series_data.dimension(0);

    if(lags_number > samples_number)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: DataSet class.\n"
               << "Tensor<type, 3> calculate_cross_correlations(const Index& lags_number) const method.\n"
               << "Lags number(" << lags_number << ") is greater than samples number (" << samples_number << ") \n";

        throw runtime_error(buffer.str());
    }

    const Index raw_variables_number = get_time_series_raw_variables_number();

    const Index input_raw_variables_number = get_input_time_series_raw_variables_number();
    const Index target_raw_variables_number = get_target_time_series_raw_variables_number();

    const Index input_target_raw_variables_number = input_raw_variables_number + target_raw_variables_number;

    const Tensor<Index, 1> input_raw_variables_indices = get_input_time_series_raw_variables_indices();
    const Tensor<Index, 1> target_raw_variables_indices = get_target_time_series_raw_variables_indices();

    Index input_target_numeric_raw_variables_number = 0;
    int counter = 0;

    for(Index i = 0; i < input_target_raw_variables_number; i++)
    {
        if(i < input_raw_variables_number)
        {
            const Index raw_variable_index = input_raw_variables_indices(i);

            const RawVariableType input_raw_variable_type = time_series_raw_variables(raw_variable_index).type;

            if(input_raw_variable_type == RawVariableType::Numeric)
            {
                input_target_numeric_raw_variables_number++;
            }
        }
        else
        {
            const Index raw_variable_index = target_raw_variables_indices(counter);

            const RawVariableType target_raw_variable_type = time_series_raw_variables(raw_variable_index).type;

            if(target_raw_variable_type == RawVariableType::Numeric)
            {
                input_target_numeric_raw_variables_number++;
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

    Tensor<type, 3> cross_correlations(input_target_numeric_raw_variables_number,
                                       input_target_numeric_raw_variables_number,
                                       new_lags_number);

    Tensor<type, 1> cross_correlations_vector(new_lags_number);

    Tensor<type, 2> input_i;
    Tensor<type, 2> input_j;

    Index counter_i = 0;
    Index counter_j = 0;

    for(Index i = 0; i < raw_variables_number; i++)
    {
        if(time_series_raw_variables(i).use != VariableUse::None
        && time_series_raw_variables(i).type == RawVariableType::Numeric)
        {
            input_i = get_time_series_raw_variable_data(i);

            if(display) cout << "Calculating " << time_series_raw_variables(i).name << " cross correlations:" << endl;

            counter_j = 0;
        }
        else
        {
            continue;
        }

        for(Index j = 0; j < raw_variables_number; j++)
        {
            if(time_series_raw_variables(j).use != VariableUse::None
            && time_series_raw_variables(j).type == RawVariableType::Numeric)
            {
                input_j = get_time_series_raw_variable_data(j);

                if(display) cout << "   vs. " << time_series_raw_variables(j).name << endl;
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
                cross_correlations(counter_i, counter_j, k) = cross_correlations_vector(k) ;
            }

            counter_j++;
        }

        counter_i++;
    }

    return cross_correlations;
}


void TimeSeriesDataSet::load_time_series_data_binary(const string& time_series_data_file_name)
{
    ifstream file;

    file.open(time_series_data_file_name.c_str(), ios::binary);

    if(!file.is_open())
        throw runtime_error("Cannot open binary file: " + time_series_data_file_name + "\n");

    streamsize size = sizeof(Index);

    Index raw_variables_number = 0;
    Index rows_number = 0;

    file.read(reinterpret_cast<char*>(&raw_variables_number), size);
    file.read(reinterpret_cast<char*>(&rows_number), size);

    size = sizeof(type);

    type value = type(0);

    time_series_data.resize(rows_number, raw_variables_number);

    for(Index i = 0; i < rows_number*raw_variables_number; i++)
    {
        file.read(reinterpret_cast<char*>(&value), size);

        time_series_data(i) = value;
    }

    file.close();
}


void TimeSeriesDataSet::save_time_series_data_binary(const string& binary_data_file_name) const
{
    ofstream file(binary_data_file_name.c_str(), ios::binary);

    if(!file.is_open())
        throw runtime_error("Cannot open data binary file.");

    // Write data

    streamsize size = sizeof(Index);

    Index raw_variables_number = time_series_data.dimension(1);
    Index rows_number = time_series_data.dimension(0);

    cout << "Saving binary data file..." << endl;

    file.write(reinterpret_cast<char*>(&raw_variables_number), size);
    file.write(reinterpret_cast<char*>(&rows_number), size);

    size = sizeof(type);

    type value;

    for(int i = 0; i < raw_variables_number; i++)
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
