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

namespace opennn
{

TimeSeriesDataSet::TimeSeriesDataSet(const Index& new_samples_number,
                                     const dimensions& new_input_dimensions,
                                     const dimensions& new_target_dimensions)
    :DataSet(new_samples_number, new_input_dimensions, new_target_dimensions)
{
}


TimeSeriesDataSet::TimeSeriesDataSet(const filesystem::path& data_path,
                                     const string& separator,
                                     const bool& has_header,
                                     const bool& has_sample_ids,
                                     const Codification& data_codification)
    :DataSet(data_path, separator, has_header, has_sample_ids, data_codification)
{
}


const Index& TimeSeriesDataSet::get_time_raw_variable_index() const
{
    return time_raw_variable_index;
}


const Index& TimeSeriesDataSet::get_group_raw_variable_index() const
{
    return group_raw_variable_index;
}


const Index& TimeSeriesDataSet::get_lags_number() const
{
    return lags_number;
}


const Index& TimeSeriesDataSet::get_steps_ahead() const
{
    return steps_ahead;
}


void TimeSeriesDataSet::set_lags_number(const Index& new_lags_number)
{
    lags_number = new_lags_number;
}


void TimeSeriesDataSet::set_steps_ahead_number(const Index& new_steps_ahead_number)
{
    steps_ahead = new_steps_ahead_number;
}


void TimeSeriesDataSet::set_time_raw_variable(const string& new_time_column)
{
//    time_column = new_time_column;
}


void TimeSeriesDataSet::set_group_by_raw_variable(const string& new_group_by_column)
{
//    group_by_column = new_group_by_column;
}


void TimeSeriesDataSet::print() const
{
    if(!display) return;

    const Index variables_number = get_variables_number();
    const Index input_variables_number = get_variables_number(VariableUse::Input);
    const Index samples_number = get_samples_number();
    const Index target_variables_bumber = get_variables_number(VariableUse::Target);

    cout << "Time series data set object summary:\n"
         << "Number of samples: " << samples_number << "\n"
         << "Number of variables: " << variables_number << "\n"
         << "Number of input variables: " << input_variables_number << "\n"
         << "Number of targets: " << target_variables_bumber << "\n"
         << "Input variables dimensions: ";

    print_vector(get_dimensions(DataSet::VariableUse::Input));

    cout << "Target variables dimensions: ";

    print_vector(get_dimensions(DataSet::VariableUse::Target));
}


void TimeSeriesDataSet::to_XML(XMLPrinter& file_stream) const
{
    /*
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
    file_stream.PushText(to_string(has_sample_ids).c_str());
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
        raw_variables[i].to_XML(file_stream);
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
            //time_series_raw_variables[i].to_XML(file_stream);
            file_stream.CloseElement();
        }

        // Close time series raw variables

        file_stream.CloseElement();
    }

    // Ids

    if(has_sample_ids)
    {
        file_stream.OpenElement("HasSamplesId");
        file_stream.PushText(string_tensor_to_string(get_sample_ids()).c_str());
        file_stream.CloseElement();
    }

    // Samples

    file_stream.OpenElement("Samples");

    // Samples number

    file_stream.OpenElement("SamplesNumber");
    file_stream.PushText(to_string(get_samples_number()).c_str());
    file_stream.CloseElement();

    // Samples uses

    file_stream.OpenElement("SampleUses");
    file_stream.PushText(tensor_to_string(get_sample_uses_vector()).c_str());
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

            for(Index j = 0; j < data_file_preview[i].size(); j++)
            {
                file_stream.PushText(data_file_preview[i][j].c_str());

                if(j != data_file_preview[i].size()-1)
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
*/
}


void TimeSeriesDataSet::from_XML(const XMLDocument& data_set_document)
{
/*
    // Data set element

    const XMLElement* data_set_element = data_set_document.FirstChildElement("DataSet");

    if(!data_set_element)
        throw runtime_error("Data set element is nullptr.\n");

    // Data file

    const XMLElement* data_source_element = data_set_element->FirstChildElement("DataSource");

    if(!data_source_element)
        throw runtime_error("Data file element is nullptr.\n");

    // Data file name

    const XMLElement* data_source_path_element = data_source_element->FirstChildElement("Path");

    if(!data_source_path_element)
        throw runtime_error("Path element is nullptr.\n");

    if(data_source_path_element->GetText())
        set_data_path(data_source_path_element->GetText());

    // Separator

    const XMLElement* separator_element = data_source_element->FirstChildElement("Separator");

    if(separator_element)
        if(separator_element->GetText())
            set_separator_name(separator_element->GetText());

    // Has raw_variables names

    const XMLElement* raw_variables_names_element = data_source_element->FirstChildElement("HasHeader");

    if(raw_variables_names_element)
        set_has_header(raw_variables_names_element->GetText() == string("1"));

    // Samples id

    const XMLElement* rows_label_element = data_source_element->FirstChildElement("HasSamplesId");

    if(rows_label_element)
        set_has_ids(rows_label_element->GetText() == string("1"));

    // Missing values label

    const XMLElement* missing_values_label_element = data_source_element->FirstChildElement("MissingValuesLabel");

    if(missing_values_label_element)
        if(missing_values_label_element->GetText())
            set_missing_values_label(missing_values_label_element->GetText());

    // Forecasting

    // Lags number

    const XMLElement* lags_number_element = data_source_element->FirstChildElement("LagsNumber");

    if(!lags_number_element)
        throw runtime_error("Lags number element is nullptr.\n");

    if(lags_number_element->GetText())
        set_lags_number(Index(atoi(lags_number_element->GetText())));

    // Steps ahead

    const XMLElement* steps_ahead_element = data_source_element->FirstChildElement("StepsAhead");

    if(!steps_ahead_element)
        throw runtime_error("Steps ahead element is nullptr.\n");

    if(steps_ahead_element->GetText())
        set_steps_ahead_number(Index(atoi(steps_ahead_element->GetText())));

    // Time raw_variable

    const XMLElement* time_raw_variable_element = data_source_element->FirstChildElement("TimeRawVariable");

    if(!time_raw_variable_element)
        throw runtime_error("Time raw_variable element is nullptr.\n");

    if(time_raw_variable_element->GetText())
        set_time_raw_variable(time_raw_variable_element->GetText());

    // Group by raw_variable

    const XMLElement* group_by_raw_variable_element = data_source_element->FirstChildElement("GroupByRawVariable");

    if(!group_by_raw_variable_element)
        throw runtime_error("Group by raw_variable element is nullptr.\n");

    if(group_by_raw_variable_element->GetText())
        set_group_by_raw_variable(group_by_raw_variable_element->GetText());

    // Codification

    const XMLElement* codification_element = data_source_element->FirstChildElement("Codification");

    if(codification_element)
        if(codification_element->GetText())
            set_codification(codification_element->GetText());

    // Raw variables

    const XMLElement* raw_variables_element = data_set_element->FirstChildElement("RawVariables");

    if(!raw_variables_element)
        throw runtime_error("RawVariables element is nullptr.\n");

    // Raw variables number

    const XMLElement* raw_variables_number_element = raw_variables_element->FirstChildElement("RawVariablesNumber");

    if(!raw_variables_number_element)
        throw runtime_error("RawVariablesNumber element is nullptr.\n");

    if(raw_variables_number_element->GetText())
        set_raw_variables_number(Index(atoi(raw_variables_number_element->GetText())));

    // Raw variables

    const XMLElement* start_element = raw_variables_number_element;

    for(Index i = 0; i < raw_variables.size(); i++)
    {
        const XMLElement* raw_variable_element = start_element->NextSiblingElement("RawVariable");
        start_element = raw_variable_element;

        if(raw_variable_element->Attribute("Item") != to_string(i+1))
            throw runtime_error("Raw variable item number (" + to_string(i+1) + ") does not match (" + raw_variable_element->Attribute("Item") + ").\n");

        // Name

        const XMLElement* name_element = raw_variable_element->FirstChildElement("Name");

        if(!name_element)
            throw runtime_error("Name element is nullptr.\n");

        if(name_element->GetText())
        {
            const string new_name = name_element->GetText();

            raw_variables[i].name = new_name;
        }

        // Scaler

        const XMLElement* scaler_element = raw_variable_element->FirstChildElement("Scaler");

        if(!scaler_element)
            throw runtime_error("Scaler element is nullptr.\n");

        if(scaler_element->GetText())
        {
            const string new_scaler = scaler_element->GetText();

            raw_variables[i].set_scaler(new_scaler);
        }

        // raw_variable use

        const XMLElement* use_element = raw_variable_element->FirstChildElement("Use");

        if(!use_element)
            throw runtime_error("Raw variable use element is nullptr.\n");

        if(use_element->GetText())
        {
            const string new_raw_variable_use = use_element->GetText();

            raw_variables[i].set_use(new_raw_variable_use);
        }

        // Type

        const XMLElement* type_element = raw_variable_element->FirstChildElement("Type");

        if(!type_element)
            throw runtime_error("Type element is nullptr.\n");

        if(type_element->GetText())
        {
            const string new_type = type_element->GetText();

            raw_variables[i].set_type(new_type);
        }

        if(raw_variables[i].type == RawVariableType::Categorical || raw_variables[i].type == RawVariableType::Binary)
        {
            // Categories

            const XMLElement* categories_element = raw_variable_element->FirstChildElement("Categories");

            if(!categories_element)
                throw runtime_error("Categories element is nullptr.\n");

            if(categories_element->GetText())
            {
                const string new_categories = categories_element->GetText();

                raw_variables[i].categories = get_tokens(new_categories, " ");
            }
        }
    }

    // Time series raw_variables

    const XMLElement* time_series_raw_variables_element = data_set_element->FirstChildElement("TimeSeriesRawVariables");

    // Time series raw_variables number

    const XMLElement* time_series_raw_variables_number_element = time_series_raw_variables_element->FirstChildElement("TimeSeriesRawVariablesNumber");

    if(!time_series_raw_variables_number_element)
        throw runtime_error("Time seires raw_variables number element is nullptr.\n");

    Index time_series_new_raw_variables_number = 0;

    if(time_series_raw_variables_number_element->GetText())
    {
        time_series_new_raw_variables_number = Index(atoi(time_series_raw_variables_number_element->GetText()));

        set_time_series_raw_variables_number(time_series_new_raw_variables_number);
    }

    // Time series raw_variables

    const XMLElement* time_series_start_element = time_series_raw_variables_number_element;

    if(time_series_new_raw_variables_number > 0)
    {
        for(Index i = 0; i < time_series_new_raw_variables_number; i++)
        {
            const XMLElement* time_series_raw_variable_element = time_series_start_element->NextSiblingElement("TimeSeriesRawVariable");
            time_series_start_element = time_series_raw_variable_element;

            if(time_series_raw_variable_element->Attribute("Item") != to_string(i+1))
                throw runtime_error("Time series raw_variable item number (" + to_string(i+1) + ") "
                                    "does not match (" + time_series_raw_variable_element->Attribute("Item") + ").\n");

            // Name

            const XMLElement* time_series_name_element = time_series_raw_variable_element->FirstChildElement("Name");

            if(!time_series_name_element)
                throw runtime_error("Time series name element is nullptr.\n");

            if(time_series_name_element->GetText())
            {
                const string time_series_new_name = time_series_name_element->GetText();

                time_series_raw_variables[i].name = time_series_new_name;
            }

            // Scaler

            const XMLElement* time_series_scaler_element = time_series_raw_variable_element->FirstChildElement("Scaler");

            if(!time_series_scaler_element)
                throw runtime_error("Time series scaler element is nullptr.\n");

            if(time_series_scaler_element->GetText())
            {
                const string time_series_new_scaler = time_series_scaler_element->GetText();

                time_series_raw_variables[i].set_scaler(time_series_new_scaler);
            }

            // raw_variable use

            const XMLElement* time_series_raw_variable_use_element = time_series_raw_variable_element->FirstChildElement("Use");

            if(!time_series_raw_variable_use_element)
                throw runtime_error("Time series raw_variable use element is nullptr.\n");

            if(time_series_raw_variable_use_element->GetText())
            {
                const string time_series_new_raw_variable_use = time_series_raw_variable_use_element->GetText();

                time_series_raw_variables[i].set_use(time_series_new_raw_variable_use);
            }

            // Type

            const XMLElement* time_series_type_element = time_series_raw_variable_element->FirstChildElement("Type");

            if(!time_series_type_element)
                throw runtime_error("Time series type element is nullptr.\n");

            if(time_series_type_element->GetText())
            {
                const string time_series_new_type = time_series_type_element->GetText();
                time_series_raw_variables[i].set_type(time_series_new_type);
            }

            if(time_series_raw_variables[i].type == RawVariableType::Categorical || time_series_raw_variables[i].type == RawVariableType::Binary)
            {
                // Categories

                const XMLElement* time_series_categories_element = time_series_raw_variable_element->FirstChildElement("Categories");

                if(!time_series_categories_element)
                    throw runtime_error("Time series categories element is nullptr.\n");

                if(time_series_categories_element->GetText())
                {
                    const string time_series_new_categories = time_series_categories_element->GetText();

                    time_series_raw_variables[i].categories = get_tokens(time_series_new_categories, ";");
                }
            }
        }
    }

    // Rows label

    if(has_sample_ids)
    {
        // Samples id begin tag

        const XMLElement* has_ids_element = data_set_element->FirstChildElement("HasSamplesId");

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

            sample_ids = get_tokens(new_rows_labels, separator);
        }
    }

    // Samples

    const XMLElement* samples_element = data_set_element->FirstChildElement("Samples");

    if(!samples_element)
        throw runtime_error("Samples element is nullptr.\n");

    // Samples number

    const XMLElement* samples_number_element = samples_element->FirstChildElement("SamplesNumber");

    if(!samples_number_element)
        throw runtime_error("Samples number element is nullptr.\n");

    if(samples_number_element->GetText())
    {
        const Index new_samples_number = Index(atoi(samples_number_element->GetText()));

        sample_uses.resize(new_samples_number);

        set(DataSet::SampleUse::Training);
    }

    // Samples uses

    const XMLElement* samples_uses_element = samples_element->FirstChildElement("SampleUses");

    if(!samples_uses_element)
        throw runtime_error("Samples uses element is nullptr.\n");

    if(samples_uses_element->GetText())
        set_sample_uses(get_tokens(samples_uses_element->GetText(), " "));

    // Missing values

    const XMLElement* missing_values_element = data_set_element->FirstChildElement("MissingValues");

    if(!missing_values_element)
        throw runtime_error("Missing values element is nullptr.\n");

    // Missing values method

    const XMLElement* missing_values_method_element = missing_values_element->FirstChildElement("MissingValuesMethod");

    if(!missing_values_method_element)
        throw runtime_error("Missing values method element is nullptr.\n");

    if(missing_values_method_element->GetText())
        set_missing_values_method(missing_values_method_element->GetText());

    // Missing values number

    const XMLElement* missing_values_number_element = missing_values_element->FirstChildElement("MissingValuesNumber");

    if(!missing_values_number_element)
        throw runtime_error("Missing values number element is nullptr.\n");

    if(missing_values_number_element->GetText())
        missing_values_number = Index(atoi(missing_values_number_element->GetText()));

    if(missing_values_number > 0)
    {
        // Raw variables Missing values number

        const XMLElement* raw_variables_missing_values_number_element = missing_values_element->FirstChildElement("RawVariablesMissingValuesNumber");

        if(!raw_variables_missing_values_number_element)
            throw runtime_error("RawVariablesMissingValuesNumber element is nullptr.\n");

        if(raw_variables_missing_values_number_element->GetText())
        {
            const vector<string> new_raw_variables_missing_values_number = get_tokens(raw_variables_missing_values_number_element->GetText(), " ");

            raw_variables_missing_values_number.resize(new_raw_variables_missing_values_number.size());

            for(Index i = 0; i < new_raw_variables_missing_values_number.size(); i++)
            {
                raw_variables_missing_values_number(i) = atoi(new_raw_variables_missing_values_number[i].c_str());
            }
        }

        // Rows missing values number

        const XMLElement* rows_missing_values_number_element
            = missing_values_element->FirstChildElement("RowsMissingValuesNumber");

        if(!rows_missing_values_number_element)
            throw runtime_error("Rows missing values number element is nullptr.\n");

        if(rows_missing_values_number_element->GetText())
            rows_missing_values_number = Index(atoi(rows_missing_values_number_element->GetText()));
    }

    // Preview data

    const XMLElement* preview_data_element = data_set_element->FirstChildElement("PreviewData");

    if(!preview_data_element)
        throw runtime_error("Preview data element is nullptr.\n");

    // Preview size

    const XMLElement* preview_size_element = preview_data_element->FirstChildElement("PreviewSize");

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
        const XMLElement* row_element = start_element->NextSiblingElement("Row");
        start_element = row_element;

        if(row_element->Attribute("Item") != to_string(i+1))
            throw runtime_error("Row item number (" + to_string(i+1) + ") "
                                "does not match (" + row_element->Attribute("Item") + ").\n");

        if(row_element->GetText())
            data_file_preview[i] = get_tokens(row_element->GetText(), " ");
    }

    // Display

    set_display(read_xml_bool(neural_network_element, "Display"));
*/
}


void TimeSeriesDataSet::impute_missing_values_mean()
{
    const vector<Index> used_sample_indices = get_used_sample_indices();
    const vector<Index> used_variable_indices = get_used_variable_indices();
    const vector<Index> input_variable_indices = get_variable_indices(DataSet::VariableUse::Input);
    const vector<Index> target_variable_indices = get_variable_indices(DataSet::VariableUse::Target);

    const Tensor<type, 1> means = mean(data, used_sample_indices, used_variable_indices);

    const Index used_samples_number = used_sample_indices.size();
    const Index used_variables_number = used_variable_indices.size();
    const Index target_variables_number = target_variable_indices.size();

    //Index current_variable_index;
    //Index current_sample_index;

    if(lags_number == 0 && steps_ahead == 0)
    {
        #pragma omp parallel for 

        for(Index j = 0; j < used_variables_number - target_variables_number; j++)
        {
            const Index current_variable_index = input_variable_indices[j];

            for(Index i = 0; i < used_samples_number; i++)
            {
                const Index current_sample_index = used_sample_indices[i];

                if(isnan(data(current_sample_index, current_variable_index)))
                    data(current_sample_index, current_variable_index) = means(j);
            }
        }

        #pragma omp parallel for 

        for(Index j = 0; j < target_variables_number; j++)
        {
            const Index current_variable_index = target_variable_indices[j];

            for(Index i = 0; i < used_samples_number; i++)
            {
                const Index current_sample_index = used_sample_indices[i];

                if(isnan(data(current_sample_index, current_variable_index)))
                    set_sample_use(i, "None");
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
                const Index current_sample_index = used_sample_indices[i];

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
                            previous_value = data(used_sample_indices[k], current_variable_index);
                        }

                        k = i;

                        while(isnan(next_value) && k < used_samples_number)
                        {
                            k++;
                            next_value = data(used_sample_indices[k], current_variable_index);
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
                            data(current_sample_index, current_variable_index) = type(next_value);
                        else if(isnan(next_value))
                            data(current_sample_index, current_variable_index) = type(previous_value);
                        else
                            data(current_sample_index, current_variable_index) = type((previous_value + next_value)/2);
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

    type new_samples_number = (end_time - start_time)/period;
    type new_variables_number = get_variables_number();

    Tensor<type, 2> new_data(new_samples_number,  new_variables_number);

    type timestamp = 0;

    type new_timestamp = 0;

    Index row_index = 0;
    Index column_index = 0;

    Tensor<type, 1> sample;

    for(Index i = 0; i < new_samples_number; i++)
    {
        new_timestamp = start_time + i*period;
        timestamp = new_data(row_index, column_index);

        if(new_timestamp == timestamp)
        {
            data.chip(i, 0) = data.chip(row_index, 0);

            row_index++;
        }
    }
}


Tensor<type, 2> TimeSeriesDataSet::calculate_autocorrelations(const Index& lags_number) const
{
    const Index samples_number = get_samples_number();

    if(lags_number > samples_number)
    {
        throw runtime_error("Lags number (" + to_string(lags_number) + ") "
                            "is greater than samples number (" + to_string(samples_number) + ") \n");
    }

    const Index raw_variables_number = get_raw_variables_number();

    const Index input_raw_variables_number = get_raw_variables_number(VariableUse::Input);
    const Index target_raw_variables_number = get_raw_variables_number(VariableUse::Target);

    const Index input_target_raw_variables_number = input_raw_variables_number + target_raw_variables_number;

    const vector<Index> input_raw_variable_indices = get_raw_variable_indices(VariableUse::Input);
    const vector<Index> target_raw_variable_indices = get_raw_variable_indices(VariableUse::Target);

    Index input_target_numeric_raw_variables_number = 0;

    int count = 0;

    for(Index i = 0; i < input_target_raw_variables_number; i++)
    {
        if(i < input_raw_variables_number)
        {
            const Index raw_variable_index = input_raw_variable_indices[i];

            const RawVariableType input_raw_variable_type = raw_variables[raw_variable_index].type;

            if(input_raw_variable_type == RawVariableType::Numeric)
                input_target_numeric_raw_variables_number++;
        }
        else
        {
            const Index raw_variable_index = target_raw_variable_indices[count];

            const RawVariableType target_raw_variable_type = raw_variables[raw_variable_index].type;

            if(target_raw_variable_type == RawVariableType::Numeric)
                input_target_numeric_raw_variables_number++;

            count++;
        }
    }

    Index new_lags_number;

    if((samples_number == lags_number || samples_number < lags_number) && lags_number > 2)
        new_lags_number = lags_number - 2;
    else if(samples_number == lags_number + 1 && lags_number > 1)
        new_lags_number = lags_number - 1;
    else
        new_lags_number = lags_number;

    Tensor<type, 2> autocorrelations(input_target_numeric_raw_variables_number, new_lags_number);
    Tensor<type, 1> autocorrelations_vector(new_lags_number);
    Tensor<type, 2> input_i;
    Index counter_i = 0;

    for(Index i = 0; i < raw_variables_number; i++)
    {
        if(raw_variables[i].use != VariableUse::None
        && raw_variables[i].type == RawVariableType::Numeric)
        {
            input_i = get_raw_variable_data(i);
            cout << "Calculating " << raw_variables[i].name << " autocorrelations" << endl;
        }
        else
        {
            continue;
        }
        
        const TensorMap<Tensor<type, 1>> current_input_i(input_i.data(), input_i.dimension(0));
        
        autocorrelations_vector = opennn::autocorrelations(thread_pool_device.get(), current_input_i, new_lags_number);
        for(Index j = 0; j < new_lags_number; j++)
            autocorrelations (counter_i, j) = autocorrelations_vector(j) ;

        counter_i++;
 
    }

    return autocorrelations;
}


Tensor<type, 3> TimeSeriesDataSet::calculate_cross_correlations(const Index& lags_number) const
{
    const Index samples_number = get_samples_number();

    if(lags_number > samples_number)
        throw runtime_error("Lags number(" + to_string(lags_number) + ") is greater than samples number (" + to_string(samples_number) + ") \n");

    const Index raw_variables_number = get_raw_variables_number();

    const Index input_raw_variables_number = get_raw_variables_number(VariableUse::Input);
    const Index target_raw_variables_number = get_raw_variables_number(VariableUse::Target);

    const Index input_target_raw_variables_number = input_raw_variables_number + target_raw_variables_number;

    const vector<Index> input_raw_variable_indices = get_raw_variable_indices(VariableUse::Input);
    const vector<Index> target_raw_variable_indices = get_raw_variable_indices(VariableUse::Input);

    Index input_target_numeric_raw_variables_number = 0;
    int count = 0;

    for(Index i = 0; i < input_target_raw_variables_number; i++)
    {
        if(i < input_raw_variables_number)
        {
            const Index raw_variable_index = input_raw_variable_indices[i];

            const RawVariableType input_raw_variable_type = raw_variables[raw_variable_index].type;

            if(input_raw_variable_type == RawVariableType::Numeric)
                input_target_numeric_raw_variables_number++;
        }
        else
        {
            const Index raw_variable_index = target_raw_variable_indices[count];

            const RawVariableType target_raw_variable_type = raw_variables[raw_variable_index].type;

            if(target_raw_variable_type == RawVariableType::Numeric)
                input_target_numeric_raw_variables_number++;

            count++;
        }
    }

    Index new_lags_number;

    if(samples_number == lags_number)
        new_lags_number = lags_number - 2;
    else if(samples_number == lags_number + 1)
        new_lags_number = lags_number - 1;
    else
        new_lags_number = lags_number;

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
        if(raw_variables[i].use == VariableUse::None
        || raw_variables[i].type != RawVariableType::Numeric)
            continue;

        input_i = get_raw_variable_data(i);

        if (display) cout << "Calculating " << raw_variables[i].name << " cross correlations:" << endl;

        counter_j = 0;

        for(Index j = 0; j < raw_variables_number; j++)
        {
            if(raw_variables[j].use == VariableUse::None
            || raw_variables[j].type == RawVariableType::Numeric)
                continue;

            input_j = get_raw_variable_data(j);

            if(display) cout << "   vs. " << raw_variables[j].name << endl;
 
            const TensorMap<Tensor<type, 1>> current_input_i(input_i.data(), input_i.dimension(0));
            const TensorMap<Tensor<type, 1>> current_input_j(input_j.data(), input_j.dimension(0));

            cross_correlations_vector = opennn::cross_correlations(thread_pool_device.get(), 
                current_input_i, current_input_j, new_lags_number);

            for(Index k = 0; k < new_lags_number; k++)
                cross_correlations(counter_i, counter_j, k) = cross_correlations_vector(k) ;

            counter_j++;
        }

        counter_i++;
    }

    return cross_correlations;
}

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2025 Artificial Intelligence Techniques, SL.
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
