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
#include "strings_utilities.h"

namespace opennn
{

TimeSeriesDataset::TimeSeriesDataset(const Index& new_samples_number,
                                     const dimensions& new_input_dimensions,
                                     const dimensions& new_target_dimensions)
    :Dataset(new_samples_number, new_input_dimensions, new_target_dimensions)
{
}


TimeSeriesDataset::TimeSeriesDataset(const filesystem::path& data_path,
                                     const string& separator,
                                     const bool& has_header,
                                     const bool& has_sample_ids,
                                     const Codification& data_codification)
    :Dataset(data_path, separator, has_header, has_sample_ids, data_codification)
{
}


const Index& TimeSeriesDataset::get_time_raw_variable_index() const
{
    return time_raw_variable_index;
}


const Index& TimeSeriesDataset::get_lags_number() const
{
    return lags_number;
}


const Index& TimeSeriesDataset::get_steps_ahead() const
{
    return steps_ahead;
}


void TimeSeriesDataset::set_lags_number(const Index& new_lags_number)
{
    lags_number = new_lags_number;
}


void TimeSeriesDataset::set_steps_ahead_number(const Index& new_steps_ahead_number)
{
    steps_ahead = new_steps_ahead_number;
}


void TimeSeriesDataset::set_time_raw_variable_index(const Index& new_time_raw_variable_index)
{
    time_raw_variable_index = new_time_raw_variable_index;
}


void TimeSeriesDataset::print() const
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
         << "Number of target variables: " << target_variables_bumber << "\n"
         << "Input variables dimensions: ";

    print_vector(get_dimensions(Dataset::VariableUse::Input));

    cout << "Target variables dimensions: ";

    print_vector(get_dimensions(Dataset::VariableUse::Target));



}


void TimeSeriesDataset::to_XML(XMLPrinter& printer) const
{
    // if(model_type != ModelType::Forecasting)
    //     throw runtime_error("No forecasting model type");

    printer.OpenElement("Dataset");

    printer.OpenElement("DataSource");
    add_xml_element(printer, "FileType", "csv");
    add_xml_element(printer, "Path", data_path.string());
    add_xml_element(printer, "Separator", get_separator_name());
    add_xml_element(printer, "HasHeader", to_string(has_header));
    add_xml_element(printer, "HasSamplesId", to_string(has_sample_ids));
    add_xml_element(printer, "MissingValuesLabel", missing_values_label);
    add_xml_element(printer, "LagsNumber", to_string(get_lags_number()));
    add_xml_element(printer, "StepsAhead", to_string(get_steps_ahead()));
//    add_xml_element(printer, "TimeRawVariable", get_time_raw_variable());
    add_xml_element(printer, "GroupByRawVariable", "");
    add_xml_element(printer, "Codification", get_codification_string());
    printer.CloseElement();

    // Group by raw_variable
    // {
    //     file_stream.OpenElement("GroupByRawVariable");

    //     buffer.str("");
    //     buffer << get_time_raw_variable();

    //     file_stream.PushText(buffer.str().c_str());

    //     file_stream.CloseElement();
    // }

    // Raw variables

    printer.OpenElement("RawVariables");
    add_xml_element(printer, "RawVariablesNumber", to_string(get_raw_variables_number()));

    const Index raw_variables_number = get_raw_variables_number();

    cout << raw_variables_number << endl;

    for(Index i = 0; i < raw_variables_number; i++)
    {
        printer.OpenElement("RawVariable");
        printer.PushAttribute("Item", to_string(i+1).c_str());
        raw_variables[i].to_XML(printer);
        printer.CloseElement();
    }

    printer.CloseElement();

    // Time series raw_variables
    //const Index time_series_raw_variables_number = get_time_series_raw_variables_number();
    // const Index time_series_raw_variables_number = get_time_raw_variable_index();

    // if(time_series_raw_variables_number > 0) - descomentar
    // {
    //     printer.OpenElement("TimeSeriesRawVariables");
    //     add_xml_element(printer, "TimeSeriesRawVariablesNumber", to_string(get_time_raw_variable_index()));


    //     for(Index i = 0; i < time_series_raw_variables_number; i++)
    //     {
    //         printer.OpenElement("TimeSeriesRawVariable");
    //         printer.PushAttribute("Item", to_string(i+1).c_str());
    //         raw_variables[i].to_XML(printer);
    //         printer.CloseElement();
    //     }
    //     printer.CloseElement();
    // }

    // Samples

    printer.OpenElement("Samples");

    add_xml_element(printer, "SamplesNumber", to_string(get_samples_number()));

    if(has_sample_ids)
        add_xml_element(printer, "SamplesId", vector_to_string(sample_ids));

    add_xml_element(printer, "SampleUses", vector_to_string(get_sample_uses_vector()));

    printer.CloseElement();

    // Missing values

    printer.OpenElement("MissingValues");
    add_xml_element(printer, "MissingValuesNumber", to_string(missing_values_number));

    if(missing_values_number > 0)
    {
        add_xml_element(printer, "MissingValuesMethod", get_missing_values_method_string());
        add_xml_element(printer, "RawVariablesMissingValuesNumber", tensor_to_string(raw_variables_missing_values_number));
        add_xml_element(printer, "RowsMissingValuesNumber", to_string(rows_missing_values_number));
    }

    printer.CloseElement();

    // Preview data

    printer.OpenElement("PreviewData");

    add_xml_element(printer, "PreviewSize", to_string(data_file_preview.size()));

    vector<string> vector_data_file_preview = convert_string_vector(data_file_preview,",");

    for(int i = 0; i < data_file_preview.size(); i++)
    {
        printer.OpenElement("Row");
        printer.PushAttribute("Item", to_string(i + 1).c_str());
        printer.PushText(vector_data_file_preview[i].data());
        printer.CloseElement();
    }

    printer.CloseElement();

    // Display

    add_xml_element(printer, "Display", to_string(display));

    // Close data set

    printer.CloseElement();

}


void TimeSeriesDataset::from_XML(const XMLDocument& data_set_document)
{
    const XMLElement* data_set_element = data_set_document.FirstChildElement("Dataset");
    if(!data_set_element)
        throw runtime_error("Data set element is nullptr.\n");

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
    set_lags_number(stoi(read_xml_string(data_source_element, "LagsNumber")));
    set_steps_ahead_number(stoi(read_xml_string(data_source_element, "StepsAhead")));
//    set_time_raw_variable(read_xml_string(data_source_element, "TimeRawVariable"));
    //set_group_by_raw_variable(read_xml_string(data_source_element, "GroupByRawVariable"));
    set_codification(read_xml_string(data_source_element, "Codification"));


    // Raw variables

    const XMLElement* raw_variables_element = data_set_element->FirstChildElement("RawVariables");

    if(!raw_variables_element)
        throw runtime_error("RawVariables element is nullptr.\n");

    set_raw_variables_number(read_xml_index(raw_variables_element, "RawVariablesNumber"));

    const XMLElement* start_element = raw_variables_element->FirstChildElement("RawVariablesNumber");

    for(size_t i = 0; i < raw_variables.size(); i++)
    {
        RawVariable& raw_variable = raw_variables[i];
        const XMLElement* raw_variable_element = start_element->NextSiblingElement("RawVariable");
        start_element = raw_variable_element;

        if(raw_variable_element->Attribute("Item") != to_string(i+1))
            throw runtime_error("Raw variable item number (" + to_string(i+1) + ") does not match (" + raw_variable_element->Attribute("Item") + ").\n");

        raw_variable.name = read_xml_string(raw_variable_element, "Name");
        raw_variable.set_scaler(read_xml_string(raw_variable_element, "Scaler"));
        raw_variable.set_use(read_xml_string(raw_variable_element, "Use"));
        raw_variable.set_type(read_xml_string(raw_variable_element, "Type"));

        if (raw_variable.type == RawVariableType::Categorical || raw_variable.type == RawVariableType::Binary)
        {
            const XMLElement* categories_element = raw_variable_element->FirstChildElement("Categories");

            if (categories_element)
                raw_variable.categories = get_tokens(read_xml_string(raw_variable_element, "Categories"), ";");
            else if (raw_variable.type == RawVariableType::Binary)
                raw_variable.categories = { "0", "1" };
            else
                throw runtime_error("Categorical RawVariable Element is nullptr: Categories");
        }
    }

    // Time series raw_variables

    // const XMLElement* time_series_raw_variables_element = data_set_element->FirstChildElement("TimeSeriesRawVariables");

    // if(!time_series_raw_variables_element)
    //     throw runtime_error("time series raw variable element is nullptr");

    // set_time_raw_variable_index(stoi(read_xml_string(time_series_raw_variables_element, "TimeSeriesRawVariablesNumber")));

    // const XMLElement* time_series_start_element = time_series_raw_variables_element->FirstChildElement("TimeSeriesRawVariablesNumber");

    // if(time_raw_variable_index > 0)
    // {
    //     for(Index i = 0; i < time_raw_variable_index; i++)
    //     {
    //         RawVariable& time_raw_variable = time_series_raw_variables[i];
    //         const XMLElement* time_series_raw_variable_element = time_series_start_element->NextSiblingElement("TimeSeriesRawVariable");
    //         time_series_start_element = time_series_raw_variable_element;

    //         if(time_series_raw_variable_element->Attribute("Item") != to_string(i+1))
    //             throw runtime_error("Time series raw_variable item number (" + to_string(i+1) + ") "
    //                                 "does not match (" + time_series_raw_variable_element->Attribute("Item") + ").\n");

    //         time_raw_variable.name = read_xml_string(time_series_raw_variable_element, "Name");
    //         time_raw_variable.set_scaler(read_xml_string(time_series_raw_variable_element, "Scaler"));
    //         time_raw_variable.set_use(read_xml_string(time_series_raw_variable_element, "Use"));
    //         time_raw_variable.set_type(read_xml_string(time_series_raw_variable_element, "Type"));

    //         if (time_raw_variable.type == RawVariableType::Categorical || time_raw_variable.type == RawVariableType::Binary)
    //         {
    //             const XMLElement* categories_element = time_series_raw_variable_element->FirstChildElement("Categories");

    //             if (categories_element)
    //                 time_raw_variable.categories = get_tokens(read_xml_string(time_series_raw_variable_element, "Categories"), ";");
    //             else if (time_raw_variable.type == RawVariableType::Binary)
    //                 time_raw_variable.categories = { "0", "1" };
    //             else
    //                 throw runtime_error("Categorical RawVariable Element is nullptr: Categories");
    //         }
    //     }
    // }

    // Samples

    const XMLElement* samples_element = data_set_element->FirstChildElement("Samples");

    if(!samples_element)
        throw runtime_error("Samples element is nullptr.\n");

    const Index samples_number = read_xml_index(samples_element, "SamplesNumber");

    if (has_sample_ids)
        sample_ids = get_tokens(read_xml_string(samples_element, "SamplesId"), " ");

    if (raw_variables.size() != 0)
    {
        const vector<vector<Index>> all_variable_indices = get_variable_indices();

        data.resize(samples_number, all_variable_indices[all_variable_indices.size() - 1][all_variable_indices[all_variable_indices.size() - 1].size() - 1] + 1);
        data.setZero();

        sample_uses.resize(samples_number);
        set_sample_uses(get_tokens(read_xml_string(samples_element, "SampleUses"), " "));
    }
    else
        data.resize(0, 0);

    // Missing values
    const XMLElement* missing_values_element = data_set_element->FirstChildElement("MissingValues");

    if(!missing_values_element)
        throw runtime_error("Missing values element is nullptr.\n");

    missing_values_number = read_xml_index(missing_values_element, "MissingValuesNumber");

    if (missing_values_number > 0)
    {
        set_missing_values_method(read_xml_string(missing_values_element, "MissingValuesMethod"));

        raw_variables_missing_values_number.resize(get_tokens(read_xml_string(missing_values_element, "RawVariablesMissingValuesNumber"), " ").size());

        for (Index i = 0; i < raw_variables_missing_values_number.size(); i++)
            raw_variables_missing_values_number(i) = stoi(get_tokens(read_xml_string(missing_values_element, "RawVariablesMissingValuesNumber"), " ")[i]);

        rows_missing_values_number = read_xml_index(missing_values_element, "RowsMissingValuesNumber");
    }

    // Preview data

    const XMLElement* preview_data_element = data_set_element->FirstChildElement("PreviewData");

    if(!preview_data_element)
        throw runtime_error("Preview data element is nullptr.\n");

    const XMLElement* preview_size_element = preview_data_element->FirstChildElement("PreviewSize");

    if(!preview_size_element)
        throw runtime_error("Preview size element is nullptr.\n");

    Index preview_size = 0;
    if (preview_size_element->GetText())
        preview_size = static_cast<Index>(atoi(preview_size_element->GetText()));

    start_element = preview_size_element;
    if(preview_size > 0){
        data_file_preview.resize(preview_size);

        for (Index i = 0; i < preview_size; ++i) {
            const XMLElement* row_data = start_element->NextSiblingElement("Row");
            start_element = row_data;

            if (row_data->Attribute("Item") != to_string(i + 1))
                throw runtime_error("Row item number (" + to_string(i + 1) + ") does not match (" + row_data->Attribute("Item") + ").\n");

            if(row_data->GetText())
                data_file_preview[i] = get_tokens(row_data->GetText(), ",");
        }
    }

    // Display

    set_display(read_xml_bool(data_set_element, "Display"));

    input_dimensions = { get_variables_number(Dataset::VariableUse::Input) };
    target_dimensions = { get_variables_number(Dataset::VariableUse::Target) };

}


void TimeSeriesDataset::impute_missing_values_mean()
{
    const vector<Index> used_sample_indices = get_used_sample_indices();
    const vector<Index> used_variable_indices = get_used_variable_indices();
    const vector<Index> input_variable_indices = get_variable_indices(Dataset::VariableUse::Input);
    const vector<Index> target_variable_indices = get_variable_indices(Dataset::VariableUse::Target);

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

                            buffer << "OpenNN Exception: Dataset class.\n"
                                   << "void Dataset::impute_missing_values_mean() const.\n"
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

void TimeSeriesDataset::fill_gaps()
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


Tensor<type, 2> TimeSeriesDataset::calculate_autocorrelations(const Index& lags_number) const
{
    const Index samples_number = get_samples_number();

    if(lags_number > samples_number)
        throw runtime_error("Lags number (" + to_string(lags_number) + ") "
                            "is greater than samples number (" + to_string(samples_number) + ") \n");

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


Tensor<type, 3> TimeSeriesDataset::calculate_cross_correlations(const Index& lags_number) const
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

    const Index new_lags_number = (samples_number == lags_number) ? (lags_number - 2)
                                : (samples_number == lags_number + 1) ? (lags_number - 1)
                                : lags_number;

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
            || raw_variables[j].type != RawVariableType::Numeric)
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
