//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   T E X T   D A T A S E T   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "text_data_set.h"
#include "strings.h"

namespace opennn
{

TextDataSet::TextDataSet() : DataSet()
{

}


const Index& TextDataSet::get_short_words_length() const
{
    return short_words_length;
}


const Index& TextDataSet::get_long_words_length() const
{
    return long_words_length;
}


const Tensor<Index,1>& TextDataSet::get_words_frequencies() const
{
    return words_frequencies;
}



/// Returns the string which will be used as separator in the data file for Text Classification.

string TextDataSet::get_text_separator_string() const
{
    switch(text_separator)
    {
    case Separator::Tab:
        return "Tab";

    case Separator::Semicolon:
        return "Semicolon";

    default:
        return string();
    }
}

/// Sets a new separator.
/// @param new_separator Separator value.

void TextDataSet::set_text_separator(const Separator& new_separator)
{
    separator = new_separator;
}


/// Sets a new separator from a string.
/// @param new_separator Char with the separator value.

void TextDataSet::set_text_separator(const string& new_separator_string)
{
    if(new_separator_string == "Tab")
    {
        text_separator = Separator::Tab;
    }
    else if(new_separator_string == "Semicolon")
    {
        text_separator = Separator::Semicolon;
    }
    else
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: DataSet class.\n"
               << "void set_text_separator(const string&) method.\n"
               << "Unknown separator: " << new_separator_string << ".\n";

        throw runtime_error(buffer.str());
    }
}

Tensor<string, 2> TextDataSet::get_text_data_file_preview() const
{
    return text_data_file_preview;
}


void TextDataSet::set_short_words_length(const Index& new_short_words_length)
{
    short_words_length = new_short_words_length;
}


void TextDataSet::set_long_words_length(const Index& new_long_words_length)
{
    long_words_length = new_long_words_length;
}


void TextDataSet::set_words_frequencies(const Tensor<Index,1>& new_words_frequencies)
{
    words_frequencies = new_words_frequencies;
}


/// Serializes the data set object into a XML document of the TinyXML library without keep the DOM tree in memory.

void TextDataSet::write_XML(tinyxml2::XMLPrinter& file_stream) const
{
/*
    ostringstream buffer;

    time_t start, finish;
    time(&start);

    file_stream.OpenElement("DataSet");

    // Data file

    file_stream.OpenElement("DataFile");

    file_stream.OpenElement("FileType");

    file_stream.PushText("csv");

    file_stream.CloseElement();

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

    // Text separator
    {
        file_stream.OpenElement("TextSeparator");

        file_stream.PushText(get_text_separator_string().c_str());

        file_stream.CloseElement();
    }

    // Columns names
    {
        file_stream.OpenElement("ColumnsNames");

        buffer.str("");
        buffer << has_raw_variables_names;

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

    // Short words length
    {
        file_stream.OpenElement("ShortWordsLength");

        buffer.str("");
        buffer << get_short_words_length();

        file_stream.PushText(buffer.str().c_str());

        file_stream.CloseElement();
    }

    // Long words length
    {
        file_stream.OpenElement("LongWordsLength");

        buffer.str("");
        buffer << get_long_words_length();

        file_stream.PushText(buffer.str().c_str());

        file_stream.CloseElement();
    }

    // Stop words list
    {
        file_stream.OpenElement("StopWordsList");

        const Index stop_words_number = stop_words.dimension(0);

        buffer.str("");

        for(Index i = 0; i < stop_words_number; i++)
        {
            buffer << stop_words(i);

            if(i != stop_words_number-1) buffer << ",";
        }

        file_stream.PushText(buffer.str().c_str());

        file_stream.CloseElement();
    }

    // Codification
    {
        file_stream.OpenElement("Codification");

        buffer.str("");
        buffer << get_codification_string();

        file_stream.PushText(buffer.str().c_str());

        file_stream.CloseElement();
    }

    // Close DataFile

    file_stream.CloseElement();

    // Columns

    file_stream.OpenElement("Columns");

    // Columns number
    {
        file_stream.OpenElement("ColumnsNumber");

        buffer.str("");
        buffer << get_raw_variables_number();

        file_stream.PushText(buffer.str().c_str());

        file_stream.CloseElement();
    }

    // Columns items

    const Index raw_variables_number = get_raw_variables_number();
    {
        for(Index i = 0; i < raw_variables_number; i++)
        {
            file_stream.OpenElement("Column");

            file_stream.PushAttribute("Item", to_string(i+1).c_str());

            raw_variables(i).write_XML(file_stream);

            file_stream.CloseElement();
        }
    }

    // Close raw_variables

    file_stream.CloseElement();

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

            const Index raw_variables_number = columns_missing_values_number.size();

            buffer.str("");

            for(Index i = 0; i < raw_variables_number; i++)
            {
                buffer << columns_missing_values_number(i);

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

    // Words frequencies
    {
        file_stream.OpenElement("WordsFrequencies");

        for(Index i = 0; i < words_frequencies.size(); i++)
        {
            buffer.str("");
            buffer << words_frequencies(i);

            file_stream.PushText(buffer.str().c_str());
            if(i != words_frequencies.size()-1) file_stream.PushText(" ");

        }

        file_stream.CloseElement();
    }

    // Preview data

    file_stream.OpenElement("PreviewData");

    file_stream.OpenElement("PreviewSize");

    buffer.str("");

    // Row and Targets
    {
        buffer << text_data_file_preview.dimension(0);

        file_stream.PushText(buffer.str().c_str());

        file_stream.CloseElement();

        for(Index i = 0; i < text_data_file_preview.dimension(0); i++)
        {
            file_stream.OpenElement("Row");

            file_stream.PushAttribute("Item", to_string(i+1).c_str());

            file_stream.PushText(text_data_file_preview(i,0).c_str());

            file_stream.CloseElement();
        }

        for(Index i = 0; i < text_data_file_preview.dimension(0); i++)
        {
            file_stream.OpenElement("Target");

            file_stream.PushAttribute("Item", to_string(i+1).c_str());

            file_stream.PushText(text_data_file_preview(i,1).c_str());

            file_stream.CloseElement();
        }
    }

    // Close preview data

    file_stream.CloseElement();

    // Close data set

    file_stream.CloseElement();

    time(&finish);
}


void TextDataSet::from_XML(const tinyxml2::XMLDocument& data_set_document)
{
    ostringstream buffer;

    // Data set element

    const tinyxml2::XMLElement* data_set_element = data_set_document.FirstChildElement("DataSet");

    if(!data_set_element)
    {
        buffer << "OpenNN Exception: DataSet class.\n"
               << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
               << "Data set element is nullptr.\n";

        throw runtime_error(buffer.str());
    }

    // Data file

    const tinyxml2::XMLElement* data_file_element = data_set_element->FirstChildElement("DataFile");

    if(!data_file_element)
    {
        buffer << "OpenNN Exception: DataSet class.\n"
               << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
               << "Data file element is nullptr.\n";

        throw runtime_error(buffer.str());
    }

    // Data file name

    const tinyxml2::XMLElement* data_file_name_element = data_file_element->FirstChildElement("DataFileName");

    if(!data_file_name_element)
    {
        buffer << "OpenNN Exception: DataSet class.\n"
               << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
               << "DataFileName element is nullptr.\n";

        throw runtime_error(buffer.str());
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

    // Text separator

    const tinyxml2::XMLElement* text_separator_element = data_file_element->FirstChildElement("TextSeparator");

    if(text_separator_element)
    {
        if(text_separator_element->GetText())
        {
            const string new_separator = text_separator_element->GetText();

            try
            {
                set_text_separator(new_separator);
            }
            catch(const exception& e)
            {
                cerr << e.what() << endl;
            }
        }
    }

    // Has raw_variables names

    const tinyxml2::XMLElement* columns_names_element = data_file_element->FirstChildElement("ColumnsNames");

    if(columns_names_element)
    {
        const string new_raw_variables_names_string = columns_names_element->GetText();

        try
        {
            set_has_raw_variables_names(new_raw_variables_names_string == "1");
        }
        catch(const exception& e)
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
        catch(const exception& e)
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

    // short words length

    const tinyxml2::XMLElement* short_words_length_element = data_file_element->FirstChildElement("ShortWordsLength");

    if(short_words_length_element)
    {
        if(short_words_length_element->GetText())
        {
            const int new_short_words_length = Index(atoi(short_words_length_element->GetText()));

            set_short_words_length(new_short_words_length);
        }
    }

    // Long words length

    const tinyxml2::XMLElement* long_words_length_element = data_file_element->FirstChildElement("LongWordsLength");

    if(long_words_length_element)
    {
        if(long_words_length_element->GetText())
        {
            const int new_long_words_length = Index(atoi(long_words_length_element->GetText()));

            set_long_words_length(new_long_words_length);
        }
    }

    // Stop words list

    const tinyxml2::XMLElement* stop_words_list_element = data_file_element->FirstChildElement("StopWordsList");

    if(stop_words_list_element)
    {
        if(stop_words_list_element->GetText())
        {
            const string new_stop_words_list = stop_words_list_element->GetText();

            stop_words = get_tokens(new_stop_words_list,",");
        }
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

        throw runtime_error(buffer.str());
    }

    // Columns number

    const tinyxml2::XMLElement* columns_number_element = columns_element->FirstChildElement("ColumnsNumber");

    if(!columns_number_element)
    {
        buffer << "OpenNN Exception: DataSet class.\n"
               << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
               << "Columns number element is nullptr.\n";

        throw runtime_error(buffer.str());
    }

    Index new_raw_variables_number = 0;

    if(columns_number_element->GetText())
    {
        new_raw_variables_number = Index(atoi(columns_number_element->GetText()));

        set_raw_variables_number(new_raw_variables_number);
    }

    // Columns

    const tinyxml2::XMLElement* start_element = columns_number_element;

    if(new_raw_variables_number > 0)
    {
        for(Index i = 0; i < new_raw_variables_number; i++)
        {
            const tinyxml2::XMLElement* column_element = start_element->NextSiblingElement("Column");
            start_element = column_element;

            if(column_element->Attribute("Item") != to_string(i+1))
            {
                buffer << "OpenNN Exception: DataSet class.\n"
                       << "void DataSet:from_XML(const tinyxml2::XMLDocument&) method.\n"
                       << "Column item number (" << i+1 << ") does not match (" << column_element->Attribute("Item") << ").\n";

                throw runtime_error(buffer.str());
            }

            // Name

            const tinyxml2::XMLElement* name_element = column_element->FirstChildElement("Name");

            if(!name_element)
            {
                buffer << "OpenNN Exception: DataSet class.\n"
                       << "void Column::from_XML(const tinyxml2::XMLDocument&) method.\n"
                       << "Name element is nullptr.\n";

                throw runtime_error(buffer.str());
            }

            if(name_element->GetText())
            {
                const string new_name = name_element->GetText();

                raw_variables(i).name = new_name;
            }

            // Scaler

            const tinyxml2::XMLElement* scaler_element = column_element->FirstChildElement("Scaler");

            if(!scaler_element)
            {
                buffer << "OpenNN Exception: DataSet class.\n"
                       << "void DataSet::from_XML(const tinyxml2::XMLDocument&) method.\n"
                       << "Scaler element is nullptr.\n";

                throw runtime_error(buffer.str());
            }

            if(scaler_element->GetText())
            {
                const string new_scaler = scaler_element->GetText();

                raw_variables(i).set_scaler(new_scaler);
            }

            // Column use

            const tinyxml2::XMLElement* raw_variable_use_element = column_element->FirstChildElement("ColumnUse");

            if(!raw_variable_use_element)
            {
                buffer << "OpenNN Exception: DataSet class.\n"
                       << "void DataSet::from_XML(const tinyxml2::XMLDocument&) method.\n"
                       << "Column use element is nullptr.\n";

                throw runtime_error(buffer.str());
            }

            if(raw_variable_use_element->GetText())
            {
                const string new_raw_variable_use = raw_variable_use_element->GetText();

                raw_variables(i).set_use(new_raw_variable_use);
            }

            // Type

            const tinyxml2::XMLElement* type_element = column_element->FirstChildElement("Type");

            if(!type_element)
            {
                buffer << "OpenNN Exception: DataSet class.\n"
                       << "void Column::from_XML(const tinyxml2::XMLDocument&) method.\n"
                       << "Type element is nullptr.\n";

                throw runtime_error(buffer.str());
            }

            if(type_element->GetText())
            {
                const string new_type = type_element->GetText();
                raw_variables(i).set_type(new_type);
            }

            if(raw_variables(i).type == RawVariableType::Categorical || raw_variables(i).type == RawVariableType::Binary)
            {
                // Categories

                const tinyxml2::XMLElement* categories_element = column_element->FirstChildElement("Categories");

                if(!categories_element)
                {
                    buffer << "OpenNN Exception: DataSet class.\n"
                           << "void Column::from_XML(const tinyxml2::XMLDocument&) method.\n"
                           << "Categories element is nullptr.\n";

                    throw runtime_error(buffer.str());
                }

                if(categories_element->GetText())
                {
                    const string new_categories = categories_element->GetText();

                    raw_variables(i).categories = get_tokens(new_categories, ';');
                }

                // Categories uses

                const tinyxml2::XMLElement* categories_uses_element = column_element->FirstChildElement("CategoriesUses");

                if(!categories_uses_element)
                {
                    buffer << "OpenNN Exception: DataSet class.\n"
                           << "void Column::from_XML(const tinyxml2::XMLDocument&) method.\n"
                           << "Categories uses element is nullptr.\n";

                    throw runtime_error(buffer.str());
                }

                if(categories_uses_element->GetText())
                {
                    const string new_categories_uses = categories_uses_element->GetText();

                    raw_variables(i).set_categories_uses(get_tokens(new_categories_uses, ';'));
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

            throw runtime_error(buffer.str());
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

        throw runtime_error(buffer.str());
    }

    // Samples number

    const tinyxml2::XMLElement* samples_number_element = samples_element->FirstChildElement("SamplesNumber");

    if(!samples_number_element)
    {
        buffer << "OpenNN Exception: DataSet class.\n"
               << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
               << "Samples number element is nullptr.\n";

        throw runtime_error(buffer.str());
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

        throw runtime_error(buffer.str());
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

        throw runtime_error(buffer.str());
    }

    // Missing values method

    const tinyxml2::XMLElement* missing_values_method_element = missing_values_element->FirstChildElement("MissingValuesMethod");

    if(!missing_values_method_element)
    {
        buffer << "OpenNN Exception: DataSet class.\n"
               << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
               << "Missing values method element is nullptr.\n";

        throw runtime_error(buffer.str());
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

        throw runtime_error(buffer.str());
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

            throw runtime_error(buffer.str());
        }

        if(columns_missing_values_number_element->GetText())
        {
            Tensor<string, 1> new_raw_variables_missing_values_number = get_tokens(columns_missing_values_number_element->GetText(), ' ');

            columns_missing_values_number.resize(new_raw_variables_missing_values_number.size());

            for(Index i = 0; i < new_raw_variables_missing_values_number.size(); i++)
            {
                columns_missing_values_number(i) = atoi(new_raw_variables_missing_values_number(i).c_str());
            }
        }

        // Rows missing values number

        const tinyxml2::XMLElement* rows_missing_values_number_element = missing_values_element->FirstChildElement("RowsMissingValuesNumber");

        if(!rows_missing_values_number_element)
        {
            buffer << "OpenNN Exception: DataSet class.\n"
                   << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
                   << "Rows missing values number element is nullptr.\n";

            throw runtime_error(buffer.str());
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

        throw runtime_error(buffer.str());
    }

    // Preview size

    const tinyxml2::XMLElement* preview_size_element = preview_data_element->FirstChildElement("PreviewSize");

    if(!preview_size_element)
    {
        buffer << "OpenNN Exception: DataSet class.\n"
               << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
               << "Preview size element is nullptr.\n";

        throw runtime_error(buffer.str());
    }

    Index new_preview_size = 0;

    if(preview_size_element->GetText())
    {
        new_preview_size = Index(atoi(preview_size_element->GetText()));

        if(new_preview_size > 0) data_file_preview.resize(new_preview_size);
        if(new_preview_size > 0) text_data_file_preview.resize(new_preview_size, 2);
    }

    // Preview data

    start_element = preview_size_element;

    if(model_type != ModelType::TextClassification)
    {
        for(Index i = 0; i < new_preview_size; i++)
        {
            const tinyxml2::XMLElement* row_element = start_element->NextSiblingElement("Row");
            start_element = row_element;

            if(row_element->Attribute("Item") != to_string(i+1))
            {
                buffer << "OpenNN Exception: DataSet class.\n"
                       << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
                       << "Row item number (" << i+1 << ") does not match (" << row_element->Attribute("Item") << ").\n";

                throw runtime_error(buffer.str());
            }

            if(row_element->GetText())
            {
                data_file_preview(i) = get_tokens(row_element->GetText(), ',');
            }
        }
    }
    else
    {
        for(Index i = 0; i < new_preview_size; i++)
        {
            const tinyxml2::XMLElement* row_element = start_element->NextSiblingElement("Row");
            start_element = row_element;

            if(row_element->Attribute("Item") != to_string(i+1))
            {
                buffer << "OpenNN Exception: DataSet class.\n"
                       << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
                       << "Row item number (" << i+1 << ") does not match (" << row_element->Attribute("Item") << ").\n";

                throw runtime_error(buffer.str());
            }

            if(row_element->GetText())
            {
                text_data_file_preview(i,0) = row_element->GetText();
            }
        }

        for(Index i = 0; i < new_preview_size; i++)
        {
            const tinyxml2::XMLElement* row_element = start_element->NextSiblingElement("Target");
            start_element = row_element;

            if(row_element->Attribute("Item") != to_string(i+1))
            {
                buffer << "OpenNN Exception: DataSet class.\n"
                       << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
                       << "Target item number (" << i+1 << ") does not match (" << row_element->Attribute("Item") << ").\n";

                throw runtime_error(buffer.str());
            }

            if(row_element->GetText())
            {
                text_data_file_preview(i,1) = row_element->GetText();
            }
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
        catch(const exception& e)
        {
            cerr << e.what() << endl;
        }
    }
*/
}


/// Transforms a sentence to Tensor, according to dataset raw_variables.
/// @param sentence Sentence that we will transform

Tensor<type,1> TextDataSet::sentence_to_data(const string& sentence) const
{
/*
    Tensor<string, 1> tokens = get_tokens(sentence,' ');

    Tensor<type, 1> vector(get_raw_variables_number() - 1);

    TextAnalytics text_analytics;
    text_analytics.set_short_words_length(short_words_length);
    text_analytics.set_long_words_length(long_words_length);

    Tensor<Tensor<string,1>,1> words = text_analytics.preprocess(tokens);

    WordBag word_bag = text_analytics.calculate_word_bag(words);

    const Tensor<string,1> columns_names = get_raw_variables_names();

    const Index words_number = word_bag.words.size();

    vector.setZero();

    for(Index i = 0; i < words_number; i++)
    {
        if( contains(columns_names, word_bag.words(i)) )
        {
            auto it = find(columns_names.data(), columns_names.data() + columns_names.size(), word_bag.words(i));
            const Index index = it - columns_names.data();

            vector(index) = type(word_bag.frequencies(i));
        }
    }

    return vector;
*/
    return Tensor<type, 1>();

};


void TextDataSet::read_txt()
{
    cout << "Reading .txt file..." << endl;
/*
    TextAnalytics text_analytics;

    text_analytics.set_separator(get_text_separator_string());
    text_analytics.set_short_words_length(short_words_length);
    text_analytics.set_long_words_length(long_words_length);
    text_analytics.set_stop_words(stop_words);

    cout << "Loading documents..." << endl;

    text_analytics.load_documents(data_source_path);

    const Tensor<Tensor<string, 1>, 1> document = text_analytics.get_documents();

    Tensor<Tensor<string, 1>, 1> targets = text_analytics.get_targets();

    const Tensor<string, 1> full_document = text_analytics.join(document);

    cout << "Processing documents..." << endl;

    const Tensor<Tensor<string, 1>, 1> document_tokens = text_analytics.preprocess(full_document);

    cout << "Calculating wordbag..." << endl;

    const WordBag document_word_bag = text_analytics.calculate_word_bag(document_tokens);
    set_words_frequencies(document_word_bag.frequencies);

    Tensor<string, 1> columns_names = document_word_bag.words;
    const Index document_words_number = document_word_bag.words.size();

    Tensor<type, 1> row(document_words_number);

    // Output

    cout << "Writting data file..." << endl;

    string transformed_data_path = data_source_path;
    replace(transformed_data_path,".txt","_data.txt");
    replace(transformed_data_path,".csv","_data.csv");

    std::ofstream file;
    file.open(transformed_data_path);

    for(Index i  = type(0); i < document_words_number; i++)
        file << columns_names(i) << ";";
    file << "target_variable" << "\n";

    // Data file preview

    Index preview_size = 4;

    text_data_file_preview.resize(preview_size,2);

    for(Index i = 0; i < preview_size - 1; i++)
    {
        text_data_file_preview(i,0) = document(0)(i);
        text_data_file_preview(i,1) = targets(0)(i);
    }

    text_data_file_preview(preview_size - 1, 0) = document(0)(document(0).size()-1);
    text_data_file_preview(preview_size - 1, 1) = targets(0)(targets(0).size()-1);

#pragma omp parallel for

    for(Index i = 0; i < document.size(); i++)
    {
        Tensor<string, 1> sheet = document(i);

        for(Index j = 0; j < sheet.size(); j++)
        {
            row.setZero();

            const string line = sheet(j);

            Tensor<string,1> tokens = get_tokens(line);

            Tensor<Tensor<string, 1>,1> processed_tokens = text_analytics.preprocess(tokens);

            WordBag line_word_bag = text_analytics.calculate_word_bag(processed_tokens);

            Tensor<string, 1> line_words = line_word_bag.words;

            Tensor<Index, 1> line_frequencies = line_word_bag.frequencies;

            for(Index k = 0; k < line_words.size(); k++)
            {
                if( contains(columns_names, line_words(k)) )
                {
                    auto it = find(columns_names.data(), columns_names.data() + document_words_number, line_words(k));

                    const Index word_index = it - columns_names.data();

                    row(word_index) = type(line_frequencies(k));
                }
            }

            for(Index k = 0; k < document_words_number; k++)
                file << row(k) << ";";
            for_each(targets(i)(j).begin(), targets(i)(j).end(), [](char & c){
                c = ::tolower(c);});
            file << "target_" + targets(i)(j) << "\n";
        }
    }

    file.close();

    data_source_path = transformed_data_path;
    separator = Separator::Semicolon;
    has_raw_variables_names = true;

    load_data(); 

    for(Index i = 0; i < get_input_raw_variables_number(); i++)
        set_raw_variable_type(i,RawVariableType::Numeric);
*/
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
