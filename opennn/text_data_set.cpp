//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   T E X T   D A T A S E T   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include <sstream>
#include <iostream>
#include <fstream>
//#include <limits>
//#include <math.h>

#include "text_data_set.h"
#include "strings_utilities.h"
#include "word_bag.h"
#include "tensors.h"

namespace opennn
{

TextDataSet::TextDataSet() : DataSet()
{
    stop_words.resize(242);

    stop_words.setValues(
        { "i", "me", "my", "myself", "we", "us", "our", "ours", "ourselves", "you", "u", "your", "yours", "yourself", "yourselves", "he",
         "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves",
         "what", "which", "who", "whom", "this", "that", "these", "those", "im", "am", "m", "is", "are", "was", "were", "be", "been", "being",
         "have", "has", "s", "ve", "re", "ll", "t", "had", "having", "do", "does", "did", "doing", "would", "d", "shall", "should", "could",
         "ought", "i'm", "you're", "he's", "she's", "it's", "we're", "they're", "i've", "you've", "we've", "they've", "i'd", "you'd", "he'd",
         "she'd", "we'd", "they'd", "i'll", "you'll", "he'll", "she'll", "we'll", "they'll", "isn't", "aren't", "wasn't", "weren't", "hasn't",
         "haven't", "hadn't", "doesn't", "don't", "didn't", "won't", "wouldn't", "shan't", "shouldn't", "can't", "cannot", "couldn't", "mustn't",
         "let's", "that's", "who's", "what's", "here's", "there's", "when's", "where's", "why's", "how's", "daren't", "needn't", "oughtn't",
         "mightn't", "shes", "its", "were", "theyre", "ive", "youve", "weve", "theyve", "id", "youd", "hed", "shed", "wed", "theyd",
         "ill", "youll", "hell", "shell", "well", "theyll", "isnt", "arent", "wasnt", "werent", "hasnt", "havent", "hadnt",
         "doesnt", "dont", "didnt", "wont", "wouldnt", "shant", "shouldnt", "cant", "cannot", "couldnt", "mustnt", "lets",
         "thats", "whos", "whats", "heres", "theres", "whens", "wheres", "whys", "hows", "darent", "neednt", "oughtnt",
         "mightnt", "a", "an", "the", "and", "n", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about",
         "against", "between", "into", "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", "on",
         "off", "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both",
         "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very" });

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

//string TextDataSet::get_text_separator_string() const
//{
//    switch(text_separator)
//    {
//    case Separator::Tab:
//        return "Tab";

//    case Separator::Semicolon:
//        return "Semicolon";

//    default:
//        return string();
//    }
//}


// Sets a new separator.
// @param new_separator Separator value.

//void TextDataSet::set_text_separator(const Separator& new_separator)
//{
//    separator = new_separator;
//}


// Sets a new separator from a string.
// @param new_separator Char with the separator value.

//void TextDataSet::set_text_separator(const string& new_separator_string)
//{
//    if(new_separator_string == "Tab")
//    {
//        text_separator = Separator::Tab;
//    }
//    else if(new_separator_string == "Semicolon")
//    {
//        text_separator = Separator::Semicolon;
//    }
//    else
//    {
//        ostringstream buffer;

//        buffer << "OpenNN Exception: DataSet class.\n"
//               << "void set_text_separator(const string&) method.\n"
//               << "Unknown separator: " << new_separator_string << ".\n";

//        throw runtime_error(buffer.str());
//    }
//}


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


//void TextDataSet::set_words_frequencies(const Tensor<Index,1>& new_words_frequencies)
//{
//    words_frequencies = new_words_frequencies;
//}


/// Serializes the data set object into a XML document of the TinyXML library without keep the DOM tree in memory.

void TextDataSet::write_XML(tinyxml2::XMLPrinter& file_stream) const
{

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
        file_stream.OpenElement("DataSourcePath");

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
        file_stream.OpenElement("Separator");

        file_stream.PushText(get_separator_string().c_str());

        file_stream.CloseElement();
    }

    // raw_variables names
    {
        file_stream.OpenElement("RawVariablesNames");

        buffer.str("");
        buffer << has_header;

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

    // raw_variables

    file_stream.OpenElement("RawVariables");

    // raw_variables number
    {
        file_stream.OpenElement("RawVariablesNumber");

        buffer.str("");
        buffer << get_raw_variables_number();

        file_stream.PushText(buffer.str().c_str());

        file_stream.CloseElement();
    }

    // raw_variables items

    const Index raw_variables_number = get_raw_variables_number();
    {
        for(Index i = 0; i < raw_variables_number; i++)
        {
            file_stream.OpenElement("RawVariable");

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
        // raw_variables missing values number
        {
            file_stream.OpenElement("raw_variablesMissingValuesNumber");

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

    const tinyxml2::XMLElement* data_file_name_element = data_file_element->FirstChildElement("DataSourcePath");

    if(!data_file_name_element)
    {
        buffer << "OpenNN Exception: DataSet class.\n"
               << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
               << "DataSourcePath element is nullptr.\n";

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

    const tinyxml2::XMLElement* text_separator_element = data_file_element->FirstChildElement("Separator");

    if(text_separator_element)
    {
        if(text_separator_element->GetText())
        {
            const string new_separator = text_separator_element->GetText();

            try
            {
                set_separator(new_separator);
            }
            catch(const exception& e)
            {
                cerr << e.what() << endl;
            }
        }
    }

    // Has raw_variables names

    const tinyxml2::XMLElement* raw_variables_names_element = data_file_element->FirstChildElement("RawVariablesNames");

    if(raw_variables_names_element)
    {
        const string new_raw_variables_names_string = raw_variables_names_element->GetText();

        try
        {
            set_has_header(new_raw_variables_names_string == "1");
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

            stop_words = get_tokens(new_stop_words_list, ",");

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

    // raw_variables

    const tinyxml2::XMLElement* raw_variables_element = data_set_element->FirstChildElement("RawVariables");

    if(!raw_variables_element)
    {
        buffer << "OpenNN Exception: DataSet class.\n"
               << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
               << "raw_variables element is nullptr.\n";

        throw runtime_error(buffer.str());
    }

    // raw_variables number

    const tinyxml2::XMLElement* raw_variables_number_element = raw_variables_element->FirstChildElement("RawVariablesNumber");

    if(!raw_variables_number_element)
    {
        buffer << "OpenNN Exception: DataSet class.\n"
               << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
               << "raw_variables number element is nullptr.\n";

        throw runtime_error(buffer.str());
    }

    Index new_raw_variables_number = 0;

    if(raw_variables_number_element->GetText())
    {
        new_raw_variables_number = Index(atoi(raw_variables_number_element->GetText()));

        set_raw_variables_number(new_raw_variables_number);
    }

    // raw_variables

    const tinyxml2::XMLElement* start_element = raw_variables_number_element;

    if(new_raw_variables_number > 0)
    {
        for(Index i = 0; i < new_raw_variables_number; i++)
        {
            const tinyxml2::XMLElement* column_element = start_element->NextSiblingElement("RawVariable");
            start_element = column_element;

            if(column_element->Attribute("Item") != to_string(i+1))
            {
                buffer << "OpenNN Exception: DataSet class.\n"
                       << "void DataSet:from_XML(const tinyxml2::XMLDocument&) method.\n"
                       << "raw_variable item number (" << i+1 << ") does not match (" << column_element->Attribute("Item") << ").\n";

                throw runtime_error(buffer.str());
            }

            // Name

            const tinyxml2::XMLElement* name_element = column_element->FirstChildElement("Name");

            if(!name_element)
            {
                buffer << "OpenNN Exception: DataSet class.\n"
                       << "void raw_variable::from_XML(const tinyxml2::XMLDocument&) method.\n"
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

            // raw_variable use

            const tinyxml2::XMLElement* raw_variable_use_element = column_element->FirstChildElement("RawVariableUse");

            if(!raw_variable_use_element)
            {
                buffer << "OpenNN Exception: DataSet class.\n"
                       << "void DataSet::from_XML(const tinyxml2::XMLDocument&) method.\n"
                       << "raw_variable use element is nullptr.\n";

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
                       << "void raw_variable::from_XML(const tinyxml2::XMLDocument&) method.\n"
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
                           << "void raw_variable::from_XML(const tinyxml2::XMLDocument&) method.\n"
                           << "Categories element is nullptr.\n";

                    throw runtime_error(buffer.str());
                }

                if(categories_element->GetText())
                {
                    const string new_categories = categories_element->GetText();

                    raw_variables(i).categories = get_tokens(new_categories, ";");

                }

                // Categories uses

                const tinyxml2::XMLElement* categories_uses_element = column_element->FirstChildElement("CategoriesUses");

                if(!categories_uses_element)
                {
                    buffer << "OpenNN Exception: DataSet class.\n"
                           << "void raw_variable::from_XML(const tinyxml2::XMLDocument&) method.\n"
                           << "Categories uses element is nullptr.\n";

                    throw runtime_error(buffer.str());
                }

                if(categories_uses_element->GetText())
                {
                    const string new_categories_uses = categories_uses_element->GetText();

                    raw_variables(i).set_categories_uses(get_tokens(new_categories_uses, ";"));
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

            string separator = ",";

            if(new_rows_labels.find(",") == string::npos
            && new_rows_labels.find(";") != string::npos) {
                separator = ";";
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
        set_samples_uses(get_tokens(samples_uses_element->GetText(), " "));
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
        // raw_variables Missing values number

        const tinyxml2::XMLElement* raw_variables_missing_values_number_element = missing_values_element->FirstChildElement("raw_variablesMissingValuesNumber");

        if(!raw_variables_missing_values_number_element)
        {
            buffer << "OpenNN Exception: DataSet class.\n"
                   << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
                   << "raw_variables missing values number element is nullptr.\n";

            throw runtime_error(buffer.str());
        }

        if(raw_variables_missing_values_number_element->GetText())
        {
            Tensor<string, 1> new_raw_variables_missing_values_number
                = get_tokens(raw_variables_missing_values_number_element->GetText(), " ");

            raw_variables_missing_values_number.resize(new_raw_variables_missing_values_number.size());

            for(Index i = 0; i < new_raw_variables_missing_values_number.size(); i++)
            {
                raw_variables_missing_values_number(i) = atoi(new_raw_variables_missing_values_number(i).c_str());
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
                data_file_preview(i) = get_tokens(row_element->GetText(), ",");
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
}


/// Transforms a sentence to Tensor, according to dataset raw_variables.
/// @param sentence Sentence that we will transform

Tensor<type,1> TextDataSet::sentence_to_data(const string& sentence) const
{
    const Index raw_variables_number = get_raw_variables_number();
    const Tensor<string,1> raw_variables_names = get_raw_variables_names();

    const Tensor<string, 1> tokens = get_tokens(sentence, " ");

    Tensor<type, 1> vectorx(raw_variables_number - 1);
    vectorx.setZero();

    const Tensor<Tensor<string,1>,1> words = preprocess(tokens);

    const WordBag word_bag = calculate_word_bag(words, get_separator_string());

    const Index words_number = word_bag.size();

    for(Index i = 0; i < words_number; i++)
    {
/*
        if( contains(raw_variables_names, word_bag.words(i)) )
        {
            auto it = find(raw_variables_names.data(), raw_variables_names.data() + raw_variables_names.size(), word_bag.words(i));
            const Index index = it - raw_variables_names.data();

            vectorx(index) = type(word_bag.frequencies(i));
        }
*/
    }

    return vectorx;
}


/// Reduces inflected (or sometimes derived) words to their word stem, base or root form (english language).
/// @param tokens

Tensor<Tensor<string,1>,1> stem(const Tensor<Tensor<string,1>,1>& tokens)
{
    const Index documents_number = tokens.size();

    Tensor<Tensor<string,1>,1> new_tokenized_documents(documents_number);

    // Set vowels and suffixes

    Tensor<string,1> vowels(6);
    vowels.setValues({"a","e","i","o","u","y"});

    Tensor<string,1> double_consonants(9);
    double_consonants.setValues({"bb", "dd", "ff", "gg", "mm", "nn", "pp", "rr", "tt"});

    Tensor<string,1> li_ending(10);
    li_ending.setValues({"c", "d", "e", "g", "h", "k", "m", "n", "r", "t"});

    const Index step0_suffixes_size = 3;

    Tensor<string,1> step0_suffixes(step0_suffixes_size);

    step0_suffixes.setValues({"'s'", "'s", "'"});

    const Index step1a_suffixes_size = 6;

    Tensor<string,1> step1a_suffixes(step1a_suffixes_size);

    step1a_suffixes.setValues({"sses", "ied", "ies", "us", "ss", "s"});

    const Index step1b_suffixes_size = 6;

    Tensor<string,1> step1b_suffixes(step1b_suffixes_size);

    step1b_suffixes.setValues({"eedly", "ingly", "edly", "eed", "ing", "ed"});

    const Index step2_suffixes_size = 25;

    Tensor<string,1> step2_suffixes(step2_suffixes_size);

    step2_suffixes.setValues({"ization",
                              "ational",
                              "fulness",
                              "ousness",
                              "iveness",
                              "tional",
                              "biliti",
                              "lessli",
                              "entli",
                              "ation",
                              "alism",
                              "aliti",
                              "ousli",
                              "iviti",
                              "fulli",
                              "enci",
                              "anci",
                              "abli",
                              "izer",
                              "ator",
                              "alli",
                              "bli",
                              "ogi",
                              "li"});

    const Index step3_suffixes_size = 9;

    Tensor<string,1> step3_suffixes(step3_suffixes_size);

    step3_suffixes.setValues({"ational", "tional", "alize", "icate", "iciti", "ative", "ical", "ness", "ful"});

    const Index step4_suffixes_size = 18;

    Tensor<string,1> step4_suffixes(step4_suffixes_size);

    step4_suffixes.setValues({"ement", "ance", "ence", "able", "ible", "ment", "ant", "ent", "ism", "ate", "iti", "ous",
                              "ive", "ize", "ion", "al", "er", "ic"});

    Tensor<string, 2> special_words(40,2);

    special_words(0,0) = "skis";        special_words(0,1) = "ski";
    special_words(1,0) = "skies";       special_words(1,1) = "sky";
    special_words(2,0) = "dying";       special_words(2,1) = "die";
    special_words(3,0) = "lying";       special_words(3,1) = "lie";
    special_words(4,0) = "tying";       special_words(4,1) = "tie";
    special_words(5,0) = "idly";        special_words(5,1) = "idl";
    special_words(6,0) = "gently";      special_words(6,1) = "gentl";
    special_words(7,0) = "ugly";        special_words(7,1) = "ugli";
    special_words(8,0) = "early";       special_words(8,1) = "earli";
    special_words(9,0) = "only";        special_words(9,1) = "onli";
    special_words(10,0) = "singly";     special_words(10,1) = "singl";
    special_words(11,0) = "sky";        special_words(11,1) = "sky";
    special_words(12,0) = "news";       special_words(12,1) = "news";
    special_words(13,0) = "howe";       special_words(13,1) = "howe";
    special_words(14,0) = "atlas";      special_words(14,1) = "atlas";
    special_words(15,0) = "cosmos";     special_words(15,1) = "cosmos";
    special_words(16,0) = "bias";       special_words(16,1) = "bias";
    special_words(17,0) = "andes";      special_words(17,1) = "andes";
    special_words(18,0) = "inning";     special_words(18,1) = "inning";
    special_words(19,0) = "innings";    special_words(19,1) = "inning";
    special_words(20,0) = "outing";     special_words(20,1) = "outing";
    special_words(21,0) = "outings";    special_words(21,1) = "outing";
    special_words(22,0) = "canning";    special_words(22,1) = "canning";
    special_words(23,0) = "cannings";   special_words(23,1) = "canning";
    special_words(24,0) = "herring";    special_words(24,1) = "herring";
    special_words(25,0) = "herrings";   special_words(25,1) = "herring";
    special_words(26,0) = "earring";    special_words(26,1) = "earring";
    special_words(27,0) = "earrings";   special_words(27,1) = "earring";
    special_words(28,0) = "proceed";    special_words(28,1) = "proceed";
    special_words(29,0) = "proceeds";   special_words(29,1) = "proceed";
    special_words(30,0) = "proceeded";  special_words(30,1) = "proceed";
    special_words(31,0) = "proceeding"; special_words(31,1) = "proceed";
    special_words(32,0) = "exceed";     special_words(32,1) = "exceed";
    special_words(33,0) = "exceeds";    special_words(33,1) = "exceed";
    special_words(34,0) = "exceeded";   special_words(34,1) = "exceed";
    special_words(35,0) = "exceeding";  special_words(35,1) = "exceed";
    special_words(36,0) = "succeed";    special_words(36,1) = "succeed";
    special_words(37,0) = "succeeds";   special_words(37,1) = "succeed";
    special_words(38,0) = "succeeded";  special_words(38,1) = "succeed";
    special_words(39,0) = "succeeding"; special_words(39,1) = "succeed";

#pragma omp parallel for
    for(Index i = 0; i < documents_number; i++)
    {
        Tensor<string,1> current_document = tokens(i);

        replace_substring(current_document, "’", "'");
        replace_substring(current_document, "‘", "'");
        replace_substring(current_document, "‛", "'");

        for(Index j = 0; j < current_document.size(); j++)
        {
            string current_word = current_document(j);

            trim(current_word);

            if(contains(special_words.chip(0,1),current_word))
            {
                auto it = find(special_words.data(), special_words.data() + special_words.size(), current_word);

                const Index word_index = it - special_words.data();

                current_document(j) = special_words(word_index, 1);

                break;
            }

            if(starts_with(current_word, "'"))
            {
                current_word = current_word.substr(1);
            }

            if(starts_with(current_word, "y"))
            {
                current_word = "Y" + current_word.substr(1);
            }

            for(size_t k = 1; k < current_word.size(); k++)
            {
                if(contains(vowels, string(1,current_word[k-1]) ) && current_word[k] == 'y')
                {
                    current_word[k] = 'Y';
                }
            }

            Tensor<string,1> r1_r2(2);

            r1_r2 = get_r1_r2(current_word, vowels);

            bool step1a_vowel_found = false;
            bool step1b_vowel_found = false;

            // Step 0

            for(Index l = 0; l < step0_suffixes_size; l++)
            {
                const string current_suffix = step0_suffixes(l);

                if(ends_with(current_word,current_suffix))
                {
                    current_word = current_word.substr(0,current_word.length()-current_suffix.length());
                    r1_r2[0] = r1_r2[0].substr(0,r1_r2[0].length()-current_suffix.length());
                    r1_r2[1] = r1_r2[1].substr(0,r1_r2[1].length()-current_suffix.length());
                    break;
                }
            }

            // Step 1a

            for(size_t l = 0; l < step1a_suffixes_size; l++)
            {
                const string current_suffix = step1a_suffixes[l];

                if(ends_with(current_word, current_suffix))
                {
                    if(current_suffix == "sses")
                    {
                        current_word = current_word.substr(0,current_word.length()-2);
                        r1_r2[0] = r1_r2[0].substr(0,r1_r2[0].length()-2);
                        r1_r2[1] = r1_r2[1].substr(0,r1_r2[1].length()-2);
                    }
                    else if(current_suffix == "ied" || current_suffix == "ies")
                    {
                        if(current_word.length() - current_suffix.length() > 1)
                        {
                            current_word = current_word.substr(0,current_word.length()-2);
                            r1_r2[0] = r1_r2[0].substr(0,r1_r2[0].length()-2);
                            r1_r2[1] = r1_r2[1].substr(0,r1_r2[1].length()-2);
                        }
                        else
                        {
                            current_word = current_word.substr(0,current_word.length()-1);
                            r1_r2[0] = r1_r2[0].substr(0,r1_r2[0].length()-1);
                            r1_r2[1] = r1_r2[1].substr(0,r1_r2[1].length()-1);
                        }
                    }
                    else if(current_suffix == "s")
                    {
                        for(size_t l = 0; l < current_word.length() - 2; l++)
                        {
                            if(contains(vowels, string(1,current_word[l])))
                            {
                                step1a_vowel_found = true;
                                break;
                            }
                        }

                        if(step1a_vowel_found)
                        {
                            current_word = current_word.substr(0,current_word.length()-1);
                            r1_r2[0] = r1_r2[0].substr(0,r1_r2[0].length()-1);
                            r1_r2[1] = r1_r2[1].substr(0,r1_r2[1].length()-1);
                        }
                    }

                    break;
                }
            }

            // Step 1b

            for(Index k = 0; k < step1b_suffixes_size; k++)
            {
                const string current_suffix = step1b_suffixes[k];

                if(ends_with(current_word, current_suffix))
                {
                    if(current_suffix == "eed" || current_suffix == "eedly")
                    {
                        if(ends_with(r1_r2[0], current_suffix))
                        {
                            current_word = current_word.substr(0,current_word.length()-current_suffix.length()) + "ee";

                            if(r1_r2[0].length() >= current_suffix.length())
                            {
                                r1_r2[0] = r1_r2[0].substr(0,r1_r2[0].length()-current_suffix.length()) + "ee";
                            }
                            else
                            {
                                r1_r2[0] = "";
                            }

                            if(r1_r2[1].length() >= current_suffix.length())
                            {
                                r1_r2[1] = r1_r2[1].substr(0,r1_r2[1].length()-current_suffix.length()) + "ee";
                            }
                            else
                            {
                                r1_r2[1] = "";
                            }
                        }
                    }
                    else
                    {
                        for(size_t l = 0; l <(current_word.length() - current_suffix.length()); l++)
                        {
                            if(contains(vowels,string(1,current_word[l])))
                            {
                                step1b_vowel_found = true;
                                break;
                            }
                        }

                        if(step1b_vowel_found)
                        {
                            current_word = current_word.substr(0,current_word.length()-current_suffix.length());
                            r1_r2[0] = r1_r2[0].substr(0,r1_r2[0].length()-current_suffix.length());
                            r1_r2[1] = r1_r2[1].substr(0,r1_r2[1].length()-current_suffix.length());

                            if(ends_with(current_word, "at") || ends_with(current_word, "bl") || ends_with(current_word, "iz"))
                            {
                                current_word = current_word + "e";
                                r1_r2[0] = r1_r2[0] + "e";

                                if(current_word.length() > 5 || r1_r2[0].length() >= 3)
                                {
                                    r1_r2[1] = r1_r2[1] + "e";
                                }
                            }
                            else if(ends_with(current_word, double_consonants))
                            {
                                current_word = current_word.substr(0,current_word.length()-1);
                                r1_r2[0] = r1_r2[0].substr(0,r1_r2[0].length()-1);
                                r1_r2[1] = r1_r2[1].substr(0,r1_r2[1].length()-1);
                            }
                            else if((r1_r2[0] == "" && current_word.length() >= 3 && !contains(vowels,string(1,current_word[current_word.length()-1])) &&
                                      !(current_word[current_word.length()-1] == 'w' || current_word[current_word.length()-1] == 'x' || current_word[current_word.length()-1] == 'Y') &&
                                      contains(vowels,string(1,current_word[current_word.length()-2])) && !contains(vowels,string(1,current_word[current_word.length()-3]))) ||
                                     (r1_r2[0] == "" && current_word.length() == 2 && contains(vowels,string(1,current_word[0])) && contains(vowels, string(1,current_word[1]))))
                            {
                                current_word = current_word + "e";

                                if(r1_r2[0].length() > 0)
                                {
                                    r1_r2[0] = r1_r2[0] + "e";
                                }

                                if(r1_r2[1].length() > 0)
                                {
                                    r1_r2[1] = r1_r2[1] + "e";
                                }
                            }
                        }
                    }

                    break;
                }
            }

            // Step 1c

            if(current_word.length() > 2 &&(current_word[current_word.length()-1] == 'y' || current_word[current_word.length()-1] == 'Y') &&
                !contains(vowels, string(1,current_word[current_word.length()-2])))
            {
                current_word = current_word.substr(0,current_word.length()-1) + "i";

                if(r1_r2[0].length() >= 1)
                {
                    r1_r2[0] = r1_r2[0].substr(0,r1_r2[0].length()-1) + "i";
                }
                else
                {
                    r1_r2[0] = "";
                }

                if(r1_r2[1].length() >= 1)
                {
                    r1_r2[1] = r1_r2[1].substr(0,r1_r2[1].length()-1) + "i";
                }
                else
                {
                    r1_r2[1] = "";
                }
            }

            // Step 2

            for(Index l = 0; l < step2_suffixes_size; l++)
            {
                const string current_suffix = step2_suffixes[l];

                if(ends_with(current_word,current_suffix) && ends_with(r1_r2[0],current_suffix))
                {
                    if(current_suffix == "tional")
                    {
                        current_word = current_word.substr(0,current_word.length()-2);
                        r1_r2[0] = r1_r2[0].substr(0,r1_r2[0].length()-2);
                        r1_r2[1] = r1_r2[1].substr(0,r1_r2[1].length()-2);
                    }
                    else if(current_suffix == "enci" || current_suffix == "anci" || current_suffix == "abli")
                    {
                        current_word = current_word.substr(0,current_word.length()-1) + "e";

                        if(r1_r2[0].length() >= 1)
                        {
                            r1_r2[0] = r1_r2[0].substr(0,r1_r2[0].length()-1) + "e";
                        }
                        else
                        {
                            r1_r2[0] = "";
                        }

                        if(r1_r2[1].length() >= 1)
                        {
                            r1_r2[1] = r1_r2[1].substr(0,r1_r2[1].length()-1) + "e";
                        }
                        else
                        {
                            r1_r2[1] = "";
                        }
                    }
                    else if(current_suffix == "entli")
                    {
                        current_word = current_word.substr(0,current_word.length()-2);
                        r1_r2[0] = r1_r2[0].substr(0,r1_r2[0].length()-2);
                        r1_r2[1] = r1_r2[1].substr(0,r1_r2[1].length()-2);
                    }
                    else if(current_suffix == "izer" || current_suffix == "ization")
                    {
                        current_word = current_word.substr(0,current_word.length()-current_suffix.length()) + "ize";

                        if(r1_r2[0].length() >= current_suffix.length())
                        {
                            r1_r2[0] = r1_r2[0].substr(0,r1_r2[0].length()-current_suffix.length()) + "ize";
                        }
                        else
                        {
                            r1_r2[0] = "";
                        }

                        if(r1_r2[1].length() >= current_suffix.length())
                        {
                            r1_r2[1] = r1_r2[1].substr(0,r1_r2[1].length()-current_suffix.length()) + "ize";
                        }
                        else
                        {
                            r1_r2[1] = "";
                        }
                    }
                    else if(current_suffix == "ational" || current_suffix == "ation" || current_suffix == "ator")
                    {
                        current_word = current_word.substr(0,current_word.length()-current_suffix.length()) + "ate";

                        if(r1_r2[0].length() >= current_suffix.length())
                        {
                            r1_r2[0] = r1_r2[0].substr(0,r1_r2[0].length()-current_suffix.length()) + "ate";
                        }
                        else
                        {
                            r1_r2[0] = "";
                        }

                        if(r1_r2[1].length() >= current_suffix.length())
                        {
                            r1_r2[1] = r1_r2[1].substr(0,r1_r2[1].length()-current_suffix.length()) + "ate";
                        }
                        else
                        {
                            r1_r2[1] = "e";
                        }
                    }
                    else if(current_suffix == "alism" || current_suffix == "aliti" || current_suffix == "alli")
                    {
                        current_word = current_word.substr(0,current_word.length()-current_suffix.length()) + "al";

                        if(r1_r2[0].length() >= current_suffix.length())
                        {
                            r1_r2[0] = r1_r2[0].substr(0,r1_r2[0].length()-current_suffix.length()) + "al";
                        }
                        else
                        {
                            r1_r2[0] = "";
                        }

                        if(r1_r2[1].length() >= current_suffix.length())
                        {
                            r1_r2[1] = r1_r2[1].substr(0,r1_r2[1].length()-current_suffix.length()) + "al";
                        }
                        else
                        {
                            r1_r2[1] = "";
                        }
                    }
                    else if(current_suffix == "fulness")
                    {
                        current_word = current_word.substr(0,current_word.length()-4);
                        r1_r2[0] = r1_r2[0].substr(0,r1_r2[0].length()-4);
                        r1_r2[1] = r1_r2[1].substr(0,r1_r2[1].length()-4);
                    }
                    else if(current_suffix == "ousli" || current_suffix == "ousness")
                    {
                        current_word = current_word.substr(0,current_word.length()-current_suffix.length()) + "ous";

                        if(r1_r2[0].length() >= current_suffix.length())
                        {
                            r1_r2[0] = r1_r2[0].substr(0,r1_r2[0].length()-current_suffix.length()) + "ous";
                        }
                        else
                        {
                            r1_r2[0] = "";
                        }

                        if(r1_r2[1].length() >= current_suffix.length())
                        {
                            r1_r2[1] = r1_r2[1].substr(0,r1_r2[1].length()-current_suffix.length()) + "ous";
                        }
                        else
                        {
                            r1_r2[1] = "";
                        }
                    }
                    else if(current_suffix == "iveness" || current_suffix == "iviti")
                    {
                        current_word = current_word.substr(0,current_word.length()-current_suffix.length()) + "ive";

                        if(r1_r2[0].length() >= current_suffix.length())
                        {
                            r1_r2[0] = r1_r2[0].substr(0,r1_r2[0].length()-current_suffix.length()) + "ive";
                        }
                        else
                        {
                            r1_r2[0] = "";
                        }

                        if(r1_r2[1].length() >= current_suffix.length())
                        {
                            r1_r2[1] = r1_r2[1].substr(0,r1_r2[1].length()-current_suffix.length()) + "ive";
                        }
                        else
                        {
                            r1_r2[1] = "e";
                        }
                    }
                    else if(current_suffix == "biliti" || current_suffix == "bli")
                    {
                        current_word = current_word.substr(0,current_word.length()-current_suffix.length()) + "ble";

                        if(r1_r2[0].length() >= current_suffix.length())
                        {
                            r1_r2[0] = r1_r2[0].substr(0,r1_r2[0].length()-current_suffix.length()) + "ble";
                        }
                        else
                        {
                            r1_r2[0] = "";
                        }

                        if(r1_r2[1].length() >= current_suffix.length())
                        {
                            r1_r2[1] = r1_r2[1].substr(0,r1_r2[1].length()-current_suffix.length()) + "ble";
                        }
                        else
                        {
                            r1_r2[1] = "";
                        }
                    }
                    else if(current_suffix == "ogi" && current_word[current_word.length()-4] == 'l')
                    {
                        current_word = current_word.substr(0,current_word.length()-1);
                        r1_r2[0] = r1_r2[0].substr(0,r1_r2[0].length()-1);
                        r1_r2[1] = r1_r2[1].substr(0,r1_r2[1].length()-1);
                    }
                    else if(current_suffix == "fulli" || current_suffix == "lessli")
                    {
                        current_word = current_word.substr(0,current_word.length()-2);
                        r1_r2[0] = r1_r2[0].substr(0,r1_r2[0].length()-2);
                        r1_r2[1] = r1_r2[1].substr(0,r1_r2[1].length()-2);
                    }
                    else if(current_suffix == "li" && contains(li_ending, string(1,current_word[current_word.length()-4])))
                    {
                        current_word = current_word.substr(0,current_word.length()-2);
                        r1_r2[0] = r1_r2[0].substr(0,r1_r2[0].length()-2);
                        r1_r2[1] = r1_r2[1].substr(0,r1_r2[1].length()-2);
                    }

                    break;
                }
            }

            // Step 3

            for(Index l = 0; l < step3_suffixes_size; l++)
            {
                const string current_suffix = step3_suffixes[l];

                if(ends_with(current_word,current_suffix) && ends_with(r1_r2[0],current_suffix))
                {
                    if(current_suffix == "tional")
                    {
                        current_word = current_word.substr(0,current_word.length()-2);
                        r1_r2[0] = r1_r2[0].substr(0,r1_r2[0].length()-2);
                        r1_r2[1] = r1_r2[1].substr(0,r1_r2[1].length()-2);
                    }
                    else if(current_suffix == "ational")
                    {
                        current_word = current_word.substr(0,current_word.length()-current_suffix.length()) + "ate";

                        if(r1_r2[0].length() >= current_suffix.length())
                        {
                            r1_r2[0] = r1_r2[0].substr(0,r1_r2[0].length()-current_suffix.length()) + "ate";
                        }
                        else
                        {
                            r1_r2[0] = "";
                        }

                        if(r1_r2[1].length() >= current_suffix.length())
                        {
                            r1_r2[1] = r1_r2[1].substr(0,r1_r2[1].length()-current_suffix.length()) + "ate";
                        }
                        else
                        {
                            r1_r2[1] = "";
                        }
                    }
                    else if(current_suffix == "alize")
                    {
                        current_word = current_word.substr(0,current_word.length()-3);
                        r1_r2[0] = r1_r2[0].substr(0,r1_r2[0].length()-3);
                        r1_r2[1] = r1_r2[1].substr(0,r1_r2[1].length()-3);
                    }
                    else if(current_suffix == "icate" || current_suffix == "iciti" || current_suffix == "ical")
                    {
                        current_word = current_word.substr(0,current_word.length()-current_suffix.length()) + "ic";

                        if(r1_r2[0].length() >= current_suffix.length())
                        {
                            r1_r2[0] = r1_r2[0].substr(0,r1_r2[0].length()-current_suffix.length()) + "ic";
                        }
                        else
                        {
                            r1_r2[0] = "";
                        }

                        if(r1_r2[1].length() >= current_suffix.length())
                        {
                            r1_r2[1] = r1_r2[1].substr(0,r1_r2[1].length()-current_suffix.length()) + "ic";
                        }
                        else
                        {
                            r1_r2[1] = "";
                        }
                    }
                    else if(current_suffix == "ful" || current_suffix == "ness")
                    {
                        current_word = current_word.substr(0,current_word.length()-current_suffix.length());
                        r1_r2[0] = r1_r2[0].substr(0,r1_r2[0].length()-current_suffix.length());
                        r1_r2[1] = r1_r2[1].substr(0,r1_r2[1].length()-current_suffix.length());
                    }
                    else if(current_suffix == "ative" && ends_with(r1_r2[1],current_suffix))
                    {
                        current_word = current_word.substr(0,current_word.length()-5);
                        r1_r2[0] = r1_r2[0].substr(0,r1_r2[0].length()-5);
                        r1_r2[1] = r1_r2[1].substr(0,r1_r2[1].length()-5);
                    }

                    break;
                }
            }

            // Step 4

            for(Index l = 0; l < step4_suffixes_size; l++)
            {
                const string current_suffix = step4_suffixes[l];

                if(ends_with(current_word,current_suffix) && ends_with(r1_r2[1],current_suffix))
                {
                    if(current_suffix == "ion" &&(current_word[current_word.length()-4] == 's' || current_word[current_word.length()-4] == 't'))
                    {
                        current_word = current_word.substr(0,current_word.length()-3);
                        r1_r2[0] = r1_r2[0].substr(0,r1_r2[0].length()-3);
                        r1_r2[1] = r1_r2[1].substr(0,r1_r2[1].length()-3);
                    }
                    else
                    {
                        current_word = current_word.substr(0,current_word.length()-current_suffix.length());
                        r1_r2[0] = r1_r2[0].substr(0,r1_r2[0].length()-current_suffix.length());
                        r1_r2[1] = r1_r2[1].substr(0,r1_r2[1].length()-current_suffix.length());
                    }

                    break;
                }
            }

            // Step 5

            if(r1_r2[1][r1_r2[1].length()-1] == 'l' && current_word[current_word.length()-2] == 'l')
            {
                current_word = current_word.substr(0,current_word.length()-1);
            }
            else if(r1_r2[1][r1_r2[1].length()-1] == 'e')
            {
                current_word = current_word.substr(0,current_word.length()-1);
            }
            else if(r1_r2[0][r1_r2[0].length()-1] == 'e')
            {
                if(current_word.length() >= 4 &&(contains(vowels, string(1,current_word[current_word.length()-2])) ||
                                                   (current_word[current_word.length()-2] == 'w' || current_word[current_word.length()-2] == 'x' ||
                                                    current_word[current_word.length()-2] == 'Y') || !contains(vowels, string(1,current_word[current_word.length()-3])) ||
                                                   contains(vowels, string(1,current_word[current_word.length()-4]))))
                {
                    current_word = current_word.substr(0,current_word.length()-1);
                }
            }

            replace(current_word,"Y","y");
            current_document(j) = current_word;
        }
        new_tokenized_documents(i) = current_document;
    }

    return new_tokenized_documents;
}


void TextDataSet::read_txt()
{
    cout << "Reading .txt file..." << endl;

    if(data_source_path.empty())
        throw runtime_error("Text file is empty");

    ifstream file(data_source_path.c_str());

    if(!file.is_open())
        throw runtime_error("Cannot open text file: " + data_source_path + "\n");

    const string separator = get_separator_string();

    Index lines_count = 0;

    string line;

    while(file.good())
    {
        getline(file, line);
        trim(line);
        erase(line, '"');

        if(line.empty()) continue;

        lines_count++;

        if(file.peek() == EOF) break;
    }

    cout << lines_count << endl;

    file.seekg (0, ios::beg);

    Tensor<string, 1> documents(lines_count);
    Tensor<string, 1> targets(lines_count);

    Index index = 0;

    while(file.good())
    {
        getline(file, line);
        trim(line);
        erase(line, '"');

        if(line.empty()) continue;

        const Index tokens_number = count_tokens(line, separator);

        if(tokens_number != 2)
            throw runtime_error("More than one separator in line: " + line + "\n");

        const Tensor<string, 1> line_tokens = get_tokens(line, separator);

        documents(index) = line_tokens(0);
        targets(index) = line_tokens(1);

        index++;

        if(file.peek() == EOF) break;
    }

    file.close();

    to_lower(documents);
    delete_punctuation(documents);
    delete_non_printable_chars(documents);
    delete_extra_spaces(documents);
    delete_non_alphanumeric(documents);

    cout << documents << endl;

    Tensor<Tensor<string,1>,1> documents_words = get_tokens(documents, " ");

    delete_words(documents_words, stop_words);

    delete_short_long_words(documents_words, short_words_length, long_words_length);

    //replace_accented(documents_words);

    //delete_emails(documents_words);

    //delete_numbers(documents_words);

    //delete_blanks(documents_words);

    /*
    documents_words = stem(documents_words);
    */

    print_tokens(documents_words);
/*
    Tensor<Tensor<string, 1>, 1> targets;
    to_lower(targets);


    cout << "Processing documents..." << endl;

    //const Tensor<Tensor<string, 1>, 1> document_tokens = preprocess(word_list);

    cout << "Calculating wordbag..." << endl;

    //const WordBag word_bag = calculate_word_bag(document_tokens);
    set_words_frequencies(word_bag.frequencies);

    const Tensor<string, 1> raw_variables_names = word_bag.words;
    const Index raw_variables_number = word_bag.size();

    Tensor<type, 1> row(raw_variables   _number);

    // Output

    cout << "Writting data file..." << endl;

    string new_data_source_path = data_source_path;
    replace(new_data_source_path, ".txt", "_data.txt");
    replace(new_data_source_path, ".csv", "_data.csv");

    ofstream file;
    file.open(new_data_source_path);

    for(Index i  = 0; i < raw_variables_number; i++)
        file << raw_variables_names(i) << ";";

    file << "target_variable" << "\n";

    // Data file preview

    const Index preview_size = 4;

    text_data_file_preview.resize(preview_size, 2);

    for(Index i = 0; i < preview_size - 1; i++)
    {
        text_data_file_preview(i, 0) = documents(0)(i);
        text_data_file_preview(i, 1) = targets(0)(i);
    }

    text_data_file_preview(preview_size - 1, 0) = documents(0)(documents(0).size() - 1);
    text_data_file_preview(preview_size - 1, 1) = targets(0)(targets(0).size() - 1);

#pragma omp parallel for

    for(Index i = 0; i < documents_number; i++)
    {
        const Tensor<string, 1> document = documents(i);

        const Index tokens_number = document.size();

        for(Index j = 0; j < tokens_number; j++)
        {
            row.setZero();

//            const string line = sheet(j);

//            const Tensor<string,1> tokens = get_tokens(line);

//            const Tensor<Tensor<string, 1>, 1> processed_tokens = preprocess(tokens);

//            const WordBag line_word_bag = calculate_word_bag(processed_tokens);

//            const Tensor<string, 1> line_words = line_word_bag.words;

//            const Tensor<Index, 1> line_frequencies = line_word_bag.frequencies;

//            for(Index k = 0; k < line_words.size(); k++)
            {

                if( contains(raw_variables_names, line_words(k)) )
                {
                    auto it = find(raw_variables_names.data(), raw_variables_names.data() + document_words_number, line_words(k));

                    const Index word_index = it - raw_variables_names.data();

                    row(word_index) = type(line_frequencies(k));
                }

            }

//            for(Index k = 0; k < document_words_number; k++)
//                file << row(k) << ";";

            file << "target_" + targets(i)(j) << "\n";
        }

    }

    file.close();

    data_source_path = new_data_source_path;
    separator = Separator::Semicolon;
    has_header = true;

    load_data(); 

    for(Index i = 0; i < get_input_raw_variables_number(); i++)
        set_raw_variable_type(i, RawVariableType::Numeric);
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
