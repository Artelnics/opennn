//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   L A N G U A G E  D A T A S E T   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "language_data_set.h"
#include "strings_utilities.h"

namespace opennn
{

LanguageDataSet::LanguageDataSet(const dimensions& new_input_dimensionms,
                                 const dimensions& new_target_dimensionms)
{
}


LanguageDataSet::LanguageDataSet(const filesystem::path& new_data_path) : DataSet()
{
    data_path = new_data_path;
    separator = DataSet::Separator::Tab;

    read_csv();
    set_raw_variable_scalers(Scaler::None);

   target_dimensions = {get_target_length()};
   decoder_dimensions = {get_target_length()};
   input_dimensions = {get_input_length()};
}


const unordered_map<string, Index>& LanguageDataSet::get_input_vocabulary() const
{
    return input_vocabulary;
}


const unordered_map<string, Index>& LanguageDataSet::get_target_vocabulary() const
{
    return target_vocabulary;
}


Index LanguageDataSet::get_input_vocabulary_size() const
{
    return input_vocabulary.size();
}

Index LanguageDataSet::get_input_length() const
{
    return maximum_input_length;
}

Index LanguageDataSet::get_target_length() const
{
    return maximum_target_length;
}

Index LanguageDataSet::get_target_vocabulary_size() const
{
    return target_vocabulary.size();
}


void LanguageDataSet::set_input_vocabulary(const unordered_map<string, Index>& new_input_vocabulary)
{
    input_vocabulary = new_input_vocabulary;
}


void LanguageDataSet::set_target_vocabulary(const unordered_map<string, Index>& new_target_vocabulary)
{
    target_vocabulary = new_target_vocabulary;
}


void LanguageDataSet::set_data_random()
{
/*

    for(Index i = 0; i < batch_samples_number; i++)
    {
        for(Index j = 0; j < decoder_length; j++)
            data(i, j) = type(rand() % context_dimension);

        for(Index j = 0; j < 2 * completion_length; j++)
            data(i, j + decoder_length) = type(rand() % completion_dimension);
    }

    for(Index i = 0; i < decoder_length; i++)
        set_raw_variable_use(i, DataSet::VariableUse::Decoder);

    for(Index i = 0; i < completion_length; i++)
        set_raw_variable_use(i + decoder_length, DataSet::VariableUse::Input);

    for(Index i = 0; i < completion_length; i++)
        set_raw_variable_use(i + decoder_length + completion_length, DataSet::VariableUse::Target);
*/
}


void LanguageDataSet::to_XML(XMLPrinter& printer) const
{
    ostringstream buffer;

    time_t start, finish;
    time(&start);

    printer.OpenElement("DataSet");

    // Data file

    printer.OpenElement("DataSource");

    add_xml_element(printer, "FileType", "csv");
    add_xml_element(printer, "DataPath", data_path.string());
    add_xml_element(printer, "Separator", get_separator_string());
    add_xml_element(printer, "HasHeader", to_string(has_header));
    add_xml_element(printer, "HasSamplesId", to_string(has_sample_ids));
    add_xml_element(printer, "MissingValuesLabel", missing_values_label);
    add_xml_element(printer, "Codification", get_codification_string());
    printer.CloseElement();

    const Index raw_variables_number = get_raw_variables_number();

    printer.OpenElement("RawVariables");
    add_xml_element(printer, "RawVariablesNumber", to_string(raw_variables_number));

    for(Index i = 0; i < raw_variables_number; i++)
    {
        printer.OpenElement("RawVariable");
        printer.PushAttribute("Item", to_string(i+1).c_str());
        raw_variables[i].to_XML(printer);
        printer.CloseElement();
    }

    printer.CloseElement();

    // Samples id

    if(has_sample_ids)
        add_xml_element(printer, "SampleIds", vector_to_string(sample_ids));

    // Samples

    printer.OpenElement("Samples");
    add_xml_element(printer, "SamplesNumber", to_string(get_samples_number()));
    add_xml_element(printer, "SampleUses", vector_to_string(get_sample_uses_vector()));
    printer.CloseElement();

    printer.OpenElement("MissingValues");
    add_xml_element(printer, "MissingValuesMethod", get_missing_values_method_string());
    add_xml_element(printer, "MissingValuesNumber", to_string(missing_values_number));

    if (missing_values_number > 0)
    {
        add_xml_element(printer, "RawVariablesMissingValuesNumber", tensor_to_string(raw_variables_missing_values_number));
        add_xml_element(printer, "RowsMissingValuesNumber", to_string(rows_missing_values_number));
    }

    printer.CloseElement();

    printer.OpenElement("MissingValues");
    add_xml_element(printer, "MissingValuesMethod", get_missing_values_method_string());
    add_xml_element(printer, "MissingValuesNumber", to_string(missing_values_number));

    if (missing_values_number > 0)
    {
        add_xml_element(printer, "RawVariablesMissingValuesNumber", tensor_to_string(raw_variables_missing_values_number));
        add_xml_element(printer, "RowsMissingValuesNumber", to_string(rows_missing_values_number));
    }

    printer.CloseElement();

    // Completion Vocabulary

    printer.OpenElement("TargetVocabulary");

//    for (const auto& word : target_vocabulary)
//        add_xml_element(printer, "Word", word);

    printer.CloseElement();

    // Decoder Vocabulary
    printer.OpenElement("DecoderVocabulary");

//    for (const auto& word : input_vocabulary)
//        add_xml_element(printer, "Word", word);

    printer.CloseElement();

    add_xml_element(printer, "TargetDimensions", to_string(maximum_target_length));
    add_xml_element(printer, "targetDimensions", to_string(maximum_input_length));

    printer.CloseElement();

    time(&finish);
}


vector<string> LanguageDataSet::tokenize(const string& document, const bool& input)
{
    vector<string> tokens;

    // if(!input)
    tokens.push_back("[START]");

    string currentToken;

    for (char c : document)
    {
        if (isalnum(c))
        {
            // Add alphanumeric characters to the current token
            currentToken += tolower(c);
        }
        else
        {
            // If the current token is not empty, add it to the tokens list
            if (!currentToken.empty())
            {
                tokens.push_back(currentToken);
                currentToken.clear();
            }
            // Treat punctuation as a separate token

            if (ispunct(c))
            {
                tokens.push_back(string(1, c));
            }
            else if (isspace(c))
            {
                // Ignore spaces, they just delimit tokens
            }
        }
    }

    // Add the last token if it's not empty
    if (!currentToken.empty())
        tokens.push_back(currentToken);

    // Add [END] token
    // if(!input)
    tokens.push_back("[END]");

    // if(tokens.size() == 3 || tokens.size() == 1)
    //     tokens

    return tokens;
}


unordered_map<string, Index> LanguageDataSet::create_vocabulary(const vector<vector<string>>& document_tokens)
{
    unordered_map<string, Index> vocabulary;
    Index id = 0;

    vocabulary["[PAD]"] = id++;
    vocabulary["[UNK]"] = id++;
    vocabulary["[START]"] = id++;
    vocabulary["[END]"] = id++;

    for (const auto& document : document_tokens)
        for (const auto& token : document)
            if (vocabulary.find(token) == vocabulary.end())
                vocabulary[token] = id++;

    return vocabulary;
}


void LanguageDataSet::print_vocabulary(const unordered_map<string, Index>& vocabulary)
{
    for (const auto& entry : vocabulary)
        cout << entry.first << " : " << entry.second << "\n";
}


void LanguageDataSet::print() const
{
    if(has_decoder)
    {
        cout << "Language data set" << endl;

        cout << "Input vocabulary size: " << get_input_vocabulary_size() << endl;
        cout << "Target vocabulary size: " << get_target_vocabulary_size() << endl;

        cout << "Input length: " << get_input_length() << endl;
        cerr << "Target length: " << get_target_length() << endl;
    }
    else
    {
        cout << "Language data set" << endl;

        cout << "Input vocabulary size: " << get_input_vocabulary_size() << endl;
        cout << "Target size: 1" << endl;

        cout << "Input lenght: " << get_input_length() << endl;
        cout << "Target categories: 0, 1"<<endl;
    }
}


void LanguageDataSet::from_XML(const XMLDocument& data_set_document)
{
    const XMLElement* data_set_element = data_set_document.FirstChildElement("DataSet");

    if(!data_set_element)
        throw runtime_error("Data set element is nullptr.\n");

    const XMLElement* data_source_element = data_set_element->FirstChildElement("DataSource");

    if(!data_source_element)
        throw runtime_error("Data file element is nullptr.\n");

    set_data_path(read_xml_string(data_source_element, "Path"));
    set_separator_string(read_xml_string(data_source_element, "Separator"));
    set_missing_values_label(read_xml_string(data_source_element, "MissingValuesLabel"));
    set_codification(read_xml_string(data_source_element, "Codification"));
    set_has_header(read_xml_bool(data_source_element, "HasHeader"));
    set_has_ids(read_xml_bool(data_source_element, "HasSamplesId"));

    // RawVariables

    const XMLElement* raw_variables_element = data_set_element->FirstChildElement("RawVariables");

    if(!raw_variables_element)
        throw runtime_error("RawVariables element is nullptr.\n");

    // Raw variables number

    const XMLElement* raw_variables_number_element = raw_variables_element->FirstChildElement("RawVariablesNumber");

    if(!raw_variables_number_element)
        throw runtime_error("RawVariablesNumber element is nullptr.\n");

    Index new_raw_variables_number = 0;

    if(raw_variables_number_element->GetText())
    {
        new_raw_variables_number = Index(atoi(raw_variables_number_element->GetText()));

        set_raw_variables_number(new_raw_variables_number);
    }

    // Raw variables

    const XMLElement* start_element = raw_variables_number_element;

    for(Index i = 0; i < new_raw_variables_number; i++)
    {
        const XMLElement* raw_variable_element = start_element->NextSiblingElement("RawVariable");
        start_element = raw_variable_element;

        if(raw_variable_element->Attribute("Item") != to_string(i+1))
            throw runtime_error("Raw_variable item number (" + to_string(i + 1) + ") exception.\n");

        RawVariable& raw_variable = raw_variables[i];
        raw_variable.name = read_xml_string(raw_variable_element, "Name");
        raw_variable.set_scaler(read_xml_string(raw_variable_element, "Scaler"));
        raw_variable.set_use(read_xml_string(raw_variable_element, "Use"));
        raw_variable.set_type(read_xml_string(raw_variable_element, "Type"));

        if(raw_variables[i].type == RawVariableType::Categorical || raw_variables[i].type == RawVariableType::Binary)
            raw_variable.categories = get_tokens(read_xml_string(raw_variable_element, "Categories"), ";");

        //        raw_variable_element = raw_variable_element->NextSiblingElement("RawVariable");
    }

    // Rows label

    if(has_sample_ids)
        sample_ids = get_tokens(read_xml_string(data_set_element, "SamplesId"), ",");

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

        const XMLElement* raw_variables_missing_values_number_element = missing_values_element->FirstChildElement("");

        if(!raw_variables_missing_values_number_element)
            throw runtime_error("RawVariablesMissingValuesNumber element is nullptr.\n");

        if(raw_variables_missing_values_number_element->GetText())
        {
            const vector<string> new_raw_variables_missing_values_number
                = get_tokens(raw_variables_missing_values_number_element->GetText(), " ");

            raw_variables_missing_values_number.resize(new_raw_variables_missing_values_number.size());

            for(size_t i = 0; i < new_raw_variables_missing_values_number.size(); i++)
                raw_variables_missing_values_number(i) = atoi(new_raw_variables_missing_values_number[i].c_str());
        }

        // Rows missing values number

        const XMLElement* rows_missing_values_number_element = missing_values_element->FirstChildElement("RowsMissingValuesNumber");

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

        if(new_preview_size > 0)
            data_file_preview.resize(new_preview_size);
    }

    // Preview data

    start_element = preview_size_element;

    if(model_type != ModelType::TextClassification)
    {
        for(Index i = 0; i < new_preview_size; i++)
        {
            const XMLElement* row_element = start_element->NextSiblingElement("Row");
            start_element = row_element;

            if(row_element->Attribute("Item") != to_string(i+1))
                throw runtime_error("Row item number (" + to_string(i + 1) + ") exception.\n");

            if(row_element->GetText())
                data_file_preview[i] = get_tokens(row_element->GetText(), ",");
        }
    }
    else
    {
        for(Index i = 0; i < new_preview_size; i++)
        {
            const XMLElement* row_element = start_element->NextSiblingElement("Row");
            start_element = row_element;

            if(row_element->Attribute("Item") != to_string(i+1))
                throw runtime_error("Row item number (" + to_string(i + 1) + ") exception.\n");

            if(row_element->GetText())
                data_file_preview[i][0] = row_element->GetText();
        }

        for(Index i = 0; i < new_preview_size; i++)
        {
            const XMLElement* row_element = start_element->NextSiblingElement("Target");
            start_element = row_element;

            if(row_element->Attribute("Item") != to_string(i+1))
                throw runtime_error("Target item number (" + to_string(i + 1) + ") exception.\n");

            if(row_element->GetText())
                data_file_preview[i][1] = row_element->GetText();
        }
    }

    // Completion Vocabulary

    const XMLElement* completion_vocabulary_element = data_set_element->FirstChildElement("CompletionVocabulary");
    if(completion_vocabulary_element)
    {
        target_vocabulary.clear();
        for (const XMLElement* word_element = completion_vocabulary_element->FirstChildElement(); word_element; word_element = word_element->NextSiblingElement())
        {
//            if (word_element->GetText())
//                target_vocabulary.push_back(word_element->GetText());
        }
    }

    // Decoder Vocabulary

    const XMLElement* context_vocabulary_element = data_set_element->FirstChildElement("DecoderVocabulary");
    if(context_vocabulary_element)
    {
        input_vocabulary.clear();

        for (const XMLElement* word_element = context_vocabulary_element->FirstChildElement(); word_element; word_element = word_element->NextSiblingElement())
        {
//            if (word_element->GetText())
//                input_vocabulary.push_back(word_element->GetText());
        }
    }

    // Decoder Dimensions

    const XMLElement* completion_dimensions_element = data_set_element->FirstChildElement("CompletionDimensions");
    if (completion_dimensions_element && completion_dimensions_element->GetText())
        maximum_target_length = atoi(completion_dimensions_element->GetText());

    // Decoder Dimensions

    const XMLElement* decoder_dimensions_element = data_set_element->FirstChildElement("DecoderDimensions");
    if (decoder_dimensions_element && decoder_dimensions_element->GetText())
        maximum_input_length = atoi(decoder_dimensions_element->GetText());

    // Display

    // set_display(read_xml_bool(data_set_element, "Display"));
}


Index LanguageDataSet::count_non_empty_lines() const
{
    ifstream file(data_path);

    if (!file.is_open())
        throw runtime_error("Cannot open file: " + data_path.string() + "\n");

    Index count = 0;

    string line;

    while (getline(file, line))
    {
        prepare_line(line);

        if (!line.empty())
            count++;
    }

    return count;
}


void LanguageDataSet::read_csv()
{
    cout << "Reading .txt file..." << endl;

    const Index samples_number = count_non_empty_lines();

    const vector<string> positive_words = { "yes", "positive", "+", "true", "1"};

    const vector<string> negative_words = { "no", "negative", "-", "false", "0"};

    ifstream file(data_path);

    if (!file.is_open())
        throw runtime_error("Cannot open data file: " + data_path.string() + "\n");

    if(data_path.extension() == ".csv")
        separator = Separator::Semicolon;

    string line;

    vector<string> tokens;

    vector<vector<string>> input_documents_tokens(samples_number);
    vector<vector<string>> target_documents_tokens(samples_number);

    const string separator_string = get_separator_string();

    Index sample_index = 0;

    while (getline(file, line))
    {
        if (line.empty()) continue;

        tokens = get_tokens(line, separator_string);

        if (tokens.size() != 2)
            throw runtime_error("Tokens number must be two.");

        input_documents_tokens[sample_index] = tokenize(tokens[0], true);

        target_documents_tokens[sample_index] = tokenize(tokens[1], false);

        sample_index++;
    }

    if (sample_index != samples_number)
        throw runtime_error("WARNING: Expected " + to_string(samples_number) + " samples, but " + to_string(sample_index) + " were processed.");

    maximum_input_length = get_maximum_size(input_documents_tokens);
    maximum_target_length = get_maximum_size(target_documents_tokens);

    input_vocabulary = create_vocabulary(input_documents_tokens);
    target_vocabulary = create_vocabulary(target_documents_tokens);

    has_decoder = target_vocabulary.size() == 6 ? false
                                                : true;
    
    input_vocabulary_size = get_input_vocabulary_size();
    target_vocabulary_size = get_target_vocabulary_size();

    const Index input_variables_number = maximum_input_length;
    const Index decoder_variables_number = has_decoder ? maximum_target_length - 1
                                                       : 0;
    const Index target_variables_number = has_decoder ? maximum_target_length - 1
                                                      : 1;
    const Index variables_number = input_variables_number + decoder_variables_number + target_variables_number;

    data.resize(samples_number, variables_number);
    data.setZero();

    raw_variables.resize(variables_number);

    for(Index i = 0; i < input_variables_number; i++)
        set_raw_variable_use(i, VariableUse::Input);

    for (Index i = input_variables_number; i < input_variables_number + decoder_variables_number; i++)
        set_raw_variable_use(i, VariableUse::Decoder);

    for (Index i = input_variables_number + decoder_variables_number; i < variables_number; i++)
        set_raw_variable_use(i, VariableUse::Target);

    Index column_index = 0;

    for (Index i = 0; i < samples_number; i++)
    {
        column_index = 0;

        const vector<string>& input_document_tokens = input_documents_tokens[i];
        const vector<string>& target_document_tokens = target_documents_tokens[i];

        // Input data

        for (Index j = 0; j < Index(input_document_tokens.size()); j++)
        {
            const auto iterator = input_vocabulary.find(input_document_tokens[j]);

            iterator != input_vocabulary.end()
                ? data(i, column_index++) = iterator->second
                : data(i, column_index++) = 1;
        }

        if(column_index < input_variables_number)
            column_index = input_variables_number;

        if(has_decoder)
        {
            // Decoder data

            for (Index j = 0; j < Index(target_document_tokens.size()); j++)
            {
                const auto iterator = target_vocabulary.find(target_document_tokens[j]);

                if(iterator->second == 3)
                    continue;

                iterator != target_vocabulary.end() && iterator->second != 3 // [END]
                    ? data(i, column_index++) = iterator->second
                    : data(i,column_index++) = 1;
            }

            if(column_index < input_variables_number + decoder_variables_number)
                column_index = input_variables_number + decoder_variables_number;

            // Target data

            for (Index j = 0; j < Index(target_document_tokens.size()); j++)
            {
                const auto iterator = target_vocabulary.find(target_document_tokens[j]);

                if(iterator->second == 2)
                    continue;

                iterator != target_vocabulary.end() && iterator->second != 2// [START]
                    ? data(i, column_index++) = iterator->second
                    : data(i,column_index++) = 1;
            }
        }
        else
        {
            // Target data

            for (Index j = 0; j < Index(target_document_tokens.size()); j++)
            {
                const auto iterator = target_vocabulary.find(target_document_tokens[j]);

                if(iterator->second == 2||iterator->second == 3)
                    continue;

                iterator != target_vocabulary.end() && contains(negative_words, iterator->first)
                    ? data(i, column_index) = 0
                    : data(i,column_index) = 1;
            }
        }
    }

    sample_uses.resize(samples_number);

    set_default_raw_variable_names();
    split_samples_random();
    set_binary_raw_variables();
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
