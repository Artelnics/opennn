//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   L A N G U A G E  D A T A S E T   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "language_dataset.h"
#include "strings_utilities.h"
#include "tensors.h"
#include "tinyxml2.h"

namespace opennn
{

LanguageDataset::LanguageDataset(const filesystem::path& new_data_path) : Dataset()
{
    data_path = new_data_path;
    separator = Dataset::Separator::Tab;

    if(!data_path.empty())
        read_csv();
}


LanguageDataset::LanguageDataset(const Index& samples_number,
                                 const Index& input_sequence_length,
                                 const Index& input_vocabulary_size) : Dataset()
{
    maximum_input_length = input_sequence_length;
    maximum_target_length = 1;

    const Index variables_number = input_sequence_length + 1;

    data.resize(samples_number, variables_number);
    raw_variables.resize(variables_number);

    set_default();

    for (Index i = 0; i < variables_number; i++)
    {
        RawVariable& raw_variable = raw_variables[i];

        raw_variable.type = RawVariableType::Numeric;
        raw_variable.name = "variable_" + to_string(i + 1);

        raw_variable.role = (i < input_sequence_length)
                               ? "Input"
                               : "Target";
    }

    sample_uses.resize(samples_number);
    split_samples_random();

    default_random_engine generator;
    uniform_int_distribution<int> dist_main(0, input_vocabulary_size - 1);
    uniform_int_distribution<int> dist_binary(0, 1);

    const Index target_column_index = data.dimension(1) - 1;

    for (Index i = 0; i < data.dimension(0); ++i)
    {
        for (Index j = 0; j < target_column_index; ++j)
            data(i, j) = static_cast<type>(dist_main(generator));

        data(i, target_column_index) = static_cast<type>(dist_binary(generator));
    }

    input_vocabulary.resize(input_vocabulary_size + reserved_tokens.size());
    target_vocabulary.resize(2);

    input_dimensions = { get_input_sequence_length() };
    target_dimensions = { get_target_sequence_length() };
    decoder_dimensions = { 0 };

    set_raw_variable_scalers("None");
    set_default_raw_variable_names();
    set_binary_raw_variables();
}


const vector<string>& LanguageDataset::get_input_vocabulary() const
{
    return input_vocabulary;
}


const vector<string>& LanguageDataset::get_target_vocabulary() const
{
    return target_vocabulary;
}


Index LanguageDataset::get_input_vocabulary_size() const
{
    return input_vocabulary.size();
}


Index LanguageDataset::get_input_sequence_length() const
{
    return maximum_input_length;
}


Index LanguageDataset::get_target_sequence_length() const
{
    return maximum_target_length;
}


Index LanguageDataset::get_target_vocabulary_size() const
{
    return target_vocabulary.size();
}


void LanguageDataset::set_input_vocabulary(const vector<string>& new_input_vocabulary)
{
    input_vocabulary = new_input_vocabulary;
}


void LanguageDataset::set_target_vocabulary(const vector<string>& new_target_vocabulary)
{
    target_vocabulary = new_target_vocabulary;
}


void LanguageDataset::to_XML(XMLPrinter& printer) const
{
    ostringstream buffer;

    time_t start, finish;
    time(&start);

    printer.OpenElement("Dataset");

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
        add_xml_element(printer, "RawVariablesMissingValuesNumber", tensor_to_string<Index, 1>(raw_variables_missing_values_number));
        add_xml_element(printer, "RowsMissingValuesNumber", to_string(rows_missing_values_number));
    }

    printer.CloseElement();

    printer.OpenElement("MissingValues");
    add_xml_element(printer, "MissingValuesMethod", get_missing_values_method_string());
    add_xml_element(printer, "MissingValuesNumber", to_string(missing_values_number));

    if (missing_values_number > 0)
    {
        add_xml_element(printer, "RawVariablesMissingValuesNumber", tensor_to_string<Index, 1>(raw_variables_missing_values_number));
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


void LanguageDataset::create_vocabulary(const vector<vector<string>>& document_tokens,
                                        vector<string>& vocabulary) const
{
    unordered_map<string, size_t> word_counts;

    for (const auto& document : document_tokens)
        for (const auto& token : document)
            ++word_counts[token];

    vector<pair<string, size_t>> sorted_words(word_counts.begin(), word_counts.end());

    sort(sorted_words.begin(), sorted_words.end(),
         [](const pair<string, size_t>& a, const pair<string, size_t>& b)
         {
             return a.second > b.second;
         });

    vocabulary.clear();

    for (const auto& token : reserved_tokens)
        vocabulary.push_back(token);

    for (const auto& entry : sorted_words)
    {
        // if (entry.second < size_t(minimum_word_frequency)) continue;
        // if (find(reserved_tokens.begin(), reserved_tokens.end(), entry.first) != reserved_tokens.end()) continue;
        // if (vocabulary.size() >= size_t(maximum_vocabulary_size)) break;

        vocabulary.push_back(entry.first);
    }
}


void LanguageDataset::encode_input_data(const vector<vector<string>>& input_document_tokens)
{
    unordered_map<string, Index> input_vocabulary_map;

    for (size_t i = 0; i < input_vocabulary.size(); ++i)
        input_vocabulary_map[input_vocabulary[i]] = i;

    const Index samples_number = get_samples_number();

    #pragma omp parallel for

    for (Index sample = 0; sample < samples_number; sample++)
    {
        data(sample, 0) = 2; // start

        const vector<string>& input_tokens = input_document_tokens[sample];

        for (Index variable = 0; variable < Index(input_tokens.size()); variable++)
        {
            auto it = find(input_vocabulary.begin(), input_vocabulary.end(), input_tokens[variable]);

            data(sample, 1+variable) = (it != input_vocabulary.end()) ? it - input_vocabulary.begin() : 1;
        }

        data(sample, input_tokens.size() + 1) = 3; // end
    }
}


void LanguageDataset::encode_target_data(const vector<vector<string>>& target_document_tokens)
{
    if(maximum_target_length == 1 && target_vocabulary.size() < 6)
    {
        throw logic_error("Encode target data: Invalid case");
    }
    else if(maximum_target_length == 1 && target_vocabulary.size() == 6) // Binary classification
    {
        // Binary classification

        for(size_t sample_index = 0; sample_index < target_document_tokens.size(); sample_index++)
        {
            const string& token = target_document_tokens[sample_index][0];

            if (contains(positive_words, token) || contains(negative_words, token))
            {
                data(sample_index, maximum_input_length) = contains(positive_words, token)
                ? 1
                : 0;
            }
            else
            {
                // @todo
            }
        }
    }
    else if(maximum_target_length == 6 && target_vocabulary.size() >= 6) // Multiple classification
    {
        for(size_t sample_index = 0; sample_index < target_document_tokens.size(); sample_index++)
        {
            const string& token = target_document_tokens[sample_index][0];

            auto iterator = find(target_vocabulary.begin(), target_vocabulary.end(), token);

            const Index token_index = (iterator != target_vocabulary.end())
                                          ? distance(target_vocabulary.begin(), iterator)
                                          : 1;

            data(sample_index, maximum_input_length + token_index - reserved_tokens.size()) = 1;
        }
    }
    else
    {
        unordered_map<string, Index> target_vocabulary_map;

        for (Index i = 0; i < Index(target_vocabulary.size()); ++i)
            target_vocabulary_map[target_vocabulary[i]] = i;

        const Index samples_number = get_samples_number();

        #pragma omp parallel for
        for (Index sample = 0; sample < samples_number; sample++)
        {
            data(sample, maximum_input_length) = 2; // start;

            const vector<string>& target_tokens = target_document_tokens[sample];

            for (Index variable = maximum_input_length + 1;
                variable < maximum_input_length + 1 + Index(target_tokens.size());
                variable++)
            {
                const Index token_index = variable - (maximum_input_length + 1);

                const string& current_token = target_tokens[token_index];

                auto it = find(target_vocabulary.begin(), target_vocabulary.end(), current_token);

                data(sample, variable) = (it != target_vocabulary.end()) ? distance(target_vocabulary.begin(), it) : 1;
            }

            data(sample, maximum_input_length + target_tokens.size() + 1) = 3; // end;
        }
    }
}


void LanguageDataset::print_input_vocabulary() const
{
    for(size_t i = 0; i < input_vocabulary.size(); i++)
        cout << i << " : " << input_vocabulary[i] << endl;
}


void LanguageDataset::print_target_vocabulary() const
{
    for(size_t i = 0; i < target_vocabulary.size(); i++)
        cout << i << " : " << target_vocabulary[i] << endl;
}


void LanguageDataset::print() const
{
    cout << "Language dataset" << endl
         << "Input vocabulary size: " << get_input_vocabulary_size() << endl
         << "Target vocabulary size: " << get_target_vocabulary_size() << endl
         << "Input length: " << get_input_sequence_length() << endl
         << "Target length: " << get_target_sequence_length() << endl
         << "Input variables number: " << get_variables_number("Input") << endl
         << "Target variables number: " << get_variables_number("Target") << endl;
}


void LanguageDataset::from_XML(const XMLDocument& data_set_document)
{
    const XMLElement* data_set_element = data_set_document.FirstChildElement("Dataset");

    if(!data_set_element)
        throw runtime_error("Dataset element is nullptr.\n");

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
        raw_variable.set_role(read_xml_string(raw_variable_element, "Role"));
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

        set_sample_uses("Training");
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
/*
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
*/
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


Index LanguageDataset::count_non_empty_lines() const
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


// @todo add "decoder variables"

void LanguageDataset::read_csv()
{
    const Index samples_number = count_non_empty_lines();

    ifstream file(data_path);

    if (!file.is_open())
        throw runtime_error("Cannot open data file: " + data_path.string());

    cout << "Reading .txt file..." << endl;

    vector<vector<string>> input_document_tokens(samples_number);
    vector<vector<string>> target_document_tokens(samples_number);

    string line;
    Index sample_index = 0;
    const string separator = get_separator_string();

    while (getline(file, line))
    {
        if (line.empty()) continue;

        const vector<string> tokens = get_tokens(line, separator);

        if (tokens.size() != 2)
            throw runtime_error("Line must contain exactly 2 fields");

        input_document_tokens[sample_index] = tokenize(tokens[0]);
        target_document_tokens[sample_index] = tokenize(tokens[1]);

        sample_index++;
    }

    if (sample_index != samples_number)
        throw runtime_error("Error: Expected " + to_string(samples_number) + " samples, but " + to_string(sample_index) + " found.");

    // set vocabulary

    create_vocabulary(input_document_tokens, input_vocabulary);
    create_vocabulary(target_document_tokens, target_vocabulary);

    // set data size

    maximum_input_length = get_maximum_size(input_document_tokens) + 2;

    if(get_maximum_size(target_document_tokens) == 1 && target_vocabulary.size() == 6)
        maximum_target_length = 1;
    else if(get_maximum_size(target_document_tokens) == 1 && target_vocabulary.size() > 6)
        maximum_target_length = target_vocabulary.size() - 4;
    else
        get_maximum_size(target_document_tokens) + 2;

    // maximum_target_length = (get_maximum_size(target_document_tokens) == 1)
    //     ? 1
    //     : get_maximum_size(target_document_tokens) + 2;

    const Index variables_number = maximum_input_length + maximum_target_length;

    data.resize(samples_number, variables_number);

    raw_variables.resize(variables_number);

    for_each(raw_variables.begin(), raw_variables.begin() + maximum_input_length,
             [](auto& var) {var.role = "Input";});

    for_each(raw_variables.begin() + maximum_input_length, raw_variables.begin() + maximum_input_length + maximum_target_length,
             [](auto& var) { var.role = "Target"; });

    // set data

    encode_input_data(input_document_tokens);
    encode_target_data(target_document_tokens);

    sample_uses.resize(samples_number);

    target_dimensions = {get_target_sequence_length()};
    decoder_dimensions = {get_target_sequence_length()};
    input_dimensions = {get_input_sequence_length()};

    set_raw_variable_scalers("None");
    set_default_raw_variable_names();
    split_samples_random();
    set_binary_raw_variables();

    cout << "Reading finished" << endl;
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
