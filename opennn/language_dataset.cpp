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
    const unordered_map<string, Index> input_vocabulary_map = create_vocabulary_map(input_vocabulary);

    const Index samples_number = get_samples_number();

    #pragma omp parallel for

    for (Index sample = 0; sample < samples_number; sample++)
    {
        data(sample, 0) = 2; // start

        const vector<string>& input_tokens = input_document_tokens[sample];

        const size_t input_tokens_number = input_tokens.size();

        for (size_t variable = 0; variable < input_tokens_number; variable++)
        {
            auto it = find(input_vocabulary.begin(), input_vocabulary.end(), input_tokens[variable]);

            data(sample, 1+variable) = (it != input_vocabulary.end()) ? it - input_vocabulary.begin() : 1;
        }

        data(sample, input_tokens_number + 1) = 3; // end
    }
}


void LanguageDataset::encode_target_data(const vector<vector<string>>& target_document_tokens)
{
    if(maximum_target_length == 1 && target_vocabulary.size() < 6)
        throw logic_error("Encode target data: Invalid case");

    const size_t target_document_tokens_number = target_document_tokens.size();

    // Binary classification

    if(maximum_target_length == 1 && target_vocabulary.size() == 6)
    {
        for(size_t sample_index = 0; sample_index < target_document_tokens_number; sample_index++)
        {
            const string& token = target_document_tokens[sample_index][0];

            if(contains(positive_words, token))
                data(sample_index, maximum_input_length) = 1;
            else if(contains(negative_words, token))
                data(sample_index, maximum_input_length) = 0;
            else
                throw runtime_error("Unknown target value");
        }

        return;
    }

    // Multiple classification

    if(maximum_target_length == 6 && target_vocabulary.size() >= 6)
    {
        for(size_t sample_index = 0; sample_index < target_document_tokens_number; sample_index++)
        {
            const string& token = target_document_tokens[sample_index][0];

            auto iterator = find(target_vocabulary.begin(), target_vocabulary.end(), token);

            const Index token_index = (iterator != target_vocabulary.end())
                                          ? distance(target_vocabulary.begin(), iterator)
                                          : 1;

            data(sample_index, maximum_input_length + token_index - reserved_tokens.size()) = 1;
        }

        return;
    }

    // Other

    {
        const unordered_map<string, Index> target_vocabulary_map = create_vocabulary_map(target_vocabulary);

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


dimensions LanguageDataset::get_input_dimensions() const
{
    return dimensions({get_input_vocabulary_size(), get_input_sequence_length()});
}


dimensions LanguageDataset::get_target_dimensions() const
{
    return dimensions({get_variables_number("Target")});
}

// @todo add "decoder variables"

void LanguageDataset::read_csv()
{   
    const Index samples_number = count_non_empty_lines(data_path);

    ifstream file(data_path);

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
            throw runtime_error("Line must contain two tokens");

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

    const Index maximum_target_document_tokens = get_maximum_size(target_document_tokens);

    if(maximum_target_document_tokens == 1 && target_vocabulary.size() == 6)
        maximum_target_length = 1;
    else if(maximum_target_document_tokens == 1 && target_vocabulary.size() > 6)
        maximum_target_length = target_vocabulary.size() - 4;
    else
        throw runtime_error("Unknown case in read_csv");

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


unordered_map<string, Index> LanguageDataset::create_vocabulary_map(const vector<string> &vocabulary)
{
    unordered_map<string, Index> vocabulary_map;

    for (Index i = 0; i < Index(vocabulary.size()); ++i)
        vocabulary_map[vocabulary[i]] = i;

    return vocabulary_map;
}


void LanguageDataset::to_XML(XMLPrinter& printer) const
{
    printer.OpenElement("Dataset");

    printer.OpenElement("DataSource");

    add_xml_element(printer, "FileType", "csv");
    add_xml_element(printer, "DataPath", data_path.string());
    add_xml_element(printer, "Separator", get_separator_string());
    add_xml_element(printer, "HasHeader", to_string(has_header));
    add_xml_element(printer, "HasSamplesId", to_string(has_sample_ids));
    add_xml_element(printer, "MissingValuesLabel", missing_values_label);
    add_xml_element(printer, "Codification", get_codification_string());
    printer.CloseElement();

    raw_variables_to_XML(printer);

    if(has_sample_ids)
        add_xml_element(printer, "SampleIds", vector_to_string(sample_ids));

    samples_to_XML(printer);

    missing_values_to_XML(printer);

    printer.OpenElement("TargetVocabulary");

    //    for (const auto& word : target_vocabulary)
    //        add_xml_element(printer, "Word", word);

    printer.CloseElement();

    printer.OpenElement("DecoderVocabulary");

    //    for (const auto& word : input_vocabulary)
    //        add_xml_element(printer, "Word", word);

    printer.CloseElement();

    // @todo look at this

    add_xml_element(printer, "TargetDimensions", to_string(maximum_target_length));
    add_xml_element(printer, "targetDimensions", to_string(maximum_input_length));

    printer.CloseElement();
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

    const XMLElement* raw_variables_element = data_set_element->FirstChildElement("RawVariables");

    raw_variables_from_XML(raw_variables_element);

    if(has_sample_ids)
        sample_ids = get_tokens(read_xml_string(data_set_element, "SamplesId"), ",");

    const XMLElement* samples_element = data_set_element->FirstChildElement("Samples");

    samples_from_XML(samples_element);

    const XMLElement* missing_values_element = data_set_element->FirstChildElement("MissingValues");

    missing_values_from_XML(missing_values_element);

    // Preview data

    const XMLElement* preview_data_element = data_set_element->FirstChildElement("PreviewData");

    preview_data_from_XML(preview_data_element);

    // Completion Vocabulary

    const XMLElement* completion_vocabulary_element = data_set_element->FirstChildElement("CompletionVocabulary");
    vocabulary_from_XML(completion_vocabulary_element, target_vocabulary);

    // Decoder Vocabulary

    const XMLElement* context_vocabulary_element = data_set_element->FirstChildElement("DecoderVocabulary");

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


void LanguageDataset::vocabulary_from_XML(const XMLElement *element, vector<string> &vocabulary)
{
    if(!element)
        return;

    vocabulary.clear();

    for (const XMLElement* word_element = element->FirstChildElement(); word_element; word_element = word_element->NextSiblingElement())
    {
        //            if (word_element->GetText())
        //                input_vocabulary.push_back(word_element->GetText());
    }
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
