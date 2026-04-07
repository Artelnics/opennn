//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   L A N G U A G E  D A T A S E T   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "language_dataset.h"
#include "string_utilities.h"
#include "random_utilities.h"
#include "tensor_utilities.h"

namespace opennn
{

LanguageDataset::LanguageDataset(const filesystem::path& new_data_path) : Dataset()
{
    data_path = new_data_path;
    separator = Dataset::Separator::Tab;

    if(!data_path.empty())
        read_csv();
}


LanguageDataset::LanguageDataset(const Index samples_number,
                                 Index input_sequence_length,
                                 Index input_vocabulary_size) : Dataset()
{
    // @todo check this code

    maximum_input_sequence_length = input_sequence_length;
    maximum_target_sequence_length = 1;

    const Index features_number = input_sequence_length + 1;

    data.resize(samples_number, features_number);
    variables.resize(features_number);

    set_default();

    for(Index i = 0; i < features_number; i++)
    {
        Variable& variable = variables[i];

        variable.type = VariableType::Numeric;
        variable.name = "variable_" + to_string(i + 1);

        variable.role = (i < input_sequence_length)
            ? "Input"
            : "Target";
    }

    sample_roles.resize(samples_number);
    split_samples_random();

    const Index target_column_index = data.cols() - 1;

    for(Index i = 0; i < data.rows(); ++i)
    {
        for(Index j = 0; j < target_column_index; ++j)
            data(i, j) = random_integer(0, input_vocabulary_size - 1);

        data(i, target_column_index) = random_integer(0, 1);
    }

    input_vocabulary.resize(input_vocabulary_size + reserved_tokens.size());
    target_vocabulary.resize(2);

    if(!variables.empty())
        variables[0].categories = input_vocabulary;

    input_shape = { get_maximum_input_sequence_length() };
    target_shape = { get_maximum_target_sequence_length() };
    decoder_shape.clear();

    set_variable_scalers("None");
    set_default_variable_names();
    set_binary_variables();
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


Index LanguageDataset::get_maximum_input_sequence_length() const
{
    return maximum_input_sequence_length;
}


Index LanguageDataset::get_maximum_target_sequence_length() const
{
    return maximum_target_sequence_length;
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
    unordered_map<string, size_t> token_count;

    for(const vector<string>& document : document_tokens)
        for(const string& token : document)
            ++token_count[token];

    vector<pair<string, size_t>> sorted_tokens(token_count.begin(), token_count.end());

    sort(sorted_tokens.begin(), sorted_tokens.end(),
         [](const pair<string, size_t>& a, const pair<string, size_t>& b){return a.second > b.second;});

    vocabulary.clear();

    for(const string& token : reserved_tokens)
        vocabulary.push_back(token);

    for(const pair<string, size_t>& token : sorted_tokens)
    {
        if (token.second < size_t(minimum_token_frequency))
            continue;

        if (find(reserved_tokens.begin(), reserved_tokens.end(), token.first) != reserved_tokens.end())
            continue;

        if (vocabulary.size() >= size_t(maximum_vocabulary_size))
            break;

        vocabulary.push_back(token.first);
    }
}


void LanguageDataset::encode_input(const vector<vector<string>>& input_document_tokens)
{
    const unordered_map<string, Index> input_vocabulary_map = create_vocabulary_map(input_vocabulary);
    const Index samples_number = get_samples_number();

    #pragma omp parallel for

    for(Index sample = 0; sample < samples_number; sample++)
    {
        data(sample, 0) = START_INDEX;

        const vector<string>& input_tokens = input_document_tokens[sample];
        const size_t input_tokens_number = input_tokens.size();

        for(size_t i = 0; i < input_tokens_number; i++)
        {
            if (1 + i >= (size_t)maximum_input_sequence_length) break;

            const unordered_map<string, Index>::const_iterator iterator = input_vocabulary_map.find(input_tokens[i]);

            data(sample, 1 + i) = (iterator != input_vocabulary_map.end())
                ? static_cast<type>(iterator->second)
                : UNK_INDEX;
        }

        if (1 + input_tokens_number < (size_t)maximum_input_sequence_length)
            data(sample, 1 + input_tokens_number) = END_INDEX;
    }
}


void LanguageDataset::encode_decoder_target_sequence_to_sequence(const vector<vector<string>>& target_document_tokens)
{
    const unordered_map<string, Index> target_vocabulary_map = create_vocabulary_map(target_vocabulary);
    const Index samples_number = get_samples_number();

    const Index decoder_offset = maximum_input_sequence_length;
    const Index target_offset = maximum_input_sequence_length + maximum_target_sequence_length;

#pragma omp parallel for
    for(Index sample = 0; sample < samples_number; sample++)
    {
        const vector<string>& target_tokens = target_document_tokens[sample];
        const Index target_tokens_number = static_cast<Index>(target_tokens.size());

        data(sample, decoder_offset) = START_INDEX;

        for(Index i = 0; i < target_tokens_number; i++)
        {
            if(i + 1 >= maximum_target_sequence_length) break;

            const unordered_map<string, Index>::const_iterator iterator = target_vocabulary_map.find(target_tokens[i]);

            data(sample, decoder_offset + 1 + i) =
                (iterator != target_vocabulary_map.end())
                    ? static_cast<type>(iterator->second)
                    : UNK_INDEX;
        }

        for(Index i = 0; i < target_tokens_number; i++)
        {
            if(i >= maximum_target_sequence_length) break;

            const unordered_map<string, Index>::const_iterator iterator = target_vocabulary_map.find(target_tokens[i]);

            data(sample, target_offset + i) =
                (iterator != target_vocabulary_map.end())
                    ? static_cast<type>(iterator->second)
                    : UNK_INDEX;
        }

        if(target_tokens_number < maximum_target_sequence_length)
            data(sample, target_offset + target_tokens_number) = END_INDEX;
    }
}


void LanguageDataset::encode_target_classification(const vector<vector<string>>& target_document_tokens)
{
    if(maximum_target_sequence_length == 1 && target_vocabulary.size() < 6)
        throw logic_error("Encode target data: Invalid case");

    const size_t target_document_tokens_number = target_document_tokens.size();

    // Binary classification

    if(maximum_target_sequence_length == 1 && target_vocabulary.size() == 6)
    {
        for(size_t sample_index = 0; sample_index < target_document_tokens_number; sample_index++)
        {
            const string& token = target_document_tokens[sample_index][0];

            if(contains(positive_words, token))
                data(sample_index, maximum_input_sequence_length) = 1;
            else if(contains(negative_words, token))
                data(sample_index, maximum_input_sequence_length) = 0;
            else
                throw runtime_error("Unknown target value");
        }

        return;
    }

    // Multiple classification

    if(maximum_target_sequence_length == 6 && target_vocabulary.size() >= 6)
    {
        for(size_t sample_index = 0; sample_index < target_document_tokens_number; sample_index++)
        {
            const string& token = target_document_tokens[sample_index][0];

            vector<string>::iterator iterator = find(target_vocabulary.begin(), target_vocabulary.end(), token);

            const Index token_index = (iterator != target_vocabulary.end())
                                          ? distance(target_vocabulary.begin(), iterator)
                                          : 1;

            data(sample_index, maximum_input_sequence_length + token_index - reserved_tokens.size()) = 1;
        }

        return;
    }

    // Other

    {
        const unordered_map<string, Index> target_vocabulary_map = create_vocabulary_map(target_vocabulary);

        const Index samples_number = get_samples_number();

        #pragma omp parallel for
        for(Index sample = 0; sample < samples_number; sample++)
        {
            data(sample, maximum_input_sequence_length) = 2; // start;

            const vector<string>& target_tokens = target_document_tokens[sample];

            for(Index variable = maximum_input_sequence_length + 1;
                variable < maximum_input_sequence_length + 1 + Index(target_tokens.size());
                variable++)
            {
                const Index token_index = variable - (maximum_input_sequence_length + 1);

                const string& current_token = target_tokens[token_index];

                vector<string>::iterator iterator = find(target_vocabulary.begin(), target_vocabulary.end(), current_token);

                data(sample, variable) = (iterator != target_vocabulary.end()) ? distance(target_vocabulary.begin(), iterator) : 1;
            }

            data(sample, maximum_input_sequence_length + target_tokens.size() + 1) = 3; // end;
        }
    }
}


void LanguageDataset::read_csv()
{
    cout << "Reading .txt file..." << endl;

    if(data_path.empty())
        throw runtime_error("Error: Data path is empty.\n");

    if(!filesystem::exists(data_path))
        throw runtime_error("Error: The file " + data_path.string() + " does not exist.\n");

    const Index samples_number = count_non_empty_lines(data_path);

    ifstream file(data_path);

    if(!file.is_open())
        throw runtime_error("Error: Cannot open file " + data_path.string());

    const string separator = get_separator_string();

    vector<vector<string>> input_document_tokens(samples_number);
    vector<vector<string>> target_document_tokens(samples_number);

    string line;
    Index sample_index = 0;

    while(getline(file, line))
    {
        if(line.empty()) continue;

        const vector<string> tokens = get_tokens(line, separator);

        if(tokens.size() != 2)
            throw runtime_error("Line must contain two fields: input and target.");

        input_document_tokens[sample_index] = tokenize(tokens[0]);
        target_document_tokens[sample_index] = tokenize(tokens[1]);

        sample_index++;
    }

    if(sample_index != samples_number)
        throw runtime_error("Error: Expected " + to_string(samples_number) + " samples, but " + to_string(sample_index) + " found.");

    create_vocabulary(input_document_tokens, input_vocabulary);
    create_vocabulary(target_document_tokens, target_vocabulary);

    maximum_input_sequence_length = get_maximum_size(input_document_tokens) + 2;

    const Index maximum_target_document_tokens = get_maximum_size(target_document_tokens);
    const Index target_vocabulary_size = get_target_vocabulary_size();

    const bool is_single_token_target = (maximum_target_document_tokens == 1);

    // Keep current behaviour for the cases that already work:
    // - binary classification
    // - one-token multiclass classification
    //
    // New behaviour:
    // - multi-token target => seq2seq / transformer
    if(is_single_token_target)
    {
        maximum_target_sequence_length = (target_vocabulary_size == 6)
        ? 1
        : target_vocabulary_size - 4;

        const Index features_number = maximum_input_sequence_length + maximum_target_sequence_length;

        data.resize(samples_number, features_number);
        data.setZero();

        variables.resize(features_number);

        for_each(variables.begin(),
                 variables.begin() + maximum_input_sequence_length,
                 [](Variable& variable) { variable.role = "Input"; });

        for_each(variables.begin() + maximum_input_sequence_length,
                 variables.begin() + maximum_input_sequence_length + maximum_target_sequence_length,
                 [](Variable& variable) { variable.role = "Target"; });

        if(!variables.empty())
            variables[0].categories = input_vocabulary;

        for(Index i = 0; i < maximum_input_sequence_length; i++)
            variables[i].name = "token_" + to_string(i + 1);

        encode_input(input_document_tokens);
        encode_target_classification(target_document_tokens);

        input_shape = { get_maximum_input_sequence_length() };
        target_shape = { get_maximum_target_sequence_length() };
        decoder_shape.clear();
    }
    else
    {
        // Seq2seq / Transformer case.
        // Assumes:
        //   decoder = [START, y1, ..., yN]
        //   target  = [y1, ..., yN, END]
        // so each sequence needs max_target_tokens + 1 positions.
        maximum_target_sequence_length = maximum_target_document_tokens + 1;

        const Index decoder_offset = maximum_input_sequence_length;
        const Index target_offset = decoder_offset + maximum_target_sequence_length;
        const Index features_number = maximum_input_sequence_length
                                      + maximum_target_sequence_length
                                      + maximum_target_sequence_length;

        data.resize(samples_number, features_number);
        data.setZero();

        variables.resize(features_number);

        for_each(variables.begin(),
                 variables.begin() + maximum_input_sequence_length,
                 [](Variable& variable) { variable.role = "Input"; });

        for_each(variables.begin() + decoder_offset,
                 variables.begin() + decoder_offset + maximum_target_sequence_length,
                 [](Variable& variable) { variable.role = "Decoder"; });

        for_each(variables.begin() + target_offset,
                 variables.begin() + target_offset + maximum_target_sequence_length,
                 [](Variable& variable) { variable.role = "Target"; });

        if(!variables.empty())
            variables[0].categories = input_vocabulary;

        for(Index i = 0; i < maximum_input_sequence_length; i++)
            variables[i].name = "input_token_" + to_string(i + 1);

        for(Index i = 0; i < maximum_target_sequence_length; i++)
            variables[decoder_offset + i].name = "decoder_token_" + to_string(i + 1);

        for(Index i = 0; i < maximum_target_sequence_length; i++)
            variables[target_offset + i].name = "target_token_" + to_string(i + 1);

        encode_input(input_document_tokens);

        // @todo This requires encode_decoder_target_sequence_to_sequence(...)
        // to be implemented consistently with:
        //   decoder = [START, y1, ..., yN]
        //   target  = [y1, ..., yN, END]
        encode_decoder_target_sequence_to_sequence(target_document_tokens);

        input_shape = { get_maximum_input_sequence_length() };
        decoder_shape = { get_maximum_target_sequence_length() };
        target_shape = { get_maximum_target_sequence_length() };
    }

    sample_roles.resize(samples_number);

    set_variable_scalers("None");
    set_default_variable_names();
    split_samples_random();
    set_binary_variables();

    cout << "Reading finished" << endl;
}


unordered_map<string, Index> LanguageDataset::create_vocabulary_map(const vector<string>& vocabulary)
{
    unordered_map<string, Index> vocabulary_map;

    for(Index i = 0; i < Index(vocabulary.size()); ++i)
        vocabulary_map[vocabulary[i]] = i;

    return vocabulary_map;
}


void LanguageDataset::to_XML(XMLPrinter& printer) const
{
    printer.OpenElement("Dataset");

    printer.OpenElement("DataSource");

    add_xml_element(printer, "FileType", "csv");
    add_xml_element(printer, "Path", data_path.string());
    add_xml_element(printer, "Separator", get_separator_name());
    add_xml_element(printer, "HasHeader", to_string(has_header));
    add_xml_element(printer, "HasSamplesId", to_string(has_sample_ids));
    add_xml_element(printer, "MissingValuesLabel", missing_values_label);
    add_xml_element(printer, "Codification", get_codification_string());
    printer.CloseElement();

    variables_to_XML(printer);

    samples_to_XML(printer);

    missing_values_to_XML(printer);

    preview_data_to_XML(printer);

    const string separator_string = get_separator_string();

    add_xml_element(printer, "InputVocabulary", vector_to_string(input_vocabulary, separator_string));
    add_xml_element(printer, "TargetVocabulary", vector_to_string(target_vocabulary, separator_string));

    add_xml_element(printer, "MaximumInputSequenceLength", to_string(maximum_input_sequence_length));
    add_xml_element(printer, "MaximumTargetSequenceLength", to_string(maximum_target_sequence_length));

    add_xml_element(printer, "Display", to_string(display));

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
    set_separator_name(read_xml_string(data_source_element, "Separator"));
    set_missing_values_label(read_xml_string(data_source_element, "MissingValuesLabel"));
    set_codification(read_xml_string(data_source_element, "Codification"));
    set_has_header(read_xml_bool(data_source_element, "HasHeader"));
    set_has_ids(read_xml_bool(data_source_element, "HasSamplesId"));

    const XMLElement* variables_element = data_set_element->FirstChildElement("Variables");
    variables_from_XML(variables_element);

    const XMLElement* samples_element = data_set_element->FirstChildElement("Samples");
    samples_from_XML(samples_element);

    const XMLElement* missing_values_element = data_set_element->FirstChildElement("MissingValues");
    missing_values_from_XML(missing_values_element);

    const XMLElement* preview_data_element = data_set_element->FirstChildElement("PreviewData");
    preview_data_from_XML(preview_data_element);

    const string separator_string = get_separator_string();

    const XMLElement* input_vocabulary_element = data_set_element->FirstChildElement("InputVocabulary");
    if(input_vocabulary_element && input_vocabulary_element->GetText())
        input_vocabulary = get_tokens(input_vocabulary_element->GetText(), separator_string);
    else
        input_vocabulary.clear();

    const XMLElement* target_vocabulary_element = data_set_element->FirstChildElement("TargetVocabulary");
    if(target_vocabulary_element && target_vocabulary_element->GetText())
        target_vocabulary = get_tokens(target_vocabulary_element->GetText(), separator_string);
    else
        target_vocabulary.clear();

    const XMLElement* input_len_element = data_set_element->FirstChildElement("MaximumInputSequenceLength");
    if(input_len_element && input_len_element->GetText())
        maximum_input_sequence_length = atoi(input_len_element->GetText());
    else
        maximum_input_sequence_length = 0;

    const XMLElement* target_len_element = data_set_element->FirstChildElement("MaximumTargetSequenceLength");
    if(target_len_element && target_len_element->GetText())
        maximum_target_sequence_length = atoi(target_len_element->GetText());
    else
        maximum_target_sequence_length = 0;

    input_shape = { get_maximum_input_sequence_length() };
    target_shape = { get_maximum_target_sequence_length() };

    const bool has_decoder_variables = (get_variables_number("Decoder") > 0);

    if(has_decoder_variables)
        decoder_shape = { get_maximum_target_sequence_length() };
    else
        decoder_shape.clear();

    if(!variables.empty() && !input_vocabulary.empty())
        variables[0].categories = input_vocabulary;

    set_display(read_xml_bool(data_set_element, "Display"));

    ifstream file(data_path);

    if(!file.is_open())
        throw runtime_error("Error: Cannot open file " + data_path.string() + "\n");

    vector<vector<string>> input_docs_tokens;
    vector<vector<string>> target_docs_tokens;

    string line;
    const string separator = get_separator_string();

    if(has_header)
        getline(file, line);

    while(getline(file, line))
    {
        if(line.empty())
            continue;

        const vector<string> tokens = get_tokens(line, separator);

        if(tokens.size() != 2)
            continue;

        input_docs_tokens.push_back(tokenize(tokens[0]));
        target_docs_tokens.push_back(tokenize(tokens[1]));
    }

    if(input_docs_tokens.size() != static_cast<size_t>(get_samples_number()))
    {
        cout << "Warning: Loaded samples count (" << get_samples_number()
        << ") does not match file lines (" << input_docs_tokens.size() << ")." << endl;
    }

    data.setZero();

    encode_input(input_docs_tokens);

    if(has_decoder_variables)
        encode_decoder_target_sequence_to_sequence(target_docs_tokens);
    else
        encode_target_classification(target_docs_tokens);
}


void LanguageDataset::print() const
{
    cout << "Language dataset" << endl
         << "Samples number: " << get_samples_number() << endl
         << "Input vocabulary size: " << get_input_vocabulary_size() << endl
         << "Target vocabulary size: " << get_target_vocabulary_size() << endl
         << "Maximum input sequence length: " << get_maximum_input_sequence_length() << endl
         << "Maximum target sequence length: " << get_maximum_target_sequence_length() << endl
         << "Input shape: " << get_shape("Input") << endl
         << "Decoder shape: " << get_shape("Decoder") << endl
         << "Target shape: " << get_shape("Target") << endl
         << "Input variables number: " << get_variables_number("Input") << endl
         << "Decoder variables number: " << get_variables_number("Decoder") << endl
         << "Target variables number: " << get_variables_number("Target") << endl
         << "Input features number: " << get_features_number("Input") << endl
         << "Decoder features number: " << get_features_number("Decoder") << endl
         << "Target features number: " << get_features_number("Target") << endl
         << "Has decoder: " << (get_variables_number("Decoder") > 0 ? "True" : "False") << endl;
}

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// This library is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 2.1 of the License, or any later version.
// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.
// You should have received a copy of the GNU Lesser General Public
// License along with this library; if not, write to the Free Software
// Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
