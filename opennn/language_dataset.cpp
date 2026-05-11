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

namespace {

template <typename T>
size_t get_maximum_size(const vector<vector<T>>& nested_values)
{
    size_t maximum_size = 0;
    for (const auto& inner : nested_values)
        if (inner.size() > maximum_size)
            maximum_size = inner.size();
    return maximum_size;
}

}

LanguageDataset::LanguageDataset(const filesystem::path& new_data_path) : Dataset()
{
    data_path = new_data_path;
    separator = Dataset::Separator::Tab;

    if (!data_path.empty())
        read_csv();
}

LanguageDataset::LanguageDataset(const filesystem::path& new_data_path,
                                 Index new_maximum_vocabulary_size) : Dataset()
{
    data_path = new_data_path;
    separator = Dataset::Separator::Tab;
    maximum_vocabulary_size = new_maximum_vocabulary_size;

    if (!data_path.empty())
        read_csv();
}

LanguageDataset::LanguageDataset(const filesystem::path& new_data_path,
                                 Index new_maximum_vocabulary_size,
                                 Index new_minimum_token_frequency) : Dataset()
{
    data_path = new_data_path;
    separator = Dataset::Separator::Tab;
    maximum_vocabulary_size = new_maximum_vocabulary_size;
    minimum_token_frequency = new_minimum_token_frequency;

    if (!data_path.empty())
        read_csv();
}

LanguageDataset::LanguageDataset(const Index samples_number,
                                 Index input_sequence_length,
                                 Index input_vocabulary_size) : Dataset()
{
    maximum_input_sequence_length = input_sequence_length;
    maximum_target_sequence_length = 1;

    const Index features_number = input_sequence_length + 1;

    data.resize(samples_number, features_number);
    variables.resize(features_number);

    set_default();

    for (Index i = 0; i < features_number; ++i)
    {
        Variable& variable = variables[i];

        variable.type = VariableType::Numeric;
        variable.name = "variable_" + to_string(i + 1);

        variable.role = (i < input_sequence_length)
            ? VariableRole::Input
            : VariableRole::Target;
    }

    sample_roles.resize(samples_number);
    split_samples_random();

    const Index target_column_index = data.cols() - 1;

    for (Index i = 0; i < data.rows(); ++i)
    {
        for (Index j = 0; j < target_column_index; ++j)
            data(i, j) = random_integer(0, input_vocabulary_size - 1);

        data(i, target_column_index) = random_integer(0, 1);
    }

    input_vocabulary.resize(input_vocabulary_size + reserved_tokens.size());
    target_vocabulary.resize(2);

    if (!variables.empty())
        variables[0].categories = input_vocabulary;

    input_shape = { get_maximum_input_sequence_length() };
    target_shape = { get_maximum_target_sequence_length() };
    decoder_shape.clear();

    set_variable_scalers("None");
    set_default_variable_names();
    set_binary_variables();

    for_each(variables.begin(),
             variables.begin() + maximum_input_sequence_length,
             [](Variable& variable) { variable.role = VariableRole::Input; });
}

void LanguageDataset::create_vocabulary(const vector<vector<string>>& document_tokens,
                                        vector<string>& vocabulary) const
{
    unordered_map<string, size_t> token_count;

    for (const vector<string>& document : document_tokens)
        for (const string& token : document)
            ++token_count[token];

    vector<pair<string, size_t>> sorted_tokens(token_count.begin(), token_count.end());

    sort(sorted_tokens.begin(), sorted_tokens.end(),
         [](const pair<string, size_t>& a, const pair<string, size_t>& b) {return a.second > b.second;});

    vocabulary.clear();

    for (const string& token : reserved_tokens)
        vocabulary.push_back(token);

    for (const pair<string, size_t>& token : sorted_tokens)
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
    const Index samples_number = get_samples_number();

    #pragma omp parallel for

    for (Index sample = 0; sample < samples_number; ++sample)
    {
        data(sample, 0) = START_INDEX;

        const vector<string>& input_tokens = input_document_tokens[sample];
        const size_t input_tokens_number = input_tokens.size();

        for (size_t i = 0; i < input_tokens_number; ++i)
        {
            if (1 + i >= static_cast<size_t>(maximum_input_sequence_length)) break;

            const auto iterator = input_vocabulary_map.find(input_tokens[i]);

            data(sample, 1 + i) = (iterator != input_vocabulary_map.end())
                ? static_cast<float>(iterator->second)
                : UNK_INDEX;
        }

        if (1 + input_tokens_number < static_cast<size_t>(maximum_input_sequence_length))
            data(sample, 1 + input_tokens_number) = END_INDEX;
    }
}

void LanguageDataset::encode_decoder_target_sequence_to_sequence(const vector<vector<string>>& target_document_tokens)
{
    const Index samples_number = get_samples_number();

    const Index decoder_offset = maximum_input_sequence_length;
    const Index target_offset = maximum_input_sequence_length + maximum_target_sequence_length;

#pragma omp parallel for
    for (Index sample = 0; sample < samples_number; ++sample)
    {
        const vector<string>& target_tokens = target_document_tokens[sample];
        const Index target_tokens_number = ssize(target_tokens);

        data(sample, decoder_offset) = START_INDEX;

        for (Index i = 0; i < target_tokens_number; ++i)
        {
            if (i >= maximum_target_sequence_length) break;

            const auto iterator = target_vocabulary_map.find(target_tokens[i]);

            const float token_index = (iterator != target_vocabulary_map.end())
                ? static_cast<float>(iterator->second)
                : UNK_INDEX;

            if (i + 1 < maximum_target_sequence_length)
                data(sample, decoder_offset + 1 + i) = token_index;

            data(sample, target_offset + i) = token_index;
        }

        if (target_tokens_number < maximum_target_sequence_length)
            data(sample, target_offset + target_tokens_number) = END_INDEX;
    }
}

void LanguageDataset::encode_target_classification(const vector<vector<string>>& target_document_tokens)
{
    if (maximum_target_sequence_length == 1 && target_vocabulary.size() < 6)
        throw logic_error("Encode target data: Invalid case");

    const size_t target_document_tokens_number = target_document_tokens.size();

    if (maximum_target_sequence_length == 1 && target_vocabulary.size() == 6)
    {
        for (size_t sample_index = 0; sample_index < target_document_tokens_number; ++sample_index)
        {
            const string& token = target_document_tokens[sample_index][0];

            if (contains(positive_words, token))
                data(sample_index, maximum_input_sequence_length) = 1;
            else if (contains(negative_words, token))
                data(sample_index, maximum_input_sequence_length) = 0;
            else
                throw runtime_error("Unknown target value");
        }

        return;
    }

    if (maximum_target_sequence_length == 6 && target_vocabulary.size() >= 6)
    {
        for (size_t sample_index = 0; sample_index < target_document_tokens_number; ++sample_index)
        {
            const string& token = target_document_tokens[sample_index][0];

            auto iterator = target_vocabulary_map.find(token);

            const Index token_index = (iterator != target_vocabulary_map.end())
                                          ? iterator->second
                                          : 1;

            data(sample_index, maximum_input_sequence_length + token_index - reserved_tokens.size()) = 1;
        }

        return;
    }

    {
        const Index samples_number = get_samples_number();

        #pragma omp parallel for
        for (Index sample = 0; sample < samples_number; ++sample)
        {
            data(sample, maximum_input_sequence_length) = 2;

            const vector<string>& target_tokens = target_document_tokens[sample];

            for (Index variable = maximum_input_sequence_length + 1;
                variable < maximum_input_sequence_length + 1 + Index(target_tokens.size());
                ++variable)
            {
                const Index token_index = variable - (maximum_input_sequence_length + 1);

                const string& current_token = target_tokens[token_index];

                auto iterator = target_vocabulary_map.find(current_token);

                data(sample, variable) = (iterator != target_vocabulary_map.end()) ? iterator->second : 1;
            }

            data(sample, maximum_input_sequence_length + target_tokens.size() + 1) = 3;
        }
    }
}

void LanguageDataset::read_csv()
{
    cout << "Reading .txt file..." << "\n";

    // Total file size is the progress bar's denominator. Skips the otherwise
    // silent pre-pass of count_non_empty_lines() — we now size the token
    // vectors with push_back during the single read.
    const auto file_size = filesystem::file_size(data_path);

    ifstream file(data_path);

    if (!file.is_open())
        throw runtime_error("Cannot open file " + data_path.string());

    const string separator = get_separator_string();

    vector<vector<string>> input_document_tokens;
    vector<vector<string>> target_document_tokens;
    input_document_tokens.reserve(1 << 16);
    target_document_tokens.reserve(1 << 16);

    string line;
    // Repaint the bar at most ~200 times across the whole file. Show 0%
    // immediately so progress is visible from the first byte.
    const auto progress_step_bytes = std::max<std::streamoff>(1, std::streamoff(file_size / 200));
    std::streamoff next_progress = progress_step_bytes;
    display_progress_bar(0, int(file_size));

    while (getline(file, line))
    {
        if (line.empty()) continue;

        const vector<string> tokens = get_tokens(line, separator);

        if (tokens.size() != 2)
            throw runtime_error("Line must contain two fields: input and target.");

        input_document_tokens.push_back(tokenize(tokens[0]));
        target_document_tokens.push_back(tokenize(tokens[1]));

        const std::streamoff pos = std::streamoff(file.tellg());
        if (pos >= next_progress || pos < 0)
        {
            display_progress_bar(int(pos > 0 ? pos : std::streamoff(file_size)),
                                 int(file_size));
            next_progress += progress_step_bytes;
        }
    }
    display_progress_bar(int(file_size), int(file_size));
    cout << "\n";

    const Index samples_number = Index(input_document_tokens.size());

    create_vocabulary(input_document_tokens, input_vocabulary);
    create_vocabulary(target_document_tokens, target_vocabulary);

    update_input_vocabulary_map();
    update_target_vocabulary_maps();

    maximum_input_sequence_length = get_maximum_size(input_document_tokens) + 2;

    const Index maximum_target_document_tokens = get_maximum_size(target_document_tokens);
    const Index target_vocabulary_size = get_target_vocabulary_size();

    const bool is_single_token_target = (maximum_target_document_tokens == 1);

    if (is_single_token_target)
    {
        maximum_target_sequence_length = (target_vocabulary_size == 6)
        ? 1
        : target_vocabulary_size - 4;

        const Index features_number = maximum_input_sequence_length + maximum_target_sequence_length;

        data = MatrixR::Zero(samples_number, features_number);

        variables.resize(features_number);

        for_each(variables.begin(),
                 variables.begin() + maximum_input_sequence_length,
                 [](Variable& variable) { variable.role = VariableRole::Input; });

        for_each(variables.begin() + maximum_input_sequence_length,
                 variables.begin() + maximum_input_sequence_length + maximum_target_sequence_length,
                 [](Variable& variable) { variable.role = VariableRole::Target; });

        if (!variables.empty())
            variables[0].categories = input_vocabulary;

        for (Index i = 0; i < maximum_input_sequence_length; ++i)
            variables[i].name = "token_" + to_string(i + 1);

        encode_input(input_document_tokens);
        encode_target_classification(target_document_tokens);

        input_shape = { get_maximum_input_sequence_length() };
        target_shape = { get_maximum_target_sequence_length() };
        decoder_shape.clear();
    }
    else
    {
        maximum_target_sequence_length = maximum_target_document_tokens + 1;

        const Index decoder_offset = maximum_input_sequence_length;
        const Index target_offset = decoder_offset + maximum_target_sequence_length;
        const Index features_number = maximum_input_sequence_length
                                      + maximum_target_sequence_length
                                      + maximum_target_sequence_length;

        data = MatrixR::Zero(samples_number, features_number);

        variables.resize(features_number);

        for_each(variables.begin(),
                 variables.begin() + maximum_input_sequence_length,
                 [](Variable& variable) { variable.role = VariableRole::Input; });

        for_each(variables.begin() + decoder_offset,
                 variables.begin() + decoder_offset + maximum_target_sequence_length,
                 [](Variable& variable) { variable.role = VariableRole::Decoder; });

        for_each(variables.begin() + target_offset,
                 variables.begin() + target_offset + maximum_target_sequence_length,
                 [](Variable& variable) { variable.role = VariableRole::Target; });

        if (!variables.empty())
            variables[0].categories = input_vocabulary;

        for (Index i = 0; i < maximum_input_sequence_length; ++i)
            variables[i].name = "input_token_" + to_string(i + 1);

        for (Index i = 0; i < maximum_target_sequence_length; ++i)
            variables[decoder_offset + i].name = "decoder_token_" + to_string(i + 1);

        for (Index i = 0; i < maximum_target_sequence_length; ++i)
            variables[target_offset + i].name = "target_token_" + to_string(i + 1);

        encode_input(input_document_tokens);

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

    for (Index i = 0; i < ssize(variables); ++i)
    {
        if (i < maximum_input_sequence_length)
            variables[i].role = VariableRole::Input;
        else if (!decoder_shape.empty() && i < maximum_input_sequence_length + maximum_target_sequence_length)
            variables[i].role = VariableRole::Decoder;
        else
            variables[i].role = VariableRole::Target;

        variables[i].type = VariableType::Numeric;
    }

    if (!variables.empty())
        variables[0].categories = input_vocabulary;

    cout << "Reading finished" << "\n";
}

void LanguageDataset::update_input_vocabulary_map()
{
    input_vocabulary_map.clear();
    input_vocabulary_map.reserve(input_vocabulary.size());

    for (Index i = 0; i < Index(input_vocabulary.size()); ++i)
        input_vocabulary_map[input_vocabulary[i]] = i;
}

void LanguageDataset::update_target_vocabulary_maps()
{
    target_vocabulary_map.clear();
    target_inverse_vocabulary_map.clear();
    target_vocabulary_map.reserve(target_vocabulary.size());
    target_inverse_vocabulary_map.reserve(target_vocabulary.size());

    for (Index i = 0; i < Index(target_vocabulary.size()); ++i)
    {
        target_vocabulary_map[target_vocabulary[i]] = i;
        target_inverse_vocabulary_map[i] = target_vocabulary[i];
    }
}

void LanguageDataset::set_input_vocabulary(const vector<string>& new_vocabulary)
{
    input_vocabulary = new_vocabulary;
    update_input_vocabulary_map();
}

void LanguageDataset::set_target_vocabulary(const vector<string>& new_vocabulary)
{
    target_vocabulary = new_vocabulary;
    update_target_vocabulary_maps();
}

void LanguageDataset::to_JSON(JsonWriter& printer) const
{
    printer.open_element("Dataset");

    printer.open_element("DataSource");

    write_json(printer, {
        {"FileType", "csv"},
        {"Path", data_path.string()},
        {"Separator", get_separator_name()},
        {"HasHeader", to_string(has_header)},
        {"HasSamplesId", to_string(has_sample_ids)},
        {"MissingValuesLabel", missing_values_label},
        {"Codification", get_codification_string()}
    });
    printer.close_element();

    variables_to_JSON(printer);

    samples_to_JSON(printer);

    missing_values_to_JSON(printer);

    preview_data_to_JSON(printer);

    const string separator_string = get_separator_string();

    write_json(printer, {
        {"InputVocabulary", vector_to_string(input_vocabulary, separator_string)},
        {"TargetVocabulary", vector_to_string(target_vocabulary, separator_string)},
        {"MaximumInputSequenceLength", to_string(maximum_input_sequence_length)},
        {"MaximumTargetSequenceLength", to_string(maximum_target_sequence_length)},
        {"Display", to_string(display)}
    });

    printer.close_element();
}

void LanguageDataset::from_JSON(const JsonDocument& data_set_document)
{
    const Json* data_set_element = get_json_root(data_set_document, "Dataset");

    const Json* data_source_element = require_json_field(data_set_element, "DataSource");

    set_data_path(read_json_string(data_source_element, "Path"));
    set_separator_name(read_json_string(data_source_element, "Separator"));
    set_missing_values_label(read_json_string(data_source_element, "MissingValuesLabel"));
    set_codification(read_json_string(data_source_element, "Codification"));
    set_has_header(read_json_bool(data_source_element, "HasHeader"));
    set_has_ids(read_json_bool(data_source_element, "HasSamplesId"));

    const Json* variables_element = data_set_element->first_child("Variables");
    variables_from_JSON(variables_element);

    const Json* samples_element = data_set_element->first_child("Samples");
    samples_from_JSON(samples_element);

    const Json* missing_values_element = data_set_element->first_child("MissingValues");
    missing_values_from_JSON(missing_values_element);

    const Json* preview_data_element = data_set_element->first_child("PreviewData");
    preview_data_from_JSON(preview_data_element);

    const string separator_string = get_separator_string();

    const string input_vocab_text = read_json_string(data_set_element, "InputVocabulary");
    if (!input_vocab_text.empty())
        input_vocabulary = get_tokens(input_vocab_text, separator_string);
    else
        input_vocabulary.clear();

    const string target_vocab_text = read_json_string(data_set_element, "TargetVocabulary");
    if (!target_vocab_text.empty())
        target_vocabulary = get_tokens(target_vocab_text, separator_string);
    else
        target_vocabulary.clear();

    update_input_vocabulary_map();
    update_target_vocabulary_maps();

    maximum_input_sequence_length  = read_json_index(data_set_element, "MaximumInputSequenceLength");
    maximum_target_sequence_length = read_json_index(data_set_element, "MaximumTargetSequenceLength");

    input_shape = { get_maximum_input_sequence_length() };
    target_shape = { get_maximum_target_sequence_length() };

    const bool has_decoder_variables = (get_variables_number("Decoder") > 0);

    if (has_decoder_variables)
        decoder_shape = { get_maximum_target_sequence_length() };
    else
        decoder_shape.clear();

    if (!variables.empty() && !input_vocabulary.empty())
        variables[0].categories = input_vocabulary;

    set_display(read_json_bool(data_set_element, "Display"));

    ifstream file(data_path);

    if (!file.is_open())
        throw runtime_error("Cannot open file " + data_path.string() + "\n");

    vector<vector<string>> input_docs_tokens;
    vector<vector<string>> target_docs_tokens;

    string line;
    const string separator = get_separator_string();

    if (has_header)
        getline(file, line);

    while (getline(file, line))
    {
        if (line.empty())
            continue;

        const vector<string> tokens = get_tokens(line, separator);

        if (tokens.size() != 2)
            continue;

        input_docs_tokens.push_back(tokenize(tokens[0]));
        target_docs_tokens.push_back(tokenize(tokens[1]));
    }

    if (input_docs_tokens.size() != static_cast<size_t>(get_samples_number()))
    {
        cout << "Warning: Loaded samples count (" << get_samples_number()
        << ") does not match file lines (" << input_docs_tokens.size() << ")." << "\n";
    }

    data.setZero();

    encode_input(input_docs_tokens);

    if (has_decoder_variables)
        encode_decoder_target_sequence_to_sequence(target_docs_tokens);
    else
        encode_target_classification(target_docs_tokens);
}

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
