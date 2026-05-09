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

void copy_padded_tokens(const vector<vector<Index>>& storage,
                        const vector<Index>& sample_indices,
                        Index seq_len,
                        float* out,
                        bool parallelize)
{
    const Index batch_size = sample_indices.size();

    std::fill_n(out, batch_size * seq_len, 0.0f);

    #pragma omp parallel for if (parallelize)
    for (Index i = 0; i < batch_size; ++i)
    {
        const vector<Index>& tokens = storage[sample_indices[i]];
        const Index n = std::min(Index(tokens.size()), seq_len);

        for (Index j = 0; j < n; ++j)
            out[i * seq_len + j] = float(tokens[j]);
    }
}

}

LanguageDataset::LanguageDataset(const filesystem::path& new_data_path, bool new_streaming) : MaterializedDataset()
{
    streaming = new_streaming;
    data_path = new_data_path;
    separator = Dataset::Separator::Tab;

    if (!data_path.empty())
        read_csv();
}

Index LanguageDataset::get_samples_number() const
{
    return streaming ? Index(sample_input_indices.size()) : data.rows();
}

LanguageDataset::LanguageDataset(const Index samples_number,
                                 Index input_sequence_length,
                                 Index input_vocabulary_size) : MaterializedDataset()
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

void LanguageDataset::create_vocabulary(const vector<vector<string_view>>& document_tokens,
                                        vector<string>& vocabulary) const
{
    unordered_map<string_view, size_t> token_count;

    for (const vector<string_view>& document : document_tokens)
        for (string_view token : document)
            ++token_count[token];

    vector<pair<string_view, size_t>> sorted_tokens(token_count.begin(), token_count.end());

    sort(sorted_tokens.begin(), sorted_tokens.end(),
         [](const auto& a, const auto& b) { return a.second > b.second; });

    vocabulary.clear();

    for (const string& token : reserved_tokens)
        vocabulary.push_back(token);

    for (const auto& [token, count] : sorted_tokens)
    {
        if (count < size_t(minimum_token_frequency))
            continue;

        if (find(reserved_tokens.begin(), reserved_tokens.end(), token) != reserved_tokens.end())
            continue;

        if (vocabulary.size() >= size_t(maximum_vocabulary_size))
            break;

        vocabulary.emplace_back(token);
    }
}

void LanguageDataset::encode_input(const vector<vector<string_view>>& input_document_tokens)
{
    const unordered_map<string_view, Index> input_vocabulary_map = create_vocabulary_map(input_vocabulary);
    const Index samples_number = get_samples_number();

    #pragma omp parallel for

    for (Index sample = 0; sample < samples_number; ++sample)
    {
        data(sample, 0) = START_INDEX;

        const vector<string_view>& input_tokens = input_document_tokens[sample];
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

void LanguageDataset::encode_decoder_target_sequence_to_sequence(const vector<vector<string_view>>& target_document_tokens)
{
    const unordered_map<string_view, Index> target_vocabulary_map = create_vocabulary_map(target_vocabulary);
    const Index samples_number = get_samples_number();

    const Index decoder_offset = maximum_input_sequence_length;
    const Index target_offset = maximum_input_sequence_length + maximum_target_sequence_length;

#pragma omp parallel for
    for (Index sample = 0; sample < samples_number; ++sample)
    {
        const vector<string_view>& target_tokens = target_document_tokens[sample];
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

void LanguageDataset::encode_target_classification(const vector<vector<string_view>>& target_document_tokens)
{
    if (maximum_target_sequence_length == 1 && target_vocabulary.size() < 6)
        throw logic_error("Encode target data: Invalid case");

    const size_t target_document_tokens_number = target_document_tokens.size();

    if (maximum_target_sequence_length == 1 && target_vocabulary.size() == 6)
    {
        for (size_t sample_index = 0; sample_index < target_document_tokens_number; ++sample_index)
        {
            const string_view token = target_document_tokens[sample_index][0];

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
        const unordered_map<string_view, Index> target_vocab_map = create_vocabulary_map(target_vocabulary);

        for (size_t sample_index = 0; sample_index < target_document_tokens_number; ++sample_index)
        {
            const string_view token = target_document_tokens[sample_index][0];

            auto iterator = target_vocab_map.find(token);

            const Index token_index = (iterator != target_vocab_map.end())
                                          ? iterator->second
                                          : 1;

            data(sample_index, maximum_input_sequence_length + token_index - reserved_tokens.size()) = 1;
        }

        return;
    }

    {
        const unordered_map<string_view, Index> target_vocabulary_map = create_vocabulary_map(target_vocabulary);

        const Index samples_number = get_samples_number();

        #pragma omp parallel for
        for (Index sample = 0; sample < samples_number; ++sample)
        {
            data(sample, maximum_input_sequence_length) = 2;

            const vector<string_view>& target_tokens = target_document_tokens[sample];

            for (Index variable = maximum_input_sequence_length + 1;
                variable < maximum_input_sequence_length + 1 + Index(target_tokens.size());
                ++variable)
            {
                const Index token_index = variable - (maximum_input_sequence_length + 1);

                const string_view current_token = target_tokens[token_index];

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

    string buffer;
    vector<vector<string_view>> input_document_tokens;
    vector<vector<string_view>> target_document_tokens;

    load_documents(buffer, input_document_tokens, target_document_tokens, false, true);

    const Index samples_number = ssize(input_document_tokens);

    create_vocabulary(input_document_tokens, input_vocabulary);
    create_vocabulary(target_document_tokens, target_vocabulary);

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

        if (!streaming)
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

        if (streaming)
        {
            encode_streaming(input_document_tokens, target_document_tokens);
        }
        else
        {
            encode_input(input_document_tokens);
            encode_target_classification(target_document_tokens);
        }

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

        if (!streaming)
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

        input_shape = { get_maximum_input_sequence_length() };
        decoder_shape = { get_maximum_target_sequence_length() };
        target_shape = { get_maximum_target_sequence_length() };

        if (streaming)
        {
            encode_streaming(input_document_tokens, target_document_tokens);
        }
        else
        {
            encode_input(input_document_tokens);
            encode_decoder_target_sequence_to_sequence(target_document_tokens);
        }
    }

    sample_roles.resize(samples_number);

    set_variable_scalers("None");
    set_default_variable_names();
    split_samples_random();
    if (!streaming)
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

unordered_map<string_view, Index> LanguageDataset::create_vocabulary_map(const vector<string>& vocabulary)
{
    unordered_map<string_view, Index> vocabulary_map;
    vocabulary_map.reserve(vocabulary.size());

    for (Index i = 0; i < Index(vocabulary.size()); ++i)
        vocabulary_map.emplace(string_view(vocabulary[i]), i);

    return vocabulary_map;
}

void LanguageDataset::load_documents(string& buffer,
                                     vector<vector<string_view>>& input_documents,
                                     vector<vector<string_view>>& target_documents,
                                     bool has_header_line,
                                     bool strict_field_count) const
{
    ifstream file(data_path, ios::binary | ios::ate);

    if (!file.is_open())
        throw runtime_error("Cannot open file " + data_path.string());

    const auto file_size = file.tellg();
    file.seekg(0);

    buffer.assign(static_cast<size_t>(file_size), '\0');
    if (file_size > 0)
        file.read(buffer.data(), file_size);
    file.close();

    for (char& c : buffer)
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));

    const string separator_string = get_separator_string();
    const char field_separator = separator_string.empty() ? '\t' : separator_string[0];

    const size_t line_count_estimate = count(buffer.begin(), buffer.end(), '\n') + 1;
    input_documents.reserve(line_count_estimate);
    target_documents.reserve(line_count_estimate);

    const string_view buffer_view(buffer);
    size_t line_start = 0;
    bool header_pending = has_header_line;

    while (line_start < buffer_view.size())
    {
        size_t line_end = buffer_view.find('\n', line_start);
        if (line_end == string_view::npos) line_end = buffer_view.size();

        string_view line = buffer_view.substr(line_start, line_end - line_start);
        line_start = line_end + 1;

        if (!line.empty() && line.back() == '\r') line.remove_suffix(1);
        if (line.empty()) continue;

        if (header_pending) { header_pending = false; continue; }

        const vector<string_view> fields = get_token_views(line, field_separator);

        if (fields.size() != 2)
        {
            if (strict_field_count)
                throw runtime_error("Line must contain two fields: input and target.");
            continue;
        }

        input_documents.push_back(tokenize_views(fields[0]));
        target_documents.push_back(tokenize_views(fields[1]));
    }
}

void LanguageDataset::to_JSON(JsonWriter& printer) const
{
    printer.open_element("Dataset");

    printer.open_element("DataSource");

    write_json(printer, {
        {"FileType", "csv"},
        {"Path", data_path.string()},
        {"Streaming", to_string(streaming)},
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

    streaming = data_source_element->has("Streaming")
        ? read_json_bool(data_source_element, "Streaming")
        : false;

    set_separator_name(read_json_string(data_source_element, "Separator"));
    set_missing_values_label(read_json_string(data_source_element, "MissingValuesLabel"));
    set_codification(read_json_string(data_source_element, "Codification"));
    set_has_header(read_json_bool(data_source_element, "HasHeader"));
    set_has_ids(read_json_bool(data_source_element, "HasSamplesId"));

    if (streaming)
    {
        set_display(read_json_bool(data_set_element, "Display"));
        read_csv();
        return;
    }

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

    string buffer;
    vector<vector<string_view>> input_docs_tokens;
    vector<vector<string_view>> target_docs_tokens;

    load_documents(buffer, input_docs_tokens, target_docs_tokens, has_header, false);

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

void LanguageDataset::encode_streaming(const vector<vector<string_view>>& input_document_tokens,
                                       const vector<vector<string_view>>& target_document_tokens)
{
    const Index samples_number = ssize(input_document_tokens);

    sample_input_indices.assign(samples_number, {});
    sample_target_indices.assign(samples_number, {});

    const unordered_map<string_view, Index> input_vocabulary_map = create_vocabulary_map(input_vocabulary);
    const unordered_map<string_view, Index> target_vocabulary_map = create_vocabulary_map(target_vocabulary);

    #pragma omp parallel for
    for (Index sample = 0; sample < samples_number; ++sample)
    {
        const vector<string_view>& tokens = input_document_tokens[sample];
        vector<Index>& dst = sample_input_indices[sample];

        dst.reserve(tokens.size() + 2);
        dst.push_back(Index(START_INDEX));

        for (size_t i = 0; i < tokens.size(); ++i)
        {
            if (1 + i >= size_t(maximum_input_sequence_length)) break;
            const auto it = input_vocabulary_map.find(tokens[i]);
            dst.push_back(it != input_vocabulary_map.end() ? it->second : Index(UNK_INDEX));
        }

        if (1 + tokens.size() < size_t(maximum_input_sequence_length))
            dst.push_back(Index(END_INDEX));
    }

    const bool has_decoder = !decoder_shape.empty();
    const Index target_vocab_size = ssize(target_vocabulary);

    if (has_decoder)
    {
        #pragma omp parallel for
        for (Index sample = 0; sample < samples_number; ++sample)
        {
            const vector<string_view>& tokens = target_document_tokens[sample];
            vector<Index>& dst = sample_target_indices[sample];

            dst.reserve(tokens.size() + 1);

            for (size_t i = 0; i < tokens.size(); ++i)
            {
                if (i >= size_t(maximum_target_sequence_length)) break;
                const auto it = target_vocabulary_map.find(tokens[i]);
                dst.push_back(it != target_vocabulary_map.end() ? it->second : Index(UNK_INDEX));
            }

            if (tokens.size() < size_t(maximum_target_sequence_length))
                dst.push_back(Index(END_INDEX));
        }
    }
    else if (maximum_target_sequence_length == 1 && target_vocab_size == 6)
    {
        for (Index sample = 0; sample < samples_number; ++sample)
        {
            const string_view token = target_document_tokens[sample][0];

            if (contains(positive_words, token))
                sample_target_indices[sample] = {1};
            else if (contains(negative_words, token))
                sample_target_indices[sample] = {0};
            else
                throw runtime_error("Unknown target value");
        }
    }
    else if (maximum_target_sequence_length == 6 && target_vocab_size >= 6)
    {
        const Index reserved_count = ssize(reserved_tokens);

        for (Index sample = 0; sample < samples_number; ++sample)
        {
            sample_target_indices[sample].assign(maximum_target_sequence_length, 0);

            const string_view token = target_document_tokens[sample][0];
            const auto it = target_vocabulary_map.find(token);
            const Index vocab_index = (it != target_vocabulary_map.end()) ? it->second : Index(UNK_INDEX);
            const Index col = vocab_index - reserved_count;

            if (col >= 0 && col < maximum_target_sequence_length)
                sample_target_indices[sample][col] = 1;
        }
    }
    else
    {
        #pragma omp parallel for
        for (Index sample = 0; sample < samples_number; ++sample)
        {
            const vector<string_view>& tokens = target_document_tokens[sample];
            vector<Index>& dst = sample_target_indices[sample];

            dst.reserve(tokens.size() + 2);
            dst.push_back(Index(START_INDEX));

            for (size_t i = 0; i < tokens.size(); ++i)
            {
                if (1 + i >= size_t(maximum_target_sequence_length)) break;
                const auto it = target_vocabulary_map.find(tokens[i]);
                dst.push_back(it != target_vocabulary_map.end() ? it->second : Index(UNK_INDEX));
            }

            if (1 + tokens.size() < size_t(maximum_target_sequence_length))
                dst.push_back(Index(END_INDEX));
        }
    }
}

void LanguageDataset::fill_inputs(const vector<Index>& sample_indices,
                                  const vector<Index>& input_indices,
                                  float* input_data,
                                  bool parallelize,
                                  int contiguous) const
{
    if (!streaming)
    {
        MaterializedDataset::fill_inputs(sample_indices, input_indices, input_data, parallelize, contiguous);
        return;
    }

    copy_padded_tokens(sample_input_indices, sample_indices, maximum_input_sequence_length, input_data, parallelize);
}

void LanguageDataset::fill_targets(const vector<Index>& sample_indices,
                                   const vector<Index>& target_indices,
                                   float* target_data,
                                   bool parallelize,
                                   int contiguous) const
{
    if (!streaming)
    {
        MaterializedDataset::fill_targets(sample_indices, target_indices, target_data, parallelize, contiguous);
        return;
    }

    copy_padded_tokens(sample_target_indices, sample_indices, maximum_target_sequence_length, target_data, parallelize);
}

void LanguageDataset::fill_decoder(const vector<Index>& sample_indices,
                                   const vector<Index>& decoder_indices,
                                   float* decoder_data,
                                   bool parallelize,
                                   int contiguous) const
{
    if (!streaming)
    {
        MaterializedDataset::fill_decoder(sample_indices, decoder_indices, decoder_data, parallelize, contiguous);
        return;
    }

    const Index batch_size = sample_indices.size();
    const Index seq_len = maximum_target_sequence_length;

    std::fill_n(decoder_data, batch_size * seq_len, 0.0f);

    #pragma omp parallel for if (parallelize)
    for (Index i = 0; i < batch_size; ++i)
    {
        decoder_data[i * seq_len] = START_INDEX;

        const vector<Index>& tokens = sample_target_indices[sample_indices[i]];
        const Index n = std::min(Index(tokens.size()), seq_len - 1);

        for (Index j = 0; j < n; ++j)
            decoder_data[i * seq_len + 1 + j] = float(tokens[j]);
    }
}

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
