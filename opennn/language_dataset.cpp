//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   L A N G U A G E  D A T A S E T   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "language_dataset.h"
#include "string_utilities.h"
#include "tensor_types.h"
#include "io_utilities.h"

namespace opennn
{

static constexpr array<char, 8> LANGUAGE_CACHE_MAGIC{'O', 'N', 'N', 'L', 'A', 'N', 'G', '3'};
static constexpr uint32_t LANGUAGE_CACHE_VERSION = 1;

LanguageDataset::LanguageDataset(const filesystem::path& new_data_path,
                                 Index new_maximum_vocabulary_size,
                                 Index new_minimum_token_frequency) : Dataset()
{
    data_path = new_data_path;
    separator = Dataset::Separator::Tab;
    maximum_vocabulary_size = new_maximum_vocabulary_size;
    minimum_token_frequency = new_minimum_token_frequency;
    storage_mode = StorageMode::BinaryFile;

    if (!data_path.empty())
        read_txt();
}

VectorI LanguageDataset::calculate_target_distribution() const
{
    if (!decoder_shape.empty()) return {};

    const Index samples_number = get_samples_number();
    const Index targets_number = maximum_target_sequence_length;
    const uint64_t record_tokens = uint64_t(maximum_input_sequence_length + targets_number);

    VectorI distribution = VectorI::Zero(targets_number == 1 ? 2 : targets_number);

    vector<int32_t> tokens(size_t(targets_number), 0);

    for (Index sample = 0; sample < samples_number; ++sample)
    {
        if (storage_mode == StorageMode::Matrix)
            for (Index j = 0; j < targets_number; ++j)
                tokens[size_t(j)] = int32_t(data(sample, maximum_input_sequence_length + j));
        else
            cache_reader.read_at(tokens.data(), size_t(targets_number) * sizeof(int32_t),
                                 (uint64_t(sample) * record_tokens
                                  + uint64_t(maximum_input_sequence_length)) * sizeof(int32_t));

        if (targets_number == 1)
            (tokens[0] < 1) ? distribution(0)++ : distribution(1)++;
        else
            for (Index j = 0; j < targets_number; ++j)
                if (tokens[size_t(j)] == 1) { distribution(j)++; break; }
    }

    return distribution;
}

void LanguageDataset::read_txt()
{
    cout << "Reading .txt file..." << "\n";

    cache_reader.close();

    const filesystem::path cache_parent = cache_directory.empty()
        ? filesystem::path(data_path.string() + ".cache")
        : cache_directory / (data_path.filename().string() + ".cache");

    cache_path = cache_parent / format(
        "tokens_v3_{}_{}_{}_{}_{}_{}_{}.bin",
        maximum_vocabulary_size,
        minimum_token_frequency,
        input_sequence_length_limit,
        classification_target,
        get_separator_name(),
        has_header,
        has_sample_ids);
    const filesystem::path metadata_path = cache_path.string() + ".meta";

    if (storage_mode == StorageMode::BinaryFile
        && is_file_current(cache_path, {data_path})
        && is_file_current(metadata_path, {data_path})
        && load_cache_metadata(metadata_path))
    {
        split_samples_random();
        cout << "Reading finished (cached)" << "\n";
        return;
    }

    vector<vector<string>> input_document_tokens;
    vector<vector<string>> target_document_tokens;

    load_documents(input_document_tokens, target_document_tokens);

    auto get_maximum_size = [](const auto& nested_values) {
        const auto it = ranges::max_element(nested_values,
                                            [](const auto& a, const auto& b) { return a.size() < b.size(); });
        return it == nested_values.end() ? size_t(0) : it->size();
    };

    const Index samples_number = ssize(input_document_tokens);

    input_tokenizer->build_vocabulary(input_document_tokens, maximum_vocabulary_size, minimum_token_frequency);
    target_tokenizer->build_vocabulary(target_document_tokens, maximum_vocabulary_size, minimum_token_frequency);

    maximum_input_sequence_length = get_maximum_size(input_document_tokens) + 2;

    if (input_sequence_length_limit > 0
     && maximum_input_sequence_length > input_sequence_length_limit)
    {
        cout << "[LanguageDataset] Input sequence length capped from "
             << maximum_input_sequence_length << " to "
             << input_sequence_length_limit
             << " tokens (longer documents are truncated)." << "\n";

        maximum_input_sequence_length = input_sequence_length_limit;
    }

    const Index maximum_target_document_tokens = get_maximum_size(target_document_tokens);
    const Index target_vocabulary_size = get_target_vocabulary_size();

    // Classification mode forces atomic (single-token) targets; a token-per-word
    // target with one token per sample is also treated as classification.
    const bool is_single_token_target = classification_target || (maximum_target_document_tokens == 1);

    // For single-token classification the target vocabulary is
    // [4 reserved tokens] + [one entry per class]. A binary problem (2 classes)
    // is encoded as a single output (probability of the positive class); an
    // N-class problem (N >= 3) as N one-hot outputs.
    const Index target_classes = target_vocabulary_size - Index(reserved_tokens.size());

    // Binary classification: both the 0/1 encoding (encode_streaming) and the
    // positive-class display name follow the vocabulary order (index 4 =
    // negative, index 5 = positive). build_vocabulary orders by frequency, so
    // when the labels have a recognizable polarity (positive_words /
    // negative_words) reorder them semantically; otherwise the frequency order
    // stands and stays consistent between encoding and naming.
    if (is_single_token_target && target_classes == 2)
    {
        vector<string> target_vocabulary = target_tokenizer->get_vocabulary();
        const size_t reserved_count = reserved_tokens.size();

        if (contains(positive_words, target_vocabulary[reserved_count])
         || contains(negative_words, target_vocabulary[reserved_count + 1]))
        {
            swap(target_vocabulary[reserved_count], target_vocabulary[reserved_count + 1]);
            target_tokenizer->set_vocabulary(target_vocabulary);
        }
    }

    maximum_target_sequence_length = is_single_token_target
        ? (target_classes == 2 ? 1 : target_classes)
        : maximum_target_document_tokens + 1;

    const bool has_decoder = !is_single_token_target;
    configure(samples_number, has_decoder);

    if (storage_mode == StorageMode::Matrix)
    {
        vector<vector<Index>> input_indices;
        vector<vector<Index>> target_indices;
        encode_streaming(input_document_tokens, target_document_tokens, input_indices, target_indices);

        const Index decoder_offset = maximum_input_sequence_length;
        const Index target_offset = has_decoder
            ? decoder_offset + maximum_target_sequence_length
            : maximum_input_sequence_length;

        data.resize(samples_number, get_features_number());
        data.setZero();

        for (Index i = 0; i < samples_number; ++i)
        {
            const vector<Index>& in = input_indices[size_t(i)];
            const Index in_n = min(ssize(in), maximum_input_sequence_length);
            for (Index j = 0; j < in_n; ++j)
                data(i, j) = float(in[size_t(j)]);

            const vector<Index>& tgt = target_indices[size_t(i)];
            const Index tgt_n = min(ssize(tgt), maximum_target_sequence_length);
            for (Index j = 0; j < tgt_n; ++j)
                data(i, target_offset + j) = float(tgt[size_t(j)]);

            if (has_decoder)
            {
                data(i, decoder_offset) = float(START_INDEX);
                const Index dec_n = min(ssize(tgt), maximum_target_sequence_length - 1);
                for (Index j = 0; j < dec_n; ++j)
                    data(i, decoder_offset + 1 + j) = float(tgt[size_t(j)]);
            }
        }
    }
    else
    {
        vector<vector<Index>> input_indices;
        vector<vector<Index>> target_indices;
        encode_streaming(input_document_tokens, target_document_tokens, input_indices, target_indices);
        write_binary_cache(input_indices, target_indices);
        save_cache_metadata(metadata_path, samples_number, has_decoder);
    }

    split_samples_random();

    cout << "Reading finished" << "\n";
}

void LanguageDataset::set_input_vocabulary(const vector<string>& new_vocabulary)
{
    input_tokenizer->set_vocabulary(new_vocabulary);
}

void LanguageDataset::set_target_vocabulary(const vector<string>& new_vocabulary)
{
    target_tokenizer->set_vocabulary(new_vocabulary);
}

void LanguageDataset::configure(Index samples_number, bool has_decoder)
{
    input_shape = {maximum_input_sequence_length};
    target_shape = {maximum_target_sequence_length};
    decoder_shape = has_decoder ? Shape{maximum_target_sequence_length} : Shape{};

    variables.assign(has_decoder ? 3 : 2, Variable());

    Variable& input_variable = variables[0];
    input_variable.name = "input_sequence";
    input_variable.role = VariableRole::Input;
    input_variable.type = VariableType::Numeric;
    input_variable.features = maximum_input_sequence_length;
    input_variable.categories = input_tokenizer->get_vocabulary();

    if (has_decoder)
    {
        Variable& decoder_variable = variables[1];
        decoder_variable.name = "decoder_sequence";
        decoder_variable.role = VariableRole::Decoder;
        decoder_variable.type = VariableType::Numeric;
        decoder_variable.features = maximum_target_sequence_length;
    }

    Variable& target_variable = variables[has_decoder ? 2 : 1];
    target_variable.name = "target_sequence";
    target_variable.role = VariableRole::Target;
    target_variable.type = VariableType::Numeric;
    target_variable.features = maximum_target_sequence_length;

    if (!has_decoder)
    {
        const vector<string>& target_vocabulary = target_tokenizer->get_vocabulary();
        throw_if(target_vocabulary.size() < reserved_tokens.size(),
                 "LanguageDataset: cached target vocabulary is incomplete.");
        target_variable.categories.assign(
            target_vocabulary.begin() + ssize(reserved_tokens),
            target_vocabulary.end());
    }

    sample_roles.resize(size_t(samples_number));
}

bool LanguageDataset::load_cache_metadata(const filesystem::path& metadata_path)
{
    ifstream file(metadata_path, ios::binary);
    if (!file) return false;

    array<char, 8> magic{};
    uint32_t version = 0;
    int64_t input_length = 0;
    int64_t target_length = 0;
    int64_t samples_number = 0;
    uint8_t has_decoder = 0;
    uint64_t input_vocabulary_size = 0;
    uint64_t target_vocabulary_size = 0;

    if (!file.read(magic.data(), magic.size())
        || !read_binary_value(file, version)
        || !read_binary_value(file, input_length)
        || !read_binary_value(file, target_length)
        || !read_binary_value(file, samples_number)
        || !read_binary_value(file, has_decoder)
        || !read_binary_value(file, input_vocabulary_size)
        || !read_binary_value(file, target_vocabulary_size)
        || magic != LANGUAGE_CACHE_MAGIC
        || version != LANGUAGE_CACHE_VERSION
        || input_length <= 0
        || target_length <= 0
        || samples_number <= 0
        || input_vocabulary_size > uint64_t(numeric_limits<Index>::max())
        || target_vocabulary_size > uint64_t(numeric_limits<Index>::max()))
        return false;

    vector<string> input_vocabulary(static_cast<size_t>(input_vocabulary_size));
    vector<string> target_vocabulary(static_cast<size_t>(target_vocabulary_size));

    for (string& token : input_vocabulary)
        if (!read_binary_string(file, token)) return false;
    for (string& token : target_vocabulary)
        if (!read_binary_string(file, token)) return false;

    maximum_input_sequence_length = Index(input_length);
    maximum_target_sequence_length = Index(target_length);
    input_tokenizer->set_vocabulary(input_vocabulary);
    target_tokenizer->set_vocabulary(target_vocabulary);

    const uintmax_t expected_bytes =
        uintmax_t(samples_number)
        * uintmax_t(input_length + target_length)
        * sizeof(int32_t);
    error_code error;
    if (filesystem::file_size(cache_path, error) != expected_bytes || error)
        return false;

    configure(Index(samples_number), has_decoder != 0);
    cache_reader.open(cache_path);
    return true;
}

void LanguageDataset::save_cache_metadata(const filesystem::path& metadata_path,
                                          Index samples_number,
                                          bool has_decoder) const
{
    FileWriter writer;
    writer.open(metadata_path.string() + ".tmp");

    writer.write(LANGUAGE_CACHE_MAGIC.data(), LANGUAGE_CACHE_MAGIC.size());
    write_binary_value(writer, LANGUAGE_CACHE_VERSION);
    write_binary_value(writer, int64_t(maximum_input_sequence_length));
    write_binary_value(writer, int64_t(maximum_target_sequence_length));
    write_binary_value(writer, int64_t(samples_number));
    write_binary_value(writer, uint8_t(has_decoder));

    const vector<string>& input_vocabulary = input_tokenizer->get_vocabulary();
    const vector<string>& target_vocabulary = target_tokenizer->get_vocabulary();
    write_binary_value(writer, uint64_t(input_vocabulary.size()));
    write_binary_value(writer, uint64_t(target_vocabulary.size()));

    for (const string& token : input_vocabulary)
        write_binary_string(writer, token);
    for (const string& token : target_vocabulary)
        write_binary_string(writer, token);

    writer.finish_with_rename(metadata_path);
}

void LanguageDataset::load_documents(vector<vector<string>>& input_documents,
                                     vector<vector<string>>& target_documents) const
{
    const string separator_string = get_separator_string();
    const char field_separator = separator_string.empty() ? '\t' : separator_string[0];

    CsvReader reader({field_separator, {}});
    CsvReader::Result result = reader.read(data_path);

    const size_t first_line = has_header ? 1 : 0;
    const size_t documents_number = result.lines.size() - min(first_line, result.lines.size());

    input_documents.resize(documents_number);
    target_documents.resize(documents_number);
    vector<unsigned char> valid(documents_number, 1);

    #pragma omp parallel if(documents_number >= 256)
    {
        string scratch;
        vector<string_view> fields;

        #pragma omp for schedule(static)
        for (Index document = 0; document < Index(documents_number); ++document)
        {
            get_token_views_maybe_quoted(result.lines[first_line + size_t(document)],
                                         field_separator,
                                         result.has_quotes,
                                         scratch,
                                         fields);

            if (fields.size() != 2)
            {
                valid[size_t(document)] = 0;
                continue;
            }

            input_documents[size_t(document)] = input_tokenizer->tokenize(fields[0]);

            if (classification_target)
            {
                target_documents[size_t(document)] = {
                    ascii_lowercase(trim_view(fields[1]))
                };
            }
            else
            {
                target_documents[size_t(document)] = target_tokenizer->tokenize(fields[1]);
            }
        }
    }

    const auto invalid = ranges::find(valid, 0);
    throw_if(invalid != valid.end(),
             "Line {} must contain exactly two fields: input and target.",
             first_line + size_t(distance(valid.begin(), invalid)) + 1);
}

void LanguageDataset::to_JSON(JsonWriter& printer) const
{
    write_json_header(printer, {
        {"FileType", "csv"},
        {"Path", data_path.string()},
        {"Separator", get_separator_name()},
        {"HasHeader", has_header},
        {"HasSamplesId", has_sample_ids},
        {"Codification", get_codification_string()},
        {"StorageMode", get_storage_mode_string()}
    });

    preview_data_to_JSON(printer);

    write_json(printer, {
        {"InputVocabulary", json_array(input_tokenizer->get_vocabulary())},
        {"TargetVocabulary", json_array(target_tokenizer->get_vocabulary())},
        {"MaximumInputSequenceLength", maximum_input_sequence_length},
        {"MaximumTargetSequenceLength", maximum_target_sequence_length},
        {"InputSequenceLengthLimit", input_sequence_length_limit},
        {"ClassificationTarget", classification_target}
    });

    write_json_footer(printer);
}

void LanguageDataset::from_JSON(const JsonDocument& data_set_document)
{
    const Json* data_set_element = get_json_root(data_set_document, "Dataset");

    const Json* data_source_element = require_json_field(data_set_element, "DataSource");

    set_data_path(read_json_string(data_source_element, "Path"));


    set_separator_name(read_json_string(data_source_element, "Separator"));
    set_codification(read_json_string(data_source_element, "Codification"));
    set_storage_mode(data_source_element->has("StorageMode")
                   ? read_json_string(data_source_element, "StorageMode")
                   : "BinaryFile");
    set_has_header(read_json_bool(data_source_element, "HasHeader"));
    set_has_ids(read_json_bool(data_source_element, "HasSamplesId"));

    set_display(read_json_bool(data_set_element, "Display"));

    // Both fields must be restored BEFORE read_txt: they change how the file is
    // tokenized (atomic class labels) and the resulting sequence length (cap).
    //
    // The truncation limit can arrive in two places: the editor writes the
    // user's import-time choice inside DataSource, while this class's own
    // to_JSON echoes it at the Dataset level. The editor's (fresher) intent
    // wins over the echo.
    if (data_source_element->has("InputSequenceLengthLimit"))
        set_input_sequence_length_limit(read_json_index(data_source_element, "InputSequenceLengthLimit"));
    else if (data_set_element->has("InputSequenceLengthLimit"))
        set_input_sequence_length_limit(read_json_index(data_set_element, "InputSequenceLengthLimit"));

    if (data_set_element->has("ClassificationTarget"))
        set_classification_target(read_json_bool(data_set_element, "ClassificationTarget"));

    read_txt();
}

void LanguageDataset::encode_streaming(const vector<vector<string>>& input_document_tokens,
                                       const vector<vector<string>>& target_document_tokens,
                                       vector<vector<Index>>& input_indices,
                                       vector<vector<Index>>& target_indices) const
{
    const Index samples_number = ssize(input_document_tokens);

    input_indices.assign(samples_number, {});
    target_indices.assign(samples_number, {});

    const auto& target_vocabulary_map = target_tokenizer->get_vocabulary_map();

    #pragma omp parallel for
    for (Index sample = 0; sample < samples_number; ++sample)
        input_indices[sample] = input_tokenizer->encode_sequence(input_document_tokens[sample],
                                                                 maximum_input_sequence_length);

    const bool has_decoder = !decoder_shape.empty();
    const Index target_vocab_size = target_tokenizer->get_vocabulary_size();

    if (has_decoder)
    {
        #pragma omp parallel for
        for (Index sample = 0; sample < samples_number; ++sample)
        {
            const vector<string>& tokens = target_document_tokens[sample];
            vector<Index>& destination = target_indices[sample];

            destination.reserve(tokens.size() + 1);

            for (size_t i = 0; i < tokens.size(); ++i)
            {
                if (i >= size_t(maximum_target_sequence_length)) break;
                const auto it = target_vocabulary_map.find(tokens[i]);
                destination.push_back(it != target_vocabulary_map.end() ? it->second : UNK_INDEX);
            }

            if (tokens.size() < size_t(maximum_target_sequence_length))
                destination.push_back(END_INDEX);
        }
    }
    else if (maximum_target_sequence_length == 1
          && target_vocab_size == ssize(reserved_tokens) + 2)
    {
        // Binary classification: one output = P(positive class). The vocabulary
        // order defines the encoding (index 4 -> 0, index 5 -> 1); read_txt
        // places the semantically positive label at index 5 when the labels'
        // polarity is recognizable, and the display-name resolution assumes the
        // same order. Works for arbitrary label pairs (e.g. spam/ham), unlike
        // the previous positive_words/negative_words lookup.
        const vector<string>& target_vocabulary = target_tokenizer->get_vocabulary();
        const size_t reserved_count = reserved_tokens.size();

        for (Index sample = 0; sample < samples_number; ++sample)
        {
            const vector<string>& sample_tokens = target_document_tokens[sample];
            throw_if(sample_tokens.empty(), "Empty target value");

            const string_view token = sample_tokens[0];

            if (token == target_vocabulary[reserved_count + 1])
                target_indices[sample] = {1};
            else if (token == target_vocabulary[reserved_count])
                target_indices[sample] = {0};
            else
                throw runtime_error(format("Unknown binary target label: {}", string(token)));
        }
    }
    else
    {
        // One-hot single-token targets: one column per non-reserved vocabulary entry.
        const Index reserved_count = ssize(reserved_tokens);

        throw_if(maximum_target_sequence_length != target_vocab_size - reserved_count,
                 "Unsupported target encoding: expected one column per target class.");

        for (Index sample = 0; sample < samples_number; ++sample)
        {
            target_indices[sample].assign(maximum_target_sequence_length, 0);

            const vector<string>& sample_tokens = target_document_tokens[sample];
            throw_if(sample_tokens.empty(), "Empty target value");

            const string& token = sample_tokens[0];
            const auto it = target_vocabulary_map.find(token);

            throw_if(it == target_vocabulary_map.end() || it->second < reserved_count,
                     "Unknown target label: {}", token);

            target_indices[sample][it->second - reserved_count] = 1;
        }
    }
}

void LanguageDataset::write_binary_cache(const vector<vector<Index>>& input_indices,
                                         const vector<vector<Index>>& target_indices)
{
    const Index samples_number = ssize(input_indices);
    const Index record_tokens = maximum_input_sequence_length + maximum_target_sequence_length;

    filesystem::create_directories(cache_path.parent_path());
    const filesystem::path tmp_path = cache_path.string() + ".tmp";

    FileWriter writer;
    writer.open(tmp_path);

    vector<int32_t> record(size_t(record_tokens), 0);

    for (Index i = 0; i < samples_number; ++i)
    {
        ranges::fill(record, 0);

        const vector<Index>& in = input_indices[size_t(i)];
        const Index in_n = min(ssize(in), maximum_input_sequence_length);
        for (Index j = 0; j < in_n; ++j)
            record[size_t(j)] = int32_t(in[size_t(j)]);

        const vector<Index>& tgt = target_indices[size_t(i)];
        const Index tgt_n = min(ssize(tgt), maximum_target_sequence_length);
        for (Index j = 0; j < tgt_n; ++j)
            record[size_t(maximum_input_sequence_length + j)] = int32_t(tgt[size_t(j)]);

        writer.write(record.data(), record.size() * sizeof(int32_t));
    }

    writer.finish_with_rename(cache_path);

    cache_reader.open(cache_path);
}

void LanguageDataset::fill_sequences(const vector<Index>& sample_indices,
                                     const vector<Index>& variable_indices,
                                     float* output_data,
                                     int contiguous,
                                     Index sequence_length,
                                     Index record_offset,
                                     Index shift,
                                     const char* context) const
{
    if (storage_mode == StorageMode::Matrix)
    {
        fill_tensor_data(data, sample_indices, variable_indices, output_data, contiguous);
        return;
    }

    const uint64_t record_tokens = uint64_t(maximum_input_sequence_length + maximum_target_sequence_length);
    const Index n = sequence_length - shift;

    if (shift > 0)
        for (Index i = 0; i < ssize(sample_indices); ++i)
            output_data[i * sequence_length] = float(START_INDEX);

    read_int32_batch(cache_reader,
                     sample_indices,
                     get_samples_number(),
                     record_tokens,
                     record_offset,
                     n,
                     output_data,
                     sequence_length,
                     shift,
                     format("LanguageDataset {}", context));
}

void LanguageDataset::fill_inputs(const vector<Index>& sample_indices,
                                  const vector<Index>& input_indices,
                                  float* input_data,
                                  FillMode,
                                  int contiguous) const
{
    fill_sequences(sample_indices, input_indices, input_data, contiguous,
                   maximum_input_sequence_length, 0, 0, "input");
}

void LanguageDataset::fill_targets(const vector<Index>& sample_indices,
                                   const vector<Index>& target_indices,
                                   float* target_data,
                                   FillMode,
                                   int contiguous) const
{
    fill_sequences(sample_indices, target_indices, target_data, contiguous,
                   maximum_target_sequence_length, maximum_input_sequence_length, 0, "target");
}

void LanguageDataset::fill_decoder(const vector<Index>& sample_indices,
                                   const vector<Index>& decoder_indices,
                                   float* decoder_data,
                                   FillMode,
                                   int contiguous) const
{
    fill_sequences(sample_indices, decoder_indices, decoder_data, contiguous,
                   maximum_target_sequence_length, maximum_input_sequence_length, 1, "decoder");
}

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
