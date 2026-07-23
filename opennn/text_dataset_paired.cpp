//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   L A N G U A G E  D A T A S E T   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "text_dataset.h"
#include "string_utilities.h"
#include "tensor_types.h"
#include "io_utilities.h"

namespace opennn::detail
{

class PairedTextDataset final : public TextDataset
{
public:

    PairedTextDataset(const filesystem::path& = "",
                      Index maximum_vocabulary_size = 20000,
                      Index minimum_token_frequency = 1);

    const vector<string>& get_input_vocabulary() const noexcept override { return input_tokenizer->get_vocabulary(); }
    const vector<string>& get_target_vocabulary() const noexcept override { return target_tokenizer->get_vocabulary(); }
    Index get_sequence_length() const noexcept override { return maximum_input_sequence_length; }
    Index get_target_vocabulary_size() const noexcept { return target_tokenizer->get_vocabulary_size(); }
    unique_ptr<TokenizerOperator> clone_input_tokenizer() const override { return input_tokenizer->clone(); }

    void set_input_vocabulary(const vector<string>&);
    void set_target_vocabulary(const vector<string>&);
    void set_maximum_vocabulary_size(Index value) { maximum_vocabulary_size = value; }
    void set_minimum_token_frequency(Index value) { minimum_token_frequency = value; }
    void set_classification_target(bool value) { classification_target = value; }
    Task get_task() const noexcept override
    {
        return classification_target ? Task::Classification : Task::SequenceToSequence;
    }

    Index get_input_sequence_length_limit() const noexcept { return input_sequence_length_limit; }
    void set_input_sequence_length_limit(Index value) { input_sequence_length_limit = value; }

    VectorI calculate_target_distribution() const override;
    void read_txt();
    void from_JSON(const JsonDocument&) override;
    void to_JSON(JsonWriter&) const override;

    void fill_inputs(const vector<Index>&, const vector<Index>&, float*, FillMode, int = -1) const override;
    void fill_targets(const vector<Index>&, const vector<Index>&, float*, FillMode, int = -1) const override;
    void fill_decoder(const vector<Index>&, const vector<Index>&, float*, FillMode, int = -1) const override;
    bool supports_bf16_inputs() const override { return false; }

    static constexpr Index UNK_INDEX = TokenizerOperator::UNK_INDEX;
    static constexpr Index START_INDEX = TokenizerOperator::START_INDEX;
    static constexpr Index END_INDEX = TokenizerOperator::END_INDEX;
    inline static const vector<string> reserved_tokens = {"[PAD]", "[UNK]", "[START]", "[END]"};

private:

    void fill_sequences(const vector<Index>&,
                        const vector<Index>&,
                        float*,
                        int,
                        Index,
                        Index,
                        Index,
                        const char*) const;

    void load_documents(vector<vector<string>>&, vector<vector<string>>&) const;
    void encode_streaming(const vector<vector<string>>&,
                          const vector<vector<string>>&,
                          vector<vector<Index>>&,
                          vector<vector<Index>>&) const;
    void write_binary_cache(const vector<vector<Index>>&, const vector<vector<Index>>&);

    unique_ptr<TokenizerOperator> input_tokenizer = make_unique<WordLevelTokenizer>();
    unique_ptr<TokenizerOperator> target_tokenizer = make_unique<WordLevelTokenizer>();
    Index maximum_input_sequence_length = 0;
    Index maximum_target_sequence_length = 0;
    Index minimum_token_frequency = 1;
    Index maximum_vocabulary_size = 20000;
    bool classification_target = false;
    Index input_sequence_length_limit = 0;
    filesystem::path cache_path;
    mutable FileReader cache_reader;
};

}

namespace opennn
{

unique_ptr<TextDataset> TextDataset::from_classification(
    const filesystem::path& data_path,
    Index maximum_vocabulary_size,
    Index minimum_token_frequency)
{
    auto dataset = make_unique<detail::PairedTextDataset>();
    dataset->set_maximum_vocabulary_size(maximum_vocabulary_size);
    dataset->set_minimum_token_frequency(minimum_token_frequency);
    dataset->set_classification_target(true);
    dataset->set_data_path(data_path);
    dataset->read_txt();
    return dataset;
}

unique_ptr<TextDataset> TextDataset::from_sequence_to_sequence(
    const filesystem::path& data_path,
    Index maximum_vocabulary_size,
    Index minimum_token_frequency)
{
    auto dataset = make_unique<detail::PairedTextDataset>();
    dataset->set_maximum_vocabulary_size(maximum_vocabulary_size);
    dataset->set_minimum_token_frequency(minimum_token_frequency);
    dataset->set_data_path(data_path);
    dataset->read_txt();
    return dataset;
}

}

namespace opennn::detail
{

PairedTextDataset::PairedTextDataset(const filesystem::path& new_data_path,
                                     Index new_maximum_vocabulary_size,
                                     Index new_minimum_token_frequency) : TextDataset(Task::SequenceToSequence)
{
    data_path = new_data_path;
    separator = Dataset::Separator::Tab;
    maximum_vocabulary_size = new_maximum_vocabulary_size;
    minimum_token_frequency = new_minimum_token_frequency;
    storage_mode = StorageMode::BinaryFile;

    if (!data_path.empty())
        read_txt();
}

VectorI PairedTextDataset::calculate_target_distribution() const
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

void PairedTextDataset::read_txt()
{
    cout << "Reading .txt file..." << "\n";

    cache_reader.close();

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
        cout << "[TextDataset] Input sequence length capped from "
             << maximum_input_sequence_length << " to "
             << input_sequence_length_limit
             << " tokens (longer documents are truncated)." << "\n";

        maximum_input_sequence_length = input_sequence_length_limit;
    }

    const Index maximum_target_document_tokens = get_maximum_size(target_document_tokens);
    const Index target_vocabulary_size = get_target_vocabulary_size();

    const bool is_single_token_target = classification_target || (maximum_target_document_tokens == 1);

    const Index target_classes = target_vocabulary_size - Index(reserved_tokens.size());

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

    if (is_single_token_target)
        decoder_shape.clear();
    else
        decoder_shape = { maximum_target_sequence_length };

    input_shape  = { maximum_input_sequence_length };
    target_shape = { maximum_target_sequence_length };

    const bool has_decoder = !decoder_shape.empty();

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

    if (is_single_token_target)
    {
        const vector<string>& target_vocabulary = target_tokenizer->get_vocabulary();
        target_variable.categories.assign(target_vocabulary.begin() + ssize(reserved_tokens),
                                          target_vocabulary.end());
    }

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
        cache_path = cache_directory.empty()
            ? filesystem::path(data_path.string() + ".cache") / "tokens.bin"
            : cache_directory / (data_path.filename().string() + ".cache") / "tokens.bin";

        const uintmax_t record_bytes = uintmax_t(maximum_input_sequence_length
                                               + maximum_target_sequence_length) * sizeof(int32_t);

        const bool cache_valid = binary_cache_is_valid(cache_path, data_path, uintmax_t(samples_number) * record_bytes);

        if (cache_valid)
        {
            cache_reader.open(cache_path);
        }
        else
        {
            vector<vector<Index>> input_indices;
            vector<vector<Index>> target_indices;
            encode_streaming(input_document_tokens, target_document_tokens, input_indices, target_indices);
            write_binary_cache(input_indices, target_indices);
        }
    }

    sample_roles.resize(samples_number);

    split_samples_random();

    cout << "Reading finished" << "\n";
}

void PairedTextDataset::set_input_vocabulary(const vector<string>& new_vocabulary)
{
    input_tokenizer->set_vocabulary(new_vocabulary);
}

void PairedTextDataset::set_target_vocabulary(const vector<string>& new_vocabulary)
{
    target_tokenizer->set_vocabulary(new_vocabulary);
}

void PairedTextDataset::load_documents(vector<vector<string>>& input_documents,
                                     vector<vector<string>>& target_documents) const
{
    const string buffer = read_text_file(data_path);
    const string separator_string = get_separator_string();
    const char field_separator = separator_string.empty() ? '\t' : separator_string[0];

    const size_t line_count_estimate = ranges::count(buffer, '\n') + 1;
    input_documents.reserve(line_count_estimate);
    target_documents.reserve(line_count_estimate);

    const string_view buffer_view(buffer);
    size_t line_start = 0;
    bool header_pending = has_header;

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

        throw_if(fields.size() != 2,
                 "Line must contain two fields: input and target.");

        input_documents.push_back(input_tokenizer->tokenize(fields[0]));

        if (classification_target)
        {
            string_view label = fields[1];
            while (!label.empty() && isspace(static_cast<unsigned char>(label.front()))) label.remove_prefix(1);
            while (!label.empty() && isspace(static_cast<unsigned char>(label.back())))  label.remove_suffix(1);

            string label_string(label);
            for (char& character : label_string)
                character = static_cast<char>(tolower(static_cast<unsigned char>(character)));

            target_documents.push_back({ move(label_string) });
        }
        else
            target_documents.push_back(target_tokenizer->tokenize(fields[1]));
    }
}

void PairedTextDataset::to_JSON(JsonWriter& printer) const
{
    printer.open_element("Dataset");

    printer.open_element("DataSource");

    write_json(printer, {
        {"FileType", "csv"},
        {"Path", data_path.string()},
        {"Separator", get_separator_name()},
        {"HasHeader", to_string(has_header)},
        {"HasSamplesId", to_string(has_sample_ids)},
        {"Codification", get_codification_string()},
        {"StorageMode", get_storage_mode_string()}
    });
    printer.close_element();

    variables_to_JSON(printer);

    samples_to_JSON(printer);

    preview_data_to_JSON(printer);

    const string separator_string = get_separator_string();

    write_json(printer, {
        {"InputVocabulary", vector_to_string(input_tokenizer->get_vocabulary(), separator_string)},
        {"TargetVocabulary", vector_to_string(target_tokenizer->get_vocabulary(), separator_string)},
        {"MaximumInputSequenceLength", to_string(maximum_input_sequence_length)},
        {"MaximumTargetSequenceLength", to_string(maximum_target_sequence_length)},
        {"InputSequenceLengthLimit", to_string(input_sequence_length_limit)},
        {"ClassificationTarget", to_string(classification_target)},
        {"Display", to_string(display)}
    });

    printer.close_element();
}

void PairedTextDataset::from_JSON(const JsonDocument& data_set_document)
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

    if (data_source_element->has("InputSequenceLengthLimit"))
        set_input_sequence_length_limit(read_json_index(data_source_element, "InputSequenceLengthLimit"));
    else if (data_set_element->has("InputSequenceLengthLimit"))
        set_input_sequence_length_limit(read_json_index(data_set_element, "InputSequenceLengthLimit"));

    if (data_set_element->has("ClassificationTarget"))
        set_classification_target(read_json_bool(data_set_element, "ClassificationTarget"));

    read_txt();
}

void PairedTextDataset::encode_streaming(const vector<vector<string>>& input_document_tokens,
                                       const vector<vector<string>>& target_document_tokens,
                                       vector<vector<Index>>& input_indices,
                                       vector<vector<Index>>& target_indices) const
{
    const Index samples_number = ssize(input_document_tokens);

    input_indices.assign(samples_number, {});
    target_indices.assign(samples_number, {});

    const unordered_map<string, Index>& target_vocabulary_map = target_tokenizer->get_vocabulary_map();

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
                     format("Unknown target label: {}", token));

            target_indices[sample][it->second - reserved_count] = 1;
        }
    }
}

void PairedTextDataset::write_binary_cache(const vector<vector<Index>>& input_indices,
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

void PairedTextDataset::fill_sequences(const vector<Index>& sample_indices,
                                     const vector<Index>& variable_indices,
                                     float* output_data,
                                     int contiguous,
                                     Index stream_length,
                                     Index record_offset,
                                     Index shift,
                                     const char* context) const
{
    if (storage_mode == StorageMode::Matrix)
    {
        fill_tensor_data(data, sample_indices, variable_indices, output_data, contiguous);
        return;
    }

    const Index batch_size = ssize(sample_indices);
    const Index samples_number = get_samples_number();
    const uint64_t record_tokens = uint64_t(maximum_input_sequence_length + maximum_target_sequence_length);
    const Index n = stream_length - shift;

    string omp_error;

    #pragma omp parallel for
    for (Index i = 0; i < batch_size; ++i)
    {
        try
        {
            if (shift > 0) output_data[i * stream_length] = float(START_INDEX);

            const Index sample_index = sample_indices[size_t(i)];
            throw_if(sample_index < 0 || sample_index >= samples_number,
                     format("TextDataset {} sample index is out of range.", context));

            thread_local vector<int32_t> buf;
            buf.resize(size_t(n));
            cache_reader.read_at(buf.data(), size_t(n) * sizeof(int32_t),
                                 (uint64_t(sample_index) * record_tokens + uint64_t(record_offset)) * sizeof(int32_t));

            for (Index j = 0; j < n; ++j)
                output_data[i * stream_length + shift + j] = float(buf[size_t(j)]);
        }
        catch (const exception& e)
        {
            #pragma omp critical
            { omp_error = e.what(); }
        }
    }

    throw_if(!omp_error.empty(),
             omp_error);
}

void PairedTextDataset::fill_inputs(const vector<Index>& sample_indices,
                                  const vector<Index>& input_indices,
                                  float* input_data,
                                  FillMode,
                                  int contiguous) const
{
    fill_sequences(sample_indices, input_indices, input_data, contiguous,
                   maximum_input_sequence_length, 0, 0, "input");
}

void PairedTextDataset::fill_targets(const vector<Index>& sample_indices,
                                   const vector<Index>& target_indices,
                                   float* target_data,
                                   FillMode,
                                   int contiguous) const
{
    fill_sequences(sample_indices, target_indices, target_data, contiguous,
                   maximum_target_sequence_length, maximum_input_sequence_length, 0, "target");
}

void PairedTextDataset::fill_decoder(const vector<Index>& sample_indices,
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
