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
#include "io_utilities.h"

#include <fstream>

namespace opennn
{

namespace {

#pragma pack(push, 1)
struct LangCacheHeader
{
    char     magic[8];          // "OPENNNTK" (8 B)
    uint32_t version;           //           (4 B)
    uint64_t num_samples;       //           (8 B)
    uint32_t input_max_len;     //           (4 B)
    uint32_t target_max_len;    //           (4 B)
    uint8_t  has_decoder;       //           (1 B)
    uint8_t  pad[35];           // pad to 64 (35 B)
};                              // total = 64
#pragma pack(pop)
static_assert(sizeof(LangCacheHeader) == 64, "LangCacheHeader must be 64 bytes");

constexpr uint32_t LANG_CACHE_VERSION = 1;
constexpr const char LANG_CACHE_MAGIC[8] = {'O','P','E','N','N','N','T','K'};

const vector<string> positive_words = {"1", "yes", "positive", "+", "true", "good", "si", "sí", "Sí"};
const vector<string> negative_words = {"0", "no", "negative", "-", "false", "bad", "not", "No"};

}

LanguageDataset::LanguageDataset(const filesystem::path& new_data_path) : Dataset()
{
    data_path = new_data_path;
    separator = Dataset::Separator::Tab;

    if (!data_path.empty())
        read_txt();
}

Index LanguageDataset::get_samples_number() const
{
    return Index(offsets_table.size());
}

LanguageDataset::LanguageDataset(const filesystem::path& new_data_path,
                                 Index new_maximum_vocabulary_size) : Dataset()
{
    data_path = new_data_path;
    separator = Dataset::Separator::Tab;
    maximum_vocabulary_size = new_maximum_vocabulary_size;

    if (!data_path.empty())
        read_txt();
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
        read_txt();
}

VectorI LanguageDataset::calculate_target_distribution() const
{
    if (maximum_target_sequence_length != 1) return {};

    const Index samples_number = get_samples_number();

    Index positives = 0;
    Index negatives = 0;

    for (Index sample = 0; sample < samples_number; ++sample)
    {
        const auto& offsets = offsets_table[size_t(sample)];
        const int64_t tgt_off = offsets[2];
        const int64_t tgt_len = offsets[3];
        if (tgt_len < 1) continue;

        int32_t token = 0;
        cache_reader.read_at(&token, sizeof(token),
                             cache_data_offset + uint64_t(tgt_off) * sizeof(int32_t));
        (token < 1) ? negatives++ : positives++;
    }

    VectorI distribution(2);
    distribution(0) = negatives;
    distribution(1) = positives;
    return distribution;
}

void LanguageDataset::create_vocabulary(const vector<vector<string_view>>& document_tokens,
                                        vector<string>& vocabulary) const
{
    unordered_map<string_view, size_t> token_count;

    for (const vector<string_view>& document : document_tokens)
        for (string_view token : document)
            ++token_count[token];

    vector<pair<string_view, size_t>> sorted_tokens(token_count.begin(), token_count.end());

    ranges::sort(sorted_tokens,
                 [](const auto& a, const auto& b) { return a.second > b.second; });

    vocabulary = reserved_tokens;

    for (const auto& [token, count] : sorted_tokens)
    {
        if (count < size_t(minimum_token_frequency))
            continue;

        if (ranges::find(reserved_tokens, token) != reserved_tokens.end())
            continue;

        if (vocabulary.size() >= size_t(maximum_vocabulary_size))
            break;

        vocabulary.emplace_back(token);
    }
}


void LanguageDataset::read_txt()
{
    cout << "Reading .txt file..." << "\n";

    cache_path = filesystem::path(data_path.string() + ".cache") / "tokens.bin";

    string buffer;
    vector<vector<string_view>> input_document_tokens;
    vector<vector<string_view>> target_document_tokens;

    load_documents(buffer, input_document_tokens, target_document_tokens, false, true);

    auto get_maximum_size = [](const auto& nested_values) {
        const auto it = ranges::max_element(nested_values,
                                            [](const auto& a, const auto& b) { return a.size() < b.size(); });
        return it == nested_values.end() ? size_t(0) : it->size();
    };

    const Index samples_number = ssize(input_document_tokens);

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

        variables.resize(features_number);

        ranges::for_each(variables | views::take(maximum_input_sequence_length),
                         [](Variable& variable) { variable.role = VariableRole::Input; });

        ranges::for_each(variables
                         | views::drop(maximum_input_sequence_length)
                         | views::take(maximum_target_sequence_length),
                         [](Variable& variable) { variable.role = VariableRole::Target; });

        if (!variables.empty())
            variables[0].categories = input_vocabulary;

        for (Index i = 0; i < maximum_input_sequence_length; ++i)
            variables[i].name = format("token_{}", i + 1);

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
                                      + 2 * maximum_target_sequence_length;

        variables.resize(features_number);

        ranges::for_each(variables | views::take(maximum_input_sequence_length),
                         [](Variable& variable) { variable.role = VariableRole::Input; });

        ranges::for_each(variables
                         | views::drop(decoder_offset)
                         | views::take(maximum_target_sequence_length),
                         [](Variable& variable) { variable.role = VariableRole::Decoder; });

        ranges::for_each(variables
                         | views::drop(target_offset)
                         | views::take(maximum_target_sequence_length),
                         [](Variable& variable) { variable.role = VariableRole::Target; });

        if (!variables.empty())
            variables[0].categories = input_vocabulary;

        for (Index i = 0; i < maximum_input_sequence_length; ++i)
            variables[i].name = format("input_token_{}", i + 1);

        for (Index i = 0; i < maximum_target_sequence_length; ++i)
            variables[decoder_offset + i].name = format("decoder_token_{}", i + 1);

        for (Index i = 0; i < maximum_target_sequence_length; ++i)
            variables[target_offset + i].name = format("target_token_{}", i + 1);

        input_shape = { get_maximum_input_sequence_length() };
        decoder_shape = { get_maximum_target_sequence_length() };
        target_shape = { get_maximum_target_sequence_length() };
    }

    if (!try_load_binary_cache(samples_number))
    {
        vector<vector<Index>> input_indices;
        vector<vector<Index>> target_indices;
        encode_streaming(input_document_tokens, target_document_tokens, input_indices, target_indices);
        write_binary_cache(input_indices, target_indices, /*has_decoder=*/!decoder_shape.empty());
    }

    sample_roles.resize(samples_number);

    set_default_variable_names();
    split_samples_random();

    for (Index i = 0; i < ssize(variables); ++i)
    {
        Variable& variable = variables[i];

        if (i < maximum_input_sequence_length)
            variable.role = VariableRole::Input;
        else if (!decoder_shape.empty() && i < maximum_input_sequence_length + maximum_target_sequence_length)
            variable.role = VariableRole::Decoder;
        else
            variable.role = VariableRole::Target;

        variable.type = VariableType::Numeric;
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

unordered_map<string_view, Index> LanguageDataset::create_vocabulary_map(const vector<string>& vocabulary) const
{
    unordered_map<string_view, Index> vocabulary_map;
    vocabulary_map.reserve(vocabulary.size());

    for (Index i = 0; i < Index(vocabulary.size()); ++i)
        vocabulary_map.emplace(string_view(vocabulary[i]), i);

    return vocabulary_map;
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

void LanguageDataset::load_documents(string& buffer,
                                     vector<vector<string_view>>& input_documents,
                                     vector<vector<string_view>>& target_documents,
                                     bool has_header_line,
                                     bool strict_field_count) const
{
    ifstream file(data_path, ios::binary | ios::ate);

    if (!file.is_open())
        throw runtime_error(format("Cannot open file {}", data_path.string()));

    const auto file_size = file.tellg();
    file.seekg(0);

    buffer.assign(static_cast<size_t>(file_size), '\0');
    if (file_size > 0)
        file.read(buffer.data(), file_size);
    file.close();

    for (char& c : buffer)
        c = static_cast<char>(tolower(static_cast<unsigned char>(c)));

    const string separator_string = get_separator_string();
    const char field_separator = separator_string.empty() ? '\t' : separator_string[0];

    const size_t line_count_estimate = ranges::count(buffer, '\n') + 1;
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
        {"Separator", get_separator_name()},
        {"HasHeader", to_string(has_header)},
        {"HasSamplesId", to_string(has_sample_ids)},
        {"Codification", get_codification_string()}
    });
    printer.close_element();

    variables_to_JSON(printer);

    samples_to_JSON(printer);

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

    if (data_source_element->has("Streaming"))
        (void)read_json_bool(data_source_element, "Streaming");

    set_separator_name(read_json_string(data_source_element, "Separator"));
    set_codification(read_json_string(data_source_element, "Codification"));
    set_has_header(read_json_bool(data_source_element, "HasHeader"));
    set_has_ids(read_json_bool(data_source_element, "HasSamplesId"));

    set_display(read_json_bool(data_set_element, "Display"));
    read_txt();
}

void LanguageDataset::encode_streaming(const vector<vector<string_view>>& input_document_tokens,
                                       const vector<vector<string_view>>& target_document_tokens,
                                       vector<vector<Index>>& input_indices,
                                       vector<vector<Index>>& target_indices) const
{
    const Index samples_number = ssize(input_document_tokens);

    input_indices.assign(samples_number, {});
    target_indices.assign(samples_number, {});

    const unordered_map<string_view, Index> input_vocabulary_map = create_vocabulary_map(input_vocabulary);
    const unordered_map<string_view, Index> target_vocabulary_map = create_vocabulary_map(target_vocabulary);

    #pragma omp parallel for
    for (Index sample = 0; sample < samples_number; ++sample)
    {
        const vector<string_view>& tokens = input_document_tokens[sample];
        vector<Index>& destination = input_indices[sample];

        destination.reserve(tokens.size() + 2);
        destination.push_back(Index(START_INDEX));

        for (size_t i = 0; i < tokens.size(); ++i)
        {
            if (1 + i >= size_t(maximum_input_sequence_length)) break;
            const auto it = input_vocabulary_map.find(tokens[i]);
            destination.push_back(it != input_vocabulary_map.end() ? it->second : Index(UNK_INDEX));
        }

        if (1 + tokens.size() < size_t(maximum_input_sequence_length))
            destination.push_back(Index(END_INDEX));
    }

    const bool has_decoder = !decoder_shape.empty();
    const Index target_vocab_size = ssize(target_vocabulary);

    if (has_decoder)
    {
        #pragma omp parallel for
        for (Index sample = 0; sample < samples_number; ++sample)
        {
            const vector<string_view>& tokens = target_document_tokens[sample];
            vector<Index>& destination = target_indices[sample];

            destination.reserve(tokens.size() + 1);

            for (size_t i = 0; i < tokens.size(); ++i)
            {
                if (i >= size_t(maximum_target_sequence_length)) break;
                const auto it = target_vocabulary_map.find(tokens[i]);
                destination.push_back(it != target_vocabulary_map.end() ? it->second : Index(UNK_INDEX));
            }

            if (tokens.size() < size_t(maximum_target_sequence_length))
                destination.push_back(Index(END_INDEX));
        }
    }
    else if (maximum_target_sequence_length == 1 && target_vocab_size == 6)
    {
        for (Index sample = 0; sample < samples_number; ++sample)
        {
            const string_view token = target_document_tokens[sample][0];

            if (contains(positive_words, token))
                target_indices[sample] = {1};
            else if (contains(negative_words, token))
                target_indices[sample] = {0};
            else
                throw runtime_error("Unknown target value");
        }
    }
    else if (maximum_target_sequence_length == 6 && target_vocab_size >= 6)
    {
        const Index reserved_count = ssize(reserved_tokens);

        for (Index sample = 0; sample < samples_number; ++sample)
        {
            target_indices[sample].assign(maximum_target_sequence_length, 0);

            const string_view token = target_document_tokens[sample][0];
            const auto it = target_vocabulary_map.find(token);
            const Index vocab_index = (it != target_vocabulary_map.end()) ? it->second : Index(UNK_INDEX);
            const Index col = vocab_index - reserved_count;

            if (col >= 0 && col < maximum_target_sequence_length)
                target_indices[sample][col] = 1;
        }
    }
    else
    {
        #pragma omp parallel for
        for (Index sample = 0; sample < samples_number; ++sample)
        {
            const vector<string_view>& tokens = target_document_tokens[sample];
            vector<Index>& destination = target_indices[sample];

            destination.reserve(tokens.size() + 2);
            destination.push_back(Index(START_INDEX));

            for (size_t i = 0; i < tokens.size(); ++i)
            {
                if (1 + i >= size_t(maximum_target_sequence_length)) break;
                const auto it = target_vocabulary_map.find(tokens[i]);
                destination.push_back(it != target_vocabulary_map.end() ? it->second : Index(UNK_INDEX));
            }

            if (1 + tokens.size() < size_t(maximum_target_sequence_length))
                destination.push_back(Index(END_INDEX));
        }
    }
}

void LanguageDataset::write_binary_cache(const vector<vector<Index>>& input_indices,
                                         const vector<vector<Index>>& target_indices,
                                         bool has_decoder)
{
    const Index samples_number = ssize(input_indices);

    offsets_table.assign(size_t(samples_number), array<int64_t, 4>{0, 0, 0, 0});

    int64_t total_in_tokens = 0;
    int64_t total_tgt_tokens = 0;
    for (Index i = 0; i < samples_number; ++i)
    {
        offsets_table[size_t(i)][0] = total_in_tokens;
        offsets_table[size_t(i)][1] = int64_t(input_indices[size_t(i)].size());
        offsets_table[size_t(i)][2] = total_tgt_tokens;
        offsets_table[size_t(i)][3] = int64_t(target_indices[size_t(i)].size());
        total_in_tokens  += int64_t(input_indices[size_t(i)].size());
        total_tgt_tokens += int64_t(target_indices[size_t(i)].size());
    }

    for (Index i = 0; i < samples_number; ++i)
        offsets_table[size_t(i)][2] += total_in_tokens;

    cache_data_offset = sizeof(LangCacheHeader)
                    + uint64_t(samples_number) * sizeof(array<int64_t, 4>);

    LangCacheHeader header{};
    memcpy(header.magic, LANG_CACHE_MAGIC, 8);
    header.version        = LANG_CACHE_VERSION;
    header.num_samples    = uint64_t(samples_number);
    header.input_max_len  = uint32_t(maximum_input_sequence_length);
    header.target_max_len = uint32_t(maximum_target_sequence_length);
    header.has_decoder    = has_decoder ? uint8_t(1) : uint8_t(0);

    filesystem::create_directories(cache_path.parent_path());
    const filesystem::path tmp_path = cache_path.string() + ".tmp";

    FileWriter writer;
    writer.open(tmp_path);
    writer.write(&header, sizeof(header));
    writer.write(offsets_table.data(), offsets_table.size() * sizeof(array<int64_t, 4>));

    vector<int32_t> chunk;
    chunk.reserve(8192);
    for (const auto& v : input_indices)
    {
        chunk.clear();
        chunk.reserve(v.size());
        for (Index t : v) chunk.push_back(int32_t(t));
        if (!chunk.empty()) writer.write(chunk.data(), chunk.size() * sizeof(int32_t));
    }
    for (const auto& v : target_indices)
    {
        chunk.clear();
        chunk.reserve(v.size());
        for (Index t : v) chunk.push_back(int32_t(t));
        if (!chunk.empty()) writer.write(chunk.data(), chunk.size() * sizeof(int32_t));
    }
    writer.finish_with_rename(cache_path);

    cache_reader.open(cache_path);
}

bool LanguageDataset::try_load_binary_cache(Index expected_samples)
{
    if (!filesystem::exists(cache_path)) return false;

    try
    {
        cache_reader.open(cache_path);
        if (cache_reader.file_size() < sizeof(LangCacheHeader)) return false;

        LangCacheHeader header{};
        cache_reader.read_at(&header, sizeof(header), 0);

        if (memcmp(header.magic, LANG_CACHE_MAGIC, 8) != 0) return false;
        if (header.version != LANG_CACHE_VERSION) return false;
        if (Index(header.num_samples) != expected_samples) return false;
        if (Index(header.input_max_len)  != maximum_input_sequence_length)  return false;
        if (Index(header.target_max_len) != maximum_target_sequence_length) return false;

        offsets_table.resize(size_t(header.num_samples));
        cache_reader.read_at(offsets_table.data(),
                             offsets_table.size() * sizeof(array<int64_t, 4>),
                             sizeof(LangCacheHeader));
        cache_data_offset = sizeof(LangCacheHeader)
                        + uint64_t(header.num_samples) * sizeof(array<int64_t, 4>);
        return true;
    }
    catch (const exception&)
    {
        cache_reader.close();
        return false;
    }
}

void LanguageDataset::fill_inputs(const vector<Index>& sample_indices,
                                  const vector<Index>& /*input_indices*/,
                                  float* input_data,
                                  bool /*is_training*/,
                                  int /*contiguous*/) const
{
    const Index batch_size = ssize(sample_indices);
    const Index seq_len = maximum_input_sequence_length;

    fill_n(input_data, batch_size * seq_len, 0.0f);

    #pragma omp parallel for
    for (Index i = 0; i < batch_size; ++i)
    {
        const auto& offsets = offsets_table[size_t(sample_indices[i])];
        const Index n = min(Index(offsets[1]), seq_len);
        if (n <= 0) continue;

        thread_local vector<int32_t> buf;
        buf.resize(size_t(n));
        cache_reader.read_at(buf.data(), size_t(n) * sizeof(int32_t),
                             cache_data_offset + uint64_t(offsets[0]) * sizeof(int32_t));

        for (Index j = 0; j < n; ++j)
            input_data[i * seq_len + j] = float(buf[size_t(j)]);
    }
}

void LanguageDataset::fill_targets(const vector<Index>& sample_indices,
                                   const vector<Index>& /*target_indices*/,
                                   float* target_data,
                                   bool /*is_training*/,
                                   int /*contiguous*/) const
{
    const Index batch_size = ssize(sample_indices);
    const Index seq_len = maximum_target_sequence_length;

    fill_n(target_data, batch_size * seq_len, 0.0f);

    #pragma omp parallel for
    for (Index i = 0; i < batch_size; ++i)
    {
        const auto& offsets = offsets_table[size_t(sample_indices[i])];
        const Index n = min(Index(offsets[3]), seq_len);
        if (n <= 0) continue;

        thread_local vector<int32_t> buf;
        buf.resize(size_t(n));
        cache_reader.read_at(buf.data(), size_t(n) * sizeof(int32_t),
                             cache_data_offset + uint64_t(offsets[2]) * sizeof(int32_t));

        for (Index j = 0; j < n; ++j)
            target_data[i * seq_len + j] = float(buf[size_t(j)]);
    }
}

void LanguageDataset::fill_decoder(const vector<Index>& sample_indices,
                                   const vector<Index>& /*decoder_indices*/,
                                   float* decoder_data,
                                   bool /*is_training*/,
                                   int /*contiguous*/) const
{
    const Index batch_size = ssize(sample_indices);
    const Index seq_len = maximum_target_sequence_length;

    fill_n(decoder_data, batch_size * seq_len, 0.0f);

    #pragma omp parallel for
    for (Index i = 0; i < batch_size; ++i)
    {
        decoder_data[i * seq_len] = START_INDEX;

        const auto& offsets = offsets_table[size_t(sample_indices[i])];
        const Index n = min(Index(offsets[3]), seq_len - 1);
        if (n <= 0) continue;

        thread_local vector<int32_t> buf;
        buf.resize(size_t(n));
        cache_reader.read_at(buf.data(), size_t(n) * sizeof(int32_t),
                             cache_data_offset + uint64_t(offsets[2]) * sizeof(int32_t));

        for (Index j = 0; j < n; ++j)
            decoder_data[i * seq_len + 1 + j] = float(buf[size_t(j)]);
    }
}

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
