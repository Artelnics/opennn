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
#include "tensor_types.h"
#include "io_utilities.h"

#include <fstream>

namespace opennn
{

namespace {

#pragma pack(push, 1)
struct LangCacheHeader
{
    char     magic[8];
    uint32_t version;
    uint64_t num_samples;
    uint32_t input_max_len;
    uint32_t target_max_len;
    uint8_t  has_decoder;
    uint8_t  pad[35];
};
#pragma pack(pop)
static_assert(sizeof(LangCacheHeader) == 64, "LangCacheHeader must be 64 bytes");

constexpr uint32_t LANG_CACHE_VERSION = 1;
constexpr const char LANG_CACHE_MAGIC[8] = {'O','P','E','N','N','N','T','K'};

}

LanguageDataset::LanguageDataset(const filesystem::path& new_data_path) : Dataset()
{
    data_path = new_data_path;
    separator = Dataset::Separator::Tab;
    storage_mode = StorageMode::BinaryFile;

    if (!data_path.empty())
        read_txt();
}

void LanguageDataset::set_storage_mode(StorageMode new_storage_mode)
{
    Dataset::set_storage_mode(new_storage_mode);

    if (new_storage_mode == StorageMode::BinaryFile)
        data.resize(0, 0);
}

Index LanguageDataset::get_samples_number() const
{
    return storage_mode == StorageMode::Matrix
        ? Index(data.rows())
        : Index(offsets_table.size());
}

LanguageDataset::LanguageDataset(const filesystem::path& new_data_path,
                                 Index new_maximum_vocabulary_size) : Dataset()
{
    data_path = new_data_path;
    separator = Dataset::Separator::Tab;
    maximum_vocabulary_size = new_maximum_vocabulary_size;
    storage_mode = StorageMode::BinaryFile;

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
    storage_mode = StorageMode::BinaryFile;

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
        int32_t token = 0;

        if (storage_mode == StorageMode::Matrix)
        {
            token = int32_t(data(sample, maximum_input_sequence_length));
        }
        else
        {
            const auto& offsets = offsets_table[size_t(sample)];
            if (offsets[3] < 1) continue;
            cache_reader.read_at(&token, sizeof(token),
                                 cache_data_offset + uint64_t(offsets[2]) * sizeof(int32_t));
        }
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

        input_shape = { get_maximum_input_sequence_length() };
        target_shape = { get_maximum_target_sequence_length() };
        decoder_shape.clear();
    }
    else
    {
        maximum_target_sequence_length = maximum_target_document_tokens + 1;

        const Index features_number = maximum_input_sequence_length
                                      + 2 * maximum_target_sequence_length;

        variables.resize(features_number);

        input_shape = { get_maximum_input_sequence_length() };
        decoder_shape = { get_maximum_target_sequence_length() };
        target_shape = { get_maximum_target_sequence_length() };
    }

    const bool has_decoder = !decoder_shape.empty();

    if (storage_mode == StorageMode::Matrix)
    {
        vector<vector<Index>> input_indices;
        vector<vector<Index>> target_indices;
        encode_streaming(input_document_tokens, target_document_tokens, input_indices, target_indices);

        const Index decoder_offset = maximum_input_sequence_length;
        const Index target_offset = has_decoder
            ? decoder_offset + maximum_target_sequence_length
            : maximum_input_sequence_length;

        data.resize(samples_number, ssize(variables));
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
                data(i, decoder_offset) = START_INDEX;
                const Index dec_n = min(ssize(tgt), maximum_target_sequence_length - 1);
                for (Index j = 0; j < dec_n; ++j)
                    data(i, decoder_offset + 1 + j) = float(tgt[size_t(j)]);
            }
        }

        offsets_table.clear();
        cache_reader.close();
    }
    else if (!try_load_binary_cache(samples_number))
    {
        vector<vector<Index>> input_indices;
        vector<vector<Index>> target_indices;
        encode_streaming(input_document_tokens, target_document_tokens, input_indices, target_indices);
        write_binary_cache(input_indices, target_indices, has_decoder);
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

    throw_if(!file.is_open(),
             format("Cannot open file {}", data_path.string()));

    const auto file_size = file.tellg();
    throw_if(file_size < 0,
             format("Cannot determine file size for {}", data_path.string()));

    file.seekg(0);

    buffer.assign(static_cast<size_t>(file_size), '\0');
    if (file_size > 0)
        file.read(buffer.data(), file_size);

    throw_if(!file,
             format("Cannot read file {}", data_path.string()));

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
            throw_if(strict_field_count,
                     "Line must contain two fields: input and target.");
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
        {"Codification", get_codification_string()},
        {"StorageMode", get_storage_mode_string()}
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
    set_storage_mode(data_source_element->has("StorageMode")
                   ? read_json_string(data_source_element, "StorageMode")
                   : "BinaryFile");
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
            const vector<string_view>& sample_tokens = target_document_tokens[sample];
            throw_if(sample_tokens.empty(),
                     "Unknown target value");

            const string_view token = sample_tokens[0];

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

            const vector<string_view>& sample_tokens = target_document_tokens[sample];
            if (sample_tokens.empty())
                continue;

            const string_view token = sample_tokens[0];
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
        const uint64_t file_bytes = cache_reader.file_size();
        if (file_bytes < sizeof(LangCacheHeader)) { cache_reader.close(); return false; }

        LangCacheHeader header{};
        cache_reader.read_at(&header, sizeof(header), 0);

        if (memcmp(header.magic, LANG_CACHE_MAGIC, 8) != 0) { cache_reader.close(); return false; }
        if (header.version != LANG_CACHE_VERSION) { cache_reader.close(); return false; }
        if (Index(header.num_samples) != expected_samples) { cache_reader.close(); return false; }
        if (Index(header.input_max_len)  != maximum_input_sequence_length)  { cache_reader.close(); return false; }
        if (Index(header.target_max_len) != maximum_target_sequence_length) { cache_reader.close(); return false; }
        if (header.has_decoder != uint8_t(!decoder_shape.empty())) { cache_reader.close(); return false; }

        const uint64_t offsets_bytes = header.num_samples * sizeof(array<int64_t, 4>);
        if (file_bytes < sizeof(LangCacheHeader) + offsets_bytes) { cache_reader.close(); return false; }

        offsets_table.resize(size_t(header.num_samples));
        cache_reader.read_at(offsets_table.data(),
                             offsets_table.size() * sizeof(array<int64_t, 4>),
                             sizeof(LangCacheHeader));
        cache_data_offset = sizeof(LangCacheHeader)
                        + uint64_t(header.num_samples) * sizeof(array<int64_t, 4>);

        int64_t token_count = 0;
        for (const array<int64_t, 4>& offsets : offsets_table)
        {
            if (offsets[0] < 0 || offsets[1] < 0 || offsets[2] < 0 || offsets[3] < 0
            ||  offsets[0] > numeric_limits<int64_t>::max() - offsets[1]
            ||  offsets[2] > numeric_limits<int64_t>::max() - offsets[3])
            {
                cache_reader.close();
                return false;
            }

            token_count = max(token_count, offsets[0] + offsets[1]);
            token_count = max(token_count, offsets[2] + offsets[3]);
        }

        const uint64_t expected_bytes = cache_data_offset + uint64_t(token_count) * sizeof(int32_t);
        if (file_bytes != expected_bytes) { cache_reader.close(); return false; }

        return true;
    }
    catch (const exception&)
    {
        cache_reader.close();
        return false;
    }
}

void LanguageDataset::fill_sequences(const vector<Index>& sample_indices,
                                     const vector<Index>& variable_indices,
                                     float* output_data,
                                     int contiguous,
                                     Index sequence_length,
                                     Index offsets_index,
                                     Index shift,
                                     const char* context) const
{
    if (storage_mode == StorageMode::Matrix)
    {
        fill_tensor_data(data, sample_indices, variable_indices, output_data, contiguous);
        return;
    }

    const Index batch_size = ssize(sample_indices);

    fill_n(output_data, batch_size * sequence_length, 0.0f);

    string omp_error;

    #pragma omp parallel for
    for (Index i = 0; i < batch_size; ++i)
    {
        try
        {
            if (shift > 0) output_data[i * sequence_length] = START_INDEX;

            const Index sample_index = sample_indices[size_t(i)];
            throw_if(sample_index < 0 || sample_index >= ssize(offsets_table),
                     format("LanguageDataset {} sample index is out of range.", context));

            const auto& offsets = offsets_table[size_t(sample_index)];
            const Index n = min(Index(offsets[size_t(offsets_index + 1)]), sequence_length - shift);
            if (n <= 0) continue;

            thread_local vector<int32_t> buf;
            buf.resize(size_t(n));
            cache_reader.read_at(buf.data(), size_t(n) * sizeof(int32_t),
                                 cache_data_offset + uint64_t(offsets[size_t(offsets_index)]) * sizeof(int32_t));

            for (Index j = 0; j < n; ++j)
                output_data[i * sequence_length + shift + j] = float(buf[size_t(j)]);
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

void LanguageDataset::fill_inputs(const vector<Index>& sample_indices,
                                  const vector<Index>& input_indices,
                                  float* input_data,
                                  bool /*is_training*/,
                                  int contiguous) const
{
    fill_sequences(sample_indices, input_indices, input_data, contiguous,
                   maximum_input_sequence_length, 0, 0, "input");
}

void LanguageDataset::fill_targets(const vector<Index>& sample_indices,
                                   const vector<Index>& target_indices,
                                   float* target_data,
                                   bool /*is_training*/,
                                   int contiguous) const
{
    fill_sequences(sample_indices, target_indices, target_data, contiguous,
                   maximum_target_sequence_length, 2, 0, "target");
}

void LanguageDataset::fill_decoder(const vector<Index>& sample_indices,
                                   const vector<Index>& decoder_indices,
                                   float* decoder_data,
                                   bool /*is_training*/,
                                   int contiguous) const
{
    fill_sequences(sample_indices, decoder_indices, decoder_data, contiguous,
                   maximum_target_sequence_length, 2, 1, "decoder");
}

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
