//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   T E X T   G E N E R A T I O N   D A T A S E T   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "text_generation_dataset.h"
#include "string_utilities.h"
#include "tensor_types.h"
#include "io_utilities.h"

#ifdef _OPENMP
#include <omp.h>
#endif

namespace opennn
{

static constexpr array<char, 8> TEXT_CACHE_MAGIC{'O', 'N', 'N', 'T', 'E', 'X', 'T', '3'};
static constexpr uint32_t TEXT_CACHE_VERSION = 1;

TextGenerationDataset::TextGenerationDataset(const filesystem::path& new_data_path,
                                             Index new_sequence_length,
                                             Index new_maximum_vocabulary_size,
                                             Index new_minimum_token_frequency) : Dataset()
{
    data_path = new_data_path;
    separator = Dataset::Separator::Space;
    sequence_length = new_sequence_length;
    maximum_vocabulary_size = new_maximum_vocabulary_size;
    minimum_token_frequency = new_minimum_token_frequency;
    storage_mode = StorageMode::BinaryFile;

    if (!data_path.empty())
        read_txt();
}

void TextGenerationDataset::create_vocabulary(const vector<string_view>& corpus_tokens)
{
    unordered_map<string_view, size_t> token_count;

#ifdef _OPENMP
    if (corpus_tokens.size() >= 10000)
    {
        vector<unordered_map<string_view, size_t>> local_counts(
            static_cast<size_t>(omp_get_max_threads()));

        #pragma omp parallel
        {
            auto& local = local_counts[size_t(omp_get_thread_num())];

            #pragma omp for schedule(static)
            for (Index i = 0; i < ssize(corpus_tokens); ++i)
                ++local[corpus_tokens[size_t(i)]];
        }

        for (const auto& local : local_counts)
            for (const auto& [token, count] : local)
                token_count[token] += count;
    }
    else
#endif
        for (string_view token : corpus_tokens)
            ++token_count[token];

    tokenizer->set_vocabulary(make_vocabulary(token_count, reserved_tokens,
                                              maximum_vocabulary_size,
                                              minimum_token_frequency));
}

void TextGenerationDataset::read_txt()
{
    cout << "Reading .txt file..." << "\n";

    throw_if(sequence_length <= 0,
             "TextGenerationDataset: sequence_length must be > 0.");

    cache_reader.close();

    const uint64_t tokenizer_fingerprint =
        fixed_vocabulary ? tokenizer->fingerprint() : 0;
    const filesystem::path cache_parent = cache_directory.empty()
        ? filesystem::path(data_path.string() + ".cache")
        : cache_directory / (data_path.filename().string() + ".cache");
    cache_path = cache_parent
        / format("lm_tokens_v3_{}_{}_{}_{}_{:016x}.bin",
                 sequence_length,
                 maximum_vocabulary_size,
                 minimum_token_frequency,
                 fixed_vocabulary ? 1 : 0,
                 tokenizer_fingerprint);
    const filesystem::path metadata_path = cache_path.string() + ".meta";

    if (storage_mode == StorageMode::BinaryFile
        && is_file_current(cache_path, {data_path})
        && is_file_current(metadata_path, {data_path})
        && load_cache_metadata(metadata_path, tokenizer_fingerprint))
    {
        split_samples_random();
        cout << "Reading finished (cached)" << "\n";
        return;
    }

    string buffer = read_text_file(data_path);

    const bool subword = fixed_vocabulary;

    vector<Index> token_ids;
    if (subword)
    {
        cout << "Tokenizing corpus (subword)..." << "\n";
        token_ids = tokenizer->encode(buffer);
    }
    else
    {
        ascii_lowercase_in_place(buffer);

        const vector<string_view> corpus_tokens = tokenize_views(buffer);
        create_vocabulary(corpus_tokens);
        token_ids = encode_corpus(corpus_tokens);
    }

    const Index record_tokens = sequence_length + 1;
    const Index samples_number = ssize(token_ids) / record_tokens;

    throw_if(samples_number == 0,
             "TextGenerationDataset: corpus has {} tokens; at least {} are needed for one sample.",
                    token_ids.size(), record_tokens);

    configure(samples_number);

    if (storage_mode == StorageMode::Matrix)
    {
        data.resize(samples_number, get_features_number());

        for (Index i = 0; i < samples_number; ++i)
        {
            const Index block_start = i * record_tokens;

            for (Index j = 0; j < sequence_length; ++j)
            {
                data(i, j) = float(token_ids[size_t(block_start + j)]);
                data(i, sequence_length + j) = float(token_ids[size_t(block_start + j + 1)]);
            }
        }
    }
    else
    {
        write_binary_cache(token_ids, samples_number);
        save_cache_metadata(metadata_path, tokenizer_fingerprint, samples_number);
    }

    split_samples_random();

    cout << "Reading finished" << "\n";
}

void TextGenerationDataset::set_tokenizer(unique_ptr<TokenizerOperator> new_tokenizer)
{
    tokenizer = new_tokenizer
        ? move(new_tokenizer)
        : make_unique<WordLevelTokenizer>(reserved_tokens);
    fixed_vocabulary = tokenizer->get_vocabulary_size() > 0;
}

void TextGenerationDataset::set_vocabulary(const vector<string>& new_vocabulary)
{
    tokenizer->set_vocabulary(new_vocabulary);
    fixed_vocabulary = true;
}

void TextGenerationDataset::configure(Index samples_number)
{
    input_shape = {sequence_length};
    target_shape = {sequence_length};
    decoder_shape.clear();

    variables.assign(2, Variable());

    Variable& input_variable = variables[0];
    input_variable.name = "input_sequence";
    input_variable.role = VariableRole::Input;
    input_variable.type = VariableType::Numeric;
    input_variable.features = sequence_length;
    input_variable.categories = tokenizer->get_vocabulary();

    Variable& target_variable = variables[1];
    target_variable.name = "target_sequence";
    target_variable.role = VariableRole::Target;
    target_variable.type = VariableType::Numeric;
    target_variable.features = sequence_length;

    sample_roles.resize(size_t(samples_number));
}

bool TextGenerationDataset::load_cache_metadata(const filesystem::path& metadata_path,
                                                uint64_t expected_fingerprint)
{
    ifstream file(metadata_path, ios::binary);
    if (!file) return false;

    array<char, 8> magic{};
    uint32_t version = 0;
    int64_t stored_sequence_length = 0;
    int64_t samples_number = 0;
    uint64_t fingerprint = 0;
    uint64_t vocabulary_size = 0;

    if (!file.read(magic.data(), magic.size())
        || !read_binary_values(file, version, stored_sequence_length, samples_number,
                               fingerprint, vocabulary_size))
        return false;

    if (magic != TEXT_CACHE_MAGIC
        || version != TEXT_CACHE_VERSION
        || stored_sequence_length != sequence_length
        || samples_number <= 0
        || fingerprint != expected_fingerprint
        || vocabulary_size > uint64_t(numeric_limits<Index>::max()))
        return false;

    vector<string> cached_vocabulary(static_cast<size_t>(vocabulary_size));
    for (string& token : cached_vocabulary)
        if (!read_binary_string(file, token)) return false;

    if (fixed_vocabulary)
    {
        if (cached_vocabulary != tokenizer->get_vocabulary()) return false;
    }
    else
        tokenizer->set_vocabulary(cached_vocabulary);

    const uintmax_t expected_bytes =
        uintmax_t(samples_number) * uintmax_t(sequence_length + 1) * sizeof(int32_t);
    error_code error;
    if (filesystem::file_size(cache_path, error) != expected_bytes || error)
        return false;

    configure(Index(samples_number));
    cache_reader.open(cache_path);
    return true;
}

void TextGenerationDataset::save_cache_metadata(const filesystem::path& metadata_path,
                                                uint64_t fingerprint,
                                                Index samples_number) const
{
    FileWriter writer;
    writer.open(metadata_path.string() + ".tmp");

    writer.write(span(TEXT_CACHE_MAGIC));
    write_binary_value(writer, TEXT_CACHE_VERSION);
    write_binary_value(writer, int64_t(sequence_length));
    write_binary_value(writer, int64_t(samples_number));
    write_binary_value(writer, fingerprint);

    const vector<string>& vocabulary = tokenizer->get_vocabulary();
    write_binary_value(writer, uint64_t(vocabulary.size()));
    for (const string& token : vocabulary)
        write_binary_string(writer, token);

    writer.finish_with_rename(metadata_path);
}

vector<Index> TextGenerationDataset::encode_corpus(const vector<string_view>& corpus_tokens) const
{
    const vector<string>& vocabulary = tokenizer->get_vocabulary();
    const unordered_map<string_view, Index> vocabulary_views = [&vocabulary]
    {
        unordered_map<string_view, Index> map;
        map.reserve(vocabulary.size());
        for (Index i = 0; i < ssize(vocabulary); ++i)
            map.emplace(string_view(vocabulary[i]), i);
        return map;
    }();

    const Index tokens_number = ssize(corpus_tokens);

    vector<Index> token_indices(corpus_tokens.size());

    #pragma omp parallel for
    for (Index i = 0; i < tokens_number; ++i)
    {
        const auto iterator = vocabulary_views.find(corpus_tokens[size_t(i)]);
        token_indices[size_t(i)] =
            iterator != vocabulary_views.end() ? iterator->second : UNK_INDEX;
    }

    return token_indices;
}

void TextGenerationDataset::write_binary_cache(const vector<Index>& token_indices, Index samples_number)
{
    const Index record_tokens = sequence_length + 1;

    filesystem::create_directories(cache_path.parent_path());
    const filesystem::path tmp_path = cache_path.string() + ".tmp";

    FileWriter writer;
    writer.open(tmp_path);

    vector<int32_t> record(size_t(record_tokens), 0);

    for (Index i = 0; i < samples_number; ++i)
    {
        const Index block_start = i * record_tokens;

        for (Index j = 0; j < record_tokens; ++j)
            record[size_t(j)] = int32_t(token_indices[size_t(block_start + j)]);

        writer.write(span(record));
    }

    writer.finish_with_rename(cache_path);

    cache_reader.open(cache_path);
}

void TextGenerationDataset::fill_blocks(const vector<Index>& sample_indices,
                                        const vector<Index>& variable_indices,
                                        float* output_data,
                                        int contiguous,
                                        Index record_offset,
                                        const char* context) const
{
    const span<float> output(output_data, size_t(ssize(sample_indices) * sequence_length));

    if (storage_mode == StorageMode::Matrix)
    {
        fill_tensor_data(data, sample_indices, variable_indices, output, contiguous);
        return;
    }

    read_int32_batch(cache_reader,
                     sample_indices,
                     get_samples_number(),
                     uint64_t(sequence_length + 1),
                     record_offset,
                     sequence_length,
                     output,
                     sequence_length,
                     0,
                     format("TextGenerationDataset {}", context));
}

void TextGenerationDataset::fill_inputs(const vector<Index>& sample_indices,
                                        const vector<Index>& input_indices,
                                        float* input_data,
                                        FillMode,
                                        int contiguous) const
{
    fill_blocks(sample_indices, input_indices, input_data, contiguous, 0, "input");
}

void TextGenerationDataset::fill_targets(const vector<Index>& sample_indices,
                                         const vector<Index>& target_indices,
                                         float* target_data,
                                         FillMode,
                                         int contiguous) const
{
    fill_blocks(sample_indices, target_indices, target_data, contiguous, 1, "target");
}

void TextGenerationDataset::to_JSON(JsonWriter& printer) const
{
    write_json_header(printer, {
        {"FileType", "txt"},
        {"Path", data_path.string()},
        {"StorageMode", get_storage_mode_string()}
    });

    preview_data_to_JSON(printer);

    write_json(printer, {
        {"Vocabulary", json_array(tokenizer->get_vocabulary())},
        {"SequenceLength", sequence_length},
        {"MaximumVocabularySize", maximum_vocabulary_size},
        {"MinimumTokenFrequency", minimum_token_frequency}
    });

    write_json_footer(printer);
}

void TextGenerationDataset::from_JSON(const JsonDocument& data_set_document)
{
    const Json* data_set_element = get_json_root(data_set_document, "Dataset");

    const Json* data_source_element = require_json_field(data_set_element, "DataSource");

    set_data_path(read_json_string(data_source_element, "Path"));

    set_storage_mode(data_source_element->has("StorageMode")
                   ? read_json_string(data_source_element, "StorageMode")
                   : "BinaryFile");

    sequence_length = Index(read_json_index(data_set_element, "SequenceLength"));
    maximum_vocabulary_size = Index(read_json_index(data_set_element, "MaximumVocabularySize"));
    minimum_token_frequency = Index(read_json_index(data_set_element, "MinimumTokenFrequency"));

    set_display(read_json_bool(data_set_element, "Display"));

    if (data_set_element->has("Vocabulary"))
        set_vocabulary(read_json_strings(data_set_element, "Vocabulary"));

    read_txt();
}

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
