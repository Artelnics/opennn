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

namespace opennn
{

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

// Near-duplicate of TokenizerOperator::build_vocabulary, kept separate on
// purpose: this version tie-breaks equal-frequency tokens alphabetically while
// the operator's does not, so unifying them would silently reorder existing
// vocabularies and invalidate the id <-> embedding-row mapping of saved weight
// files. Revisit when the word-level models are next retrained.
void TextGenerationDataset::create_vocabulary(const vector<string_view>& corpus_tokens)
{
    unordered_map<string_view, size_t> token_count;

    for (string_view token : corpus_tokens)
        ++token_count[token];

    vector<pair<string_view, size_t>> sorted_tokens(token_count.begin(), token_count.end());

    ranges::sort(sorted_tokens,
                 [](const auto& a, const auto& b)
                 { return a.second != b.second ? a.second > b.second : a.first < b.first; });

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

void TextGenerationDataset::read_txt()
{
    cout << "Reading .txt file..." << "\n";

    throw_if(sequence_length <= 0,
             "TextGenerationDataset: sequence_length must be > 0.");

    cache_reader.close();

    string buffer;
    read_file(buffer);

    // A tokenizer with a fixed loaded vocabulary (subword, e.g. byte-pair)
    // encodes the raw corpus; otherwise the corpus is lowercased and split into
    // whitespace/word-level tokens whose vocabulary is built by frequency.
    const bool subword = tokenizer && tokenizer->get_vocabulary_size() > 0;

    vector<Index> token_ids;
    string cache_tag;

    if (subword)
    {
        cout << "Tokenizing corpus (subword)..." << "\n";
        vocabulary = tokenizer->get_vocabulary();
        token_ids = tokenizer->encode(buffer);
        cache_tag = format("sub{}", vocabulary.size());
    }
    else
    {
        for (char& character : buffer)
            character = static_cast<char>(tolower(static_cast<unsigned char>(character)));

        const vector<string_view> corpus_tokens = tokenize_views(buffer);
        create_vocabulary(corpus_tokens);
        token_ids = encode_corpus(corpus_tokens);
        cache_tag = "word";
    }

    update_vocabulary_map();

    // Non-overlapping blocks of (sequence_length + 1) tokens: inputs are tokens
    // [0, T-1] and targets the same block shifted one position, [1, T].
    const Index record_tokens = sequence_length + 1;
    const Index samples_number = ssize(token_ids) / record_tokens;

    throw_if(samples_number == 0,
             format("TextGenerationDataset: corpus has {} tokens; at least {} are needed for one sample.",
                    token_ids.size(), record_tokens));

    input_shape  = { sequence_length };
    target_shape = { sequence_length };
    decoder_shape.clear();

    variables.assign(2, Variable());

    Variable& input_variable = variables[0];
    input_variable.name = "input_sequence";
    input_variable.role = VariableRole::Input;
    input_variable.type = VariableType::Numeric;
    input_variable.features = sequence_length;
    input_variable.categories = vocabulary;

    Variable& target_variable = variables[1];
    target_variable.name = "target_sequence";
    target_variable.role = VariableRole::Target;
    target_variable.type = VariableType::Numeric;
    target_variable.features = sequence_length;

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
        // BinaryFile: fixed-size records of (sequence_length + 1) int32 tokens per
        // sample. The cache name carries a tokenizer tag so switching tokenizers
        // never silently reuses a stale cache.
        cache_path = filesystem::path(data_path.string() + ".cache") / format("lm_tokens_{}.bin", cache_tag);

        const uintmax_t record_bytes = uintmax_t(record_tokens) * sizeof(int32_t);

        const bool cache_valid = filesystem::exists(cache_path)
            && filesystem::file_size(cache_path) == uintmax_t(samples_number) * record_bytes
            && filesystem::last_write_time(cache_path) >= filesystem::last_write_time(data_path);

        if (cache_valid)
            cache_reader.open(cache_path);
        else
            write_binary_cache(token_ids, samples_number);
    }

    sample_roles.resize(samples_number);

    split_samples_random();

    cout << "Reading finished" << "\n";
}

void TextGenerationDataset::update_vocabulary_map()
{
    vocabulary_map.clear();
    vocabulary_map.reserve(vocabulary.size());

    for (Index i = 0; i < ssize(vocabulary); ++i)
        vocabulary_map[vocabulary[i]] = i;
}

void TextGenerationDataset::set_vocabulary(const vector<string>& new_vocabulary)
{
    vocabulary = new_vocabulary;
    update_vocabulary_map();
}

void TextGenerationDataset::read_file(string& buffer) const
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
}

vector<Index> TextGenerationDataset::encode_corpus(const vector<string_view>& corpus_tokens) const
{
    const unordered_map<string_view, Index> corpus_vocabulary_map = [this]
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
        const auto it = corpus_vocabulary_map.find(corpus_tokens[size_t(i)]);
        token_indices[size_t(i)] = it != corpus_vocabulary_map.end() ? it->second : Index(UNK_INDEX);
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

        writer.write(record.data(), record.size() * sizeof(int32_t));
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
    if (storage_mode == StorageMode::Matrix)
    {
        fill_tensor_data(data, sample_indices, variable_indices, output_data, contiguous);
        return;
    }

    const Index batch_size = ssize(sample_indices);
    const Index samples_number = get_samples_number();
    const uint64_t record_tokens = uint64_t(sequence_length + 1);

    string omp_error;

    #pragma omp parallel for
    for (Index i = 0; i < batch_size; ++i)
    {
        try
        {
            const Index sample_index = sample_indices[size_t(i)];
            throw_if(sample_index < 0 || sample_index >= samples_number,
                     format("TextGenerationDataset {} sample index is out of range.", context));

            thread_local vector<int32_t> buf;
            buf.resize(size_t(sequence_length));
            cache_reader.read_at(buf.data(), size_t(sequence_length) * sizeof(int32_t),
                                 (uint64_t(sample_index) * record_tokens + uint64_t(record_offset)) * sizeof(int32_t));

            for (Index j = 0; j < sequence_length; ++j)
                output_data[i * sequence_length + j] = float(buf[size_t(j)]);
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

void TextGenerationDataset::fill_inputs(const vector<Index>& sample_indices,
                                        const vector<Index>& input_indices,
                                        float* input_data,
                                        bool /*is_training*/,
                                        int contiguous) const
{
    fill_blocks(sample_indices, input_indices, input_data, contiguous, 0, "input");
}

void TextGenerationDataset::fill_targets(const vector<Index>& sample_indices,
                                         const vector<Index>& target_indices,
                                         float* target_data,
                                         bool /*is_training*/,
                                         int contiguous) const
{
    fill_blocks(sample_indices, target_indices, target_data, contiguous, 1, "target");
}

void TextGenerationDataset::to_JSON(JsonWriter& printer) const
{
    printer.open_element("Dataset");

    printer.open_element("DataSource");

    write_json(printer, {
        {"FileType", "txt"},
        {"Path", data_path.string()},
        {"StorageMode", get_storage_mode_string()}
    });
    printer.close_element();

    variables_to_JSON(printer);

    samples_to_JSON(printer);

    preview_data_to_JSON(printer);

    write_json(printer, {
        {"Vocabulary", vector_to_string(vocabulary, " ")},
        {"SequenceLength", to_string(sequence_length)},
        {"MaximumVocabularySize", to_string(maximum_vocabulary_size)},
        {"MinimumTokenFrequency", to_string(minimum_token_frequency)},
        {"Display", to_string(display)}
    });

    printer.close_element();
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

    read_txt();
}

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
