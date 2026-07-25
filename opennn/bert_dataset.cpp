//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   B E R T   D A T A S E T   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "bert_dataset.h"
#include "io_utilities.h"
#include "string_utilities.h"
#include "tokenizer_operator.h"

namespace opennn
{

static constexpr array<char, 8> BERT_CACHE_MAGIC{'O', 'N', 'N', 'B', 'E', 'R', 'T', '3'};
static constexpr uint32_t BERT_CACHE_VERSION = 1;

BertDataset::BertDataset(const filesystem::path& text_file,
                         const filesystem::path& vocabulary_file,
                         Index new_sequence_length)
{
    throw_if(new_sequence_length <= 1,
             "BertDataset: sequence length must leave room for [CLS] and [SEP].");

    sequence_length = new_sequence_length;
    data_path = text_file;
    separator = Dataset::Separator::Tab;
    storage_mode = StorageMode::Matrix;

    const filesystem::path bert_cache_path =
        text_file.string() + ".bert_v3_" + to_string(sequence_length) + ".bin";

    if (!is_file_current(bert_cache_path, {text_file, vocabulary_file})
        || !load_cache(bert_cache_path))
    {
        build(text_file, vocabulary_file);
        save_cache(bert_cache_path, variables.back().categories);
    }

    split_samples_random();
}

void BertDataset::configure(const vector<string>& labels, Index samples_number)
{
    const Index target_features = labels.size() == 2 ? 1 : Index(labels.size());
    throw_if(target_features == 0, "BertDataset: no labels found.");

    variables.assign(size_t(2 * sequence_length + 1), Variable());

    for (Index i = 0; i < sequence_length; ++i)
    {
        Variable& token = variables[size_t(i)];
        token.name = format("id_{}", i);
        token.role = VariableRole::Decoder;
        token.type = VariableType::Numeric;
        token.scaler = ScalerMethod::None;

        Variable& mask = variables[size_t(sequence_length + i)];
        mask.name = format("mask_{}", i);
        mask.role = VariableRole::Input;
        mask.type = VariableType::Numeric;
        mask.scaler = ScalerMethod::None;
    }

    Variable& label = variables.back();
    label.name = "label";
    label.role = VariableRole::Target;
    label.type = labels.size() == 2 ? VariableType::Binary : VariableType::Categorical;
    label.scaler = ScalerMethod::None;
    label.categories = labels;

    input_shape = {sequence_length};
    decoder_shape = {sequence_length};
    target_shape = {target_features};

    data.resize(samples_number, 2 * sequence_length + target_features);
    data.setZero();
    sample_roles.assign(size_t(samples_number), SampleRole::Training);
}

void BertDataset::build(const filesystem::path& text_file,
                        const filesystem::path& vocabulary_file)
{
    WordPieceTokenizer tokenizer;
    tokenizer.load_vocabulary(vocabulary_file);

    const Index cls = tokenizer.token_to_id("[CLS]");
    const Index sep = tokenizer.token_to_id("[SEP]");

    throw_if(cls == tokenizer.get_unk_id() || sep == tokenizer.get_unk_id(),
             "BertDataset: vocabulary is missing [CLS]/[SEP].");
    throw_if(tokenizer.token_to_id("[PAD]") != 0,
             "BertDataset: [PAD] must be token id 0.");

    const CsvReader reader({'\t', {}});
    const CsvReader::Result source = reader.read(text_file);

    vector<pair<string_view, string_view>> rows;
    rows.reserve(source.lines.size());
    unordered_set<string> unique_labels;

    for (const string_view line : source.lines)
    {
        const size_t tab = line.rfind('\t');
        if (tab == string_view::npos) continue;

        const string_view text = trim_view(line.substr(0, tab));
        const string_view label = trim_view(line.substr(tab + 1));
        if (text.empty() || label.empty()) continue;

        rows.emplace_back(text, label);
        unique_labels.emplace(label);
    }

    vector<string> labels(unique_labels.begin(), unique_labels.end());
    ranges::sort(labels);
    configure(labels, Index(rows.size()));

    unordered_map<string_view, Index> label_indices;
    for (Index i = 0; i < ssize(labels); ++i)
        label_indices.emplace(labels[size_t(i)], i);

    const Index target_offset = 2 * sequence_length;

    #pragma omp parallel for
    for (Index row = 0; row < ssize(rows); ++row)
    {
        const auto& [text, label] = rows[size_t(row)];
        vector<Index> ids = tokenizer.encode_sequence(text, sequence_length);
        const Index real_length = ssize(ids);
        ids.resize(size_t(sequence_length), 0);

        for (Index i = 0; i < sequence_length; ++i)
        {
            data(row, i) = float(ids[size_t(i)]);
            data(row, sequence_length + i) = i < real_length ? 1.0f : 0.0f;
        }

        const Index label_index = label_indices.at(label);
        if (labels.size() == 2)
        {
            const bool positive = contains(positive_words, label);
            const bool negative = contains(negative_words, label);
            data(row, target_offset) =
                positive || (!negative && label_index == 1) ? 1.0f : 0.0f;
        }
        else
            data(row, target_offset + label_index) = 1.0f;
    }
}

bool BertDataset::load_cache(const filesystem::path& bert_cache_path)
{
    ifstream file(bert_cache_path, ios::binary);
    if (!file) return false;

    array<char, 8> magic{};
    uint32_t version = 0;
    int64_t stored_sequence_length = 0;
    int64_t samples_number = 0;
    int64_t labels_number = 0;

    if (!file.read(magic.data(), magic.size())
        || !read_binary_value(file, version)
        || !read_binary_value(file, stored_sequence_length)
        || !read_binary_value(file, samples_number)
        || !read_binary_value(file, labels_number)
        || magic != BERT_CACHE_MAGIC
        || version != BERT_CACHE_VERSION
        || stored_sequence_length != sequence_length
        || samples_number < 0
        || labels_number <= 0)
        return false;

    vector<string> labels(static_cast<size_t>(labels_number));
    for (string& label : labels)
        if (!read_binary_string(file, label)) return false;

    configure(labels, Index(samples_number));
    const streamsize bytes = streamsize(data.size() * Index(sizeof(float)));
    return bool(file.read(reinterpret_cast<char*>(data.data()), bytes));
}

void BertDataset::save_cache(const filesystem::path& bert_cache_path,
                             const vector<string>& labels) const
{
    FileWriter writer;
    writer.open(bert_cache_path.string() + ".tmp");

    writer.write(BERT_CACHE_MAGIC.data(), BERT_CACHE_MAGIC.size());
    write_binary_value(writer, BERT_CACHE_VERSION);
    write_binary_value(writer, int64_t(sequence_length));
    write_binary_value(writer, int64_t(data.rows()));
    write_binary_value(writer, int64_t(labels.size()));

    for (const string& label : labels)
        write_binary_string(writer, label);

    writer.write(data.data(), size_t(data.size()) * sizeof(float));
    writer.finish_with_rename(bert_cache_path);
}

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
