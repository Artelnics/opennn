//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   B E R T   D A T A S E T   C L A S S

#include "text_dataset.h"
#include "tokenizer_operator.h"
#include "io_utilities.h"

namespace opennn::detail
{

class BertTextDataset final : public TextDataset
{
public:

    BertTextDataset(const filesystem::path& text_file,
                    const filesystem::path& vocabulary_file,
                    Index sequence_length);

    Index get_sequence_length() const noexcept override { return sequence_length; }
    Task get_task() const noexcept override { return Task::BertClassification; }
    const vector<string>& get_input_vocabulary() const noexcept override { return bert_vocabulary; }
    const vector<string>& get_target_vocabulary() const noexcept override { return label_vocabulary; }

    void from_JSON(const JsonDocument&) override;
    void to_JSON(JsonWriter&) const override;
    void fill_inputs(const vector<Index>&, const vector<Index>&, float*, FillMode, int = -1) const override;
    void fill_decoder(const vector<Index>&, const vector<Index>&, float*, FillMode, int = -1) const override;
    void fill_targets(const vector<Index>&, const vector<Index>&, float*, FillMode, int = -1) const override;

private:

    void read_bert(const filesystem::path&, const filesystem::path&);

    Index sequence_length = 0;
    filesystem::path vocabulary_path;
    vector<string> bert_vocabulary;
    vector<string> label_vocabulary;
};

}

namespace opennn
{

unique_ptr<TextDataset> TextDataset::from_bert_classification(
    const filesystem::path& text_file,
    const filesystem::path& vocabulary_file,
    Index sequence_length)
{
    return make_unique<detail::BertTextDataset>(text_file, vocabulary_file, sequence_length);
}

}

namespace opennn::detail
{

BertTextDataset::BertTextDataset(const filesystem::path& text_file,
                                 const filesystem::path& vocabulary_file,
                                 Index new_sequence_length)
    : TextDataset(Task::BertClassification),
      sequence_length(new_sequence_length),
      vocabulary_path(vocabulary_file)
{
    data_path = text_file;
    storage_mode = StorageMode::Matrix;
    read_bert(text_file, vocabulary_file);
}

void BertTextDataset::read_bert(const filesystem::path& text_file,
                            const filesystem::path& vocabulary_file)
{
    throw_if(sequence_length < 2, "TextDataset BERT: sequence_length must be at least 2.");

    WordPieceTokenizer wordpiece_tokenizer;
    wordpiece_tokenizer.load_vocabulary(vocabulary_file);

    const Index cls = wordpiece_tokenizer.token_to_id("[CLS]");
    const Index sep = wordpiece_tokenizer.token_to_id("[SEP]");
    const Index pad = wordpiece_tokenizer.token_to_id("[PAD]");

    throw_if(cls == wordpiece_tokenizer.get_unk_id() || sep == wordpiece_tokenizer.get_unk_id(),
             "TextDataset BERT: vocabulary is missing [CLS]/[SEP].");
    throw_if(pad != 0,
             "TextDataset BERT: [PAD] must be token id 0 (attention masking convention).");

    const string buffer = read_text_file(text_file);
    vector<vector<Index>> token_rows;
    vector<string> labels;
    vector<string> categories;
    unordered_map<string, Index> category_indices;

    string_view text(buffer);
    size_t line_start = 0;
    while (line_start < text.size())
    {
        size_t line_end = text.find('\n', line_start);
        if (line_end == string_view::npos) line_end = text.size();

        string_view line = text.substr(line_start, line_end - line_start);
        line_start = line_end + 1;
        if (!line.empty() && line.back() == '\r') line.remove_suffix(1);

        const size_t tab = line.rfind('\t');
        if (tab == string_view::npos) continue;

        const string_view document = line.substr(0, tab);
        const string_view label_view = line.substr(tab + 1);
        if (document.empty() || label_view.empty()) continue;

        vector<Index> ids;
        ids.reserve(size_t(sequence_length));
        ids.push_back(cls);
        for (const Index id : wordpiece_tokenizer.encode(document))
        {
            if (ssize(ids) >= sequence_length - 1) break;
            ids.push_back(id);
        }
        ids.push_back(sep);
        ids.resize(size_t(sequence_length), pad);
        token_rows.push_back(move(ids));

        string label(label_view);
        if (!category_indices.contains(label))
        {
            category_indices.emplace(label, ssize(categories));
            categories.push_back(label);
        }
        labels.push_back(move(label));
    }

    const Index samples_number = ssize(token_rows);
    const Index classes_number = ssize(categories);
    const Index targets_number = classes_number == 2 ? 1 : classes_number;
    throw_if(samples_number == 0, "TextDataset BERT: no labelled samples were found.");
    throw_if(classes_number < 2, "TextDataset BERT: classification requires at least two labels.");

    variables.assign(3, Variable());

    variables[0].name = "input_ids";
    variables[0].role = VariableRole::Decoder;
    variables[0].type = VariableType::Numeric;
    variables[0].features = sequence_length;
    bert_vocabulary = wordpiece_tokenizer.get_vocabulary();
    label_vocabulary = categories;
    variables[0].categories = bert_vocabulary;

    variables[1].name = "attention_mask";
    variables[1].role = VariableRole::Input;
    variables[1].type = VariableType::Numeric;
    variables[1].features = sequence_length;

    variables[2].name = "label";
    variables[2].role = VariableRole::Target;
    variables[2].type = classes_number == 2 ? VariableType::Binary : VariableType::Categorical;
    variables[2].features = targets_number;
    variables[2].categories = categories;

    input_shape = {sequence_length};
    decoder_shape = {sequence_length};
    target_shape = {targets_number};

    data.resize(samples_number, 2 * sequence_length + targets_number);
    data.setZero();

    for (Index sample = 0; sample < samples_number; ++sample)
    {
        for (Index token = 0; token < sequence_length; ++token)
        {
            const Index id = token_rows[size_t(sample)][size_t(token)];
            data(sample, token) = float(id);
            data(sample, sequence_length + token) = id == pad ? 0.0f : 1.0f;
        }

        const Index category = category_indices.at(labels[size_t(sample)]);
        if (classes_number == 2)
            data(sample, 2 * sequence_length) = float(category);
        else
            data(sample, 2 * sequence_length + category) = 1.0f;
    }

    sample_roles.resize(samples_number);
    split_samples_random();
}

void BertTextDataset::to_JSON(JsonWriter& printer) const
{
    printer.open_element("Dataset");
    write_json(printer, {
        {"Task", "BertClassification"},
        {"Path", data_path.string()},
        {"VocabularyPath", vocabulary_path.string()},
        {"SequenceLength", to_string(sequence_length)}
    });
    printer.close_element();
}

void BertTextDataset::from_JSON(const JsonDocument& document)
{
    const Json* root = get_json_root(document, "Dataset");
    data_path = read_json_string(root, "Path");
    vocabulary_path = read_json_string(root, "VocabularyPath");
    sequence_length = read_json_index(root, "SequenceLength");
    read_bert(data_path, vocabulary_path);
}

void BertTextDataset::fill_inputs(const vector<Index>& sample_indices,
                              const vector<Index>& input_indices,
                              float* output,
                              FillMode,
                              int contiguous) const
{
    fill_tensor_data(data, sample_indices, input_indices, output, contiguous);
}

void BertTextDataset::fill_decoder(const vector<Index>& sample_indices,
                               const vector<Index>& decoder_indices,
                               float* output,
                               FillMode,
                               int contiguous) const
{
    fill_tensor_data(data, sample_indices, decoder_indices, output, contiguous);
}

void BertTextDataset::fill_targets(const vector<Index>& sample_indices,
                               const vector<Index>& target_indices,
                               float* output,
                               FillMode,
                               int contiguous) const
{
    fill_tensor_data(data, sample_indices, target_indices, output, contiguous);
}

}
