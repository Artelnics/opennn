//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   B E R T   D A T A S E T   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "bert_dataset.h"
#include "tokenizer.h"

namespace opennn
{

BertDataset::BertDataset(const filesystem::path& text_file,
                         const filesystem::path& vocabulary_file,
                         Index new_sequence_length)
{
    sequence_length = new_sequence_length;

    const filesystem::path csv_path =
        text_file.string() + ".bert" + to_string(sequence_length) + ".csv";

    const bool cache_valid = filesystem::exists(csv_path)
        && filesystem::last_write_time(csv_path) >= filesystem::last_write_time(text_file)
        && filesystem::last_write_time(csv_path) >= filesystem::last_write_time(vocabulary_file);

    if (!cache_valid)
        write_tokens_csv(text_file, vocabulary_file, csv_path);

    set(csv_path, ";", /*has_header*/ true, /*has_ids*/ false);

    configure_bert_roles();
}

void BertDataset::write_tokens_csv(const filesystem::path& text_file,
                                   const filesystem::path& vocabulary_file,
                                   const filesystem::path& csv_path) const
{
    WordPieceTokenizer tokenizer;
    tokenizer.load_vocabulary(vocabulary_file);

    const Index cls = tokenizer.token_to_id("[CLS]");
    const Index sep = tokenizer.token_to_id("[SEP]");
    const Index pad = 0;

    throw_if(cls == tokenizer.get_unk_id() || sep == tokenizer.get_unk_id(),
             "BertDataset: vocabulary is missing [CLS]/[SEP].");

    ifstream file(text_file);
    throw_if(!file.is_open(), format("BertDataset: cannot open {}", text_file.string()));

    ofstream out(csv_path);
    throw_if(!out.is_open(), format("BertDataset: cannot write {}", csv_path.string()));

    for (Index i = 0; i < sequence_length; ++i) out << "id_" << i << ";";
    for (Index i = 0; i < sequence_length; ++i) out << "tt_" << i << ";";
    out << "label\n";

    string line;

    while (getline(file, line))
    {
        if (!line.empty() && line.back() == '\r') line.pop_back();

        const size_t tab = line.rfind('\t');
        if (tab == string::npos) continue;

        const string text  = line.substr(0, tab);
        const string label = line.substr(tab + 1);
        if (text.empty() || label.empty()) continue;

        vector<Index> ids;
        ids.push_back(cls);
        for (const Index id : tokenizer.encode(text))
        {
            if (Index(ids.size()) >= sequence_length - 1) break;
            ids.push_back(id);
        }
        ids.push_back(sep);
        while (Index(ids.size()) < sequence_length) ids.push_back(pad);

        for (Index i = 0; i < sequence_length; ++i) out << ids[size_t(i)] << ";";
        for (Index i = 0; i < sequence_length; ++i) out << 1 << ";";
        out << label << "\n";
    }
}

void BertDataset::configure_bert_roles()
{
    const Index variables_number = get_variables_number();

    for (Index i = 0; i < 2 * sequence_length; ++i)
        set_variable_type(i, VariableType::Numeric);

    for (Index i = 0; i < sequence_length; ++i)
        set_variable_role(i, "Decoder");
    for (Index i = sequence_length; i < 2 * sequence_length; ++i)
        set_variable_role(i, "Input");
    for (Index i = 2 * sequence_length; i < variables_number; ++i)
        set_variable_role(i, "Target");

    set_shape("Input",   {sequence_length});
    set_shape("Decoder", {sequence_length});
}

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
