//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   T O K E N I Z E R   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#pragma once

#include "tensor_types.h"

namespace opennn
{

class Tokenizer
{
public:

    virtual ~Tokenizer() = default;

    virtual vector<string> tokenize(string_view text) const = 0;

    virtual void build_vocabulary(const vector<vector<string>>& documents,
                                  Index maximum_vocabulary_size,
                                  Index minimum_token_frequency);

    virtual void set_vocabulary(const vector<string>&);
    const vector<string>& get_vocabulary() const noexcept { return vocabulary; }
    const unordered_map<string, Index>& get_vocabulary_map() const noexcept { return vocabulary_map; }
    Index get_vocabulary_size() const noexcept { return Index(vocabulary.size()); }

    Index token_to_id(string_view token) const;
    const string& id_to_token(Index id) const;

    vector<Index> encode(string_view text) const;
    string decode(const vector<Index>& ids) const;

    Index get_unk_id() const noexcept { return unk_id; }
    const vector<string>& get_reserved_tokens() const noexcept { return reserved_tokens; }

protected:

    vector<string> reserved_tokens;
    Index unk_id = 0;

    vector<string> vocabulary;
    unordered_map<string, Index> vocabulary_map;

    void rebuild_map();
};

// Reserved tokens: [PAD]=0, [UNK]=1, [START]=2, [END]=3.
class WordLevelTokenizer : public Tokenizer
{
public:

    WordLevelTokenizer();

    vector<string> tokenize(string_view text) const override;
};


class WordPieceTokenizer : public Tokenizer
{
public:

    WordPieceTokenizer();
    explicit WordPieceTokenizer(const vector<string>& vocabulary);

    void load_vocabulary(const filesystem::path& vocabulary_file);

    void set_vocabulary(const vector<string>&) override;

    vector<string> tokenize(string_view text) const override;

    void build_vocabulary(const vector<vector<string>>&, Index, Index) override {}

    void set_lower_case(bool value) noexcept { do_lower_case = value; }

private:

    vector<string> basic_tokenize(string_view text) const;
    vector<string> wordpiece(const string& word) const;

    string unk_token = "[UNK]";
    string continuation_prefix = "##";
    Index  max_input_chars_per_word = 100;
    bool   do_lower_case = true;
};

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
