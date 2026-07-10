//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   T O K E N I Z E R   O P E R A T O R   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#pragma once

#include <array>

#include "operator.h"

namespace opennn
{

class TokenizerOperator : public Operator
{
public:

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
    virtual string decode(const vector<Index>& ids) const;

    Index get_unk_id() const noexcept { return unk_id; }
    const vector<string>& get_reserved_tokens() const noexcept { return reserved_tokens; }

    virtual unique_ptr<TokenizerOperator> clone() const = 0;
    virtual string get_kind() const = 0;

    void to_JSON(JsonWriter&) const override;
    void from_JSON(const Json*) override;

protected:

    vector<string> reserved_tokens;
    Index unk_id = 0;

    vector<string> vocabulary;
    unordered_map<string, Index> vocabulary_map;

    void rebuild_map();
};

unique_ptr<TokenizerOperator> make_tokenizer_operator(const string& kind);

// Reserved tokens: [PAD]=0, [UNK]=1, [START]=2, [END]=3.
class WordLevelTokenizer : public TokenizerOperator
{
public:

    WordLevelTokenizer();

    vector<string> tokenize(string_view text) const override;

    unique_ptr<TokenizerOperator> clone() const override { return make_unique<WordLevelTokenizer>(*this); }
    string get_kind() const override { return "WordLevel"; }
};


class WordPieceTokenizer : public TokenizerOperator
{
public:

    WordPieceTokenizer();
    explicit WordPieceTokenizer(const vector<string>& vocabulary);

    void load_vocabulary(const filesystem::path& vocabulary_file);

    void set_vocabulary(const vector<string>&) override;

    vector<string> tokenize(string_view text) const override;

    void build_vocabulary(const vector<vector<string>>&, Index, Index) override {}

    void set_lower_case(bool value) noexcept { do_lower_case = value; }
    bool get_lower_case() const noexcept { return do_lower_case; }

    unique_ptr<TokenizerOperator> clone() const override { return make_unique<WordPieceTokenizer>(*this); }
    string get_kind() const override { return "WordPiece"; }

    void to_JSON(JsonWriter&) const override;
    void from_JSON(const Json*) override;

private:

    vector<string> basic_tokenize(string_view text) const;
    vector<string> wordpiece(const string& word) const;

    string unk_token = "[UNK]";
    string continuation_prefix = "##";
    Index  max_input_chars_per_word = 100;
    bool   do_lower_case = true;
};


// Id 0 is reserved for [PAD]
class BytePairTokenizer : public TokenizerOperator
{
public:

    BytePairTokenizer();

    void load(const filesystem::path& vocabulary_json,
              const filesystem::path& merges_txt);

    vector<string> tokenize(string_view text) const override;
    string decode(const vector<Index>& ids) const override;

    void build_vocabulary(const vector<vector<string>>&, Index, Index) override {}

    vector<string> get_merges() const;
    void set_merges(const vector<string>&);

    unique_ptr<TokenizerOperator> clone() const override { return make_unique<BytePairTokenizer>(*this); }
    string get_kind() const override { return "BytePair"; }

    void to_JSON(JsonWriter&) const override;
    void from_JSON(const Json*) override;

    static constexpr string_view PAD_TOKEN = "[PAD]";

private:

    vector<string> pre_tokenize(string_view text) const;
    vector<string> bpe(const string& token) const;

    array<uint32_t, 256> byte_encoder{};
    unordered_map<uint32_t, unsigned char> byte_decoder;
    unordered_map<string, int> merge_ranks;
    mutable unordered_map<string, vector<string>> bpe_cache;
};

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
